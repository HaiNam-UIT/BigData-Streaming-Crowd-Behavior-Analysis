"""Đánh giá tracker trên MOTChallenge (MOTA/IDF1/HOTA) và counting (MAE/RMSE).

Đầu vào:
  --pred   tracking_results.csv sinh bởi ai_service/app.py
  --gt     gt/gt.txt của sequence MOT (hoặc thư mục sequence)
  --seqinfo (tuỳ chọn) seqinfo.ini để tự scale nếu bbox đã resize

Xử lý bắt buộc theo chuẩn MOTChallenge:
  1. Lọc gt: mark==1 AND class==1 (pedestrian), bỏ visibility thấp nếu cấu hình.
  2. Chuyển bbox xyxy (pred) -> xywh (MOT format).
  3. Tự phát hiện scale: nếu frame_width trong pred != imWidth trong seqinfo
     thì nhân bbox lên để khớp gt.
  4. Chỉ đánh giá trên các frame có trong pred (vì sample_fps < fps gốc).

Đầu ra:
  - In bảng summary ra stdout
  - Ghi JSON: <output>/tracking_metrics.json
  - Ghi CSV chi tiết theo frame: <output>/counting_per_frame.csv

Chạy:
  .\\.venv\\Scripts\\python.exe -m evaluation.evaluate_tracking `
      --pred artifacts/runs/<run>/CAM01/tracking_results.csv `
      --seq  data/public/MOT20/train/MOT20-02
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Windows console
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

MOT_COLS = ["frame", "id", "x", "y", "w", "h", "mark", "class", "visibility", "unused"]


# --------------------------------------------------------------------------- IO


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"frame_id", "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV thiếu cột: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "frame": df["frame_id"].astype(int),
            "id": df["track_id"].fillna(-1).astype(int),
            "x": df["bbox_x1"].astype(float),
            "y": df["bbox_y1"].astype(float),
            "w": (df["bbox_x2"] - df["bbox_x1"]).astype(float),
            "h": (df["bbox_y2"] - df["bbox_y1"]).astype(float),
            "conf": df.get("confidence", pd.Series(np.ones(len(df)))).astype(float),
        }
    )
    # Bỏ track_id = -1 nếu có (chưa gán)
    out = out[out["id"] >= 0]
    return out


def load_gt(gt_path: Path, min_visibility: float) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, header=None, names=MOT_COLS)
    gt = gt[(gt["mark"] == 1) & (gt["class"] == 1)]
    if min_visibility > 0:
        gt = gt[gt["visibility"] >= min_visibility]
    return gt[["frame", "id", "x", "y", "w", "h"]]


def read_seqinfo(seq_dir: Path) -> dict:
    ini = seq_dir / "seqinfo.ini"
    if not ini.is_file():
        return {}
    import configparser

    cp = configparser.ConfigParser()
    cp.read(ini, encoding="utf-8")
    s = cp["Sequence"] if "Sequence" in cp else {}
    return {
        "frame_rate": float(s.get("frameRate", 25)),
        "width": int(s.get("imWidth", 0)),
        "height": int(s.get("imHeight", 0)),
        "seq_length": int(s.get("seqLength", 0)),
    }


# --------------------------------------------------------------------------- core metric


def _iou_matrix(gt_boxes: np.ndarray, pr_boxes: np.ndarray, max_iou: float = 0.5) -> np.ndarray:
    """Distance matrix cho motmetrics: 1 - IoU, NaN nếu IoU < max_iou.

    Tự tính để tránh lỗi `np.asfarray` đã bị xoá ở NumPy >= 2.0
    (motmetrics 1.4.0 vẫn dùng API cũ).
    """
    n, m = len(gt_boxes), len(pr_boxes)
    dist = np.full((n, m), np.nan, dtype=float)
    if n == 0 or m == 0:
        return dist

    gx1, gy1 = gt_boxes[:, 0][:, None], gt_boxes[:, 1][:, None]
    gw, gh = gt_boxes[:, 2][:, None], gt_boxes[:, 3][:, None]
    gx2, gy2 = gx1 + gw, gy1 + gh

    px1, py1 = pr_boxes[:, 0][None, :], pr_boxes[:, 1][None, :]
    pw, ph = pr_boxes[:, 2][None, :], pr_boxes[:, 3][None, :]
    px2, py2 = px1 + pw, py1 + ph

    inter_w = np.clip(np.minimum(gx2, px2) - np.maximum(gx1, px1), 0, None)
    inter_h = np.clip(np.minimum(gy2, py2) - np.maximum(gy1, py1), 0, None)
    inter = inter_w * inter_h

    gt_area = np.clip(gw * gh, 1e-9, None)
    pr_area = np.clip(pw * ph, 1e-9, None)
    union = gt_area + pr_area - inter
    iou = inter / union

    mask = iou >= max_iou
    dist[mask] = 1.0 - iou[mask]
    return dist


def mot_accumulators(pred: pd.DataFrame, gt: pd.DataFrame):
    """Tính accumulators của motmetrics + các chỉ số đơn giản.

    Trả về dict metrics + chi tiết theo frame.
    """
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(set(pred["frame"]).intersection(set(gt["frame"])))
    per_frame = []

    for f in frames:
        p = pred[pred["frame"] == f]
        g = gt[gt["frame"] == f]
        # boxes dạng [x, y, w, h] cho motmetrics.to_mot
        gt_boxes = g[["x", "y", "w", "h"]].to_numpy(dtype=float)
        pr_boxes = p[["x", "y", "w", "h"]].to_numpy(dtype=float)
        distances = _iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        acc.update(
            list(g["id"].astype(int).values),
            list(p["id"].astype(int).values),
            distances,
        )
        per_frame.append(
            {
                "frame": int(f),
                "gt_count": int(len(g)),
                "pred_count": int(len(p)),
                "abs_error": int(abs(len(g) - len(p))),
            }
        )

    return acc, per_frame, frames


def summarize(acc, num_frames: int) -> dict:
    import motmetrics as mm

    mh = mm.metrics.create()
    names = [
        "num_frames",
        "mota",
        "motp",
        "idf1",
        "idp",
        "idr",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "num_detections",
        "num_objects",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
    ]
    summary = mh.compute(acc, metrics=[n for n in names if n in mh.names], name="acc")
    row = summary.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if pd.isna(v) else round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    out["frames_evaluated"] = num_frames
    # Các metric tỉ lệ -> phần trăm cho dễ đọc
    for k in ("mota", "motp", "idf1", "idp", "idr"):
        if out.get(k) is not None:
            out[f"{k}_pct"] = round(out[k] * 100, 2)
    return out


def try_hota(pred: pd.DataFrame, gt: pd.DataFrame) -> dict | None:
    """HOTA nếu cài TrackEval; không bắt buộc."""
    try:
        import trackeval  # noqa: F401
    except ImportError:
        return None
    # TrackEval yêu cầu cấu trúc thư mục cố định; bỏ qua trong script này,
    # hướng dẫn chạy riêng trong README/evaluation.
    return None


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Đánh giá tracker MOT20 vs gt.txt")
    ap.add_argument("--pred", required=True, help="tracking_results.csv")
    ap.add_argument(
        "--seq", required=True,
        help="thư mục sequence MOT (chứa gt/gt.txt + seqinfo.ini) hoặc trực tiếp gt.txt",
    )
    ap.add_argument("--out", default=None, help="thư mục xuất kết quả; mặc định cạnh pred")
    ap.add_argument(
        "--min-visibility", type=float, default=0.0,
        help="bỏ gt có visibility < giá trị này (0 = giữ hết; 0.2 như MOTChallenge hay dùng)",
    )
    ap.add_argument(
        "--scale", type=float, default=None,
        help="hệ số nhân bbox pred để khớp gt; mặc định tự suy từ frame_width/seqinfo",
    )
    args = ap.parse_args(argv)

    pred_path = Path(args.pred)
    seq_path = Path(args.seq)
    if not pred_path.is_file():
        raise SystemExit(f"Không thấy pred: {pred_path}")

    if seq_path.is_dir():
        gt_path = seq_path / "gt" / "gt.txt"
        seqinfo = read_seqinfo(seq_path)
    else:
        gt_path = seq_path
        seqinfo = {}

    if not gt_path.is_file():
        raise SystemExit(
            f"Không thấy gt.txt: {gt_path}\n"
            "Lưu ý: MOT20 test/ KHÔNG có gt — chỉ đánh giá được train/."
        )

    out_dir = Path(args.out) if args.out else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = load_pred(pred_path)
    if pred.empty:
        raise SystemExit("pred rỗng — không có track_id hợp lệ để đánh giá.")

    # --- tự xác định scale
    scale = args.scale
    if scale is None:
        frame_w = int(pd.read_csv(pred_path, usecols=["frame_width"]).iloc[0]["frame_width"])
        gt_w = seqinfo.get("width", 0)
        scale = (gt_w / frame_w) if (gt_w and frame_w and gt_w != frame_w) else 1.0
    if scale != 1.0:
        for c in ("x", "w"):
            pred[c] = pred[c] * scale
        for c in ("y", "h"):
            pred[c] = pred[c] * scale

    gt = load_gt(gt_path, args.min_visibility)
    if gt.empty:
        raise SystemExit("gt lọc xong rỗng — kiểm tra cột class/mark/visibility.")

    print("=" * 72)
    print(f"Pred   : {pred_path}")
    print(f"GT     : {gt_path}")
    print(f"Scale  : {scale:.4f}  (pred frame_w x scale = gt coords)")
    print(f"GT rows: {len(gt)}  |  GT tracks: {gt['id'].nunique()}  |  GT frames: {gt['frame'].nunique()}")
    print(f"Pred rows: {len(pred)}  |  Pred tracks: {pred['id'].nunique()}  |  frames: {pred['frame'].nunique()}")
    print(f"min_visibility: {args.min_visibility}")
    print("-" * 72)

    try:
        acc, per_frame, frames = mot_accumulators(pred, gt)
    except ImportError:
        raise SystemExit("Thiếu py-motmetrics. Chạy: pip install motmetrics")

    metrics = summarize(acc, len(frames))

    # counting
    pf = pd.DataFrame(per_frame)
    mae = float(pf["abs_error"].mean()) if not pf.empty else None
    rmse = float(np.sqrt((pf["abs_error"] ** 2).mean())) if not pf.empty else None
    metrics.update(
        {
            "counting_mae": round(mae, 3) if mae is not None else None,
            "counting_rmse": round(rmse, 3) if rmse is not None else None,
            "gt_avg_per_frame": round(float(pf["gt_count"].mean()), 2) if not pf.empty else None,
            "pred_avg_per_frame": round(float(pf["pred_count"].mean()), 2) if not pf.empty else None,
        }
    )

    print("\n[TRACKING METRICS]")
    for k in ("mota_pct", "idf1_pct", "idp_pct", "idr_pct", "motp",
              "num_switches", "num_false_positives", "num_misses",
              "num_detections", "num_objects", "frames_evaluated"):
        if k in metrics and metrics[k] is not None:
            print(f"  {k:<22} {metrics[k]}")

    print("\n[COUNTING PER FRAME]")
    print(f"  gt_avg_per_frame       {metrics['gt_avg_per_frame']}")
    print(f"  pred_avg_per_frame     {metrics['pred_avg_per_frame']}")
    print(f"  MAE                    {metrics['counting_mae']}")
    print(f"  RMSE                   {metrics['counting_rmse']}")

    metrics["meta"] = {
        "pred": str(pred_path),
        "gt": str(gt_path),
        "scale": scale,
        "min_visibility": args.min_visibility,
        "seqinfo": seqinfo,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "note": "HOTA cần TrackEval; xem hướng dẫn trong evaluation/README.",
    }

    out_json = out_dir / "tracking_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pf.to_csv(out_dir / "counting_per_frame.csv", index=False)

    print("-" * 72)
    print(f"Đã ghi: {out_json}")
    print(f"Đã ghi: {out_dir / 'counting_per_frame.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
