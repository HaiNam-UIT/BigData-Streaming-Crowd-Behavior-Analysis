"""AI service entrypoint (P2): YOLO detection + ByteTrack -> CSV.

Mục tiêu P2 theo PROJECT_DESIGN_AND_DATA.md §17:
  video -> detect người -> track cục bộ -> xuất metadata kiểm chứng được.

Output:
  artifacts/runs/<run_id>/tracking_results.csv   - một dòng/detection (§6.1, §6.3)
  artifacts/runs/<run_id>/camera_metrics.csv     - window metrics (§12.3 camera-metrics)
  artifacts/runs/<run_id>/annotated.mp4          - video có bbox + track_id
  artifacts/runs/<run_id>/run_metadata.json      - cấu hình + thống kê + thời gian chạy

Cách chạy (từ repo root):
  .\\.venv\\Scripts\\python.exe -m ai_service.app --source data/raw/sample.mp4 --camera-id CAM01

Không gọi API ngoài phạm vi detection; count/density model (CSRNet) là phase sau.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml

# Cho phép chạy từ repo root bằng `python -m ai_service.app`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_service.preprocessing import (  # noqa: E402
    VideoOpenError,
    is_mot_sequence,
    iter_frames_auto,
    read_mot_seqinfo,
    read_video_info,
)
from ai_service.motion_features import (  # noqa: E402
    MotionConfig,
    compute_motion_features,
    summarize_directions,
)
from ai_service.virtual_line import (  # noqa: E402
    LineCountingConfig,
    VirtualLine,
    Zone,
    assign_zones,
    count_crossings,
    load_lines_from_dict,
    load_zones_from_dict,
    zone_occupancy,
)

logger = logging.getLogger("ai_service.app")


DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "app.yaml"
DEFAULT_SOURCES_PATH = _REPO_ROOT / "producer" / "video_sources.yaml"
DEFAULT_GEOMETRY_PATH = _REPO_ROOT / "configs" / "geometry.yaml"


# --------------------------------------------------------------------------- config


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config phải là mapping: {path}")
    return data


def load_camera_sources(path: Path) -> tuple[str, str, dict[str, dict[str, Any]]]:
    """Trả về (site_id, session_id, {camera_id: source_dict})."""
    data = load_yaml(path)
    site_id = data.get("site_id", "PLAZA_01")
    session_id = data.get("session_id", "S01")
    cams: dict[str, dict[str, Any]] = {}
    for cam in data.get("cameras") or []:
        cam_id = cam.get("camera_id")
        if cam_id:
            cams[cam_id] = cam
    return site_id, session_id, cams


def resolve_device(device: str) -> str:
    """'auto' -> '0' nếu CUDA khả dụng, ngược lại 'cpu'."""
    if device and device.lower() != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch luôn có khi chạy thật
        return "cpu"


# --------------------------------------------------------------------------- data


@dataclass
class DetectionRow:
    site_id: str
    session_id: str
    camera_id: str
    zone_id: str | None
    frame_id: int
    timestamp: str
    track_id: int | None
    class_name: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    confidence: float
    center_x: float
    center_y: float
    bottom_center_x: float
    bottom_center_y: float
    frame_width: int
    frame_height: int
    detector: str
    tracker: str | None


@dataclass
class WindowMetric:
    site_id: str
    session_id: str
    camera_id: str
    window_start: str
    window_end: str
    unique_tracks: int
    person_count_end: int
    person_count_avg: float
    person_count_max: int
    detection_rows: int


@dataclass
class RunStats:
    frames_read: int = 0
    frames_with_detections: int = 0
    total_detections: int = 0
    unique_tracks: set[int] = field(default_factory=set)
    processing_seconds: float = 0.0
    fps_processed: float = 0.0
    started_at: str = ""
    finished_at: str = ""


# --------------------------------------------------------------------------- helpers


def _bbox_row(
    base: dict[str, Any],
    xyxy: tuple[float, float, float, float],
    confidence: float,
    track_id: int | None,
) -> DetectionRow:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w = int(base["frame_width"])
    h = int(base["frame_height"])
    return DetectionRow(
        site_id=base["site_id"],
        session_id=base["session_id"],
        camera_id=base["camera_id"],
        zone_id=base["zone_id"],
        frame_id=base["frame_id"],
        timestamp=base["timestamp"],
        track_id=track_id,
        class_name="person",
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        confidence=confidence,
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        bottom_center_x=(x1 + x2) / 2.0,
        bottom_center_y=y2,
        frame_width=w,
        frame_height=h,
        detector=base["detector"],
        tracker=base["tracker"],
    )


def _draw_box(
    frame: Any,
    xyxy: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int] = (0, 200, 0),
) -> None:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, max(th, y1 - baseline - 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


# --------------------------------------------------------------------------- core


def run_inference(
    source: str | Path,
    camera_id: str,
    site_id: str,
    session_id: str,
    zone_id: str | None,
    cfg: dict[str, Any],
    output_dir: Path,
    geometry_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chạy detect+track trên một video, ghi CSV/JSON/video. Trả về summary."""
    from ultralytics import YOLO

    video_cfg = cfg.get("video", {})
    det_cfg = cfg.get("detection", {})
    trk_cfg = cfg.get("tracking", {})
    out_cfg = cfg.get("output", {})

    model_name = det_cfg.get("model", "yolo11n.pt")
    model_path = Path(cfg.get("paths", {}).get("model_path") or model_name)
    if not model_path.is_absolute():
        model_path = _REPO_ROOT / model_path
    if not model_path.exists() and not Path(model_name).exists():
        logger.info("Model %s chưa có, ultralytics sẽ tải về (lần đầu).", model_name)

    weights = str(model_path) if model_path.exists() else model_name
    device = resolve_device(str(det_cfg.get("device", "auto")))
    logger.info("Load model=%s device=%s", weights, device)
    model = YOLO(weights, task="detect")

    tracking_enabled = bool(trk_cfg.get("enabled", True))
    tracker_name = str(trk_cfg.get("tracker", "bytetrack.yaml")) if tracking_enabled else None

    person_cls = int(det_cfg.get("person_class_id", 0))
    conf = float(det_cfg.get("conf_threshold", 0.30))
    iou = float(det_cfg.get("iou_threshold", 0.50))
    imgsz = int(det_cfg.get("imgsz", 960))
    max_det = int(det_cfg.get("max_detections", 300))

    is_mot = is_mot_sequence(source)
    if is_mot:
        seqinfo = read_mot_seqinfo(source)
        info = {
            "source_fps": seqinfo["frame_rate"],
            "width": seqinfo["width"],
            "height": seqinfo["height"],
            "total_frames": seqinfo["seq_length"],
            "duration_seconds": (
                seqinfo["seq_length"] / seqinfo["frame_rate"]
                if seqinfo["frame_rate"] > 0 else 0.0
            ),
            "mot_sequence": seqinfo["name"],
        }
        logger.info("MOT sequence info: %s", info)
    else:
        info = read_video_info(source)
        logger.info("Video info: %s", info)

    eff_fps = float(video_cfg.get("sample_fps", 5.0))
    if not (0 < eff_fps < info["source_fps"]):
        eff_fps = info["source_fps"]

    output_dir.mkdir(parents=True, exist_ok=True)
    limit_seconds = video_cfg.get("_limit_seconds")

    rows: list[DetectionRow] = []
    frames: list[tuple[datetime, int]] = []  # (timestamp, person_count) cho window metric
    stats = RunStats(started_at=datetime.now(timezone.utc).isoformat())
    t0 = time.perf_counter()

    progress_every = int(cfg.get("logging", {}).get("progress_every_frames", 100))
    video_writer = None

    try:
        for packet in iter_frames_auto(
            source,
            sample_fps=eff_fps,
            resize_width=int(video_cfg.get("resize_width", 0) or 0),
            skip_seconds=float(video_cfg.get("skip_seconds", 0.0) or 0.0),
        ):
            frame_h, frame_w = packet.frame.shape[:2]

            if limit_seconds is not None:
                elapsed_video = (packet.timestamp - frames[0][0]).total_seconds() if frames else 0.0
                if elapsed_video > float(limit_seconds):
                    logger.info("Đạt giới hạn %.1fs (smoke test), dừng.", float(limit_seconds))
                    break

            if video_writer is None and out_cfg.get("save_annotated_video", True):
                path = output_dir / out_cfg.get("annotated_video_name", "annotated.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(path), fourcc, max(eff_fps, 1.0), (frame_w, frame_h)
                )
                if not video_writer.isOpened():
                    logger.warning("Không mở được VideoWriter cho %s, tắt ghi video.", path)
                    video_writer = None

            if tracking_enabled:
                results = model.track(
                    source=packet.frame,
                    persist=True,
                    classes=[person_cls],
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    max_det=max_det,
                    device=device,
                    tracker=tracker_name,
                    verbose=False,
                )
            else:
                results = model.predict(
                    source=packet.frame,
                    classes=[person_cls],
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    max_det=max_det,
                    device=device,
                    verbose=False,
                )

            person_count = 0
            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy_all = boxes.xyxy.cpu().numpy()
                    conf_all = boxes.conf.cpu().numpy()
                    ids_all = boxes.id.cpu().numpy() if boxes.id is not None else None
                    for idx in range(len(xyxy_all)):
                        track_id = int(ids_all[idx]) if ids_all is not None else None
                        xyxy = tuple(float(v) for v in xyxy_all[idx])  # type: ignore[arg-type]
                        confidence = float(conf_all[idx])
                        base = {
                            "site_id": site_id,
                            "session_id": session_id,
                            "camera_id": camera_id,
                            "zone_id": zone_id,
                            "frame_id": packet.frame_id,
                            "timestamp": packet.timestamp.isoformat(),
                            "frame_width": frame_w,
                            "frame_height": frame_h,
                            "detector": Path(model_name).stem,
                            "tracker": Path(tracker_name).stem if tracker_name else None,
                        }
                        rows.append(_bbox_row(base, xyxy, confidence, track_id))

                        if track_id is not None:
                            stats.unique_tracks.add(track_id)
                        label = f"person {confidence:.2f}"
                        if track_id is not None:
                            label = f"ID{track_id} {confidence:.2f}"
                        if video_writer is not None:
                            _draw_box(packet.frame, xyxy, label)
                    person_count = len(xyxy_all)
                    stats.total_detections += person_count
                    if person_count > 0:
                        stats.frames_with_detections += 1

            if video_writer is not None:
                video_writer.write(packet.frame)

            frames.append((packet.timestamp, person_count))
            stats.frames_read += 1
            if progress_every and stats.frames_read % progress_every == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Đã xử lý %d frame | %.2f fps | %d detection",
                    stats.frames_read,
                    stats.frames_read / elapsed if elapsed > 0 else 0.0,
                    stats.total_detections,
                )
    except VideoOpenError:
        raise
    except KeyboardInterrupt:
        logger.warning("Người dùng dừng giữa chừng; vẫn ghi phần đã xử lý.")
    finally:
        if video_writer is not None:
            video_writer.release()

    stats.processing_seconds = time.perf_counter() - t0
    stats.fps_processed = (
        stats.frames_read / stats.processing_seconds if stats.processing_seconds > 0 else 0.0
    )
    stats.finished_at = datetime.now(timezone.utc).isoformat()

    # ---- write outputs
    tracking_df = pd.DataFrame([asdict(r) for r in rows])
    tracking_csv = None
    if out_cfg.get("save_tracking_csv", True):
        tracking_csv = output_dir / out_cfg.get("tracking_csv_name", "tracking_results.csv")
        tracking_df.to_csv(tracking_csv, index=False)
        logger.info("Đã ghi %d dòng tracking -> %s", len(rows), tracking_csv)

    metrics_csv = None
    if out_cfg.get("save_camera_metrics", True) and frames:
        metrics_csv = output_dir / out_cfg.get("camera_metrics_name", "camera_metrics.csv")
        window_s = float(out_cfg.get("metrics_window_seconds", 10.0))
        metrics = _build_window_metrics(
            frames, rows, site_id, session_id, camera_id, window_s
        )
        pd.DataFrame([asdict(m) for m in metrics]).to_csv(metrics_csv, index=False)
        logger.info("Đã ghi %d window metric -> %s", len(metrics), metrics_csv)

    # ---- motion + line + zone (F03/F04/F05)
    geometry = geometry_cfg or {}
    extra_outputs = _write_geometry_outputs(
        tracking_df=tracking_df,
        geometry=geometry,
        camera_id=camera_id,
        site_id=site_id,
        session_id=session_id,
        eff_fps=eff_fps,
        out_cfg=out_cfg,
        output_dir=output_dir,
    )

    summary = {
        "camera_id": camera_id,
        "site_id": site_id,
        "session_id": session_id,
        "zone_id": zone_id,
        "source": str(source),
        "video_info": info,
        "config": {
            "model": model_name,
            "device": device,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "tracking": tracking_enabled,
            "tracker": tracker_name,
            "sample_fps": eff_fps,
        },
        "stats": {
            "frames_read": stats.frames_read,
            "frames_with_detections": stats.frames_with_detections,
            "total_detections": stats.total_detections,
            "unique_tracks": len(stats.unique_tracks),
            "processing_seconds": round(stats.processing_seconds, 3),
            "fps_processed": round(stats.fps_processed, 3),
            "started_at": stats.started_at,
            "finished_at": stats.finished_at,
        },
        "outputs": {
            "tracking_csv": str(tracking_csv) if tracking_csv else None,
            "camera_metrics_csv": str(metrics_csv) if metrics_csv else None,
            "annotated_video": (
                str(output_dir / out_cfg.get("annotated_video_name", "annotated.mp4"))
                if out_cfg.get("save_annotated_video", True) else None
            ),
            **extra_outputs,
        },
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _build_window_metrics(
    frames: list[tuple[datetime, int]],
    rows: list[DetectionRow],
    site_id: str,
    session_id: str,
    camera_id: str,
    window_seconds: float,
) -> list[WindowMetric]:
    """Gộp frames thành window cố định, tính count/track cho từng window."""
    if window_seconds <= 0 or not frames:
        return []

    t_first = frames[0][0]
    buckets: dict[int, list[tuple[datetime, int]]] = {}
    for ts, count in frames:
        idx = int((ts - t_first).total_seconds() // window_seconds)
        buckets.setdefault(idx, []).append((ts, count))

    # tracks per window dựa trên timestamp của từng row
    track_window: dict[int, set[int]] = {}
    for row in rows:
        if row.track_id is None:
            continue
        ts = datetime.fromisoformat(row.timestamp)
        idx = int((ts - t_first).total_seconds() // window_seconds)
        track_window.setdefault(idx, set()).add(row.track_id)

    out: list[WindowMetric] = []
    for idx in sorted(buckets):
        chunk = buckets[idx]
        counts = [c for _, c in chunk]
        rows_in_window = sum(
            1
            for r in rows
            if int(
                (datetime.fromisoformat(r.timestamp) - t_first).total_seconds()
                // window_seconds
            ) == idx
        )
        out.append(
            WindowMetric(
                site_id=site_id,
                session_id=session_id,
                camera_id=camera_id,
                window_start=chunk[0][0].isoformat(),
                window_end=chunk[-1][0].isoformat(),
                unique_tracks=len(track_window.get(idx, set())),
                person_count_end=counts[-1],
                person_count_avg=round(sum(counts) / len(counts), 2),
                person_count_max=max(counts),
                detection_rows=rows_in_window,
            )
        )
    return out


def _write_geometry_outputs(
    tracking_df: "pd.DataFrame",
    geometry: dict[str, Any],
    camera_id: str,
    site_id: str,
    session_id: str,
    eff_fps: float,
    out_cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, str | None]:
    """Tính motion (F03), line crossing (F04), zone occupancy (F05) và ghi CSV.

    Trả về dict đường dẫn output để nhét vào run_metadata.
    """
    result: dict[str, str | None] = {
        "motion_metrics_csv": None,
        "motion_directions_csv": None,
        "line_crossings_csv": None,
        "zone_occupancy_csv": None,
    }
    if tracking_df.empty:
        logger.warning("Không có tracking -> bỏ qua motion/line/zone.")
        return result

    trk_cfg = geometry.get("tracking_motion", {}) or {}
    min_frames = int(trk_cfg.get("min_track_frames", 10))
    smooth = int(trk_cfg.get("smoothing_window", 5))
    stationary = float(trk_cfg.get("stationary_speed_px", 5.0))
    mpp = float(trk_cfg.get("meters_per_pixel", 0.0) or 0.0)

    # ---- motion features
    try:
        motion = compute_motion_features(
            tracking_df,
            fps=eff_fps,
            cfg=MotionConfig(
                min_track_frames=min_frames,
                smoothing_window=smooth,
                stationary_speed_px=stationary,
                meters_per_pixel=mpp,
            ),
        )
        if not motion.empty:
            p = output_dir / out_cfg.get("motion_metrics_name", "motion_metrics.csv")
            motion.to_csv(p, index=False)
            result["motion_metrics_csv"] = str(p)
            directions = summarize_directions(motion)
            dp = output_dir / out_cfg.get("motion_directions_name", "motion_directions.csv")
            directions.to_csv(dp, index=False)
            result["motion_directions_csv"] = str(dp)
            logger.info("Đã ghi motion cho %d track -> %s", len(motion), p)
    except Exception as exc:  # pragma: no cover - log rồi đi tiếp
        logger.exception("Lỗi khi tính motion features: %s", exc)

    # ---- lọc line/zone theo camera
    all_lines: list[VirtualLine] = load_lines_from_dict(geometry)
    all_zones: list[Zone] = load_zones_from_dict(geometry)
    cam_lines = [ln for ln in all_lines if ln.camera_id in (camera_id, "*")]
    cam_zones = [z for z in all_zones if z.camera_id in (camera_id, "*")]

    # ---- line crossings
    if cam_lines:
        try:
            lc_cfg_dict = geometry.get("line_counting", {}) or {}
            events = count_crossings(
                tracking_df,
                cam_lines,
                site_id=site_id,
                session_id=session_id,
                cfg=LineCountingConfig(
                    fps=eff_fps,
                    margin_ratio=float(lc_cfg_dict.get("margin_ratio", 0.05)),
                    min_consecutive=int(lc_cfg_dict.get("min_consecutive", 1)),
                ),
            )
            ev_df = pd.DataFrame([asdict(e) for e in events])
            if ev_df.empty:
                ev_df = pd.DataFrame(
                    columns=[
                        "site_id", "session_id", "camera_id", "line_id", "edge_id",
                        "track_id", "frame_id", "timestamp", "direction", "x", "y", "count",
                    ]
                )
            p = output_dir / out_cfg.get("line_crossings_name", "line_crossings.csv")
            ev_df.to_csv(p, index=False)
            result["line_crossings_csv"] = str(p)
            logger.info("Đã ghi %d line crossing -> %s", len(ev_df), p)
        except Exception as exc:  # pragma: no cover
            logger.exception("Lỗi khi đếm line crossing: %s", exc)

    # ---- zone occupancy
    if cam_zones:
        try:
            occ = zone_occupancy(
                tracking_df,
                cam_zones,
                fps=eff_fps,
                window_seconds=float(
                    (geometry.get("zone_occupancy", {}) or {}).get("window_seconds", 10.0)
                ),
            )
            p = output_dir / out_cfg.get("zone_occupancy_name", "zone_occupancy.csv")
            occ.to_csv(p, index=False)
            result["zone_occupancy_csv"] = str(p)
            logger.info("Đã ghi %d dòng zone occupancy -> %s", len(occ), p)
        except Exception as exc:  # pragma: no cover
            logger.exception("Lỗi khi tính zone occupancy: %s", exc)

    return result


# --------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P2 AI service - YOLO + ByteTrack")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--source", help="Đường dẫn video/RTSP; bỏ trống thì lấy từ --camera-id")
    g.add_argument("--all-cameras", action="store_true", help="Chạy tất cả camera trong video_sources.yaml")
    p.add_argument("--camera-id", default="CAM01", help="Camera ID (khớp video_sources.yaml nếu dùng --source mặc định)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="app.yaml")
    p.add_argument("--sources", default=str(DEFAULT_SOURCES_PATH), help="video_sources.yaml")
    p.add_argument("--run-id", default="", help="Tên thư mục run; mặc định timestamp")
    p.add_argument("--no-track", action="store_true", help="Chỉ detect, không track")
    p.add_argument("--no-video", action="store_true", help="Không ghi video annotated (chạy nhanh)")
    p.add_argument("--sample-fps", type=float, default=None, help="Ghi đè sample_fps trong config")
    p.add_argument("--device", default=None, help="cpu | cuda | 0")
    p.add_argument("--limit-seconds", type=float, default=None, help="Chỉ xử lý N giây đầu (smoke test nhanh)")
    p.add_argument(
        "--geometry", default=str(DEFAULT_GEOMETRY_PATH),
        help="config YAML chứa lines/zones; bỏ qua nếu file không tồn tại",
    )
    p.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_yaml(Path(args.config))
    site_id, session_id, cams = load_camera_sources(Path(args.sources))

    geometry_cfg: dict[str, Any] = {}
    geo_path = Path(args.geometry)
    if geo_path.exists():
        try:
            geometry_cfg = load_yaml(geo_path)
            logger.info(
                "Geometry: %d line, %d zone từ %s",
                len(geometry_cfg.get("lines") or []),
                len(geometry_cfg.get("zones") or []),
                geo_path,
            )
        except Exception as exc:
            logger.warning("Không đọc được geometry %s: %s", geo_path, exc)
    else:
        logger.info("Không có file geometry (%s) -> bỏ qua line/zone.", geo_path)

    if args.sample_fps is not None:
        cfg.setdefault("video", {})["sample_fps"] = args.sample_fps
    if args.device is not None:
        cfg.setdefault("detection", {})["device"] = args.device
    if args.no_track:
        cfg.setdefault("tracking", {})["enabled"] = False
    if args.no_video:
        cfg.setdefault("output", {})["save_annotated_video"] = False
    if args.limit_seconds is not None:
        # giới hạn thời gian không phải thuộc tính của config; ghi vào cfg riêng
        cfg.setdefault("video", {})["_limit_seconds"] = args.limit_seconds

    runs_dir = _REPO_ROOT / cfg.get("paths", {}).get("runs_dir", "artifacts/runs")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs: list[dict[str, Any]] = []
    if args.all_cameras:
        for cam_id, cam in cams.items():
            jobs.append(_job_from_cam(cam_id, cam, args.source))
    else:
        cam = cams.get(args.camera_id, {})
        jobs.append(_job_from_cam(args.camera_id, cam, args.source or cam.get("file")))

    exit_code = 0
    for job in jobs:
        camera_id = job["camera_id"]
        source = job["source"]
        if not source:
            logger.error("Camera %s không có nguồn video; bỏ qua.", camera_id)
            exit_code = 2
            continue
        src_path = Path(source)
        if not str(source).lower().startswith(("rtsp://", "http://", "https://")):
            if src_path.is_absolute():
                pass
            elif not src_path.exists():
                candidate = _REPO_ROOT / src_path
                if candidate.exists():
                    src_path = candidate
                else:
                    logger.error("Không tìm thấy nguồn: %s", src_path)
                    exit_code = 3
                    continue
            if src_path.is_dir() and not is_mot_sequence(src_path):
                logger.error(
                    "Thư mục %s không phải sequence MOT (thiếu seqinfo.ini hoặc img1/).",
                    src_path,
                )
                exit_code = 3
                continue
        output_dir = runs_dir / run_id / camera_id
        logger.info("=== Run %s | camera=%s | source=%s ===", run_id, camera_id, src_path)
        try:
            summary = run_inference(
                source=src_path,
                camera_id=camera_id,
                site_id=site_id,
                session_id=session_id,
                zone_id=job.get("zone_id"),
                cfg=cfg,
                output_dir=output_dir,
                geometry_cfg=geometry_cfg,
            )
            logger.info(
                "Xong %s: frames=%d detections=%d tracks=%d fps=%.2f -> %s",
                camera_id,
                summary["stats"]["frames_read"],
                summary["stats"]["total_detections"],
                summary["stats"]["unique_tracks"],
                summary["stats"]["fps_processed"],
                output_dir,
            )
        except VideoOpenError as exc:
            logger.error("Lỗi mở video: %s", exc)
            exit_code = 4
    return exit_code


def _job_from_cam(camera_id: str, cam: dict[str, Any], override_source: str | None) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "zone_id": cam.get("zone_id"),
        "source": override_source or cam.get("file") or cam.get("rtsp"),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
