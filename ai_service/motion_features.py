"""Motion features từ track cục bộ (F03).

Mục đích:
- Từ `tracking_results.csv` (frame_id, track_id, center/bottom_center) tính:
  * quỹ đạo theo track,
  * vận tốc (pixel/frame -> pixel/s -> m/s nếu có hiệu chuẩn),
  * hướng di chuyển (độ, 0 = đông, 90 = bắc, hình ảnh y hướng xuống).

Nguyên tắc thiết kế (PROJECT_DESIGN_AND_DATA.md):
- §6.3: track_id chỉ có ý nghĩa trong 1 camera. Không ghép xuyên camera.
- §6.4: optical flow mô tả chuyển động pixel, KHÔNG biết đích đến của từng người.
  Vận tốc ở đây là ước lượng thô từ tâm box; muốn ra m/s cần hiệu chuẩn
  mặt phẳng (homography) — ngoài phạm vi module này.
- Chỉ tính khi track đủ dài (>= min_track_frames), bỏ các track rác.

Output: `motion_metrics.csv` — mỗi dòng = 1 track, đủ feature cho F05/F06.

Không phụ thuộc ultralytics; chỉ pandas + numpy -> test được offline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hướng tham chiếu: 0° = +x (phải), 90° = -y (lên trên ảnh).
# Quy ước này khớp với góc toán học trên trục ảnh (y hướng xuống),
# nên `dy` phải đảo dấu khi tính atan2.
_DIRECTION_NAMES = [
    (22.5, "E"), (67.5, "NE"), (112.5, "N"), (157.5, "NW"),
    (202.5, "W"), (247.5, "SW"), (292.5, "S"), (337.5, "SE"), (360.1, "E"),
]


@dataclass
class MotionConfig:
    """Tham số tính motion feature."""

    min_track_frames: int = 10
    # Cửa sổ để làm mượt vận tốc (frame gần nhất)
    smoothing_window: int = 5
    # Ngưỡng coi là đứng yên (pixel/s). Dưới ngưỡng này -> direction "stationary"
    stationary_speed_px: float = 5.0
    # (tuỳ chọn) Tỉ lệ mét/pixel để đổi vận tốc sang m/s; 0 = không đổi
    meters_per_pixel: float = 0.0


def bearing_deg(dx: float, dy: float) -> float:
    """Góc di chuyển (độ, 0..360) trong hệ toạ độ ảnh (y hướng xuống)."""
    if dx == 0 and dy == 0:
        return 0.0
    deg = math.degrees(math.atan2(-dy, dx))
    return deg % 360.0


def bearing_name(deg: float) -> str:
    """Đổi góc sang 8 hướng chữ (E/NE/N/...)."""
    deg = deg % 360.0
    for limit, name in _DIRECTION_NAMES:
        if deg < limit:
            return name
    return "E"


def _fit_smoothed_velocity(df: pd.DataFrame, window: int) -> tuple[float, float]:
    """Vận tốc trung bình (px/frame) trên `window` điểm cuối, bằng hồi quy tuyến tính.

    Hồi quy bền hơn hiệu 2 điểm với track có jitter detection.
    """
    tail = df.tail(max(2, window))
    if len(tail) < 2:
        return 0.0, 0.0
    x = tail["frame_id"].to_numpy(dtype=float)
    ycx = tail["bottom_center_x"].to_numpy(dtype=float)
    ycy = tail["bottom_center_y"].to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return 0.0, 0.0
    vx = float(np.polyfit(x, ycx, 1)[0])
    vy = float(np.polyfit(x, ycy, 1)[0])
    return vx, vy


def compute_motion_features(
    tracking: pd.DataFrame,
    fps: float,
    cfg: MotionConfig | None = None,
) -> pd.DataFrame:
    """Tính motion feature cho mỗi track.

    Args:
        tracking: DataFrame đọc từ `tracking_results.csv`. Cần các cột:
            frame_id, track_id, bottom_center_x, bottom_center_y, bbox_*
        fps: fps của luồng đã lấy mẫu (không phải fps gốc) — dùng để đổi px/frame -> px/s.
        cfg: tham số; mặc định MotionConfig().

    Returns:
        DataFrame, mỗi dòng = 1 track với các cột:
        camera_id, track_id, first_frame, last_frame, num_frames, duration_s,
        start_x, start_y, end_x, end_y, net_dx, net_dy, path_length_px,
        displacement_px, speed_px_per_s, speed_mps, bearing_deg, direction,
        bearing_net_deg, direction_net, is_stationary
    """
    cfg = cfg or MotionConfig()
    if tracking.empty:
        return pd.DataFrame()

    required = {"frame_id", "track_id", "bottom_center_x", "bottom_center_y"}
    missing = required - set(tracking.columns)
    if missing:
        raise ValueError(f"tracking thiếu cột: {sorted(missing)}")

    fps = float(fps) if fps and fps > 0 else 25.0
    df = tracking[tracking["track_id"].notna()].copy()
    df["track_id"] = df["track_id"].astype(int)

    camera_id = (
        str(df["camera_id"].iloc[0]) if "camera_id" in df.columns and len(df) else "CAM00"
    )

    out_rows: list[dict[str, Any]] = []
    for tid, g in df.groupby("track_id", sort=True):
        g = g.sort_values("frame_id")
        n = len(g)
        if n < cfg.min_track_frames:
            continue

        frames = g["frame_id"].to_numpy(dtype=float)
        xs = g["bottom_center_x"].to_numpy(dtype=float)
        ys = g["bottom_center_y"].to_numpy(dtype=float)

        # --- vận tốc làm mượt
        vx, vy = _fit_smoothed_velocity(g, cfg.smoothing_window)
        speed_px_per_frame = math.hypot(vx, vy)
        speed_px_per_s = speed_px_per_frame * fps
        speed_mps = speed_px_per_s * cfg.meters_per_pixel if cfg.meters_per_pixel > 0 else None

        # --- hướng tức thời (từ vận tốc trượt)
        deg_inst = bearing_deg(vx, vy)

        # --- hướng ròng (start -> end), phản ánh "đi đâu" về tổng thể
        net_dx = float(xs[-1] - xs[0])
        net_dy = float(ys[-1] - ys[0])
        deg_net = bearing_deg(net_dx, net_dy)

        # --- độ dài quỹ đạo thực (tổng khoảng cách liên tiếp)
        if n > 1:
            seg = np.hypot(np.diff(xs), np.diff(ys))
            path_length = float(seg.sum())
        else:
            path_length = 0.0
        displacement = float(math.hypot(net_dx, net_dy))

        is_stationary = speed_px_per_s < cfg.stationary_speed_px

        out_rows.append(
            {
                "camera_id": camera_id,
                "track_id": int(tid),
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                "num_frames": n,
                "duration_s": round((frames[-1] - frames[0]) / fps, 3),
                "start_x": round(float(xs[0]), 2),
                "start_y": round(float(ys[0]), 2),
                "end_x": round(float(xs[-1]), 2),
                "end_y": round(float(ys[-1]), 2),
                "net_dx": round(net_dx, 2),
                "net_dy": round(net_dy, 2),
                "path_length_px": round(path_length, 2),
                "displacement_px": round(displacement, 2),
                "speed_px_per_s": round(speed_px_per_s, 3),
                "speed_mps": None if speed_mps is None else round(speed_mps, 3),
                "bearing_deg": round(deg_inst, 1),
                "direction": "stationary" if is_stationary else bearing_name(deg_inst),
                "bearing_net_deg": round(deg_net, 1),
                "direction_net": "stationary" if displacement < 1.0 else bearing_name(deg_net),
                "is_stationary": bool(is_stationary),
            }
        )

    result = pd.DataFrame(out_rows)
    logger.info(
        "motion_features: %d/%d track đủ dài (>= %d frame) -> %d dòng",
        len(result), df["track_id"].nunique(), cfg.min_track_frames, len(result),
    )
    return result


def summarize_directions(motion: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp số track theo 8 hướng — dùng cho dashboard motion field."""
    if motion.empty:
        return pd.DataFrame(columns=["direction", "track_count", "share"])
    counts = (
        motion["direction"].value_counts().rename_axis("direction").reset_index(name="track_count")
    )
    total = int(counts["track_count"].sum())
    counts["share"] = (counts["track_count"] / total).round(3) if total else 0.0
    return counts
