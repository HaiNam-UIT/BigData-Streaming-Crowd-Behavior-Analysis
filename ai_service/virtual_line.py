"""Virtual line counting (F04) + zone occupancy (F05).

Đếm số người đi qua một đoạn thẳng ảo theo hướng IN/OUT.

Nguyên tắc thiết kế (PROJECT_DESIGN_AND_DATA.md):
- §6.5: dùng vị trí đại diện bottom-center của bbox; tạo event khi quỹ đạo
  cắt qua line; tránh đếm lặp nhiều frame.
- §9.4: event có timestamp, edge_id, direction IN/OUT, count.
- Hướng IN/OUT phải định nghĩa theo topology, KHÔNG suy diễn.

Cách xác định cắt line an toàn với detection jitter:
1. Chiếu điểm (x, y) lên pháp tuyến của line -> giá trị signed distance `d`.
2. Dấu d đổi giữa 2 frame liên tiếp => có crossing tiềm năng.
3. Kiểm tra thêm: giao điểm phải nằm TRONG đoạn line (không nằm trên đường kéo dài).
4. Cooldown theo track: một track chỉ được tính 1 lần cho mỗi lần cắt.

IN/OUT quy ước:
- `side_in` = phía được coi là "trong" (âm hoặc dương theo pháp tuyến).
- Đi từ ngoài vào trong  => IN.
- Đi từ trong ra ngoài  => OUT.

Không phụ thuộc ultralytics; chỉ pandas + numpy -> test được offline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Point = tuple[float, float]


@dataclass(frozen=True)
class VirtualLine:
    """Đoạn thẳng ảo trên ảnh, đếm khi bottom_center của track đi qua."""

    line_id: str
    camera_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    edge_id: str | None = None
    # Phía nào của line được coi là "trong"? "positive" hoặc "negative" của signed distance.
    inside_side: str = "positive"

    def __post_init__(self) -> None:
        if self.x1 == self.x2 and self.y1 == self.y2:
            raise ValueError(f"Line {self.line_id}: hai đầu trùng nhau")
        if self.inside_side not in ("positive", "negative"):
            raise ValueError(f"Line {self.line_id}: inside_side phải là 'positive' hoặc 'negative'")

    # --- hình học ---------------------------------------------------------
    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def signed_distance(self, x: float, y: float) -> float:
        """Khoảng cách có dấu tới đường thẳng chứa line.

        Dùng pháp tuyến (-dy, dx) chuẩn hoá; dấu dương khi điểm ở "bên trái hướng line".
        """
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        norm = math.hypot(dx, dy)
        if norm == 0:
            return 0.0
        nx, ny = -dy / norm, dx / norm
        return (x - self.x1) * nx + (y - self.y1) * ny

    def _projection_t(self, x: float, y: float) -> float:
        """Tham số t trong [0,1] nếu điểm chiếu nằm trên đoạn; ngoài khoảng = ngoài đoạn."""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0:
            return -1.0
        return ((x - self.x1) * dx + (y - self.y1) * dy) / seg_len2

    def crosses(
        self,
        prev: Point,
        curr: Point,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> int | None:
        """Trả về +1 (IN), -1 (OUT) hoặc None nếu không cắt.

        t_min/t_max cho phép co lại đoạn hiệu dụng để tránh đếm sát mép ảnh.
        """
        d_prev = self.signed_distance(*prev)
        d_curr = self.signed_distance(*curr)
        # Điểm nằm ĐÚNG trên line: không sinh crossing ở bước này.
        # Quy ước: bỏ qua bước chạm line, crossing sẽ được phát hiện ở bước kế tiếp
        # khi có điểm thực sự ở phía bên kia -> tránh đếm 2 lần.
        eps = 1e-6
        if abs(d_prev) < eps or abs(d_curr) < eps:
            return None
        if (d_prev > 0) == (d_curr > 0):
            return None

        # điểm cắt
        alpha = d_prev / (d_prev - d_curr)
        cx = prev[0] + alpha * (curr[0] - prev[0])
        cy = prev[1] + alpha * (curr[1] - prev[1])
        t = self._projection_t(cx, cy)
        if not (t_min <= t <= t_max):
            return None

        positive_end = d_curr > 0
        if self.inside_side == "positive":
            return 1 if positive_end else -1
        return -1 if positive_end else 1


@dataclass
class LineCountingConfig:
    fps: float = 25.0
    # Tỉ lệ co đoạn hiệu dụng hai đầu (0.05 = bỏ 5% mỗi phía)
    margin_ratio: float = 0.05
    # Số điểm liên tiếp phải cùng phía để xác nhận (giảm nhiễu detection)
    min_consecutive: int = 1


@dataclass
class CrossingEvent:
    site_id: str
    session_id: str
    camera_id: str
    line_id: str
    edge_id: str | None
    track_id: int
    frame_id: int
    timestamp: str
    direction: str           # "IN" | "OUT"
    x: float
    y: float
    count: int = 1


def _iter_track_points(track: pd.DataFrame) -> Iterable[tuple[int, float, float, str]]:
    """Sinh (frame_id, x, y, timestamp) theo thứ tự frame."""
    for row in track.itertuples(index=False):
        ts = getattr(row, "timestamp", "")
        yield int(row.frame_id), float(row.bottom_center_x), float(row.bottom_center_y), str(ts)


def count_crossings(
    tracking: pd.DataFrame,
    lines: Sequence[VirtualLine],
    site_id: str,
    session_id: str,
    cfg: LineCountingConfig | None = None,
) -> list[CrossingEvent]:
    """Đếm sự kiện đi qua các virtual line.

    Args:
        tracking: tracking_results.csv dạng DataFrame (cần frame_id, track_id,
            bottom_center_x/y, timestamp, camera_id).
        lines: danh sách line cần đếm.
        site_id, session_id: gắn vào event.
        cfg: cấu hình.

    Returns:
        Danh sách CrossingEvent sắp theo (line_id, frame_id).
    """
    cfg = cfg or LineCountingConfig()
    if tracking.empty or not lines:
        return []

    required = {"frame_id", "track_id", "bottom_center_x", "bottom_center_y"}
    missing = required - set(tracking.columns)
    if missing:
        raise ValueError(f"tracking thiếu cột: {sorted(missing)}")

    df = tracking[tracking["track_id"].notna()].copy()
    df["track_id"] = df["track_id"].astype(int)

    events: list[CrossingEvent] = []

    for line in lines:
        t_lo, t_hi = cfg.margin_ratio, 1.0 - cfg.margin_ratio

        cam_df = df[df["camera_id"] == line.camera_id] if "camera_id" in df.columns else df
        for tid, g in cam_df.groupby("track_id", sort=True):
            g = g.sort_values("frame_id")
            pts = list(_iter_track_points(g))
            if len(pts) < 2:
                continue

            # Phía hợp lệ gần nhất (bỏ các điểm nằm chính xác trên line).
            # Crossing = khi phía hợp lệ ĐỔI dấu -> đếm 1 lần, bất kể số frame chạm line ở giữa.
            last_side: int | None = None
            last_point: tuple[float, float] | None = None

            for f_curr, x_curr, y_curr, ts_curr in pts:
                d_curr = line.signed_distance(x_curr, y_curr)
                eps = 1e-6
                if abs(d_curr) < eps:
                    # điểm chạm line -> bỏ qua, chờ điểm thực sự đổi phía
                    continue
                side = 1 if d_curr > 0 else -1

                if last_side is None or last_point is None:
                    last_side = side
                    last_point = (x_curr, y_curr)
                    continue

                if side == last_side:
                    last_point = (x_curr, y_curr)
                    continue

                # đổi phía: kiểm tra giao điểm có nằm trong đoạn line hiệu dụng
                direction = line.crosses(
                    last_point, (x_curr, y_curr), t_min=t_lo, t_max=t_hi
                )
                if direction is not None:
                    x_prev, y_prev = last_point
                    d_prev = line.signed_distance(x_prev, y_prev)
                    alpha = d_prev / (d_prev - d_curr) if (d_prev - d_curr) != 0 else 0.5
                    cx = x_prev + alpha * (x_curr - x_prev)
                    cy = y_prev + alpha * (y_curr - y_prev)
                    events.append(
                        CrossingEvent(
                            site_id=site_id,
                            session_id=session_id,
                            camera_id=line.camera_id,
                            line_id=line.line_id,
                            edge_id=line.edge_id,
                            track_id=int(tid),
                            frame_id=int(f_curr),
                            timestamp=ts_curr,
                            direction="IN" if direction > 0 else "OUT",
                            x=round(cx, 2),
                            y=round(cy, 2),
                        )
                    )

                last_side = side
                last_point = (x_curr, y_curr)

    events.sort(key=lambda e: (e.line_id, e.frame_id))
    logger.info("line counting: %d event trên %d line", len(events), len(lines))
    return events


# --------------------------------------------------------------------------- zone occupancy


@dataclass(frozen=True)
class Zone:
    """Polygon zone theo toạ độ pixel."""

    zone_id: str
    camera_id: str
    polygon: tuple[Point, ...]   # >= 3 đỉnh, thứ tự kim đồng hồ hoặc ngược
    area_m2: float | None = None  # diện tích thực để tính density (tuỳ chọn)

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"Zone {self.zone_id}: cần >= 3 đỉnh")

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon (ray casting)."""
        inside = False
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xint = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                if x < xint:
                    inside = not inside
        return inside


def assign_zones(tracking: pd.DataFrame, zones: Sequence[Zone]) -> pd.DataFrame:
    """Gán zone_id cho từng detection theo bottom_center (point-in-polygon)."""
    if tracking.empty:
        return tracking
    df = tracking.copy()
    if "zone_id" not in df.columns:
        df["zone_id"] = pd.NA

    for zone in zones:
        mask = (
            (df["camera_id"] == zone.camera_id) if "camera_id" in df.columns else True
        )
        sub = df[mask]
        if sub.empty:
            continue
        hit = [
            zone.contains(float(x), float(y))
            for x, y in zip(sub["bottom_center_x"], sub["bottom_center_y"])
        ]
        df.loc[sub.index[np.array(hit, dtype=bool)], "zone_id"] = zone.zone_id

    return df


def zone_occupancy(
    tracking: pd.DataFrame,
    zones: Sequence[Zone],
    fps: float,
    window_seconds: float = 10.0,
) -> pd.DataFrame:
    """Occupancy theo zone theo cửa sổ thời gian.

    Trả về DataFrame: camera_id, zone_id, window_start_frame, window_end_frame,
    window_start, window_end, occupancy (count trung bình), occupancy_max,
    unique_tracks, density_persons_per_m2 (nếu zone có area_m2).
    """
    if tracking.empty or not zones:
        return pd.DataFrame()

    fps = float(fps) if fps and fps > 0 else 25.0
    df = assign_zones(tracking, zones)
    df = df[df["zone_id"].notna()]
    if df.empty:
        return pd.DataFrame()

    area_by_zone = {z.zone_id: z.area_m2 for z in zones}
    rows: list[dict[str, Any]] = []

    for (cam, zid), g in df.groupby(["camera_id", "zone_id"], sort=True):
        per_frame = g.groupby("frame_id").size().sort_index()
        if per_frame.empty:
            continue
        f_start, f_end = int(per_frame.index.min()), int(per_frame.index.max())
        wframes = max(1, int(round(window_seconds * fps)))

        bucket = ((per_frame.index - f_start) // wframes).astype(int)
        for b, sub in per_frame.groupby(bucket):
            start_f = f_start + int(b) * wframes
            end_f = min(f_end, start_f + wframes - 1)
            sub_g = g[(g["frame_id"] >= start_f) & (g["frame_id"] <= end_f)]
            uniq = sub_g["track_id"].dropna().nunique()
            occ_avg = float(sub.mean())
            occ_max = int(sub.max())
            area = area_by_zone.get(zid)
            density = round(occ_avg / area, 4) if area else None

            ts_col = "timestamp" if "timestamp" in g.columns else None
            w_start = w_end = ""
            if ts_col:
                s_rows = sub_g[sub_g["frame_id"] == start_f]
                e_rows = sub_g[sub_g["frame_id"] == end_f]
                w_start = str(s_rows["timestamp"].iloc[0]) if not s_rows.empty else ""
                w_end = str(e_rows["timestamp"].iloc[0]) if not e_rows.empty else ""

            rows.append(
                {
                    "camera_id": cam,
                    "zone_id": zid,
                    "window_start_frame": start_f,
                    "window_end_frame": end_f,
                    "window_start": w_start,
                    "window_end": w_end,
                    "occupancy": round(occ_avg, 2),
                    "occupancy_max": occ_max,
                    "unique_tracks": int(uniq),
                    "density_persons_per_m2": density,
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- config


def load_lines_from_dict(data: dict) -> list[VirtualLine]:
    """Tạo danh sách VirtualLine từ config YAML đã parse."""
    out: list[VirtualLine] = []
    for item in data.get("lines", []) or []:
        out.append(
            VirtualLine(
                line_id=str(item["line_id"]),
                camera_id=str(item["camera_id"]),
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
                edge_id=item.get("edge_id"),
                inside_side=str(item.get("inside_side", "positive")),
            )
        )
    return out


def load_zones_from_dict(data: dict) -> list[Zone]:
    """Tạo danh sách Zone từ config YAML đã parse."""
    out: list[Zone] = []
    for item in data.get("zones", []) or []:
        poly = tuple((float(p[0]), float(p[1])) for p in item["polygon"])
        out.append(
            Zone(
                zone_id=str(item["zone_id"]),
                camera_id=str(item["camera_id"]),
                polygon=poly,
                area_m2=float(item["area_m2"]) if item.get("area_m2") else None,
            )
        )
    return out
