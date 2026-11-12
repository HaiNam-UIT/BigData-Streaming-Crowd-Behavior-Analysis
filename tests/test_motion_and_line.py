"""Test motion_features + virtual_line (không cần model, không cần video)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ai_service.motion_features import (
    MotionConfig,
    bearing_deg,
    bearing_name,
    compute_motion_features,
    summarize_directions,
)
from ai_service.virtual_line import (
    LineCountingConfig,
    VirtualLine,
    Zone,
    assign_zones,
    count_crossings,
    load_lines_from_dict,
    load_zones_from_dict,
    zone_occupancy,
)


# --------------------------------------------------------------------------- helpers


def _track_df(points: list[tuple[int, float, float]], track_id: int = 1, cam: str = "CAM01"):
    """points = [(frame_id, x, y)] -> DataFrame giống tracking_results.csv."""
    return pd.DataFrame(
        {
            "site_id": "P",
            "session_id": "S",
            "camera_id": cam,
            "frame_id": [p[0] for p in points],
            "track_id": [track_id] * len(points),
            "timestamp": [f"2026-01-01T00:00:{p[0]:02d}+00:00" for p in points],
            "bottom_center_x": [p[1] for p in points],
            "bottom_center_y": [p[2] for p in points],
            "bbox_x1": [p[1] - 10 for p in points],
            "bbox_y1": [p[2] - 50 for p in points],
            "bbox_x2": [p[1] + 10 for p in points],
            "bbox_y2": [p[2] for p in points],
        }
    )


# --------------------------------------------------------------------------- bearing


@pytest.mark.parametrize(
    "dx,dy,expected",
    [
        (1, 0, 0.0),         # phải -> E
        (1, -1, 45.0),       # chéo phải-lên -> NE
        (0, -1, 90.0),       # lên -> N
        (-1, -1, 135.0),     # trái-lên -> NW
        (-1, 0, 180.0),      # trái -> W
        (-1, 1, 225.0),      # trái-xuống -> SW
        (0, 1, 270.0),       # xuống -> S
        (1, 1, 315.0),       # phải-xuống -> SE
    ],
)
def test_bearing_deg(dx: float, dy: float, expected: float) -> None:
    assert math.isclose(bearing_deg(dx, dy), expected, abs_tol=0.1)


def test_bearing_name() -> None:
    assert bearing_name(0) == "E"
    assert bearing_name(90) == "N"
    assert bearing_name(180) == "W"
    assert bearing_name(270) == "S"
    assert bearing_name(359) == "E"


# --------------------------------------------------------------------------- motion features


def test_motion_features_walking_east() -> None:
    """Track đi sang phải đều -> direction E, speed hợp lý."""
    pts = [(i, 100 + i * 10, 300) for i in range(20)]
    df = _track_df(pts)
    out = compute_motion_features(df, fps=5.0, cfg=MotionConfig(min_track_frames=10))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["direction_net"] == "E"
    assert row["net_dx"] > 150
    # 10 px/frame * 5 fps = 50 px/s
    assert 45 < row["speed_px_per_s"] < 55


def test_motion_features_walking_south() -> None:
    pts = [(i, 300, 100 + i * 8) for i in range(20)]
    out = compute_motion_features(_track_df(pts), fps=5.0)
    assert out.iloc[0]["direction_net"] == "S"


def test_motion_features_filters_short_track() -> None:
    pts = [(i, 100, 100) for i in range(3)]
    out = compute_motion_features(
        _track_df(pts), fps=5.0, cfg=MotionConfig(min_track_frames=10)
    )
    assert out.empty


def test_motion_features_stationary() -> None:
    pts = [(i, 100.0, 100.0) for i in range(15)]
    out = compute_motion_features(
        _track_df(pts), fps=5.0, cfg=MotionConfig(min_track_frames=10, stationary_speed_px=5.0)
    )
    assert bool(out.iloc[0]["is_stationary"]) is True
    assert out.iloc[0]["direction"] == "stationary"


def test_summarize_directions() -> None:
    motion = pd.DataFrame({"direction": ["N", "N", "E", "stationary"]})
    s = summarize_directions(motion)
    assert dict(zip(s["direction"], s["track_count"])) == {"N": 2, "E": 1, "stationary": 1}
    assert math.isclose(s["share"].sum(), 1.0)


# --------------------------------------------------------------------------- virtual line


def test_line_rejects_identical_endpoints() -> None:
    with pytest.raises(ValueError):
        VirtualLine("L", "CAM01", 10, 10, 10, 10)


def test_line_signed_distance_sign() -> None:
    # line ngang từ (0,100)->(100,100), pháp tuyến (-dy,dx)=(0,100) -> hướng +y
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100, inside_side="positive")
    assert line.signed_distance(50, 50) < 0   # phía trên
    assert line.signed_distance(50, 150) > 0  # phía dưới


def test_crossing_down_is_in_when_inside_positive() -> None:
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100, inside_side="positive")
    assert line.crosses((50, 90), (50, 110)) == 1  # - sang + => IN
    assert line.crosses((50, 110), (50, 90)) == -1  # + sang - => OUT


def test_crossing_flips_when_inside_negative() -> None:
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100, inside_side="negative")
    assert line.crosses((50, 90), (50, 110)) == -1
    assert line.crosses((50, 110), (50, 90)) == 1


def test_crossing_outside_segment_returns_none() -> None:
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100)
    # người đi qua đường kéo dài nhưng ngoài đoạn [0,100]
    assert line.crosses((500, 90), (500, 110)) is None


def test_crossing_same_side_returns_none() -> None:
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100)
    assert line.crosses((50, 50), (50, 60)) is None


def test_count_crossings_basic() -> None:
    """1 người đi xuống qua line y=100 -> 1 event IN."""
    df = _track_df([(i, 50, 80 + i * 10) for i in range(10)])  # 80..170 -> cắt 100
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100, edge_id="E1", inside_side="positive")
    events = count_crossings(df, [line], "P", "S", LineCountingConfig(fps=5.0))
    assert len(events) == 1
    e = events[0]
    assert e.direction == "IN"
    assert e.track_id == 1
    assert e.edge_id == "E1"
    assert e.line_id == "L"


def test_count_crossings_no_double_count() -> None:
    """Đi qua rồi tiếp tục đi -> vẫn chỉ 1 event."""
    df = _track_df([(i, 50, 60 + i * 15) for i in range(10)])  # 60..195
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100)
    events = count_crossings(df, [line], "P", "S")
    assert len(events) == 1


def test_count_crossings_back_and_forth() -> None:
    """Đi xuống rồi đi lên -> 2 event (IN + OUT)."""
    pts = [(i, 50, 80 + i * 10) for i in range(6)]  # xuống 80..130
    pts += [(6 + i, 50, 130 - i * 10) for i in range(1, 6)]  # lên 120..80
    df = _track_df(pts)
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100)
    events = count_crossings(df, [line], "P", "S")
    dirs = [e.direction for e in events]
    assert dirs == ["IN", "OUT"]


def test_count_crossings_margin_filters_edge_hits() -> None:
    """Cắt gần đầu mút đoạn -> margin 0.2 loại bỏ."""
    # đi qua tại x=2 (gần đầu đoạn 0..100)
    df = _track_df([(i, 2, 80 + i * 10) for i in range(5)])
    line = VirtualLine("L", "CAM01", 0, 100, 100, 100)
    strict = count_crossings(
        df, [line], "P", "S", LineCountingConfig(fps=5.0, margin_ratio=0.2)
    )
    assert strict == []
    loose = count_crossings(
        df, [line], "P", "S", LineCountingConfig(fps=5.0, margin_ratio=0.0)
    )
    assert len(loose) == 1


# --------------------------------------------------------------------------- zone


def test_zone_contains_square() -> None:
    z = Zone("Z1", "CAM01", ((0, 0), (100, 0), (100, 100), (0, 100)))
    assert z.contains(50, 50) is True
    assert z.contains(150, 50) is False
    assert z.contains(50, 150) is False


def test_zone_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        Zone("Z1", "CAM01", ((0, 0), (10, 10)))


def test_assign_zones() -> None:
    df = _track_df([(1, 50, 50), (2, 50, 150), (3, 150, 50)], track_id=1)
    zones = [
        Zone("TOP", "CAM01", ((0, 0), (100, 0), (100, 100), (0, 100))),
        Zone("BOTTOM", "CAM01", ((0, 100), (100, 100), (100, 200), (0, 200))),
    ]
    out = assign_zones(df, zones)
    got = [None if pd.isna(v) else v for v in out["zone_id"]]
    assert got == ["TOP", "BOTTOM", None]


def test_zone_occupancy_counts() -> None:
    rows = []
    for f in range(20):
        for tid in range(5):  # 5 người cùng lúc trong zone
            rows.append((f, 50 + tid, 50))
    df = pd.DataFrame(
        {
            "camera_id": "CAM01",
            "frame_id": [r[0] for r in rows],
            "track_id": [r[1] for r in rows],
            "bottom_center_x": [r[2] for r in rows],
            "bottom_center_y": [r[2] for r in rows],
            "timestamp": [f"2026-01-01T00:00:{r[0]:02d}+00:00" for r in rows],
        }
    )
    zones = [Zone("Z1", "CAM01", ((0, 0), (100, 0), (100, 100), (0, 100)), area_m2=100.0)]
    occ = zone_occupancy(df, zones, fps=5.0, window_seconds=2.0)
    assert len(occ) >= 1
    first = occ.iloc[0]
    assert first["occupancy_max"] == 5
    assert first["unique_tracks"] == 5
    assert abs(first["density_persons_per_m2"] - 0.05) < 1e-6


# --------------------------------------------------------------------------- config loaders


def test_load_lines_from_dict() -> None:
    data = {
        "lines": [
            {"line_id": "L1", "camera_id": "CAM01", "x1": 0, "y1": 0, "x2": 10, "y2": 10}
        ]
    }
    lines = load_lines_from_dict(data)
    assert len(lines) == 1
    assert lines[0].line_id == "L1"
    assert lines[0].inside_side == "positive"


def test_load_zones_from_dict() -> None:
    data = {
        "zones": [
            {
                "zone_id": "Z1",
                "camera_id": "CAM01",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "area_m2": 50,
            }
        ]
    }
    zones = load_zones_from_dict(data)
    assert zones[0].area_m2 == 50.0
