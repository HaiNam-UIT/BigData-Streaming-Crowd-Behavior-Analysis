"""Smoke tests cho JSON schemas.

Kiểm tra: 3 file schema hợp lệ theo JSON Schema Draft 2020-12, và
một record mẫu đi kèm ai_service output khớp crowd_metadata.

Chạy: .\\.venv\\Scripts\\python.exe -m pytest tests/test_schemas.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    pytest.skip("jsonschema chưa cài", allow_module_level=True)


SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def _load(name: str) -> dict:
    path = SCHEMA_DIR / name
    assert path.exists(), f"Thiếu schema: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "filename",
    ["raw_frame.schema.json", "crowd_metadata.schema.json", "alert_event.schema.json"],
)
def test_schema_is_valid_draft_2020_12(filename: str) -> None:
    schema = _load(filename)
    # raise nếu schema hỏng
    Draft202012Validator.check_schema(schema)
    assert schema.get("$schema", "").endswith("2020-12/schema")


def test_raw_frame_sample_valid() -> None:
    schema = _load("raw_frame.schema.json")
    sample = {
        "site_id": "PLAZA_01",
        "session_id": "S01",
        "camera_id": "CAM01",
        "zone_id": "Z_N",
        "frame_id": 1200,
        "timestamp": "2026-11-05T09:30:00.000Z",
        "source_type": "group_recorded",
        "width": 1920,
        "height": 1080,
        "fps_source": 30.0,
        "fps_sampled": 5.0,
        "source_uri": "data/raw/sample.mp4",
    }
    Draft202012Validator(schema).validate(sample)


def test_crowd_metadata_sample_matches_cereal_row() -> None:
    """Record giống hệt 1 dòng tracking_results.csv phải pass."""
    schema = _load("crowd_metadata.schema.json")
    sample = {
        "site_id": "PLAZA_01",
        "session_id": "S01",
        "camera_id": "CAM01",
        "zone_id": "Z_N",
        "frame_id": 1200,
        "timestamp": "2026-11-05T09:30:00+00:00",
        "track_id": 42,
        "class_name": "person",
        "bbox_xyxy": [100.0, 80.0, 150.0, 230.0],
        "confidence": 0.94,
        "center_xy": [125.0, 155.0],
        "bottom_center_xy": [125.0, 230.0],
        "frame_width": 1920,
        "frame_height": 1080,
        "detector": "yolo11n",
        "tracker": "bytetrack",
    }
    Draft202012Validator(schema).validate(sample)


def test_crowd_metadata_allows_null_track() -> None:
    """Detector bật tracking nhưng chưa gán ID -> track_id = null phải pass."""
    schema = _load("crowd_metadata.schema.json")
    sample = {
        "site_id": "PLAZA_01",
        "session_id": "S01",
        "camera_id": "CAM02",
        "frame_id": 0,
        "timestamp": "2026-11-05T09:00:00+00:00",
        "track_id": None,
        "bbox_xyxy": [10, 20, 60, 180],
        "confidence": 0.51,
        "center_xy": [35.0, 100.0],
        "bottom_center_xy": [35.0, 180],
        "frame_width": 1280,
        "frame_height": 720,
    }
    Draft202012Validator(schema).validate(sample)


def test_crowd_metadata_rejects_missing_bbox() -> None:
    schema = _load("crowd_metadata.schema.json")
    validator = Draft202012Validator(schema)
    bad = {
        "site_id": "PLAZA_01",
        "session_id": "S01",
        "camera_id": "CAM01",
        "frame_id": 0,
        "timestamp": "2026-11-05T09:00:00+00:00",
        "track_id": 1,
        "confidence": 0.9,
        "center_xy": [0, 0],
        "bottom_center_xy": [0, 0],
        "frame_width": 100,
        "frame_height": 100,
    }
    assert list(validator.iter_errors(bad)), "Phải báo lỗi khi thiếu bbox_xyxy"


def test_alert_event_sample_valid() -> None:
    schema = _load("alert_event.schema.json")
    sample = {
        "alert_id": "abc-123",
        "site_id": "PLAZA_01",
        "zone_id": "Z_C",
        "edge_id": None,
        "timestamp": "2026-11-05T09:35:00Z",
        "alert_type": "POSSIBLE_CONGESTION",
        "severity": "warning",
        "evidence": {
            "metric": "density",
            "value": 2.1,
            "threshold": 1.5,
            "windows_persisted": 3,
        },
        "rule_id": "congestion_density_speed",
        "message": "Density vượt ngưỡng 3 window liên tiếp.",
    }
    Draft202012Validator(schema).validate(sample)


def test_alert_rejects_unknown_type() -> None:
    schema = _load("alert_event.schema.json")
    validator = Draft202012Validator(schema)
    bad = {
        "alert_id": "x",
        "site_id": "P",
        "zone_id": "Z_C",
        "timestamp": "2026-11-05T09:35:00Z",
        "alert_type": "STAMPEDE",  # ngoài enum
        "severity": "critical",
        "evidence": {"metric": "density", "value": 1, "threshold": 1},
    }
    assert list(validator.iter_errors(bad))
