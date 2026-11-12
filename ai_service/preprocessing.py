"""Video decoding + frame sampling cho AI service.

Vai trò (P2):
- Mở video file (demo) hoặc RTSP (deployment) bằng OpenCV.
- Lấy mẫu theo FPS cấu hình thay vì xử lý mọi frame.
- Sinh metadata frame: camera_id, frame_id, timestamp theo wall-clock gốc.

Không có logic AI ở đây; detection/tracking nằm ở app.py.

Tham chiếu: PROJECT_DESIGN_AND_DATA.md §8.3 (đồng bộ thời gian), §12.2 (demo mode).
"""

from __future__ import annotations

import configparser
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Union

import cv2

logger = logging.getLogger(__name__)


class VideoOpenError(RuntimeError):
    """Không mở được nguồn video."""


@dataclass(frozen=True)
class FramePacket:
    """Một frame đã lấy mẫu, sẵn sàng đưa vào detector."""

    frame: "cv2.Mat"  # noqa: F821 - type comment only
    frame_id: int          # frame index trong video gốc (0-based)
    sampled_index: int     # thứ tự trong chuỗi đã lấy mẫu (0-based)
    timestamp: datetime    # UTC, dựng từ start_time + frame_id / fps
    source_fps: float


def _resolve_source(source: Union[str, os.PathLike]) -> Union[str, int]:
    """CRTSP -> chuỗi; file -> str đường dẫn; '0' -> camera index int."""
    if isinstance(source, int):
        return source
    text = str(source)
    if text.isdigit():
        return int(text)
    return text


def read_video_info(source: Union[str, os.PathLike]) -> dict:
    """Đọc metadata cơ bản: fps, width, height, total_frames, duration_seconds."""
    resolved = _resolve_source(source)
    cap = cv2.VideoCapture(resolved)
    if not cap.isOpened():
        raise VideoOpenError(f"Không mở được video: {source}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    duration = (total / fps) if fps > 0 else 0.0
    return {
        "source_fps": fps,
        "width": width,
        "height": height,
        "total_frames": total,
        "duration_seconds": duration,
    }


def iter_frames(
    source: Union[str, os.PathLike],
    sample_fps: float = 5.0,
    resize_width: int = 0,
    skip_seconds: float = 0.0,
    start_time: datetime | None = None,
) -> Iterator[FramePacket]:
    """Sinh FramePacket theo thứ tự thời gian.

    Args:
        source: file path, RTSP URL, hoặc camera index dạng chuỗi số.
        sample_fps: tần số lấy mẫu. <= 0 hoặc lớn hơn fps gốc => giữ mọi frame.
        resize_width: nếu > 0, resize giữ tỉ lệ về chiều rộng này.
        skip_seconds: bỏ N giây đầu.
        start_time: thời điểm tương ứng frame 0. Mặc định = UTC now.
    """
    resolved = _resolve_source(source)
    cap = cv2.VideoCapture(resolved)
    if not cap.isOpened():
        raise VideoOpenError(f"Không mở được video: {source}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 25.0
        logger.warning("Không đọc được FPS gốc của %s, giả định %.1f", source, source_fps)

    effective_sample_fps = sample_fps if 0 < sample_fps < source_fps else source_fps
    step = max(1, int(round(source_fps / effective_sample_fps)))

    if start_time is None:
        start_time = datetime.now(timezone.utc)
    elif start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)

    skip_frames = max(0, int(round(skip_seconds * source_fps)))

    logger.info(
        "Mở video %s | fps=%.2f sample_fps=%.2f step=%d skip=%d",
        source, source_fps, effective_sample_fps, step, skip_frames,
    )

    frame_id = 0
    sampled_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id < skip_frames:
                frame_id += 1
                continue
            if (frame_id - skip_frames) % step != 0:
                frame_id += 1
                continue

            if resize_width and resize_width > 0 and frame.shape[1] != resize_width:
                scale = resize_width / float(frame.shape[1])
                new_h = max(1, int(round(frame.shape[0] * scale)))
                frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

            timestamp = start_time + timedelta(seconds=frame_id / source_fps)

            yield FramePacket(
                frame=frame,
                frame_id=frame_id,
                sampled_index=sampled_index,
                timestamp=timestamp,
                source_fps=source_fps,
            )
            frame_id += 1
            sampled_index += 1
    finally:
        cap.release()


def discover_video_files(directory: Union[str, os.PathLike]) -> list[Path]:
    """Liệt kê video file hỗ trợ trong thư mục (không đệ quy)."""
    exts = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


# --------------------------------------------------------------------------- MOTChallenge


def is_mot_sequence(path: Union[str, os.PathLike]) -> bool:
    """True nếu `path` là thư mục sequence MOTChallenge (có seqinfo.ini + img1/)."""
    p = Path(path)
    return p.is_dir() and (p / "seqinfo.ini").is_file() and (p / "img1").is_dir()


def read_mot_seqinfo(seq_dir: Union[str, os.PathLike]) -> dict:
    """Đọc seqinfo.ini của một sequence MOTChallenge.

    Trả về dict: name, im_dir, frame_rate, seq_length, width, height, im_ext.
    """
    seq_path = Path(seq_dir)
    ini_path = seq_path / "seqinfo.ini"
    if not ini_path.is_file():
        raise VideoOpenError(f"Không tìm thấy seqinfo.ini trong {seq_path}")

    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    if "Sequence" not in parser:
        raise VideoOpenError(f"seqinfo.ini thiếu section [Sequence]: {ini_path}")

    seq = parser["Sequence"]
    return {
        "name": seq.get("name", seq_path.name),
        "im_dir": seq.get("imDir", "img1"),
        "frame_rate": float(seq.get("frameRate", 25)),
        "seq_length": int(seq.get("seqLength", 0)),
        "width": int(seq.get("imWidth", 0)),
        "height": int(seq.get("imHeight", 0)),
        "im_ext": seq.get("imExt", ".jpg"),
    }


def list_mot_frames(seq_dir: Union[str, os.PathLike]) -> list[Path]:
    """Danh sách file ảnh theo thứ tự tên (000001.jpg, 000002.jpg, ...)."""
    info = read_mot_seqinfo(seq_dir)
    img_dir = Path(seq_dir) / info["im_dir"]
    if not img_dir.is_dir():
        raise VideoOpenError(f"Không tìm thấy thư mục ảnh: {img_dir}")
    ext = info["im_ext"].lower()
    frames = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() == ext)
    if not frames:
        raise VideoOpenError(f"Thư mục ảnh rỗng: {img_dir}")
    return frames


def iter_frames_from_image_dir(
    seq_dir: Union[str, os.PathLike],
    sample_fps: float = 5.0,
    resize_width: int = 0,
    skip_seconds: float = 0.0,
    start_time: datetime | None = None,
) -> Iterator[FramePacket]:
    """Sinh FramePacket từ thư mục ảnh của một sequence MOTChallenge.

    Khác `iter_frames()`:
    - `frame_id` khớp 1-1 với `gt/gt.txt` của MOT (1-based).
    - Chỉ đọc ảnh, không giải mã video -> giữ đúng độ phân giải gốc.

    Args:
        seq_dir: thư mục sequence (chứa seqinfo.ini và img1/).
        sample_fps: tần số lấy mẫu; <=0 hoặc >= fps gốc -> lấy mọi frame.
        resize_width: >0 thì resize giữ tỉ lệ. 0 = giữ nguyên (khuyến nghị khi
            đánh giá tracker để bbox khớp trực tiếp với gt.txt).
        skip_seconds: bỏ N giây đầu.
        start_time: mốc thời gian của frame đầu tiên; mặc định UTC now.
    """
    info = read_mot_seqinfo(seq_dir)
    frames = list_mot_frames(seq_dir)

    source_fps = info["frame_rate"] if info["frame_rate"] > 0 else 25.0
    effective_sample_fps = sample_fps if 0 < sample_fps < source_fps else source_fps
    step = max(1, int(round(source_fps / effective_sample_fps)))
    skip_frames = max(0, int(round(skip_seconds * source_fps)))

    if info["seq_length"] and len(frames) != info["seq_length"]:
        logger.warning(
            "Sequence %s: seqinfo ghi %d frame nhưng có %d ảnh (có thể giải nén thiếu).",
            info["name"], info["seq_length"], len(frames),
        )

    if start_time is None:
        start_time = datetime.now(timezone.utc)
    elif start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)

    logger.info(
        "Mở sequence MOT %s | images=%d fps=%.2f sample_fps=%.2f step=%d skip=%d size=%dx%d",
        info["name"], len(frames), source_fps, effective_sample_fps, step, skip_frames,
        info["width"], info["height"],
    )

    sampled_index = 0
    for idx, path in enumerate(frames):
        # MOT frame_id là 1-based: ảnh thứ idx -> frame_id = idx + 1
        frame_id = idx + 1
        if frame_id <= skip_frames:
            continue
        if (frame_id - 1 - skip_frames) % step != 0:
            continue

        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Không đọc được ảnh %s, bỏ qua.", path)
            continue

        if resize_width and resize_width > 0 and frame.shape[1] != resize_width:
            scale = resize_width / float(frame.shape[1])
            new_h = max(1, int(round(frame.shape[0] * scale)))
            frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

        timestamp = start_time + timedelta(seconds=(frame_id - 1) / source_fps)
        yield FramePacket(
            frame=frame,
            frame_id=frame_id,
            sampled_index=sampled_index,
            timestamp=timestamp,
            source_fps=source_fps,
        )
        sampled_index += 1


def iter_frames_auto(
    source: Union[str, os.PathLike],
    sample_fps: float = 5.0,
    resize_width: int = 0,
    skip_seconds: float = 0.0,
    start_time: datetime | None = None,
) -> Iterator[FramePacket]:
    """Chọn reader phù hợp: thư mục MOT sequence hoặc video/RTSP."""
    if is_mot_sequence(source):
        yield from iter_frames_from_image_dir(
            source,
            sample_fps=sample_fps,
            resize_width=resize_width,
            skip_seconds=skip_seconds,
            start_time=start_time,
        )
    else:
        yield from iter_frames(
            source,
            sample_fps=sample_fps,
            resize_width=resize_width,
            skip_seconds=skip_seconds,
            start_time=start_time,
        )
