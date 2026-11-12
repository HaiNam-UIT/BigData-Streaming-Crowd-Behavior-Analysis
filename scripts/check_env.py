"""Kiểm tra môi trường P2: Python, libs, GPU, video nguồn, config.

Chạy: .\\.venv\\Scripts\\python.exe scripts\\check_env.py
Exit code 0 = mọi thứ sẵn sàng; 1 = có mục cần xử lý.
"""

from __future__ import annotations

import shutil
import sys
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OK = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"

# Windows console mặc định cp1252 -> ép UTF-8 để in tiếng Việt
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def main() -> int:
    failures = 0
    warnings = 0

    print("=" * 72)
    print("SE363 - Environment check")
    print("=" * 72)

    # 1. Python
    print(f"\n[Python]")
    print(f"  version : {sys.version.split()[0]}")
    print(f"  exe     : {sys.executable}")
    print(f"  platform: {platform.platform()}")
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print(f"  {WARN} Nên dùng 3.11 hoặc 3.12 (hiện {sys.version_info.major}.{sys.version_info.minor})")
        warnings += 1
    else:
        print(f"  {OK} phiên bản ổn định")
    if ".venv" not in sys.executable.lower():
        print(f"  {WARN} Không chạy trong .venv; nên activate .venv trước")
        warnings += 1

    # 2. Libraries
    print("\n[Libraries]")
    libs = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("yaml", "pyyaml"),
        ("lap", "lapx"),
        ("jsonschema", "jsonschema"),
        ("pytest", "pytest"),
    ]
    for mod, pkg in libs:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {OK} {mod:<14} {ver:<16} ({pkg})")
        except ImportError as exc:
            print(f"  {FAIL} {mod:<14} missing ({pkg}) -> pip install {pkg}")
            print(f"         {exc}")
            failures += 1

    # 3. GPU
    print("\n[GPU]")
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  {OK} CUDA sẵn sàng: {name}")
        else:
            print(f"  {WARN} torch chỉ CPU. Nếu muốn nhanh hơn, cài torch CUDA (xem README).")
            warnings += 1
    except ImportError:
        print(f"  {FAIL} không import được torch")
        failures += 1

    # 4. OCRay
    print("\n[Video backend]")
    try:
        import cv2

        has_ffmpeg = "FFMPEG" in cv2.getBuildInformation().upper()
        print(f"  cv2 build: {'có FFmpeg' if has_ffmpeg else 'KHÔNG có FFmpeg (chỉ hỗ trợ codec cơ bản)'}")
        if not has_ffmpeg:
            print(f"  {WARN} Nên dùng opencv-python đầy đủ (FFmpeg) để đọc mp4.")
            warnings += 1
    except Exception:
        pass

    # 5. Config / schema / dirs
    print("\n[Repo files]")
    required = {
        "configs/app.yaml": True,
        "producer/video_sources.yaml": True,
        "schemas/raw_frame.schema.json": True,
        "schemas/crowd_metadata.schema.json": True,
        "schemas/alert_event.schema.json": True,
        "ai_service/preprocessing.py": True,
        "ai_service/app.py": True,
        ".env.example": True,
    }
    for rel, _ in required.items():
        p = REPO_ROOT / rel
        if p.exists():
            size = p.stat().st_size
            print(f"  {OK} {rel:<42} ({size} B)")
        else:
            print(f"  {FAIL} thiếu {rel}")
            failures += 1

    print("\n[Directories]")
    for rel in ["data/raw", "data/processed", "data/own_data/plaza_01",
                "artifacts/runs", "artifacts/models", "tests"]:
        p = REPO_ROOT / rel
        if p.exists():
            print(f"  {OK} {rel}")
        else:
            print(f"  {FAIL} thiếu thư mục {rel}")
            failures += 1

    # 6. Sample video
    print("\n[Sample video]")
    raw = REPO_ROOT / "data" / "raw"
    vids = sorted(
        list(raw.glob("*.mp4")) + list(raw.glob("*.avi")) + list(raw.glob("*.mkv")) +
        list(raw.glob("*.mov")),
    ) if raw.exists() else []
    if vids:
        for v in vids:
            size_mb = v.stat().st_size / (1024 * 1024)
            print(f"  {OK} {v.name} ({size_mb:.1f} MB)")
    else:
        print(f"  {WARN} chưa có video trong data/raw/. Xem README để biết nơi lấy.")
        warnings += 1

    # Summary
    print("\n" + "-" * 72)
    print(f"Kết quả: {failures} fail, {warnings} warn")
    if failures:
        print("-> Cần fix mục [FAIL] trước khi chạy ai_service/app.py")
    elif warnings:
        print("-> Có thể chạy app.py; các [WARN] không chặn.")
    else:
        print("-> Môi trường sẵn sàng.")
    print("-" * 72)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
