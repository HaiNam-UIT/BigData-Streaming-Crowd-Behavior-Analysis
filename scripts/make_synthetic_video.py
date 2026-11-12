"""Tạo video tổng hợp để smoke test pipeline (không phải dữ liệu thật).

Mục đích: chứng minh ai_service/app.py chạy được end-to-end khi chưa có video plaza.
Video chứa vài hình người đơn giản (rectangle) di chuyển -> YOLO có thể không detect
(person synthetic quá đơn giản), nhưng ít nhất kiểm tra được decode/ghi CSV/ghi video.

Nếu YOLO không detect ai, đó là bình thường với video synthetic.
Dùng video thật (xem README) để đánh giá chất lượng.

Chạy: .\\.venv\\Scripts\\python.exe scripts\\make_synthetic_video.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "raw" / "synthetic_smoke.mp4"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

WIDTH, HEIGHT, FPS, SECONDS = 640, 360, 20, 8


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        print(f"Không mở được VideoWriter: {OUT}")
        return 1

    total = FPS * SECONDS
    for i in range(total):
        frame = np.full((HEIGHT, WIDTH, 3), 210, dtype=np.uint8)
        # đường ngang làm mốc
        cv2.line(frame, (0, HEIGHT // 2), (WIDTH, HEIGHT // 2), (150, 150, 150), 1)

        # 3 "người": hình chữ nhật + đầu tròn, di chuyển trái->phải với tốc độ khác nhau
        t = i / FPS
        entities = [
            (int(40 + 60 * t) % WIDTH, HEIGHT // 2 + 20, (30, 80, 200)),
            (int(180 + 45 * t) % WIDTH, HEIGHT // 2 - 60, (200, 80, 30)),
            (int(320 + 70 * t) % WIDTH, HEIGHT // 2 + 60, (30, 160, 60)),
        ]
        for cx, cy, color in entities:
            x1, y1 = cx - 12, cy - 55
            x2, y2 = cx + 12, cy + 55
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.circle(frame, (cx, y1 - 10), 11, (200, 170, 140), -1)

        cv2.putText(frame, f"t={t:5.2f}s", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    size_kb = OUT.stat().st_size / 1024
    print(f"Đã tạo {OUT} ({WIDTH}x{HEIGHT}, {SECONDS}s, {size_kb:.0f} KB)")
    print("Lưu ý: video synthetic KHÔNG dùng làm kết quả AI trong báo cáo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
