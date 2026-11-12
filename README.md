# SE363 - Multi-Camera Plaza Crowd Streaming

Nền tảng phân tích mật độ và luồng di chuyển người tại quảng trường từ video đa camera,
tổng hợp bằng Kafka + Spark Structured Streaming và dự báo occupancy/flow.

Thiết kế đầy đủ: [PROJECT_DESIGN_AND_DATA.md](PROJECT_DESIGN_AND_DATA.md).

## Trạng thái hiện tại

| Phase | Nội dung | Trạng thái |
|---|---|---|
| P1 | Environment setup | ✅ `.venv` Python 3.11 + ultralytics/opencv/lap |
| P2 | Single-camera prototype (detect + track + CSV) | ⏳ code xong, chờ video để chạy |
| P3 | Public CV experiments (counting) | ⬜ |
| P4 | Plaza data collection + annotation | ⬜ |
| P5 | Multi-camera aggregation | ⬜ |
| P6 | Kafka + Spark Streaming | ⬜ |
| P7 | Forecasting | ⬜ |
| P8 | Dashboard | ⬜ |
| P9 | Evaluation | ⬜ |

## Yêu cầu hệ thống

- Windows 10/11 + PowerShell
- Python **3.11** (`py -3.11`) - ổn định nhất cho ultralytics trên Windows
- GPU tùy chọn: RTX 3050 có sẵn (torch mặc định là CPU; xem mục CUDA bên dưới)
- Không cần Docker cho P2

## Cài đặt (P2)

```powershell
# 1. Tạo venv với Python 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài dependencies
python -m pip install -U pip
python -m pip install -r ai_service\requirements.txt

# 3. Kiểm tra
python -c "import torch, ultralytics, cv2, lap, pandas; print('ok')"
```

Nếu PowerShell chặn `Activate.ps1`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Bật CUDA (tùy chọn, RTX 3050)

Torch mặc định chỉ chạy CPU. Muốn nhanh hơn ~5–10x:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision `
  --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

## Chạy P2: video -> YOLO -> ByteTrack -> CSV

1. Đặt 1 video mẫu vào `data/raw/` (xem hướng dẫn lấy video bên dưới).
2. Sửa `producer/video_sources.yaml` nếu tên file khác `sample.mp4`.
3. Chạy:

```powershell
# Smoke test 10 giây đầu, không xuất video (nhanh nhất)
.\.venv\Scripts\python.exe -m ai_service.app `
  --source data/raw/sample.mp4 --camera-id CAM01 `
  --no-video --limit-seconds 10

# Chạy đầy đủ (CSV + video annotated)
.\.venv\Scripts\python.exe -m ai_service.app `
  --source data/raw/sample.mp4 --camera-id CAM01

# Chạy mọi camera trong video_sources.yaml
.\.venv\Scripts\python.exe -m ai_service.app --all-cameras
```

Output:

```
artifacts/runs/<run_id>/CAM01/
├── tracking_results.csv    # một dòng/detection: frame, track_id, bbox, conf, center, bottom_center
├── camera_metrics.csv      # window 10s: unique_tracks, person_count_avg/max, detection_rows
├── motion_metrics.csv      # F03: mỗi track: speed, bearing, direction, path_length
├── motion_directions.csv   # F03: phân bố 8 hướng (E/NE/N/...)
├── line_crossings.csv      # F04: event IN/OUT qua virtual line (cần configs/geometry.yaml)
├── zone_occupancy.csv      # F05: occupancy/density theo zone window (cần geometry zones)
├── annotated.mp4           # video có bbox + track_id, để xem bằng mắt
└── run_metadata.json       # cấu hình, thống kê, FPS xử lý, thời gian
```

`motion_*.csv`, `line_crossings.csv`, `zone_occupancy.csv` chỉ sinh khi `configs/geometry.yaml` tồn tại và có `lines`/`zones` khớp `camera_id`.

## Cấu hình zone/line (F04 + F05)

Sửa `configs/geometry.yaml`:

```yaml
lines:
  - line_id: L_GATE_NORTH
    camera_id: CAM01
    edge_id: E_NC          # map sang zone graph trong PROJECT_DESIGN_AND_DATA.md §7
    x1: 100
    y1: 400
    x2: 860
    y2: 400
    inside_side: positive  # "trong" ở phía pháp tuyến dương
zones:
  - zone_id: Z_C
    camera_id: CAM01
    polygon: [[300, 100], [660, 100], [660, 400], [300, 400]]
    area_m2: 120.0         # điền sau hiệu chuẩn homography; null = chỉ đếm, không density
```

Toạ độ tính theo **pixel của frame SAU resize** (nếu `resize_width: 960` thì vẽ trên ảnh rộng 960px).

Để lấy toạ độ: trích 1 frame từ video (<code>cv2.imwrite</code>) rồi mở bằng Paint/Photoshop, đọc pixel.

Tắt geometry: `--geometry /dev/null` (hoặc trỏ vào file không tồn tại) → bỏ qua line/zone.

`run_metadata.json` chứa **số liệu throughput** dùng cho §15.3 của thiết kế.

## Chạy trên MOT20 (dữ liệu public có ground truth)

MOT20 nằm ở `data/public/MOT20/`. `app.py` tự nhận diện thư mục sequence qua `seqinfo.ini` — không cần convert mp4.

```powershell
# Smoke test 3 giây đầu (≈16 frame)
.\.venv\Scripts\python.exe -m ai_service.app `
  --source data\public\MOT20\train\MOT20-02 `
  --camera-id CAM_MOT02 --no-video --limit-seconds 3

# Chạy full MOT20-02 (2782 frame gốc -> ~556 frame sau khi sample 5fps)
.\.venv\Scripts\python.exe -m ai_service.app `
  --source data\public\MOT20\train\MOT20-02 `
  --camera-id CAM_MOT02 --no-video
```

### Đánh giá với ground truth

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate_tracking `
  --pred artifacts\runs\<run_id>\CAM_MOT02\tracking_results.csv `
  --seq  data\public\MOT20\train\MOT20-02
```

Sinh `tracking_metrics.json` (MOTA, IDF1, IDP/IDR, ID switches) và `counting_per_frame.csv` (MAE/RMSE).
Chi tiết: [evaluation/README.md](evaluation/README.md).

**Chỉ đánh giá được trên `train/`** (test không có gt.txt). Long sequence MOT20-05 (3315 frame) cũng dùng được.

## Lấy video mẫu

Xem mục "Nguồn video mẫu cho P2" trong README hoặc hỏi trực tiếp nhóm.
Với P2 chỉ cần **1 clip người đi bộ thực tế** (≥30 giây, có ít nhất vài người qua khung).
Ưu tiên clip plaza/nhà ga/phố đi bộ, không lấy clip xe cộ.

## Cấu trúc repo

```
ai_service/         cv: preprocessing (frame sampler), app (detect+track), crowd_counter*, motion_features*
producer/           video_sources.yaml (nguồn camera), camera_producer.py* (Kafka producer, P6)
streaming/          crowd_stream_job.py*, rules.py* (Spark Structured Streaming, P6)
schemas/            JSON schema: raw_frame, crowd_metadata, alert_event
configs/            app.yaml (config toàn cục), alert_rules.yaml*
storage/            schema.sql*, repository.py* (PostgreSQL, P6-P7)
dashboard/          app.py* (Streamlit, P8)
alerts/             notifier.py* (P8)
evaluation/         benchmark/evaluate scripts (P9)
data/               raw/ (video gốc), processed/, own_data/plaza_01/, manifests/
artifacts/          models/ (weights), runs/ (output theo run_id)
tests/              pytest
```

`*` = file khung chưa implement (sẽ làm ở phase tương ứng).

## Tiêu chí hoàn thành P2

- [ ] 1 video ≥30s chạy hết không crash
- [ ] `tracking_results.csv` có cột `track_id` không null cho phần lớn detection
- [ ] Có ≥ vài track tồn tại ≥10 frame liên tiếp
- [ ] `annotated.mp4` xem bằng mắt: bbox bao quanh người, không lệch nghiêm trọng
- [ ] `run_metadata.json` ghi được `fps_processed`
- [ ] Ghi lại kết quả vào báo cáo (không dùng số liệu ước lượng)

## Ghi chú kỹ thuật (bắt buộc nhớ)

- **Track ID chỉ có ý nghĩa trong một camera** - không ghép CAM01 và CAM02 (thiết kế §6.3).
- **Count giữa các camera không được cộng mù** - cần ROI/ownership để tránh đếm trùng (§19).
- **Synthetic chỉ dùng stress test** - không dùng làm kết quả AI (§5.3).
- `sample_fps=5` giảm tải ~6x so với 30fps; đủ cho tracking người đi bộ nhưng cần ghi rõ khi báo cáo.
- Đổi `configs/app.yaml` thay vì sửa code khi tinh chỉnh threshold.
