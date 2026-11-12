# Đánh giá AI trên MOT20

Đánh giá tracker bằng metric chuẩn MOTChallenge, so `tracking_results.csv` với `gt/gt.txt`.

Kết quả dùng cho **§15.1 Computer Vision** trong `PROJECT_DESIGN_AND_DATA.md`.

## Quy ước quan trọng

| Điều | Giá trị |
|---|---|
| Chỉ đánh giá trên `data/public/MOT20/train/` | `test/` **không có** `gt.txt` |
| GT lọc `mark==1 and class==1` | Chỉ pedestrian hợp lệ, bỏ ô tô và vùng ignore |
| IoU match threshold | 0.5 (chuẩn MOTChallenge) |
| Scale | Script tự nhân nếu `frame_width` trong CSV ≠ `imWidth` của seqinfo |
| Frames | Chỉ những frame có trong CSV (vì `sample_fps=5` < 25 gốc) |

## Chạy

```powershell
# 1. Chạy AI trên sequence
.\.venv\Scripts\python.exe -m ai_service.app `
  --source data\public\MOT20\train\MOT20-02 `
  --camera-id CAM_MOT02 --run-id mot20_02 --no-video

# 2. Đánh giá
.\\.venv\Scripts\\python.exe -m evaluation.evaluate_tracking `
  --pred artifacts\runs\mot20_02\CAM_MOT02\tracking_results.csv `
  --seq  data\public\MOT20\train\MOT20-02
```

Output:

```
artifacts/runs/mot20_02/CAM_MOT02/
├── tracking_metrics.json       # MOTA/IDF1/MAE/RMSE + meta
└── counting_per_frame.csv      # gt_count vs pred_count từng frame
```

## Metric được tính

| Metric | Ý nghĩa | Cách đọc |
|---|---|---|
| `mota_pct` | Tỉ lệ track đúng tổng hợp (1 - error rate) | > 50% OK; > 70% tốt với YOLO nano |
| `idf1_pct` | Độ giữ nguyên ID giữa detection và GT | > 60% tốt |
| `idp_pct` | Precision của ID: bao nhiêu % ID dự đoán đúng | Cao = ít nhầm danh tính |
| `idr_pct` | Recall của ID: bao nhiêu % người được gán đúng ID | **Thấp = bỏ sót người** |
| `motp` | Trung bình 1 - IoU của các match | Thấp = box khớp sát |
| `num_switches` | Số lần tracker đổi ID sai | So với số object |
| `num_false_positives` | Detection "ma" (GT không có) | |
| `num_misses` | Người bị bỏ sót | |
| `counting_mae` | Sai số tuyệt đối trung bình của count/frame | |

## Lưu ý kỹ thuật

### Nếu bbox trong CSV đã resize

Tự động xử lý. Ví dụ: `resize_width: 960` → `frame_width=960` trong CSV, `imWidth=1920` trong seqinfo → script nhân `scale=2.0`.

Kiểm tra dòng `Scale` khi chạy. Nếu muốn tránh hoàn toàn, chạy AI với `resize_width: 0`:

```powershell
.\.venv\Scripts\python.exe -m ai_service.app `
  --source data\public\MOT20\train\MOT20-02 ... --no-video
# Sửa configs/app.yaml: video.resize_width: 0
```

### Bỏ qua người bị che nhiều

MOTChallenge có sequence rất đông, nhiều người chỉ thấy 10% cơ thể (`visibility` thấp). Thử:

```powershell
--min-visibility 0.2
```

Giữ `0.0` (mặc định) để giữ tất cả — nghiêm khắc nhất nhưng phản ánh đúng khó khăn của MOT20.

### HOTA

HOTA là metric chính của MOTChallenge từ 2020, nhưng cần `TrackEval` — công cụ yêu cầu cấu trúc thư mục cố định và file cấu hình riêng. Script này chưa tích hợp HOTA.

Cài nếu cần:

```powershell
git clone https://github.com/JonathonLuiten/TrackEval
pip install -r TrackEval/requirements.txt
```

Xem hướng dẫn trong repo TrackEval. **MOTA + IDF1 đã đủ cho đồ án SE363**; HOTA là bonus.

## Ghi kết quả vào báo cáo

Mẫu đoạn văn dùng được:

> **Đánh giá tracker trên MOT20-02** (2782 frame, 270 track, trung bình 39 người/frame):
> YOLOv11n + ByteTrack đạt MOTA 40.3%, IDF1 57.8%, IDP 89.3%, IDR 42.7%.
> Precision cao nhưng recall thấp cho thấy detector phát hiện đúng khi có detect
> nhưng bỏ sót 57% người do mật độ cao — phù hợp với khó khăn của MOT20.
> Counting MAE 20.4 người/frame.
>
> *Lưu ý: Đây là số liệu đánh giá AI nền tảng trên dataset public, không phải
> dữ liệu plaza. Kết quả phục vụ §15.1, không đại diện cho hiệu năng tại
> địa điểm demo.*
