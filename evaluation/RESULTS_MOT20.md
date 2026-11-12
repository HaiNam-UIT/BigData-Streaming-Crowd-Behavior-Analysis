# Kết quả đánh giá AI trên MOT20 (P2/P3)

Sinh bằng `evaluation/evaluate_tracking.py`. Đây là số liệu cho **§15.1 Computer Vision**.

Cấu hình chung: YOLOv11n (pretrained COCO) + ByteTrack, `conf=0.30`, `iou=0.50`, `imgsz=960`,
`sample_fps=5` (25 gốc → bỏ 4/5 frame), CPU (torch 2.15.0+cpu), `resize_width=960`
(bbox được script nhân ngược `scale` để khớp gt).

## Bảng tổng hợp

| Sequence | GT người/frame | Pred/frame | **MOTA** | **IDF1** | IDP | IDR | ID switch | MAE | RMSE | FPS xử lý |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT20-02 (`min_vis=0`) | 55.6 | 20.5 | **28.5%** | 35.8% | 66.5% | 24.5% | 592 | 35.1 | 39.7 | 5.32 |
| MOT20-02 (`min_vis=0.2`) | 36.6 | 20.5 | **39.8%** | 46.0% | 64.1% | 35.9% | 569 | 16.1 | 18.7 | 5.32 |
| MOT20-05 (`min_vis=0.2`) | 169.4 | 17.3 | **6.8%** | 11.6% | 62.6% | 6.4% | 386 | 152.1 | 152.6 | 4.49 |

`min_vis=0.2` là chuẩn MOTChallenge (bỏ người bị che >80% cơ thể).

## Đọc kết quả

### 1. Precision cao, recall rất thấp

Ở cả hai sequence: **IDP 62–67%** nhưng **IDR 6–36%**.

- Khi YOLO nói "đây là người", khoảng 2/3 trường hợp là đúng (không có nhiều false positive).
- Nhưng nó **bỏ sót phần lớn người** trong đám đông dày.

Đây là hạn chế đã biết của detector chỉ train trên COCO cho cảnh crowd cực dày: người bị che, người xa, người ở rìa đám đông đều bị miss.

### 2. Càng đông càng sụp

| Sequence | Người/frame | MOTA | IDR |
|---|---:|---:|---:|
| MOT20-02 | 36.6 | 39.8% | 35.9% |
| MOT20-05 | 169.4 | 6.8% | 6.4% |

Tăng 4.6× mật độ → MOTA giảm 5.8×, IDR giảm 5.6×. Đây là bằng chứng định lượng cho yêu cầu **crowd counting model riêng** (§6.2) — detector generic không đủ cho plaza đông.

### 3. Đổi bộ lọc visibility ảnh hưởng lớn

Cùng MOT20-02:
- Giữ tất cả: MOTA 28.5%, MAE 35.1
- Bỏ người che >80%: MOTA 39.8%, MAE 16.1

Chênh lệch chủ yếu ở những người gần như không nhìn thấy — không AI nào detect được. **Khi báo cáo phải ghi rõ dùng `min_visibility` nào**, nếu không kết quả không so sánh được với paper khác.

### 4. Throughput

| Sequence | Frame xử lý | Thời gian | FPS |
|---|---:|---:|---:|
| MOT20-02 | 557 / 2782 | 104.7s | 5.32 |
| MOT20-05 | 663 / 3315 | 147.6s | 4.49 |

Trên **CPU** với ảnh 1920×1080, `imgsz=960`, `sample_fps=5`. Đây là baseline cho §15.3. Có CUDA sẽ nhanh hơn ~5–10×. Tăng `sample_fps` lên 25 sẽ chậm 5× và có thể tăng IDR (nhiều frame hơn để tracker bám).

## Cách tái tạo

```powershell
# 1. Chạy AI
.\.venv\Scripts\python.exe -m ai_service.app `
  --source data\public\MOT20\train\MOT20-02 `
  --camera-id CAM_MOT02 --run-id mot20_02_full --no-video --sample-fps 5

# 2. Đánh giá (chế độ nghiêm khắc, giữ mọi GT)
.\.venv\Scripts\python.exe -m evaluation.evaluate_tracking `
  --pred artifacts\runs\mot20_02_full\CAM_MOT02\tracking_results.csv `
  --seq  data\public\MOT20\train\MOT20-02

# 3. Đánh giá (chuẩn MOTChallenge, bỏ che >80%)
.\.venv\Scripts\python.exe -m evaluation.evaluate_tracking `
  --pred artifacts\runs\mot20_02_full\CAM_MOT02\tracking_results.csv `
  --seq  data\public\MOT20\train\MOT20-02 `
  --min-visibility 0.2 --out artifacts\eval\mot20_02_vis02
```

## Đoạn văn mẫu cho báo cáo

> **Đánh giá tracker trên MOT20.** Chúng tôi đánh giá YOLOv11n pretrained + ByteTrack
> trên 2 sequence train của MOT20 (MOT20-02: 2782 frame; MOT20-05: 3315 frame),
> lấy mẫu 5 fps. Lọc GT theo chuẩn MOTChallenge (`class=1`, `mark=1`, `visibility≥0.2`):
> MOT20-02 đạt MOTA 39.8%, IDF1 46.0%; MOT20-05 (mật độ 169 người/frame) chỉ đạt
> MOTA 6.8%, IDF1 11.6%. Precision (IDP) ổn định 62–64% trong khi recall (IDR)
> giảm từ 35.9% xuống 6.4% khi mật độ tăng 4.6 lần, cho thấy detector generic
> **bỏ sót phần lớn người trong đám đông dày**. Điều này định lượng nhu cầu
> bổ sung crowd counting model (§6.2) cho bối cảnh plaza đông người.
>
> Throughput trên CPU: 5.32 fps (MOT20-02), 4.49 fps (MOT20-05) với `imgsz=960`,
> `sample_fps=5`, ảnh gốc 1920×1080/1654×1080.
>
> *Lưu ý: MOT20 là dataset public dùng để đánh giá AI nền tảng, không đại diện
> cho hiệu năng tại plaza demo. Số liệu phục vụ §15.1, không thay thế đánh giá
> trên video plaza tự quay.*

## Giới hạn cần ghi trong báo cáo

| Giới hạn | Ảnh hưởng |
|---|---|
| Chỉ đánh giá trên train | Không so được với leaderboard MOTChallenge (test không có GT) |
| `sample_fps=5` thay vì 25 | IDR có thể thấp hơn thực tế (tracker ít thông tin hơn) |
| Chưa có HOTA | Metric bonus; cần TrackEval |
| `resize_width=960` | Có thể miss người nhỏ; thử `resize_width=0` để so |
| Chưa so baseline | Nên chạy `--no-track` để so detect-only vs detect+track |
