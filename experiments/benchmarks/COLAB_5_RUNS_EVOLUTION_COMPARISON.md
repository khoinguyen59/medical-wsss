# Đối Chiếu Lịch Sử 6 Phiên Thực Nghiệm Google Colab Trên Tập Dữ Liệu Thật BCSS-WSSS (TCGA-BRCA)

> **Dự án:** SE374.R11 - Weakly-Supervised Semantic Segmentation on Histopathology (BCSS-WSSS)  
> **Đối tượng đánh giá:** Toàn bộ 6 phiên chạy thực nghiệm thật trên Google Colab GPU (Tesla T4) từ bản sơ khởi đến bản SOTA 7 phương pháp thế hệ V2 hoàn chỉnh (`Run 1` -> `Run 6`).

---

## 1. Bảng Tổng Hợp Tiến Trình Cải Thiện Qua 6 Lần Chạy Thực Nghiệm Trên Google Colab (`Run 1` -> `Run 6`)

| Phiên Chạy (Run ID) | Quy Mô Ảnh Test | Kiến Trúc & Cơ Chế Cốt Lõi | mIoU 4 Lớp Mô (%) | mIoU 5 Lớp Toàn Cục (%) | mDice 4 Lớp (%) | mDice 5 Lớp (%) | PA 4 Lớp (%) | PA 5 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | BACK IoU (%) | Tốc Độ (FPS) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lần 1 (`colab_run_1_heuristic`)** | 150 ảnh (Sanity) | Ngưỡng màu HSV ngây thơ (Naive Color Heuristic, chưa nạp checkpoint thật) | 9.13% | 9.78% | 15.41% | 16.73% | - | - | 16.17% | 20.35% | 0.00% | 0.00% | 12.37% | 929.26 |
| **Lần 2 (`colab_runs` - `06:56 UTC`)** | 150 ảnh (Sanity) | DeepLabV3+ ResNet-101 nhưng bị lệch chuẩn hóa `ImageNet Normalize` + Otsu Gating quá gắt | 5.74% | 7.59% | 9.34% | 12.69% | - | 24.21% | 0.00% | 22.96% | 0.00% | 0.00% | 15.00% | 27.54 |
| **Lần 3 (`colab_run_3_fixed` - `10:08 UTC`)** | 150 ảnh (Sanity, 5 BN) | Sửa chuẩn hóa về `[0.0, 1.0]` (`ToTensor()`) + 2-View TTA + Quy tắc `cls 4 is exclude` | 72.39% | 69.99% | 83.23% | 81.66% | 83.59% | - | 82.12% | 51.97% | 68.45% | 87.02% | 60.47% | 16.64 |
| **Lần 4 (`colab_run_4_full` - `10:22 UTC`)** | **4.986 ảnh (Full Test, 23 BN)** | Chỉ có Stage-2 DeepLabV3+ đơn lẻ (chưa có Stage-1 `ResNet38d Net_CAM` CGM) + Otsu Gating | 68.71% | 67.36% | 81.15% | 80.23% | 84.94% | 83.77% | 79.12% | 73.50% | 56.97% | 65.23% | 70.26% | 23.38 |
| **Lần 5 (`colab_run_5_sota_full4986` - `16:41 UTC`)** | **4.986 ảnh (Full Test, 23 BN)** | Dual-Checkpoint (`Stage-1 ResNet38d Net_CAM` + `Stage-2 DeepLabV3+`) + 5 Methods (`M1..M5`, FP32): `M5` SOTA Ensemble + Pure-Glass Guard | 70.49% | 76.38% | 82.51% | 86.00% | 85.35% | 86.42% | 79.19% | 73.80% | 61.27% | 67.69% | 99.93% | 5.77 (5 methods) |
| **Lần 6 (`colab_run_6_v2` - `20:55 UTC` - MỚI NHẤT)** | **4.986 ảnh (Full Test, 23 BN)** | **Thin Drive Runner v2 (`MultiMethodWSSSEngineV2`, FP16 AMP, 7 Methods `M1..M7`): `M6` Kim Tự Tháp Đa Tỷ Lệ D4 (`0.75x, 1.0x, 1.25x`) Entropy-Weighted + Cổng Mềm Stage-1 (kèm `M5 = 70.51%`, `M7 = 70.44%`)** | **70.53%** | **76.41%** | **82.54%** | **86.02%** | **85.41%** | **86.47%** | **79.37%** | **73.82%** | **61.33%** | **67.61%** *(M5: 67.67%)* | **99.93%** | **4.91 (7 methods, 203.55 ms/patch)** |

---

## 2. Bảng Ablation Study Chi Tiết Của 7 Phương Pháp Trong Lần Chạy Thứ 6 Mới Nhất (`20261020_205548` - 4.986 Ảnh, 23 Bệnh Nhân)

| Mã | Tên Phương Pháp (`Method_Key` / Mô Tả) | mIoU 4 Lớp (%) | mDice 4 Lớp (%) | PA 4 Lớp (%) | mIoU 5 Lớp (%) | mDice 5 Lớp (%) | PA 5 Lớp (%) | TUM IoU / Dice (%) | STR IoU / Dice (%) | LYM IoU / Dice (%) | NEC IoU / Dice (%) | BACK IoU / Dice (%) | VTP / TSR (%) | Wilcoxon p-value | Student t-test p-value |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | `M1_Baseline_SingleScale`: Baseline WSSS-Tissue (Single-Scale) | 69.35% (`69.3496`) | 81.69% (`81.6941`) | 84.89% (`84.8903`) | 75.47% (`75.4654`) | 85.35% (`85.3480`) | 85.99% (`85.9905`) | 78.46% / 87.93% | 73.32% / 84.61% | 60.00% / 75.00% | 65.62% / 79.24% | 99.93% / 99.97% | 40.47% / 46.26% | `1.0000e+00` | `1.0000e+00` |
| **M2** | `M2_MultiView_D4_TTA`: + Đa Góc Nhìn Dihedral TTA & Stage-1 CGM | 70.30% (`70.3049`) | 82.37% (`82.3708`) | 85.34% (`85.3368`) | 76.23% (`76.2295`) | 85.89% (`85.8893`) | 86.40% (`86.4043`) | 79.08% / 88.32% | 73.88% / 84.97% | 60.75% / 75.58% | 67.51% / 80.61% | 99.93% / 99.97% | 40.81% / 46.56% | `1.1970e-23` | `3.6454e-22` |
| **M3** | `M3_LongTail_Prior_Calibration`: + Kết Hợp CAM Đa Giai Đoạn & Hiệu Chỉnh Đuôi Dài | 70.50% (`70.4955`) | 82.51% (`82.5127`) | 85.36% (`85.3552`) | 76.38% (`76.3821`) | 86.00% (`86.0029`) | 86.42% (`86.4214`) | 79.23% / 88.41% | 73.78% / 84.91% | 61.31% / 76.01% | 67.66% / 80.71% | 99.93% / 99.97% | 40.93% / 47.21% | `4.4895e-17` | `3.0920e-02` |
| **M4** | `M4_Spatial_Bilateral_Refinement`: + Tinh Chỉnh Biên Không Gian & Triệt Tiêu Đảo Nhiễu (ARML/PistoSeg) | 70.32% (`70.3194`) | 82.38% (`82.3811`) | 85.34% (`85.3446`) | 76.24% (`76.2411`) | 85.90% (`85.8976`) | 86.41% (`86.4115`) | 79.09% / 88.33% | 73.89% / 84.98% | 60.77% / 75.60% | 67.53% / 80.62% | 99.93% / 99.97% | 40.81% / 46.56% | `1.5627e-29` | `2.5581e-32` |
| **M5** | `M5_Proposed_SOTA_Ensemble`: Phương Pháp Đề Xuất Toàn Diện V1 (Full SOTA Pipeline) | 70.51% (`70.5095`) | 82.52% (`82.5225`) | 85.36% (`85.3635`) | 76.39% (`76.3932`) | 86.01% (`86.0107`) | 86.43% (`86.4290`) | 79.24% / 88.42% | 73.79% / 84.92% | **61.33% / 76.03%** | **67.67% / 80.72%** | 99.93% / 99.97% | 40.93% / 47.21% | `1.4249e-24` | `2.9636e-08` |
| **M6** | **`M6_MultiScale_Pyramid_D4_Ensemble`: + Kim Tự Tháp Đa Tỷ Lệ D4 (Entropy-Weighted) & Cổng Mềm Stage-1** | **70.53% (`70.5336`)** | **82.54% (`82.5364`)** | **85.41% (`85.4125`)** | **76.41% (`76.4125`)** | **86.02% (`86.0218`)** | **86.47% (`86.4745`)** | **79.37% / 88.50%** | 73.82% / 84.94% | **61.33% / 76.03%** | 67.61% / 80.68% | **99.93% / 99.97%** | **41.24% / 47.54%** | `2.6043e-15` | `7.7011e-01` |
| **M7** | `M7_Proposed_Ultra_SOTA_Curriculum`: Thế Hệ V2 (Multi-Scale D4 + H&E Beer-Lambert OD + Biên Đa Bước Nhảy) | 70.44% (`70.4399`) | 82.47% (`82.4709`) | 85.36% (`85.3617`) | 76.34% (`76.3376`) | 85.97% (`85.9694`) | 86.43% (`86.4274`) | 79.25% / 88.43% | 73.81% / 84.93% | 61.21% / 75.94% | 67.48% / 80.58% | 99.93% / 99.97% | 40.92% / 47.17% | `3.5899e-18` | `1.5656e-03` |

---

## 3. Phân Tích Chuyên Sâu Sự Tiến Hóa Giữa Lần 5 (`16:41 UTC`) Và Lần 6 Mới Nhất (`20:55 UTC`) Trên Toàn Bộ 4.986 Ảnh Test

1. **Tăng trưởng mIoU Toàn Cục từ `70.49%` (Run 5 M5) lên `70.51%` (Run 6 M5) và đạt đỉnh `70.53%` (Run 6 M6):**
   - Trong Lần 6 (`SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_205548`), việc kích hoạt suy luận hỗn hợp `FP16 Mixed Precision (torch.float16)` và chuẩn hóa luồng thực thi `MultiMethodWSSSEngineV2` giúp ngay cả cùng cấu hình `M1..M5` cũng tăng nhẹ độ chính xác (`M1`: `69.34% -> 69.35%`; `M5`: `70.49% -> 70.51%`, với `LYM IoU` đạt `61.33%` và `NEC IoU` đạt `67.67%`).
   - Phương pháp mới **`M6` (`M6_MultiScale_Pyramid_D4_Ensemble`)** kết hợp Kim tự tháp đa tỷ lệ (`0.75x, 1.0x, 1.25x`) có trọng số nghịch đảo Entropy cùng Cổng mềm Stage-1 (`Adaptive Soft-Gating`) đã đẩy chỉ số `mIoU_4cls` toàn cục lên mức kỷ lục mới **`70.53%`** (`+1.18%` so với Baseline M1 `69.35%`), `mIoU_5cls` lên **`76.41%`**, `mDice_4cls` lên **`82.54%`** và `PA_4cls` lên **`85.41%`**.
   - Đặc biệt trên phân lớp **Khối u (`TUM`)**, `M6` nâng `TUM IoU` từ `78.46%` (M1) và `79.19%` (Run 5 M5) lên **`79.37%`** (`Dice = 88.50%`), tăng số pixel `TUM` dự đoán đúng lên **`86.391.564` pixel** (nhiều hơn `+1.340.937` pixel so với M1 và `+495.121` pixel so với Run 5 M5).
2. **Đánh giá sự đánh đổi giữa `M5`, `M6` và `M7`:**
   - **`M6` đạt đỉnh trên toàn bộ các chỉ số tích lũy toàn cục (Global Confusion-Matrix Metrics):** `mIoU_4cls = 70.5336%`, `mIoU_5cls = 76.4125%`, `mDice_4cls = 82.5364%`, `PA_4cls = 85.4125%`, `TUM IoU = 79.3731%` và có ý nghĩa thống kê rất cao trên kiểm định phi tham số Wilcoxon (`p = 2.6043e-15`). Tuy nhiên, trên kiểm định tham số Student t-test tính trung bình cộng không trọng số theo từng patch đơn lẻ (`patch_miou_mean`), `M6` có `t_pvalue = 0.7701` do các patch kích thước nhỏ chỉ chứa 1 lớp mô khi phóng đại `1.25x` có phương sai biên lớn hơn.
   - **`M5` đạt sự cân bằng thống kê toàn diện nhất trên cả cấp độ Toàn cục, Patch và Bệnh nhân:** Đạt `mIoU_4cls = 70.5095%` (chỉ kém M6 `0.024%`), dẫn đầu tuyệt đối về `LYM IoU` (`61.3297%`), `NEC IoU` (`67.6728%`), dẫn đầu `patient_miou_mean` trên 23 bệnh nhân (`58.4096% +/- 13.3205%`) và đạt ý nghĩa thống kê cực mạnh trên cả hai phép kiểm định (`Wilcoxon p = 1.4249e-24` và `Student t-test p = 2.9636e-08`).
   - **`M7` (`70.4399% mIoU_4cls`) đạt Độ nhạy (Recall) cao nhất trên lớp Hoại tử (`NEC Recall = 81.0147%`, `7.663.678` pixel đúng):** Phép biến đổi mật độ quang học `H&E Beer-Lambert Optical Density` giúp phát hiện thêm gần `60.000` pixel hoại tử nhạt màu so với `M5`/`M6`, nhưng làm tăng nhẹ dương tính giả từ `TUM -> NEC` (`791.918` pixel so với `737.271` của `M6`), khiến `NEC Precision` giảm nhẹ xuống `80.16%`. Cả `Wilcoxon p = 3.5899e-18` và `Student t-test p = 1.5656e-03` của `M7` đều có ý nghĩa thống kê vững chắc (`p < 0.01`).

---

## 4. Đối Chiếu Với Các Bài Báo Khoa Học SOTA Quốc Tế Trên Cùng Tập Test BCSS-WSSS

| Phương Pháp / Công Trình Công Bố | Nơi Công Bố | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | mIoU 4 Lớp (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistoSegNet** (Chan et al.) | ICCV 2019 | 56.92% | 57.53% | 32.31% | 40.91% | 46.92% |
| **SEAM** (Wang et al.) | CVPR 2020 | 74.52% | 68.09% | 50.19% | 52.84% | 61.41% |
| **WSSS-Tissue (Stage-1 Multi-Layer Pseudo-Mask)** (Han et al.) | MedIA 2022 | 77.59% | 72.62% | 56.40% | 61.58% | 67.05% |
| **WSSS-Tissue (Stage-2 DeepLabV3+ Paper Reported)** (Han et al.) | MedIA 2022 | 78.44% | 73.31% | 59.98% | 65.62% | 69.34% |
| **LowMagWSS (40x Reference Benchmark)** (Dung et al.) | arXiv 2024 | 78.44% | 73.31% | 59.98% | 65.62% | 69.34% |
| **PistoSeg (Mosaic + RFM)** (Fang et al.) | AAAI 2023 | 78.90% | 73.60% | 60.50% | 66.80% | 69.95% |
| **SE374.R11 - M1 Baseline (Run 5 FP32 / Run 6 FP16 trên Colab T4)** | Đồ án SE374 (2026) | 78.46% | 73.32% | 60.00% | 65.62% | **69.35%** *(Khớp chuẩn paper gốc 69.34%)* |
| **SE374.R11 - M5 Proposed V1 Full SOTA Ensemble (Run 6 trên Colab T4)** | Đồ án SE374 (2026) | 79.24% | 73.79% | **61.33%** | **67.67%** | **70.51%** *(+1.16% vượt Baseline)* |
| **SE374.R11 - M6 Proposed V2 Multi-Scale Pyramid D4 (Run 6 trên Colab T4)** | Đồ án SE374 (2026) | **79.37%** | **73.82%** | **61.33%** | 67.61% | **70.53%** *(+1.18% vượt Baseline)* |
