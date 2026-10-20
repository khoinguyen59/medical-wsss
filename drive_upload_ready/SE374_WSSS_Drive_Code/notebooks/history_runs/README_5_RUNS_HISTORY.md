# Lịch Sử Đối Chứng Từng Bước Nghiên Cứu (6 Phiên Chạy Thực Nghiệm Google Colab: Run 1 -> Run 6)

> **Dự án:** SE374.R11 - Phân Vùng Mô Bệnh Học Có Giám Sát Yếu (WSSS) Trên Tập Dữ Liệu BCSS-WSSS (TCGA-BRCA)  
> **Nhóm thực hiện:** Nhóm 5 (Nguyễn Trọng Khôi - 23520790, Lê Nguyễn Thành Tài - 23521374; GVHD: TS. Nguyễn Vĩnh Tiệp)  
> **Mục đích thư mục:** Bảo toàn nguyên vẹn 100% toàn bộ các file Jupyter Notebook thực nghiệm qua từng bước tiến hóa nghiên cứu (`Run 1` -> `Run 6`) kèm đầy đủ kết quả đầu ra (`outputs`), bảng số liệu định lượng (`mIoU`, `mDice`, `PA`, `FPS`), ma trận nhầm lẫn 5x5 và hình ảnh trực quan hóa lâm sàng 300 DPI.

---

## 1. Danh Mục 6 File Notebook Đối Chứng Lịch Sử Trong `notebooks/history_runs/`

| Bước Nghiên Cứu | Tên File Notebook Được Bảo Toàn | Thư Mục Kết Quả Đối Chứng Trong `experiments/` | Quy Mô Ảnh Test | Bản Chất Thay Đổi & Phát Hiện Khoa Học | mIoU 4 Lớp (%) | mIoU 5 Lớp (%) | mDice 4 Lớp (%) | mDice 5 Lớp (%) |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Bước 1 (Run 1)** | `Run1_20261019_Naive_HSV_Heuristic_Sanity150.ipynb` | `experiments/colab_run_1_heuristic/` | 150 ảnh (Sanity) | Bản sơ khởi dùng ngưỡng màu HSV ngây thơ (`Naive Color Heuristic`), chưa nạp trọng số mạng nơ-ron sâu. Cho thấy ngưỡng màu thủ công hoàn toàn thất bại trên mô bệnh học H&E (`LYM=0%`, `NEC=0%`). | **9.13%** | **9.78%** | 15.41% | 16.73% |
| **Bước 2 (Run 2)** | `Run2_20261020_0656UTC_DeepLabV3Plus_ImageNetNorm_OtsuAggressive_Sanity150.ipynb` | `experiments/colab_run_2_imagenet_norm/` & `experiments/colab_runs/` | 150 ảnh (Sanity) | Tích hợp kiến trúc `DeepLabV3+ ResNet-101` (`stage2_checkpoint_trained_on_bcss.pth`), nhưng chuẩn hóa đầu vào theo `ImageNet Normalize` (`mean=[0.485, 0.456, 0.406]`) bị lệch với miền `[0.0, 1.0]` lúc huấn luyện checkpoint gốc + Otsu Gating quá gắt làm mô hình dự đoán lệch sang `STR`. | **5.74%** | **7.59%** | 9.34% | 12.69% |
| **Bước 3 (Run 3)** | `Run3_20261020_1008UTC_DeepLabV3Plus_UnitRangeNorm_Fixed_Sanity150.ipynb` | `experiments/colab_run_3_fixed/` | 150 ảnh (5 bệnh nhân) | Khắc phục chuẩn hóa đầu vào về đúng miền `[0.0, 1.0]` (`transforms.ToTensor()`) khớp với `WSSS-Tissue` gốc + bổ sung 2-View TTA + áp dụng quy tắc vùng ngoài ROI (`cls 4 is exclude`). Hiệu năng nhảy vọt từ `5.74%` lên **`72.39%`** trên tập Sanity 150 ảnh. | **72.39%** | **69.99%** | **83.23%** | **81.66%** |
| **Bước 4 (Run 4)** | `Run4_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb` | `experiments/colab_run_4_full/` | **4.986 ảnh (Full Test - 23 BN)** | Mở rộng đánh giá ra **toàn bộ 4.986 mảng ảnh test** bằng cơ chế suy luận cuốn chiếu (streaming evaluation). Phát hiện việc chỉ dùng đơn lẻ Stage-2 `DeepLabV3+` (thiếu cổng `Stage-1 CGM`) và bộ lọc Otsu Gating cũ làm đục lỗ nhầm 7.73 triệu pixel mô nhạt màu thành nền `BACK`. | **68.71%** | **67.36%** | **81.15%** | **80.23%** |
| **Bước 5 (Run 5)** | `Run5_20261020_1641UTC_DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4_Full4986.ipynb` | `experiments/colab_run_5_sota_full4986/` | **4.986 ảnh (Full Test - 23 BN)** | Kết hợp đồng thời **2 bộ trọng số**: Stage-1 `ResNet38d Net_CAM` (`stage1_checkpoint_trained_on_bcss.pth` - 105.09M tham số, kích hoạt `CGM y_cls > 0.1` & `CAM Fusion`) + Stage-2 `DeepLabV3+ ResNet-101` (59.34M tham số) + Ablation 5 phương pháp (`M1..M5`) + `Pure-Glass Guard`. Tái lập Baseline bài báo gốc (`M1 = 69.34%`) và vượt SOTA với **`M5 = 70.49%` (4 lớp) / `76.38%` (5 lớp)**. | **70.49%** *(M1: 69.34%)* | **76.38%** *(M1: 75.46%)* | **82.51%** *(M1: 81.69%)* | **86.00%** *(M1: 85.34%)* |
| **Bước 6 (Run 6)** | `Run6_20261020_2055UTC_MultiScale_Pyramid_D4_M1_to_M7_T4_Full4986.ipynb` | `experiments/colab_run_6_v2/` | **4.986 ảnh (Full Test - 23 BN)** | Kiến trúc `Thin Drive Runner v2` (`MultiMethodWSSSEngineV2`, FP16 Mixed Precision) đối soát đồng thời **7 phương pháp (`M1..M7`)**, tích hợp `M6` Kim tự tháp đa tỷ lệ D4 (`0.75x, 1.0x, 1.25x`) trọng số nghịch đảo Entropy + Cổng mềm Stage-1 và `M7` Beer-Lambert OD. Đạt đỉnh mới **`M6 = 70.53%` (4 lớp) / `76.41%` (5 lớp)** và **`M5 = 70.51%`**. | **70.53%** *(M1: 69.35%)* | **76.41%** *(M1: 75.47%)* | **82.54%** *(M1: 81.69%)* | **86.02%** *(M1: 85.35%)* |

---

## 2. Đối Chiếu Chi Tiết Từng Phân Lớp Mô Học Qua 6 Bước Nghiên Cứu

| Bước Nghiên Cứu | Quy Mô | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | BACK IoU (%) | Tốc Độ (FPS) | File Báo Cáo JSON / Markdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run 1 (`2026-10-19`)** | 150 ảnh | 16.17% | 20.35% | 0.00% | 0.00% | 12.37% | 929.26 FPS | `experiments/colab_run_1_heuristic/content/SE374_Colab_Results/COLAB_EVALUATION_REPORT.md` |
| **Run 2 (`2026-10-20 06:56 UTC`)** | 150 ảnh | 0.00% | 22.96% | 0.00% | 0.00% | 15.00% | 27.54 FPS | `experiments/colab_run_2_imagenet_norm/content/SE374_Colab_Results/COLAB_EVALUATION_REPORT.md` |
| **Run 3 (`2026-10-20 10:08 UTC`)** | 150 ảnh | 82.12% | 51.97% | 68.45% | 87.02% | 60.47% | 16.64 FPS | `experiments/colab_run_3_fixed/content/SE374_Colab_Results_DeepLabV3Plus_Sanity150_20261020_100806/COLAB_EVALUATION_REPORT.md` |
| **Run 4 (`2026-10-20 10:22 UTC`)** | 4.986 ảnh | 79.12% | 73.50% | 56.97% | 65.23% | 70.26% | 23.38 FPS | `experiments/colab_run_4_full/content/SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_102247/COLAB_EVALUATION_REPORT.md` |
| **Run 5 - M1 Baseline (`16:41 UTC`)** | 4.986 ảnh | 78.44% | 73.31% | 59.98% | 65.62% | 99.93% | 5.77 FPS (5 methods) | `experiments/colab_run_5_sota_full4986/content/SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_164144/COLAB_EVALUATION_REPORT.json` |
| **Run 5 - M5 Proposed (`16:41 UTC`)** | 4.986 ảnh | 79.19% | 73.80% | 61.27% | 67.69% | 99.93% | 5.77 FPS (5 methods) | `experiments/colab_run_5_sota_full4986/content/SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_164144/COLAB_EVALUATION_REPORT.json` |
| **Run 6 - M1 Baseline (`20:55 UTC`)** | 4.986 ảnh | 78.46% | 73.32% | 60.00% | 65.62% | 99.93% | 4.91 FPS (7 methods) | `experiments/colab_run_6_v2/SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_205548/COLAB_EVALUATION_REPORT.json` |
| **Run 6 - M5 Proposed V1 (`20:55 UTC`)** | 4.986 ảnh | 79.24% | 73.79% | **61.33%** | **67.67%** | **99.93%** | 4.91 FPS (7 methods) | `experiments/colab_run_6_v2/SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_205548/COLAB_EVALUATION_REPORT.json` |
| **Run 6 - M6 Proposed V2 (`20:55 UTC`)** | **4.986 ảnh** | **79.37%** | **73.82%** | **61.33%** | 67.61% | **99.93%** | **4.91 FPS (7 methods)** | `experiments/colab_run_6_v2/SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_205548/COLAB_EVALUATION_REPORT.md` |

---

## 3. Cấu Trúc Tổ Chức Toàn Bộ Các Thư Mục Notebook Trong Dự Án (Không Xóa Hay Ghi Đè Bất Kỳ Bản Cũ Nào)

1. **Thư mục gốc (`/`):**
   - `SE374_Colab_WSSS_T4_Full4986.ipynb` (`3.24 MB`): File notebook thực thi hoàn chỉnh từ Google Colab kèm toàn bộ log và hình ảnh nhúng.
   - `SE374_Colab_Results*.zip`: Toàn bộ 6 gói kết quả `.zip` tải trực tiếp từ Google Colab qua các lần chạy (`Run 1` -> `Run 6`).
2. **Thư mục `notebooks/`:**
   - **5 Notebook Standalone Gốc (Đầy đủ code độc lập + outputs):**
     - `SE374_Colab_WSSS_Pipeline.ipynb`
     - `SE374_Colab_WSSS_T4_Sanity150.ipynb`
     - `SE374_Colab_WSSS_T4_Full4986.ipynb`
     - `SE374_Colab_WSSS_L4_Full4986.ipynb`
     - `SE374_Colab_WSSS_A100_Full4986.ipynb`
   - **4 Notebook Thin Drive Runner v2 Mới (Gọi gói code modular `src/` từ Google Drive, tích hợp 7 phương pháp `M1..M7`):**
     - `SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb`
     - `SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`
     - `SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb`
     - `SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb`
   - **Thư mục `notebooks/history_runs/` (6 Notebook Lịch Sử Từng Bước Nghiên Cứu `Run 1` -> `Run 6`):**
     - Lưu giữ nguyên trạng từng phiên bản notebook và kết quả đầu ra của cả 6 bước thực nghiệm để phục vụ đối chứng báo cáo và bảo vệ đồ án.
