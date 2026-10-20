# Hướng Dẫn Triển Khai Gói Code Modular Trên Google Drive & Thin Runner Colab
## Dự Án SE374.R11: Phân Vùng Mô Bệnh Học WSSS Trên Tập Dữ Liệu BCSS-WSSS

Tài liệu này hướng dẫn cách tải gói mã nguồn modular `SE374_WSSS_Drive_Code` lên Google Drive và chạy các **Thin Runner Notebook** mới trên Google Colab mà **không nhồi nhét hàng nghìn dòng code vào trong notebook**, đồng thời bảo toàn nguyên vẹn 100% các notebook cũ và lịch sử Git.

---

### 1. Kiến Trúc Tách Biệt: Code Trong Thư Mục Drive (`src/`) + Notebook Gọi Hàm (Thin Runner)

Thay vì đặt toàn bộ kiến trúc mạng nơ-ron sâu và thuật toán xử lý hậu kỳ vào các ô (cells) của file `.ipynb`, hệ thống được tách thành 2 phần rõ ràng:

1. **Gói Mã Nguồn Modular (`SE374_WSSS_Drive_Code/src/`):**
   - `src/models/wsss_inference_methods.py`: Chứa kiến trúc Stage-1 `WSSSTissueStage1NetCAM` (`ResNet-38d` + `Net_CAM`, 105.09M tham số), Stage-2 `DeepLabV3+` (`ResNet-101`, 59.34M tham số) và bộ máy suy luận đa phương pháp `MultiMethodWSSSEngine` nâng cấp giúp tăng tỉ lệ mIoU và Macro Dice.
   - `src/pipeline/`: Chứa bộ điều phối thực nghiệm tự động tải dữ liệu `BCSS-WSSS.zip`, nạp bộ trọng số đôi (Stage-1 & Stage-2), chạy suy luận cuốn chiếu (streaming inference), xuất biểu đồ lâm sàng 300 DPI và đóng gói kết quả `.zip`.
   - `src/datasets/`, `src/losses_metrics/`, `src/visualization/`, `src/utils/`: Các dịch vụ đọc dữ liệu phân tách theo bệnh nhân (Patient-Level Split), lọc mô học Otsu Clinical Gating và kiểm định chất lượng.

2. **Các File Thin Runner Notebook Mới (Gọn Nhẹ - Không Đè Notebook Cũ):**
- `SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb`
- `SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb`
- `SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`
- `SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb`

3. **Các Phương Pháp Suy Luận SOTA Tích Hợp Trong `MultiMethodWSSSEngine`:**
- `M1_Baseline_SingleScale`
- `M2_MultiView_D4_TTA`
- `M3_LongTail_Prior_Calibration`
- `M4_Spatial_Bilateral_Refinement`
- `M5_Proposed_SOTA_Ensemble`
- `M6_MultiScale_Pyramid_D4_Ensemble`
- `M7_Proposed_Ultra_SOTA_Curriculum`

---

### 2. Hướng Dẫn Đưa Code Lên Google Drive Chỉ Với 1 Bước (Chọn 1 Trong 2 Cách)

#### Cách 1: Tải File Nén `SE374_WSSS_Drive_Code.zip` (Khuyên Dùng - Nhanh Nhất)
1. Mở Google Drive cá nhân của bạn tại `https://drive.google.com/`.
2. Tạo thư mục `SE374_WSSS_Project` ngay trong `MyDrive` (đường dẫn: `MyDrive/SE374_WSSS_Project/`).
3. Tải trực tiếp file nén **`SE374_WSSS_Drive_Code.zip`** (và các file Thin Runner `.ipynb` nếu muốn mở trực tiếp từ Drive) vào thư mục `MyDrive/SE374_WSSS_Project/`.
4. Khi chạy trên Google Colab, Ô số 1 của Thin Runner Notebook sẽ tự động tìm thấy `SE374_WSSS_Drive_Code.zip`, giải nén trực tiếp vào ổ đĩa tốc độ cao `/content/SE374_WSSS_Drive_Code` trên máy ảo Colab và gắn vào `sys.path`.

#### Cách 2: Tải Trực Tiếp Thư Mục `SE374_WSSS_Drive_Code/` Lên Google Drive
1. Kéo thả toàn bộ thư mục **`SE374_WSSS_Drive_Code/`** từ máy tính vào trong `MyDrive/SE374_WSSS_Project/` trên Google Drive (sao cho đường dẫn là `MyDrive/SE374_WSSS_Project/SE374_WSSS_Drive_Code/src/`).
2. Khi bạn sửa bất kỳ file `.py` nào trong `MyDrive/SE374_WSSS_Project/SE374_WSSS_Drive_Code/src/`, Thin Runner Notebook trên Colab sẽ tự động nhận diện ngay lập tức thông qua cơ chế `%autoreload 2`.

---

### 3. Bảng Lựa Chọn Thin Runner Notebook Theo Cấu Hình GPU Trên Google Colab

| Phiên Bản Thin Runner Notebook | Loại GPU Colab | Cần Bật High-RAM? | Quy Mô Đánh Giá | Thời Gian Ước Tính |
| :--- | :---: | :---: | :---: | :---: |
| **T4 Sanity 150** | `T4 GPU` (Free) | **KHÔNG CẦN** | 150 mảng ảnh thật (5 bệnh nhân x 30 ảnh) | ~1 - 2 phút |
| **T4 Full 4986** | `T4 GPU` (Free) | **KHÔNG CẦN** | Toàn bộ 4.986 mảng ảnh test BCSS-WSSS | ~3 - 4 phút |
| **L4 Full 4986** | `L4 GPU` (Pro) | **KHÔNG CẦN** | Toàn bộ 4.986 mảng ảnh test BCSS-WSSS | ~1.5 - 2 phút |
| **A100 Full 4986** | `A100 GPU` (Pro+) | Tùy chọn (Chỉ bật khi Train) | Toàn bộ 4.986 mảng ảnh test BCSS-WSSS | ~45 - 60 giây |

---

### 4. Quy Trình 4 Bước Chạy Thin Runner Notebook Trên Google Colab

1. **Bước 1:** Mở một trong các file Thin Runner Notebook trên Google Colab. Vào menu `Runtime` -> `Change runtime type` -> Chọn `T4 GPU`, `L4 GPU` hoặc `A100 GPU`.
2. **Bước 2:** Chạy Ô kết nối Google Drive (`drive.mount('/content/drive')`) để notebook tự động nạp gói code `src/` từ `SE374_WSSS_Drive_Code`.
3. **Bước 3:** Chạy các hàm điều phối từ `src.pipeline` để tự động tải dữ liệu thật `BCSS-WSSS.zip` (từ cổng HuggingFace CDN chống giới hạn băng thông Google Drive) và 2 bộ trọng số chính thức (`stage1_checkpoint_trained_on_bcss.pth` & `stage2_checkpoint_trained_on_bcss.pth`).
4. **Bước 4:** Chạy hàm đánh giá toàn diện các phương pháp (`M1` -> `M6`/`M7`), xem bảng so sánh định lượng mIoU/mDice ngay trên notebook và nhận file nén kết quả `.zip` tự động tải về máy tính.

---

### 5. Bảo Toàn 100% Các Notebook Cũ, Lịch Sử 5 Phiên Thực Nghiệm (Run 1 -> Run 5) & Lịch Sử Git

- **5 Notebook Standalone Gốc (`notebooks/` & `SE374_WSSS_Drive_Code/notebooks/legacy_standalone_v1/`):**
  - `SE374_Colab_WSSS_Pipeline.ipynb`, `SE374_Colab_WSSS_T4_Sanity150.ipynb`, `SE374_Colab_WSSS_T4_Full4986.ipynb`, `SE374_Colab_WSSS_L4_Full4986.ipynb`, `SE374_Colab_WSSS_A100_Full4986.ipynb` được bảo toàn nguyên vẹn 100%, không bị xóa hay ghi đè.
- **5 Notebook Lịch Sử Đối Chứng Từng Bước Nghiên Cứu (`notebooks/history_runs/` & `SE374_WSSS_Drive_Code/notebooks/history_runs/`):**
  - `Run1_20261019_Naive_HSV_Heuristic_Sanity150.ipynb` (`mIoU_4cls = 9.13%`, `mIoU_5cls = 9.78%`)
  - `Run2_20261020_0656UTC_DeepLabV3Plus_ImageNetNorm_OtsuAggressive_Sanity150.ipynb` (`mIoU_4cls = 5.74%`, `mIoU_5cls = 7.59%`)
  - `Run3_20261020_1008UTC_DeepLabV3Plus_UnitRangeNorm_Fixed_Sanity150.ipynb` (`mIoU_4cls = 72.39%`, `mIoU_5cls = 69.99%`)
  - `Run4_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb` (`mIoU_4cls = 68.71%`, `mIoU_5cls = 67.36%`)
  - `Run5_20261020_1641UTC_DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4_Full4986.ipynb` (`M1 = 69.34% -> M5 = 70.49%` 4 lớp / `76.38%` 5 lớp)
- **Báo Cáo Số Liệu Đối Chứng Qua 5 Phiên Thực Nghiệm (`experiments/` & `SE374_WSSS_Drive_Code/experiments/`):**
  - `experiments/benchmarks/COLAB_5_RUNS_EVOLUTION_COMPARISON.md` cùng đầy đủ các báo cáo `COLAB_EVALUATION_REPORT.md` và `COLAB_EVALUATION_REPORT.json` của cả 5 phiên chạy.
- Để cập nhật lại gói `drive_upload_ready/` sau khi chỉnh sửa code trong `src/`, chỉ cần chạy lệnh:
  ```bash
  cmd /c "python scripts/build_drive_package.py"
  ```
