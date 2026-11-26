# ĐẶC TẢ DỮ LIỆU Y KHOA & GIAO THỨC CHỐNG RÒ RỈ BỆNH NHÂN
## Đồ Án: Hệ Thống Phân Vùng Ngữ Nghĩa Giám Sát Yếu Trên Ảnh Tiêu Bản Mô Bệnh Học Toàn Phần (WSSS for WSI)
### Môn Học: SE374.R11 - Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa (UIT, ĐHQG-HCM, 2026)

> **Giảng viên hướng dẫn:** TS. Bùi Cao Doanh  
> **Cơ sở khoa học:** Amgad et al. (*Bioinformatics 2019*), Chu Han et al. (*MedIA 2022*, arXiv `2110.08048`)  
> **Quy chuẩn bất biến:** Patient-Level Anti-Leakage (`INVARIANT-08`), Track A Purity (`0%` TCGA-BRCA overlap), Anti-Shallow-Glob Defensive Resolution (`INVARIANT-29`), Zero Mockup (`INVARIANT-07`)

---

## 1. Khảo Sát Hai Bộ Dữ Liệu Mô Bệnh Học Tiêu Chuẩn Quốc Tế

Hệ thống SE374.R11 sử dụng 2 bộ dữ liệu mô bệnh học chuẩn quốc tế đã được đối soát trực tiếp giữa bài báo gốc Han et al. (*MedIA 2022*), kho mã nguồn tham chiếu `repos/WSSS-Tissue/`, khóa phân chia `src/config/bcss_patient_splits.json`, dữ liệu thực trên đĩa `data/BCSS-WSSS/` và bộ kiểm định chéo cơ quan `src/stage2_v2/gate7_clinical_bench.py`.

### 1.1 Bộ Dữ Liệu BCSS-WSSS (Breast Cancer Semantic Segmentation - Weakly Supervised)

- **Nguồn gốc:** Trích xuất từ `151` vùng vi trường quan tâm (ROIs) của `151` bệnh nhân ung thư vú thuộc hệ thống quốc gia Hoa Kỳ **TCGA-BRCA** (The Cancer Genome Atlas Breast Invasive Carcinoma), công bố gốc bởi Amgad et al. (*Bioinformatics 2019*) và chuẩn hóa cho bài toán giám sát yếu bởi Han et al. (*MedIA 2022*).
- **Liên kết tải chính thức:**
  - Hugging Face Hub (trong `src/colab_pkg/downloader.py`): `https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip`
  - Google Drive Mirror ZIP (trong `src/colab_pkg/downloader.py`, `GDRIVE_DATASET_ID`): `https://drive.google.com/file/d/1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn/view`
  - Google Drive gốc của tác giả MedIA 2022 (`repos/WSSS-Tissue/README.md`): `https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77?usp=sharing`
  - Checkpoints tiền huấn luyện tác giả (`repos/WSSS-Tissue/README.md`): `https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing`
- **Quy mô chính xác (xác minh trên đĩa `data/BCSS-WSSS/` và `src/config/bcss_patient_splits.json`):** **`31,826` patches** kích thước `224 x 224` pixels (`50,176` pixels/patch), độ phóng đại quang học chuẩn lâm sàng **`40x` (`0.2500 MPP` - microns per pixel)**.
- **Phân bổ phân chia cấp bệnh nhân (`151` bệnh nhân, `0%` chồng lấp):**
  - **Tập Huấn luyện (`training/`):** **`106` bệnh nhân** (`23,422` patches). Chỉ có nhãn yếu đa nhãn cấp patch `y_cls` mã hóa trong tên tệp, hoàn toàn không có mặt nạ pixel vẽ tay.
  - **Tập Kiểm định (`val/`):** **`22` bệnh nhân** (`3,418` patches trong `val/img/` và `val/mask/`, gồm `2,153` patches đa lớp). Có mặt nạ Ground Truth cấp điểm ảnh hoàn chỉnh (`{0, 1, 2, 3, 4}`), dùng cho dừng sớm, chọn TTA, hiệu chuẩn nhiệt độ `beta`, chọn công thức tổ hợp 5-fold CV và hiệu chuẩn ngưỡng CGM.
  - **Tập Kiểm thử mù (`test/`):** **`23` bệnh nhân** (`4,986` patches trong `test/img/` và `test/mask/`). Có mặt nạ Ground Truth cấp điểm ảnh hoàn chỉnh (`{0, 1, 2, 3, 4}`), đóng băng tuyệt đối cho đánh giá cuối cùng và kiểm định thống kê.
- **Danh mục 5 phân lớp mô học BCSS-WSSS (khớp `repos/WSSS-Tissue/2_generate_PM.py:29-34` và `src/training/dataset_loader.py:42-48`):**
  1. `0: TUM` (Tumor Epithelium - Biểu mô khối u ác tính) - RGB `(255, 0, 0)` (Đỏ)
  2. `1: STR` (Tumor-Associated Stroma - Mô đệm liên kết) - RGB `(0, 255, 0)` (Xanh lá)
  3. `2: LYM` (Lymphocytic Infiltrate - Bạch huyết bào xâm nhập) - RGB `(0, 0, 255)` (Xanh dương)
  4. `3: NEC` (Necrosis - Mô hoại tử) - RGB `(153, 0, 255)` (Tím)
  5. `4: BACK` (Background / Optical White Glass / Outside ROI) - RGB `(255, 255, 255)` (Trắng). Ở pha suy luận, nền kính trắng được tách độc lập bằng `OtsuClinicalGating` + `Pure-Glass Guard` (`>= 512 px`), **tuyệt đối không sao chép hay tô đè nhãn GT `label == 4` vào dự đoán AI**.

### 1.2 Bộ Dữ Liệu LUAD-HistoSeg (Lung Adenocarcinoma Tissue Segmentation)

- **Nguồn gốc:** Gồm **`49` bệnh nhân** ung thư biểu mô tuyến phổi độc lập (`29` ca từ Bệnh viện Nhân dân tỉnh Quảng Đông - **GDPH** và `20` ca từ **TCGA-LUAD**), hoàn toàn **0% chồng lấp bệnh nhân** với quần thể ung thư vú TCGA-BRCA của BCSS-WSSS.
- **Liên kết tải chính thức:**
  - Hugging Face Hub (trong `src/colab_pkg/downloader.py`): `https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/LUAD-HistoSeg.zip`
  - Google Drive Mirror ZIP (trong `src/colab_pkg/downloader.py`, `GDRIVE_LUAD_DATASET_ID`): `https://drive.google.com/file/d/1WyWfv-JHjLwkHOiN8blINhBOkKi--iA1/view`
  - Google Drive gốc của tác giả MedIA 2022 (`repos/WSSS-Tissue/README.md`): `https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr?usp=sharing`
- **Quy mô chính xác:** **`17,285` patches** kích thước `224 x 224` pixels ở độ phóng đại **`10x` (`1.0 MPP`)** trên **`49` bệnh nhân** (`29` GDPH + `20` TCGA-LUAD).
- **Phân bổ phân chia theo patch (Han et al., *MedIA 2022*, `arXiv:2110.08048`):**
  - **Tập Huấn luyện (`training/`):** **`16,678` patches** (chỉ có nhãn yếu đa nhãn cấp patch).
  - **Tập Kiểm định (`val/`):** **`300` patches** (có dense pixel mask).
  - **Tập Kiểm thử (`test/`):** **`307` patches** (có dense pixel mask).
- **Danh mục 5 phân lớp mô học chính xác (xác minh trong `repos/WSSS-Tissue/2_generate_PM.py:21-27` và `src/stage2_v2/gate7_clinical_bench.py:34-35`):**
  1. `0: TE` (Tumor Epithelial - Biểu mô u phổi) - RGB `(205, 51, 51)`
  2. `1: NEC` (Necrosis - Mô hoại tử) - RGB `(0, 255, 0)`
  3. `2: LYM` (Lymphocyte - Bạch huyết bào) - RGB `(65, 105, 225)`
  4. `3: TAS` (Tumor-Associated Stroma - Mô đệm liên kết u) - RGB `(255, 165, 0)`
  5. `4: BACK` (Background / Alveolar Air Sacs - Khoảng trắng phế nang và nền kính) - RGB `(255, 255, 255)`
- **Ánh xạ lớp chéo cơ quan LUAD sang BCSS (`src/stage2_v2/gate7_clinical_bench.py:34-35`):**
  Do thứ tự mã lớp `1` và `3` đảo ngược giữa hai bộ dữ liệu (`NEC` là lớp `1` trong LUAD nhưng là lớp `3` trong BCSS; `TAS/STR` là lớp `3` trong LUAD nhưng là lớp `1` trong BCSS), mọi đánh giá chéo cơ quan bắt buộc áp dụng bảng tra cố định:
  ```python
  # LUAD-HistoSeg [0:TE, 1:NEC, 2:LYM, 3:TAS, 4:BACK] -> BCSS [0:TUM, 1:STR, 2:LYM, 3:NEC, 4:BACK]
  LUAD_TO_BCSS_LUT = [0, 3, 2, 1, 4]
  ```
  Hàm `remap_luad_mask_to_bcss()` bảo toàn nguyên vẹn giá trị bỏ qua `ignore_index = 255`.
- **Vai trò kép trong dự án SE374.R11:**
  1. Cung cấp trọng số khởi tạo miền mô học sạch (`stage1_checkpoint_trained_on_luad.pth`, 0% chồng lấp TCGA-BRCA) cho Stage-1 Champion `se374_arm_b_full`.
  2. Làm bộ kiểm định tổng quát hóa chéo cơ quan (Cross-Organ Benchmark) qua lớp `LUADBCSSAlignedDataset` trong `src/stage2_v2/gate7_clinical_bench.py` và notebook `notebooks/04b_Stage2_Gate7_Clinical_LUAD_Bench_G4.ipynb`.

### 1.3 Bảng Đối Chiếu Tổng Hợp Hai Bộ Dữ Liệu

| Thuộc Tính | BCSS-WSSS (TCGA-BRCA) | LUAD-HistoSeg (GDPH + TCGA-LUAD) |
| :--- | :--- | :--- |
| **Cơ quan / Bệnh học** | Ung thư biểu mô vú xâm lấn (Breast) | Ung thư biểu mô tuyến phổi (Lung) |
| **Tổng số bệnh nhân** | **151** (`106` Train / `22` Val / `23` Test) | **49** (`29` GDPH + `20` TCGA-LUAD) |
| **Tổng số patches (`224x224`)** | **31,826** (`23,422 / 3,418 / 4,986`) | **17,285** (`16,678 / 300 / 307`) |
| **Độ phóng đại / MPP** | **40x (`0.2500 MPP`)** | **10x (`1.0000 MPP`)** |
| **Thứ tự mã lớp (`0-4`)** | `0:TUM, 1:STR, 2:LYM, 3:NEC, 4:BACK` | `0:TE, 1:NEC, 2:LYM, 3:TAS, 4:BACK` |
| **Ánh xạ về không gian BCSS** | Đồng nhất `[0, 1, 2, 3, 4]` | `LUAD_TO_BCSS_LUT = [0, 3, 2, 1, 4]` |
| **Rò rỉ bệnh nhân nội bộ / chéo tập** | `0%` (`patient_overlap_count = 0`) | `0%` chồng lấp với TCGA-BRCA |

---

## 2. Cấu Trúc Thư Mục Thực Tế, Bóc Tách Tên Tệp & Dò Tìm Đa Tầng

### 2.1 Cấu Trúc Thư Mục Thực Tế Trên Đĩa (`data/`)

Bộ dữ liệu đầy đủ `data/BCSS-WSSS/` được loại trừ khỏi Git qua `.gitignore` để giữ kho mã nguồn gọn nhẹ; chỉ có tập mẫu nhỏ `data/sample_dataset/` được theo dõi trong Git nhằm phục vụ kiểm thử tự động (`pytest`) và chạy thử cục bộ không cần tải toàn bộ bộ dữ liệu.

| Đường Dẫn | Trạng Thái Lưu Trữ | Mô Tả Chi Tiết |
| :--- | :---: | :--- |
| `data/BCSS-WSSS/training/` | Cục bộ (gitignored) | `23,422` ảnh huấn luyện (`106` bệnh nhân), chỉ có nhãn yếu multi-hot trong tên tệp (`TCGA-...+[abcd].png`). |
| `data/BCSS-WSSS/val/img/` & `val/mask/` | Cục bộ (gitignored) | `3,418` ảnh kiểm định (`22` bệnh nhân) kèm mặt nạ dense mask chế độ `P` (`0-3` cho 4 lớp mô, `4` cho `BACK`). |
| `data/BCSS-WSSS/test/img/` & `test/mask/` | Cục bộ (gitignored) | `4,986` ảnh kiểm thử độc lập held-out (`23` bệnh nhân) kèm mặt nạ dense mask do bác sĩ giải phẫu bệnh gán nhãn. |
| `data/BCSS-WSSS/*.lst` | Cục bộ (gitignored) | Danh sách tệp `train.lst`, `val.lst`, `test.lst`, `Readme.txt` và `Example_for_using_image-level_label.py` từ bản phát hành gốc. |
| `data/sample_dataset/` (`train/`, `valid/`, `test/`) | Theo dõi trong Git | Tập mẫu mini (`4` ảnh `train`, `2` cặp `valid/img` + `valid/mask`, `2` cặp `test/img` + `test/mask`) dùng cho `pytest` và chạy thử nhanh. |
| `data/LUAD-HistoSeg/` (ung thư phổi) | Tải trực tiếp trên Cloud/Colab | Không lưu cục bộ trong `data/`; được tải tự động trên môi trường Colab/GPU thông qua `src/colab_pkg/downloader.py` khi chạy kiểm định chéo cơ quan. |

```text
data/
|-- BCSS-WSSS/                      # Bộ dữ liệu chính ung thư vú (Han et al., MedIA 2022, gitignored)
|   |-- training/                    # 23,422 PNGs (106 bệnh nhân, chỉ có nhãn yếu [tsln])
|   |   |-- TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101].png
|   |   `-- TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+100[1100].png
|   |-- val/
|   |   |-- img/                     # 3,418 PNGs (22 bệnh nhân)
|   |   |   `-- TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0.png
|   |   `-- mask/                    # 3,418 PNGs 8-bit Palette mode "P" (giá trị 0-4)
|   |       `-- TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0.png
|   |-- test/
|   |   |-- img/                     # 4,986 PNGs (23 bệnh nhân Held-Out Test)
|   |   |   `-- TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+0.png
|   |   `-- mask/                    # 4,986 PNGs 8-bit Palette mode "P" (giá trị 0-4)
|   |       `-- TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+0.png
|   `-- train.lst / val.lst / test.lst / Readme.txt
|-- sample_dataset/                  # Tập mẫu mini theo dõi trong Git cho pytest & dry-run
|   |-- train/                       # 4 ảnh mẫu huấn luyện
|   |-- valid/                       # 2 cặp img/ + mask/
|   `-- test/                        # 2 cặp img/ + mask/
`-- LUAD-HistoSeg/                   # Bộ dữ liệu phổi kiểm định chéo cơ quan (tải qua downloader.py)
    |-- training/                    # 16,678 PNGs (nhãn yếu [a b c d])
    |-- val/                         # 300 PNGs (img/ + mask/)
    `-- test/                        # 307 PNGs (img/ + mask/)
```

### 2.2 Quy Cách Bóc Tách Metadata & Nhãn Yếu Từ Tên Tệp

Trong kho mã nguồn gốc `repos/WSSS-Tissue/tool/GenDataset.py:63-68`, định dạng nhãn trong dấu ngoặc vuông khác nhau giữa hai bộ dữ liệu:
- **BCSS-WSSS (`data/BCSS-WSSS/training/`):** Vector 4 bit viết liền `[1101]` gắn sau chỉ số patch, ví dụ `TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101].png`.
- **LUAD-HistoSeg:** Vector 4 bit tách bởi khoảng trắng `[1 0 0 1]`.

Các hàm `parse_patch_filename()` (`src/training/dataset_loader.py:75-114`) và `_extract_weak_label()` (`src/datasets/bcss_dataset.py:122-132`) dùng biểu thức chính quy chuẩn hóa để bóc tách an toàn cả hai định dạng (`[1101]`, `[1 0 0 1]`, `[1,0,0,1]`):
- **Patient ID:** Bắt chuẩn tiền tố `TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}` (hoặc `GDPH[_-][A-Za-z0-9]+` trong `extract_luad_case_id()` tại `src/stage2_v2/gate7_clinical_bench.py:51-57`).
- **Vector nhãn yếu đa nhãn:**
  $$\mathbf{y} = [y_{\text{TUM}}, y_{\text{STR}}, y_{\text{LYM}}, y_{\text{NEC}}] \in \{0, 1\}^4$$
- **Đọc mặt nạ chế độ Palette (`mode == "P"`):** Mọi DataLoader đọc mặt nạ bằng `_load_mode_p_mask_array()` (`src/training/dataset_loader.py`), tuyệt đối giữ nguyên chỉ số lớp gốc `{0, 1, 2, 3, 4}` và giá trị bỏ qua `ignore_index = 255` (không chuyển đổi `.convert("L")` lên ảnh palette gây sai lệch nhãn màu).

### 2.3 Giao Thức Dò Tìm Thư Mục Lồng Nhau & Phòng Thủ (`INVARIANT-29`)

Do các gói nén trên Google Drive và Hugging Face có thể giải nén thành nhiều cấp thư mục khác nhau (`training/`, `train/`, `train/img/`, `BCSS-WSSS/training/`, `BCSS-WSSS/train/img/`, `valid/img/`), **nghiêm cấm giả định thư mục phẳng bằng `glob("*.png")`**. Toàn bộ hệ thống tuân thủ cơ chế phòng thủ 3 tầng:

1. **Tự động phân giải đường dẫn đa tầng (`resolve_split_path` & `resolve_split_dirs`):**
   - Trong các notebook huấn luyện (`notebooks/01_*`, `02_*`, `03_*`): hàm `resolve_split_path` dò tìm tuần tự mọi cấu trúc lồng nhau cho `train / val / test`.
   - Trong gói điều phối (`src/colab_pkg/downloader.py:90-184`, `src/colab_pkg/evaluator.py`, và các notebook Gate-7 `02b`, `02c`, `02d`, `04b`): hàm `resolve_split_dirs()` quét đệ quy `**/{split}/img`, tự động bỏ qua `sample_dataset` và thư mục mã nguồn repo (`medical-wsss`, `repos`), đồng thời áp đặt ngưỡng kiểm chứng thực tế ở Pass 1 (`>= 10` ảnh và `>= 2` bệnh nhân độc lập) để chống ảo giác kiểm thử trên dữ liệu mẫu nhỏ.
2. **Quét ảnh đệ quy `rglob("*.png")`:**
   - Các bộ nạp dữ liệu lõi gồm `find_image_files()` (`src/training/dataset_loader.py:158-176`), `BCSSStage1TrainDataset` (`:435-443`), `BCSSStage1ValDataset` (`:504-506`), `memmap_dataset.py` (`:59`) và `LUADBCSSAlignedDataset` (`src/stage2_v2/gate7_clinical_bench.py:69-74`) đều tự động fallback sang `rglob("*.png")` nếu thư mục đầu vào chứa thư mục con `img/`.
3. **Chặn sập DataLoader rỗng (`Zero-File Fail-Fast`):**
   - `BCSSStage1TrainDataset` (`src/training/dataset_loader.py:440-443`) và `LUADBCSSAlignedDataset` (`src/stage2_v2/gate7_clinical_bench.py:73-74`) kiểm tra `len(files) == 0` ngay tại `__init__` và ném `FileNotFoundError` tường minh; `BCSSStage1ValDataset` (`src/training/dataset_loader.py:511-515`) ném `ValueError` khi bộ lọc bệnh nhân `strict_patient_filter=True` khớp `0` ảnh; và `SoftPseudoMaskMemmapDataset` (`src/stage2_v2/memmap_dataset.py:927-930`) ném `FileNotFoundError` ngay nếu thiếu tệp memmap hoặc manifest.

---

## 3. Hợp Đồng Chuẩn Hóa Đầu Vào (`Input Normalization Contracts`)

Hệ thống quản lý nghiêm ngặt hai miền chuẩn hóa đầu vào tương ứng với nguồn gốc trọng số khởi tạo nhằm ngăn lệch phân phối BatchNorm/LayerNorm:

| Miền Chuẩn Hóa | Công Thức Biến Đổi | Mô Hình / Checkpoint Áp Dụng | Tệp Khóa Hợp Đồng |
| :--- | :--- | :--- | :--- |
| **`zero_one` (`[0.0, 1.0]`)** | $x = \text{RGB} / 255.0$ (KHÔNG trừ `IMAGENET_MEAN/STD`) | Pretrained tác giả WSSS-Tissue (`stage1_checkpoint_trained_on_bcss.pth`, `stage2_checkpoint_trained_on_bcss.pth`, `stage1_checkpoint_trained_on_luad.pth`) và Stage-1 Champion `se374_arm_b_full` (`input_norm = "zero_one"`) | `src/colab_pkg/colab_runner.py`, `src/colab_pkg/evaluator.py`, `src/training/eval_stage1_full.py` (`load_stage1_strict`), `src/visualization/api_server.py` |
| **`imagenet` (ImageNet Norm)** | $x = (\text{RGB}/255.0 - \mu_{\text{IN}}) / \sigma_{\text{IN}}$ với $\mu_{\text{IN}} = (0.485, 0.456, 0.406)$, $\sigma_{\text{IN}} = (0.229, 0.224, 0.225)$ | Toàn bộ backbone khởi tạo từ ImageNet-1K hoặc LVD-142M: `src/training/train_stage2.py`, toàn bộ hệ thống Stage-2 v2 Gate-6 / Gate-7 Track A (`DeepLabV3PlusDualHead`, `SegFormerDualHead`, `ConvNeXtFPNDualHead`, `DINOv2AdapterDualHead`), và `LUADBCSSAlignedDataset` | `src/training/dataset_loader.py`, `src/stage2_v2/models.py`, `src/stage2_v2/gate7_clinical_bench.py`, `src/visualization/api_server.py` |

- Khi suy luận kết hợp Stage-1 (`zero_one`) với Stage-2 v2 (`imagenet`) trong `src/stage2_v2/ensemble.py` và `src/visualization/api_server.py`, tensor đầu vào được tự động phân nhánh đúng miền chuẩn hóa theo metadata `input_norm` lưu trong checkpoint (qua `load_stage1_strict`), đảm bảo nhất quán 100% giữa huấn luyện, đánh giá và triển khai lâm sàng.

---

## 4. Giao Thức Chống Rò Rỉ Bệnh Nhân & Tính Thuần Khiết Track A

### 4.1 Vấn Đề Rò Rỉ Dữ Liệu Ở Cấp Patch Trong Ảnh Y Tế

Một bệnh nhân ung thư cung cấp một hoặc nhiều tiêu bản WSI, từ đó cắt ra hàng trăm đến hàng nghìn patches `224 x 224`. Nếu xáo trộn ngẫu nhiên toàn bộ patches trước khi chia `train / val / test`, các lát cắt của cùng một khối u (cùng đột biến gen, cùng cấu trúc nền mô và cùng sắc độ thuốc nhuộm H&E) sẽ lọt vào cả tập huấn luyện và tập kiểm thử. Điều này tạo ra kết quả kiểm thử cao giả tạo nhưng thất bại trên bệnh nhân mới ngoài lâm sàng.

### 4.2 Điều Kiện Toán Học & Cơ Chế Khóa Cứng (`INVARIANT-08`)

Gọi $\mathcal{P}_{\text{train}}, \mathcal{P}_{\text{val}}, \mathcal{P}_{\text{test}}$ lần lượt là tập hợp các mã bệnh nhân (`Patient IDs`) thuộc ba phân tập:

$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$

$$\left|\mathcal{P}_{\text{train}}\right| = 106, \quad \left|\mathcal{P}_{\text{val}}\right| = 22, \quad \left|\mathcal{P}_{\text{test}}\right| = 23, \quad \left|\mathcal{P}_{\text{train}} \cup \mathcal{P}_{\text{val}} \cup \mathcal{P}_{\text{test}}\right| = 151$$

Hệ thống thực thi bảo đảm bất biến này ở 3 lớp kiểm soát độc lập:
1. **Tệp kê khai bất biến (`src/config/bcss_patient_splits.json`):** Khóa tĩnh danh sách toàn bộ `151` mã bệnh nhân TCGA-BRCA (`106 / 22 / 23`), khớp 100% với phân vùng thực trên đĩa.
2. **Kiểm định runtime tự động (`src/datasets/patient_split.py`):** Các hàm `verify_patient_leakage()` và `load_patient_splits()` kiểm tra giao tập ở mọi lần nạp dữ liệu và ném `ValueError` ngay lập tức nếu phát hiện dù chỉ 1 bệnh nhân trùng lặp. Mọi báo cáo thực nghiệm Gate-6 và Gate-7 trong `checkpoints/vps_evidence/` đều ghi nhận `patient_overlap_count = 0` và `non_finite_tensors = 0`.
3. **Kiểm thử tự động liên tục (`tests/test_patient_split.py` & `.github/workflows/ci.yml`):** Chạy bắt buộc trong mọi chu trình CI/CD và trước mỗi đợt kiểm định.

### 4.3 Nguyên Tắc Đóng Băng Tập Test & Độ Thuần Khiết Tiền Huấn Luyện Track A

- **Đóng băng tuyệt đối `23` bệnh nhân Held-Out Test (`4,986` patches):** Mọi quyết định mô hình gồm dừng sớm (`ema_only_best_selection = True`), chọn ứng viên Round-2 (`blend = 1.0` vs `0.5` tại `se374_g7_diag_r2_val`), chọn TTA theo từng nhánh, hiệu chuẩn nhiệt độ `full_val_betas`, 5-fold CV / LOPOCV tìm trọng số tổ hợp, hiệu chuẩn ngưỡng CGM (`1-DoF` / `4-DoF`) và hiệu chuẩn prototype `MP-PPC` đều thực hiện 100% trên `train` hoặc `val` (`22` bệnh nhân), **tuyệt đối không chạm vào tập `test`**.
- **Độ thuần khiết tiền huấn luyện Track A (`track_a_clean`, `0%` chồng lấp TCGA-BRCA):**
  - Toàn bộ 4 nhánh Round-1 (`arm1-arm4`) và 4 nhánh Round-2 (`arm5-arm8`) gắn cờ kiểm chứng `track = "track_a_clean"`, `tcga_pretrained = False`. Các backbone chỉ khởi tạo từ ảnh tự nhiên **ImageNet-1K** (`resnet101`, `nvidia/mit-b3`, `nvidia/mit-b5`, `convnext_base`, `convnext_large`) hoặc **LVD-142M** (`facebook/dinov2-base`, `facebook/dinov2-large`).
  - Các mô hình pathology foundation tiền huấn luyện trên dữ liệu TCGA (ví dụ `owkin/phikon-v2` trên PANCAN-XL) bị cách ly tuyệt đối sang Track B bằng `validate_track_a_arm_spec()` trong `src/stage2_v2/ensemble.py:116` (xuất lại tại `src/stage2_v2/__init__.py`) kết hợp `validate_track_name()` trong `src/stage2_v2/config.py:25`.
  - Stage-1 Champion `se374_arm_b_full` khởi tạo từ trọng số mô phổi LUAD-HistoSeg (`stage1_checkpoint_trained_on_luad.pth`: `29` GDPH + `20` TCGA-LUAD), đảm bảo **0% chồng lấp bệnh nhân TCGA-BRCA**.
