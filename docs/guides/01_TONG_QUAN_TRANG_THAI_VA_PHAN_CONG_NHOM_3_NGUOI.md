# Guide 01: Tổng Quan Dự Án, Trạng Thái, Minh Bạch Sự Cố & Phân Công Nhóm 3 Người

> **Học phần:** SE374.R11 - Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa (Khoa Kỹ Thuật Phần Mềm, UIT, ĐHQG-HCM, 2026)  
> **Giảng viên hướng dẫn:** TS. Bùi Cao Doanh  
> **Đề tài chính thức:** Đề tài 6 - Phân Vùng Ngữ Nghĩa Giám Sát Yếu Trên Tiêu Bản Mô Bệnh Học Toàn Phần WSI (`BCSS-WSSS` / `TCGA-BRCA` + kiểm định chéo cơ quan `LUAD-HistoSeg`)  
> **Thông tin hành chính:** `[CẦN XÁC NHẬN: Họ tên 3 thành viên / MSSV / GVHD]`  
> **Quay về trang chủ:** [README.md](../../README.md)

---

## 1. Hiểu Toàn Bộ Dự Án Trong 2 Phút

### 1.1 Bài Toán Y Khoa Cốt Lõi
Trong giải phẫu bệnh số (Digital Pathology), một tiêu bản sinh thiết ung thư vú nhuộm **Hematoxylin & Eosin (H&E)** quét ở độ phóng đại `40x` (`0.25 MPP`) có kích thước khổng lồ dạng gigapixel (từ `50,000 x 50,000` đến `100,000 x 100,000` điểm ảnh). Nếu yêu cầu bác sĩ giải phẫu bệnh dùng bút số khoanh vùng thủ công từng đám tế bào ở cấp từng điểm ảnh (fully-supervised) thì cần hàng trăm giờ lao động và dễ có sai khác giữa các bác sĩ.

Dự án **BCSS-WSSS (SE374.R11)** giải bài toán này bằng **Phân vùng Ngữ nghĩa Giám sát Yếu (Weakly-Supervised Semantic Segmentation - WSSS)** kết hợp **Trạm Làm Việc Giải Phẫu Bệnh Số Ngoại Tuyến**:
- **Ở giai đoạn huấn luyện (`training/`, `106` bệnh nhân, `23,422` mảnh ảnh `224 x 224` px):** Hệ thống **KHÔNG có bất kỳ mặt nạ vẽ tay cấp điểm ảnh nào**; mô hình chỉ học từ 4 bit nhãn nhị phân cấp mảnh ảnh `[TUM, STR, LYM, NEC]` (ví dụ `[1100]` nghĩa là mảnh ảnh có khối u và mô đệm).
- **Ở giai đoạn suy luận (Inference & Trạm Lâm Sàng):** Hệ thống dự đoán chính xác từng điểm ảnh cho **4 lớp mô học tiền cảnh** (`0: TUM` Biểu mô khối u, `1: STR` Mô đệm liên kết, `2: LYM` Bạch huyết bào xâm nhập, `3: NEC` Mô hoại tử) cộng thêm **lớp nền kính trắng quang học (`4: BACK`)** tách hoàn toàn độc lập không rò rỉ đáp án bác sĩ bằng `OtsuClinicalGating` + `Pure-Glass Guard >= 512 px`.
- **Giá trị chuyển dịch lâm sàng:** Tự động tính chỉ số tiên lượng **VTP** (Viable Tumor Percentage), **TSR** (Tumor-Stroma Ratio theo chuẩn Mesker 2007 / UNITED 2018 với ngưỡng nguy cơ cao `TSR <= 50%`), hỗ trợ kiểm duyệt bất định bằng **Shannon Entropy**, ghép lát cắt **WSI Multi-Tile Stitch**, và xuất **Báo cáo Lâm sàng chuẩn CAP 2 trang PDF** kèm cảnh báo Research Use Only (RUO).

### 1.2 Bảng 8 Chủ Đề Môn Học SE374.R11 & Lý Do Chọn Đề Tài 6
Theo đề bài môn học SE374.R11 (cơ cấu đánh giá học phần: `10%` điểm danh/quiz, `30%` thực hành Lab, **`60%` báo cáo chuyên đề và đồ án cuối kỳ**), môn học gồm 8 chủ đề chính đã được chuẩn hóa tiếng Việt đầy đủ dưới đây:

| STT | Tên Chủ Đề | Bản Chất Bài Toán & Phương Pháp | Bộ Dữ Liệu Tiêu Biểu | Tài Liệu Tham Khảo Chính |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Sparse-view CT reconstruction** | Bài toán nghịch đảo (inverse problem): khôi phục ảnh CT chất lượng cao từ số góc chiếu thưa để giảm liều tia X; khử nhiễu sọc (streak artifacts) bằng mạng unrolling (`CG-GLORE`, `LPD`, `FBPConvNet`). | AAPM Low-Dose CT, DeepLesion | `arXiv:2608.15246` (`CG-GLORE`), Springer MVAP 2026 |
| **2** | **Limited-angle CT reconstruction** | Khôi phục ảnh CT khi dải góc quét bị giới hạn (thiếu vùng thông tin góc quét), kết hợp bù sinogram và điều chuẩn miền ảnh. | AAPM, LoDoPaB-CT | Springer MVAP 2026, `arXiv:2608.15246` |
| **3** | **Cancer subtyping using WSI** | Phân loại phân nhóm ung thư từ ảnh mô bệnh học gigapixel; mô hình hóa túi mảnh ảnh (Multiple Instance Learning - MIL) có khái niệm dẫn đường (`CGRL`, `CLAM`, `TransMIL`). | TCGA-BRCA, TCGA-NSCLC | `arXiv:2607.12556` (`CGRL`) |
| **4** | **Survival analysis using WSI** | Dự đoán thời gian sống còn / nguy cơ tử vong (time-to-event) kết hợp học liên tục trên nhiều nhóm ung thư (`MergeSurv`, `Porpoise`, `DeepAttnMISL`). | TCGA Cohorts (`LUAD`, `BLCA`, `KIRC`) | `arXiv:2607.04747` (`MergeSurv`) |
| **5** | **Tumor segmentation using CT/fMRI (MRI)** | Phân vùng khối u (u não Glioma, u nội tạng) trên ảnh thể tích 3D bằng MONAI `Swin-UNETR`, `DynUNet`, `3D U-Net`, `nnU-Net`. | BraTS 2021/2023, LiTS | `arXiv:1804.10484`, `DBC-Lab/Brain_MRI_Enhancement` |
| **6** | **Weakly-supervised segmentation for WSI (Đề tài nhóm)** | Phân vùng tổn thương giải phẫu bệnh chỉ với nhãn cấp mảnh ảnh hoặc ảnh độ phóng đại thấp (`LowMagWSS`, `WSSS-Tissue`, `HisynSeg`, `ARML`). | **BCSS-WSSS**, **LUAD-HistoSeg**, Camelyon16 | `arXiv:2607.10783` (`LowMagWSS`), `DOI:10.1016/j.media.2022.102487` (`WSSS-Tissue`) |
| **7** | **Collateral imaging in acute ischemic stroke (MRI)** | Đánh giá tuần hoàn bàng hệ trong đột quỵ thiếu máu não cấp; lượng hóa vùng tranh tối tranh sáng (penumbra) và lõi nhồi máu (infarct core). | ISLES 2022, Perfusion MRI | Springer *Int J CARS* (`10.1007/s11548-024-03229-5`) |
| **8** | **Tự do đề xuất (Free Medical AI Proposals)** | Phát hiện phổ tự kỷ (ASD) dựa trên fMRI não (GNN/CNN trên ABIDE I) hoặc trợ lý hỏi đáp hình ảnh y khoa (VQA). | ABIDE I, VQA-RAD, PathVQA | Đề xuất mở của nhóm sinh viên |

**Vì sao nhóm chốt Đề tài 6 theo chiến lược 2 tầng (`BCSS-WSSS` + mở rộng `WSI`):**
- Bộ dữ liệu `BCSS-WSSS` (`31,826` patches `224 x 224` px, `~2.5-2.9 GB`) giúp tải nhanh trên Colab/PC sinh viên, tách sạch `151` bệnh nhân chống rò rỉ `0%`, có mã nguồn tham chiếu chuẩn mực từ `WSSS-Tissue` (MedIA 2022) và liên kết trực tiếp với hướng nghiên cứu `LowMagWSS` (MAPR 2026) của giảng viên hướng dẫn.
- Nhóm lấy `BCSS-WSSS` 40x làm lõi nghiên cứu SOTA (**Gate-7 Track A 4-Arm**, đạt `72.79%` no-CGM / `72.80%` `nec_1dof` / `72.90%` Simplex mIoU), đồng thời xây dựng hoàn chỉnh Trạm Lâm Sàng FastAPI hỗ trợ **WSI Multi-Tile Stitch**, **Hann-50% Whole-ROI**, **LUAD-HistoSeg Zero-Shot chéo cơ quan** và **CAP PDF 2 trang**.

---

## 2. Bảng Trạng Thái Thực Tế Của Toàn Dự Án

| Hạng Mục | Trạng Thái | Tóm Tắt Minh Chứng Trên Đĩa | Tài Liệu / Đường Dẫn Chi Tiết |
| :--- | :---: | :--- | :--- |
| **1. Nghiên cứu lõi Gate-7 Track A (Phase 1B)** | **[ĐÃ XONG]** | Hoàn tất `6/6` job GPU hậu bản vá (`02c_v2`, `04b_v2`, `02d_v2`, `H1`, `H2`, `H3`) + phân tích không GPU; Primary 4-Arm đạt **`72.79%` no-CGM / `72.80%` `nec_1dof`** (`84.04%-84.06%` mDice), Secondary Simplex **`72.90%`** (`84.11%` mDice); `228/228` SHA256 OK (`BAD = 0`). | [Guide 02](02_PHAN_1_NGHIEN_CUU_DATASET_MODEL_VA_19_BAI_BAO.md), [STAGE2_FINAL_EXPERIMENT_SUMMARY.md](../research/STAGE2_FINAL_EXPERIMENT_SUMMARY.md) |
| **2. Kiểm chứng trực quan `02d_v2` & Bộ ảnh minh họa** | **[ĐÃ XONG]** | `02d_v2` đạt `6/6` assertion PASS trên `4,986` patch TEST (`0%` rò rỉ GT vào `BACK=4`); xuất đủ `25` ảnh Showcase (`N1-N5`, `P1-P5` + CAP PDF 2 trang trong nhánh lưu trữ cục bộ) và chọn lọc `6` ảnh nhẹ đại diện trong `docs/assets/readme/`. | [Guide 04](04_HUONG_DAN_DOC_ANH_TRUC_QUAN_N1_N5_VA_P1_P5.md) |
| **3. Trạm Lâm Sàng FastAPI + HTML5 Canvas Offline + Docker** | **[ĐÃ XONG]** | Hệ màu chuẩn quang học **Trắng Lâm Sàng + Xanh Y Khoa** (`#FFFFFF`, `#F4FAFA`, `#0E7C86`, `#00957E`, `#2A9D8F`), tách rõ **3 chế độ** (Chẩn đoán Lâm sàng không GT / Kiểm chứng Nghiên cứu GT kèm Đối chiếu 4 Khung / Ghép lát cắt WSI), `15` HTTP routes, `18` chế độ AI, Split-Curtain, Magnifier Loupe, Pixel HUD, Compare Modal, CAP PDF 2 trang, cảnh báo RUO thường trực, fail-closed `503`, `Dockerfile` + `docker-compose.yml`. | [Guide 03](03_PHAN_2_PHAN_MEM_KIEN_TRUC_GRAPHIFY_AST_VA_VAN_HANH.md) |
| **4. Cổng kiểm định chất lượng (CI, Tests, Linter, Rubric 40+40)** | **[ĐÃ XONG]** | **`501` bài kiểm thử tự động phần mềm y khoa** (`30` bộ kiểm thử trong `tests/`), `ruff check src tests` (`0` lỗi), `DeliverableAuditor` (`0 CRITICAL, 0 WARNING`), CI tự động quét chống rò rỉ tài khoản/mockup, đạt tiêu chuẩn nghiệm thu **40/40 Nghiên cứu + 40/40 Phần mềm (`0` cờ đỏ)** qua kiểm định chéo nhiều luồng độc lập. | [Mục 5 tài liệu này](#5-lich-su-kiem-toan-minh-bach-su-co--gioi-han-khoa-hoc-mo), [SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md](../../product/reports/SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md) |
| **5. Tối ưu kho GitHub & Bộ 4 Cẩm Nang Hợp Nhất** | **[ĐANG HOÀN TẤT]** | Tinh gọn còn `16` tệp `.md` sạch tiếng Việt có dấu chuẩn, tách dữ liệu/công cụ nội bộ và tài liệu rời rạc sang nhánh lưu trữ cục bộ `local-archive/full-workspace`, giữ `4` ấn phẩm chính thức trong `product/` và `6` ảnh đại diện nhẹ trong `docs/assets/readme/`. | [README.md](../../README.md), [Guide 03](03_PHAN_2_PHAN_MEM_KIEN_TRUC_GRAPHIFY_AST_VA_VAN_HANH.md) |
| **6. Chọn ảnh minh họa đưa vào Slide / Phụ lục cuối kỳ** | **[CHỜ NHÓM CHỐT]** | Cả 3 thành viên xem `6` ảnh đại diện nhẹ và các hình `fig1-fig9` tại [Guide 04](04_HUONG_DAN_DOC_ANH_TRUC_QUAN_N1_N5_VA_P1_P5.md) (hoặc mở trọn bộ `25` ảnh từ nhánh lưu trữ cục bộ `local-archive/full-workspace`) để chốt ảnh đưa vào slide bảo vệ. | [Guide 04](04_HUONG_DAN_DOC_ANH_TRUC_QUAN_N1_N5_VA_P1_P5.md) |
| **7. Điền thông tin hành chính 3 thành viên & GVHD** | **[CHỜ NHÓM CHỐT]** | Hiện giữ đúng placeholder `[CẦN XÁC NHẬN: Họ tên sinh viên / MSSV / GVHD]` tại `README.md`, báo cáo cuối kỳ, biên bản kiểm định, proposal và slide bảo vệ để tuyệt đối không tự bịa thông tin cá nhân. | [README.md](../../README.md) |
| **8. Kiểm tra lần cuối & `git push` lên GitHub** | **[CHỜ NHÓM CHỐT]** | Chỉ đẩy nhánh GitHub sạch sau khi nhóm duyệt xong bộ tài liệu hợp nhất, điền thông tin hành chính và xác nhận `pytest`, `ruff`, `DeliverableAuditor` đều xanh. | [Mục 4 tài liệu này](#4-ban-do-tai-nguyen-co-san-khi-git-clone-vs-can-tai--luu-cuc-bo) |

---

## 3. Phân Công Lộ Trình Đọc & Trách Nhiệm Cho Nhóm 3 Người

Để 3 thành viên không phải đọc tràn lan toàn bộ kho mã nguồn, mỗi người nắm **1 trụ cột chính** dựa trên **4 Cẩm nang hợp nhất (`Guide 01-04`)** và phối hợp chặt ở buổi ghép slide / demo bảo vệ:

### 3.1 Thành Viên 1 - Phụ Trách Phần 1: Nghiên Cứu AI, Dữ Liệu & Thực Nghiệm
- **Đọc theo thứ tự (45-60 phút):**
  1. [README.md](../../README.md) (tóm tắt tổng thể & bảng số liệu chốt).
  2. [Guide 02: Phần 1 - Nghiên cứu, Dataset, Mô hình & 19 Bài báo](02_PHAN_1_NGHIEN_CUU_DATASET_MODEL_VA_19_BAI_BAO.md).
  3. [STAGE2_FINAL_EXPERIMENT_SUMMARY.md](../research/STAGE2_FINAL_EXPERIMENT_SUMMARY.md) & [DATASET_SPECIFICATION_AND_PATIENT_LEAKAGE_PROTOCOL.md](../research/DATASET_SPECIFICATION_AND_PATIENT_LEAKAGE_PROTOCOL.md).
  4. [Guide 04: Giải mã ảnh nghiên cứu `N1-N5` & `fig1-fig9`](04_HUONG_DAN_DOC_ANH_TRUC_QUAN_N1_N5_VA_P1_P5.md).
- **Trách nhiệm khi bảo vệ đồ án:**
  - Trình bày bài toán WSSS từ nhãn yếu `y_cls`, cơ chế chống rò rỉ `151` bệnh nhân (`106/22/23`), độ thuần khiết **Track A (`0%` TCGA-BRCA pretraining)** và sự khác biệt giữa miền chuẩn hóa `[0, 1]` vs `ImageNet`.
  - Giải thích bước tiến từ Author Baseline (`69.48%`) -> Run 3 `M6` (`70.53%`) -> Gate-6 Round-1 (`72.47%`) -> Gate-7 Track A Phase 1B (`72.79%` no-CGM / `72.80%` `nec_1dof` / `72.90%` Simplex), bóc tách nhân quả `H1/H2/H3`, đệm đối xứng `_pad_to_multiple_of_32` cho `arm7`, kiểm định Holm-Bonferroni (`4/4` lớp `p_holm < 0.05`), và thử nghiệm chéo cơ quan `LUAD 2x zoom` (`51.47%`).
  - Trả lời câu hỏi phản biện về 19 bài báo SOTA (bao gồm lý do minh bạch không so trực tiếp `mDice` với `SIA-WSSS` và giới hạn sống còn `n = 23`, `2` biến cố tử vong từ NCI GDC Release 46.0).

### 3.2 Thành Viên 2 - Phụ Trách Phần 2: Kiến Trúc Phần Mềm, Trạm Lâm Sàng & Docker
- **Đọc theo thứ tự (45-60 phút):**
  1. [README.md](../../README.md) (mục Phần 2 Phần mềm & Quickstart).
  2. [Guide 03: Phần 2 - Phần mềm, Bản đồ AST, Trạm Lâm Sàng & Docker](03_PHAN_2_PHAN_MEM_KIEN_TRUC_GRAPHIFY_AST_VA_VAN_HANH.md).
  3. [Guide 04: Giải mã ảnh phần mềm sống `P1-P5`](04_HUONG_DAN_DOC_ANH_TRUC_QUAN_N1_N5_VA_P1_P5.md).
- **Trách nhiệm khi bảo vệ đồ án:**
  - Khởi chạy và demo trực tiếp **Trạm Lâm Sàng FastAPI + HTML5 Canvas 100% Offline** theo hệ màu **Trắng Lâm Sàng + Xanh Y Khoa** (`http://localhost:8000/viewer` hoặc `docker compose up --build -d`): trình diễn 3 chế độ (**Chẩn đoán Lâm sàng không GT**, **Kiểm chứng Nghiên cứu 4 Khung GT**, **Ghép lát cắt WSI**), `18` chế độ AI, thanh trượt `Split-Curtain (S)`, kính lúp `Magnifier Loupe (L)`, `Pixel HUD`, `Compare Modal`, và xuất **CAP PDF 2 trang**.
  - Trình bày kiến trúc Clean Architecture `11` gói trong `src/`, luồng dữ liệu qua đồ thị AST (`scripts/build_ast_code_graph.py`), cơ chế fail-closed HTTP `503` khi thiếu trọng số, và triển khai container Docker không đặc quyền.

### 3.3 Thành Viên 3 - Phụ Trách Kiểm Định Chất Lượng, Báo Cáo, Slide & Minh Bạch Sự Cố
- **Đọc theo thứ tự (45-60 phút):**
  1. [README.md](../../README.md) & [Mục 5 tài liệu này (Minh bạch sự cố & QA)](#5-lich-su-kiem-toan-minh-bach-su-co--gioi-han-khoa-hoc-mo).
  2. [Guide 04: Phân biệt ảnh B&W học thuật `fig1-fig8` vs ngoại lệ H&E `fig9`, `N1-N5`, `P1-P5`](04_HUONG_DAN_DOC_ANH_TRUC_QUAN_N1_N5_VA_P1_P5.md).
  3. [SE374_MINI_PROPOSAL_WSSS_WSI.md](../../product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md), [SE374_FINAL_TECHNICAL_REPORT_WSSS.md](../../product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md), [SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md](../../product/reports/SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md), và [SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.md](../../product/slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.md).
- **Trách nhiệm khi bảo vệ đồ án:**
  - Tổng hợp ý kiến chọn ảnh minh họa `N1-N5` / `P1-P5`, điền thông tin hành chính chính xác cho 3 thành viên và GVHD, biên dịch lại `.docx` (`pandoc --reference-doc=docs/templates/SE374_Reference_Template.docx --syntax-highlighting=none`) và `.pptx` (`marp --pptx --no-stdin --allow-local-files`).
  - Vận hành các cổng kiểm định tự động trước khi nộp bài: `pytest`, `ruff check src tests`, `DeliverableAuditor` (`0 CRITICAL, 0 WARNING`), kiểm tra `SHA256SUMS.txt` (`228/228`), và giải trình minh bạch về quy trình kiểm định chất lượng cũng như 2 sự cố mất dữ liệu cục bộ (`2026-11-05` MedSAM & `2026-11-23` `Lab1.ipynb` / WSL2).

---

<a id="4-ban-do-tai-nguyen-co-san-khi-git-clone-vs-can-tai--luu-cuc-bo"></a>
## 4. Bản Đồ Tài Nguyên: Có Sẵn Khi `git clone` vs Lưu Nhánh Cục Bộ vs Cần Tải Ngoài

Kho lưu trữ được tổ chức theo mô hình 2 tầng sạch sẽ: **nhánh GitHub sạch** chỉ giữ mã nguồn cốt lõi, bộ kiểm thử, tài liệu hợp nhất (`16` tệp `.md`), ấn phẩm chính thức (`4` tệp `.docx`/`.pptx` + `figures/`) và mẫu ảnh nhẹ; trong khi **nhánh Git cục bộ riêng (`local-archive/full-workspace`)** giữ nguyên toàn bộ tài liệu cũ, bản lưu trung gian và tiện ích phụ trợ cục bộ (không đẩy lên GitHub).

| Tài Nguyên / Thư Mục | Có Sẵn Khi `git clone`? | Trạng Thái Lưu Trữ Thực Tế | Cách Thành Viên Lấy / Dùng |
| :--- | :---: | :--- | :--- |
| `src/`, `tests/`, `scripts/` (`build_ast_code_graph.py`, `build_drive_package.py`, ...), `notebooks/` (`01-04b`, `ThinDriveRunner_v2`, `history_runs/`) | **CÓ (100%)** | Nhánh GitHub chính | Tự động có ngay sau `git clone` |
| `README.md`, `docs/guides/01-04`, `docs/research/*.md`, `docs/assets/readme/` (`6` ảnh showcase nhẹ + minh họa) | **CÓ (100%)** | Nhánh GitHub chính | Tự động có ngay sau `git clone` |
| `product/` (`proposal/SE374_MINI_PROPOSAL_WSSS_WSI.*`, `reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.*`, `reports/SE374_MULTI_AGENT_AUDIT_*`, `reports/figures/fig1-fig9`, `slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.*`) | **CÓ (100%)** | Nhánh GitHub chính | Tự động có ngay sau `git clone` |
| `src/config/bcss_patient_splits.json` & `tcga_bcss_151_gdc_clinical.json` | **CÓ (100%)** | Nhánh GitHub chính | Tự động có ngay sau `git clone` |
| `experiments/benchmarks/COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md` & `experiments/colab_run_1-3/` (`.md`, `.json`, `.csv`) | **CÓ (100%)** | Nhánh GitHub chính | Tự động có ngay sau `git clone` |
| `data/sample_dataset/` (`12` tệp mẫu nhỏ chạy unit test) | **CÓ (100%)** | Nhánh GitHub chính | Tự động có ngay sau `git clone` |
| `docs/templates/SE374_Reference_Template.docx` | **CÓ (100%)** | Nhánh GitHub chính | Dùng trực tiếp cho `pandoc --reference-doc` |
| Bộ `25` ảnh Showcase gốc (`experiments/visual_option_showcase/`), `results_dryrun_v1/`, `docs/lectures/`, `docs/course/`, các `.md`/template cũ đã hợp nhất | **LƯU CỤC BỘ** | Lưu đầy đủ trong nhánh Git cục bộ `local-archive/full-workspace` (không push GitHub) | Xem `git show local-archive/full-workspace:<path>` khi cần đối chiếu bản gốc |
| `data/BCSS-WSSS/` (`31,826` ảnh, `~2.9 GB`) | **KHÔNG (gitignored)** | Có đủ trên máy chính | Tải từ [Hugging Face BCSS-WSSS.zip](https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip) hoặc [GDrive Mirror](https://drive.google.com/file/d/1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn/view) giải nén vào `data/BCSS-WSSS/` |
| `data/LUAD-HistoSeg/` (`17,285` ảnh phổi `10x`) | **KHÔNG (gitignored)** | **Chưa lưu trên máy Windows cục bộ** (đã chạy trên Colab/Cloud GPU ở `04b_v2`) | Tải từ [Hugging Face LUAD-HistoSeg.zip](https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/LUAD-HistoSeg.zip) nếu muốn chạy lại benchmark phổi cục bộ |
| `checkpoints/` (`stage1/2` tác giả + `vps_evidence/` `arm_b_full`, `arm5-arm8` `.pth`, `228` tệp SHA256) | **KHÔNG (gitignored)** | Có đủ bản chốt trên máy chính (`~4.63 GB` / `4.32 GiB` tổng, `vps_evidence/` `3.74 GB`) | Trọng số tác giả tải từ Hugging Face (xem [README.md](../../README.md)); trọng số Gate-7 `arm5-arm8` chia sẻ nội bộ trong nhóm kèm `SHA256SUMS.txt` |
| `docs/papers/` (`19` tệp `.md` tóm lược) & `repos/` (`WSSS-Tissue`, `LowMagWSS`) | **KHÔNG (gitignored)** | Có đủ trên máy chính | **Toàn bộ 19 bài báo đã tóm lược đầy đủ trong [Guide 02](02_PHAN_1_NGHIEN_CUU_DATASET_MODEL_VA_19_BAI_BAO.md)**; mã nguồn gốc clone từ link GitHub trong Guide 02 |
| `graphify-out/` (`graph.json`, `GRAPH_REPORT.md`) | **KHÔNG (gitignored)** | Dựng tức thì bằng script | Chạy `.\.venv\Scripts\python.exe scripts/build_ast_code_graph.py` |
| `notebooks/Lab1.ipynb` | **MẤT VĨNH VIỄN** | Mất ngày `2026-11-23` (lỗi xóa cục bộ ổ `C:\`) | Bài thực hành ngoại khóa ban đầu, **không thuộc** pipeline sản phẩm/nghiên cứu (xem Mục 5.2) |

---

<a id="5-lich-su-kiem-toan-minh-bach-su-co--gioi-han-khoa-hoc-mo"></a>
## 5. Lịch Sử Kiểm Toán, Minh Bạch Sự Cố & Giới Hạn Khoa Học Mở

> **Tinh thần cốt lõi:** Trung thực tuyệt đối (**Radical Scientific Honesty**) - thiếu báo thiếu, mất báo mất, lỗi báo lỗi, tuyệt đối không che giấu sự cố hay tô hồng số liệu. Biên bản kiểm định chính thức đi kèm: [SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md](../../product/reports/SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md) ([Word DOCX](../../product/reports/SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.docx)).

### 5.1 Quy Trình Kiểm Định Chất Lượng Phần Mềm & Thang Điểm Nghiệm Thu `40+40`
Để bảo đảm chất lượng đồng thời ở cả chiều sâu nghiên cứu lẫn độ tin cậy kỹ nghệ phần mềm, đồ án áp dụng quy trình kiểm định chuẩn Kỹ thuật Phần mềm gồm **hai tầng**:
1. **Kiểm tra chéo nhiều luồng độc lập 2 vòng:** Các luồng kiểm toán (Kiến trúc & AST, Dữ liệu & Liêm chính, Giao diện & Kiểm thử tự động) đánh giá độc lập theo nguyên tắc đồng thuận ở Vòng 1; nếu có ý kiến bất đồng (`2 vs 1`), toàn bộ các luồng bắt buộc tái đối soát trực tiếp trên bằng chứng mã nguồn (`file:line`), log thực nghiệm thật, kết quả test và mã băm SHA256 trước khi chốt quyết định ở Vòng 2 kèm ghi nhận minh bạch các điểm phản biện.
2. **Thang điểm nghiệm thu kép `40 + 40` (`0` cờ đỏ):**
   - **Khối Nghiên cứu (`40` điểm):** Anti-Leakage bệnh nhân (`106/22/23`), độ thuần khiết Track A (`0%` TCGA pretraining), thực nghiệm đa hạt giống & kiểm định thống kê Holm-Bonferroni, bóc tách nhân quả (`H1/H2/H3`), kiểm chứng trực quan không rò rỉ GT (`02d_v2`), và phân tích lâm sàng / sống còn thật (`NCI GDC 46.0` + `LUAD 2x`).
   - **Khối Phần mềm (`40` điểm):** Kiến trúc mô-đun sạch (`src/`), Trạm Lâm Sàng FastAPI + HTML5 Canvas 100% offline, đóng gói Docker an toàn (`appuser`, `/health`), `501` ca `pytest`, `0` lỗi `ruff`, `0 CRITICAL / 0 WARNING` trên `DeliverableAuditor`, và toàn vẹn mật mã `228/228 SHA256SUMS.txt`.

### 5.2 Sổ Nhật Ký 2 Sự Cố Mất Dữ Liệu Cục Bộ (Công Bố Minh Bạch 100%)
1. **Sự cố ngày `2026-11-05` - Ghi đè mất trọng số `MedSAM` (`2 x 86 MB`):**
   - **Tài sản bị mất vĩnh viễn:** Cặp tệp trọng số cục bộ (gitignored) `checkpoints/medsam_wsss_adapter_best.pth` (`86 MB`) và `checkpoints/medsam_wsss_adapter_last.pth` (`86 MB`), bị ghi đè lúc `03:40` ngày `2026-11-05` trong đợt chạy kiểm toán song song. Do không theo dõi trong git và không có bản sao lưu ngoài, cặp trọng số MedSAM cũ này **không thể khôi phục** (muốn dùng lại cần huấn luyện lại bằng `notebooks/03_Train_Proposed_MedSAM_*.ipynb`).
   - **Nguyên nhân gốc rễ (`D5` & `D16`):** Bài test cũ chưa cô lập trong `tmp_path` khi gọi script có `shutil.rmtree()`, và nhánh fallback cũ của `src/training/train_proposed_sam.py` cho phép rơi về tập giả lập rồi ghi đè `.pth` vào `checkpoints/` thật.
   - **Khắc phục & ảnh hưởng:** Đã cô lập 100% test ghi tệp vào `tmp_path`, chặn tuyệt đối fallback giả lập ghi đè checkpoint thật khi `dry_run=False`. **Không ảnh hưởng** đến toàn bộ tuyến chính Gate-7 Track A (`se374_arm_b_full`, `arm5-arm8`) hay mốc Baseline Author (`228/228` SHA256 nguyên vẹn).
2. **Sự cố ổ `C:\` ngày `2026-11-23` - Mất `Lab1.ipynb` & đĩa ảo WSL2 Ubuntu (`ext4.vhdx`):**
   - **Tài sản bị mất vĩnh viễn:** Tệp bài tập thực hành ngoại khóa cục bộ `notebooks/Lab1.ipynb` (không theo dõi trên git; không thuộc chuỗi pipeline nghiên cứu hay sản phẩm phần mềm cuối kỳ) và đĩa ảo WSL2 Ubuntu (`ext4.vhdx`) do lỗi thoát dấu ngoặc kép giữa PowerShell và `cmd` khi dọn dẹp cục bộ trên ổ `C:\`.
   - **Khôi phục & kiểm chứng:** **100% mã nguồn, tài liệu, notebook sản phẩm, báo cáo và slide** khôi phục nguyên vẹn từ lịch sử git; **100% minh chứng Phase 1B (`228/228` tệp trong `checkpoints/vps_evidence/SHA256SUMS.txt`)** được xác minh bit-exact (`BAD = 0`, `MISSING = 0`, `UNLISTED = 0`).

### 5.3 Biên Niên Sử 6 Đợt Kiểm Toán & Khắc Phục Lỗi (Tháng 10 - Tháng 11/2026)

| Mốc Thời Gian | Đợt Kiểm Toán / Sự Kiện | Phát Hiện Chính (Cờ Đỏ / Lỗi Lõi) | Kết Quả Khắc Phục & Bài Học Rút Ra |
| :--- | :--- | :--- | :--- |
| **`2026-10-24`** | Kiểm toán kết quả `Dry-Run v2` | Phát hiện bộ `v2` chỉ chạy `SANITY_DRY_RUN=True` (`46` patch), `Adaptive Router` bị kẹp `max(0.0, ...)`, và `Exp 4 LUAD` nạp nhầm checkpoint LUAD rồi gắn nhãn "zero-shot" (`78.76%`). | **Chặn tuyệt đối không đưa số dry-run vào báo cáo**; sửa nhãn đánh giá; chuyển sang đánh giá thật `4,986` patch (`Run 3 M6 = 70.53%`) và thực hiện lại Zero-Shot LUAD thật 100% ở `04b_v2` (`36.09%` tại `10x` / `51.47%` tại `2x` zoom). |
| **`2026-11-02`** | Kiểm toán kỹ thuật 6 luồng (`DEF-01` đến `DEF-18`) | Phát hiện `NameError: cm4` trong `wsi_lowmag_highmag_pipeline.py`, lỗ hổng `torch.load` (CWE-502), XML parser chưa dùng `defusedxml`, `77` lỗi `ruff`, lỗi batch `otsu_filter`, đảo chiều nguy cơ TSR, và OOM `Spatial Affinity` ở `224x224`. | Sửa triệt để `16/16` lỗi thật, thêm `safe_torch_load`, `defusedxml`, `Hypothesis` fuzz testing cho `extract_patient_id`, chuẩn hóa `TSR <= 50%` là Stroma-Rich High Risk theo Mesker/UNITED. |
| **`2026-11-05`** | Kiểm toán đối kháng 8 luồng (`D1-D16`) | Phát hiện CI `.github/workflows/ci.yml` sai nhánh (`main` thay vì `master`), test không hermetic phá đĩa (gây mất cặp `MedSAM 2x86 MB`), báo cáo cũ chỉ chạy `2` patch nhưng ghi `23` BN, lệnh gắn Google Drive qua OAuth còn sót ở notebook, và `glob("*.png")` nông. | Viết lại CI chạy trên `master`, cô lập test trong `tmp_path`, loại bỏ báo cáo cũ khỏi nhánh sạch, chuẩn hóa `resolve_split_path` + `rglob("*.png")` và loại bỏ hoàn toàn bước gắn Google Drive khỏi quy trình cloud thuần đĩa cục bộ `/content/`. |
| **`2026-11-15`** | Kiểm toán Zero-Trust (`Nhóm A-D`) | Phát hiện URL sai trong notebook Colab, xung đột hai lớp `DeepLabV3Plus`, lỗi nhầm chuẩn hóa `ImageNet` vs `[0, 1]` cho checkpoint tác giả, hardcode `4` kênh khi `num_classes=5`, `api_server.py` hardcode `"loaded": True`, và lỗi nhãn ma khi lọc bilateral trên nhãn rời rạc. | Sửa toàn bộ URL kho git, tách bạch 2 hợp đồng chuẩn hóa `[0, 1]` vs `ImageNet`, lọc song phương trên xác suất liên tục `(C, H, W)` xóa 100% nhãn ma, chuẩn hóa `pytest.ini` loại trừ `repos/`, và buộc `api_server.py` trả `503` khi thiếu weights. |
| **`2026-11-23 - 2026-11-24`** | Chiến dịch Gate-7 Phase 1A/1B | Xóa triệt để cờ đỏ sống còn (thay bằng **NCI GDC REST API Release 46.0** thật cho `151/151` BN), đo Pareto đồng chuẩn `torch.bfloat16` trên L4, vá cắt biên stride-2 `_pad_to_multiple_of_32` cho `arm7` ConvNeXt-L, bổ sung Hann-50% Whole-ROI, LUAD `2x` zoom và bóc tách `H1/H2/H3`. | Hoàn tất `6/6` job GPU thật, khóa **`228/228` tệp `SHA256SUMS.txt` (`BAD = 0`)**, `0` lỗi `ruff`, đạt chuẩn nghiệm thu **`40/40` Nghiên cứu + `40/40` Phần mềm (`0` cờ đỏ)**. |
| **`2026-11-25 - 2026-11-26`** | Chuẩn hóa Màu Y Khoa, Vệ Sinh Kho Git & Tinh Gọn Tài Liệu | Nâng cấp Trạm Lâm Sàng sang hệ màu **Trắng Lâm Sàng + Xanh Y Khoa (`#FFFFFF`, `#F4FAFA`, `#0E7C86`, `#00957E`, `#2A9D8F`)**, tách 3 chế độ (Lâm sàng / Kiểm chứng 4 Khung GT / Ghép WSI) và banner RUO; rà soát tách công cụ vận hành cá nhân khỏi kho GitHub học thuật; tinh gọn `33` tệp `.md` rời rạc thành bộ tài liệu hợp nhất chuẩn. | `501` bài kiểm thử y khoa PASS; `0` rò rỉ tài khoản/công cụ nội bộ; `0` link hỏng; `0 CRITICAL / 0 WARNING` trên `DeliverableAuditor`; giữ nguyên toàn bộ tài sản cũ tại nhánh cục bộ `local-archive/full-workspace`. |

### 5.4 Năm Giới Hạn Khoa Học Mở (Báo Cáo Thẳng Thắn Trước Hội Đồng)
1. **Cỡ mẫu sống còn Held-Out Test nhỏ (`n = 23` bệnh nhân, chỉ có `2` biến cố tử vong):** Dù dữ liệu sống còn đã đối soát sống 100% từ **NCI GDC Release 46.0** (`151/151` BN), riêng tập Test `23` bệnh nhân chỉ có `2` ca tử vong nên kiểm định Kaplan-Meier chưa đủ lực thống kê cho `p < 0.05` (`GT p = 0.1869`, `ens4_cgm p = 0.1474`).
2. **Đánh giá 3 hạt giống tập trung ở `arm5` và tổ hợp 4-Arm (`arm6/7/8` đơn seed):** Độ ổn định `3` hạt giống (`42 / 123 / 456`) đã được xác minh trên `arm5` (`72.44 +/- 0.13%` no-CGM / `72.53 +/- 0.13%` +CGM trên `02c_v2`) và bên trong tổ hợp 4-Arm (`72.74 +/- 0.05%` / `72.76 +/- 0.05%`); `arm6`, `arm7`, `arm8` huấn luyện đơn seed `42` do ngân sách thời gian GPU.
3. **Khoảng cách `-0.50%` đến `-0.61%` mIoU so với `SIA-WSSS` (MICCAI 2025):** Gate-7 Track A vượt toàn bộ SOTA 4-class pooled chuẩn 2022-2026 (`QFKF`, `ConStruct`, `HisynSeg`, `PLIMSeg`, `DualProtoSeg`, `CSS`, `C2RM-Seg`), nhưng thấp hơn `SIA-WSSS` (`73.40%` mIoU, trung bình 5 lần chạy, có thêm lớp `OTR`) và không so sánh `mDice` do khác công thức định nghĩa (`~5.1 pp`).
4. **Dữ liệu `LUAD-HistoSeg` và một số trọng số trung gian không nằm trên đĩa cục bộ:** Tập `data/LUAD-HistoSeg/` chạy trực tiếp trên GPU Cloud (`04b_v2`), các trọng số Round-1 (`arm1-arm4`), seed phụ `s123/s456` và full `.zip` của `H1/H2/H3` lưu log/JSON nhẹ cục bộ; 4 nhánh chốt `arm5-arm8` và `arm_b_full` có đầy đủ cục bộ kèm SHA256.
5. **Tác vụ cũ lưu ở nhánh lưu trữ cục bộ:** Các báo cáo smoke cũ (`EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md` `2` patch, `clinical_wsi_report.md` demo WSI `100` patch, `results_dryrun_v1/`) được tách khỏi nhánh GitHub sạch vào `local-archive/full-workspace` để không gây nhiễu số liệu chính thức.

---

## 6. Hướng Dẫn Thiết Lập Môi Trường Cục Bộ Nhanh (Windows & Linux)

> **Lưu ý môi trường Windows:** Luôn tạo và gọi tường minh môi trường ảo cục bộ **`.venv\Scripts\python.exe`** (Python `3.10-3.12`) để tránh xung đột với bản Python cài ở cấp hệ thống trên Windows.

```powershell
# 1. Clone repo và tạo môi trường ảo Python 3.10 - 3.12
git clone https://github.com/khoinguyen59/medical-wsss.git
cd medical-wsss
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt

# 2. Kiểm tra nhanh độ sạch của kho (tests, linter, kiểm định tài liệu)
.\.venv\Scripts\python.exe -m pytest tests/test_governance_invariants.py tests/test_patient_split.py -q
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\python.exe -m src.utils.deliverable_auditor README.md "docs/guides/*.md" "docs/research/*.md"

# 3. Sinh bản đồ mã nguồn AST (tạo graphify-out/graph.json và GRAPH_REPORT.md)
.\.venv\Scripts\python.exe scripts/build_ast_code_graph.py

# 4. Khởi chạy Trạm Lâm Sàng Ngoại Tuyến tại http://localhost:8000/viewer
.\.venv\Scripts\python.exe -m uvicorn src.visualization.api_server:app --host 0.0.0.0 --port 8000
```

---

## 7. Mạng Lưới Liên Kết Sang Các Cẩm Nang Tiếp Theo

- **Đọc tiếp Phần 1 (Nghiên cứu):** [Guide 02 - Dataset, Mô Hình, Các Option Nghiên Cứu & 19 Bài Báo](02_PHAN_1_NGHIEN_CUU_DATASET_MODEL_VA_19_BAI_BAO.md)
- **Đọc tiếp Phần 2 (Phần mềm):** [Guide 03 - Kiến Trúc Phần Mềm, Bản Đồ AST, API, Colab & Docker](03_PHAN_2_PHAN_MEM_KIEN_TRUC_GRAPHIFY_AST_VA_VAN_HANH.md)
- **Đọc tiếp Giải mã hình ảnh:** [Guide 04 - Hướng Dẫn Đọc Hiểu Toàn Bộ Các Loại Ảnh `N1-N5`, `P1-P5` & `fig1-fig9`](04_HUONG_DAN_DOC_ANH_TRUC_QUAN_N1_N5_VA_P1_P5.md)
