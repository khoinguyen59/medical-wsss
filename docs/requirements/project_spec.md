# Project Specification & Dynamic Knowledge Dossier

> **Specification Standard:** `agents.md` v5.0 - Dynamic Knowledge Decoupling Protocol  
> **Target File:** `docs/requirements/project_spec.md`  
> **Course Profile:** SE374.R11 - Phát triển phần mềm xử lý ảnh y khoa, Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM (2026).  
> **Faculty:** Khoa Kỹ thuật Phần mềm (Software Engineering).  
> **Primary Topic Focus:** Topic 6 - Weakly-supervised segmentation for WSI (BCSS & Camelyon16).  

---

## 1. Course Profile & Administrative Information

- **Mã lớp học phần:** SE374.R11
- **Tên môn học:** Phát triển phần mềm xử lý ảnh y khoa
- **Đơn vị đào tạo:** Khoa Kỹ thuật Phần mềm, Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM
- **Năm học:** 2026
- **Tài liệu nguồn môn học:**
  - Đề bài tổng quan: `docs/course/topic.md`
  - Danh sách nhóm & đăng ký đề tài: `docs/course/SE374.R11 - Đồ án cuối kỳ.xlsx`

---

## 2. Topic Matrix & Scope (8 Chủ Đề Môn Học SE374.R11)

| STT Topic | Tên Chủ Đề (Topic Name) | Bản Chất Bài Toán & Phương Pháp Cốt Lõi | Bộ Dữ Liệu Tiêu Biểu | Tài Liệu Tham Khảo Chính |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Sparse-view CT reconstruction** *(Trọng tâm Buổi 2)* | Bài toán nghịch đảo (inverse problem): khôi phục CT chất lượng cao từ số góc chiếu thưa để giảm liều tia X; khử streak artifacts. Kế thừa mạng unrolling (CG-GLORE, LPD, FBPConvNet). | AAPM Low-Dose CT, DeepLesion | arXiv:2608.15246 (CG-GLORE), Springer MVAP 2026 |
| **2** | **Limited-angle CT reconstruction** | Khôi phục ảnh CT khi dải góc quét bị giới hạn (thiếu wedge thông tin). Sinogram inpainting + Image-domain regularization. | AAPM, LoDoPaB-CT | Springer MVAP 2026, arXiv:2608.15246 |
| **3** | **Cancer subtyping using Whole Slide Images (WSI)** | Phân loại dưới nhóm ung thư từ ảnh mô bệnh học gigapixel; Bag of patches & Multiple Instance Learning (MIL) với concept hướng dẫn (CGRL, CLAM, TransMIL). | TCGA-BRCA, TCGA-NSCLC | arXiv:2607.12556 (CGRL) |
| **4** | **Survival analysis using Whole Slide Images (WSI)** | Dự đoán thời gian sống sót / nguy cơ tử vong (time-to-event); Continual learning trên nhiều nhóm ung thư (MergeSurv, Porpoise, DeepAttnMISL). | TCGA Cohorts (LUAD, BLCA, KIRC) | arXiv:2607.04747 (MergeSurv) |
| **5** | **Tumor segmentation using CT/fMRI (MRI)** | Phân vùng khối u (u não Glioma, u nội tạng) trên ảnh thể tích 3D. Sử dụng MONAI Swin-UNETR, DynUNet, 3D U-Net, nnU-Net. | BraTS 2021/2023, LiTS | arXiv:1804.10484, GitHub: DBC-Lab/Brain_MRI_Enhancement |
| **6** | **Weakly-supervised segmentation for WSI** | Phân vùng tổn thương giải phẫu bệnh chỉ với nhãn mức slide hoặc ảnh độ phóng đại thấp (LowMagWSS). | Camelyon16, BCSS | arXiv:2607.10783 (LowMagWSS), GitHub: Dung-Dx/LowMagWSS |
| **7** | **Collateral imaging in acute ischemic stroke using MRI** | Đánh giá tuần hoàn bàng hệ trong đột quỵ thiếu máu não cấp; lượng hóa vùng tranh tối tranh sáng (penumbra) và lõi nhồi máu (infarct core). | ISLES 2022, Perfusion MRI | Springer Int J CARS (10.1007/s11548-024-03229-5) |
| **8** | **Tự do đề xuất (Free Medical AI Proposals)** | - Phát hiện phổ tự kỷ (ASD) dựa trên fMRI não (Dataset ABIDE I, functional connectivity GNN/CNN).<br>- Ứng dụng hỏi đáp / trợ lý VQA hỗ trợ bác sĩ chẩn đoán hình ảnh. | ABIDE I, VQA-RAD, PathVQA | Đề xuất nhóm (Nhóm 7 & Nhóm 8 trong Excel) |

---

## 3. Dataset Schemas & Storage Locations

> **Trọng tâm Nghiên cứu Đồ án:** Hệ thống tập trung chuyên sâu vào **bộ dữ liệu chuẩn BCSS-WSSS** (và tập kiểm chứng chéo **LUAD-HistoSeg**). Các bộ dữ liệu khảo sát ban đầu (AAPM, BraTS, ABIDE I, Camelyon16, ISLES) thuộc các đề tài khác trong môn học đã được gác lại để dồn toàn lực tối ưu hóa SOTA cho phân vùng WSI giám sát yếu.

1. **BCSS (Breast Cancer Semantic Segmentation - WSSS Subset):**
   - Định dạng: Tệp nén `BCSS-WSSS.zip` chứa các lát cắt mô học độ phân giải $224 \times 224$ px hoặc $512 \times 512$ px.
   - Cấu trúc thư mục chuẩn: `train/img/`, `train/mask/` (hoặc nhãn mức slide), `val/img/`, `val/mask/`, `test/img/`, `test/mask/`.
   - 5 phân lớp giải phẫu bệnh:
     * Lớp 0: Khối u xâm lấn (Tumor / TUM)
     * Lớp 1: Mô đệm đan xen (Stroma / STR)
     * Lớp 2: Tế bào lympho thâm nhiễm (Inflammatory / Lymphocytic)
     * Lớp 3: Vùng hoại tử (Necrosis / NEC)
     * Lớp 4: Nền kính phết / Nền không tế bào (Background / Other)
   - Phân chia bệnh nhân: Tuân thủ nghiêm ngặt theo `src/config/bcss_patient_splits.json` (106 train / 22 val / 23 test). Tuyệt đối không để rò rỉ lát cắt giữa cùng một bệnh nhân (Patient ID) qua các tập train, val, test.

2. **LUAD-HistoSeg (Cross-Dataset Evaluation Subset):**
   - Định dạng: Các mảng ảnh mô bệnh học ung thư biểu mô tuyến phổi (Lung Adenocarcinoma) kích thước $224 \times 224$ px.
   - Mục đích: Đánh giá khả năng tổng quát hóa không gian (out-of-distribution generalization) xuyên cơ quan và xuyên phòng xét nghiệm giải phẫu bệnh.

---

## 4. Reference Repositories & External Authoritative Sources

Toàn bộ mã nguồn SOTA đã được clone và lưu trữ tại thư mục `repos/` để phục vụ việc đối soát trực tiếp theo quy trình Tầng 3 (Tier 3 Recheck):

1. **`repos/LowMagWSS` (arXiv:2607.10783):**
   - Repo gốc: `https://github.com/Dung-Dx/LowMagWSS`
   - Phương pháp: Weakly-Supervised Semantic Segmentation on WSI using Low-Magnification Cues.
   - Đường dẫn tham chiếu kiến trúc: `repos/LowMagWSS/models/`, `repos/LowMagWSS/datasets/`.

2. **`repos/CG-GLORE` (arXiv:2608.15246):**
   - Phương pháp: Coordinate Gradient Guided Global-Local Regularization for Sparse-View CT.
   - Đường dẫn tham chiếu unrolling: `repos/CG-GLORE/networks/`.

3. **`repos/MergeSurv` (arXiv:2607.04747):**
   - Phương pháp: Continual Learning for Survival Analysis on Whole Slide Images.

4. **`repos/CGRL` (arXiv:2607.12556):**
   - Phương pháp: Concept-Guided Representation Learning for WSI Cancer Subtyping.

5. **`repos/meta_tools`:**
   - Kho tiện ích bổ trợ gồm Graphify CLI, Linters, SAST script kiểm tra an toàn mã nguồn.

---

## 5. Hardware Specifications & Colab Runtime Configurations

1. **Google Colab Free Tier (Tesla T4 GPU):**
   - VRAM: 15 GB GDDR6.
   - System RAM: 12.7 GB.
   - Batch size tối ưu: $B=8$ hoặc $B=16$ (kích thước patch $224 \times 224$).
   - Thời gian giới hạn: Phiên ngắt kết nối sau 6 - 12 giờ.

2. **Google Colab Pro / Enterprise (NVIDIA L4 GPU / A100 GPU):**
   - VRAM: 24 GB (L4) / 40 GB hoặc 80 GB (A100).
   - System High-RAM: 51 - 83 GB.
   - Batch size tối ưu: $B=32$ hoặc $B=64$.
   - Mức tiêu thụ tài nguyên: ~1.5 - 2.0 compute units/giờ (L4); ~9.5 - 13.0 compute units/giờ (A100).

3. **Môi Trường Phiên Bản PyTorch / CUDA:**
   - Python: >= 3.10
   - PyTorch: >= 2.0.0
   - CUDA: >= 11.8 / 12.1
   - MONAI: >= 1.3.0
