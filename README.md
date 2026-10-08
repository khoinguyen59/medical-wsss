# SE374.R11: Đồ Án Cuối Kỳ - Trí Tuệ Nhân Tạo Trong Y Tế & Xử Lý Ảnh Y Khoa

> **Đơn vị đào tạo:** Trường Đại học Công nghệ Thông tin (UIT) - ĐHQG-HCM  
> **Khoa:** Khoa Kỹ thuật Phần mềm (Software Engineering)  
> **Môn học:** SE374.R11 - Đồ án cuối kỳ (Năm học 2025-2026)  
> **Lĩnh vực:** Trí tuệ Nhân tạo Y tế & Xử lý ảnh Y khoa (Medical AI, Medical Imaging & Computational Pathology)  
> **Hệ thống AI Agent:** `SE374MedicalAIAgent` tuân thủ tiêu chuẩn `AGENTS.md` v5.0  

---

## 1. Tổng Quan Môn Học & Danh Mục 8 Chủ Đề Nghiên Cứu

Theo hướng dẫn từ Giảng viên (`topic.md`) và danh sách đăng ký đề tài của lớp (`SE374.R11 - Đồ án cuối kỳ.xlsx`):

| STT | Tên Chủ Đề (Topic Name) | Bản Chất Bài Toán & Phương Pháp Cốt Lõi | Dữ Liệu Tham Chiếu | Tài Liệu Tham Khảo Chính |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Sparse-view CT reconstruction** *(Trọng tâm Buổi 2)* | Tái tạo ảnh cắt lớp CT chất lượng cao từ số góc chiếu thưa nhằm giảm liều chiếu xạ cho bệnh nhân; triệt tiêu streak artifacts bằng Deep Unrolling (CG-GLORE, LPD). | AAPM Low-Dose CT, DeepLesion | arXiv:2608.15246, Springer MVAP 2026 |
| **2** | **Limited-angle CT reconstruction** | Tái tạo ảnh CT khi góc quét đầu thu bị hạn chế (thiếu wedge dữ liệu); kết hợp phục hồi miền Sinogram và miền Ảnh. | AAPM, LoDoPaB-CT | Springer MVAP 2026, arXiv:2608.15246 |
| **3** | **Cancer subtyping using Whole Slide Images (WSI)** | Phân loại dưới nhóm mô ung thư từ ảnh mô bệnh học gigapixel; Bag of patches & Multiple Instance Learning có concept dẫn đường (CGRL, CLAM, TransMIL). | TCGA-BRCA, TCGA-NSCLC | arXiv:2607.12556 (CGRL) |
| **4** | **Survival analysis using Whole Slide Images (WSI)** | Phân tích thời gian sống sót và phân tầng nguy cơ tử vong của bệnh nhân ung thư; Continual Learning không quên tri thức cũ (MergeSurv). | TCGA (LUAD, BLCA, KIRC) | arXiv:2607.04747 (MergeSurv) |
| **5** | **Tumor segmentation using CT/fMRI (MRI)** | Phân vùng thể tích 3D u não (Glioma) hoặc u nội tạng; kiến trúc Swin-UNETR, 3D U-Net, DynUNet (MONAI), nnU-Net. | BraTS 2021/2023, LiTS | arXiv:1804.10484, GitHub: DBC-Lab/Brain_MRI_Enhancement |
| **6** | **Weakly-supervised segmentation for WSI** | Phân vùng tổn thương giải phẫu bệnh siêu phân giải chỉ với nhãn mức slide hoặc ảnh độ phóng đại thấp (LowMagWSS). | Camelyon16, BCSS | arXiv:2607.10783, GitHub: Dung-Dx/LowMagWSS |
| **7** | **Collateral imaging in acute ischemic stroke using MRI** | Đánh giá tuần hoàn bàng hệ trong đột quỵ thiếu máu cục bộ cấp; phân định vùng lõi nhồi máu (infarct core) và vùng tranh tối tranh sáng (penumbra). | ISLES 2022, Perfusion MRI | Springer Int J CARS (10.1007/s11548-024-03229-5) |
| **8** | **Tự do đề xuất (Free Proposals)** | - Phát hiện phổ tự kỷ (ASD) dựa trên fMRI não (Dataset ABIDE I, mạng kết nối chức năng não GNN/CNN).<br>- Ứng dụng hỏi đáp / trợ lý thị giác ngôn ngữ (VQA) hỗ trợ bác sĩ chẩn đoán hình ảnh. | ABIDE I, VQA-RAD, PathVQA | Đề xuất của sinh viên (Nhóm 7 & Nhóm 8 trong file Excel) |

---

## 2. Kiến Trúc Thư Mục Chuẩn Y Khoa (Clean Architecture)

```text
c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA/
├── AGENTS.md                          # Master Governance Specification (18 Luật Bất Biến Quản Trị)
├── README.md                          # Tài liệu tổng quan dự án (File này)
├── requirements.txt                   # Danh mục thư viện Python chuẩn mực (PyTorch, MONAI, NiBabel...)
├── topic.md                           # Danh mục topic & đường link bài báo gốc từ Giảng viên
├── SE374.R11 - Đồ án cuối kỳ.xlsx     # File Excel đăng ký đề tài của các nhóm trong lớp
├── .agents/                           # Bộ não tự trị của AI Agent
│   ├── rules/                         # 12 Luật kiến trúc & quy chuẩn học thuật y khoa
│   ├── workflows/                     # Quy trình chuẩn SOP từng bước (Workflow 00 -> 09)
│   └── skills/                        # Các kỹ năng chuyên sâu (medical-imaging-architect, etc.)
├── docs/                              # Tài liệu học thuật, phân tích chuyên sâu & templates
│   ├── SE374_TOPIC_MATRIX_AND_GUIDE.md# Cẩm nang phân tích chi tiết 8 topic, papers, datasets & SOTA code
│   ├── templates/                     # Biểu mẫu chính thức (UIT Thesis Word/LaTeX, Springer LNCS, Proposal)
│   ├── papers/                        # Lưu trữ bài báo khoa học PDF/MD
│   └── diary/                         # Nhật ký nghiên cứu & theo dõi tiến độ
├── src/                               # Mã nguồn chính theo Kiến trúc Module phân tầng
│   ├── config/                        # Cấu hình siêu tham số, random seeds, paths, device GPU
│   ├── datasets/                      # Data Ingestion (DICOM/NIfTI/WSI loaders, Patient-Level Split)
│   ├── models/                        # Kiến trúc mô hình (Reconstruction, Segmentation, WSI-MIL)
│   ├── losses_metrics/                # Loss Functions & Metrics (Dice, Focal, PSNR, SSIM, C-Index)
│   ├── visualization/                 # Trực quan hóa lát cắt 3D & Web Demo Streamlit
│   ├── pipeline/                      # Trình điều phối chạy tự động toàn chu trình
│   └── utils/                         # Hàm tiện ích y khoa (HU windowing, spacing normalization)
├── product/                           # Sản phẩm bàn giao cho Giảng viên
│   ├── proposal/                      # Đề cương Mini Proposal (DOCX / PDF)
│   ├── slides/                        # Slide thuyết trình 16:9 (Marp HTML, PDF, PPTX)
│   └── reports/                       # Báo cáo kỹ thuật cuối kỳ (Springer / UIT Word & LaTeX)
├── tests/                             # Unit tests & Sanity checks (Zero-Mockup & Leakage tests)
├── data/                              # Dữ liệu thực nghiệm (đã đưa vào .gitignore)
├── checkpoints/                       # Trọng số mô hình đã huấn luyện (đã đưa vào .gitignore)
└── repos/                             # Meta-repos tài nguyên thiết kế & công cụ ngoại tuyến
```

---

## 3. Các Nguyên Tắc Quản Trị Cốt Lõi (Core Governance Invariants)

1. **Patient-Level Anti-Leakage (Chống Rò Rỉ Dữ Liệu Bệnh Nhân):**
   - Phân chia `train / val / test` **bắt buộc theo Patient ID / Case ID**.
   - Tuyệt đối không chia ngẫu nhiên ở cấp độ lát cắt (slices) hoặc ô trích xuất (patches) để tránh rò rỉ dữ liệu cùng một bệnh nhân sang tập đánh giá.
2. **Liêm Chính Học Thuật Tuyệt Đối (Zero Mockup):**
   - 100% chỉ số (Dice, PSNR, SSIM, C-Index, AUC) phải chạy thực tế từ script trên dữ liệu thật.
   - Nghiêm cấm bịa đặt số liệu (vi phạm = 0 điểm).
3. **An Toàn Tài Nguyên & Dry-Run (Resource Safeguard):**
   - Ảnh y khoa 3D và WSI rất nặng. Luôn chạy thử nghiệm Sanity Check / Dry-run trên 1-2 batch nhỏ hoặc vài lát cắt trước khi huấn luyện quy mô lớn.
   - Luôn hỏi xác nhận từ người dùng trước khi train tốn nhiều giờ hoặc gọi API trả phí.
4. **Chuẩn Mực Công Cụ Tự Động Hóa:**
   - Biên dịch tài liệu Word bằng `Pandoc` với template UIT.
   - Thiết kế Slide 16:9 bằng `Marp CLI`.
   - Xây dựng giao diện demo bằng `Streamlit`.

---

## 4. Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt môi trường Python
```bash
cmd /c "pip install -r requirements.txt"
```

### 2. Kiểm thử tự động (CI/CD Local Gate)
```bash
cmd /c "pytest tests/"
cmd /c "flake8 src/ --max-line-length=120"
```

### 3. Cập nhật đồ thị tri thức mã nguồn (Graphify AST)
```bash
cmd /c "python scripts/update_graphify.py --profile main"
```
