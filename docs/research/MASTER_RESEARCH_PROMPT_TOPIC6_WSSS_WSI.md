# MASTER PROMPT: NGHIÊN CỨU TOÀN DIỆN VỀ WEAKLY-SUPERVISED HISTOPATHOLOGY & WSI SEGMENTATION

Tài liệu lưu trữ bộ Master Prompt chuyên dụng cho các công cụ AI Deep Research (Gemini Deep Research, ChatGPT Deep Research, Perplexity Pro, Claude 3.7/3.8, Antigravity Agent) nhằm tự động truy quét và tổng hợp toàn bộ tài nguyên học thuật, mã nguồn, mô hình và dữ liệu cho Topic 6: **Weakly-Supervised Semantic Segmentation for Histopathology / Whole Slide Images (WSI)**.

---

## BẢN 1: MASTER PROMPT TIẾNG VIỆT (TỐI ƯU HÓA CHO ĐỒ ÁN SE374.R11 - UIT)

```text
Bạn là một Tổng công trình sư AI Y tế và Kỹ sư Nghiên cứu Học sâu Giải phẫu bệnh (Senior Medical AI Architect & Computational Pathology Research Engineer). 

Tôi đang thực hiện đề tài nghiên cứu và phát triển phần mềm:
"Phân vùng ngữ nghĩa giám sát yếu trên ảnh tiêu bản mô bệnh học và tiêu bản toàn phần (Weakly-Supervised Semantic Segmentation for Histopathology Tissue & Whole Slide Images - WSI)".
Trọng tâm bài toán: Phân tầng các kiểu hình mô u (Biểu mô u, Mô đệm, Lympho bào, Hoại tử, Nền kính trắng) chỉ dựa trên nhãn phân loại nhị phân mảng ảnh (patch-level labels) hoặc nhãn yếu toàn slide mà không có mặt nạ vẽ tay cấp điểm ảnh. Các hướng phương pháp bao gồm: Cơ chế chú ý lũy tiến (Progressive Dropout Attention - PDA), Giám sát giả đa tầng (MLPS), Cổng phân loại (CGM), Trộn mẫu nhãn giả (Pseudo-Mask ClassMix), và Bộ trích xuất đặc trưng Foundation Model (CONCH, UNI, Virchow).

Hãy tiến hành một đợt Deep Research toàn diện, truy quét trên toàn mạng Internet (arXiv, PubMed, ScienceDirect/MedIA, IEEE TMI, Springer, MICCAI, GitHub, Hugging Face, Zenodo, Kaggle, GDC) và lập báo cáo chi tiết theo đúng cấu trúc 6 phần sau:

### 1. BẢNG TỔNG HỢP CÁC BÀI BÁO KHOA HỌC SOTA & KINH ĐIỂN
Hãy tìm và lập bảng so sánh các công trình hàng đầu từ 2019 đến 2026 (đặc biệt các bài báo: WSSS-Tissue MedIA 2022 của Chu Han et al., LowMagWSS arXiv:2607.10783 của Bùi Cao Doanh et al., PistoSeg AAAI 2023, HisynSeg IEEE TMI 2025, ConStruct MIDL 2026, PathMamba MICCAI 2024, PBIP CVPR 2025):
- Tên bài báo, Tác giả chính, Năm xuất bản, Tạp chí/Hội nghị, DOI hoặc link arXiv/PDF.
- Ý tưởng thuật toán cốt lõi (Cơ chế sinh nhãn giả CAM/PDA, cách tinh lọc ranh giới, hàm mất mát).
- Bộ dữ liệu thực nghiệm và chỉ số mIoU / mDice đạt được trên BCSS-WSSS, LUAD-HistoSeg, hoặc Camelyon16.
- Ưu điểm và nhược điểm thực tế khi triển khai.

### 2. KHO MÃ NGUỒN GITHUB CHẤT LƯỢNG CAO (ACTIONABLE REPOSITORIES)
Hãy tìm kiếm các repository GitHub công khai có mã nguồn PyTorch hoàn chỉnh, có thể tái lập thực nghiệm:
- Link repo GitHub chính thức của các bài báo liên quan (đặc biệt repo của WSSS-Tissue, LowMagWSS, HisynSeg, CLAM, TransMIL, OpenSlide).
- Tech stack của từng repo (PyTorch version, torchvision, timm, CUDA requirements).
- Trạng thái trọng số đã huấn luyện sẵn (Pretrained Weights / Checkpoints): Có sẵn link tải (Google Drive, OneDrive, Baidu, Hugging Face) hay phải tự train từ đầu?
- Mức độ dễ cấu hình và chạy thực tế trên máy tính cá nhân hoặc Google Colab / RunPod.

### 3. TÀI NGUYÊN TRÊN HUGGING FACE (MODELS, DATASETS & SPACES)
Hãy truy quét hệ sinh thái Hugging Face (huggingface.co) để tìm:
- Các mô hình nền tảng giải phẫu bệnh (Pathology Foundation Models) đã tải lên Hub: CONCH (MahmoodLab/CONCH), UNI (MahmoodLab/UNI), Virchow (paige-ai/Virchow), Prov-GigaPath, v.v. Kèm đường dẫn Model ID, giấy phép sử dụng (License) và cách gọi nạp feature extractor bằng thư viện `timm` hoặc `transformers`.
- Các bộ dữ liệu mô bệnh học đã được đưa lên Hugging Face Datasets (BCSS, Camelyon, TCGA patches) giúp tải nhanh bằng `datasets.load_dataset()`.
- Các ứng dụng Spaces (Gradio / Streamlit) liên quan đến phân tích WSI hoặc CAM visualization để tham khảo giao diện.

### 4. CÁC BỘ DỮ LIỆU CHUẨN MỰC & HƯỚNG DẪN TẢI THỰC TẾ
Tổng hợp thông tin chi tiết về các benchmark datasets phục vụ đề tài:
- BCSS-WSSS (Breast Cancer Semantic Segmentation): Số lượng mảng ảnh (31.826 patches), 151 ROIs, 5 phân lớp mô học, nguồn tải chuẩn xác.
- LUAD-HistoSeg (Lung Adenocarcinoma): 17.285 patches, 49 ca bệnh.
- Camelyon16 / Camelyon17 và TCGA-BRCA: Quy mô file WSI (.svs), cách tải trực tiếp hoặc qua GDC Data Transfer Tool.
- Nguyên tắc phân chia dữ liệu chống rò rỉ bệnh nhân (Patient-Level Split): Cách nhóm và chia Train/Val/Test theo Patient ID để tránh gian lận học thuật.

### 5. KIẾN TRÚC PHẦN MỀM & THƯ VIỆN CÔNG CỤ THỰC THI (SOFTWARE STACK)
Liệt kê các thư viện chuyên dụng tốt nhất thế giới hiện nay để xây dựng ứng dụng chẩn đoán:
- Thư viện đọc và tiền xử lý ảnh WSI: OpenSlide, cuCIM, TIAToolbox, Histolab (phân tích ưu nhược điểm trên Windows/Linux).
- Giải thuật tiền xử lý: Lọc nền kính trắng (Otsu Thresholding), phát hiện mô sống, cắt lưới đa tầng (Tiling).
- Trình xem ảnh tiêu bản tương tác: OpenSeadragon (DeepZoom viewer), QuPath integration.
- Dịch vụ suy luận: Đóng gói API bằng FastAPI, tối ưu hóa suy luận bằng TorchScript / ONNX Runtime, tự động đo diện tích khối u sống ($mm^2$) và tỷ lệ tải lượng u (Viable Tumor Percentage).

### 6. CÁC CẠM BẪY KỸ THUẬT & KHUYẾN NGHỊ THỰC HIỆN
- Những lỗi ngầm kinh điển trong bài toán WSSS mô học (ví dụ: lỗi Cổng CGM ép xác suất về 0 dẫn đến argmax đoán nhầm nền kính trắng thành u; lỗi rò rỉ bệnh nhân giữa train và test; lỗi trộn ảnh Mixup tuyến tính gây bóng mờ nhân tế bào).
- Lời khuyên thiết thực để một nhóm sinh viên có thể hoàn thành xuất sắc đồ án trong 10 tuần trên 1 GPU máy trạm hoặc Cloud GPU (Colab/RunPod) với chi phí thấp nhất.

Yêu cầu định dạng: Báo cáo có cấu trúc chặt chẽ, sử dụng Markdown chuẩn, có trích dẫn link cụ thể (DOI, URL GitHub, Hugging Face ID), số liệu trung thực, văn phong khoa học chuyên nghiệp.
```

---

## BẢN 2: MASTER PROMPT TIẾNG ANH (CHO GEMINI DEEP RESEARCH & CHATGPT PRO)

```text
Act as a Principal Medical AI Architect and Senior Computational Pathology Research Scientist.

I am leading an advanced research and engineering project entitled:
"Weakly-Supervised Semantic Segmentation (WSSS) for Histopathology Tissue and Whole Slide Images (WSI)".
Core Problem Formulation: Multi-class segmentation of clinically vital tissue phenotypes (Tumor Epithelium, Stroma, Tumor-Infiltrating Lymphocytes, Necrosis, and Background) using only patch-level classification labels or low-magnification slide-level tags, completely bypassing dense pixel-level manual masks. Key paradigms include: Progressive Attention Dropout (PDA), Multi-Layer Pseudo-Supervision (MLPS), Classification Gate Mechanism (CGM), Pseudo-Mask ClassMix boundary synthesis, and Pathology Foundation Model adapters (CONCH, UNI, Virchow).

Conduct an exhaustive, publication-grade deep research survey across scientific literature (arXiv, PubMed, ScienceDirect/MedIA, IEEE TMI, Springer, MICCAI, CVPR, ICCV), open-source codebases (GitHub), model hubs (Hugging Face), and biomedical repositories. Produce a comprehensive report covering the following 6 pillars:

### 1. SEMINAL & SOTA RESEARCH LANDSCAPE (2019-2026)
Provide a detailed analytical table of state-of-the-art weakly-supervised histopathology segmentation methods:
- Target papers: WSSS-Tissue (Chu Han et al., MedIA 2022, DOI: 10.1016/j.media.2022.102487), LowMagWSS (Bui et al., arXiv:2607.10783 / Springer MVAP 2026), PistoSeg (AAAI 2023), HisynSeg (IEEE TMI 2025), ConStruct (MIDL 2026), PathMamba (MICCAI 2024), and PBIP (CVPR 2025).
- For each paper: Formal method name, Authors, Venue, Year, DOI/arXiv link.
- Algorithmic innovation: Attention dropout mechanics, multi-scale feature supervision, false-positive suppression, and boundary refinement loss formulations.
- Benchmark performance: Quantitative results on BCSS-WSSS (mIoU / mDice) and LUAD-HistoSeg.
- Strengths, critical failure modes, and hardware feasibility.

### 2. OFFICIAL & COMMUNITY GITHUB REPOSITORIES
Identify verified, reproducible PyTorch GitHub repositories:
- Direct repository links for WSSS-Tissue, LowMagWSS, HisynSeg, PistoSeg, CLAM, TIAToolbox, and related weakly-supervised pipelines.
- Technical environment specifications: PyTorch version, torchvision, timm, CUDA requirements, Docker availability.
- Pretrained weights availability: Checkpoints links (Google Drive, Hugging Face, Baidu) for ResNet-38 ImageNet backbone and DeepLabV3+ stage-2 weights.
- Reproducibility score: Ease of setup on local Windows/Linux machines or cloud environments (RunPod, Lambda, Google Colab).

### 3. HUGGING FACE ECOSYSTEM (MODELS, WEIGHTS & DATASETS)
Map out existing Hugging Face assets (huggingface.co):
- Foundation Model Checkpoints: MahmoodLab/CONCH, MahmoodLab/UNI, paige-ai/Virchow, Prov-GigaPath. Document exact Hugging Face Hub IDs, access gate requirements, model architectures (ViT-B, ViT-L), and Python loading snippets using `timm` or `transformers`.
- Published Histopathology Datasets: Publicly hosted patch datasets (BCSS, Camelyon, TCGA) available via `datasets.load_dataset()`.
- Interactive Spaces: Notable Gradio / Streamlit demos showcasing WSI visualization, CAM heatmaps, or patch classification.

### 4. BENCHMARK DATASETS & PATIENT-LEVEL ANTI-LEAKAGE PROTOCOL
Compile rigorous specifications for standard evaluation benchmarks:
- BCSS-WSSS (Breast Cancer): Exact patch count (31,826 total patches: 23,422 Train, 3,418 Val, 4,986 Test), 151 ROIs, 5-class ontology (Tumor, Stroma, Lymphocyte, Necrosis, Other). Download endpoints.
- LUAD-HistoSeg: 17,285 patches across 49 patients.
- Gigapixel WSIs: Camelyon16 (svs files) and TCGA-BRCA GDC portal access procedures.
- Patient-Level Anti-Leakage Protocol: Mathematical and engineering guidelines for strict zero-leakage patient-level partitioning ($\mathcal{P}_{train} \cap \mathcal{P}_{val} = \emptyset, \mathcal{P}_{train} \cap \mathcal{P}_{test} = \emptyset$) to prevent tissue morphology memorization.

### 5. PRODUCTION SOFTWARE ARCHITECTURE & TOOLING STACK
Recommend an end-to-end clinical workstation software stack:
- WSI Processing Libraries: Comparison of OpenSlide, cuCIM, TIAToolbox, and Histolab (handling Windows compatibility, I/O bottlenecks).
- Tissue Detection: Otsu thresholding, saturation filtering, dynamic grid tiling with overlap.
- Gigapixel Web Visualization: OpenSeadragon integration, DeepZoom Image (DZI) tile serving.
- Backend & Inference Service: FastAPI async REST endpoints, TorchScript/ONNX optimization for low-latency patch inference (<1.5s on GPU/CPU), and automated clinical metric calculation:
  $$\text{Viable Tumor Area } (mm^2) = N_{tumor\_pixels} \times (\text{MPP})^2$$
  $$\text{Viable Tumor Percentage } (\%) = \frac{N_{tumor\_pixels}}{N_{tissue\_pixels}} \times 100\%$$

### 6. CRITICAL PITFALLS, HIDDEN ARTIFACTS & IMPLEMENTATION ADVICE
- Highlight obscure implementation dilemmas: e.g., the zero-probability argmax bug in Classification Gate Mechanism (where blank background patches mistakenly trigger class 0 Tumor Epithelium when all foreground channels are gated to zero), and how to resolve it with pre-inference tissue masks.
- Linear Mixup vs. Pseudo-Mask ClassMix: Why linear alpha-blending creates unphysical optical ghosting in H&E microscopy, and why spatial ClassMix preserves tissue cellular morphology.
- Strategic guidance for delivering a fully working, reproducible capstone project within 10 weeks on budget-friendly compute.

Provide your output with strict scientific rigor, explicit mathematical notation (LaTeX), direct verified hyperlinks, and publication-ready Markdown formatting.
```
