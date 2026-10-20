# BỘ SIÊU PROMPT KHẢO SÁT HỆ THỐNG CHUYÊN SÂU (DEEP RESEARCH PROMPT 2024 - 2026)

Tài liệu này cung cấp bộ chỉ thị nghiên cứu chuyên sâu (Deep Research Directive) được thiết kế đặc thù cho các công cụ AI Deep Research (Gemini Deep Research, ChatGPT Deep Research, Perplexity Pro Research, Claude with Research). 

Prompt giải quyết triệt để các hạn chế của các câu lệnh thông thường:
- Ép AI thực hiện quét đa tầng qua Google Scholar, PubMed/PMC, arXiv, Semantic Scholar, IEEE Xplore, ScienceDirect/MedIA, CVPR, ECCV, MICCAI.
- Mở rộng toàn bộ phổ thuật ngữ công nghệ mới giai đoạn 2024 - 2026 (Foundation Models, Vision-Language Adapters, Prompt-based Image Prompting, Mamba/State-Space Models, Pseudo-mask Synthesis, Topological Consistency).
- Đặt định ngạch số lượng bắt buộc (tối thiểu 20 - 30 công trình từ 2024 đến 2026) kèm số liệu định lượng tuyệt đối (mIoU %, Dice %, per-class metrics, GPU-hours).
- Truy quét mã nguồn GitHub còn hoạt động và kho mô hình Hugging Face Hub.

---

## 1. BẢN TIẾNG ANH (KHUYÊN DÙNG ĐỂ ĐẠT HIỆU SUẤT TRUY TÌM TỐI ĐA TRÊN SCHOLAR & PREPRINTS)

```text
Role: You are a Lead Research Scientist in Computational Pathology and Senior Systematic Reviewer for Medical Image Analysis (IEEE TMI / MedIA standards).

Objective: Execute an EXHAUSTIVE, QUANTITATIVE, RECURSIVE DEEP RESEARCH SURVEY on:
"Weakly-Supervised Semantic Segmentation (WSSS) and Dense Tissue Phenotyping for Histopathology Whole Slide Images (WSI): State-of-the-Art Advances, Benchmarks, and Codebases from 2024 to 2026".

### MANDATORY RECURSIVE SEARCH PROTOCOL
Do not stop after a few surface-level search queries. You must recursively execute searches across the following academic repositories and specialized databases:
1. Primary Scholarly Indexes:
   - Google Scholar (Advanced syntax: allintitle, author, venue filters)
   - PubMed Central (PMC) & Europe PMC (Search MeSH terms: "Computational Pathology", "Image Segmentation", "Machine Learning")
   - arXiv repository (Primary categories: cs.CV, eess.IV; filter strictly for years 2024, 2025, and 2026)
   - Semantic Scholar API & DBLP Computer Science Bibliography
2. Top-Tier Medical Imaging & Computer Vision Venues:
   - Journals: Medical Image Analysis (MedIA), IEEE Transactions on Medical Imaging (TMI), IEEE JBHI, Nature Biomedical Engineering, Nature Machine Intelligence, Computerized Medical Imaging and Graphics (CMIG).
   - Conferences (2024, 2025, and 2026 proceedings): MICCAI, MIDL, CVPR, ICCV, ECCV, NeurIPS, ICLR, AAAI, WACV, ISBI.
3. Open-Source Repositories & Hubs:
   - GitHub (search active repositories created or updated in 2024-2026 with tags: histopathology, wsss, wsi-segmentation, computational-pathology)
   - Hugging Face Hub (models and datasets under task: image-segmentation, tags: pathology, tissue, wsi)
   - Papers With Code (Leaderboards: BCSS, WSSS4LUAD, LUAD-HistoSeg, GlaS, Camelyon16/17, PanNuke, MoNuSeg)

### REQUISITE SEARCH OPERATORS & QUERY EXPANSION VECTORS
You must systematically query across these expanded paradigm keywords (spanning 2024-2026 terminology):
- Query Set 1 (Classic WSSS & Refinement Extensions):
  ("weakly supervised semantic segmentation" OR "WSSS" OR "patch-level supervision" OR "slide-level labels") AND ("histopathology" OR "whole slide image" OR "WSI") AND (2024 OR 2025 OR 2026)
- Query Set 2 (Foundation Models & Dense Prediction):
  ("pathology foundation model" OR "CONCH" OR "UNI" OR "Virchow" OR "Prov-GigaPath" OR "Phikon" OR "H-Optimus") AND ("semantic segmentation" OR "dense prediction" OR "token-level") AND (2024 OR 2025 OR 2026)
- Query Set 3 (Promptable & Vision-Language Segmentation):
  ("Segment Anything" OR "SAM" OR "promptable segmentation" OR "vision-language" OR "CLIP" OR "PLIP" OR "BiomedCLIP") AND ("histopathology" OR "tissue segmentation") AND (2024 OR 2025 OR 2026)
- Query Set 4 (Advanced Data-Centric & Boundary Synthesis):
  ("pseudo-label refinement" OR "class-mixing" OR "Bézier" OR "Mosaic" OR "topological consistency" OR "mutual learning" OR "activation relocalization" OR "structural distillation") AND ("histopathology" OR "WSI") AND (2024 OR 2025 OR 2026)
- Query Set 5 (State-Space Models & Long-Sequence Context):
  ("Mamba" OR "State Space Model" OR "SSM" OR "Vision Mamba" OR "VMamba") AND ("histopathology" OR "whole slide image" OR "WSI segmentation") AND (2024 OR 2025 OR 2026)

---

### MANDATORY REPORT DELIVERABLES & QUANTITATIVE TARGETS

#### PART 1: MASTER SOTA BENCHMARK TABLE (MINIMUM 20 TO 30 DISTINCT PAPERS FROM 2024-2026)
Build a comprehensive markdown table with AT LEAST 20 TO 30 PAPERS published strictly between 2024 and 2026. Every entry MUST contain verifiable numerical metrics:
Columns:
1. Paper Title & First Author
2. Publication Venue & Exact Year (e.g., CVPR 2025, MICCAI 2024, IEEE TMI 2025, MIDL 2026, arXiv 2024-2026)
3. Verified Direct Link (Valid DOI URL or arXiv ID/link)
4. Supervision Paradigm (Slide-level, Patch-level tags, Scribbles, Low-magnification WSI, Multi-modal prompt)
5. Algorithmic Innovation (e.g., Foundation token distillation, Prototype-based image prompting, Topo-consistency, Image-mixing synthesis, SSM context)
6. Benchmark Dataset(s) (e.g., BCSS, BCSS-WSSS, LUAD-HistoSeg, WSSS4LUAD, DigestPath, Camelyon16, CoNIC)
7. Exact Quantitative Metrics (Report EXACT numbers: mIoU %, Mean Dice %, Tumor IoU %, Stroma IoU %, Lymphocyte IoU %)
8. Computational Cost & Hardware (Report reported GPU types, VRAM footprint, and training duration where disclosed)

#### PART 2: ACTIVE GITHUB CODEBASES & REPRODUCIBILITY AUDIT (MINIMUM 10 REPOSITORIES)
Provide a curated directory of open-source GitHub codebases released or actively maintained in 2024-2026:
- Repository Name & Direct GitHub URL
- Corresponding Paper & Conference/Journal
- Framework & Tech Stack (PyTorch version, CUDA version, key libraries like MONAI, TIAToolbox, cuCIM, OpenSlide, timm)
- Pretrained Weights / Model Checkpoints: Direct public access links (Hugging Face, Zenodo, Baidu, Google Drive) and checkpoint formats (.pt, .pth, .ckpt)
- Hardware Requirements: Minimum VRAM for inference vs. training
- Practical Reproducibility Rating: High / Medium / Low (with concrete justification based on docstrings and environment files)

#### PART 3: HUGGING FACE COMPUTATIONAL PATHOLOGY HUB CATALOG
Compile the definitive directory of Hugging Face assets:
1. Foundation Model Weights Table:
   - Model Name (e.g., UNI, UNI v2, Virchow, Virchow2, CONCH, Prov-GigaPath, Phikon v2, H-Optimus-0)
   - Exact Hugging Face Hub Model ID (e.g., `MahmoodLab/UNI`, `paige-ai/Virchow`, `prov-gigapath/gigapath`)
   - Backbone Architecture (ViT-B/16, ViT-L/14, ViT-H/14, ConvNeXt)
   - Pretraining Scale (Number of WSIs / tiles)
   - Feature Extraction Compatibility (How to load via `timm` or `transformers`)
2. Public Pathology Datasets on Hub:
   - Dataset Name, HF Dataset ID, Number of samples, Mask annotations availability
3. Hosted Spaces & Interactive Tools:
   - Active Gradio / Streamlit pathology visualization demos on Hugging Face Spaces

#### PART 4: SYSTEMATIC PARADIGM SHIFT ANALYSIS (2024 TO 2026)
Synthesize the technical evolution into four clear thematic paradigms:
1. Beyond Classic CAM: Why activation relocalization, prototype prompting, and topological consistency replaced vanilla Grad-CAM in histology.
2. Synthetic Image-Mixing & Boundary Restoration: How Bézier curves, ClassMix, and generative self-supervision resolve the inter-cellular boundary collapse.
3. Foundation Model Adaptation for Dense Phenotyping: Methods transferring slide-level/tile-level token embeddings to pixel-dense segmentations (PEFT, LoRA, linear probing, lightweight decoders).
4. Linear Attention & State-Space Models (Mamba): Addressing gigapixel tissue scale without $O(N^2)$ GPU memory exhaustion.

#### PART 5: REPRODUCIBILITY GUIDE FOR RESOURCE-CONSTRAINED ENVIRONMENTS
- Classification matrix: Segment 2024-2026 SOTA methods into:
  * Category A: Feasible on Single Workstation / Free/Budget Cloud (8GB - 16GB VRAM, e.g., RTX 3060/4060, T4, A5000).
  * Category B: Requires Multi-GPU / High-VRAM Clusters (24GB - 80GB VRAM, e.g., A100, H100).
- Identify the Pareto-optimal architecture that maximizes mIoU while minimizing GPU memory and training time.

STRICT RESEARCH INTEGRITY CONSTRAINTS:
- ZERO hallucinated citations or fabricated numbers. If a specific metric is not reported in the paper, mark it as "NR" (Not Reported).
- NO vague qualifiers (e.g., "significantly outperforms baselines") without providing the exact comparative numeric percentages.
- Strictly prioritize literature published from January 2024 to early 2026.
```

---

## 2. BẢN TIẾNG VIỆT (DÀNH CHO CÁC MÔ HÌNH HỖ TRỢ TIẾNG VIỆT HOẶC BÁO CÁO TRỰC TIẾP)

```text
Vai trò: Bạn là Tổng công trình sư Nghiên cứu Trí tuệ Nhân tạo Y tế và Chuyên gia Khảo sát Hệ thống Độc lập (Senior Medical AI Systematic Reviewer) theo tiêu chuẩn khắt khe của IEEE TMI và Medical Image Analysis (MedIA).

Nhiệm vụ: Hãy thực hiện một chiến dịch KHẢO SÁT HỆ THỐNG VÀ THU THẬP DỮ LIỆU ĐỊNH LƯỢNG ĐA TẦNG TOÀN DIỆN NHẤT về đề tài:
"Phân vùng ngữ nghĩa giám sát yếu (Weakly-Supervised Semantic Segmentation - WSSS) và phân tầng kiểu hình mô bệnh học trên ảnh tiêu bản toàn phần (Whole Slide Images - WSI): Đột phá công nghệ, Bộ chỉ số đối chuẩn SOTA và Kho mã nguồn giai đoạn 2024 - 2026".

### GIAO THỨC TRUY TÌM SÂU ĐA TẦNG BẮT BUỘC
Tuyệt đối không dừng lại ở vài lượt tìm kiếm Google bề nổi. Bắt buộc phải thực hiện truy quét đệ quy qua các cơ sở dữ liệu học thuật quốc tế sau:
1. Các công cụ tìm kiếm và chỉ mục học thuật hàng đầu:
   - Google Scholar (sử dụng toán tử allintitle, venue, author chuyên sâu)
   - PubMed Central (PMC) và Europe PMC (các cụm từ MeSH chuyên ngành y sinh)
   - arXiv (chuyên mục cs.CV, eess.IV; lọc nghiêm ngặt các năm 2024, 2025 và 2026)
   - Semantic Scholar API và DBLP Computer Science Bibliography
2. Các diễn đàn khoa học đỉnh cao:
   - Tạp chí: Medical Image Analysis (MedIA), IEEE TMI, IEEE JBHI, Nature Biomedical Engineering, Nature Machine Intelligence, CMIG.
   - Hội nghị (kỷ yếu 2024, 2025, 2026): MICCAI, MIDL, CVPR, ICCV, ECCV, NeurIPS, ICLR, AAAI, WACV, ISBI.
3. Kho mã nguồn và nền tảng cộng đồng:
   - GitHub (truy lùng các repository có mã nguồn thực thi được, tạo hoặc cập nhật trong 2024-2026)
   - Hugging Face Hub (các checkpoint Foundation Model và Dataset chuyên ngành pathology)
   - Papers With Code (Bảng xếp hạng BCSS, WSSS4LUAD, LUAD-HistoSeg, GlaS, Camelyon16/17, PanNuke, MoNuSeg)

### TOÁN TỬ VÀ VECTƠ TỪ KHÓA MỞ RỘNG (GIAI ĐOẠN 2024 - 2026)
Phải mở rộng tìm kiếm theo các nhóm thuật ngữ công nghệ mới nhất:
- Nhóm 1 (WSSS kinh điển và biến thể):
  ("weakly supervised semantic segmentation" OR "WSSS" OR "patch-level supervision" OR "slide-level labels") AND ("histopathology" OR "whole slide image" OR "WSI") AND (2024 OR 2025 OR 2026)
- Nhóm 2 (Foundation Models và Dense Prediction):
  ("pathology foundation model" OR "CONCH" OR "UNI" OR "Virchow" OR "Prov-GigaPath" OR "Phikon" OR "H-Optimus") AND ("semantic segmentation" OR "dense prediction" OR "token-level") AND (2024 OR 2025 OR 2026)
- Nhóm 3 (Segment Anything và Vision-Language):
  ("Segment Anything" OR "SAM" OR "promptable segmentation" OR "vision-language" OR "CLIP" OR "PLIP" OR "BiomedCLIP") AND ("histopathology" OR "tissue segmentation") AND (2024 OR 2025 OR 2026)
- Nhóm 4 (Lọc nhiễu giả nhãn và Tổng hợp ranh giới mô học):
  ("pseudo-label refinement" OR "class-mixing" OR "Bézier" OR "Mosaic" OR "topological consistency" OR "mutual learning" OR "activation relocalization" OR "structural distillation") AND ("histopathology" OR "WSI") AND (2024 OR 2025 OR 2026)
- Nhóm 5 (State-Space Models / Mamba trên ảnh Gigapixel):
  ("Mamba" OR "State Space Model" OR "SSM" OR "Vision Mamba" OR "VMamba") AND ("histopathology" OR "whole slide image" OR "WSI segmentation") AND (2024 OR 2025 OR 2026)

---

### YÊU CẦU ĐẦU RA VÀ CHỈ TIÊU ĐỊNH LƯỢNG BẮT BUỘC

#### PHẦN 1: BẢNG MA TRẬN ĐỐI CHUẨN SOTA (TỐI THIỂU 20 ĐẾN 30 BÀI BÁO GIAI ĐOẠN 2024 - 2026)
Xây dựng bảng Markdown toàn diện gồm TỐI THIỂU 20 ĐẾN 30 CÔNG TRÌNH NGHIÊN CỨU xuất bản từ năm 2024 đến 2026. Mọi hàng bắt buộc phải có số liệu đo lường cụ thể:
1. Tên bài báo và Danh sách tác giả chính
2. Hội nghị / Tạp chí và Năm xuất bản chính xác (CVPR 2025, MICCAI 2024/2025, IEEE TMI 2025, MIDL 2026, arXiv 2024-2026)
3. Đường dẫn trực tiếp có thể kiểm chứng (Link DOI hoặc mã arXiv kèm URL)
4. Dạng thức giám sát (Nhãn mức slide, Nhãn mức mảng ảnh patch, Nhãn nét vẽ scribble, Ảnh độ phóng đại thấp, Prompt đa phương thức)
5. Cơ chế giải thuật cốt lõi (Chưng cất đặc trưng Foundation Model, Prompt-based image prompting, Tính nhất quán tô-pô, Tổng hợp trộn mẫu Bézier/Mosaic, SSM Mamba)
6. Bộ dữ liệu đối chuẩn thực nghiệm (BCSS, BCSS-WSSS, LUAD-HistoSeg, WSSS4LUAD, DigestPath, Camelyon16, CoNIC)
7. Kết quả số liệu định lượng chính xác (BẮT BUỘC ghi rõ con số % mIoU, % Dice; chỉ số từng lớp: Khối u, Mô đệm, Tế bào miễn dịch)
8. Chi phí phần cứng và điện toán công bố (Loại GPU, dung lượng VRAM, số giờ huấn luyện)

#### PHẦN 2: KHO MÃ NGUỒN GITHUB THỰC THI ĐƯỢC (TỐI THIỂU 10 REPOSITORIES)
Cung cấp danh mục các kho mã nguồn GitHub mở phát hành hoặc cập nhật trong 2024 - 2026:
- Tên repo và Đường dẫn URL trực tiếp trên GitHub
- Bài báo liên quan và Hội nghị/Tạp chí
- Tech Stack: Phiên bản PyTorch, CUDA, các thư viện lõi (MONAI, TIAToolbox, cuCIM, OpenSlide, timm)
- Trạng thái Checkpoints / Trọng số huấn luyện sẵn: Link tải trực tiếp (Hugging Face, Zenodo, Google Drive, Baidu) và định dạng file (.pt, .pth, .ckpt)
- Cấu hình phần cứng tối thiểu: Dung lượng VRAM cần cho inference so với khi huấn luyện
- Điểm đánh giá khả năng tái lập: Cao / Trung bình / Thấp (có giải thích cụ thể dựa trên tài liệu hướng dẫn và mã nguồn)

#### PHẦN 3: DANH MỤC HỆ SINH THÁI HUGGING FACE CHO COMPUTATIONAL PATHOLOGY
Tổng hợp danh mục chuẩn các tài sản trên Hugging Face Hub:
1. Bảng Foundation Model trọng số công khai:
   - Tên mô hình (UNI, UNI v2, Virchow, Virchow2, CONCH, Prov-GigaPath, Phikon v2, H-Optimus-0)
   - Mã định danh chính xác trên HF Hub (ví dụ: `MahmoodLab/UNI`, `paige-ai/Virchow`, `prov-gigapath/gigapath`)
   - Kiến trúc mạng (ViT-B/16, ViT-L/14, ViT-H/14, ConvNeXt)
   - Quy mô dữ liệu tiền huấn luyện (Số slide WSI / số mảng patch)
   - Cách gọi nạp đặc trưng qua thư viện Python (`timm` hoặc `transformers`)
2. Các bộ dữ liệu giải phẫu bệnh công khai trên Hugging Face Datasets có thể dùng `load_dataset()`.
3. Các bản Demo Spaces (Gradio / Streamlit) trực quan hóa WSI đang chạy trên Hugging Face.

#### PHẦN 4: BỨC TRANH DỊCH CHUYỂN CÔNG NGHỆ (PARADIGM SHIFTS 2024 - 2026)
Phân tích sâu tiến trình chuyển dịch qua 4 làn sóng công nghệ:
1. Vượt qua giới hạn của CAM kinh điển: Tại sao các kỹ thuật relocalization, prototype prompting và topological consistency thay thế Grad-CAM trên mô học.
2. Tổng hợp trộn ảnh và Phục hồi ranh giới mô học: Cơ chế đường cong Bézier, ClassMix và tự giám sát chống suy biến ranh giới tế bào.
3. Thích ứng Foundation Model cho phân vùng chi tiết: Các phương pháp đưa embedding cấp slide/patch về phân vùng cấp pixel (PEFT, LoRA, lightweight decoders).
4. Mạng chú ý tuyến tính và Mô hình không gian trạng thái (Mamba): Giải quyết ngữ cảnh WSI gigapixel mà không bị bùng nổ bộ nhớ GPU.

#### PHẦN 5: CẨM NANG TRIỂN KHAI TRÊN PHẦN CỨNG GIỚI HẠN
- Phân loại rõ: Giải pháp nào từ 2024-2026 khả thi trên 1 GPU máy trạm đơn / Cloud phổ thông (8GB - 16GB VRAM như RTX 3060/4060, T4, A5000), và giải pháp nào bắt buộc cụm A100/H100.
- Xác định kiến trúc tối ưu Pareto đạt mIoU cao nhất trên mỗi giờ tính toán GPU.

NGUYÊN TẮC LIÊM CHÍNH HỌC THUẬT BẮT BUỘC:
- TUYỆT ĐỐI KHÔNG bịa đặt trích dẫn hoặc số liệu. Nếu bài báo không công bố số liệu nào, ghi rõ "NR" (Not Reported).
- CẤM dùng các tính từ mơ hồ (như "vượt trội", "rất cao") mà không kèm con số % so sánh thực tế.
- Ưu tiên cao nhất các công trình công bố từ tháng 01/2024 đến đầu năm 2026.
```
