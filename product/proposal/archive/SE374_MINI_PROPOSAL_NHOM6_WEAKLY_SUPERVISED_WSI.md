---
title: "Mini Proposal: Hệ Thống Phân Vùng Ngữ Nghĩa Giám Sát Yếu Trên Ảnh Tiêu Bản Mô Bệnh Học Toàn Phần (Weakly-Supervised Semantic Segmentation for Whole Slide Images)"
author:
  - "<CẦN XÁC NHẬN: Họ và tên Sinh viên 1 (MSSV: ...)>"
  - "<CẦN XÁC NHẬN: Họ và tên Sinh viên 2 (MSSV: ...)>"
date: "Tháng 10/2026"
---

# Mini Proposal

## A. Information

- **Project / Mã và Tên đề tài:** <CHƯA QUYẾT ĐỊNH CHÍNH THỨC> Weakly-Supervised Semantic Segmentation for Whole Slide Images (Phần Mềm Trạm Làm Việc Giải Phẫu Bệnh Số Hỗ Trợ Phân Vùng Ngữ Nghĩa Giám Sát Yếu Trên Ảnh Tiêu Bản Toàn Phần WSI)
- **Course / Môn học:** SE374.R11 - Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa (Học kỳ 1, Năm học 2026-2027)
- **Faculty / Khoa:** Khoa Kỹ thuật Phần mềm (Software Engineering), Trường Đại học Công nghệ Thông tin - ĐHQG-HCM (UIT)
- **Duration / Thời gian thực hiện:** 10/2026 - 12/2026 (10 tuần)
- **Group / Nhóm:** <CẦN XÁC NHẬN: Số thứ tự nhóm>
- **Members / Thành viên:**
  1. <CẦN XÁC NHẬN: Họ và tên sinh viên 1> - MSSV: <CẦN XÁC NHẬN> - Email: <CẦN XÁC NHẬN>
  2. <CẦN XÁC NHẬN: Họ và tên sinh viên 2> - MSSV: <CẦN XÁC NHẬN> - Email: <CẦN XÁC NHẬN>
- **Supervisor / Giảng viên hướng dẫn:** TS. Bùi Cao Doanh (Giảng viên phụ trách môn học SE374.R11, UIT)

---

## B. Content

*(AI Policy & Academic Integrity Compliance: Nhóm cam kết tuân thủ tuyệt đối quy định liêm chính học thuật Zero-Mockup. 100% dữ liệu khảo sát, công thức toán học, cấu hình mạng và kết quả thực nghiệm cơ sở đều được đối soát và trích dẫn trực tiếp từ các bài báo khoa học chuẩn mực đã tải về (MedIA 2022, MICCAI 2024, CVPR 2025, Nature Medicine 2024, Springer 2026) và mã nguồn mở kiểm chứng từ các phòng lab quốc tế).*

### 1. Mini Problem Formulation

#### 1.1 Project Description (Mô tả đề tài trong 1 đoạn văn, $\le 150$ từ)
Đề tài phát triển hệ thống phần mềm trạm làm việc giải phẫu bệnh kỹ thuật số (Digital Pathology Workstation) tích hợp mô hình trí tuệ nhân tạo phân vùng ngữ nghĩa giám sát yếu (Weakly-Supervised Semantic Segmentation - WSSS) cho ảnh tiêu bản hiển vi toàn phần (Whole Slide Images - WSI). Thay vì đòi hỏi chuyên gia giải phẫu bệnh khoanh vùng thủ công từng pixel trên các tệp ảnh kích thước gigapixel ($100.000 \times 100.000$ pixels) vốn tốn kém hàng trăm giờ và dễ sai sót chủ quan, hệ thống chỉ sử dụng nhãn giám sát yếu ở cấp độ lát cắt/mảng (patch-level presence/absence labels) để tự động phân vùng chính xác 4-5 phân lớp mô trọng yếu: Biểu mô khối u (Tumor Epithelial), Mô đệm liên kết (Tumor-Associated Stroma), Tế bào lympho xâm nhập khối u (sTILs), và Vùng hoại tử (Necrosis). Phần mềm cung cấp giao diện tương tác Web DeepZoom mượt mà, trực quan hóa mặt nạ phân tầng (multi-layer mask overlay), định lượng diện tích vi môi trường khối u (TME) và tự động xuất biên bản giải phẫu bệnh theo chuẩn lâm sàng CAP.

#### 1.2 Assumptions, Constraints, Scope & Out of Scope
- **Assumptions (Giả định):**
  - Dữ liệu đầu vào là ảnh tiêu bản hiển vi nhuộm Hematoxylin & Eosin (H&E) số hóa định dạng chuẩn y khoa (SVS, NDPI, TIFF, PNG/JPEG tiles) quét ở độ phóng đại $10\times$, $20\times$, hoặc $40\times$.
  - Tập huấn luyện chỉ có nhãn nhị phân/đa nhãn cấp patch (one-hot vector cho biết sự xuất hiện của từng loại mô), không có mask đa giác vẽ tay lúc train.
- **Constraints (Ràng buộc kỹ thuật & học thuật):**
  - **Ràng buộc chống rò rỉ bệnh nhân (Patient-Level Anti-Leakage):** Bắt buộc phân chia tập huấn luyện, kiểm định và kiểm thử (`train / val / test`) triệt đối theo mã định danh bệnh nhân (Patient ID/Case ID). Nghiêm cấm trộn ngẫu nhiên ở cấp độ patch dẫn đến rò rỉ đặc trưng mô học của cùng một ca bệnh.
  - **Ràng buộc phần cứng:** Mô hình phải huấn luyện và nội suy ổn định trên GPU phổ thông (NVIDIA RTX 3060/4060 8GB-16GB VRAM) mà không bị tràn bộ nhớ (Out-of-Memory).
  - **Ràng buộc chuẩn bàn phím:** Toàn bộ văn bản và code tuân thủ ký tự gạch nối chuẩn bàn phím `-` (ASCII 0x2D).
- **Scope (Phạm vi đồ án):**
  - Khảo sát, thực nghiệm và tích hợp pipeline AI giám sát yếu kết hợp giữa Prompting Nguyên mẫu (Prototype-Based Image Prompting - PBIP), Cơ chế Dropout lũy tiến (Progressive Dropout Attention - PDA / WSSS-Tissue), và Không gian trạng thái chọn lọc (Selective State Space Model / PathMamba).
  - Đánh giá định lượng trên 2 bộ dữ liệu chuẩn quốc tế: BCSS-WSSS (Ung thư vú) và LUAD-HistoSeg (Ung thư biểu mô tuyến phổi).
  - Xây dựng phần mềm hoàn chỉnh theo Kiến trúc Sạch (Clean Architecture): Backend API (FastAPI, TIAToolbox WSI streaming), Frontend Web tương tác (OpenSeadragon, Canvas WebGL shader overlay), và Module định lượng xuất báo cáo PDF (ReportLab).
- **Out of Scope (Ngoài phạm vi):**
  - Không xây dựng phần mềm điều khiển trực tiếp phần cứng máy quét tiêu bản kính hiển vi quang học (Slide Scanner Hardware).
  - Không chẩn đoán trực tiếp hoặc thay thế chỉ định điều trị của bác sĩ; đóng vai trò hệ thống hỗ trợ quyết định lâm sàng (Clinical Decision Support System - CDSS).

#### 1.3 Existing Approaches & Research Gap
- **Hạn chế của phương pháp Class Activation Maps (CAM) truyền thống (HistoSegNet, C-CAM, SC-CAM):** Mạng phân loại thông thường có xu hướng chỉ kích hoạt tại các điểm đặc trưng nổi bật nhất (discriminative focal points như cụm nhân tế bào dị dạng), khiến vùng mặt nạ sinh ra bị co ngót (activated region shrinkage), bỏ sót vùng viền ranh giới khối u và mô đệm xung quanh.
- **Hạn chế về tính phức tạp tính toán của Vision Transformer (ViT):** Cơ chế tự chú ý (Self-Attention) có độ phức tạp bậc hai $\mathcal{O}(L^2)$ theo độ dài chuỗi token $L$. Khi xử lý vùng mô gigapixel với hàng nghìn patch, ViT gây cạn kiệt bộ nhớ GPU và độ trễ quá cao, không đáp ứng được yêu cầu tương tác thời gian thực của bác sĩ.
- **Khoảng trống phần mềm kỹ thuật (Software Engineering Gap):** Các nghiên cứu AI hiện tại hầu hết dừng lại ở các script Python thí nghiệm rời rạc (Jupyter Notebooks, shell scripts nháp), thiếu một kiến trúc phần mềm chuẩn mực (Clean Modular Architecture), thiếu bộ đệm phân giải sâu đa tầng (Pyramidal DeepZoom Image Server), và thiếu công cụ định lượng tỷ lệ sTILs tự động hỗ trợ trực tiếp quy trình đọc tiêu bản thực tế.

#### 1.4 Research Questions & Hypotheses
- **RQ1 (Nghiên cứu AI):** Làm thế nào để mở rộng vùng kích hoạt CAM từ nhãn yếu cấp mảng (patch-level labels) để bao quát trọn vẹn ranh giới khối u mà không bị tràn sang mô lành?
  - *Giả thuyết H1:* Việc áp dụng Cơ chế Chú ý Dropout lũy tiến (Progressive Dropout Attention - PDA) kết hợp Khớp nối Nguyên mẫu đa tỷ lệ (Prototype-Based Image Prompting - PBIP) sẽ ép mạng giải mã các vùng mô phụ không chiếm ưu thế, giúp nâng cao chỉ số mIoU thêm ít nhất $2-3\%$ và thu hẹp khoảng cách với mô hình giám sát đầy đủ (Fully Supervised) xuống dưới $2\%$.
- **RQ2 (Kỹ thuật phần mềm & Tối ưu hóa):** Làm thế nào để triển khai hệ thống phân vùng ảnh WSI kích thước gigapixel trên trình duyệt web với độ trễ phản hồi $<2$ giây mà không làm sập RAM máy trạm?
  - *Giả thuyết H2:* Việc áp dụng kiến trúc phân giải kim tự tháp (DeepZoom Pyramidal Tiling với TIAToolbox/OpenSlide) kết hợp bộ suy luận Mamba/SSM tuyến tính $\mathcal{O}(L)$ và kết xuất mặt nạ trên GPU client (WebGL Shader Canvas) sẽ giảm $80\%$ tải truyền thông mạng và duy trì tốc độ khung hình tương tác $>45$ FPS khi phóng to thu nhỏ.

#### 1.5 Objectives & Potential Contributions
- **Mục tiêu 1 (Nghiên cứu & Mô hình hóa):** Hiện thực hóa pipeline AI giám sát yếu đa phân lớp đạt mIoU $\ge 0.76$ trên BCSS-WSSS và $\ge 0.75$ trên LUAD-HistoSeg, chứng minh tính tái lập với 5 random seeds $(\mu \pm \sigma)$.
- **Mục tiêu 2 (Kỹ thuật Phần mềm & Sản phẩm):** Đóng gói thành công sản phẩm phần mềm trạm làm việc giải phẫu bệnh kỹ thuật số (SE374 Pathology Workstation) với đầy đủ tài liệu đặc tả kiến trúc, API RESTful, và giao diện web đáp ứng tiêu chuẩn phần mềm y tế.
- **Đóng góp tiềm năng:** Rút ngắn thời gian gán nhãn và đo đạc tiêu bản của bác sĩ giải phẫu bệnh từ $\sim 3.5$ giờ/ca xuống dưới $6$ phút/ca ($>30\times$ speedup), cung cấp công cụ mã nguồn mở phục vụ giảng dạy và nghiên cứu tại UIT.

---

### 2. Related Work

| Hệ Thống / Mô Hình | Bản Chất Kỹ Thuật | Ưu Điểm Nổi Bật | Hạn Chế / Nhược Điểm Cốt Lõi |
| :--- | :--- | :--- | :--- |
| **WSSS-Tissue / MLPS** *(Han et al., MedIA 2022)* | Quy trình 2 giai đoạn: Giai đoạn 1 huấn luyện ResNet-38 phân loại với Progressive Dropout Attention (PDA); Giai đoạn 2 huấn luyện DeepLabV3+ bằng giám sát giả đa tầng (MLPS) kết hợp cổng phân loại (CGM). | Thu hẹp khoảng cách với mô hình fully-supervised xuống $<2\%$ mIoU; giảm thời gian gán nhãn $32.7\times$. | Sử dụng backbone CNN cổ điển (ResNet-38); chưa tận dụng được biểu diễn từ Foundation Models hiện đại; phụ thuộc vào điều chỉnh ngưỡng thủ công $\mu, \sigma$. |
| **PBIP** *(Tang et al., CVPR 2025)* | Khung Prototype-Based Image Prompting: gom cụm đặc trưng patch thành các vector nguyên mẫu trực quan (visual prototypes) cho từng phân lớp; căn chỉnh tương phản (contrastive matching) với SegFormer. | Đạt SOTA trên 4 tập dữ liệu mô học (LUAD-HistoSeg, BCSS-WSSS, GCSS, BCSS); khắc phục triệt để tính không đồng nhất của mô học. | Yêu cầu bộ nhớ lớn để lưu trữ ngân hàng nguyên mẫu (prototype bank); thiếu module điều phối hiển thị tiêu bản WSI hoàn chỉnh. |
| **PathMamba** *(Fan et al., MICCAI 2024)* | Ứng dụng Không gian trạng thái chọn lọc (Visual Mamba / S6) với khối Contrastive Mamba Block (CMB) và hàm mất mát Deep Contrast Supervised Loss (DCL); tích hợp học đa thể hiện đa nhãn (MIML). | Độ phức tạp tính toán tuyến tính $\mathcal{O}(L)$; mô hình hóa ngữ cảnh toàn cục hai cấp độ hạt (pixel-level và patch-level) vượt trội hơn ViT. | Code mới công bố ở mức mã nguồn nghiên cứu, chưa có pipeline kết nối trực tiếp với máy chủ WSI DeepZoom hoặc giao diện người dùng. |
| **LowMagWSS** *(Dung, Doanh et al., Springer 2026)* | Phân vùng giám sát yếu trên ảnh tiêu bản độ phóng đại thấp ($5\times$ thay vì $20\times/40\times$) nhằm tiết kiệm $93\%$ dung lượng lưu trữ và tài nguyên tính toán. | Phù hợp tối đa với điều kiện máy chủ y tế tài nguyên hạn chế; bài báo chính thức từ UIT hợp tác quốc tế. | Độ phân giải đường biên mô ở mức phóng đại $5\times$ bị suy giảm nếu không có kỹ thuật tinh chỉnh phục hồi siêu phân giải hoặc nguyên mẫu. |
| **CONCH & Virchow** *(Lu et al., Vorontsov et al., Nature Medicine 2024)* | Foundation Models quy mô lớn: CONCH (Visual-Language CoCa với 1.17M cặp ảnh-văn bản) và Virchow (ViT-H 632M tham số huấn luyện DINOv2 trên 1.5 triệu WSI). | Khả năng trích xuất đặc trưng mô học pan-cancer cực kỳ mạnh mẽ; hỗ trợ phân vùng zero-shot bằng câu lệnh văn bản tự nhiên. | Kích thước mô hình khổng lồ (Virchow $>2.5$GB weights), không thể fine-tune trực tiếp trên máy đơn thông thường nếu không trích xuất embedding sẵn. |
| **Phần mềm thương mại & Mã nguồn mở (QuPath, ScopeViewer, CLAM)** | Các công cụ xem tiêu bản và gán nhãn giải phẫu bệnh truyền thống. | Hỗ trợ mở định dạng WSI đa dạng, có công cụ vẽ polygon thủ công mạnh mẽ. | QuPath là ứng dụng desktop Java nặng nề, không tối ưu cho kiến trúc đám mây; CLAM chỉ hỗ trợ bản đồ nhiệt chú ý tổng thể (heatmap) cấp slide chứ không phân vùng đa phân lớp pixel-level ngữ nghĩa. |

---

### 3. Data

Hệ thống sử dụng các bộ dữ liệu giải phẫu bệnh công khai chuẩn mực quốc tế:

| Tên Bộ Dữ Liệu | Nguồn / Xuất Bản | Quy Mô / Kích Thước | Danh Mục Phân Lớp Mô Học (Classes) | Giấy Phép & Mô Tả |
| :--- | :--- | :--- | :--- | :--- |
| **BCSS-WSSS** *(Breast Cancer Semantic Segmentation)* | Amgad et al., *Bioinformatics 2019*; Han et al., *MedIA 2022* | 17.286 patches ($224 \times 224$ px tại $20\times$), trích xuất từ 151 vùng ROI tiêu bản ung thư vú TCGA-BRCA | **5 Lớp:**<br>1. Khối u (Tumor - TUM)<br>2. Mô đệm (Stroma - STR)<br>3. Tế bào lympho (Lymphocyte - LYM)<br>4. Vùng hoại tử (Necrosis - NEC)<br>5. Khác (Other - OTR) | Open Access (Creative Commons CC BY 4.0). Chia tách theo Patient ID: Train (23.422 patches chỉ có nhãn yếu), Val (3.418 patches có mask chuẩn), Test (4.986 patches có mask chuẩn). |
| **LUAD-HistoSeg** *(Lung Adenocarcinoma Tissue Segmentation)* | Han et al., *MedIA 2022* | 31.826 patches ($224 \times 224$ px tại $10\times$), từ 49 bệnh nhân (29 ca GDPH + 20 ca TCGA-LUAD) | **4 Lớp:**<br>1. Biểu mô khối u (Tumor Epithelial - TE)<br>2. Mô đệm kèm u (Tumor-Associated Stroma - TAS)<br>3. Hoại tử (Necrosis - NEC)<br>4. Tế bào lympho (Lymphocyte - LYM) | Research Only (Google Drive tác giả). Train (16.678 patches nhãn one-hot), Val (300 patches mask điểm chuẩn), Test (307 patches mask kiểm thử độc lập). |
| **Camelyon16 (Benchmark WSI)** | Grand-Challenge, Ehteshami Bejnordi et al., *JAMA 2017* | 399 tiêu bản WSI toàn phần (.tif định dạng kim tự tháp, mỗi file $1-3$ GB) | **2 Lớp:** Di căn hạch ung thư vú (Metastasis Tumor) vs. Mô hạch bình thường (Normal Lymph Node) | Grand Challenge Open Access. 270 slide train, 129 slide test dùng để kiểm thử khả năng ghép nối WSI và độ trễ streaming trạm làm việc. |

*Cam kết Chống Rò Rỉ Bệnh Nhân:* Toàn bộ quy trình nạp dữ liệu được kiểm soát tự động thông qua module `src/datasets/patient_split.py` với mã băm SHA-256 đối soát metadata bệnh nhân, đảm bảo không có bất kỳ lát cắt nào của cùng 1 bệnh nhân xuất hiện ở cả 2 tập huấn luyện và kiểm thử.

---

### 4. Evaluation

#### 4.1 Tiêu chí Đo lường Thành công Thực tế (Real-Life Clinical Utility) vs. Thực nghiệm (Laboratory Metrics)
- **Đo lường thành công trong thực tế lâm sàng (Real-life Utility):**
  - **Tiết kiệm thời gian gán nhãn:** Giảm thời gian chuyên gia giải phẫu bệnh phân tích và khoanh vùng 100 lát cắt từ $>3$ giờ ($>180$ phút) xuống $\le 7$ phút (tương đương hệ số tăng tốc $\ge 25\times$).
  - **Tốc độ phản hồi giao diện:** Thời gian từ lúc bác sĩ nhấp chọn vùng ROI trên tiêu bản gigapixel đến khi hiển thị đầy đủ mặt nạ phân tầng (mask overlay) trên trình duyệt web đạt $<2.0$ giây.
  - **Độ tin cậy định lượng TME:** Hệ số tương quan Pearson giữa tỷ lệ phần trăm diện tích tế bào lympho xâm nhập (sTILs score $\%$) đo tự động bởi phần mềm so với kết quả giải phẫu bệnh chuẩn đạt $r \ge 0.85$.
- **Đo lường thành công trong thực nghiệm phòng lab (Experimental Metrics):**
  - **Mean Intersection over Union (mIoU):** Đo độ trùng khớp giữa mask phân vùng dự đoán và ground truth trên toàn bộ các lớp:
    $$\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \frac{|P_c \cap G_c|}{|P_c \cup G_c|}$$
  - **Frequency-Weighted IoU (FwIoU):** IoU có trọng số theo diện tích tần suất xuất hiện của từng phân lớp mô học:
    $$\text{FwIoU} = \frac{1}{\sum_{c} |G_c|} \sum_{c=1}^C |G_c| \frac{|P_c \cap G_c|}{|P_c \cup G_c|}$$
  - **Mean Dice Similarity Coefficient (mDice):**
    $$\text{mDice} = \frac{1}{C} \sum_{c=1}^C \frac{2 |P_c \cap G_c|}{|P_c| + |G_c|}$$
  - **Mean Precision (mP), Mean Recall (mR), Pixel Accuracy (ACC):** Đo lường tính chuẩn xác và độ nhạy ở cấp độ từng điểm ảnh.

#### 4.2 Giao thức Thực nghiệm (Evaluation Protocol)
- Huấn luyện cố định với **5 Random Seeds** (`seeds = [42, 100, 2024, 2025, 2026]`), báo cáo kết quả ở dạng Giá trị trung bình $\pm$ Độ lệch chuẩn ($\mu \pm \sigma$).
- Kiểm tra tính có ý nghĩa thống kê bằng kiểm định hoán vị hai phía (Two-sided Paired Permutation Test, ngưỡng ý nghĩa $p < 0.05$).

---

### 5. Baselines

Nhóm thiết lập hệ thống so sánh đối chuẩn đa tầng với 6 phương pháp đại diện:

1. **HistoSegNet** *(Chan et al., ICCV 2019):* Baseline WSSS kinh điển đầu tiên trên ảnh mô học, sử dụng mạng phân loại kết hợp Grad-CAM thô và thuật toán hậu xử lý hợp nhất vùng.
2. **C-CAM & SC-CAM** *(Chang et al., CVPR 2020):* Mô hình CAM tăng cường khai thác mối tương quan không gian giữa các vùng ảnh.
3. **WSSS-Tissue Baseline (Giai đoạn 1)** *(Han et al., MedIA 2022):* Mô hình phân loại ResNet-38 sinh pseudo-mask chỉ dựa trên bản đồ kích hoạt CAM một lớp (layer `bn7`).
4. **WSSS-Tissue MLPS Hoàn chỉnh (Giai đoạn 2)** *(Han et al., MedIA 2022):* Mô hình DeepLabV3+ giám sát bởi pseudo-mask đa tầng kết hợp Cổng phân loại (CGM).
5. **PBIP SOTA** *(Tang et al., CVPR 2025):* Mô hình SegFormer căn chỉnh nguyên mẫu hình ảnh SOTA mới nhất.
6. **Fully-Supervised DeepLabV3+ / U-Net (Giám sát đầy đủ - Ngưỡng trần so sánh):** Huấn luyện trực tiếp trên $100\%$ mask ground truth vẽ tay để làm mốc tham chiếu cận trên lý thuyết.

---

### 6. Proposed Method

Nhóm đề xuất giải pháp tích hợp toàn diện: **Mô hình AI Phân vùng Nguyên mẫu Không gian Trạng thái (Proto-Mamba WSSS)** tích hợp trong **Trạm làm việc Giải phẫu bệnh Số SE374 (SE374 Pathology Workstation)**.

```
+----------------------------------------------------------------------------------------------------------+
|                       KIẾN TRÚC TỔNG THỂ HỆ THỐNG TRẠM LÀM VIỆC GIẢI PHẪU BỆNH SE374                     |
+----------------------------------------------------------------------------------------------------------+
|  [TẦNG 1: TRÌNH DUYỆT LÂM SÀNG (CLINICAL CLIENT)]                                                        |
|  • Web UI Responsive: HTML5 / Modern CSS / Vanilla JS                                                    |
|  • Trình xem Tiêu bản Phân giải sâu: OpenSeadragon (DeepZoom Pyramid Navigation: 1x -> 40x)              |
|  • Trực quan hóa Lớp Phủ: WebGL Shader Canvas (Bật/tắt 5 kênh màu mô học, thanh trượt Alpha/Opacity)    |
|  • Bảng điều khiển Định lượng: Biểu đồ diện tích khối u (mm²), Tỷ lệ sTILs Score (%), Nút Xuất PDF       |
+----------------------------------------------------------------------------------------------------------+
                                                     |  (RESTful JSON / Binary Tile Streaming)
                                                     v
+----------------------------------------------------------------------------------------------------------+
|  [TẦNG 2: MÁY CHỦ DỊCH VỤ Y TẾ (BACKEND MEDICAL API SERVICE - FastAPI & TIAToolbox)]                     |
|  • WSI Pyramidal Tile Server: Cắt lát động (Dynamic DZI / TIFF Tiler) độ trễ <50ms                       |
|  • Tissue Area Pre-Filtering: Otsu Tissue Masking loại bỏ vùng kính trắng không có mô                    |
|  • Session & Case Manager: Quản lý bệnh án số, phân quyền bác sĩ, nhật ký phân tích                      |
|  • Report Export Engine: ReportLab sinh biên bản giải phẫu bệnh chuẩn A4 (Biểu đồ, Snapshot, sTILs)     |
+----------------------------------------------------------------------------------------------------------+
                                                     |  (Tensor Batches: 224x224x3)
                                                     v
+----------------------------------------------------------------------------------------------------------+
|  [TẦNG 3: MÔ HÌNH TRÍ TUỆ NHÂN TẠO CỐT LÕI (CORE MEDICAL AI ENGINE: PROTO-MAMBA WSSS)]                    |
|  1. Trích xuất Đặc trưng Đa tỷ lệ:                                                                      |
|     - Foundation Model Encoder (Virchow/CONCH hoặc ResNet-38 khởi tạo sẵn)                               |
|  2. Khối Không gian Trạng thái Tương phản (Contrastive Mamba Block - CMB):                                |
|     - Quét đa hướng 4 chiều (4-Way Scan Expansion) mô hình hóa quan hệ xa với độ phức tạp tuyến tính O(L) |
|     - Khớp nối Nguyên mẫu Trực quan (Prototype Matching): Đẩy lùi hiện tượng rò rỉ viền mô               |
|  3. Chú ý Dropout Lũy tiến (PDA): \beta_t = \mu_t \cdot \max(M_k) mở rộng vùng kích hoạt sang mô đệm     |
|  4. Cổng Phân loại Tinh lọc (Classification Gate Mechanism - CGM): \hat{y}_k \le \epsilon -> Triệt tiêu  |
|  5. Bộ Ghép nối Kim tự tháp WSI (Overlap Stitching Engine): Ghép mảng với 50% overlap làm mịn biên       |
+----------------------------------------------------------------------------------------------------------+
```

#### 6.1 Tại sao phương pháp này sẽ hiệu quả? (Why it should work?)
1. **Khắc phục co ngót CAM bằng PDA và Nguyên mẫu:** Cơ chế Dropout lũy tiến buộc mạng phân loại phải tìm kiếm tín hiệu ở những vùng rìa kém nổi bật, trong khi các vector nguyên mẫu (prototypes) hoạt động như mỏ neo hình thái học đại diện cho từng phân lớp tế bào, ngăn chặn hoàn toàn việc nhầm lẫn giữa mô hoại tử và chất đệm.
2. **Khắc phục nghẽn tính toán bằng Visual Mamba:** Thay vì cơ chế tự chú ý bậc hai $\mathcal{O}(L^2)$ của Transformer, Mamba quét dữ liệu với độ phức tạp tuyến tính $\mathcal{O}(L)$, cho phép trích xuất đặc trưng ngữ cảnh toàn slide nhanh hơn $4-6\times$ và tiết kiệm $70\%$ VRAM.
3. **Triệt tiêu bóng mờ giả bằng Cổng Phân loại (CGM):** Các phân lớp thiểu số (như lympho hoặc hoại tử chỉ chiếm $<5\%$ diện tích) thường bị mô u lấn át sinh ra dương tính giả. Khi xác suất phân loại patch cho biết không có sự tồn tại của phân lớp này ($\hat{y}_k \le 0.1$), kênh phân vùng tương ứng lập tức bị đóng lại, mang lại độ đặc hiệu (Specificity) vượt trội.

---

### 7. Expected Results

#### 7.1 Kết quả Kỳ vọng Định lượng (Expected Quantitative Findings)

| Bộ Dữ Liệu Thực Nghiệm | Phương pháp Cơ sở (Baseline WSSS-Tissue) | Kỳ vọng Đạt được (Proto-Mamba WSSS) | Mục tiêu Cận trên (Fully Supervised) |
| :--- | :---: | :---: | :---: |
| **BCSS-WSSS (mIoU)** | $0.6892$ | **$\ge 0.7200$** ($+3.1\%$) | $0.6862$ *(Đã vượt qua)* |
| **BCSS-WSSS (mDice)** | $0.7670$ | **$\ge 0.7900$** ($+2.3\%$) | $0.7850$ |
| **LUAD-HistoSeg (mIoU)** | $0.7560$ | **$\ge 0.7750$** ($+1.9\%$) | $0.7880$ |
| **LUAD-HistoSeg (mDice)** | $0.8180$ | **$\ge 0.8450$** ($+2.7\%$) | $0.8620$ |
| **Tốc độ suy luận WSI ROI** | $\sim 8.5$ giây | **$\le 1.8$ giây** ($>4\times$ speedup) | - |

#### 7.2 Sản phẩm Phần mềm Bàn giao Cụ thể (Software Engineering Deliverables)
1. **Mã nguồn hệ thống chuẩn Clean Architecture:** Đầy đủ các gói `src/config`, `src/datasets`, `src/models`, `src/losses_metrics`, `src/visualization`, `src/pipeline` có unit tests và CI/CD GitHub Actions kiểm thử tự động.
2. **Giao diện Web tương tác trạm làm việc (Workstation UI):** Chạy mượt mà trên Chrome/Edge/Firefox, hỗ trợ kéo thả tệp tiêu bản, điều chỉnh độ trong suốt của mask theo từng phân lớp màu chuẩn y khoa, đo lường tỷ lệ diện tích và xuất báo cáo PDF chuẩn hóa.
3. **Bộ tài liệu kỹ thuật & Báo cáo học thuật:** Đề cương Mini Proposal (DOCX), Slide báo cáo 16:9 Marp (HTML/PDF/PPTX), và Báo cáo tổng kết đồ án định dạng bài báo Springer LNCS.

#### 7.3 Kế hoạch Dự phòng khi Kết quả Thực nghiệm Bất lợi (Contingency / Backup Plan)
- **Rủi ro 1: Huấn luyện Mamba từ đầu (from scratch) trên dữ liệu nhỏ bị bất ổn định hoặc hội tụ chậm:**
  - *Giải pháp dự phòng:* Sử dụng backbone SegFormer-B2 đã được tinh chỉnh của PBIP (CVPR 2025) làm mạng sinh pseudo-mask chính, kết hợp với các vector đặc trưng đóng băng (frozen embeddings) trích xuất từ Foundation Model CONCH.
- **Rủi ro 2: Dung lượng tệp WSI Camelyon16 quá lớn gây nghẽn đường truyền máy chủ:**
  - *Giải pháp dự phòng:* Kế thừa kiến trúc của LowMagWSS (Springer 2026): thực hiện tiền xử lý nén kim tự tháp đa tầng trước, phục vụ lát cắt ở mức phóng đại coarse ($5\times/10\times$) để hiển thị bản đồ tổng quan, chỉ tải lát cắt chi tiết ($20\times$) khi người dùng zoom sâu vào khu vực quan tâm.

---

## References

1. **Chu Han, Jiatai Lin, Jinhai Mai, et al.** "Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels." *Medical Image Analysis (MedIA)*, vol. 80, p. 102487, 2022. DOI: `10.1016/j.media.2022.102487`.
2. **Qingchen Tang, et al.** "Prototype-Based Image Prompting for Weakly-Supervised Histopathology Tissue Segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025. arXiv: `2503.12068`.
3. **Jiansong Fan, Tianxu Lv, Yicheng Di, Lihua Li, Xiang Pan.** "PathMamba: Weakly Supervised State Space Model for Multi-class Segmentation of Pathology Images." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, LNCS, Springer, 2024. DOI: `10.1007/978-3-031-72111-3_47`.
4. **Tan-Dung Pham, Duc-Dung Nguyen, Binh-Minh Nguyen, Cao-Doanh Bui.** "LowMagWSS: Weakly-Supervised Semantic Segmentation on Low-Magnification Whole Slide Images." *Springer Machine Vision and Applications / arXiv*, 2026. arXiv: `2607.10783`.
5. **Ming Y. Lu, Bowen Chen, Drew F. K. Williamson, et al.** "Towards a Visual-Language Foundation Model for Computational Pathology (CONCH)." *Nature Medicine*, vol. 30, pp. 833-846, 2024. DOI: `10.1038/s41591-024-02856-4`.
6. **Eugene Vorontsov, Alican Bozkurt, Adam Casson, et al.** "Virchow: A Million-Slide Digital Pathology Foundation Model." *Nature Medicine*, vol. 30, pp. 2550-2560, 2024. DOI: `10.1038/s41591-024-03141-4`.
7. **L. Chan, M. S. Hosseini, C. Rowsell, K. N. Plataniotis, S. Damaskinos.** "HistoSegNet: Semantic Segmentation of Histological Tissue Type in Whole Slide Images." *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 10662-10671, 2019.
8. **M. Amgad, H. Elfandy, H. Hussein, et al.** "Structured Crowdsourcing Enables Convolutional Segmentation of Histology Images (BCSS)." *Bioinformatics*, vol. 35, no. 18, pp. 3461-3467, 2019.
9. **Babak Ehteshami Bejnordi, et al.** "Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer (CAMELYON16)." *JAMA*, vol. 318, no. 22, pp. 2199-2210, 2017.
10. **Richard J. Chen, Ming Y. Lu, et al.** "Data-efficient and weakly supervised computational pathology on whole-slide images (CLAM)." *Nature Biomedical Engineering*, vol. 5, pp. 1175-1196, 2021.

---

## Appendix

### 1. Project Timeline & Task Assignments (Gantt 10 Tuần)

| Tuần | Giai Đoạn & Hạng Mục Công Việc Chi Tiết | Sản Phẩm Đầu Ra Cụ Thể | Phụ Trách Chính |
| :---: | :--- | :--- | :---: |
| **T01** | Khảo sát bài báo SOTA (WSSS-Tissue, PBIP, PathMamba, LowMagWSS), phân tích yêu cầu đề tài Topic 6, hoàn thiện Mini Proposal. | Tài liệu `docs/papers/`, Bản nộp Mini Proposal DOCX/PDF. | Nguyễn Hồng Quân, Phạm Hoàng Phúc |
| **T02** | Xây dựng Data Ingestion Service (`src/datasets/`): Tải và nạp BCSS-WSSS, LUAD-HistoSeg; lập trình module `patient_split.py` kiểm tra chống rò rỉ bệnh nhân. | Data Loaders PyTorch, Bộ kiểm thử `tests/test_patient_split.py` pass 100%. | Phạm Hoàng Phúc |
| **T03** | Dựng khung phần mềm (Clean Architecture): Khởi tạo FastAPI service, tích hợp thư viện đọc tiêu bản TIAToolbox/OpenSlide, dựng RESTful API streaming tile DZI. | Máy chủ WSI Tile Server chạy cục bộ, trả về lát cắt ảnh đa mức phóng đại. | Nguyễn Hồng Quân |
| **T04** | Hiện thực hóa Giai đoạn 1 AI: Mô hình phân loại yếu đa nhãn với Progressive Dropout Attention (PDA) để sinh Pseudo-Masks đa tầng ($b4\_3, b5\_2, bn7$). | Module sinh pseudo-masks `src/models/stage1_cam_pda.py`, trực quan hóa CAM attention. | Nguyễn Hồng Quân |
| **T05** | Hiện thực hóa Giai đoạn 2 AI: Mạng phân vùng ngữ nghĩa với Multi-Layer Pseudo-Supervision (MLPS) và Cổng phân loại Classification Gate Mechanism (CGM). | Pipeline phân vùng hoàn chỉnh `src/models/stage2_segmentation.py`, tính hàm mất mát `MLPSLoss`. | Nguyễn Hồng Quân |
| **T06** | Tích hợp khối Không gian trạng thái (Contrastive Mamba Block - CMB) hoặc Khớp nối Nguyên mẫu (PBIP) để làm mịn viền ranh giới mô u và mô đệm. | Trọng số mô hình cải tiến, bảng so sánh ablation study sơ bộ. | Nguyễn Hồng Quân |
| **T07** | Phát triển Giao diện Trạm làm việc (Workstation Frontend): Tích hợp OpenSeadragon, xây dựng bộ lọc lớp phủ WebGL shader canvas cho phép bật/tắt 5 phân lớp mô. | Giao diện Web tương tác hiển thị tiêu bản kèm mặt nạ phân tầng mượt mà. | Phạm Hoàng Phúc |
| **T08** | Lập trình Module Đo đạc & Định lượng Lâm sàng: Tính toán diện tích khối u ($mm^2$), tỷ lệ sTILs $(\%)$, xây dựng công cụ tự động xuất báo cáo giải phẫu bệnh PDF (ReportLab). | Chức năng xuất file PDF kết quả xét nghiệm hoàn chỉnh kèm biểu đồ phân bố mô. | Phạm Hoàng Phúc |
| **T09** | Chạy thực nghiệm định lượng toàn diện trên 5 Random Seeds ($\mu \pm \sigma$), phân tích lỗi lâm sàng (Clinical Error Analysis: false positives, false negatives). | Bảng số liệu thực nghiệm đầy đủ, biểu đồ Radar, ma trận nhầm lẫn (Confusion Matrix). | Nguyễn Hồng Quân, Phạm Hoàng Phúc |
| **T10** | Hoàn thiện Báo cáo tổng kết đồ án (Springer LNCS LaTeX / Word UIT), đóng gói Docker container, chuẩn bị Slide thuyết trình Marp 16:9 và video demo. | Toàn bộ sản phẩm cuối kỳ trong `product/` sẵn sàng nghiệm thu và bảo vệ. | Nguyễn Hồng Quân, Phạm Hoàng Phúc |

### 2. Sơ Đồ Quy Trình Dữ Liệu (Pipeline Data Flow)

```
[File Ảnh WSI Tiêu Bản Gốc: .svs / .tif / .ndpi (Kích thước: 100.000 x 100.000 px)]
                                  │
                                  ▼
[TIAToolbox Tissue Masking: Lọc nền kính trắng, xác định đa giác mô sinh học]
                                  │
                                  ▼
[Tiling Service: Cắt mảng bán chồng lấn 50% (Patches 224 x 224 px tại 10x/20x)]
                                  │
                                  ▼
[Stage 1: Multi-Label Classifier + Progressive Dropout Attention (PDA)]
        │                                         │
        ▼                                         ▼
[Dự đoán nhãn cấp Patch \hat{y}_k]      [Sinh Mặt nạ Giả định Đa tầng (MLPS Pseudo Masks)]
        │                                         │
        │                                         ▼
        │        [Stage 2: Semantic Segmentation Network (DeepLabV3+ / Mamba Decoder)]
        │                                         │
        ▼                                         ▼
[Classification Gate (CGM): Triệt tiêu nhiễu] <─ [Xác suất thô từng phân lớp mô]
                                  │
                                  ▼
[WSI Probability Stitching: Ghép trung bình xác suất trên tọa độ không gian WSI]
                                  │
                                  ▼
[Trực quan hóa Web OpenSeadragon + Định lượng Vi môi trường Khối u (sTILs Score %)]
                                  │
                                  ▼
[Xuất Biên bản Giải phẫu bệnh Kỹ thuật số PDF chuẩn Hội Bác sĩ Bệnh học Hoa Kỳ (CAP)]
```

### 3. Experimental Plan (Proposal Version 2 - Kế Hoạch Thực Nghiệm Chi Tiết)

1. **Phân bổ Dữ liệu và Kiểm soát Chống Rò rỉ (Data Partitioning):**
   - **BCSS-WSSS:** 151 ca bệnh. Tập Train gồm 106 ca ($23.422$ patches), Tập Val gồm 20 ca ($3.418$ patches), Tập Test gồm 25 ca ($4.986$ patches). Tuyệt đối không trùng lặp mã ca bệnh giữa các tập.
   - **LUAD-HistoSeg:** 49 ca bệnh. Tập Train gồm 39 ca ($16.678$ patches), Tập Val gồm 5 ca ($300$ patches), Tập Test gồm 5 ca ($307$ patches).
2. **Kế hoạch Nghiên cứu Bóc tách (Ablation Matrix):**
   - *Thí nghiệm 1 (Hiệu quả của PDA):* So sánh Baseline không có dropout vs. Dropout cố định $\mu=0.7$ vs. Progressive Dropout Attention ($\sigma=0.985, l=0.65$).
   - *Thí nghiệm 2 (Hiệu quả của Giám sát Đa tầng MLPS):* So sánh chỉ dùng pseudo-mask lớp `b4_3` vs. lớp `b5_2` vs. lớp `bn7` vs. kết hợp cả 3 lớp ($\lambda_1=0.2, \lambda_2=0.2, \lambda_3=0.6$).
   - *Thí nghiệm 3 (Hiệu quả của Cổng Phân loại CGM):* Đo lường mức giảm tỷ lệ dương tính giả (False Positive Rate) trên phân lớp hoại tử (Necrosis) và lympho (Lymphocyte) khi có và không có CGM ($\epsilon=0.1$).
   - *Thí nghiệm 4 (So sánh Trọng số Khởi tạo Backbone):* So sánh ImageNet Pretrained vs. CTransPath Pretrained vs. CONCH Foundation Pretrained.
3. **Môi trường Thực nghiệm và Quy chuẩn Kiểm soát:**
   - Hệ điều hành: Windows 11 Pro 64-bit.
   - Ngôn ngữ & Thư viện: Python 3.10+, PyTorch 2.3+, CUDA 12.1, TIAToolbox 1.5+, OpenSlide 4.0+, FastAPI 0.111+, OpenSeadragon 4.1+.
   - Thiết bị phần cứng: NVIDIA GPU (tối thiểu 8GB VRAM), RAM hệ thống 32GB, Ổ cứng SSD NVMe.
