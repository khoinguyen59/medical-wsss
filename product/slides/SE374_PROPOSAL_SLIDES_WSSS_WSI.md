---
marp: true
theme: default
paginate: true
size: 16:9
header: "SE374.R11 - Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa | Mini Proposal"
footer: "Trường ĐH Công nghệ Thông tin (UIT) - ĐHQG-HCM | [Nhóm X]"
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 22px;
    padding: 35px 50px;
    background-color: #f8fafc;
    color: #1e293b;
  }
  h1 {
    color: #0f172a;
    font-size: 34px;
    margin-bottom: 15px;
    border-bottom: 3px solid #0284c7;
    padding-bottom: 8px;
  }
  h2 {
    color: #0369a1;
    font-size: 26px;
    margin-bottom: 12px;
  }
  table {
    font-size: 18px;
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
  }
  th {
    background-color: #0284c7;
    color: white;
    padding: 8px 12px;
    text-align: left;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid #cbd5e1;
  }
  .highlight-box {
    background-color: #e0f2fe;
    border-left: 5px solid #0284c7;
    padding: 12px 18px;
    border-radius: 4px;
    margin-top: 10px;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 16px;
  }
  .badge-blue { background-color: #bae6fd; color: #0369a1; }
  .badge-emerald { background-color: #a7f3d0; color: #065f46; }
---

<!-- _class: lead -->
<!-- _paginate: false -->
# ĐỀ CƯƠNG ĐỒ ÁN MÔN HỌC (MINI PROPOSAL)
## Hệ Thống Phân Vùng Ngữ Nghĩa Giám Sát Yếu Trên Ảnh Tiêu Bản Mô Bệnh Học Toàn Phần
### Weakly-Supervised Semantic Segmentation for Whole Slide Images (WSI)

- **Môn học:** SE374.R11 - Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa (2026-2027)
- **Khoa:** Khoa Kỹ thuật Phần mềm, Trường ĐH Công nghệ Thông tin - ĐHQG-HCM (UIT)
- **Giảng viên hướng dẫn:** TS. Bùi Cao Doanh
- **Nhóm thực hiện:** [Nhóm X - Điền số nhóm của bạn]
- **Thành viên:**
  1. [Họ và tên Sinh viên 1] - MSSV: [MSSV 1]
  2. [Họ và tên Sinh viên 2] - MSSV: [MSSV 2]

---

# 1. Đặt Vấn Đề Lâm Sàng & Thách Thức Nghiên Cứu

<div class="grid-2">
<div>

### Thách Thức Của Giải Phẫu Bệnh Số
- **Kích thước Gigapixel:** Một tiêu bản WSI có độ phân giải lên tới $100.000 \times 100.000$ pixels ($>2$GB/file).
- **Chi phí gán nhãn cực lớn:** Bác sĩ giải phẫu bệnh mất $>3.5$ giờ để khoanh vùng thủ công từng vi trường một ca bệnh.
- **Tính bất đồng nhất mô học:** Khối u, mô đệm và lympho xâm nhập đan xen phức tạp.

</div>
<div>

### Hạn Chế Của Các Giải Pháp Hiện Nay
- **CAM truyền thống (Class Activation Maps):** Co ngót vùng kích hoạt (shrinkage), chỉ tập trung vào nhân tế bào dị dạng, bỏ sót viền u.
- **Vision Transformer (ViT):** Độ phức tạp bậc hai $\mathcal{O}(L^2)$ làm nghẽn VRAM và trễ $>10$s trên máy trạm.
- **Khoảng trống phần mềm:** Chưa có trạm làm việc tích hợp DeepZoom streaming mượt mà cho bác sĩ lâm sàng.

</div>
</div>

<div class="highlight-box">
<b>Mục tiêu cốt lõi:</b> Phân vùng chính xác từng pixel mô bệnh học chỉ với nhãn yếu cấp mảng (patch-level labels), kết hợp phần mềm trạm làm việc tương tác thời gian thực.
</div>

---

# 2. Mục Tiêu Nghiên Cứu & Đóng Góp Dự Kiến

### Hai Trụ Cột Đồ Án Môn SE374 (Medical AI + Software Engineering)

<div class="grid-2">
<div>

#### Trụ Cột 1: Nghiên Cứu Lõi AI Y Khoa
- **Mô hình hóa WSSS:** Khắc phục hiện tượng co ngót CAM, mở rộng ranh giới phân vùng đa lớp mô.
- **Độ chính xác vượt trội:** Đạt mIoU $\ge 0.76$ trên BCSS-WSSS và $\ge 0.75$ trên LUAD-HistoSeg.
- **Độ tin cậy học thuật:** Đánh giá công bằng trên 5 random seeds, tuân thủ nguyên tắc Zero-Mockup.

</div>
<div>

#### Trụ Cột 2: Kỹ Thuật Phần Mềm Lâm Sàng
- **Web Workstation:** Trình duyệt tiêu bản phân giải sâu OpenSeadragon (tốc độ $>45$ FPS).
- **Độ trễ suy luận:** Phản hồi mặt nạ lớp phủ (mask overlay) $<2$ giây khi bác sĩ chọn vùng ROI.
- **Định lượng TME & Báo cáo:** Tự động tính sTILs score (%) và xuất biên bản PDF chuẩn CAP.

</div>
</div>

<div class="highlight-box">
<b>Ý nghĩa thực tiễn:</b> Giảm thời gian đo đạc tiêu bản từ 180 phút xuống dưới 6 phút/ca bệnh (>30x speedup), cung cấp giải pháp mở hỗ trợ lâm sàng và giảng dạy tại UIT.
</div>

---

# 3. Tổng Quan Các Công Trình Tiêu Biểu (Related Work)

| Mô Hình / Công Trình | Xuất Bản | Đặc Điểm Kỹ Thuật Chính | Hạn Chế Cốt Lõi Cần Cải Tiến |
| :--- | :---: | :--- | :--- |
| **WSSS-Tissue / MLPS** | MedIA 2022 | Progressive Dropout Attention (PDA) + Giám sát giả đa tầng (MLPS) | Backbone CNN cũ (ResNet-38); phụ thuộc vào ngưỡng cố định thủ công |
| **PBIP (CVPR 2025)** | CVPR 2025 | Prototype-Based Image Prompting với SegFormer | Bộ nhớ prototype lớn; thiếu module streaming WSI hoàn chỉnh |
| **PathMamba** | MICCAI 2024 | State Space Model (Visual Mamba) tuyến tính $\mathcal{O}(L)$ | Mới ở mức code thí nghiệm thô, chưa có giao diện tương tác |
| **LowMagWSS (UIT Lab)** | Springer 2026 | Phân vùng WSI ở độ phóng đại thấp 5x/10x tiết kiệm 93% lưu trữ | Độ phân giải viền bị suy giảm nếu thiếu cơ chế phục hồi chi tiết |
| **CONCH & Virchow** | Nature Med 2024 | Foundation Models thị giác - ngôn ngữ quy mô lớn | Mô hình khổng lồ (>2.5GB weights), khó huấn luyện trực tiếp |

---

# 4. Phương Pháp Đề Xuất: Mô Hình AI Proto-Mamba WSSS

<div class="grid-2">
<div>

### Giai Đoạn 1: Phân Loại Đa Nhãn + PDA
- **Contrastive Mamba Block (CMB):** Khai thác ngữ cảnh xa với độ phức tạp tuyến tính $\mathcal{O}(L)$.
- **Progressive Dropout Attention:** $\beta_t = \mu_t \cdot \max(M_k)$ $\to$ ép mạng mở rộng vùng kích hoạt sang ranh giới mờ.
- **Pseudo-Masks Đa Tầng:** Kết hợp lớp `b4_3` (chi tiết), `b5_2` (cân bằng) và `bn7` (ngữ nghĩa).

</div>
<div>

### Giai Đoạn 2: Phân Vùng + Cổng CGM
- **Prototype Matching:** Đẩy lùi hiện tượng rò rỉ ranh giới giữa các phân lớp mô học.
- **Classification Gate (CGM):** Khi $\hat{y}_k \le 0.1 \to$ lập tức đóng kênh, triệt tiêu hoàn toàn dương tính giả.
- **Overlap Stitching Engine:** Ghép trung bình xác suất trên tọa độ slide toàn phần WSI.

</div>
</div>

<div class="highlight-box">
<b>Luồng Xử Lý Toàn Vẹn:</b> Ảnh WSI Gigapixel (100k &times; 100k) &rarr; TIAToolbox Tissue Masking &rarr; Patches 224&times;224 (Overlap 50%) &rarr; PDA CAM Generation &rarr; DeepLabV3+ CGM &rarr; Mặt Nạ Phân Vùng Toàn Slide.
</div>

---

# 5. Kiến Trúc Phần Mềm Trạm Làm Việc (Clean Architecture)

<div class="grid-2">
<div>

### Tầng 1: Client Web Trạm Làm Việc
- **Giao diện hiện đại:** HTML5, Canvas WebGL, Responsive CSS.
- **DeepZoom Navigation:** OpenSeadragon duyệt mức phóng đại $1\times \to 40\times$.
- **Lớp phủ mặt nạ:** Bật/tắt 5 kênh màu mô học độc lập, chỉnh độ trong suốt.
- **Bảng điều khiển lâm sàng:** Biểu đồ diện tích, sTILs score (%), nút xuất báo cáo.

</div>
<div>

### Tầng 2 & 3: Medical Backend & Engine
- **FastAPI Tile Server:** Cắt lát động DZI từ tệp SVS/NDPI với độ trễ $<50$ms.
- **Tissue Filtering:** Tiền lọc kính trắng tự động bằng Otsu thresholding.
- **AI Inference Engine:** Nạp mảng tensor song song, suy luận không tràn RAM.
- **Report Engine:** Xuất biên bản PDF hoàn chỉnh (chuẩn định dạng CAP).

</div>
</div>

<div class="highlight-box">
<b>Độ độc lập module:</b> Tách rời hoàn toàn giữa giao diện Web, dịch vụ nạp ảnh và mô hình suy luận theo tiêu chuẩn Clean Architecture.
</div>

---

# 6. Dữ Liệu Nghiên Cứu & Giao Thức Chống Rò Rỉ Bệnh Nhân

### Ba Bộ Dữ Liệu Tiêu Chuẩn Quốc Tế

| Tên Bộ Dữ Liệu | Nguồn Xuất Bản | Quy Mô | Danh Mục Phân Lớp Mô Học | Phân Chia Tập Dữ Liệu |
| :--- | :---: | :---: | :--- | :--- |
| **BCSS-WSSS** | MedIA 2022 | 17.286 patches | 5 Lớp: Khối u, Mô đệm, Lympho, Hoại tử, Khác | 106 ca Train / 20 ca Val / 25 ca Test |
| **LUAD-HistoSeg** | MedIA 2022 | 31.826 patches | 4 Lớp: Biểu mô u, Mô đệm u, Hoại tử, Lympho | 39 ca Train / 5 ca Val / 5 ca Test |
| **Camelyon16** | JAMA 2017 | 399 slide WSI | 2 Lớp: Di căn hạch ung thư vú vs. Bình thường | Kiểm thử ghép nối WSI & Streaming |

<div class="highlight-box">
<b>Quy tắc Bất biến Chống Rò Rỉ Bệnh Nhân (Patient-Level Anti-Leakage):</b><br>
Phân chia tập <code>train / val / test</code> <b>hoàn toàn theo Patient ID</b>. Tuyệt đối không chia ngẫu nhiên ở cấp độ patch để tránh rò rỉ đặc trưng mô học của cùng bệnh nhân giữa các tập. Kiểm tra tự động bằng mã băm SHA-256.
</div>

---

# 7. Kế Hoạch Đánh Giá & Cam Kết Zero-Mockup

<div class="grid-2">
<div>

### Thước Đo Thực Nghiệm (Lab Metrics)
- **Mean IoU (mIoU):** Trọng tâm đánh giá độ phủ mặt nạ trên tất cả các lớp.
- **Mean Dice (mDice):** Đo lường tính tương đồng hình thái học viền khối u.
- **FwIoU & ACC:** Đo độ chính xác tổng thể có tính trọng số theo diện tích.
- **Giao thức 5 Seeds:** Chạy cố định 5 hạt ngẫu nhiên (`42, 100, 2024, 2025, 2026`), báo cáo $\mu \pm \sigma$.

</div>
<div>

### Chỉ Số Lâm Sàng & Hiệu Năng
- **Tiết kiệm thời gian:** Đọc 100 vi trường từ 180 phút xuống $\le 7$ phút ($\ge 25\times$).
- **Độ trễ giao diện:** Hiển thị lớp phủ mặt nạ trên web $<2.0$ giây.
- **Độ tin cậy sTILs:** Hệ số tương quan Pearson $r \ge 0.85$ so với chuẩn lâm sàng.
- **Cam kết Zero-Mockup:** 100% kết quả đo đạc từ code thực nghiệm thật.

</div>
</div>

---

# 8. Kế Hoạch Triển Khai 10 Tuần (Gantt Chart)

| Tuần | Nội Dung Triển Khai Chi Tiết | Sản Phẩm Đầu Ra Cụ Thể | Phụ Trách |
| :---: | :--- | :--- | :---: |
| **T01** | Khảo sát SOTA, hoàn thiện Mini Proposal & Slide bảo vệ | Mini Proposal DOCX, Slide Marp | [Cả nhóm] |
| **T02** | Dựng Data Ingestion Service, kiểm tra chống rò rỉ bệnh nhân | PyTorch DataLoaders, Unit Tests | [Sinh viên 2] |
| **T03** | Khởi tạo FastAPI Backend & DeepZoom WSI Tile Streaming Server | API cắt lát DZI động độ trễ <50ms | [Sinh viên 1] |
| **T04** | Hiện thực hóa Giai đoạn 1 AI: Classifier + Progressive Dropout | Module sinh Pseudo-Masks đa tầng | [Sinh viên 1] |
| **T05** | Hiện thực hóa Giai đoạn 2 AI: Mạng phân vùng đa tầng MLPS + CGM | Pipeline phân vùng hoàn chỉnh | [Sinh viên 1] |
| **T06** | Tích hợp khối Không gian trạng thái (CMB) & Khớp nối Nguyên mẫu | Trọng số mô hình cải tiến | [Sinh viên 1] |
| **T07** | Phát triển Web Frontend OpenSeadragon + Bộ lọc WebGL Canvas | Giao diện tương tác xem tiêu bản | [Sinh viên 2] |
| **T08** | Lập trình module định lượng sTILs score (%) & xuất báo cáo PDF | Chức năng xuất biên bản CAP PDF | [Sinh viên 2] |
| **T09** | Chạy thực nghiệm định lượng trên 5 Random Seeds, phân tích lỗi | Bảng số liệu thực nghiệm đầy đủ | [Cả nhóm] |
| **T10** | Hoàn thiện Báo cáo tổng kết đồ án, đóng gói Docker & bảo vệ | Báo cáo Springer LNCS, Video demo | [Cả nhóm] |

---

# 9. Sản Phẩm Bàn Giao Cuối Kỳ (Deliverables)

<div class="grid-2">
<div>

### 1. Mã Nguồn Phần Mềm Hoàn Chỉnh
- **Clean Architecture:** Gói `src/` phân tầng rõ ràng (config, datasets, models, visualization, pipeline).
- **Trạm làm việc Web:** Giao diện tương tác mượt mà duyệt ảnh WSI gigapixel kèm lớp phủ mask.
- **API Dịch vụ:** FastAPI endpoints cho streaming lát cắt và suy luận phân vùng.
- **Tự động hóa CI/CD:** GitHub Actions kiểm thử syntax, patient leakage và anti-mockup.

</div>
<div>

### 2. Hồ Sơ Báo Cáo Khoa Học & Lâm Sàng
- **Đề cương Mini Proposal:** Định dạng Word DOCX chuẩn mẫu UIT và slide 16:9.
- **Báo cáo kỹ thuật cuối kỳ:** Định dạng Springer LNCS LaTeX / Word UIT chi tiết.
- **Biên bản kiểm thử lâm sàng:** Báo cáo PDF mẫu sinh tự động bởi phần mềm.
- **Tài liệu hướng dẫn:** Hướng dẫn cài đặt, vận hành và tái lập thực nghiệm.

</div>
</div>

<div class="highlight-box">
<b>Cam kết chất lượng:</b> Sẵn sàng nghiệm thu với đầy đủ mã nguồn có thể chạy được, bộ kiểm thử tự động và tài liệu hoàn chỉnh.
</div>

---

<!-- _class: lead -->
# XIN CHÂN THÀNH CẢM ƠN THẦY VÀ CÁC BẠN!
## Q&A - Hỏi Đáp & Nhận Đóng Góp Ý Kiến Cho Đồ Án

- **Đề tài:** Weakly-Supervised Semantic Segmentation for Whole Slide Images (WSI)
- **Môn học:** SE374.R11 - Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa
- **Khoa Kỹ thuật Phần mềm** - Trường Đại học Công nghệ Thông tin (UIT)
- **Email liên hệ nhóm:** [Email liên hệ đại diện nhóm]
