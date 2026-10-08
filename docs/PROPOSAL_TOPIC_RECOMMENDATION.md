# BÁO CÁO PHÂN TÍCH ĐỀ TÀI & ĐỀ XUẤT CHIẾN LƯỢC
## Môn học: Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa (SE374.R11)
### Khoa Kỹ thuật Phần mềm - Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM

> **Nguồn đối soát:** `SE374.R11 - Đồ án cuối kỳ.xlsx` & `topic.md`  
> **Giảng viên / Phòng nghiên cứu phụ trách:** TS. Bùi Cao Doanh (`doanhbc@uit.edu.vn`) & PGS. TS. Nguyễn Khang (`khangnttm@uit.edu.vn`)  
> **Đặc thù môn học:** Kết hợp giữa **Thuật toán Xử lý ảnh Y tế / AI Y khoa (Medical AI Core)** và **Kỹ thuật phát triển phần mềm lâm sàng hoàn chỉnh (Clinical Medical Software Engineering)**.  

---

## 1. Review & Phân Tích Hiện Trạng Đăng Ký Của Các Nhóm Trong Lớp

Dựa trên bảng theo dõi `SE374.R11 - Đồ án cuối kỳ.xlsx`, lớp hiện có 9 nhóm được phân bổ như sau:

| Nhóm | Thành Viên Đăng Ký | Tên Đề Tài & Phân Loại Topic | Nhận Xét Độc Lập & Mức Độ Cạnh Tranh |
| :---: | :--- | :--- | :--- |
| **Nhóm 1** | *(Chưa điền tên thành viên)* | **Sparse-view CT reconstruction** *(Topic 1)* | **Vị trí chiến lược:** Đề tài trọng tâm Buổi 2 của Thầy Doanh. Chưa có tên thành viên cụ thể $\rightarrow$ Cơ hội rất lớn nếu nhóm của bạn chọn nhận chủ đề này. |
| **Nhóm 2** | 24520554 - Nguyễn Huy Hoàng | **Tumor segmentation using CT/MRI** *(Topic 5)* | **Hướng kinh điển:** Sử dụng BraTS dataset. Bài toán phân vùng u não 3D rất quen thuộc, thư viện MONAI hỗ trợ mạnh mẽ, nhưng tính đột phá mới lạ không cao. |
| **Nhóm 3** | 24520336 - Huỳnh Đức Dũng | *Tên đề tài gì cho hay nhỉ* | **Chưa chốt đề tài:** Đang tìm kiếm định hướng. |
| **Nhóm 4** | *(Trống)* | *(Chưa có đăng ký)* | Vị trí hoàn toàn trống. |
| **Nhóm 5** | 23520922 - Hà Nhật Minh<br>23521267 - Nguyễn Việt Kỳ Quân | **Whole-Slide Image–Based Survival Risk Analysis Using Attention Multiple Instance Learning** *(Topic 4)* | **Hướng nghiên cứu hay:** Dự đoán thời gian sống sót bằng WSI giải phẫu bệnh (TCGA). Nhóm chọn phương pháp Attention-MIL kinh điển, có thể đối mặt với thách thức tài nguyên khi xử lý gigapixel WSI. |
| **Nhóm 6** | 23521262 - Nguyễn Hồng Quân<br>23521214 - Phạm Hoàng Phúc | **Weakly-supervised segmentation for whole slide images** *(Topic 6)* | **Bám sát đề bài:** Kế thừa paper LowMagWSS (7/2026 của lab Thầy Doanh/Thầy Khang). Hướng đi thông minh, giải quyết bài toán giảm dung lượng lưu trữ WSI. |
| **Nhóm 7** | 23520707 - Phạm Tuấn Khang<br>Phạm Hùng<br>Nguyễn Đình Tri | **Ứng dụng hỏi đáp dành cho bác sĩ chẩn đoán hình ảnh** *(Topic 8 - Tự do)* | **Medical VQA / Chatbot:** Ý tưởng hấp dẫn về mặt phần mềm, nhưng thách thức rất lớn về mặt y khoa là hiện tượng ảo giác (hallucination) của LLM và đòi hỏi tài nguyên GPU lớn nếu tự train mô hình đa phương thức (Med-VLM). |
| **Nhóm 8** | 23520826 - Võ Tuấn Kiệt | **Ứng dụng phát hiện người mắc rối loạn phổ tự kỉ dựa trên ảnh fMRI** *(Topic 8 - Tự do)* | **Brain fMRI & Web Demo:** Sử dụng dataset ABIDE I để khai thác functional connectivity não, định hướng làm web demo. Rất đúng trọng tâm môn phần mềm xử lý ảnh y tế. |
| **Nhóm 9** | 24520601 - Nguyễn Đại Hưng<br>24521489 - Dương Nguyễn Phú Quý<br>25520115 - Vũ Xuân Bách<br>25521929 - Bảo Trình | **Limited-angle CT reconstruction** *(Topic 2)* | **Nhóm đông thành viên nhất (4 bạn):** Tái tạo CT góc hạn chế. Đề tài toán học và xử lý ảnh nghịch đảo rất nặng, cần sự phối hợp tốt giữa sinogram inpainting và tái tạo ảnh. |

### 📌 Các "Kho Báu" Đang Bị Bỏ Trống (Unclaimed Strategic Gaps):
1. **Topic 1 (Sparse-view CT reconstruction):** Giảng viên giảng rất kỹ ở Buổi 2 và vừa công bố paper đỉnh cao **CG-GLORE** (15/08/2026). Nhóm 1 hiện chưa có danh sách thành viên!
2. **Topic 3 (Cancer subtyping using Whole Slide Images):** **Chưa có bất kỳ nhóm nào chọn!** Trong khi phòng lab của Thầy Doanh vừa ra mắt bài báo **CGRL** (16/07/2026) kèm mã nguồn GitHub chính thức của sinh viên UIT (`ThucHuynh44/CGRL`).
3. **Topic 7 (Collateral imaging in acute ischemic stroke using MRI):** **Chưa có bất kỳ nhóm nào chọn!** Đánh giá tuần hoàn bàng hệ trong cấp cứu đột quỵ thiếu máu não là bài toán lâm sàng cực kỳ cấp thiết, cứu sống bệnh nhân trong "cửa sổ vàng".

---

## 2. Tiêu Chí "Kép" Của Môn "Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa"

Vì đây là môn học của **Khoa Kỹ thuật Phần mềm (Software Engineering)**, đồ án đạt điểm xuất sắc ($\ge 9.5$) không chỉ dừng lại ở việc chạy code trong Jupyter Notebook, mà bắt buộc phải thỏa mãn **2 trụ cột**:

```mermaid
graph LR
    subgraph Trụ Cột 1: Thuật Toán Lõi Y Khoa
        A1["Dữ liệu Y tế Chuẩn<br>(DICOM / NIfTI / WSI)"]
        A2["Deep Learning SOTA<br>(Unrolling / MIL / 3D Swin)"]
        A3["Chống Rò Rỉ Bệnh Nhân<br>(Patient-Level Split)"]
    end
    
    subgraph Trụ Cột 2: Kỹ Thuật Phần Mềm Lâm Sàng
        B1["Giao Diện Xem Ảnh Y Tế<br>(MPR 3 mặt cắt, HU Windowing)"]
        B2["Pipeline Xử Lý Mượt Mà<br>(FastAPI Engine / Streamlit)"]
        B3["Trực Quan Hóa Tương Tác<br>(Overlay Mask, Heatmap, Báo Cáo PDF)"]
    end
    
    Trụ Cột 1 --> C["Phần Mềm Xử Lý Ảnh Y Khoa Đạt Điểm A+"]
    Trụ Cột 2 --> C
```

---

## 3. Top 4 Đề Xuất Chủ Đề Tối Ưu Nhất Cho Nhóm Của Bạn

---

### 🌟 LỰA CHỌN 1 (KHUYÊN DÙNG HÀNG ĐẦU): Topic 1 - Sparse-View CT Reconstruction
**Tên đề tài phần mềm:**  
> **"Phát triển phần mềm tái tạo ảnh cắt lớp vi tính liều thấp (Sparse-View CT Reconstruction Studio) dựa trên mạng Unrolling CG-GLORE"**

- **Bản chất bài toán:** Khôi phục ảnh CT chất lượng cao từ số góc chiếu thưa (32, 64 view thay vì 720 view) nhằm giảm thiểu liều tia X cho bệnh nhân; triệt tiêu hoàn toàn vệt nhiễu streak artifacts.
- **Phương pháp SOTA lõi:** Kế thừa mạng **CG-GLORE** (arXiv:2608.15246, công bố tháng 8/2026 bởi chính TS. Bùi Cao Doanh và PGS. TS. Nguyễn Khang). Giải hệ Hessian surrogate bằng thuật toán Conjugate Gradient (CG) kết hợp khối điều chuẩn GLORE (Nyström attention + CNN).
- **Bộ dữ liệu chuẩn:** AAPM Low-Dose CT Grand Challenge (Mayo Clinic), DeepLesion.
- **Thiết kế phần mềm (Software Engineering Deliverable):**
  * Xây dựng ứng dụng Web tương tác (Streamlit / Next.js):
    1. Bác sĩ tải lên ảnh sinogram thưa hoặc ảnh CT gốc để mô phỏng liều tia.
    2. Kéo thanh trượt tùy chỉnh số lượng góc chiếu ($n_v = 32, 64, 128$).
    3. Hiển thị đối chiếu 3 khung nhìn song song: **FBP (kinh điển bị nhiễu)** vs **CG-GLORE (tái tạo sắc nét)** vs **Ground Truth**.
    4. Cung cấp công cụ Hounsfield Unit Windowing (Brain, Bone, Soft Tissue, Lung).
    5. Hiển thị bản đồ sai lệch (Error Difference Map) và chỉ số PSNR / SSIM trực tiếp.
- **Điểm thuyết phục Giảng viên:** 
  * Đánh đúng 100% nội dung Giảng viên giảng dạy tại Buổi 2.
  * Kế thừa trực tiếp công trình nghiên cứu mới nhất của Giảng viên.

---

### 🌟 LỰA CHỌN 2 (TIÊN PHONG & CÓ SẴN SOURCE LAB): Topic 3 - Cancer Subtyping Using WSI
**Tên đề tài phần mềm:**  
> **"Phần mềm hỗ trợ phân loại phân nhóm ung thư biểu mô từ tiêu bản giải phẫu bệnh số (WSI Pathology Subtyping Assistant) ứng dụng mô hình CGRL"**

- **Bản chất bài toán:** Phân loại phân nhóm ung thư vú (IDC vs ILC) hoặc ung thư phổi (LUAD vs LUSC) từ ảnh tiêu bản mô bệnh học gigapixel ($100,000 \times 100,000$ pixels).
- **Phương pháp SOTA lõi:** Kế thừa mô hình **CGRL** (arXiv:2607.12556, công bố tháng 7/2026 bởi học viên UIT Thuc Huynh và TS. Bùi Cao Doanh). Tỉa bớt patch theo concept bệnh học (Concept-Relevance Pruning Top-$K$) và học biểu diễn tương phản.
- **Mã nguồn kế thừa:** Đã có sẵn repo GitHub của lab: `https://github.com/ThucHuynh44/CGRL`.
- **Bộ dữ liệu chuẩn:** TCGA-BRCA, TCGA-NSCLC.
- **Thiết kế phần mềm (Software Engineering Deliverable):**
  * Giao diện WSI Viewer (sử dụng OpenSeadragon / Streamlit):
    1. Cho phép zoom vô cực (Deep Zoom $1\times \rightarrow 40\times$) trên tiêu bản gigapixel.
    2. Hiển thị Top-$K$ patches có trọng số chú ý bệnh lý cao nhất do thuật toán Concept Pruning trích xuất.
    3. Phủ bản đồ nhiệt (Attention Heatmap Overlay) thể hiện mật độ tế bào ác tính.
    4. Bảng chẩn đoán phân nhóm ung thư tự động kèm độ tin cậy xác suất (Probability Confidence).
- **Điểm thuyết phục Giảng viên:**
  * Chủ đề đang **hoàn toàn trống trong lớp** (chưa nhóm nào đăng ký).
  * Mã nguồn tác giả sinh viên UIT sẵn có, giảm thiểu rủi ro kỹ thuật, tập trung tối đa vào việc đóng gói thành phần mềm y tế hoàn chỉnh.

---

### 🌟 LỰA CHỌN 3 (TÍNH NHÂN VĂN & CẤP CỨU LÂM SÀNG CAO): Topic 7 - Ischemic Stroke Collateral MRI
**Tên đề tài phần mềm:**  
> **"Hệ thống phần mềm lượng hóa tổn thương đột quỵ và tuần hoàn bàng hệ trên ảnh cộng hưởng từ MRI (Acute Stroke Neuro-Imaging CADe/CADx)"**

- **Bản chất bài toán:** Đánh giá hệ thống mạch máu tuần hoàn bàng hệ trong đột quỵ thiếu máu não cấp. Đây là chỉ số sống còn giúp bác sĩ cấp cứu quyết định bệnh nhân có còn cơ hội can thiệp lấy huyết khối tĩnh mạch/động mạch hay không.
- **Phương pháp SOTA lõi:** Kế thừa bài báo Springer IJCARS (2024) do Giảng viên chỉ định trong `topic.md`. Sử dụng mạng học sâu phân đoạn vùng lõi nhồi máu (infarct core) và vùng tranh tối tranh sáng (penumbra) trên các chuỗi xung DWI, PWI, FLAIR.
- **Bộ dữ liệu chuẩn:** ISLES 2022 (Ischemic Stroke Lesion Segmentation Challenge).
- **Thiết kế phần mềm (Software Engineering Deliverable):**
  * Giao diện chuyên dụng cho phòng cấp cứu thần kinh:
    1. Nhập đồng thời các chuỗi xung MRI (DWI, ADC, FLAIR) của bệnh nhân đột quỵ.
    2. Tự động tính toán và hiển thị thể tích vùng lõi tổn thương ($cm^3$) và vùng mô não còn khả năng cứu sống (Mismatch volume).
    3. Đưa ra điểm tuần hoàn bàng hệ tự động (Collateral Score).
    4. Xuất báo cáo tóm tắt lâm sàng chuẩn y tế phục vụ hội chẩn nhanh.
- **Điểm thuyết phục Giảng viên:**
  * Chủ đề **chưa có nhóm nào đăng ký**, bài toán cấp cứu có ý nghĩa thực tiễn cứu người rất cao.

---

### 🌟 LỰA CHỌN 4 (PHẦN MỀM THỰC THỤ DỄ DEMO NHẤT): Topic 5 Biến Thể - 3D Brain Tumor & PACS Viewer
**Tên đề tài phần mềm:**  
> **"Phần mềm máy trạm chẩn đoán u não 3D (Neuro-Oncology 3D DICOM Workstation) tích hợp mô hình MONAI Swin-UNETR"**

- **Bản chất bài toán:** Tự động phân đoạn 3 thành phần của u não Glioma (vùng phù nề peritumoral edema, vùng u bắt thuốc GD-enhancing tumor, và vùng u hoại tử necrotic core) trên ảnh MRI thể tích 3D 4 chuỗi xung (T1, T1ce, T2, FLAIR).
- **Phương pháp SOTA lõi:** MONAI Swin-UNETR hoặc nnU-Net với hàm mất mát DiceFocalLoss.
- **Bộ dữ liệu chuẩn:** BraTS 2021 / BraTS 2023.
- **Thiết kế phần mềm (Software Engineering Deliverable):**
  * Xây dựng máy trạm y tế hoàn chỉnh (Mini PACS Workstation):
    1. Đọc trực tiếp tệp NIfTI (`.nii.gz`) và thư mục DICOM.
    2. Cắt lát đồng bộ theo 3 mặt phẳng giải phẫu (Axial, Coronal, Sagittal) với thanh cuộn slice mượt mà.
    3. Lớp phủ mặt nạ 3 màu đại diện cho 3 phân vùng u não.
    4. Tự động đo kích thước lớn nhất khối u theo tiêu chuẩn RECIST 1.1 và tính tổng thể tích khối u ($cm^3$).
- **Điểm thuyết phục Giảng viên:**
  * Giao diện phần mềm cực kỳ trực quan, dễ "gây ấn tượng mạnh" (wow effect) khi thuyết trình slide và bảo vệ trước hội đồng.

---

## 4. Bảng So Sánh Chiến Lược Chọn Đề Tài

| Tiêu Chí So Sánh | Lựa Chọn 1: Sparse-View CT (CG-GLORE) | Lựa Chọn 2: WSI Subtyping (CGRL) | Lựa Chọn 3: Stroke MRI Collateral | Lựa Chọn 4: 3D Tumor Workstation |
| :--- | :---: | :---: | :---: | :---: |
| **Mức độ trùng lặp với lớp** | **Thấp** (Nhóm 1 chưa có tên) | **0%** (Chưa ai chọn) | **0%** (Chưa ai chọn) | Trung bình (Nhóm 2 đã đăng ký) |
| **Mức độ quan tâm của Giảng viên** | **Tối đa (10/10)** (Trọng tâm Buổi 2) | **Rất cao (9.5/10)** (Paper lab 7/2026) | **Cao (8.5/10)** (Đề bài chỉ định) | Khá (8.0/10) (Kinh điển) |
| **Mã nguồn tham khảo sẵn có** | Paper chi tiết (arXiv 8/2026) | Repo GitHub UIT đầy đủ | Baseline ISLES mở | MONAI tutorials đầy đủ |
| **Yêu cầu phần cứng GPU** | Vừa phải (6-8 GB VRAM) | Thấp (nhờ Concept Pruning) | Vừa phải (8-12 GB VRAM) | Khá (12-16 GB VRAM) |
| **Khả năng làm demo phần mềm** | Rất tốt (Kéo thanh trượt góc) | Rất tốt (WSI Deep Zoom) | Rất tốt (Báo cáo cấp cứu) | **Tuyệt đỉnh (Viewer 3 mặt cắt)** |
