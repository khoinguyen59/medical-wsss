# Phân Tích Độ Khả Thi Toàn Diện: Đối Chiếu Hai Bản Mini Proposal SE374.R11

Tài liệu đối chiếu và đánh giá khả thi kỹ thuật giữa hai bản đề cương:
- **Bản 1:** `SE374_MINI_PROPOSAL_WSSS_WSI` (Bản Toàn Diện / SOTA Mở Rộng - Digital Pathology Workstation)
- **Bản 2:** `SE374_MINI_PROPOSAL_WSSS_TISSUE` (Bản Vừa Sức / Thực Dụng - WSSS-Tissue MLPS & CGM)

---

## 1. Bảng Ma Trận So Sánh Tổng Hợp Đa Chiều

| Tiêu Chí Đánh Giá | Bản 1: Toàn Diện SOTA (WSSS-WSI) | Bản 2: Vừa Sức Khả Thi Cao (WSSS-Tissue) | Nhận Định & Lời Khuyên |
| :--- | :--- | :--- | :--- |
| **1. Định hướng bài toán** | Trạm làm việc giải phẫu bệnh số (Digital Pathology Workstation) trên ảnh gigapixel WSI; tổng hợp đa phương pháp SOTA (Foundation Models CONCH/UNI, HisynSeg, ConStruct, PathMamba). | Phân vùng mô học 2 giai đoạn (MLPS & CGM) bám sát bài báo Han et al. (*MedIA 2022*); cải tiến nhẹ khâu điều hòa biên (Tissue Mixing). | Bản 2 tập trung vào 1 bài toán lõi rõ ràng; Bản 1 mang tính bao quát một hệ sinh thái lớn. |
| **2. Độ phức tạp Dataset** | **Rất cao:** Ảnh WSI thô (.svs, .ndpi, .tiff), kích thước gigapixel ($100.000 \times 100.000$ px), dung lượng hàng trăm GB đến TB (TCGA-BRCA, Camelyon16). Đòi hỏi cắt patch, lọc background, xử lý kim tự tháp đa tầng. | **Vừa phải:** Bộ dữ liệu chuẩn BCSS-WSSS (151 ROIs) đã được bóc tách sẵn thành $17.286$ mảng ảnh chuẩn $224 \times 224$ px. Dung lượng gọn nhẹ (~2.5 GB). | Bản 2 nạp dữ liệu tức thì, không tắc nghẽn I/O; Bản 1 mất từ 1 đến 2 tuần chỉ để tải và tiền xử lý WSI. |
| **3. Độ khó giải thuật** | **Rất khó:** Chưng cất cấu trúc (Structural Distillation) từ Vision Transformer lớn sang SegFormer; tổng hợp ranh giới Bézier; điều phối đa mô hình Foundation Model. | **Vừa tầm:** ResNet-38 kết hợp Progressive Dropout Attention (PDA) ở Giai đoạn 1; DeepLabV3+ kết hợp Cổng phân loại CGM ở Giai đoạn 2. | Bản 2 có mã nguồn mở kiểm chứng đầy đủ; Bản 1 phụ thuộc nhiều module thuật toán phức tạp chưa hoàn toàn đồng nhất. |
| **4. Độ sâu nghiên cứu** | Rất sâu về mặt lý thuyết bao quát; khảo sát rộng các hướng tiếp cận SOTA mới nhất (2024-2026). | Sâu và chắc về mặt thực nghiệm tái lập (reproducibility), phân tích cơ chế từng tầng mạng ($b4\_5, b5\_2, bn7$) và ablation study. | Bản 1 thích hợp làm khóa luận tốt nghiệp / đề tài nghiên cứu dài hạn; Bản 2 tối ưu cho đồ án môn học 10 tuần. |
| **5. Nhu cầu phần cứng & GPU** | Cần GPU VRAM lớn ($\ge 16\text{GB} - 24\text{GB}$, như RTX 4090, A100/V100) nếu fine-tune hoặc suy luận Foundation Models quy mô lớn. | 1 GPU phổ thông ($\mathbf{8\text{GB} - 12\text{GB}}$ VRAM, như RTX 3060, RTX 4060, RTX 2060 Super) là hoàn toàn đủ. | Bản 2 tương thích 100% máy trạm sinh viên UIT. |
| **6. Khả năng chạy trên Colab / Kaggle** | - Colab Free (T4 15GB): Dễ bị tràn RAM/Disk khi xử lý file .svs hàng chục GB; giới hạn 12h timeout dễ làm đứt đoạn tiền xử lý.<br>- Kaggle: 20GB disk không đủ lưu dataset WSI lớn. | - **Colab Free:** Chạy mượt mà (Dataset 2.5 GB tải trong 3 phút, batch size 16-32 trên T4 vận hành ổn định).<br>- **Kaggle (2x T4 hoặc P100 16GB):** Huấn luyện Stage 1 và Stage 2 cực kỳ nhanh và miễn phí 100%. | Bản 2 tận dụng 100% tài nguyên miễn phí của Colab và Kaggle mà không lo ngắt kết nối. |
| **7. Chi phí tài nguyên** | Có thể phát sinh chi phí thuê GPU đám mây (RunPod, Lambda Labs, Google Cloud) hoặc tài khoản HuggingFace xin cấp quyền trọng số đóng (CONCH, UNI). | **0 VNĐ (Hoàn toàn miễn phí):** Dùng máy cá nhân hoặc Colab/Kaggle miễn phí; trọng số backbone ResNet-38 và DeepLabV3+ tải tự do. | Bản 2 an toàn tuyệt đối về mặt tài chính sinh viên. |
| **8. Ngăn xếp phần mềm (Software Stack)** | Phức tạp: OpenSlide / cuCIM, DeepZoom tile server, FastAPI, WebGL viewer hỗ trợ zoom đa tầng gigapixel, xuất báo cáo chuẩn CAP. | Gọn gàng: PyTorch, TorchScript, OpenCV, FastAPI, giao diện Web trực quan kéo thả mảng ảnh $224 \times 224$ px, hiển thị CAM và mặt nạ phân tầng. | Bản 2 hoàn thành sản phẩm phần mềm trọn vẹn trong 2 tuần cuối; Bản 1 có nguy cơ trễ hạn phần giao diện. |
| **9. Tiến độ 10 tuần** | Quá tải: Rủi ro cao bị nghẽn ở tuần 3-5 (khâu nạp WSI và cấu hình trọng số mô hình nền tảng). | Cân đối: 2 tuần nạp data và baseline, 3 tuần huấn luyện & cải tiến, 2 tuần xây dựng phần mềm, 2 tuần đánh giá & báo cáo. | Bản 2 bảo đảm 100% tiến độ nộp bài. |
| **10. Khả năng bảo vệ & Điểm số** | Ý tưởng hoành tráng, ấn tượng nếu làm được trọn vẹn. Tuy nhiên nếu không kịp hoàn thành sẽ dễ bị trừ điểm do thiếu minh chứng thực nghiệm. | Thực dụng, chắc chắn: 100% số liệu thực nghiệm thật, kiểm định 5 seeds, có phần mềm demo chạy trơn tru, không có số liệu ảo. | Bản 2 rất dễ thuyết phục hội đồng vì tính hoàn thiện cao và độ trung thực học thuật tuyệt đối. |

---

## 2. Phân Tích Chi Tiết Từng Khía Cạnh

### 2.1 Khía Cạnh Dữ Liệu (Dataset Size & Pipeline Complexity)
- **Bản 1 (WSI Gigapixel):**
  - Kích thước 1 tiêu bản WSI trung bình từ $500\text{MB}$ đến $3\text{GB}$. 100 ca bệnh cần khoảng $100\text{GB} - 300\text{GB}$ bộ nhớ lưu trữ.
  - Cần cài đặt thư viện C-level `OpenSlide` (thường gặp lỗi driver trên Windows).
  - Phải lập trình pipeline phát hiện vùng mô hữu ích (Tissue Thresholding/Otsu), sau đó cắt thành hàng triệu patch nhỏ, lưu trữ metadata tọa độ cực kỳ cồng kềnh.
- **Bản 2 (BCSS-WSSS Patches):**
  - Bài báo Han et al. (*MedIA 2022*) đã trích xuất sẵn 151 vùng mô tiêu chuẩn thành $17.286$ mảng ảnh $224 \times 224$ pixels.
  - Tổng kích thước toàn bộ tập dữ liệu chỉ khoảng **$2.5\text{GB}$**.
  - Tải về trong 3-5 phút, giải nén trực tiếp trên máy cục bộ hoặc Google Drive / Kaggle Dataset. Không gặp bất kỳ rào cản kỹ thuật nào về I/O hay thư viện C.

### 2.2 Khía Cạnh Phần Cứng & Nền Tảng Tính Toán (GPU, Colab, Kaggle)
- **Bản 1:**
  - Nếu sử dụng mô hình nền tảng như CONCH (ViT-B-32) hoặc UNI (ViT-Large với hơn 300 triệu tham số), việc trích xuất đặc trưng offline cho hàng triệu patch đòi hỏi hàng chục giờ tính toán trên GPU mạnh.
  - Môi trường Colab Free (12.7 GB RAM hệ thống và 15 GB VRAM T4) rất dễ bị dừng đột ngột (Kernel Died / Out of RAM) khi đọc đồng thời nhiều file WSI lớn.
- **Bản 2:**
  - Backbone ResNet-38 có kiến trúc vừa phải (khoảng 26 triệu tham số).
  - Giai đoạn 1 (Progressive Dropout Attention): Tiêu tốn khoảng $4.5\text{GB} - 6\text{GB}$ VRAM với batch size 16.
  - Giai đoạn 2 (DeepLabV3+): Tiêu tốn khoảng $5.5\text{GB} - 7\text{GB}$ VRAM với batch size 8.
  - Máy trạm có GPU đơn như NVIDIA RTX 3060 (12GB), RTX 4060 (8GB) hoặc thậm chí tài khoản Google Colab Free / Kaggle GPU đều huấn luyện hoàn toàn ổn định từ đầu đến cuối mà không lo OOM.

### 2.3 Khía Cạnh Độ Khó Giải Thuật & Nguy Cơ Nghẽn Tiến Độ
- **Bản 1:**
  - Cần hiện thực đồng thời nhiều kỹ thuật phức tạp: tổng hợp ảnh ghép Bézier, chưng cất đặc trưng Foundation Model, mô hình hóa chuỗi dài với PathMamba.
  - Có sự phân tán về mục tiêu: vừa phải giải bài toán Gigapixel WSI, vừa phải giải bài toán phân vùng yếu, vừa phải làm trạm làm việc DeepZoom. Nguy cơ "lan man, dở dang" ở tuần thứ 6-7 là rất lớn.
- **Bản 2:**
  - Bài toán 2 giai đoạn của WSSS-Tissue được thiết kế theo cấu trúc tách rời độc lập (Decoupled Architecture):
    - Giai đoạn 1 huấn luyện xong $\rightarrow$ Sinh và lưu toàn bộ nhãn giả vào đĩa $\rightarrow$ Kiểm tra chất lượng CAM.
    - Giai đoạn 2 huấn luyện mạng phân vùng DeepLabV3+ trên tập nhãn giả đó kết hợp Cổng phân loại CGM.
  - Tính module hóa cao: nếu một thành phần gặp sự cố, các thành phần khác vẫn hoạt động độc lập. Tiến độ kiểm soát được 100%.

### 2.4 Khía Cạnh Phần Mềm Ứng Dụng (Software Deliverable)
- **Bản 1:**
  - Trạm làm việc WSI đòi hỏi xây dựng Tile Server (Dynamic Patch Serving) theo chuẩn DeepZoom Image (DZI) để trình duyệt web có thể pan/zoom mượt mà trên ảnh $100.000 \times 100.000$ px. Đây là khối lượng công việc phần mềm tương đương một đồ án công nghệ phần mềm riêng biệt.
- **Bản 2:**
  - Ứng dụng tập trung vào mảng ảnh quan tâm (ROI Patch Assistant): Bác sĩ tải lên ảnh mảng H&E $224 \times 224$ px hoặc chọn từ danh sách ca bệnh mẫu; backend FastAPI xử lý trong $<1.5$ giây; giao diện hiển thị ảnh gốc, bản đồ CAM và phân lớp mô kèm tỷ lệ diện tích u sống.
  - Gọn nhẹ, trực quan, dễ demo trước giảng viên mà không sợ lỗi nghẽn mạng hay trễ hiển thị.

---

## 3. Khuyến Nghị Chiến Lược Cho Nhóm Sinh Viên

1. **Lựa chọn tối ưu:** **Bản 2 (`SE374_MINI_PROPOSAL_WSSS_TISSUE`)** là lựa chọn an toàn, thực tế và mang lại xác suất đạt điểm cao nhất cho đồ án SE374.R11.
2. **Chiến lược "Thực dụng đỉnh cao":**
   - Lấy Bản 2 làm khung xương triển khai thực tế 100% mã nguồn, thực nghiệm và phần mềm.
   - Trích dẫn tầm nhìn và một số so sánh lý thuyết của Bản 1 vào phần "Hướng phát triển tương lai" (Future Work) trong báo cáo cuối kỳ để gây ấn tượng mạnh về độ rộng kiến thức trước Hội đồng chấm thi!
