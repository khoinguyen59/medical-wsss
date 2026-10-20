# Rule 15: Proactive Compute, Data & Execution Lifecycle Governance

> **Specification Standard:** `agents.md` v5.0 - Professional Deep Learning & MLOps Standard  
> **Core Mandate:** Khắc phục triệt để thói quen thụ động. Agent phải luôn đi trước một bước, tính toán định mức phần cứng, dữ liệu, chi phí, thời gian và hoàn thiện vòng đời thực nghiệm khép kín từ A đến Z mà không đợi người dùng phải nhắc.

---

## 1. Bản Đồ 6 Chiều Kích Chủ Động Bắt Buộc (The 6 Proactive Dimensions)

Khi đề xuất hoặc chuẩn bị bất kỳ tác vụ thực nghiệm nào (Google Colab, Cloud compute, Local training, Docker/Cluster), Agent BẮT BUỘC phải chủ động cung cấp đầy đủ 6 thông số:

### 1.1 Khảo Sát Hiện Trạng Dữ Liệu Thật (Data Status & Source)
- **Tình trạng lưu trữ:** Khẳng định dữ liệu đã có trong máy hay chưa, nằm tại thư mục nào, dung lượng nén (`.zip`) và sau giải nén.
- **Tính xác thực y khoa:** Bắt buộc dùng $100\%$ dữ liệu bệnh nhân thật (như TCGA-BRCA, BCSS-WSSS). Cấm tuyệt đối dùng synthetic data (như vẽ hình bằng OpenCV) nếu người dùng không yêu cầu.
- **Phương thức tải tự động:** Cung cấp link Google Drive chính thức kèm mã File ID cụ thể (ví dụ: `1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn`) và lệnh tải trực tiếp qua `gdown` hoặc mount Google Drive.

### 1.2 Định Danh & Tính Toán Phần Cứng GPU (GPU & VRAM Sizing)
- **Tên GPU cụ thể:** Tesla T4, NVIDIA L4, NVIDIA A100 SXM, NVIDIA V100.
- **Dung lượng VRAM:** 15/16 GB (T4), 24 GB (L4), 40/80 GB (A100).
- **Mức tiêu tốn VRAM đỉnh:** Tính toán theo kích thước ảnh ($224 \times 224$ pixels) và kích thước batch ($B=8, 16, 20, 32$).

### 1.3 Phân Định Rõ Ràng Gói Free vs Gói Trả Phí (Free vs Paid Tier Breakdown)
- **Gói Miễn phí (Free Tier - Tesla T4):** Chi phí $0$ đồng, $0$ compute units, giới hạn phiên $6 - 12$ tiếng. Phù hợp cho Sanity Run và đánh giá mẫu vừa.
- **Gói Trả phí (Pro/Pro+ - NVIDIA L4 / A100):** Mức trừ compute units (L4: $\sim 1.5 - 2.0$ units/giờ; A100: $\sim 9.5 - 13.0$ units/giờ), dung lượng High-RAM ($51 - 83$ GB). Phù hợp cho huấn luyện toàn diện.

### 1.4 Ước Tính Thời Gian & Môi Trường Phiên Bản (Runtime & Time Estimation)
- **Môi trường:** Khai báo rõ phiên bản Python (3.10/3.11/3.12), PyTorch ($\ge 2.0$), CUDA ($\ge 11.8/12.1$).
- **Thời gian chạy ước tính:** Chia nhỏ theo từng giai đoạn (Stage 1 train: bao nhiêu phút/giờ; Sinh nhãn giả: bao nhiêu phút; Stage 2 train: bao nhiêu phút/giờ; Suy luận test: bao nhiêu giây/ảnh).

### 1.5 Vòng Đời Thực Nghiệm Khép Kín Từ A Đến Z (Closed-Loop Lifecycle)
Mọi Jupyter Notebook hoặc Execution Script BẮT BUỘC phải có đầy đủ 7 bước khép kín:
1. `!nvidia-smi` và kiểm tra CUDA/VRAM.
2. Cài đặt thư viện (`pip install`).
3. Tải và giải nén dữ liệu thật (`gdown --id` hoặc mount Drive).
4. Tải trọng số mô hình đã huấn luyện sẵn SOTA (`.pth`).
5. Thực thi suy luận / đánh giá chỉ số y khoa thật (mIoU, Dice, FPS).
6. Trực quan hóa hình ảnh lâm sàng độ phân giải cao ($\ge 300$ DPI).
7. **ĐÓNG GÓI ZIP VÀ TỰ ĐỘNG TẢI VỀ MÁY TÍNH (`files.download`):** Tuyệt đối không dừng lại ở khâu hiển thị trên màn hình mà bắt buộc phải có ô nén kết quả thành `.zip` và tự động kích hoạt tải về máy tính người dùng.

### 1.6 Tiên Liệu Lỗi & Kỹ Thuật Dự Phòng (Defensive Engineering)
- Cung cấp sẵn cả cách tự động (pop-up tải file) và hướng dẫn cách tải thủ công bằng chuột từ thanh công cụ bên trái (Files Panel) đề phòng trình duyệt chặn pop-up.

---

## 2. Checklist Kiểm Định Trước Khi Bàn Giao Notebook / Script
- [ ] Đã kiểm tra dữ liệu là dữ liệu thật $100\%$, không có code vẽ ảnh giả lập?
- [ ] Đã ghi rõ cấu hình GPU khuyên dùng và phân biệt gói Free/Pro?
- [ ] Đã có File ID chính thức để tải tự động, không bắt người dùng upload thủ công?
- [ ] Đã có ô cuối cùng tự động tải file zip kết quả về máy tính người dùng?
