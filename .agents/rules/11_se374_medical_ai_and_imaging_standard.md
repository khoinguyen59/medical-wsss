---
id: rule-11-se374-medical-ai-and-imaging-standard
title: SE374.R11 Medical AI & Medical Imaging Engineering Standard
activation: always_on
description: "Quy chuẩn kỹ thuật, pipeline khép kín, phân chia chống rò rỉ bệnh nhân (Patient-Level Anti-Leakage), liêm chính y khoa và ma trận công cụ cho Đồ án môn học SE374.R11 - Trí tuệ nhân tạo trong Y tế & Xử lý ảnh Y khoa (UIT 2026)."
---

# Rule 11: SE374.R11 - Medical AI & Medical Imaging Standard

## 1. Nguyên Tắc Cốt Lõi: Tận Dụng Tri Thức Sẵn Có, Bảo Toàn Sáng Tạo & An Toàn Tài Nguyên

1. **Hỏi Ý Kiến Người Dùng Trước Quyết Định Quan Trọng & Huấn Luyện Nặng (Human-in-the-Loop & Resource Safeguard):**
   - **Các quyết định bắt buộc phải hỏi người dùng:** Chốt đề tài lựa chọn (trong 8 topic của môn học), chốt bộ dữ liệu (AAPM, DeepLesion, TCGA, BraTS, ABIDE I, ISLES...), chốt kiến trúc baseline và proposed method.
   - **Bắt buộc hỏi trước khi chạy tác vụ tốn VRAM/GPU hoặc thời gian:**
     * *Dữ liệu y tế (3D CT/MRI, WSI gigapixel):* Rất nặng và tốn VRAM. Trước khi train full-resolution hoặc full-epoch, bắt buộc phải chạy Sanity Run / Dry Run (1-2 batch nhỏ, downsampled slice hoặc tập con bệnh nhân) để kiểm tra pipeline.
     * Báo cáo ước tính thời gian, dung lượng bộ nhớ và xin xác nhận của người dùng trước khi chạy training dài hạn hoặc multi-seed (5 seeds).
     * Tuyệt đối không tự ý gọi API trả phí hoặc cloud compute khi chưa được duyệt.

2. **Chống Rò Rỉ Dữ Liệu Bệnh Nhân Tuyệt Đối (Patient-Level Anti-Leakage):**
   - Trong ảnh y khoa (CT, MRI, fMRI, WSI), một bệnh nhân có thể có hàng trăm lát cắt (slices), hàng nghìn mảng nhỏ (patches), hoặc nhiều chuỗi xung (sequences).
   - **Quy tắc bất biến:** Phân chia tập `train / val / test` **BẮT BUỘC theo Patient ID / Case ID**, tuyệt đối không chia ngẫu nhiên ở cấp độ slice hay patch.
   - Nếu lát cắt của cùng một bệnh nhân xuất hiện ở cả tập train và test $\rightarrow$ Vi phạm nghiêm trọng về rò rỉ dữ liệu (Data Leakage) và khiến mô hình mất giá trị đánh giá lâm sàng thực tế.
   - Mọi bước chuẩn hóa (Hounsfield Unit clipping, z-score normalization, min-max scaling) phải tính toán tham số trên tập train của bệnh nhân rồi mới áp dụng sang val/test.

3. **Liêm Chính Học Thuật & Y Khoa Tuyệt Đối (Zero Mockup):**
   - Mọi kết quả thực nghiệm, bảng chỉ số (Dice, IoU, PSNR, SSIM, C-Index, AUC, Macro-F1) và hình ảnh trực quan hóa tổn thương/tái tạo bắt buộc phải chạy từ script thật trên dữ liệu thật.
   - Tuyệt đối nghiêm cấm việc dùng dummy metrics, fake score hay số liệu tự suy đoán. Vi phạm tính trung thực khoa học bị xử lý 0 điểm theo quy chế của Giảng viên và Nhà trường.

4. **Kế Thừa Tối Đa Nền Tảng Mã Nguồn Mở Uy Tín (MCP & GitHub-First):**
   - Ưu tiên sử dụng và kế thừa các thư viện chuẩn mực quốc tế:
     * **MONAI (`monai`):** Framework chuẩn y khoa của PyTorch (transforms, backbones Swin-UNETR, DynUNet, losses DiceFocal, sliding window inference).
     * **Xử lý định dạng y tế:** `nibabel` (NIfTI .nii/.nii.gz), `pydicom` (DICOM .dcm), `SimpleITK`, `tifffile` / `openslide` (Whole Slide Images .svs/.tif).
     * **Tái tạo CT:** Kế thừa phương pháp tối ưu unrolling như CG-GLORE, Learned Primal-Dual (LPD), FBPConvNet.
     * **Giải phẫu bệnh số (WSI):** Kế thừa các framework MIL như CGRL, CLAM, MergeSurv, LowMagWSS.
   - Không tự viết lại các hàm toán học nền tảng khi cộng đồng đã tối ưu bằng C/CUDA.

5. **Kiến Trúc Module Phân Tầng Độc Lập (Decoupled Modular Architecture):**
   - Mã nguồn trong `src/` phải được tổ chức thành các dịch vụ độc lập, ranh giới rõ ràng:
     - `src/config/`: Cấu hình đường dẫn dataset, siêu tham số, seed, device (GPU/CPU).
     - `src/datasets/`: Dịch vụ nạp dữ liệu y tế, chuẩn hóa HU/spacing, và phân chia theo Patient ID.
     - `src/models/`: Dịch vụ mô hình (Reconstruction, 3D Segmentation, WSI-MIL, Brain Connectivity).
     - `src/losses_metrics/`: Định nghĩa các hàm mất mát y khoa (Dice, Focal, Perceptual) và chỉ số định lượng (Dice, PSNR, SSIM, C-Index).
     - `src/visualization/`: Dịch vụ hiển thị lát cắt y tế (Axial/Coronal/Sagittal), overlay mask tổn thương, heatmap chú ý và Web Demo (Streamlit).
     - `src/pipeline/`: Trình điều phối chạy tự động toàn bộ chu trình end-to-end.

6. **Quy Chuẩn Bắt Buộc: Tải Paper Về Máy, Đọc Bằng Vision & Viết File Markdown Toàn Văn (Vision-to-Markdown Protocol):**
   - **Tải file PDF:** Mọi tài liệu nghiên cứu, bài báo y khoa (arXiv, Springer, IEEE, PubMed) bắt buộc phải tải về lưu tại `docs/papers/<paper_id>.pdf`.
   - **Đọc trực quan bằng Vision (Multimodal Reading):** Dùng tính năng Vision / đa phương thức để duyệt từng trang, đọc sơ đồ kiến trúc khối giải phẫu/mô hình, bảng đối chứng định lượng (PSNR, Dice, C-Index) và công thức toán học (Radon, FBP, Hessian unrolling).
   - **Tạo file Markdown đối ứng:** Bắt buộc viết file `docs/papers/<paper_id>.md` ghi lại toàn văn cấu trúc hóa chi tiết để Agent đọc lại nhanh chóng, trích dẫn chính xác và tích hợp liền mạch vào đề cương Proposal và báo cáo đồ án.
   - **Lý do tối thượng:** AI đọc file PDF dạng ảnh hoặc cột đôi rất kém, dễ nuốt mất ký hiệu toán học hoặc tráo đổi thứ tự cột; việc đọc trực tiếp file PDF dung lượng lớn mỗi lần cần tham chiếu gây tràn Context Window và giảm sút độ chính xác. File `.md` cấu trúc hóa cho phép tái sử dụng tối ưu trong suốt quá trình nghiên cứu.

---

## 2. Chuẩn Mực Bàn Giao (Deliverables) Cho Môn Học SE374.R11

Mọi đồ án cần hướng tới bộ sản phẩm hoàn chỉnh:
1. **Bản Đề Cương (Mini Proposal):** Trình bày tổng quan bài toán, mục tiêu, dataset, phương pháp đề xuất và kế hoạch thực hiện theo chuẩn mẫu UIT / Giảng viên.
2. **Slide Báo Cáo Thuyết Trình (16:9):** Soạn bằng Marp CLI / Slidev, cấu trúc cô đọng, có hình ảnh minh họa y khoa trực quan và bảng số liệu so sánh rõ ràng.
3. **Báo Cáo Kỹ Thuật Cuối Kỳ (Final Technical Report):** Soạn theo chuẩn Springer LNCS hoặc mẫu Báo cáo đồ án tốt nghiệp UIT (Khoa Kỹ thuật Phần mềm), đầy đủ cơ sở lý thuyết, kiến trúc chi tiết, phân tích bóc tách (Ablation Study) và phân tích lỗi lâm sàng (Error Analysis).
4. **Mã Nguồn & Web Demo Trực Quan:** Mã nguồn sạch sẽ trên GitHub kèm CI/CD Actions, kèm giao diện Web Demo nhỏ gọn (Streamlit) cho phép bác sĩ / người dùng tải ảnh y tế lên và xem kết quả phân tích / tái tạo.
