# QUY CHUẨN KIỂM SOÁT CHẤT LƯỢNG ẤN PHẨM HỌC THUẬT & HÌNH ẢNH KHOA HỌC
## Standard Specification: Publication-Grade Visuals & Independent Quality Gate

> **Mã quy chuẩn:** `RULE-13-DELIVERABLE-QUALITY-GATE`  
> **Áp dụng cho:** Toàn bộ đề cương Proposal (`product/proposal/`), Báo cáo đồ án (`product/reports/`), Slide thuyết trình (`product/slides/`), và Tài liệu nghiên cứu (`docs/`).

---

### 1. QUY TẮC BẤT BIẾN VỀ HÌNH ẢNH KHOA HỌC (SCIENTIFIC VISUALS INVARIANT)

1. **Cấm tuyệt đối ảnh chụp đời thường / ảnh điện thoại / chụp máy chiếu:**
   - TUYỆT ĐỐI NGHIÊM CẤM đưa ảnh chụp từ camera điện thoại, ảnh chụp màn hình máy chiếu lớp học mờ, méo mó hoặc chứa yếu tố ngoại cảnh vào báo cáo/proposal.
   - Bất kỳ sơ đồ nào do Giảng viên hoặc bài báo cung cấp dưới dạng ảnh chụp/ảnh thô BẮT BUỘC phải được vẽ lại bằng công cụ đồ họa chuyên nghiệp (Matplotlib, Draw.io, TikZ, Figma, Illustrator).
2. **Tiêu chuẩn kỹ thuật cho hình ảnh khoa học:**
   - Định dạng: Đồ họa vector (SVG) hoặc ảnh raster độ nét cao (PNG với độ phân giải $\ge 300$ DPI, chiều rộng tối thiểu 1200 px).
   - Màu sắc: Phối màu chuyên nghiệp chuẩn y khoa (H&E pink/purple, deep blue, clean green/orange), nền trắng thuần `#FFFFFF` hoặc trong suốt.
   - Kiểu chữ: Rõ ràng, sans-serif hiện đại, kích thước tối thiểu 8pt để đọc được khi in ra giấy A4.
   - Chú thích (Caption): Bắt buộc có chú thích khoa học đầy đủ ("Hình X: ... / Fig. X: ...") giải thích rõ các khối chức năng.

---

### 2. CƠ CHẾ VẬN HÀNH CỦA HỆ THỐNG KIỂM SOÁT ĐỘC LẬP (`deliverable_auditor.py`)

Hệ thống `src/utils/deliverable_auditor.py` kiểm tra tự động 4 tầng trước mọi lần nộp bài:

1. **Tầng 1 - Kiểm soát Chất lượng Hình ảnh (Visual & Image Gate):**
   - Quét metadata EXIF: Phát hiện và chặn tức thì ảnh có thông tin máy ảnh (Apple, Samsung, Canon...).
   - Quét mẫu tên tệp: Chặn các tệp có tiền tố ảnh thô (`media_*`, `IMG_*`, `PXL_*`, `screenshot_*`).
   - Quét độ phân giải: Cảnh báo ảnh dưới $600 \times 300$ px.
   - Kiểm tra chú thích: Chặn hình ảnh có caption trống `![]()`.
2. **Tầng 2 - Kiểm soát Chính tả, Mã ký tự & Typography (Typography Gate):**
   - Quét dấu gạch nối: Chặn em-dash `—` (U+2014) và en-dash `–` (U+2013), chỉ cho phép ASCII `-` (0x2D).
   - Quét vỡ font: Phát hiện lỗi mã hóa UTF-8 (mojibake).
   - Quét lỗi dấu câu: Phát hiện lặp dấu `..`, `,,`, `??`.
3. **Tầng 3 - Kiểm soát Liêm chính Học thuật (Zero-Mockup Gate):**
   - Quét từ khóa giả lập: Chặn `lorem ipsum`, `dummy metric`, `placeholder data`.
   - Kiểm tra công thức toán: Đảm bảo cân bằng cặp dấu `$`.
4. **Tầng 4 - Kiểm soát Định dạng Văn bản (Structure Gate):**
   - Kiểm tra cấu trúc tiêu đề: Đảm bảo không nhảy cóc thứ bậc (H1 -> H2 -> H3).
   - Kiểm tra bảng biểu: Đảm bảo bảng Markdown hợp lệ.

---

### 3. QUY TRÌNH THỰC THI (VERIFICATION WORKFLOW)

- Trước khi nghiệm thu bất kỳ tài liệu nào trong `product/`, bắt buộc chạy lệnh:
  ```bash
  python -m src.utils.deliverable_auditor <duong_dan_file_markdown>
  ```
- Toàn bộ bài kiểm tra phải trả về `[DELIVERABLE QUALITY AUDIT: PASSED]` với `0 CRITICAL ISSUES`.
