# SOP 03: Skill & Rule Evolution Workflow

## 1. Mục Tiêu & Phạm Vi
Quy trình giúp hệ thống Agent liên tục học hỏi và tiến hóa trong suốt quá trình phát triển dự án, đảm bảo các quy tắc (Rules) và kỹ năng (Skills) luôn bám sát sự phát triển của codebase.

---

## 2. Quy Trình Thực Thi

### Bước 1: Giám Sát Nhu Cầu Mới
Khi dự án:
- Thêm một module công nghệ mới (ví dụ: thêm Redis cache, thêm Celery worker, tích hợp Stripe thanh toán).
- Xuất hiện lỗi lặp lại nhiều lần (Recurring bugs).
- Được người dùng nhắc nhở hoặc điều chỉnh phong cách code.

### Bước 2: Nạp Thêm Kỹ Năng Bổ Trợ
1. Tra cứu trong `C:\Users\Nguyen Trong Khoi\.gemini\config\skills\` xem có kỹ năng tương ứng chưa.
2. Nạp kỹ năng bằng lệnh copy chuẩn hoặc đọc trực tiếp:
   ```bash
   cmd /c "xcopy /E /I /Y \"C:\Users\Nguyen Trong Khoi\.gemini\config\skills\<new_skill>\" \".agents\skills\<new_skill>\""
   ```

### Bước 3: Chuẩn Hóa Thành Rule Mới
Nếu phát hiện một mẫu thiết kế hoặc quy ước cần tuân thủ nghiêm ngặt:
1. Tạo file `.agents/rules/06_<ten_quy_uoc>.md`.
2. Định nghĩa rõ ràng: Mục đích, Ràng buộc kỹ thuật, Khối so sánh sai/đúng (`diff`), và Lệnh kiểm tra đối kháng.
3. Cập nhật `Task Routing Matrix` trong file `AGENTS.md`.
