---
name: stackoverflow-bug-hunter
description: "Kỹ năng truy vết và xử lý bug chuyên sâu bằng cách tra cứu StackOverflow, GitHub Issues và các diễn đàn lập trình viên bằng chữ ký lỗi thực tế."
---

# StackOverflow Bug Hunter Skill

Kỹ năng này cung cấp phương pháp luận giải quyết triệt để các lỗi khó (Stubborn Bugs, Build Errors, Type Mismatches, Deadlocks) bằng cách khai thác hàng triệu giải pháp đã được chứng thực trên **StackOverflow** và **GitHub Issues**.

## 1. Khi Nào Kích Hoạt Kỹ Năng Này?
- Khi gặp lỗi biên dịch hoặc runtime sau 1 lần tự sửa mà vẫn không thành công.
- Khi gặp lỗi đặc thù của framework (ví dụ: Next.js hydration, React hooks dependency loop, Hardhat revert without reason, Docker network issue).
- Khi gặp lỗi version mismatch hoặc deprecated method giữa các phiên bản thư viện.

---

## 2. Quy Trình Truy Vết Lỗi 3 Bước

### Bước 1: Trích Xuất Chữ Ký Lỗi Độc Nhất (Sanitizing Error)
- Lọc bỏ đường dẫn tệp máy tính người dùng (`C:\Users\...`) và các số dòng ngẫu nhiên.
- Giữ lại: Tên loại lỗi + Đoạn thông điệp lỗi trọng tâm + Tên framework/thư viện.
- **Ví dụ:**
  - Gốc: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! at line 42`
  - Chữ ký tìm kiếm: `"Expected all tensors to be on the same device" "cuda" "pytorch"`

### Bước 2: Tìm Kiếm Với Toán Tử Chính Xác Qua Công Cụ Có Sẵn
Sử dụng trực tiếp công cụ `search_web` với toán tử `site:stackoverflow.com` hoặc GitHub MCP:
```python
# Cú pháp tìm StackOverflow trực tiếp bằng search_web
search_web(query='site:stackoverflow.com "Expected all tensors to be on the same device" "pytorch"')
```
*Tra cứu GitHub Issues:* Sử dụng `call_mcp_tool` với server `github`, tool `search_issues` hoặc `search_web` với `site:github.com/<org>/<repo>/issues`.

### Bước 3: Phân Tích & Chọn Bản Vá Chuẩn
- Chọn câu trả lời có dấu tích xanh (Accepted Answer) hoặc câu trả lời có số upvotes cao nhất.
- Đọc phần giải thích "Why this happens" (Tại sao lỗi xảy ra).
- Áp dụng thay đổi vào mã nguồn và chạy lại test để xác nhận.

---

## 3. Tài Liệu Tham Khảo Trong Kỹ Năng
- [Mẫu cú pháp truy vấn StackOverflow theo từng ngôn ngữ/lỗi](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/stackoverflow-bug-hunter/references/debugging_search_patterns.md)
