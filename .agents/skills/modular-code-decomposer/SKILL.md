---
name: modular-code-decomposer
description: "Kỹ năng phân rã kiến trúc module, chia nhỏ các chức năng phức tạp thành các file độc lập và kiểm soát nghiêm ngặt giới hạn 400 dòng code cho mỗi tệp."
---

# Modular Code Decomposer Skill

Kỹ năng này chịu trách nhiệm giám sát độ phức tạp và kích thước của mã nguồn trong toàn bộ dự án, thực hiện phân rã các lớp và hàm theo nguyên tắc Single Responsibility, bảo đảm không một file nào vượt quá 400 dòng code.

## 1. Khi Nào Kích Hoạt Kỹ Năng Này?
- Khi tạo mới cấu trúc thư mục cho dự án (thiết lập các tầng modular).
- Khi viết mã nguồn và nhận thấy một file đang vượt qua ngưỡng 300 dòng code.
- Trước khi bàn giao sản phẩm: Chạy script kiểm tra dòng code toàn dự án.

---

## 2. Quy Trình Phân Rã 3 Bước (Refactoring SOP)

### Bước 1: Quét Toàn Bộ File Mã Nguồn
Chạy script kiểm tra dòng code:
```bash
cmd /c "python .agents\skills\modular-code-decomposer\scripts\check_file_lines.py src"
```
Script sẽ in ra danh sách các file vượt quá 400 dòng (nếu có) và đánh dấu đỏ (FAILED).

### Bước 2: Kỹ Thuật Tách File (Extraction Patterns)
- **Tách Data Validation:** Đưa các hàm kiểm tra dữ liệu đầu vào ra `validators/<feature>Validator.py`.
- **Tách Business Rules:** Đưa các thuật toán tính toán, logic điều kiện sang `services/<feature>Rules.py`.
- **Tách Database Access:** Đưa các câu lệnh SQL/Prisma/Mongoose sang `repositories/<feature>Repo.py`.
- **Tách UI Component:** Tách các sub-component (Header, List, Card, Dialog) thành các tệp riêng trong `components/`.

### Bước 3: Kiểm Tra Lại & Cập Nhật Đồ Thị AST
- Chạy lại script `check_file_lines.py` để đảm bảo 100% file < 400 dòng.
- Chạy `cmd /c "graphify update ."` để đồ thị AST ghi nhận cấu trúc module mới.

---

## 3. Tài Liệu Tham Khảo Trong Kỹ Năng
- [Các mẫu phân rã module kinh điển](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/modular-code-decomposer/references/decomposition_patterns.md)
