---
name: graphify-codebase-mapper
description: "Kỹ năng lập bản đồ cây AST mã nguồn, dựng Đồ Thị Tri Thức kiến trúc (Knowledge Graph), tìm kiếm quan hệ phụ thuộc liên tệp và xác định các God Nodes bằng Graphify."
---

# Graphify Codebase Mapper Skill

Kỹ năng này cung cấp quy trình và các câu lệnh chuẩn xác nhất để Agent khai thác sức mạnh của công cụ **Graphify**, biến codebase thành đồ thị AST trực quan, bảo toàn trí nhớ kiến trúc qua các phiên làm việc.

## 1. Khi Nào Kích Hoạt Kỹ Năng Này?
- Ngay khi tiếp nhận một dự án mới (Pha 3 của Bootstrap Protocol).
- Trước khi thực hiện tái cấu trúc (Refactoring) mã nguồn quan trọng.
- Khi cần tìm hiểu luồng gọi hàm giữa Controller -> Service -> Repository -> Database.
- Sau khi hoàn thành việc thêm tính năng mới để cập nhật đồ thị tri thức.

---

## 2. Quy Trình Vận Hành 3 Bước

### Bước 1: Trích Xuất Cây AST Ban Đầu
Chạy lệnh phân tích tĩnh toàn bộ mã nguồn:
```bash
cmd /c "graphify extract . --code-only"
```
*Lưu ý:* Cờ `--code-only` giúp phân tích cú pháp AST cục bộ cực nhanh mà không cần API key LLM.

### Bước 2: Phân Tích Hub Kiến Trúc (God Nodes)
Xác định 10 tệp/lớp có liên kết nhiều nhất trong toàn bộ hệ thống:
```bash
cmd /c "graphify god-nodes --top 10"
```
Các node xuất hiện trong danh sách này là các điểm nhạy cảm, phải cẩn trọng tối đa khi chỉnh sửa.

### Bước 3: Tìm Đường Đi Quan Hệ (Path Analysis)
Khi cần biết Module A gọi Module B như thế nào:
```bash
cmd /c "graphify path \"ModuleA\" \"ModuleB\""
```

---

## 3. Cập Nhật Đồ Thị Sau Khi Sửa Đổi
```bash
cmd /c "graphify update ."
```

---

## 4. Tài Liệu Tham Khảo Trong Kỹ Năng
- [Cheatsheet các lệnh Graphify CLI](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/graphify-codebase-mapper/references/graphify_cli_cheatsheet.md)
- [Cookbook truy vấn AST nâng cao](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/graphify-codebase-mapper/references/ast_query_cookbook.md)
