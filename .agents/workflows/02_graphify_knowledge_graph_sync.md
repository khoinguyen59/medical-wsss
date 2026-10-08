# SOP 02: Graphify Knowledge Graph Synchronization & Query Workflow

## 1. Mục Tiêu & Phạm Vi
Quy trình hướng dẫn Agent cách quản lý, đồng bộ và truy vấn Đồ thị Tri thức AST bằng công cụ Graphify trong suốt vòng đời phát triển dự án.

---

## 2. Quy Trình Thực Thi

### Bước 1: Khởi Tạo Đồ Thị Lần Đầu
Nếu dự án chưa có thư mục `graphify-out/`:
```bash
cmd /c "graphify extract . --code-only"
```
Kết quả sẽ sinh ra:
- `graphify-out/graph.json`
- `graphify-out/GRAPH_REPORT.md`

### Bước 2: Truy Vấn Kiến Trúc Khi Tiếp Nhận Yêu Cầu Mới
1. **Tìm kiếm các Hub trung tâm (God Nodes):**
   ```bash
   cmd /c "graphify god-nodes --top 10"
   ```
2. **Tìm đường dẫn phụ thuộc giữa 2 tệp/module:**
   ```bash
   cmd /c "graphify path \"<FileA>\" \"<FileB>\""
   ```
3. **Phân tích chi tiết một thành phần:**
   ```bash
   cmd /c "graphify explain \"<ComponentName>\""
   ```

### Bước 3: Cập Nhật Đồ Thị Sau Khi Sửa Đổi Code
Bắt buộc chạy sau mỗi lần thêm tệp mới, refactor hoặc thay đổi cấu trúc hàm:
```bash
cmd /c "graphify update ."
```

---

## 3. Danh Mục Kiểm Tra Nghiệm Thu (Acceptance Checklist)
- [ ] File `graphify-out/graph.json` tồn tại và không bị rỗng.
- [ ] Không có tệp mã nguồn mới nào bị bỏ sót trong đồ thị tri thức.
