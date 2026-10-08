---
id: rule-02-graphify-ast-knowledge-graph
title: Codebase AST Knowledge Graph Mapping & Query Standards (Graphify)
activation: always_on
description: "Quy chuẩn bắt buộc về việc lập bản đồ cây AST mã nguồn bằng Graphify, xác định God Nodes, duy trì tri thức liên tệp và cập nhật liên tục đồ thị tri thức tại graphify-out/."
---

# Rule 02: Codebase AST Knowledge Graph Mapping (Graphify)

## 1. Mục Đích & Nguyên Tắc Vận Hành
Mọi codebase phức tạp đều có mạng lưới quan hệ đa chiều giữa các tệp, hàm, lớp và cấu trúc dữ liệu.
- Thay vì đọc mã nguồn rời rạc, tốn token và dễ sót dependency, Agent **bắt buộc sử dụng Graphify** để biến mã nguồn thành Đồ Thị Tri Thức (Knowledge Graph AST).
- Đồ thị tri thức này được lưu trữ bền vững tại thư mục `graphify-out/` và là nguồn sự thật kiến trúc (Architectural Ground Truth).

---

## 2. Quy Trình Khởi Tạo & Đồng Bộ Đồ Thị

### 2.1. Khởi Tạo Lần Đầu (Initial Extraction)
Khi tiếp quản dự án mới, Agent chạy lệnh:
```bash
# Trích xuất toàn bộ cây AST mã nguồn không phụ thuộc vào LLM API key
cmd /c "graphify extract . --code-only"
```
Lệnh này phân tích AST của Python, JavaScript, TypeScript, Go, Rust, Java, C/C++... và xuất ra:
- `graphify-out/graph.json`: Toàn bộ các đỉnh (nodes) và cạnh (edges - imports, calls, extends).
- `graphify-out/GRAPH_REPORT.md`: Báo cáo kiến trúc tổng quan.

### 2.2. Đồng Bộ Liên Tục (Continuous Sync)
Sau khi thực hiện thêm mới tệp, refactor hoặc thay đổi chữ ký hàm (function signatures), Agent bắt buộc chạy:
```bash
cmd /c "graphify update ."
```

---

## 3. Tiêu Chuẩn Khai Thác Đồ Thị AST (Architecture Query SOP)

1. **Phân tích Kiến Trúc Trung Tâm (God Nodes):**
   - Chạy lệnh tìm các thành phần có nhiều kết nối nhất (Hubs):
     ```bash
     cmd /c "graphify god-nodes --top 10"
     ```
   - Đây là các module tối quan trọng, bất kỳ thay đổi nào tại đây đều có nguy cơ gây ảnh hưởng dây chuyền.

2. **Tìm Luồng Tương Tác Giữa 2 Module (Path Query):**
   ```bash
   cmd /c "graphify path \"ModuleA\" \"ModuleB\""
   ```

3. **Truy Vấn Giải Thích Module Cụ Thể:**
   ```bash
   cmd /c "graphify explain \"ClassNameOrFunction\""
   ```

---

## 4. Anti-Pattern vs Positive Pattern

```diff
- // ANTI-PATTERN: Thay đổi code của file Controller nhưng không biết Service nào bị ảnh hưởng
- // Hậu quả: Gây ra lỗi ngầm (silent breakage) ở các module phụ thuộc.

+ // POSITIVE PATTERN: Kiểm tra quan hệ AST qua Graphify trước khi sửa đổi
+ cmd /c "graphify path \"AuthController\" \"DatabaseService\""
+ cmd /c "graphify update ."
```

---

## 5. Lệnh Kiểm Tra Đối Kháng
```bash
# Kiểm tra sự tồn tại của đồ thị AST
cmd /c "dir /b graphify-out\graph.json"
```
