---
name: ast-graphify-mapper
description: Hướng dẫn quản trị và kiểm soát đồ thị tri thức mã nguồn (AST Knowledge Graph) bằng Graphify CLI, phát hiện các điểm nút trung tâm (God-Nodes) và ngăn ngừa lỗi nhập vòng (Circular Dependencies).
---

# AST Graphify Mapper Skill

## Mục đích
Giúp AI Agent nắm bắt toàn bộ kiến trúc mã nguồn của dự án thông qua đồ thị tri thức AST mà không làm nghẽn bộ nhớ ngữ cảnh (Context Window).

## Hai Profile Đồ Thị Chuẩn Hóa
1. **Profile `main` (`graphify-out/`):** Chỉ quét thư mục mã nguồn chính (`src/`), tập trung vào kiến trúc các service.
2. **Profile `all` (`graphify-out-all/`):** Quét toàn bộ codebase (bao gồm cả `experiments/`, `tests/`).

## Lệnh Cập Nhật Đồ Thị
Sau mỗi lần chỉnh sửa kiến trúc lớn hoặc thêm module mới, chạy script:
```bash
python scripts/update_graphify.py --profile both
```

## Cách AI Khai Thác Đồ Thị AST
1. **Tìm các điểm nút trung tâm (Architectural Hubs):**
   ```bash
   graphify god-nodes --graph graphify-out/graph.json --top 10
   ```
2. **Kiểm tra vòng lặp phụ thuộc (Circular Imports):**
   * Nếu có hai module trỏ chéo lẫn nhau, Graphify sẽ cảnh báo để Agent tách Data Transfer Object (DTO) độc lập.
3. **Mở cây trực quan D3.js:**
   * Mở file `graphify-out/GRAPH_TREE.html` trên trình duyệt để quan sát cấu trúc cây phân cấp trực quan.
