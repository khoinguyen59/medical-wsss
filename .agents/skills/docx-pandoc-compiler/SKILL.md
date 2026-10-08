---
name: docx-pandoc-compiler
description: Hướng dẫn quy trình biên dịch tài liệu học thuật từ Markdown sang Word DOCX chuẩn mẫu bằng Pandoc, tự động chuyển đổi công thức toán LaTeX sang chuẩn Word OMML.
---

# DOCX Pandoc Compiler Skill

## Mục đích
Thay thế hoàn toàn việc viết script `python-docx` thủ công 700 dòng bằng công cụ Pandoc tiêu chuẩn công nghiệp.

## Quy Trình Thực Hiện

### Bước 1: Chuẩn Bị File Mẫu Word Tham Chiếu (Reference Docx)
Đặt file mẫu Word chính thức của trường hoặc hội nghị vào thư mục `templates/` (ví dụ `templates/reference_template.docx`). File này quy định sẵn:
* Font chữ (Times New Roman / Calibri / Palatino).
* Kích thước lề trang (Margins: Top/Bottom/Left/Right = 1 inch).
* Header, Footer, số trang, định dạng tiêu đề Heading 1, 2, 3.

### Bước 2: Soạn Thảo Tài Liệu Bằng Markdown
Viết nội dung hoàn toàn bằng Markdown chuẩn:
* Công thức toán viết bằng LaTeX:
  * Inline: `$E = mc^2$`
  * Block: `$$F_{\text{PNS}} = \frac{2 \cdot PN \cdot PS}{PN + PS}$$`
* Bảng biểu viết bằng cú pháp GFM Markdown.

### Bước 3: Lệnh Biên Dịch Pandoc 1 Dòng
Chạy lệnh chuyển đổi trong terminal:
```bash
pandoc input.md --reference-doc=templates/reference_template.docx -o output.docx
```

### Bước 4: Kiểm Tra Định Dạng Đầu Ra
1. Kiểm tra bảng biểu và công thức toán trong Word (đảm bảo công thức là Math Equation gốc, không phải hình ảnh).
2. Kiểm tra ngân sách số trang (Page Budget) để tránh rớt dòng mồ côi sang trang mới.
