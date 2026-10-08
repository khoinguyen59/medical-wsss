---
name: slide-marp-generator
description: Hướng dẫn tạo slide thuyết trình chuyên nghiệp tỷ lệ 16:9 từ Markdown bằng Marp hoặc Slidev, xuất ra cả PowerPoint PPTX, PDF và HTML trình chiếu.
---

# Slide Marp & Slidev Generator Skill

## Mục đích
Tạo slide báo cáo tiến độ, pitching 5 phút và bảo vệ đồ án chuẩn tỷ lệ 16:9 trực tiếp từ Markdown, loại bỏ việc chỉnh sửa thủ công trên phần mềm PowerPoint.

## Quy Trình Thực Hiện

### Bước 1: Soạn Thảo Slide Bằng Markdown (Marp Format)
Tạo file `slides.md` với frontmatter chuẩn:
```markdown
---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #f5f5f5
color: #333333
---

# Tên Đề Tài / Bài Thuyết Trình
### Báo Cáo Tiến Độ / Pitching 5 Phút

---

## 1. Bài Toán & Động Lực Nghiên Cứu
* Vấn đề thực tế cần giải quyết.
* Điểm hạn chế của các phương pháp hiện có.

---

## 2. Kết Quả Thực Nghiệm Đối Chứng
![bg right:40% 80%](images/chart_result.png)
* Baseline vs Proposed Method
* Cải thiện hiệu năng rõ rệt
```

### Bước 2: Lệnh Xuất Bản Đa Định Dạng
1. Xuất ra file PowerPoint PPTX để nộp/chỉnh sửa:
   ```bash
   marp slides.md --pptx -o slides.pptx
   ```
2. Xuất ra file PDF chất lượng cao để in ấn:
   ```bash
   marp slides.md --pdf -o slides.pdf
   ```
3. Xuất ra HTML để thuyết trình trực tiếp trên trình duyệt:
   ```bash
   marp slides.md -o slides.html
   ```

### Bước 3: Nguyên Tắc Thiết Kế Slide Pitching
* **Quy tắc 1 slide / 1 phút:** Pitch 5 phút chỉ dùng đúng 5 - 6 slides chính.
* **Quy tắc 5x5:** Không quá 5 ý mỗi slide, không quá 5 - 7 từ mỗi dòng.
* **Trực quan hóa:** Dùng biểu đồ kết quả, ảnh kiến trúc, sơ đồ dòng chảy thay cho các đoạn văn dài.
