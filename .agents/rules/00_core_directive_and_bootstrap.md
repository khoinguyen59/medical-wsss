---
id: rule-00-core-directive-and-bootstrap
title: Core Directive & Dual-Engine Bootstrap Protocol
activation: always_on
description: "Quy chuẩn chỉ thị tối cao và quy trình tự kích hoạt Bootstrap khi Agent tiếp quản dự án: hỗ trợ cả dự án mới toanh chỉ có đề bài/yêu cầu (Greenfield Spec-Driven) và dự án đã có sẵn code (Brownfield AST)."
---

# Rule 00: Core Directive & Dual-Engine Bootstrap Protocol

## 1. Mục Đích & Nguyên Tắc Tối Cao
Rule này quy định hành vi khởi đầu của Agent khi được sao chép vào bất kỳ thư mục dự án nào.
- **Trường hợp phổ biến:** Dự án mới nhận thường **chỉ là một bản đề bài, rubric, tệp tài liệu yêu cầu (`.pdf`, `.docx`, `.md`, `.txt`) hoặc một đoạn prompt yêu cầu của người dùng**, hoàn toàn chưa có bất kỳ dòng code hay file cấu hình nào!
- **Nguyên tắc vàng:** Agent không được lúng túng hoặc dừng lại khi thấy thư mục rỗng. Agent phải tự động trở thành **Chuyên Gia Phân Rã Đề Bài**, chủ động đọc tài liệu, dùng Firecrawl và Context7 nghiên cứu giải pháp, và tự động tạo dựng toàn bộ khung dự án (Scaffolding) từ đầu đến cuối.

---

## 2. Quy Trình Khởi Động Kép (Dual-Engine Bootstrap)

Ngay khi được khởi động trong thư mục mới, Agent tự động phân loại và thực thi:

### 🟢 Nhánh A: Dự Án Mới Toanh (Chỉ có Đề Bài / Yêu Cầu / Prompt)

1. **Bước 1: Quét & Đọc Tài Liệu Đề Bài:**
   - Tìm kiếm các tệp tài liệu: `*.pdf`, `*.docx`, `*.md`, `*.txt`, `docs/`, `specs/`.
   - Nếu là `.md` hoặc `.txt`: Đọc trực tiếp bằng công cụ `view_file`.
   - Nếu là `.docx` hoặc `.pdf`: Dùng script Python trích xuất văn bản sạch.
   - Bóc tách: Mục tiêu dự án, Danh sách chức năng bắt buộc, Đối tượng người dùng (Actors/Roles), Tiêu chuẩn chấm điểm (Rubric).

2. **Bước 2: Tự Động Nghiên Cứu Chuyên Sâu (Autonomous Deep Research):**
   - **Tìm paper học thuật & tiêu chuẩn:** Gọi `firecrawl_research_search_papers` với chủ đề của đề bài.
   - **Tìm repo tham chiếu tương tự:** Gọi `firecrawl_research_search_github` với từ khóa đề bài để xem các dự án điểm cao/chuẩn mực thiết kế kiến trúc ra sao.
   - **Tra cứu thư viện tối ưu:** Gọi `context7` lấy API docs chính xác cho các thư viện dự kiến sử dụng.

3. **Bước 3: Tự Động Dựng Khung Dự Án Chuẩn (Zero-to-One Scaffolding):**
   - Tạo các file manifest chuẩn: `package.json`, `requirements.txt`, `pyproject.toml`, hoặc `Cargo.toml`.
   - Thiết lập cấu trúc thư mục phân tầng (Clean Architecture: Controller -> Service -> Repository -> Model).
   - Thiết lập môi trường cấu hình: `.env.example`, `.gitignore`, `README.md`.

4. **Bước 4: Nạp Kỹ Năng Đỉnh Cao Tương Ứng & Graphify:**
   - Nạp kỹ năng phù hợp từ `C:\Users\Nguyen Trong Khoi\.gemini\config\skills\` vào `.agents/skills/`.
   - Chạy `cmd /c "graphify extract . --code-only"` để thiết lập bản đồ AST khởi đầu.

5. **Bước 5: Tự Tối Ưu Hóa AGENTS.md Thành Chuyên Gia Đề Bài:**
   - Đổi tên Agent gắn liền với đề bài (ví dụ: `SmartLogisticsAgent`, `GenomicsAgent`...).
   - Đưa các tiêu chí trong đề bài thành các **System Invariants** bắt buộc tuân thủ.

---

### 🔵 Nhánh B: Dự Án Đã Có Mã Nguồn (Brownfield)

1. Quét manifest (`package.json`, `pyproject.toml`...).
2. Dựng đồ thị AST bằng `cmd /c "graphify extract . --code-only"`.
3. Phân tích các God Nodes và nạp kỹ năng bổ trợ tương ứng.

---

## 3. Anti-Pattern vs Positive Pattern

```diff
- // ANTI-PATTERN: Thấy thư mục chỉ có file "De_bai.pdf", agent bảo "Dự án trống, hãy cho tôi biết bạn muốn làm gì"
- // Hậu quả: Thụ động, bắt người dùng phải tóm tắt lại những gì đề bài đã ghi rất rõ.

+ // POSITIVE PATTERN: Tự động phát hiện "De_bai.pdf", đọc toàn bộ nội dung, trích xuất rubric,
+ // Dùng Firecrawl tìm repo tham chiếu, tự đề xuất kiến trúc và dựng sẵn khung source code hoàn chỉnh!
```
