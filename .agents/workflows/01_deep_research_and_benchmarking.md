# SOP 01: Deep Research & Benchmarking Workflow

## 1. Mục Tiêu & Phạm Vi
Hướng dẫn Agent thực hiện quy trình nghiên cứu giải pháp kỹ thuật, so sánh benchmark và thu thập tài liệu chuẩn mực khi nhận được yêu cầu công nghệ mới hoặc phức tạp.

---

## 2. Quy Trình Thực Thi

### Bước 1: Xác Định Bài Toán & Từ Khóa Cốt Lõi
- Phân tích yêu cầu của người dùng để xác định:
  - Khái niệm kỹ thuật chính (Core concept).
  - Yêu cầu phi chức năng (Latency, Throughput, Security, Scalability).

### Bước 2: Truy Vấn Tài Liệu Bằng Firecrawl MCP
1. **Tìm kiếm các công trình khoa học / chuẩn đặc tả:**
   - Gọi `firecrawl_research_search_papers` với query rõ ràng.
   - Ví dụ: `firecrawl_research_search_papers(query="OAuth2 PKCE best practices RFC")`
2. **Khảo sát giải pháp thực tế từ GitHub:**
   - Gọi `firecrawl_research_search_github` để xem cách triển khai thực tế.
   - Ví dụ: `firecrawl_research_search_github(query="FastAPI OAuth2 PKCE implementation")`

### Bước 3: Tra Cứu API Specification Bằng Context7
1. Gọi `resolve-library-id` cho các thư viện liên quan.
2. Gọi `query-docs` để kiểm tra phương thức khởi tạo, các tham số bắt buộc và cách xử lý exception.

### Bước 4: Soạn Thảo Đề Xuất Kỹ Thuật (Architecture Proposal)
- Viết tài liệu tổng hợp vào `docs/research/<feature_name>.md`.
- Đính kèm bảng so sánh các giải pháp (Ưu điểm, Nhược điểm, Mức độ phù hợp với dự án).
- Trình bày tóm tắt cho người dùng trước khi triển khai code.
