---
name: autonomous-deep-researcher
description: "Kỹ năng nghiên cứu công nghệ độc lập: khai thác Firecrawl MCP (tìm kiếm paper học thuật, GitHub benchmark repos, crawl docs), Context7 (tra cứu API specs chính thức), và Repomix."
---

# Autonomous Deep Researcher Skill

Kỹ năng này hướng dẫn Agent cách thực hiện nghiên cứu kỹ thuật chuyên sâu trước khi bắt đầu lập trình hoặc thiết kế kiến trúc trong bất kỳ dự án nào.

## 1. Khi Nào Kích Hoạt Kỹ Năng Này?
- Khi tiếp nhận một dự án mới và cần tìm hiểu các chuẩn công nghệ tốt nhất.
- Khi gặp thư viện, framework hoặc thuật toán chưa từng sử dụng trước đây.
- Khi cần tra cứu chuẩn giao thức (RFC, IEEE, NIST, OpenZeppelin, GA4GH).
- Khi cần tìm kiếm các repository mã nguồn mở chuẩn mực (Gold Standard Repositories) để học hỏi mô hình kiến trúc.

---

## 2. Quy Trình Nghiên Cứu 4 Bước

### Bước 1: Tra cứu Chuẩn Học Thuật & Công Nghiệp (Academic & Industry Search)
Sử dụng công cụ `firecrawl_research_search_papers`:
```json
{
  "query": "<chủ đề cần nghiên cứu + công nghệ>",
  "limit": 5
}
```
Trích xuất:
- Các thuật toán cốt lõi đã được peer-reviewed.
- Các tiêu chuẩn dữ liệu và định dạng payload chính thức.
- Các khuyến nghị bảo mật và phương thức phòng ngừa rủi ro.

### Bước 2: Tải Bài Báo Về Máy & Dùng Vision Chuyển Hóa Thành Markdown (Vision-to-Markdown)
- **Quy tắc bắt buộc:**
  1. Tải toàn văn bài báo (PDF) về lưu tại `docs/papers/<paper_id>.pdf`.
  2. Sử dụng tính năng Vision / đọc trực quan đa phương thức để duyệt từng trang PDF, nhận diện chính xác sơ đồ kiến trúc khối, bảng số liệu thực nghiệm, và công thức toán học.
  3. Viết lại một file Markdown đối ứng `docs/papers/<paper_id>.md` chứa toàn văn nội dung cấu trúc hóa chi tiết của bài báo.
- **Lý do kỹ thuật tối thượng:**
  - AI đọc PDF dạng ảnh hoặc cột đôi rất kém, thường xuyên bị sót ký hiệu toán học, tráo đổi thứ tự cột hoặc mất dữ liệu bảng.
  - Việc đọc đi đọc lại file PDF dung lượng lớn mỗi khi cần tra cứu làm lãng phí Context Window và thiếu hiệu quả khi tái sử dụng.
  - Một file Markdown cấu trúc sạch sẽ giúp AI Agent đọc lại tức thì với chi phí token tối thiểu, dễ dàng trích dẫn khi viết proposal và báo cáo.

### Bước 3: Khảo Sát Kho Mã Nguồn Đối Chuẩn (GitHub Benchmark Repos)
Sử dụng công cụ `firecrawl_research_search_github`:
```json
{
  "query": "<tên công nghệ + architecture/starter/best-practices>",
  "limit": 5
}
```
Phân tích:
- Cách tổ chức thư mục của các repo top-starred.
- Cách thiết lập file cấu hình (`pyproject.toml`, `settings.py`, `environment.yml`).
- Các thư viện bổ trợ phổ biến nhất (de-facto standards).

### Bước 4: Tra Cứu Tài Liệu API Chính Thức (Context7 Official Docs)
Gọi công cụ Context7:
1. `resolve-library-id`: Tìm library ID chính xác.
2. `query-docs`: Lấy tài liệu trực tiếp về các hàm/module cụ thể.

### Bước 5: Lưu Trữ Bền Vững Tri Thức
Tổng hợp tài liệu thu thập được vào `docs/research/<topic_name>.md` hoặc `docs/papers/<paper_name>.md` để toàn bộ team và các agent khác có thể tham khảo lại mà không cần tốn token tra cứu lại từ đầu.

---

## 3. Tài Liệu Tham Khảo Trong Kỹ Năng
- [Hồ sơ sử dụng Firecrawl MCP](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/autonomous-deep-researcher/references/firecrawl_mcp_cookbook.md)
- [Mẫu tra cứu Context7 API](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/autonomous-deep-researcher/references/context7_api_patterns.md)
- [Chuẩn trích xuất bài báo khoa học](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/autonomous-deep-researcher/references/academic_paper_extraction.md)
