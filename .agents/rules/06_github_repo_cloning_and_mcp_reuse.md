---
id: rule-06-github-repo-cloning-and-mcp-reuse
title: GitHub Open-Source Repository Reuse & MCP Integration Protocol
activation: always_on
description: "Quy chuẩn tìm kiếm, clone, bóc tách và tái sử dụng các repository mã nguồn mở xuất sắc trên GitHub và các MCP Server chuyên biệt (thay vì tự viết lại từ đầu)."
---

# Rule 06: GitHub Open-Source Repository Reuse & MCP Integration Protocol

## 1. Mục Đích & Triết Lý "Đứng Trên Vai Người Khổng Lồ"
Trong phát triển phần mềm hiện đại, hầu hết các bài toán chuyên biệt (tạo slide thuyết trình, hiệu ứng UI đỉnh cao, bóc tách PDF, đồ họa 3D, crypto primitives) đều đã có các dự án mã nguồn mở xuất sắc được cộng đồng tối ưu qua hàng năm trời.
- **Quy tắc vàng:** Không bao giờ tự viết lại từ đầu (reinvent the wheel) một tính năng phức tạp nếu trên GitHub đã có các công cụ hoặc repository tiêu chuẩn (Battle-tested Open-Source).
- **Hành động bắt buộc:** Khi nhận một yêu cầu chuyên biệt, Agent phải chủ động tìm kiếm các repository hàng đầu trên GitHub, clone hoặc bóc tách mã nguồn mẫu, và tích hợp vào dự án.

---

## 2. Bảng Danh Mục Các Nhóm Bài Toán & Repository Chuẩn Mực

| Nhóm Nhiệm Vụ | Giải Pháp GitHub & Open-Source Khuyên Dùng | Công Cụ Khảo Sát / Tái Sử Dụng |
| :--- | :--- | :--- |
| **Tạo Slide Thuyết Trình** | `marp-team/marp`, `slidevjs/slidev`, `hakimel/reveal.js`, `gitbrent/PptxGenJS` | Clone template, biên dịch HTML/PDF slides chuyên nghiệp. |
| **Giao Diện & Hiệu Ứng Đỉnh Cao (UI/UX)** | `shadcn-ui/ui`, `magicuidesign/magicui`, `aceternity/ui`, `lucide-icons/lucide` | Copy component Tailwind/React sẵn có, không cần CSS thủ công từ đầu. |
| **Xử Lý & Chuyển Đổi Tài Liệu** | `jgm/pandoc`, `jsvine/pdfplumber`, `ankitrohatgi/tar` | Tái sử dụng pipeline xử lý văn bản, PDF, Word, Excel chuẩn. |
| **Đồ Họa 3D & Animation** | `mrdoob/three.js`, `pmndrs/react-three-fiber`, `framer/motion` | Dùng các scene và animation hook đã được tối ưu hiệu năng. |
| **Mật Mã & Bảo Mật Chuyên Sâu** | `OpenZeppelin/openzeppelin-contracts`, `nucypher/pyUmbral`, `cryptography` | Sử dụng thư viện đã được audit bảo mật, cấm tự chế thuật toán mật mã. |
| **Tự Động Hóa & Web Scraping** | `puppeteer/puppeteer`, `microsoft/playwright`, `crawlee` | Sử dụng Playwright / Puppeteer scripts có sẵn. |

---

## 3. Quy Trình Khảo Sát, Clone & Bóc Tách Repo GitHub

### Bước 1: Tìm Kiếm Repository Hàng Đầu
Sử dụng công cụ `firecrawl_research_search_github` hoặc GitHub MCP:
```json
{
  "query": "<chủ đề> <ngôn ngữ> stars:>500 best practices"
}
```

### Bước 2: Bóc Tách Nhanh Không Cần Tải Nặng
- **Cách 1 (Dùng Repomix):** Gọi `pack_remote_repository` với URL GitHub của repo đó để nén toàn bộ kiến trúc vào 1 tệp Markdown phân tích.
- **Cách 2 (Dùng GitHub MCP):** Gọi `get_file_contents` để đọc trực tiếp file cấu hình hoặc component cụ thể từ repo mà không cần tải cả thư mục.

### Bước 3: Shallow Clone (Nếu cần chạy thử hoặc lấy toàn bộ)
Khi cần lấy tài nguyên hoặc chạy thử công cụ, sử dụng lệnh clone nông (chỉ lấy commit mới nhất, giảm 90% thời gian tải):
```bash
cmd /c "git clone --depth 1 https://github.com/<author>/<repo>.git temp_tools/<repo_name>"
```
Sau khi trích xuất component hoặc chạy xong, dọn dẹp thư mục tạm để tránh làm nặng repo chính.

---

## 4. Tận Dụng Các MCP Server Sẵn Có
- **Figma MCP (`figma_read`, `figma_write`):** Trích xuất thiết kế UI trực tiếp từ file Figma của khách hàng hoặc đề bài.
- **Firecrawl MCP (`firecrawl_scrape`, `firecrawl_crawl`):** Cào tài liệu mẫu hoặc nội dung web phức tạp.
- **Repomix MCP (`pack_codebase`):** Đóng gói nhanh các module tham chiếu để so sánh đối chiếu.

---

## 5. Giao Thuc Tim Kiem Thong Tin Sieu Toc Tu Repositories (`repos/` & `repos/meta_tools/`)

Khi can tim bat ky cong cu, thu vien hoac ma nguon tham khao nao:
1. **Buoc 1 - Tra cuu Index trung tam (`repos/INDEX.md`):** Doc bang ma tran dieu huong de xac dinh ngay repo muc tieu va file can mo.
2. **Buoc 2 - Ripgrep chinh xac (`grep_search`):** Gioi han `SearchPath` vao file `README.md` cua repo can tra (vi du `SearchPath: "repos/meta_tools/awesome-python/README.md"`). Thoi gian tra cuu < 20ms, khong scan toan bo thu muc tranh tran bo nho va lang phi token.
3. **Buoc 3 - Ke thua truc tiep (`view_file`):** Mo dong code mau de tai su dung, tuyet doi khong viet script Python nhap (`INVARIANT-15`).

---

## 6. Lenh Kiem Tra Doi Khang
```bash
# Kiem tra su ton tai cua thu muc cong cu tham chieu hoac vendor
cmd /c "dir /b repos || echo Clean"
```

