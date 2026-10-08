# SOP 05: GitHub Repository Cloning & Open-Source Reuse Workflow

## 1. Mục Tiêu & Phạm Vi
Quy trình hướng dẫn Agent cách tìm kiếm, clone nông (shallow clone) và tái sử dụng các repository chất lượng cao trên GitHub cho các tính năng cụ thể (làm slide, hiệu ứng UI, bóc tách tài liệu, crypto) mà không cần viết lại từ đầu.

---

## 2. Quy Trình Thực Thi Chi Tiết

```
[NHẬN YÊU CẦU TÍNH NĂNG ĐẶC THÙ]
(Ví dụ: tạo slide, giao diện animation, bộ parser)
              │
              ▼
[BƯỚC 1: TÌM KIẾM REPO MÃ NGUỒN MỞ XUẤT SẮC]
- Gọi firecrawl_research_search_github hoặc GitHub MCP:
  Query: "<chủ_đề> <ngôn ngữ> stars:>500"
- Chọn repo uy tín, có giấy phép MIT/Apache 2.0, cộng đồng bảo trì tốt.
              │
              ▼
[BƯỚC 2: KHẢO SÁT & BÓC TÁCH MÃ NGUỒN]
- Nếu chỉ cần đọc vài file: Dùng GitHub MCP get_file_contents.
- Nếu cần phân tích sâu cấu trúc: Dùng Repomix pack_remote_repository.
- Nếu cần chạy thử CLI hoặc lấy cả thư mục: 
  cmd /c "git clone --depth 1 <repo_url> temp_tools/<repo_name>"
              │
              ▼
[BƯỚC 3: TÁI SỬ DỤNG & TÍCH HỢP VÀO DỰ ÁN CHÍNH]
- Sao chép các component / template cần thiết vào src/ hoặc tools/.
- Cài đặt dependency tương ứng vào package.json hoặc requirements.txt.
- Tùy biến mã nguồn cho phù hợp với nghiệp vụ của dự án.
              │
              ▼
[BƯỚC 4: DỌN DẸP TÀI NGUYÊN TẠM]
- Xóa thư mục clone tạm: cmd /c "rmdir /s /q temp_tools"
- Chạy kiểm thử để xác nhận module tích hợp hoạt động hoàn hảo.
```

---

## 3. Ví Dụ Cụ Thể

### Ví Dụ A: Yêu Cầu Tạo Slide Thuyết Trình Từ Báo Cáo / Đề Bài
1. Tìm repo: `marp-team/marp-cli` hoặc `slidevjs/slidev`.
2. Dùng CLI: Cài đặt npm package `npm install -D @marp-team/marp-cli`.
3. Soạn nội dung Markdown với directive Marp (theme gaia, uncover...).
4. Biên dịch: `npx @marp-team/marp-cli presentation.md --html --pdf`.

### Ví Dụ B: Yêu Cầu Giao Diện Đẳng Cấp Cao (UI/UX)
1. Tham chiếu repo: `shadcn-ui/ui` hoặc `magicuidesign/magicui`.
2. Lấy component mẫu (Card, Button, Bento Grid, Animated List) tích hợp trực tiếp vào dự án React/Next.js.
3. Chạy dev server và kiểm tra giao diện mượt mà.
