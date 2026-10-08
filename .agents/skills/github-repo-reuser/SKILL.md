---
name: github-repo-reuser
description: "Kỹ năng tìm kiếm, shallow clone, bóc tách và tái sử dụng các repository mã nguồn mở xuất sắc trên GitHub (như làm slide, làm UI/UX cao cấp, converter, 3D animations) và kết nối các công cụ MCP."
---

# GitHub Repo Reuser Skill

Kỹ năng này hướng dẫn Agent cách tìm kiếm, clone nông (shallow clone) và tái sử dụng các công cụ mã nguồn mở hàng đầu trên GitHub cho các bài toán chuyên biệt mà không cần lập trình lại từ đầu.

## 1. Khi Nào Sử Dụng Kỹ Năng Này?
- Khi cần tạo **Slide thuyết trình ấn tượng**: Dùng `marp`, `slidev`, hoặc `reveal.js`.
- Khi cần xây dựng **Giao diện người dùng (UI/UX) đỉnh cao**: Dùng các component mẫu từ `shadcn/ui`, `magicui`, `aceternity/ui`.
- Khi cần **Chuyển đổi hoặc bóc tách tài liệu phức tạp**: Tái sử dụng các script từ các repo parsing uy tín.
- Khi cần **Mô phỏng hoặc thực hiện thuật toán mật mã phức tạp**: Tái sử dụng code mẫu từ OpenZeppelin, NuCypher TACo, PyUmbral.

---

## 2. Quy Trình Vận Hành 3 Bước

### Bước 1: Khảo Sát Repo Trên GitHub
Sử dụng GitHub MCP hoặc Firecrawl:
- Tìm repo có trên 500+ stars, có tài liệu README rõ ràng, cập nhật gần đây.
- Ví dụ tìm công cụ tạo slide:
  `firecrawl_research_search_github(query="markdown presentation slides CLI open source")`

### Bước 2: Clone Nông Bằng Git CLI Chuẩn (Shallow Clone)
Sử dụng trực tiếp lệnh Git CLI chuẩn của hệ thống (luôn dùng cờ `--depth 1` để tải cực nhanh trong vài giây mà không cần script phụ):
```bash
cmd /c "git clone --depth 1 https://github.com/marp-team/marp-cli.git temp_tools/marp-cli"
```

### Bước 3: Trích Xuất & Tích Hợp
- Trích xuất các component, template, hoặc file cấu hình cần thiết vào thư mục dự án chính.
- Dọn dẹp thư mục tạm sau khi hoàn tất.

---

## 3. Tài Liệu Tham Khảo Trong Kỹ Năng
- [Danh mục các Repository nổi tiếng theo từng bài toán](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/github-repo-reuser/references/popular_repos_directory.md)
