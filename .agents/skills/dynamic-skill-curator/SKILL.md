---
name: dynamic-skill-curator
description: "Kỹ năng quét, sàng lọc và nạp các bộ kỹ năng chuyên sâu từ kho 100+ kỹ năng toàn cục (tại ~/.gemini/config/skills/) vào dự án hiện tại dựa trên phân loại Tech Stack."
---

# Dynamic Skill Curator Skill

Kỹ năng này chịu trách nhiệm biến Agent từ một thực thể chung chung thành một chuyên gia công nghệ mũi nhọn bằng cách kết nối với kho tri thức kỹ năng khổng lồ đã được chuẩn hóa.

## 1. Vị Trí Kho Kỹ Năng Toàn Cục (Global Skills Hub)
Hệ thống lưu trữ toàn bộ các skill đỉnh cao tại:
`C:\Users\Nguyen Trong Khoi\.gemini\config\skills\`

Trong kho này có sẵn hơn 100 kỹ năng chuyên gia:
- **Ngôn ngữ lập trình:** `python-pro`, `golang-pro`, `rust-pro`, `typescript-pro`, `javascript-pro`, `csharp-pro`, `java-pro`, `cpp-pro`, `c-pro`, `php-pro`, `ruby-pro`, `elixir-pro`...
- **Frameworks & Web:** `react-patterns`, `nextjs-best-practices`, `fastapi-pro`, `django-pro`, `dotnet-backend`, `sveltekit`, `flutter-expert`...
- **Thiết kế & Trải nghiệm:** `ui-ux-pro-max`, `frontend-design`, `css-expert`, `tailwind-patterns`, `threejs-skills`...
- **Cơ sở dữ liệu & Hạ tầng:** `database-admin`, `postgres-best-practices`, `sql-pro`, `docker-expert`, `k8s-security-policies`, `terraform-skill`...
- **Bảo mật & Kiểm thử:** `security-auditor`, `web-security-testing`, `api-security-testing`, `tdd-workflow`, `systematic-debugging`...

---

## 2. Quy Trình Tuyển Chọn Kỹ Năng (Curating SOP)

### Bước 1: Quét Tech Stack Dự Án
Agent kiểm tra danh sách dependencies của dự án:
- Có React / Next.js / Tailwind? -> Kích hoạt `ui-ux-pro-max`, `react-patterns`, `nextjs-best-practices`.
- Có FastAPI / PyTorch / Pandas? -> Kích hoạt `fastapi-pro`, `python-pro`, `machine-learning-ops-ml-pipeline`.
- Có Go / Gin / Gorm? -> Kích hoạt `golang-pro`, `database-admin`.
- Có Rust / Cargo? -> Kích hoạt `rust-pro`, `systems-programming-rust-project`.
- Có Smart Contract / Hardhat / Foundry? -> Kích hoạt `genomics-smart-contracts`, `web3-testing`.

### Bước 2: Nạp Vào Không Gian Làm Việc Hoặc Đọc Trực Tiếp
Agent có thể copy kỹ năng vào dự án bằng lệnh chuẩn của Windows hoặc đọc trực tiếp:
```bash
cmd /c "xcopy /E /I /Y \"C:\Users\Nguyen Trong Khoi\.gemini\config\skills\ui-ux-pro-max\" \".agents\skills\ui-ux-pro-max\""
```
*Gợi ý:* Để giữ cấu trúc dự án tinh gọn, Agent có thể đọc trực tiếp file `SKILL.md` tại `C:\Users\Nguyen Trong Khoi\.gemini\config\skills\<skill-name>\SKILL.md` mà không cần copy thư mục nếu không có yêu cầu đóng gói độc lập.

### Bước 3: Đọc Và Kích Hoạt Chỉ Dẫn
Agent đọc trực tiếp `SKILL.md` của các kỹ năng vừa nạp để hấp thu toàn bộ nguyên tắc thiết kế và mẫu code chuẩn.

---

## 3. Tài Liệu Tham Khảo Trong Kỹ Năng
- [Phân loại toàn bộ kho kỹ năng (Taxonomy)](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/dynamic-skill-curator/references/global_skills_taxonomy.md)
- [Ma trận ghép nối kỹ năng theo domain bài toán](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/dynamic-skill-curator/references/domain_matching_matrix.md)
