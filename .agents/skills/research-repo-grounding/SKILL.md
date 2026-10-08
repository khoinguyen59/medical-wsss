---
name: research-repo-grounding
description: Quy trình bắt buộc đối soát mã nguồn từ các GitHub Repositories chính thức của tác giả, triệt tiêu căn bệnh tự viết code từ trí nhớ (Zero Scratch Drift).
---

# Research Repo Grounding Skill

## Mục đích
Đảm bảo AI Agent không bao giờ tự ý tạo các file `scratch_*.py` để code thuật toán từ trí nhớ xác suất. Mọi thuật toán, hàm mục tiêu, kiến trúc mô hình phải được đối soát dòng-sang-dòng (Code-to-Code Parity) từ repo gốc của tác giả.

## Quy Trình 4 Bước

### Bước 1: Tìm kiếm Repo Chính Thức
1. Dùng `github` MCP (`search_repositories`) hoặc `firecrawl` MCP (`firecrawl_developer_search`) để tìm URL chính thức của tác giả bài báo.
2. Kiểm tra uy tín: Số stars, hội nghị xuất bản (NeurIPS, ICML, ICLR, KDD, ACL, CVPR).

### Bước 2: Clone Repo về Thư Mục `repos/`
1. Clone repo với tham số nhẹ:
   ```bash
   git clone --depth 1 <repo_url> repos/<repo_name>
   ```
2. Đảm bảo `.gitignore` đã có `repos/*` để không ảnh hưởng đến git chính của dự án.

### Bước 3: Đối Soát Code-to-Code
1. Mở file mã nguồn gốc của tác giả trong `repos/` bằng `view_file`.
2. Kiểm tra:
   * Cách tác giả định nghĩa hàm loss và hàm thưởng (Reward).
   * Cách tác giả xử lý đầu vào và tách ma trận cạnh.
   * Các siêu tham số mặc định.
3. Kế thừa trực tiếp cấu trúc hoặc class chuẩn (như `torch_geometric.explain`), không tự viết lại hàm thô sơ.

### Bước 4: Viết Unit Test Đối Chứng
1. Viết test case so sánh output của module mới với output của repo tác giả trên cùng một mẫu dữ liệu nhỏ (dummy / small graph).
2. Khi test passed 100%, mới tích hợp vào `src/` chính thức.
