---
name: bespoke-asset-designer
description: "Ky nang tim kiem, clone va tich hop cac repository open-source tren GitHub ve Logo, Banner SVG va Icon cao cap (SVGL, Simple Icons, Capsule Render, Lucide, Iconsax), tuyet doi khong dung script tu che tho so."
---

# Bespoke Asset Designer Skill

Ky nang nay huong dan Agent cach khai thac cac repository ma nguon mo chat luong cao tren GitHub de lay logo, banner va icon vector chuyen nghiep cho du an, tuan thu nghiem ngat nguyen tac "Khong tu ve tho so, tan dung giai phap open-source xuat sac co san tren GitHub".

## 1. Khi Nao Kich Hoat Ky Nang Nay?
- Khi du an can logo thuong hieu, favicon, tech stack icons cho trang web / dashboard.
- Khi can tao banner, header sinh dong cho trang chu, landing page hoac README.
- Khi can bo icon he thong: thay the toan bo emoji hay icon co ban bang vector tu cac repo GitHub uy tin.

---

## 2. Danh Muc Cac Repository GitHub Hang Dau Ve Asset

### A. Kho Logo Vector SVG:
- **SVGL (`pheralb/svgl`):** Thu vien logo vector SVG tuyet dep va duoc cap nhat lien tuc danh cho web hien dai.
- **Simple Icons (`simple-icons/simple-icons`):** Hon 3100+ icon vector SVG chinh thuc cua cac thuong hieu, ngon ngu va cong nghe.
- **Gilbarbara Logos (`gilbarbara/logos`):** Kho chua hang nghin logo SVG vector goc.

### B. Kho Banner & Dynamic Header SVG:
- **Capsule Render (`kyechan99/capsule-render`):** Cong cu open-source sinh dynamic SVG banner voi hieu ung song nuoc (waving), gradient tu nhien, ho tro moi kich thuoc web.
- **Readme Typing SVG (`DenverCoder1/readme-typing-svg`):** Dynamic SVG tao hieu ung go chu muot ma danh cho tieu de.
- **Markdown Badges (`Ileriayo/markdown-badges`):** Kho badge va banner vector phang hien dai.

### C. Kho Icon He Thong:
- **Lucide Icons (`lucide-icons/lucide`):** Bo icon vector hien dai, stroke manh, sieu nhe.
- **Iconsax (`lusaxweb/iconsax`):** 6 phong cach icon nghe thuat (Linear, Bold, Bulk, Two-tone).
- **Tabler Icons (`tabler/tabler-icons`):** 4500+ vector icon pixel-perfect.

---

## 3. Quy Trinh 3 Buoc Tich Hop Tu GitHub

### Buoc 1: Tim Kiem Asset Trong Repo GitHub
- Dung GitHub MCP hoac Firecrawl de tim kiem logo/banner trong cac repo tren.
- Hoac tra cuu nhanh trong package `simple-icons` / `svgl`.

### Buoc 2: Clone Nong Hoac Cai Dat Package
- **Cai dat qua Package Manager:**
  ```bash
  cmd /c "npm install simple-icons lucide-react"
  # hoac
  cmd /c "pip install simple-icons"
  ```
- **Hoac Shallow Clone repo ve thu muc tam de trich xuat SVG:**
  ```bash
  cmd /c "git clone --depth 1 https://github.com/pheralb/svgl.git temp_tools/svgl"
  ```
  Sau khi copy file SVG can thiet vao `src/assets/`, xoa ngay thu muc `temp_tools/`.

### Buoc 3: Tich Hop Banner Dong (Dynamic SVG)
- Su dung template SVG tu `capsule-render` hoac tich hop component banner tu repo de render header chat luong cao ma khong bi vo hat.

---

## 4. Tai Lieu Tham Khao Trong Ky Nang
- [Chi tiet cac repo logo, banner va icon tren GitHub](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/bespoke-asset-designer/references/icon_repos_cheatsheet.md)
