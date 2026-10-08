---
name: stitch-ui-architect
description: "Kỹ năng kiến trúc giao diện theo tiêu chuẩn Google Stitch: tạo bản đặc tả DESIGN.md, cấu trúc luồng màn hình, thiết kế thẻ Bento Grid, và tách nhỏ Presentation Components (< 200 dòng)."
---

# Stitch UI Architect Skill

Kỹ năng này hướng dẫn Agent cách áp dụng tư duy thiết kế của **Google Stitch** (công cụ AI UI thế hệ mới của Google Labs) vào việc xây dựng giao diện ứng dụng web và di động đẳng cấp, có cấu trúc chặt chẽ và không bao giờ sơ sài.

## 1. Khi Nào Kích Hoạt Kỹ Năng Này?
- Khi dự án có phần giao diện người dùng (Web Frontend, Dashboard, Mobile App).
- Khi khởi tạo hệ thống thiết kế cho dự án mới (viết file `DESIGN.md`).
- Khi cần tạo giao diện mới từ bản vẽ, wireframe hoặc mô tả đề bài.

---

## 2. 5 Nguyên Tắc Google Stitch Cốt Lõi

1. **Specific & Context-Rich (Đặc tả chi tiết, không chung chung):**
   - Không bao giờ thiết kế "dashboard chung chung".
   - Luôn xác định rõ: Ai dùng? Có những thẻ chỉ số nào? Bố cục lưới ra sao? Màu sắc chủ đạo là gì?
2. **Semantic Design Tokens (`DESIGN.md`):**
   - Mọi dự án phải có file `DESIGN.md` quy định: Bảng màu HSL/HEX, Typography (Google Fonts), Spacing, Border Radii, Box Shadows.
3. **Bento Grid & Card Hierarchy:**
   - Sử dụng thẻ bo góc mềm mại (`rounded-2xl`), đường viền siêu mỏng (`border border-white/10`), và đổ bóng phân tầng.
4. **Presentation vs Container Separation:**
   - Presentation Components: Chỉ nhận `props` và render HTML/CSS, tuyệt đối không chứa logic gọi API, độ dài luôn `< 200 dòng`.
   - Custom Hooks / Context: Chứa toàn bộ state và gọi API, độ dài luôn `< 150 dòng`.
5. **Interactive Feedback:**
   - 100% nút bấm, card, ô input phải có hiệu ứng micro-animations khi hover/focus/active.
   - Có đầy đủ trạng thái Loading (Skeleton shimmer) và Empty State (Hình minh họa + nút kêu gọi hành động).

---

## 3. Tài Liệu Tham Khảo Trong Kỹ Năng
- [Cheatsheet viết Prompt theo chuẩn Google Stitch](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/stitch-ui-architect/references/stitch_prompting_cheatsheet.md)
- [Template file DESIGN.md chuẩn](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/stitch-ui-architect/references/design_md_template.md)
