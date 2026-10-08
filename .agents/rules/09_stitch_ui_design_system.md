---
id: rule-09-stitch-ui-design-system
title: Google Stitch UI Design System & Component Architecture
activation: model_decision
description: "Quy chuẩn thiết kế giao diện theo triết lý Google Stitch (Stitches UI): thiết lập file DESIGN.md, bố cục Bento Grid/Card-based, phân tách Presentation Component nhỏ (< 200 dòng), cấm màu tím đen AI và dùng icon vector tự thiết kế."
---

# Rule 09: Google Stitch UI Design System & Component Architecture

## 1. Mục Đích & Triết Lý Thiết Kế Google Stitch
Google Stitch là chuẩn mực thiết kế giao diện hiện đại kết hợp sức mạnh của AI UI Generation với tư duy hệ thống thành phần (Component-Driven Design).
- **Tránh giao diện generic, sơ sài:** Không bao giờ chấp nhận giao diện với các màu cơ bản (đỏ, xanh, vàng thuần), chữ mặc định trình duyệt, hay các khối hộp đơn điệu.
- **Cấm Tuyệt Đối Tông Màu Tím Đen AI:** Nghiêm cấm sử dụng tông màu tím đen rập khuôn của AI. Bắt buộc dùng bảng màu tự nhiên theo từng domain (Oceanic Navy, Forest Emerald, Slate & Amber, Charcoal & Cobalt).
- **Cấm Icon Thô Sơ - Bắt Buộc Dùng Logo/Icon Tự Thiết Kế:** Dùng Lucide, Iconsax, Tabler hoặc vector SVG tự vẽ. Cấm emoji làm icon chính.
- **Dùng Dấu '-' Chuẩn Bàn Phím:** Tuyệt đối không dùng dấu gạch ngang dài tự động của AI trong nội dung UI.
- **Thiết lập tài liệu `DESIGN.md`:** Mọi dự án có giao diện người dùng phải khởi tạo một file `DESIGN.md` quy định rõ ràng bảng màu, Typography, Spacing, và Phong cách thị giác trước khi code.
- **Thành phần giao diện nhỏ & cô đọng (< 200 dòng):** Mọi component UI phải là các Presentation Components nhỏ gọn, tách rời khỏi logic gọi API.

---

## 2. Các Nguyên Tắc Thiết Kế Google Stitch Cốt Lõi

### 2.1. Bố Cục Thẻ & Lưới Bento (Card-Based & Bento Grid Layout)
- Sử dụng Bento Grid để trình bày thông tin đa chiều một cách trực quan, hiện đại.
- Mỗi thẻ (Card) phải có:
  - Đường viền tinh tế (`border border-slate-800` hoặc glassmorphic `backdrop-blur-md`).
  - Đổ bóng mềm mại (`shadow-sm hover:shadow-md transition-shadow`).
  - Khoảng đệm (padding) thoáng đãng, cân đối (`p-5` hoặc `p-6`).

### 2.2. Bảng Màu & Ánh Sáng Tự Nhiên Theo Domain (Không Dùng Tím Đen)
- Xây dựng bảng màu dựa trên bài toán thực tế (Domain-driven color palette).
- Hiệu ứng kính mờ (Glassmorphism): `bg-white/5 backdrop-blur-lg border-white/10`.

### 2.3. Micro-Animations & Trạng Thái Tương Tác Đầy Đủ
Mọi thành phần tương tác (Button, Card, Input, Tab) bắt buộc phải có:
- Trạng thái Hover: Thay đổi màu nhẹ, nâng thẻ nhẹ (`hover:-translate-y-0.5 transition-transform`).
- Trạng thái Active / Focus: Vòng sáng bao quanh (`focus:ring-2 focus:ring-primary/50`).
- Trạng thái Loading: Skeleton shimmer effect (cấm để trang trắng khi đang tải dữ liệu).
- Trạng thái Rỗng (Empty State): Minh họa trực quan + Lời kêu gọi hành động (CTA) rõ ràng.

---

## 3. Cấu Trúc Thư Mục Giao Diện Chuẩn (Google Stitch Pattern)

```
frontend/src/ (hoặc components/)
├── assets/             # Logo SVG, Banner SVG tự thiết kế, Icons vector
├── components/         # Các thành phần tái sử dụng (< 200 dòng/file)
│   ├── common/         # Button, Input, Modal, Badge, Skeleton
│   ├── layout/         # Navbar, Sidebar, Footer, PageHeader
│   └── dashboard/      # StatCard, TrendChart, ActivityFeed, QuickActions
├── hooks/              # Toàn bộ logic state & API calls (< 150 dòng/file)
│   ├── useAuth.ts
│   └── useMarketplaceData.ts
├── context/            # Global state (Theme, Auth, Language)
├── styles/             # Global CSS tokens, animations
└── DESIGN.md           # Bản đặc tả hệ thống thiết kế Google Stitch
```

---

## 4. Mẫu Cấu Trúc File `DESIGN.md` Bắt Buộc

```markdown
# Design System Specification (Google Stitch Standard)

## 1. Visual Direction & Aesthetic
- Style: Oceanic Clean Bento Grid & Precision Surface
- Personality: High-tech, Trustworthy, Professional, Fluid
- Anti-AI Rule: Khong dung mau tim den AI. Khong dung icon tho so. Dung dau '-' ban phim.

## 2. Color Palette (Domain Grounded)
- Surface Primary: #0A1128 (Deep Oceanic Navy)
- Surface Secondary: rgba(255, 255, 255, 0.04) (Glass Card)
- Primary Accent: #0284C7 (Cobalt Sky)
- Success / Positive: #059669 (Emerald)
- Border Subtlety: rgba(2, 132, 199, 0.18)

## 3. Typography
- Heading: 'Plus Jakarta Sans', sans-serif (Font weights: 600, 700)
- Body / Data: 'Inter', sans-serif (Font weights: 400, 500)
- Monospace / Hash: 'JetBrains Mono', monospace

## 4. Layout Breakpoints & Spacing
- Mobile: 320px - 640px
- Tablet: 641px - 1024px
- Desktop: 1025px - 1440px+
```
