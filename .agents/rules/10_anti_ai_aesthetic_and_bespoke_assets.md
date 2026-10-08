---
id: rule-10-anti-ai-aesthetic-and-bespoke-assets
title: Anti-AI Aesthetic, Bespoke Assets & Standard Hyphen Rules
activation: always_on
description: "Quy chuẩn loại bỏ hoàn toàn phong cách AI rập khuôn: cấm tông màu tím đen AI, cấm icon cơ bản đơn điệu, bắt buộc dùng logo/icon/banner thiết kế riêng và bắt buộc dùng dấu gạch ngang chuẩn phím (-)."
---

# Rule 10: Anti-AI Aesthetic, Bespoke Assets & Standard Hyphen Rules

## 1. Mục Đích & Nguyên Tắc Tối Thượng
Các sản phẩm tạo ra bởi AI thường mắc phải 3 "dấu vết rập khuôn" (AI Hallmarks) gây cảm giác rẻ tiền và thiếu tính chuyên nghiệp:
1. **Tông màu tím đen rập khuôn (Cliché AI Purple-Black):** Nền đen xì kết hợp tím neon/hồng cyberpunk xuất hiện khắp nơi.
2. **Icon cơ bản đơn điệu:** Sử dụng emoji hoặc các icon hình học thô sơ (hình tròn tick, mũi tên thô).
3. **Dấu gạch ngang dài tự động (Em-dash/En-dash):** Trình tạo văn bản AI hay lạm dụng dấu gạch ngang dài khiến câu văn thiếu tự nhiên và dễ lỗi font trên một số hệ thống.

Rule này ban hành các quy định nghiêm ngặt để triệt tiêu hoàn toàn 3 lỗi trên.

---

## 2. Các Ràng Buộc Kỹ Thuật Chi Tiết

### 2.1. Cấm Tuyệt Đối Tông Màu Tím Đen AI (Anti-Purple-Black Cliché)
- **Cấm:** Không dùng nền đen tuyền kết hợp tím neon (`#8B5CF6`, `#A855F7`, `#7C3AED` trên nền `#000000` hoặc `#090D16`).
- **Bắt buộc dùng Bảng màu theo Domain Thực Tế (Domain-Grounded Palettes):**
 - **Nông nghiệp / Môi trường / Dược liệu:** Xanh ngọc lục bảo (Emerald `#059669`), Xanh xô thơm (Sage `#10B981`), Vàng lúa chín (`#D97706`), Kem ấm (`#FDFBF7`).
 - **Tài chính / Ngân hàng / DeSci / Crypto:** Xanh biển sâu (Deep Navy `#0A192F`), Thép lạnh (Slate `#334155`), Vàng kim (Amber `#F59E0B`), Trắng băng (`#F8FAFC`).
 - **Doanh nghiệp / SaaS / Quản lý:** Xám than chì (Charcoal `#1E293B`), Xanh Cobalt (`#2563EB`), Trắng ngà (`#F1F5F9`).
 - **Y tế / Sinh học phân tử:** Xanh Cyan (`#0891B2`), Xanh lam phẫu thuật (`#0284C7`), Bạc sáng (`#E2E8F0`).

### 2.2. Cấm Dùng Icon Cơ Bản - Bắt Buộc Dùng Logo & Icon Tự Thiết Kế / Repo Chuẩn
- **Cấm:** Không dùng emoji làm icon chính của giao diện (như biểu tượng tên lửa, ngôi sao, tia sét emoji), không dùng các hình SVG tròn/vuông cơ bản.
- **Bắt buộc:**
 - Sử dụng các thư viện icon vector mã nguồn mở chất lượng cao: **Lucide Icons** (`lucide-react`), **Iconsax** (`iconsax-react`), **Tabler Icons**, **Phosphor Icons**, hoặc **Simple Icons** (cho logo các hãng).
 - Tự thiết kế logo vector SVG riêng cho dự án (đặt tại `assets/logo.svg`, `assets/banner.svg`), có tỷ lệ hình học chuẩn, gradient tinh tế và đường nét hiện đại.
 - Sử dụng các repo chuyên về banner, logo vector mẫu trên GitHub để tích hợp vào header và landing page.

### 2.3. Bắt Buộc Dùng Dấu Gạch Ngang Bàn Phím (-) Chuẩn
- **Cấm:** Tuyệt đối không dùng dấu gạch ngang dài tự động của AI (em-dash và en-dash).
- **Bắt buộc:** Luôn sử dụng dấu gạch ngang ngắn chuẩn bàn máy tính: `-` (mã ASCII 0x2D).
- Quy định này áp dụng cho toàn bộ: Tiêu đề tài liệu, văn bản giải thích, commit message, code comments và nội dung hiển thị trên UI.

---

## 3. Anti-Pattern vs Positive Pattern

```diff
- // ANTI-PATTERN: Nền đen tím, icon cơ bản, dùng dấu gạch dài
- const theme = { bg: "#000000", primary: "#9333ea" }; // Tím đen AI
- <button>⚡ Mua ngay - Giá rẻ</button> // Emoji thô + gạch ngang dài

+ // POSITIVE PATTERN: Bảng màu theo domain, icon Lucide/Iconsax, dấu gạch phím -
+ const theme = { bg: "#0b1329", primary: "#0284c7" }; // Navy & Cyan
+ <button><ShoppingCartIcon className="w-5 h-5 mr-2" /> Mua ngay - Giá ưu đãi</button>
```

---

## 4. Lệnh Kiểm Tra Đối Kháng (Verification)
Kiểm tra xem mã nguồn hoặc tài liệu có còn chứa dấu gạch ngang dài không:
```bash
# Quét tìm dấu gạch dài em-dash hoặc en-dash (kết quả phải là 0)
cmd /c "git grep -P \"\x{2014}|\x{2013}\" -- src/ docs/ || echo Clean"
```
