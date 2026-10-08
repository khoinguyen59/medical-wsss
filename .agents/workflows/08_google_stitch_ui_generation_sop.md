# SOP 08: Google Stitch UI Generation & Component Architecture

## 1. Mục Tiêu & Phạm Vi
Quy trình hướng dẫn Agent thiết kế và xây dựng giao diện ứng dụng từ đầu theo chuẩn mực **Google Stitch (Stitches UI)**, đảm bảo tính thẩm mỹ cao cấp, cấu trúc module nhỏ gọn (< 200 dòng/component) và tách biệt hoàn toàn giữa giao diện và logic dữ liệu.

---

## 2. Quy Trình Thực Thi 4 Bước

```
[BẮT ĐẦU XÂY DỰNG GIAO DIỆN MỚI]
                │
                ▼
[BƯỚC 1: KHỞI TẠO BẢN ĐẶC TẢ DESIGN.MD]
- Viết file frontend/DESIGN.md: Bảng màu HSL/HEX, Font chữ, Spacing, Shadows.
- Cài đặt Google Fonts trong index.html hoặc CSS (@import).
                │
                ▼
[BƯỚC 2: PHÂN RÃ LUỒNG MÀN HÌNH THEO BENTO GRID]
- Chia màn hình thành các Card độc lập:
  + StatCard.jsx (< 100 dòng)
  + ChartCard.jsx (< 150 dòng)
  + ActivityList.jsx (< 120 dòng)
- Dùng layout CSS Grid / Flexbox thoáng đãng.
                │
                ▼
[BƯỚC 3: TÁCH RIÊNG LOGIC STATE SANG CUSTOM HOOKS]
- Tuyệt đối không viết hàm fetch/axios trực tiếp trong Presentation Component.
- Đưa toàn bộ useEffect, useState, handlers sang hooks/use<Feature>.js (< 150 dòng).
- Presentation Component chỉ nhận props và render.
                │
                ▼
[BƯỚC 4: POLISH TRẢI NGHIỆM VỚI HIỆU ỨNG THỊ GIÁC]
- Thêm hiệu ứng kính mờ (backdrop-blur-md, border-white/10).
- Thêm micro-animations (hover:-translate-y-0.5, transition-all).
- Thêm Skeleton Shimmer khi loading và Empty State khi không có data.
```

---

## 3. Tiêu Chí Nghiệm Thu Giao Diện Google Stitch
- [ ] File `DESIGN.md` tồn tại trong thư mục frontend.
- [ ] 100% component giao diện có độ dài dưới 200 dòng code.
- [ ] Không có component nào bị lỗi giao diện trên màn hình mobile (Responsive test).
- [ ] Toàn bộ các tương tác đều có visual feedback (Hover/Active/Focus).
