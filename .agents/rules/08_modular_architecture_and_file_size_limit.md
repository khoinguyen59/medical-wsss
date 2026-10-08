---
id: rule-08-modular-architecture-and-file-size-limit
title: Modular Architecture & 400-Line Code Cap Standard
activation: always_on
description: "Quy chuẩn phân rã cấu trúc thư mục có tổ chức, phân tách phân tầng rõ ràng (Clean Architecture / Microservices) và LUẬT BẤT BIẾN: Không một file mã nguồn nào được vượt quá 400 dòng code."
---

# Rule 08: Modular Architecture & 400-Line Code Cap Standard

## 1. Mục Đích & Nguyên Tắc Tối Thượng
Codebase phình to, lộn xộn và chứa các "God Files" (file hàng ngàn dòng chứa đủ thứ logic từ DB, API đến UI) là nguyên nhân số 1 gây ra bug ngầm, khó bảo trì và làm cạn kiệt context window của AI.
- **Luật Giới Hạn 400 Dòng (The 400-Line Hard Cap):** **Tuyệt đối không một tệp mã nguồn nào được phép vượt quá 400 dòng code**. Ngưỡng cảnh báo refactor là 300 dòng.
- **Phân Rã Trách Nhiệm Đơn Nhất (Single Responsibility & Modularization):** Mọi chức năng phải được chia nhỏ thành các tệp độc lập, có phạm vi rõ ràng.
- **Cấu Trúc Thư Mục Chặt Chẽ:** Tuyệt đối cấm vứt file bừa bãi ở thư mục gốc (root) hoặc gom chung các thành phần không cùng tầng trách nhiệm.

---

## 2. Tiêu Chuẩn Cấu Trúc Thư Mục Phân Tầng Chuẩn (Clean Layered / Modular)

Mọi dự án phải tuân theo cấu trúc phân tầng rõ ràng:

```
src/
├── config/             # Cấu hình môi trường, hằng số hệ thống (< 150 dòng/file)
├── controllers/        # Tiếp nhận request, validate input sơ bộ, gọi service (< 200 dòng/file)
├── services/           # Nghiệp vụ cốt lõi (Business logic) - chia nhỏ theo domain
│   ├── auth/           # Mỗi service chỉ làm 1 việc cụ thể
│   │   ├── token_service.py       # (< 200 dòng)
│   │   ├── password_service.py    # (< 150 dòng)
│   │   └── oauth_service.py       # (< 250 dòng)
├── repositories/       # Tương tác Database / ORM / Blockchain queries (< 250 dòng/file)
├── models/             # Schema, Entities, Pydantic/TypeScript interfaces (< 200 dòng/file)
├── middlewares/        # Auth guard, Rate limiting, Error handler (< 150 dòng/file)
├── utils/              # Helper functions dùng chung (< 150 dòng/file)
└── types/              # Định nghĩa types, enums (< 150 dòng/file)
```

---

## 3. Chiến Lược Phân Rã Khi File Chạm Ngưỡng 300-400 Dòng

Khi một file tiến gần đến 350-400 dòng, Agent **bắt buộc phải thực hiện phân rã (Decomposition)**:

1. **Phân rã Service lớn thành Sub-Services:**
   - Thay vì: `OrderService.ts` (700 dòng gồm tính giá, kiểm tra kho, tạo hóa đơn, gửi mail).
   - Tách thành:
     - `orderPricing.ts`: Tính toán giảm giá, thuế, tổng tiền (< 150 dòng).
     - `orderInventory.ts`: Kiểm tra và trừ tồn kho (< 120 dòng).
     - `orderNotification.ts`: Gửi email/webhook xác nhận (< 100 dòng).
     - `orderService.ts`: Điều phối (orchestrator) gọi các sub-services trên (< 180 dòng).
2. **Phân rã Component UI lớn:**
   - Tách các modal, bảng dữ liệu, thanh tìm kiếm, card con thành các component riêng biệt trong thư mục con `components/<feature>/`.
3. **Phân rã Router / Controller:**
   - Dùng sub-routers (ví dụ: `auth_router`, `user_router`, `item_router` gom về `api_router`).

---

## 4. Anti-Pattern vs Positive Pattern

```diff
- // ANTI-PATTERN: Một file UserService.js chứa 650 dòng code gồm cả hash password, gửi mail,
- // validate CCCD, tạo token JWT, query MongoDB và render HTML email.
- // Hậu quả: Dễ gây conflict, AI đọc hiểu tốn token, khó viết unit test.

+ // POSITIVE PATTERN: Tách thành các file nhỏ < 250 dòng:
+ // - authService.js (xác thực & token)
+ // - userValidation.js (kiểm tra dữ liệu)
+ // - mailerService.js (gửi email thông báo)
+ // - userRepository.js (truy vấn DB)
```

---

## 5. Lệnh Kiểm Tra Đối Kháng Tự Động (Lệnh Quét Giới Hạn 400 Dòng)

Agent bắt buộc tích hợp lệnh kiểm tra này vào Cổng nghiệm thu đối kháng trước khi bàn giao:

```bash
# Quét toàn bộ mã nguồn và báo lỗi nếu có bất kỳ file nào vượt quá 400 dòng
cmd /c "python .agents\skills\modular-code-decomposer\scripts\check_file_lines.py src"
```
