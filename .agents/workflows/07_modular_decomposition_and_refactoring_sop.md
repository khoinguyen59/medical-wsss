# SOP 07: Modular Decomposition & 400-Line Code Cap Enforcement

## 1. Mục Tiêu & Phạm Vi
Quy trình đảm bảo toàn bộ mã nguồn của dự án luôn duy trì cấu trúc phân rã module sạch sẽ, không có tệp nào vượt quá **400 dòng code**, ngăn chặn triệt để hiện tượng "God Files" gây lỗi ngầm và tràn bộ nhớ context.

---

## 2. Quy Trình Thực Thi Chi Tiết

```
[KIỂM TRA DÒNG CODE TRONG MỌI PHIÊN LÀM VIỆC]
                    │
                    ▼
[BƯỚC 1: CHẠY QUÉT KIỂM TOÁN DÒNG CODE]
cmd /c "python .agents\skills\modular-code-decomposer\scripts\check_file_lines.py src"
                    │
                    ▼
     {Có file nào vượt quá 400 dòng?}
       /                          \
   [CÓ - VI PHẠM]              [KHÔNG - ĐẠT]
     │                               │
     ▼                               ▼
[BƯỚC 2: PHÂN RÃ THEO PATTERN]   [HOÀN TẤT & ĐỒNG BỘ AST]
- Tách Service thành Sub-Services cmd /c "graphify update ."
- Tách Component UI thành Sub-cards
- Tách Data validation ra Validator
- Tách DB queries ra Repository
     │
     ▼
[BƯỚC 3: CẬP NHẬT IMPORT & KIỂM THỬ]
- Sửa lại các đường dẫn import liên quan
- Chạy lại Unit / Integration Test
- Chạy lại check_file_lines.py xác nhận exit code 0
```

---

## 3. Checklist Giám Sát Khi Viết Code Mới
- [ ] Khi code một file chạm 300 dòng: Lập tức cân nhắc tạo file vệ tinh để chứa hàm phụ trợ.
- [ ] Không bao giờ để logic controller gọi trực tiếp database query (phải qua repository).
- [ ] Không bao giờ để component React chứa cả JSX layout lẫn hàm tính toán toán học phức tạp (tách ra utils).
- [ ] Lệnh `check_file_lines.py` phải trả về `[PASSED]` trước khi bàn giao task.
