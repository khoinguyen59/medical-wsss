# SOP 04: Pre-Handover Adversarial Verification Workflow

## 1. Mục Tiêu & Phạm Vi
Quy trình kiểm định đối kháng 6 bước bắt buộc. Mọi nhiệm vụ, tính năng mới hoặc bản sửa lỗi đều phải chạy qua quy trình này trước khi thông báo hoàn thành cho người dùng.

---

## 2. Quy Trình 6 Bước Kiểm Định Đối Kháng

```
[BẮT ĐẦU NGHIỆM THU]
         │
         ▼
[BƯỚC 1: KIỂM TOÁN GIỚI HẠN 400 DÒNG CODE] ──(Có file > 400 dòng)──> [Phân rã module ngay]
         │ (100% Passed)
         ▼
[BƯỚC 2: KIỂM THỬ ĐƠN VỊ & TÍCH HỢP] ──(Thất bại)──> [Dừng & Sửa lỗi code/test]
         │ (Passed 100%)
         ▼
[BƯỚC 3: BIÊN DỊCH PRODUCTION BUILD] ──(Thất bại)──> [Dừng & Sửa lỗi build/types]
         │ (Exit code 0)
         ▼
[BƯỚC 4: PHÂN TÍCH TĨNH & LINTER] ──(Thất bại)──> [Dừng & Chuẩn hóa style/lint]
         │ (0 errors)
         ▼
[BƯỚC 5: QUÉT ĐỐI KHÁNG ANTI-MOCKUP] ──(Có mock)──> [Dừng & Thay bằng code thật]
         │ (0 mockups)
         ▼
[BƯỚC 6: ĐỒNG BỘ ĐỒ THỊ AST GRAPHIFY]
         │
         ▼
[XUẤT BẢNG BIÊN BẢN NGHIỆM THU & BÀN GIAO]
```

### Chi Tiết Từng Lệnh Kiểm Tra:

#### 1. Kiểm toán Giới hạn 400 Dòng Code:
```bash
cmd /c "python .agents\skills\modular-code-decomposer\scripts\check_file_lines.py src"
```

#### 2. Kiểm thử Unit & Integration Tests:
```bash
cmd /c "<test_command>"
```

#### 3. Biên dịch Production:
```bash
cmd /c "<build_command>"
```

#### 4. Phân tích tĩnh & Kiểm tra kiểu:
```bash
cmd /c "<lint_command>"
```

#### 5. Quét đối kháng chống Mockup & Hardcoded:
```bash
cmd /c "git grep -iE \"TODO: mock|fakeData|dummyData|mockUser\" -- src/ || echo Clean"
```

#### 6. Đồng bộ đồ thị AST Graphify:
```bash
cmd /c "graphify update ."
```

---

## 3. Báo Cáo Nghiệm Thu Bắt Buộc
Đính kèm bảng kết quả trong phản hồi cuối cùng:
```markdown
### 🛡️ Biên Bản Kiểm Định Đối Kháng (Pre-Handover Verification Matrix)

| STT | Tiêu Chí | Lệnh Thực Thi | Kết Quả Thực Tế | Trạng Thái |
| :---: | :--- | :--- | :--- | :---: |
| 1 | Line Cap Audit (<= 400 lines) | `check_file_lines.py src` | 100% files <= 400 lines | ✅ PASS |
| 2 | Unit/Integration Test | `cmd /c "<test_cmd>"` | X/X tests passed | ✅ PASS |
| 3 | Production Build | `cmd /c "<build_cmd>"` | Exit code 0 | ✅ PASS |
| 4 | Linter & Typecheck | `cmd /c "<lint_cmd>"` | 0 errors | ✅ PASS |
| 5 | Anti-Mockup Scan | `git grep -i "fakeData"` | 0 matches | ✅ PASS |
| 6 | Graphify Sync | `graphify update .` | Synced successfully | ✅ PASS |
```
