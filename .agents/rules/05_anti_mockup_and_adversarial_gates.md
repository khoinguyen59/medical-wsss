---
id: rule-05-anti-mockup-and-adversarial-gates
title: Anti-Mockup Law, Line Cap Enforcement & Adversarial Gates
activation: always_on
description: "Luật cấm tuyệt đối sinh mã giả lập (mockup/stub/hardcoded state), kiểm soát giới hạn 400 dòng code và quy chuẩn kiểm định đối kháng 6 bước bắt buộc qua shell trước khi bàn giao."
---

# Rule 05: Anti-Mockup Law, Line Cap Enforcement & Adversarial Gates

## 1. Mục Đích & Nguyên Tắc Tối Thượng
Một trợ lý kỹ thuật cấp cao tuyệt đối không bao giờ được "lừa dối" người dùng bằng dữ liệu ảo và không bao giờ để mã nguồn phình to vô tổ chức.
- **Không Mockup:** Mọi thao tác phải kết nối đến DB thật, API thật, File system thật.
- **Không File Quá 400 Dòng:** Mọi tệp mã nguồn phải được chia nhỏ gọn gàng. File chạm 300 dòng phải lên kế hoạch tách module.
- **Cổng Nghiệm Thu 6 Bước:** Không bao giờ tuyên bố hoàn thành nếu chưa chạy đủ 6 lệnh shell kiểm thử thực tế.

---

## 2. Cổng Nghiệm Thu 6 Bước Bắt Buộc (Mandatory Pre-Handover Gate)

Agent **nghiêm cấm tuyên bố hoàn thành** nếu chưa chạy thành công 6 lệnh shell sau:

```bash
# 1. Kiểm tra Giới hạn 400 dòng code (100% Files <= 400 dòng)
cmd /c "python .agents\skills\modular-code-decomposer\scripts\check_file_lines.py src"

# 2. Kiểm thử Logic & Nghiệp vụ (100% Passed)
cmd /c "<test_command>"

# 3. Biên dịch Production Build (Exit code 0)
cmd /c "<build_command>"

# 4. Phân tích tĩnh, Typecheck & Linter (0 Lỗi)
cmd /c "<lint_command>"

# 5. Quét đối kháng chống Mockup & Dữ liệu giả (0 kết quả)
cmd /c "git grep -iE \"fakeData|dummyData|mockUser\" -- src/ || echo Clean"

# 6. Cập nhật Đồ thị Tri thức AST
cmd /c "graphify update ."
```

---

## 3. Biên Bản Kiểm Định Đối Kháng Mẫu

Mọi phản hồi bàn giao phải đính kèm bảng:

```markdown
### 🛡️ Biên Bản Kiểm Định Đối Kháng (Pre-Handover Verification Matrix)

| STT | Tiêu Chí Kiểm Định | Lệnh Shell Thực Thi | Kết Quả Thực Tế | Trạng Thái |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Line Cap Audit (<= 400 lines)** | `check_file_lines.py src` | 100% files <= 400 lines | ✅ PASS |
| 2 | **Unit / Integration Tests** | `cmd /c "<test_cmd>"` | Passed X/X tests | ✅ PASS |
| 3 | **Production Build** | `cmd /c "<build_cmd>"` | Exit code 0, 0 errors | ✅ PASS |
| 4 | **Linter & Type Check** | `cmd /c "<lint_cmd>"` | 0 errors, 0 warnings | ✅ PASS |
| 5 | **Anti-Mockup Scan** | `git grep -i "fakeData"` | 0 match detected | ✅ PASS |
| 6 | **AST Graph Sync** | `graphify update .` | Graph updated | ✅ PASS |
```
