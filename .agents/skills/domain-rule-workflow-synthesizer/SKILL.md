---
name: domain-rule-workflow-synthesizer
description: "Kỹ năng tổng hợp và tự động viết lại rules, workflows và đặc tả AGENTS.md dựa trên dữ liệu khảo sát và nghiên cứu của dự án cụ thể."
---

# Domain Rule & Workflow Synthesizer Skill

Kỹ năng này hướng dẫn Agent cách thực hiện khâu cuối cùng của Bootstrap Protocol: Tự động tổng hợp và ghi đè các cấu hình mẫu thành các quy tắc và quy trình nghiệp vụ sắc bén cho riêng dự án đó.

## 1. Khi Nào Kích Hoạt Kỹ Năng Này?
- Trong Pha 5 của Bootstrap Protocol.
- Khi người dùng thay đổi yêu cầu kiến trúc lớn hoặc thêm công nghệ mới.
- Khi cần chuẩn hóa thêm các Invariants đặc thù cho domain (ví dụ: HIPAA cho y tế, PCI-DSS cho tài chính).

---

## 2. Quy Trình Tổng Hợp 3 Bước

### Bước 1: Trích Xuất Dữ Liệu Dự Án
Thu thập các tham số thực tế từ môi trường:
- Tên dự án: Đọc từ `package.json` / `pyproject.toml` / tên thư mục gốc.
- Các lệnh thực thi thật: Đọc các script từ `package.json` (`test`, `build`, `lint`) hoặc `Makefile`, `pytest.ini`.
- Các cổng RPC / API / Dev server: Đọc từ `.env` hoặc file cấu hình (`vite.config.js`, `main.py`).

### Bước 2: Hiệu Chỉnh File `AGENTS.md`
Cập nhật:
1. Đặt lại tiêu đề và tên Agent: `Agent Name: <Domain>Agent`
2. Bổ sung các Invariant đặc thù từ [INVARIANT-11] trở đi.
3. Thay thế các placeholder `<test_command>` bằng lệnh thực tế.

### Bước 3: Tinh Chỉnh Workflow Kiểm Định
Cập nhật file `.agents/workflows/04_pre_handover_adversarial_verification.md` với các lệnh kiểm thử thật để đảm bảo Cổng nghiệm thu đối kháng hoạt động trơn tru.

---

## 3. Tài Liệu Tham Khảo Trong Kỹ Năng
- [Chuẩn cấu trúc Rule trong Antigravity](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/domain-rule-workflow-synthesizer/references/rule_template_standards.md)
- [Mẫu thiết kế SOP Workflow](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/BLOCKCHAIN/UNIVERSAL_AGENT_SYSTEM/.agents/skills/domain-rule-workflow-synthesizer/references/sop_workflow_templates.md)
