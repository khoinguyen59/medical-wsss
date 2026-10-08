---
id: rule-04-self-optimization-and-evolution
title: Self-Optimization & Domain Adaptation Protocol
activation: always_on
description: "Quy chuẩn tự điều chỉnh và tiến hóa: Agent tự động cập nhật AGENTS.md, hiệu chỉnh rules và tinh chỉnh workflows phù hợp với miền bài toán cụ thể của dự án mới."
---

# Rule 04: Self-Optimization & Domain Adaptation Protocol

## 1. Mục Đích & Nguyên Tắc Vận Hành
Bộ khung `UNIVERSAL_AGENT_SYSTEM` là một thực thể sống có khả năng tự tiến hóa (Self-Evolving Framework).
- Khi vừa được sao chép vào một dự án mới, nó ở trạng thái "Đa Năng Tổng Quát" (Generic Meta-Agent).
- Sau khi hoàn thành Pha 1 (Reconnaissance) và Pha 2 (Deep Research), Agent **bắt buộc phải tự tối ưu hóa toàn bộ hệ thống** để biến thành "Chuyên Gia Đặc Thù" (Domain-Specific Specialist) của dự án đó.

---

## 2. Quy Trình Tự Tối Ưu Hóa (Self-Calibration SOP)

### 2.1. Cập Nhật Danh Tính & Thông Số Trong `AGENTS.md`
Agent tiến hành chỉnh sửa các trường sau trong `AGENTS.md`:
1. **Agent Name:** Đặt tên chuyên biệt (ví dụ: `CapNongCommerceAgent`, `FintechRiskAgent`, `GenomicsWeb3Agent`).
2. **Target Project:** Tên dự án và đường dẫn tuyệt đối của thư mục làm việc hiện tại.
3. **Environment Parameters:** Cập nhật phiên bản cụ thể của Node.js, Python, cơ sở dữ liệu, cổng chạy dev server và port backend.
4. **Project Invariants:** Bổ sung các ràng buộc nghiệp vụ đặc thù của domain (ví dụ: "Dữ liệu y tế phải mã hóa AES-GCM", "Giao dịch thanh toán phải có idempotency key", "Tỷ lệ model accuracy >= 95%").

### 2.2. Tinh Chỉnh Cổng Nghiệm Thu Trong `workflows/04_pre_handover_adversarial_verification.md`
Thay thế các lệnh kiểm thử placeholder bằng các lệnh kiểm thử thực tế của dự án:
- Dự án Node/TypeScript: `npm test`, `npm run build`, `npm run lint`, `npm audit`.
- Dự án Python/FastAPI: `pytest`, `mypy src/`, `flake8`, `pip audit`.
- Dự án Rust: `cargo test`, `cargo build --release`, `cargo clippy`, `cargo audit`.
- Dự án Go: `go test -v ./...`, `go build`, `golangci-lint run`.

### 2.3. Tối Ưu Hóa Các Rules Cục Bộ
- Nếu dự án có các chuẩn kiến trúc riêng biệt (ví dụ: Clean Architecture, CQRS, Event-Driven), Agent tạo thêm rule tương ứng (ví dụ: `.agents/rules/06_clean_architecture.md`) để bảo đảm toàn bộ mã nguồn viết ra tuân thủ nghiêm ngặt.

---

## 3. Anti-Pattern vs Positive Pattern

```diff
- // ANTI-PATTERN: Giữ nguyên file AGENTS.md với các giá trị chung chung, không cập nhật lệnh test
- // Hậu quả: Agent không thể chạy kiểm thử tự động, dễ bỏ sót lỗi khi bàn giao.

+ // POSITIVE PATTERN: Tự động ghi đè lệnh test thực tế vào Workflow và AGENTS.md
+ // Ví dụ: cmd /c "pytest tests/ --cov=src" thay vì placeholder generic.
```
