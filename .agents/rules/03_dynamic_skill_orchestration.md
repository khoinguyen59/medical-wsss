---
id: rule-03-dynamic-skill-orchestration
title: Dynamic Skill Orchestration & Global Catalog Ingestion
activation: always_on
description: "Quy chuẩn tự động quét, tuyển chọn và tích hợp các kỹ năng chuyên sâu từ kho 100+ kỹ năng toàn cục vào không gian làm việc của dự án theo đúng domain kỹ thuật."
---

# Rule 03: Dynamic Skill Orchestration & Global Catalog Ingestion

## 1. Mục Đích & Nguyên Tắc Vận Hành
Không một AI nào có thể nắm trọn mọi best practice của hàng trăm công nghệ mà không có bộ kỹ năng chuyên biệt (Specialized Skills).
- Hệ thống duy trì kho kỹ năng tinh hoa toàn cục tại `C:\Users\Nguyen Trong Khoi\.gemini\config\skills\`.
- Agent **có trách nhiệm tự động quét, phân loại và nạp các kỹ năng chính xác** vào dự án dựa trên kết quả khảo sát Tech Stack ở Pha 1.

---

## 2. Ma Trận Ghép Nối Kỹ Năng Theo Domain (Domain-Skill Matrix)

| Lĩnh Vực Dự Án | Bộ Kỹ Năng Toàn Cục Khuyên Dùng | Mục Đích Sử Dụng |
| :--- | :--- | :--- |
| **Giao diện & Trải nghiệm (Frontend & UI/UX)** | `ui-ux-pro-max`<br>`react-patterns`<br>`nextjs-best-practices`<br>`css-expert` | Thiết kế giao diện đẳng cấp, Responsive, Accessible, Micro-animations. |
| **Hệ thống Backend & API** | `fastapi-pro`<br>`python-pro`<br>`golang-pro`<br>`rust-pro`<br>`nodejs-best-practices` | Xây dựng REST/GraphQL API hiệu năng cao, Concurrency, Clean Architecture. |
| **Cơ Sở Dữ Liệu & Lưu Trữ** | `database-admin`<br>`database-design`<br>`postgres-best-practices`<br>`redis-patterns` | Thiết kế Schema chuẩn hóa, Indexing tối ưu, Connection Pooling, Migrations an toàn. |
| **Bảo Mật & Kiểm Toán Mã Nguồn** | `security-auditor`<br>`api-security-best-practices`<br>`web-security-testing` | Quét lỗ hổng OWASP Top 10, SQLi, XSS, CSRF, IDOR, rò rỉ khóa bí mật. |
| **Kiểm Thử & Đảm Bảo Chất Lượng** | `tdd-workflow`<br>`systematic-debugging`<br>`e2e-testing-patterns` | Viết unit/integration tests đầy đủ, chu trình Red-Green-Refactor. |
| **Kiến Trúc & Tối Ưu Hệ Thống** | `architect-review`<br>`performance-engineer`<br>`monorepo-architect` | Đánh giá kiến trúc phân tầng, Clean Code, giải quyết thắt cổ chai hiệu năng. |
| **Dữ Liệu Lớn & Trí Tuệ Nhân Tạo (AI/Data)**| `machine-learning-ops-ml-pipeline`<br>`embedding-strategies`<br>`llm-app-patterns` | Pipeline huấn luyện, Vector Search, MLOps, Prompt Engineering. |
| **Blockchain & Web3** | `genomics-smart-contracts`<br>`genomics-data-crypto`<br>`ipfs-decentralized-storage` | Smart Contract Solidity, ERC-20/721, Web3.js, IPFS storage. |

---

## 3. Quy Trình Nạp Kỹ Năng Tự Động (Ingestion Protocol)

1. **Khảo sát:** Đọc danh sách kỹ năng hiện có trong `C:\Users\Nguyen Trong Khoi\.gemini\config\skills\`.
2. **Lựa chọn:** Xác định 2-4 kỹ năng cốt lõi cho dự án hiện tại.
3. **Kích hoạt:** 
   - Nếu cần nhúng trực tiếp vào dự án: Sử dụng lệnh xcopy chuẩn `cmd /c "xcopy /E /I /Y \"C:\Users\Nguyen Trong Khoi\.gemini\config\skills\<skill_name>\" \".agents\skills\<skill_name>\""` mà không cần cài đặt thêm script phụ.
   - Nếu kích hoạt qua ngữ cảnh: Đọc trực tiếp file `SKILL.md` của kỹ năng đó tại đường dẫn toàn cục để ghi nhớ và áp dụng toàn bộ quy tắc.

---

## 4. Anti-Pattern vs Positive Pattern

```diff
- // ANTI-PATTERN: Viết code giao diện bằng HTML/CSS đơn giản theo phong cách mặc định cũ kỹ
- // Kết quả: Giao diện sơ sài, người dùng thất vọng, vi phạm tiêu chuẩn sản phẩm.

+ // POSITIVE PATTERN: Kích hoạt ngay skill 'ui-ux-pro-max' và 'react-patterns'
+ // Áp dụng bảng màu HSL, typography Google Fonts cao cấp, micro-animations và card glassmorphism.
```
