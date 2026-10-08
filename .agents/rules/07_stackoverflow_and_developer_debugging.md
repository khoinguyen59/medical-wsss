---
id: rule-07-stackoverflow-and-developer-debugging
title: StackOverflow & Developer Community Debugging Protocol
activation: always_on
description: "Quy chuẩn khai thác tri thức cộng đồng lập trình viên: tra cứu StackOverflow và GitHub Issues bằng chuỗi lỗi thực tế để tìm nguyên nhân gốc rễ và giải pháp sửa lỗi chuẩn xác."
---

# Rule 07: StackOverflow & Developer Community Debugging Protocol

## 1. Mục Đích & Nguyên Tắc Tối Thượng
Khi lập trình với các framework hiện đại (React, Next.js, FastAPI, Hardhat, Docker, Spring Boot...), hầu hết các lỗi khó (Edge-case bugs, Circular dependencies, Hydration errors, Async deadlocks, Type mismatch) đều đã từng được các lập trình viên khác trên thế giới gặp phải và giải quyết.
- **Cấm đoán mò (No Guesswork Debugging):** Khi gặp một lỗi lạ hoặc test case thất bại sau 1 lần thử sửa chữa, Agent **bắt buộc phải tra cứu tri thức cộng đồng**.
- **Hiểu rõ nguyên nhân gốc rễ (Root Cause First):** Không được sửa mù quáng bằng cách bọc bừa `try/catch` hoặc tắt cảnh báo linter. Phải tìm ra vì sao lỗi phát sinh dựa trên các thảo luận được chứng thực trên StackOverflow hoặc GitHub Issues.

---

## 2. Quy Trình Tra Cứu Lỗi 4 Bước Chuẩn Mực

### Bước 1: Trích Xuất Chữ Ký Lỗi Độc Nhất (Error Signature)
- Loại bỏ các đường dẫn tệp cục bộ (ví dụ: `C:\Users\...`) để giữ lại thông điệp lỗi cốt lõi.
- **Ví dụ chuỗi lỗi gốc:**
  `TypeError: Cannot read properties of undefined (reading 'callStatic') at ContractFactory.deploy (C:\Users\Nguyen Trong Khoi\...)`
- **Chữ ký lỗi trích xuất:**
  `"Cannot read properties of undefined (reading 'callStatic')" "ethers.js v6"`

### Bước 2: Truy Vấn StackOverflow Có Mục Tiêu (Targeted Query)
Sử dụng công cụ `search_web` hoặc `firecrawl_developer_search` với cú pháp chuyên gia:
```python
# Cú pháp tìm kiếm StackOverflow
search_web(query='site:stackoverflow.com "Cannot read properties of undefined" "callStatic" "ethers v6"')

# Cú pháp tìm kiếm GitHub Issues của thư viện
search_web(query='site:github.com/ethers-io/ethers.js/issues "callStatic"')
```

### Bước 3: Đánh Giá Câu Trả Lời Chuẩn Mực
Ưu tiên lựa chọn:
1. **Dấu tích xanh (Accepted Answer):** Câu trả lời được người đặt câu hỏi xác nhận hiệu quả.
2. **Số lượng Upvote cao (>= 10 upvotes):** Thể hiện sự đồng thuận của cộng đồng.
3. **Phù hợp với phiên bản hiện tại:** Đọc kỹ năm trả lời (ưu tiên từ 2023 trở lại đây nếu là thư viện mới).

### Bước 4: Áp Dụng Bản Vá & Chạy Lại Kiểm Định
- Sửa mã nguồn theo đúng nguyên lý giải thích từ StackOverflow.
- Chạy lại lệnh test qua shell ngay lập tức để xác minh lỗi đã được triệt tiêu 100%.

---

## 3. Anti-Pattern vs Positive Pattern

```diff
- // ANTI-PATTERN: Gặp lỗi Hydration của Next.js, agent tự ý thêm 'suppressHydrationWarning' bừa bãi
- // Hậu quả: Lỗi gốc rễ không được giải quyết, gây ra lỗi layout hoặc memory leak ngầm.

+ // POSITIVE PATTERN: Tra cứu StackOverflow -> Hiểu nguyên nhân do dùng window.localStorage trong render server
+ // Sửa bằng useEffect hoặc dynamic import với { ssr: false } theo đúng best practice cộng đồng.
```

---

## 4. Lệnh Kiểm Tra Sau Khi Fix Bug
```bash
# Bắt buộc chạy lại test suite sau khi áp dụng giải pháp từ StackOverflow
cmd /c "<test_command>"
```
