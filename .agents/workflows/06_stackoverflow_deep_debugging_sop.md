# SOP 06: StackOverflow & Developer Community Debugging Workflow

## 1. Mục Tiêu & Phạm Vi
Quy trình hướng dẫn Agent cách cô lập, tra cứu và giải quyết triệt để các lỗi kỹ thuật phức tạp (runtime exceptions, build failures, type errors) bằng cách sử dụng dữ liệu từ **StackOverflow** và **GitHub Issues**.

---

## 2. Quy Trình Thực Thi 4 Bước Chi Tiết

```
[GẶP LỖI BIÊN DỊCH / RUNTIME / TEST THẤT BẠI]
                     │
                     ▼
[BƯỚC 1: TRÍCH XUẤT CHỮ KÝ LỖI ĐỘC NHẤT (ERROR SIGNATURE)]
- Lọc bỏ đường dẫn máy cá nhân (C:\Users\...) và số dòng ngẫu nhiên.
- Giữ lại: [Loại lỗi] + [Thông điệp lỗi cốt lõi] + [Tên thư viện/framework].
                     │
                     ▼
[BƯỚC 2: TRA CỨU STACKOVERFLOW & GITHUB ISSUES QUA CÔNG CỤ CÓ SẴN]
- Gọi search_web hoặc firecrawl_developer_search:
  site:stackoverflow.com "<error_signature>" "<framework>"
- Hoặc dùng github MCP tra cứu:
  search_issues(query="repo:<owner>/<repo> <error_signature>")
                     │
                     ▼
[BƯỚC 3: PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ & CHỌN GIẢI PHÁP]
- Đọc câu trả lời có dấu tích xanh (Accepted Answer) hoặc nhiều upvotes nhất.
- Nắm rõ: Tại sao lỗi này phát sinh? (Do lifecycle, async, kiểu dữ liệu, hay version conflict?)
- Tránh các giải pháp "chữa cháy" tạm thời (như tắt cảnh báo linter, bắt lỗi rỗng).
                     │
                     ▼
[BƯỚC 4: TRIỂN KHAI BẢN VÁ & CHẠY LẠI KIỂM THỬ THỰC TẾ]
- Chỉnh sửa code theo đúng giải pháp chuẩn từ cộng đồng.
- Chạy lại lệnh test shell ngay lập tức (cmd /c "<test_cmd>").
- Xác nhận lỗi đã được khắc phục hoàn toàn.
```

---

## 3. Bảng Kiểm Tra Sau Khi Sửa Lỗi (Post-Fix Checklist)
- [ ] Lỗi ban đầu không còn tái diễn khi chạy lại lệnh test.
- [ ] Không sinh thêm lỗi phụ hoặc làm vỡ các module liên quan khác.
- [ ] Code sửa đổi tuân thủ đúng nguyên lý khuyến nghị từ StackOverflow.
- [ ] Đã chạy `graphify update .` nếu việc sửa lỗi làm thay đổi cấu trúc tệp hoặc hàm.
