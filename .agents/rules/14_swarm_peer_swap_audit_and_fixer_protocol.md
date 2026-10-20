# Rule 14: Automated Triple-Blind Peer-Swap Audit & Repo-Grounded Fixer Swarm Protocol

> **Specification Standard:** `agents.md` v5.0 - Multi-Agent Autonomous Swarm Protocol  
> **Trigger Phrases:** Khi người dùng ra các lệnh rút gọn như:  
> - "kêu gọi team recheck"  
> - "triệu tập team kiểm tra"  
> - "team recheck"  
> - "kêu team check lại"  
> - "/teamwork-preview /boost"  
> Hệ thống BẮT BUỘC tự động kích hoạt và tuân thủ 100% quy trình sau mà người dùng không cần gõ lại prompt dài.

---

## 1. Cơ Cấu Tổ Chức Bắt Buộc

1. **Lead Agent (Tổng Chỉ Huy - Tuyệt Đối Không Trực Tiếp Sửa Code):**
   - Chỉ đóng vai trò điều phối: tiếp nhận danh sách lỗi đã qua kiểm định, phân rã công việc thành các gói độc lập, giao cho Nhóm Fixer, nhận nghiệm thu từ Nhóm Recheck và lập Báo cáo Markdown cho User.
2. **Audit Swarm (Nhóm Tác Tử Kiểm Định):**
   - Bắt buộc khởi tạo số lượng từ **6 đến 10 Agent và LUÔN LÀ SỐ CHẴN** ($N \in \{6, 8, 10\}$) để đảm bảo việc bắt cặp đối soát chéo hoàn hảo, không có tác tử lẻ.
3. **Fixer Swarm (Nhóm Tác Tử Sửa Lỗi - Triệu Tập Theo Nhu Cầu):**
   - Tiếp nhận các gói lỗi độc lập từ Lead và thực thi sửa song song cùng lúc (Parallel Fixing).
4. **Nguyên Tắc Tham Chiếu Tri Thức Nguồn Bắt Buộc (Repo & Paper Grounding):**
   - Cả Nhóm Check và Nhóm Fix BẮT BUỘC đối soát trực tiếp với:
     * Toàn bộ mã nguồn từ các GitHub repository đã clone về trong `repos/`.
     * Toàn bộ các bài báo khoa học, công thức toán và tài liệu đã tải về trong `docs/papers/`.
   - Mọi kết luận lỗi và giải pháp fix bắt buộc phải ghi rõ nguồn trích dẫn: `[Tên Repo / File code / Bài báo / arXiv hoặc DOI / Công thức tham chiếu]`.

---

## 2. Quy Trình Vận Hành 5 Giai Đoạn Khép Kín

### Giai Đoạn 1: Săn Lỗi Đồng Thời Kèm Đối Soát Tri Thức Gốc (Parallel Grounded Hunting)
- $N$ Audit Agent đồng loạt chạy song song quét toàn bộ codebase và tài liệu theo 6 chiều kích (Đề bài, Kiến trúc, Toán học/Loss, Rò rỉ dữ liệu, Test đối kháng, Bộ nhớ VRAM).
- Mỗi agent độc lập viết các unit test đối kháng mới (cấm chỉ chạy lại test cũ) và đối soát với repo/paper gốc để lập danh mục nghi vấn.

### Giai Đoạn 2: Ghép Cặp Hoán Đổi Vòng 1 (FIFO Dynamic Pairwise Swap)
- Bất kỳ 2 Audit Agent nào xong trước lập tức ghép cặp và hoán đổi toàn bộ danh mục lỗi cho nhau để recheck (Ví dụ: $A_1 \leftrightarrow A_2, A_3 \leftrightarrow A_4, A_5 \leftrightarrow A_6$).
- Từng agent trực tiếp thẩm tra và chạy thử test case lỗi của bạn mình để phản biện.

### Giai Đoạn 3: Kiểm Soát Chéo Vòng Cuối Hoán Vị Mới (Disjoint Final Cross-Shuffle)
- Sau khi xong Vòng 1, các Agent BẮT BUỘC thực hiện Vòng Kiểm Tra Chéo Quyết Định.
- **LUẬT BẤT BIẾN:** Tuyệt đối KHÔNG ĐƯỢC kiểm tra lại đối tác đã kiểm ở Vòng 1. Bắt buộc hoán vị sang một Agent hoàn toàn mới (Ví dụ: $A_1 \leftrightarrow A_3, A_2 \leftrightarrow A_5, A_4 \leftrightarrow A_6$).
- **Tiêu chuẩn xác thực:** Một lỗi chỉ được coi là Lỗi Xác Thực (True Defect) để trình lên Lead khi được ít nhất 3 Agent độc lập đồng thuận xác nhận và có bằng chứng đối soát từ repo/paper gốc.
- Nhóm Audit thống nhất và gửi Biên bản Lỗi Xác Thực duy nhất lên Lead.

### Giai Đoạn 4: Phân Rã & Sửa Lỗi Song Song Chuẩn SOTA (Lead Dispatch & Parallel Fix)
- Lead tiếp nhận danh sách lỗi đã xác thực, phân rã thành các gói công việc độc lập.
- Lead giao cho Fixer Swarm sửa các lỗi khác nhau cùng một lúc.
- Fixer Swarm tra cứu giải pháp chuẩn từ repo/paper gốc để patch mã nguồn đúng bản chất, không dùng dummy code.

### Giai Đoạn 5: Tái Kiểm Định Hồi Quy & Vòng Lặp Khép Kín (Regression Re-Audit Loop)
- Nhóm Audit Swarm ban đầu quay lại kiểm tra mã nguồn thực tế sau khi fix (Fixer không được tự duyệt code).
- Chạy toàn bộ test suite (test kế thừa + test mới sinh) để kiểm tra tính hồi quy.
- **Nhánh rẽ:**
  * **Nếu còn lỗi:** Nhóm Audit quay lại quy trình hoán đổi chéo xác thực lỗi -> Gửi Lead -> Lead giao Fixer sửa tiếp.
  * **Nếu 100% Pass:** Nhóm Audit ký nghiệm thu gửi Lead. Lead lập Báo cáo Markdown tổng hợp kèm trích dẫn nguồn.

---

## 3. Điều Kiện Dừng Tuyệt Đối (Hard Stop Gate)
- Lead Agent xuất một bản Báo cáo Markdown duy nhất (ghi rõ lỗi phát hiện, biên bản hoán đổi 2 vòng, nguồn repo/paper tham chiếu, nhật ký sửa của Fixer Swarm, và log chạy test thật 100% pass).
- **DỪNG HOÀN TOÀN** tại bước tệp Markdown này để User trực tiếp kiểm tra và nghiệm thu, tuyệt đối không tự ý chuyển sang bước tiếp theo.
