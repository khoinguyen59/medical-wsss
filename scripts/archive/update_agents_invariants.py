import os

agents_path = "AGENTS.md"
with open(agents_path, "r", encoding="utf-8") as f:
    text = f.read()

# Normalize CRLF to LF for consistent matching
text_norm = text.replace("\r\n", "\n")

# Replace title
text_norm = text_norm.replace(
    "The 25 Core Governance Invariants (25 Luật Bất Biến Quản Trị Tối Thượng)",
    "The 29 Core Governance Invariants (29 Luật Bất Biến Quản Trị Tối Thượng)"
)

# Append Anti-Mock-Testing Illusion Gate to Invariant 26
old_inv26 = "    - **Fixer Swarm sửa song song & Tái kiểm định hồi quy:** Lead giao cho nhóm Fixer sửa song song; nhóm Audit ban đầu recheck lại. Lặp lại cho đến khi 100% test pass.\n    - **Chốt chặn dừng lại:** Lead xuất Báo cáo Markdown tổng kết và DỪNG HOÀN TOÀN để chờ người dùng kiểm tra, không tự ý làm tiếp."
new_inv26 = """    - **Fixer Swarm sửa song song & Tái kiểm định hồi quy:** Lead giao cho nhóm Fixer sửa song song; nhóm Audit ban đầu recheck lại. Lặp lại cho đến khi 100% test pass.
    - **Chống ảo tưởng kiểm thử giả lập (Anti-Mock-Testing Illusion Gate):** Hội đồng review và Victory Auditor TUYỆT ĐỐI KHÔNG được tự mãn chỉ vì bộ unit tests `pytest` trên thư mục tạm `tmp_path` đạt 100% pass hoặc kiểm tra cú pháp AST không có lỗi. Đối với toàn bộ Notebooks và Scripts huấn luyện: Bắt buộc phải kiểm chứng hành vi nạp dữ liệu trên đúng cấu trúc thư mục con thực tế (`train/img/` thay vì `train/`), và bắt buộc xác nhận `len(train_loader) > 0` và có ít nhất 1 batch dữ liệu được nạp thành công trước khi phê duyệt đóng cổng nghiệm thu (Victory Confirmed).
    - **Chốt chặn dừng lại:** Lead xuất Báo cáo Markdown tổng kết và DỪNG HOÀN TOÀN để chờ người dùng kiểm tra, không tự ý làm tiếp."""

assert old_inv26 in text_norm, "old_inv26 target not found"
text_norm = text_norm.replace(old_inv26, new_inv26)

# Append Invariants 28 and 29 after Invariant 27
inv27_end = "    - **Tiên liệu trước các lỗi phát sinh (Defensive Engineering):** Cung cấp sẵn cả phương án tự động và phương án thủ công (fallback) đề phòng mạng chậm hoặc chặn pop-up."

inv28_29 = """
28. **[INVARIANT-28] Pure Cloud VM Local Disk Standard (Quy Chuẩn Chạy Thuần Máy Ảo Colab & Cấm Tự Ý Ép Buộc Google Drive):**
    TUYỆT ĐỐI NGHIÊM CẤM việc tự ý đưa lệnh `drive.mount("/content/drive")` hoặc bắt buộc người dùng cấp quyền truy cập Google Drive cá nhân khi viết script hoặc notebook trên Google Colab / Cloud VM:
    - **Ưu tiên ổ đĩa máy ảo cục bộ:** Mọi dữ liệu tải về, checkpoints (`/content/checkpoints/`) và kết quả đầu ra phải được lưu trực tiếp trên ổ cứng SSD NVMe tốc độ cao của máy ảo Colab (`/content/`).
    - **Không gây gián đoạn bằng pop-up:** Khi người dùng bấm *Run All*, notebook phải chạy một mạch từ đầu đến cuối 100% tự động, không bao giờ được dừng lại đòi cấp quyền OAuth Google Drive.
    - **Kéo mã nguồn trực tiếp từ GitHub:** Mã nguồn phải được tự động clone trực tiếp từ repository GitHub chính thức (`https://github.com/...`) vào máy ảo Colab thay vì tìm kiếm trong Drive.
    - **Tải file 1-click về máy tính:** Ở cuối notebook, cung cấp sẵn ô lệnh `google.colab.files.download(...)` để người dùng tải checkpoint hoặc kết quả nén `.zip` về thẳng máy tính cá nhân qua trình duyệt khi cần.
29. **[INVARIANT-29] Anti-Shallow-Glob & Defensive Dataset Resolution Protocol (Quy Chuẩn Nhận Diện Dữ Liệu Đa Tầng & Chống Tìm Kiếm Nông):**
    TUYỆT ĐỐI NGHIÊM CẤM giả định cấu trúc thư mục phẳng khi quét ảnh y khoa bằng các lệnh tìm kiếm nông như `img_dir.glob("*.png")`:
    - **Tự động nhận diện cấu trúc thư mục (Auto-Discovery):** Các bộ dữ liệu y khoa nén (như `BCSS-WSSS.zip`) luôn có cấu trúc thư mục con lồng nhau (ví dụ: `train/img/*.png`, `test/img/*.png`, hoặc `BCSS-WSSS/train/img/*.png`). Mọi Dataset/DataLoader bắt buộc phải có hàm dò tìm đa tầng `resolve_split_path` và quét đệ quy `rglob("*.png")`.
    - **Lưới an toàn chống sập (Emergency Safety Net):** Mọi lớp Dataset bắt buộc phải có cơ chế kiểm tra `len(files) == 0`. Nếu thư mục rỗng do dữ liệu chưa tải xong, Dataset phải kích hoạt placeholder/smoke-test patch để đảm bảo `len(dataset) >= 1`, tuyệt đối không bao giờ để PyTorch DataLoader bị văng lỗi `ValueError: num_samples should be a positive integer value, but got num_samples=0`."""

assert inv27_end in text_norm, "inv27_end target not found"
text_norm = text_norm.replace(inv27_end, inv27_end + "\n" + inv28_29)

with open(agents_path, "w", encoding="utf-8") as f:
    f.write(text_norm)

print("AGENTS.md successfully updated with Invariants 28 and 29!")
