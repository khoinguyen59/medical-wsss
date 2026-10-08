---
id: rule-01-autonomous-research-protocol
title: Autonomous Research & Tooling Protocol (Firecrawl, Context7, Repomix)
activation: always_on
description: "Quy chuẩn bắt buộc về nghiên cứu độc lập: khai thác Firecrawl MCP để tìm paper học thuật và benchmark GitHub dựa trên đề tài đề bài, tra cứu API qua Context7, và cơ chế fallback an toàn."
---

# Rule 01: Autonomous Research & Tooling Protocol

## 1. Mục Đích & Nguyên Tắc Vận Hành
Khi nhận một đề tài mới toanh hoặc một yêu cầu bài toán:
- Tuyệt đối không chọn bừa bãi công nghệ hay dựng kiến trúc ngẫu hứng.
- **Bắt buộc phải nghiên cứu thực tế:**
  1. Đề tài này trên thế giới đã có những công trình khoa học nào (IEEE, arXiv, PubMed, ACM)?
  2. Những repository mã nguồn mở nào trên GitHub đang giải quyết bài toán này tốt nhất (Gold Standard Repositories)?
  3. Phiên bản API mới nhất của các công nghệ dự kiến sử dụng là gì (Context7)?

---

## 2. Tiêu Chuẩn Nghiên Cứu Khi Tiếp Nhận Đề Bài (Spec-Driven Research)

### 2.1. Tra Cứu Giải Pháp Học Thuật Theo Đề Tài
Sử dụng `firecrawl_research_search_papers`:
- **Ví dụ Đề tài Hệ thống Quản lý Logistics:**
  ```python
  firecrawl_research_search_papers(query="Smart Logistics Supply Chain Architecture IEEE", limit=3)
  ```
- **Ví dụ Đề tài Nhận diện Xâm nhập mạng (IDS) Big Data:**
  ```python
  firecrawl_research_search_papers(query="IoT Network Intrusion Detection Machine Learning CSE-CIC-IDS2018", limit=3)
  ```
- Trích xuất: Các pipeline xử lý dữ liệu chuẩn mực, các độ đo đánh giá (Precision, Recall, F1, Latency).

### 2.2. Khảo Sát Repository GitHub Đối Chuẩn (Benchmark Repos)
Sử dụng `firecrawl_research_search_github`:
- **Tìm kiếm repo đã hoàn thiện bài toán tương tự:**
  ```python
  firecrawl_research_search_github(query="<chủ_đề_đề_bài> clean architecture production", limit=5)
  ```
- Phân tích cấu trúc thư mục, các dependencies đã chọn và luồng dữ liệu (Data Flow) để học hỏi.

### 2.3. Tra Cứu Tài Liệu API Bằng Context7
- Dùng `resolve-library-id` và `query-docs` để kiểm tra chuẩn cú pháp của các framework dự kiến dùng trong đề tài (FastAPI, React, Spring Boot, Pydantic, Ethers...).

### 2.4. Quy Chuẩn Bắt Buộc: Tải Bài Báo Về Máy, Đọc Bằng Vision & Viết File Markdown (Vision-to-Markdown Protocol)
- **Quy tắc bắt buộc:** Mọi tài liệu nghiên cứu và bài báo khoa học (từ arXiv, IEEE, Springer, PubMed hoặc do Giảng viên cung cấp):
  1. **Tải về lưu trữ:** Bắt buộc tải file PDF gốc về lưu trong thư mục `docs/papers/<paper_id>.pdf`.
  2. **Đọc trực quan bằng Vision (Multimodal Reading):** Dùng tính năng Vision / đa phương thức để duyệt từng trang, đọc sơ đồ kiến trúc khối (pipeline diagrams), biểu đồ suy giảm, bảng so sánh định lượng và công thức toán học.
  3. **Viết file Markdown đối ứng:** Bắt buộc tạo file `docs/papers/<paper_id>.md` ghi lại toàn văn cấu trúc hóa:
     - Tên bài báo, tác giả, hội nghị/tạp chí, năm công bố, mã định danh (DOI / arXiv ID).
     - Bối cảnh & Động lực nghiên cứu (Problem Motivation & Formulation).
     - Đóng góp cốt lõi (Key Contributions).
     - Mô hình kiến trúc chi tiết (Model Architecture, Components, Mathematical Loss Functions).
     - Bộ dữ liệu thực nghiệm (Datasets, Preprocessing, Patient Splits).
     - Bảng số liệu đối chứng thực nghiệm (Benchmark Tables & SOTA Comparisons).
     - Điểm mạnh, điểm yếu và khả năng kế thừa cho đồ án.
- **Lý do kỹ thuật tối thượng:**
  * **AI đọc PDF ảnh / cột đôi rất kém:** Các file PDF nghiên cứu thường định dạng 2 cột, chứa phương trình toán học và ảnh chụp bảng biểu. Đọc PDF thô bằng công cụ text parser thường xuyên bị nuốt ký hiệu, tráo thứ tự dòng giữa 2 cột, hoặc mất bảng biểu.
  * **Tái sử dụng thiếu hiệu quả & Lãng phí Context Token:** Việc đọc lại file PDF dung lượng lớn mỗi khi cần viết Proposal hoặc báo cáo gây tắc nghẽn Context Window và suy giảm độ chính xác.
  * **Hiệu quả tái sử dụng vượt trội:** Một file `.md` chuẩn hóa giúp AI Agent đọc lại tức thì với chi phí token tối thiểu, trích xuất số liệu chuẩn xác 100% để đối chiếu học thuật.

---

## 3. Quy Trình Fallback An Toàn

```
[Bắt đầu nghiên cứu Đề Tài]
           │
           ▼
    {Có Firecrawl MCP?}
      /            \
   [Có]            [Không / Lỗi Token]
    │                      │
    ▼                      ▼
[Gọi Firecrawl]    [Kích hoạt Fallback Đa Tầng]
                   1. search_web (Tìm tài liệu chính thức & bài báo)
                   2. Context7 (query-docs)
                   3. Đọc tài liệu local trong docs/research/
```

---

## 4. Lệnh Kiểm Tra Đối Kháng
Mọi kết quả nghiên cứu đề tài phải được ghi nhận vào `docs/research/`:
```bash
cmd /c "dir /b docs\research || mkdir docs\research"
```


---

## 5. Quy Chuẩn 4 Trụ Cột Nghiên Cứu Chuyên Sâu (The 4 Research & Engineering Pillars)

Để tránh bẫy đọc lướt bề nổi, mất thông tin trực quan và biến dạng công thức toán học khi đọc PDF thô, mọi quy trình nghiên cứu và cài đặt bắt buộc phải tuân thủ nghiêm ngặt 4 trụ cột:

1. **Trụ Cột 1: Đọc Đa Phương Thức (Multimodal Visual Reading):**
   - Render các trang chứa Figure 1 (Sơ đồ kiến trúc), Figure 2 (Ví dụ minh họa), Algorithm 1 thành ảnh PNG độ phân giải cao (`docs/papers/figures/[paper_id]/`).
   - AI Agent kích hoạt thị giác máy tính (`view_file` trên ảnh PNG) để đọc trực tiếp sơ đồ, cấu trúc đồ thị con và luồng xử lý nguyên bản của tác giả.

2. **Trụ Cột 2: Tận Dụng Toàn Diện 19 Bài Báo Qua Thẻ Phân Tích Chuyên Sâu (Deep Analysis Cards):**
   - Không bỏ sót bất kỳ bài báo nào trong số 19 bài báo đỉnh cao (2019 - 2026).
   - Mỗi bài báo được bóc tách thành một thẻ phân tích chuyên sâu lưu tại `docs/research_notes/cards/[paper_id].md` gồm 5 thành phần: (1) Tóm tắt toàn diện không thiếu sót; (2) Phân tích sơ đồ kiến trúc trích từ ảnh gốc; (3) Công thức toán học 100% LaTeX chuẩn xác; (4) Những điểm cụ thể kế thừa vào đồ án C.10; (5) Đường dẫn đối chiếu ngược về tệp PDF và ảnh sơ đồ.
   - **Quy trình đọc:** Khi nghiên cứu & code, đọc các thẻ phân tích này trước. Ưu tiên các bài nòng cốt để code, khi cần tối ưu/cải tiến thì đọc tiếp các bài còn lại. Nếu thẻ còn thiếu thông tin ngách nào, quay lại đọc trực tiếp tệp PDF gốc.

3. **Trụ Cột 3: Đối Soát Mã Nguồn Từng Dòng (Code-to-Code Inheritance Mapping):**
   - Mở trực tiếp các file mã nguồn đã clone trong `repos/` (DIG, EiG-Search, GraphFramEx, GraphNarrator).
   - Thiết lập bảng ánh xạ kế thừa (Code Inheritance Matrix), đối chiếu từng hàm toán học với dòng code thực thi trong `src/`, kế thừa logic chuẩn của tác giả thay vì tự tưởng tượng code.

4. **Trụ Cột 4: Quy Trình Nghiệm Thu 3 Bước Có Chốt Chặn (Gated 3-Step Verification):**
   - Chấm dứt việc "code một lần" (One-shot Coding).
   - Mỗi module khi cài đặt hoặc tinh chỉnh phải qua 3 bước:
     * Bước 1 (Lý thuyết & Trực quan): Trình bày ảnh sơ đồ + công thức toán LaTeX phục hồi từ bài báo.
     * Bước 2 (Mã nguồn đối chứng): Chỉ ra đoạn code mẫu tương ứng trong `repos/` của tác giả.
     * Bước 3 (Thực nghiệm kiểm chứng): Chạy unit test kiểm nghiệm đối chiếu với baseline trước khi coi là hoàn thành.
