# SOP 00: Project Bootstrap & Autonomous Adaptation (Dual-Engine)

## 1. Mục Tiêu & Phạm Vi
Quy trình giúp Agent tự động thích ứng với dự án mới trong **lượt tương tác đầu tiên**, đặc biệt tối ưu cho trường hợp **dự án mới chỉ có Đề bài, Tài liệu yêu cầu (`.pdf`, `.docx`, `.md`, `.txt`) hoặc Prompt mô tả**.

---

## 2. Quy Trình Thực Thi Chi Tiết Theo Trạng Thái Dự Án

### 🟢 KỊCH BẢN 1: DỰ ÁN MỚI CHỈ CÓ ĐỀ BÀI / SPECS (GREENFIELD)

```
[QUÉT THẤY THƯ MỤC CHỈ CÓ ĐỀ BÀI]
               │
               ▼
[BƯỚC 1: ĐỌC & BÓC TÁCH ĐỀ BÀI (SPEC PARSING)]
- Tìm tất cả file tài liệu: *.pdf, *.docx, *.md, *.txt trong thư mục
- Đọc nội dung và lập Ma Trận Yêu Cầu (Requirements Matrix):
  + Tên đề tài & mục tiêu cốt lõi
  + Các Actor/Roles và phạm vi chức năng (Functional Requirements)
  + Tiêu chí đánh giá & chấm điểm (Rubric)
  + Ràng buộc công nghệ bắt buộc (nếu có)
               │
               ▼
[BƯỚC 2: NGHIÊN CỨU GIẢI PHÁP CHUẨN QUA FIRECRAWL & CONTEXT7]
- Dùng firecrawl_research_search_papers: Tìm các paper học thuật về đề tài
- Dùng firecrawl_research_search_github: Tìm 2-3 repo GitHub chuẩn mực cùng chủ đề
- Dùng Context7: Tra cứu tài liệu API mới nhất của các thư viện lựa chọn
- Lưu đề xuất kiến trúc vào docs/research/architecture_proposal.md
               │
               ▼
[BƯỚC 3: DỰNG BỘ KHUNG MÃ NGUỒN CHUẨN (SCAFFOLDING)]
- Tạo package.json / requirements.txt / Cargo.toml với dependencies mới nhất
- Dựng cấu trúc thư mục Clean Architecture:
  + src/controllers/ (hoặc routers/)
  + src/services/ (business logic)
  + src/models/ (schemas/entities)
  + src/repositories/ (database access)
  + tests/ (unit & integration tests)
- Tạo .env.example, .gitignore, README.md sơ bộ
               │
               ▼
[BƯỚC 4: NẠP SKILLS TƯƠNG ỨNG TỪ KHO 100+ SKILL & LẬP ĐỒ THỊ AST]
- Quét C:\Users\Nguyen Trong Khoi\.gemini\config\skills\
- Nạp các kỹ năng chuyên môn tương ứng (ví dụ: ui-ux-pro-max, fastapi-pro, database-admin)
- Chạy: cmd /c "graphify extract . --code-only" để thiết lập bản đồ AST baseline
               │
               ▼
[BƯỚC 5: TỰ TỐI ƯU HÓA AGENTS.md THEO ĐỀ BÀI]
- Cập nhật AGENTS.md: Đặt tên Agent theo đề tài, ghi rõ các Invariants của đề bài
- Cập nhật lệnh test thực tế vào Cổng nghiệm thu đối kháng 5 bước
```

---

### 🔵 KỊCH BẢN 2: DỰ ÁN ĐÃ CÓ SẴN MÃ NGUỒN (BROWNFIELD)

1. **Khảo sát mã nguồn:** Đọc `package.json`, `pyproject.toml`, cấu trúc thư mục hiện có.
2. **Lập đồ thị AST:** Chạy `cmd /c "graphify extract . --code-only"` để sinh `graphify-out/graph.json`.
3. **Phân tích God Nodes:** Chạy `cmd /c "graphify god-nodes --top 10"`.
4. **Nạp kỹ năng bổ trợ:** Nạp các skill phù hợp với stack đang dùng.
5. **Cập nhật AGENTS.md:** Tối ưu hóa các quy chuẩn và lệnh test.

---

## 3. Danh Mục Nghiệm Thu Khởi Tạo Dự Án (Bootstrap Checklist)
- [ ] Toàn bộ yêu cầu của đề bài đã được đọc và chuyển hóa thành Ma trận yêu cầu.
- [ ] Kiến trúc dự án đã được nghiên cứu đối chuẩn qua Firecrawl / Context7.
- [ ] Khung mã nguồn đã được tạo đầy đủ với dependencies không xung đột.
- [ ] File `graphify-out/graph.json` đã được tạo thành công.
- [ ] Các kỹ năng chuyên sâu đã được nạp vào `.agents/skills/`.
- [ ] File `AGENTS.md` đã được tối ưu hóa 100% bám sát đề bài.
