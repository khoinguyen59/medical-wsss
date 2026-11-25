# Thư Mục Sản Phẩm Bàn Giao (Deliverables & Submission Products)

Thư mục `product/` lưu trữ toàn bộ ấn phẩm học thuật và tài liệu kỹ thuật chính thức cho đồ án môn học **SE374.R11 - Phát triển phần mềm xử lý ảnh y khoa (UIT, ĐHQG-HCM, 2026)** - Đề tài 6: Phân vùng ngữ nghĩa giám sát yếu (WSSS) trên tiêu bản giải phẫu bệnh ung thư vú **BCSS-WSSS (TCGA-BRCA)**.

- **Thông tin hành chính:** `[CẦN XÁC NHẬN: Họ tên sinh viên / MSSV / GVHD]`
- **Kho minh chứng mật mã (Phase 1B khóa cứng):** `checkpoints/vps_evidence/SHA256SUMS.txt` (`228/228` tệp bit-exact, `BAD = 0`, `UNLISTED = 0`) gồm `6/6` phiên GPU hậu bản vá (`02c_v2`, `04b_v2`, `02d_v2`, `H1`, `H2`, `H3`) cùng phân tích thống kê không cần GPU `phase1b_zero_gpu_analytics.json`.
- **Chỉ số đại diện đã khóa (`02c_v2`, `n = 4,986` Held-Out Test / `23` BN, `0%` TCGA-BRCA overlap):**
  - Primary 4-Arm Track A (`Uniform`, `cal_logprob_eps1e3`, `12x ms_3_hvflip`): **`72.79%` mIoU (`84.04%` mDice)** ở `no-CGM`, **`72.80%` mIoU (`84.06%` mDice, `NEC = 70.95%`, xóa sạch `6/8` BN vắng NEC về `0 px`)** ở `cgm_nec_focused_1dof` (`t_nec = 0.40`).
  - Secondary Simplex-CV 4-Arm (`[0.61, 0.06, 0.05, 0.28]`): **`72.90%` mIoU (`84.11%` mDice)**; `arm7` ConvNeXt-L hậu bản vá `_pad_to_multiple_of_32` đạt **`71.88%` mIoU** (`202.51 FPS` BF16 trên `NVIDIA L4`).
  - Chéo cơ quan Zero-Shot `BCSS (40x Vú) -> LUAD-HistoSeg (10x Phổi)` (`04b_v2`, `n = 307`, `20` ca): **`36.09%`** (`arm5_cgm` ở `10x`) và **`51.47%`** (`arm5_no_cgm_2x_zoom`) / **`49.92%`** (`ens4_uncal_linear_2x_zoom`) khi phóng đại `2x` không huấn luyện lại.
  - Phủ kiểm thử tự động & phần mềm (Stage 2): **`541` bài kiểm thử** (`505` ca cục bộ tại `tests/` + `36` ca tại `colab_runner_local/`, xác minh **`36/36` PASS** trên Linux VPS), Trạm Lâm Sàng FastAPI + HTML5 Canvas 100% ngoại tuyến (Docker Compose) và bộ điều phối `colab_runner_local/` (`colab_account`, `H100 -> G4,A100,L4`, bảo vệ P0 external_service).

```text
product/
|-- proposal/    # Đề cương Mini Proposal (Markdown .md, Word .docx, PDF .pdf, HTML .html)
|-- reports/     # Báo cáo kỹ thuật cuối kỳ & Báo cáo kiểm định lỗi (.md, .docx) + 9 hình 300-DPI (figures/)
`-- slides/      # Slide bảo vệ đồ án 16:9 Academic Monochrome (Marp .md, .pptx, .pdf, .html)
```

---

## 1. Chi Tiết Các Phân Mục Bàn Giao

### 1.1 Đề Cương Mini Proposal (`product/proposal/`)

- `SE374_MINI_PROPOSAL_WSSS_WSI.md` (`.docx`, `.pdf`, `.html`): Đề cương nghiên cứu tích hợp WSSS-Tissue và WSI.
- `SE374_MINI_PROPOSAL_WSSS_TISSUE.md` & `SE374_MINI_PROPOSAL_PHASED_ROADMAP_WSSS.md` (`.docx`, `.pdf`, `.html`): Lộ trình triển khai theo giai đoạn.

### 1.2 Báo Cáo Kỹ Thuật Cuối Kỳ & Kiểm Định Đa Tác Tử (`product/reports/`)

- [SE374_FINAL_TECHNICAL_REPORT_WSSS.md](reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md) & `SE374_FINAL_TECHNICAL_REPORT_WSSS.docx`: Báo cáo kỹ thuật tổng kết toàn diện đồng bộ 100% số liệu Phase 1B (`docs/research/STAGE2_FINAL_EXPERIMENT_SUMMARY.md`) và sản phẩm Stage 2, biên dịch chuẩn mẫu UIT (`--reference-doc=docs/templates/SE374_Reference_Template.docx --syntax-highlighting=none`).
- [SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md](reports/SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md) & `SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.docx`: Biên bản kiểm toán đa tác tử và nhật ký khắc phục lỗi xuyên suốt từ kiểm toán 6 Teams ban đầu đến Phase 1B (`1d3f738`, `8dabf65`, `c0edb2f`), Stage 1 (`47a1bab`) và Stage 2 (`079fee0`).
- `figures/` (`fig1` đến `fig9`): Bộ `8` hình khoa học Đơn Sắc Học Thuật (`#FFFFFF`, `300 DPI` PNG + vector SVG cho `fig1-fig4`) tái sinh trực tiếp từ JSON thực nghiệm Phase 1B (`02c_v2`, `04b_v2`, `02d_v2`) cùng `fig9_whole_roi_he_overlay_audit.png` (kiểm chứng trực quan lớp phủ H&E `TCGA-EW-A1P4` `14x17` tiles).

### 1.3 Slide Thuyết Trình Bảo Vệ (`product/slides/`)

- [SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.md](slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.md): Mã nguồn Marp 16:9 chuẩn Academic Monochrome nền trắng `#FFFFFF`, đồng bộ 100% số liệu Phase 1B và Stage 2.
- Biên dịch hoàn chỉnh sang ba định dạng trình chiếu: `SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.pptx`, `SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.pdf`, và `SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.html`.
- `SE374_PROPOSAL_SLIDES_WSSS_WSI.md` (`.pptx`, `.html`): Bộ slide đề cương giai đoạn đầu.

---

## 2. Lệnh Biên Dịch & Kiểm Định Tự Động

```powershell
# 1. Biên dịch báo cáo Word (.docx) theo mẫu chuẩn UIT (tắt syntax-highlighting để giữ mực đen #000000):
pandoc product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md --reference-doc=docs/templates/SE374_Reference_Template.docx "--resource-path=product/reports;product/reports/figures;." --syntax-highlighting=none -o product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.docx

# 2. Biên dịch Slide bảo vệ 16:9 (Marp CLI):
marp product/slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.md --allow-local-files --pptx -o product/slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.pptx
marp product/slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.md --allow-local-files --pdf -o product/slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.pdf
marp product/slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.md --allow-local-files --html -o product/slides/SE374_FINAL_DEFENSE_SLIDES_WSSS_WSI.html

# 3. Kiểm định 4 tầng độc lập (DeliverableAuditor - bắt buộc 0 CRITICAL, 0 WARNING):
python src/utils/deliverable_auditor.py README.md product/README.md "product/reports/*.md" "product/reports/*.docx" "product/slides/*.md" "docs/architecture/*.md" "docs/research/*.md"
```
