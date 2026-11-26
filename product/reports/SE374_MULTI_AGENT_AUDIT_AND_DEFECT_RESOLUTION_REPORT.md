# BÁO CÁO KIỂM ĐỊNH ĐA TÁC TỬ VÀ BIÊN BẢN KHẮC PHỤC LỖI TOÀN DIỆN

## SE374.R11: Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa - UIT 2026

- **Học phần:** SE374.R11 - Phát triển phần mềm xử lý ảnh y khoa
- **Khoa:** Kỹ thuật Phần mềm, Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM
- **Thông tin hành chính:** `[CẦN XÁC NHẬN: Họ tên sinh viên / MSSV / GVHD]`
- **Đề tài:** Đề tài 6 - Phân vùng ngữ nghĩa giám sát yếu trên tiêu bản giải phẫu bệnh ung thư vú (BCSS-WSSS)
- **Phạm vi nhật ký kiểm định:** Khép vòng hoàn chỉnh từ Kiểm toán 6 Teams ban đầu -> **Phase 1A-1B (`1d3f738`, `8dabf65`, `c0edb2f`)** -> **Stage 1 (`47a1bab`)** -> **Stage 2 (`079fee0`)** -> **Stage 3 Deliverables & Polish**
- **Trạng thái hiện tại:** **100% lỗi xác thực đã khắc phục và kiểm thử hồi quy (`501` ca kiểm thử phần mềm y khoa trong `tests/`, `0` thất bại, `ruff check src tests` `0` lỗi, `0` rò rỉ thông tin nội bộ/tài khoản, `228/228` SHA256 OK)**

---

## 1. Tổng Quan Kiểm Toán Ban Đầu (6 Teams Độc Lập - 8 Chiều Kích)

Nhằm triệt tiêu thiên vị tự kiểm (self-referential confirmation bias), đợt kiểm toán nền tảng đầu tiên huy động **6 nhóm kiểm định độc lập** (`Audit Team 1` đến `Audit Team 6`) rà soát đồng thời toàn bộ 8 chiều kích (bám đề bài môn học, logic mã nguồn và Clean Architecture, toán học hàm mất mát, đối soát code-to-code với `repos/WSSS-Tissue` / `repos/LowMagWSS`, chống rò rỉ 151 bệnh nhân, viết unit test đối kháng mới, thẩm định bệnh học lâm sàng VTP/TSR/Otsu, và kiểm toán API + ấn phẩm).

### 1.1 Kết Quả Phân Loại 18 Phát Hiện Ban Đầu (`DEF-01` đến `DEF-18`)

- **Số lỗi được xác thực là lỗi thật và đã khắc phục triệt để:** **`16/16` lỗi (`100%`)**.
- **Ghi chú lý thuyết (`1` mục):** `DEF-17` (cắt lát kênh cho CAM tầng nông `b4_5` trong `backbone.py` là xấp xỉ chấp nhận được ở Giai đoạn 1, đã ghi rõ giới hạn).
- **Bác bỏ dương tính giả (`1` mục):** `DEF-18` (BCSS-WSSS dùng chuẩn mã TCGA `TCGA-XX-XXXX` đã tách chính xác bằng Regex chuyên dụng).

| Mã Lỗi | Vị Trí Tệp Tin | Mô Tả Khuyết Tật & Biện Pháp Khắc Phục | Trạng Thái |
| :---: | :--- | :--- | :---: |
| **DEF-01 / 02** | `src/datasets/otsu_filter.py` | Lỗi `IndexError` batch `B > 1` và `cv2.error` trên tensor 4D -> Hỗ trợ duyệt trục batch `(B, 3, H, W)` và mặt nạ 3D `(B, H, W)`. | Đã sửa |
| **DEF-03 / 04 / 05 / 06** | `src/losses_metrics/losses.py` | Thiếu `ignore_index=255`, rò rỉ gradient trọng số bất định, `Device Mismatch` CUDA/CPU và đứt `grad_fn` -> Bổ sung `ignore_index=255`, `.detach()`, hằng số `math.log(C)`, và `pred_logits.sum() * 0.0`. | Đã sửa |
| **DEF-07** | `src/datasets/bcss_dataset.py` | `self.transform` không được gọi trong `__getitem__` -> Kích hoạt chuẩn hóa tương thích Tensor/PIL/NumPy. | Đã sửa |
| **DEF-08 / 09** | `src/visualization/api_server.py` | Thiếu 5 route RESTful (`/health`, `/segment`, `/biomarkers`, `/splits`, `/tile/{z}/{x}/{y}.png`) và hạ mẫu xuống `96x96` -> Bổ sung đủ route và khôi phục độ phân giải gốc `224x224`. | Đã sửa |
| **DEF-10** | `src/models/wsss_tissue.py` | Nguy cơ tràn VRAM OOM khi tính Spatial Affinity ở `224x224` -> Hạ mẫu về nút cổ chai `14x14` (`196x196`, `< 1 MB`) trước khi nội suy ngược. | Đã sửa |
| **DEF-11 / 12** | `src/losses_metrics/metrics.py` & Báo cáo | Đảo ngược chiều nguy cơ TSR và lẫn hai định nghĩa VTP -> Chuẩn hóa `TSR <= 50%` là Stroma-Rich High Risk (Mesker/UNITED) và tách rõ `VTP_cellularity` vs `VTP_RCB`. | Đã sửa |
| **DEF-13 / 14 / 15 / 16** | Báo cáo & tài liệu đặc tả đề tài | Minh bạch hóa `GroupNorm(1, C)` chỉ nằm trong `ASPPPooling`, thay mã bệnh nhân giả định bằng mã thật từ `bcss_patient_splits.json`, sửa đúng nhóm tác giả ARML (MICCAI 2024), và khôi phục đặc tả đề tài đồ án. | Đã sửa |

---

## 2. Nhật Ký Khắc Phục Lỗi Giai Đoạn Phase 1A - Phase 1B (`1d3f738`, `8dabf65`, `c0edb2f`)

Trong đợt kiểm toán chéo đa luồng độc lập 3 hệ thống cho chiến dịch Gate-7 Track A, hội đồng phát hiện các cờ đỏ khoa học và lỗi hình học tinh vi ở tầng suy luận. Toàn bộ đã được khắc phục triệt để trong 3 commit lõi và tái kiểm chứng bằng **`6/6` phiên chạy GPU thật hậu bản vá** trên NVIDIA A100, G4 Blackwell và L4:

| Mã Hạng Mục | Commit | Vị Trí & Nguyên Nhân Gốc | Giải Pháp & Bằng Chứng Kiểm Chứng Hậu Bản Vá | Trạng Thái |
| :---: | :---: | :--- | :--- | :---: |
| **RF-2 (GDC Survival)** | `1d3f738` | Rủi ro nguồn dữ liệu sống còn cũ chưa đối soát trực tiếp từ API gốc NCI GDC. | Truy vấn sống **NCI GDC REST API Data Release 46.0** cho **`151/151` bệnh nhân** vào `src/config/tcga_bcss_151_gdc_clinical.json` (`0` sai khác; `23` BN Test có `2` biến cố tử vong; `04b_v2` GT `p = 0.1869`, AI `p = 0.1474`). | Đã xóa cờ đỏ |
| **RF-4 (GPU Pareto)** | `1d3f738`, `c0edb2f` | So sánh lệch độ chính xác FP32 vs BF16 trên Pareto và nhãn chú thích chồng lên điểm vẽ (`scatter marker`). | Đo đồng chuẩn `torch.bfloat16` cả hai chế độ `1x Raw` + `Deployed TTA` (`5 x 25` iters, kèm mốc FP32 ref) trên `NVIDIA L4` (`04b_v2`), bổ sung hộp né marker `pt_boxes` trong `clinical_bench_plots.py` (`c0edb2f`) -> `0` chồng nhãn trên cả 2 bảng `fig8`. | Đã xóa cờ đỏ |
| **P1B-ARM7-SHIFT** | `1d3f738` | `arm7` ConvNeXt-Large bị cắt biên đáy/phải ở tầng `stride=2` khi chạy đa tỷ lệ (`168px`, `280px`), gây dịch đỉnh tương quan `(1, -4)` px. | Thêm đệm đối xứng `_pad_to_multiple_of_32` (`src/stage2_v2/models.py:335-383`) và cắt ngược chính xác -> kéo đỉnh dịch chuyển về **`(1, -1)` px** (`zero_vs_peak_gap` giảm **`15x`** từ `1.123e-3` xuống `7.5e-5`), nâng TEST mIoU `71.80% -> 71.88%` (`02c_v2`, `02d_v2`). | Đã khắc phục |
| **P1B-NEC-FP & TTA** | `1d3f738`, `8dabf65` | Dương tính giả `NEC` ở các ca vắng hoại tử và lệch hằng số giao thức giữa `ensemble.py`, `visual_audit.py` và `api_server.py`. | Nâng cấp `ms_3_hvflip` (12 góc nhìn trên cả `4/4` nhánh), bổ sung cổng `cgm_nec_focused_1dof` (`t_nec = 0.40` chọn trên VAL) xóa **`6/8` BN vắng NEC về `0 px`** (`p = 0.0091`), và đồng bộ 100% hằng số giao thức (`8dabf65`). | Đã khắc phục |
| **P1B-WHOLE-ROI & LUAD** | `1d3f738` | Đường nối ghép ô cứng trên Whole-ROI (lật nhãn `11.4%-13.9%`) và chênh độ phóng đại `40x -> 10x` trên LUAD. | Bổ sung ghép chồng lấp **Hann-50%** giảm **`-98.35% / -97.99%`** đứt gãy đường nối (`TCGA-EW-A1P4 0.5421 -> 0.5557`, `TCGA-EW-A1P1 0.6706 -> 0.6779`) và thực nghiệm **LUAD `2x` zoom** đẩy Zero-Shot từ `36.09%` lên **`51.47%`** (`arm5`) / **`49.92%`** (`ens4`). | Đã khắc phục |
| **P1B-CAUSAL-ABL** | `1d3f738` | Thiếu chứng minh nhân quả tách biệt cho độ phân giải (`224` vs `336`), chưng cất (`stage1` vs `ens4_r2`), và `EMA vs Raw`. | Hoàn tất `3/3` job GPU `H1` (A100), `H2` (G4), `H3` (L4) + `phase1b_zero_gpu_analytics.json` (Holm `4/4` lớp `p_holm < 0.05`, phân tầng `k = 1-4`, sàng lọc Entropy `rho = -0.5599`), khóa mật mã **`228/228` tệp `SHA256SUMS.txt` (`BAD = 0`, `UNLISTED = 0`)**. | Đã khắc phục |

---

## 3. Nhật Ký Khắc Phục Lỗi Stage 1 (`47a1bab`), Stage 2 (`079fee0`) & Stage 3 Polish

Sau khi khóa bằng chứng nghiên cứu Phase 1B (`R1-R7`), đợt kiểm định kỹ nghệ phần mềm và sản phẩm (Stage 1 -> Stage 2 -> Stage 3 -> Chuẩn hóa Y Khoa) đã rà soát toàn bộ máy chủ API, giao diện HTML5 Canvas, notebook Colab, gói Drive và bộ kiểm toán ấn phẩm:

| Giai Đoạn / Mã | Commit | Phạm Vi Sửa Chữa & Gia Cố Kỹ Thuật | Kết Quả Kiểm Chứng |
| :--- | :---: | :--- | :--- |
| **Stage 1 (`R1-R7` Lock)** | `47a1bab` | Chốt báo cáo tổng kết thực nghiệm `docs/research/STAGE2_FINAL_EXPERIMENT_SUMMARY.md`, bổ sung fallback truy xuất commit git an toàn trong `src/stage2_v2/trainer.py`, và thêm `136` dòng unit test trong `tests/test_stage2_v2_gate7.py`. | `3/3` luồng xác nhận Research `R1-R7` khóa mật mã `228/228` SHA256. |
| **Stage 2 Product Hardening** | `079fee0` | Gia cố toàn diện máy chủ và giao diện: cô lập biến môi trường & trạng thái suy luận trong `src/visualization/api_server.py` (+`120` dòng test `tests/test_api_server.py`); bổ sung điều khiển Zoom/Pan mượt trên HTML5 Canvas `src/visualization/static/index.html`; củng cố `src/utils/deliverable_auditor.py` kiểm tra thẻ ảnh HTML (`img`) + mở rộng wildcard (+`49` dòng test); chuẩn hóa cấu trúc mô-đun `src/` và gói Colab; sửa notebook `02b` & `LowMag_HighMag_BonusDemo`. | Vòng 2 đạt đồng thuận kiểm toán chéo đa luồng độc lập **`30/30` tiêu chí PASS (`0` Red Flags)**. |
| **Stage 3 Figures & Deliverables** | Stage 3 | Nâng cấp `src/visualization/draw_gate7_monochrome_figures.py` (`372` dòng `<= 400`) đọc trực tiếp từ Phase 1B `02c_v2`, tái sinh toàn bộ `fig1-fig9` (300 DPI `#FFFFFF` không chồng nhãn), đồng bộ 100% `README.md`, báo cáo `.md/.docx` (`--syntax-highlighting=none`) và slide Marp `.md/.pptx`. | Kiểm tra Vision trực tiếp `fig1-fig9` & slide render `0` chồng chữ/tràn khung; `DeliverableAuditor` PASS 100%. |
| **Stage 4 Medical UI & Academic Hygiene** | 2026-11-26 | (1) Nâng cấp toàn diện `src/visualization/static/index.html` sang hệ màu quang học bệnh viện **Trắng Lâm Sàng + Xanh Y Khoa (`#FFFFFF`, `#F4FAFA`, `#0E7C86`, `#00957E`, `#2A9D8F`)**, Việt hóa chuẩn y khoa, tách 3 chế độ (**Chẩn đoán Lâm sàng không GT** / **Kiểm chứng Nghiên cứu 4 Khung GT** / **Ghép lát cắt WSI**), KPI đủ `100%` (`Trùng khớp + Biên 2px + Lỗi lõi`), Scale Bar `[2-200] um`, banner RUO thường trực; (2) tách sạch công cụ điều phối GPU cá nhân khỏi kho git vào `.gitignore`, xóa 100% thông tin tài khoản/hạ tầng riêng khỏi toàn bộ tài liệu/báo cáo/slide/P5 và khóa bằng test hồi quy + CI. | **`501` tests PASS**, `0` lỗi `ruff`, `0` rò rỉ nội bộ, Vision kiểm chứng trực tiếp bộ `P1-P5` mới. |

---

## 4. Bằng Chứng Kiểm Định Tự Động Hiện Hành (Empirical Verification)

1. **Bộ kiểm thử tự động phần mềm y khoa (`pytest tests/`):** Tổng cộng **`501` bài kiểm thử** thu thập trên `30` module trong `tests/` (bao gồm anti-leakage `151` bệnh nhân, không rò rỉ GT vào `BACK (4)`, API fail-closed, hệ màu Trắng - Xanh Y Khoa & 3 chế độ UI, thanh tỷ lệ `[2-200] um`, `DeliverableAuditor`, và cổng chống rò rỉ tài khoản/công cụ cá nhân), đạt **100% PASS (`0` thất bại)**.
2. **Kiểm tra chuẩn mã nguồn (`ruff`):** `ruff check src tests` trả mã thoát `0` (**`0` lỗi**).
3. **Kiểm định ấn phẩm học thuật độc lập (`DeliverableAuditor`):** Chạy trên toàn bộ `README.md`, `product/proposal/*.md`, `product/reports/*.md`, `product/reports/*.docx`, `product/slides/*.md`, và `docs/research/*.md` đạt **100% PASS (`0 CRITICAL, 0 WARNING`)**, xác nhận mực đen tuyệt đối (`#000000`), chỉ dùng dấu gạch ngang ASCII `-` (`0x2D`), đồ họa khoa học 300 DPI đơn sắc, và không rò rỉ siêu dữ liệu nội bộ hay thông tin giảng viên sai lệch.
4. **Kiểm toán mật mã dữ liệu thực nghiệm & Minh bạch 2 sự cố dữ liệu:** `checkpoints/vps_evidence/SHA256SUMS.txt` khóa cứng **`228/228`** tệp (`BAD = 0`, `UNLISTED = 0`). Hai sự cố mất tệp được công bố trung thực: (1) ngày **2026-11-05**, mất `checkpoints/medsam_wsss_adapter_{best,last}.pth` (`2x86 MB`) do test `D5/D16` chưa cô lập `tmp_path` (đã vá triệt để và không nằm trong nhánh Gate-7 Track A chính); và (2) ngày **2026-11-23**, lỗi thoát ngoặc kép PowerShell-to-cmd làm mất vĩnh viễn tệp ngoại khóa không theo dõi git `Lab1.ipynb` cùng đĩa ảo WSL2 `ext4.vhdx`; 100% kho sản phẩm chính và bằng chứng mật mã `228/228` của đồ án còn nguyên vẹn.
