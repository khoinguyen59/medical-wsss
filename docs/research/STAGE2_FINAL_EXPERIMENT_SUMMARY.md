# Báo Cáo Tổng Kết Thực Nghiệm Stage-2 BCSS-WSSS (SE374.R11 - UIT)

> **Mã học phần:** SE374.R11 - Phát triển phần mềm xử lý ảnh y khoa (UIT, ĐHQG-HCM, 2026)
> **Bộ dữ liệu chuẩn hóa:** BCSS-WSSS (`151` bệnh nhân TCGA-BRCA, `31,826` ảnh mẫu: `106` Train `23,422` / `22` Val `3,418` / `23` Test `4,986`; xác minh `patient_overlap_count = 0` theo `src/config/bcss_patient_splits.json`)
> **Commit mã nguồn chốt Phase 1A-1B:** `1d3f738` (bổ sung đệm đối xứng `_pad_to_multiple_of_32` xóa cắt biên stride-2 cho `arm7` ConvNeXt-Large, tích hợp sống còn thật NCI GDC Release 46.0 cho 151/151 bệnh nhân, Pareto BF16 đồng chuẩn, `ms_3_hvflip` 12 góc nhìn, Hann-50% Whole-ROI, LUAD `2x` zoom), `8dabf65e` (đồng bộ hằng số giao thức hậu Job E `02c_v2` vào `visual_audit.py` và `api_server.py`), và `c0edb2f` (chống chồng nhãn điểm vẽ trên biểu đồ Pareto 300 DPI)
> **Kho minh chứng mật mã:** `checkpoints/vps_evidence/` (100% chạy thực nghiệm thật trên GPU `NVIDIA A100-SXM4-40GB`, `NVIDIA RTX PRO 6000 Blackwell G4`, và `NVIDIA L4 24GB`, đối chiếu SHA256 bit-exact)
> **Trạng thái cập nhật Phase 1B:** Đã khóa đủ **`6/6` job GPU lớn hậu bản vá (`Job E 02c_v2`, `Job F 04b_v2`, `Job G 02d_v2`, `Job H1 arm5_224px`, `Job H2 arm5_s1 336px`, `Job H3 arm5_raw_vs_ema 336px`)**, xuất và kiểm định **`19/19`** hình kết quả Phase 1B (`1` biểu đồ CI của Job E + `4` Job F + `11` Job G + `3` đường cong hội tụ `H1/H2/H3`), và hoàn tất **`3/3`** nhóm phân tích thống kê không cần GPU trong `phase1b_zero_gpu_analytics.json` (`per_class_wilcoxon_holm`, `complexity_k_1_to_4`, `entropy_triage_spearman`; đánh đổi bộ lọc kính trắng `glass_guard_ablation` nằm trực tiếp trong `02d_v2`). Toàn bộ bằng chứng Giai đoạn 1 (`R1-R7`) đã đóng băng mật mã với `SHA256SUMS.txt` (`228/228` tệp, `BAD = 0`, `UNLISTED = 0`).

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Toàn bộ chiến dịch nghiên cứu và thực nghiệm Stage-1 -> Stage-2 trên bài toán phân vùng ngữ nghĩa giám sát yếu mô bệnh học ung thư vú (BCSS-WSSS, 4 lớp mô tiền cảnh `TUM`, `STR`, `LYM`, `NEC` + nền kính `BACK`) đã hoàn tất qua 7 cổng kiểm định (Gates 1-7) và đợt tái kiểm định hậu bản vá **Phase 1B** với nguyên tắc **chống rò rỉ bệnh nhân tuyệt đối** (`0` bệnh nhân trùng lặp giữa Train/Val/Test), **0% chồng lấp tiền huấn luyện TCGA-BRCA** trên toàn bộ backbone Stage-2 (Track A sạch `track_a_clean`: chỉ dùng ImageNet-1K và LVD-142M), và **0% dữ liệu lâm sàng giả lập** (`151/151` bệnh nhân đối soát trực tiếp từ NCI GDC API Release 46.0).

Các cột mốc định lượng đã xác minh bằng log thực nghiệm thật và chữ ký SHA256:

1. **Stage-1 Đột phá (`se374_arm_b_full`, Ep 5 EMA + Dual-Expert Classifier):**
   - Trên giao thức đối chuẩn Gate-4 `ms_tta_sg_dap` (`se374_gate4_bench`), `arm_b_full` nâng chất lượng mặt nạ giả trên toàn bộ tập VAL (`3,418` patches) từ **`72.03%`** (Author official) lên **`73.18%` mIoU (`+1.15%`)**, và trên đường ống memmap mềm có cổng `y_gt` ở `T = 0.25` (`se374_g7_diag_r2_val`, `q_s1_sgdap`) đạt **`74.13%` mIoU (`85.07%` mDice)**.
2. **Gate-6 Round-1 4-Arm Cross-Family Ensemble (`se374_s2v2_ensemble`, 4 họ kiến trúc):**
   - Đạt **`72.47%` TEST mIoU (`83.84%` mDice, `95% CI [66.96%, 76.43%]`)** trên 23 bệnh nhân TEST độc lập (`4,986` patches), vượt Author Official Pure (`69.47%` L4) **`+2.99%`** (`perm_p_holm = 0.0004`, `wilcoxon_p_holm = 0.000105`, `19W/0T/4L`) và vượt Step-0 Upgraded Author (`70.42%` L4) **`+2.04%`**.
3. **Gate-7 Step-0 VAL Ceiling Diagnostic (`se374_g7_diag_r2_val`, 0% TEST):**
   - Chứng minh trên `3,418` patches VAL (`2,153` đa lớp) rằng mặt nạ mềm Round-2 `p_ens4_uncal_ygated` (`blend = 1.0`, y-gated + fallback `p_sum < 0.01` trên chỉ `0.0054%` pixel) đạt **`76.03%` VAL mIoU (`+1.90%` so với Stage-1 `74.13%`)**, **`65.89%` multi-class mIoU (`+2.57%`)**, giảm ECE **35%** (`0.0489` vs `0.0751`).
4. **Đột phá đệm đối xứng xóa cắt biên stride-2 cho `arm7` ConvNeXt-Large & Gate-7 Track A Hậu Bản Vá (`Job E: se374_g7_phase1b_02c_v2`, `NVIDIA A100`, `EXIT_CODE = 0`):**
   - Bổ sung đệm đối xứng `_pad_to_multiple_of_32` (`src/stage2_v2/models.py:335-383`) đã loại bỏ hoàn toàn hiện tượng cắt biên đáy/phải ở các tầng stride-2 (`8 px` tại `0.75x = 168px`, `24 px` tại `1.25x = 280px`), kéo đỉnh dịch chuyển không gian quan sát của **`arm7_convnext_large_r2`** từ **`(1, -4)` px (`zero_vs_peak_gap = 1.123e-3`)** ở bản trước vá `se374_g7_visual_audit` về **`(1, -1)` px (`zero_vs_peak_gap = 7.5e-5`, giảm `15x`)** trong `02d_v2`, đồng thời nâng Held-Out TEST mIoU của `arm7` từ `71.80%` lên **`71.88%` (`83.45%` mDice, CGM `71.90%`; `TUM 80.46% -> 80.83%`, `LYM = 62.13%` cao nhất trong cả 4 mô hình đơn, vượt cả DINOv2-Large `61.13%`)**.
   - Cả `4/4` nhánh (`arm5`, `arm6`, `arm7`, `arm8`) tự động chọn **`ms_3_hvflip` (12 góc nhìn)** trên VAL không rò rỉ.
   - **Primary 4-Arm Track A Ensemble (`uniform [0.25]*4`, `cal_logprob_eps1e3`, `no_cgm`):** Đạt **`72.79%` TEST mIoU (`0.727865`, `95% CI [67.17%, 76.83%]`)**, **`84.04%` mDice** (`TUM 82.12 / STR 75.85 / LYM 62.30 / NEC 70.89`); với `cgm_val_calibrated` đạt **`72.80%` (`84.06%` mDice)**; với **`cgm_nec_focused_1dof` (`t_nec = 0.40`)** đạt **`72.80%` (`NEC = 70.95%`)** và xóa sạch dương tính giả `NEC` về `0 px` trên **`6/8` bệnh nhân TEST vắng NEC** (so với `3/8` ở `no_cgm` và `4/8` ở `cgm_val_calibrated`; Wilcoxon vs `no_cgm` **`p = 0.0091`**, `20W/3L`).
   - **Thứ cấp / Thăm dò (Simplex-CV 4-Arm `[0.61, 0.06, 0.05, 0.28]`):** Đạt **`72.85%`** (`no_cgm`) / **`72.88%`** (`cgm_val_calibrated`) / **`72.90%` (`84.11%` mDice, `NEC = 71.01%`)** ở `cgm_nec_focused_1dof`.
   - **Độ ổn định 3 hạt giống (`seed 42 / 123 / 456`) của `arm5` bên trong tổ hợp 4-Arm:** Giảm độ lệch chuẩn **`2.6x`** từ `0.7253 +/- 0.0013` (arm5 đơn) xuống **`0.7274 +/- 0.0005`** (`primary_4arm_no_cgm`) và **`0.7276 +/- 0.0005`** (`primary_4arm_cgm_nec_focused_1dof`).
   - **Kiểm định ý nghĩa toàn cục & từng lớp (`23` BN):** Vượt Step-0 Author (`70.43%`) **`+2.35%`** (`perm_p_holm = 0.0005`, `17W/6L`); vượt Pure Author `1x` (`69.48%`) **`+3.31%`** (`perm_p_holm = 0.0004`, `wilcoxon_p_holm = 0.000511`, `21W/2L`); vượt `arm5` đơn cùng chế độ `no_cgm` **`+0.24%`** (`wilcoxon_p = 0.0327`, `17W/6L`); và đạt **Wilcoxon hiệu chỉnh Holm `p_holm < 0.05` trên cả `4/4` lớp mô** (`TUM 0.000426`, `STR 0.00498`, `LYM 0.0306`, `NEC 0.0306`, trong khi `arm5` đơn không đạt ý nghĩa ở `LYM` với `p_holm = 0.0532`).
5. **Bóc Tách Độ Phân Giải `224px` vs `336px` & `EMA-Only` vs `Best-Raw` (`Job H1: se374_g7_phase1b_h1_arm5_224px`, `NVIDIA A100`, `EXIT_CODE = 0`):**
   - Trong cùng một phiên chạy (`224px`, `seed = 42`, `ens4_r2`), `Best EMA` (Ep 7, VAL `70.65%`) vượt `Best Raw` (Ep 8, VAL `70.38%`) trên mọi chế độ TEST: **Raw 1x `+0.56%`** (`71.99%` vs `71.43%`), **MS-TTA `+0.47%`** (`72.40%` vs `71.93%`), **MS-TTA+CGM cố định `+0.43%`** (`72.53%` vs `72.10%`), và **MS-TTA+CGM hiệu chuẩn `+0.24%`** (`72.56%` vs `72.32%`).
   - So sánh đồng chuẩn `02b` (`test_eval_calibrated.ms_tta_cgm`) giữa `Job H1 224px` (`Scale = 1.0`, TEST `raw 71.99%`, `ms_tta_cgm 72.56%`, `NEC 69.78%`) với `336px` (`Scale = 1.5`, `se374_g7_arm5_dinov2l`: `raw 72.24%`, `ms_tta_cgm 72.74%`, **`NEC 70.78%`, `+1.00%`**; và `Job H3 336px`: `raw 72.35%`, `ms_tta_cgm 72.88%`, **`NEC 71.15%`, `+1.37%`**; đối chiếu `02c_v2` `ms_3_hvflip` `arm5` CGM: `72.63%`, `NEC 70.56%`) xác nhận lợi ích rõ nét nhất của độ phân giải nội bộ `336px` nằm ở lớp đuôi dài `NEC` (`+1.00%` đến `+1.37%` trên cùng giao thức `02b`).
   - **Chứng minh nhân quả Chưng cất Round-2 `ens4_r2` vs `stage1` & `EMA vs Raw` ở `336px` (`Job H2: se374_g7_phase1b_h2_arm5_s1` trên `G4` & `Job H3: se374_g7_phase1b_h3_arm5_raw_vs_ema` trên `L4`, `seed = 42`, `EXIT_CODE = 0`):** Trên cùng cấu hình `arm5_dinov2_large_r2` `336px` (`SaveBestRaw = 1`), thay `stage1` (`Job H2`) bằng `ens4_r2` (`Job H3`) nâng Held-Out TEST ở mọi chế độ: **Raw 1x `+0.94%`** (`72.35%` vs `71.41%`), **MS-TTA `+0.84%`** (`72.60%` vs `71.76%`), **MS-TTA+CGM cố định `+0.85%`** (`72.74%` vs `71.89%`), và **MS-TTA+CGM hiệu chuẩn `+0.65%`** (`72.88%` vs `72.23%`). Trong chính phiên `Job H3` (`336px`, `ens4_r2`), `Best EMA` (Ep 8, VAL `70.94%`) tiếp tục vượt `Best Raw` (Ep 9, VAL `70.92%`) ở Raw 1x **`+0.19%`** (`72.35%` vs `72.16%`, `NEC +1.01%`), MS-TTA **`+0.09%`** (`72.60%` vs `72.51%`, `NEC +0.69%`), và MS-TTA+CGM cố định `0.10` **`+0.10%`** (`72.74%` vs `72.64%`), đồng thời ngang bằng ở MS-TTA+CGM hiệu chuẩn (`72.88%` vs `72.88%` nhưng `NEC` cao hơn `+0.46%`: `71.15%` vs `70.69%`).
6. **Kiểm Chứng Trực Quan 4,986 Patch, Hann-50% Whole-ROI & Sàng Lọc Entropy (`Job G: se374_g7_phase1b_02d_v2`, `NVIDIA G4`, `35m 20s`, `EXIT_CODE = 0`):**
   - **`6/6` Assertion PASS** (đối soát `pooled_miou` khớp Job E tới `<= 2.05e-4`, `0%` rò rỉ GT vào `BACK = 4`, đã soi Vision đủ **`11/11`** ảnh `01-11`).
   - **`Boundary F1@2px = 53.30%`** vs Author `50.67%` (Wilcoxon **`p = 0.000298`**, `20W/3L`; vượt `arm5` `52.72%` với **`p = 0.00345`**, `19W/4L`).
   - **Tâm khối u `TUM` trên tập giao chung không thiên vị (`n = 2,847` patches, `23` BN):** Giảm độ lệch trung vị từ `8.93 px` (Author, recall `93.68%`) xuống **`7.60 px`** (`4-Arm`, recall `95.80%-96.19%`, `p = 0.0327`, `19/23` thắng) và **`7.13 px`** (`Arm 5`, recall `95.73%`, `p = 0.0123`, `16/23` thắng).
   - **Ghép Whole-ROI chồng lấp Hann-50%:** Giảm tỷ lệ đứt gãy đường nối tile **`98.35%`** trên `TCGA-EW-A1P4` (`0.5421 -> 0.5557`, `+1.35%`) và **`97.99%`** trên `TCGA-EW-A1P1` (`0.6706 -> 0.6779`, `+0.73%`), tiệm cận mức tự nhiên của GT (`0.11%-0.26%`).
   - **Sàng lọc QA bằng Entropy (`n = 4,986`):** `Spearman rho = -0.5599` (`Confidence rho = +0.5708`, `p < 1e-300`); nhóm `20%` entropy thấp nhất đạt trung bình **`91.15%`** patch mIoU so với **`53.59%`** ở nhóm `20%` entropy cao nhất (`+37.56%` tách biệt).
7. **Chỉ Số Lâm Sàng, Sống Còn Thật NCI GDC Release 46.0, Hiệu Chuẩn Xác Suất, Pareto BF16 Đồng Chuẩn & LUAD `2x` Zoom (`Job F: se374_g7_phase1b_04b_v2`, `NVIDIA L4`, `65m 21s`, `EXIT_CODE = 0`):**
   - **Hiệu chuẩn điểm ảnh (`4,986` patches):** `arm5_cal` hạ **ECE xuống `0.00408`** (`-92.1%` so với `0.05142`), `ens4_cal_logprob_eps1e3` đạt **ECE = `0.00586`**, Brier `0.1922`, NLL `0.3405` (so với `author_step0` ECE `0.01812`, NLL `0.4206`).
   - **Chỉ số lâm sàng (GT-Free `Cand1` & `ROI-Bounded`, `23` BN) & Sống còn thật NCI GDC Release 46.0:** `ens4_cgm` đạt **đồng thuận phân tầng nguy cơ Mesker 2007 / UNITED 2018 = `91.30%` (`21/23`, Cohen's `kappa = 0.8271` vs Author `86.96%`, `kappa = 0.7435`)**, VTP `r = 0.9186` (`CCC = 0.9062`, ROI `r = 0.9297`), TSR `r = 0.9121` (`CCC = 0.9074`, ROI `r = 0.9234`), **VTP_RCB `r = 0.9543`** (cao nhất mọi mô hình); `arm5_cgm` đạt VTP `r = 0.9362` (`CCC = 0.9182`), TSR `r = 0.9260` (`CCC = 0.9169`). Sống còn thật GDC (`n = 23`, `2` biến cố tử vong): GT `p = 0.1869` (`chi2 = 1.74`), `ens4_cgm` `p = 0.1474` (`chi2 = 2.10`) - báo cáo trung thực cỡ mẫu nhỏ chưa đủ công suất cho `p < 0.05`.
   - **Pareto GPU đồng chuẩn `torch.bfloat16` trên `NVIDIA L4` (xóa nhiễu FP32/BF16):** `arm7` 1x Raw (`4.94 ms`, **`202.51 FPS`**, `71.61%` mIoU) **trội Pareto tuyệt đối** so với Author 1x Raw (`5.67 ms`, `176.63 FPS`, `69.34%`); `primary_4arm` 1x Raw (`36.75 ms`, **`27.22 FPS`**, **`72.68%` mIoU**) ngang tốc độ Author Deployed TTA (`27.60 FPS`, `70.42%`) nhưng cao hơn **`+2.26%` mIoU**.
   - **Chéo cơ quan Zero-Shot `BCSS (40x Vú) -> LUAD-HistoSeg (10x Phổi)` (`n = 307`, `20` ca):** Ở `10x` gốc: Author sập ở `15.05%` (`LYM = 0.00%`), `ens4_cgm = 25.56%`, **`arm5_cgm = 36.09%`** (`p = 3.81e-6`, `19W/0T/1L`); khi áp dụng **phóng đại `2x` (`10x -> 20x`, không huấn luyện lại)**: **`arm5_no_cgm_2x_zoom` vọt lên `51.47%` mIoU (`67.90%` mDice)** và **`ens4_uncal_linear_2x_zoom` vọt lên `49.92%` (`66.56%` mDice, `+25.01%`)**!

---

## 2. Giao Thức Chống Rò Rỉ & Nguồn Gốc Dữ Liệu (Track A & NCI GDC Provenance)

- **Phân chia cấp bệnh nhân nghiêm ngặt (`INVARIANT-08`):** Khóa cứng tại `src/config/bcss_patient_splits.json` (`151` bệnh nhân TCGA-BRCA hoàn toàn tách biệt: `106` Train `23,422` / `22` Val `3,418` / `23` Test `4,986`, `patient_overlap_count = 0`, `non_finite_tensors = 0`).
- **Tính thuần khiết Track A (`track_a_clean`, 0% chồng lấp tiền huấn luyện TCGA-BRCA):** Toàn bộ `arm1-arm4` (Round-1) và `arm5-arm8` (Round-2) chỉ dùng trọng số ImageNet-1K (`resnet101`, `mit-b3`, `mit-b5`, `convnext_base`, `convnext_large`) hoặc LVD-142M (`dinov2-base`, `dinov2-large`). Mọi foundation model tiền huấn luyện trên TCGA (như `phikon-v2`) bị chặn tuyệt đối bởi `validate_track_a_arm_spec()`. Stage-1 ResNet38d khởi tạo từ `stage1_checkpoint_trained_on_luad.pth` (`0%` bệnh nhân TCGA-BRCA).
- **Nguồn gốc dữ liệu sống còn thật (`RF-2` đã xóa triệt để):** `src/config/tcga_bcss_151_gdc_clinical.json` được truy vấn trực tiếp và kiểm chứng sống (`0` sai khác trên `151/151` bệnh nhân) từ **NCI Genomic Data Commons (GDC) REST API - Data Release 46.0** (`tag 8.5.0`, commit `8f7c2a51ab...`, queried `2026-11-24T04:44:41Z`; tổng số ca tử vong thật: `20` Train / `1` Val / `2` Test).

---

## 3. Diễn Tiến Thực Nghiệm & Bóc Tách Thành Phần (Phase 1B Hậu Bản Vá)

### 3.1. Gate-7 Step-0: Trần Pseudo-Mask Round-2 Trên Tập VAL (`3,418` Patches, 0% TEST)

| Ứng viên Pseudo-Mask trên VAL (`3,418` patches) | Overall mIoU | mDice | TUM | STR | LYM | NEC | Multi-class mIoU | ECE (thấp = tốt) | Entropy (Multi-class) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `q_s1_sgdap` (Stage-1, `blend = 0.0`) | 74.13% | 85.07% | 80.14% | 74.72% | 67.86% | 73.81% | 63.32% | 0.0751 | 0.2244 |
| `q_r2` (`blend = 0.50`: 50% `p_ens4` + 50% `q_s1`) | 75.39% (+1.26%) | 85.89% | 81.27% | 75.92% | **68.82%** | 75.54% | 65.00% | 0.0527 | 0.2847 |
| **`p_ens4_uncal_ygated` (`blend = 1.00`, y-gated + fallback)** | **76.03% (+1.90%)** | **86.30%** | **82.19%** | **76.69%** | 68.71% | **76.54%** | **65.89%** | **0.0489** | **0.2883** |

### 3.2. Tiến Hóa Đơn Mô Hình: Round-1 (Gate-6) vs Round-2 Hậu Bản Vá `arm7` (Job E `02c_v2`, `ms_3_hvflip`, `A100`)

| Họ kiến trúc | Round-1 Arm (Gate-6) | R1 Matched TEST | Round-2 Track A Arm (Hậu Bản Vá, `02c_v2`) | R2 VAL (`ms_3_hvflip`) | R2 No-CGM TEST | R2 CGM TEST | R2 Per-Class IoU (No-CGM: TUM / STR / LYM / NEC) | R2 vs R1 |
|---|---|---:|---|---:|---:|---:|---|---:|
| Classic ASPP CNN | `arm1_deeplab_cleankd` (R101) | 71.11% | `arm8_deeplab_r101_r2` (R101) | 69.31% | 70.70% | **71.09%** | 80.55 / 74.35 / 61.92 / 65.99 (CGM NEC: `67.06%`) | -0.02% |
| Hierarchical ViT | `arm2_segformer_oeem` (MiT-B3) | 71.84% | `arm6_segformer_b5_r2` (MiT-B5) | 70.31% | **72.04%** | 72.10% | 81.37 / 75.30 / 61.53 / 69.97 | **+0.20%** |
| Modern ConvNet | `arm3_convnext_selftrain` (Base) | 70.73% | **`arm7_convnext_large_r2` (Large, đã vá `_pad_to_multiple_of_32`)** | 69.65% | **71.88%** | 71.90% | 80.83 / 74.89 / **62.13** / 69.67 | **+1.15%** |
| SSL Vision Transformer | `arm4_dinov2_adapter` (ViT-B/14) | 72.36% | **`arm5_dinov2_large_r2` (ViT-L/14, `336px`)** | **71.21%** | **72.55%** | **72.63%** | **82.59 / 76.03** / 61.13 / **70.45** (CGM: `82.66/76.10/61.23/70.56`) | **+0.27%** |

### 3.3. Kiểm Chứng Đa Hạt Giống (`INVARIANT-09`) Trên Đơn Mô Hình & Trong Tổ Hợp 4-Arm (`02c_v2`)

- **Round-1 (`arm4`, 3 seeds `42/123/456`, chọn lẫn Raw/EMA):** `71.70% +/- 0.69%` (s456 rơi `-1.1%` do chọn trọng số Raw Ep 5).
- **Gate-7 `arm5` độc lập (`EMA-only`, 3 seeds `42/123/456`, đánh giá trên `02b`):** `72.68% +/- 0.09%` (`72.74%` s42 / `72.58%` s123 / `72.73%` s456; `raw = 72.12% +/- 0.15%`, giảm độ lệch chuẩn `7.7x` so với `arm4`).
- **Độ ổn định 3 hạt giống `arm5` BÊN TRONG Tổ Hợp 4-Arm (`02c_v2`, `test_multi_seed_4arm_variance`, `ms_3_hvflip`):**

| Cấu hình (`n = 3` hạt giống `arm5`: `42, 123, 456`) | Mean TEST mIoU | Std (`ddof=1`) | Min - Max | TUM / STR / LYM / NEC (Mean) |
|---|---:|---:|---|---|
| `arm5_no_cgm` (đơn mô hình) | 72.44% | 0.13% | 72.29% - 72.55% | 82.58 / 75.92 / 60.92 / 70.35 |
| `arm5_cgm_val_calibrated` (đơn mô hình) | 72.53% | 0.13% | 72.39% - 72.63% | 82.64 / 75.98 / 61.02 / 70.46 |
| **`primary_4arm_no_cgm` (4-Arm Uniform)** | **72.74%** | **0.05%** (`2.6x` thấp hơn) | 72.68% - 72.79% | 82.07 / 75.81 / **62.28** / 70.80 |
| **`primary_4arm_cgm_val_calibrated`** | **72.75%** | **0.05%** | 72.70% - 72.80% | 82.07 / 75.82 / **62.29** / 70.83 |
| **`primary_4arm_cgm_nec_focused_1dof` (`t_nec=0.40`)** | **72.76%** | **0.05%** | **72.70% - 72.80%** | 82.07 / 75.82 / 62.28 / **70.88** |

### 3.4. Bóc Tách Độ Phân Giải, Chọn Trọng Số `EMA` vs `Raw` & Nguồn Pseudo-Mask (`Jobs H1, H2, H3`)

| Thực nghiệm (`arm5_dinov2_large_r2`, `seed = 42`) | Độ phân giải | Pseudo-Mask | Best VAL (EMA / Raw) | TEST `raw` (EMA / Raw) | TEST `ms_tta` (EMA / Raw) | TEST `ms_tta_cgm` Cal (EMA / Raw) | Trạng thái |
|---|---|---|---|---:|---:|---:|---|
| **Job H1 (`se374_g7_phase1b_h1_arm5_224px`, `A100`)** | `224x224` (`1.0x`) | `ens4_r2` | **70.65%** (Ep7) / 70.38% (Ep8) | **71.99%** / 71.43% (`+0.56%`) | **72.40%** / 71.93% (`+0.47%`) | **72.56%** / 72.32% (`+0.24%`; fixed `72.53%` vs `72.10%`) | **HOÀN TẤT (`EXIT_CODE=0`)** |
| **Job H2 (`se374_g7_phase1b_h2_arm5_s1`, `G4`)** | `336x336` (`1.5x`) | `stage1` | **70.60%** (Ep5) / 70.32% (Ep5) | **71.41%** / 70.66% (`+0.75%`) | **71.76%** / 71.05% (`+0.71%`) | **72.23%** / 71.68% (`+0.55%`; fixed `71.89%` vs `71.25%`) | **HOÀN TẤT (`EXIT_CODE=0`)** |
| **Job H3 (`se374_g7_phase1b_h3_arm5_raw_vs_ema`, `L4`)** | `336x336` (`1.5x`) | `ens4_r2` | **70.94%** (Ep8) / 70.92% (Ep9) | **72.35%** / 72.16% (`+0.19%`) | **72.60%** / 72.51% (`+0.09%`) | **72.88%** / 72.88% (`fixed`: **72.74%** vs 72.64%, `+0.10%`; `NEC 71.15%` vs `70.69%`) | **HOÀN TẤT (`EXIT_CODE=0`)** |
| Mốc chuẩn đã chốt `se374_g7_arm5_dinov2l` (`G4`) | `336x336` (`1.5x`) | `ens4_r2` | 70.95% (Ep10 EMA) | 72.24% | 72.53% | 72.74% (`02c_v2` `ms_3_hvflip`: `72.63%`) | Đã chốt (`33f4d623...`) |

---

## 4. Bảng Kết Quả Chung Kết Gate-7 Track A Hậu Bản Vá (`Job E: se374_g7_phase1b_02c_v2`, TEST `n = 4,986`)

### 4.1. Giao Thức Đóng Băng Chọn 0% Rò Rỉ Trên Tập VAL (`3,418` Patches, 22 Bệnh Nhân)

- **TTA chọn trên VAL:** Cả `4/4` nhánh chọn **`ms_3_hvflip`** (`12` góc nhìn: 3 tỷ lệ `0.75/1.0/1.25` x 4 phép lật H/V).
- **Hiệu chuẩn nhiệt độ & trộn log-prob (`cal_logprob_eps1e3`):** `full_val_betas = [0.6386 (arm5), 0.7029 (arm6), 0.6747 (arm7), 0.8782 (arm8)]`; 5-fold CV OOF `cal_logprob_eps1e3` (`0.711209` uniform / `0.711328` simplex) vượt `cal_linear` (`0.709838`) và `uncal_linear` (`0.709311`).
- **Trọng số & LOPOCV (`22`-fold Leave-One-Patient-Out):** `simplex_bagged_weights = [0.61, 0.06, 0.05, 0.28]` (`arm5-arm8`); LOPOCV `weights_mean = [0.5818, 0.0523, 0.0500, 0.3159]` (`oof_miou = 0.710641`). Vì chênh lệch 5-fold CV `simplex - uniform = +0.00012` dưới ngưỡng bảo thủ, giao thức khóa **`uniform [0.25, 0.25, 0.25, 0.25]`** và **`use_cgm = False`** làm Primary 4-Arm, đồng thời công bố đầy đủ `cgm_val_calibrated` (`fixed_0.10`), `cgm_nec_focused_1dof` (`t_nec = 0.40`), và `simplex_cv_4arm`.

### 4.2. Bảng Tổng Hợp Chính (Primary vs Anchors vs Secondary/Exploratory)

| Phương pháp / Giao thức (Hậu Bản Vá `02c_v2` trên `A100`) | Vai trò đánh giá | TEST mIoU | TEST mDice | 95% Patient Boot CI | TUM / STR / LYM / NEC IoU |
|---|---|---:|---:|---|---|
| Author Official Pure (`1x`, `cgm_author_mul0`) | Mốc tác giả gốc | 69.48% | 81.79% | `[63.63, 73.59]` | (Raw `1x` không CGM: `69.35%`) |
| Author Official Pure (`MS-TTA + CGM`) | Mốc tác giả tăng cường | 70.31% | 82.36% | `[64.57, 74.32]` | - |
| Author Step-0 Upgraded Anchor (`arm_b_full` + `MS-TTA+CGM`) | Mốc đối chứng chính (`m=5`) | 70.43% | 82.45% | `[64.72, 74.47]` | - |
| Gate-6 Round-1 Primary 4-Arm (`se374_s2v2_ensemble`) | Mốc Round-1 | 72.47% | 83.84% | `[66.96, 76.43]` | 81.55 / 75.51 / 62.34 / 70.47 |
| Gate-7 Track A Single Champion (`arm5`, `ms_3_hvflip` + CGM `0.10`) | Đơn mô hình tiền đăng ký | 72.63% (`no_cgm`: `72.55%`) | 83.90% | - | **82.66 / 76.10** / 61.23 / 70.56 |
| Gate-7 Track A VAL-Selected CNN+ViT Pair (`arm5+arm8`, `no_cgm`) | Cặp tiền đăng ký trên VAL | 72.69% (CGM: `72.72%`) | 83.97% | `[66.95, 76.71]` | 82.36 / 75.92 / 62.10 / 70.40 |
| **Gate-7 Track A Primary 4-Arm (`Uniform`, `cal_logprob`, `no_cgm`)** | **Mô hình tổ hợp chính** | **72.79%** (`0.727865`) | **84.04%** | **`[67.17, 76.83]`** | 82.12 / 75.85 / **62.30** / 70.89 |
| **Gate-7 Track A Primary 4-Arm + `cgm_val_calibrated` (`0.10`)** | **Tổ hợp chính + CGM 5-chuyên gia** | **72.80%** (`0.728022`) | **84.06%** | **`[67.19, 76.85]`** | 82.12 / 75.86 / **62.32** / 70.91 |
| **Gate-7 Track A Primary 4-Arm + `cgm_nec_focused_1dof` (`t_nec=0.40`)** | **Chế độ lâm sàng sạch FP NEC** | **72.80%** (`0.728045`) | **84.06%** | **`[67.19, 76.86]`** | 82.12 / 75.85 / 62.30 / **70.95** |
| Gate-7 Track A Simplex-CV 4-Arm (`[0.61,0.06,0.05,0.28]`, `no_cgm` / `cgm_cal` / `nec1dof`) | Thứ cấp / Thăm dò | 72.85% / 72.88% / **72.90%** | 84.07% / 84.09% / **84.11%** | `[67.28, 76.89]` | 82.55 / 76.13 / 61.93 / **71.01** |

- **Nhóm patch đa lớp phức tạp (`multiclass_k_ge_2`, `n = 2,984 / 4,986`):** Primary 4-Arm `cgm_val_calibrated` đạt **`65.33%` mIoU (`78.89%` mDice)** (`TUM 73.70 / STR 67.10 / LYM 58.44 / NEC 62.09`) vượt `arm5` đơn (`65.26%` / `78.81%`, `LYM 57.88`, `NEC 61.48`), thể hiện ưu thế tổ hợp đa họ ở `LYM` (`+0.56%`) và `NEC` (`+0.61%`).
- **Đa dạng & Giới hạn trên (`02c_v2` TEST):** Bất đồng theo cặp `3.61%` (`arm6-arm7`, `kappa = 0.9410`) đến `6.14%` (`arm5-arm8`, `kappa = 0.8999`); **Pixel Oracle Accuracy = `90.35%`**, **Patch Oracle Upper Bound = `75.69%` mIoU**.

### 4.3. Kiểm Định Ý Nghĩa Thống Kê Toàn Cục & Từng Lớp (Holm-Bonferroni, `23` Bệnh Nhân)

| So sánh (`02c_v2` & `phase1b_zero_gpu_analytics.json`) | Chênh lệch | `perm_p_holm` (pooled) | `wilcoxon_p` / `p_holm` (per-patient) | Cohen's `d_z` | Thắng/Hòa/Thua |
|---|---:|---:|---:|---:|---|
| **Primary 4-Arm (`no_cgm`) vs Step-0 Author (`70.43%`)** [`m=5`] | **+2.35%** | **0.0005** | `0.0650` / `0.2813` | +0.1619 | 17W / 0T / 6L |
| **Primary 4-Arm (`no_cgm`) vs Pure Author `1x` (`69.48%`)** [`m=4`] | **+3.31%** | **0.0004** | **`0.000511`** (Holm) | **+0.5817** | **21W / 0T / 2L** |
| **Primary 4-Arm (`no_cgm`) vs Pure Author `MS-TTA` (`70.31%`)** [`m=4`] | **+2.48%** | **0.0004** | **`0.0217`** (Holm) | **+0.3312** | **20W / 0T / 3L** |
| **Pure Ensemble: Primary 4-Arm (`no_cgm`) vs `arm5` (`no_cgm`)** | **+0.24%** | `p_raw = 0.6029` | **`0.0327`** (raw) | **+0.4008** | **17W / 0T / 6L** |
| **Primary 4-Arm `nec_1dof` vs `no_cgm`** | +0.02% | `p_raw = 0.2510` | **`0.0091`** (raw) | **+0.3603** | **20W / 0T / 3L** |
| **Primary 4-Arm `nec_1dof` vs Best Single (`arm5` CGM)** | +0.17% | `p_raw = 0.7170` | **`0.0327`** (raw) | **+0.5707** | **15W / 0T / 8L** |

**Kiểm định Wilcoxon từng lớp mô hiệu chỉnh Holm-Bonferroni (`m = 4`, `ens4_cgm` vs `author_step0` trên `per_patient_class_ious_all_models.json`):**

| Lớp mô học | Số BN hoạt động | Mean Delta IoU | Median Delta IoU | W / T / L | `p_raw` | **`p_holm` (`ens4_cgm` vs Author)** | Đối chiếu `arm5_cgm` vs Author (`p_holm`) |
|---|---:|---:|---:|---|---:|---:|---:|
| **`TUM` (Tumor Epithelium)** | 23 | **+2.42%** | +2.11% | 21 / 0 / 2 | 0.000107 | **0.000426** (`***`) | 0.005115 (`**`) |
| **`STR` (Stroma)** | 23 | **+1.58%** | +1.60% | 18 / 0 / 5 | 0.001660 | **0.004980** (`**`) | 0.005115 (`**`) |
| **`LYM` (Lymphocytic Infiltrate)** | 22 | **+1.96%** | +1.49% | 15 / 2 / 5 | 0.015312 | **0.030624** (`*`) | 0.053169 (`ns`) |
| **`NEC` (Necrosis)** | 20 | **+4.48%** | +0.10% | 10 / 6 / 4 | 0.024536 | **0.030624** (`*`) | 0.037076 (`*`) |

- **Ý nghĩa khoa học:** Tổ hợp 4 họ kiến trúc (`ens4_cgm`) là phương pháp duy nhất đạt ý nghĩa thống kê sau hiệu chỉnh Holm trên **cả `4/4` lớp mô** (`p_holm <= 0.0306`), khắc phục điểm yếu của đơn mô hình `arm5` ở lớp miễn dịch `LYM` (`p_holm = 0.0532 > 0.05`).

---

## 5. Khử Dương Tính Giả NEC 3 Chiều & Phân Tích Lỗi Lâm Sàng (`INVARIANT-10`)

- **Cơ chế khử dương tính giả `NEC` trên `8` bệnh nhân Held-Out Test không có mô hoại tử (`GT-NEC == 0`):**
  - `primary_4arm_no_cgm`: sạch `0 px` ở **`3/8`** ca (`D8-A27H`, `E2-A14R`, `EW-A1OV`), tổng FP = **`121,071 px`** (còn sót `E2-A158: 215 px`, `E2-A1LS: 375 px`, `EW-A6SB: 376 px`, `EW-A1P4: 27,771 px`, `E2-A573: 92,334 px`).
  - `primary_4arm_cgm_val_calibrated` (`t = 0.10`): xóa sạch thêm `E2-A158` -> **`4/8`** ca sạch, tổng FP = **`119,400 px`**.
  - **`primary_4arm_cgm_nec_focused_1dof` (`t_nec = 0.40`, chọn 0% rò rỉ trên VAL):** xóa sạch thêm `E2-A1LS` và `EW-A6SB` -> **`6/8` ca sạch tuyệt đối `0 px`**, tổng FP giảm xuống **`110,076 px`** (chỉ còn 2 ca khó thực sự là `E2-A573` và `EW-A1P4`), đồng thời nâng `NEC IoU` lên `70.95%` (Uniform) / `71.01%` (Simplex) và đạt Wilcoxon `p = 0.0091` (`20W/3L`) trước `no_cgm`.
- **Phân tầng độ phức tạp `k = 1-4` lớp đồng xuất hiện (`n = 4,986` patches, `phase1b_zero_gpu_analytics.json`):**
  - `k = 1` (`n = 2,002`): Author `0.9410` | Arm 5 `0.9449` | **4-Arm+CGM `0.9473`** (`+0.24%` vs Arm 5).
  - `k = 2` (`n = 2,543`): Author `0.5399` | Arm 5 `0.5685` | 4-Arm+CGM `0.5616` (lưu ý tính theo pooled pixel trên `k >= 2`, 4-Arm `0.6533` thắng Arm 5 `0.6526`).
  - `k = 3` (`n = 431`): Author `0.3540` | Arm 5 `0.3927` | 4-Arm+CGM `0.3879`.
  - `k = 4` (`n = 10`, đủ 4 lớp trong 1 patch): Author `0.2365` | Arm 5 `0.2503` | **4-Arm+CGM `0.2646`** (`+1.43%` vs Arm 5, `+2.81%` vs Author).
- **Sàng lọc QA lâm sàng bằng Entropy (`n = 4,986` patches):** `ens4_mean_entropy` tương quan nghịch chặt với độ chính xác phân vùng (**`Spearman rho = -0.5599`, `p < 1e-300`**; `ens4_mean_confidence` đạt **`rho = +0.5708`**). Nhóm `20%` patch có độ bất định thấp nhất đạt trung bình **`91.15%`** patch mIoU, trong khi nhóm `20%` bất định cao nhất chỉ đạt **`53.59%`** (top `10%` bất định cao nhất: **`51.46%`**), chứng minh giá trị thực tiễn của bản đồ Entropy trên Trạm Lâm Sàng để tự động khoanh vùng cần bác sĩ giải phẫu bệnh kiểm duyệt ưu tiên.

---

## 6. Kiểm Chứng Trực Quan (`02d_v2`), Lâm Sàng GDC, Pareto BF16 & LUAD `2x` Zoom (`04b_v2`)

### 6.1. Kiểm Chứng Trực Quan 4,986 Patch & Căn Chỉnh Không Gian (`Job G: se374_g7_phase1b_02d_v2`, `G4`)

- **6/6 Assertion PASS (`d933a89ebec0578c`):** GT PNG palette gốc (`4,986/4,986`), SHA256 khớp tuyệt đối, đối xứng `flip_rot90 >= 0.7931` (`pixel_agreement >= 0.8995`), dịch chuyển không gian **mọi mô hình đều nằm trong bán kính `<= 2 px` quanh tâm `(0, 0)`** (`arm7/arm5/ens4 = (1, -1)` px, `arm8 = (1, 0)` px, `arm6 = (2, -1)` px, `author_step0 = (1, 1)` px; toàn bộ có `zero_vs_peak_gap <= 1.39e-4`, riêng `arm7 = 7.5e-5` và `ens4_cgm = 7.7e-5`), `gt_parameter_free = True`, và đối soát `pooled_miou` chênh lệch `<= 2.05e-4` so với `02c_v2`.
- **Độ sắc nét biên (`Boundary F1@2px`) & Tâm khối u trên tập giao chung không thiên vị (`n = 2,847` common TUM patches, `23` BN):**

| Mô hình (`02d_v2`, `n = 4,986`) | Macro Boundary F1@2px (`95% CI`) | TUM / STR / LYM / NEC F1@2px | TUM Patch Recall (`n = 3,069`) | Matched Median / Mean TUM Offset (`n = 2,847`) | Wilcoxon vs Author (BF1 / Centroid) |
|---|---:|---|---:|---:|---|
| `author_step0` | 50.67% `[45.73, 54.21]` | 60.77 / 56.31 / 33.41 / 52.19 | 93.68% | 8.93 px / 17.55 px | Mốc đối chứng |
| `arm5_cgm` | 52.72% `[48.14, 56.18]` | **63.93** / 57.12 / 35.63 / 54.22 | 95.73% | **7.13 px / 15.52 px** | `p = 0.1695` / **`p = 0.0123`** (`16W/7L`) |
| `ens4_no_cgm` | 53.21% `[48.59, 56.64]` | 63.61 / 57.66 / 36.27 / 55.31 | **96.19%** | 7.60 px / 16.19 px | **`p = 0.0277`** / **`p = 0.0327`** |
| **`ens4_cgm`** | **53.30% `[48.67, 56.73]`** | 63.67 / **57.73 / 36.35 / 55.44** | 95.80% | 7.60 px / 16.19 px | **`p = 0.000298` (`20W/3L`)** / **`p = 0.0327` (`19W/4L`)** |

*(Lưu ý: `ens4_cgm` đồng thời vượt `arm5_cgm` về `Boundary F1@2px` ở cấp bệnh nhân với `p = 0.00345`, `19W/4L`.)*

- **Ghép Whole-ROI & Bóc Tách Chồng Lấp Hann-50% (`09`, `10`, `whole_roi_audit.json`):**
  - `TCGA-EW-A1P4` (`14x17 = 238` tiles, `3808x3136 px`): Legacy Downscale `0.2020` -> Native Tiled `0.5421` -> **`Hann-50% = 0.5557` (`+1.35%`)**; tỷ lệ lật nhãn tại đường nối giảm từ `13.91%` xuống **`0.23%` (`-98.35%`)**, sát mức GT `0.11%`.
  - `TCGA-EW-A1P1` (`16x18 = 288` tiles, `4032x3584 px`): Legacy Downscale `0.0749` -> Native Tiled `0.6706` -> **`Hann-50% = 0.6779` (`+0.73%`)**; tỷ lệ lật nhãn tại đường nối giảm từ `11.38%` xuống **`0.23%` (`-97.99%`)**, sát mức GT `0.26%`.
- **Ablation bộ lọc kính trắng không rò rỉ GT trên toàn bộ `4,986` patch (`glass_guard_ablation`):**
  - `none`: `72.80%` mIoU (`0%` ức chế kính, `0%` xói mòn).
  - **`pure_glass`**: **`72.80%`** mIoU (`-0.00%`, `1.04%` kính sáng, `0.005%` FG erosion, **`0.0%` NEC erosion**).
  - `cand1_bright_glass`: `72.22%` (`-0.58%`, `48.55%` kính sáng, `0.793%` FG erosion, `2.808%` NEC erosion).
  - `otsu`: `70.82%` (`-1.98%`, `91.12%` kính sáng, `3.336%` FG erosion, `8.258%` NEC erosion).

### 6.2. Chỉ Số Lâm Sàng, Hiệu Chuẩn Xác Suất & Sống Còn Thật NCI GDC (`Job F: se374_g7_phase1b_04b_v2`, `L4`)

| Mô hình (`n = 23` BN, GT-Free `Cand1` / `ROI-Bounded`) | VTP `r` / `CCC` (MAE) | TSR `r` / `CCC` (MAE) | VTP_RCB `r` | Mesker/UNITED 50% TSR Concordance (`kappa`) | Live GDC KM Log-Rank `p` |
|---|---:|---:|---:|---:|---:|
| `author_step0` | 0.8888 / 0.8782 (5.57%) | 0.8994 / 0.8889 (6.73%) | 0.9358 | 86.96% (`20/23`, `0.7435`) | 0.1116 |
| **`arm5_cgm`** | **0.9362 / 0.9182 (4.99%)** (ROI `0.9401`) | **0.9260 / 0.9169 (6.31%)** (ROI `0.9371`) | 0.9329 | 86.96% (`20/23`, `0.7435`) | 0.1116 |
| **`ens4_cgm` (`ens4_no_cgm` tương đương)** | 0.9186 / 0.9062 (5.08%) (ROI `0.9297`) | 0.9121 / 0.9074 (**6.30%**, ROI `0.9234`) | **0.9543** | **91.30% (`21/23`, `0.8271`)** | 0.1474 (GT: `0.1869`, `2` biến cố) |

- **Hiệu chuẩn xác suất điểm ảnh (`4,986` patches, `bcss_pixel_calibration`):**
  - `author_raw`: ECE `0.02816` | Brier `0.2256` | NLL `0.4195`.
  - `author_step0` (có CGM): ECE `0.01812` | Brier `0.2140` | NLL `0.4206`.
  - `arm5_uncal` -> **`arm5_cal`**: ECE `0.05142` -> **`0.00408` (`-92.1%`)** | Brier `0.1986` -> **`0.1914`** | NLL `0.3788` -> **`0.3392`**.
  - `ens4_uncal_linear` -> **`ens4_cal_logprob_eps1e3`**: ECE `0.02571` -> **`0.00586` (`-77.2%`)** | Brier `0.1948` -> **`0.1922`** | NLL `0.3530` -> **`0.3405`**.

### 6.3. Hiệu Năng GPU Đồng Chuẩn `torch.bfloat16` Trên `NVIDIA L4` (`04b_v2`, `5 x 25` iters, `batch_size = 4`)

| Mô hình (`NVIDIA L4`, `torch.bfloat16` đồng chuẩn) | 1x Raw Latency (ms) | 1x Raw FPS | 1x Raw mIoU | Deployed TTA Latency (ms) | Deployed FPS (`views`) | Deployed mIoU |
|---|---:|---:|---:|---:|---:|---:|
| `author_stage2` (DeepLabV3+ R101) | 5.67 ms | 176.6 FPS | 69.34% | 36.24 ms | 27.6 FPS (`6x` + S1) | 70.42% |
| `arm8_deeplab_r101_r2` | 5.92 ms | 169.2 FPS | 69.58% | 68.52 ms | 14.6 FPS (`12x`) | 70.70% |
| **`arm7_convnext_large_r2` (Pareto-Dominant 1x)** | **4.94 ms** | **202.5 FPS** | **71.61%** | 66.90 ms | 15.0 FPS (`12x`) | 71.88% |
| `arm6_segformer_b5_r2` | 19.03 ms | 52.6 FPS | 71.95% | 228.91 ms | 4.4 FPS (`12x`) | 72.04% |
| **`arm5_dinov2_large_r2` (`336px`)** | **13.66 ms** | **73.2 FPS** | **72.24%** | 173.89 ms | 5.8 FPS (`12x`) | 72.65% |
| **`primary_4arm_ensemble` (`no-CGM`)** | **36.75 ms** | **27.2 FPS** | **72.68%** | 534.59 ms | 1.9 FPS (`48x`) | **72.79%** |

*(Ghi chú giao thức `04b_v2`: Mốc `arm5` `12x` Deployed `72.65%` dùng `s2_self` CGM với `0` lượt suy luận Stage-1 bổ sung, khớp độ trễ `173.89 ms` - so với `72.55%` `no_cgm` và `72.63%` `s1_s2_50_50` CGM trong `02c_v2`. Bảng đo đồng chuẩn `torch.bfloat16` cho mọi mô hình gồm `Author 1x Raw BF16 = 5.67 ms / 176.63 FPS` (`69.34%`); đối chiếu thêm mốc FP32 thuần `author_stage2_fp32 = 3.80 ms / 263.27 FPS` tương ứng chế độ `1x` raw **`69.34%`** trên L4 (`69.35%` trên A100), trong khi mốc `69.47%-69.48%` cần bổ sung Stage-1 CGM (`cgm_author_mul0`), và kết quả đo Wave-2B trên GPU `NVIDIA RTX PRO 6000 Blackwell G4`: `arm5` 1x Raw `334.7 FPS`, `primary_4arm_ensemble` 1x Raw `110.3 FPS`.)*

### 6.4. Chéo Cơ Quan Zero-Shot `BCSS (40x Vú) -> LUAD-HistoSeg (10x Phổi)` & Phóng Đại `2x` (`04b_v2`, `n = 307`, `20` Ca)

| Mô hình Zero-Shot (`BCSS -> LUAD`) | Độ phóng đại đầu vào | Pooled 4-Cls mIoU | Pooled mDice | TUM / STR / LYM / NEC IoU | Paired vs Author (`n = 20` ca) |
|---|---|---:|---:|---|---|
| `author_step0` (ResNet-101 CNN) | Gốc `10x` | 15.05% | 24.78% | 26.79 / 20.72 / **0.00** / 12.69 | Mốc đối chứng |
| `ens4_cgm` (`ens4_no_cgm`) | Gốc `10x` | 25.56% | 40.09% | 35.00 / 31.42 / 16.03 / 19.79 | `+10.51%`, `p_wilc = 0.0033` (`15W/1T/4L`) |
| `arm5_cgm` (`arm5_no_cgm = 35.47%`) | Gốc `10x` | **36.09%** (`2.40x`) | **52.52%** | 28.45 / 49.99 / 34.53 / 31.40 | **`+21.04%`, `p_wilc = 3.81e-6` (`19W/0T/1L`)** |
| **`ens4_uncal_linear_2x_zoom`** | **Phóng `2x` (`10x -> 20x`, 0 train)** | **49.92%** (`+25.01%`) | **66.56%** | 48.70 / 52.55 / **46.39** / 52.03 | Giải quyết khoảng cách độ phóng đại `4x` |
| **`arm5_no_cgm_2x_zoom`** | **Phóng `2x` (`10x -> 20x`, 0 train)** | **51.47%** (`+16.00%`) | **67.90%** | **54.87 / 52.72** / 46.15 / **52.12** | **Vượt `3.42x` Author `10x` hoàn toàn zero-shot!** |

---

## 7. Đối Chiếu Trung Thực Với SOTA 2025-2026 & Giới Hạn Mở

| Công trình / Phương pháp | Nguồn công bố | Giao thức / Backbone | BCSS-WSSS mIoU | Nhận xét so sánh công bằng |
|---|---|---|---:|---|
| WSSS-Tissue (Baseline gốc) | Han et al., MedIA 2022 (`2110.08048`) | ResNet38d + DeepLabV3+ R101 | 69.48% (tái tạo sạch) | Mốc đối chứng chuẩn 4-class pooled |
| QFKF | BMVC 2025 (`docs/papers/QFKF_BMVC2025.md`) | 4-class pooled BCSS-WSSS | 70.92% | Gate-7 Track A vượt **`+1.87%`** Primary / **`+1.88%`** CGM |
| ConStruct | MIDL 2026 (`arXiv:2512.10316`) | 4-class pooled BCSS-WSSS | 70.96% | Gate-7 Track A vượt **`+1.83%`** Primary / **`+1.84%`** CGM |
| HisynSeg | IEEE TMI 2025 (`arXiv:2412.20924`) | 4-class pooled BCSS-WSSS (`5` seeds Table II / CSS re-eval) | 70.97% +/- 0.29% (`70.73%` CSS re-eval) | Gate-7 Track A vượt **`+1.82%`** Primary / **`+1.83%`** CGM (và `+2.06%`/`+2.07%` so với CSS re-eval) |
| PLIMSeg | BSPC Vol. 112, 2026 | 4-class pooled BCSS-WSSS | 71.07% | Gate-7 Track A vượt **`+1.72%`** Primary / **`+1.73%`** CGM |
| DualProtoSeg | arXiv:2512.10314 (2025) | 4-class pooled BCSS-WSSS | 71.35% | Gate-7 Track A vượt **`+1.44%`** Primary / **`+1.45%`** CGM |
| CSS | Bioinformatics 2026 (`PMC13614711`) | 4-class pooled BCSS-WSSS | 71.72% | Gate-7 Track A vượt **`+1.07%`** Primary / **`+1.08%`** CGM / **`+1.18%`** Simplex |
| C2RM-Seg | MICCAI 2026 (`arXiv:2606.25508`) | 4-class pooled BCSS-WSSS (`5` runs, ResNeSt-200e + DINOv3-Base) | 72.17% +/- 0.83% | Gate-7 Track A vượt **`+0.62%`** Primary / **`+0.63%`** CGM / **`+0.73%`** Simplex với độ lệch chuẩn thấp hơn **`16.6x`** (`0.05%` vs `0.83%`) |
| **Đề xuất Gate-7 Track A Hậu Bản Vá (`02c_v2`)** | **SE374.R11 UIT (2026)** | **0% TCGA-BRCA overlap, 23 BN tách biệt, 3-seed `0.7274 +/- 0.0005`** | **72.79% / 72.80%** (`72.90%` secondary Simplex) | **Vượt mọi SOTA 4-class pooled chuẩn 2025-2026** |
| SIA-WSSS | MICCAI 2025 (`docs/papers/SIA_WSSS_MICCAI2025.md`, p. 6 Table 1) | 5-run avg (gồm lớp `OTR`, khác định nghĩa Dice) | 73.40% | Gate-7 Track A thấp hơn **`-0.61%`** Primary / **`-0.60%`** CGM / **`-0.50%`** Simplex `nec1dof` |

- **Cảnh báo trung thực về định nghĩa mDice với `SIA-WSSS` (MICCAI 2025):** Không so sánh trực tiếp `84.04%-84.11%` mDice của hệ thống này với `79.9%` mDice của SIA-WSSS vì SIA-WSSS báo cáo `WSSS-Tissue` ở `0.697` mIoU / `0.767` mDice (chênh `~5.1` điểm phần trăm định nghĩa mDice). So sánh công bằng duy nhất giữa hai công trình là trên **mIoU**.

---

## 8. Danh Mục Minh Chứng Mật Mã Phase 1B (Đã Xác Minh SHA256 Trên Đĩa Cục Bộ & VPS)

| Phiên chạy (`checkpoints/vps_evidence/`) | GPU Thực Thi | Tệp minh chứng chính | SHA256 (16 ký tự đầu) | Trạng thái |
|---|---|---|---|---|
| Stage-1 Champion (`se374_arm_b_full`) | G4 | `stage1_retrained_best.pth` / `stage1_best_cls.pth` | `a182d8890c09d7b8` / `e1a88293aff0a0e3` | Khóa cứng |
| Gate-7 Track A Arms (`arm5/6/7/8`) | G4 / L4 | `stage2_retrained_best.pth` (`arm5/6/7/8`) | `33f4d623` / `3055643d` / `c64b97c7` / `9fd6e121` | Khóa cứng |
| **Job E: Post-Fix 4-Arm & 3-Seed (`se374_g7_phase1b_02c_v2`)** | **NVIDIA A100** | `STAGE2_TRAINING_METRICS.json` / `SE374_Stage2_Training_Results.zip` | `5188cd5527353b08` / `3385df05acf95188` | **HOÀN TẤT (`6` tệp lõi / `11` mục manifest, `1/1` hình CI)** |
| **Job F: Clinical, GDC, Pareto & LUAD (`se374_g7_phase1b_04b_v2`)** | **NVIDIA L4** | `STAGE2_TRAINING_METRICS.json` / `SE374_Stage2_Training_Results.zip` | `fe9763a1ac87c000` / `38f449542efb138b` | **HOÀN TẤT (`6` tệp lõi / `18` mục manifest, `4/4` Vision)** |
| **Job G: Visual Overlay & Hann-50% (`se374_g7_phase1b_02d_v2`)** | **NVIDIA G4** | `STAGE2_TRAINING_METRICS.json` / `SE374_Stage2_Training_Results.zip` | `d933a89ebec0578c` / `f04e79118cf04f3c` | **HOÀN TẤT (`6` tệp lõi / `23` mục manifest, `11/11` Vision)** |
| **Job H1: `224px` + `EMA vs Raw` (`se374_g7_phase1b_h1_arm5_224px`)** | **NVIDIA A100** | `STAGE2_TRAINING_METRICS.json` / `run.log` | `6437dcb80113848d` / `ee71cf3a3965567f` | **HOÀN TẤT (`5/5` tệp nhẹ SHA256 cục bộ, `.zip` lưu VPS)** |
| Zero-GPU Post-Fix Analytics (`phase1b_zero_gpu_analytics.json`) | Local CPU | `phase1b_zero_gpu_analytics.json` | `3dbd09a43dce436f` | **HOÀN TẤT (`3/3` hạng mục thống kê)** |
| **Job H2: `stage1` Memmap 336px (`se374_g7_phase1b_h2_arm5_s1`)** | **NVIDIA G4** | `STAGE2_TRAINING_METRICS.json` / `stage2_training_loss.png` | `ca49f55783603e09` / `eb974f754131584c` | **HOÀN TẤT (`5/5` tệp nhẹ SHA256 cục bộ, `EMA 72.23%` vs `Raw 71.68%`)** |
| **Job H3: `336px` `EMA vs Raw` (`se374_g7_phase1b_h3_arm5_raw_vs_ema`)** | **NVIDIA L4** | `STAGE2_TRAINING_METRICS.json` / `stage2_training_loss.png` | `d42d061bdc7483fb` / `9bedcf6be19928f2` | **HOÀN TẤT (`5/5` tệp nhẹ SHA256 cục bộ, Vision PASS)** |

*(Ghi chú truy xuất nguồn gốc: Ba thực nghiệm ablation `H1`, `H2`, `H3` lưu cục bộ `5` tệp minh chứng nhẹ `5/5` SHA256 trong `checkpoints/vps_evidence/` - tệp nén `.zip` trọng số ablation lưu trữ trên VPS - và đều chạy trên bản sao mã nguồn đã xác minh `local_repo.bundle` SHA256 `4da1d4849fefbad7` tương ứng commit **`8dabf65e`**. Bảng băm toàn kho `checkpoints/vps_evidence/SHA256SUMS.txt` khóa đủ **`228/228`** tệp, `BAD = 0`, `UNLISTED = 0`.)*
