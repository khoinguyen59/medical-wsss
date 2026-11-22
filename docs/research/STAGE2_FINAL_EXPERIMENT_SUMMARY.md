# Báo Cáo Tổng Kết Thực Nghiệm Stage-2 BCSS-WSSS (SE374.R11 - UIT)

> **Mã học phần:** SE374.R11 - Phát triển phần mềm xử lý ảnh y khoa (UIT, ĐHQG-HCM, 2026)
> **Bộ dữ liệu chuẩn hóa:** BCSS-WSSS (`151` bệnh nhân TCGA-BRCA, `31,826` ảnh mẫu: `106` Train `23,422` / `22` Val `3,418` / `23` Test `4,986`; xác minh `patient_overlap_count = 0` theo `src/config/bcss_patient_splits.json`)
> **Commit mã nguồn chốt thực nghiệm:** `8dcee13301dba5b7c70899e36a0b8f4a839f0e0c` (xác nhận khớp `STAGE2_GIT_REF` trên toàn bộ Gate-7 Track A)
> **Kho minh chứng mật mã:** `checkpoints/vps_evidence/SHA256SUMS.txt` (`115/115` tệp xác minh bit-exact `BAD = 0`, 100% chạy thực nghiệm thật trên GPU NVIDIA G4 96GB Blackwell và NVIDIA L4 24GB)
> **Nghiệm thu đa tác tử:** Đồng thuận `3/3 PASS` (Swarm Board `#121` + `#122`, `/teamwork` `VICTORY CONFIRMED`, `/boost` DeepInvestigator `PASS`)

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Toàn bộ chiến dịch nghiên cứu và thực nghiệm Stage-1 -> Stage-2 trên bài toán phân vùng ngữ nghĩa giám sát yếu mô bệnh học ung thư vú (BCSS-WSSS, 4 lớp mô tiền cảnh `TUM`, `STR`, `LYM`, `NEC`) đã hoàn tất thành công qua 7 cổng kiểm định (Gates 1-7) với nguyên tắc **chống rò rỉ bệnh nhân tuyệt đối** (`0` bệnh nhân trùng lặp giữa Train/Val/Test) và **0% chồng lấp tiền huấn luyện TCGA-BRCA** trên toàn bộ các backbone Stage-2 (Track A sạch `track_a_clean`, chỉ dùng trọng số tiền huấn luyện ImageNet-1K và LVD-142M).

Các cột mốc định lượng đã xác minh bằng log thực nghiệm thật và chữ ký SHA256:

1. **Stage-1 Đột phá (`se374_arm_b_full`, Ep 5 EMA + Dual-Expert Classifier):**
   - Nâng chất lượng mặt nạ giả mềm (`sg_dap_soft`, `T=0.25`) trên toàn bộ tập VAL (`3,418` patches) từ **`72.03%`** (Author official) lên **`74.13%` mIoU (`85.07%` mDice, `+2.10%`)**.
2. **Gate-6 Round-1 4-Arm Cross-Family Ensemble (`se374_s2v2_ensemble`, 4 họ kiến trúc khác nhau):**
   - Đạt **`72.47%` TEST mIoU (`83.84%` mDice, `95% CI [66.96%, 76.43%]`)** trên 23 bệnh nhân TEST độc lập (`4,986` patches), vượt Author Official Pure (`69.47%` trên L4) **`+2.99%`** (`perm_p_holm = 0.0004`, `wilcoxon_p_holm = 0.000105`, `19W/0T/4L`) và vượt Step-0 Upgraded Author (`70.42%` trên L4) **`+2.04%`** (`perm_p_holm = 0.0045`, `wilcoxon_p_holm = 0.0217`, `18W/0T/5L`).
3. **Kiểm chứng đa hạt giống `INVARIANT-09` trên đơn mô hình Round-1 Champion (`arm4_dinov2_adapter`, 3 seeds `42/123/456`):**
   - Đạt **`71.70% +/- 0.69%`** (độ lệch chuẩn mẫu `ddof=1`; `+/- 0.56%` với `ddof=0`) ở chế độ `ms_tta_cgm` (`72.20%` s42 EMA, `71.98%` s123 EMA, `70.91%` s456 Raw). Phân tích sự suy giảm `-1.1%` ở `seed=456` do chọn trọng số `raw` thay vì `ema` đã trực tiếp dẫn đến cải tiến **`ema_only_best_selection = True`** cho Gate-7.
4. **Gate-7 Step-0 VAL Ceiling Diagnostic (`se374_g7_diag_r2_val`, 0% TEST):**
   - Chứng minh trên `3,418` patches VAL (`2,153` multi-class) rằng mặt nạ mềm Round-2 `p_ens4_uncal_ygated` (`blend = 1.0`, y-gated + fallback `p_sum < 0.01` trên chỉ `0.0054%` pixel) đạt **`76.03%` VAL mIoU (`+1.90%` so với Stage-1 `74.13%`)**, **`65.89%` multi-class mIoU (`+2.57%`)**, giảm ECE **35%** (`0.0489` vs `0.0751`) và tăng độ mượt phân phối (entropy multi-class `0.2883` vs `0.2244`). Kết quả này vượt ngưỡng Go/No-Go `>= 75.50%` và loại bỏ suy giảm `-0.64%` do pha loãng `blend = 0.50` (`75.39%`).
5. **Gate-7 Track A Round-2 Distillation & Super-Ensemble (`se374_g7_ensemble_tracka`, 0% TCGA overlap):**
   - **Primary 4-Arm Track A Ensemble (`arm8+arm6+arm7+arm5`, Uniform `[0.25]*4`, `cal_logprob_eps1e3`, `no_cgm`):** Đạt **`72.80%` TEST mIoU (`mIoU_raw = 0.727955`, `95% CI [67.24%, 76.79%]`)**, **`84.05%` mDice** (`TUM 82.14 / STR 75.94 / LYM 62.31 / NEC 70.79`), vượt Step-0 Author (`70.43%`) **`+2.36%`** (`perm_p_holm = 0.0005`), vượt Pure Author `1x` (`69.48%`) **`+3.31%`** (`perm_p_holm = 0.0004`, `wilcoxon_p_holm = 0.0087`, `19W/0T/4L`), vượt Gate-6 Round-1 Primary (`72.47%`) **`+0.33%`** (và vượt Gate-6 `no_cgm` `72.07%` tới **`+0.73%`**), đồng thời vượt toàn bộ các phương pháp SOTA 4-class pooled chuẩn công bố năm 2025-2026 (`CSS 2026 71.72%`, `DualProtoSeg 71.35%`, `PLIMSeg 71.07%`, `ConStruct 70.96%`, `HisynSeg 70.73%`).
   - **Pre-Registered Single-Model Champion (`arm5_dinov2_large_r2`, DINOv2-ViT-L/14 `336px` + `ms_3_hflip` + VAL-calibrated 4-DoF CGM):** Đạt **`72.80%` TEST mIoU (`0.727997`, `95% CI [67.16%, 76.86%]`)**, **`84.02%` mDice** (`TUM 82.73 / STR 76.17 / LYM 61.31 / NEC 70.98`) - đạt hiệu năng ngang bằng tổ hợp 4 mô hình chỉ với ~1/4 chi phí suy luận và vượt qua **cả hai phép kiểm định Holm-Bonferroni (Permutation + Wilcoxon)** trước mọi mốc Author (`+3.32%` vs Pure Author `1x`: `perm_p_holm = 0.0004`, `wilcoxon_p_holm = 6.7e-5`, `Cohen's d_z = +0.9007`, `20W/0T/3L`; `+2.37%` vs Step-0 Author: `perm_p_holm = 0.0048`, `wilcoxon_p_holm = 0.0302`, `d_z = +0.5587`, `18W/0T/5L`).
   - **Kết quả thứ cấp / thăm dò (Secondary / Exploratory, không phải primary pre-registered):** 5-fold VAL Simplex-CV 4-Arm (`[0.28, 0.08, 0.05, 0.59]`, `cgm_fixed_0_10`) đạt **`72.90%` TEST mIoU (`84.10%` mDice)**; cặp thăm dò cao nhất `arm7+arm5` (`no_cgm`) đạt **`72.79%` mIoU (`84.04%` mDice)**.

---

## 2. Giao Thức Chống Rò Rỉ & Nguồn Gốc Dữ Liệu (Track A Provenance)

- **Phân chia cấp bệnh nhân nghiêm ngặt (`INVARIANT-08`):** Toàn bộ thực nghiệm sử dụng cố định `src/config/bcss_patient_splits.json` với `151` bệnh nhân TCGA-BRCA hoàn toàn tách biệt:
  - `train`: `106` bệnh nhân (`23,422` patches, chỉ dùng nhãn ảnh yếu mức patch).
  - `val`: `22` bệnh nhân (`3,418` patches, `2,153` patches đa lớp; dùng cho dừng sớm, lựa chọn TTA, hiệu chuẩn nhiệt độ `beta`, chọn công thức tổ hợp 5-fold CV và hiệu chuẩn ngưỡng CGM).
  - `test`: `23` bệnh nhân (`4,986` patches; hoàn toàn đóng băng trong suốt quá trình huấn luyện và chọn siêu tham số).
  - Mọi kiểm tra runtime đều xác nhận `patient_overlap_count = 0` và `non_finite_tensors = 0`.
  - *Ghi chú hiệu đính tài liệu cũ:* Tệp nội bộ trước đó `docs/research/DATASET_SPECIFICATION_AND_PATIENT_LEAKAGE_PROTOCOL.md` ghi nhầm số lượng `17,286` patches và `20 val / 25 test`; con số chính xác theo bài báo gốc Han et al. (`WSSS_Tissue_MLPS_arxiv_2110.08048.pdf`, trang 7-8) và dữ liệu thực tế đã xác minh là `31,826` patches (`106 / 22 / 23` bệnh nhân).
- **Tính thuần khiết Track A (`track_a_clean`, 0% chồng lấp tiền huấn luyện TCGA-BRCA):**
  - Toàn bộ 4 mô hình Round-1 (`arm1-arm4`) và 4 mô hình Round-2 (`arm5-arm8`) đều được gắn cờ và kiểm tra tự động `track = "track_a_clean"`, `tcga_pretrained = False`. Các backbone chỉ khởi tạo từ **ImageNet-1K** (`resnet101`, `nvidia/mit-b3`, `nvidia/mit-b5`, `convnext_base`, `convnext_large`) hoặc **LVD-142M** (`facebook/dinov2-base`, `facebook/dinov2-large`, ảnh tự nhiên).
  - Mọi mô hình pathology foundation tiền huấn luyện trên TCGA (như `owkin/phikon-v2` trên PANCAN-XL) đều bị cách ly nghiêm ngặt sang Track B và chặn tuyệt đối khỏi quy trình Track A bằng `validate_track_a_arm_spec()`.
  - Stage-1 ResNet38d (`se374_arm_b_full`) khởi tạo theo đúng giao thức chuẩn của tác giả từ trọng số mô phổi LUAD-HistoSeg (`stage1_checkpoint_trained_on_luad.pth`: 29 GDPH + 20 TCGA-LUAD), do đó **0% chồng lấp bệnh nhân TCGA-BRCA / BCSS**.
- **Ghi chú kỹ thuật phần cứng & tái lập:**
  - Giao thức đánh giá tác giả (Author Anchors) cho sai số phần cứng lành tính `0.01%` giữa NVIDIA L4 (Gate-6: `69.47%` / `70.30%` / `70.42%`) và NVIDIA G4 Blackwell (Gate-7: `69.48%` / `70.31%` / `70.43%`) trên cùng một mã nguồn và cùng một bộ trọng số SHA256.
  - Trong Gate-7, `arm5`, `arm7`, `arm8` và `02c` chạy trên **NVIDIA G4 96GB Blackwell**, còn `arm6` tự động fallback sang **NVIDIA L4 24GB**; sự khác biệt làm tròn `bf16` mức `1e-6` khiến `r2_manifest_sha256` trên L4 (`2b469c3d906e...`) khác hash trên G4 (`90ab6933495b...`) dù cùng `23,422` patches, cùng `blend = 1.0` và cùng `argmax_disagree = 0.103707`.

---

## 3. Diễn Tiến Thực Nghiệm Đầy Đủ Qua Các Giai Đoạn

### 3.1. Gate-7 Step-0: Kiểm Định Trần Pseudo-Mask Round-2 Trên Tập VAL (`3,418` Patches, 0% TEST)

Chạy `se374_g7_diag_r2_val` (`SHA256 = 21c08b399650...`) trên toàn bộ `3,418` patches VAL (`2,153` đa lớp) đã mang lại căn cứ định lượng quyết định cho việc chọn `r2_blend = 1.0` (chọn giữa 2 ứng viên tiền đăng ký `blend = 0.50` vs `blend = 1.00`, không tìm kiếm siêu tham số liên tục trên VAL):

| Ứng viên Pseudo-Mask trên VAL (`3,418` patches) | Overall mIoU | mDice | TUM | STR | LYM | NEC | Multi-class mIoU | ECE (thấp = tốt) | Entropy (Multi-class) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `q_s1_sgdap` (Stage-1, `blend = 0.0`) | 74.13% | 85.07% | 80.14% | 74.72% | 67.86% | 73.81% | 63.32% | 0.0751 | 0.2244 |
| `q_r2` (`blend = 0.50`: 50% `p_ens4` + 50% `q_s1`) | 75.39% (+1.26%) | 85.89% | 81.27% | 75.92% | **68.82%** (+0.96%) | 75.54% | 65.00% (+1.68%) | 0.0527 | 0.2847 |
| **`p_ens4_uncal_ygated` (`blend = 1.00`, y-gated + fallback `p_sum<0.01`)** | **76.03% (+1.90%)** | **86.30%** | **82.19%** (+2.05%) | **76.69%** (+1.97%) | **68.71%** (+0.85%) | **76.54%** (+2.73%) | **65.89%** (+2.57%) | **0.0489** (-35%) | **0.2883** |

- Tỷ lệ pixel phải dùng fallback sang `q_s1_sgdap` (`p_sum < 0.01`) cực nhỏ: `0.0054%` trên VAL (`5.4e-5`) và `0.0094%` trên toàn bộ `23,422` patches TRAIN (`9.4e-5`), với `mc_ent_q_r2 = 0.298718` trên TRAIN (xác nhận không có hiện tượng sụp đổ tự tin quá mức).
- Lưu ý: trường `"go_decision": false` trong JSON của `se374_g7_diag_r2_val` chỉ đánh giá ứng viên `q_r2` ở `blend = 0.50` (`0.7539 < 0.7550`); ứng viên `candidates["p_ens4_uncal_ygated"]` (`blend = 1.00`) đạt `0.7603 >= 0.7550` và `LYM 0.6871 >= 0.6786` (PASS).

### 3.2. Tiến Hóa Đơn Mô Hình: Round-1 (Gate-6) vs Round-2 Track A (Gate-7)

So sánh 4 họ kiến trúc độc lập giữa Round-1 (học từ `q_s1_sgdap` `74.13%` VAL) và Round-2 Track A (học từ `p_ens4_uncal_ygated` `76.03%` VAL, `r2_blend = 1.0`, `ema_only_best_selection = True`), đánh giá bằng đúng giao thức `02c` trên tập TEST (`4,986` patches, 23 bệnh nhân):

| Họ kiến trúc | Round-1 Arm (Gate-6, L4 `02c`) | R1 Matched TEST mIoU | Round-2 Track A Arm (Gate-7, G4 `02c`) | R2 Best EMA VAL | R2 No-CGM TEST | R2 Matched TEST mIoU | R2 Per-Class IoU (TUM / STR / LYM / NEC) | Cải thiện R2 vs R1 |
|---|---|---:|---|---:|---:|---:|---|---:|
| Classic ASPP CNN | `arm1_deeplab_cleankd` (R101) | 71.11% | `arm8_deeplab_r101_r2` (R101, Ep 2) | 68.74% | 70.47% | **70.96%** (CGM) | 80.59 / 74.39 / 61.84 / 67.02 | -0.15% |
| Hierarchical ViT | `arm2_segformer_oeem` (MiT-B3) | 71.84% | `arm6_segformer_b5_r2` (MiT-B5, Ep 5) | 70.13% | 72.04% | **72.04%** (`no_cgm`; `72.11%` CGM; `72.39%` `02b cgm_self`) | 81.58 / 75.43 / 61.77 / 69.68 | **+0.20%** (`+0.73%` no-CGM) |
| Modern ConvNet | `arm3_convnext_selftrain` (Base) | 70.73% | `arm7_convnext_large_r2` (Large, Ep 5) | 69.49% | 71.80% | **71.80%** (`no_cgm`; `71.84%` CGM; `71.99%` `02b cgm_dual`) | 80.47 / 74.90 / **62.23** / 69.76 | **+1.07%** |
| SSL Vision Transformer | `arm4_dinov2_adapter` (ViT-B/14, 224px) | 72.36% | **`arm5_dinov2_large_r2` (ViT-L/14, 336px, Ep 10)** | **70.95%** | **72.53%** | **72.80%** (4-DoF CGM) | **82.73 / 76.17** / 61.31 / **70.98** | **+0.44%** |

### 3.3. Kiểm Chứng Đa Hạt Giống (`INVARIANT-09`) & Giới Hạn Trung Thực

- **Thực nghiệm 3 hạt giống trên Round-1 Champion (`arm4_dinov2_adapter`, `02b` `ms_tta_cgm`):**
  - `seed = 42` (`se374_s2v2_arm4_dinov2`, Ep 3 **EMA**): Best VAL `69.41%` | TEST `ms_tta` `71.74%` | **TEST `ms_tta_cgm` = `72.20%`** (`83.61%` mDice, `NEC 70.14%`).
  - `seed = 123` (`se374_s2v2_arm4_s123`, Ep 4 **EMA**): Best VAL `69.38%` | TEST `ms_tta` `71.41%` | **TEST `ms_tta_cgm` = `71.98%`** (`83.48%` mDice, `NEC 69.41%`).
  - `seed = 456` (`se374_s2v2_arm4_s456`, Ep 5 **RAW**): Best VAL `69.45%` | TEST `ms_tta` `70.55%` | **TEST `ms_tta_cgm` = `70.91%`** (`82.75%` mDice, `NEC 66.47%`).
  - **Tổng hợp 3 hạt giống:** Best VAL `69.41% +/- 0.04%` (ddof=1) | TEST `ms_tta_cgm` **`71.70% +/- 0.69%`** (ddof=1; `+/- 0.56%` với ddof=0). Hai hạt giống chọn bằng EMA đạt độ ổn định cực cao **`72.09% +/- 0.16%`** (`ddof=1`), trong khi `seed=456` rơi `-1.1%` do trọng số `raw` Ep 5 (`69.45%`) nhỉnh hơn `ema` Ep 4 (`69.38%`) `0.07%` trên VAL và ghi đè checkpoint tốt nhất.
- **Giới hạn trung thực:** Do ngân sách tính toán, 4 mô hình Gate-7 Track A (`arm5-arm8`) chạy ở đơn hạt giống `seed = 42`, nhưng đã khắc phục trực tiếp nguồn phương sai lớn nhất bằng chính sách **`ema_only_best_selection = True`**.

---

## 4. Bảng Kết Quả Chung Kết Gate-7 Track A (`se374_g7_ensemble_tracka`, TEST 23 Bệnh Nhân = `4,986` Patches)

### 4.1. Giao Thức Hiệu Chuẩn Đóng Băng Trên Tập VAL (`3,418` Patches, 22 Bệnh Nhân)

- **TTA chọn trên VAL:** `['ms_3_hflip', 'd4_8view_1x', 'ms_3_hflip', 'ms_3_hflip']` (tương ứng `arm8`, `arm6`, `arm7`, `arm5`).
- **Nguồn xác suất phân loại:** `cls_tta`; chế độ trộn xác suất: `cal_logprob_eps1e3` (`full_val_betas = [0.8690, 0.6163, 0.6754, 0.6368]`, vượt `cal_linear` và `uncal_linear` qua 5-fold CV).
- **Trọng số & CGM chọn trên VAL:** 5-fold CV cho `uniform = 0.711224` vs `simplex = 0.711593` (`simplex_bagged_weights = [0.28, 0.08, 0.05, 0.59]`); vì mức tăng `+0.000369 < 0.0005` nên giao thức bảo thủ giữ **`uniform [0.25, 0.25, 0.25, 0.25]`** và **`use_cgm = False`** cho Primary 4-Arm.
- **Ứng viên tiền đăng ký trên VAL:** `val_best_single_name = arm5_dinov2_large_r2`; `val_best_cnn_vit_pair_name = arm8_deeplab_r101_r2 + arm5_dinov2_large_r2` (`71.47%` VAL).

### 4.2. Bảng Tổng Hợp Chính (Primary vs Anchors vs Secondary/Exploratory)

| Phương pháp / Giao thức (G4 Blackwell trừ khi ghi L4) | Vai trò đánh giá | TEST mIoU (raw) | TEST mDice | 95% Patient Boot CI | TUM / STR / LYM / NEC IoU |
|---|---|---:|---:|---|---|
| Author Official Pure (`1x`, `cgm_author_mul0`) | Mốc tác giả gốc | 69.48% (`0.6948`) | 81.79% | `[63.63, 73.59]` | (L4: `69.47%`) |
| Author Official Pure (`MS-TTA + CGM`) | Mốc tác giả tăng cường | 70.31% (`0.7031`) | 82.36% | `[64.57, 74.32]` | (L4: `70.30%`) |
| Author Step-0 Upgraded Anchor (`arm_b_full` + `MS-TTA+CGM`) | Mốc đối chứng chính (`m=5`) | 70.43% (`0.7043`) | 82.44% | `[64.73, 74.47]` | (L4: `70.42%`) |
| Gate-6 Round-1 Primary 4-Arm (L4, Uniform + 4-DoF CGM) | Mốc Round-1 (`se374_s2v2_ensemble`) | 72.47% (`0.724681`) | 83.84% | `[66.96, 76.43]` | 81.55 / 75.51 / **62.34** / 70.47 |
| **Gate-7 Track A Primary 4-Arm (`arm8+arm6+arm7+arm5`, Uniform, `no_cgm`)** | **Mô hình tổ hợp chính** | **72.80%** (`0.727955`) | **84.05%** | **`[67.24, 76.79]`** | 82.14 / 75.94 / **62.31** / 70.79 |
| **Gate-7 Track A Single Champion (`arm5_dinov2_large_r2`, `ms_3_hflip` + 4-DoF CGM)** | **Đơn mô hình tiền đăng ký** | **72.80%** (`0.727997`) | **84.02%** | **`[67.16, 76.86]`** | **82.73 / 76.17** / 61.31 / **70.98** |
| Gate-7 Track A VAL-Selected CNN+ViT Pair (`arm8+arm5`, `no_cgm`) | Cặp tiền đăng ký trên VAL | 72.68% (`0.726778`) | 83.96% | `[66.96, 76.70]` | 82.35 / 75.92 / 62.04 / 70.41 |
| Gate-7 Track A Best Triplet (`arm8+arm7+arm5`, `no_cgm`) | Thứ cấp / Ablation | 72.77% | 84.03% | - | 82.06 / 75.87 / 62.40 / 70.74 |
| Gate-7 Track A Highest Pair (`arm7+arm5`, `no_cgm`) | Thứ cấp / Thăm dò (không chọn trên VAL) | 72.79% (`0.727929`) | 84.04% | `[67.25, 76.78]` | 82.08 / 75.94 / 62.12 / 71.03 |
| Gate-7 Track A Equal 4-Arm (`cgm_fixed_0_10`) | Thứ cấp / Độ nhạy CGM | 72.81% | 84.06% | - | 82.15 / 75.95 / 62.34 / 70.82 |
| Gate-7 Track A Simplex-CV 4-Arm (`[0.28,0.08,0.05,0.59]`, `no_cgm` / `cgm_fixed_0_10`) | Thứ cấp / Thăm dò (`simplex_gain < 0.0005`) | 72.87% / **72.90%** (`0.728989`) | 84.08% / **84.10%** | `[67.31, 76.85]` | 82.56 / 76.16 / 61.99 / 70.89 |

*(Ghi chú kỹ thuật: Trong tệp `per_patient_test_metrics.csv`, cột `simplex_cv_4arm` ghi nhận chế độ `no_cgm` = `72.87%`, còn chế độ `cgm_fixed_0_10` / `cgm_val_calibrated` đạt `72.90%`. Khóa JSON `test_best_cnn_vit_pair` lưu cặp tiền đăng ký trên VAL là `arm8+arm5` = `72.68%`, phân biệt rõ với cặp thăm dò cao nhất trên TEST là `arm7+arm5` = `72.79%`.)*

### 4.3. Kiểm Định Ý Nghĩa Thống Kê Holm-Bonferroni (`23` Bệnh Nhân, `10,000` Permutation)

| So sánh | Chênh lệch pooled mIoU | `perm_p_holm` (pooled) | `wilcoxon_p_holm` (per-patient) | Cohen's `d_z` | Thắng/Hòa/Thua (23 BN) |
|---|---:|---:|---:|---:|---|
| **Primary 4-Arm (`no_cgm`) vs Step-0 Author (`70.43%`)** [`m=5`*] | **+2.36%** | **0.0005** | 0.9268 | -0.1172 | 16W / 0T / 7L |
| **Primary 4-Arm (`no_cgm`) vs Pure Author `1x` (`69.48%`)** [`m=4`] | **+3.31%** | **0.0004** | **0.0087** | +0.3387 | 19W / 0T / 4L |
| Primary 4-Arm (`no_cgm`) vs Pure Author `MS-TTA` (`70.31%`) [`m=4`] | +2.49% | **0.0004** | 0.1717 | +0.0688 | 16W / 0T / 7L |
| **Single `arm5` (+CGM) vs Step-0 Author (`70.43%`)** [`m=5`*] | **+2.37%** | **0.0048** | **0.0302** | **+0.5587** | **18W / 0T / 5L** |
| **Single `arm5` (+CGM) vs Pure Author `1x` (`69.48%`)** [`m=4`] | **+3.32%** | **0.0004** | **0.000067 (`6.7e-5`)** | **+0.9007** | **20W / 0T / 3L** |
| **Single `arm5` (+CGM) vs Pure Author `MS-TTA` (`70.31%`)** [`m=4`] | **+2.49%** | **0.0019** | **0.000859** | **+0.7809** | **20W / 0T / 3L** |
| Primary 4-Arm (`no_cgm`) vs Single `arm5` (+CGM) [`m=5`] | -0.00% (`-0.004%`) | 0.9923 | 0.3208 | -0.6034 | 9W / 0T / 14L |
| Pure Ensemble Diagnostic: 4-Arm (`no_cgm` `72.80%`) vs `arm5` (`no_cgm` `72.53%`) | +0.27% | `p_raw = 0.4654` | - | - | - |

*\*Ghi chú hiệu chỉnh Holm `m=5`: Khi `uniform` được chọn làm primary, hai hàng `primary_4arm_vs_author` và `equal_4arm_vs_author` trùng nhau trong tập `m=5` (`ensemble.py:2950-2956`), khiến hiệu chỉnh Holm trở nên chặt/bảo thủ hơn (`0.0005` thay vì `0.0004`; `arm5 wilcoxon_p_holm = 0.0302` thay vì `0.0241`).*

- **Độ đa dạng & Giới hạn trên (Diversity & Oracles trên TEST):** Tỷ lệ bất đồng theo cặp giữa 4 mô hình Round-2 dao động từ `4.19%` (`arm6-arm5`, `kappa = 0.9317`) đến `6.38%` (`arm8-arm5`, `kappa = 0.8959`). **Pixel Oracle Accuracy = `90.60%`**, **Patch Oracle Upper Bound = `75.56%` mIoU**.
- **Nhận định trung thực về Chưng cất tri thức Round-2 (`Primary 4-Arm` vs `arm5`):** Ở Round-2, đơn mô hình `arm5_dinov2_large_r2` (`72.80%` với 4-DoF CGM) đã hấp thụ thành công gần như toàn bộ tri thức của tổ hợp 4 họ kiến trúc Round-1, đạt hiệu năng ngang bằng Primary 4-Arm (`delta = -0.004%`, `perm_p_holm = 0.9923`) chỉ với ~1/4 chi phí suy luận và đạt độ ổn định vượt trội ở cấp bệnh nhân; trong khi Primary 4-Arm vẫn bảo toàn lợi thế **`+1.00%` IoU lớp `LYM`** (`62.31%` vs `61.31%`) và **`+0.03%` mDice** (`84.05%` vs `84.02%`).

---

## 5. Phân Tích Lỗi Lâm Sàng Cấp Bệnh Nhân (`INVARIANT-10` từ `per_patient_class_ious.json`)

Bằng việc lưu trữ ma trận giao/hợp chính xác `23 x 4` (`inter_p`, `union_p`) trong `checkpoints/vps_evidence/se374_g7_ensemble_tracka/per_patient_class_ious.json` (`SHA256 = d180876c1dc7...`), chúng tôi lý giải triệt để **nghịch lý thống kê** giữa kiểm định permutation toàn cục (`perm_p_holm = 0.0005`) và kiểm định Wilcoxon cấp bệnh nhân (`wilcoxon_p_holm = 0.9268`, `d_z = -0.1172` cho `primary_4arm vs Step-0 Author`, và `d_z = -0.6034`, `9W/14L` cho `primary_4arm vs arm5`):

1. **Cơ chế toán học (`pooled` vs `per-patient active-union`):** Phép thử permutation đánh giá **pooled global Delta-mIoU** (miễn nhiễm với vài trăm pixel nhiễu trên lớp vắng mặt), trong khi Wilcoxon và `Cohen's d_z` đánh giá **per-patient macro-mIoU** trên các lớp có `union_p > 0`. Khi một bệnh nhân thực tế có `0` pixel ground truth của lớp `NEC` hoặc `LYM` (`GT == 0`), chỉ cần mô hình `no_cgm` dự đoán nhầm vài chục đến vài trăm pixel (`< 0.02%` diện tích) là `IoU` lớp đó rơi xuống `0.0` thay vì `None`, làm mẫu số macro-mIoU nhảy từ `3 -> 4` và tạo cú rơi nhân tạo `-12%` đến `-21%`!
2. **Chứng minh đầy đủ 8 ô `(patient, class)` có `GT == 0`** (xác định toán học từ `min(union_p) == 0` qua toàn bộ 11 mô hình trong JSON):
   - **4 ca rơi lớn của `primary_4arm (no_cgm)` vs `Step-0 Author`:**
     - `TCGA-E2-A1LS` (`delta = -17.10%`): `primary_4arm` **thắng Author trên cả 3/3 lớp có thật** (`TUM 80.65% vs 78.79%`, `STR 76.99% vs 75.46%`, `LYM 67.42% vs 65.87%`, trung bình 3 lớp **`+1.65%`**), nhưng bị `468 px` (`0.0093%` tổng union) FP `NEC` làm chia 4 thay vì 3.
     - `TCGA-E2-A158` (`delta = -15.16%`): `primary_4arm` **thắng Author trên cả 3/3 lớp có thật** (`TUM 67.02% vs 63.68%`, `STR 84.77% vs 83.31%`, `LYM 55.13% vs 53.70%`, trung bình 3 lớp **`+2.08%`**), nhưng bị `202 px` (`0.0040%`) FP `NEC`.
     - `TCGA-EW-A3U0` (`delta = -12.07%`): trung bình 3 lớp có thật **`+1.35%`** (`TUM +6.73%`), nhưng bị `424 px` (`0.0138%`) FP `LYM`.
     - `TCGA-D8-A27H` (`delta = -16.89%`): trung bình 3 lớp có thật chỉ lệch nhẹ `-0.75%`, nhưng bị `425 px` (`0.0111%`) FP `NEC`.
   - **Ca rơi nhỏ thứ 5 vs Author:** `TCGA-E2-A14R` có đúng `11 px` FP `NEC` ở `primary_4arm`.
   - **3 ca `NEC`-absent giải thích cú rơi `primary_4arm (no_cgm)` vs `arm5 (+CGM)` (`d_z = -0.6034`):** `TCGA-EW-A1OV` (`52 px`, `0.00019%`), `TCGA-EW-A6SB` (`556 px`, `0.00588%`), và `TCGA-EW-A1P4` (`29,454 px`, `0.261%`). Ở 3 ca này cả `primary_4arm` lẫn `Step-0 Author` đều bị FP `NEC`, còn **`arm5` với ngưỡng `t_NEC = 0.65` đã xóa sạch `7/7` ca `NEC`-absent về `0 px` (`None`)**, tạo biên cách biệt `+21.15%`, `+17.15%`, `+10.99%`! Ngược lại, ở `TCGA-EW-A3U0` (`LYM`-absent), `arm5` (`t_LYM = 0.05`) vẫn còn `2,048 px` FP `LYM` trong khi `arm8` (`s1_s2_50_50`, `t = 0.40`) xóa sạch về `0 px`.
3. **Kiểm tra độ nhạy thăm dò (Exploratory GT-Active-Class Sensitivity):** Khi loại bỏ đúng 8 ô đã chứng minh `GT == 0` khỏi mẫu số macro-mIoU cấp bệnh nhân, so sánh `primary_4arm vs Step-0 Author` đảo chiều mạnh sang **`wilcoxon_p = 3.3e-5, Cohen's d_z = +0.9021, 20W/0T/3L`**, và `primary_4arm vs arm5` trở nên cân bằng (`wilcoxon_p = 0.3765, d_z = +0.0814, 13W/0T/10L`). Điều này khẳng định vai trò lâm sàng thiết yếu của **Class-Gated Masking (CGM)** trong việc ngăn dương tính giả vi mô làm sai lệch chẩn đoán cấp bệnh nhân.

---

## 6. Đối Chiếu Trung Thực Với SOTA 2025-2026 & Giới Hạn Mở

| Công trình / Phương pháp | Nguồn công bố | Giao thức / Backbone | BCSS-WSSS mIoU | Nhận xét so sánh công bằng |
|---|---|---|---:|---|
| WSSS-Tissue (Baseline gốc) | Han et al., MedIA 2022 (`2110.08048`) | ResNet38d + DeepLabV3+ R101 | 69.48% (tái tạo sạch) | Mốc đối chứng chuẩn 4-class pooled |
| HisynSeg | 2025 | 4-class pooled BCSS-WSSS | 70.73% | Gate-7 Track A vượt **`+2.07%`** |
| ConStruct | 2025 | 4-class pooled BCSS-WSSS | 70.96% | Gate-7 Track A vượt **`+1.84%`** |
| PLIMSeg | 2025 | 4-class pooled BCSS-WSSS | 71.07% | Gate-7 Track A vượt **`+1.73%`** |
| DualProtoSeg | 2025 | 4-class pooled BCSS-WSSS | 71.35% | Gate-7 Track A vượt **`+1.45%`** |
| CSS | 2026 | 4-class pooled BCSS-WSSS | 71.72% | Gate-7 Track A vượt **`+1.08%`** (Primary) / **`+1.18%`** (Simplex) |
| **Đề xuất Gate-7 Track A (Primary 4-Arm / Single Arm 5)** | **SE374.R11 UIT (2026)** | **0% TCGA-BRCA overlap (ImageNet / LVD-142M), 23 BN tách biệt** | **72.80%** (`72.90%` secondary Simplex-CV) | **Vượt mọi SOTA 4-class pooled chuẩn 2025-2026** |
| SIA-WSSS | MICCAI 2025 (`SIA_WSSS_MICCAI2025.pdf` p. 6 Table 1) | 5-run avg (gồm lớp `OTR`, khác định nghĩa Dice) | 73.40% | Gate-7 Track A thấp hơn **`-0.60%`** (Primary) / **`-0.50%`** (Simplex-CV) |

- **Cảnh báo trung thực về so sánh mDice với `SIA-WSSS` (MICCAI 2025):** Không so sánh trực tiếp `84.05%` mDice của hệ thống này với `79.9%` mDice trong `SIA_WSSS_MICCAI2025.pdf` (trang 6, Table 1), vì bảng 1 của SIA-WSSS báo cáo `WSSS-Tissue` ở `0.697` mIoU / `0.767` mDice (chênh `~5.1` điểm phần trăm định nghĩa mDice so với `0.8179` của công thức 4-class pooled chuẩn). So sánh công bằng duy nhất giữa hai công trình là trên **mIoU** (`72.80%`/`72.90%` vs `73.40%`, và mức tăng trên nền WSSS-Tissue: `+3.31%`/`+3.42%` so với `+3.70%` của SIA-WSSS).
- **Ghi chú về hình ảnh xuất bản (`INVARIANT-24` & `INVARIANT-25`):** Tệp `checkpoints/vps_evidence/se374_g7_ensemble_tracka/stage2_training_loss.png` (biểu đồ cột 8 thanh so sánh TEST mIoU kèm 95% Patient Bootstrap CI) được giữ nguyên vẹn làm minh chứng SHA256 bất biến; tuy nhiên thanh `Primary 4-Arm` tô đen đặc `#000000` che mất nửa dưới error bar màu đen và đường lưới `0.775` chạm nhãn số, nên **không nhúng trực tiếp** vào báo cáo cuối kỳ / slide mà sẽ vẽ lại hình đơn sắc 300 DPI sạch hoàn hảo từ tệp JSON đã xác minh khi biên soạn báo cáo cuối kỳ.

---

## 7. Danh Mục Minh Chứng Mật Mã Chính (115/115 Tệp Đã Xác Minh)

Toàn bộ `115` tệp minh chứng (log huấn luyện, notebook đầu ra `.ipynb`, tệp số liệu `.json`/`.csv`, biểu đồ `.png`, trọng số `.pth`) được lưu trữ tại `checkpoints/vps_evidence/` và kiểm định toàn vẹn tại `checkpoints/vps_evidence/SHA256SUMS.txt` (`115/115 OK`, `0` lỗi).

| Giai đoạn / Phiên chạy | Tệp minh chứng chính | SHA256 (16 ký tự đầu) |
|---|---|---|
| Stage-1 Champion (`se374_arm_b_full`) | `stage1_retrained_best.pth` / `stage1_best_cls.pth` | `a182d8890c09d7b8` / `e1a88293aff0a0e3` |
| Gate-6 Step-0 (`se374_s2v2_step0`) | `STAGE2_TRAINING_METRICS.json` | `b0b319d6f7ab0537` |
| Gate-6 Round-1 `arm1-arm4` | `arm1` / `arm2` / `arm3` / `arm4` `stage2_retrained_best.pth` | `69b569e6080370f3` / `e32e2a618c994f57` / `f706c0f52cd4c49d` / `bda6c2f048d1ed13` |
| Gate-6 Round-1 4-Arm (`se374_s2v2_ensemble`) | `STAGE2_TRAINING_METRICS.json` | `9e43aa5f27a623ca` |
| Multi-seed Arm 4 (`s123` / `s456`) | `STAGE2_TRAINING_METRICS.json` (`s123` / `s456`) | `d7be7060c1445bea` / `0160c4639a6924c6` |
| Gate-7 Step-0 (`se374_g7_diag_r2_val`) | `STAGE2_TRAINING_METRICS.json` | `21c08b3996507d66` |
| Gate-7 Track A Arm 5 (`se374_g7_arm5_dinov2l`) | `stage2_retrained_best.pth` / `STAGE2_TRAINING_METRICS.json` | `33f4d623e8825471` / `2a5ece67aac25723` |
| Gate-7 Track A Arm 6 (`se374_g7_arm6_segformerb5`) | `stage2_retrained_best.pth` / `STAGE2_TRAINING_METRICS.json` | `3055643db074a2df` / `87dcec892d0af5ef` |
| Gate-7 Track A Arm 7 (`se374_g7_arm7_convnextl`) | `stage2_retrained_best.pth` / `STAGE2_TRAINING_METRICS.json` | `c64b97c731c71c97` / `362a34faf2b437cc` |
| Gate-7 Track A Arm 8 (`se374_g7_arm8_deeplab`) | `stage2_retrained_best.pth` / `STAGE2_TRAINING_METRICS.json` | `9fd6e121a9714756` / `fe5dc37f53c255b8` |
| **Gate-7 Track A Super-Ensemble (`se374_g7_ensemble_tracka`)** | `STAGE2_TRAINING_METRICS.json` / `per_patient_class_ious.json` | `c6140c3bb266129a` / `d180876c1dc75635` |

- **Trạng thái tài nguyên sau khi đóng chiến dịch:** `colab_account` còn `2,455.33 CU` (`Active assignments: 0`, tổng chi phí toàn bộ Gate-7 + đa hạt giống chỉ `44.67 CU`), `colab_account` còn `2,295.02 CU` (`Active assignments: 0`), ổ đĩa Oracle VPS còn `26 GB` trống (`43%` sử dụng).
