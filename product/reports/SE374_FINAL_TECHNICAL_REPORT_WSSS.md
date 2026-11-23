# BÁO CÁO KỸ THUẬT TỔNG KẾT ĐỒ ÁN MÔN HỌC SE374.R11

## HỆ THỐNG PHÂN VÙNG NGỮ NGHĨA GIÁM SÁT YẾU CHO ẢNH MÔ BỆNH HỌC UNG THƯ VÚ (BCSS-WSSS): TỪ CHƯNG CẤT ĐA KIẾN TRÚC GATE-7 TRACK A ĐẾN TRẠM LÀM VIỆC LÂM SÀNG

> **Trường:** Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM - Khoa Kỹ thuật Phần mềm  
> **Học phần:** SE374.R11 - Phát triển phần mềm xử lý ảnh y khoa (Năm học 2026)  
> **Đề tài:** Đề tài 6 - Phân vùng ngữ nghĩa giám sát yếu trên ảnh tiêu bản giải phẫu bệnh toàn phần WSI (BCSS-WSSS / TCGA-BRCA)  
> **Giảng viên hướng dẫn:** TS. Bùi Cao Doanh (`doanhbc@uit.edu.vn`)  
> **Nhóm thực hiện & Thành viên:** [Chờ xác nhận danh sách thành viên / MSSV]  
> **Commit mã nguồn chốt thực nghiệm Gate-7:** `8dcee13301dba5b7c70899e36a0b8f4a839f0e0c` (chuỗi tích hợp sản phẩm `4c2da59`, `c9ec17a`)  
> **Kho minh chứng mật mã:** `checkpoints/vps_evidence/SHA256SUMS.txt` (`123/123` tệp cục bộ xác minh bit-exact, `BAD = 0`, `MISSING = 0`)  

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Trong giải phẫu bệnh số (Digital Pathology), tiêu bản mô học nhuộm Hematoxylin và Eosin (H&E) toàn phần (Whole Slide Images - WSI) đạt kích thước gigapixel (lên tới $100{,}000 \times 100{,}000$ điểm ảnh). Việc khoanh vùng thủ công ở cấp độ điểm ảnh để định lượng vi môi trường khối u tốn nhiều giờ công bác sĩ giải phẫu bệnh cho mỗi ca bệnh và chịu biến thiên đáng kể giữa các chuyên gia. Đồ án nghiên cứu, thiết kế và hiện thực hóa hoàn chỉnh hệ thống **Phân vùng Ngữ nghĩa Giám sát Yếu (Weakly-Supervised Semantic Segmentation - WSSS)** trên bộ dữ liệu chuẩn ung thư vú **BCSS-WSSS (TCGA-BRCA, 4 lớp mô tiền cảnh `TUM`, `STR`, `LYM`, `NEC` cùng lớp nền kính quang học `BACK`)**, chỉ sử dụng nhãn nhị phân cấp mảng ảnh ở giai đoạn huấn luyện nhưng xuất đầy đủ mặt nạ phân vùng mật độ cao và các dấu ấn sinh học tiên lượng trên trạm làm việc lâm sàng ngoại tuyến.

Toàn bộ chiến dịch nghiên cứu và thực nghiệm được vận hành qua 7 cổng kiểm định (Gates 1-7) trên dữ liệu bệnh nhân thật với hai ranh giới khoa học tuyệt đối: **(1) chống rò rỉ cấp bệnh nhân 0%** trên toàn bộ `151` bệnh nhân (`106` Train / `22` Val / `23` Held-out Test, `31,826` mảng ảnh), và **(2) tính thuần khiết Track A (`track_a_clean`, 0% chồng lấp tiền huấn luyện TCGA-BRCA)** khi toàn bộ các backbone Giai đoạn 2 chỉ khởi tạo từ ảnh tự nhiên ImageNet-1K và LVD-142M.

Các kết quả định lượng và kỹ thuật cốt lõi đã được kiểm chứng bằng nhật ký thực nghiệm thật trên GPU NVIDIA G4 96GB Blackwell, NVIDIA L4 24GB và NVIDIA Tesla T4 kèm chữ ký mật mã SHA256 bao gồm:

1. **Đột phá Giai đoạn 1 (`se374_arm_b_full`, ResNet-38d + PDA + D4 Equivariance + EMA):** Nâng chất lượng mặt nạ giả mềm `q_s1_sgdap` (`T = 0.25`) trên toàn bộ `3,418` mảng ảnh tập VAL từ `72.03%` (mốc tác giả gốc) lên **`74.13%` mIoU (`85.07%` mDice, tăng `+2.10%`)**.
2. **Tổ hợp 4 họ kiến trúc Vòng 1 - Gate-6 (`se374_s2v2_ensemble`, NVIDIA L4):** Đạt **`72.47%` TEST mIoU (`83.84%` mDice, 95% CI `[66.96%, 76.43%]`)** trên 23 bệnh nhân TEST độc lập (`4,986` patches), vượt mốc Author Official Pure `1x` (`69.47%` trên L4) **`+2.99%`** (`perm_p_holm = 0.0004`, `wilcoxon_p_holm = 0.000105`, `19W/0T/4L`) và vượt mốc Step-0 Upgraded Author (`70.42%` trên L4) **`+2.04%`** (`perm_p_holm = 0.0045`, `wilcoxon_p_holm = 0.0217`, `18W/0T/5L`).
3. **Kiểm chứng đa hạt giống và cơ chế chọn EMA duy nhất:** Thử nghiệm 3 hạt giống (`42`, `123`, `456`) trên đơn mô hình Vòng 1 `arm4_dinov2_adapter` đạt **`71.70% +/- 0.69%`** (`ddof=1`; hai hạt giống chọn bằng EMA đạt `72.09% +/- 0.16%`). Phân tích nguyên nhân sụt giảm `-1.1%` ở `seed=456` do chọn nhầm trọng số `raw` đã trực tiếp dẫn đến cải tiến **`ema_only_best_selection = True`** cho toàn bộ Gate-7.
4. **Kiểm định trần mặt nạ giả Vòng 2 - Gate-7 Step-0 (`se374_g7_diag_r2_val`, 0% TEST):** Mặt nạ mềm `p_ens4_uncal_ygated` (`r2_blend = 1.0`, y-gated kèm dự phòng `p_sum < 0.01` trên chỉ `0.0054%` điểm ảnh VAL và `0.0094%` TRAIN) đạt **`76.03%` VAL mIoU (`+1.90%` so với Giai đoạn 1)**, **`65.89%` multi-class mIoU (`+2.57%`)** và giảm sai số hiệu chuẩn kỳ vọng ECE **35%** (`0.0751` xuống `0.0489`).
5. **Chưng cất Vòng 2 và Tổ hợp Gate-7 Track A (`se374_g7_ensemble_tracka`, 0% TCGA-BRCA overlap):**
   - **Tổ hợp chính Primary 4-Arm (`arm8 + arm6 + arm7 + arm5`, Uniform `[0.25, 0.25, 0.25, 0.25]`, `cal_logprob_eps1e3`, `no_cgm`):** Đạt **`72.80%` TEST mIoU (`mIoU_raw = 0.727955`, 95% CI `[67.24%, 76.79%]`)**, **`84.05%` mDice** (`TUM 82.14% / STR 75.94% / LYM 62.31% / NEC 70.79%`), vượt Step-0 Author (`70.43%`) **`+2.36%`** (`perm_p_holm = 0.0005`), vượt Pure Author `1x` (`69.48%`) **`+3.31%`** (`perm_p_holm = 0.0004`, `wilcoxon_p_holm = 0.0087`, `19W/0T/4L`), và vượt toàn bộ các công bố SOTA 4-class pooled chuẩn năm 2025-2026 (`CSS 2026 71.72%`, `DualProtoSeg 71.35%`, `PLIMSeg 71.07%`, `ConStruct 70.96%`, `HisynSeg 70.73%`).
   - **Đơn mô hình chủ lực tiền đăng ký (`arm5_dinov2_large_r2`, DINOv2-ViT-L/14 `336px` + `ms_3_hflip` + 4-DoF CGM):** Đạt **`72.80%` TEST mIoU (`0.727997`, 95% CI `[67.16%, 76.86%]`)**, **`84.02%` mDice** (`TUM 82.73% / STR 76.17% / LYM 61.31% / NEC 70.98%`) - đạt hiệu năng tương đương tổ hợp 4 mô hình với khoảng 1/4 chi phí suy luận và vượt qua cả hai phép kiểm định Holm-Bonferroni (`perm_p_holm = 0.0048`, `wilcoxon_p_holm = 0.0302`, `Cohen's d_z = +0.5587`, `18W/0T/5L` so với Step-0 Author; `perm_p_holm = 0.0004`, `wilcoxon_p_holm = 6.7e-5`, `d_z = +0.9007`, `20W/0T/3L` so với Pure Author `1x`).
   - **Kết quả thăm dò thứ cấp (Secondary / Exploratory):** Tổ hợp Simplex-CV 4-Arm (`[0.28, 0.08, 0.05, 0.59]`, `cgm_fixed_0_10`) đạt **`72.90%` TEST mIoU (`0.728989`, `84.10%` mDice)**.
6. **Hệ sinh thái sản phẩm phần mềm lâm sàng và kiểm thử đa nền tảng:** Đóng gói Trạm Làm Việc Giải Phẫu Bệnh Số (`FastAPI` + `HTML5 Canvas` 100% ngoại tuyến với 9 chế độ QA, Split-Curtain, Magnifier Loupe, WSI Multi-Tile Stitch, xuất báo cáo PDF chuẩn CAP 2 trang kèm cảnh báo RUO và container hóa `Dockerfile` / `docker-compose.yml`), hệ thống điều phối đa GPU `colab_runner_local/` bảo vệ P0 chống tắt nhầm phiên ngoại lai external_service, và bộ kiểm thử tự động `531` bài kiểm thử (`496` bài trong `30` module tại `tests/` và `35` bài trong `2` module tại `colab_runner_local/`, trong đó `35/35` bài test chạy đạt 100% trên môi trường Linux ARM64 thực tế).

---

## 2. Tổng Quan Tài Liệu & Đối Chiếu Mã Nguồn Tham Chiếu

### 2.1 Khảo Sát Văn Hiến & Mã Nguồn Gốc

Toàn bộ quyết định kiến trúc, hàm mất mát và giao thức đánh giá của hệ thống được xây dựng dựa trên việc khảo sát trực tiếp 14 công trình chuyên sâu đã được lưu trữ và tóm lược có cấu trúc trong `docs/papers/` (bao gồm `SIA_WSSS_MICCAI2025.md`, `DualProtoSeg_arxiv_2512.10314.md`, `ARML_MICCAI2024.md`, `TPRO_MICCAI2024.md`, `S2C_From_SAM_to_CAMs_CVPR2024.md`, `D_CAM_MICCAI2025.md`, `DBDC_WSL_MICCAI2025.md`, `KMG_Mamba_MICCAI2025.md`, `M3amba_CVPR2025.md`, `MambaMIL_arxiv_2403.06800.md`, `QFKF_BMVC2025.md`, `LPD_arxiv_2512.18342.md`, `C2RM_Seg_arxiv_2606.25508.md`, và `PFM_DenseBench_arxiv_2602.03887.md`), kết hợp đối soát mã nguồn trực tiếp với hai kho tham chiếu `repos/WSSS-Tissue` (Han et al., *Medical Image Analysis* 2022) và `repos/LowMagWSS` (Do et al.).

Qua quá trình đối chiếu bài báo và thực nghiệm chẩn đoán mã nguồn, nhóm xác định 4 điểm nghẽn căn bản của bài toán phân vùng giám sát yếu trong giải phẫu bệnh số:

1. **Suy biến tiêu điểm bản đồ kích hoạt lớp (CAM Peak Collapse):** Bộ phân loại nhị phân đa nhãn chỉ cần kích hoạt vào một số cụm nhân tế bào dị dạng nổi bật nhất để tối thiểu hóa mất mát phân loại, bỏ sót vùng bào tương xâm lấn và ranh giới tiếp giáp giữa biểu mô u (`TUM`) với mô đệm (`STR`).
2. **Suy giảm nhận diện các lớp đuôi dài (`LYM` và `NEC`):** Tế bào lympho xâm nhập có kích thước nhỏ, phân bố phân tán giữa nền collagen mô đệm; mô hoại tử thường vắng mặt ở nhiều bệnh nhân nhưng khi xuất hiện lại có biên hình thái phức tạp, dễ bị các lớp chiếm ưu thế lấn át hoặc sinh dương tính giả vi mô.
3. **Lỗ hổng nền kính trắng (The Background Score Gap):** Do tập huấn luyện BCSS-WSSS chỉ giám sát 4 lớp mô (`0` đến `3`), mô hình phân vùng Stage-2 thuần túy không có lớp nền quang học rõ ràng, dẫn tới gán nhầm vùng lam kính sạch thành mô đệm hoặc khối u nếu thiếu cổng lọc quang học `OtsuClinicalGating` và `Pure-Glass Guard`.
4. **Rủi ro rò rỉ dữ liệu kép (Patient Leakage & Pretraining Overlap):** Chia ngẫu nhiên ở cấp mảnh ảnh làm các patch của cùng một bệnh nhân xuất hiện ở cả Train và Test; đồng thời việc sử dụng các mô hình nền tảng giải phẫu bệnh tiền huấn luyện trên kho PANCAN/TCGA (như `owkin/phikon-v2`) gây nguy cơ chồng lấp gián tiếp với tập kiểm thử TCGA-BRCA. Vì lý do đó, đồ án thiết lập ranh giới **Track A sạch (`track_a_clean`)** tách biệt hoàn toàn khỏi Track B.

### 2.2 Đối Chiếu Công Bằng Với Các Công Trình SOTA (2019-2026)

Bảng 1 tổng hợp so sánh định lượng trên cùng chuẩn đánh giá BCSS-WSSS giữa các công trình tiêu biểu công bố giai đoạn 2019-2026 và hệ thống Gate-7 Track A của đồ án.

| Công trình / Phương pháp | Nguồn công bố | Giao thức / Backbone | BCSS-WSSS mIoU (%) | Nhận xét so sánh công bằng |
| :--- | :--- | :--- | :---: | :--- |
| **HistoSegNet** (Chan et al.) | ICCV 2019 (`pp. 10662-10671`) | Patch-level CAM + Grad-CAM | 46.92% | Mốc sơ khởi WSSS mô bệnh học |
| **SEAM** (Wang et al.) | CVPR 2020 | ResNet-38 Equivariant Attention | 61.41% | Mở rộng đồng biến biến đổi tỷ lệ |
| **WSSS-Tissue** (Han et al.) | MedIA 2022 (`arXiv:2110.08048`) | ResNet-38d + DeepLabV3+ R101 | 69.34% / 69.48%* | Mốc đối chứng chuẩn 4-class pooled |
| **PistoSeg** (Fang et al.) | AAAI 2023 | Mosaic Synthesis + RFM | 69.95% | Tổng hợp ranh giới mô nhân tạo |
| **HisynSeg** (Fang et al.) | IEEE TMI 2025 | 4-class pooled BCSS-WSSS | 70.73% | Gate-7 Track A vượt **`+2.07%`** |
| **ConStruct** (Le et al.) | MIDL 2026 (`arXiv:2512.10316`) | 4-class pooled BCSS-WSSS | 70.96% | Gate-7 Track A vượt **`+1.84%`** |
| **PLIMSeg** | BSPC Vol. 112, 2026 | 4-class pooled BCSS-WSSS | 71.07% | Gate-7 Track A vượt **`+1.73%`** |
| **DualProtoSeg** (Vu et al.) | 2025 (`arXiv:2512.10314`) | 4-class pooled BCSS-WSSS | 71.35% | Gate-7 Track A vượt **`+1.45%`** |
| **CSS** | 2026 | 4-class pooled BCSS-WSSS | 71.72% | Gate-7 Track A vượt **`+1.08%`** (Primary) / **`+1.18%`** (Simplex) |
| **Đề xuất Gate-7 Track A (Primary 4-Arm / Single Arm 5)** | **SE374.R11 UIT (2026)** | **0% TCGA-BRCA overlap (ImageNet / LVD-142M), 23 BN tách biệt** | **72.80%** (`72.90%` Secondary) | **Vượt mọi SOTA 4-class pooled chuẩn 2025-2026** |
| **SIA-WSSS** (Fan et al.) | MICCAI 2025 (`10.1007/978-3-032-05141-7_56`) | ViT-B 5-run avg (gồm lớp `OTR`, khác định nghĩa Dice) | 73.40% | Gate-7 Track A thấp hơn **`-0.60%`** (Primary) / **`-0.50%`** (Simplex) |

*\*Ghi chú tái lập mốc tác giả:* `69.34%` là kết quả tái lập không dùng CGM trên Tesla T4 (`M1`), còn `69.48%` là kết quả Author Official Pure `1x` có cổng CGM (`cgm_author_mul0`) trên NVIDIA G4 Blackwell (`69.47%` trên L4).

> **Cảnh báo trung thực về định nghĩa thước đo mDice khi đối chiếu với `SIA-WSSS` (MICCAI 2025):** Không so sánh trực tiếp chỉ số `84.05%` mDice của hệ thống này với `79.90%` mDice trong `SIA_WSSS_MICCAI2025.pdf` (trang 6, Table 1), bởi vì bảng 1 của `SIA-WSSS` báo cáo cùng mốc `WSSS-Tissue` ở mức `0.697` mIoU và `0.767` mDice (chênh khoảng `5.1` điểm phần trăm theo định nghĩa mDice so với `0.8179` của công thức 4-class pooled chuẩn). Do đó, phép so sánh khoa học công bằng duy nhất giữa hai công trình là trên thước đo **mIoU** (`72.80%` Primary / `72.90%` Secondary so với `73.40%`, và biên độ cải thiện trên nền WSSS-Tissue tương ứng là `+3.31%` / `+3.42%` so với `+3.70%` của `SIA-WSSS`).

---

## 3. Dữ Liệu BCSS-WSSS, Chống Rò Rỉ Bệnh Nhân & Tính Thuần Khiết Track A

### 3.1 Đặc Tả Bộ Dữ Liệu & Phân Chia Khóa Cứng 151 Bệnh Nhân

Bộ dữ liệu chuẩn **BCSS-WSSS** trích xuất từ kho giải phẫu bệnh ung thư vú **TCGA-BRCA** (Amgad et al., *Bioinformatics* 2019; Han et al., *MedIA* 2022) ở độ phóng đại chuẩn lâm sàng **40x (`0.2500 MPP`)**, kích thước cố định `224 x 224` điểm ảnh (`50,176` điểm ảnh mỗi mảng), gồm 4 phân lớp mô học tiền cảnh và 1 lớp nền quang học:

- **Lớp `0` (`TUM` - Tumor Epithelium):** Biểu mô khối u ác tính với nhân dị dạng, tỷ lệ nhân trên bào tương cao, bắt màu tím Hematoxylin đậm.
- **Lớp `1` (`STR` - Stroma):** Mô đệm sợi liên kết giàu collagen bao quanh ổ u, bắt màu hồng Eosin đặc trưng.
- **Lớp `2` (`LYM` - Lymphocytic Infiltrate):** Bạch huyết bào xâm nhập miễn dịch có nhân tròn nhỏ, đồng dạng, sắp xếp thành cụm hoặc phân tán.
- **Lớp `3` (`NEC` - Necrosis):** Vùng mô hoại tử đặc trưng bởi hiện tượng vỡ nhân (`karyorrhexis`), tan nhân (`karyolysis`) trên nền mảnh vụn ưa Eosin.
- **Lớp `4` (`BACK` - White Glass / Background):** Nền kính trắng quang học và vùng ngoài ROI; ở pha suy luận được tách độc lập bởi cổng quang học `OtsuClinicalGating` kết hợp `Pure-Glass Guard` (`>= 512 px`), tuyệt đối không sao chép nhãn Ground Truth `label == 4` vào dự đoán AI.

Toàn bộ danh sách chia tách theo định danh bệnh nhân (`TCGA-XX-XXXX`) được khóa cứng tại `src/config/bcss_patient_splits.json` và kiểm định tự động ở mọi phiên chạy qua `src/datasets/patient_split.py` và `tests/test_patient_split.py`. Mọi nhật ký Gate-6 và Gate-7 đều ghi nhận tường minh `patient_overlap_count = 0` và `non_finite_tensors = 0`.

| Phân tập dữ liệu | Nguồn bệnh nhân | Số bệnh nhân | Số mảng ảnh (`224x224`) | Định dạng nhãn | Rò rỉ bệnh nhân | Vai trò trong pipeline |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **Train** | BCSS-WSSS (TCGA-BRCA) | 106 | 23,422 | Nhãn yếu đa lớp `y_cls` `[4]` trong tên tệp | 0% | Huấn luyện Stage-1, sinh mặt nạ mềm Round-1 và Round-2 |
| **Validation** | BCSS-WSSS (TCGA-BRCA) | 22 | 3,418 (`2,153` đa lớp) | Mặt nạ điểm ảnh đầy đủ `{0, 1, 2, 3, 4}` | 0% | Dừng sớm EMA, chọn TTA, hiệu chuẩn `beta`, CV 5-fold, CGM |
| **Held-Out Test** | BCSS-WSSS (TCGA-BRCA) | 23 | 4,986 | Mặt nạ điểm ảnh đầy đủ `{0, 1, 2, 3, 4}` | 0% | Đánh giá mù cuối cùng, bootstrap CI và Holm-Bonferroni |
| **Tổng cộng** | **BCSS-WSSS** | **151** | **31,826** | **Bất đối xứng (Yếu / Đầy đủ)** | **0% Tuyệt đối** | Chuẩn đánh giá chính thức của toàn hệ thống |
| **Cross-Organ** | LUAD-HistoSeg (Ung thư phổi) | Cohort độc lập | Tập Test mô phổi | Ánh xạ `LUAD_TO_BCSS_LUT = [0, 3, 2, 1, 4]` | 0% | Kiểm định tổng quát hóa chéo cơ quan Zero-Shot (`04b`) |

*Ghi chú hiệu đính tài liệu nội bộ cũ:* Tài liệu đặc tả nháp sớm trước đây từng ghi nhầm `17,286` patches và phân bổ `20 val / 25 test`. Số liệu chính xác đối chiếu trực tiếp từ bài báo gốc Han et al. (`arXiv:2110.08048`, trang 7-8) và kiểm đếm thực tế trên kho dữ liệu `data/BCSS-WSSS/` là **`31,826` patches (`106 / 22 / 23` bệnh nhân)**.

### 3.2 Tính Thuần Khiết Track A & Kỷ Luật Miền Chuẩn Hóa Đầu Vào

Để loại trừ mọi nghi ngờ về rò rỉ tri thức gián tiếp từ các kho giải phẫu bệnh tiền huấn luyện chứa TCGA-BRCA, toàn bộ 4 nhánh Round-1 (`arm1` đến `arm4`) và 4 nhánh Round-2 (`arm5` đến `arm8`) đều mang siêu dữ liệu bắt buộc `track = "track_a_clean"`, `tcga_pretrained = False` và được kiểm tra tự động bằng hàm `validate_track_a_arm_spec()` trong `src/stage2_v2/ensemble.py`:

- **Backbone Stage-2 Track A:** Chỉ khởi tạo từ trọng số ảnh tự nhiên **ImageNet-1K** (`resnet101`, `nvidia/mit-b3`, `nvidia/mit-b5`, `convnext_base`, `convnext_large`) hoặc **LVD-142M** (`facebook/dinov2-base`, `facebook/dinov2-large`). Mọi mô hình foundation tiền huấn luyện trên kho bệnh học TCGA (chẳng hạn `owkin/phikon-v2`) bị cách ly hoàn toàn sang Track B.
- **Backbone Stage-1 (`se374_arm_b_full`):** Khởi tạo theo đúng giao thức công bố gốc của tác giả từ trọng số mô phổi `stage1_checkpoint_trained_on_luad.pth` (`29` ca GDPH và `20` ca TCGA-LUAD, `228/228` tầng khớp), bảo đảm **0% trùng lặp bệnh nhân TCGA-BRCA**.
- **Hợp đồng miền chuẩn hóa đầu vào (`Input Normalization Contract`):**
  - Các trọng số pretrained chính thức của WSSS-Tissue (`stage1_checkpoint_trained_on_bcss.pth`, `stage2_checkpoint_trained_on_bcss.pth`) và Stage-1 `se374_arm_b_full` sử dụng chuẩn hóa trực tiếp về đoạn `[0.0, 1.0]` (`img / 255.0`), **không** trừ trung bình ImageNet. Lịch sử thực nghiệm Colab Run 2 đã chứng minh việc áp nhầm `IMAGENET_MEAN/STD` lên checkpoint `[0.0, 1.0]` làm phân phối BatchNorm sụp đổ xuống `5.74%` mIoU.
  - Toàn bộ 8 mô hình Stage-2 v2 (`arm1` đến `arm8`) khởi tạo từ ImageNet-1K / LVD-142M sử dụng đúng chuẩn hóa `IMAGENET_MEAN = (0.485, 0.456, 0.406)` và `IMAGENET_STD = (0.229, 0.224, 0.225)` có ghi nhận tường minh trường `"input_norm": "imagenet"` trong siêu dữ liệu checkpoint.
- **Phòng vệ nạp thư mục đa tầng:** Toàn bộ DataLoader trong `src/datasets/`, `src/training/dataset_loader.py` và `src/stage2_v2/` áp dụng `resolve_split_path` cùng `rglob("*.png")` kèm kiểm tra tập rỗng, loại bỏ hoàn toàn lỗi sập DataLoader do giả định thư mục phẳng `glob("*.png")` trên cấu trúc lồng `train/img/*.png`.

---

## 4. Phương Pháp Luận & Kiến Trúc Gate-7 Track A

Kiến trúc tổng thể 6 tầng của hệ thống Gate-7 Track A kết nối liền mạch từ huấn luyện nhãn yếu Giai đoạn 1, chưng cất tri thức hai vòng đa họ kiến trúc, tổ hợp xác suất hiệu chuẩn 5 chuyên gia cho tới Trạm Làm Việc Lâm Sàng FastAPI được trình bày tại **Hình 4**.

![Hình 4: Sơ đồ kiến trúc đơn sắc 6 tầng của hệ thống SE374.R11 Gate-7 Track A từ Giai đoạn 1 SG-DAP, chưng cất hai vòng đa họ kiến trúc đến Trạm Làm Việc Lâm Sàng FastAPI và HTML5 Canvas ngoại tuyến.](figures/fig4_gate7_pipeline_architecture_bw.png)

### 4.1 Giai Đoạn 1: Mạng Phân Loại Đa Nhãn, PDA & Mặt Nạ Giả Mềm SG-DAP

Giai đoạn 1 (`se374_arm_b_full`, kiến trúc `WSSSTissueStage1NetCAM` trên nền `ResNet-38d`, `105.09M` tham số) nhận ảnh H&E đầu vào $X$ và vector nhãn yếu cấp mảng $y \in \{0, 1\}^4$. Khác với cấu hình SGD truyền thống dễ hội tụ cục bộ, cấu hình đột phá `arm_b_full` tích hợp đồng thời 4 cơ chế:

1. **Hàm mất mát phân loại biên mềm (Multi-Label Soft Margin Loss)** kết hợp bộ tối ưu AdamW (`lr_bb = 8e-5`, `lr_fc8 = 2.4e-4`, lịch Cosine Annealing, khởi động ấm 2 epoch).
2. **Chú ý xóa mờ lũy tiến nhân tính (Multiplicative Progressive Dropout Attention - PDA):** Kích hoạt từ epoch 3 với ngưỡng suy giảm $\mu_t = \max(1.0 \cdot 0.98^{t-3}, 0.65)$, tạm thời che các đỉnh kích hoạt mạnh nhất để buộc mạng mở rộng bản đồ CAM sang toàn bộ ranh giới biểu mô và mô đệm.
3. **Tính nhất quán đồng biến biến đổi hình học ($D_4$ + đa tỷ lệ $\{1.0, 1.25\}$):** Do tiêu bản cắt lát mô học bất biến dưới phép quay và phản xạ gương, thành phần phạt nhất quán đồng biến ($\lambda_{\text{eq}} \le 0.25$, tăng dần từ epoch 2 đến epoch 5) buộc bản đồ CAM khớp nhau giữa các góc nhìn, đồng thời duy trì trung bình trượt hàm mũ **EMA** (`decay = 0.995`) cho toàn bộ trọng số.
4. **Tinh chỉnh mặt nạ mềm dẫn đường sắc tố H&E (`SG-DAP`, nhiệt độ $T = 0.25$):** Sử dụng phân rã quang học Beer-Lambert (`StainDeconvLayer`, $\mathbf{OD} = -\log_{10}(I/I_0)$, $\mathbf{S} = \max(\mathbf{M}_{\text{H\&E}}^{-1}\mathbf{OD}, 0)$) tách riêng mật độ sắc tố nhân Hematoxylin ($S_H$) và bào tương Eosin ($S_E$), khuếch tán ái lực cục bộ theo cấu trúc sắc tố để bảo toàn ranh giới sắc nét cho cụm tế bào `LYM` và ổ `NEC`. Nhờ đó, chất lượng mặt nạ giả mềm `q_s1_sgdap` trên toàn bộ `3,418` patches VAL đạt **`74.13%` mIoU (`85.07%` mDice)**, tăng **`+2.10%`** so với mốc tác giả gốc (`72.03%`).

### 4.2 Chưng Cất Hai Vòng & Bốn Họ Kiến Trúc Trực Giao Dual-Head

Thay vì phụ thuộc vào một kiến trúc CNN đơn lẻ dễ mắc lỗi hệ thống (inductive bias), tầng `src/stage2_v2/` xây dựng tổ hợp 4 họ kiến trúc hoàn toàn độc lập, mỗi mô hình đều mang thiết kế **hai đầu ra đồng thời (Dual-Head)**: đầu phân vùng mật độ cao xuất bản đồ xác suất 4 lớp $(B, 4, H, W)$ và đầu phân loại phụ xuất logit cấp mảng $(B, 4)$ phục vụ cổng CGM:

| Nhánh Gate-7 Track A | Họ kiến trúc | Backbone (0% TCGA) | Độ phân giải | Chính sách chọn trọng số | Best EMA VAL mIoU |
| :--- | :--- | :--- | :---: | :--- | :---: |
| `arm8_deeplab_r101_r2` | Classic ASPP CNN | `resnet101` (ImageNet-1K) | `224 x 224` | `ema_only` (Ep 2) | 68.74% |
| `arm6_segformer_b5_r2` | Hierarchical ViT | `nvidia/mit-b5` (ImageNet-1K, `drop_path = 0.15`) | `224 x 224` | `ema_only` (Ep 5) | 70.13% |
| `arm7_convnext_large_r2` | Modern Pure ConvNet | `convnext_large` (ImageNet-1K, `stoch_depth = 0.30`) | `224 x 224` | `ema_only` (Ep 5) | 69.49% |
| **`arm5_dinov2_large_r2`** | **Self-Supervised ViT** | **`facebook/dinov2-large` (LVD-142M, mở khóa 8 khối cuối)** | **`336 x 336`** (`scale = 1.5`) | **`ema_only` (Ep 10)** | **70.95%** |

Quy trình chưng cất hai vòng (Two-Round Cross-Family Distillation) diễn ra sạch hoàn toàn khỏi tập TEST:

1. **Vòng 1 (Gate-6):** Huấn luyện 4 nhánh đầu tiên (`arm1` DeepLab-R101, `arm2` SegFormer-MiT-B3, `arm3` ConvNeXt-Base, `arm4` DINOv2-ViT-B/14) từ mặt nạ giả mềm Giai đoạn 1 `q_s1_sgdap` (`74.13%` VAL), kết hợp mất mát chưng cất mềm KL-OEEM / Soft-KD / Soft-SCE, Dice mềm, MIL và phân loại phụ.
2. **Sinh mục tiêu mềm Vòng 2 (`r2_blend = 1.0`, y-gated):** Lấy trung bình xác suất dự đoán của 4 giáo viên Vòng 1 trên `23,422` mảng ảnh TRAIN, nhân cổng nhãn yếu thật $y_{\text{cls}} \in \{0, 1\}^4$ để khóa tuyệt đối các lớp vắng mặt, và chuẩn hóa lại phân phối:
$$q_{\text{R2}}(u, c) = \frac{y_c \cdot \bar{p}_{\text{ens4}}(u, c)}{\sum_{k=0}^{3} y_k \cdot \bar{p}_{\text{ens4}}(u, k)} \quad \text{khi } \sum_{k=0}^{3} y_k \cdot \bar{p}_{\text{ens4}}(u, k) \ge 0.01$$
Tại những điểm ảnh hiếm hoi có tổng xác suất dưới ngưỡng `0.01` (chỉ chiếm `0.0054%` trên VAL và `0.0094%` trên toàn bộ `23,422` patches TRAIN), hệ thống tự động lùi an toàn về phân phối `q_s1_sgdap`. Trên TRAIN, entropy đa lớp đạt `mc_ent_q_r2 = 0.298718`, khẳng định nhãn mềm giàu thông tin và không bị sụp đổ tự tin quá mức.
3. **Chính sách tuyển chọn `ema_only_best_selection = True`:** Được thiết lập sau phát hiện phương sai đa hạt giống ở Vòng 1 (Mục 5.6), buộc vòng huấn luyện Gate-7 chỉ lưu checkpoint tốt nhất theo nhánh EMA trên tập VAL nhằm triệt tiêu hiện tượng dao động cuối epoch của trọng số `raw`.

### 4.3 Tổ Hợp Log-Xác Suất Hiệu Chuẩn, Cổng CGM 5 Chuyên Gia & Lọc Quang Học

Toàn bộ tham số suy luận của Gate-7 Track A được đóng băng 100% trên 22 bệnh nhân VAL (`3,418` patches) thông qua kiểm định chéo 5-fold theo bệnh nhân và kiểm định chéo bỏ từng bệnh nhân (`22-fold LOPOCV`) trước khi chạm tập TEST:

- **Lựa chọn TTA theo từng nhánh trên VAL:** `ms_3_hflip` cho `arm8`, `arm7`, `arm5` (đa tỷ lệ `{0.75, 1.0, 1.25}` kèm lật ngang) và `d4_8view_1x` cho `arm6` (nhóm nhị diện 8 góc nhìn). Nguồn xác suất phân loại tối ưu là `cls_tta` (`mean_macro_auroc = 0.9340`, `mean_brier = 0.0761`, vượt `cls_1x` `0.9324` và `cls_d4` `0.9333`).
- **Trộn log-xác suất hiệu chuẩn nhiệt độ (`cal_logprob_eps1e3`):** Mỗi nhánh $m \in \{1, 2, 3, 4\}$ được gán hệ số nhiệt độ tối ưu $\beta_m$ khớp trên VAL (`full_val_betas = [0.8690, 0.6163, 0.6754, 0.6368]`) và hợp nhất trong miền log-xác suất làm trơn $\epsilon = 10^{-3}$:
$$\log \tilde{p}_{\text{ens}}(u, c) = \sum_{m=1}^{4} w_m \cdot \beta_m \log\left(p_m(u, c) + \epsilon\right)$$
Chế độ `cal_logprob_eps1e3` đạt điểm OOF 5-fold CV cao nhất (`0.711224` uniform / `0.711593` simplex), vượt cả `cal_linear` (`0.710122` / `0.710309`) và `uncal_linear` (`0.709606` / `0.709562`). Do mức cải thiện của trọng số simplex bagged `[0.28, 0.08, 0.05, 0.59]` chỉ đạt `+0.000369 < 0.0005` so với trọng số đồng đều, quy tắc bảo thủ tiền đăng ký chốt **`uniform [0.25, 0.25, 0.25, 0.25]`** và **`no_cgm`** làm mô hình tổ hợp chính Primary 4-Arm, trong khi `arm5_dinov2_large_r2` sử dụng **4-DoF CGM** hiệu chuẩn trên VAL (`t = [0.50, 0.60, 0.05, 0.65]`).
- **Tách nền kính quang học & Dấu ấn sinh học lâm sàng:** Ở pha suy luận lâm sàng, `OtsuClinicalGating` kết hợp `Pure-Glass Guard` (`gray >= 242`, `sat <= 8`, vùng liên thông `>= 512 px`) tách sạch lớp `BACK (4)` mà không sử dụng bất kỳ thông tin nào từ mặt nạ Ground Truth. Từ mặt nạ 5 lớp, hệ thống tính tự động hai dấu ấn sinh học chuẩn hóa theo khuyến cáo Hội Bác sĩ Giải phẫu bệnh Hoa Kỳ (CAP) và hướng dẫn quốc tế (Symmans et al., *JCO* 2007; van Pelt et al., *Virchows Archiv* 2018):
$$\text{VTP}_{\text{cellularity}} = \frac{N_{\text{TUM}}}{N_{\text{TUM}} + N_{\text{STR}} + N_{\text{LYM}} + N_{\text{NEC}}} \times 100\%, \quad \text{VTP}_{\text{RCB}} = \frac{N_{\text{TUM}}}{N_{\text{TUM}} + N_{\text{NEC}}} \times 100\%$$
$$\text{TSR} = \frac{N_{\text{TUM}}}{N_{\text{TUM}} + N_{\text{STR}}} \times 100\% \quad (\text{TSR} \le 50\% \Rightarrow \text{Nhóm nguy cơ cao Stroma-Rich})$$

---

## 5. Kết Quả Thực Nghiệm Định Lượng & Kiểm Định Thống Kê

### 5.1 Nền Tảng Lịch Sử Thực Nghiệm Colab (Run 1 đến Run 5 & Ablation M1-M7)

Trước khi bước vào thiết kế chưng cất đa họ kiến trúc Stage-2 v2, đội ngũ đã thực hiện 5 phiên thực nghiệm tiến hóa có đối chứng trên GPU Tesla T4 (`experiments/benchmarks/COLAB_5_RUNS_EVOLUTION_COMPARISON.md`) nhằm kiểm chứng chuỗi tiền xử lý, chuẩn hóa và suy luận đa phương pháp `M1` đến `M7` trên toàn bộ `4,986` ảnh TEST:

| Phiên chạy / Cấu hình | Quy mô ảnh TEST | Cơ chế kỹ thuật cốt lõi | 4-Cls mIoU (%) | 5-Cls mIoU (%) | 4-Cls mDice (%) | BACK IoU (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| Run 1 (`colab_run_1_heuristic`) | 150 (Sanity) | Ngưỡng màu HSV ngây thơ (chưa nạp trọng số) | 9.13% | 9.78% | ~16.5% | 12.37% |
| Run 2 (`colab_runs`) | 150 (Sanity) | Lệch miền `ImageNet Normalize` trên checkpoint `[0, 1]` | 5.74% | 7.59% | 9.34% | 15.00% |
| Run 4 (`colab_run_4_full`) | 4,986 (Full) | Chỉ Stage-2 DeepLabV3+ đơn lẻ + Otsu cũ | 68.71% | 67.36% | 81.15% | 70.26% |
| Run 5 (`M1_Baseline_SingleScale`) | 4,986 (Full) | Tái lập WSSS-Tissue 1x + Pure-Glass Guard | 69.34% | 75.46% | 81.69% | 99.93% |
| Run 5 (`M5_Proposed_SOTA`) | 4,986 (Full) | + Stage-1 CGM + D4 TTA + CAM Fusion + Prior Calib + Bilateral | 70.49% | 76.38% | 82.51% | 99.93% |
| Run 3 v2 (`M6_MultiScale_Pyramid`) | 4,986 (Full) | Multi-Scale `{0.75, 1.0, 1.25}` x D4 + Entropy Weighting | **70.53%** | **76.41%** | **82.54%** | **99.93%** |

Chuỗi thực nghiệm nền tảng này xác nhận ba bài học kỹ thuật quyết định: (1) tuân thủ tuyệt đối chuẩn hóa `[0.0, 1.0]` với checkpoint gốc (`5.74% -> 69.34%`), (2) `Pure-Glass Guard` nâng `BACK IoU` từ `70.26%` lên `99.93%` (`100.00%` Recall trên `18,279,223` điểm ảnh kính trắng) và xóa sạch `781,168` điểm ảnh `NEC` từng bị đục lỗ nhầm ở Run 4, và (3) tổ hợp suy luận `M6` đạt `70.53%` 4-class mIoU (Wilcoxon $p = 2.60 \times 10^{-15}$ so với `M1`), nhưng vẫn bị giới hạn bởi trần biểu diễn của một kiến trúc CNN Stage-2 duy nhất.

### 5.2 Gate-7 Step-0: Kiểm Định Trần Mặt Nạ Giả Vòng 2 Trên Tập VAL (0% TEST)

Chạy kiểm định `se374_g7_diag_r2_val` (`SHA256 = 21c08b3996507d66`) trên toàn bộ `3,418` patches VAL (`2,153` patches đa lớp, hoàn toàn không truy cập tập TEST) giữa hai ứng viên tiền đăng ký `blend = 0.50` và `blend = 1.00`:

| Ứng viên mặt nạ mềm trên VAL (`3,418` patches) | Overall mIoU | mDice | TUM | STR | LYM | NEC | Multi-class mIoU | ECE (thấp = tốt) | Entropy (Đa lớp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `q_s1_sgdap` (Giai đoạn 1, `blend = 0.00`) | 74.13% | 85.07% | 80.14% | 74.72% | 67.86% | 73.81% | 63.32% | 0.0751 | 0.2244 |
| `q_r2` (`blend = 0.50`: 50% `p_ens4` + 50% `q_s1`) | 75.39% (+1.26%) | 85.89% | 81.27% | 75.92% | **68.82%** | 75.54% | 65.00% (+1.68%) | 0.0527 | 0.2847 |
| **`p_ens4_uncal_ygated` (`blend = 1.00`, y-gated)** | **76.03% (+1.90%)** | **86.30%** | **82.19%** | **76.69%** | 68.71% | **76.54%** | **65.89% (+2.57%)** | **0.0489 (-35%)** | **0.2883** |

Kết quả trên VAL vượt rõ ràng ngưỡng Go/No-Go `>= 75.50%` và `LYM >= 67.86%`, đồng thời chỉ ra việc pha loãng 50% về Giai đoạn 1 làm suy giảm `-0.64%` mIoU (`75.39%` so với `76.03%`). Lưu ý minh bạch trong tệp JSON `se374_g7_diag_r2_val`: cờ `"go_decision": false` được script đánh giá tự động trên ứng viên mặc định `blend = 0.50` (`0.7539 < 0.7550`), trong khi ứng viên `candidates["p_ens4_uncal_ygated"]` (`blend = 1.00`) đạt `0.7603 >= 0.7550` và được khóa làm mục tiêu chưng cất chính thức cho Gate-7.

### 5.3 Tiến Hóa 4 Họ Kiến Trúc: Round-1 (Gate-6) vs Round-2 Track A (Gate-7)

Bảng dưới đối chiếu từng họ kiến trúc độc lập giữa Round-1 (Gate-6, đánh giá trên L4) và Round-2 Track A (Gate-7, đánh giá trên G4 Blackwell) theo đúng giao thức `02c` trên tập TEST `4,986` patches (`23` bệnh nhân):

| Họ kiến trúc | Round-1 Arm (Gate-6) | R1 TEST mIoU | Round-2 Track A Arm (Gate-7) | R2 Best EMA VAL | R2 No-CGM TEST | R2 Matched TEST mIoU | R2 Per-Class IoU (TUM / STR / LYM / NEC) | Tăng R2 vs R1 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| Classic ASPP CNN | `arm1_deeplab_cleankd` (R101) | 71.11% | `arm8_deeplab_r101_r2` (R101, Ep 2) | 68.74% | 70.47% | **70.96%** (CGM) | 80.59 / 74.39 / 61.84 / 67.02 | -0.15% |
| Hierarchical ViT | `arm2_segformer_oeem` (MiT-B3) | 71.84% | `arm6_segformer_b5_r2` (MiT-B5, Ep 5) | 70.13% | 72.04% | **72.04%** (`no_cgm`; `72.11%` CGM) | 81.55 / 75.40 / 61.69 / 69.51 | **+0.20%** |
| Modern ConvNet | `arm3_convnext_selftrain` (Base) | 70.73% | `arm7_convnext_large_r2` (Large, Ep 5) | 69.49% | 71.80% | **71.80%** (`no_cgm`; `71.84%` CGM) | 80.46 / 74.89 / **62.21** / 69.64 | **+1.07%** |
| SSL Vision Transformer | `arm4_dinov2_adapter` (ViT-B/14, 224px) | 72.36% | **`arm5_dinov2_large_r2` (ViT-L/14, 336px, Ep 10)** | **70.95%** | **72.53%** | **72.80%** (4-DoF CGM) | **82.73 / 76.17** / 61.31 / **70.98** | **+0.44%** |

*(Lưu ý số liệu từng lớp trong bảng trên trích nguyên bản từ nhánh `no_cgm` được VAL chọn cho `arm6` và `arm7` trong `STAGE2_TRAINING_METRICS.json`; ở chế độ `cgm_val_calibrated`, `arm6` đạt `81.58 / 75.43 / 61.77 / 69.68` và `arm7` đạt `80.47 / 74.90 / 62.23 / 69.76`.)*

### 5.4 Bảng Tổng Hợp Kết Quả Chung Kết Gate-7 Track A (Held-Out TEST)

Toàn bộ kết quả trên `4,986` patches (`23` bệnh nhân độc lập) của chiến dịch `se374_g7_ensemble_tracka` được tổng hợp trong bảng dưới đây và minh họa trực quan ở **Hình 1** (so sánh mIoU kèm khoảng tin cậy 95% Patient Bootstrap CI) và **Hình 2** (tiến hóa IoU theo từng lớp mô học `TUM`, `STR`, `LYM`, `NEC`).

| Phương pháp / Giao thức (G4 Blackwell trừ khi ghi L4) | Vai trò đánh giá | TEST mIoU (raw) | TEST mDice | 95% Patient Boot CI | TUM / STR / LYM / NEC IoU (%) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Author Official Pure (`1x`, `cgm_author_mul0`) | Mốc tác giả gốc | 69.48% (`0.6948`) | 81.79% | `[63.63, 73.59]` | 78.44 / 73.33 / 60.06 / 66.09 |
| Author Official Pure (`MS-TTA + CGM`) | Mốc tác giả tăng cường | 70.31% (`0.7031`) | 82.36% | `[64.57, 74.32]` | (L4: `70.30%`) |
| Author Step-0 Upgraded Anchor (`arm_b_full` + `MS-TTA+CGM`) | Mốc đối chứng chính (`m=5`) | 70.43% (`0.7043`) | 82.45% | `[64.73, 74.47]` | 79.46 / 74.13 / 60.55 / 67.60 |
| Gate-6 Round-1 Primary 4-Arm (L4, Uniform + 4-DoF CGM) | Mốc Round-1 (`se374_s2v2_ensemble`) | 72.47% (`0.724681`) | 83.84% | `[66.96, 76.43]` | 81.55 / 75.51 / **62.34** / 70.47 |
| **Gate-7 Track A Primary 4-Arm (`arm8+arm6+arm7+arm5`, Uniform, `no_cgm`)** | **Mô hình tổ hợp chính** | **72.80%** (`0.727955`) | **84.05%** | **`[67.24, 76.79]`** | 82.14 / 75.94 / **62.31** / 70.79 |
| **Gate-7 Track A Single Champion (`arm5_dinov2_large_r2`, `ms_3_hflip` + 4-DoF CGM)** | **Đơn mô hình tiền đăng ký** | **72.80%** (`0.727997`) | **84.02%** | **`[67.16, 76.86]`** | **82.73 / 76.17** / 61.31 / **70.98** |
| Gate-7 Track A VAL-Selected CNN+ViT Pair (`arm8+arm5`, `no_cgm`) | Cặp tiền đăng ký trên VAL | 72.68% (`0.726778`) | 83.96% | `[66.96, 76.70]` | 82.35 / 75.92 / 62.04 / 70.41 |
| Gate-7 Track A Best Triplet (`arm8+arm7+arm5`, `no_cgm`) | Thứ cấp / Ablation | 72.77% | 84.03% | - | 82.06 / 75.87 / 62.40 / 70.74 |
| Gate-7 Track A Highest Pair (`arm7+arm5`, `no_cgm`) | Thứ cấp / Thăm dò | 72.79% (`0.727929`) | 84.04% | `[67.25, 76.78]` | 82.08 / 75.94 / 62.12 / 71.03 |
| Gate-7 Track A Equal 4-Arm (`cgm_fixed_0_10`) | Thứ cấp / Độ nhạy CGM | 72.81% (`0.728143`) | 84.06% | `[67.26, 76.81]` | 82.15 / 75.95 / 62.34 / 70.82 |
| Gate-7 Track A Simplex-CV 4-Arm (`[0.28,0.08,0.05,0.59]`, `no_cgm` / `cgm_fixed_0_10`) | Thứ cấp / Thăm dò | 72.87% / **72.90%** (`0.728989`) | 84.08% / **84.10%** | `[67.31, 76.85]` | 82.56 / 76.16 / 61.99 / 70.89 |

![Hình 1: Đối chiếu 4-Class Pooled TEST mIoU kèm khoảng tin cậy 95% Patient Bootstrap CI trên 23 bệnh nhân độc lập (4,986 patches) giữa các mốc tác giả, Gate-6 Round-1 và Gate-7 Track A.](figures/fig1_gate7_test_miou_ci_bw.png)

![Hình 2: Tiến hóa IoU từng phân lớp mô học (TUM, STR, LYM, NEC) và Pooled 4-Class mIoU từ mốc Author Pure 1x qua Step-0 Author, Gate-6 Round-1 đến Gate-7 Arm 5 và Primary 4-Arm.](figures/fig2_gate7_per_class_evolution_bw.png)

### 5.5 Kiểm Định Ý Nghĩa Thống Kê Holm-Bonferroni & Độ Đa Dạng Tổ Hợp

Toàn bộ giả thuyết chính được kiểm định trên `23` bệnh nhân độc lập bằng phép thử hoán vị ghép cặp theo bệnh nhân (`10,000` lần hoán vị cho pooled Delta-mIoU) và kiểm định dấu hạng Wilcoxon (cho per-patient macro-mIoU), kèm hiệu chỉnh đa kiểm định **Holm-Bonferroni**:

| Cặp so sánh | Chênh lệch pooled mIoU | `perm_p_holm` (pooled) | `wilcoxon_p_holm` (bệnh nhân) | Cohen's `d_z` | Thắng/Hòa/Thua (23 BN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary 4-Arm (`no_cgm`) vs Step-0 Author (`70.43%`)** [`m=5`*] | **+2.36%** | **0.0005** | 0.9268 | -0.1172 | 16W / 0T / 7L |
| **Primary 4-Arm (`no_cgm`) vs Pure Author `1x` (`69.48%`)** [`m=4`] | **+3.31%** | **0.0004** | **0.0087** | +0.3387 | 19W / 0T / 4L |
| Primary 4-Arm (`no_cgm`) vs Pure Author `MS-TTA` (`70.31%`) [`m=4`] | +2.49% | **0.0004** | 0.1717 | +0.0688 | 16W / 0T / 7L |
| **Single `arm5` (+4-DoF CGM) vs Step-0 Author (`70.43%`)** [`m=5`*] | **+2.37%** | **0.0048** | **0.0302** | **+0.5587** | **18W / 0T / 5L** |
| **Single `arm5` (+4-DoF CGM) vs Pure Author `1x` (`69.48%`)** [`m=4`] | **+3.32%** | **0.0004** | **0.000067 (`6.7e-5`)** | **+0.9007** | **20W / 0T / 3L** |
| **Single `arm5` (+4-DoF CGM) vs Pure Author `MS-TTA` (`70.31%`)** [`m=4`] | **+2.49%** | **0.0019** | **0.000859** | **+0.7809** | **20W / 0T / 3L** |
| Primary 4-Arm (`no_cgm`) vs Single `arm5` (+4-DoF CGM) [`m=5`] | -0.00% (`-0.004%`) | 0.9923 | 0.3208 | -0.6034 | 9W / 0T / 14L |
| Chẩn đoán tổ hợp thuần: 4-Arm (`no_cgm` `72.80%`) vs `arm5` (`no_cgm` `72.53%`) | +0.27% | `p_raw = 0.4654` | `p_raw = 0.0698` | +0.1858 | 14W / 0T / 9L |

*\*Ghi chú hiệu chỉnh Holm `m=5`: Khi `uniform` được chọn làm cấu hình chính, hai hàng `primary_4arm_vs_author` và `equal_4arm_vs_author` trùng nhau trong họ `m=5`, làm phép hiệu chỉnh Holm trở nên chặt và bảo thủ hơn.*

- **Độ đa dạng và giới hạn trên (Diversity & Oracles trên TEST):** Tỷ lệ bất đồng theo cặp giữa 4 mô hình Gate-7 dao động từ `4.19%` (`arm6 - arm5`, $\kappa = 0.9317$) đến `6.38%` (`arm8 - arm5`, $\kappa = 0.8959$). **Pixel Oracle Accuracy = `90.60%`** và **Patch Oracle Upper Bound = `75.56%` mIoU** (tăng nhẹ so với Gate-6 lần lượt là `90.35%` và `75.35%`).
- **Nhận định khoa học về hiệu quả chưng cất Vòng 2:** Ở Vòng 2, đơn mô hình `arm5_dinov2_large_r2` (`72.80%` với 4-DoF CGM) đã hấp thụ thành công gần trọn tri thức của tổ hợp 4 họ kiến trúc Vòng 1, đạt hiệu năng ngang bằng Primary 4-Arm (`delta = -0.004%`, `perm_p_holm = 0.9923`) chỉ với khoảng 1/4 chi phí suy luận và thắng thuyết phục ở cả hai phép thử cấp bệnh nhân. Trong khi đó, Primary 4-Arm vẫn bảo toàn lợi thế quan trọng **`+1.00%` IoU lớp tế bào miễn dịch `LYM`** (`62.31%` so với `61.31%`) và **`+0.03%` mDice** (`84.05%` so với `84.02%`).

### 5.6 Kiểm Chứng Đa Hạt Giống (Multi-Seed Reproducibility)

Để kiểm chứng độ ổn định trước dao động khởi tạo ngẫu nhiên và thứ tự nạp batch, hệ thống đã hoàn thành đầy đủ bộ 3 hạt giống trên mô hình chủ lực Vòng 1 (`arm4_dinov2_adapter`, giao thức `02b` `ms_tta_cgm`) và đang tiếp tục mở rộng kiểm chứng 3 hạt giống cho mô hình chủ lực Vòng 2 (`arm5_dinov2_large_r2`):

| Mô hình / Chiến dịch | Hạt giống (Seed) | Epoch & Trọng số chọn | Best VAL mIoU | TEST `ms_tta` | TEST `ms_tta_cgm` | mDice | Trạng thái |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `arm4_dinov2_adapter` (R1) | `seed = 42` | Ep 3 **EMA** | 69.41% | 71.74% | **72.20%** | 83.61% | Đã xác minh (`SHA256` OK) |
| `arm4_dinov2_adapter` (R1) | `seed = 123` | Ep 4 **EMA** | 69.38% | 71.41% | **71.98%** | 83.48% | Đã xác minh (`SHA256` OK) |
| `arm4_dinov2_adapter` (R1) | `seed = 456` | Ep 5 **RAW** | 69.45% | 70.55% | **70.91%** | 82.75% | Đã xác minh (`SHA256` OK) |
| **Tổng hợp `arm4` (3 seeds)** | **42 / 123 / 456** | **2 EMA + 1 RAW** | **69.41% +/- 0.04%** | **71.23% +/- 0.61%** | **71.70% +/- 0.69%** (`ddof=1`) | **83.28%** | **2 EMA đạt `72.09% +/- 0.16%`** |
| `arm5_dinov2_large_r2` (R2) | `seed = 42` | Ep 10 **EMA-only** | 70.95% | 72.53% (`02c` `no_cgm`) | **72.80%** (`02c` 4-DoF CGM) | 84.02% | Đã xác minh (`SHA256` OK) |
| `arm5_dinov2_large_r2` (R2) | `seed = 123` | **EMA-only** | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | Đang huấn luyện trực tiếp trên L4 (`colab_account`) |
| `arm5_dinov2_large_r2` (R2) | `seed = 456` | **EMA-only** | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | Đang xếp hàng chờ GPU (`colab_account`) |
| **Tổng hợp `arm5` (3 seeds)** | **42 / 123 / 456** | **3/3 EMA-only** | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | `[PENDING LIVE GPU RUN]` | Chờ hoàn tất `seed=123` và `seed=456` |

---

## 6. Phân Tích Lỗi Lâm Sàng Cấp Bệnh Nhân & Chiến Dịch Kiểm Chứng GPU Đang Chạy

### 6.1 Giải Phẫu Nghịch Lý Thống Kê Cấp Bệnh Nhân (`GT = 0` Micro-FP & Tác Động Của CGM)

Từ bảng thống kê Mục 5.5 xuất hiện một nghịch lý đáng chú ý: `Primary 4-Arm (no_cgm)` vượt `Step-0 Author` tới `+2.36%` trên pooled mIoU với độ ý nghĩa cực cao (`perm_p_holm = 0.0005`), nhưng kiểm định Wilcoxon cấp bệnh nhân lại cho `wilcoxon_p_holm = 0.9268` và `Cohen's d_z = -0.1172`; tương tự khi so với `arm5 (+4-DoF CGM)` cho `d_z = -0.6034` (`9W/14L`).

Bằng việc lưu trữ chính xác ma trận giao/hợp `23 x 4` (`inter_p`, `union_p`) trong `checkpoints/vps_evidence/se374_g7_ensemble_tracka/artifacts/per_patient_class_ious.json` (`SHA256 = d180876c1dc75635`), chúng tôi đã chứng minh trọn vẹn cơ chế toán học đứng sau nghịch lý này (minh họa trực quan tại **Hình 3**):

- **Cơ chế phân mẫu active-union:** Phép thử hoán vị đánh giá pooled global Delta-mIoU (miễn nhiễm với lượng nhỏ điểm ảnh nhiễu trên một lớp vắng mặt). Ngược lại, thước đo macro-mIoU cấp bệnh nhân chỉ tính trung bình trên các lớp có `union_p > 0`. Kiểm toán trực tiếp trên toàn bộ `4,986` mặt nạ Ground Truth gốc (`data/BCSS-WSSS/test/mask/*.png`) xác nhận trong ma trận `23 x 4 = 92` ô có chính xác **`11` ô `GT == 0` thực tế trên đĩa** (`8` ca vắng `NEC` + `3` ca vắng `LYM` trên `10` bệnh nhân riêng biệt, do `TCGA-EW-A6SB` đồng thời vắng cả `LYM` lẫn `NEC`), tách thành hai tập con rạch ròi:
  1. **`8` ô `GT == 0` bất đối xứng mẫu số (`min(union_p) == 0` qua `10` mô hình cấp đầu / `12` bản ghi trong JSON: `7` ô `NEC` + `1` ô `LYM`):** Khi ít nhất một mô hình dự đoán sạch `0 px` (`None`) còn mô hình `no_cgm` phát sinh dương tính giả, IoU lớp đó rơi xuống `0.0` làm mẫu số macro-mIoU nhảy từ `3` lên `4`. Cụ thể, trong `8` ô bất đối xứng này, có **`7/8` ô chỉ phát sinh vi dương tính giả rất nhỏ từ `11` đến `556` điểm ảnh (`< 0.02%` tổng `union_p`, trong đó `11` đến `468` điểm ảnh trên `5` ca Author sạch tạo cú sụt giảm nhân tạo `primary_4arm (no_cgm)` so với `Step-0 Author` từ `-9.12%` đến `-17.10%`, gồm `4` ca sụt nặng `-12.07%` đến `-17.10%`)**, còn ô thứ 8 (`TCGA-EW-A1P4`) có `29,454` điểm ảnh (`0.261%` tổng `union_p`); riêng mức chênh lệch `-21.15%` (`TCGA-EW-A1OV`) là chênh lệch của `primary_4arm (no_cgm)` so với `arm5 (+4-DoF CGM)`, không phải so với `Step-0 Author`.
  2. **`3` ô `GT == 0` dương tính giả đồng loạt (`min(union_p) > 0`, cả `12/12` bản ghi mô hình đều dự đoán FP lớn nên giữ nguyên cùng mẫu số giữa các mô hình):** `TCGA-E2-A573` (`NEC`: `prim = 92,539 px`, `arm5 = 103,799 px`, `auth = 98,339 px`), `TCGA-E2-A159` (`LYM`: `prim = 653,710 px`, `arm5 = 916,141 px`, `auth = 503,723 px`), và `TCGA-EW-A6SB` (`LYM`: `prim = 255,472 px`, `arm5 = 321,734 px`, `auth = 193,150 px`).
- **Bảng giải phẫu `8` ô `GT == 0` bất đối xứng mẫu số (`min(union_p) == 0`, gồm `7` ca vắng `NEC` và `1` ca vắng `LYM`):**

| Bệnh nhân (Held-Out Test) | Lớp vắng mặt bất đối xứng (`GT = 0`) | Số pixel FP ở `Primary 4-Arm` (`no_cgm`) | Raw Delta vs Author | Delta trừ `8` ô bất đối xứng (trừ cả `11` ô) | Hành vi của `arm5` (`+4-DoF CGM`, `t_NEC = 0.65`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `TCGA-E2-A1LS` | `NEC` | 468 px (`0.0093%`) | -17.10% | **+1.65%** (thắng 3/3 lớp thật) | Xóa sạch về `0 px` (`+1.84%` vs Author) |
| `TCGA-E2-A158` | `NEC` | 202 px (`0.0040%`) | -15.16% | **+2.08%** (thắng 3/3 lớp thật) | Xóa sạch về `0 px` (`+1.56%` vs Author) |
| `TCGA-EW-A3U0` | `LYM` | 424 px (`0.01385%`) | -12.07% | **+1.36%** (`TUM +6.73%`) | Còn `2,048 px` FP `LYM` (`t_LYM = 0.05`, `-9.37%`) |
| `TCGA-D8-A27H` | `NEC` | 425 px (`0.0111%`) | -16.89% | -0.75% | Xóa sạch về `0 px` (`-0.65%` vs Author) |
| `TCGA-E2-A14R` | `NEC` | 11 px (`0.0003%`) | -9.12% | **+5.31%** | Xóa sạch về `0 px` (`+6.06%` vs Author) |
| `TCGA-EW-A1OV` | `NEC` | 52 px (`0.00019%`) | +0.88% | **+1.17%** | Xóa sạch về `0 px` (`+22.03%` vs Author) |
| `TCGA-EW-A6SB`* | `NEC` (và `LYM` FP đồng loạt) | 556 px (`0.00588%`) | +1.17% | **+1.56%** (**+2.34%** khi trừ cả 11 ô) | Xóa sạch `NEC` về `0 px` (`+12.16%` vs Author); còn `321,734 px` FP `LYM` |
| `TCGA-EW-A1P4` | `NEC` | 29,454 px (`0.261%`) | +4.42% | **+5.90%** | Xóa sạch về `0 px` (`+21.57%` vs Author) |

*\*Ghi chú bảng: Do `TCGA-EW-A6SB` đồng thời có `GT == 0` ở cả `NEC` (bất đối xứng mẫu số) lẫn `LYM` (FP đồng loạt `255,472 px` / `321,734 px` / `193,150 px`), mức chênh lệch của bệnh nhân này là `+1.56%` khi chỉ loại 8 ô bất đối xứng và tăng lên `+2.34%` khi loại toàn bộ 11 ô `GT == 0` thực tế. Tổng hợp trên cả 11 ô thực tế, `arm5 (+4-DoF CGM, t_NEC = 0.65)` xóa sạch **`7/8` ca `NEC`-absent thực tế (`7/7` ca bất đối xứng mẫu số; còn FP `103,799 px` ở `TCGA-E2-A573`)** và vẫn còn FP `LYM` trên **`3/3` ca `LYM`-absent thực tế** (`EW-A3U0: 2,048 px`, `EW-A6SB: 321,734 px`, `E2-A159: 916,141 px`; `1/1` trong tập bất đối xứng).*

![Hình 3: Phân tích độ nhạy cấp bệnh nhân trên 23 bệnh nhân Held-Out Test phân tách rõ 8 ô GT = 0 bất đối xứng mẫu số trong tổng số 11 ô GT = 0 thực tế trên đĩa ở Primary 4-Arm (Panel A) và khả năng xóa sạch 7/8 ca NEC-absent thực tế (7/7 ca bất đối xứng, còn FP trên TCGA-E2-A573) của Arm 5 tích hợp 4-DoF CGM (Panel B).](figures/fig3_patient_gt0_sensitivity_bw.png)

- **Kiểm chứng độ nhạy thăm dò (Exploratory GT-Active Sensitivity - công bố cả hai mốc):**
  - **Loại trừ `8` ô `GT == 0` bất đối xứng mẫu số (`min(union_p) == 0`):** `primary_4arm vs Step-0 Author` đảo chiều mạnh sang **`wilcoxon_p = 3.3e-5` (`3.27e-5`), `Cohen's d_z = +0.9021`, `20W/0T/3L`** (`TCGA-EW-A6SB = +1.56%`), và `primary_4arm vs arm5` trở nên cân bằng (`wilcoxon_p = 0.3765`, `d_z = +0.0814`, `13W/0T/10L`).
  - **Loại trừ toàn bộ `11` ô `GT == 0` thực tế trên đĩa (`4,986` mặt nạ gốc):** `primary_4arm vs Step-0 Author` càng khẳng định mạnh hơn với **`wilcoxon_p = 2.6e-5` (`2.62e-5`), `Cohen's d_z = +0.9383`, `20W/0T/3L`** (`TCGA-EW-A6SB = +2.34%`), và `primary_4arm vs arm5` tiếp tục cân bằng (`wilcoxon_p = 0.3294`, `d_z = +0.1011`, `13W/0T/10L`).
- **Ý nghĩa lâm sàng:** Trên thực tế giải phẫu bệnh, việc kích hoạt **Class-Gated Masking (CGM)** ở pha triển khai đóng vai trò bảo vệ sống còn, ngăn các đốm dương tính giả vi mô trên mô hoại tử vắng mặt làm sai lệch ước lượng tỷ lệ u sống $\text{VTP}_{\text{RCB}}$.

### 6.2 Ba Chiến Dịch Thực Nghiệm GPU Bổ Sung Đang Chạy Trực Tiếp Trên `vps-colab` (`colab_account`)

Nhằm hoàn thiện toàn diện các trụ cột phương sai đa hạt giống, kiểm chứng trực quan không chọn lọc toàn tập 4,986 ảnh và kiểm định lâm sàng chéo cơ quan, 3 chiến dịch GPU đang được điều phối trực tiếp trên tài khoản `colab_account` (`vps-colab`). Tuân thủ tuyệt đối nguyên tắc không bịa đặt số liệu, các hạng mục chưa hoàn tất được trình bày minh bạch với trạng thái **`[PENDING LIVE GPU RUN]`**:

#### 6.2.1 Chiến Dịch Kiểm Chứng Trực Quan Toàn Tập (`02d_Stage2_Visual_Overlay_Audit_G4.ipynb`, `src/stage2_v2/visual_audit.py`)

| Hạng mục kiểm định trực quan & hình học (`n = 4,986` patches TEST) | Tiêu chí xác minh bắt buộc | Kết quả đo đạc thực tế |
| :--- | :--- | :--- |
| 6 khẳng định toàn vẹn fail-fast (Mode `'P'` palette, SHA-256, `0` GT leakage vào `BACK`) | 6/6 PASS (`verify_zero_gt_leakage_glass_guard`) | `[PENDING LIVE GPU RUN]` (Job C) |
| Boundary F1@2px (Primary 4-Arm & Arm 5 vs Author Anchor) | Đo trực tiếp trên toàn bộ biên lớp mô 4,986 patches | `[PENDING LIVE GPU RUN]` (Job C) |
| Độ lệch tâm khối u (TUM Centroid Offset, pixel / um) | Đo sai số định vị hình học vùng biểu mô u | `[PENDING LIVE GPU RUN]` (Job C) |
| Ma trận bất biến biến đổi `raw_1x` Flip/Rot90 & dịch chuyển nội vi `(dx, dy)` | Xác nhận 0 lỗi xoay/lật kênh trên 4 nhánh Gate-7 | `[PENDING LIVE GPU RUN]` (Job C) |
| 11 bảng ảnh đối chiếu 300-DPI (01-08 Contact Sheets, 09-10 Whole-ROI Stitches, 11 Alignment) | Pixel nearest-neighbor sắc nét, 0 chọn lọc thiên lệch | `[PENDING LIVE GPU RUN]` (Job C) |

#### 6.2.2 Chiến Dịch Đánh Giá Lâm Sàng, Hiệu Năng GPU & Chéo Cơ Quan (`04b_Stage2_Gate7_Clinical_LUAD_Bench_G4.ipynb`, `src/stage2_v2/gate7_clinical_bench.py`)

| Hạng mục đánh giá lâm sàng & triển khai | Giao thức đo lường | Kết quả đo đạc thực tế |
| :--- | :--- | :--- |
| Tương quan dấu ấn sinh học VTP / TSR cấp bệnh nhân (Pearson $r$, Spearman $\rho$, Bland-Altman) | Đối chiếu GT bác sĩ vs Gate-7 trên 23 bệnh nhân TEST | `[PENDING LIVE GPU RUN]` (Job D) |
| Phân tầng nguy cơ CAP (`TSR <= 50%`, Cohen's $\kappa$) & đường cong sống còn Kaplan-Meier / Log-rank | Sử dụng `src/clinical/survival_analysis.py` | `[PENDING LIVE GPU RUN]` (Job D) |
| Biên Pareto hiệu năng GPU (`1x` forward latency ms/patch, VRAM đỉnh MB vs TEST TTA mIoU) | So sánh Author S2, Arm 5-8 và Primary 4-Arm | `[PENDING LIVE GPU RUN]` (Job D) |
| Tổng quát hóa chéo cơ quan Zero-Shot trên **LUAD-HistoSeg** (mô phổi, ánh xạ `LUAD_TO_BCSS_LUT`) | Đánh giá trực tiếp không tinh chỉnh trên tập Test LUAD | `[PENDING LIVE GPU RUN]` (Job D) |

---

## 7. Kiến Trúc Sản Phẩm Phần Mềm, Trạm Lâm Sàng & Hạ Tầng Đa GPU

### 7.1 Kiến Trúc Clean Architecture & Đồng Bộ Mã Nguồn

Kho mã nguồn `src/` được phân tách rõ ràng theo nguyên lý trách nhiệm đơn nhất (Single Responsibility Principle) thành 11 gói dịch vụ độc lập:

1. **`src/config/`:** Lưu trữ cấu hình phân chia 151 bệnh nhân bất biến (`bcss_patient_splits.json`) và cấu hình thực nghiệm Stage-1 (`stage1_experiments.py`).
2. **`src/datasets/`:** Dịch vụ nạp dữ liệu phòng vệ đa tầng (`bcss_dataset.py`), quản lý chống rò rỉ bệnh nhân (`patient_split.py`), và bộ lọc quang học (`otsu_filter.py`).
3. **`src/models/`:** Kiến trúc nền tảng `WSSSTissueStage1NetCAM`, `DeepLabV3Plus`, `MultiMethodWSSSEngineV2` (`M1` đến `M7`), tách sắc tố Beer-Lambert `StainDeconvLayer` (`stain_deconv.py`), tinh chỉnh ái lực sắc tố `SG-DAP` (`sg_dap.py`), và mô-đun thích ứng `MedSAM_WSSS_Adapter` (`sam_wsss_adapter.py`).
4. **`src/stage2_v2/`:** Động cơ nghiên cứu và đánh giá Gate-7 Track A gồm cấu hình bất biến (`config.py`), 4 kiến trúc Dual-Head kèm bộ ánh xạ tương thích chéo `transformers` v4/v5 (`models.py`: `remap_hf_state_dict_keys`), các hàm mất mát mềm (`losses.py`), bộ nhớ đĩa `memmap` chống tràn RAM (`memmap_dataset.py`), vòng huấn luyện EMA-only (`trainer.py`), đánh giá chuẩn hóa (`evaluator.py`), tổ hợp 5 chuyên gia (`ensemble.py`), kiểm chứng trực quan (`visual_audit.py`), và chuẩn đo lâm sàng + LUAD (`gate7_clinical_bench.py`).
5. **`src/losses_metrics/` & `src/clinical/`:** Ma trận nhầm lẫn chuẩn hóa, tính dấu ấn VTP/TSR, kiểm định thống kê Wilcoxon/Holm-Bonferroni, và phân tích sống còn Kaplan-Meier (`survival_analysis.py`).
6. **`src/training/`, `src/pipeline/` & `src/colab_pkg/`:** Script huấn luyện/đánh giá chuẩn hóa, điều phối chuỗi WSI đa độ phóng đại (`wsi_lowmag_highmag_pipeline.py`), và gói tự động hóa Colab (`downloader.py` với `resolve_split_dirs`, `model_loader.py`, `evaluator.py`, `visualizer.py`, `colab_runner.py`).
7. **`src/visualization/` & `src/utils/`:** Máy chủ Trạm Lâm Sàng FastAPI (`api_server.py`), bộ kết xuất kiểm chứng trực quan (`clinical_viewer.py`), giao diện ngoại tuyến (`static/index.html`), công cụ vẽ hình học thuật đơn sắc 300-DPI (`draw_gate7_monochrome_figures.py`), và bộ kiểm toán chất lượng tài liệu 4 tầng (`deliverable_auditor.py`).

Toàn bộ `63/63` tệp trong `src/` (`60` tệp Python + `3` tệp cấu hình/tài liệu/HTML, chuẩn hóa LF) được kiểm tra đồng bộ hóa SHA256 và cây cú pháp AST với gói phân phối Colab `drive_upload_ready/SE374_WSSS_Drive_Code/src/` và kho nén `.zip` kèm theo; bộ kiểm tra chuẩn mã nguồn `ruff check src tests colab_runner_local` đạt **0 lỗi**.

### 7.2 Trạm Làm Việc Lâm Sàng FastAPI, Báo Cáo PDF Chuẩn CAP & Docker

Trạm Làm Việc Giải Phẫu Bệnh Số (`src/visualization/api_server.py` và `src/visualization/static/index.html`) vận hành **100% ngoại tuyến** không phụ thuộc CDN bên ngoài, cho phép bác sĩ và kỹ sư kiểm chứng trực tiếp trên ảnh tiêu bản H&E thật:

| Endpoint | Phương thức | Chức năng chuyên môn & Hợp đồng dữ liệu |
| :--- | :---: | :--- |
| `/viewer` & `/` | `GET` | Giao diện HTML5 Canvas ngoại tuyến: 9 chế độ QA trực quan (`Blurred Dual QA`, `Match vs Error`, `Seg Mask`, `GT + AI Blend`, `Contours`, `Entropy`, `Stage-1 CAM`, `Spotlight`, `Doctor GT`), Split-Curtain (`S`), Magnifier Loupe (`L`), Pixel Inspector HUD, Scale Bar (`50 um, 0.25 MPP`) |
| `/docs` | `GET` | Tài liệu tương tác OpenAPI (Swagger UI) |
| `/api/v1/health` & `/health` | `GET` | Báo cáo trạng thái nạp trọng số thật (`gate7_available_arms`, `gate7_loaded_arms`, `stage2_loaded`), thiết bị GPU/CPU và danh sách chế độ suy luận |
| `/api/v1/samples`, `/api/v1/samples/{id}` | `GET` | Duyệt và lọc tập Test theo `patient_id`, hình thái mô, phân bố GT và dấu ấn sinh học chuẩn |
| `/api/v1/predict/patch` & `/segment` | `POST` | Suy luận trên ảnh H&E hỗ trợ **7 chế độ Gate-7 Track A** (`gate7_arm5`, `gate7_arm5_tta`, `gate7_arm6`, `gate7_arm7`, `gate7_arm8`, `gate7_4arm`, `gate7_4arm_cgm`) cùng 11 chế độ cổ điển; fail-closed trả HTTP `503` khi `strict_weights=True` thiếu trọng số |
| `/api/v1/compare` & `/biomarkers` | `POST` | Đối chiếu song song đa mô hình và tính định lượng VTP, TSR, phân tầng nguy cơ CAP |
| `/api/v1/predict/wsi` & `/tile/{z}/{x}/{y}.png` | `POST` / `GET` | Ghép nối đa ô WSI Multi-Tile Stitch (tối đa 64 tiles, trộn biên cosine feathering) và phục vụ lát cắt DeepZoom |
| **`/api/v1/export/report`** | **`POST`** | **Xuất báo cáo kiểm định 300-DPI PNG, JSON, CSV hoặc Báo cáo Lâm sàng PDF 2 trang chuẩn CAP** (`BCSS-WSSS CLINICAL PATHOLOGY QA REPORT`, tự động ghi VTP/TSR, phân tầng CAP/UNITED theo ngưỡng 50% TSR, ảnh đối chiếu và khuyến cáo RUO độc lập, không rò rỉ nhãn học phần) |
| `/splits` | `GET` | Xuất công khai bảng chia 151 bệnh nhân phục vụ kiểm toán chống rò rỉ |

Hệ thống được đóng gói hoàn chỉnh bằng `Dockerfile` (nền `python:3.11-slim`, chạy dưới tài khoản không đặc quyền `appuser`, cấu hình biến `WSSS_GATE7_DIR=/app/checkpoints/vps_evidence` và `HEALTHCHECK` tự động gọi `/health` mỗi 30 giây) cùng `docker-compose.yml` gắn chỉ đọc (`ro`) hai thư mục `./checkpoints` và `./data`, giúp triển khai tái lập bằng một lệnh `docker compose up --build`.

### 7.3 Hệ Thống Điều Phối Đa GPU `colab_runner_local/` & Kiểm Thử Tự Động

Nhằm vận hành ổn định các chiến dịch huấn luyện và đánh giá kéo dài nhiều giờ trên Google Colab GPU mà không phụ thuộc đường truyền máy trạm cục bộ, dự án xây dựng bộ điều phối chuyên biệt `colab_runner_local/` chạy trên Linux VPS:

- **Chính sách ưu tiên tài nguyên tường minh:** Mọi lệnh truyền tường minh `--credentials ~/.config/gcloud/colab_adc.json` (tài khoản `colab_account`, không phụ thuộc ADC mặc định của hệ thống) và chuỗi GPU ưu tiên `--gpu H100 --gpu-fallback G4,A100,L4`.
- **Cơ chế bảo vệ P0 chống tắt nhầm phiên ngoại lai (`runtime_guard.py`):** Cô lập trạng thái theo tài khoản (`acct_<sha256_12>`), khóa cấp phát độc quyền bằng `flock`, chụp danh sách `protected_endpoints` bất biến theo nguyên tắc fail-closed trước mọi lệnh cấp phát (bảo vệ tuyệt đối các phiên ngoại lai như external_service A100), đăng ký danh sách trắng `owned_sessions`, và áp dụng quy tắc tắt phiên khớp kép (chỉ dừng khi cả `session_name` lẫn `endpoint` thuộc `owned_sessions` và không nằm trong `protected_endpoints`).
- **Toàn vẹn checkpoint và giám sát web:** Cơ chế `sync_remote_file_atomic` tải tệp sang vùng tạm, đối chiếu `sha256sum` khớp 100% trước khi thay thế nguyên tử; tự động resume tối đa 3 lần khi mất kết nối; đồng hồ cắt cứng `--max-minutes`; cảnh báo sớm fail-fast khi phát hiện `NaN` hoặc sụp đổ hiệu chuẩn; và bảng điều khiển web `dashboard.py` bảo mật băm PBKDF2 (`local_auth.py`) qua Cloudflare Quick Tunnel.
- **Phủ kiểm thử tự động đa nền tảng:** Hệ thống tích hợp **32 tệp kiểm thử (`30` module / `496` ca trong `tests/` + `2` module / `35` ca trong `colab_runner_local/`) với `531` test cases thu thập được**, kết hợp đường ống CI `.github/workflows/ci.yml`. Trên môi trường Windows PowerShell cục bộ và môi trường Linux ARM64 thực tế trên `vps-colab` (`35/35` bài kiểm thử `test_vps_colab_runner.py` và `test_dashboard.py` đạt 100%, `0 skipped`, `0 failed`), toàn bộ các kiểm định chống rò rỉ bệnh nhân, API, trực quan, bảo vệ P0 và chuẩn mã nguồn đều vượt qua.

---

## 8. Tính Tái Lập Mật Mã, Minh Bạch Sự Cố & Giới Hạn Mở

### 8.1 Kiểm Toán Mật Mã SHA256 (123/123 Tệp Cục Bộ + Trọng Số VPS)

Toàn bộ `123` tệp minh chứng cục bộ (gồm nhật ký huấn luyện `.log`, báo cáo phiên `.md`, số liệu `.json`/`.csv`, biểu đồ `.png` và `6` tệp trọng số `.pth` chủ lực gồm Stage-1 Champion `se374_arm_b_full` cùng 4 nhánh Gate-7 Track A `arm5` đến `arm8`) được lập chỉ mục tại `checkpoints/vps_evidence/SHA256SUMS.txt` và xác minh toàn vẹn bit-exact **`123/123 OK` (`BAD = 0`, `MISSING = 0`)**. Ngoài ra, toàn bộ `4/4` tệp trọng số Gate-6 Round-1 (`arm1` đến `arm4`) lưu trên `vps-colab` đã được đối chiếu khớp đủ `64/64` ký tự hex SHA256.

| Giai đoạn / Phiên chạy | Tệp minh chứng trọng yếu | Mã băm SHA256 (16 ký tự đầu) |
| :--- | :--- | :--- |
| Stage-1 Champion (`se374_arm_b_full`) | `stage1_retrained_best.pth` / `stage1_best_cls.pth` | `a182d8890c09d7b8` / `e1a88293aff0a0e3` |
| Gate-6 Step-0 (`se374_s2v2_step0`) | `STAGE2_TRAINING_METRICS.json` | `b0b319d6f7ab0537` |
| Gate-6 Round-1 `arm1` đến `arm4` (`vps-colab`) | `arm1` / `arm2` / `arm3` / `arm4` `stage2_retrained_best.pth` | `69b569e6080370f3` / `e32e2a618c994f57` / `f706c0f52cd4c49d` / `bda6c2f048d1ed13` |
| Gate-6 Round-1 4-Arm (`se374_s2v2_ensemble`) | `STAGE2_TRAINING_METRICS.json` | `9e43aa5f27a623ca` |
| Multi-seed Arm 4 (`s123` / `s456`) | `STAGE2_TRAINING_METRICS.json` (`s123` / `s456`) | `d7be7060c1445bea` / `0160c4639a6924c6` |
| Gate-7 Step-0 (`se374_g7_diag_r2_val`) | `STAGE2_TRAINING_METRICS.json` | `21c08b3996507d66` |
| Gate-7 Arm 5 (`se374_g7_arm5_dinov2l`) | `stage2_retrained_best.pth` / `STAGE2_TRAINING_METRICS.json` | `33f4d623e8825471` / `2a5ece67aac25723` |
| Gate-7 Arm 6 (`se374_g7_arm6_segformerb5`) | `stage2_retrained_best.pth` / `STAGE2_TRAINING_METRICS.json` | `3055643db074a2df` / `87dcec892d0af5ef` |
| Gate-7 Arm 7 (`se374_g7_arm7_convnextl`) | `stage2_retrained_best.pth` / `STAGE2_TRAINING_METRICS.json` | `c64b97c731c71c97` / `362a34faf2b437cc` |
| Gate-7 Arm 8 (`se374_g7_arm8_deeplab`) | `stage2_retrained_best.pth` / `STAGE2_TRAINING_METRICS.json` | `9fd6e121a9714756` / `fe5dc37f53c255b8` |
| **Gate-7 Track A Ensemble (`se374_g7_ensemble_tracka`)** | `STAGE2_TRAINING_METRICS.json` / `per_patient_class_ious.json` | `c6140c3bb266129a` / `d180876c1dc75635` |

*Ghi chú phần cứng và tái lập:* Giao thức đánh giá Author Anchors ghi nhận sai khác làm tròn phần cứng lành tính `0.01%` giữa NVIDIA L4 (`69.47%` / `70.30%` / `70.42%`) và NVIDIA G4 Blackwell (`69.48%` / `70.31%` / `70.43%`) trên cùng mã nguồn và trọng số. Trong Gate-7, `arm5`, `arm7`, `arm8` và `02c` chạy trên G4 Blackwell, còn `arm6` tự động chuyển sang L4 nên sai khác làm tròn `bf16` mức $10^{-6}$ tạo hai mã băm `r2_manifest_sha256` khác nhau (`90ab6933495b` trên G4 so với `2b469c3d906e` trên L4) dù cùng `23,422` patches, cùng `blend = 1.0` và cùng `argmax_disagree = 0.103707`.

### 8.2 Ghi Chú Minh Bạch Về Sự Cố Mất Dữ Liệu Ngày 2026-11-23 & Kiểm Toán Khôi Phục

Vì tính trung thực khoa học và quản trị kỹ thuật tuyệt đối, báo cáo ghi nhận công khai sự cố vận hành máy trạm Windows ngày **2026-11-23**: một lệnh dọn dẹp cục bộ bị lỗi thoát dấu ngoặc kép giữa PowerShell và `cmd` đã xóa dữ liệu trên phân vùng `C:\`, dẫn đến việc **mất vĩnh viễn tệp không được theo dõi trên git là `Lab1.ipynb`** (tệp bài tập thực hành cục bộ ban đầu, không nằm trong lịch sử remote GitHub và không thể tái tạo bản gốc).

Ngay sau sự cố, toàn bộ kho dự án nghiên cứu và sản phẩm chính đã trải qua quy trình phục hồi và kiểm toán toàn vẹn chặt chẽ:

- Khôi phục đầy đủ toàn bộ các mô-đun cốt lõi từ lịch sử git xác thực và đồng bộ lại 10 mô-đun nâng cấp (bao gồm `MultiMethodWSSSEngineV2`, tương thích chéo `transformers` v4/v5, sửa bảng màu `STR/LYM` và cổng `0/4` trong `wsi_lowmag_highmag_pipeline.py`, giao diện chọn chế độ Gate-7 trong `static/index.html`, và bộ kiểm chứng `02d_Stage2_Visual_Overlay_Audit_G4.ipynb`).
- Kiểm toán mật mã độc lập xác nhận **`123/123` tệp minh chứng tại `checkpoints/vps_evidence/SHA256SUMS.txt` nguyên vẹn 100% (`BAD = 0`, `MISSING = 0`)** cùng `4/4` tệp trọng số Gate-6 trên `vps-colab` khớp toàn phần `64/64` ký tự hex.
- Chạy kiểm thử hồi quy đa luồng trên dữ liệu bệnh nhân BCSS-WSSS thật (`len(train_loader) = 5,856` batches ở `batch_size = 4`) và môi trường Linux ARM64 độc lập, xác nhận không có hư hại đối với bất kỳ kết quả thực nghiệm hay thành phần sản phẩm chính nào của đồ án.

### 8.3 Giới Hạn Hiện Tại & Hướng Phát Triển Tương Lai

1. **Giới hạn khoảng cách so với `SIA-WSSS` (MICCAI 2025):** Ở cấu hình tiền đăng ký Track A không chồng lấp TCGA, hệ thống đạt `72.80%` mIoU (`72.90%` Secondary Simplex-CV), còn thấp hơn `-0.60%` (`-0.50%`) so với mức `73.40%` mIoU của `SIA-WSSS` (5-run average). Việc kết hợp cơ chế đồ thị kích hoạt - ức chế đồng bộ ngay từ Giai đoạn 1 là hướng nghiên cứu đầy hứa hẹn để vượt ngưỡng `73.5%` trong Track A.
2. **Hoàn tất 3 chiến dịch GPU đang thực thi:** Khi `seed=123/456` của `arm5_dinov2_large_r2`, kiểm chứng trực quan `02d` toàn tập `4,986` patches và đánh giá lâm sàng / chéo cơ quan LUAD-HistoSeg `04b` hoàn thành trên `vps-colab` (`colab_account`), các ô `[PENDING LIVE GPU RUN]` trong Chương 5 và Chương 6 sẽ được cập nhật trực tiếp từ nhật ký JSON có ký số SHA256.
3. **Mở rộng Track B có đối chứng và triển khai đa trung tâm:** Thiết lập nhánh thực nghiệm Track B tách biệt hoàn toàn (đánh giá công bằng các mô hình nền tảng giải phẫu bệnh như UNI/Virchow/Phikon-v2), kết hợp suy luận WSI toàn phần độ phóng đại thấp từ `LowMagWSS` để rút ngắn thời gian xử lý toàn slide gigapixel, và tổ chức đánh giá mù với hội đồng bác sĩ giải phẫu bệnh lâm sàng.

---

## 9. Tài Liệu Tham Khảo (References)

1. **Chu Han, Jiatai Lin, Jinhai Mai, et al.** "Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels." *Medical Image Analysis (MedIA)*, vol. 80, p. 102487, 2022. DOI: `10.1016/j.media.2022.102487` (`arXiv:2110.08048`).
2. **Mohamed Amgad, Habiba Elfandy, Hagar Hussein, et al.** "Structured crowdsourcing enables convolutional segmentation of histology images." *Bioinformatics*, vol. 35, no. 18, pp. 3461-3467, 2019. DOI: `10.1093/bioinformatics/btz083`.
3. **Lyndon Chan, Mahdi S. Hosseini, Corwyn Rowsell, Konstantinos N. Plataniotis, Savvas Damaskinos.** "HistoSegNet: Semantic Segmentation of Histological Tissue Type in Whole Slide Images." *ICCV*, pp. 10662-10671, 2019. DOI: `10.1109/ICCV.2019.01076`.
4. **Yude Wang, Jie Zhang, Meina Kan, Shiguang Shan, Xilin Chen.** "Self-Supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation (SEAM)." *CVPR*, pp. 12275-12284, 2020. DOI: `10.1109/CVPR42600.2020.01229`.
5. **Zijie Fang, Yang Chen, Yifeng Wang, Zhi Wang, Xiangyang Ji, Yongbing Zhang.** "Weakly-Supervised Semantic Segmentation for Histopathology Images Based on Dataset Synthesis and Feature Consistency Constraint (PistoSeg)." *AAAI*, vol. 37, no. 1, pp. 606-613, 2023. DOI: `10.1609/aaai.v37i1.25136`.
6. **Zijie Fang, Yifeng Wang, Peizhang Xie, Zhi Wang, Yongbing Zhang.** "HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization." *IEEE Transactions on Medical Imaging (TMI)*, vol. 44, no. 4, pp. 1765-1782, 2025. DOI: `10.1109/TMI.2024.3520129`.
7. **Siyang Feng, Jiale Chen, Zhenbing Liu, et al.** "Mining Gold from the Sand: Weakly Supervised Histological Tissue Segmentation with Activation Relocalization and Mutual Learning (ARML)." *MICCAI 2024*, Springer LNCS vol. 15008, pp. 414-423, 2024. DOI: `10.1007/978-3-031-72111-3_39`.
8. **Jiansong Fan, Yicheng Di, Jiayu Bao, Lihua Li, Xiang Pan.** "Synchronous Inhibition and Activation for Weakly Supervised Semantic Segmentation of Pathology Images." *MICCAI 2025*, Springer LNCS, 2025. DOI: `10.1007/978-3-032-05141-7_56`.
9. **Anh M. Vu, Khang P. Le, et al.** "DualProtoSeg: Simple and Efficient Design with Text- and Image-Guided Prototype Learning for Weakly Supervised Histopathology Image Segmentation." *arXiv preprint arXiv:2512.10314*, 2025.
10. **Khang Le, Ha Thach, Anh M. Vu, et al.** "ConStruct: Structural Distillation of Foundation Models for Prototype-Based Weakly Supervised Histopathology Segmentation." *Medical Imaging with Deep Learning (MIDL)*, `arXiv:2512.10316`, 2026.
11. **Dung Minh Do, Nhat-Thanh Huynh, Duc Minh Huynh, Doanh C. Bui, Khang Nguyen.** "Toward Efficient Weakly Supervised Semantic Segmentation Using Only Low-Magnification Histopathological Images." *2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)*, pp. 334-339, 2026. DOI: `10.1109/MAPR72750.2026.11685691`.
12. **Maxime Oquab, Timothee Darcet, Theo Moutakanni, et al.** "DINOv2: Learning Robust Visual Features without Supervision." *Transactions on Machine Learning Research (TMLR)*, 2024 (`arXiv:2304.07193`).
13. **Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo.** "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *NeurIPS*, vol. 34, pp. 12077-12090, 2021.
14. **Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.** "A ConvNet for the 2020s (ConvNeXt)." *CVPR*, pp. 11976-11986, 2022. DOI: `10.1109/CVPR52688.2022.01167`.
15. **Arnout C. Ruifrok, Dennis A. Johnston.** "Quantification of histochemical staining by color deconvolution." *Analytical and Quantitative Cytology and Histology*, vol. 23, no. 4, pp. 291-299, 2001.
16. **W. Fraser Symmans, Florentia Peintinger, Christos Hatzis, et al.** "Measurement of Residual Breast Cancer Burden to Predict Survival After Neoadjuvant Chemotherapy." *Journal of Clinical Oncology*, vol. 25, no. 28, pp. 4414-4422, 2007. DOI: `10.1200/JCO.2007.10.6823`.
17. **Gabi W. van Pelt, Sanne Kjaer-Frifeldt, J. Han J. M. van Krieken, et al.** "Scoring the tumor-stroma ratio in colon cancer: procedure and recommendations." *Virchows Archiv*, vol. 473, no. 4, pp. 405-412, 2018. DOI: `10.1007/s00428-018-2408-z`.
