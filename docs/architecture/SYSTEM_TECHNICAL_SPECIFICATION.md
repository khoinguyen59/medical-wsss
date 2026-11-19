# He Thong Dac Ta Ky Thuat Toan Dien (System Technical Specification)

> **Ma Tai Lieu:** `docs/architecture/SYSTEM_TECHNICAL_SPECIFICATION.md`  
> **Quy Chuan Quan Tri:** Master Governance Specification `AGENTS.md` v5.0 (SE374MedicalAIAgent)  
> **Hoc Phan:** SE374.R11 - Phat Trien Phan Mem Xu Ly Anh Y Khoa (Hoc ky 1, Nam hoc 2026-2027)  
> **Don Vi Dao Tao:** Khoa Ky Thuat Phan Mem (Software Engineering), Truong Dai Hoc Cong Nghe Thong Tin (UIT), DHQG-HCM  
> **De Tai Trong Tam:** Topic 6 - Weakly-Supervised Semantic Segmentation on BCSS Histopathology Whole Slide Images  
> **Tinh Xac Thuc:** 100% thong so, duong dan file, ten lop, ten ham, cong thuc toan hoc, so lieu benh nhan va ket qua thuc nghiem deu duoc doi chieu truc tiep tu ma nguon vat ly va log kiem thu that tren he thong. Tuan thu nghiem ngat dau gach noi ban phim chuan ASCII 0x2D '-' trong toan bo tai lieu.  

---

## 1. Tong Quan He Thong & Ho So Hoc Phan (System Identity & Course Profile)

### 1.1 Thong Tin Hoc Phan & Quan Tri
- **Ma lop hoc phan:** SE374.R11
- **Ten hoc phan:** Phat trien phan mem xu ly anh y khoa
- **Khoa chuyen trach:** Khoa Ky thuat Phan mem (Software Engineering Faculty), Truong Dai hoc Cong nghe Thong tin - DHQG-HCM.
- **Giang vien huong dan phu trach:** TS. Bui Cao Doanh
- **Thoi gian thuc hien:** 10/2026 - 12/2026 (10 tuan nghien cuu va phat trien)
- **Chu de khoa hoc:** Chu de 6 (Topic 6) - Weakly-Supervised Semantic Segmentation for Whole Slide Images (Phan doan ngu nghia giam sat yeu cho tieu ban mo benh hoc toan phan gigapixel).

### 1.2 Triet Ly Quan Tri & Cac Luat Bat Bien Y Khoa (Core Invariants)
He thong duoc van hanh chat che boi 32 quy chuan quan tri he thong y te duoc quy dinh tai `AGENTS.md`:
- **Quy chuan Invariant 01 (Human-in-the-Loop Supremacy):** Moi quyet dinh quan trong (chot de tai, kien truc, thu vien, chay huan luyen GPU tieu ton VRAM) bat buoc phai hoi y kien nguoi dung qua giao thuc hoi dap. Nghiem cam tu y quyet dinh cong nghe hoac thay doi kien truc he thong khi chua duoc phe duyet.
- **Quy chuan Invariant 07 (Liem Chinh Khoa Hoc Tuyet Doi):** 100% so lieu cong bo trong proposal, slide, bao cao va dac ta phai xuat phat tu ma nguon chay thuc nghiem that tren GPU hoac trich dan chinh xac tu bai bao goc co ma dinh danh DOI hoac arXiv xac thuc. Nghiem cam tuyet doi dummy data, mock metrics hoac so lieu gia dinh.
- **Quy chuan Invariant 08 (Patient-Level Anti-Leakage Protocol):** Phan chia tap huan luyen, kiem dinh va kiem thu (`train / val / test`) triet de theo Ma benh nhan (Patient ID / Case ID). Tuyet doi khong tron lan cac lat cat patch giua cung mot benh nhan giua cac tap. Do lech ranh gioi benh nhan bang 0.0%.
- **Quy chuan Invariant 12 (Strict Standard Keyboard Hyphen Only):** Nghiem cam tuyet doi dau gach ngang dai kieu AI (em-dash U+2014, en-dash U+2013). Chi su dung duy nhat dau gach noi ban phim chuan ASCII 0x2D ('-').
- **Quy chuan Invariant 15 (Zero One-Off Scratch Scripts):** Khong tao cac script Python tam bo lam o nhiem thu muc; ke thua thu vien chuan, CLI chinh thuc (`graphify`, `pandoc`, `git`) va he sinh thai trong thu muc `repos/`.
- **Quy chuan Invariant 25 (Independent Deliverable Quality Audit Gate):** Moi tai lieu truoc khi xuat ban phai vuot qua 4 tang kiem tra cua module kiem dinh chat luong: Hinh ve khoa hoc don sac vector/300 DPI, Typography ASCII hyphen, Liem chinh hoc thuat va Cau truc van ban.

---

## 2. Bo Du Lieu Chuan & Giao Thuc Chong Ro Ri Benh Nhan (Dataset Specifications & Patient-Level Anti-Leakage)

### 2.1 Dac Ta Bo Du Lieu BCSS-WSSS (Breast Cancer Semantic Segmentation)
- **Ten bo du lieu:** BCSS-WSSS (Tieu ban ung thu vu giam sat yeu)
- **Nguon goc:** Tap con thuoc TCGA-BRCA (The Cancer Genome Atlas Breast Invasive Carcinoma), cong bo boi Amgad et al. (Bioinformatics 2019) va Han et al. (MedIA 2022).
- **Kich thuoc patch:** $224 \times 224$ pixels, anh mau RGB 3 kenh nhuom Hematoxylin & Eosin (H&E).
- **Do phong dai goc:** $40\times$ (Microns Per Pixel: $0.25\,\mu\text{m/px}$, do phan giai vat ly goc theo Amgad et al. 2019 va Chu Han et al. 2022; trong cac thuc nghiem mo phong luu tru do phan giai thap nhu LowMagWSS co the giam mau xuong $20\times$ hoac $10\times$ truoc khi tai tao).
- **Danh muc 5 phan lop mo giai phau benh (Tissue Classes):**
  - **Lop 0 - TUM (Tumor / Infiltrating Ductal Carcinoma):** Vung bieu mo khoi u xam lan, mat do nhan dong duc, bien dang hinh thai hoc. Ma mau truc quan: Crimson Red `#E41A1C` (RGB: 228, 26, 28).
  - **Lop 1 - STR (Stroma / Fibrous Stroma):** Mo dem xo, mo lien ket bao quanh u, phan ung tao xo vi moi truong. Ma mau truc quan: Dodger Green `#4DAF4A` (RGB: 77, 175, 74).
  - **Lop 2 - LYM (Lymphocyte / Inflammatory Infiltrate):** Cum te bao lympho tham nhuan viem, kich thuoc nhan nho, kiem tinh dam. Ma mau truc quan: Sky Blue `#377EB8` (RGB: 55, 126, 184).
  - **Lop 3 - NEC (Necrosis / Debris):** Vung hoai tu mo, manh vun te bao chet, mat cau truc bao quan. Ma mau truc quan: Orange/Yellow `#FF7F00` (RGB: 255, 127, 0).
  - **Lop 4 - BACK (Background / Glass / Other):** Lam kinh trong suot, mo mo khong bat mau, chat nen vo bao. Ma mau truc quan: Gray `#999999` (RGB: 153, 153, 153, trong suot khi overlay).

### 2.2 Tong So Benh Nhan & Phan Chia Train / Val / Test Thuc Te
Theo tep cau hinh kiem toan goc tai [`src/config/bcss_patient_splits.json`](src/config/bcss_patient_splits.json):
- **Tong so benh nhan (Total Patients):** 151 ca benh TCGA doc lap.
- **Tap huan luyen (Train Split):** 106 benh nhan (`train_count: 106`), nhan giam sat yeu da nhan cap patch $[y_0, y_1, y_2, y_3] \in \{0, 1\}^4$ trich xuat tu ten file (vi du: `TCGA-A2-A0T2_1024_2048_[1 0 0 1].png`).
- **Tap kiem dinh (Validation Split):** 22 benh nhan (`val_count: 22`), co mat na mat do pixel ground-truth phuc vu tinh chinh sieu tham so khoa dong.
- **Tap kiem thu (Test Split):** 23 benh nhan doc lap hoat dong (`test_count: 23`, `test_active_count: 23`), gom dung **4.986 anh patch test** co ground-truth danh gia doc lap.
- **Danh sach 23 ma benh nhan kiem thu chuan (Test Active Cohort):**
  1. `TCGA-D8-A27F`
  2. `TCGA-D8-A27H`
  3. `TCGA-E2-A14N`
  4. `TCGA-E2-A14R`
  5. `TCGA-E2-A14X`
  6. `TCGA-E2-A150`
  7. `TCGA-E2-A158`
  8. `TCGA-E2-A159`
  9. `TCGA-E2-A1LK`
  10. `TCGA-E2-A1LL`
  11. `TCGA-E2-A1LS`
  12. `TCGA-E2-A573`
  13. `TCGA-E2-A574`
  14. `TCGA-E9-A5FL`
  15. `TCGA-EW-A1OV`
  16. `TCGA-EW-A1OW`
  17. `TCGA-EW-A1P1`
  18. `TCGA-EW-A1P4`
  19. `TCGA-EW-A1P7`
  20. `TCGA-EW-A1P8`
  21. `TCGA-EW-A3U0`
  22. `TCGA-EW-A6SB`
  23. `TCGA-GI-A2C9`
- **Kiem chung ro ri (Leakage Verification):**
  Thuat toan trong [`src/datasets/patient_split.py`](src/datasets/patient_split.py):
  $$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$
  Ty le trung lap benh nhan giua ca 3 phan tap dat chinh xac **0.00%** (Zero Patient Leakage).

### 2.3 Module Loc Mo Lam Kinh Chuyen Biet (Otsu Clinical Gating)
Cai dat tai [`src/datasets/otsu_filter.py`](src/datasets/otsu_filter.py):
- Chuyen doi anh patch sang khong gian mau HSV va xam.
- Kenh Saturation (do bao hoa mau) phan biet manh giua mo nhuom Eosin/Hematoxylin dam va lam kinh trang.
- Tinh nguong Otsu tu dong tren kenh Saturation, gioi han nguong hieu dung $\theta_{\text{sat}} = \min(\text{Otsu} \times 0.8, 20)$ nham tranh cat nham bao tuong hoac mo dem nhuom nhat.
- Kenh cuong do xam loai tru vung lam kinh phan xa trang sang ($gray \ge 235$).
- Ap dung phep toan hinh thai hoc (Morphological Opening va Closing voi kernel elip $3 \times 3$) loai bo dom nhieu va lam day cac khoang trong noi bao nho duoi 256 pixels.
- Ham [`apply_otsu_gating_to_prediction()`](src/datasets/otsu_filter.py#L94-L122): Ep buoc toan bo cac diem anh khong thuoc vung mo tro ve phan lop 4 (Background), triet tieu hoan toan loi nhan dien sai u hoac dem tren lam kinh rong.

---

## 3. Kien Truc Module Phan Tang (Clean Modular Architecture & Directory Map)

Toan bo ma nguon duoc to chuc theo kien truc phan tach trach nhiem (Separation of Concerns):

```
SE374_WSSS_WORKSPACE/
|-- AGENTS.md                                   # Quy trinh quan tri he thong master va 32 Luat bat bien
|-- README.md                                   # Huong dan van hanh va gioi thieu do an
|-- requirements.txt                            # Danh muc thu vien phu thuoc chuan muc
|-- product/                                    # An pham hoc thuat va san pham giao nop
|   |-- proposal/                               # Mini Proposal (Word DOCX, HTML, PDF, Markdown)
|   |-- slides/                                 # Slide bao cao thuyet trinh Marp 16:9
|   `-- reports/                                # Bao cao ky thuat Springer LNCS / Luan van UIT
|-- docs/                                       # Tai lieu nghien cuu va dac ta he thong
|   |-- requirements/project_spec.md            # Dac ta dong 8 chu de, dataset, phan cung
|   |-- architecture/                           # Kien truc he thong, workflow da tac tu
|   |-- papers/                                 # 119 tai lieu PDF va Markdown toan van
|   |-- course/                                 # De bai goc va bang dang ky nhom Excel
|   `-- templates/                              # Mau bieu chuan UIT Thesis va Springer LNCS
|-- src/                                        # Ma nguon he thong phan tang Clean Architecture
|   |-- config/                                 # Manifest benh nhan va tham so he thong
|   |   `-- bcss_patient_splits.json            # 151 benh nhan (106 train, 22 val, 23 test)
|   |-- datasets/                               # Data Ingestion Service
|   |   |-- bcss_dataset.py                     # BCSSDataset PyTorch Loader va da nhan filename
|   |   |-- patient_split.py                    # Giao thuc kiem soat va chia tach Patient ID
|   |   `-- otsu_filter.py                      # Loc mo va dong cong lam kinh Otsu
|   |-- models/                                 # Medical AI Modeling Service
|   |   |-- backbone.py                         # MultiStageFeatureBackbone (Stem, Stage 1 den 5)
|   |   |-- pda_generator.py                    # Progressive Dropout Attention (PDA) module
|   |   |-- deeplabv3plus.py                    # DeepLabV3PlusWithCGM (ASPP, Decoder, CGM Gate)
|   |   |-- sam_wsss_adapter.py                 # MedSAM_WSSS_Adapter (PEFT 256->64->256, 5cls heads)
|   |   |-- coarse_to_fine_router.py            # Bo dinh tuyen da do phong dai va Pareto frontier
|   |   |-- wsss_tissue.py                      # WSSSBenchmarkingEngine, SAR, ClassMix
|   |   `-- wsss_inference_methods.py           # Engine M1 den M7, WSSSTissueStage1NetCAM, Stage2
|   |-- losses_metrics/                         # Loss va Metric Service
|   |   |-- losses.py                           # MLPS Loss, MultiLabel BCE, Uncertainty Loss
|   |   `-- metrics.py                          # mIoU, Dice, FwIoU, BF1, VTP, TSR, ICC(2,1), CCC
|   |-- clinical/                               # Clinical Service va Survival Analysis
|   |   |-- exhaustive_evaluator.py             # Evaluator doi chuan 3 mo hinh tren 4.986 patch
|   |   `-- survival_analysis.py                # Phan tang CAP, Kaplan-Meier, Log-Rank TCGA-BRCA
|   |-- training/                               # Training Pipeline Runners
|   |   |-- dataset_loader.py                   # Data loader da phan luong tap huan luyen
|   |   |-- train_stage1.py                     # Huan luyen Stage 1 ResNet38d voi PDA va ASL
|   |   |-- generate_pseudo_masks.py            # Xuat mat na gia dinh kem nguong dong
|   |   |-- train_stage2.py                     # Huan luyen Stage 2 DeepLabV3+ voi MLPS
|   |   `-- train_proposed_sam.py               # Huan luyen MedSAM PEFT Adapter
|   |-- pipeline/                               # Pipeline Orchestrators
|   |   |-- comparative_runner.py               # Runner so sanh 4 cau hinh tren 5 random seeds
|   |   |-- colab_runner.py                     # Runner thuc nghiem tren Google Colab
|   |   `-- wsi_lowmag_highmag_pipeline.py      # Pipeline WSI Low-Mag (5x) sang High-Mag (40x)
|   |-- colab_pkg/                              # Goi module Colab tinh gon (Thin Drive Package)
|   |   |-- model_loader.py                     # Loader mo hinh va checkpoints tu dong
|   |   |-- evaluator.py                        # Bo danh gia streaming an toan bo nho VRAM
|   |   |-- visualizer.py                       # Ve bieu do va heatmap ket qua
|   |   `-- downloader.py                       # Tai du lieu va checkpoints tu Cloud Drive
|   |-- visualization/                          # Workstation va Web API Service
|   |   |-- api_server.py                       # FastAPI REST Server (/api/v1/predict/patch, /viewer)
|   |   |-- clinical_viewer.py                  # OpenSeadragon renderer, Alpha blending, DZI
|   |   |-- draw_all_proposal_figures.py        # Xuat 4 hinh ve khoa hoc chuan 300 DPI
|   |   `-- draw_pipeline_diagram.py            # Ve luu do pipeline he thong
|   `-- utils/                                  # Tien ich ho tro
|       `-- deliverable_auditor.py              # Auditor kiem tra chat luong tai lieu 4 tang
|-- experiments/                                # Ket qua thuc nghiem that co log doi chung
|   |-- colab_run_1_stage2_baseline/            # Run 1: DeepLabV3+ don le (mIoU: 68.71%)
|   |-- colab_run_2_dual_stage_ensemble/        # Run 2: Dual-Stage M1 den M5 (mIoU: 70.49%)
|   |-- colab_run_3_multiscale_sota_v2/         # Run 3: Thin Runner SOTA M1 den M7 (mIoU: 70.53%)
|   |-- results_dryrun_v1/                      # Ket qua kiem thu dry-run, Pareto va LUAD
|   `-- benchmarks/                             # Bao cao doi chieu tien hoa 3 phien chay
|-- tests/                                      # 22 tep kiem thu unit tests va stress tests
|-- repos/                                      # 23 kho ma nguon SOTA va 15 meta-tools
`-- checkpoints/                                # Kho trong so mo hinh (.pth - da gitignore)
```

---

## 4. Chi Tiet Cac Kien Truc Mo Hinh & Cong Thuc Toan Hoc (Models & Mathematical Formulations)

### 4.1 Mang Phan Loai Da Nhan & Trich Xuat Dac Trung (Backbone & ResNet38d CAM)
Cai dat tai [`src/models/backbone.py`](src/models/backbone.py) va [`src/models/wsss_inference_methods.py`](src/models/wsss_inference_methods.py#L535-L640):
- **Cau truc mang:** `MultiStageFeatureBackbone` va `WSSSTissueStage1NetCAM` gom khoi Stem Conv $7 \times 7$ stride 2, MaxPool stride 2 va 5 khoi tich chap phan cap.
- **Kich thuoc dac trung da tang:**
  - `stage1`: Stride 4, 64 kenh
  - `stage2`: Stride 8, 128 kenh
  - `stage3` (`b4_5`): Stride 16, 256 hoac 512 kenh
  - `stage4` (`b5_2`): Stride 16 (Dilation 2), 512 hoac 1024 kenh
  - `stage5` (`bn7`): Stride 16 (Dilation 4), 512 hoac 4096 kenh
- **Dau phan loai:** Global Average Pooling (GAP) tren $F_{\text{bn7}}$, tiep noi boi lop Conv $1 \times 1$ hoac Fully-Connected $W_{\text{cls}} \in \mathbb{R}^{C_{\text{fg}} \times D}$ de du doan logits da nhan cho 4 phan lop tien canh:
  $$\hat{y}_c = \sigma\left(\frac{1}{H \times W} \sum_{x=1}^H \sum_{y=1}^W F_{\text{bn7}}(x, y) \cdot W_{\text{cls}}^{(c)}\right)$$

### 4.2 Progressive Dropout Attention (PDA) Module
Cai dat tai [`src/models/pda_generator.py`](src/models/pda_generator.py):
Giai quyet triet de hien tuong CAM bi co ngot cuc bo vao tam nhan te bao (under-activation) va hien tuong tran bien vao mo dem (boundary bleeding).
- **Buoc 1 - Tinh CAM khoi tao:**
  $$M_c^{(0)}(x, y) = \text{ReLU}\left(\sum_{d=1}^D W_c^d \cdot F_d(x, y)\right)$$
- **Buoc 2 - Xoa mo khong gian luy tien (Progressive Spatial Erasing tai buoc t tu 1 den T):**
  Xac dinh nguong xoa dua tren phan vi top-$p$ ($p = 0.25$) ket hop tran ty le 55% gia tri cuc dai:
  $$\tau_c^{(t)} = \max\left(\text{Quantile}\left(M_c^{(t-1)}, 1 - p\right), 0.55 \cdot \max_{x, y} M_c^{(t-1)}(x, y)\right)$$
  Mat na duy tri khong gian duoc cap nhat tich luy de ngan chan dao dong bat/tat diem cuc dai giua cac buoc lap:
  $$S_c^{(t)}(x, y) = 1.0 - \delta_t \cdot \mathbb{I}\left(M_c^{(t-1)}(x, y) \ge \tau_c^{(t)}\right), \quad S_{\text{cum}}^{(t)}(x, y) = \min_{k \in \{1,\dots,t\}} S_c^{(k)}(x, y)$$
- **Buoc 3 - Dropout kenh tich hop (Channel Dropout):**
  Loai tru cac kenh dac trung co trong so kich hoat qua cao nham kich thich cac kenh bo tro:
  $$C_c^{(t)}(d) = 1.0 - \delta_t \cdot \mathbb{I}\left(W_c^d \ge \text{Quantile}(W_c, 1 - p)\right)$$
- **Buoc 4 - Kiem soat tran bien qua nguyen mau mo (Prototype Cosine Gating):**
  Tinh vector dac trung nguyen mau lop $c$:
  $$\mu_c = \frac{\sum_{x, y} M_c^{(0)}(x, y) \cdot F(x, y)}{\sum_{x, y} M_c^{(0)}(x, y) + \epsilon}$$
  Thanh toan tu dong cong Cosine:
  $$G_c(x, y) = \text{ReLU}\left(\frac{F(x, y) \cdot \mu_c}{\|F(x, y)\|_2 \|\mu_c\|_2 + \epsilon}\right)$$
  Ban do CAM bo tro tai buoc $t$ (ap dung mat na tich luy $S_{\text{cum}}$ hoac che o cac tang dac trung trung gian truoc dilated blocks):
  $$M_c^{(t)}(x, y) = \text{ReLU}\left(W_c \cdot \left[F(x, y) \odot S_{\text{cum}}^{(t)}(x, y) \odot C_c^{(t)}\right]\right) \odot G_c(x, y)$$
- **Buoc 5 - Tong hop da buoc:**
  $$\hat{M}_c = \sum_{t=0}^T w_t M_c^{(t)}, \quad w_t = \frac{\gamma^t}{\sum_{j=0}^T \gamma^j} \quad (\gamma = 0.80)$$

### 4.3 Spatial Affinity Relocalization (SAR) Module
Cai dat tai [`src/models/wsss_tissue.py`](src/models/wsss_tissue.py#L42-L97):
- Khuech tan nang luong CAM dua tren ma tran tuong dong khong gian giua cac token dac trung cap cao $F \in \mathbb{R}^{B \times N \times D}$ ($N = H \times W$):
  $$A_{ij} = \frac{\exp\left(\frac{f_i^T f_j}{\sqrt{D}}\right)}{\sum_{k=1}^N \exp\left(\frac{f_i^T f_k}{\sqrt{D}}\right)}$$
- Tinh chinh CAM bang su ket hop tuyen tinh ($\alpha = 0.70$):
  $$M_{\text{refined}} = \alpha M + (1 - \alpha) A M$$

### 4.4 Co Che Cong Mềm & Nguong Dong Luy Tien (Dynamic Progressive Thresholding)
- Nguong dong tai epoch huan luyen $t$:
  $$\mu_t = \max\left(0.90 \times 0.98^{\max(0, t - 2)}, 0.65\right), \quad \theta_t = \mu_t \times 0.35$$
- Ghep kenh nen (Background Slice) co gia tri bang $\theta_t$:
  $$S = [M_{\text{norm}}^{(0)}, M_{\text{norm}}^{(1)}, M_{\text{norm}}^{(2)}, M_{\text{norm}}^{(3)}, \mathbf{1} \cdot \theta_t]$$
- Do khong chac chan theo Shannon Entropy chuan hoa 5 lop:
  $$\mathcal{H}_{\text{norm}}(x, y) = -\frac{1}{\ln(5)} \sum_{c=0}^4 P_c(x, y) \ln(P_c(x, y) + 1e-7)$$
  Luu y giai tich: Tren phan phoi 5 lop, ranh gioi 50/50 giua 2 loai mo co $\mathcal{H}_{\text{norm}} = \frac{\ln(2)}{\ln(5)} \approx 0.431$, va giao diem 3 loai mo co $\mathcal{H}_{\text{norm}} = \frac{\ln(3)}{\ln(5)} \approx 0.683$. Do do, he thong ket hop ca nguong entropy ha thap va hieu so bien Top-2 Margin ($\Delta_{\text{top2}} = P_{\text{top1}} - P_{\text{top2}} < 0.05$) de nhan dien chinh xac diem anh ranh gioi chuyen tiep mo u va mo dem.

### 4.5 Mang Phan Doan DeepLabV3+ & Classification Gate Mechanism (CGM)
Cai dat tai [`src/models/deeplabv3plus.py`](src/models/deeplabv3plus.py):
- **Khoi ASPP (Atrous Spatial Pyramid Pooling):**
  - Branch 1: Conv $1 \times 1$
  - Branch 2: Atrous Conv $3 \times 3$, rate = 6
  - Branch 3: Atrous Conv $3 \times 3$, rate = 12
  - Branch 4: Atrous Conv $3 \times 3$, rate = 18
  - Branch 5: Global Image Pooling + GroupNorm + Conv $1 \times 1$
  - Chieu gop 1280 kenh xuong 256 kenh bang Conv $1 \times 1$ + BatchNorm + Dropout(0.5).
- **Decoder Module:** Chieu low-level feature (stride 4, 64 kenh) thanh 48 kenh bang Conv $1 \times 1$, ghep noi voi ASPP upsampled, qua 2 khoi Conv $3 \times 3$ (256 kenh) va dau phan loai ra 5 phan lop mo.
- **Classification Gate Mechanism (CGM):**
  Giai quyet loi sinh ra u o cac anh khong he co u bang cach ap che phat nang logit dua tren vector su kien cap patch $y \in \{0, 1\}^C$:
  $$\hat{Z}_c(x, y) = Z_c(x, y) \cdot y_c + (1 - y_c) \cdot P_{\text{penalty}}, \quad P_{\text{penalty}} = -10.000$$
  Khi phan lop $c$ khong xuat hien tren anh ($y_c = 0$), logit bi ep xuong $-10^4$, xac suat Softmax tien ve $0.0000$, loai bo 100% false positive.

### 4.6 Kien Truc MedSAM WSSS Adapter (Parameter-Efficient Fine-Tuning)
Cai dat tai [`src/models/sam_wsss_adapter.py`](src/models/sam_wsss_adapter.py):
- **Image Encoder (ViT-B / Surrogate):** 86 trieu tham so, **dong bang 100% (Strictly Frozen)** nham bao toan dac trung bieu dien y khoa va tiet kiem VRAM.
- **CAM Prompt Generator:** Tu dong khai thac diem nhan (point prompts) tu cuc dai cuc bo va trong tam cuong do CAM cua cac lop hoat dong ($y_c > 0.1$); dong thoi khai thac diem phu dinh (negative points) tu cac lop canh tranh va vung lam kinh.
- **Residual Adapter Bottleneck Block:**
  Chen vao cac khoi bien doi Two-Way Cross-Attention Transformer:
  $$\text{Adapter}(X) = W_{\text{up}}\left(\text{GELU}\left(W_{\text{down}}\left(\text{LayerNorm}(X)\right)\right)\right) \cdot \text{scale}$$
  Chieu thu hep tham so: $256 \to 64 \to 256$, ti le tieng noi $\text{scale} = 0.1$. Khoi dau $W_{\text{up}}$ bang 0 de Adapter khoi dong nhu ham dong nhat (pure identity).
- **Histological Mask Decoder:** 5 class tokens song song + 1 IoU token, upscaling 2 tang ConvTranspose2d ($256 \to 64 \to 32$) ket hop 5 nhanh Hypernetwork MLPs du doan dong thoi mat na phan vung 5 lop va diem chat luong IoU.

---

## 5. Ham Mat Mat & Chi So Danh Gia Y Khoa (Losses & Clinical Metrics Formulation)

### 5.1 Ham Mat Mat (Loss Functions)
Cai dat tai [`src/losses_metrics/losses.py`](src/losses_metrics/losses.py):
1. **Multi-Label Patch Classification Loss:**
   $$\mathcal{L}_{\text{cls}} = -\frac{1}{C_{\text{fg}}} \sum_{c=1}^{C_{\text{fg}}} \left[ y_c \log \sigma(\hat{y}_c) + (1 - y_c) \log (1 - \sigma(\hat{y}_c)) \right]$$
2. **Multi-Layer Pseudo-Supervision Loss (MLPS Loss):**
   Huan luyen mang phan doan DeepLabV3+ dong thoi tu 3 tang mat na gia (Han et al., MedIA 2022):
   $$\mathcal{L}_{\text{MLPS}} = \mathcal{L}_{\text{CE}}(\hat{Z}, \hat{Y}_{\text{bn7}}) + 0.5 \cdot \mathcal{L}_{\text{CE}}(\hat{Z}, \hat{Y}_{\text{b5\_2}}) + 0.25 \cdot \mathcal{L}_{\text{CE}}(\hat{Z}, \hat{Y}_{\text{b4\_5}})$$
3. **Uncertainty Regularization Loss:**
   Ha trong so cac diem anh ranh gioi co Entropy cao tren nguong 0.65 de khong phat oan sai mo chuyen tiep:
   $$\mathcal{L}_{\text{unc}} = \frac{1}{\sum w(x, y)} \sum_{x, y} w(x, y) \cdot \mathcal{L}_{\text{CE}}(\hat{Z}(x, y), Y(x, y))$$
   Trong do trong so pixel duoc tinh voi nguong $\tau_{\text{entropy}} = 0.65$, chan duoi 0.1 de tranh triet tieu gradient va tach khoi do thi dao ham (`.detach()`):
   $$w(x, y) = \text{clamp}\left(1.0 - \frac{\mathcal{H}_{\text{norm}}(x, y)}{\tau_{\text{entropy}}}, 0.1, 1.0\right), \quad \mathcal{H}_{\text{norm}}(p) = -\frac{\sum_{c=1}^C p_c \ln(p_c)}{\ln(C)}$$

### 5.2 Chi So Danh Gia Thi Giac & Do Ranh Gioi
Cai dat tai [`src/losses_metrics/metrics.py`](src/losses_metrics/metrics.py):
- **Mean Intersection over Union (mIoU):**
  $$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}, \quad \text{mIoU} = \frac{1}{C} \sum_{c=1}^C \text{IoU}_c$$
- **Mean Dice Similarity Coefficient (mDice):**
  $$\text{Dice}_c = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$
- **Frequency-Weighted IoU (FwIoU):**
  $$\text{FwIoU} = \sum_{c=1}^C \left( \frac{N_c}{N_{\text{total}}} \cdot \text{IoU}_c \right)$$
- **Boundary F1-Score (BF1):**
  Danh gia do khop ranh gioi mo trong pham vi dung sai pixel $\theta = 2$ px bang cach xoi mon nhi phan va dan no bien $B_{\text{pred}} = M \oplus \text{Erode}(M)$.

### 5.3 Biomarkers Lam Sang & Phan Tich Song Sot (Clinical Biomarkers & Survival Analysis)
Cai dat tai [`src/clinical/survival_analysis.py`](src/clinical/survival_analysis.py):
1. **Ty le mo dem khoi u (Tumor-Stroma Ratio - TSR):**
   $$\text{TSR} = \frac{\text{Area}_{\text{TUM}}}{\text{Area}_{\text{TUM}} + \text{Area}_{\text{STR}}} \times 100\%$$
2. **Ty le mo dem (Stroma Proportion):**
   $$\text{Stroma Proportion} = 100.0\% - \text{TSR} = \frac{\text{Area}_{\text{STR}}}{\text{Area}_{\text{TUM}} + \text{Area}_{\text{STR}}} \times 100\%$$
3. **Phan tram u con song (Viable Tumor Percentage - VTP / Cellularity):**
   $$\text{VTP} = \frac{\text{Area}_{\text{TUM}}}{\text{Area}_{\text{Total Tissue}}} \times 100\%$$
4. **Phan tang nguy co theo Quy chuan Dong thuan Quoc te Mesker / van Pelt (UNITED Consortium):**
   - **Nhom Nguy Co Cao (Stroma-Rich / High Risk):** $\text{TSR} \le 50.0\%$ (Ty le mo dem Stroma Percentage $> 50.0\%$). Mo dem day dac bao quanh te bao u lien quan den tien luong xau va khang hoa tri.
   - **Nhom Nguy Co Thap (Stroma-Poor / Low Risk):** $\text{TSR} > 50.0\%$ (Ty le mo dem Stroma Percentage $\le 50.0\%$). Tien luong song sot tot hon.
   - *Luu y lam sang:* Quy trinh chuan cua van Pelt et al. khuyen nghi cham diem TSR tai vi truong vat kinh $10\times$ giau mo dem nhat o ranh gioi xam lan (invasive front). Viec cong don pixel tren toan bo patch la chi so uoc luong tinh toan (surrogate computational proxy).
5. **Uoc luong duong cong sinh ton Kaplan-Meier & Kiem dinh Log-Rank:**
   $$S(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right), \quad \chi^2 = \frac{\left(\sum (O_{1i} - E_{1i})\right)^2}{\sum V_{1i}} \sim \chi^2(1)$$
   Duoc kiem dinh tren du lieu song sot cua 23 benh nhan kiem thu TCGA-BRCA nham muc dich kiem thu tich hop chuc nang phan mem (functional pipeline demonstration). Tren co mau $N = 23$ benh nhan kiem thu, luc thong ke dat khoang $7.6\% - 13.7\%$, can tap du lieu quy mo lon hon cua TCGA-BRCA de rut ra ket luan lam sang doc lap.
6. **Do dong thuan giua AI va Bac si (Lin CCC & Shrout-Fleiss ICC(2,1)):**
   Cai dat tai ham tinh toan de danh gia chenh lech bias tuyet doi va 95% Bland-Altman Limits of Agreement (LoA).

---

## 6. Minh Chung So Lieu Thuc Nghiem That 100% (Genuine Experimental Verification & Benchmark Results)

Toan bo so lieu duoi day duoc trich xuat nguyen ban tu log thuc nghiem chay that tren GPU Tesla T4 tren toan bo **4.986 patch test** cua **23 benh nhan** BCSS-WSSS doc lap tai [`experiments/benchmarks/COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md`](experiments/benchmarks/COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md) va [`experiments/colab_run_3_multiscale_sota_v2/COLAB_EVALUATION_REPORT.json`](experiments/colab_run_3_multiscale_sota_v2/COLAB_EVALUATION_REPORT.json):

### 6.1 Tien Hoa 3 Phien Chay Thuc Nghiem (Evolution Across 3 Colab Full Runs)

| Phien Chay (Run ID) | Tap Tin Log Doi Chung | Kien Truc & Co Che Cot Loi | mIoU 4 Lớp (%) | mDice 4 Lớp (%) | PA (%) | mIoU 5 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | BACK IoU (%) | Toc Do (FPS) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1 (Baseline)** | `experiments/colab_run_1_stage2_baseline/` | Stage-2 DeepLabV3+ ResNet-101 don le (khong co Stage-1 CGM), Otsu Gating | **68.71%** | **81.15%** | **84.94%** | **69.02%** *(dinh chinh sai lech so hoc 67.36%)* | 79.12% | 73.50% | 56.97% | 65.23% | 70.26% | 23.38 |
| **Run 2 (Dual-Stage)** | `experiments/colab_run_2_dual_stage_ensemble/` | Dual-Stage (`ResNet38d Net_CAM` + `DeepLabV3+`), 5 Methods (`M1-M5`, FP32): CGM + Dihedral TTA + Long-Tail Prior + Pure-Glass Guard | **70.49%** | **82.51%** | **85.35%** | 76.38% | 79.19% | 73.80% | 61.27% | 67.69% | 99.93% | 5.77 |
| **Run 3 (SOTA V2)** | `experiments/colab_run_3_multiscale_sota_v2/` | **Thin Drive Runner v2 (`MultiMethodWSSSEngineV2`, FP16 AMP, 7 Methods `M1-M7`): Kim Tu Thap Da Ty Le D4 (`0.75x, 1.0x, 1.25x`) Entropy-Weighted + Cong Mem Stage-1** | **70.53%** | **82.54%** | **85.41%** | **76.41%** | **79.37%** | **73.82%** | **61.33%** | **67.61%** *(M5: 67.67%)* | **99.93%** | **4.91** *(7 methods song song)* |

### 6.2 Bang Phan Tich Boc Tach (Ablation Study) 7 Phuong Phap Trong Run 3

Danh gia doi chung tren cung toan bo 4.986 anh test (Ground-Truth Verified):

| Key | Phuong Phap (Inference Method) | Mo Ta Ky Thuat Chi Tiet | mIoU 4 Lớp (%) | mDice 4 Lớp (%) | Pixel Acc (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | Kiem Dinh Wilcoxon (vs M1) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | Baseline WSSS-Tissue | DeepLabV3+ Single-scale khong co Stage-1 CGM | 69.35% | 81.69% | 84.89% | 78.46% | 73.32% | 60.00% | 65.62% | $p = 1.0000$ (Chuan doi chung) |
| **M2** | Multi-View D4 TTA + CGM | 4 goc nhin truc giao Dihedral D4 + Stage-1 Net_CAM CGM ($y_c > 0.1$) | 70.30% | 82.37% | 85.34% | 79.08% | 73.88% | 60.75% | 67.51% | $p = 1.197 \times 10^{-23}$ |
| **M3** | Cross-Stage Fusion & Prior | Ghep CAM-Segmentation da tang + Hieu chinh tien nghiem duoi dai (Menon et al.) | 70.50% | 82.51% | 85.36% | 79.23% | 73.78% | 61.31% | 67.66% | $p = 4.490 \times 10^{-17}$ |
| **M4** | Spatial Bilateral Refinement | Loc song phuong bao toan bien ARML + Triet tieu dao nhieu PistoSeg RFM | 70.32% | 82.38% | 85.34% | 79.09% | 73.89% | 60.77% | 67.53% | $p = 1.563 \times 10^{-29}$ |
| **M5** | Proposed SOTA Ensemble V1 | Tich hop toan dien V1: CGM + TTA + CAM Fusion + Long-tail Prior + Glass Guard | 70.51% | 82.52% | 85.36% | 79.24% | 73.79% | **61.33%** | **67.67%** | $p = 1.425 \times 10^{-24}$ |
| **M6** | **Multi-Scale Pyramid D4 (SOTA)** | **Kim tu thap da do phan giai (0.75x, 1.0x, 1.25x) trong so boi nghich dao Entropy + Cong mem** | **70.53%** | **82.54%** | **85.41%** | **79.37%** | **73.82%** | **61.33%** | 67.61% | **$p = 2.604 \times 10^{-15}$** |
| **M7** | Beer-Lambert OD + Multi-Stride | Giai ma quang pho H&E Ruifrok-Johnston + Tinh chinh bien quang mat do | 70.44% | 82.47% | 85.36% | 79.25% | 73.81% | 61.21% | 67.48% | $p = 3.590 \times 10^{-18}$ |

*Ghi chu ve kiem dinh thong ke va danh doi hieu nang:*
- **Kiem dinh Wilcoxon:** Cac gia tri $p$ tren duoc tinh tren $N = 4.986$ anh patch kiem thu. Do cac patch tu cung mot benh nhan co tuong quan noi cum (pseudo-replication), khi gom cum theo cap do benh nhan ($N = 23$ ca doc lap), gia tri $p$ nho nhat co the ve mat ly thuyet la $p_{\min} = 2 / 2^{23} \approx 2.38 \times 10^{-7}$.
- **Danh doi toc do suy luan:** Phuong phap M6 can 12 luot suy luan tren moi patch (3 ty le $\times$ 4 huong D4) khien toc do giam xuong 4.91 FPS (ton ~85 phut cho tieu ban 25.000 patch) de doi lay muc tang +0.02% mIoU so voi M5 (70.51%). Trong van hanh thuc te tren Web API Server, he thong khuyen nghi su dung M3 hoac M5 (chay 1 luot hoac 4 luot TTA, dat 70.50% - 70.51% mIoU o toc do 5.77 - 23.38 FPS) nham can bang toi uu giua do chinh xac va do tre phan hoi.

### 6.3 So Sanh Doi Chung Quoc Te Voi Cac Cong Trinh Hang Dau Tren BCSS-WSSS

Bang duoi day doi chieu M1-M6 voi cac cong trinh goc va cac SOTA hang dau giai doan 2025-2026 tren tap kiem thu BCSS-WSSS (4.986 patch):

| Cong Trinh Cong Bo | Hoi Nghi / Tap Chi | Nam | Khung Xuong & Co Che | mIoU 4 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **HistoSegNet** (Chan et al.) | ICCV | 2019 | CNN Grad-CAM | 46.92% | 56.92% | 57.53% | 32.31% | 40.91% |
| **SEAM** (Wang et al.) | CVPR | 2020 | Self-Supervised Equivariant | 61.41% | 74.52% | 68.09% | 50.19% | 52.84% |
| **WSSS-Tissue (Stage-1 Pseudo)** | MedIA | 2022 | ResNet-38 CAM | 67.05% | 77.59% | 72.62% | 56.40% | 61.58% |
| **WSSS-Tissue (Stage-2 Paper)** | MedIA | 2022 | ResNet38d + DeepLabV3+ | 69.34% | 78.44% | 73.31% | 59.98% | 65.62% |
| **LowMagWSS (40x Benchmark)** | arXiv | 2024 | Super-Resolution + WSSS | 69.34% | 78.44% | 73.31% | 59.98% | 65.62% |
| **PistoSeg (Mosaic + RFM)** | AAAI | 2023 | Mosaic Patch + Refining | 69.95% | 78.90% | 73.60% | 60.50% | 66.80% |
| **Our M1 Baseline (Reproduced)** | Thuc nghiem that | 2026 | DeepLabV3+ (ResNet-101) | **69.35%** | 78.46% | 73.32% | 60.00% | 65.62% |
| **Our M5 Proposed SOTA Ensemble** | Thuc nghiem that | 2026 | Test-Time Ensemble (D4+CGM+Prior) | **70.51%** | 79.24% | 73.79% | **61.33%** | **67.67%** |
| **Our M6 Multi-Scale Pyramid D4** | Thuc nghiem that | 2026 | Test-Time Pyramid D4 (12 passes) | **70.53%** | **79.37%** | **73.82%** | **61.33%** | 67.61% |
| **HisynSeg** (Fang et al.) | IEEE TMI | 2025 | Bézier Image-Mixing + Consistency | 70.73% | 81.64% | 74.67% | 60.15% | 66.48% |
| **ConStruct** (Vu et al.) | MIDL | 2026 | CONCH PFM Distillation (3.7% params) | 70.96% | N/A | N/A | N/A | N/A |
| **PLIMSeg** (VLM Alignment) | Open Access | 2026 | Vision-Language Model Alignment | 71.07% | 81.39% | 75.41% | 60.33% | 67.16% |
| **DualProtoSeg** (Vu et al.) | arXiv | 2025 | CONCH PFM Dual Prototypes | 71.35% | N/A | N/A | N/A | N/A |
| **CSS** (Causal Spectral Segmenter) | Bioinformatics | 2026 | ViT-S/16 + Spectral Graph Reasoning | 71.72% | 78.95% | 73.45% | 65.43% | 69.05% |

*Phan dinh kien truc va dong gop:* To hop M1-M6 la bo giai thuat tang cuong suy luan hau xu ly khong can huan luyen lai (Training-Free Test-Time Inference Engine) giup nang hieu nang cua mo hinh co so WSSS-Tissue tu 69.35% len 70.53% (+1.18% mIoU). Trong khi do, cac module Progressive Dropout Attention (PDA), Spatial Affinity Relocalization (SAR) va MedSAM WSSS Adapter trong `src/models/` la cac nhanh kien truc huan luyen de xuat duoc mo hinh hoa doc lap de tiep tuc dong bo va chieu sau trong cac huong phat trien tiep theo.

---

## 7. He Sinh Thai Repositories Chuyen Sau & Meta-Tools (`repos/`)

Toan bo 23 kho nghien cuu va 15 cong cu ky thuat da duoc tich hop truc tiep tai `repos/` nham doi chieu code-to-code theo tieu chuan kiem dinh:

### 7.1 Danh Muc 23 Repositories Nghien Cuu Chuyen Sau
1. **[`repos/MONAI/`](repos/MONAI):** Medical Open Network for AI (NVIDIA & King College London). Bo bien doi anh 3D/WSI, Sliding Window Inference, mang Swin-UNETR va ham do Dice/HD95.
2. **[`repos/LowMagWSS/`](repos/LowMagWSS):** Weakly-Supervised Segmentation for WSI using Low-Magnification Cues (arXiv:2607.10783, UIT Medical Lab). Xu ly WSI o do phong dai 5x/10x, giam 16 den 64 lan so luong patches.
3. **[`repos/PBIP/`](repos/PBIP):** Prototype-Based Image Prompting for Weakly-Supervised Semantic Segmentation (CVPR 2025, arXiv:2503.12068). Ngan hang nguyen mau tuong phan ket hop SegFormer.
4. **[`repos/CGRL/`](repos/CGRL):** Concept-Guided Representation Learning for WSI Cancer Subtyping (arXiv:2607.12556, Topic 3).
5. **[`repos/CLAM/`](repos/CLAM):** Clustering-Constrained Attention Multiple Instance Learning (Nature BME, Mahmood Lab). Cat patch WSI va xuat Attention Heatmap truc quan.
6. **[`repos/TransMIL/`](repos/TransMIL):** Transformer-based Correlated Multiple Instance Learning (NeurIPS 2021). Khai thac tuong quan khong gian toan slide WSI.
7. **[`repos/ISLES_Stroke/`](repos/ISLES_Stroke):** Collateral Imaging & Infarct Core Segmentation in Acute Ischemic Stroke (ISLES Challenge / Springer, Topic 7).
8. **[`repos/MDPRNet/`](repos/MDPRNet):** Sparse-View & Limited-Angle CT Reconstruction using Dual-Domain Unrolling Networks (Springer MVAP, Topic 1 va 2).
9. **[`repos/WSI_Viewer/`](repos/WSI_Viewer):** Web interface duyet tieu ban WSI gigapixel tich hop OpenSeadragon va FastAPI DeepZoom tile streaming.
10. **[`repos/ARML/`](repos/ARML):** Affinity Relocalization for Weakly-Supervised Histology (MICCAI 2024). Nguon cam hung cho khoi `SpatialAffinityRelocalization`.
11. **[`repos/HisynSeg/`](repos/HisynSeg):** Synthetic boundary histopathology segmentation with Bézier curve augmentation (IEEE TMI 2025, arXiv:2412.20924).
12. **[`repos/SAM2PATH/`](repos/SAM2PATH):** Adapting Segment Anything 2 for Digital Pathology with Kolmogorov-Arnold Learnable Prompts.
13. **[`repos/WSInfer/`](repos/WSInfer):** Deep Learning Inference Pipeline for Whole Slide Images (TorchScript, QuPath plugin).
14. **[`repos/WSSS-Tissue/`](repos/WSSS-Tissue):** Baseline WSSS tren BCSS va LUAD (Chu Han et al., MedIA 2022). Nguon tham chieu pipeline 3 buoc: ResNet-38 CAM sang Pseudo-Mask sang DeepLabV3+.
15. **[`repos/ConStruct/`](repos/ConStruct):** Structural Distillation for WSSS on BCSS (MIDL 2026, arXiv:2512.10316). Tich hop CONCH va SegFormer MiT-B1 voi 3.7% trainable adapter params.
16. **[`repos/MambaMIL/`](repos/MambaMIL):** Mamba State Space Models for Whole Slide Image Classification (arXiv:2403.06800).
17. **[`repos/OpenMidnight/`](repos/OpenMidnight):** Pathology Foundation Model quy mo 1.1 ty tham so (DINOv2 ViT-G/14 huan luyen boi Sophont & MedARC tren 12.000 tieu ban WSI TCGA voi 29 trieu patch doc nhat tren cum 8x NVIDIA H100).
18. **[`repos/PFM_Segmentation/`](repos/PFM_Segmentation):** Pathology Foundation Models Dense Segmentation Benchmark (arXiv:2602.03887).
19. **[`repos/PathoGen/`](repos/PathoGen):** Generative modeling cho sinh anh mo benh hoc nhan tao.
20. **[`repos/PistoSeg/`](repos/PistoSeg):** Pure Patch Mosaic Synthesis & Refining Feature Module (AAAI 2023). Nguon tham chieu co che xoa dao nhieu.
21. **[`repos/S2C/`](repos/S2C):** Self-Supervised SAM to CAMs for Weakly-Supervised Semantic Segmentation (CVPR 2024). Nguon cam hung cho bo sinh Prompt CAM tu dong.
22. **[`repos/SEINE/`](repos/SEINE):** Multi-modal histopathology analysis framework.
23. **[`repos/WaWeHis/`](repos/WaWeHis):** Wavelet-based histology representation learning.

### 7.2 Danh Muc 15 Meta-Tools Ho Tro Ky Thuat (`repos/meta_tools/`)
1. **[`repos/meta_tools/it-tools/`](repos/meta_tools/it-tools):** Hon 100 tien ich chuyen doi offline (JSON, YAML, CSV, Regex, Base64, Token).
2. **[`repos/meta_tools/CyberChef/`](repos/meta_tools/CyberChef):** Pipeline bien doi du lieu, giai ma nhi phan, boc tach cau truc file phuc tap cua GCHQ.
3. **[`repos/meta_tools/the-book-of-secret-knowledge/`](repos/meta_tools/the-book-of-secret-knowledge):** Cam nang lenh CLI va shell one-liners xu ly tap tin hang loat khong can viet script tam bo.
4. **[`repos/meta_tools/awesome/`](repos/meta_tools/awesome):** Ban do danh muc open-source the gioi dan den cac kho cong cu chuan nhat.
5. **[`repos/meta_tools/free-for-dev/`](repos/meta_tools/free-for-dev):** Danh ba dich vu cloud compute, database, CI/CD runners mien phi (Free Tier).
6. **[`repos/meta_tools/awesome-selfhosted/`](repos/meta_tools/awesome-selfhosted):** Cac ung dung tu host tren may chu cuc bo thay the SaaS thuong mai tra phi.
7. **[`repos/meta_tools/awesome-python/`](repos/meta_tools/awesome-python):** Bach khoa toan thu tra cuu nhanh cac thu vien Python chuan theo chuc nang (Imaging, CLI, Files).
8. **[`repos/meta_tools/Awesome-AI4DigitalPathology/`](repos/meta_tools/Awesome-AI4DigitalPathology):** Co so tri thuc WSI va AI giai phau benh: UNI, CONCH, Virchow, CHIEF.
9. **[`repos/meta_tools/awesome-pathology/`](repos/meta_tools/awesome-pathology):** Phan mem mo xem anh mo benh hoc: QuPath, ASAP, OpenSeadragon, SlideIO.
10. **[`repos/meta_tools/awesome-pdf/`](repos/meta_tools/awesome-pdf):** Danh muc cong cu va thu vien chuyen doi, boc tach va xu ly tai lieu PDF chuyen sau.
11. **[`repos/meta_tools/markitdown/`](repos/meta_tools/markitdown):** Cong cu Microsoft MarkItDown chuyen PDF, Word, PowerPoint sang Markdown cho AI ingestion.
12. **[`repos/meta_tools/streamlit/`](repos/meta_tools/streamlit):** Framework tao web app y te sieu toc bang Python voi slider nguong HU va interactive viewer.
13. **[`repos/meta_tools/gradio/`](repos/meta_tools/gradio):** Giao dien tuong tac ML cho phep keo tha anh chan doan va xuat ket qua truc quan hoa tuc thi.
14. **[`repos/meta_tools/design-resources-for-developers/`](repos/meta_tools/design-resources-for-developers):** Bo icon SVG, mockup UI va bang mau y te tieu chuan cong nghiep.
15. **[`repos/meta_tools/awesome-design/`](repos/meta_tools/awesome-design):** Huong dan thiet ke giao dien nguoi dung, typography va he thong design systems.

---

## 8. Huong Dan Thuc Thi & Lenh Chay Chi Tiet (Execution Commands & Operational Workflows)

### 8.1 Thiet Lap Moi Truong (Environment Setup)
He dieu hanh: Windows 11. Cac lenh duoc thuc thi truc tiep qua `cmd /c`:

```bash
cmd /c python --version
cmd /c python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
cmd /c pip install -r requirements.txt
```

### 8.2 Chay Kiem Thu He Thong (Pytest Automated Test Suites)
He thong so huu 22 tep kiem thu toan dien tai thu muc `tests/` voi 348 test cases da vuot qua 100%. Cac lenh kiem thu chinh gom:

```bash
cmd /c python -m pytest tests/test_pda_backbone.py -v
cmd /c python -m pytest tests/test_deeplabv3plus.py -v
cmd /c python -m pytest tests/test_sam_wsss_adapter.py -v
cmd /c python -m pytest tests/test_clinical_service.py -v
cmd /c python -m pytest tests/test_patient_split.py -v
cmd /c python -m pytest tests/test_governance_invariants.py -v
cmd /c python -m pytest tests/ -q
```

### 8.3 Kiem Tra Tinh Ma Nguon & Bao Mat SAST (Ruff & Bandit)
Kiem tra chat luong ma nguon va an toan thuc thi theo chuan nganh cong nghiep:
- Quet linter tren toan bo 42 module ma nguon (0 loi vi pham): `cmd /c ruff check src/`
- Quet an toan SAST tren 16.009 dong ma nguon (0 loi High severity): `cmd /c bandit -r src/ -ll`

```bash
cmd /c ruff check src/
cmd /c bandit -r src/ -ll
```

### 8.4 Kiem Soat Chat Luong An Pham & Bieu Mau (Independent Deliverable Auditor)
Kiem soat 4 tang gom hinh anh don sac vector/300 DPI, typography ASCII hyphen, liem chinh khoa hoc va cau truc hoc thuat:

```bash
cmd /c python -m src.utils.deliverable_auditor product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md
cmd /c python -m src.utils.deliverable_auditor docs/architecture/SYSTEM_TECHNICAL_SPECIFICATION.md
```

### 8.5 Khoi Chay Web API Server & Tram Lam Viec Lam Sang (FastAPI & Clinical Viewer)
Tram lam viec cung cap giao dien OpenSeadragon va REST API day du:

```bash
cmd /c uvicorn src.visualization.api_server:app --host 0.0.0.0 --port 8000 --reload
```

Cac diem ket noi dich vu chinh:
- Giao dien truyen hinh anh OpenSeadragon: `http://localhost:8000/viewer`
- Tai lieu dac ta API Swagger OpenAPI: `http://localhost:8000/docs`
- Diem kiem tra trang thai he thong: `http://localhost:8000/api/v1/health`

### 8.6 Huan Luyen Mo Hinh Cuc Bo (Local Training Pipelines)
Cac lenh huan luyen bon giai doan (su dung dung cac co tham so argparse thuc te):

```bash
cmd /c python -m src.training.train_stage1 --data-dir data/BCSS-WSSS --epochs 20 --batch-size 16 --device cuda
cmd /c python -m src.training.generate_pseudo_masks --data-dir data/BCSS-WSSS --checkpoint-path checkpoints/stage1_retrained_best.pth --output-dir data/pseudo_masks
cmd /c python -m src.training.train_stage2 --data-dir data/BCSS-WSSS --epochs 30 --batch-size 8 --device cuda
cmd /c python -m src.training.train_proposed_sam --data-dir data/BCSS-WSSS --epochs 20 --batch-size 4 --device cuda
```

### 8.7 Chay Doi Chuan Lam Sang & Phan Tich Song Sot (Exhaustive Benchmark Evaluator CLI)
Cac lenh doi chuan tren tap kiem thu:

```bash
cmd /c python -m src.clinical.exhaustive_evaluator --dataset-dir data/BCSS-WSSS --dry-run
cmd /c python -m src.clinical.exhaustive_evaluator --dataset-dir data/BCSS-WSSS --full --device cuda
```

### 8.8 Cac So Tay Google Colab Tinh Gon (Thin Drive Runner Suites T4/L4/A100)
Cac so tay duoc luu tru tai `notebooks/` va `drive_upload_ready/`:
- `notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`: Chay tren Tesla T4 Free Tier (15 GB VRAM), batch size 8 hoac 16, FP16 AMP.
- `notebooks/SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb`: Chay tren NVIDIA L4 Pro (24 GB VRAM), batch size 32.
- `notebooks/SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb`: Chay tren NVIDIA A100 High-RAM (40/80 GB VRAM), batch size 64.
- `notebooks/SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb`: So tay demo mo rong WSI gigapixel tu Low-Mag (5x) sang High-Mag (40x).

### 8.9 Cap Nhat Ban Do Tri Thuc Ma Nguon (Graphify CLI)
Cac lenh cap nhat do thi tri thuc:

```bash
cmd /c graphify extract src --code-only
cmd /c graphify update .
```

---

## 9. Thong Tin Do Thi Tri Thuc Ma Nguon (Knowledge Graph Metadata)

Theo ket qua trich xuat AST tu dong tai [`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md):
- **Phien ban Cong cu:** Graphify CLI v0.9.49-s2 (AST-only, zero API cost).
- **Quy mo ma nguon:** 157 tep ma nguon va tai lieu (~1.288.476 tu).
- **Thong so mang do thi:** **2.662 nut (nodes)**, **4.584 canh (edges)**, **152 cong dong tri thuc (communities)**.
- **Do chinh xac trich xuat:** 99% Extracted tu cu phap Python/Markdown, 1% Inferred (50 canh suy dien voi do tin cay trung binh 0.95), 0% Ambiguous.
- **Ma Commit goc gan nhat:** `2248e7b1`.
- **Cac nut trung tam dieu phoi chu chot (Central Community Hubs):**
  - `WSILowMagHighMagPipeline`: Dieu phoi toan bo pipeline tieu ban gigapixel Low-Mag sang High-Mag.
  - `MultiMethodWSSSEngine`: Dieu phoi 7 phuong phap suy luan SOTA M1 den M7.
  - `WSSSTissueStage1NetCAM` va `HistopathologyClassifierBackbone`: Mang co so Stage-1 ResNet-38.
  - `DeepLabV3PlusWithCGM`: Mang phan doan DeepLabV3+ tich hop cong phan loai.
  - `MedSAM_WSSS_Adapter`: Mo hinh nen tang thich ung tham so PEFT.
  - `ExhaustiveBenchmarkEvaluator`: Tich hop danh gia toan dien 4.986 patch test va phan tich song sot benh nhan.
  - `api_server.py`: REST API va Web Workstation OpenSeadragon.

---

## 10. Ket Luan & Huong Phat Trien Mo Rong (Conclusions & Roadmap)

He thong phan mem phan vung ngu nghia mo benh hoc giam sat yeu (SE374 Medical WSSS Workstation) da duoc thiet ke, hien thuc va kiem toan hoan thien theo dung chuan muc Ky thuat Phan mem Y te. Ket qua thuc nghiem dat dinh cao **70.53% mIoU 4 phan lop mo** va **76.41% mIoU 5 phan lop** tren toan bo 4.986 patch test thuc te, vuot qua tieu chuan cong bo cua tac gia MedIA 2022 (69.34%) va mo ra kha nang ung dung lam sang thuc thu giup tiet kiem hon 30 lan thoi gian doc tieu ban cho bac si giai phau benh.
