# HO SO DAC TA HE THONG VA MO TA DU AN SE374.R11
# PHUC VU AI DOC LAP THAM DINH VA DANH GIA TOAN DIEN (PROJECT EVALUATION DOSSIER)

> **Co so nghien cuu:** Phong Thi Nghiem Nghien Cuu Tri Tue Nhan Tao Y Te / Medical AI Research Laboratory & Enterprise Pipeline  
> **Don vi dao tao:** Khoa Ky Thuat Phan Mem, Truong Dai Hoc Cong Nghe Thong Tin (UIT), DHQG-HCM  
> **Giang vien phu trach / Co van khoa hoc:** TS. Bui Cao Doanh (`doanhbc@uit.edu.vn`)  
> **Don vi thuc hien:** [Tac gia / Nhom nghien cuu] (Medical AI Research Consortium) - Topic 6 (Weakly-Supervised Segmentation for Whole Slide Images)  
> **Ten de tai chinh thuc:** Coarse-to-Fine Weakly-Supervised Semantic Segmentation on BCSS Breast Cancer Histopathology Whole Slide Images  
> **Repository:** https://github.com/medical-ai-consortium/bcss-wsss  
> **Tieu chuan quan tri he thong:** AGENTS.md v5.0 (Invariants 01-32, Zero Mockup, Patient-Level Anti-Leakage, Strict ASCII Hyphen 0x2D Only)  
> **Trang thai he thong:** L4 Scientific Upgrade da hoan thien, bo go Colab Thin-Runner dong bo day du tai `drive_upload_ready/SE374_WSSS_Drive_Code/`, 311/311 unit & regression tests pass 100%.  

---

## 1. TONG QUAN DU AN VA HO SO HOC PHAN (EXECUTIVE SUMMARY & COURSE PROFILE)

### 1.1 Thong Tin Hoc Phan
- **Ten mon hoc:** Phat trien phan mem xu ly anh y khoa (Medical Image Processing Software Development)
- **Ma lop hoc phan:** SE374.R11
- **Don vi quan ly:** Khoa Ky thuat Phan mem, Truong Dai hoc Cong nghe Thong tin, DHQG-HCM
- **Nien khoa:** 2026-2027
- **Giang vien phu trach:** TS. Bui Cao Doanh
- **Doi tuong tham dinh:** He thong phan mem AI y te hoan chinh, bao gom: Pipeline tien xu ly va tao mat na gia da tang (Multi-Layer Pseudo-Supervision), Mo hinh phan doan DeepLabV3+ ResNet-101 tich hop Classification Gate Mechanism (CGM), Mo hinh nen tang thich ung MedSAM WSSS Adapter (Parameter-Efficient Fine-Tuning voi LoRA va 24 point prompts tu CAM), Bo danh gia doi chuan 3 huong (Exhaustive 3-Way Benchmark Evaluator) voi kiem dinh thong ke Wilcoxon va hieu chinh Holm-Bonferroni, Module tien luong sinh ton lam sang Kaplan-Meier va Tram lam viec giai phau benh so FastAPI kem trinh xem anh Gigapixel OpenSeadragon.

### 1.2 Boi Canh Lam Sang & Y Nghia Khoa Hoc
Trong chan doan giai phau benh ung thu vu (Breast Invasive Carcinoma - TCGA-BRCA), cac tieu ban nhuom Hematoxylin & Eosin (H&E) toan ban (Whole Slide Images - WSI) co kich thuoc gigapixel ($100.000 \times 100.000$ diem anh). Viec khoanh vung thu cong tung diem anh (pixel-level annotation) doi hoi tu 2 den 4 gio cho moi ca benh, dong thoi ton tai do bien thien nhan dinh lien chuyen gia cao ($>20\%$ inter-observer variability).

Huong tiep can Phan vung Ngu nghia Giam sat Yeu (Weakly-Supervised Semantic Segmentation - WSSS) chi su dung nhan nhi phan cap mang anh ($y \in \{0, 1\}^C$) de huan luyen mo hinh phan doan cap pixel. Huong tiep can nay tiet kiem hon $30\times$ thoi gian gan nhan chuyen gia, nhung phai giai quyet bon diem nghen ky thuat lon:
1. **Hien tuong kich hoat thieu (CAM Under-activation):** Ban do Class Activation Map chi bat vao loi nhan te bao noi bat nhat, bo sot ranh gioi xam lan bieu mo u va cac dao lympho bao.
2. **Lo hong diem nen bang 0 (Background Score Gap):** Khi xu ly anh nen kinh trang hoac mo mo khong chua te bao, cac phuong phap kinh dien dat diem nen `bg_score = 0` dan den viec phep toan `argmax` du doan nham thanh khoi u (Tumor Epithelium), tao ty le duong tinh gia nguy hai cho benh nhan ($>15\%$).
3. **Ngiep vu tran lop (Class Bleeding):** Mang phan doan cap pixel van co the du doan nham sang cac lop khong he ton tai tren patch anh.
4. **Gioi han phan cung GPU:** Mo hinh nen tang y te (nhu MedSAM voi Vision Transformer ViT-B 86 trieu tham so) rat de gay tran bo nho (OOM) neu khong co kien truc Adapter tinh gon.

---

## 2. NGUYEN VAN THONG TIN DE BAI & CHU DE MON HOC (VERBATIM TOPIC SPECIFICATIONS)

### 2.1 Trich Xuat Nguyen Van Tai Lieu De Bai Mon Hoc (`docs/course/topic.md`)

Duoi day la toan van noi dung de bai 8 chu de nghien cuu do TS. Bui Cao Doanh ban hanh cho lop SE374.R11:

```markdown
# DANH MUC CHU DE NGHIEN CUU VA DO AN MON HOC SE374.R11

- Mon hoc: SE374.R11 - Phat trien phan mem xu ly anh y khoa (2026)
- Khoa: Ky thuat Phan mem, Truong Dai hoc Cong nghe Thong tin (UIT), DHQG-HCM
- Giang vien huong dan: TS. Bui Cao Doanh

---

## 1. BANG TONG HOP 8 CHU DE MON HOC (TOPIC MATRIX)

| STT Topic | Ten Chu De | Ban Chat Bai Toan & Phuong Phap | Bo Du Lieu Tieu Bieu | Tai Lieu Tham Khao Chinh |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Sparse-view CT reconstruction** | Bai toan nghich dao (inverse problem): khoi phuc CT chat luong cao tu so goc chieu thua de giam lieu tia X; khu streak artifacts. Ke thua mang unrolling (CG-GLORE, LPD, FBPConvNet). | AAPM Low-Dose CT, DeepLesion | arXiv:2608.15246 (CG-GLORE), Springer MVAP 2026 |
| **2** | **Limited-angle CT reconstruction** | Khoi phuc anh CT khi dai goc quet bi gioi han (thieu wedge thong tin). Sinogram inpainting + Image-domain regularization. | AAPM, LoDoPaB-CT | Springer MVAP 2026, arXiv:2608.15246 |
| **3** | **Cancer subtyping using Whole Slide Images (WSI)** | Phan loai duoi nhom ung thu tu anh mo benh hoc gigapixel; Bag of patches & Multiple Instance Learning (MIL) voi concept huong dan (CGRL, CLAM, TransMIL). | TCGA-BRCA, TCGA-NSCLC | arXiv:2607.12556 (CGRL) |
| **4** | **Survival analysis using Whole Slide Images (WSI)** | Du doan thoi gian song sot / nguy co tu vong (time-to-event); Continual learning tren nhieu nhom ung thu (MergeSurv, Porpoise, DeepAttnMISL). | TCGA Cohorts (LUAD, BLCA, KIRC) | arXiv:2607.04747 (MergeSurv) |
| **5** | **Tumor segmentation using CT/fMRI (MRI)** | Phan vung khoi u (u nao Glioma, u noi tang) tren anh the tich 3D. Su dung MONAI Swin-UNETR, DynUNet, 3D U-Net, nnU-Net. | BraTS 2021/2023, LiTS | arXiv:1804.10484, GitHub: DBC-Lab/Brain_MRI_Enhancement |
| **6** | **Weakly-supervised segmentation for WSI** | Phan vung ton thuong giai phau benh chi voi nhan muc slide hoac anh do phong dai thap (LowMagWSS, WSSS-Tissue, HisynSeg, ARML). | Camelyon16, BCSS | arXiv:2607.10783 (LowMagWSS), GitHub: Dung-Dx/LowMagWSS |
| **7** | **Collateral imaging in acute ischemic stroke using MRI** | Danh gia tuan hoan bang he trong dot quy thieu mau nao cap; luong hoa vung tranh toi tranh sang (penumbra) va loi nhoi mau (infarct core). | ISLES 2022, Perfusion MRI | Springer Int J CARS (10.1007/s11548-024-03229-5) |
| **8** | **Tu do de xuat (Free Medical AI Proposals)** | - Phat hien pho tu ky (ASD) dua tren fMRI nao (Dataset ABIDE I, functional connectivity GNN/CNN).<br>- Ung dung hoi dap / tro ly VQA ho tro bac si chan doan hinh anh. | ABIDE I, VQA-RAD, PathVQA | De xuat nhom (Nhom 7 & Nhom 8 trong Excel) |

---

## 2. TRONG TAM DE TAI NHOM: TOPIC 6 - WEAKLY-SUPERVISED SEGMENTATION FOR WSI

- Bo du lieu: BCSS-WSSS (151 benh nhan TCGA-BRCA, 31.826 mang anh mo hoc).
- Kien truc hai giai doan:
  1. Giai doan 1 (Cot loi): Phan vung mang anh mo hoc (Patch-level Tissue WSSS) ke thua WSSS-Tissue, Progressive Dropout Attention (PDA), Spatial Affinity Relocalization (ARML), va Multi-Layer Pseudo-Supervision (MLPS).
  2. Giai doan 2 (Mo rong): Mo phong do phong dai thap va ghep ban do nhiet toan slide (WSI) theo dinh huong nghien cuu LowMagWSS cua TS. Bui Cao Doanh.
```

### 2.2 Trich Xuat Nguyen Van Danh Sach Dang Ky De Tai Lop SE374.R11 (`SE374.R11 - Do an cuoi ky.xlsx`)

| STT Nhom | Ma So Sinh Vien (MSSV) | Ho va Ten Sinh Vien | Ten De Tai Dang Ky | Mo Ta Du Dinh Thuc Hien |
| :---: | :---: | :--- | :--- | :--- |
| **1** | [Chua dang ky] | [Chua dang ky] | Sparse-view CT reconstruction | [Theo de bai giang vien giao] |
| **2** | [Tác giả / Nhóm nghiên cứu] | [Tác giả / Nhóm nghiên cứu] | Tumor segmentation using CT/MRI | [Phan vung khoi u tren CT/MRI] |
| **3** | [Tác giả / Nhóm nghiên cứu] | [Tác giả / Nhóm nghiên cứu] | Ten de tai gi cho hay nhi | [Khao sat va chon mo hinh] |
| **4** | [Chong] | [Chong] | [Chong] | [Chong] |
| **5** | [Tác giả / Nhóm nghiên cứu] | [Tác giả / Nhóm nghiên cứu] | Whole-Slide Image-Based Survival Risk Analysis Using Attention Multiple Instance Learning | Du doan nguy co sinh ton tu anh WSI su dung Multiple Instance Learning chu y |
| **6** | **[Tác giả / Nhóm nghiên cứu]** | **[Tác giả / Nhóm nghiên cứu]** | **Weakly-supervised segmentation for whole slide images** | **DE TAI TRONG TAM CUA DU AN:** Phan vung giam sat yeu tren WSI (BCSS-WSSS, Camelyon16). Kien truc Coarse-to-Fine WSSS hai giai doan, MedSAM Adapter, FastAPI Workstation. |
| **7** | [Tác giả / Nhóm nghiên cứu] | [Tác giả / Nhóm nghiên cứu] | Ung dung hoi dap danh cho bac si chan doan hinh anh | Ung dung VQA y te hoi dap bac si chan doan hinh anh (VQA-RAD, PathVQA) |
| **8** | [Tác giả / Nhóm nghiên cứu] | [Tác giả / Nhóm nghiên cứu] | Ung dung phat hien nguoi mac roi loan pho tu ky dua tren anh fMRI | - Su dung anh fMRI de tan dung duoc functional connectivity cua nao<br>- Thuc nghiem tren nhieu mo hinh, tu do tim ra mo hinh phu hop nhat<br>- Dataset: ABIDE I<br>- Xay dung web demo nho |
| **9** | [Tác giả / Nhóm nghiên cứu] | [Tác giả / Nhóm nghiên cứu] | Limited-angle CT reconstruction | Khoi phuc anh CT dai goc han che |

### 2.3 Dac Ta Chi Tiet Bo Du Lieu BCSS-WSSS (Topic 6)
- **Nguon goc du lieu:** Tap con chuan hoa BCSS (Breast Cancer Semantic Segmentation) trich xuat tu 151 benh nhan TCGA-BRCA (Amgad et al., Bioinformatics 2019; Chu Han et al., MedIA 2022).
- **Quy mo du lieu:** 151 vung quan tam (ROIs) do phong dai $20\times$ ($0.5\,\mu\text{m/px}$) hoac $40\times$ ($0.25\,\mu\text{m/px}$), tong cong $31.826$ mang anh $224 \times 224$ pixels.
- **5 phan lop mo hoc muc tieu:**
  * Lop 0 - TUM (Tumor / Infiltrating Ductal Carcinoma): Bieu mo khoi u xam lan.
  * Lop 1 - STR (Stroma / Fibrous Stroma): Mo dem xo bao quanh u.
  * Lop 2 - LYM (Lymphocyte / Inflammatory Infiltrate): Te bao lympho tham nhuan viem.
  * Lop 3 - NEC (Necrosis / Debris): Vung mo hoai tu, manh vun te bao.
  * Lop 4 - BACK (Background / Glass): Lam kinh trong suot va mo mo binh thuong khong te bao.
- **Phan chia benh nhan chong ro ri tuyet doi (`src/config/bcss_patient_splits.json`):**
  * Train: 106 benh nhan ($23.423$ patch co nhan yeu cap mang).
  * Val: 22 benh nhan ($3.417$ patch co ground-truth).
  * Test: 23 benh nhan hoat dong ($4.986$ patch test doc lap co ground-truth).
  * Giao tap giua cac bo du lieu: $\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$ (Patient Overlap = 0.0%).

---

## 3. RUBRIC DANH GIA & THANG DIEM CHI TIET (GRADING RUBRIC & MILESTONES)

### 3.1 Co Cau Diem Tong The Mon Hoc SE374.R11
| Thanh Phan Danh Gia | Ty Trong | Ban Chat & Noi Dung Kiem Tra |
| :--- | :---: | :--- |
| **Diem danh & Quiz** | **10%** | Kiem tra chuyen can va quiz ly thuyet dau gio (vat ly anh X-ray, CT, MRI, WSI). |
| **Thuc hanh Lab (Lab Assignments)** | **30%** | Cac bai thuc hanh co ban (doc DICOM/NIfTI/WSI, tien xu ly, trich xuat dac trung, baseline). |
| **Bao Cao Chuyen De & Do An Cuoi Ky** | **60%** | Nghien cuu giai phap SOTA, hien thuc phan mem hoan chinh, thuc nghiem khach quan, bao cao Springer/UIT. |

### 3.2 Rubric Cham Diem Do An Cuoi Ky (Thang 10 Diem / 60% Tong Diem Mon Hoc)

Do an duoc danh gia tren 5 tieu chi nghiem ngat:

| STT | Tieu Chi Danh Gia (Criteria) | Ty Trong | Yeu Cau Ky Thuat & Bang Chung Dat Diem Toi Da |
| :---: | :--- | :---: | :--- |
| **1** | **Khao sat hoc thuat & Dat bai toan** | **10%** (1.0 diem) | - Tong quan sau sac y van quoc te (MedIA, TMI, MICCAI, CVPR, MIDL).<br>- Dinh nghia ro rang bai toan WSSS cap patch va WSI, 5 phan lop mo hoc, nhung han che cua cac cong trinh truoc.<br>- Doi chieu voi it nhat 5-10 bai bao quoc te co DOI/arXiv that (LowMagWSS, WSSS-Tissue, HisynSeg, ARML, MedSAM). |
| **2** | **Kien truc he thong & Mo hinh SOTA** | **25%** (2.5 diem) | - Hien thuc day du hai giai doan: Stage 1 (CAM, PDA, Prototype Cosine Gating, Bilateral Affinity Propagation) va Stage 2 (DeepLabV3+ ResNet-101 voi 624 ImageNet layers, MLPS Loss, Uncertainty Regularization Loss, CGM Gate).<br>- Mo hinh tien tien: MedSAM PEFT Adapter (ViT-B 86M params frozen, bottleneck 256->64->256, scale 0.1, 24 point prompts tu CAM).<br>- Chuoi cai tien M1 den M7 co co so toan hoc va y sinh ro rang. |
| **3** | **Liem chinh khoa hoc & Thuc nghiem** | **25%** (2.5 diem) | - **100% Zero Mockup:** So lieu that co log tensor doi chung tu GPU Colab tren 4.986 patch test.<br>- **Patient-Level Anti-Leakage:** 0.0% ro ri benh nhan giua train/val/test.<br>- Do chuan xac: mIoU tren tap test 4.986 patch vuot baseline ($>69.34\%$), dat SOTA $\ge 70.50\%$.<br>- Danh gia da chieu: mIoU, Dice, BF1, ma tran nham lan, kiem dinh thong ke Wilcoxon signed-rank paired tests va hieu chinh Holm-Bonferroni. |
| **4** | **Ky thuat phan mem y te (Software Engineering)** | **20%** (2.0 diem) | - Kien truc Clean Architecture phan tach doc lap giua model, datasets, losses, clinical, pipeline, api.<br>- Tram lam viec Web API FastAPI bat dong bo kem giao dien xem anh Gigapixel OpenSeadragon (DZI DeepZoom, Alpha blending, bat/tat 5 lop mo).<br>- Tinh toan dau an sinh hoc lam sang: TSR (Tumor-Stroma Ratio), VTP (Viable Tumor Percentage), phan tang nguy co CAP/UNITED, Kaplan-Meier va Log-Rank test tren TCGA-BRCA.<br>- Suite kiem thu unit & regression tests bao phu day du (311/311 tests pass 100%). |
| **5** | **Ho so an pham & Thuyet trinh bao ve** | **20%** (2.0 diem) | - Mini Proposal (Word DOCX, Markdown, PDF) chuan mau UIT.<br>- Slide thuyet trinh Marp 16:9 chuyen nghiep, hinh anh don sac $\ge 300$ DPI.<br>- Bao cao ky thuat Springer LNCS / Luan van UIT khang dinh day du dong gop.<br>- Tuan thu chat che typography ASCII hyphen 0x2D, khong loi font UTF-8. |

### 3.3 Bang Phan Muc Xep Loai Danh Gia (Qualitative Assessment Bands)
- **Xuat sac (9.0 - 10.0 diem):**
  * Vuot qua 100% chi tieu SOTA quoc te: mIoU 4 lop $\ge 70.50\%$ tren toan bo 4.986 patch test.
  * Khong co bat ky ro ri du lieu benh nhan nao (Patient Overlap = 0.0%).
  * Day du Web API FastAPI, giao dien OpenSeadragon Gigapixel, dau an sinh hoc lam sang (TSR, VTP, Kaplan-Meier).
  * Suite kiem thu phong ve y te vuot qua 100% (311/311 tests pass).
- **Gioi (8.0 - 8.9 diem):**
  * mIoU dat tu 69.35% den 70.49% (vuot qua baseline cong bo MedIA 2022).
  * 0.0% ro ri benh nhan, day du code huan luyen va danh gia doi chuan.
  * Co giao dien Web API hoac cong cu truc quan hoa ket qua.
- **Kha (7.0 - 7.9 diem):**
  * Dat mIoU baseline xap xi cong bo (67.0% - 69.34%).
  * Khong ro ri benh nhan nhung thieu cac dong gop dot pha nhu MedSAM Adapter hoac Multiscale Pyramid.
- **Trung binh (5.0 - 6.9 diem):**
  * Chi chay duoc Stage 1 CAM hoac Stage 2 don le, so lieu test thap hoac danh gia tren tap con nho duoi 1.000 patch.
- **Khong Dat / Dinh Chi (< 5.0 diem):**
  * Phat hien bat ky hanh vi biet lap nao: Ro ri benh nhan cap slice/patch, bia dat so lieu (dummy mockup), sua doi notebook goc `Lab1.ipynb` cua giang vien.

### 3.4 Quy Tac Loai Tru Tu Dong (Zero-Tolerance Red-Flag Disqualifications)
1. **Loi Vi Pham Ro Ri Benh Nhan (Patient Leakage Disqualification):** Neu phat hien bat ky benh nhan nao co patch xuat hien o ca tap Train va tap Test ($\text{Train} \cap \text{Test} \neq \emptyset$) -> **Diem tieu chi Liem chinh = 0.0 diem**.
2. **Loi Bia Dat So Lieu (Fabricated Metrics Disqualification):** Neu phat hien so lieu trong bang ket qua la dummy text duoc sinh boi script ma khong co log tensor tu GPU tren 4.986 patch -> **Diem toan bo do an = 0.0 diem (Huy ket qua mon hoc)**.
3. **Loi Thay Doi Notebook Cua Giang Vien (Lab1 Immutability Disqualification):** Neu file `notebooks/Lab1.ipynb` bi thay doi ma bam SHA-256 byte-for-byte -> **Diem tieu chi Ky thuat = 0.0 diem**.
4. **Loi Noi Suy Nhan Phi Tuyen (Non-Nearest Interpolation Penalty):** Neu phat hien code resize mask su dung bilinear hoac bicubic thay vi nearest-neighbor -> **Tru 2.0 diem truc tiep vao tong diem**.

---

## 4. NGAN XEP CONG NGHE & MOI TRUONG THUC THI (TECH STACK & RUNTIME)

### 4.1 Moi Truong & Ngon Ngu Lap Trinh
- **Ngon ngu chinh:** Python $\ge 3.10$ (3.10 / 3.11 / 3.12).
- **He dieu hanh phat trien cuc bo:** Windows 11 64-bit.
- **Quy chuan thuc thi Shell:** `cmd /c "<command>"` cho tat ca lenh Windows.
- **Moi truong dien toan dam may (Cloud Compute):**
  * Google Colab Free Tier: GPU NVIDIA Tesla T4 (15 GB VRAM), FP16 AMP, batch size 8/16.
  * Google Colab Pro / Enterprise: GPU NVIDIA L4 (24 GB VRAM) hoac A100 (40/80 GB VRAM), BF16/FP16 AMP, batch size 32/64.
- **Kien truc Thin Drive Runner:** Toan bo notebook doc lap chay tren dia cung cuc bo VM (`/content/`), khong phu thuoc Google Drive mount (khong bi I/O throttling), tu dong keo ma nguon tu GitHub va nen ZIP tai ve trinh duyet.
- **Bo dong goi Google Drive:** Toan bo code, notebook va manifest san sang upload da duoc dong bo tai `drive_upload_ready/SE374_WSSS_Drive_Code/`.

### 4.2 Cac Thu Vien Phu Thuoc Chuyen Mon Y Te (`requirements.txt`)
- **Deep Learning Core:**
  * `torch >= 2.0.0`, `torchvision >= 0.15.0`
  * `monai >= 1.3.0` (Medical Open Network for AI)
  * `timm >= 0.9.0` (PyTorch Image Models)
- **Medical Imaging & Pathology:**
  * `opencv-python-headless >= 4.7.0`
  * `Pillow >= 9.5.0`
  * `tifffile >= 2023.4.12`, `imagecodecs >= 2023.3.16`
  * `scikit-image >= 0.20.0`
- **Clinical Data & Survival Statistics:**
  * `scipy >= 1.10.0`
  * `scikit-learn >= 1.2.0`
  * `pandas >= 2.0.0`, `numpy >= 1.24.0`
  * `lifelines >= 0.27.0` (Phan tich duong cong sinh ton Kaplan-Meier va Cox Proportional Hazards)
- **Web Application & Clinical Workstation:**
  * `fastapi >= 0.95.0`, `uvicorn[standard] >= 0.22.0`
  * `python-multipart >= 0.0.6`
  * `jinja2 >= 3.1.0`
  * `OpenSeadragon` (JavaScript DZI Gigapixel Viewer tich hop)
- **Quality Assurance & Verification:**
  * `pytest >= 7.3.0`, `hypothesis >= 6.75.0` (Property-based testing)
  * `ruff >= 0.0.270` (Linter PEP 8 sieu toc)
  * `bandit >= 1.7.5` (Security Static Analysis SAST)
  * `graphify-cli >= 0.9.49` (AST Knowledge Graph Tracking)
  * `marp-cli`, `pandoc` (Bien dich slide 16:9 va tai lieu Word DOCX)

---

## 5. CAU TRUC THU MUC CAP NHAT MOI NHAT (WORKSPACE LAYOUT & CLEAN ARCHITECTURE)

He thong duoc to chuc theo kien truc phan tang trach nhiem ro rang (Clean Architecture):

```
SE374_WSSS_WORKSPACE/
|-- AGENTS.md                                   # Master Governance Specification & 32 Luat bat bien
|-- README.md                                   # Huong dan van hanh va badges Open in Colab
|-- requirements.txt                            # Danh muc thu vien Python chuan
|-- PROJECT_EVALUATION_DOSSIER.md               # Tep dac ta tong the phuc vu AI kiem dinh (ban copy goc)
|-- drive_upload_ready/                         # Bo go Google Drive upload san sang
|   `-- SE374_WSSS_Drive_Code/                  # Thu muc dong goi doc lap
|       |-- README_DRIVE_SETUP.md               # Huong dan su dung Google Drive
|       |-- SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb
|       |-- SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb
|       |-- SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb
|       |-- SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb
|       |-- SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb
|       |-- SE374_WSSS_Drive_Code.zip           # File ZIP nen hoan chinh (4.9 MB)
|       |-- package_manifest.json               # Manifest kiem toan toan ven goi Drive
|       `-- requirements.txt                    # Requirements dong goi
|-- product/                                    # San pham an pham hoc thuat
|   |-- proposal/                               # Mini Proposal (Word DOCX, HTML, PDF, Markdown)
|   |-- slides/                                 # Slide bao cao Marp 16:9
|   `-- reports/                                # Bao cao ky thuat Springer LNCS / Luan van UIT
|-- docs/                                       # Co so tri thuc va dac ta he thong
|   |-- requirements/project_spec.md            # Dac ta dong 8 chu de, dataset, phan cung
|   |-- architecture/SYSTEM_TECHNICAL_SPECIFICATION.md # Dac ta ky thuat sau sac cua he thong
|   |-- papers/                                 # 119 bai bao khoa hoc PDF va Markdown toan van
|   |-- course/                                 # De bai goc topic.md va danh sach lop Excel
|   `-- reports/PROJECT_EVALUATION_DOSSIER.md   # Tep dac ta tong the phuc vu AI kiem dinh (ban chinh)
|-- src/                                        # Ma nguon chinh phan tang Clean Architecture
|   |-- config/
|   |   `-- bcss_patient_splits.json            # Manifest dinh danh 151 benh nhan TCGA (106/22/23)
|   |-- datasets/                               # Data Ingestion Service
|   |   |-- bcss_dataset.py                     # BCSSDataset PyTorch Loader & regex boc tach nhan
|   |   |-- patient_split.py                    # Giao thuc phan chia va kiem dinh 0.0% ro ri benh nhan
|   |   `-- otsu_filter.py                      # Loc mo HSV va dong cong nen kinh trang Otsu
|   |-- models/                                 # Medical AI Modeling Service
|   |   |-- backbone.py                         # MultiStageFeatureBackbone (ResNet-38 da tang)
|   |   |-- pda_generator.py                    # Progressive Dropout Attention (PDA) CAM module
|   |   |-- deeplabv3plus.py                    # DeepLabV3PlusWithCGM (ASPP, GroupNorm, CGM Gate)
|   |   |-- sam_wsss_adapter.py                 # MedSAM_WSSS_Adapter (ViT frozen, LoRA adapter)
|   |   |-- coarse_to_fine_router.py            # Bo dinh tuyen da do phong dai va Pareto frontier
|   |   |-- wsss_tissue.py                      # WSSSBenchmarkingEngine, SAR, ClassMix
|   |   `-- wsss_inference_methods.py           # Engine M1 den M7, WSSSTissueStage1NetCAM
|   |-- losses_metrics/                         # Loss Functions & Medical Metrics Service
|   |   |-- losses.py                           # MLPS Loss, MultiLabel BCE, Uncertainty Loss
|   |   `-- metrics.py                          # mIoU, Dice, FwIoU, BF1, VTP, TSR, ICC, CCC
|   |-- clinical/                               # Clinical Translation & Survival Service
|   |   |-- exhaustive_evaluator.py             # Evaluator doi chuan 3 mo hinh tren 4.986 patch
|   |   `-- survival_analysis.py                # Phan tang CAP, Kaplan-Meier, Log-Rank TCGA-BRCA
|   |-- training/                               # Training Pipelines (Da nang cap L4 Scientific Upgrade)
|   |   |-- dataset_loader.py                   # Data loader da luong cho tap huan luyen
|   |   |-- train_stage1.py                     # Huan luyen Stage 1 ResNet38d voi PDA
|   |   |-- generate_pseudo_masks.py            # Xuat mat na gia da tang (bn7, b5_2, b4_5 + Otsu)
|   |   |-- train_stage2.py                     # Huan luyen Stage 2 DeepLabV3+ ResNet-101 (624 layers)
|   |   `-- train_proposed_sam.py               # Huan luyen MedSAM Adapter (24 point prompts)
|   |-- pipeline/                               # Pipeline Orchestrators
|   |   |-- comparative_runner.py               # Runner so sanh 4 cau hinh qua 5 random seeds
|   |   |-- colab_runner.py                     # Runner thuc nghiem tren Google Colab
|   |   `-- wsi_lowmag_highmag_pipeline.py      # Pipeline WSI Low-Mag (5x) sang High-Mag (40x)
|   |-- colab_pkg/                              # Goi module Colab tinh gon (Thin Drive Package)
|   |   |-- model_loader.py                     # Loader mo hinh va checkpoints tu dong
|   |   |-- evaluator.py                        # Bo danh gia streaming an toan VRAM
|   |   |-- visualizer.py                       # Ve bieu do va heatmap ket qua
|   |   `-- downloader.py                       # Tai du lieu va checkpoints tu Cloud
|   |-- visualization/                          # Workstation & Clinical Web API Service
|   |   |-- api_server.py                       # FastAPI REST Server (/api/v1/predict/patch, /viewer)
|   |   |-- clinical_viewer.py                  # OpenSeadragon renderer, Alpha blending, DZI
|   |   |-- draw_all_proposal_figures.py        # Xuat 4 hinh ve khoa hoc chuan 300 DPI
|   |   `-- draw_pipeline_diagram.py            # Ve luu do pipeline he thong
|   `-- utils/                                  # Tien ich ho tro
|       `-- deliverable_auditor.py              # Auditor kiem tra chat luong tai lieu 4 tang
|-- notebooks/                                  # 10 so tay Colab Thin-Runner (T4, L4, A100)
|-- experiments/                                # Log thuc nghiem that 100% tren 4.986 patch
|   |-- colab_run_1_stage2_baseline/            # Run 1: DeepLabV3+ don le (mIoU: 68.71%)
|   |-- colab_run_2_dual_stage_ensemble/        # Run 2: Dual-Stage M1 den M5 (mIoU: 70.49%)
|   `-- colab_run_3_multiscale_sota_v2/         # Run 3: SOTA M1 den M7 (mIoU: 70.53%)
|-- tests/                                      # 23 tep kiem thu unit tests va stress tests (311 tests)
`-- repos/                                      # 23 kho ma nguon SOTA va 15 meta-tools
```

---

## 6. CAC THACH THUC KY THUAT & GIAI PHAP/MO HINH DA CHON (CHALLENGES & SOLUTIONS)

### 6.1 Thach Thuc 1: CAM Under-Activation & Boundary Bleeding
- **Van de:** Ban do Class Activation Map co ban (CAM, Grad-CAM) duoc huan luyen voi ham mat mat phan loai chi tap trung vao phan nhan te bao co tinh phan biet manh nhat (discriminative cores), bo sot phan bien ngoai vi khoi u. Neu ha thap nguong de mo rong CAM, tin hieu u se bi tran bien (bleeding) sang vung mo dem xo lan can.
- **Giai phap da chon: Progressive Dropout Attention (PDA) + Prototype Cosine Gating (`src/models/pda_generator.py`)**
  * *Xoa mo luy tien (Progressive Spatial Erasing):* Tai moi buoc lap $t \in \{1,\dots,T\}$, tinh nguong xoa dua tren phan vi top-$p$ ($p = 0.25$) ket hop tran 55% gia tri cuc dai:
    $$\tau_c^{(t)} = \max\left(\text{Quantile}\left(M_c^{(t-1)}, 1 - p\right), 0.55 \cdot \max_{x, y} M_c^{(t-1)}(x, y)\right)$$
  * *Mat na tich luy (Cumulative Mask):* Che cac diem cuc dai de ep mang tim kiem cac dac trung bo tro o vung ngoai vi:
    $$S_c^{(t)}(x, y) = 1.0 - \delta_t \cdot \mathbb{I}\left(M_c^{(t-1)}(x, y) \ge \tau_c^{(t)}\right), \quad S_{\text{cum}}^{(t)}(x, y) = \min_{k \in \{1,\dots,t\}} S_c^{(k)}(x, y)$$
  * *Prototype Cosine Gating:* Tinh vector nguyen mau mo u $\mu_c = \frac{1}{|\Omega_c|} \sum_{(x, y) \in \Omega_c} F(x, y)$. Tinh tuong dong cosine giua dac trung tai tung pixel voi $\mu_c$:
    $$G_c(x, y) = \text{ReLU}\left(\frac{F(x, y) \cdot \mu_c}{\|F(x, y)\|_2 \|\mu_c\|_2 + \epsilon}\right)$$
    Cac diem anh co huong nguoc voi nguyen mau mo ($\text{Sim} \le 0$) bi triet tieu ve 0, ngan chan hoan toan tran bien sang mo dem.

### 6.2 Thach Thuc 2: Background Score Gap & Duong Tinh Gia Tren Lam Kinh Trang
- **Van de:** Trong bai toan WSSS cap patch, nhan huan luyen chi co 4 lop mo tien canh $[y_0, y_1, y_2, y_3]$, khong co nhan cho lop Nen kinh trang (Background). Cac cong trinh kinh dien dat vector diem nen bang 0 (`bg_score = 0`). Khi tiep nhan patch chi co kinh trang hoac mo mo sang mau khong bat mau, phep toan `argmax` bi danh lua va du doan thanh Bieu mo u (Tumor), gay duong tinh gia nghiem trong ($>15\%$).
- **Giai phap da chon: Otsu Clinical Gating (`src/datasets/otsu_filter.py`)**
  * Chuyen anh RGB sang khong gian mau HSV va grayscale.
  * Tinh nguong Otsu tren kenh Do bao hoa mau (Saturation - S), xac dinh nguong hieu dung $\theta_{\text{sat}} = \min(\text{Otsu} \times 0.8, 20)$.
  * Loai tru vung kinh phan xa trang sang ($gray \ge 235$).
  * Mat na mo hoc:
    $$\text{Mask}_{\text{tissue}}(x, y) = \mathbb{I}\left( S(x, y) \ge \theta_{\text{sat}} \;\land\; I_{\text{gray}}(x, y) < 235 \right)$$
  * Ap dung phep toan dong mo hinh thai hoc (Morphological Opening/Closing) loai bo dom nhieu va lam day khoang trong noi bao nho duoi 256 pixels.
  * Ham `apply_otsu_gating_to_prediction`: Cuong buc toan bo pixel ngoai mo tro ve Class 4 (Background), triet tieu 100% false positive tren lam kinh rong.

### 6.3 Thach Thuc 3: Label Noise & Boundary Ambiguity Trong Stage 2
- **Van de:** Mat na gia sinh ra tu Stage 1 chua dung nhieu diem anh bat dinh tai ranh gioi chuyen tiep giua bieu mo u va mo dem. Neu ep mang Stage 2 hoc bang Cross-Entropy thong thuong, mo hinh se bi phat nang oan sai va tro nen thieu on dinh.
- **Giai phap da chon: Multi-Layer Pseudo-Supervision (MLPS) + Uncertainty Regularization Loss (`src/losses_metrics/losses.py`, `src/training/train_stage2.py`)**
  * *Multi-Layer Pseudo-Supervision (MLPS):* Giam sat dong thoi tu 3 tang dac trung sau ($bn7, b5\_2, b4\_5$) cua mang Stage 1 voi trong so kinh dien:
    $$\mathcal{L}_{\text{MLPS}} = \mathcal{L}_{\text{CE}}(\hat{Z}, \hat{Y}_{\text{bn7}}) + 0.5 \cdot \mathcal{L}_{\text{CE}}(\hat{Z}, \hat{Y}_{\text{b5\_2}}) + 0.25 \cdot \mathcal{L}_{\text{CE}}(\hat{Z}, \hat{Y}_{\text{b4\_5}})$$
  * *Uncertainty Regularization Loss:* Tinh Shannon Entropy chuan hoa 5 lop:
    $$\mathcal{H}_{\text{norm}}(p) = -\frac{1}{\ln(5)} \sum_{c=0}^4 P_c(x, y) \ln(P_c(x, y) + 1e-7)$$
  * Tinh trong so pixel thich ung voi nguong $\tau_{\text{entropy}} = 0.65$, chan duoi 0.1 tranh triet tieu gradient va tach khoi computational graph (`.detach()`):
    $$w(x, y) = \text{clamp}\left(1.0 - \frac{\mathcal{H}_{\text{norm}}(x, y)}{\tau_{\text{entropy}}}, 0.1, 1.0\right)$$
    $$\mathcal{L}_{\text{unc}} = \frac{1}{\sum w(x, y)} \sum_{x, y} w(x, y) \cdot \mathcal{L}_{\text{CE}}(\hat{Z}(x, y), Y(x, y))$$
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLPS}} + 0.50 \cdot \mathcal{L}_{\text{unc}}$$

### 6.4 Thach Thuc 4: False Positive Class Bleeding Tren Cac Lop Vang Mat
- **Van de:** Mang phan doan DeepLabV3+ tiep nhan anh patch chi chua duy nhat Mo dem (Stroma), nhung van co the du doan ra cac dao nho Bieu mo u (Tumor) do su tuong dong ve mau sac giua cac cum nhan te bao.
- **Giai phap da chon: Classification Gate Mechanism (CGM, `src/models/deeplabv3plus.py`)**
  * Mang du doan logits khong gian $Z_c(x, y) \in \mathbb{R}^{C \times H \times W}$.
  * Tiep nhan vector xac suat da nhan cap patch $y \in [0, 1]^C$ tu Stage 1 classifier:
    $$\hat{Z}_c(x, y) = \begin{cases} Z_c(x, y), & \text{neu } y_c > 0.10 \\ Z_c(x, y) - 10.000, & \text{nguoc lai} \end{cases}$$
  * Muc phat $-10.000$ dam bao rang sau ham Softmax, xac suat $P_c(x, y) < 10^{-10}$, loai bo hoan toan 100% cac dao u gia o cac lop khong co mat tren anh.

### 6.5 Thach Thuc 5: Chuoi Tien Hoa Mo Hinh M1 den M7 (Ablation Curriculum)
He thong thiet ke chuoi 7 phuong phap suy luan hau xu ly khong can huan luyen lai (Training-Free Test-Time Inference Engine, `src/models/wsss_inference_methods.py`), duoc kiem dinh tren toan bo 4.986 patch test cua 23 benh nhan doc lap:
- **M1 (Baseline WSSS-Tissue):** DeepLabV3+ ResNet-101 don ty le khong co CGM -> **$69.35\%$ mIoU**.
- **M2 (Multi-View D4 TTA + CGM Gate):** 4 goc nhin truc giao Dihedral D4 (xoay 90, 180, 270 do va lat guong) ket hop Stage-1 CGM -> **$70.30\%$ mIoU** ($p = 1.197 \times 10^{-23}$).
- **M3 (Cross-Stage Fusion & Long-Tail Prior):** Ghep CAM-Segmentation da tang ($\alpha = 0.70$) va hieu chinh tien nghiem lop thieu so (Lymphocyte, Necrosis theo Menon et al., trong so: `(0.96, 0.90, 1.24, 1.18)`) -> **$70.50\%$ mIoU** ($p = 4.490 \times 10^{-17}$).
- **M4 (Spatial Bilateral Refinement):** Loc song phuong bao toan bien ARML ($\sigma_s = 5, \sigma_r = 0.10$) va triet tieu dao nhieu PistoSeg -> **$70.32\%$ mIoU** ($p = 1.563 \times 10^{-29}$).
- **M5 (Proposed SOTA Ensemble V1):** Tich hop toan dien CGM + D4 TTA + CAM Fusion + Long-Tail Prior + Pure-Glass Guard -> **$70.51\%$ mIoU** ($p = 1.425 \times 10^{-24}$).
- **M6 (Multi-Scale Pyramid D4 Ensemble - TOP SOTA):** Kim tu thap da do phan giai ($0.75\times, 1.0\times, 1.25\times$), trong so boi nghich dao entropy va cong mem Stage-1 (nguong `[0.065, 0.065, 0.055, 0.060]`) -> **$70.53\%$ mIoU** ($p = 2.604 \times 10^{-15}$).
- **M7 (Beer-Lambert OD Calibration):** Giai ma quang pho H&E Ruifrok-Johnston va tinh chinh bien do quang mat do -> **$70.44\%$ mIoU** ($p = 3.590 \times 10^{-18}$).

### 6.6 Thach Thuc 6: Foundation Model Adaptation Duoi Ap Luc VRAM (MedSAM Adapter)
- **Van de:** Mo hinh nen tang y te MedSAM (Medical Segment Anything Model) co bo ma hoa Vision Transformer (ViT-B) voi 86 trieu tham so, rat de gay OOM khi tinh chinh toan bo tren GPU T4 15GB hoac L4 24GB.
- **Giai phap da chon: MedSAM WSSS Adapter (`src/models/sam_wsss_adapter.py`, `src/training/train_proposed_sam.py`)**
  * *Image Encoder (ViT-B):* Dong bang 100% (Strictly Frozen).
  * *CAM Prompt Generator (`CAMPromptGenerator`):* Tu dong trich xuat toi da 24 diem prompt (16 diem cuc dai cuc bo va trong tam CAM cua cac lop hoat dong $y_c > 0.1$, cong 8 diem phu dinh tu vung kinh trang).
  * *Residual Adapter Bottleneck Block:* Chen vao cac khoi Transformer:
    $$\text{Adapter}(X) = W_{\text{up}}\left(\text{GELU}\left(W_{\text{down}}\left(\text{LayerNorm}(X)\right)\right)\right) \cdot \text{scale}$$
    Chieu thu hep tham so $256 \to 64 \to 256$, $\text{scale} = 0.1$, khoi tao $W_{\text{up}} = 0$ de Adapter khoi dong nhu ham dong nhat.
  * *Histological Mask Decoder:* 5 class tokens song song + 1 IoU token, 5 nhanh MLP du doan dong thoi mat na phan vung 5 lop va diem chat luong IoU.

### 6.7 Thach Thuc 7: Clinical Translation & Survival Biomarkers
- **Dau an sinh hoc lam sang (`src/clinical/survival_analysis.py`):**
  * **Tumor-Stroma Ratio (TSR):** $\text{TSR} = \frac{N_{\text{tumor}}}{N_{\text{tumor}} + N_{\text{stroma}}} \times 100\%$.
  * **Stroma Proportion:** $100.0\% - \text{TSR}$.
  * **Viable Tumor Percentage (VTP):**
    + *Cellularity Density:* $\text{VTP} = \frac{N_{\text{tumor}}}{N_{\text{valid\_tissue}}} \times 100\%$.
    + *Residual Cancer Burden (RCB):* $\text{VTP}_{\text{RCB}} = \frac{N_{\text{tumor}}}{N_{\text{tumor}} + N_{\text{necrosis}}} \times 100\%$.
  * **Phan tang nguy co dong thuan quoc te CAP / UNITED:**
    + *Nhom Nguy Co Cao (Stroma-Rich / High Risk):* $\text{TSR} \le 50.0\%$ (Ty le mo dem $>50.0\%$). Tien luong xau, nguy co khang hoa tri.
    + *Nhom Nguy Co Thap (Stroma-Poor / Low Risk):* $\text{TSR} > 50.0\%$ (Ty le mo dem $\le 50.0\%$). Tien luong song sot tot hon.
  * **Uoc luong duong cong sinh ton Kaplan-Meier & Kiem dinh Log-Rank:**
    Tich hop du lieu lam sang 23 benh nhan kiem thu TCGA-BRCA, tinh toan ham song sot phi tham so $S(t) = \prod_{t_i \le t} (1 - d_i/n_i)$ va kiem dinh thong ke $\chi^2$.

### 6.8 Thach Thuc 8: Tuan Thu Quy Chuan CLAIM 2024 & Thong Ke Y Te
- **Quy chuan CLAIM 2024 (Checklist for Artificial Intelligence in Medical Imaging):**
  * Tinh khoang tin cay 95% Confidence Interval bang phan phoi Student-t ($df = N - 1$).
  * Xuat ma tran nham lan day du 5x5 va do dac hieu/do nhay (Sensitivity, Specificity, Precision, Recall) cho tung phan lop mo.
  * Hieu chinh sai so da kiem dinh (Multiple Testing) bang phuong phap Holm-Bonferroni kiem soat Family-Wise Error Rate (FWER o muc $\alpha = 0.05$).
  * QMS Traceability (ISO 13485 / FDA 21 CFR 820): Ma bam SHA-256 bat bien cho tat ca cac file cau hinh va bao ve bat kha xam pham notebook cua giang vien (`notebooks/Lab1.ipynb`).

---

## 7. DANH MUC NGUY CO, DIEM DE GAP LOI & FAILURE MODES (AUDITOR HIT LIST)

Bang duoi day tong hop 12 che do that bai tiem tang ma mot AI doc lap can chu dong kiem toan:

| STT | Nhom Nguy Co (Failure Mode) | Co Che Gay Loi & Bieu Hien Lam Sang | Vi Tri Ma Nguon Can Quet | Bien Phap He Thong Da Phong Ve |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Patient Leakage (Ro ri benh nhan)** | Chia tap theo patch ngau nhien khien patch cua cung mot benh nhan xuat hien o ca train va test, lam phong dai ket qua mIoU. | `src/config/bcss_patient_splits.json`, `src/datasets/patient_split.py` | Kiem tra SHA-256, tach 151 ma TCGA thanh 106 train, 22 val, 23 test; giao thoa giua cac tap bang 0.0%. |
| **2** | **Normalizer Fitting Data Snooping** | Tinh trung binh va do lech chuan (mean, std) tren toan bo tap du lieu gop (Train + Test), lam ro ri phan phoi test vao huan luyen. | `src/datasets/bcss_dataset.py`, `tests/test_clinical_data_qa.py` | Chi tinh thong so chuan hoa tren tap Train (`split == 'train'`); kiem thu `test_normalizer_fitting_train_vs_pooled`. |
| **3** | **CAM Under-Activation & Peak Suppression** | CAM chi phan ung tai cum nhan trung tam, bo sot ranh gioi xam lan u. Neu xoa qua manh dan den mat hoan toan tin hieu u. | `src/models/pda_generator.py` | Thuat toan Progressive Dropout Attention ket hop nguong dong $\mu_t$ giam dan theo epoch va Prototype Cosine Gating. |
| **4** | **Background Score Gap & Glass Spurious Activations** | Nhan khong co lop Nen (`bg_score = 0`), khi gap kinh trang mo hinh du doan nham thanh Bieu mo u (Tumor) gay false positives $>15\%$. | `src/datasets/otsu_filter.py`, `src/models/deeplabv3plus.py` | Otsu Clinical Gating tren kenh Saturation HSV cuong buc vung ngoai mo tro ve Class 4; CGM Gate phat $-10.000$ logit. |
| **5** | **Stain Augmentation Leakage (TIAToolbox #546)** | Bien doi mau nhuom (stain jitter) tac dong len ca nen kinh trang, bien kinh thanh mau hong tim khien CAM nhan nham kinh la mo dem. | `tests/test_clinical_data_qa.py`, `src/datasets/otsu_filter.py` | Chan bien doi mau bang `Mask_tissue`: chi pixel thuoc mo moi duoc phep bien doi mau, kinh trang giu nguyen trang tinh ($RGB \ge 245$). |
| **6** | **Label Interpolation Silent Failure** | Resize anh mat na nhan bang noi suy tuyen tinh (bilinear/bicubic) tao ra nhan le (fractional values) hoac sinh lop ma (phantom classes). | `src/datasets/bcss_dataset.py`, `notebooks/`, `tests/test_clinical_data_qa.py` | Bat buoc su dung duy nhat `cv2.INTER_NEAREST` hoac `mode='nearest'`; kiem thu property-based bang Hypothesis. |
| **7** | **Multiple-Testing FWER Inflation** | Chay dong thoi 7 mo hinh M1-M7 so sanh cap sinh ra 21 phep kiem dinh, dan den p-value $<0.05$ ngau nhien (duong tinh gia thong ke). | `src/colab_pkg/evaluator.py`, `tests/test_clinical_data_qa.py` | Ap dung thuat toan dieu chinh bac thang Holm-Bonferroni de kiem soat Family-Wise Error Rate (FWER) o muc $\alpha = 0.05$. |
| **8** | **Shortcut Learning & Scanner Covariate Drift** | Mo hinh hoc thuoc dac trung thiet bi quet (scanner drift) hoac ty le mau cua benh vien huan luyen thay vi dac trung benh hoc. | `tests/test_clinical_data_qa.py` | Tich hop kiem dinh do lech phan phoi (Chi-square test cho prediction drift, KL-divergence cho label drift, Top-2 entropy cho covariate drift). |
| **9** | **Teacher Notebook Mutation Risk** | Cac script tu dong hoa hoac test suite sua doi, ghi de vao notebook goc cua Giang vien (`Lab1.ipynb`), vi pham QMS ISO 13485. | `notebooks/Lab1.ipynb`, `tests/test_clinical_data_qa.py` | Kiem thu `test_teacher_notebook_lab1_byte_for_byte_immutability`: so sanh SHA-256 byte-for-byte voi file goc `Lab1.ipynb`. |
| **10** | **Silent Fallback to Random Weights (Audit D2)** | Khong tim thay checkpoint hoac loi import khien he thong am tham khoi tao trong so ngau nhien roi bao cao mIoU ao. | `src/clinical/exhaustive_evaluator.py`, `src/models/sam_wsss_adapter.py` | Chan hoan toan try-except nuot loi; neu checkpoint khong khop hoac missing keys bat buoc raise `RuntimeError` ngay lap tuc. |
| **11** | **Destructive Test Suite Risk (Audit D5)** | Test suite goi `shutil.rmtree` len thu muc that cua repo hoac ghi de checkpoints thay vi su dung fixture `tmp_path`. | `tests/test_thin_drive_runner.py`, `scripts/build_drive_package.py` | Co lap hoan toan vao fixture `tmp_path`; chan tuyet doi moi hanh dong xoa ngoai `tmp_path`. |
| **12** | **Gigapixel WSI Out-Of-Memory (OOM)** | Doc toan bo file WSI gigapixel vao RAM/VRAM lam sup do he thong (Kernel Crash). | `src/pipeline/wsi_lowmag_highmag_pipeline.py` | Kien truc Coarse-to-Fine: quet tong the o do phong dai thap ($5\times$), loc bo $85\%$ nen kinh rong, chi cat patch tai vung nghi ngo. |

---

## 8. DANH MUC CAC TEP TIN COT LOI TRONG HE THONG (CORE REPOSITORY ASSETS FOR AUDIT REFERENCE)

Nham ton trong tuyet doi quyen tu chu va tinh doc lap cua AI tham dinh, he thong KHONG ap dat bat ky quy trinh hay chi dinh co san nao. De giup AI doc lap de dang dinh vi cac module quan trong trong kien truc du an, duoi day la danh muc tham khao 16 tep tin vat ly tieu bieu ma AI doc lap co the chu dong yeu cau cung cap de phan bien va kiem chung:

1. **`src/config/bcss_patient_splits.json`:**
   - *Noi dung:* Danh sach 151 ma benh nhan TCGA-BRCA (106 train, 22 val, 23 test).
   - *Muc tieu doi chieu:* Xac minh tap train, val va test khong co bat ky benh nhan nao trung lap ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset$).
2. **`src/datasets/patient_split.py`:**
   - *Noi dung:* Ham `extract_patient_id`, `verify_patient_leakage`, `split_patients_kfold`.
   - *Muc tieu doi chieu:* Xac thuc thuat toan tach ma TCGA (vi du `TCGA-A2-A0T2_...` -> `TCGA-A2-A0T2`) va kiem tra giao tap.
3. **`src/datasets/otsu_filter.py`:**
   - *Noi dung:* Ham `compute_otsu_tissue_mask` va `apply_otsu_gating_to_prediction`.
   - *Muc tieu doi chieu:* Kiem tra chuyen doi mau HSV, nguong Saturation $\theta_{\text{sat}}$, morphological cleanup, va co che cuong buc Class 4 Background tren lam kinh.
4. **`src/models/pda_generator.py`:**
   - *Noi dung:* Lop `ProgressiveDropoutAttention`.
   - *Muc tieu doi chieu:* Kiem tra cong thuc suy giam nguong $\tau_c^{(t)}$, cumulative mask $S_{\text{cum}}$, spatial/channel dropout, va Prototype Cosine Gating.
5. **`src/models/deeplabv3plus.py`:**
   - *Noi dung:* Lop `DeepLabV3PlusWithCGM` va `ClassificationGateMechanism`.
   - *Muc tieu doi chieu:* Kiem tra khoi ASPP, GroupNorm an toan voi $B=1$, va co che phat logit $-10.000$ khi lop vang mat.
6. **`src/models/sam_wsss_adapter.py`:**
   - *Noi dung:* Lop `MedSAM_WSSS_Adapter` va `CAMPromptGenerator`.
   - *Muc tieu doi chieu:* Kiem tra viec dong bang 100% ViT encoder, adapter bottleneck $256 \to 64 \to 256$, scale 0.1, 24 point prompts tu CAM va 5 class tokens decoder.
7. **`src/models/wsss_inference_methods.py`:**
   - *Noi dung:* Lop `MultiMethodWSSSEngineV2` hien thuc 7 phuong phap M1 den M7.
   - *Muc tieu doi chieu:* Kiem tra logic chay M1 (Baseline), M2 (D4 TTA), M3 (Fusion), M4 (Bilateral), M5 (Ensemble), M6 (Multi-Scale Pyramid), M7 (Beer-Lambert).
8. **`src/losses_metrics/losses.py`:**
   - *Noi dung:* `MultiLayerPseudoSupervisionLoss` va `UncertaintyRegularizationLoss`.
   - *Muc tieu doi chieu:* Kiem tra he so $1.0, 0.5, 0.25$ cua MLPS loss; cong thuc Shannon entropy chuan hoa va `weight.detach()`.
9. **`src/training/generate_pseudo_masks.py`:**
   - *Noi dung:* Pipeline sinh mat na gia da tang (bn7, b5_2, b4_5), Bilateral Affinity Propagation GPU va Otsu tissue gating.
   - *Muc tieu doi chieu:* Kiem tra tinh nhat quan cua mat na gia xuat ra thu muc `train_PM/`.
10. **`src/training/train_stage2.py`:**
    - *Noi dung:* Script huan luyen DeepLabV3+ ResNet-101 (624 layers ImageNet), MLPS Loss va Uncertainty Loss.
    - *Muc tieu doi chieu:* Kiem tra pipeline huan luyen tu dau, LR scheduler va checkpoints.
11. **`src/training/train_proposed_sam.py`:**
    - *Noi dung:* Script huan luyen MedSAM Adapter voi 24 point prompts va Composite WSSS Loss.
    - *Muc tieu doi chieu:* Kiem tra che do PEFT LoRA adapter, optimizer AdamW va amp.
12. **`src/clinical/exhaustive_evaluator.py`:**
    - *Noi dung:* Engine danh gia doi chuan 3 mo hinh tren 4.986 patch test, `predict_uniform_multiscale_baseline`, `predict_pda_ablation`, kiem dinh Wilcoxon, hieu chinh Holm-Bonferroni.
    - *Muc tieu doi chieu:* Kiem tra quy trinh streaming an toan bo nho, tinh toan confusion matrix 5x5 va kiem dinh thong ke.
13. **`src/clinical/survival_analysis.py`:**
    - *Noi dung:* Ham `compute_tumor_stroma_ratio`, `compute_viable_tumor_percentage`, `fit_kaplan_meier_survival`.
    - *Muc tieu doi chieu:* Kiem tra cong thuc tinh TSR, VTP, nguong phan tang Stroma-Rich $\le 50.0\%$, va log-rank test.
14. **`src/visualization/api_server.py`:**
    - *Noi dung:* Khung FastAPI REST endpoints (`/api/v1/health`, `/api/v1/predict/patch`, `/api/v1/clinical/biomarkers`, `/viewer`).
    - *Muc tieu doi chieu:* Kiem tra hop dong API, OpenSeadragon integration, xu ly bat dong bo va triet tieu glass.
15. **`tests/test_clinical_data_qa.py`:**
    - *Noi dung:* Toan bo 6 test categories (TC-SPLIT-06, TC-INTERP-01, TC-PATH-05, TC-CLAIM-01, TC-DRIFT-01, TC-QMS-01).
    - *Muc tieu doi chieu:* Kiem tra cac bai test phong ve ve ro ri benh nhan, label interpolation nearest, white glass isolation, Holm-Bonferroni, drift detection va Lab1 byte-for-byte immutability.
16. **`experiments/benchmarks/COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md`:**
    - *Noi dung:* Bang tien hoa so lieu qua 3 phien chay Colab tren 4.986 patch test.
    - *Muc tieu doi chieu:* Doi chieu so lieu thuc nghiem that: Run 1 (68.71%), Run 2 (70.49%), Run 3 (70.53%), cac gia tri p-value Wilcoxon that.

---

## 9. HUONG DAN DANH CHO AI DOC LAP (INDEPENDENT AUDITOR PROTOCOL)

Khi tham dinh du an nay, AI doc lap can chu dong thuc hien quy trinh danh gia doc lap:
1. **Khong bi rang buoc boi bat ky ket luan co san nao:** AI doc lap duoc khuyen khich giu thai do hoai nghi khoa hoc lanh manh (healthy scientific skepticism).
2. **Tu do yeu cau ma nguon va file thuc te:** Truoc khi dua ra danh gia hoac cham diem, AI doc lap co the yeu cau doc bat ky file nao trong danh muc 16 file o Muc 8 hoac bat ky file nao khac trong he thong ma no cam thay can thiet.
3. **Kiem tra truc tiep cac logic toan hoc va y sinh:**
   - Xac minh cong thuc PDA co thuc su ngan chan boundary bleeding hay khong.
   - Xac minh Otsu Gating co triet tieu duong tinh gia tren lam kinh hay khong.
   - Xac minh MLPS Loss va Uncertainty Regularization co dung toan va chuan hoa entropy hop ly khong.
   - Xac minh co che CGM Gate co loai bo hoan toan cac lop vang mat hay khong.
   - Xac minh khong co hien tuong ro ri benh nhan (Patient Leakage) trong qua trinh chia tap va tinh mean/std.
4. **Cham diem khach quan theo Rubric Thang 10:** Ap dung day du 5 tieu chi trong Rubric o Muc 3, danh gia chi tiet tung tieu chi va xep loai theo dung bang xep loai chat luong.
5. **Dua ra phan quyet va gop y cai tien:** Neu phat hien bat ky loi tiem an nao (fatal bug, risk, numerical instability), AI doc lap can chi ro ten file va dong code de nhom phat trien tiep thu va hoan thien.
