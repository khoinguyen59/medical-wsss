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
