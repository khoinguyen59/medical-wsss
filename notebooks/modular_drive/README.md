# SE374.R11 - Thin Drive Runner v2 Colab Notebook Suite

Thu muc nay chua bo 4 Google Colab Notebooks the he moi (**Thin Drive Runner v2**) danh cho de tai **Topic 6: Weakly-Supervised Segmentation for Whole Slide Images (LowMagWSS)** cua **Nhom 5** (Nguyễn Trọng Khôi - 23520790, Lê Nguyễn Thành Tài - 23521374; GVHD: TS. Nguyen Vinh Tiep).

## 1. Diem Khac Biet Cua Kien Truc Thin Drive Runner v2
- **Khong nhung code mo hinh khong lo vao o notebook:** Thay vi nhung hang nghin dong dinh nghia class (`WSSSTissueStage1NetCAM`, `DeepLabV3Plus`, `MultiMethodWSSSEngine`) truc tiep trong `.ipynb`, toan bo logic duoc dat trong thu muc code `src/` tren Google Drive (`src/colab_pkg/`, `src/pipeline/colab_runner.py`, `src/models/wsss_inference_methods.py`).
- **Moi o code cuc ky ngan gon (`<= 25` dong/o):** De doc, de bao tri, khong bao gio bi loi cu phap do sua nham code dai trong Colab.
- **Tich hop 7 Phuong Phap SOTA & Ablation (`MultiMethodWSSSEngineV2`):**
  1. `M1_Baseline_SingleScale`: Stage-2 DeepLabV3+ ResNet-101 Single-Scale Baseline (`Is_GM=False`)
  2. `M2_MultiView_D4_TTA`: Stage-1 ResNet38d Net_CAM Classification Gate (`y_cls > 0.1`) + Dihedral D4 TTA
  3. `M3_LongTail_Prior_Calibration`: Stage-1 CGM + Cross-Stage CAM Fusion + Long-Tail Prior Calibration
  4. `M4_Spatial_Bilateral_Refinement`: Stage-1 CGM + D4 TTA + PistoSeg/ARML Bilateral Spatial Refinement
  5. `M5_Proposed_SOTA_Ensemble`: Proposed V1 SOTA Ensemble (CGM + CAM Fusion + D8 TTA + Prior Calib + Spatial Refinement + Glass Guard)
  6. `M6_MultiScale_Pyramid_D4_Ensemble`: Multi-Scale Pyramid (`0.75x, 1.0x, 1.25x`) Inverse-Entropy D4 + Adaptive Soft-Gating
  7. `M7_Proposed_Ultra_SOTA_Curriculum`: **Proposed V2 Ultra-SOTA Curriculum (Multi-Scale D4 + Adaptive Soft-Gate + Entropy CAM Fusion + H&E Beer-Lambert Optical Density Calib + Multi-Stride Edge-Guided Refinement)**

## 2. Danh Sach 4 Notebook Theo Cau Hinh GPU

| File Notebook | GPU Muc Tieu | Che Do Danh Gia | Precision | Thoi Gian Du Kien |
| :--- | :---: | :---: | :---: | :---: |
| `SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb` | **NVIDIA T4 (16GB)** | Fast Sanity 150 anh | FP16 (`float16`) | ~1.5 - 2.5 phut |
| `SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb` | **NVIDIA T4 (16GB)** | Full 4,986 anh | FP16 (`float16`) | ~25 - 35 phut |
| `SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb` | **NVIDIA L4 (24GB)** | Full 4,986 anh | BF16 + TF32 | ~10 - 14 phut |
| `SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb` | **NVIDIA A100 (40/80GB)** | Full 4,986 anh | BF16 + TF32 | ~5 - 8 phut |

## 3. Huong Dan Su Dung Tren Google Drive + Google Colab (3 Buoc)

1. **Dong goi thu muc code de upload len Google Drive:**
   ```cmd
   cmd /c "python scripts/build_drive_package.py"
   ```
   Lenh tren tao thu muc `drive_upload_ready/SE374_WSSS_Drive_Code/` va file nen `drive_upload_ready/SE374_WSSS_Drive_Code.zip` chua day du `src/` va 4 notebook Thin Runner v2.
2. **Upload len Google Drive:**
   Upload thu muc `SE374_WSSS_Drive_Code` (hoac file `SE374_WSSS_Drive_Code.zip`) len `/content/drive/MyDrive/SE374_WSSS_Project/` hoac `/content/drive/MyDrive/`.
3. **Mo Notebook va bam `Runtime -> Run all`:**
   Chon 1 trong 4 file `*_ThinDriveRunner_v2.ipynb` phu hop voi GPU tren Colab va chay tu tren xuong duoi.

## 4. Bao Toan Nguyen Ven 100% Cac Notebook Cu & Lich Su 5 Buoc Nghien Cuu (Run 1 -> Run 5)

- **5 Notebook Standalone Goc (`notebooks/` & `SE374_WSSS_Drive_Code/notebooks/legacy_standalone_v1/`):**
  - `SE374_Colab_WSSS_Pipeline.ipynb`, `SE374_Colab_WSSS_T4_Sanity150.ipynb`, `SE374_Colab_WSSS_T4_Full4986.ipynb`, `SE374_Colab_WSSS_L4_Full4986.ipynb`, `SE374_Colab_WSSS_A100_Full4986.ipynb` (va ban 3.24 MB tai thu muc goc `/SE374_Colab_WSSS_T4_Full4986.ipynb`) duoc giu nguyen 100% kem day du cell outputs va hinh anh 300 DPI.
- **5 Notebook Lich Su Doi Chung Tung Buoc Nghien Cuu (`notebooks/history_runs/` & `SE374_WSSS_Drive_Code/notebooks/history_runs/`):**
  1. `Run1_20261019_Naive_HSV_Heuristic_Sanity150.ipynb` (`mIoU_4cls = 9.13%`, `mIoU_5cls = 9.78%`)
  2. `Run2_20261020_0656UTC_DeepLabV3Plus_ImageNetNorm_OtsuAggressive_Sanity150.ipynb` (`mIoU_4cls = 5.74%`, `mIoU_5cls = 7.59%`)
  3. `Run3_20261020_1008UTC_DeepLabV3Plus_UnitRangeNorm_Fixed_Sanity150.ipynb` (`mIoU_4cls = 72.39%`, `mIoU_5cls = 69.99%`)
  4. `Run4_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb` (`mIoU_4cls = 68.71%`, `mIoU_5cls = 67.36%`)
  5. `Run5_20261020_1641UTC_DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4_Full4986.ipynb` (`M1 = 69.34% -> M5 = 70.49%` 4 lop / `76.38%` 5 lop)
- **Bao Cao Doi Chung Chi Tiet (`experiments/benchmarks/COLAB_5_RUNS_EVOLUTION_COMPARISON.md`):** Luu day du bang doi sanh va ma tran nham lan cua ca 5 lan chay.

