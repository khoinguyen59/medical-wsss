# Scope: SE374 Colab WSSS Pipeline Notebook Refactoring & SOTA Integration

## Architecture
- Colab Notebook: `notebooks/SE374_Colab_WSSS_Pipeline.ipynb` (nbformat 4)
- Deep Learning Architecture: DeepLabV3+ with ResNet-38 / ResNet backbone and ASPP (matching `stage2_checkpoint_trained_on_bcss.pth`)
- Pretrained Weights: Google Drive ID `1cXm29ko8UevbaeW6YreSg-vuNaQiN_je` (`stage2_checkpoint_trained_on_bcss.pth`)
- Real Data Source: Google Drive ID `1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn` (`BCSS-WSSS.zip`) -> `/content/BCSS-WSSS/`
- Clinical Gating: Otsu saturation thresholding in HSV color space to force non-tissue background to Class 4 (Background)
- Evaluation: 5x5 confusion matrix, per-class IoU, global mIoU, mDice, FPS
- Visualization: 300 DPI publication-grade side-by-side plots (H&E, Ground Truth, DeepLabV3+ Prediction)
- Closed-Loop Packaging: Auto-generate `COLAB_EVALUATION_REPORT.md`, summary JSON, auto-zip into `/content/SE374_Colab_Results.zip`, and trigger `google.colab.files.download()`

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Automated Dependency & Weight Setup | Automated pip install, Drive download of `stage2_checkpoint_trained_on_bcss.pth` | M1 | R1 |
| 2 | DeepLabV3+ Architecture & Ingestion | Exact PyTorch DeepLabV3+ matching checkpoint state_dict keys, GPU forward pass | M1 | R1 |
| 3 | Automated BCSS-WSSS Data Pipeline | Download `BCSS-WSSS.zip`, patient-level indexing, dual mode (Sanity T4 vs Full L4) | M2 | R2 |
| 4 | Otsu Clinical Tissue Gating | HSV saturation Otsu gating to isolate tissue and extinguish glass false positives | M2 | R2 |
| 5 | Quantitative Clinical Metrics | 5x5 confusion matrix, per-class IoU (TUM, STR, LYM, NEC, BACK), mIoU, mDice, FPS | M3 | R3 |
| 6 | Publication-Grade Visualizations | 300 DPI H&E vs Pathologist GT vs DeepLabV3+ AI prediction comparisons | M4 | R4 |
| 7 | Closed-Loop Packaging & Download | Markdown and JSON evaluation reports, automatic zip, and `google.colab.files.download()` | M4 | R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Architecture & Checkpoint Ingestion | DeepLabV3+ PyTorch architecture matching checkpoint keys | none | DONE |
| 2 | Real Data Pipeline & Otsu Gating | Automated dataset download, patient split, and Otsu gating | none | DONE |
| 3 | Quantitative Evaluation & Metrics | Confusion matrix, per-class IoU, global mIoU, mDice, FPS | M1, M2 | DONE |
| 4 | Clinical Visualization & Packaging | Side-by-side 300 DPI plots, report generation, and auto-download | M3 | DONE |
| 5 | Adversarial Verification & Integrity Audit | Tri-party adversarial verification (syntax, weights, mockup, hyphens) | M1, M2, M3, M4 | DONE |

## Interface Contracts
### Checkpoint Ingestion Contract
- Expected input: `stage2_checkpoint_trained_on_bcss.pth`
- State dict keys: keys from official WSSS stage2 training (ResNet / ASPP / decoder)
- Output classes: 5 classes (0: TUM, 1: STR, 2: LYM, 3: NEC, 4: BACK)

### Real Data Pipeline Contract
- Input zip: `BCSS-WSSS.zip` unpacked into `/content/BCSS-WSSS/`
- Dual-mode selector: `EXECUTION_MODE = 'SANITY'` (150 real patches) vs `'FULL'` (4986 test patches)
- Data format: `(3, H, W)` normalized tensors, `(H, W)` ground truth masks
