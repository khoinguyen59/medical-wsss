# Coarse-to-Fine Weakly-Supervised Segmentation for Whole Slide Images

[![Course: SE374.R11](https://img.shields.io/badge/UIT-SE374.R11-0052cc.svg)](https://www.uit.edu.vn/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset BCSS-WSSS](https://img.shields.io/badge/Dataset-BCSS--WSSS-success.svg)](https://github.com/PathologyDataScience/BCSS)
[![Benchmark 70.53% mIoU](https://img.shields.io/badge/BCSS_mIoU-70.53%25-brightgreen.svg)](#benchmark-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Môn học:** SE374.R11 - Phát triển phần mềm xử lý ảnh y khoa (Khoa Kỹ thuật Phần mềm, UIT - ĐHQG-HCM)  
> **Tên đề tài chính thức:** `Coarse-to-Fine Weakly-Supervised Segmentation for Whole Slide Images`  
> **Chủ đề nghiên cứu:** Topic 6 - Weakly-supervised segmentation for whole slide images  

### Mô tả dự định đồ án (Project Scope & Objectives)
- Xây dựng quy trình phân đoạn mô học ung thư vú giám sát yếu 2 giai đoạn (Net_CAM + DeepLabV3+) chỉ từ nhãn mức ảnh trên bộ dữ liệu BCSS.
- Nghiên cứu và đối chuẩn các kỹ thuật cải tiến SOTA: Tăng cường suy luận (Multi-View D4 TTA), kim tự tháp đa tỷ lệ (Multi-Scale Pyramid) và hiệu chỉnh phân phối đuôi dài (Long-Tail Calibration).
- Mở rộng thành hệ thống lâm sàng 2 pha: Quét nhanh góc rộng Low-Mag (5x) để sàng lọc vùng nghi ngờ, kết hợp zoom sâu High-Mag (40x) trên tiêu bản Gigapixel TCGA-BRCA.
- Tự động trích xuất các dấu ấn sinh học lâm sàng (Viable Tumor Percentage - VTP, Tumor-Stroma Ratio - TSR) và tái cấu trúc bản đồ phân đoạn toàn tiêu bản.

A state-of-the-art, weakly-supervised semantic segmentation (WSSS) framework designed for gigapixel breast cancer histopathology (TCGA-BRCA) Whole Slide Images (WSI). The system implements an end-to-end dual-stage pipeline combining **Stage-1 Multi-Layer Class Activation Maps (Net_CAM)** and **Stage-2 DeepLabV3+**, augmented by an advanced inference engine featuring Dihedral D4 Test-Time Augmentation, Multi-Scale Pyramid Fusion, Stage-1 Soft-Gating, and Histopathology Prior Refinement.

---

## Key Highlights

- **Dual-Stage Architecture**: Seamlessly bridges Stage-1 Weakly-Supervised CAM generation (`ResNet38d Net_CAM`) with Stage-2 Dense Segmentation (`DeepLabV3+ ResNet-101`).
- **Benchmark-Setting Accuracy**: Achieves **70.53% 4-class mIoU** on the full 4,986 test patches across 23 independent patients on the official BCSS-WSSS benchmark, surpassing published SOTA baselines.
- **Patient-Level Anti-Leakage**: Strict zero-leakage patient-level stratification across all splits (no patch-level shuffling across patient boundaries).
- **Modular Thin-Runner Design**: Clean architecture separating core neural services (`src/colab_pkg/`, `src/models/`, `src/datasets/`) from Google Colab notebooks, enabling 1-click execution without monolithic code dumping.
- **Clinical WSI End-to-End Pipeline (Bonus Extension)**: Implements Low-to-High magnification guidance (`src/pipeline/wsi_lowmag_highmag_pipeline.py`) inspired by LowMagWSS (arXiv 2024), scanning macro slides at 5x to filter glass and saving 80% to 92% of dense compute before zooming into 40x candidate patches.

---

## Benchmark Results (Full BCSS-WSSS Test Set: 4,986 Patches, 23 Patients)

### 1. Evolution Across 3 Real Experimental Runs

All 3 runs evaluate the entire official BCSS-WSSS test partition (**4,986 patches, 224x224, 23 patients**) on a Google Colab NVIDIA Tesla T4 GPU:

| Run ID | Experiment Directory | Core Mechanism & Architecture | 4-Class mIoU (%) | 4-Class mDice (%) | 4-Class PA (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1** | `experiments/colab_run_1_stage2_baseline/` | Stage-2 DeepLabV3+ single checkpoint, Otsu gating | **68.71%** | 81.15% | 84.94% | 79.12% | 73.50% | 56.97% | 65.23% |
| **Run 2** | `experiments/colab_run_2_dual_stage_ensemble/` | Dual-Stage (`ResNet38d Net_CAM` + `DeepLabV3+`), M1-M5 ablation suite (FP32) | **70.49%** | 82.51% | 85.35% | 79.19% | 73.80% | 61.27% | 67.69% |
| **Run 3** | `experiments/colab_run_3_multiscale_sota_v2/` | **Thin Drive Runner v2, M1-M7 suite (FP16 AMP): Multi-Scale Pyramid D4 Ensemble** | **70.53%** | **82.54%** | **85.41%** | **79.37%** | **73.82%** | **61.33%** | **67.61%** *(M5: 67.67%)* |

### 2. Comparison with Published International SOTA (BCSS-WSSS Benchmark)

| Method / Published Work | Venue | 4-Class mIoU (%) | Tumor (TUM) IoU | Stroma (STR) IoU | Lymphocyte (LYM) IoU | Necrosis (NEC) IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistoSegNet** (Chan et al.) | ICCV 2019 | 46.92% | 56.92% | 57.53% | 32.31% | 40.91% |
| **SEAM** (Wang et al.) | CVPR 2020 | 61.41% | 74.52% | 68.09% | 50.19% | 52.84% |
| **WSSS-Tissue (Stage-1)** (Han et al.) | MedIA 2022 | 67.05% | 77.59% | 72.62% | 56.40% | 61.58% |
| **WSSS-Tissue (Stage-2 Paper)** (Han et al.) | MedIA 2022 | 69.34% | 78.44% | 73.31% | 59.98% | 65.62% |
| **LowMagWSS (40x Benchmark)** (Dung et al.) | arXiv 2024 | 69.34% | 78.44% | 73.31% | 59.98% | 65.62% |
| **PistoSeg (Mosaic + RFM)** (Fang et al.) | AAAI 2023 | 69.95% | 78.90% | 73.60% | 60.50% | 66.80% |
| **Ours - M1 Baseline (Run 2 & Run 3)** | This Repo | **69.35%** | 78.46% | 73.32% | 60.00% | 65.62% |
| **Ours - M5 Proposed SOTA Ensemble** | This Repo | **70.51%** | 79.24% | 73.79% | **61.33%** | **67.67%** |
| **Ours - M6 Proposed Multi-Scale Pyramid D4** | This Repo | **70.53%** | **79.37%** | **73.82%** | **61.33%** | 67.61% |

### 3. Detailed Ablation Study (Run 3 - 4,986 Test Patches)

| Method Key | Technical Description | mIoU 4-cls | mDice 4-cls | PA 4-cls | TUM IoU | STR IoU | LYM IoU | NEC IoU | Wilcoxon p-value |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | Baseline Single-Scale Forward Pass | 69.35% | 81.69% | 84.89% | 78.46% | 73.32% | 60.00% | 65.62% | 1.0000 |
| **M2** | + Dihedral D4 TTA & Stage-1 Classification Gate | 70.30% | 82.37% | 85.34% | 79.08% | 73.88% | 60.75% | 67.51% | 1.197e-23 |
| **M3** | + Cross-Stage CAM Fusion & Long-Tail Minorities | 70.50% | 82.51% | 85.36% | 79.23% | 73.78% | 61.31% | 67.66% | 4.490e-17 |
| **M4** | + Spatial Bilateral Refinement & Island Suppression | 70.32% | 82.38% | 85.34% | 79.09% | 73.89% | 60.77% | 67.53% | 1.563e-29 |
| **M5** | Proposed SOTA Full Pipeline V1 | 70.51% | 82.52% | 85.36% | 79.24% | 73.79% | **61.33%** | **67.67%** | 1.425e-24 |
| **M6** | **Multi-Scale Pyramid D4 (Entropy-Weighted) & Soft-Gating** | **70.53%** | **82.54%** | **85.41%** | **79.37%** | **73.82%** | **61.33%** | 67.61% | **2.604e-15** |
| **M7** | Multi-Scale D4 + H&E Beer-Lambert OD Prior Refinement | 70.44% | 82.47% | 85.36% | 79.25% | 73.81% | 61.21% | 67.48% | 3.590e-18 |

---

## Directory Structure

```text
medical-wsss/
├── .github/workflows/                 # Automated CI/CD (lint, unit tests, anti-leakage audit)
├── src/                               # Decoupled Modular Core Source
│   ├── colab_pkg/                     # Colab-ready modular package (Downloader, Loader, Evaluator, Visualizer)
│   ├── config/                        # Configuration & BCSS patient-level split definitions
│   ├── datasets/                      # BCSS Dataset loader, Otsu filtering & patient splitting
│   ├── models/                        # DeepLabV3+, Stage-1 Net_CAM & MultiMethodWSSSEngineV2
│   ├── losses_metrics/                # Loss functions and confusion matrix metrics (IoU, Dice, PA)
│   ├── pipeline/                      # End-to-end automated pipeline runner & WSI multi-mag coordinator
│   │   ├── colab_runner.py            # Automated Colab experiment runner
│   │   ├── comparative_runner.py      # Multi-method ablation orchestrator
│   │   └── wsi_lowmag_highmag_pipeline.py # Clinical WSI Low-to-High magnification pipeline
│   ├── utils/                         # Medical utilities, logging, normalization & auditing
│   └── visualization/                 # Clinical viewer, proposal figures & heatmap generators
├── notebooks/                         # Interactive Jupyter / Colab Notebooks
│   ├── SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb  # Primary Thin Runner notebook (Tesla T4)
│   ├── SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb  # Thin Runner notebook (NVIDIA L4)
│   ├── SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb # Thin Runner notebook (NVIDIA A100)
│   ├── SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb         # Bonus WSI Low-to-High clinical demo
│   ├── modular_drive/                 # Standalone drive-mounting notebook variants
│   └── history_runs/                  # Executed notebooks with full output logs for Runs 1, 2, 3
├── experiments/                       # Quantitative Results Across the 3 Real Runs
│   ├── colab_run_1_stage2_baseline/   # Run 1 results: Stage-2 baseline (68.71% mIoU)
│   ├── colab_run_2_dual_stage_ensemble/ # Run 2 results: Dual-stage M1-M5 suite (70.49% mIoU)
│   ├── colab_run_3_multiscale_sota_v2/ # Run 3 results: Multi-Scale Pyramid D4 SOTA (70.53% mIoU)
│   └── benchmarks/                    # Comparative analysis report and cross-run evolution
├── product/                           # Deliverables
│   ├── proposal/                      # Project proposal (DOCX, PDF, HTML, Markdown)
│   ├── slides/                        # Presentation slides (Marp HTML, PDF, PPTX)
│   └── reports/                       # Final technical report (SE374_FINAL_TECHNICAL_REPORT_WSSS.md)
├── docs/                              # Project Documentation
│   ├── course/                        # Course syllabus, topics & survey documents
│   ├── design/                        # System architecture & pipeline dataflow diagrams
│   ├── research/                      # WSSS literature review & roadmap
│   └── templates/                     # Academic LaTeX / Word templates
├── scripts/                           # Automation Scripts
│   ├── build_drive_package.py         # Automates packaging for Google Drive upload
│   ├── generate_colab_variants.py     # Generates GPU-specific notebook variants
│   └── generate_thin_colab_notebooks.py # Builds thin runner notebooks
├── data/                              # Sample clinical dataset for local unit testing
├── tests/                             # Unit tests & sanity suites
├── requirements.txt                   # Production Python dependencies
├── .gitignore                         # Strict exclusion for heavy binaries, weights, and caches
└── README.md                          # Project documentation
```

---

## Quickstart & Reproducibility

### Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/khoinguyen59/medical-wsss.git
cd medical-wsss

# Install dependencies
pip install -r requirements.txt

# Run unit tests and sanity checks
pytest tests/ -v
```

### Running on Google Colab GPU

1. Open [`notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`](notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb) in Google Colab.
2. Select a GPU runtime (Runtime > Change runtime type > **T4 GPU** or **L4 GPU**).
3. Run the cells sequentially:
   - **Cell 1**: Mount Google Drive and auto-install dependencies (`gdown`, `albumentations`).
   - **Cell 2**: Auto-download official BCSS test images and pretrained checkpoints (`stage1_checkpoint_trained_on_bcss.pth` and `stage2_checkpoint_trained_on_bcss.pth`).
   - **Cell 3**: Execute the M1-M7 evaluation pipeline with Mixed Precision FP16.
   - **Cell 4**: Display the comparative ablation study table and clinical visual comparison grids.
   - **Cell 5**: Export results, figures, CSVs, and markdown reports directly to Google Drive.

---

## Pretrained Model Weights

Official BCSS-trained checkpoints are hosted on Google Drive:
- **Stage-1 `ResNet38d Net_CAM`**: `118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc`
- **Stage-2 `DeepLabV3+ ResNet-101`**: `1cXm29ko8UevbaeW6YreSg-vuNaQiN_je`

Both weights are automatically verified and fetched with sha256 checksums by `src/colab_pkg/downloader.py`.

---

## License & Citation

This project is released under the [MIT License](LICENSE).

```bibtex
@misc{medical_wsss_2026,
  author = {Khoi, Nguyen Trong},
  title = {Medical-WSSS: Weakly-Supervised Semantic Segmentation on Histopathology Whole Slide Images},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/khoinguyen59/medical-wsss}}
}
```
