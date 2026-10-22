# Đối Chiếu 3 Phiên Thực Nghiệm Thật Trên Toàn Bộ Tập Dữ Liệu BCSS-WSSS (4.986 Ảnh Test, 23 Bệnh Nhân)

> **Dự án:** Medical WSSS - Weakly-Supervised Semantic Segmentation on Histopathology Whole Slide Images  
> **Bộ dữ liệu chuẩn:** BCSS-WSSS (Breast Cancer Semantic Segmentation - TCGA-BRCA)  
> **Quy mô tập Test chuẩn mực:** **4.986 ảnh patch (224x224)** thuộc **23 bệnh nhân độc lập** (phân tách theo Patient ID để chống rò rỉ dữ liệu).  
> **Phần cứng thực nghiệm:** Google Colab GPU NVIDIA Tesla T4 (16GB VRAM).

---

## 1. Bảng Tổng Hợp 3 Lần Chạy Thật Trên Toàn Bộ 4.986 Ảnh Test

| Phiên Chạy Thật (Run ID) | Thư Mục Lưu Trữ | Kiến Trúc & Kỹ Thuật Đột Phá | mIoU 4 Lớp (%) | mDice 4 Lớp (%) | PA 4 Lớp (%) | mIoU 5 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | BACK IoU (%) | Tốc Độ (FPS) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lần 1 (Baseline Full)** | `experiments/colab_run_1_stage2_baseline` | Stage-2 DeepLabV3+ ResNet-101 đơn lẻ (chưa tích hợp Stage-1 CGM), Otsu Gating | **68.71%** | **81.15%** | **84.94%** | 67.36% | 79.12% | 73.50% | 56.97% | 65.23% | 70.26% | 23.38 |
| **Lần 2 (Dual-Stage M1-M5)** | `experiments/colab_run_2_dual_stage_ensemble` | Dual-Stage (`ResNet38d Net_CAM` + `DeepLabV3+`), 5 Methods (`M1..M5`, FP32): Cổng phân loại CGM + Multi-View TTA + Long-Tail Prior + Pure-Glass Guard | **70.49%** | **82.51%** | **85.35%** | 76.38% | 79.19% | 73.80% | 61.27% | 67.69% | 99.93% | 5.77 |
| **Lần 3 (Thin Runner M1-M7 SOTA)** | `experiments/colab_run_3_multiscale_sota_v2` | **Thin Drive Runner v2 (`MultiMethodWSSSEngineV2`, FP16 AMP, 7 Methods `M1..M7`): Kim Tự Tháp Đa Tỷ Lệ D4 (`0.75x, 1.0x, 1.25x`) Entropy-Weighted + Cổng Mềm Stage-1** | **70.53%** | **82.54%** | **85.41%** | **76.41%** | **79.37%** | **73.82%** | **61.33%** | **67.61%** *(M5: 67.67%)* | **99.93%** | **4.91** *(7 methods)* |

---

## 2. Bảng Phân Tích Bóc Tách (Ablation Study) 7 Phương Pháp Trong Lần Chạy Thứ 3 (SOTA)

Đánh giá thực nghiệm trên toàn bộ 4.986 ảnh test thật của tập BCSS-WSSS:

| Phương Pháp (`Method_Key`) | Mô Tả Kỹ Thuật | mIoU 4 Lớp (%) | mDice 4 Lớp (%) | PA 4 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | BACK IoU (%) | Wilcoxon p-value vs M1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | Baseline WSSS-Tissue Single-Scale | 69.35% | 81.69% | 84.89% | 78.46% | 73.32% | 60.00% | 65.62% | 99.93% | 1.0000 |
| **M2** | + Đa Góc Nhìn Dihedral TTA & Stage-1 CGM | 70.30% | 82.37% | 85.34% | 79.08% | 73.88% | 60.75% | 67.51% | 99.93% | 1.197e-23 |
| **M3** | + Kết Hợp CAM Đa Giai Đoạn & Hiệu Chỉnh Đuôi Dài | 70.50% | 82.51% | 85.36% | 79.23% | 73.78% | 61.31% | 67.66% | 99.93% | 4.490e-17 |
| **M4** | + Tinh Chỉnh Biên Không Gian & Triệt Tiêu Đảo Nhiễu | 70.32% | 82.38% | 85.34% | 79.09% | 73.89% | 60.77% | 67.53% | 99.93% | 1.563e-29 |
| **M5** | Phương Pháp Đề Xuất Toàn Diện V1 (Full SOTA Pipeline) | 70.51% | 82.52% | 85.36% | 79.24% | 73.79% | **61.33%** | **67.67%** | 99.93% | 1.425e-24 |
| **M6** | **Kim Tự Tháp Đa Tỷ Lệ D4 (Entropy-Weighted) & Cổng Mềm** | **70.53%** | **82.54%** | **85.41%** | **79.37%** | **73.82%** | **61.33%** | 67.61% | **99.93%** | **2.604e-15** |
| **M7** | Multi-Scale D4 + H&E Beer-Lambert OD + Biên Đa Bước Nhảy | 70.44% | 82.47% | 85.36% | 79.25% | 73.81% | 61.21% | 67.48% | 99.93% | 3.590e-18 |

---

## 3. So Sánh Với Các Công Trình Khoa Học Quốc Tế Công Bố Trên Cùng Tập Test BCSS

| Phương Pháp / Công Trình Công Bố | Nơi Công Bố | mIoU 4 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistoSegNet** (Chan et al.) | ICCV 2019 | 46.92% | 56.92% | 57.53% | 32.31% | 40.91% |
| **SEAM** (Wang et al.) | CVPR 2020 | 61.41% | 74.52% | 68.09% | 50.19% | 52.84% |
| **WSSS-Tissue (Stage-1 Pseudo-Mask)** | MedIA 2022 | 67.05% | 77.59% | 72.62% | 56.40% | 61.58% |
| **WSSS-Tissue (Stage-2 DeepLabV3+ Paper)** | MedIA 2022 | 69.34% | 78.44% | 73.31% | 59.98% | 65.62% |
| **LowMagWSS (40x Benchmark)** | arXiv 2024 | 69.34% | 78.44% | 73.31% | 59.98% | 65.62% |
| **PistoSeg (Mosaic + RFM)** | AAAI 2023 | 69.95% | 78.90% | 73.60% | 60.50% | 66.80% |
| **Our M1 Baseline (Lần 2 & Lần 3)** | Dự án này | **69.35%** | 78.46% | 73.32% | 60.00% | 65.62% |
| **Our M5 Proposed SOTA Ensemble** | Dự án này | **70.51%** | 79.24% | 73.79% | **61.33%** | **67.67%** |
| **Our M6 Proposed Multi-Scale Pyramid D4** | Dự án này | **70.53%** | **79.37%** | **73.82%** | **61.33%** | 67.61% |
