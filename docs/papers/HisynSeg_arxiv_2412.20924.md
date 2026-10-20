# HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization (IEEE TMI 2025)

> **Paper Metadata & Provenance**  
> - **Title:** HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization  
> - **Authors:** Zijie Fang, Yifeng Wang, Peizhang Xie, Zhi Wang, Yongbing Zhang  
> - **Affiliation:** Tsinghua Shenzhen International Graduate School, Tsinghua University; Harbin Institute of Technology, Shenzhen  
> - **Journal:** IEEE Transactions on Medical Imaging (TMI), Volume 44, Issue 4, Pages 1765-1782, 2025  
> - **Preprint DOI:** [arXiv:2412.20924](https://arxiv.org/abs/2412.20924) | [IEEE TMI DOI](https://doi.org/10.1109/TMI.2024.3520129)  
> - **Official Code Repository:** [GitHub: Vison307/HisynSeg](https://github.com/Vison307/HisynSeg)  
> - **Local Artifacts:** [PDF](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/docs/papers/HisynSeg_arxiv_2412.20924.pdf) | [Source Code](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/repos/HisynSeg)

---

## 1. Executive Summary & Paradigm Shift

Weakly-supervised semantic segmentation (WSSS) in digital pathology aims to delineate dense histological tissue boundaries (tumor epithelium, stroma, necrosis, lymphocytes) using only cheap image-level or patch-level category labels. Traditional WSSS relies on Class Activation Maps (CAM). However, in computational pathology:
1. **Under-activation:** CAMs only activate the most discriminative morphological subregions (e.g., dense hyperchromatic cancer clusters), neglecting transitional invasive borders.
2. **Over-activation:** CAMs bleed into adjacent normal stroma due to stain color similarities across different cellular types (inter-class visual homogeneity).

**HisynSeg** fundamentally sidesteps CAM degradation by converting the weakly-supervised problem into a **synthetic fully-supervised problem**. By assembling single-tissue (homogeneous) patches via random **Bézier curves** and **Mosaic blending**, HisynSeg synthesizes realistic histology images with automatic pixel-exact ground-truth masks. To eliminate domain artifacts between synthetic and real tissues, it enforces a self-supervised **Feature Consistency Regularization** on real unannotated patches.

---

## 2. Core Architecture & Mathematical Formulation

### 2.1 Image-Mixing Synthesis Module (Mosaic & Bézier Curves)
Given a collection of homogeneous tissue patches $P_c$ of class $c \in \{1, \dots, C\}$:
1. **Bézier Boundary Generation:** Smooth morphological boundaries are synthesized by cubic Bézier curves:
   $$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3, \quad t \in [0, 1]$$
   where $P_0, P_1, P_2, P_3$ are randomized control points generating realistic histological interface geometries.
2. **Synthetic Image Assembly:** The synthetic patch $I_{syn}$ and its ground truth mask $M_{syn}$ are formed by:
   $$I_{syn} = \sum_{c=1}^C M_c \odot I_c, \quad M_{syn} = \sum_{c=1}^C c \cdot M_c$$
   where $\odot$ denotes element-wise multiplication and $M_c \in \{0, 1\}^{H \times W}$ is the non-overlapping binary partition mask.

### 2.2 Dual-Stream Training & Consistency Regularization
To train the segmentation network $f_\theta$ (e.g., DeepLabV3+ with EfficientNet-B6 backbone):
1. **Supervised Loss on Synthetic Data ($\mathcal{L}_{sup}$):**
   $$\mathcal{L}_{sup} = \mathcal{L}_{CE}(f_\theta(I_{syn}), M_{syn}) + \mathcal{L}_{Dice}(f_\theta(I_{syn}), M_{syn})$$
2. **Feature Consistency Regularization on Real Data ($\mathcal{L}_{con}$):**
   Real patches $I_{real}$ (without dense labels) are perturbed with affine transformations and stain color jittering $T(\cdot)$:
   $$\mathcal{L}_{con} = \| f_\theta(T(I_{real})) - T(f_\theta(I_{real})) \|_2^2$$
   This prevents the network from overfitting to artificial seam lines while aligning spatial feature distributions across real whole slide images.
3. **Total Objective Function:**
   $$\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_{con} \mathcal{L}_{con}$$

---

## 3. Benchmark Experiments & Quantitative Results

### 3.1 BCSS-WSSS Benchmark (Breast Cancer Microenvironment)
Evaluated on the official BCSS-WSSS dataset across 4 tissue classes (Tumor, Stroma, Inflammatory, Necrosis):

| Method | Supervision Type | Backbone | mIoU (%) | mDice (%) | Tumor IoU (%) | Stroma IoU (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Fully Supervised Baseline** | Pixel-level (Dense) | DeepLabV3+ | 73.10 | 84.45 | 81.20 | 72.85 |
| **WSSS-Tissue** (Han et al., MedIA 2022) | Patch-level (Weak) | ResNet-38 | 64.20 | 77.10 | 74.30 | 63.80 |
| **PistoSeg** (Fang et al., AAAI 2023) | Patch-level (Weak) | EfficientNet | 70.75 | 81.90 | 79.80 | 70.20 |
| **PathMamba** (MICCAI 2024) | Patch-level (Weak) | Mamba SSM | 68.90 | 80.40 | 78.10 | 68.30 |
| **HisynSeg (Ours)** | Patch-level (Weak) | EfficientNet-B6 | **70.82** | **82.15** | **80.15** | **70.64** |

> **Key Observation:** HisynSeg reaches within **$2.28\%$ mIoU** of the fully supervised upper bound, proving that synthetic boundary modeling surpasses heuristic CAM post-processing.

---

## 4. Relevance & Integration in SE374.R11 (Topic 6)

1. **Direct Pipeline Inheritance:** We directly utilize `repos/HisynSeg/create_synthesis_datasets/` to generate high-fidelity synthetic patches for our Stage 2 segmentation model.
2. **Zero Need for Manual Pixel Masks:** Overcomes the primary bottleneck of SE374.R11 by using readily available slide-level TCGA annotations.
3. **Reproducibility Guarantee:** Confirmed codebase in [`repos/HisynSeg`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/repos/HisynSeg) with complete training and test scripts.

The above content shows the entire, complete file contents of the requested file.