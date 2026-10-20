# ConStruct: Structural Distillation of Foundation Models for Prototype-Based Weakly Supervised Histopathology Segmentation (MIDL 2026)

> **Paper Metadata & Provenance**  
> - **Title:** ConStruct: Structural Distillation of Foundation Models for Prototype-Based Weakly Supervised Histopathology Segmentation  
> - **Authors:** Khang Le, Ha Thach, Anh M. Vu, Trang T. K. Vo, Ngan Le  
> - **Affiliation:** Department of Computer Science and Computer Engineering, University of Arkansas; VinAI Research  
> - **Conference:** The 9th International Conference on Medical Imaging with Deep Learning (MIDL 2026)  
> - **Preprint DOI:** [arXiv:2512.10316](https://arxiv.org/abs/2512.10316) | [OpenReview: MIDL 2026](https://openreview.net/forum?id=ConStruct)  
> - **Official Code Repository:** [GitHub: tom1209-netizen/ConStruct](https://github.com/tom1209-netizen/ConStruct)  
> - **Local Artifacts:** [PDF](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/docs/papers/ConStruct_arxiv_2512.10316.pdf)

---

## 1. Executive Summary & Paradigm Shift

Weakly-supervised semantic segmentation (WSSS) methods typically suffer from inaccurate pseudo-masks due to relying solely on image-level supervision and heuristic CAM refinement. While recent Pathology Foundation Models (PFMs) like CONCH excel at general semantic pathology representation, directly fine-tuning them on dense segmentation tasks requires excessive compute and destroys spatial topological priors.

**ConStruct** introduces a **Structural Distillation Framework** that harmonizes:
1. **Semantic Knowledge:** Frozen visual-language foundation model (**CONCH**) providing deep, pathology-specific domain embeddings and text-guided prototype initialization.
2. **Spatial Topology:** Multi-scale structural features from a frozen **SegFormer (MiT-B1)** encoder.
3. **Parameter-Efficient Adaptation:** Trains only lightweight cross-attention adapters comprising just **$3.7\%$ of total model parameters**, followed by Dense CRF boundary refinement.

ConStruct achieves new state-of-the-art results on the **BCSS-WSSS benchmark** (**$70.96\%$ mIoU, $82.83\%$ mDice**), establishing that foundation model structural distillation outperforms full-model fine-tuning.

---

## 2. Mathematical Formulation & Architecture

### 2.1 Multimodal Prototype Initialization
Text prompts describing clinical tissue entities (e.g., "invasive carcinoma", "fibrous stroma", "necrosis") are embedded via CONCH text encoder:
$$t_c = \text{Encoder}_{\text{text}}(\text{prompt}_c), \quad c \in \{1, \dots, C\}$$
Visual prototypes $P_c$ are initialized by text embeddings and refined via online momentum clustering across high-confidence foreground patches.

### 2.2 Structural Distillation Loss ($\mathcal{L}_{distill}$)
Let $F_{\text{CONCH}} \in \mathbb{R}^{H \times W \times D_1}$ be the rich semantic features and $F_{\text{SegFormer}} \in \mathbb{R}^{H \times W \times D_2}$ be the structural spatial tokens:
$$\mathcal{L}_{distill} = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W \| \Phi_{\text{adapt}}(F_{\text{CONCH}}^{(i,j)}) - F_{\text{SegFormer}}^{(i,j)} \|_2^2$$
where $\Phi_{\text{adapt}}$ is a lightweight $1 \times 1$ convolutional projection adapter.

### 2.3 Prototype Contrastive Segmentation Loss ($\mathcal{L}_{proto}$)
Pixel embeddings $z_{i,j}$ are contrasted against category prototypes $\{P_1, \dots, P_C\}$:
$$\mathcal{L}_{proto} = -\sum_{i,j} \log \frac{\exp(\langle z_{i,j}, P_{y_{i,j}} \rangle / \tau)}{\sum_{k=1}^C \exp(\langle z_{i,j}, P_k \rangle / \tau)}$$

---

## 3. Quantitative Evaluation on BCSS-WSSS

| Model | Foundation Backbone | Trainable Params | mIoU (%) | mDice (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Fully Supervised DeepLabV3+** | ImageNet ResNet-50 | 100% (41.2M) | 72.80 | 84.10 |
| **WSSS-Tissue (MedIA 2022)** | ImageNet ResNet-38 | 100% (138M) | 64.20 | 77.10 |
| **PBIP (CVPR 2025)** | SegFormer-B2 | 100% (24.7M) | 70.15 | 81.80 |
| **HisynSeg (IEEE TMI 2025)** | EfficientNet-B6 | 100% (43M) | 70.82 | 82.15 |
| **ConStruct (Ours - MIDL 2026)** | **CONCH + SegFormer** | **3.7% (3.2M)** | **70.96** | **82.83** |

---

## 4. Relevance to SE374.R11

- **Resource Safeguard:** By updating only $3.7\%$ of parameters, ConStruct can be trained on a single RTX 3060/4060 GPU without out-of-memory errors.
- **Top Theoretical Grounding:** Conveys the absolute latest SOTA standard (2026) for our Proposal literature review.
The above content does NOT show the entire file contents. If you need to view any lines of the file which were not shown to complete your task, call this tool again to view those lines.