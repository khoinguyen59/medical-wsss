# SE374.R11 - INDEPENDENT COMPREHENSIVE 3-TIER CODEBASE RECHECK & VERIFICATION REPORT

**Course Profile:** SE374.R11 - Phat Trien Phan Mem Xu Ly Anh Y Khoa (UIT 2026)  
**Topic:** Topic 6 - Weakly-Supervised Semantic Segmentation on BCSS Histopathology  
**Audit Protocol:** 3-Tier Source Code Recheck (Tier 1: Automated Test Suites, Tier 2: Semantic Code Inspection, Tier 3: SAST Tools & Reference Codebases)  
**Governance Invariants:** [INVARIANT-12] (ASCII Hyphen '-' 0x2D only) & [INVARIANT-07] (Zero Mockup / Genuine Metrics)  
**Audit Mode:** Rigorous Skeptical Review & Gap-Resolution Synthesis  
**Date:** 2026-11-02  
**Target Repository:** `SE374_WSSS_WORKSPACE`  
**AST Knowledge Graph:** 2662 nodes, 4584 edges, 152 communities (`graphify-out/`)  

---

## 1. Executive Summary & Review of Prior Findings

An independent, rigorous 3-tier recheck of the medical AI codebase was conducted across `src/`, `tests/`, `notebooks/`, and `repos/`. The recheck verified the state of the codebase following recent defect resolutions and addressed specific omissions:

1. **Repository Inventory Clarification:** Exactly 23 specialized medical research repositories in `repos/` plus 15 curated technical meta-tools in `repos/meta_tools/` (total 38 sub-repositories).
2. **Code Citations Grounding:** Verified actual implementation classes (e.g., `ClassificationGateMechanism` defined at `src/models/deeplabv3plus.py:L246-L313`).
3. **Loss Formulations Grounding:** Verified the exact normalized Shannon entropy formula $H_{\text{norm}}(p) = -\frac{\sum p_c \ln(p_c)}{\ln(C)}$ and the gradient detachment guarantee `.detach()` in `src/losses_metrics/losses.py:L108-L115`.
4. **Pytest Warning Root Causes Traced:** Fully traced the mechanics of the 2 runtime warnings captured during test execution.
5. **Full Test & SAST Verification:** All 348 unit/integration tests in `tests/` passed cleanly (0 failures). Ruff reported 0 lint errors across all 42 modules in `src/`. Bandit confirmed 0 High vulnerabilities across 16,009 SLOC (19,550 raw lines).
6. **Governance Hygiene:** 100% compliance with INVARIANT-12 (0 forbidden unicode dashes across all workspace files) and INVARIANT-07 (zero dummy metrics or synthetic patient injection).

---

## 2. Tier 1: Automated Test Suites (`pytest tests/`)

### 2.1 Execution Summary
- **Command:** `cmd /c python -m pytest tests/`
- **Environment:** Windows, Python 3.13.9, pytest-8.4.2, hypothesis-6.170.14, jaxtyping-0.3.11, typeguard-4.6.0
- **Duration:** 104.15 seconds
- **Summary:** **348 passed, 0 failed, 2 warnings**

### 2.2 Complete Test Suite Inventory (22 Test Modules, 348 Tests)

| Test Module | Tests | Status | Scope & Validated Capabilities |
| :--- | :---: | :---: | :--- |
| `tests/test_adversarial_clinical_stress.py` | 13 | PASSED | Otsu tissue gating under illumination/stain shift, GroupNorm vs BatchNorm single-batch stability, absent class gating. |
| `tests/test_api_server.py` | 8 | PASSED | FastAPI clinical inference endpoints, base64 payload encoding/decoding, DZI manifest generation. |
| `tests/test_audit_team_1.py` | 18 | PASSED | Zero patient leakage, Otsu gating on pale stroma/adipose, MLPS loss mathematical bounds, DeepLabV3+ CGM gradient flow. |
| `tests/test_audit_team_2.py` | 19 | PASSED | 151-case manifest determinism, batch size 1 dynamics, PDA prototype cosine gating, TSR/VTP boundary checks. |
| `tests/test_audit_team_3.py` | 17 | PASSED | Multi-layer loss weighting, Spatial Affinity Relocalization token normalization, Invariant-12 ASCII hyphen check. |
| `tests/test_audit_team_4.py` | 18 | PASSED | PDA numerical stability with zero variance, non-square tensor spatial refinement, API contract validation. |
| `tests/test_audit_team_5.py` | 9 | PASSED | Single-batch evaluation dynamics, uncertainty regularization loss gradient detachment, biomarker parity. |
| `tests/test_audit_team_6.py` | 14 | PASSED | Cross-stage CAM fusion consistency, device portability (CPU/CUDA), manifest disjoint partition guarantees. |
| `tests/test_benchmark_components.py` | 7 | PASSED | Streaming evaluation metrics accumulator, confusion matrix bincount arithmetic, per-class and mean IoU/Dice. |
| `tests/test_clinical_service.py` | 14 | PASSED | TCGA-BRCA 23-patient clinical manifest, Kaplan-Meier fitter and analytical product-limit fallback, Mantel-Haenszel log-rank test. |
| `tests/test_colab_notebook.py` | 30 | PASSED | Jupyter Notebook JSON structure (nbformat 4), AST syntax compilation, elimination of naive HSV heuristics, DeepLabV3+ forward verification. |
| `tests/test_colab_pro_notebooks.py` | 71 | PASSED | Colab Pro suite (Stages 1 to 4 across CPU, T4, L4 variants): AST compilation, Drive mount fallbacks, checkpoint resumption. |
| `tests/test_deeplabv3plus.py` | 11 | PASSED | DeepLabV3+ ASPP multi-dilation rates [1, 6, 12, 18], Classification Gate Mechanism penalty injection (-1e4), decoder channel fusion. |
| `tests/test_deliverable_auditor.py` | 11 | PASSED | 4-stage quality auditor gate: blocking phone snapshots, enforcing scientific captions, preventing em-dashes/en-dashes. |
| `tests/test_governance_invariants.py` | 4 | PASSED | Global Invariant 12 hyphen scan across `*.md`, `*.py`, `*.ipynb`; Invariants 24 and 25 presence in `AGENTS.md`. |
| `tests/test_patient_split.py` | 9 | PASSED | TCGA patient ID extraction regex, sample dataset anti-leakage verification, 151-patient split manifest integrity, Hypothesis fuzz tests. |
| `tests/test_pda_backbone.py` | 14 | PASSED | MultiStageFeatureBackbone, Progressive Dropout Attention (PDA) spatial erasing and channel dropout, prototype cosine gating. |
| `tests/test_sam_wsss_adapter.py` | 12 | PASSED | MedSAM PEFT Residual Adapter Bottleneck (256 -> 64 -> 256), CAM-to-prompt conversion, multi-class mask decoding. |
| `tests/test_scientific_upgrades.py` | 14 | PASSED | Multi-magnification simulation (5x/10x/20x), coarse-to-fine adaptive router, Pareto frontier optimization, LUAD-HistoSeg cross-dataset loader, BF1, FwIoU. |
| `tests/test_thin_drive_runner.py` | 10 | PASSED | Thin Drive runner package assembly (`build_drive_package.py`), ZIP extraction round-trip, modular `src` packaging, M1 to M7 engine invocation. |
| `tests/test_training_service.py` | 16 | PASSED | Stage-1 classifier training dry run, Stage-2 DeepLabV3+ training dry run, multi-layer pseudo-mask generation dry run, Asymmetric Loss (ASL), D4 group sync. |
| `tests/test_wsi_pipeline_extension.py` | 9 | PASSED | Gigapixel WSI overview generation, tissue patch extraction, low-mag to high-mag hierarchical inference, whole-ROI stitching report. |

### 2.3 Warning Analysis
1. **`RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation`**:
   - *Source:* `scipy/stats/_axis_nan_policy.py:423` called from `tests/test_clinical_service.py:357`.
   - *Cause:* Mock test cohort contains 5 identical images, yielding variance approaching zero. In clinical cohorts, patient-to-patient biological variance naturally prevents this cancellation.
2. **`DeprecationWarning: scipy.misc is deprecated and will be removed in 2.0.0`**:
   - *Source:* `repos/WSSS-Tissue/tool/infer_utils.py:6` called from `tests/test_colab_notebook.py:558`.
   - *Cause:* Originates from the 2022 external baseline repo `repos/WSSS-Tissue/`. Production code in `src/` does not import `scipy.misc`.

---

## 3. Tier 2: Direct Semantic Code Inspection (`src/` & `notebooks/`)

### 3.1 Patient-Level Anti-Leakage Protocol ([INVARIANT-08])
- **Manifest:** `src/config/bcss_patient_splits.json`
- **Distribution:**
  - Total Unique Cases: 151
  - Train: 106 cases (70.2%)
  - Validation: 22 cases (14.6%)
  - Test: 23 cases (15.2%)
- **Zero Leakage:** $P_{\text{train}} \cap P_{\text{val}} = \emptyset, P_{\text{train}} \cap P_{\text{test}} = \emptyset, P_{\text{val}} \cap P_{\text{test}} = \emptyset$.
- **Verification:** `verify_patient_leakage()` in `src/datasets/patient_split.py:L29-L74` enforces strict disjointness.

### 3.2 Zero Mockup & Academic Scientific Integrity ([INVARIANT-07])
- **Loss Formulations:**
  - `src/losses_metrics/losses.py:L33-L78`: `MultiLayerPseudoSupervisionLoss` calculates multi-resolution cross-entropy ($L_{\text{total}} = 1.0 \cdot L_{\text{bn7}} + 0.5 \cdot L_{\text{b5\_2}} + 0.25 \cdot L_{\text{b4\_5}}$).
  - `src/losses_metrics/losses.py:L80-L133`: `UncertaintyRegularizationLoss` computes normalized Shannon entropy $H_{\text{norm}}(p) = -\frac{\sum p_c \ln(p_c)}{\ln(C)}$ with detached adaptive weighting (`.detach()`).
- **Defensive Safeguards:**
  - `src/clinical/exhaustive_evaluator.py:L621-L683`: When survival data is missing or empty, status is recorded as `"Insufficient stratified cohorts for logrank test; no dummy patients injected."` rather than fabricating survival months.

### 3.3 Architecture & Model Implementation
- **Classification Gate Mechanism (CGM):** `src/models/deeplabv3plus.py:L246-L313`. Penalty injection (-10^4) on absent classes suppresses false positive activations.
- **Progressive Dropout Attention (PDA):** `src/models/pda_generator.py:L38-L81`. Iterative spatial top-$p$ erasing ($p = 0.25$) and channel dropout ($0.50$).
- **MedSAM PEFT Adapter:** `src/models/sam_wsss_adapter.py:L38-L90`. `ResidualAdapterBottleneck` ($256 \to 64 \to 256$) with `nn.init.zeros_` ensuring identity initialization.

### 3.4 Notebook Semantic Audit
- **Total Notebooks:** Exactly 32 `.ipynb` files (12 primary staged, 8 Thin Drive runners, 4 modular templates, 3 history runs, 3 research runners, 2 archive).
- All parse cleanly into valid AST trees under nbformat 4.

---

## 4. Tier 3: External SAST Tools & Reference Codebases

### 4.1 Ruff Static Analysis
- **Command:** `cmd /c python -m ruff check src/`
- **Result:** **All checks passed! (0 lint errors across all 42 modules in `src/`)**

### 4.2 Bandit Security SAST Analysis
- **Command:** `cmd /c python -m bandit -r src/`
- **Result:**
  - **High:** 0
  - **Medium:** 3 (B310 URL retrieval and B615 dataset download in `src/colab_pkg/downloader.py` using trusted Hugging Face base URLs)
  - **Low:** 71 (assert statements in non-production blocks, defensive exception handlers)

### 4.3 Hypothesis Property-Based Fuzz Testing
- **Test File:** `tests/test_patient_split.py:L158-L175`
- Fuzzing with TCGA regex and arbitrary UTF-8 strings: 100 iterations passed with zero failures.

### 4.4 Reference Repositories Ecosystem (`repos/`)
- 23 specialized medical research repositories in `repos/` (including `WSSS-Tissue`, `HisynSeg`, `LowMagWSS`, `PBIP`, `SAM2PATH`, `MONAI`, `WSInfer`, `CLAM`, `TransMIL`, `MambaMIL`, `ConStruct`, `CGRL`, `ISLES_Stroke`).
- 15 curated technical meta-tools in `repos/meta_tools/`.

---

## 5. INVARIANT-12 Character Hygiene Audit

- **Rule:** Strict standard ASCII 0x2D keyboard hyphen `-` only.
- **Scan Result:** **0 non-ASCII hyphens found across the entire workspace.**

---

## 6. Verification Matrix

| Category / Invariant | Baseline (Prior Audit) | Recheck Verification | Status |
| :--- | :---: | :---: | :---: |
| **Pytest Test Suite** | 332 Passed / 14 Failed | **348 Passed / 0 Failed** | **PASS (100%)** |
| **Bandit Security SAST** | 17 Medium Issues | **0 High / 3 Medium (Trusted HF)** | **SECURE** |
| **Ruff Linter Cleanliness** | 77 Errors | **0 Errors across `src/`** | **CLEAN** |
| **Patient-Level Leakage** | 0.0% Leakage | **0.0% Strict Leakage (151 Cases)** | **VERIFIED** |
| **Hypothesis Property Tests** | None | **Integrated & Passing (100 Iterations)** | **ROBUST** |
| **INVARIANT-12 ASCII Hyphens** | Zero non-ASCII | **0 non-ASCII Hyphens (0x2D Only)** | **COMPLIANT** |
| **AST Knowledge Graph** | 2644 nodes | **2662 nodes, 4584 edges** | **SYNCED** |

---
*Report certified under Master Governance Specification `AGENTS.md` v5.0 and Rule 14.*
