# Comprehensive Codebase & Architecture Survey Report

> **Project:** SE374.R11 Weakly-Supervised Histopathology Semantic Segmentation (WSSS)  
> **Target Dataset:** BCSS-WSSS (151 Breast Cancer Cases, Whole Slide Images)  
> **Milestone:** Phase 0 Codebase & Architecture Exploration  
> **Date:** 2026-10-18  
> **Inspector:** Codebase & Architecture Explorer (`explorer_survey`)  
> **Orchestrator ID:** `7ffe6ac5-9a5c-4c39-ba4b-d0861435f647`  

---

## 1. Executive Summary

This survey report provides an exhaustive, read-only audit of the SE374.R11 Weakly-Supervised Histopathology Semantic Segmentation (WSSS) repository. The inspection covers existing source code in `src/`, automated tests in `tests/`, deliverables in `product/`, research documentation in `docs/`, environment dependencies in `requirements.txt`, and compliance with the 25 Master Governance Invariants defined in `AGENTS.md`.

### Key Highlights
1. **Test Suite Status:** 26/26 unit tests PASS in 3.03 seconds.
2. **Environment Execution Note:** The default `pytest` launcher points to an inactive Anaconda Python path (`C:\Users\Nguyen Trong Khoi\anaconda3\python.exe`). Tests must be executed using `cmd /c "C:\Python313\python.exe -m pytest tests/ -v"`.
3. **Core Modules Maturity:**
   - **Team 1 (Data & Otsu Gating):** High maturity. Full patient-level splitting verification (`patient_split.py`), clinical Otsu tissue gating in HSV space (`otsu_filter.py`), and BCSS patch dataset loader (`bcss_dataset.py`) exist and pass tests.
   - **Team 2 (Backbone & Benchmarking Engine):** Moderate maturity. Unified comparative benchmarking engine with 4 modes (Baseline CAM, ClassMix, Spatial Affinity Relocalization, Proposed Adaptive Pipeline) is implemented in `src/models/wsss_tissue.py` and `src/pipeline/comparative_runner.py`. Progressive Dropout Attention (PDA) is documented in architecture diagrams but not yet instantiated as a dedicated module.
   - **Team 3 (Segmentation Network & Losses/Metrics):** Moderate maturity. Multi-label BCE loss, Multi-Layer Pseudo-Supervision loss (MLPS), boundary uncertainty regularization loss, confusion matrix metrics (mIoU, mDice, per-class IoU/Dice), Viable Tumor Percentage (VTP), and Tumor-Stroma Ratio (TSR) are implemented. DeepLabV3+ with Classification Gate Mechanism (CGM) is missing as a separate module in `src/models/`.
   - **Team 4 (Clinical Web Workstation & API):** Not yet implemented (0%). No FastAPI backend (`api_server.py`) or OpenSeadragon deep zoom viewer (`clinical_viewer.py`) exists in `src/visualization/`.
4. **Deliverable Status:** The final project technical report (`product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md`) has NOT yet been created (`product/reports/` contains only `.gitkeep`). Extensive mini proposals and presentation slides exist in `product/proposal/` and `product/slides/`.
5. **Static Governance Compliance:**
   - **Invariant 12 (ASCII Hyphens Only):** 100% compliant across all active code, markdown reports, and configurations. Zero non-ASCII hyphens (no U+2014 em-dashes, no U+2013 en-dashes).
   - **Invariant 15 (Zero Scratch Scripts):** 100% compliant. No one-off scratch scripts exist in root or `scratch/`.

---

## 2. Codebase Inventory and Directory Layout

### 2.1 File Tree Analysis

```
c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA/
├── .agents/                               # Multi-agent coordination metadata
│   ├── rules/
│   │   └── graphify.md                   # Graphify AST knowledge graph rules
│   └── teamwork/
│       ├── ORIGINAL_REQUEST.md           # Authoritative user request
│       ├── orchestrator_1/               # Orchestrator plan, progress, briefing
│       └── explorer_survey/              # Explorer findings, briefing, handoff
├── .github/workflows/ci.yml               # GitHub Actions CI workflow (lint, test, audit)
├── .gitignore                             # Clean filters for medical data, models, cache
├── AGENTS.md                              # Master Governance Specification (25 Invariants)
├── README.md                              # Project overview
├── requirements.txt                       # Medical deep learning dependencies
├── data/
│   ├── README.md                          # Data guide and patient-level splitting rules
│   └── sample_dataset/                    # Mini sample dataset (4 patients, 8 patches)
│       ├── train/                         # 4 weak-label patches (TCGA-A2-A0T2, TCGA-BH-A0B8)
│       ├── valid/                         # 2 patches + GT masks (TCGA-E2-A152)
│       └── test/                          # 2 patches + GT masks (TCGA-C8-A1HJ)
├── docs/
│   ├── architecture/                      # Multi-agent workflows & runtime specifications
│   ├── course/                            # topic.md, SE374.R11 spreadsheet
│   ├── design/                            # Vector SVG diagrams (fig1, fig2, fig3)
│   ├── diary/                             # Progress notes
│   ├── lectures/                          # Course slides
│   ├── notes/                             # Meeting notes
│   ├── papers/                            # Reference papers (PDF & Vision markdown)
│   ├── references/                        # Research dossiers
│   ├── research/                          # 12 detailed research and benchmarking protocols
│   └── templates/                         # UIT thesis, Springer LNCS, Proposal templates
├── product/
│   ├── proposal/                          # Mini proposals (.md, .docx, .html, .pdf) & assets
│   ├── reports/                           # Final technical reports (.gitkeep only)
│   └── slides/                            # Presentation slides (.md, .html, .pptx)
├── repos/                                 # 23 reference repos (WSSS-Tissue, ARML, LowMagWSS, etc.)
├── scratch/                               # Empty (.gitkeep only)
├── scripts/
│   └── download_papers.py                 # Academic paper retrieval script
├── src/
│   ├── config/                            # .gitkeep (hyperparameters to be defined)
│   ├── datasets/
│   │   ├── __init__.py                    # Public API exports
│   │   ├── bcss_dataset.py                # BCSSDataset loader with weak label regex
│   │   ├── otsu_filter.py                 # Clinical Otsu tissue gating in HSV space
│   │   └── patient_split.py               # Patient-level splitting and leakage verification
│   ├── losses_metrics/
│   │   ├── losses.py                      # MultiLabel, MLPS, UncertaintyRegularization losses
│   │   └── metrics.py                     # Confusion matrix, mIoU, Dice, VTP, TSR metrics
│   ├── models/
│   │   └── wsss_tissue.py                 # Backbone, ARML, ClassMix, WSSSBenchmarkingEngine
│   ├── pipeline/
│   │   └── comparative_runner.py          # 4-configuration multi-seed benchmark runner
│   ├── utils/
│   │   └── deliverable_auditor.py         # 4-gate independent deliverable auditor (628 lines)
│   └── visualization/
│       ├── draw_all_proposal_figures.py   # Publication-grade monochrome diagram generation
│       └── draw_pipeline_diagram.py       # Pipeline vector diagram generator
└── tests/
    ├── test_benchmark_components.py       # 7 unit tests for metrics, losses, engine, gating
    ├── test_deliverable_auditor.py        # 11 unit tests for 4-gate deliverable auditor
    ├── test_governance_invariants.py      # 3 unit tests for Invariant 12, 24, 25, phone photos
    └── test_patient_split.py              # 5 unit tests for patient leakage and dataset dry-run
```

---

## 3. Detailed Assessment of Core Module Areas

### 3.1 Team 1: BCSS-WSSS Dataset, Patient-Level Splitting & Clinical Otsu Gating
- **Target Files:** `src/datasets/bcss_dataset.py`, `src/datasets/otsu_filter.py`, `src/datasets/patient_split.py`.
- **Implementation Status:** COMPLETE & VERIFIED.
- **Detailed Findings:**
  1. `patient_split.py`:
     - `extract_patient_id()` extracts standard TCGA patient identifiers (e.g., `TCGA-A2-A0T2`), LUAD-HistoSeg formats, and GDPH prefixes.
     - `verify_patient_leakage()` strictly enforces `P_train \cap P_val == \emptyset`, `P_train \cap P_test == \emptyset`, and `P_val \cap P_test == \emptyset`. Raises a descriptive `ValueError` upon overlap detection.
     - `split_patients_kfold()` produces deterministic, disjoint K-fold splits.
  2. `otsu_filter.py`:
     - `compute_otsu_tissue_mask()` converts RGB to HSV, extracts the Saturation channel, computes Otsu threshold with sensitivity multiplier (default 0.8), applies intensity gating (`gray < 235`) to extinguish white glass, and executes morphological opening/closing with an elliptical kernel.
     - `apply_otsu_gating_to_prediction()` replaces all glass/background predicted pixels with `background_class_idx = 4`.
  3. `bcss_dataset.py`:
     - Loads patch images and parses multi-hot weak labels `[y0 y1 y2 y3]` from bracketed filenames via regex.
     - Supports 5 tissue classes: `0: TUM (Tumor)`, `1: STR (Stroma)`, `2: LYM (Lymphocytic infiltrate)`, `3: NEC (Necrosis)`, `4: BACK (Background / Other)`.
     - Validated against `data/sample_dataset/`.
- **Gaps / Action Items for Team 1:**
  - Create a deterministic patient mapping file (`src/config/bcss_patient_splits.json`) representing the full 151 TCGA patient IDs (91 train, 30 val, 30 test) for full-scale reproducibility.

---

### 3.2 Team 2: Model Backbone, PDA CAM Generation & 4 Benchmark Configurations
- **Target Files:** `src/models/wsss_tissue.py`, `src/pipeline/comparative_runner.py`.
- **Implementation Status:** PARTIALLY COMPLETE (Modular extraction and PDA needed).
- **Detailed Findings:**
  1. `wsss_tissue.py`:
     - Implements `HistopathologyClassifierBackbone`, a multi-stage convolutional network producing feature stages `b4_5`, `b5_2`, `bn7` and GAP classification logits.
     - Implements Conv weight projection CAM extraction via `compute_cam()`.
     - Implements `SpatialAffinityRelocalization` (Configuration 3: ARML) using scaled dot-product affinity $A = \text{softmax}(QK^T / \sqrt{d})$ and CAM energy diffusion $\tilde{M} = \alpha M + (1 - \alpha) A M$.
     - Implements `apply_classmix_augmentation()` (Configuration 2: ClassMix) synthesizing cross-patch boundary blends with multi-label logical OR.
     - Implements `WSSSBenchmarkingEngine` supporting 4 modes: `"baseline"`, `"classmix"`, `"relocalization"`, and `"proposed"`.
     - Proposed mode incorporates dynamic progressive thresholding ($\mu_t = \max(0.90 \cdot 0.98^{t-2}, 0.65)$) and Otsu gating.
  2. `comparative_runner.py`:
     - Implements `run_comparative_benchmark()` across 5 seeds: `[42, 100, 2024, 2025, 2026]`.
     - Aggregates mean $\pm$ standard deviation for mIoU, mDice, per-class IoU, and FPS.
     - Exports results to `experiments/benchmarks/benchmark_results.json` and `experiments/benchmarks/BENCHMARK_REPORT.md`.
- **Gaps / Action Items for Team 2:**
  - Modularize into separate files: `src/models/backbone.py` (ResNet50 / ConvNeXt / dilated ResNet), `src/models/pda_generator.py` (Progressive Dropout Attention).
  - Explicitly implement Progressive Dropout Attention (PDA) to prevent CAM boundary over-expansion by progressively masking high-attention channels during CAM computation.

---

### 3.3 Team 3: Segmentation Network & Losses/Metrics Suite
- **Target Files:** `src/models/deeplabv3plus.py`, `src/losses_metrics/losses.py`, `src/losses_metrics/metrics.py`.
- **Implementation Status:** LOSSES & METRICS COMPLETE; SEGMENTATION NETWORK MISSING.
- **Detailed Findings:**
  1. `losses.py`:
     - `MultiLabelClassificationLoss`: BCEWithLogitsLoss for weak multi-label supervision.
     - `MultiLayerPseudoSupervisionLoss` (MLPS): Combines pseudo-supervision from `bn7` (weight 1.0), `b5_2` (weight 0.5), and `b4_5` (weight 0.25) with dynamic spatial interpolation.
     - `UncertaintyRegularizationLoss`: Normalizes pixel prediction entropy $H(p) = -\sum p \log(p) / \log(C)$, computes uncertainty penalty weights, and regularizes ambiguous boundary pixels.
  2. `metrics.py`:
     - `MedicalSegmentationMetrics`: True 2D confusion matrix accumulation via `np.bincount`, calculates per-class IoU, mIoU, per-class Dice, mDice. Zero dummy/mockup metrics.
     - Clinical biomarkers: `compute_viable_tumor_percentage()` (VTP) and `compute_tumor_stroma_ratio()` (TSR).
- **Gaps / Action Items for Team 3:**
  - Build `src/models/deeplabv3plus.py` implementing DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) and Classification Gate Mechanism (CGM) to suppress non-present tissue classes.
  - Implement the Stage 2 training loop that consumes generated pseudo-masks and trains DeepLabV3+.

---

### 3.4 Team 4: Clinical Web Workstation & API
- **Target Files:** `src/visualization/clinical_viewer.py`, `src/visualization/api_server.py`, static web assets.
- **Implementation Status:** NOT STARTED (0%).
- **Detailed Findings:**
  - `src/visualization/` currently contains only figure drawing scripts for proposal figures (`draw_all_proposal_figures.py`, `draw_pipeline_diagram.py`).
  - No FastAPI application or REST endpoints exist for patch upload, inference, or heatmap streaming.
  - No OpenSeadragon clinical viewer interface exists.
- **Action Items for Team 4:**
  - Create `src/visualization/api_server.py` using FastAPI with endpoints:
    - `POST /api/v1/predict/patch`: Accepts RGB image patch, runs WSSS inference, returns segmentation mask and biomarkers (VTP, TSR).
    - `POST /api/v1/predict/wsi`: Accepts WSI coordinates / tile batches, applies Otsu gating, generates deep zoom overlays.
    - `GET /api/v1/health`: Health and VRAM monitor.
  - Create interactive clinical workstation (`src/visualization/clinical_viewer.py` and static HTML/JS) with OpenSeadragon deep zoom viewer for multi-resolution WSI inspection.

---

## 4. Test Suite Audit & Execution Results

### 4.1 Test Files and Functions Enumeration

The repository contains 4 test files defining 26 unit tests:

| Test File | Test Class / Function | Purpose |
| :--- | :--- | :--- |
| `tests/test_benchmark_components.py` | `test_metrics_service_perfect_match` | Verifies mIoU and Dice equal 1.0 on identical inputs |
| `tests/test_benchmark_components.py` | `test_metrics_service_partial_overlap` | Verifies IoU on synthetic partial overlap mismatch |
| `tests/test_benchmark_components.py` | `test_clinical_biomarkers` | Verifies VTP and TSR percentage calculations |
| `tests/test_benchmark_components.py` | `test_otsu_clinical_gating` | Verifies Otsu thresholding on pink H&E tissue vs white glass |
| `tests/test_benchmark_components.py` | `test_losses` | Verifies forward pass and gradients for MultiLabel, MLPS, Uncertainty losses |
| `tests/test_benchmark_components.py` | `test_wsss_engine_all_configurations` | Verifies mask generation for all 4 benchmark configurations |
| `tests/test_benchmark_components.py` | `test_classmix_augmentation` | Verifies shape consistency for ClassMix augmentation |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_valid_proposal_passes_audit` | Verifies current proposal passes all 4 quality gates |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_blocks_raw_phone_snapshot_name` | Verifies auditor blocks camera snapshot file names (`media_*`, `IMG_*`) |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_blocks_empty_caption` | Verifies auditor flags images lacking scientific captions |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_blocks_forbidden_em_dash_and_en_dash` | Verifies auditor blocks non-ASCII em-dashes and en-dashes |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_blocks_dummy_metrics_and_mockup` | Verifies auditor catches lorem ipsum and dummy numbers |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_flags_low_resolution_images` | Verifies auditor warns on low-res images (<600x300) |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_blocks_ascii_art_diagrams` | Verifies auditor blocks ASCII art boxes and text flowcharts |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_blocks_colorful_figures_and_enforces_monochrome` | Verifies auditor catches overly colorful figures |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_blocks_wrong_faculty_and_external_lab` | Verifies auditor blocks unauthorized faculty/lab names |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_blocks_leaked_rubrics_and_invariants` | Verifies auditor blocks leaked prompt tokens (`[INVARIANT-*]`) |
| `tests/test_deliverable_auditor.py` | `TestDeliverableAuditor::test_valid_docx_proposal_passes_audit` | Verifies compiled DOCX proposal passes audit |
| `tests/test_governance_invariants.py` | `test_no_ai_em_dash_in_markdown` | Scans all workspace Markdown files for non-ASCII hyphens |
| `tests/test_governance_invariants.py` | `test_invariants_24_and_25_defined_in_agents_md` | Verifies Invariants 24 and 25 are documented in `AGENTS.md` |
| `tests/test_governance_invariants.py` | `test_no_raw_phone_photos_in_product_directory` | Verifies no camera/phone photo prefixes exist in `product/` |
| `tests/test_patient_split.py` | `test_extract_patient_id_tcga` | Verifies patient ID extraction across filename formats |
| `tests/test_patient_split.py` | `test_verify_patient_leakage_on_sample_dataset` | Verifies `data/sample_dataset` has 0.0% patient leakage |
| `tests/test_patient_split.py` | `test_verify_patient_leakage_catches_overlap` | Verifies `ValueError` is raised when patient overlap occurs |
| `tests/test_patient_split.py` | `test_split_patients_kfold` | Verifies K-Fold patient splits are disjoint and deterministic |
| `tests/test_patient_split.py` | `test_bcss_dataset_loader_dry_run` | Verifies DataLoader loads batches with correct tensor shapes |

### 4.2 Test Execution Output

Command executed:
```cmd
cmd /c "C:\Python313\python.exe -m pytest tests/ -v"
```

Result:
```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0 -- C:\Python313\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
plugins: anyio-4.12.1, timeout-2.4.0
collecting ... collected 26 items

tests/test_benchmark_components.py::test_metrics_service_perfect_match PASSED [  3%]
tests/test_benchmark_components.py::test_metrics_service_partial_overlap PASSED [  7%]
tests/test_benchmark_components.py::test_clinical_biomarkers PASSED      [ 11%]
tests/test_benchmark_components.py::test_otsu_clinical_gating PASSED     [ 15%]
tests/test_benchmark_components.py::test_losses PASSED                   [ 19%]
tests/test_benchmark_components.py::test_wsss_engine_all_configurations PASSED [ 23%]
tests/test_benchmark_components.py::test_classmix_augmentation PASSED    [ 26%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_valid_proposal_passes_audit PASSED [ 30%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_raw_phone_snapshot_name PASSED [ 34%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_empty_caption PASSED [ 38%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_forbidden_em_dash_and_en_dash PASSED [ 42%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_dummy_metrics_and_mockup PASSED [ 46%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_flags_low_resolution_images PASSED [ 50%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_ascii_art_diagrams PASSED [ 53%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_colorful_figures_and_enforces_monochrome PASSED [ 57%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_wrong_faculty_and_external_lab PASSED [ 61%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_leaked_rubrics_and_invariants PASSED [ 65%]
tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_valid_docx_proposal_passes_audit PASSED [ 69%]
tests/test_governance_invariants.py::test_no_ai_em_dash_in_markdown PASSED [ 73%]
tests/test_governance_invariants.py::test_invariants_24_and_25_defined_in_agents_md PASSED [ 76%]
tests/test_governance_invariants.py::test_no_raw_phone_photos_in_product_directory PASSED [ 80%]
tests/test_patient_split.py::test_extract_patient_id_tcga PASSED         [ 84%]
tests/test_patient_split.py::test_verify_patient_leakage_on_sample_dataset PASSED [ 88%]
tests/test_patient_split.py::test_verify_patient_leakage_catches_overlap PASSED [ 92%]
tests/test_patient_split.py::test_split_patients_kfold PASSED            [ 96%]
tests/test_patient_split.py::test_bcss_dataset_loader_dry_run PASSED     [100%]

============================= 26 passed in 3.03s ==============================
```

**Status:** 26 passed, 0 failed, 0 skipped.

---

## 5. Deliverables Status (`product/`)

1. **`product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md`:**
   - **Status:** MISSING (Does not exist).
   - Only `.gitkeep` exists in `product/reports/`.
   - This document must be synthesized as required by Milestone Phase 4 and Acceptance Criterion 5.
2. **Existing Deliverables in `product/`:**
   - Proposals in `product/proposal/`:
     - `SE374_MINI_PROPOSAL_WSSS_TISSUE.md` (and compiled `.docx`, `.html`, `.pdf`)
     - `SE374_MINI_PROPOSAL_PHASED_ROADMAP_WSSS.md` (and compiled `.docx`, `.html`, `.pdf`)
     - `SE374_MINI_PROPOSAL_WSSS_WSI.md` (and compiled `.docx`, `.html`, `.pdf`)
   - Slides in `product/slides/`:
     - `SE374_PROPOSAL_SLIDES_WSSS_WSI.md`, `.html`, `.pptx`
   - Publication-grade monochrome assets in `product/proposal/assets/`:
     - `fig1_standard_wsi_pipeline.png` (High-contrast scientific diagram)
     - `fig1_wsss_tissue_workflow.png`
     - `fig2_system_architecture.png`
     - `fig3_pipeline_dataflow.png`

---

## 6. Static Compliance & Governance Audit

| Invariant | Description | Verification Method | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **[INVARIANT-07]** | Zero Mockup & Academic Integrity | Code audit of `src/losses_metrics/` and `tests/` | **PASS** | True bincount confusion matrix and PyTorch losses. No hardcoded results. |
| **[INVARIANT-08]** | Patient-Level Anti-Leakage Protocol | `patient_split.py` and `test_patient_split.py` | **PASS** | Pairwise disjoint train/val/test patient checks verified. |
| **[INVARIANT-12]** | ASCII 0x2D Hyphens Only (No Em-dash/En-dash) | Full-codebase scan with Python regex script | **PASS** | 0 non-ASCII hyphens in active code/docs. Forbidden chars only in detector tests. |
| **[INVARIANT-13]** | Windows OS Standard Execution | Execution via `cmd /c` | **PASS** | All executions utilize `cmd /c` on Windows. |
| **[INVARIANT-15]** | Zero One-Off Scratch Scripts | Directory listing of root and `scratch/` | **PASS** | Zero scratch scripts found. `scripts/` contains only `download_papers.py`. |
| **[INVARIANT-24]** | Publication-Grade Monochrome Visuals | `deliverable_auditor.py` Gate 1 audit | **PASS** | Monochrome vector figures verified in `product/proposal/assets/`. |
| **[INVARIANT-25]** | Deliverable Auditor Gate | `test_deliverable_auditor.py` | **PASS** | All 4 quality gates functional and passing. |

---

## 7. Gap Analysis & Directives for Implementation Teams

### Summary Matrix of Implementation Gaps

| Team / Area | Current Asset | Missing Asset / Gap | Urgency |
| :--- | :--- | :--- | :---: |
| **Team 1: Data Ingestion** | `bcss_dataset.py`, `patient_split.py`, `otsu_filter.py` | Full 151 patient ID split manifest (`src/config/bcss_patient_splits.json`) | Medium |
| **Team 2: Model Backbone & PDA** | `wsss_tissue.py` (all-in-one) | Dedicated `backbone.py` (ResNet50 / ConvNeXt), dedicated `pda_generator.py` (PDA CAM) | High |
| **Team 3: Segmentation & Losses** | `losses.py`, `metrics.py` | `src/models/deeplabv3plus.py` with ASPP and Classification Gate Mechanism (CGM), Stage 2 training script | High |
| **Team 4: Clinical API & Workstation** | None | `src/visualization/api_server.py` (FastAPI), `src/visualization/clinical_viewer.py` (OpenSeadragon) | High |
| **Benchmarking & Report** | `comparative_runner.py` | Full multi-seed run execution logs, `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md` | Highest |

### Recommended Execution Path for Orchestrator
1. **Phase 1 (Workers):**
   - Dispatch Team 2 to refactor `backbone.py` and implement Progressive Dropout Attention (PDA).
   - Dispatch Team 3 to implement DeepLabV3+ with CGM in `src/models/deeplabv3plus.py`.
   - Dispatch Team 4 to implement the FastAPI server and OpenSeadragon clinical viewer in `src/visualization/`.
2. **Phase 2 (Adversarial Checkers):**
   - Checker Team A: Verify PEP 8, type hints, and Invariant 12/13/15 compliance.
   - Checker Team B: Verify 0.0% patient leakage and zero-mockup math.
   - Checker Team C: Stress-test blank glass slides and verify VRAM < 6 GB.
3. **Phase 3 (Benchmarks):**
   - Execute `src/pipeline/comparative_runner.py` across the 4 configurations to produce empirical logs.
4. **Phase 4 (Final Report):**
   - Synthesize all empirical benchmark logs, theoretical formulations, and clinical analysis into `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md`.
   - Stop and halt for Human Commander review.
