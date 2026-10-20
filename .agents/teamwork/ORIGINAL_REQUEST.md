# Original User Request

## 2026-10-18T15:53:06Z

# Teamwork Project Prompt - Launched

> Status: Launched
> Goal: Multi-team implementation with 1 Lead, 4 Implementation Teams, and 3 Adversarial Checker Teams
> Requested team: Lead-directed multi-agent system with 3 independent adversarial verification teams

Build, refine, and verify the complete Weakly-Supervised Histopathology Semantic Segmentation (WSSS) codebase and produce the comprehensive Project Technical Report markdown (product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md) for SE374.R11 on BCSS-WSSS and Whole Slide Images. Operate through an iterative Lead-driven implementation loop guarded by 3 independent cross-checking teams, perfecting the code until all tests pass and stopping at the final project markdown report for human review.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
Integrity mode: development

## Team Structure & Operational Hierarchy

- **Team Lead (Orchestrator):** Breaks down the WSSS pipeline into modular work packets, issues precise coding specifications, coordinates task dependencies, routes defect reports from the 3 checker teams into fix directives, and stops when the code is verified and the comprehensive project report in Markdown is generated.
- **Module Implementation Teams:**
  - **Team 1 (Data Ingestion & Otsu Gating):** Manages BCSS-WSSS dataset with strict patient-level splitting (zero patient leakage across 151 cases) and implements clinical Otsu tissue gating in HSV space to extinguish background false positives.
  - **Team 2 (Model Backbone & PDA Generator):** Builds the multi-stage feature backbone, Progressive Dropout Attention (PDA) CAM generation, and implements the 4 benchmark configurations: Baseline CAM, Data-Centric ClassMix, Spatial Affinity Relocalization (ARML), and Proposed Adaptive Pipeline.
  - **Team 3 (Segmentation Network & Losses/Metrics):** Implements DeepLabV3+ with Classification Gate Mechanism (CGM), multi-layer pseudo-supervision loss, boundary uncertainty regularization, and the zero-mockup metrics suite (mIoU, mDice, per-class IoU, VTP, TSR).
  - **Team 4 (Clinical Web Workstation & API):** Packages the FastAPI backend and interactive clinical interface with OpenSeadragon deep zoom viewing for ROI patches and WSI slides.
- **3 Independent Checker Teams (Tri-Party Adversarial Verification):**
  - **Checker Team A (Code Quality & Invariants):** Enforces static code quality, type annotations, PEP 8, Invariant 12 (100% ASCII 0x2D hyphens, 0 em-dashes/en-dashes), Invariant 13 (Windows `cmd /c`), and Invariant 15 (zero scratch scripts).
  - **Checker Team B (Scientific Integrity & Leakage Auditor):** Verifies zero patient-level leakage across splits (Invariant 08), zero dummy/fabricated metrics (Invariant 07 - Zero Mockup), and mathematical correctness of loss/metric formulations.
  - **Checker Team C (Adversarial Clinical & Stress Tester):** Executes edge case stress tests (blank glass slides, adipose tissue, single-item batches), verifies memory safety (zero CUDA OOM, < 6 GB VRAM inference), and acts as a skeptical pathologist critiquing tumor boundaries.

## Requirements

### R1. Complete End-to-End Implementation of the 4 Benchmark Configurations
- Implement, verify, and run the 4 configurations in `src/models/wsss_tissue.py` and `src/pipeline/comparative_runner.py`:
  1. Configuration 1: Vanilla CAM Baseline (Han et al., MedIA 2022).
  2. Configuration 2: Data-Centric Augmentation (Baseline + ClassMix).
  3. Configuration 3: Spatial Affinity Relocalization (ARML, MICCAI 2024).
  4. Configuration 4: Proposed Adaptive Pipeline (Otsu Gating + Dynamic Progressive Thresholding + Uncertainty Regularization).

### R2. Closed-Loop Adversarial Critique and Self-Correction
- Every module must be independently audited by Checker Team A, Checker Team B, and Checker Team C.
- Any defect detected must generate a structured Defect Report to the Team Lead, who issues a Fix Directive to the responsible module team.
- The loop re-verifies until all 3 Checker Teams agree on 100% PASS with 0 defects.

### R3. Final Project Technical Report in Markdown
- Synthesize all theoretical analysis, architectural diagrams, mathematical formulations, experimental benchmark logs, and clinical biomarker results into the comprehensive final project report in markdown: `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md`.
- Stop at this markdown report stage for user review before any further execution.

## Acceptance Criteria

- [ ] 100% of unit tests pass in `pytest tests/` (26+ tests).
- [ ] Checker Team A verifies 0 non-ASCII hyphens, 100% ASCII 0x2D hyphens across all code and docs.
- [ ] Checker Team B verifies 0.0% patient-level leakage between Train, Val, and Test splits.
- [ ] Checker Team C verifies that Otsu gating eliminates 100% of false positives on blank glass and inference consumes < 6 GB VRAM.
- [ ] Comprehensive Project Technical Report markdown is generated at `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md`.
- [ ] Execution halts at the final project markdown report for human commander inspection.


## 2026-10-19T19:11:48Z

Audit, refactor, and rigorously rewrite the Google Colab execution notebook (`notebooks/SE374_Colab_WSSS_Pipeline.ipynb`) to integrate the full DeepLabV3+ deep neural network architecture with its official pretrained SOTA weights (`stage2_checkpoint_trained_on_bcss.pth`). Eliminate the naive color heuristic, evaluate on 100% real TCGA-BRCA patient data with Otsu clinical gating, achieve clinical-grade mIoU, and deliver an end-to-end automated download pipeline.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\notebooks
Integrity mode: development

## Requirements

### R1. DeepLabV3+ Architecture & Pretrained Weight Ingestion
- Implement the exact PyTorch DeepLabV3+ architecture with Atrous Spatial Pyramid Pooling (ASPP) and ResNet backbone matching the tensor shapes in `stage2_checkpoint_trained_on_bcss.pth` (downloaded via Google Drive ID `1cXm29ko8UevbaeW6YreSg-vuNaQiN_je`).
- Eliminate the naive HSV color thresholding in inference; forward pass must execute directly through the deep neural network on GPU.

### R2. Real TCGA-BRCA Data Pipeline & Otsu Clinical Gating
- Automate direct download of the official `BCSS-WSSS.zip` (Google Drive ID `1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn`) and unzipping into `/content/BCSS-WSSS/`.
- Provide seamless dual-mode execution:
  - Sanity Mode (T4 Free): Evaluates a subset of real patient test patches (e.g. 150 real patches from patients like `TCGA-AQ-A04H`, `TCGA-C8-A1HJ`).
  - Full Mode (L4 Pro): Evaluates the complete set of 4,986 real test patches.
- Apply Otsu clinical gating in HSV saturation space to force glass/background regions to Class 4 (Background), extinguishing false positives.

### R3. Quantitative Clinical Metric Evaluation & Logging
- Compute the full 5x5 confusion matrix over valid pixels across the 5 histopathology classes: Tumor (TUM), Stroma (STR), Lymphocyte (LYM), Necrosis (NEC), Background (BACK).
- Calculate and log per-class IoU, global mIoU, mDice, and inference throughput (FPS).

### R4. Clinical Visualizations & Closed-Loop Packaging
- Render 300 DPI publication-grade side-by-side comparisons: Original Patient H&E vs Pathologist Ground Truth vs WSSS AI Discrete Segmentation Mask.
- Save summary reports in both Markdown (`COLAB_EVALUATION_REPORT.md`) and JSON.
- Automatically compress all outputs into `/content/SE374_Colab_Results.zip` and trigger direct browser download via `google.colab.files.download()`.

## Acceptance Criteria

### Technical & Execution Readiness
- [ ] Valid Jupyter Notebook JSON structure (`nbformat: 4`), parses cleanly with zero syntax errors.
- [ ] Model architecture matches state dictionary keys in `stage2_checkpoint_trained_on_bcss.pth` without key mismatch errors.
- [ ] No dummy data or synthetic image generation: 100% real TCGA-BRCA pathology patches.
- [ ] Otsu gating function handles boundary conditions and maintains (H, W) spatial alignment.
- [ ] Cell 9 automatically executes `zip` and `files.download()` to prompt user download without manual file-browser navigation.
- [ ] Strictly standard keyboard ASCII hyphens `-` (0x2D) only, 0 non-ASCII em-dashes (`\u2014`) or en-dashes (`\u2013`).


## 2026-10-20T11:36:26Z

Use a full multi-agent team with a Leader to divide work across a group of worker agents and a dedicated group of recheck/auditor agents to independently verify and cross-check all work. The project upgrades our Weakly-Supervised Semantic Segmentation (WSSS) Histopathology pipeline (`SE374.R11` Topic 6 on BCSS-WSSS) by grounding every method directly in the 30 academic papers (`docs/papers/`) and 24 official GitHub repositories (`repos/`), restoring all remaining 0-byte research/paper Markdown dossiers, and benchmarking genuine SOTA multi-method WSSS architectures across all 4 Google Colab GPU notebook variants.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
Integrity mode: development

## Verification Resources
- Official Repositories in Workspace:
  - `repos/WSSS-Tissue/` (`3_train_stage2.py`, `network/resnet38d.py`, `network/resnet38_cls.py`)
  - `repos/LowMagWSS/` (`SegMethod/WSSS-Tissue/visualize_predictions_v3.py`)
  - `repos/ARML/` (`3_train_stage2.py`, `tool/cca.py`, `tool/amm.py`)
  - `repos/PistoSeg/` (`models/revise_net.py`, `segmentation_test.py`)
  - `repos/HisynSeg/` (`models/smp_model.py`, `segmentation_train.py`)
  - `repos/PBIP/` (`utils/fgbg_feature.py`, `utils/validate.py`)
  - `repos/ConStruct/` (`README.md`)
- Official Pretrained Checkpoints (BCSS-WSSS):
  - Stage-1 `ResNet38d` `Net_CAM` (`stage1_checkpoint_trained_on_bcss.pth`): Google Drive ID `118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc`
  - Stage-2 `DeepLabV3+` `ResNet-101` (`stage2_checkpoint_trained_on_bcss.pth`): Google Drive ID `1cXm29ko8UevbaeW6YreSg-vuNaQiN_je`
- Automated Test Suites:
  - `tests/test_colab_notebook.py`
  - `tests/test_governance_invariants.py`
  - `src/utils/deliverable_auditor.py`

## Requirements

### R1. Full Restoration & Cross-Verification of Academic Paper & Research Dossiers (`docs/`)
Restore all 9 remaining 0-byte Markdown files in `docs/papers/` (`CLAM_arxiv_2004.09666.md`, `HoVerNet_arxiv_1812.06499.md`, `PathMamba_MICCAI2024.md`, `UNI_arxiv_2308.05814.md`, `Virchow_arxiv_2309.07778.md`) and `docs/references/` (`deep-research-report.md`, `Digital_Pathology_WSI_Resources.md`, `Gigapixel_WSI_Segmentation_Survey.md`, `Nghien_Cuu_AI_WSI.md`) from the corresponding PDFs (`docs/papers/*.pdf`) and workspace research sources so that every file has a complete, structured academic Markdown synthesis (`> 2,500` bytes) with zero non-ASCII dashes. Also update `docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md` with a comprehensive code-to-code comparative matrix across the cloned repositories (`WSSS-Tissue`, `LowMagWSS`, `ARML`, `PistoSeg`, `HisynSeg`, `PBIP`, `ConStruct`, `S2C`, `SAM2PATH`, `PFM_Segmentation`, `SEINE`, `WaWeHis`).

### R2. Code-to-Code Repo-Grounded Multi-Method WSSS Engine (`src/models/wsss_inference_methods.py`)
Cross-reference the actual source code in `repos/WSSS-Tissue`, `repos/LowMagWSS`, `repos/ARML`, `repos/PistoSeg`, `repos/HisynSeg`, `repos/PBIP`, and `repos/ConStruct` to implement and integrate genuine repo-grounded SOTA architectures and inference mechanisms in `src/models/wsss_inference_methods.py`:
- Include the exact `ResNet38d` + `Net_CAM` Stage-1 architecture (`WSSSTissueStage1NetCAM`) ported directly from `repos/WSSS-Tissue/network/resnet38d.py` and `repos/WSSS-Tissue/network/resnet38_cls.py` so that `stage1_checkpoint_trained_on_bcss.pth` loads with 100% key match (`strict=True`).
- Implement 5 genuine repo-grounded SOTA methods in `MultiMethodWSSSEngine` (compatible both when `model_stage1` is loaded on GPU/CPU and with graceful fallback in unit tests when `model_stage1=None`):
  1. `M1_Baseline_SingleScale`: Stage-2 DeepLabV3+ raw single-scale forward pass (`Is_GM=False` in `repos/WSSS-Tissue/3_train_stage2.py`, reproducing the `68.71%` mIoU baseline).
  2. `M2_MultiView_D4_TTA`: Stage-2 DeepLabV3+ + Stage-1 `ResNet38d` `Net_CAM` Classification Gate Mechanism (`CGM`, `Is_GM=True`, `y_cls > 0.1` from `repos/WSSS-Tissue/3_train_stage2.py` lines 158-167 & `repos/LowMagWSS/SegMethod/WSSS-Tissue/visualize_predictions_v3.py` lines 256-262) + Multi-View Dihedral TTA (`repos/ConStruct/README.md` & `repos/PistoSeg/segmentation_test.py`).
  3. `M3_LongTail_Prior_Calibration`: Stage-1 CGM + Multi-View TTA + Cross-Stage CAM-Segmentation Probability Consistency Fusion (`repos/HisynSeg/models/smp_model.py` & `repos/PBIP/utils/validate.py`) + Long-Tail Prior Calibration (`LYM`, `NEC` minority recovery).
  4. `M4_Spatial_Bilateral_Refinement`: Stage-1 CGM + Multi-View TTA + PistoSeg Refining Feature Module (`RFM` non-maximum suppression & affinity propagation from `repos/PistoSeg/models/revise_net.py` lines 29-96) & Edge-Preserving Bilateral / Island Suppression (`repos/ARML/3_train_stage2.py`).
  5. `M5_Proposed_SOTA_Ensemble`: Full Proposed Pipeline combining Stage-1 `ResNet38d` `Net_CAM` CGM + Cross-Stage CAM Fusion + Multi-View TTA + Long-Tail Prior Calibration + RFM/Edge-Preserving Spatial Refinement + Ultra-Safe Pure-Glass Guard.

### R3. Synchronized Google Colab GPU Notebook Suite (`notebooks/` & `scripts/generate_colab_variants.py`)
Update `scripts/generate_colab_variants.py` and regenerate all 4 GPU-specific Colab notebooks (`SE374_Colab_WSSS_T4_Sanity150.ipynb`, `SE374_Colab_WSSS_T4_Full4986.ipynb`, `SE374_Colab_WSSS_L4_Full4986.ipynb`, `SE374_Colab_WSSS_A100_Full4986.ipynb`) so that they:
- Automatically download and verify **both** official BCSS checkpoints (`stage1_checkpoint_trained_on_bcss.pth` ID `118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc` and `stage2_checkpoint_trained_on_bcss.pth` ID `1cXm29ko8UevbaeW6YreSg-vuNaQiN_je`).
- Instantiate both `model` (Stage-2 DeepLabV3+ ResNet-101) and `model_stage1` (Stage-1 ResNet-38d `Net_CAM`) and pass `model_stage1` into `MultiMethodWSSSEngine`.
- Run all 5 repo-grounded SOTA methods on 100% real BCSS-WSSS images, export complete ablation tables (mIoU, Macro Dice, per-class IoU, paired Wilcoxon/t-test p-value vs M1), and render publication-grade multi-method visual comparison grids.

## Acceptance Criteria

### Independent Recheck & Programmatic Verification
- [ ] Zero 0-byte `.md` files exist anywhere under `docs/` (`docs/papers/`, `docs/research/`, `docs/references/`, `docs/architecture/`). Every paper `.md` file in `docs/papers/` is `>= 1,900` bytes.
- [ ] `WSSSTissueStage1NetCAM` in `src/models/wsss_inference_methods.py` has exact parameter shapes matching `repos/WSSS-Tissue/network/resnet38d.py` (`conv1a`, `b2`..`b7`, `bn7`, `fc8`) and passes forward/CAM unit tests in `tests/test_colab_notebook.py`.
- [ ] All 4 Colab notebooks in `notebooks/` are regenerated via `cmd /c "python scripts/generate_colab_variants.py"` and contain both Google Drive checkpoint IDs (`118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc` and `1cXm29ko8UevbaeW6YreSg-vuNaQiN_je`) and `WSSSTissueStage1NetCAM`.
- [ ] 100% of unit and governance tests pass when running `cmd /c "python -m pytest tests/test_colab_notebook.py tests/test_governance_invariants.py -v"`.
- [ ] Zero non-ASCII dashes (`\u2013`, `\u2014`, `\u2012`, `\u2015`, `\u2212`) exist in any modified `.py`, `.md`, or `.ipynb` file (Invariant 12), and `graphify update .` succeeds.
- [ ] Note on Windows shell execution: Always wrap shell commands in `cmd /c "..."` (or `cmd /c '...'` in PowerShell) per `AGENTS.md` Invariant 13, and do not leave any temporary scratch scripts in the workspace per Invariant 15.


## 2026-10-20T11:39:40Z

URGENT DIRECTIVE FROM USER - STRICT 3-TEAM ORGANIZATIONAL STRUCTURE REQUIRED:

The user explicitly requires the Project Orchestrator / Leader (6de01969-97e8-4fac-b013-ba88aa268158) to organize and divide the subagents into 3 distinct, specialized teams (do NOT lump everything into generic workers):

1. Team 1 - Research Team (Team Research):
   - Dedicated strictly to reading the 30 academic papers in docs/papers/ (and docs/references/), cross-referencing the 24 cloned GitHub repositories in repos/ (WSSS-Tissue, LowMagWSS, ARML, PistoSeg, HisynSeg, PBIP, ConStruct, S2C, SAM2PATH, PFM_Segmentation, SEINE, WaWeHis), restoring the 9 remaining 0-byte Markdown dossiers in docs/papers/ and docs/references/, updating docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md, and formulating the exact mathematical & architectural specifications to improve our 68.71% mIoU baseline.

2. Team 2 - Implementation / Fix Team (Team Implement/Fix):
   - Dedicated strictly to coding, implementing, and fixing based on Team Research's specifications:
     - Porting WSSSTissueStage1NetCAM (ResNet38d + Net_CAM from repos/WSSS-Tissue/network/resnet38d.py & resnet38_cls.py) and the 5 repo-grounded SOTA methods into src/models/wsss_inference_methods.py.
     - Updating scripts/generate_colab_variants.py and regenerating all 4 GPU Colab notebooks (SE374_Colab_WSSS_T4_Sanity150.ipynb, SE374_Colab_WSSS_T4_Full4986.ipynb, SE374_Colab_WSSS_L4_Full4986.ipynb, SE374_Colab_WSSS_A100_Full4986.ipynb) with both Stage-1 (118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc) and Stage-2 (1cXm29ko8UevbaeW6YreSg-vuNaQiN_je) checkpoints.
     - Updating unit tests in tests/test_colab_notebook.py.

3. Team 3 - Recheck / Audit Team (Team Recheck/Audit):
   - Independent verification and auditing team that NEVER writes the initial implementation, but strictly cross-checks Team 1 (Research) and Team 2 (Implement/Fix) against all acceptance criteria:
     - Verifying zero 0-byte .md files in docs/ and all docs/papers/*.md >= 1,900 bytes.
     - Verifying exact layer/parameter parity of WSSSTissueStage1NetCAM against repos/WSSS-Tissue.
     - Running cmd /c "python -m pytest tests/test_colab_notebook.py tests/test_governance_invariants.py -v".
     - Auditing 100% compliance with AGENTS.md Invariant 07 (Zero Mockup), Invariant 08 (Patient-Level Split), Invariant 12 (ASCII hyphen 0x2D only), Invariant 15 (Zero scratch scripts), and running graphify update ..
     - Sending any defects back to Team 2 (Team Implement/Fix) until 100% pass.


## 2026-10-20T11:52:35Z

URGENT HEADCOUNT MANDATE FROM USER - EXPAND TEAM SIZES IMMEDIATELY:

The user explicitly clarified the required minimum headcount for each of the 3 teams under the Project Orchestrator (6de01969-97e8-4fac-b013-ba88aa268158):
- Team 1 (Team Research): MUST have >= 2 agents (currently has 2 agents: team1_researcher_papers & team1_researcher_repos - optionally add a 3rd research specialist team1_researcher_math_ablation if helpful).
- Team 2 (Team Implement/Fix): MUST have >= 4 agents (4+ distinct subagents working in parallel, NOT just 1 or 2!). Immediately instruct the Orchestrator (6de01969-97e8-4fac-b013-ba88aa268158) to spawn at least 2 more Team Implement/Fix subagents so Team 2 has at least 4 agents:
  1. team2_fixer_engine (already active: Stage-1 WSSSTissueStage1NetCAM & Stage-2 core model architecture in src/models/wsss_inference_methods.py)
  2. team2_fixer_colab (already active: scripts/generate_colab_variants.py & 4 GPU Colab notebooks)
  3. team2_fixer_sota_modules (NEW - spawn immediately: dedicated to verifying and refining the 5 repo-grounded SOTA methods M1-M5, RFM affinity module from PistoSeg, Cross-Stage CAM fusion from HisynSeg/PBIP, and Long-Tail Prior Calibration in src/models/)
  4. team2_fixer_tests_integration (NEW - spawn immediately: dedicated to writing and expanding comprehensive unit tests in tests/test_colab_notebook.py for WSSSTissueStage1NetCAM, dual-checkpoint loading, and M1-M5 numerical invariants, and fixing any integration issues)
- Team 3 (Team Recheck/Audit): MUST have >= 4 agents (4+ distinct independent auditor subagents, NOT a single auditor!). Immediately instruct the Orchestrator to spawn at least 4 distinct auditor subagents for Team 3:
  1. team3_auditor_papers_docs: Audits all 30 paper .md files in docs/papers/, docs/references/, and docs/research/ (verifies zero 0-byte files, >= 1,900 bytes each, accuracy against PDFs, and zero non-ASCII dashes).
  2. team3_auditor_repo_parity: Audits code-to-code parity of src/models/wsss_inference_methods.py (WSSSTissueStage1NetCAM and M1-M5) line-by-line against repos/WSSS-Tissue, repos/LowMagWSS, repos/ARML, repos/PistoSeg, repos/HisynSeg, repos/PBIP, and repos/ConStruct).
  3. team3_auditor_colab_notebooks: Audits scripts/generate_colab_variants.py and all 4 generated .ipynb notebooks in notebooks/ (verifies dual checkpoint IDs 118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc and 1cXm29ko8UevbaeW6YreSg-vuNaQiN_je, hardware configs for T4/L4/A100, and JSON syntax validity).
  4. team3_auditor_governance_pytest: Runs full pytest suite (tests/test_colab_notebook.py, tests/test_governance_invariants.py), audits all 25 AGENTS.md invariants (Zero Mockup, Patient-Level Split, ASCII hyphen 0x2D only, Zero scratch scripts), and runs graphify update ..

Relay this immediately to the Project Orchestrator (6de01969-97e8-4fac-b013-ba88aa268158) so that Team 1 has >= 2 agents, Team 2 has >= 4 agents, and Team 3 has >= 4 agents (total >= 10 specialized subagents under the Orchestrator), and report back the conversation IDs of all spawned agents!


## 2026-10-20T12:23:01Z

USER TIME CHECK: "hơi lâu rồi á, dự kiến bao lâu xong"
- Team 1 (Research) is 100% DONE.
- Team 2 (Implement/Fix) is 100% DONE.
- Team 3 (Recheck/Audit: repo parity, colab notebooks, governance pytest) is 100% APPROVED (42/42 pytest passed).
- Final step: team1_remediation_worker wrap-up, team3_auditor_papers_docs sign-off, and independent Victory Audit.
- Target completion: within 2-4 minutes.
