# Orchestration Plan - SE374.R11 WSSS Project

## Objective
Deliver a production-ready, clinically sound Weakly-Supervised Histopathology Semantic Segmentation (WSSS) system for BCSS-WSSS and Whole Slide Images, rigorously validated through tri-party adversarial checking, and document all findings in product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md.

## Architecture & Work Packets
1. **Module 1 (Data Ingestion & Clinical Tissue Gating)**:
   - Target files: `src/datasets/bcss_dataset.py`, `src/datasets/tissue_gating.py`
   - Patient-level splitting (zero leakage across 151 patients: 91 train, 30 val, 30 test)
   - Clinical Otsu tissue gating in HSV space (Saturation & Value channels) to eliminate blank glass slide artifacts
   - Mask and patch extraction pipeline
2. **Module 2 (Model Backbone & PDA CAM Generator)**:
   - Target files: `src/models/backbone.py`, `src/models/pda_generator.py`, `src/models/wsss_tissue.py`
   - Multi-stage feature backbone (ResNet50 / ConvNeXt with dilated convs)
   - Progressive Dropout Attention (PDA) for CAM expansion without boundary bleeding
   - 4 Benchmark Configurations:
     * Config 1: Vanilla CAM Baseline (Han et al., MedIA 2022)
     * Config 2: Data-Centric Augmentation (Baseline + ClassMix)
     * Config 3: Spatial Affinity Relocalization (ARML, MICCAI 2024)
     * Config 4: Proposed Adaptive Pipeline (Otsu Gating + Dynamic Progressive Thresholding + Uncertainty Regularization)
3. **Module 3 (Segmentation Network & Losses/Metrics)**:
   - Target files: `src/models/deeplabv3plus.py`, `src/losses_metrics/losses.py`, `src/losses_metrics/metrics.py`
   - DeepLabV3+ with Classification Gate Mechanism (CGM)
   - Multi-layer pseudo-supervision loss and boundary uncertainty regularization
   - Zero-mockup clinical metrics: mIoU, mDice, per-class IoU, Volume-to-Perimeter ratio (VTP), Tumor Stroma Ratio (TSR)
4. **Module 4 (Clinical Web Workstation & API)**:
   - Target files: `src/visualization/clinical_viewer.py`, `src/visualization/api_server.py`, static web assets
   - FastAPI REST API for slide ingestion, patch inference, and heatmap streaming
   - Interactive clinical viewer with OpenSeadragon deep zoom support for WSI and ROI patches
5. **Pipeline & Comparative Benchmarks**:
   - Target file: `src/pipeline/comparative_runner.py`
   - Orchestrated execution of the 4 benchmark configurations with logging and artifact capture
6. **Tri-Party Adversarial Verification**:
   - Checker Team A: Code Quality, PEP 8, Invariant 12 (100% ASCII 0x2D hyphens), Invariant 13 (cmd /c), Invariant 15 (zero scratch scripts)
   - Checker Team B: Scientific Integrity, Patient-Level Zero Leakage (Invariant 08), Zero Dummy Metrics (Invariant 07), Math correctness
   - Checker Team C: Adversarial Clinical & Stress Tester (Blank glass slides, adipose tissue, single-item batches, memory safety < 6 GB VRAM)
7. **Final Technical Report**:
   - Target file: `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md`
   - Complete academic report synthesising theory, architecture, experiments, metrics, clinical interpretation, and discussion.

## Phase Execution Roadmap
- **Phase 0**: Codebase Survey (Explorer) -> examine existing src/, tests/, data/, and docs/.
- **Phase 1**: Implementation & Unit Testing (Workers for Teams 1, 2, 3, 4).
- **Phase 2**: Tri-Party Adversarial Verification (Checkers A, B, C) & Closed-Loop Fixes.
- **Phase 3**: Benchmark Execution & Results Logging (`src/pipeline/comparative_runner.py`).
- **Phase 4**: Final Technical Report Generation (`product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md`).
- **Phase 5**: Final Review and Halting for Human Commander Inspection.
