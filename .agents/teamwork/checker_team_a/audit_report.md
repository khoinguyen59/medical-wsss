# Comprehensive Audit Report - Checker Team A: Code Quality & Master Invariants

- **Auditor**: Checker Team A (Code Quality & Master Invariants Auditor)
- **Role**: Reviewer & Adversarial Critic
- **Target Repository**: SE374.R11 WSSS Medical AI Codebase
- **Audit Date**: 2026-10-18T16:32:00Z
- **Verdict**: **REQUEST_CHANGES**

---

## 1. Executive Summary

Checker Team A performed an independent, adversarial audit of the SE374.R11 WSSS codebase across six core dimensions:
1. Static Code Quality, Architecture & PEP 8 Conformance in `src/` and `tests/`.
2. Master Invariant 12 Compliance (Strict ASCII 0x2D hyphens across code and docs).
3. Master Invariant 13 Compliance (Windows OS `cmd /c` shell execution).
4. Master Invariant 15 Compliance (Zero one-off scratch scripts).
5. Comprehensive Test Suite Execution (`pytest tests/ -v`).
6. Codebase Integrity and absence of facade implementations.

### Summary Scorecard

| Audit Dimension | Standard / Invariant | Status | Key Findings |
| :--- | :--- | :---: | :--- |
| **Static Code Quality** | PEP 8, Clean Architecture, Type Annotations | **PASS with Minor Notices** | 100% valid syntax, zero import errors across 15 modules. Minor unused imports and typing modernization notices from Ruff. |
| **Hyphen Compliance** | Invariant 12 (ASCII 0x2D '-' only) | **DEFECT DETECTED** | 7 raw non-ASCII hyphen characters found across 2 files (`src/utils/deliverable_auditor.py` and `tests/test_deliverable_auditor.py`). |
| **Windows Shell Execution** | Invariant 13 (`cmd /c` prefix) | **PASS (100%)** | All documented commands in `README.md` and documentation use `cmd /c`. Zero bare subprocess calls. |
| **Scratch Script Hygiene** | Invariant 15 (Zero scratch scripts) | **PASS (100%)** | Zero temporary/scratch scripts in root, `src/`, or `scratch/`. All code lives in structured modules. |
| **Test Suite Execution** | 61+ unit tests passing | **PASS (100%)** | 61 of 61 tests passed in 9.43s with zero failures or skips. |
| **Integrity & Real Logic** | Invariant 07 (Zero Mockup) | **PASS (100%)** | Genuine PyTorch tensor math, real confusion matrices, authentic ASPP and PDA implementations. |

---

## 2. Detailed Audit Findings

### Finding A-01: [CRITICAL] Non-ASCII Hyphen Characters in Source and Test Files (Invariant 12 Violation)

- **Severity**: Critical (Violates Master Governance Invariant 12).
- **Rule Violated**: `[INVARIANT-12] Strict Standard Keyboard Hyphen Only (No AI Em-Dash/En-Dash)`. Standard ASCII 0x2D '-' is the only permitted hyphen character.
- **Observations**:
  A regex scan across all source files revealed exactly 7 non-ASCII hyphen/dash characters:
  1. `src/utils/deliverable_auditor.py:393`:
     ```python
     # Forbidden dashes: em-dash (— \u2014), en-dash (– \u2013), horizontal bar (― \u2015)
     ```
     Contains literal em-dash `—` (U+2014), en-dash `–` (U+2013), and horizontal bar `―` (U+2015) in a comment.
  2. `tests/test_deliverable_auditor.py:58`:
     ```python
     """Auditor must enforce ASCII hyphen and block em-dash (—) and en-dash (–)."""
     ```
     Contains literal em-dash `—` (U+2014) and en-dash `–` (U+2013) in docstring.
  3. `tests/test_deliverable_auditor.py:62`:
     ```python
     f.write("# Báo cáo — Vi phạm\n\nKhoảng cách 10 – 20px.\n")
     ```
     Contains literal em-dash `—` (U+2014) and en-dash `–` (U+2013) written as raw UTF-8 characters instead of ASCII escape sequences.
- **Why This Is a Problem**:
  Invariant 12 prohibits raw non-ASCII dashes in code, markdown, proposals, slides, and messages. In Python files, literal Unicode dashes cause encoding issues on Windows consoles (e.g. `cp1252` encoding errors) and violate the strict standard keyboard hyphen mandate.
- **Required Remediation**:
  1. In `src/utils/deliverable_auditor.py:393`, rewrite the comment to use pure ASCII notation:
     `# Forbidden dashes: em-dash (\u2014), en-dash (\u2013), horizontal bar (\u2015)`
  2. In `tests/test_deliverable_auditor.py:58`, rewrite docstring to:
     `"""Auditor must enforce ASCII hyphen and block em-dash (\u2014) and en-dash (\u2013)."""`
  3. In `tests/test_deliverable_auditor.py:62`, escape the test payload characters:
     `f.write("# Báo cáo \u2014 Vi phạm\n\nKhoảng cách 10 \u2013 20px.\n")`
  4. In `tests/test_governance_invariants.py`, update `test_no_ai_em_dash_in_markdown` or add `test_no_ai_em_dash_in_python_code` to scan both `.md` and `.py` files to prevent future regressions.

---

### Finding A-02: [MINOR] Unused Imports and Unused Unpacked Variables (PEP 8 / Ruff)

- **Severity**: Minor (Code cleanliness & style conformance).
- **Observations**:
  Ruff linter flagged unused imports and unused unpacked variables in test and visualization files:
  1. `tests/test_deliverable_auditor.py:12`: `import pytest` unused.
  2. `tests/test_patient_split.py:1`: `import numpy as np` unused.
  3. `tests/test_pda_backbone.py:18, 24, 28, 30`: `torch.nn as nn`, `ResidualConvBlock`, `SpatialAffinityRelocalization`, `apply_classmix_augmentation` unused.
  4. `tests/test_patient_split.py:80, 89`: Unused unpacked variables `fname`, `fname_v` (should be `_fname`, `_fname_v`).
  5. `tests/test_deeplabv3plus.py:202, 220, 232`: Unused unpacked variable `logits` (should be `_, loss = ...`).
  6. `src/visualization/draw_all_proposal_figures.py:45, 208, 329`: Implicit Optional type annotations `output_svg: str = None` (PEP 484 recommends `output_svg: Optional[str] = None` or `output_svg: str | None = None`).
  7. `src/visualization/draw_pipeline_diagram.py:33`: Local variable `font_item` assigned but never used.
- **Why This Is a Problem**:
  Unused imports and variables introduce clutter and trigger static analysis warnings during automated CI/CD scans.
- **Suggested Fix**:
  Clean up unused imports and prefix unused unpack targets with an underscore `_`.

---

## 3. Invariant Verification Evidence Chain

### Invariant 12: Standard Keyboard Hyphen (ASCII 0x2D)
- **Method**: Exhaustive character code inspection using Python script scanning `src/`, `tests/`, `product/`, `docs/`, `README.md`, and `requirements.txt`.
- **Target Chars Tested**:
  - U+2010 (Hyphen)
  - U+2011 (Non-breaking hyphen)
  - U+2012 (Figure dash)
  - U+2013 (En-dash)
  - U+2014 (Em-dash)
  - U+2015 (Horizontal bar)
  - U+2212 (Minus sign)
  - U+FE58, U+FE63, U+FF0D (Small / fullwidth dash variants)
- **Results**:
  - `product/`: **0 violations**. All markdown and HTML documents are 100% compliant.
  - `docs/`: **0 violations**.
  - Root files (`README.md`, `requirements.txt`): **0 violations**.
  - `src/`: **3 violations** in `src/utils/deliverable_auditor.py:393` (comment).
  - `tests/`: **4 violations** in `tests/test_deliverable_auditor.py:58, 62` (docstring and test string).
  - Total violations: **7**.

### Invariant 13: Windows OS Standard Execution (`cmd /c`)
- **Method**: Regex inspection of markdown bash/shell code blocks and Python source files for subprocess/os.system invocations.
- **Results**:
  - `README.md`: All terminal instructions explicitly prefixed with `cmd /c` (e.g. `cmd /c "pip install -r requirements.txt"`, `cmd /c "pytest tests/"`, `cmd /c "flake8 src/ --max-line-length=120"`).
  - Source & Test files: Zero raw `subprocess.run`, `subprocess.Popen`, or `os.system` calls. All tests interface natively with model classes and functions in pure Python.
  - Status: **100% PASS**.

### Invariant 15: Zero One-Off Scratch Scripts
- **Method**: Complete filesystem walk across repository root, `src/`, `scratch/`, `scripts/`, and `experiments/`.
- **Results**:
  - Root directory contains zero `.py`, `.sh`, or `.bat` files.
  - `src/` contains 16 modular architectural components organized by Clean Architecture.
  - `scratch/` contains only `.gitkeep`.
  - `experiments/` contains only `experiments/benchmarks/`.
  - `scripts/` contains only `download_papers.py` and `.gitkeep`.
  - Status: **100% PASS**.

---

## 4. Test Suite Execution & Verification Log

- **Command**: `cmd /c "C:\Python313\python.exe -m pytest tests/ -v"`
- **Execution Output**:
  ```
  platform win32 -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0 -- C:\Python313\python.exe
  cachedir: .pytest_cache
  rootdir: C:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
  plugins: anyio-4.12.1, timeout-2.4.0
  collected 61 items

  tests/test_api_server.py::test_api_health_endpoint PASSED                [  1%]
  tests/test_api_server.py::test_predict_patch_json_base64 PASSED          [  3%]
  tests/test_api_server.py::test_predict_patch_multipart_file PASSED       [  4%]
  tests/test_api_server.py::test_predict_patch_otsu_gating_blank_glass PASSED [  6%]
  tests/test_api_server.py::test_predict_patch_missing_data_error PASSED   [  8%]
  tests/test_api_server.py::test_predict_wsi_multi_patch PASSED            [  9%]
  tests/test_api_server.py::test_serve_viewer_html PASSED                  [ 11%]
  tests/test_api_server.py::test_clinical_viewer_utilities PASSED          [ 13%]
  tests/test_benchmark_components.py::test_metrics_service_perfect_match PASSED [ 14%]
  tests/test_benchmark_components.py::test_metrics_service_partial_overlap PASSED [ 16%]
  tests/test_benchmark_components.py::test_clinical_biomarkers PASSED      [ 18%]
  tests/test_benchmark_components.py::test_otsu_clinical_gating PASSED     [ 19%]
  tests/test_benchmark_components.py::test_losses PASSED                   [ 21%]
  tests/test_benchmark_components.py::test_wsss_engine_all_configurations PASSED [ 22%]
  tests/test_benchmark_components.py::test_classmix_augmentation PASSED    [ 24%]
  tests/test_deeplabv3plus.py::test_aspp_output_shapes PASSED              [ 26%]
  tests/test_deeplabv3plus.py::test_decoder_output_shapes PASSED           [ 27%]
  tests/test_deeplabv3plus.py::test_deeplabv3plus_forward_pass PASSED      [ 29%]
  tests/test_deeplabv3plus.py::test_deeplabv3plus_gradient_flow PASSED     [ 31%]
  tests/test_deeplabv3plus.py::test_classification_gate_mechanism_direct PASSED [ 32%]
  tests/test_deeplabv3plus.py::test_cgm_extinguishes_absent_class_in_segmentation PASSED [ 34%]
  tests/test_deeplabv3plus.py::test_cgm_with_none_presence PASSED          [ 36%]
  tests/test_deeplabv3plus.py::test_deeplabv3plus_loss_ce PASSED           [ 37%]
  tests/test_deeplabv3plus.py::test_deeplabv3plus_loss_mlps PASSED         [ 39%]
  tests/test_deeplabv3plus.py::test_deeplabv3plus_loss_uncertainty PASSED  [ 40%]
  tests/test_deeplabv3plus.py::test_single_item_batch_edge_case PASSED     [ 42%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_valid_proposal_passes_audit PASSED [ 44%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_raw_phone_snapshot_name PASSED [ 45%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_empty_caption PASSED [ 47%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_forbidden_em_dash_and_en_dash PASSED [ 49%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_dummy_metrics_and_mockup PASSED [ 50%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_flags_low_resolution_images PASSED [ 52%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_ascii_art_diagrams PASSED [ 54%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_colorful_figures_and_enforces_monochrome PASSED [ 55%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_wrong_faculty_and_external_lab PASSED [ 57%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_blocks_leaked_rubrics_and_invariants PASSED [ 59%]
  tests/test_deliverable_auditor.py::TestDeliverableAuditor::test_valid_docx_proposal_passes_audit PASSED [ 60%]
  tests/test_governance_invariants.py::test_no_ai_em_dash_in_markdown PASSED [ 62%]
  tests/test_governance_invariants.py::test_invariants_24_and_25_defined_in_agents_md PASSED [ 63%]
  tests/test_governance_invariants.py::test_no_raw_phone_photos_in_product_directory PASSED [ 65%]
  tests/test_patient_split.py::test_extract_patient_id_tcga PASSED         [ 67%]
  tests/test_patient_split.py::test_verify_patient_leakage_on_sample_dataset PASSED [ 68%]
  tests/test_patient_split.py::test_verify_patient_leakage_catches_overlap PASSED [ 70%]
  tests/test_patient_split.py::test_split_patients_kfold PASSED            [ 72%]
  tests/test_patient_split.py::test_bcss_dataset_loader_dry_run PASSED     [ 73%]
  tests/test_patient_split.py::test_151_patient_split_manifest_zero_leakage PASSED [ 75%]
  tests/test_patient_split.py::test_load_patient_splits_invalid_path PASSED [ 77%]
  tests/test_pda_backbone.py::test_backbone_forward_and_feature_shapes PASSED [ 78%]
  tests/test_pda_backbone.py::test_backbone_custom_channels_and_dropout PASSED [ 80%]
  tests/test_pda_backbone.py::test_backbone_cam_computation PASSED         [ 81%]
  tests/test_pda_backbone.py::test_backbone_backward_compatibility_alias PASSED [ 83%]
  tests/test_pda_backbone.py::test_pda_initialization_and_forward_shapes PASSED [ 85%]
  tests/test_pda_backbone.py::test_pda_expansion_property PASSED           [ 86%]
  tests/test_pda_backbone.py::test_pda_boundary_bleeding_control PASSED    [ 88%]
  tests/test_pda_backbone.py::test_pda_generate_from_backbone PASSED       [ 90%]
  tests/test_pda_backbone.py::test_wsss_engine_all_configurations_clean_execution[baseline] PASSED [ 91%]
  tests/test_pda_backbone.py::test_wsss_engine_all_configurations_clean_execution[classmix] PASSED [ 93%]
  tests/test_pda_backbone.py::test_wsss_engine_all_configurations_clean_execution[relocalization] PASSED [ 95%]
  tests/test_pda_backbone.py::test_wsss_engine_all_configurations_clean_execution[proposed] PASSED [ 96%]
  tests/test_pda_backbone.py::test_wsss_engine_proposed_mode_uncertainty PASSED [ 98%]
  tests/test_pda_backbone.py::test_wsss_engine_dynamic_progressive_thresholding PASSED [100%]

  ============================= 61 passed in 9.43s ==============================
  ```

---

## 5. Adversarial Stress-Testing & Failure Mode Analysis

As an adversarial critic, Checker Team A conducted the following stress-tests:
1. **Hyphen Leakage in Test Payloads**:
   The existing `test_no_ai_em_dash_in_markdown` in `tests/test_governance_invariants.py` only scanned `**/*.md`. This allowed raw non-ASCII unicode characters inside Python test payloads and docstrings to pass unnoticed. The test must be strengthened to cover `.py` files as well.
2. **Empty / Absent Class Logits in CGM**:
   Tested `ClassificationGateMechanism` with all-zeros presence vector. The penalty `-1e4` successfully extinguished absent logits without gradient overflow or NaN values.
3. **Single-Item Batch Handling**:
   Confirmed that `DeepLabV3PlusWithCGM` supports single-item batches `(1, 3, 224, 224)` across all loss modes (`ce`, `mlps`, `uncertainty`), avoiding standard PyTorch BatchNorm batch size 1 failure by using appropriate model configuration.
4. **Patient Leakage Boundary**:
   Confirmed `verify_patient_leakage` correctly raises `ValueError` on any patient overlap and verified that the official 151-case BCSS manifest exhibits 0% overlap.

---

## 6. Actionable Fix Directives for Team Lead

To achieve 100% compliance and obtain an unconditional **APPROVE** verdict, the following fixes must be directed:
1. **Fix Invariant 12 in `src/utils/deliverable_auditor.py` line 393**:
   Replace raw Unicode dash characters with ASCII escape representations `\u2014`, `\u2013`, `\u2015`.
2. **Fix Invariant 12 in `tests/test_deliverable_auditor.py` lines 58 & 62**:
   Replace raw Unicode dash characters with ASCII escape sequences in docstrings and string literals (`\u2014`, `\u2013`).
3. **Strengthen Governance Tests in `tests/test_governance_invariants.py`**:
   Expand `test_no_ai_em_dash_in_markdown` to also scan `.py` files across `src/` and `tests/`.
4. **Optional PEP 8 Polish**:
   Remove unused imports in `tests/test_deliverable_auditor.py`, `tests/test_patient_split.py`, and `tests/test_pda_backbone.py`.
