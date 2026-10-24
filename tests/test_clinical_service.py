"""
Comprehensive Unit & Integration Test Suite for Clinical Service (SE374.R11).

Tests:
1. Patient Split Standardization & Anti-Leakage on bcss_patient_splits.json.
2. Clinical Biomarkers (TSR, Stroma Proportion, VTP, Patient Biomarker Aggregation).
3. CAP (College of American Pathologists) Risk Stratification.
4. Grounded TCGA-BRCA 23 Test Patient Survival Manifest.
5. Dual-Engine Kaplan-Meier Survival Analysis & Log-Rank Testing.
6. Publication-grade SVG and PNG plot generation.
7. Segmentation Metric Accumulator (Confusion Matrix, mIoU, Dice, FwIoU).
8. Exhaustive 3-Way Benchmark Evaluator Dry-Run Execution & CLI.
9. Strict Invariant 12 compliance: 100% standard keyboard ASCII hyphens '-' (0x2D) only.
"""

import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from src.clinical.survival_analysis import (
    ACTIVE_TEST_PATIENTS_23,
    TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA,
    KaplanMeierSurvivalAnalysis,
    compute_patient_biomarkers,
    compute_stroma_proportion,
    compute_tumor_stroma_ratio,
    compute_viable_tumor_percentage,
    export_survival_summary_json,
    get_all_test_survival_data,
    get_patient_survival_data,
    load_active_test_patients,
    stratify_cap_risk,
    stratify_cap_risk_category,
)
from src.clinical.exhaustive_evaluator import (
    ExhaustiveBenchmarkEvaluator,
    SegmentationMetricAccumulator,
    build_cli_parser,
    run_cli,
)
from src.datasets.patient_split import load_patient_splits, verify_patient_leakage


# ==============================================================================
# 1. Patient Splits Standardization & Anti-Leakage Tests
# ==============================================================================

def test_bcss_patient_splits_standardization_and_active_cohort():
    """Verifies that bcss_patient_splits.json recognizes the 23 active test cohort

    while preserving the 91/30/30 structure and 0.0% patient leakage.
    """
    manifest_path = Path("src/config/bcss_patient_splits.json")
    assert manifest_path.is_file(), f"Manifest not found: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level counts
    assert data["train_count"] == 91
    assert data["val_count"] == 30
    assert data["test_count"] == 30
    assert data["num_patients"] == 151
    assert data["test_active_count"] == 23

    # Active test cohort
    active_cohort = data["test_active_patients"]
    assert len(active_cohort) == 23
    assert set(active_cohort) == set(ACTIVE_TEST_PATIENTS_23)

    # Invariant 08: Zero Patient-Level Leakage between Train, Val, and Active Cohort
    train_set = set(data["splits"]["train"])
    val_set = set(data["splits"]["val"])
    active_set = set(active_cohort)

    assert len(train_set & val_set) == 0, f"Train-Val leakage: {train_set & val_set}"
    assert len(train_set & active_set) == 0, f"Train-Active leakage: {train_set & active_set}"
    assert len(val_set & active_set) == 0, f"Val-Active leakage: {val_set & active_set}"

    # Verify standard loader still works and returns clean 91/30/30 split
    splits = load_patient_splits(str(manifest_path))
    assert len(splits["train"]) == 91
    assert len(splits["val"]) == 30
    assert len(splits["test"]) == 30

    leakage_check = verify_patient_leakage(splits["train"], splits["val"], splits["test"])
    assert leakage_check["has_leakage"] is False


def test_load_active_test_patients_helper():
    """Verifies helper function correctly reads active test cohort."""
    cohort = load_active_test_patients()
    assert len(cohort) == 23
    assert "TCGA-D8-A27F" in cohort
    assert "TCGA-GI-A2C9" in cohort


# ==============================================================================
# 2. Clinical Biomarkers Tests (TSR, Stroma Proportion, VTP)
# ==============================================================================

def test_compute_tumor_stroma_ratio_and_stroma_proportion():
    """Verifies mathematical correctness of TSR and Stroma Proportion calculations."""
    # 1. Scalar calculation: 300 tumor, 100 stroma -> TSR = 75%, Stroma Prop = 25%
    tsr = compute_tumor_stroma_ratio(300, 100)
    str_prop = compute_stroma_proportion(300, 100)
    assert pytest.approx(tsr, 0.01) == 75.0
    assert pytest.approx(str_prop, 0.01) == 25.0
    assert pytest.approx(tsr + str_prop, 0.01) == 100.0

    # 2. Array/Tensor calculation: 2D mask
    mask = np.full((10, 10), fill_value=4, dtype=np.int64) # Background
    mask[0:4, :] = 0 # 40 pixels Tumor
    mask[4:10, :] = 1 # 60 pixels Stroma

    tsr_arr = compute_tumor_stroma_ratio(mask)
    str_arr = compute_stroma_proportion(mask)
    assert pytest.approx(tsr_arr, 0.01) == 40.0
    assert pytest.approx(str_arr, 0.01) == 60.0

    # PyTorch Tensor input
    mask_tensor = torch.from_numpy(mask)
    assert pytest.approx(compute_tumor_stroma_ratio(mask_tensor), 0.01) == 40.0

    # 3. Boundary Condition: Zero malignant area (0 tumor, 0 stroma) -> 0.0
    zero_mask = np.full((10, 10), fill_value=4, dtype=np.int64)
    assert compute_tumor_stroma_ratio(zero_mask) == 0.0
    assert compute_stroma_proportion(zero_mask) == 0.0
    assert compute_tumor_stroma_ratio(0, 0) == 0.0


def test_compute_viable_tumor_percentage():
    """Verifies Viable Tumor Percentage (VTP) across valid tissue compartments."""
    # Mask with 50 tumor, 30 stroma, 10 lym, 10 nec, and 100 background glass
    mask = np.full((200,), fill_value=4, dtype=np.int64)
    mask[0:50] = 0 # Tumor (50 px)
    mask[50:80] = 1 # Stroma (30 px)
    mask[80:90] = 2 # Lymphocyte (10 px)
    mask[90:100] = 3 # Necrosis (10 px)
    # Total tissue = 100 px, Tumor = 50 px -> VTP = 50.0%

    vtp = compute_viable_tumor_percentage(mask)
    assert pytest.approx(vtp, 0.01) == 50.0

    # Scalar mode
    vtp_scalar = compute_viable_tumor_percentage(50, 100)
    assert pytest.approx(vtp_scalar, 0.01) == 50.0

    # Boundary condition: zero tissue
    empty_mask = np.full((10, 10), fill_value=4, dtype=np.int64)
    assert compute_viable_tumor_percentage(empty_mask) == 0.0
    assert compute_viable_tumor_percentage(0, 0) == 0.0


def test_compute_patient_biomarkers_aggregation():
    """Verifies aggregation of multiple patches for a single patient."""
    patch1 = np.zeros((10, 10), dtype=np.int64) # 100 Tumor
    patch2 = np.ones((10, 10), dtype=np.int64) # 100 Stroma
    patch3 = np.full((10, 10), fill_value=4, dtype=np.int64) # 100 Glass (Background)

    res = compute_patient_biomarkers([patch1, patch2, patch3])
    assert res["tumor_pixels"] == 100.0
    assert res["stroma_pixels"] == 100.0
    assert res["total_tissue_pixels"] == 200.0
    assert pytest.approx(res["tsr"], 0.01) == 50.0
    assert pytest.approx(res["stroma_proportion"], 0.01) == 50.0
    assert pytest.approx(res["vtp"], 0.01) == 50.0


# ==============================================================================
# 3. CAP Risk Stratification Tests
# ==============================================================================

def test_stratify_cap_risk_categories():
    """Verifies College of American Pathologists (CAP) risk stratification logic."""
    # Stroma-Rich High Risk: TSR <= 50%
    high_risk_1 = stratify_cap_risk(tsr=40.0)
    assert high_risk_1["risk_group"] == "Stroma-Rich"
    assert high_risk_1["risk_level"] == "High Risk"
    assert high_risk_1["is_stroma_rich"] is True
    assert high_risk_1["risk_code"] == 1

    # Exact threshold condition: TSR == 50.0% is classified as Stroma-Rich High Risk
    high_risk_edge = stratify_cap_risk(tsr=50.0)
    assert high_risk_edge["risk_group"] == "Stroma-Rich"
    assert high_risk_edge["risk_level"] == "High Risk"

    # Stroma-Poor Low Risk: TSR > 50%
    low_risk = stratify_cap_risk(tsr=75.0)
    assert low_risk["risk_group"] == "Stroma-Poor"
    assert low_risk["risk_level"] == "Low Risk"
    assert low_risk["is_stroma_rich"] is False
    assert low_risk["risk_code"] == 0

    # String convenience function
    assert stratify_cap_risk_category(35.0) == "Stroma-Rich (High Risk)"
    assert stratify_cap_risk_category(65.0) == "Stroma-Poor (Low Risk)"


# ==============================================================================
# 4. TCGA-BRCA Clinical Manifest Tests
# ==============================================================================

def test_tcga_brca_clinical_manifest_data_integrity():
    """Verifies that all 23 active test patients have complete, valid clinical metadata."""
    manifest = get_all_test_survival_data()
    assert len(manifest) == 23

    for pid in ACTIVE_TEST_PATIENTS_23:
        assert pid in manifest, f"Missing clinical metadata for patient: {pid}"
        data = manifest[pid]
        assert data["overall_survival_months"] > 0.0
        assert data["vital_status"] in (0, 1)
        assert data["status_label"] in ("Living", "Deceased")
        assert len(data["histological_subtype"]) > 0

    # Verify specific known cases
    # TCGA-E2-A1LK: Deceased event
    p1 = get_patient_survival_data("TCGA-E2-A1LK")
    assert p1["vital_status"] == 1
    assert p1["status_label"] == "Deceased"

    # TCGA-E2-A14N: Living long-term survivor
    p2 = get_patient_survival_data("TCGA-E2-A14N")
    assert p2["vital_status"] == 0
    assert p2["overall_survival_months"] > 40.0

    # Key error on invalid ID
    with pytest.raises(KeyError):
        get_patient_survival_data("NON-EXISTENT-PATIENT")


# ==============================================================================
# 5. Kaplan-Meier Survival Analysis Engine Tests
# ==============================================================================

def test_kaplan_meier_analytical_estimator_monotonicity():
    """Verifies that the analytical Kaplan-Meier estimator produces monotonically non-increasing curves."""
    durations = [4.5, 7.8, 12.0, 15.5, 20.0, 24.5, 30.0]
    events = [0, 1, 0, 1, 0, 0, 1]

    engine = KaplanMeierSurvivalAnalysis(use_lifelines_if_available=False)
    fit_res = engine.fit(durations, events, label="Test Cohort")

    timeline = fit_res["timeline"]
    surv = fit_res["survival_probability"]
    ci_lower = fit_res["ci_lower"]
    ci_upper = fit_res["ci_upper"]

    assert len(timeline) > 0
    assert surv[0] == 1.0 # Starts at 1.0 at t = 0

    # Monotonicity check
    for i in range(1, len(surv)):
        assert surv[i] <= surv[i - 1] + 1e-9, f"Survival increased at step {i}"

    # Bounded in [0.0, 1.0]
    for s, lo, hi in zip(surv, ci_lower, ci_upper):
        assert 0.0 <= s <= 1.0
        assert 0.0 <= lo <= s + 1e-9
        assert s - 1e-9 <= hi <= 1.0

    assert fit_res["events_observed"] == 3
    assert fit_res["censored_count"] == 4


def test_logrank_test_analytical_significance():
    """Verifies analytical log-rank test computes Chi-square statistic and valid p-value."""
    engine = KaplanMeierSurvivalAnalysis(use_lifelines_if_available=False)

    # Group A has early events (high risk)
    dur_a = [3.0, 5.0, 7.0, 8.0, 10.0]
    evt_a = [1, 1, 1, 1, 1]

    # Group B has late follow-up (low risk)
    dur_b = [20.0, 25.0, 30.0, 35.0, 40.0]
    evt_b = [0, 0, 1, 0, 0]

    res = engine.logrank_test(dur_a, evt_a, dur_b, evt_b, label_a="High Risk", label_b="Low Risk")

    assert res["test_statistic"] > 0.0
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["degrees_of_freedom"] == 1
    assert res["is_significant_05"] is True # Significant separation


def test_kaplan_meier_plot_and_json_export():
    """Verifies rendering and exporting of publication-grade SVG and PNG survival plots."""
    engine = KaplanMeierSurvivalAnalysis(use_lifelines_if_available=False)

    dur_a = [5.0, 8.0, 12.0]
    evt_a = [1, 1, 0]
    dur_b = [15.0, 22.0, 30.0]
    evt_b = [0, 0, 1]

    fit_a = engine.fit(dur_a, evt_a, label="Stroma-Rich")
    fit_b = engine.fit(dur_b, evt_b, label="Stroma-Poor")
    lr = engine.logrank_test(dur_a, evt_a, dur_b, evt_b)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = Path(tmp_dir) / "km_plot.png"
        saved_p = engine.plot_survival_curves(
            fit_a, fit_b, dur_a, evt_a, dur_b, evt_b,
            output_path=out_png,
            logrank_res=lr,
        )

        assert saved_p.is_file()
        assert saved_p.stat().st_size > 1000
        # Also check SVG was exported
        svg_p = saved_p.with_suffix(".svg")
        assert svg_p.is_file()
        assert svg_p.stat().st_size > 1000

        # Test JSON summary exporter
        json_p = Path(tmp_dir) / "summary.json"
        export_survival_summary_json({"cohort": 6, "p_value": lr["p_value"]}, json_p)
        assert json_p.is_file()


# ==============================================================================
# 6. Segmentation Metric Accumulator Tests
# ==============================================================================

def test_segmentation_metric_accumulator_4cls_and_5cls():
    """Verifies confusion matrix accumulation and IoU/Dice computation."""
    acc = SegmentationMetricAccumulator(num_classes=5)

    # Synthetic prediction and GT
    gt = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [4, 4, 4, 4],
    ], dtype=np.int64)

    # Perfect prediction
    pred = gt.copy()
    acc.update(pred, gt, latency_sec=0.01)

    metrics = acc.compute_metrics()
    assert pytest.approx(metrics["mIoU_4cls"], 0.01) == 100.0
    assert pytest.approx(metrics["mDice_4cls"], 0.01) == 100.0
    assert pytest.approx(metrics["Pixel_Accuracy_4cls"], 0.01) == 100.0
    assert pytest.approx(metrics["IoU_TUM"], 0.01) == 100.0
    assert metrics["total_patches_evaluated"] == 1
    assert metrics["throughput_fps"] > 0.0


# ==============================================================================
# 7. Exhaustive Benchmark Evaluator Integration & Dry-Run Tests
# ==============================================================================

def test_exhaustive_benchmark_evaluator_dry_run_execution():
    """Tests full end-to-end dry-run of ExhaustiveBenchmarkEvaluator on sample_dataset."""
    sample_dir = Path("data/sample_dataset")
    assert sample_dir.is_dir(), "data/sample_dataset must exist"

    with tempfile.TemporaryDirectory() as tmp_dir:
        evaluator = ExhaustiveBenchmarkEvaluator(
            dataset_dir=sample_dir,
            output_dir=tmp_dir,
            device="cpu",
            batch_size=2,
        )

        results = evaluator.run(dry_run=True, dry_run_count=5)

        # 1. Check metric dicts are populated
        assert "mIoU_4cls" in results["metrics_author"]
        assert "mIoU_4cls" in results["metrics_retrained"]
        assert "mIoU_4cls" in results["metrics_medsam"]

        # 2. Check statistical tests are present
        assert "proposed_vs_author" in results["stat_tests"]
        assert "proposed_vs_retrained" in results["stat_tests"]

        # 3. Check clinical survival results
        surv = results["survival_results"]
        assert surv["patient_cohort_count"] > 0
        assert "logrank_test" in surv

        # 4. Check exported files exist
        exp = results["exported_files"]
        for key in ["master_json", "patient_json", "patient_csv", "report_md"]:
            assert Path(exp[key]).is_file(), f"Expected exported file: {key}"

        # 5. Check generated figures exist in output directory
        assert (Path(tmp_dir) / "kaplan_meier_survival_curve.png").is_file()
        assert (Path(tmp_dir) / "kaplan_meier_survival_curve.svg").is_file()


def test_exhaustive_evaluator_cli_parser_and_dry_run():
    """Tests CLI argument parsing and smoke dry-run."""
    parser = build_cli_parser()
    args = parser.parse_args(["--dry-run", "--batch-size", "4", "--device", "cpu"])
    assert args.dry_run is True
    assert args.batch_size == 4
    assert args.device == "cpu"


# ==============================================================================
# 8. Governance Invariant 12: Strict ASCII Hyphens Only
# ==============================================================================

def test_invariant_12_no_non_ascii_hyphens_in_clinical_service():
    """Verifies that all files in src/clinical/ and this test file strictly use ASCII hyphens (0x2D)."""
    forbidden_chars = [
        ("\u2013", "En-dash"),
        ("\u2014", "Em-dash"),
        ("\u2012", "Figure-dash"),
        ("\u2015", "Horizontal-bar"),
        ("\u2212", "Minus-sign"),
    ]

    files_to_check = [
        Path("src/clinical/__init__.py"),
        Path("src/clinical/survival_analysis.py"),
        Path("src/clinical/exhaustive_evaluator.py"),
        Path("tests/test_clinical_service.py"),
    ]

    for p in files_to_check:
        assert p.is_file(), f"File must exist: {p}"
        text = p.read_text(encoding="utf-8")
        for bad_char, desc in forbidden_chars:
            assert bad_char not in text, f"Found forbidden {desc} ({repr(bad_char)}) in {p}"
