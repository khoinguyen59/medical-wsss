"""
Clinical Service & Exhaustive Evaluator Package (SE374.R11).

Provides:
- Clinical Biomarkers: TSR, Stroma Proportion, VTP.
- CAP (College of American Pathologists) Risk Stratification.
- Dual-Engine Kaplan-Meier Survival Analysis with Log-Rank Testing.
- Grounded TCGA-BRCA 23 Test Patient Survival Manifest.
- Exhaustive 3-Way Benchmark Evaluator across test patches.

Strict Invariant 12 compliance: 100% standard keyboard ASCII hyphens '-' (0x2D) only.
"""

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

__all__ = [
    "ACTIVE_TEST_PATIENTS_23",
    "TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA",
    "KaplanMeierSurvivalAnalysis",
    "compute_patient_biomarkers",
    "compute_stroma_proportion",
    "compute_tumor_stroma_ratio",
    "compute_viable_tumor_percentage",
    "export_survival_summary_json",
    "get_all_test_survival_data",
    "get_patient_survival_data",
    "load_active_test_patients",
    "stratify_cap_risk",
    "stratify_cap_risk_category",
    "ExhaustiveBenchmarkEvaluator",
    "SegmentationMetricAccumulator",
    "build_cli_parser",
    "run_cli",
]
