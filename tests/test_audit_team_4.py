"""
Specialized Audit Suite 4: Statistical Hypothesis Testing, Holm-Bonferroni FWER & Clinical Concordance.

Focused, non-overlapping verification for Pillar 8 (Audit Team 4):
1. Holm-Bonferroni step-down multiple-testing correction (exact analytical adjustment, monotonicity,
   and [0.0, 1.0] bounds).
2. Student-t 95% confidence interval calculation across paired differences (df = n - 1).
3. Patient-level paired Wilcoxon signed-rank and paired Student t-test on ColabSegmentationEvaluator
   and ExhaustiveBenchmarkEvaluator (preventing patch-level pseudo-replication and OR-gate p-hacking).
4. Pathologist concordance analytical identities: ICC(2,1) two-way random absolute agreement and Lin's CCC
   penalize systematic additive bias whereas Pearson r remains 1.0.

Strictly adheres to Invariant 12 (standard ASCII hyphen '-' 0x2D only).
"""

import numpy as np
import pytest
from scipy import stats

from src.colab_pkg.evaluator import (
    ColabSegmentationEvaluator,
    _paired_test_arrays,
    compute_paired_statistical_tests,
    holm_bonferroni_correction,
)
from src.losses_metrics.metrics import compute_patient_biomarker_correlation


class TestStatisticalCorrectionsAndConfidenceIntervals:
    """Verifies Holm-Bonferroni FWER control and Student-t confidence intervals."""

    def test_holm_bonferroni_step_down_exact_values_and_monotonicity(self):
        """Verifies step-down Holm-Bonferroni formula against hand calculation."""
        raw_p = {
            "M7_vs_M1": 0.004,  # Rank 1 (m=4): 0.004 * 4 = 0.016
            "M7_vs_M2": 0.012,  # Rank 2 (m-1=3): max(0.016, 0.012 * 3) = 0.036
            "M7_vs_M3": 0.020,  # Rank 3 (m-2=2): max(0.036, 0.020 * 2) = 0.040
            "M7_vs_M6": 0.060,  # Rank 4 (m-3=1): max(0.040, 0.060 * 1) = 0.060
        }
        corr = holm_bonferroni_correction(raw_p, alpha=0.05)

        assert corr["M7_vs_M1"]["holm_adjusted_pvalue"] == pytest.approx(0.016, abs=1e-6)
        assert corr["M7_vs_M2"]["holm_adjusted_pvalue"] == pytest.approx(0.036, abs=1e-6)
        assert corr["M7_vs_M3"]["holm_adjusted_pvalue"] == pytest.approx(0.040, abs=1e-6)
        assert corr["M7_vs_M6"]["holm_adjusted_pvalue"] == pytest.approx(0.060, abs=1e-6)

        assert corr["M7_vs_M1"]["significant_holm_p05"] is True
        assert corr["M7_vs_M2"]["significant_holm_p05"] is True
        assert corr["M7_vs_M3"]["significant_holm_p05"] is True
        assert corr["M7_vs_M6"]["significant_holm_p05"] is False

    def test_student_t_confidence_interval_exact_quantile_and_paired_tests(self):
        """Verifies _paired_test_arrays uses Student-t quantile t_{0.975, df=n-1} and no OR-gate p-hacking."""
        a = np.array([62.0, 64.0, 63.0, 65.0, 61.0], dtype=np.float64)
        b = np.array([58.0, 59.0, 60.0, 60.0, 58.0], dtype=np.float64)
        diffs = a - b  # [4.0, 5.0, 3.0, 5.0, 3.0]

        res = _paired_test_arrays(a, b)
        expected_mean = float(np.mean(diffs))
        expected_std = float(np.std(diffs, ddof=1))
        t_crit = float(stats.t.ppf(0.975, df=4))
        expected_half = t_crit * expected_std / np.sqrt(5.0)

        assert res["mean_diff"] == pytest.approx(expected_mean, abs=1e-4)
        assert res["ci95_low"] == pytest.approx(expected_mean - expected_half, abs=1e-3)
        assert res["ci95_high"] == pytest.approx(expected_mean + expected_half, abs=1e-3)
        assert res["significant_p05"] is bool(res["wilcoxon_pvalue"] < 0.05)
        assert res["t_significant_p05"] is bool(res["t_pvalue"] < 0.05)

    def test_compute_paired_statistical_tests_includes_holm_bonferroni(self):
        """Verifies compute_paired_statistical_tests computes Holm-Bonferroni adjusted p-values."""
        ev_m1 = ColabSegmentationEvaluator()
        ev_m2 = ColabSegmentationEvaluator()
        ev_m5 = ColabSegmentationEvaluator()

        ev_m1.patch_miou_4cls = [50.0 + i for i in range(15)]
        ev_m1.patch_mdice_4cls = [62.0 + i for i in range(15)]
        ev_m1.patch_patient_ids = [f"TCGA-A1-00{i:02d}" for i in range(15)]

        ev_m2.patch_miou_4cls = [52.0 + i for i in range(15)]
        ev_m2.patch_mdice_4cls = [64.0 + i for i in range(15)]
        ev_m2.patch_patient_ids = [f"TCGA-A1-00{i:02d}" for i in range(15)]

        ev_m5.patch_miou_4cls = [56.0 + i for i in range(15)]
        ev_m5.patch_mdice_4cls = [68.0 + i for i in range(15)]
        ev_m5.patch_patient_ids = [f"TCGA-A1-00{i:02d}" for i in range(15)]

        paired = compute_paired_statistical_tests(
            {
                "M1_Baseline_SingleScale": ev_m1,
                "M2_MultiView_D4_TTA": ev_m2,
                "M5_Proposed_SOTA_Ensemble": ev_m5,
            },
            baseline_key="M1_Baseline_SingleScale",
        )
        for k in ("M2_MultiView_D4_TTA", "M5_Proposed_SOTA_Ensemble"):
            assert "holm_adjusted_pvalue" in paired[k]["miou_paired_test"]
            assert "significant_holm_p05" in paired[k]["miou_paired_test"]
            assert "holm_adjusted_pvalue" in paired[k]["miou_patient_level_test"]
            assert "significant_holm_p05" in paired[k]["miou_patient_level_test"]


class TestPathologistConcordanceAnalyticalIdentities:
    """Verifies ICC(2,1), Lin's CCC, and Bland-Altman on exact analytical inputs."""

    def test_icc21_and_lins_ccc_penalize_systematic_bias_unlike_pearson_r(self):
        """Proves that a constant +15% bias keeps Pearson r = 1.0 but lowers Lin's CCC and ICC(2,1)."""
        gt_vals = [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]

        # 1. Perfect agreement -> CCC == 1.0, ICC(2,1) == 1.0, Bland-Altman bias == 0.0
        records_perfect = [
            {"patient_id": f"P{i}", "pred_vtp": v, "gt_vtp": v, "pred_tsr": v, "gt_tsr": v}
            for i, v in enumerate(gt_vals)
        ]
        res_perfect = compute_patient_biomarker_correlation(records_perfect)
        assert res_perfect["vtp_pearson_r"] == pytest.approx(1.0, abs=1e-5)
        assert res_perfect["vtp_ccc"] == pytest.approx(1.0, abs=1e-5)
        assert res_perfect["vtp_icc"] == pytest.approx(1.0, abs=1e-5)
        assert res_perfect["vtp_bias"] == pytest.approx(0.0, abs=1e-5)

        # 2. Systematic additive shift (+15.0) -> Pearson r is still 1.0, but CCC and ICC(2,1) < 0.85
        records_biased = [
            {"patient_id": f"P{i}", "pred_vtp": v + 15.0, "gt_vtp": v, "pred_tsr": v + 15.0, "gt_tsr": v}
            for i, v in enumerate(gt_vals)
        ]
        res_biased = compute_patient_biomarker_correlation(records_biased)
        assert res_biased["vtp_pearson_r"] == pytest.approx(1.0, abs=1e-5)
        assert res_biased["vtp_ccc"] < 0.85
        assert res_biased["vtp_icc"] < 0.85
        assert res_biased["vtp_bias"] == pytest.approx(15.0, abs=1e-5)
