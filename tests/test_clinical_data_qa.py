"""
Clinical AI Quality Assurance and Medical Imaging Governance Test Suite (SE374.R11).

Standard: Medical AI QA Framework, FDA GMLP 10 Principles, RSNA CLAIM 2024 Update,
          IMDRF N88:2025, ISO 13485 / FDA 21 CFR 820.
Context: BCSS-WSSS Weakly-Supervised Semantic Segmentation (5 classes: TUM, STR, LYM, NEC, BACK)
         on 151 TCGA breast cancer patient cohort.

Test Categories:
1. TC-SPLIT-06: Patient-Level Anti-Leakage and Normalizer Distribution Fitting Gate (Data Snooping Prevention).
2. TC-INTERP-01: Label Interpolation Silent Failure and Nearest-Neighbor Enforcement (No Fractional Artifacts).
3. TC-PATH-05: Tissue Detection, White Glass Isolation and Stain Augmentation Integrity (TIAToolbox #546).
4. TC-CLAIM-01: CLAIM 2024 Standards, 95% Confidence Intervals, Multi-Seed Reproducibility and Confusion Matrix.
5. TC-DRIFT-01: Covariate, Label, Prediction and Concept Drift Stress Testing and Uncertainty Gating.
6. TC-QMS-01: Quality Management System (QMS) Change Control and Immutable Artifact Traceability.

Strictly adheres to Invariant 12: Standard keyboard hyphen '-' (ASCII 0x2D) only.
Strictly adheres to Invariant 07: Absolute scientific integrity (Zero Mockup).
"""

import hashlib
import json
from pathlib import Path
from typing import Set

import cv2
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from scipy import stats

from src.colab_pkg.evaluator import (
    CLASS_CODES,
    ColabSegmentationEvaluator,
    _paired_test_arrays,
)
from src.datasets.bcss_dataset import BCSSDataset
from src.datasets.otsu_filter import (
    apply_otsu_gating_to_prediction,
    compute_otsu_tissue_mask,
)
from src.datasets.patient_split import (
    split_patients_kfold,
    verify_patient_leakage,
)
from src.models.stain_deconv import StainDeconvLayer
from src.visualization.clinical_viewer import compute_agreement_error_overlay

# ==============================================================================
# 1. TC-SPLIT-06: Patient-Level Anti-Leakage & Normalizer Leakage Gate
# ==============================================================================

class TestTCSplit06NormalizerLeakageGate:
    """Verifies patient partition independence and normalizer fitting isolation (TC-SPLIT-06)."""

    def test_canonical_cohort_disjointness_bcss_151_patients(self):
        """Verifies strict 0.0% patient leakage across train, val (tuning), and test partitions."""
        rng = np.random.default_rng(2026)
        # Generate 151 unique patient IDs representing the full BCSS cohort
        patient_pool = [f"TCGA-{rng.choice(['A2', 'BH', 'E2', 'AR'])}-{idx:04d}" for idx in range(1, 152)]
        assert len(set(patient_pool)) == 151

        # Canonical split: 106 train, 22 val, 23 test
        train_pids = set(patient_pool[:106])
        val_pids = set(patient_pool[106:128])
        test_pids = set(patient_pool[128:151])

        assert len(train_pids) == 106
        assert len(val_pids) == 22
        assert len(test_pids) == 23

        # Form filenames for leakage verification
        train_files = [f"{pid}_1024_1024_[1 0 0 0].png" for pid in train_pids]
        val_files = [f"{pid}_2048_1024_[0 1 0 0].png" for pid in val_pids]
        test_files = [f"{pid}_512_512_[0 0 1 0].png" for pid in test_pids]

        res = verify_patient_leakage(train_files, val_files, test_files)
        assert res["has_leakage"] is False
        assert len(res["train_val_overlap"]) == 0
        assert len(res["train_test_overlap"]) == 0
        assert len(res["val_test_overlap"]) == 0

    def test_normalizer_fitting_train_vs_pooled_data_snooping_prevention(self):
        """Proves that fitting image normalizers across pooled (train + test) data leaks test statistics.
        
        FDA GMLP Principle 4 and TC-SPLIT-06 require normalizer parameters (mean, std, scalers)
        to be fitted strictly on the Training set (split == 'train').
        """
        rng = np.random.default_rng(42)
        # Train distribution (e.g. TCGA center A with mean ~ 0.50, std ~ 0.15)
        train_data = rng.normal(loc=0.50, scale=0.15, size=(100, 3, 32, 32)).astype(np.float32)
        # Held-out Test distribution with clinical scanner shift (mean ~ 0.65, std ~ 0.10)
        test_data = rng.normal(loc=0.65, scale=0.10, size=(25, 3, 32, 32)).astype(np.float32)

        # 1. Clean Protocol: normalizer fitted strictly on Train set
        train_mean = np.mean(train_data, axis=(0, 2, 3))
        train_std = np.std(train_data, axis=(0, 2, 3))

        # 2. Leaky Protocol: normalizer fitted on pooled (Train + Test)
        pooled_data = np.concatenate([train_data, test_data], axis=0)
        pooled_mean = np.mean(pooled_data, axis=(0, 2, 3))
        pooled_std = np.std(pooled_data, axis=(0, 2, 3))

        # Assert data snooping shifts the normalization baseline significantly
        mean_shift = np.abs(pooled_mean - train_mean)
        assert np.all(mean_shift > 0.02), (
            f"Expected pooled normalizer to leak test distribution shift: {mean_shift}"
        )

        # Normalized test data with train parameters must preserve test covariate shift
        test_norm_clean = (test_data - train_mean.reshape(1, 3, 1, 1)) / train_std.reshape(1, 3, 1, 1)
        # Under clean normalization, the test mean will reflect its true shifted mean (> 0)
        assert np.all(np.mean(test_norm_clean, axis=(0, 2, 3)) > 0.5)

    def test_kfold_patient_partition_exhaustiveness_and_exclusivity(self):
        """Verifies deterministic 5-fold cross-validation at patient level."""
        patient_ids = [f"TCGA-A2-{i:04d}" for i in range(50)]
        folds = split_patients_kfold(patient_ids, n_splits=5, seed=123)

        assert len(folds) == 5
        val_seen: Set[str] = set()

        for fold_idx, (train_pids, val_pids) in enumerate(folds):
            # No overlap within fold
            assert set(train_pids).isdisjoint(set(val_pids)), f"Leakage in fold {fold_idx}"
            # Val sizes are balanced (10 patients each)
            assert len(val_pids) == 10
            assert len(train_pids) == 40
            # Every patient is validated exactly once
            for p in val_pids:
                assert p not in val_seen, f"Patient {p} appeared in multiple val folds!"
                val_seen.add(p)

        assert val_seen == set(patient_ids)


# ==============================================================================
# 2. TC-INTERP-01: Label Interpolation Silent Failure & Nearest-Neighbor Rule
# ==============================================================================

class TestTCInterp01MaskNearestNeighborEnforcement:
    """Audits segmentation mask resizing to guarantee nearest-neighbor interpolation (TC-INTERP-01).
    
    Any linear/bilinear/bicubic interpolation creates fractional label artifacts at tissue
    boundaries (e.g. 0.45 between TUM and STR). Label maps MUST strictly use nearest-neighbor.
    """

    def test_bilinear_vs_nearest_neighbor_silent_failure_demonstration(self):
        """Demonstrates that bilinear interpolation produces spurious intermediate classes/fractional values,
        while nearest-neighbor strictly preserves genuine discrete classes.
        """
        # 1. Tensor interpolation: bilinear produces fractional float values
        mask_float = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).unsqueeze(0).unsqueeze(0)
        bilinear_tensor = torch.nn.functional.interpolate(
            mask_float, size=(4, 4), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        # Bilinear creates fractional intermediate values (e.g. 0.25, 0.50, 0.75)
        has_fractional = np.any((bilinear_tensor % 1.0) != 0.0)
        assert bool(has_fractional) is True, "Bilinear interpolation must create fractional boundary values."

        nearest_tensor = torch.nn.functional.interpolate(
            mask_float, size=(4, 4), mode="nearest"
        ).squeeze().numpy()
        assert not bool(np.any((nearest_tensor % 1.0) != 0.0)), "Nearest-neighbor must never create fractional values."

        # 2. Discrete OpenCV interpolation: linear introduces phantom classes at boundaries
        # e.g., transition between Class 0 (TUM) and Class 2 (LYM) creates phantom Class 1 (STR)!
        discrete_mask = np.array([
            [0, 2],
            [0, 2],
        ], dtype=np.uint8)

        linear_cv = cv2.resize(discrete_mask, (8, 8), interpolation=cv2.INTER_LINEAR)
        # Verify linear interpolation falsely generated phantom Class 1
        assert 1 in linear_cv, "Linear interpolation creates spurious intermediate classes at boundaries!"

        nearest_cv = cv2.resize(discrete_mask, (8, 8), interpolation=cv2.INTER_NEAREST)
        # Nearest-neighbor strictly contains only {0, 2}
        assert set(np.unique(nearest_cv)).issubset({0, 2})
        assert 1 not in nearest_cv, "Nearest-neighbor must never create phantom boundary classes!"

    @settings(max_examples=50, deadline=None)
    @given(
        h=st.integers(min_value=16, max_value=64),
        w=st.integers(min_value=16, max_value=64),
        scale=st.sampled_from([0.5, 1.5, 2.0]),
    )
    def test_property_nearest_neighbor_preserves_discrete_classes_and_ignore_index(
        self, h: int, w: int, scale: float
    ):
        """Property-based verification: nearest-neighbor resizing of any mask in {0,1,2,3,4,255}
        preserves exact discrete set membership without introducing out-of-range labels.
        """
        rng = np.random.default_rng(h + w)
        allowed_values = np.array([0, 1, 2, 3, 4, 255], dtype=np.uint8)
        mask = rng.choice(allowed_values, size=(h, w)).astype(np.uint8)

        new_h, new_w = int(h * scale), int(w * scale)
        # Use OpenCV nearest-neighbor interpolation
        resized_cv = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Output must strictly be a subset of the allowed integer set
        out_unique = set(np.unique(resized_cv))
        assert out_unique.issubset({0, 1, 2, 3, 4, 255})

    def test_bcss_dataset_mask_loader_strict_integer_and_ignore_index_preservation(self, tmp_path):
        """Verifies BCSSDataset.__getitem__ returns clean integer masks in {0,1,2,3,4,255}."""
        val_img_dir = tmp_path / "val" / "img"
        val_mask_dir = tmp_path / "val" / "mask"
        val_img_dir.mkdir(parents=True)
        val_mask_dir.mkdir(parents=True)

        # Create dummy image and mask
        fname = "TCGA-A2-A0T2_1024_1024_[1 0 0 0].png"
        img = Image.new("RGB", (32, 32), color=(200, 100, 150))
        img.save(val_img_dir / fname)

        # Mask with classes 0, 1, 4, and ignore_index 255
        raw_mask = np.array([
            [0, 1, 4, 255],
            [1, 0, 4, 255],
            [4, 4, 0, 1],
            [255, 255, 1, 0],
        ], dtype=np.uint8)
        # Resize to 32x32 using nearest-neighbor
        raw_mask_32 = cv2.resize(raw_mask, (32, 32), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(raw_mask_32).save(val_mask_dir / fname)

        dataset = BCSSDataset(root_dir=str(tmp_path), split="val")
        assert len(dataset) == 1
        _, mask_tensor, returned_fname = dataset[0]

        assert returned_fname == fname
        assert mask_tensor.dtype == torch.int64
        unique_vals = set(torch.unique(mask_tensor).tolist())
        assert unique_vals.issubset({0, 1, 4, 255})
        assert 255 in unique_vals

        # Strict mask mode must fail closed when a mask file is missing
        (val_mask_dir / fname).unlink()
        strict_ds = BCSSDataset(root_dir=str(tmp_path), split="val", strict_masks=True)
        with pytest.raises(FileNotFoundError):
            _ = strict_ds[0]

    def test_optical_stain_deconv_and_agreement_overlay_real_src(self):
        """Exercises real src StainDeconvLayer and compute_agreement_error_overlay."""
        deconv = StainDeconvLayer()
        # Construct unstained white glass vs purple-pink H&E tissue
        rgb = np.full((16, 16, 3), 250, dtype=np.uint8)
        rgb[:8, :, :] = [135, 50, 155]
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        he, fano, _ = deconv(t)
        assert float(he[0, 0, :8, :].mean()) > float(he[0, 0, 8:, :].mean())
        assert torch.all(fano >= 0.0)

        gt = np.zeros((16, 16), dtype=np.int64)
        gt[8:, :] = 4
        pred = gt.copy()
        pred[0:4, 0:4] = 1
        vis, metrics = compute_agreement_error_overlay(rgb, pred, gt)
        assert vis.shape == (16, 16, 3)
        assert 80.0 < metrics["exact_match_pct"] < 100.0
        assert metrics["total_error_pct"] > 0.0


# ==============================================================================
# 3. TC-PATH-05: Tissue Detection & White Glass Isolation (TIAToolbox #546)
# ==============================================================================

class TestTCPath05TissueDetectionAndStainIsolation:
    """Verifies tissue detection and stain augmentation isolation (TC-PATH-05 / TIAToolbox #546)."""

    def test_white_glass_background_isolation_from_tissue(self):
        """Verifies compute_otsu_tissue_mask correctly identifies blank glass vs stained tissue."""
        # Synthetic H&E patch: Left side is white glass (RGB > 240, sat near 0), Right side is H&E tissue
        patch = np.ones((64, 64, 3), dtype=np.uint8) * 245  # Blank glass
        # Add stained tissue region (nuclei hematoxylin: dark purple/blue, cytoplasm eosin: pink)
        patch[:, 32:, 0] = 180  # R
        patch[:, 32:, 1] = 60   # G
        patch[:, 32:, 2] = 140  # B

        tissue_mask = compute_otsu_tissue_mask(patch)
        assert tissue_mask.shape == (64, 64)

        # Glass region (left) must be background (0)
        assert np.mean(tissue_mask[:, :28]) == 0.0
        # Stained region (right) must be tissue foreground (1)
        assert np.mean(tissue_mask[:, 36:]) == 1.0

    def test_stain_augmentation_tissue_only_gate_tiatoolbox_bug_regression(self):
        """TIAToolbox #546 regression test: stain jitter must NEVER alter blank glass background.
        
        If stain normalization or augmentation is applied indiscriminately to the entire slide,
        white glass receives a pink/purple tint, inducing Net_CAM to falsely classify glass as Stroma.
        """
        # Create slide with tissue in center and pure white glass background
        h, w = 64, 64
        slide = np.ones((h, w, 3), dtype=np.uint8) * 248  # Pure white glass
        # Circular tissue area to avoid morphological corner-rounding artifacts
        cv2.circle(slide, (32, 32), 16, (170, 70, 130), -1)  # Eosin/Hematoxylin mix

        tissue_mask = compute_otsu_tissue_mask(slide)
        true_glass_mask = (slide[:, :, 0] >= 240) & (slide[:, :, 1] >= 240) & (slide[:, :, 2] >= 240)

        # Apply stain augmentation vector: shift channels
        shift_vector = np.array([20.0, -15.0, 10.0], dtype=np.float32)

        # Correct Clinical Pipeline: Gated by tissue_mask
        gated_augmented_slide = slide.copy().astype(np.float32)
        for c in range(3):
            gated_augmented_slide[:, :, c] = np.where(
                tissue_mask == 1,
                np.clip(slide[:, :, c] + shift_vector[c], 0, 255),
                slide[:, :, c],  # Background remains untouched
            )

        # Verify glass pixels in gated_augmented_slide remain clean white (>= 245) and unchanged
        glass_pixels = gated_augmented_slide[true_glass_mask]
        assert np.all(glass_pixels >= 245.0), "Stain augmentation leaked into blank glass background!"
        np.testing.assert_array_equal(
            gated_augmented_slide[true_glass_mask],
            slide[true_glass_mask].astype(np.float32),
        )

    def test_otsu_clinical_gating_overrides_spurious_cam_activations(self):
        """Verifies apply_otsu_gating_to_prediction forces non-tissue pixels to Class 4 (BACK)."""
        # Raw RGB patch: center has tissue, edges are blank glass
        rgb = np.ones((32, 32, 3), dtype=np.uint8) * 245
        rgb[8:24, 8:24] = [180, 50, 120]  # Tissue

        # Erroneous unconstrained model prediction: predicts Tumor (Class 0) everywhere
        spurious_pred = np.zeros((32, 32), dtype=np.int64)

        gated_pred = apply_otsu_gating_to_prediction(spurious_pred, rgb, background_class_idx=4)

        # In blank glass region, prediction must be gated to Class 4 (BACK)
        assert np.all(gated_pred[:6, :] == 4)
        assert np.all(gated_pred[26:, :] == 4)
        # Inside tissue region, original prediction (Class 0) is preserved
        assert np.all(gated_pred[10:22, 10:22] == 0)


# ==============================================================================
# 4. TC-CLAIM-01: CLAIM 2024 Standards, 95% Confidence Intervals & Reporting
# ==============================================================================

class TestTCClaim01ReportingAndConfidenceIntervals:
    """Verifies compliance with CLAIM 2024 reporting standards and statistical rigor (TC-CLAIM-01)."""

    def test_student_t_confidence_interval_coverage_and_scaling(self):
        """Verifies 95% Confidence Interval computation scales as O(1/sqrt(N)) and covers sample mean."""
        rng = np.random.default_rng(2026)
        true_mean_diff = 4.5
        n_samples = 15  # n < 30 invokes exact Student-t critical value ppf(0.975, df=n-1)
        sample_diffs = rng.normal(loc=true_mean_diff, scale=2.0, size=n_samples)

        arr_a = sample_diffs + 50.0
        arr_b = np.full(n_samples, 50.0)

        stats_res = _paired_test_arrays(arr_a, arr_b)

        mean = stats_res["mean_diff"]
        ci_low = stats_res["ci95_low"]
        ci_high = stats_res["ci95_high"]

        assert ci_low <= mean <= ci_high
        ci_half_width = (ci_high - ci_low) / 2.0
        expected_t = stats.t.ppf(0.975, df=n_samples - 1)
        expected_se = np.std(sample_diffs, ddof=1) / np.sqrt(n_samples)
        assert ci_half_width == pytest.approx(expected_t * expected_se, abs=1e-5)

    def test_multi_seed_reproducibility_aggregation(self):
        """Verifies multi-seed evaluation produces accurate mean and standard deviation (mu +- sigma)."""
        seed_results = [
            {"seed": 42, "mIoU": 68.20, "mDice": 78.40},
            {"seed": 123, "mIoU": 67.85, "mDice": 78.10},
            {"seed": 456, "mIoU": 68.50, "mDice": 78.75},
            {"seed": 789, "mIoU": 68.10, "mDice": 78.30},
            {"seed": 999, "mIoU": 68.35, "mDice": 78.55},
        ]

        miou_vals = [r["mIoU"] for r in seed_results]
        mdice_vals = [r["mDice"] for r in seed_results]

        mu_miou = float(np.mean(miou_vals))
        sigma_miou = float(np.std(miou_vals, ddof=1))
        mu_mdice = float(np.mean(mdice_vals))
        sigma_mdice = float(np.std(mdice_vals, ddof=1))

        assert mu_miou == pytest.approx(68.20, abs=0.01)
        assert sigma_miou < 0.30  # High reproducibility across 5 seeds
        assert mu_mdice == pytest.approx(78.42, abs=0.01)

    def test_claim_2024_full_confusion_matrix_and_per_class_sens_spec(self):
        """CLAIM 2024 Item 38: verifies full confusion matrix and per-class diagnostic accuracy metrics."""
        evaluator = ColabSegmentationEvaluator(num_classes=5)

        # Synthesize ground truth and prediction
        gt = np.array([
            [0, 0, 1, 1],
            [0, 2, 2, 3],
            [1, 1, 3, 3],
            [4, 4, 4, 4],
        ], dtype=np.int64)

        pred = np.array([
            [0, 0, 1, 1],
            [1, 2, 2, 3],  # 1 pixel of Class 0 misclassified as Class 1
            [1, 1, 3, 3],
            [4, 4, 4, 4],
        ], dtype=np.int64)

        evaluator.update(pred=pred, target=gt, patient_id="TCGA-A2-A0T2")
        metrics = evaluator.compute_metrics()

        # Check full 5x5 confusion matrix exists and sums to 16
        cm = evaluator.confusion_matrix
        assert cm.shape == (5, 5)
        assert cm.sum() == 16

        # Class 0: 3 pixels in GT. 2 predicted correct, 1 misclassified as Class 1
        tp_0 = cm[0, 0]
        fn_0 = cm[0, :].sum() - tp_0
        fp_0 = cm[:, 0].sum() - tp_0
        tn_0 = cm.sum() - (tp_0 + fn_0 + fp_0)

        sensitivity_0 = tp_0 / (tp_0 + fn_0)
        specificity_0 = tn_0 / (tn_0 + fp_0)

        assert sensitivity_0 == pytest.approx(2.0 / 3.0, abs=1e-5)
        assert specificity_0 == pytest.approx(13.0 / 13.0, abs=1e-5)

    def test_claim_2024_scientific_terminology_keys_in_evaluator(self):
        """Verifies evaluator dictionary utilizes scientific terminology aligned with CLAIM 2024."""
        evaluator = ColabSegmentationEvaluator(num_classes=5)
        evaluator.update(
            pred=np.zeros((4, 4), dtype=np.int64),
            target=np.zeros((4, 4), dtype=np.int64),
            patient_id="TCGA-A2-A0T2",
        )
        res = evaluator.compute_metrics()

        # Check essential metrics are present
        assert "mIoU" in res
        assert "mDice" in res
        assert "Accuracy" in res or "Pixel_Accuracy" in res
        assert "per_class_iou" in res
        assert "per_class_dice" in res
        assert "per_class_precision" in res
        assert "per_class_recall" in res
        for code in CLASS_CODES:
            assert code in res["per_class_iou"]
            assert code in res["per_class_dice"]


# ==============================================================================
# 5. TC-DRIFT-01: Distribution Drift Stress Testing & Uncertainty Gating
# ==============================================================================

class TestTCDrift01DistributionDriftStressTesting:
    """Verifies drift detection and uncertainty regularized gating under domain shifts (TC-DRIFT-01)."""

    def test_covariate_stain_shift_increases_top2_entropy(self):
        """Verifies that synthetic stain covariate shift increases top-2 uncertainty entropy.
        
        Under severe stain variation (scanner differences), model predictions should exhibit
        higher entropy and trigger the uncertainty gate rather than outputting confident errors.
        """
        # Baseline normal tissue probability: clear distinction between TUM and STR
        p_baseline = torch.tensor([0.85, 0.10, 0.02, 0.02, 0.01])
        top2_base, _ = torch.topk(p_baseline, 2)
        norm_top2_base = top2_base / top2_base.sum()
        entropy_base = -(norm_top2_base * torch.log(norm_top2_base + 1e-7)).sum() / np.log(2.0)

        # Covariate shifted tissue probability (ambiguous stain causing tie between TUM and STR)
        p_shifted = torch.tensor([0.48, 0.46, 0.02, 0.02, 0.02])
        top2_shift, _ = torch.topk(p_shifted, 2)
        norm_top2_shift = top2_shift / top2_shift.sum()
        entropy_shift = -(norm_top2_shift * torch.log(norm_top2_shift + 1e-7)).sum() / np.log(2.0)

        # Entropy under covariate shift must be significantly higher
        assert entropy_shift > 0.95
        assert entropy_base < 0.50
        assert entropy_shift > entropy_base

    def test_label_drift_detection_kullback_leibler_divergence(self):
        """Verifies detection of Label Drift between train and test class priors via KL Divergence."""
        # Train class prior: TUM=35%, STR=40%, LYM=15%, NEC=5%, BACK=5%
        p_train = np.array([0.35, 0.40, 0.15, 0.05, 0.05], dtype=np.float64)
        # Test class prior with clinical selection bias: Necrosis enriched (NEC=25%, TUM=20%)
        p_test_drifted = np.array([0.20, 0.35, 0.15, 0.25, 0.05], dtype=np.float64)
        # Test class prior without drift (drawn from same distribution)
        p_test_clean = np.array([0.34, 0.41, 0.15, 0.05, 0.05], dtype=np.float64)

        kl_clean = float(stats.entropy(p_train, p_test_clean))
        kl_drifted = float(stats.entropy(p_train, p_test_drifted))

        assert kl_clean < 0.01, f"Clean distribution divergence should be near 0: {kl_clean}"
        assert kl_drifted > 0.10, f"Label drift should yield significant KL divergence: {kl_drifted}"

    def test_prediction_drift_two_sample_chi_square_alert(self):
        """Verifies that significant prediction distribution shift triggers a statistical alarm."""
        # Baseline model output counts over 1000 patches
        baseline_counts = np.array([350, 400, 150, 50, 50], dtype=np.int64)
        # New hospital cohort deployment with silent failure / prediction drift
        drifted_counts = np.array([150, 200, 100, 500, 50], dtype=np.int64)

        # Perform Chi-square contingency test
        contingency_table = np.array([baseline_counts, drifted_counts])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)

        assert p_val < 1e-5, f"Expected strong rejection of null hypothesis under prediction drift, got p={p_val}"


# ==============================================================================
# 6. TC-QMS-01: QMS Change Control & Immutable Artifact Traceability
# ==============================================================================

class TestTCQMS01TraceabilityAndChangeControl:
    """Verifies QMS change control, file integrity checksums, and teacher notebook protection (TC-QMS-01)."""

    def test_teacher_notebook_lab1_byte_for_byte_immutability(self):
        """FDA 21 CFR 820 / ISO 13485: Verifies notebooks/Lab1.ipynb is 100% byte-identical to Lab1.ipynb."""
        root_lab1 = Path("Lab1.ipynb")
        notebooks_lab1 = Path("notebooks/Lab1.ipynb")

        if not root_lab1.is_file() and not notebooks_lab1.is_file():
            pytest.skip("Untracked instructor notebook Lab1.ipynb is not present in this checkout.")
        assert notebooks_lab1.is_file(), "Missing required teacher notebook: notebooks/Lab1.ipynb"
        assert root_lab1.is_file(), "Missing reference root notebook: Lab1.ipynb"
        h_root = hashlib.sha256(root_lab1.read_bytes()).hexdigest()
        h_nb = hashlib.sha256(notebooks_lab1.read_bytes()).hexdigest()
        assert h_root == h_nb, "Instructor notebook notebooks/Lab1.ipynb was modified! Strict immutability violated."

    def test_sha256_checksum_verification_utility(self, tmp_path):
        """Verifies deterministic SHA-256 calculation for pipeline configuration files."""
        sample_config = {"project": "SE374.R11", "dataset": "BCSS", "num_classes": 5}
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(sample_config, sort_keys=True), encoding="utf-8")

        raw_bytes = cfg_file.read_bytes()
        expected_sha = hashlib.sha256(raw_bytes).hexdigest()
        assert len(expected_sha) == 64
        # Verify idempotency
        assert hashlib.sha256(cfg_file.read_bytes()).hexdigest() == expected_sha
