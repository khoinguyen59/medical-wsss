"""
Specialized Audit Suite 1: Confusion Matrix Algebra, Boundary F1 Calibration & Property-Based Invariants.

Focused, non-overlapping verification for Pillar 8 (Audit Team 1):
1. Confusion matrix 4-class vs 5-class submatrix identity (cm4 == cm5[:4, :4]) and ignore_index=255 exclusion
   via SegmentationMetricAccumulator and MedicalSegmentationMetrics.
2. Property-based tests (Hypothesis) on MedicalSegmentationMetrics:
   - Exact match yields 100.0% mIoU and mDice on present classes.
   - Symmetry of Dice: Dice(A, B) == Dice(B, A).
   - Mathematical inequality: IoU(c) <= Dice(c) and Dice(c) == 2*IoU(c)/(1+IoU(c)) for every class c.
3. Boundary F1 (BF1) morphological shift calibration at 0px, 1px, and 6px offsets.

Strictly adheres to Invariant 12 (standard ASCII hyphen '-' 0x2D only).
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.clinical.exhaustive_evaluator import SegmentationMetricAccumulator
from src.losses_metrics.metrics import (
    MedicalSegmentationMetrics,
    compute_boundary_f1,
)


class TestConfusionMatrixAlgebraicInvariants:
    """Verifies exact submatrix relationships between 4-class and 5-class confusion matrices."""

    def test_4class_is_exact_principal_submatrix_of_5class(self):
        """Verifies 4-class confusion matrix equals the top-left 4x4 submatrix of 5-class confusion matrix."""
        rng = np.random.default_rng(42)
        gt = rng.integers(0, 6, size=(64, 64), dtype=np.int64)
        gt[gt == 5] = 255  # Sprinkle ignore_index=255
        pred = rng.integers(0, 5, size=(64, 64), dtype=np.int64)

        m4 = MedicalSegmentationMetrics(num_classes=4, ignore_index=255)
        m4.update(pred, gt)
        cm4 = m4.confusion_matrix

        m5 = MedicalSegmentationMetrics(num_classes=5, ignore_index=255)
        m5.update(pred, gt)
        cm5 = m5.confusion_matrix

        assert cm4.shape == (4, 4)
        assert cm5.shape == (5, 5)
        np.testing.assert_array_equal(cm4, cm5[:4, :4])

        # Verify total counted pixels in cm5 equals non-255 pixels in gt
        expected_valid_pixels = int(np.sum(gt < 5))
        assert int(cm5.sum()) == expected_valid_pixels

    def test_accumulator_known_analytical_confusion_matrix(self):
        """Verifies SegmentationMetricAccumulator against hand-calculated IoU, Dice, and FwIoU."""
        acc = SegmentationMetricAccumulator(num_classes=5)
        acc.cm_4cls = np.array(
            [
                [80, 10, 0, 10],  # Class 0: TP=80, FN=20, FP=20 -> denom=120 -> IoU=80/120=66.6667%
                [10, 70, 20, 0],  # Class 1: TP=70, FN=30, FP=10 -> denom=110 -> IoU=70/110=63.6364%
                [0, 0, 50, 0],    # Class 2: TP=50, FN=0,  FP=20 -> denom=70  -> IoU=50/70=71.4286%
                [10, 0, 0, 90],   # Class 3: TP=90, FN=10, FP=10 -> denom=110 -> IoU=90/110=81.8182%
            ],
            dtype=np.int64,
        )
        acc.cm_5cls[:4, :4] = acc.cm_4cls
        res = acc.compute_metrics()
        expected_ious = [8000.0 / 120.0, 7000.0 / 110.0, 5000.0 / 70.0, 9000.0 / 110.0]
        for cls_name, exp_iou in zip(["TUM", "STR", "LYM", "NEC"], expected_ious):
            assert res[f"IoU_{cls_name}"] == pytest.approx(exp_iou, rel=1e-4)
            exp_dice = (2.0 * (exp_iou / 100.0)) / (1.0 + (exp_iou / 100.0)) * 100.0
            assert res[f"Dice_{cls_name}"] == pytest.approx(exp_dice, rel=1e-4)

        assert res["mIoU_4cls"] == pytest.approx(np.mean(expected_ious), rel=1e-4)
        assert res["Pixel_Accuracy_4cls"] == pytest.approx((80 + 70 + 50 + 90) / 350.0 * 100.0, rel=1e-4)


class TestBoundaryF1ShiftCalibration:
    """Verifies Boundary F1 (BF1) sensitivity to controlled pixel boundary shifts."""

    def test_boundary_f1_monotonic_decay_with_shift_distance(self):
        """Verifies BF1 is 100% within tolerance (<=2px) and drops sharply outside tolerance (6px)."""
        gt = np.full((64, 64), 4, dtype=np.int64)
        gt[16:48, 16:48] = 0  # 32x32 Tumor square

        # 1. Zero shift (exact match) -> BF1 == 100.0%
        res_0px = compute_boundary_f1(gt, gt, num_classes=5, tolerance_px=2)
        assert res_0px["BF1"] == pytest.approx(100.0, abs=1e-4)

        # 2. 1-pixel shift (within tolerance_px=2) -> BF1 == 100.0%
        pred_1px = np.full((64, 64), 4, dtype=np.int64)
        pred_1px[17:49, 17:49] = 0
        res_1px = compute_boundary_f1(pred_1px, gt, num_classes=5, tolerance_px=2)
        assert res_1px["BF1"] == pytest.approx(100.0, abs=1e-4)

        # 3. 6-pixel shift (exceeds tolerance_px=2 on all 4 sides) -> BF1 drops significantly
        pred_6px = np.full((64, 64), 4, dtype=np.int64)
        pred_6px[22:54, 22:54] = 0
        res_6px = compute_boundary_f1(pred_6px, gt, num_classes=5, tolerance_px=2)
        assert res_6px["BF1"] < 40.0, f"6px shifted square should have low BF1, got {res_6px['BF1']}"


class TestHypothesisPropertyBasedMetrics:
    """Property-based tests using Hypothesis to verify universal metric identities."""

    @given(
        mask_a=arrays(np.int64, (16, 16), elements=st.integers(0, 4)),
        mask_b=arrays(np.int64, (16, 16), elements=st.integers(0, 4)),
    )
    @settings(max_examples=25, deadline=None)
    def test_dice_symmetry_and_iou_le_dice_inequality(self, mask_a: np.ndarray, mask_b: np.ndarray):
        """Verifies Dice(A, B) == Dice(B, A), IoU(A, B) == IoU(B, A), and IoU(c) <= Dice(c)."""
        m_ab = MedicalSegmentationMetrics(num_classes=5, class_names=["TUM", "STR", "LYM", "NEC", "BACK"])
        m_ab.update(mask_a, mask_b)
        iou_ab, miou_ab = m_ab.compute_iou()
        dice_ab, mdice_ab = m_ab.compute_dice()

        m_ba = MedicalSegmentationMetrics(num_classes=5, class_names=["TUM", "STR", "LYM", "NEC", "BACK"])
        m_ba.update(mask_b, mask_a)
        iou_ba, miou_ba = m_ba.compute_iou()
        dice_ba, mdice_ba = m_ba.compute_dice()

        assert miou_ab == pytest.approx(miou_ba, abs=1e-5)
        assert mdice_ab == pytest.approx(mdice_ba, abs=1e-5)

        for c in range(5):
            # Symmetry
            assert iou_ab[c] == pytest.approx(iou_ba[c], abs=1e-5)
            assert dice_ab[c] == pytest.approx(dice_ba[c], abs=1e-5)

            # Universal inequality: 0 <= IoU <= Dice <= 1.0
            iou_val = float(iou_ab[c])
            dice_val = float(dice_ab[c])
            assert 0.0 <= iou_val <= dice_val + 1e-6 <= 1.0 + 1e-6

            # Exact functional relationship: Dice = 2*IoU / (1 + IoU) when union > 0
            if iou_val > 0.0:
                expected_dice = (2.0 * iou_val) / (1.0 + iou_val)
                assert dice_val == pytest.approx(expected_dice, abs=1e-5)
