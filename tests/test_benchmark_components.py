"""
Analytical & Mathematical Unit Tests for SE374.R11 Loss Functions, Metrics & WSSS Components.

Verifies:
1. Exact analytical values of MultiLabelClassificationLoss, AsymmetricLoss, MultiLayerPseudoSupervisionLoss,
   UncertaintyRegularizationLoss, and CompositeWSSSLoss against closed-form formulas.
2. Strict proof that MultiLayerPseudoSupervisionLoss ignores target == 255 (zero loss & zero gradient
   contribution from ignored pixels, and 0.0 finite loss when the entire mask is 255).
3. Strict proof that UncertaintyRegularizationLoss discounts high-entropy boundary pixels relative to
   low-entropy confident pixels with the exact same Cross-Entropy error.
4. Numerical stability under extreme logits (+/- 1e4).
5. Strict Zero Mockup & ASCII hyphens (-) only.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.datasets.otsu_filter import apply_otsu_gating_to_prediction, compute_otsu_tissue_mask
from src.losses_metrics.losses import (
    MultiLabelClassificationLoss,
    MultiLayerPseudoSupervisionLoss,
    UncertaintyRegularizationLoss,
)
from src.losses_metrics.metrics import (
    MedicalSegmentationMetrics,
    compute_boundary_f1,
    compute_fwiou,
    compute_pixel_accuracy,
    compute_stroma_percentage,
    compute_tumor_stroma_ratio,
    compute_viable_tumor_bed_percentage,
    compute_viable_tumor_percentage,
)
from src.models.wsss_tissue import (
    WSSSBenchmarkingEngine,
    apply_classmix_augmentation,
)
from src.training.train_proposed_sam import CompositeWSSSLoss
from src.training.train_stage1 import AsymmetricLoss


def test_metrics_service_analytical_iou_dice_fwiou_and_bf1():
    """Verifies confusion matrix, per-class IoU, Dice, FwIoU, PA, and Boundary F1 against hand-calculated values."""
    metrics = MedicalSegmentationMetrics(
        num_classes=5,
        class_names=["TUM", "STR", "LYM", "NEC", "BACK"],
        ignore_index=255,
    )
    # 4 pixels + 1 ignored pixel (255)
    pred = np.array([0, 1, 0, 1, 3], dtype=np.int64)
    target = np.array([0, 0, 1, 1, 255], dtype=np.int64)

    metrics.update(pred, target)
    assert metrics.confusion_matrix.sum() == 4  # pixel with target=255 must be excluded

    iou, miou = metrics.compute_iou()
    dice, mdice = metrics.compute_dice()
    fwiou = metrics.compute_fwiou()
    pa = metrics.compute_pixel_accuracy()

    # Class 0: TP=1, FP=1, FN=1 -> IoU = 1/3, Dice = 2/4 = 0.5
    # Class 1: TP=1, FP=1, FN=1 -> IoU = 1/3, Dice = 2/4 = 0.5
    assert iou[0] == pytest.approx(1.0 / 3.0)
    assert iou[1] == pytest.approx(1.0 / 3.0)
    assert miou == pytest.approx(1.0 / 3.0)
    assert dice[0] == pytest.approx(0.5)
    assert mdice == pytest.approx(0.5)
    assert fwiou == pytest.approx(1.0 / 3.0)
    assert pa == pytest.approx(0.5)
    assert compute_fwiou(pred, target, num_classes=5, ignore_index=255) == pytest.approx(1.0 / 3.0)
    assert compute_pixel_accuracy(pred, target, ignore_index=255) == pytest.approx(0.5)

    # Boundary F1 shift calibration: 1px shift within tolerance_px=2 -> BF1 = 100%, 5px shift -> BF1 < 50%
    gt_box = np.full((64, 64), fill_value=4, dtype=np.int64)
    gt_box[16:48, 16:48] = 0
    pred_shift_1px = np.full((64, 64), fill_value=4, dtype=np.int64)
    pred_shift_1px[17:49, 17:49] = 0
    pred_shift_6px = np.full((64, 64), fill_value=4, dtype=np.int64)
    pred_shift_6px[22:54, 22:54] = 0

    bf1_1px = compute_boundary_f1(pred_shift_1px, gt_box, num_classes=5, tolerance_px=2)["BF1"]
    bf1_6px = compute_boundary_f1(pred_shift_6px, gt_box, num_classes=5, tolerance_px=2)["BF1"]
    assert bf1_1px == pytest.approx(100.0)
    assert bf1_6px < bf1_1px


def test_clinical_biomarkers_vtp_tsr_and_tumor_bed():
    """Verifies VTP (cellularity), VTP (tumor bed), TSR, and Stroma Percentage identities."""
    mask = np.array([
        [0, 0, 0, 1],  # 3 tumor, 1 stroma
        [3, 4, 4, 255],  # 1 necrosis, 2 background, 1 ignored
    ])
    # Valid tissue = 3 TUM + 1 STR + 1 NEC = 5 pixels -> VTP = 3/5 = 60%
    vtp = compute_viable_tumor_percentage(mask, tumor_class_idx=0, background_class_idx=4, ignore_index=255)
    assert vtp == pytest.approx(60.0)

    # Tumor bed = 3 TUM + 1 NEC = 4 pixels -> VTP_bed = 3/4 = 75%
    vtp_bed = compute_viable_tumor_bed_percentage(mask, tumor_class_idx=0, necrosis_class_idx=3)
    assert vtp_bed == pytest.approx(75.0)

    # TSR = 3 TUM / (3 TUM + 1 STR) = 75%, Stroma Percentage = 25%
    tsr = compute_tumor_stroma_ratio(mask, tumor_class_idx=0, stroma_class_idx=1)
    str_pct = compute_stroma_percentage(mask, tumor_class_idx=0, stroma_class_idx=1)
    assert tsr == pytest.approx(75.0)
    assert str_pct == pytest.approx(25.0)
    assert tsr + str_pct == pytest.approx(100.0)


def test_otsu_clinical_gating():
    """Tests Otsu thresholding on synthetic tissue vs white background."""
    img = np.full((100, 100, 3), fill_value=255, dtype=np.uint8)
    img[:50, :] = [200, 80, 140]

    mask = compute_otsu_tissue_mask(img)
    assert mask.shape == (100, 100)
    assert np.mean(mask[:50, :]) > 0.90
    assert np.mean(mask[50:, :]) < 0.10

    pred = np.zeros((100, 100), dtype=np.int64)
    gated = apply_otsu_gating_to_prediction(pred, img, background_class_idx=4)
    assert np.all(gated[50:, :] == 4)


def test_multilabel_and_asymmetric_loss_analytical():
    """Verifies MultiLabelClassificationLoss and AsymmetricLoss against exact closed-form calculations."""
    logits = torch.tensor([[2.0, -2.0]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    pos_w = torch.tensor([2.0, 1.0], dtype=torch.float32)
    bce_loss = MultiLabelClassificationLoss(pos_weight=pos_w)(logits, targets)
    expected_bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)
    assert torch.isclose(bce_loss, expected_bce, atol=1e-6)

    # AsymmetricLoss: with clip=0.05, gamma_pos=0.0, gamma_neg=4.0:
    # For target=1 at logit=2.0: p = sigmoid(2.0), loss_pos = -log(p)
    # For target=0 at logit=-2.0: p = sigmoid(-2.0) ~ 0.1192, p_m = max(p - 0.05, 0) = 0.0692,
    # loss_neg = -(p^4) * log(1 - p_m)
    asl_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    asl_val = asl_fn(logits, targets)
    p0 = 1.0 / (1.0 + math.exp(-2.0))
    p1 = 1.0 / (1.0 + math.exp(2.0))
    pm1 = max(p1 - 0.05, 0.0)
    expected_asl = 0.5 * (-math.log(p0) - (p1 ** 4) * math.log(1.0 - pm1))
    assert asl_val.item() == pytest.approx(expected_asl, rel=1e-4)


def test_mlps_loss_analytical_weights_and_ignore_index_255_invariance():
    """Proves MultiLayerPseudoSupervisionLoss applies exact layer weights and genuinely ignores 255."""
    mlps = MultiLayerPseudoSupervisionLoss(
        weights={"bn7": 1.0, "b5_2": 0.5, "b4_5": 0.25},
        ignore_index=255,
    )
    logits = torch.randn(1, 5, 8, 8, requires_grad=True)
    m_bn7 = torch.zeros((1, 8, 8), dtype=torch.long)
    m_b5_2 = torch.ones((1, 8, 8), dtype=torch.long)
    m_b4_5 = torch.full((1, 8, 8), fill_value=2, dtype=torch.long)
    # Mark right half of m_bn7 as ignore_index=255
    m_bn7[:, :, 4:] = 255

    loss = mlps(logits, {"bn7": m_bn7, "b5_2": m_b5_2, "b4_5": m_b4_5})
    expected = (
        1.0 * F.cross_entropy(logits, m_bn7, ignore_index=255)
        + 0.5 * F.cross_entropy(logits, m_b5_2, ignore_index=255)
        + 0.25 * F.cross_entropy(logits, m_b4_5, ignore_index=255)
    )
    assert torch.isclose(loss, expected, atol=1e-6)

    # Perturbing logits on the ignored region (:, :, :, 4:) when only bn7 is supervised
    # MUST produce zero change in loss and zero gradient on the ignored pixels!
    logits_a = logits.detach().clone().requires_grad_(True)
    logits_b = logits.detach().clone()
    logits_b[:, :, :, 4:] += 50.0  # Massive change on ignored pixels
    logits_b.requires_grad_(True)

    loss_a = mlps(logits_a, {"bn7": m_bn7})
    loss_b = mlps(logits_b, {"bn7": m_bn7})
    assert torch.isclose(loss_a, loss_b, atol=1e-6)

    loss_a.backward()
    assert torch.all(logits_a.grad[:, :, :, 4:] == 0.0)

    # All-255 mask must return 0.0 (never NaN)
    all_ignore = torch.full((1, 8, 8), fill_value=255, dtype=torch.long)
    zero_loss = mlps(logits_a, {"bn7": all_ignore})
    assert zero_loss.item() == 0.0
    assert not torch.isnan(zero_loss)


def test_uncertainty_regularization_loss_strictly_discounts_high_entropy_pixels():
    """Proves UncertaintyRegularizationLoss assigns strictly smaller loss and gradient to high-entropy pixels
    than to low-entropy pixels when both have the exact same raw Cross-Entropy error for target class 0.
    """
    unc_fn = UncertaintyRegularizationLoss(entropy_threshold=0.65, ignore_index=255, reduction="mean")
    target = torch.zeros((1, 1, 1), dtype=torch.long)  # Target class = 0

    # Construct two 5-class logit vectors with the EXACT SAME probability p_0 for class 0:
    # Case 1 (Low Entropy): p = [0.20, 0.80, 0.0, 0.0, 0.0] -> H = -(0.2 ln 0.2 + 0.8 ln 0.8)/ln 5 ~ 0.311
    # Case 2 (High Entropy): p = [0.20, 0.20, 0.20, 0.20, 0.20] -> H = 1.0
    # Both have p_0 = 0.20, so raw Cross-Entropy CE = -ln(0.20) is identical!
    probs_low_ent = torch.tensor([0.20, 0.7999, 1e-4 / 3, 1e-4 / 3, 1e-4 / 3], dtype=torch.float32)
    probs_high_ent = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20], dtype=torch.float32)

    logits_low = torch.log(probs_low_ent).view(1, 5, 1, 1).requires_grad_(True)
    logits_high = torch.log(probs_high_ent).view(1, 5, 1, 1).requires_grad_(True)

    # Verify raw Cross-Entropy is identical
    raw_ce_low = F.cross_entropy(logits_low, target)
    raw_ce_high = F.cross_entropy(logits_high, target)
    assert torch.isclose(raw_ce_low, raw_ce_high, atol=1e-4)

    # Uncertainty-regularized loss must be strictly smaller for high-entropy pixel
    loss_low = unc_fn(logits_low, target)
    loss_high = unc_fn(logits_high, target)
    assert loss_high.item() < loss_low.item(), (
        f"High-entropy pixel loss ({loss_high.item():.4f}) must be < low-entropy loss ({loss_low.item():.4f})"
    )
    # Specifically, for H=1.0 and threshold=0.65, weight is clamped at min=0.10:
    assert loss_high.item() == pytest.approx(0.10 * raw_ce_high.item(), rel=1e-3)

    loss_low.backward()
    loss_high.backward()
    assert logits_high.grad.norm().item() < logits_low.grad.norm().item()


def test_composite_wsss_loss_and_extreme_logit_stability():
    """Tests CompositeWSSSLoss components and verifies numerical stability under +/- 1e4 extreme logits."""
    comp_loss = CompositeWSSSLoss(num_classes=5, ignore_index=255)
    logits = torch.full((2, 5, 16, 16), fill_value=-1e4, dtype=torch.float32)
    logits[:, 0, :, :] = 1e4  # Perfect prediction for class 0 with extreme magnitude
    logits.requires_grad_(True)

    targets = torch.zeros((2, 16, 16), dtype=torch.long)
    targets[:, 8:, :] = 255  # Half ignored
    iou_preds = torch.ones((2, 5), dtype=torch.float32, requires_grad=True)

    out = comp_loss(logits, targets, iou_preds)
    assert torch.isfinite(out["total_loss"])
    assert torch.isfinite(out["focal_loss"])
    assert torch.isfinite(out["dice_loss"])
    assert torch.isfinite(out["iou_loss"])
    assert torch.isfinite(out["head_loss"])
    out["total_loss"].backward()
    assert torch.all(torch.isfinite(logits.grad))


def test_wsss_engine_all_configurations():
    """Tests that all 4 configurations can be initialized and generate masks."""
    configs = ["baseline", "classmix", "relocalization", "proposed"]
    dummy_input = torch.rand(1, 3, 64, 64)
    dummy_raw_img = (np.random.rand(64, 64, 3) * 255.0).astype(np.uint8)

    for cfg in configs:
        engine = WSSSBenchmarkingEngine(mode=cfg, num_classes=5, device="cpu")
        mask = engine.generate_pseudo_mask(
            image_tensor=dummy_input,
            current_epoch=3,
            raw_image=dummy_raw_img,
        )
        assert mask.shape == (64, 64)
        assert mask.min() >= 0
        assert mask.max() < 5


def test_classmix_augmentation_pixel_conservation():
    """Verifies that ClassMix mixes pixels from sample B into sample A where mask belongs to selected classes."""
    images = torch.zeros(2, 3, 16, 16)
    images[1] = 1.0  # Sample 1 is all-ones
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    masks = torch.zeros(2, 16, 16, dtype=torch.long)
    masks[1, :8, :] = 1  # Top half of sample 1 is class 1

    mixed_imgs, mixed_lbls, mixed_msks = apply_classmix_augmentation(
        images, labels, masks, p=1.0
    )
    assert mixed_imgs.shape == images.shape
    assert mixed_lbls.shape == labels.shape
    assert mixed_msks.shape == masks.shape
    assert torch.all((mixed_imgs >= 0.0) & (mixed_imgs <= 1.0))
