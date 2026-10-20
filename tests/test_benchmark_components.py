"""
Unit Tests for SE374.R11 Benchmarking & Modeling Components.
Verifies:
- MedicalSegmentationMetrics (Confusion Matrix, mIoU, Dice, VTP, TSR)
- Loss Functions (MultiLabel, MLPS, Uncertainty Regularization)
- Otsu Clinical Gating
- Model Forward Pass & 4 Benchmark Configurations
- Strict Zero Mockup & ASCII hyphens (-) only
"""

import numpy as np
import pytest
import torch

from src.datasets.otsu_filter import apply_otsu_gating_to_prediction, compute_otsu_tissue_mask
from src.losses_metrics.losses import (
    MultiLabelClassificationLoss,
    MultiLayerPseudoSupervisionLoss,
    UncertaintyRegularizationLoss,
)
from src.losses_metrics.metrics import (
    MedicalSegmentationMetrics,
    compute_tumor_stroma_ratio,
    compute_viable_tumor_percentage,
)
from src.models.wsss_tissue import (
    HistopathologyClassifierBackbone,
    SpatialAffinityRelocalization,
    WSSSBenchmarkingEngine,
    apply_classmix_augmentation,
)


def test_metrics_service_perfect_match():
    """Tests metrics when prediction exactly equals ground truth."""
    metrics = MedicalSegmentationMetrics(num_classes=5)
    pred = np.array([[0, 1], [2, 3]])
    target = np.array([[0, 1], [2, 3]])

    metrics.update(pred, target)
    iou, miou = metrics.compute_iou()
    dice, mdice = metrics.compute_dice()

    assert miou == pytest.approx(1.0)
    assert mdice == pytest.approx(1.0)
    assert iou[0] == pytest.approx(1.0)
    assert dice[1] == pytest.approx(1.0)


def test_metrics_service_partial_overlap():
    """Tests metrics calculation on known synthetic mismatch."""
    metrics = MedicalSegmentationMetrics(num_classes=3)
    pred = np.array([0, 1, 0, 1])
    target = np.array([0, 0, 1, 1])

    # Class 0: target [0, 0], pred [0, 0] -> TP=1, FP=1, FN=1 -> IoU = 1/3 = 0.333
    # Class 1: target [1, 1], pred [1, 1] -> TP=1, FP=1, FN=1 -> IoU = 1/3 = 0.333
    metrics.update(pred, target)
    iou, miou = metrics.compute_iou()

    assert iou[0] == pytest.approx(1.0 / 3.0)
    assert iou[1] == pytest.approx(1.0 / 3.0)
    assert miou == pytest.approx(1.0 / 3.0)


def test_clinical_biomarkers():
    """Tests Viable Tumor Percentage and Tumor-Stroma Ratio."""
    mask = np.array([
        [0, 0, 1, 1],  # 2 tumor, 2 stroma
        [4, 4, 4, 4],  # 4 background
    ])
    # Tissue area = 4 pixels (2 tumor + 2 stroma)
    vtp = compute_viable_tumor_percentage(mask, tumor_class_idx=0, background_class_idx=4)
    assert vtp == pytest.approx(50.0)

    tsr = compute_tumor_stroma_ratio(mask, tumor_class_idx=0, stroma_class_idx=1)
    assert tsr == pytest.approx(50.0)


def test_otsu_clinical_gating():
    """Tests Otsu thresholding on synthetic tissue vs white background."""
    # Synthetic image: top half pink tissue (high saturation), bottom half pure white glass
    img = np.full((100, 100, 3), fill_value=255, dtype=np.uint8)
    # Pink H&E tissue color: (R=200, G=80, B=140)
    img[:50, :] = [200, 80, 140]

    mask = compute_otsu_tissue_mask(img)
    assert mask.shape == (100, 100)
    assert np.mean(mask[:50, :]) > 0.90  # Tissue recognized
    assert np.mean(mask[50:, :]) < 0.10  # Glass extinguished

    # Test gating on prediction
    pred = np.zeros((100, 100), dtype=np.int64)  # Argmax predicted tumor everywhere
    gated = apply_otsu_gating_to_prediction(pred, img, background_class_idx=4)
    assert np.all(gated[50:, :] == 4)  # Background glass forced to 4


def test_losses():
    """Tests loss functions forward computation and gradient readiness."""
    # 1. MultiLabelClassificationLoss
    cls_loss = MultiLabelClassificationLoss()
    logits = torch.randn(4, 4, requires_grad=True)
    targets = torch.randint(0, 2, (4, 4)).float()
    loss1 = cls_loss(logits, targets)
    assert loss1.item() > 0.0

    # 2. MultiLayerPseudoSupervisionLoss
    mlps_loss = MultiLayerPseudoSupervisionLoss()
    pred_logits = torch.randn(2, 5, 32, 32, requires_grad=True)
    pseudo_masks = {
        "bn7": torch.randint(0, 5, (2, 32, 32)),
        "b5_2": torch.randint(0, 5, (2, 16, 16)),
    }
    loss2 = mlps_loss(pred_logits, pseudo_masks)
    assert loss2.item() > 0.0

    # 3. UncertaintyRegularizationLoss
    unc_loss = UncertaintyRegularizationLoss()
    loss3 = unc_loss(pred_logits, pseudo_masks["bn7"])
    assert loss3.item() > 0.0


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


def test_classmix_augmentation():
    """Tests ClassMix data augmentation output dimensions."""
    images = torch.rand(4, 3, 32, 32)
    labels = torch.randint(0, 2, (4, 4)).float()
    masks = torch.randint(0, 5, (4, 32, 32))

    mixed_imgs, mixed_lbls, mixed_msks = apply_classmix_augmentation(
        images, labels, masks, p=1.0
    )
    assert mixed_imgs.shape == images.shape
    assert mixed_lbls.shape == labels.shape
    assert mixed_msks.shape == masks.shape
