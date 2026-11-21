"""
Unit Tests for Loss Functions and MultiPrototypePPCLoss Upgrades (SE374.R11).

Covers:
- MultiPrototypePPCLoss empirical BCSS hard-pair weights ((0,1)=2.0, (1,2)=2.5, (0,3)=2.0, (2,3)=1.0)
- Single-class anti-glass guard (pre-simplex channel-max CAM >= 0.35 + non-white tissue mask)
- Warm-started EMA centroids for num_prototypes=1
- MultiLabelClassificationLoss, MLPS, UncertaintyRegularizationLoss, and UABL edge cases
- Pure ASCII hyphen '-' (0x2D) invariant check (Invariant 12)
"""

from pathlib import Path

import pytest
import torch

from src.losses_metrics.losses import (
    MultiLabelClassificationLoss,
    MultiLayerPseudoSupervisionLoss,
    MultiPrototypePPCLoss,
    UncertaintyAwareBoundaryLoss,
    UncertaintyRegularizationLoss,
)


def test_mpppc_hard_pair_weights_match_bcss_confusion_matrix() -> None:
    """Verify MultiPrototypePPCLoss hard_weights match BCSS confusion matrix."""
    ppc = MultiPrototypePPCLoss(num_classes=4, num_prototypes=1, feat_dim=32)
    hw = ppc.hard_weights

    assert hw[0, 1].item() == pytest.approx(2.0)
    assert hw[1, 0].item() == pytest.approx(2.0)
    assert hw[1, 2].item() == pytest.approx(2.5)
    assert hw[2, 1].item() == pytest.approx(2.5)
    assert hw[0, 3].item() == pytest.approx(2.0)
    assert hw[3, 0].item() == pytest.approx(2.0)
    assert hw[2, 3].item() == pytest.approx(1.0)
    assert hw[3, 2].item() == pytest.approx(1.0)
    assert torch.allclose(hw, hw.t())


def test_mpppc_single_class_anti_glass_guard() -> None:
    """Verify single-class [1,0,0,0] patches reject bright white glass and low-CAM pixels."""
    torch.manual_seed(7)
    ppc = MultiPrototypePPCLoss(
        num_classes=4,
        num_prototypes=1,
        feat_dim=16,
        conf_thresh=0.70,
        max_samples=256,
    )
    ppc.train()

    b, d, h, w = 2, 16, 16, 16
    proj_feats = torch.randn(b, d, h, w, requires_grad=True)
    # Single-class patch: valid_probs[:, 0] becomes 1.0 everywhere after gating
    img_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    cam_probs = torch.full((b, 4, h, w), 0.25)

    # 1. Pure white glass RGB (mean >= 0.90, sat <= 0.08) + low CAM -> rejected -> 0.0 loss
    glass_rgb = torch.full((b, 3, 64, 64), 0.96)
    low_cams = torch.zeros((b, 4, h, w))
    low_cams[:, 0, 0, 0] = 10.0  # One tiny spike, all other pixels ~0.0 (< 0.35 max-norm)
    loss_glass = ppc(proj_feats, cam_probs, img_labels, raw_cams=low_cams, rgb_images=glass_rgb)
    assert loss_glass.item() == pytest.approx(0.0, abs=1e-7)

    # 2. Bright glass rejected even if CAM is uniformly active
    uniform_high_cams = torch.ones((b, 4, h, w))
    loss_glass_only = ppc(proj_feats, cam_probs, img_labels, raw_cams=uniform_high_cams, rgb_images=glass_rgb)
    assert loss_glass_only.item() == pytest.approx(0.0, abs=1e-7)

    # 2b. ImageNet-normalized all-white glass (rgb_01=0.95 -> min~2.03, max~2.42) must also be rejected
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imagenet_glass_rgb = (torch.full((b, 3, 64, 64), 0.95) - mean_t) / std_t
    assert float(imagenet_glass_rgb.min()) > 1.5 and float(imagenet_glass_rgb.max()) < 3.0
    loss_imagenet_glass = ppc(
        proj_feats, cam_probs, img_labels, raw_cams=uniform_high_cams, rgb_images=imagenet_glass_rgb
    )
    assert loss_imagenet_glass.item() == pytest.approx(0.0, abs=1e-7)
    assert not bool(ppc.prototype_initialized.any().item())

    # 3. Non-white H&E tissue RGB + high CAM -> accepted -> positive finite loss + valid grad
    tissue_rgb = torch.full((b, 3, 64, 64), 0.55)
    tissue_rgb[:, 0, :, :] = 0.72  # Pinkish eosin stain (sat > 0.08, mean < 0.90)
    loss_tissue = ppc(proj_feats, cam_probs, img_labels, raw_cams=uniform_high_cams, rgb_images=tissue_rgb)
    assert torch.isfinite(loss_tissue)
    assert loss_tissue.item() > 0.0
    loss_tissue.backward()
    assert proj_feats.grad is not None and torch.isfinite(proj_feats.grad).all()


def test_mpppc_num_prototypes_one_warm_starts_centroid() -> None:
    """Verify num_prototypes=1 warm-starts uninitialized class centroids on first batch instead of 0.99*random."""
    torch.manual_seed(123)
    ppc = MultiPrototypePPCLoss(
        num_classes=4,
        num_prototypes=1,
        feat_dim=16,
        momentum=0.99,
        conf_thresh=0.70,
        warm_start=True,
    )
    ppc.train()
    assert not bool(ppc.prototype_initialized.any().item())

    b, d, h, w = 2, 16, 8, 8
    # Construct pure unit features along axis 0 for class 0, axis 1 for class 1
    feats = torch.zeros(b, d, h, w)
    feats[0, 0, :, :] = 3.0
    feats[1, 1, :, :] = 3.0
    cam_probs = torch.full((b, 4, h, w), 0.01)
    cam_probs[0, 0, :, :] = 0.97
    cam_probs[1, 1, :, :] = 0.97
    img_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    loss = ppc(feats, cam_probs, img_labels)
    assert torch.isfinite(loss)
    assert bool(ppc.prototype_initialized[0, 0].item()) is True
    assert bool(ppc.prototype_initialized[1, 0].item()) is True
    assert bool(ppc.prototype_initialized[2, 0].item()) is False

    # Warm-started prototype must match the data centroid exactly (not 0.99*random + 0.01*data)
    expected_c0 = torch.zeros(d)
    expected_c0[0] = 1.0
    expected_c1 = torch.zeros(d)
    expected_c1[1] = 1.0
    assert torch.allclose(ppc.prototypes[0, 0], expected_c0, atol=1e-5)
    assert torch.allclose(ppc.prototypes[1, 0], expected_c1, atol=1e-5)

    # Subsequent update uses momentum=0.99 EMA blending
    feats_step2 = torch.zeros(b, d, h, w)
    feats_step2[0, 2, :, :] = 3.0
    feats_step2[1, 1, :, :] = 3.0
    _ = ppc(feats_step2, cam_probs, img_labels)
    assert ppc.prototypes[0, 0, 0].item() > 0.95  # Kept ~99% of warm-started centroid


def test_mpppc_resume_before_init_still_warm_starts() -> None:
    """Verify Arm C MP-PPC warm-start fires after state_dict save/reload before first PPC epoch."""
    torch.manual_seed(11)
    ppc_ep2 = MultiPrototypePPCLoss(
        num_classes=4,
        num_prototypes=1,
        feat_dim=16,
        momentum=0.99,
        conf_thresh=0.70,
        warm_start=True,
    )
    assert not hasattr(ppc_ep2, "_init_protos_ref")
    saved_state = {k: v.clone() for k, v in ppc_ep2.state_dict().items()}
    assert "_init_protos_ref" not in saved_state
    assert not bool(saved_state["prototype_initialized"].any().item())

    # Simulate fresh process at Ep 3 with different random init before loading Ep 2 checkpoint
    torch.manual_seed(999)
    ppc_resumed = MultiPrototypePPCLoss(
        num_classes=4,
        num_prototypes=1,
        feat_dim=16,
        momentum=0.99,
        conf_thresh=0.70,
        warm_start=True,
    )
    assert not torch.allclose(ppc_resumed.prototypes, saved_state["prototypes"])
    ppc_resumed.load_state_dict(saved_state, strict=True)
    ppc_resumed.train()

    b, d, h, w = 2, 16, 8, 8
    feats = torch.zeros(b, d, h, w)
    feats[0, 0, :, :] = 3.0
    feats[1, 1, :, :] = 3.0
    cam_probs = torch.full((b, 4, h, w), 0.01)
    cam_probs[0, 0, :, :] = 0.97
    cam_probs[1, 1, :, :] = 0.97
    img_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    loss = ppc_resumed(feats, cam_probs, img_labels)
    assert torch.isfinite(loss)
    assert bool(ppc_resumed.prototype_initialized[0, 0].item()) is True
    assert bool(ppc_resumed.prototype_initialized[1, 0].item()) is True

    expected_c0 = torch.zeros(d)
    expected_c0[0] = 1.0
    expected_c1 = torch.zeros(d)
    expected_c1[1] = 1.0
    assert torch.allclose(ppc_resumed.prototypes[0, 0], expected_c0, atol=1e-5)
    assert torch.allclose(ppc_resumed.prototypes[1, 0], expected_c1, atol=1e-5)

    # Second batch after init uses EMA momentum=0.99
    feats_next = torch.zeros(b, d, h, w)
    feats_next[0, 2, :, :] = 3.0
    feats_next[1, 1, :, :] = 3.0
    _ = ppc_resumed(feats_next, cam_probs, img_labels)
    assert ppc_resumed.prototypes[0, 0, 0].item() > 0.95


def test_auxiliary_losses_edge_cases_and_gradients() -> None:
    """Verify MultiLabelClassificationLoss, MLPS, UncertaintyRegularization, and UABL."""
    cls_loss = MultiLabelClassificationLoss()
    logits = torch.randn(4, 4, requires_grad=True)
    targets = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
    l_cls = cls_loss(logits, targets)
    assert torch.isfinite(l_cls) and l_cls.item() > 0.0

    # All-ignore pseudo-masks must return zero finite loss without NaN
    seg_logits = torch.randn(2, 5, 16, 16, requires_grad=True)
    all_ignore = torch.full((2, 16, 16), 255, dtype=torch.long)
    mlps = MultiLayerPseudoSupervisionLoss()
    unc = UncertaintyRegularizationLoss()
    l_mlps = mlps(seg_logits, {"bn7": all_ignore, "b5_2": all_ignore, "b4_5": all_ignore})
    l_unc = unc(seg_logits, all_ignore)
    assert l_mlps.item() == pytest.approx(0.0)
    assert l_unc.item() == pytest.approx(0.0)

    uabl = UncertaintyAwareBoundaryLoss()
    probs = torch.softmax(seg_logits, dim=1)
    l_uabl = uabl(probs, torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16))
    assert torch.isfinite(l_uabl) and l_uabl.item() >= 0.0
    l_uabl.backward()
    assert seg_logits.grad is not None and torch.isfinite(seg_logits.grad).all()


def test_invariant_12_losses_metrics_ascii_hyphen_only() -> None:
    """Verify zero non-ASCII dashes in src/losses_metrics/losses.py and tests/test_losses_metrics.py."""
    forbidden = ["\u2013", "\u2014", "\u2012", "\u2015", "\u2212"]
    root = Path(__file__).resolve().parent.parent
    for path in (
        root / "src" / "losses_metrics" / "losses.py",
        root / "tests" / "test_losses_metrics.py",
    ):
        text = path.read_text(encoding="utf-8")
        for bad in forbidden:
            assert bad not in text, f"Forbidden dash {repr(bad)} found in {path}"
