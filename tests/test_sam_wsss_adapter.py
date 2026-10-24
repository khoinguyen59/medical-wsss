"""
Unit Tests for Proposed MedSAM WSSS Adapter, Prompt Generator, and Composite Loss.

Course: SE374.R11 - Medical Image Processing Software Development
Verifies:
- CAMPromptGenerator: local maxima / centroid positive prompts and negative glass prompts
- ResidualAdapterBottleneck: shape preservation and identity initialization
- MedSAM_WSSS_Adapter: frozen image encoder, adapter cross-attention, 5-class decoding
- Parameter freezing: 100% frozen ViT encoder with zero gradient leak
- CompositeWSSSLoss: Multi-Class Focal (15:1 imbalance) + Soft Dice + IoU + Head MSE
- Training pipeline: 1-step dry run, checkpointing, and resume verification
- Invariant 12: strictly ASCII hyphens (0x2D) only

Strict Invariant 12 compliance: 100% ASCII 0x2D hyphens only.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

# Safeguard against OpenMP library collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models.sam_wsss_adapter import (
    CAMPromptGenerator,
    ResidualAdapterBottleneck,
    HistologicalMaskDecoder,
    MedSAM_WSSS_Adapter,
)
from src.training.train_proposed_sam import (
    CompositeWSSSLoss,
    SyntheticHistologyDataset,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)


# ==============================================================================
# 1. Tests for CAMPromptGenerator
# ==============================================================================

def test_cam_prompt_generator_shapes_and_coordinates():
    """Verifies output shapes, coordinate conventions, and label values."""
    gen = CAMPromptGenerator(
        num_classes=4,
        max_pos_points=3,
        max_neg_points=3,
        target_size=(1024, 1024),
    )

    b, c, h, w = 2, 4, 128, 128
    torch.manual_seed(42)
    cams = torch.rand(b, c, h, w)
    cls_scores = torch.tensor([[0.9, 0.8, 0.05, 0.0], [0.1, 0.9, 0.7, 0.8]])

    results = gen(cams=cams, cls_scores=cls_scores)

    assert len(results) == 4
    total_pts = 3 + 3

    for class_idx in range(4):
        coords = results[class_idx]["point_coords"]
        labels = results[class_idx]["point_labels"]
        active = results[class_idx]["active_mask"]

        # Check tensor shapes
        assert coords.shape == (b, total_pts, 2)
        assert labels.shape == (b, total_pts)
        assert active.shape == (b,)

        # Check coordinate bounds in target frame (1024 x 1024)
        assert (coords[:, :, 0] >= 0.0).all() and (coords[:, :, 0] <= 1024.0).all()
        assert (coords[:, :, 1] >= 0.0).all() and (coords[:, :, 1] <= 1024.0).all()

        # Check label values belong to {-1, 0, 1}
        unique_labels = torch.unique(labels).cpu().tolist()
        for ul in unique_labels:
            assert ul in [-1, 0, 1]


def test_cam_prompt_generator_active_vs_inactive():
    """Verifies that inactive classes do not mine positive prompts."""
    gen = CAMPromptGenerator(num_classes=4, max_pos_points=2, max_neg_points=2)
    cams = torch.rand(1, 4, 64, 64)
    # Class 0 and 1 active; Class 2 and 3 inactive
    cls_scores = torch.tensor([[0.95, 0.85, 0.02, 0.01]])

    res = gen(cams=cams, cls_scores=cls_scores)

    assert res[0]["active_mask"][0].item() is True
    assert res[1]["active_mask"][0].item() is True
    assert res[2]["active_mask"][0].item() is False
    assert res[3]["active_mask"][0].item() is False

    # Inactive class should not have positive label (+1)
    lbls_inactive = res[2]["point_labels"][0]
    assert 1 not in lbls_inactive.cpu().tolist()


def test_cam_prompt_generator_blank_glass_suppression():
    """Verifies prompt generation on low-activation blank glass."""
    gen = CAMPromptGenerator(num_classes=4, max_pos_points=2, max_neg_points=2)
    # All zeros CAM (blank background)
    cams = torch.zeros(1, 4, 64, 64)
    cls_scores = torch.zeros(1, 4)
    sat = torch.zeros(1, 64, 64)  # Zero saturation (pure glass)

    res = gen(cams=cams, cls_scores=cls_scores, hsv_saturation=sat)

    for c in range(4):
        assert res[c]["active_mask"][0].item() is False
        # Coordinates must be valid non-NaN numbers
        assert not torch.isnan(res[c]["point_coords"]).any()


# ==============================================================================
# 2. Tests for Residual Adapter Bottleneck
# ==============================================================================

def test_residual_adapter_bottleneck_shape_and_identity():
    """Verifies shape preservation and identity initialization."""
    adapter = ResidualAdapterBottleneck(in_dim=256, mid_dim=64, scale=0.1)

    # 3D token input: (B, N, C)
    x3d = torch.randn(2, 16, 256)
    out3d = adapter(x3d)
    assert out3d.shape == (2, 16, 256)

    # Due to zero initialization of up_proj, output should be identical to input at init
    assert torch.allclose(out3d, x3d, atol=1e-6)

    # 4D spatial feature input: (B, C, H, W)
    x4d = torch.randn(2, 256, 14, 14)
    out4d = adapter(x4d)
    assert out4d.shape == (2, 256, 14, 14)
    assert torch.allclose(out4d, x4d, atol=1e-6)


def test_residual_adapter_gradient_flow():
    """Verifies gradient propagation through adapter layers."""
    adapter = ResidualAdapterBottleneck(in_dim=256, mid_dim=64, scale=0.1)
    x = torch.randn(2, 8, 256, requires_grad=True)
    out = adapter(x)
    loss = out.sum()
    loss.backward()

    assert adapter.down_proj.weight.grad is not None
    assert adapter.up_proj.weight.grad is not None
    assert x.grad is not None


# ==============================================================================
# 3. Tests for MedSAM_WSSS_Adapter Architecture
# ==============================================================================

def test_medsam_wsss_adapter_surrogate_forward():
    """Verifies surrogate forward pass, output shapes, and Softmax normalization."""
    model = MedSAM_WSSS_Adapter(
        checkpoint_path=None,
        freeze_image_encoder=True,
        adapter_mid_dim=64,
        num_classes=5,
    )

    images = torch.randn(2, 3, 224, 224)
    out = model(images)

    assert "logits" in out
    assert "probabilities" in out
    assert "iou_predictions" in out

    logits = out["logits"]
    probs = out["probabilities"]
    iou_preds = out["iou_predictions"]

    assert logits.shape == (2, 5, 224, 224)
    assert probs.shape == (2, 5, 224, 224)
    assert iou_preds.shape == (2, 5)

    # Probabilities must sum to 1.0 along channel axis
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_medsam_wsss_adapter_prompt_guided_forward():
    """Verifies forward pass with point prompts."""
    model = MedSAM_WSSS_Adapter(checkpoint_path=None, num_classes=5)
    images = torch.randn(2, 3, 224, 224)

    # Sparse point prompts
    coords = torch.tensor([
        [[30.0, 40.0], [100.0, 120.0], [0.0, 0.0]],
        [[50.0, 60.0], [80.0, 90.0], [0.0, 0.0]],
    ])
    labels = torch.tensor([[1, 0, -1], [1, 1, -1]])

    out = model(images, point_coords=coords, point_labels=labels)
    assert out["logits"].shape == (2, 5, 224, 224)


def test_medsam_wsss_adapter_parameter_freezing_and_gradients():
    """Verifies strict freezing of image encoder and active gradient flow to adapters."""
    model = MedSAM_WSSS_Adapter(checkpoint_path=None, freeze_image_encoder=True)

    # Check that image encoder parameters are frozen
    for name, param in model.image_encoder.named_parameters():
        assert not param.requires_grad, f"Parameter {name} in image encoder should be frozen!"

    # Check that mask decoder adapter parameters are trainable
    trainable_found = False
    for name, param in model.mask_decoder.named_parameters():
        if "adapter" in name or "mask_tokens" in name:
            if param.requires_grad:
                trainable_found = True
    assert trainable_found, "Mask decoder adapters should have trainable parameters!"

    # Verify gradient backpropagation
    images = torch.randn(2, 3, 112, 112)
    out = model(images)
    loss = out["logits"].sum()
    loss.backward()

    # Image encoder must have zero gradients
    for name, param in model.image_encoder.named_parameters():
        assert param.grad is None, f"Frozen parameter {name} received unexpected gradient!"

    # Mask decoder adapters must receive gradients
    has_grad = False
    for param in model.mask_decoder.parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Trainable mask decoder parameters should receive gradients!"


# ==============================================================================
# 4. Tests for Composite WSSS Loss
# ==============================================================================

def test_composite_wsss_loss_components():
    """Verifies Focal, Soft Dice, IoU, and Head MSE loss components."""
    criterion = CompositeWSSSLoss(
        num_classes=5,
        focal_weight=1.0,
        dice_weight=1.0,
        iou_weight=0.5,
        head_weight=0.2,
    )

    pred_logits = torch.randn(2, 5, 64, 64)
    target_labels = torch.randint(0, 5, (2, 64, 64), dtype=torch.long)
    # Include ignore index 255
    target_labels[:, :5, :5] = 255

    iou_predictions = torch.rand(2, 5)

    loss_dict = criterion(pred_logits, target_labels, iou_predictions)

    assert "total_loss" in loss_dict
    assert "focal_loss" in loss_dict
    assert "dice_loss" in loss_dict
    assert "iou_loss" in loss_dict
    assert "head_loss" in loss_dict

    total = loss_dict["total_loss"]
    assert total.item() > 0.0
    assert not torch.isnan(total)
    assert not torch.isinf(total)


def test_composite_wsss_loss_exact_match():
    """Verifies that Dice and IoU losses are minimal when predictions match targets."""
    criterion = CompositeWSSSLoss(num_classes=5)

    # Construct one-hot logits with huge confidence
    targets = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    logits = torch.zeros(1, 5, 2, 2)
    for r in range(2):
        for c in range(2):
            lbl = targets[r, c].item()
            logits[0, lbl, r, c] = 20.0  # High confidence

    loss_dict = criterion(logits, targets.unsqueeze(0))

    # Dice loss on foreground should be close to 0
    assert loss_dict["dice_loss"].item() < 0.05
    assert loss_dict["iou_loss"].item() < 0.05


# ==============================================================================
# 5. Tests for Training & Resume Pipeline
# ==============================================================================

def test_training_pipeline_dry_run_and_checkpointing():
    """Verifies 1-step training, evaluation, saving, and resumption."""
    device = torch.device("cpu")
    model = MedSAM_WSSS_Adapter(checkpoint_path=None, num_classes=5).to(device)
    prompt_gen = CAMPromptGenerator(num_classes=4)
    criterion = CompositeWSSSLoss(num_classes=5).to(device)

    dataset = SyntheticHistologyDataset(num_samples=4, image_size=64)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)

    # 1. Run one epoch
    train_stats = train_one_epoch(
        model=model,
        prompt_generator=prompt_gen,
        dataloader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=None,
        device=device,
        accum_steps=1,
        max_steps=2,
    )

    assert "loss" in train_stats
    assert train_stats["loss"] > 0.0

    # 2. Evaluate
    val_stats = evaluate(model=model, dataloader=loader, device=device, max_steps=2)
    assert "mIoU_fg" in val_stats
    assert "mIoU_all" in val_stats

    # 3. Save checkpoint
    with tempfile.TemporaryDirectory() as tmp_dir:
        state = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mIoU": val_stats["mIoU_fg"],
        }
        best_path = save_checkpoint(state, is_best=True, checkpoint_dir=tmp_dir)
        assert os.path.exists(best_path)

        # 4. Resume checkpoint
        new_model = MedSAM_WSSS_Adapter(checkpoint_path=None, num_classes=5).to(device)
        new_opt = torch.optim.AdamW(new_model.get_trainable_parameters(), lr=1e-4)
        loaded = load_checkpoint(best_path, new_model, new_opt)

        assert loaded["epoch"] == 0
        assert loaded["best_mIoU"] == val_stats["mIoU_fg"]


# ==============================================================================
# 6. Governance Invariant 12 Verification (Strict ASCII Hyphen Only)
# ==============================================================================

def test_governance_invariant_12_strict_ascii_hyphens():
    """Audits modified files to enforce 0 non-ASCII dashes."""
    forbidden = ["\u2013", "\u2014", "\u2012", "\u2015", "\u2212"]
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    target_files = [
        os.path.join(tests_dir, "..", "src", "models", "sam_wsss_adapter.py"),
        os.path.join(tests_dir, "..", "src", "training", "train_proposed_sam.py"),
        os.path.join(tests_dir, "test_sam_wsss_adapter.py"),
    ]

    for fpath in target_files:
        norm_path = os.path.abspath(fpath)
        assert os.path.exists(norm_path), f"File not found: {norm_path}"
        with open(norm_path, "r", encoding="utf-8") as f:
            content = f.read()

        for char in forbidden:
            count = content.count(char)
            assert count == 0, (
                f"Invariant 12 Violation: Found {count} instances of forbidden character "
                f"U+{ord(char):04X} in {norm_path}"
            )
