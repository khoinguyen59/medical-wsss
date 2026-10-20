"""
Unit Tests for DeepLabV3PlusWithCGM (SE374.R11 WSSS).

Verifies:
1. ASPP module output shapes and multi-scale context aggregation.
2. Decoder module feature fusion and spatial dimension handling.
3. End-to-end forward pass with input tensor (B, 3, H, W).
4. Classification Gate Mechanism (CGM): mathematical logit gating and absolute elimination
   of false-positive pixels for absent tissue classes.
5. Multi-loss training support: CrossEntropyLoss, MultiLayerPseudoSupervisionLoss (MLPS),
   and UncertaintyRegularizationLoss.
6. Inference methods: predict() argmax masks and predict_proba() softmax maps.
7. Single-item batch (B=1) edge cases for clinical reliability.

Strictly adheres to ASCII hyphens (-) only.
"""

from typing import Dict
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.losses_metrics.losses import (
    MultiLayerPseudoSupervisionLoss,
    UncertaintyRegularizationLoss,
)
from src.models.deeplabv3plus import (
    ASPP,
    ClassificationGateMechanism,
    DeepLabV3PlusDecoder,
    DeepLabV3PlusWithCGM,
    ResNetBackbone,
)


def test_aspp_output_shapes():
    """Verifies ASPP module output dimensions and dilation rates [1, 6, 12, 18]."""
    in_channels = 512
    out_channels = 256
    batch_size = 2
    height, width = 16, 16

    aspp = ASPP(
        in_channels=in_channels,
        out_channels=out_channels,
        dilations=[1, 6, 12, 18],
        dropout_rate=0.2,
    )
    x = torch.randn(batch_size, in_channels, height, width)
    out = aspp(x)

    assert out.shape == (batch_size, out_channels, height, width)
    assert not torch.isnan(out).any()


def test_decoder_output_shapes():
    """Verifies Decoder module fusing low-level and ASPP features."""
    batch_size = 2
    num_classes = 5
    aspp_feat = torch.randn(batch_size, 256, 16, 16)
    low_level_feat = torch.randn(batch_size, 64, 64, 64)

    decoder = DeepLabV3PlusDecoder(
        low_level_in_channels=64,
        aspp_in_channels=256,
        num_classes=num_classes,
        low_level_projected_channels=48,
        decoder_channels=256,
    )

    out = decoder(aspp_feat, low_level_feat)
    assert out.shape == (batch_size, num_classes, 64, 64)


def test_deeplabv3plus_forward_pass():
    """Tests end-to-end forward pass on (B, 3, H, W) input tensor."""
    model = DeepLabV3PlusWithCGM(
        num_classes=5,
        in_channels=3,
        backbone_channels=512,
        low_level_channels=64,
    )
    model.eval()

    # Test standard resolution
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 5, 128, 128)
    assert not torch.isnan(logits).any()

    # Test dictionary output
    output_dict = model(x, return_dict=True)
    assert isinstance(output_dict, dict)
    assert "logits" in output_dict
    assert "raw_logits" in output_dict
    assert "features" in output_dict
    assert "aspp_features" in output_dict
    assert output_dict["logits"].shape == (2, 5, 128, 128)
    assert "b4_5" in output_dict["features"]
    assert "b5_2" in output_dict["features"]
    assert "bn7" in output_dict["features"]


def test_deeplabv3plus_gradient_flow():
    """Verifies that gradients propagate through the entire network during training."""
    model = DeepLabV3PlusWithCGM(num_classes=5)
    model.train()

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    targets = torch.randint(0, 5, (2, 64, 64))

    logits, loss = model(x, targets=targets, return_loss=True)
    assert loss.item() > 0.0

    loss.backward()
    assert x.grad is not None
    # Check that decoder weights receive gradients
    assert model.decoder.classifier.weight.grad is not None
    assert model.aspp.branch1[0].weight.grad is not None


def test_classification_gate_mechanism_direct():
    """Tests CGM mathematical gating formula: Z_hat_c = Z_c * y_c + (1 - y_c) * penalty."""
    cgm = ClassificationGateMechanism(penalty_value=-1e4, threshold=0.5)

    # Synthetic logits (B=2, C=5, H=4, W=4)
    logits = torch.full((2, 5, 4, 4), fill_value=2.5)

    # Batch item 0: class 1 and 3 absent (0)
    # Batch item 1: class 0, 2, 4 absent (0)
    presence = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
    ])

    gated = cgm(logits, presence)

    # Check batch item 0
    assert torch.all(gated[0, 0] == 2.5)
    assert torch.all(gated[0, 1] == -1e4)
    assert torch.all(gated[0, 2] == 2.5)
    assert torch.all(gated[0, 3] == -1e4)
    assert torch.all(gated[0, 4] == 2.5)

    # Check batch item 1
    assert torch.all(gated[1, 0] == -1e4)
    assert torch.all(gated[1, 1] == 2.5)
    assert torch.all(gated[1, 2] == -1e4)
    assert torch.all(gated[1, 3] == 2.5)
    assert torch.all(gated[1, 4] == -1e4)


def test_cgm_extinguishes_absent_class_in_segmentation():
    """Verifies that absent classes in y produce exactly zero pixels in predicted masks."""
    model = DeepLabV3PlusWithCGM(num_classes=5)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    # Mark class 1 (STR) and class 3 (NEC) absent
    class_presence = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
    ])

    pred_masks = model.predict(x, class_presence=class_presence)  # (2, 64, 64)

    # For item 0: class 1 and class 3 must have 0 pixels
    assert (pred_masks[0] == 1).sum().item() == 0, "Class 1 must have 0 pixels when absent"
    assert (pred_masks[0] == 3).sum().item() == 0, "Class 3 must have 0 pixels when absent"

    # For item 1: class 1, 2, 3 must have 0 pixels
    assert (pred_masks[1] == 1).sum().item() == 0, "Class 1 must have 0 pixels when absent"
    assert (pred_masks[1] == 2).sum().item() == 0, "Class 2 must have 0 pixels when absent"
    assert (pred_masks[1] == 3).sum().item() == 0, "Class 3 must have 0 pixels when absent"

    # Verify softmax probabilities approach zero for absent classes
    probs = model.predict_proba(x, class_presence=class_presence)
    assert probs[:, 1].max().item() < 1e-4, "Softmax prob for absent class 1 must be near 0"
    assert probs[:, 3].max().item() < 1e-4, "Softmax prob for absent class 3 must be near 0"


def test_cgm_with_none_presence():
    """Verifies that CGM behaves as an identity operation when class_presence is None."""
    model = DeepLabV3PlusWithCGM(num_classes=5)
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    out_dict = model(x, class_presence=None, return_dict=True)

    # When class_presence is None, gated logits must equal raw logits
    assert torch.allclose(out_dict["logits"], out_dict["raw_logits"])


def test_deeplabv3plus_loss_ce():
    """Tests training loss computation with standard CrossEntropyLoss."""
    model = DeepLabV3PlusWithCGM(num_classes=5)
    x = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 5, (2, 64, 64))

    logits, loss = model(x, targets=targets, loss_type="ce", return_loss=True)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_deeplabv3plus_loss_mlps():
    """Tests end-to-end training with MultiLayerPseudoSupervisionLoss."""
    model = DeepLabV3PlusWithCGM(num_classes=5)
    x = torch.randn(2, 3, 64, 64)

    # Pseudo-masks from multi-resolution feature stages
    pseudo_masks: Dict[str, torch.Tensor] = {
        "bn7": torch.randint(0, 5, (2, 64, 64)),
        "b5_2": torch.randint(0, 5, (2, 32, 32)),
        "b4_5": torch.randint(0, 5, (2, 16, 16)),
    }

    logits, loss = model(x, pseudo_masks=pseudo_masks, loss_type="mlps", return_loss=True)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_deeplabv3plus_loss_uncertainty():
    """Tests end-to-end training with UncertaintyRegularizationLoss."""
    model = DeepLabV3PlusWithCGM(num_classes=5)
    x = torch.randn(2, 3, 64, 64)
    pseudo_mask = torch.randint(0, 5, (2, 64, 64))

    logits, loss = model(x, pseudo_masks=pseudo_mask, loss_type="uncertainty", return_loss=True)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_single_item_batch_edge_case():
    """Tests single-item batch (B=1) edge case for memory safety and clinical deployment."""
    model = DeepLabV3PlusWithCGM(num_classes=5)
    model.eval()

    x = torch.randn(1, 3, 96, 96)
    presence = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])

    mask = model.predict(x, class_presence=presence)
    assert mask.shape == (1, 96, 96)
    assert (mask == 2).sum().item() == 0
    assert (mask == 3).sum().item() == 0
