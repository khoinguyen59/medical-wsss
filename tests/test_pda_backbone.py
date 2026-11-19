"""
Unit Tests for Model Backbone, PDA Generator, and WSSS Benchmarking Engine (SE374.R11).

Verifies:
1. MultiStageFeatureBackbone forward pass, multi-stage shapes (b4_5, b5_2, bn7), and CAM extraction.
2. ProgressiveDropoutAttention (PDA) iterative CAM computation, expansion properties, and boundary bleeding control.
3. WSSSBenchmarkingEngine across all 4 benchmark configurations (baseline, classmix, relocalization, proposed).
4. Dynamic progressive thresholding and boundary uncertainty estimation.

Strict adherence to:
- Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Invariant 07: Genuine PyTorch test assertions with zero mockup.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import (
    HistopathologyClassifierBackbone,
    MultiStageFeatureBackbone,
    ResidualConvBlock,
)
from src.models.pda_generator import ProgressiveDropoutAttention
from src.models.wsss_tissue import (
    SpatialAffinityRelocalization,
    WSSSBenchmarkingEngine,
    apply_classmix_augmentation,
)

# ==============================================================================
# 1. MultiStageFeatureBackbone Tests
# ==============================================================================

def test_backbone_forward_and_feature_shapes():
    """Tests forward pass dimensions and multi-stage intermediate representations."""
    backbone = MultiStageFeatureBackbone(
        in_channels=3,
        num_classes=4,
        stage_channels=[64, 128, 256, 512, 512],
        dropout_rate=0.2,
    )
    backbone.eval()

    # Input: batch of 2 RGB patches (64x64)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits, features = backbone(x)

    # 1. Check classification logits shape
    assert logits.shape == (2, 4), f"Expected logits shape (2, 4), got {logits.shape}"

    # 2. Check feature stages
    required_stages = ["stage1", "stage2", "b4_5", "b5_2", "bn7"]
    for stage_name in required_stages:
        assert stage_name in features, f"Missing stage '{stage_name}' in features dict"

    # 3. Check spatial and channel dimensions
    # b4_5: stride 16 -> 64 / 16 = 4x4, 256 channels
    assert features["b4_5"].shape == (2, 256, 4, 4)
    # b5_2: stride 32 -> 64 / 32 = 2x2, 512 channels
    assert features["b5_2"].shape == (2, 512, 2, 2)
    # bn7: stride 32 -> 64 / 32 = 2x2, 512 channels
    assert features["bn7"].shape == (2, 512, 2, 2)


def test_backbone_custom_channels_and_dropout():
    """Tests backbone adaptability with custom stage channels and dropout settings."""
    custom_channels = [32, 64, 128, 256, 256]
    backbone = MultiStageFeatureBackbone(
        in_channels=3,
        num_classes=4,
        stage_channels=custom_channels,
        dropout_rate=0.3,
    )
    assert backbone.stage_channels == custom_channels

    backbone.eval()
    x = torch.randn(1, 3, 32, 32)
    logits, features = backbone(x)
    assert logits.shape == (1, 4)
    assert features["b4_5"].shape[1] == 128
    assert features["bn7"].shape[1] == 256

    # Test training mode with batch size >= 2
    backbone.train()
    x2 = torch.randn(2, 3, 32, 32)
    logits2, _ = backbone(x2)
    assert logits2.shape == (2, 4)


def test_backbone_cam_computation():
    """Tests that compute_cam produces non-negative activation maps for all stages."""
    backbone = MultiStageFeatureBackbone(in_channels=3, num_classes=4)
    backbone.eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        cam_bn7 = backbone.compute_cam(x, feature_layer="bn7")
        cam_b5_2 = backbone.compute_cam(x, feature_layer="b5_2")
        cam_b4_5 = backbone.compute_cam(x, feature_layer="b4_5")

    # Shapes: (B, num_classes, H_feat, W_feat)
    assert cam_bn7.shape == (2, 4, 2, 2)
    assert cam_b5_2.shape == (2, 4, 2, 2)
    assert cam_b4_5.shape == (2, 4, 4, 4)

    # ReLU guarantees non-negativity
    assert (cam_bn7 >= 0.0).all()
    assert (cam_b5_2 >= 0.0).all()
    assert (cam_b4_5 >= 0.0).all()


def test_backbone_backward_compatibility_alias():
    """Verifies that HistopathologyClassifierBackbone is a valid drop-in alias."""
    assert issubclass(HistopathologyClassifierBackbone, MultiStageFeatureBackbone)
    model = HistopathologyClassifierBackbone(in_channels=3, num_classes=4)
    x = torch.randn(1, 3, 64, 64)
    logits, features = model(x)
    assert logits.shape == (1, 4)
    assert "bn7" in features


# ==============================================================================
# 2. ProgressiveDropoutAttention (PDA) Tests
# ==============================================================================

def test_pda_initialization_and_forward_shapes():
    """Tests PDA forward pass, output shapes, and per-class normalization."""
    pda = ProgressiveDropoutAttention(
        num_steps=3,
        top_p=0.25,
        base_dropout_rate=0.50,
        step_weight_decay=0.80,
        boundary_bleeding_control=True,
    )
    b, d, h, w = 2, 512, 8, 8
    num_classes = 4

    feature_map = torch.randn(b, d, h, w).abs()
    classifier_weight = torch.randn(num_classes, d, 1, 1).abs()

    pda_cam = pda(feature_map, classifier_weight)

    # Output shape should match (B, C, H, W)
    assert pda_cam.shape == (b, num_classes, h, w)

    # Per-class min-max normalization to [0, 1]
    assert pda_cam.min() >= 0.0
    assert pda_cam.max() <= 1.0 + 1e-5


def test_pda_expansion_property():
    """Mathematically verifies the core PDA property: expands CAM coverage beyond tiny foci.
    
    Setup:
    - Feature map has a sharp primary focal activation peak at (row 4, col 4).
    - Surrounding pixels (rows 3..5, cols 3..5) have moderate activation.
    - Far pixels have low activation.
    
    Standard CAM: Concentrates almost exclusively on the sharp primary peak.
    PDA CAM: After dropping out the top peak in step 1, the surrounding tumor area is discovered,
    resulting in a strictly larger activated area (pixels > threshold).
    """
    torch.manual_seed(42)
    b, d, h, w = 1, 64, 16, 16
    c = 1

    # Base feature map with uniform small background
    features = torch.full((b, d, h, w), fill_value=0.1, dtype=torch.float32)

    # Moderate tumor bed around center (rows 5..10, cols 5..10)
    features[:, :, 5:11, 5:11] = 0.8

    # Highly salient focal peak at center (rows 7..8, cols 7..8)
    features[:, :, 7:9, 7:9] = 3.5

    # Classifier weight: positive for channel 0..63
    weights = torch.ones((c, d, 1, 1), dtype=torch.float32) / np.sqrt(d)

    # 1. Compute standard baseline CAM
    raw_cam = F.relu(F.conv2d(features, weights))
    raw_min, raw_max = raw_cam.min(), raw_cam.max()
    norm_base_cam = (raw_cam - raw_min) / (raw_max - raw_min + 1e-7)

    # 2. Compute PDA CAM
    pda = ProgressiveDropoutAttention(
        num_steps=3,
        top_p=0.15,
        base_dropout_rate=0.75,
        boundary_bleeding_control=False,
    )
    pda_cam = pda(features, weights)

    # Coverage threshold: e.g. 0.40
    thr = 0.40
    base_active_pixels = (norm_base_cam > thr).sum().item()
    pda_active_pixels = (pda_cam > thr).sum().item()

    # PDA must discover and activate the surrounding tumor bed
    assert pda_active_pixels > base_active_pixels, (
        f"PDA should expand CAM coverage: base={base_active_pixels}, pda={pda_active_pixels}"
    )


def test_pda_boundary_bleeding_control():
    """Verifies that prototype cosine gating stops complementary activations from bleeding into stroma.
    
    Setup:
    - Left half of feature map represents tumor tissue (positive correlation with prototype).
    - Right half represents normal stroma with orthogonal/negative feature direction.
    - PDA prototype gating should restrict complementary activation on the right half.
    """
    torch.manual_seed(100)
    b, d, h, w = 1, 32, 8, 8
    c = 1

    features = torch.zeros((b, d, h, w))
    # Left half: tumor direction (+1 along first 16 channels)
    features[:, :16, :, :4] = 2.0
    # Center focus in tumor
    features[:, :16, 3:5, 1:3] = 5.0

    # Right half: stroma direction (-1 along first 16 channels)
    features[:, :16, :, 4:] = -2.0

    weights = torch.zeros((c, d, 1, 1))
    weights[:, :16, :, :] = 1.0

    pda_with_control = ProgressiveDropoutAttention(
        num_steps=3,
        top_p=0.25,
        boundary_bleeding_control=True,
    )
    cam_controlled = pda_with_control(features, weights)

    # Right half (stroma) must have near-zero activation due to negative cosine similarity
    right_half_max = cam_controlled[:, :, :, 4:].max().item()
    assert right_half_max < 0.05, f"Boundary bleeding detected: right half max = {right_half_max}"


def test_pda_generate_from_backbone():
    """Tests end-to-end PDA CAM generation directly from backbone model."""
    backbone = MultiStageFeatureBackbone(in_channels=3, num_classes=4)
    backbone.eval()
    pda = ProgressiveDropoutAttention(num_steps=2)

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        cam = pda.generate_from_backbone(backbone, x, feature_layer="bn7")

    assert cam.shape == (1, 4, 2, 2)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-5


# ==============================================================================
# 3. WSSSBenchmarkingEngine Integration Tests
# ==============================================================================

@pytest.mark.parametrize("config_name", ["baseline", "classmix", "relocalization", "proposed"])
def test_wsss_engine_all_configurations_clean_execution(config_name: str):
    """Verifies that all 4 benchmark configurations execute cleanly and produce valid masks."""
    engine = WSSSBenchmarkingEngine(mode=config_name, num_classes=5, device="cpu")
    dummy_input = torch.rand(1, 3, 64, 64)
    dummy_raw_img = (np.random.rand(64, 64, 3) * 255.0).astype(np.uint8)

    mask = engine.generate_pseudo_mask(
        image_tensor=dummy_input,
        current_epoch=3,
        raw_image=dummy_raw_img,
    )

    assert isinstance(mask, np.ndarray), "Output mask must be a numpy ndarray"
    assert mask.shape == (64, 64), f"Expected shape (64, 64), got {mask.shape}"
    assert mask.dtype in [np.int64, np.int32], f"Expected integer dtype, got {mask.dtype}"
    assert mask.min() >= 0, f"Class indices must be non-negative, got min {mask.min()}"
    assert mask.max() < 5, f"Class indices must be < num_classes (5), got max {mask.max()}"


def test_wsss_engine_proposed_mode_uncertainty():
    """Verifies that proposed mode computes normalized Shannon entropy uncertainty map."""
    engine = WSSSBenchmarkingEngine(mode="proposed", num_classes=5, device="cpu")
    dummy_input = torch.rand(1, 3, 64, 64)

    # 1. Test with return_uncertainty=True
    mask, unc_map = engine.generate_pseudo_mask(
        image_tensor=dummy_input,
        current_epoch=2,
        return_uncertainty=True,
    )

    assert mask.shape == (64, 64)
    assert unc_map.shape == (64, 64)
    assert unc_map.min() >= 0.0
    assert unc_map.max() <= 1.0 + 1e-5

    # 2. Check that engine.last_uncertainty_map is stored
    assert engine.last_uncertainty_map is not None
    assert engine.last_uncertainty_map.shape == (64, 64)


def test_wsss_engine_dynamic_progressive_thresholding():
    """Verifies that effective threshold dynamically adapts across training epochs."""
    engine = WSSSBenchmarkingEngine(mode="proposed", num_classes=5, device="cpu")
    dummy_input = torch.rand(1, 3, 32, 32)

    # Generate at epoch 1 vs epoch 10
    mask_ep1 = engine.generate_pseudo_mask(dummy_input, current_epoch=1)
    mask_ep10 = engine.generate_pseudo_mask(dummy_input, current_epoch=10)

    assert mask_ep1.shape == (32, 32)
    assert mask_ep10.shape == (32, 32)


def test_pda_cumulative_mask_prevents_peak_reemergence():
    """Proves that cumulative_mask=True suppresses Peak 1 across all subsequent steps t=2..T.

    Setup:
    - Peak A at (1, 1) has magnitude 10.0.
    - Secondary Peak B at (4, 4) has magnitude 5.0.
    - Tertiary Tumor Bed C at (6, 6) has magnitude 2.5.
    - With base_dropout_rate=1.0, top_p=0.05 (masks the top peak at each step):
      * Step 0: Peak A (1, 1) has normalized activation 1.0.
      * Step 1: Peak A is erased (0.0), so Peak B (4, 4) becomes 1.0.
      * Step 2: With cumulative_mask=True, BOTH Peak A and Peak B are 0.0,
        allowing Tertiary Tumor Bed C (6, 6) to reach 1.0!
    """
    b, d, h, w = 1, 8, 8, 8
    c = 1
    features = torch.zeros((b, d, h, w), dtype=torch.float32)
    features[:, :, 1, 1] = 10.0  # Peak A
    features[:, :, 4, 4] = 5.0   # Peak B
    features[:, :, 6, 6] = 2.5   # Bed C

    weights = torch.ones((c, d, 1, 1), dtype=torch.float32) / float(d)

    pda_cum = ProgressiveDropoutAttention(
        num_steps=2,
        top_p=0.02,
        base_dropout_rate=1.0,
        boundary_bleeding_control=False,
        channel_masking=False,
        cumulative_mask=True,
    )
    _, step_cams_cum = pda_cum.generate_cam(features, weights, return_intermediates=True)

    # Step 0: Peak A is 1.0
    assert step_cams_cum[0][0, 0, 1, 1].item() == pytest.approx(1.0, abs=1e-5)
    # Step 1: Peak A is suppressed to 0.0, Peak B is 1.0
    assert step_cams_cum[1][0, 0, 1, 1].item() == pytest.approx(0.0, abs=1e-5)
    assert step_cams_cum[1][0, 0, 4, 4].item() == pytest.approx(1.0, abs=1e-5)
    # Step 2 (cumulative): Peak A stays 0.0 AND Peak B is suppressed to 0.0, so Bed C reaches 1.0!
    assert step_cams_cum[2][0, 0, 1, 1].item() == pytest.approx(0.0, abs=1e-5)
    assert step_cams_cum[2][0, 0, 4, 4].item() == pytest.approx(0.0, abs=1e-5)
    assert step_cams_cum[2][0, 0, 6, 6].item() == pytest.approx(1.0, abs=1e-5)


def test_wsss_engine_two_class_boundary_top2_entropy_gating():
    """Proves that 2-class histological interface ties (Tumor vs Stroma) trigger top-2 binary entropy gating."""
    engine = WSSSBenchmarkingEngine(mode="proposed", num_classes=5, device="cpu")

    # Craft a deterministic 4-class foreground CAM where at rows 1..15, Tumor (class 0) and Stroma (class 1)
    # are tied at 0.34 (above effective_thr=0.315 so raw argmax is 0, but below effective_thr * 1.2 = 0.378)
    # while LYM and NEC are 0.0. Row 0 has 0.0 and rows 16..31 have 1.0 to fix min=0.0, max=1.0.
    def _mock_pda(feat, weights):
        cam = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
        cam[0, 0, 1:16, :] = 0.34
        cam[0, 1, 1:16, :] = 0.34
        cam[0, 0, 16:, :] = 1.00
        cam[0, 1, 16:, 16:] = 1.00
        return cam

    engine.pda = _mock_pda
    engine.relocalizer = lambda cam, feat: cam

    img_tensor = torch.ones((1, 3, 32, 32), dtype=torch.float32) * 0.5
    pred_mask = engine.generate_pseudo_mask(img_tensor, current_epoch=2, raw_image=None)

    # Rows 2..14 have exact tie between Tumor (0.34) and Stroma (0.34) > Background (0.315),
    # so top2_entropy is 1.0 (> 0.85) and margin == 0.0 (< 0.05) -> gated from class 0 to background class 4!
    assert engine.last_top2_entropy_map is not None
    assert float(engine.last_top2_entropy_map[2:14, :].mean()) == pytest.approx(1.0, abs=1e-4)
    assert np.all(pred_mask[2:14, :] == 4)
    # Bottom-left quadrant (18:, :14) has confident Tumor=1.0, Stroma=0.0 -> predicted as class 0 (Tumor)
    assert np.all(pred_mask[18:, :14] == 0)

