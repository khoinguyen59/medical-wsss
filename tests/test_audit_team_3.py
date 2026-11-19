"""
Specialized Audit Suite 3: Stage-2 DeepLabV3+ Architectures & MultiMethodWSSSEngineV2 (M1-M7).

Focused, non-overlapping verification for Pillar 8 (Audit Team 3):
1. Stage-2 DeepLabV3Plus (src/colab_pkg/model_loader.py) and DeepLabV3PlusWithCGM (src/models/deeplabv3plus.py)
   forward passes, state-dict round-trip compatibility, and B=1 stability.
2. Classification Gate Mechanism (CGM) hard (-1e4 logit penalty) and soft (P * gate) gating of
   absent classes (y_c = 0).
3. MultiMethodWSSSEngineV2 end-to-end execution across all 7 methods (M1..M7) on real histopathology
   patches from data/sample_dataset/.

Strictly adheres to Invariant 12 (standard ASCII hyphen '-' 0x2D only).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.colab_pkg.model_loader import (
    DeepLabV3Plus,
    MultiMethodWSSSEngineV2,
    WSSSTissueStage1NetCAM,
)
from src.models.deeplabv3plus import ClassificationGateMechanism, DeepLabV3PlusWithCGM


class TestStage2DeepLabV3PlusAndCGM:
    """Verifies Stage-2 DeepLabV3Plus and DeepLabV3PlusWithCGM architectures and gating."""

    def test_stage2_deeplabv3plus_forward_and_state_dict_roundtrip(self):
        """Verifies colab_pkg DeepLabV3Plus forward pass and strict state_dict reload."""
        model = DeepLabV3Plus(num_classes=4, output_stride=16)
        model.eval()

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 64, 64)
        assert torch.all(torch.isfinite(out))

        # Verify state_dict round-trip
        state = model.state_dict()
        model_reloaded = DeepLabV3Plus(num_classes=4, output_stride=16)
        missing, unexpected = model_reloaded.load_state_dict(state, strict=True)
        assert len(missing) == 0 and len(unexpected) == 0

    def test_cgm_hard_and_soft_gating_suppress_absent_classes(self):
        """Verifies both hard CGM (-1e4 logit penalty) and soft CGM (prob * gate) suppress absent classes."""
        # 1. Hard CGM in DeepLabV3PlusWithCGM
        cgm = ClassificationGateMechanism(penalty_value=-1e4, threshold=0.5, default_background_idx=4)
        logits = torch.ones(2, 5, 16, 16) * 3.0
        presence = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])

        gated = cgm(logits, class_presence=presence)
        probs = F.softmax(gated, dim=1)
        # Sample 0: classes 1, 2, 3 are absent -> softmax probability < 1e-10
        assert float(probs[0, 1:4].max().item()) < 1e-10
        # Sample 1: all classes 0 -> fallback protects background class 4
        assert torch.all(gated[1].argmax(dim=0) == 4)

        # 2. Soft CGM in MultiMethodWSSSEngineV2.apply_cgm_gate
        probs_4cls = np.full((4, 16, 16), 0.25, dtype=np.float32)
        cgm_gate = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)
        gated_probs = MultiMethodWSSSEngineV2.apply_cgm_gate(probs_4cls, cgm_gate)
        assert float(gated_probs[1].max()) == 0.0
        assert float(gated_probs[2].max()) == 0.0
        assert np.allclose(gated_probs.sum(axis=0), 1.0, atol=1e-5)

        # 3. Training-mode B=1 gradient check on DeepLabV3PlusWithCGM
        seg_model = DeepLabV3PlusWithCGM(num_classes=5, in_channels=3)
        seg_model.train()
        img_b1 = torch.randn(1, 3, 64, 64, requires_grad=True)
        target_b1 = torch.randint(0, 5, (1, 64, 64))
        out_logits, loss = seg_model(img_b1, class_presence=presence[:1], targets=target_b1, return_loss=True)
        assert out_logits.shape == (1, 5, 64, 64)
        loss.backward()
        assert img_b1.grad is not None and torch.all(torch.isfinite(img_b1.grad))


class TestMultiMethodWSSSEngineV2RealPatches:
    """Verifies MultiMethodWSSSEngineV2 across all 7 benchmark methods (M1..M7) on real patches."""

    def test_all_seven_methods_m1_to_m7_on_real_bcss_patch(self):
        """Runs M1..M7 on a real BCSS sample patch and verifies 4-class and 5-class predictions."""
        from pathlib import Path

        sample_files = sorted(Path("data/sample_dataset/valid/img").glob("*.png"))
        assert len(sample_files) > 0
        img_rgb = np.array(Image.open(sample_files[0]).convert("RGB").resize((64, 64)), dtype=np.uint8)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

        stage2_model = DeepLabV3Plus(num_classes=4, output_stride=16).eval()
        stage1_model = WSSSTissueStage1NetCAM(n_class=4).eval()

        engine = MultiMethodWSSSEngineV2(include_v2=True)

        preds = engine.predict_all_methods_v2(
            model=stage2_model,
            img_tensor=img_tensor,
            raw_rgb=img_rgb,
            model_stage1=stage1_model,
        )
        expected_keys = {
            "M1_Baseline_SingleScale",
            "M2_MultiView_D4_TTA",
            "M3_LongTail_Prior_Calibration",
            "M4_Spatial_Bilateral_Refinement",
            "M5_Proposed_SOTA_Ensemble",
            "M6_MultiScale_Pyramid_D4_Ensemble",
            "M7_Proposed_Ultra_SOTA_Curriculum",
        }
        assert set(preds.keys()) == expected_keys
        for m_name, mask in preds.items():
            assert mask.shape == (64, 64), f"{m_name} shape mismatch: {mask.shape}"
            assert mask.dtype == np.uint8
            assert set(np.unique(mask)).issubset({0, 1, 2, 3, 4}), f"{m_name} invalid labels: {np.unique(mask)}"

        # Verify pure glass slide is mapped to class 4 (Background) by M5, M6, M7 glass guard
        glass_rgb = np.full((64, 64, 3), 255, dtype=np.uint8)
        glass_tensor = torch.ones((1, 3, 64, 64), dtype=torch.float32)
        glass_preds = engine.predict_all_methods_v2(
            model=stage2_model,
            img_tensor=glass_tensor,
            raw_rgb=glass_rgb,
            model_stage1=stage1_model,
        )
        assert np.all(glass_preds["M5_Proposed_SOTA_Ensemble"] == 4)
        assert np.all(glass_preds["M6_MultiScale_Pyramid_D4_Ensemble"] == 4)
        assert np.all(glass_preds["M7_Proposed_Ultra_SOTA_Curriculum"] == 4)
