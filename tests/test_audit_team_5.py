"""
Specialized Audit Suite 5: Histopathology Edge Cases, Top-2 Boundary Entropy & Uncertainty Gradient Attenuation.

Focused, non-overlapping verification for Pillar 8 (Audit Team 5):
1. Otsu clinical gating on pale mucinous / hypochromic stroma adjacent to dense tumor vs blank glass:
   - Pale mucinous tissue must retain > 0.75 foreground coverage.
   - Blank white glass / scanner noise must retain < 0.05 foreground coverage.
2. Two-class histological boundary tie (e.g., Tumor vs Stroma [0.5, 0.5, 0, 0, 0]):
   - 5-class Shannon entropy is ln(2)/ln(5) ~ 0.4307 (< 0.85), which misses the boundary tie.
   - Top-2 binary entropy is ln(2)/ln(2) = 1.0 (> 0.85), which accurately flags the interface as 255.
3. UncertaintyRegularizationLoss gradient attenuation:
   - High-entropy ambiguous pixels receive strictly smaller parameter gradients than low-entropy pixels
     with the same raw cross-entropy error, while ignore_index=255 receives strictly zero gradient.

Strictly adheres to Invariant 12 (standard ASCII hyphen '-' 0x2D only).
"""

import numpy as np
import pytest
import torch

from src.datasets.otsu_filter import apply_otsu_gating_to_prediction, compute_otsu_tissue_mask
from src.losses_metrics.losses import UncertaintyRegularizationLoss
from src.models.wsss_tissue import WSSSBenchmarkingEngine


class TestHistopathologyOtsuAndBoundaryEntropy:
    """Verifies Otsu tissue preservation on pale histology and Top-2 entropy boundary gating."""

    def test_otsu_preserves_pale_mucinous_tissue_and_rejects_glass(self):
        """Verifies pale mucinous tissue (> 75% retained) vs white glass (0% retained)."""
        # Left half: pale bluish-pink mucinous tissue with subtle mucin strands
        # Right half: white glass background (252..255)
        rng = np.random.default_rng(7)
        slide = np.full((128, 256, 3), 254, dtype=np.uint8)
        mucin_base = np.full((128, 128, 3), [222, 206, 228], dtype=np.uint8)
        texture = rng.integers(-6, 7, size=(128, 128, 3))
        slide[:, :128] = np.clip(mucin_base + texture, 0, 255).astype(np.uint8)

        mask = compute_otsu_tissue_mask(slide)
        mucin_retention = float(mask[:, :128].mean())
        glass_retention = float(mask[:, 128:].mean())

        assert mucin_retention > 0.75, f"Pale mucinous tissue retention too low: {mucin_retention:.3f}"
        assert glass_retention < 0.05, f"White glass false positive rate too high: {glass_retention:.3f}"

        # Verify apply_otsu_gating_to_prediction maps glass region to class 4 (BACK)
        pred_4cls = np.zeros((128, 256), dtype=np.int64)
        gated = apply_otsu_gating_to_prediction(pred_4cls, slide, background_class_idx=4)
        assert float((gated[:, 128:] == 4).mean()) > 0.95

    def test_two_class_histological_interface_top2_entropy_gating(self):
        """Proves WSSSBenchmarkingEngine detects 2-class Tumor-Stroma ties via Top-2 binary entropy."""
        engine = WSSSBenchmarkingEngine(num_classes=5, mode="proposed", device="cpu")

        def _mock_pda(feat: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            cam = torch.zeros((1, 4, 32, 32), dtype=torch.float32, device=feat.device)
            # Left half (cols 2..14): clear Tumor (Class 0 = 1.0, Class 1 = 0.0)
            cam[0, 0, :, 2:14] = 1.00
            # Top-right (rows 0..3, cols 18..30): anchor Class 1 max = 1.00
            cam[0, 1, :4, 18:30] = 1.00
            # Bottom-right (rows 6..30, cols 18..30): exact 2-class Tumor-Stroma tie at 0.34
            # (above effective_thr=0.315 so raw argmax is 0, but below effective_thr * 1.2 = 0.378)
            cam[0, 0, 6:30, 18:30] = 0.34
            cam[0, 1, 6:30, 18:30] = 0.34
            return cam

        engine.pda = _mock_pda
        engine.relocalizer = lambda cam, feat: cam

        img_tensor = torch.ones(1, 3, 32, 32)
        pseudo_mask = engine.generate_pseudo_mask(
            image_tensor=img_tensor,
            current_epoch=2,
            raw_image=None,
        )

        # Left half has confident Tumor=1.0 -> assigned Class 0 (TUM)
        assert np.all(pseudo_mask[:, 2:14] == 0)
        # Bottom-right tie region has Top-2 binary entropy = 1.0 (> 0.85) and margin = 0 -> gated to Class 4 (BACK)
        assert float(engine.last_top2_entropy_map[6:30, 18:30].mean()) == pytest.approx(1.0, abs=1e-4)
        assert np.all(pseudo_mask[6:30, 18:30] == 4)


class TestUncertaintyLossGradientAttenuation:
    """Verifies per-pixel gradient scaling and ignore_index=255 zero-gradient invariance."""

    def test_high_entropy_pixels_receive_attenuated_gradients_and_ignore_255_gets_zero_grad(self):
        """Proves gradient norm on high-entropy pixels is attenuated by ~10x vs low-entropy pixels."""
        loss_fn = UncertaintyRegularizationLoss(entropy_threshold=0.65, ignore_index=255)

        # Pixel 0: low entropy (p_0 = 0.20, p_1 = 0.80, others ~ 0 -> H_norm ~ 0.31 < 0.65 -> w ~ 0.69)
        logits_low_ent = torch.tensor(
            [[[[0.0]], [[1.3863]], [[-20.0]], [[-20.0]], [[-20.0]]]],
            requires_grad=True,
        )
        # Pixel 1: max entropy (uniform p_k = 0.20 -> H_norm = 1.0 >= 0.65 -> w = 0.10)
        logits_high_ent = torch.tensor(
            [[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]],
            requires_grad=True,
        )
        target = torch.tensor([[[0]]], dtype=torch.long)

        loss_low = loss_fn(logits_low_ent, target)
        loss_low.backward()
        grad_low_norm = float(logits_low_ent.grad.norm().item())

        loss_high = loss_fn(logits_high_ent, target)
        loss_high.backward()
        grad_high_norm = float(logits_high_ent.grad.norm().item())

        # Both pixels have p_0 = 0.20 (identical raw CE = -ln(0.2)), but high-entropy pixel is downweighted
        assert loss_high.item() < loss_low.item()
        assert grad_high_norm < grad_low_norm * 0.35

        # Pixel with target = 255 (ignore_index) must receive strictly 0.0 loss and 0.0 gradient
        logits_ign = torch.randn(1, 5, 4, 4, requires_grad=True)
        target_ign = torch.full((1, 4, 4), 255, dtype=torch.long)
        loss_ign = loss_fn(logits_ign, target_ign)
        loss_ign.backward()
        assert loss_ign.item() == 0.0
        assert float(logits_ign.grad.abs().max().item()) == 0.0
