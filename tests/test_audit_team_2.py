"""
Specialized Audit Suite 2: Stage-1 ResNet38d Net_CAM, Multi-Layer Fusion & Bilateral Affinity Propagation.

Focused, non-overlapping verification for Pillar 8 (Audit Team 2):
1. WSSSTissueStage1NetCAM multi-layer feature and CAM extraction (b4_5, b5_2, bn7) via MultiLayerCAMExtractor.
2. bilateral_affinity_propagation_gpu edge-preserving refinement and mathematical boundedness
   (convex combination guarantees refined_cams stays within [cams.min(), cams.max()]).
3. assemble_pseudo_mask Otsu background assignment and multi-layer CAM assembly.
4. Palette PNG round-trip fidelity via save_palette_mask (exact class index preservation).

Strictly adheres to Invariant 12 (standard ASCII hyphen '-' 0x2D only).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.training.dataset_loader import BCSS_PALETTE
from src.training.generate_pseudo_masks import (
    MultiLayerCAMExtractor,
    assemble_pseudo_mask,
    bilateral_affinity_propagation_gpu,
    save_palette_mask,
)


class TestStage1NetCAMArchitectureAndFusion:
    """Tests WSSSTissueStage1NetCAM forward feature pyramids and multi-layer CAM fusion."""

    def test_stage1_netcam_and_multilayer_cam_extractor(self):
        """Verifies WSSSTissueStage1NetCAM and MultiLayerCAMExtractor extract b4_5, b5_2, bn7 CAMs."""
        model = WSSSTissueStage1NetCAM(num_classes=4)
        model.eval()

        extractor = MultiLayerCAMExtractor(model=model, device=torch.device("cpu"))
        x = torch.randn(1, 3, 64, 64)
        label = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        cams = extractor.extract_cams(x, label)
        extractor.remove_hooks()

        assert set(cams.keys()) == {"bn7", "b5_2", "b4_5"}
        for layer_name, cam_tensor in cams.items():
            assert cam_tensor.shape == (1, 4, 64, 64), f"{layer_name} CAM shape mismatch: {cam_tensor.shape}"
            assert torch.all(torch.isfinite(cam_tensor))
            # Classes 2 and 3 have weak_label == 0 -> must be strictly 0.0 across all layers
            assert float(cam_tensor[:, 2].abs().max().item()) == 0.0
            assert float(cam_tensor[:, 3].abs().max().item()) == 0.0

    def test_bilateral_affinity_propagation_boundedness_and_edge_preservation(self):
        """Verifies bilateral_affinity_propagation_gpu preserves sharp colour boundaries and stays bounded."""
        h, w = 32, 32
        # Left half dark purple tumor [120, 40, 140], right half pale pink stroma [230, 180, 200]
        img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        img_rgb[:, :16] = [120, 40, 140]
        img_rgb[:, 16:] = [230, 180, 200]
        images_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

        cams = torch.zeros((1, 4, h, w), dtype=torch.float32)
        cams[0, 0, :, :16] = 0.90
        cams[0, 0, :, 16:] = 0.10
        cams[0, 1, :, 16:] = 0.85

        refined = bilateral_affinity_propagation_gpu(
            cams=cams,
            images=images_t,
            radius=2,
            sigma_color=0.10,
            sigma_pos=2.0,
            alpha=0.60,
            iterations=3,
        )

        assert refined.shape == (1, 4, h, w)
        # Row-normalized kernel guarantees output stays within [min(cams), max(cams)]
        assert float(refined.min().item()) >= float(cams.min().item()) - 1e-5
        assert float(refined.max().item()) <= float(cams.max().item()) + 1e-5
        # Sharp colour boundary between x=15 and x=16 prevents Tumor CAM from bleeding into Stroma
        assert float(refined[0, 0, :, :15].mean().item()) > 0.80
        assert float(refined[0, 0, :, 17:].mean().item()) < 0.20

    def test_assemble_pseudo_mask_weak_label_and_otsu_gating(self):
        """Verifies assemble_pseudo_mask gates white glass to class 4 while preserving tissue classes."""
        h, w = 32, 32
        img_rgb = np.full((1, h, w, 3), 255, dtype=np.uint8)  # Right half white glass
        img_rgb[0, :, :16] = [130, 50, 150]  # Left half dense tissue

        # Gated CAM where Class 0 (Tumor) is active on left half and Class 2 (LYM) is zeroed out
        cams = torch.zeros((1, 4, h, w), dtype=torch.float32)
        cams[0, 0, :, :16] = 0.85  # Tumor

        masks = assemble_pseudo_mask(
            cams=cams,
            raw_images_np=img_rgb,
            background_threshold=0.20,
        )
        mask = masks[0]

        assert mask.shape == (h, w)
        # Left tissue half must predict Class 0 (TUM)
        assert int(np.sum(mask[:, :16] == 0)) > 0
        # Right white glass half must be gated to Class 4 (BACK)
        assert np.all(mask[:, 16:] == 4)

    def test_palette_png_roundtrip_on_real_sample_patch(self, tmp_path: Path):
        """Verifies save_palette_mask lossless indexed PNG round-trip on real patch mask."""
        sample_files = sorted(Path("data/sample_dataset/train").glob("*.png"))
        assert len(sample_files) > 0
        sample_img = Image.open(sample_files[0]).convert("RGB")
        img_rgb = np.array(sample_img.resize((64, 64)), dtype=np.uint8)[np.newaxis, ...]

        cams = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
        cams[0, 0, :32, :] = 0.90
        cams[0, 1, 32:, :] = 0.85

        pred_mask = assemble_pseudo_mask(
            cams=cams,
            raw_images_np=img_rgb,
            background_threshold=0.20,
        )[0].astype(np.uint8)
        assert set(np.unique(pred_mask)).issubset({0, 1, 4})

        out_png = tmp_path / "pseudo_mask.png"
        save_palette_mask(pred_mask, out_png)
        assert out_png.is_file()

        reloaded_pil = Image.open(out_png)
        assert reloaded_pil.mode == "P"
        reloaded_arr = np.array(reloaded_pil, dtype=np.uint8)
        np.testing.assert_array_equal(reloaded_arr, pred_mask)
        assert len(BCSS_PALETTE) == 15
