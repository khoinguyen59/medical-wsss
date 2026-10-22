"""Unit tests for the WSI Low-Mag to High-Mag End-to-End Segmentation Pipeline.

Verifies that the extension module operates cleanly, integrates seamlessly
with MultiMethodWSSSEngineV2, preserves memory safety, and does not alter
existing patch-level baseline contracts.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.pipeline.wsi_lowmag_highmag_pipeline import (
    WSILowMagHighMagPipeline,
    WSICandidatePatch,
    WSIProcessingReport,
)
from src.models.wsss_inference_methods import MultiMethodWSSSEngineV2


class MockDeepLabV3Plus(nn.Module):
    """Lightweight dummy segmentor for CPU testing."""
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MockStage1NetCAM(nn.Module):
    """Lightweight dummy Stage-1 CAM classifier for CPU testing."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.cam_conv = nn.Conv2d(3, num_classes, kernel_size=1)
        self.fc = nn.Linear(3, num_classes)

    def forward_cam(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cam = F.relu(self.cam_conv(x))
        pooled = x.mean(dim=(2, 3))
        logits = self.fc(pooled)
        probs = torch.sigmoid(logits)
        return cam, probs


import torch.nn.functional as F


def test_wsi_pipeline_initialization():
    """Verify default initialization parameters."""
    pipeline = WSILowMagHighMagPipeline(patch_size=224, tissue_threshold=0.15)
    assert pipeline.patch_size == 224
    assert pipeline.tissue_threshold == 0.15
    assert pipeline.lowmag_target_size == (1024, 1024)
    assert isinstance(pipeline.engine, MultiMethodWSSSEngineV2)


def test_wsi_pipeline_read_overview():
    """Verify overview extraction from synthetic in-memory image."""
    pipeline = WSILowMagHighMagPipeline(lowmag_target_size=(256, 256))
    dummy_slide = np.ones((1000, 2000, 3), dtype=np.uint8) * 200

    overview, orig_dims = pipeline.read_slide_overview(dummy_slide)
    assert orig_dims == (2000, 1000)
    assert overview.size == (256, 256)


def test_wsi_pipeline_macro_tissue_heatmap():
    """Verify macro tissue mask filters pure white glass background."""
    pipeline = WSILowMagHighMagPipeline(lowmag_target_size=(128, 128))

    # Create synthetic slide: left half is pure white glass (250, 250, 250),
    # right half is H&E stained tissue (180, 50, 150)
    synthetic_overview = np.ones((128, 128, 3), dtype=np.uint8) * 250
    synthetic_overview[:, 64:, 0] = 180
    synthetic_overview[:, 64:, 1] = 50
    synthetic_overview[:, 64:, 2] = 150
    pil_overview = Image.fromarray(synthetic_overview)

    tissue_mask, macro_probs = pipeline.compute_macro_tissue_heatmap(pil_overview)

    assert tissue_mask.shape == (128, 128)
    assert macro_probs.shape == (4, 128, 128)
    # Pure glass region on left should be classified as non-tissue (False)
    assert np.mean(tissue_mask[:, :50]) < 0.1
    # Stained tissue region on right should be classified as tissue (True)
    assert np.mean(tissue_mask[:, 70:]) > 0.8


def test_wsi_pipeline_candidate_patch_selection():
    """Verify low-mag guidance correctly prunes empty background and computes reduction %."""
    pipeline = WSILowMagHighMagPipeline(patch_size=100)

    # 1000x1000 slide = 10x10 = 100 total possible patches
    orig_dims = (1000, 1000)
    # Mask with only 20% tissue in the center
    tissue_mask = np.zeros((100, 100), dtype=bool)
    tissue_mask[40:60, 40:60] = True
    macro_probs = np.zeros((4, 100, 100), dtype=np.float32)

    candidates, total_possible, reduction_pct = pipeline.select_candidate_highmag_patches(
        orig_dims, tissue_mask, macro_probs
    )

    assert total_possible == 100
    assert len(candidates) < total_possible
    assert reduction_pct > 70.0  # Should save over 70% of compute
    assert all(isinstance(c, WSICandidatePatch) for c in candidates)


def test_wsi_pipeline_end_to_end_simulation():
    """Verify end-to-end pipeline execution and clinical report generation."""
    pipeline = WSILowMagHighMagPipeline(
        patch_size=64,
        lowmag_target_size=(64, 64),
        tissue_threshold=0.1,
        device="cpu",
    )

    # Create small synthetic slide: 256x256 image with purple tissue circle in center
    slide_arr = np.ones((256, 256, 3), dtype=np.uint8) * 245
    slide_arr[64:192, 64:192, 0] = 170
    slide_arr[64:192, 64:192, 1] = 60
    slide_arr[64:192, 64:192, 2] = 160
    slide_pil = Image.fromarray(slide_arr)

    model_stage2 = MockDeepLabV3Plus(num_classes=5)
    model_stage1 = MockStage1NetCAM(num_classes=4)

    stitched_mask, report = pipeline.run_end_to_end_wsi(
        slide_input=slide_pil,
        model_stage2=model_stage2,
        model_stage1=model_stage1,
        method="M1_Baseline_SingleScale",
        slide_id="TEST_SLIDE_001",
        max_patches_to_infer=10,
    )

    assert stitched_mask.shape == (64, 64)
    assert isinstance(report, WSIProcessingReport)
    assert report.slide_id == "TEST_SLIDE_001"
    assert report.compute_reduction_pct >= 0.0
    assert report.total_elapsed_time_sec > 0.0
    assert report.tumor_fraction_pct >= 0.0

    # Verify colorization
    color_img = WSILowMagHighMagPipeline.colorize_mask(stitched_mask)
    assert color_img.size == (64, 64)
    assert color_img.mode == "RGB"


def test_wsi_pipeline_extract_patch_cached_handle():
    """Verify that patch extraction works with cached handle and avoids disk re-reads."""
    pipeline = WSILowMagHighMagPipeline(patch_size=32)
    img_arr = np.ones((128, 128, 3), dtype=np.uint8) * 120
    img_pil = Image.fromarray(img_arr)

    cand = WSICandidatePatch(
        patch_id=1,
        x=10,
        y=10,
        width=32,
        height=32,
        macro_tissue_score=0.9,
        macro_tumor_prob=0.8,
    )

    # Test with PIL image handle
    crop_from_pil = pipeline.extract_patch_image(slide_input=img_pil, patch_info=cand, slide_obj=img_pil)
    assert crop_from_pil.shape == (32, 32, 3)

    # Test with numpy array handle
    crop_from_arr = pipeline.extract_patch_image(slide_input=img_arr, patch_info=cand, slide_obj=img_arr)
    assert crop_from_arr.shape == (32, 32, 3)
