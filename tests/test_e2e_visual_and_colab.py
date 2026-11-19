"""End-to-End (4-Tier) Opaque-Box Test Suite for Visual Verification, Clinical Workstation & Colab Readiness.

Covers:
- Tier 1 (Feature Coverage):
  1. Blurred & dimmed dual verification overlay (blur_sigma, dim_factor, gt_alpha, pred_alpha, tolerance_px).
  2. Unified miou_gt_present vs miou_penalized and pixel breakdown across clinical_viewer & colab_pkg.
  3. Entropy heatmap overlay & src.colab_pkg.render_visual_verification_diagnostic_figure(method_label=...).
  4. FastAPI /api/v1/samples (filters) & /api/v1/predict/patch (blurred_dual, cam, index maps, gt_biomarkers).
  5. FastAPI /api/v1/compare (m1..m6), /api/v1/predict/wsi, and /api/v1/export/report (png, json, csv).
  6. 5-class Stage-2 inference [:, :4] slicing in MultiMethodWSSSEngineV2 + DeepLabV3Plus(num_classes=5).
  7. Stage-1 [0, 1] unnormalization in Stage2Trainer.evaluate & train_proposed_sam.train_one_epoch.
  8. MedSAM ViT-B 1024x1024 resize contract in MedSAM_WSSS_Adapter.forward when is_surrogate=False.
- Tier 2 (Boundary & Corner Cases):
  - Blurred dual overlay boundary inputs (sigma=0, dim=0/1, alpha=0/1, tol=0, ignore_index=255, all-BACK).
  - Unified mIoU divergence on absent-GT false positives (>= 50 px) & degenerate all-255/single-class masks.
  - Custom upload on /api/v1/predict/patch without GT vs with GT (including ignore_index=255).
  - Invalid WSI mode and invalid export report format rejections (400/422).
  - CORS allow_credentials != true with wildcard origin, security headers, upload limit, zero __globals__[.
- Tier 3 (Cross-Feature Combinations & Enterprise Hygiene):
  - Zero SE374/UIT leaks in user-facing UI/API/visualizer/colab_pkg files, zero alert(, zero bad dashes/emojis.
  - Index PNGs (pred_mask_index_base64, gt_mask_index_base64, confidence_map_base64) vs agreement_summary & export.
  - Case 1 default sample ordering sanity & multi-method m1..m6 compare overlays.
  - HTML/JS workstation enterprise UI contracts (offline canvas fallback, zoom guard, sliders, spotlight, a11y).
  - Colab notebooks 01..04* & scripts/build_drive_package.py contracts (repo URL, ignore_index=255, ckpt keys, AST).
- Tier 4 (Real-World Clinical Workstation & Colab Pipeline Scenarios):
  - Scenario 1: Pathologist Patch QA + Spotlight + Report Export workflow.
  - Scenario 2: Multi-Method Comparative Audit workflow.
  - Scenario 3: WSI Multi-Tile Grid Stitching workflow.
  - Scenario 4: Real BCSS / sample_dataset Mode-P Lossless Mask + Blurred Dual Overlay smoke.
  - Scenario 5: End-to-End Colab Training-to-Visualization Contract Chain & Self-Governance Audit.
"""

import ast
import base64
import copy
import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import src.colab_pkg as colab_pkg
import src.colab_pkg.visualizer as colab_visualizer
import src.visualization.api_server as api_server
import src.visualization.clinical_viewer as clinical_viewer
from src.clinical.comparative_ablation_evaluator import ComparativeAblationEvaluator
from src.models.sam_wsss_adapter import MedSAM_WSSS_Adapter
from src.models.wsss_inference_methods import DeepLabV3Plus, MultiMethodWSSSEngineV2
from src.training.dataset_loader import BCSS_PALETTE, IMAGENET_MEAN, IMAGENET_STD
from src.training.train_proposed_sam import (
    CAMPromptGenerator,
    CompositeWSSSLoss,
)
from src.training.train_proposed_sam import (
    train_one_epoch as sam_train_one_epoch,
)
from src.training.train_stage2 import Stage2Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_DASHES: Tuple[str, ...] = ("\u2014", "\u2013", "\u2012", "\u2015", "\u2212")
FORBIDDEN_EMOJI_CHARS: Tuple[str, ...] = (
    "\u2744",
    "\u2728",
    "\u26a1",
    "\U0001f525",
    "\U0001f916",
    "\U0001f680",
    "\U0001f419",
    "\U0001f5c4",
)


_MODULE_SAMPLE_CATALOG_SNAPSHOT: Optional[List[Dict[str, Any]]] = None


# -----------------------------------------------------------------------------
# Deterministic CPU Helpers & Lightweight Model Stubs
# -----------------------------------------------------------------------------


def _decode_data_url_to_pil(data_url: str) -> Image.Image:
    assert isinstance(data_url, str) and data_url.startswith("data:image/")
    b64_part = data_url.split(",", 1)[1]
    raw = base64.b64decode(b64_part)
    return Image.open(io.BytesIO(raw))


def _encode_rgb_data_url(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_mask_p_data_url(mask: np.ndarray) -> str:
    pil_m = Image.fromarray(mask.astype(np.uint8), mode="P")
    pil_m.putpalette(BCSS_PALETTE)
    buf = io.BytesIO()
    pil_m.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_synthetic_he_and_gt(
    height: int = 64,
    width: int = 64,
    include_ignore: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds a deterministic H&E-stained tissue RGB patch + mode-P compatible GT mask."""
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    gt = np.full((height, width), 4, dtype=np.int64)

    q_h = height // 2
    q_w = width // 2

    # Top-left: TUM (0) dark purple nuclei
    rgb[:q_h, :q_w] = (145, 55, 130)
    gt[:q_h, :q_w] = 0

    # Bottom-left: STR (1) pink eosinophilic stroma
    rgb[q_h:, :q_w] = (205, 145, 180)
    gt[q_h:, :q_w] = 1

    # Top-center: LYM (2) dense blue-purple lymphocyte cluster
    rgb[:q_h, q_w : q_w + width // 4] = (90, 60, 155)
    gt[:q_h, q_w : q_w + width // 4] = 2

    # Bottom-center: NEC (3) pale eosinophilic necrosis
    rgb[q_h:, q_w : q_w + width // 4] = (210, 165, 150)
    gt[q_h:, q_w : q_w + width // 4] = 3

    # Right strip: BACK (4) pure white glass
    rgb[:, q_w + width // 4 :] = (250, 250, 250)
    gt[:, q_w + width // 4 :] = 4

    if include_ignore:
        gt[:2, :] = 255
        gt[-2:, :] = 255

    return rgb, gt


class _TinyDecoderStub(nn.Module):
    """Minimal DeepLab-like decoder exposing last_conv[8] for 256-dim feature hooks."""

    def __init__(self, out_channels: int = 5) -> None:
        super().__init__()
        self.last_conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=1),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_conv(x)


class _Tiny5ClassStage2Stub(nn.Module):
    """Fast deterministic 5-class Stage-2 stub outputting (B, 5, H, W) logits on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = _TinyDecoderStub(out_channels=5)
        with torch.no_grad():
            self.decoder.last_conv[0].weight.zero_()
            self.decoder.last_conv[0].bias.zero_()
            # Encode simple RGB color cues into first 4 feature channels
            self.decoder.last_conv[0].weight[0, 0, 0, 0] = 2.0
            self.decoder.last_conv[0].weight[1, 1, 0, 0] = 2.0
            self.decoder.last_conv[0].weight[2, 2, 0, 0] = 2.0
            self.decoder.last_conv[0].weight[3, 0, 0, 0] = 1.0
            self.decoder.last_conv[0].weight[3, 1, 0, 0] = 1.0

            self.decoder.last_conv[8].weight.zero_()
            self.decoder.last_conv[8].bias.zero_()
            for c_idx in range(4):
                self.decoder.last_conv[8].weight[c_idx, c_idx, 0, 0] = 2.5
            self.decoder.last_conv[8].bias[4] = -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class _Tiny4ClassStage1Stub(nn.Module):
    """Fast deterministic 4-class Stage-1 CAM stub accepting [0, 1] tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.n_class = 4
        self.num_classes = 4
        self.proj = nn.Conv2d(3, 4, kernel_size=1, bias=True)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[0, 0, 0, 0] = 1.2
            self.proj.weight[1, 1, 0, 0] = 1.2
            self.proj.weight[2, 2, 0, 0] = 1.2
            self.proj.weight[3, 0, 0, 0] = 0.8
            self.proj.bias.fill_(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cam, probs = self.forward_cam(x)
        return probs

    def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = F.avg_pool2d(x, kernel_size=4, stride=4) if min(x.shape[-2:]) >= 8 else x
        cam = F.relu(self.proj(feat))
        probs = torch.sigmoid(cam.mean(dim=(2, 3)))
        return cam, probs


@pytest.fixture()
def isolated_api_client(monkeypatch: pytest.MonkeyPatch):
    """Provides a fast isolated FastAPI TestClient with 5-class Stage-2 & 4-class Stage-1 stubs."""
    global _MODULE_SAMPLE_CATALOG_SNAPSHOT
    stub_s2 = _Tiny5ClassStage2Stub().eval()
    stub_s1 = _Tiny4ClassStage1Stub().eval()

    mgr = api_server.ClinicalWorkstationInferenceManager()
    mgr.model_stage2 = stub_s2  # type: ignore[assignment]
    mgr.model_stage1 = stub_s1  # type: ignore[assignment]
    mgr.stage2_loaded = True
    mgr.stage1_loaded = True
    mgr.sota_engine = MultiMethodWSSSEngineV2(model_stage1=stub_s1, use_amp=False)
    ev = ComparativeAblationEvaluator(feat_dim=256, device=torch.device("cpu"))
    ev.attach_classifier_head(stub_s2.decoder.last_conv[8])
    ev.prototypes_calibrated = True
    mgr._ablation_eval = ev
    mgr._initialized = True

    orig_cache = api_server._sample_catalog_cache
    monkeypatch.setattr(api_server, "_manager", mgr)
    try:
        if _MODULE_SAMPLE_CATALOG_SNAPSHOT is None:
            api_server._sample_catalog_cache = None
            _MODULE_SAMPLE_CATALOG_SNAPSHOT = copy.deepcopy(api_server._discover_clinical_samples())
        api_server._sample_catalog_cache = copy.deepcopy(_MODULE_SAMPLE_CATALOG_SNAPSHOT)
        yield TestClient(api_server.app)
    finally:
        api_server._sample_catalog_cache = orig_cache


# -----------------------------------------------------------------------------
# TIER 1: Feature Coverage (Happy Path)
# -----------------------------------------------------------------------------


def test_tier1_blurred_dual_verification_overlay_modes():
    """Verify blurred dual verification overlay blurs layers, dims H&E, and tints TP/Error/Margin cues."""
    fn = getattr(clinical_viewer, "compute_blurred_dual_verification_overlay", None)
    assert fn is not None, "Missing M2 function: clinical_viewer.compute_blurred_dual_verification_overlay"

    rgb, gt = _make_synthetic_he_and_gt(64, 64)
    # Create prediction: left TUM matches GT, bottom-left core STR misclassified as TUM, boundary perturbed
    pred = gt.copy()
    pred[40:58, 4:24] = 0  # Core misclassification inside STR -> Red
    pred[32:34, 0:32] = 1  # 2px boundary shift along horizontal seam -> Amber

    out_sharp = fn(
        rgb, pred, gt, blur_sigma=0.0, dim_factor=0.55, gt_alpha=0.45, pred_alpha=0.55, tolerance_px=2
    )
    out_blur = fn(
        rgb, pred, gt, blur_sigma=4.0, dim_factor=0.55, gt_alpha=0.45, pred_alpha=0.55, tolerance_px=2
    )
    out_dim = fn(
        rgb, pred, gt, blur_sigma=4.0, dim_factor=0.95, gt_alpha=0.25, pred_alpha=0.25, tolerance_px=2
    )

    out_no_err = fn(
        rgb, pred, gt, blur_sigma=4.0, dim_factor=0.55, gt_alpha=0.45, pred_alpha=0.55, error_alpha=0.0, tolerance_px=2
    )

    assert out_sharp.shape == (64, 64, 3) and out_sharp.dtype == np.uint8
    assert out_blur.shape == (64, 64, 3) and out_blur.dtype == np.uint8
    assert not np.array_equal(out_sharp, out_blur), "blur_sigma=0.0 vs 4.0 must produce distinct overlays"
    assert not np.array_equal(out_blur, out_dim), "dim_factor/alphas must alter the composite overlay"

    # Check TP green cue in exact TUM region (8:24, 8:24) vs no-error baseline and vs core-error region
    tp_patch = out_blur[8:24, 8:24].astype(np.float32)
    tp_no_err = out_no_err[8:24, 8:24].astype(np.float32)
    err_patch = out_blur[44:54, 8:20].astype(np.float32)
    assert float(tp_patch[:, :, 1].mean()) > float(tp_no_err[:, :, 1].mean()), (
        "TP match region must apply emerald green tint compared to error_alpha=0.0"
    )
    assert float(tp_patch[:, :, 1].mean()) > float(err_patch[:, :, 1].mean()), (
        "TP match region should have stronger green channel than core error region"
    )

    # Check Core error red cue in misclassified STR core (44:54, 8:20)
    assert float(err_patch[:, :, 0].mean()) > float(err_patch[:, :, 1].mean()), (
        "Core error region should exhibit red dominance"
    )

    # Check Amber cue on boundary margin row (32:34, 4:28): Red & Green both > Blue
    margin_patch = out_blur[32:34, 4:28].astype(np.float32)
    assert float(margin_patch[:, :, 0].mean()) > float(margin_patch[:, :, 2].mean())
    assert float(margin_patch[:, :, 1].mean()) > float(margin_patch[:, :, 2].mean())


def test_tier1_unified_miou_and_pixel_breakdown_across_viewer_and_colab(tmp_path: Path):
    """Verify unified miou_gt_present vs miou_penalized and pixel counts across clinical_viewer and colab_pkg."""
    rgb, gt = _make_synthetic_he_and_gt(224, 224)
    # Restrict GT strictly to TUM (0) + 4px LYM sliver (< 50 px) + BACK (4)
    gt_clean = np.full((224, 224), 4, dtype=np.int64)
    gt_clean[10:110, 10:110] = 0  # 10,000 px TUM
    gt_clean[115:116, 10:20] = 2  # 10 px sliver (< 50 px)

    # Prediction: 100% TUM match, misses 10px LYM sliver, plus 100px FP of STR (1) on glass
    pred = np.full((224, 224), 4, dtype=np.int64)
    pred[10:110, 10:110] = 0
    pred[130:140, 130:140] = 1  # 100 px FP on BACK

    _, summary = clinical_viewer.compute_agreement_error_overlay(rgb, pred, gt_clean)
    miou_gt = summary.get("miou_gt_present", summary.get("miou_4cls"))
    miou_pen = summary.get("miou_penalized_4cls", summary.get("miou_penalized"))

    assert miou_gt == pytest.approx(100.0, abs=0.2), f"Expected miou_gt_present=100.0, got {summary}"
    assert miou_pen == pytest.approx(50.0, abs=0.2), f"Expected miou_penalized=50.0, got {summary}"
    assert summary["tp_px"] == 10000
    assert summary["fp_glass_px"] == 100
    fn_key = "fn_missed_px" if "fn_missed_px" in summary else "fn_px"
    assert summary[fn_key] == 10
    assert summary["misclass_px"] == 0
    assert summary["valid_tissue_px"] == 10010

    # Verify colab_pkg.visualizer.render_clinical_comparison_figure reports unified miou_gt_present (100.0%)
    img_dir = tmp_path / "img"
    mask_dir = tmp_path / "mask"
    out_dir = tmp_path / "out"
    img_dir.mkdir()
    mask_dir.mkdir()

    patch_path = img_dir / "TCGA-E2-A14X_0_0_[1 0 1 0].png"
    # Paint H&E stain over FP region so pure_glass_guard keeps STR (1)
    rgb_stained = np.full((224, 224, 3), (160, 80, 145), dtype=np.uint8)
    Image.fromarray(rgb_stained, mode="RGB").save(patch_path)
    pil_gt = Image.fromarray(gt_clean.astype(np.uint8), mode="P")
    pil_gt.putpalette(BCSS_PALETTE)
    pil_gt.save(mask_dir / patch_path.name)

    class _EngineStub:
        def predict_all_methods(self, model, t_tensor, raw_rgb=None, model_stage1=None):
            return {
                "M1_Baseline_SingleScale": pred.astype(np.uint8),
                "M6_MultiScale_Pyramid_D4_Ensemble": pred.astype(np.uint8),
            }

        def apply_pure_glass_guard(self, pred_arr, raw_rgb, background_class_idx=4):
            return pred_arr

    dummy_model = nn.Conv2d(3, 5, 1)
    fig, saved_path = colab_visualizer.render_clinical_comparison_figure(
        active_test_files=[patch_path],
        test_mask_dir=mask_dir,
        model=dummy_model,
        wsss_engine=_EngineStub(),
        output_dir=out_dir,
        sample_count=1,
        show_plots=False,
    )
    assert fig is not None and saved_path is not None and saved_path.is_file()
    titles = " | ".join(ax.get_title() for ax in fig.axes)
    assert "100.0%" in titles, (
        f"colab_pkg.visualizer must use unified miou_gt_present (min_px=50 sliver guard); titles={titles}"
    )


def test_tier1_entropy_heatmap_and_colab_public_diagnostic_export(tmp_path: Path):
    """Verify entropy heatmap over dimmed grayscale and public colab_pkg diagnostic export with method_label."""
    rgb, gt = _make_synthetic_he_and_gt(64, 64)

    probs_uniform = np.full((4, 64, 64), 0.25, dtype=np.float32)
    probs_conf = np.full((64, 64, 4), 0.01, dtype=np.float32)
    probs_conf[:, :, 0] = 0.97

    ent_u_img, ent_u = clinical_viewer.compute_entropy_heatmap_overlay(rgb, probs_uniform, alpha=0.55)
    ent_c_img, ent_c = clinical_viewer.compute_entropy_heatmap_overlay(rgb, probs_conf, alpha=0.55)

    assert ent_u_img.shape == (64, 64, 3) and ent_u_img.dtype == np.uint8
    assert ent_c_img.shape == (64, 64, 3) and ent_c_img.dtype == np.uint8
    assert ent_u > 0.95
    assert ent_c < 0.25

    export_fn = getattr(colab_pkg, "render_visual_verification_diagnostic_figure", None)
    assert export_fn is not None, "Missing M1 export: src.colab_pkg.render_visual_verification_diagnostic_figure"

    out_png = tmp_path / "diagnostic_sheet.png"
    cases = [
        {
            "patient_id": "TCGA-TEST-0001",
            "tag": "Synthetic",
            "image": rgb,
            "gt": gt,
            "pred": gt.copy(),
            "probs": probs_conf.transpose(2, 0, 1),
        }
    ]
    fig, saved = export_fn(
        cases=cases,
        output_path=out_png,
        show_plots=False,
        method_label="AI Prediction (M7 Custom)",
    )
    assert fig is not None and saved is not None and saved.is_file()
    assert saved.stat().st_size > 1024
    all_titles = " | ".join(ax.get_title() for ax in fig.axes)
    assert "AI Prediction (M7 Custom)" in all_titles


def test_tier1_api_samples_and_predict_patch_visual_verification_contract(isolated_api_client: TestClient):
    """Verify /api/v1/samples filtering and /api/v1/predict/patch M2 visual verification & index outputs."""
    resp_all = isolated_api_client.get("/api/v1/samples")
    assert resp_all.status_code == 200
    all_data = resp_all.json()
    assert all_data["count"] >= 1
    first = all_data["samples"][0]

    # Filtered /api/v1/samples contract
    resp_filt = isolated_api_client.get(
        "/api/v1/samples",
        params={"patient_id": first["patient_id"], "limit": 1, "offset": 0},
    )
    assert resp_filt.status_code == 200
    filt_data = resp_filt.json()
    assert len(filt_data["samples"]) == 1
    assert filt_data["samples"][0]["patient_id"] == first["patient_id"]

    # Morphology filter sanity
    if first.get("present_classes"):
        morph = first["present_classes"][0]
        resp_m = isolated_api_client.get("/api/v1/samples", params={"morphology": morph, "limit": 2})
        assert resp_m.status_code == 200
        for item in resp_m.json()["samples"]:
            assert morph in item["present_classes"]

    # Predict patch with blurred dual parameters
    pred_resp = isolated_api_client.post(
        "/api/v1/predict/patch",
        json={
            "sample_id": first["sample_id"],
            "mode": "m6",
            "blur_sigma": 4.5,
            "dim_factor": 0.6,
            "gt_alpha": 0.4,
            "pred_alpha": 0.6,
            "tolerance_px": 2,
            "target_class": 0,
        },
    )
    assert pred_resp.status_code == 200
    data = pred_resp.json()

    h, w = data["mask_shape"]
    assert data.get("blurred_dual_overlay_base64"), "Expected blurred_dual_overlay_base64 in /predict/patch"
    assert _decode_data_url_to_pil(data["blurred_dual_overlay_base64"]).size == (w, h)

    assert data.get("cam_overlay_base64"), "Expected cam_overlay_base64 when stage1_loaded=True"
    assert _decode_data_url_to_pil(data["cam_overlay_base64"]).size == (w, h)

    assert data.get("pred_mask_index_base64"), "Expected pred_mask_index_base64"
    pred_idx_pil = _decode_data_url_to_pil(data["pred_mask_index_base64"])
    assert pred_idx_pil.mode == "L"
    pred_idx_arr = np.array(pred_idx_pil, dtype=np.int64)
    assert pred_idx_arr.shape == (h, w)
    assert set(np.unique(pred_idx_arr).tolist()).issubset({0, 1, 2, 3, 4})

    assert data.get("gt_mask_index_base64"), "Expected gt_mask_index_base64 for sample with GT"
    gt_idx_pil = _decode_data_url_to_pil(data["gt_mask_index_base64"])
    assert gt_idx_pil.mode == "L"

    assert data.get("confidence_map_base64"), "Expected confidence_map_base64"
    conf_pil = _decode_data_url_to_pil(data["confidence_map_base64"])
    assert conf_pil.mode == "L"
    assert conf_pil.size == (w, h)

    assert data.get("gt_distribution") is not None
    assert data.get("gt_biomarkers") is not None

    assert "agreement_summary" in data, (
        "Expected 'agreement_summary' in /api/v1/predict/patch response (PROJECT.md)"
    )
    summary = data["agreement_summary"]
    assert isinstance(summary, dict)
    for req_key in (
        "exact_match_pct",
        "boundary_margin_pct",
        "core_error_pct",
        "miou_4cls",
        "miou_penalized_4cls",
        "mdice_4cls",
        "boundary_f1",
        "per_class_iou",
        "top_confusions",
        "valid_tissue_px",
        "tp_px",
        "margin_err_px",
        "core_err_px",
        "fp_glass_px",
        "misclass_px",
    ):
        assert req_key in summary, f"Missing {req_key} in agreement_summary"
    assert ("fn_missed_px" in summary) or ("fn_px" in summary)


def test_tier1_api_compare_wsi_and_export_report(isolated_api_client: TestClient):
    """Verify /api/v1/compare (m1..m6), /api/v1/predict/wsi, and /api/v1/export/report (png/json/csv)."""
    rgb, gt = _make_synthetic_he_and_gt(64, 64)
    img_b64 = _encode_rgb_data_url(rgb)
    gt_b64 = _encode_mask_p_data_url(gt)

    # 1. Compare m1..m6
    cmp_resp = isolated_api_client.post(
        "/api/v1/compare",
        json={
            "image_base64": img_b64,
            "gt_mask_base64": gt_b64,
            "methods": ["m1", "m2", "m3", "m4", "m5", "m6"],
        },
    )
    assert cmp_resp.status_code == 200
    cmp_data = cmp_resp.json()
    assert len(cmp_data["comparisons"]) == 6
    for item in cmp_data["comparisons"]:
        for key in (
            "mask_base64",
            "agreement_overlay_base64",
            "blurred_dual_overlay_base64",
            "contour_overlay_base64",
            "entropy_overlay_base64",
        ):
            assert item.get(key), f"Missing {key} for method {item.get('requested_method')}"

    # 2. WSI multi-tile predict
    wsi_resp = isolated_api_client.post(
        "/api/v1/predict/wsi",
        json={
            "slide_id": "SLIDE-E2E-01",
            "mode": "m6",
            "tiles": [
                {"id": "t0", "x": 0, "y": 0, "image_base64": img_b64},
                {"id": "t1", "x": 48, "y": 0, "image_base64": img_b64},
            ],
        },
    )
    assert wsi_resp.status_code == 200
    wsi_data = wsi_resp.json()
    assert wsi_data["composite_shape"] == [64, 112]
    assert wsi_data.get("composite_mask_base64")
    assert wsi_data.get("composite_heatmap_base64"), "Expected composite_heatmap_base64 in /predict/wsi"
    assert any(k in wsi_data for k in ("region_summary", "regions", "tile_summaries")), (
        "Expected region summary in /predict/wsi"
    )

    # 3. Export report PNG / JSON / CSV
    png_resp = isolated_api_client.post(
        "/api/v1/export/report",
        json={"mode": "m6", "image_base64": img_b64, "gt_mask_base64": gt_b64, "format": "png"},
    )
    assert png_resp.status_code == 200
    assert "image/png" in png_resp.headers.get("content-type", "")
    assert png_resp.content.startswith(b"\x89PNG")
    assert Image.open(io.BytesIO(png_resp.content)).width >= 400

    json_resp = isolated_api_client.post(
        "/api/v1/export/report",
        json={"mode": "m6", "image_base64": img_b64, "gt_mask_base64": gt_b64, "format": "json"},
    )
    assert json_resp.status_code == 200
    parsed_json = json.loads(json_resp.text)
    assert "biomarkers" in parsed_json

    csv_resp = isolated_api_client.post(
        "/api/v1/export/report",
        json={"mode": "m6", "image_base64": img_b64, "gt_mask_base64": gt_b64, "format": "csv"},
    )
    assert csv_resp.status_code == 200
    rows = list(csv.reader(io.StringIO(csv_resp.text)))
    assert len(rows) >= 2


def test_tier1_five_class_stage2_inference_slice_no_cgm_crash():
    """Verify 5-class DeepLabV3Plus(num_classes=5) works in MultiMethodWSSSEngineV2 with CGM gating."""
    torch.manual_seed(42)
    model_s2 = DeepLabV3Plus(num_classes=5, output_stride=16).eval()
    model_s1 = _Tiny4ClassStage1Stub().eval()
    engine = MultiMethodWSSSEngineV2(model_stage1=model_s1, use_amp=False)

    rgb, _ = _make_synthetic_he_and_gt(64, 64)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    preds = engine.predict_all_methods_v2(model_s2, img_tensor=x, raw_rgb=rgb, model_stage1=model_s1)
    assert len(preds) == 7
    for key, mask in preds.items():
        assert mask.shape == (64, 64), f"Unexpected shape for {key}: {mask.shape}"
        assert set(np.unique(mask).tolist()).issubset({0, 1, 2, 3, 4})


def test_tier1_stage1_zero_one_unnormalization_contract():
    """Verify Stage2Trainer.evaluate and train_proposed_sam.train_one_epoch pass [0, 1] tensors to forward_cam."""
    observed_ranges: List[Tuple[float, float]] = []

    class _SpyStage1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_class = 4
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            observed_ranges.append((float(x.min().item()), float(x.max().item())))
            b, _, h, w = x.shape
            cam = torch.ones((b, 4, max(1, h // 8), max(1, w // 8)), device=x.device)
            probs = torch.full((b, 4), 0.75, device=x.device)
            return cam, probs

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    img_01 = torch.linspace(0.05, 0.95, 3 * 32 * 32).view(1, 3, 32, 32)
    img_norm = (img_01 - mean) / std
    assert float(img_norm.min()) < -1.0 and float(img_norm.max()) > 1.5

    # 1. Stage2Trainer.evaluate
    class _Stage2ValDS(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            return (
                img_norm.squeeze(0),
                torch.zeros((32, 32), dtype=torch.long),
                torch.ones((4,), dtype=torch.float32),
                "TCGA-XX-0001_0_0_[1 1 1 1].png",
            )

    s2_stub = _Tiny5ClassStage2Stub()
    trainer = Stage2Trainer(
        model=s2_stub,  # type: ignore[arg-type]
        optimizer=torch.optim.SGD(s2_stub.parameters(), lr=1e-3),
        stage1_model=_SpyStage1(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        enable_amp=False,
    )
    val_loader = DataLoader(_Stage2ValDS(), batch_size=1)
    trainer.evaluate(val_loader, use_cgm=True)

    assert len(observed_ranges) == 1
    s2_min, s2_max = observed_ranges[0]
    assert s2_min >= -1e-4 and s2_max <= 1.0 + 1e-4, (
        f"Stage2Trainer.evaluate passed unnormalized tensor [{s2_min:.3f}, {s2_max:.3f}] to stage1.forward_cam"
    )

    # 2. train_proposed_sam.train_one_epoch
    observed_ranges.clear()

    class _SamValDS(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            return {
                "image": img_norm.squeeze(0),
                "label": torch.zeros((32, 32), dtype=torch.long),
                "cls_label": torch.ones((4,), dtype=torch.float32),
            }

    class _MiniSamStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 1)

        def forward(self, images, point_coords=None, point_labels=None):
            logits = self.conv(images)
            return {
                "logits": logits,
                "probabilities": F.softmax(logits, dim=1),
                "iou_predictions": torch.ones((images.shape[0], 5), device=images.device) * 0.5,
            }

    sam_stub = _MiniSamStub()
    sam_loader = DataLoader(_SamValDS(), batch_size=1)
    sam_train_one_epoch(
        model=sam_stub,
        prompt_generator=CAMPromptGenerator(num_classes=4, max_pos_points=3, max_neg_points=3),
        dataloader=sam_loader,
        criterion=CompositeWSSSLoss(num_classes=5),
        optimizer=torch.optim.SGD(sam_stub.parameters(), lr=1e-3),
        scaler=None,
        device=torch.device("cpu"),
        stage1_model=_SpyStage1(),
        accum_steps=1,
        max_steps=1,
    )

    assert len(observed_ranges) == 1
    sam_min, sam_max = observed_ranges[0]
    assert sam_min >= -1e-4 and sam_max <= 1.0 + 1e-4, (
        f"train_proposed_sam.train_one_epoch passed [{sam_min:.3f}, {sam_max:.3f}] to stage1.forward_cam"
    )


def test_tier1_medsam_vit_b_1024_resize_contract():
    """Verify MedSAM_WSSS_Adapter resizes inputs to (1024, 1024) for image_encoder when is_surrogate=False."""
    observed_shapes: List[Tuple[int, ...]] = []

    class _ViTBShapeSpyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.freeze_marker = nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            observed_shapes.append(tuple(x.shape))
            assert x.shape[-2:] == (1024, 1024), f"ViT-B requires 1024x1024, got {tuple(x.shape)}"
            return torch.zeros((x.shape[0], 256, 64, 64), device=x.device, dtype=x.dtype)

    adapter = MedSAM_WSSS_Adapter(num_classes=5, use_surrogate_if_missing=True)
    adapter.image_encoder = _ViTBShapeSpyEncoder()
    adapter.is_surrogate = False

    x_224 = torch.rand((1, 3, 224, 224))
    out = adapter(x_224)

    assert len(observed_shapes) == 1
    assert observed_shapes[0] == (1, 3, 1024, 1024)
    assert out["logits"].shape == (1, 5, 224, 224)
    assert out["probabilities"].shape == (1, 5, 224, 224)


# -----------------------------------------------------------------------------
# TIER 2: Boundary & Corner Cases
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blur_sigma,dim_factor,gt_alpha,pred_alpha,tolerance_px",
    [
        (0.0, 0.0, 0.0, 0.0, 0),
        (0.0, 1.0, 1.0, 1.0, 0),
        (8.0, 0.0, 1.0, 0.0, 5),
        (12.0, 1.0, 0.0, 1.0, 1),
    ],
)
def test_tier2_blurred_dual_overlay_boundary_and_ignore_255_and_all_bg(
    blur_sigma: float,
    dim_factor: float,
    gt_alpha: float,
    pred_alpha: float,
    tolerance_px: int,
):
    """Verify blurred dual overlay boundary parameters, ignore_index=255, and all-BACK masks never fail."""
    fn = getattr(clinical_viewer, "compute_blurred_dual_verification_overlay", None)
    assert fn is not None, "Missing M2 function: compute_blurred_dual_verification_overlay"

    rgb, gt_ignore = _make_synthetic_he_and_gt(64, 64, include_ignore=True)
    pred = gt_ignore.copy()
    pred[gt_ignore == 255] = 0

    out_ign = fn(
        rgb,
        pred,
        gt_ignore,
        blur_sigma=blur_sigma,
        dim_factor=dim_factor,
        gt_alpha=gt_alpha,
        pred_alpha=pred_alpha,
        tolerance_px=tolerance_px,
    )
    assert out_ign.shape == (64, 64, 3) and out_ign.dtype == np.uint8

    all_bg = np.full((64, 64), 4, dtype=np.int64)
    out_bg = fn(
        rgb,
        all_bg,
        all_bg,
        blur_sigma=blur_sigma,
        dim_factor=dim_factor,
        gt_alpha=gt_alpha,
        pred_alpha=pred_alpha,
        tolerance_px=tolerance_px,
    )
    assert out_bg.shape == (64, 64, 3) and out_bg.dtype == np.uint8


def test_tier2_unified_miou_divergence_and_degenerate_masks():
    """Verify absent-GT >= 50px FP divergence and ZeroDivisionError safety on all-255 and single-class masks."""
    rgb = np.full((224, 224, 3), 180, dtype=np.uint8)
    gt = np.full((224, 224), 4, dtype=np.int64)
    gt[20:120, 20:120] = 0  # Only TUM present in foreground

    pred = gt.copy()
    pred[150:160, 150:160] = 2  # 100 px false-positive LYM (>= 50 px) on glass

    _, summary = clinical_viewer.compute_agreement_error_overlay(rgb, pred, gt)
    miou_gt = summary.get("miou_gt_present", summary.get("miou_4cls"))
    miou_pen = summary.get("miou_penalized_4cls", summary.get("miou_penalized"))
    assert miou_gt > miou_pen
    assert summary["fp_glass_px"] == 100

    # Degenerate masks: all-255 ignore, all-BACK (4), and all-TUM (0)
    for mask_val in (255, 4, 0):
        m = np.full((64, 64), mask_val, dtype=np.int64)
        out_img, deg_sum = clinical_viewer.compute_agreement_error_overlay(rgb[:64, :64], m, m)
        assert out_img.shape == (64, 64, 3)
        assert isinstance(deg_sum["exact_match_pct"], float)
        dist = clinical_viewer.compute_class_distribution(m)
        assert all(0.0 <= v <= 100.0 for v in dist.values())

    empty_dist = clinical_viewer.compute_class_distribution(np.zeros((0, 0), dtype=np.int64))
    assert all(v == 0.0 for v in empty_dist.values())


def test_tier2_custom_upload_without_gt_vs_with_gt_ignore_255(isolated_api_client: TestClient):
    """Verify /predict/patch custom upload behavior without GT vs with GT (including ignore_index=255)."""
    rgb, gt_ign = _make_synthetic_he_and_gt(64, 64, include_ignore=True)
    img_b64 = _encode_rgb_data_url(rgb)
    gt_b64 = _encode_mask_p_data_url(gt_ign)

    # Without GT
    resp_no_gt = isolated_api_client.post(
        "/api/v1/predict/patch",
        json={"image_base64": img_b64, "mode": "m6"},
    )
    assert resp_no_gt.status_code == 200
    d0 = resp_no_gt.json()
    assert d0["gt_mask_index_base64"] is None
    assert d0["gt_distribution"] is None
    assert d0["gt_biomarkers"] is None
    assert d0["blurred_dual_overlay_base64"] is None
    assert d0["pred_mask_index_base64"] is not None
    assert d0["confidence_map_base64"] is not None

    exp_no_gt = isolated_api_client.post(
        "/api/v1/export/report",
        json={"image_base64": img_b64, "gt_mask_base64": None, "mode": "m6", "format": "png"},
    )
    assert exp_no_gt.status_code == 200
    assert "image/png" in exp_no_gt.headers.get("content-type", "")
    assert exp_no_gt.content.startswith(b"\x89PNG")

    # With GT containing ignore=255 border
    resp_with_gt = isolated_api_client.post(
        "/api/v1/predict/patch",
        json={"image_base64": img_b64, "gt_mask_base64": gt_b64, "mode": "m6"},
    )
    assert resp_with_gt.status_code == 200
    d1 = resp_with_gt.json()
    assert d1["gt_mask_index_base64"] is not None
    assert d1["gt_distribution"] is not None
    assert d1["gt_biomarkers"] is not None
    assert d1["blurred_dual_overlay_base64"] is not None
    assert "agreement_summary" in d1 and d1["agreement_summary"] is not None
    summary = d1["agreement_summary"]
    # Ignore border (4 rows * 64 = 256 px) must not count in valid_tissue_px
    expected_valid_tissue = int(np.sum((gt_ign >= 0) & (gt_ign < 4)))
    assert summary["valid_tissue_px"] == expected_valid_tissue


def test_tier2_invalid_wsi_mode_and_export_format_rejection(isolated_api_client: TestClient):
    """Verify /predict/wsi rejects invalid mode/negative coords and /export/report rejects invalid format."""
    rgb, _ = _make_synthetic_he_and_gt(64, 64)
    img_b64 = _encode_rgb_data_url(rgb)

    wsi_bad = isolated_api_client.post(
        "/api/v1/predict/wsi",
        json={
            "slide_id": "SLIDE-BAD",
            "mode": "nonexistent_mode_xyz",
            "tiles": [{"id": "t0", "x": 0, "y": 0, "image_base64": img_b64}],
        },
    )
    assert wsi_bad.status_code in (400, 422), f"Expected 400/422 on invalid WSI mode, got {wsi_bad.status_code}"

    wsi_neg = isolated_api_client.post(
        "/api/v1/predict/wsi",
        json={
            "slide_id": "SLIDE-NEG",
            "mode": "m6",
            "tiles": [{"id": "t0", "x": -1, "y": 0, "image_base64": img_b64}],
        },
    )
    assert wsi_neg.status_code == 422, f"Expected 422 on negative WSI tile x=-1, got {wsi_neg.status_code}"

    exp_bad = isolated_api_client.post(
        "/api/v1/export/report",
        json={"mode": "m6", "image_base64": img_b64, "format": "xml"},
    )
    assert exp_bad.status_code in (400, 422), f"Expected 400/422 on invalid export format, got {exp_bad.status_code}"


def test_tier2_security_cors_headers_upload_limit_and_no_globals(isolated_api_client: TestClient):
    """Verify CORS allow_credentials=False, security headers, upload size limit, and zero __globals__[."""
    resp = isolated_api_client.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    acao = resp.headers.get("access-control-allow-origin", "")
    acac = resp.headers.get("access-control-allow-credentials", "").lower()
    if acao == "*":
        assert acac != "true", "Insecure CORS: allow_credentials must not be true when allow_origin is '*'"

    assert resp.headers.get("x-content-type-options", "").lower() == "nosniff"

    # Oversized upload guard (24 MB dummy payload)
    oversized_bytes = b"0" * (24 * 1024 * 1024)
    up_resp = isolated_api_client.post(
        "/api/v1/predict/patch",
        files={"file": ("huge.png", oversized_bytes, "image/png")},
    )
    assert up_resp.status_code in (400, 413), f"Expected 400/413 for oversized upload, got {up_resp.status_code}"

    api_src = (PROJECT_ROOT / "src" / "visualization" / "api_server.py").read_text(encoding="utf-8")
    assert "__globals__[" not in api_src, "Unsafe __globals__ reflection must be removed from api_server.py"


# -----------------------------------------------------------------------------
# TIER 3: Cross-Feature Combinations & Enterprise Hygiene
# -----------------------------------------------------------------------------


def test_tier3_zero_coursework_leaks_dashes_emojis_and_alerts_across_workstation_and_colab():
    """Verify user-facing UI, API, visualizer, and colab_pkg files have zero SE374/UIT leaks, alert(, dashes, emojis."""
    targets = [
        PROJECT_ROOT / "src" / "visualization" / "static" / "index.html",
        PROJECT_ROOT / "src" / "visualization" / "api_server.py",
        PROJECT_ROOT / "src" / "visualization" / "clinical_viewer.py",
        PROJECT_ROOT / "src" / "visualization" / "draw_pipeline_diagram.py",
        PROJECT_ROOT / "src" / "visualization" / "__init__.py",
        PROJECT_ROOT / "src" / "colab_pkg" / "__init__.py",
        PROJECT_ROOT / "src" / "colab_pkg" / "visualizer.py",
        PROJECT_ROOT / "src" / "colab_pkg" / "evaluator.py",
        PROJECT_ROOT / "src" / "colab_pkg" / "downloader.py",
        PROJECT_ROOT / "src" / "clinical" / "exhaustive_evaluator.py",
    ]
    leak_rgx = re.compile(
        r"SE374(?!_WSSS_Drive_Code|_WSSS_Project|_Colab_|_Stage|_MedSAM|_Exhaustive)|\bUIT\b"
    )

    violations: List[str] = []
    for path in targets:
        assert path.is_file(), f"Missing target file: {path}"
        text = path.read_text(encoding="utf-8")
        if leak_rgx.search(text):
            violations.append(f"Coursework leak in {path.relative_to(PROJECT_ROOT)}")
        for ch in FORBIDDEN_DASHES:
            if ch in text:
                violations.append(f"Non-ASCII dash U+{ord(ch):04X} in {path.relative_to(PROJECT_ROOT)}")
        for em in FORBIDDEN_EMOJI_CHARS:
            if em in text:
                violations.append(f"AI emoji U+{ord(em):04X} in {path.relative_to(PROJECT_ROOT)}")
        if path.name == "index.html" and "alert(" in text:
            violations.append("Blocking alert( found in index.html")

    assert not violations, "Hygiene violations:\n" + "\n".join(violations)


def test_tier3_predict_patch_index_maps_consistent_with_agreement_and_export(isolated_api_client: TestClient):
    """Verify decoded pred/gt index maps and confidence map match agreement_summary and export/report values."""
    rgb, gt = _make_synthetic_he_and_gt(64, 64)
    img_b64 = _encode_rgb_data_url(rgb)
    gt_b64 = _encode_mask_p_data_url(gt)

    resp = isolated_api_client.post(
        "/api/v1/predict/patch",
        json={"image_base64": img_b64, "gt_mask_base64": gt_b64, "mode": "m6"},
    )
    assert resp.status_code == 200
    data = resp.json()

    pred_arr = np.array(_decode_data_url_to_pil(data["pred_mask_index_base64"]), dtype=np.int64)
    gt_arr = np.array(_decode_data_url_to_pil(data["gt_mask_index_base64"]), dtype=np.int64)
    conf_arr = np.array(_decode_data_url_to_pil(data["confidence_map_base64"]), dtype=np.uint8)

    assert pred_arr.shape == gt_arr.shape == conf_arr.shape == (64, 64)
    valid = (gt_arr >= 0) & (gt_arr <= 4)
    tp_recomputed = int(np.sum((pred_arr == gt_arr) & (gt_arr < 4) & valid))
    fp_glass_recomputed = int(np.sum((gt_arr == 4) & (pred_arr < 4) & valid))
    fn_recomputed = int(np.sum((gt_arr < 4) & (pred_arr == 4) & valid))
    mis_recomputed = int(np.sum((gt_arr < 4) & (pred_arr < 4) & (pred_arr != gt_arr) & valid))

    assert "agreement_summary" in data and data["agreement_summary"] is not None
    summary = data["agreement_summary"]
    assert summary["tp_px"] == tp_recomputed
    assert summary["fp_glass_px"] == fp_glass_recomputed
    fn_key = "fn_missed_px" if "fn_missed_px" in summary else "fn_px"
    assert summary[fn_key] == fn_recomputed
    assert summary["misclass_px"] == mis_recomputed

    exp_json = isolated_api_client.post(
        "/api/v1/export/report",
        json={"mode": "m6", "image_base64": img_b64, "gt_mask_base64": gt_b64, "format": "json"},
    ).json()
    assert exp_json["biomarkers"] == data["biomarkers"]


def test_tier3_samples_case1_default_ordering_and_compare_m1_to_m6_overlays(isolated_api_client: TestClient):
    """Verify Case 1 default sample is tissue-rich and /compare returns complete m1..m6 overlays."""
    samples_resp = isolated_api_client.get("/api/v1/samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()["samples"]
    assert len(samples) >= 1
    case1 = samples[0]
    assert case1["gt_available"] is True

    pred_resp = isolated_api_client.post(
        "/api/v1/predict/patch",
        json={"sample_id": case1["sample_id"], "mode": "m6"},
    )
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    h, w = pred_data["mask_shape"]
    total_px = h * w
    assert "agreement_summary" in pred_data and pred_data["agreement_summary"] is not None
    summary = pred_data["agreement_summary"]
    assert summary["valid_tissue_px"] >= int(0.10 * total_px), (
        f"Case 1 default sample must have substantial tissue (>= 10%), got {summary['valid_tissue_px']}/{total_px}"
    )

    cmp_resp = isolated_api_client.post(
        "/api/v1/compare",
        json={"sample_id": case1["sample_id"], "methods": ["m1", "m2", "m3", "m4", "m5", "m6"]},
    )
    assert cmp_resp.status_code == 200
    cmp_items = cmp_resp.json()["comparisons"]
    assert [it["method_used"] for it in cmp_items] == ["m1", "m2", "m3", "m4", "m5", "m6"]


def test_tier3_html_workstation_enterprise_ui_contracts(isolated_api_client: TestClient):
    """Verify GET / and GET /viewer serve enterprise HTML/JS controls, offline fallback, a11y, and responsive CSS."""
    for route in ("/", "/viewer"):
        resp = isolated_api_client.get(route)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        html = resp.text

        assert "<canvas" in html or "createElement(\"canvas\")" in html or "createElement('canvas')" in html
        assert "zoomAtPoint" in html and "fitToViewport" in html
        assert "lastLoadedImageKey" in html
        assert "pixelInspector" in html
        assert "btnClearUpload" in html
        assert "blurred_dual" in html
        assert "blur_sigma" in html and "dim_factor" in html
        assert "gt_alpha" in html and "pred_alpha" in html and "tolerance_px" in html
        assert "pred_mask_index_base64" in html and "confidence_map_base64" in html
        for method_val in ('value="m2"', 'value="m3"', 'value="m4"'):
            assert method_val in html, f"Missing method option {method_val} in UI"
        assert "/api/v1/export/report" in html
        assert "/api/v1/predict/wsi" in html
        assert re.search(r"@media\s*\(\s*max-width\s*:\s*1180px\s*\)", html)
        assert "<button" in html and re.search(r'class=["\'][^"\']*legend-item', html)
        assert "keydown" in html, "Expected keyboard shortcuts handler"
        assert "alert(" not in html


def test_tier3_colab_notebooks_and_drive_packager_contracts():
    """Verify notebooks 01..04* and build_drive_package.py repo URLs, ignore_index=255, AST, and paths."""
    nb_paths = sorted(PROJECT_ROOT.glob("notebooks/0[1-4]*.ipynb"))
    assert len(nb_paths) >= 12, f"Expected 12 notebooks under notebooks/01..04*, found {len(nb_paths)}"

    for nb_p in nb_paths:
        raw_text = nb_p.read_text(encoding="utf-8")
        assert "medical-ai-consortium/bcss-wsss" not in raw_text, f"Broken URL remains in {nb_p.name}"
        assert "khoinguyen59/medical-wsss.git" in raw_text, f"Missing khoinguyen59/medical-wsss.git in {nb_p.name}"
        for ch in FORBIDDEN_DASHES:
            assert ch not in raw_text, f"Non-ASCII dash in {nb_p.name}"

        nb_json = json.loads(raw_text)
        code_src = "\n".join(
            "".join(c.get("source", []))
            for c in nb_json.get("cells", [])
            if c.get("cell_type") == "code"
        )
        # AST syntax check on every code cell (commenting Colab magics)
        for idx, cell in enumerate(nb_json.get("cells", [])):
            if cell.get("cell_type") == "code":
                cleaned = [
                    f"# {ln.strip()}" if ln.strip().startswith(("!", "%")) else ln
                    for ln in "".join(cell.get("source", [])).splitlines()
                ]
                ast.parse("\n".join(cleaned), filename=f"{nb_p.name}_cell_{idx}")

        if nb_p.name.startswith(("02_", "03_")):
            assert not re.search(r"np\.clip\s*\([^,]+,\s*0\s*,\s*4\s*\)", code_src), (
                f"{nb_p.name} must preserve ignore_index=255 (remove np.clip(..., 0, 4))"
            )
        if nb_p.name.startswith("03_"):
            assert "segment-anything" in code_src and "medsam_vit_b.pth" in code_src, (
                f"{nb_p.name} must wire official segment-anything and medsam_vit_b.pth"
            )
        if nb_p.name.startswith("04_"):
            assert "state_dict" in code_src and "model_state_dict" in code_src, (
                f"{nb_p.name} must handle 'state_dict' and 'model_state_dict' checkpoint keys"
            )
            assert (
                "ComparativeAblationEvaluator" in code_src
                or "render_visual_verification_diagnostic_figure" in code_src
            ), f"{nb_p.name} must wire ComparativeAblationEvaluator or render_visual_verification_diagnostic_figure"

    packager_src = (PROJECT_ROOT / "scripts" / "build_drive_package.py").read_text(encoding="utf-8")
    assert "medical-ai-consortium/bcss-wsss" not in packager_src, "Broken URL remains in build_drive_package.py"
    assert "SE374_WSSS_Drive_Code" in packager_src
    assert "SE374_WSSS_Drive_Code.zip" in packager_src
    assert "SE374_Colab_" in packager_src


# -----------------------------------------------------------------------------
# TIER 4: Real-World Clinical Workstation & Colab Pipeline Scenarios
# -----------------------------------------------------------------------------


def test_tier4_scenario1_pathologist_patch_qa_spotlight_and_export(isolated_api_client: TestClient):
    """Scenario 1: Health -> Samples -> Detail -> Predict Patch -> Spotlight Change -> Export PNG/JSON/CSV."""
    assert isolated_api_client.get("/api/v1/health").json()["status"] == "healthy"
    samples = isolated_api_client.get("/api/v1/samples").json()["samples"]
    sid = samples[0]["sample_id"]

    detail = isolated_api_client.get(f"/api/v1/samples/{sid}")
    assert detail.status_code == 200
    assert detail.json()["gt_mask_base64"] is not None

    r_tum = isolated_api_client.post(
        "/api/v1/predict/patch",
        json={"sample_id": sid, "mode": "m6", "target_class": 0, "blur_sigma": 3.0},
    )
    r_str = isolated_api_client.post(
        "/api/v1/predict/patch",
        json={"sample_id": sid, "mode": "m6", "target_class": 1, "blur_sigma": 6.0},
    )
    assert r_tum.status_code == 200 and r_str.status_code == 200
    assert r_tum.json()["spotlight_stats"]["class_name"] == "TUM"
    assert r_str.json()["spotlight_stats"]["class_name"] == "STR"
    assert r_tum.json()["blurred_dual_overlay_base64"] != r_str.json()["blurred_dual_overlay_base64"]

    for fmt in ("png", "json", "csv"):
        exp = isolated_api_client.post(
            "/api/v1/export/report",
            json={"sample_id": sid, "mode": "m6", "format": fmt},
        )
        assert exp.status_code == 200 and len(exp.content) > 64


def test_tier4_scenario2_multi_method_comparative_audit(isolated_api_client: TestClient):
    """Scenario 2: Filtered cohort selection -> Multi-method m1..m6 side-by-side audit."""
    samples = isolated_api_client.get("/api/v1/samples", params={"limit": 3, "offset": 0}).json()["samples"]
    assert len(samples) >= 1
    sid = samples[0]["sample_id"]

    cmp_resp = isolated_api_client.post(
        "/api/v1/compare",
        json={"sample_id": sid, "methods": ["m1", "m2", "m3", "m4", "m5", "m6"]},
    )
    assert cmp_resp.status_code == 200
    payload = cmp_resp.json()
    assert payload["gt_available"] is True
    assert len(payload["comparisons"]) == 6
    for entry in payload["comparisons"]:
        assert _decode_data_url_to_pil(entry["blurred_dual_overlay_base64"]).mode == "RGB"
        assert _decode_data_url_to_pil(entry["entropy_overlay_base64"]).mode == "RGB"
        entry_metrics = entry.get("agreement_summary") or entry.get("verification_metrics")
        assert entry_metrics is not None


def test_tier4_scenario3_wsi_multitile_grid_stitching(isolated_api_client: TestClient):
    """Scenario 3: 2x2 overlapping WSI tile grid stitching & composite heatmap generation."""
    rgb, _ = _make_synthetic_he_and_gt(64, 64)
    b64 = _encode_rgb_data_url(rgb)
    tiles = [
        {"id": "r0_c0", "x": 0, "y": 0, "image_base64": b64},
        {"id": "r0_c1", "x": 48, "y": 0, "image_base64": b64},
        {"id": "r1_c0", "x": 0, "y": 48, "image_base64": b64},
        {"id": "r1_c1", "x": 48, "y": 48, "image_base64": b64},
    ]
    resp = isolated_api_client.post(
        "/api/v1/predict/wsi",
        json={"slide_id": "TCGA-WSI-GRID", "mode": "m6", "tiles": tiles},
    )
    assert resp.status_code == 200
    wsi = resp.json()
    assert wsi["num_tiles"] == 4
    assert wsi["composite_shape"] == [112, 112]
    assert _decode_data_url_to_pil(wsi["composite_mask_base64"]).size == (112, 112)
    assert _decode_data_url_to_pil(wsi["composite_heatmap_base64"]).size == (112, 112)


def test_tier4_scenario4_real_bcss_and_sample_dataset_mode_p_blurred_dual_smoke():
    """Scenario 4: Lossless mode-P mask loading and blurred dual verification on real sample_dataset & BCSS."""
    fn = getattr(clinical_viewer, "compute_blurred_dual_verification_overlay", None)
    assert fn is not None

    dataset_pairs = [
        (
            PROJECT_ROOT / "data" / "sample_dataset" / "test" / "img",
            PROJECT_ROOT / "data" / "sample_dataset" / "test" / "mask",
            True,
        ),
        (
            PROJECT_ROOT / "data" / "BCSS-WSSS" / "test" / "img",
            PROJECT_ROOT / "data" / "BCSS-WSSS" / "test" / "mask",
            False,
        ),
    ]
    checked = 0
    for img_dir, mask_dir, required in dataset_pairs:
        if not (img_dir.is_dir() and mask_dir.is_dir()):
            if required:
                pytest.fail(f"Committed sample_dataset missing: {img_dir}")
            continue
        img_p = sorted(img_dir.glob("*.png"))[0]
        mask_p = mask_dir / img_p.name
        assert mask_p.is_file()

        rgb = np.array(Image.open(img_p).convert("RGB"), dtype=np.uint8)[:64, :64]
        m_img = Image.open(mask_p)
        if not required:
            assert m_img.mode == "P", f"Expected mode-P paletted mask at {mask_p}, got {m_img.mode}"
        else:
            assert m_img.mode in ("P", "L"), f"Expected mode-P or mode-L mask at {mask_p}, got {m_img.mode}"
        if m_img.mode != "P":
            m_img = m_img.convert("L")
        gt = np.array(m_img, dtype=np.int64)[:64, :64]
        assert set(np.unique(gt).tolist()).issubset({0, 1, 2, 3, 4, 255})

        pred = np.where(gt == 255, 4, gt)
        blurred = fn(rgb, pred, gt, blur_sigma=4.0, dim_factor=0.55)
        agree_img, summary = clinical_viewer.compute_agreement_error_overlay(rgb, pred, gt)
        assert blurred.shape == rgb.shape and agree_img.shape == rgb.shape
        assert summary["exact_match_pct"] == pytest.approx(100.0, abs=0.01)
        checked += 1

    assert checked >= 1


def test_tier4_scenario5_colab_training_to_visualization_contract_chain_and_self_hygiene(tmp_path: Path):
    """Scenario 5: End-to-end 5-class inference -> colab_pkg diagnostic export + self-governance hygiene check."""
    model_s2 = _Tiny5ClassStage2Stub().eval()
    model_s1 = _Tiny4ClassStage1Stub().eval()
    engine = MultiMethodWSSSEngineV2(model_stage1=model_s1, use_amp=False)

    rgb, gt = _make_synthetic_he_and_gt(64, 64)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    preds = engine.predict_all_methods_v2(model_s2, img_tensor=x, raw_rgb=rgb, model_stage1=model_s1)
    pred_m6 = preds["M6_MultiScale_Pyramid_D4_Ensemble"]

    probs_one_hot = np.zeros((4, 64, 64), dtype=np.float32)
    for c_idx in range(4):
        probs_one_hot[c_idx] = np.where(pred_m6 == c_idx, 0.85, 0.05)

    export_fn = getattr(colab_pkg, "render_visual_verification_diagnostic_figure", None)
    assert export_fn is not None
    out_png = tmp_path / "tier4_chain.png"
    fig, saved = export_fn(
        cases=[
            {
                "patient_id": "TCGA-CHAIN-01",
                "tag": "M6",
                "image": rgb,
                "gt": gt,
                "pred": pred_m6,
                "probs": probs_one_hot,
            }
        ],
        output_path=out_png,
        show_plots=False,
        method_label="AI Prediction (M6 Tier4)",
    )
    assert fig is not None and saved is not None and saved.is_file()

    # Self-governance check on this E2E test file
    self_text = Path(__file__).read_text(encoding="utf-8")
    for ch in FORBIDDEN_DASHES:
        assert ch not in self_text, f"Forbidden non-ASCII dash U+{ord(ch):04X} in {__file__}"
    for em in FORBIDDEN_EMOJI_CHARS:
        assert em not in self_text, f"Forbidden emoji U+{ord(em):04X} in {__file__}"
