"""
Unit Tests for FastAPI Clinical Web Workstation & Inference API (SE374.R11).
Verifies:
- GET /api/v1/health returns system status, model state, and memory/VRAM usage.
- POST /api/v1/predict/patch accepts RGB image (multipart & JSON base64),
  computes WSSS segmentation, class distributions, VTP, and TSR.
- Clinical Otsu gating extinguishes background false positives on negative control slides.
- POST /api/v1/predict/wsi combines multi-patch tiles and returns composite heatmap.
- GET / and GET /viewer serve interactive OpenSeadragon clinical workstation HTML.
- Invariant 12 compliance: 100% standard ASCII hyphens (-), zero em-dashes/en-dashes.
"""

import base64
import csv
import io
import json
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image

import src.visualization.api_server as api_mod
from src.visualization.api_server import app
from src.visualization.clinical_viewer import (
    CLASS_COLORS_HEX,
    blend_overlay,
    compute_agreement_error_overlay,
    compute_blurred_dual_verification_overlay,
    compute_cam_overlay,
    compute_class_distribution,
    compute_contour_comparison_overlay,
    compute_dual_mask_blend,
    compute_entropy_heatmap_overlay,
    compute_patch_iou_metrics,
    compute_per_class_spotlight,
    decode_base64_to_image,
    encode_image_to_base64,
    generate_dzi_manifest,
    mask_to_color_image,
    mask_to_rgba_overlay,
    render_single_patch_diagnostic_sheet,
    stitch_tiles_heatmap,
)

client = TestClient(app)


def create_synthetic_patch_bytes(
    height: int = 224,
    width: int = 224,
    color: tuple = (210, 160, 190),
) -> bytes:
    """Helper creating raw PNG bytes for a synthetic H&E patch."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_api_health_endpoint():
    """Tests GET /api/v1/health returns status healthy and expected metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "model" in data
    assert data["model"]["num_classes"] == 5
    assert data["model"]["classes"] == ["TUM", "STR", "LYM", "NEC", "BACK"]
    assert data["model"]["otsu_gating"] is True
    assert "system" in data
    assert "torch_version" in data["system"]
    assert "cuda_available" in data["system"]
    assert isinstance(data["system"]["vram_allocated_mb"], (int, float))


def test_predict_patch_json_base64():
    """Tests POST /api/v1/predict/patch with JSON base64 encoded patch."""
    patch_bytes = create_synthetic_patch_bytes(224, 224, color=(180, 50, 120))
    raw_np = np.array(Image.open(io.BytesIO(patch_bytes)), dtype=np.uint8)
    b64_str = encode_image_to_base64(raw_np, format="PNG")

    payload = {
        "image_base64": b64_str,
        "mode": "proposed",
        "apply_otsu_gating": True,
    }
    response = client.post("/api/v1/predict/patch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["mask_shape"] == [224, 224]
    assert data["applied_gating"] is True

    # Validate class distribution
    dist = data["class_distribution"]
    for name in ["TUM", "STR", "LYM", "NEC", "BACK"]:
        assert name in dist
        assert 0.0 <= dist[name] <= 100.0

    total_pct = sum(dist.values())
    assert total_pct == pytest.approx(100.0, abs=1.0)

    # Validate clinical biomarkers
    biomarkers = data["biomarkers"]
    assert "viable_tumor_percentage" in biomarkers
    assert "tumor_stroma_ratio" in biomarkers
    assert 0.0 <= biomarkers["viable_tumor_percentage"] <= 100.0
    assert 0.0 <= biomarkers["tumor_stroma_ratio"] <= 100.0

    # Validate returned base64 image strings
    assert data["mask_base64"].startswith("data:image/png;base64,")
    assert data["overlay_base64"].startswith("data:image/png;base64,")


def test_predict_patch_multipart_file():
    """Tests POST /api/v1/predict/patch with multipart/form-data upload."""
    patch_bytes = create_synthetic_patch_bytes(224, 224, color=(160, 90, 150))
    files = {"file": ("test_patch.png", patch_bytes, "image/png")}

    response = client.post("/api/v1/predict/patch", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["mask_shape"] == [224, 224]
    assert "biomarkers" in data
    assert "class_distribution" in data


def test_predict_patch_otsu_gating_blank_glass():
    """Tests that Otsu clinical gating on blank glass slide forces pixels to background class."""
    blank_glass_bytes = create_synthetic_patch_bytes(224, 224, color=(250, 250, 250))
    blank_np = np.array(Image.open(io.BytesIO(blank_glass_bytes)), dtype=np.uint8)
    b64_str = encode_image_to_base64(blank_np, format="PNG")

    payload = {
        "image_base64": b64_str,
        "mode": "proposed",
        "apply_otsu_gating": True,
    }
    response = client.post("/api/v1/predict/patch", json=payload)
    assert response.status_code == 200

    data = response.json()
    dist = data["class_distribution"]
    # Blank glass has near-zero saturation, Otsu gating extinguishes all false positives
    assert dist["BACK"] >= 95.0
    assert data["biomarkers"]["viable_tumor_percentage"] == 0.0


def test_predict_patch_missing_data_error():
    """Tests POST /api/v1/predict/patch returns 400 when no image is provided."""
    response = client.post("/api/v1/predict/patch", json={})
    assert response.status_code == 400
    assert "Missing image data" in response.json()["detail"]


def test_predict_wsi_multi_patch():
    """Tests POST /api/v1/predict/wsi with multi-patch tile batch."""
    tile1_bytes = create_synthetic_patch_bytes(100, 100, color=(180, 80, 140))
    tile2_bytes = create_synthetic_patch_bytes(100, 100, color=(150, 110, 160))

    tile1_np = np.array(Image.open(io.BytesIO(tile1_bytes)), dtype=np.uint8)
    tile2_np = np.array(Image.open(io.BytesIO(tile2_bytes)), dtype=np.uint8)

    payload = {
        "slide_id": "TCGA-TEST-SLIDE",
        "tiles": [
            {"id": "tile_1", "x": 0, "y": 0, "image_base64": encode_image_to_base64(tile1_np)},
            {"id": "tile_2", "x": 100, "y": 0, "image_base64": encode_image_to_base64(tile2_np)},
        ],
        "mode": "proposed",
    }
    response = client.post("/api/v1/predict/wsi", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["slide_id"] == "TCGA-TEST-SLIDE"
    assert data["num_tiles"] == 2
    assert data["composite_shape"] == [100, 200]
    assert "biomarkers" in data
    assert "composite_mask_base64" in data
    assert "composite_overlay_base64" in data


def test_serve_viewer_html():
    """Tests GET / and GET /viewer serve the interactive HTML OpenSeadragon workstation."""
    for path in ["/", "/viewer"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        content = response.text
        # Assert OpenSeadragon presence
        assert "openseadragon" in content.lower()
        # Assert 5 tissue classes in legend
        for hex_code in CLASS_COLORS_HEX.values():
            assert hex_code.lower() in content.lower()
        # Assert opacity slider element exists
        assert "opacityslider" in content.lower()
        # Assert biomarker elements exist
        assert "valvtp" in content.lower()
        assert "valtsr" in content.lower()


def test_clinical_viewer_utilities():
    """Tests utility functions in clinical_viewer.py for shapes, colors, and DZI manifests."""
    synthetic_mask = np.array([
        [0, 1, 2],
        [3, 4, 0],
    ], dtype=np.int64)

    # 1. Test mask to RGB
    color_img = mask_to_color_image(synthetic_mask)
    assert color_img.shape == (2, 3, 3)
    assert color_img.dtype == np.uint8
    # Pixel [0, 0] is class 0 (TUM: #E41A1C -> (228, 26, 28))
    assert tuple(color_img[0, 0]) == (228, 26, 28)

    # 2. Test mask to RGBA overlay (background class 4 transparent)
    rgba = mask_to_rgba_overlay(synthetic_mask, alpha=0.5, bg_transparent=True)
    assert rgba.shape == (2, 3, 4)
    assert rgba[1, 1, 3] == 0  # Background pixel alpha is 0
    assert rgba[0, 0, 3] > 0   # Foreground tumor pixel alpha > 0

    # 3. Test DZI manifest generation
    dzi = generate_dzi_manifest(width=1000, height=800, tile_size=256)
    assert dzi["Image"]["Size"]["Width"] == "1000"
    assert dzi["Image"]["Size"]["Height"] == "800"
    assert dzi["Image"]["TileSize"] == "256"

    # 4. Test tile stitching
    t1 = {"x": 0, "y": 0, "mask": np.zeros((50, 50), dtype=np.int64)}
    t2 = {"x": 50, "y": 0, "mask": np.ones((50, 50), dtype=np.int64)}
    stitched_mask, stitched_rgb = stitch_tiles_heatmap([t1, t2])
    assert stitched_mask.shape == (50, 100)
    assert stitched_rgb.shape == (50, 100, 3)


def test_api_error_handling_and_real_sample_patch_inference():
    """Tests FastAPI error responses (400) on corrupted inputs and valid inference on real sample_dataset patch."""
    from pathlib import Path

    # 1. Empty multipart upload -> HTTP 400
    resp_empty = client.post("/api/v1/predict/patch", files={"file": ("empty.png", b"", "image/png")})
    assert resp_empty.status_code == 400

    # 2. Non-image corrupted binary upload -> HTTP 400
    resp_bad_bytes = client.post(
        "/api/v1/predict/patch",
        files={"file": ("corrupt.png", b"not-a-valid-png-header", "image/png")},
    )
    assert resp_bad_bytes.status_code == 400

    # 3. Empty WSI tiles list -> HTTP 400
    resp_empty_wsi = client.post("/api/v1/predict/wsi", json={"slide_id": "EMPTY", "tiles": []})
    assert resp_empty_wsi.status_code == 400

    # 4. Invalid base64 in WSI tile -> HTTP 400
    resp_bad_tile = client.post(
        "/api/v1/predict/wsi",
        json={
            "slide_id": "BAD_TILE",
            "tiles": [{"id": "t0", "x": 0, "y": 0, "image_base64": "!!!not_base64!!!"}],
        },
    )
    assert resp_bad_tile.status_code == 400

    # 5. Real histopathology patch from data/sample_dataset/test/img/
    sample_pngs = sorted(Path("data/sample_dataset/test/img").glob("*.png"))
    assert len(sample_pngs) > 0
    raw_bytes = sample_pngs[0].read_bytes()
    resp_real = client.post(
        "/api/v1/predict/patch",
        files={"file": (sample_pngs[0].name, raw_bytes, "image/png")},
    )
    assert resp_real.status_code == 200
    body = resp_real.json()
    assert body["success"] is True
    assert body["mask_shape"] == [224, 224]
    assert 0.0 <= body["biomarkers"]["viable_tumor_percentage"] <= 100.0
    assert 0.0 <= body["biomarkers"]["tumor_stroma_ratio"] <= 100.0


def test_visual_diagnostic_overlays_and_samples_compare_api():
    """Verifies visual correctness overlays (Green=Match, Red=Error, Contours, Entropy, Spotlight)
    and real held-out cohort /api/v1/samples + /api/v1/compare endpoints."""
    # 1. Unit verify Green=Match / Red=Mismatch / Amber=Margin logic
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    gt = np.zeros((64, 64), dtype=np.int64)
    gt[32:, :] = 1
    pred = gt.copy()
    pred[:16, :16] = 2  # Core region mismatch

    agree_rgb, summary = compute_agreement_error_overlay(img, pred, gt, alpha=1.0, tolerance_px=2)
    assert agree_rgb.shape == (64, 64, 3)
    # Matching pixels have higher G channel than R; core mismatch has higher R than G
    assert float(agree_rgb[48, 48, 1]) > float(agree_rgb[48, 48, 0])
    assert float(agree_rgb[4, 4, 0]) > float(agree_rgb[4, 4, 1])
    assert summary["core_error_pct"] > 0.0
    assert 80.0 < summary["exact_match_pct"] < 100.0

    dual_rgb = compute_dual_mask_blend(img, pred, gt)
    contour_rgb = compute_contour_comparison_overlay(img, pred, gt)
    spot_rgb, spot_stats = compute_per_class_spotlight(img, pred, gt, target_class=0)
    probs = np.full((4, 64, 64), 0.25, dtype=np.float32)
    ent_rgb, mean_ent = compute_entropy_heatmap_overlay(img, probs)
    assert dual_rgb.shape == (64, 64, 3)
    assert contour_rgb.shape == (64, 64, 3)
    assert spot_rgb.shape == (64, 64, 3) and spot_stats["iou_pct"] < 100.0
    assert ent_rgb.shape == (64, 64, 3) and mean_ent > 0.9

    # 2. Verify /api/v1/samples cohort discovery and /api/v1/predict/patch with sample_id + GT QA
    samples_resp = client.get("/api/v1/samples")
    assert samples_resp.status_code == 200
    samples_data = samples_resp.json()
    assert samples_data["count"] > 0
    first_id = samples_data["samples"][0]["sample_id"]

    detail_resp = client.get(f"/api/v1/samples/{first_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["image_base64"].startswith("data:image/png;base64,")

    pred_resp = client.post(
        "/api/v1/predict/patch",
        json={"sample_id": first_id, "mode": "m6", "apply_otsu_gating": True},
    )
    assert pred_resp.status_code == 200
    pdata = pred_resp.json()
    assert pdata["entropy_overlay_base64"].startswith("data:image/png;base64,")
    assert pdata["contour_overlay_base64"].startswith("data:image/png;base64,")
    if detail["gt_mask_base64"] is not None:
        assert pdata["agreement_overlay_base64"].startswith("data:image/png;base64,")
        assert pdata["verification_metrics"]["exact_match_pct"] >= 0.0

    # 3. Verify side-by-side method comparator /api/v1/compare
    cmp_resp = client.post(
        "/api/v1/compare",
        json={"sample_id": first_id, "methods": ["m1", "m6"], "apply_otsu_gating": True},
    )
    assert cmp_resp.status_code == 200
    cdata = cmp_resp.json()
    assert len(cdata["comparisons"]) == 2


def test_composable_methods_and_default_compare_no_batch4_crash():
    """Regression test for P0 batch-4 hook crash and P1 prototype calibration across sg_dap/mpppc/combined."""
    from src.visualization.api_server import ClinicalWorkstationInferenceManager, _manager

    samples_resp = client.get("/api/v1/samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()["samples"]
    assert len(samples) > 0
    sample_id = samples[0]["sample_id"]

    # 1. Verify sg_dap, mpppc, and combined all succeed without shape mismatch
    for method in ("sg_dap", "mpppc", "combined"):
        resp = client.post(
            "/api/v1/predict/patch",
            json={"sample_id": sample_id, "mode": method, "apply_otsu_gating": True},
        )
        assert resp.status_code == 200, f"Mode {method} failed: {resp.text}"
        body = resp.json()
        assert body["success"] is True
        assert body["method_used"] == method
        assert body["mask_shape"] == [224, 224]
        assert body["mask_base64"].startswith("data:image/png;base64,")
        assert body["entropy_overlay_base64"].startswith("data:image/png;base64,")

    # 2. Verify UI default 4-method compare payload succeeds
    cmp_resp = client.post(
        "/api/v1/compare",
        json={
            "sample_id": sample_id,
            "methods": ["m1", "m6", "sg_dap", "combined"],
            "apply_otsu_gating": True,
        },
    )
    assert cmp_resp.status_code == 200, f"Default compare crashed: {cmp_resp.text}"
    cmp_data = cmp_resp.json()
    assert len(cmp_data["comparisons"]) == 4
    assert [c["method_used"] for c in cmp_data["comparisons"]] == ["m1", "m6", "sg_dap", "combined"]

    # 3. Verify MP-PPC prototypes are deterministically calibrated (when checkpoints loaded)
    _manager.ensure_initialized()
    if _manager.stage2_loaded:
        ev = _manager._get_ablation_evaluator()
        assert ev.prototypes_calibrated is True
        health = client.get("/api/v1/health").json()
        assert health["model"]["prototypes_calibrated"] is True

        # Verify deterministic classifier-weight fallback when training dir is absent
        import torch.nn as nn
        import torch.nn.functional as F

        mock_mgr = ClinicalWorkstationInferenceManager()
        mock_mgr.model_stage2 = _manager.model_stage2
        mock_mgr.stage2_loaded = True
        head = mock_mgr.model_stage2.decoder.last_conv[8]
        # Force training path miss by pointing PROJECT_ROOT temporarily
        import src.visualization.api_server as api_mod

        orig_root = api_mod.PROJECT_ROOT
        try:
            api_mod.PROJECT_ROOT = orig_root / "__nonexistent_root__"
            fallback_ev = mock_mgr._get_ablation_evaluator()
            assert fallback_ev.prototypes_calibrated is True
            assert isinstance(head, nn.Module)
            expected_w = F.normalize(head.weight.detach()[:4, :256].reshape(4, 256).float(), dim=-1, eps=1e-6)
            assert np.allclose(
                fallback_ev.mpppc.prototypes[:, 0, :].detach().cpu().numpy(),
                expected_w.cpu().numpy(),
                atol=1e-5,
            )
        finally:
            api_mod.PROJECT_ROOT = orig_root


def test_api_manager_startup_ordering_and_concurrent_hook_isolation():
    """Regression test: ensure_initialized calibrates prototypes before _initialized=True and hooks stay isolated."""
    from concurrent.futures import ThreadPoolExecutor

    from src.visualization.api_server import ClinicalWorkstationInferenceManager, _manager

    _manager.ensure_initialized()
    if not _manager.stage2_loaded or _manager.model_stage2 is None:
        pytest.skip("Verified Stage-2 checkpoint not available in this environment.")

    # 1. Verify ensure_initialized completes _get_ablation_evaluator() before _initialized becomes True
    fresh_mgr = ClinicalWorkstationInferenceManager()
    seen_initialized_during_cal = []
    orig_get_ablation = fresh_mgr._get_ablation_evaluator

    def _wrapped_get_ablation():
        seen_initialized_during_cal.append(fresh_mgr._initialized)
        return orig_get_ablation()

    fresh_mgr._get_ablation_evaluator = _wrapped_get_ablation  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(fresh_mgr.ensure_initialized)
        f2 = pool.submit(fresh_mgr.ensure_initialized)
        f1.result()
        f2.result()

    assert seen_initialized_during_cal == [False]
    assert fresh_mgr._initialized is True
    assert fresh_mgr._ablation_eval is not None
    assert getattr(fresh_mgr._ablation_eval, "prototypes_calibrated", False) is True

    # 2. Verify no permanent hook remains on last_conv[8] and concurrent m6 + combined calls remain deterministic
    last_head = _manager.model_stage2.decoder.last_conv[8]
    assert len(last_head._forward_hooks) == 0

    patch_rgb = np.full((128, 128, 3), 140, dtype=np.uint8)
    patch_rgb[20:90, 20:90] = [190, 110, 160]
    seq_mask, _, _, _ = _manager.predict(patch_rgb, mode="combined", enforce_otsu=True)
    assert len(last_head._forward_hooks) == 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_m6 = pool.submit(_manager.predict, patch_rgb, "m6", True)
        fut_comb = pool.submit(_manager.predict, patch_rgb, "combined", True)
        mask_m6, probs_m6, m_m6, ck_m6 = fut_m6.result()
        mask_comb, probs_comb, m_comb, ck_comb = fut_comb.result()

    assert len(last_head._forward_hooks) == 0
    assert m_m6 == "m6" and ck_m6 is True
    assert m_comb == "combined" and ck_comb is True
    assert mask_m6.shape == (128, 128) and probs_m6.shape == (4, 128, 128)
    assert np.array_equal(mask_comb, seq_mask)


def test_segment_and_biomarkers_json_and_multipart():
    """Regression test: POST /segment and POST /biomarkers must handle JSON and multipart without Form(None) leak."""
    patch_bytes = create_synthetic_patch_bytes(64, 64, color=(180, 60, 120))
    raw_np = np.array(Image.open(io.BytesIO(patch_bytes)), dtype=np.uint8)
    b64_str = encode_image_to_base64(raw_np, format="PNG")

    # 1. POST /segment (JSON)
    seg_json = client.post("/segment", json={"image_base64": b64_str, "mode": "m6", "apply_otsu_gating": True})
    assert seg_json.status_code == 200, f"/segment JSON failed: {seg_json.text}"
    seg_json_data = seg_json.json()
    assert seg_json_data["success"] is True
    assert seg_json_data["mask_shape"] == [64, 64]

    # 2. POST /segment (multipart)
    seg_mp = client.post("/segment", files={"file": ("patch.png", patch_bytes, "image/png")})
    assert seg_mp.status_code == 200, f"/segment multipart failed: {seg_mp.text}"
    seg_mp_data = seg_mp.json()
    assert seg_mp_data["success"] is True
    assert seg_mp_data["mask_shape"] == [64, 64]

    # 3. POST /biomarkers (JSON)
    bio_json = client.post("/biomarkers", json={"image_base64": b64_str, "mode": "m6", "apply_otsu_gating": True})
    assert bio_json.status_code == 200, f"/biomarkers JSON failed: {bio_json.text}"
    bio_json_data = bio_json.json()
    assert 0.0 <= bio_json_data["viable_tumor_percentage"] <= 100.0
    assert 0.0 <= bio_json_data["tumor_stroma_ratio"] <= 100.0

    # 4. POST /biomarkers (multipart)
    bio_mp = client.post("/biomarkers", files={"file": ("patch.png", patch_bytes, "image/png")})
    assert bio_mp.status_code == 200, f"/biomarkers multipart failed: {bio_mp.text}"
    bio_mp_data = bio_mp.json()
    assert 0.0 <= bio_mp_data["viable_tumor_percentage"] <= 100.0
    assert 0.0 <= bio_mp_data["tumor_stroma_ratio"] <= 100.0


def test_predict_wsi_and_stitch_tiles_reject_negative_coordinates():
    """Regression test: negative tile coordinates must be rejected cleanly at API and Python helper layers."""
    patch_bytes = create_synthetic_patch_bytes(64, 64, color=(180, 80, 140))
    raw_np = np.array(Image.open(io.BytesIO(patch_bytes)), dtype=np.uint8)
    b64_str = encode_image_to_base64(raw_np, format="PNG")

    for bad_x, bad_y in [(-10, 0), (0, -1), (-5, -5)]:
        resp = client.post(
            "/api/v1/predict/wsi",
            json={
                "slide_id": "NEG_COORDS",
                "tiles": [{"id": "t_neg", "x": bad_x, "y": bad_y, "image_base64": b64_str}],
            },
        )
        assert resp.status_code in (400, 422), f"Expected 400/422 for ({bad_x}, {bad_y}), got {resp.status_code}"

    with pytest.raises(ValueError, match="non-negative"):
        stitch_tiles_heatmap([{"x": -10, "y": 0, "mask": np.zeros((16, 16), dtype=np.int64)}])
    with pytest.raises(ValueError, match="non-negative"):
        stitch_tiles_heatmap([{"x": 0, "y": -1, "mask": np.zeros((16, 16), dtype=np.int64)}])


def test_multipart_form_parity_across_patch_segment_and_biomarkers():
    """Regression test: /segment and /biomarkers must honor mode_form, sample_id_form, and target_class_form."""
    samples_resp = client.get("/api/v1/samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()["samples"]
    assert len(samples) > 0
    sample_id = samples[0]["sample_id"]

    patch_bytes = create_synthetic_patch_bytes(64, 64, color=(180, 60, 120))
    form_data = {
        "mode_form": "m1",
        "sample_id_form": sample_id,
        "target_class_form": "2",
    }

    r_patch = client.post(
        "/api/v1/predict/patch",
        files={"file": ("patch.png", patch_bytes, "image/png")},
        data=form_data,
    )
    r_seg = client.post(
        "/segment",
        files={"file": ("patch.png", patch_bytes, "image/png")},
        data=form_data,
    )
    r_bio = client.post(
        "/biomarkers",
        files={"file": ("patch.png", patch_bytes, "image/png")},
        data=form_data,
    )

    assert r_patch.status_code == 200, r_patch.text
    assert r_seg.status_code == 200, r_seg.text
    assert r_bio.status_code == 200, r_bio.text

    d_patch = r_patch.json()
    d_seg = r_seg.json()
    d_bio = r_bio.json()

    assert d_patch["method_used"] == "m1"
    assert d_seg["method_used"] == "m1"
    assert d_patch["spotlight_stats"]["target_class"] == 2
    assert d_seg["spotlight_stats"]["target_class"] == 2
    if samples[0].get("gt_available"):
        assert d_patch["verification_metrics"] is not None
        assert d_seg["verification_metrics"] is not None
        assert d_patch["gt_color_base64"] is not None
        assert d_seg["gt_color_base64"] is not None
    assert d_seg["biomarkers"] == d_patch["biomarkers"]
    assert d_bio == d_patch["biomarkers"]

    for endpoint in ("/api/v1/predict/patch", "/segment", "/biomarkers"):
        bad_resp = client.post(
            endpoint,
            files={"file": ("patch.png", patch_bytes, "image/png")},
            data={"target_class_form": "abc"},
        )
        assert bad_resp.status_code == 422, f"{endpoint} returned {bad_resp.status_code}: {bad_resp.text}"


def test_json_validation_guards_across_patch_segment_and_biomarkers():
    """Regression test: invalid JSON target_class / non-dict payloads return 422 (never 500), and null defaults cleanly."""
    patch_bytes = create_synthetic_patch_bytes(64, 64, color=(180, 60, 120))
    raw_np = np.array(Image.open(io.BytesIO(patch_bytes)), dtype=np.uint8)
    b64_str = encode_image_to_base64(raw_np, format="PNG")

    samples_resp = client.get("/api/v1/samples")
    assert samples_resp.status_code == 200
    sample_id = samples_resp.json()["samples"][0]["sample_id"]

    invalid_target_values = ["not_an_int", True, False, [1], {"k": 1}, 2.5]
    for endpoint in ("/api/v1/predict/patch", "/segment", "/biomarkers"):
        for bad_tc in invalid_target_values:
            resp = client.post(
                endpoint,
                json={"image_base64": b64_str, "target_class": bad_tc},
            )
            assert resp.status_code == 422, f"{endpoint} with target_class={bad_tc!r} gave {resp.status_code}"

        # Non-dict JSON payloads must fail cleanly with 422
        for non_dict in ([1, 2], "not_a_dict", 123):
            resp_nd = client.post(endpoint, json=non_dict)
            assert resp_nd.status_code == 422, f"{endpoint} with non-dict {non_dict!r} gave {resp_nd.status_code}"
        resp_json_null = client.post(endpoint, content="null", headers={"Content-Type": "application/json"})
        assert resp_json_null.status_code == 422, f"{endpoint} with JSON null gave {resp_json_null.status_code}"

        # JSON target_class=null must default cleanly to 0 (HTTP 200)
        resp_null_tc = client.post(
            endpoint,
            json={"image_base64": b64_str, "target_class": None},
        )
        assert resp_null_tc.status_code == 200, f"{endpoint} null target_class failed: {resp_null_tc.text}"
        if endpoint != "/biomarkers":
            assert resp_null_tc.json()["spotlight_stats"]["target_class"] == 0

        # JSON mode=null must default cleanly rather than turning into "None"
        resp_null_mode = client.post(
            endpoint,
            json={"sample_id": sample_id, "mode": None},
        )
        assert resp_null_mode.status_code == 200, f"{endpoint} null mode failed: {resp_null_mode.text}"
        if endpoint != "/biomarkers":
            assert resp_null_mode.json()["method_used"] == "m6"


class _UnitDecoderStub(nn.Module):
    """Minimal decoder exposing last_conv[8] for 256-dim feature hooks."""

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


class _UnitStage2Stub(nn.Module):
    """Fast deterministic 5-class Stage-2 stub outputting (B, 5, H, W) logits on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = _UnitDecoderStub(out_channels=5)
        with torch.no_grad():
            self.decoder.last_conv[0].weight.zero_()
            self.decoder.last_conv[0].bias.zero_()
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


class _UnitStage1Stub(nn.Module):
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
        _, probs = self.forward_cam(x)
        return probs

    def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = F.avg_pool2d(x, kernel_size=4, stride=4) if min(x.shape[-2:]) >= 8 else x
        cam = F.relu(self.proj(feat))
        probs = torch.sigmoid(cam.mean(dim=(2, 3)))
        return cam, probs


def _fast_stub_client(monkeypatch: pytest.MonkeyPatch) -> Tuple[TestClient, api_mod.ClinicalWorkstationInferenceManager]:
    """Monkeypatches api_server._manager with tiny deterministic stubs for <1s isolated API tests."""
    from src.clinical.comparative_ablation_evaluator import ComparativeAblationEvaluator
    from src.models.wsss_inference_methods import MultiMethodWSSSEngineV2

    stub_s2 = _UnitStage2Stub().eval()
    stub_s1 = _UnitStage1Stub().eval()

    mgr = api_mod.ClinicalWorkstationInferenceManager()
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

    monkeypatch.setattr(api_mod, "_manager", mgr)
    return TestClient(app), mgr


def _decode_gray_l_b64(b64_str: str) -> np.ndarray:
    raw = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
    img = Image.open(io.BytesIO(base64.b64decode(raw)))
    assert img.mode == "L"
    return np.array(img, dtype=np.uint8)


def test_clinical_viewer_blurred_dual_cam_and_unified_miou():
    """Unit tests M2 blurred dual QA overlay, Stage-1 CAM overlay, unified mIoU, and 2x4 diagnostic PNG."""
    img = np.full((224, 224, 3), 185, dtype=np.uint8)
    gt = np.full((224, 224), 4, dtype=np.int64)
    gt[20:140, 20:140] = 0  # Large TUM block
    gt[:4, :] = 255         # ignore_index=255 region

    pred = gt.copy()
    pred[40:70, 40:70] = 1      # Core tissue misclass (TUM -> STR), >2px from GT boundary
    pred[160:190, 160:190] = 0  # Glass FP (BACK -> TUM), >2px from GT boundary
    pred[139:141, 60:90] = 0    # 1px boundary shift at GT edge (within tolerance_px=2)

    out_blur = compute_blurred_dual_verification_overlay(
        img, pred, gt, blur_sigma=4.0, dim_factor=0.55, gt_alpha=0.45, pred_alpha=0.55, error_alpha=0.65, tolerance_px=2
    )
    out_sharp = compute_blurred_dual_verification_overlay(
        img, pred, gt, blur_sigma=0.0, dim_factor=0.55, gt_alpha=0.45, pred_alpha=0.55, error_alpha=0.65, tolerance_px=2
    )
    out_dim = compute_blurred_dual_verification_overlay(
        img, pred, gt, blur_sigma=4.0, dim_factor=0.90, gt_alpha=0.20, pred_alpha=0.20, error_alpha=0.65, tolerance_px=2
    )
    assert out_blur.shape == (224, 224, 3) and out_blur.dtype == np.uint8
    assert not np.array_equal(out_blur, out_sharp)
    assert not np.array_equal(out_blur, out_dim)

    # TP (Emerald Green), Core tissue misclass (Red), Glass FP (Magenta), Boundary shift (Amber)
    tp_px = out_blur[28, 28]
    mis_px = out_blur[55, 55]
    fp_px = out_blur[175, 175]
    amb_px = out_blur[140, 75]
    assert int(tp_px[1]) > int(mis_px[1])
    assert int(mis_px[0]) > int(mis_px[1])
    assert int(fp_px[0]) > int(fp_px[1]) and int(fp_px[2]) > int(fp_px[1])
    assert int(amb_px[0]) > int(amb_px[2]) and int(amb_px[1]) > int(amb_px[2])

    # compute_cam_overlay on 3D (4, 32, 32) and 2D (32, 32)
    cam_3d = np.zeros((4, 32, 32), dtype=np.float32)
    cam_3d[0, 4:16, 4:16] = 0.9
    cam_rgb_0, stats_0 = compute_cam_overlay(img, cam_3d, target_class=0)
    cam_rgb_max, stats_max = compute_cam_overlay(img, cam_3d, target_class=None)
    cam_rgb_2d, stats_2d = compute_cam_overlay(img, cam_3d[0], target_class=None)
    assert cam_rgb_0.shape == (224, 224, 3)
    assert cam_rgb_max.shape == (224, 224, 3)
    assert cam_rgb_2d.shape == (224, 224, 3)
    for st in (stats_0, stats_max, stats_2d):
        assert {"target_class", "peak_activation", "mean_activation", "active_area_pct"} <= set(st.keys())
    assert stats_0["target_class"] == 0 and stats_max["target_class"] is None

    # compute_patch_iou_metrics: <50px GT sliver vs >=50px absent-class FP
    gt_iou = np.full((224, 224), 4, dtype=np.int64)
    gt_iou[20:100, 20:100] = 0   # 6400 px class 0
    gt_iou[10:15, 10:15] = 2     # 25 px sliver (< 50 px)
    pred_iou = gt_iou.copy()
    pred_iou[140:160, 140:160] = 1  # 400 px FP of absent class 1
    iou_res = compute_patch_iou_metrics(pred_iou, gt_iou)
    assert iou_res["miou_gt_present"] == pytest.approx(100.0, abs=0.2)
    assert iou_res["miou_penalized_4cls"] == pytest.approx(50.0, abs=0.2)

    # render_single_patch_diagnostic_sheet with and without GT
    probs = np.full((4, 224, 224), 0.25, dtype=np.float32)
    sheet_gt = render_single_patch_diagnostic_sheet(
        image=img,
        pred_mask=pred,
        probs=probs,
        gt_mask=gt,
        cam_map=cam_3d,
        method_name="m6",
        sample_id="unit_patch",
        biomarkers={"viable_tumor_percentage": 25.0, "tumor_stroma_ratio": 55.0},
        gt_biomarkers={"viable_tumor_percentage": 30.0, "tumor_stroma_ratio": 50.0},
        dpi=150,
    )
    sheet_no_gt = render_single_patch_diagnostic_sheet(
        image=img,
        pred_mask=pred,
        probs=probs,
        gt_mask=None,
        cam_map=None,
        method_name="m6",
        sample_id="unit_no_gt",
        dpi=150,
    )
    for png_bytes in (sheet_gt, sheet_no_gt):
        assert png_bytes.startswith(b"\x89PNG")
        pil_sheet = Image.open(io.BytesIO(png_bytes))
        assert pil_sheet.size[0] > 1000 and pil_sheet.size[1] > 600


def test_predict_patch_m2_visual_fields_index_maps_and_strict_weights(monkeypatch: pytest.MonkeyPatch):
    """Tests M2 overlay fields, pixel-index maps, agreement_summary parity, and strict_weights=True 503."""
    stub_client, mgr = _fast_stub_client(monkeypatch)

    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    img[:, :, 0] = 200
    gt = np.full((64, 64), 4, dtype=np.uint8)
    gt[8:48, 8:48] = 0
    gt[:4, :] = 255  # ignore_index=255 rows
    gt_buf = io.BytesIO()
    Image.fromarray(gt, mode="L").save(gt_buf, format="PNG")
    gt_b64 = "data:image/png;base64," + base64.b64encode(gt_buf.getvalue()).decode("ascii")

    resp = stub_client.post(
        "/api/v1/predict/patch",
        json={
            "image_base64": encode_image_to_base64(img, format="PNG"),
            "gt_mask_base64": gt_b64,
            "mode": "m6",
            "apply_otsu_gating": True,
            "blur_sigma": 3.5,
            "dim_factor": 0.6,
            "gt_alpha": 0.4,
            "pred_alpha": 0.6,
            "error_alpha": 0.7,
            "tolerance_px": 2,
            "error_breakdown": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["blurred_dual_overlay_base64"].startswith("data:image/png;base64,")
    assert data["cam_overlay_base64"].startswith("data:image/png;base64,")
    assert data["pred_mask_index_base64"].startswith("data:image/png;base64,")
    assert data["gt_mask_index_base64"].startswith("data:image/png;base64,")
    assert data["confidence_map_base64"].startswith("data:image/png;base64,")
    assert isinstance(data["gt_distribution"], dict)
    assert isinstance(data["gt_biomarkers"], dict)

    pred_idx = _decode_gray_l_b64(data["pred_mask_index_base64"])
    gt_idx = _decode_gray_l_b64(data["gt_mask_index_base64"])
    conf_idx = _decode_gray_l_b64(data["confidence_map_base64"])
    assert pred_idx.shape == (64, 64)
    assert gt_idx.shape == (64, 64)
    assert conf_idx.shape == (64, 64)
    assert set(np.unique(pred_idx).tolist()).issubset({0, 1, 2, 3, 4})

    summary = data["agreement_summary"]
    assert summary == data["verification_metrics"]
    required_keys = {
        "miou_4cls",
        "miou_gt_present",
        "miou_penalized_4cls",
        "miou_penalized",
        "mdice_4cls",
        "boundary_f1",
        "per_class_iou",
        "top_confusions",
        "valid_tissue_px",
        "tp_px",
        "margin_err_px",
        "core_err_px",
        "fp_glass_px",
        "fn_missed_px",
        "misclass_px",
    }
    assert required_keys <= set(summary.keys())

    _, recomputed = compute_agreement_error_overlay(
        img,
        pred_idx.astype(np.int64),
        gt_idx.astype(np.int64),
        tolerance_px=2,
        error_breakdown=True,
    )
    for k in ("valid_tissue_px", "tp_px", "margin_err_px", "core_err_px", "fp_glass_px", "fn_missed_px", "misclass_px"):
        assert summary[k] == recomputed[k]

    # Without GT: index & confidence maps exist, GT/dual overlays are None
    resp_no_gt = stub_client.post(
        "/api/v1/predict/patch",
        json={"image_base64": encode_image_to_base64(img, format="PNG"), "mode": "m6"},
    )
    assert resp_no_gt.status_code == 200
    d_no_gt = resp_no_gt.json()
    assert d_no_gt["blurred_dual_overlay_base64"] is None
    assert d_no_gt["gt_mask_index_base64"] is None
    assert d_no_gt["gt_distribution"] is None
    assert d_no_gt["gt_biomarkers"] is None
    assert d_no_gt["agreement_summary"] is None
    assert d_no_gt["pred_mask_index_base64"].startswith("data:image/png;base64,")
    assert d_no_gt["confidence_map_base64"].startswith("data:image/png;base64,")

    # strict_weights=True returns HTTP 503 when stage2_loaded=False
    mgr.stage2_loaded = False
    resp_503 = stub_client.post(
        "/api/v1/predict/patch",
        json={
            "image_base64": encode_image_to_base64(img, format="PNG"),
            "mode": "m6",
            "strict_weights": True,
        },
    )
    assert resp_503.status_code == 503


def test_export_report_png_json_csv_and_validation(monkeypatch: pytest.MonkeyPatch):
    """Tests POST /api/v1/export/report across png/json/csv formats and 400/422 guards."""
    stub_client, _ = _fast_stub_client(monkeypatch)
    img = np.full((64, 64, 3), 150, dtype=np.uint8)
    img[:, :, 0] = 210
    img_b64 = encode_image_to_base64(img, format="PNG")
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[32:, :] = 1
    gt_buf = io.BytesIO()
    Image.fromarray(gt, mode="L").save(gt_buf, format="PNG")
    gt_b64 = "data:image/png;base64," + base64.b64encode(gt_buf.getvalue()).decode("ascii")

    r_png = stub_client.post(
        "/api/v1/export/report",
        json={"image_base64": img_b64, "gt_mask_base64": gt_b64, "mode": "m6", "format": "png"},
    )
    assert r_png.status_code == 200
    assert r_png.headers["content-type"] == "image/png"
    assert r_png.headers["content-disposition"].endswith('_diagnostic_sheet.png"')
    assert r_png.content.startswith(b"\x89PNG")
    assert Image.open(io.BytesIO(r_png.content)).size[0] >= 1000

    r_json = stub_client.post(
        "/api/v1/export/report",
        json={"image_base64": img_b64, "gt_mask_base64": gt_b64, "mode": "m6", "format": "json"},
    )
    assert r_json.status_code == 200
    assert "application/json" in r_json.headers["content-type"]
    jdoc = json.loads(r_json.content.decode("utf-8"))
    for key in ("biomarkers", "gt_biomarkers", "class_distribution", "gt_distribution", "agreement_summary", "verification_metrics"):
        assert key in jdoc

    r_csv = stub_client.post(
        "/api/v1/export/report",
        json={"image_base64": img_b64, "gt_mask_base64": gt_b64, "mode": "m6", "format": "csv"},
    )
    assert r_csv.status_code == 200
    assert "text/csv" in r_csv.headers["content-type"]
    rows = list(csv.reader(io.StringIO(r_csv.content.decode("utf-8"))))
    assert rows[0] == ["section", "metric", "value"]
    sections = {row[0] for row in rows[1:] if row}
    assert {"pred_biomarkers", "gt_biomarkers", "pred_class_pct"} <= sections
    assert any(s.startswith("verification") for s in sections)

    r_bad_fmt = stub_client.post("/api/v1/export/report", json={"image_base64": img_b64, "format": "xml"})
    assert r_bad_fmt.status_code == 422

    r_missing = stub_client.post("/api/v1/export/report", json={"format": "json"})
    assert r_missing.status_code == 400


def test_samples_filtering_pagination_and_detail():
    """Tests /api/v1/samples filtering, pagination, invalid bounds 422, and /api/v1/samples/{id}."""
    base_resp = client.get("/api/v1/samples")
    assert base_resp.status_code == 200
    base_data = base_resp.json()
    assert base_data["total"] >= base_data["count"] > 0

    first_item = base_data["samples"][0]
    first_pid = first_item["patient_id"]
    first_morph = first_item["present_classes"][0]

    pid_resp = client.get("/api/v1/samples", params={"patient_id": first_pid})
    assert pid_resp.status_code == 200
    pid_data = pid_resp.json()
    assert 0 < pid_data["total"] <= base_data["total"]
    assert all(first_pid.lower() in s["patient_id"].lower() or first_pid.lower() in s["sample_id"].lower() for s in pid_data["samples"])

    morph_resp = client.get("/api/v1/samples", params={"morphology": first_morph})
    assert morph_resp.status_code == 200
    assert morph_resp.json()["total"] > 0

    if base_data["total"] >= 3:
        p0 = client.get("/api/v1/samples", params={"limit": 2, "offset": 0}).json()["samples"]
        p1 = client.get("/api/v1/samples", params={"limit": 2, "offset": 1}).json()["samples"]
        assert len(p0) == 2 and len(p1) == 2
        assert p1[0]["sample_id"] == p0[1]["sample_id"]

    far_resp = client.get("/api/v1/samples", params={"offset": 999999})
    assert far_resp.status_code == 200
    assert far_resp.json()["count"] == 0
    assert far_resp.json()["samples"] == []

    for bad_params in ({"limit": 0}, {"limit": 501}, {"offset": -1}):
        bad_r = client.get("/api/v1/samples", params=bad_params)
        assert bad_r.status_code == 422

    detail_r = client.get(f"/api/v1/samples/{first_item['sample_id']}")
    assert detail_r.status_code == 200
    detail = detail_r.json()
    assert "gt_distribution" in detail and "gt_biomarkers" in detail
    assert client.get("/api/v1/samples/nonexistent_sample_id_999").status_code == 404


def test_predict_wsi_and_compare_m2_contracts(monkeypatch: pytest.MonkeyPatch):
    """Tests WSI multi-tile composite_heatmap/region_summary, upfront 422 guards, and /compare 6-method cap."""
    stub_client, _ = _fast_stub_client(monkeypatch)
    t1 = np.full((64, 64, 3), 150, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 165, dtype=np.uint8)
    b1 = encode_image_to_base64(t1, format="PNG")
    b2 = encode_image_to_base64(t2, format="PNG")

    r_wsi = stub_client.post(
        "/api/v1/predict/wsi",
        json={
            "slide_id": "SLIDE_M2_STITCH",
            "mode": "m6",
            "tiles": [
                {"id": "t0", "x": 0, "y": 0, "image_base64": b1},
                {"id": "t1", "x": 32, "y": 0, "image_base64": b2},
            ],
        },
    )
    assert r_wsi.status_code == 200, r_wsi.text
    wdata = r_wsi.json()
    assert wdata["composite_shape"] == [64, 96]
    assert wdata["composite_heatmap_base64"].startswith("data:image/png;base64,")
    assert decode_base64_to_image(wdata["composite_heatmap_base64"]).shape == (64, 96, 3)
    rs = wdata["region_summary"]
    assert rs["canvas_width"] == 96 and rs["canvas_height"] == 64
    assert {"tissue_coverage_pct", "dominant_class", "method_used", "tiles"} <= set(rs.keys())
    assert len(rs["tiles"]) == 2
    assert {"vtp", "tsr", "dominant_class"} <= set(rs["tiles"][0].keys())

    # Upfront WSI guards must fail fast with 0 process_single_patch calls
    patch_calls = []

    def _spy_process(*args, **kwargs):
        patch_calls.append(1)
        return {}

    with monkeypatch.context() as mp:
        mp.setattr(api_mod, "process_single_patch", _spy_process)
        assert stub_client.post("/api/v1/predict/wsi", json={"mode": "bad_mode", "tiles": [{"x": 0, "y": 0, "image_base64": b1}]}).status_code == 422
        assert stub_client.post("/api/v1/predict/wsi", json={"mode": "m6", "tiles": [{"id": f"t{i}", "x": 0, "y": 0, "image_base64": b1} for i in range(65)]}).status_code == 422
        assert stub_client.post("/api/v1/predict/wsi", json={"mode": "m6", "tiles": [{"x": 8192, "y": 0, "image_base64": b1}]}).status_code == 422
        assert stub_client.post("/api/v1/predict/wsi", json={"mode": "m6", "tiles": [{"x": 8160, "y": 0, "image_base64": b1}]}).status_code == 422
    assert len(patch_calls) == 0

    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[32:, :] = 1
    gt_buf = io.BytesIO()
    Image.fromarray(gt, mode="L").save(gt_buf, format="PNG")
    gt_b64 = "data:image/png;base64," + base64.b64encode(gt_buf.getvalue()).decode("ascii")

    r_cmp = stub_client.post(
        "/api/v1/compare",
        json={
            "image_base64": b1,
            "gt_mask_base64": gt_b64,
            "methods": ["m1", "m2", "m3", "m4", "m5", "m6", "m7"],
        },
    )
    assert r_cmp.status_code == 200, r_cmp.text
    cdata = r_cmp.json()
    assert cdata["gt_available"] is True
    assert cdata["raw_image_base64"].startswith("data:image/png;base64,")
    assert cdata["gt_color_base64"].startswith("data:image/png;base64,")
    assert len(cdata["comparisons"]) == 6
    for entry in cdata["comparisons"]:
        assert entry["blurred_dual_overlay_base64"].startswith("data:image/png;base64,")
        assert entry["cam_overlay_base64"].startswith("data:image/png;base64,")
        assert entry["agreement_summary"] == entry["verification_metrics"]

    assert stub_client.post("/api/v1/compare", json={"methods": ["m1", "m6"]}).status_code == 400


def test_nan_inf_and_out_of_range_float_422_handling(monkeypatch: pytest.MonkeyPatch):
    """Tests NaN/Infinity/-Infinity and out-of-range numeric bounds return clean 422 JSON across M2 endpoints."""
    stub_client, _ = _fast_stub_client(monkeypatch)
    img_b64 = encode_image_to_base64(np.full((32, 32, 3), 160, dtype=np.uint8), format="PNG")

    endpoints = ("/api/v1/predict/patch", "/api/v1/compare", "/api/v1/export/report")
    for endpoint in endpoints:
        for bad_num in ("NaN", "Infinity", "-Infinity"):
            for key in ("blur_sigma", "dim_factor", "gt_alpha", "pred_alpha", "error_alpha"):
                raw_payload = f'{{"image_base64": "{img_b64}", "{key}": {bad_num}}}'
                resp = stub_client.post(endpoint, content=raw_payload, headers={"Content-Type": "application/json"})
                assert resp.status_code == 422, f"{endpoint} {key}={bad_num} gave {resp.status_code}"
                assert "detail" in resp.json()

        for bad_payload in (
            {"image_base64": img_b64, "blur_sigma": -0.5},
            {"image_base64": img_b64, "blur_sigma": 26.0},
            {"image_base64": img_b64, "dim_factor": 1.5},
            {"image_base64": img_b64, "tolerance_px": 11},
        ):
            resp_oob = stub_client.post(endpoint, json=bad_payload)
            assert resp_oob.status_code == 422, f"{endpoint} {bad_payload} gave {resp_oob.status_code}"






