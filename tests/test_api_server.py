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

import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.visualization.api_server import app
from src.visualization.clinical_viewer import (
    CLASS_COLORS_HEX,
    CLASS_NAMES,
    blend_overlay,
    compute_class_distribution,
    decode_base64_to_image,
    encode_image_to_base64,
    generate_dzi_manifest,
    mask_to_color_image,
    mask_to_rgba_overlay,
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
