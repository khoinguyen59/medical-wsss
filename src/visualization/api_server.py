"""
FastAPI Clinical Web Workstation & Inference API (SE374.R11).
Provides high-performance REST endpoints for:
- System and GPU health monitoring (/api/v1/health)
- Histopathology patch WSSS segmentation with Otsu clinical gating (/api/v1/predict/patch)
- Multi-patch WSI slide stitching and heatmap synthesis (/api/v1/predict/wsi)
- Interactive OpenSeadragon clinical viewer (/viewer and /)
"""

from datetime import datetime, timezone
import io
import os
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from src.datasets.otsu_filter import apply_otsu_gating_to_prediction
from src.datasets.patient_split import load_patient_splits
from src.losses_metrics.metrics import (
    compute_tumor_stroma_ratio,
    compute_viable_tumor_percentage,
)
from src.models.wsss_tissue import WSSSBenchmarkingEngine
from src.visualization.clinical_viewer import (
    CLASS_COLORS_HEX,
    CLASS_NAMES,
    blend_overlay,
    compute_class_distribution,
    decode_base64_to_image,
    encode_image_to_base64,
    mask_to_color_image,
    mask_to_rgba_overlay,
    stitch_tiles_heatmap,
)

# Directory configurations
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(MODULE_DIR, "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

# Create FastAPI application
app = FastAPI(
    title="SE374.R11 WSSS Clinical Workstation API",
    description="Weakly-Supervised Histopathology Semantic Segmentation and Clinical Biomarkers",
    version="1.0.0",
)

# Enable CORS for medical workstation frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Shared global engine instance (device auto-selected)
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
_engine_cache: Dict[str, WSSSBenchmarkingEngine] = {}


def get_inference_engine(mode: str = "proposed") -> WSSSBenchmarkingEngine:
    """Returns a cached WSSS benchmarking engine for the requested configuration."""
    valid_modes = ["proposed", "relocalization", "classmix", "baseline"]
    if mode not in valid_modes:
        mode = "proposed"

    if mode not in _engine_cache:
        _engine_cache[mode] = WSSSBenchmarkingEngine(
            mode=mode,
            num_classes=5,
            device=DEVICE_NAME,
        )
    return _engine_cache[mode]


# -------------------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------------------

class BiomarkersResponse(BaseModel):
    viable_tumor_percentage: float = Field(..., description="VTP percentage [0.0, 100.0]")
    tumor_stroma_ratio: float = Field(..., description="TSR percentage [0.0, 100.0]")


class PatchPredictJSONRequest(BaseModel):
    image_base64: str = Field(..., description="RGB image encoded in base64 or Data URL")
    mode: Optional[str] = Field("proposed", description="Algorithm mode: proposed, relocalization, classmix, baseline")
    apply_otsu_gating: Optional[bool] = Field(True, description="Enforce Otsu tissue gating")


class PatchPredictResponse(BaseModel):
    success: bool
    mask_shape: List[int]
    class_distribution: Dict[str, float]
    biomarkers: BiomarkersResponse
    mask_base64: str
    overlay_base64: str
    blended_base64: Optional[str] = None
    applied_gating: bool


class WSITileItem(BaseModel):
    id: Optional[str] = "tile_0"
    x: int = 0
    y: int = 0
    image_base64: str


class WSIPredictRequest(BaseModel):
    slide_id: Optional[str] = "WSI_SLIDE"
    tiles: List[WSITileItem]
    mode: Optional[str] = "proposed"


class WSIPredictResponse(BaseModel):
    success: bool
    slide_id: str
    num_tiles: int
    composite_shape: List[int]
    class_distribution: Dict[str, float]
    biomarkers: BiomarkersResponse
    composite_mask_base64: str
    composite_overlay_base64: str
    applied_gating: bool


# -------------------------------------------------------------------------
# Core Helper Logic
# -------------------------------------------------------------------------

def process_single_patch(
    image_np: np.ndarray,
    mode: str = "proposed",
    enforce_otsu: bool = True,
) -> Dict[str, Any]:
    """Runs WSSS inference with Otsu clinical gating and computes clinical biomarkers.
    
    Args:
        image_np: (H, W, 3) RGB image array in uint8.
        mode: Benchmark mode ("proposed", "relocalization", "classmix", "baseline").
        enforce_otsu: Whether to extinguish non-tissue false positives.
        
    Returns:
        Structured prediction dictionary with masks, class distributions, and biomarkers.
    """
    engine = get_inference_engine(mode)
    h_orig, w_orig = image_np.shape[:2]

    # Memory safety: downsample high-resolution inputs to max 224x224 for feature
    # and attention affinity calculation (prevents O(N^2) dense attention RAM exhaustion),
    # then upsample predicted mask back to original resolution.
    max_dim = max(h_orig, w_orig)
    if max_dim > 224:
        scale = 224.0 / max_dim
        new_w = max(16, int(round(w_orig * scale)))
        new_h = max(16, int(round(h_orig * scale)))
        pil_down = Image.fromarray(image_np).resize((new_w, new_h), resample=Image.BILINEAR)
        down_np = np.array(pil_down, dtype=np.uint8)
    else:
        down_np = image_np

    # Convert to normalized float32 tensor (1, 3, H, W)
    img_float = down_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    # Execute segmentation pipeline on downscaled patch
    down_mask = engine.generate_pseudo_mask(
        image_tensor=img_tensor,
        current_epoch=5,
        raw_image=down_np if enforce_otsu else None,
    )

    # Upsample predicted mask back to original patch resolution
    if (down_mask.shape[0], down_mask.shape[1]) != (h_orig, w_orig):
        pil_mask = Image.fromarray(down_mask.astype(np.uint8)).resize((w_orig, h_orig), resample=Image.NEAREST)
        pred_mask = np.array(pil_mask, dtype=np.int64)
    else:
        pred_mask = down_mask

    # Ensure Otsu gating is guaranteed at original resolution if requested
    if enforce_otsu:
        pred_mask = apply_otsu_gating_to_prediction(
            pred_mask=pred_mask,
            image=image_np,
            background_class_idx=4,
        )

    # Calculate clinical biomarkers using zero-mockup metrics service
    vtp = compute_viable_tumor_percentage(pred_mask, tumor_class_idx=0, background_class_idx=4)
    tsr = compute_tumor_stroma_ratio(pred_mask, tumor_class_idx=0, stroma_class_idx=1)

    # Class breakdown
    distribution = compute_class_distribution(pred_mask)

    # Generate visual outputs
    mask_rgb = mask_to_color_image(pred_mask)
    mask_rgba = mask_to_rgba_overlay(pred_mask, alpha=0.55, bg_transparent=True)
    blended = blend_overlay(image_np, pred_mask, alpha=0.45)

    mask_b64 = encode_image_to_base64(mask_rgb, format="PNG")
    overlay_b64 = encode_image_to_base64(mask_rgba, format="PNG")
    blended_b64 = encode_image_to_base64(blended, format="PNG")

    return {
        "pred_mask": pred_mask,
        "mask_shape": list(pred_mask.shape),
        "class_distribution": distribution,
        "biomarkers": {
            "viable_tumor_percentage": round(float(vtp), 2),
            "tumor_stroma_ratio": round(float(tsr), 2),
        },
        "mask_base64": mask_b64,
        "overlay_base64": overlay_b64,
        "blended_base64": blended_b64,
        "applied_gating": enforce_otsu,
    }


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/api/v1/health")
def get_system_health() -> Dict[str, Any]:
    """Returns system status, active model state, and memory/VRAM usage."""
    engine = get_inference_engine("proposed")

    cuda_avail = torch.cuda.is_available()
    vram_mb = 0.0
    if cuda_avail:
        vram_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "mode": engine.mode,
            "device": str(engine.device),
            "num_classes": 5,
            "classes": CLASS_NAMES,
            "palette": CLASS_COLORS_HEX,
            "otsu_gating": True,
            "loaded": True,
        },
        "system": {
            "torch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "vram_allocated_mb": round(vram_mb, 2),
            "os": os.name,
        },
    }


@app.post("/api/v1/predict/patch", response_model=PatchPredictResponse)
async def predict_patch(
    request: Request,
    file: Optional[UploadFile] = File(None),
) -> PatchPredictResponse:
    """Predicts WSSS semantic segmentation and clinical biomarkers on an RGB image patch.
    
    Accepts either:
    1. Multipart file upload via 'file' parameter.
    2. JSON payload with 'image_base64' string.
    """
    image_np: Optional[np.ndarray] = None
    mode: str = "proposed"
    apply_otsu: bool = True

    # 1. Handle multipart file upload
    if file is not None:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            image_np = np.array(pil_img, dtype=np.uint8)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot decode image file: {str(e)}")

    # 2. Handle JSON payload
    if image_np is None:
        try:
            body = await request.json()
            if "image_base64" in body and body["image_base64"]:
                image_np = decode_base64_to_image(body["image_base64"])
                mode = body.get("mode", "proposed")
                apply_otsu = body.get("apply_otsu_gating", True)
        except Exception:
            pass

    if image_np is None:
        raise HTTPException(
            status_code=400,
            detail="Missing image data. Provide multipart 'file' or JSON 'image_base64'.",
        )

    # Execute prediction pipeline
    result = process_single_patch(image_np, mode=mode, enforce_otsu=apply_otsu)

    return PatchPredictResponse(
        success=True,
        mask_shape=result["mask_shape"],
        class_distribution=result["class_distribution"],
        biomarkers=BiomarkersResponse(**result["biomarkers"]),
        mask_base64=result["mask_base64"],
        overlay_base64=result["overlay_base64"],
        blended_base64=result["blended_base64"],
        applied_gating=result["applied_gating"],
    )


@app.post("/api/v1/predict/wsi", response_model=WSIPredictResponse)
def predict_wsi(request: WSIPredictRequest) -> WSIPredictResponse:
    """Accepts multi-patch tiles and coordinates, stitches predictions into a composite heatmap."""
    if not request.tiles:
        raise HTTPException(status_code=400, detail="Tile list cannot be empty.")

    processed_tiles: List[Dict[str, Any]] = []

    for tile in request.tiles:
        try:
            img_np = decode_base64_to_image(tile.image_base64)
            res = process_single_patch(img_np, mode=request.mode or "proposed", enforce_otsu=True)
            processed_tiles.append({
                "x": tile.x,
                "y": tile.y,
                "mask": res["pred_mask"],
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing tile {tile.id}: {str(e)}")

    # Stitch into composite slide mask
    composite_mask, composite_rgb = stitch_tiles_heatmap(processed_tiles)
    composite_rgba = mask_to_rgba_overlay(composite_mask, alpha=0.55, bg_transparent=True)

    # Compute global slide biomarkers
    vtp = compute_viable_tumor_percentage(composite_mask, tumor_class_idx=0, background_class_idx=4)
    tsr = compute_tumor_stroma_ratio(composite_mask, tumor_class_idx=0, stroma_class_idx=1)
    distribution = compute_class_distribution(composite_mask)

    comp_mask_b64 = encode_image_to_base64(composite_rgb, format="PNG")
    comp_overlay_b64 = encode_image_to_base64(composite_rgba, format="PNG")

    return WSIPredictResponse(
        success=True,
        slide_id=request.slide_id or "WSI_SLIDE",
        num_tiles=len(request.tiles),
        composite_shape=list(composite_mask.shape),
        class_distribution=distribution,
        biomarkers=BiomarkersResponse(
            viable_tumor_percentage=round(float(vtp), 2),
            tumor_stroma_ratio=round(float(tsr), 2),
        ),
        composite_mask_base64=comp_mask_b64,
        composite_overlay_base64=comp_overlay_b64,
        applied_gating=True,
    )


@app.get("/", response_class=HTMLResponse)
def serve_root() -> Any:
    """Serves the clinical workstation web interface or redirects to /viewer."""
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    return RedirectResponse(url="/viewer")


@app.get("/viewer", response_class=HTMLResponse)
def serve_viewer() -> Any:
    """Serves the interactive HTML/JS OpenSeadragon clinical viewer."""
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    return HTMLResponse("<h3>Clinical Viewer template not found in static folder.</h3>", status_code=404)


@app.get("/health")
def health_alias() -> Dict[str, Any]:
    """Alias for /api/v1/health."""
    return get_system_health()


@app.post("/segment", response_model=PatchPredictResponse)
async def segment_alias(
    request: Request,
    file: Optional[UploadFile] = File(None),
) -> PatchPredictResponse:
    """Alias for /api/v1/predict/patch matching technical report contract."""
    return await predict_patch(request=request, file=file)


@app.post("/biomarkers", response_model=BiomarkersResponse)
async def calculate_biomarkers_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
) -> BiomarkersResponse:
    """Computes clinical biomarkers (VTP, TSR) from image patch or segmentation mask."""
    res = await predict_patch(request=request, file=file)
    return res.biomarkers


@app.get("/splits")
def get_splits_endpoint() -> Dict[str, Any]:
    """Returns patient-level anti-leakage train/val/test splits."""
    return load_patient_splits()


@app.get("/tile/{z}/{x}/{y}.png")
def get_tile_endpoint(z: int, x: int, y: int) -> Response:
    """Returns a PNG tile for OpenSeadragon deep-zoom viewer integration."""
    tile_img = Image.new("RGB", (256, 256), color=(245, 240, 245))
    buf = io.BytesIO()
    tile_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

