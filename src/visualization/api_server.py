"""
Clinical Digital Pathology Workstation & Inference REST API.
Provides endpoints for:
- GET  /api/v1/health         : Honest checkpoint state, active models, device, and VRAM usage
- GET  /api/v1/samples        : Held-out TCGA-BRCA histopathology cohort with pathologist GT masks
- GET  /api/v1/samples/{id}   : Raw H&E patch, Ground Truth mask, and GT biomarker reference
- POST /api/v1/predict/patch  : WSSS segmentation, visual verification overlays (blurred dual QA,
                                Green/Red match, contours, entropy uncertainty, Stage-1 CAM,
                                class spotlight), pixel-index maps, VTP & TSR biomarkers
- POST /api/v1/compare        : Multi-method A/B side-by-side comparison on the same tissue patch
- POST /api/v1/export/report  : Downloadable 300-DPI PNG Diagnostic Sheet, JSON, or CSV audit report
- POST /api/v1/predict/wsi    : Multi-patch WSI slide stitching and composite heatmap synthesis
- GET  /viewer and /          : Enterprise self-contained HTML5 clinical workstation UI
- Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

import base64
import csv
import io
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from src.colab_pkg.downloader import extract_patient_id
from src.datasets.otsu_filter import apply_otsu_gating_to_prediction
from src.datasets.patient_split import load_patient_splits
from src.losses_metrics.metrics import (
    compute_tumor_stroma_ratio,
    compute_viable_tumor_percentage,
)
from src.models.wsss_inference_methods import (
    DeepLabV3Plus,
    MultiMethodWSSSEngineV2,
    WSSSTissueStage1NetCAM,
    build_and_load_deeplabv3plus,
    build_and_load_stage1_netcam,
)
from src.models.wsss_tissue import WSSSBenchmarkingEngine
from src.visualization.clinical_viewer import (
    CLASS_COLORS_HEX,
    CLASS_NAMES,
    blend_overlay,
    compute_agreement_error_overlay,
    compute_blurred_dual_verification_overlay,
    compute_cam_overlay,
    compute_class_distribution,
    compute_contour_comparison_overlay,
    compute_dual_mask_blend,
    compute_entropy_heatmap_overlay,
    compute_per_class_spotlight,
    decode_base64_to_image,
    encode_image_to_base64,
    mask_to_color_image,
    mask_to_rgba_overlay,
    render_single_patch_diagnostic_sheet,
    stitch_tiles_heatmap,
)

# Directory configurations
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(MODULE_DIR).resolve().parent.parent
STATIC_DIR = os.path.join(MODULE_DIR, "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_NAME)

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_BASE64_CHARS = 24 * 1024 * 1024
MAX_WSI_TILES = 64

app = FastAPI(
    title="BCSS-WSSS Clinical Digital Pathology Workstation API",
    description="Weakly-Supervised Histopathology Semantic Segmentation, Visual Verification & Prognostic Biomarkers",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def _sanitize_validation_obj(obj: Any) -> Any:
    """Recursively converts non-finite floats and non-JSON primitives for safe 422 JSON responses."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_validation_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_validation_obj(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, BaseException):
        return str(obj)
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    sanitized_errors = _sanitize_validation_obj(exc.errors())
    safe_content = jsonable_encoder(
        {"detail": sanitized_errors},
        custom_encoder={float: lambda v: v if math.isfinite(v) else str(v)},
    )
    return JSONResponse(status_code=422, content=safe_content)


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

REAL_SOTA_METHODS = {
    "proposed",
    "m6",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m7",
    "sg_dap",
    "od_prior",
    "mpppc",
    "combined",
}
LEGACY_CAM_MODES = {"legacy_proposed", "relocalization", "classmix", "baseline"}


class ClinicalWorkstationInferenceManager:
    """Thread-safe model registry serving verified Stage-1/Stage-2 checkpoints and multi-method inference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self.model_stage1: Optional[WSSSTissueStage1NetCAM] = None
        self.model_stage2: Optional[DeepLabV3Plus] = None
        self.sota_engine: Optional[MultiMethodWSSSEngineV2] = None
        self.stage1_loaded = False
        self.stage2_loaded = False
        self.stage1_ckpt_path: Optional[str] = None
        self.stage2_ckpt_path: Optional[str] = None
        self._legacy_engines: Dict[str, WSSSBenchmarkingEngine] = {}
        self._ablation_eval: Any = None
        self._ablation_lock = threading.Lock()

    @property
    def real_checkpoints_loaded(self) -> bool:
        return bool(self.stage2_loaded)

    def _resolve_checkpoint_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        s2_candidates = [
            os.environ.get("WSSS_STAGE2_CKPT", ""),
            str(PROJECT_ROOT / "checkpoints" / "stage2_checkpoint_trained_on_bcss.pth"),
        ]
        s1_candidates = [
            os.environ.get("WSSS_STAGE1_CKPT", ""),
            os.environ.get("WSSS_CHECKPOINT_PATH", ""),
            str(PROJECT_ROOT / "checkpoints" / "stage1_checkpoint_trained_on_bcss.pth"),
        ]
        s2_p = next((Path(p) for p in s2_candidates if p and Path(p).is_file()), None)
        s1_p = next((Path(p) for p in s1_candidates if p and Path(p).is_file()), None)
        return s2_p, s1_p

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            s2_p, s1_p = self._resolve_checkpoint_paths()
            if s2_p is not None:
                try:
                    self.model_stage2 = build_and_load_deeplabv3plus(s2_p, device=DEVICE, strict=True)
                    self.stage2_loaded = True
                    self.stage2_ckpt_path = str(s2_p)
                except Exception:
                    self.model_stage2 = None
                    self.stage2_loaded = False

            if s1_p is not None:
                try:
                    self.model_stage1 = build_and_load_stage1_netcam(s1_p, device=DEVICE, strict=True)
                    self.stage1_loaded = True
                    self.stage1_ckpt_path = str(s1_p)
                except Exception:
                    self.model_stage1 = None
                    self.stage1_loaded = False

            self.sota_engine = MultiMethodWSSSEngineV2(model_stage1=self.model_stage1, use_amp=False)
            if self.stage2_loaded:
                self._get_ablation_evaluator()
            self._initialized = True

    def _get_legacy_engine(self, mode: str) -> WSSSBenchmarkingEngine:
        engine_mode = "proposed" if mode == "legacy_proposed" else mode
        if engine_mode not in ("proposed", "relocalization", "classmix", "baseline"):
            engine_mode = "proposed"
        if engine_mode not in self._legacy_engines:
            self._legacy_engines[engine_mode] = WSSSBenchmarkingEngine(
                mode=engine_mode,
                num_classes=5,
                device=DEVICE_NAME,
            )
        return self._legacy_engines[engine_mode]

    def _get_ablation_evaluator(self):
        if self._ablation_eval is not None:
            return self._ablation_eval
        with self._ablation_lock:
            if self._ablation_eval is not None:
                return self._ablation_eval
            from src.clinical.comparative_ablation_evaluator import ComparativeAblationEvaluator

            ev = ComparativeAblationEvaluator(feat_dim=256, device=DEVICE)
            head: Optional[torch.nn.Module] = None
            if (
                self.model_stage2 is not None
                and hasattr(self.model_stage2, "decoder")
                and hasattr(self.model_stage2.decoder, "last_conv")
                and len(self.model_stage2.decoder.last_conv) >= 9
            ):
                head = self.model_stage2.decoder.last_conv[8]
                ev.attach_classifier_head(head)
                if hasattr(head, "weight") and head.weight.shape[0] >= 4 and head.weight.shape[1] == 256:
                    with torch.no_grad():
                        w_norm = F.normalize(head.weight.detach()[:4, :256].reshape(4, 256).float(), dim=-1, eps=1e-6)
                        ev.mpppc.prototypes.copy_(
                            w_norm.unsqueeze(1).expand(4, ev.mpppc.num_prototypes, 256).clone().to(DEVICE)
                        )
                        ev.prototypes_calibrated = True

            # Deterministically calibrate MP-PPC sub-prototypes from unlabeled training patches (zero test leakage)
            train_candidates = [
                PROJECT_ROOT / "data" / "BCSS-WSSS" / "training",
                PROJECT_ROOT / "data" / "BCSS-WSSS" / "train" / "img",
            ]
            train_img_dir = next((c for c in train_candidates if c.is_dir()), None)
            if self.model_stage2 is not None and train_img_dir is not None:
                all_tr = sorted(train_img_dir.glob("*.png"))
                if all_tr:
                    cal_paths: List[Path] = []
                    seen_train_pids = set()
                    step_tr = max(1, len(all_tr) // 400)
                    for p in all_tr[::step_tr]:
                        tpid = extract_patient_id(p.name)
                        if tpid not in seen_train_pids:
                            cal_paths.append(p)
                            seen_train_pids.add(tpid)
                        if len(cal_paths) >= 16:
                            break
                    if cal_paths:
                        cal_hook_box: Dict[str, torch.Tensor] = {}
                        cal_hook = None
                        if head is not None:
                            def _cal_cap(_m: torch.nn.Module, inp: Tuple[torch.Tensor, ...], _o: torch.Tensor) -> None:
                                cal_hook_box["feat"] = inp[0].detach()

                            cal_hook = head.register_forward_hook(_cal_cap)
                        try:
                            cal_feats: List[torch.Tensor] = []
                            cal_probs: List[torch.Tensor] = []
                            with torch.no_grad():
                                for i in range(0, len(cal_paths), 8):
                                    batch_imgs = []
                                    for tr_p in cal_paths[i : i + 8]:
                                        tr_rgb = np.array(Image.open(tr_p).convert("RGB"), dtype=np.uint8)
                                        batch_imgs.append(
                                            torch.from_numpy(tr_rgb.transpose(2, 0, 1)).float().div(255.0)
                                        )
                                    tr_batch = torch.stack(batch_imgs, dim=0).to(DEVICE)
                                    cal_hook_box.clear()
                                    logits = self.model_stage2(tr_batch)[:, :4]
                                    cal_probs.append(F.softmax(logits, dim=1).detach().cpu())
                                    feat = cal_hook_box.get(
                                        "feat",
                                        F.interpolate(tr_batch, size=(56, 56), mode="bilinear", align_corners=False),
                                    )
                                    cal_feats.append(feat.detach().cpu())
                            cal_count = ev.fit_anchor_prototypes(cal_feats, cal_probs, conf_thresh=0.85)
                            if cal_count == 0 and head is not None:
                                ev.prototypes_calibrated = True
                        finally:
                            if cal_hook is not None:
                                cal_hook.remove()

            self._ablation_eval = ev
        return self._ablation_eval

    def available_methods(self) -> List[str]:
        if self.stage2_loaded:
            return ["m6", "m1", "m2", "m3", "m4", "m5", "m7", "od_prior", "sg_dap", "mpppc", "combined"]
        return ["proposed", "relocalization", "classmix", "baseline"]

    @torch.no_grad()
    def predict(
        self,
        down_np: np.ndarray,
        mode: str = "m6",
        enforce_otsu: bool = True,
        strict_weights: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, str, bool]:
        """Runs patch inference and returns (pred_mask, probs_4cls_or_5cls, resolved_method, ckpt_verified)."""
        pred_mask, probs_arr, method_key, ckpt_verified, _ = self.predict_with_aux(
            down_np=down_np,
            mode=mode,
            enforce_otsu=enforce_otsu,
            strict_weights=strict_weights,
            precomputed_cache=None,
        )
        return pred_mask, probs_arr, method_key, ckpt_verified

    @torch.no_grad()
    def predict_with_aux(
        self,
        down_np: np.ndarray,
        mode: str = "m6",
        enforce_otsu: bool = True,
        strict_weights: bool = False,
        precomputed_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str, bool, Dict[str, Any]]:
        """Runs patch inference and returns (pred_mask, probs, resolved_method, ckpt_verified, aux)."""
        self.ensure_initialized()
        req_mode = (mode or "m6").lower().strip()

        if req_mode not in REAL_SOTA_METHODS and req_mode not in LEGACY_CAM_MODES:
            req_mode = "m6" if self.stage2_loaded else "proposed"

        if strict_weights and not self.stage2_loaded:
            raise HTTPException(
                status_code=503,
                detail="Verified Stage-2 DeepLabV3+ checkpoint not loaded on server (strict_weights=True).",
            )

        # Normalize to WSSS-Tissue [0.0, 1.0] standard and move onto inference device
        img_float = down_np.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        if self.stage2_loaded and self.model_stage2 is not None and req_mode not in LEGACY_CAM_MODES:
            engine = self.sota_engine
            assert engine is not None
            h, w = down_np.shape[:2]

            # Alias "proposed" -> "m6" when real checkpoints are loaded
            method_key = "m6" if req_mode == "proposed" else req_mode

            need_ms_d4 = method_key in ("m6", "m7", "od_prior", "sg_dap", "mpppc", "combined")
            need_s2_feat = method_key in ("od_prior", "sg_dap", "mpppc", "combined")
            s2_feat: Optional[torch.Tensor] = None
            probs_ms_d4: Optional[np.ndarray] = None
            cache_key = (h, w)

            with self._lock:
                cached_s2 = precomputed_cache.get(("s2", cache_key)) if precomputed_cache is not None else None
                if cached_s2 is not None and (not need_s2_feat or cached_s2[2] is not None):
                    probs_orig, probs_d4, s2_feat = cached_s2
                else:
                    local_hook_box: Dict[str, torch.Tensor] = {}
                    local_hook_handle: Any = None
                    if (
                        (need_s2_feat or precomputed_cache is not None)
                        and hasattr(self.model_stage2, "decoder")
                        and hasattr(self.model_stage2.decoder, "last_conv")
                        and len(self.model_stage2.decoder.last_conv) >= 9
                    ):
                        def _cap(_m: torch.nn.Module, inp: Tuple[torch.Tensor, ...], _o: torch.Tensor) -> None:
                            local_hook_box["feat"] = inp[0][:1].detach()

                        local_hook_handle = self.model_stage2.decoder.last_conv[8].register_forward_hook(_cap)
                    try:
                        probs_orig, probs_d4 = engine.forward_d4_tta(
                            self.model_stage2, img_tensor, use_amp=False
                        )
                    finally:
                        if local_hook_handle is not None:
                            local_hook_handle.remove()

                    s2_feat = local_hook_box.get("feat")
                    if s2_feat is not None and s2_feat.shape[0] != 1:
                        s2_feat = s2_feat[:1]
                    if precomputed_cache is not None:
                        precomputed_cache[("s2", cache_key)] = (probs_orig, probs_d4, s2_feat)

                if need_ms_d4:
                    cached_ms = precomputed_cache.get(("ms", cache_key)) if precomputed_cache is not None else None
                    if cached_ms is not None:
                        probs_ms_d4 = cached_ms
                    else:
                        _, _, probs_ms_d4 = engine.forward_multiscale_d4_tta(
                            self.model_stage2, img_tensor, precomputed_s1_d4=(probs_orig, probs_d4)
                        )
                        if precomputed_cache is not None:
                            precomputed_cache[("ms", cache_key)] = probs_ms_d4

            probs_4cls = probs_orig
            pred_raw: Optional[np.ndarray] = None
            cam_out: Optional[np.ndarray] = None
            cam_conf_out: Optional[np.ndarray] = None

            def _get_s1_v1() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                if precomputed_cache is not None and ("s1_v1", cache_key) in precomputed_cache:
                    return precomputed_cache[("s1_v1", cache_key)]
                res_v1 = engine.extract_stage1_cam_and_gate(self.model_stage1, img_tensor, (h, w))
                if precomputed_cache is not None:
                    precomputed_cache[("s1_v1", cache_key)] = res_v1
                return res_v1

            def _get_s1_v2() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
                if precomputed_cache is not None and ("s1_v2", cache_key) in precomputed_cache:
                    return precomputed_cache[("s1_v2", cache_key)]
                res_v2 = engine.extract_stage1_adaptive_cam_and_gate(self.model_stage1, img_tensor, (h, w))
                if precomputed_cache is not None:
                    precomputed_cache[("s1_v2", cache_key)] = res_v2
                return res_v2

            if method_key == "m1":
                probs_4cls = probs_orig
                pred_raw = np.argmax(probs_4cls, axis=0).astype(np.uint8)
                if self.model_stage1 is not None:
                    cam_v2_opt, _, cam_conf_opt = _get_s1_v2()
                    cam_out = cam_v2_opt
                    cam_conf_out = cam_conf_opt
            elif method_key == "m2":
                cam_p, cgm_gate = _get_s1_v1()
                cam_out = cam_p
                probs_4cls = engine.apply_cgm_gate(probs_d4, cgm_gate)
                pred_raw = np.argmax(probs_4cls, axis=0).astype(np.uint8)
            elif method_key == "m3":
                cam_p, cgm_gate = _get_s1_v1()
                cam_out = cam_p
                probs_m2 = engine.apply_cgm_gate(probs_d4, cgm_gate)
                probs_4cls = engine.apply_prior_calibration(engine.apply_cam_fusion(probs_m2, cam_p))
                pred_raw = np.argmax(probs_4cls, axis=0).astype(np.uint8)
            elif method_key == "m4":
                cam_p, cgm_gate = _get_s1_v1()
                cam_out = cam_p
                probs_4cls = engine.apply_cgm_gate(probs_d4, cgm_gate)
                pred_raw = engine.apply_spatial_refinement(probs_4cls, raw_rgb=down_np)
            elif method_key == "m5":
                cam_p, cgm_gate = _get_s1_v1()
                cam_out = cam_p
                probs_m2 = engine.apply_cgm_gate(probs_d4, cgm_gate)
                probs_4cls = engine.apply_prior_calibration(engine.apply_cam_fusion(probs_m2, cam_p))
                pred_raw = engine.apply_spatial_refinement(probs_4cls, raw_rgb=down_np)
            elif method_key in ("m6", "m7", "od_prior", "sg_dap", "mpppc", "combined"):
                assert probs_ms_d4 is not None
                cam_v2, soft_gate, cam_conf = _get_s1_v2()
                cam_out = cam_v2
                cam_conf_out = cam_conf
                probs_gated = engine.apply_cgm_gate(probs_ms_d4, soft_gate)

                if method_key == "m6":
                    probs_4cls = engine.apply_prior_calibration(
                        engine.apply_cam_fusion(probs_gated, cam_v2, alpha=0.22)
                    )
                    pred_raw = np.argmax(probs_4cls, axis=0).astype(np.uint8)
                elif method_key == "m7":
                    probs_fused = engine.apply_entropy_guided_cam_fusion(
                        probs_gated, cam_v2, cam_conf=cam_conf
                    )
                    probs_4cls = engine.apply_he_optical_density_calibration(
                        probs_fused, raw_rgb=down_np, cgm_gate=soft_gate
                    )
                    pred_raw = engine.apply_advanced_spatial_refinement(probs_4cls, raw_rgb=down_np)
                else:
                    # Composable experimental directions built on top of real M6 features & probabilities
                    probs_m6 = engine.apply_prior_calibration(
                        engine.apply_cam_fusion(probs_gated, cam_v2, alpha=0.22)
                    )
                    ablation_eval = self._get_ablation_evaluator()
                    if s2_feat is None:
                        s2_feat = F.interpolate(
                            img_tensor.repeat(1, 86, 1, 1)[:, :256],
                            size=(max(1, h // 4), max(1, w // 4)),
                            mode="bilinear",
                            align_corners=False,
                        )
                    init_cam_t = torch.from_numpy(probs_m6).unsqueeze(0).to(DEVICE)
                    if method_key == "od_prior":
                        steps = ("od_prior",)
                    elif method_key == "sg_dap":
                        steps = ("sg_dap",)
                    elif method_key == "mpppc":
                        steps = ("sg_dap", "mpppc")
                    else:
                        steps = ("od_prior", "sg_dap", "mpppc")
                    refined_t, pred_t = ablation_eval.compose_pipeline(
                        rgb_patch=img_tensor,
                        init_cam=init_cam_t,
                        backbone_feats=s2_feat,
                        steps=steps,
                    )
                    probs_4cls = refined_t.squeeze(0).cpu().numpy().astype(np.float32)
                    pred_raw = pred_t.squeeze(0).cpu().numpy().astype(np.uint8)

            assert pred_raw is not None
            pred_mask = (
                engine.apply_pure_glass_guard(pred_raw, down_np, background_class_idx=4)
                if enforce_otsu
                else pred_raw.astype(np.int64)
            )
            aux = {"cam": cam_out, "cam_conf": cam_conf_out}
            return pred_mask.astype(np.int64), probs_4cls, method_key, True, aux

        # Fallback to legacy Stage-1 CAM benchmarking engine (only when requested or Stage-2 absent)
        legacy_engine = self._get_legacy_engine(req_mode)
        pred_mask = legacy_engine.generate_pseudo_mask(
            image_tensor=img_tensor,
            current_epoch=5,
            raw_image=down_np if enforce_otsu else None,
        )
        # Synthesize smooth probability simplex from binary mask + last_uncertainty_map for entropy readout
        unc = getattr(legacy_engine, "last_uncertainty_map", None)
        h_m, w_m = pred_mask.shape
        probs_5 = np.full((5, h_m, w_m), 0.02, dtype=np.float32)
        for cls_i in range(5):
            probs_5[cls_i, pred_mask == cls_i] = 0.92
        if unc is not None and unc.shape == (h_m, w_m):
            unc_clip = np.clip(unc, 0.0, 0.9)
            probs_5 = probs_5 * (1.0 - 0.6 * unc_clip[None, ...]) + 0.12 * unc_clip[None, ...]
        probs_5 /= np.maximum(probs_5.sum(axis=0, keepdims=True), 1e-6)
        return pred_mask.astype(np.int64), probs_5, f"legacy_{legacy_engine.mode}", False, {"cam": None, "cam_conf": None}


_manager = ClinicalWorkstationInferenceManager()
_sample_catalog_cache: Optional[List[Dict[str, Any]]] = None


def _encode_gray_l_png(arr_u8: np.ndarray) -> str:
    """Encodes a 2D uint8 array as a lossless single-channel 'L' PNG Data URL."""
    pil_img = Image.fromarray(np.asarray(arr_u8, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _discover_clinical_samples() -> List[Dict[str, Any]]:
    """Discovers diverse held-out BCSS test patches (or sample_dataset fallback) with Ground Truth."""
    global _sample_catalog_cache
    if _sample_catalog_cache is not None:
        return _sample_catalog_cache

    candidates = [
        (PROJECT_ROOT / "data" / "BCSS-WSSS" / "test" / "img", PROJECT_ROOT / "data" / "BCSS-WSSS" / "test" / "mask"),
        (PROJECT_ROOT / "data" / "sample_dataset" / "test" / "img", PROJECT_ROOT / "data" / "sample_dataset" / "test" / "mask"),
    ]
    selected: List[Dict[str, Any]] = []
    for img_dir, mask_dir in candidates:
        if not img_dir.is_dir():
            continue
        pngs = sorted(img_dir.glob("*.png"))
        if not pngs:
            continue

        # Scan across the full test cohort so every patient is represented cleanly without sliver bias
        step = max(1, len(pngs) // 480)
        by_patient: Dict[str, List[Tuple[int, float, Path, Optional[Path], List[str]]]] = {}
        for p in pngs[::step]:
            pid = extract_patient_id(p.name)
            mp = mask_dir / p.name
            if mp.is_file():
                m_arr = np.array(Image.open(mp), dtype=np.int64)
                fg_shares = [float((m_arr == c).mean()) for c in range(4)]
                uniq_major = [c for c in range(4) if fg_shares[c] >= 0.12]
                uniq_minor = [c for c in range(4) if 0.05 <= fg_shares[c] < 0.12]
                tiny_slivers = [c for c in range(4) if 0.002 <= fg_shares[c] < 0.08]
                uniq_all = [c for c in range(4) if fg_shares[c] >= 0.05]
                present = [CLASS_NAMES[x] for x in uniq_all]
                back_mask = m_arr == 4
                back_frac = float(np.mean(back_mask))
                true_glass_frac = 1.0
                if back_frac > 0.05:
                    rgb_arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
                    gray = np.mean(rgb_arr, axis=2)
                    true_glass_frac = float(np.mean(gray[back_mask] >= 235))
                    if true_glass_frac >= 0.55 and back_frac >= 0.10:
                        present.append("BACK")
                if not present:
                    present = ["BACK"]

                max_fg = max(fg_shares) if fg_shares else 0.0
                outside_roi_penalty = -60 if (back_frac > 0.05 and true_glass_frac < 0.45) else 0
                skew_penalty = -16 if max_fg > 0.78 else 0
                sliver_penalty = -6 * len(tiny_slivers)
                roi_bonus = 15 if (1.0 - back_frac) >= 0.55 else 0
                glass_bonus = 12 if (0.12 <= back_frac <= 0.55 and true_glass_frac >= 0.65) else 0
                balance_bonus = int((1.0 - max_fg) * 20)
                score = (
                    len(uniq_major) * 14
                    + len(uniq_minor) * 5
                    + balance_bonus
                    + roi_bonus
                    + glass_bonus
                    + outside_roi_penalty
                    + skew_penalty
                    + sliver_penalty
                )
            else:
                present = []
                back_frac = 0.0
                score = 1
            by_patient.setdefault(pid, []).append((score, back_frac, p, mp if mp.is_file() else None, present))

        ranked_patients = sorted(
            by_patient.keys(),
            key=lambda k: (-max(x[0] for x in by_patient[k]), k),
        )
        for idx, pid in enumerate(ranked_patients):
            best_item = sorted(by_patient[pid], key=lambda item: (-item[0], item[2].name))[0]
            _, back_frac, img_p, mask_p, present = best_item
            phenotype = "+".join([c for c in present if c != "BACK"]) or "Glass/Background"
            if back_frac >= 0.10:
                phenotype += " + Glass Margin"
            selected.append(
                {
                    "sample_id": img_p.stem,
                    "filename": img_p.name,
                    "patient_id": pid,
                    "present_classes": present,
                    "morphology": phenotype,
                    "label": f"Case {idx + 1}: {pid} ({phenotype})",
                    "gt_available": mask_p is not None,
                    "_img_path": str(img_p),
                    "_mask_path": str(mask_p) if mask_p is not None else None,
                }
            )
        if selected:
            break

    _sample_catalog_cache = selected
    return _sample_catalog_cache


def _get_sample_record(sample_id: str) -> Dict[str, Any]:
    for item in _discover_clinical_samples():
        if item["sample_id"] == sample_id or item["filename"] == sample_id:
            return item
    raise HTTPException(status_code=404, detail=f"Sample ID '{sample_id}' not found in test catalog.")


def _load_mask_array(path_str: str) -> np.ndarray:
    m_img = Image.open(path_str)
    arr = np.array(m_img if m_img.mode == "P" else m_img.convert("L"), dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _decode_gt_base64(gt_b64: str) -> np.ndarray:
    if len(gt_b64) > MAX_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="GT mask payload exceeds 16 MB limit.")
    if "," in gt_b64:
        gt_b64 = gt_b64.split(",", 1)[1]
    raw_bytes = io.BytesIO(base64.b64decode(gt_b64))
    img = Image.open(raw_bytes)
    arr = np.array(img, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


# -------------------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------------------

class BiomarkersResponse(BaseModel):
    viable_tumor_percentage: float = Field(..., description="VTP percentage [0.0, 100.0]")
    tumor_stroma_ratio: float = Field(..., description="TSR percentage [0.0, 100.0]")


class PatchPredictJSONRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    image_base64: Optional[str] = Field(None, description="RGB image encoded in base64 or Data URL")
    sample_id: Optional[str] = Field(None, description="Held-out cohort sample ID from /api/v1/samples")
    gt_mask_base64: Optional[str] = Field(None, description="Optional index GT mask for visual error overlay")
    mode: Optional[str] = Field("m6", description="Algorithm mode (m6, m1..m7, od_prior, sg_dap, mpppc, combined)")
    apply_otsu_gating: Optional[bool] = Field(True, description="Enforce Pure-Glass Otsu gating")
    target_class: Optional[int] = Field(0, ge=0, le=4, description="Target class [0..4] for TP/FP/FN spotlight")
    strict_weights: Optional[bool] = Field(False, description="Require verified Stage-2 checkpoint (503 if absent)")
    blur_sigma: Optional[float] = Field(4.0, ge=0.0, le=25.0)
    dim_factor: Optional[float] = Field(0.55, ge=0.0, le=1.0)
    gt_alpha: Optional[float] = Field(0.45, ge=0.0, le=1.0)
    pred_alpha: Optional[float] = Field(0.55, ge=0.0, le=1.0)
    error_alpha: Optional[float] = Field(0.65, ge=0.0, le=1.0)
    tolerance_px: Optional[int] = Field(2, ge=0, le=10)
    error_breakdown: Optional[bool] = Field(False)


class PatchPredictResponse(BaseModel):
    success: bool
    mask_shape: List[int]
    class_distribution: Dict[str, float]
    biomarkers: BiomarkersResponse
    mask_base64: str
    overlay_base64: str
    blended_base64: Optional[str] = None
    applied_gating: bool
    method_used: Optional[str] = None
    checkpoint_verified: bool = False
    latency_ms: float = 0.0
    mean_entropy: float = 0.0
    entropy_overlay_base64: Optional[str] = None
    contour_overlay_base64: Optional[str] = None
    gt_color_base64: Optional[str] = None
    agreement_overlay_base64: Optional[str] = None
    dual_blend_base64: Optional[str] = None
    blurred_dual_overlay_base64: Optional[str] = None
    cam_overlay_base64: Optional[str] = None
    pred_mask_index_base64: Optional[str] = None
    gt_mask_index_base64: Optional[str] = None
    confidence_map_base64: Optional[str] = None
    gt_distribution: Optional[Dict[str, float]] = None
    gt_biomarkers: Optional[BiomarkersResponse] = None
    spotlight_overlay_base64: Optional[str] = None
    spotlight_stats: Optional[Dict[str, Any]] = None
    agreement_summary: Optional[Dict[str, Any]] = None
    verification_metrics: Optional[Dict[str, Any]] = None


class MethodCompareRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    sample_id: Optional[str] = None
    image_base64: Optional[str] = None
    gt_mask_base64: Optional[str] = None
    methods: List[str] = Field(default_factory=lambda: ["m1", "m6", "sg_dap", "combined"])
    apply_otsu_gating: bool = True
    strict_weights: bool = False
    blur_sigma: float = Field(4.0, ge=0.0, le=25.0)
    dim_factor: float = Field(0.55, ge=0.0, le=1.0)
    gt_alpha: float = Field(0.45, ge=0.0, le=1.0)
    pred_alpha: float = Field(0.55, ge=0.0, le=1.0)
    error_alpha: float = Field(0.65, ge=0.0, le=1.0)
    tolerance_px: int = Field(2, ge=0, le=10)
    error_breakdown: bool = False


class ExportReportRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    sample_id: Optional[str] = None
    image_base64: Optional[str] = None
    gt_mask_base64: Optional[str] = None
    mode: Optional[str] = "m6"
    apply_otsu_gating: bool = True
    target_class: int = Field(0, ge=0, le=4)
    blur_sigma: float = Field(4.0, ge=0.0, le=25.0)
    dim_factor: float = Field(0.55, ge=0.0, le=1.0)
    gt_alpha: float = Field(0.45, ge=0.0, le=1.0)
    pred_alpha: float = Field(0.55, ge=0.0, le=1.0)
    error_alpha: float = Field(0.65, ge=0.0, le=1.0)
    tolerance_px: int = Field(2, ge=0, le=10)
    format: str = "png"


class WSITileItem(BaseModel):
    id: Optional[str] = "tile_0"
    x: int = Field(default=0, ge=0)
    y: int = Field(default=0, ge=0)
    image_base64: str


class WSIPredictRequest(BaseModel):
    slide_id: Optional[str] = "WSI_SLIDE"
    tiles: List[WSITileItem]
    mode: Optional[str] = "m6"


class WSIPredictResponse(BaseModel):
    success: bool
    slide_id: str
    num_tiles: int
    composite_shape: List[int]
    class_distribution: Dict[str, float]
    biomarkers: BiomarkersResponse
    composite_mask_base64: str
    composite_overlay_base64: str
    composite_heatmap_base64: Optional[str] = None
    region_summary: Optional[Dict[str, Any]] = None
    applied_gating: bool


# -------------------------------------------------------------------------
# Core Helper Logic
# -------------------------------------------------------------------------

def process_single_patch(
    image_np: np.ndarray,
    mode: str = "proposed",
    enforce_otsu: bool = True,
    gt_mask: Optional[np.ndarray] = None,
    target_class: int = 0,
    strict_weights: bool = False,
    blur_sigma: float = 4.0,
    dim_factor: float = 0.55,
    gt_alpha: float = 0.45,
    pred_alpha: float = 0.55,
    error_alpha: float = 0.65,
    tolerance_px: int = 2,
    error_breakdown: bool = False,
    precomputed_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runs WSSS inference with optional GT visual verification overlays and clinical biomarkers."""
    h_orig, w_orig = image_np.shape[:2]

    max_dim = max(h_orig, w_orig)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        new_w = max(16, int(round(w_orig * scale)))
        new_h = max(16, int(round(h_orig * scale)))
        pil_down = Image.fromarray(image_np).resize((new_w, new_h), resample=Image.BILINEAR)
        down_np = np.array(pil_down, dtype=np.uint8)
    else:
        down_np = image_np

    t0 = time.perf_counter()
    down_mask, probs_arr, resolved_method, ckpt_verified, aux = _manager.predict_with_aux(
        down_np=down_np,
        mode=mode,
        enforce_otsu=enforce_otsu,
        strict_weights=strict_weights,
        precomputed_cache=precomputed_cache,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    cam_aligned: Optional[np.ndarray] = aux.get("cam")
    if (down_mask.shape[0], down_mask.shape[1]) != (h_orig, w_orig):
        pil_mask = Image.fromarray(down_mask.astype(np.uint8)).resize((w_orig, h_orig), resample=Image.NEAREST)
        pred_mask = np.array(pil_mask, dtype=np.int64)
        probs_t = torch.from_numpy(probs_arr).unsqueeze(0).float()
        probs_up = F.interpolate(probs_t, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        probs_arr = probs_up.squeeze(0).numpy()
        if cam_aligned is not None and cam_aligned.ndim == 3:
            cam_t = torch.from_numpy(cam_aligned).unsqueeze(0).float()
            cam_up = F.interpolate(cam_t, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
            cam_aligned = cam_up.squeeze(0).numpy()
    else:
        pred_mask = down_mask

    if enforce_otsu:
        if ckpt_verified:
            if (down_mask.shape[0], down_mask.shape[1]) != (h_orig, w_orig):
                pred_mask = MultiMethodWSSSEngineV2.apply_pure_glass_guard(
                    pred_mask, image_np, background_class_idx=4
                )
        else:
            pred_mask = apply_otsu_gating_to_prediction(
                pred_mask=pred_mask,
                image=image_np,
                background_class_idx=4,
            )

    vtp = compute_viable_tumor_percentage(pred_mask, tumor_class_idx=0, background_class_idx=4)
    tsr = compute_tumor_stroma_ratio(pred_mask, tumor_class_idx=0, stroma_class_idx=1)
    distribution = compute_class_distribution(pred_mask)

    mask_rgb = mask_to_color_image(pred_mask)
    mask_rgba = mask_to_rgba_overlay(pred_mask, alpha=0.55, bg_transparent=True)
    blended = blend_overlay(image_np, pred_mask, alpha=0.45)

    entropy_rgb, mean_entropy = compute_entropy_heatmap_overlay(image_np, probs_arr, alpha=0.55)
    cam_b64: Optional[str] = None
    if cam_aligned is not None:
        cam_rgb, _ = compute_cam_overlay(
            image_np,
            cam_aligned,
            target_class=target_class if 0 <= int(target_class) < 4 else None,
            alpha=0.60,
            dim_factor=dim_factor,
        )
        cam_b64 = encode_image_to_base64(cam_rgb, format="PNG")

    pred_index_b64 = _encode_gray_l_png(np.clip(pred_mask, 0, 4).astype(np.uint8))
    p_max = np.max(probs_arr, axis=0)
    conf_u8 = np.clip(np.round(p_max * 255.0), 0, 255).astype(np.uint8)
    conf_map_b64 = _encode_gray_l_png(conf_u8)

    # Align optional Ground Truth mask to patch shape and render visual verification suite
    gt_aligned: Optional[np.ndarray] = None
    if gt_mask is not None:
        gt_aligned = np.asarray(gt_mask, dtype=np.int64)
        if gt_aligned.shape[:2] != (h_orig, w_orig):
            gt_aligned = np.array(
                Image.fromarray(gt_aligned.astype(np.uint8)).resize((w_orig, h_orig), resample=Image.NEAREST),
                dtype=np.int64,
            )

    contour_rgb = compute_contour_comparison_overlay(image_np, pred_mask, gt_aligned)
    spotlight_rgb, spotlight_stats = compute_per_class_spotlight(
        image_np, pred_mask, gt_aligned, target_class=target_class
    )

    gt_color_b64: Optional[str] = None
    gt_index_b64: Optional[str] = None
    gt_dist: Optional[Dict[str, float]] = None
    gt_bio: Optional[Dict[str, float]] = None
    agree_b64: Optional[str] = None
    dual_blend_b64: Optional[str] = None
    blurred_dual_b64: Optional[str] = None
    verif_metrics: Optional[Dict[str, Any]] = None

    if gt_aligned is not None:
        gt_rgb = mask_to_color_image(gt_aligned)
        agree_rgb, verif_metrics = compute_agreement_error_overlay(
            image_np,
            pred_mask,
            gt_aligned,
            tolerance_px=tolerance_px,
            error_breakdown=error_breakdown,
        )
        dual_rgb = compute_dual_mask_blend(image_np, pred_mask, gt_aligned)
        blurred_rgb = compute_blurred_dual_verification_overlay(
            image_np,
            pred_mask,
            gt_aligned,
            blur_sigma=blur_sigma,
            dim_factor=dim_factor,
            gt_alpha=gt_alpha,
            pred_alpha=pred_alpha,
            error_alpha=error_alpha,
            tolerance_px=tolerance_px,
        )
        gt_color_b64 = encode_image_to_base64(gt_rgb, format="PNG")
        gt_index_b64 = _encode_gray_l_png(np.clip(gt_aligned, 0, 255).astype(np.uint8))
        gt_dist = compute_class_distribution(gt_aligned)
        gt_bio = {
            "viable_tumor_percentage": round(float(compute_viable_tumor_percentage(gt_aligned)), 2),
            "tumor_stroma_ratio": round(float(compute_tumor_stroma_ratio(gt_aligned)), 2),
        }
        agree_b64 = encode_image_to_base64(agree_rgb, format="PNG")
        dual_blend_b64 = encode_image_to_base64(dual_rgb, format="PNG")
        blurred_dual_b64 = encode_image_to_base64(blurred_rgb, format="PNG")

    return {
        "pred_mask": pred_mask,
        "_probs_arr": probs_arr,
        "_cam_aligned": cam_aligned,
        "_gt_aligned": gt_aligned,
        "mask_shape": list(pred_mask.shape),
        "class_distribution": distribution,
        "biomarkers": {
            "viable_tumor_percentage": round(float(vtp), 2),
            "tumor_stroma_ratio": round(float(tsr), 2),
        },
        "mask_base64": encode_image_to_base64(mask_rgb, format="PNG"),
        "overlay_base64": encode_image_to_base64(mask_rgba, format="PNG"),
        "blended_base64": encode_image_to_base64(blended, format="PNG"),
        "applied_gating": enforce_otsu,
        "method_used": resolved_method,
        "checkpoint_verified": ckpt_verified,
        "latency_ms": round(float(latency_ms), 2),
        "mean_entropy": mean_entropy,
        "entropy_overlay_base64": encode_image_to_base64(entropy_rgb, format="PNG"),
        "contour_overlay_base64": encode_image_to_base64(contour_rgb, format="PNG"),
        "gt_color_base64": gt_color_b64,
        "agreement_overlay_base64": agree_b64,
        "dual_blend_base64": dual_blend_b64,
        "blurred_dual_overlay_base64": blurred_dual_b64,
        "cam_overlay_base64": cam_b64,
        "pred_mask_index_base64": pred_index_b64,
        "gt_mask_index_base64": gt_index_b64,
        "confidence_map_base64": conf_map_b64,
        "gt_distribution": gt_dist,
        "gt_biomarkers": gt_bio,
        "spotlight_overlay_base64": encode_image_to_base64(spotlight_rgb, format="PNG"),
        "spotlight_stats": spotlight_stats,
        "agreement_summary": verif_metrics,
        "verification_metrics": verif_metrics,
    }


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/api/v1/health")
def get_system_health() -> Dict[str, Any]:
    """Returns honest system status, verified checkpoint state, available methods, and VRAM usage."""
    _manager.ensure_initialized()

    cuda_avail = torch.cuda.is_available()
    vram_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0) if cuda_avail else 0.0

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "mode": "m6" if _manager.stage2_loaded else "legacy_fallback",
            "device": DEVICE_NAME,
            "num_classes": 5,
            "classes": CLASS_NAMES,
            "palette": CLASS_COLORS_HEX,
            "otsu_gating": True,
            "loaded": _manager.real_checkpoints_loaded,
            "stage1_loaded": _manager.stage1_loaded,
            "stage2_loaded": _manager.stage2_loaded,
            "prototypes_calibrated": bool(
                _manager._ablation_eval is not None
                and getattr(_manager._ablation_eval, "prototypes_calibrated", False)
            ),
            "weights_source": (
                "pretrained_bcss_stage1_stage2" if _manager.real_checkpoints_loaded else "untrained_fallback"
            ),
            "available_methods": _manager.available_methods(),
        },
        "system": {
            "torch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "vram_allocated_mb": round(vram_mb, 2),
            "os": os.name,
        },
    }


@app.get("/api/v1/samples")
def list_clinical_samples(
    patient_id: Optional[str] = None,
    morphology: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Lists curated real held-out BCSS test samples with optional filtering & pagination."""
    items = _discover_clinical_samples()
    filtered = [
        {k: v for k, v in item.items() if not k.startswith("_")}
        for item in items
    ]
    if patient_id:
        pid_q = patient_id.strip().lower()
        filtered = [
            it for it in filtered
            if pid_q in it.get("patient_id", "").lower() or pid_q in it.get("sample_id", "").lower()
        ]
    if morphology:
        morph_q = morphology.strip().lower()
        filtered = [
            it for it in filtered
            if morph_q in it.get("label", "").lower()
            or morph_q in it.get("morphology", "").lower()
            or any(morph_q in cls_name.lower() for cls_name in it.get("present_classes", []))
        ]
    total = len(filtered)
    paged = filtered[offset : offset + limit] if limit is not None else filtered[offset:]
    return {"count": len(paged), "total": total, "samples": paged}


@app.get("/api/v1/samples/{sample_id}")
def get_clinical_sample_detail(sample_id: str) -> Dict[str, Any]:
    """Returns raw H&E image, pathologist Ground Truth mask, and GT biomarkers for a sample ID."""
    rec = _get_sample_record(sample_id)
    img_np = np.array(Image.open(rec["_img_path"]).convert("RGB"), dtype=np.uint8)

    gt_b64: Optional[str] = None
    gt_color_b64: Optional[str] = None
    gt_dist: Optional[Dict[str, float]] = None
    gt_bio: Optional[Dict[str, float]] = None

    if rec["_mask_path"] and os.path.isfile(rec["_mask_path"]):
        gt_arr = _load_mask_array(rec["_mask_path"])
        gt_b64 = _encode_gray_l_png(np.clip(gt_arr, 0, 255).astype(np.uint8))
        gt_color_b64 = encode_image_to_base64(mask_to_color_image(gt_arr), format="PNG")
        gt_dist = compute_class_distribution(gt_arr)
        gt_bio = {
            "viable_tumor_percentage": round(float(compute_viable_tumor_percentage(gt_arr)), 2),
            "tumor_stroma_ratio": round(float(compute_tumor_stroma_ratio(gt_arr)), 2),
        }

    return {
        "sample_id": rec["sample_id"],
        "filename": rec["filename"],
        "patient_id": rec["patient_id"],
        "present_classes": rec["present_classes"],
        "morphology": rec.get("morphology"),
        "image_base64": encode_image_to_base64(img_np, format="PNG"),
        "gt_mask_base64": gt_b64,
        "gt_color_base64": gt_color_b64,
        "gt_distribution": gt_dist,
        "gt_biomarkers": gt_bio,
    }


def _parse_json_float(
    body: Dict[str, Any],
    key: str,
    default: float,
    min_val: float,
    max_val: float,
) -> float:
    raw = body.get(key)
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise HTTPException(status_code=422, detail=f"Invalid '{key}': must be a number.")
    val = float(raw)
    if not math.isfinite(val) or val < min_val or val > max_val:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid '{key}': must be a finite number in [{min_val}, {max_val}].",
        )
    return val


async def _predict_patch_impl(
    request: Request,
    file: Optional[UploadFile] = None,
    mode_form: Optional[str] = None,
    sample_id_form: Optional[str] = None,
    target_class_form: Optional[int] = None,
) -> PatchPredictResponse:
    """Shared implementation for patch prediction, /segment, and /biomarkers."""
    image_np: Optional[np.ndarray] = None
    gt_np: Optional[np.ndarray] = None
    mode: str = mode_form or "proposed"
    apply_otsu: bool = True
    target_class: int = int(target_class_form) if target_class_form is not None else 0
    if target_class < 0 or target_class > 4:
        raise HTTPException(status_code=422, detail="Invalid 'target_class': must be in [0, 4].")
    strict_weights: bool = False
    blur_sigma: float = 4.0
    dim_factor: float = 0.55
    gt_alpha: float = 0.45
    pred_alpha: float = 0.55
    error_alpha: float = 0.65
    tolerance_px: int = 2
    error_breakdown: bool = False

    if file is not None:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds 16 MB limit.")
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            image_np = np.array(pil_img, dtype=np.uint8)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot decode image file: {str(e)}")

        if sample_id_form:
            rec = _get_sample_record(sample_id_form)
            if rec["_mask_path"] and os.path.isfile(rec["_mask_path"]):
                gt_np = _load_mask_array(rec["_mask_path"])

    if image_np is None:
        try:
            body = await request.json()
        except Exception:
            body = {}

        if not isinstance(body, dict):
            raise HTTPException(
                status_code=422,
                detail="JSON request body must be a JSON object.",
            )

        raw_mode = body.get("mode")
        if raw_mode is not None:
            if not isinstance(raw_mode, str):
                raise HTTPException(status_code=422, detail="Invalid 'mode': must be a string.")
            if raw_mode.strip():
                mode = raw_mode

        raw_otsu = body.get("apply_otsu_gating")
        if raw_otsu is not None:
            if not isinstance(raw_otsu, bool):
                raise HTTPException(status_code=422, detail="Invalid 'apply_otsu_gating': must be a boolean.")
            apply_otsu = raw_otsu

        raw_target_class = body.get("target_class")
        if raw_target_class is not None:
            if isinstance(raw_target_class, bool) or not isinstance(raw_target_class, (int, str)):
                raise HTTPException(status_code=422, detail="Invalid 'target_class': must be an integer.")
            try:
                target_class = int(raw_target_class)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail="Invalid 'target_class': must be an integer.")
            if target_class < 0 or target_class > 4:
                raise HTTPException(status_code=422, detail="Invalid 'target_class': must be in [0, 4].")

        raw_strict = body.get("strict_weights")
        if raw_strict is not None:
            if not isinstance(raw_strict, bool):
                raise HTTPException(status_code=422, detail="Invalid 'strict_weights': must be a boolean.")
            strict_weights = raw_strict

        blur_sigma = _parse_json_float(body, "blur_sigma", 4.0, 0.0, 25.0)
        dim_factor = _parse_json_float(body, "dim_factor", 0.55, 0.0, 1.0)
        gt_alpha = _parse_json_float(body, "gt_alpha", 0.45, 0.0, 1.0)
        pred_alpha = _parse_json_float(body, "pred_alpha", 0.55, 0.0, 1.0)
        error_alpha = _parse_json_float(body, "error_alpha", 0.65, 0.0, 1.0)

        raw_tol = body.get("tolerance_px")
        if raw_tol is not None:
            if isinstance(raw_tol, bool) or not isinstance(raw_tol, int):
                raise HTTPException(status_code=422, detail="Invalid 'tolerance_px': must be an integer.")
            if raw_tol < 0 or raw_tol > 10:
                raise HTTPException(status_code=422, detail="Invalid 'tolerance_px': must be in [0, 10].")
            tolerance_px = raw_tol

        raw_eb = body.get("error_breakdown")
        if raw_eb is not None:
            if not isinstance(raw_eb, bool):
                raise HTTPException(status_code=422, detail="Invalid 'error_breakdown': must be a boolean.")
            error_breakdown = raw_eb

        raw_sample_id = body.get("sample_id")
        if raw_sample_id is not None:
            if not isinstance(raw_sample_id, str):
                raise HTTPException(status_code=422, detail="Invalid 'sample_id': must be a string.")
            if raw_sample_id:
                rec = _get_sample_record(raw_sample_id)
                image_np = np.array(Image.open(rec["_img_path"]).convert("RGB"), dtype=np.uint8)
                if rec["_mask_path"] and os.path.isfile(rec["_mask_path"]):
                    gt_np = _load_mask_array(rec["_mask_path"])

        raw_image_b64 = body.get("image_base64")
        if image_np is None and raw_image_b64 is not None:
            if not isinstance(raw_image_b64, str):
                raise HTTPException(status_code=422, detail="Invalid 'image_base64': must be a string.")
            if len(raw_image_b64) > MAX_BASE64_CHARS:
                raise HTTPException(status_code=413, detail="Image payload exceeds 16 MB limit.")
            if raw_image_b64:
                try:
                    image_np = decode_base64_to_image(raw_image_b64)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid image_base64: {str(e)}")

        raw_gt_b64 = body.get("gt_mask_base64")
        if raw_gt_b64 is not None:
            if not isinstance(raw_gt_b64, str):
                raise HTTPException(status_code=422, detail="Invalid 'gt_mask_base64': must be a string.")
            if raw_gt_b64:
                try:
                    gt_np = _decode_gt_base64(raw_gt_b64)
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid gt_mask_base64: {str(e)}")

    if image_np is None:
        raise HTTPException(
            status_code=400,
            detail="Missing image data. Provide multipart 'file', JSON 'image_base64', or valid 'sample_id'.",
        )

    result = await run_in_threadpool(
        process_single_patch,
        image_np,
        mode=mode,
        enforce_otsu=apply_otsu,
        gt_mask=gt_np,
        target_class=target_class,
        strict_weights=strict_weights,
        blur_sigma=blur_sigma,
        dim_factor=dim_factor,
        gt_alpha=gt_alpha,
        pred_alpha=pred_alpha,
        error_alpha=error_alpha,
        tolerance_px=tolerance_px,
        error_breakdown=error_breakdown,
    )

    return PatchPredictResponse(
        success=True,
        mask_shape=result["mask_shape"],
        class_distribution=result["class_distribution"],
        biomarkers=BiomarkersResponse(**result["biomarkers"]),
        mask_base64=result["mask_base64"],
        overlay_base64=result["overlay_base64"],
        blended_base64=result["blended_base64"],
        applied_gating=result["applied_gating"],
        method_used=result["method_used"],
        checkpoint_verified=result["checkpoint_verified"],
        latency_ms=result["latency_ms"],
        mean_entropy=result["mean_entropy"],
        entropy_overlay_base64=result["entropy_overlay_base64"],
        contour_overlay_base64=result["contour_overlay_base64"],
        gt_color_base64=result["gt_color_base64"],
        agreement_overlay_base64=result["agreement_overlay_base64"],
        dual_blend_base64=result["dual_blend_base64"],
        blurred_dual_overlay_base64=result["blurred_dual_overlay_base64"],
        cam_overlay_base64=result["cam_overlay_base64"],
        pred_mask_index_base64=result["pred_mask_index_base64"],
        gt_mask_index_base64=result["gt_mask_index_base64"],
        confidence_map_base64=result["confidence_map_base64"],
        gt_distribution=result["gt_distribution"],
        gt_biomarkers=BiomarkersResponse(**result["gt_biomarkers"]) if result["gt_biomarkers"] else None,
        spotlight_overlay_base64=result["spotlight_overlay_base64"],
        spotlight_stats=result["spotlight_stats"],
        agreement_summary=result["agreement_summary"],
        verification_metrics=result["verification_metrics"],
    )


@app.post("/api/v1/predict/patch", response_model=PatchPredictResponse)
async def predict_patch(
    request: Request,
    file: Optional[UploadFile] = File(None),
    mode_form: Optional[str] = Form(None),
    sample_id_form: Optional[str] = Form(None),
    target_class_form: Optional[int] = Form(None),
) -> PatchPredictResponse:
    """Predicts WSSS segmentation, visual verification overlays, and biomarkers on an RGB patch."""
    return await _predict_patch_impl(
        request=request,
        file=file,
        mode_form=mode_form,
        sample_id_form=sample_id_form,
        target_class_form=target_class_form,
    )


@app.post("/api/v1/compare")
async def compare_methods_endpoint(payload: MethodCompareRequest) -> Dict[str, Any]:
    """Runs multiple WSSS methods side-by-side on one patch for empirical visual comparison."""
    image_np: Optional[np.ndarray] = None
    gt_np: Optional[np.ndarray] = None

    if payload.sample_id:
        rec = _get_sample_record(payload.sample_id)
        image_np = np.array(Image.open(rec["_img_path"]).convert("RGB"), dtype=np.uint8)
        if rec["_mask_path"] and os.path.isfile(rec["_mask_path"]):
            gt_np = _load_mask_array(rec["_mask_path"])
    elif payload.image_base64:
        if len(payload.image_base64) > MAX_BASE64_CHARS:
            raise HTTPException(status_code=413, detail="Image payload exceeds 16 MB limit.")
        try:
            image_np = decode_base64_to_image(payload.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {str(e)}")

    if image_np is None:
        raise HTTPException(status_code=400, detail="Provide 'sample_id' or 'image_base64' for method comparison.")

    if payload.gt_mask_base64:
        try:
            gt_np = _decode_gt_base64(payload.gt_mask_base64)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid gt_mask_base64: {str(e)}")

    methods = payload.methods[:6] if payload.methods else ["m1", "m6"]

    def _run_compare() -> Dict[str, Any]:
        shared_cache: Dict[str, Any] = {}
        comparisons = []
        for m in methods:
            res = process_single_patch(
                image_np,
                mode=m,
                enforce_otsu=payload.apply_otsu_gating,
                gt_mask=gt_np,
                strict_weights=payload.strict_weights,
                blur_sigma=payload.blur_sigma,
                dim_factor=payload.dim_factor,
                gt_alpha=payload.gt_alpha,
                pred_alpha=payload.pred_alpha,
                error_alpha=payload.error_alpha,
                tolerance_px=payload.tolerance_px,
                error_breakdown=payload.error_breakdown,
                precomputed_cache=shared_cache,
            )
            comparisons.append(
                {
                    "requested_method": m,
                    "method_used": res["method_used"],
                    "checkpoint_verified": res["checkpoint_verified"],
                    "latency_ms": res["latency_ms"],
                    "mean_entropy": res["mean_entropy"],
                    "biomarkers": res["biomarkers"],
                    "mask_base64": res["mask_base64"],
                    "agreement_overlay_base64": res["agreement_overlay_base64"],
                    "blurred_dual_overlay_base64": res["blurred_dual_overlay_base64"],
                    "contour_overlay_base64": res["contour_overlay_base64"],
                    "entropy_overlay_base64": res["entropy_overlay_base64"],
                    "cam_overlay_base64": res["cam_overlay_base64"],
                    "agreement_summary": res["agreement_summary"],
                    "verification_metrics": res["verification_metrics"],
                }
            )
        return {
            "success": True,
            "gt_available": gt_np is not None,
            "raw_image_base64": encode_image_to_base64(image_np, format="PNG"),
            "gt_color_base64": encode_image_to_base64(mask_to_color_image(gt_np), format="PNG")
            if gt_np is not None
            else None,
            "comparisons": comparisons,
        }

    return await run_in_threadpool(_run_compare)


@app.post("/api/v1/export/report")
async def export_report_endpoint(payload: ExportReportRequest) -> Response:
    """Exports a 300-DPI PNG Diagnostic Comparison Sheet, JSON audit, or CSV metrics report."""
    fmt = (payload.format or "png").lower().strip()
    if fmt not in {"png", "json", "csv"}:
        raise HTTPException(status_code=422, detail=f"Invalid report format '{payload.format}'. Allowed: png, json, csv.")

    image_np: Optional[np.ndarray] = None
    gt_np: Optional[np.ndarray] = None
    patient_id = "custom_upload"
    sample_label = payload.sample_id or "custom_patch"

    if payload.sample_id:
        rec = _get_sample_record(payload.sample_id)
        patient_id = rec["patient_id"]
        sample_label = rec["sample_id"]
        image_np = np.array(Image.open(rec["_img_path"]).convert("RGB"), dtype=np.uint8)
        if rec["_mask_path"] and os.path.isfile(rec["_mask_path"]):
            gt_np = _load_mask_array(rec["_mask_path"])
    elif payload.image_base64:
        if len(payload.image_base64) > MAX_BASE64_CHARS:
            raise HTTPException(status_code=413, detail="Image payload exceeds 16 MB limit.")
        try:
            image_np = decode_base64_to_image(payload.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {str(e)}")

    if image_np is None:
        raise HTTPException(status_code=400, detail="Provide 'sample_id' or 'image_base64' to export report.")

    if payload.gt_mask_base64:
        try:
            gt_np = _decode_gt_base64(payload.gt_mask_base64)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid gt_mask_base64: {str(e)}")

    safe_tag = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in sample_label)[:64] or "patch"

    def _build_export() -> Tuple[bytes, str, str]:
        res = process_single_patch(
            image_np,
            mode=payload.mode or "m6",
            enforce_otsu=payload.apply_otsu_gating,
            gt_mask=gt_np,
            target_class=payload.target_class,
            blur_sigma=payload.blur_sigma,
            dim_factor=payload.dim_factor,
            gt_alpha=payload.gt_alpha,
            pred_alpha=payload.pred_alpha,
            error_alpha=payload.error_alpha,
            tolerance_px=payload.tolerance_px,
            error_breakdown=True,
        )
        if fmt == "png":
            png_bytes = render_single_patch_diagnostic_sheet(
                image=image_np,
                pred_mask=res["pred_mask"],
                probs=res["_probs_arr"],
                gt_mask=res["_gt_aligned"],
                cam_map=res["_cam_aligned"],
                method_name=str(res["method_used"] or "m6"),
                sample_id=sample_label,
                biomarkers=res["biomarkers"],
                gt_biomarkers=res["gt_biomarkers"],
                blur_sigma=payload.blur_sigma,
                dim_factor=payload.dim_factor,
                gt_alpha=payload.gt_alpha,
                pred_alpha=payload.pred_alpha,
                error_alpha=payload.error_alpha,
                tolerance_px=payload.tolerance_px,
                dpi=300,
            )
            return png_bytes, "image/png", f"{safe_tag}_diagnostic_sheet.png"

        audit_doc = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sample_id": sample_label,
            "patient_id": patient_id,
            "method_used": res["method_used"],
            "checkpoint_verified": res["checkpoint_verified"],
            "applied_gating": res["applied_gating"],
            "latency_ms": res["latency_ms"],
            "mean_entropy": res["mean_entropy"],
            "biomarkers": res["biomarkers"],
            "gt_biomarkers": res["gt_biomarkers"],
            "class_distribution": res["class_distribution"],
            "gt_distribution": res["gt_distribution"],
            "agreement_summary": res["agreement_summary"],
            "verification_metrics": res["verification_metrics"],
        }
        if fmt == "json":
            return (
                json.dumps(audit_doc, indent=2).encode("utf-8"),
                "application/json",
                f"{safe_tag}_audit_report.json",
            )

        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["section", "metric", "value"])
        writer.writerow(["meta", "timestamp_utc", audit_doc["timestamp_utc"]])
        writer.writerow(["meta", "sample_id", sample_label])
        writer.writerow(["meta", "patient_id", patient_id])
        writer.writerow(["meta", "method_used", res["method_used"]])
        writer.writerow(["meta", "checkpoint_verified", res["checkpoint_verified"]])
        writer.writerow(["meta", "mean_entropy", res["mean_entropy"]])
        for k, v in res["biomarkers"].items():
            writer.writerow(["pred_biomarkers", k, v])
        if res["gt_biomarkers"]:
            for k, v in res["gt_biomarkers"].items():
                writer.writerow(["gt_biomarkers", k, v])
        for k, v in res["class_distribution"].items():
            writer.writerow(["pred_class_pct", k, v])
        if res["gt_distribution"]:
            for k, v in res["gt_distribution"].items():
                writer.writerow(["gt_class_pct", k, v])
        if res["verification_metrics"]:
            for k, v in res["verification_metrics"].items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        writer.writerow([f"verification_{k}", sub_k, sub_v])
                elif not isinstance(v, list):
                    writer.writerow(["verification", k, v])
        return csv_buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"{safe_tag}_audit_report.csv"

    content, media_type, filename = await run_in_threadpool(_build_export)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/v1/predict/wsi", response_model=WSIPredictResponse)
def predict_wsi(request: WSIPredictRequest) -> WSIPredictResponse:
    """Accepts multi-patch tiles and coordinates, stitches predictions into a composite heatmap."""
    if not request.tiles:
        raise HTTPException(status_code=400, detail="Tile list cannot be empty.")
    if len(request.tiles) > MAX_WSI_TILES:
        raise HTTPException(status_code=422, detail=f"Too many tiles ({len(request.tiles)}); maximum is {MAX_WSI_TILES}.")

    req_mode = (request.mode or "m6").lower().strip()
    if req_mode not in REAL_SOTA_METHODS and req_mode not in LEGACY_CAM_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid WSI mode '{request.mode}'. Allowed: {sorted(REAL_SOTA_METHODS | LEGACY_CAM_MODES)}",
        )

    max_wsi_dim = 8192
    for tile in request.tiles:
        if tile.x >= max_wsi_dim or tile.y >= max_wsi_dim:
            raise HTTPException(
                status_code=422,
                detail=f"Tile {tile.id} coordinates ({tile.x}, {tile.y}) exceed maximum canvas bound ({max_wsi_dim}).",
            )
        if len(tile.image_base64) > MAX_BASE64_CHARS:
            raise HTTPException(status_code=413, detail=f"Tile {tile.id} payload exceeds 16 MB limit.")

    decoded_tiles: List[Tuple[WSITileItem, np.ndarray]] = []
    for tile in request.tiles:
        try:
            img_np = decode_base64_to_image(tile.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error decoding tile {tile.id}: {str(e)}")
        th, tw = img_np.shape[:2]
        if (int(tile.x) + int(tw)) > max_wsi_dim or (int(tile.y) + int(th)) > max_wsi_dim:
            raise HTTPException(
                status_code=422,
                detail=f"Composite WSI canvas exceeds {max_wsi_dim}x{max_wsi_dim} limit at tile {tile.id}.",
            )
        decoded_tiles.append((tile, img_np))

    processed_tiles: List[Dict[str, Any]] = []
    raw_tiles: List[Tuple[int, int, np.ndarray]] = []
    tile_summaries: List[Dict[str, Any]] = []
    resolved_method_name = req_mode

    for tile, img_np in decoded_tiles:
        try:
            res = process_single_patch(img_np, mode=req_mode, enforce_otsu=True)
            resolved_method_name = str(res["method_used"] or req_mode)
            processed_tiles.append({
                "x": tile.x,
                "y": tile.y,
                "mask": res["pred_mask"],
                "probs": res["_probs_arr"],
            })
            raw_tiles.append((int(tile.x), int(tile.y), img_np))
            dom_cls = max(res["class_distribution"].items(), key=lambda kv: kv[1])[0]
            tile_summaries.append({
                "id": tile.id or f"tile_{len(tile_summaries)}",
                "x": int(tile.x),
                "y": int(tile.y),
                "shape": res["mask_shape"],
                "vtp": res["biomarkers"]["viable_tumor_percentage"],
                "tsr": res["biomarkers"]["tumor_stroma_ratio"],
                "dominant_class": dom_cls,
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing tile {tile.id}: {str(e)}")

    composite_mask, composite_rgb = stitch_tiles_heatmap(processed_tiles)
    composite_rgba = mask_to_rgba_overlay(composite_mask, alpha=0.55, bg_transparent=True)

    ch, cw = composite_mask.shape[:2]
    composite_raw = np.full((ch, cw, 3), 245, dtype=np.uint8)
    for tx, ty, raw_tile_img in raw_tiles:
        th, tw = raw_tile_img.shape[:2]
        composite_raw[ty : ty + th, tx : tx + tw] = raw_tile_img[:, :, :3]
    composite_heatmap = blend_overlay(composite_raw, composite_mask, alpha=0.50)

    vtp = compute_viable_tumor_percentage(composite_mask, tumor_class_idx=0, background_class_idx=4)
    tsr = compute_tumor_stroma_ratio(composite_mask, tumor_class_idx=0, stroma_class_idx=1)
    distribution = compute_class_distribution(composite_mask)
    tissue_cov = round(100.0 - float(distribution.get("BACK", 0.0)), 2)
    dominant_cls = max(distribution.items(), key=lambda kv: kv[1])[0]

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
        composite_mask_base64=encode_image_to_base64(composite_rgb, format="PNG"),
        composite_overlay_base64=encode_image_to_base64(composite_rgba, format="PNG"),
        composite_heatmap_base64=encode_image_to_base64(composite_heatmap, format="PNG"),
        region_summary={
            "canvas_width": int(cw),
            "canvas_height": int(ch),
            "tissue_coverage_pct": tissue_cov,
            "dominant_class": dominant_cls,
            "method_used": resolved_method_name,
            "tiles": tile_summaries,
        },
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
    mode_form: Optional[str] = Form(None),
    sample_id_form: Optional[str] = Form(None),
    target_class_form: Optional[int] = Form(None),
) -> PatchPredictResponse:
    """Alias for /api/v1/predict/patch matching API contract."""
    return await _predict_patch_impl(
        request=request,
        file=file,
        mode_form=mode_form,
        sample_id_form=sample_id_form,
        target_class_form=target_class_form,
    )


@app.post("/biomarkers", response_model=BiomarkersResponse)
async def calculate_biomarkers_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    mode_form: Optional[str] = Form(None),
    sample_id_form: Optional[str] = Form(None),
    target_class_form: Optional[int] = Form(None),
) -> BiomarkersResponse:
    """Computes clinical biomarkers (VTP, TSR) from image patch or segmentation mask."""
    res = await _predict_patch_impl(
        request=request,
        file=file,
        mode_form=mode_form,
        sample_id_form=sample_id_form,
        target_class_form=target_class_form,
    )
    return res.biomarkers


@app.get("/splits")
def get_splits_endpoint() -> Dict[str, Any]:
    """Returns patient-level anti-leakage train/val/test splits."""
    return load_patient_splits()


@app.get("/tile/{z}/{x}/{y}.png")
def get_tile_endpoint(z: int, x: int, y: int) -> Response:
    """Serves real cached DeepZoom WSI PNG tile if present; returns 404 otherwise."""
    tile_path = os.path.join(os.path.dirname(os.path.dirname(MODULE_DIR)), "data", "tiles", str(z), str(x), f"{y}.png")
    if os.path.isfile(tile_path):
        return FileResponse(tile_path, media_type="image/png")
    raise HTTPException(status_code=404, detail=f"WSI tile ({z}/{x}/{y}) not found. Load a slide first.")
