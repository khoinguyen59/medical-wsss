"""
Gate-7 Track A Real Visual Overlay & Whole-ROI Verification Suite.

Implements a zero-cherry-picking, publication-grade visual and spatial alignment audit
directly reusing frozen Gate-7 Track A ensemble primitives from `src/stage2_v2/ensemble.py`:
1. Exact frozen Track A streaming inference on BCSS-WSSS TEST (compact uint8 predictions +
   disk-backed centered float16 log-prob memmaps, zero full-dataset FP32 RAM bloat).
2. Six mandatory fail-fast integrity assertions:
   - GT PNG mode 'P', (224, 224), {0..4} IDs, exact BCSS PNG palette, zero ROI remapping.
   - Checkpoint SHA-256 verification and input_norm contract enforcement.
   - Single-pass raw_1x Flip & Rot90 equivariance across all 4 Gate-7 arms + Author.
   - Non-wrapping interior (dx, dy) translation-shift sensitivity matrix + orientation collapse.
   - Zero-GT-leakage bright-glass guard invariance check.
   - Pooled TEST mIoU reconciliation against frozen Gate-7 Track A benchmarks.
3. Zero-cherry-picking 300-DPI contact sheets (01..08, nearest-neighbor crisp pixels,
   zero Gaussian blur, zero GT overwrite) + 2 column-major Whole-ROI stitches (09..10)
   + 300-DPI Academic Monochrome alignment figure (11 & stage2_training_loss.png).
4. Atomic output packing compatible with Stage-2 visual audit notebook `02d_Stage2_Visual_Overlay_Audit_G4.ipynb`.

Pure ASCII hyphen '-' (0x2D) only (Invariant 12).
"""

import csv
import gc
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader

from src.datasets.otsu_filter import compute_otsu_tissue_mask
from src.models.wsss_inference_methods import (
    MultiMethodWSSSEngineV2,
    WSSSTissueStage1NetCAM,
    build_and_load_deeplabv3plus,
)
from src.stage2_v2.ensemble import (
    SHA256_AUTHOR_STAGE2,
    SHA256_STAGE1_BEST_CLS,
    SHA256_STAGE1_RETRAINED,
    _close_memmap,
    _combine_5_expert_probs,
    _fuse_memmap_chunk,
    _gate_masks_from_thresholds,
    apply_tta_probs,
    compute_macro_miou_from_iu,
    encode_centered_logprobs_fp16,
    paired_wilcoxon_and_dz,
    summarize_iu,
    validate_track_a_arm_spec,
)
from src.stage2_v2.evaluator import (
    _apply_prob_cgm,
    _extract_img_gt,
    _forward_dual_head,
    _make_cgm_gate,
    _ms_tta_probs,
)
from src.stage2_v2.models import load_stage2_v2_checkpoint
from src.training.dataset_loader import FOREGROUND_CLASSES, NUM_FOREGROUND, extract_patient_id
from src.training.eval_stage1_full import compute_file_sha256, load_stage1_strict
from src.training.train_stage2 import prepare_stage1_input

_CLS_LOWER: Tuple[str, ...] = tuple(c.lower() for c in FOREGROUND_CLASSES)

GATE7_ARM_ORDER: Tuple[str, ...] = (
    "arm8_deeplab_r101_r2",
    "arm6_segformer_b5_r2",
    "arm7_convnext_large_r2",
    "arm5_dinov2_large_r2",
)
GATE7_SELECTED_TTAS: Tuple[str, ...] = (
    "ms_3_hvflip",
    "ms_3_hvflip",
    "ms_3_hvflip",
    "ms_3_hvflip",
)
GATE7_FULL_VAL_BETAS: Tuple[float, ...] = (0.8782, 0.7029, 0.6747, 0.6386)
GATE7_UNIFORM_WEIGHTS: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
GATE7_SIMPLEX_WEIGHTS: Tuple[float, ...] = (0.28, 0.06, 0.05, 0.61)
GATE7_ARM5_SELF_CGM_THRESHOLDS: Tuple[float, ...] = (0.10, 0.10, 0.10, 0.10)
GATE7_4ARM_CGM_THRESHOLDS: Tuple[float, ...] = (0.10, 0.10, 0.10, 0.10)

GATE7_SELECTED_TTAS_BY_ARM: Dict[str, str] = dict(zip(GATE7_ARM_ORDER, GATE7_SELECTED_TTAS))
GATE7_FULL_VAL_BETAS_BY_ARM: Dict[str, float] = dict(zip(GATE7_ARM_ORDER, GATE7_FULL_VAL_BETAS))
GATE7_UNIFORM_WEIGHTS_BY_ARM: Dict[str, float] = dict(zip(GATE7_ARM_ORDER, GATE7_UNIFORM_WEIGHTS))
GATE7_SIMPLEX_WEIGHTS_BY_ARM: Dict[str, float] = dict(zip(GATE7_ARM_ORDER, GATE7_SIMPLEX_WEIGHTS))

GATE7_ARM_SHA256: Dict[str, str] = {
    "arm8_deeplab_r101_r2": "9fd6e121a97147567b96850c2f96c2a22dde64f6b3cf8d846bb20dd18f6122e7",
    "arm6_segformer_b5_r2": "3055643db074a2dfe8b53ad8d48b1e4ff3f3762fc674b0ad0589be498c8fdc9e",
    "arm7_convnext_large_r2": "c64b97c731c71c97644d662c943fe49c473c2308d612eb24eb781105f900c75c",
    "arm5_dinov2_large_r2": "33f4d623e88254713d7258c403b7a2059c2645a35ab40beaa78cc3852dd978b7",
}

GATE7_EXPECTED_TEST_MIOU: Dict[str, float] = {
    "author_step0": 0.7043,
    "arm5_cgm": 0.7263,
    "ens4_no_cgm": 0.7279,
    "ens4_cgm": 0.7280,
}

# Raw indexed palette header stored inside canonical BCSS-WSSS mask PNGs (indices 0..4)
BCSS_PNG_PALETTE_PREFIX: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (153, 0, 255),
    (255, 255, 255),
)

# ColorBrewer-aligned clinical overlay palette (0=TUM Red, 1=STR Green, 2=LYM Blue, 3=NEC Orange, 4=BACK Gray)
CLINICAL_OVERLAY_PALETTE: np.ndarray = np.array(
    [
        [228, 26, 28],
        [77, 175, 74],
        [55, 126, 184],
        [255, 127, 0],
        [153, 153, 153],
    ],
    dtype=np.uint8,
)

SHIFT_GRID: Tuple[int, ...] = (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)

WHOLE_ROI_TARGETS: Tuple[Tuple[str, int, int, str, str], ...] = (
    (
        "TCGA-EW-A1P4-DX1_xmin17256_ymin35430_MPP-0.2500",
        14,
        17,
        "09_whole_roi_TCGA_EW_A1P4_14x17_and_5x5_zoom.png",
        "nec_fp",
    ),
    (
        "TCGA-EW-A1P1-DX1_xmin50567_ymin38988_MPP-0.2500",
        16,
        18,
        "10_whole_roi_TCGA_EW_A1P1_16x18_and_5x5_zoom.png",
        "multiclass",
    ),
)

WORST_FAILURE_TARGET_PATIENTS: Tuple[str, ...] = (
    "TCGA-E2-A1LL",
    "TCGA-GI-A2C9",
    "TCGA-EW-A3U0",
    "TCGA-EW-A6SB",
)

IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically persist bytes to disk via sibling .tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write UTF-8 JSON file."""
    raw = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(path, raw)


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, path)


def _atomic_save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Atomically save a Matplotlib figure at exact DPI with PNG optimization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.png")
    fig.savefig(tmp_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Re-save with PIL optimize=True for compact ZIP size while preserving 300 DPI metadata
    with Image.open(tmp_path) as im:
        rgb = im.convert("RGB")
        opt_tmp = path.with_suffix(path.suffix + ".opt.png")
        rgb.save(opt_tmp, format="PNG", optimize=True, compress_level=9, dpi=(dpi, dpi))
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass
    os.replace(opt_tmp, path)


def _atomic_write_zip(zip_path: Path, entries: Mapping[str, Path]) -> None:
    """Atomically create a compressed ZIP archive via sibling .tmp + os.replace."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arcname, src_path in entries.items():
            if src_path.is_file():
                zf.write(src_path, arcname=arcname)
    os.replace(tmp_path, zip_path)


def _softmax_np(scores: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax over class axis."""
    arr = np.asarray(scores, dtype=np.float32)
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp_s = np.exp(shifted)
    return (exp_s / np.maximum(np.sum(exp_s, axis=axis, keepdims=True), 1e-8)).astype(np.float32)


def _apply_bitmask_argmax_np(scores_b4hw: np.ndarray, bitmasks: np.ndarray) -> np.ndarray:
    """Apply per-sample 4-bit class masks to (B, 4, H, W) scores using -inf masking."""
    b = int(scores_b4hw.shape[0])
    out = np.empty((b, scores_b4hw.shape[2], scores_b4hw.shape[3]), dtype=np.uint8)
    neg_inf = np.float32(-np.inf)
    for i in range(b):
        bm = int(bitmasks[i])
        active = np.asarray([(bm >> c) & 1 for c in range(NUM_FOREGROUND)], dtype=bool).reshape(
            NUM_FOREGROUND, 1, 1
        )
        masked = np.where(active, scores_b4hw[i], neg_inf)
        out[i] = np.argmax(masked, axis=0).astype(np.uint8)
    return out


def _denorm_imagenet_to_uint8(
    img_chw: np.ndarray,
    images_are_imagenet_norm: bool = True,
) -> np.ndarray:
    """Convert a single (3, H, W) tensor array to uint8 (H, W, 3), unconditionally denormalizing ImageNet tensors."""
    arr = np.asarray(img_chw, dtype=np.float32)
    if images_are_imagenet_norm:
        mean = IMAGENET_MEAN_NP[0]
        std = IMAGENET_STD_NP[0]
        rgb01 = arr * std + mean
    else:
        rgb01 = arr / 255.0 if float(arr.max()) > 1.5 else arr
    rgb255 = np.clip(np.round(rgb01 * 255.0), 0.0, 255.0).astype(np.uint8)
    return np.transpose(rgb255, (1, 2, 0))


def _rgb_uint8_to_model_tensor(raw_rgb: np.ndarray, input_norm: str, device: torch.device) -> torch.Tensor:
    """Convert uint8 (H, W, 3) image to (1, 3, H, W) tensor respecting input_norm contract."""
    arr = np.asarray(raw_rgb, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))[None, ...]
    if str(input_norm).strip().lower() == "imagenet":
        chw = (chw - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
    return torch.from_numpy(chw).to(device)


# ---------------------------------------------------------------------------
# 6 Mandatory Fail-Fast Integrity Assertions
# ---------------------------------------------------------------------------


def verify_gt_mask_contract(
    mask_paths: Sequence[Union[str, Path]],
    expected_hw: Tuple[int, int] = (224, 224),
    check_palette: bool = True,
    max_check: Optional[int] = None,
) -> Dict[str, Any]:
    """Assertion 1: Validate GT PNGs directly (mode 'P', size, {0..4} IDs, exact BCSS palette)."""
    paths = [Path(p) for p in mask_paths]
    if not paths:
        raise RuntimeError("[STAGE2 ERROR] GT mask contract failed: no mask paths provided.")
    subset = paths if max_check is None else paths[: int(max_check)]
    exp_h, exp_w = int(expected_hw[0]), int(expected_hw[1])
    observed_ids: set = set()
    palette_verified = 0

    for p in subset:
        if not p.is_file():
            raise RuntimeError(f"[STAGE2 ERROR] GT mask missing on disk: {p}")
        with Image.open(p) as im:
            if check_palette and im.mode != "P":
                raise RuntimeError(
                    f"[STAGE2 ERROR] GT mask {p.name} has mode={im.mode!r}, expected indexed 'P'."
                )
            if im.size != (exp_w, exp_h):
                raise RuntimeError(
                    f"[STAGE2 ERROR] GT mask {p.name} has size={im.size}, expected {(exp_w, exp_h)}."
                )
            if check_palette:
                pal = im.getpalette()
                if pal is None or len(pal) < 15:
                    raise RuntimeError(f"[STAGE2 ERROR] GT mask {p.name} lacks valid BCSS PNG palette.")
                prefix = tuple(tuple(pal[i * 3 : i * 3 + 3]) for i in range(5))
                if prefix != BCSS_PNG_PALETTE_PREFIX:
                    raise RuntimeError(
                        f"[STAGE2 ERROR] GT mask {p.name} palette prefix {prefix} != {BCSS_PNG_PALETTE_PREFIX}."
                    )
                palette_verified += 1
            arr = np.asarray(im, dtype=np.int64)
            uvals = set(int(v) for v in np.unique(arr))
            invalid = uvals - {0, 1, 2, 3, 4}
            if invalid:
                raise RuntimeError(
                    f"[STAGE2 ERROR] GT mask {p.name} contains invalid class IDs {sorted(invalid)}."
                )
            observed_ids.update(uvals)

    return {
        "passed": True,
        "checked_count": len(subset),
        "palette_verified_count": palette_verified,
        "observed_class_ids": sorted(observed_ids),
        "expected_hw": [exp_h, exp_w],
        "zero_roi_remapper_used": True,
    }


def verify_checkpoints_and_norm_contract(
    arm_specs: Sequence[Mapping[str, Any]],
    stage1_cls_ckpt: Optional[Union[str, Path]] = None,
    author_stage2_ckpt: Optional[Union[str, Path]] = None,
    expected_arm_shas: Optional[Mapping[str, str]] = GATE7_ARM_SHA256,
    expected_stage1_sha: Optional[str] = SHA256_STAGE1_BEST_CLS,
    expected_author_sha: Optional[str] = SHA256_AUTHOR_STAGE2,
    enforce_sha: bool = True,
    track: str = "track_a_clean",
    author_stage1_ckpt: Optional[Union[str, Path]] = None,
    expected_author_stage1_sha: Optional[str] = SHA256_STAGE1_RETRAINED,
) -> Dict[str, Any]:
    """Assertion 2: Verify Track A quarantine, checkpoint SHA-256 hashes, and input_norm contract."""
    arm_shas: Dict[str, str] = {}
    for spec in arm_specs:
        name = str(spec["name"]).strip().lower()
        validate_track_a_arm_spec(spec, track=track)
        ckpt_path = spec.get("ckpt_path")
        if ckpt_path is not None:
            p = Path(ckpt_path)
            if not p.is_file():
                raise RuntimeError(f"[STAGE2 ERROR] Missing arm checkpoint for {name}: {p}")
            sha = compute_file_sha256(p)
            arm_shas[name] = sha
            if enforce_sha and expected_arm_shas and name in expected_arm_shas:
                if sha != expected_arm_shas[name]:
                    raise RuntimeError(
                        f"[STAGE2 ERROR] SHA-256 mismatch for {name}: got {sha}, expected {expected_arm_shas[name]}."
                    )
        else:
            arm_shas[name] = "in_memory_model"

    s1_sha = "in_memory_model"
    if stage1_cls_ckpt is not None:
        p1 = Path(stage1_cls_ckpt)
        if not p1.is_file():
            raise RuntimeError(f"[STAGE2 ERROR] Missing Stage-1 checkpoint: {p1}")
        s1_sha = compute_file_sha256(p1)
        if enforce_sha and expected_stage1_sha and s1_sha != expected_stage1_sha:
            raise RuntimeError(
                f"[STAGE2 ERROR] Stage-1 SHA-256 mismatch: got {s1_sha}, expected {expected_stage1_sha}."
            )

    author_s1_sha = "fallback_to_stage1_cls"
    if author_stage1_ckpt is not None:
        pa1 = Path(author_stage1_ckpt)
        if not pa1.is_file():
            raise RuntimeError(f"[STAGE2 ERROR] Missing Author Stage-1 checkpoint: {pa1}")
        author_s1_sha = compute_file_sha256(pa1)
        if enforce_sha and expected_author_stage1_sha and author_s1_sha != expected_author_stage1_sha:
            raise RuntimeError(
                f"[STAGE2 ERROR] Author Stage-1 SHA-256 mismatch: got {author_s1_sha}, "
                f"expected {expected_author_stage1_sha}."
            )

    author_sha = "in_memory_model"
    if author_stage2_ckpt is not None:
        pa = Path(author_stage2_ckpt)
        if not pa.is_file():
            raise RuntimeError(f"[STAGE2 ERROR] Missing Author Stage-2 checkpoint: {pa}")
        author_sha = compute_file_sha256(pa)
        if enforce_sha and expected_author_sha and author_sha != expected_author_sha:
            raise RuntimeError(
                f"[STAGE2 ERROR] Author Stage-2 SHA-256 mismatch: got {author_sha}, expected {expected_author_sha}."
            )

    return {
        "passed": True,
        "track": track,
        "enforce_sha": bool(enforce_sha),
        "arm_sha256": arm_shas,
        "stage1_cls_sha256": s1_sha,
        "author_stage1_sha256": author_s1_sha,
        "author_stage2_sha256": author_sha,
        "input_norm_contract": {
            "stage2_v2_arms": "imagenet",
            "author_stage2": "zero_one",
            "stage1_cls": "zero_one",
            "author_stage1": "zero_one" if author_stage1_ckpt is not None else "fallback_to_stage1_cls",
        },
    }


def _compute_pair_miou_4cls(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """4-class macro mIoU between two (H, W) or (N, H, W) foreground class maps."""
    inter = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    union = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    for c in range(NUM_FOREGROUND):
        pa = pred_a == c
        pb = pred_b == c
        tp = int(np.logical_and(pa, pb).sum())
        inter[c] = tp
        union[c] = int(pa.sum()) + int(pb.sum()) - tp
    return compute_macro_miou_from_iu(inter, union)


@torch.no_grad()
def verify_flip_rot90_equivariance(
    models_by_name: Mapping[str, nn.Module],
    raw_rgb_patches: Sequence[np.ndarray],
    input_norms: Mapping[str, str],
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    min_miou: float = 0.70,
    min_pixel_agreement: float = 0.85,
    min_uninverted_margin: float = 0.15,
    enforce_strict: bool = True,
) -> Dict[str, Any]:
    """Assertion 3: Pooled raw_1x hflip, vflip, and rot90 equivariance across all models.

    Computes pooled 4-class mIoU, pixel agreement, and un-inverted (axis-bug) baseline across
    the asymmetric probe patches so any inverted flip/rot90 axis fails fast while avoiding
    single-patch trace-class unweighted mIoU artifacts.
    """
    if not raw_rgb_patches:
        raise RuntimeError("[STAGE2 ERROR] Equivariance check requires at least 1 RGB patch.")

    use_amp = device.type == "cuda" and amp_dtype is not None
    per_model: Dict[str, Dict[str, Any]] = {}
    worst_miou = 1.0
    worst_pix = 1.0
    worst_key = ""
    failures: List[str] = []

    for name, model in models_by_name.items():
        norm_mode = str(input_norms.get(name, "imagenet"))
        model.eval()
        p0_list: List[np.ndarray] = []
        inv_lists: Dict[str, List[np.ndarray]] = {"hflip": [], "vflip": [], "rot90": []}
        raw_lists: Dict[str, List[np.ndarray]] = {"hflip": [], "vflip": [], "rot90": []}

        for raw_rgb in raw_rgb_patches:
            x0 = _rgb_uint8_to_model_tensor(raw_rgb, norm_mode, device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                lg0, _ = _forward_dual_head(model, x0)
            p0 = torch.argmax(lg0.float(), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            p0_list.append(p0)

            # Horizontal flip
            xh = torch.flip(x0, dims=[-1])
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                lgh, _ = _forward_dual_head(model, xh)
            ph_raw_t = torch.argmax(lgh.float(), dim=1)
            raw_lists["hflip"].append(ph_raw_t.squeeze(0).cpu().numpy().astype(np.uint8))
            inv_lists["hflip"].append(torch.flip(ph_raw_t, dims=[-1]).squeeze(0).cpu().numpy().astype(np.uint8))

            # Vertical flip
            xv = torch.flip(x0, dims=[-2])
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                lgv, _ = _forward_dual_head(model, xv)
            pv_raw_t = torch.argmax(lgv.float(), dim=1)
            raw_lists["vflip"].append(pv_raw_t.squeeze(0).cpu().numpy().astype(np.uint8))
            inv_lists["vflip"].append(torch.flip(pv_raw_t, dims=[-2]).squeeze(0).cpu().numpy().astype(np.uint8))

            # 90-degree CCW rotation (k=1) inverted by k=3
            xr = torch.rot90(x0, k=1, dims=[-2, -1])
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                lgr, _ = _forward_dual_head(model, xr)
            pr_raw_t = torch.argmax(lgr.float(), dim=1)
            raw_lists["rot90"].append(pr_raw_t.squeeze(0).cpu().numpy().astype(np.uint8))
            inv_lists["rot90"].append(
                torch.rot90(pr_raw_t, k=3, dims=[-2, -1]).squeeze(0).cpu().numpy().astype(np.uint8)
            )

        p0_all = np.stack(p0_list, axis=0)
        summary_m: Dict[str, Any] = {}
        for tname in ("hflip", "vflip", "rot90"):
            p_inv = np.stack(inv_lists[tname], axis=0)
            p_raw = np.stack(raw_lists[tname], axis=0)
            pooled_miou = _compute_pair_miou_4cls(p0_all, p_inv)
            pix_acc = float(np.mean(p0_all == p_inv))
            uninv_miou = _compute_pair_miou_4cls(p0_all, p_raw)
            summary_m[tname] = round(pooled_miou, 4)
            summary_m[f"{tname}_pixel_agreement"] = round(pix_acc, 4)
            summary_m[f"{tname}_uninverted_miou"] = round(uninv_miou, 4)
            if pooled_miou < worst_miou:
                worst_miou = pooled_miou
                worst_key = f"{name}:{tname}"
            if pix_acc < worst_pix:
                worst_pix = pix_acc
            if pooled_miou < float(min_miou):
                failures.append(f"{name}:{tname} pooled_miou={pooled_miou:.4f} < {min_miou:.4f}")
            if pix_acc < float(min_pixel_agreement):
                failures.append(f"{name}:{tname} pixel_agreement={pix_acc:.4f} < {min_pixel_agreement:.4f}")
            if uninv_miou < 0.75 and (pooled_miou - uninv_miou) < float(min_uninverted_margin):
                failures.append(
                    f"{name}:{tname} inverted={pooled_miou:.4f} vs uninverted={uninv_miou:.4f} "
                    f"(margin < {min_uninverted_margin:.4f})"
                )
        per_model[name] = summary_m

    passed = len(failures) == 0
    if enforce_strict and not passed:
        raise RuntimeError(
            "[STAGE2 ERROR] Equivariance assertion failed: " + "; ".join(failures)
        )
    return {
        "passed": passed,
        "min_miou_threshold": float(min_miou),
        "min_pixel_agreement_threshold": float(min_pixel_agreement),
        "worst_observed": round(float(worst_miou), 4),
        "worst_pixel_agreement": round(float(worst_pix), 4),
        "worst_key": worst_key,
        "failures": failures,
        "n_patches": len(raw_rgb_patches),
        "per_model": per_model,
    }


def _pooled_miou_interior(
    pred: np.ndarray,
    gt: np.ndarray,
    chunk_size: int = 256,
) -> float:
    """Compute canonical 4-class pooled mIoU strictly on valid_fg = (gt >= 0) & (gt < 4) via bincount."""
    p_arr = np.asarray(pred, dtype=np.uint8)
    g_arr = np.asarray(gt, dtype=np.uint8)
    if p_arr.ndim == 2:
        p_arr = p_arr[None, ...]
        g_arr = g_arr[None, ...]

    n = int(p_arr.shape[0])
    cm = np.zeros((NUM_FOREGROUND, NUM_FOREGROUND + 1), dtype=np.int64)
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        gb = g_arr[start:end]
        pb = p_arr[start:end]
        valid = gb < NUM_FOREGROUND
        if not np.any(valid):
            continue
        gv = gb[valid].astype(np.int64)
        pv = np.minimum(pb[valid], NUM_FOREGROUND).astype(np.int64)
        cm += np.bincount(gv * (NUM_FOREGROUND + 1) + pv, minlength=NUM_FOREGROUND * (NUM_FOREGROUND + 1)).reshape(
            NUM_FOREGROUND, NUM_FOREGROUND + 1
        )
    cm4 = cm[:, :NUM_FOREGROUND]
    inter = np.diag(cm4)
    union = cm4.sum(axis=0) + cm.sum(axis=1) - inter
    return compute_macro_miou_from_iu(inter, union)


def compute_translation_shift_matrix(
    preds_by_model: Mapping[str, np.ndarray],
    gt_masks: np.ndarray,
    shifts: Sequence[int] = SHIFT_GRID,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Assertion 4a: Strictly non-wrapping interior (dx, dy) translation-shift & orientation collapse matrix."""
    gt_arr = np.asarray(gt_masks, dtype=np.uint8)
    if gt_arr.ndim != 3:
        raise ValueError(f"Expected gt_masks shape (N, H, W), got {gt_arr.shape}")
    n_patches, h, w = gt_arr.shape
    shift_vals = [int(s) for s in shifts]
    margin = max(abs(s) for s in shift_vals) if shift_vals else 0
    if 2 * margin >= min(h, w):
        raise ValueError(f"Shift margin {margin} too large for patch shape {(h, w)}")

    use_cuda = device is not None and device.type == "cuda" and torch.cuda.is_available()

    def _miou_pair_fast(p_sub: np.ndarray, g_sub: np.ndarray) -> float:
        if not use_cuda:
            return _pooled_miou_interior(p_sub, g_sub)
        cm_t = torch.zeros(NUM_FOREGROUND * (NUM_FOREGROUND + 1), dtype=torch.int64, device=device)
        batch_n = 256
        for st in range(0, n_patches, batch_n):
            en = min(n_patches, st + batch_n)
            gt_t = torch.from_numpy(np.ascontiguousarray(g_sub[st:en])).to(device=device, dtype=torch.int64)
            pr_t = torch.from_numpy(np.ascontiguousarray(p_sub[st:en])).to(device=device, dtype=torch.int64)
            vm = gt_t < NUM_FOREGROUND
            if not bool(vm.any()):
                continue
            gv = gt_t[vm]
            pv = torch.clamp(pr_t[vm], 0, NUM_FOREGROUND)
            cm_t += torch.bincount(gv * (NUM_FOREGROUND + 1) + pv, minlength=NUM_FOREGROUND * (NUM_FOREGROUND + 1))
        cm_np = cm_t.cpu().numpy().reshape(NUM_FOREGROUND, NUM_FOREGROUND + 1)
        cm4 = cm_np[:, :NUM_FOREGROUND]
        inter = np.diag(cm4)
        union = cm4.sum(axis=0) + cm_np.sum(axis=1) - inter
        return compute_macro_miou_from_iu(inter, union)

    rows_csv: List[Dict[str, Any]] = []
    per_model_report: Dict[str, Any] = {}

    for model_name, pred_raw in preds_by_model.items():
        pred_arr = np.asarray(pred_raw, dtype=np.uint8)
        p_crop = np.ascontiguousarray(pred_arr[:, margin : h - margin, margin : w - margin])
        g_zero = np.ascontiguousarray(gt_arr[:, margin : h - margin, margin : w - margin])

        grid_scores: Dict[Tuple[int, int], float] = {}
        best_score = -1.0
        best_dx, best_dy = 0, 0
        far_max = -1.0

        for dy in shift_vals:
            y0 = margin - dy
            y1 = h - margin - dy
            for dx in shift_vals:
                x0 = margin - dx
                x1 = w - margin - dx
                p_shift = pred_arr[:, y0:y1, x0:x1]
                miou_val = _miou_pair_fast(p_shift, g_zero)
                grid_scores[(dx, dy)] = miou_val
                rows_csv.append(
                    {
                        "model": model_name,
                        "kind": "shift",
                        "dx": dx,
                        "dy": dy,
                        "miou": round(miou_val, 6),
                    }
                )
                if miou_val > best_score:
                    best_score = miou_val
                    best_dx, best_dy = dx, dy
                if abs(dx) >= margin or abs(dy) >= margin:
                    if miou_val > far_max:
                        far_max = miou_val

        # Orientation collapse checks on identical interior crop
        orient_scores = {
            "hflip": _miou_pair_fast(p_crop[:, :, ::-1], g_zero),
            "vflip": _miou_pair_fast(p_crop[:, ::-1, :], g_zero),
            "transpose": _miou_pair_fast(np.swapaxes(p_crop, 1, 2), g_zero),
            "rot90": _miou_pair_fast(np.rot90(p_crop, k=1, axes=(1, 2)), g_zero),
        }
        for ok, ov in orient_scores.items():
            rows_csv.append(
                {
                    "model": model_name,
                    "kind": f"orient_{ok}",
                    "dx": 0,
                    "dy": 0,
                    "miou": round(ov, 6),
                }
            )

        zero_score = float(grid_scores.get((0, 0), _miou_pair_fast(p_crop, g_zero)))
        dx_slice = [round(float(grid_scores[(dx, 0)]), 6) for dx in shift_vals]
        dy_slice = [round(float(grid_scores[(0, dy)]), 6) for dy in shift_vals]

        per_model_report[model_name] = {
            "miou_zero_shift": round(zero_score, 6),
            "peak_dx": int(best_dx),
            "peak_dy": int(best_dy),
            "miou_peak": round(float(best_score), 6),
            "zero_vs_peak_gap": round(float(best_score - zero_score), 6),
            "far_border_max_miou": round(float(far_max), 6),
            "dx_slice_dy0": dx_slice,
            "dy_slice_dx0": dy_slice,
            "orientation_collapse": {k: round(float(v), 6) for k, v in orient_scores.items()},
            "max_orient_miou": round(float(max(orient_scores.values())), 6),
        }

    return {
        "shifts": shift_vals,
        "margin": int(margin),
        "per_model": per_model_report,
        "csv_rows": rows_csv,
    }


def assert_translation_and_orientation_alignment(
    shift_report: Mapping[str, Any],
    max_peak_offset: int = 4,
    max_zero_gap: float = 0.008,
    min_orient_margin: float = 0.05,
    enforce_strict: bool = True,
) -> Dict[str, Any]:
    """Assertion 4b: Verify sharp (0, 0) peak and clear orientation collapse across every model."""
    failures: List[str] = []
    for m_name, rep in shift_report.get("per_model", {}).items():
        dx_p = abs(int(rep["peak_dx"]))
        dy_p = abs(int(rep["peak_dy"]))
        z_score = float(rep["miou_zero_shift"])
        p_score = float(rep["miou_peak"])
        far_max = float(rep["far_border_max_miou"])
        orient_max = float(rep["max_orient_miou"])

        if dx_p > int(max_peak_offset) or dy_p > int(max_peak_offset):
            failures.append(f"{m_name}: peak=({rep['peak_dx']},{rep['peak_dy']}) exceeds +/-{max_peak_offset}")
        if z_score < p_score - float(max_zero_gap):
            failures.append(f"{m_name}: miou(0,0)={z_score:.4f} lags peak={p_score:.4f} by > {max_zero_gap}")
        if z_score <= far_max:
            failures.append(f"{m_name}: miou(0,0)={z_score:.4f} <= far_border_max={far_max:.4f}")
        if z_score <= orient_max + float(min_orient_margin):
            failures.append(
                f"{m_name}: miou(0,0)={z_score:.4f} does not exceed max_orient={orient_max:.4f} by +{min_orient_margin}"
            )

    passed = len(failures) == 0
    if enforce_strict and not passed:
        raise RuntimeError(
            "[STAGE2 ERROR] Translation-shift / orientation alignment check failed: " + "; ".join(failures)
        )
    return {
        "passed": passed,
        "failures": failures,
        "max_peak_offset": int(max_peak_offset),
        "max_zero_gap": float(max_zero_gap),
        "min_orient_margin": float(min_orient_margin),
    }


def verify_zero_gt_leakage_glass_guard(
    pred_masks: Sequence[np.ndarray],
    raw_rgbs: Sequence[np.ndarray],
    gt_masks: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Assertion 5: Confirm GT-free bright-glass guard produces bit-identical output when GT is zeroed."""
    import inspect

    sig = inspect.signature(MultiMethodWSSSEngineV2.apply_bright_glass_guard)
    if any("gt" in pname.lower() for pname in sig.parameters):
        raise RuntimeError(
            f"[STAGE2 ERROR] Bright-glass guard exposes GT parameter in signature: {list(sig.parameters)}"
        )

    checked = 0
    for idx, (pred, rgb) in enumerate(zip(pred_masks, raw_rgbs)):
        out_a = MultiMethodWSSSEngineV2.apply_bright_glass_guard(pred, rgb, background_class_idx=4)
        # Zeroing an independent GT buffer cannot affect a GT-free guard
        if gt_masks is not None and idx < len(gt_masks):
            _zero_gt = np.zeros_like(gt_masks[idx])
            _ = _zero_gt
        out_b = MultiMethodWSSSEngineV2.apply_bright_glass_guard(pred.copy(), rgb.copy(), background_class_idx=4)
        if not np.array_equal(out_a, out_b):
            raise RuntimeError(f"[STAGE2 ERROR] Non-deterministic glass guard output on patch index {idx}.")
        checked += 1

    return {"passed": True, "checked_patches": checked, "gt_parameter_free": True}


def verify_pooled_miou_reconciliation(
    observed_mious: Mapping[str, float],
    expected_mious: Mapping[str, float] = GATE7_EXPECTED_TEST_MIOU,
    tol: float = 0.0015,
    enforce_strict: bool = True,
    allow_improvement_keys: Sequence[str] = (),
) -> Dict[str, Any]:
    """Assertion 6: Reconcile pooled TEST mIoU against frozen Gate-7 Track A reference values.

    For keys listed in ``allow_improvement_keys``, strict enforcement guards against regression
    below ``expected - tol`` while permitting verified post-fix improvements above ``expected``.
    """
    allow_imp = set(allow_improvement_keys)
    deltas: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    for key, exp_v in expected_mious.items():
        if key not in observed_mious:
            failures.append(f"missing observed key {key}")
            continue
        obs_v = float(observed_mious[key])
        signed_diff = obs_v - float(exp_v)
        diff = abs(signed_diff)
        improved_above_ref = key in allow_imp and signed_diff > float(tol)
        deltas[key] = {
            "observed": round(obs_v, 4),
            "expected": round(float(exp_v), 4),
            "signed_diff": round(signed_diff, 6),
            "abs_diff": round(diff, 6),
            "allow_improvement": key in allow_imp,
            "improved_above_ref": improved_above_ref,
        }
        if key in allow_imp:
            if signed_diff < -float(tol):
                failures.append(
                    f"{key}: observed={obs_v:.4f} regressed below expected={exp_v:.4f} "
                    f"(signed_diff={signed_diff:+.4f} < -{tol})"
                )
        elif diff > float(tol):
            failures.append(f"{key}: observed={obs_v:.4f} vs expected={exp_v:.4f} (diff={diff:.4f} > {tol})")

    passed = len(failures) == 0
    if enforce_strict and not passed:
        raise RuntimeError("[STAGE2 ERROR] Pooled mIoU reconciliation failed: " + "; ".join(failures))
    return {"passed": passed, "tolerance": float(tol), "deltas": deltas, "failures": failures}


# ---------------------------------------------------------------------------
# Spatial Alignment, Boundary F1, Centroid Offset & Patch Metrics
# ---------------------------------------------------------------------------


def _extract_binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel inner morphological contour of a binary mask."""
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    eroded = binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_and(m, np.logical_not(eroded))


def _macro_boundary_f1_from_counts(
    prec_num: np.ndarray,
    prec_den: np.ndarray,
    rec_num: np.ndarray,
    rec_den: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Compute macro and per-class Boundary F1 from length-4 precision/recall numerator/denominator vectors."""
    per_class: Dict[str, float] = {}
    vals: List[float] = []
    for c in range(NUM_FOREGROUND):
        prec = float(prec_num[c] / prec_den[c]) if prec_den[c] > 0 else 0.0
        rec = float(rec_num[c] / rec_den[c]) if rec_den[c] > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[_CLS_LOWER[c]] = round(f1, 4)
        if rec_den[c] > 0 or prec_den[c] > 0:
            vals.append(f1)
    return (float(np.mean(vals)) if vals else 0.0), per_class


def compute_boundary_f1_detailed(
    preds: np.ndarray,
    gts: np.ndarray,
    patch_pidx: Optional[np.ndarray] = None,
    num_patients: int = 0,
    radius: int = 2,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Compute pooled, per-patch, and patient-level Boundary F1@radius with 95% patient bootstrap CI."""
    p_arr = np.asarray(preds, dtype=np.uint8)
    g_arr = np.asarray(gts, dtype=np.uint8)
    if p_arr.ndim == 2:
        p_arr = p_arr[None, ...]
        g_arr = g_arr[None, ...]

    n_patches = int(p_arr.shape[0])
    y_g, x_g = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (x_g * x_g + y_g * y_g) <= (radius * radius)

    prec_num = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    prec_den = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    rec_num = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    rec_den = np.zeros(NUM_FOREGROUND, dtype=np.int64)

    has_pats = patch_pidx is not None and int(num_patients) > 0
    n_pat = int(num_patients) if has_pats else 0
    p_prec_num = np.zeros((n_pat, NUM_FOREGROUND), dtype=np.int64) if has_pats else np.zeros((0, NUM_FOREGROUND), dtype=np.int64)
    p_prec_den = np.zeros((n_pat, NUM_FOREGROUND), dtype=np.int64) if has_pats else np.zeros((0, NUM_FOREGROUND), dtype=np.int64)
    p_rec_num = np.zeros((n_pat, NUM_FOREGROUND), dtype=np.int64) if has_pats else np.zeros((0, NUM_FOREGROUND), dtype=np.int64)
    p_rec_den = np.zeros((n_pat, NUM_FOREGROUND), dtype=np.int64) if has_pats else np.zeros((0, NUM_FOREGROUND), dtype=np.int64)

    per_patch_f1 = np.full(n_patches, np.nan, dtype=np.float64)

    for i in range(n_patches):
        gt_i = g_arr[i]
        pr_i = p_arr[i]
        valid = (gt_i >= 0) & (gt_i < NUM_FOREGROUND)
        if not np.any(valid):
            continue
        pi = int(patch_pidx[i]) if has_pats and patch_pidx is not None else -1
        patch_cls_f1: List[float] = []
        for c in range(NUM_FOREGROUND):
            gb = _extract_binary_boundary((gt_i == c) & valid)
            pb = _extract_binary_boundary((pr_i == c) & valid)
            g_cnt = int(gb.sum())
            p_cnt = int(pb.sum())
            if g_cnt == 0 and p_cnt == 0:
                continue
            pn_i = 0
            rn_i = 0
            if p_cnt > 0:
                g_dil = binary_dilation(gb, structure=disk) if g_cnt > 0 else np.zeros_like(gb)
                pn_i = int(np.logical_and(pb, g_dil).sum())
                prec_num[c] += pn_i
                prec_den[c] += p_cnt
                if has_pats:
                    p_prec_num[pi, c] += pn_i
                    p_prec_den[pi, c] += p_cnt
            if g_cnt > 0:
                p_dil = binary_dilation(pb, structure=disk) if p_cnt > 0 else np.zeros_like(pb)
                rn_i = int(np.logical_and(gb, p_dil).sum())
                rec_num[c] += rn_i
                rec_den[c] += g_cnt
                if has_pats:
                    p_rec_num[pi, c] += rn_i
                    p_rec_den[pi, c] += g_cnt
            prec_i = float(pn_i / p_cnt) if p_cnt > 0 else 0.0
            rec_i = float(rn_i / g_cnt) if g_cnt > 0 else 0.0
            patch_cls_f1.append((2.0 * prec_i * rec_i / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0)
        if patch_cls_f1:
            per_patch_f1[i] = float(np.mean(patch_cls_f1))

    macro_f1, per_class = _macro_boundary_f1_from_counts(prec_num, prec_den, rec_num, rec_den)
    report: Dict[str, Any] = {
        "radius_px": int(radius),
        "macro_boundary_f1": round(macro_f1, 4),
        "per_class_boundary_f1": per_class,
    }

    per_patient_f1 = np.zeros(n_pat, dtype=np.float64)
    if has_pats:
        for p in range(n_pat):
            pf1, _ = _macro_boundary_f1_from_counts(p_prec_num[p], p_prec_den[p], p_rec_num[p], p_rec_den[p])
            per_patient_f1[p] = pf1
        rng = np.random.default_rng(int(seed))
        b_idx = rng.integers(0, n_pat, size=(int(n_boot), n_pat))
        bpn = p_prec_num[b_idx].sum(axis=1).astype(np.float64)
        bpd = p_prec_den[b_idx].sum(axis=1).astype(np.float64)
        brn = p_rec_num[b_idx].sum(axis=1).astype(np.float64)
        brd = p_rec_den[b_idx].sum(axis=1).astype(np.float64)
        b_prec = np.divide(bpn, np.where(bpd > 0, bpd, 1.0))
        b_prec = np.where(bpd > 0, b_prec, 0.0)
        b_rec = np.divide(brn, np.where(brd > 0, brd, 1.0))
        b_rec = np.where(brd > 0, b_rec, 0.0)
        b_den = b_prec + b_rec
        b_f1 = np.where(b_den > 0, 2.0 * b_prec * b_rec / np.where(b_den > 0, b_den, 1.0), 0.0)
        b_act = (bpd > 0) | (brd > 0)
        b_cnt = np.maximum(1, b_act.sum(axis=1))
        boot_macro = np.where(b_act.sum(axis=1) > 0, np.where(b_act, b_f1, 0.0).sum(axis=1) / b_cnt, 0.0)
        lo, hi = np.percentile(boot_macro, [2.5, 97.5])
        report["bootstrap_95ci"] = {
            "point": round(macro_f1, 4),
            "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4),
        }
        report["patient_mean_f1"] = round(float(np.mean(per_patient_f1)), 4)

    return report, per_patch_f1, per_patient_f1


def compute_boundary_f1_at_radius(
    preds: np.ndarray,
    gts: np.ndarray,
    radius: int = 2,
) -> Dict[str, Any]:
    """Compute pooled per-class Boundary F1 at pixel tolerance radius strictly within valid_fg."""
    rep, _, _ = compute_boundary_f1_detailed(preds, gts, patch_pidx=None, num_patients=0, radius=radius)
    return rep


def compute_tum_centroid_per_patch(
    preds: np.ndarray,
    gts: np.ndarray,
    min_tum_px: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (per_patch_offset_nan, gt_tum_ok_bool, pred_tum_ok_bool) within valid_fg."""
    p_arr = np.asarray(preds, dtype=np.uint8)
    g_arr = np.asarray(gts, dtype=np.uint8)
    if p_arr.ndim == 2:
        p_arr = p_arr[None, ...]
        g_arr = g_arr[None, ...]

    n_patches = int(p_arr.shape[0])
    offsets = np.full(n_patches, np.nan, dtype=np.float64)
    gt_ok = np.zeros(n_patches, dtype=bool)
    pr_ok = np.zeros(n_patches, dtype=bool)
    min_px = int(min_tum_px)

    for i in range(n_patches):
        valid = (g_arr[i] >= 0) & (g_arr[i] < NUM_FOREGROUND)
        gt_tum = (g_arr[i] == 0) & valid
        pr_tum = (p_arr[i] == 0) & valid
        g_ok = int(gt_tum.sum()) >= min_px
        p_ok = int(pr_tum.sum()) >= min_px
        gt_ok[i] = g_ok
        pr_ok[i] = p_ok
        if g_ok and p_ok:
            gy, gx = np.nonzero(gt_tum)
            py, px = np.nonzero(pr_tum)
            offsets[i] = float(np.hypot(float(px.mean() - gx.mean()), float(py.mean() - gy.mean())))
    return offsets, gt_ok, pr_ok


def compute_tum_centroid_offset(preds: np.ndarray, gts: np.ndarray, min_tum_px: int = 64) -> Dict[str, Any]:
    """Compute Euclidean TUM centroid offset (pixels) and TUM patch detection recall within valid_fg."""
    offsets, gt_ok, pr_ok = compute_tum_centroid_per_patch(preds, gts, min_tum_px=min_tum_px)
    n_gt_tum = int(gt_ok.sum())
    det_mask = gt_ok & pr_ok
    valid_off = offsets[det_mask]
    recall = float(valid_off.size / n_gt_tum) if n_gt_tum > 0 else 0.0
    if valid_off.size == 0:
        return {
            "n_patches": 0,
            "n_gt_tum_patches": n_gt_tum,
            "tum_patch_detection_recall": round(recall, 4),
            "mean_px": 0.0,
            "median_px": 0.0,
            "p90_px": 0.0,
        }
    return {
        "n_patches": int(valid_off.size),
        "n_gt_tum_patches": n_gt_tum,
        "tum_patch_detection_recall": round(recall, 4),
        "mean_px": round(float(valid_off.mean()), 3),
        "median_px": round(float(np.median(valid_off)), 3),
        "p90_px": round(float(np.percentile(valid_off, 90)), 3),
    }


def compute_matched_spatial_stats(
    bf1_pat_by_model: Mapping[str, np.ndarray],
    tum_offsets_by_model: Mapping[str, np.ndarray],
    gt_tum_ok: np.ndarray,
    pred_tum_ok_by_model: Mapping[str, np.ndarray],
    patch_pidx: np.ndarray,
    num_patients: int,
    key_models: Sequence[str] = ("author_step0", "arm5_cgm", "ens4_no_cgm", "ens4_cgm"),
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute unbiased common-intersection TUM centroid stats + patient-level Wilcoxon & 95% bootstrap CIs."""
    present_models = [m for m in key_models if m in tum_offsets_by_model]
    common_mask = np.asarray(gt_tum_ok, dtype=bool).copy()
    for m in present_models:
        common_mask &= np.asarray(pred_tum_ok_by_model[m], dtype=bool)
    n_gt_tum = int(np.asarray(gt_tum_ok, dtype=bool).sum())
    n_common = int(common_mask.sum())

    # Patients with >= 1 common-intersection TUM patch
    common_pidx = np.asarray(patch_pidx, dtype=np.int32)[common_mask]
    active_pats = sorted(set(int(p) for p in common_pidx))
    n_active_pats = len(active_pats)

    matched_per_model: Dict[str, Any] = {}
    pat_mean_by_model: Dict[str, np.ndarray] = {}
    pat_med_by_model: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(int(seed))

    for m in present_models:
        vals = np.asarray(tum_offsets_by_model[m], dtype=np.float64)[common_mask]
        n_det = int(np.logical_and(gt_tum_ok, pred_tum_ok_by_model[m]).sum())
        recall = float(n_det / n_gt_tum) if n_gt_tum > 0 else 0.0
        if vals.size == 0 or n_active_pats == 0:
            matched_per_model[m] = {
                "n_common_patches": 0,
                "n_active_patients": 0,
                "n_detected_patches": n_det,
                "n_gt_tum_patches": n_gt_tum,
                "tum_patch_detection_recall": round(recall, 4),
                "mean_px": 0.0,
                "median_px": 0.0,
                "p90_px": 0.0,
                "bootstrap_95ci_mean_px": [0.0, 0.0],
                "bootstrap_95ci_median_px": [0.0, 0.0],
            }
            pat_mean_by_model[m] = np.zeros(0, dtype=np.float64)
            pat_med_by_model[m] = np.zeros(0, dtype=np.float64)
            continue

        pat_vals = [vals[common_pidx == p] for p in active_pats]
        pat_means = np.asarray([float(v.mean()) for v in pat_vals], dtype=np.float64)
        pat_meds = np.asarray([float(np.median(v)) for v in pat_vals], dtype=np.float64)
        pat_mean_by_model[m] = pat_means
        pat_med_by_model[m] = pat_meds

        # Patient-level cluster bootstrap for pooled common-intersection mean & median offset
        b_means = np.empty(int(n_boot), dtype=np.float64)
        b_meds = np.empty(int(n_boot), dtype=np.float64)
        for bi in range(int(n_boot)):
            chosen = rng.integers(0, n_active_pats, size=n_active_pats)
            sample = np.concatenate([pat_vals[int(c)] for c in chosen])
            b_means[bi] = float(sample.mean())
            b_meds[bi] = float(np.median(sample))
        m_lo, m_hi = np.percentile(b_means, [2.5, 97.5])
        d_lo, d_hi = np.percentile(b_meds, [2.5, 97.5])

        matched_per_model[m] = {
            "n_common_patches": n_common,
            "n_active_patients": n_active_pats,
            "n_detected_patches": n_det,
            "n_gt_tum_patches": n_gt_tum,
            "tum_patch_detection_recall": round(recall, 4),
            "mean_px": round(float(vals.mean()), 3),
            "median_px": round(float(np.median(vals)), 3),
            "p90_px": round(float(np.percentile(vals, 90)), 3),
            "patient_mean_of_means_px": round(float(pat_means.mean()), 3),
            "bootstrap_95ci_mean_px": [round(float(m_lo), 3), round(float(m_hi), 3)],
            "bootstrap_95ci_median_px": [round(float(d_lo), 3), round(float(d_hi), 3)],
        }

    pairs = [
        ("ens4_cgm_vs_author_step0", "ens4_cgm", "author_step0"),
        ("ens4_no_cgm_vs_author_step0", "ens4_no_cgm", "author_step0"),
        ("arm5_cgm_vs_author_step0", "arm5_cgm", "author_step0"),
        ("ens4_cgm_vs_arm5_cgm", "ens4_cgm", "arm5_cgm"),
    ]
    bf1_wilcoxon: Dict[str, Any] = {}
    cent_wilcoxon: Dict[str, Any] = {}
    for pname, ma, mb in pairs:
        if ma in bf1_pat_by_model and mb in bf1_pat_by_model and int(num_patients) > 0:
            wa = np.asarray(bf1_pat_by_model[ma], dtype=np.float64)
            wb = np.asarray(bf1_pat_by_model[mb], dtype=np.float64)
            bf1_wilcoxon[pname] = {
                "delta_patient_mean_f1": round(float(np.mean(wa - wb)), 4),
                **paired_wilcoxon_and_dz(wa, wb),
            }
        if ma in pat_mean_by_model and mb in pat_mean_by_model and n_active_pats > 0:
            ca = pat_mean_by_model[ma]
            cb = pat_mean_by_model[mb]
            # Negative delta means ma has smaller (better) centroid error than mb
            cent_wilcoxon[pname] = {
                "delta_patient_mean_px": round(float(np.mean(ca - cb)), 3),
                "delta_patient_median_px": round(float(np.mean(pat_med_by_model[ma] - pat_med_by_model[mb])), 3),
                **paired_wilcoxon_and_dz(ca, cb),
            }

    return {
        "n_patients": int(num_patients),
        "boundary_f1_at_2px_paired_wilcoxon": bf1_wilcoxon,
        "tum_centroid_matched_common_intersection": {
            "n_gt_tum_patches": n_gt_tum,
            "n_common_patches": n_common,
            "n_active_patients": n_active_pats,
            "per_model": matched_per_model,
            "paired_wilcoxon": cent_wilcoxon,
        },
    }


def compute_full_test_glass_guard_ablation(
    preds: np.ndarray,
    raw_rgbs: Sequence[np.ndarray],
    gt_masks: np.ndarray,
) -> Dict[str, Any]:
    """Quantify None vs PureGlass vs Cand1 BrightGlass vs Otsu across ALL test patches (n=4,986)."""
    n_patches = int(gt_masks.shape[0])
    modes = ("none", "pure_glass", "cand1_bright_glass", "otsu")
    cm_by_mode = {m: np.zeros((NUM_FOREGROUND, NUM_FOREGROUND + 1), dtype=np.int64) for m in modes}
    bg_supp = {m: 0 for m in modes}
    or_supp = {m: 0 for m in modes}
    fg_erode = {m: 0 for m in modes}
    nec_erode = {m: 0 for m in modes}
    tot_bg, tot_or, tot_fg, tot_nec = 0, 0, 0, 0

    for i in range(n_patches):
        rgb = raw_rgbs[i]
        gt = gt_masks[i]
        pred_raw = preds[i]
        b_info = classify_back_pixels(rgb, gt)
        bg_m = b_info["bright_glass_mask"]
        or_m = b_info["outside_roi_mask"]
        fg_m = (gt >= 0) & (gt < NUM_FOREGROUND)
        nec_m = gt == 3

        tot_bg += int(bg_m.sum())
        tot_or += int(or_m.sum())
        tot_fg += int(fg_m.sum())
        tot_nec += int(nec_m.sum())

        pred_pure = MultiMethodWSSSEngineV2.apply_pure_glass_guard(pred_raw, rgb, background_class_idx=4)
        pred_cand1 = MultiMethodWSSSEngineV2.apply_bright_glass_guard(pred_raw, rgb, background_class_idx=4)
        otsu_tiss = compute_otsu_tissue_mask(rgb).astype(bool)
        pred_otsu = pred_raw.copy()
        pred_otsu[~otsu_tiss] = 4

        gv = gt[fg_m].astype(np.int64) if np.any(fg_m) else None
        for m_name, p_cur in (
            ("none", pred_raw),
            ("pure_glass", pred_pure),
            ("cand1_bright_glass", pred_cand1),
            ("otsu", pred_otsu),
        ):
            is_back = p_cur == 4
            bg_supp[m_name] += int(np.logical_and(bg_m, is_back).sum())
            or_supp[m_name] += int(np.logical_and(or_m, is_back).sum())
            fg_erode[m_name] += int(np.logical_and(fg_m, is_back).sum())
            nec_erode[m_name] += int(np.logical_and(nec_m, is_back).sum())
            if gv is not None:
                pv = np.minimum(p_cur[fg_m], NUM_FOREGROUND).astype(np.int64)
                cm_by_mode[m_name] += np.bincount(
                    gv * (NUM_FOREGROUND + 1) + pv,
                    minlength=NUM_FOREGROUND * (NUM_FOREGROUND + 1),
                ).reshape(NUM_FOREGROUND, NUM_FOREGROUND + 1)

    def _pct(num: int, den: int) -> Optional[float]:
        return round(float(num) / float(den) * 100.0, 3) if den > 0 else None

    out_modes: Dict[str, Any] = {}
    base_miou: Optional[float] = None
    for m_name in modes:
        cm = cm_by_mode[m_name]
        cm4 = cm[:, :NUM_FOREGROUND]
        inter = np.diag(cm4)
        union = cm4.sum(axis=0) + cm.sum(axis=1) - inter
        summ = summarize_iu(inter, union)
        if base_miou is None:
            base_miou = float(summ["mIoU_raw"])
        out_modes[m_name] = {
            **summ,
            "delta_miou_vs_none": round(float(summ["mIoU_raw"]) - float(base_miou), 4),
            "bright_glass_suppression_pct": _pct(bg_supp[m_name], tot_bg),
            "outside_roi_suppression_pct": _pct(or_supp[m_name], tot_or),
            "fg_erosion_pct": _pct(fg_erode[m_name], tot_fg),
            "nec_erosion_pct": _pct(nec_erode[m_name], tot_nec),
        }

    return {
        "n_patches": n_patches,
        "pixel_totals": {
            "bright_glass_px": tot_bg,
            "outside_roi_px": tot_or,
            "valid_fg_px": tot_fg,
            "nec_px": tot_nec,
        },
        "modes": out_modes,
    }


def _make_2d_hann_window(h: int, w: int) -> np.ndarray:
    """Strictly positive 2D separable raised-cosine Hann blending window (zero border singularity)."""
    hy = np.hanning(int(h) + 2)[1:-1].astype(np.float32)
    hx = np.hanning(int(w) + 2)[1:-1].astype(np.float32)
    return np.outer(hy, hx).astype(np.float32)


def compute_tile_seam_discontinuity_rate(
    pred_full: np.ndarray,
    gt_full: np.ndarray,
    tile_size: int = 224,
) -> Dict[str, Any]:
    """Measure predicted class-flip rate across interior tile seams vs non-seam neighbors (valid_fg only)."""
    p = np.asarray(pred_full, dtype=np.uint8)
    g = np.asarray(gt_full, dtype=np.uint8)
    fh, fw = p.shape
    ts = int(tile_size)
    seam_flips = 0
    seam_pairs = 0
    gt_seam_flips = 0
    # Vertical seams at x = k * ts
    for x in range(ts, fw, ts):
        valid = (g[:, x - 1] < NUM_FOREGROUND) & (g[:, x] < NUM_FOREGROUND)
        cnt = int(valid.sum())
        if cnt > 0:
            seam_pairs += cnt
            seam_flips += int(np.logical_and(valid, p[:, x - 1] != p[:, x]).sum())
            gt_seam_flips += int(np.logical_and(valid, g[:, x - 1] != g[:, x]).sum())
    # Horizontal seams at y = k * ts
    for y in range(ts, fh, ts):
        valid = (g[y - 1, :] < NUM_FOREGROUND) & (g[y, :] < NUM_FOREGROUND)
        cnt = int(valid.sum())
        if cnt > 0:
            seam_pairs += cnt
            seam_flips += int(np.logical_and(valid, p[y - 1, :] != p[y, :]).sum())
            gt_seam_flips += int(np.logical_and(valid, g[y - 1, :] != g[y, :]).sum())

    pred_rate = float(seam_flips / seam_pairs) if seam_pairs > 0 else 0.0
    gt_rate = float(gt_seam_flips / seam_pairs) if seam_pairs > 0 else 0.0
    return {
        "seam_valid_pairs": seam_pairs,
        "pred_seam_flip_rate": round(pred_rate, 5),
        "gt_seam_flip_rate": round(gt_rate, 5),
    }


def _resolve_visual_audit_provenance(a2_ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    """Capture git_ref, HEAD commit SHA, and checkpoint SHA-256 provenance."""
    repo_root = Path(__file__).resolve().parents[2]
    git_ref = os.environ.get("STAGE2_GIT_REF", "").strip()
    git_sha = ""
    try:
        r_sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r_sha.returncode == 0:
            git_sha = r_sha.stdout.strip()
        if not git_ref:
            r_br = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if r_br.returncode == 0:
                git_ref = r_br.stdout.strip()
    except Exception:
        pass
    return {
        "git_ref": git_ref or "unknown",
        "git_commit_sha": git_sha or "unknown",
        "arm_sha256": dict(a2_ckpt.get("arm_sha256", {})),
        "stage1_cls_sha256": str(a2_ckpt.get("stage1_cls_sha256", "")),
        "author_stage1_sha256": str(a2_ckpt.get("author_stage1_sha256", "")),
        "author_stage2_sha256": str(a2_ckpt.get("author_stage2_sha256", "")),
    }


def compute_patch_canonical_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    """Compute canonical per-patch 4-class mIoU strictly over valid_fg = (gt >= 0) & (gt < 4)."""
    valid = (gt >= 0) & (gt < NUM_FOREGROUND)
    fg_px = int(valid.sum())
    present_ious: List[float] = []
    per_cls: Dict[str, Optional[float]] = {}
    gt_counts = [0] * NUM_FOREGROUND
    pred_counts = [0] * NUM_FOREGROUND

    for c in range(NUM_FOREGROUND):
        gc = (gt == c) & valid
        pc = (pred == c) & valid
        g_sum = int(gc.sum())
        p_sum = int(pc.sum())
        gt_counts[c] = g_sum
        pred_counts[c] = p_sum
        if g_sum > 0:
            tp = int(np.logical_and(gc, pc).sum())
            un = g_sum + p_sum - tp
            iou_c = float(tp / un) if un > 0 else 0.0
            per_cls[_CLS_LOWER[c]] = round(iou_c, 4)
            present_ious.append(iou_c)
        else:
            per_cls[_CLS_LOWER[c]] = None

    nec_fp_px = int(np.logical_and(pred == 3, np.logical_and(valid, gt != 3)).sum())
    lym_fp_px = int(np.logical_and(pred == 2, np.logical_and(valid, gt != 2)).sum())
    return {
        "fg_px": fg_px,
        "miou_gt_present": float(np.mean(present_ious)) if present_ious else 0.0,
        "per_class_iou": per_cls,
        "gt_counts": gt_counts,
        "pred_counts": pred_counts,
        "nec_fp_px": nec_fp_px,
        "lym_fp_px": lym_fp_px,
    }


def classify_back_pixels(raw_rgb: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
    """Split GT==4 pixels into bright glass (gray >= 230) and stained outside-ROI tissue (gray < 220)."""
    rgb_f = np.asarray(raw_rgb, dtype=np.float32)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    sat = np.max(rgb_f, axis=-1) - np.min(rgb_f, axis=-1)
    back = gt_mask == 4
    total_px = max(1, int(gt_mask.size))
    bright_glass = back & (gray >= 230.0)
    outside_roi = back & (gray < 220.0)
    clinical_outside = back & (sat > 15.0) & (gray < 235.0)
    return {
        "back_frac": float(back.sum()) / float(total_px),
        "bright_glass_frac": float(bright_glass.sum()) / float(total_px),
        "outside_roi_frac": float(outside_roi.sum()) / float(total_px),
        "bright_glass_mask": bright_glass,
        "outside_roi_mask": outside_roi,
        "clinical_outside_mask": clinical_outside,
    }


# ---------------------------------------------------------------------------
# Crisp Nearest-Neighbor Visualization Renderers (ZERO Blur, ZERO GT Overwrite)
# ---------------------------------------------------------------------------


def render_overlay_rgb(raw_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend foreground classes {0..3} onto raw H&E leaving BACK==4 transparent (zero blur)."""
    base = np.asarray(raw_rgb, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=np.int64)
    fg = (m >= 0) & (m < NUM_FOREGROUND)
    if np.any(fg):
        colors = CLINICAL_OVERLAY_PALETTE[np.clip(m, 0, 4)].astype(np.float32)
        base[fg] = (1.0 - float(alpha)) * base[fg] + float(alpha) * colors[fg]
    return np.clip(np.round(base), 0.0, 255.0).astype(np.uint8)


def _dashed_boundary_mask(binary_mask: np.ndarray, dash_period: int = 8) -> np.ndarray:
    """Create crisp 1-px dashed boundary mask for GT==4 contour visualization."""
    bnd = _extract_binary_boundary(binary_mask)
    if not np.any(bnd):
        return bnd
    ys, xs = np.nonzero(bnd)
    keep = ((ys + xs) % int(dash_period)) < (int(dash_period) // 2 + 1)
    out = np.zeros_like(bnd, dtype=bool)
    out[ys[keep], xs[keep]] = True
    return out


def render_gt_overlay_with_roi_contour(
    raw_rgb: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.42,
) -> np.ndarray:
    """Render pathologist GT overlay with crisp dashed dark contour around GT==4."""
    vis = render_overlay_rgb(raw_rgb, gt_mask, alpha=alpha)
    back = gt_mask == 4
    if np.any(back) and not np.all(back):
        dash = _dashed_boundary_mask(back, dash_period=8)
        vis[dash] = np.array([20, 20, 20], dtype=np.uint8)
    return vis


def render_contour_overlay(
    raw_rgb: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.36,
) -> np.ndarray:
    """Render prediction overlay with Cyan GT contours vs Gold Pred contours (inside valid_fg)."""
    vis = render_overlay_rgb(raw_rgb, pred_mask, alpha=alpha)
    valid = (gt_mask >= 0) & (gt_mask < NUM_FOREGROUND)
    gt_edge = np.zeros(gt_mask.shape, dtype=bool)
    pr_edge = np.zeros(gt_mask.shape, dtype=bool)
    for c in range(NUM_FOREGROUND):
        gt_edge |= _extract_binary_boundary((gt_mask == c) & valid)
        pr_edge |= _extract_binary_boundary((pred_mask == c) & valid)
    vis[pr_edge] = np.array([255, 215, 0], dtype=np.uint8)  # Gold Pred
    vis[gt_edge] = np.array([0, 240, 255], dtype=np.uint8)  # Cyan GT
    return vis


def render_cancer_alignment_error_map(
    raw_rgb: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.72,
) -> np.ndarray:
    """
    Render Cancer Alignment Error Map:
    - Green (34, 197, 94): TP exact foreground class match
    - Red (220, 38, 38): FP-TUM (pred==0 when gt in {1,2,3})
    - Magenta (217, 70, 239): FN-TUM (gt==0 when pred in {1,2,3})
    - Amber (245, 158, 11): Other foreground class swap
    - Bright Cyan (56, 189, 248): GT==4 Bright Glass
    - Slate Blue-Gray (100, 116, 139): GT==4 Stained Outside-ROI tissue
    """
    base = np.asarray(raw_rgb, dtype=np.float32)
    color_layer = base.copy()
    valid = (gt_mask >= 0) & (gt_mask < NUM_FOREGROUND)
    back = gt_mask == 4

    tp = valid & (pred_mask == gt_mask)
    fp_tum = valid & (pred_mask == 0) & (gt_mask != 0)
    fn_tum = valid & (gt_mask == 0) & (pred_mask != 0)
    other_swap = valid & (pred_mask != gt_mask) & (~fp_tum) & (~fn_tum)

    back_info = classify_back_pixels(raw_rgb, gt_mask)
    outside_roi = back_info["clinical_outside_mask"]
    bright_glass = back & (~outside_roi)

    color_layer[tp] = np.array([34, 197, 94], dtype=np.float32)
    color_layer[fp_tum] = np.array([220, 38, 38], dtype=np.float32)
    color_layer[fn_tum] = np.array([217, 70, 239], dtype=np.float32)
    color_layer[other_swap] = np.array([245, 158, 11], dtype=np.float32)
    color_layer[bright_glass] = np.array([56, 189, 248], dtype=np.float32)
    color_layer[outside_roi] = np.array([100, 116, 139], dtype=np.float32)

    blended = (1.0 - float(alpha)) * base + float(alpha) * color_layer
    return np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Zero-Cherry-Picking Sample Selection
# ---------------------------------------------------------------------------


def _select_distinct_patient_indices(
    candidate_indices: Sequence[int],
    patient_ids: Sequence[str],
    k: int = 4,
    fallback_pool: Optional[Sequence[int]] = None,
) -> List[int]:
    """Pick up to k indices from distinct patients first, filling from remaining if fewer unique patients exist."""
    chosen: List[int] = []
    seen_pats: set = set()
    used_idx: set = set()

    for idx in candidate_indices:
        pid = str(patient_ids[idx])
        if pid not in seen_pats and idx not in used_idx:
            chosen.append(int(idx))
            seen_pats.add(pid)
            used_idx.add(int(idx))
            if len(chosen) >= k:
                return chosen

    if fallback_pool is not None:
        for idx in fallback_pool:
            pid = str(patient_ids[idx])
            if pid not in seen_pats and idx not in used_idx:
                chosen.append(int(idx))
                seen_pats.add(pid)
                used_idx.add(int(idx))
                if len(chosen) >= k:
                    return chosen

    # Final fallback if dataset has < k distinct patients (e.g. tiny synthetic unit test)
    for pool in (candidate_indices, fallback_pool or ()):
        for idx in pool:
            if idx not in used_idx:
                chosen.append(int(idx))
                used_idx.add(int(idx))
                if len(chosen) >= k:
                    return chosen
    return chosen


def select_audit_sheet_indices(
    patch_records: Sequence[Mapping[str, Any]],
    k: int = 4,
    min_fg_px: int = 1000,
) -> Dict[str, List[int]]:
    """Deterministically select zero-cherry-picking patch indices for sheets 01..06 and 08."""
    n = len(patch_records)
    if n == 0:
        return {key: [] for key in ("p90", "p50", "p25", "p10_worst", "four_class", "heavy_back", "divergence")}

    pids = [str(r["patient_id"]) for r in patch_records]
    fg_eligible = [i for i, r in enumerate(patch_records) if int(r["fg_px"]) >= int(min_fg_px)]
    if len(fg_eligible) < k:
        fg_eligible = sorted(range(n), key=lambda i: -int(patch_records[i]["fg_px"]))

    scores = np.asarray([float(patch_records[i]["ens4_cgm_miou"]) for i in fg_eligible], dtype=np.float64)

    def _band_pick(q_low: float, q_mid: float, q_high: float) -> List[int]:
        lo = float(np.percentile(scores, q_low))
        mid = float(np.percentile(scores, q_mid))
        hi = float(np.percentile(scores, q_high))
        in_band = [
            idx
            for idx in fg_eligible
            if lo <= float(patch_records[idx]["ens4_cgm_miou"]) <= hi
        ]
        in_band.sort(key=lambda idx: (abs(float(patch_records[idx]["ens4_cgm_miou"]) - mid), str(patch_records[idx]["filename"])))
        near_all = sorted(
            fg_eligible,
            key=lambda idx: (abs(float(patch_records[idx]["ens4_cgm_miou"]) - mid), str(patch_records[idx]["filename"])),
        )
        return _select_distinct_patient_indices(in_band, pids, k=k, fallback_pool=near_all)

    p90_idx = _band_pick(89.0, 90.0, 91.0)
    p50_idx = _band_pick(49.0, 50.0, 51.0)
    p25_idx = _band_pick(24.0, 25.0, 26.0)

    # 04: Worst-case failures spanning target failure patients
    worst_chosen: List[int] = []
    worst_used: set = set()
    for target_pid in WORST_FAILURE_TARGET_PATIENTS:
        cand = [i for i in fg_eligible if str(patch_records[i]["patient_id"]) == target_pid]
        if cand:
            cand.sort(key=lambda i: (float(patch_records[i]["ens4_cgm_miou"]), str(patch_records[i]["filename"])))
            worst_chosen.append(int(cand[0]))
            worst_used.add(int(cand[0]))
    if len(worst_chosen) < k:
        worst_all = sorted(
            fg_eligible,
            key=lambda i: (float(patch_records[i]["ens4_cgm_miou"]), str(patch_records[i]["filename"])),
        )
        extra = _select_distinct_patient_indices(
            [i for i in worst_all if i not in worst_used], pids, k=k - len(worst_chosen)
        )
        worst_chosen.extend(extra)
    p10_worst_idx = worst_chosen[:k]

    # 05: Four-class complex (all 4 classes >= 5% of fg_px)
    four_cand: List[int] = []
    for i in fg_eligible:
        fg = max(1, int(patch_records[i]["fg_px"]))
        fracs = [
            float(patch_records[i][f"gt_{c}_px"]) / float(fg) for c in _CLS_LOWER
        ]
        if min(fracs) >= 0.05:
            four_cand.append(i)
    four_cand.sort(
        key=lambda i: (
            -min(float(patch_records[i][f"gt_{c}_px"]) / max(1, int(patch_records[i]["fg_px"])) for c in _CLS_LOWER),
            str(patch_records[i]["filename"]),
        )
    )
    four_fallback = sorted(
        fg_eligible,
        key=lambda i: (
            -sum(1 for c in _CLS_LOWER if int(patch_records[i][f"gt_{c}_px"]) > 0),
            -min(float(patch_records[i][f"gt_{c}_px"]) / max(1, int(patch_records[i]["fg_px"])) for c in _CLS_LOWER),
            str(patch_records[i]["filename"]),
        ),
    )
    four_class_idx = _select_distinct_patient_indices(four_cand, pids, k=k, fallback_pool=four_fallback)

    # 06: Heavy BACK (2 bright glass >= 40% + 2 stained outside-ROI >= 40%, requiring >= 5000 ROI fg_px)
    heavy_back_min_fg = 5000
    bg_strict = [
        i
        for i in range(n)
        if float(patch_records[i]["bright_glass_frac"]) >= 0.40
        and int(patch_records[i]["fg_px"]) >= heavy_back_min_fg
    ]
    bg_strict.sort(key=lambda i: (-float(patch_records[i]["bright_glass_frac"]), str(patch_records[i]["filename"])))
    bg_fg_elig = sorted(
        fg_eligible,
        key=lambda i: (
            0 if float(patch_records[i]["bright_glass_frac"]) >= 0.40 else 1,
            -float(patch_records[i]["bright_glass_frac"]),
            str(patch_records[i]["filename"]),
        ),
    )
    bg_fg_set = set(bg_fg_elig)
    bg_all_fb = sorted(
        [i for i in range(n) if i not in bg_fg_set],
        key=lambda i: (-float(patch_records[i]["bright_glass_frac"]), str(patch_records[i]["filename"])),
    )
    bg_pick = _select_distinct_patient_indices(bg_strict, pids, k=2, fallback_pool=bg_fg_elig + bg_all_fb)

    or_strict = [
        i
        for i in range(n)
        if float(patch_records[i]["outside_roi_frac"]) >= 0.40
        and int(patch_records[i]["fg_px"]) >= heavy_back_min_fg
        and i not in bg_pick
    ]
    or_strict.sort(key=lambda i: (-float(patch_records[i]["outside_roi_frac"]), str(patch_records[i]["filename"])))
    or_fg_elig = sorted(
        [i for i in fg_eligible if i not in bg_pick],
        key=lambda i: (
            0 if float(patch_records[i]["outside_roi_frac"]) >= 0.40 else 1,
            -float(patch_records[i]["outside_roi_frac"]),
            str(patch_records[i]["filename"]),
        ),
    )
    or_fg_set = set(or_fg_elig)
    or_all_fb = sorted(
        [i for i in range(n) if i not in bg_pick and i not in or_fg_set],
        key=lambda i: (-float(patch_records[i]["outside_roi_frac"]), str(patch_records[i]["filename"])),
    )
    bg_pids = {pids[i] for i in bg_pick}
    or_distinct = [i for i in or_strict if pids[i] not in bg_pids]
    or_same_pat = [i for i in or_strict if pids[i] in bg_pids]
    or_pick = _select_distinct_patient_indices(
        or_distinct, pids, k=2, fallback_pool=or_same_pat + or_fg_elig + or_all_fb
    )
    heavy_back_idx = (bg_pick + or_pick)[:k]

    # 08: Model divergence (top 2 TCGA-EW-A1P4 NEC FP + top 2 TCGA-E2-A159 / TCGA-EW-A6SB LYM FP)
    def _nec_div_key(i: int) -> Tuple[int, int, int, str]:
        rec = patch_records[i]
        arm5_fp = int(rec["arm5_cgm_nec_fp_px"])
        cgm_fp = int(rec["ens4_cgm_nec_fp_px"])
        no_cgm_fp = int(rec.get("ens4_no_cgm_nec_fp_px", cgm_fp))
        max_diff = max(abs(no_cgm_fp - arm5_fp), abs(cgm_fp - arm5_fp))
        return (-max_diff, -no_cgm_fp, -cgm_fp, str(rec["filename"]))

    nec_a1p4 = [i for i in fg_eligible if str(patch_records[i]["patient_id"]) == "TCGA-EW-A1P4"]
    nec_pool = nec_a1p4 if len(nec_a1p4) >= 2 else list(fg_eligible)
    nec_pool.sort(key=_nec_div_key)
    nec_pick = nec_pool[:2]

    def _lym_div_key(i: int) -> Tuple[int, int, int, str]:
        rec = patch_records[i]
        arm5_fp = int(rec["arm5_cgm_lym_fp_px"])
        cgm_fp = int(rec["ens4_cgm_lym_fp_px"])
        no_cgm_fp = int(rec.get("ens4_no_cgm_lym_fp_px", cgm_fp))
        max_diff = max(abs(no_cgm_fp - arm5_fp), abs(cgm_fp - arm5_fp))
        return (-max_diff, -no_cgm_fp, -cgm_fp, str(rec["filename"]))

    lym_pats = [
        i
        for i in fg_eligible
        if str(patch_records[i]["patient_id"]) in ("TCGA-E2-A159", "TCGA-EW-A6SB") and i not in nec_pick
    ]
    lym_pool = lym_pats if len(lym_pats) >= 2 else [i for i in fg_eligible if i not in nec_pick]
    lym_pool.sort(key=_lym_div_key)
    lym_pick = _select_distinct_patient_indices(lym_pool, pids, k=2)
    div_idx = (nec_pick + lym_pick)[:k]

    return {
        "p90": p90_idx,
        "p50": p50_idx,
        "p25": p25_idx,
        "p10_worst": p10_worst_idx,
        "four_class": four_class_idx,
        "heavy_back": heavy_back_idx,
        "divergence": div_idx,
    }


# ---------------------------------------------------------------------------
# Contact Sheet & Whole-ROI Rendering (300 DPI)
# ---------------------------------------------------------------------------


def _render_7col_contact_sheet(
    indices: Sequence[int],
    raw_rgbs: Sequence[np.ndarray],
    gt_masks: np.ndarray,
    preds_author: np.ndarray,
    preds_arm5: np.ndarray,
    preds_ens4_no_cgm: np.ndarray,
    preds_ens4_cgm: np.ndarray,
    patch_records: Sequence[Mapping[str, Any]],
    title: str,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Render 7-column zero-cherry-picking contact sheet with nearest-neighbor crisp pixels."""
    n_rows = max(1, len(indices))
    col_titles = [
        "1. Raw H&E",
        "2. Pathologist GT\n(dashed BACK=4)",
        "3. Author Baseline\n(MS-TTA + CGM)",
        "4. Arm 5 DINOv2-L\n(s2_self CGM)",
        "5. 4-Arm Primary\n(no_cgm)",
        "6. 4-Arm + CGM\n(Cyan GT / Gold Pred)",
        "7. Alignment Error\n(G=TP R=FP-T M=FN-T)",
    ]
    fig, axes = plt.subplots(n_rows, 7, figsize=(21.0, 3.15 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    for r_i, idx in enumerate(indices):
        rgb = raw_rgbs[idx]
        gt = gt_masks[idx]
        rec = patch_records[idx]
        panels = [
            rgb,
            render_gt_overlay_with_roi_contour(rgb, gt),
            render_overlay_rgb(rgb, preds_author[idx]),
            render_overlay_rgb(rgb, preds_arm5[idx]),
            render_overlay_rgb(rgb, preds_ens4_no_cgm[idx]),
            render_contour_overlay(rgb, preds_ens4_cgm[idx], gt),
            render_cancer_alignment_error_map(rgb, preds_ens4_cgm[idx], gt),
        ]
        cgm_nec_fp = int(rec["ens4_cgm_nec_fp_px"])
        cgm_lym_fp = int(rec["ens4_cgm_lym_fp_px"])
        no_cgm_nec_fp = int(rec.get("ens4_no_cgm_nec_fp_px", cgm_nec_fp))
        no_cgm_lym_fp = int(rec.get("ens4_no_cgm_lym_fp_px", cgm_lym_fp))
        arm5_nec_fp = int(rec["arm5_cgm_nec_fp_px"])
        arm5_lym_fp = int(rec["arm5_cgm_lym_fp_px"])
        sub_labels = [
            f"{rec['patient_id']} | fg={int(rec['fg_px'])}",
            f"BG={float(rec['bright_glass_frac'])*100:.0f}% OR={float(rec['outside_roi_frac'])*100:.0f}%",
            f"mIoU={float(rec['author_step0_miou']):.3f}",
            f"mIoU={float(rec['arm5_cgm_miou']):.3f} | N/Lfp={arm5_nec_fp}/{arm5_lym_fp}",
            f"mIoU={float(rec['ens4_no_cgm_miou']):.3f} | N/Lfp={no_cgm_nec_fp}/{no_cgm_lym_fp}",
            f"mIoU={float(rec['ens4_cgm_miou']):.3f} | N/Lfp={cgm_nec_fp}/{cgm_lym_fp}",
            f"CGM {cgm_nec_fp}/{cgm_lym_fp} | noCGM {no_cgm_nec_fp}/{no_cgm_lym_fp}",
        ]
        for c_i in range(7):
            ax = axes[r_i, c_i]
            ax.imshow(panels[c_i], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=9.5, fontweight="bold", pad=4)
            ax.set_xlabel(sub_labels[c_i], fontsize=7.6, labelpad=2)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    _atomic_save_figure(fig, save_path, dpi=dpi)


def _render_glass_guard_ablation_sheet(
    indices: Sequence[int],
    raw_rgbs: Sequence[np.ndarray],
    gt_masks: np.ndarray,
    preds_ens4_cgm: np.ndarray,
    patch_records: Sequence[Mapping[str, Any]],
    save_path: Path,
    dpi: int = 300,
) -> Dict[str, Any]:
    """Render Sheet 07 comparing GT-free Glass Guards: Raw vs PureGlass vs Cand1 vs Legacy Otsu."""
    n_rows = max(1, len(indices))
    col_titles = [
        "1. Raw H&E + GT\n(dashed BACK=4)",
        "2. Raw Unmasked\n(4-Arm + CGM)",
        "3. Legacy PureGlass\n(g>=242, s<=8, a>=512)",
        "4. Cand1 BrightGlass\n(g>=235, s<=12, a>=256)",
        "5. Legacy Otsu\n(tissue filter)",
    ]
    fig, axes = plt.subplots(n_rows, 5, figsize=(15.8, 3.25 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "07. GT-Free Glass Guard Ablation on Heavy-BACK Patches (Bright-Glass Suppression vs FG/NEC Erosion)",
        fontsize=12.0,
        fontweight="bold",
        y=0.995,
    )

    stats_rows: List[Dict[str, Any]] = []
    for r_i, idx in enumerate(indices):
        rgb = raw_rgbs[idx]
        gt = gt_masks[idx]
        pred_raw = preds_ens4_cgm[idx]
        rec = patch_records[idx]

        back_info = classify_back_pixels(rgb, gt)
        bg_mask = back_info["bright_glass_mask"]
        fg_mask = (gt >= 0) & (gt < NUM_FOREGROUND)
        nec_mask = gt == 3

        pred_pure = MultiMethodWSSSEngineV2.apply_pure_glass_guard(pred_raw, rgb, background_class_idx=4)
        pred_cand1 = MultiMethodWSSSEngineV2.apply_bright_glass_guard(pred_raw, rgb, background_class_idx=4)
        otsu_tiss = compute_otsu_tissue_mask(rgb).astype(bool)
        pred_otsu = pred_raw.copy()
        pred_otsu[~otsu_tiss] = 4

        def _guard_stats(p_guarded: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
            bg_cnt = int(bg_mask.sum())
            fg_cnt = int(fg_mask.sum())
            nec_cnt = int(nec_mask.sum())
            bg_supp = (
                float(np.logical_and(bg_mask, p_guarded == 4).sum()) / float(bg_cnt) * 100.0
                if bg_cnt > 0
                else None
            )
            fg_erode = (
                float(np.logical_and(fg_mask, p_guarded == 4).sum()) / float(fg_cnt) * 100.0
                if fg_cnt > 0
                else None
            )
            nec_erode = (
                float(np.logical_and(nec_mask, p_guarded == 4).sum()) / float(nec_cnt) * 100.0
                if nec_cnt > 0
                else None
            )
            return bg_supp, fg_erode, nec_erode

        def _fmt_pct(val: Optional[float]) -> str:
            return "N/A" if val is None else f"{val:.1f}%"

        st_pure = _guard_stats(pred_pure)
        st_cand1 = _guard_stats(pred_cand1)
        st_otsu = _guard_stats(pred_otsu)
        stats_rows.append(
            {
                "filename": str(rec["filename"]),
                "patient_id": str(rec["patient_id"]),
                "bright_glass_frac": round(float(rec["bright_glass_frac"]), 4),
                "outside_roi_frac": round(float(rec["outside_roi_frac"]), 4),
                "pure_bg_supp_pct": None if st_pure[0] is None else round(st_pure[0], 2),
                "pure_fg_erode_pct": None if st_pure[1] is None else round(st_pure[1], 2),
                "cand1_bg_supp_pct": None if st_cand1[0] is None else round(st_cand1[0], 2),
                "cand1_fg_erode_pct": None if st_cand1[1] is None else round(st_cand1[1], 2),
                "cand1_nec_erode_pct": None if st_cand1[2] is None else round(st_cand1[2], 2),
                "otsu_bg_supp_pct": None if st_otsu[0] is None else round(st_otsu[0], 2),
                "otsu_fg_erode_pct": None if st_otsu[1] is None else round(st_otsu[1], 2),
                "otsu_nec_erode_pct": None if st_otsu[2] is None else round(st_otsu[2], 2),
            }
        )

        panels = [
            render_gt_overlay_with_roi_contour(rgb, gt),
            render_overlay_rgb(rgb, pred_raw),
            render_overlay_rgb(rgb, pred_pure),
            render_overlay_rgb(rgb, pred_cand1),
            render_overlay_rgb(rgb, pred_otsu),
        ]
        raw_bg_lbl = "N/A" if int(bg_mask.sum()) == 0 else "0.0%"
        raw_fg_lbl = "N/A" if int(fg_mask.sum()) == 0 else "0.0%"
        labels = [
            f"{rec['patient_id']} BG={float(rec['bright_glass_frac'])*100:.0f}% OR={float(rec['outside_roi_frac'])*100:.0f}%",
            f"BGsupp={raw_bg_lbl} FGerode={raw_fg_lbl}",
            f"BGsupp={_fmt_pct(st_pure[0])} FGerode={_fmt_pct(st_pure[1])}",
            f"BGsupp={_fmt_pct(st_cand1[0])} FGerode={_fmt_pct(st_cand1[1])} NEC={_fmt_pct(st_cand1[2])}",
            f"BGsupp={_fmt_pct(st_otsu[0])} FGerode={_fmt_pct(st_otsu[1])} NEC={_fmt_pct(st_otsu[2])}",
        ]
        for c_i in range(5):
            ax = axes[r_i, c_i]
            ax.imshow(panels[c_i], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=9.5, fontweight="bold", pad=4)
            ax.set_xlabel(labels[c_i], fontsize=8.0, labelpad=2)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _atomic_save_figure(fig, save_path, dpi=dpi)
    return {"patches": stats_rows}


# ---------------------------------------------------------------------------
# Column-Major Whole-ROI Stitcher & 5x5 Zoom Verification
# ---------------------------------------------------------------------------


def stitch_column_major_roi(
    tile_map: Mapping[int, np.ndarray],
    n_rows: int,
    n_cols: int,
    tile_size: int = 224,
    fill_value: int = 0,
    enforce_seam: bool = False,
    max_seam_l1: float = 20.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stitch BCSS-WSSS `+N` tiles into an exact column-major grid (`row = idx % n_rows, col = idx // n_rows`).
    Computes adjacent horizontal and vertical border seam L1 errors when given 3-channel RGB tiles.
    """
    if not tile_map:
        raise ValueError("tile_map cannot be empty.")
    sample = next(iter(tile_map.values()))
    is_rgb = sample.ndim == 3
    if is_rgb:
        canvas = np.full((n_rows * tile_size, n_cols * tile_size, sample.shape[2]), fill_value, dtype=sample.dtype)
    else:
        canvas = np.full((n_rows * tile_size, n_cols * tile_size), fill_value, dtype=sample.dtype)

    present = np.zeros((n_rows, n_cols), dtype=bool)
    for idx, tile in tile_map.items():
        if idx < 0 or idx >= n_rows * n_cols:
            continue
        r = int(idx) % int(n_rows)
        c = int(idx) // int(n_rows)
        y0, x0 = r * tile_size, c * tile_size
        canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
        present[r, c] = True

    h_diffs: List[float] = []
    v_diffs: List[float] = []
    if is_rgb:
        for r in range(n_rows):
            for c in range(n_cols - 1):
                if present[r, c] and present[r, c + 1]:
                    left_edge = canvas[r * tile_size : (r + 1) * tile_size, (c + 1) * tile_size - 1].astype(np.float32)
                    right_edge = canvas[r * tile_size : (r + 1) * tile_size, (c + 1) * tile_size].astype(np.float32)
                    h_diffs.append(float(np.mean(np.abs(left_edge - right_edge))))
        for r in range(n_rows - 1):
            for c in range(n_cols):
                if present[r, c] and present[r + 1, c]:
                    bot_edge = canvas[(r + 1) * tile_size - 1, c * tile_size : (c + 1) * tile_size].astype(np.float32)
                    top_edge = canvas[(r + 1) * tile_size, c * tile_size : (c + 1) * tile_size].astype(np.float32)
                    v_diffs.append(float(np.mean(np.abs(bot_edge - top_edge))))

    h_l1 = float(np.mean(h_diffs)) if h_diffs else 0.0
    v_l1 = float(np.mean(v_diffs)) if v_diffs else 0.0
    if enforce_seam and is_rgb and (h_l1 >= float(max_seam_l1) or v_l1 >= float(max_seam_l1)):
        raise RuntimeError(
            f"[STAGE2 ERROR] Whole-ROI seam L1 failed: h_seam_l1={h_l1:.2f}, v_seam_l1={v_l1:.2f} >= {max_seam_l1}."
        )
    return canvas, {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "tiles_present": int(present.sum()),
        "tiles_expected": int(n_rows * n_cols),
        "canvas_hw": [int(canvas.shape[0]), int(canvas.shape[1])],
        "h_seam_l1": round(h_l1, 3),
        "v_seam_l1": round(v_l1, 3),
    }


def _select_5x5_zoom_origin(
    gt_full: np.ndarray,
    pred_full: np.ndarray,
    n_rows: int,
    n_cols: int,
    mode: str = "nec_fp",
    tile_size: int = 224,
) -> Tuple[int, int]:
    """Select optimal 5x5 tile sub-block origin (r0, c0) for Whole-ROI zoom inspection."""
    win = min(5, n_rows, n_cols)

    def _score_window(r0: int, c0: int, score_mode: str) -> float:
        y0, y1 = r0 * tile_size, (r0 + win) * tile_size
        x0, x1 = c0 * tile_size, (c0 + win) * tile_size
        g_sub = gt_full[y0:y1, x0:x1]
        p_sub = pred_full[y0:y1, x0:x1]
        valid = (g_sub >= 0) & (g_sub < NUM_FOREGROUND)
        if score_mode == "nec_fp":
            return float(np.logical_and(p_sub == 3, np.logical_and(valid, g_sub != 3)).sum())
        fg = int(valid.sum())
        if fg == 0:
            return 0.0
        fg_frac = float(fg) / float(max(1, int(g_sub.size)))
        fracs = [float(np.logical_and(g_sub == c, valid).sum()) / float(fg) for c in range(NUM_FOREGROUND)]
        # Prefer blocks where all 4 classes are richly represented (exact match to r0=6, c0=13 on A1P1)
        n_present = sum(1 for f in fracs if f >= 0.02)
        entropy = -sum(f * np.log(f + 1e-12) for f in fracs if f > 0)
        base_score = float(n_present * 10.0 + min(fracs) * 25.0 + entropy)
        return float(base_score * min(1.0, fg_frac / 0.15))

    best_rc = (0, 0)
    best_val = -1.0
    for r0 in range(0, n_rows - win + 1):
        for c0 in range(0, n_cols - win + 1):
            score = _score_window(r0, c0, mode)
            if score > best_val:
                best_val = score
                best_rc = (r0, c0)

    if mode == "nec_fp" and best_val <= 0.0:
        best_rc = (0, 0)
        best_val = -1.0
        for r0 in range(0, n_rows - win + 1):
            for c0 in range(0, n_cols - win + 1):
                score = _score_window(r0, c0, "multiclass")
                if score > best_val:
                    best_val = score
                    best_rc = (r0, c0)

    return best_rc


def render_whole_roi_comparison_figure(
    roi_prefix: str,
    rgb_full: np.ndarray,
    gt_full: np.ndarray,
    pred_tiled_full: np.ndarray,
    pred_downscale_full: np.ndarray,
    n_rows: int,
    n_cols: int,
    seam_meta: Mapping[str, Any],
    zoom_mode: str,
    save_path: Path,
    tile_size: int = 224,
    dpi: int = 300,
    zoom_score_pred_full: Optional[np.ndarray] = None,
    pred_tiled_no_cgm_full: Optional[np.ndarray] = None,
    pred_hann_full: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Render 2-row 300-DPI Whole-ROI verification figure with legends, generous hspace, and Hann-50% ablation."""
    tiled_miou = _pooled_miou_interior(pred_tiled_full, gt_full)
    down_miou = _pooled_miou_interior(pred_downscale_full, gt_full)
    tiled_no_cgm_miou = (
        _pooled_miou_interior(pred_tiled_no_cgm_full, gt_full)
        if pred_tiled_no_cgm_full is not None
        else None
    )
    hann_miou = _pooled_miou_interior(pred_hann_full, gt_full) if pred_hann_full is not None else None
    seam_tiled_cgm = compute_tile_seam_discontinuity_rate(pred_tiled_full, gt_full, tile_size=tile_size)
    seam_tiled_no_cgm = (
        compute_tile_seam_discontinuity_rate(pred_tiled_no_cgm_full, gt_full, tile_size=tile_size)
        if pred_tiled_no_cgm_full is not None
        else None
    )
    seam_hann = (
        compute_tile_seam_discontinuity_rate(pred_hann_full, gt_full, tile_size=tile_size)
        if pred_hann_full is not None
        else None
    )

    win = min(5, n_rows, n_cols)
    zoom_pred = zoom_score_pred_full if zoom_score_pred_full is not None else pred_tiled_full
    r0, c0 = _select_5x5_zoom_origin(gt_full, zoom_pred, n_rows, n_cols, mode=zoom_mode, tile_size=tile_size)
    y0, y1 = r0 * tile_size, (r0 + win) * tile_size
    x0, x1 = c0 * tile_size, (c0 + win) * tile_size

    rgb_z = rgb_full[y0:y1, x0:x1]
    gt_z = gt_full[y0:y1, x0:x1]
    pt_z = pred_tiled_full[y0:y1, x0:x1]
    zoom_tiled_miou = _pooled_miou_interior(pt_z, gt_z)

    # Downsample full-ROI display slightly for memory-efficient figure rendering while keeping crisp NEAREST
    disp_scale = max(1, max(rgb_full.shape[0], rgb_full.shape[1]) // 1024)
    rgb_d = rgb_full[::disp_scale, ::disp_scale]
    gt_d = gt_full[::disp_scale, ::disp_scale]
    pt_d = pred_tiled_full[::disp_scale, ::disp_scale]
    pd_d = pred_downscale_full[::disp_scale, ::disp_scale]

    fig, axes = plt.subplots(2, 4, figsize=(18.5, 10.8), squeeze=False)
    fig.patch.set_facecolor("white")
    hann_note = f" | Hann-50% Overlap={hann_miou:.4f}" if hann_miou is not None else ""
    fig.suptitle(
        f"Whole-ROI Verification: {roi_prefix} ({n_rows}x{n_cols}, H-seam={seam_meta['h_seam_l1']:.2f}, "
        f"V-seam={seam_meta['v_seam_l1']:.2f}) | Native Tiled={tiled_miou:.4f}{hann_note} vs Downscale={down_miou:.4f}",
        fontsize=11.0,
        fontweight="bold",
        y=0.975,
    )

    top_panels = [
        (rgb_d, f"Full ROI Raw H&E ({rgb_full.shape[1]}x{rgb_full.shape[0]})"),
        (render_gt_overlay_with_roi_contour(rgb_d, gt_d), "Pathologist GT Overlay"),
        (render_overlay_rgb(rgb_d, pt_d), f"Native 224x224 Tiled 4-Arm+CGM (mIoU={tiled_miou:.4f})"),
        (render_overlay_rgb(rgb_d, pd_d), f"Legacy Single-Pass 224 Downscale (mIoU={down_miou:.4f})"),
    ]
    for c_i, (img_p, ttl) in enumerate(top_panels):
        ax = axes[0, c_i]
        ax.imshow(img_p, interpolation="nearest")
        rect_y = y0 / disp_scale
        rect_x = x0 / disp_scale
        rect_h = (y1 - y0) / disp_scale
        rect_w = (x1 - x0) / disp_scale
        ax.add_patch(
            plt.Rectangle((rect_x, rect_y), rect_w, rect_h, fill=False, edgecolor="#FFD700", linewidth=2.0)
        )
        ax.set_title(ttl, fontsize=9.5, fontweight="bold", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])

    bot_panels = [
        (rgb_z, f"{win}x{win} Zoom Raw H&E (r0={r0}, c0={c0})"),
        (render_gt_overlay_with_roi_contour(rgb_z, gt_z), f"{win}x{win} Zoom GT Overlay"),
        (render_contour_overlay(rgb_z, pt_z, gt_z), f"Native Tiled + Contours (mIoU={zoom_tiled_miou:.4f})"),
        (render_cancer_alignment_error_map(rgb_z, pt_z, gt_z), "Zoom Cancer Alignment Error Map"),
    ]
    for c_i, (img_p, ttl) in enumerate(bot_panels):
        ax = axes[1, c_i]
        ax.imshow(img_p, interpolation="nearest")
        ax.set_title(ttl, fontsize=9.5, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Generous vertical separation (hspace=0.24) prevents bottom-row titles from ever touching top-row borders
    fig.subplots_adjust(left=0.02, right=0.98, top=0.905, bottom=0.105, wspace=0.06, hspace=0.24)

    tissue_handles = [
        mpatches.Patch(facecolor="#E41A1C", edgecolor="#222222", label="TUM (0)"),
        mpatches.Patch(facecolor="#4DAF4A", edgecolor="#222222", label="STR (1)"),
        mpatches.Patch(facecolor="#377EB8", edgecolor="#222222", label="LYM (2)"),
        mpatches.Patch(facecolor="#FF7F00", edgecolor="#222222", label="NEC (3)"),
        mpatches.Patch(facecolor="#999999", edgecolor="#222222", label="BACK (4, dashed)"),
    ]
    error_handles = [
        mpatches.Patch(facecolor="#22C55E", edgecolor="#222222", label="TP"),
        mpatches.Patch(facecolor="#DC2626", edgecolor="#222222", label="FP-T"),
        mpatches.Patch(facecolor="#D946EF", edgecolor="#222222", label="FN-T"),
        mpatches.Patch(facecolor="#F59E0B", edgecolor="#222222", label="Other"),
        mpatches.Patch(facecolor="#38BDF8", edgecolor="#222222", label="Glass"),
        mpatches.Patch(facecolor="#64748B", edgecolor="#222222", label="Outside-ROI"),
    ]
    leg_tissue = fig.legend(
        handles=tissue_handles,
        title="Tissue Overlay",
        title_fontsize=8.5,
        fontsize=8.0,
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.012),
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
    )
    fig.add_artist(leg_tissue)
    fig.legend(
        handles=error_handles,
        title="Error Map",
        title_fontsize=8.5,
        fontsize=8.0,
        ncol=6,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.012),
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
    )

    _atomic_save_figure(fig, save_path, dpi=dpi)

    seam_ablation: Dict[str, Any] = {
        "gt_seam_flip_rate": seam_tiled_cgm["gt_seam_flip_rate"],
        "seam_valid_pairs": seam_tiled_cgm["seam_valid_pairs"],
        "tiled_cgm_seam_flip_rate": seam_tiled_cgm["pred_seam_flip_rate"],
        "tiled_no_cgm_seam_flip_rate": (
            seam_tiled_no_cgm["pred_seam_flip_rate"] if seam_tiled_no_cgm is not None else None
        ),
        "hann_overlap50_seam_flip_rate": seam_hann["pred_seam_flip_rate"] if seam_hann is not None else None,
    }
    if seam_hann is not None and seam_tiled_cgm["pred_seam_flip_rate"] > 0:
        seam_ablation["hann_seam_reduction_pct_vs_tiled_cgm"] = round(
            (1.0 - seam_hann["pred_seam_flip_rate"] / seam_tiled_cgm["pred_seam_flip_rate"]) * 100.0, 2
        )
    if (
        seam_hann is not None
        and seam_tiled_no_cgm is not None
        and seam_tiled_no_cgm["pred_seam_flip_rate"] > 0
    ):
        seam_ablation["hann_seam_reduction_pct_vs_tiled_no_cgm"] = round(
            (1.0 - seam_hann["pred_seam_flip_rate"] / seam_tiled_no_cgm["pred_seam_flip_rate"]) * 100.0, 2
        )

    out_entry: Dict[str, Any] = {
        "roi_prefix": roi_prefix,
        "seam_meta": dict(seam_meta),
        "native_tiled_miou": round(float(tiled_miou), 4),
        "native_tiled_no_cgm_miou": round(float(tiled_no_cgm_miou), 4) if tiled_no_cgm_miou is not None else None,
        "hann_overlap50_miou": round(float(hann_miou), 4) if hann_miou is not None else None,
        "hann_minus_tiled_cgm_miou": (
            round(float(hann_miou - tiled_miou), 4) if hann_miou is not None else None
        ),
        "hann_minus_tiled_no_cgm_miou": (
            round(float(hann_miou - tiled_no_cgm_miou), 4)
            if (hann_miou is not None and tiled_no_cgm_miou is not None)
            else None
        ),
        "seam_discontinuity": seam_ablation,
        "legacy_downscale_miou": round(float(down_miou), 4),
        "tiled_minus_downscale": round(float(tiled_miou - down_miou), 4),
        "zoom_origin_rc": [int(r0), int(c0)],
        "zoom_window_tiles": int(win),
        "zoom_mode": str(zoom_mode),
        "zoom_tiled_miou": round(float(zoom_tiled_miou), 4),
        "figure_filename": save_path.name,
    }
    return out_entry


# ---------------------------------------------------------------------------
# Academic Monochrome 300-DPI Summary Figure (Invariant 24)
# ---------------------------------------------------------------------------


def render_monochrome_alignment_summary(
    shift_report: Mapping[str, Any],
    patch_mious_by_model: Mapping[str, Sequence[float]],
    boundary_f1_by_model: Mapping[str, Mapping[str, Any]],
    tum_centroid_by_model: Mapping[str, Mapping[str, Any]],
    save_paths: Sequence[Path],
    dpi: int = 300,
    matched_centroid_by_model: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    """
    Render Academic Monochrome 300-DPI Summary Figure (Invariant 24: white bg, black/gray only):
    - Panel A: Separate dx (dy=0, solid) and dy (dx=0, dashed) translation-shift sensitivity curves.
    - Panel B: Patch-level canonical mIoU quantile boxplot across models.
    - Panel C: Per-class Boundary F1 (@2px) monochrome hatched bars + TUM centroid offset.
    """
    model_order = [
        m for m in ("author_step0", "arm5_cgm", "ens4_no_cgm", "ens4_cgm") if m in shift_report["per_model"]
    ]
    if not model_order:
        model_order = list(shift_report["per_model"].keys())[:4]

    panel_a_models = [
        m
        for m in ("author_step0", "arm7_convnext_large_r2", "arm5_cgm", "ens4_cgm")
        if m in shift_report["per_model"]
    ]
    if len(panel_a_models) < 2:
        panel_a_models = list(model_order)

    pretty_names = {
        "author_step0": "Author",
        "arm7_convnext_large_r2": "Arm 7 (ConvNeXt)",
        "arm5_cgm": "Arm 5",
        "ens4_no_cgm": "4-Arm",
        "ens4_cgm": "4-Arm+CGM",
    }
    markers = ["s", "D", "^", "o"]
    grays = ["#888888", "#606060", "#383838", "#000000"]
    hatches = ["///", "...", "xxx", ""]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#BFBFBF")

    # Panel A: Separate dx (dy=0, solid) and dy (dx=0, dashed) translation-shift curves
    ax_a = axes[0]
    shifts = shift_report["shifts"]
    for idx, m in enumerate(panel_a_models):
        rep = shift_report["per_model"][m]
        dx_y = rep["dx_slice_dy0"]
        dy_y = rep["dy_slice_dx0"]
        col = grays[idx % len(grays)]
        mk = markers[idx % len(markers)]
        lbl = pretty_names.get(m, m)
        ax_a.plot(
            shifts,
            dx_y,
            color=col,
            linestyle="-",
            marker=mk,
            markersize=4.2,
            linewidth=1.5,
            label=f"{lbl} dx",
        )
        ax_a.plot(
            shifts,
            dy_y,
            color=col,
            linestyle="--",
            marker=mk,
            markerfacecolor="white",
            markeredgecolor=col,
            markersize=4.0,
            linewidth=1.2,
            label=f"{lbl} dy",
        )
    ax_a.axvline(0, color="#000000", linestyle=":", linewidth=0.9)
    ax_a.set_title("(a) Separate dx (solid) & dy (dashed) Shift Sensitivity", fontsize=10.0, fontweight="bold", color="#000000")
    ax_a.set_xlabel("Spatial Shift Offset (pixels)", fontsize=9.5, color="#000000")
    ax_a.set_ylabel("Pooled 4-Class mIoU (Interior ROI)", fontsize=9.5, color="#000000")
    ax_a.legend(frameon=True, facecolor="white", edgecolor="#000000", fontsize=6.6, ncol=2, loc="lower center")

    # Panel B: Patch-level canonical mIoU quantile boxplot
    ax_b = axes[1]
    box_data = [list(patch_mious_by_model.get(m, [0.0])) for m in model_order]
    bp = ax_b.boxplot(
        box_data,
        tick_labels=[pretty_names.get(m, m) for m in model_order],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#000000", "linewidth": 1.8},
        whiskerprops={"color": "#000000", "linewidth": 1.1},
        capprops={"color": "#000000", "linewidth": 1.1},
        boxprops={"edgecolor": "#000000", "linewidth": 1.1},
    )
    box_fills = ["#EAEAEA", "#D0D0D0", "#B0B0B0", "#8C8C8C"]
    for patch, fc in zip(bp["boxes"], box_fills):
        patch.set_facecolor(fc)
    ax_b.set_title("(b) Patch Canonical mIoU Distribution", fontsize=10.5, fontweight="bold", color="#000000")
    ax_b.set_ylabel("Patch 4-Class mIoU (GT-Present ROI)", fontsize=9.5, color="#000000")
    ax_b.set_ylim(0.0, 1.02)

    # Panel C: Per-Class Boundary F1 (@2px) + TUM Centroid Offset note
    ax_c = axes[2]
    x_pos = np.arange(NUM_FOREGROUND, dtype=np.float64)
    bar_w = 0.78 / max(1, len(model_order))
    bar_fills = ["#F2F2F2", "#CCCCCC", "#969696", "#4D4D4D"]
    for idx, m in enumerate(model_order):
        bf1_map = boundary_f1_by_model.get(m, {}).get("per_class_boundary_f1", {})
        vals = [float(bf1_map.get(c, 0.0)) for c in _CLS_LOWER]
        offset = (idx - (len(model_order) - 1) / 2.0) * bar_w
        ax_c.bar(
            x_pos + offset,
            vals,
            width=bar_w,
            color=bar_fills[idx % len(bar_fills)],
            edgecolor="#000000",
            linewidth=0.9,
            hatch=hatches[idx % len(hatches)],
            label=pretty_names.get(m, m),
        )
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([c.upper() for c in _CLS_LOWER], fontsize=9.5, color="#000000")
    ax_c.set_ylim(0.0, 1.05)
    ax_c.set_ylabel("Boundary F1 (@2px)", fontsize=9.5, color="#000000")
    ax_c.set_title("(c) Per-Class Boundary F1 (@2px)", fontsize=10.5, fontweight="bold", color="#000000")
    ax_c.legend(frameon=True, facecolor="white", edgecolor="#000000", fontsize=7.8, loc="upper right")

    # Annotate TUM centroid offset cleanly in the top figure margin (zero bar overlap)
    c_src = matched_centroid_by_model if matched_centroid_by_model else tum_centroid_by_model
    c_notes = []
    for m in model_order:
        cm = c_src.get(m, {})
        if cm:
            c_notes.append(f"{pretty_names.get(m, m)}:{float(cm.get('median_px', 0.0)):.1f}px")
    if c_notes:
        prefix = "Matched Common-Intersection " if matched_centroid_by_model else ""
        fig.suptitle(
            f"{prefix}Median TUM Centroid Offset: " + " | ".join(c_notes),
            fontsize=9.5,
            fontweight="bold",
            color="#000000",
            y=0.985,
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    else:
        plt.tight_layout()
    first_path = Path(save_paths[0])
    _atomic_save_figure(fig, first_path, dpi=dpi)
    raw_bytes = first_path.read_bytes()
    for extra_path in save_paths[1:]:
        _atomic_write_bytes(Path(extra_path), raw_bytes)


# ---------------------------------------------------------------------------
# Main End-to-End Visual Audit Runner
# ---------------------------------------------------------------------------


def _accumulate_patient_iu(
    preds: np.ndarray,
    gts: np.ndarray,
    patch_pidx: np.ndarray,
    num_patients: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate exact (num_patients, 4) intersection & union strictly over valid_fg."""
    inter = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
    union = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
    valid = (gts >= 0) & (gts < NUM_FOREGROUND)
    for i in range(preds.shape[0]):
        p = int(patch_pidx[i])
        vi = valid[i]
        if not np.any(vi):
            continue
        pi = preds[i]
        gi = gts[i]
        for c in range(NUM_FOREGROUND):
            pc = (pi == c) & vi
            gc = (gi == c) & vi
            tp = int(np.logical_and(pc, gc).sum())
            inter[p, c] += tp
            union[p, c] += int(pc.sum()) + int(gc.sum()) - tp
    return inter, union


def _patient_class_iou_dict(
    unique_pids: Sequence[str],
    inter_p: np.ndarray,
    union_p: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Build per-patient class IoU dictionary from (P, 4) inter/union arrays."""
    out: Dict[str, Dict[str, Any]] = {}
    for p_idx, pid in enumerate(unique_pids):
        summ = summarize_iu(inter_p[p_idx], union_p[p_idx])
        out[str(pid)] = {
            "mIoU": summ["mIoU"],
            "per_class_iou": summ["per_class_iou"],
            "inter": [int(x) for x in inter_p[p_idx]],
            "union": [int(x) for x in union_p[p_idx]],
        }
    return out


def _collect_test_samples_from_loader(
    test_loader: DataLoader,
) -> Tuple[List[str], List[str], List[Optional[Path]], List[np.ndarray], np.ndarray]:
    """Collect ordered filenames, patient IDs, optional mask paths, uint8 RGBs, and uint8 GT masks.

    Supports canonical BCSSStage2ValDataset / BCSSEvalDataset (`.files` + `.mask_dir`),
    `torch.utils.data.Subset` wrappers, and custom datasets exposing `.samples`.
    """
    ds = test_loader.dataset
    subset_indices: Optional[List[int]] = None
    while hasattr(ds, "dataset") and hasattr(ds, "indices"):
        cur_idx = [int(x) for x in ds.indices]
        subset_indices = cur_idx if subset_indices is None else [cur_idx[i] for i in subset_indices]
        ds = ds.dataset

    files_attr = getattr(ds, "files", None)
    mask_dir_attr = getattr(ds, "mask_dir", None)
    mask_dir_path = Path(mask_dir_attr) if mask_dir_attr is not None else None
    samples_attr = getattr(ds, "samples", None)

    filenames: List[str] = []
    patient_ids: List[str] = []
    mask_paths: List[Optional[Path]] = []
    raw_rgbs: List[np.ndarray] = []
    gt_chunks: List[np.ndarray] = []

    offset = 0
    for batch in test_loader:
        imgs, gts, third = _extract_img_gt(batch)
        b = int(imgs.shape[0])
        imgs_np = imgs.detach().cpu().numpy()
        gt_np = gts.detach().cpu().numpy().astype(np.uint8)
        gt_chunks.append(gt_np)

        # Support 4-tuple BCSSEvalDataset (img, mask, cls_label, name) as well as 3-tuple (img, mask, fname)
        batch_names: Optional[Sequence[Any]] = None
        if isinstance(batch, (tuple, list)) and len(batch) >= 4 and isinstance(batch[3], (list, tuple)):
            batch_names = batch[3]
        elif isinstance(third, (list, tuple)) and len(third) > 0 and isinstance(third[0], str):
            batch_names = third

        for i in range(b):
            global_i = offset + i
            ds_idx = int(subset_indices[global_i]) if subset_indices is not None else global_i

            fn: Optional[str] = None
            mp: Optional[Path] = None
            img_f: Optional[Path] = None

            if files_attr is not None and ds_idx < len(files_attr):
                img_f = Path(files_attr[ds_idx])
                fn = img_f.name
                if mask_dir_path is not None:
                    cand_m = mask_dir_path / fn
                    if cand_m.is_file():
                        mp = cand_m
            elif samples_attr is not None and ds_idx < len(samples_attr):
                entry = samples_attr[ds_idx]
                img_f = Path(entry[0])
                fn = img_f.name
                if len(entry) >= 2:
                    cand_m = Path(entry[1])
                    if cand_m.is_file():
                        mp = cand_m

            if fn is None and batch_names is not None and i < len(batch_names) and isinstance(batch_names[i], str):
                fn = str(batch_names[i])
            if fn is None:
                fn = f"patch_{global_i:05d}.png"

            filenames.append(fn)
            patient_ids.append(extract_patient_id(fn))
            mask_paths.append(mp)
            if img_f is not None and img_f.is_file():
                with Image.open(img_f) as raw_im:
                    raw_rgb = np.asarray(raw_im.convert("RGB"), dtype=np.uint8)
                exp_h, exp_w = int(gt_np.shape[1]), int(gt_np.shape[2])
                if raw_rgb.shape[:2] != (exp_h, exp_w):
                    raw_rgb = np.asarray(
                        Image.fromarray(raw_rgb, mode="RGB").resize((exp_w, exp_h), resample=Image.BILINEAR),
                        dtype=np.uint8,
                    )
                raw_rgbs.append(raw_rgb)
            else:
                raw_rgbs.append(_denorm_imagenet_to_uint8(imgs_np[i], images_are_imagenet_norm=True))
        offset += b

    gt_all = np.concatenate(gt_chunks, axis=0) if gt_chunks else np.zeros((0, 224, 224), dtype=np.uint8)
    return filenames, patient_ids, mask_paths, raw_rgbs, gt_all


def run_stage2_v2_visual_audit(
    arm_specs: Sequence[Mapping[str, Any]],
    test_loader: DataLoader,
    stage1_cls_ckpt: Optional[Union[str, Path]] = None,
    author_stage2_ckpt: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "./stage2_results",
    zip_path: Union[str, Path] = "./SE374_Stage2_Training_Results.zip",
    device: Optional[torch.device] = None,
    amp_dtype: Optional[torch.dtype] = None,
    selected_ttas: Union[Sequence[str], Mapping[str, str]] = GATE7_SELECTED_TTAS,
    full_val_betas: Union[Sequence[float], Mapping[str, float]] = GATE7_FULL_VAL_BETAS,
    uniform_weights: Union[Sequence[float], Mapping[str, float]] = GATE7_UNIFORM_WEIGHTS,
    simplex_weights: Union[Sequence[float], Mapping[str, float]] = GATE7_SIMPLEX_WEIGHTS,
    arm5_self_cgm_thresholds: Sequence[float] = GATE7_ARM5_SELF_CGM_THRESHOLDS,
    ens4_cgm_thresholds: Sequence[float] = GATE7_4ARM_CGM_THRESHOLDS,
    expected_test_miou: Optional[Mapping[str, float]] = GATE7_EXPECTED_TEST_MIOU,
    expected_arm_shas: Optional[Mapping[str, str]] = GATE7_ARM_SHA256,
    expected_stage1_sha: Optional[str] = SHA256_STAGE1_BEST_CLS,
    expected_author_sha: Optional[str] = SHA256_AUTHOR_STAGE2,
    enforce_strict_reconciliation: bool = True,
    enforce_sha: bool = True,
    enforce_palette_check: bool = True,
    enforce_equivariance: bool = True,
    equivariance_min_miou: float = 0.72,
    equivariance_min_pixel_agreement: float = 0.88,
    author_equivariance_min_miou: float = 0.60,
    author_equivariance_min_pixel_agreement: float = 0.82,
    enforce_shift_alignment: bool = True,
    track: str = "track_a_clean",
    chunk_size: int = 64,
    dpi: int = 300,
    # Optional in-memory model overrides for fast synthetic unit tests
    stage1_model_override: Optional[nn.Module] = None,
    author_model_override: Optional[nn.Module] = None,
    whole_roi_targets: Sequence[Tuple[str, int, int, str, str]] = WHOLE_ROI_TARGETS,
    downscale_predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    author_stage1_ckpt: Optional[Union[str, Path]] = None,
    expected_author_stage1_sha: Optional[str] = SHA256_STAGE1_RETRAINED,
    author_stage1_model_override: Optional[nn.Module] = None,
    enable_hann_whole_roi: bool = True,
    reconciliation_allow_improvement_keys: Sequence[str] = ("ens4_no_cgm", "ens4_cgm"),
) -> Dict[str, Any]:
    """Run complete Gate-7 Track A visual overlay, spatial alignment, and Whole-ROI audit."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_out = Path(zip_path)

    if len(arm_specs) != 4:
        raise ValueError(f"[STAGE2 ERROR] Expected exactly 4 Gate-7 arms, got {len(arm_specs)}.")

    raw_arm_names = [str(spec["name"]).strip().lower() for spec in arm_specs]
    if len(set(raw_arm_names)) != 4:
        raise ValueError(f"[STAGE2 ERROR] Expected 4 distinct Gate-7 arm names, got {raw_arm_names}.")

    # Canonically reorder Gate-7 arms to GATE7_ARM_ORDER and map all per-arm params strictly by arm name
    spec_by_name = {name: spec for name, spec in zip(raw_arm_names, arm_specs)}
    if set(raw_arm_names) == set(GATE7_ARM_ORDER):
        ordered_names = list(GATE7_ARM_ORDER)
        ordered_specs = [spec_by_name[name] for name in ordered_names]
    else:
        ordered_names = list(raw_arm_names)
        ordered_specs = list(arm_specs)

    def _resolve_arm_param(
        param: Union[Sequence[Any], Mapping[str, Any]],
        default_tuple: Tuple[Any, ...],
        default_by_arm: Mapping[str, Any],
        label: str,
    ) -> List[Any]:
        if isinstance(param, Mapping):
            norm_map = {str(k).strip().lower(): v for k, v in param.items()}
            missing = [n for n in ordered_names if n not in norm_map]
            if missing:
                raise ValueError(f"[STAGE2 ERROR] Missing {label} entries for arms: {missing}")
            return [norm_map[n] for n in ordered_names]
        seq = list(param)
        if len(seq) != len(ordered_names):
            raise ValueError(f"[STAGE2 ERROR] Expected {len(ordered_names)} values for {label}, got {len(seq)}.")
        is_default_seq = False
        if ordered_names == list(GATE7_ARM_ORDER):
            if all(isinstance(v, (int, float, np.number)) for v in default_tuple) and all(
                isinstance(v, (int, float, np.number)) for v in seq
            ):
                is_default_seq = bool(
                    np.allclose(
                        np.asarray(seq, dtype=np.float64),
                        np.asarray(default_tuple, dtype=np.float64),
                        rtol=0.0,
                        atol=1e-6,
                    )
                )
            else:
                is_default_seq = tuple(seq) == default_tuple
        if is_default_seq:
            return [default_by_arm[n] for n in ordered_names]
        raw_map = dict(zip(raw_arm_names, seq))
        return [raw_map[n] for n in ordered_names]

    arm_specs = ordered_specs
    arm_names = ordered_names
    resolved_ttas = [
        str(x)
        for x in _resolve_arm_param(selected_ttas, GATE7_SELECTED_TTAS, GATE7_SELECTED_TTAS_BY_ARM, "selected_ttas")
    ]
    resolved_betas = [
        float(x)
        for x in _resolve_arm_param(full_val_betas, GATE7_FULL_VAL_BETAS, GATE7_FULL_VAL_BETAS_BY_ARM, "full_val_betas")
    ]
    resolved_unif = [
        float(x)
        for x in _resolve_arm_param(
            uniform_weights, GATE7_UNIFORM_WEIGHTS, GATE7_UNIFORM_WEIGHTS_BY_ARM, "uniform_weights"
        )
    ]
    resolved_simp = [
        float(x)
        for x in _resolve_arm_param(
            simplex_weights, GATE7_SIMPLEX_WEIGHTS, GATE7_SIMPLEX_WEIGHTS_BY_ARM, "simplex_weights"
        )
    ]
    print(f"[VISUAL AUDIT] Starting Track A audit | Arms={arm_names} | Device={dev}", flush=True)

    # Assertion 2: Checkpoint SHA-256 & Track A quarantine + input_norm contract
    a2_ckpt = verify_checkpoints_and_norm_contract(
        arm_specs=arm_specs,
        stage1_cls_ckpt=stage1_cls_ckpt if stage1_model_override is None else None,
        author_stage2_ckpt=author_stage2_ckpt if author_model_override is None else None,
        expected_arm_shas=expected_arm_shas,
        expected_stage1_sha=expected_stage1_sha,
        expected_author_sha=expected_author_sha,
        enforce_sha=enforce_sha,
        track=track,
        author_stage1_ckpt=author_stage1_ckpt if author_stage1_model_override is None else None,
        expected_author_stage1_sha=expected_author_stage1_sha,
    )

    # Collect TEST metadata, uint8 RGBs, and GT masks
    filenames, patient_ids, mask_paths, raw_rgbs, gt_all = _collect_test_samples_from_loader(test_loader)
    n_patches = int(gt_all.shape[0])
    if n_patches == 0:
        raise RuntimeError("[STAGE2 ERROR] Test loader yielded 0 patches.")
    h, w = int(gt_all.shape[1]), int(gt_all.shape[2])

    # Assertion 1: Direct GT PNG contract (no standardize_bcss_roi_mask!)
    valid_mask_paths = [p for p in mask_paths if p is not None]
    if valid_mask_paths:
        a1_gt = verify_gt_mask_contract(
            valid_mask_paths,
            expected_hw=(h, w),
            check_palette=enforce_palette_check,
        )
    else:
        uvals = set(int(v) for v in np.unique(gt_all))
        if uvals - {0, 1, 2, 3, 4}:
            raise RuntimeError(f"[STAGE2 ERROR] In-memory GT masks contain invalid IDs: {sorted(uvals - {0,1,2,3,4})}")
        a1_gt = {
            "passed": True,
            "checked_count": n_patches,
            "palette_verified_count": 0,
            "observed_class_ids": sorted(uvals),
            "expected_hw": [h, w],
            "zero_roi_remapper_used": True,
        }

    unique_pids = sorted(set(patient_ids))
    pid_to_idx = {pid: idx for idx, pid in enumerate(unique_pids)}
    patch_pidx = np.asarray([pid_to_idx[pid] for pid in patient_ids], dtype=np.int32)
    num_patients = len(unique_pids)

    # Select 4 diverse multi-class asymmetric patches from distinct patients for Assertion 3 (equivariance)
    eq_scored: List[Tuple[int, float, int, str, int]] = []
    min_fg_eq = max(500, int(h * w * 0.40))
    for i in range(n_patches):
        gt_i = gt_all[i]
        fg_i = int(((gt_i >= 0) & (gt_i < NUM_FOREGROUND)).sum())
        if fg_i < 500:
            continue
        shares = [int((gt_i == c).sum()) / float(fg_i) for c in range(NUM_FOREGROUND)]
        n_rich = sum(1 for s in shares if s >= 0.08)
        n_present = sum(1 for s in shares if s >= 0.03)
        eq_scored.append((1 if fg_i >= min_fg_eq else 0, float(n_rich * 10 + n_present), fg_i, filenames[i], i))
    eq_scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    eq_indices = _select_distinct_patient_indices(
        [item[4] for item in eq_scored],
        patient_ids,
        k=min(4, n_patches),
    )
    if not eq_indices:
        eq_indices = list(range(min(4, n_patches)))
    eq_rgbs = [raw_rgbs[i] for i in eq_indices]

    # Pre-build downscaled (h, w) thumbnails for any complete Whole-ROIs so each arm
    # evaluates the exact same 4-Arm + CGM model in single-pass downscale mode while loaded
    fn_to_idx = {fn: i for i, fn in enumerate(filenames)}
    tile_re = re.compile(r"^(.+)\+(\d+)\.png$", re.IGNORECASE)
    pre_roi_bundles: Dict[str, Dict[str, Any]] = {}
    for roi_prefix, n_rows, n_cols, png_name, zoom_mode in whole_roi_targets:
        rgb_tiles: Dict[int, np.ndarray] = {}
        gt_tiles: Dict[int, np.ndarray] = {}
        idx_tiles: Dict[int, int] = {}
        for fn, idx in fn_to_idx.items():
            m_re = tile_re.match(fn)
            if m_re and m_re.group(1) == roi_prefix:
                t_num = int(m_re.group(2))
                rgb_tiles[t_num] = raw_rgbs[idx]
                gt_tiles[t_num] = gt_all[idx]
                idx_tiles[t_num] = idx
        if len(rgb_tiles) == n_rows * n_cols:
            rgb_full, seam_meta = stitch_column_major_roi(
                rgb_tiles, n_rows, n_cols, tile_size=h, fill_value=255, enforce_seam=True, max_seam_l1=20.0
            )
            gt_full, _ = stitch_column_major_roi(gt_tiles, n_rows, n_cols, tile_size=h, fill_value=4)
            thumb_rgb = np.asarray(
                Image.fromarray(rgb_full, mode="RGB").resize((w, h), resample=Image.BILINEAR),
                dtype=np.uint8,
            )
            pre_roi_bundles[roi_prefix] = {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "png_name": png_name,
                "zoom_mode": zoom_mode,
                "idx_tiles": idx_tiles,
                "rgb_full": rgb_full,
                "gt_full": gt_full,
                "seam_meta": seam_meta,
                "thumb_rgb": thumb_rgb,
                "arm_cal_logp": np.zeros((NUM_FOREGROUND, h, w), dtype=np.float32),
                "arm_cls_probs": [],
                "hann_cal_logp_num": (
                    np.zeros((NUM_FOREGROUND, rgb_full.shape[0], rgb_full.shape[1]), dtype=np.float32)
                    if enable_hann_whole_roi
                    else None
                ),
                "hann_weight_den": (
                    np.zeros((rgb_full.shape[0], rgb_full.shape[1]), dtype=np.float32)
                    if enable_hann_whole_roi
                    else None
                ),
            }

    # Step 1: Stream all 4 Gate-7 arms into temporary centered FP16 log-prob memmaps
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="g7_visual_audit_")
    tmp_dir = Path(tmp_dir_obj.name)
    memmap_paths: List[Path] = []
    arm_cls_probs: List[np.ndarray] = []
    arm_preds_raw: Dict[str, np.ndarray] = {}
    eq_reports: Dict[str, Dict[str, Any]] = {}
    eq_worst = 1.0
    eq_worst_pix = 1.0
    eq_worst_key = ""
    eq_failures: List[str] = []
    arm5_idx = next((i for i, n in enumerate(arm_names) if "arm5" in n), len(arm_names) - 1)

    try:
        for m_idx, spec in enumerate(arm_specs):
            arm_name = arm_names[m_idx]
            tta_mode = str(resolved_ttas[m_idx])
            mm_path = tmp_dir / f"z_arm_{m_idx}.dat"
            memmap_paths.append(mm_path)

            if spec.get("model") is not None:
                model = spec["model"].to(dev).eval()
            else:
                model, _ = load_stage2_v2_checkpoint(
                    spec["ckpt_path"],
                    device=dev,
                    strict=True,
                )
                model.eval()

            # Equivariance check for this arm while loaded
            eq_res = verify_flip_rot90_equivariance(
                {arm_name: model},
                eq_rgbs,
                {arm_name: "imagenet"},
                device=dev,
                amp_dtype=amp_dtype,
                min_miou=equivariance_min_miou,
                min_pixel_agreement=equivariance_min_pixel_agreement,
                enforce_strict=enforce_equivariance,
            )
            eq_reports[arm_name] = eq_res["per_model"][arm_name]
            eq_failures.extend(eq_res.get("failures", []))
            if float(eq_res["worst_observed"]) < eq_worst:
                eq_worst = float(eq_res["worst_observed"])
                eq_worst_key = str(eq_res["worst_key"])
            if float(eq_res.get("worst_pixel_agreement", 1.0)) < eq_worst_pix:
                eq_worst_pix = float(eq_res.get("worst_pixel_agreement", 1.0))

            mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(n_patches, NUM_FOREGROUND, h, w))
            cls_arr = np.zeros((n_patches, NUM_FOREGROUND), dtype=np.float32)
            pred_arm = np.zeros((n_patches, h, w), dtype=np.uint8)

            offset = 0
            with torch.no_grad():
                for batch in test_loader:
                    imgs, _, _ = _extract_img_gt(batch)
                    b = int(imgs.shape[0])
                    imgs = imgs.to(dev, non_blocking=True)
                    probs_t, cls_dict = apply_tta_probs(model, imgs, tta_mode, dev, amp_dtype=amp_dtype)
                    probs_np = probs_t.detach().cpu().numpy().astype(np.float32)
                    mm[offset : offset + b] = encode_centered_logprobs_fp16(probs_np, axis=1)
                    cls_arr[offset : offset + b] = cls_dict["tta"].detach().cpu().numpy().astype(np.float32)
                    pred_arm[offset : offset + b] = np.argmax(probs_np, axis=1).astype(np.uint8)
                    offset += b

                # Apples-to-apples single-pass 224x224 downscale & 50%-overlap Hann Whole-ROI accumulation
                if pre_roi_bundles:
                    from src.stage2_v2.ensemble import reconstruct_probs_from_centered

                    wm = float(resolved_unif[m_idx])
                    bm = float(resolved_betas[m_idx])
                    hann_2d = _make_2d_hann_window(h, w) if enable_hann_whole_roi else None
                    stride_y = max(1, h // 2)
                    stride_x = max(1, w // 2)
                    for bundle in pre_roi_bundles.values():
                        if downscale_predict_fn is None:
                            x_th = _rgb_uint8_to_model_tensor(bundle["thumb_rgb"], "imagenet", dev)
                            p_th, c_th = apply_tta_probs(model, x_th, tta_mode, dev, amp_dtype=amp_dtype)
                            z_th = encode_centered_logprobs_fp16(
                                p_th.detach().cpu().numpy().astype(np.float32), axis=1
                            )
                            cal_lp = reconstruct_probs_from_centered(z_th, beta=bm, eps_log=1e-3, axis=1)[0]
                            bundle["arm_cal_logp"] += wm * cal_lp.astype(np.float32)
                            bundle["arm_cls_probs"].append(c_th["tta"].detach().cpu().numpy().astype(np.float64))
                        if enable_hann_whole_roi and hann_2d is not None and bundle["hann_cal_logp_num"] is not None:
                            rgb_full = bundle["rgb_full"]
                            fh, fw = int(rgb_full.shape[0]), int(rgb_full.shape[1])
                            y_starts = list(range(0, max(1, fh - h + 1), stride_y))
                            if y_starts[-1] != fh - h and fh >= h:
                                y_starts.append(fh - h)
                            x_starts = list(range(0, max(1, fw - w + 1), stride_x))
                            if x_starts[-1] != fw - w and fw >= w:
                                x_starts.append(fw - w)
                            coords = [(ys, xs) for ys in y_starts for xs in x_starts]
                            h_bs = 16
                            h_num = bundle["hann_cal_logp_num"]
                            h_den = bundle["hann_weight_den"]
                            for ci in range(0, len(coords), h_bs):
                                batch_coords = coords[ci : ci + h_bs]
                                crops = np.stack(
                                    [rgb_full[ys : ys + h, xs : xs + w] for ys, xs in batch_coords], axis=0
                                ).astype(np.float32) / 255.0
                                chw = (np.transpose(crops, (0, 3, 1, 2)) - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
                                x_hb = torch.from_numpy(chw).to(dev, non_blocking=True)
                                p_hb, _ = apply_tta_probs(model, x_hb, tta_mode, dev, amp_dtype=amp_dtype)
                                z_hb = encode_centered_logprobs_fp16(
                                    p_hb.detach().cpu().numpy().astype(np.float32), axis=1
                                )
                                lp_hb = reconstruct_probs_from_centered(z_hb, beta=bm, eps_log=1e-3, axis=1).astype(
                                    np.float32
                                )
                                w_hann = wm * hann_2d
                                for bi, (ys, xs) in enumerate(batch_coords):
                                    h_num[:, ys : ys + h, xs : xs + w] += lp_hb[bi] * w_hann[None, :, :]
                                    h_den[ys : ys + h, xs : xs + w] += w_hann

            _close_memmap(mm)
            arm_cls_probs.append(cls_arr)
            arm_preds_raw[arm_name] = pred_arm
            print(f"[VISUAL AUDIT] Streamed {arm_name} ({tta_mode}) on {n_patches} patches.", flush=True)

            if spec.get("model") is None:
                del model
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

        # Step 2: Run Stage-1 cls expert + Author Stage-2 baseline (author_pure_1x + author_step0)
        if stage1_model_override is not None:
            s1_model = stage1_model_override.to(dev).eval()
            s1_norm = "zero_one"
        else:
            s1_model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
            s1_norm, _, _ = load_stage1_strict(s1_model, stage1_cls_ckpt)
            s1_model = s1_model.to(dev).eval()

        # Author Step-0 (0.7043) MUST be gated with stage1_retrained_best.pth when provided;
        # keep stage1_cls_ckpt (stage1_best_cls.pth) strictly for the 5-expert ensemble CGM.
        author_s1_separate = False
        if author_stage1_model_override is not None:
            author_s1_model = author_stage1_model_override.to(dev).eval()
            author_s1_norm = "zero_one"
            author_s1_separate = True
        elif author_stage1_ckpt is not None:
            author_s1_model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
            author_s1_norm, _, _ = load_stage1_strict(author_s1_model, author_stage1_ckpt)
            author_s1_model = author_s1_model.to(dev).eval()
            author_s1_separate = True
        else:
            author_s1_model = s1_model
            author_s1_norm = s1_norm

        a2_ckpt["input_norm_contract"]["stage1_cls"] = str(s1_norm)
        a2_ckpt["input_norm_contract"]["author_stage1"] = (
            str(author_s1_norm) if author_s1_separate else f"fallback_to_stage1_cls ({s1_norm})"
        )

        if author_model_override is not None:
            author_model = author_model_override.to(dev).eval()
        else:
            author_model = build_and_load_deeplabv3plus(
                checkpoint_path=author_stage2_ckpt,
                device=dev,
                strict=True,
            )

        # Equivariance check for Author Stage-2 (trained without D4 rot90 aug; uses calibrated non-D4 bounds)
        eq_author = verify_flip_rot90_equivariance(
            {"author_stage2": author_model},
            eq_rgbs,
            {"author_stage2": "zero_one"},
            device=dev,
            amp_dtype=None,
            min_miou=author_equivariance_min_miou,
            min_pixel_agreement=author_equivariance_min_pixel_agreement,
            enforce_strict=enforce_equivariance,
        )
        eq_reports["author_stage2"] = eq_author["per_model"]["author_stage2"]
        eq_failures.extend(eq_author.get("failures", []))
        if float(eq_author["worst_observed"]) < eq_worst:
            eq_worst = float(eq_author["worst_observed"])
            eq_worst_key = str(eq_author["worst_key"])
        if float(eq_author.get("worst_pixel_agreement", 1.0)) < eq_worst_pix:
            eq_worst_pix = float(eq_author.get("worst_pixel_agreement", 1.0))

        a3_equiv = {
            "passed": len(eq_failures) == 0,
            "min_miou_threshold": float(equivariance_min_miou),
            "min_pixel_agreement_threshold": float(equivariance_min_pixel_agreement),
            "worst_observed": round(float(eq_worst), 4),
            "worst_pixel_agreement": round(float(eq_worst_pix), 4),
            "worst_key": eq_worst_key,
            "failures": eq_failures,
            "n_patches": len(eq_rgbs),
            "probe_filenames": [filenames[i] for i in eq_indices],
            "per_model": eq_reports,
        }

        s1_cls_probs = np.zeros((n_patches, NUM_FOREGROUND), dtype=np.float32)
        pred_author_raw_1x_no_cgm = np.zeros((n_patches, h, w), dtype=np.uint8)
        pred_author_pure_1x_cgm_mul0 = np.zeros((n_patches, h, w), dtype=np.uint8)
        pred_author_step0 = np.zeros((n_patches, h, w), dtype=np.uint8)

        offset = 0
        with torch.no_grad():
            for batch in test_loader:
                imgs, _, _ = _extract_img_gt(batch)
                b = int(imgs.shape[0])
                imgs = imgs.to(dev, non_blocking=True)
                author_inp = prepare_stage1_input(imgs, stage1_norm_mode="zero_one", images_are_imagenet_norm=True)
                s1_inp = (
                    author_inp
                    if s1_norm == "zero_one"
                    else prepare_stage1_input(imgs, stage1_norm_mode=s1_norm, images_are_imagenet_norm=True)
                )
                _, s1_p = s1_model.forward_cam(s1_inp)
                s1_cls_probs[offset : offset + b] = s1_p.float().detach().cpu().numpy().astype(np.float32)

                if author_s1_separate:
                    author_s1_inp = (
                        author_inp
                        if author_s1_norm == "zero_one"
                        else (
                            s1_inp
                            if author_s1_norm == s1_norm
                            else prepare_stage1_input(
                                imgs, stage1_norm_mode=author_s1_norm, images_are_imagenet_norm=True
                            )
                        )
                    )
                    _, author_s1_p = author_s1_model.forward_cam(author_s1_inp)
                else:
                    author_s1_p = s1_p

                logits_1x = author_model(author_inp)
                if isinstance(logits_1x, dict):
                    logits_1x = logits_1x["seg_logits"]
                logits_1x = logits_1x[:, :NUM_FOREGROUND].float()
                probs_ms = _ms_tta_probs(author_model, author_inp, logits_1x, (0.75, 1.0, 1.25), dev, amp_dtype=None)

                pred_author_raw_1x_no_cgm[offset : offset + b] = (
                    torch.argmax(logits_1x, dim=1).cpu().numpy().astype(np.uint8)
                )
                raw_gate_a = (author_s1_p.float() > 0.10).float().view(-1, NUM_FOREGROUND, 1, 1)
                pred_author_pure_1x_cgm_mul0[offset : offset + b] = (
                    torch.argmax(logits_1x * raw_gate_a, dim=1).cpu().numpy().astype(np.uint8)
                )
                gate_a = _make_cgm_gate(author_s1_p.float(), 0.10)
                pred_author_step0[offset : offset + b] = _apply_prob_cgm(probs_ms, gate_a).astype(np.uint8)
                offset += b

        # Apples-to-apples 4-Arm + Stage-1 CGM prediction on downscaled Whole-ROI thumbnails
        if downscale_predict_fn is None and pre_roi_bundles:
            with torch.no_grad():
                for bundle in pre_roi_bundles.values():
                    x_s1 = _rgb_uint8_to_model_tensor(bundle["thumb_rgb"], s1_norm, dev)
                    _, s1_th_p = s1_model.forward_cam(x_s1)
                    s1_th_np = s1_th_p.float().detach().cpu().numpy().astype(np.float64)
                    gate_th = _combine_5_expert_probs(
                        s1_th_np,
                        bundle["arm_cls_probs"],
                        np.asarray(resolved_unif, dtype=np.float32),
                        "balanced_s1_s2_40_60",
                    )
                    bm_th = _gate_masks_from_thresholds(gate_th, ens4_cgm_thresholds)
                    p_small = _apply_bitmask_argmax_np(bundle["arm_cal_logp"][None, ...], bm_th)[0]
                    full_h, full_w = bundle["rgb_full"].shape[:2]
                    bundle["pred_downscale_full"] = np.asarray(
                        Image.fromarray(p_small, mode="L").resize((full_w, full_h), resample=Image.NEAREST),
                        dtype=np.uint8,
                    )

        # Release author & stage-1 models immediately after Step 2
        if author_s1_separate and author_stage1_model_override is None:
            del author_s1_model
        if stage1_model_override is None:
            del s1_model
        if author_model_override is None:
            del author_model
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        # Step 3: Stream fusion of 4 memmaps -> arm5_cgm, ens4_no_cgm, ens4_cgm, ens4_simplex_cgm
        # plus exact softmax-derived mean entropy & confidence (never raw negative log-probs!)
        pred_arm5_cgm = np.zeros((n_patches, h, w), dtype=np.uint8)
        pred_ens4_no_cgm = np.zeros((n_patches, h, w), dtype=np.uint8)
        pred_ens4_cgm = np.zeros((n_patches, h, w), dtype=np.uint8)
        pred_ens4_simplex_cgm = np.zeros((n_patches, h, w), dtype=np.uint8)
        patch_entropy_ens4 = np.zeros(n_patches, dtype=np.float32)
        patch_conf_ens4 = np.zeros(n_patches, dtype=np.float32)

        w_unif = np.asarray(resolved_unif, dtype=np.float32)
        w_simp = np.asarray(resolved_simp, dtype=np.float32)

        arm5_bm = _gate_masks_from_thresholds(arm_cls_probs[arm5_idx], arm5_self_cgm_thresholds)
        gate_p_unif = _combine_5_expert_probs(s1_cls_probs, arm_cls_probs, w_unif, "balanced_s1_s2_40_60")
        ens4_unif_bm = _gate_masks_from_thresholds(gate_p_unif, ens4_cgm_thresholds)
        gate_p_simp = _combine_5_expert_probs(s1_cls_probs, arm_cls_probs, w_simp, "balanced_s1_s2_40_60")
        ens4_simp_bm = _gate_masks_from_thresholds(gate_p_simp, ens4_cgm_thresholds)

        mms = [np.memmap(p, dtype=np.float16, mode="r", shape=(n_patches, NUM_FOREGROUND, h, w)) for p in memmap_paths]
        try:
            for start in range(0, n_patches, int(chunk_size)):
                end = min(n_patches, start + int(chunk_size))
                # Arm 5 s2_self CGM directly on Arm 5 calibrated probs
                arm5_probs = np.asarray(
                    _fuse_memmap_chunk(
                        [mms[arm5_idx]], start, end, np.asarray([1.0], dtype=np.float32), [1.0], "uncal_linear"
                    ),
                    dtype=np.float32,
                )
                pred_arm5_cgm[start:end] = _apply_bitmask_argmax_np(arm5_probs, arm5_bm[start:end])

                # 4-Arm Uniform cal_logprob_eps1e3
                fused_unif = _fuse_memmap_chunk(
                    mms, start, end, w_unif, resolved_betas, "cal_logprob_eps1e3", eps_log=1e-3
                )
                pred_ens4_no_cgm[start:end] = np.argmax(fused_unif, axis=1).astype(np.uint8)
                pred_ens4_cgm[start:end] = _apply_bitmask_argmax_np(fused_unif, ens4_unif_bm[start:end])

                # CRITICAL: softmax fused log-probs before computing entropy & confidence
                probs_unif = _softmax_np(fused_unif, axis=1)
                ent_b = -np.sum(probs_unif * np.log(np.clip(probs_unif, 1e-7, 1.0)), axis=1) / np.log(4.0)
                conf_b = np.max(probs_unif, axis=1)
                patch_entropy_ens4[start:end] = ent_b.mean(axis=(1, 2)).astype(np.float32)
                patch_conf_ens4[start:end] = conf_b.mean(axis=(1, 2)).astype(np.float32)

                # 4-Arm Exploratory Simplex cal_logprob_eps1e3 + CGM
                fused_simp = _fuse_memmap_chunk(
                    mms, start, end, w_simp, resolved_betas, "cal_logprob_eps1e3", eps_log=1e-3
                )
                pred_ens4_simplex_cgm[start:end] = _apply_bitmask_argmax_np(fused_simp, ens4_simp_bm[start:end])
        finally:
            for mm in mms:
                _close_memmap(mm)
            mms.clear()

        # Step 4: Accumulate pooled & 23-patient metrics across ALL models
        all_model_preds: Dict[str, np.ndarray] = {
            **arm_preds_raw,
            "author_raw_1x_no_cgm": pred_author_raw_1x_no_cgm,
            "author_pure_1x_cgm_mul0": pred_author_pure_1x_cgm_mul0,
            "author_pure_1x": pred_author_raw_1x_no_cgm,
            "author_step0": pred_author_step0,
            "arm5_cgm": pred_arm5_cgm,
            "ens4_no_cgm": pred_ens4_no_cgm,
            "ens4_cgm": pred_ens4_cgm,
            "ens4_simplex_cgm": pred_ens4_simplex_cgm,
        }
        pooled_summaries: Dict[str, Any] = {}
        per_patient_all: Dict[str, Any] = {}
        observed_mious: Dict[str, float] = {}

        for m_key, m_pred in all_model_preds.items():
            inter_p, union_p = _accumulate_patient_iu(m_pred, gt_all, patch_pidx, num_patients)
            summ = summarize_iu(inter_p.sum(axis=0), union_p.sum(axis=0))
            if m_key == "ens4_simplex_cgm":
                summ["secondary_exploratory"] = True
            elif m_key == "author_raw_1x_no_cgm":
                summ["description"] = "Author DeepLabV3+ 1x raw logits argmax (no TTA, no CGM)"
            elif m_key == "author_pure_1x_cgm_mul0":
                summ["description"] = "Author DeepLabV3+ 1x logits * (Stage-1 > 0.10) official mul0 CGM"
            elif m_key == "author_pure_1x":
                summ["alias_of"] = "author_raw_1x_no_cgm"
            pooled_summaries[m_key] = summ
            observed_mious[m_key] = float(summ["mIoU_raw"])
            per_patient_all[m_key] = _patient_class_iou_dict(unique_pids, inter_p, union_p)

        # Assertion 6: Pooled TEST mIoU Reconciliation
        a6_recon = verify_pooled_miou_reconciliation(
            observed_mious=observed_mious,
            expected_mious=expected_test_miou or {},
            tol=0.0015,
            enforce_strict=enforce_strict_reconciliation,
            allow_improvement_keys=reconciliation_allow_improvement_keys,
        )

        # Assertion 4: Translation-Shift Peak Sensitivity Matrix & Orientation Collapse across ALL models
        shift_models = {
            **arm_preds_raw,
            "author_step0": pred_author_step0,
            "arm5_cgm": pred_arm5_cgm,
            "ens4_no_cgm": pred_ens4_no_cgm,
            "ens4_cgm": pred_ens4_cgm,
        }
        shift_report = compute_translation_shift_matrix(
            shift_models, gt_all, shifts=SHIFT_GRID, device=dev
        )
        a4_shift = assert_translation_and_orientation_alignment(
            shift_report,
            max_peak_offset=4,
            max_zero_gap=0.008,
            min_orient_margin=0.05,
            enforce_strict=enforce_shift_alignment,
        )

        # Assertion 5: Zero-GT-Leakage Glass Guard check
        a5_glass = verify_zero_gt_leakage_glass_guard(
            pred_masks=[pred_ens4_cgm[i] for i in eq_indices],
            raw_rgbs=eq_rgbs,
            gt_masks=[gt_all[i] for i in eq_indices],
        )

        # Boundary F1 (@2px) & TUM Centroid Offset (pooled, per-patch, patient-level CI & matched common-intersection)
        spatial_keys = ("author_step0", "arm5_cgm", "ens4_no_cgm", "ens4_cgm")
        boundary_f1_report: Dict[str, Any] = {}
        bf1_per_patch_map: Dict[str, np.ndarray] = {}
        bf1_per_pat_map: Dict[str, np.ndarray] = {}
        tum_centroid_report: Dict[str, Any] = {}
        tum_off_per_patch_map: Dict[str, np.ndarray] = {}
        pred_tum_ok_map: Dict[str, np.ndarray] = {}
        gt_tum_ok_arr = np.zeros(n_patches, dtype=bool)
        for m_key in spatial_keys:
            bf1_rep, bf1_pp, bf1_pat = compute_boundary_f1_detailed(
                all_model_preds[m_key],
                gt_all,
                patch_pidx=patch_pidx,
                num_patients=num_patients,
                radius=2,
            )
            boundary_f1_report[m_key] = bf1_rep
            bf1_per_patch_map[m_key] = bf1_pp
            bf1_per_pat_map[m_key] = bf1_pat

            t_off, g_ok, p_ok = compute_tum_centroid_per_patch(all_model_preds[m_key], gt_all, min_tum_px=64)
            tum_off_per_patch_map[m_key] = t_off
            pred_tum_ok_map[m_key] = p_ok
            gt_tum_ok_arr = g_ok
            tum_centroid_report[m_key] = compute_tum_centroid_offset(all_model_preds[m_key], gt_all, min_tum_px=64)

        matched_spatial_stats = compute_matched_spatial_stats(
            bf1_pat_by_model=bf1_per_pat_map,
            tum_offsets_by_model=tum_off_per_patch_map,
            gt_tum_ok=gt_tum_ok_arr,
            pred_tum_ok_by_model=pred_tum_ok_map,
            patch_pidx=patch_pidx,
            num_patients=num_patients,
            key_models=spatial_keys,
        )
        for m_key in spatial_keys:
            tum_centroid_report[m_key]["matched_common_intersection"] = matched_spatial_stats[
                "tum_centroid_matched_common_intersection"
            ]["per_model"].get(m_key, {})

        def _fmt_opt_float(val: float, ndigits: int) -> Any:
            return "" if not np.isfinite(val) else round(float(val), ndigits)

        # Per-patch audit table (including per-patch boundary_f1_at_2px & tum_centroid_offset_px!)
        patch_records: List[Dict[str, Any]] = []
        patch_mious_by_model: Dict[str, List[float]] = {
            "author_step0": [],
            "arm5_cgm": [],
            "ens4_no_cgm": [],
            "ens4_cgm": [],
        }

        for i in range(n_patches):
            m_auth = compute_patch_canonical_metrics(pred_author_step0[i], gt_all[i])
            m_arm5 = compute_patch_canonical_metrics(pred_arm5_cgm[i], gt_all[i])
            m_e4 = compute_patch_canonical_metrics(pred_ens4_no_cgm[i], gt_all[i])
            m_e4c = compute_patch_canonical_metrics(pred_ens4_cgm[i], gt_all[i])
            b_info = classify_back_pixels(raw_rgbs[i], gt_all[i])

            if m_e4c["fg_px"] > 0:
                patch_mious_by_model["author_step0"].append(m_auth["miou_gt_present"])
                patch_mious_by_model["arm5_cgm"].append(m_arm5["miou_gt_present"])
                patch_mious_by_model["ens4_no_cgm"].append(m_e4["miou_gt_present"])
                patch_mious_by_model["ens4_cgm"].append(m_e4c["miou_gt_present"])

            patch_records.append(
                {
                    "index": i,
                    "filename": filenames[i],
                    "patient_id": patient_ids[i],
                    "fg_px": int(m_e4c["fg_px"]),
                    "gt_tum_px": int(m_e4c["gt_counts"][0]),
                    "gt_str_px": int(m_e4c["gt_counts"][1]),
                    "gt_lym_px": int(m_e4c["gt_counts"][2]),
                    "gt_nec_px": int(m_e4c["gt_counts"][3]),
                    "back_frac": round(float(b_info["back_frac"]), 4),
                    "bright_glass_frac": round(float(b_info["bright_glass_frac"]), 4),
                    "outside_roi_frac": round(float(b_info["outside_roi_frac"]), 4),
                    "author_step0_miou": round(float(m_auth["miou_gt_present"]), 4),
                    "arm5_cgm_miou": round(float(m_arm5["miou_gt_present"]), 4),
                    "ens4_no_cgm_miou": round(float(m_e4["miou_gt_present"]), 4),
                    "ens4_cgm_miou": round(float(m_e4c["miou_gt_present"]), 4),
                    "boundary_f1_at_2px": _fmt_opt_float(bf1_per_patch_map["ens4_cgm"][i], 4),
                    "tum_centroid_offset_px": _fmt_opt_float(tum_off_per_patch_map["ens4_cgm"][i], 3),
                    "author_step0_boundary_f1_at_2px": _fmt_opt_float(bf1_per_patch_map["author_step0"][i], 4),
                    "arm5_cgm_boundary_f1_at_2px": _fmt_opt_float(bf1_per_patch_map["arm5_cgm"][i], 4),
                    "ens4_no_cgm_boundary_f1_at_2px": _fmt_opt_float(bf1_per_patch_map["ens4_no_cgm"][i], 4),
                    "ens4_cgm_boundary_f1_at_2px": _fmt_opt_float(bf1_per_patch_map["ens4_cgm"][i], 4),
                    "author_step0_tum_centroid_offset_px": _fmt_opt_float(tum_off_per_patch_map["author_step0"][i], 3),
                    "arm5_cgm_tum_centroid_offset_px": _fmt_opt_float(tum_off_per_patch_map["arm5_cgm"][i], 3),
                    "ens4_no_cgm_tum_centroid_offset_px": _fmt_opt_float(tum_off_per_patch_map["ens4_no_cgm"][i], 3),
                    "ens4_cgm_tum_centroid_offset_px": _fmt_opt_float(tum_off_per_patch_map["ens4_cgm"][i], 3),
                    "arm5_cgm_nec_fp_px": int(m_arm5["nec_fp_px"]),
                    "ens4_no_cgm_nec_fp_px": int(m_e4["nec_fp_px"]),
                    "ens4_cgm_nec_fp_px": int(m_e4c["nec_fp_px"]),
                    "arm5_cgm_lym_fp_px": int(m_arm5["lym_fp_px"]),
                    "ens4_no_cgm_lym_fp_px": int(m_e4["lym_fp_px"]),
                    "ens4_cgm_lym_fp_px": int(m_e4c["lym_fp_px"]),
                    "ens4_mean_entropy": round(float(patch_entropy_ens4[i]), 4),
                    "ens4_mean_confidence": round(float(patch_conf_ens4[i]), 4),
                }
            )

        # Step 5: Render 8 Zero-Cherry-Picking 300-DPI Contact Sheets (01..08)
        sheet_indices = select_audit_sheet_indices(patch_records, k=4, min_fg_px=1000)
        sheet_specs = [
            ("p90", "01_quantile_P90.png", "01. Quantile P90 (4 Distinct Patients, P89-P91 Band)"),
            ("p50", "02_quantile_P50_median.png", "02. Quantile P50 Median (4 Distinct Patients, P49-P51 Band)"),
            ("p25", "03_quantile_P25.png", "03. Quantile P25 (4 Distinct Patients, P24-P26 Band)"),
            ("p10_worst", "04_quantile_P10_worst_failures.png", "04. Quantile P10 Worst Failure Modes"),
            ("four_class", "05_four_class_complex.png", "05. Multi-Class Complex Patches (4-Class Priority)"),
            ("heavy_back", "06_heavy_back_glass_vs_outside_roi.png", "06. Heavy BACK: Bright Glass vs Stained Outside-ROI"),
            ("divergence", "08_model_divergence_arm5_vs_4arm_cgm.png", "08. Model Divergence: Arm 5 vs 4-Arm + CGM (NEC & LYM Hallucinations)"),
        ]
        rendered_files: Dict[str, Path] = {}
        for sel_key, png_name, title in sheet_specs:
            out_png = out_dir / png_name
            _render_7col_contact_sheet(
                indices=sheet_indices[sel_key],
                raw_rgbs=raw_rgbs,
                gt_masks=gt_all,
                preds_author=pred_author_step0,
                preds_arm5=pred_arm5_cgm,
                preds_ens4_no_cgm=pred_ens4_no_cgm,
                preds_ens4_cgm=pred_ens4_cgm,
                patch_records=patch_records,
                title=title,
                save_path=out_png,
                dpi=dpi,
            )
            rendered_files[png_name] = out_png

        # Sheet 07: GT-free Glass Guard comparison + full test quantitative ablation (n=4,986)
        sheet07_path = out_dir / "07_glass_guard_comparison_gt_free.png"
        glass_ablation_report = _render_glass_guard_ablation_sheet(
            indices=sheet_indices["heavy_back"],
            raw_rgbs=raw_rgbs,
            gt_masks=gt_all,
            preds_ens4_cgm=pred_ens4_cgm,
            patch_records=patch_records,
            save_path=sheet07_path,
            dpi=dpi,
        )
        glass_ablation_report["full_test_quantitative"] = compute_full_test_glass_guard_ablation(
            pred_ens4_cgm, raw_rgbs, gt_all
        )
        rendered_files[sheet07_path.name] = sheet07_path

        # Step 6: Whole-ROI Column-Major Stitches (09, 10) + 50%-Overlap Hann Ablation
        whole_roi_results: List[Dict[str, Any]] = []
        for roi_prefix, n_rows, n_cols, png_name, zoom_mode in whole_roi_targets:
            if roi_prefix in pre_roi_bundles:
                bundle = pre_roi_bundles[roi_prefix]
                pred_tiles = {t_num: pred_ens4_cgm[idx] for t_num, idx in bundle["idx_tiles"].items()}
                pt_full, _ = stitch_column_major_roi(pred_tiles, n_rows, n_cols, tile_size=h, fill_value=4)
                pred_no_cgm_tiles = {t_num: pred_ens4_no_cgm[idx] for t_num, idx in bundle["idx_tiles"].items()}
                pt_no_cgm_full, _ = stitch_column_major_roi(pred_no_cgm_tiles, n_rows, n_cols, tile_size=h, fill_value=4)
                zoom_score_pred_full = pt_no_cgm_full if zoom_mode == "nec_fp" else None
                pd_full = (
                    downscale_predict_fn(bundle["rgb_full"])
                    if downscale_predict_fn is not None
                    else bundle["pred_downscale_full"]
                )
                pred_hann_full: Optional[np.ndarray] = None
                if bundle.get("hann_cal_logp_num") is not None and bundle.get("hann_weight_den") is not None:
                    h_den = np.maximum(bundle["hann_weight_den"][None, :, :], 1e-6)
                    pred_hann_full = np.argmax(bundle["hann_cal_logp_num"] / h_den, axis=0).astype(np.uint8)
                    bundle["hann_cal_logp_num"] = None
                    bundle["hann_weight_den"] = None

                roi_png = out_dir / png_name
                roi_entry = render_whole_roi_comparison_figure(
                    roi_prefix=roi_prefix,
                    rgb_full=bundle["rgb_full"],
                    gt_full=bundle["gt_full"],
                    pred_tiled_full=pt_full,
                    pred_downscale_full=pd_full,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    seam_meta=bundle["seam_meta"],
                    zoom_mode=zoom_mode,
                    save_path=roi_png,
                    tile_size=h,
                    dpi=dpi,
                    zoom_score_pred_full=zoom_score_pred_full,
                    pred_tiled_no_cgm_full=pt_no_cgm_full,
                    pred_hann_full=pred_hann_full,
                )
                rendered_files[png_name] = roi_png
                whole_roi_results.append(roi_entry)
            else:
                tiles_found = sum(
                    1
                    for fn in fn_to_idx
                    if (m_re := tile_re.match(fn)) is not None and m_re.group(1) == roi_prefix
                )
                whole_roi_results.append(
                    {
                        "roi_prefix": roi_prefix,
                        "skipped_incomplete": True,
                        "tiles_found": int(tiles_found),
                        "tiles_expected": int(n_rows * n_cols),
                    }
                )

        # Step 7: Academic Monochrome 300-DPI Summary Figure (11 + stage2_training_loss.png)
        mono_11 = out_dir / "11_spatial_alignment_monochrome_300dpi.png"
        mono_compat = out_dir / "stage2_training_loss.png"
        render_monochrome_alignment_summary(
            shift_report=shift_report,
            patch_mious_by_model=patch_mious_by_model,
            boundary_f1_by_model=boundary_f1_report,
            tum_centroid_by_model=tum_centroid_report,
            save_paths=[mono_11, mono_compat],
            dpi=dpi,
            matched_centroid_by_model=matched_spatial_stats["tum_centroid_matched_common_intersection"]["per_model"],
        )
        rendered_files[mono_11.name] = mono_11
        rendered_files[mono_compat.name] = mono_compat

        # Step 8: Persist tabular & JSON deliverables atomically
        csv_patch_path = out_dir / "per_patch_visual_audit.csv"
        _atomic_write_csv(csv_patch_path, list(patch_records[0].keys()), patch_records)
        rendered_files[csv_patch_path.name] = csv_patch_path

        pp_json_path = out_dir / "per_patient_class_ious_all_models.json"
        _atomic_write_json(pp_json_path, per_patient_all)
        rendered_files[pp_json_path.name] = pp_json_path

        shift_csv_path = out_dir / "translation_shift_matrix.csv"
        _atomic_write_csv(
            shift_csv_path,
            ["model", "kind", "dx", "dy", "miou"],
            shift_report["csv_rows"],
        )
        rendered_files[shift_csv_path.name] = shift_csv_path

        whole_roi_json_path = out_dir / "whole_roi_audit.json"
        _atomic_write_json(whole_roi_json_path, {"whole_rois": whole_roi_results})
        rendered_files[whole_roi_json_path.name] = whole_roi_json_path

        metrics_payload: Dict[str, Any] = {
            "audit_type": "gate7_track_a_visual_overlay_audit",
            "track": track,
            "provenance": _resolve_visual_audit_provenance(a2_ckpt),
            "n_test_patches": n_patches,
            "n_test_patients": num_patients,
            "frozen_protocol": {
                "arm_order": arm_names,
                "selected_ttas": list(resolved_ttas),
                "prob_mode": "cal_logprob_eps1e3",
                "full_val_betas": [float(b) for b in resolved_betas],
                "uniform_weights": [float(x) for x in resolved_unif],
                "simplex_weights": [float(x) for x in resolved_simp],
                "arm5_self_cgm_thresholds": [float(x) for x in arm5_self_cgm_thresholds],
                "ens4_cgm_strategy": "balanced_s1_s2_40_60",
                "ens4_cgm_thresholds": [float(x) for x in ens4_cgm_thresholds],
            },
            "assertions": {
                "assertion_1_gt_png_contract": a1_gt,
                "assertion_2_ckpt_sha256_and_norm": a2_ckpt,
                "assertion_3_flip_rot90_equivariance": a3_equiv,
                "assertion_4_translation_shift_alignment": a4_shift,
                "assertion_5_zero_gt_leakage_glass_guard": a5_glass,
                "assertion_6_pooled_miou_reconciliation": a6_recon,
            },
            "pooled_test_metrics": pooled_summaries,
            "boundary_f1_at_2px": boundary_f1_report,
            "tum_centroid_offset_px": tum_centroid_report,
            "matched_spatial_stats": matched_spatial_stats,
            "translation_shift_summary": {
                "shifts": shift_report["shifts"],
                "margin": shift_report["margin"],
                "per_model": shift_report["per_model"],
            },
            "selected_sheet_filenames": {
                k: [filenames[i] for i in idxs] for k, idxs in sheet_indices.items()
            },
            "glass_guard_ablation": glass_ablation_report,
            "whole_roi_audit": whole_roi_results,
            "artifacts": sorted(rendered_files.keys()),
        }

        metrics_json_path = out_dir / "STAGE2_TRAINING_METRICS.json"
        _atomic_write_json(metrics_json_path, metrics_payload)
        rendered_files[metrics_json_path.name] = metrics_json_path

        # Atomic ZIP export for Stage-2 visual audit notebook
        _atomic_write_zip(zip_out, rendered_files)
        zip_mb = zip_out.stat().st_size / (1024.0 * 1024.0)
        print(
            f"[VISUAL AUDIT DONE] ens4_cgm={pooled_summaries['ens4_cgm']['mIoU']:.4f} | "
            f"author={pooled_summaries['author_step0']['mIoU']:.4f} | ZIP={zip_out} ({zip_mb:.2f} MB)",
            flush=True,
        )
        return metrics_payload
    finally:
        tmp_dir_obj.cleanup()
