"""
Clinical Viewer Service for WSSS Histopathology.
Provides utilities for color-coded segmentation overlays, OpenSeadragon DZI manifests,
and clinical biomarker visualization for Whole Slide Images (WSI) and ROI patches.
"""

import base64
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.losses_metrics.metrics import compute_boundary_f1

# 5 Tissue Classes and Standard Color Palette
# 0: TUM (Tumor) - Red #E41A1C
# 1: STR (Stroma) - Green #4DAF4A
# 2: LYM (Lymphocyte) - Blue #377EB8
# 3: NEC (Necrosis) - Yellow/Orange #FF7F00
# 4: BACK (Background) - Gray #999999 (Transparent in overlay)

CLASS_NAMES: List[str] = ["TUM", "STR", "LYM", "NEC", "BACK"]

CLASS_COLORS_HEX: Dict[int, str] = {
    0: "#E41A1C",  # Tumor (Red)
    1: "#4DAF4A",  # Stroma (Green)
    2: "#377EB8",  # Lymphocytic Infiltrate (Blue)
    3: "#FF7F00",  # Necrosis (Yellow/Orange)
    4: "#999999",  # Background (Gray)
}

CLASS_COLORS_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (228, 26, 28),    # Red
    1: (77, 175, 74),    # Green
    2: (55, 126, 184),   # Blue
    3: (255, 127, 0),    # Orange/Yellow
    4: (153, 153, 153),  # Gray
}


def mask_to_color_image(mask: np.ndarray) -> np.ndarray:
    """Converts a 2D integer class mask (H, W) to an RGB image (H, W, 3).
    
    Args:
        mask: 2D array of class indices [0, 4].
        
    Returns:
        RGB uint8 array of shape (H, W, 3).
    """
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS_RGB.items():
        rgb[mask == cls_idx] = color
    return rgb


def mask_to_rgba_overlay(
    mask: np.ndarray,
    alpha: float = 0.5,
    bg_transparent: bool = True,
) -> np.ndarray:
    """Converts a 2D class mask to an RGBA overlay with transparency.
    
    Args:
        mask: 2D array of class indices [0, 4].
        alpha: Overlay opacity for foreground classes in range [0.0, 1.0].
        bg_transparent: If True, background (class 4) is 100% transparent.
        
    Returns:
        RGBA uint8 array of shape (H, W, 4).
    """
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    alpha_byte = int(np.clip(alpha, 0.0, 1.0) * 255)

    for cls_idx, color in CLASS_COLORS_RGB.items():
        match_pixels = (mask == cls_idx)
        rgba[match_pixels, :3] = color
        if cls_idx == 4 and bg_transparent:
            rgba[match_pixels, 3] = 0
        else:
            rgba[match_pixels, 3] = alpha_byte

    return rgba


def blend_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blends an RGB image with the color-coded segmentation mask.
    
    Args:
        image: RGB image (H, W, 3) in uint8 [0, 255].
        mask: 2D class mask (H, W).
        alpha: Opacity factor for the segmentation mask overlay.
        
    Returns:
        Blended RGB uint8 image of shape (H, W, 3).
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    color_mask = mask_to_color_image(mask)
    # Blend foreground classes; keep background tissue untouched or slightly shaded
    blended = image.copy()
    fg_mask = (mask != 4)
    if np.any(fg_mask):
        blended[fg_mask] = (
            (1.0 - alpha) * image[fg_mask].astype(np.float32) +
            alpha * color_mask[fg_mask].astype(np.float32)
        ).astype(np.uint8)

    return blended


def encode_image_to_base64(image: np.ndarray, format: str = "PNG") -> str:
    """Encodes a NumPy image array to a base64 Data URL string.
    
    Args:
        image: NumPy array (RGB or RGBA uint8).
        format: Image format ("PNG" or "JPEG").
        
    Returns:
        Base64 Data URL string (e.g. 'data:image/png;base64,...').
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    mode = "RGBA" if (image.ndim == 3 and image.shape[2] == 4) else "RGB"
    pil_img = Image.fromarray(image, mode=mode)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """Decodes a base64 string or Data URL to an RGB NumPy array.
    
    Args:
        base64_str: Base64 string, optionally with 'data:image/...;base64,' prefix.
        
    Returns:
        RGB NumPy array uint8 of shape (H, W, 3).
    """
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    raw_bytes = base64.b64decode(base64_str)
    buffer = io.BytesIO(raw_bytes)
    pil_img = Image.open(buffer).convert("RGB")
    return np.array(pil_img, dtype=np.uint8)


def generate_dzi_manifest(
    width: int,
    height: int,
    tile_size: int = 256,
    overlap: int = 1,
    fmt: str = "png",
) -> Dict[str, Any]:
    """Generates an OpenSeadragon Deep Zoom Image (DZI) tile source manifest.
    
    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        tile_size: Deep zoom tile size (default: 256).
        overlap: Tile overlap in pixels (default: 1).
        fmt: Tile format (default: 'png').
        
    Returns:
        Dictionary compatible with OpenSeadragon tileSources configuration.
    """
    return {
        "Image": {
            "xmlns": "http://schemas.microsoft.com/deepzoom/2008",
            "Format": fmt,
            "Overlap": str(overlap),
            "TileSize": str(tile_size),
            "Size": {
                "Width": str(width),
                "Height": str(height),
            },
        }
    }


def compute_class_distribution(mask: np.ndarray) -> Dict[str, float]:
    """Computes percentage distribution for all 5 tissue classes in the mask.
    
    Args:
        mask: 2D integer class mask (H, W).
        
    Returns:
        Dictionary mapping class name (TUM, STR, LYM, NEC, BACK) to percentage [0.0, 100.0].
    """
    total_pixels = mask.size
    if total_pixels == 0:
        return {name: 0.0 for name in CLASS_NAMES}

    distribution: Dict[str, float] = {}
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(mask == idx))
        pct = (count / total_pixels) * 100.0
        distribution[name] = round(pct, 2)

    return distribution


def _build_tile_feather_window(height: int, width: int, min_weight: float = 0.05) -> np.ndarray:
    """Builds a 2D separable cosine/tent feather window for seam-free WSI tile blending."""
    if height <= 2:
        wy = np.ones((height,), dtype=np.float32)
    else:
        edge_y = max(2, height // 6)
        wy = np.ones((height,), dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, edge_y, dtype=np.float32)))
        ramp = np.clip(ramp, min_weight, 1.0)
        wy[:edge_y] = ramp
        wy[-edge_y:] = np.minimum(wy[-edge_y:], ramp[::-1])

    if width <= 2:
        wx = np.ones((width,), dtype=np.float32)
    else:
        edge_x = max(2, width // 6)
        wx = np.ones((width,), dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, edge_x, dtype=np.float32)))
        ramp = np.clip(ramp, min_weight, 1.0)
        wx[:edge_x] = ramp
        wx[-edge_x:] = np.minimum(wx[-edge_x:], ramp[::-1])

    return np.outer(wy, wx).astype(np.float32)


def stitch_tiles_heatmap(
    tiles: List[Dict[str, Any]],
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stitches multi-patch tile predictions into a combined slide-level mask and color heatmap.

    Overlapping tiles are blended using a 2D separable cosine feather window over tile
    probabilities (if provided) or one-hot class votes, avoiding hard seam jumps.

    Args:
        tiles: List of dicts, each containing:
            - 'x': X-coordinate offset (pixels).
            - 'y': Y-coordinate offset (pixels).
            - 'mask': 2D integer class mask.
            - 'probs' (optional): 3D float probability map `(C, H, W)` or `(H, W, C)`.
        canvas_width: Optional canvas width. If None, computed from tile extents.
        canvas_height: Optional canvas height. If None, computed from tile extents.

    Returns:
        Tuple of:
            - composite_mask: 2D array of class indices (H, W).
            - composite_color: RGB uint8 array (H, W, 3).
    """
    if not tiles:
        # Default empty canvas
        w = canvas_width or 256
        h = canvas_height or 256
        empty_mask = np.full((h, w), 4, dtype=np.int64)
        return empty_mask, mask_to_color_image(empty_mask)

    for t in tiles:
        if int(t["x"]) < 0 or int(t["y"]) < 0:
            raise ValueError(
                f"Tile coordinates must be non-negative, got x={t['x']}, y={t['y']}."
            )

    max_x = max(int(t["x"]) + t["mask"].shape[1] for t in tiles)
    max_y = max(int(t["y"]) + t["mask"].shape[0] for t in tiles)

    w = max(canvas_width or 0, max_x)
    h = max(canvas_height or 0, max_y)

    prob_accum = np.zeros((5, h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)
    win_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for t in tiles:
        tx, ty = int(t["x"]), int(t["y"])
        t_mask = np.asarray(t["mask"], dtype=np.int64)
        th, tw = t_mask.shape[:2]
        win = win_cache.get((th, tw))
        if win is None:
            win = _build_tile_feather_window(th, tw)
            win_cache[(th, tw)] = win

        t_probs_raw = t.get("probs")
        tile_5ch = np.zeros((5, th, tw), dtype=np.float32)
        if t_probs_raw is not None:
            p_arr = np.asarray(t_probs_raw, dtype=np.float32)
            if p_arr.ndim == 3 and p_arr.shape[2] <= 10 and p_arr.shape[0] == th and p_arr.shape[1] == tw:
                p_arr = np.transpose(p_arr, (2, 0, 1))
            if p_arr.ndim == 3 and p_arr.shape[1:] == (th, tw) and p_arr.shape[0] >= 4:
                if p_arr.shape[0] >= 5:
                    tile_5ch = p_arr[:5]
                else:
                    is_glass = (t_mask == 4).astype(np.float32)
                    p_sum = np.maximum(np.sum(p_arr[:4], axis=0, keepdims=True), 1e-6)
                    fg_norm = p_arr[:4] / p_sum
                    tile_5ch[:4] = fg_norm * (1.0 - is_glass[None, ...])
                    tile_5ch[4] = is_glass
            else:
                safe_m = np.clip(t_mask, 0, 4)
                for c_idx in range(5):
                    tile_5ch[c_idx] = (safe_m == c_idx).astype(np.float32)
        else:
            safe_m = np.clip(t_mask, 0, 4)
            for c_idx in range(5):
                tile_5ch[c_idx] = (safe_m == c_idx).astype(np.float32)

        prob_accum[:, ty : ty + th, tx : tx + tw] += tile_5ch * win[None, ...]
        weight_accum[ty : ty + th, tx : tx + tw] += win

    composite_mask = np.where(weight_accum > 0.0, np.argmax(prob_accum, axis=0), 4).astype(np.int64)
    composite_color = mask_to_color_image(composite_mask)
    return composite_mask, composite_color


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Converts normalized float or uint8 image arrays safely to uint8 RGB."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.size > 0 and float(np.max(arr)) <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr[:, :, :3]


def _dimmed_grayscale_base(image: np.ndarray, he_weight: float = 0.45) -> np.ndarray:
    """Creates a muted H&E cellular backdrop so green/red accuracy overlays remain legible."""
    rgb = _ensure_uint8_rgb(image).astype(np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    return np.clip(he_weight * rgb + (1.0 - he_weight) * (0.85 * gray_rgb), 0, 255).astype(np.float32)


def compute_patch_iou_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    min_px: Optional[int] = None,
) -> Dict[str, Any]:
    """Computes unified per-patch IoU and Dice metrics across tissue classes {0,1,2,3} and BACK {4}.

    Unifies the metric contract across clinical_viewer.py and colab_pkg/visualizer.py:
    - miou_gt_present: averages strictly over GT-present tissue classes {0,1,2,3} (falling back
      to min_px=1 if all GT foreground classes have fewer than eff_min_px pixels).
    - miou_penalized_4cls: includes 0.0 penalties for tissue classes predicted >= eff_min_px
      when absent in GT (< eff_min_px), transparently surfacing false-positive hallucination.
    """
    pred = np.asarray(pred_mask, dtype=np.int64)
    gt = np.asarray(gt_mask, dtype=np.int64)
    valid = (gt >= 0) & (gt <= 4)
    n_valid = int(np.sum(valid))
    n_fg = int(np.sum((gt < 4) & valid))

    eff_min_px = (50 if n_valid >= 4096 else 1) if min_px is None else max(1, int(min_px))

    per_class_iou: Dict[str, float] = {}
    iou_gt_present_list: List[float] = []
    dice_gt_present_list: List[float] = []
    iou_penalized_list: List[float] = []
    dice_penalized_list: List[float] = []
    iou_5_list: List[float] = []

    for c_idx, c_name in enumerate(CLASS_NAMES):
        p_c = (pred == c_idx) & valid
        g_c = (gt == c_idx) & valid
        p_cnt = int(np.sum(p_c))
        g_cnt = int(np.sum(g_c))
        inter = int(np.sum(p_c & g_c))
        union = int(np.sum(p_c | g_c))
        denom_d = p_cnt + g_cnt
        if union > 0 and (g_cnt >= eff_min_px or p_cnt >= eff_min_px):
            iou_pct = float(inter / union * 100.0)
            dice_pct = float(2.0 * inter / denom_d * 100.0) if denom_d > 0 else 0.0
            per_class_iou[c_name] = round(iou_pct, 2)
            iou_5_list.append(iou_pct)
            if c_idx < 4 and g_cnt >= eff_min_px:
                iou_gt_present_list.append(iou_pct)
                dice_gt_present_list.append(dice_pct)
                iou_penalized_list.append(iou_pct)
                dice_penalized_list.append(dice_pct)
            elif c_idx < 4 and p_cnt >= eff_min_px:
                iou_penalized_list.append(0.0)
                dice_penalized_list.append(0.0)

    if not iou_gt_present_list and n_fg > 0:
        for c_idx, c_name in enumerate(CLASS_NAMES[:4]):
            p_c = (pred == c_idx) & valid
            g_c = (gt == c_idx) & valid
            p_cnt = int(np.sum(p_c))
            g_cnt = int(np.sum(g_c))
            inter = int(np.sum(p_c & g_c))
            union = int(np.sum(p_c | g_c))
            denom_d = p_cnt + g_cnt
            if g_cnt >= 1 and union > 0:
                iou_pct = float(inter / union * 100.0)
                dice_pct = float(2.0 * inter / denom_d * 100.0) if denom_d > 0 else 0.0
                per_class_iou.setdefault(c_name, round(iou_pct, 2))
                iou_gt_present_list.append(iou_pct)
                dice_gt_present_list.append(dice_pct)

    miou_gt_present = float(np.mean(iou_gt_present_list)) if iou_gt_present_list else 0.0
    miou_penalized_4cls = float(np.mean(iou_penalized_list)) if iou_penalized_list else 0.0
    mdice_4cls = float(np.mean(dice_gt_present_list)) if dice_gt_present_list else 0.0
    miou_5cls = float(np.mean(iou_5_list)) if iou_5_list else 0.0

    return {
        "miou_gt_present": round(miou_gt_present, 2),
        "miou_penalized_4cls": round(miou_penalized_4cls, 2),
        "mdice_4cls": round(mdice_4cls, 2),
        "miou_5cls": round(miou_5cls, 2),
        "per_class_iou": per_class_iou,
    }


def compute_agreement_error_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.62,
    tolerance_px: int = 2,
    error_breakdown: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generates a clinical visual correctness overlay comparing prediction vs Doctor Ground Truth.

    Color encoding (default error_breakdown=False):
    - Emerald Green (16, 185, 129): Exact foreground tissue match (pred == gt, gt < 4)
    - Neutral Crisp Glass (226, 232, 240 light tint): Exact glass background match (pred == 4, gt == 4)
    - Amber (245, 158, 11): Mismatch within thin <= tolerance_px anatomical boundary margin
    - Crimson Red (239, 68, 68): Core region classification error / glass false alarm

    When error_breakdown=True, core errors are further differentiated:
    - Crimson Red (239, 68, 68): Foreground tissue misclassification (gt < 4, pred < 4, pred != gt)
    - Magenta (217, 70, 239): False-positive tissue on clean glass (gt == 4, pred < 4)
    - Dodger Blue (59, 130, 246): Missed tissue / false negative (gt < 4, pred == 4)
    """
    rgb_u8 = _ensure_uint8_rgb(image)
    pred = np.asarray(pred_mask, dtype=np.int64)
    gt = np.asarray(gt_mask, dtype=np.int64)
    valid = (gt >= 0) & (gt <= 4)

    # Identify morphological boundary zone within tolerance_px around GT label transitions
    tol = max(0, int(tolerance_px))
    if tol > 0:
        k = max(3, 2 * tol + 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        gt_safe = np.where(valid, gt, 4).astype(np.uint8)
        gt_dil = cv2.dilate(gt_safe, kernel, iterations=1)
        gt_ero = cv2.erode(gt_safe, kernel, iterations=1)
        boundary_zone = (gt_dil != gt_ero) & valid
    else:
        boundary_zone = np.zeros_like(valid, dtype=bool)

    match_fg = (pred == gt) & (gt < 4) & valid
    match_bg = (pred == 4) & (gt == 4) & valid
    mismatch = (pred != gt) & valid
    mismatch_margin = mismatch & boundary_zone
    mismatch_core = mismatch & (~boundary_zone)

    fp_glass_mask = (gt == 4) & (pred < 4) & valid
    fn_missed_mask = (gt < 4) & (pred == 4) & valid
    misclass_mask = (gt < 4) & (pred < 4) & (pred != gt) & valid

    base = _dimmed_grayscale_base(rgb_u8, he_weight=0.40)
    out = rgb_u8.astype(np.float32).copy()

    color_green = np.array([16.0, 185.0, 129.0], dtype=np.float32)
    color_amber = np.array([245.0, 158.0, 11.0], dtype=np.float32)
    color_red = np.array([239.0, 68.0, 68.0], dtype=np.float32)
    color_magenta = np.array([217.0, 70.0, 239.0], dtype=np.float32)
    color_blue = np.array([59.0, 130.0, 246.0], dtype=np.float32)
    color_glass = np.array([226.0, 232.0, 240.0], dtype=np.float32)

    if np.any(match_bg):
        out[match_bg] = 0.85 * out[match_bg] + 0.15 * color_glass
    if np.any(match_fg):
        out[match_fg] = (1.0 - alpha * 0.80) * base[match_fg] + (alpha * 0.80) * color_green
    if np.any(mismatch_margin):
        out[mismatch_margin] = (1.0 - alpha * 0.88) * base[mismatch_margin] + (alpha * 0.88) * color_amber

    core_w = min(0.88, alpha + 0.14)
    if error_breakdown:
        core_misclass = misclass_mask & (~boundary_zone)
        core_fp_glass = fp_glass_mask & (~boundary_zone)
        core_fn_missed = fn_missed_mask & (~boundary_zone)
        if np.any(core_misclass):
            out[core_misclass] = (1.0 - core_w) * base[core_misclass] + core_w * color_red
        if np.any(core_fp_glass):
            out[core_fp_glass] = (1.0 - core_w) * base[core_fp_glass] + core_w * color_magenta
        if np.any(core_fn_missed):
            out[core_fn_missed] = (1.0 - core_w) * base[core_fn_missed] + core_w * color_blue
    elif np.any(mismatch_core):
        out[mismatch_core] = (1.0 - core_w) * base[mismatch_core] + core_w * color_red

    # Quantitative diagnostic summary
    n_valid = int(np.sum(valid))
    valid_tissue_px = int(np.sum((gt < 4) & valid))
    tp_px = int(np.sum(match_fg))
    margin_err_px = int(np.sum(mismatch_margin))
    core_err_px = int(np.sum(mismatch_core))
    fp_glass_px = int(np.sum(fp_glass_mask))
    fn_missed_px = int(np.sum(fn_missed_mask))
    misclass_px = int(np.sum(misclass_mask))

    exact_match_pct = (float(np.sum(match_fg | match_bg)) / n_valid * 100.0) if n_valid > 0 else 0.0
    fg_match_pct = (float(tp_px) / valid_tissue_px * 100.0) if valid_tissue_px > 0 else 100.0
    margin_err_pct = (float(margin_err_px) / n_valid * 100.0) if n_valid > 0 else 0.0
    core_err_pct = (float(core_err_px) / n_valid * 100.0) if n_valid > 0 else 0.0

    iou_stats = compute_patch_iou_metrics(pred, gt)
    bf1_res = compute_boundary_f1(pred, gt, num_classes=4, tolerance_px=tolerance_px)

    # Top confusion pairs (GT -> Pred) among mismatched pixels
    confusions: List[Dict[str, Any]] = []
    if n_valid > 0 and np.any(mismatch):
        g_m = gt[mismatch]
        p_m = pred[mismatch]
        for g_c in range(5):
            for p_c in range(5):
                if g_c == p_c:
                    continue
                cnt = int(np.sum((g_m == g_c) & (p_m == p_c)))
                if cnt > 0:
                    confusions.append(
                        {
                            "gt_class": CLASS_NAMES[g_c],
                            "pred_class": CLASS_NAMES[p_c],
                            "pixels": cnt,
                            "pct_of_patch": round(cnt / n_valid * 100.0, 2),
                        }
                    )
        confusions.sort(key=lambda item: -item["pixels"])

    summary = {
        "exact_match_pct": round(exact_match_pct, 2),
        "foreground_match_pct": round(fg_match_pct, 2),
        "boundary_margin_pct": round(margin_err_pct, 2),
        "core_error_pct": round(core_err_pct, 2),
        "total_error_pct": round(margin_err_pct + core_err_pct, 2),
        "miou_4cls": iou_stats["miou_gt_present"],
        "miou_gt_present": iou_stats["miou_gt_present"],
        "miou_penalized_4cls": iou_stats["miou_penalized_4cls"],
        "miou_penalized": iou_stats["miou_penalized_4cls"],
        "miou_5cls": iou_stats["miou_5cls"],
        "mdice_4cls": iou_stats["mdice_4cls"],
        "boundary_f1": round(float(bf1_res.get("BF1", 0.0)), 2),
        "per_class_iou": iou_stats["per_class_iou"],
        "top_confusions": confusions[:3],
        "valid_tissue_px": valid_tissue_px,
        "tp_px": tp_px,
        "margin_err_px": margin_err_px,
        "core_err_px": core_err_px,
        "fp_glass_px": fp_glass_px,
        "fn_missed_px": fn_missed_px,
        "misclass_px": misclass_px,
    }
    return np.clip(out, 0, 255).astype(np.uint8), summary


def compute_dual_mask_blend(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blends dimmed H&E with averaged (GT + AI Prediction) masks and outlines mismatch borders."""
    rgb = _ensure_uint8_rgb(image).astype(np.float32)
    gt = np.asarray(gt_mask, dtype=np.int64)
    pred = np.asarray(pred_mask, dtype=np.int64)
    valid = (gt >= 0) & (gt <= 4)

    gt_col = mask_to_color_image(np.where(valid, gt, 4)).astype(np.float32)
    pred_col = mask_to_color_image(np.clip(pred, 0, 4)).astype(np.float32)
    dual_col = 0.50 * gt_col + 0.50 * pred_col

    out = rgb.copy()
    fg_or_err = ((gt < 4) | (pred < 4)) & valid
    if np.any(fg_or_err):
        out[fg_or_err] = (1.0 - alpha) * rgb[fg_or_err] + alpha * dual_col[fg_or_err]

    mismatch = ((pred != gt) & valid).astype(np.uint8)
    if np.any(mismatch):
        contours, _ = cv2.findContours(mismatch, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255.0, 56.0, 56.0), 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_blurred_dual_verification_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    blur_sigma: float = 4.0,
    dim_factor: float = 0.55,
    gt_alpha: float = 0.45,
    pred_alpha: float = 0.55,
    error_alpha: float = 0.65,
    tolerance_px: int = 2,
) -> np.ndarray:
    """Blurs GT and AI Prediction masks over dimmed H&E with Green/Red/Amber/Magenta alignment cues.

    Uses premultiplied-alpha Gaussian blur so soft region halos never darken at tissue/glass
    edges, then superimposes crisp TP (Emerald Green), core tissue misclass (Crimson Red),
    glass false-positive (Magenta), and boundary margin / missed-tissue FN (Amber) highlights
    with 1px error contours.
    """
    rgb_u8 = _ensure_uint8_rgb(image)
    rgb_f = rgb_u8.astype(np.float32)
    gt = np.asarray(gt_mask, dtype=np.int64)
    pred = np.asarray(pred_mask, dtype=np.int64)
    valid = (gt >= 0) & (gt <= 4)

    dim = float(np.clip(dim_factor, 0.0, 1.0))
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_dark = np.stack([gray, gray, gray], axis=-1) * (1.0 - 0.30 * dim)
    out = (1.0 - dim) * rgb_f + dim * gray_dark

    gt_safe = np.where(valid, gt, 4)
    pred_safe = np.clip(pred, 0, 4)
    gt_fg = ((gt_safe >= 0) & (gt_safe < 4)).astype(np.float32)
    pred_fg = ((pred_safe >= 0) & (pred_safe < 4)).astype(np.float32)

    gt_col = mask_to_color_image(gt_safe).astype(np.float32) * gt_fg[..., None]
    pred_col = mask_to_color_image(pred_safe).astype(np.float32) * pred_fg[..., None]

    sigma = float(max(0.0, blur_sigma))
    if sigma > 0.0:
        gt_col_b = cv2.GaussianBlur(gt_col, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        gt_w_b = cv2.GaussianBlur(gt_fg, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        gt_soft_rgb = np.where(gt_w_b[..., None] > 1e-6, gt_col_b / np.maximum(gt_w_b[..., None], 1e-6), 0.0)
        gt_soft_w = np.clip(gt_w_b, 0.0, 1.0)

        pred_col_b = cv2.GaussianBlur(pred_col, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        pred_w_b = cv2.GaussianBlur(pred_fg, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        pred_soft_rgb = np.where(pred_w_b[..., None] > 1e-6, pred_col_b / np.maximum(pred_w_b[..., None], 1e-6), 0.0)
        pred_soft_w = np.clip(pred_w_b, 0.0, 1.0)
    else:
        gt_soft_rgb = gt_col
        gt_soft_w = gt_fg
        pred_soft_rgb = pred_col
        pred_soft_w = pred_fg

    g_a = float(np.clip(gt_alpha, 0.0, 1.0)) * gt_soft_w[..., None]
    out = (1.0 - g_a) * out + g_a * gt_soft_rgb

    p_a = float(np.clip(pred_alpha, 0.0, 1.0)) * pred_soft_w[..., None]
    out = (1.0 - p_a) * out + p_a * pred_soft_rgb

    e_alpha = float(np.clip(error_alpha, 0.0, 1.0))
    if e_alpha > 0.0:
        tol = max(0, int(tolerance_px))
        if tol > 0:
            k = max(3, 2 * tol + 1)
            kernel = np.ones((k, k), dtype=np.uint8)
            gt_u8 = gt_safe.astype(np.uint8)
            boundary_zone = (cv2.dilate(gt_u8, kernel) != cv2.erode(gt_u8, kernel)) & valid
        else:
            boundary_zone = np.zeros_like(valid, dtype=bool)

        match_fg = (pred == gt) & (gt < 4) & valid
        mismatch = (pred != gt) & valid
        margin_or_fn = (mismatch & boundary_zone) | ((gt < 4) & (pred == 4) & valid)
        fp_glass = (gt == 4) & (pred < 4) & valid & (~margin_or_fn)
        core_misclass = (gt < 4) & (pred < 4) & (pred != gt) & valid & (~margin_or_fn)

        c_green = np.array([16.0, 185.0, 129.0], dtype=np.float32)
        c_red = np.array([239.0, 68.0, 68.0], dtype=np.float32)
        c_magenta = np.array([217.0, 70.0, 239.0], dtype=np.float32)
        c_amber = np.array([245.0, 158.0, 11.0], dtype=np.float32)

        w_tp = 0.55 * e_alpha
        w_amb = 0.75 * e_alpha
        w_mag = 0.80 * e_alpha
        w_red = 0.80 * e_alpha
        if np.any(match_fg):
            out[match_fg] = (1.0 - w_tp) * out[match_fg] + w_tp * c_green
        if np.any(margin_or_fn):
            out[margin_or_fn] = (1.0 - w_amb) * out[margin_or_fn] + w_amb * c_amber
        if np.any(fp_glass):
            out[fp_glass] = (1.0 - w_mag) * out[fp_glass] + w_mag * c_magenta
            if e_alpha >= 0.2:
                contours_fp, _ = cv2.findContours(fp_glass.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours_fp, -1, (192.0, 38.0, 211.0), 1)
        if np.any(core_misclass):
            out[core_misclass] = (1.0 - w_red) * out[core_misclass] + w_red * c_red
            if e_alpha >= 0.2:
                contours_mis, _ = cv2.findContours(
                    core_misclass.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(out, contours_mis, -1, (220.0, 38.0, 38.0), 1)

    return np.clip(out, 0, 255).astype(np.uint8)


def compute_contour_comparison_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draws crisp boundary contours (Solid Cyan = Doctor GT, Dashed Gold = AI Prediction) with dark halo."""
    out_f = _ensure_uint8_rgb(image).astype(np.float32)
    pred = np.asarray(pred_mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)

    gt_edges = np.zeros(pred.shape[:2], dtype=bool)
    if gt_mask is not None:
        gt = np.asarray(gt_mask, dtype=np.int64)
        valid = (gt >= 0) & (gt <= 4)
        gt_u8 = np.where(valid, gt, 4).astype(np.uint8)
        gt_edges = (cv2.dilate(gt_u8, kernel) != cv2.erode(gt_u8, kernel)) & valid

    pred_edges = cv2.dilate(pred, kernel) != cv2.erode(pred, kernel)
    yy, xx = np.indices(pred.shape[:2])
    dash_pattern = ((yy + xx) % 6) < 4
    pred_edges_dashed = pred_edges & dash_pattern

    all_edges = gt_edges | pred_edges_dashed
    if np.any(all_edges):
        halo_mask = (cv2.dilate(all_edges.astype(np.uint8), kernel, iterations=1) > 0) & (~all_edges)
        if np.any(halo_mask):
            slate_halo = np.array([15.0, 23.0, 42.0], dtype=np.float32)
            out_f[halo_mask] = 0.15 * out_f[halo_mask] + 0.85 * slate_halo

    out = np.clip(out_f, 0, 255).astype(np.uint8)
    if np.any(gt_edges):
        out[gt_edges] = np.array([6, 182, 212], dtype=np.uint8)  # Solid Cyan (#06B6D4)
    if np.any(pred_edges_dashed):
        out[pred_edges_dashed] = np.array([250, 204, 21], dtype=np.uint8)  # Dashed Gold (#FACC15)
    return out


def compute_per_class_spotlight(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    target_class: int = 0,
    alpha: float = 0.65,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Highlights TP (Green), FP (Red), FN (Amber) for a single target tissue class over grayscale H&E."""
    cls_idx = int(np.clip(target_class, 0, 4))
    rgb = _ensure_uint8_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32) * 0.85
    out = base.copy()

    pred_c = np.asarray(pred_mask, dtype=np.int64) == cls_idx
    if gt_mask is None:
        col = np.array(CLASS_COLORS_RGB[cls_idx], dtype=np.float32)
        if np.any(pred_c):
            out[pred_c] = (1.0 - alpha) * base[pred_c] + alpha * col
        stats_dict = {
            "target_class": cls_idx,
            "class_name": CLASS_NAMES[cls_idx],
            "pred_pct": round(float(np.mean(pred_c) * 100.0), 2),
            "pred_px": int(np.sum(pred_c)),
        }
        return np.clip(out, 0, 255).astype(np.uint8), stats_dict

    gt = np.asarray(gt_mask, dtype=np.int64)
    valid = (gt >= 0) & (gt <= 4)
    gt_c = (gt == cls_idx) & valid
    pred_valid = pred_c & valid

    tp = pred_valid & gt_c
    fp = pred_valid & (~gt_c)
    fn = (~pred_valid) & gt_c

    if np.any(tp):
        out[tp] = (1.0 - alpha) * base[tp] + alpha * np.array([16.0, 185.0, 129.0], dtype=np.float32)
    if np.any(fp):
        out[fp] = (1.0 - alpha) * base[fp] + alpha * np.array([239.0, 68.0, 68.0], dtype=np.float32)
    if np.any(fn):
        out[fn] = (1.0 - alpha) * base[fn] + alpha * np.array([245.0, 158.0, 11.0], dtype=np.float32)

    n_tp = int(np.sum(tp))
    n_fp = int(np.sum(fp))
    n_fn = int(np.sum(fn))
    n_valid = max(1, int(np.sum(valid)))
    union = n_tp + n_fp + n_fn

    stats_dict = {
        "target_class": cls_idx,
        "class_name": CLASS_NAMES[cls_idx],
        "iou_pct": round(n_tp / union * 100.0, 2) if union > 0 else 100.0,
        "precision_pct": round(n_tp / max(1, n_tp + n_fp) * 100.0, 2) if (n_tp + n_fp) > 0 else 0.0,
        "recall_pct": round(n_tp / max(1, n_tp + n_fn) * 100.0, 2) if (n_tp + n_fn) > 0 else 0.0,
        "tp_pct": round(n_tp / n_valid * 100.0, 2),
        "fp_pct": round(n_fp / n_valid * 100.0, 2),
        "fn_pct": round(n_fn / n_valid * 100.0, 2),
        "tp_px": n_tp,
        "fp_px": n_fp,
        "fn_px": n_fn,
        "valid_px": int(np.sum(valid)),
    }
    return np.clip(out, 0, 255).astype(np.uint8), stats_dict


def compute_entropy_heatmap_overlay(
    image: np.ndarray,
    probs: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "inferno",
) -> Tuple[np.ndarray, float]:
    """Renders pixel-wise normalized Shannon entropy uncertainty heatmap over dimmed grayscale H&E."""
    rgb = _ensure_uint8_rgb(image)
    base = _dimmed_grayscale_base(rgb, he_weight=0.25)
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim == 3 and p.shape[0] <= 10 and p.shape[1] == rgb.shape[0]:
        # (C, H, W)
        pass
    elif p.ndim == 3 and p.shape[2] <= 10:
        p = np.transpose(p, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected probability map shape {p.shape} for image {rgb.shape}")

    num_cls = max(2, int(p.shape[0]))
    p_safe = np.clip(p, 1e-7, 1.0)
    p_safe = p_safe / np.maximum(np.sum(p_safe, axis=0, keepdims=True), 1e-7)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=0) / math.log(float(num_cls))
    entropy = np.clip(entropy, 0.0, 1.0)

    ent_u8 = (entropy * 255.0).astype(np.uint8)
    cmap_code = cv2.COLORMAP_MAGMA if str(colormap).lower().strip() == "magma" else cv2.COLORMAP_INFERNO
    heat_bgr = cv2.applyColorMap(ent_u8, cmap_code)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    blend_w = np.clip(alpha * (0.20 + 0.80 * entropy), 0.0, 0.88)[..., None]
    out = (1.0 - blend_w) * base + blend_w * heat_rgb
    return np.clip(out, 0, 255).astype(np.uint8), round(float(np.mean(entropy)), 4)


def compute_cam_overlay(
    image: np.ndarray,
    cam_map: np.ndarray,
    target_class: Optional[int] = None,
    alpha: float = 0.60,
    dim_factor: float = 0.55,
    colormap: str = "inferno",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Renders Stage-1 Class Activation Map (CAM) heatmap over dimmed grayscale H&E."""
    rgb = _ensure_uint8_rgb(image)
    h, w = rgb.shape[:2]
    cam = np.asarray(cam_map, dtype=np.float32)

    resolved_target: Optional[int] = None
    if cam.ndim == 3:
        if cam.shape[0] <= 10 and cam.shape[1] > 10:
            pass
        elif cam.shape[2] <= 10:
            cam = np.transpose(cam, (2, 0, 1))
        if target_class is not None and 0 <= int(target_class) < cam.shape[0]:
            resolved_target = int(target_class)
            cam_2d = cam[resolved_target]
        else:
            cam_2d = np.max(cam, axis=0)
    elif cam.ndim == 2:
        cam_2d = cam
    else:
        raise ValueError(f"Unexpected CAM shape {cam.shape} for image {rgb.shape}")

    if cam_2d.shape[:2] != (h, w):
        cam_2d = cv2.resize(cam_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    c_min = float(np.min(cam_2d))
    c_max = float(np.max(cam_2d))
    if c_max - c_min > 1e-6:
        cam_norm = np.clip((cam_2d - c_min) / (c_max - c_min), 0.0, 1.0)
    else:
        cam_norm = np.zeros((h, w), dtype=np.float32)

    he_w = max(0.15, 1.0 - 0.75 * float(np.clip(dim_factor, 0.0, 1.0)))
    base = _dimmed_grayscale_base(rgb, he_weight=he_w)

    cam_u8 = (cam_norm * 255.0).astype(np.uint8)
    cmap_code = cv2.COLORMAP_MAGMA if str(colormap).lower().strip() == "magma" else cv2.COLORMAP_INFERNO
    heat_bgr = cv2.applyColorMap(cam_u8, cmap_code)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    blend_w = np.clip(float(alpha) * (0.15 + 0.85 * cam_norm), 0.0, 0.90)[..., None]
    out = (1.0 - blend_w) * base + blend_w * heat_rgb

    stats = {
        "target_class": resolved_target,
        "peak_activation": round(c_max, 4),
        "mean_activation": round(float(np.mean(cam_norm)), 4),
        "active_area_pct": round(float(np.mean(cam_norm >= 0.50) * 100.0), 2),
    }
    return np.clip(out, 0, 255).astype(np.uint8), stats


def render_single_patch_diagnostic_sheet(
    image: np.ndarray,
    pred_mask: np.ndarray,
    probs: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    cam_map: Optional[np.ndarray] = None,
    method_name: str = "m6",
    sample_id: str = "patch",
    biomarkers: Optional[Dict[str, float]] = None,
    gt_biomarkers: Optional[Dict[str, float]] = None,
    blur_sigma: float = 4.0,
    dim_factor: float = 0.55,
    gt_alpha: float = 0.45,
    pred_alpha: float = 0.55,
    error_alpha: float = 0.65,
    tolerance_px: int = 2,
    dpi: int = 300,
) -> bytes:
    """Renders a 300-DPI 2x4 Diagnostic Comparison Sheet PNG using thread-safe OO Matplotlib."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rgb = _ensure_uint8_rgb(image)
    pred = np.asarray(pred_mask, dtype=np.int64)
    pred_rgb = mask_to_color_image(pred)
    contour_rgb = compute_contour_comparison_overlay(rgb, pred, gt_mask)

    if probs is not None:
        entropy_rgb, mean_ent = compute_entropy_heatmap_overlay(rgb, probs, alpha=0.55)
    else:
        entropy_rgb = _dimmed_grayscale_base(rgb, he_weight=0.25).astype(np.uint8)
        mean_ent = 0.0

    cam_rgb = None
    if cam_map is not None:
        cam_rgb, _ = compute_cam_overlay(rgb, cam_map, alpha=0.60, dim_factor=dim_factor)

    agree_summary: Optional[Dict[str, Any]] = None
    if gt_mask is not None:
        gt_arr = np.asarray(gt_mask, dtype=np.int64)
        gt_rgb = mask_to_color_image(np.where((gt_arr >= 0) & (gt_arr <= 4), gt_arr, 4))
        blurred_rgb = compute_blurred_dual_verification_overlay(
            rgb,
            pred,
            gt_arr,
            blur_sigma=blur_sigma,
            dim_factor=dim_factor,
            gt_alpha=gt_alpha,
            pred_alpha=pred_alpha,
            error_alpha=error_alpha,
            tolerance_px=tolerance_px,
        )
        agree_rgb, agree_summary = compute_agreement_error_overlay(
            rgb,
            pred,
            gt_arr,
            tolerance_px=tolerance_px,
            error_breakdown=True,
        )
    else:
        gt_rgb = None
        blurred_rgb = blend_overlay(rgb, pred, alpha=0.50)
        agree_rgb = pred_rgb

    fig = Figure(figsize=(16.0, 10.2), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#FFFFFF")
    axes = fig.subplots(2, 4)

    def _show(ax: Any, arr: np.ndarray, title: str) -> None:
        ax.imshow(arr)
        ax.set_title(title, fontsize=10, fontweight="bold", color="#0F172A", pad=6)
        ax.axis("off")

    _show(axes[0, 0], rgb, "1. Raw H&E Patch")
    if gt_rgb is not None:
        _show(axes[0, 1], gt_rgb, "2. Pathologist Ground Truth")
    else:
        axes[0, 1].set_facecolor("#F8FAFC")
        axes[0, 1].text(0.5, 0.5, "No Reference GT\nProvided", ha="center", va="center", fontsize=11, color="#475569")
        axes[0, 1].set_title("2. Pathologist Ground Truth", fontsize=10, fontweight="bold", color="#0F172A", pad=6)
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])

    pred_title = f"3. AI Prediction ({method_name.upper()})"
    if agree_summary is not None:
        pred_title += f" | mIoU: {agree_summary['miou_4cls']:.1f}%"
    _show(axes[0, 2], pred_rgb, pred_title)
    _show(
        axes[0, 3],
        blurred_rgb,
        f"4. Blurred Dual QA (sigma={blur_sigma:.1f})" if gt_rgb is not None else "4. Blended Prediction",
    )

    _show(
        axes[1, 0],
        agree_rgb,
        "5. Match vs Error Breakdown" if gt_rgb is not None else "5. Class Segmentation Mask",
    )
    _show(axes[1, 1], contour_rgb, "6. Boundary Contours (GT Cyan / AI Gold)")
    _show(axes[1, 2], entropy_rgb, f"7. Shannon Entropy (mean H={mean_ent:.3f})")

    if cam_rgb is not None:
        _show(axes[1, 3], cam_rgb, "8. Stage-1 CAM Activation")
    else:
        axes[1, 3].set_facecolor("#F8FAFC")
        axes[1, 3].set_title("8. Quantitative Audit Summary", fontsize=10, fontweight="bold", color="#0F172A", pad=6)
        axes[1, 3].set_xticks([])
        axes[1, 3].set_yticks([])

    vtp = float((biomarkers or {}).get("viable_tumor_percentage", 0.0))
    tsr = float((biomarkers or {}).get("tumor_stroma_ratio", 0.0))
    footer_lines = [
        f"Sample: {sample_id}  |  Method: {method_name.upper()}  |  Pred VTP: {vtp:.1f}%  |  Pred TSR: {tsr:.1f}%"
    ]
    if gt_biomarkers is not None:
        gvtp = float(gt_biomarkers.get("viable_tumor_percentage", 0.0))
        gtsr = float(gt_biomarkers.get("tumor_stroma_ratio", 0.0))
        footer_lines[0] += f"  |  GT VTP: {gvtp:.1f}%  |  GT TSR: {gtsr:.1f}%"
    if agree_summary is not None:
        footer_lines.append(
            f"Exact Match: {agree_summary['exact_match_pct']:.1f}%  |  "
            f"mIoU (GT-present): {agree_summary['miou_4cls']:.1f}%  |  "
            f"mIoU (Penalized): {agree_summary['miou_penalized_4cls']:.1f}%  |  "
            f"BF1: {agree_summary['boundary_f1']:.1f}%  |  "
            f"Pixels: valid_tissue={agree_summary['valid_tissue_px']}, tp={agree_summary['tp_px']}, "
            f"margin={agree_summary['margin_err_px']}, fp_glass={agree_summary['fp_glass_px']}, "
            f"fn_missed={agree_summary['fn_missed_px']}, misclass={agree_summary['misclass_px']}"
        )

    fig.suptitle(
        "BCSS-WSSS Clinical Visual Verification & Diagnostic Comparison Sheet",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )
    fig.text(
        0.5,
        0.02,
        "\n".join(footer_lines),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#1E293B",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F1F5F9", "edgecolor": "#CBD5E1"},
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.11, hspace=0.28, wspace=0.08)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#FFFFFF")
    return buf.getvalue()
