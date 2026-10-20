"""
Clinical Viewer Service for WSSS Histopathology (SE374.R11).
Provides utilities for color-coded segmentation overlays, OpenSeadragon DZI manifests,
and clinical biomarker visualization for Whole Slide Images (WSI) and ROI patches.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

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


def stitch_tiles_heatmap(
    tiles: List[Dict[str, Any]],
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stitches multi-patch tile predictions into a combined slide-level mask and color heatmap.
    
    Args:
        tiles: List of dicts, each containing:
            - 'x': X-coordinate offset (pixels).
            - 'y': Y-coordinate offset (pixels).
            - 'mask': 2D integer class mask.
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

    max_x = max(t["x"] + t["mask"].shape[1] for t in tiles)
    max_y = max(t["y"] + t["mask"].shape[0] for t in tiles)

    w = max(canvas_width or 0, max_x)
    h = max(canvas_height or 0, max_y)

    composite_mask = np.full((h, w), 4, dtype=np.int64)

    for t in tiles:
        tx, ty = t["x"], t["y"]
        t_mask = t["mask"]
        th, tw = t_mask.shape
        composite_mask[ty : ty + th, tx : tx + tw] = t_mask

    composite_color = mask_to_color_image(composite_mask)
    return composite_mask, composite_color
