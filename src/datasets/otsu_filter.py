"""
Otsu Clinical Gating Service for Histopathology (SE374.R11).
Segments tissue foreground from glass background / adipose tissue on H&E images
using classical Otsu thresholding in HSV / Grayscale space.

Prevents CAM argmax errors on empty background slides.
"""

from typing import Tuple, Union
import cv2
import numpy as np
import torch


def compute_otsu_tissue_mask(
    image: Union[np.ndarray, torch.Tensor],
    saturation_threshold_multiplier: float = 0.8,
) -> np.ndarray:
    """Computes a binary tissue mask where 1 indicates tissue and 0 indicates glass/background.
    
    Args:
        image: RGB image as (H, W, 3), (3, H, W), or batched (B, C, H, W) / (B, H, W, 3).
        saturation_threshold_multiplier: Sensitivity multiplier for Otsu threshold.
        
    Returns:
        binary_mask: (H, W) or (B, H, W) uint8 mask (1 for tissue, 0 for glass).
    """
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            masks = [
                compute_otsu_tissue_mask(img_slice, saturation_threshold_multiplier)
                for img_slice in image
            ]
            return np.stack(masks, axis=0)
        elif image.ndim == 3 and image.shape[0] in [1, 3]:
            # (C, H, W) -> (H, W, C)
            image = image.permute(1, 2, 0).detach().cpu().numpy()
        else:
            image = image.detach().cpu().numpy()

    if isinstance(image, np.ndarray) and image.ndim == 4:
        masks = [
            compute_otsu_tissue_mask(img_slice, saturation_threshold_multiplier)
            for img_slice in image
        ]
        return np.stack(masks, axis=0)

    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            image_uint8 = (image * 255.0).astype(np.uint8)
        else:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image

    if image_uint8.ndim == 2:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    elif image_uint8.shape[-1] == 1:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    # Convert to HSV: Tissue exhibits significant Saturation; glass is near 0
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    # Otsu thresholding on saturation channel, capped at 20 so unimodal tissue
    # patches do not split high-saturation nuclei from pale cytoplasm/stroma
    otsu_thresh, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    effective_thresh = min(int(otsu_thresh * saturation_threshold_multiplier), 20)

    # Invert grayscale for supplementary intensity gating (glass is bright white >= 235)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    is_non_white = gray < 235
    is_stained_or_dark = (sat >= effective_thresh) | (gray < 220)

    tissue_mask = (is_stained_or_dark & is_non_white).astype(np.uint8)

    # Morphological clean-up: remove tiny specks, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    # Suppress small intracellular vacuoles/clearings (< 256 px) inside larger tissue patches
    if tissue_mask.size >= 1024 and np.any(tissue_mask == 1):
        bg_mask = (tissue_mask == 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < 256:
                tissue_mask[labels == lbl] = 1

    return tissue_mask


def apply_otsu_gating_to_prediction(
    pred_mask: Union[np.ndarray, torch.Tensor],
    image: Union[np.ndarray, torch.Tensor],
    background_class_idx: int = 4,
) -> np.ndarray:
    """Hard-gates semantic prediction by forcing glass/non-tissue pixels to background class.
    
    Args:
        pred_mask: (H, W) or (B, H, W) class prediction indices.
        image: Corresponding RGB image or batch of images.
        background_class_idx: Index of background/other class (default: 4 for BCSS).
        
    Returns:
        gated_mask: Array with non-tissue areas forced to background_class_idx.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    gated_mask = pred_mask.copy()
    if gated_mask.ndim == 3:
        for b in range(gated_mask.shape[0]):
            sub_img = image[b] if (hasattr(image, "shape") and image.ndim == 4) else image
            t_mask = compute_otsu_tissue_mask(sub_img)
            gated_mask[b][t_mask == 0] = background_class_idx
    else:
        tissue_mask = compute_otsu_tissue_mask(image)
        gated_mask[tissue_mask == 0] = background_class_idx
    return gated_mask
