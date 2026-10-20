"""
Metrics Service for Medical Histopathology Semantic Segmentation (SE374.R11).
Provides mathematically rigorous, zero-mockup implementations of:
- Confusion Matrix computation
- Mean Intersection over Union (mIoU) and per-class IoU
- Mean Dice Similarity Coefficient (mDice) and per-class Dice
- Clinical Biomarkers: Viable Tumor Percentage (VTP) and Tumor-Stroma Ratio (TSR)

All calculations strictly use standard PyTorch / NumPy arrays.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class MedicalSegmentationMetrics:
    """Calculates multi-class semantic segmentation metrics and clinical tissue ratios."""

    def __init__(
        self,
        num_classes: int = 5,
        class_names: Optional[List[str]] = None,
        ignore_index: Optional[int] = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Resets the internal confusion matrix accumulator."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Updates confusion matrix with batch predictions and ground truth.
        
        Args:
            pred: Predicted class indices (B, H, W) or (H, W).
            target: Ground truth class indices (B, H, W) or (H, W).
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred = pred.astype(np.int64).flatten()
        target = target.astype(np.int64).flatten()

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            pred = pred[valid_mask]
            target = target[valid_mask]

        # Filter out of range values
        valid_range = (target >= 0) & (target < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        pred = pred[valid_range]
        target = target[valid_range]

        # Fast 2D bincount accumulation
        indices = target * self.num_classes + pred
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape((self.num_classes, self.num_classes))

    def compute_iou(self) -> Tuple[np.ndarray, float]:
        """Computes per-class IoU and mean IoU (mIoU).
        
        IoU_c = TP_c / (TP_c + FP_c + FN_c)
        """
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        denominator = tp + fp + fn

        iou = np.zeros(self.num_classes, dtype=np.float64)
        valid_classes = denominator > 0
        iou[valid_classes] = tp[valid_classes] / denominator[valid_classes]

        # mIoU over present classes
        miou = float(np.mean(iou[valid_classes])) if np.any(valid_classes) else 0.0
        return iou, miou

    def compute_dice(self) -> Tuple[np.ndarray, float]:
        """Computes per-class Dice coefficient and mean Dice (mDice).
        
        Dice_c = 2 * TP_c / (2 * TP_c + FP_c + FN_c)
        """
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        denominator = 2 * tp + fp + fn

        dice = np.zeros(self.num_classes, dtype=np.float64)
        valid_classes = denominator > 0
        dice[valid_classes] = (2.0 * tp[valid_classes]) / denominator[valid_classes]

        mdice = float(np.mean(dice[valid_classes])) if np.any(valid_classes) else 0.0
        return dice, mdice

    def get_summary_dict(self) -> Dict[str, float]:
        """Returns structured dictionary with all evaluation metrics."""
        per_class_iou, miou = self.compute_iou()
        per_class_dice, mdice = self.compute_dice()

        results: Dict[str, float] = {
            "mIoU": miou * 100.0,
            "mDice": mdice * 100.0,
        }

        for idx, name in enumerate(self.class_names):
            results[f"IoU_{name}"] = float(per_class_iou[idx] * 100.0)
            results[f"Dice_{name}"] = float(per_class_dice[idx] * 100.0)

        return results


def compute_viable_tumor_percentage(
    pred_mask: Union[torch.Tensor, np.ndarray],
    tumor_class_idx: int = 0,
    background_class_idx: int = 4,
    ignore_index: int = 255,
) -> float:
    """Computes the Viable Tumor Percentage (VTP) across total tissue area (Cellularity).
    
    Formula:
        VTP_cellularity = N_tumor / N_valid_tissue * 100%
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    tumor_pixels = np.sum(pred_mask == tumor_class_idx)
    tissue_pixels = np.sum((pred_mask != background_class_idx) & (pred_mask != ignore_index))

    if tissue_pixels == 0:
        return 0.0
    return float((tumor_pixels / tissue_pixels) * 100.0)


def compute_viable_tumor_bed_percentage(
    pred_mask: Union[torch.Tensor, np.ndarray],
    tumor_class_idx: int = 0,
    necrosis_class_idx: int = 3,
) -> float:
    """Computes Viable Tumor in Tumor Bed (Residual Cancer Burden standard, Symmans et al.).
    
    Formula:
        VTP_tumor_bed = N_tumor / (N_tumor + N_necrosis) * 100%
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    tumor_pixels = np.sum(pred_mask == tumor_class_idx)
    necrosis_pixels = np.sum(pred_mask == necrosis_class_idx)
    tumor_bed = tumor_pixels + necrosis_pixels

    if tumor_bed == 0:
        return 0.0
    return float((tumor_pixels / tumor_bed) * 100.0)


def compute_tumor_stroma_ratio(
    pred_mask: Union[torch.Tensor, np.ndarray],
    tumor_class_idx: int = 0,
    stroma_class_idx: int = 1,
) -> float:
    """Computes the Tumor-Stroma Ratio (TSR) in malignant areas.
    
    Formula:
        TSR = N_tumor / (N_tumor + N_stroma) * 100%
        
    Clinical Note:
        Under International TSR Guidelines (van Pelt et al.), Stroma-Rich High Risk
        corresponds to Stroma > 50% (equivalent to TSR < 50%).
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    tumor_pixels = np.sum(pred_mask == tumor_class_idx)
    stroma_pixels = np.sum(pred_mask == stroma_class_idx)
    total = tumor_pixels + stroma_pixels

    if total == 0:
        return 0.0
    return float((tumor_pixels / total) * 100.0)


def compute_stroma_percentage(
    pred_mask: Union[torch.Tensor, np.ndarray],
    tumor_class_idx: int = 0,
    stroma_class_idx: int = 1,
) -> float:
    """Computes the Stroma Percentage relative to the tumor-stroma compartment.
    
    Formula:
        Stroma_Pct = N_stroma / (N_tumor + N_stroma) * 100% = 100.0% - TSR
    """
    tsr = compute_tumor_stroma_ratio(pred_mask, tumor_class_idx, stroma_class_idx)
    return 100.0 - tsr if tsr > 0 or np.sum((pred_mask == tumor_class_idx) | (pred_mask == stroma_class_idx)) > 0 else 0.0

