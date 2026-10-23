"""
Metrics Service for Medical Histopathology Semantic Segmentation (SE374.R11).
Provides mathematically rigorous, zero-mockup implementations of:
- Confusion Matrix computation
- Mean Intersection over Union (mIoU) and per-class IoU
- Mean Dice Similarity Coefficient (mDice) and per-class Dice
- Clinical Biomarkers: Viable Tumor Percentage (VTP) and Tumor-Stroma Ratio (TSR)

All calculations strictly use standard PyTorch / NumPy arrays.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from scipy import ndimage, stats


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

    def compute_fwiou(self) -> float:
        """Computes Frequency-Weighted Intersection over Union (FwIoU).
        
        Formula (LowMagWSS & WSSS-Tissue standard):
            FwIoU = sum_c [ (N_c / N_total) * IoU_c ]
        """
        total_pixels = float(np.sum(self.confusion_matrix))
        if total_pixels == 0.0:
            return 0.0
        class_pixel_counts = np.sum(self.confusion_matrix, axis=1)
        freq = class_pixel_counts / total_pixels
        iou, _ = self.compute_iou()
        valid = class_pixel_counts > 0
        return float(np.sum(freq[valid] * iou[valid]))

    def compute_pixel_accuracy(self) -> float:
        """Computes overall Pixel Accuracy (PA = sum(TP) / total_pixels)."""
        total_pixels = float(np.sum(self.confusion_matrix))
        if total_pixels == 0.0:
            return 0.0
        tp = np.diag(self.confusion_matrix)
        return float(np.sum(tp) / total_pixels)

    def get_summary_dict(self) -> Dict[str, float]:
        """Returns structured dictionary with all evaluation metrics."""
        per_class_iou, miou = self.compute_iou()
        per_class_dice, mdice = self.compute_dice()
        fwiou = self.compute_fwiou()
        pa = self.compute_pixel_accuracy()

        results: Dict[str, float] = {
            "mIoU": miou * 100.0,
            "mDice": mdice * 100.0,
            "FwIoU": fwiou * 100.0,
            "Pixel_Accuracy": pa * 100.0,
        }

        for idx, name in enumerate(self.class_names):
            results[f"IoU_{name}"] = float(per_class_iou[idx] * 100.0)
            results[f"Dice_{name}"] = float(per_class_dice[idx] * 100.0)

        return results


def compute_fwiou(
    pred_mask: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    num_classes: int = 5,
    ignore_index: Optional[int] = 255,
) -> float:
    """Computes Frequency-Weighted IoU (FwIoU) for a pair of prediction and ground truth masks."""
    evaluator = MedicalSegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    evaluator.update(pred_mask, gt_mask)
    return evaluator.compute_fwiou()


def compute_pixel_accuracy(
    pred_mask: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    ignore_index: Optional[int] = 255,
) -> float:
    """Computes global Pixel Accuracy between prediction and ground truth masks."""
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    pred_mask = pred_mask.squeeze()
    gt_mask = gt_mask.squeeze()

    if ignore_index is not None:
        valid = gt_mask != ignore_index
        if not np.any(valid):
            return 0.0
        return float(np.mean(pred_mask[valid] == gt_mask[valid]))
    return float(np.mean(pred_mask == gt_mask))


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


def compute_boundary_f1(
    pred_mask: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    num_classes: int = 5,
    tolerance_px: int = 2,
    ignore_index: Optional[int] = 255,
) -> Dict[str, float]:
    """Computes Boundary Precision, Boundary Recall, and Boundary F1-Score (BF1).

    Evaluates morphological boundary alignment along tissue and tumor interfaces
    within a pixel tolerance theta (default theta = 2 pixels).
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    pred_mask = pred_mask.squeeze()
    gt_mask = gt_mask.squeeze()

    struct_elem = ndimage.generate_binary_structure(2, 1)
    if tolerance_px > 1:
        struct_elem = ndimage.iterate_structure(struct_elem, tolerance_px)

    precisions = []
    recalls = []
    f1s = []

    valid_mask = np.ones_like(gt_mask, dtype=bool)
    if ignore_index is not None:
        valid_mask = gt_mask != ignore_index

    for c in range(num_classes):
        # Foreground binary masks for class c
        bin_pred = (pred_mask == c) & valid_mask
        bin_gt = (gt_mask == c) & valid_mask

        if not np.any(bin_gt) and not np.any(bin_pred):
            continue

        # Extract 1-pixel boundary: mask XOR eroded(mask)
        eroded_pred = ndimage.binary_erosion(bin_pred)
        bound_pred = bin_pred ^ eroded_pred

        eroded_gt = ndimage.binary_erosion(bin_gt)
        bound_gt = bin_gt ^ eroded_gt

        n_pred_bound = float(np.sum(bound_pred))
        n_gt_bound = float(np.sum(bound_gt))

        if n_pred_bound == 0.0 and n_gt_bound == 0.0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        elif n_pred_bound == 0.0 or n_gt_bound == 0.0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        # Dilate ground truth boundary by tolerance to form hit region
        dilated_gt_bound = ndimage.binary_dilation(bound_gt, structure=struct_elem)
        dilated_pred_bound = ndimage.binary_dilation(bound_pred, structure=struct_elem)

        precision = float(np.sum(bound_pred & dilated_gt_bound) / n_pred_bound)
        recall = float(np.sum(bound_gt & dilated_pred_bound) / n_gt_bound)

        if precision + recall > 0.0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    mean_f1 = float(np.mean(f1s) * 100.0) if f1s else 0.0
    mean_precision = float(np.mean(precisions) * 100.0) if precisions else 0.0
    mean_recall = float(np.mean(recalls) * 100.0) if recalls else 0.0

    return {
        "BF1": mean_f1,
        "Boundary_Precision": mean_precision,
        "Boundary_Recall": mean_recall,
    }


def compute_patient_biomarker_correlation(
    patient_records: Union[List[Dict[str, Any]], Dict[str, Dict[str, float]]],
    gt_biomarkers: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Computes statistical correlation and clinical risk concordance across patients.

    Parameters
    ----------
    patient_records : Union[List[Dict[str, Any]], Dict[str, Dict[str, float]]]
        Either a list of patient record dicts with keys 'pred_vtp', 'gt_vtp', 'pred_tsr', 'gt_tsr',
        or a dictionary mapping patient_id to {"vtp": float, "tsr": float} (when gt_biomarkers is also provided).
    gt_biomarkers : Optional[Dict[str, Dict[str, float]]]
        Optional dictionary mapping patient_id to ground truth {"vtp": float, "tsr": float}.

    Returns
    -------
    Dict[str, Any]
        Dictionary with Pearson r, Spearman rho, MAE, RMSE, and CAP concordance.
    """
    if gt_biomarkers is not None and isinstance(patient_records, dict):
        formatted_records = []
        for pid in sorted(patient_records.keys()):
            if pid in gt_biomarkers:
                formatted_records.append({
                    "patient_id": pid,
                    "pred_vtp": float(patient_records[pid].get("vtp", 0.0)),
                    "gt_vtp": float(gt_biomarkers[pid].get("vtp", 0.0)),
                    "pred_tsr": float(patient_records[pid].get("tsr", 0.0)),
                    "gt_tsr": float(gt_biomarkers[pid].get("tsr", 0.0)),
                })
        patient_records = formatted_records

    if not patient_records:
        return {
            "patient_count": 0,
            "vtp_pearson_r": 0.0,
            "vtp_pearson_p": 1.0,
            "vtp_spearman_rho": 0.0,
            "vtp_spearman_p": 1.0,
            "vtp_mae": 0.0,
            "vtp_rmse": 0.0,
            "tsr_pearson_r": 0.0,
            "tsr_pearson_p": 1.0,
            "tsr_spearman_rho": 0.0,
            "tsr_spearman_p": 1.0,
            "tsr_mae": 0.0,
            "tsr_rmse": 0.0,
            "cap_risk_concordance_pct": 0.0,
        }

    pred_vtp = np.array([r["pred_vtp"] for r in patient_records], dtype=np.float64)
    gt_vtp = np.array([r["gt_vtp"] for r in patient_records], dtype=np.float64)
    pred_tsr = np.array([r["pred_tsr"] for r in patient_records], dtype=np.float64)
    gt_tsr = np.array([r["gt_tsr"] for r in patient_records], dtype=np.float64)

    n = len(patient_records)

    # VTP correlations and errors
    vtp_mae = float(np.mean(np.abs(pred_vtp - gt_vtp)))
    vtp_rmse = float(np.sqrt(np.mean((pred_vtp - gt_vtp) ** 2)))

    if n >= 2 and np.std(pred_vtp) > 1e-6 and np.std(gt_vtp) > 1e-6:
        vtp_pearson = stats.pearsonr(pred_vtp, gt_vtp)
        vtp_spearman = stats.spearmanr(pred_vtp, gt_vtp)
        vtp_r, vtp_r_p = float(vtp_pearson.statistic), float(vtp_pearson.pvalue)
        vtp_rho, vtp_rho_p = float(vtp_spearman.statistic), float(vtp_spearman.pvalue)
    else:
        vtp_r, vtp_r_p = 1.0 if np.allclose(pred_vtp, gt_vtp) else 0.0, 1.0
        vtp_rho, vtp_rho_p = 1.0 if np.allclose(pred_vtp, gt_vtp) else 0.0, 1.0

    # TSR correlations and errors
    tsr_mae = float(np.mean(np.abs(pred_tsr - gt_tsr)))
    tsr_rmse = float(np.sqrt(np.mean((pred_tsr - gt_tsr) ** 2)))

    if n >= 2 and np.std(pred_tsr) > 1e-6 and np.std(gt_tsr) > 1e-6:
        tsr_pearson = stats.pearsonr(pred_tsr, gt_tsr)
        tsr_spearman = stats.spearmanr(pred_tsr, gt_tsr)
        tsr_r, tsr_r_p = float(tsr_pearson.statistic), float(tsr_pearson.pvalue)
        tsr_rho, tsr_rho_p = float(tsr_spearman.statistic), float(tsr_spearman.pvalue)
    else:
        tsr_r, tsr_r_p = 1.0 if np.allclose(pred_tsr, gt_tsr) else 0.0, 1.0
        tsr_rho, tsr_rho_p = 1.0 if np.allclose(pred_tsr, gt_tsr) else 0.0, 1.0

    # CAP Risk Category Concordance: Stroma-Rich High Risk if TSR <= 50%
    pred_high_risk = pred_tsr <= 50.0
    gt_high_risk = gt_tsr <= 50.0
    cap_concordance = float(np.mean(pred_high_risk == gt_high_risk) * 100.0)

    return {
        "patient_count": n,
        "vtp_pearson_r": vtp_r,
        "vtp_pearson_p": vtp_r_p,
        "vtp_spearman_rho": vtp_rho,
        "vtp_spearman_p": vtp_rho_p,
        "vtp_mae": vtp_mae,
        "vtp_rmse": vtp_rmse,
        "tsr_pearson_r": tsr_r,
        "tsr_pearson_p": tsr_r_p,
        "tsr_spearman_rho": tsr_rho,
        "tsr_spearman_p": tsr_rho_p,
        "tsr_mae": tsr_mae,
        "tsr_rmse": tsr_rmse,
        "cap_risk_concordance_pct": cap_concordance,
    }

