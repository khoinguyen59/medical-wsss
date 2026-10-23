"""
Adaptive Coarse-to-Fine Multi-Magnification Router & Pareto Benchmark Engine (SE374.R11).

Implements:
1. Multi-magnification histopathology simulation (40x, 20x, 10x, 5x) following
   LowMagWSS (Do et al., MAPR 2026 / arXiv:2607.10783).
2. Entropy- and Boundary-guided Adaptive Patch Difficulty Scoring.
3. Pareto Frontier Benchmarking (Accuracy vs. Compute Budget) across test/val cohorts.

Strict adherence to Invariant 12: Standard keyboard hyphen ASCII 0x2D ('-') only.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses_metrics.metrics import MedicalSegmentationMetrics, compute_boundary_f1


def simulate_magnification_patch(
    img_rgb: np.ndarray,
    target_mag: int = 40,
    orig_mag: int = 40,
    interpolation: str = "bilinear",
    *,
    base_mag: Optional[int] = None,
) -> np.ndarray:
    """Simulates lower magnification storage degradation and reconstruction back to 224x224.

    Following LowMagWSS benchmark protocol:
    - 40x -> 40x: No degradation (original 224x224).
    - 40x -> 20x: Downsampled to 112x112 (4x compression), reconstructed to 224x224.
    - 40x -> 10x: Downsampled to 56x56 (16x compression), reconstructed to 224x224.
    - 40x -> 5x:  Downsampled to 28x28 (64x compression), reconstructed to 224x224.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image of shape (H, W, 3) in uint8 [0..255].
    target_mag : int
        Target magnification in {5, 10, 20, 40}.
    orig_mag : int
        Source magnification (default 40).
    interpolation : str
        Reconstruction method ('bilinear' or 'bicubic').

    Returns
    -------
    np.ndarray
        Reconstructed RGB image of original shape (H, W, 3).
    """
    if base_mag is not None:
        orig_mag = base_mag

    if target_mag == orig_mag:
        return img_rgb.copy()

    h, w = img_rgb.shape[:2]
    down_scale = float(target_mag) / float(orig_mag)
    target_h = max(4, int(round(h * down_scale)))
    target_w = max(4, int(round(w * down_scale)))

    # Step 1: Simulate optical downsampling / low-resolution storage
    down_img = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Step 2: Reconstruct back to original input resolution
    interp_flag = cv2.INTER_CUBIC if interpolation.lower() == "bicubic" else cv2.INTER_LINEAR
    reconstructed = cv2.resize(down_img, (w, h), interpolation=interp_flag)

    return reconstructed


class CoarseToFineAdaptiveRouter:
    """Orchestrates adaptive Coarse-to-Fine routing between Low-Mag and High-Mag models."""

    def __init__(
        self,
        num_classes: int = 5,
        entropy_weight: float = 0.50,
        boundary_weight: float = 0.30,
        tumor_weight: float = 0.20,
    ) -> None:
        self.num_classes = num_classes
        self.entropy_weight = entropy_weight
        self.boundary_weight = boundary_weight
        self.tumor_weight = tumor_weight

    def compute_patch_difficulty_score(
        self,
        prob_map: np.ndarray,
        pred_mask: np.ndarray,
        cam_prob: Optional[np.ndarray] = None,
    ) -> float:
        """Computes a scalar difficulty score S_route in [0..1] for a patch.

        Higher score indicates complex tumor boundaries or high predictive uncertainty,
        prioritizing the patch for fine 40x M6 multi-scale magnification.

        Parameters
        ----------
        prob_map : np.ndarray
            Class probability distribution (C, H, W) or (H, W, C).
        pred_mask : np.ndarray
            Predicted discrete mask (H, W).
        cam_prob : Optional[np.ndarray]
            Optional Stage-1 image-level probability vector (C,).

        Returns
        -------
        float
            Routing difficulty score in [0.0..1.0].
        """
        if prob_map.ndim == 3 and prob_map.shape[-1] == self.num_classes:
            prob_map = prob_map.transpose(2, 0, 1)

        c, h, w = prob_map.shape
        eps = 1e-7

        # 1. Normalized Shannon Entropy component
        entropy_pixel = -np.sum(prob_map * np.log(prob_map + eps), axis=0) / math.log(c)
        valid_tissue = pred_mask != 4
        mean_entropy = float(np.mean(entropy_pixel[valid_tissue])) if np.any(valid_tissue) else 0.0

        # 2. Boundary transition ratio (morphological gradient)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(pred_mask.astype(np.uint8), kernel)
        eroded = cv2.erode(pred_mask.astype(np.uint8), kernel)
        boundary_pixels = (dilated != eroded) & valid_tissue
        boundary_ratio = float(np.sum(boundary_pixels)) / float(max(1, np.sum(valid_tissue)))

        # 3. Tumor / Minority class presence
        tumor_prob = 0.0
        if cam_prob is not None and len(cam_prob) > 0:
            tumor_prob = float(cam_prob[0])
        else:
            tumor_prob = float(np.mean(prob_map[0, valid_tissue])) if np.any(valid_tissue) else 0.0

        score = (
            self.entropy_weight * mean_entropy
            + self.boundary_weight * boundary_ratio
            + self.tumor_weight * tumor_prob
        )
        return float(np.clip(score, 0.0, 1.0))

    def evaluate_pareto_curve(
        self,
        coarse_preds: Optional[List[np.ndarray]] = None,
        fine_preds: Optional[List[np.ndarray]] = None,
        gt_masks: Optional[List[np.ndarray]] = None,
        difficulty_scores: Optional[List[float]] = None,
        budgets: Optional[List[float]] = None,
        coarse_latency_ms: float = 23.4,
        fine_latency_ms: float = 203.6,
        seed: int = 42,
        *,
        engine: Optional[Any] = None,
        model: Optional[nn.Module] = None,
        patches_40x: Optional[List[np.ndarray]] = None,
        patches_10x: Optional[List[np.ndarray]] = None,
        ground_truths: Optional[List[np.ndarray]] = None,
        budget_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Evaluates accuracy vs compute trade-off curve across budgets beta in [0..1].

        Can be called either with precomputed predictions and difficulty scores,
        or with raw image patches (patches_40x, patches_10x) and inference engine/model.
        """
        if patches_40x is not None and patches_10x is not None and engine is not None and model is not None:
            gt_masks = ground_truths if ground_truths is not None else []
            dev = device if device is not None else torch.device("cpu")
            coarse_preds = []
            fine_preds = []
            difficulty_scores = []
            if budget_steps is not None and budgets is None:
                budgets = np.linspace(0.0, 1.0, budget_steps).tolist()

            with torch.no_grad():
                for p40, p10 in zip(patches_40x, patches_10x):
                    t10 = torch.from_numpy(p10.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(dev)
                    t40 = torch.from_numpy(p40.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(dev)

                    c_mask = engine.predict_m1(model, t10)
                    if hasattr(engine, "predict_m6"):
                        f_mask = engine.predict_m6(model, t40, raw_rgb=p40)
                    else:
                        f_mask = engine.predict_m1(model, t40)

                    coarse_preds.append(c_mask)
                    fine_preds.append(f_mask)

                    # Compute difficulty score
                    prob_map = np.zeros((self.num_classes, c_mask.shape[0], c_mask.shape[1]), dtype=np.float32)
                    for c_idx in range(min(self.num_classes, 4)):
                        prob_map[c_idx][c_mask == c_idx] = 0.8
                    score = self.compute_patch_difficulty_score(prob_map, c_mask)
                    difficulty_scores.append(score)

        if coarse_preds is None or fine_preds is None or gt_masks is None or difficulty_scores is None:
            raise ValueError("Phải cung cấp đủ coarse_preds, fine_preds, gt_masks, difficulty_scores.")
        if budgets is None:
            budgets = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

        n_samples = len(gt_masks)
        sorted_indices = np.argsort(difficulty_scores)[::-1]

        rng = np.random.RandomState(seed)
        random_indices = rng.permutation(n_samples)

        adaptive_results = []
        random_results = []

        for beta in budgets:
            k_fine = int(round(beta * n_samples))
            fine_set_adaptive = set(sorted_indices[:k_fine])
            fine_set_random = set(random_indices[:k_fine])

            # 1. Adaptive Routing Evaluation
            eval_adapt = MedicalSegmentationMetrics(num_classes=self.num_classes)
            bf1_adapt_list = []
            for i in range(n_samples):
                chosen_mask = fine_preds[i] if i in fine_set_adaptive else coarse_preds[i]
                eval_adapt.update(chosen_mask, gt_masks[i])
                bf1_res = compute_boundary_f1(chosen_mask, gt_masks[i], num_classes=self.num_classes)
                bf1_adapt_list.append(bf1_res["BF1"])

            summary_adapt = eval_adapt.get_summary_dict()
            effective_latency = coarse_latency_ms + (beta * fine_latency_ms)
            compute_savings = (1.0 - (effective_latency / fine_latency_ms)) * 100.0

            adaptive_results.append(
                {
                    "budget_pct": beta * 100.0,
                    "mIoU": summary_adapt["mIoU"],
                    "mDice": summary_adapt["mDice"],
                    "FwIoU": summary_adapt["FwIoU"],
                    "Pixel_Accuracy": summary_adapt["Pixel_Accuracy"],
                    "BF1": float(np.mean(bf1_adapt_list)),
                    "effective_latency_ms": effective_latency,
                    "effective_fps": 1000.0 / effective_latency,
                    "compute_savings_pct": max(0.0, compute_savings),
                    "IoU_TUM": summary_adapt.get("IoU_Class_0", 0.0),
                    "IoU_STR": summary_adapt.get("IoU_Class_1", 0.0),
                    "IoU_LYM": summary_adapt.get("IoU_Class_2", 0.0),
                    "IoU_NEC": summary_adapt.get("IoU_Class_3", 0.0),
                }
            )

            # 2. Random Routing Baseline Evaluation
            eval_rand = MedicalSegmentationMetrics(num_classes=self.num_classes)
            bf1_rand_list = []
            for i in range(n_samples):
                chosen_mask = fine_preds[i] if i in fine_set_random else coarse_preds[i]
                eval_rand.update(chosen_mask, gt_masks[i])
                bf1_res = compute_boundary_f1(chosen_mask, gt_masks[i], num_classes=self.num_classes)
                bf1_rand_list.append(bf1_res["BF1"])

            summary_rand = eval_rand.get_summary_dict()
            random_results.append(
                {
                    "budget_pct": beta * 100.0,
                    "mIoU": summary_rand["mIoU"],
                    "mDice": summary_rand["mDice"],
                    "FwIoU": summary_rand["FwIoU"],
                    "Pixel_Accuracy": summary_rand["Pixel_Accuracy"],
                    "BF1": float(np.mean(bf1_rand_list)),
                    "effective_latency_ms": effective_latency,
                    "effective_fps": 1000.0 / effective_latency,
                    "compute_savings_pct": max(0.0, compute_savings),
                }
            )

        adaptive_miou = [r["mIoU"] for r in adaptive_results]
        random_miou = [r["mIoU"] for r in random_results]
        auc_adapt = float(np.trapezoid(adaptive_miou, budgets)) if hasattr(np, "trapezoid") else float(np.trapz(adaptive_miou, budgets))
        auc_rand = float(np.trapezoid(random_miou, budgets)) if hasattr(np, "trapezoid") else float(np.trapz(random_miou, budgets))
        auc_gain = max(0.0, float(auc_adapt - auc_rand))

        return {
            "num_samples": n_samples,
            "budgets": budgets,
            "adaptive_routing": adaptive_results,
            "random_routing": random_results,
            "adaptive_miou": adaptive_miou,
            "random_miou": random_miou,
            "auc_gain": auc_gain,
        }
