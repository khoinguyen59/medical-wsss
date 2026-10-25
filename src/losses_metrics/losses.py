"""
Loss Functions for Weakly-Supervised Histopathology Segmentation (SE374.R11).
Provides mathematically grounded losses:
- Multi-layer pseudo-supervision loss (MLPS Loss)
- Boundary uncertainty regularization loss
- Multi-label patch classification loss
- Hybrid Cross-Entropy and Dice loss for pseudo-mask supervision
"""

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelClassificationLoss(nn.Module):
    """Multi-label soft-margin binary cross entropy for weak patch labels."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: (B, C) classification logits.
            targets: (B, C) multi-hot float targets.
        """
        return self.loss_fn(logits, targets)


class MultiLayerPseudoSupervisionLoss(nn.Module):
    """Loss for training segmentation networks using pseudo-masks from multiple feature layers.
    
    Formula from Han et al. (MedIA 2022):
        L_total = L_bn7 + alpha * L_b5_2 + beta * L_b4_5
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.weights = weights or {"bn7": 1.0, "b5_2": 0.5, "b4_5": 0.25}
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        pred_logits: torch.Tensor,
        pseudo_masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Args:
            pred_logits: (B, C, H, W) model prediction logits.
            pseudo_masks: Dictionary with keys 'bn7', 'b5_2', 'b4_5' containing (B, H, W) masks.
        """
        total_loss = pred_logits.sum() * 0.0
        has_layer = False
        for layer_name, weight in self.weights.items():
            if layer_name in pseudo_masks:
                has_layer = True
                target_mask = pseudo_masks[layer_name]
                # If target resolution differs, interpolate logits to target resolution
                if pred_logits.shape[-2:] != target_mask.shape[-2:]:
                    scaled_logits = F.interpolate(
                        pred_logits,
                        size=target_mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    scaled_logits = pred_logits
                target_clamped = target_mask.long().clamp(0, scaled_logits.shape[1] - 1)
                loss = self.ce(scaled_logits, target_clamped)
                total_loss = total_loss + weight * loss

        return total_loss


class UncertaintyRegularizationLoss(nn.Module):
    """Boundary uncertainty-guided regularization loss.
    
    Down-weights high-entropy boundary pixels where pseudo-labels are inherently ambiguous,
    preventing over-penalization of transitional tumor-stroma margins.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.65,
        ignore_index: int = 255,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        pred_logits: torch.Tensor,
        pseudo_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            pred_logits: (B, C, H, W) prediction logits.
            pseudo_mask: (B, H, W) pseudo-mask labels.
        """
        probs = F.softmax(pred_logits, dim=1)  # (B, C, H, W)
        # Compute normalized entropy: H(p) = - sum(p * log(p)) / log(C)
        num_classes = probs.shape[1]
        log_probs = torch.log(probs.clamp(min=1e-7))
        norm_factor = math.log(float(max(num_classes, 2))) + 1e-7
        entropy = -torch.sum(probs * log_probs, dim=1) / norm_factor  # (B, H, W)

        # Detach weight tensor so gradients do not flow through the uncertainty weighting term
        weight = torch.clamp(1.0 - (entropy / self.entropy_threshold), min=0.1, max=1.0).detach()  # (B, H, W)

        ce_per_pixel = F.cross_entropy(
            pred_logits,
            pseudo_mask.long(),
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)

        # Mask out ignore_index pixels
        valid_mask = (pseudo_mask != self.ignore_index).float()
        weighted_loss = ce_per_pixel * weight * valid_mask

        if self.reduction == "mean":
            valid_count = valid_mask.sum().clamp(min=1.0)
            return weighted_loss.sum() / valid_count
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss
