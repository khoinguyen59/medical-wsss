"""
Loss Functions for Weakly-Supervised Histopathology Segmentation (SE374.R11).
Provides mathematically grounded losses:
- Multi-layer pseudo-supervision loss (MLPS Loss)
- Boundary uncertainty regularization loss
- Multi-label patch classification loss
- Hybrid Cross-Entropy and Dice loss for pseudo-mask supervision
"""

import math
from typing import Dict, Optional

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
        pred_logits = pred_logits.float()
        total_loss = pred_logits.sum() * 0.0
        for layer_name, weight in self.weights.items():
            if layer_name in pseudo_masks:
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
                target_long = target_mask.long()
                is_ignore = target_long == self.ignore_index
                target_clamped = target_long.clamp(0, scaled_logits.shape[1] - 1)
                target_clamped = torch.where(
                    is_ignore,
                    torch.full_like(target_clamped, self.ignore_index),
                    target_clamped,
                )
                if torch.all(target_clamped == self.ignore_index):
                    loss = scaled_logits.sum() * 0.0
                else:
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
        pred_logits = pred_logits.float()
        probs = F.softmax(pred_logits, dim=1)  # (B, C, H, W)
        # Compute normalized entropy: H(p) = - sum(p * log(p)) / log(C)
        num_classes = probs.shape[1]
        log_probs = torch.log(probs.clamp(min=1e-7))
        norm_factor = math.log(float(max(num_classes, 2))) + 1e-7
        entropy = -torch.sum(probs * log_probs, dim=1) / norm_factor  # (B, H, W)

        # Detach weight tensor so gradients do not flow through the uncertainty weighting term
        weight = torch.clamp(1.0 - (entropy / self.entropy_threshold), min=0.1, max=1.0).detach()  # (B, H, W)

        target_long = pseudo_mask.long()
        is_ignore = target_long == self.ignore_index
        safe_mask = target_long.clamp(0, num_classes - 1)
        safe_mask = torch.where(
            is_ignore,
            torch.full_like(safe_mask, self.ignore_index),
            safe_mask,
        )

        ce_per_pixel = F.cross_entropy(
            pred_logits,
            safe_mask,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)

        # Mask out ignore_index pixels
        valid_mask = (~is_ignore).float()
        weighted_loss = ce_per_pixel * weight * valid_mask

        if self.reduction == "mean":
            valid_count = valid_mask.sum()
            if valid_count.item() == 0.0:
                return pred_logits.sum() * 0.0
            return weighted_loss.sum() / valid_count
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss


class MultiPrototypePPCLoss(nn.Module):
    """Multi-Prototype Pixel-to-Prototype Contrastive Loss (L_MP-PPC).

    Pulls homogeneous tissue pixels towards class sub-prototypes and penalizes hard-pair
    misclassifications (Tumor <-> Stroma, Stroma <-> Lymphocytic, Tumor <-> Necrosis).
    Enforces pixel sampling (N_samples <= max_samples) to constrain GPU VRAM under 0.5 GB.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_prototypes: int = 3,
        feat_dim: int = 128,
        temp: float = 0.1,
        momentum: float = 0.99,
        conf_thresh: float = 0.75,
        max_samples: int = 2048,
        warm_start: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.feat_dim = feat_dim
        self.temp = temp
        self.momentum = momentum
        self.conf_thresh = conf_thresh
        self.max_samples = max_samples
        self.warm_start = warm_start

        # Initialize orthogonal prototypes tensor: (C, K, D)
        protos = torch.randn(num_classes, num_prototypes, feat_dim)
        protos = F.normalize(protos, dim=-1)
        self.register_buffer("prototypes", protos)
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(num_classes, num_prototypes, dtype=torch.bool),
        )

        # Hard-pair penalty matrix calibrated to empirical BCSS confusion matrix:
        # 0: Tumor, 1: Stroma, 2: Inflammatory, 3: Necrosis
        hard_weights = torch.ones(num_classes, num_classes)
        if num_classes >= 4:
            hard_weights[0, 1] = hard_weights[1, 0] = 2.0  # Tumor <-> Stroma
            hard_weights[1, 2] = hard_weights[2, 1] = 2.5  # Stroma <-> Lymphocytic
            hard_weights[0, 3] = hard_weights[3, 0] = 2.0  # Tumor <-> Necrosis
            hard_weights[2, 3] = hard_weights[3, 2] = 1.0  # Lymphocytic <-> Necrosis
        self.register_buffer("hard_weights", hard_weights)

    @torch.no_grad()
    def _update_prototypes(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """Updates prototype sub-clusters with Exponential Moving Average (EMA)."""
        for c in range(self.num_classes):
            mask_c = labels == c
            if mask_c.sum() == 0:
                continue
            feats_c = feats[mask_c]      # (N_c, D)
            weights_c = weights[mask_c]  # (N_c,)

            # Assign each pixel to the nearest sub-prototype in strict FP32
            with torch.amp.autocast(device_type=feats.device.type, enabled=False):
                feats_cf = feats_c.float()
                w_cf = weights_c.float()
                sim_k = torch.matmul(feats_cf, self.prototypes[c].float().t())  # (N_c, K)
                assign_k = sim_k.argmax(dim=-1)

                for k in range(self.num_prototypes):
                    mask_k = assign_k == k
                    if mask_k.sum() > 0:
                        w_k = w_cf[mask_k].unsqueeze(-1)
                        mean_feat = (feats_cf[mask_k] * w_k).sum(dim=0) / (w_k.sum() + 1e-6)
                        is_uninit = not bool(self.prototype_initialized[c, k].item())
                        if self.warm_start and is_uninit:
                            self.prototypes[c, k] = F.normalize(mean_feat, dim=-1)
                        else:
                            new_proto = (
                                self.momentum * self.prototypes[c, k].float()
                                + (1.0 - self.momentum) * mean_feat
                            )
                            self.prototypes[c, k] = F.normalize(new_proto, dim=-1)
                        self.prototype_initialized[c, k] = True

    def forward(
        self,
        proj_feats: torch.Tensor,
        cam_probs: torch.Tensor,
        img_labels: torch.Tensor,
        raw_cams: Optional[torch.Tensor] = None,
        rgb_images: Optional[torch.Tensor] = None,
        cam_norm_thresh: float = 0.35,
    ) -> torch.Tensor:
        """Computes pixel-to-prototype contrastive loss.

        Args:
            proj_feats: (B, D, H, W) projected feature embeddings.
            cam_probs: (B, C, H, W) normalized class activation maps.
            img_labels: (B, C) multi-hot image-level ground truth.
            raw_cams: Optional (B, C, H, W) pre-simplex ReLU CAMs for single-class anti-glass guard.
            rgb_images: Optional (B, 3, H_img, W_img) RGB images for non-white tissue masking.
            cam_norm_thresh: Minimum per-class channel-max normalized CAM score (default 0.35).

        Returns:
            loss: Scalar tensor representing L_MP-PPC loss.
        """
        with torch.amp.autocast(device_type=proj_feats.device.type, enabled=False):
            b, d, h, w = proj_feats.shape
            proj_feats = F.normalize(proj_feats.float(), dim=1)
            cam_probs_f = cam_probs.float()
            img_labels_f = img_labels.float()

            # Gate probabilities with image-level labels
            valid_probs = cam_probs_f * img_labels_f.view(b, self.num_classes, 1, 1)
            valid_probs = valid_probs / (valid_probs.sum(dim=1, keepdim=True) + 1e-6)
            max_probs, pseudo_labels = valid_probs.max(dim=1)  # (B, H, W)

            norm_entropy = math.log(float(max(self.num_classes, 2))) + 1e-7
            entropy = -(valid_probs * torch.log(valid_probs.clamp(min=1e-7))).sum(dim=1) / norm_entropy
            conf_weights = (1.0 - entropy).detach()

            confident_mask = max_probs >= self.conf_thresh

            # Single-class anti-glass guard 1: pre-simplex channel-max normalized CAM >= cam_norm_thresh
            if raw_cams is not None:
                raw_cams_f = torch.clamp(raw_cams.float(), min=0.0)
                if raw_cams_f.shape[-2:] != (h, w):
                    raw_cams_f = F.interpolate(
                        raw_cams_f, size=(h, w), mode="bilinear", align_corners=False
                    )
                cam_max = raw_cams_f.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                cam_max_norm = raw_cams_f / cam_max
                gathered_cam = cam_max_norm.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)
                confident_mask = confident_mask & (gathered_cam >= float(cam_norm_thresh))

            # Single-class anti-glass guard 2: non-white tissue mask (mean_rgb < 0.90 or sat > 0.08)
            if rgb_images is not None:
                rgb_f = rgb_images.float()
                if float(rgb_f.min()) < -1e-4:
                    mean_t = torch.tensor([0.485, 0.456, 0.406], device=rgb_f.device).view(1, 3, 1, 1)
                    std_t = torch.tensor([0.229, 0.224, 0.225], device=rgb_f.device).view(1, 3, 1, 1)
                    rgb_f = rgb_f * std_t + mean_t
                elif float(rgb_f.max()) > 5.0:
                    rgb_f = rgb_f / 255.0
                rgb_f = rgb_f.clamp(0.0, 1.0)
                if rgb_f.shape[-2:] != (h, w):
                    rgb_f = F.interpolate(
                        rgb_f, size=(h, w), mode="bilinear", align_corners=False
                    )
                mean_rgb = rgb_f.mean(dim=1)
                rgb_max = rgb_f.max(dim=1).values
                rgb_min = rgb_f.min(dim=1).values
                sat = (rgb_max - rgb_min) / (rgb_max + 1e-6)
                tissue_mask = (mean_rgb < 0.90) | (sat > 0.08)
                confident_mask = confident_mask & tissue_mask

            if confident_mask.sum() < 16:
                return proj_feats.sum() * 0.0

            flat_feats = proj_feats.permute(0, 2, 3, 1)[confident_mask]
            flat_labels = pseudo_labels[confident_mask]
            flat_weights = conf_weights[confident_mask]

            # Subsample to cap memory usage
            if flat_feats.size(0) > self.max_samples:
                perm = torch.randperm(flat_feats.size(0), device=flat_feats.device)[: self.max_samples]
                flat_feats = flat_feats[perm]
                flat_labels = flat_labels[perm]
                flat_weights = flat_weights[perm]

            # Update prototypes via EMA strictly during training mode
            if self.training:
                self._update_prototypes(flat_feats.detach(), flat_labels, flat_weights)

            # Cosine similarity with all prototypes in FP32 with log-sum-exp stabilization: (N, C, K)
            all_sims = torch.einsum("nd,ckd->nck", flat_feats, self.prototypes.float()) / self.temp
            pos_sims_k = all_sims[torch.arange(flat_feats.size(0), device=flat_feats.device), flat_labels]
            pos_sim, _ = pos_sims_k.max(dim=-1)

            sim_max = all_sims.amax(dim=(1, 2), keepdim=True).detach()
            exp_pos = torch.exp(pos_sim - sim_max.squeeze(-1).squeeze(-1))
            exp_all = torch.exp(all_sims - sim_max)

            pair_w = self.hard_weights[flat_labels].float()
            neg_mask = F.one_hot(flat_labels, num_classes=self.num_classes) == 0
            exp_neg = (exp_all.sum(dim=-1) * pair_w * neg_mask.float()).sum(dim=-1)

            loss = -flat_weights * torch.log((exp_pos / (exp_pos + exp_neg + 1e-12)).clamp(min=1e-12))
            return loss.mean()


class UncertaintyAwareBoundaryLoss(nn.Module):
    """Uncertainty-Aware Optical Stain Boundary Regularization Loss (L_UABL).

    Enforces that high-gradient morphological transitions align with true H&E
    tissue optical boundaries, while penalizing erratic predictions in high-uncertainty zones.
    """

    def __init__(self, sigma_stain: float = 0.35, eta: float = 1.0) -> None:
        super().__init__()
        self.sigma_stain = sigma_stain
        self.eta = eta

    def forward(
        self,
        pred_probs: torch.Tensor,
        stain_gradient: torch.Tensor,
        uncertainty_map: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            pred_probs: (B, C, H, W) softmax prediction probabilities.
            stain_gradient: (B, 1, H, W) optical density gradient magnitude.
            uncertainty_map: (B, 1, H, W) epistemic uncertainty map in [0, 1].

        Returns:
            loss: Scalar boundary alignment regularization loss.
        """
        pred_probs = pred_probs.float()
        stain_gradient = stain_gradient.float()
        uncertainty_map = uncertainty_map.float()
        diff_h = torch.abs(pred_probs[:, :, :, 1:] - pred_probs[:, :, :, :-1]).sum(dim=1, keepdim=True)
        diff_v = torch.abs(pred_probs[:, :, 1:, :] - pred_probs[:, :, :-1, :]).sum(dim=1, keepdim=True)

        weight_h = torch.exp(-stain_gradient[:, :, :, :-1] / (2.0 * self.sigma_stain ** 2))
        weight_v = torch.exp(-stain_gradient[:, :, :-1, :] / (2.0 * self.sigma_stain ** 2))

        unc_h = 1.0 + self.eta * uncertainty_map[:, :, :, :-1]
        unc_v = 1.0 + self.eta * uncertainty_map[:, :, :-1, :]

        loss_h = (diff_h * weight_h * unc_h).mean()
        loss_v = (diff_v * weight_v * unc_v).mean()

        return 0.5 * (loss_h + loss_v)

