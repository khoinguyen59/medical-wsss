"""
Stage-2 v2 FP32 Loss Functions for Soft Pseudo-Label Knowledge Distillation.

All losses run inside torch.amp.autocast(device_type=..., enabled=False) in FP32
to avoid BF16/FP16 log-softmax, entropy, and top-k MIL underflow:
- SoftCrossEntropyLoss
- KLSoftOEEMLoss (zero-bias KL(q||p) with detached adaptive weighting on multi-class patches)
- SoftSymmetricCELoss (robust to pseudo-label noise without spatial .view() corruption)
- SoftMulticlassDiceLoss
- MILPatchPresenceLoss (top-1% present MIL + negative-log-1-minus-p absent suppression)
- Stage2V2CompositeLoss

Pure ASCII hyphen '-' only (Invariant 12).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stage2_v2.config import Stage2V2Config


def _masked_mean(loss_map: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    if weight is None:
        return loss_map.mean()
    w = weight.float()
    return (loss_map * w).sum() / w.sum().clamp(min=1.0)


class SoftCrossEntropyLoss(nn.Module):
    """Pixel-wise soft cross-entropy -sum_c q_c * log_softmax(logits)_c in FP32."""

    def forward(
        self,
        logits: torch.Tensor,
        target_soft: torch.Tensor,
        valid_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            log_p = F.log_softmax(logits.float(), dim=1)
            q = target_soft.float().clamp(min=0.0)
            ce_pix = -(q * log_p).sum(dim=1)
            return _masked_mean(ce_pix, valid_weight)


class KLSoftOEEMLoss(nn.Module):
    """Zero-bias KL(q || p) loss with Online Easy Example Mining (OEEM) pixel weighting.

    - Computes kl_pix = (H(q, p) - H(q)).clamp(min=0.0), which is strictly 0.0 when p == q.
    - Computes detached per-image weights w = softmax(-kl_pix.detach() / tau) * (H * W)
      clamped to [0.25, 2.0] ONLY on multi-class patches (y_gt.sum(1) >= 2) after warmup.
    - Strictly keeps w = 1.0 on single-class patches (y_gt.sum(1) <= 1) and during
      epoch < oeem_warmup_epochs.
    """

    def __init__(self, tau: float = 1.5, oeem_warmup_epochs: int = 2) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError(f"oeem tau must be > 0, got {tau}")
        self.tau = float(tau)
        self.oeem_warmup_epochs = int(oeem_warmup_epochs)

    def compute_kl_and_weights(
        self,
        logits: torch.Tensor,
        target_soft: torch.Tensor,
        y_gt: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            logits_f = logits.float()
            log_p = F.log_softmax(logits_f, dim=1)
            q = target_soft.float().clamp(min=0.0)
            q_safe = q.clamp(min=1e-6)
            h_qp = -(q * log_p).sum(dim=1)
            h_q = -(q * torch.log(q_safe)).sum(dim=1)
            kl_pix = (h_qp - h_q).clamp(min=0.0)

            b, _, h, w = logits_f.shape
            if epoch < self.oeem_warmup_epochs or y_gt is None:
                weights = torch.ones_like(kl_pix)
            else:
                kl_det = kl_pix.detach().reshape(b, h * w)
                raw_w = F.softmax(-kl_det / self.tau, dim=-1) * float(h * w)
                w_mc = raw_w.clamp(min=0.25, max=2.0).reshape(b, h, w)
                is_multi = (y_gt.float().sum(dim=1) >= 2.0).reshape(b, 1, 1)
                weights = torch.where(is_multi, w_mc, torch.ones_like(w_mc))
            return kl_pix, weights

    def forward(
        self,
        logits: torch.Tensor,
        target_soft: torch.Tensor,
        y_gt: Optional[torch.Tensor] = None,
        valid_weight: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            kl_pix, oeem_w = self.compute_kl_and_weights(
                logits=logits,
                target_soft=target_soft,
                y_gt=y_gt,
                epoch=epoch,
            )
            eff_w = oeem_w if valid_weight is None else (oeem_w * valid_weight.float())
            return _masked_mean(kl_pix, eff_w)


class SoftSymmetricCELoss(nn.Module):
    """Symmetric Cross-Entropy (alpha * CE(q, p) + beta * RCE(p, q)) for noisy soft masks."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.15) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(
        self,
        logits: torch.Tensor,
        target_soft: torch.Tensor,
        valid_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            logits_f = logits.float()
            log_p = F.log_softmax(logits_f, dim=1)
            p = torch.softmax(logits_f, dim=1)
            q = target_soft.float().clamp(min=0.0)
            l_ce = -(q * log_p).sum(dim=1)
            l_rce = -(p * torch.log(q.clamp(min=1e-4))).sum(dim=1)
            loss_pix = self.alpha * l_ce + self.beta * l_rce
            return _masked_mean(loss_pix, valid_weight)


class SoftMulticlassDiceLoss(nn.Module):
    """Smooth multi-class soft Dice loss across 4 foreground tissue classes in FP32."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        target_soft: torch.Tensor,
        valid_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            p = torch.softmax(logits.float(), dim=1)
            q = target_soft.float().clamp(min=0.0)
            if valid_weight is not None:
                vw = valid_weight.float().unsqueeze(1)
                inter = (vw * p * q).sum(dim=(0, 2, 3))
                denom = (vw * (p + q)).sum(dim=(0, 2, 3))
            else:
                inter = (p * q).sum(dim=(0, 2, 3))
                denom = p.sum(dim=(0, 2, 3)) + q.sum(dim=(0, 2, 3))
            dice_per_cls = (2.0 * inter + self.eps) / (denom + self.eps)
            return 1.0 - dice_per_cls.mean()


class MILPatchPresenceLoss(nn.Module):
    """Multiple-Instance Learning (MIL) patch-level presence/absence loss in FP32.

    - Absent classes (y_c = 0): masked mean of -mean_{x,y} log(1 - P_c).
    - Present classes (y_c = 1): dynamic top-1% pixel mean p_mil, masked mean of -log(p_mil).
    """

    def forward(self, logits: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            p = torch.softmax(logits.float(), dim=1).clamp(1e-6, 1.0 - 1e-6)
            y = (y_gt.float() > 0.5).float()
            b, c, h, w = p.shape

            # Absent suppression
            absent_mask = 1.0 - y
            absent_nll = -torch.log(1.0 - p).mean(dim=(2, 3))
            l_absent = (absent_nll * absent_mask).sum() / absent_mask.sum().clamp(min=1.0)

            # Present top-1% MIL pooling
            k = max(1, int(round(0.01 * h * w)))
            p_flat = p.reshape(b, c, h * w)
            p_mil = torch.topk(p_flat, k, dim=-1).values.mean(dim=-1)
            present_nll = -torch.log(p_mil.clamp(min=1e-6))
            l_present = (present_nll * y).sum() / y.sum().clamp(min=1.0)

            return l_absent + l_present


class Stage2V2CompositeLoss(nn.Module):
    """Composite Stage-2 v2 training loss combining soft seg, Dice, MIL, and cls BCE."""

    def __init__(self, cfg: Stage2V2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_mode = str(cfg.loss_mode).strip().lower()
        if self.loss_mode == "soft_kd_dice_mil":
            self.seg_loss: nn.Module = SoftCrossEntropyLoss()
        elif self.loss_mode == "kl_oeem_dice_mil":
            self.seg_loss = KLSoftOEEMLoss(
                tau=cfg.oeem_tau,
                oeem_warmup_epochs=cfg.oeem_warmup_epochs,
            )
        elif self.loss_mode == "soft_sce_dice_mil":
            self.seg_loss = SoftSymmetricCELoss(alpha=cfg.sce_alpha, beta=cfg.sce_beta)
        else:
            raise ValueError(f"Unsupported loss_mode: {cfg.loss_mode!r}.")

        self.dice_loss = SoftMulticlassDiceLoss()
        self.mil_loss = MILPatchPresenceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_soft: torch.Tensor,
        y_gt: torch.Tensor,
        valid_weight: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        seg_logits = outputs["seg_logits"]
        cls_logits = outputs["cls_logits"]

        with torch.amp.autocast(device_type=seg_logits.device.type, enabled=False):
            if isinstance(self.seg_loss, KLSoftOEEMLoss):
                l_seg = self.seg_loss(
                    seg_logits,
                    target_soft,
                    y_gt=y_gt,
                    valid_weight=valid_weight,
                    epoch=epoch,
                )
            else:
                l_seg = self.seg_loss(seg_logits, target_soft, valid_weight=valid_weight)

            l_dice = self.dice_loss(seg_logits, target_soft, valid_weight=valid_weight)
            l_mil = self.mil_loss(seg_logits, y_gt)
            l_cls = self.bce_loss(cls_logits.float().flatten(1), y_gt.float().flatten(1))

            total = (
                l_seg
                + float(self.cfg.lambda_dice) * l_dice
                + float(self.cfg.lambda_mil) * l_mil
                + float(self.cfg.lambda_cls) * l_cls
            )

        metrics = {
            "seg": float(l_seg.detach().item()),
            "dice": float(l_dice.detach().item()),
            "mil": float(l_mil.detach().item()),
            "cls": float(l_cls.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, metrics
