"""
Stain-Guided Deep Affinity Propagation (SG-DAP) for Histopathological Pseudo-Mask Refinement.
Pure-PyTorch GPU-accelerated replacement for DenseCRF (SE374.R11):
- Optical density Beer-Lambert stain weighting
- Deep feature semantic affinity combined with H&E morphological kernel
- Anchored Markov Random Walk with semantic anchor preservation
- Dual-mode boundary uncertainty gating via return_ignore_mask parameter
- Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.stain_deconv import StainDeconvLayer


class StainGuidedAffinityRefiner(nn.Module):
    """Stain-Guided Deep Affinity Propagation (SG-DAP) Module.

    Replaces heuristic CPU-bound DenseCRF with a 100% GPU-accelerated, biologically
    grounded affinity propagation kernel tailored for Whole Slide Image patches.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        alpha_rw: float = 0.85,
        beta: float = 2.0,
        num_iters: int = 10,
        sigma_stain: float = 0.35,
        conf_ignore_thresh: float = 0.55,
        margin_ignore_thresh: float = 0.15,
        ignore_index: int = 255,
        use_deep_affinity: bool = False,
    ) -> None:
        super().__init__()
        self.ks = kernel_size
        self.pad = kernel_size // 2
        self.alpha = alpha_rw
        self.beta = beta
        self.num_iters = num_iters
        self.sigma_stain = sigma_stain
        self.conf_ignore_thresh = conf_ignore_thresh
        self.margin_ignore_thresh = margin_ignore_thresh
        self.ignore_index = ignore_index
        self.use_deep_affinity = bool(use_deep_affinity)

        self.stain_extractor = StainDeconvLayer()

    def forward(
        self,
        rgb_img: torch.Tensor,
        deep_feats: Optional[torch.Tensor],
        init_cam: torch.Tensor,
        return_ignore_mask: bool = False,
        active_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs stain-guided affinity diffusion and boundary uncertainty gating.

        Args:
            rgb_img: (B, 3, H, W) normalized input image in [0.0, 1.0].
            deep_feats: Optional (B, D, h, w) high-level feature activations from backbone.
                        Bypassed when use_deep_affinity is False or deep_feats is None.
            init_cam: (B, C, h, w) initial class activation probability maps.
            return_ignore_mask: If False (evaluation/test mode), returns pure discrete
                                argmax mask strictly in {0, ..., C-1}. If True (training
                                pseudo-mask generation), marks boundary uncertainty with 255.
            active_labels: Optional (B, C) binary indicator of classes present in each patch.
                           When provided, absent channels (<= 0.5) are zeroed before simplex
                           normalization, excluded from uniform fallback prior, kept strictly
                           0.0 across all random-walk iterations, and masked to -1.0 before
                           argmax so 0% absent-class pixels leak.

        Returns:
            refined_cam: (B, C, h, w) diffused probability maps.
            discrete_mask: (B, h, w) discrete segmentation mask.
        """
        b, c, h, w = init_cam.shape

        # Optional ground-truth active-class mask (prevents absent-class uniform-prior leak)
        active_mask: Optional[torch.Tensor] = None
        if active_labels is not None:
            active_bool = (active_labels.to(device=init_cam.device) > 0.5).view(b, c, 1, 1)
            has_any_active = active_bool.any(dim=1, keepdim=True)
            effective_active = torch.where(has_any_active, active_bool, torch.ones_like(active_bool))
            active_mask = effective_active.to(dtype=torch.float32)

        # Normalize init_cam safely into a valid pixel-wise probability simplex
        cam_pos = torch.clamp(init_cam.float(), min=0.0)
        if active_mask is not None:
            cam_pos = cam_pos * active_mask
        cam_sum = cam_pos.sum(dim=1, keepdim=True)
        if active_mask is not None:
            active_count = active_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            uniform_prior = (active_mask / active_count).expand_as(cam_pos)
        else:
            uniform_prior = torch.full_like(cam_pos, 1.0 / float(max(c, 1)))
        init_probs = torch.where(
            cam_sum > 1e-6,
            cam_pos / cam_sum.clamp(min=1e-6),
            uniform_prior,
        )

        # Resize RGB image and deep features to match CAM spatial resolution if needed
        if rgb_img.shape[-2:] != (h, w):
            rgb_small = F.interpolate(rgb_img, size=(h, w), mode="bilinear", align_corners=False)
        else:
            rgb_small = rgb_img

        # Extract H&E optical density and chromatin dispersion features
        _, _, stain_features = self.stain_extractor(rgb_small)        # (B, 6, h, w)

        k2 = self.ks * self.ks
        stain_unfold = F.unfold(stain_features, kernel_size=self.ks, padding=self.pad).view(b, 6, k2, h * w)
        stain_center = stain_features.view(b, 6, 1, h * w)
        dist_stain = ((stain_center - stain_unfold) ** 2).sum(dim=1)      # (B, K2, h*w)
        sim_stain = torch.exp(-dist_stain / (2.0 * (self.sigma_stain ** 2)))

        if self.use_deep_affinity and deep_feats is not None:
            # Memory-safe per-sample k x k deep-feature cosine similarity (avoids huge B*D*H*W allocations)
            sim_deep = torch.empty((b, k2, h * w), device=init_cam.device, dtype=torch.float32)
            for bi in range(b):
                feat_i = deep_feats[bi : bi + 1]
                if feat_i.shape[-2:] != (h, w):
                    feat_i = F.interpolate(feat_i, size=(h, w), mode="bilinear", align_corners=False)
                norm_i = F.normalize(feat_i.float(), dim=1, eps=1e-6)  # (1, D, h, w)
                idx_k = 0
                for dy in range(-self.pad, self.pad + 1):
                    src_y0 = max(0, dy)
                    src_y1 = h + min(0, dy)
                    dst_y0 = max(0, -dy)
                    dst_y1 = h - max(0, dy)
                    for dx in range(-self.pad, self.pad + 1):
                        src_x0 = max(0, dx)
                        src_x1 = w + min(0, dx)
                        dst_x0 = max(0, -dx)
                        dst_x1 = w - max(0, dx)
                        dot_map = torch.zeros((h, w), device=init_cam.device, dtype=torch.float32)
                        dot_valid = (
                            norm_i[0, :, dst_y0:dst_y1, dst_x0:dst_x1]
                            * norm_i[0, :, src_y0:src_y1, src_x0:src_x1]
                        ).sum(dim=0).clamp(min=0.0)
                        dot_map[dst_y0:dst_y1, dst_x0:dst_x1] = dot_valid
                        sim_deep[bi, idx_k] = dot_map.reshape(-1)
                        idx_k += 1

            # Hybrid affinity matrix with row-stochastic center self-loop fallback when row_sum == 0
            affinity = (sim_deep * sim_stain) ** self.beta
        else:
            # Pure H&E stain-guided affinity (sim_deep = 1.0, skips 25-shift 4096-D upsampling)
            affinity = sim_stain ** self.beta
        row_sum = affinity.sum(dim=1, keepdim=True)
        identity_loop = torch.zeros_like(affinity)
        identity_loop[:, k2 // 2, :] = 1.0
        trans_matrix = torch.where(
            row_sum > 1e-6,
            affinity / row_sum.clamp(min=1e-6),
            identity_loop,
        )  # (B, K2, h*w)

        # Anchored Markov Random Walk diffusion
        cam_curr = init_probs
        for _ in range(self.num_iters):
            cam_unfold = F.unfold(cam_curr, kernel_size=self.ks, padding=self.pad).view(b, c, k2, h * w)
            cam_prop = (cam_unfold * trans_matrix.unsqueeze(1)).sum(dim=2).view(b, c, h, w)
            cam_curr = self.alpha * cam_prop + (1.0 - self.alpha) * init_probs
            if active_mask is not None:
                cam_curr = cam_curr * active_mask
            cam_curr = cam_curr / cam_curr.sum(dim=1, keepdim=True).clamp(min=1e-6)

        if active_mask is not None:
            cam_for_argmax = torch.where(active_mask > 0.5, cam_curr, torch.full_like(cam_curr, -1.0))
        else:
            cam_for_argmax = cam_curr
        discrete_mask = torch.argmax(cam_for_argmax, dim=1)

        # Boundary uncertainty masking: only apply when explicitly requested for training
        if return_ignore_mask:
            sorted_probs, _ = torch.sort(cam_curr, dim=1, descending=True)
            top1_prob = sorted_probs[:, 0, :, :]
            top2_prob = sorted_probs[:, 1, :, :] if c > 1 else torch.zeros_like(top1_prob)
            margin = top1_prob - top2_prob

            uncertain_mask = (top1_prob < self.conf_ignore_thresh) | (margin < self.margin_ignore_thresh)
            discrete_mask = discrete_mask.clone()
            discrete_mask[uncertain_mask] = self.ignore_index

        return cam_curr, discrete_mask
