"""
Progressive Dropout Attention (PDA) for Weakly-Supervised Histopathology Segmentation (SE374.R11).

Mathematical Formulation:
1. Initial CAM Generation:
   M_c^(0)(x, y) = ReLU(sum_d W_c^d * F_d(x, y))

2. Progressive Multi-Step Attention Expansion (t = 1 ... T):
   - Spatial Erasing:
     tau_c^(t) = Quantile(M_c^(t-1), 1 - p)
     S_c^(t)(x, y) = 1 - delta_t * Indicator(M_c^(t-1)(x, y) >= tau_c^(t))
   - Channel Dropout:
     C_c^(t)(d) = 1 - delta_t * Indicator(W_c^d >= Quantile(W_c, 1 - p))
   - Complementary Feature & CAM Computation:
     F_c^(t) = F * S_c^(t) * C_c^(t)
     M_c^(t) = ReLU(W_c * F_c^(t))
   - Boundary Bleeding Control via Prototype Gating:
     mu_c = sum_(x,y) [M_c^(0)(x, y) * F(x, y)] / (sum M_c^(0) + eps)
     G_c(x, y) = ReLU(CosineSimilarity(F(x, y), mu_c))
     M_c^(t) = M_c^(t) * G_c

3. Aggregation & Normalization:
   hat{M}_c = sum_(t=0)^T w_t * M_c^(t)
   hat{M}_c = (hat{M}_c - min) / (max - min + eps)

Strict adherence to:
- Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Invariant 07: Genuine PyTorch implementation with zero mockup.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveDropoutAttention(nn.Module):
    """Progressive Dropout Attention (PDA) generator for Class Activation Maps.
    
    Prevents CAM from over-concentrating on small discriminative foci (under-activation)
    while preventing boundary bleeding into adjacent fibrous stroma via prototype-guided gating.
    """

    def __init__(
        self,
        num_steps: int = 3,
        top_p: float = 0.25,
        base_dropout_rate: float = 0.50,
        step_weight_decay: float = 0.80,
        boundary_bleeding_control: bool = True,
        spatial_masking: bool = True,
        channel_masking: bool = True,
    ) -> None:
        """Args:
            num_steps: Number of progressive dropout iterations T (default: 3).
            top_p: Fraction of highest-activation regions/channels to mask (default: 0.25).
            base_dropout_rate: Initial dropout attenuation rate delta_0 (default: 0.50).
            step_weight_decay: Decay factor gamma for step weights w_t (default: 0.80).
            boundary_bleeding_control: Whether to constrain complementary CAMs with prototype cosine gating.
            spatial_masking: Whether to apply progressive spatial masking.
            channel_masking: Whether to apply progressive channel dropout.
        """
        super().__init__()
        assert num_steps >= 1, f"num_steps must be at least 1, got {num_steps}"
        assert 0.0 < top_p < 1.0, f"top_p must be between 0 and 1, got {top_p}"
        assert 0.0 <= base_dropout_rate <= 1.0, f"base_dropout_rate must be in [0, 1], got {base_dropout_rate}"

        self.num_steps = num_steps
        self.top_p = top_p
        self.base_dropout_rate = base_dropout_rate
        self.step_weight_decay = step_weight_decay
        self.boundary_bleeding_control = boundary_bleeding_control
        self.spatial_masking = spatial_masking
        self.channel_masking = channel_masking

        # Precompute normalized step weights: w_t = gamma^t / sum(gamma^j)
        raw_weights = [self.step_weight_decay ** t for t in range(self.num_steps + 1)]
        sum_w = sum(raw_weights)
        self.step_weights = [w / sum_w for w in raw_weights]

    def _compute_raw_cam(
        self,
        feature_map: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Computes raw CAM via Conv2d weight projection followed by ReLU.
        
        Args:
            feature_map: (B, D, H, W) feature tensor.
            weight: (C, D, 1, 1) or (C, D) classifier weight tensor.
            
        Returns:
            cam: (B, C, H, W) non-negative activation map.
        """
        if weight.dim() == 2:
            weight = weight.unsqueeze(-1).unsqueeze(-1)
        
        d_feat = feature_map.shape[1]
        d_weight = weight.shape[1]
        if d_feat != d_weight:
            weight = weight[:, :d_feat, :, :]

        cam = F.conv2d(feature_map, weight)
        return F.relu(cam)

    def _compute_spatial_mask(
        self,
        cam_t: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        """Computes spatial retention mask extinguishing top-p percentile activations.
        
        Args:
            cam_t: (B, C, H, W) activation tensor from current/previous step.
            delta_t: Dropout attenuation intensity for this step.
            
        Returns:
            spatial_mask: (B, C, H, W) retention mask in [1 - delta_t, 1.0].
        """
        b, c, h, w = cam_t.shape
        flat_cam = cam_t.view(b, c, -1)  # (B, C, H*W)

        # Compute relative peak and quantile thresholds
        max_vals = flat_cam.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (B, C, 1, 1)
        q = float(1.0 - self.top_p)
        q_thresholds = torch.quantile(flat_cam, q=q, dim=-1, keepdim=True).unsqueeze(-1)  # (B, C, 1, 1)

        # Combine relative peak threshold with quantile to avoid over-erasing sub-dominant tumor bed
        rel_thresholds = 0.55 * max_vals
        thresholds = torch.maximum(q_thresholds, rel_thresholds)
        thresholds = torch.clamp(thresholds, min=1e-6)

        # High activation region indicator: 1.0 where cam >= threshold
        high_activation = (cam_t >= thresholds).float()

        # Retention mask: 1.0 where unmasked, (1.0 - delta_t) where masked
        spatial_mask = 1.0 - delta_t * high_activation
        return spatial_mask

    def _compute_channel_mask(
        self,
        weight: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        """Computes channel-wise retention mask for top-activating classifier weights.
        
        Args:
            weight: (C, D, 1, 1) or (C, D) classification weights.
            delta_t: Attenuation intensity.
            
        Returns:
            channel_mask: (C, D, 1, 1) retention mask.
        """
        if weight.dim() == 4:
            w_flat = weight.squeeze(-1).squeeze(-1)  # (C, D)
        else:
            w_flat = weight  # (C, D)

        q = float(1.0 - self.top_p)
        thresholds = torch.quantile(w_flat, q=q, dim=-1, keepdim=True)  # (C, 1)
        w_min = w_flat.min(dim=-1, keepdim=True)[0]                     # (C, 1)
        w_max = w_flat.max(dim=-1, keepdim=True)[0]                     # (C, 1)

        # Only apply channel masking if weights have variance across channels
        has_variance = (w_max - w_min) > 1e-5                            # (C, 1)
        high_channels = ((w_flat >= thresholds) & has_variance).float() # (C, D)
        channel_mask = 1.0 - delta_t * high_channels
        return channel_mask.unsqueeze(-1).unsqueeze(-1)  # (C, D, 1, 1)

    def _compute_prototype_gating(
        self,
        feature_map: torch.Tensor,
        cam_base: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cosine similarity gating factor between features and initial class prototypes.
        
        Prevents expanded complementary activations from spilling into non-tumor stroma
        by verifying histological feature coherence.
        
        Args:
            feature_map: (B, D, H, W) base deep feature map.
            cam_base: (B, C, H, W) initial baseline CAM.
            
        Returns:
            gating_factor: (B, C, H, W) cosine gating factor in [0, 1].
        """
        b, d, h, w = feature_map.shape
        _, c, _, _ = cam_base.shape

        # Reshape features to (B, D, N)
        feat_flat = feature_map.view(b, d, -1)  # (B, D, N)
        cam_flat = cam_base.view(b, c, -1)      # (B, C, N)

        # Compute weighted prototype vector mu_c = sum(cam * feat) / sum(cam)
        # (B, C, N) x (B, N, D) = (B, C, D)
        cam_sum = cam_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, C, 1)
        prototypes = torch.bmm(cam_flat, feat_flat.transpose(1, 2)) / cam_sum  # (B, C, D)

        # Normalize prototypes and features for cosine similarity
        proto_norm = F.normalize(prototypes, p=2, dim=-1)  # (B, C, D)
        feat_norm = F.normalize(feat_flat, p=2, dim=1)     # (B, D, N)

        # Cosine similarity: (B, C, D) x (B, D, N) = (B, C, N)
        cos_sim = torch.bmm(proto_norm, feat_norm).view(b, c, h, w)

        # ReLU gating: only positive semantic correlation to prototype is allowed
        gating_factor = F.relu(cos_sim)
        return gating_factor

    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """Min-max normalizes CAM to [0, 1] per sample and class."""
        b, c, h, w = cam.shape
        flat = cam.view(b, c, -1)
        min_val = flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        max_val = flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        range_val = torch.clamp(max_val - min_val, min=1e-7)
        return (cam - min_val) / range_val

    def forward(
        self,
        feature_map: torch.Tensor,
        classifier_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass generating Progressive Dropout Attention CAM.
        
        Args:
            feature_map: (B, D, H, W) deep feature tensor (e.g. layer bn7).
            classifier_weight: (C, D, 1, 1) or (C, D) classifier weight tensor.
            
        Returns:
            pda_cam: (B, C, H, W) normalized expanded Class Activation Map.
        """
        cams = self.generate_cam(
            feature_map=feature_map,
            classifier_weight=classifier_weight,
            return_intermediates=False,
        )
        assert isinstance(cams, torch.Tensor)
        return cams

    def generate_cam(
        self,
        feature_map: torch.Tensor,
        classifier_weight: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Generates progressive dropout attention CAM with iterative expansion.
        
        Args:
            feature_map: (B, D, H, W) feature map.
            classifier_weight: (C, D, 1, 1) or (C, D) classification weights.
            return_intermediates: If True, returns (aggregated_cam, list_of_step_cams).
            
        Returns:
            aggregated_cam: (B, C, H, W) CAM normalized per class.
            intermediates (optional): List of (num_steps + 1) CAM tensors.
        """
        if classifier_weight.dim() == 2:
            weight = classifier_weight.unsqueeze(-1).unsqueeze(-1)
        else:
            weight = classifier_weight

        b, d, h, w = feature_map.shape
        c = weight.shape[0]

        # Step 0: Compute initial baseline CAM
        cam_0 = self._compute_raw_cam(feature_map, weight)
        cam_0_norm = self._normalize_cam(cam_0)
        step_cams: List[torch.Tensor] = [cam_0_norm]

        # Prototype gating for boundary bleeding prevention
        if self.boundary_bleeding_control:
            proto_gating = self._compute_prototype_gating(feature_map, cam_0)
        else:
            proto_gating = torch.ones((b, c, h, w), device=feature_map.device, dtype=feature_map.dtype)

        # Iterative steps t = 1 ... T
        prev_cam = cam_0_norm
        for t in range(1, self.num_steps + 1):
            # Progressive dropout rate increases with step depth
            delta_t = min(self.base_dropout_rate * (1.0 + 0.20 * (t - 1)), 1.0)

            # 1. Spatial Masking based on active regions of previous step
            if self.spatial_masking:
                spatial_mask = self._compute_spatial_mask(prev_cam, delta_t=delta_t)
            else:
                spatial_mask = torch.ones((b, c, h, w), device=feature_map.device, dtype=feature_map.dtype)

            # 2. Channel Masking
            if self.channel_masking:
                channel_mask = self._compute_channel_mask(weight, delta_t=delta_t)
            else:
                channel_mask = torch.ones_like(weight)

            # 3. Compute Complementary CAM per class
            step_cam_classes = []
            for class_idx in range(c):
                s_mask_c = spatial_mask[:, class_idx : class_idx + 1, :, :]  # (B, 1, H, W)
                c_mask_c = channel_mask[class_idx : class_idx + 1, :, :, :]  # (1, D, 1, 1)
                w_c = weight[class_idx : class_idx + 1, :, :, :]             # (1, D, 1, 1)

                masked_feat = feature_map * s_mask_c * c_mask_c
                comp_cam_c = F.relu(F.conv2d(masked_feat, w_c))  # (B, 1, H, W)

                # 4. Apply boundary bleeding control
                if self.boundary_bleeding_control:
                    g_c = proto_gating[:, class_idx : class_idx + 1, :, :]
                    comp_cam_c = comp_cam_c * g_c

                step_cam_classes.append(comp_cam_c)

            cam_t = torch.cat(step_cam_classes, dim=1)  # (B, C, H, W)
            cam_t_norm = self._normalize_cam(cam_t)
            step_cams.append(cam_t_norm)
            prev_cam = cam_t_norm

        # Weighted aggregation: hat_M = sum_t (w_t * M^(t))
        agg_cam = torch.zeros_like(cam_0_norm)
        for t, cam_step in enumerate(step_cams):
            w_t = self.step_weights[t]
            agg_cam = agg_cam + w_t * cam_step

        # Final min-max normalization to [0, 1]
        norm_agg_cam = self._normalize_cam(agg_cam)

        if return_intermediates:
            return norm_agg_cam, step_cams
        return norm_agg_cam

    def generate_from_backbone(
        self,
        backbone: nn.Module,
        x: torch.Tensor,
        feature_layer: str = "bn7",
    ) -> torch.Tensor:
        """Extracts features from backbone and generates Progressive Dropout Attention CAM.
        
        Args:
            backbone: Feature extractor module (e.g. MultiStageFeatureBackbone).
            x: Input image tensor (B, 3, H, W).
            feature_layer: Feature layer key ('bn7', 'b5_2', 'b4_5').
            
        Returns:
            pda_cam: (B, C, H, W) normalized CAM tensor.
        """
        _, features = backbone(x)
        feat = features[feature_layer]
        classifier_weight = backbone.fc_head.weight if hasattr(backbone, "fc_head") else backbone.get_classifier_weights()
        return self.forward(feat, classifier_weight)
