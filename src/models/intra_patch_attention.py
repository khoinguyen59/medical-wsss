"""
Intra-Patch Dual-Branch Multi-Field-of-View Attention Module (SE374.R11).
Provides hierarchical cross-attention between Macro-Context (downsampled patch)
and Micro-Detail (full-resolution patch) without requiring external .svs gigapixel files:
- Macro Branch: Downsamples patch to capture global tumor-stroma architecture
- Micro Branch: High-resolution local cellular morphology and nuclear pleomorphism
- Cross-Attention Spatial Modulation: Guides fine CAM boundaries with macro context priors
- Residual Identity Initialization (gamma = 0) to ensure zero perturbation during baseline testing
- Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraPatchDualBranchAttention(nn.Module):
    """Intra-Patch Multi-Field-of-View Cross-Attention Network.

    Implements zero-download multi-scale learning directly on existing 512x512 patches.
    Avoids multi-gigabyte .svs downloads and TCGA coordinate re-projection errors.
    Uses learnable residual scalar gamma initialized to 0.0 for safe pass-through.
    """

    def __init__(
        self,
        in_channels: int = 512,
        macro_size: Tuple[int, int] = (16, 16),
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.macro_size = macro_size

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        num_groups = next((g for g in (8, 4, 2, 1) if in_channels % g == 0), 1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Residual gating scale initialized to 0.0 for deterministic identity pass-through
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, micro_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs hierarchical cross-attention between micro features and downsampled macro features.

        Args:
            micro_feats: (B, C, H, W) high-resolution micro feature map.

        Returns:
            enhanced_feats: (B, C, H, W) contextualized feature map.
            spatial_gate: (B, 1, H, W) learned macro-to-micro guidance gate.
        """
        b, c, h, w = micro_feats.shape

        # Downsample to extract macro-context representation
        macro_feats = F.adaptive_avg_pool2d(micro_feats, self.macro_size)  # (B, C, Hm, Wm)

        # Micro features serve as Query; Macro features serve as Key and Value
        q = self.query_conv(micro_feats).view(b, c, h * w).permute(0, 2, 1)        # (B, H*W, C)
        k = self.key_conv(macro_feats).view(b, c, -1).permute(0, 2, 1)            # (B, Hm*Wm, C)
        v = self.value_conv(macro_feats).view(b, c, -1).permute(0, 2, 1)          # (B, Hm*Wm, C)

        attn_out, _ = self.mha(query=q, key=k, value=v)                            # (B, H*W, C)
        attn_map = attn_out.permute(0, 2, 1).view(b, c, h, w)                      # (B, C, H, W)

        # Spatial gating modulation: combines macro context with fine features
        concat_feats = torch.cat([micro_feats, attn_map], dim=1)
        spatial_gate = self.gate_conv(concat_feats)                                # (B, 1, H, W)

        # When gamma == 0, enhanced is strictly equal to micro_feats (perfect residual identity)
        enhanced = micro_feats + self.gamma * (spatial_gate * attn_map)

        return enhanced, spatial_gate
