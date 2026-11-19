"""
Optical Density Stain Deconvolution and Histopathological Chromatin Entropy Service.
Provides mathematically grounded, differentiable H&E deconvolution (SE374.R11):
- Beer-Lambert Optical Density (OD) transformation
- Normalized Ruifrok & Johnston stain absorption matrix inversion
- Fano Factor Chromatin Dispersion index (D_H = var_H / (mean_H + eps)) to distinguish
  Karyorrhectic Necrosis from Monomorphic Inflammatory Lymphocytic infiltration.
- Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StainDeconvLayer(nn.Module):
    """Beer-Lambert Optical Density H&E Stain Deconvolution Layer.

    Decouples raw RGB microphotographs into orthogonal Hematoxylin (nuclear chromatin)
    and Eosin (cytoplasmic/stromal extracellular matrix) optical density channels.
    Purely algebraic and fully differentiable on GPU with zero learned weights.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

        # Standard Ruifrok & Johnston optical density absorption matrix (rows = unit stain vectors):
        # Row 0: Hematoxylin [0.650, 0.704, 0.286]
        # Row 1: Eosin       [0.072, 0.990, 0.105]
        # Row 2: Residual    [0.268, 0.570, 0.776] (Background/DAB)
        raw_matrix = torch.tensor(
            [
                [0.650, 0.704, 0.286],
                [0.072, 0.990, 0.105],
                [0.268, 0.570, 0.776],
            ],
            dtype=torch.float32,
        )
        he_matrix = F.normalize(raw_matrix, p=2, dim=1)
        # Column-vector Beer-Lambert model: od_col = M^T @ s_col => s_col = (M^T)^(-1) @ od_col
        he_inv = torch.linalg.inv(he_matrix.t()).contiguous()
        # Register as persistent buffers for GPU/CPU device consistency and inverse verification
        self.register_buffer("he_matrix", he_matrix)
        self.register_buffer("he_inv", he_inv.view(3, 3, 1, 1))

        # Sobel kernels for optical stain boundary gradient computation
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32) / 8.0
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def compute_optical_density(self, rgb_img: torch.Tensor) -> torch.Tensor:
        """Transforms normalized RGB [0, 1] (or ImageNet-standardized RGB) into Optical Density (OD) space.

        Args:
            rgb_img: (B, 3, H, W) tensor in [0.0, 1.0] or ImageNet z-score space.

        Returns:
            od: (B, 3, H, W) optical density tensor.
        """
        img_01 = rgb_img
        if float(img_01.min().detach()) < -0.05 or float(img_01.max().detach()) > 1.05:
            if float(img_01.max().detach()) > 5.0 and float(img_01.min().detach()) >= 0.0:
                # Rescale [0, 255] uint8-range floats into [0, 1]
                img_01 = img_01 / 255.0
            else:
                # Reverse standard ImageNet normalization cleanly
                mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_img.device, dtype=rgb_img.dtype).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=rgb_img.device, dtype=rgb_img.dtype).view(1, 3, 1, 1)
                img_01 = img_01 * std + mean
        img_clamped = torch.clamp(img_01, min=1e-3, max=1.0)
        return -torch.log10(img_clamped)

    def compute_stain_gradient(self, he_channels: torch.Tensor) -> torch.Tensor:
        """Computes spatial Sobel optical density gradient magnitude across H&E channels.

        Args:
            he_channels: (B, 2, H, W) Hematoxylin and Eosin concentration channels.

        Returns:
            grad_mag: (B, 1, H, W) optical stain boundary gradient magnitude >= 0.
        """
        b, c, h, w = he_channels.shape
        flat = he_channels.reshape(b * c, 1, h, w)
        gx = F.conv2d(flat, self.sobel_x.to(dtype=flat.dtype), padding=1)
        gy = F.conv2d(flat, self.sobel_y.to(dtype=flat.dtype), padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + self.eps).reshape(b, c, h, w)
        return mag.mean(dim=1, keepdim=True)

    def compute_fano_factor(
        self,
        h_channel: torch.Tensor,
        kernel_size: int = 7,
    ) -> torch.Tensor:
        """Computes the Fano Factor (Index of Dispersion) of nuclear chromatin:

            D_H(i) = sigma_H^2(i) / (mu_H(i) + eps)

        Biological rationale:
        - Inflammatory (Lymphocytes): Uniform, hyperchromatic spherical nuclei ->
          high mu_H, periodic low sigma_H^2 -> stable, bounded D_H.
        - Necrosis: Karyorrhexis/Karyolysis with fragmented nuclear debris on eosinophilic
          background -> high spatial variance sigma_H^2 relative to low mu_H -> erratic D_H.

        Args:
            h_channel: (B, 1, H, W) Hematoxylin optical density channel.
            kernel_size: Sliding window size for local neighborhood statistics.

        Returns:
            fano_factor: (B, 1, H, W) local dispersion tensor.
        """
        pad = kernel_size // 2
        mu_h = torch.clamp(
            F.avg_pool2d(h_channel, kernel_size=kernel_size, stride=1, padding=pad),
            min=0.0,
        )
        mu_h_sq = F.avg_pool2d(h_channel ** 2, kernel_size=kernel_size, stride=1, padding=pad)
        var_h = torch.clamp(mu_h_sq - mu_h ** 2, min=0.0)
        return var_h / (mu_h + self.eps)

    def forward(self, rgb_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs optical density deconvolution and morphological feature extraction.

        Args:
            rgb_img: (B, 3, H, W) normalized image tensor in [0.0, 1.0].

        Returns:
            stain_channels: (B, 2, H, W) tensor containing [H_channel, E_channel].
            fano_factor: (B, 1, H, W) nuclear chromatin dispersion index D_H.
            stain_features: (B, 6, H, W) concatenated morphological descriptor:
                [H_mean, E_mean, H_std, E_std, H_raw, E_raw].
        """
        od = self.compute_optical_density(rgb_img)
        # Physical stain concentrations cannot be negative (Beer-Lambert absorbance S >= 0)
        stains = torch.clamp(F.conv2d(od, self.he_inv), min=0.0)
        h_channel = stains[:, 0:1, :, :]
        e_channel = stains[:, 1:2, :, :]
        he_channels = torch.cat([h_channel, e_channel], dim=1)  # (B, 2, H, W)

        # Morphological descriptors via 5x5 spatial pooling
        mean_local = F.avg_pool2d(he_channels, kernel_size=5, stride=1, padding=2)
        sq_mean_local = F.avg_pool2d(he_channels ** 2, kernel_size=5, stride=1, padding=2)
        std_local = torch.sqrt(torch.clamp(sq_mean_local - mean_local ** 2, min=self.eps))

        stain_features = torch.cat([mean_local, std_local, he_channels], dim=1)  # (B, 6, H, W)
        fano_factor = self.compute_fano_factor(h_channel, kernel_size=7)         # (B, 1, H, W)

        return he_channels, fano_factor, stain_features
