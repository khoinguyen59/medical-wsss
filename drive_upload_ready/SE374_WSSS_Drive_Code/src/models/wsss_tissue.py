"""
Weakly-Supervised Histopathology Semantic Segmentation Engine (SE374.R11).

Implements the 4 Comparative Benchmarking Configurations:
1. Configuration 1: Vanilla CAM Baseline (Han et al., MedIA 2022)
2. Configuration 2: Data-Centric (+ ClassMix augmentation)
3. Configuration 3: Spatial Affinity Relocalization (ARML-inspired, MICCAI 2024)
4. Configuration 4: Proposed Adaptive Pipeline (Otsu Gating + PDA + Dynamic Progressive Thresholding + Uncertainty Regularization)

Strict adherence to:
- Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Invariant 07: Genuine PyTorch implementation with zero mockup.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.otsu_filter import apply_otsu_gating_to_prediction
from src.models.backbone import (
    HistopathologyClassifierBackbone,
    MultiStageFeatureBackbone,
    ResidualConvBlock,
    SimpleConvBlock,
)
from src.models.pda_generator import ProgressiveDropoutAttention


class SpatialAffinityRelocalization(nn.Module):
    """Configuration 3: Spatial Affinity Matrix Relocalization (ARML-inspired, MICCAI 2024).
    
    Diffuses CAM energy across spatially correlated feature tokens to eliminate under-activation.
    Formula:
        A_ij = exp(f_i^T f_j / sqrt(d)) / sum_k(exp(f_i^T f_k / sqrt(d)))
        M_refined = alpha * M + (1 - alpha) * A * M
    """

    def __init__(self, alpha: float = 0.70) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, cam: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """Args:
            cam: (B, C, H, W) Class activation maps.
            feature_map: (B, D, H, W) High-level deep feature map.
            
        Returns:
            relocalized_cam: (B, C, H, W) Energy-diffused activation maps.
        """
        b, c, h, w = cam.shape
        _, d, h_f, w_f = feature_map.shape

        # Memory safety: if spatial resolution is large, diffuse at bottleneck resolution (< 1 MB)
        if (h, w) != (h_f, w_f):
            if max(h, w) > 32 and max(h_f, w_f) <= 32:
                cam_low = F.interpolate(cam, size=(h_f, w_f), mode="bilinear", align_corners=False)
                cam_tokens = cam_low.flatten(2)
                tokens = feature_map.flatten(2).permute(0, 2, 1)
                scale = 1.0 / np.sqrt(d)
                attn = torch.bmm(tokens, tokens.transpose(1, 2)) * scale
                attn = F.softmax(attn, dim=-1)
                diffused_tokens = torch.bmm(cam_tokens, attn.transpose(1, 2))
                diffused_cam_low = diffused_tokens.view(b, c, h_f, w_f)
                diffused_cam = F.interpolate(diffused_cam_low, size=(h, w), mode="bilinear", align_corners=False)
                return self.alpha * cam + (1.0 - self.alpha) * diffused_cam
            else:
                feature_map = F.interpolate(feature_map, size=(h, w), mode="bilinear", align_corners=False)

        # Reshape to token matrices
        tokens = feature_map.flatten(2).permute(0, 2, 1)  # (B, N, D), N = H*W
        cam_tokens = cam.flatten(2)  # (B, C, N)

        # Normalized cosine affinity: A = softmax(Q K^T / sqrt(D))
        scale = 1.0 / np.sqrt(d)
        attn = torch.bmm(tokens, tokens.transpose(1, 2)) * scale  # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        # Diffuse CAM: (B, C, N) x (B, N, N)^T = (B, C, N)
        diffused_tokens = torch.bmm(cam_tokens, attn.transpose(1, 2))
        diffused_cam = diffused_tokens.view(b, c, h, w)

        refined = self.alpha * cam + (1.0 - self.alpha) * diffused_cam
        return refined


def apply_classmix_augmentation(
    images: torch.Tensor,
    labels: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Configuration 2: ClassMix Data Augmentation for Histopathology.
    
    Synthesizes sharp tissue boundaries between two patches in the batch:
        x_mixed = B_c * x_A + (1 - B_c) * x_B
        y_mixed = y_A | y_B (logical OR for multi-label classification)
    """
    if torch.rand(1).item() > p or images.size(0) < 2:
        return images, labels, masks

    batch_size = images.size(0)
    perm = torch.randperm(batch_size)

    # Pick a random class mask region if masks are provided, else spatial half-split
    if masks is not None:
        target_class = torch.randint(0, 4, (1,)).item()
        binary_mask = (masks == target_class).unsqueeze(1).float()  # (B, 1, H, W)
    else:
        # Spatial half mask when masks are not yet available (Stage 1)
        h, w = images.shape[-2:]
        binary_mask = torch.zeros((batch_size, 1, h, w), device=images.device)
        split_h = torch.randint(h // 4, 3 * h // 4, (1,)).item()
        binary_mask[:, :, :split_h, :] = 1.0

    mixed_images = binary_mask * images + (1.0 - binary_mask) * images[perm]
    # Multi-label logical OR
    mixed_labels = torch.clamp(labels + labels[perm], max=1.0)

    mixed_masks = None
    if masks is not None:
        mixed_masks = (binary_mask.squeeze(1) * masks + (1.0 - binary_mask.squeeze(1)) * masks[perm]).long()

    return mixed_images, mixed_labels, mixed_masks


class WSSSBenchmarkingEngine:
    """Unified Orchestration Engine supporting all 4 Comparative Benchmarking Configurations.
    
    Configuration 1: Baseline CAM (Han et al., MedIA 2022) with static threshold 0.20
    Configuration 2: Data-Centric (+ ClassMix augmentation)
    Configuration 3: Spatial Affinity Relocalization (ARML-inspired) + dynamic threshold
    Configuration 4: Proposed Adaptive Pipeline (PDA + Relocalization + Dynamic Progressive Thresholding + Uncertainty Regularization + Otsu Gating)
    """

    def __init__(
        self,
        mode: str = "proposed",
        num_classes: int = 5,
        device: str = "cpu",
    ) -> None:
        assert mode in ["baseline", "classmix", "relocalization", "proposed"], f"Invalid mode: {mode}"
        self.mode = mode
        self.num_classes = num_classes
        self.device = torch.device(device)

        # Multi-stage feature backbone (num_classes - 1 foreground classes)
        self.classifier = MultiStageFeatureBackbone(in_channels=3, num_classes=num_classes - 1).to(self.device)

        # Progressive Dropout Attention generator
        self.pda = ProgressiveDropoutAttention(
            num_steps=3,
            top_p=0.25,
            base_dropout_rate=0.50,
            step_weight_decay=0.80,
            boundary_bleeding_control=True,
        ).to(self.device)

        # Spatial affinity relocalizer
        self.relocalizer = SpatialAffinityRelocalization(alpha=0.70).to(self.device)

        # Uncertainty map tracker for clinical transparency
        self.last_uncertainty_map: Optional[np.ndarray] = None

    def generate_pseudo_mask(
        self,
        image_tensor: torch.Tensor,
        current_epoch: int = 1,
        raw_image: Optional[np.ndarray] = None,
        return_uncertainty: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generates pseudo semantic segmentation mask according to selected benchmark configuration.
        
        Args:
            image_tensor: (1, 3, H, W) normalized tensor in [0, 1].
            current_epoch: Training epoch for dynamic progressive thresholding.
            raw_image: Original RGB numpy image (H, W, 3) for Otsu gating.
            return_uncertainty: If True, returns (pred_mask, uncertainty_map).
            
        Returns:
            pseudo_mask: (H, W) array with class indices 0..(num_classes - 1).
        """
        self.classifier.eval()
        with torch.no_grad():
            img = image_tensor.to(self.device)
            logits, features = self.classifier(img)
            h_orig, w_orig = image_tensor.shape[-2:]

            if self.mode == "proposed":
                # Step 1: Progressive Dropout Attention (PDA) CAM generation
                # Operates on deep bn7 feature map with progressive spatial & channel erasing
                cam_feat = self.pda(features["bn7"], self.classifier.fc_head.weight)

                # Step 2: Spatial Affinity Relocalization at feature token resolution BEFORE upsampling
                cam_feat = self.relocalizer(cam_feat, features["bn7"])
                cam_upsampled = F.interpolate(cam_feat, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
            else:
                # Standard Vanilla CAM from bn7 Conv projection at feature token resolution
                cam_feat = self.classifier.compute_cam(img, feature_layer="bn7")

                if self.mode == "relocalization":
                    # Spatial Affinity Relocalization at feature token resolution BEFORE upsampling
                    cam_feat = self.relocalizer(cam_feat, features["bn7"])

                cam_upsampled = F.interpolate(cam_feat, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

            # Step 3: Per-class min-max normalization
            cam_np = cam_upsampled.squeeze(0).cpu().numpy()  # (C_fg, H, W)
            norm_cam = np.zeros_like(cam_np)
            for c in range(cam_np.shape[0]):
                c_min, c_max = cam_np[c].min(), cam_np[c].max()
                if c_max > c_min:
                    norm_cam[c] = (cam_np[c] - c_min) / (c_max - c_min)
                else:
                    norm_cam[c] = 0.0

            # Step 4: Thresholding Mechanism
            if self.mode in ["baseline", "classmix"]:
                # Static baseline threshold: theta = 0.20
                effective_thr = 0.20
            else:
                # Dynamic Progressive Thresholding:
                # mu_t = max(0.90 * (0.98 ** max(0, current_epoch - 2)), 0.65)
                # theta_t = mu_t * 0.35
                mu_t = max(0.90 * (0.98 ** max(0, current_epoch - 2)), 0.65)
                effective_thr = float(mu_t * 0.35)

            # Step 5: Background slice concatenation
            bg_score = np.full((1, h_orig, w_orig), fill_value=effective_thr, dtype=np.float32)
            all_scores = np.concatenate([norm_cam, bg_score], axis=0)  # (num_classes, H, W)

            # Step 6: Uncertainty Regularization & Shannon Entropy Computation
            # P_c(x, y) = softmax(all_scores)
            scores_tensor = torch.from_numpy(all_scores).unsqueeze(0)  # (1, num_classes, H, W)
            probs = F.softmax(scores_tensor * 5.0, dim=1)              # Scaled softmax for sharper probability
            log_probs = torch.log(probs.clamp(min=1e-7))
            num_cls = float(all_scores.shape[0])
            norm_entropy = -torch.sum(probs * log_probs, dim=1) / (np.log(num_cls) + 1e-7)  # (1, H, W)
            uncertainty_map = norm_entropy.squeeze(0).cpu().numpy()                         # (H, W) in [0, 1]
            self.last_uncertainty_map = uncertainty_map

            # Argmax prediction
            pred_mask = np.argmax(all_scores, axis=0).astype(np.int64)

            # Step 7: Apply Uncertainty Regularization on ambiguous boundary pixels
            if self.mode == "proposed":
                # For high-uncertainty boundary pixels (entropy > 0.85) where top-1 and top-2 scores
                # are within margin 0.05, regularize prediction to prevent boundary bleeding
                sorted_scores = np.sort(all_scores, axis=0)
                margin = sorted_scores[-1, :, :] - sorted_scores[-2, :, :]
                boundary_ambiguous = (uncertainty_map > 0.85) & (margin < 0.05)
                # If ambiguous between foreground tissue and background, favor background gating
                bg_class_idx = self.num_classes - 1
                pred_mask[boundary_ambiguous & (sorted_scores[-1, :, :] < effective_thr * 1.2)] = bg_class_idx

            # Step 8: Apply Clinical Otsu Gating to extinguish empty glass false positives
            if self.mode == "proposed" and raw_image is not None:
                pred_mask = apply_otsu_gating_to_prediction(
                    pred_mask=pred_mask,
                    image=raw_image,
                    background_class_idx=self.num_classes - 1,
                )

            if return_uncertainty:
                return pred_mask, uncertainty_map
            return pred_mask
