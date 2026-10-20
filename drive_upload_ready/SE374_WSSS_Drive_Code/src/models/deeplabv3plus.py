"""
DeepLabV3+ with Classification Gate Mechanism (CGM) for SE374.R11 WSSS.

This module implements the DeepLabV3PlusWithCGM architecture for weakly-supervised
histopathology semantic segmentation on the BCSS dataset.

Key Architectural Components:
1. Feature Backbone: Multi-stage residual network extracting low-level and high-level representations.
2. Atrous Spatial Pyramid Pooling (ASPP): Multi-scale context aggregation with dilation rates [1, 6, 12, 18],
   batch normalization, and global image pooling.
3. Decoder: Fuses low-level spatial details and upsampled ASPP representations.
4. Classification Gate Mechanism (CGM): Gates segmentation logits using image-level class presence vectors
   y in {0, 1}^C:
       Z_hat_c = Z_c * y_c + (1 - y_c) * penalty_value (-1e4)
   Guarantees zero false-positive segmentations for absent tissue types.
5. Multi-Loss Training Support: CrossEntropyLoss, MultiLayerPseudoSupervisionLoss (MLPS),
   and UncertaintyRegularizationLoss.

Standard ASCII hyphens (-) are strictly enforced.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses_metrics.losses import (
    MultiLayerPseudoSupervisionLoss,
    UncertaintyRegularizationLoss,
)


class ASPPConv(nn.Sequential):
    """Atrous (dilated) convolution block with batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ) -> None:
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """Global image pooling branch for ASPP."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = super().forward(x)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module.

    Captures multi-scale contextual information using parallel atrous convolutions
    with rates [1, 6, 12, 18] and image-level pooled features.
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 256,
        dilations: Optional[List[int]] = None,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 6, 12, 18]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = ASPPConv(in_channels, out_channels, dilation=dilations[1])
        self.branch3 = ASPPConv(in_channels, out_channels, dilation=dilations[2])
        self.branch4 = ASPPConv(in_channels, out_channels, dilation=dilations[3])
        self.branch5 = ASPPPooling(in_channels, out_channels)

        total_channels = out_channels * 5
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: High-level feature map of shape (B, in_channels, H, W).

        Returns:
            Aggregated multi-scale features of shape (B, out_channels, H, W).
        """
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        f5 = self.branch5(x)

        combined = torch.cat([f1, f2, f3, f4, f5], dim=1)
        return self.project(combined)


class DeepLabV3PlusDecoder(nn.Module):
    """Decoder module fusing low-level feature maps and upsampled ASPP features.

    Refines spatial boundaries and projects aggregated representations to class logits.
    """

    def __init__(
        self,
        low_level_in_channels: int = 64,
        aspp_in_channels: int = 256,
        num_classes: int = 5,
        low_level_projected_channels: int = 48,
        decoder_channels: int = 256,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # 1x1 projection for low-level features
        self.project_low = nn.Sequential(
            nn.Conv2d(
                low_level_in_channels,
                low_level_projected_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(low_level_projected_channels),
            nn.ReLU(inplace=True),
        )

        # Refinement convolution layers
        fused_channels = low_level_projected_channels + aspp_in_channels
        self.refinement = nn.Sequential(
            nn.Conv2d(
                fused_channels,
                decoder_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Final classification head
        self.classifier = nn.Conv2d(
            decoder_channels,
            num_classes,
            kernel_size=1,
            stride=1,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        aspp_feat: torch.Tensor,
        low_level_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            aspp_feat: ASPP output tensor of shape (B, aspp_in_channels, H_aspp, W_aspp).
            low_level_feat: Low-level feature tensor of shape (B, low_level_in_channels, H_low, W_low).

        Returns:
            Pre-upsampling logits of shape (B, num_classes, H_low, W_low).
        """
        low_proj = self.project_low(low_level_feat)
        aspp_upsampled = F.interpolate(
            aspp_feat,
            size=low_level_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = torch.cat([aspp_upsampled, low_proj], dim=1)
        refined = self.refinement(fused)
        return self.classifier(refined)


class ClassificationGateMechanism(nn.Module):
    """Classification Gate Mechanism (CGM) for weakly-supervised segmentation.

    Enforces zero false-positive segmentations for absent tissue classes by penalizing
    their logits based on image-level class presence y in {0, 1}^C:
        Z_hat_c = Z_c * y_c + (1 - y_c) * penalty_value

    When class c is marked absent (y_c = 0), Z_hat_c is forced to a large negative penalty
    (-1e4), ensuring softmax probability approaches 0.0 and argmax prediction never selects c.
    """

    def __init__(
        self,
        penalty_value: float = -1e4,
        threshold: float = 0.5,
        default_background_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.penalty_value = penalty_value
        self.threshold = threshold
        self.default_background_idx = default_background_idx

    def forward(
        self,
        logits: torch.Tensor,
        class_presence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            logits: Segmentation logits of shape (B, C, H, W).
            class_presence: Optional image-level presence indicators.
                Shapes supported: (B, C), (C,), or (B, C, 1, 1).
                Values can be binary {0, 1} or probabilities in [0, 1].

        Returns:
            Gated logits of shape (B, C, H, W).
        """
        if class_presence is None:
            return logits

        # Ensure class_presence is on same device and dtype
        presence = class_presence.to(device=logits.device, dtype=logits.dtype)

        # Standardize shape to (B, C, 1, 1)
        if presence.dim() == 1:
            presence = presence.unsqueeze(0)  # (1, C)

        if presence.dim() == 2:
            presence = presence.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Binarize presence indicators using threshold
        gate = (presence >= self.threshold).to(dtype=logits.dtype)

        # Handle edge case where entire presence vector is zero and default background is specified
        if self.default_background_idx is not None:
            sum_presence = gate.sum(dim=1, keepdim=True)  # (B, 1, 1, 1)
            all_zero = (sum_presence == 0).to(dtype=logits.dtype)
            gate[:, self.default_background_idx : self.default_background_idx + 1, :, :] = (
                gate[:, self.default_background_idx : self.default_background_idx + 1, :, :] + all_zero
            )
            gate = torch.clamp(gate, max=1.0)

        # Apply Classification Gate equation:
        # Z_hat_c = Z_c * y_c + (1 - y_c) * penalty_value
        penalty = torch.tensor(self.penalty_value, device=logits.device, dtype=logits.dtype)
        gated_logits = logits * gate + (1.0 - gate) * penalty

        return gated_logits


class ResidualBlock(nn.Module):
    """Basic residual block for the feature extraction backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.relu(out)


class ResNetBackbone(nn.Module):
    """Multi-stage residual backbone for DeepLabV3+.

    Extracts:
    - low_level_features: Stride 4 representation for spatial edge refinement.
    - intermediate_features: Stages corresponding to MLPS levels (stage3/b4_5, stage4/b5_2, stage5/bn7).
    - high_level_features: Stride 16 deep representation for ASPP input.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        # Stem: Conv 7x7 stride 2 + MaxPool stride 2 -> Stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 1: Stride 4, Channels = 64 (Low-Level Features)
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels, stride=1),
            ResidualBlock(base_channels, base_channels, stride=1),
        )

        # Stage 2: Stride 8, Channels = 128
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2, stride=1),
        )

        # Stage 3: Stride 16, Channels = 256 (b4_5 equivalent for MLPS)
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=1),
        )

        # Stage 4: Stride 16, Channels = 512 (b5_2 equivalent for MLPS)
        self.stage4 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 8, stride=1, dilation=2),
            ResidualBlock(base_channels * 8, base_channels * 8, stride=1, dilation=2),
        )

        # Stage 5: Stride 16, Channels = 512 (bn7 equivalent for ASPP input)
        self.stage5 = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 8, stride=1, dilation=4),
            ResidualBlock(base_channels * 8, base_channels * 8, stride=1, dilation=4),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            high_level_feat: Stride 16 tensor (B, 512, H/16, W/16).
            low_level_feat: Stride 4 tensor (B, 64, H/4, W/4).
            intermediate_dict: Dictionary with keys 'stage3'/'b4_5', 'stage4'/'b5_2', 'stage5'/'bn7'.
        """
        stem = self.stem(x)
        low_level = self.stage1(stem)  # (B, 64, H/4, W/4)
        s2 = self.stage2(low_level)    # (B, 128, H/8, W/8)
        s3 = self.stage3(s2)           # (B, 256, H/16, W/16) - b4_5
        s4 = self.stage4(s3)           # (B, 512, H/16, W/16) - b5_2
        high_level = self.stage5(s4)   # (B, 512, H/16, W/16) - bn7

        intermediates = {
            "b4_5": s3,
            "stage3": s3,
            "b5_2": s4,
            "stage4": s4,
            "bn7": high_level,
            "stage5": high_level,
        }

        return high_level, low_level, intermediates


class DeepLabV3PlusWithCGM(nn.Module):
    """DeepLabV3+ Semantic Segmentation Network with Classification Gate Mechanism (CGM).

    Tailored for weakly-supervised histopathology on BCSS-WSSS (5 classes: TUM, STR, LYM, NEC, BACK).

    Features:
    - ASPP with atrous dilations [1, 6, 12, 18] and image pooling.
    - Decoder fusing high-resolution low-level features and high-level context.
    - Classification Gate Mechanism (CGM) extinguishing absent class false positives.
    - Full compatibility with MultiLayerPseudoSupervisionLoss, UncertaintyRegularizationLoss,
      and standard CrossEntropyLoss.
    - Inference utilities: predict() for argmax masks, predict_proba() for softmax maps.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        backbone_channels: int = 512,
        low_level_channels: int = 64,
        aspp_dilations: Optional[List[int]] = None,
        cgm_penalty: float = -1e4,
        ignore_index: int = 255,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.ignore_index = ignore_index

        # 1. Feature Backbone
        self.backbone = ResNetBackbone(in_channels=in_channels, base_channels=low_level_channels)

        # 2. ASPP Module
        self.aspp = ASPP(
            in_channels=backbone_channels,
            out_channels=256,
            dilations=aspp_dilations or [1, 6, 12, 18],
            dropout_rate=dropout_rate,
        )

        # 3. Decoder Module
        self.decoder = DeepLabV3PlusDecoder(
            low_level_in_channels=low_level_channels,
            aspp_in_channels=256,
            num_classes=num_classes,
            low_level_projected_channels=48,
            decoder_channels=256,
            dropout_rate=dropout_rate,
        )

        # 4. Classification Gate Mechanism
        self.cgm = ClassificationGateMechanism(
            penalty_value=cgm_penalty,
            default_background_idx=num_classes - 1,
        )

        # 5. Loss Modules for Training
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mlps_loss = MultiLayerPseudoSupervisionLoss(ignore_index=ignore_index)
        self.uncertainty_loss = UncertaintyRegularizationLoss()

    def forward(
        self,
        x: torch.Tensor,
        class_presence: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        pseudo_masks: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        loss_type: str = "ce",
        return_dict: bool = False,
        return_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """Forward pass of DeepLabV3PlusWithCGM.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
            class_presence: Optional image-level multi-hot labels or probabilities (B, C).
            targets: Optional ground truth / pseudo-mask labels (B, H, W) for CE loss.
            pseudo_masks: Optional pseudo-masks dictionary or tensor for MLPS / uncertainty loss.
            loss_type: "ce", "mlps", or "uncertainty".
            return_dict: When True, returns a dictionary with all intermediate artifacts.
            return_loss: When True, returns (logits, loss).

        Returns:
            Gated logits (B, num_classes, H, W) by default, or structured output.
        """
        input_size = x.shape[-2:]

        # Extract features
        high_level, low_level, intermediate_dict = self.backbone(x)

        # Multi-scale context aggregation via ASPP
        aspp_features = self.aspp(high_level)

        # Spatial fusion via Decoder
        decoder_logits = self.decoder(aspp_features, low_level)

        # Upsample logits to original input resolution
        raw_logits = F.interpolate(
            decoder_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        # Apply Classification Gate Mechanism
        gated_logits = self.cgm(raw_logits, class_presence=class_presence)

        # Compute loss if requested
        loss: Optional[torch.Tensor] = None
        if return_loss or targets is not None or pseudo_masks is not None:
            loss = self.compute_loss(
                logits=gated_logits,
                targets=targets,
                pseudo_masks=pseudo_masks,
                loss_type=loss_type,
            )

        if return_dict:
            return {
                "logits": gated_logits,
                "raw_logits": raw_logits,
                "loss": loss,
                "features": intermediate_dict,
                "aspp_features": aspp_features,
            }

        if return_loss:
            if loss is None:
                raise ValueError("return_loss=True requires targets or pseudo_masks to be provided.")
            return gated_logits, loss

        return gated_logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        pseudo_masks: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        loss_type: str = "ce",
    ) -> torch.Tensor:
        """Computes training loss using pseudo-masks or ground truth targets.

        Supports:
        - "ce": Standard Cross-Entropy with ignore_index.
        - "mlps": Multi-Layer Pseudo-Supervision Loss across feature stages.
        - "uncertainty": Boundary uncertainty-regularized loss.

        Args:
            logits: Predicted segmentation logits (B, C, H, W).
            targets: Optional standard labels tensor (B, H, W).
            pseudo_masks: Optional multi-layer dict or single pseudo-mask tensor.
            loss_type: Loss formulation identifier.

        Returns:
            Scalar loss tensor.
        """
        if loss_type == "ce":
            supervision = targets if targets is not None else pseudo_masks
            if supervision is None:
                raise ValueError("targets or pseudo_masks required for CrossEntropyLoss.")
            if isinstance(supervision, dict):
                # Default to bn7 or first available key
                key = "bn7" if "bn7" in supervision else list(supervision.keys())[0]
                supervision = supervision[key]
            return self.ce_loss(logits, supervision.long())

        elif loss_type == "mlps":
            if pseudo_masks is None or not isinstance(pseudo_masks, dict):
                raise ValueError("pseudo_masks dictionary required for MLPS loss.")
            return self.mlps_loss(logits, pseudo_masks)

        elif loss_type == "uncertainty":
            supervision = targets if targets is not None else pseudo_masks
            if supervision is None:
                raise ValueError("targets or pseudo_masks required for UncertaintyRegularizationLoss.")
            if isinstance(supervision, dict):
                key = "bn7" if "bn7" in supervision else list(supervision.keys())[0]
                supervision = supervision[key]
            return self.uncertainty_loss(logits, supervision.long())

        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'ce', 'mlps', or 'uncertainty'.")

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        class_presence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs inference and returns discrete class segmentation masks.

        Args:
            x: Input tensor of shape (B, 3, H, W).
            class_presence: Optional presence vectors (B, C).

        Returns:
            Class index predictions of shape (B, H, W).
        """
        logits = self.forward(x, class_presence=class_presence)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        class_presence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs inference and returns softmax class probability maps.

        Args:
            x: Input tensor of shape (B, 3, H, W).
            class_presence: Optional presence vectors (B, C).

        Returns:
            Probability tensor of shape (B, num_classes, H, W).
        """
        logits = self.forward(x, class_presence=class_presence)
        return F.softmax(logits, dim=1)
