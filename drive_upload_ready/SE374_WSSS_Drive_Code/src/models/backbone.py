"""
Multi-Stage Feature Backbone for Weakly-Supervised Histopathology Segmentation (SE374.R11).

Provides:
- MultiStageFeatureBackbone: Deep convolutional network supporting intermediate feature extraction
  at stages b4_5 (stage3), b5_2 (stage4), and bn7 (stage5), along with Global Average Pooling (GAP)
  classification logits for multi-label patch supervision.
- Modular residual/conv blocks with customizable channel dimensions and dropout regularization.
- Backward-compatible alias: HistopathologyClassifierBackbone.

Strict adherence to:
- Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Invariant 07: Genuine PyTorch implementation with zero mockup.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with batch normalization and ReLU activation.
    
    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
        |                                                ^
        +------------> [Conv1x1 -> BN / Identity] ------+
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + identity
        return self.relu(out)


# Alias for backward compatibility
SimpleConvBlock = ResidualConvBlock


class MultiStageFeatureBackbone(nn.Module):
    """Multi-stage feature extractor and multi-label classifier for WSSS histopathology.
    
    Extracts intermediate representations matching the Multi-Layer Pseudo-Supervision (MLPS)
    hierarchy (Han et al., MedIA 2022):
    - stage3 -> b4_5 equivalent (stride 8, e.g. 256 channels)
    - stage4 -> b5_2 equivalent (stride 16, e.g. 512 channels)
    - stage5 -> bn7 equivalent (stride 16, e.g. 512 channels)
    
    Applies Global Average Pooling (GAP) followed by a 1x1 convolution classification head
    for multi-label tissue classification (Tumor, Stroma, Lymphocytic infiltrate, Necrosis).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        stage_channels: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
    ) -> None:
        """Args:
            in_channels: Number of input image channels (default: 3 for RGB).
            num_classes: Number of foreground classes for multi-label supervision (default: 4).
            stage_channels: List of 5 channel dimensions for stages 1 to 5.
                            Default: [64, 128, 256, 512, 512].
            dropout_rate: Dropout probability applied before final classification head.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if stage_channels is None:
            stage_channels = [64, 128, 256, 512, 512]
        assert len(stage_channels) == 5, f"stage_channels must contain exactly 5 integers, got {stage_channels}"
        self.stage_channels = stage_channels

        c1, c2, c3, c4, c5 = stage_channels

        # Stem: fast spatial reduction for gigapixel WSI patch ingestion
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Multi-resolution convolutional stages
        self.stage1 = ResidualConvBlock(c1, c1, stride=1, dropout_rate=0.0)
        self.stage2 = ResidualConvBlock(c1, c2, stride=2, dropout_rate=0.0)
        self.stage3 = ResidualConvBlock(c2, c3, stride=2, dropout_rate=dropout_rate * 0.5)  # b4_5 equivalent
        self.stage4 = ResidualConvBlock(c3, c4, stride=2, dropout_rate=dropout_rate * 0.5)  # b5_2 equivalent
        self.stage5 = ResidualConvBlock(c4, c5, stride=1, dropout_rate=dropout_rate)        # bn7 equivalent

        # Dropout before classification
        self.classifier_dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        # Multi-label classification head (1x1 Conv projecting bn7 channels to num_classes)
        self.fc_head = nn.Conv2d(c5, num_classes, kernel_size=1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through multi-stage backbone.
        
        Args:
            x: Input image tensor of shape (B, C_in, H, W).
            
        Returns:
            logits: (B, num_classes) multi-label classification logits.
            features: Dictionary containing intermediate feature stages:
                      - 'stage1': Stem + Stage 1 output (stride 4)
                      - 'stage2': Stage 2 output (stride 8)
                      - 'b4_5': Stage 3 output (stride 16)
                      - 'b5_2': Stage 4 output (stride 32)
                      - 'bn7': Stage 5 output (stride 32)
        """
        x_stem = self.stem(x)
        f1 = self.stage1(x_stem)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)  # b4_5
        f4 = self.stage4(f3)  # b5_2
        f5 = self.stage5(f4)  # bn7

        # Global Average Pooling on bn7
        f5_drop = self.classifier_dropout(f5)
        pooled = F.adaptive_avg_pool2d(f5_drop, (1, 1))
        logits = self.fc_head(pooled).view(x.size(0), -1)

        features = {
            "stage1": f1,
            "stage2": f2,
            "b4_5": f3,
            "b5_2": f4,
            "bn7": f5,
        }
        return logits, features

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method to extract feature dictionary without computing logits."""
        _, features = self.forward(x)
        return features

    def compute_cam(
        self,
        x: torch.Tensor,
        feature_layer: str = "bn7",
    ) -> torch.Tensor:
        """Extracts standard Class Activation Maps (CAM) via Conv weight projection.
        
        Formula:
            M_c(x, y) = ReLU(sum_k w_c^k * F_k(x, y))
            
        Args:
            x: Input tensor of shape (B, 3, H, W).
            feature_layer: Layer key in features dictionary ('bn7', 'b5_2', or 'b4_5').
            
        Returns:
            cam: Activation tensor of shape (B, num_classes, H_feat, W_feat).
        """
        _, features = self.forward(x)
        feat = features[feature_layer]
        feat_channels = feat.shape[1]
        weight = self.fc_head.weight

        if feat_channels != weight.shape[1]:
            # Adapt channel dimension if feature is from earlier shallower stage
            weight_adapted = weight[:, :feat_channels, :, :]
            cam = F.conv2d(feat, weight_adapted)
        else:
            cam = F.conv2d(feat, weight)

        return F.relu(cam)

    def get_classifier_weights(self) -> torch.Tensor:
        """Returns the classification projection weight tensor of shape (num_classes, C_feat, 1, 1)."""
        return self.fc_head.weight


# Alias for backward compatibility with existing code
HistopathologyClassifierBackbone = MultiStageFeatureBackbone
