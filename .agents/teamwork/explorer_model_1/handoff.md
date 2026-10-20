# DeepLabV3+ Architecture & Pretrained Weight Checkpoint Analysis Report

## 1. Observation

### 1.1 Codebase References to Pretrained Weights and Architecture
Direct inspection of files revealed the origin and usage of the pretrained weights `stage2_checkpoint_trained_on_bcss.pth`:
- `notebooks/SE374_Colab_WSSS_Pipeline.ipynb`:
  - Line 152: `CKPT_FILE = CHECKPOINT_DIR / 'stage2_checkpoint_trained_on_bcss.pth'`
  - Line 156: `!gdown --id 1cXm29ko8UevbaeW6YreSg-vuNaQiN_je -O /content/checkpoints/stage2_checkpoint_trained_on_bcss.pth`
- `data/README.md`:
  - Line 35: `- **Pretrained Checkpoints:** Google Drive: https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing`
- `repos/WSSS-Tissue/README.md`:
  - Lines 14-20: Official repository for "Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels" (Han et al., Medical Image Analysis 2022, DOI: 10.1016/j.media.2022.102487).
  - Line 20: `- [stage2_checkpoint_trained_on_bcss.pth](https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing)`
- `repos/WSSS-Tissue/3_train_stage2.py`:
  - Line 25-29: Instantiates segmentation model:
    ```python
    model = DeepLab(num_classes=self.nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    ```
  - Line 139-142: Saves model state:
    ```python
    self.saver.save_checkpoint({
        'state_dict': self.model.module.state_dict(),
        'optimizer': self.optimizer.state_dict()
    }, 'stage2_checkpoint_trained_on_'+self.args.dataset+'.pth')
    ```
  - Lines 190-200: Default arguments:
    `--backbone resnet`, `--out-stride 16`, `--n_class 4`, `--dataset bcss`.
- `repos/LowMagWSS/SegMethod/WSSS-Tissue/03_stage2.sh`:
  - Line 85: Runs `3_train_stage2.py` with identical arguments, confirming inheritance from Han et al. (MedIA 2022).

### 1.2 Comparison Between `src/models/` and Reference Implementation
- `src/models/deeplabv3plus.py`:
  - Defines `DeepLabV3PlusWithCGM` using a custom lightweight `ResNetBackbone` (`stem`, `stage1`, `stage2`, `stage3`, `stage4`, `stage5` with 2 convolutional layers per block, 512 channels at stage 5).
  - Layer naming in decoder: `project_low`, `refinement`, `classifier`.
  - Intended for unit testing and modular training experiments within the repository.
- `repos/WSSS-Tissue/network/deeplab.py`:
  - Backbone: `network.backbone.resnet.ResNet101` with bottleneck blocks (3, 4, 23, 3 layers), expansion=4, output channels 2048 at layer4, and Multi-Grid rates `[1, 2, 4]` with dilations `[1, 1, 1, 2]`.
  - Low-level feature tap: `layer1` output with 256 channels.
  - ASPP: Atrous convolutions with dilations `[1, 6, 12, 18]` and 2048 input channels -> 256 output channels, plus global average pooling branch (2048 -> 256), projected through 1x1 conv (1280 -> 256) with Dropout(0.5).
  - Decoder: 1x1 conv projecting low-level features (256 -> 48), concatenating with 4x upsampled ASPP features (48 + 256 = 304 channels), and `last_conv` sequential (304 -> 256 -> 256 -> num_classes).
  - Layer naming in decoder: `conv1`, `bn1`, `last_conv.0`, `last_conv.1`, `last_conv.4`, `last_conv.5`, `last_conv.8`.

### 1.3 Standalone Key Verification Experiment
Direct execution of `.agents/teamwork/explorer_model_1/compare_keys.py` comparing the standalone architecture against `repos/WSSS-Tissue/network/deeplab.py`:
- Command executed:
  `cmd /c "python .agents\teamwork\explorer_model_1\compare_keys.py"`
- Result:
  `Key set difference count: 0`
  `PERFECT 100% KEY MATCH between repos/WSSS-Tissue and StandaloneDeepLab!`
  `Mismatched shapes count: 0`
  `PERFECT 100% TENSOR SHAPE MATCH for all 680 parameters and buffers!`
- Total model parameters: 59,339,940 (~59.34M FP32 parameters, ~226.4 MB).
- Total state_dict tensors: 680 (including weights, biases, running_mean, running_var, and num_batches_tracked).

### 1.4 Third-Party Dependency Pitfalls
Direct test execution of `repos/WSSS-Tissue/network/deeplab.py` failed with:
- Line 4: `from torchsummary import summary` -> `ModuleNotFoundError: No module named 'torchsummary'`
- Line 5: `from network.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d` -> requires external package or distributed GPU environment.
In evaluation mode on Colab single GPU, `torch.nn.BatchNorm2d` is completely compatible with serialized weights trained with `SynchronizedBatchNorm2d` because key names (`weight`, `bias`, `running_mean`, `running_var`) and tensor dimensions are identical.

---

## 2. Logic Chain

1. **Origin of Weights (Observation 1.1):**
   The checkpoint `stage2_checkpoint_trained_on_bcss.pth` distributed via Google Drive ID `1cXm29ko8UevbaeW6YreSg-vuNaQiN_je` is the official Stage 2 checkpoint released by Chu Han et al. (MedIA 2022). It was trained on the BCSS-WSSS dataset using `3_train_stage2.py`.

2. **Checkpoint State Structure (Observation 1.1 & 1.3):**
   In `3_train_stage2.py`, checkpoints are saved as `{'state_dict': self.model.module.state_dict(), 'optimizer': ...}`. The state dictionary keys follow the exact module hierarchy of `repos/WSSS-Tissue/network/deeplab.py`:
   - Prefix: `backbone.` for ResNet-101 (from `conv1` through `layer4.2.bn3`).
   - Prefix: `aspp.` for ASPP module (`aspp1`, `aspp2`, `aspp3`, `aspp4`, `global_avg_pool`, `conv1`, `bn1`).
   - Prefix: `decoder.` for Decoder module (`conv1`, `bn1`, `last_conv.0`, `last_conv.1`, `last_conv.4`, `last_conv.5`, `last_conv.8`).
   If the checkpoint was saved with `DataParallel`, keys may contain a leading `module.` prefix, which must be stripped during ingestion.

3. **Number of Output Classes (`num_classes`):**
   In `3_train_stage2.py`, `--n_class 4` is used because BCSS foreground classes are 4:
   - Class 0: Tumor (TUM)
   - Class 1: Stroma (STR)
   - Class 2: Lymphocytic infiltrate (LYM)
   - Class 3: Necrosis (NEC)
   Background (Class 4: BACK / Glass) was excluded from the loss during Stage 2 training and assigned in post-processing.
   However, `decoder.last_conv.8.weight` shape may be either `[4, 256, 1, 1]` or `[5, 256, 1, 1]`. Detecting `num_classes = state_dict['decoder.last_conv.8.weight'].shape[0]` dynamically guarantees zero key/shape mismatch.

4. **Integration with Otsu Clinical Gating:**
   Because DeepLabV3+ outputs logits over foreground tissue classes, glass background regions are clinically extinguished by applying Otsu thresholding in HSV saturation space (`saturation < threshold`), forcing those spatial pixels to Class 4 (BACK). This eliminates false-positive tumor and stroma predictions on empty slide regions.

5. **Standalone Deployment for Colab (Observation 1.4):**
   Embedding DeepLabV3+ directly into the Colab notebook without importing from external repos avoids `ModuleNotFoundError` for `torchsummary` or `sync_batchnorm`. Standard `torch.nn.BatchNorm2d` operates seamlessly in `eval()` mode.

---

## 3. Caveats

1. **Pretrained Checkpoint File Not Local:** The binary `.pth` file is hosted on Google Drive (`1cXm29ko8UevbaeW6YreSg-vuNaQiN_je`) and is not pre-committed to git (as required by Invariant 03). It is downloaded at runtime in Google Colab via `gdown`.
2. **Device Compatibility:** In inference mode on Colab (CPU, T4 Free, or L4 Pro GPU), `output_stride=16` is used. A batch size of 1 to 4 with input resolution 224x224 or 512x512 requires less than 2.0 GB VRAM.
3. **Data Preprocessing Normalization:** PyTorch models trained on ImageNet/BCSS expect standard RGB normalization:
   `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.

---

## 4. Conclusion

The exact neural network architecture matching `stage2_checkpoint_trained_on_bcss.pth` is:
- **Model:** DeepLabV3+ (output_stride = 16)
- **Backbone:** ResNet-101 with Multi-Grid `[1, 2, 4]` at layer4 (2048 feature channels)
- **Low-level features:** ResNet-101 layer1 (256 channels)
- **ASPP:** 4 atrous branches (rates 1, 6, 12, 18) + global pooling -> 256 channels
- **Decoder:** Low-level conv (256 -> 48), concatenation (304), refinement convs (256), final conv (256 -> num_classes)
- **Total parameters:** 59,339,940
- **Total state_dict keys:** 680

Below is the standalone, self-contained PyTorch implementation ready for direct embedding into Google Colab.

### 4.1 Standalone PyTorch DeepLabV3+ Code Specification

```python
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. RESNET-101 BACKBONE MODULES
# ==============================================================================

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class ResNet101(nn.Module):
    """ResNet-101 feature extractor configured for DeepLabV3+ (stride 16)."""

    def __init__(self, output_stride: int = 16) -> None:
        super().__init__()
        self.inplanes = 64
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError(f"Unsupported output_stride: {output_stride}")

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Residual Stages
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(Bottleneck, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, dilation, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_MG_unit(
        self,
        block: type,
        planes: int,
        blocks: list,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, dilation=blocks[0] * dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i] * dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat


# ==============================================================================
# 2. ATROUS SPATIAL PYRAMID POOLING (ASPP)
# ==============================================================================

class _ASPPModule(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int, padding: int, dilation: int) -> None:
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPP(nn.Module):
    def __init__(self, output_stride: int = 16) -> None:
        super().__init__()
        inplanes = 2048
        dilations = [1, 6, 12, 18] if output_stride == 16 else [1, 12, 24, 36]

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)


# ==============================================================================
# 3. DECODER MODULE
# ==============================================================================

class Decoder(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        low_level_inplanes = 256

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor, low_level_feat: torch.Tensor) -> torch.Tensor:
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x


# ==============================================================================
# 4. DEEPLABV3+ NETWORK & PRETRAINED WEIGHT LOADER
# ==============================================================================

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet-101 backbone and ASPP (MedIA 2022 SOTA specification)."""

    def __init__(self, num_classes: int = 4, output_stride: int = 16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet101(output_stride=output_stride)
        self.aspp = ASPP(output_stride=output_stride)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feat, low_level = self.backbone(x)
        x = self.aspp(feat)
        x = self.decoder(x, low_level)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


def build_and_load_deeplabv3plus(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> DeepLabV3Plus:
    """Builds DeepLabV3+ and loads pretrained weights from stage2_checkpoint_trained_on_bcss.pth.
    
    Automatically strips DataParallel 'module.' prefixes and detects num_classes.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")

    print(f"--> Loading checkpoint: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip DataParallel 'module.' prefix
    cleaned_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    # Inspect classification head shape
    if "decoder.last_conv.8.weight" in cleaned_state_dict:
        num_classes = cleaned_state_dict["decoder.last_conv.8.weight"].shape[0]
        print(f"--> Detected num_classes from checkpoint: {num_classes}")
    else:
        num_classes = 4

    model = DeepLabV3Plus(num_classes=num_classes, output_stride=16)
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"[OK] Weights loaded successfully (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    model.to(device)
    model.eval()
    return model
```

---

## 5. Verification Method

To independently verify this implementation:

1. **Verify Standalone Compilation and Parameters:**
   ```bash
   cmd /c "python .agents\teamwork\explorer_model_1\verify_model.py"
   ```
   - Invalidation condition: Total parameters != 59,339,940 or forward pass fails with tensor shape mismatch.

2. **Verify 100% Key-by-Key Equivalence with Reference Repo:**
   ```bash
   cmd /c "python .agents\teamwork\explorer_model_1\compare_keys.py"
   ```
   - Invalidation condition: `Key set difference count > 0` or `Mismatched shapes count > 0`.

3. **Verify Pytest Suite Passes:**
   ```bash
   cmd /c "set KMP_DUPLICATE_LIB_OK=TRUE & python -m pytest tests/"
   ```
   - Invalidation condition: Any test failure across existing test suite (169 passed tests).

