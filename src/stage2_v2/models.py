"""
Stage-2 v2 Dual-Head Segmentation & Patch-Classification Models.

Unified interface across all 4 Gate-5 architectures:
- DeepLabV3PlusDualHead (Arm 1: ResNet-101 OS=16 + ASPP + Decoder + GAP cls_head)
- SegFormerDualHead (Arm 2: MiT-B3 encoder + MLP decoder + GAP cls_head)
- ConvNeXtFPNDualHead (Arm 3: ConvNeXt-Base stochastic_depth_prob=0.2 + GroupNorm FPN + GAP cls_head)
- DINOv2AdapterDualHead (Arm 4: DINOv2 ViT-B/14 + Spatial Prior Stem + Per-Stage LayerNorm FPN + GAP cls_head)

Every model accepts ImageNet-normalized (B, 3, H, W) tensors and returns:
  {"seg_logits": (B, 4, H, W), "cls_logits": (B, 4)}
and exposes forward_logits(x) -> (B, 4, H, W).

Tier-1 packages ONLY (torch, torchvision, transformers). Pure ASCII hyphen '-' only.
"""

import contextlib
import math
import re
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.ops import FeaturePyramidNetwork
from transformers import (
    Dinov2Config,
    Dinov2Model,
    SegformerConfig,
    SegformerForSemanticSegmentation,
)

from src.models.wsss_inference_methods import DeepLabV3Plus, safe_torch_load
from src.stage2_v2.config import Stage2V2Config
from src.training.train_stage2 import load_imagenet_pretrained_resnet101


def _split_wd_param_groups(
    params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Split parameters into decay and no-decay (1D/bias/norm) optimizer groups."""
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    seen = set()
    for p in params:
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "lr": float(lr), "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "lr": float(lr), "weight_decay": 0.0})
    return groups


class BaseStage2DualHead(nn.Module):
    """Base contract for Stage-2 v2 dual-head segmentation models."""

    arch: str = "base"
    backbone_id: str = "base"
    num_classes: int = 4
    input_norm: str = "imagenet"
    has_cls_head: bool = True
    track: str = "track_a_clean"
    tcga_pretrained: bool = False

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return only dense segmentation logits (B, num_classes, H, W)."""
        return self.forward(x)["seg_logits"]

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    def head_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    def get_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[Dict[str, Any]]:
        """Build differential learning-rate parameter groups for backbone vs head."""
        groups: List[Dict[str, Any]] = []
        groups.extend(_split_wd_param_groups(self.backbone_parameters(), lr_backbone, weight_decay))
        groups.extend(_split_wd_param_groups(self.head_parameters(), lr_head, weight_decay))
        return groups

    def model_metadata(self) -> Dict[str, Any]:
        """Return architecture metadata for atomic checkpoint persistence."""
        return {
            "arch": self.arch,
            "backbone_id": self.backbone_id,
            "num_classes": int(self.num_classes),
            "input_norm": self.input_norm,
            "has_cls_head": bool(self.has_cls_head),
            "track": str(self.track),
            "tcga_pretrained": bool(self.tcga_pretrained),
        }


class DeepLabV3PlusDualHead(BaseStage2DualHead):
    """Arm 1 / Arm 8: DeepLabV3+ ResNet-101 (OS=16) with decoupled GAP classification head."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        track: str = "track_a_clean",
        tcga_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.arch = "deeplabv3plus_r101"
        self.backbone_id = "resnet101"
        self.num_classes = int(num_classes)
        self.pretrained_init = bool(pretrained)
        self.track = str(track)
        self.tcga_pretrained = bool(tcga_pretrained)

        self.seg_net = DeepLabV3Plus(num_classes=self.num_classes, output_stride=16)
        self._init_head_weights()
        if pretrained:
            load_imagenet_pretrained_resnet101(self.seg_net, allow_random_init_fallback=False)

        self.cls_head = nn.Linear(2048, self.num_classes)
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.cls_head.bias)

    def _init_head_weights(self) -> None:
        for mod in (self.seg_net.aspp, self.seg_net.decoder):
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        feat, low_level = self.seg_net.backbone(x)
        cls_feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        cls_logits = self.cls_head(cls_feat)
        aspp_out = self.seg_net.aspp(feat)
        dec_out = self.seg_net.decoder(aspp_out, low_level)
        seg_logits = F.interpolate(dec_out, size=(h, w), mode="bilinear", align_corners=True)
        return {"seg_logits": seg_logits, "cls_logits": cls_logits}

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.seg_net.backbone.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.seg_net.aspp.parameters(),
            self.seg_net.decoder.parameters(),
            self.cls_head.parameters(),
        )

    def model_metadata(self) -> Dict[str, Any]:
        meta = super().model_metadata()
        meta["model_kwargs"] = {
            "num_classes": self.num_classes,
            "track": self.track,
            "tcga_pretrained": self.tcga_pretrained,
        }
        return meta


class SegFormerDualHead(BaseStage2DualHead):
    """Arm 2 / Arm 6: SegFormer (MiT-B3 / MiT-B5) with stage-4 GAP classification head."""

    def __init__(
        self,
        backbone_id: str = "nvidia/mit-b3",
        num_classes: int = 4,
        pretrained: bool = True,
        tiny_offline_config: bool = False,
        arch: str = "segformer_b3",
        drop_path_rate: Optional[float] = None,
        track: str = "track_a_clean",
        tcga_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.arch = str(arch)
        self.backbone_id = str(backbone_id)
        self.num_classes = int(num_classes)
        self.tiny_offline_config = bool(tiny_offline_config or not pretrained)
        self.drop_path_rate = float(drop_path_rate) if drop_path_rate is not None else None
        self.track = str(track)
        self.tcga_pretrained = bool(tcga_pretrained)

        if pretrained:
            fp_kwargs: Dict[str, Any] = {
                "num_labels": self.num_classes,
                "ignore_mismatched_sizes": True,
                "output_loading_info": True,
            }
            if self.drop_path_rate is not None:
                fp_kwargs["drop_path_rate"] = self.drop_path_rate
            model_and_info = SegformerForSemanticSegmentation.from_pretrained(
                self.backbone_id,
                **fp_kwargs,
            )
            if isinstance(model_and_info, tuple):
                self.segformer, loading_info = model_and_info
            else:
                self.segformer, loading_info = model_and_info, {}
            missing_enc = {
                k for k in loading_info.get("missing_keys", []) if str(k).startswith("segformer.")
            }
            enc_keys = [k for k in self.segformer.state_dict() if k.startswith("segformer.")]
            matched_enc = len(enc_keys) - len(missing_enc)
            if matched_enc < 300:
                raise RuntimeError(
                    f"[STAGE2 ERROR] SegFormer matched only {matched_enc} encoder keys "
                    f"(expected >= 300) for {self.backbone_id}."
                )
        else:
            cfg_kwargs: Dict[str, Any] = {
                "num_labels": self.num_classes,
                "depths": [1, 1, 1, 1],
                "hidden_sizes": [16, 32, 64, 128],
                "decoder_hidden_size": 32,
                "num_attention_heads": [1, 2, 4, 8],
                "sr_ratios": [8, 4, 2, 1],
            }
            if self.drop_path_rate is not None:
                cfg_kwargs["drop_path_rate"] = self.drop_path_rate
            cfg = SegformerConfig(**cfg_kwargs)
            self.segformer = SegformerForSemanticSegmentation(cfg)

        c4 = int(self.segformer.config.hidden_sizes[-1])
        self.cls_head = nn.Linear(c4, self.num_classes)
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.cls_head.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.segformer(pixel_values=x, output_hidden_states=True)
        seg_logits = F.interpolate(out.logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        stage4 = out.hidden_states[-1]
        cls_feat = F.adaptive_avg_pool2d(stage4, (1, 1)).flatten(1)
        cls_logits = self.cls_head(cls_feat)
        return {"seg_logits": seg_logits, "cls_logits": cls_logits}

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.segformer.segformer.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return chain(self.segformer.decode_head.parameters(), self.cls_head.parameters())

    def model_metadata(self) -> Dict[str, Any]:
        meta = super().model_metadata()
        meta["model_kwargs"] = {
            "arch": self.arch,
            "backbone_id": self.backbone_id,
            "num_classes": self.num_classes,
            "tiny_offline_config": self.tiny_offline_config,
            "drop_path_rate": self.drop_path_rate,
            "track": self.track,
            "tcga_pretrained": self.tcga_pretrained,
        }
        return meta


class ConvNeXtFPNDualHead(BaseStage2DualHead):
    """Arm 3 / Arm 7: ConvNeXt-Base/Large + GroupNorm FPN + GAP cls_head."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        arch: str = "convnext_base_fpn",
        backbone_id: str = "convnext_base",
        stochastic_depth_prob: float = 0.2,
        track: str = "track_a_clean",
        tcga_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.arch = str(arch)
        self.backbone_id = str(backbone_id)
        self.num_classes = int(num_classes)
        self.stochastic_depth_prob = float(stochastic_depth_prob)
        self.track = str(track)
        self.tcga_pretrained = bool(tcga_pretrained)

        bb = self.backbone_id.strip().lower()
        if bb == "convnext_base":
            weights = tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.convnext_base(
                weights=weights,
                stochastic_depth_prob=self.stochastic_depth_prob,
            )
            in_channels_list = [128, 256, 512, 1024]
        elif bb == "convnext_large":
            weights = tv_models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.convnext_large(
                weights=weights,
                stochastic_depth_prob=self.stochastic_depth_prob,
            )
            in_channels_list = [192, 384, 768, 1536]
        else:
            raise ValueError(f"Unsupported ConvNeXt backbone_id: {backbone_id!r}.")

        self.in_channels_list: List[int] = list(in_channels_list)
        self.features = backbone.features
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=256,
        )
        self.decode_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, self.num_classes, kernel_size=1),
        )
        self.cls_head = nn.Linear(self.in_channels_list[-1], self.num_classes)
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.cls_head.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        tapped: Dict[str, torch.Tensor] = {}
        stage_map = {1: "c2", 3: "c3", 5: "c4", 7: "c5"}
        cur = x
        for idx, layer in enumerate(self.features):
            cur = layer(cur)
            if idx in stage_map:
                tapped[stage_map[idx]] = cur

        c5 = tapped["c5"]
        cls_feat = F.adaptive_avg_pool2d(c5, (1, 1)).flatten(1)
        cls_logits = self.cls_head(cls_feat)

        fpn_feats = self.fpn(tapped)
        target_hw = fpn_feats["c2"].shape[-2:]
        fused_list = [fpn_feats["c2"]]
        for key in ("c3", "c4", "c5"):
            fused_list.append(
                F.interpolate(fpn_feats[key], size=target_hw, mode="bilinear", align_corners=False)
            )
        fused = torch.cat(fused_list, dim=1)
        logits_low = self.decode_head(fused)
        seg_logits = F.interpolate(logits_low, size=(h, w), mode="bilinear", align_corners=False)
        return {"seg_logits": seg_logits, "cls_logits": cls_logits}

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.features.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.fpn.parameters(),
            self.decode_head.parameters(),
            self.cls_head.parameters(),
        )

    def model_metadata(self) -> Dict[str, Any]:
        meta = super().model_metadata()
        meta["model_kwargs"] = {
            "arch": self.arch,
            "backbone_id": self.backbone_id,
            "num_classes": self.num_classes,
            "stochastic_depth_prob": self.stochastic_depth_prob,
            "in_channels_list": list(self.in_channels_list),
            "track": self.track,
            "tcga_pretrained": self.tcga_pretrained,
        }
        return meta


def _compute_evenly_spaced_stages(num_layers: int) -> Tuple[int, int, int, int]:
    """Select 4 hidden-state indices ([3,6,9,12] for 12-layer ViT, scaled for tiny test configs)."""
    if num_layers == 12:
        return (3, 6, 9, 12)
    if num_layers < 4:
        return tuple(min(num_layers, max(1, i + 1)) for i in range(4))  # type: ignore[return-value]
    return (
        max(1, int(round(num_layers * 0.25))),
        max(1, int(round(num_layers * 0.50))),
        max(1, int(round(num_layers * 0.75))),
        num_layers,
    )


class DINOv2AdapterDualHead(BaseStage2DualHead):
    """Arm 4 / 5 / 9: DINOv2 ViT with Spatial Prior Stem, Per-Stage LayerNorm FPN & GAP cls_head."""

    def __init__(
        self,
        backbone_id: str = "facebook/dinov2-base",
        num_classes: int = 4,
        pretrained: bool = True,
        unfreeze_last_n_blocks: int = 4,
        tiny_offline_config: bool = False,
        arch: str = "dinov2_vitb14_adapter",
        fpn_dim: int = 128,
        internal_scale: float = 1.0,
        drop_path_rate: Optional[float] = None,
        track: str = "track_a_clean",
        tcga_pretrained: bool = False,
    ) -> None:
        super().__init__()
        if float(internal_scale) <= 0.0:
            raise ValueError(f"internal_scale must be > 0, got {internal_scale}.")
        if int(fpn_dim) <= 0 or int(fpn_dim) % 16 != 0:
            raise ValueError(f"fpn_dim must be a positive multiple of 16, got {fpn_dim}.")
        self.arch = str(arch)
        self.backbone_id = str(backbone_id)
        self.num_classes = int(num_classes)
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)
        self.tiny_offline_config = bool(tiny_offline_config or not pretrained)
        self.fpn_dim = int(fpn_dim)
        self.internal_scale = float(internal_scale)
        self.drop_path_rate = float(drop_path_rate) if drop_path_rate is not None else None
        self.track = str(track)
        self.tcga_pretrained = bool(tcga_pretrained)

        if pretrained:
            fp_kwargs: Dict[str, Any] = {"output_loading_info": True}
            if self.drop_path_rate is not None:
                fp_kwargs["drop_path_rate"] = self.drop_path_rate
            model_and_info = Dinov2Model.from_pretrained(
                self.backbone_id,
                **fp_kwargs,
            )
            if isinstance(model_and_info, tuple):
                self.backbone, loading_info = model_and_info
            else:
                self.backbone, loading_info = model_and_info, {}
            missing_keys = set(loading_info.get("missing_keys", []))
            all_keys = list(self.backbone.state_dict().keys())
            matched = len(all_keys) - len(missing_keys)
            if matched < 200:
                raise RuntimeError(
                    f"[STAGE2 ERROR] DINOv2 matched only {matched} keys "
                    f"(expected >= 200) for {self.backbone_id}."
                )
        else:
            patch_sz = 16 if ("vitl16" in self.arch.lower() or "phikon" in self.backbone_id.lower()) else 14
            cfg_kwargs: Dict[str, Any] = {
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "intermediate_size": 128,
                "image_size": 224,
                "patch_size": patch_sz,
            }
            if self.drop_path_rate is not None:
                cfg_kwargs["drop_path_rate"] = self.drop_path_rate
            cfg = Dinov2Config(**cfg_kwargs)
            self.backbone = Dinov2Model(cfg)

        self.patch_size = int(getattr(self.backbone.config, "patch_size", 14))
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.num_prefix_tokens = 1 + self.num_register_tokens
        num_layers = int(self.backbone.config.num_hidden_layers)
        self.stage_indices = _compute_evenly_spaced_stages(num_layers)
        self._supports_interp_pos = self._probe_interp_pos_support()

        self._apply_freeze_policy()

        self.stage_norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(4)])
        self.stage_projs = nn.ModuleList(
            [nn.Conv2d(self.hidden_size, self.fpn_dim, kernel_size=1, bias=False) for _ in range(4)]
        )
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, self.fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, self.fpn_dim),
            nn.GELU(),
        )
        self.fpn_refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(16, self.fpn_dim),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        self.decode_head = nn.Sequential(
            nn.Conv2d(self.fpn_dim * 2, self.fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, self.fpn_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.fpn_dim, self.num_classes, kernel_size=1),
        )
        self.cls_head = nn.Linear(self.hidden_size, self.num_classes)
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.cls_head.bias)

    def _probe_interp_pos_support(self) -> bool:
        """Probe runtime support for interpolate_pos_encoding without inspect.signature."""
        first_param = next(self.backbone.parameters(), None)
        probe_dtype = first_param.dtype if first_param is not None else torch.float32
        probe_device = first_param.device if first_param is not None else torch.device("cpu")
        probe = torch.zeros(
            1, 3, self.patch_size * 2, self.patch_size * 2, dtype=probe_dtype, device=probe_device
        )
        with torch.no_grad():
            try:
                self.backbone(pixel_values=probe, interpolate_pos_encoding=True)
                return True
            except (TypeError, ValueError):
                return False

    def _apply_freeze_policy(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = self.backbone.encoder.layer
        n_unfreeze = max(0, min(self.unfreeze_last_n_blocks, len(layers)))
        if n_unfreeze > 0:
            for block in layers[-n_unfreeze:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True) -> "DINOv2AdapterDualHead":
        super().train(mode)
        if mode:
            self.backbone.embeddings.eval()
            layers = self.backbone.encoder.layer
            n_frozen = max(0, len(layers) - max(0, min(self.unfreeze_last_n_blocks, len(layers))))
            for block in layers[:n_frozen]:
                block.eval()
        return self

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        ps = self.patch_size
        pad_h = (ps - (h % ps)) % ps
        pad_w = (ps - (w % ps)) % ps
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x_pad, h, w

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig_h, orig_w = x.shape[2:]
        scale_f = float(self.internal_scale)
        do_scale = abs(scale_f - 1.0) > 1e-6
        if do_scale:
            scaled_h = max(1, int(round(orig_h * scale_f)))
            scaled_w = max(1, int(round(orig_w * scale_f)))
            x_work = F.interpolate(x, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
        else:
            scaled_h, scaled_w = orig_h, orig_w
            x_work = x
        x_pad, _, _ = self._pad_to_patch_multiple(x_work)
        b, _, pad_h, pad_w = x_pad.shape
        h_tok = pad_h // self.patch_size
        w_tok = pad_w // self.patch_size

        vit_kwargs: Dict[str, Any] = {"pixel_values": x_pad, "output_hidden_states": True}
        if self._supports_interp_pos:
            vit_kwargs["interpolate_pos_encoding"] = True
        out = self.backbone(**vit_kwargs)

        # cls_head consumes post-layernorm last_hidden_state so backbone.layernorm receives grads
        last_spatial = out.last_hidden_state[:, self.num_prefix_tokens :, :]
        cls_feat = last_spatial.mean(dim=1)
        cls_logits = self.cls_head(cls_feat)

        stem_feat = self.spatial_stem(x_pad)  # (B, fpn_dim, pad_h//4, pad_w//4)
        base_h, base_w = stem_feat.shape[-2:]
        pyramid_sizes = [
            (base_h, base_w),
            (max(1, base_h // 2), max(1, base_w // 2)),
            (max(1, base_h // 4), max(1, base_w // 4)),
            (max(1, base_h // 8), max(1, base_w // 8)),
        ]

        stage_feats: List[torch.Tensor] = []
        for idx, norm, proj, target_sz in zip(
            self.stage_indices, self.stage_norms, self.stage_projs, pyramid_sizes
        ):
            tokens = out.hidden_states[idx][:, self.num_prefix_tokens :, :]
            tokens = norm(tokens)
            spatial = tokens.transpose(1, 2).contiguous().view(b, self.hidden_size, h_tok, w_tok)
            proj_feat = proj(spatial)
            if proj_feat.shape[-2:] != target_sz:
                proj_feat = F.interpolate(
                    proj_feat, size=target_sz, mode="bilinear", align_corners=False
                )
            stage_feats.append(proj_feat)

        # Progressive top-down FPN fusion into H_pad/4 resolution
        fused = self.fpn_refine[3](stage_feats[3])
        for level in (2, 1, 0):
            up = F.interpolate(
                fused, size=stage_feats[level].shape[-2:], mode="bilinear", align_corners=False
            )
            fused = self.fpn_refine[level](stage_feats[level] + up)

        combined = torch.cat([fused, stem_feat], dim=1)
        logits_pad = F.interpolate(
            self.decode_head(combined),
            size=(pad_h, pad_w),
            mode="bilinear",
            align_corners=False,
        )
        logits_scaled = logits_pad[:, :, :scaled_h, :scaled_w]
        if do_scale:
            seg_logits = F.interpolate(
                logits_scaled,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            seg_logits = logits_scaled
        return {"seg_logits": seg_logits, "cls_logits": cls_logits}

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.backbone.parameters() if p.requires_grad)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.stage_norms.parameters(),
            self.stage_projs.parameters(),
            self.spatial_stem.parameters(),
            self.fpn_refine.parameters(),
            self.decode_head.parameters(),
            self.cls_head.parameters(),
        )

    def model_metadata(self) -> Dict[str, Any]:
        meta = super().model_metadata()
        meta["model_kwargs"] = {
            "arch": self.arch,
            "backbone_id": self.backbone_id,
            "num_classes": self.num_classes,
            "unfreeze_last_n_blocks": self.unfreeze_last_n_blocks,
            "tiny_offline_config": self.tiny_offline_config,
            "fpn_dim": self.fpn_dim,
            "internal_scale": self.internal_scale,
            "drop_path_rate": self.drop_path_rate,
            "track": self.track,
            "tcga_pretrained": self.tcga_pretrained,
        }
        return meta


class Stage2ModelEMA:
    """Clean Exponential Moving Average (EMA) wrapper for Stage-2 v2 models.

    Tracks trainable floating-point parameters via EMA and copies frozen params/buffers,
    exporting strict=True clean model state dicts with zero 2x checkpoint bloat.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.trainable_names = trainable_names
        d = self.decay
        for k, v in model.state_dict().items():
            v_det = v.detach()
            if k not in self.shadow:
                self.shadow[k] = v_det.clone()
                continue
            target = self.shadow[k]
            v_aligned = v_det.to(device=target.device, dtype=target.dtype)
            if k in trainable_names and v_det.is_floating_point():
                target.mul_(d).add_(v_aligned, alpha=1.0 - d)
            else:
                target.copy_(v_aligned)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    @contextlib.contextmanager
    def swap_to_ema(self, model: nn.Module):
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            self.apply_to(model)
            yield model
        finally:
            model.load_state_dict(raw_state, strict=True)

    def clean_model_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def state_dict(
        self,
        trainable_only: bool = False,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        if trainable_only:
            t_names = (
                {n for n, p in model.named_parameters() if p.requires_grad}
                if model is not None
                else self.trainable_names
            )
            shadow_out = {
                k: v.detach().cpu().clone()
                for k, v in self.shadow.items()
                if k in t_names
            }
            return {
                "decay": self.decay,
                "trainable_only": True,
                "shadow": shadow_out,
            }
        return {
            "decay": self.decay,
            "trainable_only": False,
            "shadow": self.clean_model_state_dict(),
        }

    def sync_non_loaded_from_model(
        self,
        model: nn.Module,
        loaded_keys: Optional[Iterable[str]] = None,
    ) -> None:
        loaded_set = set(loaded_keys) if loaded_keys is not None else set()
        for k, v in model.state_dict().items():
            if k in loaded_set:
                continue
            v_det = v.detach()
            if k in self.shadow:
                self.shadow[k].copy_(
                    v_det.to(device=self.shadow[k].device, dtype=self.shadow[k].dtype)
                )
            else:
                self.shadow[k] = v_det.clone()

    def load_state_dict(
        self,
        state: Dict[str, Any],
        model: Optional[nn.Module] = None,
    ) -> None:
        if bool(state.get("trainable_only", False)) and model is None:
            raise ValueError(
                "Stage2ModelEMA.load_state_dict requires model=... when loading a trainable_only=True EMA state."
            )
        if "decay" in state:
            self.decay = float(state["decay"])
        loaded_shadow = state.get("shadow", state)
        if model is not None:
            self.sync_non_loaded_from_model(model, loaded_keys=loaded_shadow.keys())
        for k, v in loaded_shadow.items():
            if k in self.shadow:
                self.shadow[k].copy_(
                    v.to(device=self.shadow[k].device, dtype=self.shadow[k].dtype)
                )
            else:
                self.shadow[k] = v.detach().clone()


def build_stage2_v2_model(
    cfg: Stage2V2Config,
    pretrained: bool = True,
) -> BaseStage2DualHead:
    """Instantiate the Stage-2 v2 dual-head model specified by cfg."""
    arch = str(cfg.arch).strip().lower()
    if arch == "deeplabv3plus_r101":
        return DeepLabV3PlusDualHead(
            num_classes=cfg.num_classes,
            pretrained=pretrained,
            track=cfg.track,
            tcga_pretrained=cfg.tcga_pretrained,
        )
    if arch in ("segformer_b3", "segformer_b5"):
        return SegFormerDualHead(
            backbone_id=cfg.backbone_id,
            num_classes=cfg.num_classes,
            pretrained=pretrained,
            tiny_offline_config=not pretrained,
            arch=arch,
            drop_path_rate=cfg.drop_path_rate,
            track=cfg.track,
            tcga_pretrained=cfg.tcga_pretrained,
        )
    if arch in ("convnext_base_fpn", "convnext_large_fpn"):
        return ConvNeXtFPNDualHead(
            num_classes=cfg.num_classes,
            pretrained=pretrained,
            arch=arch,
            backbone_id=cfg.backbone_id,
            stochastic_depth_prob=cfg.stochastic_depth_prob,
            track=cfg.track,
            tcga_pretrained=cfg.tcga_pretrained,
        )
    if arch in ("dinov2_vitb14_adapter", "dinov2_vitl14_adapter", "dinov2_vitl16_adapter"):
        return DINOv2AdapterDualHead(
            backbone_id=cfg.backbone_id,
            num_classes=cfg.num_classes,
            pretrained=pretrained,
            unfreeze_last_n_blocks=cfg.unfreeze_last_n_blocks,
            tiny_offline_config=not pretrained,
            arch=arch,
            fpn_dim=cfg.fpn_dim,
            internal_scale=cfg.internal_scale,
            drop_path_rate=cfg.drop_path_rate,
            track=cfg.track,
            tcga_pretrained=cfg.tcga_pretrained,
        )
    raise ValueError(f"Unsupported Stage-2 v2 arch: {cfg.arch!r}.")


def remap_hf_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    target_keys: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Bidirectionally remaps HuggingFace transformers v4 <-> v5 keys for DINOv2 & SegFormer.

    Keys already present in target_keys are returned verbatim (no-op). Unmatched keys are
    tested against deterministic v5 -> v4 and v4 -> v5 rewrites and remapped only when the
    rewritten key exists in target_keys.
    """
    target_set = set(target_keys)
    if not target_set or set(state_dict.keys()).issubset(target_set):
        return dict(state_dict)

    def _to_v4(key: str) -> str:
        nk = key
        nk = re.sub(r"(^|\.)decode_head\.linear_projections\.", r"\1decode_head.linear_c.", nk)
        nk = re.sub(r"(^|\.)stages\.(\d+)\.patch_embeddings\.", r"\1encoder.patch_embeddings.\2.", nk)
        nk = re.sub(r"(^|\.)stages\.(\d+)\.layer_norm\.", r"\1encoder.layer_norm.\2.", nk)
        if re.search(r"(^|\.)stages\.\d+\.blocks\.\d+\.", nk):
            nk = re.sub(r"(^|\.)stages\.(\d+)\.blocks\.(\d+)\.", r"\1encoder.block.\2.\3.", nk)
            nk = nk.replace(".layernorm_before.", ".layer_norm_1.")
            nk = nk.replace(".layernorm_after.", ".layer_norm_2.")
            nk = nk.replace(".attention.q_proj.", ".attention.self.query.")
            nk = nk.replace(".attention.k_proj.", ".attention.self.key.")
            nk = nk.replace(".attention.v_proj.", ".attention.self.value.")
            nk = nk.replace(".attention.o_proj.", ".attention.output.dense.")
            nk = nk.replace(
                ".attention.sequence_reduction.sequence_reduction.",
                ".attention.self.sr.",
            )
            nk = nk.replace(
                ".attention.sequence_reduction.layer_norm.",
                ".attention.self.layer_norm.",
            )
            nk = nk.replace(".mlp.fc1.", ".mlp.dense1.")
            nk = nk.replace(".mlp.fc2.", ".mlp.dense2.")
        else:
            nk = nk.replace(".attention.q_proj.", ".attention.attention.query.")
            nk = nk.replace(".attention.k_proj.", ".attention.attention.key.")
            nk = nk.replace(".attention.v_proj.", ".attention.attention.value.")
            nk = nk.replace(".attention.o_proj.", ".attention.output.dense.")
        return nk

    def _to_v5(key: str) -> str:
        nk = key
        nk = re.sub(r"(^|\.)decode_head\.linear_c\.", r"\1decode_head.linear_projections.", nk)
        nk = re.sub(r"(^|\.)encoder\.patch_embeddings\.(\d+)\.", r"\1stages.\2.patch_embeddings.", nk)
        nk = re.sub(r"(^|\.)encoder\.layer_norm\.(\d+)\.", r"\1stages.\2.layer_norm.", nk)
        if re.search(r"(^|\.)encoder\.block\.\d+\.\d+\.", nk):
            nk = re.sub(r"(^|\.)encoder\.block\.(\d+)\.(\d+)\.", r"\1stages.\2.blocks.\3.", nk)
            nk = nk.replace(".layer_norm_1.", ".layernorm_before.")
            nk = nk.replace(".layer_norm_2.", ".layernorm_after.")
            nk = nk.replace(".attention.self.query.", ".attention.q_proj.")
            nk = nk.replace(".attention.self.key.", ".attention.k_proj.")
            nk = nk.replace(".attention.self.value.", ".attention.v_proj.")
            nk = nk.replace(".attention.output.dense.", ".attention.o_proj.")
            nk = nk.replace(
                ".attention.self.sr.",
                ".attention.sequence_reduction.sequence_reduction.",
            )
            nk = nk.replace(
                ".attention.self.layer_norm.",
                ".attention.sequence_reduction.layer_norm.",
            )
            nk = nk.replace(".mlp.dense1.", ".mlp.fc1.")
            nk = nk.replace(".mlp.dense2.", ".mlp.fc2.")
        else:
            nk = nk.replace(".attention.attention.query.", ".attention.q_proj.")
            nk = nk.replace(".attention.attention.key.", ".attention.k_proj.")
            nk = nk.replace(".attention.attention.value.", ".attention.v_proj.")
            nk = nk.replace(".attention.output.dense.", ".attention.o_proj.")
        return nk

    remapped: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k in target_set:
            remapped[k] = v
            continue
        cand_v4 = _to_v4(k)
        if cand_v4 in target_set and cand_v4 not in remapped and cand_v4 not in state_dict:
            remapped[cand_v4] = v
            continue
        cand_v5 = _to_v5(k)
        if cand_v5 in target_set and cand_v5 not in remapped and cand_v5 not in state_dict:
            remapped[cand_v5] = v
            continue
        remapped[k] = v
    return remapped


def load_stage2_v2_checkpoint(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Tuple[BaseStage2DualHead, Dict[str, Any]]:
    """Load a Stage-2 v2 checkpoint strictly with metadata verification."""
    ckpt_p = Path(ckpt_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Stage-2 v2 checkpoint not found: {ckpt_p}")

    dev = torch.device(device)
    payload = safe_torch_load(ckpt_p, device=torch.device("cpu"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload in {ckpt_p}: expected dict.")

    arch = str(payload.get("arch", "")).strip().lower()
    if not arch:
        raise RuntimeError(f"Checkpoint {ckpt_p} missing required 'arch' metadata.")
    backbone_id = str(payload.get("backbone_id", ""))
    num_classes = int(payload.get("num_classes", 4))
    model_kwargs = dict(payload.get("model_kwargs") or {})
    tiny_cfg = bool(model_kwargs.get("tiny_offline_config", False))
    track = str(model_kwargs.get("track") or payload.get("track") or "track_a_clean")
    tcga_pretrained = bool(model_kwargs.get("tcga_pretrained", payload.get("tcga_pretrained", False)))

    if arch == "deeplabv3plus_r101":
        model: BaseStage2DualHead = DeepLabV3PlusDualHead(
            num_classes=num_classes,
            pretrained=False,
            track=track,
            tcga_pretrained=tcga_pretrained,
        )
    elif arch in ("segformer_b3", "segformer_b5"):
        bb_default = "nvidia/mit-b5" if arch == "segformer_b5" else "nvidia/mit-b3"
        bb_resolved = str(model_kwargs.get("backbone_id") or backbone_id or bb_default)
        dp_raw = model_kwargs.get("drop_path_rate", None)
        dp_rate = float(dp_raw) if dp_raw is not None else None
        if tiny_cfg:
            model = SegFormerDualHead(
                backbone_id=bb_resolved,
                num_classes=num_classes,
                pretrained=False,
                tiny_offline_config=True,
                arch=arch,
                drop_path_rate=dp_rate,
                track=track,
                tcga_pretrained=tcga_pretrained,
            )
        else:
            fp_cfg_kw: Dict[str, Any] = {"num_labels": num_classes}
            if dp_rate is not None:
                fp_cfg_kw["drop_path_rate"] = dp_rate
            cfg_s = SegformerConfig.from_pretrained(bb_resolved, **fp_cfg_kw)
            model = SegFormerDualHead.__new__(SegFormerDualHead)
            BaseStage2DualHead.__init__(model)
            model.arch = arch
            model.backbone_id = bb_resolved
            model.num_classes = num_classes
            model.tiny_offline_config = False
            model.drop_path_rate = dp_rate
            model.track = track
            model.tcga_pretrained = tcga_pretrained
            model.segformer = SegformerForSemanticSegmentation(cfg_s)
            c4 = int(model.segformer.config.hidden_sizes[-1])
            model.cls_head = nn.Linear(c4, num_classes)
    elif arch in ("convnext_base_fpn", "convnext_large_fpn"):
        bb_default = "convnext_large" if arch == "convnext_large_fpn" else "convnext_base"
        bb_id = str(model_kwargs.get("backbone_id") or backbone_id or bb_default)
        sd_prob = float(model_kwargs.get("stochastic_depth_prob", 0.2))
        model = ConvNeXtFPNDualHead(
            num_classes=num_classes,
            pretrained=False,
            arch=arch,
            backbone_id=bb_id,
            stochastic_depth_prob=sd_prob,
            track=track,
            tcga_pretrained=tcga_pretrained,
        )
        if "in_channels_list" in model_kwargs:
            expected_ch = [int(c) for c in model_kwargs["in_channels_list"]]
            if expected_ch != list(model.in_channels_list):
                raise RuntimeError(
                    f"ConvNeXt channel mismatch in {ckpt_p}: {expected_ch} vs {model.in_channels_list}"
                )
    elif arch in ("dinov2_vitb14_adapter", "dinov2_vitl14_adapter", "dinov2_vitl16_adapter"):
        bb_map = {
            "dinov2_vitb14_adapter": "facebook/dinov2-base",
            "dinov2_vitl14_adapter": "facebook/dinov2-large",
            "dinov2_vitl16_adapter": "owkin/phikon-v2",
        }
        bb_resolved = str(model_kwargs.get("backbone_id") or backbone_id or bb_map[arch])
        unfreeze_default = 8 if "vitl" in arch else 4
        unfreeze_n = int(model_kwargs.get("unfreeze_last_n_blocks", unfreeze_default))
        fpn_dim = int(model_kwargs.get("fpn_dim", 128))
        internal_scale = float(model_kwargs.get("internal_scale", 1.0))
        dp_raw = model_kwargs.get("drop_path_rate", None)
        dp_rate = float(dp_raw) if dp_raw is not None else None
        if tiny_cfg:
            model = DINOv2AdapterDualHead(
                backbone_id=bb_resolved,
                num_classes=num_classes,
                pretrained=False,
                unfreeze_last_n_blocks=unfreeze_n,
                tiny_offline_config=True,
                arch=arch,
                fpn_dim=fpn_dim,
                internal_scale=internal_scale,
                drop_path_rate=dp_rate,
                track=track,
                tcga_pretrained=tcga_pretrained,
            )
        else:
            cfg_d_kw: Dict[str, Any] = {}
            if dp_rate is not None:
                cfg_d_kw["drop_path_rate"] = dp_rate
            cfg_d = Dinov2Config.from_pretrained(bb_resolved, **cfg_d_kw)
            model = DINOv2AdapterDualHead.__new__(DINOv2AdapterDualHead)
            BaseStage2DualHead.__init__(model)
            model.arch = arch
            model.backbone_id = bb_resolved
            model.num_classes = num_classes
            model.unfreeze_last_n_blocks = unfreeze_n
            model.tiny_offline_config = False
            model.fpn_dim = fpn_dim
            model.internal_scale = internal_scale
            model.drop_path_rate = dp_rate
            model.track = track
            model.tcga_pretrained = tcga_pretrained
            model.backbone = Dinov2Model(cfg_d)
            model.patch_size = int(getattr(cfg_d, "patch_size", 14))
            model.hidden_size = int(cfg_d.hidden_size)
            model.num_register_tokens = int(getattr(cfg_d, "num_register_tokens", 0))
            model.num_prefix_tokens = 1 + model.num_register_tokens
            model.stage_indices = _compute_evenly_spaced_stages(int(cfg_d.num_hidden_layers))
            model._supports_interp_pos = model._probe_interp_pos_support()
            model._apply_freeze_policy()
            model.stage_norms = nn.ModuleList([nn.LayerNorm(model.hidden_size) for _ in range(4)])
            model.stage_projs = nn.ModuleList(
                [nn.Conv2d(model.hidden_size, model.fpn_dim, 1, bias=False) for _ in range(4)]
            )
            model.spatial_stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(16, 128),
                nn.GELU(),
                nn.Conv2d(128, model.fpn_dim, 3, stride=1, padding=1, bias=False),
                nn.GroupNorm(16, model.fpn_dim),
                nn.GELU(),
            )
            model.fpn_refine = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(model.fpn_dim, model.fpn_dim, 3, padding=1, bias=False),
                        nn.GroupNorm(16, model.fpn_dim),
                        nn.GELU(),
                    )
                    for _ in range(4)
                ]
            )
            model.decode_head = nn.Sequential(
                nn.Conv2d(model.fpn_dim * 2, model.fpn_dim, 3, padding=1, bias=False),
                nn.GroupNorm(16, model.fpn_dim),
                nn.GELU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(model.fpn_dim, num_classes, 1),
            )
            model.cls_head = nn.Linear(model.hidden_size, num_classes)
    else:
        raise ValueError(f"Unsupported arch in checkpoint {ckpt_p}: {arch!r}.")

    raw_sd = payload.get("state_dict", payload.get("model", payload))
    cleaned_sd: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        ck = str(k)
        while ck.startswith("module.") or ck.startswith("model."):
            ck = ck[7:] if ck.startswith("module.") else ck[6:]
        if isinstance(v, torch.Tensor) and v.is_floating_point() and not torch.isfinite(v).all():
            raise ValueError(f"Non-finite tensor {ck!r} in checkpoint {ckpt_p}.")
        cleaned_sd[ck] = v

    cleaned_sd = remap_hf_state_dict_keys(cleaned_sd, model.state_dict().keys())
    model.load_state_dict(cleaned_sd, strict=strict)
    model.to(dev).eval()

    meta = {
        "arch": arch,
        "backbone_id": model.backbone_id,
        "num_classes": num_classes,
        "input_norm": payload.get("input_norm", "imagenet"),
        "has_cls_head": bool(payload.get("has_cls_head", True)),
        "stage1_sha256": payload.get("stage1_sha256", ""),
        "memmap_source": payload.get("memmap_source", ""),
        "r2_manifest_sha256": payload.get("r2_manifest_sha256", ""),
        "track": track,
        "tcga_pretrained": tcga_pretrained,
        "model_kwargs": model.model_metadata().get("model_kwargs", {}),
        "seed": payload.get("seed"),
        "preset": payload.get("preset"),
        "selected_weights": payload.get("selected_weights"),
        "epoch": payload.get("epoch"),
        "val_miou": payload.get("val_miou"),
        "non_finite_tensors": 0,
        "math_pi_check": math.pi > 3.0,
    }
    return model, meta
