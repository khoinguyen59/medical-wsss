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
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

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
        }


class DeepLabV3PlusDualHead(BaseStage2DualHead):
    """Arm 1: DeepLabV3+ ResNet-101 (OS=16) with decoupled GAP classification head."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.arch = "deeplabv3plus_r101"
        self.backbone_id = "resnet101"
        self.num_classes = int(num_classes)
        self.pretrained_init = bool(pretrained)

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
        meta["model_kwargs"] = {"num_classes": self.num_classes}
        return meta


class SegFormerDualHead(BaseStage2DualHead):
    """Arm 2: SegFormer-B3 (MiT-B3) with stage-4 GAP classification head."""

    def __init__(
        self,
        backbone_id: str = "nvidia/mit-b3",
        num_classes: int = 4,
        pretrained: bool = True,
        tiny_offline_config: bool = False,
    ) -> None:
        super().__init__()
        self.arch = "segformer_b3"
        self.backbone_id = str(backbone_id)
        self.num_classes = int(num_classes)
        self.tiny_offline_config = bool(tiny_offline_config or not pretrained)

        if pretrained:
            model_and_info = SegformerForSemanticSegmentation.from_pretrained(
                self.backbone_id,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
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
            cfg = SegformerConfig(
                num_labels=self.num_classes,
                depths=[1, 1, 1, 1],
                hidden_sizes=[16, 32, 64, 128],
                decoder_hidden_size=32,
                num_attention_heads=[1, 2, 4, 8],
                sr_ratios=[8, 4, 2, 1],
            )
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
            "backbone_id": self.backbone_id,
            "num_classes": self.num_classes,
            "tiny_offline_config": self.tiny_offline_config,
        }
        return meta


class ConvNeXtFPNDualHead(BaseStage2DualHead):
    """Arm 3: ConvNeXt-Base (stochastic_depth_prob=0.2) + GroupNorm FPN + GAP cls_head."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.arch = "convnext_base_fpn"
        self.backbone_id = "convnext_base"
        self.num_classes = int(num_classes)
        self.stochastic_depth_prob = 0.2

        weights = tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.convnext_base(
            weights=weights,
            stochastic_depth_prob=self.stochastic_depth_prob,
        )
        self.features = backbone.features
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 1024],
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
        self.cls_head = nn.Linear(1024, self.num_classes)
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
        meta["model_kwargs"] = {"num_classes": self.num_classes}
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
    """Arm 4: DINOv2 ViT-B/14 with Spatial Prior Stem, Per-Stage LayerNorm FPN & GAP cls_head."""

    def __init__(
        self,
        backbone_id: str = "facebook/dinov2-base",
        num_classes: int = 4,
        pretrained: bool = True,
        unfreeze_last_n_blocks: int = 4,
        tiny_offline_config: bool = False,
    ) -> None:
        super().__init__()
        self.arch = "dinov2_vitb14_adapter"
        self.backbone_id = str(backbone_id)
        self.num_classes = int(num_classes)
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)
        self.tiny_offline_config = bool(tiny_offline_config or not pretrained)

        if pretrained:
            model_and_info = Dinov2Model.from_pretrained(
                self.backbone_id,
                output_loading_info=True,
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
            cfg = Dinov2Config(
                hidden_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=128,
                image_size=224,
                patch_size=14,
            )
            self.backbone = Dinov2Model(cfg)

        self.patch_size = int(getattr(self.backbone.config, "patch_size", 14))
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.num_prefix_tokens = 1 + self.num_register_tokens
        num_layers = int(self.backbone.config.num_hidden_layers)
        self.stage_indices = _compute_evenly_spaced_stages(num_layers)
        self._supports_interp_pos = self._probe_interp_pos_support()

        self._apply_freeze_policy()

        self.fpn_dim = 128
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
        probe = torch.zeros(1, 3, self.patch_size * 2, self.patch_size * 2)
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
        x_pad, orig_h, orig_w = self._pad_to_patch_multiple(x)
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
        seg_logits = logits_pad[:, :, :orig_h, :orig_w]
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
            "backbone_id": self.backbone_id,
            "num_classes": self.num_classes,
            "unfreeze_last_n_blocks": self.unfreeze_last_n_blocks,
            "tiny_offline_config": self.tiny_offline_config,
        }
        return meta


class Stage2ModelEMA:
    """Clean Exponential Moving Average (EMA) wrapper for Stage-2 v2 models.

    Tracks trainable floating-point parameters via EMA and copies frozen params/buffers,
    exporting strict=True clean model state dicts with zero 2x checkpoint bloat.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
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

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.clean_model_state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "decay" in state:
            self.decay = float(state["decay"])
        loaded_shadow = state.get("shadow", state)
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
        return DeepLabV3PlusDualHead(num_classes=cfg.num_classes, pretrained=pretrained)
    if arch == "segformer_b3":
        return SegFormerDualHead(
            backbone_id=cfg.backbone_id,
            num_classes=cfg.num_classes,
            pretrained=pretrained,
            tiny_offline_config=not pretrained,
        )
    if arch == "convnext_base_fpn":
        return ConvNeXtFPNDualHead(num_classes=cfg.num_classes, pretrained=pretrained)
    if arch == "dinov2_vitb14_adapter":
        return DINOv2AdapterDualHead(
            backbone_id=cfg.backbone_id,
            num_classes=cfg.num_classes,
            pretrained=pretrained,
            unfreeze_last_n_blocks=cfg.unfreeze_last_n_blocks,
            tiny_offline_config=not pretrained,
        )
    raise ValueError(f"Unsupported Stage-2 v2 arch: {cfg.arch!r}.")


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

    if arch == "deeplabv3plus_r101":
        model: BaseStage2DualHead = DeepLabV3PlusDualHead(
            num_classes=num_classes,
            pretrained=False,
        )
    elif arch == "segformer_b3":
        if tiny_cfg:
            model = SegFormerDualHead(
                backbone_id=backbone_id or "nvidia/mit-b3",
                num_classes=num_classes,
                pretrained=False,
                tiny_offline_config=True,
            )
        else:
            cfg = SegformerConfig.from_pretrained(
                backbone_id or "nvidia/mit-b3",
                num_labels=num_classes,
            )
            model = SegFormerDualHead.__new__(SegFormerDualHead)
            BaseStage2DualHead.__init__(model)
            model.arch = "segformer_b3"
            model.backbone_id = backbone_id or "nvidia/mit-b3"
            model.num_classes = num_classes
            model.tiny_offline_config = False
            model.segformer = SegformerForSemanticSegmentation(cfg)
            c4 = int(model.segformer.config.hidden_sizes[-1])
            model.cls_head = nn.Linear(c4, num_classes)
    elif arch == "convnext_base_fpn":
        model = ConvNeXtFPNDualHead(num_classes=num_classes, pretrained=False)
    elif arch == "dinov2_vitb14_adapter":
        unfreeze_n = int(model_kwargs.get("unfreeze_last_n_blocks", 4))
        if tiny_cfg:
            model = DINOv2AdapterDualHead(
                backbone_id=backbone_id or "facebook/dinov2-base",
                num_classes=num_classes,
                pretrained=False,
                unfreeze_last_n_blocks=unfreeze_n,
                tiny_offline_config=True,
            )
        else:
            cfg_d = Dinov2Config.from_pretrained(backbone_id or "facebook/dinov2-base")
            model = DINOv2AdapterDualHead.__new__(DINOv2AdapterDualHead)
            BaseStage2DualHead.__init__(model)
            model.arch = "dinov2_vitb14_adapter"
            model.backbone_id = backbone_id or "facebook/dinov2-base"
            model.num_classes = num_classes
            model.unfreeze_last_n_blocks = unfreeze_n
            model.tiny_offline_config = False
            model.backbone = Dinov2Model(cfg_d)
            model.patch_size = int(getattr(cfg_d, "patch_size", 14))
            model.hidden_size = int(cfg_d.hidden_size)
            model.num_register_tokens = int(getattr(cfg_d, "num_register_tokens", 0))
            model.num_prefix_tokens = 1 + model.num_register_tokens
            model.stage_indices = _compute_evenly_spaced_stages(int(cfg_d.num_hidden_layers))
            model._supports_interp_pos = model._probe_interp_pos_support()
            model._apply_freeze_policy()
            model.fpn_dim = 128
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

    model.load_state_dict(cleaned_sd, strict=strict)
    model.to(dev).eval()

    meta = {
        "arch": arch,
        "backbone_id": backbone_id,
        "num_classes": num_classes,
        "input_norm": payload.get("input_norm", "imagenet"),
        "has_cls_head": bool(payload.get("has_cls_head", True)),
        "stage1_sha256": payload.get("stage1_sha256", ""),
        "seed": payload.get("seed"),
        "preset": payload.get("preset"),
        "selected_weights": payload.get("selected_weights"),
        "epoch": payload.get("epoch"),
        "val_miou": payload.get("val_miou"),
        "non_finite_tensors": 0,
        "math_pi_check": math.pi > 3.0,
    }
    return model, meta
