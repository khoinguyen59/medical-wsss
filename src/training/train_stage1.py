"""
Stage-1 Multi-Label Classification Training Engine for Histopathology WSSS.

Trains the 105M-parameter ResNet-38d backbone (WSSSTissueStage1NetCAM) with:
1. Pretrained ResNet-38d weight initialization (ImageNet .params/.pth or cross-organ
   histopathology domain-pretrained backbone) with explicit layer count logging and
   anti-silent-random-init guard (Rule 17).
2. Progressive Dropout Attention (PDA) activation schedule (Han et al., MedIA 2022).
3. Baseline-parity MultiLabelSoftMarginLoss default (with optional calibrated AsymmetricLoss).
4. Independent validation loop after every epoch measuring val mAP, Macro-AUROC,
   Best-Threshold Macro-F1, and val Gated Pseudo-Mask mIoU (Rule 17 Anti-Blind-Training).
5. AdamW optimizer with decoupled layer-wise learning rates (10x lr on fc8 projection head)
   and Cosine Annealing Learning Rate Scheduler.
6. Deterministic checkpoint saving to checkpoints/stage1_retrained_best.pth selected strictly
   by validation metric with explicit 'input_norm' metadata for Stage-2 alignment.
7. Zero patient-level leakage enforcement (Invariant 08).
8. Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

import argparse
import contextlib
import math
import random
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config.stage1_experiments import (
    Stage1ExperimentConfig,
    get_stage1_preset,
    resolve_stage1_config_from_env,
)
from src.losses_metrics.losses import MultiPrototypePPCLoss
from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.training.dataset_loader import (
    NUM_FOREGROUND,
    create_stage1_dataloaders,
)

HF_LUAD_DOMAIN_PRETRAINED_URL: str = (
    "https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/"
    "stage1_checkpoint_trained_on_luad.pth"
)


def convert_mxnet_params_to_resnet38d_state_dict(params_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Convert MXNet ResNet-38d .params file (ilsvrc-cls_rna-a1_cls1000_ep-0001.params) to PyTorch state_dict.

    Matches the canonical key renaming in repos/WSSS-Tissue/network/resnet38d.py.
    """
    try:
        import mxnet  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading raw MXNet .params requires the 'mxnet' package, which is not installed. "
            "Please provide a converted PyTorch .pth weight file (e.g., resnet38d_ilsvrc.pth) "
            "or enable cross-organ domain-pretrained .pth initialization."
        ) from exc

    save_dict = mxnet.nd.load(str(params_path))
    renamed_dict: Dict[str, torch.Tensor] = {}
    bn_param_mx_pt = {"beta": "bias", "gamma": "weight", "mean": "running_mean", "var": "running_var"}

    for k, v in save_dict.items():
        tensor_v = torch.from_numpy(v.asnumpy())
        toks = k.split("_")
        if "conv1a" in toks[0]:
            renamed_dict["conv1a.weight"] = tensor_v
        elif "linear1000" in toks[0]:
            continue
        elif len(toks) > 1 and "branch" in toks[1]:
            pt_name: List[str] = []
            if toks[0][-1] != "a":
                pt_name.append("b" + toks[0][-3] + "_" + toks[0][-1])
            else:
                pt_name.append("b" + toks[0][-2])
            if "res" in toks[0]:
                layer_type = "conv"
                last_name = "weight"
            else:
                layer_type = "bn"
                last_name = bn_param_mx_pt[toks[-1]]
            pt_name.append(layer_type + "_" + toks[1])
            pt_name.append(last_name)
            renamed_dict[".".join(pt_name)] = tensor_v
        elif toks[-1] in bn_param_mx_pt:
            renamed_dict["bn7." + bn_param_mx_pt[toks[-1]]] = tensor_v

    return renamed_dict


def load_pretrained_resnet38d_weights(
    model: WSSSTissueStage1NetCAM,
    pretrained_path: Optional[Union[str, Path]] = None,
    allow_random_init: bool = False,
    reset_classifier_head: bool = True,
    min_loaded_layers: int = 100,
    auto_download_domain_pretrained: bool = False,
    cache_dir: Union[str, Path] = "checkpoints",
) -> Dict[str, Any]:
    """Load pretrained ResNet-38d weights with explicit layer count audit and anti-silent-fallback gate.

    Search order when pretrained_path is None:
    1. init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
    2. init_weights/resnet38d_ilsvrc.pth
    3. <cache_dir>/resnet38d_ilsvrc.pth
    4. <cache_dir>/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth
    5. <cache_dir>/stage1_checkpoint_trained_on_luad.pth (cross-organ LUAD histopathology backbone, fc8 reset)
    6. Optional auto-download of stage1_checkpoint_trained_on_luad.pth when auto_download_domain_pretrained=True.
    """
    model_dict = model.state_dict()
    total_backbone_layers = sum(1 for k in model_dict if not k.startswith("fc8."))
    resolved_path: Optional[Path] = None
    source_desc = "random_init"
    cache_path = Path(cache_dir)

    if pretrained_path is not None:
        cand = Path(pretrained_path)
        if not cand.is_file():
            if allow_random_init:
                warnings.warn(
                    f"[STAGE 1 WARN] Requested pretrained weights not found at {cand}; "
                    "proceeding with random initialization (allow_random_init=True).",
                    UserWarning,
                    stacklevel=2,
                )
                return {
                    "loaded_layers": 0,
                    "total_backbone_layers": total_backbone_layers,
                    "source": "random_init",
                    "path": None,
                }
            raise FileNotFoundError(
                f"[STAGE 1 ERROR] Pretrained ResNet-38d weights not found at {cand}. "
                "Refusing silent random-init training (Rule 17)."
            )
        resolved_path = cand
        source_desc = f"explicit:{cand.name}"
    else:
        candidates = [
            Path("init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params"),
            Path("init_weights/resnet38d_ilsvrc.pth"),
            cache_path / "resnet38d_ilsvrc.pth",
            cache_path / "ilsvrc-cls_rna-a1_cls1000_ep-0001.pth",
            cache_path / "stage1_checkpoint_trained_on_luad.pth",
            Path("checkpoints/stage1_checkpoint_trained_on_luad.pth"),
        ]
        for cand in candidates:
            if cand.is_file() and cand.stat().st_size > 1_000_000:
                resolved_path = cand
                source_desc = f"local:{cand.name}"
                break

        if resolved_path is None and auto_download_domain_pretrained:
            dl_target = cache_path / "stage1_checkpoint_trained_on_luad.pth"
            dl_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                print(
                    f"[STAGE 1] Downloading cross-organ LUAD histopathology pretrained ResNet-38d "
                    f"backbone (fc8 will be reset) -> {dl_target} ..."
                )
                urllib.request.urlretrieve(HF_LUAD_DOMAIN_PRETRAINED_URL, str(dl_target))  # nosec B310
                if dl_target.is_file() and dl_target.stat().st_size > 1_000_000:
                    resolved_path = dl_target
                    source_desc = "domain_pretrained:luad_resnet38d_backbone"
            except Exception as exc:
                warnings.warn(
                    f"[STAGE 1 WARN] Failed to download domain-pretrained ResNet-38d ({exc}).",
                    UserWarning,
                    stacklevel=2,
                )

    if resolved_path is None:
        if allow_random_init:
            warnings.warn(
                "[STAGE 1 WARN] No pretrained ResNet-38d weights loaded (0 layers). "
                "Continuing with random initialization ONLY because allow_random_init=True.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "loaded_layers": 0,
                "total_backbone_layers": total_backbone_layers,
                "source": "random_init",
                "path": None,
            }
        raise RuntimeError(
            "[STAGE 1 ERROR] No pretrained ResNet-38d weights found (0 layers loaded). "
            "Refusing silent training from random weights on 105M-parameter ResNet-38d (Rule 17). "
            "Provide --pretrained-path (ilsvrc-cls_rna-a1_cls1000_ep-0001.params or .pth), "
            "enable auto_download_domain_pretrained=True, or pass --allow-random-init explicitly."
        )

    try:
        if resolved_path.suffix.lower() == ".params":
            raw_dict = convert_mxnet_params_to_resnet38d_state_dict(resolved_path)
        else:
            try:
                loaded_obj = torch.load(resolved_path, map_location="cpu", weights_only=True)
            except Exception:
                loaded_obj = torch.load(resolved_path, map_location="cpu", weights_only=False)  # nosec B614
            if isinstance(loaded_obj, dict):
                raw_dict = loaded_obj.get("state_dict", loaded_obj.get("model", loaded_obj))
            else:
                raw_dict = loaded_obj
    except Exception as exc:
        if allow_random_init:
            warnings.warn(
                f"[STAGE 1 WARN] Failed loading pretrained weights from {resolved_path} ({exc}); "
                "continuing with random init (allow_random_init=True).",
                UserWarning,
                stacklevel=2,
            )
            return {
                "loaded_layers": 0,
                "total_backbone_layers": total_backbone_layers,
                "source": "random_init",
                "path": None,
            }
        raise RuntimeError(
            f"[STAGE 1 ERROR] Failed to load pretrained ResNet-38d weights from {resolved_path}: {exc}"
        ) from exc

    filtered: Dict[str, torch.Tensor] = {}
    for k, v in raw_dict.items():
        clean_k = k.replace("module.", "").replace("model.", "")
        if reset_classifier_head and clean_k.startswith("fc8."):
            continue
        if clean_k in model_dict and isinstance(v, torch.Tensor) and v.shape == model_dict[clean_k].shape:
            filtered[clean_k] = v

    loaded_layers = len(filtered)
    if loaded_layers < min_loaded_layers:
        if allow_random_init:
            warnings.warn(
                f"[STAGE 1 WARN] Loaded only {loaded_layers}/{total_backbone_layers} ResNet-38d "
                f"layers from {resolved_path} (expected >= {min_loaded_layers}).",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise RuntimeError(
                f"[STAGE 1 ERROR] Loaded only {loaded_layers}/{total_backbone_layers} ResNet-38d "
                f"layers from {resolved_path} (expected >= {min_loaded_layers}). "
                "Refusing silent under-initialized training (Rule 17)."
            )

    if loaded_layers > 0:
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)

    print(
        f"[STAGE 1] Loaded {loaded_layers}/{total_backbone_layers} pretrained ResNet-38d layers "
        f"from {resolved_path} (source={source_desc}, reset_fc8={reset_classifier_head})."
    )
    return {
        "loaded_layers": loaded_layers,
        "total_backbone_layers": total_backbone_layers,
        "source": source_desc,
        "path": str(resolved_path),
    }


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute exact Average Precision (AP) for a single binary label vector."""
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = (y_true[order] == 1).astype(np.float64)
    tp_cum = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    precisions = tp_cum / ranks
    return float((precisions * y_sorted).sum() / float(n_pos))


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute exact ROC-AUC via average-rank Mann-Whitney U statistic."""
    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    scores_sorted = y_score[order]
    n = len(scores_sorted)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1
    orig_ranks = np.empty(n, dtype=np.float64)
    orig_ranks[order] = ranks
    sum_pos_ranks = float(orig_ranks[pos_mask].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_multilabel_validation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute threshold-invariant (mAP, Macro-AUROC, Best-Threshold Macro-F1) and @0.5 metrics."""
    y_true_bin = (y_true > 0.5).astype(np.int64)
    y_prob_f = y_prob.astype(np.float64)
    n_samples, n_classes = y_true_bin.shape

    preds_05 = (y_prob_f > 0.5).astype(np.int64)
    exact_match = float(np.all(preds_05 == y_true_bin, axis=1).mean()) if n_samples > 0 else 0.0

    f1_05_list: List[float] = []
    ap_list: List[float] = []
    auc_list: List[float] = []
    best_f1_list: List[float] = []
    best_thresh_list: List[float] = []
    thresholds = np.linspace(0.05, 0.95, 19)

    for c in range(n_classes):
        yt = y_true_bin[:, c]
        yp = y_prob_f[:, c]
        tp05 = float(np.logical_and(preds_05[:, c] == 1, yt == 1).sum())
        denom05 = float((preds_05[:, c] == 1).sum() + (yt == 1).sum())
        f1_05_list.append((2.0 * tp05 / denom05) if denom05 > 0 else 0.0)

        ap = _average_precision_binary(yt, yp)
        if ap is not None:
            ap_list.append(ap)
        auc = _roc_auc_binary(yt, yp)
        if auc is not None:
            auc_list.append(auc)

        best_c_f1 = 0.0
        best_c_thr = 0.5
        for thr in thresholds:
            pb = (yp >= thr).astype(np.int64)
            tp = float(np.logical_and(pb == 1, yt == 1).sum())
            den = float((pb == 1).sum() + (yt == 1).sum())
            f1_t = (2.0 * tp / den) if den > 0 else 0.0
            if f1_t > best_c_f1:
                best_c_f1 = f1_t
                best_c_thr = float(thr)
        best_f1_list.append(best_c_f1)
        best_thresh_list.append(round(best_c_thr, 2))

    return {
        "val_exact_match": exact_match,
        "val_macro_f1": float(np.mean(f1_05_list)) if f1_05_list else 0.0,
        "val_mAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "val_auroc": float(np.mean(auc_list)) if auc_list else 0.0,
        "val_best_f1": float(np.mean(best_f1_list)) if best_f1_list else 0.0,
        "val_best_thresholds": best_thresh_list,
        "val_f1_tum": f1_05_list[0] if len(f1_05_list) > 0 else 0.0,
        "val_f1_str": f1_05_list[1] if len(f1_05_list) > 1 else 0.0,
        "val_f1_lym": f1_05_list[2] if len(f1_05_list) > 2 else 0.0,
        "val_f1_nec": f1_05_list[3] if len(f1_05_list) > 3 else 0.0,
    }


class AttentionModule(nn.Module):
    """Progressive Dropout Attention (PDA) Module for ResNet-38d.

    Supports both legacy global-average mode (when active_labels=None) and
    Label-Gated Multiplicative Peak-Drop (when active_labels is provided),
    where only ground-truth active classes (y_c == 1) drop their spatial peaks
    via elementwise min so single-class peaks drop strictly to 0.0.
    """

    def __init__(self, pda_mode: str = "multiplicative") -> None:
        super().__init__()
        self.pda_mode = str(pda_mode or "multiplicative").strip().lower()

    def forward(
        self,
        x: torch.Tensor,
        fc_weights: torch.Tensor,
        gamma: float = 1.0,
        active_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: (B, 4096, H, W) feature map from bn7.
            fc_weights: (4, 4096, 1, 1) weights from fc8.
            gamma: Dropout threshold factor in [0.65, 1.0].
            active_labels: Optional (B, C) multi-hot binary labels for label-gated
                multiplicative peak-drop across active classes only.

        Returns:
            masked_x: (B, 4096, H, W) feature map modulated by bounded attention mask.
        """
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            gamma_f = float(gamma)
            cams = F.relu(F.conv2d(x_f, fc_weights.float().detach()))  # (B, C, H, W)
            b, c, h, w = cams.size()

            if gamma_f >= 1.0 or c == 0:
                return x_f

            ones = torch.ones((b, h, w), device=x_f.device, dtype=x_f.dtype)
            zeros = torch.zeros((b, h, w), device=x_f.device, dtype=x_f.dtype)

            if active_labels is not None and self.pda_mode == "multiplicative":
                labels_f = active_labels.to(device=x_f.device, dtype=x_f.dtype)
                drop_mask = ones.clone()
                for i in range(c):
                    sub_cam = cams[:, i, :, :]
                    sub_cam_max = torch.amax(sub_cam.view(b, -1), dim=-1).view(b, 1, 1)
                    cls_active = (labels_f[:, i].view(b, 1, 1) > 0.5) & (sub_cam_max > 1e-6)
                    thr = sub_cam_max * gamma_f
                    class_mask = torch.where(cls_active & (sub_cam > thr), zeros, ones)
                    drop_mask = torch.minimum(drop_mask, class_mask)
                mask = drop_mask.unsqueeze(1)
            else:
                mean_drop_mask = torch.zeros((b, h, w), device=x_f.device, dtype=x_f.dtype)
                for i in range(c):
                    sub_cam = cams[:, i, :, :]
                    sub_cam_max = torch.amax(sub_cam.view(b, -1), dim=-1).view(b, 1, 1)
                    active = sub_cam_max > 1e-6
                    thr = sub_cam_max * gamma_f
                    class_mask = torch.where(active & (sub_cam > thr), zeros, ones)
                    mean_drop_mask = mean_drop_mask + class_mask
                mask = (mean_drop_mask / float(c)).unsqueeze(1)

            scale = mask.mean(dim=(2, 3), keepdim=True).clamp(min=0.25)

            # Return FP32 with inverted-dropout rescaling to preserve expected feature magnitude
            return x_f * (mask / scale)


class ModelEMA:
    """Exponential Moving Average (EMA) shadow tracker for WSSSTissueStage1NetCAM.

    Tracks strictly the base model's state_dict keys so exported EMA weights
    are 100% strict=True compatible with WSSSTissueStage1NetCAM.
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
                self.shadow[k].copy_(v.to(device=self.shadow[k].device, dtype=self.shadow[k].dtype))
            else:
                self.shadow[k] = v.detach().clone()


class ProjectionHead(nn.Module):
    """Training-only 1x1 projection head for Arm C MP-PPC contrastive learning."""

    def __init__(
        self,
        in_dim: int = 4096,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_groups: int = 16,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label learning with negative margin shifting.

    Reference: Ridnik et al., ICCV 2021.
    Suppresses easy negative gradients to address severe tissue class imbalance.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: (B, C) unnormalized raw predictions.
            targets: (B, C) multi-hot binary ground truth in {0, 1}.
        """
        # Compute loss strictly in FP32 with autocast disabled so eps never underflows
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            logits_f = logits.float()
            targets_f = targets.float()
            eps = float(self.eps)
            probs = torch.sigmoid(logits_f)

            # Positive margin
            pos_probs = probs
            pos_loss = targets_f * torch.log(pos_probs.clamp(min=eps))
            if self.gamma_pos > 0:
                pos_weight = (1.0 - pos_probs).clamp(min=0.0).pow(self.gamma_pos)
                pos_loss = pos_loss * pos_weight

            # Negative margin with hard probability clipping
            neg_probs = (1.0 - probs).clamp(min=eps)
            if self.clip > 0:
                asym_probs = (probs - float(self.clip)).clamp(min=0.0)
                neg_loss = (1.0 - targets_f) * torch.log((1.0 - asym_probs).clamp(min=eps))
            else:
                neg_loss = (1.0 - targets_f) * torch.log(neg_probs)

            if self.gamma_neg > 0:
                neg_weight = probs.clamp(min=0.0).pow(self.gamma_neg)
                neg_loss = neg_loss * neg_weight

            total_loss = -(pos_loss + neg_loss)
            return total_loss.mean()


class Stage1Trainer:
    """Orchestrates Stage-1 ResNet38d classifier training and validation."""

    def __init__(
        self,
        model: WSSSTissueStage1NetCAM,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        enable_amp: bool = True,
        pda_start_epoch: int = 3,
        initial_gamma: float = 1.0,
        gamma_decay: float = 0.985,
        min_gamma: float = 0.65,
        pda_mode: str = "multiplicative",
        step_scheduler_per_batch: bool = False,
        model_ema: Optional[ModelEMA] = None,
        proj_head: Optional[ProjectionHead] = None,
        mpppc_loss: Optional[MultiPrototypePPCLoss] = None,
        lambda_ppc: float = 0.0,
        ppc_start_epoch: int = 3,
        use_equiv_consistency: bool = False,
        equiv_scales: Tuple[float, ...] = (1.0, 1.25),
        lambda_eq_max: float = 0.0,
        eq_ramp_start_epoch: int = 2,
        eq_ramp_end_epoch: int = 5,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.enable_amp = enable_amp and (self.device.type == "cuda")
        if self.enable_amp:
            if amp_dtype is not None:
                self.amp_dtype = amp_dtype
            else:
                try:
                    major_cc = torch.cuda.get_device_capability(self.device)[0] if torch.cuda.is_available() else 0
                except Exception:
                    major_cc = 0
                self.amp_dtype = (
                    torch.bfloat16
                    if (major_cc >= 8 and torch.cuda.is_bf16_supported(including_emulation=False))
                    else torch.float16
                )
        else:
            self.amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.enable_amp and self.amp_dtype == torch.float16),
        )

        self.attention_module = AttentionModule(pda_mode=pda_mode).to(self.device)
        self.pda_start_epoch = pda_start_epoch
        self.gamma = initial_gamma
        self.gamma_decay = gamma_decay
        self.min_gamma = min_gamma
        self.step_scheduler_per_batch = bool(step_scheduler_per_batch)
        self.model_ema = model_ema
        self.proj_head = proj_head.to(self.device) if proj_head is not None else None
        self.mpppc_loss = mpppc_loss.to(self.device) if mpppc_loss is not None else None
        self.lambda_ppc = float(lambda_ppc)
        self.ppc_start_epoch = int(ppc_start_epoch)
        self.use_equiv_consistency = bool(use_equiv_consistency)
        self.equiv_scales = tuple(float(s) for s in equiv_scales)
        self.lambda_eq_max = float(lambda_eq_max)
        self.eq_ramp_start_epoch = int(eq_ramp_start_epoch)
        self.eq_ramp_end_epoch = int(eq_ramp_end_epoch)

    @staticmethod
    def _apply_d4_tensor(
        tensor: torch.Tensor,
        k_rot: int,
        do_hflip: bool,
        do_vflip: bool,
    ) -> torch.Tensor:
        out = tensor
        if k_rot % 4 != 0:
            out = torch.rot90(out, k=k_rot % 4, dims=(-2, -1))
        if do_hflip:
            out = torch.flip(out, dims=[-1])
        if do_vflip:
            out = torch.flip(out, dims=[-2])
        return out

    @staticmethod
    def _inverse_d4_tensor(
        tensor: torch.Tensor,
        k_rot: int,
        do_hflip: bool,
        do_vflip: bool,
    ) -> torch.Tensor:
        out = tensor
        if do_vflip:
            out = torch.flip(out, dims=[-2])
        if do_hflip:
            out = torch.flip(out, dims=[-1])
        if k_rot % 4 != 0:
            out = torch.rot90(out, k=(4 - (k_rot % 4)) % 4, dims=(-2, -1))
        return out

    def get_lambda_eq(self, epoch: int) -> float:
        if not self.use_equiv_consistency or self.lambda_eq_max <= 0.0:
            return 0.0
        if epoch < self.eq_ramp_start_epoch:
            return 0.0
        if epoch >= self.eq_ramp_end_epoch:
            return self.lambda_eq_max
        span = max(self.eq_ramp_end_epoch - self.eq_ramp_start_epoch, 1)
        frac = float(epoch - self.eq_ramp_start_epoch) / float(span)
        return self.lambda_eq_max * frac

    def forward_dual_branches(
        self,
        images: torch.Tensor,
        enable_pda: bool,
        targets: Optional[torch.Tensor] = None,
        return_conv6: bool = False,
    ) -> Tuple[Any, ...]:
        """Forward pass returning clean logits/probs and optional PDA-masked logits/probs."""
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp):
            conv6 = self.model.forward_backbone(images)  # (B, 4096, 28, 28)

        with torch.amp.autocast(device_type=conv6.device.type, enabled=False):
            conv6_f = conv6.float()
            fc8_w = self.model.fc8.weight.float()

            x_clean = self.model.dropout7(conv6_f)
            x_clean = self.model.pool(x_clean)
            logits_clean = F.conv2d(x_clean, fc8_w).view(x_clean.size(0), -1)
            probs_clean = torch.sigmoid(logits_clean)

            if not enable_pda:
                if return_conv6:
                    return logits_clean, probs_clean, None, None, conv6_f
                return logits_clean, probs_clean, None, None

            conv6_pda = self.attention_module(conv6_f, fc8_w, self.gamma, active_labels=targets)
            x_pda = self.model.dropout7(conv6_pda)
            x_pda = self.model.pool(x_pda)
            logits_pda = F.conv2d(x_pda, fc8_w).view(x_pda.size(0), -1)
            probs_pda = torch.sigmoid(logits_pda)
            if return_conv6:
                return logits_clean, probs_clean, logits_pda, probs_pda, conv6_f
            return logits_clean, probs_clean, logits_pda, probs_pda

    def forward_with_pda(
        self,
        images: torch.Tensor,
        enable_pda: bool,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with conditional Progressive Dropout Attention."""
        logits_clean, probs_clean, logits_pda, probs_pda = self.forward_dual_branches(
            images, enable_pda=enable_pda, targets=targets
        )
        if enable_pda and logits_pda is not None and probs_pda is not None:
            return logits_pda, probs_pda
        return logits_clean, probs_clean

    def compute_equivariant_branch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        conv6_clean: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute transformed classification loss L_cls(x_t) and equivariant CAM loss L_eq."""
        b, _, h_in, w_in = images.shape
        scale = float(random.choice(self.equiv_scales)) if self.equiv_scales else 1.0
        k_rot = random.randint(0, 3)
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        if scale == 1.0 and k_rot == 0 and not do_hflip and not do_vflip:
            k_rot = 1

        x_t = self._apply_d4_tensor(images, k_rot, do_hflip, do_vflip)
        if abs(scale - 1.0) > 1e-4:
            new_h = max(8, int(round(h_in * scale)))
            new_w = max(8, int(round(w_in * scale)))
            x_t = F.interpolate(x_t, size=(new_h, new_w), mode="bilinear", align_corners=False)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp):
            conv6_t = self.model.forward_backbone(x_t)

        with torch.amp.autocast(device_type=conv6_clean.device.type, enabled=False):
            conv6_clean_f = conv6_clean.float()
            targets_f = targets.float()
            conv6_tf = conv6_t.float()
            fc8_w = self.model.fc8.weight.float()
            x_t_pool = self.model.pool(self.model.dropout7(conv6_tf))
            logits_t = F.conv2d(x_t_pool, fc8_w).view(b, -1)
            loss_cls_t = self.criterion(logits_t, targets_f)

            cams_clean = F.relu(F.conv2d(conv6_clean_f, fc8_w))
            cams_t = F.relu(F.conv2d(conv6_tf, fc8_w))
            if cams_t.shape[-2:] != cams_clean.shape[-2:]:
                cams_t = F.interpolate(
                    cams_t, size=cams_clean.shape[-2:], mode="bilinear", align_corners=False
                )
            cams_t_inv = self._inverse_d4_tensor(cams_t, k_rot, do_hflip, do_vflip).detach()

            m_clean = cams_clean / cams_clean.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            m_t_inv = cams_t_inv / cams_t_inv.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

            active_mask = (targets_f > 0.5).float().view(b, cams_clean.size(1), 1, 1)
            denom = (active_mask.sum() * float(cams_clean.size(-2) * cams_clean.size(-1))).clamp(min=1.0)
            diff_eq = F.smooth_l1_loss(m_clean, m_t_inv, reduction="none", beta=1.0) * active_mask
            loss_eq = diff_eq.sum() / denom
            return loss_cls_t, loss_eq

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        if self.proj_head is not None:
            self.proj_head.train()
        if self.mpppc_loss is not None:
            self.mpppc_loss.train()

        enable_pda = (epoch >= self.pda_start_epoch)
        lambda_eq = self.get_lambda_eq(epoch)
        enable_ppc = (
            self.proj_head is not None
            and self.mpppc_loss is not None
            and self.lambda_ppc > 0.0
            and epoch >= self.ppc_start_epoch
        )

        total_loss = 0.0
        total_eq_loss = 0.0
        total_ppc_loss = 0.0
        exact_matches = 0
        total_samples = 0
        per_class_tp = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        per_class_pred = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        per_class_gt = np.zeros(NUM_FOREGROUND, dtype=np.int64)

        all_trainable_params = list(self.model.parameters())
        if self.proj_head is not None:
            all_trainable_params.extend(list(self.proj_head.parameters()))

        for step, (images, targets, _) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            logits_clean, probs_clean, logits_pda, _, conv6_f = self.forward_dual_branches(
                images, enable_pda=enable_pda, targets=targets, return_conv6=True
            )
            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                if enable_pda and logits_pda is not None:
                    loss_cls = 0.5 * self.criterion(logits_clean, targets) + 0.5 * self.criterion(
                        logits_pda, targets
                    )
                else:
                    loss_cls = self.criterion(logits_clean, targets)
                loss = loss_cls

                l_eq_val = 0.0
                if self.use_equiv_consistency:
                    loss_cls_t, loss_eq = self.compute_equivariant_branch(images, targets, conv6_f)
                    loss = loss + 0.25 * loss_cls_t + lambda_eq * loss_eq
                    l_eq_val = float(loss_eq.detach().item())

                l_ppc_val = 0.0
                if enable_ppc and self.proj_head is not None and self.mpppc_loss is not None:
                    proj_feats = self.proj_head(conv6_f)
                    with torch.no_grad():
                        fc8_w_det = self.model.fc8.weight.float().detach()
                        cam_logits = F.conv2d(conv6_f.detach(), fc8_w_det)
                        raw_cams = F.relu(cam_logits)
                        cam_probs = F.softmax(cam_logits, dim=1)
                    loss_ppc = self.mpppc_loss(
                        proj_feats,
                        cam_probs,
                        targets,
                        raw_cams=raw_cams,
                        rgb_images=images,
                    )
                    loss = loss + self.lambda_ppc * loss_ppc
                    l_ppc_val = float(loss_ppc.detach().item())

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else None
            self.scaler.scale(loss).backward()
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            step_executed = (not self.scaler.is_enabled()) or (
                scale_before is not None and self.scaler.get_scale() >= scale_before
            )
            if step_executed and self.model_ema is not None:
                self.model_ema.update(self.model)
            if step_executed and self.scheduler is not None and self.step_scheduler_per_batch:
                self.scheduler.step()

            # Batch metrics (computed on clean branch matching evaluate() calibration)
            bs = images.size(0)
            total_loss += float(loss.item()) * bs
            total_eq_loss += l_eq_val * bs
            total_ppc_loss += l_ppc_val * bs
            total_samples += bs

            binary_preds = (probs_clean > 0.5).long().cpu().numpy()
            binary_gt = targets.long().cpu().numpy()

            # Exact match (all 4 classes correct)
            em_batch = np.all(binary_preds == binary_gt, axis=1).sum()
            exact_matches += int(em_batch)

            for c in range(NUM_FOREGROUND):
                per_class_tp[c] += np.logical_and(binary_preds[:, c] == 1, binary_gt[:, c] == 1).sum()
                per_class_pred[c] += (binary_preds[:, c] == 1).sum()
                per_class_gt[c] += (binary_gt[:, c] == 1).sum()

        if self.scheduler is not None and not self.step_scheduler_per_batch:
            self.scheduler.step()

        # Decay gamma from Epoch 0 per Paper Eq. 11 so first PDA epoch activates with gamma < 1.0
        if self.gamma > self.min_gamma:
            self.gamma = max(self.min_gamma, self.gamma * self.gamma_decay)

        # Compute F1 scores
        f1_scores = []
        for c in range(NUM_FOREGROUND):
            denom = per_class_pred[c] + per_class_gt[c]
            f1 = (2.0 * per_class_tp[c] / denom) if denom > 0 else 0.0
            f1_scores.append(float(f1))

        avg_loss = total_loss / max(total_samples, 1)
        em_rate = exact_matches / max(total_samples, 1)
        macro_f1 = float(np.mean(f1_scores))

        return {
            "loss": avg_loss,
            "eq_loss": total_eq_loss / max(total_samples, 1),
            "ppc_loss": total_ppc_loss / max(total_samples, 1),
            "lambda_eq": lambda_eq,
            "exact_match": em_rate,
            "macro_f1": macro_f1,
            "f1_tum": f1_scores[0],
            "f1_str": f1_scores[1],
            "f1_lym": f1_scores[2],
            "f1_nec": f1_scores[3],
            "gamma": self.gamma,
        }

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Evaluate multi-label metrics, per-channel & global GatedPM_mIoU, and fc8 diagnostics."""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_probs: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_logits: List[np.ndarray] = []

        pm_inter = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_union = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_dice_num = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_dice_den = np.zeros(NUM_FOREGROUND, dtype=np.float64)

        pm_inter_g = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_union_g = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_dice_num_g = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_dice_den_g = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        has_valid_masks = False

        for batch in val_loader:
            if len(batch) == 4:
                images, targets, gt_masks, _ = batch
            else:
                images, targets, _ = batch
                gt_masks = None

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp):
                conv6 = self.model.forward_backbone(images)
            with torch.amp.autocast(device_type=conv6.device.type, enabled=False):
                conv6_f = conv6.float()
                fc_w = self.model.fc8.weight.float()
                pooled = self.model.pool(conv6_f)
                logits = F.conv2d(pooled, fc_w).view(pooled.size(0), -1)
                probs = torch.sigmoid(logits)
                loss = self.criterion(logits, targets)

                total_loss += float(loss.item()) * images.size(0)
                total_samples += images.size(0)
                all_logits.append(logits.detach().cpu().numpy())
                all_probs.append(probs.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

                if gt_masks is not None:
                    gt_np = gt_masks.cpu().numpy()
                    valid_roi = (gt_np >= 0) & (gt_np < NUM_FOREGROUND)
                    if np.any(valid_roi):
                        has_valid_masks = True
                        cams = F.relu(F.conv2d(conv6_f, fc_w))
                        h_gt, w_gt = gt_np.shape[-2], gt_np.shape[-1]
                        cam_up = F.interpolate(cams, size=(h_gt, w_gt), mode="bilinear", align_corners=False)
                        gate = targets.float().view(-1, NUM_FOREGROUND, 1, 1)
                        cam_gated = cam_up * gate
                        b_sz = cam_gated.size(0)

                        # 1. Per-channel min-max normalization
                        cmin = cam_gated.view(b_sz, NUM_FOREGROUND, -1).min(dim=-1)[0].view(b_sz, NUM_FOREGROUND, 1, 1)
                        cmax = cam_gated.view(b_sz, NUM_FOREGROUND, -1).max(dim=-1)[0].view(b_sz, NUM_FOREGROUND, 1, 1)
                        cam_norm = ((cam_gated - cmin) / (cmax - cmin).clamp(min=1e-7)) * gate
                        # Mask inactive classes to -1.0 so zero-CAM ties never leak Class 0 (TUM)
                        cam_masked = torch.where(gate > 0.5, cam_norm, torch.full_like(cam_norm, -1.0))
                        pred_pm = torch.argmax(cam_masked, dim=1).cpu().numpy()

                        # 2. Global min-max normalization across all channels (author parity)
                        gmin = cam_gated.view(b_sz, -1).min(dim=-1)[0].view(b_sz, 1, 1, 1)
                        gmax = cam_gated.view(b_sz, -1).max(dim=-1)[0].view(b_sz, 1, 1, 1)
                        cam_norm_g = ((cam_gated - gmin) / (gmax - gmin).clamp(min=1e-7)) * gate
                        cam_masked_g = torch.where(gate > 0.5, cam_norm_g, torch.full_like(cam_norm_g, -1.0))
                        pred_pm_g = torch.argmax(cam_masked_g, dim=1).cpu().numpy()

                        for c in range(NUM_FOREGROUND):
                            gc = (gt_np == c) & valid_roi

                            pc = (pred_pm == c) & valid_roi
                            tp = float(np.logical_and(pc, gc).sum())
                            pm_inter[c] += tp
                            pm_union[c] += float(np.logical_or(pc, gc).sum())
                            pm_dice_num[c] += 2.0 * tp
                            pm_dice_den[c] += float(pc.sum() + gc.sum())

                            pc_g = (pred_pm_g == c) & valid_roi
                            tp_g = float(np.logical_and(pc_g, gc).sum())
                            pm_inter_g[c] += tp_g
                            pm_union_g[c] += float(np.logical_or(pc_g, gc).sum())
                            pm_dice_num_g[c] += 2.0 * tp_g
                            pm_dice_den_g[c] += float(pc_g.sum() + gc.sum())

        avg_loss = total_loss / max(total_samples, 1)
        if all_probs:
            y_prob = np.concatenate(all_probs, axis=0)
            y_true = np.concatenate(all_targets, axis=0)
            y_logits = np.concatenate(all_logits, axis=0)
            ml_metrics = compute_multilabel_validation_metrics(y_true, y_prob)
            logits_mean = float(np.mean(y_logits))
        else:
            ml_metrics = compute_multilabel_validation_metrics(
                np.zeros((0, NUM_FOREGROUND), dtype=np.float32),
                np.zeros((0, NUM_FOREGROUND), dtype=np.float32),
            )
            logits_mean = 0.0

        fc8_w_flat = self.model.fc8.weight.detach().float().view(NUM_FOREGROUND, -1)
        fc8_std = float(fc8_w_flat.std().item())
        fc8_norms = torch.linalg.norm(fc8_w_flat, dim=1)

        out: Dict[str, Any] = {
            "val_loss": avg_loss,
            "val_logits_mean": logits_mean,
            "fc8_std": fc8_std,
            "fc8_norm_tum": float(fc8_norms[0].item()),
            "fc8_norm_str": float(fc8_norms[1].item()),
            "fc8_norm_lym": float(fc8_norms[2].item()),
            "fc8_norm_nec": float(fc8_norms[3].item()),
            **ml_metrics,
        }
        if has_valid_masks:
            pm_ious = [
                float(pm_inter[c] / pm_union[c]) if pm_union[c] > 0 else 0.0
                for c in range(NUM_FOREGROUND)
            ]
            pm_dices = [
                float(pm_dice_num[c] / pm_dice_den[c]) if pm_dice_den[c] > 0 else 0.0
                for c in range(NUM_FOREGROUND)
            ]
            active_ious = [pm_ious[c] for c in range(NUM_FOREGROUND) if pm_union[c] > 0]
            active_dices = [pm_dices[c] for c in range(NUM_FOREGROUND) if pm_dice_den[c] > 0]
            out["val_gated_pm_miou"] = float(np.mean(active_ious)) if active_ious else 0.0
            out["val_gated_pm_mdice"] = float(np.mean(active_dices)) if active_dices else 0.0
            out["val_pm_iou_tum"] = pm_ious[0]
            out["val_pm_iou_str"] = pm_ious[1]
            out["val_pm_iou_lym"] = pm_ious[2]
            out["val_pm_iou_nec"] = pm_ious[3]

            pm_ious_g = [
                float(pm_inter_g[c] / pm_union_g[c]) if pm_union_g[c] > 0 else 0.0
                for c in range(NUM_FOREGROUND)
            ]
            pm_dices_g = [
                float(pm_dice_num_g[c] / pm_dice_den_g[c]) if pm_dice_den_g[c] > 0 else 0.0
                for c in range(NUM_FOREGROUND)
            ]
            active_ious_g = [pm_ious_g[c] for c in range(NUM_FOREGROUND) if pm_union_g[c] > 0]
            active_dices_g = [pm_dices_g[c] for c in range(NUM_FOREGROUND) if pm_dice_den_g[c] > 0]
            out["val_gated_pm_miou_global"] = float(np.mean(active_ious_g)) if active_ious_g else 0.0
            out["val_gated_pm_mdice_global"] = float(np.mean(active_dices_g)) if active_dices_g else 0.0
            out["val_pm_iou_global_tum"] = pm_ious_g[0]
            out["val_pm_iou_global_str"] = pm_ious_g[1]
            out["val_pm_iou_global_lym"] = pm_ious_g[2]
            out["val_pm_iou_global_nec"] = pm_ious_g[3]
        return out


def build_optimizer_and_scheduler(
    model: WSSSTissueStage1NetCAM,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    *,
    optimizer_type: str = "adamw",
    scheduler_type: str = "cosine",
    lr_fc8: Optional[float] = None,
    wd_fc8: Optional[float] = None,
    momentum: float = 0.9,
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    poly_power: float = 0.9,
    eta_min: float = 1e-6,
    extra_params: Optional[List[nn.Parameter]] = None,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Build optimizer (AdamW or PolySGD) and scheduler (Cosine or PolyLR).

    Applies weight_decay to trainable weights and 0.0 to biases or 1D parameters.
    Frozen parameters (requires_grad=False) are excluded.
    """
    eff_lr_fc8 = (10.0 * lr) if lr_fc8 is None else float(lr_fc8)
    eff_wd_fc8 = float(weight_decay) if wd_fc8 is None else float(wd_fc8)

    backbone_decay: List[nn.Parameter] = []
    backbone_no_decay: List[nn.Parameter] = []
    head_decay: List[nn.Parameter] = []
    head_no_decay: List[nn.Parameter] = []
    extra_decay: List[nn.Parameter] = []
    extra_no_decay: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = name.endswith(".bias") or param.ndim <= 1
        if "fc8" in name:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    if extra_params:
        for param in extra_params:
            if not param.requires_grad:
                continue
            if param.ndim <= 1:
                extra_no_decay.append(param)
            else:
                extra_decay.append(param)

    param_groups: List[Dict[str, Any]] = []
    if backbone_decay:
        param_groups.append({"params": backbone_decay, "lr": lr, "weight_decay": weight_decay})
    if backbone_no_decay:
        param_groups.append({"params": backbone_no_decay, "lr": lr, "weight_decay": 0.0})
    if head_decay:
        param_groups.append({"params": head_decay, "lr": eff_lr_fc8, "weight_decay": eff_wd_fc8})
    if head_no_decay:
        param_groups.append({"params": head_no_decay, "lr": eff_lr_fc8, "weight_decay": 0.0})
    if extra_decay:
        param_groups.append({"params": extra_decay, "lr": lr, "weight_decay": weight_decay})
    if extra_no_decay:
        param_groups.append({"params": extra_no_decay, "lr": lr, "weight_decay": 0.0})

    opt_key = str(optimizer_type or "adamw").strip().lower()
    if opt_key in ("polysgd", "sgd"):
        optimizer: torch.optim.Optimizer = torch.optim.SGD(param_groups, momentum=float(momentum))
    else:
        optimizer = torch.optim.AdamW(param_groups)

    sched_key = str(scheduler_type or "cosine").strip().lower()
    eff_warmup = max(int(warmup_steps), 0)
    if sched_key == "poly":
        eff_total = max(int(total_steps if total_steps is not None else epochs), 1)
        power_f = float(poly_power)

        def _poly_lambda(step: int) -> float:
            if eff_warmup > 0 and step < eff_warmup:
                return float(step + 1) / float(eff_warmup)
            denom = max(eff_total - eff_warmup, 1)
            progress = min(max(float(step - eff_warmup) / float(denom), 0.0), 1.0)
            return max((1.0 - progress) ** power_f, 0.0)

        scheduler: Any = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly_lambda)
    elif eff_warmup > 0 and total_steps is not None and total_steps > 0:
        eff_total = max(int(total_steps), 1)
        min_ratio = min(max(float(eta_min) / max(float(lr), 1e-12), 0.0), 1.0)

        def _cosine_warmup_lambda(step: int) -> float:
            if step < eff_warmup:
                return float(step + 1) / float(eff_warmup)
            denom = max(eff_total - eff_warmup, 1)
            progress = min(max(float(step - eff_warmup) / float(denom), 0.0), 1.0)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_warmup_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(eta_min),
        )

    return optimizer, scheduler


def train_stage1_pipeline(
    data_dir: Union[str, Path],
    checkpoint_dir: Union[str, Path] = "checkpoints",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    loss_type: Optional[str] = None,
    resume_path: Optional[Union[str, Path]] = None,
    pretrained_path: Optional[Union[str, Path]] = None,
    allow_random_init: Optional[bool] = None,
    auto_download_domain_pretrained: bool = False,
    input_norm: Optional[str] = None,
    dry_run: bool = False,
    device_str: Optional[str] = None,
    seed: Optional[int] = None,
    pda_start_epoch: Optional[int] = None,
    initial_gamma: Optional[float] = None,
    gamma_decay: Optional[float] = None,
    min_gamma: Optional[float] = None,
    num_workers: Optional[int] = None,
    enable_regression_guard: bool = True,
    preset: Optional[str] = None,
    experiment_config: Optional[Stage1ExperimentConfig] = None,
    stop_after_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute end-to-end Stage 1 classifier training pipeline with Rule 17 validation gate."""
    if experiment_config is not None:
        cfg = experiment_config
    elif preset is not None:
        cfg = get_stage1_preset(preset)
    else:
        cfg = get_stage1_preset("default")

    eff_seed = cfg.seed if seed is None else int(seed)
    eff_epochs = cfg.max_epochs if epochs is None else int(epochs)
    eff_stop_after = cfg.stop_after_epoch if stop_after_epoch is None else int(stop_after_epoch)
    if eff_stop_after is not None and eff_stop_after < 1:
        raise ValueError(f"stop_after_epoch must be >= 1, got {eff_stop_after}.")
    eff_bs_cfg = cfg.batch_size if batch_size is None else int(batch_size)
    eff_lr_bb = cfg.lr_bb if lr is None else float(lr)
    eff_lr_fc8 = cfg.lr_fc8 if (lr is None or preset is not None or experiment_config is not None) else (10.0 * float(lr))
    eff_loss_type = cfg.loss_type if loss_type is None else str(loss_type)
    eff_input_norm = cfg.input_norm if input_norm is None else str(input_norm)
    eff_pda_start = cfg.pda_start_epoch if pda_start_epoch is None else int(pda_start_epoch)
    eff_init_gamma = cfg.initial_gamma if initial_gamma is None else float(initial_gamma)
    eff_gamma_decay = cfg.gamma_decay if gamma_decay is None else float(gamma_decay)
    eff_min_gamma = cfg.min_gamma if min_gamma is None else float(min_gamma)

    random.seed(eff_seed)
    np.random.seed(eff_seed)
    torch.manual_seed(eff_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eff_seed)

    norm_mode = str(eff_input_norm or "zero_one").strip().lower()
    if norm_mode not in ("imagenet", "zero_one"):
        raise ValueError(f"Unsupported input_norm={eff_input_norm!r}. Expected 'imagenet' or 'zero_one'.")

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Initialize model on target device BEFORE loading resume state or building optimizer
    model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND).to(device)

    # Load pretrained backbone unless resuming from an existing Stage-1 checkpoint
    start_epoch = 0
    best_val_score = -1.0
    best_cls_score = -1.0
    best_macro_f1 = 0.0
    best_val_loss_seen = float("inf")
    best_val_map_seen = -1.0
    best_val_miou_seen = -1.0
    epochs_without_improve = 0
    joint_no_improve = 0
    resumed_gamma = eff_init_gamma
    history: List[Dict[str, Any]] = []
    pretrained_info: Dict[str, Any] = {
        "loaded_layers": 0,
        "total_backbone_layers": sum(1 for k in model.state_dict() if not k.startswith("fc8.")),
        "source": "resume" if (resume_path and Path(resume_path).exists()) else "random_init",
        "path": str(resume_path) if (resume_path and Path(resume_path).exists()) else None,
    }

    loaded_resume_obj: Optional[Dict[str, Any]] = None
    if resume_path and Path(resume_path).exists():
        try:
            loaded = torch.load(resume_path, map_location=device, weights_only=True)
        except Exception:
            loaded = torch.load(resume_path, map_location=device, weights_only=False)  # nosec B614
        if isinstance(loaded, dict):
            loaded_resume_obj = loaded
            state_dict = loaded.get("state_dict", loaded)
            start_epoch = int(loaded.get("epoch", -1)) + 1
            best_macro_f1 = float(loaded.get("macro_f1", 0.0))
            best_val_score = float(loaded.get("best_val_score", best_macro_f1))
            best_cls_score = float(loaded.get("best_cls_score", loaded.get("val_mAP", -1.0)))
            epochs_without_improve = int(loaded.get("epochs_without_improve", 0))
            joint_no_improve = int(loaded.get("joint_no_improve", epochs_without_improve))
            resumed_gamma = float(loaded.get("pda_gamma", eff_init_gamma))
            best_val_loss_seen = float(loaded.get("best_val_loss_seen", float("inf")))
            best_val_map_seen = float(loaded.get("best_val_map_seen", best_cls_score))
            best_val_miou_seen = float(loaded.get("best_val_miou_seen", best_val_score))
            if isinstance(loaded.get("history"), list):
                history = list(loaded["history"])
        else:
            state_dict = loaded
        model.load_state_dict(state_dict, strict=False)
    else:
        effective_allow_random = dry_run if allow_random_init is None else bool(allow_random_init)
        pretrained_info = load_pretrained_resnet38d_weights(
            model,
            pretrained_path=pretrained_path,
            allow_random_init=effective_allow_random,
            reset_classifier_head=True,
            auto_download_domain_pretrained=auto_download_domain_pretrained,
            cache_dir=checkpoint_path,
        )
        src_lower = str(pretrained_info.get("source", "")).lower()
        path_lower = str(pretrained_info.get("path", "") or "").lower()
        if norm_mode == "imagenet" and ("luad" in src_lower or "luad" in path_lower or "bcss" in path_lower):
            warnings.warn(
                "[STAGE 1 WARN] Domain-pretrained histopathology ResNet-38d checkpoints expect "
                "input_norm='zero_one' ([0, 1]), but input_norm='imagenet' was requested. "
                "Pass input_norm='zero_one' for paper-parity convergence.",
                UserWarning,
                stacklevel=2,
            )

    # Loss function (Baseline Parity default: MultiLabelSoftMarginLoss)
    loss_key = str(eff_loss_type or "soft_margin").strip().lower()
    if loss_key == "asl":
        criterion: nn.Module = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.02)
    elif loss_key == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        loss_key = "soft_margin"
        criterion = nn.MultiLabelSoftMarginLoss()

    # DataLoaders first so per-step scheduler knows exact steps_per_epoch
    max_train = 4 if dry_run else None
    max_val = 4 if dry_run else None
    effective_epochs = 1 if dry_run else eff_epochs
    effective_bs = 2 if dry_run else eff_bs_cfg
    effective_workers = (0 if dry_run else cfg.num_workers) if num_workers is None else int(num_workers)

    train_loader, val_loader = create_stage1_dataloaders(
        data_dir=data_dir,
        batch_size=effective_bs,
        num_workers=effective_workers,
        pin_memory=False if dry_run else True,
        max_train_samples=max_train,
        max_val_samples=max_val,
        input_norm=norm_mode,
        return_val_masks=True,
        strict_patient_filter=not dry_run,
        use_d4_aug=cfg.use_d4_aug,
    )

    if not dry_run and (val_loader is None or len(val_loader) == 0):
        raise RuntimeError(
            "[STAGE 1 ERROR] Validation split not found or empty. "
            "Refusing blind training without an independent validation loop (Rule 17)."
        )

    # Optional Arm C modules (stored separately from WSSSTissueStage1NetCAM)
    proj_head: Optional[ProjectionHead] = None
    mpppc_loss: Optional[MultiPrototypePPCLoss] = None
    if cfg.use_mpppc:
        proj_head = ProjectionHead(in_dim=4096, hidden_dim=256, out_dim=128, num_groups=16).to(device)
        mpppc_loss = MultiPrototypePPCLoss(
            num_classes=NUM_FOREGROUND,
            num_prototypes=cfg.ppc_num_prototypes,
            feat_dim=128,
            conf_thresh=cfg.ppc_conf_thresh,
            warm_start=True,
        ).to(device)
        if loaded_resume_obj is not None:
            if "proj_head_state_dict" in loaded_resume_obj:
                proj_head.load_state_dict(loaded_resume_obj["proj_head_state_dict"])
            if "mpppc_state_dict" in loaded_resume_obj:
                mpppc_loss.load_state_dict(loaded_resume_obj["mpppc_state_dict"])

    # Move all trainable modules to target device BEFORE building optimizer or loading optimizer state
    model = model.to(device)
    if proj_head is not None:
        proj_head = proj_head.to(device)
    if mpppc_loss is not None:
        mpppc_loss = mpppc_loss.to(device)

    # Call model.train() BEFORE building optimizer so frozen BN & not_training params are excluded
    model.train()
    steps_per_epoch = max(len(train_loader), 1)
    step_scheduler_per_batch = (cfg.scheduler_type == "poly") or (cfg.warmup_epochs > 0.0)
    warmup_steps = int(round(float(cfg.warmup_epochs) * steps_per_epoch)) if step_scheduler_per_batch else 0
    total_steps = (effective_epochs * steps_per_epoch) if step_scheduler_per_batch else None

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr=eff_lr_bb,
        weight_decay=cfg.wd_bb,
        epochs=effective_epochs,
        optimizer_type=cfg.optimizer_type,
        scheduler_type=cfg.scheduler_type,
        lr_fc8=eff_lr_fc8,
        wd_fc8=cfg.wd_fc8,
        momentum=cfg.momentum,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        poly_power=cfg.poly_power,
        eta_min=cfg.eta_min,
        extra_params=list(proj_head.parameters()) if proj_head is not None else None,
    )
    if loaded_resume_obj is not None:
        if "optimizer" in loaded_resume_obj:
            optimizer.load_state_dict(loaded_resume_obj["optimizer"])
            for state_vals in optimizer.state.values():
                for sk, sv in state_vals.items():
                    if isinstance(sv, torch.Tensor) and sk != "step":
                        state_vals[sk] = sv.to(device)
        if "scheduler" in loaded_resume_obj:
            scheduler.load_state_dict(loaded_resume_obj["scheduler"])

    model_ema: Optional[ModelEMA] = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    if model_ema is not None and loaded_resume_obj is not None and "ema_state_dict" in loaded_resume_obj:
        model_ema.load_state_dict(loaded_resume_obj["ema_state_dict"])

    trainer = Stage1Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        enable_amp=not dry_run,
        pda_start_epoch=eff_pda_start,
        initial_gamma=resumed_gamma,
        gamma_decay=eff_gamma_decay,
        min_gamma=eff_min_gamma,
        pda_mode=cfg.pda_mode,
        step_scheduler_per_batch=step_scheduler_per_batch,
        model_ema=model_ema,
        proj_head=proj_head,
        mpppc_loss=mpppc_loss,
        lambda_ppc=cfg.lambda_ppc,
        ppc_start_epoch=cfg.ppc_start_epoch,
        use_equiv_consistency=cfg.use_equiv_consistency,
        equiv_scales=cfg.equiv_scales,
        lambda_eq_max=cfg.lambda_eq_max,
        eq_ramp_start_epoch=cfg.eq_ramp_start_epoch,
        eq_ramp_end_epoch=cfg.eq_ramp_end_epoch,
    )
    if (
        loaded_resume_obj is not None
        and trainer.scaler.is_enabled()
        and isinstance(loaded_resume_obj.get("scaler"), dict)
        and loaded_resume_obj["scaler"]
    ):
        trainer.scaler.load_state_dict(loaded_resume_obj["scaler"])

    best_save_path = checkpoint_path / "stage1_retrained_best.pth"
    best_cls_save_path = checkpoint_path / "stage1_best_cls.pth"
    latest_save_path = checkpoint_path / "stage1_latest.pth"
    selection_metric = "val_mAP"
    final_selected_weights = (
        str(loaded_resume_obj.get("selected_weights", "raw"))
        if loaded_resume_obj is not None
        else "raw"
    )

    loop_should_run = True
    if not dry_run and eff_stop_after is not None and start_epoch >= eff_stop_after:
        print(
            f"[STOP AFTER EPOCH] Start epoch {start_epoch + 1} already past "
            f"stop_after_epoch={eff_stop_after} (LR schedule horizon={effective_epochs}). Halting cleanly.",
            flush=True,
        )
        loop_should_run = False

    for epoch in (range(start_epoch, effective_epochs) if loop_should_run else ()):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)

        raw_val_metrics: Dict[str, Any] = {}
        ema_val_metrics: Dict[str, Any] = {}
        if val_loader is not None and len(val_loader) > 0:
            raw_val_metrics = trainer.evaluate(val_loader)
            if model_ema is not None:
                with model_ema.swap_to_ema(model):
                    ema_val_metrics = trainer.evaluate(val_loader)

        # Pick winning weights (raw vs ema) for primary selection metric
        def _score_from(vm: Dict[str, Any]) -> Tuple[str, float]:
            if "val_gated_pm_miou" in vm:
                return "val_gated_pm_miou", float(vm["val_gated_pm_miou"])
            if "val_mAP" in vm:
                return "val_mAP", float(vm["val_mAP"])
            if "val_macro_f1" in vm:
                return "val_macro_f1", float(vm["val_macro_f1"])
            return "train_macro_f1_dry_run", float(train_metrics["macro_f1"])

        selection_metric, raw_score = _score_from(raw_val_metrics)
        _, ema_score = _score_from(ema_val_metrics) if ema_val_metrics else (selection_metric, -1.0)
        if model_ema is not None and ema_val_metrics and ema_score >= raw_score:
            selected_weights = "ema"
            val_metrics = ema_val_metrics
            current_val_score = ema_score
            winning_state_dict = model_ema.clean_model_state_dict()
        else:
            selected_weights = "raw"
            val_metrics = raw_val_metrics
            current_val_score = raw_score
            winning_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Best classification checkpoint (by val_mAP across raw & ema)
        raw_cls = float(raw_val_metrics.get("val_mAP", raw_val_metrics.get("val_macro_f1", train_metrics["macro_f1"])))
        ema_cls = float(ema_val_metrics.get("val_mAP", ema_val_metrics.get("val_macro_f1", -1.0))) if ema_val_metrics else -1.0
        if model_ema is not None and ema_val_metrics and ema_cls >= raw_cls:
            cls_selected_weights = "ema"
            current_cls_score = ema_cls
            cls_state_dict = model_ema.clean_model_state_dict()
            cls_metrics_ref = ema_val_metrics
        else:
            cls_selected_weights = "raw"
            current_cls_score = raw_cls
            cls_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            cls_metrics_ref = raw_val_metrics

        ema_summary = {
            f"ema_{k}": v
            for k, v in ema_val_metrics.items()
            if k in (
                "val_loss",
                "val_mAP",
                "val_auroc",
                "val_best_f1",
                "val_macro_f1",
                "val_gated_pm_miou",
                "val_gated_pm_miou_global",
                "fc8_std",
            )
        }
        epoch_record = {
            **train_metrics,
            **raw_val_metrics,
            **ema_summary,
            "selected_weights": selected_weights,
            "exp_preset": cfg.preset_name,
            "epoch": epoch,
        }
        history.append(epoch_record)

        current_f1 = float(val_metrics.get("val_best_f1", val_metrics.get("val_macro_f1", train_metrics["macro_f1"])))
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1

        improved = current_val_score >= best_val_score
        if improved:
            best_val_score = max(best_val_score, current_val_score)
            epochs_without_improve = 0
            final_selected_weights = selected_weights
        else:
            epochs_without_improve += 1

        cls_improved = current_cls_score >= best_cls_score
        if cls_improved:
            best_cls_score = max(best_cls_score, current_cls_score)

        # Joint early-stop patience across {val_gated_pm_miou, val_mAP, val_loss}
        cur_miou = float(val_metrics.get("val_gated_pm_miou", -1.0))
        cur_map = float(val_metrics.get("val_mAP", -1.0))
        cur_vloss = float(val_metrics.get("val_loss", float("inf")))
        any_joint_improve = False
        if cur_miou > best_val_miou_seen + 1e-6:
            best_val_miou_seen = cur_miou
            any_joint_improve = True
        if cur_map > best_val_map_seen + 1e-6:
            best_val_map_seen = cur_map
            any_joint_improve = True
        if cur_vloss < best_val_loss_seen - 1e-6:
            best_val_loss_seen = cur_vloss
            any_joint_improve = True
        if any_joint_improve:
            joint_no_improve = 0
        else:
            joint_no_improve += 1

        # Rich per-epoch diagnostic logging
        raw_pm = float(raw_val_metrics.get("val_gated_pm_miou", 0.0))
        raw_pm_g = float(raw_val_metrics.get("val_gated_pm_miou_global", 0.0))
        print(
            f"[STAGE 1] Ep {epoch + 1}/{effective_epochs} ({cfg.preset_name}) | "
            f"TrainLoss={train_metrics['loss']:.4f} | "
            f"ValLoss={raw_val_metrics.get('val_loss', 0.0):.4f} | "
            f"mAP={raw_val_metrics.get('val_mAP', 0.0):.4f} | "
            f"AUROC={raw_val_metrics.get('val_auroc', 0.0):.4f} | "
            f"BestF1={raw_val_metrics.get('val_best_f1', 0.0):.4f} | "
            f"F1@0.5={raw_val_metrics.get('val_macro_f1', 0.0):.4f} | "
            f"LogitMean={raw_val_metrics.get('val_logits_mean', 0.0):.4f} | "
            f"GatedPM_mIoU={raw_pm:.4f} (Global={raw_pm_g:.4f}) | "
            f"fc8_std={raw_val_metrics.get('fc8_std', 0.0):.5f} | "
            f"fc8_norms=[{raw_val_metrics.get('fc8_norm_tum', 0.0):.3f},"
            f"{raw_val_metrics.get('fc8_norm_str', 0.0):.3f},"
            f"{raw_val_metrics.get('fc8_norm_lym', 0.0):.3f},"
            f"{raw_val_metrics.get('fc8_norm_nec', 0.0):.3f}]",
            flush=True,
        )
        if ema_val_metrics:
            print(
                f"  [EMA] ValLoss={ema_val_metrics.get('val_loss', 0.0):.4f} | "
                f"mAP={ema_val_metrics.get('val_mAP', 0.0):.4f} | "
                f"AUROC={ema_val_metrics.get('val_auroc', 0.0):.4f} | "
                f"BestF1={ema_val_metrics.get('val_best_f1', 0.0):.4f} | "
                f"GatedPM_mIoU={ema_val_metrics.get('val_gated_pm_miou', 0.0):.4f} "
                f"(Global={ema_val_metrics.get('val_gated_pm_miou_global', 0.0):.4f}) | "
                f"selected={selected_weights}",
                flush=True,
            )

        compact_payload = {
            "epoch": epoch,
            "exp_preset": cfg.preset_name,
            "selected_weights": selected_weights,
            "macro_f1": current_f1,
            "best_val_score": max(0.0, best_val_score),
            "best_val_metric": selection_metric,
            "val_mAP": float(val_metrics.get("val_mAP", 0.0)),
            "val_auroc": float(val_metrics.get("val_auroc", 0.0)),
            "val_best_f1": float(val_metrics.get("val_best_f1", 0.0)),
            "val_gated_pm_miou": float(val_metrics.get("val_gated_pm_miou", 0.0)),
            "val_gated_pm_miou_global": float(val_metrics.get("val_gated_pm_miou_global", 0.0)),
            "fc8_std": float(val_metrics.get("fc8_std", 0.0)),
            "pda_gamma": float(trainer.gamma),
            "input_norm": norm_mode,
            "loss_type": loss_key,
            "pretrained_source": pretrained_info["source"],
            "pretrained_loaded_layers": pretrained_info["loaded_layers"],
            "state_dict": winning_state_dict,
            "metrics": epoch_record,
        }

        if not dry_run:
            latest_payload: Dict[str, Any] = {
                "epoch": epoch,
                "exp_preset": cfg.preset_name,
                "selected_weights": selected_weights,
                "macro_f1": current_f1,
                "best_val_score": max(0.0, best_val_score),
                "best_cls_score": max(0.0, best_cls_score),
                "best_val_metric": selection_metric,
                "best_val_loss_seen": best_val_loss_seen,
                "best_val_map_seen": best_val_map_seen,
                "best_val_miou_seen": best_val_miou_seen,
                "epochs_without_improve": epochs_without_improve,
                "joint_no_improve": joint_no_improve,
                "pda_gamma": float(trainer.gamma),
                "input_norm": norm_mode,
                "loss_type": loss_key,
                "pretrained_source": pretrained_info["source"],
                "pretrained_loaded_layers": pretrained_info["loaded_layers"],
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "history": history,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": trainer.scaler.state_dict(),
            }
            if model_ema is not None:
                latest_payload["ema_state_dict"] = model_ema.state_dict()
            if proj_head is not None:
                latest_payload["proj_head_state_dict"] = {
                    k: v.detach().cpu().clone() for k, v in proj_head.state_dict().items()
                }
            if mpppc_loss is not None:
                latest_payload["mpppc_state_dict"] = {
                    k: v.detach().cpu().clone() for k, v in mpppc_loss.state_dict().items()
                }
            torch.save(latest_payload, latest_save_path)

        if improved or dry_run or not best_save_path.is_file():
            torch.save(compact_payload, best_save_path)

        if cls_improved or dry_run or not best_cls_save_path.is_file():
            cls_payload = {
                "epoch": epoch,
                "exp_preset": cfg.preset_name,
                "selected_weights": cls_selected_weights,
                "best_val_metric": "val_mAP",
                "best_val_score": max(0.0, best_cls_score),
                "val_mAP": float(cls_metrics_ref.get("val_mAP", 0.0)),
                "val_auroc": float(cls_metrics_ref.get("val_auroc", 0.0)),
                "val_best_f1": float(cls_metrics_ref.get("val_best_f1", 0.0)),
                "val_gated_pm_miou": float(cls_metrics_ref.get("val_gated_pm_miou", 0.0)),
                "fc8_std": float(cls_metrics_ref.get("fc8_std", 0.0)),
                "input_norm": norm_mode,
                "loss_type": loss_key,
                "state_dict": cls_state_dict,
            }
            torch.save(cls_payload, best_cls_save_path)

        if enable_regression_guard and not dry_run:
            val_best_f1 = float(val_metrics.get("val_best_f1", 1.0))
            val_loss = float(val_metrics.get("val_loss", 0.0))
            val_map = float(val_metrics.get("val_mAP", 1.0))
            is_nan = (not math.isfinite(val_loss)) or (not math.isfinite(float(train_metrics["loss"])))
            is_collapse = (epoch >= 4) and (val_best_f1 < 0.55 or val_loss > 1.20 or val_map < 0.65)
            if is_nan or is_collapse:
                print(
                    f"[CALIBRATION COLLAPSE ALARM] Epoch {epoch + 1}: Calibration/collapse detected "
                    f"(val_best_f1={val_best_f1:.4f}, val_loss={val_loss:.4f}, val_mAP={val_map:.4f}). Halting.",
                    flush=True,
                )
                break
        if not dry_run and eff_stop_after is not None and (epoch + 1) >= eff_stop_after:
            print(
                f"[STOP AFTER EPOCH] Epoch {epoch + 1}: Completed target stop_after_epoch={eff_stop_after} "
                f"(LR schedule horizon={effective_epochs}). Halting cleanly.",
                flush=True,
            )
            break
        if enable_regression_guard and not dry_run:
            if (epoch + 1) >= cfg.min_epochs_early_stop and joint_no_improve >= cfg.early_stop_patience:
                print(
                    f"[EARLY STOP PATIENCE] Epoch {epoch + 1}: No joint improvement across "
                    f"{{val_gated_pm_miou, val_mAP, val_loss}} for {joint_no_improve} epochs "
                    f"(min_epochs={cfg.min_epochs_early_stop}, best {selection_metric}={best_val_score:.4f}). Halting.",
                    flush=True,
                )
                break

    final_fc8_std = float(history[-1].get("fc8_std", 0.0)) if history else 0.0
    return {
        "exp_preset": cfg.preset_name,
        "selected_weights": final_selected_weights,
        "best_macro_f1": best_macro_f1,
        "best_val_score": max(0.0, best_val_score),
        "best_cls_score": max(0.0, best_cls_score),
        "best_val_metric": selection_metric,
        "fc8_std": final_fc8_std,
        "input_norm": norm_mode,
        "loss_type": loss_key,
        "pretrained_info": pretrained_info,
        "checkpoint_path": str(best_save_path),
        "best_cls_checkpoint_path": str(best_cls_save_path) if best_cls_save_path.is_file() else None,
        "latest_checkpoint_path": str(latest_save_path) if latest_save_path.is_file() else None,
        "max_epochs": effective_epochs,
        "stop_after_epoch": None if dry_run else eff_stop_after,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 ResNet-38d Classifier on BCSS-WSSS")
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="Dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--preset", type=str, default=None, help="Stage-1 experiment preset name")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (LR schedule horizon)")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="Optional 1-indexed epoch to stop cleanly after")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--loss-type",
        type=str,
        default=None,
        choices=["soft_margin", "asl", "bce"],
        help="Loss type (default: soft_margin for paper parity)",
    )
    parser.add_argument(
        "--input-norm",
        type=str,
        default=None,
        choices=["imagenet", "zero_one"],
        help="Input normalization mode saved in checkpoint metadata (default: zero_one for paper parity)",
    )
    parser.add_argument("--pretrained-path", type=str, default=None, help="Pretrained ResNet-38d .params or .pth path")
    parser.add_argument("--allow-random-init", action="store_true", help="Explicitly allow training from random weights")
    parser.add_argument(
        "--auto-download-domain-pretrained",
        action="store_true",
        help="Auto-download cross-organ LUAD ResNet-38d backbone if no local weights exist",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-step dry run for testing")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_stage1_config_from_env(
        overrides={
            "preset_name": args.preset,
            "max_epochs": args.epochs,
            "stop_after_epoch": args.stop_after_epoch,
            "batch_size": args.batch_size,
            "loss_type": args.loss_type,
            "input_norm": args.input_norm,
            "seed": args.seed,
        }
    )
    results = train_stage1_pipeline(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        resume_path=args.resume,
        pretrained_path=args.pretrained_path,
        allow_random_init=args.allow_random_init if args.allow_random_init else None,
        auto_download_domain_pretrained=args.auto_download_domain_pretrained,
        dry_run=args.dry_run,
        device_str=args.device,
        experiment_config=cfg,
    )
    print("Stage 1 Training Complete.")
    print(f"Best Val Metric ({results['best_val_metric']}): {results['best_val_score']:.4f}")
    print(f"Best Macro F1: {results['best_macro_f1']:.4f}")
    print(f"Saved Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
