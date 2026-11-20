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

    Prevents network from focusing solely on the most discriminative focal points
    by adaptively dropping out high-activation regions of the Class Activation Map.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        fc_weights: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Args:
            x: (B, 4096, H, W) feature map from bn7.
            fc_weights: (4, 4096, 1, 1) weights from fc8.
            gamma: Dropout threshold factor in [0.65, 1.0].

        Returns:
            masked_x: (B, 4096, H, W) feature map modulated by attention mask.
        """
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            cams = F.conv2d(x_f, fc_weights.float())
            cams = F.relu(cams)  # (B, 4, H, W)
            b, c, h, w = cams.size()

            cam_mean = torch.mean(cams, dim=1)  # (B, H, W)
            zero = torch.zeros_like(cam_mean)
            mean_drop_cam = zero.clone()

            for i in range(c):
                sub_cam = cams[:, i, :, :]  # (B, H, W)
                sub_cam_max = torch.amax(sub_cam.view(b, -1), dim=-1).view(b, 1, 1)
                thr = sub_cam_max * float(gamma)
                thr = thr.expand_as(sub_cam)
                sub_cam_with_drop = torch.where(sub_cam > thr, zero, sub_cam)
                mean_drop_cam = mean_drop_cam + sub_cam_with_drop

            mean_drop_cam = mean_drop_cam / max(c, 1)
            mean_drop_cam = mean_drop_cam.unsqueeze(1)  # (B, 1, H, W)

            # Return FP32 to prevent float16 >65504 overflow on unnormalized 4096-ch CAM products
            return x_f * mean_drop_cam


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
        pda_start_epoch: int = 2,
        initial_gamma: float = 1.0,
        gamma_decay: float = 0.98,
        min_gamma: float = 0.65,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.enable_amp = enable_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_amp)

        self.attention_module = AttentionModule().to(self.device)
        self.pda_start_epoch = pda_start_epoch
        self.gamma = initial_gamma
        self.gamma_decay = gamma_decay
        self.min_gamma = min_gamma

    def forward_with_pda(
        self,
        images: torch.Tensor,
        enable_pda: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with conditional Progressive Dropout Attention."""
        conv6 = self.model.forward_backbone(images)  # (B, 4096, 28, 28)

        with torch.amp.autocast(device_type=conv6.device.type, enabled=False):
            conv6_f = conv6.float()
            if enable_pda:
                conv6_f = self.attention_module(conv6_f, self.model.fc8.weight.float(), self.gamma)

            x = self.model.dropout7(conv6_f)
            x = self.model.pool(x)
            logits = F.conv2d(x, self.model.fc8.weight.float()).view(x.size(0), -1)  # (B, 4)
            probs = torch.sigmoid(logits)
            return logits, probs

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        enable_pda = (epoch >= self.pda_start_epoch)

        total_loss = 0.0
        exact_matches = 0
        total_samples = 0
        per_class_tp = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        per_class_pred = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        per_class_gt = np.zeros(NUM_FOREGROUND, dtype=np.int64)

        for step, (images, targets, _) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.enable_amp):
                logits, probs = self.forward_with_pda(images, enable_pda=enable_pda)
                loss = self.criterion(logits, targets)

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Batch metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            binary_preds = (probs > 0.5).long().cpu().numpy()
            binary_gt = targets.long().cpu().numpy()

            # Exact match (all 4 classes correct)
            em_batch = np.all(binary_preds == binary_gt, axis=1).sum()
            exact_matches += int(em_batch)

            for c in range(NUM_FOREGROUND):
                per_class_tp[c] += np.logical_and(binary_preds[:, c] == 1, binary_gt[:, c] == 1).sum()
                per_class_pred[c] += (binary_preds[:, c] == 1).sum()
                per_class_gt[c] += (binary_gt[:, c] == 1).sum()

        if self.scheduler is not None:
            self.scheduler.step()

        # Update gamma schedule for PDA
        if enable_pda and self.gamma > self.min_gamma:
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
        """Evaluate multi-label metrics (mAP, AUROC, Best-F1) and Gated Pseudo-Mask mIoU on val cohort."""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_probs: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        pm_inter = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_union = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_dice_num = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        pm_dice_den = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        has_valid_masks = False

        for batch in val_loader:
            if len(batch) == 4:
                images, targets, gt_masks, _ = batch
            else:
                images, targets, _ = batch
                gt_masks = None

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

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
                        cmin = cam_gated.view(b_sz, NUM_FOREGROUND, -1).min(dim=-1)[0].view(b_sz, NUM_FOREGROUND, 1, 1)
                        cmax = cam_gated.view(b_sz, NUM_FOREGROUND, -1).max(dim=-1)[0].view(b_sz, NUM_FOREGROUND, 1, 1)
                        cam_norm = ((cam_gated - cmin) / (cmax - cmin).clamp(min=1e-7)) * gate
                        # Mask inactive classes to -1.0 so zero-CAM ties never leak Class 0 (TUM)
                        cam_masked = torch.where(gate > 0.5, cam_norm, torch.full_like(cam_norm, -1.0))
                        pred_pm = torch.argmax(cam_masked, dim=1).cpu().numpy()

                        for c in range(NUM_FOREGROUND):
                            pc = (pred_pm == c) & valid_roi
                            gc = (gt_np == c) & valid_roi
                            tp = float(np.logical_and(pc, gc).sum())
                            pm_inter[c] += tp
                            pm_union[c] += float(np.logical_or(pc, gc).sum())
                            pm_dice_num[c] += 2.0 * tp
                            pm_dice_den[c] += float(pc.sum() + gc.sum())

        avg_loss = total_loss / max(total_samples, 1)
        if all_probs:
            y_prob = np.concatenate(all_probs, axis=0)
            y_true = np.concatenate(all_targets, axis=0)
            ml_metrics = compute_multilabel_validation_metrics(y_true, y_prob)
        else:
            ml_metrics = compute_multilabel_validation_metrics(
                np.zeros((0, NUM_FOREGROUND), dtype=np.float32),
                np.zeros((0, NUM_FOREGROUND), dtype=np.float32),
            )

        out: Dict[str, Any] = {
            "val_loss": avg_loss,
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
        return out


def build_optimizer_and_scheduler(
    model: WSSSTissueStage1NetCAM,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 20,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Build AdamW optimizer with 10x higher learning rate on classification head."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc8" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": 10.0 * lr, "weight_decay": 0.0},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    return optimizer, scheduler


def train_stage1_pipeline(
    data_dir: Union[str, Path],
    checkpoint_dir: Union[str, Path] = "checkpoints",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1.4e-4,
    loss_type: str = "soft_margin",
    resume_path: Optional[Union[str, Path]] = None,
    pretrained_path: Optional[Union[str, Path]] = None,
    allow_random_init: Optional[bool] = None,
    auto_download_domain_pretrained: bool = False,
    input_norm: str = "zero_one",
    dry_run: bool = False,
    device_str: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end Stage 1 classifier training pipeline with Rule 17 validation gate.

    Args:
        data_dir: Root dataset folder containing train/ and val/.
        checkpoint_dir: Folder to save checkpoints.
        epochs: Number of epochs to train (25 for BS=32 matches paper gradient step budget).
        batch_size: Mini-batch size.
        lr: Base learning rate (1.4e-4 for BS=32 via sqrt scaling from 1e-4 at BS=16).
        loss_type: 'soft_margin' (default, paper parity), 'asl', or 'bce'.
        resume_path: Checkpoint path to resume from.
        pretrained_path: Optional path to ImageNet (.params/.pth) or domain-pretrained ResNet-38d weights.
        allow_random_init: Allow training without pretrained backbone (defaults to dry_run).
        auto_download_domain_pretrained: Auto-download LUAD histopathology backbone if no local weights exist.
        input_norm: Input normalization contract ('zero_one' default for paper parity, or 'imagenet').
        dry_run: If True, executes 1 batch per epoch on 4 samples for smoke testing.
        device_str: 'cuda', 'cpu', etc.
        seed: Random seed for reproducibility.

    Returns:
        Summary metrics dictionary.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    norm_mode = str(input_norm or "zero_one").strip().lower()
    if norm_mode not in ("imagenet", "zero_one"):
        raise ValueError(f"Unsupported input_norm={input_norm!r}. Expected 'imagenet' or 'zero_one'.")

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)

    # Load pretrained backbone unless resuming from an existing Stage-1 checkpoint
    start_epoch = 0
    best_val_score = -1.0
    best_macro_f1 = 0.0
    pretrained_info: Dict[str, Any] = {
        "loaded_layers": 0,
        "total_backbone_layers": sum(1 for k in model.state_dict() if not k.startswith("fc8.")),
        "source": "resume" if (resume_path and Path(resume_path).exists()) else "random_init",
        "path": str(resume_path) if (resume_path and Path(resume_path).exists()) else None,
    }

    loaded_resume_obj: Optional[Dict[str, Any]] = None
    if resume_path and Path(resume_path).exists():
        try:
            loaded = torch.load(resume_path, map_location="cpu", weights_only=True)
        except Exception:
            loaded = torch.load(resume_path, map_location="cpu", weights_only=False)  # nosec B614
        if isinstance(loaded, dict):
            loaded_resume_obj = loaded
            state_dict = loaded.get("state_dict", loaded)
            start_epoch = int(loaded.get("epoch", -1)) + 1
            best_macro_f1 = float(loaded.get("macro_f1", 0.0))
            best_val_score = float(loaded.get("best_val_score", best_macro_f1))
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
    loss_key = str(loss_type or "soft_margin").strip().lower()
    if loss_key == "asl":
        criterion: nn.Module = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.02)
    elif loss_key == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        loss_key = "soft_margin"
        criterion = nn.MultiLabelSoftMarginLoss()

    optimizer, scheduler = build_optimizer_and_scheduler(model, lr=lr, epochs=epochs)
    if loaded_resume_obj is not None:
        if "optimizer" in loaded_resume_obj:
            optimizer.load_state_dict(loaded_resume_obj["optimizer"])
        if "scheduler" in loaded_resume_obj:
            scheduler.load_state_dict(loaded_resume_obj["scheduler"])

    # DataLoaders
    max_train = 4 if dry_run else None
    max_val = 4 if dry_run else None
    effective_epochs = 1 if dry_run else epochs
    effective_bs = 2 if dry_run else batch_size

    train_loader, val_loader = create_stage1_dataloaders(
        data_dir=data_dir,
        batch_size=effective_bs,
        num_workers=0 if dry_run else 2,
        pin_memory=False if dry_run else True,
        max_train_samples=max_train,
        max_val_samples=max_val,
        input_norm=norm_mode,
        return_val_masks=True,
    )

    if not dry_run and (val_loader is None or len(val_loader) == 0):
        raise RuntimeError(
            "[STAGE 1 ERROR] Validation split not found or empty. "
            "Refusing blind training without an independent validation loop (Rule 17)."
        )

    trainer = Stage1Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        enable_amp=not dry_run,
    )

    history: List[Dict[str, Any]] = []
    best_save_path = checkpoint_path / "stage1_retrained_best.pth"
    latest_save_path = checkpoint_path / "stage1_latest.pth"
    selection_metric = "val_mAP"

    for epoch in range(start_epoch, effective_epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)

        val_metrics: Dict[str, Any] = {}
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = trainer.evaluate(val_loader)

        epoch_record = {**train_metrics, **val_metrics, "epoch": epoch}
        history.append(epoch_record)

        # Select best checkpoint strictly by independent validation metric (Rule 17)
        if "val_gated_pm_miou" in val_metrics:
            selection_metric = "val_gated_pm_miou"
            current_val_score = float(val_metrics["val_gated_pm_miou"])
        elif "val_mAP" in val_metrics:
            selection_metric = "val_mAP"
            current_val_score = float(val_metrics["val_mAP"])
        else:
            selection_metric = "val_macro_f1" if "val_macro_f1" in val_metrics else "train_macro_f1_dry_run"
            current_val_score = float(val_metrics.get("val_macro_f1", train_metrics["macro_f1"]))

        current_f1 = float(val_metrics.get("val_best_f1", val_metrics.get("val_macro_f1", train_metrics["macro_f1"])))
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1

        improved = current_val_score >= best_val_score
        if improved:
            best_val_score = max(best_val_score, current_val_score)

        compact_payload = {
            "epoch": epoch,
            "macro_f1": current_f1,
            "best_val_score": max(0.0, best_val_score),
            "best_val_metric": selection_metric,
            "val_mAP": float(val_metrics.get("val_mAP", 0.0)),
            "val_auroc": float(val_metrics.get("val_auroc", 0.0)),
            "val_best_f1": float(val_metrics.get("val_best_f1", 0.0)),
            "val_gated_pm_miou": float(val_metrics.get("val_gated_pm_miou", 0.0)),
            "input_norm": norm_mode,
            "loss_type": loss_key,
            "pretrained_source": pretrained_info["source"],
            "pretrained_loaded_layers": pretrained_info["loaded_layers"],
            "state_dict": model.state_dict(),
            "metrics": epoch_record,
        }

        if not dry_run:
            latest_payload = {
                **compact_payload,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(latest_payload, latest_save_path)

        if improved or dry_run or not best_save_path.is_file():
            torch.save(compact_payload, best_save_path)

    return {
        "best_macro_f1": best_macro_f1,
        "best_val_score": max(0.0, best_val_score),
        "best_val_metric": selection_metric,
        "input_norm": norm_mode,
        "loss_type": loss_key,
        "pretrained_info": pretrained_info,
        "checkpoint_path": str(best_save_path),
        "latest_checkpoint_path": str(latest_save_path) if latest_save_path.is_file() else None,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 ResNet-38d Classifier on BCSS-WSSS")
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="Dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1.4e-4, help="Learning rate")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="soft_margin",
        choices=["soft_margin", "asl", "bce"],
        help="Loss type (default: soft_margin for paper parity)",
    )
    parser.add_argument(
        "--input-norm",
        type=str,
        default="zero_one",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_stage1_pipeline(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss_type,
        resume_path=args.resume,
        pretrained_path=args.pretrained_path,
        allow_random_init=args.allow_random_init if args.allow_random_init else None,
        auto_download_domain_pretrained=args.auto_download_domain_pretrained,
        input_norm=args.input_norm,
        dry_run=args.dry_run,
        device_str=args.device,
        seed=args.seed,
    )
    print("Stage 1 Training Complete.")
    print(f"Best Val Metric ({results['best_val_metric']}): {results['best_val_score']:.4f}")
    print(f"Best Macro F1: {results['best_macro_f1']:.4f}")
    print(f"Saved Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
