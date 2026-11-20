"""
Stage-2 DeepLabV3+ Segmentation Training Engine for Histopathology WSSS.

Trains the 59M-parameter DeepLabV3+ (ImageNet-pretrained ResNet-101 backbone + ASPP) on generated
multi-layer pseudo-masks with:
1. Multi-Layer Pseudo-Supervision (MLPS) Loss (Han et al., MedIA 2022):
   L_total = L_mlps + 0.20 * L_unc (+ optional uabl_weight * L_uabl)
   with paper-parity layer weights: bn7=0.6, b5_2=0.2, b4_5=0.2.
2. Boundary Uncertainty Regularization Loss: down-weights ambiguous transition pixels.
3. Classification Gate Mechanism (CGM) during validation and evaluation.
4. Synchronized patient-level separation and zero patient leakage (Invariant 08).
5. Poly/Cosine learning rate scheduling with layer-wise multiplier (10x on decoder).
6. Compact inference-only checkpoint saving to checkpoints/stage2_retrained_best.pth
   (with optimizer/scheduler state persisted in checkpoints/stage2_latest.pth for resume).
7. Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

import argparse
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.losses_metrics.losses import (
    MultiLayerPseudoSupervisionLoss,
    UncertaintyAwareBoundaryLoss,
    UncertaintyRegularizationLoss,
)
from src.models.stain_deconv import StainDeconvLayer
from src.models.wsss_inference_methods import DeepLabV3Plus, WSSSTissueStage1NetCAM
from src.training.dataset_loader import (
    NUM_CLASSES,
    NUM_FOREGROUND,
    create_eval_dataloader,
    create_stage2_dataloader,
    load_bcss_patient_splits,
)

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
VALID_STAGE1_NORM_MODES: Tuple[str, ...] = ("auto", "zero_one", "imagenet")


def infer_stage1_norm_mode(
    checkpoint_obj: Any,
    checkpoint_path: Optional[Union[str, Path]] = None,
    requested_mode: str = "auto",
) -> str:
    """Infers expected Stage-1 input normalization mode ('zero_one' or 'imagenet')."""
    mode = str(requested_mode or "auto").strip().lower()
    if mode not in VALID_STAGE1_NORM_MODES:
        raise ValueError(
            f"Unsupported stage1_norm_mode={requested_mode!r}. Expected one of {VALID_STAGE1_NORM_MODES}."
        )
    if mode != "auto":
        return mode

    if isinstance(checkpoint_obj, dict):
        if "input_norm" in checkpoint_obj and checkpoint_obj["input_norm"] is not None:
            norm_val = str(checkpoint_obj["input_norm"]).strip().lower()
            if norm_val not in ("zero_one", "imagenet"):
                raise ValueError(
                    f"Unsupported checkpoint input_norm={checkpoint_obj['input_norm']!r}. "
                    "Expected 'zero_one' or 'imagenet'."
                )
            return norm_val
        if "state_dict" in checkpoint_obj and any(
            k in checkpoint_obj for k in ("epoch", "macro_f1", "optimizer", "metrics")
        ):
            return "imagenet"

    return "zero_one"


def prepare_stage1_input(
    images: torch.Tensor,
    stage1_norm_mode: str,
    images_are_imagenet_norm: bool = True,
) -> torch.Tensor:
    """Aligns input tensor range/normalization with Stage-1 checkpoint expectations."""
    mode = str(stage1_norm_mode or "zero_one").strip().lower()
    view_shape = (1, 3, 1, 1) if images.dim() == 4 else (3, 1, 1)

    if mode == "zero_one" and images_are_imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(*view_shape)
        std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(*view_shape)
        return torch.clamp(images * std + mean, 0.0, 1.0)

    if mode == "imagenet" and not images_are_imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(*view_shape)
        std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(*view_shape)
        return (images - mean) / std

    return images


def load_stage1_checkpoint_into_model(
    model: WSSSTissueStage1NetCAM,
    checkpoint_path: Union[str, Path],
    stage1_norm_mode: str = "auto",
    min_matched_layers: int = 100,
    allow_random_init: bool = False,
) -> Tuple[str, int]:
    """Loads Stage-1 ResNet-38d checkpoint with prefix stripping and >=100-layer verification.

    Strips leading 'module.' / 'model.' prefixes, checks tensor shape compatibility, and raises
    FileNotFoundError / RuntimeError on missing, corrupt, or under-loaded checkpoints unless
    allow_random_init=True is explicitly requested.
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        msg = f"[STAGE 1 ERROR] Stage-1 checkpoint not found at: {ckpt_p}"
        if allow_random_init:
            warnings.warn(
                f"{msg}; continuing with random Stage-1 weights ONLY because allow_random_init=True.",
                UserWarning,
                stacklevel=2,
            )
            fallback_norm = infer_stage1_norm_mode(
                None, checkpoint_path=ckpt_p, requested_mode=stage1_norm_mode
            )
            return fallback_norm, 0
        raise FileNotFoundError(msg)

    try:
        try:
            state = torch.load(ckpt_p, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(ckpt_p, map_location="cpu", weights_only=False)  # nosec B614
    except Exception as exc:
        msg = f"[STAGE 1 ERROR] Failed to load corrupt Stage-1 checkpoint {ckpt_p}: {exc}"
        if allow_random_init:
            warnings.warn(
                f"{msg}; continuing with random Stage-1 weights ONLY because allow_random_init=True.",
                UserWarning,
                stacklevel=2,
            )
            fallback_norm = infer_stage1_norm_mode(
                None, checkpoint_path=ckpt_p, requested_mode=stage1_norm_mode
            )
            return fallback_norm, 0
        raise RuntimeError(msg) from exc

    resolved_norm = infer_stage1_norm_mode(
        state,
        checkpoint_path=ckpt_p,
        requested_mode=stage1_norm_mode,
    )
    if (
        str(stage1_norm_mode or "auto").strip().lower() == "auto"
        and ckpt_p.name == "stage1_checkpoint_trained_on_bcss.pth"
        and not (isinstance(state, dict) and "input_norm" in state)
    ):
        resolved_norm = "zero_one"

    raw_sd = state.get("state_dict", state.get("model", state)) if isinstance(state, dict) else state
    if not isinstance(raw_sd, dict):
        msg = (
            f"[STAGE 1 ERROR] Invalid Stage-1 checkpoint payload in {ckpt_p}: "
            f"expected state_dict mapping, got {type(raw_sd).__name__}."
        )
        if allow_random_init:
            warnings.warn(
                f"{msg} Continuing with random Stage-1 weights ONLY because allow_random_init=True.",
                UserWarning,
                stacklevel=2,
            )
            return resolved_norm, 0
        raise RuntimeError(msg)

    cleaned_sd: Dict[str, Any] = {}
    for k, v in raw_sd.items():
        clean_k = str(k)
        while clean_k.startswith("module.") or clean_k.startswith("model."):
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            elif clean_k.startswith("model."):
                clean_k = clean_k[6:]
        cleaned_sd[clean_k] = v

    model_dict = model.state_dict()
    matched = {
        k: v
        for k, v in cleaned_sd.items()
        if k in model_dict and isinstance(v, torch.Tensor) and v.shape == model_dict[k].shape
    }
    matched_count = len(matched)
    if matched_count < min_matched_layers:
        msg = (
            f"[STAGE 1 ERROR] Loaded only {matched_count}/{len(model_dict)} matched ResNet-38d layers "
            f"from {ckpt_p} (expected >= {min_matched_layers}). Refusing under-loaded Stage-1 weights."
        )
        if allow_random_init:
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            raise RuntimeError(msg)

    model.load_state_dict(matched, strict=False)
    return resolved_norm, matched_count


def load_imagenet_pretrained_resnet101(
    model: DeepLabV3Plus,
    allow_random_init_fallback: bool = False,
    min_loaded_layers: int = 100,
) -> int:
    """Loads torchvision ImageNet pretrained weights into DeepLabV3Plus ResNet101 backbone.

    Raises RuntimeError on failure or under-loading unless allow_random_init_fallback=True
    (prevents silent training from random ResNet-101 weights per Rule 17).
    """
    model_dict = model.backbone.state_dict()
    try:
        import torchvision.models as tv_models
        try:
            weights = tv_models.ResNet101_Weights.IMAGENET1K_V1
            tv_model = tv_models.resnet101(weights=weights)
        except Exception:
            tv_model = tv_models.resnet101(pretrained=True)
        pretrained_dict = tv_model.state_dict()
        filtered = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        loaded_count = len(filtered)
        if loaded_count < min_loaded_layers:
            msg = (
                f"[STAGE 2 ERROR] Loaded only {loaded_count}/{len(model_dict)} ImageNet ResNet-101 "
                f"layers (expected >= {min_loaded_layers}). Refusing silent random-init fallback (Rule 17)."
            )
            if allow_random_init_fallback:
                warnings.warn(msg, UserWarning, stacklevel=2)
                return loaded_count
            raise RuntimeError(msg)
        model_dict.update(filtered)
        model.backbone.load_state_dict(model_dict)
        print(f"[STAGE 2] Loaded {loaded_count}/{len(model_dict)} layers from ImageNet ResNet-101 into backbone.")
        return loaded_count
    except RuntimeError:
        raise
    except Exception as e:
        msg = (
            f"[STAGE 2 ERROR] Failed to load ImageNet ResNet-101 pretrained weights ({e}). "
            "Pass allow_random_init_fallback=True only for offline/unit-test runs."
        )
        if allow_random_init_fallback:
            warnings.warn(
                f"[STAGE 2 WARN] ImageNet weights not loaded ({e}); continuing with initialized "
                "backbone ONLY because allow_random_init_fallback=True.",
                UserWarning,
                stacklevel=2,
            )
            return 0
        raise RuntimeError(msg) from e


class Stage2Trainer:
    """Orchestrates Stage-2 DeepLabV3+ segmentation training and evaluation."""

    def __init__(
        self,
        model: DeepLabV3Plus,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        stage1_model: Optional[WSSSTissueStage1NetCAM] = None,
        device: Optional[torch.device] = None,
        enable_amp: bool = True,
        unc_weight: float = 0.20,
        uabl_weight: float = 0.0,
        stage1_norm_mode: str = "zero_one",
        mlps_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stage1_model = stage1_model.to(self.device).eval() if stage1_model else None
        self.stage1_norm_mode = str(stage1_norm_mode or "zero_one").strip().lower()

        self.enable_amp = enable_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_amp)

        self.mlps_loss = MultiLayerPseudoSupervisionLoss(
            weights=mlps_weights or {"bn7": 0.6, "b5_2": 0.2, "b4_5": 0.2},
            ignore_index=255,
        )
        self.unc_loss = UncertaintyRegularizationLoss(entropy_threshold=0.65, ignore_index=255)
        self.unc_weight = unc_weight
        self.uabl_weight = uabl_weight
        self.uabl_loss = UncertaintyAwareBoundaryLoss() if uabl_weight > 0.0 else None
        self.stain_deconv = StainDeconvLayer().to(self.device).eval() if uabl_weight > 0.0 else None

    def _prepare_logits(self, logits: torch.Tensor, target_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ensure logits have 5 channels without leaking target_masks into model predictions."""
        if logits.shape[1] == 4:
            # Append neutral background logit (0.0) without reading target_masks (Zero GT leakage)
            bg_channel = torch.zeros(
                (logits.shape[0], 1, logits.shape[2], logits.shape[3]),
                device=logits.device,
                dtype=logits.dtype,
            )
            return torch.cat([logits, bg_channel], dim=1)
        return logits

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Execute one complete training epoch over pseudo-masks."""
        self.model.train()
        total_loss = 0.0
        total_mlps = 0.0
        total_unc = 0.0
        batch_count = 0

        for sample in train_loader:
            images = sample["image"].to(self.device, non_blocking=True)
            mask_bn7 = sample["label"].to(self.device, non_blocking=True)
            mask_b5_2 = sample["label_a"].to(self.device, non_blocking=True)
            mask_b4_5 = sample["label_b"].to(self.device, non_blocking=True)

            pseudo_masks = {
                "bn7": mask_bn7,
                "b5_2": mask_b5_2,
                "b4_5": mask_b4_5,
            }

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.enable_amp):
                raw_logits = self.model(images)
                full_logits = self._prepare_logits(raw_logits, mask_bn7)

                l_mlps = self.mlps_loss(full_logits, pseudo_masks)
                l_unc = self.unc_loss(full_logits, mask_bn7)
                loss = l_mlps + self.unc_weight * l_unc
                if self.uabl_loss is not None and self.stain_deconv is not None:
                    # Compute optical stain boundary gradient in strict FP32 with autocast disabled
                    with torch.amp.autocast(device_type=images.device.type, enabled=False):
                        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=torch.float32).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=torch.float32).view(1, 3, 1, 1)
                        img_01 = torch.clamp(images.float() * std + mean, 0.0, 1.0)
                        he_channels, _, _ = self.stain_deconv(img_01)
                        stain_grad = self.stain_deconv.compute_stain_gradient(he_channels)
                        pred_probs = F.softmax(full_logits.float(), dim=1)
                        entropy = -(pred_probs * torch.log(pred_probs.clamp(min=1e-6))).sum(dim=1, keepdim=True)
                        entropy_norm = entropy / np.log(float(full_logits.shape[1]))
                        l_uabl = self.uabl_loss(pred_probs, stain_grad, entropy_norm.detach())
                    loss = loss + self.uabl_weight * l_uabl

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_mlps += l_mlps.item()
            total_unc += l_unc.item()
            batch_count += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "train_loss": total_loss / max(batch_count, 1),
            "mlps_loss": total_mlps / max(batch_count, 1),
            "unc_loss": total_unc / max(batch_count, 1),
        }

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        use_cgm: bool = True,
    ) -> Dict[str, Any]:
        """Compute pixel-level confusion matrix and per-class IoU on validation/test set."""
        self.model.eval()
        confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        for images, targets, cls_labels, _ in val_loader:
            images = images.to(self.device, non_blocking=True)
            targets_np = targets.numpy()  # (B, H, W)

            raw_logits = self.model(images)

            # Apply Classification Gate Mechanism (CGM) if available
            if use_cgm and self.stage1_model is not None:
                s1_images = prepare_stage1_input(
                    images,
                    stage1_norm_mode=self.stage1_norm_mode,
                    images_are_imagenet_norm=True,
                )
                stage1_cam, stage1_probs = self.stage1_model.forward_cam(s1_images)
                active_classes = (stage1_probs > 0.10).float()
                # Suppress inactive foreground channels (0..3)
                for c in range(NUM_FOREGROUND):
                    gate = active_classes[:, c].view(-1, 1, 1, 1)
                    raw_logits[:, c : c + 1] = torch.where(
                        gate == 1.0,
                        raw_logits[:, c : c + 1],
                        torch.full_like(raw_logits[:, c : c + 1], -1e4),
                    )

            if raw_logits.shape[1] == 4:
                # Add background channel with neutral logit 0.0
                bg_logit = torch.zeros((raw_logits.shape[0], 1, raw_logits.shape[2], raw_logits.shape[3]), device=self.device)
                pred_logits = torch.cat([raw_logits, bg_logit], dim=1)
            else:
                pred_logits = raw_logits

            preds = torch.argmax(pred_logits, dim=1).cpu().numpy()  # (B, H, W)
            # Exclude outside-ROI background (class 4) per repos/WSSS-Tissue/3_train_stage2.py
            preds_roi = preds.copy()
            preds_roi[targets_np == 4] = 4

            # Accumulate 5x5 confusion matrix (with outside-ROI class 4 excluded from foreground FP)
            mask = (targets_np >= 0) & (targets_np < NUM_CLASSES)
            cm_batch = np.bincount(
                NUM_CLASSES * targets_np[mask] + preds_roi[mask],
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            confusion_matrix += cm_batch

        # Calculate per-class IoU and Dice
        ious = []
        dices = []
        for c in range(NUM_CLASSES):
            tp = confusion_matrix[c, c]
            fp = confusion_matrix[:, c].sum() - tp
            fn = confusion_matrix[c, :].sum() - tp
            denom = tp + fp + fn
            iou = (tp / denom) if denom > 0 else 0.0
            dice = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
            ious.append(float(iou))
            dices.append(float(dice))

        # Primary 4-class foreground mIoU and Macro-Dice (excluding class 4 background)
        fg_ious = [ious[c] for c in range(NUM_FOREGROUND) if (confusion_matrix[c, :].sum() + confusion_matrix[:, c].sum()) > 0]
        fg_dices = [dices[c] for c in range(NUM_FOREGROUND) if (confusion_matrix[c, :].sum() + confusion_matrix[:, c].sum()) > 0]
        miou_4cls = float(np.mean(fg_ious)) if fg_ious else 0.0
        mdice_4cls = float(np.mean(fg_dices)) if fg_dices else 0.0
        miou_5cls = float(np.mean(ious))

        return {
            "mIoU": miou_4cls,
            "mDice": mdice_4cls,
            "mIoU_4cls": miou_4cls,
            "mDice_4cls": mdice_4cls,
            "mIoU_5cls": miou_5cls,
            "iou_tum": ious[0],
            "iou_str": ious[1],
            "iou_lym": ious[2],
            "iou_nec": ious[3],
            "iou_back": ious[4],
            "confusion_matrix": confusion_matrix.tolist(),
        }


def build_stage2_optimizer_and_scheduler(
    model: DeepLabV3Plus,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 30,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Build AdamW optimizer with 10x higher learning rate on ASPP and Decoder."""
    backbone_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr, "weight_decay": weight_decay},
        {"params": decoder_params, "lr": 10.0 * lr, "weight_decay": 0.0},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    return optimizer, scheduler


def train_stage2_pipeline(
    data_dir: Union[str, Path],
    pm_dir: Optional[Union[str, Path]] = None,
    stage1_checkpoint: Optional[Union[str, Path]] = "checkpoints/stage1_checkpoint_trained_on_bcss.pth",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-4,
    resume_path: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    device_str: Optional[str] = None,
    seed: int = 42,
    use_uabl: bool = False,
    stage1_norm_mode: str = "auto",
    allow_random_init_fallback: Optional[bool] = None,
    allow_random_init: bool = False,
    unc_weight: float = 0.20,
    uabl_weight: float = 0.05,
) -> Dict[str, Any]:
    """Execute end-to-end Stage 2 DeepLabV3+ segmentation training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    if (data_path / "training").exists():
        train_img_dir = data_path / "training"
    elif (data_path / "train").exists():
        train_img_dir = data_path / "train"
    elif (data_path / "BCSS-WSSS" / "training").exists():
        train_img_dir = data_path / "BCSS-WSSS" / "training"
    elif (data_path / "BCSS-WSSS" / "train").exists():
        train_img_dir = data_path / "BCSS-WSSS" / "train"
    else:
        train_img_dir = data_path

    val_img_dir = None
    val_mask_dir = None
    for cand_v, cand_m in [
        (data_path / "val" / "img", data_path / "val" / "mask"),
        (data_path / "valid" / "img", data_path / "valid" / "mask"),
        (data_path / "BCSS-WSSS" / "val" / "img", data_path / "BCSS-WSSS" / "val" / "mask"),
        (data_path / "BCSS-WSSS" / "valid" / "img", data_path / "BCSS-WSSS" / "valid" / "mask"),
    ]:
        if cand_v.exists():
            val_img_dir = cand_v
            val_mask_dir = cand_m if cand_m.exists() else None
            break

    # Load patient splits for strict separation
    splits = load_bcss_patient_splits()
    train_patients = splits.get("train", [])
    val_patients = splits.get("val", [])

    # Initialize Stage 2 DeepLabV3Plus (5 classes: TUM, STR, LYM, NEC, BACK)
    model = DeepLabV3Plus(num_classes=5, output_stride=16)
    effective_fallback = dry_run if allow_random_init_fallback is None else bool(allow_random_init_fallback)
    loaded_backbone_layers = load_imagenet_pretrained_resnet101(
        model,
        allow_random_init_fallback=effective_fallback,
    )

    # Optionally load Stage 1 model for CGM gating during eval
    stage1_model: Optional[WSSSTissueStage1NetCAM] = None
    resolved_stage1_norm = infer_stage1_norm_mode(
        None, checkpoint_path=stage1_checkpoint, requested_mode=stage1_norm_mode
    )
    if stage1_checkpoint is not None:
        stage1_model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
        resolved_stage1_norm, _ = load_stage1_checkpoint_into_model(
            stage1_model,
            checkpoint_path=stage1_checkpoint,
            stage1_norm_mode=stage1_norm_mode,
            min_matched_layers=100,
            allow_random_init=allow_random_init,
        )

    # Resume from checkpoint if requested
    start_epoch = 0
    best_miou = 0.0
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
            best_miou = float(loaded.get("mIoU", 0.0))
        else:
            state_dict = loaded
        model.load_state_dict(state_dict, strict=False)

    optimizer, scheduler = build_stage2_optimizer_and_scheduler(model, lr=lr, epochs=epochs)
    if loaded_resume_obj is not None:
        if "optimizer" in loaded_resume_obj:
            optimizer.load_state_dict(loaded_resume_obj["optimizer"])
        if "scheduler" in loaded_resume_obj:
            scheduler.load_state_dict(loaded_resume_obj["scheduler"])

    effective_epochs = 1 if dry_run else epochs
    effective_bs = 2 if dry_run else batch_size
    max_train = 4 if dry_run else None
    max_val = 4 if dry_run else None

    train_loader = create_stage2_dataloader(
        img_dir=train_img_dir,
        pm_root_dir=pm_dir,
        batch_size=effective_bs,
        num_workers=0 if dry_run else 2,
        pin_memory=False if dry_run else True,
        patient_ids=train_patients if train_patients else None,
        max_samples=max_train,
    )

    val_loader: Optional[DataLoader] = None
    if val_img_dir and val_img_dir.exists():
        val_loader = create_eval_dataloader(
            img_dir=val_img_dir,
            mask_dir=val_mask_dir,
            batch_size=effective_bs,
            num_workers=0 if dry_run else 2,
            pin_memory=False if dry_run else True,
            patient_ids=val_patients if val_patients else None,
            max_samples=max_val,
        )

    trainer = Stage2Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stage1_model=stage1_model,
        device=device,
        enable_amp=not dry_run,
        unc_weight=unc_weight,
        uabl_weight=uabl_weight if use_uabl else 0.0,
        stage1_norm_mode=resolved_stage1_norm,
    )

    history: List[Dict[str, Any]] = []
    best_save_path = checkpoint_path / "stage2_retrained_best.pth"
    latest_save_path = checkpoint_path / "stage2_latest.pth"

    for epoch in range(start_epoch, effective_epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)

        eval_metrics: Dict[str, Any] = {}
        if val_loader is not None and len(val_loader) > 0:
            eval_metrics = trainer.evaluate(val_loader, use_cgm=(stage1_model is not None))

        epoch_record = {**train_metrics, **eval_metrics, "epoch": epoch}
        history.append(epoch_record)

        current_miou = eval_metrics.get("mIoU", 0.0)
        improved = current_miou >= best_miou
        if improved:
            best_miou = current_miou

        compact_payload = {
            "epoch": epoch,
            "mIoU": best_miou,
            "input_norm": "imagenet",
            "pretrained_backbone_layers": loaded_backbone_layers,
            "stage1_checkpoint": str(stage1_checkpoint) if stage1_checkpoint else None,
            "stage1_norm_mode": resolved_stage1_norm,
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
        "best_mIoU": best_miou,
        "pretrained_backbone_layers": loaded_backbone_layers,
        "checkpoint_path": str(best_save_path),
        "latest_checkpoint_path": str(latest_save_path) if latest_save_path.is_file() else None,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 DeepLabV3+ on Pseudo-Masks")
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="Dataset directory")
    parser.add_argument("--pm-dir", type=str, default=None, help="Pseudo-masks root directory")
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default="checkpoints/stage1_checkpoint_trained_on_bcss.pth",
        help="Stage-1 checkpoint path (default: verified author baseline)",
    )
    parser.add_argument(
        "--stage1-norm-mode",
        type=str,
        default="auto",
        choices=["auto", "zero_one", "imagenet"],
        help="Stage-1 input normalization mode (auto, zero_one, or imagenet)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-step dry run for testing")
    parser.add_argument("--use-uabl", action="store_true", help="Enable Uncertainty-Aware Optical Stain Boundary Loss")
    parser.add_argument("--allow-random-init", action="store_true", help="Allow offline/random ResNet-101 init fallback")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_stage2_pipeline(
        data_dir=args.data_dir,
        pm_dir=args.pm_dir,
        stage1_checkpoint=args.stage1_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_path=args.resume,
        dry_run=args.dry_run,
        device_str=args.device,
        seed=args.seed,
        use_uabl=args.use_uabl,
        stage1_norm_mode=args.stage1_norm_mode,
        allow_random_init_fallback=args.allow_random_init if args.allow_random_init else None,
        allow_random_init=bool(args.allow_random_init),
    )
    print("Stage 2 Training Complete.")
    print(f"Best mIoU: {results['best_mIoU']:.4f}")
    print(f"Saved Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
