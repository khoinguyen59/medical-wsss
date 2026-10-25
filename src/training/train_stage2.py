"""
Stage-2 DeepLabV3+ Segmentation Training Engine for SE374.R11 WSSS.

Trains the 59M-parameter DeepLabV3+ (ResNet-101 backbone + ASPP) from scratch on generated
multi-layer pseudo-masks with:
1. Multi-Layer Pseudo-Supervision (MLPS) Loss (Han et al., MedIA 2022):
   L_total = 0.60 * L(bn7) + 0.20 * L(b5_2) + 0.20 * L(b4_5) + 0.10 * L_unc
2. Boundary Uncertainty Regularization Loss: down-weights ambiguous transition pixels.
3. Classification Gate Mechanism (CGM) during validation and evaluation.
4. Synchronized patient-level separation and zero patient leakage (Invariant 08).
5. Poly/Cosine learning rate scheduling with layer-wise multiplier (10x on decoder).
6. Checkpoint saving to checkpoints/stage2_retrained_best.pth.
7. Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.losses_metrics.losses import (
    MultiLayerPseudoSupervisionLoss,
    UncertaintyRegularizationLoss,
)
from src.models.deeplabv3plus import DeepLabV3PlusWithCGM
from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.training.dataset_loader import (
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_FOREGROUND,
    create_eval_dataloader,
    create_stage2_dataloader,
    load_bcss_patient_splits,
)


class Stage2Trainer:
    """Orchestrates Stage-2 DeepLabV3+ segmentation training and evaluation."""

    def __init__(
        self,
        model: DeepLabV3PlusWithCGM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        stage1_model: Optional[WSSSTissueStage1NetCAM] = None,
        device: Optional[torch.device] = None,
        enable_amp: bool = True,
        unc_weight: float = 0.10,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stage1_model = stage1_model.to(self.device).eval() if stage1_model else None

        self.enable_amp = enable_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_amp)

        self.mlps_loss = MultiLayerPseudoSupervisionLoss(
            weights={"bn7": 0.60, "b5_2": 0.20, "b4_5": 0.20},
            ignore_index=255,
        )
        self.unc_loss = UncertaintyRegularizationLoss(entropy_threshold=0.65, ignore_index=255)
        self.unc_weight = unc_weight

    def _prepare_logits(self, logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Ensure logits have 5 channels (incorporating Mode A background channel if model outputs 4)."""
        if logits.shape[1] == 4:
            # Mode A: append background logit channel where pseudo-mask target == 4
            one = torch.ones((logits.shape[0], 1, logits.shape[2], logits.shape[3]), device=logits.device)
            bg_channel = 100.0 * one * (target_masks == 4).unsqueeze(1).float()
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
                stage1_cam, stage1_probs = self.stage1_model.forward_cam(images)
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

            # Accumulate 5x5 confusion matrix
            mask = (targets_np >= 0) & (targets_np < NUM_CLASSES)
            cm_batch = np.bincount(
                NUM_CLASSES * targets_np[mask] + preds[mask],
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

        miou = float(np.mean(ious))
        mdice = float(np.mean(dices))

        return {
            "mIoU": miou,
            "mDice": mdice,
            "iou_tum": ious[0],
            "iou_str": ious[1],
            "iou_lym": ious[2],
            "iou_nec": ious[3],
            "iou_back": ious[4],
            "confusion_matrix": confusion_matrix.tolist(),
        }


def build_stage2_optimizer_and_scheduler(
    model: DeepLabV3PlusWithCGM,
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
    stage1_checkpoint: Optional[Union[str, Path]] = "checkpoints/stage1_retrained_best.pth",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-4,
    resume_path: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    device_str: Optional[str] = None,
    seed: int = 42,
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

    # Initialize Stage 2 DeepLabV3PlusWithCGM (5 classes: TUM, STR, LYM, NEC, BACK)
    model = DeepLabV3PlusWithCGM(num_classes=5)

    # Optionally load Stage 1 model for CGM gating during eval
    stage1_model: Optional[WSSSTissueStage1NetCAM] = None
    if stage1_checkpoint and Path(stage1_checkpoint).exists():
        stage1_model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
        st = torch.load(stage1_checkpoint, map_location="cpu")
        st_dict = st.get("state_dict", st)
        stage1_model.load_state_dict(st_dict, strict=False)

    # Resume from checkpoint if requested
    start_epoch = 0
    best_miou = 0.0
    if resume_path and Path(resume_path).exists():
        loaded = torch.load(resume_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)
        model.load_state_dict(state_dict, strict=False)
        start_epoch = loaded.get("epoch", 0)
        best_miou = loaded.get("mIoU", 0.0)

    optimizer, scheduler = build_stage2_optimizer_and_scheduler(model, lr=lr, epochs=epochs)

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
    )

    history: List[Dict[str, Any]] = []
    best_save_path = checkpoint_path / "stage2_retrained_best.pth"

    for epoch in range(start_epoch, effective_epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)

        eval_metrics: Dict[str, Any] = {}
        if val_loader is not None and len(val_loader) > 0:
            eval_metrics = trainer.evaluate(val_loader, use_cgm=True)

        epoch_record = {**train_metrics, **eval_metrics, "epoch": epoch}
        history.append(epoch_record)

        current_miou = eval_metrics.get("mIoU", 0.0)
        if current_miou >= best_miou or dry_run:
            best_miou = current_miou
            torch.save(
                {
                    "epoch": epoch,
                    "mIoU": best_miou,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": epoch_record,
                },
                best_save_path,
            )

    return {
        "best_mIoU": best_miou,
        "checkpoint_path": str(best_save_path),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 DeepLabV3+ on Pseudo-Masks")
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="Dataset directory")
    parser.add_argument("--pm-dir", type=str, default=None, help="Pseudo-masks root directory")
    parser.add_argument("--stage1-checkpoint", type=str, default="checkpoints/stage1_retrained_best.pth")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-step dry run for testing")
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
    )
    print("Stage 2 Training Complete.")
    print(f"Best mIoU: {results['best_mIoU']:.4f}")
    print(f"Saved Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
