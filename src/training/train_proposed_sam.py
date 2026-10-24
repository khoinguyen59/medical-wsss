"""
Training Script for Proposed MedSAM WSSS Adapter with Parameter-Efficient Fine-Tuning.

Course: SE374.R11 - Medical Image Processing Software Development
Grounding:
- S2C (CVPR 2024), SAM2PATH / MedSAM (Nature Communications 2024)
- Multi-Class Composite Loss: Focal Loss (15:1 imbalance) + Soft Dice + Mask IoU + IoU Head MSE
- Optimizer: AdamW + Cosine Annealing LR Schedule + Mixed Precision (AMP)
- Epoch checkpointing with resume capability and --dry-run validation

Strict Invariant 12 compliance: 100% ASCII 0x2D hyphens only.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Safeguard against OpenMP library collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.sam_wsss_adapter import (
    CAMPromptGenerator,
    MedSAM_WSSS_Adapter,
)


# ==============================================================================
# 1. Composite WSSS Loss Function
# ==============================================================================

class CompositeWSSSLoss(nn.Module):
    """Composite Loss for Histopathology Weakly-Supervised Semantic Segmentation.

    Components:
        1. Multi-Class Focal Loss: Mitigates 15:1 imbalance for LYM and NEC classes.
        2. Multi-Class Soft Dice Loss: Maximizes spatial overlap of tissue contours.
        3. Multi-Class IoU / Jaccard Loss: Directly optimizes benchmark mIoU metric.
        4. IoU Prediction Head MSE Loss: Calibrates predicted mask quality confidence.

    Total Loss:
        L_total = lambda_focal * L_focal + lambda_dice * L_dice + lambda_iou * L_iou + lambda_head * L_head
    """

    def __init__(
        self,
        num_classes: int = 5,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        iou_weight: float = 0.5,
        head_weight: float = 0.2,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.head_weight = head_weight
        self.focal_gamma = focal_gamma
        self.ignore_index = ignore_index
        self.epsilon = epsilon

        # Default class weights prioritizing minority classes: TUM=1.0, STR=1.0, LYM=2.5, NEC=2.5, BACK=0.5
        if class_weights is None:
            class_weights = [1.0, 1.0, 2.5, 2.5, 0.5]
        self.register_buffer("alpha", torch.tensor(class_weights, dtype=torch.float32))

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
        iou_predictions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            pred_logits: Tensor of shape (B, 5, H, W).
            target_labels: Tensor of shape (B, H, W) int64 in {0, 1, 2, 3, 4, 255}.
            iou_predictions: Optional tensor of shape (B, 5) from mask decoder.

        Returns:
            Dict containing 'total_loss', 'focal_loss', 'dice_loss', 'iou_loss', 'head_loss'.
        """
        device = pred_logits.device
        b, num_cls, h, w = pred_logits.shape

        # Valid pixel mask (excluding ignore_index 255)
        valid_mask = (target_labels != self.ignore_index)
        num_valid = valid_mask.sum().clamp(min=1.0).float()

        # Probabilities via softmax
        probs = F.softmax(pred_logits, dim=1)

        # ----------------------------------------------------------------------
        # 1. Multi-Class Focal Loss
        # ----------------------------------------------------------------------
        # Clamp target labels for one-hot encoding
        safe_targets = target_labels.clone()
        safe_targets[~valid_mask] = 0

        # One-hot encoding of targets: (B, 5, H, W)
        targets_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Cross entropy term: sum_c targets_c * log(probs_c + eps)
        log_probs = torch.log(probs.clamp(min=1e-7, max=1.0))
        pt = (probs * targets_one_hot).sum(dim=1)  # (B, H, W)

        # Focal modulating factor: (1 - pt)^gamma
        focal_factor = (1.0 - pt).clamp(min=0.0).pow(self.focal_gamma)

        # Alpha weighting per class
        alpha_t = (self.alpha.view(1, num_cls, 1, 1) * targets_one_hot).sum(dim=1)

        pixel_focal_loss = -alpha_t * focal_factor * (targets_one_hot * log_probs).sum(dim=1)
        focal_loss = (pixel_focal_loss * valid_mask.float()).sum() / num_valid

        # ----------------------------------------------------------------------
        # 2. Multi-Class Soft Dice Loss & IoU Loss
        # ----------------------------------------------------------------------
        dice_per_class: List[torch.Tensor] = []
        iou_per_class: List[torch.Tensor] = []

        # Mask probabilities and targets by valid pixels
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(probs).float()
        masked_probs = probs * valid_mask_expanded
        masked_targets = targets_one_hot * valid_mask_expanded

        for c in range(self.num_classes):
            p_c = masked_probs[:, c]
            y_c = masked_targets[:, c]

            intersection = (p_c * y_c).sum(dim=(1, 2))
            cardinality = (p_c.pow(2) + y_c.pow(2)).sum(dim=(1, 2))
            union = (p_c + y_c - p_c * y_c).sum(dim=(1, 2))

            dice_c = (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)
            iou_c = (intersection + self.epsilon) / (union + self.epsilon)

            dice_per_class.append(dice_c)
            iou_per_class.append(iou_c)

        # Macro average across foreground classes (0..3)
        fg_dice = torch.stack(dice_per_class[:4], dim=1).mean()
        dice_loss = 1.0 - fg_dice

        fg_iou = torch.stack(iou_per_class[:4], dim=1).mean()
        iou_loss = 1.0 - fg_iou

        # ----------------------------------------------------------------------
        # 3. IoU Prediction Head MSE Loss
        # ----------------------------------------------------------------------
        head_loss = torch.tensor(0.0, device=device)
        if iou_predictions is not None:
            # Actual computed IoU per class detached (stop-gradient)
            actual_ious = torch.stack(iou_per_class, dim=1).detach()
            head_loss = F.mse_loss(iou_predictions, actual_ious)

        # ----------------------------------------------------------------------
        # Total Weighted Composite Loss
        # ----------------------------------------------------------------------
        total_loss = (
            self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
            + self.iou_weight * iou_loss
            + self.head_weight * head_loss
        )

        return {
            "total_loss": total_loss,
            "focal_loss": focal_loss,
            "dice_loss": dice_loss,
            "iou_loss": iou_loss,
            "head_loss": head_loss,
        }


# ==============================================================================
# 2. Histopathology Synthetic / Real Dataset Loader
# ==============================================================================

class SyntheticHistologyDataset(Dataset):
    """Deterministic synthetic histology dataset for dry-run and local smoke testing."""

    def __init__(self, num_samples: int = 16, image_size: int = 224) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42 + idx)
        # Synthetic RGB image
        img = torch.rand(3, self.image_size, self.image_size)
        # Synthetic discrete label mask: classes 0..4, occasionally 255
        mask = torch.randint(0, 5, (self.image_size, self.image_size), dtype=torch.long)
        # Add random ignore pixels along borders
        mask[:4, :] = 255
        mask[-4:, :] = 255
        # Multi-hot label vector for 4 foreground classes
        weak_label = (torch.rand(4) > 0.4).float()
        return img, mask, weak_label


# ==============================================================================
# 3. Training & Validation Epoch Loops
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    prompt_generator: CAMPromptGenerator,
    dataloader: DataLoader,
    criterion: CompositeWSSSLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    accum_steps: int = 2,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Runs a single training epoch with gradient accumulation and AMP."""
    model.train()
    total_loss_accum = 0.0
    focal_loss_accum = 0.0
    dice_loss_accum = 0.0
    iou_loss_accum = 0.0

    optimizer.zero_grad()
    step_count = 0

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break

        images, masks, weak_labels = batch
        images = images.to(device)
        masks = masks.to(device)
        weak_labels = weak_labels.to(device)

        # Generate prompts from pseudo CAMs or synthetic CAM activations
        # When CAMs are not precomputed, construct surrogate CAMs from weak labels
        b, _, h, w = images.shape
        surrogate_cams = torch.zeros(b, 4, h, w, device=device)
        for c in range(4):
            active_b = weak_labels[:, c].view(b, 1, 1)
            surrogate_cams[:, c] = (images.mean(dim=1) * active_b).clamp(0.0, 1.0)

        prompt_dict = prompt_generator(
            cams=surrogate_cams,
            cls_scores=weak_labels,
        )

        # Concatenate positive/negative prompts from primary active class
        primary_class = 0
        coords = prompt_dict[primary_class]["point_coords"]
        labels = prompt_dict[primary_class]["point_labels"]

        use_cuda_amp = (device.type == "cuda") and (scaler is not None)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda_amp):
            outputs = model(
                images=images,
                point_coords=coords,
                point_labels=labels,
            )
            logits = outputs["logits"]
            iou_pred = outputs["iou_predictions"]

            loss_dict = criterion(
                pred_logits=logits,
                target_labels=masks,
                iou_predictions=iou_pred,
            )
            loss = loss_dict["total_loss"] / accum_steps

        if scaler is not None and use_cuda_amp:
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        total_loss_accum += loss_dict["total_loss"].item()
        focal_loss_accum += loss_dict["focal_loss"].item()
        dice_loss_accum += loss_dict["dice_loss"].item()
        iou_loss_accum += loss_dict["iou_loss"].item()
        step_count += 1

    denom = max(1, step_count)
    return {
        "loss": total_loss_accum / denom,
        "focal": focal_loss_accum / denom,
        "dice": dice_loss_accum / denom,
        "iou": iou_loss_accum / denom,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluates model performance across 5 histological classes."""
    model.eval()
    num_classes = 5
    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break

        images, masks, _ = batch
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images=images)
        preds = outputs["logits"].argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()

        for c in range(num_classes):
            valid = (targets != 255)
            pred_c = (preds == c) & valid
            target_c = (targets == c) & valid

            intersection[c] += np.logical_and(pred_c, target_c).sum()
            union[c] += np.logical_or(pred_c, target_c).sum()

    per_class_iou = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        per_class_iou[c] = intersection[c] / max(1.0, union[c])

    # mIoU over 4 foreground classes (TUM, STR, LYM, NEC)
    miou_fg = float(per_class_iou[:4].mean())
    miou_all = float(per_class_iou.mean())

    return {
        "mIoU_fg": miou_fg,
        "mIoU_all": miou_all,
        "iou_TUM": float(per_class_iou[0]),
        "iou_STR": float(per_class_iou[1]),
        "iou_LYM": float(per_class_iou[2]),
        "iou_NEC": float(per_class_iou[3]),
        "iou_BACK": float(per_class_iou[4]),
    }


# ==============================================================================
# 4. Checkpoint Management
# ==============================================================================

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
    filename: str = "medsam_wsss_adapter_last.pth",
    best_filename: str = "medsam_wsss_adapter_best.pth",
) -> str:
    """Saves checkpoint dictionary to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_path)
        return best_path
    return last_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Loads checkpoint state dictionary and restores optimizer and scheduler."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# ==============================================================================
# 5. CLI & Execution Pipeline
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train proposed MedSAM WSSS Adapter with Parameter-Efficient Fine-Tuning."
    )
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="Path to BCSS-WSSS dataset.")
    parser.add_argument("--checkpoint-sam", type=str, default=None, help="Path to official SAM / MedSAM weights.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per worker.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for adapter parameters.")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1-step dry run for validation.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Initializing MedSAM WSSS Adapter on {device}...")

    # Instantiate Model
    model = MedSAM_WSSS_Adapter(
        checkpoint_path=args.checkpoint_sam,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        adapter_mid_dim=64,
        num_classes=5,
    ).to(device)

    param_summary = model.count_parameters()
    print(f"[INFO] Total Parameters: {param_summary['total']:,}")
    print(f"[INFO] Trainable Adapter Parameters: {param_summary['trainable']:,} ({param_summary['trainable']/param_summary['total']*100:.2f}%)")
    print(f"[INFO] Frozen Parameters: {param_summary['frozen']:,}")

    # Prompt Generator
    prompt_generator = CAMPromptGenerator(num_classes=4, max_pos_points=3, max_neg_points=3)

    # Loss, Optimizer, Scheduler
    criterion = CompositeWSSSLoss(
        num_classes=5,
        focal_weight=1.0,
        dice_weight=1.0,
        iou_weight=0.5,
        head_weight=0.2,
    ).to(device)

    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Resume capability
    start_epoch = 0
    best_miou = 0.0
    if args.resume is not None:
        print(f"[INFO] Resuming training from {args.resume}...")
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_mIoU", 0.0)
        print(f"[INFO] Resumed at epoch {start_epoch} with best mIoU = {best_miou:.4f}")

    # Dataloaders: use real data if exists, else synthetic fallback
    if os.path.exists(args.data_dir) and not args.dry_run:
        from src.datasets.bcss_dataset import BCSSDataset
        train_ds = BCSSDataset(root_dir=args.data_dir, split="train")
        val_ds = BCSSDataset(root_dir=args.data_dir, split="valid")
    else:
        num_samples = 4 if args.dry_run else 32
        print(f"[INFO] Using SyntheticHistologyDataset with {num_samples} samples.")
        train_ds = SyntheticHistologyDataset(num_samples=num_samples, image_size=224)
        val_ds = SyntheticHistologyDataset(num_samples=num_samples // 2, image_size=224)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    total_epochs = 1 if args.dry_run else args.epochs
    max_steps = 2 if args.dry_run else None

    print(f"[INFO] Starting training for {total_epochs} epoch(s) (dry_run={args.dry_run})...")

    for epoch in range(start_epoch, start_epoch + total_epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            prompt_generator=prompt_generator,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accum_steps=args.accum_steps,
            max_steps=max_steps,
        )

        val_stats = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            max_steps=max_steps,
        )

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d} [{elapsed:.1f}s]: "
            f"Train Loss={train_stats['loss']:.4f} (Focal={train_stats['focal']:.4f}, Dice={train_stats['dice']:.4f}) | "
            f"Val mIoU_fg={val_stats['mIoU_fg']:.4f} (TUM={val_stats['iou_TUM']:.4f}, STR={val_stats['iou_STR']:.4f}, "
            f"LYM={val_stats['iou_LYM']:.4f}, NEC={val_stats['iou_NEC']:.4f})"
        )

        is_best = val_stats["mIoU_fg"] > best_miou
        if is_best:
            best_miou = val_stats["mIoU_fg"]

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_mIoU": best_miou,
                "val_stats": val_stats,
            },
            is_best=is_best,
            checkpoint_dir=args.output_dir,
        )

    print(f"[INFO] Completed successfully. Best mIoU: {best_miou:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
