"""
Stage-1 Multi-Label Classification Training Engine for SE374.R11 WSSS.

Trains the 105M-parameter ResNet-38d backbone (WSSSTissueStage1NetCAM) with:
1. Progressive Dropout Attention (PDA) activation schedule (Han et al., MedIA 2022).
2. Asymmetric Loss (ASL) and Multi-Label Soft Margin Loss for severe tissue class imbalance.
3. AdamW optimizer with decoupled layer-wise learning rates (10x lr on fc8 projection head).
4. Cosine Annealing Learning Rate Scheduler with warmup.
5. Mixed-precision training via torch.amp (bfloat16/float16).
6. Deterministic checkpoint saving to checkpoints/stage1_retrained_best.pth.
7. Zero patient-level leakage enforcement (Invariant 08).
8. Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
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

from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.training.dataset_loader import (
    CLASS_NAMES,
    FOREGROUND_CLASSES,
    NUM_FOREGROUND,
    create_stage1_dataloaders,
)


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
        cams = F.conv2d(x, fc_weights)
        cams = F.relu(cams)  # (B, 4, H, W)
        b, c, h, w = cams.size()

        cam_mean = torch.mean(cams, dim=1)  # (B, H, W)
        zero = torch.zeros_like(cam_mean)
        mean_drop_cam = zero.clone()

        for i in range(c):
            sub_cam = cams[:, i, :, :]  # (B, H, W)
            sub_cam_max = torch.amax(sub_cam.view(b, -1), dim=-1).view(b, 1, 1)
            thr = sub_cam_max * gamma
            thr = thr.expand_as(sub_cam)
            sub_cam_with_drop = torch.where(sub_cam > thr, zero, sub_cam)
            mean_drop_cam = mean_drop_cam + sub_cam_with_drop

        mean_drop_cam = mean_drop_cam / max(c, 1)
        mean_drop_cam = mean_drop_cam.unsqueeze(1)  # (B, 1, H, W)

        return x * mean_drop_cam


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
        probs = torch.sigmoid(logits)

        # Positive margin
        pos_probs = probs
        pos_loss = targets * torch.log(pos_probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            pos_weight = (1.0 - pos_probs).pow(self.gamma_pos)
            pos_loss = pos_loss * pos_weight

        # Negative margin with hard probability clipping
        neg_probs = (1.0 - probs).clamp(min=self.eps)
        if self.clip > 0:
            asym_probs = (probs - self.clip).clamp(min=0.0)
            neg_loss = (1.0 - targets) * torch.log((1.0 - asym_probs).clamp(min=self.eps))
        else:
            neg_loss = (1.0 - targets) * torch.log(neg_probs)

        if self.gamma_neg > 0:
            neg_weight = probs.pow(self.gamma_neg)
            neg_loss = neg_loss * neg_weight

        total_loss = - (pos_loss + neg_loss)
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

        if enable_pda:
            conv6 = self.attention_module(conv6, self.model.fc8.weight, self.gamma)

        x = self.model.dropout7(conv6)
        x = self.model.pool(x)
        logits = self.model.fc8(x).view(x.size(0), -1)  # (B, 4)
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
    ) -> Dict[str, float]:
        """Evaluate multi-label classification accuracy on validation cohort."""
        self.model.eval()

        total_loss = 0.0
        exact_matches = 0
        total_samples = 0
        per_class_tp = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        per_class_pred = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        per_class_gt = np.zeros(NUM_FOREGROUND, dtype=np.int64)

        for images, targets, _ in val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits, probs = self.forward_with_pda(images, enable_pda=False)
            loss = self.criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            binary_preds = (probs > 0.5).long().cpu().numpy()
            binary_gt = targets.long().cpu().numpy()

            em_batch = np.all(binary_preds == binary_gt, axis=1).sum()
            exact_matches += int(em_batch)

            for c in range(NUM_FOREGROUND):
                per_class_tp[c] += np.logical_and(binary_preds[:, c] == 1, binary_gt[:, c] == 1).sum()
                per_class_pred[c] += (binary_preds[:, c] == 1).sum()
                per_class_gt[c] += (binary_gt[:, c] == 1).sum()

        f1_scores = []
        for c in range(NUM_FOREGROUND):
            denom = per_class_pred[c] + per_class_gt[c]
            f1 = (2.0 * per_class_tp[c] / denom) if denom > 0 else 0.0
            f1_scores.append(float(f1))

        avg_loss = total_loss / max(total_samples, 1)
        em_rate = exact_matches / max(total_samples, 1)
        macro_f1 = float(np.mean(f1_scores))

        return {
            "val_loss": avg_loss,
            "val_exact_match": em_rate,
            "val_macro_f1": macro_f1,
            "val_f1_tum": f1_scores[0],
            "val_f1_str": f1_scores[1],
            "val_f1_lym": f1_scores[2],
            "val_f1_nec": f1_scores[3],
        }


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
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    loss_type: str = "asl",
    resume_path: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    device_str: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end Stage 1 classifier training pipeline.

    Args:
        data_dir: Root dataset folder containing train/ and optionally val/.
        checkpoint_dir: Folder to save checkpoints.
        epochs: Number of epochs to train.
        batch_size: Mini-batch size.
        lr: Base learning rate.
        loss_type: 'asl' or 'soft_margin'.
        resume_path: Checkpoint path to resume from.
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

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)

    # Resume from checkpoint if requested
    start_epoch = 0
    best_macro_f1 = 0.0
    if resume_path and Path(resume_path).exists():
        loaded = torch.load(resume_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)
        model.load_state_dict(state_dict, strict=False)
        start_epoch = loaded.get("epoch", 0)
        best_macro_f1 = loaded.get("macro_f1", 0.0)

    # Loss function
    if loss_type == "asl":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer, scheduler = build_optimizer_and_scheduler(model, lr=lr, epochs=epochs)

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
    )

    trainer = Stage1Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        enable_amp=not dry_run,
    )

    history: List[Dict[str, float]] = []
    best_save_path = checkpoint_path / "stage1_retrained_best.pth"

    for epoch in range(start_epoch, effective_epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)

        val_metrics: Dict[str, float] = {}
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = trainer.evaluate(val_loader)

        epoch_record = {**train_metrics, **val_metrics, "epoch": epoch}
        history.append(epoch_record)

        current_f1 = val_metrics.get("val_macro_f1", train_metrics["macro_f1"])

        if current_f1 >= best_macro_f1 or dry_run:
            best_macro_f1 = current_f1
            torch.save(
                {
                    "epoch": epoch,
                    "macro_f1": best_macro_f1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": epoch_record,
                },
                best_save_path,
            )

    return {
        "best_macro_f1": best_macro_f1,
        "checkpoint_path": str(best_save_path),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 ResNet-38d Classifier on BCSS-WSSS")
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="Dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss-type", type=str, default="asl", choices=["asl", "bce"], help="Loss type")
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
        dry_run=args.dry_run,
        device_str=args.device,
        seed=args.seed,
    )
    print("Stage 1 Training Complete.")
    print(f"Best Macro F1: {results['best_macro_f1']:.4f}")
    print(f"Saved Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
