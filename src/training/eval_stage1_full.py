"""
Full Validation Benchmark Engine for Stage-1 Histopathology WSSS Checkpoints.

Provides apples-to-apples evaluation on BCSS-WSSS validation cohort (3,418 patches)
across 4 inference modes (strictly excluding outside-ROI gt == 4):
1. Raw 1x CAM (both per-channel min-max and global min-max normalization)
2. SG-DAP 1x (StainGuidedAffinityRefiner, return_ignore_mask=False, active_labels)
3. MS-TTA (multi-scale + HFlip CAM aggregation, per-channel and global)
4. MS-TTA + SG-DAP (multi-scale CAM refined by StainGuidedAffinityRefiner)

Also audits checkpoint integrity: SHA256, strict=True load into WSSSTissueStage1NetCAM,
fc8_std, per-class fc8 norms, bn7_var, and threshold-invariant classification metrics.

Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.sg_dap import StainGuidedAffinityRefiner
from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.training.dataset_loader import (
    FOREGROUND_CLASSES,
    NUM_FOREGROUND,
    BCSSStage1ValDataset,
    load_bcss_patient_splits,
)
from src.training.train_stage1 import compute_multilabel_validation_metrics
from src.training.train_stage2 import infer_stage1_norm_mode, prepare_stage1_input


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute SHA256 hex digest of a file in streaming 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_stage1_strict(
    model: WSSSTissueStage1NetCAM,
    checkpoint_path: Union[str, Path],
    requested_norm: str = "auto",
) -> Tuple[str, str, Dict[str, Any]]:
    """Load Stage-1 checkpoint with strict=True verification into WSSSTissueStage1NetCAM."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found at: {ckpt_p}")

    sha256 = compute_file_sha256(ckpt_p)
    try:
        raw_obj = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    except Exception:
        raw_obj = torch.load(ckpt_p, map_location="cpu", weights_only=False)  # nosec B614

    norm_mode = infer_stage1_norm_mode(raw_obj, checkpoint_path=ckpt_p, requested_mode=requested_norm)
    if (
        str(requested_norm or "auto").strip().lower() == "auto"
        and ckpt_p.name == "stage1_checkpoint_trained_on_bcss.pth"
        and not (isinstance(raw_obj, dict) and "input_norm" in raw_obj)
    ):
        norm_mode = "zero_one"

    raw_sd = raw_obj.get("state_dict", raw_obj.get("model", raw_obj)) if isinstance(raw_obj, dict) else raw_obj
    if not isinstance(raw_sd, dict):
        raise RuntimeError(f"Invalid checkpoint payload in {ckpt_p}: expected dict, got {type(raw_sd).__name__}.")

    cleaned_sd: Dict[str, torch.Tensor] = {}
    non_finite_keys: List[str] = []
    for k, v in raw_sd.items():
        clean_k = str(k)
        while clean_k.startswith("module.") or clean_k.startswith("model."):
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            elif clean_k.startswith("model."):
                clean_k = clean_k[6:]
        if isinstance(v, torch.Tensor) and v.is_floating_point() and not torch.isfinite(v).all():
            non_finite_keys.append(clean_k)
        cleaned_sd[clean_k] = v

    if non_finite_keys:
        raise ValueError(
            f"Non-finite tensors ({len(non_finite_keys)}) detected in {ckpt_p}: {non_finite_keys[:5]}"
        )

    model.load_state_dict(cleaned_sd, strict=True)

    meta: Dict[str, Any] = {
        "strict_load_ok": True,
        "non_finite_tensors": 0,
        "num_state_dict_keys": len(cleaned_sd),
        "ckpt_epoch": int(raw_obj.get("epoch", -1)) if isinstance(raw_obj, dict) and "epoch" in raw_obj else None,
        "ckpt_exp_preset": raw_obj.get("exp_preset") if isinstance(raw_obj, dict) else None,
        "ckpt_selected_weights": raw_obj.get("selected_weights") if isinstance(raw_obj, dict) else None,
    }
    return norm_mode, sha256, meta


class _ConfusionAccumulator:
    """Streaming 4-class IoU and Dice accumulator strictly excluding gt == 4."""

    def __init__(self, num_classes: int = NUM_FOREGROUND) -> None:
        self.num_classes = num_classes
        self.inter = np.zeros(num_classes, dtype=np.float64)
        self.union = np.zeros(num_classes, dtype=np.float64)
        self.dice_num = np.zeros(num_classes, dtype=np.float64)
        self.dice_den = np.zeros(num_classes, dtype=np.float64)

    def update(self, pred_np: np.ndarray, gt_np: np.ndarray, valid_roi: np.ndarray) -> None:
        for c in range(self.num_classes):
            pc = (pred_np == c) & valid_roi
            gc = (gt_np == c) & valid_roi
            tp = float(np.logical_and(pc, gc).sum())
            self.inter[c] += tp
            self.union[c] += float(np.logical_or(pc, gc).sum())
            self.dice_num[c] += 2.0 * tp
            self.dice_den[c] += float(pc.sum() + gc.sum())

    def summary(self) -> Dict[str, Any]:
        ious = [
            float(self.inter[c] / self.union[c]) if self.union[c] > 0 else 0.0
            for c in range(self.num_classes)
        ]
        dices = [
            float(self.dice_num[c] / self.dice_den[c]) if self.dice_den[c] > 0 else 0.0
            for c in range(self.num_classes)
        ]
        active_ious = [ious[c] for c in range(self.num_classes) if self.union[c] > 0]
        active_dices = [dices[c] for c in range(self.num_classes) if self.dice_den[c] > 0]
        per_class_iou = {FOREGROUND_CLASSES[c].lower(): ious[c] for c in range(self.num_classes)}
        per_class_dice = {FOREGROUND_CLASSES[c].lower(): dices[c] for c in range(self.num_classes)}
        return {
            "mIoU": float(np.mean(active_ious)) if active_ious else 0.0,
            "mDice": float(np.mean(active_dices)) if active_dices else 0.0,
            "per_class_iou": per_class_iou,
            "per_class_dice": per_class_dice,
        }


def _cam_to_discrete_pred(
    cam_up: torch.Tensor,
    targets: torch.Tensor,
    norm_kind: str = "per_channel",
) -> Tuple[np.ndarray, torch.Tensor]:
    """Gate CAMs by active labels, normalize (per_channel or global), and argmax safely."""
    b_sz = cam_up.size(0)
    gate = (targets.float() > 0.5).float().view(b_sz, NUM_FOREGROUND, 1, 1)
    cam_gated = torch.clamp(cam_up.float(), min=0.0) * gate
    if norm_kind == "global":
        gmin = cam_gated.view(b_sz, -1).min(dim=-1)[0].view(b_sz, 1, 1, 1)
        gmax = cam_gated.view(b_sz, -1).max(dim=-1)[0].view(b_sz, 1, 1, 1)
        cam_norm = ((cam_gated - gmin) / (gmax - gmin).clamp(min=1e-7)) * gate
    else:
        cmin = cam_gated.view(b_sz, NUM_FOREGROUND, -1).min(dim=-1)[0].view(b_sz, NUM_FOREGROUND, 1, 1)
        cmax = cam_gated.view(b_sz, NUM_FOREGROUND, -1).max(dim=-1)[0].view(b_sz, NUM_FOREGROUND, 1, 1)
        cam_norm = ((cam_gated - cmin) / (cmax - cmin).clamp(min=1e-7)) * gate
    cam_masked = torch.where(gate > 0.5, cam_norm, torch.full_like(cam_norm, -1.0))
    pred = torch.argmax(cam_masked, dim=1).cpu().numpy()
    return pred, cam_norm


def _extract_ms_tta_cam(
    model: WSSSTissueStage1NetCAM,
    images: torch.Tensor,
    target_hw: Tuple[int, int],
    scales: Sequence[float] = (0.75, 1.0, 1.25),
    use_hflip: bool = True,
) -> torch.Tensor:
    """Compute Multi-Scale + HFlip Test-Time Augmentation raw CAM upsampled to target_hw."""
    fc8_w = model.fc8.weight.float()
    b, _, h_in, w_in = images.shape
    cam_sum = torch.zeros((b, NUM_FOREGROUND, target_hw[0], target_hw[1]), device=images.device, dtype=torch.float32)
    count = 0

    flip_opts = (False, True) if use_hflip else (False,)
    for scale in scales:
        s = float(scale)
        if abs(s - 1.0) > 1e-4:
            nh = max(8, int(round(h_in * s)))
            nw = max(8, int(round(w_in * s)))
            x_s = F.interpolate(images, size=(nh, nw), mode="bilinear", align_corners=False)
        else:
            x_s = images

        for do_flip in flip_opts:
            x_in = torch.flip(x_s, dims=[-1]) if do_flip else x_s
            conv6 = model.forward_backbone(x_in).float()
            cam = F.relu(F.conv2d(conv6, fc8_w))
            if do_flip:
                cam = torch.flip(cam, dims=[-1])
            cam_up = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
            cam_sum = cam_sum + cam_up
            count += 1

    return cam_sum / float(max(count, 1))


@torch.no_grad()
def evaluate_stage1_checkpoint(
    checkpoint_path: Union[str, Path],
    data_dir: Union[str, Path] = "data/BCSS-WSSS",
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    device_str: Optional[str] = None,
    scales: Sequence[float] = (0.75, 1.0, 1.25),
    run_sg_dap: bool = True,
    num_workers: int = 0,
    input_norm: str = "auto",
    strict_patient_filter: bool = True,
) -> Dict[str, Any]:
    """Run full Stage-1 validation benchmark on a strict=True loaded checkpoint."""
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
    norm_mode, sha256, ckpt_meta = load_stage1_strict(
        model, checkpoint_path=checkpoint_path, requested_norm=input_norm
    )
    model = model.to(device).eval()

    data_path = Path(data_dir)
    val_img_dir = data_path / "val" / "img"
    val_mask_dir = data_path / "val" / "mask"
    if not val_img_dir.is_dir():
        val_img_dir = data_path / "BCSS-WSSS" / "val" / "img"
        val_mask_dir = data_path / "BCSS-WSSS" / "val" / "mask"

    splits = load_bcss_patient_splits()
    val_patients = splits.get("val", [])
    val_ds = BCSSStage1ValDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir if val_mask_dir.is_dir() else None,
        patient_ids=val_patients if val_patients else None,
        limit=max_samples,
        input_norm=norm_mode,
        strict_patient_filter=strict_patient_filter,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    sg_dap = StainGuidedAffinityRefiner(kernel_size=5, num_iters=10).to(device).eval() if run_sg_dap else None

    acc_raw_pc = _ConfusionAccumulator()
    acc_raw_gl = _ConfusionAccumulator()
    acc_sgdap = _ConfusionAccumulator()
    acc_mstta_pc = _ConfusionAccumulator()
    acc_mstta_gl = _ConfusionAccumulator()
    acc_mstta_sgdap = _ConfusionAccumulator()

    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    total_patches = 0

    fc8_w = model.fc8.weight.float()

    for _, images, targets, gt_masks in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        gt_np = gt_masks.cpu().numpy()
        valid_roi = (gt_np >= 0) & (gt_np < NUM_FOREGROUND)
        target_hw = (gt_np.shape[-2], gt_np.shape[-1])
        total_patches += images.size(0)

        conv6_f = model.forward_backbone(images).float()
        pooled = model.pool(conv6_f)
        logits = F.conv2d(pooled, fc8_w).view(pooled.size(0), -1)
        probs = torch.sigmoid(logits)
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        if not np.any(valid_roi):
            continue

        # 1. Raw 1x CAM (per-channel & global)
        cams_1x = F.relu(F.conv2d(conv6_f, fc8_w))
        cam_1x_up = F.interpolate(cams_1x, size=target_hw, mode="bilinear", align_corners=False)
        pred_1x_pc, cam_1x_norm_pc = _cam_to_discrete_pred(cam_1x_up, targets, norm_kind="per_channel")
        pred_1x_gl, _ = _cam_to_discrete_pred(cam_1x_up, targets, norm_kind="global")
        acc_raw_pc.update(pred_1x_pc, gt_np, valid_roi)
        acc_raw_gl.update(pred_1x_gl, gt_np, valid_roi)

        # 2. MS-TTA CAM (per-channel & global)
        cam_ms_up = _extract_ms_tta_cam(model, images, target_hw=target_hw, scales=scales, use_hflip=True)
        pred_ms_pc, cam_ms_norm_pc = _cam_to_discrete_pred(cam_ms_up, targets, norm_kind="per_channel")
        pred_ms_gl, _ = _cam_to_discrete_pred(cam_ms_up, targets, norm_kind="global")
        acc_mstta_pc.update(pred_ms_pc, gt_np, valid_roi)
        acc_mstta_gl.update(pred_ms_gl, gt_np, valid_roi)

        # 3 & 4. SG-DAP 1x and MS-TTA+SG-DAP
        if sg_dap is not None:
            rgb_01 = prepare_stage1_input(
                images,
                stage1_norm_mode="zero_one",
                images_are_imagenet_norm=(norm_mode == "imagenet"),
            )
            _, mask_sg_1x = sg_dap(
                rgb_01,
                conv6_f,
                cam_1x_norm_pc,
                return_ignore_mask=False,
                active_labels=targets,
            )
            acc_sgdap.update(mask_sg_1x.cpu().numpy(), gt_np, valid_roi)

            _, mask_sg_ms = sg_dap(
                rgb_01,
                conv6_f,
                cam_ms_norm_pc,
                return_ignore_mask=False,
                active_labels=targets,
            )
            acc_mstta_sgdap.update(mask_sg_ms.cpu().numpy(), gt_np, valid_roi)

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, NUM_FOREGROUND))
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, NUM_FOREGROUND))
    y_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, NUM_FOREGROUND))
    cls_metrics = compute_multilabel_validation_metrics(y_true, y_prob)

    fc8_flat = model.fc8.weight.detach().float().view(NUM_FOREGROUND, -1)
    fc8_norms = torch.linalg.norm(fc8_flat, dim=1)
    bn7_var = model.bn7.running_var.detach().float()

    report: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "sha256": sha256,
        "input_norm": norm_mode,
        "num_patches": total_patches,
        "scales": [float(s) for s in scales],
        **ckpt_meta,
        "fc8_std": float(fc8_flat.std().item()),
        "fc8_norms": {
            FOREGROUND_CLASSES[c].lower(): float(fc8_norms[c].item()) for c in range(NUM_FOREGROUND)
        },
        "bn7_var_mean": float(bn7_var.mean().item()),
        "bn7_var_min": float(bn7_var.min().item()),
        "bn7_var_max": float(bn7_var.max().item()),
        "classification": {
            **cls_metrics,
            "val_logits_mean": float(np.mean(y_logits)) if y_logits.size > 0 else 0.0,
        },
        "modes": {
            "raw_1x_per_channel": acc_raw_pc.summary(),
            "raw_1x_global": acc_raw_gl.summary(),
            "ms_tta_per_channel": acc_mstta_pc.summary(),
            "ms_tta_global": acc_mstta_gl.summary(),
        },
    }
    if sg_dap is not None:
        report["modes"]["sg_dap_1x"] = acc_sgdap.summary()
        report["modes"]["ms_tta_sg_dap"] = acc_mstta_sgdap.summary()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full Stage-1 Validation Benchmark (BCSS-WSSS)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage-1 checkpoint .pth path")
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="BCSS-WSSS root directory")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON report")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional patch cap for quick checks")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], help="MS-TTA scales")
    parser.add_argument("--no-sg-dap", action="store_true", help="Skip SG-DAP refinement modes")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--input-norm",
        type=str,
        default="auto",
        choices=["auto", "zero_one", "imagenet"],
        help="Input normalization mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_stage1_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device_str=args.device,
        scales=args.scales,
        run_sg_dap=not args.no_sg_dap,
        num_workers=args.num_workers,
        input_norm=args.input_norm,
    )
    print(f"Checkpoint: {report['checkpoint_path']} | SHA256: {report['sha256'][:16]}... | strict={report['strict_load_ok']}")
    print(f"Patches={report['num_patches']} | norm={report['input_norm']} | fc8_std={report['fc8_std']:.5f} | bn7_var={report['bn7_var_mean']:.5f}")
    cls = report["classification"]
    print(f"Cls: mAP={cls['val_mAP']:.4f} | AUROC={cls['val_auroc']:.4f} | BestF1={cls['val_best_f1']:.4f} | F1@0.5={cls['val_macro_f1']:.4f}")
    for mode_name, mode_stats in report["modes"].items():
        ious = mode_stats["per_class_iou"]
        print(
            f"  {mode_name:20s}: mIoU={mode_stats['mIoU']:.4f} | mDice={mode_stats['mDice']:.4f} | "
            f"TUM={ious['tum']:.4f} STR={ious['str']:.4f} LYM={ious['lym']:.4f} NEC={ious['nec']:.4f}"
        )
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_p}")


if __name__ == "__main__":
    main()
