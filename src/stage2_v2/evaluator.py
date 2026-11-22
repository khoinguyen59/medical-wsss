"""
Stage-2 v2 Unified 4-Class Evaluator, VAL-Only CGM Calibrator & Step-0 Anchor Engine.

Enforces strict 4-class BCSS-WSSS evaluation (gt >= 0 & gt < 4, strictly excluding
outside-ROI gt == 4 and ignore gt == 255) across:
- raw: 1x segmentation logits[:, :4]
- cgm_s1: gated by Stage-1 classifier probabilities (with top-1 fallback)
- cgm_self: gated by Stage-2 Dual-Head cls_logits (with top-1 fallback)
- cgm_dual: gated by 0.5 * s1_probs + 0.5 * self_probs
- ms_tta: 6-view MS-TTA ([0.75, 1.0, 1.25] + HFlip)
- ms_tta_cgm: 6-view MS-TTA with CGM gating

Also provides:
- calibrate_cgm_thresholds_val_only: Coordinate-descent grid search over
  (t_TUM, t_STR, t_LYM, t_NEC) in [0.05, ..., 0.55] strictly on VAL.
- run_step0_calibration_and_author_anchor: Official Author Stage-2 anchor benchmark
  + Champion bn7 soft-target recipe sweep (raw_l1, softmax_t T=0.15/0.25/0.40, sg_dap_soft).

Pure ASCII hyphen '-' only (Invariant 12).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.sg_dap import StainGuidedAffinityRefiner
from src.models.wsss_inference_methods import (
    WSSSTissueStage1NetCAM,
    build_and_load_deeplabv3plus,
)
from src.stage2_v2.memmap_dataset import _cams_to_soft_probs, _extract_bn7_6view_global_cam
from src.training.dataset_loader import FOREGROUND_CLASSES, NUM_FOREGROUND
from src.training.train_stage2 import prepare_stage1_input

_CLS_LOWER: Tuple[str, ...] = tuple(c.lower() for c in FOREGROUND_CLASSES)
CGM_GRID: Tuple[float, ...] = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
)


class _SegConfusion4:
    """Streaming 4-class foreground confusion accumulator (excludes gt == 4 and gt == 255)."""

    def __init__(self) -> None:
        self.inter = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        self.union = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        self.dnum = np.zeros(NUM_FOREGROUND, dtype=np.float64)
        self.dden = np.zeros(NUM_FOREGROUND, dtype=np.float64)

    def update(self, pred: np.ndarray, gt: np.ndarray, valid_fg: np.ndarray) -> None:
        if not np.any(valid_fg):
            return
        for c in range(NUM_FOREGROUND):
            pc = (pred == c) & valid_fg
            gc = (gt == c) & valid_fg
            tp = float(np.logical_and(pc, gc).sum())
            self.inter[c] += tp
            self.union[c] += float(np.logical_or(pc, gc).sum())
            self.dnum[c] += 2.0 * tp
            self.dden[c] += float(pc.sum() + gc.sum())

    def summary(self) -> Dict[str, Any]:
        ious = [
            float(self.inter[c] / self.union[c]) if self.union[c] > 0 else 0.0
            for c in range(NUM_FOREGROUND)
        ]
        dices = [
            float(self.dnum[c] / self.dden[c]) if self.dden[c] > 0 else 0.0
            for c in range(NUM_FOREGROUND)
        ]
        active_ious = [ious[c] for c in range(NUM_FOREGROUND) if self.union[c] > 0]
        active_dices = [dices[c] for c in range(NUM_FOREGROUND) if self.dden[c] > 0]
        return {
            "mIoU": round(float(np.mean(active_ious)) if active_ious else 0.0, 4),
            "mDice": round(float(np.mean(active_dices)) if active_dices else 0.0, 4),
            "per_class_iou": {_CLS_LOWER[c]: round(ious[c], 4) for c in range(NUM_FOREGROUND)},
            "per_class_dice": {_CLS_LOWER[c]: round(dices[c], 4) for c in range(NUM_FOREGROUND)},
            "class_ious": [round(x, 4) for x in ious],
        }


def _extract_img_gt(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Support 2-tuple (BCSSStage2ValDataset) and 4-tuple (BCSSEvalDataset) batches."""
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            return batch[0], batch[1], None
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
    raise ValueError(f"Unsupported evaluation batch format: {type(batch)}")


def _make_cgm_gate(
    probs: torch.Tensor,
    thresholds: Union[float, Sequence[float]],
) -> torch.Tensor:
    """Build (B, 4, 1, 1) binary active-class mask with top-1 fallback if all below threshold."""
    if isinstance(thresholds, (int, float)):
        thr_t = torch.full((1, NUM_FOREGROUND), float(thresholds), device=probs.device, dtype=probs.dtype)
    else:
        thr_t = torch.tensor(list(thresholds), device=probs.device, dtype=probs.dtype).view(1, NUM_FOREGROUND)
    active = probs > thr_t
    has_any = active.any(dim=1, keepdim=True)
    top1 = F.one_hot(torch.argmax(probs, dim=1), num_classes=NUM_FOREGROUND).bool()
    return torch.where(has_any, active, top1).float().view(-1, NUM_FOREGROUND, 1, 1)


def _apply_logit_cgm(logits_4: torch.Tensor, gate: torch.Tensor) -> np.ndarray:
    gated = torch.where(gate == 1.0, logits_4, torch.full_like(logits_4, -1e4))
    return torch.argmax(gated, dim=1).cpu().numpy()


def _apply_prob_cgm(probs_4: torch.Tensor, gate: torch.Tensor) -> np.ndarray:
    gated = torch.where(gate == 1.0, probs_4, torch.full_like(probs_4, -1.0))
    return torch.argmax(gated, dim=1).cpu().numpy()


def _forward_dual_head(
    model: nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = model(x)
    if isinstance(out, dict):
        return out["seg_logits"][:, :NUM_FOREGROUND], out.get("cls_logits")
    return out[:, :NUM_FOREGROUND], None


def _ms_tta_probs(
    model: nn.Module,
    imgs: torch.Tensor,
    logits_1x: torch.Tensor,
    scales: Sequence[float],
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    b, _, h, w = imgs.shape
    probs_sum = torch.zeros((b, NUM_FOREGROUND, h, w), device=device, dtype=torch.float32)
    count = 0
    use_amp = device.type == "cuda" and amp_dtype is not None
    for sc in scales:
        sc_f = float(sc)
        if abs(sc_f - 1.0) > 1e-4:
            nh = max(16, int(round(h * sc_f)))
            nw = max(16, int(round(w * sc_f)))
            x_s = F.interpolate(imgs, size=(nh, nw), mode="bilinear", align_corners=False)
        else:
            x_s = imgs
        for do_flip in (False, True):
            x_in = torch.flip(x_s, dims=[-1]) if do_flip else x_s
            if abs(sc_f - 1.0) < 1e-4 and not do_flip:
                lg = logits_1x
            else:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    lg, _ = _forward_dual_head(model, x_in)
            pr = torch.softmax(lg.float(), dim=1)
            if do_flip:
                pr = torch.flip(pr, dims=[-1])
            if pr.shape[-2:] != (h, w):
                pr = F.interpolate(pr, size=(h, w), mode="bilinear", align_corners=False)
            probs_sum = probs_sum + pr
            count += 1
    return probs_sum / float(max(count, 1))


@torch.no_grad()
def evaluate_stage2_v2_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stage1_model: Optional[WSSSTissueStage1NetCAM] = None,
    stage1_norm_mode: str = "zero_one",
    stage1_cls_model: Optional[WSSSTissueStage1NetCAM] = None,
    cgm_thresholds: Union[float, Sequence[float]] = 0.10,
    scales: Sequence[float] = (0.75, 1.0, 1.25),
    run_ms_tta: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """Evaluate a Stage-2 v2 dual-head model across raw, CGM, and MS-TTA modes."""
    model.eval()
    s1_expert = stage1_cls_model if stage1_cls_model is not None else stage1_model
    if s1_expert is not None:
        s1_expert.eval()

    acc: Dict[str, _SegConfusion4] = {"raw": _SegConfusion4(), "cgm_self": _SegConfusion4()}
    if s1_expert is not None:
        acc["cgm_s1"] = _SegConfusion4()
        acc["cgm_dual"] = _SegConfusion4()
    if run_ms_tta:
        acc["ms_tta"] = _SegConfusion4()
        acc["ms_tta_cgm"] = _SegConfusion4()

    use_amp = device.type == "cuda" and amp_dtype is not None
    n_patches = 0

    for batch in loader:
        imgs, gts, _ = _extract_img_gt(batch)
        imgs = imgs.to(device, non_blocking=True)
        gt_np = gts.cpu().numpy()
        valid_fg = (gt_np >= 0) & (gt_np < NUM_FOREGROUND)
        n_patches += imgs.size(0)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits_1x, cls_logits = _forward_dual_head(model, imgs)
        logits_1x_f = logits_1x.float()

        pred_raw = torch.argmax(logits_1x_f, dim=1).cpu().numpy()
        acc["raw"].update(pred_raw, gt_np, valid_fg)

        self_probs = (
            torch.sigmoid(cls_logits.float())
            if cls_logits is not None
            else torch.softmax(logits_1x_f, dim=1).amax(dim=(2, 3))
        )
        gate_self = _make_cgm_gate(self_probs, cgm_thresholds)
        acc["cgm_self"].update(_apply_logit_cgm(logits_1x_f, gate_self), gt_np, valid_fg)

        gate_best = gate_self
        if s1_expert is not None:
            s1_inp = prepare_stage1_input(
                imgs, stage1_norm_mode=stage1_norm_mode, images_are_imagenet_norm=True
            )
            _, s1_probs = s1_expert.forward_cam(s1_inp)
            gate_s1 = _make_cgm_gate(s1_probs, cgm_thresholds)
            acc["cgm_s1"].update(_apply_logit_cgm(logits_1x_f, gate_s1), gt_np, valid_fg)
            dual_probs = 0.5 * s1_probs + 0.5 * self_probs
            gate_dual = _make_cgm_gate(dual_probs, cgm_thresholds)
            acc["cgm_dual"].update(_apply_logit_cgm(logits_1x_f, gate_dual), gt_np, valid_fg)
            gate_best = gate_dual

        if run_ms_tta:
            probs_ms = _ms_tta_probs(model, imgs, logits_1x, scales, device, amp_dtype)
            acc["ms_tta"].update(torch.argmax(probs_ms, dim=1).cpu().numpy(), gt_np, valid_fg)
            acc["ms_tta_cgm"].update(_apply_prob_cgm(probs_ms, gate_best), gt_np, valid_fg)

    results: Dict[str, Any] = {"num_patches": n_patches}
    for key, conf in acc.items():
        results[key] = conf.summary()
    return results


@torch.no_grad()
def calibrate_cgm_thresholds_val_only(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    stage1_model: Optional[WSSSTissueStage1NetCAM] = None,
    stage1_norm_mode: str = "zero_one",
    gate_source: str = "dual",
    grid: Sequence[float] = CGM_GRID,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """Coordinate-descent search for per-class CGM thresholds strictly on VAL.

    Precomputes exact per-image confusion vectors across all 15 non-empty 4-class
    bitmasks in a single VAL pass (O(1) RAM), then runs 2 coordinate-descent sweeps
    over [0.05, ..., 0.55]^4 maximizing 4-class VAL mIoU.
    """
    model.eval()
    if stage1_model is not None:
        stage1_model.eval()

    use_amp = device.type == "cuda" and amp_dtype is not None
    all_probs: List[np.ndarray] = []
    all_inter: List[np.ndarray] = []
    all_union: List[np.ndarray] = []

    bitmasks = [m for m in range(1, 16)]
    mask_tensors = [
        torch.tensor([[(m >> c) & 1 for c in range(NUM_FOREGROUND)]], device=device, dtype=torch.bool).view(
            1, NUM_FOREGROUND, 1, 1
        )
        for m in bitmasks
    ]

    for batch in val_loader:
        imgs, gts, _ = _extract_img_gt(batch)
        imgs = imgs.to(device, non_blocking=True)
        gt_np = gts.cpu().numpy()
        valid_fg = (gt_np >= 0) & (gt_np < NUM_FOREGROUND)
        b = imgs.size(0)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits_1x, cls_logits = _forward_dual_head(model, imgs)
        logits_f = logits_1x.float()

        self_probs = (
            torch.sigmoid(cls_logits.float())
            if cls_logits is not None
            else torch.softmax(logits_f, dim=1).amax(dim=(2, 3))
        )
        if gate_source == "s1" and stage1_model is not None:
            s1_inp = prepare_stage1_input(imgs, stage1_norm_mode=stage1_norm_mode, images_are_imagenet_norm=True)
            _, gate_probs = stage1_model.forward_cam(s1_inp)
        elif gate_source == "dual" and stage1_model is not None:
            s1_inp = prepare_stage1_input(imgs, stage1_norm_mode=stage1_norm_mode, images_are_imagenet_norm=True)
            _, s1_probs = stage1_model.forward_cam(s1_inp)
            gate_probs = 0.5 * s1_probs + 0.5 * self_probs
        else:
            gate_probs = self_probs

        all_probs.append(gate_probs.cpu().numpy())

        batch_inter = np.zeros((b, 16, NUM_FOREGROUND), dtype=np.float64)
        batch_union = np.zeros((b, 16, NUM_FOREGROUND), dtype=np.float64)
        for m, m_mask in zip(bitmasks, mask_tensors):
            masked_lg = torch.where(m_mask, logits_f, torch.full_like(logits_f, -1e4))
            pred_m = torch.argmax(masked_lg, dim=1).cpu().numpy()
            for c in range(NUM_FOREGROUND):
                pc = (pred_m == c) & valid_fg
                gc = (gt_np == c) & valid_fg
                batch_inter[:, m, c] = np.logical_and(pc, gc).sum(axis=(1, 2))
                batch_union[:, m, c] = np.logical_or(pc, gc).sum(axis=(1, 2))

        all_inter.append(batch_inter)
        all_union.append(batch_union)

    probs_arr = np.concatenate(all_probs, axis=0)
    inter_arr = np.concatenate(all_inter, axis=0)
    union_arr = np.concatenate(all_union, axis=0)
    n_imgs = probs_arr.shape[0]
    top1_cls = np.argmax(probs_arr, axis=1)
    idx_range = np.arange(n_imgs)

    def _eval_thresholds(thr_vec: List[float]) -> float:
        thr_np = np.asarray(thr_vec, dtype=np.float32).reshape(1, NUM_FOREGROUND)
        active = probs_arr > thr_np
        masks = (
            active[:, 0].astype(np.int32)
            | (active[:, 1].astype(np.int32) << 1)
            | (active[:, 2].astype(np.int32) << 2)
            | (active[:, 3].astype(np.int32) << 3)
        )
        fallback = 1 << top1_cls
        masks = np.where(masks > 0, masks, fallback)
        tot_inter = inter_arr[idx_range, masks].sum(axis=0)
        tot_union = union_arr[idx_range, masks].sum(axis=0)
        valid_c = tot_union > 0
        if not np.any(valid_c):
            return 0.0
        return float(np.mean(tot_inter[valid_c] / tot_union[valid_c]))

    best_thr = [0.10, 0.10, 0.10, 0.10]
    best_miou = _eval_thresholds(best_thr)
    for _sweep in range(2):
        for c in range(NUM_FOREGROUND):
            for cand in grid:
                trial = list(best_thr)
                trial[c] = float(cand)
                score = _eval_thresholds(trial)
                if score > best_miou + 1e-6:
                    best_miou = score
                    best_thr = trial

    return {
        "gate_source": gate_source if stage1_model is not None else "self",
        "thresholds": [round(float(t), 4) for t in best_thr],
        "per_class_thresholds": {_CLS_LOWER[c]: round(float(best_thr[c]), 4) for c in range(NUM_FOREGROUND)},
        "val_miou": round(best_miou, 4),
    }


def _compute_ece_and_entropy(
    probs_flat: np.ndarray,
    gt_flat: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and mean Shannon entropy on ROI pixels."""
    if probs_flat.shape[0] == 0:
        return 0.0, 0.0
    p_clip = np.clip(probs_flat, 1e-6, 1.0)
    entropy = float(-np.sum(p_clip * np.log(p_clip), axis=1).mean())
    conf = probs_flat.max(axis=1)
    pred = probs_flat.argmax(axis=1)
    acc = (pred == gt_flat).astype(np.float64)

    ece = 0.0
    n = float(len(gt_flat))
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        in_bin = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        if np.any(in_bin):
            ece += (float(in_bin.sum()) / n) * abs(float(acc[in_bin].mean()) - float(conf[in_bin].mean()))
    return round(float(ece), 4), round(float(entropy), 4)


@torch.no_grad()
def _eval_author_stage2_split(
    author_model: nn.Module,
    stage1_model: Optional[WSSSTissueStage1NetCAM],
    loader: DataLoader,
    device: torch.device,
    stage1_norm_mode: str,
    scales: Sequence[float] = (0.75, 1.0, 1.25),
) -> Dict[str, Any]:
    """Evaluate official Author DeepLabV3+ Stage-2 checkpoint on a split."""
    author_model.eval()
    if stage1_model is not None:
        stage1_model.eval()

    modes = ["raw", "ms_tta"]
    if stage1_model is not None:
        modes.extend(["cgm_author_mul0", "cgm_neg1e4", "ms_tta_cgm_neg1e4"])
    acc = {m: _SegConfusion4() for m in modes}

    for batch in loader:
        imgs, gts, _ = _extract_img_gt(batch)
        imgs = imgs.to(device, non_blocking=True)
        gt_np = gts.cpu().numpy()
        valid_fg = (gt_np >= 0) & (gt_np < NUM_FOREGROUND)

        author_imgs = prepare_stage1_input(
            imgs, stage1_norm_mode="zero_one", images_are_imagenet_norm=True
        )
        logits_1x = author_model(author_imgs)[:, :NUM_FOREGROUND].float()
        acc["raw"].update(torch.argmax(logits_1x, dim=1).cpu().numpy(), gt_np, valid_fg)

        probs_ms = _ms_tta_probs(author_model, author_imgs, logits_1x, scales, device, amp_dtype=None)
        acc["ms_tta"].update(torch.argmax(probs_ms, dim=1).cpu().numpy(), gt_np, valid_fg)

        if stage1_model is not None:
            s1_inp = prepare_stage1_input(imgs, stage1_norm_mode=stage1_norm_mode, images_are_imagenet_norm=True)
            _, s1_probs = stage1_model.forward_cam(s1_inp)
            raw_gate = (s1_probs > 0.10).float().view(-1, NUM_FOREGROUND, 1, 1)
            pred_mul0 = torch.argmax(logits_1x * raw_gate, dim=1).cpu().numpy()
            acc["cgm_author_mul0"].update(pred_mul0, gt_np, valid_fg)
            gate = _make_cgm_gate(s1_probs, 0.10)
            acc["cgm_neg1e4"].update(_apply_logit_cgm(logits_1x, gate), gt_np, valid_fg)
            acc["ms_tta_cgm_neg1e4"].update(_apply_prob_cgm(probs_ms, gate), gt_np, valid_fg)

    return {m: acc[m].summary() for m in modes}


@torch.no_grad()
def run_step0_calibration_and_author_anchor(
    stage1_model: WSSSTissueStage1NetCAM,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    author_stage2_ckpt: Optional[Union[str, Path]] = None,
    stage1_norm_mode: str = "zero_one",
    device: Optional[torch.device] = None,
    output_json_path: Union[str, Path] = "/content/stage2_results/STAGE2_TRAINING_METRICS.json",
    max_ece_pixels: int = 200000,
    stage1_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Step-0 Author Stage-2 anchor + Champion bn7 soft-target calibration sweep."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_model = stage1_model.to(dev).eval()

    report: Dict[str, Any] = {
        "step": "step0_calibration_and_author_anchor",
        "stage1_norm_mode": stage1_norm_mode,
        "stage1_sha256": stage1_sha256,
        "author_stage2_anchor": {},
        "bn7_soft_recipe_sweep_val": {},
    }

    if author_stage2_ckpt is not None and Path(author_stage2_ckpt).is_file():
        author_m = build_and_load_deeplabv3plus(
            checkpoint_path=author_stage2_ckpt, device=dev, strict=True
        )
        report["author_stage2_anchor"]["val"] = _eval_author_stage2_split(
            author_m, stage1_model, val_loader, dev, stage1_norm_mode
        )
        if test_loader is not None:
            report["author_stage2_anchor"]["test"] = _eval_author_stage2_split(
                author_m, stage1_model, test_loader, dev, stage1_norm_mode
            )

    recipes: List[Tuple[str, str, float]] = [
        ("raw_l1", "raw_l1", 1.0),
        ("softmax_t_0.15", "softmax_t", 0.15),
        ("softmax_t_0.25", "softmax_t", 0.25),
        ("softmax_t_0.40", "softmax_t", 0.40),
        ("sg_dap_soft", "sg_dap_soft", 0.25),
    ]
    sg_dap = StainGuidedAffinityRefiner(
        kernel_size=5, num_iters=10, use_deep_affinity=False
    ).to(dev).eval()
    acc_map = {name: _SegConfusion4() for name, _, _ in recipes}
    ece_probs: Dict[str, List[np.ndarray]] = {name: [] for name, _, _ in recipes}
    ece_gts: List[np.ndarray] = []
    collected_px = 0

    for batch in val_loader:
        imgs, gts, _ = _extract_img_gt(batch)
        imgs = imgs.to(dev, non_blocking=True)
        gt_np = gts.cpu().numpy()
        valid_fg = (gt_np >= 0) & (gt_np < NUM_FOREGROUND)
        y_gt = torch.stack(
            [(gts == c).any(dim=(-2, -1)) for c in range(NUM_FOREGROUND)], dim=1
        ).float().to(dev)

        rgb_01 = prepare_stage1_input(imgs, stage1_norm_mode="zero_one", images_are_imagenet_norm=True)
        s1_inp = prepare_stage1_input(imgs, stage1_norm_mode=stage1_norm_mode, images_are_imagenet_norm=True)
        cam_ms = _extract_bn7_6view_global_cam(stage1_model, s1_inp, target_hw=gt_np.shape[-2:])

        take_px = collected_px < max_ece_pixels and bool(np.any(valid_fg))
        if take_px:
            gt_v = gt_np[valid_fg]
            if gt_v.shape[0] > 900:
                step = max(1, gt_v.shape[0] // 900)
                gt_sub = gt_v[::step][:900]
            else:
                step = 1
                gt_sub = gt_v
            ece_gts.append(gt_sub)
            collected_px += len(gt_sub)

        for name, smode, stemp in recipes:
            q_soft = _cams_to_soft_probs(
                cam_ms=cam_ms,
                y_gt=y_gt,
                rgb_01=rgb_01,
                soft_mode=smode,
                soft_temp=stemp,
                sg_dap=sg_dap if smode == "sg_dap_soft" else None,
            )
            pred = torch.argmax(q_soft, dim=1).cpu().numpy()
            acc_map[name].update(pred, gt_np, valid_fg)
            if take_px:
                q_np = np.transpose(q_soft.cpu().numpy(), (0, 2, 3, 1))[valid_fg]
                if step > 1:
                    q_np = q_np[::step]
                q_np = q_np[:900]
                ece_probs[name].append(q_np)

    gt_cat = np.concatenate(ece_gts, axis=0) if ece_gts else np.zeros((0,), dtype=np.int64)
    for name, _, _ in recipes:
        res = acc_map[name].summary()
        p_cat = np.concatenate(ece_probs[name], axis=0) if ece_probs[name] else np.zeros((0, 4), dtype=np.float32)
        ece, ent = _compute_ece_and_entropy(p_cat, gt_cat)
        res["ECE"] = ece
        res["entropy"] = ent
        report["bn7_soft_recipe_sweep_val"][name] = res

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.with_suffix(out_p.suffix + ".tmp")
    tmp_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp_p, out_p)
    return report
