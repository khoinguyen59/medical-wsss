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

import gc
import json
import os
import tempfile
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
from src.stage2_v2.config import VALID_SOFT_MODES, validate_track_name
from src.stage2_v2.memmap_dataset import (
    _cams_to_soft_probs,
    _close_memmap,
    _extract_bn7_6view_global_cam,
    validate_track_a_teacher_checkpoint,
)
from src.training.dataset_loader import (
    FOREGROUND_CLASSES,
    NUM_FOREGROUND,
    extract_patient_id,
    load_bcss_patient_splits,
)
from src.training.eval_stage1_full import compute_file_sha256
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


def _collect_loader_patient_ids(loader: DataLoader) -> List[str]:
    ds: Any = loader.dataset
    indices: Optional[List[int]] = None
    while isinstance(ds, torch.utils.data.Subset):
        sub_idx = [int(i) for i in ds.indices]
        if indices is None:
            indices = sub_idx
        else:
            indices = [sub_idx[i] for i in indices]
        ds = ds.dataset

    files = getattr(ds, "files", None)
    if files is not None:
        if indices is not None:
            return [extract_patient_id(Path(files[i]).name) for i in indices]
        return [extract_patient_id(Path(f).name) for f in files]

    samples = getattr(ds, "samples", None)
    if samples is not None:
        sel = [samples[i] for i in indices] if indices is not None else samples
        pids: List[str] = []
        for item in sel:
            p = item[0] if isinstance(item, (tuple, list)) else item
            pids.append(extract_patient_id(Path(p).name))
        return pids
    return []


@torch.no_grad()
def evaluate_r2_pseudo_mask_val_diagnostic(
    val_loader: DataLoader,
    teacher_ckpt_paths: Sequence[Union[str, Path]],
    stage1_model: Optional[WSSSTissueStage1NetCAM] = None,
    stage1_norm_mode: str = "zero_one",
    stage1_sha256: Optional[str] = None,
    soft_mode: str = "sg_dap_soft",
    soft_temp: float = 0.25,
    blend: float = 0.5,
    weights: Optional[Sequence[float]] = None,
    r2_temp: float = 1.0,
    tail_noise_thresh: float = 1e-2,
    p_sum_diag_thresh: float = 0.05,
    ceiling_gate_thresh: float = 0.755,
    track: str = "track_a_clean",
    device: Optional[torch.device] = None,
    amp_dtype: Optional[torch.dtype] = None,
    output_json_path: Optional[Union[str, Path]] = "/content/stage2_results/STAGE2_R2_VAL_DIAGNOSTIC.json",
    max_ece_pixels: int = 200000,
) -> Dict[str, Any]:
    """Evaluate q_s1_sgdap vs p_ens4_uncal_ygated vs q_r2 strictly on val_loader (zero TEST access)."""
    if len(teacher_ckpt_paths) == 0:
        raise ValueError("teacher_ckpt_paths must contain at least 1 teacher checkpoint.")
    n_teachers = len(teacher_ckpt_paths)
    if weights is None:
        w_list = [1.0 / float(n_teachers)] * n_teachers
    else:
        if len(weights) != n_teachers:
            raise ValueError(
                f"weights length {len(weights)} != teacher_ckpt_paths length {n_teachers}."
            )
        w_list = [float(w) for w in weights]
    if any(w < 0.0 for w in w_list) or abs(sum(w_list) - 1.0) > 1e-5:
        raise ValueError(f"weights must be non-negative and sum to 1.0, got {w_list}.")
    alpha = float(blend)
    temp_val = float(r2_temp)
    if temp_val <= 0.0:
        raise ValueError(f"r2_temp must be > 0, got {r2_temp}.")
    soft_mode_norm = str(soft_mode).strip().lower()
    if soft_mode_norm not in VALID_SOFT_MODES:
        raise ValueError(
            f"Unsupported soft_mode={soft_mode!r} in evaluate_r2_pseudo_mask_val_diagnostic. "
            f"Expected one of {VALID_SOFT_MODES}."
        )
    track_norm = validate_track_name(track)
    enforce_track_a = track_norm == "track_a_clean"

    # Strict VAL-only anti-leakage check (zero TEST or TRAIN patients)
    splits = load_bcss_patient_splits()
    test_pids = {str(p).upper() for p in splits["test"]}
    train_pids = {str(p).upper() for p in splits["train"]}
    val_pids_found = {p.upper() for p in _collect_loader_patient_ids(val_loader)}
    if len(val_pids_found) == 0:
        raise RuntimeError(
            "[STAGE2 ANTI-LEAKAGE ERROR] evaluate_r2_pseudo_mask_val_diagnostic could not verify "
            "patient IDs from val_loader.dataset (empty or missing .files/.samples)."
        )
    leaked = sorted(val_pids_found & (test_pids | train_pids))
    if leaked:
        raise RuntimeError(
            f"[STAGE2 ANTI-LEAKAGE ERROR] evaluate_r2_pseudo_mask_val_diagnostic received TEST/TRAIN patients: {leaked[:5]}"
        )

    teacher_paths = [Path(p) for p in teacher_ckpt_paths]
    teacher_sha256s: List[str] = []
    for p in teacher_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {p}")
        if enforce_track_a:
            validate_track_a_teacher_checkpoint(p)
        teacher_sha256s.append(compute_file_sha256(p))

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda" and amp_dtype is not None
    if stage1_model is not None:
        stage1_model = stage1_model.to(dev).eval()

    n_val = 0
    h = 0
    w = 0
    for batch in val_loader:
        imgs_b, _, _ = _extract_img_gt(batch)
        if n_val == 0:
            h, w = int(imgs_b.shape[-2]), int(imgs_b.shape[-1])
        n_val += int(imgs_b.shape[0])
    if n_val == 0 or h == 0 or w == 0:
        raise RuntimeError("val_loader yielded 0 batches for evaluate_r2_pseudo_mask_val_diagnostic.")

    from src.stage2_v2.models import load_stage2_v2_checkpoint

    acc_mm: Optional[np.memmap] = None
    tmp_acc_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=".tmp_val_diag_ens4_", suffix=".dat", delete=False
        ) as tfp:
            tmp_acc_path = Path(tfp.name)

        acc_mm = np.memmap(
            tmp_acc_path,
            dtype=np.float32,
            mode="w+",
            shape=(n_val, NUM_FOREGROUND, h, w),
        )
        acc_mm[:] = 0.0

        for t_idx, ckpt_p in enumerate(teacher_paths):
            teacher_m = None
            try:
                teacher_m, t_meta = load_stage2_v2_checkpoint(ckpt_p, device=dev, strict=True)
                if enforce_track_a:
                    validate_track_a_teacher_checkpoint(ckpt_p, payload_meta=t_meta)
                teacher_m.eval()

                w_t = float(w_list[t_idx])
                pos = 0
                for batch in val_loader:
                    imgs_b, _, _ = _extract_img_gt(batch)
                    b = int(imgs_b.size(0))
                    imgs_b = imgs_b.to(dev, non_blocking=True)
                    views = (
                        imgs_b,
                        torch.flip(imgs_b, dims=[-1]),
                        torch.flip(imgs_b, dims=[-2]),
                        torch.rot90(imgs_b, k=1, dims=[-2, -1]),
                    )
                    probs_sum = torch.zeros(
                        (b, NUM_FOREGROUND, h, w), device=dev, dtype=torch.float32
                    )
                    for v_idx, x_v in enumerate(views):
                        with torch.amp.autocast(
                            device_type=dev.type, dtype=amp_dtype, enabled=use_amp
                        ):
                            lg_v, _ = _forward_dual_head(teacher_m, x_v)
                        pr_v = torch.softmax(lg_v[:, :NUM_FOREGROUND].float(), dim=1)
                        if v_idx == 1:
                            pr_v = torch.flip(pr_v, dims=[-1])
                        elif v_idx == 2:
                            pr_v = torch.flip(pr_v, dims=[-2])
                        elif v_idx == 3:
                            pr_v = torch.rot90(pr_v, k=3, dims=[-2, -1])
                        probs_sum = probs_sum + pr_v

                    p_d4 = (probs_sum / 4.0).cpu().numpy()
                    acc_mm[pos : pos + b] = acc_mm[pos : pos + b] + w_t * p_d4
                    pos += b
            finally:
                del teacher_m
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

        acc_mm.flush()

        cand_names = ("q_s1_sgdap", "p_ens4_uncal_ygated", "q_r2")
        overall_acc = {name: _SegConfusion4() for name in cand_names}
        multi_acc = {name: _SegConfusion4() for name in cand_names}
        ece_probs: Dict[str, List[np.ndarray]] = {name: [] for name in cand_names}
        ece_gts: List[np.ndarray] = []
        multi_ent_sum = {name: 0.0 for name in cand_names}
        multi_ent_px = 0
        collected_px = 0
        n_val_multi = 0
        psum_lt_0_05_px = 0
        psum_lt_0_01_px = 0
        psum_diag_total_px = 0
        psum_all_lt_0_05_px = 0
        psum_all_lt_0_01_px = 0
        psum_all_total_px = 0

        sg_dap = (
            StainGuidedAffinityRefiner(kernel_size=5, num_iters=10, use_deep_affinity=False)
            .to(dev)
            .eval()
            if (stage1_model is not None and soft_mode_norm == "sg_dap_soft")
            else None
        )

        pos = 0
        for batch in val_loader:
            imgs_b, gts_b, _ = _extract_img_gt(batch)
            b = int(imgs_b.size(0))
            imgs_b = imgs_b.to(dev, non_blocking=True)
            gt_np = gts_b.cpu().numpy()
            valid_fg = (gt_np >= 0) & (gt_np < NUM_FOREGROUND)

            y_gt = torch.stack(
                [(gts_b == c).any(dim=(-2, -1)) for c in range(NUM_FOREGROUND)], dim=1
            ).float().to(dev)
            is_multi_b = (y_gt > 0.5).sum(dim=1) >= 2
            is_multi_np = is_multi_b.cpu().numpy()
            n_val_multi += int(np.sum(is_multi_np))

            raw_gate = (y_gt > 0.5).float().view(b, NUM_FOREGROUND, 1, 1)
            has_any = raw_gate.any(dim=1, keepdim=True)
            gate = torch.where(has_any, raw_gate, torch.ones_like(raw_gate))
            uniform_gt = gate / gate.sum(dim=1, keepdim=True).clamp(min=1.0)

            if stage1_model is not None:
                rgb_01 = prepare_stage1_input(
                    imgs_b, stage1_norm_mode="zero_one", images_are_imagenet_norm=True
                )
                s1_inp = prepare_stage1_input(
                    imgs_b, stage1_norm_mode=stage1_norm_mode, images_are_imagenet_norm=True
                )
                cam_ms = _extract_bn7_6view_global_cam(stage1_model, s1_inp, target_hw=(h, w))
                q_s1 = _cams_to_soft_probs(
                    cam_ms=cam_ms,
                    y_gt=y_gt,
                    rgb_01=rgb_01,
                    soft_mode=soft_mode_norm,
                    soft_temp=soft_temp,
                    sg_dap=sg_dap,
                )
            else:
                q_s1 = uniform_gt

            p_ens4 = torch.from_numpy(np.array(acc_mm[pos : pos + b], dtype=np.float32)).to(dev)
            pos += b
            if abs(temp_val - 1.0) > 1e-6:
                p_ens4 = p_ens4.clamp(min=1e-8) ** (1.0 / temp_val)
                p_ens4 = p_ens4 / p_ens4.sum(dim=1, keepdim=True).clamp(min=1e-8)

            p_g = p_ens4 * gate
            p_sum = p_g.sum(dim=1, keepdim=True)
            tail_mask = p_sum < float(tail_noise_thresh)
            p_ens4_ygated = torch.where(
                tail_mask,
                q_s1,
                p_g / p_sum.clamp(min=float(tail_noise_thresh)),
            )

            q_r2 = (1.0 - alpha) * q_s1 + alpha * p_ens4_ygated
            q_r2 = q_r2 * gate
            r2_sum = q_r2.sum(dim=1, keepdim=True)
            q_r2 = torch.where(r2_sum > 1e-6, q_r2 / r2_sum.clamp(min=1e-6), uniform_gt)

            # Track p_sum tail-noise diagnostics strictly on multi-class valid pixels
            # (split-level fallback to valid_fg only if n_val_multi == 0 across entire split)
            p_sum_np = p_sum.squeeze(1).cpu().numpy()
            if np.any(valid_fg):
                psum_all_total_px += int(np.sum(valid_fg))
                psum_all_lt_0_05_px += int(np.sum(p_sum_np[valid_fg] < float(p_sum_diag_thresh)))
                psum_all_lt_0_01_px += int(np.sum(p_sum_np[valid_fg] < float(tail_noise_thresh)))
            diag_mask = valid_fg & is_multi_np[:, None, None]
            if np.any(diag_mask):
                psum_diag_total_px += int(np.sum(diag_mask))
                psum_lt_0_05_px += int(np.sum(p_sum_np[diag_mask] < float(p_sum_diag_thresh)))
                psum_lt_0_01_px += int(np.sum(p_sum_np[diag_mask] < float(tail_noise_thresh)))

            take_px = collected_px < max_ece_pixels and bool(np.any(valid_fg))
            step = 1
            if take_px:
                gt_v = gt_np[valid_fg]
                if gt_v.shape[0] > 900:
                    step = max(1, gt_v.shape[0] // 900)
                    gt_sub = gt_v[::step][:900]
                else:
                    gt_sub = gt_v
                ece_gts.append(gt_sub)
                collected_px += len(gt_sub)

            cand_tensors = {
                "q_s1_sgdap": q_s1,
                "p_ens4_uncal_ygated": p_ens4_ygated,
                "q_r2": q_r2,
            }
            if np.any(is_multi_np):
                multi_valid_fg = valid_fg & is_multi_np[:, None, None]
                multi_ent_px += int(np.sum(multi_valid_fg))

            for cname, c_tensor in cand_tensors.items():
                pred = torch.argmax(c_tensor, dim=1).cpu().numpy()
                overall_acc[cname].update(pred, gt_np, valid_fg)
                if np.any(is_multi_np):
                    multi_acc[cname].update(pred, gt_np, valid_fg & is_multi_np[:, None, None])
                    c_np_full = np.transpose(c_tensor.cpu().numpy(), (0, 2, 3, 1))
                    m_probs = np.clip(c_np_full[valid_fg & is_multi_np[:, None, None]], 1e-6, 1.0)
                    if m_probs.shape[0] > 0:
                        multi_ent_sum[cname] += float(-np.sum(m_probs * np.log(m_probs)))
                if take_px:
                    q_np = np.transpose(c_tensor.cpu().numpy(), (0, 2, 3, 1))[valid_fg]
                    if step > 1:
                        q_np = q_np[::step]
                    ece_probs[cname].append(q_np[:900])

        gt_cat = np.concatenate(ece_gts, axis=0) if ece_gts else np.zeros((0,), dtype=np.int64)
        candidates_report: Dict[str, Any] = {}
        for cname in cand_names:
            ov = overall_acc[cname].summary()
            mv = multi_acc[cname].summary()
            p_cat = (
                np.concatenate(ece_probs[cname], axis=0)
                if ece_probs[cname]
                else np.zeros((0, NUM_FOREGROUND), dtype=np.float32)
            )
            ece, ent = _compute_ece_and_entropy(p_cat, gt_cat)
            multi_ent = (
                round(float(multi_ent_sum[cname] / float(multi_ent_px)), 4)
                if multi_ent_px > 0
                else ent
            )
            candidates_report[cname] = {
                **ov,
                "multiclass_mIoU": mv["mIoU"],
                "multiclass_mDice": mv["mDice"],
                "multiclass_per_class_iou": mv["per_class_iou"],
                "multiclass_class_ious": mv["class_ious"],
                "ECE": ece,
                "entropy": ent,
                "multiclass_entropy": multi_ent,
            }

        if n_val_multi > 0:
            eff_diag_total = psum_diag_total_px
            eff_lt_0_05 = psum_lt_0_05_px
            eff_lt_0_01 = psum_lt_0_01_px
        else:
            eff_diag_total = psum_all_total_px
            eff_lt_0_05 = psum_all_lt_0_05_px
            eff_lt_0_01 = psum_all_lt_0_01_px

        frac_005 = (
            round(float(eff_lt_0_05) / float(eff_diag_total), 6)
            if eff_diag_total > 0
            else 0.0
        )
        frac_001 = (
            round(float(eff_lt_0_01) / float(eff_diag_total), 6)
            if eff_diag_total > 0
            else 0.0
        )
        q_r2_miou = float(candidates_report["q_r2"]["mIoU"])
        q_r2_lym = float(candidates_report["q_r2"]["per_class_iou"]["lym"])
        s1_lym = float(candidates_report["q_s1_sgdap"]["per_class_iou"]["lym"])
        ceiling_gate_pass = bool(q_r2_miou >= float(ceiling_gate_thresh))
        go_decision = bool(ceiling_gate_pass and q_r2_lym >= s1_lym)

        report: Dict[str, Any] = {
            "step": "diag_r2_val",
            "split": "val_only",
            "num_val_patches": n_val,
            "num_multiclass_val_patches": n_val_multi,
            "track": track_norm,
            "soft_mode": soft_mode_norm,
            "soft_temp": float(soft_temp),
            "blend": alpha,
            "r2_temp": temp_val,
            "tail_noise_thresh": float(tail_noise_thresh),
            "stage1_norm_mode": stage1_norm_mode,
            "stage1_sha256": stage1_sha256,
            "teacher_sha256s": teacher_sha256s,
            "frac_psum_lt_0_05": frac_005,
            "frac_psum_lt_0_01": frac_001,
            "candidates": candidates_report,
            "ceiling_gate_thresh": float(ceiling_gate_thresh),
            "ceiling_gate_pass": ceiling_gate_pass,
            "go_decision": go_decision,
        }

        if output_json_path is not None:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(report, indent=2)
            tmp_p = out_p.with_suffix(out_p.suffix + ".tmp")
            tmp_p.write_text(serialized, encoding="utf-8")
            os.replace(tmp_p, out_p)
            diag_sidecar = out_p.parent / "STAGE2_R2_VAL_DIAGNOSTIC.json"
            if diag_sidecar.resolve() != out_p.resolve():
                tmp_sidecar = diag_sidecar.with_suffix(".json.tmp")
                tmp_sidecar.write_text(serialized, encoding="utf-8")
                os.replace(tmp_sidecar, diag_sidecar)

        return report
    finally:
        _close_memmap(acc_mm)
        del acc_mm
        if tmp_acc_path is not None:
            try:
                tmp_acc_path.unlink(missing_ok=True)
            except OSError:
                pass
