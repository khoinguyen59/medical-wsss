"""
Zero-Leakage 4-Arm Cross-Family Ensemble & Publication-Grade Evaluation Suite.

Implements the Gate-5 post-Wave-2B cross-family ensemble pipeline:
1. Canonical BCSSStage2ValDataset patient-level anti-leakage verification.
2. Per-arm discrete TTA selection strictly on VAL (raw_1x, d4_8view_1x, ms_up_hflip, ms_3_hflip)
   plus VAL patch AUROC/Brier verification for cls_probs (1.0x vs 1.0x-D4 vs selected-TTA).
3. Disk-streaming centered float16 log-prob memmaps (zero underflow, < 4 GB VRAM)
   with 1D bounded NLL temperature calibration (beta_m in [0.25, 4.0]) on deterministic
   stride-8 ROI pixels.
4. Exact disagreement-pixel acceleration across all 969 simplex weight vectors
   (step=0.05, floor>=0.05) + Uniform [0.25]*4 inside zero-leakage 5-fold stratified
   patient CV and N_val-fold LOPOCV stability check.
5. 5-expert CGM calibration (stage1_best_cls.pth + 4 Stage-2 cls_heads) directly on
   fused TTA probs over ENSEMBLE_CGM_GRID = (0.05, ..., 0.65) with 15-bitmask exact parity.
6. Complete TEST ablations + BOTH Official Author Anchors + 23-patient publication stats
   (10,000-bootstrap 95% CI, exact paired patient-swap pooled permutation test,
   two-sided Wilcoxon signed-rank + Cohen's dz, and m=5 Holm-Bonferroni) + 300 DPI
   Academic Monochrome figure and compact results ZIP.

Pure ASCII hyphen '-' (0x2D) only (Invariant 12).
"""

import csv
import gc
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, wilcoxon
from torch.utils.data import DataLoader

from src.models.wsss_inference_methods import (
    WSSSTissueStage1NetCAM,
    build_and_load_deeplabv3plus,
)
from src.stage2_v2.evaluator import (
    _apply_logit_cgm,
    _apply_prob_cgm,
    _extract_img_gt,
    _forward_dual_head,
    _make_cgm_gate,
    _ms_tta_probs,
)
from src.stage2_v2.models import BaseStage2DualHead, load_stage2_v2_checkpoint
from src.training.dataset_loader import FOREGROUND_CLASSES, NUM_FOREGROUND, extract_patient_id
from src.training.eval_stage1_full import compute_file_sha256, load_stage1_strict
from src.training.train_stage2 import prepare_stage1_input

_CLS_LOWER: Tuple[str, ...] = tuple(c.lower() for c in FOREGROUND_CLASSES)

ENSEMBLE_CGM_GRID: Tuple[float, ...] = (
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
    0.60,
    0.65,
)

TTA_CANDIDATES: Tuple[str, ...] = ("raw_1x", "d4_8view_1x", "ms_up_hflip", "ms_3_hflip")
TTA_VIEW_COUNTS: Dict[str, int] = {
    "raw_1x": 1,
    "ms_up_hflip": 4,
    "ms_3_hflip": 6,
    "d4_8view_1x": 8,
}

DEFAULT_ARM_NAMES: Tuple[str, ...] = (
    "arm1_deeplab_cleankd",
    "arm2_segformer_oeem",
    "arm3_convnext_selftrain",
    "arm4_dinov2_adapter",
)
DEFAULT_ARM_FAMILIES: Dict[str, str] = {
    "arm1_deeplab_cleankd": "cnn",
    "arm2_segformer_oeem": "vit",
    "arm3_convnext_selftrain": "cnn",
    "arm4_dinov2_adapter": "vit",
}

SHA256_STAGE1_BEST_CLS = "e1a88293aff0a0e3b8422b08140346464ea4c95fddb301aac71ad5a860c7dfb2"
SHA256_STAGE1_RETRAINED = "a182d8890c09d7b88ee06810769ed7ac1bd84280df0beddb7898d5a43125de0b"
SHA256_AUTHOR_STAGE1 = "1b5800e80be88ff552640f16c4b9efbe96e5b8969571e158c3ee51eb283da333"
SHA256_AUTHOR_STAGE2 = "f2c09b5395927c48a7060ae5b08381069f498d0108b35789d60f152a22319eb4"


def _close_memmap(mm: Any) -> None:
    """Explicitly flush and close numpy memmap handle for Windows cleanup safety."""
    if mm is None:
        return
    try:
        mm.flush()
    except Exception:
        pass
    inner = getattr(mm, "_mmap", None)
    if inner is not None:
        try:
            inner.close()
        except Exception:
            pass


def compute_macro_miou_from_iu(inter: np.ndarray, union: np.ndarray) -> float:
    """Compute 4-class macro mIoU from intersection and union sums, guarding union == 0."""
    inter_a = np.asarray(inter, dtype=np.float64)
    union_a = np.asarray(union, dtype=np.float64)
    valid_c = union_a > 0
    if not np.any(valid_c):
        return 0.0
    return float(np.mean(inter_a[valid_c] / union_a[valid_c]))


def summarize_iu(inter: np.ndarray, union: np.ndarray) -> Dict[str, Any]:
    """Build mIoU, mDice, and per-class IoU/Dice summary from (4,) inter/union arrays."""
    inter_a = np.asarray(inter, dtype=np.float64)
    union_a = np.asarray(union, dtype=np.float64)
    ious = [float(inter_a[c] / union_a[c]) if union_a[c] > 0 else 0.0 for c in range(NUM_FOREGROUND)]
    dices = [
        float((2.0 * inter_a[c]) / (union_a[c] + inter_a[c])) if (union_a[c] + inter_a[c]) > 0 else 0.0
        for c in range(NUM_FOREGROUND)
    ]
    active_ious = [ious[c] for c in range(NUM_FOREGROUND) if union_a[c] > 0]
    active_dices = [dices[c] for c in range(NUM_FOREGROUND) if (union_a[c] + inter_a[c]) > 0]
    return {
        "mIoU": round(float(np.mean(active_ious)) if active_ious else 0.0, 4),
        "mIoU_raw": float(np.mean(active_ious)) if active_ious else 0.0,
        "mDice": round(float(np.mean(active_dices)) if active_dices else 0.0, 4),
        "per_class_iou": {_CLS_LOWER[c]: round(ious[c], 4) for c in range(NUM_FOREGROUND)},
        "per_class_dice": {_CLS_LOWER[c]: round(dices[c], 4) for c in range(NUM_FOREGROUND)},
        "class_ious": [round(x, 4) for x in ious],
    }


def compute_per_patient_miou(inter_p: np.ndarray, union_p: np.ndarray) -> np.ndarray:
    """Compute per-patient macro mIoU across active classes (union_p[p, c] > 0)."""
    inter_a = np.asarray(inter_p, dtype=np.float64)
    union_a = np.asarray(union_p, dtype=np.float64)
    out = np.zeros(inter_a.shape[0], dtype=np.float64)
    for p in range(inter_a.shape[0]):
        out[p] = compute_macro_miou_from_iu(inter_a[p], union_a[p])
    return out


def generate_simplex_grid(
    num_arms: int = 4,
    step: float = 0.05,
    min_weight: float = 0.05,
) -> np.ndarray:
    """Generate all simplex weight vectors with exact step and floor (969 for 4 arms, step=0.05, min=0.05)."""
    units = int(round(1.0 / float(step)))
    min_u = int(round(float(min_weight) / float(step)))
    if not math.isclose(units * step, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"step={step} must evenly divide 1.0")
    if min_u * num_arms > units:
        raise ValueError("min_weight * num_arms cannot exceed 1.0")

    rows: List[List[int]] = []

    def _rec(idx: int, rem: int, cur: List[int]) -> None:
        if idx == num_arms - 1:
            if rem >= min_u:
                rows.append([*cur, rem])
            return
        max_here = rem - min_u * (num_arms - 1 - idx)
        for v in range(min_u, max_here + 1):
            cur.append(v)
            _rec(idx + 1, rem - v, cur)
            cur.pop()

    _rec(0, units, [])
    return np.asarray(rows, dtype=np.float32) * np.float32(step)


def _forward_single_view(
    model: nn.Module,
    x_in: torch.Tensor,
    target_hw: Tuple[int, int],
    k_rot: int = 0,
    do_flip: bool = False,
    device: Optional[torch.device] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one transformed view and invert spatial transformation cleanly in FP32."""
    dev = device or x_in.device
    use_amp = dev.type == "cuda" and amp_dtype is not None
    with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
        seg_lg, cls_lg = _forward_dual_head(model, x_in)
    pr = torch.softmax(seg_lg.float(), dim=1)
    if do_flip:
        pr = torch.flip(pr, dims=[-1])
    if k_rot % 4 != 0:
        pr = torch.rot90(pr, k=(4 - (k_rot % 4)) % 4, dims=[-2, -1])
    if pr.shape[-2:] != target_hw:
        pr = F.interpolate(pr, size=target_hw, mode="bilinear", align_corners=False)
    if cls_lg is not None:
        cls_pr = torch.sigmoid(cls_lg.float())
    else:
        cls_pr = pr.amax(dim=(2, 3))
    return pr, cls_pr


@torch.no_grad()
def apply_tta_probs(
    model: nn.Module,
    imgs: torch.Tensor,
    tta_mode: str,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    include_d4_cls: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run discrete TTA mode and return FP32 seg probs + dict of cls probs (1x, d4, tta)."""
    mode = str(tta_mode).strip().lower()
    if mode not in TTA_CANDIDATES:
        raise ValueError(f"Unsupported TTA mode {tta_mode!r}; expected one of {TTA_CANDIDATES}")
    b, _, h, w = imgs.shape
    target_hw = (h, w)
    seg_sum = torch.zeros((b, NUM_FOREGROUND, h, w), device=device, dtype=torch.float32)
    cls_sum = torch.zeros((b, NUM_FOREGROUND), device=device, dtype=torch.float32)
    cls_1x: Optional[torch.Tensor] = None
    count = 0

    if mode == "raw_1x":
        views: List[Tuple[float, int, bool]] = [(1.0, 0, False)]
    elif mode == "d4_8view_1x":
        views = [(1.0, k, flp) for k in range(4) for flp in (False, True)]
    elif mode == "ms_up_hflip":
        views = [(sc, 0, flp) for sc in (1.0, 1.25) for flp in (False, True)]
    else:
        views = [(sc, 0, flp) for sc in (0.75, 1.0, 1.25) for flp in (False, True)]

    d4_cache: Dict[Tuple[int, bool], torch.Tensor] = {}
    for sc, k_rot, do_flip in views:
        if abs(sc - 1.0) > 1e-4:
            nh = max(16, int(round(h * sc)))
            nw = max(16, int(round(w * sc)))
            x_v = F.interpolate(imgs, size=(nh, nw), mode="bilinear", align_corners=False)
        else:
            x_v = imgs
        if k_rot % 4 != 0:
            x_v = torch.rot90(x_v, k=k_rot, dims=[-2, -1])
        if do_flip:
            x_v = torch.flip(x_v, dims=[-1])
        pr, cls_pr = _forward_single_view(
            model, x_v, target_hw=target_hw, k_rot=k_rot, do_flip=do_flip, device=device, amp_dtype=amp_dtype
        )
        if abs(sc - 1.0) < 1e-4:
            d4_cache[(k_rot, do_flip)] = cls_pr
            if k_rot == 0 and not do_flip:
                cls_1x = cls_pr
        seg_sum = seg_sum + pr
        cls_sum = cls_sum + cls_pr
        count += 1

    seg_avg = seg_sum / float(max(1, count))
    cls_tta = cls_sum / float(max(1, count))
    if cls_1x is None:
        _, cls_1x = _forward_single_view(
            model, imgs, target_hw=target_hw, k_rot=0, do_flip=False, device=device, amp_dtype=amp_dtype
        )
        d4_cache[(0, False)] = cls_1x
    if mode == "d4_8view_1x":
        cls_d4 = cls_tta
    elif include_d4_cls:
        for k in range(4):
            x_r = torch.rot90(imgs, k=k, dims=[-2, -1]) if k != 0 else imgs
            for flp in (False, True):
                if (k, flp) in d4_cache:
                    continue
                x_v = torch.flip(x_r, dims=[-1]) if flp else x_r
                _, cpr = _forward_single_view(
                    model, x_v, target_hw=target_hw, k_rot=k, do_flip=flp, device=device, amp_dtype=amp_dtype
                )
                d4_cache[(k, flp)] = cpr
        cls_d4 = sum(d4_cache.values()) / float(len(d4_cache))
    else:
        cls_d4 = cls_tta
    return seg_avg, {"1x": cls_1x, "d4": cls_d4, "tta": cls_tta}


@torch.no_grad()
def _collect_all_tta_candidates_batch(
    model: nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute all 4 discrete TTA candidates in 12 shared views (vs 19 separate views)."""
    _, _, h, w = imgs.shape
    target_hw = (h, w)

    # 1.0x D4 group (8 views)
    d4_seg: Dict[Tuple[int, bool], torch.Tensor] = {}
    d4_cls: Dict[Tuple[int, bool], torch.Tensor] = {}
    for k in range(4):
        x_r = torch.rot90(imgs, k=k, dims=[-2, -1]) if k != 0 else imgs
        for flp in (False, True):
            x_v = torch.flip(x_r, dims=[-1]) if flp else x_r
            pr, cpr = _forward_single_view(
                model, x_v, target_hw=target_hw, k_rot=k, do_flip=flp, device=device, amp_dtype=amp_dtype
            )
            d4_seg[(k, flp)] = pr
            d4_cls[(k, flp)] = cpr

    # 1.25x and 0.75x HFlip (4 additional views)
    scaled_seg: Dict[Tuple[float, bool], torch.Tensor] = {}
    scaled_cls: Dict[Tuple[float, bool], torch.Tensor] = {}
    for sc in (0.75, 1.25):
        nh = max(16, int(round(h * sc)))
        nw = max(16, int(round(w * sc)))
        x_s = F.interpolate(imgs, size=(nh, nw), mode="bilinear", align_corners=False)
        for flp in (False, True):
            x_v = torch.flip(x_s, dims=[-1]) if flp else x_s
            pr, cpr = _forward_single_view(
                model, x_v, target_hw=target_hw, k_rot=0, do_flip=flp, device=device, amp_dtype=amp_dtype
            )
            scaled_seg[(sc, flp)] = pr
            scaled_cls[(sc, flp)] = cpr

    seg_raw = d4_seg[(0, False)]
    cls_raw = d4_cls[(0, False)]

    seg_d4 = sum(d4_seg.values()) / 8.0
    cls_d4 = sum(d4_cls.values()) / 8.0

    up_views_s = [d4_seg[(0, False)], d4_seg[(0, True)], scaled_seg[(1.25, False)], scaled_seg[(1.25, True)]]
    up_views_c = [d4_cls[(0, False)], d4_cls[(0, True)], scaled_cls[(1.25, False)], scaled_cls[(1.25, True)]]
    seg_ms_up = sum(up_views_s) / 4.0
    cls_ms_up = sum(up_views_c) / 4.0

    ms3_views_s = up_views_s + [scaled_seg[(0.75, False)], scaled_seg[(0.75, True)]]
    ms3_views_c = up_views_c + [scaled_cls[(0.75, False)], scaled_cls[(0.75, True)]]
    seg_ms_3 = sum(ms3_views_s) / 6.0
    cls_ms_3 = sum(ms3_views_c) / 6.0

    seg_out = {
        "raw_1x": seg_raw,
        "d4_8view_1x": seg_d4,
        "ms_up_hflip": seg_ms_up,
        "ms_3_hflip": seg_ms_3,
    }
    cls_out = {
        "raw_1x": cls_raw,
        "d4_8view_1x": cls_d4,
        "ms_up_hflip": cls_ms_up,
        "ms_3_hflip": cls_ms_3,
    }
    return seg_out, cls_out


def encode_centered_logprobs_fp16(probs: np.ndarray, eps: float = 1e-7, axis: int = 1) -> np.ndarray:
    """Encode probabilities into centered float16 log-probs in [-16.12, 0.0] with zero underflow."""
    p_clip = np.clip(np.asarray(probs, dtype=np.float32), float(eps), 1.0)
    log_p = np.log(p_clip)
    z = log_p - np.max(log_p, axis=axis, keepdims=True)
    return z.astype(np.float16)


def reconstruct_probs_from_centered(
    z_fp16: np.ndarray,
    beta: float = 1.0,
    eps_log: Optional[float] = None,
    axis: int = 1,
) -> np.ndarray:
    """Reconstruct softmax(beta * z) or log(clamp(softmax(beta * z), eps_log)) from centered float16 log-probs."""
    logits = np.asarray(z_fp16, dtype=np.float32) * float(beta)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l, axis=axis, keepdims=True)
    if eps_log is not None:
        return np.log(np.clip(probs, float(eps_log), 1.0))
    return probs


def fit_temperature_beta_nll(
    z_roi_fp16: np.ndarray,
    gt_roi: np.ndarray,
    bounds: Tuple[float, float] = (0.25, 4.0),
) -> float:
    """Fit scalar inverse-temperature beta = 1/T in [0.25, 4.0] minimizing ROI NLL."""
    z = np.asarray(z_roi_fp16, dtype=np.float32)
    gt = np.asarray(gt_roi, dtype=np.int64)
    if z.shape[0] == 0:
        return 1.0
    idx = np.arange(z.shape[0])
    z_gt = z[idx, gt]

    def _nll(beta: float) -> float:
        bz = z * float(beta)
        m = np.max(bz, axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.sum(np.exp(bz - m), axis=1))
        return float(np.mean(lse - float(beta) * z_gt))

    res = minimize_scalar(_nll, bounds=bounds, method="bounded")
    return float(np.clip(float(res.x), bounds[0], bounds[1]))


def fit_binary_cls_temperature_bce(
    probs: np.ndarray,
    targets: np.ndarray,
    bounds: Tuple[float, float] = (0.25, 4.0),
) -> float:
    """Fit scalar temperature beta on multi-label patch logits minimizing BCE."""
    p_clip = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(targets, dtype=np.float64)
    if p_clip.shape[0] == 0:
        return 1.0
    logits = np.log(p_clip / (1.0 - p_clip))

    def _bce(beta: float) -> float:
        z = logits * float(beta)
        # Stable BCEWithLogits: max(z, 0) - z * y + log(1 + exp(-|z|))
        loss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
        return float(np.mean(loss))

    res = minimize_scalar(_bce, bounds=bounds, method="bounded")
    return float(np.clip(float(res.x), bounds[0], bounds[1]))


def compute_patch_auroc_and_brier(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute macro patch AUROC (via Mann-Whitney U / rankdata) and Brier score across 4 classes."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if p.shape[0] == 0:
        return {"macro_auroc": 0.0, "brier": 0.0}
    brier = float(np.mean((p - y) ** 2))
    aurocs: List[float] = []
    for c in range(NUM_FOREGROUND):
        yc = y[:, c] > 0.5
        n_pos = int(yc.sum())
        n_neg = int((~yc).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(p[:, c], method="average")
        u_stat = float(ranks[yc].sum()) - (n_pos * (n_pos + 1)) / 2.0
        aurocs.append(u_stat / float(n_pos * n_neg))
    macro_auroc = float(np.mean(aurocs)) if aurocs else 0.0
    return {"macro_auroc": round(macro_auroc, 4), "brier": round(brier, 4)}


def build_patient_folds(
    patch_patient_ids: Sequence[str],
    nec_pixels_per_patch: np.ndarray,
    n_folds: int = 5,
) -> Tuple[List[str], np.ndarray, List[List[int]]]:
    """Stratified balanced patient fold assignment sorted by (nec_pixels, patch_count)."""
    unique_pids = sorted(set(str(p) for p in patch_patient_ids))
    pid_to_idx = {pid: idx for idx, pid in enumerate(unique_pids)}
    patch_pidx = np.asarray([pid_to_idx[str(p)] for p in patch_patient_ids], dtype=np.int32)
    n_pat = len(unique_pids)
    nec_p = np.zeros(n_pat, dtype=np.int64)
    cnt_p = np.zeros(n_pat, dtype=np.int64)
    for i, pidx in enumerate(patch_pidx):
        nec_p[pidx] += int(nec_pixels_per_patch[i])
        cnt_p[pidx] += 1

    k_folds = max(1, min(int(n_folds), n_pat))
    order = sorted(range(n_pat), key=lambda p: (-int(nec_p[p]), -int(cnt_p[p]), unique_pids[p]))
    folds: List[List[int]] = [[] for _ in range(k_folds)]
    fold_nec = [0] * k_folds
    fold_cnt = [0] * k_folds
    for p in order:
        best_f = min(range(k_folds), key=lambda f: (fold_nec[f], fold_cnt[f], len(folds[f]), f))
        folds[best_f].append(p)
        fold_nec[best_f] += int(nec_p[p])
        fold_cnt[best_f] += int(cnt_p[p])
    return unique_pids, patch_pidx, folds


def _concat_and_drain(
    chunks: List[np.ndarray],
    tail_shape: Tuple[int, ...],
    dtype: Any,
) -> np.ndarray:
    """Pre-allocate exact output buffer and drain chunk list in place to avoid 2x concat RAM spike."""
    if not chunks:
        return np.zeros((0, *tail_shape), dtype=dtype)
    total = sum(int(c.shape[0]) for c in chunks)
    out = np.empty((total, *tail_shape), dtype=dtype)
    pos = 0
    while chunks:
        c = chunks.pop(0)
        n_c = int(c.shape[0])
        out[pos : pos + n_c] = c
        pos += n_c
    return out


def _extract_disagreement_and_unanimous(
    z_memmaps: Sequence[np.ndarray],
    gt_masks: np.ndarray,
    patch_patient_idx: np.ndarray,
    num_patients: int,
    batch_size: int = 64,
    stride_cal: int = 8,
) -> Dict[str, Any]:
    """Stream over centered float16 log-prob memmaps to split unanimous vs disagreement ROI pixels."""
    n_patches = int(gt_masks.shape[0])
    num_arms = len(z_memmaps)
    inter_u = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
    union_u = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)

    z_dis_chunks: List[np.ndarray] = []
    gt_dis_chunks: List[np.ndarray] = []
    pat_dis_chunks: List[np.ndarray] = []

    z_cal_chunks: List[np.ndarray] = []
    gt_cal_chunks: List[np.ndarray] = []
    pat_cal_chunks: List[np.ndarray] = []

    pair_agree = np.zeros((num_arms, num_arms), dtype=np.int64)
    pair_conf = np.zeros((num_arms, num_arms, NUM_FOREGROUND, NUM_FOREGROUND), dtype=np.int64)
    total_roi_pixels = 0

    for start in range(0, n_patches, int(batch_size)):
        end = min(n_patches, start + int(batch_size))
        gt_raw = gt_masks[start:end]
        valid_b = (gt_raw >= 0) & (gt_raw < NUM_FOREGROUND)
        gt_b = np.where(valid_b, gt_raw, 255).astype(np.uint8)
        pidx_b = patch_patient_idx[start:end]

        # Stack centered log-probs: (B, M, 4, H, W)
        z_b = np.stack([np.asarray(mm[start:end], dtype=np.float16) for mm in z_memmaps], axis=1)
        preds_b = np.argmax(z_b, axis=2).astype(np.uint8)  # (B, M, H, W)

        # Stride-8 calibration subset
        gt_raw_s = gt_raw[:, ::stride_cal, ::stride_cal]
        val_s = (gt_raw_s >= 0) & (gt_raw_s < NUM_FOREGROUND)
        gt_s = gt_b[:, ::stride_cal, ::stride_cal]
        if np.any(val_s):
            z_s = np.transpose(z_b[:, :, :, ::stride_cal, ::stride_cal], (0, 3, 4, 1, 2))
            p_s = np.broadcast_to(pidx_b[:, None, None], gt_s.shape)
            z_cal_chunks.append(z_s[val_s].astype(np.float16))
            gt_cal_chunks.append(gt_s[val_s].astype(np.uint8))
            pat_cal_chunks.append(p_s[val_s].astype(np.int32))

        if not np.any(valid_b):
            continue

        total_roi_pixels += int(valid_b.sum())
        p_roi = preds_b.transpose(0, 2, 3, 1)[valid_b]  # (K_roi, M)
        for a in range(num_arms):
            pa = p_roi[:, a].astype(np.int32)
            for b_arm in range(a, num_arms):
                pb = p_roi[:, b_arm].astype(np.int32)
                ag = int(np.sum(pa == pb))
                pair_agree[a, b_arm] += ag
                pair_agree[b_arm, a] = pair_agree[a, b_arm]
                cm = np.bincount(pa * NUM_FOREGROUND + pb, minlength=NUM_FOREGROUND * NUM_FOREGROUND).reshape(
                    NUM_FOREGROUND, NUM_FOREGROUND
                )
                pair_conf[a, b_arm] += cm
                if b_arm != a:
                    pair_conf[b_arm, a] += cm.T

        unan_b = valid_b & np.all(preds_b == preds_b[:, :1], axis=1)
        dis_b = valid_b & ~unan_b

        if np.any(unan_b):
            pred_u = preds_b[:, 0]
            for p_local in np.unique(pidx_b):
                pm = (pidx_b == p_local)[:, None, None] & unan_b
                if not np.any(pm):
                    continue
                pu = pred_u[pm]
                gu = gt_b[pm]
                for c in range(NUM_FOREGROUND):
                    pc = pu == c
                    gc = gu == c
                    tp = int(np.logical_and(pc, gc).sum())
                    inter_u[p_local, c] += tp
                    union_u[p_local, c] += int(pc.sum() + gc.sum() - tp)

        if np.any(dis_b):
            p_grid = np.broadcast_to(pidx_b[:, None, None], gt_b.shape)
            z_dis_t = np.transpose(z_b, (0, 3, 4, 1, 2))  # (B, H, W, M, 4)
            z_dis_chunks.append(z_dis_t[dis_b].astype(np.float16))
            gt_dis_chunks.append(gt_b[dis_b].astype(np.uint8))
            pat_dis_chunks.append(p_grid[dis_b].astype(np.int32))

    z_dis = _concat_and_drain(z_dis_chunks, (num_arms, NUM_FOREGROUND), np.float16)
    gt_dis = _concat_and_drain(gt_dis_chunks, (), np.uint8)
    pat_dis = _concat_and_drain(pat_dis_chunks, (), np.int32)

    z_cal = _concat_and_drain(z_cal_chunks, (num_arms, NUM_FOREGROUND), np.float16)
    gt_cal = _concat_and_drain(gt_cal_chunks, (), np.uint8)
    pat_cal = _concat_and_drain(pat_cal_chunks, (), np.int32)

    return {
        "inter_u": inter_u,
        "union_u": union_u,
        "z_dis": z_dis,
        "gt_dis": gt_dis,
        "pat_dis": pat_dis,
        "z_cal": z_cal,
        "gt_cal": gt_cal,
        "pat_cal": pat_cal,
        "total_roi_pixels": total_roi_pixels,
        "pair_agree": pair_agree,
        "pair_conf": pair_conf,
    }


def _transform_centered_disagreement_features(
    z_dis_chunk: np.ndarray,
    betas: Sequence[float],
    prob_mode: str,
    eps_log: float = 1e-3,
) -> np.ndarray:
    """Transform (N_px, M, 4) centered float16 log-probs into (N_px, M, 4) float32 fusion inputs."""
    z_f = np.asarray(z_dis_chunk, dtype=np.float32)
    mode = str(prob_mode).strip().lower()
    if mode == "uncal_linear":
        eff_betas = np.ones((1, z_f.shape[1], 1), dtype=np.float32)
    else:
        eff_betas = np.asarray(betas, dtype=np.float32).reshape(1, -1, 1)
    logits = z_f * eff_betas
    logits = logits - np.max(logits, axis=2, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l, axis=2, keepdims=True)
    if mode == "cal_logprob_eps1e3":
        return np.log(np.clip(probs, float(eps_log), 1.0))
    return probs


def evaluate_weights_on_disagreement(
    z_dis: np.ndarray,
    gt_dis: np.ndarray,
    pat_dis: np.ndarray,
    inter_u: np.ndarray,
    union_u: np.ndarray,
    weights: np.ndarray,
    betas: Sequence[float],
    prob_mode: str = "cal_linear",
    eps_log: float = 1e-3,
    chunk_pixels: int = 2048,
    patient_subset: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate W weight vectors over disagreement pixels + unanimous base -> (W, P, 4) inter/union."""
    w_arr = np.ascontiguousarray(weights, dtype=np.float32)
    num_w, num_arms = int(w_arr.shape[0]), int(w_arr.shape[1])
    num_p = int(inter_u.shape[0])
    inter_tot = np.broadcast_to(inter_u[None, :, :], (num_w, num_p, NUM_FOREGROUND)).copy()
    union_tot = np.broadcast_to(union_u[None, :, :], (num_w, num_p, NUM_FOREGROUND)).copy()

    if z_dis.shape[0] == 0:
        return inter_tot, union_tot

    pids = list(range(num_p)) if patient_subset is None else [int(p) for p in patient_subset]
    use_cuda = device is not None and device.type == "cuda"
    if use_cuda:
        mode = str(prob_mode).strip().lower()
        w_t = torch.from_numpy(w_arr).to(device=device, dtype=torch.float32)
        if mode == "uncal_linear":
            eff_betas = torch.ones((1, num_arms, 1), device=device, dtype=torch.float32)
        else:
            eff_betas = torch.tensor(list(betas), device=device, dtype=torch.float32).view(1, num_arms, 1)
        gpu_chunk = max(int(chunk_pixels), 16384)
        for p in pids:
            pmask = pat_dis == p
            if not np.any(pmask):
                continue
            zp_all = z_dis[pmask]
            gp_all = gt_dis[pmask]
            for g_cls in range(NUM_FOREGROUND):
                gmask = gp_all == g_cls
                if not np.any(gmask):
                    continue
                zp_g = zp_all[gmask]
                n_px = zp_g.shape[0]
                for start in range(0, n_px, gpu_chunk):
                    end = min(n_px, start + gpu_chunk)
                    n_ch = end - start
                    zf = torch.from_numpy(np.ascontiguousarray(zp_g[start:end])).to(
                        device=device, dtype=torch.float32
                    )
                    logits = zf * eff_betas
                    logits = logits - logits.amax(dim=2, keepdim=True)
                    probs = torch.softmax(logits, dim=2)
                    xp = torch.log(torch.clamp(probs, min=float(eps_log))) if mode == "cal_logprob_eps1e3" else probs
                    xp_2d = xp.permute(1, 0, 2).contiguous().view(num_arms, n_ch * NUM_FOREGROUND)
                    fused = torch.mm(w_t, xp_2d).view(num_w, n_ch, NUM_FOREGROUND)
                    pred = torch.argmax(fused, dim=2)
                    counts = np.stack(
                        [(pred == c).sum(dim=1).cpu().numpy().astype(np.int64) for c in range(NUM_FOREGROUND)],
                        axis=1,
                    )
                    inter_tot[:, p, g_cls] += counts[:, g_cls]
                    union_tot[:, p, :] += counts
                    union_tot[:, p, g_cls] += n_ch - counts[:, g_cls]
        return inter_tot, union_tot

    row_offsets = (np.arange(num_w, dtype=np.int32) * NUM_FOREGROUND)[:, None]
    for p in pids:
        pmask = pat_dis == p
        if not np.any(pmask):
            continue
        zp_all = z_dis[pmask]
        gp_all = gt_dis[pmask]
        for g_cls in range(NUM_FOREGROUND):
            gmask = gp_all == g_cls
            if not np.any(gmask):
                continue
            zp_g = zp_all[gmask]
            n_px = zp_g.shape[0]
            for start in range(0, n_px, int(chunk_pixels)):
                end = min(n_px, start + int(chunk_pixels))
                n_ch = end - start
                xp = _transform_centered_disagreement_features(zp_g[start:end], betas, prob_mode, eps_log=eps_log)
                # BLAS sgemm: (W, M) @ (M, N_ch * C) -> (W, N_ch, C)
                xp_2d = np.ascontiguousarray(xp.transpose(1, 0, 2)).reshape(num_arms, n_ch * NUM_FOREGROUND)
                fused = (w_arr @ xp_2d).reshape(num_w, n_ch, NUM_FOREGROUND)
                pred = np.argmax(fused, axis=2)  # (W, N_ch)
                counts = np.bincount(
                    (row_offsets + pred.astype(np.int32)).ravel(),
                    minlength=num_w * NUM_FOREGROUND,
                ).reshape(num_w, NUM_FOREGROUND)
                inter_tot[:, p, g_cls] += counts[:, g_cls]
                union_tot[:, p, :] += counts
                union_tot[:, p, g_cls] += n_ch - counts[:, g_cls]

    return inter_tot, union_tot


def _best_simplex_index(
    inter_w: np.ndarray,
    union_w: np.ndarray,
    weights: np.ndarray,
) -> int:
    """Pick weight index maximizing macro mIoU, breaking ties toward Uniform [0.25]*4."""
    valid = union_w > 0
    ratios = np.divide(inter_w, np.where(valid, union_w, 1.0), dtype=np.float64)
    ratios = np.where(valid, ratios, 0.0)
    counts = np.maximum(1, valid.sum(axis=1))
    mious = ratios.sum(axis=1) / counts
    max_val = float(np.max(mious))
    cands = np.where(mious >= max_val - 1e-12)[0]
    if len(cands) == 1:
        return int(cands[0])
    uni = np.full((1, weights.shape[1]), 1.0 / float(weights.shape[1]), dtype=np.float32)
    dists = np.sum((weights[cands] - uni) ** 2, axis=1)
    return int(cands[int(np.argmin(dists))])


def run_val_cv_and_lopocv(
    dis_data: Mapping[str, Any],
    folds: Sequence[Sequence[int]],
    num_arms: int = 4,
    adoption_delta: float = 0.0005,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run leak-free 5-fold stratified CV across 3 prob modes + Uniform/Simplex, plus N-fold LOPOCV."""
    simplex_w = generate_simplex_grid(num_arms=num_arms, step=0.05, min_weight=0.05)
    uniform_w = np.full((1, num_arms), 1.0 / float(num_arms), dtype=np.float32)
    all_w = np.vstack([uniform_w, simplex_w])  # index 0 is Uniform, 1..969 are simplex

    inter_u = dis_data["inter_u"]
    union_u = dis_data["union_u"]
    z_dis = dis_data["z_dis"]
    gt_dis = dis_data["gt_dis"]
    pat_dis = dis_data["pat_dis"]
    z_cal = dis_data["z_cal"]
    gt_cal = dis_data["gt_cal"]
    pat_cal = dis_data["pat_cal"]
    num_patients = int(inter_u.shape[0])

    modes = ("uncal_linear", "cal_linear", "cal_logprob_eps1e3")
    mode_results: Dict[str, Any] = {}

    for mode in modes:
        oof_inter_uni = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        oof_union_uni = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        oof_inter_sim = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        oof_union_sim = np.zeros(NUM_FOREGROUND, dtype=np.int64)
        fold_weights: List[List[float]] = []
        fold_betas: List[List[float]] = []

        uncal_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if mode == "uncal_linear":
            uncal_cache = evaluate_weights_on_disagreement(
                z_dis, gt_dis, pat_dis, inter_u, union_u, all_w, [1.0] * num_arms, prob_mode=mode, device=device
            )

        for val_pats in folds:
            val_set = set(int(p) for p in val_pats)
            train_pats = [p for p in range(num_patients) if p not in val_set]
            if not train_pats:
                train_pats = list(val_pats)

            # Fit beta_m strictly on train-fold stride-8 ROI pixels
            if mode == "uncal_linear":
                betas = [1.0] * num_arms
                assert uncal_cache is not None
                inter_p, union_p = uncal_cache
                tr_inter = inter_p[1:, train_pats, :].sum(axis=1)
                tr_union = union_p[1:, train_pats, :].sum(axis=1)
                best_sim_rel = _best_simplex_index(tr_inter, tr_union, simplex_w)
                best_idx = 1 + best_sim_rel
                val_list = list(val_set)
                oof_inter_uni += inter_p[0, val_list, :].sum(axis=0)
                oof_union_uni += union_p[0, val_list, :].sum(axis=0)
                oof_inter_sim += inter_p[best_idx, val_list, :].sum(axis=0)
                oof_union_sim += union_p[best_idx, val_list, :].sum(axis=0)
            else:
                tr_mask = np.isin(pat_cal, train_pats)
                betas = [
                    fit_temperature_beta_nll(z_cal[tr_mask, m, :], gt_cal[tr_mask])
                    for m in range(num_arms)
                ]
                inter_tr, union_tr = evaluate_weights_on_disagreement(
                    z_dis,
                    gt_dis,
                    pat_dis,
                    inter_u,
                    union_u,
                    simplex_w,
                    betas,
                    prob_mode=mode,
                    patient_subset=train_pats,
                    device=device,
                )
                tr_inter = inter_tr[:, train_pats, :].sum(axis=1)
                tr_union = union_tr[:, train_pats, :].sum(axis=1)
                best_sim_rel = _best_simplex_index(tr_inter, tr_union, simplex_w)
                eval_pair = np.vstack([uniform_w, simplex_w[best_sim_rel : best_sim_rel + 1]])
                val_list = list(val_set)
                inter_val, union_val = evaluate_weights_on_disagreement(
                    z_dis,
                    gt_dis,
                    pat_dis,
                    inter_u,
                    union_u,
                    eval_pair,
                    betas,
                    prob_mode=mode,
                    patient_subset=val_list,
                    device=device,
                )
                oof_inter_uni += inter_val[0, val_list, :].sum(axis=0)
                oof_union_uni += union_val[0, val_list, :].sum(axis=0)
                oof_inter_sim += inter_val[1, val_list, :].sum(axis=0)
                oof_union_sim += union_val[1, val_list, :].sum(axis=0)

            fold_betas.append([round(b, 4) for b in betas])
            fold_weights.append([round(float(x), 4) for x in simplex_w[best_sim_rel]])

        uni_miou = compute_macro_miou_from_iu(oof_inter_uni, oof_union_uni)
        sim_miou = compute_macro_miou_from_iu(oof_inter_sim, oof_union_sim)
        bagged_w = np.mean(np.asarray(fold_weights, dtype=np.float64), axis=0)
        bagged_w = bagged_w / np.sum(bagged_w)
        mode_results[mode] = {
            "uniform_oof_miou": round(uni_miou, 6),
            "simplex_oof_miou": round(sim_miou, 6),
            "simplex_gain_over_uniform": round(sim_miou - uni_miou, 6),
            "bagged_weights": [round(float(x), 4) for x in bagged_w],
            "fold_weights": fold_weights,
            "fold_betas": fold_betas,
        }

    # Select best prob mode by max OOF VAL mIoU across (Uniform, Simplex)
    def _mode_best_score(m_name: str) -> float:
        r = mode_results[m_name]
        return max(float(r["uniform_oof_miou"]), float(r["simplex_oof_miou"]))

    best_mode = max(modes, key=_mode_best_score)
    best_r = mode_results[best_mode]
    adopt_simplex = (
        float(best_r["simplex_oof_miou"]) - float(best_r["uniform_oof_miou"])
    ) >= float(adoption_delta)
    selected_strategy = "simplex_cv" if adopt_simplex else "uniform"
    primary_weights = (
        list(best_r["bagged_weights"]) if adopt_simplex else [round(1.0 / num_arms, 4)] * num_arms
    )

    # Full-VAL betas for frozen inference
    if best_mode == "uncal_linear":
        full_val_betas = [1.0] * num_arms
    else:
        full_val_betas = [
            fit_temperature_beta_nll(z_cal[:, m, :], gt_cal) for m in range(num_arms)
        ]

    # N_val-fold LOPOCV stability check on best_mode (single precomputed (969, P, 4) pass)
    lopocv_weights: List[List[float]] = []
    lopocv_inter = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    lopocv_union = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    if num_patients >= 2:
        inter_l, union_l = evaluate_weights_on_disagreement(
            z_dis, gt_dis, pat_dis, inter_u, union_u, simplex_w, full_val_betas, prob_mode=best_mode, device=device
        )
        tot_i_all = inter_l.sum(axis=1)
        tot_u_all = union_l.sum(axis=1)
        for held_p in range(num_patients):
            tr_i = tot_i_all - inter_l[:, held_p, :]
            tr_u = tot_u_all - union_l[:, held_p, :]
            best_l = _best_simplex_index(tr_i, tr_u, simplex_w)
            lopocv_weights.append([float(x) for x in simplex_w[best_l]])
            lopocv_inter += inter_l[best_l, held_p, :]
            lopocv_union += union_l[best_l, held_p, :]
        w_mat = np.asarray(lopocv_weights, dtype=np.float64)
        lopo_mean = [round(float(x), 4) for x in np.mean(w_mat, axis=0)]
        lopo_std = [round(float(x), 4) for x in np.std(w_mat, axis=0)]
        lopo_miou = round(compute_macro_miou_from_iu(lopocv_inter, lopocv_union), 6)
    else:
        lopo_mean = list(primary_weights)
        lopo_std = [0.0] * num_arms
        lopo_miou = float(best_r["uniform_oof_miou"])

    return {
        "n_cv_folds": len(folds),
        "n_lopocv_folds": num_patients,
        "prob_mode_comparison": mode_results,
        "selected_prob_mode": best_mode,
        "selected_weight_strategy": selected_strategy,
        "primary_weights": primary_weights,
        "simplex_bagged_weights": list(best_r["bagged_weights"]),
        "uniform_weights": [round(1.0 / num_arms, 4)] * num_arms,
        "full_val_betas": [round(float(b), 4) for b in full_val_betas],
        "lopocv": {
            "weights_mean": lopo_mean,
            "weights_std": lopo_std,
            "oof_miou": lopo_miou,
        },
    }


def _eval_bitmasks_on_probs(
    fused_probs: np.ndarray,
    gt_masks: np.ndarray,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact (N, 16, 4) inter/union across all 15 non-empty bitmasks in bounded FP32 chunks."""
    n = int(fused_probs.shape[0])
    inter_arr = np.zeros((n, 16, NUM_FOREGROUND), dtype=np.int64)
    union_arr = np.zeros((n, 16, NUM_FOREGROUND), dtype=np.int64)
    neg_inf = np.float32(-np.inf)
    active_masks = [
        np.asarray([(m >> c) & 1 for c in range(NUM_FOREGROUND)], dtype=bool).reshape(1, NUM_FOREGROUND, 1, 1)
        for m in range(16)
    ]
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        fp_b = np.asarray(fused_probs[start:end], dtype=np.float32)
        gt_raw = gt_masks[start:end]
        valid_fg = (gt_raw >= 0) & (gt_raw < NUM_FOREGROUND)
        gt_b = np.where(valid_fg, gt_raw, 255).astype(np.uint8)
        for m in range(1, 16):
            masked_p = np.where(active_masks[m], fp_b, neg_inf)
            pred_m = np.argmax(masked_p, axis=1).astype(np.uint8)
            for c in range(NUM_FOREGROUND):
                pc = (pred_m == c) & valid_fg
                gc = (gt_b == c) & valid_fg
                tp = np.logical_and(pc, gc).sum(axis=(1, 2), dtype=np.int64)
                inter_arr[start:end, m, c] = tp
                union_arr[start:end, m, c] = (
                    pc.sum(axis=(1, 2), dtype=np.int64) + gc.sum(axis=(1, 2), dtype=np.int64) - tp
                )
    return inter_arr, union_arr


def _fuse_memmap_chunk(
    mms: Sequence[np.memmap],
    start: int,
    end: int,
    w_arr: np.ndarray,
    betas: Sequence[float],
    prob_mode: str,
    eps_log: float = 1e-3,
) -> np.ndarray:
    """Reconstruct and fuse one [start:end] chunk of centered float16 memmaps into FP32 (B, 4, H, W)."""
    _, c_fg, h, w = mms[0].shape
    acc = np.zeros((end - start, c_fg, h, w), dtype=np.float32)
    mode = str(prob_mode).strip().lower()
    for m, mm in enumerate(mms):
        wm = float(w_arr[m])
        if abs(wm) < 1e-12:
            continue
        feat = reconstruct_probs_from_centered(
            mm[start:end],
            beta=1.0 if mode == "uncal_linear" else float(betas[m]),
            eps_log=eps_log if mode == "cal_logprob_eps1e3" else None,
            axis=1,
        )
        acc += wm * feat
    return acc


def _eval_bitmasks_from_memmaps(
    memmap_paths: Sequence[Path],
    shape: Tuple[int, int, int, int],
    weights: Sequence[float],
    betas: Sequence[float],
    prob_mode: str,
    gt_masks: np.ndarray,
    eps_log: float = 1e-3,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stream memmaps in bounded chunks to compute exact (N, 16, 4) bitmask inter/union without full FP32."""
    n_patches = int(shape[0])
    w_arr = np.asarray(weights, dtype=np.float32)
    inter_bm = np.zeros((n_patches, 16, NUM_FOREGROUND), dtype=np.int64)
    union_bm = np.zeros((n_patches, 16, NUM_FOREGROUND), dtype=np.int64)
    mms = [np.memmap(p, dtype=np.float16, mode="r", shape=shape) for p in memmap_paths]
    try:
        for start in range(0, n_patches, int(batch_size)):
            end = min(n_patches, start + int(batch_size))
            fused_b = _fuse_memmap_chunk(mms, start, end, w_arr, betas, prob_mode, eps_log=eps_log)
            ib, ub = _eval_bitmasks_on_probs(fused_b, gt_masks[start:end], batch_size=int(batch_size))
            inter_bm[start:end] = ib
            union_bm[start:end] = ub
    finally:
        for mm in mms:
            _close_memmap(mm)
        mms.clear()
    return inter_bm, union_bm


def _combine_5_expert_probs(
    s1_probs: np.ndarray,
    s2_probs_list: Sequence[np.ndarray],
    arm_weights: Sequence[float],
    strategy: str,
    bce_betas: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Fuse 1 Stage-1 cls expert + 4 Stage-2 cls heads -> (N, 4)."""
    w = np.asarray(arm_weights, dtype=np.float64)
    w = w / np.sum(w)
    s2_stack = np.stack([np.asarray(x, dtype=np.float64) for x in s2_probs_list], axis=0)  # (4, N, 4)
    s1 = np.asarray(s1_probs, dtype=np.float64)

    if strategy == "equal_5":
        return (s1 + np.sum(s2_stack, axis=0)) / 5.0
    if strategy == "balanced_s1_s2_40_60":
        s2_w = np.tensordot(w, s2_stack, axes=([0], [0]))
        return 0.4 * s1 + 0.6 * s2_w
    if strategy == "balanced_s1_s2_50_50":
        s2_w = np.tensordot(w, s2_stack, axes=([0], [0]))
        return 0.5 * s1 + 0.5 * s2_w
    if strategy == "bce_cal_equal_5":
        experts = [s1] + [s2_stack[m] for m in range(len(s2_probs_list))]
        betas = list(bce_betas) if bce_betas is not None else [1.0] * len(experts)
        cal = []
        for ex, b in zip(experts, betas):
            ec = np.clip(ex, 1e-6, 1.0 - 1e-6)
            lg = np.log(ec / (1.0 - ec)) * float(b)
            cal.append(1.0 / (1.0 + np.exp(-lg)))
        return np.mean(np.stack(cal, axis=0), axis=0)
    raise ValueError(f"Unsupported 5-expert fusion strategy: {strategy!r}")


def _gate_masks_from_thresholds(probs_arr: np.ndarray, thr_vec: Sequence[float]) -> np.ndarray:
    """Return bitmask indices (1..15) for (N, 4) gate probs with top-1 fallback."""
    thr_np = np.asarray(thr_vec, dtype=np.float64).reshape(1, NUM_FOREGROUND)
    active = probs_arr > thr_np
    masks = (
        active[:, 0].astype(np.int32)
        | (active[:, 1].astype(np.int32) << 1)
        | (active[:, 2].astype(np.int32) << 2)
        | (active[:, 3].astype(np.int32) << 3)
    )
    top1 = np.argmax(probs_arr, axis=1)
    fallback = 1 << top1
    return np.where(masks > 0, masks, fallback)


def _run_cgm_bitmask_cv_for_strategy(
    inter_bm: np.ndarray,
    union_bm: np.ndarray,
    patch_patient_idx: np.ndarray,
    folds: Sequence[Sequence[int]],
    gate_p_fn: Callable[[Optional[np.ndarray]], Tuple[np.ndarray, Optional[List[float]]]],
    grid: Sequence[float] = ENSEMBLE_CGM_GRID,
    adoption_delta: float = 0.0005,
) -> Dict[str, Any]:
    """Run 5-fold VAL CV over {fixed_0.10, 1_dof, 4_dof} with monotonic complexity ladder."""
    n_patches = int(inter_bm.shape[0])

    def _score_subset(bm_idx: np.ndarray, patch_idx: np.ndarray) -> float:
        tot_i = inter_bm[patch_idx, bm_idx[patch_idx], :].sum(axis=0)
        tot_u = union_bm[patch_idx, bm_idx[patch_idx], :].sum(axis=0)
        return compute_macro_miou_from_iu(tot_i, tot_u)

    oof_iu: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        m: (np.zeros(NUM_FOREGROUND, dtype=np.int64), np.zeros(NUM_FOREGROUND, dtype=np.int64))
        for m in ("fixed_0.10", "1_dof", "4_dof")
    }

    for val_pats in folds:
        val_set = set(int(p) for p in val_pats)
        val_patch_mask = np.isin(patch_patient_idx, list(val_set))
        tr_patch_mask = ~val_patch_mask
        if not np.any(tr_patch_mask):
            tr_patch_mask = val_patch_mask
        tr_idx = np.where(tr_patch_mask)[0]
        val_idx = np.where(val_patch_mask)[0]

        gate_p, _ = gate_p_fn(tr_idx)

        bm_fix = _gate_masks_from_thresholds(gate_p, [0.10] * NUM_FOREGROUND)
        oof_iu["fixed_0.10"][0][:] += inter_bm[val_idx, bm_fix[val_idx], :].sum(axis=0)
        oof_iu["fixed_0.10"][1][:] += union_bm[val_idx, bm_fix[val_idx], :].sum(axis=0)

        best_t = 0.10
        best_s1 = -1.0
        for t in grid:
            bm_t = _gate_masks_from_thresholds(gate_p, [float(t)] * NUM_FOREGROUND)
            sc = _score_subset(bm_t, tr_idx)
            if sc > best_s1 + 1e-9:
                best_s1 = sc
                best_t = float(t)
        bm_1 = _gate_masks_from_thresholds(gate_p, [best_t] * NUM_FOREGROUND)
        oof_iu["1_dof"][0][:] += inter_bm[val_idx, bm_1[val_idx], :].sum(axis=0)
        oof_iu["1_dof"][1][:] += union_bm[val_idx, bm_1[val_idx], :].sum(axis=0)

        thr4 = [best_t] * NUM_FOREGROUND
        best_s4 = best_s1
        for _ in range(2):
            for c in range(NUM_FOREGROUND):
                for cand in grid:
                    trial = list(thr4)
                    trial[c] = float(cand)
                    bm_c = _gate_masks_from_thresholds(gate_p, trial)
                    sc = _score_subset(bm_c, tr_idx)
                    if sc > best_s4 + 1e-9:
                        best_s4 = sc
                        thr4 = trial
        bm_4 = _gate_masks_from_thresholds(gate_p, thr4)
        oof_iu["4_dof"][0][:] += inter_bm[val_idx, bm_4[val_idx], :].sum(axis=0)
        oof_iu["4_dof"][1][:] += union_bm[val_idx, bm_4[val_idx], :].sum(axis=0)

    oof_scores = {k: round(compute_macro_miou_from_iu(v[0], v[1]), 6) for k, v in oof_iu.items()}
    chosen_dof = "fixed_0.10"
    chosen_oof = float(oof_scores["fixed_0.10"])
    if (float(oof_scores["1_dof"]) - chosen_oof) >= float(adoption_delta):
        chosen_dof = "1_dof"
        chosen_oof = float(oof_scores["1_dof"])
    if (float(oof_scores["4_dof"]) - chosen_oof) >= float(adoption_delta):
        chosen_dof = "4_dof"
        chosen_oof = float(oof_scores["4_dof"])

    full_gate_p, full_bce_betas = gate_p_fn(None)
    all_idx = np.arange(n_patches)
    t_star = 0.10
    s_star = -1.0
    for t in grid:
        sc = _score_subset(_gate_masks_from_thresholds(full_gate_p, [float(t)] * NUM_FOREGROUND), all_idx)
        if sc > s_star + 1e-9:
            s_star = sc
            t_star = float(t)
    thr4_full = [t_star] * NUM_FOREGROUND
    s4_full = s_star
    for _ in range(2):
        for c in range(NUM_FOREGROUND):
            for cand in grid:
                trial = list(thr4_full)
                trial[c] = float(cand)
                sc = _score_subset(_gate_masks_from_thresholds(full_gate_p, trial), all_idx)
                if sc > s4_full + 1e-9:
                    s4_full = sc
                    thr4_full = trial

    return {
        "oof_miou": oof_scores,
        "chosen_dof": chosen_dof,
        "chosen_oof_miou": oof_scores[chosen_dof],
        "full_val_1dof_t": round(t_star, 4),
        "full_val_4dof_thresholds": [round(x, 4) for x in thr4_full],
        "full_val_bce_betas": [round(x, 4) for x in full_bce_betas] if full_bce_betas is not None else None,
    }


def calibrate_ensemble_cgm_cv(
    fused_probs: Optional[np.ndarray],
    gt_masks: np.ndarray,
    patch_patient_idx: np.ndarray,
    folds: Sequence[Sequence[int]],
    s1_cls_probs: np.ndarray,
    s2_cls_probs: Sequence[np.ndarray],
    y_gt_patches: np.ndarray,
    arm_weights: Sequence[float],
    grid: Sequence[float] = ENSEMBLE_CGM_GRID,
    adoption_delta: float = 0.0005,
    precomputed_bitmasks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    """5-fold VAL CV over 5-expert cls fusion x {fixed_0.10, 1_dof, 4_dof} on probs or precomputed bitmasks."""
    if precomputed_bitmasks is not None:
        inter_bm, union_bm = precomputed_bitmasks
    else:
        if fused_probs is None:
            raise ValueError("Either fused_probs or precomputed_bitmasks must be provided.")
        inter_bm, union_bm = _eval_bitmasks_on_probs(fused_probs, gt_masks)
    no_cgm_miou = compute_macro_miou_from_iu(inter_bm[:, 15, :].sum(axis=0), union_bm[:, 15, :].sum(axis=0))

    strategies = ("equal_5", "balanced_s1_s2_40_60", "balanced_s1_s2_50_50", "bce_cal_equal_5")
    strat_reports: Dict[str, Any] = {}

    for strat in strategies:
        def _gate_fn(tr_idx: Optional[np.ndarray], _strat: str = strat) -> Tuple[np.ndarray, Optional[List[float]]]:
            bce_betas: Optional[List[float]] = None
            if _strat == "bce_cal_equal_5":
                experts_all = [s1_cls_probs] + list(s2_cls_probs)
                if tr_idx is None:
                    bce_betas = [fit_binary_cls_temperature_bce(ex, y_gt_patches) for ex in experts_all]
                else:
                    bce_betas = [
                        fit_binary_cls_temperature_bce(ex[tr_idx], y_gt_patches[tr_idx]) for ex in experts_all
                    ]
            gate_p = _combine_5_expert_probs(s1_cls_probs, s2_cls_probs, arm_weights, _strat, bce_betas=bce_betas)
            return gate_p, bce_betas

        strat_reports[strat] = _run_cgm_bitmask_cv_for_strategy(
            inter_bm,
            union_bm,
            patch_patient_idx,
            folds,
            _gate_fn,
            grid=grid,
            adoption_delta=adoption_delta,
        )

    best_strat = max(strategies, key=lambda s: float(strat_reports[s]["chosen_oof_miou"]))
    best_sr = strat_reports[best_strat]
    best_cgm_oof = float(best_sr["chosen_oof_miou"])
    gate_margin = float(adoption_delta) if float(adoption_delta) > 0.0 else 1e-6
    use_cgm = (best_cgm_oof - no_cgm_miou) >= gate_margin
    best_dof = str(best_sr["chosen_dof"])
    if best_dof == "4_dof":
        selected_thresholds = list(best_sr["full_val_4dof_thresholds"])
    elif best_dof == "1_dof":
        selected_thresholds = [float(best_sr["full_val_1dof_t"])] * NUM_FOREGROUND
    else:
        selected_thresholds = [0.10] * NUM_FOREGROUND

    return {
        "no_cgm_val_miou": round(no_cgm_miou, 6),
        "strategies": strat_reports,
        "selected_expert_strategy": best_strat,
        "selected_dof": best_dof,
        "selected_thresholds": selected_thresholds,
        "selected_bce_betas": best_sr["full_val_bce_betas"],
        "use_cgm": bool(use_cgm),
    }


def _calibrate_subset_cgm_cv(
    inter_bm: np.ndarray,
    union_bm: np.ndarray,
    patch_patient_idx: np.ndarray,
    folds: Sequence[Sequence[int]],
    s1_cls_probs: np.ndarray,
    s2_cls_weighted: np.ndarray,
    grid: Sequence[float] = ENSEMBLE_CGM_GRID,
    adoption_delta: float = 0.0005,
) -> Dict[str, Any]:
    """VAL-only 5-fold CV CGM calibration over (s2_self, s1_s2_50_50) for single arm or pair."""
    no_cgm_miou = compute_macro_miou_from_iu(inter_bm[:, 15, :].sum(axis=0), union_bm[:, 15, :].sum(axis=0))
    s1_arr = np.asarray(s1_cls_probs, dtype=np.float64)
    s2_arr = np.asarray(s2_cls_weighted, dtype=np.float64)
    gate_map: Dict[str, np.ndarray] = {
        "s2_self": s2_arr,
        "s1_s2_50_50": 0.5 * s1_arr + 0.5 * s2_arr,
    }
    strat_reports: Dict[str, Any] = {}
    for g_name, g_probs in gate_map.items():
        strat_reports[g_name] = _run_cgm_bitmask_cv_for_strategy(
            inter_bm,
            union_bm,
            patch_patient_idx,
            folds,
            lambda _tr_idx, _gp=g_probs: (_gp, None),
            grid=grid,
            adoption_delta=adoption_delta,
        )

    best_source = max(("s2_self", "s1_s2_50_50"), key=lambda s: float(strat_reports[s]["chosen_oof_miou"]))
    best_sr = strat_reports[best_source]
    chosen_oof = float(best_sr["chosen_oof_miou"])
    gate_margin = float(adoption_delta) if float(adoption_delta) > 0.0 else 1e-6
    use_cgm = (chosen_oof - no_cgm_miou) >= gate_margin
    best_dof = str(best_sr["chosen_dof"])
    if best_dof == "4_dof":
        selected_thresholds = list(best_sr["full_val_4dof_thresholds"])
    elif best_dof == "1_dof":
        selected_thresholds = [float(best_sr["full_val_1dof_t"])] * NUM_FOREGROUND
    else:
        selected_thresholds = [0.10] * NUM_FOREGROUND

    effective_val_miou = chosen_oof if use_cgm else no_cgm_miou
    return {
        "no_cgm_val_miou": round(no_cgm_miou, 6),
        "selected_gate_source": best_source,
        "selected_dof": best_dof,
        "chosen_oof_miou": round(chosen_oof, 6),
        "effective_val_miou": round(effective_val_miou, 6),
        "selected_thresholds": selected_thresholds,
        "use_cgm": bool(use_cgm),
        "strategies": strat_reports,
    }


def _apply_cgm_probs_np(
    probs: np.ndarray,
    gate_probs: np.ndarray,
    thresholds: Union[float, Sequence[float]],
) -> np.ndarray:
    """Apply CGM gating to (N, 4, H, W) probs -> (N, H, W) uint8 predicted class labels in FP32."""
    probs_f = np.asarray(probs, dtype=np.float32)
    thr_vec = [float(thresholds)] * NUM_FOREGROUND if isinstance(thresholds, (int, float)) else thresholds
    bm = _gate_masks_from_thresholds(gate_probs, thr_vec)
    preds = np.empty((probs_f.shape[0], probs_f.shape[2], probs_f.shape[3]), dtype=np.uint8)
    neg_inf = np.float32(-np.inf)
    for m in np.unique(bm):
        m_int = int(m)
        idx_m = bm == m_int
        active = np.asarray([(m_int >> c) & 1 for c in range(NUM_FOREGROUND)], dtype=bool).reshape(
            1, NUM_FOREGROUND, 1, 1
        )
        preds[idx_m] = np.argmax(np.where(active, probs_f[idx_m], neg_inf), axis=1).astype(np.uint8)
    return preds


def _accumulate_patient_iu_into(
    preds_b: np.ndarray,
    gt_b: np.ndarray,
    pidx_b: np.ndarray,
    inter_p: np.ndarray,
    union_p: np.ndarray,
) -> None:
    """Accumulate (B, H, W) predictions into existing (num_patients, 4) integer inter/union arrays."""
    valid_fg = (gt_b >= 0) & (gt_b < NUM_FOREGROUND)
    for p in np.unique(pidx_b):
        p_int = int(p)
        pm = pidx_b == p_int
        pr_p = preds_b[pm]
        gt_p = gt_b[pm]
        vf_p = valid_fg[pm]
        for c in range(NUM_FOREGROUND):
            pc = (pr_p == c) & vf_p
            gc = (gt_p == c) & vf_p
            tp = int(np.logical_and(pc, gc).sum())
            inter_p[p_int, c] += tp
            union_p[p_int, c] += int(pc.sum() + gc.sum() - tp)


def _accumulate_patient_iu(
    preds: np.ndarray,
    gt_masks: np.ndarray,
    patch_patient_idx: np.ndarray,
    num_patients: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate (num_patients, 4) integer inter and union from (N, H, W) predictions."""
    inter_p = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
    union_p = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
    _accumulate_patient_iu_into(preds, gt_masks, patch_patient_idx, inter_p, union_p)
    return inter_p, union_p


def _eval_cgm_from_memmaps(
    memmap_paths: Sequence[Path],
    shape: Tuple[int, int, int, int],
    w_vec: Sequence[float],
    betas: Sequence[float],
    prob_mode: str,
    gt_masks: np.ndarray,
    patch_patient_idx: np.ndarray,
    num_patients: int,
    gate_probs: np.ndarray,
    threshold_specs: Mapping[str, Optional[Union[float, Sequence[float]]]],
    eps_log: float = 1e-3,
    batch_size: int = 128,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Stream memmaps in bounded chunks to evaluate one or more CGM threshold specs with zero full FP32."""
    n_patches = int(shape[0])
    w_arr = np.asarray(w_vec, dtype=np.float32)
    out_iu: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        k: (
            np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64),
            np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64),
        )
        for k in threshold_specs
    }
    mms = [np.memmap(p, dtype=np.float16, mode="r", shape=shape) for p in memmap_paths]
    try:
        for start in range(0, n_patches, int(batch_size)):
            end = min(n_patches, start + int(batch_size))
            fused_b = _fuse_memmap_chunk(mms, start, end, w_arr, betas, prob_mode, eps_log=eps_log)
            gt_b = gt_masks[start:end]
            pidx_b = patch_patient_idx[start:end]
            gp_b = gate_probs[start:end]
            for mode_key, thr_spec in threshold_specs.items():
                if thr_spec is None:
                    pred_b = np.argmax(fused_b, axis=1).astype(np.uint8)
                else:
                    pred_b = _apply_cgm_probs_np(fused_b, gp_b, thr_spec)
                _accumulate_patient_iu_into(pred_b, gt_b, pidx_b, out_iu[mode_key][0], out_iu[mode_key][1])
    finally:
        for mm in mms:
            _close_memmap(mm)
        mms.clear()
    return out_iu


def bootstrap_pooled_miou_ci(
    inter_p: np.ndarray,
    union_p: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute exact patient-level pooled bootstrap 95% CI with valid_c = tot_U > 0 guard."""
    inter_a = np.asarray(inter_p, dtype=np.float64)
    union_a = np.asarray(union_p, dtype=np.float64)
    n_pat = int(inter_a.shape[0])
    point = compute_macro_miou_from_iu(inter_a.sum(axis=0), union_a.sum(axis=0))
    if n_pat == 0:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n_pat, size=(int(n_boot), n_pat))
    b_inter = inter_a[idx].sum(axis=1)  # (B, 4)
    b_union = union_a[idx].sum(axis=1)  # (B, 4)
    valid = b_union > 0
    ratios = np.divide(b_inter, np.where(valid, b_union, 1.0))
    ratios = np.where(valid, ratios, 0.0)
    counts = valid.sum(axis=1)
    boot_mious = np.where(counts > 0, ratios.sum(axis=1) / np.maximum(1, counts), 0.0)
    lo, hi = np.percentile(boot_mious, [2.5, 97.5])
    return {
        "point": round(float(point), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
    }


def paired_patient_swap_permutation_test(
    inter_a: np.ndarray,
    union_a: np.ndarray,
    inter_b: np.ndarray,
    union_b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Exact paired patient-swap pooled permutation test on global Delta_mIoU."""
    ia = np.asarray(inter_a, dtype=np.float64)
    ua = np.asarray(union_a, dtype=np.float64)
    ib = np.asarray(inter_b, dtype=np.float64)
    ub = np.asarray(union_b, dtype=np.float64)
    miou_a = compute_macro_miou_from_iu(ia.sum(axis=0), ua.sum(axis=0))
    miou_b = compute_macro_miou_from_iu(ib.sum(axis=0), ub.sum(axis=0))
    delta_obs = miou_a - miou_b
    n_pat = int(ia.shape[0])
    if n_pat == 0:
        return {"delta_miou": 0.0, "p_value": 1.0}

    rng = np.random.default_rng(int(seed))
    swaps = rng.integers(0, 2, size=(int(n_perm), n_pat), dtype=np.int8)  # 1 => swap A and B for patient p
    ia_p = np.where(swaps[:, :, None] == 1, ib[None, :, :], ia[None, :, :]).sum(axis=1)
    ua_p = np.where(swaps[:, :, None] == 1, ub[None, :, :], ua[None, :, :]).sum(axis=1)
    ib_p = np.where(swaps[:, :, None] == 1, ia[None, :, :], ib[None, :, :]).sum(axis=1)
    ub_p = np.where(swaps[:, :, None] == 1, ua[None, :, :], ub[None, :, :]).sum(axis=1)

    def _batch_miou(bi: np.ndarray, bu: np.ndarray) -> np.ndarray:
        val = bu > 0
        rat = np.where(val, np.divide(bi, np.where(val, bu, 1.0)), 0.0)
        cnt = val.sum(axis=1)
        return np.where(cnt > 0, rat.sum(axis=1) / np.maximum(1, cnt), 0.0)

    deltas = _batch_miou(ia_p, ua_p) - _batch_miou(ib_p, ub_p)
    ge_count = int(np.sum(np.abs(deltas) >= abs(delta_obs) - 1e-12))
    p_val = float(ge_count + 1) / float(int(n_perm) + 1)
    return {"delta_miou": round(float(delta_obs), 4), "p_value": float(p_val)}


def paired_wilcoxon_and_dz(
    per_patient_a: np.ndarray,
    per_patient_b: np.ndarray,
) -> Dict[str, Any]:
    """Two-sided Wilcoxon signed-rank (all-zero diffs guarded) + Cohen's dz + win/tie/loss."""
    a = np.asarray(per_patient_a, dtype=np.float64)
    b = np.asarray(per_patient_b, dtype=np.float64)
    diffs = a - b
    wins = int(np.sum(diffs > 1e-12))
    losses = int(np.sum(diffs < -1e-12))
    ties = int(len(diffs) - wins - losses)
    if len(diffs) > 1:
        sd = float(np.std(diffs, ddof=1))
        dz = float(np.mean(diffs) / sd) if sd > 1e-12 else 0.0
    else:
        dz = 0.0
    if len(diffs) == 0 or np.allclose(diffs, 0.0, atol=1e-12):
        p_val = 1.0
    else:
        p_val = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
    return {
        "wilcoxon_p": float(p_val),
        "cohens_dz": round(float(dz), 4),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Step-down Holm-Bonferroni adjusted p-values preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda kv: kv[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        corrected = min(1.0, (m - rank) * p_val)
        running_max = max(running_max, corrected)
        adj[orig_idx] = float(running_max)
    return adj


def _cohens_kappa_from_conf(conf: np.ndarray) -> float:
    """Compute Cohen's kappa from (4, 4) confusion matrix."""
    total = float(np.sum(conf))
    if total <= 0:
        return 1.0
    po = float(np.trace(conf)) / total
    pe = float(np.sum(conf.sum(axis=0) * conf.sum(axis=1))) / (total * total)
    denom = 1.0 - pe
    if denom <= 1e-12:
        return 1.0
    return float((po - pe) / denom)


def _compute_diversity_and_oracles(
    dis_data: Mapping[str, Any],
    z_memmaps: Sequence[np.ndarray],
    gt_masks: np.ndarray,
    patch_patient_idx: np.ndarray,
    num_patients: int,
    arm_names: Sequence[str],
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Compute 4x4 disagreement & Cohen's kappa matrix + pixel/patch oracle upper bounds."""
    num_arms = len(arm_names)
    total_roi = max(1, int(dis_data["total_roi_pixels"]))
    pair_agree = dis_data["pair_agree"]
    pair_conf = dis_data["pair_conf"]

    disagreement_matrix = [
        [round(1.0 - float(pair_agree[i, j]) / float(total_roi), 4) for j in range(num_arms)]
        for i in range(num_arms)
    ]
    kappa_matrix = [
        [round(_cohens_kappa_from_conf(pair_conf[i, j]), 4) for j in range(num_arms)]
        for i in range(num_arms)
    ]

    # Pixel oracle & Patch oracle upper bounds
    pixel_oracle_correct = 0
    patch_oracle_inter = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    patch_oracle_union = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    n_patches = int(gt_masks.shape[0])

    for start in range(0, n_patches, int(batch_size)):
        end = min(n_patches, start + int(batch_size))
        gt_raw = gt_masks[start:end]
        valid_b = (gt_raw >= 0) & (gt_raw < NUM_FOREGROUND)
        gt_b = np.where(valid_b, gt_raw, 255).astype(np.uint8)
        z_b = np.stack([np.asarray(mm[start:end], dtype=np.float16) for mm in z_memmaps], axis=1)
        preds_b = np.argmax(z_b, axis=2).astype(np.uint8)  # (B, M, H, W)
        any_correct = np.any(preds_b == gt_b[:, None, :, :], axis=1) & valid_b
        pixel_oracle_correct += int(any_correct.sum())

        for bi in range(end - start):
            vf = valid_b[bi]
            if not np.any(vf):
                continue
            gt_i = gt_b[bi]
            best_m = 0
            best_miou = -1.0
            best_iu: Optional[Tuple[np.ndarray, np.ndarray]] = None
            for m in range(num_arms):
                pr_i = preds_b[bi, m]
                inter_m = np.zeros(NUM_FOREGROUND, dtype=np.int64)
                union_m = np.zeros(NUM_FOREGROUND, dtype=np.int64)
                for c in range(NUM_FOREGROUND):
                    pc = (pr_i == c) & vf
                    gc = (gt_i == c) & vf
                    tp = int(np.logical_and(pc, gc).sum())
                    inter_m[c] = tp
                    union_m[c] = int(pc.sum() + gc.sum() - tp)
                sc = compute_macro_miou_from_iu(inter_m, union_m)
                if sc > best_miou:
                    best_miou = sc
                    best_m = m
                    best_iu = (inter_m, union_m)
            _ = best_m
            if best_iu is not None:
                patch_oracle_inter += best_iu[0]
                patch_oracle_union += best_iu[1]

    return {
        "arms": list(arm_names),
        "pairwise_disagreement_rate": disagreement_matrix,
        "pairwise_cohens_kappa": kappa_matrix,
        "pixel_oracle_accuracy": round(float(pixel_oracle_correct) / float(total_roi), 4),
        "patch_oracle_upper_bound_miou": round(
            compute_macro_miou_from_iu(patch_oracle_inter, patch_oracle_union), 4
        ),
    }


def _stream_test_diversity_and_no_cgm_weights(
    memmap_paths: Sequence[Path],
    shape: Tuple[int, int, int, int],
    gt_masks: np.ndarray,
    patch_patient_idx: np.ndarray,
    num_patients: int,
    arm_names: Sequence[str],
    weight_matrix: np.ndarray,
    betas: Sequence[float],
    prob_mode: str,
    eps_log: float = 1e-3,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Single-pass chunked TEST stream: diversity/oracles + all no-CGM weight vectors without storing test_dis."""
    n_patches = int(gt_masks.shape[0])
    num_arms = len(arm_names)
    num_w = int(weight_matrix.shape[0])

    total_roi_pixels = 0
    pair_agree = np.zeros((num_arms, num_arms), dtype=np.int64)
    pair_conf = np.zeros((num_arms, num_arms, NUM_FOREGROUND, NUM_FOREGROUND), dtype=np.int64)
    pixel_oracle_correct = 0
    patch_oracle_inter = np.zeros(NUM_FOREGROUND, dtype=np.int64)
    patch_oracle_union = np.zeros(NUM_FOREGROUND, dtype=np.int64)

    inter_tot = np.zeros((num_w, num_patients, NUM_FOREGROUND), dtype=np.int64)
    union_tot = np.zeros((num_w, num_patients, NUM_FOREGROUND), dtype=np.int64)

    mms = [np.memmap(p, dtype=np.float16, mode="r", shape=shape) for p in memmap_paths]
    try:
        for start in range(0, n_patches, int(batch_size)):
            end = min(n_patches, start + int(batch_size))
            gt_raw = gt_masks[start:end]
            valid_b = (gt_raw >= 0) & (gt_raw < NUM_FOREGROUND)
            if not np.any(valid_b):
                continue
            gt_b = np.where(valid_b, gt_raw, 255).astype(np.uint8)
            pidx_b = patch_patient_idx[start:end]

            z_b = np.stack([np.asarray(mm[start:end], dtype=np.float16) for mm in mms], axis=1)
            preds_b = np.argmax(z_b, axis=2).astype(np.uint8)

            total_roi_pixels += int(valid_b.sum())
            p_roi = preds_b.transpose(0, 2, 3, 1)[valid_b]
            for a in range(num_arms):
                pa = p_roi[:, a].astype(np.int32)
                for b_arm in range(a, num_arms):
                    pb = p_roi[:, b_arm].astype(np.int32)
                    ag = int(np.sum(pa == pb))
                    pair_agree[a, b_arm] += ag
                    pair_agree[b_arm, a] = pair_agree[a, b_arm]
                    cm = np.bincount(
                        pa * NUM_FOREGROUND + pb,
                        minlength=NUM_FOREGROUND * NUM_FOREGROUND,
                    ).reshape(NUM_FOREGROUND, NUM_FOREGROUND)
                    pair_conf[a, b_arm] += cm
                    if b_arm != a:
                        pair_conf[b_arm, a] += cm.T

            any_correct = np.any(preds_b == gt_b[:, None, :, :], axis=1) & valid_b
            pixel_oracle_correct += int(any_correct.sum())

            for bi in range(end - start):
                vf = valid_b[bi]
                if not np.any(vf):
                    continue
                gt_i = gt_b[bi]
                best_miou = -1.0
                best_iu: Optional[Tuple[np.ndarray, np.ndarray]] = None
                for m in range(num_arms):
                    pr_i = preds_b[bi, m]
                    inter_m = np.zeros(NUM_FOREGROUND, dtype=np.int64)
                    union_m = np.zeros(NUM_FOREGROUND, dtype=np.int64)
                    for c in range(NUM_FOREGROUND):
                        pc = (pr_i == c) & vf
                        gc = (gt_i == c) & vf
                        tp = int(np.logical_and(pc, gc).sum())
                        inter_m[c] = tp
                        union_m[c] = int(pc.sum() + gc.sum() - tp)
                    sc = compute_macro_miou_from_iu(inter_m, union_m)
                    if sc > best_miou:
                        best_miou = sc
                        best_iu = (inter_m, union_m)
                if best_iu is not None:
                    patch_oracle_inter += best_iu[0]
                    patch_oracle_union += best_iu[1]

            unan_b = valid_b & np.all(preds_b == preds_b[:, :1], axis=1)
            dis_b = valid_b & ~unan_b
            inter_u_b = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
            union_u_b = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
            if np.any(unan_b):
                pred_u = preds_b[:, 0]
                for p_local in np.unique(pidx_b):
                    p_int = int(p_local)
                    pm = (pidx_b == p_int)[:, None, None] & unan_b
                    if not np.any(pm):
                        continue
                    pu = pred_u[pm]
                    gu = gt_b[pm]
                    for c in range(NUM_FOREGROUND):
                        pc = pu == c
                        gc = gu == c
                        tp = int(np.logical_and(pc, gc).sum())
                        inter_u_b[p_int, c] += tp
                        union_u_b[p_int, c] += int(pc.sum() + gc.sum() - tp)

            if np.any(dis_b):
                p_grid = np.broadcast_to(pidx_b[:, None, None], gt_b.shape)
                z_dis_b = np.transpose(z_b, (0, 3, 4, 1, 2))[dis_b].astype(np.float16)
                gt_dis_b = gt_b[dis_b].astype(np.uint8)
                pat_dis_b = p_grid[dis_b].astype(np.int32)
                pat_sub = [int(x) for x in np.unique(pat_dis_b)]
            else:
                z_dis_b = np.zeros((0, num_arms, NUM_FOREGROUND), dtype=np.float16)
                gt_dis_b = np.zeros((0,), dtype=np.uint8)
                pat_dis_b = np.zeros((0,), dtype=np.int32)
                pat_sub = []

            inter_c, union_c = evaluate_weights_on_disagreement(
                z_dis_b,
                gt_dis_b,
                pat_dis_b,
                inter_u_b,
                union_u_b,
                weight_matrix,
                betas,
                prob_mode=prob_mode,
                eps_log=eps_log,
                patient_subset=pat_sub,
                device=device,
            )
            inter_tot += inter_c
            union_tot += union_c
    finally:
        for mm in mms:
            _close_memmap(mm)
        mms.clear()

    total_roi = max(1, int(total_roi_pixels))
    disagreement_matrix = [
        [round(1.0 - float(pair_agree[i, j]) / float(total_roi), 4) for j in range(num_arms)]
        for i in range(num_arms)
    ]
    kappa_matrix = [
        [round(_cohens_kappa_from_conf(pair_conf[i, j]), 4) for j in range(num_arms)]
        for i in range(num_arms)
    ]
    diversity = {
        "arms": list(arm_names),
        "pairwise_disagreement_rate": disagreement_matrix,
        "pairwise_cohens_kappa": kappa_matrix,
        "pixel_oracle_accuracy": round(float(pixel_oracle_correct) / float(total_roi), 4),
        "patch_oracle_upper_bound_miou": round(
            compute_macro_miou_from_iu(patch_oracle_inter, patch_oracle_union), 4
        ),
    }
    return diversity, inter_tot, union_tot


def _evaluate_author_anchors_shared(
    author_stage2_ckpt: Union[str, Path],
    stage1_ckpts: Mapping[str, Union[str, Path]],
    loader: DataLoader,
    patch_patient_idx: np.ndarray,
    num_patients: int,
    device: torch.device,
    scales: Sequence[float] = (0.75, 1.0, 1.25),
) -> Dict[str, Dict[str, Any]]:
    """Evaluate official Author DeepLabV3+ Stage-2 once (1x + 6-view MS-TTA) across all Stage-1 experts."""
    author_m = build_and_load_deeplabv3plus(checkpoint_path=author_stage2_ckpt, device=device, strict=True)
    author_sha = compute_file_sha256(author_stage2_ckpt)

    s1_experts: Dict[str, Tuple[WSSSTissueStage1NetCAM, str, str]] = {}
    for anchor_key, ckpt_path in stage1_ckpts.items():
        s1_m = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
        s1_norm, s1_sha, _ = load_stage1_strict(s1_m, ckpt_path)
        s1_m = s1_m.to(device).eval()
        s1_experts[str(anchor_key)] = (s1_m, str(s1_norm), str(s1_sha))

    inter_raw = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
    union_raw = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
    inter_ms = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)
    union_ms = np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64)

    anchor_iu: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        ak: {
            mode_k: (
                np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64),
                np.zeros((num_patients, NUM_FOREGROUND), dtype=np.int64),
            )
            for mode_k in ("cgm_author_mul0", "cgm_neg1e4", "ms_tta_cgm_neg1e4")
        }
        for ak in s1_experts
    }

    offset = 0
    with torch.no_grad():
        for batch in loader:
            imgs, gts, _ = _extract_img_gt(batch)
            b = imgs.size(0)
            imgs = imgs.to(device, non_blocking=True)
            gt_np = gts.cpu().numpy()
            pidx_b = patch_patient_idx[offset : offset + b]
            offset += b

            author_inp = prepare_stage1_input(imgs, stage1_norm_mode="zero_one", images_are_imagenet_norm=True)
            logits_1x = author_m(author_inp)[:, :NUM_FOREGROUND].float()
            probs_ms = _ms_tta_probs(author_m, author_inp, logits_1x, scales, device, amp_dtype=None)

            pred_raw = torch.argmax(logits_1x, dim=1).cpu().numpy().astype(np.uint8)
            pred_ms = torch.argmax(probs_ms, dim=1).cpu().numpy().astype(np.uint8)
            _accumulate_patient_iu_into(pred_raw, gt_np, pidx_b, inter_raw, union_raw)
            _accumulate_patient_iu_into(pred_ms, gt_np, pidx_b, inter_ms, union_ms)

            s1_inputs: Dict[str, torch.Tensor] = {"zero_one": author_inp}
            for ak, (s1_m, s1_norm, _) in s1_experts.items():
                if s1_norm not in s1_inputs:
                    s1_inputs[s1_norm] = prepare_stage1_input(
                        imgs, stage1_norm_mode=s1_norm, images_are_imagenet_norm=True
                    )
                _, s1_probs = s1_m.forward_cam(s1_inputs[s1_norm])
                raw_gate = (s1_probs > 0.10).float().view(-1, NUM_FOREGROUND, 1, 1)
                pred_mul0 = torch.argmax(logits_1x * raw_gate, dim=1).cpu().numpy().astype(np.uint8)
                gate = _make_cgm_gate(s1_probs, 0.10)
                pred_neg = _apply_logit_cgm(logits_1x, gate).astype(np.uint8)
                pred_ms_neg = _apply_prob_cgm(probs_ms, gate).astype(np.uint8)

                _accumulate_patient_iu_into(
                    pred_mul0, gt_np, pidx_b, *anchor_iu[ak]["cgm_author_mul0"]
                )
                _accumulate_patient_iu_into(
                    pred_neg, gt_np, pidx_b, *anchor_iu[ak]["cgm_neg1e4"]
                )
                _accumulate_patient_iu_into(
                    pred_ms_neg, gt_np, pidx_b, *anchor_iu[ak]["ms_tta_cgm_neg1e4"]
                )

    s1_shas = {ak: vals[2] for ak, vals in s1_experts.items()}
    del author_m, s1_experts
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_entry = {**summarize_iu(inter_raw.sum(axis=0), union_raw.sum(axis=0)), "inter_p": inter_raw, "union_p": union_raw}
    ms_entry = {**summarize_iu(inter_ms.sum(axis=0), union_ms.sum(axis=0)), "inter_p": inter_ms, "union_p": union_ms}

    out: Dict[str, Dict[str, Any]] = {}
    for ak, iu_modes in anchor_iu.items():
        # Copy shared baseline arrays/dicts per anchor so mutating/stripping one anchor cannot affect another
        out[ak] = {
            "author_stage2_sha256": author_sha,
            "stage1_sha256": s1_shas[ak],
            "raw": {**raw_entry, "inter_p": inter_raw.copy(), "union_p": union_raw.copy()},
            "ms_tta": {**ms_entry, "inter_p": inter_ms.copy(), "union_p": union_ms.copy()},
            "cgm_author_mul0": {
                **summarize_iu(iu_modes["cgm_author_mul0"][0].sum(axis=0), iu_modes["cgm_author_mul0"][1].sum(axis=0)),
                "inter_p": iu_modes["cgm_author_mul0"][0],
                "union_p": iu_modes["cgm_author_mul0"][1],
            },
            "cgm_neg1e4": {
                **summarize_iu(iu_modes["cgm_neg1e4"][0].sum(axis=0), iu_modes["cgm_neg1e4"][1].sum(axis=0)),
                "inter_p": iu_modes["cgm_neg1e4"][0],
                "union_p": iu_modes["cgm_neg1e4"][1],
            },
            "ms_tta_cgm_neg1e4": {
                **summarize_iu(
                    iu_modes["ms_tta_cgm_neg1e4"][0].sum(axis=0),
                    iu_modes["ms_tta_cgm_neg1e4"][1].sum(axis=0),
                ),
                "inter_p": iu_modes["ms_tta_cgm_neg1e4"][0],
                "union_p": iu_modes["ms_tta_cgm_neg1e4"][1],
            },
        }
    return out


def _evaluate_author_anchor(
    author_stage2_ckpt: Union[str, Path],
    stage1_ckpt: Union[str, Path],
    loader: DataLoader,
    patch_patient_idx: np.ndarray,
    num_patients: int,
    device: torch.device,
    scales: Sequence[float] = (0.75, 1.0, 1.25),
) -> Dict[str, Any]:
    """Evaluate official Author DeepLabV3+ Stage-2 anchor (1x + 6-view MS-TTA) with Stage-1 CGM."""
    return _evaluate_author_anchors_shared(
        author_stage2_ckpt=author_stage2_ckpt,
        stage1_ckpts={"anchor": stage1_ckpt},
        loader=loader,
        patch_patient_idx=patch_patient_idx,
        num_patients=num_patients,
        device=device,
        scales=scales,
    )["anchor"]


def save_ensemble_comparison_plot(
    entries: Sequence[Tuple[str, float, float, float]],
    plot_path: Union[str, Path],
) -> Path:
    """Save 300 DPI Academic Monochrome comparison bar chart with 95% bootstrap CIs (INVARIANT-24)."""
    out_p = Path(plot_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        return out_p

    labels = [e[0] for e in entries]
    pts = np.asarray([e[1] for e in entries], dtype=np.float64)
    err_lo = np.maximum(0.0, pts - np.asarray([e[2] for e in entries], dtype=np.float64))
    err_hi = np.maximum(0.0, np.asarray([e[3] for e in entries], dtype=np.float64) - pts)

    shades = ["#B8B8B8", "#909090", "#707070", "#555555", "#383838", "#000000"]
    hatches = ["", "//", "\\\\", "xx", "..", ""]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    x = np.arange(n)
    for i in range(n):
        sh = shades[min(i, len(shades) - 1)]
        ht = hatches[min(i, len(hatches) - 1)]
        ax.bar(
            x[i],
            pts[i],
            width=0.58,
            color=sh,
            edgecolor="#000000",
            linewidth=0.9,
            hatch=ht,
            zorder=3,
        )
    ax.errorbar(
        x,
        pts,
        yerr=np.vstack([err_lo, err_hi]),
        fmt="none",
        ecolor="#000000",
        elinewidth=1.2,
        capsize=4,
        zorder=4,
    )
    for i, v in enumerate(pts):
        ax.text(
            x[i],
            pts[i] + err_hi[i] + 0.004,
            f"{v * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#000000",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5, color="#000000")
    ax.set_ylabel("4-Class Pooled Macro mIoU", color="#000000")
    ax.set_title("BCSS-WSSS Stage-2 v2 Test Comparison (95% Patient Bootstrap CI)", color="#000000")
    ymin = max(0.0, float(np.min(pts - err_lo)) - 0.04)
    ymax = min(1.0, float(np.max(pts + err_hi)) + 0.03)
    if ymax > ymin:
        ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="#B0B0B0", alpha=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#000000")

    fig.tight_layout()
    fig.savefig(out_p, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)
    return out_p


def _extract_patient_ids_from_loader(loader: DataLoader) -> List[str]:
    """Extract canonical patient IDs aligned 1-to-1 with DataLoader dataset files."""
    ds = loader.dataset
    files = getattr(ds, "files", None)
    if files is None:
        raise RuntimeError("Dataset must expose .files for patient-level anti-leakage verification.")
    return [extract_patient_id(Path(f).name) for f in files]


def _stream_split_models(
    arm_specs: Sequence[Dict[str, Any]],
    loader: DataLoader,
    arm_tta_modes: Sequence[str],
    device: torch.device,
    memmap_dir: Path,
    split_tag: str,
    amp_dtype: Optional[torch.dtype] = None,
    select_tta_on_split: bool = False,
    include_d4_cls: bool = False,
) -> Dict[str, Any]:
    """Stream 1 arm at a time on GPU -> centered float16 log-prob memmaps + cls probs + gt/y_gt."""
    n_patches = len(loader.dataset)  # type: ignore[arg-type]
    gt_all: Optional[np.ndarray] = None
    h = 0
    w = 0
    y_gt_all = np.zeros((n_patches, NUM_FOREGROUND), dtype=np.float32)
    nec_pixels = np.zeros(n_patches, dtype=np.int64)

    offset = 0
    for batch in loader:
        _, gts_b, _ = _extract_img_gt(batch)
        b = int(gts_b.shape[0])
        if gt_all is None:
            h, w = int(gts_b.shape[-2]), int(gts_b.shape[-1])
            gt_all = np.empty((n_patches, h, w), dtype=np.uint8)
        g_np = gts_b.cpu().numpy()
        gt_all[offset : offset + b] = np.clip(g_np, 0, 255).astype(np.uint8)
        for c in range(NUM_FOREGROUND):
            y_gt_all[offset : offset + b, c] = np.any(g_np == c, axis=(1, 2)).astype(np.float32)
        nec_pixels[offset : offset + b] = np.sum(g_np == 3, axis=(1, 2))
        offset += b

    if gt_all is None:
        raise RuntimeError("DataLoader yielded 0 batches; cannot infer spatial dimensions.")

    memmap_paths: List[Path] = []
    selected_ttas: List[str] = list(arm_tta_modes)
    tta_val_tables: Dict[str, Any] = {}
    cls_1x_list: List[np.ndarray] = []
    cls_d4_list: List[np.ndarray] = []
    cls_tta_list: List[np.ndarray] = []
    arm_metas: List[Dict[str, Any]] = []

    for m_idx, spec in enumerate(arm_specs):
        arm_name = str(spec["name"])
        mm_path = memmap_dir / f"{split_tag}_{m_idx}_{arm_name}.dat"
        mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(n_patches, NUM_FOREGROUND, h, w))

        model_obj: Optional[BaseStage2DualHead] = spec.get("model")
        loaded_here = False
        if model_obj is None:
            model_obj, ck_meta = load_stage2_v2_checkpoint(spec["ckpt_path"], device=device, strict=True)
            ck_meta["ckpt_sha256"] = compute_file_sha256(spec["ckpt_path"])
            loaded_here = True
        else:
            model_obj = model_obj.to(device).eval()
            ck_meta = model_obj.model_metadata()
            ck_meta["ckpt_sha256"] = ""
        arm_metas.append({"name": arm_name, "family": spec.get("family", "unknown"), **ck_meta})

        cls_1x_arr = np.zeros((n_patches, NUM_FOREGROUND), dtype=np.float32)
        cls_d4_arr = np.zeros((n_patches, NUM_FOREGROUND), dtype=np.float32)
        cls_tta_arr = np.zeros((n_patches, NUM_FOREGROUND), dtype=np.float32)

        try:
            if select_tta_on_split:
                cand_inter = {t: np.zeros(NUM_FOREGROUND, dtype=np.int64) for t in TTA_CANDIDATES}
                cand_union = {t: np.zeros(NUM_FOREGROUND, dtype=np.int64) for t in TTA_CANDIDATES}
                cand_cls: Dict[str, List[np.ndarray]] = {t: [] for t in TTA_CANDIDATES}
                # Temporary memmaps per candidate so we select winning TTA without running twice
                cand_mms = {
                    t: np.memmap(
                        memmap_dir / f"tmp_{split_tag}_{m_idx}_{t}.dat",
                        dtype=np.float16,
                        mode="w+",
                        shape=(n_patches, NUM_FOREGROUND, h, w),
                    )
                    for t in TTA_CANDIDATES
                }
                try:
                    pos = 0
                    with torch.no_grad():
                        for batch in loader:
                            imgs_b, gts_b, _ = _extract_img_gt(batch)
                            b = imgs_b.size(0)
                            imgs_b = imgs_b.to(device, non_blocking=True)
                            gt_np = gts_b.cpu().numpy()
                            vf = (gt_np >= 0) & (gt_np < NUM_FOREGROUND)
                            seg_dict, cls_dict = _collect_all_tta_candidates_batch(
                                model_obj, imgs_b, device=device, amp_dtype=amp_dtype
                            )
                            for t in TTA_CANDIDATES:
                                p_np = seg_dict[t].cpu().numpy()
                                cand_mms[t][pos : pos + b] = encode_centered_logprobs_fp16(p_np)
                                cand_cls[t].append(cls_dict[t].cpu().numpy())
                                pred_t = np.argmax(p_np, axis=1).astype(np.uint8)
                                for c in range(NUM_FOREGROUND):
                                    pc = (pred_t == c) & vf
                                    gt_c = (gt_np == c) & vf
                                    tp = int(np.logical_and(pc, gt_c).sum())
                                    cand_inter[t][c] += tp
                                    cand_union[t][c] += int(pc.sum() + gt_c.sum() - tp)
                            pos += b

                    cand_scores = {
                        t: compute_macro_miou_from_iu(cand_inter[t], cand_union[t]) for t in TTA_CANDIDATES
                    }
                    # Select best TTA on VAL; break ties by smallest view count
                    win_tta = max(TTA_CANDIDATES, key=lambda t: (round(cand_scores[t], 6), -TTA_VIEW_COUNTS[t]))
                    selected_ttas[m_idx] = win_tta
                    cls_1x_arr = np.concatenate(cand_cls["raw_1x"], axis=0)
                    cls_d4_arr = np.concatenate(cand_cls["d4_8view_1x"], axis=0)
                    cls_tta_arr = np.concatenate(cand_cls[win_tta], axis=0)

                    # Copy winning TTA memmap into canonical mm
                    for st in range(0, n_patches, 128):
                        en = min(n_patches, st + 128)
                        mm[st:en] = cand_mms[win_tta][st:en]
                    mm.flush()

                    tta_val_tables[arm_name] = {
                        "selected_tta": win_tta,
                        "candidates": {
                            t: {
                                "mIoU": round(cand_scores[t], 4),
                                "cls_auroc_brier": compute_patch_auroc_and_brier(
                                    np.concatenate(cand_cls[t], axis=0), y_gt_all
                                ),
                            }
                            for t in TTA_CANDIDATES
                        },
                    }
                finally:
                    for t in TTA_CANDIDATES:
                        _close_memmap(cand_mms[t])
                    cand_mms.clear()
                    for t in TTA_CANDIDATES:
                        tmp_p = memmap_dir / f"tmp_{split_tag}_{m_idx}_{t}.dat"
                        if tmp_p.exists():
                            try:
                                tmp_p.unlink()
                            except OSError:
                                pass
            else:
                mode_m = selected_ttas[m_idx]
                pos = 0
                with torch.no_grad():
                    for batch in loader:
                        imgs_b, _, _ = _extract_img_gt(batch)
                        b = imgs_b.size(0)
                        imgs_b = imgs_b.to(device, non_blocking=True)
                        seg_p, cls_d = apply_tta_probs(
                            model_obj,
                            imgs_b,
                            tta_mode=mode_m,
                            device=device,
                            amp_dtype=amp_dtype,
                            include_d4_cls=include_d4_cls,
                        )
                        mm[pos : pos + b] = encode_centered_logprobs_fp16(seg_p.cpu().numpy())
                        cls_1x_arr[pos : pos + b] = cls_d["1x"].cpu().numpy()
                        cls_d4_arr[pos : pos + b] = cls_d["d4"].cpu().numpy()
                        cls_tta_arr[pos : pos + b] = cls_d["tta"].cpu().numpy()
                        pos += b
                mm.flush()
        finally:
            _close_memmap(mm)
            del mm
            if loaded_here:
                del model_obj
            else:
                model_obj.cpu()
            if loaded_here and device.type == "cuda":
                gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        memmap_paths.append(mm_path)
        cls_1x_list.append(cls_1x_arr)
        cls_d4_list.append(cls_d4_arr)
        cls_tta_list.append(cls_tta_arr)

    return {
        "memmap_paths": memmap_paths,
        "shape": (n_patches, NUM_FOREGROUND, h, w),
        "gt_all": gt_all,
        "y_gt_all": y_gt_all,
        "nec_pixels": nec_pixels,
        "selected_ttas": selected_ttas,
        "tta_val_tables": tta_val_tables,
        "cls_1x": cls_1x_list,
        "cls_d4": cls_d4_list,
        "cls_tta": cls_tta_list,
        "arm_metas": arm_metas,
    }


@torch.no_grad()
def _extract_stage1_cls_probs(
    stage1_cls_model: Optional[WSSSTissueStage1NetCAM],
    stage1_norm_mode: str,
    loader: DataLoader,
    n_patches: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    if stage1_cls_model is None:
        return None
    stage1_cls_model = stage1_cls_model.to(device).eval()
    probs_list: List[np.ndarray] = []
    for batch in loader:
        imgs, _, _ = _extract_img_gt(batch)
        imgs = imgs.to(device, non_blocking=True)
        inp = prepare_stage1_input(imgs, stage1_norm_mode=stage1_norm_mode, images_are_imagenet_norm=True)
        _, s1_pr = stage1_cls_model.forward_cam(inp)
        probs_list.append(s1_pr.cpu().numpy())
    stage1_cls_model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = np.concatenate(probs_list, axis=0)
    assert out.shape == (n_patches, NUM_FOREGROUND)
    return out


def _fuse_memmaps_to_probs(
    memmap_paths: Sequence[Path],
    shape: Tuple[int, int, int, int],
    weights: Sequence[float],
    betas: Sequence[float],
    prob_mode: str,
    eps_log: float = 1e-3,
    batch_size: int = 128,
) -> np.ndarray:
    """Reconstruct and fuse M centered float16 memmaps into (N, 4, H, W) float32 normalized score/probs."""
    n_patches, c_fg, h, w = shape
    w_arr = np.asarray(weights, dtype=np.float32)
    out = np.empty((n_patches, c_fg, h, w), dtype=np.float32)
    mms = [np.memmap(p, dtype=np.float16, mode="r", shape=shape) for p in memmap_paths]
    try:
        for start in range(0, n_patches, int(batch_size)):
            end = min(n_patches, start + int(batch_size))
            out[start:end] = _fuse_memmap_chunk(mms, start, end, w_arr, betas, prob_mode, eps_log=eps_log)
    finally:
        for mm in mms:
            _close_memmap(mm)
        mms.clear()
    return out


def run_stage2_v2_ensemble_eval(
    arm_specs: Sequence[Dict[str, Any]],
    val_loader: DataLoader,
    test_loader: DataLoader,
    stage1_cls_ckpt: Optional[Union[str, Path]] = None,
    stage1_cls_model: Optional[WSSSTissueStage1NetCAM] = None,
    stage1_norm_mode: str = "zero_one",
    author_stage2_ckpt: Optional[Union[str, Path]] = None,
    author_stage1_retrained_ckpt: Optional[Union[str, Path]] = None,
    author_stage1_official_ckpt: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "/content/stage2_results",
    zip_path: Union[str, Path] = "/content/SE374_Stage2_Training_Results.zip",
    work_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    amp_dtype: Optional[torch.dtype] = None,
    n_bootstrap: int = 10000,
    n_perm: int = 10000,
    seed: int = 42,
    val_patient_ids: Optional[Sequence[str]] = None,
    test_patient_ids: Optional[Sequence[str]] = None,
    before_test_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run complete zero-leakage 4-arm cross-family ensemble evaluation and export artifacts."""
    if len(arm_specs) != 4:
        raise ValueError(f"Expected exactly 4 Stage-2 arms, got {len(arm_specs)}")
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_p = Path(zip_path)
    zip_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Patient-level anti-leakage verification
    v_pids = list(val_patient_ids) if val_patient_ids is not None else _extract_patient_ids_from_loader(val_loader)
    t_pids = list(test_patient_ids) if test_patient_ids is not None else _extract_patient_ids_from_loader(test_loader)
    overlap = set(v_pids) & set(t_pids)
    if overlap:
        raise RuntimeError(f"Patient leakage detected between VAL and TEST: {sorted(overlap)}")

    if stage1_cls_model is None and stage1_cls_ckpt is not None and Path(stage1_cls_ckpt).is_file():
        s1_cls_m = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
        s1_norm_loaded, s1_cls_sha, _ = load_stage1_strict(s1_cls_m, stage1_cls_ckpt)
        stage1_norm_mode = s1_norm_loaded
    else:
        s1_cls_m = stage1_cls_model
        s1_cls_sha = compute_file_sha256(stage1_cls_ckpt) if stage1_cls_ckpt and Path(stage1_cls_ckpt).is_file() else ""

    temp_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    if work_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="s2v2_ens_")
        wdir = Path(temp_ctx.name)
    else:
        wdir = Path(work_dir)
        wdir.mkdir(parents=True, exist_ok=True)

    try:
        # 2. VAL-ONLY Phase: TTA selection -> centered float16 memmaps -> CV -> CGM calibration
        val_stream = _stream_split_models(
            arm_specs=arm_specs,
            loader=val_loader,
            arm_tta_modes=list(TTA_CANDIDATES),
            device=dev,
            memmap_dir=wdir,
            split_tag="val",
            amp_dtype=amp_dtype,
            select_tta_on_split=True,
        )
        val_unique_pids, val_pidx, val_folds = build_patient_folds(
            v_pids, val_stream["nec_pixels"], n_folds=5
        )
        n_val_pat = len(val_unique_pids)

        # Select cls_probs view source strictly on VAL AUROC / Brier
        cls_src_scores: Dict[str, Dict[str, float]] = {}
        for src_k in ("cls_1x", "cls_d4", "cls_tta"):
            per_arm_m = [
                compute_patch_auroc_and_brier(val_stream[src_k][m], val_stream["y_gt_all"])
                for m in range(4)
            ]
            cls_src_scores[src_k] = {
                "mean_macro_auroc": round(float(np.mean([d["macro_auroc"] for d in per_arm_m])), 4),
                "mean_brier": round(float(np.mean([d["brier"] for d in per_arm_m])), 4),
            }
        # Prefer cls_tta/cls_d4 over cls_1x only if AUROC improves or ties with lower Brier
        selected_cls_source = max(
            ("cls_1x", "cls_d4", "cls_tta"),
            key=lambda k: (cls_src_scores[k]["mean_macro_auroc"], -cls_src_scores[k]["mean_brier"]),
        )

        val_mms = [
            np.memmap(p, dtype=np.float16, mode="r", shape=val_stream["shape"])
            for p in val_stream["memmap_paths"]
        ]
        try:
            val_dis = _extract_disagreement_and_unanimous(
                val_mms, val_stream["gt_all"], val_pidx, n_val_pat
            )
            val_diversity = _compute_diversity_and_oracles(
                val_dis,
                val_mms,
                val_stream["gt_all"],
                val_pidx,
                n_val_pat,
                [s["name"] for s in arm_specs],
            )
        finally:
            for mm in val_mms:
                _close_memmap(mm)
            val_mms.clear()
            gc.collect()

        arm_names = [str(s["name"]) for s in arm_specs]
        arm_families = [str(s.get("family", DEFAULT_ARM_FAMILIES.get(s["name"], "unknown"))) for s in arm_specs]
        all_pair_specs: List[Tuple[str, int, int, List[float]]] = []
        cnn_vit_pairs: List[str] = []
        for i in range(4):
            for j in range(i + 1, 4):
                pname = f"{arm_names[i]}+{arm_names[j]}"
                w_pair = [0.5 if k in (i, j) else 0.0 for k in range(4)]
                all_pair_specs.append((pname, i, j, w_pair))
                if arm_families[i] != arm_families[j]:
                    cnn_vit_pairs.append(pname)
        pair_indices_map = {pname: (i, j) for pname, i, j, _ in all_pair_specs}
        pair_weights_map = {pname: w_pair for pname, _, _, w_pair in all_pair_specs}
        pair_pool = cnn_vit_pairs if cnn_vit_pairs else [pname for pname, _, _, _ in all_pair_specs]

        val_cv = run_val_cv_and_lopocv(val_dis, val_folds, num_arms=4, adoption_delta=0.0005, device=dev)
        primary_weights = list(val_cv["primary_weights"])
        primary_mode = str(val_cv["selected_prob_mode"])
        primary_betas = list(val_cv["full_val_betas"])

        # Evaluate VAL no-CGM scores for 4 singles + candidate CNN+ViT pairs before freeing val_dis
        val_cand_w = np.asarray(
            [[1.0 if k == m else 0.0 for k in range(4)] for m in range(4)]
            + [pair_weights_map[pn] for pn in pair_pool],
            dtype=np.float32,
        )
        cand_ip_val, cand_up_val = evaluate_weights_on_disagreement(
            val_dis["z_dis"],
            val_dis["gt_dis"],
            val_dis["pat_dis"],
            val_dis["inter_u"],
            val_dis["union_u"],
            val_cand_w,
            primary_betas,
            prob_mode=primary_mode,
            device=dev,
        )
        val_single_no_cgm_mious = {
            arm_names[m]: compute_macro_miou_from_iu(cand_ip_val[m].sum(axis=0), cand_up_val[m].sum(axis=0))
            for m in range(4)
        }
        val_pair_no_cgm_mious = {
            pn: compute_macro_miou_from_iu(cand_ip_val[4 + idx].sum(axis=0), cand_up_val[4 + idx].sum(axis=0))
            for idx, pn in enumerate(pair_pool)
        }
        val_best_single_no_cgm_name = max(arm_names, key=lambda n: round(val_single_no_cgm_mious[n], 6))
        val_best_cnn_vit_pair_no_cgm_name = max(pair_pool, key=lambda n: round(val_pair_no_cgm_mious[n], 6))

        # Free val_dis before CGM bitmask calibration so they never coexist in RAM
        del val_dis, cand_ip_val, cand_up_val
        gc.collect()

        val_s1_cls = _extract_stage1_cls_probs(
            s1_cls_m, stage1_norm_mode, val_loader, val_stream["shape"][0], dev
        )
        if val_s1_cls is None:
            # Fallback neutral expert if Stage-1 cls model omitted in minimal test
            val_s1_cls = np.mean(np.stack(val_stream[selected_cls_source], axis=0), axis=0)

        val_bitmasks = _eval_bitmasks_from_memmaps(
            val_stream["memmap_paths"],
            val_stream["shape"],
            primary_weights,
            primary_betas,
            primary_mode,
            val_stream["gt_all"],
            batch_size=128,
        )
        val_cgm = calibrate_ensemble_cgm_cv(
            fused_probs=None,
            gt_masks=val_stream["gt_all"],
            patch_patient_idx=val_pidx,
            folds=val_folds,
            s1_cls_probs=val_s1_cls,
            s2_cls_probs=val_stream[selected_cls_source],
            y_gt_patches=val_stream["y_gt_all"],
            arm_weights=primary_weights,
            grid=ENSEMBLE_CGM_GRID,
            adoption_delta=0.0005,
            precomputed_bitmasks=val_bitmasks,
        )
        del val_bitmasks
        gc.collect()

        # Calibrate VAL-only CGM policies for all 4 individual arms and candidate CNN+ViT pairs
        val_indiv_cgm_policies: Dict[str, Any] = {}
        for m in range(4):
            w_one = [1.0 if k == m else 0.0 for k in range(4)]
            ibm_m, ubm_m = _eval_bitmasks_from_memmaps(
                val_stream["memmap_paths"],
                val_stream["shape"],
                w_one,
                primary_betas,
                primary_mode,
                val_stream["gt_all"],
                batch_size=128,
            )
            val_indiv_cgm_policies[arm_names[m]] = _calibrate_subset_cgm_cv(
                ibm_m,
                ubm_m,
                val_pidx,
                val_folds,
                val_s1_cls,
                val_stream[selected_cls_source][m],
                grid=ENSEMBLE_CGM_GRID,
                adoption_delta=0.0005,
            )
            del ibm_m, ubm_m

        val_pair_cgm_policies: Dict[str, Any] = {}
        for pn in pair_pool:
            pi, pj = pair_indices_map[pn]
            ibm_p, ubm_p = _eval_bitmasks_from_memmaps(
                val_stream["memmap_paths"],
                val_stream["shape"],
                pair_weights_map[pn],
                primary_betas,
                primary_mode,
                val_stream["gt_all"],
                batch_size=128,
            )
            s2_pair = 0.5 * val_stream[selected_cls_source][pi] + 0.5 * val_stream[selected_cls_source][pj]
            val_pair_cgm_policies[pn] = _calibrate_subset_cgm_cv(
                ibm_p,
                ubm_p,
                val_pidx,
                val_folds,
                val_s1_cls,
                s2_pair,
                grid=ENSEMBLE_CGM_GRID,
                adoption_delta=0.0005,
            )
            del ibm_p, ubm_p

        val_best_single_name = max(
            arm_names, key=lambda n: round(val_indiv_cgm_policies[n]["effective_val_miou"], 6)
        )
        val_best_cnn_vit_pair_name = max(
            pair_pool, key=lambda n: round(val_pair_cgm_policies[n]["effective_val_miou"], 6)
        )

        # Freeze immutable VAL protocol selection BEFORE touching TEST
        val_protocol_selection: Dict[str, Any] = {
            "selected_ttas": list(val_stream["selected_ttas"]),
            "selected_cls_source": selected_cls_source,
            "selected_prob_mode": primary_mode,
            "selected_weight_strategy": str(val_cv["selected_weight_strategy"]),
            "primary_weights": list(primary_weights),
            "simplex_bagged_weights": list(val_cv["simplex_bagged_weights"]),
            "uniform_weights": list(val_cv["uniform_weights"]),
            "full_val_betas": list(primary_betas),
            "cgm_expert_strategy": str(val_cgm["selected_expert_strategy"]),
            "cgm_dof": str(val_cgm["selected_dof"]),
            "cgm_thresholds": list(val_cgm["selected_thresholds"]),
            "cgm_bce_betas": val_cgm["selected_bce_betas"],
            "use_cgm": bool(val_cgm["use_cgm"]),
            "val_best_single_name": val_best_single_name,
            "val_best_single_no_cgm_name": val_best_single_no_cgm_name,
            "val_best_cnn_vit_pair_name": val_best_cnn_vit_pair_name,
            "val_best_cnn_vit_pair_no_cgm_name": val_best_cnn_vit_pair_no_cgm_name,
            "individual_arm_cgm_policies": val_indiv_cgm_policies,
            "best_cnn_vit_pair_cgm_policy": val_pair_cgm_policies[val_best_cnn_vit_pair_name],
            "pair_cgm_policies": val_pair_cgm_policies,
        }

        # Clean up VAL memmaps and ground truth before TEST
        val_stream.pop("gt_all", None)
        for p in val_stream["memmap_paths"]:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        gc.collect()

        if before_test_hook is not None:
            before_test_hook(dict(val_protocol_selection))

        # 3. FROZEN TEST EVALUATION (Zero selection/fitting on TEST!)
        test_stream = _stream_split_models(
            arm_specs=arm_specs,
            loader=test_loader,
            arm_tta_modes=val_protocol_selection["selected_ttas"],
            device=dev,
            memmap_dir=wdir,
            split_tag="test",
            amp_dtype=amp_dtype,
            select_tta_on_split=False,
            include_d4_cls=(val_protocol_selection["selected_cls_source"] == "cls_d4"),
        )
        test_unique_pids, test_pidx, _ = build_patient_folds(
            t_pids, test_stream["nec_pixels"], n_folds=5
        )
        n_test_pat = len(test_unique_pids)
        test_gt = test_stream["gt_all"]

        # Single-pass chunked TEST streaming: diversity/oracles + ALL 16 no-CGM weight vectors
        # Order: 0..3 singles, 4..9 pairs, 10..13 triplets, 14 uniform 4-arm, 15 simplex 4-arm
        triplet_specs: List[Tuple[str, List[float]]] = []
        for drop_idx in range(4):
            kept = [arm_names[k] for k in range(4) if k != drop_idx]
            triplet_specs.append(("+".join(kept), [0.0 if k == drop_idx else 1.0 / 3.0 for k in range(4)]))

        w_all_rows: List[List[float]] = (
            [[1.0 if j == m else 0.0 for j in range(4)] for m in range(4)]
            + [w_pair for _, _, _, w_pair in all_pair_specs]
            + [w_trip for _, w_trip in triplet_specs]
            + [list(val_protocol_selection["uniform_weights"]), list(val_protocol_selection["simplex_bagged_weights"])]
        )
        w_all = np.asarray(w_all_rows, dtype=np.float32)

        test_diversity, test_inter_all, test_union_all = _stream_test_diversity_and_no_cgm_weights(
            memmap_paths=test_stream["memmap_paths"],
            shape=test_stream["shape"],
            gt_masks=test_gt,
            patch_patient_idx=test_pidx,
            num_patients=n_test_pat,
            arm_names=arm_names,
            weight_matrix=w_all,
            betas=primary_betas,
            prob_mode=primary_mode,
            batch_size=128,
            device=dev,
        )

        def _iu_summary(ip: np.ndarray, up: np.ndarray) -> Dict[str, Any]:
            return {
                **summarize_iu(ip.sum(axis=0), up.sum(axis=0)),
                "bootstrap_95ci": bootstrap_pooled_miou_ci(ip, up, n_boot=n_bootstrap, seed=seed),
                "inter_p": ip,
                "union_p": up,
            }

        # Extract Stage-1 cls probs on TEST (needed for 5-expert and s1_s2_50_50 CGM gates)
        test_s1_cls = _extract_stage1_cls_probs(
            s1_cls_m, stage1_norm_mode, test_loader, test_stream["shape"][0], dev
        )
        if test_s1_cls is None:
            test_s1_cls = np.mean(np.stack(test_stream[selected_cls_source], axis=0), axis=0)

        # Individual 4 arms (selected TTA: both No-CGM and VAL-calibrated CGM)
        indiv_results: Dict[str, Any] = {}
        for m in range(4):
            a_name = arm_names[m]
            w_one = [1.0 if j == m else 0.0 for j in range(4)]
            ip_no, up_no = test_inter_all[m], test_union_all[m]
            no_cgm_entry = _iu_summary(ip_no, up_no)

            arm_pol = val_indiv_cgm_policies[a_name]
            s2_arm = test_stream[selected_cls_source][m]
            gate_arm = (
                s2_arm
                if arm_pol["selected_gate_source"] == "s2_self"
                else (0.5 * np.asarray(test_s1_cls, dtype=np.float64) + 0.5 * np.asarray(s2_arm, dtype=np.float64))
            )
            cgm_iu_arm = _eval_cgm_from_memmaps(
                test_stream["memmap_paths"],
                test_stream["shape"],
                w_one,
                primary_betas,
                primary_mode,
                test_gt,
                test_pidx,
                n_test_pat,
                gate_arm,
                {"cgm_val_calibrated": arm_pol["selected_thresholds"]},
                batch_size=128,
            )
            ip_cal, up_cal = cgm_iu_arm["cgm_val_calibrated"]
            cal_entry = _iu_summary(ip_cal, up_cal)
            arm_mode_key = "cgm_val_calibrated" if arm_pol["use_cgm"] else "no_cgm"
            matched_arm = cal_entry if arm_mode_key == "cgm_val_calibrated" else no_cgm_entry
            indiv_results[a_name] = {
                **matched_arm,
                "cgm_mode": arm_mode_key,
                "cgm_policy": arm_pol,
                "no_cgm": no_cgm_entry,
                "cgm_val_calibrated": cal_entry,
            }
        best_single_name = val_best_single_name

        # All 6 pairwise blends (No-CGM) + VAL pre-registered Best CNN+ViT pair (No-CGM + VAL-calibrated CGM)
        pairwise_results: Dict[str, Any] = {}
        for idx_p, (pname, _, _, _) in enumerate(all_pair_specs):
            ip_ij, up_ij = test_inter_all[4 + idx_p], test_union_all[4 + idx_p]
            pairwise_results[pname] = _iu_summary(ip_ij, up_ij)

        best_cnn_vit_pair_name = val_best_cnn_vit_pair_name
        pair_pol = val_pair_cgm_policies[best_cnn_vit_pair_name]
        pi_best, pj_best = pair_indices_map[best_cnn_vit_pair_name]
        s2_pair_test = (
            0.5 * np.asarray(test_stream[selected_cls_source][pi_best], dtype=np.float64)
            + 0.5 * np.asarray(test_stream[selected_cls_source][pj_best], dtype=np.float64)
        )
        gate_pair = (
            s2_pair_test
            if pair_pol["selected_gate_source"] == "s2_self"
            else (0.5 * np.asarray(test_s1_cls, dtype=np.float64) + 0.5 * s2_pair_test)
        )
        cgm_iu_pair = _eval_cgm_from_memmaps(
            test_stream["memmap_paths"],
            test_stream["shape"],
            pair_weights_map[best_cnn_vit_pair_name],
            primary_betas,
            primary_mode,
            test_gt,
            test_pidx,
            n_test_pat,
            gate_pair,
            {"cgm_val_calibrated": pair_pol["selected_thresholds"]},
            batch_size=128,
        )
        pair_no_cgm_entry = pairwise_results[best_cnn_vit_pair_name]
        pair_cal_entry = _iu_summary(*cgm_iu_pair["cgm_val_calibrated"])
        best_pair_mode_key = "cgm_val_calibrated" if pair_pol["use_cgm"] else "no_cgm"
        best_pair_entry = pair_cal_entry if best_pair_mode_key == "cgm_val_calibrated" else pair_no_cgm_entry
        pairwise_results[best_cnn_vit_pair_name] = {
            **pair_no_cgm_entry,
            "cgm_mode": best_pair_mode_key,
            "cgm_policy": pair_pol,
            "no_cgm": dict(pair_no_cgm_entry),
            "cgm_val_calibrated": pair_cal_entry,
        }

        # All 4 triplets
        triplet_results: Dict[str, Any] = {}
        for idx_t, (tname, _) in enumerate(triplet_specs):
            ip_t, up_t = test_inter_all[10 + idx_t], test_union_all[10 + idx_t]
            triplet_results[tname] = {
                **summarize_iu(ip_t.sum(axis=0), up_t.sum(axis=0)),
                "bootstrap_95ci": bootstrap_pooled_miou_ci(ip_t, up_t, n_boot=n_bootstrap, seed=seed),
            }
        best_triplet_name = max(triplet_results.keys(), key=lambda n: triplet_results[n]["mIoU_raw"])

        # Equal 4-Arm & VAL-Weighted 4-Arm (No-CGM, fixed-0.10, primary CGM) via chunked memmap streaming
        def _eval_4arm_with_cgm(
            w_vec: Sequence[float],
            ip_no: np.ndarray,
            up_no: np.ndarray,
            gt_arr: np.ndarray,
        ) -> Dict[str, Any]:
            gate_p = _combine_5_expert_probs(
                test_s1_cls,
                test_stream[selected_cls_source],
                w_vec,
                val_protocol_selection["cgm_expert_strategy"],
                bce_betas=val_protocol_selection["cgm_bce_betas"],
            )
            cgm_res = _eval_cgm_from_memmaps(
                test_stream["memmap_paths"],
                test_stream["shape"],
                w_vec,
                primary_betas,
                primary_mode,
                gt_arr,
                test_pidx,
                n_test_pat,
                gate_p,
                {
                    "cgm_fixed_0_10": 0.10,
                    "cgm_val_calibrated": val_protocol_selection["cgm_thresholds"],
                },
                batch_size=128,
            )
            ip_fix, up_fix = cgm_res["cgm_fixed_0_10"]
            ip_cal, up_cal = cgm_res["cgm_val_calibrated"]
            return {
                "no_cgm": _iu_summary(ip_no, up_no),
                "cgm_fixed_0_10": _iu_summary(ip_fix, up_fix),
                "cgm_val_calibrated": _iu_summary(ip_cal, up_cal),
            }

        equal_4arm_res = _eval_4arm_with_cgm(
            val_protocol_selection["uniform_weights"],
            test_inter_all[14],
            test_union_all[14],
            test_gt,
        )
        simplex_4arm_res = _eval_4arm_with_cgm(
            val_protocol_selection["simplex_bagged_weights"],
            test_inter_all[15],
            test_union_all[15],
            test_gt,
        )
        primary_4arm_res = (
            simplex_4arm_res
            if val_protocol_selection["selected_weight_strategy"] == "simplex_cv"
            else equal_4arm_res
        )
        primary_mode_key = "cgm_val_calibrated" if val_protocol_selection["use_cgm"] else "no_cgm"
        primary_entry = primary_4arm_res[primary_mode_key]
        equal_entry = equal_4arm_res[primary_mode_key]
        best_single_entry = indiv_results[best_single_name][indiv_results[best_single_name]["cgm_mode"]]

        # Clean up TEST memmaps and gt_all promptly after all memmap evals complete
        del test_gt, test_inter_all, test_union_all
        test_stream.pop("gt_all", None)
        for p in test_stream["memmap_paths"]:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        gc.collect()

        # Official Author Anchors (shared single forward pass for 1x + 6-view MS-TTA)
        author_anchors: Dict[str, Any] = {}
        anchor_ckpts: Dict[str, Union[str, Path]] = {}
        if author_stage2_ckpt is not None and Path(author_stage2_ckpt).is_file():
            if author_stage1_retrained_ckpt is not None and Path(author_stage1_retrained_ckpt).is_file():
                anchor_ckpts["author_step0_anchor_7043"] = author_stage1_retrained_ckpt
            if author_stage1_official_ckpt is not None and Path(author_stage1_official_ckpt).is_file():
                anchor_ckpts["author_official_pure"] = author_stage1_official_ckpt
            if anchor_ckpts:
                author_anchors = _evaluate_author_anchors_shared(
                    author_stage2_ckpt=author_stage2_ckpt,
                    stage1_ckpts=anchor_ckpts,
                    loader=test_loader,
                    patch_patient_idx=test_pidx,
                    num_patients=n_test_pat,
                    device=dev,
                )

        # Primary Author Anchor for m=5 Holm-Bonferroni confirmatory family (Step-0 TTA+CGM anchor)
        ref_anchor_key = (
            "author_step0_anchor_7043"
            if "author_step0_anchor_7043" in author_anchors
            else ("author_official_pure" if "author_official_pure" in author_anchors else None)
        )
        ref_anchor_mode = "ms_tta_cgm_neg1e4" if ref_anchor_key is not None else indiv_results[best_single_name]["cgm_mode"]
        ref_anchor_iu = (
            author_anchors[ref_anchor_key]["ms_tta_cgm_neg1e4"]
            if ref_anchor_key is not None
            else best_single_entry
        )

        def _run_paired_comparison_suite(
            specs: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any], str, str]],
        ) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            raw_perm_ps: List[float] = []
            raw_wilc_ps: List[float] = []
            for comp_name, res_a, res_b, mode_a, mode_b in specs:
                perm = paired_patient_swap_permutation_test(
                    res_a["inter_p"], res_a["union_p"], res_b["inter_p"], res_b["union_p"], n_perm=n_perm, seed=seed
                )
                pa_miou = compute_per_patient_miou(res_a["inter_p"], res_a["union_p"])
                pb_miou = compute_per_patient_miou(res_b["inter_p"], res_b["union_p"])
                wilc = paired_wilcoxon_and_dz(pa_miou, pb_miou)
                raw_perm_ps.append(perm["p_value"])
                raw_wilc_ps.append(wilc["wilcoxon_p"])
                rows.append(
                    {
                        "comparison": comp_name,
                        "mode_a": mode_a,
                        "mode_b": mode_b,
                        "delta_miou": perm["delta_miou"],
                        "perm_p_raw": round(perm["p_value"], 6),
                        **wilc,
                    }
                )
            adj_perm = holm_bonferroni(raw_perm_ps)
            adj_wilc = holm_bonferroni(raw_wilc_ps)
            for idx_c, row in enumerate(rows):
                row["perm_p_holm"] = round(adj_perm[idx_c], 6)
                row["wilcoxon_p_holm"] = round(adj_wilc[idx_c], 6)
            return rows

        # m = 5 Pre-Registered Confirmatory Comparisons (strictly matched TTA+CGM vs TTA+CGM)
        best_single_mode_k = str(indiv_results[best_single_name]["cgm_mode"])
        comparisons_spec = [
            ("primary_4arm_vs_author", primary_entry, ref_anchor_iu, primary_mode_key, ref_anchor_mode),
            ("equal_4arm_vs_author", equal_entry, ref_anchor_iu, primary_mode_key, ref_anchor_mode),
            ("best_single_arm_vs_author", best_single_entry, ref_anchor_iu, best_single_mode_k, ref_anchor_mode),
            ("best_cnn_vit_pair_vs_author", best_pair_entry, ref_anchor_iu, best_pair_mode_key, ref_anchor_mode),
            ("primary_4arm_vs_best_single_arm", primary_entry, best_single_entry, primary_mode_key, best_single_mode_k),
        ]
        confirmatory_stats = _run_paired_comparison_suite(comparisons_spec)

        # Secondary paired stats across BOTH Author anchors and BOTH modes (ms_tta_cgm_neg1e4 & cgm_author_mul0)
        secondary_anchor_stats: Dict[str, Any] = {}
        for ak_sec in ("author_step0_anchor_7043", "author_official_pure"):
            if ak_sec not in author_anchors:
                continue
            for amode in ("ms_tta_cgm_neg1e4", "cgm_author_mul0"):
                anch_sec = author_anchors[ak_sec][amode]
                sec_specs = [
                    ("primary_4arm_vs_author", primary_entry, anch_sec, primary_mode_key, amode),
                    ("equal_4arm_vs_author", equal_entry, anch_sec, primary_mode_key, amode),
                    ("best_single_arm_vs_author", best_single_entry, anch_sec, best_single_mode_k, amode),
                    ("best_cnn_vit_pair_vs_author", best_pair_entry, anch_sec, best_pair_mode_key, amode),
                ]
                secondary_anchor_stats[f"{ak_sec}__{amode}"] = {
                    "anchor_key": ak_sec,
                    "anchor_mode": amode,
                    "anchor_miou": anch_sec["mIoU"],
                    "anchor_bootstrap_95ci": bootstrap_pooled_miou_ci(
                        anch_sec["inter_p"], anch_sec["union_p"], n_boot=n_bootstrap, seed=seed
                    ),
                    "comparisons": _run_paired_comparison_suite(sec_specs),
                }

        # Pure-ensemble No-CGM diagnostic (outside Holm m=5 confirmatory family)
        no_cgm_ens = primary_4arm_res["no_cgm"]
        no_cgm_best_s = indiv_results[val_best_single_no_cgm_name]["no_cgm"]
        pure_perm = paired_patient_swap_permutation_test(
            no_cgm_ens["inter_p"],
            no_cgm_ens["union_p"],
            no_cgm_best_s["inter_p"],
            no_cgm_best_s["union_p"],
            n_perm=n_perm,
            seed=seed,
        )
        pure_wilc = paired_wilcoxon_and_dz(
            compute_per_patient_miou(no_cgm_ens["inter_p"], no_cgm_ens["union_p"]),
            compute_per_patient_miou(no_cgm_best_s["inter_p"], no_cgm_best_s["union_p"]),
        )
        pure_ensemble_diagnostic = {
            "primary_4arm_no_cgm_vs_best_single_no_cgm": {
                "comparison": "primary_4arm_no_cgm_vs_best_single_no_cgm",
                "best_single_no_cgm_name": val_best_single_no_cgm_name,
                "primary_4arm_no_cgm_miou": no_cgm_ens["mIoU"],
                "best_single_no_cgm_miou": no_cgm_best_s["mIoU"],
                "delta_miou": pure_perm["delta_miou"],
                "perm_p_raw": round(pure_perm["p_value"], 6),
                **pure_wilc,
            }
        }

        # Per-patient CSV export (matched TTA+CGM)
        csv_path = out_dir / "per_patient_test_metrics.csv"
        primary_pp = compute_per_patient_miou(primary_entry["inter_p"], primary_entry["union_p"])
        equal_pp = compute_per_patient_miou(equal_entry["inter_p"], equal_entry["union_p"])
        best_single_pp = compute_per_patient_miou(best_single_entry["inter_p"], best_single_entry["union_p"])
        author_pp = (
            compute_per_patient_miou(ref_anchor_iu["inter_p"], ref_anchor_iu["union_p"])
            if ref_anchor_key is not None
            else None
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            header = ["patient_id", "primary_4arm_miou", "equal_4arm_miou", "best_single_miou"]
            if author_pp is not None:
                header.append(f"{ref_anchor_key}_miou")
            writer.writerow(header)
            for p_i, pid_s in enumerate(test_unique_pids):
                r_csv = [pid_s, f"{primary_pp[p_i]:.6f}", f"{equal_pp[p_i]:.6f}", f"{best_single_pp[p_i]:.6f}"]
                if author_pp is not None:
                    r_csv.append(f"{author_pp[p_i]:.6f}")
                writer.writerow(r_csv)

        # Publication-grade 300 DPI Monochrome Figure (all bars matched TTA+CGM)
        plot_entries: List[Tuple[str, float, float, float]] = []
        if ref_anchor_key is not None:
            a_ci = bootstrap_pooled_miou_ci(
                ref_anchor_iu["inter_p"], ref_anchor_iu["union_p"], n_boot=n_bootstrap, seed=seed
            )
            plot_entries.append(("Author Anchor", a_ci["point"], a_ci["ci_low"], a_ci["ci_high"]))
        for a_name in arm_names:
            ci = indiv_results[a_name][indiv_results[a_name]["cgm_mode"]]["bootstrap_95ci"]
            short_lbl = a_name.split("_")[0].upper()
            plot_entries.append((short_lbl, ci["point"], ci["ci_low"], ci["ci_high"]))
        pair_ci = best_pair_entry["bootstrap_95ci"]
        plot_entries.append(("Best CNN+ViT", pair_ci["point"], pair_ci["ci_low"], pair_ci["ci_high"]))
        eq_ci = equal_entry["bootstrap_95ci"]
        plot_entries.append(("Equal 4-Arm", eq_ci["point"], eq_ci["ci_low"], eq_ci["ci_high"]))
        pr_ci = primary_entry["bootstrap_95ci"]
        plot_entries.append(("Primary 4-Arm", pr_ci["point"], pr_ci["ci_low"], pr_ci["ci_high"]))

        plot_path = out_dir / "stage2_training_loss.png"
        save_ensemble_comparison_plot(plot_entries, plot_path)

        # Recursively strip raw numpy inter_p/union_p arrays before JSON serialization
        def _strip_iu(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _strip_iu(v) for k, v in obj.items() if k not in ("inter_p", "union_p")}
            if isinstance(obj, list):
                return [_strip_iu(x) for x in obj]
            return obj

        clean_author = {}
        for ak, av in author_anchors.items():
            clean_author[ak] = {
                "author_stage2_sha256": av["author_stage2_sha256"],
                "stage1_sha256": av["stage1_sha256"],
                **{
                    mode_k: {
                        **_strip_iu(av[mode_k]),
                        "bootstrap_95ci": bootstrap_pooled_miou_ci(
                            av[mode_k]["inter_p"], av[mode_k]["union_p"], n_boot=n_bootstrap, seed=seed
                        ),
                    }
                    for mode_k in ("raw", "ms_tta", "cgm_author_mul0", "cgm_neg1e4", "ms_tta_cgm_neg1e4")
                },
            }

        report: Dict[str, Any] = {
            "pipeline": "stage2_v2_4arm_cross_family_ensemble",
            "anti_leakage": {
                "val_patches": int(val_stream["shape"][0]),
                "val_patients": n_val_pat,
                "test_patches": int(test_stream["shape"][0]),
                "test_patients": n_test_pat,
                "patient_overlap_count": len(overlap),
            },
            "stage1_cls_sha256": s1_cls_sha,
            "arms": test_stream["arm_metas"],
            "val_tta_selection": val_stream["tta_val_tables"],
            "val_cls_source_comparison": cls_src_scores,
            "val_cv_and_lopocv": val_cv,
            "val_cgm_calibration": val_cgm,
            "val_protocol_selection": val_protocol_selection,
            "val_diversity_and_oracles": val_diversity,
            "test_diversity_and_oracles": test_diversity,
            "test_primary_author_anchor": (
                {"anchor_key": ref_anchor_key, "mode": "ms_tta_cgm_neg1e4"}
                if ref_anchor_key is not None
                else None
            ),
            "test_author_anchors": clean_author,
            "test_individual_arms": {k: _strip_iu(v) for k, v in indiv_results.items()},
            "test_best_single_arm": best_single_name,
            "test_pairwise_blends": {k: _strip_iu(v) for k, v in pairwise_results.items()},
            "test_best_cnn_vit_pair": best_cnn_vit_pair_name,
            "test_val_best_cnn_vit_pair": {
                "pair_name": best_cnn_vit_pair_name,
                "cgm_mode": best_pair_mode_key,
                "cgm_policy": pair_pol,
                "no_cgm": _strip_iu(pair_no_cgm_entry),
                "cgm_val_calibrated": _strip_iu(pair_cal_entry),
                **_strip_iu(best_pair_entry),
            },
            "test_triplet_blends": triplet_results,
            "test_best_triplet": best_triplet_name,
            "test_equal_4arm": {k: _strip_iu(v) for k, v in equal_4arm_res.items()},
            "test_simplex_cv_4arm": {k: _strip_iu(v) for k, v in simplex_4arm_res.items()},
            "test_primary_4arm": {
                "weight_strategy": val_protocol_selection["selected_weight_strategy"],
                "cgm_mode": primary_mode_key,
                **_strip_iu(primary_entry),
            },
            "test_confirmatory_stats_holm_m5": confirmatory_stats,
            "test_secondary_author_anchor_stats": secondary_anchor_stats,
            "test_pure_ensemble_diagnostic": pure_ensemble_diagnostic,
        }

        json_path = out_dir / "STAGE2_TRAINING_METRICS.json"
        tmp_json = json_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(tmp_json, json_path)

        with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
            for artifact in (json_path, csv_path, plot_path):
                if artifact.is_file():
                    zf.write(artifact, arcname=artifact.name)

        return report
    finally:
        for p in wdir.glob("*.dat"):
            try:
                p.unlink()
            except OSError:
                pass
        if temp_ctx is not None:
            try:
                temp_ctx.cleanup()
            except OSError:
                pass
