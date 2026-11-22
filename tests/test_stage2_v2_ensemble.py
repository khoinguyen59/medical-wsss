"""Fast CPU unit and synthetic E2E tests for src/stage2_v2/ensemble.py and 02c notebook."""

import ast
import json
import time
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.stage2_v2 import (
    ENSEMBLE_CGM_GRID,
    bootstrap_pooled_miou_ci,
    build_patient_folds,
    calibrate_ensemble_cgm_cv,
    compute_macro_miou_from_iu,
    encode_centered_logprobs_fp16,
    evaluate_weights_on_disagreement,
    fit_temperature_beta_nll,
    generate_simplex_grid,
    holm_bonferroni,
    paired_patient_swap_permutation_test,
    paired_wilcoxon_and_dz,
    reconstruct_probs_from_centered,
    run_stage2_v2_ensemble_eval,
    run_val_cv_and_lopocv,
)
from src.stage2_v2.ensemble import (
    _accumulate_patient_iu,
    _apply_cgm_probs_np,
    _compute_diversity_and_oracles,
    _eval_bitmasks_from_memmaps,
    _eval_bitmasks_on_probs,
    _eval_cgm_from_memmaps,
    _evaluate_author_anchor,
    _evaluate_author_anchors_shared,
    _extract_disagreement_and_unanimous,
    _fuse_memmaps_to_probs,
    _gate_masks_from_thresholds,
    _stream_split_models,
    _stream_test_diversity_and_no_cgm_weights,
    apply_tta_probs,
)
from src.stage2_v2.evaluator import _apply_logit_cgm, _eval_author_stage2_split, _make_cgm_gate
from src.stage2_v2.models import BaseStage2DualHead, SegFormerDualHead
from src.training.dataset_loader import BCSSStage2ValDataset
from src.utils.deliverable_auditor import DeliverableAuditor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _TinyDummyArm(BaseStage2DualHead):
    """Tiny deterministic dual-head model (< 1 KB) for fast CPU ensemble tests."""

    def __init__(self, arm_name: str, bias_vec: tuple[float, float, float, float]) -> None:
        super().__init__()
        self.arch = "tiny_dummy"
        self.backbone_id = arm_name
        self.num_classes = 4
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        self.cls_fc = nn.Linear(3, 4)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, 0, 0] = torch.tensor(
                [[0.6, -0.2, -0.2], [-0.2, 0.6, -0.2], [-0.2, -0.2, 0.6], [0.2, 0.2, -0.4]],
                dtype=torch.float32,
            )
            self.conv.bias.copy_(torch.tensor(bias_vec, dtype=torch.float32))
            self.cls_fc.weight.zero_()
            self.cls_fc.bias.copy_(torch.tensor(bias_vec, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg = self.conv(x)
        cls_lg = self.cls_fc(x.mean(dim=(2, 3)))
        return {"seg_logits": seg, "cls_logits": cls_lg}


class _TinyDummyStage1Cls(nn.Module):
    """Minimal Stage-1 classifier stub exposing forward_cam(x)."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward_cam(self, x: torch.Tensor):
        b, _, h, w = x.shape
        cams = torch.ones((b, 4, h, w), device=x.device, dtype=torch.float32) * 0.5
        probs = torch.full((b, 4), 0.55, device=x.device, dtype=torch.float32)
        return cams, probs


def test_centered_fp16_reconstruction_and_temperature_fit() -> None:
    """Centered float16 log-probs reconstruct softmax(beta*z) within < 1e-3 with zero underflow."""
    rng = np.random.default_rng(7)
    logits = rng.normal(0.0, 3.5, size=(16, 4, 24, 24)).astype(np.float32)
    # Force an extreme low-probability pixel (1e-6) that would underflow raw fp16
    logits[0, :, 0, 0] = np.array([13.8, 0.0, -0.5, -1.0], dtype=np.float32)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    z_fp16 = encode_centered_logprobs_fp16(probs, eps=1e-7, axis=1)
    assert z_fp16.dtype == np.float16
    assert np.all(np.isfinite(z_fp16))
    assert float(z_fp16.min()) >= -16.5 and float(z_fp16.max()) == 0.0

    for beta in (0.5, 1.0, 2.0):
        rec = reconstruct_probs_from_centered(z_fp16, beta=beta, axis=1)
        ref_l = np.log(np.clip(probs, 1e-7, 1.0)) * beta
        ref_l -= ref_l.max(axis=1, keepdims=True)
        ref = np.exp(ref_l) / np.exp(ref_l).sum(axis=1, keepdims=True)
        max_err = float(np.max(np.abs(rec - ref)))
        assert max_err < 1e-3, f"Reconstruction error {max_err} >= 1e-3 for beta={beta}"

    # Fit beta recovering sharpening from noisy logits
    gt = np.argmax(logits[:, :, ::4, ::4], axis=1).reshape(-1)
    z_roi = np.transpose(z_fp16[:, :, ::4, ::4], (0, 2, 3, 1)).reshape(-1, 4)
    beta_fit = fit_temperature_beta_nll(z_roi, gt, bounds=(0.25, 4.0))
    assert 0.25 <= beta_fit <= 4.0


def test_disagreement_acceleration_exact_parity_all_969_simplex() -> None:
    """Disagreement-pixel acceleration matches brute-force (969, P, 4) confusion with 0.0 exact diff."""
    simplex = generate_simplex_grid(num_arms=4, step=0.05, min_weight=0.05)
    assert simplex.shape == (969, 4)
    assert np.all(simplex >= 0.05 - 1e-6)
    assert np.allclose(simplex.sum(axis=1), 1.0, atol=1e-6)

    rng = np.random.default_rng(123)
    n_patches, num_arms, h, w = 9, 4, 12, 12
    num_pat = 3
    patch_pidx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    gt = rng.integers(0, 4, size=(n_patches, h, w), dtype=np.uint8)
    gt[0, :3, :3] = 4  # outside-ROI ignore
    gt[1, :3, :3] = 255  # ignore

    z_arms = []
    for _ in range(num_arms):
        lg = rng.normal(0.0, 1.5, size=(n_patches, 4, h, w)).astype(np.float32)
        pr = np.exp(lg - lg.max(axis=1, keepdims=True))
        pr /= pr.sum(axis=1, keepdims=True)
        z_arms.append(encode_centered_logprobs_fp16(pr, axis=1))

    dis = _extract_disagreement_and_unanimous(z_arms, gt, patch_pidx, num_pat, batch_size=4, stride_cal=4)
    betas = [0.8, 1.1, 1.3, 0.9]

    for mode in ("uncal_linear", "cal_linear", "cal_logprob_eps1e3"):
        inter_acc, union_acc = evaluate_weights_on_disagreement(
            dis["z_dis"],
            dis["gt_dis"],
            dis["pat_dis"],
            dis["inter_u"],
            dis["union_u"],
            simplex,
            betas,
            prob_mode=mode,
            chunk_pixels=128,
        )

        # Brute-force reference over all pixels and all 969 weights
        feats = [
            reconstruct_probs_from_centered(
                z_arms[m],
                beta=1.0 if mode == "uncal_linear" else betas[m],
                eps_log=1e-3 if mode == "cal_logprob_eps1e3" else None,
                axis=1,
            )
            for m in range(num_arms)
        ]
        x_full = np.stack(feats, axis=1)  # (N, 4, 4, H, W)
        valid = (gt >= 0) & (gt < 4)
        inter_bf = np.zeros_like(inter_acc)
        union_bf = np.zeros_like(union_acc)
        x_2d = np.ascontiguousarray(x_full.transpose(1, 0, 2, 3, 4)).reshape(num_arms, -1)
        fused_all = (simplex @ x_2d).reshape(len(simplex), n_patches, 4, h, w)
        pred_all = np.argmax(fused_all, axis=2)  # (969, N, H, W)
        for p in range(num_pat):
            pm = patch_pidx == p
            gp = gt[pm]
            vp = valid[pm]
            prp = pred_all[:, pm, :, :]
            for c in range(4):
                pc = (prp == c) & vp[None, :, :, :]
                gc = (gp == c) & vp
                tp = np.logical_and(pc, gc[None, :, :, :]).sum(axis=(1, 2, 3))
                inter_bf[:, p, c] = tp
                union_bf[:, p, c] = pc.sum(axis=(1, 2, 3)) + gc.sum() - tp

        assert int(np.max(np.abs(inter_acc - inter_bf))) == 0, f"mode={mode} inter diff={np.argwhere(inter_acc != inter_bf)[:5]}"
        assert int(np.max(np.abs(union_acc - union_bf))) == 0, f"mode={mode} union diff={np.argwhere(union_acc != union_bf)[:5]}"


def test_zero_leakage_5fold_cv_lopocv_and_tta_d4() -> None:
    """5-fold stratified CV + 22-fold LOPOCV refits betas per fold, plus exact D4 TTA invertibility."""
    pids = [f"TCGA-BB-{i:04d}" for i in range(22) for _ in range(2)]
    nec = np.arange(44, dtype=np.int64) * 10
    unique_p, pat_idx, folds = build_patient_folds(pids, nec, n_folds=5)
    assert len(unique_p) == 22 and len(folds) == 5
    assert sorted(p for f in folds for p in f) == list(range(22))

    # Build tiny synthetic disagreement structure where arm 0 is clearly superior
    rng = np.random.default_rng(9)
    n_p, h, w = 44, 8, 8
    gt = rng.integers(0, 4, size=(n_p, h, w), dtype=np.uint8)
    z_arms = []
    for m in range(4):
        pr = np.full((n_p, 4, h, w), 0.1, dtype=np.float32)
        for i in range(n_p):
            for c in range(4):
                mask = gt[i] == c
                if m == 0:
                    pr[i, c, mask] = 0.7
                else:
                    pr[i, (c + 1) % 4, mask] = 0.7
        pr /= pr.sum(axis=1, keepdims=True)
        z_arms.append(encode_centered_logprobs_fp16(pr, axis=1))

    dis = _extract_disagreement_and_unanimous(z_arms, gt, pat_idx, 22, batch_size=16, stride_cal=2)
    cv_res = run_val_cv_and_lopocv(dis, folds, num_arms=4, adoption_delta=0.0005)
    assert cv_res["n_cv_folds"] == 5 and cv_res["n_lopocv_folds"] == 22
    assert cv_res["selected_weight_strategy"] == "simplex_cv"
    assert cv_res["primary_weights"][0] >= 0.5

    # Verify D4 8-view invariance on a symmetric dummy model
    m = _TinyDummyArm("arm_sym", (0.1, 0.0, 0.0, 0.0))
    x = torch.randn(2, 3, 16, 16)
    seg_d4, cls_d = apply_tta_probs(m, x, "d4_8view_1x", torch.device("cpu"), include_d4_cls=True)
    assert seg_d4.shape == (2, 4, 16, 16)
    assert torch.allclose(seg_d4.sum(dim=1), torch.ones((2, 16, 16)), atol=1e-5)
    assert cls_d["1x"].shape == (2, 4) and cls_d["d4"].shape == (2, 4)


def test_5expert_cgm_bitmask_parity_and_grid_to_065() -> None:
    """5-expert CGM 15-bitmask cache matches brute force; grid spans up to 0.65 and initializes 4-DOF from 1-DOF."""
    assert ENSEMBLE_CGM_GRID[-1] == 0.65 and len(ENSEMBLE_CGM_GRID) == 13
    rng = np.random.default_rng(42)
    n_p, h, w = 12, 12, 12
    gt = rng.integers(0, 4, size=(n_p, h, w), dtype=np.uint8)
    probs = rng.uniform(0.05, 1.0, size=(n_p, 4, h, w)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    inter_bm, union_bm = _eval_bitmasks_on_probs(probs, gt)

    # Verify exact parity against _apply_cgm_probs_np for random thresholds
    gate_p = rng.uniform(0.0, 0.8, size=(n_p, 4)).astype(np.float32)
    thr = [0.20, 0.45, 0.60, 0.35]
    pred_direct = _apply_cgm_probs_np(probs, gate_p, thr)
    pat_idx = np.arange(n_p) % 4
    ip_dir, up_dir = _accumulate_patient_iu(pred_direct, gt, pat_idx, 4)

    bm = _gate_masks_from_thresholds(gate_p, thr)
    ip_bm = inter_bm[np.arange(n_p), bm, :].sum(axis=0)
    up_bm = union_bm[np.arange(n_p), bm, :].sum(axis=0)
    assert np.array_equal(ip_dir.sum(axis=0), ip_bm) and np.array_equal(up_dir.sum(axis=0), up_bm)

    # Calibration CV test across all 4 strategies
    y_gt = np.stack([np.any(gt == c, axis=(1, 2)) for c in range(4)], axis=1).astype(np.float32)
    s1_p = np.clip(y_gt * 0.75 + 0.1, 0.05, 0.95)
    s2_list = [np.clip(y_gt * 0.70 + 0.12, 0.05, 0.95) for _ in range(4)]
    folds = [[0], [1], [2], [3]]
    cgm_res = calibrate_ensemble_cgm_cv(
        fused_probs=probs,
        gt_masks=gt,
        patch_patient_idx=pat_idx,
        folds=folds,
        s1_cls_probs=s1_p,
        s2_cls_probs=s2_list,
        y_gt_patches=y_gt,
        arm_weights=[0.25, 0.25, 0.25, 0.25],
    )
    assert set(cgm_res["strategies"].keys()) == {
        "equal_5",
        "balanced_s1_s2_40_60",
        "balanced_s1_s2_50_50",
        "bce_cal_equal_5",
    }
    assert cgm_res["selected_dof"] in ("fixed_0.10", "1_dof", "4_dof")
    assert all(0.05 <= t <= 0.65 for t in cgm_res["selected_thresholds"])


def test_cgm_cal_logprob_eps1e3_negative_values_no_sentinel_leak() -> None:
    """Negative cal_logprob_eps1e3 (< -1.0) must never pick CGM-suppressed classes in bitmask or inference."""
    # Exact Challenger probe: active classes {2, 3} both have logp < -1.0 while suppressed class 0 > -1.0
    p_probe1 = np.array([[[[0.70]], [[0.05]], [[0.15]], [[0.10]]]], dtype=np.float32)
    logp_probe1 = np.log(np.clip(p_probe1, 1e-3, 1.0))
    assert float(logp_probe1[0, 2, 0, 0]) < -1.0 and float(logp_probe1[0, 3, 0, 0]) < -1.0
    gate1 = np.array([[0.01, 0.01, 0.50, 0.50]], dtype=np.float32)  # active bitmask = 12 ({2, 3})

    pred1 = _apply_cgm_probs_np(logp_probe1, gate1, 0.10)
    gate1_t = _make_cgm_gate(torch.from_numpy(gate1), 0.10)
    ref1 = _apply_logit_cgm(torch.from_numpy(logp_probe1), gate1_t)
    assert int(pred1[0, 0, 0]) == 2
    assert np.array_equal(pred1, ref1)

    gt1 = np.array([[[2]]], dtype=np.uint8)
    inter1, union1 = _eval_bitmasks_on_probs(logp_probe1, gt1)
    assert np.array_equal(inter1[0, 12], np.array([0, 0, 1, 0], dtype=np.int64))
    assert np.array_equal(union1[0, 12], np.array([0, 0, 1, 0], dtype=np.int64))

    # Centered fp16 -> cal_logprob_eps1e3 reconstruction probe suppressing class 0
    p_probe2 = np.array([[[[0.65]], [[0.15]], [[0.12]], [[0.08]]]], dtype=np.float32)
    z_probe2 = encode_centered_logprobs_fp16(p_probe2, axis=1)
    logp_probe2 = reconstruct_probs_from_centered(z_probe2, beta=1.0, eps_log=1e-3, axis=1)
    gate2 = np.array([[0.01, 0.40, 0.40, 0.40]], dtype=np.float32)
    pred2 = _apply_cgm_probs_np(logp_probe2, gate2, 0.10)
    ref2 = _apply_logit_cgm(torch.from_numpy(logp_probe2), _make_cgm_gate(torch.from_numpy(gate2), 0.10))
    assert int(pred2[0, 0, 0]) == 1
    assert np.array_equal(pred2, ref2)

    # Full batch parity on cal_logprob_eps1e3 features (< -1.0 active pixels) across CV + bitmasks
    rng = np.random.default_rng(77)
    n_p, h, w = 12, 10, 10
    gt = rng.integers(0, 4, size=(n_p, h, w), dtype=np.uint8)
    raw_p = rng.uniform(0.01, 0.45, size=(n_p, 4, h, w)).astype(np.float32)
    raw_p[:, 0] = 0.70  # bias suppressed class 0 so active classes often have logp < -1.0
    raw_p /= raw_p.sum(axis=1, keepdims=True)
    logp_batch = reconstruct_probs_from_centered(
        encode_centered_logprobs_fp16(raw_p, axis=1), beta=1.0, eps_log=1e-3, axis=1
    )
    assert float(np.max(logp_batch)) <= 0.0 and int(np.sum(logp_batch < -1.0)) > 0

    gate_b = rng.uniform(0.0, 0.7, size=(n_p, 4)).astype(np.float32)
    gate_b[:, 0] = 0.01  # suppress class 0 on most patches
    thr = [0.20, 0.30, 0.40, 0.25]
    pred_np = _apply_cgm_probs_np(logp_batch, gate_b, thr)
    ref_logit = _apply_logit_cgm(torch.from_numpy(logp_batch), _make_cgm_gate(torch.from_numpy(gate_b), thr))
    assert np.array_equal(pred_np, ref_logit)

    inter_bm, union_bm = _eval_bitmasks_on_probs(logp_batch, gt)
    bm = _gate_masks_from_thresholds(gate_b, thr)
    pat_idx = np.arange(n_p) % 4
    ip_dir, up_dir = _accumulate_patient_iu(pred_np, gt, pat_idx, 4)
    assert np.array_equal(ip_dir.sum(axis=0), inter_bm[np.arange(n_p), bm, :].sum(axis=0))
    assert np.array_equal(up_dir.sum(axis=0), union_bm[np.arange(n_p), bm, :].sum(axis=0))

    y_gt = np.stack([np.any(gt == c, axis=(1, 2)) for c in range(4)], axis=1).astype(np.float32)
    s1_p = np.clip(y_gt * 0.75 + 0.1, 0.05, 0.95)
    s2_list = [np.clip(y_gt * 0.70 + 0.12, 0.05, 0.95) for _ in range(4)]
    cgm_res = calibrate_ensemble_cgm_cv(
        fused_probs=logp_batch,
        gt_masks=gt,
        patch_patient_idx=pat_idx,
        folds=[[0], [1], [2], [3]],
        s1_cls_probs=s1_p,
        s2_cls_probs=s2_list,
        y_gt_patches=y_gt,
        arm_weights=[0.25, 0.25, 0.25, 0.25],
    )
    ip_no, up_no = _accumulate_patient_iu(np.argmax(logp_batch, axis=1), gt, pat_idx, 4)
    expected_no_cgm = round(compute_macro_miou_from_iu(ip_no.sum(axis=0), up_no.sum(axis=0)), 6)
    assert cgm_res["no_cgm_val_miou"] == expected_no_cgm
    assert all(0.05 <= t <= 0.65 for t in cgm_res["selected_thresholds"])


def test_publication_stats_bootstrap_perm_wilcoxon_holm() -> None:
    """10,000-bootstrap CI, exact patient-swap permutation test, Wilcoxon zero-diff guard, and m=5 Holm-Bonferroni."""
    rng = np.random.default_rng(42)
    inter_a = rng.integers(80, 120, size=(23, 4), dtype=np.int64)
    union_a = inter_a + rng.integers(20, 40, size=(23, 4), dtype=np.int64)
    inter_b = rng.integers(50, 80, size=(23, 4), dtype=np.int64)
    union_b = inter_b + rng.integers(40, 70, size=(23, 4), dtype=np.int64)

    ci = bootstrap_pooled_miou_ci(inter_a, union_a, n_boot=10000, seed=42)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert compute_macro_miou_from_iu(np.zeros(4), np.zeros(4)) == 0.0

    perm = paired_patient_swap_permutation_test(inter_a, union_a, inter_b, union_b, n_perm=10000, seed=42)
    assert perm["delta_miou"] > 0.1 and perm["p_value"] < 0.01

    perm_same = paired_patient_swap_permutation_test(inter_a, union_a, inter_a, union_a, n_perm=500, seed=42)
    assert perm_same["delta_miou"] == 0.0 and perm_same["p_value"] == 1.0

    # Zero-diff guard in Wilcoxon
    w_zero = paired_wilcoxon_and_dz(np.ones(23) * 0.7, np.ones(23) * 0.7)
    assert w_zero["wilcoxon_p"] == 1.0 and w_zero["cohens_dz"] == 0.0 and w_zero["ties"] == 23

    w_pos = paired_wilcoxon_and_dz(np.linspace(0.75, 0.85, 23), np.linspace(0.65, 0.75, 23))
    assert w_pos["wins"] == 23 and w_pos["losses"] == 0 and w_pos["wilcoxon_p"] < 1e-4

    # m = 5 Holm-Bonferroni check
    adj = holm_bonferroni([0.001, 0.04, 0.03, 0.20, 0.005])
    # sorted: 0.001*5=0.005, 0.005*4=0.02, 0.03*3=0.09, 0.04*2=0.09 (monotone!), 0.20*1=0.20
    assert np.allclose(adj, [0.005, 0.09, 0.09, 0.20, 0.02], atol=1e-9)


def _make_tiny_cohort(root: Path, n_val: int = 32, n_test: int = 32) -> tuple[Path, Path, Path, Path, list[str], list[str]]:
    val_img, val_msk = root / "val" / "img", root / "val" / "mask"
    test_img, test_msk = root / "test" / "img", root / "test" / "mask"
    for d in (val_img, val_msk, test_img, test_msk):
        d.mkdir(parents=True, exist_ok=True)
    val_pids = [f"TCGA-BB-{i:04d}" for i in range(1, 5)]
    test_pids = [f"TCGA-CC-{i:04d}" for i in range(1, 5)]
    rng = np.random.default_rng(11)
    for split_n, pids, im_d, mk_d in (
        (n_val, val_pids, val_img, val_msk),
        (n_test, test_pids, test_img, test_msk),
    ):
        for idx in range(split_n):
            pid = pids[idx % len(pids)]
            fname = f"{pid}_dx1_x{idx}.png"
            arr = rng.integers(30, 220, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr, mode="RGB").save(im_d / fname)
            mk = rng.integers(0, 4, size=(32, 32), dtype=np.uint8)
            mk[:4, :4] = 4
            Image.fromarray(mk, mode="P").save(mk_d / fname)
    return val_img, val_msk, test_img, test_msk, val_pids, test_pids


def test_02c_notebook_validity_and_64patch_cpu_e2e_dryrun(tmp_path: Path) -> None:
    """Validate 02c notebook JSON/AST/INVARIANT-12 and run <20s 64-patch CPU E2E ensemble + INVARIANT-24 check."""
    nb_path = PROJECT_ROOT / "notebooks" / "02c_Stage2_Ensemble_Eval_G4.ipynb"
    nb_text = nb_path.read_text(encoding="utf-8")
    for bad in ("\u2012", "\u2013", "\u2014", "\u2015", "\u2212"):
        assert bad not in nb_text, f"Non-ASCII dash {bad!r} in 02c notebook"
    nb_data = json.loads(nb_text)
    assert nb_data["nbformat"] == 4 and len(nb_data["cells"]) == 6
    for cell in nb_data["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    cell1_src = "".join(nb_data["cells"][0]["source"])
    cell2_src = "".join(nb_data["cells"][1]["source"])
    assert "min_ver=(5, 0, 0)" in cell1_src and "transformers>=5.0.0" in cell1_src
    assert "persistent_workers" not in cell2_src

    # Also verify SegFormer tiny_offline_config checkpoint loading compatibility
    from src.stage2_v2 import load_stage2_v2_checkpoint

    seg_m = SegFormerDualHead(num_classes=4, pretrained=False, tiny_offline_config=True)
    seg_ckpt = tmp_path / "seg_tiny.pth"
    torch.save({**seg_m.model_metadata(), "state_dict": seg_m.state_dict()}, seg_ckpt)
    _, seg_meta = load_stage2_v2_checkpoint(seg_ckpt, device="cpu", strict=True)
    assert seg_meta["arch"] == "segformer_b3"

    v_im, v_mk, te_im, te_mk, v_pids, te_pids = _make_tiny_cohort(tmp_path / "bcss", n_val=32, n_test=32)
    val_ds_full = BCSSStage2ValDataset(v_im, v_mk, patient_ids=v_pids, strict_patient_filter=True)
    test_ds_full = BCSSStage2ValDataset(te_im, te_mk, patient_ids=te_pids, strict_patient_filter=True)
    assert len(val_ds_full) == 32 and len(test_ds_full) == 32

    class _Fast16Dataset(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = list(files)
            self._cache = []
            for p in self.files:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB").resize((16, 16)), dtype=np.float32) / 255.0
                img_t = torch.from_numpy(arr.transpose(2, 0, 1))
                mk_p = p.parent.parent / "mask" / p.name
                with Image.open(mk_p) as m_im:
                    gt_t = torch.from_numpy(np.asarray(m_im.resize((16, 16), Image.NEAREST), dtype=np.int64))
                self._cache.append((img_t, gt_t))

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, idx: int):
            return self._cache[idx]

    val_ds = _Fast16Dataset(val_ds_full.files)
    test_ds = _Fast16Dataset(test_ds_full.files)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # Track when test_loader is touched to prove zero TEST access during VAL calibration
    val_frozen_snapshot: list[dict] = []
    test_iter_calls_before_hook: list[int] = []
    orig_iter = test_ds.__getitem__

    def _guarded_getitem(idx: int):
        if not val_frozen_snapshot:
            test_iter_calls_before_hook.append(idx)
        return orig_iter(idx)

    test_ds.__getitem__ = _guarded_getitem  # type: ignore[method-assign]

    arm_specs = [
        {"name": "arm1_deeplab_cleankd", "family": "cnn", "model": _TinyDummyArm("a1", (0.2, 0.0, 0.0, 0.0))},
        {"name": "arm2_segformer_oeem", "family": "vit", "model": _TinyDummyArm("a2", (0.0, 0.2, 0.0, 0.0))},
        {"name": "arm3_convnext_selftrain", "family": "cnn", "model": _TinyDummyArm("a3", (0.0, 0.0, 0.2, 0.0))},
        {"name": "arm4_dinov2_adapter", "family": "vit", "model": _TinyDummyArm("a4", (0.0, 0.0, 0.0, 0.2))},
    ]

    out_dir = tmp_path / "stage2_results"
    zip_path = tmp_path / "SE374_Stage2_Training_Results.zip"
    t0 = time.perf_counter()
    report = run_stage2_v2_ensemble_eval(
        arm_specs=arm_specs,
        val_loader=val_loader,
        test_loader=test_loader,
        stage1_cls_model=_TinyDummyStage1Cls(),
        output_dir=out_dir,
        zip_path=zip_path,
        work_dir=tmp_path / "ens_work",
        device=torch.device("cpu"),
        n_bootstrap=200,
        n_perm=200,
        seed=42,
        before_test_hook=lambda sel: val_frozen_snapshot.append(sel),
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0, f"64-patch CPU E2E took {elapsed:.2f}s (> 20s)"
    assert len(val_frozen_snapshot) == 1
    assert "val_best_single_name" in val_frozen_snapshot[0]
    assert "val_best_cnn_vit_pair_name" in val_frozen_snapshot[0]
    assert test_iter_calls_before_hook == [], "TEST dataset accessed during VAL phase!"
    assert report["anti_leakage"] == {
        "val_patches": 32,
        "val_patients": 4,
        "test_patches": 32,
        "test_patients": 4,
        "patient_overlap_count": 0,
    }
    assert len(report["test_individual_arms"]) == 4
    for arm_entry in report["test_individual_arms"].values():
        assert "no_cgm" in arm_entry and "cgm_val_calibrated" in arm_entry and "cgm_mode" in arm_entry
    assert len(report["test_pairwise_blends"]) == 6
    assert len(report["test_triplet_blends"]) == 4
    assert len(report["test_confirmatory_stats_holm_m5"]) == 5
    assert "primary_4arm_no_cgm_vs_best_single_no_cgm" in report["test_pure_ensemble_diagnostic"]

    json_p = out_dir / "STAGE2_TRAINING_METRICS.json"
    png_p = out_dir / "stage2_training_loss.png"
    assert json_p.is_file() and png_p.is_file() and zip_path.is_file()
    assert zip_path.stat().st_size < 1_000_000
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert names == {"STAGE2_TRAINING_METRICS.json", "per_patient_test_metrics.csv", "stage2_training_loss.png"}

    # Verify INVARIANT-24 monochrome compliance on generated plot
    auditor = DeliverableAuditor(str(PROJECT_ROOT))
    auditor.audit_single_image_file(str(png_p), caption="Stage-2 v2 Test Comparison")
    assert auditor.issues == [], f"INVARIANT-24 issues on plot: {auditor.issues}"


def test_chunked_streaming_exact_parity_and_single_loader_pass(tmp_path: Path) -> None:
    """Chunked bitmask/TEST/CGM streaming matches dense reference to 0.0 and iterates loader once."""
    rng = np.random.default_rng(101)
    n_p, c_fg, h, w = 15, 4, 12, 12
    num_pat = 3
    gt = rng.integers(0, 4, size=(n_p, h, w), dtype=np.uint8)
    gt[0, :2, :2] = 4
    gt[1, :2, :2] = 255
    pat_idx = np.arange(n_p, dtype=np.int32) % num_pat

    mm_paths = []
    for m in range(4):
        lg = rng.normal(0.0, 1.4, size=(n_p, c_fg, h, w)).astype(np.float32)
        pr = np.exp(lg - lg.max(axis=1, keepdims=True))
        pr /= pr.sum(axis=1, keepdims=True)
        p_dat = tmp_path / f"arm_{m}.dat"
        mm = np.memmap(p_dat, dtype=np.float16, mode="w+", shape=(n_p, c_fg, h, w))
        mm[:] = encode_centered_logprobs_fp16(pr, axis=1)
        mm.flush()
        del mm
        mm_paths.append(p_dat)

    weights = [0.4, 0.25, 0.2, 0.15]
    betas = [0.9, 1.1, 1.2, 0.85]
    y_gt = np.stack([np.any(gt == c, axis=(1, 2)) for c in range(4)], axis=1).astype(np.float32)
    s1_p = np.clip(y_gt * 0.7 + 0.1, 0.05, 0.95)
    s2_list = [np.clip(y_gt * 0.65 + 0.15, 0.05, 0.95) for _ in range(4)]
    folds = [[0], [1], [2]]

    for mode in ("cal_linear", "cal_logprob_eps1e3"):
        fused_dense = _fuse_memmaps_to_probs(mm_paths, (n_p, c_fg, h, w), weights, betas, mode, batch_size=5)
        ibm_dense, ubm_dense = _eval_bitmasks_on_probs(fused_dense, gt, batch_size=5)
        ibm_chunk, ubm_chunk = _eval_bitmasks_from_memmaps(
            mm_paths, (n_p, c_fg, h, w), weights, betas, mode, gt, batch_size=4
        )
        assert np.array_equal(ibm_chunk, ibm_dense) and np.array_equal(ubm_chunk, ubm_dense)

        res_dense = calibrate_ensemble_cgm_cv(
            fused_probs=fused_dense,
            gt_masks=gt,
            patch_patient_idx=pat_idx,
            folds=folds,
            s1_cls_probs=s1_p,
            s2_cls_probs=s2_list,
            y_gt_patches=y_gt,
            arm_weights=weights,
        )
        res_chunk = calibrate_ensemble_cgm_cv(
            fused_probs=None,
            gt_masks=gt,
            patch_patient_idx=pat_idx,
            folds=folds,
            s1_cls_probs=s1_p,
            s2_cls_probs=s2_list,
            y_gt_patches=y_gt,
            arm_weights=weights,
            precomputed_bitmasks=(ibm_chunk, ubm_chunk),
        )
        assert res_chunk == res_dense

        # TEST single-pass streaming vs dense reference
        w_mat = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                weights,
            ],
            dtype=np.float32,
        )
        arm_names = ["arm1", "arm2", "arm3", "arm4"]
        div_stream, ip_stream, up_stream = _stream_test_diversity_and_no_cgm_weights(
            mm_paths, (n_p, c_fg, h, w), gt, pat_idx, num_pat, arm_names, w_mat, betas, mode, batch_size=4
        )
        mms_r = [np.memmap(p, dtype=np.float16, mode="r", shape=(n_p, c_fg, h, w)) for p in mm_paths]
        try:
            dis_ref = _extract_disagreement_and_unanimous(mms_r, gt, pat_idx, num_pat, batch_size=5)
            div_ref = _compute_diversity_and_oracles(dis_ref, mms_r, gt, pat_idx, num_pat, arm_names, batch_size=5)
            ip_ref, up_ref = evaluate_weights_on_disagreement(
                dis_ref["z_dis"], dis_ref["gt_dis"], dis_ref["pat_dis"],
                dis_ref["inter_u"], dis_ref["union_u"], w_mat, betas, prob_mode=mode,
            )
        finally:
            for mm in mms_r:
                if getattr(mm, "_mmap", None) is not None:
                    mm._mmap.close()
        assert div_stream == div_ref
        assert np.array_equal(ip_stream, ip_ref) and np.array_equal(up_stream, up_ref)

        # CGM chunked streaming vs dense _apply_cgm_probs_np (and uint8 output check)
        gate_p = rng.uniform(0.0, 0.8, size=(n_p, 4)).astype(np.float32)
        thr = [0.20, 0.45, 0.35, 0.55]
        pred_dense = _apply_cgm_probs_np(fused_dense, gate_p, thr)
        assert pred_dense.dtype == np.uint8
        ip_cgm_ref, up_cgm_ref = _accumulate_patient_iu(pred_dense, gt, pat_idx, num_pat)
        iu_chunk = _eval_cgm_from_memmaps(
            mm_paths, (n_p, c_fg, h, w), weights, betas, mode, gt, pat_idx, num_pat, gate_p,
            {"cal": thr}, batch_size=4,
        )
        assert np.array_equal(iu_chunk["cal"][0], ip_cgm_ref)
        assert np.array_equal(iu_chunk["cal"][1], up_cgm_ref)

    # Single-iteration spy proving _stream_split_models does not discard next(iter(loader))
    class _SpySingleIterLoader:
        def __init__(self) -> None:
            self.iter_count = 0
            self.dataset = [None] * 4
            self._batches = [
                (torch.rand(2, 3, 8, 8), torch.randint(0, 4, (2, 8, 8), dtype=torch.int64)),
                (torch.rand(2, 3, 8, 8), torch.randint(0, 4, (2, 8, 8), dtype=torch.int64)),
            ]

        def __iter__(self):
            self.iter_count += 1
            return iter(self._batches)

    spy_mm_dir = tmp_path / "spy_mm"
    spy_mm_dir.mkdir(parents=True, exist_ok=True)
    spy_loader = _SpySingleIterLoader()
    res_split = _stream_split_models(
        arm_specs=[{"name": "arm1", "family": "cnn", "model": _TinyDummyArm("a1", (0.2, 0.0, 0.0, 0.0))}],
        loader=spy_loader,  # type: ignore[arg-type]
        arm_tta_modes=["raw_1x"],
        device=torch.device("cpu"),
        memmap_dir=spy_mm_dir,
        split_tag="spy",
        select_tta_on_split=False,
    )
    # 1 pass for gt_all + 1 pass for the single arm = 2 total iterations (NOT 3 from discarded next(iter))
    assert spy_loader.iter_count == 2
    assert res_split["gt_all"].dtype == np.uint8


def test_author_anchor_shared_6view_ms_tta_parity_and_single_pass(tmp_path: Path, monkeypatch) -> None:
    """Shared Author evaluator runs 6-view MS-TTA once per batch and matches evaluator.py identically."""
    class _DummyAuthorStage2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert float(x.min()) >= -1e-5 and float(x.max()) <= 1.0 + 1e-5
            self.forward_calls += 1
            b, _, h, w = x.shape
            out = torch.zeros((b, 5, h, w), dtype=torch.float32, device=x.device)
            # Negative logits on class 0 so cgm_author_mul0 and cgm_neg1e4 diverge meaningfully
            out[:, 0] = -1.2 + 0.4 * x[:, 0]
            out[:, 1] = -2.0 + 0.8 * x[:, 1]
            out[:, 2] = -0.8 + 0.3 * x[:, 2]
            out[:, 3] = -1.5 + 0.5 * (x[:, 0] - x[:, 1])
            return out

    class _DummyS1Expert(nn.Module):
        def __init__(self, probs_vec: tuple[float, float, float, float]) -> None:
            super().__init__()
            self.probs_vec = probs_vec

        def forward_cam(self, x: torch.Tensor):
            b, _, h, w = x.shape
            cams = torch.ones((b, 4, h, w), dtype=torch.float32, device=x.device)
            probs = torch.tensor(self.probs_vec, dtype=torch.float32, device=x.device).unsqueeze(0).expand(b, 4)
            return cams, probs

    author_model = _DummyAuthorStage2()
    s1_a = _DummyS1Expert((0.25, 0.02, 0.30, 0.02))
    s1_b = _DummyS1Expert((0.02, 0.35, 0.02, 0.25))

    p_a2 = tmp_path / "author_s2.pth"
    p_s1_a = tmp_path / "s1_retrained.pth"
    p_s1_b = tmp_path / "s1_official.pth"
    for p in (p_a2, p_s1_a, p_s1_b):
        p.write_bytes(b"stub")

    monkeypatch.setattr("src.stage2_v2.ensemble.build_and_load_deeplabv3plus", lambda **kwargs: author_model)
    monkeypatch.setattr("src.stage2_v2.ensemble.WSSSTissueStage1NetCAM", lambda n_class=4: _DummyS1Expert((0.1, 0.1, 0.1, 0.1)))

    def _fake_load_s1(model_obj, ckpt_path):
        name = Path(ckpt_path).name
        model_obj.probs_vec = s1_a.probs_vec if "retrained" in name else s1_b.probs_vec
        return "zero_one", f"sha_{name}", {}

    monkeypatch.setattr("src.stage2_v2.ensemble.load_stage1_strict", _fake_load_s1)

    rng = np.random.default_rng(7)
    imgs = torch.from_numpy(rng.uniform(-1.5, 1.5, size=(6, 3, 16, 16)).astype(np.float32))
    gts = torch.from_numpy(rng.integers(0, 4, size=(6, 16, 16), dtype=np.int64))
    gts[0, :3, :3] = 4
    gts[1, :3, :3] = 255
    ds = torch.utils.data.TensorDataset(imgs, gts)
    loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    pidx = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)

    shared = _evaluate_author_anchors_shared(
        author_stage2_ckpt=p_a2,
        stage1_ckpts={"anchor_a": p_s1_a, "anchor_b": p_s1_b},
        loader=loader,
        patch_patient_idx=pidx,
        num_patients=3,
        device=torch.device("cpu"),
    )
    # 2 batches * 6 MS-TTA views = 12 forward calls total even with 2 Stage-1 experts
    assert author_model.forward_calls == 12

    ref_a = _eval_author_stage2_split(author_model, s1_a, loader, torch.device("cpu"), "zero_one")
    ref_b = _eval_author_stage2_split(author_model, s1_b, loader, torch.device("cpu"), "zero_one")
    for mode_k in ("raw", "ms_tta", "cgm_author_mul0", "cgm_neg1e4", "ms_tta_cgm_neg1e4"):
        assert shared["anchor_a"][mode_k]["mIoU"] == ref_a[mode_k]["mIoU"]
        assert shared["anchor_b"][mode_k]["mIoU"] == ref_b[mode_k]["mIoU"]
        assert shared["anchor_a"][mode_k]["class_ious"] == ref_a[mode_k]["class_ious"]
        assert shared["anchor_b"][mode_k]["class_ious"] == ref_b[mode_k]["class_ious"]

    single_wrap = _evaluate_author_anchor(p_a2, p_s1_a, loader, pidx, 3, torch.device("cpu"))
    assert single_wrap["ms_tta_cgm_neg1e4"]["mIoU"] == shared["anchor_a"]["ms_tta_cgm_neg1e4"]["mIoU"]


def test_fixed_010_selectable_in_cgm_ladder() -> None:
    """When 1_dof and 4_dof do not beat fixed_0.10 by adoption_delta, chosen_dof must be fixed_0.10."""
    n_p, h, w = 8, 8, 8
    gt = np.zeros((n_p, h, w), dtype=np.uint8)
    for i in range(n_p):
        gt[i, :, :4] = 0
        gt[i, :, 4:] = 1
    probs = np.full((n_p, 4, h, w), 0.05, dtype=np.float32)
    probs[:, 0, :, :4] = 0.85
    probs[:, 1, :, 4:] = 0.85
    # Gate probs >= 0.66 for active classes {0, 1} and <= 0.02 for absent {2, 3}: every threshold <= 0.65 gives exact same bitmask
    gate_p = np.full((n_p, 4), 0.01, dtype=np.float32)
    gate_p[:, 0] = 0.80
    gate_p[:, 1] = 0.80
    y_gt = np.zeros((n_p, 4), dtype=np.float32)
    y_gt[:, 0] = 1.0
    y_gt[:, 1] = 1.0

    res = calibrate_ensemble_cgm_cv(
        fused_probs=probs,
        gt_masks=gt,
        patch_patient_idx=np.arange(n_p) % 4,
        folds=[[0], [1], [2], [3]],
        s1_cls_probs=gate_p,
        s2_cls_probs=[gate_p for _ in range(4)],
        y_gt_patches=y_gt,
        arm_weights=[0.25, 0.25, 0.25, 0.25],
        adoption_delta=0.0005,
    )
    assert res["selected_dof"] == "fixed_0.10"
    assert res["selected_thresholds"] == [0.10, 0.10, 0.10, 0.10]


def test_val_preregistration_matched_cgm_and_author_stats_e2e(tmp_path: Path, monkeypatch) -> None:
    """VAL pre-registration freezes single/pair before TEST and aligns m=5 + secondary Author stats."""
    class _DummyAuthorStage2(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, h, w = x.shape
            out = torch.zeros((b, 5, h, w), dtype=torch.float32, device=x.device)
            out[:, 0] = 0.5
            out[:, 1] = -0.2
            return out

    class _DummyS1Expert(nn.Module):
        def __init__(self, probs_vec: tuple[float, float, float, float]) -> None:
            super().__init__()
            self.probs_vec = probs_vec

        def forward_cam(self, x: torch.Tensor):
            b, _, h, w = x.shape
            cams = torch.ones((b, 4, h, w), dtype=torch.float32, device=x.device)
            probs = torch.tensor(self.probs_vec, dtype=torch.float32, device=x.device).unsqueeze(0).expand(b, 4)
            return cams, probs

    p_a2 = tmp_path / "author_s2.pth"
    p_s1_ret = tmp_path / "stage1_retrained_best.pth"
    p_s1_off = tmp_path / "stage1_checkpoint_trained_on_bcss.pth"
    for p in (p_a2, p_s1_ret, p_s1_off):
        p.write_bytes(b"stub")

    monkeypatch.setattr("src.stage2_v2.ensemble.build_and_load_deeplabv3plus", lambda **kwargs: _DummyAuthorStage2())
    monkeypatch.setattr("src.stage2_v2.ensemble.WSSSTissueStage1NetCAM", lambda n_class=4: _DummyS1Expert((0.2, 0.2, 0.2, 0.2)))
    monkeypatch.setattr(
        "src.stage2_v2.ensemble.load_stage1_strict",
        lambda m_obj, ck_p: (
            setattr(m_obj, "probs_vec", (0.55, 0.55, 0.02, 0.02) if "retrained" in Path(ck_p).name else (0.55, 0.02, 0.55, 0.02))
            or ("zero_one", f"sha_{Path(ck_p).stem}", {})
        ),
    )

    v_im, v_mk, te_im, te_mk, v_pids, te_pids = _make_tiny_cohort(tmp_path / "bcss", n_val=8, n_test=8)
    val_ds_f = BCSSStage2ValDataset(v_im, v_mk, patient_ids=v_pids, strict_patient_filter=True)
    test_ds_f = BCSSStage2ValDataset(te_im, te_mk, patient_ids=te_pids, strict_patient_filter=True)

    class _Fast16(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = list(files)
            self._cache = []
            for p in self.files:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB").resize((16, 16)), dtype=np.float32) / 255.0
                mk_p = p.parent.parent / "mask" / p.name
                with Image.open(mk_p) as m_im:
                    mk = np.asarray(m_im.resize((16, 16), Image.NEAREST), dtype=np.int64)
                self._cache.append((torch.from_numpy(arr.transpose(2, 0, 1)), torch.from_numpy(mk)))

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, idx: int):
            return self._cache[idx]

    captured_plot_entries: list[list] = []
    monkeypatch.setattr(
        "src.stage2_v2.ensemble.save_ensemble_comparison_plot",
        lambda entries, plot_path: (captured_plot_entries.append(list(entries)) or Path(plot_path)),
    )

    val_loader = DataLoader(_Fast16(val_ds_f.files), batch_size=8, shuffle=False, num_workers=0)
    test_ds = _Fast16(test_ds_f.files)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    frozen_sel: list[dict] = []
    test_touched_early: list[int] = []
    orig_get = test_ds.__getitem__

    def _spy_get(idx: int):
        if not frozen_sel:
            test_touched_early.append(idx)
        return orig_get(idx)

    test_ds.__getitem__ = _spy_get  # type: ignore[method-assign]

    arm_specs = [
        {"name": "arm1_deeplab_cleankd", "family": "cnn", "model": _TinyDummyArm("a1", (0.4, 0.0, 0.0, 0.0))},
        {"name": "arm2_segformer_oeem", "family": "vit", "model": _TinyDummyArm("a2", (0.0, 0.4, 0.0, 0.0))},
        {"name": "arm3_convnext_selftrain", "family": "cnn", "model": _TinyDummyArm("a3", (0.0, 0.0, 0.4, 0.0))},
        {"name": "arm4_dinov2_adapter", "family": "vit", "model": _TinyDummyArm("a4", (0.0, 0.0, 0.0, 0.4))},
    ]
    report = run_stage2_v2_ensemble_eval(
        arm_specs=arm_specs,
        val_loader=val_loader,
        test_loader=test_loader,
        stage1_cls_model=_TinyDummyStage1Cls(),
        author_stage2_ckpt=p_a2,
        author_stage1_retrained_ckpt=p_s1_ret,
        author_stage1_official_ckpt=p_s1_off,
        output_dir=tmp_path / "out",
        zip_path=tmp_path / "out.zip",
        work_dir=tmp_path / "work",
        device=torch.device("cpu"),
        n_bootstrap=100,
        n_perm=100,
        seed=42,
        before_test_hook=lambda sel: frozen_sel.append(sel),
    )
    assert test_touched_early == []
    sel = frozen_sel[0]
    assert report["test_best_single_arm"] == sel["val_best_single_name"]
    assert report["test_best_cnn_vit_pair"] == sel["val_best_cnn_vit_pair_name"]
    assert "individual_arm_cgm_policies" in sel and "best_cnn_vit_pair_cgm_policy" in sel
    assert report["test_primary_author_anchor"] == {
        "anchor_key": "author_step0_anchor_7043",
        "mode": "ms_tta_cgm_neg1e4",
    }
    for ak in ("author_step0_anchor_7043", "author_official_pure"):
        assert set(report["test_author_anchors"][ak].keys()) >= {
            "raw", "ms_tta", "cgm_author_mul0", "cgm_neg1e4", "ms_tta_cgm_neg1e4"
        }
        assert "bootstrap_95ci" in report["test_author_anchors"][ak]["ms_tta_cgm_neg1e4"]

    m5 = report["test_confirmatory_stats_holm_m5"]
    assert len(m5) == 5
    # Confirmatory stats must compare against ms_tta_cgm_neg1e4 and match mode of pre-registered arm/pair
    for row in m5[:4]:
        assert row["mode_b"] == "ms_tta_cgm_neg1e4"
    best_s_mode = report["test_individual_arms"][sel["val_best_single_name"]]["cgm_mode"]
    assert m5[2]["mode_a"] == best_s_mode
    assert m5[4]["mode_b"] == best_s_mode

    sec = report["test_secondary_author_anchor_stats"]
    assert {
        "author_step0_anchor_7043__ms_tta_cgm_neg1e4",
        "author_step0_anchor_7043__cgm_author_mul0",
        "author_official_pure__ms_tta_cgm_neg1e4",
        "author_official_pure__cgm_author_mul0",
    }.issubset(set(sec.keys()))
    diag = report["test_pure_ensemble_diagnostic"]["primary_4arm_no_cgm_vs_best_single_no_cgm"]
    assert diag["best_single_no_cgm_name"] == sel["val_best_single_no_cgm_name"]
    assert len(captured_plot_entries) == 1
    assert [e[0] for e in captured_plot_entries[0]] == [
        "Author Anchor", "ARM1", "ARM2", "ARM3", "ARM4", "Best CNN+ViT", "Equal 4-Arm", "Primary 4-Arm"
    ]
