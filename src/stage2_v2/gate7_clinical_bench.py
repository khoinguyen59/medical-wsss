"""Gate-7 Track A Clinical Biomarkers, Survival, GPU Efficiency & LUAD Cross-Organ Benchmark (ASCII '-' only)."""

import gc
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, Dataset

from src.clinical import survival_analysis as surv_mod
from src.models import wsss_inference_methods as wim
from src.stage2_v2 import ensemble as ens
from src.stage2_v2 import evaluator as s2_eval
from src.stage2_v2 import visual_audit as va
from src.stage2_v2.models import load_stage2_v2_checkpoint
from src.training import dataset_loader as dl
from src.training.eval_stage1_full import load_stage1_strict
from src.training.train_stage2 import prepare_stage1_input

# LUAD-HistoSeg [0:TE, 1:NEC, 2:LYM, 3:TAS, 4:BACK] -> BCSS [0:TUM, 1:STR, 2:LYM, 3:NEC, 4:BACK]
LUAD_TO_BCSS_LUT: np.ndarray = np.array([0, 3, 2, 1, 4], dtype=np.int64)
_LEG_KW: Dict[str, Any] = {"frameon": True, "facecolor": "#FFFFFF", "edgecolor": "#999999", "fontsize": 7.8}
_SHORT_LABELS: Dict[str, str] = {"author_stage2": "Author S2", "arm8_deeplab_r101_r2": "Arm-8 DLv3+", "arm6_segformer_b5_r2": "Arm-6 SegF-B5", "arm7_convnext_large_r2": "Arm-7 ConvNeXt", "arm5_dinov2_large_r2": "Arm-5 DINOv2-L", "primary_4arm_ensemble": "4-Arm Ens (no-CGM)"}

def _free_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

def _load_arm(spec: Mapping[str, Any], device: torch.device) -> nn.Module:
    return spec["model"].to(device).eval() if spec.get("model") is not None else load_stage2_v2_checkpoint(spec["ckpt_path"], device=device, strict=True)[0].eval()

def _run_autocast_fwd(mod: nn.Module, inp: torch.Tensor, device: torch.device, amp_dtype: Optional[torch.dtype], use_amp: bool) -> Any:
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        return mod(inp)

def extract_luad_case_id(filepath_or_name: Union[str, Path]) -> str:
    """Extract case/patient ID from LUAD-HistoSeg filenames (TCGA, GDPH_<id>, or <case>-<x>-<y>)."""
    stem = os.path.splitext(os.path.basename(str(filepath_or_name)))[0]
    for pat in (r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", r"^(GDPH[_-][A-Za-z0-9]+)", r"^([A-Za-z0-9]+)-\d+-\d+"):
        if (m := re.search(pat, stem, re.IGNORECASE)):
            return m.group(1).upper()
    return dl.extract_patient_id(stem)

def remap_luad_mask_to_bcss(mask_arr: np.ndarray) -> np.ndarray:
    """Map LUAD-HistoSeg discrete class IDs into BCSS 4-class + BACK space, preserving 255."""
    arr = np.asarray(mask_arr, dtype=np.int64)
    return np.where((arr >= 0) & (arr <= 4), LUAD_TO_BCSS_LUT[np.clip(arr, 0, 4)], 255).astype(np.int64)

class LUADBCSSAlignedDataset(Dataset):
    """Defensive nested LUAD-HistoSeg dataset returning ImageNet tensors + BCSS-mapped GT masks."""

    def __init__(self, img_dir: Union[str, Path], mask_dir: Union[str, Path], target_size: Tuple[int, int] = (224, 224), limit: Optional[int] = None) -> None:
        self.img_dir, self.mask_dir, self.target_size = Path(img_dir), Path(mask_dir), target_size
        mask_map = {p.name: p for p in sorted(self.mask_dir.rglob("*.png"))} if self.mask_dir.is_dir() else {}
        files = [p for p in sorted(self.img_dir.rglob("*.png")) if p.name in mask_map]
        self.files = files[: int(limit)] if (limit is not None and int(limit) > 0) else files
        self.mask_paths = [mask_map[p.name] for p in self.files]
        if not self.files:
            raise FileNotFoundError(f"[LUAD ERROR] Zero paired PNG patches under {self.img_dir} / {self.mask_dir}")
        self._mean, self._std = np.asarray(dl.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1), np.asarray(dl.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_p, mask_p = self.files[idx], self.mask_paths[idx]
        with Image.open(img_p) as im:
            rgb = im.convert("RGB")
            rgb = rgb.resize(self.target_size, Image.BILINEAR) if rgb.size != self.target_size else rgb
            arr = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((arr - self._mean) / self._std), torch.from_numpy(remap_luad_mask_to_bcss(dl._load_mode_p_mask_array(mask_p, target_size=self.target_size))), img_p.name

def compute_cohens_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute exact Cohen's kappa with deterministic guard for degenerate single-class cases."""
    a, b = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if a.size == 0 or a.size != b.size:
        return 0.0
    p_o = float(np.mean(a == b))
    p_e = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in sorted(set(a.tolist()) | set(b.tolist())))
    return (1.0 if abs(p_o - 1.0) < 1e-12 else 0.0) if abs(1.0 - p_e) < 1e-12 else float((p_o - p_e) / (1.0 - p_e))

def compute_bland_altman_and_pearson(pred_vals: Sequence[float], gt_vals: Sequence[float]) -> Dict[str, float]:
    """Compute Pearson r/p, Spearman rho/p, Bland-Altman bias, 95% LoA, MAE, and RMSE."""
    p, g = np.asarray(pred_vals, dtype=np.float64), np.asarray(gt_vals, dtype=np.float64)
    n = int(min(p.size, g.size))
    p, g = p[:n], g[:n]
    if n == 0:
        return {"n": 0, "pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0, "mean_bias": 0.0, "sd_diff": 0.0, "loa_lower": 0.0, "loa_upper": 0.0, "mae": 0.0, "rmse": 0.0}
    diffs = p - g
    bias, sd = float(np.mean(diffs)), (float(np.std(diffs, ddof=1)) if n > 1 else 0.0)
    nonconst = n >= 2 and float(np.std(p)) > 1e-12 and float(np.std(g)) > 1e-12
    r_val, p_val = stats.pearsonr(p, g) if nonconst else (0.0, 1.0)
    rho_val, sp_val = stats.spearmanr(p, g) if nonconst else (0.0, 1.0)
    return {"n": n, "pearson_r": round(float(r_val), 4), "pearson_p": float(p_val), "spearman_rho": round(float(rho_val), 4), "spearman_p": float(sp_val),
            "mean_bias": round(bias, 4), "sd_diff": round(sd, 4), "loa_lower": round(bias - 1.96 * sd, 4), "loa_upper": round(bias + 1.96 * sd, 4),
            "mae": round(float(np.mean(np.abs(diffs))), 4), "rmse": round(float(np.sqrt(np.mean(diffs ** 2))), 4)}

def _new_bw_fig(nrows: int, ncols: int, figsize: Tuple[float, float], dpi: int, **kw: Any) -> Tuple[Any, Any]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, **kw)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", color="#CCCCCC", linewidth=0.8)
    return fig, axes

def render_bland_altman_bw(patient_rows: Sequence[Mapping[str, Any]], model_stats: Mapping[str, Any], primary_model: str, save_path: Path, dpi: int = 300) -> Path:
    """Render 300-DPI Academic Monochrome B&W (#FFFFFF) Bland-Altman & correlation plot for VTP and TSR."""
    fig, axes = _new_bw_fig(2, 2, (10.6, 8.4), dpi)
    for row_i, (metric, label) in enumerate((("vtp", "VTP (%)"), ("tsr", "TSR (%)"))):
        gt_v, pr_v = np.asarray([float(r[f"gt_{metric}"]) for r in patient_rows], dtype=np.float64), np.asarray([float(r[f"{primary_model}_{metric}"]) for r in patient_rows], dtype=np.float64)
        diffs, st, ax_s, ax_b = pr_v - gt_v, model_stats[metric], axes[row_i, 0], axes[row_i, 1]
        ax_s.scatter(gt_v, pr_v, color="#111111", edgecolor="#FFFFFF", s=44, zorder=3)
        ax_s.plot([0, 100], [0, 100], linestyle="--", color="#555555", linewidth=1.2, label="Identity (y=x)")
        ax_s.set(xlim=(-3, 103), ylim=(-3, 103), xlabel=f"Pathologist GT {label}", ylabel=f"Gate-7 {label}", title=f"{metric.upper()} Correlation (r={st['pearson_r']:.3f}, p={st['pearson_p']:.2e})")
        ax_s.legend(loc="lower right", **_LEG_KW)
        ax_b.scatter(0.5 * (gt_v + pr_v), diffs, color="#222222", marker="s", edgecolor="#FFFFFF", s=42, zorder=3)
        for val, col, ls, lbl in ((st["mean_bias"], "#000000", "-", f"Bias={st['mean_bias']:+.2f}%"), (st["loa_upper"], "#666666", "--", f"+1.96 SD={st['loa_upper']:+.2f}%"), (st["loa_lower"], "#666666", "--", f"-1.96 SD={st['loa_lower']:+.2f}%"), (0.0, "#AAAAAA", ":", None)):
            ax_b.axhline(val, color=col, linestyle=ls, linewidth=1.3 if lbl else 1.0, label=lbl)
        y_hi, y_lo = max(float(st["loa_upper"]), float(np.max(diffs)) if diffs.size else 1.0, 1.0), min(float(st["loa_lower"]), float(np.min(diffs)) if diffs.size else -1.0, -1.0)
        span = max(y_hi - y_lo, 4.0)
        ax_b.set(ylim=(y_lo - 0.16 * span, y_hi + 0.36 * span), xlabel=f"Mean GT & Gate-7 {label}", ylabel=f"Gate-7 - GT {label}", title=f"{metric.upper()} Bland-Altman (MAE={st['mae']:.2f}%)")
        ax_b.legend(loc="upper right", **_LEG_KW)
    fig.suptitle(f"Patient-Level Clinical Biomarker Agreement ({primary_model}, n={len(patient_rows)} patients)", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    va._atomic_save_figure(fig, save_path, dpi=dpi)
    return save_path

def render_km_survival_bw(km_by_source: Mapping[str, Mapping[str, Any]], primary_model: str, save_path: Path, dpi: int = 300) -> Path:
    """Render 300-DPI Academic Monochrome B&W (#FFFFFF) side-by-side Kaplan-Meier survival curves."""
    fig, axes = _new_bw_fig(1, 2, (11.2, 4.9), dpi, sharey=True)
    for ax, (src_key, title) in zip(axes, (("gt", "Pathologist GT TSR (50% cut-off)"), (primary_model, f"Gate-7 AI ({primary_model}) TSR (50% cut-off)"))):
        entry = km_by_source.get(src_key, {})
        for grp_k, ls, col, lbl in (("fit_high", "-", "#000000", "Stroma-Rich High Risk"), ("fit_low", "--", "#666666", "Stroma-Poor Low Risk")):
            fit, durs, evts = entry.get(grp_k, {}), entry.get("durations_high" if grp_k == "fit_high" else "durations_low", []), entry.get("events_high" if grp_k == "fit_high" else "events_low", [])
            tl, sp = fit.get("timeline", []), fit.get("survival_probability", [])
            if tl and sp:
                ax.step(tl, sp, where="post", linestyle=ls, color=col, linewidth=2.0, label=f"{lbl} (n={len(durs)}, evt={fit.get('events_observed', 0)})")
                for d, e in zip(durs, evts):
                    if int(e) == 0:
                        ax.plot(d, sp[max(0, min(int(np.searchsorted(tl, d, side="right")) - 1, len(sp) - 1))], marker="+", markersize=7, markeredgewidth=1.4, color=col)
        lr = entry.get("logrank", {})
        ax.text(0.96, 0.06, f"Log-Rank p = {float(lr.get('p_value', 1.0)):.4f}\nChi-sq = {float(lr.get('test_statistic', 0.0)):.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6, bbox={"boxstyle": "round,pad=0.38", "facecolor": "#F5F5F5", "edgecolor": "#888888"})
        ax.set(ylim=(-0.05, 1.06), xlabel="Overall Survival Time (Months)", title=title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower left", **_LEG_KW)
    axes[0].set_ylabel("Survival Probability S(t)", fontsize=9.5)
    fig.suptitle("Kaplan-Meier Overall Survival Stratification (TCGA-BRCA Held-Out Test Cohort)", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    va._atomic_save_figure(fig, save_path, dpi=dpi)
    return save_path

def render_gpu_pareto_bw(eff_results: Mapping[str, Any], save_path: Path, dpi: int = 300) -> Path:
    """Render 300-DPI Academic Monochrome B&W (#FFFFFF) Pareto chart (Latency vs real BCSS mIoU)."""
    fig, ax = _new_bw_fig(1, 1, (8.8, 5.8), dpi)
    rows = sorted([(k, v) for k, v in eff_results.get("models", {}).items() if v.get("reference_miou_pct") is not None], key=lambda kv: (float(kv[1]["latency_ms_per_patch"]), float(kv[1]["reference_miou_pct"])))
    markers, pts, frontier, best_y = ("o", "s", "^", "D", "v", "P"), [], [], -1e9
    lats, mious = [float(v["latency_ms_per_patch"]) for _, v in rows] or [0.0, 10.0], [float(v["reference_miou_pct"]) for _, v in rows] or [70.0, 73.0]
    xs, ys = max(max(lats) - min(lats), 1.0), max(max(mious) - min(mious), 1.0)
    ax.set(xlim=(max(0.0, min(lats) - 0.08 * xs), max(lats) + 0.28 * xs), ylim=(min(mious) - 0.16 * ys, max(mious) + 0.22 * ys))
    offsets = [(12, -10), (14, -18), (14, 18 if (len(mious) > 3 and mious[2] >= mious[3]) else -22), (14, -22 if (len(mious) > 3 and mious[2] >= mious[3]) else 18), (14, -18), (-12, -26)]
    for idx, (m_key, info) in enumerate(rows):
        lat, miou, vram, fps = float(info["latency_ms_per_patch"]), float(info["reference_miou_pct"]), float(info["peak_vram_mb"]), float(info["throughput_fps"])
        pts.append((lat, miou))
        ax.scatter(lat, miou, s=max(55.0, min(240.0, 55.0 + vram * 0.04)), marker=markers[idx % len(markers)], facecolor="#222222" if ("ens" in m_key or "arm5" in m_key) else "#FFFFFF", edgecolor="#000000", linewidth=1.4, zorder=4)
        right = idx == len(rows) - 1 and len(rows) > 1
        dx, dy = (-12, -26) if right else offsets[idx % len(offsets)]
        ax.annotate(f"{_SHORT_LABELS.get(m_key, m_key)}\n({lat:.1f} ms | {fps:.0f} FPS | {vram:.0f} MB)", (lat, miou), textcoords="offset points", xytext=(dx, dy), ha="right" if right else "left", va="center", fontsize=7.5, bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.6}, arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.7})
    for lat, miou in sorted(pts, key=lambda t: (t[0], -t[1])):
        if miou > best_y:
            frontier.append((lat, miou))
            best_y = miou
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], linestyle="--", color="#444444", linewidth=1.4, label="Pareto Frontier")
        ax.legend(loc="lower right", **_LEG_KW)
    ax.set(xlabel="1x Forward Latency per Patch (ms, lower is faster)", ylabel="BCSS TEST TTA 4-Class mIoU (%)", title=f"Gate-7 GPU Efficiency Pareto ({eff_results.get('device_name', 'CPU')})")
    fig.tight_layout()
    va._atomic_save_figure(fig, save_path, dpi=dpi)
    return save_path

def _patient_bio_from_counts(c: Sequence[int]) -> Tuple[float, float, Dict[str, Any]]:
    vtp, tsr, sp = surv_mod.compute_viable_tumor_percentage(c[0], sum(c)), surv_mod.compute_tumor_stroma_ratio(c[0], c[1]), surv_mod.compute_stroma_proportion(c[0], c[1])
    return vtp, tsr, surv_mod.stratify_cap_risk(tsr, stroma_proportion=sp, threshold=50.0)

def compute_gate7_clinical_biomarkers(patient_ids: Sequence[str], raw_rgbs: Sequence[np.ndarray], gt_masks: np.ndarray, model_preds: Mapping[str, np.ndarray], output_dir: Union[str, Path], primary_model: str = "ens4_cgm", clinical_manifest: Optional[Mapping[str, Mapping[str, Any]]] = None, dpi: int = 300) -> Dict[str, Any]:
    """Compute per-patient VTP/TSR agreement, CAP concordance, and KM survival (zero GT leakage)."""
    if not model_preds:
        raise ValueError("model_preds must contain at least one model prediction array.")
    out_p, manifest, model_keys = Path(output_dir), (clinical_manifest if clinical_manifest is not None else surv_mod.TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA), list(model_preds.keys())
    out_p.mkdir(parents=True, exist_ok=True)
    primary_model, by_pid = (primary_model if primary_model in model_preds else model_keys[0]), {}
    for idx, pid in enumerate(patient_ids):
        by_pid.setdefault(str(pid), []).append(idx)
    patient_rows: List[Dict[str, Any]] = []
    for pid, idxs in sorted(by_pid.items()):
        gt_vtp, gt_tsr, gt_cap = _patient_bio_from_counts([sum(int(np.sum(gt_masks[i] == c)) for i in idxs) for c in range(dl.NUM_FOREGROUND)])
        surv = manifest.get(pid, {})
        row: Dict[str, Any] = {"patient_id": pid, "n_patches": len(idxs), "gt_vtp": round(gt_vtp, 3), "gt_tsr": round(gt_tsr, 3), "gt_cap_risk": gt_cap["risk_level"], "gt_cap_code": int(gt_cap["risk_code"]), "overall_survival_months": surv.get("overall_survival_months"), "vital_status": surv.get("vital_status")}
        for mk in model_keys:
            guarded = [wim.MultiMethodWSSSEngineV2.apply_pure_glass_guard(model_preds[mk][i], raw_rgbs[i], background_class_idx=4) for i in idxs]
            p_vtp, p_tsr, p_cap = _patient_bio_from_counts([sum(int(np.sum(g == c)) for g in guarded) for c in range(dl.NUM_FOREGROUND)])
            row.update({f"{mk}_vtp": round(p_vtp, 3), f"{mk}_tsr": round(p_tsr, 3), f"{mk}_cap_risk": p_cap["risk_level"], f"{mk}_cap_code": int(p_cap["risk_code"]), f"{mk}_cap_match": int(p_cap["risk_code"] == gt_cap["risk_code"])})
        patient_rows.append(row)
    km_svc, km_by_source, matched = surv_mod.KaplanMeierSurvivalAnalysis(use_lifelines_if_available=True), {}, [r for r in patient_rows if r["overall_survival_months"] is not None and r["vital_status"] is not None]
    for src in ["gt"] + model_keys:
        col = "gt_cap_code" if src == "gt" else f"{src}_cap_code"
        dh, eh = [float(r["overall_survival_months"]) for r in matched if int(r[col]) == 1], [int(r["vital_status"]) for r in matched if int(r[col]) == 1]
        dl_v, el = [float(r["overall_survival_months"]) for r in matched if int(r[col]) == 0], [int(r["vital_status"]) for r in matched if int(r[col]) == 0]
        lr = km_svc.logrank_test(dh, eh, dl_v, el, label_a="High Risk", label_b="Low Risk") if (dh and dl_v) else {"test_statistic": 0.0, "p_value": 1.0, "note": "Single risk stratum"}
        km_by_source[src] = {"n_high_risk": len(dh), "n_low_risk": len(dl_v), "events_high_risk": int(sum(eh)), "events_low_risk": int(sum(el)), "logrank": lr, "fit_high": km_svc.fit(dh, eh, label=f"{src}_High"), "fit_low": km_svc.fit(dl_v, el, label=f"{src}_Low"), "durations_high": dh, "events_high": eh, "durations_low": dl_v, "events_low": el}
    per_model_stats: Dict[str, Any] = {}
    for mk in model_keys:
        pairs = [(int(r["gt_cap_code"]), int(r[f"{mk}_cap_code"])) for r in patient_rows if int(r["gt_cap_code"]) in (0, 1) and int(r[f"{mk}_cap_code"]) in (0, 1)]
        yt, yp = [x[0] for x in pairs], [x[1] for x in pairs]
        per_model_stats[mk] = {"vtp": compute_bland_altman_and_pearson([r[f"{mk}_vtp"] for r in patient_rows], [r["gt_vtp"] for r in patient_rows]), "tsr": compute_bland_altman_and_pearson([r[f"{mk}_tsr"] for r in patient_rows], [r["gt_tsr"] for r in patient_rows]),
                               "cap_concordance": {"n_measurable": len(pairs), "n_concordant": int(sum(a == b for a, b in pairs)), "accuracy": round(float(np.mean(np.asarray(yt) == np.asarray(yp))) if pairs else 0.0, 4), "cohens_kappa": round(compute_cohens_kappa(yt, yp), 4)}, "km_logrank_p": float(km_by_source[mk]["logrank"]["p_value"])}
    csv_p = out_p / "patient_clinical_risk_summary.csv"
    va._atomic_write_csv(csv_p, list(patient_rows[0].keys()) if patient_rows else ["patient_id"], patient_rows)
    ba_png, km_png = render_bland_altman_bw(patient_rows, per_model_stats[primary_model], primary_model, out_p / "bland_altman_tsr_vtp_bw.png", dpi=dpi), render_km_survival_bw(km_by_source, primary_model, out_p / "km_survival_tsr_bw.png", dpi=dpi)
    summary = {"n_patients": len(patient_rows), "n_patches": len(patient_ids), "primary_model": primary_model, "gt_free_glass_guard": "MultiMethodWSSSEngineV2.apply_pure_glass_guard", "models": per_model_stats, "kaplan_meier": km_by_source, "patients": patient_rows, "artifacts": [csv_p.name, ba_png.name, km_png.name]}
    va._atomic_write_json(out_p / "gate7_clinical_biomarkers.json", summary)
    return summary

def _benchmark_forward_fn(fn: Any, n_patches: int, param_m: float, device: torch.device, n_warmup: int, n_iters: int) -> Dict[str, Any]:
    with torch.no_grad():
        for _ in range(max(1, n_warmup)):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(max(1, n_iters)):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - t0, 1e-9)
    n_tot = max(1, int(n_patches)) * max(1, int(n_iters))
    return {"latency_ms_per_patch": round((elapsed * 1000.0) / n_tot, 3), "throughput_fps": round(n_tot / elapsed, 2), "peak_vram_mb": round(float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0, 2), "param_count_m": round(float(param_m), 2)}

def benchmark_gate7_gpu_efficiency(arm_specs: Sequence[Mapping[str, Any]], author_stage2_ckpt: Optional[Union[str, Path]], sample_imgs: torch.Tensor, device: torch.device, amp_dtype: Optional[torch.dtype], output_dir: Union[str, Path], reference_mious: Optional[Mapping[str, float]] = None, n_warmup: int = 5, n_iters: int = 20, author_model_override: Optional[nn.Module] = None, dpi: int = 300) -> Dict[str, Any]:
    """Measure real GPU latency (ms/patch), FPS, and Peak VRAM (MB) with Pareto B&W chart (zero fake mIoU)."""
    out_p, ref, results = Path(output_dir), dict(reference_mious or {}), {}
    out_p.mkdir(parents=True, exist_ok=True)
    _free_cuda(device)
    inp_imnet, b_sz, h_in, w_in, use_amp = sample_imgs.to(device), int(sample_imgs.shape[0]), int(sample_imgs.shape[2]), int(sample_imgs.shape[3]), (device.type == "cuda" and amp_dtype is not None)
    inp_01 = prepare_stage1_input(inp_imnet, stage1_norm_mode="zero_one", images_are_imagenet_norm=True)
    if author_model_override is not None or author_stage2_ckpt is not None:
        auth_m = author_model_override.to(device).eval() if author_model_override is not None else wim.build_and_load_deeplabv3plus(author_stage2_ckpt, device=device, strict=True)
        results["author_stage2"] = {**_benchmark_forward_fn(lambda m=auth_m: m(inp_01), b_sz, sum(p.numel() for p in auth_m.parameters()) / 1e6, device, n_warmup, n_iters), "effective_resolution": f"{h_in}x{w_in}", "reference_miou_pct": round(float(ref["author_step0"]), 2) if ref.get("author_step0") is not None else None}
        if author_model_override is None:
            del auth_m
            _free_cuda(device)
    lat_sum, vram_max, param_sum = 0.0, 0.0, 0.0
    for spec in arm_specs:
        name, mod = str(spec["name"]).strip().lower(), _load_arm(spec, device)
        pm, sc, ref_v = sum(p.numel() for p in mod.parameters()) / 1e6, float(getattr(mod, "internal_scale", 1.0)), ref.get(name, ref.get("arm5_cgm") if "arm5" in name else None)
        r_m = {**_benchmark_forward_fn(lambda m=mod: _run_autocast_fwd(m, inp_imnet, device, amp_dtype, use_amp), b_sz, pm, device, n_warmup, n_iters), "effective_resolution": f"{int(round(h_in * sc))}x{int(round(w_in * sc))}", "reference_miou_pct": round(float(ref_v), 2) if ref_v is not None else None}
        results[name], lat_sum, vram_max, param_sum = r_m, lat_sum + r_m["latency_ms_per_patch"], max(vram_max, r_m["peak_vram_mb"]), param_sum + r_m["param_count_m"]
        if spec.get("model") is None:
            del mod
            _free_cuda(device)
    if arm_specs:
        loaded: List[nn.Module] = []
        try:
            loaded = [_load_arm(sp, device) for sp in arm_specs]
            ens_fn = lambda: torch.stack([F.softmax((o["seg_logits"] if isinstance(o := _run_autocast_fwd(m, inp_imnet, device, amp_dtype, use_amp), dict) else o)[:, : dl.NUM_FOREGROUND].float(), dim=1) for m in loaded], dim=0).mean(dim=0)  # noqa: E731
            e_m = {**_benchmark_forward_fn(ens_fn, b_sz, param_sum, device, n_warmup, n_iters), "measurement_mode": "resident_multi_arm_forward"}
        except RuntimeError:
            e_m = {"latency_ms_per_patch": round(lat_sum, 3), "throughput_fps": round(1000.0 / max(lat_sum, 1e-9), 2), "peak_vram_mb": round(vram_max, 2), "param_count_m": round(param_sum, 2), "measurement_mode": "sequential_oom_fallback"}
        finally:
            loaded.clear()
            _free_cuda(device)
        results["primary_4arm_ensemble"] = {**e_m, "reference_miou_pct": round(float(ens_ref), 2) if (ens_ref := ref.get("ens4_no_cgm") if ref.get("ens4_no_cgm") is not None else ref.get("ens4_cgm")) is not None else None}
    payload = {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "amp_dtype": str(amp_dtype), "batch_size": b_sz, "n_warmup": int(n_warmup), "n_iters": int(n_iters), "models": results}
    va._atomic_write_json(out_p / "gate7_gpu_efficiency.json", payload)
    render_gpu_pareto_bw(payload, out_p / "gpu_pareto_efficiency_bw.png", dpi=dpi)
    return payload

def _load_s1_expert(override: Optional[nn.Module], ckpt: Optional[Union[str, Path]], device: torch.device) -> Tuple[nn.Module, str]:
    if override is not None:
        return override.to(device).eval(), "zero_one"
    if ckpt is None:
        raise RuntimeError("[STAGE2 ERROR] stage1_cls_ckpt is required when stage1_model_override is None.")
    mod = wim.WSSSTissueStage1NetCAM(n_class=dl.NUM_FOREGROUND)
    return mod.to(device).eval(), str(load_stage1_strict(mod, ckpt)[0] or "zero_one")

def _stream_gate7_predictions(arm_specs: Sequence[Mapping[str, Any]], loader: DataLoader, stage1_cls_ckpt: Optional[Union[str, Path]], author_stage1_ckpt: Optional[Union[str, Path]], author_stage2_ckpt: Optional[Union[str, Path]], device: torch.device, amp_dtype: Optional[torch.dtype], selected_ttas: Optional[Sequence[str]] = None, track: str = "track_a_clean", stage1_model_override: Optional[nn.Module] = None, author_stage1_model_override: Optional[nn.Module] = None, author_model_override: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Stream frozen Gate-7 Track A + Author predictions via compact FP16 disk memmaps."""
    if not arm_specs:
        raise ValueError("arm_specs must contain at least one Gate-7 arm specification.")
    for sp in arm_specs:
        ens.validate_track_a_arm_spec(sp, track=track)
    arm_names = [str(sp["name"]).strip().lower() for sp in arm_specs]
    ttas, betas = (list(selected_ttas) if selected_ttas is not None else [va.GATE7_SELECTED_TTAS_BY_ARM.get(n, "ms_3_hflip") for n in arm_names]), [va.GATE7_FULL_VAL_BETAS_BY_ARM.get(n, 0.7) for n in arm_names]
    w_raw = np.asarray([va.GATE7_UNIFORM_WEIGHTS_BY_ARM.get(n, 1.0 / len(arm_names)) for n in arm_names], dtype=np.float32)
    weights, (filenames, patient_ids, _, raw_rgbs, gt_all) = w_raw / max(float(w_raw.sum()), 1e-6), va._collect_test_samples_from_loader(loader)
    n, h, w, arm5_idx = int(gt_all.shape[0]), int(gt_all.shape[1]), int(gt_all.shape[2]), next((i for i, nm in enumerate(arm_names) if "arm5" in nm), len(arm_names) - 1)
    tmp_obj, mm_paths, arm_cls, s1_m, a_s1_m, auth_m = tempfile.TemporaryDirectory(prefix="g7_clin_bench_"), [], [], None, None, None
    try:
        for idx, spec in enumerate(arm_specs):
            mm_p, mod = Path(tmp_obj.name) / f"arm_{idx}.dat", _load_arm(spec, device)
            mm_paths.append(mm_p)
            mm, cls_a, off = np.memmap(mm_p, dtype=np.float16, mode="w+", shape=(n, dl.NUM_FOREGROUND, h, w)), np.zeros((n, dl.NUM_FOREGROUND), dtype=np.float32), 0
            with torch.no_grad():
                for batch in loader:
                    imgs = s2_eval._extract_img_gt(batch)[0].to(device, non_blocking=True)
                    b, (p_t, c_d) = int(imgs.shape[0]), ens.apply_tta_probs(mod, imgs, ttas[idx], device, amp_dtype=amp_dtype)
                    mm[off : off + b], cls_a[off : off + b], off = ens.encode_centered_logprobs_fp16(p_t.detach().cpu().numpy().astype(np.float32), axis=1), c_d["tta"].detach().cpu().numpy().astype(np.float32), off + b
            ens._close_memmap(mm)
            arm_cls.append(cls_a)
            if spec.get("model") is None:
                del mod
                _free_cuda(device)
        s1_m, s1_norm = _load_s1_expert(stage1_model_override, stage1_cls_ckpt, device)
        a_s1_m, a_s1_norm = _load_s1_expert(author_stage1_model_override, author_stage1_ckpt, device) if (author_stage1_model_override is not None or author_stage1_ckpt is not None) else (s1_m, s1_norm)
        auth_m = author_model_override.to(device).eval() if author_model_override is not None else (wim.build_and_load_deeplabv3plus(author_stage2_ckpt, device=device, strict=True) if author_stage2_ckpt else None)
        s1_cls, pred_auth, off = np.zeros((n, dl.NUM_FOREGROUND), dtype=np.float32), (np.zeros((n, h, w), dtype=np.uint8) if auth_m is not None else None), 0
        with torch.no_grad():
            for batch in loader:
                imgs = s2_eval._extract_img_gt(batch)[0].to(device, non_blocking=True)
                b, in01 = int(imgs.shape[0]), prepare_stage1_input(imgs, stage1_norm_mode="zero_one", images_are_imagenet_norm=True)
                s1_cls[off : off + b] = s1_m.forward_cam(in01 if s1_norm == "zero_one" else prepare_stage1_input(imgs, s1_norm, True))[1].float().cpu().numpy()
                if auth_m is not None and pred_auth is not None:
                    as1_p = a_s1_m.forward_cam(in01 if a_s1_norm == "zero_one" else prepare_stage1_input(imgs, a_s1_norm, True))[1].float()
                    lg1 = (o["seg_logits"] if isinstance(o := auth_m(in01), dict) else o)[:, : dl.NUM_FOREGROUND].float()
                    pred_auth[off : off + b] = s2_eval._apply_prob_cgm(s2_eval._ms_tta_probs(auth_m, in01, lg1, (0.75, 1.0, 1.25), device, amp_dtype=None), s2_eval._make_cgm_gate(as1_p, 0.10)).astype(np.uint8)
                off += b
        pred_arm5, pred_ens4_no, pred_ens4 = np.zeros((n, h, w), dtype=np.uint8), np.zeros((n, h, w), dtype=np.uint8), np.zeros((n, h, w), dtype=np.uint8)
        bm5, bm4 = ens._gate_masks_from_thresholds(arm_cls[arm5_idx], va.GATE7_ARM5_SELF_CGM_THRESHOLDS), ens._gate_masks_from_thresholds(ens._combine_5_expert_probs(s1_cls, arm_cls, weights, "balanced_s1_s2_40_60"), va.GATE7_4ARM_CGM_THRESHOLDS)
        arm_inter, arm_union = np.zeros((len(arm_names), dl.NUM_FOREGROUND), dtype=np.int64), np.zeros((len(arm_names), dl.NUM_FOREGROUND), dtype=np.int64)
        mms = [np.memmap(p, dtype=np.float16, mode="r", shape=(n, dl.NUM_FOREGROUND, h, w)) for p in mm_paths]
        try:
            for st in range(0, n, 256):
                en, z_idx = min(n, st + 256), np.zeros(min(n, st + 256) - st, dtype=np.int32)
                for m_i, mm_i in enumerate(mms):
                    ip_m, up_m = va._accumulate_patient_iu(np.argmax(np.asarray(mm_i[st:en], dtype=np.float16), axis=1).astype(np.uint8), gt_all[st:en], z_idx, 1)
                    arm_inter[m_i], arm_union[m_i] = arm_inter[m_i] + ip_m[0], arm_union[m_i] + up_m[0]
                pred_arm5[st:en] = va._apply_bitmask_argmax_np(np.asarray(ens._fuse_memmap_chunk([mms[arm5_idx]], st, en, np.asarray([1.0], dtype=np.float32), [1.0], "uncal_linear"), dtype=np.float32), bm5[st:en])
                f4 = ens._fuse_memmap_chunk(mms, st, en, weights, betas, "cal_logprob_eps1e3", eps_log=1e-3)
                pred_ens4_no[st:en], pred_ens4[st:en] = np.argmax(f4, axis=1).astype(np.uint8), va._apply_bitmask_argmax_np(f4, bm4[st:en])
        finally:
            for mm in mms:
                ens._close_memmap(mm)
        arm_mious = {nm: round(float(ens.summarize_iu(arm_inter[i], arm_union[i])["mIoU"]) * 100.0, 2) for i, nm in enumerate(arm_names)}
        return {"filenames": filenames, "patient_ids": patient_ids, "raw_rgbs": raw_rgbs, "gt_all": gt_all, "arm_pooled_miou_pct": arm_mious, "preds": {"arm5_cgm": pred_arm5, "ens4_no_cgm": pred_ens4_no, "ens4_cgm": pred_ens4, **({"author_step0": pred_auth} if pred_auth is not None else {})}}
    finally:
        s1_m = a_s1_m = auth_m = None
        tmp_obj.cleanup()
        _free_cuda(device)

def evaluate_gate7_luad_cross_organ(arm_specs: Sequence[Mapping[str, Any]], luad_loader: DataLoader, stage1_cls_ckpt: Optional[Union[str, Path]], author_stage1_ckpt: Optional[Union[str, Path]], author_stage2_ckpt: Optional[Union[str, Path]], output_dir: Union[str, Path], device: torch.device, amp_dtype: Optional[torch.dtype] = None, **kwargs: Any) -> Dict[str, Any]:
    """Run full zero-shot cross-organ LUAD-HistoSeg validation in BCSS-aligned 4-class space."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    streamed = _stream_gate7_predictions(arm_specs, luad_loader, stage1_cls_ckpt, author_stage1_ckpt, author_stage2_ckpt, device, amp_dtype, **kwargs)
    gt_all, pids = streamed["gt_all"], [extract_luad_case_id(fn) for fn in streamed["filenames"]]
    uniq = sorted(set(pids))
    p_idx, models_out = np.asarray([uniq.index(p) for p in pids], dtype=np.int32), {}
    for mk, pred_arr in streamed["preds"].items():
        ip, up = va._accumulate_patient_iu(pred_arr, gt_all, p_idx, len(uniq))
        summ, pm = ens.summarize_iu(ip.sum(axis=0), up.sum(axis=0)), ens.compute_per_patient_miou(ip, up)
        models_out[mk] = {**summ, "patient_macro_miou": round(float(np.mean(pm)) if pm.size else 0.0, 4), "patient_macro_std": round(float(np.std(pm, ddof=1)) if pm.size > 1 else 0.0, 4)}
    payload = {"benchmark": "zero_shot_bcss_to_luad_histoseg", "label_mapping": "LUAD [0:TE, 1:NEC, 2:LYM, 3:TAS, 4:BACK] -> BCSS [0:TUM, 1:STR, 2:LYM, 3:NEC, 4:BACK]", "n_patches": int(gt_all.shape[0]), "n_patients": len(uniq), "models": models_out}
    va._atomic_write_json(out_p / "gate7_luad_cross_organ.json", payload)
    print(f"[LUAD CROSS-ORGAN] n_patches={payload['n_patches']} n_patients={payload['n_patients']} ens4_cgm_mIoU={models_out['ens4_cgm']['mIoU']:.4f}", flush=True)
    return payload

def run_gate7_clinical_and_luad_benchmark(arm_specs: Sequence[Mapping[str, Any]], bcss_test_loader: DataLoader, luad_test_loader: Optional[DataLoader], stage1_cls_ckpt: Optional[Union[str, Path]], author_stage1_ckpt: Optional[Union[str, Path]], author_stage2_ckpt: Optional[Union[str, Path]], output_dir: Union[str, Path], zip_path: Union[str, Path], device: torch.device, amp_dtype: Optional[torch.dtype] = None, selected_ttas: Optional[Sequence[str]] = None, track: str = "track_a_clean", n_warmup: int = 5, n_iters: int = 20, dpi: int = 300, stage1_model_override: Optional[nn.Module] = None, author_stage1_model_override: Optional[nn.Module] = None, author_model_override: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Execute full Gate-7 Clinical Biomarkers, Survival, GPU Efficiency & LUAD Suite + ZIP export."""
    out_p, skw = Path(output_dir), dict(selected_ttas=selected_ttas, track=track, stage1_model_override=stage1_model_override, author_stage1_model_override=author_stage1_model_override, author_model_override=author_model_override)
    out_p.mkdir(parents=True, exist_ok=True)
    bcss = _stream_gate7_predictions(arm_specs, bcss_test_loader, stage1_cls_ckpt, author_stage1_ckpt, author_stage2_ckpt, device, amp_dtype, **skw)
    uniq_b = sorted(set(bcss["patient_ids"]))
    p_idx_b = np.asarray([uniq_b.index(p) for p in bcss["patient_ids"]], dtype=np.int32)
    bcss_mious = {**bcss["arm_pooled_miou_pct"], **{mk: round(float(ens.summarize_iu(*[x.sum(axis=0) for x in va._accumulate_patient_iu(pr, bcss["gt_all"], p_idx_b, len(uniq_b))])["mIoU"]) * 100.0, 2) for mk, pr in bcss["preds"].items()}}
    clin_res = compute_gate7_clinical_biomarkers(bcss["patient_ids"], bcss["raw_rgbs"], bcss["gt_all"], bcss["preds"], out_p, primary_model="ens4_cgm", dpi=dpi)
    gpu_res = benchmark_gate7_gpu_efficiency(arm_specs, author_stage2_ckpt, s2_eval._extract_img_gt(next(iter(bcss_test_loader)))[0][:4], device, amp_dtype, out_p, reference_mious=bcss_mious, n_warmup=n_warmup, n_iters=n_iters, author_model_override=author_model_override, dpi=dpi)
    luad_res = evaluate_gate7_luad_cross_organ(arm_specs, luad_test_loader, stage1_cls_ckpt, author_stage1_ckpt, author_stage2_ckpt, out_p, device, amp_dtype, **skw) if luad_test_loader is not None else None
    shutil.copy2(out_p / "bland_altman_tsr_vtp_bw.png", out_p / "stage2_training_loss.png")
    combined = {"benchmark_suite": "gate7_track_a_clinical_gpu_luad", "track": track, "arm_order": [str(s["name"]).strip().lower() for s in arm_specs], "bcss_pooled_miou_pct": bcss_mious, "clinical_biomarkers": clin_res, "gpu_efficiency": gpu_res, "luad_cross_organ": luad_res}
    for fn in ("GATE7_CLINICAL_BENCH_METRICS.json", "STAGE2_TRAINING_METRICS.json"):
        va._atomic_write_json(out_p / fn, combined)
    va._atomic_write_zip(Path(zip_path), {p.name: p for p in sorted(out_p.iterdir()) if p.is_file()})
    return combined
