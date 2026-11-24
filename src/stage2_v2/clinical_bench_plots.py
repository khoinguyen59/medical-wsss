"""Gate-7 Clinical Benchmark Dataset, Stats, Calibration & 300-DPI Publication Plots (ASCII '-' only)."""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats
from torch.utils.data import Dataset

from src.stage2_v2 import visual_audit as va
from src.training import dataset_loader as dl
from src.training.eval_stage1_full import compute_file_sha256

# LUAD-HistoSeg [0:TE, 1:NEC, 2:LYM, 3:TAS, 4:BACK] -> BCSS [0:TUM, 1:STR, 2:LYM, 3:NEC, 4:BACK]
LUAD_TO_BCSS_LUT: np.ndarray = np.array([0, 3, 2, 1, 4], dtype=np.int64)
GUIDELINE_ATTRIBUTION: str = "Mesker et al. 2007 / van Pelt et al. 2018 UNITED (not CAP)"
_LEG_KW: Dict[str, Any] = {"frameon": True, "facecolor": "#FFFFFF", "edgecolor": "#999999", "fontsize": 7.6}
_SHORT_LABELS: Dict[str, str] = {
    "author_stage2": "Author S2",
    "author_stage2_fp32": "Author S2 (FP32)",
    "arm8_deeplab_r101_r2": "Arm-8 DLv3+",
    "arm6_segformer_b5_r2": "Arm-6 SegF-B5",
    "arm7_convnext_large_r2": "Arm-7 ConvNeXt",
    "arm5_dinov2_large_r2": "Arm-5 DINOv2-L",
    "arm5_cgm": "Arm-5 + CGM",
    "primary_4arm_ensemble": "4-Arm Ens (no-CGM)",
}


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
        self._mean = np.asarray(dl.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(dl.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

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


def compute_lins_ccc(pred_vals: Sequence[float], gt_vals: Sequence[float]) -> float:
    """Compute Lin's Concordance Correlation Coefficient (Lin 1989) with zero-variance guard."""
    p, g = np.asarray(pred_vals, dtype=np.float64), np.asarray(gt_vals, dtype=np.float64)
    n = int(min(p.size, g.size))
    if n < 2:
        return 0.0
    p, g = p[:n], g[:n]
    mu_p, mu_g = float(np.mean(p)), float(np.mean(g))
    denom = float(np.var(p, ddof=0)) + float(np.var(g, ddof=0)) + (mu_p - mu_g) ** 2
    return 0.0 if denom <= 1e-12 else float((2.0 * float(np.mean((p - mu_p) * (g - mu_g)))) / denom)


def compute_bland_altman_and_pearson(pred_vals: Sequence[float], gt_vals: Sequence[float]) -> Dict[str, float]:
    """Compute Pearson r/p, Spearman rho/p, Lin's CCC, Bland-Altman bias, 95% LoA, MAE, and RMSE."""
    p, g = np.asarray(pred_vals, dtype=np.float64), np.asarray(gt_vals, dtype=np.float64)
    n = int(min(p.size, g.size))
    p, g = p[:n], g[:n]
    if n == 0:
        return {"n": 0, "pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0, "lins_ccc": 0.0, "mean_bias": 0.0, "sd_diff": 0.0, "loa_lower": 0.0, "loa_upper": 0.0, "mae": 0.0, "rmse": 0.0}
    diffs = p - g
    bias, sd = float(np.mean(diffs)), (float(np.std(diffs, ddof=1)) if n > 1 else 0.0)
    nonconst = n >= 2 and float(np.std(p)) > 1e-12 and float(np.std(g)) > 1e-12
    r_val, p_val = stats.pearsonr(p, g) if nonconst else (0.0, 1.0)
    rho_val, sp_val = stats.spearmanr(p, g) if nonconst else (0.0, 1.0)
    ccc_val = compute_lins_ccc(p, g) if nonconst else 0.0
    return {
        "n": n, "pearson_r": round(float(r_val), 4), "pearson_p": float(p_val),
        "spearman_rho": round(float(rho_val), 4), "spearman_p": float(sp_val), "lins_ccc": round(float(ccc_val), 4),
        "mean_bias": round(bias, 4), "sd_diff": round(sd, 4), "loa_lower": round(bias - 1.96 * sd, 4),
        "loa_upper": round(bias + 1.96 * sd, 4), "mae": round(float(np.mean(np.abs(diffs))), 4), "rmse": round(float(np.sqrt(np.mean(diffs ** 2))), 4),
    }


class PixelCalibrationAccumulator:
    """Streaming 15-bin ECE, multi-class Brier, and NLL accumulator strictly over valid_fg (0..3)."""

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = int(n_bins)
        self.bin_cnt = np.zeros(self.n_bins, dtype=np.int64)
        self.bin_acc, self.bin_conf = np.zeros(self.n_bins, dtype=np.float64), np.zeros(self.n_bins, dtype=np.float64)
        self.brier_sum, self.nll_sum, self.total_px = 0.0, 0.0, 0

    def update(self, probs_b4hw: np.ndarray, gt_bhw: np.ndarray) -> None:
        p_arr, g_arr = np.asarray(probs_b4hw, dtype=np.float32), np.asarray(gt_bhw, dtype=np.int64)
        valid = (g_arr >= 0) & (g_arr < dl.NUM_FOREGROUND)
        if not np.any(valid):
            return
        p_v = np.transpose(p_arr, (0, 2, 3, 1))[valid].astype(np.float64)
        p_v = p_v / np.maximum(np.sum(p_v, axis=1, keepdims=True), 1e-12)
        g_v = g_arr[valid]
        n_v = int(g_v.size)
        self.total_px += n_v
        conf, pred = np.max(p_v, axis=1), np.argmax(p_v, axis=1)
        b_idx = np.clip((conf * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
        self.bin_cnt += np.bincount(b_idx, minlength=self.n_bins)
        self.bin_acc += np.bincount(b_idx, weights=(pred == g_v).astype(np.float64), minlength=self.n_bins)
        self.bin_conf += np.bincount(b_idx, weights=conf, minlength=self.n_bins)
        p_true = p_v[np.arange(n_v), g_v]
        self.brier_sum += float(np.sum(np.sum(p_v ** 2, axis=1) - 2.0 * p_true + 1.0))
        self.nll_sum += float(np.sum(-np.log(np.clip(p_true, 1e-7, 1.0))))

    def compute(self) -> Dict[str, Any]:
        if self.total_px <= 0:
            return {"n_valid_pixels": 0, "n_bins": self.n_bins, "ece": 0.0, "brier": 0.0, "nll": 0.0}
        nz = self.bin_cnt > 0
        ece = float(np.sum(np.abs(self.bin_acc[nz] - self.bin_conf[nz])) / float(self.total_px))
        return {"n_valid_pixels": int(self.total_px), "n_bins": self.n_bins, "ece": round(ece, 6), "brier": round(self.brier_sum / float(self.total_px), 6), "nll": round(self.nll_sum / float(self.total_px), 6)}


def resolve_provenance_metadata(
    arm_specs: Sequence[Mapping[str, Any]], stage1_cls_ckpt: Optional[Union[str, Path]] = None,
    author_stage1_ckpt: Optional[Union[str, Path]] = None, author_stage2_ckpt: Optional[Union[str, Path]] = None,
    gdc_manifest_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Resolve current git ref and SHA-256 checksums for all checkpoints and manifests."""
    git_ref, git_sha = os.environ.get("STAGE2_GIT_REF", "").strip(), ""
    try:
        res = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=5)
        if res.returncode == 0:
            git_sha = res.stdout.strip()
    except Exception:
        pass
    def _sha(p: Optional[Union[str, Path]]) -> str:
        return "in_memory_or_none" if p is None else (compute_file_sha256(Path(p)) if Path(p).is_file() else f"missing:{Path(p).name}")
    return {
        "git_ref": git_ref or (git_sha[:12] if git_sha else "unknown"), "git_commit_sha": git_sha,
        "checkpoint_sha256": {
            "arms": {str(sp["name"]).strip().lower(): _sha(sp.get("ckpt_path")) for sp in arm_specs},
            "stage1_cls": _sha(stage1_cls_ckpt), "author_stage1": _sha(author_stage1_ckpt),
            "author_stage2": _sha(author_stage2_ckpt), "gdc_clinical_manifest": _sha(gdc_manifest_path),
        },
    }


def _new_bw_fig(nrows: int, ncols: int, figsize: Tuple[float, float], dpi: int, **kw: Any) -> Tuple[Any, Any]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, **kw)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("#FFFFFF")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, linestyle=":", color="#CCCCCC", linewidth=0.8)
    return fig, axes


def render_bland_altman_bw(patient_rows: Sequence[Mapping[str, Any]], model_stats: Mapping[str, Any], primary_model: str, save_path: Path, dpi: int = 300) -> Path:
    """Render 300-DPI Academic Monochrome B&W (#FFFFFF) Bland-Altman & correlation plot for VTP and TSR."""
    fig, axes = _new_bw_fig(2, 2, (10.8, 8.6), dpi)
    for row_i, (metric, label) in enumerate((("vtp", "VTP (%)"), ("tsr", "TSR (%)"))):
        gt_v = np.asarray([float(r[f"gt_{metric}"]) for r in patient_rows], dtype=np.float64)
        pr_v = np.asarray([float(r[f"{primary_model}_{metric}"]) for r in patient_rows], dtype=np.float64)
        diffs, st, ax_s, ax_b = pr_v - gt_v, model_stats[metric], axes[row_i, 0], axes[row_i, 1]
        ax_s.scatter(gt_v, pr_v, color="#111111", edgecolor="#FFFFFF", s=44, zorder=3)
        ax_s.plot([0, 100], [0, 100], linestyle="--", color="#555555", linewidth=1.2, label="Identity (y=x)")
        ccc_s = f", CCC={st['lins_ccc']:.3f}" if "lins_ccc" in st else ""
        ax_s.set(xlim=(-3, 103), ylim=(-3, 103), xlabel=f"Pathologist GT {label}", ylabel=f"Gate-7 {label}", title=f"{metric.upper()} Agreement (r={st['pearson_r']:.3f}{ccc_s}, p={st['pearson_p']:.1e})")
        ax_s.legend(loc="lower right", **_LEG_KW)
        ax_b.scatter(0.5 * (gt_v + pr_v), diffs, color="#222222", marker="s", edgecolor="#FFFFFF", s=42, zorder=3)
        for val, col, ls, lbl in ((st["mean_bias"], "#000000", "-", f"Bias={st['mean_bias']:+.2f}%"), (st["loa_upper"], "#666666", "--", f"+1.96 SD={st['loa_upper']:+.2f}%"), (st["loa_lower"], "#666666", "--", f"-1.96 SD={st['loa_lower']:+.2f}%"), (0.0, "#AAAAAA", ":", None)):
            ax_b.axhline(val, color=col, linestyle=ls, linewidth=1.3 if lbl else 1.0, label=lbl)
        y_hi, y_lo = max(float(st["loa_upper"]), float(np.max(diffs)) if diffs.size else 1.0, 1.0), min(float(st["loa_lower"]), float(np.min(diffs)) if diffs.size else -1.0, -1.0)
        span = max(y_hi - y_lo, 4.0)
        ax_b.set(ylim=(y_lo - 0.16 * span, y_hi + 0.38 * span), xlabel=f"Mean GT & Gate-7 {label}", ylabel=f"Gate-7 - GT {label}", title=f"{metric.upper()} Bland-Altman (MAE={st['mae']:.2f}%)")
        ax_b.legend(loc="upper right", **_LEG_KW)
    fig.suptitle(f"Patient-Level Clinical Biomarker Agreement ({primary_model}, n={len(patient_rows)} | Mesker 2007 / UNITED 2018)", fontsize=10.8, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    va._atomic_save_figure(fig, save_path, dpi=dpi)
    return save_path


def render_km_survival_bw(km_by_source: Mapping[str, Mapping[str, Any]], primary_model: str, save_path: Path, dpi: int = 300) -> Path:
    """Render 300-DPI Academic Monochrome B&W (#FFFFFF) side-by-side Kaplan-Meier survival curves."""
    fig, axes = _new_bw_fig(1, 2, (11.4, 5.0), dpi, sharey=True)
    for ax, (src_key, title) in zip(axes, (("gt", "Pathologist GT - Mesker 2007 / UNITED 2018 (50% TSR)"), (primary_model, f"Gate-7 AI ({primary_model}) - Mesker 2007 / UNITED 2018 (50% TSR)"))):
        entry = km_by_source.get(src_key, {})
        for grp_k, ls, col, lbl in (("fit_high", "-", "#000000", "Stroma-Rich High Risk"), ("fit_low", "--", "#666666", "Stroma-Poor Low Risk")):
            fit = entry.get(grp_k, {})
            durs = entry.get("durations_high" if grp_k == "fit_high" else "durations_low", [])
            evts = entry.get("events_high" if grp_k == "fit_high" else "events_low", [])
            tl, sp = fit.get("timeline", []), fit.get("survival_probability", [])
            if tl and sp:
                ax.step(tl, sp, where="post", linestyle=ls, color=col, linewidth=2.0, label=f"{lbl} (n={len(durs)}, evt={fit.get('events_observed', 0)})")
                for d, e in zip(durs, evts):
                    if int(e) == 0:
                        ax.plot(d, sp[max(0, min(int(np.searchsorted(tl, d, side="right")) - 1, len(sp) - 1))], marker="+", markersize=7, markeredgewidth=1.4, color=col)
        lr = entry.get("logrank", {})
        ax.text(0.96, 0.06, f"Log-Rank p = {float(lr.get('p_value', 1.0)):.4f}\nChi-sq = {float(lr.get('test_statistic', 0.0)):.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4, bbox={"boxstyle": "round,pad=0.38", "facecolor": "#F5F5F5", "edgecolor": "#888888"})
        ax.set(ylim=(-0.05, 1.06), xlabel="Overall Survival Time (Months, Live NCI GDC)", title=title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower left", **_LEG_KW)
    axes[0].set_ylabel("Survival Probability S(t)", fontsize=9.5)
    fig.suptitle("Kaplan-Meier Overall Survival Stratification (TCGA-BRCA Test | Mesker 2007 / UNITED 2018 50% TSR)", fontsize=10.8, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    va._atomic_save_figure(fig, save_path, dpi=dpi)
    return save_path


def _plot_pareto_panel(fig: Any, ax: Any, models_map: Mapping[str, Mapping[str, Any]], xlabel: str, ylabel: str, title: str) -> None:
    """Populate a single monochrome Pareto axis with greedy zero-collision label placement."""
    rows = sorted([(k, v) for k, v in models_map.items() if v.get("reference_miou_pct") is not None], key=lambda kv: (float(kv[1]["latency_ms_per_patch"]), float(kv[1]["reference_miou_pct"])))
    markers = ("o", "s", "^", "D", "v", "P", "X")
    lats, mious = [float(v["latency_ms_per_patch"]) for _, v in rows] or [0.0, 10.0], [float(v["reference_miou_pct"]) for _, v in rows] or [70.0, 73.0]
    xs, ys = max(max(lats) - min(lats), 1.0), max(max(mious) - min(mious), 1.0)
    p01, p23 = (len(mious) > 1 and mious[0] >= mious[1]), (len(mious) > 3 and mious[2] >= mious[3])
    xlim = (max(0.0, min(lats) - 0.08 * xs), max(lats) + 0.30 * xs)
    ax.set(xlim=xlim, ylim=(min(mious) - 0.24 * ys, max(mious) + 0.26 * ys), xlabel=xlabel, ylabel=ylabel, title=title)
    pts = [(float(v["latency_ms_per_patch"]), float(v["reference_miou_pct"])) for _, v in rows]
    for i, (m_key, info) in enumerate(rows):
        lat, miou = pts[i]
        if float(info.get("latency_ms_std", 0.0) or 0.0) > 0.0:
            ax.errorbar(lat, miou, xerr=float(info["latency_ms_std"]), fmt="none", ecolor="#555555", elinewidth=1.0, capsize=2.5, zorder=3)
        ax.scatter(lat, miou, s=max(52.0, min(220.0, 52.0 + float(info.get("peak_vram_mb", 0.0)) * 0.038)), marker=markers[i % len(markers)], facecolor="#222222" if ("ens" in m_key or "arm5" in m_key) else "#FFFFFF", edgecolor="#000000", linewidth=1.35, zorder=4)
    best_y, frontier = -1e9, []
    for la, mi in sorted(pts, key=lambda t: (t[0], -t[1])):
        if mi > best_y:
            frontier.append((la, mi))
            best_y = mi
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], linestyle="--", color="#444444", linewidth=1.35, label="Pareto Frontier")
        ax.legend(loc="lower right", **_LEG_KW)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_box, leg = ax.get_window_extent(rend), ax.get_legend()
    placed = [leg.get_window_extent(rend)] if leg is not None else []
    offsets = [(14, 24 if p01 else -18), (14, -18 if p01 else 24), (14, 24 if p23 else -22), (14, -22 if p23 else 24), (14, -18), (-12, -26)]
    cands = [(14, 24, "left"), (14, -18, "left"), (14, 46, "left"), (14, -40, "left"), (-12, -26, "right"), (-12, 24, "right"), (-12, 46, "right"), (-12, -46, "right"), (14, 68, "left"), (14, -62, "left"), (-12, 68, "right"), (-12, -62, "right")]
    for idx, (m_key, info) in enumerate(rows):
        lat, miou, vram, fps = float(info["latency_ms_per_patch"]), float(info["reference_miou_pct"]), float(info.get("peak_vram_mb", 0.0)), float(info.get("throughput_fps", 0.0))
        vtag = f" {int(info['tta_views'])}x" if info.get("tta_views") else ""
        pref = (-12, -26, "right") if (idx == len(rows) - 1 and len(rows) > 1 and (lat - xlim[0]) > 0.52 * (xlim[1] - xlim[0])) else (*offsets[idx % len(offsets)], "left")
        ann = ax.annotate(f"{_SHORT_LABELS.get(m_key, m_key)}{vtag}\n({lat:.1f} ms | {fps:.0f} FPS | {vram:.0f} MB)", (lat, miou), textcoords="offset points", xytext=pref[:2], ha=pref[2], va="center", fontsize=7.2, bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.5}, arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.7})
        scored = []
        for ci, (dx, dy, ha) in enumerate([pref] + cands):
            ann.set(position=(dx, dy), ha=ha)
            fig.canvas.draw()
            bb = ann.get_bbox_patch().get_window_extent(rend)
            oob, ovl = int(bb.x0 < ax_box.x0 or bb.x1 > ax_box.x1 or bb.y0 < ax_box.y0 or bb.y1 > ax_box.y1), sum(bb.overlaps(p) for p in placed)
            scored.append(((oob, ovl, ci), (dx, dy, ha), bb))
            if (oob, ovl) == (0, 0):
                break
        _, (bx, by, bha), bbb = min(scored, key=lambda x: x[0])
        ann.set(position=(bx, by), ha=bha)
        placed.append(bbb)


def render_gpu_pareto_bw(eff_results: Mapping[str, Any], save_path: Path, dpi: int = 300) -> Path:
    """Render 300-DPI Academic Monochrome B&W (#FFFFFF) Pareto chart (2-panel Raw 1x + Deployed TTA, or legacy 1-panel)."""
    dev_nm, raw_map, tta_map = str(eff_results.get("device_name", "CPU")), eff_results.get("raw_1x"), eff_results.get("deployed_tta")
    has_two = isinstance(raw_map, Mapping) and isinstance(tta_map, Mapping) and any(v.get("reference_miou_pct") is not None for v in raw_map.values()) and any(v.get("reference_miou_pct") is not None for v in tta_map.values())
    if has_two:
        fig, axes = _new_bw_fig(1, 2, (14.8, 5.8), dpi)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        _plot_pareto_panel(fig, axes[0], raw_map, "1x Forward Latency per Patch (ms, lower is faster)", "BCSS TEST 1x Raw 4-Class mIoU (%)", "(a) Matched 1x Raw Pareto (Same AMP Precision)")
        _plot_pareto_panel(fig, axes[1], tta_map, "Deployed TTA+CGM Latency per Patch (ms, lower is faster)", "BCSS TEST Deployed TTA 4-Class mIoU (%)", "(b) Matched Deployed TTA+CGM Pareto")
        fig.suptitle(f"Gate-7 GPU Efficiency Pareto ({dev_nm})", fontsize=11.5, fontweight="bold")
    else:
        fig, ax = _new_bw_fig(1, 1, (8.8, 5.8), dpi)
        fig.tight_layout()
        _plot_pareto_panel(fig, ax, eff_results.get("models", {}), "1x Forward Latency per Patch (ms, lower is faster)", "BCSS TEST TTA 4-Class mIoU (%)", f"Gate-7 GPU Efficiency Pareto ({dev_nm})")
    va._atomic_save_figure(fig, save_path, dpi=dpi)
    return save_path


def render_luad_overlays_sheet(raw_rgbs: Sequence[np.ndarray], gt_masks: np.ndarray, preds_by_model: Mapping[str, np.ndarray], filenames: Sequence[str], case_ids: Sequence[str], save_path: Path, k: int = 4, dpi: int = 300) -> List[str]:
    """Render 300-DPI LUAD-HistoSeg zero-shot visual overlay sheet across 4 diverse cases (nearest-neighbor)."""
    n = int(gt_masks.shape[0])
    if n == 0:
        return []
    scored = sorted([(sum(1 for c in range(dl.NUM_FOREGROUND) if int((gt_masks[i] == c).sum()) >= 128), int(((gt_masks[i] >= 0) & (gt_masks[i] < dl.NUM_FOREGROUND)).sum()), str(filenames[i]), i) for i in range(n)], key=lambda t: (-t[0], -t[1], t[2]))
    chosen = va._select_distinct_patient_indices([t[3] for t in scored], case_ids, k=min(int(k), n))
    avail = [c for c in (("raw", "1. LUAD H&E (10x)"), ("gt", "2. LUAD GT (BCSS Map)"), ("author_step0", "3. Author Step-0"), ("arm5_no_cgm", "4. Arm-5 DINOv2-L"), ("ens4_uncal_linear", "5. 4-Arm Linear"), ("ens4_cgm", "6. 4-Arm LogProb+CGM")) if c[0] in ("raw", "gt") or c[0] in preds_by_model]
    fig, axes = plt.subplots(len(chosen), len(avail), figsize=(3.1 * len(avail), 3.3 * len(chosen)), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("LUAD-HistoSeg Zero-Shot Cross-Organ Segmentation Overlays (BCSS-Aligned 4-Class)", fontsize=11.8, fontweight="bold", y=0.992)
    for r_i, idx in enumerate(chosen):
        rgb, gt = raw_rgbs[idx], gt_masks[idx]
        for c_i, (ck, ttl) in enumerate(avail):
            ax = axes[r_i, c_i]
            panel, sub = (rgb, f"{case_ids[idx]} | {Path(filenames[idx]).stem[:18]}") if ck == "raw" else ((va.render_gt_overlay_with_roi_contour(rgb, gt), "GT (dashed BACK=4)") if ck == "gt" else (va.render_overlay_rgb(rgb, preds_by_model[ck][idx]), f"mIoU={va.compute_patch_canonical_metrics(preds_by_model[ck][idx], gt)['miou_gt_present']:.3f}"))
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(ttl, fontsize=9.0, fontweight="bold", pad=4)
            ax.set_xlabel(sub, fontsize=7.6, labelpad=2)
    handles = [mpatches.Patch(facecolor=np.asarray(va.CLINICAL_OVERLAY_PALETTE[c]) / 255.0, edgecolor="#222222", label=lbl) for c, lbl in enumerate(("TUM (TE)", "STR (TAS)", "LYM", "NEC", "BACK"))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True, facecolor="#FFFFFF", edgecolor="#888888", fontsize=8.2, bbox_to_anchor=(0.5, 0.002))
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965))
    va._atomic_save_figure(fig, save_path, dpi=dpi)
    return [str(filenames[i]) for i in chosen]
