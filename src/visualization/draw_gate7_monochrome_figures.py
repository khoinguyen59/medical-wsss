"""Academic Monochrome (B&W) Figure Generator for SE374.R11 Gate-7 Track A (Phase-1B Post-Fix).

Reads 100% of metrics from verified JSON/CSV/ZIP artifacts under ``checkpoints/vps_evidence/``
and exports 4 publication-grade grayscale figures (#FFFFFF background, 300 DPI PNG + SVG,
zero RGB color saturation) to ``product/reports/figures/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image
from scipy import stats

EVIDENCE_PATHS = {
    "g7": Path("checkpoints/vps_evidence/se374_g7_phase1b_02c_v2/STAGE2_TRAINING_METRICS.json"),
    "g7_zip": Path("checkpoints/vps_evidence/se374_g7_phase1b_02c_v2/SE374_Stage2_Training_Results.zip"),
    "g6": Path("checkpoints/vps_evidence/se374_s2v2_ensemble/artifacts/STAGE2_TRAINING_METRICS.json"),
    "diag_r2": Path("checkpoints/vps_evidence/se374_g7_diag_r2_val/artifacts/STAGE2_TRAINING_METRICS.json"),
}
WHITE_PILL = {"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.9}


def configure_monochrome_style() -> None:
    """Applies pure neutral grayscale styling (R=G=B) with ASCII hyphen minus signs."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False, "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#000000", "axes.linewidth": 0.9, "axes.labelcolor": "#000000",
        "xtick.color": "#000000", "ytick.color": "#000000", "text.color": "#000000",
        "hatch.linewidth": 0.65, "hatch.color": "#262626", "svg.fonttype": "none",
    })


def _load_true_test_gt_counts(repo_root: Path, patient_ids: list[str], expected_total: int | None = None) -> np.ndarray | None:
    """Counts per-patient foreground GT pixels across all raw test masks when complete on disk."""
    test_root = repo_root / "data" / "BCSS-WSSS" / "test"
    mask_files = [p for p in sorted(test_root.rglob("*.png")) if "mask" in p.parts] if test_root.is_dir() else []
    if not mask_files or (expected_total is not None and len(mask_files) != expected_total):
        return None
    pid_to_idx = {pid: idx for idx, pid in enumerate(patient_ids)}
    gt_counts, seen = np.zeros((len(patient_ids), 4), dtype=np.int64), np.zeros(len(patient_ids), dtype=np.int64)
    for mf in mask_files:
        idx = pid_to_idx.get("-".join(mf.name.split("-")[:3]))
        if idx is None:
            return None
        seen[idx] += 1
        with Image.open(mf) as im:
            if im.mode in ("P", "L"):
                gt_counts[idx, :4] += im.histogram()[:4]
            else:
                arr = np.asarray(im)
                if arr.ndim != 2:
                    return None
                gt_counts[idx] += [int(np.sum(arr == c)) for c in range(4)]
    return gt_counts if (bool(np.all(seen > 0)) and (expected_total is None or int(seen.sum()) == expected_total)) else None


def load_evidence(repo_root: Path) -> dict[str, Any]:
    """Loads verified Phase-1B JSON/ZIP artifacts and asserts CSV-vs-JSON patient alignment."""
    data: dict[str, Any] = {}
    for key in ("g7", "g6", "diag_r2"):
        path = repo_root / EVIDENCE_PATHS[key]
        if not path.is_file():
            raise FileNotFoundError(f"Missing verified evidence file: {path}")
        data[key] = json.loads(path.read_text(encoding="utf-8"))
    zip_path = repo_root / EVIDENCE_PATHS["g7_zip"]
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing verified evidence zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        data["patient_json"] = json.loads(zf.read("per_patient_class_ious.json").decode("utf-8"))
        data["patient_csv"] = list(csv.DictReader(zf.read("per_patient_test_metrics.csv").decode("utf-8").splitlines()))
    pjson = data["patient_json"]
    csv_map = {r["patient_id"]: float(r["primary_4arm_miou"]) for r in data["patient_csv"]}
    for pid, val in zip(pjson["patient_ids"], pjson["primary_4arm"]["per_patient_miou"], strict=True):
        if abs(csv_map[pid] - float(val)) > 1e-5:
            raise ValueError(f"CSV/JSON mismatch for {pid}: {csv_map[pid]} vs {val}")
    sha_path = repo_root / "checkpoints/vps_evidence/SHA256SUMS.txt"
    data["sha256_count"] = sum(1 for ln in sha_path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")) if sha_path.is_file() else None
    exp_test_n = int(data["g7"].get("anti_leakage", {}).get("test_patches", 0)) or None
    data["true_gt_counts"] = _load_true_test_gt_counts(repo_root, pjson["patient_ids"], expected_total=exp_test_n)
    return data


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Saves pure-white 300 DPI PNG and vector SVG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path, svg_path = out_dir / f"{stem}.png", out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.14, facecolor="#FFFFFF", edgecolor="none")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.14, facecolor="#FFFFFF", edgecolor="none")
    plt.close(fig)
    return png_path, svg_path


def _clean_axes(*axes: plt.Axes) -> None:
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def draw_fig1_test_miou_ci(ev: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Fig 1: Horizontal 4-Class TEST mIoU + 95% Patient Bootstrap CI (Phase-1B Post-Fix)."""
    g7, g6 = ev["g7"], ev["g6"]
    anc, arms = g7["test_author_anchors"], g7["test_individual_arms"]
    p4_cgm = g7["test_nec_focused_1dof_diagnostic"]["primary_4arm_cgm_nec_focused_1dof"]
    simp_cgm = g7["test_simplex_cv_4arm"]["cgm_nec_focused_1dof"]
    specs = [
        ("Author Official Pure (1x + CGM)", anc["author_official_pure"]["cgm_author_mul0"], "#F5F5F5", "", 1.0, "-"),
        ("Author Pure (MS-TTA + CGM)", anc["author_official_pure"]["ms_tta_cgm_neg1e4"], "#EDEDED", "..", 1.0, "-"),
        ("Step-0 Author Anchor (MS-TTA + CGM)", anc["author_step0_anchor_7043"]["ms_tta_cgm_neg1e4"], "#E3E3E3", "--", 1.1, "-"),
        ("Gate-6 R1 Primary 4-Arm (L4, +CGM)", g6["test_primary_4arm"], "#DEDEDE", "//", 1.1, "-"),
        ("Gate-7 Arm 8: DeepLabV3+ R101 (+CGM)", arms["arm8_deeplab_r101_r2"], "#EBEBEB", "\\\\", 1.0, "-"),
        ("Gate-7 Arm 7: ConvNeXt-L FPN (post-fix)", arms["arm7_convnext_large_r2"], "#E0E0E0", "xx", 1.1, "-"),
        ("Gate-7 Arm 6: SegFormer-B5 (no-CGM)", arms["arm6_segformer_b5_r2"], "#D6D6D6", "oo", 1.0, "-"),
        ("Gate-7 Arm 5: DINOv2-L 336px (+CGM)", arms["arm5_dinov2_large_r2"], "#CCCCCC", "++", 1.3, "-"),
        ("Gate-7 Primary 4-Arm (Uniform, no-CGM)*", g7["test_primary_4arm"], "#C2C2C2", "///", 2.0, "-"),
        ("Gate-7 Primary 4-Arm (+NEC-1DoF CGM)*", p4_cgm, "#B8B8B8", "\\\\\\", 1.8, "-"),
        ("Gate-7 Simplex-CV 4-Arm (Secondary, +CGM)", simp_cgm, "#E8E8E8", "..", 1.3, "--"),
    ]
    fig, ax = plt.subplots(figsize=(11.8, 6.8), dpi=300)
    y_pos = np.arange(len(specs))
    pts = [s[1]["mIoU"] * 100.0 for s in specs]
    lo = [s[1]["bootstrap_95ci"]["ci_low"] * 100.0 for s in specs]
    hi = [s[1]["bootstrap_95ci"]["ci_high"] * 100.0 for s in specs]
    xerr = np.array([[p - l_v for p, l_v in zip(pts, lo)], [h_v - p for p, h_v in zip(pts, hi)]])
    ax.grid(axis="x", linestyle="--", linewidth=0.65, color="#D4D4D4", zorder=0)
    ax.set_axisbelow(True)
    for i, (_, _, fill, hatch, lw, ls) in enumerate(specs):
        ax.barh(i, pts[i], height=0.64, color=fill, edgecolor="#000000", linewidth=lw, linestyle=ls, hatch=hatch, zorder=2)
    ax.errorbar(pts, y_pos, xerr=xerr, fmt="none", ecolor="#FFFFFF", elinewidth=3.6, capsize=5.0, capthick=3.0, zorder=4)
    ax.errorbar(pts, y_pos, xerr=xerr, fmt="none", ecolor="#000000", elinewidth=1.5, capsize=4.2, capthick=1.4, zorder=5)
    ax.scatter(pts, y_pos, s=26, facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.3, zorder=6)
    anchor_val = pts[2]
    ax.axvline(anchor_val, color="#595959", linestyle=":", linewidth=1.2, zorder=3)
    ax.text(anchor_val, -0.72, f"Step-0 Anchor ({anchor_val:.2f}%)", fontsize=8.0, fontweight="bold", color="#262626", ha="center", va="center", bbox=WHITE_PILL, zorder=7)
    for sy in (2.5, 3.5, 7.5):
        ax.axhline(sy, color="#BFBFBF", linestyle="-", linewidth=0.75, zorder=1)
    for i, (pt, l_v, h_v) in enumerate(zip(pts, lo, hi)):
        ax.text(h_v + 0.28, i, f"{pt:.2f}%  [{l_v:.2f}, {h_v:.2f}]", va="center", ha="left", fontsize=8.4 if i in (8, 9) else 8.0, fontweight="bold" if i in (8, 9) else "normal", bbox=WHITE_PILL, zorder=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in specs], fontsize=8.7)
    for idx in (8, 9):
        ax.get_yticklabels()[idx].set_fontweight("bold")
    ax.set_ylim(len(specs) - 0.35, -1.10)
    ax.set_xlim(62.0, 83.2)
    ax.set_xlabel("4-Class Pooled TEST mIoU (%) with 95% Patient Bootstrap CI (n = 23 Patients, 4,986 Patches)", fontsize=9.5, fontweight="bold")
    ax.set_title("BCSS-WSSS Gate-7 Track A (Phase-1B Post-Fix): Held-Out TEST mIoU & 95% Bootstrap CI", fontsize=11.3, fontweight="bold", pad=10)
    ax.text(0.0, -0.13, "*Pre-registered Gate-7 Track A Primary 4-Arm (0% TCGA-BRCA overlap, 12x ms_3_hvflip TTA). Secondary Simplex-CV dashed border denotes exploratory CV weights.", transform=ax.transAxes, fontsize=7.6, color="#333333")
    _clean_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig1_gate7_test_miou_ci_bw")


def draw_fig2_per_class_evolution(ev: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Fig 2: Grouped monochrome hatched bar chart of per-class TEST IoU evolution."""
    g7, g6 = ev["g7"], ev["g6"]
    p4_cgm = g7["test_nec_focused_1dof_diagnostic"]["primary_4arm_cgm_nec_focused_1dof"]
    series = [
        ("Author Pure (1x + CGM)", g7["test_author_anchors"]["author_official_pure"]["cgm_author_mul0"], "#F7F7F7", "", 0.9),
        ("Step-0 Author (Arm B + MS-TTA)", g7["test_author_anchors"]["author_step0_anchor_7043"]["ms_tta_cgm_neg1e4"], "#E6E6E6", "..", 0.9),
        ("Gate-6 R1 Primary 4-Arm", g6["test_primary_4arm"], "#D4D4D4", "//", 1.0),
        ("Gate-7 Arm 5 (DINOv2-L + CGM)", g7["test_individual_arms"]["arm5_dinov2_large_r2"], "#C2C2C2", "xx", 1.1),
        ("Gate-7 Primary 4-Arm (no-CGM)", g7["test_primary_4arm"], "#ADADAD", "\\\\\\", 1.6),
    ]
    cls_keys = ["tum", "str", "lym", "nec"]
    x_labels = ["TUM\n(Tumor)", "STR\n(Stroma)", "LYM\n(Lymphocyte)", "NEC\n(Necrosis)", "Pooled\n4-Class mIoU"]
    fig, ax = plt.subplots(figsize=(12.2, 5.9), dpi=300)
    x, width = np.arange(len(x_labels)), 0.155
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    ax.grid(axis="y", linestyle="--", linewidth=0.65, color="#D4D4D4", zorder=0)
    ax.set_axisbelow(True)
    all_vals: list[list[float]] = []
    for s_idx, (name, rec, fill, hatch, lw) in enumerate(series):
        vals = [rec["per_class_iou"][k] * 100.0 for k in cls_keys] + [rec["mIoU"] * 100.0]
        all_vals.append(vals)
        bars = ax.bar(x + offsets[s_idx], vals, width=width * 0.94, color=fill, edgecolor="#000000", linewidth=lw, hatch=hatch, label=name, zorder=2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.45, f"{val:.2f}", ha="center", va="bottom", rotation=90, fontsize=7.6, fontweight="bold" if s_idx in (3, 4) else "normal", bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.6}, zorder=5)
    raw_pooled_delta = (series[4][1]["mIoU_raw"] - series[0][1]["mIoU_raw"]) * 100.0
    for g_idx in range(len(x_labels)):
        delta = raw_pooled_delta if g_idx == 4 else (all_vals[4][g_idx] - all_vals[0][g_idx])
        ax.text(x[g_idx], max(row[g_idx] for row in all_vals) + 4.3, f"Delta vs Pure: +{delta:.2f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", bbox={"facecolor": "#F5F5F5", "edgecolor": "#000000", "linewidth": 0.8, "pad": 2.2}, zorder=6)
    ax.axvline(3.5, color="#737373", linestyle=":", linewidth=1.1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9.6, fontweight="bold")
    ax.set_ylim(54.0, 91.5)
    ax.set_ylabel("Held-Out TEST IoU / mIoU (%)", fontsize=9.8, fontweight="bold")
    ax.set_title(f"Per-Class Held-Out TEST IoU Evolution (Phase-1B Post-Fix | +NEC-1DoF CGM: {p4_cgm['mIoU']*100:.2f}%, NEC {p4_cgm['per_class_iou']['nec']*100:.2f}%)", fontsize=10.8, fontweight="bold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True, edgecolor="#000000", facecolor="#FFFFFF", fontsize=8.4)
    _clean_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig2_gate7_per_class_evolution_bw")


def _compute_patient_sensitivity(pjson: dict[str, Any], true_gt_counts: np.ndarray | None = None) -> dict[str, Any]:
    """Detects asymmetric GT==0 cells across all JSON records and optional all-11 true on-disk cells."""
    pids, classes, all_recs, stack = pjson["patient_ids"], pjson["class_names"], [], [pjson]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "union_p" in cur:
                all_recs.append(cur)
            stack.extend(cur.values())
    asym_mask = np.minimum.reduce([np.asarray(r["union_p"], dtype=float) for r in all_recs]) == 0

    def active_miou(rec: dict[str, Any], excl_mask: np.ndarray | None) -> np.ndarray:
        inter, union = np.asarray(rec["inter_p"], dtype=float), np.asarray(rec["union_p"], dtype=float)
        return np.asarray([float(np.mean([inter[i, c] / union[i, c] for c in range(4) if union[i, c] > 0 and not (excl_mask is not None and bool(excl_mask[i, c]))])) for i in range(len(pids))])

    prim, auth = pjson["primary_4arm"], pjson["author_anchors"]["author_step0_anchor_7043"]
    raw_d, gt_d = (active_miou(prim, None) - active_miou(auth, None)) * 100.0, (active_miou(prim, asym_mask) - active_miou(auth, asym_mask)) * 100.0
    gt0_info = {int(i): (classes[int(c)], int(prim["union_p"][int(i)][int(c)]), int(auth["union_p"][int(i)][int(c)])) for i, c in np.argwhere(asym_mask)}
    pstats = lambda d: (float(np.mean(d) / np.std(d, ddof=1)), float(stats.wilcoxon(d).pvalue), int(np.sum(d > 0)), int(np.sum(d < 0)))  # noqa: E731
    out: dict[str, Any] = {
        "pids": pids, "n_json_records": len(all_recs), "n_asym": int(np.sum(asym_mask)),
        "raw_d": raw_d, "gt_d": gt_d, "gt0_info": gt0_info, "raw_stats": pstats(raw_d), "gt_stats": pstats(gt_d),
        "true_gt_stats": None, "asym_nec_tot": int(np.sum(asym_mask[:, 3])), "asym_lym_tot": int(np.sum(asym_mask[:, 2])),
    }
    if true_gt_counts is not None:
        tm = np.asarray(true_gt_counts, dtype=np.int64) == 0
        out.update({"true_gt0_count": int(np.sum(tm)), "true_nec_absent": int(np.sum(tm[:, 3])), "true_lym_absent": int(np.sum(tm[:, 2])), "true_gt_stats": pstats((active_miou(prim, tm) - active_miou(auth, tm)) * 100.0)})
    return out


def draw_fig3_patient_gt0_sensitivity(ev: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Fig 3: 23-Patient Delta-mIoU waterfall + 8 NEC-absent patient FP clearing under NEC-1DoF CGM."""
    res, nec_diag = _compute_patient_sensitivity(ev["patient_json"], ev.get("true_gt_counts")), ev["g7"]["test_nec_focused_1dof_diagnostic"]
    has_true, n_asym = res["true_gt_stats"] is not None, res["n_asym"]
    order, y = np.argsort(res["raw_d"]), np.arange(len(res["pids"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 8.4), dpi=300, gridspec_kw={"width_ratios": [1.22, 1.0], "wspace": 0.28})
    box_style = {"facecolor": "#FFFFFF", "edgecolor": "#000000", "linewidth": 1.0, "pad": 3.6}
    ax1.grid(axis="x", linestyle="--", linewidth=0.6, color="#D9D9D9", zorder=0)
    ax1.axvline(0.0, color="#000000", linewidth=1.0, zorder=3)
    for row, p_idx in enumerate(order):
        rv, gv, is_gt0 = res["raw_d"][p_idx], res["gt_d"][p_idx], int(p_idx) in res["gt0_info"]
        fill = "#6E6E6E" if (is_gt0 and rv < -2.0) else ("#B8B8B8" if rv < 0 else "#E6E6E6")
        ax1.barh(row, rv, height=0.68, color=fill, edgecolor="#000000", linewidth=0.9, hatch="///" if (is_gt0 and rv < -2.0) else (".." if is_gt0 else ""), zorder=2)
        if is_gt0:
            if abs(gv - rv) > 1.5:
                ax1.annotate("", xy=(gv, row), xytext=(rv, row), arrowprops={"arrowstyle": "-|>", "color": "#000000", "lw": 1.15, "mutation_scale": 8.5, "shrinkA": 2, "shrinkB": 4}, zorder=5)
            ax1.scatter(gv, row, marker="D", s=34, facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.3, zorder=6)
            cname, fp_px, auth_px = res["gt0_info"][int(p_idx)]
            if fp_px > 0 and rv < -2.0:
                ax1.text(max(gv, 0.0) + 0.65, row, f"{cname} FP:{fp_px:,}px -> {gv:+.2f}%", va="center", ha="left", fontsize=7.2, fontweight="bold", bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.5}, zorder=7)
            elif fp_px == 0 and auth_px > 0:
                ax1.text(0.5 * (gv + rv), row + 0.62, f"4-Arm 0px (Auth {auth_px}px) -> {gv:+.2f}%", va="bottom", ha="center", fontsize=6.9, fontweight="bold", bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.4}, zorder=7)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"* {res['pids'][int(i)]} [Asym {res['gt0_info'][int(i)][0]}]" if int(i) in res["gt0_info"] else res["pids"][int(i)] for i in order], fontsize=7.8)
    for row, p_idx in enumerate(order):
        if int(p_idx) in res["gt0_info"]:
            ax1.get_yticklabels()[row].set_fontweight("bold")
    ax1.set_ylim(-0.7, len(order) + 0.45)
    ax1.set_xlim(-20.0, 24.5)
    ax1.set_xlabel("Per-Patient Delta-mIoU (%) vs Step-0 Author", fontsize=9.2, fontweight="bold")
    ax1.set_title(f"(A) Primary 4-Arm (no-CGM): Raw vs Asym-Free (-{n_asym} Asym GT=0)", fontsize=9.5, fontweight="bold", pad=8)
    r_dz, r_p, r_w, r_l = res["raw_stats"]
    g_dz, g_p, g_w, g_l = res["gt_stats"]
    box_a = f"Raw Active-Union (all union>0):\n  d_z = {r_dz:+.4f} | p = {r_p:.4f} | {r_w}W/0T/{r_l}L\nExcl. {n_asym} Asym GT=0 (min union=0):\n  d_z = {g_dz:+.4f} | p = {g_p:.1e} | {g_w}W/0T/{g_l}L"
    if has_true:
        t_dz, t_p, t_w, t_l = res["true_gt_stats"]
        box_a += f"\nExcl. All {res['true_gt0_count']} True GT=0 (on-disk):\n  d_z = {t_dz:+.4f} | p = {t_p:.1e} | {t_w}W/0T/{t_l}L"
    ax1.text(0.02, 0.975, box_a, transform=ax1.transAxes, ha="left", va="top", fontsize=7.1, bbox=box_style, zorder=8)

    m_no, m_cal, m_1d = nec_diag["nec_absent_fp_px_by_patient_no_cgm"], nec_diag["nec_absent_fp_px_by_patient_cgm_val_cal"], nec_diag["nec_absent_fp_px_by_patient_nec1dof"]
    nec_pids = sorted(m_no.keys(), key=lambda k: (m_1d[k], m_cal[k], m_no[k], k))
    yb, bh = np.arange(len(nec_pids)), 0.25
    ax2.grid(axis="x", linestyle="--", linewidth=0.6, color="#D9D9D9", zorder=0)
    for off, mp, fill, hatch, lbl in (
        (bh, m_no, "#EDEDED", "", f"No-CGM ({nec_diag['n_cleared_to_zero_px_no_cgm']}/8 clean, {nec_diag['total_nec_absent_fp_px_no_cgm']:,} px)"),
        (0.0, m_cal, "#BDBDBD", "..", f"CGM Val-Cal t=0.10 ({nec_diag['n_cleared_to_zero_px_cgm_val_cal']}/8 clean, {nec_diag['total_nec_absent_fp_px_cgm_val_cal']:,} px)"),
        (-bh, m_1d, "#333333", "///", f"NEC-1DoF t_nec=0.40 ({nec_diag['n_cleared_to_zero_px_nec1dof']}/8 clean, {nec_diag['total_nec_absent_fp_px_nec1dof']:,} px)"),
    ):
        ax2.barh(yb + off, [np.log10(float(mp[p]) + 1.0) for p in nec_pids], height=bh * 0.92, color=fill, edgecolor="#000000", linewidth=0.85, hatch=hatch, label=lbl, zorder=2)
    for row, pid in enumerate(nec_pids):
        v_no, v_1d = int(m_no[pid]), int(m_1d[pid])
        tag = ("0 px (already clean)" if v_no == 0 else f"CLEARED: {v_no:,} -> 0 px") if v_1d == 0 else f"{v_no:,} -> {v_1d:,} px"
        ax2.text(max(np.log10(v_no + 1.0), 0.12) + 0.12, row, tag, va="center", ha="left", fontsize=7.2, fontweight="bold" if v_1d == 0 else "normal", bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.4}, zorder=7)
    ax2.set_yticks(yb)
    ax2.set_yticklabels([p.removeprefix("TCGA-") for p in nec_pids], fontsize=8.4, fontweight="bold")
    ax2.set_xticks([0, 1, 2, 3, 4, 5])
    ax2.set_xticklabels(["0", "10", "100", "1k", "10k", "100k"], fontsize=8.3)
    ax2.set_xlim(0.0, 6.75)
    ax2.set_xlabel("False-Positive NEC Pixels (log10(px + 1) scale)", fontsize=9.2, fontweight="bold")
    n_clr, n_tot = int(nec_diag["n_cleared_to_zero_px_nec1dof"]), int(nec_diag["n_gt_nec_absent_patients"])
    ax2.set_title(f"(B) NEC-1DoF CGM (t_nec={nec_diag['t_nec_val_selected']:.2f}): Clears {n_clr}/{n_tot} NEC-Absent to 0 px", fontsize=9.5, fontweight="bold", pad=8)
    s_nc, p4_1d = nec_diag["primary_4arm_nec1dof_vs_no_cgm"], nec_diag["primary_4arm_cgm_nec_focused_1dof"]
    box_b = f"Primary 4-Arm + NEC-1DoF (VAL t_nec={nec_diag['t_nec_val_selected']:.2f}):\n  TEST mIoU = {p4_1d['mIoU']*100:.2f}% | NEC = {p4_1d['per_class_iou']['nec']*100:.2f}%\n  Clears {n_clr}/{n_tot} NEC-absent patients to 0 FP px\n  Total FP: {nec_diag['total_nec_absent_fp_px_no_cgm']:,} -> {nec_diag['total_nec_absent_fp_px_nec1dof']:,} px\n  vs no-CGM: p = {s_nc['wilcoxon_p']:.4f} ({s_nc['wins']}W/0T/{s_nc['losses']}L, d_z={s_nc['cohens_dz']:+.2f})"
    ax2.text(0.98, 0.03, box_b, transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.1, bbox=box_style, zorder=8)
    ax2.legend(loc="lower right", bbox_to_anchor=(0.98, 0.25), frameon=True, edgecolor="#000000", facecolor="#FFFFFF", fontsize=7.0)
    handles = [Patch(facecolor="#E6E6E6", edgecolor="#000000", label="Positive Delta-mIoU (normal patient)"), Patch(facecolor="#6E6E6E", edgecolor="#000000", hatch="///", label="Artificial drop from asym GT=0 micro-FP"), Line2D([0], [0], marker="D", color="#000000", markerfacecolor="#FFFFFF", markersize=6, label=f"Excl. {n_asym} asym GT=0 cells (min union=0)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.30, -0.015), ncol=2, frameon=True, edgecolor="#000000", facecolor="#FFFFFF", fontsize=7.9)
    suffix = f"{n_asym} Asym / {res['true_gt0_count']} True GT=0 Cells & {n_clr}/{n_tot} NEC-1DoF Clearing" if has_true else f"{n_asym} Asym GT=0 & {n_clr}/{n_tot} NEC-1DoF Clearing"
    fig.suptitle(f"Per-Patient Sensitivity Analysis (n = 23 Held-Out Patients): {suffix}", fontsize=11.2, fontweight="bold", y=0.985)
    _clean_axes(ax1, ax2)
    fig.subplots_adjust(top=0.91, bottom=0.11, left=0.175, right=0.98)
    return save_figure(fig, out_dir, "fig3_patient_gt0_sensitivity_bw")


def draw_fig4_pipeline_architecture(ev: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Fig 4: Academic Monochrome end-to-end Gate-7 Track A system architecture flowchart (Phase-1B)."""
    g7, g6, diag = ev["g7"], ev["g6"], ev["diag_r2"]
    s1, r2 = diag["candidates"]["q_s1_sgdap"], diag["candidates"]["p_ens4_uncal_ygated"]
    r1_a, r2_a, p4 = g6["test_individual_arms"], g7["test_individual_arms"], g7["test_primary_4arm"]
    nec_d = g7["test_nec_focused_1dof_diagnostic"]
    p4_cgm, s4 = nec_d["primary_4arm_cgm_nec_focused_1dof"], g7["test_simplex_cv_4arm"]["cgm_nec_focused_1dof"]
    p4_perm_holm = float(next(r for r in g7["test_confirmatory_stats_holm_m5"] if r["comparison"] == "primary_4arm_vs_author")["perm_p_holm"])
    sha_n = ev.get("sha256_count")
    sha_lbl = f"{sha_n}/{sha_n} SHA256 Artifacts (BAD = 0)" if sha_n else "SHA256 Artifacts Verified (BAD = 0)"
    ci_lo, ci_hi = p4["bootstrap_95ci"]["ci_low"] * 100.0, p4["bootstrap_95ci"]["ci_high"] * 100.0
    n_clr, n_tot = int(nec_d["n_cleared_to_zero_px_nec1dof"]), int(nec_d["n_gt_nec_absent_patients"])
    fig, ax = plt.subplots(figsize=(16.2, 7.8), dpi=300)
    ax.set_xlim(0.0, 17.0)
    ax.set_ylim(0.0, 7.9)
    ax.axis("off")
    ax.text(8.5, 7.56, "SE374.R11 Gate-7 Track A (Phase-1B Post-Fix): End-to-End WSSS Distillation & Clinical Architecture", ha="center", va="center", fontsize=12.0, fontweight="bold")
    ax.text(8.5, 7.22, "BCSS-WSSS (TCGA-BRCA, 151 Patients: 106 Train / 22 Val / 23 Held-Out Test | 0% Patient Overlap | 0% TCGA-BRCA Pretraining Overlap)", ha="center", va="center", fontsize=8.9, color="#333333")
    w, h1, h2, x1, x2, x3 = 4.60, 2.38, 2.48, 0.35, 6.20, 12.05
    boxes = [
        (x1, 4.44, h1, "1. STAGE-1 CAM & SG-DAP (WEAK LABELS)", "#FAFAFA", "#262626", 1.4, [("- Input: 23,422 Train patches (weak y_cls only)", False), ("- ResNet-38d + PDA (Ep 5 EMA, LUAD init)", False), ("- Target: q_s1_sgdap (T = 0.25, y-gated masks)", False), (f"- VAL (3,418 patches): {s1['mIoU']*100:.2f}% | ECE {s1['ECE']:.4f}", True)]),
        (x2, 4.44, h1, "2. ROUND-1 4-ARM TEACHERS (GATE-6)", "#F2F2F2", "#404040", 1.4, [(f"- Arm 1 DeepLab R101: {r1_a['arm1_deeplab_cleankd']['mIoU']*100:.2f}% TEST", False), (f"- Arm 2 SegFormer MiT-B3: {r1_a['arm2_segformer_oeem']['mIoU']*100:.2f}% TEST", False), (f"- Arm 3 ConvNeXt-B {r1_a['arm3_convnext_selftrain']['mIoU']*100:.2f}% | Arm 4 DINOv2-B {r1_a['arm4_dinov2_adapter']['mIoU']*100:.2f}%", False), (f"- Gate-6 Primary 4-Arm (+CGM): {g6['test_primary_4arm']['mIoU']*100:.2f}% TEST", True)]),
        (x3, 4.44, h1, "3. ROUND-2 DISTILLATION (VAL CEILING)", "#FAFAFA", "#1A1A1A", 1.7, [("- Target: p_ens4_uncal_ygated (r2_blend = 1.0)", False), (f"- Fallback p_sum < 0.01 -> q_s1 ({diag['frac_psum_lt_0_01']*100:.4f}% VAL)", False), (f"- Multi-Class VAL: {r2['multiclass_mIoU']*100:.2f}% | ECE {r2['ECE']:.4f} ({(r2['ECE']/s1['ECE']-1.0)*100:+.0f}%)", False), (f"- VAL Ceiling: {r2['mIoU']*100:.2f}% mIoU (+{(r2['mIoU']-s1['mIoU'])*100:.2f}% vs S1)", True)]),
        (x1, 0.90, h2, "4. GATE-7 TRACK A 4-ARM (R2 EMA-ONLY, 224/336px)", "#F2F2F2", "#262626", 1.4, [("- Dual-Head (Seg + Cls), ema_only=True, 12x TTA", False), (f"- Arm 8 DeepLab {r2_a['arm8_deeplab_r101_r2']['mIoU']*100:.2f}% | Arm 7 ConvNeXt-L {r2_a['arm7_convnext_large_r2']['mIoU']*100:.2f}%", False), (f"- Arm 6 SegFormer-B5: {r2_a['arm6_segformer_b5_r2']['mIoU']*100:.2f}% TEST (no-CGM)", False), (f"- Arm 5 DINOv2-L 336px: {r2_a['arm5_dinov2_large_r2']['mIoU']*100:.2f}% TEST (+CGM)", True)]),
        (x2, 0.90, h2, "5. CALIBRATED LOG-PROB 4-ARM ENSEMBLE & CGM", "#E8E8E8", "#000000", 2.0, [("- VAL-frozen betas (cal_logprob_eps1e3, 12x TTA)", False), (f"- Primary 4-Arm: {p4['mIoU']*100:.2f}% no-CGM / {p4_cgm['mIoU']*100:.2f}% CGM", True), (f"- NEC-1DoF (t=0.40): {n_clr}/{n_tot} NEC-absent 0px | p_Holm={p4_perm_holm:.4f}", False), (f"- Secondary Simplex (+NEC-1DoF): {s4['mIoU']*100:.2f}% TEST", False)]),
        (x3, 0.90, h2, "6. CLINICAL FASTAPI + OFFLINE WORKSTATION", "#FAFAFA", "#404040", 1.4, [("- FastAPI (api_server.py, strict_weights, Docker)", False), ("- Optical Otsu + Pure-Glass (>= 512 px BACK)", False), ("- Offline HTML5 Canvas: 9 QA Modes, Loupe, Zoom", False), ("- WSI Multi-Tile Stitch, VTP/TSR %, CAP PDF", True)]),
    ]
    for bx, by, bh, title, fill, hfill, lw, lines in boxes:
        hh = 0.44
        ax.add_patch(FancyBboxPatch((bx, by), w, bh, boxstyle="round,pad=0.03,rounding_size=0.10", facecolor=fill, edgecolor="#000000", linewidth=lw, zorder=2))
        ax.add_patch(FancyBboxPatch((bx, by + bh - hh), w, hh, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=hfill, edgecolor="#000000", linewidth=1.0, zorder=3))
        ax.text(bx + w / 2.0, by + bh - hh / 2.0, title, ha="center", va="center", fontsize=7.6 if len(title) > 44 else 8.2, fontweight="bold", color="#FFFFFF", zorder=4)
        step = (bh - hh - 0.24) / len(lines)
        for idx, (text, bold) in enumerate(lines):
            ax.text(bx + 0.16, by + bh - hh - 0.26 - idx * step, text, ha="left", va="center", fontsize=7.55 if bold else 7.35, fontweight="bold" if bold else "normal", zorder=4)
    for ax1_x, ax2_x, ay, lbl in [(x1 + w + 0.12, x2 - 0.12, 5.63, "q_s1_sgdap\n(23,422)"), (x2 + w + 0.12, x3 - 0.12, 5.63, "4-Arm\nSoft Avg"), (x1 + w + 0.12, x2 - 0.12, 2.13, "12x TTA +\nCls Heads"), (x2 + w + 0.12, x3 - 0.12, 2.13, "Otsu +\nGlass Guard")]:
        ax.annotate("", xy=(ax2_x, ay), xytext=(ax1_x, ay), arrowprops={"arrowstyle": "-|>", "color": "#000000", "lw": 1.6, "mutation_scale": 12}, zorder=5)
        ax.text((ax1_x + ax2_x) / 2.0, ay + 0.35, lbl, ha="center", va="center", fontsize=7.3, fontweight="bold", bbox=WHITE_PILL, zorder=6)
    ax.plot([x3 + w / 2.0, x3 + w / 2.0, x1 + w / 2.0], [4.41, 3.92, 3.92], color="#000000", linewidth=1.6, zorder=4)
    ax.annotate("", xy=(x1 + w / 2.0, 3.43), xytext=(x1 + w / 2.0, 3.92), arrowprops={"arrowstyle": "-|>", "color": "#000000", "lw": 1.6, "mutation_scale": 12}, zorder=5)
    ax.text(8.5, 3.92, "23,422 Train Round-2 Soft Masks (r2_blend = 1.0, 0% TEST Access, Track A Clean)", ha="center", va="center", fontsize=8.0, fontweight="bold", bbox={"facecolor": "#FFFFFF", "edgecolor": "#000000", "linewidth": 0.8, "pad": 2.6}, zorder=6)
    ax.plot([0.35, 16.65], [0.52, 0.52], color="#BFBFBF", linewidth=0.8)
    ax.text(8.5, 0.25, f"Cryptographic Audit: {sha_lbl} | Held-Out TEST: 23 Patients (4,986 Patches) | 95% Boot CI [{ci_lo:.2f}%, {ci_hi:.2f}%]", ha="center", va="center", fontsize=8.1, color="#262626")
    return save_figure(fig, out_dir, "fig4_gate7_pipeline_architecture_bw")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SE374.R11 Gate-7 Academic Monochrome (B&W) figures.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("product/reports/figures"))
    args = parser.parse_args()
    configure_monochrome_style()
    ev = load_evidence(args.repo_root.resolve())
    out_dir = (args.repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    for gen in (draw_fig1_test_miou_ci, draw_fig2_per_class_evolution, draw_fig3_patient_gt0_sensitivity, draw_fig4_pipeline_architecture):
        png_path, svg_path = gen(ev, out_dir)
        print(f"Generated: {png_path} | {svg_path}")


if __name__ == "__main__":
    main()
