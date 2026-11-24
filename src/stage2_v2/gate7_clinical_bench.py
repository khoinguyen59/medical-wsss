"""Gate-7 Track A Clinical Biomarkers, Live-GDC Survival, Matched GPU Pareto, Calibration & LUAD Suite (ASCII '-' only)."""

import gc
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.clinical import survival_analysis as surv_mod
from src.models import wsss_inference_methods as wim
from src.stage2_v2 import clinical_bench_plots as cbp
from src.stage2_v2 import ensemble as ens
from src.stage2_v2 import evaluator as s2_eval
from src.stage2_v2 import visual_audit as va
from src.stage2_v2.models import load_stage2_v2_checkpoint
from src.training import dataset_loader as dl
from src.training.eval_stage1_full import load_stage1_strict
from src.training.train_stage2 import prepare_stage1_input

_TTA_VIEW_COUNTS: Dict[str, int] = {"raw_1x": 1, "ms_up_hflip": 4, "ms_3_hflip": 6, "d4_8view_1x": 8, "ms_3_hvflip": 12}
_SHORT_LABELS, GUIDELINE_ATTRIBUTION, LUAD_TO_BCSS_LUT = cbp._SHORT_LABELS, cbp.GUIDELINE_ATTRIBUTION, cbp.LUAD_TO_BCSS_LUT
LUADBCSSAlignedDataset, PixelCalibrationAccumulator = cbp.LUADBCSSAlignedDataset, cbp.PixelCalibrationAccumulator
compute_cohens_kappa, compute_lins_ccc, compute_bland_altman_and_pearson = cbp.compute_cohens_kappa, cbp.compute_lins_ccc, cbp.compute_bland_altman_and_pearson
extract_luad_case_id, remap_luad_mask_to_bcss, resolve_provenance_metadata = cbp.extract_luad_case_id, cbp.remap_luad_mask_to_bcss, cbp.resolve_provenance_metadata
render_bland_altman_bw, render_km_survival_bw, render_gpu_pareto_bw, render_luad_overlays_sheet = cbp.render_bland_altman_bw, cbp.render_km_survival_bw, cbp.render_gpu_pareto_bw, cbp.render_luad_overlays_sheet

def _free_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

def _load_arm(spec: Mapping[str, Any], device: torch.device) -> nn.Module:
    return spec["model"].to(device).eval() if spec.get("model") is not None else load_stage2_v2_checkpoint(spec["ckpt_path"], device=device, strict=True)[0].eval()

def _run_autocast_fwd(fn: Any, inp: torch.Tensor, device: torch.device, amp_dtype: Optional[torch.dtype], use_amp: bool) -> Any:
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        return fn(inp)

def _opt_round(val: Optional[float]) -> Optional[float]:
    return round(float(val), 2) if val is not None else None

def _close_mms(mms: Sequence[Any]) -> None:
    for mm in mms:
        ens._close_memmap(mm)

def _pooled_miou_pct(pred: np.ndarray, gt: np.ndarray) -> float:
    ip, up = va._accumulate_patient_iu(pred, gt, np.zeros(int(gt.shape[0]), dtype=np.int32), 1)
    return round(float(ens.summarize_iu(ip[0], up[0])["mIoU"]) * 100.0, 2)

def _load_s1_expert(override: Optional[nn.Module], ckpt: Optional[Union[str, Path]], device: torch.device) -> Tuple[nn.Module, str]:
    if override is not None:
        return override.to(device).eval(), "zero_one"
    if ckpt is None:
        raise RuntimeError("[STAGE2 ERROR] stage1_cls_ckpt is required when stage1_model_override is None.")
    mod = wim.WSSSTissueStage1NetCAM(n_class=dl.NUM_FOREGROUND)
    return mod.to(device).eval(), str(load_stage1_strict(mod, ckpt)[0] or "zero_one")

def _patient_bio_from_masks(masks: Sequence[np.ndarray]) -> Dict[str, Any]:
    c = [sum(int(np.sum(m == k)) for m in masks) for k in range(dl.NUM_FOREGROUND)]
    tsr, sp = surv_mod.compute_tumor_stroma_ratio(c[0], c[1]), surv_mod.compute_stroma_proportion(c[0], c[1])
    return {"vtp": surv_mod.compute_viable_tumor_percentage(c[0], sum(c)), "vtp_rcb": surv_mod.compute_vtp_rcb(c[0], c[3]), "tsr": tsr, "hotspot_tsr": float(surv_mod.compute_hotspot_tumor_stroma_ratio(masks)["hotspot_tsr"]), "risk": surv_mod.stratify_cap_risk(tsr, stroma_proportion=sp, threshold=50.0)}

def _summarize_bio_agreement(patient_rows: Sequence[Mapping[str, Any]], model_keys: Sequence[str], suffix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mk in model_keys:
        pref = f"{mk}{suffix}"
        pairs = [(int(r["gt_cap_code"]), int(r[f"{pref}_cap_code"])) for r in patient_rows if int(r["gt_cap_code"]) in (0, 1) and int(r[f"{pref}_cap_code"]) in (0, 1)]
        yt, yp = [x[0] for x in pairs], [x[1] for x in pairs]
        out[mk] = {**{m: compute_bland_altman_and_pearson([r[f"{pref}_{m}"] for r in patient_rows], [r[f"gt_{m}"] for r in patient_rows]) for m in ("vtp", "vtp_rcb", "tsr", "hotspot_tsr")}, "cap_concordance": {"guideline_attribution": GUIDELINE_ATTRIBUTION, "n_measurable": len(pairs), "n_concordant": int(sum(a == b for a, b in pairs)), "accuracy": round(float(np.mean(np.asarray(yt) == np.asarray(yp))) if pairs else 0.0, 4), "cohens_kappa": round(compute_cohens_kappa(yt, yp), 4)}}
    return out


def compute_gate7_clinical_biomarkers(patient_ids: Sequence[str], raw_rgbs: Sequence[np.ndarray], gt_masks: np.ndarray, model_preds: Mapping[str, np.ndarray], output_dir: Union[str, Path], primary_model: str = "ens4_cgm", clinical_manifest: Optional[Mapping[str, Mapping[str, Any]]] = None, dpi: int = 300) -> Dict[str, Any]:
    """Compute per-patient VTP/VTP_RCB/TSR/Hotspot-TSR (GT-free Cand1 + ROI-bounded) & live-GDC KM survival."""
    if not model_preds:
        raise ValueError("model_preds must contain at least one model prediction array.")
    out_p, manifest, model_keys = Path(output_dir), (clinical_manifest if clinical_manifest is not None else surv_mod.TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA), list(model_preds.keys())
    out_p.mkdir(parents=True, exist_ok=True)
    primary_model, by_pid = (primary_model if primary_model in model_preds else model_keys[0]), {}
    for idx, pid in enumerate(patient_ids):
        by_pid.setdefault(str(pid), []).append(idx)
    patient_rows: List[Dict[str, Any]] = []
    for pid, idxs in sorted(by_pid.items()):
        gt_b, surv = _patient_bio_from_masks([gt_masks[i] for i in idxs]), manifest.get(pid, {})
        row: Dict[str, Any] = {"patient_id": pid, "n_patches": len(idxs), "gt_vtp": round(gt_b["vtp"], 3), "gt_vtp_rcb": round(gt_b["vtp_rcb"], 3), "gt_tsr": round(gt_b["tsr"], 3), "gt_hotspot_tsr": round(gt_b["hotspot_tsr"], 3), "gt_cap_risk": gt_b["risk"]["risk_level"], "gt_cap_code": int(gt_b["risk"]["risk_code"]), "overall_survival_months": surv.get("overall_survival_months"), "vital_status": surv.get("vital_status"), "clinical_stage": surv.get("clinical_stage"), "histological_subtype": surv.get("histological_subtype")}
        for mk in model_keys:
            cand1_m = [wim.MultiMethodWSSSEngineV2.apply_bright_glass_guard(model_preds[mk][i], raw_rgbs[i], background_class_idx=4) for i in idxs]
            roi_m = [np.where((gt_masks[i] >= 0) & (gt_masks[i] < dl.NUM_FOREGROUND), model_preds[mk][i], 4) for i in idxs]
            for suf, mlist in (("", cand1_m), ("_roi", roi_m)):
                pb = _patient_bio_from_masks(mlist)
                row.update({f"{mk}{suf}_vtp": round(pb["vtp"], 3), f"{mk}{suf}_vtp_rcb": round(pb["vtp_rcb"], 3), f"{mk}{suf}_tsr": round(pb["tsr"], 3), f"{mk}{suf}_hotspot_tsr": round(pb["hotspot_tsr"], 3), f"{mk}{suf}_cap_risk": pb["risk"]["risk_level"], f"{mk}{suf}_cap_code": int(pb["risk"]["risk_code"]), f"{mk}{suf}_cap_match": int(pb["risk"]["risk_code"] == gt_b["risk"]["risk_code"])})
        patient_rows.append(row)
    km_svc, km_by_source, matched = surv_mod.KaplanMeierSurvivalAnalysis(use_lifelines_if_available=True), {}, [r for r in patient_rows if r["overall_survival_months"] is not None and r["vital_status"] is not None]
    for src in ["gt"] + model_keys:
        col = "gt_cap_code" if src == "gt" else f"{src}_cap_code"
        dh, eh = [float(r["overall_survival_months"]) for r in matched if int(r[col]) == 1], [int(r["vital_status"]) for r in matched if int(r[col]) == 1]
        dl_v, el = [float(r["overall_survival_months"]) for r in matched if int(r[col]) == 0], [int(r["vital_status"]) for r in matched if int(r[col]) == 0]
        lr = km_svc.logrank_test(dh, eh, dl_v, el, label_a="High Risk", label_b="Low Risk") if (dh and dl_v) else {"test_statistic": 0.0, "p_value": 1.0, "note": "Single risk stratum"}
        km_by_source[src] = {"n_high_risk": len(dh), "n_low_risk": len(dl_v), "events_high_risk": int(sum(eh)), "events_low_risk": int(sum(el)), "logrank": lr, "fit_high": km_svc.fit(dh, eh, label=f"{src}_High"), "fit_low": km_svc.fit(dl_v, el, label=f"{src}_Low"), "durations_high": dh, "events_high": eh, "durations_low": dl_v, "events_low": el}
    per_model_stats, per_model_roi = _summarize_bio_agreement(patient_rows, model_keys, ""), _summarize_bio_agreement(patient_rows, model_keys, "_roi")
    for mk in model_keys:
        per_model_stats[mk]["km_logrank_p"] = float(km_by_source[mk]["logrank"]["p_value"])
    csv_p = out_p / "patient_clinical_risk_summary.csv"
    va._atomic_write_csv(csv_p, list(patient_rows[0].keys()) if patient_rows else ["patient_id"], patient_rows)
    ba_png, km_png = render_bland_altman_bw(patient_rows, per_model_stats[primary_model], primary_model, out_p / "bland_altman_tsr_vtp_bw.png", dpi=dpi), render_km_survival_bw(km_by_source, primary_model, out_p / "km_survival_tsr_bw.png", dpi=dpi)
    summary = {"n_patients": len(patient_rows), "n_patches": len(patient_ids), "primary_model": primary_model, "guideline_attribution": GUIDELINE_ATTRIBUTION, "gt_free_glass_guard": "MultiMethodWSSSEngineV2.apply_bright_glass_guard", "gdc_provenance": surv_mod.GDC_CLINICAL_MANIFEST_META, "models": per_model_stats, "models_roi_bounded": per_model_roi, "kaplan_meier": km_by_source, "patients": patient_rows, "artifacts": [csv_p.name, ba_png.name, km_png.name]}
    va._atomic_write_json(out_p / "gate7_clinical_biomarkers.json", summary)
    return summary


def _benchmark_forward_fn(fn: Any, n_patches: int, param_m: float, device: torch.device, n_warmup: int, n_iters: int, n_repeats: int = 5) -> Dict[str, Any]:
    lats, fps_list, peak_mb = [], [], 0.0
    with torch.no_grad():
        for _ in range(max(1, n_warmup)):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(max(1, int(n_repeats))):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(max(1, n_iters)):
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            el, n_tot = max(time.perf_counter() - t0, 1e-9), max(1, int(n_patches)) * max(1, int(n_iters))
            lats.append((el * 1000.0) / n_tot)
            fps_list.append(n_tot / el)
        if device.type == "cuda":
            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    la, fa = np.asarray(lats, dtype=np.float64), np.asarray(fps_list, dtype=np.float64)
    return {"latency_ms_per_patch": round(float(la.mean()), 3), "latency_ms_std": round(float(la.std(ddof=1)) if la.size > 1 else 0.0, 3), "throughput_fps": round(float(fa.mean()), 2), "throughput_fps_std": round(float(fa.std(ddof=1)) if fa.size > 1 else 0.0, 2), "peak_vram_mb": round(peak_mb, 2), "param_count_m": round(float(param_m), 2)}


def _oom_fallback_row(lat_ms: float, vram_mb: float, param_m: float, prec_lbl: str, tta_views: int) -> Dict[str, Any]:
    return {"latency_ms_per_patch": round(lat_ms, 3), "latency_ms_std": 0.0, "throughput_fps": round(1000.0 / max(lat_ms, 1e-9), 2), "throughput_fps_std": 0.0, "peak_vram_mb": round(vram_mb, 2), "param_count_m": round(param_m, 2), "precision": prec_lbl, "tta_views": int(tta_views), "measurement_mode": "sequential_oom_fallback"}


def benchmark_gate7_gpu_efficiency(arm_specs: Sequence[Mapping[str, Any]], author_stage2_ckpt: Optional[Union[str, Path]], sample_imgs: torch.Tensor, device: torch.device, amp_dtype: Optional[torch.dtype], output_dir: Union[str, Path], reference_mious: Optional[Mapping[str, float]] = None, reference_mious_raw_1x: Optional[Mapping[str, float]] = None, selected_ttas: Optional[Sequence[str]] = None, stage1_cls_ckpt: Optional[Union[str, Path]] = None, stage1_model_override: Optional[nn.Module] = None, n_warmup: int = 5, n_iters: int = 25, n_repeats: int = 5, author_model_override: Optional[nn.Module] = None, dpi: int = 300) -> Dict[str, Any]:
    """Measure matched-AMP GPU latency (mean+/-std, 5x25 iters) for BOTH Raw 1x and Deployed TTA+CGM Pareto curves."""
    out_p, ref_tta, ref_raw = Path(output_dir), dict(reference_mious or {}), (dict(reference_mious_raw_1x) if reference_mious_raw_1x is not None else None)
    out_p.mkdir(parents=True, exist_ok=True)
    _free_cuda(device)
    inp_imnet, b_sz, h_in, w_in = sample_imgs.to(device), int(sample_imgs.shape[0]), int(sample_imgs.shape[2]), int(sample_imgs.shape[3])
    use_amp = device.type == "cuda" and amp_dtype is not None
    prec_lbl, inp_01 = (str(amp_dtype) if use_amp else "torch.float32"), prepare_stage1_input(inp_imnet, stage1_norm_mode="zero_one", images_are_imagenet_norm=True)
    arm_names = [str(sp["name"]).strip().lower() for sp in arm_specs]
    ttas = list(selected_ttas) if selected_ttas is not None else [va.GATE7_SELECTED_TTAS_BY_ARM.get(n, "ms_3_hflip") for n in arm_names]
    betas = [va.GATE7_FULL_VAL_BETAS_BY_ARM.get(n, 0.7) for n in arm_names]
    w_raw = np.asarray([va.GATE7_UNIFORM_WEIGHTS_BY_ARM.get(n, 1.0 / max(1, len(arm_names))) for n in arm_names], dtype=np.float32)
    weights = w_raw / max(float(w_raw.sum()), 1e-6)
    s1_m = _load_s1_expert(stage1_model_override, stage1_cls_ckpt, device)[0] if (stage1_model_override is not None or stage1_cls_ckpt is not None) else None
    s1_pm = (sum(p.numel() for p in s1_m.parameters()) / 1e6) if s1_m is not None else 0.0
    raw_1x, deployed_tta, legacy_models, fp32_ref = {}, {}, {}, None
    nw_t, ni_t, nr_t = max(1, n_warmup // 2), max(1, n_iters // 2), max(1, min(n_repeats, 3))
    if author_model_override is not None or author_stage2_ckpt is not None:
        auth_m = author_model_override.to(device).eval() if author_model_override is not None else wim.build_and_load_deeplabv3plus(author_stage2_ckpt, device=device, strict=True)
        pm_a = sum(p.numel() for p in auth_m.parameters()) / 1e6
        r_a = {**_benchmark_forward_fn(lambda m=auth_m: _run_autocast_fwd(m, inp_01, device, amp_dtype, use_amp), b_sz, pm_a, device, n_warmup, n_iters, n_repeats), "precision": prec_lbl, "effective_resolution": f"{h_in}x{w_in}", "tta_views": 1}
        fp32_ref = {**_benchmark_forward_fn(lambda m=auth_m: m(inp_01), b_sz, pm_a, device, nw_t, ni_t, nr_t), "precision": "torch.float32", "effective_resolution": f"{h_in}x{w_in}", "tta_views": 1}
        raw_1x["author_stage2"] = {**r_a, "reference_miou_pct": _opt_round(ref_raw.get("author_raw") if ref_raw is not None else ref_tta.get("author_step0"))}
        legacy_models["author_stage2"] = {**r_a, "reference_miou_pct": _opt_round(ref_tta.get("author_step0"))}
        def _auth_tta() -> Any:
            lg = (o["seg_logits"] if isinstance(o := _run_autocast_fwd(auth_m, inp_01, device, amp_dtype, use_amp), dict) else o)[:, : dl.NUM_FOREGROUND].float()
            pr = s2_eval._ms_tta_probs(auth_m, inp_01, lg, (0.75, 1.0, 1.25), device, amp_dtype=amp_dtype if use_amp else None)
            return (pr, _run_autocast_fwd(s1_m.forward_cam, inp_01, device, amp_dtype, use_amp)) if s1_m is not None else pr
        deployed_tta["author_stage2"] = {**_benchmark_forward_fn(_auth_tta, b_sz, pm_a + s1_pm, device, nw_t, ni_t, nr_t), "precision": prec_lbl, "tta_mode": "ms_3_hflip+s1_cgm" if s1_m is not None else "ms_3_hflip", "tta_views": 6, "uses_stage1_cgm": s1_m is not None, "reference_miou_pct": _opt_round(ref_tta.get("author_step0"))}
        if author_model_override is None:
            del auth_m
            _free_cuda(device)
    lat_sum, vram_max, param_sum, tta_lat_sum = 0.0, 0.0, 0.0, 0.0
    for idx, spec in enumerate(arm_specs):
        name, mod, tta_m = arm_names[idx], _load_arm(spec, device), ttas[idx]
        pm, sc = sum(p.numel() for p in mod.parameters()) / 1e6, float(getattr(mod, "internal_scale", 1.0))
        r_m = {**_benchmark_forward_fn(lambda m=mod: _run_autocast_fwd(m, inp_imnet, device, amp_dtype, use_amp), b_sz, pm, device, n_warmup, n_iters, n_repeats), "precision": prec_lbl, "effective_resolution": f"{int(round(h_in * sc))}x{int(round(w_in * sc))}", "tta_views": 1}
        ref_v = ref_tta.get(name, ref_tta.get("arm5_cgm") if "arm5" in name else None)
        raw_1x[name], legacy_models[name] = {**r_m, "reference_miou_pct": _opt_round(ref_raw.get(name) if ref_raw is not None else ref_v)}, {**r_m, "reference_miou_pct": _opt_round(ref_v)}
        t_m = {**_benchmark_forward_fn(lambda m=mod, tm=tta_m: ens.apply_tta_probs(m, inp_imnet, tm, device, amp_dtype=amp_dtype if use_amp else None), b_sz, pm, device, nw_t, ni_t, nr_t), "precision": prec_lbl, "tta_mode": tta_m, "tta_views": _TTA_VIEW_COUNTS.get(tta_m, 6), "reference_miou_pct": _opt_round(ref_tta.get("arm5_cgm") if ("arm5" in name and ref_tta.get("arm5_cgm") is not None) else ref_v)}
        deployed_tta[name], lat_sum, vram_max, param_sum, tta_lat_sum = t_m, lat_sum + r_m["latency_ms_per_patch"], max(vram_max, r_m["peak_vram_mb"]), param_sum + pm, tta_lat_sum + t_m["latency_ms_per_patch"]
        if spec.get("model") is None:
            del mod
            _free_cuda(device)
    if arm_specs:
        loaded, tot_views = [], sum(_TTA_VIEW_COUNTS.get(t, 6) for t in ttas)
        ens_uses_s1_cgm = bool(s1_m is not None and ref_tta.get("ens4_no_cgm") is None and ref_tta.get("ens4_cgm") is not None)
        ens_tta_pm = param_sum + (s1_pm if ens_uses_s1_cgm else 0.0)
        try:
            loaded = [_load_arm(sp, device) for sp in arm_specs]
            def _ens_1x() -> torch.Tensor:
                return sum(float(wi) * torch.log(torch.clamp(F.softmax(((o["seg_logits"] if isinstance(o := _run_autocast_fwd(m, inp_imnet, device, amp_dtype, use_amp), dict) else o)[:, : dl.NUM_FOREGROUND].float()) * float(bi), dim=1), min=1e-3)) for m, wi, bi in zip(loaded, weights, betas))
            def _ens_tta() -> Any:
                acc = sum(float(wi) * torch.log(torch.clamp(F.softmax(torch.log(torch.clamp(ens.apply_tta_probs(m, inp_imnet, tm, device, amp_dtype=amp_dtype if use_amp else None)[0], min=1e-7)) * float(bi), dim=1), min=1e-3)) for m, tm, wi, bi in zip(loaded, ttas, weights, betas))
                return (acc, _run_autocast_fwd(s1_m.forward_cam, inp_01, device, amp_dtype, use_amp)) if (ens_uses_s1_cgm and s1_m is not None) else acc
            e_m = {**_benchmark_forward_fn(_ens_1x, b_sz, param_sum, device, n_warmup, n_iters, n_repeats), "precision": prec_lbl, "tta_views": 1, "measurement_mode": "resident_multi_arm_forward"}
            et_m = {**_benchmark_forward_fn(_ens_tta, b_sz, ens_tta_pm, device, nw_t, ni_t, nr_t), "precision": prec_lbl, "tta_mode": "4arm_tta+s1_cgm" if ens_uses_s1_cgm else "4arm_tta_no_cgm", "tta_views": tot_views, "uses_stage1_cgm": ens_uses_s1_cgm, "measurement_mode": "resident_multi_arm_tta"}
        except RuntimeError:
            e_m, et_m = _oom_fallback_row(lat_sum, vram_max, param_sum, prec_lbl, 1), _oom_fallback_row(tta_lat_sum, vram_max, ens_tta_pm, prec_lbl, tot_views)
        finally:
            loaded.clear()
            _free_cuda(device)
        ens_ref_tta = ref_tta.get("ens4_no_cgm") if ref_tta.get("ens4_no_cgm") is not None else ref_tta.get("ens4_cgm")
        raw_1x["primary_4arm_ensemble"] = {**e_m, "reference_miou_pct": _opt_round(ref_raw.get("ens4_raw_1x") if ref_raw is not None else ens_ref_tta)}
        legacy_models["primary_4arm_ensemble"], deployed_tta["primary_4arm_ensemble"] = {**e_m, "reference_miou_pct": _opt_round(ens_ref_tta)}, {**et_m, "reference_miou_pct": _opt_round(ens_ref_tta)}
    if s1_m is not None and stage1_model_override is None:
        del s1_m
        _free_cuda(device)
    payload = {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "amp_dtype": prec_lbl, "batch_size": b_sz, "n_warmup": int(n_warmup), "n_iters": int(n_iters), "n_repeats": int(n_repeats), "models": legacy_models, "raw_1x": raw_1x, "deployed_tta": deployed_tta, **({"author_stage2_fp32": fp32_ref} if fp32_ref is not None else {})}
    va._atomic_write_json(out_p / "gate7_gpu_efficiency.json", payload)
    render_gpu_pareto_bw(payload, out_p / "gpu_pareto_efficiency_bw.png", dpi=dpi)
    return payload


def _stream_gate7_predictions(arm_specs: Sequence[Mapping[str, Any]], loader: DataLoader, stage1_cls_ckpt: Optional[Union[str, Path]], author_stage1_ckpt: Optional[Union[str, Path]], author_stage2_ckpt: Optional[Union[str, Path]], device: torch.device, amp_dtype: Optional[torch.dtype], selected_ttas: Optional[Sequence[str]] = None, track: str = "track_a_clean", stage1_model_override: Optional[nn.Module] = None, author_stage1_model_override: Optional[nn.Module] = None, author_model_override: Optional[nn.Module] = None, collect_raw_1x_and_cal: bool = True) -> Dict[str, Any]:
    """Stream frozen Gate-7 Track A + Author predictions (TTA + 1x raw + linear/logprob + pixel calibration)."""
    if not arm_specs:
        raise ValueError("arm_specs must contain at least one Gate-7 arm specification.")
    for sp in arm_specs:
        ens.validate_track_a_arm_spec(sp, track=track)
    arm_names = [str(sp["name"]).strip().lower() for sp in arm_specs]
    ttas = list(selected_ttas) if selected_ttas is not None else [va.GATE7_SELECTED_TTAS_BY_ARM.get(n, "ms_3_hflip") for n in arm_names]
    betas = [va.GATE7_FULL_VAL_BETAS_BY_ARM.get(n, 0.7) for n in arm_names]
    w_raw = np.asarray([va.GATE7_UNIFORM_WEIGHTS_BY_ARM.get(n, 1.0 / len(arm_names)) for n in arm_names], dtype=np.float32)
    weights, (filenames, patient_ids, _, raw_rgbs, gt_all) = w_raw / max(float(w_raw.sum()), 1e-6), va._collect_test_samples_from_loader(loader)
    n, h, w, arm5_idx = int(gt_all.shape[0]), int(gt_all.shape[1]), int(gt_all.shape[2]), next((i for i, nm in enumerate(arm_names) if "arm5" in nm), len(arm_names) - 1)
    use_amp, tmp_obj = (device.type == "cuda" and amp_dtype is not None), tempfile.TemporaryDirectory(prefix="g7_clin_bench_")
    mm_paths, mm_raw_paths, arm_cls, arm_raw_mious = [], [], [], {}
    cal_accs = {k: PixelCalibrationAccumulator(15) for k in ("author_raw", "author_step0", "arm5_uncal", "arm5_cal", "ens4_uncal_linear", "ens4_cal_logprob_eps1e3")}
    try:
        for idx, spec in enumerate(arm_specs):
            mm_p, mm_rp, mod = Path(tmp_obj.name) / f"arm_{idx}.dat", Path(tmp_obj.name) / f"arm_raw_{idx}.dat", _load_arm(spec, device)
            mm_paths.append(mm_p)
            mm = np.memmap(mm_p, dtype=np.float16, mode="w+", shape=(n, dl.NUM_FOREGROUND, h, w))
            mm_r = np.memmap(mm_rp, dtype=np.float16, mode="w+", shape=(n, dl.NUM_FOREGROUND, h, w)) if collect_raw_1x_and_cal else None
            if collect_raw_1x_and_cal:
                mm_raw_paths.append(mm_rp)
            cls_a, r_i, r_u, off = np.zeros((n, dl.NUM_FOREGROUND), dtype=np.float32), np.zeros(dl.NUM_FOREGROUND, dtype=np.int64), np.zeros(dl.NUM_FOREGROUND, dtype=np.int64), 0
            with torch.no_grad():
                for batch in loader:
                    imgs = s2_eval._extract_img_gt(batch)[0].to(device, non_blocking=True)
                    b, (p_t, c_d) = int(imgs.shape[0]), ens.apply_tta_probs(mod, imgs, ttas[idx], device, amp_dtype=amp_dtype)
                    p_np = p_t.detach().cpu().numpy().astype(np.float32)
                    mm[off : off + b], cls_a[off : off + b] = ens.encode_centered_logprobs_fp16(p_np, axis=1), c_d["tta"].detach().cpu().numpy().astype(np.float32)
                    if mm_r is not None:
                        pr1 = p_np if ttas[idx] == "raw_1x" else F.softmax((o1["seg_logits"] if isinstance(o1 := _run_autocast_fwd(mod, imgs, device, amp_dtype, use_amp), dict) else o1)[:, : dl.NUM_FOREGROUND].float(), dim=1).detach().cpu().numpy().astype(np.float32)
                        mm_r[off : off + b] = ens.encode_centered_logprobs_fp16(pr1, axis=1)
                        ip_r, up_r = va._accumulate_patient_iu(np.argmax(pr1, axis=1).astype(np.uint8), gt_all[off : off + b], np.zeros(b, dtype=np.int32), 1)
                        r_i, r_u = r_i + ip_r[0], r_u + up_r[0]
                    off += b
            _close_mms([mm] + ([mm_r] if mm_r is not None else []))
            if mm_r is not None:
                arm_raw_mious[arm_names[idx]] = round(float(ens.summarize_iu(r_i, r_u)["mIoU"]) * 100.0, 2)
            arm_cls.append(cls_a)
            if spec.get("model") is None:
                del mod
                _free_cuda(device)
        s1_m, s1_norm = _load_s1_expert(stage1_model_override, stage1_cls_ckpt, device)
        a_s1_m, a_s1_norm = _load_s1_expert(author_stage1_model_override, author_stage1_ckpt, device) if (author_stage1_model_override is not None or author_stage1_ckpt is not None) else (s1_m, s1_norm)
        auth_m = author_model_override.to(device).eval() if author_model_override is not None else (wim.build_and_load_deeplabv3plus(author_stage2_ckpt, device=device, strict=True) if author_stage2_ckpt else None)
        s1_cls, pred_auth_raw, pred_auth, off = np.zeros((n, dl.NUM_FOREGROUND), dtype=np.float32), (np.zeros((n, h, w), dtype=np.uint8) if auth_m is not None else None), (np.zeros((n, h, w), dtype=np.uint8) if auth_m is not None else None), 0
        with torch.no_grad():
            for batch in loader:
                imgs = s2_eval._extract_img_gt(batch)[0].to(device, non_blocking=True)
                b, in01 = int(imgs.shape[0]), prepare_stage1_input(imgs, stage1_norm_mode="zero_one", images_are_imagenet_norm=True)
                s1_cls[off : off + b] = s1_m.forward_cam(in01 if s1_norm == "zero_one" else prepare_stage1_input(imgs, s1_norm, True))[1].float().cpu().numpy()
                if auth_m is not None and pred_auth is not None and pred_auth_raw is not None:
                    as1_p = a_s1_m.forward_cam(in01 if a_s1_norm == "zero_one" else prepare_stage1_input(imgs, a_s1_norm, True))[1].float()
                    lg1 = (o["seg_logits"] if isinstance(o := auth_m(in01), dict) else o)[:, : dl.NUM_FOREGROUND].float()
                    pr_ms = s2_eval._ms_tta_probs(auth_m, in01, lg1, (0.75, 1.0, 1.25), device, amp_dtype=None)
                    gate_a = s2_eval._make_cgm_gate(as1_p, 0.10)
                    pred_auth_raw[off : off + b], pred_auth[off : off + b] = torch.argmax(lg1, dim=1).cpu().numpy().astype(np.uint8), s2_eval._apply_prob_cgm(pr_ms, gate_a).astype(np.uint8)
                    if collect_raw_1x_and_cal:
                        cal_accs["author_raw"].update(F.softmax(lg1, dim=1).cpu().numpy(), gt_all[off : off + b])
                        cal_accs["author_step0"].update((torch.clamp(pr_ms, min=1e-7) * gate_a).cpu().numpy(), gt_all[off : off + b])
                off += b
        pred_arms, pred_arm5, pred_ens4_lin, pred_ens4_no, pred_ens4 = {nm: np.zeros((n, h, w), dtype=np.uint8) for nm in arm_names}, np.zeros((n, h, w), dtype=np.uint8), np.zeros((n, h, w), dtype=np.uint8), np.zeros((n, h, w), dtype=np.uint8), np.zeros((n, h, w), dtype=np.uint8)
        bm5, bm4 = ens._gate_masks_from_thresholds(arm_cls[arm5_idx], va.GATE7_ARM5_SELF_CGM_THRESHOLDS), ens._gate_masks_from_thresholds(ens._combine_5_expert_probs(s1_cls, arm_cls, weights, "balanced_s1_s2_40_60"), va.GATE7_4ARM_CGM_THRESHOLDS)
        arm_inter, arm_union = np.zeros((len(arm_names), dl.NUM_FOREGROUND), dtype=np.int64), np.zeros((len(arm_names), dl.NUM_FOREGROUND), dtype=np.int64)
        mms = [np.memmap(p, dtype=np.float16, mode="r", shape=(n, dl.NUM_FOREGROUND, h, w)) for p in mm_paths]
        try:
            for st in range(0, n, 256):
                en, z_idx = min(n, st + 256), np.zeros(min(n, st + 256) - st, dtype=np.int32)
                for m_i, mm_i in enumerate(mms):
                    pr_m = np.argmax(np.asarray(mm_i[st:en], dtype=np.float16), axis=1).astype(np.uint8)
                    pred_arms[arm_names[m_i]][st:en] = pr_m
                    ip_m, up_m = va._accumulate_patient_iu(pr_m, gt_all[st:en], z_idx, 1)
                    arm_inter[m_i], arm_union[m_i] = arm_inter[m_i] + ip_m[0], arm_union[m_i] + up_m[0]
                a5_p = ens._fuse_memmap_chunk([mms[arm5_idx]], st, en, np.asarray([1.0], dtype=np.float32), [1.0], "uncal_linear")
                f4_lin, f4_lp = ens._fuse_memmap_chunk(mms, st, en, weights, betas, "uncal_linear"), ens._fuse_memmap_chunk(mms, st, en, weights, betas, "cal_logprob_eps1e3", eps_log=1e-3)
                pred_arm5[st:en], pred_ens4_lin[st:en], pred_ens4_no[st:en], pred_ens4[st:en] = va._apply_bitmask_argmax_np(a5_p, bm5[st:en]), np.argmax(f4_lin, axis=1).astype(np.uint8), np.argmax(f4_lp, axis=1).astype(np.uint8), va._apply_bitmask_argmax_np(f4_lp, bm4[st:en])
                if collect_raw_1x_and_cal:
                    cal_accs["arm5_uncal"].update(a5_p, gt_all[st:en])
                    cal_accs["arm5_cal"].update(ens.reconstruct_probs_from_centered(mms[arm5_idx][st:en], beta=betas[arm5_idx], axis=1), gt_all[st:en])
                    cal_accs["ens4_uncal_linear"].update(f4_lin, gt_all[st:en])
                    cal_accs["ens4_cal_logprob_eps1e3"].update(va._softmax_np(f4_lp, axis=1), gt_all[st:en])
        finally:
            _close_mms(mms)
        if mm_raw_paths:
            r_ei, r_eu = np.zeros(dl.NUM_FOREGROUND, dtype=np.int64), np.zeros(dl.NUM_FOREGROUND, dtype=np.int64)
            mms_r = [np.memmap(p, dtype=np.float16, mode="r", shape=(n, dl.NUM_FOREGROUND, h, w)) for p in mm_raw_paths]
            try:
                for st in range(0, n, 256):
                    en = min(n, st + 256)
                    ip_e, up_e = va._accumulate_patient_iu(np.argmax(ens._fuse_memmap_chunk(mms_r, st, en, weights, betas, "cal_logprob_eps1e3", eps_log=1e-3), axis=1).astype(np.uint8), gt_all[st:en], np.zeros(en - st, dtype=np.int32), 1)
                    r_ei, r_eu = r_ei + ip_e[0], r_eu + up_e[0]
            finally:
                _close_mms(mms_r)
            arm_raw_mious["ens4_raw_1x"] = round(float(ens.summarize_iu(r_ei, r_eu)["mIoU"]) * 100.0, 2)
        if pred_auth_raw is not None:
            arm_raw_mious["author_raw"] = _pooled_miou_pct(pred_auth_raw, gt_all)
        arm_mious = {nm: round(float(ens.summarize_iu(arm_inter[i], arm_union[i])["mIoU"]) * 100.0, 2) for i, nm in enumerate(arm_names)}
        preds_all: Dict[str, np.ndarray] = {**pred_arms, "arm5_no_cgm": pred_arms[arm_names[arm5_idx]], "arm5_cgm": pred_arm5, "ens4_uncal_linear": pred_ens4_lin, "ens4_no_cgm": pred_ens4_no, "ens4_cgm": pred_ens4, **({"author_raw": pred_auth_raw, "author_step0": pred_auth} if (pred_auth_raw is not None and pred_auth is not None) else {})}
        return {"filenames": filenames, "patient_ids": patient_ids, "raw_rgbs": raw_rgbs, "gt_all": gt_all, "arm_pooled_miou_pct": arm_mious, "raw_1x_miou_pct": arm_raw_mious, "pixel_calibration": {k: acc.compute() for k, acc in cal_accs.items() if acc.total_px > 0}, "preds": preds_all}
    finally:
        s1_m = a_s1_m = auth_m = None
        tmp_obj.cleanup()
        _free_cuda(device)


def _run_luad_2x_zoom_ablation(arm_specs: Sequence[Mapping[str, Any]], luad_loader: DataLoader, gt_all: np.ndarray, p_idx: np.ndarray, n_pat: int, device: torch.device, amp_dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    """Magnification-matched 2x zoom (10x->20x) quadrant-stitched ablation on arm5_no_cgm & ens4_uncal_linear."""
    n, h, w = int(gt_all.shape[0]), int(gt_all.shape[1]), int(gt_all.shape[2])
    h2, w2 = h // 2, w // 2
    arm_names = [str(sp["name"]).strip().lower() for sp in arm_specs]
    arm5_idx, use_amp = next((i for i, nm in enumerate(arm_names) if "arm5" in nm), len(arm_names) - 1), (device.type == "cuda" and amp_dtype is not None)
    sum_p, pred_a5 = np.zeros((n, dl.NUM_FOREGROUND, h, w), dtype=np.float32), np.zeros((n, h, w), dtype=np.uint8)
    for idx, spec in enumerate(arm_specs):
        mod, off = _load_arm(spec, device), 0
        with torch.no_grad():
            for batch in luad_loader:
                imgs = s2_eval._extract_img_gt(batch)[0].to(device, non_blocking=True)
                b = int(imgs.shape[0])
                quads = (imgs[:, :, :h2, :w2], imgs[:, :, :h2, w2:], imgs[:, :, h2:, :w2], imgs[:, :, h2:, w2:])
                q_pr = [F.interpolate(F.softmax((o["seg_logits"] if isinstance(o := _run_autocast_fwd(mod, F.interpolate(q, size=(h, w), mode="bilinear", align_corners=False), device, amp_dtype, use_amp), dict) else o)[:, : dl.NUM_FOREGROUND].float(), dim=1), size=(int(q.shape[2]), int(q.shape[3])), mode="bilinear", align_corners=False) for q in quads]
                full_p = torch.cat([torch.cat([q_pr[0], q_pr[1]], dim=3), torch.cat([q_pr[2], q_pr[3]], dim=3)], dim=2).cpu().numpy()
                sum_p[off : off + b] += full_p / float(len(arm_specs))
                if idx == arm5_idx:
                    pred_a5[off : off + b] = np.argmax(full_p, axis=1).astype(np.uint8)
                off += b
        if spec.get("model") is None:
            del mod
            _free_cuda(device)
    out_z: Dict[str, Any] = {}
    for key, pr_arr in (("arm5_no_cgm_2x_zoom", pred_a5), ("ens4_uncal_linear_2x_zoom", np.argmax(sum_p, axis=1).astype(np.uint8))):
        ip, up = va._accumulate_patient_iu(pr_arr, gt_all, p_idx, n_pat)
        pm = ens.compute_per_patient_miou(ip, up)
        out_z[key] = {**ens.summarize_iu(ip.sum(axis=0), up.sum(axis=0)), "patient_macro_miou": round(float(np.mean(pm)) if pm.size else 0.0, 4)}
    return out_z


def evaluate_gate7_luad_cross_organ(arm_specs: Sequence[Mapping[str, Any]], luad_loader: DataLoader, stage1_cls_ckpt: Optional[Union[str, Path]], author_stage1_ckpt: Optional[Union[str, Path]], author_stage2_ckpt: Optional[Union[str, Path]], output_dir: Union[str, Path], device: torch.device, amp_dtype: Optional[torch.dtype] = None, run_zoom_ablation: bool = True, dpi: int = 300, **kwargs: Any) -> Dict[str, Any]:
    """Run full zero-shot cross-organ LUAD-HistoSeg validation, 2x zoom ablation, paired stats & overlay sheet."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    streamed = _stream_gate7_predictions(arm_specs, luad_loader, stage1_cls_ckpt, author_stage1_ckpt, author_stage2_ckpt, device, amp_dtype, collect_raw_1x_and_cal=False, **kwargs)
    gt_all, pids = streamed["gt_all"], [extract_luad_case_id(fn) for fn in streamed["filenames"]]
    uniq = sorted(set(pids))
    p_idx, models_out, per_case_out, iu_by_model = np.asarray([uniq.index(p) for p in pids], dtype=np.int32), {}, {}, {}
    for mk, pred_arr in streamed["preds"].items():
        ip, up = va._accumulate_patient_iu(pred_arr, gt_all, p_idx, len(uniq))
        iu_by_model[mk], (summ, pm) = (ip, up), (ens.summarize_iu(ip.sum(axis=0), up.sum(axis=0)), ens.compute_per_patient_miou(ip, up))
        models_out[mk] = {**summ, "patient_macro_miou": round(float(np.mean(pm)) if pm.size else 0.0, 4), "patient_macro_std": round(float(np.std(pm, ddof=1)) if pm.size > 1 else 0.0, 4), "bootstrap_95ci": ens.bootstrap_pooled_miou_ci(ip, up, n_boot=2000, seed=42)}
        per_case_out[mk] = va._patient_class_iou_dict(uniq, ip, up)
    paired_stats = {f"{a}_vs_{b}": {**ens.paired_patient_swap_permutation_test(*iu_by_model[a], *iu_by_model[b], n_perm=2000, seed=42), **ens.paired_wilcoxon_and_dz(ens.compute_per_patient_miou(*iu_by_model[a]), ens.compute_per_patient_miou(*iu_by_model[b]))} for a, b in (("arm5_no_cgm", "author_step0"), ("arm5_cgm", "author_step0"), ("ens4_uncal_linear", "author_step0"), ("ens4_no_cgm", "author_step0"), ("ens4_cgm", "author_step0"), ("ens4_uncal_linear", "ens4_no_cgm")) if a in iu_by_model and b in iu_by_model}
    zoom_res = _run_luad_2x_zoom_ablation(arm_specs, luad_loader, gt_all, p_idx, len(uniq), device, amp_dtype) if run_zoom_ablation else None
    ov_png = out_p / "luad_zero_shot_overlays.png"
    ov_files = render_luad_overlays_sheet(streamed["raw_rgbs"], gt_all, streamed["preds"], streamed["filenames"], pids, ov_png, k=4, dpi=dpi)
    va._atomic_write_json(out_p / "luad_per_patient_metrics.json", {"case_ids": uniq, "per_case": per_case_out})
    payload = {"benchmark": "zero_shot_bcss_to_luad_histoseg", "label_mapping": "LUAD [0:TE, 1:NEC, 2:LYM, 3:TAS, 4:BACK] -> BCSS [0:TUM, 1:STR, 2:LYM, 3:NEC, 4:BACK]", "n_patches": int(gt_all.shape[0]), "n_patients": len(uniq), "case_ids": uniq, "models": models_out, "paired_patient_stats": paired_stats, "magnification_2x_zoom_ablation": zoom_res, "per_case_metrics": per_case_out, "overlay_sheet": ov_png.name, "overlay_filenames": ov_files}
    va._atomic_write_json(out_p / "gate7_luad_cross_organ.json", payload)
    print(f"[LUAD CROSS-ORGAN] n_patches={payload['n_patches']} n_patients={payload['n_patients']} arm5={models_out['arm5_no_cgm']['mIoU']:.4f} ens4_lin={models_out['ens4_uncal_linear']['mIoU']:.4f} ens4_cgm={models_out['ens4_cgm']['mIoU']:.4f}", flush=True)
    return payload


def run_gate7_clinical_and_luad_benchmark(arm_specs: Sequence[Mapping[str, Any]], bcss_test_loader: DataLoader, luad_test_loader: Optional[DataLoader], stage1_cls_ckpt: Optional[Union[str, Path]], author_stage1_ckpt: Optional[Union[str, Path]], author_stage2_ckpt: Optional[Union[str, Path]], output_dir: Union[str, Path], zip_path: Union[str, Path], device: torch.device, amp_dtype: Optional[torch.dtype] = None, selected_ttas: Optional[Sequence[str]] = None, track: str = "track_a_clean", n_warmup: int = 5, n_iters: int = 25, n_repeats: int = 5, dpi: int = 300, stage1_model_override: Optional[nn.Module] = None, author_stage1_model_override: Optional[nn.Module] = None, author_model_override: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Execute full Gate-7 Clinical Biomarkers, GDC Survival, Matched GPU Pareto, Calibration & LUAD Suite + ZIP."""
    out_p, skw = Path(output_dir), dict(selected_ttas=selected_ttas, track=track, stage1_model_override=stage1_model_override, author_stage1_model_override=author_stage1_model_override, author_model_override=author_model_override)
    out_p.mkdir(parents=True, exist_ok=True)
    prov = resolve_provenance_metadata(arm_specs, stage1_cls_ckpt, author_stage1_ckpt, author_stage2_ckpt, surv_mod.GDC_CLINICAL_MANIFEST_PATH)
    bcss = _stream_gate7_predictions(arm_specs, bcss_test_loader, stage1_cls_ckpt, author_stage1_ckpt, author_stage2_ckpt, device, amp_dtype, collect_raw_1x_and_cal=True, **skw)
    bcss_mious = {**bcss["arm_pooled_miou_pct"], **{mk: _pooled_miou_pct(pr, bcss["gt_all"]) for mk, pr in bcss["preds"].items()}}
    clin_preds = {k: bcss["preds"][k] for k in ("arm5_cgm", "ens4_no_cgm", "ens4_cgm", "author_step0") if k in bcss["preds"]}
    clin_res = compute_gate7_clinical_biomarkers(bcss["patient_ids"], bcss["raw_rgbs"], bcss["gt_all"], clin_preds, out_p, primary_model="ens4_cgm", dpi=dpi)
    gpu_res = benchmark_gate7_gpu_efficiency(arm_specs, author_stage2_ckpt, s2_eval._extract_img_gt(next(iter(bcss_test_loader)))[0][:4], device, amp_dtype, out_p, reference_mious=bcss_mious, reference_mious_raw_1x=bcss["raw_1x_miou_pct"], selected_ttas=selected_ttas, stage1_cls_ckpt=stage1_cls_ckpt, stage1_model_override=stage1_model_override, n_warmup=n_warmup, n_iters=n_iters, n_repeats=n_repeats, author_model_override=author_model_override, dpi=dpi)
    luad_res = evaluate_gate7_luad_cross_organ(arm_specs, luad_test_loader, stage1_cls_ckpt, author_stage1_ckpt, author_stage2_ckpt, out_p, device, amp_dtype, dpi=dpi, **skw) if luad_test_loader is not None else None
    shutil.copy2(out_p / "bland_altman_tsr_vtp_bw.png", out_p / "stage2_training_loss.png")
    combined = {"benchmark_suite": "gate7_track_a_clinical_gpu_luad", "track": track, "provenance": prov, "arm_order": [str(s["name"]).strip().lower() for s in arm_specs], "bcss_pooled_miou_pct": bcss_mious, "bcss_raw_1x_miou_pct": bcss["raw_1x_miou_pct"], "bcss_pixel_calibration": bcss["pixel_calibration"], "clinical_biomarkers": clin_res, "gpu_efficiency": gpu_res, "luad_cross_organ": luad_res}
    for fn in ("GATE7_CLINICAL_BENCH_METRICS.json", "STAGE2_TRAINING_METRICS.json"):
        va._atomic_write_json(out_p / fn, combined)
    va._atomic_write_zip(Path(zip_path), {p.name: p for p in sorted(out_p.iterdir()) if p.is_file()})
    return combined
