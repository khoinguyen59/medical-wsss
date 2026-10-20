"""
High-Level End-to-End Orchestrator for SE374.R11 BCSS-WSSS Google Colab Pipeline.

Enables 1-to-3 line execution from any Google Colab notebook while keeping all
heavy logic cleanly modularized inside the Google Drive / repository code folder:
- downloader.py: Resilient HuggingFace CDN + Google Drive dataset & checkpoint fetching
- model_loader.py: Stage-1 ResNet38d Net_CAM, Stage-2 DeepLabV3+, Otsu Gating, M1..M5 Engine
- evaluator.py: 4-class & 5-class metrics, paired Wilcoxon / Student t-tests, JSON/MD/CSV export
- visualizer.py: 300-DPI clinical figures, confusion matrix, ablation chart, and ZIP archiver

All hyphens strictly use standard ASCII 0x2D ('-').
"""

import datetime
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def _ensure_utf8_stdio() -> None:
    """Reconfigures stdout and stderr to UTF-8 on Windows consoles to prevent cp1252 UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_ensure_utf8_stdio()

try:
    from .downloader import (
        DEFAULT_MAX_SANITY_SAMPLES,
        DEFAULT_MIN_SANITY_PATIENTS,
        TARGET_PATIENTS,
        ensure_bcss_dataset,
        ensure_checkpoints,
        extract_patient_id,
        is_valid_torch_checkpoint,
        is_valid_zip_archive,
        select_patient_files,
    )
    from .evaluator import (
        CLASS_CODES,
        CLASS_LABELS,
        ColabSegmentationEvaluator,
        compute_paired_statistical_tests,
        export_evaluation_artifacts,
        print_evaluation_summary,
    )
    from .model_loader import (
        MultiMethodWSSSEngine,
        MultiMethodWSSSEngineV2,
        OtsuClinicalGating,
        load_wsss_models,
    )
    from .visualizer import (
        package_and_download_results,
        render_ablation_bar_chart,
        render_clinical_comparison_figure,
        render_confusion_matrix_heatmap,
    )
except ImportError:
    from downloader import (
        DEFAULT_MAX_SANITY_SAMPLES,
        DEFAULT_MIN_SANITY_PATIENTS,
        TARGET_PATIENTS,
        ensure_bcss_dataset,
        ensure_checkpoints,
        extract_patient_id,
        is_valid_torch_checkpoint,
        is_valid_zip_archive,
        select_patient_files,
    )
    from evaluator import (
        CLASS_CODES,
        CLASS_LABELS,
        ColabSegmentationEvaluator,
        compute_paired_statistical_tests,
        export_evaluation_artifacts,
        print_evaluation_summary,
    )
    from model_loader import (
        MultiMethodWSSSEngine,
        MultiMethodWSSSEngineV2,
        OtsuClinicalGating,
        load_wsss_models,
    )
    from visualizer import (
        package_and_download_results,
        render_ablation_bar_chart,
        render_clinical_comparison_figure,
        render_confusion_matrix_heatmap,
    )


def set_reproducible_seed(seed: int = 42, verbose: bool = True) -> Dict[str, Any]:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch and reports GPU diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_ok = torch.cuda.is_available()
    gpu_name = "CPU"
    vram_gb = 0.0

    if cuda_ok:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if verbose:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available : {cuda_ok}")
        if cuda_ok:
            print(f"GPU Model      : {gpu_name} ({vram_gb:.2f} GB VRAM)")
            print(f"CUDA Devices   : {torch.cuda.device_count()}")
        else:
            print(
                "[CẢNH BÁO]: Đang chạy trên CPU! "
                "Trên Colab vui lòng chọn Runtime -> Change runtime type -> T4/L4/A100 GPU."
            )

    return {
        "seed": seed,
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "device": torch.device("cuda" if cuda_ok else "cpu"),
    }


def run_streaming_evaluation(
    active_test_files: List[Path],
    test_mask_dir: Union[str, Path],
    model: torch.nn.Module,
    model_stage1: Optional[torch.nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    wsss_engine: Optional[Any] = None,
    use_full_d8_in_m5: bool = False,
    include_v2: bool = False,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    progress_interval: int = 25,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs streaming evaluation across active_test_files for WSSS methods (M1..M5 or V2 M1..M7)."""
    mask_dir = Path(test_mask_dir)
    if device is None:
        dev = next(model.parameters()).device
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device

    if wsss_engine is not None:
        engine = wsss_engine
        if use_amp and hasattr(engine, "use_amp"):
            engine.use_amp = bool(use_amp)
        if amp_dtype is not None and hasattr(engine, "amp_dtype"):
            engine.amp_dtype = amp_dtype
    else:
        engine = MultiMethodWSSSEngine(
            model_stage1=model_stage1,
            use_full_d8_in_m5=use_full_d8_in_m5,
            include_v2=include_v2,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    active_method_keys: List[str] = list(
        getattr(engine, "METHOD_KEYS", MultiMethodWSSSEngine.METHOD_KEYS)
    )
    run_v2_eval = bool(
        include_v2 or getattr(engine, "include_v2", False) or len(active_method_keys) > 5
    )
    if run_v2_eval and len(active_method_keys) < 7:
        for k in MultiMethodWSSSEngine.METHOD_KEYS_V2:
            if k not in active_method_keys:
                active_method_keys.append(k)

    method_names_vi: Dict[str, str] = dict(MultiMethodWSSSEngine.METHOD_NAMES_VI)
    method_names_vi.update(getattr(engine, "METHOD_NAMES_VI", {}))

    method_evaluators: Dict[str, ColabSegmentationEvaluator] = {
        m_key: ColabSegmentationEvaluator(num_classes=5, class_names=CLASS_CODES)
        for m_key in active_method_keys
    }

    transform = transforms.Compose([transforms.ToTensor()])

    if verbose:
        print(
            f"--> Bắt đầu thực thi đánh giá so sánh {len(active_method_keys)} Phương Pháp Nghiên Cứu "
            f"(Ablation & Multi-Method Benchmark) trên {len(active_test_files)} mảng ảnh bệnh nhân thật..."
        )
        print(f"--> Tiến trình chạy trên {dev} bằng PyTorch DeepLabV3+ ResNet-101 thực thụ...")

    total_inference_time = 0.0
    evaluated_count = 0

    for idx, img_path in enumerate(active_test_files):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue

        raw_img = Image.open(img_path).convert("RGB")
        raw_np = np.array(raw_img)
        gt_mask = np.array(Image.open(mask_path), dtype=np.int64)
        patient_id = extract_patient_id(img_path.name)

        img_tensor = transform(raw_img).unsqueeze(0).to(dev)

        t_start = time.perf_counter()
        if include_v2 and hasattr(engine, "predict_all_methods_v2"):
            preds_dict = engine.predict_all_methods_v2(
                model, img_tensor, raw_rgb=raw_np, model_stage1=model_stage1
            )
        else:
            preds_dict = engine.predict_all_methods(
                model, img_tensor, raw_rgb=raw_np, model_stage1=model_stage1
            )

        raw_4cls_dict = getattr(engine, "_last_raw_4cls_preds", {})

        for m_key, pred_4cls in preds_dict.items():
            if m_key not in method_evaluators:
                method_evaluators[m_key] = ColabSegmentationEvaluator(
                    num_classes=5, class_names=CLASS_CODES
                )
                if m_key not in active_method_keys:
                    active_method_keys.append(m_key)

            raw_base = raw_4cls_dict.get(m_key, pred_4cls)
            pred_raw_roi = raw_base.copy()
            pred_raw_roi[gt_mask == 4] = 4
            pred_final = engine.apply_pure_glass_guard(
                pred_4cls, raw_np, background_class_idx=4
            )
            pred_final[gt_mask == 4] = 4
            method_evaluators[m_key].update(
                pred_final,
                gt_mask,
                pred_raw_4cls=pred_raw_roi,
                patient_id=patient_id,
            )

        t_end = time.perf_counter()
        total_inference_time += t_end - t_start
        evaluated_count += 1

        if verbose and (
            (idx + 1) % progress_interval == 0 or (idx + 1) == len(active_test_files)
        ):
            width_digits = len(str(len(active_test_files)))
            print(
                f"  * Tiến trình: [{idx + 1:>{width_digits}}/{len(active_test_files)}] "
                f"mảng ảnh đã được đánh giá qua {len(preds_dict)} phương pháp..."
            )

    # Filter active_method_keys to those that were evaluated (or all if evaluated_count == 0)
    if evaluated_count > 0:
        evaluated_keys = [
            k for k in active_method_keys if method_evaluators[k].confusion_matrix.sum() > 0
        ]
        if evaluated_keys:
            active_method_keys = evaluated_keys
            method_evaluators = {k: method_evaluators[k] for k in active_method_keys}

    all_methods_metrics: Dict[str, Dict[str, Any]] = {}
    for m_key in active_method_keys:
        all_methods_metrics[m_key] = method_evaluators[m_key].compute_metrics()

    best_method_key = max(
        all_methods_metrics.keys(), key=lambda k: all_methods_metrics[k]["mIoU_4cls"]
    )
    baseline_key = (
        "M1_Baseline_SingleScale"
        if "M1_Baseline_SingleScale" in all_methods_metrics
        else active_method_keys[0]
    )
    baseline_metrics = all_methods_metrics[baseline_key]
    # Copy via dict() to prevent any circular self-reference
    metrics = dict(all_methods_metrics[best_method_key])

    fps = evaluated_count / max(total_inference_time, 1e-4)
    latency_ms = (total_inference_time / max(evaluated_count, 1)) * 1000.0

    statistical_tests = compute_paired_statistical_tests(
        method_evaluators, baseline_key=baseline_key
    )

    metrics["fps"] = fps
    metrics["latency_ms"] = latency_ms
    metrics["evaluated_count"] = evaluated_count
    metrics["total_inference_time"] = total_inference_time
    metrics["best_method_key"] = best_method_key
    metrics["best_method_name"] = method_names_vi.get(best_method_key, best_method_key)
    metrics["ablation_methods"] = {
        k: {mk: mv for mk, mv in v.items() if mk != "ablation_methods"}
        for k, v in all_methods_metrics.items()
    }
    metrics["statistical_tests"] = statistical_tests

    if verbose:
        print_evaluation_summary(
            metrics=metrics,
            all_methods_metrics=all_methods_metrics,
            method_keys=active_method_keys,
            method_names_vi=method_names_vi,
            statistical_tests=statistical_tests,
            device_name=str(dev),
        )

    return {
        "metrics": metrics,
        "all_methods_metrics": all_methods_metrics,
        "baseline_metrics": baseline_metrics,
        "best_method_key": best_method_key,
        "statistical_tests": statistical_tests,
        "method_evaluators": method_evaluators,
        "method_keys": active_method_keys,
        "method_names_vi": method_names_vi,
        "wsss_engine": engine,
        "evaluated_count": evaluated_count,
        "total_inference_time": total_inference_time,
    }


def run_bcss_wsss_experiment(
    sanity_mode: bool = False,
    mode: Optional[str] = None,
    mode_tag: Optional[str] = None,
    max_sanity_samples: int = DEFAULT_MAX_SANITY_SAMPLES,
    min_sanity_patients: int = DEFAULT_MIN_SANITY_PATIENTS,
    target_patients: Optional[List[str]] = None,
    data_root: Union[str, Path] = Path("/content/BCSS-WSSS"),
    zip_path: Union[str, Path] = Path("/content/BCSS-WSSS.zip"),
    extract_dir: Union[str, Path] = Path("/content"),
    checkpoint_dir: Union[str, Path] = Path("/content/checkpoints"),
    output_base_dir: Union[str, Path] = Path("/content"),
    legacy_dir: Optional[Union[str, Path]] = Path("/content/SE374_Colab_Results"),
    drive_cache_dir: Optional[Union[str, Path]] = None,
    drive_export_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
    use_full_d8_in_m5: bool = False,
    include_v2: bool = True,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    wsss_engine: Optional[Any] = None,
    vis_sample_count: int = 4,
    show_plots: bool = True,
    trigger_download: bool = True,
    progress_interval: int = 25,
) -> Dict[str, Any]:
    """High-level 1-line orchestrator for the BCSS-WSSS pipeline on Google Colab or local GPU/CPU.

    Supports both `sanity_mode=True/False` and `mode="smoke"|"quick"|"sanity"|"full"`.

    Example usage in a Google Colab notebook:
        from colab_pkg import run_bcss_wsss_experiment
        results = run_bcss_wsss_experiment(sanity_mode=False, mode_tag="T4_Full4986")
    """
    if mode is not None:
        m_norm = mode.strip().lower()
        if m_norm in ("smoke", "quick", "sanity"):
            sanity_mode = True
            if m_norm == "smoke" and max_sanity_samples == DEFAULT_MAX_SANITY_SAMPLES:
                max_sanity_samples = min(20, DEFAULT_MAX_SANITY_SAMPLES)
            if mode_tag is None:
                mode_tag = "T4_Smoke" if m_norm == "smoke" else "T4_Sanity150"
        elif m_norm == "full":
            sanity_mode = False
            if mode_tag is None:
                mode_tag = "T4_Full4986"
        else:
            raise ValueError(
                f"Unsupported mode={mode!r}. Expected one of: 'smoke', 'quick', 'sanity', 'full'."
            )

    # 1. Seed & Hardware Diagnostics
    env_info = set_reproducible_seed(seed=seed, verbose=True)
    dev = env_info["device"] if device is None else (
        torch.device(device) if isinstance(device, str) else device
    )
    gpu_name = env_info["gpu_name"]

    if mode_tag is None:
        mode_tag = "T4_Sanity150" if sanity_mode else "T4_Full4986"

    # 2. Dataset & Patient-Level Split Selection
    resolved_root, test_img_dir, test_mask_dir = ensure_bcss_dataset(
        data_root=data_root,
        zip_path=zip_path,
        extract_dir=extract_dir,
        drive_cache_dir=drive_cache_dir,
    )
    active_test_files, patient_to_files = select_patient_files(
        test_img_dir=test_img_dir,
        sanity_mode=sanity_mode,
        max_sanity_samples=max_sanity_samples,
        min_sanity_patients=min_sanity_patients,
        target_patients=target_patients,
    )

    # 3. Checkpoints & Models Loading
    ckpt_paths = ensure_checkpoints(
        checkpoint_dir=checkpoint_dir,
        drive_cache_dir=drive_cache_dir,
        require_stage2=True,
    )
    assert ckpt_paths["stage2"] is not None
    model_stage2, model_stage1, dev = load_wsss_models(
        stage2_ckpt=ckpt_paths["stage2"],
        stage1_ckpt=ckpt_paths["stage1"],
        device=dev,
    )

    # 4. Streaming Multi-Method Quantitative Evaluation & Statistical Audit
    eval_out = run_streaming_evaluation(
        active_test_files=active_test_files,
        test_mask_dir=test_mask_dir,
        model=model_stage2,
        model_stage1=model_stage1,
        device=dev,
        wsss_engine=wsss_engine,
        use_full_d8_in_m5=use_full_d8_in_m5,
        include_v2=include_v2,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        progress_interval=progress_interval,
        verbose=True,
    )

    metrics = eval_out["metrics"]
    all_methods_metrics = eval_out["all_methods_metrics"]
    best_method_key = eval_out["best_method_key"]
    statistical_tests = eval_out["statistical_tests"]
    active_engine = eval_out["wsss_engine"]
    active_method_keys = eval_out["method_keys"]
    active_method_names_vi = eval_out["method_names_vi"]
    evaluated_count = eval_out["evaluated_count"]

    # 5. Prepare Timestamped Output Directory
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"SE374_Colab_Results_DeepLabV3Plus_{mode_tag}_{timestamp_str}"
    output_dir = Path(output_base_dir) / run_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Render 300-DPI Academic Figures
    _, vis_img_path = render_clinical_comparison_figure(
        active_test_files=active_test_files,
        test_mask_dir=test_mask_dir,
        model=model_stage2,
        wsss_engine=active_engine,
        best_method_key=best_method_key,
        model_stage1=model_stage1,
        device=dev,
        output_dir=output_dir,
        legacy_dir=legacy_dir,
        sample_count=vis_sample_count,
        show_plots=show_plots,
    )
    _, cm_img_path = render_confusion_matrix_heatmap(
        confusion_matrix=metrics["confusion_matrix"],
        class_codes=CLASS_CODES,
        output_dir=output_dir,
        legacy_dir=legacy_dir,
        show_plots=show_plots,
    )
    _, bar_img_path = render_ablation_bar_chart(
        all_methods_metrics=all_methods_metrics,
        method_keys=active_method_keys,
        output_dir=output_dir,
        legacy_dir=legacy_dir,
        show_plots=show_plots,
    )

    # 7. Export JSON, Markdown, and CSV Reports
    exported_files = export_evaluation_artifacts(
        output_dir=output_dir,
        run_folder_name=run_folder_name,
        mode_tag=mode_tag,
        sanity_mode=sanity_mode,
        evaluated_count=evaluated_count,
        metrics=metrics,
        all_methods_metrics=all_methods_metrics,
        method_keys=active_method_keys,
        method_names_vi=active_method_names_vi,
        statistical_tests=statistical_tests,
        gpu_name=gpu_name,
        legacy_dir=legacy_dir,
    )

    # 8. Package Results into ZIP and Trigger Download
    zip_file_path = package_and_download_results(
        output_dir=output_dir,
        drive_export_dir=drive_export_dir,
        trigger_browser_download=trigger_download,
    )

    return {
        "run_id": run_folder_name,
        "output_dir": output_dir,
        "zip_path": zip_file_path,
        "data_root": resolved_root,
        "test_img_dir": test_img_dir,
        "test_mask_dir": test_mask_dir,
        "active_test_files": active_test_files,
        "patient_to_files": patient_to_files,
        "metrics": metrics,
        "all_methods_metrics": all_methods_metrics,
        "statistical_tests": statistical_tests,
        "exported_files": exported_files,
        "figures": {
            "clinical_comparison": vis_img_path,
            "confusion_matrix": cm_img_path,
            "ablation_bar_chart": bar_img_path,
        },
    }


def run_package_self_test() -> bool:
    """Executes a comprehensive self-test of all modules, edge cases, and end-to-end orchestration."""
    import json
    import sys
    import tempfile
    import zipfile

    pkg_dir = Path(__file__).resolve().parent
    pipeline_runner = pkg_dir.parent / "pipeline" / "colab_runner.py"

    # 1. Invariant 12 check on all package files
    files_to_check = list(pkg_dir.glob("*.py")) + list(pkg_dir.glob("*.md"))
    if pipeline_runner.is_file():
        files_to_check.append(pipeline_runner)
    forbidden = ("\u2014", "\u2013", "\u2015", "\u2212")
    for fpath in files_to_check:
        text = fpath.read_text(encoding="utf-8")
        for ch in forbidden:
            assert ch not in text, f"Forbidden dash {hex(ord(ch))} found in {fpath}"

    # 2. Verify parameter counts of Stage-1 and Stage-2 architectures
    try:
        from .evaluator import sanitize_for_json
        from .model_loader import DeepLabV3Plus, WSSSTissueStage1NetCAM
    except ImportError:
        from evaluator import sanitize_for_json
        from model_loader import DeepLabV3Plus, WSSSTissueStage1NetCAM

    m2 = DeepLabV3Plus(num_classes=4, output_stride=16)
    m1 = WSSSTissueStage1NetCAM(n_class=4)
    assert sum(p.numel() for p in m2.parameters()) == 59339940
    assert sum(p.numel() for p in m1.parameters()) == 105087296

    # 3. Verify OtsuClinicalGating & PureGlassGuard edge cases
    white_glass = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert np.all(OtsuClinicalGating.compute_tissue_mask(white_glass) == 0)
    assert np.all(
        OtsuClinicalGating.apply_gating(np.zeros((64, 64), dtype=np.uint8), white_glass) == 4
    )
    pale_nec = np.full((64, 64, 3), [215, 195, 210], dtype=np.uint8)
    assert np.all(
        MultiMethodWSSSEngine.apply_pure_glass_guard(
            np.full((64, 64), 3, dtype=np.uint8), pale_nec, background_class_idx=4
        )
        == 3
    )
    for bad_shape in [(32, 32), (32, 32, 4)]:
        try:
            OtsuClinicalGating.compute_tissue_mask(np.zeros(bad_shape, dtype=np.uint8))
            raise AssertionError(f"Expected ValueError for shape {bad_shape}")
        except ValueError:
            pass

    # 4. Verify circular-reference & NaN/Inf safety in sanitize_for_json and statistical test edge cases
    cyclic: Dict[str, Any] = {"a": 1, "nan_val": float("nan"), "inf_val": float("inf")}
    cyclic["self"] = cyclic
    sanitized_cyclic = sanitize_for_json(cyclic)
    assert json.dumps(sanitized_cyclic, allow_nan=False) is not None
    assert sanitized_cyclic["nan_val"] == 0.0 and sanitized_cyclic["inf_val"] == 0.0

    eval_empty_a = ColabSegmentationEvaluator()
    eval_empty_b = ColabSegmentationEvaluator()
    stats_empty = compute_paired_statistical_tests(
        {"M1_Baseline_SingleScale": eval_empty_a, "M2": eval_empty_b}
    )
    assert stats_empty["M2"]["miou_paired_test"]["wilcoxon_pvalue"] == 1.0
    assert "Macro_F1_4cls" in eval_empty_a.compute_metrics()

    # 5. Verify rectangular (H != W) 8-view Dihedral D4/D8 TTA and predict_m1..m7 parity
    engine = MultiMethodWSSSEngine(model_stage1=m1, use_full_d8_in_m5=True)
    m1.eval()
    m2.eval()
    sq_inp = torch.rand(1, 3, 32, 32)
    sq_rgb = (sq_inp.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    preds_sq = engine.predict_all_methods(m2, sq_inp, raw_rgb=sq_rgb, model_stage1=m1)
    assert np.array_equal(engine.predict_m1(m2, sq_inp), preds_sq["M1_Baseline_SingleScale"])
    assert np.array_equal(
        engine.predict_m2(m2, sq_inp, model_stage1=m1), preds_sq["M2_MultiView_D4_TTA"]
    )
    assert np.array_equal(
        engine.predict_m3(m2, sq_inp, model_stage1=m1),
        preds_sq["M3_LongTail_Prior_Calibration"],
    )
    assert np.array_equal(
        engine.predict_m4(m2, sq_inp, raw_rgb=sq_rgb, model_stage1=m1),
        preds_sq["M4_Spatial_Bilateral_Refinement"],
    )
    assert np.array_equal(
        engine.predict_m5(m2, sq_inp, raw_rgb=sq_rgb, model_stage1=m1),
        preds_sq["M5_Proposed_SOTA_Ensemble"],
    )

    rect_inp = torch.rand(3, 33, 47)
    rect_rgb = np.zeros((33, 47, 3), dtype=np.uint8)
    p_orig_r, p_d4_r, p_d8_r = MultiMethodWSSSEngine.forward_all_tta_views(
        m2, rect_inp, full_d8=True
    )
    assert p_orig_r.shape == (4, 33, 47)
    assert p_d4_r.shape == (4, 33, 47)
    assert p_d8_r.shape == (4, 33, 47)
    _, p_tta_r = MultiMethodWSSSEngine.forward_d4_tta(m2, rect_inp, full_d8=True)
    assert np.allclose(p_d8_r, p_tta_r, atol=1e-6)

    preds_rect_v2 = engine.predict_all_methods_v2(m2, rect_inp, raw_rgb=rect_rgb, model_stage1=m1)
    for k in MultiMethodWSSSEngine.METHOD_KEYS_V2:
        assert preds_rect_v2[k].shape == (33, 47)

    # 6. End-to-End Orchestrator Test on Synthetic Dataset & Mock Checkpoints (V1 & V2 & corruption check)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        corrupt_zip = tmp_p / "corrupt.zip"
        corrupt_zip.write_bytes(b"NOT_A_ZIP_HEADER" * 200)
        assert not is_valid_zip_archive(corrupt_zip, min_valid_bytes=100)
        assert not is_valid_torch_checkpoint(corrupt_zip, min_valid_bytes=100)

        data_root = tmp_p / "BCSS-WSSS"
        img_dir = data_root / "test" / "img"
        mask_dir = data_root / "test" / "mask"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        patients = [
            "TCGA-AQ-A04H",
            "TCGA-C8-A1HJ",
            "TCGA-AR-A0TQ",
            "TCGA-D8-A27F",
            "TCGA-E2-A14X",
        ]
        for pid in patients:
            for s_idx in range(2):
                fname = f"{pid}_patch_{s_idx:02d}.png"
                rgb_arr = np.random.randint(40, 220, (32, 32, 3), dtype=np.uint8)
                mask_arr = np.zeros((32, 32), dtype=np.uint8)
                mask_arr[:16, :16] = 0
                mask_arr[:16, 16:] = 1
                mask_arr[16:, :16] = 2
                mask_arr[16:28, 16:] = 3
                mask_arr[28:, 16:] = 4
                Image.fromarray(rgb_arr).save(img_dir / fname)
                Image.fromarray(mask_arr).save(mask_dir / fname)

        ckpt_dir = tmp_p / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        s1_file = ckpt_dir / "stage1_checkpoint_trained_on_bcss.pth"
        s2_file = ckpt_dir / "stage2_checkpoint_trained_on_bcss.pth"
        torch.save({"state_dict": {f"module.{k}": v for k, v in m1.state_dict().items()}}, s1_file)
        torch.save({"state_dict": {f"module.{k}": v for k, v in m2.state_dict().items()}}, s2_file)
        assert is_valid_torch_checkpoint(s1_file, min_valid_bytes=100)
        assert is_valid_torch_checkpoint(s2_file, min_valid_bytes=100)

        out_base = tmp_p / "outputs"
        legacy_p = tmp_p / "legacy_results"
        drive_backup = tmp_p / "drive_backup"

        res = run_bcss_wsss_experiment(
            mode="smoke",
            mode_tag="SelfTest_Sanity",
            max_sanity_samples=10,
            min_sanity_patients=5,
            data_root=data_root,
            extract_dir=tmp_p,
            checkpoint_dir=ckpt_dir,
            output_base_dir=out_base,
            legacy_dir=legacy_p,
            drive_export_dir=drive_backup,
            seed=42,
            device="cpu",
            include_v2=True,
            vis_sample_count=2,
            show_plots=False,
            trigger_download=False,
        )

        assert res["zip_path"].is_file()
        assert is_valid_zip_archive(res["zip_path"], min_valid_bytes=100)
        assert (drive_backup / res["zip_path"].name).is_file()
        assert res["exported_files"]["json_path"].is_file()
        assert res["exported_files"]["md_path"].is_file()
        assert res["exported_files"]["csv_ablation_path"].is_file()
        assert res["exported_files"]["csv_class_path"].is_file()
        assert res["figures"]["clinical_comparison"].is_file()
        assert res["figures"]["confusion_matrix"].is_file()
        assert res["figures"]["ablation_bar_chart"].is_file()

        loaded_json = json.loads(res["exported_files"]["json_path"].read_text(encoding="utf-8"))
        assert loaded_json["evaluated_patches"] == 10
        assert "M5_Proposed_SOTA_Ensemble" in loaded_json["metrics"]["ablation_methods"]
        assert "M7_Proposed_Ultra_SOTA_Curriculum" in loaded_json["metrics"]["ablation_methods"]
        assert "Macro_F1_4cls" in loaded_json["metrics"]

        with zipfile.ZipFile(res["zip_path"], "r") as zf:
            names = zf.namelist()
            assert len(names) == 7, f"Expected 7 files in ZIP archive, got {len(names)}: {names}"

    # 7. Verify direct folder import simulation (as if src/colab_pkg is uploaded to Google Drive)
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    import colab_runner as direct_runner  # noqa: F401

    assert hasattr(direct_runner, "run_bcss_wsss_experiment")

    print("[SELF-TEST PASSED] Toàn bộ 5 module trong src/colab_pkg hoạt động hoàn hảo 100%!")
    return True

