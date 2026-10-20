"""
Pipeline Orchestrator Wrapper for Google Colab & Drive WSSS Experiments (SE374.R11).

Re-exports the modular Google Drive package (`src.colab_pkg`) so callers can invoke
`run_bcss_wsss_experiment(...)` from either `src.pipeline.colab_runner` or `src.colab_pkg`.

All hyphens strictly use standard ASCII 0x2D ('-').
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colab_pkg import (  # noqa: E402
    ASPP,
    CLASS_CODES,
    CLASS_LABELS,
    DEFAULT_MAX_SANITY_SAMPLES,
    DEFAULT_MIN_SANITY_PATIENTS,
    DEFAULT_MULTISCALE_FACTORS,
    DEFAULT_MULTISCALE_WEIGHTS,
    DEFAULT_PRIOR_WEIGHTS,
    DEFAULT_SOFT_CLS_HIGH_THRESHOLDS,
    DEFAULT_SOFT_CLS_LOW_THRESHOLDS,
    DEFAULT_V2_PRIOR_WEIGHTS,
    DISCRETE_PALETTE,
    GDRIVE_CKPT_ID,
    GDRIVE_DATASET_ID,
    GDRIVE_STAGE1_ARML_ID,
    GDRIVE_STAGE1_ID,
    GDRIVE_STAGE2_ID,
    HE_DECONV_MATRIX,
    HF_DATASET_URL,
    HF_STAGE1_URL,
    HF_STAGE2_URL,
    SOTA_ENSEMBLE_PRIOR_WEIGHTS,
    TARGET_PATIENTS,
    Bottleneck,
    ColabSegmentationEvaluator,
    Decoder,
    DeepLabV3Plus,
    MultiMethodWSSSEngine,
    MultiMethodWSSSEngineV2,
    OtsuClinicalGating,
    ResBlock,
    ResBlockBot,
    ResBlockBotStage1,
    ResBlockStage1,
    ResNet101,
    WSSSTissueStage1NetCAM,
    build_and_load_deeplabv3plus,
    build_and_load_stage1_netcam,
    colorize_mask,
    compute_paired_statistical_tests,
    download_checkpoint_resilient,
    download_dataset_resilient,
    ensure_bcss_dataset,
    ensure_checkpoints,
    export_evaluation_artifacts,
    extract_patient_id,
    is_valid_torch_checkpoint,
    is_valid_zip_archive,
    load_wsss_models,
    package_and_download_results,
    prepare_dataset_and_splits,
    print_evaluation_summary,
    render_ablation_bar_chart,
    render_clinical_comparison_figure,
    render_confusion_matrix_heatmap,
    resolve_test_dirs,
    run_bcss_wsss_experiment,
    run_package_self_test,
    run_streaming_evaluation,
    safe_torch_load,
    sanitize_for_json,
    select_diverse_clinical_samples,
    select_patient_files,
    set_reproducible_seed,
)

__all__ = [
    "ASPP",
    "Bottleneck",
    "CLASS_CODES",
    "CLASS_LABELS",
    "ColabSegmentationEvaluator",
    "DEFAULT_MAX_SANITY_SAMPLES",
    "DEFAULT_MIN_SANITY_PATIENTS",
    "DEFAULT_MULTISCALE_FACTORS",
    "DEFAULT_MULTISCALE_WEIGHTS",
    "DEFAULT_PRIOR_WEIGHTS",
    "DEFAULT_SOFT_CLS_HIGH_THRESHOLDS",
    "DEFAULT_SOFT_CLS_LOW_THRESHOLDS",
    "DEFAULT_V2_PRIOR_WEIGHTS",
    "DISCRETE_PALETTE",
    "Decoder",
    "DeepLabV3Plus",
    "GDRIVE_CKPT_ID",
    "GDRIVE_DATASET_ID",
    "GDRIVE_STAGE1_ARML_ID",
    "GDRIVE_STAGE1_ID",
    "GDRIVE_STAGE2_ID",
    "HE_DECONV_MATRIX",
    "HF_DATASET_URL",
    "HF_STAGE1_URL",
    "HF_STAGE2_URL",
    "MultiMethodWSSSEngine",
    "MultiMethodWSSSEngineV2",
    "OtsuClinicalGating",
    "ResBlock",
    "ResBlockBot",
    "ResBlockBotStage1",
    "ResBlockStage1",
    "ResNet101",
    "SOTA_ENSEMBLE_PRIOR_WEIGHTS",
    "TARGET_PATIENTS",
    "WSSSTissueStage1NetCAM",
    "build_and_load_deeplabv3plus",
    "build_and_load_stage1_netcam",
    "colorize_mask",
    "compute_paired_statistical_tests",
    "download_checkpoint_resilient",
    "download_dataset_resilient",
    "ensure_bcss_dataset",
    "ensure_checkpoints",
    "export_evaluation_artifacts",
    "extract_patient_id",
    "is_valid_torch_checkpoint",
    "is_valid_zip_archive",
    "load_wsss_models",
    "package_and_download_results",
    "prepare_dataset_and_splits",
    "print_evaluation_summary",
    "render_ablation_bar_chart",
    "render_clinical_comparison_figure",
    "render_confusion_matrix_heatmap",
    "resolve_test_dirs",
    "run_bcss_wsss_experiment",
    "run_package_self_test",
    "run_streaming_evaluation",
    "safe_torch_load",
    "sanitize_for_json",
    "select_diverse_clinical_samples",
    "select_patient_files",
    "set_reproducible_seed",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SE374.R11 BCSS-WSSS Modular Colab Runner")
    parser.add_argument("--self-test", action="store_true", help="Run full package self-test")
    parser.add_argument("--sanity", action="store_true", help="Run 150-patch Sanity Mode")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "quick", "sanity", "full"],
        default=None,
        help="Execution mode: smoke (20 patches), quick/sanity (150 patches), or full (4986 patches)",
    )
    parser.add_argument("--v2", action="store_true", default=True, help="Include V2 Ultra-SOTA methods M6 and M7 (default: True)")
    parser.add_argument("--no-v2", action="store_true", help="Evaluate only V1 methods M1..M5")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed-precision autocast on GPU")
    parser.add_argument("--mode-tag", type=str, default=None, help="Mode tag for output folder")
    parser.add_argument("--data-root", type=str, default="/content/BCSS-WSSS")
    parser.add_argument("--checkpoint-dir", type=str, default="/content/checkpoints")
    parser.add_argument("--output-dir", type=str, default="/content")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Disable interactive plt.show()")
    parser.add_argument("--no-download", action="store_true", help="Disable browser file download")
    args = parser.parse_args()

    if args.self_test:
        run_package_self_test()
    else:
        run_bcss_wsss_experiment(
            sanity_mode=args.sanity,
            mode=args.mode,
            mode_tag=args.mode_tag,
            data_root=args.data_root,
            checkpoint_dir=args.checkpoint_dir,
            output_base_dir=args.output_dir,
            device=args.device,
            include_v2=(not args.no_v2),
            use_amp=args.amp,
            show_plots=not args.no_plots,
            trigger_download=not args.no_download,
        )
