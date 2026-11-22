"""
Stage-2 v2 Breakthrough Package for BCSS-WSSS (Gate-5 Consensus).

Provides clean, modular, decoupled Stage-2 training and evaluation across 4 arms:
- arm1_deeplab_cleankd: DeepLabV3+ R101 + Clean bn7 Soft-KD + Dice + MIL + PolySGD + EMA
- arm2_segformer_oeem: SegFormer-B3 Dual-Head + KL-Soft OEEM + Dice + MIL + AdamW + EMA
- arm3_convnext_selftrain: ConvNeXt-Base FPN + Soft-SCE + Round-2 EMA Self-Training
- arm4_dinov2_adapter: DINOv2-ViT-B/14 Spatial-Prior Adapter + KL-Soft OEEM + EMA

All hyphens strictly use standard ASCII 0x2D ('-').
"""

from src.stage2_v2.config import (
    STAGE2_V2_PRESETS,
    Stage2V2Config,
    get_stage2_v2_preset,
    resolve_stage2_v2_config_from_env,
)
from src.stage2_v2.ensemble import (
    DEFAULT_ARM_NAMES,
    ENSEMBLE_CGM_GRID,
    TTA_CANDIDATES,
    apply_tta_probs,
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
    save_ensemble_comparison_plot,
)
from src.stage2_v2.evaluator import (
    calibrate_cgm_thresholds_val_only,
    evaluate_stage2_v2_model,
    run_step0_calibration_and_author_anchor,
)
from src.stage2_v2.losses import (
    KLSoftOEEMLoss,
    MILPatchPresenceLoss,
    SoftCrossEntropyLoss,
    SoftMulticlassDiceLoss,
    SoftSymmetricCELoss,
    Stage2V2CompositeLoss,
)
from src.stage2_v2.memmap_dataset import (
    BCSSSoftMemmapDataset,
    generate_bn7_soft_memmap,
    generate_round2_ema_memmap,
)
from src.stage2_v2.models import (
    ConvNeXtFPNDualHead,
    DeepLabV3PlusDualHead,
    DINOv2AdapterDualHead,
    SegFormerDualHead,
    Stage2ModelEMA,
    build_stage2_v2_model,
    load_stage2_v2_checkpoint,
)
from src.stage2_v2.trainer import train_stage2_v2

__all__ = [
    "STAGE2_V2_PRESETS",
    "Stage2V2Config",
    "get_stage2_v2_preset",
    "resolve_stage2_v2_config_from_env",
    "DeepLabV3PlusDualHead",
    "SegFormerDualHead",
    "ConvNeXtFPNDualHead",
    "DINOv2AdapterDualHead",
    "Stage2ModelEMA",
    "build_stage2_v2_model",
    "load_stage2_v2_checkpoint",
    "SoftCrossEntropyLoss",
    "KLSoftOEEMLoss",
    "SoftSymmetricCELoss",
    "SoftMulticlassDiceLoss",
    "MILPatchPresenceLoss",
    "Stage2V2CompositeLoss",
    "BCSSSoftMemmapDataset",
    "generate_bn7_soft_memmap",
    "generate_round2_ema_memmap",
    "evaluate_stage2_v2_model",
    "calibrate_cgm_thresholds_val_only",
    "run_step0_calibration_and_author_anchor",
    "train_stage2_v2",
    "DEFAULT_ARM_NAMES",
    "ENSEMBLE_CGM_GRID",
    "TTA_CANDIDATES",
    "apply_tta_probs",
    "bootstrap_pooled_miou_ci",
    "build_patient_folds",
    "calibrate_ensemble_cgm_cv",
    "compute_macro_miou_from_iu",
    "encode_centered_logprobs_fp16",
    "evaluate_weights_on_disagreement",
    "fit_temperature_beta_nll",
    "generate_simplex_grid",
    "holm_bonferroni",
    "paired_patient_swap_permutation_test",
    "paired_wilcoxon_and_dz",
    "reconstruct_probs_from_centered",
    "run_stage2_v2_ensemble_eval",
    "run_val_cv_and_lopocv",
    "save_ensemble_comparison_plot",
]
