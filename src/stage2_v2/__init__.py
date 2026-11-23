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
    VALID_MEMMAP_SOURCES,
    VALID_SOFT_MODES,
    VALID_TRACKS,
    Stage2V2Config,
    get_stage2_v2_preset,
    resolve_stage2_v2_config_from_env,
    validate_track_name,
)
from src.stage2_v2.ensemble import (
    DEFAULT_ARM_FAMILIES,
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
    validate_track_a_arm_spec,
)
from src.stage2_v2.evaluator import (
    calibrate_cgm_thresholds_val_only,
    evaluate_r2_pseudo_mask_val_diagnostic,
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
    generate_cross_family_r2_memmap,
    generate_round2_ema_memmap,
    validate_track_a_teacher_checkpoint,
)
from src.stage2_v2.models import (
    BaseStage2DualHead,
    ConvNeXtFPNDualHead,
    DeepLabV3PlusDualHead,
    DINOv2AdapterDualHead,
    SegFormerDualHead,
    Stage2ModelEMA,
    build_stage2_v2_model,
    load_stage2_v2_checkpoint,
    remap_hf_state_dict_keys,
)
from src.stage2_v2.trainer import train_stage2_v2
from src.stage2_v2.visual_audit import run_stage2_v2_visual_audit

__all__ = [
    "STAGE2_V2_PRESETS",
    "VALID_MEMMAP_SOURCES",
    "VALID_SOFT_MODES",
    "VALID_TRACKS",
    "Stage2V2Config",
    "get_stage2_v2_preset",
    "resolve_stage2_v2_config_from_env",
    "validate_track_name",
    "BaseStage2DualHead",
    "DeepLabV3PlusDualHead",
    "SegFormerDualHead",
    "ConvNeXtFPNDualHead",
    "DINOv2AdapterDualHead",
    "Stage2ModelEMA",
    "build_stage2_v2_model",
    "load_stage2_v2_checkpoint",
    "remap_hf_state_dict_keys",
    "run_stage2_v2_visual_audit",
    "SoftCrossEntropyLoss",
    "KLSoftOEEMLoss",
    "SoftSymmetricCELoss",
    "SoftMulticlassDiceLoss",
    "MILPatchPresenceLoss",
    "Stage2V2CompositeLoss",
    "BCSSSoftMemmapDataset",
    "generate_bn7_soft_memmap",
    "generate_cross_family_r2_memmap",
    "generate_round2_ema_memmap",
    "validate_track_a_teacher_checkpoint",
    "evaluate_stage2_v2_model",
    "evaluate_r2_pseudo_mask_val_diagnostic",
    "calibrate_cgm_thresholds_val_only",
    "run_step0_calibration_and_author_anchor",
    "train_stage2_v2",
    "DEFAULT_ARM_FAMILIES",
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
    "validate_track_a_arm_spec",
]
