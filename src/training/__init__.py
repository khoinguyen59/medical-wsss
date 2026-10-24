"""
Training Service Package for SE374.R11 Weakly-Supervised Semantic Segmentation.

Provides decoupled, high-performance training engines and data loaders:
- dataset_loader: High-throughput PyTorch datasets with zero patient-level leakage.
- train_stage1: ResNet-38d classifier training with PDA and Asymmetric Loss.
- generate_pseudo_masks: Multi-layer CAM extraction, GPU affinity matting, and Otsu gating.
- train_stage2: DeepLabV3+ segmentation training with MLPS loss, CGM, and uncertainty regularization.
"""

from src.training.dataset_loader import (
    BCSS_PALETTE,
    CLASS_NAMES,
    FOREGROUND_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    NUM_FOREGROUND,
    BCSSEvalDataset,
    BCSSStage1Dataset,
    BCSSStage2MultiMaskDataset,
    create_eval_dataloader,
    create_stage1_dataloaders,
    create_stage2_dataloader,
    extract_patient_id,
    find_image_files,
    load_bcss_patient_splits,
    parse_patch_filename,
)
from src.training.generate_pseudo_masks import (
    MultiLayerCAMExtractor,
    assemble_pseudo_mask,
    bilateral_affinity_propagation_gpu,
    generate_pseudo_masks_pipeline,
    save_palette_mask,
)
from src.training.train_stage1 import (
    AsymmetricLoss,
    AttentionModule,
    Stage1Trainer,
    build_optimizer_and_scheduler,
    train_stage1_pipeline,
)
from src.training.train_stage2 import (
    Stage2Trainer,
    build_stage2_optimizer_and_scheduler,
    train_stage2_pipeline,
)
from src.training.train_proposed_sam import (
    CompositeWSSSLoss,
    SyntheticHistologyDataset,
    evaluate as evaluate_proposed_sam,
    train_one_epoch as train_proposed_sam_epoch,
)

__all__ = [
    "BCSS_PALETTE",
    "CLASS_NAMES",
    "FOREGROUND_CLASSES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NUM_CLASSES",
    "NUM_FOREGROUND",
    "BCSSEvalDataset",
    "BCSSStage1Dataset",
    "BCSSStage2MultiMaskDataset",
    "create_eval_dataloader",
    "create_stage1_dataloaders",
    "create_stage2_dataloader",
    "extract_patient_id",
    "find_image_files",
    "load_bcss_patient_splits",
    "parse_patch_filename",
    "AsymmetricLoss",
    "AttentionModule",
    "Stage1Trainer",
    "build_optimizer_and_scheduler",
    "train_stage1_pipeline",
    "MultiLayerCAMExtractor",
    "assemble_pseudo_mask",
    "bilateral_affinity_propagation_gpu",
    "generate_pseudo_masks_pipeline",
    "save_palette_mask",
    "Stage2Trainer",
    "build_stage2_optimizer_and_scheduler",
    "train_stage2_pipeline",
    "CompositeWSSSLoss",
    "SyntheticHistologyDataset",
    "train_proposed_sam_epoch",
    "evaluate_proposed_sam",
]
