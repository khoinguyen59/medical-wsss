"""
Comprehensive Unit Test Suite for SE374.R11 Training Service.

Tests all components of src/training/:
1. dataset_loader.py: Patient ID extraction, filename label parsing, patient anti-leakage,
   BCSSStage1Dataset, BCSSStage2MultiMaskDataset, BCSSEvalDataset.
2. train_stage1.py: PDA AttentionModule, AsymmetricLoss (ASL), 1-step dry run, checkpointing.
3. generate_pseudo_masks.py: MultiLayerCAMExtractor, GPU Bilateral Affinity Matting, Otsu gating,
   palette mask generation.
4. train_stage2.py: DeepLabV3PlusWithCGM training loop, MLPS loss, uncertainty regularization,
   1-step dry run, checkpointing.
5. Invariant 12 audit: Zero non-ASCII dashes across all training files.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from src.training.dataset_loader import (
    BCSS_PALETTE,
    CLASS_NAMES,
    FOREGROUND_CLASSES,
    NUM_CLASSES,
    NUM_FOREGROUND,
    BCSSEvalDataset,
    BCSSStage1Dataset,
    BCSSStage2MultiMaskDataset,
    create_eval_dataloader,
    create_stage1_dataloaders,
    create_stage2_dataloader,
    extract_patient_id,
    load_bcss_patient_splits,
    parse_patch_filename,
)
from src.training.generate_pseudo_masks import (
    assemble_pseudo_mask,
    bilateral_affinity_propagation_gpu,
    generate_pseudo_masks_pipeline,
    save_palette_mask,
)
from src.training.train_stage1 import (
    AsymmetricLoss,
    AttentionModule,
    train_stage1_pipeline,
)
from src.training.train_stage2 import (
    train_stage2_pipeline,
)
from src.training.train_proposed_sam import (
    CompositeWSSSLoss,
)


@pytest.fixture
def mock_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary BCSS-WSSS dataset hierarchy with mock patches and masks."""
    root = tmp_path / "mock_bcss"
    train_dir = root / "train"
    val_img_dir = root / "val" / "img"
    val_mask_dir = root / "val" / "mask"

    for d in (train_dir, val_img_dir, val_mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 4 train patches from 2 distinct patients (Patient-level clean)
    train_filenames = [
        "TCGA-A1-A0SK_1024_2048_[1 0 0 0].png",
        "TCGA-A1-A0SK_1024_2272_[1 1 0 0].png",
        "TCGA-A2-A04N_512_1024_[0 1 0 1].png",
        "TCGA-A2-A04N_512_1248_[0 0 1 0].png",
    ]

    for fname in train_filenames:
        # Create 224x224 RGB image with synthetic tissue-like pattern
        arr = np.random.randint(100, 220, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(train_dir / fname)

    # 2 val patches from a 3rd distinct patient
    val_filenames = [
        "TCGA-AN-A03X_1024_512_[1 0 0 0].png",
        "TCGA-AN-A03X_1024_736_[0 1 0 0].png",
    ]

    for fname in val_filenames:
        arr = np.random.randint(120, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(val_img_dir / fname)

        # Create corresponding ground truth mask (values in 0..4)
        mask_arr = np.random.randint(0, 5, (224, 224), dtype=np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="P")
        mask_img.putpalette(BCSS_PALETTE)
        mask_img.save(val_mask_dir / fname)

    return root


def test_extract_patient_id():
    """Verify patient identifier extraction across multiple naming patterns."""
    assert extract_patient_id("TCGA-A2-A0T2_1024_2048_[1 0 0 1].png") == "TCGA-A2-A0T2"
    assert extract_patient_id("TCGA-A1-A0SK_512_1024_[1000].png") == "TCGA-A1-A0SK"
    assert extract_patient_id("tcga-an-a03x_100_200_[1 1 1 1].png") == "TCGA-AN-A03X"
    assert extract_patient_id("GDPH_1234_patch_01.png") == "GDPH"


def test_parse_patch_filename():
    """Verify coordinates and multi-hot label parsing from filename."""
    pid, x, y, label = parse_patch_filename("TCGA-A2-A0T2_1024_2048_[1 0 0 1].png")
    assert pid == "TCGA-A2-A0T2"
    assert x == 1024
    assert y == 2048
    assert np.array_equal(label, np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))

    # Test contiguous digits without spaces
    pid2, x2, y2, label2 = parse_patch_filename("TCGA-A1-A0SK_512_1024_[0110].png")
    assert pid2 == "TCGA-A1-A0SK"
    assert np.array_equal(label2, np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))


def test_patient_splits_zero_leakage():
    """Verify zero patient-level leakage across train, val, and test splits (Invariant 08)."""
    splits = load_bcss_patient_splits()
    train_set = set(splits.get("train", []))
    val_set = set(splits.get("val", []))
    test_set = set(splits.get("test", []))

    assert len(train_set) > 0, "Train patient split cannot be empty"
    assert len(val_set) > 0, "Val patient split cannot be empty"
    assert len(test_set) > 0, "Test patient split cannot be empty"

    # Enforce strictly empty intersections
    assert train_set.isdisjoint(val_set), f"Train-Val patient leakage detected: {train_set & val_set}"
    assert train_set.isdisjoint(test_set), f"Train-Test patient leakage detected: {train_set & test_set}"
    assert val_set.isdisjoint(test_set), f"Val-Test patient leakage detected: {val_set & test_set}"


def test_bcss_stage1_dataset(mock_dataset_dir: Path):
    """Verify BCSSStage1Dataset item shapes and multi-label tensor values."""
    train_dir = mock_dataset_dir / "train"
    dataset = BCSSStage1Dataset(root_dir=train_dir, is_train=True)
    assert len(dataset) == 4

    image, label, name = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32

    assert isinstance(label, torch.Tensor)
    assert label.shape == (4,)
    assert label.dtype == torch.float32

    assert isinstance(name, str)
    assert len(name) > 0


def test_bcss_stage2_multi_mask_dataset(mock_dataset_dir: Path):
    """Verify synchronized loading of RGB patch and 3 multi-layer pseudo-masks."""
    train_dir = mock_dataset_dir / "train"
    pm_root = mock_dataset_dir / "train_PM"

    # Synthesize pseudo-masks for the 4 patches
    for layer in ("PM_bn7", "PM_b5_2", "PM_b4_5"):
        layer_dir = pm_root / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        for p in train_dir.glob("*.png"):
            mask_arr = np.random.randint(0, 5, (224, 224), dtype=np.uint8)
            img = Image.fromarray(mask_arr, mode="P")
            img.putpalette(BCSS_PALETTE)
            img.save(layer_dir / p.name)

    dataset = BCSSStage2MultiMaskDataset(
        img_dir=train_dir,
        pm_root_dir=pm_root,
        is_train=True,
    )
    assert len(dataset) == 4

    sample = dataset[0]
    assert "image" in sample
    assert "label" in sample
    assert "label_a" in sample
    assert "label_b" in sample
    assert "cls_label" in sample

    assert sample["image"].shape == (3, 224, 224)
    assert sample["label"].shape == (224, 224)
    assert sample["label_a"].shape == (224, 224)
    assert sample["label_b"].shape == (224, 224)
    assert sample["cls_label"].shape == (4,)


def test_bcss_eval_dataset(mock_dataset_dir: Path):
    """Verify BCSSEvalDataset loading of images and ground truth masks."""
    val_img_dir = mock_dataset_dir / "val" / "img"
    val_mask_dir = mock_dataset_dir / "val" / "mask"

    dataset = BCSSEvalDataset(img_dir=val_img_dir, mask_dir=val_mask_dir)
    assert len(dataset) == 2

    img, mask, cls_label, name = dataset[0]
    assert img.shape == (3, 224, 224)
    assert mask.shape == (224, 224)
    assert mask.dtype == torch.int64
    assert cls_label.shape == (4,)


def test_attention_module_pda():
    """Verify Progressive Dropout Attention (PDA) feature modulation and thresholding."""
    module = AttentionModule()
    x = torch.randn(2, 4096, 28, 28)
    fc_weights = torch.randn(4, 4096, 1, 1)
    gamma = 0.80

    out = module(x, fc_weights, gamma=gamma)
    assert out.shape == x.shape
    # Attention module must zero out or suppress high-activation regions
    assert not torch.allclose(out, x)


def test_asymmetric_loss():
    """Verify Asymmetric Loss (ASL) mathematical properties and gradient flow."""
    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    logits = torch.randn(4, 4, requires_grad=True)
    targets = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_bilateral_affinity_propagation_gpu():
    """Verify pure-PyTorch GPU/CPU bilateral affinity propagation on CAMs."""
    cams = torch.rand(2, 4, 32, 32)
    images = torch.rand(2, 3, 32, 32)

    refined = bilateral_affinity_propagation_gpu(
        cams=cams,
        images=images,
        radius=2,
        iterations=1,
    )
    assert refined.shape == cams.shape
    assert torch.isfinite(refined).all()


def test_assemble_pseudo_mask():
    """Verify assembly of 4-class CAMs into 5-class discrete mask with Otsu gating."""
    cams = torch.rand(2, 4, 64, 64)
    # 1st image tissue-like, 2nd image bright white glass
    imgs_np = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    imgs_np[0] = 120  # Tissue
    imgs_np[1] = 250  # Glass

    masks = assemble_pseudo_mask(cams, imgs_np, background_threshold=0.20)
    assert masks.shape == (2, 64, 64)
    # Second image (glass) must be completely assigned to background class (4)
    assert np.all(masks[1] == 4)


def test_train_stage1_dry_run(mock_dataset_dir: Path, tmp_path: Path):
    """Execute 1-step dry run of Stage-1 classifier training engine."""
    ckpt_dir = tmp_path / "checkpoints_s1"
    results = train_stage1_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_dir=ckpt_dir,
        epochs=1,
        batch_size=2,
        lr=1e-4,
        dry_run=True,
        device_str="cpu",
    )
    assert "best_macro_f1" in results
    assert "checkpoint_path" in results
    saved_path = Path(results["checkpoint_path"])
    assert saved_path.exists(), f"Checkpoint was not created at {saved_path}"

    # Verify checkpoint contents
    ckpt = torch.load(saved_path, map_location="cpu")
    assert "state_dict" in ckpt
    assert "macro_f1" in ckpt


def test_generate_pseudo_masks_dry_run(mock_dataset_dir: Path, tmp_path: Path):
    """Execute dry-run pseudo-mask generation and verify exported paletted PNGs."""
    out_dir = tmp_path / "generated_pm"
    res = generate_pseudo_masks_pipeline(
        data_dir=mock_dataset_dir,
        output_dir=out_dir,
        batch_size=2,
        dry_run=True,
        device_str="cpu",
    )
    assert res["processed_count"] > 0
    assert (out_dir / "PM_bn7").exists()
    assert (out_dir / "PM_b5_2").exists()
    assert (out_dir / "PM_b4_5").exists()

    # Check that output masks are valid 8-bit PNG images
    mask_files = list((out_dir / "PM_bn7").glob("*.png"))
    assert len(mask_files) > 0
    test_img = Image.open(mask_files[0])
    assert test_img.size == (224, 224)
    arr = np.array(test_img)
    assert np.all((arr >= 0) & (arr <= 4))


def test_train_stage2_dry_run(mock_dataset_dir: Path, tmp_path: Path):
    """Execute 1-step dry run of Stage-2 DeepLabV3+ segmentation training engine."""
    # First generate pseudo-masks in mock directory
    pm_dir = mock_dataset_dir / "train_PM"
    generate_pseudo_masks_pipeline(
        data_dir=mock_dataset_dir,
        output_dir=pm_dir,
        batch_size=2,
        dry_run=True,
        device_str="cpu",
    )

    ckpt_dir = tmp_path / "checkpoints_s2"
    results = train_stage2_pipeline(
        data_dir=mock_dataset_dir,
        pm_dir=pm_dir,
        stage1_checkpoint=None,
        checkpoint_dir=ckpt_dir,
        epochs=1,
        batch_size=2,
        lr=1e-4,
        dry_run=True,
        device_str="cpu",
    )
    assert "best_mIoU" in results
    assert "checkpoint_path" in results
    saved_path = Path(results["checkpoint_path"])
    assert saved_path.exists(), f"Stage 2 checkpoint was not created at {saved_path}"


def test_governance_invariants_training_files():
    """Verify Invariant 12: pure ASCII hyphen '-' (0x2D) only across all files."""
    forbidden = ["\u2013", "\u2014", "\u2012", "\u2015", "\u2212"]
    root = Path(__file__).resolve().parent.parent

    target_files = [
        root / "src" / "training" / "__init__.py",
        root / "src" / "training" / "dataset_loader.py",
        root / "src" / "training" / "train_stage1.py",
        root / "src" / "training" / "generate_pseudo_masks.py",
        root / "src" / "training" / "train_stage2.py",
        root / "src" / "training" / "train_proposed_sam.py",
        root / "tests" / "test_training_service.py",
    ]

    for fpath in target_files:
        assert fpath.exists(), f"File {fpath} must exist"
        content = fpath.read_text(encoding="utf-8")
        for bad_char in forbidden:
            assert bad_char not in content, (
                f"Invariant 12 violation: Found forbidden char {repr(bad_char)} in {fpath}"
            )


def test_d4_dihedral_augmentation_synchronization(mock_dataset_dir: Path):
    """Verify synchronized D4 group transformations across images and all 3 pseudo-masks."""
    train_dir = mock_dataset_dir / "train"
    pm_root = mock_dataset_dir / "train_PM"

    # Synthesize masks with known asymmetrical corner markers
    for layer in ("PM_bn7", "PM_b5_2", "PM_b4_5"):
        layer_dir = pm_root / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        for p in train_dir.glob("*.png"):
            mask_arr = np.zeros((224, 224), dtype=np.uint8)
            # Mark top-left quadrant with class 1, top-right with class 2
            mask_arr[:50, :50] = 1
            mask_arr[:50, 150:] = 2
            img = Image.fromarray(mask_arr, mode="P")
            img.putpalette(BCSS_PALETTE)
            img.save(layer_dir / p.name)

    dataset = BCSSStage2MultiMaskDataset(
        img_dir=train_dir,
        pm_root_dir=pm_root,
        is_train=True,
    )

    # Sample multiple items to ensure random D4 transforms fire and all masks match
    for _ in range(8):
        sample = dataset[0]
        lbl = sample["label"].numpy()
        lbl_a = sample["label_a"].numpy()
        lbl_b = sample["label_b"].numpy()

        # All 3 pseudo-masks must have identical spatial orientation
        assert np.array_equal(lbl, lbl_a), "PM_bn7 and PM_b5_2 spatial mismatch under D4"
        assert np.array_equal(lbl, lbl_b), "PM_bn7 and PM_b4_5 spatial mismatch under D4"
        assert sample["image"].shape == (3, 224, 224)


def test_composite_wsss_loss_backward():
    """Verify CompositeWSSSLoss forward computation and backward gradient propagation."""
    criterion = CompositeWSSSLoss(
        num_classes=5,
        focal_weight=1.0,
        dice_weight=1.0,
        iou_weight=0.5,
        head_weight=0.2,
    )

    logits = torch.randn(2, 5, 32, 32, requires_grad=True)
    masks = torch.randint(0, 5, (2, 32, 32), dtype=torch.long)
    masks[0, :4, :] = 255  # Test ignore index handling
    iou_preds = torch.rand(2, 5, requires_grad=True)

    loss_dict = criterion(
        pred_logits=logits,
        target_labels=masks,
        iou_predictions=iou_preds,
    )

    assert "total_loss" in loss_dict
    assert "focal_loss" in loss_dict
    assert "dice_loss" in loss_dict
    assert "iou_loss" in loss_dict
    assert "head_loss" in loss_dict

    total = loss_dict["total_loss"]
    assert total.dim() == 0
    assert torch.isfinite(total)
    assert total.item() > 0.0

    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert iou_preds.grad is not None
    assert torch.isfinite(iou_preds.grad).all()

