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
import pytest
import torch
import torch.nn as nn
from PIL import Image

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
from src.training.train_proposed_sam import (
    CompositeWSSSLoss,
)
from src.training.train_stage1 import (
    AsymmetricLoss,
    AttentionModule,
    train_stage1_pipeline,
)
from src.training.train_stage2 import (
    infer_stage1_norm_mode,
    train_stage2_pipeline,
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

    # Verify compact inference-only best checkpoint contents and zero_one paper-parity default
    ckpt = torch.load(saved_path, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt
    assert "macro_f1" in ckpt
    assert "optimizer" not in ckpt, "stage1_retrained_best.pth must be compact (no optimizer bloat)"
    assert "scheduler" not in ckpt, "stage1_retrained_best.pth must be compact (no scheduler bloat)"
    assert ckpt.get("input_norm") == "zero_one"
    assert infer_stage1_norm_mode(ckpt, saved_path, "auto") == "zero_one"


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
    ckpt_s2 = torch.load(saved_path, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt_s2
    assert "optimizer" not in ckpt_s2, "stage2_retrained_best.pth must be compact (no optimizer bloat)"
    assert "scheduler" not in ckpt_s2, "stage2_retrained_best.pth must be compact (no scheduler bloat)"


def test_governance_invariants_training_files():
    """Verify Invariant 12: pure ASCII hyphen '-' (0x2D) only across all files."""
    forbidden = ["\u2013", "\u2014", "\u2012", "\u2015", "\u2212"]
    root = Path(__file__).resolve().parent.parent

    target_files = [
        root / "src" / "config" / "__init__.py",
        root / "src" / "config" / "stage1_experiments.py",
        root / "src" / "training" / "__init__.py",
        root / "src" / "training" / "dataset_loader.py",
        root / "src" / "training" / "train_stage1.py",
        root / "src" / "training" / "eval_stage1_full.py",
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


def test_one_batch_overfit_sanity_loss_strictly_decreases():
    """Overfit Sanity Check: 4 steps of gradient descent on a fixed real batch from data/sample_dataset
    must strictly decrease the training loss for both Stage-1 and MedSAM Adapter.
    """
    from src.models.sam_wsss_adapter import MedSAM_WSSS_Adapter
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.dataset_loader import BCSSStage1Dataset

    torch.manual_seed(42)
    sample_train_dir = Path("data/sample_dataset/train")
    ds = BCSSStage1Dataset(root_dir=sample_train_dir, is_train=False, max_samples=2)
    img0, lbl0, _ = ds[0]
    img1, lbl1, _ = ds[1]
    batch_imgs = torch.stack([img0, img1], dim=0)
    batch_lbls = torch.stack([lbl0, lbl1], dim=0)

    # 1. Stage-1 ResNet38d Net_CAM 1-batch overfit check
    s1_model = WSSSTissueStage1NetCAM(num_classes=4)
    s1_model.train()
    s1_opt = torch.optim.AdamW(s1_model.parameters(), lr=1e-4)
    s1_crit = AsymmetricLoss(gamma_neg=2.0, clip=0.05)

    crop_imgs = batch_imgs[:, :, :64, :64]
    s1_losses = []
    for _ in range(4):
        s1_opt.zero_grad()
        logits = s1_model(crop_imgs)
        loss = s1_crit(logits, batch_lbls)
        loss.backward()
        s1_opt.step()
        s1_losses.append(loss.item())

    assert s1_losses[-1] < s1_losses[0], (
        f"Stage-1 overfit sanity failed: initial loss={s1_losses[0]:.4f}, final loss={s1_losses[-1]:.4f}"
    )

    # 2. Proposed MedSAM_WSSS_Adapter 1-batch overfit check on 64x64 crop
    sam_model = MedSAM_WSSS_Adapter(
        num_classes=5,
        freeze_image_encoder=True,
        use_surrogate_if_missing=True,
    )
    sam_model.train()
    trainable_params = [p for p in sam_model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0
    sam_opt = torch.optim.AdamW(trainable_params, lr=5e-4)
    sam_crit = CompositeWSSSLoss(num_classes=5)

    target_masks = torch.zeros((2, 64, 64), dtype=torch.long)
    target_masks[:, 32:, :] = 1

    sam_losses = []
    for _ in range(4):
        sam_opt.zero_grad()
        out = sam_model(crop_imgs)
        loss_dict = sam_crit(out["logits"], target_masks, out["iou_predictions"])
        loss_dict["total_loss"].backward()
        sam_opt.step()
        sam_losses.append(loss_dict["total_loss"].item())

    assert sam_losses[-1] < sam_losses[0], (
        f"MedSAM Adapter overfit sanity failed: initial={sam_losses[0]:.4f}, final={sam_losses[-1]:.4f}"
    )


def test_generate_pseudo_masks_stage1_norm_routing(
    mock_dataset_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify generate_pseudo_masks_pipeline routes [0,1] vs ImageNet tensors based on Stage-1 checkpoint metadata."""
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM

    s1_dummy = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
    raw_ckpt = tmp_path / "official_stage1_raw.pth"
    torch.save(s1_dummy.state_dict(), raw_ckpt)

    retrained_ckpt = tmp_path / "stage1_retrained_best.pth"
    torch.save(
        {
            "epoch": 0,
            "macro_f1": 0.5,
            "input_norm": "imagenet",
            "state_dict": s1_dummy.state_dict(),
        },
        retrained_ckpt,
    )

    captured_batches: List[torch.Tensor] = []
    orig_forward_backbone = WSSSTissueStage1NetCAM.forward_backbone

    def _spy_forward_backbone(self_mod, x: torch.Tensor) -> torch.Tensor:
        captured_batches.append(x.detach().cpu().clone())
        return orig_forward_backbone(self_mod, x)

    monkeypatch.setattr(WSSSTissueStage1NetCAM, "forward_backbone", _spy_forward_backbone)

    # 1. Official raw state_dict checkpoint -> unnormalized [0, 1] tensors
    res_raw = generate_pseudo_masks_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_path=raw_ckpt,
        output_dir=tmp_path / "pm_raw",
        batch_size=2,
        dry_run=True,
        device_str="cpu",
        stage1_norm_mode="auto",
    )
    assert res_raw["stage1_norm_mode"] == "zero_one"
    assert len(captured_batches) > 0
    for batch_t in captured_batches:
        assert float(batch_t.min()) >= 0.0
        assert float(batch_t.max()) <= 1.0

    captured_batches.clear()

    # 2. Retrained Stage-1 checkpoint ('input_norm': 'imagenet') -> ImageNet-normalized tensors
    res_imgnet = generate_pseudo_masks_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_path=retrained_ckpt,
        output_dir=tmp_path / "pm_imgnet",
        batch_size=2,
        dry_run=True,
        device_str="cpu",
        stage1_norm_mode="auto",
    )
    assert res_imgnet["stage1_norm_mode"] == "imagenet"
    assert len(captured_batches) > 0
    assert any(float(batch_t.min()) < 0.0 for batch_t in captured_batches)

    # 3. Invalid stage1_norm_mode raises ValueError
    with pytest.raises(ValueError, match="Unsupported stage1_norm_mode"):
        generate_pseudo_masks_pipeline(
            data_dir=mock_dataset_dir,
            checkpoint_path=raw_ckpt,
            output_dir=tmp_path / "pm_bad",
            batch_size=2,
            dry_run=True,
            device_str="cpu",
            stage1_norm_mode="invalid",
        )


def test_stage1_val_mask_label_derivation_and_zero_one_norm(tmp_path: Path):
    """Verify BCSSStage1Dataset & BCSSEvalDataset derive true multi-hot labels from mode-P masks when filename lacks brackets."""
    val_img_dir = tmp_path / "val" / "img"
    val_mask_dir = tmp_path / "val" / "mask"
    val_img_dir.mkdir(parents=True)
    val_mask_dir.mkdir(parents=True)

    # Filename without [a b c d] brackets (matches real data/BCSS-WSSS/val/img/*.png)
    fname = "TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0.png"
    Image.fromarray(np.full((224, 224, 3), 160, dtype=np.uint8)).save(val_img_dir / fname)

    # Mask containing strictly STR (1), NEC (3), and BACK (4) - NO TUM (0) or LYM (2)
    mask_arr = np.full((224, 224), 4, dtype=np.uint8)
    mask_arr[:80, :80] = 1
    mask_arr[100:160, 100:160] = 3
    m_img = Image.fromarray(mask_arr, mode="P")
    m_img.putpalette(BCSS_PALETTE)
    m_img.save(val_mask_dir / fname)

    ds_s1 = BCSSStage1Dataset(
        root_dir=val_img_dir,
        mask_dir=val_mask_dir,
        is_train=False,
        return_mask=True,
        input_norm="zero_one",
    )
    img_t, lbl_t, mask_t, _ = ds_s1[0]
    assert torch.equal(lbl_t, torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)), (
        f"Expected mask-derived [0, 1, 0, 1], got {lbl_t.tolist()}"
    )
    assert float(img_t.min()) >= 0.0 and float(img_t.max()) <= 1.0
    assert set(torch.unique(mask_t).tolist()) == {1, 3, 4}

    ds_eval = BCSSEvalDataset(img_dir=val_img_dir, mask_dir=val_mask_dir)
    _, eval_mask, eval_cls, _ = ds_eval[0]
    assert torch.equal(eval_cls, torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32))
    assert set(torch.unique(eval_mask).tolist()) == {1, 3, 4}


def test_stage1_multilabel_validation_metrics_and_edge_cases():
    """Verify threshold-invariant mAP, Macro-AUROC, Best-F1 (resistant to threshold shift), and degenerate edge cases."""
    from src.training.train_stage1 import compute_multilabel_validation_metrics

    y_true = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=np.float32)

    # Shifted probabilities: all positives at 0.35, all negatives at 0.10 (@0.5 F1 would be 0.0)
    y_prob_shifted = np.where(y_true > 0.5, 0.35, 0.10).astype(np.float32)
    m = compute_multilabel_validation_metrics(y_true, y_prob_shifted)
    assert m["val_macro_f1"] == 0.0
    assert m["val_mAP"] == pytest.approx(1.0, abs=1e-6)
    assert m["val_auroc"] == pytest.approx(1.0, abs=1e-6)
    assert m["val_best_f1"] == pytest.approx(1.0, abs=1e-6)

    # Edge case: empty arrays and single-class-all-zero / all-one
    m_empty = compute_multilabel_validation_metrics(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
    )
    assert m_empty["val_mAP"] == 0.0 and m_empty["val_auroc"] == 0.0

    y_deg = np.array([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.float32)
    p_deg = np.array([[0.2, 0.9, 0.8, 0.1], [0.1, 0.7, 0.2, 0.3]], dtype=np.float32)
    m_deg = compute_multilabel_validation_metrics(y_deg, p_deg)
    assert np.isfinite(m_deg["val_mAP"]) and np.isfinite(m_deg["val_auroc"]) and np.isfinite(m_deg["val_best_f1"])


def test_stage1_pretrained_loader_blocks_silent_random_init_and_loads_pth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify load_pretrained_resnet38d_weights blocks silent random init and resets fc8 when loading backbone .pth."""
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.train_stage1 import load_pretrained_resnet38d_weights

    model = WSSSTissueStage1NetCAM(n_class=4)

    # 1. Explicit missing path raises FileNotFoundError when allow_random_init=False
    with pytest.raises(FileNotFoundError):
        load_pretrained_resnet38d_weights(model, pretrained_path=tmp_path / "missing.pth", allow_random_init=False)

    # 2. No local pretrained file raises RuntimeError when allow_random_init=False
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Refusing silent training from random weights"):
        load_pretrained_resnet38d_weights(model, pretrained_path=None, allow_random_init=False, cache_dir=tmp_path)

    # 3. Valid .pth backbone loads >100 layers while keeping fc8 reset
    src_model = WSSSTissueStage1NetCAM(n_class=4)
    with torch.no_grad():
        src_model.conv1a.weight.fill_(0.1234)
        src_model.fc8.weight.fill_(9.9999)
    ckpt_file = tmp_path / "resnet38d_mock_pretrained.pth"
    torch.save({"state_dict": {f"module.{k}": v for k, v in src_model.state_dict().items()}}, ckpt_file)

    orig_fc8 = model.fc8.weight.detach().clone()
    info = load_pretrained_resnet38d_weights(
        model,
        pretrained_path=ckpt_file,
        allow_random_init=False,
        reset_classifier_head=True,
    )
    assert info["loaded_layers"] >= 100
    assert torch.allclose(model.conv1a.weight, torch.full_like(model.conv1a.weight, 0.1234))
    assert torch.equal(model.fc8.weight, orig_fc8), "fc8 classifier head must stay reset on backbone transfer"

    # 4. Partial/under-loaded checkpoint raises RuntimeError unless allow_random_init=True
    tiny_ckpt = tmp_path / "tiny_bad.pth"
    torch.save({"conv1a.weight": torch.zeros(64, 3, 3, 3)}, tiny_ckpt)
    with pytest.raises(RuntimeError, match="Loaded only 1/"):
        load_pretrained_resnet38d_weights(model, pretrained_path=tiny_ckpt, allow_random_init=False)
    with pytest.warns(UserWarning, match="Loaded only 1/"):
        warn_info = load_pretrained_resnet38d_weights(model, pretrained_path=tiny_ckpt, allow_random_init=True)
    assert warn_info["loaded_layers"] == 1


def test_stage1_dry_run_soft_margin_zero_one_and_val_gated_pm_miou(mock_dataset_dir: Path, tmp_path: Path):
    """Verify train_stage1_pipeline saves input_norm='zero_one', loss_type='soft_margin', and selects best ckpt by val_gated_pm_miou."""
    # Add two validation patches whose patient ID belongs to the official val split (disjoint from train)
    splits = load_bcss_patient_splits()
    val_pid = splits["val"][0]
    val_img_dir = mock_dataset_dir / "val" / "img"
    val_mask_dir = mock_dataset_dir / "val" / "mask"
    for idx in range(2):
        fname = f"{val_pid}-DX1_xmin100_ymin{idx * 224}_MPP-0.2500+{idx}.png"
        Image.fromarray(np.random.randint(100, 210, (224, 224, 3), dtype=np.uint8)).save(val_img_dir / fname)
        mask_arr = np.full((224, 224), 4, dtype=np.uint8)
        mask_arr[:112, :] = idx  # Class 0 (TUM) on idx=0, Class 1 (STR) on idx=1
        mask_arr[112:180, :] = (idx + 2) % 4  # Class 2 (LYM) or 3 (NEC)
        m_img = Image.fromarray(mask_arr, mode="P")
        m_img.putpalette(BCSS_PALETTE)
        m_img.save(val_mask_dir / fname)

    ckpt_dir = tmp_path / "ckpt_s1_zero_one"
    res = train_stage1_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_dir=ckpt_dir,
        epochs=1,
        batch_size=2,
        lr=1.4e-4,
        loss_type="soft_margin",
        input_norm="zero_one",
        dry_run=True,
        device_str="cpu",
    )
    assert res["input_norm"] == "zero_one"
    assert res["loss_type"] == "soft_margin"
    assert res["best_val_metric"] == "val_gated_pm_miou"
    assert "val_mAP" in res["history"][0]
    assert "val_auroc" in res["history"][0]
    assert "val_best_f1" in res["history"][0]
    assert "val_gated_pm_miou" in res["history"][0]
    assert 0.0 <= res["history"][0]["val_gated_pm_miou"] <= 1.0

    ckpt = torch.load(res["checkpoint_path"], map_location="cpu", weights_only=False)
    assert ckpt["input_norm"] == "zero_one"
    assert ckpt["best_val_metric"] == "val_gated_pm_miou"
    assert infer_stage1_norm_mode(ckpt, res["checkpoint_path"], "auto") == "zero_one"


def test_stage2_imagenet_guard_and_gt4_exclusion(monkeypatch: pytest.MonkeyPatch):
    """Verify load_imagenet_pretrained_resnet101 blocks silent 0-layer fallback and Stage2Trainer.evaluate excludes gt==4."""
    from torch.utils.data import DataLoader, Dataset

    from src.models.wsss_inference_methods import DeepLabV3Plus
    from src.training.train_stage2 import Stage2Trainer, load_imagenet_pretrained_resnet101

    model = DeepLabV3Plus(num_classes=5, output_stride=16)

    # 1. Simulate offline torchvision download failure -> RuntimeError by default, warning with fallback=True
    import torchvision.models as tv_models

    def _boom(*args, **kwargs):
        raise OSError("Simulated offline environment")

    monkeypatch.setattr(tv_models, "resnet101", _boom)
    with pytest.raises(RuntimeError, match="Failed to load ImageNet ResNet-101"):
        load_imagenet_pretrained_resnet101(model, allow_random_init_fallback=False)
    with pytest.warns(UserWarning, match="ImageNet weights not loaded"):
        loaded = load_imagenet_pretrained_resnet101(model, allow_random_init_fallback=True)
    assert loaded == 0

    # 2. Verify Stage2Trainer.evaluate strictly excludes gt==4 (repos/WSSS-Tissue/3_train_stage2.py)
    gt_mask = torch.full((16, 16), 4, dtype=torch.long)
    gt_mask[0:4, :] = 0  # TUM
    gt_mask[4:8, :] = 1  # STR
    gt_mask[8:12, :] = 2  # LYM
    gt_mask[12:14, :] = 3  # NEC
    # Rows 14:16 remain class 4 (outside-ROI background)

    class _ExactOnFgButTUMOnBgModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, h, w = x.shape
            logits = torch.full((b, 5, h, w), -5.0, device=x.device)
            logits[:, 0, 0:4, :] = 5.0
            logits[:, 1, 4:8, :] = 5.0
            logits[:, 2, 8:12, :] = 5.0
            logits[:, 3, 12:14, :] = 5.0
            # Deliberately predict TUM (0) on outside-ROI background rows 14:16
            logits[:, 0, 14:16, :] = 5.0
            return logits

    class _SingleEvalDS(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            return torch.zeros((3, 16, 16)), gt_mask, torch.ones(4), "sample"

    stub = _ExactOnFgButTUMOnBgModel()
    trainer = Stage2Trainer(
        model=stub,  # type: ignore[arg-type]
        optimizer=torch.optim.SGD(stub.parameters(), lr=1e-3),
        stage1_model=None,
        device=torch.device("cpu"),
        enable_amp=False,
    )
    metrics = trainer.evaluate(DataLoader(_SingleEvalDS(), batch_size=1), use_cgm=False)
    assert metrics["mIoU"] == pytest.approx(1.0, abs=1e-6), (
        f"Outside-ROI gt==4 must be excluded from 4-class mIoU (expected 1.0, got {metrics['mIoU']})"
    )
    assert metrics["mDice"] == pytest.approx(1.0, abs=1e-6)


def test_gated_cam_zero_tie_never_leaks_absent_tumor():
    """Verify zero-CAM tie breaking never hallucinates absent Class 0 (TUM) in Stage1Trainer.evaluate or assemble_pseudo_mask."""
    from torch.utils.data import DataLoader, Dataset

    from src.training.train_stage1 import Stage1Trainer

    # 1. assemble_pseudo_mask with background_threshold=0.0 and active_labels=[0, 1, 0, 0]
    # All-zero CAMs on an all-STR tissue patch must predict Class 1 (STR), NEVER absent Class 0 (TUM)
    zero_cams = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    tissue_img = np.full((1, 32, 32, 3), 120, dtype=np.uint8)
    tissue_img[..., 0] = 180  # Non-glass H&E-like saturation
    active_lbl = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    pm = assemble_pseudo_mask(zero_cams, tissue_img, background_threshold=0.0, active_labels=active_lbl)
    assert int((pm == 0).sum()) == 0, "Absent Class 0 (TUM) leaked on zero-CAM tie in assemble_pseudo_mask"
    assert np.all(pm == 1), f"Expected all tissue pixels assigned to sole active class 1 (STR), got {np.unique(pm)}"

    # 2. Stage1Trainer.evaluate on a non-TUM patch where ReLU CAM is 0 everywhere
    class _ZeroCAMStage1Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc8 = nn.Conv2d(4, 4, kernel_size=1, bias=False)
            with torch.no_grad():
                self.fc8.weight.fill_(-1.0)  # Negative weights -> F.relu(conv2d) == 0 everywhere
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout7 = nn.Identity()

        def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones((x.size(0), 4, 4, 4), device=x.device)

    class _SingleValDS(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            img = torch.zeros((3, 16, 16))
            lbl = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)  # Only STR present
            gt = torch.ones((16, 16), dtype=torch.long)  # All STR
            return img, lbl, gt, "non_tum_patch"

    stub = _ZeroCAMStage1Stub()
    trainer = Stage1Trainer(
        model=stub,  # type: ignore[arg-type]
        optimizer=torch.optim.SGD(stub.parameters(), lr=1e-3),
        criterion=nn.MultiLabelSoftMarginLoss(),
        device=torch.device("cpu"),
        enable_amp=False,
    )
    res = trainer.evaluate(DataLoader(_SingleValDS(), batch_size=1))
    assert res["val_pm_iou_tum"] == 0.0
    assert res["val_pm_iou_str"] == pytest.approx(1.0, abs=1e-6), (
        f"Zero-CAM tie leaked absent Class 0 instead of sole active Class 1 (got val_pm_iou_str={res['val_pm_iou_str']})"
    )


def test_sg_dap_chunked_batch_matches_single_sample_exactly():
    """Verify per-sample chunked StainGuidedAffinityRefiner produces identical results to B=1 execution."""
    from src.models.sg_dap import StainGuidedAffinityRefiner

    torch.manual_seed(7)
    refiner = StainGuidedAffinityRefiner(kernel_size=5, num_iters=4).eval()
    rgb = torch.rand(3, 3, 24, 24)
    deep = torch.randn(3, 32, 6, 6)
    cam = torch.rand(3, 4, 24, 24)

    with torch.no_grad():
        batch_probs, batch_masks = refiner(rgb, deep, cam, return_ignore_mask=True)
        for i in range(3):
            single_probs, single_masks = refiner(
                rgb[i : i + 1], deep[i : i + 1], cam[i : i + 1], return_ignore_mask=True
            )
            assert torch.allclose(batch_probs[i : i + 1], single_probs, atol=1e-6)
            assert torch.equal(batch_masks[i : i + 1], single_masks)


def test_multilayer_cam_extractor_uses_gradcam_for_b5_2_and_b4_5():
    """Verify MultiLayerCAMExtractor uses authentic Grad-CAM for b5_2/b4_5 (not fc8 slicing) and clears grads."""
    from src.models.pda_generator import ProgressiveDropoutAttention
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.generate_pseudo_masks import MultiLayerCAMExtractor

    torch.manual_seed(42)
    model = WSSSTissueStage1NetCAM(n_class=4).eval()
    extractor = MultiLayerCAMExtractor(model, device=torch.device("cpu"))

    images = torch.rand(2, 3, 64, 64)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

    # 1. Works cleanly even if caller wraps in torch.no_grad() and produces valid [0, 1] gated CAMs
    with torch.no_grad():
        cams_1 = extractor.extract_cams(images, labels)

    for layer_key in ("bn7", "b5_2", "b4_5"):
        cam = cams_1[layer_key]
        assert cam.shape == (2, 4, 64, 64)
        assert torch.isfinite(cam).all()
        assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0 + 1e-5
        assert not cam.requires_grad
        # Absent classes must be strictly zero
        assert float(cam[0, 1].abs().sum()) == 0.0
        assert float(cam[0, 3].abs().sum()) == 0.0
        assert float(cam[1, 0].abs().sum()) == 0.0
        assert float(cam[1, 2].abs().sum()) == 0.0
        assert float(cam[1, 3].abs().sum()) == 0.0

    # 2. All model parameter grads must remain clean (None) and stored features detached
    assert all(p.grad is None for p in model.parameters())
    for lk in ("bn7", "b5_2", "b4_5"):
        assert not extractor.features[lk].requires_grad

    # 3. Modifying deeper b7 weights MUST change b5_2 and b4_5 Grad-CAMs (proves genuine backward gradient flow;
    # the old invalid fc8[:, :D] slicing ignored b6/b7 completely!)
    with torch.no_grad():
        model.b7.conv_branch2b2.weight.mul_(-1.0)
    cams_2 = extractor.extract_cams(images, labels)
    assert not torch.allclose(cams_1["b5_2"][0, 0], cams_2["b5_2"][0, 0], atol=1e-5), (
        "b5_2 CAM did not respond to deeper b7 weight change; Grad-CAM gradient flow broken!"
    )
    assert not torch.allclose(cams_1["b4_5"][0, 0], cams_2["b4_5"][0, 0], atol=1e-5), (
        "b4_5 CAM did not respond to deeper b7 weight change; Grad-CAM gradient flow broken!"
    )

    # 3b. Batch independence (B=2 matches per-sample B=1) & frozen-parameter compatibility
    cams_s0 = extractor.extract_cams(images[0:1], labels[0:1])
    cams_s1 = extractor.extract_cams(images[1:2], labels[1:2])
    for lk in ("bn7", "b5_2", "b4_5"):
        assert torch.allclose(cams_2[lk][0:1], cams_s0[lk], atol=1e-5)
        assert torch.allclose(cams_2[lk][1:2], cams_s1[lk], atol=1e-5)
    for p in model.parameters():
        p.requires_grad_(False)
    cams_frozen = extractor.extract_cams(images, labels)
    for lk in ("bn7", "b5_2", "b4_5"):
        assert torch.allclose(cams_2[lk], cams_frozen[lk], atol=1e-5)
    cams_all_zero = extractor.extract_cams(images, torch.zeros_like(labels))
    for lk in ("bn7", "b5_2", "b4_5"):
        assert torch.isfinite(cams_all_zero[lk]).all() and float(cams_all_zero[lk].abs().max()) == 0.0

    extractor.remove_hooks()
    assert len(extractor.features) == 0, "remove_hooks() must clear cached intermediate feature tensors"

    # 4. ProgressiveDropoutAttention must reject mismatched feature/weight channel counts
    pda = ProgressiveDropoutAttention(num_steps=2)
    with pytest.raises(ValueError, match="Channel mismatch"):
        pda(torch.randn(1, 512, 8, 8), torch.randn(4, 4096, 1, 1))


def test_sg_dap_active_labels_eliminates_absent_class_leak():
    """Verify StainGuidedAffinityRefiner with active_labels leaks 0% absent-class pixels under both ignore modes."""
    from src.models.sg_dap import StainGuidedAffinityRefiner

    torch.manual_seed(11)
    refiner = StainGuidedAffinityRefiner(kernel_size=5, num_iters=4).eval()
    rgb = torch.rand(2, 3, 24, 24).clamp(0.05, 0.95)
    deep = torch.randn(2, 32, 6, 6)

    # Sample 0: only STR (1) active; Sample 1: STR (1) and NEC (3) active
    active_labels = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)

    # Case A: All-zero CAMs (triggers uniform_prior fallback)
    zero_cam = torch.zeros(2, 4, 24, 24, dtype=torch.float32)
    # Case B: Noisy CAM with strong spurious activation on absent Class 0 (TUM) and Class 2 (LYM)
    noisy_cam = torch.zeros(2, 4, 24, 24, dtype=torch.float32)
    noisy_cam[:, 0, :, :] = 5.0
    noisy_cam[:, 2, :, :] = 3.0
    noisy_cam[0, 1, :12, :] = 0.8
    noisy_cam[1, 1, :, :12] = 0.9
    noisy_cam[1, 3, :, 12:] = 0.9

    for cam_in in (zero_cam, noisy_cam):
        for return_ign in (False, True):
            with torch.no_grad():
                probs, mask = refiner(
                    rgb, deep, cam_in, return_ignore_mask=return_ign, active_labels=active_labels
                )
            # Probability simplex conserved everywhere
            assert torch.allclose(probs.sum(dim=1), torch.ones(2, 24, 24), atol=1e-5)
            # Absent channels strictly 0.0
            assert float(probs[0, [0, 2, 3]].abs().max()) == 0.0
            assert float(probs[1, [0, 2]].abs().max()) == 0.0
            # Zero absent-class pixels in discrete mask
            valid_0 = set(torch.unique(mask[0]).tolist()) - ({255} if return_ign else set())
            valid_1 = set(torch.unique(mask[1]).tolist()) - ({255} if return_ign else set())
            assert valid_0.issubset({1}), f"Absent class leaked in sample 0 (return_ignore={return_ign}): {valid_0}"
            assert valid_1.issubset({1, 3}), f"Absent class leaked in sample 1 (return_ignore={return_ign}): {valid_1}"
            # Single-active-class sample 0 must be 100% Class 1 even under return_ignore_mask=True
            assert torch.all(mask[0] == 1)

    # Edge case: all-zero active_labels must gracefully avoid NaN/div-by-zero
    all_zero_lbl = torch.zeros(2, 4, dtype=torch.float32)
    with torch.no_grad():
        probs_z, mask_z = refiner(rgb, deep, zero_cam, return_ignore_mask=False, active_labels=all_zero_lbl)
    assert torch.isfinite(probs_z).all()
    assert torch.allclose(probs_z.sum(dim=1), torch.ones(2, 24, 24), atol=1e-5)
    assert mask_z.shape == (2, 24, 24)


def test_stage1_checkpoint_load_parity_strict_validation(mock_dataset_dir: Path, tmp_path: Path):
    """Verify strict Stage-1 checkpoint validation (prefix stripping, >=100 layers, missing/corrupt errors)."""
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.train_stage2 import load_stage1_checkpoint_into_model

    # 1. Missing checkpoint path raises FileNotFoundError in both pipelines
    missing_ckpt = tmp_path / "does_not_exist.pth"
    with pytest.raises(FileNotFoundError):
        generate_pseudo_masks_pipeline(
            data_dir=mock_dataset_dir,
            checkpoint_path=missing_ckpt,
            output_dir=tmp_path / "pm_missing",
            dry_run=True,
            device_str="cpu",
        )
    with pytest.raises(FileNotFoundError):
        generate_pseudo_masks_pipeline(
            data_dir=mock_dataset_dir,
            checkpoint_path=None,
            output_dir=tmp_path / "pm_none",
            dry_run=False,
            device_str="cpu",
        )
    with pytest.raises(FileNotFoundError):
        train_stage2_pipeline(
            data_dir=mock_dataset_dir,
            pm_dir=mock_dataset_dir / "train_PM",
            stage1_checkpoint=missing_ckpt,
            checkpoint_dir=tmp_path / "ckpt_missing",
            dry_run=True,
            device_str="cpu",
        )

    # 2. Corrupt checkpoint file raises RuntimeError by default, warns when allow_random_init=True
    corrupt_ckpt = tmp_path / "corrupt_stage1.pth"
    corrupt_ckpt.write_bytes(b"not a valid torch checkpoint payload")
    with pytest.raises(RuntimeError, match="Failed to load corrupt Stage-1 checkpoint"):
        load_stage1_checkpoint_into_model(WSSSTissueStage1NetCAM(n_class=4), corrupt_ckpt)
    with pytest.warns(UserWarning, match="Failed to load corrupt Stage-1 checkpoint"):
        norm_c, matched_c = load_stage1_checkpoint_into_model(
            WSSSTissueStage1NetCAM(n_class=4), corrupt_ckpt, allow_random_init=True
        )
    assert norm_c == "zero_one" and matched_c == 0
    with pytest.warns(UserWarning, match="Stage-1 checkpoint not found"):
        norm_m, matched_m = load_stage1_checkpoint_into_model(
            WSSSTissueStage1NetCAM(n_class=4), missing_ckpt, allow_random_init=True
        )
    assert norm_m == "zero_one" and matched_m == 0

    # 3. Under-loaded checkpoint (<100 matched layers) raises RuntimeError by default, warns when allow_random_init=True
    tiny_ckpt = tmp_path / "tiny_stage1.pth"
    torch.save({"module.conv1a.weight": torch.zeros(64, 3, 3, 3)}, tiny_ckpt)
    with pytest.raises(RuntimeError, match="Loaded only 1/"):
        load_stage1_checkpoint_into_model(WSSSTissueStage1NetCAM(n_class=4), tiny_ckpt)
    with pytest.warns(UserWarning, match="Loaded only 1/"):
        _, matched_t = load_stage1_checkpoint_into_model(
            WSSSTissueStage1NetCAM(n_class=4), tiny_ckpt, allow_random_init=True
        )
    assert matched_t == 1

    # 4. Valid module./model.-prefixed checkpoint loads >=100 layers cleanly
    src_m = WSSSTissueStage1NetCAM(n_class=4)
    with torch.no_grad():
        src_m.conv1a.weight.fill_(0.321)
    prefixed_ckpt = tmp_path / "prefixed_stage1.pth"
    torch.save(
        {"state_dict": {f"module.model.{k}": v for k, v in src_m.state_dict().items()}},
        prefixed_ckpt,
    )
    dst_m = WSSSTissueStage1NetCAM(n_class=4)
    norm_mode, matched = load_stage1_checkpoint_into_model(dst_m, prefixed_ckpt)
    assert matched >= 100
    assert norm_mode == "zero_one"
    assert torch.allclose(dst_m.conv1a.weight, torch.full_like(dst_m.conv1a.weight, 0.321))

    # 5. Author checkpoint filename defaults to zero_one under 'auto', but honors explicit 'imagenet' override
    author_named_ckpt = tmp_path / "stage1_checkpoint_trained_on_bcss.pth"
    torch.save(src_m.state_dict(), author_named_ckpt)
    norm_auto, matched_auto = load_stage1_checkpoint_into_model(dst_m, author_named_ckpt, stage1_norm_mode="auto")
    assert norm_auto == "zero_one" and matched_auto >= 100
    norm_explicit, matched_exp = load_stage1_checkpoint_into_model(
        dst_m, author_named_ckpt, stage1_norm_mode="imagenet"
    )
    assert norm_explicit == "imagenet" and matched_exp >= 100

    # 6. End-to-end generate_pseudo_masks_pipeline(..., use_sg_dap=True) leaks 0 absent-class pixels across all 3 layers
    sg_data = tmp_path / "sg_dap_ds"
    (sg_data / "train").mkdir(parents=True)
    patch_name = "TCGA-A1-A0SK_0_0_[0 1 0 0].png"  # Only STR (1) active
    tissue = np.full((64, 64, 3), 120, dtype=np.uint8)
    tissue[..., 0] = 190
    Image.fromarray(tissue).save(sg_data / "train" / patch_name)
    sg_out = tmp_path / "sg_dap_pm"
    sg_res = generate_pseudo_masks_pipeline(
        data_dir=sg_data,
        checkpoint_path=author_named_ckpt,
        output_dir=sg_out,
        batch_size=1,
        dry_run=True,
        device_str="cpu",
        use_sg_dap=True,
    )
    assert sg_res["processed_count"] == 1
    for sub in ("PM_bn7", "PM_b5_2", "PM_b4_5"):
        arr = np.array(Image.open(sg_out / sub / patch_name))
        assert set(np.unique(arr).tolist()).issubset({1, 4, 255}), (
            f"Absent class leaked in {sub}: {np.unique(arr).tolist()}"
        )


def test_stage2_dataset_resizes_non_224_masks_nearest(tmp_path: Path):
    """Verify BCSSStage2MultiMaskDataset resizes non-224x224 images and masks with NEAREST preserving discrete IDs."""
    img_dir = tmp_path / "train"
    pm_root = tmp_path / "train_PM"
    img_dir.mkdir(parents=True)
    fname = "TCGA-A1-A0SK_100_200_[1 1 0 1].png"

    # Non-224x224 rectangular patch (160x192)
    Image.fromarray(np.random.randint(80, 200, (192, 160, 3), dtype=np.uint8)).save(img_dir / fname)
    for layer in ("PM_bn7", "PM_b5_2", "PM_b4_5"):
        ldir = pm_root / layer
        ldir.mkdir(parents=True, exist_ok=True)
        m_arr = np.full((192, 160), 4, dtype=np.uint8)
        m_arr[:60, :80] = 0
        m_arr[60:120, :80] = 1
        m_arr[120:160, 80:] = 3
        m_arr[160:, :] = 255  # ignore_index
        save_palette_mask(m_arr, ldir / fname)

    ds = BCSSStage2MultiMaskDataset(img_dir=img_dir, pm_root_dir=pm_root, is_train=True)
    sample = ds[0]
    assert sample["image"].shape == (3, 224, 224)
    for key in ("label", "label_a", "label_b"):
        assert sample[key].shape == (224, 224)
        uniq = set(torch.unique(sample[key]).tolist())
        assert uniq.issubset({0, 1, 3, 4, 255}), f"Unexpected interpolated mask values in {key}: {uniq}"
        assert 255 in uniq, f"ignore_index=255 lost during NEAREST resize in {key}"


def test_stage1_bn_and_early_stages_freeze_parity():
    """Verify WSSSTissueStage1NetCAM.train() freezes all BatchNorm2d and not_training stages matching resnet38d.py."""
    from src.colab_pkg.model_loader import WSSSTissueStage1NetCAM as ColabStage1
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM as SrcStage1
    from src.training.train_stage1 import build_optimizer_and_scheduler

    for ModelCls in (SrcStage1, ColabStage1):
        model = ModelCls(n_class=4)
        model.train()

        # 1. Early stages (conv1a, b2, b2_1, b2_2) are frozen & eval
        for mod in (model.conv1a, model.b2, model.b2_1, model.b2_2):
            assert mod.training is False, f"{ModelCls.__name__} failed to set early stage {mod} to eval"
            for p in mod.parameters():
                assert p.requires_grad is False

        # 2. All BatchNorm2d layers stay in eval mode with frozen affine weights/biases
        bn_count = 0
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                bn_count += 1
                assert mod.training is False
                if mod.weight is not None:
                    assert mod.weight.requires_grad is False
                if mod.bias is not None:
                    assert mod.bias.requires_grad is False
        assert bn_count >= 30

        # 3. bn7.running_var & running_mean stay bit-exact across forward+backward steps
        bn7_var_before = model.bn7.running_var.detach().clone()
        bn7_mean_before = model.bn7.running_mean.detach().clone()
        conv1a_before = model.conv1a.weight.detach().clone()

        opt, _ = build_optimizer_and_scheduler(model, lr=1e-4, weight_decay=1e-4, epochs=5)
        # Verify fc8.weight gets non-zero weight_decay and frozen params are excluded
        opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        assert id(model.conv1a.weight) not in opt_param_ids
        assert id(model.bn7.weight) not in opt_param_ids
        assert id(model.fc8.weight) in opt_param_ids
        fc8_group = next(g for g in opt.param_groups if any(p is model.fc8.weight for p in g["params"]))
        assert fc8_group["weight_decay"] == pytest.approx(1e-4)

        x = torch.randn(2, 3, 64, 64) * 5.0 + 2.0
        y = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        for _ in range(3):
            opt.zero_grad()
            conv6 = model.forward_backbone(x)
            logits = model.fc8(model.pool(conv6)).view(2, 4)
            loss = nn.MultiLabelSoftMarginLoss()(logits, y)
            loss.backward()
            opt.step()

        assert torch.equal(model.bn7.running_var, bn7_var_before), "bn7.running_var drifted during training!"
        assert torch.equal(model.bn7.running_mean, bn7_mean_before), "bn7.running_mean drifted during training!"
        assert torch.equal(model.conv1a.weight, conv1a_before), "conv1a.weight mutated despite not_training!"


def test_attention_module_bounded_identity_and_mean_preservation():
    """Verify AttentionModule returns identity at gamma=1.0 and bounded mean-preserving features at gamma<1.0."""
    torch.manual_seed(17)
    attn = AttentionModule()
    x = torch.rand(2, 64, 14, 14) * 2.0
    # Give spatial hotspots so CAMs have clear peaks above threshold
    x[:, :, 2:5, 2:5] = 8.0
    x.requires_grad_(True)
    w = torch.randn(4, 64, 1, 1) * 0.2
    w.requires_grad_(True)

    # 1. gamma >= 1.0 -> exact identity (zero shock on activation)
    out_id = attn(x, w, gamma=1.0)
    assert torch.equal(out_id, x.float())

    # 2. gamma < 1.0 -> drops peaks, preserves expected spatial magnitude, no fc_weight grad leak
    out_drop = attn(x, w, gamma=0.80)
    assert out_drop.shape == x.shape
    assert torch.isfinite(out_drop).all()
    assert not torch.allclose(out_drop, x)
    # Bounded inverted-dropout scale means total feature mean stays close to input mean (never 2x-38x explosion)
    in_mean = float(x.detach().mean())
    out_mean = float(out_drop.detach().mean())
    assert 0.5 * in_mean <= out_mean <= 1.5 * in_mean, f"Feature mean drifted: in={in_mean:.4f}, out={out_mean:.4f}"
    assert float(out_drop.detach().max()) <= float(x.detach().max()) * 4.0 + 1e-5
    # Verify detached fc_weights receive zero gradients from AttentionModule while x receives grads
    out_drop.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert w.grad is None, "AttentionModule must detach fc_weights so CAM thresholding never leaks grads into fc8!"

    # 3. Inactive class with zero CAM must not zero out features
    w_inactive = torch.zeros(4, 64, 1, 1)
    out_inactive = attn(x, w_inactive, gamma=0.70)
    assert torch.allclose(out_inactive, x.float())


def test_stage1_multiepoch_pda_calibration_no_collapse(tmp_path: Path):
    """Verify 5-epoch Stage1Trainer (pda_start_epoch=3) keeps logits_clean and logits_pda calibrated without fc8 collapse."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.train_stage1 import Stage1Trainer, build_optimizer_and_scheduler

    torch.manual_seed(42)
    # Construct separable synthetic dataset (20 train, 12 val) with distinct color/pattern signatures per class
    n_train, n_val = 20, 12
    train_imgs = torch.rand(n_train, 3, 64, 64) * 0.2
    train_lbls = torch.zeros(n_train, 4)
    val_imgs = torch.rand(n_val, 3, 64, 64) * 0.2
    val_lbls = torch.zeros(n_val, 4)
    for i in range(n_train):
        c = i % 4
        train_lbls[i, c] = 1.0
        train_imgs[i, c % 3, 16:48, 16:48] += 0.7 if c < 3 else 0.0
        if c == 3:
            train_imgs[i, :, 16:48, 16:48] += 0.7
    for i in range(n_val):
        c = i % 4
        val_lbls[i, c] = 1.0
        val_imgs[i, c % 3, 16:48, 16:48] += 0.7 if c < 3 else 0.0
        if c == 3:
            val_imgs[i, :, 16:48, 16:48] += 0.7

    train_loader = DataLoader(
        TensorDataset(train_imgs.clamp(0, 1), train_lbls, torch.zeros(n_train)),
        batch_size=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_imgs.clamp(0, 1), val_lbls, torch.zeros(n_val)),
        batch_size=4,
        shuffle=False,
    )

    model = WSSSTissueStage1NetCAM(n_class=4)
    model.train()
    bn7_var_init = model.bn7.running_var.detach().clone()
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr=2e-4, weight_decay=1e-4, epochs=5)
    trainer = Stage1Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MultiLabelSoftMarginLoss(),
        scheduler=scheduler,
        device=torch.device("cpu"),
        enable_amp=False,
        pda_start_epoch=3,
        initial_gamma=1.0,
        gamma_decay=0.985,
    )

    val_maps = []
    for ep in range(5):
        if ep == 3:
            assert trainer.gamma < 1.0, f"Expected gamma < 1.0 at Ep 4 PDA activation, got {trainer.gamma}"
        trainer.train_epoch(train_loader, epoch=ep)
        vm = trainer.evaluate(val_loader)
        val_maps.append(vm["val_mAP"])
        assert np.isfinite(vm["val_loss"])
        assert abs(vm["val_logits_mean"]) < 10.0, f"Logits collapsed at epoch {ep+1}: {vm['val_logits_mean']}"

    # 1. bn7.running_var must remain completely frozen (0% drift)
    assert torch.equal(model.bn7.running_var, bn7_var_init)
    # 2. Clean vs PDA branch logits stay well-calibrated (no 2x-38x explosion)
    model.eval()
    with torch.no_grad():
        l_clean, _, l_pda, _ = trainer.forward_dual_branches(val_imgs[:4].clamp(0, 1), enable_pda=True)
        assert l_pda is not None
        logit_diff = float((l_clean - l_pda).abs().mean())
        assert logit_diff < 5.0, f"Clean vs PDA logit divergence too large: {logit_diff}"
    # 3. fc8.weight mean stays balanced near zero (no catastrophic negative shift)
    assert float(model.fc8.weight.detach().mean()) > -0.1
    assert val_maps[-1] >= 0.40


def test_stage1_spawn_persistent_workers_dataloader_3_epochs(mock_dataset_dir: Path):
    """Verify BCSSStage1TrainDataset & BCSSStage1ValDataset run 3 epochs cleanly with spawn + persistent_workers."""
    from torch.utils.data import DataLoader

    from src.training.dataset_loader import BCSSStage1TrainDataset, BCSSStage1ValDataset

    train_ds = BCSSStage1TrainDataset(
        mock_dataset_dir / "train",
        patient_ids=["TCGA-A1-A0SK", "TCGA-A2-A04N"],
        strict_patient_filter=True,
    )
    val_ds = BCSSStage1ValDataset(
        mock_dataset_dir / "val" / "img",
        mock_dataset_dir / "val" / "mask",
        patient_ids=["TCGA-AN-A03X"],
        strict_patient_filter=True,
    )
    # Strict non-empty patient filtering must raise ValueError on unmatched patient IDs
    with pytest.raises(ValueError, match="Patient ID filter matched 0"):
        BCSSStage1TrainDataset(
            mock_dataset_dir / "train",
            patient_ids=["TCGA-NONEXISTENT"],
            strict_patient_filter=True,
        )

    mp_kwargs = {"persistent_workers": True, "prefetch_factor": 2, "multiprocessing_context": "spawn"}
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, **mp_kwargs)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, **mp_kwargs)

    for _ in range(3):
        train_batches = list(train_loader)
        val_batches = list(val_loader)
        assert len(train_batches) == 2
        assert len(val_batches) == 1
        fnames, imgs, lbls = train_batches[0]
        assert len(fnames) == 2 and imgs.shape == (2, 3, 224, 224) and lbls.shape == (2, 4)
        vfnames, vimgs, vlbls, vmasks = val_batches[0]
        assert len(vfnames) == 2 and vimgs.shape == (2, 3, 224, 224) and vmasks.shape == (2, 224, 224)


def test_stage1_experiment_presets_and_env_resolution(monkeypatch: pytest.MonkeyPatch):
    """Verify Stage1ExperimentConfig presets (default, Arms A/B/C) and env variable resolution."""
    from src.config import STAGE1_PRESETS, get_stage1_preset, resolve_stage1_config_from_env

    assert set(STAGE1_PRESETS.keys()) == {
        "default",
        "arm_a_polysgd",
        "arm_b_equiv_adamw",
        "arm_c_mpppc_polysgd",
    }

    arm_a = get_stage1_preset("arm_a_polysgd")
    assert arm_a.optimizer_type == "polysgd"
    assert arm_a.scheduler_type == "poly"
    assert arm_a.lr_bb == pytest.approx(1.5e-3)
    assert arm_a.lr_fc8 == pytest.approx(1.5e-2)
    assert arm_a.wd_bb == pytest.approx(5e-4)
    assert arm_a.use_d4_aug is True
    assert arm_a.pda_mode == "multiplicative"
    assert arm_a.use_ema is True
    assert arm_a.ema_decay == pytest.approx(0.995)

    arm_b = get_stage1_preset("arm_b_equiv_adamw")
    assert arm_b.optimizer_type == "adamw"
    assert arm_b.lr_bb == pytest.approx(8e-5)
    assert arm_b.lr_fc8 == pytest.approx(2.4e-4)
    assert arm_b.wd_fc8 == pytest.approx(5e-2)
    assert arm_b.use_equiv_consistency is True
    assert arm_b.equiv_scales == (1.0, 1.25)
    assert 0.75 not in arm_b.equiv_scales

    arm_c = get_stage1_preset("arm_c_mpppc_polysgd")
    assert arm_c.optimizer_type == "polysgd"
    assert arm_c.use_mpppc is True
    assert arm_c.ppc_num_prototypes == 1
    assert arm_c.lambda_ppc == pytest.approx(0.10)
    assert arm_c.ppc_start_epoch == 3

    monkeypatch.setenv("STAGE1_EXP_PRESET", "arm_b_equiv_adamw")
    monkeypatch.setenv("STAGE1_MAX_EPOCHS", "12")
    monkeypatch.setenv("STAGE1_SEED", "77")
    monkeypatch.setenv("STAGE1_BATCH_SIZE", "16")
    monkeypatch.setenv("STAGE1_INPUT_NORM", "zero_one")
    monkeypatch.setenv("STAGE1_LOSS_TYPE", "soft_margin")
    monkeypatch.setenv("STAGE1_GIT_REF", "abc1234")
    monkeypatch.setenv("STAGE1_NUM_WORKERS", "4")
    resolved = resolve_stage1_config_from_env()
    assert resolved.preset_name == "arm_b_equiv_adamw"
    assert resolved.max_epochs == 12
    assert resolved.seed == 77
    assert resolved.batch_size == 16
    assert resolved.git_ref == "abc1234"
    assert resolved.num_workers == 4


def test_attention_module_label_gated_multiplicative_drops_single_class_peak_to_zero():
    """Verify Label-Gated Multiplicative Peak-Drop drops active single-class peak strictly to 0.0."""
    torch.manual_seed(5)
    attn = AttentionModule(pda_mode="multiplicative")

    # Feature map with hotspot at (5:8, 5:8) aligned with class 0 weights
    x = torch.ones(2, 16, 14, 14)
    x[:, :, 5:8, 5:8] = 10.0
    w = torch.zeros(4, 16, 1, 1)
    w[0, :, 0, 0] = 1.0  # Active class 0 has sharp peak at 5:8, 5:8
    w[1, :, 0, 0] = 0.0  # Inactive classes

    # Single-class GT label [1, 0, 0, 0]
    active_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    # Without active_labels (legacy mean mode): 3 inactive channels contribute 1.0 -> peak drops only to 0.75 / scale
    out_legacy = attn(x, w, gamma=0.70)
    assert float(out_legacy[:, :, 6, 6].min()) > 0.0

    # With active_labels (Label-Gated Multiplicative): single-class peak drops strictly to 0.0
    out_gated = attn(x, w, gamma=0.70, active_labels=active_labels)
    assert float(out_gated[:, :, 6, 6].abs().max()) == 0.0, (
        f"Expected single-class peak dropped to 0.0, got {float(out_gated[:, :, 6, 6].abs().max())}"
    )
    # Non-peak background remains strictly positive (rescaled)
    assert float(out_gated[:, :, 0, 0].min()) > 0.0


def test_random_d4_transform_picklable_and_spawn_loader(mock_dataset_dir: Path):
    """Verify RandomD4Transform is picklable and runs cleanly under spawn DataLoader."""
    import pickle

    from torch.utils.data import DataLoader

    from src.training.dataset_loader import BCSSStage1TrainDataset, RandomD4Transform

    d4 = RandomD4Transform()
    restored = pickle.loads(pickle.dumps(d4))
    sample_img = Image.new("RGB", (64, 64), color=(120, 80, 140))
    out_img = restored(sample_img)
    assert out_img.size == (64, 64)

    ds = BCSSStage1TrainDataset(
        mock_dataset_dir / "train",
        patient_ids=["TCGA-A1-A0SK", "TCGA-A2-A04N"],
        use_d4_aug=True,
    )
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context="spawn",
    )
    fnames, imgs, lbls = next(iter(loader))
    assert len(fnames) == 2 and imgs.shape == (2, 3, 224, 224) and lbls.shape == (2, 4)


def test_arm_b_equivariant_cam_consistency_zero_nan_grad_flow():
    """Verify D4 invertibility and Arm B Scale {1.0x, 1.25x} + D4 L_eq zero-NaN gradient flow."""
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.train_stage1 import Stage1Trainer, build_optimizer_and_scheduler

    torch.manual_seed(9)
    # 1. Exact D4 group invertibility check across all 16 combinations
    x_rand = torch.randn(2, 4, 16, 16)
    for k_rot in range(4):
        for hf in (False, True):
            for vf in (False, True):
                xt = Stage1Trainer._apply_d4_tensor(x_rand, k_rot, hf, vf)
                xr = Stage1Trainer._inverse_d4_tensor(xt, k_rot, hf, vf)
                assert torch.allclose(xr, x_rand, atol=1e-6)

    # 2. Equivariant branch loss & gradient flow at epoch=5 (lambda_eq = 0.25)
    model = WSSSTissueStage1NetCAM(n_class=4)
    model.train()
    opt, sched = build_optimizer_and_scheduler(
        model, lr=8e-5, weight_decay=1e-3, lr_fc8=2.4e-4, wd_fc8=5e-2, epochs=18
    )
    trainer = Stage1Trainer(
        model=model,
        optimizer=opt,
        criterion=nn.MultiLabelSoftMarginLoss(),
        scheduler=sched,
        device=torch.device("cpu"),
        enable_amp=False,
        use_equiv_consistency=True,
        equiv_scales=(1.0, 1.25),
        lambda_eq_max=0.25,
        eq_ramp_start_epoch=2,
        eq_ramp_end_epoch=5,
    )
    assert trainer.get_lambda_eq(0) == pytest.approx(0.0)
    assert trainer.get_lambda_eq(2) == pytest.approx(0.0)
    assert trainer.get_lambda_eq(5) == pytest.approx(0.25)

    imgs = torch.rand(2, 3, 64, 64)
    targets = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    opt.zero_grad()
    _, _, _, _, conv6_f = trainer.forward_dual_branches(imgs, enable_pda=True, targets=targets, return_conv6=True)
    l_cls_t, l_eq = trainer.compute_equivariant_branch(imgs, targets, conv6_f)
    total = l_cls_t + 0.25 * l_eq
    assert torch.isfinite(total)
    total.backward()
    assert model.fc8.weight.grad is not None and torch.isfinite(model.fc8.weight.grad).all()
    assert model.b7.conv_branch2b2.weight.grad is not None and torch.isfinite(model.b7.conv_branch2b2.weight.grad).all()


def test_arms_a_b_c_dry_runs_and_strict_ema_checkpoint_export(mock_dataset_dir: Path, tmp_path: Path):
    """Verify dry-run parity for Arm A, Arm B, Arm C, ModelEMA strict=True export (zero proj_head leakage), and eval_stage1_full."""
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.eval_stage1_full import evaluate_stage1_checkpoint

    # Add val patches matching official val split so GatedPM_mIoU is computed
    splits = load_bcss_patient_splits()
    val_pid = splits["val"][0]
    val_img_dir = mock_dataset_dir / "val" / "img"
    val_mask_dir = mock_dataset_dir / "val" / "mask"
    for idx in range(2):
        fname = f"{val_pid}-DX1_xmin200_ymin{idx * 224}_MPP-0.2500+{idx}.png"
        Image.fromarray(np.random.randint(90, 200, (224, 224, 3), dtype=np.uint8)).save(val_img_dir / fname)
        mask_arr = np.full((224, 224), 4, dtype=np.uint8)
        mask_arr[:112, :] = idx
        mask_arr[112:180, :] = (idx + 2) % 4
        m_img = Image.fromarray(mask_arr, mode="P")
        m_img.putpalette(BCSS_PALETTE)
        m_img.save(val_mask_dir / fname)

    for preset_name in ("arm_a_polysgd", "arm_b_equiv_adamw", "arm_c_mpppc_polysgd"):
        ckpt_dir = tmp_path / f"ckpt_{preset_name}"
        res = train_stage1_pipeline(
            data_dir=mock_dataset_dir,
            checkpoint_dir=ckpt_dir,
            preset=preset_name,
            dry_run=True,
            device_str="cpu",
        )
        assert res["exp_preset"] == preset_name
        assert res["selected_weights"] in ("raw", "ema")
        assert "fc8_std" in res and np.isfinite(res["fc8_std"])
        assert "val_gated_pm_miou_global" in res["history"][0]
        assert "ema_val_mAP" in res["history"][0]

        best_p = Path(res["checkpoint_path"])
        best_cls_p = Path(res["best_cls_checkpoint_path"])
        assert best_p.is_file() and best_cls_p.is_file()

        for p in (best_p, best_cls_p):
            payload = torch.load(p, map_location="cpu", weights_only=False)
            sd = payload["state_dict"]
            assert not any("proj_head" in k for k in sd.keys()), (
                f"proj_head leaked into exported checkpoint {p}"
            )
            net = WSSSTissueStage1NetCAM(n_class=4)
            net.load_state_dict(sd, strict=True)

    # Also test eval_stage1_full on the Arm C exported strict=True checkpoint
    arm_c_ckpt = tmp_path / "ckpt_arm_c_mpppc_polysgd" / "stage1_retrained_best.pth"
    eval_report = evaluate_stage1_checkpoint(
        checkpoint_path=arm_c_ckpt,
        data_dir=mock_dataset_dir,
        batch_size=2,
        max_samples=2,
        device_str="cpu",
        scales=(1.0, 1.25),
        run_sg_dap=True,
    )
    assert eval_report["strict_load_ok"] is True
    assert len(eval_report["sha256"]) == 64
    for mode_key in (
        "raw_1x_per_channel",
        "raw_1x_global",
        "ms_tta_per_channel",
        "ms_tta_global",
        "sg_dap_1x",
        "ms_tta_sg_dap",
    ):
        assert mode_key in eval_report["modes"]
        assert 0.0 <= eval_report["modes"][mode_key]["mIoU"] <= 1.0


def test_polysgd_scheduler_ema_bit_exact_and_arm_c_ppc_epoch_activation(
    mock_dataset_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """Verify PolySGD+PolyLR warmup/decay, ModelEMA bn7 bit-exactness, Arm C PPC at Ep 4, and early stop logs."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.config import resolve_stage1_config_from_env
    from src.losses_metrics.losses import MultiPrototypePPCLoss
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.train_stage1 import (
        ModelEMA,
        ProjectionHead,
        Stage1Trainer,
        build_optimizer_and_scheduler,
    )

    model = WSSSTissueStage1NetCAM(n_class=4)
    model.train()
    proj_head = ProjectionHead()
    ema = ModelEMA(model, decay=0.995)
    bn7_var_init = model.bn7.running_var.detach().clone()

    opt, sched = build_optimizer_and_scheduler(
        model,
        lr=1.5e-3,
        weight_decay=5e-4,
        epochs=20,
        optimizer_type="polysgd",
        scheduler_type="poly",
        lr_fc8=1.5e-2,
        wd_fc8=5e-4,
        momentum=0.9,
        warmup_steps=2,
        total_steps=10,
        poly_power=0.9,
        extra_params=list(proj_head.parameters()),
    )
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == pytest.approx(1.5e-3 * 0.5)  # step 0 of warmup=2 -> 0.5x

    ppc = MultiPrototypePPCLoss(num_classes=4, num_prototypes=1, feat_dim=128, conf_thresh=0.1)
    trainer = Stage1Trainer(
        model=model,
        optimizer=opt,
        criterion=nn.MultiLabelSoftMarginLoss(),
        scheduler=sched,
        device=torch.device("cpu"),
        enable_amp=False,
        pda_start_epoch=3,
        step_scheduler_per_batch=True,
        model_ema=ema,
        proj_head=proj_head,
        mpppc_loss=ppc,
        lambda_ppc=0.10,
        ppc_start_epoch=3,
    )

    imgs = torch.rand(2, 3, 64, 64) * 0.6 + 0.1  # non-white tissue
    lbls = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    loader = DataLoader(TensorDataset(imgs, lbls, torch.zeros(2)), batch_size=2)

    m_ep3 = trainer.train_epoch(loader, epoch=3)
    assert np.isfinite(m_ep3["loss"]) and np.isfinite(m_ep3["ppc_loss"])
    assert torch.equal(ema.shadow["bn7.running_var"], bn7_var_init)

    # swap_to_ema must restore exact raw weights on exit
    raw_fc8 = model.fc8.weight.detach().clone()
    with ema.swap_to_ema(model):
        assert torch.allclose(model.fc8.weight, ema.shadow["fc8.weight"])
    assert torch.equal(model.fc8.weight, raw_fc8)

    # Full pipeline 1-epoch non-dry-run with Arm C saves stage1_latest.pth containing ema_state_dict + proj_head
    splits = load_bcss_patient_splits()
    val_pid = splits["val"][0]
    val_img_dir = mock_dataset_dir / "val" / "img"
    val_mask_dir = mock_dataset_dir / "val" / "mask"
    fname = f"{val_pid}-DX1_xmin300_ymin300_MPP-0.2500+0.png"
    Image.fromarray(np.full((224, 224, 3), 120, dtype=np.uint8)).save(val_img_dir / fname)
    m_arr = np.zeros((224, 224), dtype=np.uint8)
    m_arr[112:, :] = 1
    m_img = Image.fromarray(m_arr, mode="P")
    m_img.putpalette(BCSS_PALETTE)
    m_img.save(val_mask_dir / fname)

    cfg = resolve_stage1_config_from_env(
        overrides={
            "preset_name": "arm_c_mpppc_polysgd",
            "max_epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "ppc_start_epoch": 0,
            "pda_start_epoch": 0,
            "min_epochs_early_stop": 1,
            "early_stop_patience": 1,
        }
    )
    ckpt_dir = tmp_path / "arm_c_full_1ep"
    res = train_stage1_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_dir=ckpt_dir,
        allow_random_init=True,
        device_str="cpu",
        experiment_config=cfg,
    )
    latest_p = Path(res["latest_checkpoint_path"])
    assert latest_p.is_file()
    latest_obj = torch.load(latest_p, map_location="cpu", weights_only=False)
    assert "ema_state_dict" in latest_obj
    assert "proj_head_state_dict" in latest_obj

    captured = capsys.readouterr().out
    assert "[EARLY REGRESSION ALARM]" not in captured
    assert "fc8_std=" in captured
    assert "fc8_norms=" in captured
    assert "Global=" in captured
    assert "[EMA]" in captured
    assert "selected=" in captured


def test_stage1_resume_device_placement_order_and_end_to_end(
    mock_dataset_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify train_stage1_pipeline moves modules to target device BEFORE optimizer build/load and resumes cleanly."""
    from src.config import resolve_stage1_config_from_env
    from src.losses_metrics.losses import MultiPrototypePPCLoss
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training import train_stage1 as ts1

    splits = load_bcss_patient_splits()
    val_pid = splits["val"][0]
    val_img_dir = mock_dataset_dir / "val" / "img"
    val_mask_dir = mock_dataset_dir / "val" / "mask"
    fname = f"{val_pid}-DX1_xmin400_ymin400_MPP-0.2500+0.png"
    Image.fromarray(np.full((224, 224, 3), 120, dtype=np.uint8)).save(val_img_dir / fname)
    m_arr = np.zeros((224, 224), dtype=np.uint8)
    m_arr[112:, :] = 1
    m_img = Image.fromarray(m_arr, mode="P")
    m_img.putpalette(BCSS_PALETTE)
    m_img.save(val_mask_dir / fname)

    ckpt_dir = tmp_path / "arm_c_resume_ckpt"
    cfg_ep1 = resolve_stage1_config_from_env(
        overrides={
            "preset_name": "arm_c_mpppc_polysgd",
            "max_epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "ppc_start_epoch": 0,
            "pda_start_epoch": 0,
            "min_epochs_early_stop": 5,
            "early_stop_patience": 5,
        }
    )
    res1 = train_stage1_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_dir=ckpt_dir,
        allow_random_init=True,
        device_str="cpu",
        experiment_config=cfg_ep1,
    )
    latest_p = Path(res1["latest_checkpoint_path"])
    assert latest_p.is_file()
    assert len(res1["history"]) == 1

    # Spy on device-placement order, torch.load map_location, and optimizer.load_state_dict during resume
    events: List[str] = []
    seen_map_locations: List[object] = []
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_model_to = WSSSTissueStage1NetCAM.to
    orig_proj_to = ts1.ProjectionHead.to
    orig_ppc_to = MultiPrototypePPCLoss.to
    orig_build_opt = ts1.build_optimizer_and_scheduler
    orig_sgd_load = torch.optim.SGD.load_state_dict
    orig_torch_load = ts1.torch.load

    def spy_model_to(self_m, *args, **kwargs):
        events.append("model.to")
        return orig_model_to(self_m, *args, **kwargs)

    def spy_proj_to(self_m, *args, **kwargs):
        events.append("proj_head.to")
        return orig_proj_to(self_m, *args, **kwargs)

    def spy_ppc_to(self_m, *args, **kwargs):
        events.append("mpppc.to")
        return orig_ppc_to(self_m, *args, **kwargs)

    def spy_torch_load(f, *args, **kwargs):
        seen_map_locations.append(kwargs.get("map_location"))
        return orig_torch_load(f, *args, **kwargs)

    def spy_build_opt(model_arg, *args, **kwargs):
        events.append("build_optimizer")
        # Confirm every model + extra param is already on target_device before optimizer construction
        for p in model_arg.parameters():
            assert p.device == target_device
        for ep in kwargs.get("extra_params") or []:
            assert ep.device == target_device
        return orig_build_opt(model_arg, *args, **kwargs)

    def spy_sgd_load(self_opt, state_dict):
        events.append("optimizer.load_state_dict")
        ret = orig_sgd_load(self_opt, state_dict)
        for group in self_opt.param_groups:
            for p in group["params"]:
                st = self_opt.state.get(p, {})
                for sk, sv in st.items():
                    if isinstance(sv, torch.Tensor) and sk != "step":
                        assert sv.device == p.device == target_device
        return ret

    monkeypatch.setattr(WSSSTissueStage1NetCAM, "to", spy_model_to)
    monkeypatch.setattr(ts1.ProjectionHead, "to", spy_proj_to)
    monkeypatch.setattr(MultiPrototypePPCLoss, "to", spy_ppc_to)
    monkeypatch.setattr(ts1.torch, "load", spy_torch_load)
    monkeypatch.setattr(ts1, "build_optimizer_and_scheduler", spy_build_opt)
    monkeypatch.setattr(torch.optim.SGD, "load_state_dict", spy_sgd_load)

    cfg_ep2 = resolve_stage1_config_from_env(
        overrides={
            "preset_name": "arm_c_mpppc_polysgd",
            "max_epochs": 2,
            "batch_size": 2,
            "num_workers": 0,
            "ppc_start_epoch": 0,
            "pda_start_epoch": 0,
            "min_epochs_early_stop": 5,
            "early_stop_patience": 5,
        }
    )
    res2 = train_stage1_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_dir=ckpt_dir,
        resume_path=latest_p,
        allow_random_init=True,
        device_str=str(target_device),
        experiment_config=cfg_ep2,
    )

    assert len(res2["history"]) == 2
    assert np.isfinite(res2["history"][-1]["loss"])
    assert target_device in seen_map_locations
    build_idx = events.index("build_optimizer")
    load_idx = events.index("optimizer.load_state_dict")
    assert events.index("model.to") < build_idx < load_idx
    assert events.index("proj_head.to") < build_idx < load_idx
    assert events.index("mpppc.to") < build_idx < load_idx


def test_stage1_stop_after_epoch_decoupled_lr_schedule_and_resume(
    mock_dataset_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Verify STAGE1_STOP_AFTER_EPOCH decouples loop stop from MAX_EPOCHS LR horizon with zero LR shock on resume."""
    from src.config import resolve_stage1_config_from_env
    from src.training import train_stage1 as ts1

    # 1. Verify env parsing for STAGE1_STOP_AFTER_EPOCH
    monkeypatch.setenv("STAGE1_EXP_PRESET", "arm_a_polysgd")
    monkeypatch.setenv("STAGE1_MAX_EPOCHS", "20")
    monkeypatch.setenv("STAGE1_STOP_AFTER_EPOCH", "5")
    cfg_stop = resolve_stage1_config_from_env()
    assert cfg_stop.max_epochs == 20
    assert cfg_stop.stop_after_epoch == 5

    monkeypatch.setenv("STAGE1_STOP_AFTER_EPOCH", "0")
    with pytest.raises(ValueError, match="STAGE1_STOP_AFTER_EPOCH must be >= 1"):
        resolve_stage1_config_from_env()

    monkeypatch.delenv("STAGE1_STOP_AFTER_EPOCH", raising=False)
    cfg_resume_env = resolve_stage1_config_from_env()
    assert cfg_resume_env.max_epochs == 20
    assert cfg_resume_env.stop_after_epoch is None

    # Prepare val masks on official val patient so evaluate() runs full GatedPM_mIoU path
    splits = load_bcss_patient_splits()
    val_pid = splits["val"][0]
    val_img_dir = mock_dataset_dir / "val" / "img"
    val_mask_dir = mock_dataset_dir / "val" / "mask"
    fname = f"{val_pid}-DX1_xmin500_ymin500_MPP-0.2500+0.png"
    Image.fromarray(np.full((224, 224, 3), 120, dtype=np.uint8)).save(val_img_dir / fname)
    m_arr = np.zeros((224, 224), dtype=np.uint8)
    m_arr[112:, :] = 1
    m_img = Image.fromarray(m_arr, mode="P")
    m_img.putpalette(BCSS_PALETTE)
    m_img.save(val_mask_dir / fname)

    # Compact ResNet38d-compatible module so 20-epoch full schedule runs end-to-end quickly on CPU
    class _CompactStage1Net(nn.Module):
        def __init__(self, n_class: int = 4, num_classes: int = 4) -> None:
            super().__init__()
            out_c = n_class if n_class else num_classes
            self.conv1a = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
            self.bn7 = nn.BatchNorm2d(8)
            self.dropout7 = nn.Identity()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc8 = nn.Conv2d(8, out_c, kernel_size=1, bias=False)

        def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.bn7(self.conv1a(x)))

    monkeypatch.setattr(ts1, "WSSSTissueStage1NetCAM", _CompactStage1Net)

    recorded_lrs: List[float] = []
    orig_sgd_step = torch.optim.SGD.step

    def spy_sgd_step(self_opt, *args, **kwargs):
        recorded_lrs.append(float(self_opt.param_groups[0]["lr"]))
        return orig_sgd_step(self_opt, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", spy_sgd_step)

    # 2. Uninterrupted 20-epoch reference run (arm_a_polysgd)
    cfg_full = resolve_stage1_config_from_env(
        overrides={
            "preset_name": "arm_a_polysgd",
            "max_epochs": 20,
            "stop_after_epoch": None,
            "batch_size": 2,
            "num_workers": 0,
            "seed": 42,
        }
    )
    res_full = train_stage1_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_dir=tmp_path / "full_20ep",
        allow_random_init=True,
        device_str="cpu",
        enable_regression_guard=False,
        experiment_config=cfg_full,
    )
    assert len(res_full["history"]) == 20
    lrs_full = list(recorded_lrs)
    steps_per_ep = len(lrs_full) // 20
    assert steps_per_ep >= 1 and len(lrs_full) == 20 * steps_per_ep

    # 3. Split run part 1: max_epochs=20, stop_after_epoch=5
    recorded_lrs.clear()
    _ = capsys.readouterr()
    cfg_part1 = resolve_stage1_config_from_env(
        overrides={
            "preset_name": "arm_a_polysgd",
            "max_epochs": 20,
            "stop_after_epoch": 5,
            "batch_size": 2,
            "num_workers": 0,
            "seed": 42,
        }
    )
    ckpt_split_dir = tmp_path / "split_20ep"
    res_part1 = train_stage1_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_dir=ckpt_split_dir,
        allow_random_init=True,
        device_str="cpu",
        enable_regression_guard=False,
        experiment_config=cfg_part1,
    )
    part1_out = capsys.readouterr().out
    assert "[STOP AFTER EPOCH]" in part1_out
    assert "[EARLY STOP" not in part1_out
    assert len(res_part1["history"]) == 5
    assert Path(res_part1["latest_checkpoint_path"]).is_file()
    assert Path(res_part1["checkpoint_path"]).is_file()
    assert Path(res_part1["best_cls_checkpoint_path"]).is_file()
    lrs_part1 = list(recorded_lrs)
    assert len(lrs_part1) == 5 * steps_per_ep

    # 4. Split run part 2: resume with max_epochs=20, stop_after_epoch=None
    recorded_lrs.clear()
    cfg_part2 = resolve_stage1_config_from_env(
        overrides={
            "preset_name": "arm_a_polysgd",
            "max_epochs": 20,
            "stop_after_epoch": None,
            "batch_size": 2,
            "num_workers": 0,
            "seed": 42,
        }
    )
    res_part2 = train_stage1_pipeline(
        data_dir=mock_dataset_dir,
        checkpoint_dir=ckpt_split_dir,
        resume_path=res_part1["latest_checkpoint_path"],
        allow_random_init=True,
        device_str="cpu",
        enable_regression_guard=False,
        experiment_config=cfg_part2,
    )
    assert len(res_part2["history"]) == 20
    lrs_part2 = list(recorded_lrs)
    assert len(lrs_part2) == 15 * steps_per_ep

    # Ep 6 Step 1 (index 5 * steps_per_ep in uninterrupted run) must match to < 1e-9 (zero LR shock)
    ep6_step1_idx = 5 * steps_per_ep
    assert lrs_part2[0] == pytest.approx(lrs_full[ep6_step1_idx], abs=1e-9)
    assert lrs_part1 + lrs_part2 == pytest.approx(lrs_full, abs=1e-9)









