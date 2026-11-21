import glob
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image
from torch.utils.data import DataLoader

from src.colab_pkg.downloader import resolve_split_dirs
from src.datasets.bcss_dataset import BCSSDataset
from src.datasets.otsu_filter import compute_otsu_tissue_mask
from src.datasets.patient_split import (
    extract_patient_id,
    load_patient_splits,
    split_patients_kfold,
    verify_patient_leakage,
)
from src.training.dataset_loader import parse_patch_filename


def test_extract_patient_id_tcga_and_luad():
    """Verify TCGA and LUAD patient ID extraction from diverse filename patterns."""
    f1 = "TCGA-A2-A0T2_1024_2048_[1 0 0 1].png"
    assert extract_patient_id(f1) == "TCGA-A2-A0T2"

    f2 = "data/raw/TCGA-BH-A0B8_2048_4096_[0 1 1 0].png"
    assert extract_patient_id(f2) == "TCGA-BH-A0B8"

    f3 = "TCGA-E2-A152+001+[1 0 0 0].png"
    assert extract_patient_id(f3) == "TCGA-E2-A152"

    f4 = "GDPH_1234_patch_05.png"
    assert extract_patient_id(f4) == "GDPH"


def test_parse_patch_filename_all_label_formats():
    """Verify parse_patch_filename handles space-separated, comma-separated, and contiguous labels."""
    pid1, x1, y1, lbl1 = parse_patch_filename("TCGA-A2-A0T2_1024_2048_[1 0 1 0].png")
    assert pid1 == "TCGA-A2-A0T2"
    assert (x1, y1) == (1024, 2048)
    np.testing.assert_array_equal(lbl1, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))

    pid2, x2, y2, lbl2 = parse_patch_filename("TCGA-BH-A0B8_512_256_[0,1,0,1].png")
    assert pid2 == "TCGA-BH-A0B8"
    assert (x2, y2) == (512, 256)
    np.testing.assert_array_equal(lbl2, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))

    pid3, _, _, lbl3 = parse_patch_filename("TCGA-E2-A14N_0_0_[1100].png")
    assert pid3 == "TCGA-E2-A14N"
    np.testing.assert_array_equal(lbl3, np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_verify_patient_leakage_on_sample_dataset():
    """Verify that sample_dataset passes Patient-Level Anti-Leakage protocol."""
    sample_root = os.path.join("data", "sample_dataset")
    assert os.path.exists(sample_root), "data/sample_dataset must exist"

    train_files = glob.glob(os.path.join(sample_root, "train", "*.png"))
    val_files = glob.glob(os.path.join(sample_root, "valid", "img", "*.png"))
    test_files = glob.glob(os.path.join(sample_root, "test", "img", "*.png"))

    assert len(train_files) > 0, "train_files should not be empty"
    assert len(val_files) > 0, "val_files should not be empty"
    assert len(test_files) > 0, "test_files should not be empty"

    result = verify_patient_leakage(train_files, val_files, test_files)
    assert result["has_leakage"] is False
    assert result["train_patients_count"] == 2
    assert result["val_patients_count"] == 1
    assert result["test_patients_count"] == 1


def test_verify_patient_leakage_catches_overlap():
    """Adversarial check: ensure ValueError is raised when patient overlap exists."""
    train_files = ["TCGA-A2-A0T2_01.png", "TCGA-BH-A0B8_01.png"]
    val_files = ["TCGA-E2-A152_01.png"]
    test_files = ["TCGA-A2-A0T2_02.png"]  # Overlaps with train!

    with pytest.raises(ValueError) as excinfo:
        verify_patient_leakage(train_files, val_files, test_files)
    assert "Patient leakage detected" in str(excinfo.value)


def test_split_patients_kfold():
    """Verify K-Fold patient splits are deterministic and disjoint."""
    patients = [f"TCGA-P{i:02d}" for i in range(10)]
    folds = split_patients_kfold(patients, n_splits=5, seed=42)

    assert len(folds) == 5
    for train_p, val_p in folds:
        assert len(set(train_p).intersection(set(val_p))) == 0
        assert len(train_p) + len(val_p) == 10


def test_bcss_dataset_loader_and_dataloader_batching():
    """Load real sample_dataset via BCSSDataset and PyTorch DataLoader(batch_size=2)."""
    sample_root = os.path.join("data", "sample_dataset")

    train_ds = BCSSDataset(root_dir=sample_root, split="train")
    assert len(train_ds) == 4
    loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    imgs, labels, fnames = next(iter(loader))

    assert imgs.shape == (2, 3, 224, 224)
    assert imgs.dtype == torch.float32
    assert labels.shape == (2, 4)
    assert len(fnames) == 2

    val_ds = BCSSDataset(root_dir=sample_root, split="valid")
    assert len(val_ds) == 2
    img_v, mask_v, _ = val_ds[0]
    assert img_v.shape == (3, 224, 224)
    assert mask_v.shape == (224, 224)
    assert set(np.unique(mask_v.numpy())).issubset({0, 1, 2, 3, 4, 255})


def test_bcss_dataset_nested_topologies_and_missing_mask_with_transform(tmp_path: Path):
    """Test BCSSDataset across nested BCSS-WSSS/train/img topologies, custom transform, and missing mask fallback."""
    # Topology 1: Nested BCSS-WSSS/train/img/*.png
    nested_train = tmp_path / "BCSS-WSSS" / "train" / "img"
    nested_train.mkdir(parents=True)
    rgb = np.full((64, 64, 3), fill_value=180, dtype=np.uint8)
    Image.fromarray(rgb).save(nested_train / "TCGA-A2-A0T2_0_0_[1 0 1 0].png")

    ds_train = BCSSDataset(root_dir=str(tmp_path), split="train")
    assert len(ds_train) == 1
    img_t, lbl_t, _ = ds_train[0]
    assert img_t.shape == (3, 64, 64)
    np.testing.assert_array_equal(lbl_t.numpy(), np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))

    # Topology 2: Test split with custom transform AND missing mask directory (regression test for img_np UnboundLocalError)
    nested_test_img = tmp_path / "BCSS-WSSS" / "test" / "img"
    nested_test_img.mkdir(parents=True)
    Image.fromarray(rgb).save(nested_test_img / "TCGA-D8-A27F_0_0.png")

    def custom_transform(pil_img: Image.Image) -> torch.Tensor:
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    ds_test_no_mask = BCSSDataset(
        root_dir=str(tmp_path),
        split="test",
        transform=custom_transform,
    )
    assert len(ds_test_no_mask) == 1
    img_te, mask_te, fname_te = ds_test_no_mask[0]
    assert img_te.shape == (3, 64, 64)
    assert mask_te.shape == (64, 64)
    assert torch.all(mask_te == 0)
    assert fname_te == "TCGA-D8-A27F_0_0.png"

    # Topology 3: Mask containing ignore_index=255 must preserve 255 while clamping 0..4
    nested_test_mask = tmp_path / "BCSS-WSSS" / "test" / "mask"
    nested_test_mask.mkdir(parents=True)
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_arr[:20, :] = 1
    mask_arr[20:40, :] = 255
    Image.fromarray(mask_arr, mode="L").save(nested_test_mask / "TCGA-D8-A27F_0_0.png")

    ds_test_with_mask = BCSSDataset(root_dir=str(tmp_path), split="test", transform=custom_transform)
    _, loaded_mask, _ = ds_test_with_mask[0]
    assert torch.all(loaded_mask[:20, :] == 1)
    assert torch.all(loaded_mask[20:40, :] == 255)


def test_resolve_split_dirs_nested_archive_and_mock_rejection(tmp_path: Path):
    """Verify resolve_split_dirs finds deeply nested cohorts while rejecting sample_dataset mocks."""
    # 1. Create a mock sample_dataset directory that must be ignored
    mock_img_dir = tmp_path / "sample_dataset" / "test" / "img"
    mock_img_dir.mkdir(parents=True)
    for i in range(12):
        Image.new("RGB", (16, 16)).save(mock_img_dir / f"TCGA-AA-A0{i:02d}_{i}.png")

    root_found, img_found, _ = resolve_split_dirs(tmp_path, split="test")
    assert root_found is None and img_found is None

    # 2. Create a genuine nested archive structure with >= 10 images across >= 2 patients
    real_img_dir = tmp_path / "extracted_archive" / "BCSS-WSSS" / "test" / "img"
    real_mask_dir = tmp_path / "extracted_archive" / "BCSS-WSSS" / "test" / "mask"
    real_img_dir.mkdir(parents=True)
    real_mask_dir.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (16, 16)).save(real_img_dir / f"TCGA-D8-A27F_{i}.png")
        Image.new("RGB", (16, 16)).save(real_img_dir / f"TCGA-E2-A14N_{i}.png")

    root_found, img_found, mask_found = resolve_split_dirs(tmp_path, split="test")
    assert root_found == tmp_path / "extracted_archive" / "BCSS-WSSS"
    assert img_found == real_img_dir
    assert mask_found == real_mask_dir


def test_otsu_tissue_preservation_on_pale_mucinous_vs_glass():
    """Verify Otsu tissue mask retains >75% of pale mucinous H&E tissue while rejecting >90% of white glass."""
    # Left half: pale lavender/pink mucinous tumor (RGB ~ [225, 195, 220], low-moderate saturation, gray < 220)
    # Right half: bright white microscope glass slide (RGB = [250, 250, 250])
    img = np.full((128, 128, 3), fill_value=250, dtype=np.uint8)
    img[:, :64] = [225, 195, 220]

    mask = compute_otsu_tissue_mask(img)
    pale_tissue_retained = float(np.mean(mask[:, :64]))
    glass_retained = float(np.mean(mask[:, 64:]))

    assert pale_tissue_retained > 0.75, (
        f"Pale mucinous tissue was over-erased by Otsu filter: retained={pale_tissue_retained:.3f}"
    )
    assert glass_retained < 0.10, (
        f"Pure glass background was not rejected by Otsu filter: glass={glass_retained:.3f}"
    )


def test_151_patient_split_manifest_exact_canonical_counts_and_zero_leakage():
    """Verify that the canonical 151-patient split manifest has exact 106/22/23 counts and 0.0% leakage."""
    manifest_path = os.path.join("src", "config", "bcss_patient_splits.json")
    assert os.path.exists(manifest_path), f"Manifest file must exist at {manifest_path}"

    splits = load_patient_splits(manifest_path)
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    # Enforce the single canonical 106 / 22 / 23 = 151 patient split
    assert len(train) == 106, f"Expected exactly 106 train patients, got {len(train)}"
    assert len(val) == 22, f"Expected exactly 22 val patients, got {len(val)}"
    assert len(test) == 23, f"Expected exactly 23 test patients, got {len(test)}"
    assert len(train) + len(val) + len(test) == 151

    train_set, val_set, test_set = set(train), set(val), set(test)
    assert len(train_set | val_set | test_set) == 151
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0

    leakage_result = verify_patient_leakage(train, val, test)
    assert leakage_result["has_leakage"] is False
    assert leakage_result["train_patients_count"] == 106
    assert leakage_result["val_patients_count"] == 22
    assert leakage_result["test_patients_count"] == 23


def test_load_patient_splits_invalid_path():
    """Verify FileNotFoundError is raised when config path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_patient_splits("non_existent_manifest.json")


@given(st.from_regex(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}_[0-9]+_[0-9]+_\[[01 ]+\]\.png$", fullmatch=True))
def test_hypothesis_extract_patient_id_tcga_format(filename):
    """Property test: TCGA filenames always yield valid 12-char patient ID."""
    pid = extract_patient_id(filename)
    assert pid.startswith("TCGA-")
    assert len(pid) == 12


@given(st.text())
def test_hypothesis_extract_patient_id_never_crashes(text):
    """Property test: extract_patient_id must be resilient to arbitrary text input."""
    result = extract_patient_id(text)
    assert isinstance(result, str)


def test_stage2_datasets_patient_anti_leakage_fail_fast(tmp_path: Path):
    """Verify BCSSStage2Dataset and BCSSStage2ValDataset raise RuntimeError on 0-match patient_ids."""
    from src.training.dataset_loader import (
        BCSS_PALETTE,
        BCSSStage2Dataset,
        BCSSStage2ValDataset,
    )

    train_img_dir = tmp_path / "train"
    pm_dir = tmp_path / "train_PM"
    val_img_dir = tmp_path / "val" / "img"
    val_mask_dir = tmp_path / "val" / "mask"
    for d in (train_img_dir, pm_dir / "PM_bn7", pm_dir / "PM_b5_2", pm_dir / "PM_b4_5", val_img_dir, val_mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_fname = "TCGA-A2-A0T2_0_0_[1 0 0 0].png"
    other_fname = "TCGA-BH-A0B8_0_0_[0 1 0 0].png"
    for fname in (train_fname, other_fname):
        Image.fromarray(np.full((64, 64, 3), 140, dtype=np.uint8)).save(train_img_dir / fname)
        for sub in ("PM_bn7", "PM_b5_2", "PM_b4_5"):
            m_img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="P")
            m_img.putpalette(BCSS_PALETTE)
            m_img.save(pm_dir / sub / fname)

    val_fname = "TCGA-AN-A03X_0_0.png"
    Image.fromarray(np.full((64, 64, 3), 140, dtype=np.uint8)).save(val_img_dir / val_fname)
    vm_img = Image.fromarray(np.ones((64, 64), dtype=np.uint8), mode="P")
    vm_img.putpalette(BCSS_PALETTE)
    vm_img.save(val_mask_dir / val_fname)

    # P0 Fail-Fast: 0 matching patients must raise RuntimeError (no silent unfiltered fallback)
    with pytest.raises(RuntimeError, match="matched 0 Stage-2 train pseudo-masks"):
        BCSSStage2Dataset(
            img_dir=train_img_dir,
            pm_dir=pm_dir,
            patient_ids=["TCGA-ZZ-9999"],
        )
    with pytest.raises(RuntimeError, match="matched 0 Stage-2 val patches"):
        BCSSStage2ValDataset(
            img_dir=val_img_dir,
            mask_dir=val_mask_dir,
            patient_ids=["TCGA-ZZ-9999"],
        )

    # Valid filter restricts strictly to allowed patient IDs
    ds_train = BCSSStage2Dataset(
        img_dir=train_img_dir,
        pm_dir=pm_dir,
        patient_ids=["TCGA-A2-A0T2"],
        is_train=False,
    )
    assert len(ds_train) == 1
    assert extract_patient_id(ds_train.files[0].name) == "TCGA-A2-A0T2"

    ds_val = BCSSStage2ValDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        patient_ids=["TCGA-AN-A03X"],
    )
    assert len(ds_val) == 1
    assert extract_patient_id(ds_val.files[0].name) == "TCGA-AN-A03X"
