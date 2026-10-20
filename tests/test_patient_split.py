import numpy as np
import torch
import pytest
import os
import glob


from src.datasets.patient_split import (
    extract_patient_id,
    load_patient_splits,
    verify_patient_leakage,
    split_patients_kfold,
)
from src.datasets.bcss_dataset import BCSSDataset


def test_extract_patient_id_tcga():
    """Verify TCGA patient ID extraction from diverse filename patterns."""
    f1 = "TCGA-A2-A0T2_1024_2048_[1 0 0 1].png"
    assert extract_patient_id(f1) == "TCGA-A2-A0T2"

    f2 = "data/raw/TCGA-BH-A0B8_2048_4096_[0 1 1 0].png"
    assert extract_patient_id(f2) == "TCGA-BH-A0B8"

    f3 = "TCGA-E2-A152+001+[1 0 0 0].png"
    assert extract_patient_id(f3) == "TCGA-E2-A152"

    f4 = "GDPH_1234_patch_05.png"
    assert extract_patient_id(f4) == "GDPH"


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


def test_bcss_dataset_loader_dry_run():
    """Sanity dry-run: load batches and verify tensor shapes and memory."""
    sample_root = os.path.join("data", "sample_dataset")

    train_ds = BCSSDataset(root_dir=sample_root, split="train")
    assert len(train_ds) == 4
    img, label, fname = train_ds[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(label, torch.Tensor)
    assert label.shape == (4,)

    val_ds = BCSSDataset(root_dir=sample_root, split="valid")
    assert len(val_ds) == 2
    img_v, mask_v, fname_v = val_ds[0]
    assert img_v.shape == (3, 224, 224)
    assert mask_v.shape == (224, 224)


def test_151_patient_split_manifest_zero_leakage():
    """Verify that the 151-patient split manifest exists, has exactly 151 distinct patients,
    adheres to the 91/30/30 split ratio, and achieves 0.0% patient leakage.
    """
    manifest_path = os.path.join("src", "config", "bcss_patient_splits.json")
    assert os.path.exists(manifest_path), f"Manifest file must exist at {manifest_path}"

    # Load splits using loader function
    splits = load_patient_splits(manifest_path)

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    # Check exact counts per specification
    assert len(train) == 91, f"Expected 91 train patients, got {len(train)}"
    assert len(val) == 30, f"Expected 30 val patients, got {len(val)}"
    assert len(test) == 30, f"Expected 30 test patients, got {len(test)}"

    # Check total patient count
    total_count = len(train) + len(val) + len(test)
    assert total_count == 151, f"Expected exactly 151 patients, got {total_count}"

    # Check all IDs are unique (no duplicates within or across splits)
    all_unique = set(train) | set(val) | set(test)
    assert len(all_unique) == 151, f"Expected 151 unique patient IDs, got {len(all_unique)}"

    # Check pairwise disjoint sets (Zero Patient Leakage)
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    assert len(train_val_overlap) == 0, f"Patient leakage between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Patient leakage between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Patient leakage between Val and Test: {val_test_overlap}"

    # Verify via verify_patient_leakage utility
    leakage_result = verify_patient_leakage(train, val, test)
    assert leakage_result["has_leakage"] is False
    assert leakage_result["train_patients_count"] == 91
    assert leakage_result["val_patients_count"] == 30
    assert leakage_result["test_patients_count"] == 30

    # Verify sample dataset patients consistency
    assert "TCGA-A2-A0T2" in train_set
    assert "TCGA-BH-A0B8" in train_set
    assert "TCGA-E2-A152" in val_set
    assert "TCGA-C8-A1HJ" in test_set


def test_load_patient_splits_invalid_path():
    """Verify FileNotFoundError is raised when config path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_patient_splits("non_existent_manifest.json")

