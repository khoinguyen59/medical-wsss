import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple


def extract_patient_id(filepath_or_name: str) -> str:
    """Extract patient/case identifier from histopathology patch filename.
    
    Standard BCSS-WSSS format:
        TCGA-A2-A0T2_1024_2048_[1 0 0 1].png -> 'TCGA-A2-A0T2'
    LUAD-HistoSeg format:
        TCGA-XX-XXXX+001+[1 0 0 0].png -> 'TCGA-XX-XXXX'
        GDPH_1234_patch_01.png -> 'GDPH_1234'
    """
    basename = os.path.basename(filepath_or_name)
    fname = os.path.splitext(basename)[0]
    
    # Check standard TCGA pattern (e.g. TCGA-A2-A0T2)
    tcga_match = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", fname, re.IGNORECASE)
    if tcga_match:
        return tcga_match.group(1).upper()

    # Fallback: split by first underscore or plus sign
    token = re.split(r"[_+]", fname)[0]
    return token.strip()


def verify_patient_leakage(
    train_paths: List[str],
    val_paths: List[str],
    test_paths: List[str],
) -> Dict[str, Any]:
    """Verify Patient-Level Anti-Leakage protocol across train, val, and test splits.
    
    Enforces [INVARIANT-08]:
        P_train intersection P_val == empty
        P_train intersection P_test == empty
        P_val intersection P_test == empty
    """
    train_patients: Set[str] = {extract_patient_id(p) for p in train_paths}
    val_patients: Set[str] = {extract_patient_id(p) for p in val_paths}
    test_patients: Set[str] = {extract_patient_id(p) for p in test_paths}

    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)

    has_leakage = (
        len(train_val_overlap) > 0
        or len(train_test_overlap) > 0
        or len(val_test_overlap) > 0
    )

    result = {
        "has_leakage": has_leakage,
        "train_patients_count": len(train_patients),
        "val_patients_count": len(val_patients),
        "test_patients_count": len(test_patients),
        "train_val_overlap": sorted(list(train_val_overlap)),
        "train_test_overlap": sorted(list(train_test_overlap)),
        "val_test_overlap": sorted(list(val_test_overlap)),
    }

    if has_leakage:
        raise ValueError(
            f"Patient leakage detected! "
            f"Train-Val: {train_val_overlap}, "
            f"Train-Test: {train_test_overlap}, "
            f"Val-Test: {val_test_overlap}"
        )

    return result


def split_patients_kfold(
    patient_ids: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """Perform deterministic K-Fold split at the Patient ID level."""
    unique_patients = sorted(list(set(patient_ids)))
    import random
    rng = random.Random(seed)
    shuffled = list(unique_patients)
    rng.shuffle(shuffled)

    folds = []
    fold_size = len(shuffled) // n_splits
    remainder = len(shuffled) % n_splits

    start = 0
    for i in range(n_splits):
        end = start + fold_size + (1 if i < remainder else 0)
        val_fold = shuffled[start:end]
        train_fold = [p for p in shuffled if p not in val_fold]
        folds.append((train_fold, val_fold))
        start = end

    return folds


def load_patient_splits(
    config_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Load and verify deterministic patient-level splits manifest for BCSS-WSSS.

    Parameters
    ----------
    config_path : Optional[str]
        Path to bcss_patient_splits.json. If None, defaults to
        src/config/bcss_patient_splits.json.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing 'train', 'val', 'valid', and 'test' patient ID lists.

    Raises
    ------
    FileNotFoundError
        If config_path does not exist.
    ValueError
        If patient leakage is detected across splits.
    """
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config", "bcss_patient_splits.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Split configuration not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "splits" in data and isinstance(data["splits"], dict):
        train_ids = data["splits"].get("train", [])
        val_ids = data["splits"].get("val", data["splits"].get("valid", []))
        test_ids = data["splits"].get("test", [])
    else:
        train_ids = data.get("train", [])
        val_ids = data.get("val", data.get("valid", []))
        test_ids = data.get("test", [])

    # Enforce patient-level anti-leakage verification
    verify_patient_leakage(train_ids, val_ids, test_ids)

    return {
        "train": train_ids,
        "val": val_ids,
        "valid": val_ids,
        "test": test_ids,
    }

