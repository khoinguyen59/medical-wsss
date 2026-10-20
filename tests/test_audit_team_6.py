"""
Rigorous Adversarial Unit Test Suite by Audit Team 6 (Auditor 6).
Evaluates edge cases, mathematical integrity, device consistency, patient anti-leakage,
and deliverable contracts for SE374.R11 Weakly-Supervised Histopathology Segmentation.

Strict adherence to:
- Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Invariant 07: Zero Mockup, true PyTorch computation.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi.testclient import TestClient

from src.datasets.bcss_dataset import BCSSDataset
from src.datasets.otsu_filter import apply_otsu_gating_to_prediction, compute_otsu_tissue_mask
from src.datasets.patient_split import (
    extract_patient_id,
    load_patient_splits,
    verify_patient_leakage,
)
from src.losses_metrics.losses import (
    MultiLayerPseudoSupervisionLoss,
    UncertaintyRegularizationLoss,
)
from src.losses_metrics.metrics import (
    MedicalSegmentationMetrics,
    compute_tumor_stroma_ratio,
    compute_viable_tumor_percentage,
)
from src.models.backbone import MultiStageFeatureBackbone
from src.models.deeplabv3plus import DeepLabV3PlusWithCGM
from src.models.pda_generator import ProgressiveDropoutAttention
from src.visualization.api_server import app


# =========================================================================
# TEST SUITE 1: ADVERSARIAL PATIENT LEAKAGE & PATIENT ID EXTRACTION
# =========================================================================

def test_adversarial_patient_leakage_catches_all_pairwise_overlaps():
    """Verify verify_patient_leakage detects subtle cross-split patient contaminations."""
    # Clean baseline
    train_ids = ["TCGA-A1-0001", "TCGA-A1-0002", "TCGA-A1-0003"]
    val_ids = ["TCGA-A2-0001", "TCGA-A2-0002"]
    test_ids = ["TCGA-A3-0001", "TCGA-A3-0002"]

    res = verify_patient_leakage(train_ids, val_ids, test_ids)
    assert not res["has_leakage"]

    # Contaminate Train and Val
    with pytest.raises(ValueError, match="Patient leakage detected"):
        verify_patient_leakage(train_ids + ["TCGA-A2-0001"], val_ids, test_ids)

    # Contaminate Train and Test
    with pytest.raises(ValueError, match="Patient leakage detected"):
        verify_patient_leakage(train_ids, val_ids, test_ids + ["TCGA-A1-0001"])

    # Contaminate Val and Test
    with pytest.raises(ValueError, match="Patient leakage detected"):
        verify_patient_leakage(train_ids, val_ids + ["TCGA-A3-0002"], test_ids)


def test_adversarial_patient_id_parsing_robustness():
    """Tests patient ID extraction across standard, lowercase, path-prefixed, and malformed names."""
    # Standard format
    assert extract_patient_id("TCGA-A2-A0T2_1024_2048_[1 0 0 1].png") == "TCGA-A2-A0T2"
    # Lowercase should be normalized to uppercase
    assert extract_patient_id("tcga-a2-a0t2_1024_2048_[1 0 0 1].png") == "TCGA-A2-A0T2"
    # Full nested directory path
    assert extract_patient_id("/deep/nested/path/to/TCGA-BH-A0B8_crop.png") == "TCGA-BH-A0B8"
    assert extract_patient_id("C:\\Windows\\Path\\TCGA-BH-A0B8_crop.png") == "TCGA-BH-A0B8"
    # Plus delimiter format
    assert extract_patient_id("TCGA-XX-XXXX+001+[1 0 0 0].png") == "TCGA-XX-XXXX"
    # Non-TCGA hospital format fallback
    assert extract_patient_id("GDPH_1234_patch_01.png") == "GDPH"


def test_bcss_151_manifest_strict_disjointness():
    """Audits src/config/bcss_patient_splits.json to guarantee 0.0% patient leakage and 151 total count."""
    splits = load_patient_splits()
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Exact counts
    assert len(train_set) == 91, f"Expected 91 train patients, got {len(train_set)}"
    assert len(val_set) == 30, f"Expected 30 val patients, got {len(val_set)}"
    assert len(test_set) == 30, f"Expected 30 test patients, got {len(test_set)}"
    assert len(train_set) + len(val_set) + len(test_set) == 151

    # Disjointness checks
    assert len(train_set.intersection(val_set)) == 0, "Train and Val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test overlap!"


# =========================================================================
# TEST SUITE 2: MATHEMATICAL FORMULATIONS & LOSS INTEGRITY
# =========================================================================

def test_uncertainty_loss_device_handling_scalar_tensor():
    """Verifies gradient flow through UncertaintyRegularizationLoss."""
    loss_module = UncertaintyRegularizationLoss(entropy_threshold=0.65)
    
    # Simulate forward pass with float logits
    logits = torch.randn(2, 5, 32, 32, requires_grad=True)
    mask = torch.randint(0, 5, (2, 32, 32))
    
    loss_val = loss_module(logits, mask)
    assert loss_val.ndim == 0
    assert not torch.isnan(loss_val)
    assert not torch.isinf(loss_val)
    loss_val.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_uncertainty_loss_with_ignore_index_255_exposes_index_error():
    """Verifies that UncertaintyRegularizationLoss supports ignore_index=255 without crashing."""
    loss_module = UncertaintyRegularizationLoss(entropy_threshold=0.65, ignore_index=255)
    logits = torch.randn(2, 5, 16, 16)
    mask = torch.randint(0, 5, (2, 16, 16))
    mask[0, :4, :4] = 255  # Standard ignore index

    loss = loss_module(logits, mask)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_mlps_loss_empty_masks_gradient_graph():
    """Checks MultiLayerPseudoSupervisionLoss behavior when pseudo_masks dict has unexpected keys."""
    mlps = MultiLayerPseudoSupervisionLoss(weights={"bn7": 1.0, "b5_2": 0.5, "b4_5": 0.25})
    logits = torch.randn(2, 5, 32, 32, requires_grad=True)
    
    # Case 1: Matching masks -> gradient flows
    masks_valid = {
        "bn7": torch.randint(0, 5, (2, 32, 32)),
        "b5_2": torch.randint(0, 5, (2, 16, 16)),
    }
    loss_valid = mlps(logits, masks_valid)
    loss_valid.backward()
    assert logits.grad is not None

    # Case 2: No matching keys -> total_loss is initialized properly connected to logits graph
    logits_zero = torch.randn(2, 5, 32, 32, requires_grad=True)
    masks_empty = {"unrecognized_layer": torch.randint(0, 5, (2, 32, 32))}
    loss_empty = mlps(logits_zero, masks_empty)
    assert loss_empty.item() == 0.0
    loss_empty.backward()
    assert logits_zero.grad is not None


# =========================================================================
# TEST SUITE 3: CODE ARCHITECTURE & TENSOR DIMENSION CONTRACTS
# =========================================================================

def test_backbone_cam_intermediate_layer_channel_adaptation():
    """Audits backbone.py line 212 weight slicing for b4_5 layer (256 channels vs 512 classifier channels)."""
    backbone = MultiStageFeatureBackbone(in_channels=3, num_classes=4, stage_channels=[64, 128, 256, 512, 512])
    backbone.eval()
    x = torch.randn(1, 3, 128, 128)

    # Stage 5 CAM (native 512 channels)
    cam_bn7 = backbone.compute_cam(x, feature_layer="bn7")
    assert cam_bn7.shape == (1, 4, 4, 4)

    # Stage 3 CAM (b4_5 has 256 channels, fc_head has 512 channels)
    # Audits if compute_cam runs and executes the weight slice logic
    cam_b4_5 = backbone.compute_cam(x, feature_layer="b4_5")
    assert cam_b4_5.shape[1] == 4
    assert cam_b4_5.shape[-2:] == (8, 8)


def test_otsu_filter_batch_dimension_crash_exposed():
    """Verifies that apply_otsu_gating_to_prediction supports batch dimensions."""
    img_single = np.ones((64, 64, 3), dtype=np.uint8) * 200
    pred_batch = np.zeros((3, 64, 64), dtype=np.int64)

    res = apply_otsu_gating_to_prediction(pred_batch, img_single, background_class_idx=4)
    assert res.shape == (3, 64, 64)


def test_bcss_dataset_transform_is_ignored_by_design_defect():
    """Verifies that BCSSDataset correctly invokes self.transform when provided."""
    transform_invoked = False

    def dummy_transform(img):
        nonlocal transform_invoked
        transform_invoked = True
        return img

    dataset = BCSSDataset(root_dir="data/sample_dataset", split="valid", transform=dummy_transform)
    if len(dataset) > 0:
        _ = dataset[0]
        assert transform_invoked, "BCSSDataset applies self.transform!"


# =========================================================================
# TEST SUITE 4: CLINICAL PATHOLOGY METRICS & CHALLENGING HISTOLOGY
# =========================================================================

def test_viable_tumor_percentage_with_ignore_index_pollution():
    """Demonstrates that ignore_index (255) is safely excluded from VTP calculation."""
    # 100x100 mask: 10 tumor pixels (class 0), 90 background (class 4)
    pred_clean = np.full((100, 100), fill_value=4, dtype=np.int64)
    pred_clean[:10, :1] = 0  # 10 tumor pixels
    vtp_clean = compute_viable_tumor_percentage(pred_clean, tumor_class_idx=0, background_class_idx=4)
    assert vtp_clean == 100.0  # 10 / 10 * 100%

    # Now add ignore pixels (value 255)
    pred_polluted = pred_clean.copy()
    pred_polluted[50:, :] = 255  # 5000 ignore pixels
    vtp_polluted = compute_viable_tumor_percentage(pred_polluted, tumor_class_idx=0, background_class_idx=4, ignore_index=255)
    
    # When ignore_index=255 is properly excluded, VTP remains 100.0%
    assert vtp_polluted == 100.0


def test_tumor_stroma_ratio_excluding_lymphocytes():
    """Demonstrates TSR ignores class 2 (Lymphocytic infiltrate) in the denominator."""
    # Mask with 50 tumor, 0 stroma, 50 lymphocytes
    pred = np.zeros((10, 10), dtype=np.int64)
    pred[:5, :] = 0  # 50 tumor pixels
    pred[5:, :] = 2  # 50 lymphocyte pixels (STR is 0)

    tsr = compute_tumor_stroma_ratio(pred, tumor_class_idx=0, stroma_class_idx=1)
    # Tumor / (Tumor + Stroma) = 50 / (50 + 0) = 100%
    # But 50% of the lesion is actually dense lymphocytic infiltration
    assert tsr == 100.0, "TSR ignored lymphocyte infiltration component"


def test_otsu_gating_adipose_tissue_wiped_out():
    """Demonstrates that mature adipose tissue (gray >= 235) is 100% wiped out by Otsu gating.
    
    In breast lumpectomy specimens, adipose tissue makes up > 50-80% of normal parenchyma.
    Line 53 of otsu_filter.py enforces is_non_white = gray < 235.
    Lipid vacuoles in fat cells have gray in [238, 245], so fat retention is exactly 0.0%.
    """
    adipose_patch = np.full((128, 128, 3), fill_value=238, dtype=np.uint8)
    tissue_mask = compute_otsu_tissue_mask(adipose_patch)
    fat_retention_ratio = np.sum(tissue_mask) / (128 * 128)

    assert fat_retention_ratio == 0.0, "Confirmed: Mature breast adipose tissue is 100% wiped out"


# =========================================================================
# TEST SUITE 5: API SPECIFICATION CONTRACT AUDIT
# =========================================================================

def test_api_reported_endpoints_contract_existence():
    """Audits whether the endpoints claimed in SE374_FINAL_TECHNICAL_REPORT_WSSS.md actually exist."""
    client = TestClient(app)
    
    # Claimed: GET /health -> Actual: GET /api/v1/health
    resp_v1_health = client.get("/api/v1/health")
    assert resp_v1_health.status_code == 200

    resp_reported_health = client.get("/health")
    reported_health_exists = (resp_reported_health.status_code == 200)

    # Claimed: POST /segment
    resp_segment = client.post("/segment")
    segment_exists = (resp_segment.status_code != 404)

    # Claimed: POST /biomarkers
    resp_biomarkers = client.post("/biomarkers")
    biomarkers_exists = (resp_biomarkers.status_code != 404)

    # Claimed: GET /splits
    resp_splits = client.get("/splits")
    splits_exists = (resp_splits.status_code != 404)

    # Claimed: GET /tile/0/0/0.png
    resp_tile = client.get("/tile/0/0/0.png")
    tile_exists = (resp_tile.status_code != 404)

    # Document missing endpoints
    missing_endpoints = []
    if not reported_health_exists:
        missing_endpoints.append("GET /health")
    if not segment_exists:
        missing_endpoints.append("POST /segment")
    if not biomarkers_exists:
        missing_endpoints.append("POST /biomarkers")
    if not splits_exists:
        missing_endpoints.append("GET /splits")
    if not tile_exists:
        missing_endpoints.append("GET /tile/{z}/{x}/{y}.png")

    assert len(missing_endpoints) == 0, (
        f"API contract discrepancy! Reported endpoints missing in api_server.py: {missing_endpoints}"
    )


# =========================================================================
# TEST SUITE 6: INVARIANT 12 STRICT ASCII HYPHEN GOVERNANCE AUDIT
# =========================================================================

def test_invariant_12_strict_ascii_hyphen_in_source_and_reports():
    """Verifies that NO em-dash (U+2014) or en-dash (U+2013) exists across src/ and product/."""
    forbidden_chars = {
        "\u2014": "Em-dash (U+2014)",
        "\u2013": "En-dash (U+2013)",
        "\u2015": "Horizontal bar (U+2015)",
        "\u2212": "Minus sign (U+2212)",
    }

    project_root = Path(__file__).resolve().parent.parent
    scan_dirs = [project_root / "src", project_root / "product" / "reports"]

    violations: List[str] = []
    for sdir in scan_dirs:
        for fpath in sdir.rglob("*"):
            if fpath.is_file() and fpath.suffix in [".py", ".md", ".json"]:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    for char, name in forbidden_chars.items():
                        if char in content:
                            violations.append(f"{fpath.relative_to(project_root)}: contains forbidden {name}")
                except Exception:
                    pass

    assert len(violations) == 0, f"Invariant 12 violations detected:\n" + "\n".join(violations)
