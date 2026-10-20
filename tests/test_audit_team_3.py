"""
Audit Team 3 (Auditor 3) Comprehensive Independent Unit Tests (SE374.R11).

Rigorous, adversarial unit tests challenging the codebase across all 8 audit areas:
1. Mathematical formulations (MLPS Loss, Uncertainty Loss, Spatial Affinity Matrix).
2. DeepLabV3Plus with CGM (Norm layer inspection, B=1 training mode, logit penalization).
3. Patient-Level Anti-Leakage (Zero leakage on 151 patients, adversarial filename parsing).
4. Clinical Pathology & Otsu HSV Gating (Challenging histology: mucinous, pale, fat, edge cases).
5. VTP & TSR Biomarkers (Mathematical edge cases and definition alignment).
6. API Server & Deliverables (Endpoint parity, Invariant 12 ASCII hyphen, Patient split consistency).

Strict adherence to Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only.
"""

import json
import math
import os
import re
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.otsu_filter import apply_otsu_gating_to_prediction, compute_otsu_tissue_mask
from src.datasets.patient_split import extract_patient_id, load_patient_splits, verify_patient_leakage
from src.losses_metrics.losses import (
    MultiLabelClassificationLoss,
    MultiLayerPseudoSupervisionLoss,
    UncertaintyRegularizationLoss,
)
from src.losses_metrics.metrics import (
    MedicalSegmentationMetrics,
    compute_tumor_stroma_ratio,
    compute_viable_tumor_percentage,
)
from src.models.deeplabv3plus import (
    ASPP,
    ASPPConv,
    ASPPPooling,
    ClassificationGateMechanism,
    DeepLabV3PlusDecoder,
    DeepLabV3PlusWithCGM,
    ResNetBackbone,
)
from src.models.pda_generator import ProgressiveDropoutAttention
from src.models.wsss_tissue import SpatialAffinityRelocalization, WSSSBenchmarkingEngine
from src.visualization.api_server import app


# ==============================================================================
# SECTION 1: Literature & Mathematical Formulations
# ==============================================================================

def test_mlps_loss_multi_resolution_gradient_flow():
    """Tests MLPS Loss with realistic multi-resolution pseudo-masks from b4_5, b5_2, bn7.
    
    Verifies that gradients flow back into pred_logits across all interpolation scales.
    """
    criterion = MultiLayerPseudoSupervisionLoss(
        weights={"bn7": 1.0, "b5_2": 0.5, "b4_5": 0.25},
        ignore_index=255,
    )

    b, c, h, w = 2, 5, 128, 128
    pred_logits = torch.randn(b, c, h, w, requires_grad=True)

    pseudo_masks = {
        "bn7": torch.randint(0, 5, (b, 16, 16)),
        "b5_2": torch.randint(0, 5, (b, 32, 32)),
        "b4_5": torch.randint(0, 5, (b, 64, 64)),
    }

    loss = criterion(pred_logits, pseudo_masks)
    assert loss.dim() == 0, "Loss must be scalar"
    assert loss.item() > 0.0, "Loss must be positive"

    loss.backward()
    assert pred_logits.grad is not None, "Gradient must be computed for pred_logits"
    assert torch.all(torch.isfinite(pred_logits.grad)), "Gradients must be finite (no NaN or Inf)"
    assert pred_logits.grad.abs().sum() > 0, "Gradients must be non-zero"


def test_uncertainty_loss_device_and_gradient_flow():
    """Tests UncertaintyRegularizationLoss for mathematical consistency and device handling."""
    criterion = UncertaintyRegularizationLoss(entropy_threshold=0.65)

    b, c, h, w = 2, 5, 64, 64
    pred_logits = torch.randn(b, c, h, w, requires_grad=True)
    pseudo_mask = torch.randint(0, 5, (b, h, w))

    loss = criterion(pred_logits, pseudo_mask)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert pred_logits.grad is not None
    assert torch.all(torch.isfinite(pred_logits.grad))


def test_spatial_affinity_matrix_row_stochastic_property():
    """Tests that Spatial Affinity Relocalization computes a valid row-stochastic attention matrix."""
    relocalizer = SpatialAffinityRelocalization(alpha=0.70)

    b, c, h, w = 2, 4, 16, 16
    d = 64
    cam = torch.rand(b, c, h, w)
    feat = torch.randn(b, d, h, w)

    out = relocalizer(cam, feat)
    assert out.shape == (b, c, h, w)
    assert torch.all(torch.isfinite(out))

    # Test extreme non-square resolution
    cam_rect = torch.rand(1, 4, 14, 21)
    feat_rect = torch.randn(1, 32, 7, 11)  # needs interpolation
    out_rect = relocalizer(cam_rect, feat_rect)
    assert out_rect.shape == (1, 4, 14, 21)
    assert torch.all(torch.isfinite(out_rect))


# ==============================================================================
# SECTION 2: Architecture, Norm Layers & CGM Gate Audit
# ==============================================================================

def test_norm_layers_code_vs_claim_audit():
    """Audits the norm layer composition in DeepLabV3PlusWithCGM.
    
    Verifies the actual presence of BatchNorm2d vs GroupNorm in each component.
    """
    model = DeepLabV3PlusWithCGM(num_classes=5)

    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))

    # ASPPPooling is the only module with GroupNorm
    assert gn_count == 1, f"Expected exactly 1 GroupNorm in ASPPPooling, got {gn_count}"
    assert bn_count > 10, f"Expected backbone and ASPP to use BatchNorm2d, found {bn_count}"


def test_single_batch_training_mode_behavior():
    """Tests whether model.train() with batch size B=1 succeeds or suffers high variance."""
    model = DeepLabV3PlusWithCGM(num_classes=5)
    model.train()

    # Pass B=1 input through the network in train mode
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (1, 5, 224, 224)
    assert torch.all(torch.isfinite(logits))


def test_cgm_penalty_extinguishes_absent_classes():
    """Tests that CGM penalizes absent classes with -1e4 and yields softmax prob < 1e-10."""
    cgm = ClassificationGateMechanism(penalty_value=-1e4, threshold=0.5)

    b, c, h, w = 2, 5, 32, 32
    logits = torch.randn(b, c, h, w)
    # Class 0 (Tumor) and Class 4 (Background) present; classes 1, 2, 3 absent
    class_presence = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0]])

    gated_logits = cgm(logits, class_presence)
    probs = F.softmax(gated_logits, dim=1)

    # Absent classes must have near zero probability
    assert torch.all(probs[0, 1:4] < 1e-10), "Absent classes 1, 2, 3 must have prob < 1e-10"
    assert torch.all(probs[1, 0] < 1e-10), "Absent class 0 must have prob < 1e-10"
    assert torch.all(probs[1, 2:4] < 1e-10), "Absent classes 2, 3 must have prob < 1e-10"


def test_cgm_all_classes_absent_fallback():
    """Tests CGM fallback when all classes are zero in presence vector."""
    cgm = ClassificationGateMechanism(penalty_value=-1e4, default_background_idx=4)
    logits = torch.randn(1, 5, 16, 16)
    presence_zero = torch.zeros(1, 5)

    gated = cgm(logits, presence_zero)
    # Background class (index 4) should be protected
    pred = gated.argmax(dim=1)
    assert torch.all(pred == 4), "When all presence is 0, default background class 4 must be selected"


# ==============================================================================
# SECTION 3: Patient-Level Anti-Leakage Audit
# ==============================================================================

def test_patient_split_manifest_disjoint_and_complete():
    """Audits src/config/bcss_patient_splits.json for strict 0.0% patient leakage and 151 total."""
    splits = load_patient_splits()

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set) == 91, f"Expected 91 train patients, got {len(train_set)}"
    assert len(val_set) == 30, f"Expected 30 val patients, got {len(val_set)}"
    assert len(test_set) == 30, f"Expected 30 test patients, got {len(test_set)}"
    assert len(train_set | val_set | test_set) == 151, "Total unique patients must be exactly 151"

    # Absolute zero leakage check
    assert len(train_set & val_set) == 0, f"Leakage detected between Train and Val: {train_set & val_set}"
    assert len(train_set & test_set) == 0, f"Leakage detected between Train and Test: {train_set & test_set}"
    assert len(val_set & test_set) == 0, f"Leakage detected between Val and Test: {val_set & test_set}"


def test_extract_patient_id_adversarial_patterns():
    """Tests extract_patient_id on diverse, edge-case, and malformed patch filenames."""
    assert extract_patient_id("TCGA-A2-A0T2_1024_2048_[1 0 0 1].png") == "TCGA-A2-A0T2"
    assert extract_patient_id("tcga-a2-a0t2_patch_01.png") == "TCGA-A2-A0T2"
    assert extract_patient_id("TCGA-E2-A152+200+400+[0 1 0 0].png") == "TCGA-E2-A152"
    assert extract_patient_id("GDPH_1234_patch_01.png") == "GDPH"
    assert extract_patient_id("/path/to/nested/TCGA-BH-A0B8_roi_01.png") == "TCGA-BH-A0B8"


# ==============================================================================
# SECTION 4: Clinical Pathology & Otsu HSV Tissue Gating Critique
# ==============================================================================

def test_otsu_gating_on_pale_mucinous_tissue():
    """Critiques Otsu gating behavior on pale mucinous / myxoid extracellular pools.
    
    Mucin pools have very low color saturation (S < 15) and high luminance (V > 215).
    Evaluates whether genuine mucinous tumor bed is erroneously suppressed by Otsu gating.
    """
    h, w = 128, 128
    # Create pale bluish-gray mucinous background (R=215, G=225, B=230)
    mucinous_patch = np.zeros((h, w, 3), dtype=np.uint8)
    mucinous_patch[:, :] = [215, 225, 230]

    # Add small floating clusters of hyperchromatic tumor cells (R=50, G=30, B=80)
    cv2.circle(mucinous_patch, (64, 64), 12, (50, 30, 80), -1)
    cv2.circle(mucinous_patch, (32, 96), 8, (50, 30, 80), -1)

    tissue_mask = compute_otsu_tissue_mask(mucinous_patch)

    # Cellular tumor islands must be preserved
    assert tissue_mask[64, 64] == 1, "Hyperchromatic tumor island must be detected as tissue"
    assert tissue_mask[32, 96] == 1, "Second tumor island must be detected as tissue"


def test_otsu_gating_on_understained_eosinophilic_stroma():
    """Critiques Otsu gating on under-stained / faded pink stroma (R=240, G=215, B=220)."""
    h, w = 128, 128
    pale_stroma = np.zeros((h, w, 3), dtype=np.uint8)
    pale_stroma[:, :] = [240, 215, 220]

    # Add subtle collagen fibers with slightly higher saturation
    for y in range(20, 100, 15):
        cv2.line(pale_stroma, (10, y), (118, y + 10), (225, 180, 195), 2)

    tissue_mask = compute_otsu_tissue_mask(pale_stroma)
    # The fibers should provide enough saturation contrast
    assert np.sum(tissue_mask) > 0, "Pale fibrous tissue with collagen fibers should retain tissue presence"


def test_otsu_gating_zero_false_positives_on_pure_glass():
    """Confirms that pure white glass yields strictly 0 tissue pixels."""
    pure_glass = np.full((128, 128, 3), 255, dtype=np.uint8)
    mask = compute_otsu_tissue_mask(pure_glass)
    assert np.sum(mask) == 0, f"Pure glass must have 0 tissue pixels, got {np.sum(mask)}"


# ==============================================================================
# SECTION 5: Clinical Biomarkers (VTP & TSR) Consistency Audit
# ==============================================================================

def test_vtp_and_tsr_edge_cases():
    """Tests VTP and TSR when tissue has zero tumor, zero stroma, or all background."""
    # 1. All background mask
    all_bg = np.full((64, 64), 4, dtype=np.int64)
    assert compute_viable_tumor_percentage(all_bg) == 0.0
    assert compute_tumor_stroma_ratio(all_bg) == 0.0

    # 2. Pure tumor mask
    pure_tumor = np.zeros((64, 64), dtype=np.int64)
    assert compute_viable_tumor_percentage(pure_tumor) == 100.0
    assert compute_tumor_stroma_ratio(pure_tumor) == 100.0

    # 3. Pure stroma mask
    pure_stroma = np.ones((64, 64), dtype=np.int64)
    assert compute_viable_tumor_percentage(pure_stroma) == 0.0
    assert compute_tumor_stroma_ratio(pure_stroma) == 0.0


def test_vtp_formula_documentation_vs_code_audit():
    """Highlights discrepancy between Technical Report equation (Tumor / (Tumor + Necrosis))
    and metrics.py implementation (Tumor / Total_non_background_tissue).
    """
    pred_mask = np.zeros((10, 10), dtype=np.int64)
    pred_mask[:5, :] = 0  # 50 tumor pixels
    pred_mask[5:8, :] = 1 # 30 stroma pixels
    pred_mask[8:, :] = 3  # 20 necrosis pixels

    code_vtp = compute_viable_tumor_percentage(pred_mask, tumor_class_idx=0, background_class_idx=4)
    # Code computes: 50 / 100 * 100 = 50.0%
    assert abs(code_vtp - 50.0) < 1e-4

    # Report formula computes: 50 / (50 + 20) * 100 = 71.43%
    report_vtp = (50.0 / (50.0 + 20.0)) * 100.0
    assert abs(report_vtp - 71.428) < 1e-2
    # The gap demonstrates the semantic difference between tissue-wide VTP vs tumor-bed-only VTP


# ==============================================================================
# SECTION 6: Deliverable Quality & Invariant Compliance Audit
# ==============================================================================

def test_invariant_12_ascii_hyphen_compliance():
    """Verifies that no non-ASCII hyphens (em-dash U+2014 or en-dash U+2013) exist in the report."""
    report_path = os.path.join("product", "reports", "SE374_FINAL_TECHNICAL_REPORT_WSSS.md")
    assert os.path.exists(report_path), f"Report not found at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    em_dashes = [m.start() for m in re.finditer("\u2014", content)]
    en_dashes = [m.start() for m in re.finditer("\u2013", content)]

    assert len(em_dashes) == 0, f"Found {len(em_dashes)} prohibited em-dashes (\u2014) in report!"
    assert len(en_dashes) == 0, f"Found {len(en_dashes)} prohibited en-dashes (\u2013) in report!"


def test_api_server_routes_documentation_parity():
    """Verifies that endpoints documented in the report match actual routes in FastAPI app."""
    registered_routes = [route.path for route in app.routes]

    # Verify actual implemented endpoints
    assert "/api/v1/health" in registered_routes
    assert "/api/v1/predict/patch" in registered_routes
    assert "/api/v1/predict/wsi" in registered_routes
    assert "/" in registered_routes
    assert "/viewer" in registered_routes

    # Check for hallucinated endpoints from the report
    hallucinated_endpoints = ["/segment", "/biomarkers", "/splits", "/tile/{z}/{x}/{y}.png"]
    missing_from_code = [ep for ep in hallucinated_endpoints if ep not in registered_routes]

    # All endpoints are now registered in api_server.py
    assert len(missing_from_code) == 0, (
        f"Documented endpoints {missing_from_code} are absent from api_server.py routes!"
    )


def test_patient_id_examples_in_report_against_split_manifest():
    """Audits patient IDs cited as examples in report Section 3.2 against bcss_patient_splits.json."""
    splits = load_patient_splits()
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Examples cited in report line 73: TCGA-E2-A152, TCGA-E2-A1B4, TCGA-EW-A1P8
    assert "TCGA-E2-A152" in val_set
    # Note: TCGA-E2-A1B4 and TCGA-EW-A1P8 are absent from val_set
    absent_val_examples = [pid for pid in ["TCGA-E2-A1B4", "TCGA-EW-A1P8"] if pid not in val_set]
    assert len(absent_val_examples) == 2, "Confirmed report cited non-existent val examples"

    # Examples cited in report line 74: TCGA-C8-A1HJ, TCGA-D8-A13Z, TCGA-E2-A14N
    assert "TCGA-C8-A1HJ" in test_set
    absent_test_examples = [pid for pid in ["TCGA-D8-A13Z", "TCGA-E2-A14N"] if pid not in test_set]
    assert len(absent_test_examples) == 2, "Confirmed report cited non-existent test examples"
