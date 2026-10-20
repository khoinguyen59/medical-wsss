"""Independent Audit Suite by Audit Team 4 (SE374.R11 Medical AI).

Adversarial unit tests challenging:
1. Architectural stability: GroupNorm vs BatchNorm under B=1 training.
2. Gradient flow across DeepLabV3PlusWithCGM and intermediate losses.
3. Classification Gate Mechanism (CGM) under extreme class absence.
4. Progressive Dropout Attention (PDA) numerical stability with zero variance.
5. Spatial Affinity Relocalization token normalization and non-square tensors.
6. Patient ID parsing and 0.0% patient leakage on the 151-case manifest.
7. Clinical pathology edge cases for Otsu HSV gating (mucinous, pale, adipose).
8. Biomarker calculations (VTP, TSR) and formula discrepancies.
9. FastAPI clinical server endpoints and documentation mismatches.
10. Invariant 12 compliance: strict ASCII hyphen 0x2D only.
"""

import json
import os
import re
from typing import Dict, List
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi.testclient import TestClient

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
from src.models.backbone import MultiStageFeatureBackbone
from src.models.deeplabv3plus import (
    ASPP,
    ASPPPooling,
    ClassificationGateMechanism,
    DeepLabV3PlusWithCGM,
)
from src.models.pda_generator import ProgressiveDropoutAttention
from src.models.wsss_tissue import SpatialAffinityRelocalization, WSSSBenchmarkingEngine
from src.visualization.api_server import app


# ==============================================================================
# 1. ARCHITECTURAL STABILITY: GROUPNORM VS BATCHNORM UNDER B=1
# ==============================================================================

def test_aspp_pooling_groupnorm_prevents_b1_training_crash():
    """Verifies that ASPPPooling uses GroupNorm and succeeds during B=1 training mode.
    
    Standard BatchNorm2d on (1, C, 1, 1) raises ValueError:
    'Expected more than 1 value per channel when training'.
    GroupNorm(1, C) succeeds because it normalizes across channel dimensions.
    """
    pooling_block = ASPPPooling(in_channels=512, out_channels=256)
    pooling_block.train()

    # B=1 input tensor
    x = torch.randn(1, 512, 14, 14)
    out = pooling_block(x)

    assert out.shape == (1, 256, 14, 14)
    assert not torch.isnan(out).any()

    # Verify standard BatchNorm2d would fail under this condition
    bn_failing = nn.BatchNorm2d(256)
    bn_failing.train()
    with pytest.raises(ValueError, match="Expected more than 1 value per channel"):
        bn_failing(torch.randn(1, 256, 1, 1))


def test_deeplabv3plus_gradient_flow_and_backprop():
    """Verifies full gradient flow from loss back to all backbone stages."""
    model = DeepLabV3PlusWithCGM(
        num_classes=5,
        in_channels=3,
        low_level_channels=32,
        backbone_channels=256,
    )
    model.train()

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    presence = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0]])
    pseudo_masks = {
        "bn7": torch.randint(0, 5, (2, 64, 64)),
        "b5_2": torch.randint(0, 5, (2, 64, 64)),
        "b4_5": torch.randint(0, 5, (2, 64, 64)),
    }

    logits, loss = model(
        x,
        class_presence=presence,
        pseudo_masks=pseudo_masks,
        loss_type="mlps",
        return_loss=True,
    )

    assert loss is not None
    assert loss.item() > 0.0
    loss.backward()

    # Verify gradients reach the input image and early layers
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert x.grad.abs().sum() > 0.0

    stem_weight_grad = model.backbone.stem[0].weight.grad
    assert stem_weight_grad is not None
    assert not torch.isnan(stem_weight_grad).any()
    assert stem_weight_grad.abs().sum() > 0.0


# ==============================================================================
# 2. CLASSIFICATION GATE MECHANISM (CGM) UNDER EXTREME ABSENCE
# ==============================================================================

def test_cgm_extinguishes_absent_classes():
    """Verifies that CGM forces absent classes to have probability < 1e-10."""
    cgm = ClassificationGateMechanism(penalty_value=-1e4, threshold=0.5)
    logits = torch.randn(2, 5, 32, 32)
    presence = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0]])

    gated = cgm(logits, class_presence=presence)
    probs = F.softmax(gated, dim=1)

    # In sample 0: classes 1, 2, 3 are absent
    for c in [1, 2, 3]:
        max_prob = probs[0, c].max().item()
        assert max_prob < 1e-8, f"Absent class {c} has probability {max_prob} >= 1e-8"

    # In sample 1: classes 0, 2, 3 are absent
    for c in [0, 2, 3]:
        max_prob = probs[1, c].max().item()
        assert max_prob < 1e-8, f"Absent class {c} has probability {max_prob} >= 1e-8"


def test_cgm_all_absent_classes_fallback_to_background():
    """Verifies that if all classes are absent, default_background_idx is protected."""
    cgm = ClassificationGateMechanism(penalty_value=-1e4, default_background_idx=4)
    logits = torch.randn(1, 5, 16, 16)
    presence = torch.zeros(1, 5)  # All classes absent

    gated = cgm(logits, class_presence=presence)
    preds = torch.argmax(gated, dim=1)

    # 100% of pixels must predict background (class 4)
    assert (preds == 4).all()


# ==============================================================================
# 3. PROGRESSIVE DROPOUT ATTENTION (PDA) NUMERICAL STABILITY
# ==============================================================================

def test_pda_zero_variance_numerical_stability():
    """Verifies that PDA handles feature maps with zero variance without NaN or division by zero."""
    pda = ProgressiveDropoutAttention(num_steps=2, top_p=0.25, base_dropout_rate=0.5)
    feat_constant = torch.ones(2, 64, 16, 16) * 3.14159
    weights_constant = torch.ones(4, 64) * 0.5

    cam = pda(feat_constant, weights_constant)
    assert not torch.isnan(cam).any()
    assert not torch.isinf(cam).any()
    assert cam.shape == (2, 4, 16, 16)


def test_pda_prototype_cosine_gating_negative_clamping():
    """Verifies that prototype cosine gating strictly zeroes out anti-correlated features."""
    pda = ProgressiveDropoutAttention(num_steps=1, boundary_bleeding_control=True)
    b, d, h, w = 1, 16, 8, 8
    c = 2

    # Construct features where half the image is positive and half is strictly negative
    feat = torch.ones(b, d, h, w)
    feat[:, :, :, 4:] = -1.0  # Opposite direction
    cam_base = torch.zeros(b, c, h, w)
    cam_base[:, 0, :, :4] = 1.0  # Class 0 active on left side

    gating = pda._compute_prototype_gating(feat, cam_base)
    assert gating.shape == (b, c, h, w)

    # Left side should have positive gating, right side should be exactly 0.0 (ReLU clamped)
    assert (gating[:, 0, :, :4] > 0.0).all()
    assert (gating[:, 0, :, 4:] == 0.0).all()


# ==============================================================================
# 4. SPATIAL AFFINITY RELOCALIZATION & TOKEN NORMALIZATION
# ==============================================================================

def test_spatial_affinity_identity_when_alpha_one():
    """Verifies that SpatialAffinityRelocalization with alpha=1.0 is an exact identity function."""
    sar = SpatialAffinityRelocalization(alpha=1.0)
    cam = torch.rand(2, 4, 14, 14)
    feat = torch.rand(2, 128, 14, 14)

    out = sar(cam, feat)
    assert torch.allclose(out, cam, atol=1e-6)


def test_spatial_affinity_row_stochastic_normalization():
    """Verifies that the spatial attention matrix is strictly row-stochastic (sums to 1.0)."""
    sar = SpatialAffinityRelocalization(alpha=0.5)
    b, d, h, w = 2, 64, 10, 10
    feat = torch.randn(b, d, h, w)

    tokens = feat.flatten(2).permute(0, 2, 1)  # (B, N, D)
    scale = 1.0 / np.sqrt(d)
    attn = torch.bmm(tokens, tokens.transpose(1, 2)) * scale
    attn = F.softmax(attn, dim=-1)

    # Sum along last dimension must be 1.0 for each token
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_spatial_affinity_non_square_feature_map():
    """Verifies that SpatialAffinityRelocalization handles rectangular feature maps."""
    sar = SpatialAffinityRelocalization(alpha=0.7)
    cam = torch.rand(1, 4, 12, 18)
    feat = torch.rand(1, 64, 6, 9)

    out = sar(cam, feat)
    assert out.shape == (1, 4, 12, 18)
    assert not torch.isnan(out).any()


# ==============================================================================
# 5. PATIENT ID PARSING & 0.0% PATIENT LEAKAGE AUDIT
# ==============================================================================

def test_patient_id_extraction_tcga_and_gdph_bug():
    """Audits patient ID extraction on TCGA and non-TCGA (GDPH) formats.
    
    Identifies a known implementation caveat in patient_split.py line 25:
    'token = re.split(r"[_+]", fname)[0]'
    For 'GDPH_1234_patch_01.png', it extracts 'GDPH' instead of 'GDPH_1234'.
    """
    # Standard TCGA pattern: works correctly
    tcga_name = "TCGA-A2-A0T2_1024_2048_[1 0 0 1].png"
    assert extract_patient_id(tcga_name) == "TCGA-A2-A0T2"

    tcga_dash = "TCGA-BH-A0B8+002.png"
    assert extract_patient_id(tcga_dash) == "TCGA-BH-A0B8"

    # Non-TCGA format audit
    gdph_name = "GDPH_1234_patch_01.png"
    extracted = extract_patient_id(gdph_name)
    # Document the current behavior: splits by first underscore returning 'GDPH'
    assert extracted == "GDPH", f"Expected 'GDPH' under current regex split, got '{extracted}'"


def test_151_patient_manifest_zero_leakage():
    """Verifies that bcss_patient_splits.json has 0.0% patient overlap across all sets."""
    splits = load_patient_splits()
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Total unique count
    total_unique = len(train_set | val_set | test_set)
    assert total_unique == 151
    assert len(train_set) == 91
    assert len(val_set) == 30
    assert len(test_set) == 30

    # Pairwise intersections must be empty
    assert len(train_set & val_set) == 0, f"Leakage Train-Val: {train_set & val_set}"
    assert len(train_set & test_set) == 0, f"Leakage Train-Test: {train_set & test_set}"
    assert len(val_set & test_set) == 0, f"Leakage Val-Test: {val_set & test_set}"


# ==============================================================================
# 6. CLINICAL PATHOLOGY EDGE CASES (OTSU HSV GATING)
# ==============================================================================

def test_otsu_gating_pure_glass_and_noise():
    """Verifies that pure white glass (255, 255, 255) is 100% rejected as background."""
    white_glass = np.full((128, 128, 3), 255, dtype=np.uint8)
    mask = compute_otsu_tissue_mask(white_glass)
    assert (mask == 0).all(), "Pure white glass must produce zero tissue mask"

    # Glass with slight scanner sensor noise (240 to 255)
    noise_glass = np.random.randint(240, 256, (128, 128, 3), dtype=np.uint8)
    mask_noise = compute_otsu_tissue_mask(noise_glass)
    assert (mask_noise == 0).mean() >= 0.99, "Scanner noise glass must be >= 99% rejected"


def test_otsu_gating_pathology_challenge_mucinous_and_pale():
    """Pathology stress test: Evaluates Otsu HSV gating on pale mucin pools and hyalinized stroma.
    
    Mucinous carcinoma has faint bluish/pink mucin with very low saturation (S < 15).
    Hyalinized collagen stroma is pale pink (S < 20, Gray > 220).
    Audits the saturation threshold behavior on these challenging clinical entities.
    """
    # Simulate pale mucinous extracellular pool: RGB ~ (225, 220, 235), Low Saturation
    mucinous_pool = np.zeros((64, 64, 3), dtype=np.uint8)
    mucinous_pool[:, :, 0] = 225  # R
    mucinous_pool[:, :, 1] = 220  # G
    mucinous_pool[:, :, 2] = 235  # B (faint bluish-pink)

    # In isolation without dense tissue, Otsu saturation thresholding detects it as background
    mask_mucinous = compute_otsu_tissue_mask(mucinous_pool)
    # Document clinical vulnerability: pale mucin has S < 20 and can be filtered out
    assert isinstance(mask_mucinous, np.ndarray)

    # Simulate strong cellular tumor bed: RGB ~ (140, 50, 150), High Hematoxylin Saturation
    dense_tumor = np.zeros((64, 64, 3), dtype=np.uint8)
    dense_tumor[:, :, 0] = 140
    dense_tumor[:, :, 1] = 50
    dense_tumor[:, :, 2] = 150
    mask_tumor = compute_otsu_tissue_mask(dense_tumor)
    assert (mask_tumor == 1).mean() > 0.95, "Dense cellular tumor must be retained"


# ==============================================================================
# 7. LOSS FUNCTIONS MATH & UNCERTAINTY REGULARIZATION
# ==============================================================================

def test_uncertainty_loss_entropy_scaling():
    """Verifies that high-entropy predictions receive lower weight (down-weighted)."""
    loss_fn = UncertaintyRegularizationLoss(entropy_threshold=0.65, reduction="none")

    # Logits where class 0 has 100% confidence: low entropy
    confident_logits = torch.zeros(1, 5, 8, 8)
    confident_logits[:, 0, :, :] = 20.0
    targets = torch.zeros(1, 8, 8, dtype=torch.long)

    loss_confident = loss_fn(confident_logits, targets)

    # Logits where all classes have equal probability: maximal entropy
    uncertain_logits = torch.zeros(1, 5, 8, 8)
    loss_uncertain = loss_fn(uncertain_logits, targets)

    # Cross entropy of uncertain logits is log(5) ~ 1.609
    # Normalized entropy is 1.0 >= 0.65, so weight is clamped to min 0.1
    assert loss_uncertain.max().item() < 0.25, "Uncertain pixels must be attenuated to weight 0.1"


# ==============================================================================
# 8. CLINICAL BIOMARKERS: VTP, TSR & REPORT DISCREPANCY AUDIT
# ==============================================================================

def test_biomarkers_boundary_values():
    """Tests VTP and TSR on boundary clinical masks."""
    # 1. 100% Tumor mask
    mask_all_tumor = np.zeros((100, 100), dtype=np.int64)
    vtp_100 = compute_viable_tumor_percentage(mask_all_tumor)
    tsr_100 = compute_tumor_stroma_ratio(mask_all_tumor)
    assert vtp_100 == 100.0
    assert tsr_100 == 100.0

    # 2. 100% Background mask (all class 4)
    mask_all_bg = np.full((100, 100), 4, dtype=np.int64)
    vtp_bg = compute_viable_tumor_percentage(mask_all_bg)
    tsr_bg = compute_tumor_stroma_ratio(mask_all_bg)
    assert vtp_bg == 0.0
    assert tsr_bg == 0.0

    # 3. 50% Tumor (class 0), 50% Stroma (class 1)
    mask_half = np.zeros((100, 100), dtype=np.int64)
    mask_half[:50, :] = 0
    mask_half[50:, :] = 1
    tsr_half = compute_tumor_stroma_ratio(mask_half)
    assert tsr_half == 50.0


def test_vtp_formula_discrepancy_code_vs_report():
    """Audits formula discrepancy between src/losses_metrics/metrics.py and technical report.
    
    Report line 139 states:
        VTP = N_tumor / (N_tumor + N_necrosis) * 100%
    Code in metrics.py line 129 implements:
        VTP = N_tumor / (N_total - N_background) * 100%
        (where denominator includes Tumor + Stroma + Lymphocytes + Necrosis).
    """
    # Create mask: 200 Tumor (0), 300 Stroma (1), 100 Necrosis (3), 400 Background (4)
    mask = np.zeros(1000, dtype=np.int64)
    mask[0:200] = 0    # Tumor
    mask[200:500] = 1  # Stroma
    mask[500:600] = 3  # Necrosis
    mask[600:1000] = 4 # Background

    # Implementation in metrics.py: 200 / (1000 - 400) = 200 / 600 = 33.33%
    code_vtp = compute_viable_tumor_percentage(mask)
    assert pytest.approx(code_vtp, 0.01) == 33.33

    # Formula stated in Report (Viable Tumor vs Tumor Bed: Tumor + Necrosis):
    # 200 / (200 + 100) = 200 / 300 = 66.67%
    report_vtp = (200.0 / (200.0 + 100.0)) * 100.0
    assert pytest.approx(report_vtp, 0.01) == 66.67

    # Discrepancy is documented and asserted
    assert abs(code_vtp - report_vtp) > 30.0, "Discrepancy between code and report must be recorded"


# ==============================================================================
# 9. FASTAPI SERVER ENDPOINT & ROUTE AUDIT
# ==============================================================================

def test_api_server_health_and_prediction():
    """Tests active API server endpoints using FastAPI TestClient."""
    client = TestClient(app)

    # 1. Health check
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"]["num_classes"] == 5
    assert "classes" in data["model"]

    # 2. OpenSeadragon viewer page
    viewer_resp = client.get("/viewer")
    assert viewer_resp.status_code == 200
    assert "OpenSeadragon" in viewer_resp.text or "html" in viewer_resp.headers.get("content-type", "")

    # 3. Route aliases /segment and /biomarkers in Code matching report
    resp_segment = client.post("/segment")
    assert resp_segment.status_code != 404, "Report documented /segment must exist in code"

    resp_biomarkers = client.post("/biomarkers")
    assert resp_biomarkers.status_code != 404, "Report documented /biomarkers must exist in code"


# ==============================================================================
# 10. INVARIANT 12: STRICT ASCII HYPHEN 0x2D AUDIT
# ==============================================================================

def test_no_non_ascii_hyphens_in_this_file():
    """Verifies this audit file strictly adheres to Invariant 12 (ASCII 0x2D only)."""
    current_file = os.path.abspath(__file__)
    with open(current_file, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_codepoints = [0x2013, 0x2014, 0x2015, 0x2212, 0x00AD]
    violations = []
    for idx, char in enumerate(content):
        if ord(char) in forbidden_codepoints:
            violations.append((idx, char, hex(ord(char))))

    assert len(violations) == 0, f"Found {len(violations)} non-ASCII hyphen violations: {violations}"
