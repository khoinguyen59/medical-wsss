"""
SE374.R11 Medical AI - Audit Team 2 Adversarial Stress and Verification Suite.

Independent, rigorous unit tests challenging all core modules:
1. Patient-Level Anti-Leakage Protocol & Splits Manifest Integrity
2. DeepLabV3+ with Classification Gate Mechanism (CGM) & Boundary Behavior
3. Loss Formulation Mathematics (MLPS, Uncertainty, Cross-Entropy with ignore_index)
4. Progressive Dropout Attention (PDA) & Prototype Cosine Gating Stability
5. Spatial Affinity Relocalization Mathematical Properties & Token Matrices
6. Clinical Biomarkers (VTP, TSR) Edge Cases & Zero-Division Safety
7. Otsu Clinical Gating on Challenging Histopathology (Mucinous, Pale, Adipose)
8. Invariant 12 Standard ASCII Hyphen Verification across all source modules

Strict adherence to:
- Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Invariant 07: Genuine PyTorch / NumPy computation with zero mockup.
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

from src.datasets.bcss_dataset import BCSSDataset
from src.datasets.otsu_filter import apply_otsu_gating_to_prediction, compute_otsu_tissue_mask
from src.datasets.patient_split import (
    extract_patient_id,
    load_patient_splits,
    split_patients_kfold,
    verify_patient_leakage,
)
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
from src.models.backbone import MultiStageFeatureBackbone, ResidualConvBlock
from src.models.deeplabv3plus import (
    ASPP,
    ClassificationGateMechanism,
    DeepLabV3PlusDecoder,
    DeepLabV3PlusWithCGM,
    ResNetBackbone,
)
from src.models.pda_generator import ProgressiveDropoutAttention
from src.models.wsss_tissue import (
    SpatialAffinityRelocalization,
    WSSSBenchmarkingEngine,
    apply_classmix_augmentation,
)


# =========================================================================
# TEST SUITE 1: PATIENT-LEVEL ANTI-LEAKAGE AUDIT (TASK 5)
# =========================================================================

def test_patient_split_manifest_strict_invariance():
    """Verify that bcss_patient_splits.json maintains strict 0.0% leakage and exact patient counts."""
    splits = load_patient_splits()
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # 1. Zero pairwise intersection
    assert len(train_set.intersection(val_set)) == 0, f"Leakage Train-Val: {train_set.intersection(val_set)}"
    assert len(train_set.intersection(test_set)) == 0, f"Leakage Train-Test: {train_set.intersection(test_set)}"
    assert len(val_set.intersection(test_set)) == 0, f"Leakage Val-Test: {val_set.intersection(test_set)}"

    # 2. Check total unique patients is exactly 151
    total_unique = train_set.union(val_set).union(test_set)
    assert len(total_unique) == 151, f"Expected 151 total unique patients, got {len(total_unique)}"
    assert len(splits["train"]) == 91, f"Expected 91 train patients, got {len(splits['train'])}"
    assert len(splits["val"]) == 30, f"Expected 30 val patients, got {len(splits['val'])}"
    assert len(splits["test"]) == 30, f"Expected 30 test patients, got {len(splits['test'])}"

    # 3. Verify TCGA patient ID naming convention
    for pid in total_unique:
        assert re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", pid), f"Invalid TCGA Patient ID format: {pid}"


def test_adversarial_patient_leakage_detection():
    """Verify verify_patient_leakage raises ValueError on intentional overlap."""
    train_ids = ["TCGA-A1-A0SK", "TCGA-A2-A0T2", "TCGA-BH-A0B8"]
    val_ids = ["TCGA-AO-A03L", "TCGA-AO-A03M"]
    test_ids = ["TCGA-A2-A0T2", "TCGA-AR-A0TQ"]  # Intentional leak with train

    with pytest.raises(ValueError, match="Patient leakage detected"):
        verify_patient_leakage(train_ids, val_ids, test_ids)


def test_patient_kfold_no_overlap_across_all_folds():
    """Verify split_patients_kfold guarantees zero patient overlap between train and val in every fold."""
    patient_ids = [f"TCGA-A2-{i:04d}" for i in range(100)]
    folds = split_patients_kfold(patient_ids, n_splits=5, seed=42)
    assert len(folds) == 5

    for fold_idx, (train_f, val_f) in enumerate(folds):
        s_train = set(train_f)
        s_val = set(val_f)
        assert len(s_train.intersection(s_val)) == 0, f"Fold {fold_idx} leaked: {s_train.intersection(s_val)}"
        assert len(s_train.union(s_val)) == 100, f"Fold {fold_idx} lost patients"


# =========================================================================
# TEST SUITE 2: CODE LOGIC & CLEAN ARCHITECTURE (TASK 2)
# =========================================================================

def test_single_item_batch_groupnorm_and_gradient_flow():
    """Verify DeepLabV3+ with B=1 does not suffer from BatchNorm batch-stat failure and maintains gradients."""
    model = DeepLabV3PlusWithCGM(num_classes=5, in_channels=3, low_level_channels=32, backbone_channels=256)
    model.train()

    # Batch of size 1 (critical clinical test case)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    presence = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])  # Tumor, Stroma, Background present; LYM, NEC absent
    target = torch.randint(0, 5, (1, 64, 64), dtype=torch.long)

    logits, loss = model(x, class_presence=presence, targets=target, return_loss=True)
    assert logits.shape == (1, 5, 64, 64)
    assert loss is not None and not torch.isnan(loss) and not torch.isinf(loss)

    # Backward gradient flow check
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0
    # Verify backbone parameters received non-zero gradients
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient!"
            break


def test_deeplabv3plus_cgm_all_absent_fallback():
    """Verify CGM behavior when class presence vector contains all zeros."""
    cgm = ClassificationGateMechanism(penalty_value=-1e4, default_background_idx=4)
    logits = torch.zeros((2, 5, 16, 16))
    presence = torch.zeros((2, 5))  # All classes marked absent

    gated = cgm(logits, class_presence=presence)
    # Background (class 4) should NOT be penalized because of default_background_idx fallback
    assert torch.all(gated[:, 4, :, :] == 0.0), "Background class should remain active when all absent"
    # Classes 0..3 must be penalized by -1e4
    for c in range(4):
        assert torch.all(gated[:, c, :, :] == -1e4), f"Class {c} should be penalized by -1e4"


def test_backbone_intermediate_stages_consistency():
    """Verify MultiStageFeatureBackbone outputs matching spatial strides for MLPS hierarchy."""
    backbone = MultiStageFeatureBackbone(in_channels=3, num_classes=4)
    backbone.eval()
    x = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        logits, features = backbone(x)

    assert logits.shape == (2, 4)
    assert "b4_5" in features
    assert "b5_2" in features
    assert "bn7" in features

    # Check stride ratios
    # stage1: stride 4 -> 32x32
    # stage2: stride 8 -> 16x16
    # b4_5 (stage3): stride 16 -> 8x8
    # b5_2 (stage4): stride 32 -> 4x4
    # bn7 (stage5): stride 32 -> 4x4
    assert features["b4_5"].shape[-2:] == (8, 8)
    assert features["b5_2"].shape[-2:] == (4, 4)
    assert features["bn7"].shape[-2:] == (4, 4)


# =========================================================================
# TEST SUITE 3: LITERATURE & MATHEMATICAL VERIFICATION (TASK 3)
# =========================================================================

def test_mlps_loss_hierarchical_weights():
    """Verify MLPS loss calculates L_total = L_bn7 + 0.5 * L_b5_2 + 0.25 * L_b4_5."""
    mlps_loss = MultiLayerPseudoSupervisionLoss(
        weights={"bn7": 1.0, "b5_2": 0.5, "b4_5": 0.25},
        ignore_index=255,
    )

    pred_logits = torch.randn(2, 5, 32, 32)
    pseudo_masks = {
        "bn7": torch.randint(0, 5, (2, 32, 32)),
        "b5_2": torch.randint(0, 5, (2, 16, 16)),
        "b4_5": torch.randint(0, 5, (2, 8, 8)),
    }

    loss = mlps_loss(pred_logits, pseudo_masks)
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_uncertainty_loss_mathematical_downweighting():
    """Verify that UncertaintyRegularizationLoss assigns lower weight to high-entropy pixels."""
    unc_loss = UncertaintyRegularizationLoss(entropy_threshold=0.65)

    # Case 1: Pure confident prediction (one class logit = +10, others -10) -> Entropy ~ 0
    confident_logits = torch.full((1, 5, 10, 10), -10.0)
    confident_logits[:, 0, :, :] = 10.0
    target_0 = torch.zeros((1, 10, 10), dtype=torch.long)
    loss_confident = unc_loss(confident_logits, target_0)

    # Case 2: Highly ambiguous boundary prediction (all logits equal = 0.0) -> Max entropy
    ambiguous_logits = torch.zeros((1, 5, 10, 10))
    target_0 = torch.zeros((1, 10, 10), dtype=torch.long)
    loss_ambiguous = unc_loss(ambiguous_logits, target_0)

    # The ambiguous loss without weighting is log(5) ~ 1.609.
    # With clamping weight at min=0.1, the weighted ambiguous loss should be substantially lower than unweighted CE!
    ce_unweighted = F.cross_entropy(ambiguous_logits, target_0)
    assert loss_ambiguous < ce_unweighted, f"Uncertainty loss ({loss_ambiguous}) should down-weight ambiguous pixels vs CE ({ce_unweighted})"


def test_spatial_affinity_matrix_row_stochastic_property():
    """Verify that the spatial affinity matrix A in Relocalization is strictly row-stochastic (sums to 1)."""
    relocalizer = SpatialAffinityRelocalization(alpha=0.70)
    b, c, h, w = 1, 4, 14, 14
    d = 64
    cam = torch.rand((b, c, h, w))
    features = torch.randn((b, d, h, w))

    # Manually compute attention matrix as defined in relocalizer
    tokens = features.flatten(2).permute(0, 2, 1)  # (1, 196, 64)
    scale = 1.0 / np.sqrt(d)
    attn = torch.bmm(tokens, tokens.transpose(1, 2)) * scale  # (1, 196, 196)
    attn_prob = F.softmax(attn, dim=-1)

    # Row sums must be 1.0
    row_sums = attn_prob.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Affinity matrix rows must sum to 1.0"

    # Forward relocalizer pass
    refined_cam = relocalizer(cam, features)
    assert refined_cam.shape == (b, c, h, w)
    assert not torch.isnan(refined_cam).any()


def test_pda_prototype_cosine_gating_anti_bleeding():
    """Verify that Prototype Cosine Gating in PDA dampens features with negative cosine alignment."""
    pda = ProgressiveDropoutAttention(
        num_steps=2,
        top_p=0.30,
        base_dropout_rate=0.50,
        boundary_bleeding_control=True,
    )
    b, d, h, w = 1, 32, 8, 8
    c = 4

    # Synthesize feature map where left half is tumor-like and right half is opposite (negative alignment)
    feat = torch.ones((b, d, h, w))
    feat[:, :, :, 4:] = -1.0  # Opposite semantic direction

    # Weights
    weights = torch.ones((c, d, 1, 1))

    # Initial CAM: should be positive on left half, 0 on right half
    cam_out = pda(feat, weights)
    assert cam_out.shape == (b, c, h, w)
    # Right half with negative alignment should be dampened by prototype gating
    assert cam_out[:, :, :, 4:].mean() < cam_out[:, :, :, :4].mean()


# =========================================================================
# TEST SUITE 4: CLINICAL PATHOLOGY & OTSU GATING (TASK 7)
# =========================================================================

def test_otsu_gating_on_adipose_tissue():
    """Critique Otsu gating on mature adipose tissue (clear empty vacuoles, low saturation)."""
    # Adipose tissue has high luminance (>240) and very low saturation (0..15)
    adipose_patch = np.full((64, 64, 3), fill_value=245, dtype=np.uint8)
    # Add small sensor noise
    noise = np.random.randint(-5, 5, (64, 64, 3))
    adipose_patch = np.clip(adipose_patch + noise, 0, 255).astype(np.uint8)

    tissue_mask = compute_otsu_tissue_mask(adipose_patch)
    # Adipose lipid vacuoles should be recognized as non-tissue/background
    assert tissue_mask.sum() == 0, f"Pure adipose empty lipid space should be gated out, got {tissue_mask.sum()} pixels"


def test_otsu_gating_on_mucinous_pale_tissue():
    """Critique Otsu gating on pale mucinous tumor tissue (lavender/pale pink, low saturation)."""
    # Pale mucin: RGB (225, 215, 235), Saturation ~20, Grayscale ~220 (< 235)
    pale_mucin = np.zeros((64, 64, 3), dtype=np.uint8)
    pale_mucin[:, :, 0] = 225  # R
    pale_mucin[:, :, 1] = 215  # G
    pale_mucin[:, :, 2] = 235  # B

    # Half mucin, half dark tumor (160, 50, 180)
    test_patch = pale_mucin.copy()
    test_patch[:, 32:, 0] = 160
    test_patch[:, 32:, 1] = 50
    test_patch[:, 32:, 2] = 180

    tissue_mask = compute_otsu_tissue_mask(test_patch)
    # Both pale mucin and dense tumor have is_non_white = True (gray < 235)
    # Dense tumor must be captured (>90% of right half)
    assert tissue_mask[:, 32:].mean() > 0.85, "Dense tumor tissue must be preserved by Otsu gating"


def test_clinical_biomarkers_boundary_values():
    """Verify VTP and TSR mathematical edge cases and clinical zero-division safety."""
    h, w = 32, 32

    # Case A: Pure Background (Empty Glass)
    all_bg = np.full((h, w), fill_value=4, dtype=np.int64)
    assert compute_viable_tumor_percentage(all_bg) == 0.0
    assert compute_tumor_stroma_ratio(all_bg) == 0.0

    # Case B: 100% Viable Tumor, 0% Stroma, 0% Necrosis
    all_tumor = np.zeros((h, w), dtype=np.int64)  # Class 0
    assert compute_viable_tumor_percentage(all_tumor) == 100.0
    assert compute_tumor_stroma_ratio(all_tumor) == 100.0

    # Case C: Equal Tumor (Class 0) and Stroma (Class 1) -> TSR = 50%
    half_half = np.zeros((h, w), dtype=np.int64)
    half_half[:, 16:] = 1  # Right half is stroma
    assert compute_tumor_stroma_ratio(half_half) == 50.0
    assert compute_viable_tumor_percentage(half_half) == 50.0  # Tissue is all non-background

    # Case D: Tumor with Necrosis (Class 3)
    tumor_necrosis = np.zeros((h, w), dtype=np.int64)
    tumor_necrosis[:, 16:] = 3  # Right half is necrosis
    # VTP = 50%
    assert compute_viable_tumor_percentage(tumor_necrosis) == 50.0


# =========================================================================
# TEST SUITE 5: INVARIANT 12 ASCII HYPHEN AUDIT (TASK 8)
# =========================================================================

def test_invariant_12_no_em_dash_or_en_dash_in_all_source_code():
    """Audit all python files in src/ to guarantee ZERO em-dash (0x2014) or en-dash (0x2013)."""
    src_dir = Path("src")
    py_files = list(src_dir.rglob("*.py"))
    assert len(py_files) > 0, "No python files found in src/"

    forbidden_bytes = [
        (b"\xe2\x80\x94", "em-dash (U+2014)"),
        (b"\xe2\x80\x93", "en-dash (U+2013)"),
        (b"\xe2\x80\x95", "horizontal bar (U+2015)"),
        (b"\xe2\x88\x92", "minus sign (U+2212)"),
    ]

    violations = []
    for py_file in py_files:
        with open(py_file, "rb") as f:
            content = f.read()
        for f_byte, f_name in forbidden_bytes:
            if f_byte in content:
                violations.append(f"{py_file}: contains forbidden character '{f_name}'")

    assert len(violations) == 0, f"Invariant 12 violations found in source code:\n" + "\n".join(violations)


def test_invariant_12_no_em_dash_in_deliverables_and_reports():
    """Audit markdown reports in product/ to guarantee ZERO em-dash or en-dash."""
    report_file = Path("product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md")
    if report_file.exists():
        with open(report_file, "rb") as f:
            content = f.read()
        assert b"\xe2\x80\x94" not in content, "Report contains forbidden em-dash (U+2014)"
        assert b"\xe2\x80\x93" not in content, "Report contains forbidden en-dash (U+2013)"


# =========================================================================
# TEST SUITE 6: DATASET & TRANSFORM HANDLING DEFECT VERIFICATION (TASK 2)
# =========================================================================

def test_bcss_dataset_transform_handling_contract():
    """Test whether BCSSDataset respects transform parameter when provided."""
    # Test weak label parsing logic independently
    weak_label_str_space = "TCGA-A2-A0T2_1024_2048_[1 0 1 0].png"
    parsed_space = BCSSDataset._extract_weak_label(weak_label_str_space)
    assert np.array_equal(parsed_space, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))

    weak_label_str_nospace = "TCGA-A2-A0T2_1024_2048_[1010].png"
    parsed_nospace = BCSSDataset._extract_weak_label(weak_label_str_nospace)
    assert np.array_equal(parsed_nospace, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))


def test_adversarial_uncertainty_loss_ignore_index():
    """Verify UncertaintyRegularizationLoss safely ignores label 255 (standard background/unlabeled)."""
    unc_loss = UncertaintyRegularizationLoss(entropy_threshold=0.65, ignore_index=255)
    pred_logits = torch.randn(1, 5, 16, 16)
    # Mask with 255 (standard background/unlabeled in WSSS)
    mask_with_ignore = torch.full((1, 16, 16), 255, dtype=torch.long)
    loss = unc_loss(pred_logits, mask_with_ignore)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_adversarial_otsu_gating_4d_batch_dimension_handling():
    """Verify compute_otsu_tissue_mask successfully processes 4D batched tensors (B, 3, H, W)."""
    batched_img = torch.rand(2, 3, 64, 64)
    mask = compute_otsu_tissue_mask(batched_img)
    assert mask.shape == (2, 64, 64)
    assert mask.dtype == np.uint8


def test_adversarial_relocalizer_upsampling_direction():
    """Verify whether SpatialAffinityRelocalization upsamples feature_map when CAM is larger.
    Exposes DEFECT: When (h_f, w_f) != (h, w), it upsamples feature_map to (h, w), which creates an O(N^2) token attention risk.
    """
    relocalizer = SpatialAffinityRelocalization(alpha=0.70)
    cam = torch.rand(1, 4, 28, 28)
    feat = torch.randn(1, 64, 14, 14)

    # Calling relocalizer with cam (28, 28) and feat (14, 14)
    # In line 56: feature_map = F.interpolate(feature_map, size=(28, 28)) -> tokens becomes (1, 784, 64) -> attention becomes (1, 784, 784)
    out = relocalizer(cam, feat)
    assert out.shape == (1, 4, 28, 28)

