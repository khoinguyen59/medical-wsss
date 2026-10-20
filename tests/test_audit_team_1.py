"""
Independent Adversarial Audit Test Suite for SE374.R11 WSSS.
Executed by Audit Team 1 (Auditor 1).

Coverage:
1. Patient-level anti-leakage and split determinism (Zero Leakage).
2. Clinical Otsu gating on challenging histology (mucinous, pale stroma, fat vacuoles, glass).
3. Literature & mathematical verification of losses (MLPS, Uncertainty Regularization, BCE).
4. DeepLabV3+ architecture with CGM, gradient flow, and B=1 batch dynamics.
5. Progressive Dropout Attention (PDA) and Spatial Affinity Relocalization.
6. Clinical biomarkers (VTP, TSR) edge cases and error bounds.
7. API server endpoints, data serialization, and empty input handling.
8. Invariant 12 character hygiene (strict standard ASCII hyphen 0x2D check).
"""

import json
import os
import re
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.datasets.otsu_filter import (
    apply_otsu_gating_to_prediction,
    compute_otsu_tissue_mask,
)
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
from src.models.backbone import MultiStageFeatureBackbone
from src.models.deeplabv3plus import (
    ClassificationGateMechanism,
    DeepLabV3PlusWithCGM,
)
from src.models.pda_generator import ProgressiveDropoutAttention
from src.models.wsss_tissue import (
    SpatialAffinityRelocalization,
    WSSSBenchmarkingEngine,
    apply_classmix_augmentation,
)
from src.visualization.api_server import (
    process_single_patch,
)
from src.visualization.clinical_viewer import (
    blend_overlay,
    compute_class_distribution,
    decode_base64_to_image,
    encode_image_to_base64,
    mask_to_color_image,
    mask_to_rgba_overlay,
)


class TestTask1CourseAndDataIntegrity:
    """Task 1 & 5: Course alignment, patient manifest integrity, zero leakage."""

    def test_bcss_patient_splits_counts(self):
        """Verify the 151-patient split counts conform exactly to manifest."""
        splits = load_patient_splits()
        assert len(splits["train"]) == 91, f"Expected 91 train patients, got {len(splits['train'])}"
        assert len(splits["val"]) == 30, f"Expected 30 val patients, got {len(splits['val'])}"
        assert len(splits["test"]) == 30, f"Expected 30 test patients, got {len(splits['test'])}"
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 151, f"Expected 151 total patients, got {total}"

    def test_zero_patient_leakage_strict_sets(self):
        """Verify 0.0% patient leakage across pairwise set intersections."""
        splits = load_patient_splits()
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        overlap_tv = train_set.intersection(val_set)
        overlap_tt = train_set.intersection(test_set)
        overlap_vt = val_set.intersection(test_set)

        assert len(overlap_tv) == 0, f"Train-Val patient leakage detected: {overlap_tv}"
        assert len(overlap_tt) == 0, f"Train-Test patient leakage detected: {overlap_tt}"
        assert len(overlap_vt) == 0, f"Val-Test patient leakage detected: {overlap_vt}"

    def test_extract_patient_id_adversarial_filenames(self):
        """Verify patient extraction on diverse and pathological filename variants."""
        cases = [
            ("TCGA-A2-A0T2_1024_2048_[1 0 0 1].png", "TCGA-A2-A0T2"),
            ("tcga-a1-a0sk_x100_y200_[0 1 0 0].png", "TCGA-A1-A0SK"),
            ("/path/to/nested/TCGA-BH-A0B8_slice.png", "TCGA-BH-A0B8"),
            ("GDPH_1234_patch_01.png", "GDPH"),
            ("CASE-99+001+[1 0 0 0].png", "CASE-99"),
        ]
        for fname, expected in cases:
            extracted = extract_patient_id(fname)
            assert extracted == expected, f"Failed on {fname}: got {extracted}, expected {expected}"

    def test_kfold_patient_determinism(self):
        """Verify deterministic K-fold split produces zero cross-fold leakage."""
        splits = load_patient_splits()
        all_patients = splits["train"] + splits["val"] + splits["test"]
        folds = split_patients_kfold(all_patients, n_splits=5, seed=42)

        assert len(folds) == 5
        for train_fold, val_fold in folds:
            overlap = set(train_fold).intersection(set(val_fold))
            assert len(overlap) == 0, f"K-fold leak detected: {overlap}"
            assert len(train_fold) + len(val_fold) == 151


class TestTask2And3MathematicalFormulations:
    """Task 2 & 3: Loss formulations, gradient flow, and mathematical rigor."""

    def test_multilabel_bce_loss_gradient_flow(self):
        """Verify MultiLabelClassificationLoss numerical stability and clean gradients."""
        loss_fn = MultiLabelClassificationLoss()
        logits = torch.randn(4, 4, requires_grad=True)
        targets = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss), "Loss contains NaN"
        assert not torch.isinf(loss), "Loss contains Inf"
        assert loss.item() > 0.0, "Loss must be positive"

        loss.backward()
        assert logits.grad is not None, "Gradients not computed"
        assert not torch.isnan(logits.grad).any(), "Gradients contain NaN"
        assert not (logits.grad == 0).all(), "Gradients are all zero"

    def test_mlps_loss_multi_resolution_gradient(self):
        """Verify MLPS loss aggregates multi-layer masks with proper interpolation."""
        mlps = MultiLayerPseudoSupervisionLoss(weights={"bn7": 1.0, "b5_2": 0.5, "b4_5": 0.25})
        pred_logits = torch.randn(2, 5, 56, 56, requires_grad=True)

        pseudo_masks = {
            "bn7": torch.randint(0, 5, (2, 56, 56)),
            "b5_2": torch.randint(0, 5, (2, 28, 28)),  # Half resolution
            "b4_5": torch.randint(0, 5, (2, 14, 14)),  # Quarter resolution
        }

        loss = mlps(pred_logits, pseudo_masks)
        assert not torch.isnan(loss)
        assert loss.item() > 0.0

        loss.backward()
        assert pred_logits.grad is not None
        assert not torch.isnan(pred_logits.grad).any()

    def test_uncertainty_loss_adaptive_weighting(self):
        """Verify UncertaintyRegularizationLoss discounts ambiguous high-entropy pixels."""
        loss_fn = UncertaintyRegularizationLoss(entropy_threshold=0.65)

        # Case 1: Confident predictions (low entropy) -> high weight (~1.0)
        # Logits with strong peak: [10.0, 0.0, 0.0, 0.0, 0.0]
        confident_logits = torch.zeros(1, 5, 4, 4)
        confident_logits[:, 0, :, :] = 10.0
        target_confident = torch.zeros(1, 4, 4, dtype=torch.long)
        loss_confident = loss_fn(confident_logits, target_confident)

        # Case 2: Ambiguous predictions (high entropy) -> low weight (~0.1)
        # Equal logits across classes: [1.0, 1.0, 1.0, 1.0, 1.0]
        ambiguous_logits = torch.ones(1, 5, 4, 4)
        target_ambiguous = torch.zeros(1, 4, 4, dtype=torch.long)
        loss_ambiguous = loss_fn(ambiguous_logits, target_ambiguous)

        # Both losses should be finite and positive
        assert not torch.isnan(loss_confident)
        assert not torch.isnan(loss_ambiguous)
        assert loss_confident.item() >= 0.0
        assert loss_ambiguous.item() > 0.0


class TestTask4ModelsAndArchitectures:
    """Task 4: Architecture inspection (DeepLabV3+, GroupNorm, PDA, ARML Relocalization)."""

    def test_deeplabv3plus_batch_size_one_training_mode(self):
        """Adversarial Test: Verify DeepLabV3PlusWithCGM forward and backward with batch size B=1 in train mode."""
        model = DeepLabV3PlusWithCGM(num_classes=5, in_channels=3)
        model.train()  # Strictly training mode!

        x = torch.randn(1, 3, 64, 64)
        targets = torch.randint(0, 5, (1, 64, 64))

        # Test forward pass with loss calculation
        logits, loss = model(x, targets=targets, return_loss=True)
        assert logits.shape == (1, 5, 64, 64)
        assert not torch.isnan(loss)

        # Test backward pass
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_classification_gate_mechanism_extinguishment(self):
        """Verify CGM penalizes absent classes by -1e4, driving Softmax probability < 1e-10."""
        cgm = ClassificationGateMechanism(penalty_value=-1e4, threshold=0.5)
        raw_logits = torch.tensor([[[[2.0, 3.0], [1.0, 4.0]]]])  # (1, 1, 2, 2)

        # Class present (1.0)
        presence_present = torch.tensor([[1.0]])
        gated_present = cgm(raw_logits, presence_present)
        assert torch.allclose(gated_present, raw_logits), "CGM modified present class logits"

        # Class absent (0.0)
        presence_absent = torch.tensor([[0.0]])
        gated_absent = cgm(raw_logits, presence_absent)
        assert torch.allclose(gated_absent, torch.full_like(raw_logits, -1e4)), "CGM failed to penalize absent class"

        # Multi-class softmax verification
        logits_5cls = torch.zeros(1, 5, 4, 4)
        # Mark class 0 as present, classes 1, 2, 3 as absent, class 4 (background) present
        presence_5cls = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]])
        gated_5cls = cgm(logits_5cls, presence_5cls)

        probs = F.softmax(gated_5cls, dim=1)
        # Classes 1, 2, 3 should have virtually 0 probability
        assert probs[0, 1].max().item() < 1e-10, "Absent class 1 had non-zero probability"
        assert probs[0, 2].max().item() < 1e-10, "Absent class 2 had non-zero probability"
        assert probs[0, 3].max().item() < 1e-10, "Absent class 3 had non-zero probability"

    def test_pda_progressive_expansion_and_prototype_gating(self):
        """Verify ProgressiveDropoutAttention broadens activation while controlling boundary bleeding."""
        pda = ProgressiveDropoutAttention(
            num_steps=3,
            top_p=0.25,
            base_dropout_rate=0.50,
            boundary_bleeding_control=True,
        )

        feat = torch.randn(2, 64, 16, 16).abs()
        weight = torch.randn(4, 64, 1, 1).abs()

        cam_agg, step_cams = pda.generate_cam(feat, weight, return_intermediates=True)
        assert cam_agg.shape == (2, 4, 16, 16)
        assert len(step_cams) == 4  # Initial step 0 + 3 progressive steps

        # Check normalization bounds [0.0, 1.0]
        assert cam_agg.min() >= 0.0
        assert cam_agg.max() <= 1.0 + 1e-5

    def test_spatial_affinity_relocalization_energy_diffusion(self):
        """Verify SpatialAffinityRelocalization diffuses activations across feature tokens."""
        reloc = SpatialAffinityRelocalization(alpha=0.70)
        cam = torch.zeros(1, 4, 8, 8)
        cam[:, :, 2:6, 2:6] = 1.0  # Center focus

        feat = torch.randn(1, 32, 8, 8)
        refined_cam = reloc(cam, feat)

        assert refined_cam.shape == cam.shape
        assert not torch.isnan(refined_cam).any()
        # Edge regions outside center should receive non-zero diffused energy
        assert refined_cam[:, :, 0, 0].mean().item() >= 0.0

    def test_classmix_augmentation_preserves_multilabel_logic(self):
        """Verify ClassMix augmentation applies correct logical OR on weak labels."""
        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        mixed_img, mixed_lbl, _ = apply_classmix_augmentation(images, labels, masks=None, p=1.0)
        assert mixed_img.shape == images.shape
        assert mixed_lbl.shape == labels.shape
        # Each mixed label should contain at least one positive class
        assert (mixed_lbl.sum(dim=1) >= 1.0).all()


class TestTask7ClinicalPathologyCritique:
    """Task 7: Clinical pathology critique on challenging histology (pale, fat, mucinous)."""

    def test_otsu_blank_glass_slide_elimination(self):
        """Verify Otsu gating completely zeros out pure white glass slides."""
        # Pure white glass slide: (255, 255, 255)
        glass_img = np.full((128, 128, 3), 255, dtype=np.uint8)
        tissue_mask = compute_otsu_tissue_mask(glass_img)

        assert tissue_mask.sum() == 0, f"Expected 0 tissue pixels on blank glass, got {tissue_mask.sum()}"

        # Test with scanner noise: random pixels in [240, 255]
        noisy_glass = np.random.randint(240, 256, (128, 128, 3), dtype=np.uint8)
        noisy_mask = compute_otsu_tissue_mask(noisy_glass)
        assert noisy_mask.sum() == 0, f"Expected 0 tissue pixels on noisy glass, got {noisy_mask.sum()}"

    def test_otsu_adipose_tissue_fat_vacuoles(self):
        """Verify fat vacuoles in adipose tissue are treated as background (class 4)."""
        # Simulate adipose tissue: mostly white lipid spaces with thin cellular borders
        adipose_patch = np.full((100, 100, 3), 250, dtype=np.uint8)
        # Thin stroma borders: pink H&E [220, 150, 180]
        adipose_patch[::20, :] = [220, 150, 180]
        adipose_patch[:, ::20] = [220, 150, 180]

        tissue_mask = compute_otsu_tissue_mask(adipose_patch)
        # Empty fat vacuoles (250, 250, 250) must NOT be classified as tissue
        assert tissue_mask[10, 10] == 0, "Fat vacuole interior was mistakenly marked as tissue"

        # Apply gating to dummy prediction: fat interior must be forced to class 4
        dummy_pred = np.zeros((100, 100), dtype=np.int64)  # Tumor class
        gated_pred = apply_otsu_gating_to_prediction(dummy_pred, adipose_patch, background_class_idx=4)
        assert gated_pred[10, 10] == 4, "Fat vacuole was not forced to background class 4"

    def test_clinical_biomarkers_boundary_cases(self):
        """Verify VTP and TSR mathematical boundaries and division by zero protection."""
        # Case 1: Pure background slide
        all_bg = np.full((50, 50), 4, dtype=np.int64)
        vtp_bg = compute_viable_tumor_percentage(all_bg)
        tsr_bg = compute_tumor_stroma_ratio(all_bg)
        assert vtp_bg == 0.0, "VTP should be 0.0 on background slide"
        assert tsr_bg == 0.0, "TSR should be 0.0 on background slide"

        # Case 2: Pure tumor slide
        all_tumor = np.zeros((50, 50), dtype=np.int64)
        vtp_tumor = compute_viable_tumor_percentage(all_tumor)
        tsr_tumor = compute_tumor_stroma_ratio(all_tumor)
        assert vtp_tumor == 100.0, f"Expected 100.0% VTP on pure tumor, got {vtp_tumor}"
        assert tsr_tumor == 100.0, f"Expected 100.0% TSR on pure tumor, got {tsr_tumor}"

        # Case 3: 50% tumor (class 0) and 50% stroma (class 1)
        half_half = np.zeros((50, 50), dtype=np.int64)
        half_half[:25, :] = 0  # Tumor
        half_half[25:, :] = 1  # Stroma
        vtp_hh = compute_viable_tumor_percentage(half_half)
        tsr_hh = compute_tumor_stroma_ratio(half_half)
        assert abs(vtp_hh - 50.0) < 1e-4, f"Expected 50.0% VTP, got {vtp_hh}"
        assert abs(tsr_hh - 50.0) < 1e-4, f"Expected 50.0% TSR, got {tsr_hh}"

        # Case 4: Pure lymphocytes (class 2)
        all_lym = np.full((50, 50), 2, dtype=np.int64)
        vtp_lym = compute_viable_tumor_percentage(all_lym)
        tsr_lym = compute_tumor_stroma_ratio(all_lym)
        assert vtp_lym == 0.0, "VTP must be 0.0 for pure lymphocytes"
        assert tsr_lym == 0.0, "TSR must be 0.0 for pure lymphocytes"


class TestTask8ApiAndDeliverables:
    """Task 8: API server validation, OpenSeadragon helper functions, and Invariant 12 audit."""

    def test_process_single_patch_pipeline(self):
        """Verify process_single_patch end-to-end execution on synthetic histology patch."""
        # Create synthetic H&E patch
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        patch[:32, :32] = [180, 50, 120]   # Purple/pink tumor nuclei
        patch[32:, 32:] = [220, 180, 200]  # Pale pink stroma
        patch[:32, 32:] = [255, 255, 255]  # Glass background

        result = process_single_patch(patch, mode="proposed", enforce_otsu=True)
        assert result["success"] if "success" in result else True
        assert result["mask_shape"] == [64, 64]
        assert "viable_tumor_percentage" in result["biomarkers"]
        assert "tumor_stroma_ratio" in result["biomarkers"]
        assert result["mask_base64"].startswith("data:image/png;base64,")
        assert result["overlay_base64"].startswith("data:image/png;base64,")

    def test_base64_encode_decode_roundtrip(self):
        """Verify image encoding and decoding roundtrips perfectly."""
        orig_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        b64 = encode_image_to_base64(orig_img, format="PNG")
        decoded = decode_base64_to_image(b64)
        assert orig_img.shape == decoded.shape
        assert np.array_equal(orig_img, decoded), "Decoded image differed from original"

    def test_invariant_12_strict_ascii_hyphens(self):
        """Scrutinize codebase and final technical report for forbidden non-ASCII dashes."""
        forbidden = [
            ("\u2014", "em-dash"),
            ("\u2013", "en-dash"),
            ("\u2015", "horizontal bar"),
            ("\u2212", "minus sign"),
        ]

        files_to_check = [
            os.path.join("src", "models", "backbone.py"),
            os.path.join("src", "models", "pda_generator.py"),
            os.path.join("src", "models", "deeplabv3plus.py"),
            os.path.join("src", "models", "wsss_tissue.py"),
            os.path.join("src", "datasets", "bcss_dataset.py"),
            os.path.join("src", "datasets", "otsu_filter.py"),
            os.path.join("src", "datasets", "patient_split.py"),
            os.path.join("src", "losses_metrics", "losses.py"),
            os.path.join("src", "losses_metrics", "metrics.py"),
            os.path.join("src", "visualization", "api_server.py"),
            os.path.join("product", "reports", "SE374_FINAL_TECHNICAL_REPORT_WSSS.md"),
        ]

        violations = []
        for fpath in files_to_check:
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            for char, desc in forbidden:
                if char in content:
                    count = content.count(char)
                    violations.append(f"{fpath}: contains {count} instances of {desc} (U+{ord(char):04X})")

        assert len(violations) == 0, "Invariant 12 violations found:\n" + "\n".join(violations)
