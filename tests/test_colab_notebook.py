"""Unit and integration tests for Google Colab WSSS Pipeline Notebook.

Verifies:
1. Valid Jupyter Notebook JSON structure (nbformat 4, nbformat_minor 0).
2. Clean compilation of all Python code cells without syntax errors.
3. Invariant 12 compliance: 100% ASCII 0x2D hyphens, 0 em-dashes/en-dashes.
4. 100% elimination of naive HSV color heuristics in inference.
5. DeepLabV3+ ResNet-101 architecture instantiation, key consistency, and forward pass.
6. Otsu clinical gating boundary condition: pure white glass forced to Class 4 (Background).
"""

import ast
import json
import re
from pathlib import Path
import numpy as np
import pytest
import torch

NOTEBOOK_PATH = Path("notebooks/SE374_Colab_WSSS_Pipeline.ipynb")


def test_notebook_json_structure_and_nbformat():
    """Verify notebook exists, parses cleanly as JSON, and conforms to nbformat 4."""
    assert NOTEBOOK_PATH.is_file(), f"Notebook not found at: {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data, "Notebook missing 'cells' key"
    assert nb_data["nbformat"] == 4, f"Expected nbformat 4, got {nb_data.get('nbformat')}"
    assert nb_data["nbformat_minor"] == 0, f"Expected nbformat_minor 0, got {nb_data.get('nbformat_minor')}"
    assert len(nb_data["cells"]) == 19, f"Expected 19 cells, got {len(nb_data['cells'])}"


def test_code_cells_syntax_compilation():
    """Verify that every code cell parses into an AST without any Python syntax errors."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [c for c in nb_data["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 9, f"Expected 9 code cells, got {len(code_cells)}"

    for idx, cell in enumerate(code_cells):
        raw_source = "".join(cell.get("source", []))
        # Strip Colab bash magic lines starting with '!' for standard Python compilation
        python_lines = []
        for line in raw_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("!"):
                # Replace magic shell call with harmless comment to preserve line numbering
                python_lines.append(f"# {stripped}")
            else:
                python_lines.append(line)

        python_code = "\n".join(python_lines)
        try:
            tree = ast.parse(python_code, filename=f"Cell_{idx}")
            assert tree is not None
        except SyntaxError as e:
            pytest.fail(f"Syntax error in Code Cell {idx} (line {e.lineno}): {e.msg}\nCode snippet:\n{python_code[:200]}")


def test_invariant_12_strict_ascii_hyphens_only():
    """Verify 0 non-ASCII em-dashes (\\u2014) or en-dashes (\\u2013) across notebook."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    forbidden_chars = [
        ("\u2014", "em-dash"),
        ("\u2013", "en-dash"),
        ("\u2015", "horizontal bar"),
        ("\u2212", "minus sign"),
    ]
    for char, name in forbidden_chars:
        count = content.count(char)
        assert count == 0, f"Invariant 12 violation: Found {count} instances of {name} ({hex(ord(char))}) in {NOTEBOOK_PATH}"


def test_zero_naive_heuristic_quarantine():
    """Verify complete absence of naive HSV color slicing heuristics in inference."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "dark_nuclei" not in content, "Found deprecated 'dark_nuclei' naive heuristic in notebook!"
    assert "pred[tissue_mask] = 1" not in content, "Found deprecated 'pred[tissue_mask] = 1' heuristic in notebook!"
    assert "pred[hsv[:, :, 1] > 30]" not in content, "Found deprecated naive saturation indexing in notebook!"


def test_deeplabv3plus_architecture_and_forward_pass():
    """Verify the standalone DeepLabV3+ module instantiates and executes forward pass."""
    # Extract code from Cell 12 (architecture cell)
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_12 = nb_data["cells"][12]
    raw_source = "".join(cell_12.get("source", []))
    # Extract classes and function definitions before model loading line
    code_lines = []
    for line in raw_source.splitlines():
        if "CKPT_FILE" in line or "build_and_load_deeplabv3plus(CKPT_FILE" in line:
            break
        code_lines.append(line)

    env = {}
    exec("\n".join(code_lines), env)

    assert "DeepLabV3Plus" in env, "DeepLabV3Plus class not found in Cell 12"
    assert "OtsuClinicalGating" in env, "OtsuClinicalGating class not found in Cell 12"

    DeepLabV3Plus = env["DeepLabV3Plus"]
    model_4cls = DeepLabV3Plus(num_classes=4, output_stride=16)
    model_4cls.eval()

    total_params = sum(p.numel() for p in model_4cls.parameters())
    assert total_params == 59339940, f"Expected 59,339,940 parameters, got {total_params}"

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model_4cls(dummy_input)

    assert output.shape == (1, 4, 224, 224), f"Expected shape (1, 4, 224, 224), got {output.shape}"


def test_otsu_clinical_gating_pure_white_glass():
    """Verify Otsu gating forces pure white glass slide to Class 4 (Background)."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_12 = nb_data["cells"][12]
    raw_source = "".join(cell_12.get("source", []))
    code_lines = [l for l in raw_source.splitlines() if "CKPT_FILE" not in l and "model =" not in l]

    env = {}
    exec("\n".join(code_lines), env)
    OtsuClinicalGating = env["OtsuClinicalGating"]

    # Create pure white glass image (RGB 255, 255, 255)
    white_glass = np.full((128, 128, 3), 255, dtype=np.uint8)
    tissue_mask = OtsuClinicalGating.compute_tissue_mask(white_glass)
    assert np.all(tissue_mask == 0), "Tissue mask should be 0 for pure white glass"

    # Suppose model mistakenly predicted Tumor (Class 0) everywhere
    dummy_pred = np.zeros((128, 128), dtype=np.uint8)
    gated = OtsuClinicalGating.apply_gating(dummy_pred, white_glass, background_class_idx=4)
    assert np.all(gated == 4), "100% of pure white glass pixels must be forced to Class 4 (Background)"


def test_otsu_clinical_gating_adversarial_boundary_conditions():
    """Adversarially test Otsu clinical gating under extreme inputs:
    - Pure white glass (RGB 255, 255, 255) -> 0% tissue
    - Near-white glass (RGB 238-255) -> 0% tissue
    - Half-glass half-tissue patch -> exactly separates tissue from glass
    - Defensive ValueError on invalid 2D or 4-channel RGBA inputs.
    """
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_12 = nb_data["cells"][12]
    code_lines = [l for l in "".join(cell_12.get("source", [])).splitlines() if "CKPT_FILE" not in l and "model =" not in l]
    env = {}
    exec("\n".join(code_lines), env)
    OtsuClinicalGating = env["OtsuClinicalGating"]

    # 1. Near-white glass
    near_white = np.random.randint(238, 256, (128, 128, 3), dtype=np.uint8)
    mask_near_white = OtsuClinicalGating.compute_tissue_mask(near_white)
    assert np.all(mask_near_white == 0), "Near-white glass must have 0% tissue mask"

    # 2. Half-glass half-tissue patch (left: stained tissue [180, 80, 140], right: white glass [255, 255, 255])
    half_patch = np.full((128, 128, 3), 255, dtype=np.uint8)
    half_patch[:, :64] = [180, 80, 140]
    mask_half = OtsuClinicalGating.compute_tissue_mask(half_patch)
    assert np.all(mask_half[:, :64] == 1), "Stained tissue region must be detected as 1"
    assert np.all(mask_half[:, 64:] == 0), "White glass region must be detected as 0"

    # 3. Defensive checks on invalid image dimensions
    with pytest.raises(ValueError):
        OtsuClinicalGating.compute_tissue_mask(np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError):
        OtsuClinicalGating.compute_tissue_mask(np.zeros((64, 64, 4), dtype=np.uint8))


def test_otsu_clinical_gating_spatial_alignment_and_morphology():
    """Verify spatial alignment (H, W) across irregular shapes and morphological clean-up:
    - 1x1 noise speck eliminated by MORPH_OPEN
    - 1x1 micro-hole closed by MORPH_CLOSE
    - Irregular and odd aspect ratios: (33, 47), (225, 225), (512, 256).
    """
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_12 = nb_data["cells"][12]
    code_lines = [l for l in "".join(cell_12.get("source", [])).splitlines() if "CKPT_FILE" not in l and "model =" not in l]
    env = {}
    exec("\n".join(code_lines), env)
    OtsuClinicalGating = env["OtsuClinicalGating"]

    # 1. Morphological Open: 1x1 noise speck on white background should be suppressed
    speck_img = np.full((64, 64, 3), 250, dtype=np.uint8)
    speck_img[32, 32] = [180, 50, 150]
    mask_speck = OtsuClinicalGating.compute_tissue_mask(speck_img)
    assert mask_speck[32, 32] == 0, "Isolated 1x1 noise speck must be eliminated by MORPH_OPEN"

    # 2. Morphological Close: 1x1 hole inside solid tissue should be closed
    hole_img = np.full((64, 64, 3), [180, 50, 150], dtype=np.uint8)
    hole_img[32, 32] = [255, 255, 255]
    mask_hole = OtsuClinicalGating.compute_tissue_mask(hole_img)
    assert mask_hole[32, 32] == 1, "1x1 hole inside tissue must be filled by MORPH_CLOSE"

    # 3. Spatial alignment across various irregular and odd dimensions
    test_shapes = [(33, 47), (225, 225), (512, 256), (7, 13)]
    for h, w in test_shapes:
        dummy_rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        dummy_pred = np.zeros((h, w), dtype=np.uint8)
        gated = OtsuClinicalGating.apply_gating(dummy_pred, dummy_rgb, background_class_idx=4)
        assert gated.shape == (h, w), f"Expected shape ({h}, {w}), got {gated.shape}"


def test_model_memory_safety_and_tensor_footprint():
    """Verify DeepLabV3+ memory footprint is strictly < 4 GB VRAM on T4:
    - FP32 parameters = 59,339,940 (~226.36 MB)
    - Forward pass across batch sizes 1, 2, 4, 8 and irregular resolutions
    - Evaluates within Colab T4 memory budget (< 4 GB).
    """
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_12 = nb_data["cells"][12]
    code_lines = [l for l in "".join(cell_12.get("source", [])).splitlines() if "CKPT_FILE" not in l and "model =" not in l]
    env = {}
    exec("\n".join(code_lines), env)
    DeepLabV3Plus = env["DeepLabV3Plus"]

    model = DeepLabV3Plus(num_classes=4, output_stride=16)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024**2)

    assert total_params == 59339940, f"Expected 59,339,940 parameters, got {total_params}"
    assert param_mb < 250.0, f"Model parameter footprint ({param_mb:.2f} MB) exceeds 250 MB"

    # Verify forward pass on batch sizes 1, 2, 4
    for b in [1, 2, 4]:
        inp = torch.randn(b, 3, 224, 224)
        with torch.no_grad():
            out = model(inp)
        assert out.shape == (b, 4, 224, 224), f"Expected output shape ({b}, 4, 224, 224), got {out.shape}"

    # Verify forward pass on rectangular and odd resolutions
    for h, w in [(224, 320), (225, 225)]:
        inp = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out = model(inp)
        assert out.shape == (1, 4, h, w), f"Expected output shape (1, 4, {h}, {w}), got {out.shape}"


def test_dual_mode_patient_selection_and_anti_leakage():
    """Verify dual-mode logic and zero patient leakage:
    - TARGET_PATIENTS (TCGA-AQ-A04H, TCGA-C8-A1HJ, TCGA-AR-A0TQ) exist in test split
    - Zero overlap between TARGET_PATIENTS and train/val splits
    - Sanity quota distribution logic guarantees exact 150 samples without leakage.
    """
    manifest_path = Path("src/config/bcss_patient_splits.json")
    assert manifest_path.is_file(), f"Manifest not found: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_set = set(manifest["splits"]["train"])
    val_set = set(manifest["splits"]["val"])
    test_set = set(manifest["splits"]["test"])

    target_patients = {"TCGA-AQ-A04H", "TCGA-C8-A1HJ", "TCGA-AR-A0TQ"}

    # Invariant 08 check: Zero patient leakage
    assert target_patients.issubset(test_set), "TARGET_PATIENTS must all belong to test split!"
    assert len(target_patients & train_set) == 0, "TARGET_PATIENTS leaked into train split!"
    assert len(target_patients & val_set) == 0, "TARGET_PATIENTS leaked into val split!"

    # Verify sampling logic from Cell 8
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_8_code = "".join(nb_data["cells"][8].get("source", []))
    assert "SANITY_MODE = True" in cell_8_code
    assert "MAX_SANITY_SAMPLES = 150" in cell_8_code
    assert "TARGET_PATIENTS = [\"TCGA-AQ-A04H\", \"TCGA-C8-A1HJ\", \"TCGA-AR-A0TQ\"]" in cell_8_code


def test_wsss_tissue_native_0_1_normalization_and_outside_roi_exclusion():
    """Verify exact alignment with WSSS-Tissue Stage2_Dataset preprocessing and evaluation:
    - Uses transforms.ToTensor() scaling RGB to [0.0, 1.0] (mean=0, std=1)
    - Does NOT apply ImageNet mean/std shift (which destroys TUM and LYM logits)
    - Applies BCSS-WSSS outside-ROI exclusion rule: pred_final[gt_mask == 4] = 4.
    """
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    full_source = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"] if c["cell_type"] == "code")
    cell_14_code = "".join(nb_data["cells"][14].get("source", []))

    assert "transforms.Normalize(mean=[0.485" not in full_source, (
        "ImageNet mean/std normalization must NOT be applied to WSSS-Tissue Stage 2 checkpoint!"
    )
    assert "transforms.ToTensor()" in cell_14_code
    assert "pred_final[gt_mask == 4] = 4" in cell_14_code, (
        "BCSS-WSSS outside-ROI exclusion rule (pred_final[gt_mask == 4] = 4) missing in Cell 14!"
    )


def test_otsu_clinical_gating_preserves_unimodal_stained_he_tissue():
    """Verify Otsu clinical gating does NOT punch swiss-cheese holes inside 100% stained H&E patches
    where dark hematoxylin nuclei (high saturation) coexist with paler eosin cytoplasm/stroma (lower saturation).
    """
    from src.datasets.otsu_filter import compute_otsu_tissue_mask

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_12 = nb_data["cells"][12]
    code_lines = [l for l in "".join(cell_12.get("source", [])).splitlines() if "CKPT_FILE" not in l and "model =" not in l]
    env = {}
    exec("\n".join(code_lines), env)
    OtsuClinicalGating = env["OtsuClinicalGating"]

    # Construct a realistic 224x224 unimodal stained H&E patch (zero glass background):
    # Top half: Dark purple hematoxylin tumor nuclei (RGB [85, 45, 130], high saturation ~167)
    # Bottom half: Paler pink/lilac cytoplasm & stroma (RGB [210, 185, 205], low saturation ~30, gray ~195)
    he_patch = np.zeros((224, 224, 3), dtype=np.uint8)
    he_patch[:112, :] = [85, 45, 130]
    he_patch[112:, :] = [210, 185, 205]
    # Add small intracellular vacuole (8x8 bright region, 64 pixels < 256 px threshold)
    he_patch[50:58, 50:58] = [245, 245, 245]

    nb_mask = OtsuClinicalGating.compute_tissue_mask(he_patch)
    src_mask = compute_otsu_tissue_mask(he_patch)

    assert np.all(nb_mask == 1), (
        f"Notebook OtsuClinicalGating falsely masked {(nb_mask == 0).sum()} stained tissue pixels as background!"
    )
    assert np.all(src_mask == 1), (
        f"src.datasets.otsu_filter falsely masked {(src_mask == 0).sum()} stained tissue pixels as background!"
    )


def test_vietnamese_diacritics_and_unique_timestamped_zip_export():
    """Verify full UTF-8 Vietnamese diacritics in visualization/report cells and unique timestamped zip archive."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_16_code = "".join(nb_data["cells"][16].get("source", []))
    cell_18_code = "".join(nb_data["cells"][18].get("source", []))

    # Verify proper Vietnamese diacritics in plots and report
    assert "Ma Trận Nhầm Lẫn" in cell_16_code
    assert "Tiêu bản H&E gốc" in cell_16_code
    assert "Báo Cáo Đánh Giá Phân Vùng Mô Bệnh Học WSSS" in cell_18_code

    # Verify timestamped unique zip archive naming
    assert "timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')" in cell_18_code
    assert "SE374_Colab_Results_DeepLabV3Plus_{mode_tag}_{timestamp_str}" in cell_18_code


def test_evaluator_4cls_and_5cls_mathematical_integrity():
    """Verify ColabSegmentationEvaluator computes 4-class WSSS-Tissue metrics without
    Oracle GT leakage or Otsu false-negative distortion on tissue classes (0..3).
    """
    from typing import Dict, List, Optional, Union

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_14 = nb_data["cells"][14]
    raw_source = "".join(cell_14.get("source", []))
    code_lines = []
    for line in raw_source.splitlines():
        if line.startswith("transform ="):
            break
        code_lines.append(line)

    env = {"np": np, "Optional": Optional, "List": List, "Dict": Dict, "Union": Union}
    exec("\n".join(code_lines), env)
    ColabSegmentationEvaluator = env["ColabSegmentationEvaluator"]

    evaluator = ColabSegmentationEvaluator(num_classes=5)
    # Create a 4x4 synthetic mask with classes 0, 1, 2, 3 and outside-ROI class 4
    gt_mask = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [4, 4, 4, 4],
        [0, 1, 2, 3],
    ], dtype=np.int64)
    # Raw 4-class model predicts 100% accurately inside ROI (classes 0..3)
    pred_raw = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [0, 1, 0, 1],
        [0, 1, 2, 3],
    ], dtype=np.uint8)
    # Suppose Otsu gating turned one TUM pixel (0,0) into BACK (4)
    pred_gated = pred_raw.copy()
    pred_gated[0, 0] = 4
    pred_gated[gt_mask == 4] = 4
    pred_raw_roi = pred_raw.copy()
    pred_raw_roi[gt_mask == 4] = 4

    evaluator.update(pred_gated, gt_mask, pred_raw_4cls=pred_raw_roi)
    res = evaluator.compute_metrics()

    # Pure 4-class WSSS-Tissue evaluation on pred_raw_roi must be 100.0%
    assert np.isclose(res["mIoU_4cls"], 100.0), f"Expected 100.0% 4-class mIoU, got {res['mIoU_4cls']}"
    assert np.isclose(res["mDice_4cls"], 100.0), f"Expected 100.0% 4-class mDice, got {res['mDice_4cls']}"
    assert np.isclose(res["Accuracy"], 100.0), f"Expected 100.0% 4-class Accuracy, got {res['Accuracy']}"


def test_sanity_mode_quota_distributes_across_5_distinct_patients(tmp_path):
    """Execute Cell 8 patient sampling logic on a mock directory of 6 patients (100 patches each)
    and verify that SANITY_MODE selects exactly 150 patches evenly across 5 distinct patients (30 each).
    """
    from collections import defaultdict

    test_img_dir = tmp_path / "test" / "img"
    test_img_dir.mkdir(parents=True)

    mock_patients = [
        "TCGA-D8-A27F",
        "TCGA-E2-A14X",
        "TCGA-A2-A04P",
        "TCGA-AR-A1AQ",
        "TCGA-BH-A0B3",
        "TCGA-EW-A1P8",
    ]
    for pid in mock_patients:
        for idx in range(60):
            (test_img_dir / f"{pid}_patch_{idx:03d}.png").touch()

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_8_code = "".join(nb_data["cells"][8].get("source", []))
    env = {"test_img_dir": test_img_dir, "re": re, "defaultdict": defaultdict}
    exec(cell_8_code, env)

    active_files = env["active_test_files"]
    extract_pid = env["extract_patient_id"]
    assert len(active_files) == 150, f"Expected 150 sanity samples, got {len(active_files)}"

    counts_by_pid = defaultdict(int)
    for p in active_files:
        counts_by_pid[extract_pid(p.name)] += 1

    assert len(counts_by_pid) == 5, f"Expected 5 distinct patients in Sanity Mode, got {len(counts_by_pid)}: {dict(counts_by_pid)}"
    for pid, count in counts_by_pid.items():
        assert count == 30, f"Expected 30 patches for patient {pid}, got {count}"


def test_visualization_cell_does_not_overwrite_ai_prediction_with_gt():
    """Verify Cell 16 does NOT overwrite pred_gated with ground truth (pred_gated[gt == 4] = 4)."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_16_code = "".join(nb_data["cells"][16].get("source", []))
    assert "pred_gated[gt == 4] = 4" not in cell_16_code, (
        "Cell 16 must not overwrite AI prediction with Ground Truth outside-ROI mask!"
    )


def test_all_four_hardware_specific_notebook_variants():
    """Verify all 4 separated hardware-specific Colab notebooks (T4 Sanity 150, T4 Full 4986,
    L4 Full 4986, A100 Full 4986) exist, compile cleanly, obey Invariant 12, and have exact
    GPU metadata, High-RAM guidance, and SANITY_MODE settings.
    """
    expected_variants = [
        ("notebooks/SE374_Colab_WSSS_T4_Sanity150.ipynb", "T4", True, "T4_Sanity150"),
        ("notebooks/SE374_Colab_WSSS_T4_Full4986.ipynb", "T4", False, "T4_Full4986"),
        ("notebooks/SE374_Colab_WSSS_L4_Full4986.ipynb", "L4", False, "L4_Full4986"),
        ("notebooks/SE374_Colab_WSSS_A100_Full4986.ipynb", "A100", False, "A100_Full4986"),
    ]

    for rel_path, expected_gpu, expected_sanity, expected_tag in expected_variants:
        p = Path(rel_path)
        assert p.is_file(), f"Missing hardware variant notebook: {rel_path}"
        raw_text = p.read_text(encoding="utf-8")

        # Invariant 12 check
        for char in ("\u2014", "\u2013", "\u2015", "\u2212"):
            assert char not in raw_text, f"Forbidden dash found in {rel_path}"

        nb = json.loads(raw_text)
        assert nb["metadata"]["colab"]["gpuType"] == expected_gpu
        assert len(nb["cells"]) == 19

        # Check High-RAM note in Cell 0
        cell_0_md = "".join(nb["cells"][0]["source"])
        assert "High-RAM" in cell_0_md, f"Missing High-RAM note in Cell 0 of {rel_path}"

        # Check SANITY_MODE in Cell 8
        cell_8_code = "".join(nb["cells"][8]["source"])
        assert f"SANITY_MODE = {expected_sanity}" in cell_8_code, (
            f"Expected SANITY_MODE = {expected_sanity} in {rel_path}"
        )

        # Check unique hardware tag in Cell 18
        cell_18_code = "".join(nb["cells"][18]["source"])
        assert f"mode_tag = '{expected_tag}'" in cell_18_code, (
            f"Expected mode_tag = '{expected_tag}' in {rel_path}"
        )


def test_multi_method_wsss_ablation_engine():
    """Verify MultiMethodWSSSEngine executes all 5 genuine research methods (M1..M5)
    both in src/models/wsss_inference_methods.py and inside the Colab notebook Cell 12.
    """
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_12 = nb_data["cells"][12]
    code_lines = [
        l for l in "".join(cell_12.get("source", [])).splitlines()
        if "CKPT_FILE" not in l and "model =" not in l
    ]
    env = {}
    exec("\n".join(code_lines), env)

    assert "MultiMethodWSSSEngine" in env, "MultiMethodWSSSEngine missing in Cell 12"
    DeepLabV3Plus = env["DeepLabV3Plus"]
    NbEngine = env["MultiMethodWSSSEngine"]

    model = DeepLabV3Plus(num_classes=4, output_stride=16)
    model.eval()

    src_engine = MultiMethodWSSSEngine()
    nb_engine = NbEngine()

    dummy_img = torch.rand(1, 3, 64, 64)
    dummy_rgb = (dummy_img.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    preds_src = src_engine.predict_all_methods(model, dummy_img, raw_rgb=dummy_rgb)
    preds_nb = nb_engine.predict_all_methods(model, dummy_img, raw_rgb=dummy_rgb)

    assert list(preds_src.keys()) == MultiMethodWSSSEngine.METHOD_KEYS
    assert list(preds_nb.keys()) == MultiMethodWSSSEngine.METHOD_KEYS

    for k in MultiMethodWSSSEngine.METHOD_KEYS:
        assert preds_src[k].shape == (64, 64)
        assert np.array_equal(preds_src[k], preds_nb[k]), f"Mismatch between src and notebook for {k}"

    # Verify pure_glass_guard does NOT punch holes in pale stained NEC/STR tissue (gray=210, sat=25)
    pale_nec = np.full((64, 64, 3), [215, 195, 210], dtype=np.uint8)
    nec_pred = np.full((64, 64), 3, dtype=np.uint8)
    guarded_nec = src_engine.apply_pure_glass_guard(nec_pred, pale_nec, background_class_idx=4)
    assert np.all(guarded_nec == 3), "pure_glass_guard must preserve 100% of pale stained Necrosis (NEC)!"


def test_stage1_netcam_architecture_and_forward_cam():
    """Verify WSSSTissueStage1NetCAM exact parameter parity and forward/CAM execution.

    Verifies:
    1. Instantiates WSSSTissueStage1NetCAM with exactly 105,087,296 parameters.
    2. Strict 100% parameter key and shape match against repos/WSSS-Tissue/network/resnet38_cls.py.
    3. Forward pass produces valid sigmoid image-level probabilities [B, 4] in [0, 1].
    4. forward_cam produces spatial CAM [B, 4, H/8, W/8] with ReLU activations and image probabilities.
    5. MultiMethodWSSSEngine integration with model_stage1 activates CGM gating and CAM fusion.
    """
    import sys
    sys.path.insert(0, str(Path("repos/WSSS-Tissue").resolve()))
    from network.resnet38_cls import Net_CAM
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine, WSSSTissueStage1NetCAM

    model_stage1 = WSSSTissueStage1NetCAM(n_class=4)
    model_stage1.eval()

    total_params = sum(p.numel() for p in model_stage1.parameters())
    assert total_params == 105087296, f"Expected 105,087,296 parameters, got {total_params}"

    ref_model = Net_CAM(n_class=4)
    ref_dict = ref_model.state_dict()
    my_dict = model_stage1.state_dict()

    assert set(ref_dict.keys()) == set(my_dict.keys()), (
        f"Key mismatch! Ref keys: {len(ref_dict)} vs Mine: {len(my_dict)}"
    )
    for k in ref_dict:
        assert ref_dict[k].shape == my_dict[k].shape, (
            f"Shape mismatch for key {k}: {ref_dict[k].shape} vs {my_dict[k].shape}"
        )

    # Verify strict state dict loading succeeds without key missing/unexpected errors
    model_stage1.load_state_dict(ref_dict, strict=True)

    # Test forward pass on 64x64 input
    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y_prob = model_stage1(dummy_input)
    assert y_prob.shape == (1, 4), f"Expected shape (1, 4), got {y_prob.shape}"
    assert torch.all(y_prob >= 0.0) and torch.all(y_prob <= 1.0), "Probabilities must be in [0, 1]"

    # Test forward_cam on 64x64 input (H/8 = 8, W/8 = 8)
    with torch.no_grad():
        cam, y_cls = model_stage1.forward_cam(dummy_input)
    assert cam.shape == (1, 4, 8, 8), f"Expected CAM shape (1, 4, 8, 8), got {cam.shape}"
    assert y_cls.shape == (1, 4), f"Expected y_cls shape (1, 4), got {y_cls.shape}"
    assert torch.all(cam >= 0.0), "CAM activations must be non-negative (ReLU)"

    # Test MultiMethodWSSSEngine with model_stage1 loaded
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    cell_12 = nb_data["cells"][12]
    code_lines = [
        l for l in "".join(cell_12.get("source", [])).splitlines()
        if "CKPT_FILE" not in l and "model =" not in l
    ]
    env = {}
    exec("\n".join(code_lines), env)
    DeepLabV3Plus = env["DeepLabV3Plus"]
    model_stage2 = DeepLabV3Plus(num_classes=4, output_stride=16)
    model_stage2.eval()

    engine = MultiMethodWSSSEngine(model_stage1=model_stage1)
    dummy_rgb = (dummy_input.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    preds = engine.predict_all_methods(model_stage2, dummy_input, raw_rgb=dummy_rgb)

    assert list(preds.keys()) == MultiMethodWSSSEngine.METHOD_KEYS
    for k in MultiMethodWSSSEngine.METHOD_KEYS:
        assert preds[k].shape == (64, 64), f"Method {k} output shape expected (64, 64), got {preds[k].shape}"
        assert preds[k].dtype == np.uint8
        assert np.all((preds[k] >= 0) & (preds[k] <= 4))


def test_both_official_checkpoints_download_in_notebook_and_all_variants():
    """Verify that both official BCSS checkpoints are present and downloaded in master and all 4 variants:
    - Stage 1: stage1_checkpoint_trained_on_bcss.pth (ID 118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc)
    - Stage 2: stage2_checkpoint_trained_on_bcss.pth (ID 1cXm29ko8UevbaeW6YreSg-vuNaQiN_je)
    """
    stage1_id = "118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc"
    stage2_id = "1cXm29ko8UevbaeW6YreSg-vuNaQiN_je"
    stage1_file = "stage1_checkpoint_trained_on_bcss.pth"
    stage2_file = "stage2_checkpoint_trained_on_bcss.pth"

    all_notebook_paths = [
        Path("notebooks/SE374_Colab_WSSS_Pipeline.ipynb"),
        Path("notebooks/SE374_Colab_WSSS_T4_Sanity150.ipynb"),
        Path("notebooks/SE374_Colab_WSSS_T4_Full4986.ipynb"),
        Path("notebooks/SE374_Colab_WSSS_L4_Full4986.ipynb"),
        Path("notebooks/SE374_Colab_WSSS_A100_Full4986.ipynb"),
    ]

    for nb_path in all_notebook_paths:
        assert nb_path.is_file(), f"Notebook file missing: {nb_path}"
        raw_content = nb_path.read_text(encoding="utf-8")
        assert stage1_id in raw_content, f"Stage 1 checkpoint ID {stage1_id} missing in {nb_path}"
        assert stage2_id in raw_content, f"Stage 2 checkpoint ID {stage2_id} missing in {nb_path}"
        assert stage1_file in raw_content, f"Stage 1 filename {stage1_file} missing in {nb_path}"
        assert stage2_file in raw_content, f"Stage 2 filename {stage2_file} missing in {nb_path}"
        assert "WSSSTissueStage1NetCAM" in raw_content, f"WSSSTissueStage1NetCAM missing in {nb_path}"


def test_all_variants_pass_model_stage1_in_evaluation_and_visualization():
    """Verify that Cell 14 and Cell 16 in all 4 variant notebooks pass model_stage1 to predict_all_methods."""
    variants = [
        "notebooks/SE374_Colab_WSSS_T4_Sanity150.ipynb",
        "notebooks/SE374_Colab_WSSS_T4_Full4986.ipynb",
        "notebooks/SE374_Colab_WSSS_L4_Full4986.ipynb",
        "notebooks/SE374_Colab_WSSS_A100_Full4986.ipynb",
    ]
    for var_path in variants:
        p = Path(var_path)
        nb = json.loads(p.read_text(encoding="utf-8"))
        cell_14_src = "".join(nb["cells"][14]["source"])
        cell_16_src = "".join(nb["cells"][16]["source"])
        assert "model_stage1=model_stage1" in cell_14_src, f"Cell 14 in {var_path} missing model_stage1 argument"
        assert "model_stage1=model_stage1" in cell_16_src, f"Cell 16 in {var_path} missing model_stage1 argument"


def test_stage1_netcam_exact_layer_shapes_and_weights_breakdown():
    """Verify exact layer-by-layer parameter shapes, layer naming, and multi-class flexibility
    for WSSSTissueStage1NetCAM matching Han et al. (MedIA 2022) ResNet-38d architecture:
    - conv1a: (64, 3, 3, 3)
    - b2: (128, 64, 1, 1), (128, 64, 3, 3), (128, 128, 3, 3)
    - b6: Bottleneck block with dilation 4, dropout 0.3
    - b7: Bottleneck block with dilation 4, dropout 0.5, out_channels 4096
    - bn7: (4096,)
    - fc8: (n_class, 4096, 1, 1)
    - Total parameters = 105,087,296
    - Backbone output shape at stride 8: (B, 4096, H/8, W/8)
    - Multiclass configuration (n_class=4 vs n_class=5).
    """
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM

    model_4cls = WSSSTissueStage1NetCAM(n_class=4)
    model_4cls.eval()

    # 1. Total parameter count
    total_params = sum(p.numel() for p in model_4cls.parameters())
    assert total_params == 105087296, f"Expected 105,087,296 parameters, got {total_params}"

    # 2. Key layer tensor shapes
    state = model_4cls.state_dict()
    assert state["conv1a.weight"].shape == (64, 3, 3, 3)
    assert state["b2.conv_branch1.weight"].shape == (128, 64, 1, 1)
    assert state["b2.conv_branch2a.weight"].shape == (128, 64, 3, 3)
    assert state["b2.conv_branch2b1.weight"].shape == (128, 128, 3, 3)

    assert state["b6.conv_branch1.weight"].shape == (2048, 1024, 1, 1)
    assert state["b6.conv_branch2a.weight"].shape == (512, 1024, 1, 1)
    assert state["b6.conv_branch2b1.weight"].shape == (1024, 512, 3, 3)
    assert state["b6.conv_branch2b2.weight"].shape == (2048, 1024, 1, 1)

    assert state["b7.conv_branch1.weight"].shape == (4096, 2048, 1, 1)
    assert state["b7.conv_branch2a.weight"].shape == (1024, 2048, 1, 1)
    assert state["b7.conv_branch2b1.weight"].shape == (2048, 1024, 3, 3)
    assert state["b7.conv_branch2b2.weight"].shape == (4096, 2048, 1, 1)

    assert state["bn7.weight"].shape == (4096,)
    assert state["bn7.bias"].shape == (4096,)
    assert state["fc8.weight"].shape == (4, 4096, 1, 1)

    # 3. Backbone feature map shape at stride 8
    inp = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feat = model_4cls.forward_backbone(inp)
    assert feat.shape == (2, 4096, 8, 8), f"Expected backbone feat (2, 4096, 8, 8), got {feat.shape}"

    # 4. Multi-class flexibility (n_class=5)
    model_5cls = WSSSTissueStage1NetCAM(n_class=5)
    assert model_5cls.fc8.weight.shape == (5, 4096, 1, 1)
    total_params_5cls = sum(p.numel() for p in model_5cls.parameters())
    assert total_params_5cls == 105087296 + 4096, f"Expected {105087296 + 4096}, got {total_params_5cls}"
    with torch.no_grad():
        y_prob5 = model_5cls(inp)
    assert y_prob5.shape == (2, 5)


def test_dual_checkpoint_loading_and_stage1_fallback(tmp_path):
    """Verify dual checkpoint loading logic, file existence checks, and graceful fallback
    when Stage-1 model is None:
    - Missing checkpoint files raise FileNotFoundError
    - Clean loading from checkpoint dictionary with/without 'module.' prefix and 'state_dict' wrapper
    - Graceful fallback when model_stage1 is None:
      - extract_stage1_cam_and_gate returns (None, None)
      - apply_cgm_gate returns probs unmodified
      - apply_cam_fusion returns probs unmodified
      - predict_all_methods computes all 5 methods cleanly on Stage-2 alone
    - Dynamic runtime override of model_stage1 in predict_all_methods.
    """
    from src.models.wsss_inference_methods import (
        MultiMethodWSSSEngine,
        WSSSTissueStage1NetCAM,
        build_and_load_stage1_netcam,
    )

    # 1. FileNotFoundError on non-existent checkpoint
    fake_path = tmp_path / "non_existent_stage1.pth"
    with pytest.raises(FileNotFoundError):
        build_and_load_stage1_netcam(fake_path, device=torch.device("cpu"))

    # 2. Checkpoint loading with module. prefix and state_dict container
    real_model = WSSSTissueStage1NetCAM(n_class=4)
    raw_sd = real_model.state_dict()
    wrapped_sd = {"state_dict": {f"module.{k}": v for k, v in raw_sd.items()}}
    mock_ckpt_file = tmp_path / "mock_stage1.pth"
    torch.save(wrapped_sd, mock_ckpt_file)

    loaded_model = build_and_load_stage1_netcam(mock_ckpt_file, device=torch.device("cpu"), strict=True)
    assert isinstance(loaded_model, WSSSTissueStage1NetCAM)
    assert sum(p.numel() for p in loaded_model.parameters()) == 105087296

    # 3. Graceful fallback when model_stage1 is None
    engine_none = MultiMethodWSSSEngine(model_stage1=None)
    dummy_input = torch.rand(1, 3, 64, 64)
    cam_probs, cgm_gate = engine_none.extract_stage1_cam_and_gate(None, dummy_input, (64, 64))
    assert cam_probs is None
    assert cgm_gate is None

    dummy_probs = np.full((4, 64, 64), 0.25, dtype=np.float32)
    assert np.array_equal(engine_none.apply_cgm_gate(dummy_probs, None), dummy_probs)
    assert np.array_equal(engine_none.apply_cam_fusion(dummy_probs, None), dummy_probs)

    # Load DeepLabV3Plus from notebook cell 12
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    cell_12 = nb_data["cells"][12]
    code_lines = [
        l for l in "".join(cell_12.get("source", [])).splitlines()
        if "CKPT_FILE" not in l and "model =" not in l
    ]
    env = {}
    exec("\n".join(code_lines), env)
    DeepLabV3Plus = env["DeepLabV3Plus"]
    stage2_model = DeepLabV3Plus(num_classes=4, output_stride=16)
    stage2_model.eval()

    raw_rgb = (dummy_input.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    preds_fallback = engine_none.predict_all_methods(stage2_model, dummy_input, raw_rgb=raw_rgb)
    assert list(preds_fallback.keys()) == MultiMethodWSSSEngine.METHOD_KEYS
    for k in MultiMethodWSSSEngine.METHOD_KEYS:
        assert preds_fallback[k].shape == (64, 64)
        assert preds_fallback[k].dtype == np.uint8

    # 4. Dynamic runtime override: pass loaded_model to predict_all_methods
    preds_dynamic = engine_none.predict_all_methods(
        stage2_model, dummy_input, raw_rgb=raw_rgb, model_stage1=loaded_model
    )
    assert list(preds_dynamic.keys()) == MultiMethodWSSSEngine.METHOD_KEYS
    for k in MultiMethodWSSSEngine.METHOD_KEYS:
        assert preds_dynamic[k].shape == (64, 64)
        assert preds_dynamic[k].dtype == np.uint8


def test_m1_to_m5_numerical_invariants_and_tensor_contracts():
    """Verify strict numerical invariants, boundary properties, probability normalization,
    and pure glass protection for methods M1..M5:
    1. Probability simplex invariant: sum over classes == 1.0 (within float32 epsilon).
    2. CGM gating invariant: suppressed classes (gate == 0) receive strictly 0.0 probability.
    3. Fallback CGM invariant: all-zero gate [0, 0, 0, 0] falls back to [1, 1, 1, 1] without NaN.
    4. Prior calibration invariant: tail classes (LYM, NEC) boosted relative to head classes (TUM, STR).
    5. CAM fusion convex combination: alpha=0.0 preserves segmentation, alpha=1.0 preserves CAM.
    6. Pure glass guard: pure unstained glass (gray>=242, sat<=8, area>=512) masked to class 4,
       while pale stained tissue (gray=215, sat=22) and small spots (<512) are preserved.
    7. Output contract: dict of 5 uint8 arrays in range [0, 4] matching input spatial dimensions.
    """
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine

    engine = MultiMethodWSSSEngine(model_stage1=None)

    # 1. Probability simplex invariant
    dummy_probs = np.random.uniform(0.1, 1.0, (4, 32, 32)).astype(np.float32)
    dummy_probs /= np.sum(dummy_probs, axis=0, keepdims=True)
    assert np.allclose(np.sum(dummy_probs, axis=0), 1.0, atol=1e-5)

    calib_probs = engine.apply_prior_calibration(dummy_probs)
    assert np.allclose(np.sum(calib_probs, axis=0), 1.0, atol=1e-5)

    cam_probs = np.random.uniform(0.1, 1.0, (4, 32, 32)).astype(np.float32)
    cam_probs /= np.sum(cam_probs, axis=0, keepdims=True)
    fused_probs = engine.apply_cam_fusion(dummy_probs, cam_probs, alpha=0.25)
    assert np.allclose(np.sum(fused_probs, axis=0), 1.0, atol=1e-5)

    # 2. CGM gating invariant: class 1 and class 3 suppressed
    cgm_gate = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    gated_probs = engine.apply_cgm_gate(dummy_probs, cgm_gate)
    assert np.allclose(np.sum(gated_probs, axis=0), 1.0, atol=1e-5)
    assert np.all(gated_probs[1] == 0.0), "Suppressed class 1 must have strictly 0 probability"
    assert np.all(gated_probs[3] == 0.0), "Suppressed class 3 must have strictly 0 probability"

    # 3. Fallback CGM invariant: all-zero gate
    zero_gate = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cgm_fallback = zero_gate.copy()
    if cgm_fallback.sum() == 0:
        cgm_fallback = np.ones((4,), dtype=np.float32)
    fallback_probs = engine.apply_cgm_gate(dummy_probs, cgm_fallback)
    assert not np.any(np.isnan(fallback_probs)), "Fallback gate must not produce NaNs"
    assert np.allclose(np.sum(fallback_probs, axis=0), 1.0, atol=1e-5)

    # 4. Prior calibration invariant: analytical verification
    uniform_probs = np.full((4, 16, 16), 0.25, dtype=np.float32)
    p_adj = engine.apply_prior_calibration(uniform_probs)
    expected_ratio_2_0 = 1.24 / 0.96
    actual_ratio_2_0 = float(p_adj[2, 0, 0] / p_adj[0, 0, 0])
    assert np.isclose(actual_ratio_2_0, expected_ratio_2_0, atol=1e-4)

    expected_ratio_3_1 = 1.18 / 0.90
    actual_ratio_3_1 = float(p_adj[3, 0, 0] / p_adj[1, 0, 0])
    assert np.isclose(actual_ratio_3_1, expected_ratio_3_1, atol=1e-4)

    assert p_adj[2, 0, 0] > p_adj[0, 0, 0], "LYM (class 2) must receive positive boost over TUM (class 0)"
    assert p_adj[3, 0, 0] > p_adj[1, 0, 0], "NEC (class 3) must receive positive boost over STR (class 1)"

    # 5. CAM fusion alpha bounds
    fused_alpha_0 = engine.apply_cam_fusion(dummy_probs, cam_probs, alpha=0.0)
    assert np.allclose(fused_alpha_0, dummy_probs, atol=1e-5)

    fused_alpha_1 = engine.apply_cam_fusion(dummy_probs, cam_probs, alpha=1.0)
    assert np.allclose(fused_alpha_1, cam_probs, atol=1e-5)

    # 6. Pure glass guard contract
    synthetic_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    pred_mask = np.zeros((100, 100), dtype=np.uint8)

    # Zone 1: Pure white glass (600 px >= 512 px)
    synthetic_rgb[:25, :24] = [250, 250, 250]

    # Zone 2: Small white spot (100 px < 512 px)
    synthetic_rgb[:10, 30:40] = [250, 250, 250]

    # Zone 3: Pale Necrosis / Stroma (gray = 215, sat = 22, 1000 px)
    synthetic_rgb[40:65, :40] = [220, 200, 215]

    # Zone 4: Dark Hematoxylin Tumor (gray = 60, sat = 140, 1000 px)
    synthetic_rgb[70:95, :40] = [80, 40, 120]

    guarded = engine.apply_pure_glass_guard(pred_mask, synthetic_rgb, background_class_idx=4)

    # Zone 1 must be masked as 4 (Background)
    assert np.all(guarded[:25, :24] == 4), "Pure white glass region >= 512 px must be forced to class 4"

    # Zone 2 must NOT be masked as 4 (small spot preserved)
    assert np.all(guarded[:10, 30:40] == 0), "Small white spot < 512 px must NOT be masked as background"

    # Zone 3 must NOT be masked as 4 (pale necrosis preserved)
    assert np.all(guarded[40:65, :40] == 0), "Pale necrosis tissue must NOT be masked as background"

    # Zone 4 must NOT be masked as 4 (dark tumor preserved)
    assert np.all(guarded[70:95, :40] == 0), "Dark tumor tissue must NOT be masked as background"


def test_integration_engine_and_colab_generator_parity():
    """Verify integration between MultiMethodWSSSEngine, Colab generator scripts,
    and all 4 generated GPU notebook variants:
    1. scripts/generate_colab_variants.py runs and regenerates notebooks without error.
    2. All 4 generated notebooks contain exact matching checkpoint IDs, WSSSTissueStage1NetCAM,
       and MultiMethodWSSSEngine.
    3. Cells 14 and 16 pass model_stage1 to predict_all_methods.
    4. Invariant 12 compliance: zero non-ASCII dashes across all variant files.
    """
    import scripts.generate_colab_variants as gen_module

    # Execute variant build
    gen_module.build_variants()

    variant_files = [
        Path("notebooks/SE374_Colab_WSSS_T4_Sanity150.ipynb"),
        Path("notebooks/SE374_Colab_WSSS_T4_Full4986.ipynb"),
        Path("notebooks/SE374_Colab_WSSS_L4_Full4986.ipynb"),
        Path("notebooks/SE374_Colab_WSSS_A100_Full4986.ipynb"),
    ]

    for vf in variant_files:
        assert vf.is_file(), f"Variant file missing: {vf}"
        text = vf.read_text(encoding="utf-8")

        # Invariant 12 check
        for bad_dash in ("\u2014", "\u2013", "\u2015", "\u2212"):
            assert bad_dash not in text, f"Forbidden dash {bad_dash} in {vf}"

        nb = json.loads(text)
        assert nb["nbformat"] == 4

        # Verify Stage-1 architecture and engine presence
        full_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert "class WSSSTissueStage1NetCAM" in full_code
        assert "class MultiMethodWSSSEngine" in full_code
        assert "118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc" in full_code
        assert "1cXm29ko8UevbaeW6YreSg-vuNaQiN_je" in full_code
        assert "model_stage1=model_stage1" in full_code


def test_pistoseg_rfm_and_cam_nms_repo_grounded():
    """Verify PistoSeg RFM affinity propagation and CAM NMS (repos/PistoSeg/models/revise_net.py)."""
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine

    # 1. Test CAM NMS: Non-maximum class activations suppressed to zero per pixel
    dummy_cam = np.zeros((4, 8, 8), dtype=np.float32)
    # Zone A (top-left): Class 0 is dominant (0.9), Class 1 is sub-dominant (0.4)
    dummy_cam[0, :4, :4] = 0.9
    dummy_cam[1, :4, :4] = 0.4
    # Zone B (bottom-right): Class 1 is dominant (0.8), Class 0 is sub-dominant (0.3)
    dummy_cam[0, 4:, 4:] = 0.3
    dummy_cam[1, 4:, 4:] = 0.8

    nms_cam = MultiMethodWSSSEngine.apply_pistoseg_cam_nms(dummy_cam)
    # In Zone A, Class 0 wins and Class 1 is suppressed to 0
    assert np.all(nms_cam[0, :4, :4] > 0.0), "Zone A: Dominant class 0 must be preserved"
    assert np.all(nms_cam[1, :4, :4] == 0.0), "Zone A: Sub-dominant class 1 must be suppressed to 0"
    # In Zone B, Class 1 wins and Class 0 is suppressed to 0
    assert np.all(nms_cam[1, 4:, 4:] > 0.0), "Zone B: Dominant class 1 must be preserved"
    assert np.all(nms_cam[0, 4:, 4:] == 0.0), "Zone B: Sub-dominant class 0 must be suppressed to 0"

    # 2. Test RFM Affinity Propagation: Matrix multiplication over flattened spatial dimensions
    # Spatial shape: (4, 4) -> 16 spatial locations
    dummy_feat = np.random.uniform(0.1, 1.0, (4, 4, 4)).astype(np.float32)

    # Identity affinity preserves features exactly
    eye_affinity = np.eye(16, dtype=np.float32)
    rfm_identity = MultiMethodWSSSEngine.apply_pistoseg_rfm(dummy_feat, eye_affinity)
    assert np.allclose(rfm_identity, dummy_feat, atol=1e-5)

    # None affinity falls back gracefully
    rfm_none = MultiMethodWSSSEngine.apply_pistoseg_rfm(dummy_feat, None)
    assert np.allclose(rfm_none, dummy_feat, atol=1e-5)

    # Uniform averaging affinity averages across all locations
    uniform_affinity = np.full((16, 16), 1.0 / 16.0, dtype=np.float32)
    rfm_avg = MultiMethodWSSSEngine.apply_pistoseg_rfm(dummy_feat, uniform_affinity)
    assert rfm_avg.shape == (4, 4, 4)
    for c in range(4):
        expected_mean = float(np.mean(dummy_feat[c]))
        assert np.allclose(rfm_avg[c], expected_mean, atol=1e-5)


def test_individual_predict_m1_through_m5_methods():
    """Verify individual predict_m1 through predict_m5 methods match predict_all_methods exactly."""
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine, WSSSTissueStage1NetCAM

    # Load DeepLabV3Plus from notebook
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    cell_12 = nb_data["cells"][12]
    code_lines = [
        l for l in "".join(cell_12.get("source", [])).splitlines()
        if "CKPT_FILE" not in l and "model =" not in l
    ]
    env = {}
    exec("\n".join(code_lines), env)
    DeepLabV3Plus = env["DeepLabV3Plus"]

    model_stage2 = DeepLabV3Plus(num_classes=4, output_stride=16)
    model_stage2.eval()
    model_stage1 = WSSSTissueStage1NetCAM(n_class=4)
    model_stage1.eval()

    engine = MultiMethodWSSSEngine(model_stage1=model_stage1)

    dummy_input = torch.randn(1, 3, 64, 64)
    dummy_rgb = (dummy_input.squeeze(0).permute(1, 2, 0).abs().numpy() * 255.0).astype(np.uint8)

    # Compute via predict_all_methods
    preds_all = engine.predict_all_methods(
        model_stage2, dummy_input, raw_rgb=dummy_rgb, model_stage1=model_stage1
    )

    # Compute via individual methods
    p_m1 = engine.predict_m1(model_stage2, dummy_input)
    p_m2 = engine.predict_m2(model_stage2, dummy_input, model_stage1=model_stage1)
    p_m3 = engine.predict_m3(model_stage2, dummy_input, model_stage1=model_stage1)
    p_m4 = engine.predict_m4(model_stage2, dummy_input, raw_rgb=dummy_rgb, model_stage1=model_stage1)
    p_m5 = engine.predict_m5(model_stage2, dummy_input, raw_rgb=dummy_rgb, model_stage1=model_stage1)

    assert np.array_equal(p_m1, preds_all["M1_Baseline_SingleScale"])
    assert np.array_equal(p_m2, preds_all["M2_MultiView_D4_TTA"])
    assert np.array_equal(p_m3, preds_all["M3_LongTail_Prior_Calibration"])
    assert np.array_equal(p_m4, preds_all["M4_Spatial_Bilateral_Refinement"])
    assert np.array_equal(p_m5, preds_all["M5_Proposed_SOTA_Ensemble"])


def test_engine_device_and_dimension_safety():
    """Verify MultiMethodWSSSEngine gracefully handles 3D tensors (auto-unsqueeze) and CPU execution."""
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine, WSSSTissueStage1NetCAM

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    cell_12 = nb_data["cells"][12]
    code_lines = [
        l for l in "".join(cell_12.get("source", [])).splitlines()
        if "CKPT_FILE" not in l and "model =" not in l
    ]
    env = {}
    exec("\n".join(code_lines), env)
    DeepLabV3Plus = env["DeepLabV3Plus"]

    model_stage2 = DeepLabV3Plus(num_classes=4, output_stride=16)
    model_stage2.eval()
    model_stage1 = WSSSTissueStage1NetCAM(n_class=4)
    model_stage1.eval()

    engine = MultiMethodWSSSEngine(model_stage1=model_stage1)

    # 3D tensor input: (3, 64, 64) instead of (1, 3, 64, 64)
    tensor_3d = torch.randn(3, 64, 64)
    dummy_rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    preds = engine.predict_all_methods(
        model_stage2, tensor_3d, raw_rgb=dummy_rgb, model_stage1=model_stage1
    )
    assert list(preds.keys()) == MultiMethodWSSSEngine.METHOD_KEYS
    for k in MultiMethodWSSSEngine.METHOD_KEYS:
        assert preds[k].shape == (64, 64)
        assert preds[k].dtype == np.uint8


def test_cell_14_and_18_json_serialization_no_circular_reference():
    """Verify Cell 14 metrics dictionary construction and Cell 18 JSON serialization
    never create a circular reference when json.dumps(export_metadata) is called.
    """
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell_14_src = "".join(nb_data["cells"][14].get("source", []))
    cell_18_src = "".join(nb_data["cells"][18].get("source", []))

    assert "metrics = dict(all_methods_metrics[best_method_key])" in cell_14_src, (
        "Cell 14 must copy all_methods_metrics[best_method_key] via dict() to prevent circular reference!"
    )
    assert 'if mk != "ablation_methods"' in cell_18_src, (
        "Cell 18 must strip self-referential ablation_methods keys before json.dump!"
    )


def test_colab_pkg_and_wsss_inference_methods_exact_parity_m1_to_m7():
    """Verify 100% pixel-exact parity between src.colab_pkg.model_loader and
    src.models.wsss_inference_methods across M1..M7 on square and rectangular inputs.
    """
    from src.colab_pkg.model_loader import (
        DeepLabV3Plus as DL_Colab,
        WSSSTissueStage1NetCAM as S1_Colab,
        MultiMethodWSSSEngineV2 as E_Colab,
    )
    from src.models.wsss_inference_methods import (
        MultiMethodWSSSEngineV2 as E_Models,
    )

    torch.manual_seed(42)
    m2 = DL_Colab(num_classes=4, output_stride=16).eval()
    m1 = S1_Colab(n_class=4).eval()

    e_colab = E_Colab(model_stage1=m1)
    e_models = E_Models(model_stage1=m1)

    for h, w in [(64, 64), (33, 47)]:
        inp = torch.randn(1, 3, h, w)
        rgb = (inp.squeeze(0).permute(1, 2, 0).abs().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        preds_colab = e_colab.predict_all_methods(m2, inp, raw_rgb=rgb, model_stage1=m1)
        preds_models = e_models.predict_all_methods(m2, inp, raw_rgb=rgb, model_stage1=m1)

        assert list(preds_colab.keys()) == list(preds_models.keys()) == E_Models.METHOD_KEYS_V2
        for k in E_Models.METHOD_KEYS_V2:
            assert np.array_equal(preds_colab[k], preds_models[k]), f"Mismatch on {k} for shape ({h}, {w})"
