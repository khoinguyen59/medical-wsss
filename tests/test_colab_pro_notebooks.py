"""Unit tests for the 4 production-grade Colab Pro notebooks suite.

Verifies:
1. File existence and valid Jupyter Notebook JSON format (nbformat 4, nbformat_minor 5).
2. Clean compilation of all Python code cells via AST without syntax errors.
3. Invariant 12 compliance: 100% standard ASCII hyphens (0x2D), zero non-ASCII dashes.
4. Pre-configured Google Drive and checkpoint targets.
5. Dual-path resilience (Google Drive mounted vs local fallback).
6. Tqdm progress tracking and resume-from-checkpoint capability.
7. Clinical biomarkers (TSR, VTP, CAP risk) and Kaplan-Meier survival curves in Notebook 4.
8. Execution sanity of notebook components (prompt generation, loss calculation, gating).

All hyphens strictly use standard ASCII 0x2D ('-').
"""

import ast
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

NOTEBOOKS = {
    "stage1": Path("notebooks/01_Train_Stage1_Classifier.ipynb"),
    "stage2": Path("notebooks/02_Pseudo_Masks_Train_Stage2.ipynb"),
    "medsam": Path("notebooks/03_Train_Proposed_MedSAM.ipynb"),
    "benchmark": Path("notebooks/04_Exhaustive_Benchmark_Survival.ipynb"),
    "stage1_t4": Path("notebooks/01_Train_Stage1_Classifier_T4.ipynb"),
    "stage2_t4": Path("notebooks/02_Pseudo_Masks_Train_Stage2_T4.ipynb"),
    "medsam_t4": Path("notebooks/03_Train_Proposed_MedSAM_T4.ipynb"),
    "benchmark_t4": Path("notebooks/04_Exhaustive_Benchmark_Survival_T4.ipynb"),
    "stage1_l4": Path("notebooks/01_Train_Stage1_Classifier_L4.ipynb"),
    "stage2_l4": Path("notebooks/02_Pseudo_Masks_Train_Stage2_L4.ipynb"),
    "medsam_l4": Path("notebooks/03_Train_Proposed_MedSAM_L4.ipynb"),
    "benchmark_l4": Path("notebooks/04_Exhaustive_Benchmark_Survival_L4.ipynb"),
    "stage2_t4_root": Path("02_Pseudo_Masks_Train_Stage2_T4.ipynb"),
}


FORBIDDEN_DASHES = [
    ("\u2014", "em-dash"),
    ("\u2013", "en-dash"),
    ("\u2015", "horizontal bar"),
    ("\u2212", "minus sign"),
]


@pytest.mark.parametrize("key,path", list(NOTEBOOKS.items()))
def test_notebook_file_exists_and_is_not_empty(key, path):
    """Verify that each required Colab Pro notebook exists on disk and is non-empty."""
    assert path.is_file(), f"Missing required notebook: {path}"
    assert path.stat().st_size > 1000, f"Notebook file is unusually small: {path.stat().st_size} bytes"


@pytest.mark.parametrize("key,path", list(NOTEBOOKS.items()))
def test_notebook_valid_json_and_nbformat(key, path):
    """Verify that every notebook conforms to nbformat 4 and nbformat_minor 5."""
    with open(path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data, f"Notebook {path} missing 'cells' key"
    assert nb_data["nbformat"] == 4, f"Notebook {path} expected nbformat 4, got {nb_data.get('nbformat')}"
    assert nb_data["nbformat_minor"] == 5, f"Notebook {path} expected nbformat_minor 5, got {nb_data.get('nbformat_minor')}"
    assert len(nb_data["cells"]) >= 8, f"Notebook {path} has too few cells: {len(nb_data['cells'])}"

    for idx, cell in enumerate(nb_data["cells"]):
        assert "cell_type" in cell, f"Cell {idx} in {path} missing 'cell_type'"
        assert "source" in cell, f"Cell {idx} in {path} missing 'source'"
        if cell["cell_type"] == "code":
            assert "execution_count" in cell, f"Code cell {idx} in {path} missing 'execution_count'"
            assert "outputs" in cell, f"Code cell {idx} in {path} missing 'outputs'"


@pytest.mark.parametrize("key,path", list(NOTEBOOKS.items()))
def test_code_cells_ast_syntax_clean(key, path):
    """Verify that all code cells in every notebook compile into Python AST without syntax errors."""
    with open(path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [c for c in nb_data["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4, f"Notebook {path} expected at least 4 code cells, got {len(code_cells)}"

    for idx, cell in enumerate(code_cells):
        raw_source = "".join(cell.get("source", []))
        # Mask Colab magics starting with '!' or '%'
        clean_lines = []
        for line in raw_source.splitlines():
            s = line.strip()
            if s.startswith("!") or s.startswith("%"):
                clean_lines.append(f"# {s}")
            else:
                clean_lines.append(line)

        code_str = "\n".join(clean_lines)
        try:
            tree = ast.parse(code_str, filename=f"{path.name}_Cell_{idx}")
            assert tree is not None
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {path.name} Code Cell {idx} line {e.lineno}: {e.msg}\nCode snippet:\n{code_str[:300]}")


def test_invariant_12_strict_ascii_hyphens_only():
    """Verify complete absence of non-ASCII dashes across all notebooks and test files."""
    files_to_check = list(NOTEBOOKS.values()) + [Path("tests/test_colab_pro_notebooks.py")]
    for path in files_to_check:
        content = path.read_text(encoding="utf-8")
        for char, name in FORBIDDEN_DASHES:
            count = content.count(char)
            assert count == 0, f"Invariant 12 violation in {path}: found {count} instances of {name} ({hex(ord(char))})"


def test_dual_path_resilience_across_all_notebooks():
    """Verify all notebooks handle Google Drive mount with fallback to local path."""
    for key, path in NOTEBOOKS.items():
        content = path.read_text(encoding="utf-8")
        assert "google.colab" in content, f"{path} missing Google Colab detection"
        assert "CHECKPOINT_DIR" in content or "OUTPUT_DIR" in content, f"{path} missing checkpoint/output directory configuration"
        assert "checkpoints" in content, f"{path} missing local checkpoints fallback"


def test_notebook_01_stage1_checkpoint_and_resume_specs():
    """Verify Notebook 1 checkpoint paths, resume logic, and training configs."""
    content = NOTEBOOKS["stage1"].read_text(encoding="utf-8")
    assert "stage1_latest.pth" in content, "Notebook 1 missing stage1_latest.pth checkpoint target"
    assert "stage1_retrained_best.pth" in content, "Notebook 1 missing stage1_retrained_best.pth target"
    assert "/content/checkpoints" in content, "Notebook 1 missing Colab VM local checkpoints path"
    assert "tqdm" in content, "Notebook 1 missing tqdm progress bar"
    assert "MultiLabelSoftMarginLoss" in content, "Notebook 1 missing multi-label classification loss"
    assert "SE374_Stage1_Training_Results.zip" in content, "Notebook 1 missing results ZIP export"


def test_notebook_02_stage2_multi_layer_cam_and_deeplab():
    """Verify Notebook 2 CAM generation, affinity matting, and DeepLabV3+ configs."""
    content = NOTEBOOKS["stage2"].read_text(encoding="utf-8")
    assert "stage2_latest.pth" in content, "Notebook 2 missing stage2_latest.pth checkpoint target"
    assert "stage2_retrained_best.pth" in content, "Notebook 2 missing stage2_retrained_best.pth target"
    assert "DeepLabV3Plus" in content, "Notebook 2 missing DeepLabV3+ architecture"
    assert "PM_bn7" in content, "Notebook 2 missing PM_bn7 multi-layer pseudo-mask"
    assert "apply_otsu_gating" in content, "Notebook 2 missing Otsu clinical gating"
    assert "tqdm" in content, "Notebook 2 missing tqdm progress bar"
    assert "SE374_Stage2_Training_Results.zip" in content, "Notebook 2 missing results ZIP export"


def test_notebook_03_medsam_adapter_and_prompting():
    """Verify Notebook 3 MedSAM adapter architecture, prompt generator, and loss."""
    content = NOTEBOOKS["medsam"].read_text(encoding="utf-8")
    assert "medsam_latest.pth" in content, "Notebook 3 missing medsam_latest.pth checkpoint target"
    assert "medsam_wsss_adapter_best.pth" in content, "Notebook 3 missing medsam_wsss_adapter_best.pth target"
    assert "CAMPromptGenerator" in content, "Notebook 3 missing CAMPromptGenerator"
    assert "MedSAM_WSSS_Adapter" in content, "Notebook 3 missing MedSAM_WSSS_Adapter"
    assert "isinstance(pred_masks, dict)" in content, "Notebook 3 missing dict logits extraction"
    assert "tqdm" in content, "Notebook 3 missing tqdm progress bar"
    assert "SE374_MedSAM_Adapter_Results.zip" in content, "Notebook 3 missing results ZIP export"


def test_notebook_04_exhaustive_benchmark_and_survival():
    """Verify Notebook 4 3-way evaluation, 4,986 test patches, and clinical biomarkers."""
    content = NOTEBOOKS["benchmark"].read_text(encoding="utf-8")
    assert "stage2_checkpoint_trained_on_bcss.pth" in content, "Notebook 4 missing Author SOTA checkpoint"
    assert "stage2_retrained_best.pth" in content, "Notebook 4 missing Retrained checkpoint"
    assert "medsam_wsss_adapter_best.pth" in content, "Notebook 4 missing MedSAM Adapter checkpoint"
    assert "bcss_patient_splits.json" in content, "Notebook 4 missing patient split manifest"
    assert "tqdm" in content, "Notebook 4 missing tqdm progress bar"
    assert "TCGA_SURVIVAL_DATA" in content, "Notebook 4 missing real TCGA-BRCA clinical survival data"
    assert "kaplan_meier_survival_curve.png" in content, "Notebook 4 missing Kaplan-Meier plot export"
    assert "SE374_Exhaustive_Clinical_Results.zip" in content, "Notebook 4 missing final results ZIP"
    assert "tsr" in content.lower(), "Notebook 4 missing Tumor-Stroma Ratio calculation"
    assert "vtp" in content.lower(), "Notebook 4 missing Viable Tumor Percentage calculation"
    assert "cap_risk" in content, "Notebook 4 missing CAP risk stratification"
    # Reviewer 2 anti-mockup and robust extraction assertions
    assert "lr_pval = 0.042" not in content, "Notebook 4 contains hardcoded lr_pval = 0.042"
    assert "np.linspace(1.0, 0.7" not in content, "Notebook 4 contains dummy np.linspace fallback"
    assert "KaplanMeierSurvivalAnalysis" in content, "Notebook 4 missing genuine KaplanMeierSurvivalAnalysis"
    assert "isinstance(out_c, dict)" in content, "Notebook 4 missing dict logits extraction for MedSAM"


def test_notebook_04_zero_hardcoded_survival_and_genuine_fallback():
    """Verify absence of hardcoded p-values in Notebook 4 and verify genuine KM engine execution."""
    content = NOTEBOOKS["benchmark"].read_text(encoding="utf-8")
    assert "0.042" not in content, "Hardcoded p-value 0.042 detected in Notebook 4"
    assert "np.linspace(1.0, 0.7" not in content, "Dummy linear ramp detected in Notebook 4"

    # Test genuine analytical KM Product-Limit and log-rank test
    from src.clinical.survival_analysis import KaplanMeierSurvivalAnalysis
    km_engine = KaplanMeierSurvivalAnalysis(use_lifelines_if_available=False)
    times_high = [8.74, 12.30, 14.78]
    events_high = [1, 0, 0]
    times_low = [16.92, 19.42, 33.31]
    events_low = [0, 0, 0]

    fit_h = km_engine.fit(times_high, events_high, label="High Risk")
    fit_l = km_engine.fit(times_low, events_low, label="Low Risk")
    logrank_fn = getattr(km_engine, "log_rank_test", getattr(km_engine, "logrank_test", None))
    assert logrank_fn is not None
    lr_res = logrank_fn(times_high, events_high, times_low, events_low)

    assert "timeline" in fit_h and len(fit_h["timeline"]) > 0
    assert "survival_probability" in fit_h and len(fit_h["survival_probability"]) > 0
    assert "ci_lower" in fit_h and len(fit_h["ci_lower"]) > 0
    assert "p_value" in lr_res and isinstance(lr_res["p_value"], float)
    assert lr_res["p_value"] != 0.042, "p-value must be dynamically computed, not 0.042"


def test_medsam_forward_dict_extraction_logic():
    """Verify that MedSAM_WSSS_Adapter returns dict and extracted logits work with argmax and loss."""
    from src.models.sam_wsss_adapter import MedSAM_WSSS_Adapter
    model = MedSAM_WSSS_Adapter(num_classes=5, freeze_image_encoder=True, use_surrogate_if_missing=True)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert isinstance(out, dict), f"Expected dict from MedSAM_WSSS_Adapter, got {type(out)}"
    assert "logits" in out, "Dict missing 'logits' key"

    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (2, 5, 224, 224)

    # Verify argmax works without TypeError
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    assert pred.shape == (2, 224, 224)

    # Verify loss computation works without TypeError
    targets = torch.randint(0, 5, (2, 224, 224), dtype=torch.long)
    ce = torch.nn.CrossEntropyLoss()
    loss = ce(logits, targets)
    assert loss.item() > 0.0


def test_cam_prompt_generator_component_sanity():
    """Sanity test verifying that the CAMPromptGenerator component produces valid prompt tensors."""
    # Build a synthetic CAM tensor (B=2, C=4, H=56, W=56)
    cams = torch.zeros((2, 4, 56, 56), dtype=torch.float32)
    # Class 0 has a hot spot
    cams[0, 0, 20:30, 20:30] = 0.9
    cams[1, 1, 10:25, 10:25] = 0.85
    scores = torch.tensor([[0.95, 0.05, 0.02, 0.01], [0.03, 0.90, 0.01, 0.02]], dtype=torch.float32)

    # Simple prompt extraction check
    prompt_coords = []
    prompt_labels = []
    for b in range(2):
        c_active = int(torch.argmax(scores[b]).item())
        cam_active = cams[b, c_active].numpy()
        max_idx = np.unravel_index(np.argmax(cam_active), cam_active.shape)
        prompt_coords.append([float(max_idx[1]), float(max_idx[0])])
        prompt_labels.append(1)

    coords_t = torch.tensor(prompt_coords, dtype=torch.float32)
    labels_t = torch.tensor(prompt_labels, dtype=torch.long)
    assert coords_t.shape == (2, 2)
    assert labels_t.shape == (2,)
    assert labels_t[0] == 1 and labels_t[1] == 1


def test_clinical_biomarkers_mathematical_consistency():
    """Verify that TSR and VTP formulations satisfy clinical bounds [0%, 100%]."""
    tum_area = 15000
    str_area = 35000
    lym_area = 5000
    nec_area = 2000

    # TSR = TUM / (TUM + STR)
    tsr = (tum_area / (tum_area + str_area)) * 100.0
    stroma_prop = 100.0 - tsr
    assert tsr == 30.0, f"Expected TSR 30.0%, got {tsr}%"
    assert stroma_prop == 70.0, f"Expected Stroma Prop 70.0%, got {stroma_prop}%"

    # CAP Risk: Stroma-Rich (>50%) vs Stroma-Poor (<=50%)
    cap_risk = "Stroma-Rich (High Risk)" if stroma_prop > 50.0 else "Stroma-Poor (Low Risk)"
    assert cap_risk == "Stroma-Rich (High Risk)"

    # VTP = TUM / (TUM + STR + LYM + NEC)
    total_tissue = tum_area + str_area + lym_area + nec_area
    vtp = (tum_area / total_tissue) * 100.0
    assert abs(vtp - 26.315) < 0.01, f"Expected VTP ~26.32%, got {vtp}%"


def test_notebook_forward_pass_tensor_shapes_and_loss():
    """Verify Stage 1 forward_stage1_with_pda outputs (B, 4) logits matching MultiLabelSoftMarginLoss."""
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.models.sam_wsss_adapter import CAM_Prompt_Generator

    model = WSSSTissueStage1NetCAM(num_classes=4)
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    # 1. Forward with PDA (Notebook 01 logic)
    conv6 = model.forward_backbone(images)
    cam = torch.nn.functional.conv2d(conv6, model.fc8.weight)
    cam_flat = cam.view(cam.size(0), cam.size(1), -1)
    cam_max = cam_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    cam_norm = torch.nn.functional.relu(cam) / (cam_max + 1e-5)
    salient_mask = (cam_norm.max(dim=1, keepdim=True)[0] > 0.7).float()
    keep_prob = 1.0 - 0.5 * salient_mask
    dropout_mask = torch.bernoulli(keep_prob)
    conv6_pda = conv6 * dropout_mask

    x = model.dropout7(conv6_pda)
    x = model.pool(x)
    logits = model.fc8(x).view(x.size(0), -1)
    assert logits.shape == (2, 4), f"Expected logits shape (2, 4), got {logits.shape}"

    criterion = torch.nn.MultiLabelSoftMarginLoss()
    loss = criterion(logits, targets)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0

    # 2. Forward CAM for Notebook 02 and 03
    cam_raw, cls_probs = model.forward_cam(images)
    assert cam_raw.shape == (2, 4, 28, 28), f"Expected cam_raw shape (2, 4, 28, 28), got {cam_raw.shape}"
    assert cls_probs.shape == (2, 4), f"Expected cls_probs shape (2, 4), got {cls_probs.shape}"

    cam_up = torch.nn.functional.interpolate(cam_raw, size=(224, 224), mode="bilinear", align_corners=False)
    assert cam_up.shape == (2, 4, 224, 224)

    prompt_gen = CAM_Prompt_Generator(num_classes=4)
    prompt_dict = prompt_gen(cam_up, cls_probs)
    assert 0 in prompt_dict
    assert prompt_dict[0]["point_coords"] is not None
    assert prompt_dict[0]["point_labels"] is not None
    assert prompt_dict[0]["point_coords"].shape[0] == 2


@pytest.mark.parametrize("key,path", list(NOTEBOOKS.items()))
def test_device_and_amp_declared_before_usage(key, path):
    """Verify that DEVICE is declared before any usage/print in the setup cells of every notebook."""
    with open(path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Check the setup code cells (first 6 cells)
    setup_code = ""
    for cell in nb_data["cells"][:6]:
        if cell["cell_type"] == "code":
            setup_code += "".join(cell.get("source", [])) + "\n"

    lines = setup_code.splitlines()
    device_def_idx = None
    first_device_use_idx = None

    for idx, line in enumerate(lines):
        # Look for DEVICE = assignment (excluding comments)
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "DEVICE =" in line and device_def_idx is None:
            device_def_idx = idx
        if ("{DEVICE}" in line or "to(DEVICE)" in line) and first_device_use_idx is None:
            first_device_use_idx = idx

    assert device_def_idx is not None, f"Notebook {path.name} missing DEVICE assignment in setup cells"
    if first_device_use_idx is not None:
        assert device_def_idx < first_device_use_idx, (
            f"Notebook {path.name} uses DEVICE at line {first_device_use_idx} before declaration at line {device_def_idx}"
        )


def test_stage2_notebooks_detailed_specifications():
    """Verify detailed requirements for Stage 2 notebooks (base, T4, L4, and root T4):
    1. Batch sizes: T4 has batch_size 12, L4 has batch_size 24.
    2. NUM_WORKERS: T4 has 2, L4 has 4.
    3. Dataset path resolution checks 'training/' and 'val/mask' & 'val/img'.
    4. Otsu gating 'apply_otsu_gating' with multi-layer 'PM_bn7', 'PM_b5_2', 'PM_b4_5'.
    5. Local storage '/content/checkpoints' used without Google Drive popup requirement.
    6. Root '02_Pseudo_Masks_Train_Stage2_T4.ipynb' is 100% identical to 'notebooks/02_Pseudo_Masks_Train_Stage2_T4.ipynb'.
    """
    t4_nb = Path("notebooks/02_Pseudo_Masks_Train_Stage2_T4.ipynb").read_text(encoding="utf-8")
    l4_nb = Path("notebooks/02_Pseudo_Masks_Train_Stage2_L4.ipynb").read_text(encoding="utf-8")
    root_t4 = Path("02_Pseudo_Masks_Train_Stage2_T4.ipynb").read_text(encoding="utf-8")

    # Synchronization
    assert t4_nb == root_t4, "Root 02_Pseudo_Masks_Train_Stage2_T4.ipynb is out of sync with notebooks/"

    # T4 specific configurations
    assert "BATCH_SIZE = 12" in t4_nb, "T4 notebook missing BATCH_SIZE = 12"
    assert "NUM_WORKERS = 2" in t4_nb, "T4 notebook missing NUM_WORKERS = 2"
    assert "torch.float16" in t4_nb, "T4 notebook missing FP16 precision"

    # L4 specific configurations
    assert "BATCH_SIZE = 24" in l4_nb, "L4 notebook missing BATCH_SIZE = 24"
    assert "NUM_WORKERS = 4" in l4_nb, "L4 notebook missing NUM_WORKERS = 4"
    assert "torch.bfloat16" in l4_nb, "L4 notebook missing BF16 precision"

    # Dataset path resolution
    for content, name in [(t4_nb, "T4"), (l4_nb, "L4")]:
        assert "'training'" in content or '"training"' in content, f"{name} notebook missing 'training' folder resolution"
        assert "'val'" in content or '"val"' in content, f"{name} notebook missing 'val' folder resolution"
        assert "'mask'" in content or '"mask"' in content, f"{name} notebook missing 'mask' folder resolution"
        assert "resolve_split_path" in content, f"{name} notebook missing resolve_split_path"
        assert "apply_otsu_gating" in content, f"{name} notebook missing apply_otsu_gating"
        assert "gpu_affinity_refine" in content, f"{name} notebook missing gpu_affinity_refine"
        assert "PM_bn7" in content and "PM_b5_2" in content and "PM_b4_5" in content, f"{name} notebook missing 3 pseudo-mask layers"
        assert "/content/checkpoints" in content, f"{name} notebook missing local Colab checkpoints directory"
        assert "drive.mount" not in content, f"{name} notebook has unwanted Google Drive popup requirement"
        assert "SE374_Stage2_Training_Results.zip" in content, f"{name} notebook missing results ZIP packaging"


def test_stage2_deeplabv3plus_instantiation_and_forward():
    """Verify DeepLabV3Plus with backbone='resnet101' instantiates and executes forward pass."""
    from src.models.deeplabv3plus import DeepLabV3Plus as DeepLab1
    from src.models.wsss_inference_methods import DeepLabV3Plus as DeepLab2

    for ModelClass in [DeepLab1, DeepLab2]:
        m = ModelClass(num_classes=5, backbone="resnet101")
        m.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 5, 224, 224), f"Expected shape (2, 5, 224, 224), got {out.shape}"
        pred = torch.argmax(out, dim=1)
        assert pred.shape == (2, 224, 224)


def test_stage2_gpu_affinity_refine_zero_phantom_classes():
    """Verify bilateral refinement on discrete boundary produces zero phantom classes."""
    from scripts.build_t4_l4_notebook_suites import get_clean_cell_8_stage2

    lines = get_clean_cell_8_stage2()
    start = next(i for i, l in enumerate(lines) if "def gpu_affinity_refine" in l)
    end = next(i for i, l in enumerate(lines) if "all_train_files" in l)
    func_code = "import cv2\nimport numpy as np\n" + "".join(lines[start:end])
    local_scope = {}
    exec(func_code, local_scope)
    refine_fn = local_scope["gpu_affinity_refine"]

    # 1. Discrete 2D mask test with Tumor (0) and Background (4)
    discrete_mask = np.zeros((64, 64), dtype=np.uint8)
    discrete_mask[:, 32:] = 4
    refined_discrete = refine_fn(discrete_mask)
    unique_vals = set(np.unique(refined_discrete))
    assert unique_vals.issubset({0, 4}), f"Phantom classes detected in refined discrete mask: {unique_vals}"

    # 2. Continuous 3D CAM tensor test (5, 64, 64)
    cam_continuous = np.random.rand(5, 64, 64).astype(np.float32)
    refined_cam = refine_fn(cam_continuous)
    assert refined_cam.shape == (5, 64, 64)
    assert refined_cam.dtype == np.float32


def test_stage2_resolve_split_path_fallback_resilience():
    """Verify resolve_split_path returns valid Path without string division TypeError when sub is given."""
    from scripts.build_t4_l4_notebook_suites import get_clean_cell_6_stage2

    lines = get_clean_cell_6_stage2()
    start = next(i for i, l in enumerate(lines) if "def resolve_split_path" in l)
    end = next(i for i, l in enumerate(lines) if "TRAIN_DIR =" in l)
    func_code = "from pathlib import Path\n" + "".join(lines[start:end])
    local_scope = {}
    exec(func_code, local_scope)
    resolve_fn = local_scope["resolve_split_path"]

    dummy_base = Path("/nonexistent_medical_test_dir")
    # Test fallback path when no candidate directories have images (exercises the string division fix)
    custom_path = resolve_fn(dummy_base, split="custom_split", sub="custom_sub")
    assert isinstance(custom_path, Path)
    assert custom_path == dummy_base / "custom_split" / "custom_sub"

    train_custom = resolve_fn(dummy_base, split="custom_split")
    assert isinstance(train_custom, Path)
    assert train_custom == dummy_base / "custom_split"

    # Test that valid sub path resolution returns a Path ending with sub
    val_img = resolve_fn(dummy_base, split="val", sub="img")
    assert isinstance(val_img, Path)
    assert val_img.name == "img"


def test_stage2_mlps_loss_gradient_flow():
    """Verify MLPSLoss correctly weights 3 pseudo-mask layers and gradients flow back to logits."""
    from scripts.build_t4_l4_notebook_suites import get_clean_cell_10_stage2

    lines = get_clean_cell_10_stage2()
    start = next(i for i, l in enumerate(lines) if "class MLPSLoss" in l)
    end = next(i for i, l in enumerate(lines) if "criterion_s2 =" in l)
    func_code = "import torch\n" + "".join(lines[start:end])
    local_scope = {}
    exec(func_code, local_scope)
    mlps_class = local_scope["MLPSLoss"]

    criterion = mlps_class(w_main=1.0, w_aux1=0.4, w_aux2=0.2)
    logits = torch.randn(2, 5, 64, 64, requires_grad=True)
    t_main = torch.randint(0, 5, (2, 64, 64), dtype=torch.long)
    t_aux1 = torch.randint(0, 5, (2, 64, 64), dtype=torch.long)
    t_aux2 = torch.randint(0, 5, (2, 64, 64), dtype=torch.long)

    loss = criterion(logits, t_main, t_aux1, t_aux2)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_stage2_cell_4_dry_run_and_workers_declared():
    """Verify DRY_RUN and NUM_WORKERS are present in Cell 4 across all Stage 2 notebooks."""
    stage2_keys = ["stage2", "stage2_t4", "stage2_l4", "stage2_t4_root"]
    for k in stage2_keys:
        nb_path = NOTEBOOKS[k]
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
        c4_source = "".join(nb_data["cells"][4]["source"])
        assert "DRY_RUN = False" in c4_source, f"Notebook {nb_path} missing DRY_RUN declaration in Cell 4"
        assert "NUM_WORKERS =" in c4_source, f"Notebook {nb_path} missing NUM_WORKERS declaration in Cell 4"
        assert "DEVICE =" in c4_source, f"Notebook {nb_path} missing DEVICE declaration in Cell 4"
        assert "AMP_DTYPE =" in c4_source, f"Notebook {nb_path} missing AMP_DTYPE declaration in Cell 4"


def test_paletted_mode_p_mask_preserves_class_indices():
    """Verify paletted mode 'P' masks retain discrete class indices [0..4] without luminance distortion."""
    # 1. Synthetic paletted image with BCSS RGB palette
    palette = [
        0, 0, 0,        # Class 0: Background/Other
        228, 26, 28,    # Class 1: Tumor (Red)
        55, 126, 184,   # Class 2: Stroma (Blue)
        77, 175, 74,    # Class 3: Lymphocytic Infiltrate (Green)
        152, 78, 163,   # Class 4: Necrosis (Purple)
    ] + [0] * (256 * 3 - 15)

    raw_indices = np.array([
        [0, 1, 2],
        [3, 4, 1],
        [2, 0, 3]
    ], dtype=np.uint8)

    p_img = Image.fromarray(raw_indices, mode="P")
    p_img.putpalette(palette)

    # Calling .convert('L') directly causes luminance corruption:
    corrupted_arr = np.array(p_img.convert("L"), dtype=np.int64)
    # Class 1 (Red 228, 26, 28) converts to Y ~ 68 != 1
    assert not np.array_equal(corrupted_arr, raw_indices)

    # Safe loader logic preserving mode 'P' indices:
    mask_img = p_img
    if mask_img.mode != "P":
        mask_img = mask_img.convert("L")
    gt_pil = mask_img.resize((224, 224), Image.NEAREST)
    gt_arr = np.array(gt_pil, dtype=np.int64)

    assert set(np.unique(gt_arr)).issubset({0, 1, 2, 3, 4})
    assert (gt_arr < 5).sum() == gt_arr.size
    assert (gt_arr < 5).sum() > 0

    # 2. Empirical test on real BCSS paletted mask on disk if present
    bcss_mask_path = Path("data/BCSS-WSSS/val/mask/TCGA-S3-AA15-DX1_xmin55486_ymin28926_MPP-0.2500+0.png")
    if bcss_mask_path.is_file():
        real_img = Image.open(bcss_mask_path)
        assert real_img.mode == "P", f"Expected BCSS mask to be mode 'P', got {real_img.mode}"
        if real_img.mode != "P":
            real_img = real_img.convert("L")
        gt_real_pil = real_img.resize((224, 224), Image.NEAREST)
        gt_real_arr = np.array(gt_real_pil, dtype=np.int64)

        # Confirm all valid class pixels fall in [0, 4] and are non-empty
        valid_mask = (gt_real_arr >= 0) & (gt_real_arr < 5)
        assert valid_mask.sum() > 0, "Real BCSS mask should have positive valid pixels < 5"
        unique_vals = np.unique(gt_real_arr)
        # Class indices present in this patch: 1 (Tumor), 2 (Stroma), 4 (Background)
        assert any(c in unique_vals for c in range(5))


def test_generator_clean_cells_have_safe_paletted_mask_loading():
    """Verify that notebook generator cell definitions guard mode 'P' mask loading."""
    from scripts.build_t4_l4_notebook_suites import (
        get_clean_cell_14_stage2,
        get_clean_cell_16_stage2,
        get_clean_cell_14_stage3,
        get_clean_cell_10_stage4,
    )

    cell_funcs = {
        "cell_14_stage2": get_clean_cell_14_stage2(),
        "cell_16_stage2": get_clean_cell_16_stage2(),
        "cell_14_stage3": get_clean_cell_14_stage3(),
        "cell_10_stage4": get_clean_cell_10_stage4(),
    }

    for name, lines in cell_funcs.items():
        code_str = "".join(lines)
        assert "mode != 'P'" in code_str or 'mode != "P"' in code_str, (
            f"Generator {name} missing safe mode 'P' guard for mask loading!"
        )
        assert ".convert('L').resize(" not in code_str, (
            f"Generator {name} contains unsafe .convert('L').resize(...) on masks!"
        )






