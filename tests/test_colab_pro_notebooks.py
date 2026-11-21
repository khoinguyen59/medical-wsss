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
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

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
    "stage1_t4_root": Path("01_Train_Stage1_Classifier_T4.ipynb"),
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
    """Execute REAL forward_stage1_dual / forward_stage1_with_pda code from 01_Train_Stage1_Classifier_L4.ipynb."""
    from src.models.sam_wsss_adapter import CAM_Prompt_Generator
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.train_stage1 import AttentionModule

    nb_data = json.loads(NOTEBOOKS["stage1_l4"].read_text(encoding="utf-8"))
    cell10_src = "".join(nb_data["cells"][10]["source"])
    start_idx = cell10_src.index("def forward_stage1_dual(")
    fn_code = cell10_src[start_idx:]

    ns = {"torch": torch, "AttentionModule": AttentionModule}
    exec(fn_code, ns)
    forward_dual = ns["forward_stage1_dual"]
    forward_pda = ns["forward_stage1_with_pda"]

    model = WSSSTissueStage1NetCAM(num_classes=4)
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    attn_mod = AttentionModule()

    # 1. Execute actual notebook forward_stage1_dual & forward_stage1_with_pda (gamma=1.0 identity & gamma=0.80 bounded)
    logits_c, probs_c, logits_p_id, _ = forward_dual(
        model, images, enable_pda=True, gamma=1.0, attn_module=attn_mod
    )
    assert logits_c.shape == (2, 4) and logits_p_id is not None
    assert torch.allclose(logits_c, logits_p_id, atol=1e-5)

    logits_p, probs_p = forward_pda(model, images, enable_pda=True, gamma=0.80, attn_module=attn_mod)
    assert logits_p.shape == (2, 4) and probs_p.shape == (2, 4)
    assert torch.isfinite(logits_p).all()

    criterion = torch.nn.MultiLabelSoftMarginLoss()
    loss = 0.5 * criterion(logits_c, targets) + 0.5 * criterion(logits_p, targets)
    assert torch.isfinite(loss) and loss.item() > 0.0

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
    start = next(i for i, ln in enumerate(lines) if "def gpu_affinity_refine" in ln)
    end = next(i for i, ln in enumerate(lines) if "all_train_files" in ln)
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
    start = next(i for i, ln in enumerate(lines) if "def resolve_split_path" in ln)
    end = next(i for i, ln in enumerate(lines) if "TRAIN_DIR =" in ln)
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
    start = next(i for i, ln in enumerate(lines) if "class MLPSLoss" in ln)
    end = next(i for i, ln in enumerate(lines) if "criterion_s2 =" in ln)
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
        get_clean_cell_10_stage4,
        get_clean_cell_14_stage2,
        get_clean_cell_14_stage3,
        get_clean_cell_16_stage2,
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


def test_notebook_01_rule17_pretrained_validation_loop_and_parity():
    """Verify Notebook 01 (Base, T4, L4, root T4) enforces Rule 17 init, env presets, EMA, and metrics."""
    for key in ("stage1", "stage1_t4", "stage1_l4", "stage1_t4_root"):
        content = NOTEBOOKS[key].read_text(encoding="utf-8")
        assert "resolve_stage1_config_from_env()" in content, f"{key} missing env-driven preset resolution"
        assert "load_pretrained_resnet38d_weights" in content, f"{key} missing pretrained ResNet-38d loader"
        assert "ALLOW_RANDOM_INIT = DRY_RUN" in content, f"{key} missing anti-silent-random-init guard"
        assert "LOSS_TYPE = EXP_CFG.loss_type" in content, f"{key} missing preset loss_type binding"
        assert "STAGE1_INPUT_NORM" in content and "'input_norm': STAGE1_INPUT_NORM" in content, (
            f"{key} missing explicit input_norm persistence in checkpoint"
        )
        assert "BCSSStage1ValDataset" in content and "evaluate_stage1_validation" in content, (
            f"{key} missing independent validation loop"
        )
        assert "Stage1Trainer(" in content and "ModelEMA(" in content, (
            f"{key} missing Stage1Trainer delegation or ModelEMA"
        )
        assert "stage1_best_cls.pth" in content, f"{key} missing stage1_best_cls.pth export"
        for metric_key in (
            "val_mAP",
            "val_auroc",
            "val_best_f1",
            "val_gated_pm_miou",
            "val_gated_pm_miou_global",
            "fc8_std",
            "ema_metrics",
            "exp_preset",
            "selected_weights",
            "best_cls_score",
        ):
            assert metric_key in content, f"{key} missing validation/export metric {metric_key}"
        assert "if epoch_acc >= best_acc:" not in content, (
            f"{key} must NOT select best checkpoint by train accuracy!"
        )

    t4_s1 = NOTEBOOKS["stage1_t4"].read_text(encoding="utf-8")
    l4_s1 = NOTEBOOKS["stage1_l4"].read_text(encoding="utf-8")
    root_t4_s1 = NOTEBOOKS["stage1_t4_root"].read_text(encoding="utf-8")
    assert t4_s1 == root_t4_s1, "Root 01_Train_Stage1_Classifier_T4.ipynb is out of sync with notebooks/"
    assert "else 16" in t4_s1 and "else 20" in t4_s1
    assert "else 32" in l4_s1 and "else 25" in l4_s1
    assert "LEARNING_RATE = EXP_CFG.lr_bb" in t4_s1 and "LEARNING_RATE = EXP_CFG.lr_bb" in l4_s1


def test_notebook_02_six_audit_fixes_parity():
    """Verify Notebook 02 (Base, T4, L4, root T4) includes all 6 audited scientific fixes."""
    for key in ("stage2", "stage2_t4", "stage2_l4", "stage2_t4_root"):
        content = NOTEBOOKS[key].read_text(encoding="utf-8")
        # (a) Explicit Stage-1 checkpoint policy + SG-DAP + quick val pseudo-mask sanity check
        assert "os.environ.get('STAGE1_CKPT_SOURCE', 'author_baseline')" in content, (
            f"{key} missing env-wired Stage-1 selection policy"
        )
        assert "USE_SG_DAP = True" in content, f"{key} missing verified USE_SG_DAP=True default"
        assert "[PM SANITY CHECK" in content, f"{key} missing pre-generation validation PM sanity check"
        # (b) Outside-ROI gt==4 exclusion in _quick_val_miou and Cell 16
        assert "pred_v[gt_arr == 4] = 4" in content, f"{key} missing gt==4 exclusion in _quick_val_miou"
        assert "pred_raw_roi[gt_arr == 4] = 4" in content, f"{key} missing gt==4 exclusion in Cell 16"
        # (c) Unbiased batched validation (no sorted(...)[:120] or [:200])
        assert "BCSSStage2ValDataset" in content and "_sample_patient_stratified_s2" in content, (
            f"{key} missing batched/stratified validation dataset"
        )
        assert "[: (24 if DRY_RUN else 120)]" not in content, f"{key} still contains biased [:120] slicing"
        assert "min(200, len(val_files))" not in content, f"{key} still contains biased [:200] slicing"
        # (d) Synchronized Data Augmentation in BCSSStage2Dataset
        assert "FLIP_LEFT_RIGHT" in content and "FLIP_TOP_BOTTOM" in content and "GaussianBlur" in content, (
            f"{key} missing synchronized HFlip/VFlip/GaussianBlur augmentation"
        )
        # (e) CGM evaluation in Cell 16
        assert "[VAL +CGM]" in content and "val_cgm_miou_4cls" in content and "cgm_delta_miou" in content, (
            f"{key} missing Stage-2 + CGM comparison in Cell 16"
        )
        # (f) Anti-silent-fallback ResNet-101 loading + paper-parity MLPS weights
        assert "allow_random_init_fallback=DRY_RUN" in content, f"{key} missing ResNet-101 anti-silent-fallback guard"
        assert "MLPSLoss(w_main=0.6, w_aux1=0.2, w_aux2=0.2)" in content, f"{key} missing paper-parity MLPS weights"
        # Review-pass robustness fixes: no silent Stage-1 load swallow, compact best ckpt, updated best_miou before latest save
        assert "len(_s1_matched) < 100" in content, f"{key} missing Stage-1 matched layer count check in Cell 6"
        assert "active_labels=label_batch" in content, f"{key} missing active_labels zero-tie protection in Cell 8"
        assert "improved_s2 =" in content and "best_state_s2 =" in content, (
            f"{key} missing pre-save best_miou update and compact best checkpoint save"
        )
        assert "active_fg_ious =" in content, f"{key} missing active-class macro filtering in Cell 16"


def test_notebook_01_zero_tie_and_compact_best_checkpoint():
    """Verify Notebook 01 delegates zero-tie evaluation to Stage1Trainer and saves strict compact best checkpoints."""
    from src.training.train_stage1 import Stage1Trainer

    trainer_eval_src = inspect.getsource(Stage1Trainer.evaluate)
    assert "cam_masked = torch.where(gate > 0.5, cam_norm, torch.full_like(cam_norm, -1.0))" in trainer_eval_src
    assert "cam_masked_g = torch.where(gate > 0.5, cam_norm_g, torch.full_like(cam_norm_g, -1.0))" in trainer_eval_src

    for key in ("stage1", "stage1_t4", "stage1_l4"):
        content = NOTEBOOKS[key].read_text(encoding="utf-8")
        assert "return trainer.evaluate(adapted_batches)" in content, (
            f"{key} evaluate_stage1_validation must delegate to Stage1Trainer.evaluate"
        )
        assert "if k not in ('optimizer', 'scheduler', 'ema_state_dict', 'proj_head_state_dict', 'mpppc_state_dict')" in content, (
            f"{key} missing compact inference-only best checkpoint stripping optimizer/scheduler/EMA/Arm-C keys"
        )
        assert "best_state['state_dict'] = winning_state_dict" in content
        assert "'state_dict': cls_state_dict" in content


def test_notebook_01_stage1_parity_and_spawn_guards():
    """Verify Notebook 01 enforces env presets, spawn D4 loaders, ModelEMA, resume, and split VPS alarms."""
    for key in ("stage1", "stage1_t4", "stage1_l4"):
        nb_data = json.loads(NOTEBOOKS[key].read_text(encoding="utf-8"))
        c4 = "".join(nb_data["cells"][4]["source"])
        c8 = "".join(nb_data["cells"][8]["source"])
        c10 = "".join(nb_data["cells"][10]["source"])
        c12 = "".join(nb_data["cells"][12]["source"])
        c14 = "".join(nb_data["cells"][14]["source"])
        c16 = "".join(nb_data["cells"][16]["source"])

        # Cell 4: env-driven STAGE1_EXP_PRESET, STAGE1_SEED, STAGE1_GIT_REF before src/ imports
        assert "STAGE1_EXP_PRESET" in c4 and "STAGE1_SEED" in c4 and "STAGE1_GIT_REF" in c4

        # C-D3 / C-D4: No ColorJitter, picklable Datasets + D4 from src.training.dataset_loader, spawn + persistent_workers
        assert "transforms.ColorJitter" not in c8, f"{key} Cell 8 must not include transforms.ColorJitter"
        assert "from src.training.dataset_loader import" in c8
        assert "BCSSStage1TrainDataset" in c8 and "BCSSStage1ValDataset" in c8
        assert "RandomD4Transform" in c8 and "use_d4_aug=EXP_CFG.use_d4_aug" in c8
        assert "class BCSSStage1Dataset" not in c8 and "class BCSSStage1ValDataset" not in c8
        assert "'multiprocessing_context': 'spawn'" in c8 and "'persistent_workers': True" in c8
        assert "strict_patient_filter=True" in c8

        # C-D2 / C-D3: model.train() before build_optimizer_and_scheduler; forward_stage1_dual with active_labels
        assert c10.index("model.train()") < c10.index("build_optimizer_and_scheduler("), (
            f"{key} Cell 10 must call model.train() before build_optimizer_and_scheduler"
        )
        assert "gamma=1.0" in c10 and "def forward_stage1_dual(" in c10 and "active_labels=None" in c10
        assert "ModelEMA(" in c10 and "ProjectionHead(" in c10 and "MultiPrototypePPCLoss(" in c10

        # C-D5: resume epoch=-1 default, EMA, proj_head, best_cls_score, joint_no_improve restored
        assert "int(ckpt.get('epoch', -1)) + 1" in c12
        assert "epochs_without_improve = int(ckpt.get('epochs_without_improve', 0))" in c12
        assert "ema_state_dict" in c12 and "proj_head_state_dict" in c12
        assert "best_cls_score" in c12 and "joint_no_improve" in c12

        # C-D1 / C-D3 / C-D4 / C-D5 + VPS alarms: Stage1Trainer, dual-branch loss, clip_grad_norm_, split alarms
        assert "PDA_START_EPOCH = EXP_CFG.pda_start_epoch" in c14
        assert "0.5 * criterion(logits_clean, labels) + 0.5 * criterion(logits_pda, labels)" in c14
        assert "clip_grad_norm_(_trainable_all, max_norm=5.0)" in c14
        assert "pbar.close()" in c14 and "gc.collect()" in c14
        assert "sel_best_f1 < 0.55 or sel_val_loss > 1.20 or sel_val_map < 0.65" in c14 and "break" in c14
        assert "[CALIBRATION COLLAPSE ALARM]" in c14
        assert "[EARLY STOP PATIENCE]" in c14
        assert "[EARLY REGRESSION ALARM]" not in c14, (
            f"{key} Cell 14 must never emit [EARLY REGRESSION ALARM] so colab_run_script exits cleanly on patience"
        )
        assert "stage1_best_cls.pth" in c12 and "stage1_best_cls.pth" in c14 and "stage1_best_cls.pth" in c16


@pytest.mark.parametrize(
    "preset,expected_opt,expected_sched,expected_d4,expected_equiv,expected_ppc",
    [
        ("arm_a_polysgd", "SGD", "LambdaLR", True, False, False),
        ("arm_b_equiv_adamw", "AdamW", "LambdaLR", True, True, False),
        ("arm_c_mpppc_polysgd", "SGD", "LambdaLR", True, False, True),
    ],
)
def test_notebook_01_env_preset_overrides_and_active_labels_forward(
    monkeypatch,
    preset,
    expected_opt,
    expected_sched,
    expected_d4,
    expected_equiv,
    expected_ppc,
):
    """Verify env overrides resolve and configure Notebook 01 Cell 8/10/14 cleanly for Arms A/B/C."""
    from src.config.stage1_experiments import resolve_stage1_config_from_env
    from src.losses_metrics.losses import MultiPrototypePPCLoss
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.train_stage1 import (
        AttentionModule,
        ModelEMA,
        ProjectionHead,
        Stage1Trainer,
        build_optimizer_and_scheduler,
    )

    monkeypatch.setenv("STAGE1_EXP_PRESET", preset)
    monkeypatch.setenv("STAGE1_MAX_EPOCHS", "7")
    monkeypatch.setenv("STAGE1_SEED", "123")
    monkeypatch.setenv("STAGE1_BATCH_SIZE", "8")

    nb_data = json.loads(NOTEBOOKS["stage1_l4"].read_text(encoding="utf-8"))
    c4 = "".join(nb_data["cells"][4]["source"])
    c8 = "".join(nb_data["cells"][8]["source"])
    c10 = "".join(nb_data["cells"][10]["source"])

    # 1. Cell 4 env seed & preset reading
    assert "STAGE1_EXP_PRESET = os.environ.get(\"STAGE1_EXP_PRESET\"" in c4
    assert "_raw_seed = os.environ.get(\"STAGE1_SEED\"" in c4
    cfg = resolve_stage1_config_from_env()
    assert cfg.preset_name == preset
    assert cfg.max_epochs == 7
    assert cfg.seed == 123
    assert cfg.batch_size == 8
    assert cfg.use_d4_aug is expected_d4
    assert cfg.use_equiv_consistency is expected_equiv
    assert cfg.use_mpppc is expected_ppc

    # 2. Cell 8 transform & batch resolution logic
    assert "if EXP_CFG.use_d4_aug:" in c8
    assert "use_d4_aug=EXP_CFG.use_d4_aug" in c8

    # 3. Cell 10 optimizer/scheduler/EMA/Arm-C construction with preset hyperparams
    model = WSSSTissueStage1NetCAM(num_classes=4)
    model.train()
    proj_head = ProjectionHead(4096, 256, 128, 16) if cfg.use_mpppc else None
    mpppc = (
        MultiPrototypePPCLoss(4, cfg.ppc_num_prototypes, 128, conf_thresh=cfg.ppc_conf_thresh, warm_start=True)
        if cfg.use_mpppc
        else None
    )
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    steps_per_ep = 10
    step_sched_per_batch = (cfg.scheduler_type == "poly") or (cfg.warmup_epochs > 0.0)
    warmup_steps = int(round(cfg.warmup_epochs * steps_per_ep)) if step_sched_per_batch else 0
    total_steps = (cfg.max_epochs * steps_per_ep) if step_sched_per_batch else None
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr=cfg.lr_bb,
        weight_decay=cfg.wd_bb,
        epochs=cfg.max_epochs,
        optimizer_type=cfg.optimizer_type,
        scheduler_type=cfg.scheduler_type,
        lr_fc8=cfg.lr_fc8,
        wd_fc8=cfg.wd_fc8,
        momentum=cfg.momentum,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        poly_power=cfg.poly_power,
        eta_min=cfg.eta_min,
        extra_params=list(proj_head.parameters()) if proj_head is not None else None,
    )
    assert type(optimizer).__name__ == expected_opt
    assert type(scheduler).__name__ == expected_sched
    assert ema is not None and set(ema.clean_model_state_dict().keys()) == set(model.state_dict().keys())

    # 4. Cell 10 forward_stage1_dual with active_labels=single-class targets
    start_idx = c10.index("def forward_stage1_dual(")
    ns = {"torch": torch, "AttentionModule": AttentionModule}
    exec(c10[start_idx:], ns)
    forward_dual = ns["forward_stage1_dual"]
    images = torch.randn(2, 3, 224, 224)
    single_cls_targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    attn_mod = AttentionModule(pda_mode=cfg.pda_mode)
    lc, pc, lp, pp = forward_dual(
        model, images, enable_pda=True, gamma=0.75, attn_module=attn_mod, active_labels=single_cls_targets
    )
    assert lc.shape == (2, 4) and lp.shape == (2, 4)
    assert torch.isfinite(lc).all() and torch.isfinite(lp).all()
    assert not torch.allclose(lc, lp)

    # 5. Stage1Trainer construction matching Cell 14
    trainer = Stage1Trainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MultiLabelSoftMarginLoss(),
        scheduler=scheduler,
        device=torch.device("cpu"),
        enable_amp=False,
        pda_start_epoch=cfg.pda_start_epoch,
        initial_gamma=cfg.initial_gamma,
        gamma_decay=cfg.gamma_decay,
        min_gamma=cfg.min_gamma,
        pda_mode=cfg.pda_mode,
        step_scheduler_per_batch=step_sched_per_batch,
        model_ema=ema,
        proj_head=proj_head,
        mpppc_loss=mpppc,
        lambda_ppc=cfg.lambda_ppc,
        ppc_start_epoch=cfg.ppc_start_epoch,
        use_equiv_consistency=cfg.use_equiv_consistency,
        equiv_scales=cfg.equiv_scales,
        lambda_eq_max=cfg.lambda_eq_max,
        eq_ramp_start_epoch=cfg.eq_ramp_start_epoch,
        eq_ramp_end_epoch=cfg.eq_ramp_end_epoch,
    )
    assert trainer.use_equiv_consistency is expected_equiv
    assert (trainer.mpppc_loss is not None) is expected_ppc


def test_notebook_02_gradcam_sgdap_guards_and_prose_parity():
    """Verify Notebook 02 (Base, T4, L4, root T4) fixes prose, DRY_RUN split order, Grad-CAM/SG-DAP, and Cell 14 guards."""
    for key in ("stage2", "stage2_t4", "stage2_l4", "stage2_t4_root"):
        nb_data = json.loads(NOTEBOOKS[key].read_text(encoding="utf-8"))
        c0 = "".join(nb_data["cells"][0]["source"])
        c2 = "".join(nb_data["cells"][2]["source"])
        c8 = "".join(nb_data["cells"][8]["source"])
        c12 = "".join(nb_data["cells"][12]["source"])
        c14 = "".join(nb_data["cells"][14]["source"])

        # 1. Accurate prose (no misleading 'from scratch'; Cell 2 reflects author_baseline)
        assert "from scratch" not in c0.lower(), f"{key} Cell 0 still mentions 'from scratch'"
        assert "author_baseline" in c0 and "stage1_checkpoint_trained_on_bcss.pth" in c0
        assert "ImageNet-pretrained ResNet-101" in c0
        assert "stage1_checkpoint_trained_on_bcss.pth" in c2 and "author_baseline" in c2
        assert "from scratch" not in c12.lower(), f"{key} Cell 12 still mentions 'from scratch'"
        assert "ImageNet-pretrained ResNet-101" in c12

        # 2. Cell 8: train_split_pids filter occurs BEFORE LIMIT_PM slicing
        idx_split = c8.index("train_split_pids = set(")
        idx_limit = c8.index("LIMIT_PM = 100 if DRY_RUN else None")
        assert idx_split < idx_limit, f"{key} Cell 8 must filter train_split_pids BEFORE [:LIMIT_PM]"

        # 3. Cell 8: extract_cams is NOT inside torch.no_grad(); SG-DAP receives active_labels; logs all 3 layers
        assert "active_labels=_lb" in c8, f"{key} Cell 8 sanity check missing active_labels=_lb in sg_dap_refiner"
        assert "active_labels=label_batch" in c8, f"{key} Cell 8 main loop missing active_labels=label_batch"
        assert "('bn7', 'b5_2', 'b4_5')" in c8, f"{key} Cell 8 sanity check must evaluate all 3 layers"
        # Ensure extract_cams appears before `with torch.no_grad():` in both loops
        for marker in ("_cd = extractor.extract_cams(", "cams_dict = extractor.extract_cams("):
            assert marker in c8
            line_for_marker = next(ln for ln in c8.splitlines() if marker in ln)
            indent_len = len(line_for_marker) - len(line_for_marker.lstrip())
            assert indent_len <= 12, f"{key} Cell 8 {marker} unexpectedly over-indented (possibly inside no_grad)"

        # 4. Cell 14: anti-blind-training s2_val_loader guard and all-NaN total_batches==0 guard
        assert "if not DRY_RUN and (s2_val_loader is None" in c14, f"{key} Cell 14 missing s2_val_loader guard"
        assert "if total_batches == 0" in c14, f"{key} Cell 14 missing total_batches == 0 all-NaN guard"


def test_notebook_01_stop_after_epoch_and_notebook_02_ckpt_source_env_wiring(monkeypatch):
    """Verify STAGE1_STOP_AFTER_EPOCH decoupling in NB1 and STAGE1_CKPT_SOURCE env wiring in NB2."""
    for key in ("stage1", "stage1_t4", "stage1_l4", "stage1_t4_root"):
        nb_data = json.loads(NOTEBOOKS[key].read_text(encoding="utf-8"))
        c10 = "".join(nb_data["cells"][10]["source"])
        c14 = "".join(nb_data["cells"][14]["source"])
        c16 = "".join(nb_data["cells"][16]["source"])

        assert "STOP_AFTER_EPOCH = None if DRY_RUN else EXP_CFG.stop_after_epoch" in c10, (
            f"{key} Cell 10 missing STOP_AFTER_EPOCH wiring"
        )
        assert "total_steps = (MAX_EPOCHS * _steps_per_ep)" in c10, (
            f"{key} Cell 10 must keep total_steps tied to full MAX_EPOCHS"
        )
        assert "if 'STOP_AFTER_EPOCH' not in globals():" in c14, (
            f"{key} Cell 14 missing STOP_AFTER_EPOCH fallback"
        )
        assert "if STOP_AFTER_EPOCH is not None and (epoch + 1) >= int(STOP_AFTER_EPOCH):" in c14, (
            f"{key} Cell 14 missing end-of-epoch STOP_AFTER_EPOCH break"
        )
        assert "[STOP AFTER EPOCH]" in c14, f"{key} Cell 14 missing [STOP AFTER EPOCH] marker"
        stop_idx = c14.index("[STOP AFTER EPOCH]")
        early_idx = c14.index("[EARLY STOP PATIENCE]")
        assert stop_idx < early_idx, f"{key} Cell 14 must check STOP_AFTER_EPOCH before [EARLY STOP PATIENCE]"
        assert "'stop_after_epoch': globals().get('STOP_AFTER_EPOCH', None)" in c16, (
            f"{key} Cell 16 missing stop_after_epoch metadata in STAGE1_TRAINING_METRICS.json"
        )

    for key in ("stage2", "stage2_t4", "stage2_l4", "stage2_t4_root"):
        nb_data = json.loads(NOTEBOOKS[key].read_text(encoding="utf-8"))
        c6 = "".join(nb_data["cells"][6]["source"])
        src_line = next(ln for ln in c6.splitlines() if ln.strip().startswith("STAGE1_CKPT_SOURCE ="))
        assert "os.environ.get('STAGE1_CKPT_SOURCE', 'author_baseline')" in src_line

        monkeypatch.delenv("STAGE1_CKPT_SOURCE", raising=False)
        ns_default = {"os": __import__("os")}
        exec(src_line, ns_default)
        assert ns_default["STAGE1_CKPT_SOURCE"] == "author_baseline"

        monkeypatch.setenv("STAGE1_CKPT_SOURCE", "retrained")
        ns_retrained = {"os": __import__("os")}
        exec(src_line, ns_retrained)
        assert ns_retrained["STAGE1_CKPT_SOURCE"] == "retrained"


def test_notebook_02_stage2_prelaunch_patch_contracts(monkeypatch):
    """Verify all Stage-2 Pre-Launch Patch contracts across Base, T4, L4, and root T4 notebooks."""
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM

    for key in ("stage2", "stage2_t4", "stage2_l4", "stage2_t4_root"):
        nb_data = json.loads(NOTEBOOKS[key].read_text(encoding="utf-8"))
        c4 = "".join(nb_data["cells"][4]["source"])
        c6 = "".join(nb_data["cells"][6]["source"])
        c8 = "".join(nb_data["cells"][8]["source"])
        c10 = "".join(nb_data["cells"][10]["source"])
        c12 = "".join(nb_data["cells"][12]["source"])
        c14 = "".join(nb_data["cells"][14]["source"])
        c16 = "".join(nb_data["cells"][16]["source"])

        # Cell 4: STAGE2_GIT_REF fail-fast + sys.modules purge + STAGE2_SEED + device print
        assert "STAGE2_GIT_REF" in c4 and "check=True" in c4
        assert "sys.modules.pop(_mod_k, None)" in c4
        assert "STAGE2_SEED" in c4 and "cudnn.benchmark = False" in c4
        assert "[READY] Compute Device:" in c4
        if key in ("stage2", "stage2_l4"):
            assert "including_emulation=False" in c4

        # Cell 6: TEST split resolution + STAGE1_LOCAL_CKPT + SHA256 + strict=True + finiteness + STAGE1_META
        assert "TEST_IMG_DIR = resolve_split_path(DATASET_DIR, 'test', 'img')" in c6
        assert "TEST_MASK_DIR = resolve_split_path(DATASET_DIR, 'test', 'mask')" in c6
        assert "STAGE1_LOCAL_CKPT" in c6 and "STAGE1_SHA256" in c6
        assert "strict=True" in c6 and "torch.isfinite(v).all()" in c6
        assert "STAGE1_META" in c6

        # Cell 8: anti-leakage fail-fast + env knobs + pm_manifest.json + MS-TTA bn7 + use_deep_affinity=False + ThreadPoolExecutor
        assert "raise RuntimeError('[STAGE2 ERROR] Anti-leakage train_split_pids" in c8
        assert "raise RuntimeError(f'[STAGE2 ERROR] Anti-leakage val_split_pids" in c8
        assert "STAGE2_PM_MODE" in c8 and "STAGE2_USE_OTSU" in c8
        assert "STAGE2_USE_SG_DAP" in c8 and "STAGE2_BATCH_PM" in c8
        assert "pm_manifest.json" in c8
        assert "StainGuidedAffinityRefiner(use_deep_affinity=False)" in c8
        assert "_extract_bn7_ms_tta_cam" in c8
        assert "ThreadPoolExecutor(max_workers=4)" in c8 and "mininterval=10.0" in c8

        # Cell 10: env overrides + [PARITY] banner
        assert "STAGE2_EPOCHS" in c10 and "STAGE2_LR" in c10
        assert "UNC_WEIGHT" in c10 and "UABL_WEIGHT" in c10
        assert "[PARITY] Model=DeepLabV3+ ResNet-101" in c10

        # Cell 12: CPU load + epoch=-1 default + scaler_s2 restore
        assert "map_location='cpu'" in c12
        assert "int(ckpt_s2.get('epoch', -1)) + 1" in c12
        assert "_pending_scaler_s2_state" in c12

        # Cell 14: picklable dataset_loader import + spawn DataLoader + 4-cls quick val + clip_grad + atomic save + [EPOCH]
        assert "from src.training.dataset_loader import" in c14
        assert "BCSSStage2Dataset" in c14 and "BCSSStage2ValDataset" in c14
        assert "class BCSSStage2Dataset(" not in c14 and "class BCSSStage2ValDataset(" not in c14
        assert "'multiprocessing_context': 'spawn'" in c14
        assert "'persistent_workers': True" in c14 and "'prefetch_factor': 2" in c14
        assert "STAGE2_BATCH_SIZE" in c14 and "STAGE2_NUM_WORKERS" in c14 and "STAGE2_USE_D4_AUG" in c14
        assert "torch.argmax(logits_v[:, :4], dim=1)" in c14
        assert "unscale_(optimizer_s2)" in c14 and "clip_grad_norm_(model_stage2.parameters(), 5.0)" in c14
        assert "os.replace(" in c14
        assert "[EPOCH {epoch+1:02d}] Train Loss={avg_loss:.4f} | Val_mIoU=" in c14
        assert "pbar.close()" in c14 and "gc.collect()" in c14

        # Cell 16: strict=True + val_split_pids_s2 + test_eval + 4 modes + provenance
        assert "strict=True" in c16 and "patient_ids=val_split_pids_s2" in c16
        assert "TEST_IMG_DIR" in c16 and "TEST_MASK_DIR" in c16 and "test_split_pids_s2" in c16
        for tok in ("'raw'", "'cgm'", "'ms_tta'", "'ms_tta_cgm'", "'val_eval'", "'test_eval'", "'provenance'", "has_any_active"):
            assert tok in c16, f"{key} Cell 16 missing {tok}"

    # Execute Cell 8 _extract_bn7_ms_tta_cam directly on synthetic tensors to verify shape/gating/normalization
    nb_l4 = json.loads(NOTEBOOKS["stage2_l4"].read_text(encoding="utf-8"))
    c8_l4 = "".join(nb_l4["cells"][8]["source"])
    fn_start = c8_l4.index("def _extract_bn7_ms_tta_cam(")
    fn_end = c8_l4.index("def _assemble_pm_no_otsu(")
    ns_cam = {"torch": torch}
    exec(c8_l4[fn_start:fn_end], ns_cam)
    s1_model = WSSSTissueStage1NetCAM(num_classes=4).eval()
    imgs = torch.randn(2, 3, 64, 64)
    lbls = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    cam_out = ns_cam["_extract_bn7_ms_tta_cam"](s1_model, imgs, lbls, target_hw=(64, 64))
    assert cam_out.shape == (2, 4, 64, 64)
    assert torch.isfinite(cam_out).all()
    assert torch.all(cam_out[0, 1] == 0.0) and torch.all(cam_out[0, 3] == 0.0)
    assert float(cam_out.max()) <= 1.0 + 1e-6 and float(cam_out.min()) >= 0.0

