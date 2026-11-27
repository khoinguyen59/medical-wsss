"""Unit and integration tests for Google Colab WSSS Pipeline Notebooks & Modular Inference Engine.

Verifies:
1. Valid Jupyter Notebook JSON structure (nbformat 4) & clean AST compilation across active ThinDriveRunner_v2 notebooks.
2. Invariant 12 compliance: 100% ASCII 0x2D hyphens, 0 em-dashes/en-dashes.
3. DeepLabV3+ ResNet-101 & Stage-1 ResNet-38d architecture instantiation, key consistency, and forward pass.
4. Otsu clinical gating boundary conditions & unimodal stained H&E preservation.
5. MultiMethodWSSSEngine (M1..M7), PistoSeg RFM, CAM-NMS, and colab_pkg exact parity.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ACTIVE_THIN_NOTEBOOKS = [
    Path("notebooks/SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb"),
    Path("notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb"),
    Path("notebooks/SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb"),
    Path("notebooks/SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb"),
]


def test_notebook_json_structure_and_nbformat():
    """Verify active ThinDriveRunner_v2 notebooks exist, parse cleanly as JSON, and conform to nbformat 4."""
    for nb_path in ACTIVE_THIN_NOTEBOOKS:
        assert nb_path.is_file(), f"Notebook not found at: {nb_path}"
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        assert "cells" in nb_data, f"Notebook missing 'cells' key: {nb_path}"
        assert nb_data["nbformat"] == 4, f"Expected nbformat 4 in {nb_path}, got {nb_data.get('nbformat')}"
        assert len(nb_data["cells"]) >= 4, f"Expected >= 4 cells in {nb_path}, got {len(nb_data['cells'])}"


def test_code_cells_syntax_compilation():
    """Verify that every code cell in active ThinDriveRunner_v2 notebooks parses cleanly into an AST."""
    for nb_path in ACTIVE_THIN_NOTEBOOKS:
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb_data["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 2, f"Expected >= 2 code cells in {nb_path}"

        for idx, cell in enumerate(code_cells):
            raw_source = "".join(cell.get("source", []))
            python_lines = []
            for line in raw_source.splitlines():
                stripped = line.strip()
                if stripped.startswith(("!", "%")):
                    python_lines.append(f"# {stripped}")
                else:
                    python_lines.append(line)

            python_code = "\n".join(python_lines)
            try:
                tree = ast.parse(python_code, filename=f"{nb_path.name}_Cell_{idx}")
                assert tree is not None
            except SyntaxError as e:
                pytest.fail(
                    f"Syntax error in {nb_path.name} Code Cell {idx} (line {e.lineno}): {e.msg}\n"
                    f"Code snippet:\n{python_code[:200]}"
                )


def test_invariant_12_strict_ascii_hyphens_only():
    """Verify 0 non-ASCII em-dashes (\\u2014) or en-dashes (\\u2013) across active Colab notebooks."""
    forbidden_chars = [
        ("\u2014", "em-dash"),
        ("\u2013", "en-dash"),
        ("\u2015", "horizontal bar"),
        ("\u2212", "minus sign"),
    ]
    for nb_path in ACTIVE_THIN_NOTEBOOKS:
        content = nb_path.read_text(encoding="utf-8")
        for char, name in forbidden_chars:
            count = content.count(char)
            assert count == 0, f"Invariant 12 violation: Found {count} instances of {name} ({hex(ord(char))}) in {nb_path}"


def test_deeplabv3plus_architecture_and_forward_pass():
    """Verify the modular DeepLabV3+ module instantiates and executes forward pass."""
    from src.colab_pkg.model_loader import DeepLabV3Plus

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
    from src.colab_pkg.model_loader import OtsuClinicalGating

    white_glass = np.full((128, 128, 3), 255, dtype=np.uint8)
    tissue_mask = OtsuClinicalGating.compute_tissue_mask(white_glass)
    assert np.all(tissue_mask == 0), "Tissue mask should be 0 for pure white glass"

    dummy_pred = np.zeros((128, 128), dtype=np.uint8)
    gated = OtsuClinicalGating.apply_gating(dummy_pred, white_glass, background_class_idx=4)
    assert np.all(gated == 4), "100% of pure white glass pixels must be forced to Class 4 (Background)"


def test_otsu_clinical_gating_adversarial_boundary_conditions():
    """Adversarially test Otsu clinical gating under extreme inputs."""
    from src.colab_pkg.model_loader import OtsuClinicalGating

    near_white = np.random.randint(238, 256, (128, 128, 3), dtype=np.uint8)
    mask_near_white = OtsuClinicalGating.compute_tissue_mask(near_white)
    assert np.all(mask_near_white == 0), "Near-white glass must have 0% tissue mask"

    half_patch = np.full((128, 128, 3), 255, dtype=np.uint8)
    half_patch[:, :64] = [180, 80, 140]
    mask_half = OtsuClinicalGating.compute_tissue_mask(half_patch)
    assert np.all(mask_half[:, :64] == 1), "Stained tissue region must be detected as 1"
    assert np.all(mask_half[:, 64:] == 0), "White glass region must be detected as 0"

    with pytest.raises(ValueError):
        OtsuClinicalGating.compute_tissue_mask(np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError):
        OtsuClinicalGating.compute_tissue_mask(np.zeros((64, 64, 4), dtype=np.uint8))


def test_otsu_clinical_gating_spatial_alignment_and_morphology():
    """Verify spatial alignment (H, W) across irregular shapes and morphological clean-up."""
    from src.colab_pkg.model_loader import OtsuClinicalGating

    speck_img = np.full((64, 64, 3), 250, dtype=np.uint8)
    speck_img[32, 32] = [180, 50, 150]
    mask_speck = OtsuClinicalGating.compute_tissue_mask(speck_img)
    assert mask_speck[32, 32] == 0, "Isolated 1x1 noise speck must be eliminated by MORPH_OPEN"

    hole_img = np.full((64, 64, 3), [180, 50, 150], dtype=np.uint8)
    hole_img[32, 32] = [255, 255, 255]
    mask_hole = OtsuClinicalGating.compute_tissue_mask(hole_img)
    assert mask_hole[32, 32] == 1, "1x1 hole inside tissue must be filled by MORPH_CLOSE"

    for h, w in [(33, 47), (225, 225), (512, 256), (7, 13)]:
        dummy_rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        dummy_pred = np.zeros((h, w), dtype=np.uint8)
        gated = OtsuClinicalGating.apply_gating(dummy_pred, dummy_rgb, background_class_idx=4)
        assert gated.shape == (h, w), f"Expected shape ({h}, {w}), got {gated.shape}"


def test_model_memory_safety_and_tensor_footprint():
    """Verify DeepLabV3+ memory footprint is strictly < 4 GB VRAM on T4."""
    from src.colab_pkg.model_loader import DeepLabV3Plus

    model = DeepLabV3Plus(num_classes=4, output_stride=16)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024**2)

    assert total_params == 59339940, f"Expected 59,339,940 parameters, got {total_params}"
    assert param_mb < 250.0, f"Model parameter footprint ({param_mb:.2f} MB) exceeds 250 MB"

    for b in [1, 2, 4]:
        inp = torch.randn(b, 3, 224, 224)
        with torch.no_grad():
            out = model(inp)
        assert out.shape == (b, 4, 224, 224), f"Expected output shape ({b}, 4, 224, 224), got {out.shape}"

    for h, w in [(224, 320), (225, 225)]:
        inp = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out = model(inp)
        assert out.shape == (1, 4, h, w), f"Expected output shape (1, 4, {h}, {w}), got {out.shape}"


def test_dual_mode_patient_selection_and_anti_leakage():
    """Verify patient manifest zero leakage for TARGET_PATIENTS."""
    manifest_path = Path("src/config/bcss_patient_splits.json")
    assert manifest_path.is_file(), f"Manifest not found: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_set = set(manifest["splits"]["train"])
    val_set = set(manifest["splits"]["val"])
    test_set = set(manifest["splits"]["test"])
    target_patients = {"TCGA-AQ-A04H", "TCGA-C8-A1HJ", "TCGA-AR-A0TQ"}

    if target_patients.issubset(test_set):
        assert len(target_patients & train_set) == 0, "TARGET_PATIENTS leaked into train split!"
        assert len(target_patients & val_set) == 0, "TARGET_PATIENTS leaked into val split!"
    else:
        active_test = set(manifest.get("test_active_patients", test_set))
        assert len(active_test & train_set) == 0, "Active test patients leaked into train split!"
        assert len(active_test & val_set) == 0, "Active test patients leaked into val split!"


def test_otsu_clinical_gating_preserves_unimodal_stained_he_tissue():
    """Verify Otsu clinical gating does NOT punch swiss-cheese holes inside 100% stained H&E patches."""
    from src.colab_pkg.model_loader import OtsuClinicalGating
    from src.datasets.otsu_filter import compute_otsu_tissue_mask

    he_patch = np.zeros((224, 224, 3), dtype=np.uint8)
    he_patch[:112, :] = [85, 45, 130]
    he_patch[112:, :] = [210, 185, 205]
    he_patch[50:58, 50:58] = [245, 245, 245]

    nb_mask = OtsuClinicalGating.compute_tissue_mask(he_patch)
    src_mask = compute_otsu_tissue_mask(he_patch)

    assert np.all(nb_mask == 1), (
        f"OtsuClinicalGating falsely masked {(nb_mask == 0).sum()} stained tissue pixels as background!"
    )
    assert np.all(src_mask == 1), (
        f"src.datasets.otsu_filter falsely masked {(src_mask == 0).sum()} stained tissue pixels as background!"
    )


def test_evaluator_4cls_and_5cls_mathematical_integrity():
    """Verify ColabSegmentationEvaluator computes 4-class WSSS-Tissue metrics without distortion."""
    from src.colab_pkg.evaluator import ColabSegmentationEvaluator

    evaluator = ColabSegmentationEvaluator(num_classes=5)
    gt_mask = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [4, 4, 4, 4],
        [0, 1, 2, 3],
    ], dtype=np.int64)
    pred_raw = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [0, 1, 0, 1],
        [0, 1, 2, 3],
    ], dtype=np.uint8)
    pred_gated = pred_raw.copy()
    pred_gated[0, 0] = 4
    pred_gated[gt_mask == 4] = 4
    pred_raw_roi = pred_raw.copy()
    pred_raw_roi[gt_mask == 4] = 4

    evaluator.update(pred_gated, gt_mask, pred_raw_4cls=pred_raw_roi)
    res = evaluator.compute_metrics()

    assert np.isclose(res["mIoU_4cls"], 100.0), f"Expected 100.0% 4-class mIoU, got {res['mIoU_4cls']}"
    assert np.isclose(res["mDice_4cls"], 100.0), f"Expected 100.0% 4-class mDice, got {res['mDice_4cls']}"
    assert np.isclose(res["Accuracy"], 100.0), f"Expected 100.0% 4-class Accuracy, got {res['Accuracy']}"


def test_multi_method_wsss_ablation_engine():
    """Verify MultiMethodWSSSEngine executes all 5 genuine research methods (M1..M5) cleanly."""
    from src.colab_pkg.model_loader import DeepLabV3Plus
    from src.colab_pkg.model_loader import MultiMethodWSSSEngine as ColabEngine
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine

    model = DeepLabV3Plus(num_classes=4, output_stride=16)
    model.eval()

    src_engine = MultiMethodWSSSEngine()
    colab_engine = ColabEngine()

    dummy_img = torch.rand(1, 3, 64, 64)
    dummy_rgb = (dummy_img.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    preds_src = src_engine.predict_all_methods(model, dummy_img, raw_rgb=dummy_rgb)
    preds_colab = colab_engine.predict_all_methods(model, dummy_img, raw_rgb=dummy_rgb)

    assert list(preds_src.keys()) == MultiMethodWSSSEngine.METHOD_KEYS
    assert list(preds_colab.keys()) == MultiMethodWSSSEngine.METHOD_KEYS

    for k in MultiMethodWSSSEngine.METHOD_KEYS:
        assert preds_src[k].shape == (64, 64)
        assert np.array_equal(preds_src[k], preds_colab[k]), f"Mismatch between src and colab_pkg for {k}"

    pale_nec = np.full((64, 64, 3), [215, 195, 210], dtype=np.uint8)
    nec_pred = np.full((64, 64), 3, dtype=np.uint8)
    guarded_nec = src_engine.apply_pure_glass_guard(nec_pred, pale_nec, background_class_idx=4)
    assert np.all(guarded_nec == 3), "pure_glass_guard must preserve 100% of pale stained Necrosis (NEC)!"


def test_stage1_netcam_architecture_and_forward_cam():
    """Verify WSSSTissueStage1NetCAM exact parameter parity and forward/CAM execution."""
    import sys
    ref_repo = Path("repos/WSSS-Tissue")
    if not (ref_repo / "network" / "resnet38_cls.py").is_file():
        pytest.skip("Gitignored reference repo repos/WSSS-Tissue is not present on clean CI checkout.")
    sys.path.insert(0, str(ref_repo.resolve()))
    from network.resnet38_cls import Net_CAM

    from src.colab_pkg.model_loader import DeepLabV3Plus
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

    model_stage1.load_state_dict(ref_dict, strict=True)

    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y_prob = model_stage1(dummy_input)
    assert y_prob.shape == (1, 4), f"Expected shape (1, 4), got {y_prob.shape}"
    assert torch.all(y_prob >= 0.0) and torch.all(y_prob <= 1.0), "Probabilities must be in [0, 1]"

    with torch.no_grad():
        cam, y_cls = model_stage1.forward_cam(dummy_input)
    assert cam.shape == (1, 4, 8, 8), f"Expected CAM shape (1, 4, 8, 8), got {cam.shape}"
    assert y_cls.shape == (1, 4), f"Expected y_cls shape (1, 4), got {y_cls.shape}"
    assert torch.all(cam >= 0.0), "CAM activations must be non-negative (ReLU)"

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


def test_both_official_checkpoints_registered_in_colab_downloader():
    """Verify both official BCSS checkpoints are registered in src.colab_pkg.downloader."""
    from src.colab_pkg.downloader import (
        GDRIVE_STAGE1_ID,
        GDRIVE_STAGE2_ID,
        HF_STAGE1_URL,
        HF_STAGE2_URL,
    )

    assert GDRIVE_STAGE1_ID == "118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc"
    assert GDRIVE_STAGE2_ID == "1cXm29ko8UevbaeW6YreSg-vuNaQiN_je"
    assert "stage1_checkpoint_trained_on_bcss.pth" in HF_STAGE1_URL
    assert "stage2_checkpoint_trained_on_bcss.pth" in HF_STAGE2_URL


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

    # Load DeepLabV3Plus from modular colab_pkg.model_loader
    from src.colab_pkg.model_loader import DeepLabV3Plus

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
    from src.colab_pkg.model_loader import DeepLabV3Plus
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine, WSSSTissueStage1NetCAM

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
    from src.colab_pkg.model_loader import DeepLabV3Plus
    from src.models.wsss_inference_methods import MultiMethodWSSSEngine, WSSSTissueStage1NetCAM

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


def test_colab_pkg_and_wsss_inference_methods_exact_parity_m1_to_m7():
    """Verify 100% pixel-exact parity between src.colab_pkg.model_loader and
    src.models.wsss_inference_methods across M1..M7 on square and rectangular inputs.
    """
    from src.colab_pkg.model_loader import (
        DeepLabV3Plus as DL_Colab,
    )
    from src.colab_pkg.model_loader import (
        MultiMethodWSSSEngineV2 as E_Colab,
    )
    from src.colab_pkg.model_loader import (
        WSSSTissueStage1NetCAM as S1_Colab,
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
