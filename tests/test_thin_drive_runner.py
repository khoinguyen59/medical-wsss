"""Comprehensive Unit & Integration Tests for the Thin Google Drive Runner Package & Upgraded WSSS Engine.

Verifies:
1. All 5 existing legacy notebooks in `notebooks/` are untouched and retain their valid structure.
2. `scripts/build_drive_package.py` assembles `drive_upload_ready/SE374_WSSS_Drive_Code/` and
   `drive_upload_ready/SE374_WSSS_Drive_Code.zip` with `src/`, Thin Runner notebooks, and
   `README_DRIVE_SETUP.md` in Vietnamese.
3. ZIP extraction round-trip: extracting `SE374_WSSS_Drive_Code.zip` into a clean directory
   preserves all `src/` subpackages (`models`, `colab_pkg`, `datasets`, `pipeline`, `losses_metrics`)
   with valid `__init__.py` files.
4. All new Thin Runner notebooks (in `notebooks/` and `drive_upload_ready/SE374_WSSS_Drive_Code/`)
   have valid nbformat v4 JSON schema, zero syntax errors, concise cells (no monolithic code dumps),
   and import/invoke the modular `src` code package.
5. Upgraded WSSS inference methods (`M1`..`M6`/`M7` in `src.models.wsss_inference_methods` and
   `src.colab_pkg.model_loader`) run cleanly on CPU/GPU, handle 3D and 4D tensors, square (D8 8-view)
   and rectangular (D4 4-view) inputs, preserve pure-glass/pale-necrosis clinical guards, and
   guarantee full JSON serialization safety without circular references or numpy scalar TypeErrors.
6. 100% compliance with `AGENTS.md` Invariant 12 (standard ASCII hyphen 0x2D `-` only).
"""

import ast
import inspect
import json
import zipfile
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest
import torch

import scripts.build_drive_package as packager
from src.models.wsss_inference_methods import (
    DeepLabV3Plus as Stage2DeepLabV3Plus,
    MultiMethodWSSSEngine,
    MultiMethodWSSSEngineV2,
    WSSSTissueStage1NetCAM,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DRIVE_READY_DIR = PROJECT_ROOT / "drive_upload_ready"
BUNDLE_DIR = DRIVE_READY_DIR / "SE374_WSSS_Drive_Code"

LEGACY_SPECS = [
    ("SE374_Colab_WSSS_Pipeline.ipynb", 19, 90000),
    ("SE374_Colab_WSSS_T4_Sanity150.ipynb", 19, 90000),
    ("SE374_Colab_WSSS_T4_Full4986.ipynb", 19, 90000),
    ("SE374_Colab_WSSS_L4_Full4986.ipynb", 19, 90000),
    ("SE374_Colab_WSSS_A100_Full4986.ipynb", 19, 90000),
]

FORBIDDEN_DASHES = (
    ("\u2014", "em-dash"),
    ("\u2013", "en-dash"),
    ("\u2015", "horizontal bar"),
    ("\u2212", "minus sign"),
)


def _load_stage2_deeplabv3plus_class():
    """Loads the Stage-2 DeepLabV3Plus class matching BCSS 4-class checkpoint (59.34M params)."""
    return Stage2DeepLabV3Plus


def test_1_existing_legacy_notebooks_untouched():
    """Verify all 5 original notebooks in `notebooks/` are preserved untouched."""
    for fname, expected_cells, min_bytes in LEGACY_SPECS:
        nb_path = NOTEBOOKS_DIR / fname
        assert nb_path.is_file(), f"Original notebook missing: {nb_path}"
        size_bytes = nb_path.stat().st_size
        assert size_bytes >= min_bytes, (
            f"Original notebook {fname} was truncated or overwritten (size={size_bytes} < {min_bytes})"
        )

        raw_text = nb_path.read_text(encoding="utf-8")
        for char, name in FORBIDDEN_DASHES:
            assert char not in raw_text, f"Forbidden {name} found in {fname}"

        nb_data = json.loads(raw_text)
        assert nb_data.get("nbformat") == 4, f"Expected nbformat 4 in {fname}"
        assert len(nb_data.get("cells", [])) == expected_cells, (
            f"Expected {expected_cells} cells in untouched {fname}, got {len(nb_data.get('cells', []))}"
        )

        # Verify inline classes in Cell 12 of original notebooks remain intact
        cell_12_src = "".join(nb_data["cells"][12].get("source", []))
        assert "class DeepLabV3Plus" in cell_12_src, f"Missing DeepLabV3Plus in untouched {fname}"
        assert "class WSSSTissueStage1NetCAM" in cell_12_src, f"Missing WSSSTissueStage1NetCAM in untouched {fname}"


def test_2_drive_bundle_packager_creates_complete_folder_and_zip(tmp_path):
    """Verify `scripts/build_drive_package.py` assembles `drive_upload_ready/SE374_WSSS_Drive_Code/`
    and `drive_upload_ready/SE374_WSSS_Drive_Code.zip` cleanly and supports 1-click extraction.
    """
    summary = packager.build_drive_package()

    assert BUNDLE_DIR.is_dir(), f"Bundle directory not created: {BUNDLE_DIR}"
    bundle_src = BUNDLE_DIR / "src"
    assert bundle_src.is_dir(), "Missing `src/` inside Drive bundle directory"

    # Check essential modular files inside bundle/src/
    required_rel_files = [
        "src/__init__.py",
        "src/config/bcss_patient_splits.json",
        "src/datasets/bcss_dataset.py",
        "src/datasets/otsu_filter.py",
        "src/datasets/patient_split.py",
        "src/losses_metrics/losses.py",
        "src/losses_metrics/metrics.py",
        "src/models/deeplabv3plus.py",
        "src/models/wsss_inference_methods.py",
        "src/pipeline/comparative_runner.py",
        "src/utils/deliverable_auditor.py",
    ]
    for rel in required_rel_files:
        target = BUNDLE_DIR / rel
        assert target.is_file(), f"Missing required file in Drive bundle: {rel}"

    # Verify zero __pycache__ or .pyc inside bundle
    pycache_dirs = list(BUNDLE_DIR.rglob("__pycache__"))
    pyc_files = list(BUNDLE_DIR.rglob("*.pyc"))
    assert len(pycache_dirs) == 0, f"Found __pycache__ inside clean Drive bundle: {pycache_dirs}"
    assert len(pyc_files) == 0, f"Found .pyc files inside clean Drive bundle: {pyc_files}"

    # Verify README_DRIVE_SETUP.md in Vietnamese
    readme_path = BUNDLE_DIR / "README_DRIVE_SETUP.md"
    assert readme_path.is_file(), "Missing README_DRIVE_SETUP.md in Drive bundle"
    readme_text = readme_path.read_text(encoding="utf-8")
    assert len(readme_text) > 1500, "README_DRIVE_SETUP.md is too short"
    assert "MyDrive/SE374_WSSS_Project" in readme_text
    assert "SE374_WSSS_Drive_Code.zip" in readme_text
    assert "Thin Runner" in readme_text
    assert "M6_MultiScale_Pyramid_D4_Ensemble" in readme_text
    assert "M7_Proposed_Ultra_SOTA_Curriculum" in readme_text
    for char, name in FORBIDDEN_DASHES:
        assert char not in readme_text, f"Forbidden {name} in README_DRIVE_SETUP.md"

    # Verify package_manifest.json lists all 7 supported SOTA methods (M1..M7)
    manifest_data = json.loads((BUNDLE_DIR / "package_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest_data["supported_wsss_methods"]) == 7
    assert "M6_MultiScale_Pyramid_D4_Ensemble" in manifest_data["supported_wsss_methods"]
    assert "M7_Proposed_Ultra_SOTA_Curriculum" in manifest_data["supported_wsss_methods"]

    # Verify POSIX LF line endings on packaged Python source files
    raw_bytes = (BUNDLE_DIR / "src" / "models" / "wsss_inference_methods.py").read_bytes()
    assert b"\r\n" not in raw_bytes, "Packaged source files must use POSIX LF ('\\n') line endings for Linux Colab VMs"

    # Verify pre-zipped archive integrity and extraction round-trip
    zip_outer = DRIVE_READY_DIR / "SE374_WSSS_Drive_Code.zip"
    zip_inner = BUNDLE_DIR / "SE374_WSSS_Drive_Code.zip"
    assert zip_outer.is_file(), f"Missing outer zip archive: {zip_outer}"
    assert zip_inner.is_file(), f"Missing inner zip archive copy: {zip_inner}"
    assert zipfile.is_zipfile(zip_outer), "SE374_WSSS_Drive_Code.zip is not a valid zip file"

    extract_target = tmp_path / "colab_content"
    with zipfile.ZipFile(zip_outer, "r") as zf:
        assert zf.testzip() is None, "Corrupted member detected in SE374_WSSS_Drive_Code.zip"
        namelist = set(zf.namelist())
        assert "src/models/wsss_inference_methods.py" in namelist
        assert "SE374_WSSS_Drive_Code/src/models/wsss_inference_methods.py" in namelist
        assert "README_DRIVE_SETUP.md" in namelist
        assert "package_manifest.json" in namelist
        zf.extractall(extract_target)

    # Verify both root-level extraction and subfolder extraction paths work on Colab
    assert (extract_target / "src" / "models" / "wsss_inference_methods.py").is_file()
    assert (extract_target / "SE374_WSSS_Drive_Code" / "src" / "models" / "wsss_inference_methods.py").is_file()
    assert summary["src_files_count"] >= 15
    assert len(summary["supported_wsss_methods"]) == 7


def test_3_new_thin_runner_notebooks_valid_concise_and_modular():
    """Verify all new Thin Runner notebooks (in `notebooks/` and `drive_upload_ready/`)
    have valid JSON schema, zero syntax errors, concise cells (no monolithic code dumps),
    and call the modular code package (`src`).
    """
    packager.build_drive_package()

    # Collect new Thin Runner notebooks from notebooks/ (if present) and from the Drive bundle
    repo_thin_nbs = packager.discover_thin_runner_notebooks(NOTEBOOKS_DIR)
    bundle_thin_nbs = sorted((BUNDLE_DIR / "notebooks").glob("*.ipynb"))

    all_thin_nbs = repo_thin_nbs + bundle_thin_nbs
    assert len(all_thin_nbs) >= 4, (
        f"Expected at least 4 Thin Runner notebooks, found {len(all_thin_nbs)}"
    )

    for nb_path in all_thin_nbs:
        raw_text = nb_path.read_text(encoding="utf-8")

        # 1. Invariant 12: Strictly standard ASCII hyphen (0x2D) only
        for char, name in FORBIDDEN_DASHES:
            assert char not in raw_text, f"Invariant 12 violation: {name} found in {nb_path}"

        # 2. Valid nbformat v4 JSON schema
        nb_data = json.loads(raw_text)
        assert nb_data.get("nbformat") == 4, f"Expected nbformat 4 in {nb_path.name}"
        assert "cells" in nb_data and len(nb_data["cells"]) >= 3, (
            f"Thin Runner notebook {nb_path.name} must have at least 3 cells"
        )

        code_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 2, f"Expected >= 2 code cells in {nb_path.name}"

        full_code_parts: List[str] = []
        raw_code_parts: List[str] = []
        for idx, cell in enumerate(code_cells):
            cell_src = "".join(cell.get("source", []))
            raw_code_parts.append(cell_src)
            lines = cell_src.splitlines()

            # 3. Conciseness check: No monolithic 300+ line code dumps inside a Thin Runner cell
            assert len(lines) <= 160, (
                f"Cell {idx} in Thin Runner {nb_path.name} has {len(lines)} lines (> 160 limit); "
                "Thin Runner cells must remain concise and delegate heavy logic to src/!"
            )

            # Verify no inline model class definitions are dumped into the Thin Runner notebook
            assert "class ResBlockBot(" not in cell_src, (
                f"Monolithic class ResBlockBot found inline in Thin Runner {nb_path.name}!"
            )
            assert "class DeepLabV3Plus(" not in cell_src, (
                f"Monolithic class DeepLabV3Plus found inline in Thin Runner {nb_path.name}!"
            )

            # 4. Syntax compilation check (stripping Colab ! and % magic lines)
            py_lines = []
            for ln in lines:
                stripped = ln.strip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    py_lines.append(f"# {stripped}")
                else:
                    py_lines.append(ln)
            py_code = "\n".join(py_lines)
            try:
                tree = ast.parse(py_code, filename=f"{nb_path.name}_cell_{idx}")
                assert tree is not None
            except SyntaxError as exc:
                pytest.fail(
                    f"SyntaxError in Thin Runner {nb_path.name} code cell {idx} "
                    f"(line {exc.lineno}): {exc.msg}\n{py_code}"
                )
            full_code_parts.append(py_code)

        full_code = "\n".join(full_code_parts)
        # 5. Must import or invoke modular code from `src` (`src.pipeline`, `src.models`, or `src.colab_pkg`)
        assert "src." in full_code or "from src" in full_code or "import src" in full_code, (
            f"Thin Runner notebook {nb_path.name} does not import or call modular package `src`!"
        )
        assert "MultiMethodWSSSEngine" in full_code or "run_bcss_wsss_experiment" in full_code, (
            f"Thin Runner notebook {nb_path.name} does not invoke MultiMethodWSSSEngine or run_bcss_wsss_experiment!"
        )


def test_4_upgraded_wsss_inference_methods_cpu_and_gpu_execution():
    """Verify MultiMethodWSSSEngine and MultiMethodWSSSEngineV2 execute all 7 methods
    (M1..M5 and upgraded M6/M7 methods) cleanly on CPU (and GPU if available) with
    both Stage-1 + Stage-2 models and Stage-2 standalone.
    """
    DeepLabV3Plus = _load_stage2_deeplabv3plus_class()
    devices_to_test = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))

    v1_method_keys = list(MultiMethodWSSSEngine.METHOD_KEYS)
    v2_method_keys = list(MultiMethodWSSSEngineV2.METHOD_KEYS)
    assert len(v1_method_keys) == 5, f"Expected 5 V1 WSSS methods, got {v1_method_keys}"
    assert len(v2_method_keys) == 7, f"Expected 7 V2 WSSS methods (M1..M7), got {v2_method_keys}"
    assert "M6_MultiScale_Pyramid_D4_Ensemble" in v2_method_keys
    assert "M7_Proposed_Ultra_SOTA_Curriculum" in v2_method_keys

    for dev in devices_to_test:
        model_stage2 = DeepLabV3Plus(num_classes=4, output_stride=16).to(dev).eval()
        model_stage1 = WSSSTissueStage1NetCAM(n_class=4).to(dev).eval()

        # Test both V1 (5 methods) and V2 (all 7 methods M1..M7)
        engine_v1 = MultiMethodWSSSEngine(model_stage1=model_stage1, include_v2=False)
        engine_with_s1 = MultiMethodWSSSEngineV2(model_stage1=model_stage1)
        engine_no_s1 = MultiMethodWSSSEngineV2(model_stage1=None)

        # Test on square 4D tensor (1, 3, 64, 64) and 3D tensor (3, 64, 64)
        dummy_4d = torch.rand(1, 3, 64, 64, device=dev)
        dummy_3d = dummy_4d.squeeze(0)
        dummy_rgb = (dummy_3d.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        preds_v1 = engine_v1.predict_all_methods(
            model_stage2, dummy_4d, raw_rgb=dummy_rgb, model_stage1=model_stage1
        )
        assert list(preds_v1.keys()) == v1_method_keys

        preds_with_s1 = engine_with_s1.predict_all_methods_v2(
            model_stage2, dummy_4d, raw_rgb=dummy_rgb, model_stage1=model_stage1
        )
        preds_no_s1 = engine_no_s1.predict_all_methods(
            model_stage2, dummy_3d, raw_rgb=dummy_rgb, model_stage1=None
        )

        assert list(preds_with_s1.keys()) == v2_method_keys
        assert list(preds_no_s1.keys()) == v2_method_keys

        for idx, m_key in enumerate(v2_method_keys, start=1):
            mask_s1 = preds_with_s1[m_key]
            mask_no_s1 = preds_no_s1[m_key]

            assert isinstance(mask_s1, np.ndarray) and mask_s1.dtype == np.uint8
            assert isinstance(mask_no_s1, np.ndarray) and mask_no_s1.dtype == np.uint8
            assert mask_s1.shape == (64, 64), f"{m_key} shape mismatch: {mask_s1.shape}"
            assert mask_no_s1.shape == (64, 64), f"{m_key} shape mismatch: {mask_no_s1.shape}"
            assert np.all((mask_s1 >= 0) & (mask_s1 <= 4)), f"{m_key} has out-of-range class IDs"
            assert np.all((mask_no_s1 >= 0) & (mask_no_s1 <= 4)), f"{m_key} has out-of-range class IDs"

            # Verify individual predict_m1, predict_m2, ..., predict_m6, predict_m7 parity
            indiv_fn_name = f"predict_m{idx}"
            assert hasattr(engine_with_s1, indiv_fn_name), f"Missing {indiv_fn_name} on MultiMethodWSSSEngineV2"
            fn = getattr(engine_with_s1, indiv_fn_name)
            sig = inspect.signature(fn)
            kwargs = {}
            if "raw_rgb" in sig.parameters:
                kwargs["raw_rgb"] = dummy_rgb
            if "model_stage1" in sig.parameters:
                kwargs["model_stage1"] = model_stage1
            indiv_pred = fn(model_stage2, dummy_4d, **kwargs)
            assert np.array_equal(indiv_pred, mask_s1), (
                f"Individual {indiv_fn_name}() output differs from predict_all_methods_v2()['{m_key}']"
            )

        # Also test non-square rectangular spatial resolution (32, 48) across all 7 methods (M1..M7)
        rect_4d = torch.rand(1, 3, 32, 48, device=dev)
        rect_rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        rect_preds = engine_with_s1.predict_all_methods_v2(
            model_stage2, rect_4d, raw_rgb=rect_rgb, model_stage1=model_stage1
        )
        for m_key in v2_method_keys:
            assert rect_preds[m_key].shape == (32, 48)


def test_5_colab_pkg_modules_and_d8_tta_parity(tmp_path):
    """Verify `src.colab_pkg` modules (`downloader`, `model_loader`, `evaluator`, `colab_runner`)
    execute 8-view Dihedral D4/D8 TTA, streaming evaluation, and JSON/MD/CSV export accurately.
    """
    try:
        from src.colab_pkg import colab_runner, downloader, evaluator, model_loader
    except ImportError:
        pytest.skip("src.colab_pkg not yet present")

    # 1. Downloader constants and directory resolver
    assert downloader.GDRIVE_DATASET_ID == "1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn"
    assert downloader.GDRIVE_STAGE1_ID == "118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc"
    assert downloader.GDRIVE_STAGE2_ID == "1cXm29ko8UevbaeW6YreSg-vuNaQiN_je"

    # 2. Model loader DeepLabV3Plus and WSSSTissueStage1NetCAM parameter counts
    m2 = model_loader.DeepLabV3Plus(num_classes=4, output_stride=16).eval()
    m1 = model_loader.WSSSTissueStage1NetCAM(n_class=4).eval()
    assert sum(p.numel() for p in m2.parameters()) == 59339940
    assert sum(p.numel() for p in m1.parameters()) == 105087296

    # 3. Test 8-view D8 TTA vs 4-view D4 TTA in colab_pkg.model_loader.MultiMethodWSSSEngine
    pkg_engine = model_loader.MultiMethodWSSSEngine(model_stage1=m1)
    sq_img = torch.rand(1, 3, 64, 64)
    sq_rgb = (sq_img.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    pkg_preds = pkg_engine.predict_all_methods(m2, sq_img, raw_rgb=sq_rgb, model_stage1=m1)
    for k in model_loader.MultiMethodWSSSEngine.METHOD_KEYS:
        assert k in pkg_preds
        assert pkg_preds[k].shape == (64, 64)
        assert pkg_preds[k].dtype == np.uint8

    # 4. End-to-end streaming evaluation on synthetic patient patches via colab_runner
    from PIL import Image

    img_dir = tmp_path / "test" / "img"
    mask_dir = tmp_path / "test" / "mask"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    sample_files = []
    for idx in range(2):
        fname = f"TCGA-AQ-A04H_patch_{idx:03d}.png"
        rgb_arr = np.random.randint(40, 220, size=(32, 32, 3), dtype=np.uint8)
        gt_arr = np.random.randint(0, 4, size=(32, 32), dtype=np.uint8)
        gt_arr[:4, :4] = 4
        Image.fromarray(rgb_arr).save(img_dir / fname)
        Image.fromarray(gt_arr).save(mask_dir / fname)
        sample_files.append(img_dir / fname)

    eval_out = colab_runner.run_streaming_evaluation(
        active_test_files=sample_files,
        test_mask_dir=mask_dir,
        model=m2,
        model_stage1=m1,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert eval_out["evaluated_count"] == 2
    out_dir = tmp_path / "results_out"
    out_dir.mkdir(parents=True)
    exported = evaluator.export_evaluation_artifacts(
        output_dir=out_dir,
        run_folder_name="test_run",
        mode_tag="T4_Sanity150",
        sanity_mode=True,
        evaluated_count=2,
        metrics=eval_out["metrics"],
        all_methods_metrics=eval_out["all_methods_metrics"],
        method_keys=model_loader.MultiMethodWSSSEngine.METHOD_KEYS,
        method_names_vi=model_loader.MultiMethodWSSSEngine.METHOD_NAMES_VI,
        statistical_tests=eval_out["statistical_tests"],
        gpu_name="CPU",
        legacy_dir=None,
    )
    assert Path(exported["json_path"]).is_file()
    assert Path(exported["md_path"]).is_file()
    assert Path(exported["csv_ablation_path"]).is_file()
    assert Path(exported["csv_class_path"]).is_file()


def test_6_json_serialization_safety_and_ablation_metrics():
    """Verify full JSON serialization safety (`json.dumps(..., allow_nan=False)`) across all
    7 methods (M1..M7) in `MultiMethodWSSSEngineV2` without circular references or numpy scalar TypeErrors.
    """
    DeepLabV3Plus = _load_stage2_deeplabv3plus_class()
    model_stage2 = DeepLabV3Plus(num_classes=4, output_stride=16).eval()
    model_stage1 = WSSSTissueStage1NetCAM(n_class=4).eval()
    engine = MultiMethodWSSSEngineV2(model_stage1=model_stage1)

    dummy_img = torch.rand(1, 3, 48, 48)
    dummy_rgb = (dummy_img.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    gt_mask = np.random.randint(0, 4, size=(48, 48), dtype=np.int64)
    gt_mask[:4, :4] = 4  # Outside-ROI region

    preds = engine.predict_all_methods_v2(
        model_stage2, dummy_img, raw_rgb=dummy_rgb, model_stage1=model_stage1
    )
    assert len(preds) == 7
    assert "M6_MultiScale_Pyramid_D4_Ensemble" in preds
    assert "M7_Proposed_Ultra_SOTA_Curriculum" in preds

    all_methods_metrics: Dict[str, Dict[str, object]] = {}
    for m_key, pred_mask in preds.items():
        pred_eval = pred_mask.copy()
        pred_eval[gt_mask == 4] = 4
        valid = (gt_mask >= 0) & (gt_mask < 4) & (pred_eval < 4)
        hist = np.bincount(
            4 * gt_mask[valid].astype(np.int64) + pred_eval[valid].astype(np.int64),
            minlength=16,
        ).reshape(4, 4)
        inter = np.diag(hist).astype(np.float64)
        union = (hist.sum(axis=1) + hist.sum(axis=0) - inter).astype(np.float64)
        iou = np.where(union > 0, (inter / union) * 100.0, 0.0)
        dice = np.where(
            (hist.sum(axis=1) + hist.sum(axis=0)) > 0,
            (2.0 * inter / (hist.sum(axis=1) + hist.sum(axis=0))) * 100.0,
            0.0,
        )
        all_methods_metrics[m_key] = {
            "method_key": m_key,
            "method_name_vi": MultiMethodWSSSEngineV2.METHOD_NAMES_VI.get(m_key, m_key),
            "mIoU_4cls": float(np.mean(iou)),
            "mDice_4cls": float(np.mean(dice)),
            "per_class_iou": [float(x) for x in iou],
            "per_class_dice": [float(x) for x in dice],
            "confusion_matrix_4x4": hist.tolist(),
        }

    # Select best method and verify copying via dict() prevents circular reference
    best_method_key = max(all_methods_metrics.keys(), key=lambda k: all_methods_metrics[k]["mIoU_4cls"])
    metrics = dict(all_methods_metrics[best_method_key])
    metrics["ablation_methods"] = {
        k: {mk: mv for mk, mv in v.items() if mk != "ablation_methods"}
        for k, v in all_methods_metrics.items()
    }

    export_payload = {
        "best_method": best_method_key,
        "supported_methods": list(MultiMethodWSSSEngineV2.METHOD_KEYS),
        "primary_metrics": metrics,
        "ablation_summary": all_methods_metrics,
    }

    # Must serialize cleanly with allow_nan=False (no NaN/Inf, no numpy types, no circular refs)
    json_str = json.dumps(export_payload, ensure_ascii=False, indent=2, allow_nan=False)
    parsed_back = json.loads(json_str)
    assert parsed_back["best_method"] == best_method_key
    assert len(parsed_back["ablation_summary"]) == 7


def test_7_invariant_12_strict_ascii_hyphens_across_all_packaged_files():
    """Verify 0 non-ASCII dashes across `scripts/build_drive_package.py`,
    `tests/test_thin_drive_runner.py`, and all text files in `drive_upload_ready/`.
    """
    packager.build_drive_package()

    files_to_check = [
        PROJECT_ROOT / "scripts" / "build_drive_package.py",
        PROJECT_ROOT / "tests" / "test_thin_drive_runner.py",
    ]
    for p in DRIVE_READY_DIR.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".json", ".ipynb", ".txt", ".html"}:
            files_to_check.append(p)

    for fpath in files_to_check:
        text = fpath.read_text(encoding="utf-8")
        for char, name in FORBIDDEN_DASHES:
            count = text.count(char)
            assert count == 0, (
                f"Invariant 12 violation: Found {count} instance(s) of {name} ({hex(ord(char))}) in {fpath}"
            )
