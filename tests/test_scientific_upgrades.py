"""
Unit and Integration Tests for Scientific Research Upgrades in SE374.R11.

Covers:
1. Multi-Magnification Simulation & Coarse-to-Fine Adaptive Router (Pareto Frontier).
2. Val-to-Test Evaluation Protocol, FwIoU, and Boundary F1 (BF1).
3. Patient-Level Clinical Biomarker Correlation (Pearson r, Spearman rho, CAP risk).
4. Cross-Dataset LUAD-HistoSeg Downloader & Model Loading Architecture.
5. 100% Tissue Coverage WSI Stitching & Whole-ROI Evaluation.
6. Publication Figures (Pareto curve, Bland-Altman regression, Whole-ROI comparison).

All hyphens strictly use standard ASCII 0x2D ('-').
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.colab_pkg.downloader import (
    extract_patient_id,
    resolve_split_dirs,
)
from src.colab_pkg.evaluator import (
    ColabSegmentationEvaluator,
    run_biomarker_correlation_study,
    run_val_to_test_scientific_protocol,
)
from src.colab_pkg.visualizer import (
    plot_biomarker_bland_altman_and_regression,
    plot_coarse_to_fine_pareto_curve,
    plot_whole_roi_ground_truth_comparison,
)
from src.losses_metrics.metrics import (
    compute_boundary_f1,
    compute_fwiou,
    compute_patient_biomarker_correlation,
    compute_pixel_accuracy,
)
from src.models.coarse_to_fine_router import (
    CoarseToFineAdaptiveRouter,
    simulate_magnification_patch,
)
from src.pipeline.wsi_lowmag_highmag_pipeline import (
    WSILowMagHighMagPipeline,
)


class DummySegmentationModel(nn.Module):
    """Lightweight dummy segmentor outputting deterministic 4-class logits."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        logits = torch.zeros((b, self.num_classes, h, w), device=x.device)
        logits[:, 0, :, :] = 2.0  # TUM prior
        logits[:, 1, :, :] = 1.0  # STR prior
        return logits

    def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        cam = torch.zeros((b, self.num_classes, max(1, h // 8), max(1, w // 8)), device=x.device)
        cam[:, 0, :, :] = 3.0
        cam[:, 1, :, :] = 2.5
        y_cls = torch.full((b, self.num_classes), 0.85, device=x.device)
        return cam, y_cls


class DummyWSSSEngine:
    """Mock engine simulating MultiMethodWSSSEngineV2."""

    def predict_m1(self, model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
        b, c, h, w = tensor.shape
        pred = np.zeros((h, w), dtype=np.uint8)
        pred[: h // 2, :] = 0  # Tumor
        pred[h // 2 :, :] = 1  # Stroma
        return pred

    def predict_m6(
        self,
        model: nn.Module,
        tensor: torch.Tensor,
        raw_rgb: Optional[np.ndarray] = None,
        model_stage1: Optional[nn.Module] = None,
    ) -> np.ndarray:
        return self.predict_m1(model, tensor)

    def predict_all_methods(self, *args: Any, **kwargs: Any) -> Dict[str, np.ndarray]:
        t = kwargs.get("img_tensor", args[1] if len(args) > 1 else None)
        h, w = (224, 224) if t is None else (t.shape[2], t.shape[3])
        p = np.zeros((h, w), dtype=np.uint8)
        p[: h // 2, :] = 0
        p[h // 2 :, :] = 1
        return {
            "M1_Baseline_SingleScale": p,
            "M6_MultiScale_Pyramid_D4_Ensemble": p,
        }


def test_fwiou_and_boundary_f1_computation() -> None:
    """Verifies FwIoU, Pixel Accuracy, and Boundary F1 on known synthetic masks."""
    h, w = 100, 100
    gt = np.zeros((h, w), dtype=np.int64)
    gt[:50, :] = 0  # TUM
    gt[50:, :] = 1  # STR

    pred = gt.copy()
    # Perfect match
    fwiou_perfect = compute_fwiou(pred, gt, num_classes=5)
    pa_perfect = compute_pixel_accuracy(pred, gt)
    bf1_perfect = compute_boundary_f1(pred, gt, num_classes=5, tolerance_px=2)

    assert fwiou_perfect > 0.99
    assert pa_perfect == 1.0
    assert bf1_perfect["BF1"] > 95.0

    # Perturbed prediction
    pred[:20, :] = 1
    fwiou_pert = compute_fwiou(pred, gt, num_classes=5)
    pa_pert = compute_pixel_accuracy(pred, gt)
    bf1_pert = compute_boundary_f1(pred, gt, num_classes=5, tolerance_px=2)

    assert fwiou_pert < fwiou_perfect
    assert pa_pert == 0.80
    assert bf1_pert["BF1"] < bf1_perfect["BF1"]

    # Verify ColabSegmentationEvaluator.update() handles dict returns without TypeError and computes non-zero Boundary_F1
    evaluator_perfect = ColabSegmentationEvaluator(num_classes=5)
    evaluator_perfect.update(gt, gt, patient_id="TCGA-AQ-A04H")
    metrics_perfect = evaluator_perfect.compute_metrics()
    assert metrics_perfect["Boundary_F1"] == 100.0, f"Expected 100.0% Boundary_F1 on identical masks, got {metrics_perfect['Boundary_F1']}"
    assert metrics_perfect["FwIoU"] > 99.0, f"Expected >99% FwIoU on identical masks, got {metrics_perfect['FwIoU']}"

    # Verify perturbed prediction reduces Boundary_F1 while remaining non-zero
    evaluator_pert = ColabSegmentationEvaluator(num_classes=5)
    evaluator_pert.update(pred, gt, patient_id="TCGA-AQ-A04H")
    metrics_pert = evaluator_pert.compute_metrics()
    assert 0.0 < metrics_pert["Boundary_F1"] < metrics_perfect["Boundary_F1"]


def test_coarse_to_fine_router_and_pareto() -> None:
    """Verifies magnification simulation and Pareto efficiency curve calculation."""
    rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sim_10x = simulate_magnification_patch(rgb, target_mag=10, base_mag=40)
    assert sim_10x.shape == rgb.shape
    assert sim_10x.dtype == np.uint8

    router = CoarseToFineAdaptiveRouter()
    prob_map = np.full((5, 224, 224), 0.2, dtype=np.float32)
    pred_mask = np.zeros((224, 224), dtype=np.uint8)
    score = router.compute_patch_difficulty_score(prob_map, pred_mask)
    assert 0.0 <= score <= 1.0

    model = DummySegmentationModel()
    engine = DummyWSSSEngine()
    patches_40x = [rgb, rgb]
    patches_10x = [sim_10x, sim_10x]
    gt = np.zeros((224, 224), dtype=np.int64)
    gt[:112, :] = 0
    gt[112:, :] = 1
    gts = [gt, gt]

    pareto = router.evaluate_pareto_curve(
        engine=engine,
        model=model,
        patches_40x=patches_40x,
        patches_10x=patches_10x,
        ground_truths=gts,
        budget_steps=5,
        device=torch.device("cpu"),
    )

    assert len(pareto["budgets"]) == 5
    assert len(pareto["adaptive_miou"]) == 5
    assert len(pareto["random_miou"]) == 5
    assert pareto["auc_gain"] >= 0.0


def test_biomarker_correlation_computation() -> None:
    """Verifies Pearson r, Spearman rho, MAE, and CAP concordance calculation."""
    records = [
        {"patient_id": "P1", "pred_vtp": 30.0, "gt_vtp": 32.0, "pred_tsr": 60.0, "gt_tsr": 65.0},
        {"patient_id": "P2", "pred_vtp": 50.0, "gt_vtp": 48.0, "pred_tsr": 40.0, "gt_tsr": 42.0},
        {"patient_id": "P3", "pred_vtp": 10.0, "gt_vtp": 12.0, "pred_tsr": 20.0, "gt_tsr": 25.0},
        {"patient_id": "P4", "pred_vtp": 70.0, "gt_vtp": 68.0, "pred_tsr": 80.0, "gt_tsr": 78.0},
    ]

    res = compute_patient_biomarker_correlation(records)
    assert res["patient_count"] == 4
    assert res["vtp_pearson_r"] > 0.95
    assert res["tsr_pearson_r"] > 0.95
    assert res["vtp_mae"] < 3.0
    assert res["tsr_mae"] < 5.0
    assert res["cap_risk_concordance_pct"] == 100.0


def test_run_biomarker_correlation_study_with_csv(tmp_path: Path) -> None:
    """Verifies run_biomarker_correlation_study exports valid CSV artifact."""
    p_preds = {
        "TCGA-AQ-A04H": [np.zeros((50, 50), dtype=np.uint8)],
        "TCGA-C8-A1HJ": [np.ones((50, 50), dtype=np.uint8)],
    }
    p_gts = {
        "TCGA-AQ-A04H": [np.zeros((50, 50), dtype=np.int64)],
        "TCGA-C8-A1HJ": [np.ones((50, 50), dtype=np.int64)],
    }

    out_csv = tmp_path / "biomarker_correlation.csv"
    res = run_biomarker_correlation_study(p_preds, p_gts, output_csv_path=out_csv)

    assert out_csv.is_file()
    assert "correlation_metrics" in res
    assert "TCGA-AQ-A04H" in res["pred_biomarkers"]


def test_wsi_100_percent_tissue_coverage_and_priority() -> None:
    """Verifies 100% tissue coverage stitching and priority ordering in WSI pipeline."""
    pipeline = WSILowMagHighMagPipeline(
        patch_size=64,
        lowmag_target_size=(128, 128),
        tissue_threshold=0.10,
        engine=DummyWSSSEngine(),
        device="cpu",
    )

    # Synthetic 256x256 slide with tissue in center
    slide_rgb = np.full((256, 256, 3), 245, dtype=np.uint8)  # White glass
    slide_rgb[64:192, 64:192] = [180, 50, 120]  # Stained tissue core

    model = DummySegmentationModel()
    stitched_mask, report = pipeline.run_end_to_end_wsi(
        slide_input=slide_rgb,
        model_stage2=model,
        slide_id="TEST_SLIDE_01",
        max_patches_to_infer=2,  # Intentional small cap to test coarse base map
    )

    assert stitched_mask.shape == (128, 128)
    # The tissue core at (64:192, 64:192) downscaled to (32:96, 32:96) must NOT be class 4 (background)
    tissue_center = stitched_mask[32:96, 32:96]
    non_bg_fraction = np.mean(tissue_center != 4)
    assert non_bg_fraction > 0.85, "100% tissue coverage must prevent un-inferred tissue from being Background"
    assert report.total_possible_patches > 0


def test_evaluate_wsi_with_ground_truth_roi() -> None:
    """Verifies evaluate_wsi_with_ground_truth_roi against synthetic ROI."""
    pipeline = WSILowMagHighMagPipeline(device="cpu")
    gt = np.zeros((100, 100), dtype=np.int64)
    gt[:50, :] = 0  # TUM
    gt[50:, :] = 1  # STR
    pred = gt.copy()

    metrics = pipeline.evaluate_wsi_with_ground_truth_roi(pred, gt)
    assert metrics["mIoU_4cls"] == 100.0
    assert metrics["mDice_4cls"] == 100.0
    assert metrics["Pixel_Accuracy"] == 100.0
    assert metrics["FwIoU"] > 99.0
    assert metrics["Boundary_F1"] > 90.0


def test_scientific_visualization_figures(tmp_path: Path) -> None:
    """Verifies Pareto curve, Bland-Altman, and Whole-ROI comparison plotting."""
    pareto_data = {
        "budgets": [0.0, 0.25, 0.5, 0.75, 1.0],
        "adaptive_miou": [60.0, 71.0, 75.5, 77.8, 78.5],
        "random_miou": [60.0, 64.5, 69.2, 73.8, 78.5],
    }
    fig1, p1 = plot_coarse_to_fine_pareto_curve(pareto_data, output_dir=tmp_path)
    assert p1 is not None and p1.is_file()

    p_preds = {
        "P1": {"vtp": 30.0, "tsr": 60.0},
        "P2": {"vtp": 50.0, "tsr": 40.0},
        "P3": {"vtp": 70.0, "tsr": 80.0},
    }
    p_gts = {
        "P1": {"vtp": 32.0, "tsr": 65.0},
        "P2": {"vtp": 48.0, "tsr": 42.0},
        "P3": {"vtp": 68.0, "tsr": 78.0},
    }
    fig2, p2 = plot_biomarker_bland_altman_and_regression(p_preds, p_gts, output_dir=tmp_path)
    assert p2 is not None and p2.is_file()

    rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gt = np.zeros((100, 100), dtype=np.int64)
    pred = gt.copy()
    fig3, p3 = plot_whole_roi_ground_truth_comparison(rgb, pred, gt, output_dir=tmp_path)
    assert p3 is not None and p3.is_file()


def test_val_to_test_scientific_protocol(tmp_path: Path) -> None:
    """Verifies run_val_to_test_scientific_protocol on synthetic val and test cohorts."""
    val_img_dir = tmp_path / "val" / "img"
    val_mask_dir = tmp_path / "val" / "mask"
    test_img_dir = tmp_path / "test" / "img"
    test_mask_dir = tmp_path / "test" / "mask"

    val_img_dir.mkdir(parents=True)
    val_mask_dir.mkdir(parents=True)
    test_img_dir.mkdir(parents=True)
    test_mask_dir.mkdir(parents=True)

    val_files = []
    for i in range(2):
        img_p = val_img_dir / f"TCGA-AQ-A04H_patch_{i}.png"
        mask_p = val_mask_dir / f"TCGA-AQ-A04H_patch_{i}.png"
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img_p)
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(mask_p)
        val_files.append(img_p)

    test_files = []
    for i in range(2):
        img_p = test_img_dir / f"TCGA-C8-A1HJ_patch_{i}.png"
        mask_p = test_mask_dir / f"TCGA-C8-A1HJ_patch_{i}.png"
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img_p)
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(mask_p)
        test_files.append(img_p)

    model = DummySegmentationModel()
    engine = DummyWSSSEngine()

    res = run_val_to_test_scientific_protocol(
        engine=engine,
        model_stage2=model,
        val_files=val_files,
        test_files=test_files,
        val_mask_dir=val_mask_dir,
        test_mask_dir=test_mask_dir,
        device="cpu",
    )

    assert "val_metrics" in res
    assert "test_metrics" in res
    assert "generalization_gap" in res
    assert "protocol_passed" in res


def test_scientific_runner_notebook_end_to_end_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executes all code cells of SE374_Colab_Scientific_Research_Runner.ipynb end-to-end against mocked local datasets and models."""
    import json

    import src.colab_pkg.downloader as dl_mod
    import src.colab_pkg.model_loader as ml_mod

    nb_path = Path(__file__).resolve().parent.parent / "SE374_Colab_Scientific_Research_Runner.ipynb"
    assert nb_path.is_file()
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))

    # Verify Cell 2 never calls `pip install -r requirements.txt` on Colab (prevents NumPy 2.x ABI downgrade)
    cell2_src = "".join(nb_data["cells"][2]["source"])
    assert '"requirements.txt"' not in cell2_src
    assert "drive.mount" not in cell2_src

    # Create mock BCSS dataset
    bcss_root = tmp_path / "BCSS-WSSS"
    bcss_test_img = bcss_root / "test" / "img"
    bcss_test_mask = bcss_root / "test" / "mask"
    bcss_test_img.mkdir(parents=True)
    bcss_test_mask.mkdir(parents=True)

    pids = ["TCGA-A1-A0SD", "TCGA-A2-A04P", "TCGA-AQ-A04H", "TCGA-B6-A0I6", "TCGA-C8-A1HJ", "TCGA-D8-A142"]
    for idx, pid in enumerate(pids):
        for j in range(2):
            img_arr = np.full((64, 64, 3), 180, dtype=np.uint8)
            mask_arr = np.zeros((64, 64), dtype=np.uint8)
            mask_arr[:32, :] = idx % 2
            mask_arr[32:, :] = (idx + 1) % 3
            fname = f"{pid}_patch_{j}.png"
            Image.fromarray(img_arr).save(bcss_test_img / fname)
            Image.fromarray(mask_arr).save(bcss_test_mask / fname)

    # Create mock LUAD dataset
    luad_root = tmp_path / "LUAD-HistoSeg"
    luad_test_img = luad_root / "test" / "img"
    luad_test_mask = luad_root / "test" / "mask"
    luad_test_img.mkdir(parents=True)
    luad_test_mask.mkdir(parents=True)
    for j in range(2):
        Image.fromarray(np.full((64, 64, 3), 170, dtype=np.uint8)).save(luad_test_img / f"LUAD_p{j}.png")
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(luad_test_mask / f"LUAD_p{j}.png")

    # Create mock Whole-ROI sample
    roi_img_p = tmp_path / "roi_img.png"
    roi_mask_p = tmp_path / "roi_mask.png"
    Image.fromarray(np.full((128, 128, 3), 175, dtype=np.uint8)).save(roi_img_p)
    Image.fromarray(np.ones((128, 128), dtype=np.uint8)).save(roi_mask_p)

    monkeypatch.setattr(dl_mod, "ensure_bcss_dataset", lambda **kw: (bcss_root, bcss_test_img, bcss_test_mask))
    monkeypatch.setattr(dl_mod, "ensure_checkpoints", lambda **kw: {"stage1": tmp_path / "s1.pth", "stage2": tmp_path / "s2.pth"})
    monkeypatch.setattr(dl_mod, "ensure_luad_dataset", lambda **kw: (luad_root, luad_test_img, luad_test_mask))
    monkeypatch.setattr(dl_mod, "ensure_luad_checkpoints", lambda **kw: {"stage1": tmp_path / "ls1.pth", "stage2": tmp_path / "ls2.pth"})
    monkeypatch.setattr(
        dl_mod,
        "download_bcss_whole_roi_cohort",
        lambda **kw: [{"img": roi_img_p, "mask": roi_mask_p, "patient_id": "TCGA-A1-A0SD", "filename": "roi.png"}],
    )

    class MockLoader:
        def __init__(self, device: str = "cpu") -> None:
            self.device = device

        def load_stage1_netcam(self, ckpt_path: object) -> DummySegmentationModel:
            return DummySegmentationModel()

        def load_stage2_deeplabv3plus(self, ckpt_path: object) -> DummySegmentationModel:
            return DummySegmentationModel()

    monkeypatch.setattr(ml_mod, "ColabModelLoader", MockLoader)
    monkeypatch.setattr(
        ml_mod,
        "load_luad_models",
        lambda **kw: (DummySegmentationModel(), DummySegmentationModel(), torch.device("cpu")),
    )

    ns: dict = {"__name__": "__main__"}
    code_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) == 9

    for idx, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        if idx == 1:
            # Redirect WORK_ROOT to tmp_path and use tiny DRY_RUN_SAMPLES=4 for fast test execution
            src = src.replace(
                'Path("/content") if IN_COLAB else (_repo_for_commit / "se374_scientific_work")',
                f"Path(r'{tmp_path}')",
            )
            src = src.replace('Path("/content" if IN_COLAB else "./se374_scientific_work")', f"Path(r'{tmp_path}')")
            assert "se374_scientific_work" not in src, "WORK_ROOT redirect failed in test"
            src = src.replace("DRY_RUN_SAMPLES = 50", "DRY_RUN_SAMPLES = 4")
        exec(compile(src, f"<notebook_cell_{idx}>", "exec"), ns)

    assert "zip_out" in ns
    assert Path(ns["zip_out"]).is_file()
    assert Path(ns["zip_out"]).stat().st_size > 0

    import zipfile
    with zipfile.ZipFile(ns["zip_out"], "r") as zf:
        namelist = zf.namelist()
        normalized_names = [name.replace("\\", "/") for name in namelist]

        assert any(n.endswith("val_to_test/val_to_test_report.json") for n in normalized_names), (
            f"Missing val_to_test/val_to_test_report.json in results zip. Contents: {normalized_names}"
        )
        assert any(n.endswith("pareto/coarse_to_fine_pareto_curve.png") for n in normalized_names), (
            f"Missing pareto/coarse_to_fine_pareto_curve.png in results zip. Contents: {normalized_names}"
        )
        assert any(n.endswith("pareto/pareto_results.json") for n in normalized_names), (
            f"Missing pareto/pareto_results.json in results zip. Contents: {normalized_names}"
        )
        assert any(n.endswith("biomarkers/patient_biomarker_correlation.csv") for n in normalized_names), (
            f"Missing biomarkers/patient_biomarker_correlation.csv in results zip. Contents: {normalized_names}"
        )
        assert any(n.endswith("biomarkers/biomarker_bland_altman_regression.png") for n in normalized_names), (
            f"Missing biomarkers/biomarker_bland_altman_regression.png in results zip. Contents: {normalized_names}"
        )
        assert any("whole_roi/" in n for n in normalized_names), (
            f"Missing whole_roi/ deliverables in results zip. Contents: {normalized_names}"
        )
        assert any(n.endswith("scientific_benchmark_summary.json") for n in normalized_names), (
            f"Missing scientific_benchmark_summary.json in results zip. Contents: {normalized_names}"
        )

    # Verify patient diversity in SANITY_DRY_RUN mode during end-to-end execution
    val_pids_final = set(dl_mod.extract_patient_id(p.name) for p in ns["val_files"])
    test_pids_final = set(dl_mod.extract_patient_id(p.name) for p in ns["test_files"])
    assert len(val_pids_final) >= 2, f"Expected val patient diversity >= 2, got {len(val_pids_final)}"
    assert len(test_pids_final) >= 2, f"Expected test patient diversity >= 2, got {len(test_pids_final)}"
    assert val_pids_final.isdisjoint(test_pids_final), "Zero patient leakage: val and test must be disjoint"


def test_resolve_split_dirs_ignores_repo_sample_dataset_and_single_patient_fallback(tmp_path: Path) -> None:
    """Verifies resolve_split_dirs ignores cloned medical-wsss/data/sample_dataset under /content and never yields 0 patches."""
    # Simulate /content/medical-wsss/data/sample_dataset/test/img (the cloned GitHub repo inside /content)
    fake_repo_sample = tmp_path / "medical-wsss" / "data" / "sample_dataset" / "test" / "img"
    fake_repo_mask = tmp_path / "medical-wsss" / "data" / "sample_dataset" / "test" / "mask"
    fake_repo_sample.mkdir(parents=True)
    fake_repo_mask.mkdir(parents=True)
    Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(fake_repo_sample / "TCGA-C8-A1HJ_1.png")
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(fake_repo_mask / "TCGA-C8-A1HJ_1.png")

    # Searching from tmp_path (simulating /content) must NOT match medical-wsss/data/sample_dataset
    root_found, img_dir, mask_dir = resolve_split_dirs(tmp_path, split="test")
    assert root_found is None
    assert img_dir is None

    # When real BCSS-WSSS is extracted under tmp_path / "BCSS-WSSS", resolve_split_dirs must find it
    real_bcss_img = tmp_path / "BCSS-WSSS" / "test" / "img"
    real_bcss_mask = tmp_path / "BCSS-WSSS" / "test" / "mask"
    real_bcss_img.mkdir(parents=True)
    real_bcss_mask.mkdir(parents=True)
    Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(real_bcss_img / "TCGA-A1-A0SD_1.png")
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(real_bcss_mask / "TCGA-A1-A0SD_1.png")

    root_found2, img_dir2, mask_dir2 = resolve_split_dirs(tmp_path, split="test")
    assert root_found2 == tmp_path / "BCSS-WSSS"
    assert img_dir2 == real_bcss_img
    assert mask_dir2 == real_bcss_mask


def test_sanity_dry_run_patient_diversity_balanced_sampling() -> None:
    """Verifies that in SANITY_DRY_RUN = True, balanced patient sampling preserves patient diversity.

    Both val and test cohorts must contain distinct patient count >= 2 with zero patient leakage (Invariant 08).
    """
    from collections import defaultdict

    pids = [
        "TCGA-A1-A0SD",
        "TCGA-A2-A04P",
        "TCGA-AQ-A04H",
        "TCGA-B6-A0I6",
        "TCGA-C8-A1HJ",
        "TCGA-D8-A142",
    ]
    patient_map = defaultdict(list)
    for pid in pids:
        for idx in range(10):
            patient_map[pid].append(Path(f"/content/BCSS-WSSS/test/img/{pid}_patch_{idx}.png"))

    unique_pids = sorted(list(patient_map.keys()))
    n_val_pids = max(1, len(unique_pids) // 3)
    val_pids = sorted(list(unique_pids[:n_val_pids]))
    test_pids = sorted(list(unique_pids[n_val_pids:]))

    dry_run_samples = 4
    quota_val = max(1, dry_run_samples // len(val_pids))
    val_files = [p for pid in val_pids for p in patient_map[pid][:quota_val]][:dry_run_samples]
    quota_test = max(1, dry_run_samples // len(test_pids))
    test_files = [p for pid in test_pids for p in patient_map[pid][:quota_test]][:dry_run_samples]

    val_distinct_patients = set(extract_patient_id(p.name) for p in val_files)
    test_distinct_patients = set(extract_patient_id(p.name) for p in test_files)

    assert len(val_distinct_patients) >= 2, f"Expected val patient diversity >= 2, got {len(val_distinct_patients)}"
    assert len(test_distinct_patients) >= 2, f"Expected test patient diversity >= 2, got {len(test_distinct_patients)}"
    assert val_distinct_patients.isdisjoint(test_distinct_patients), "Patient overlap detected between val and test!"


def test_bcss_official_22class_roi_remapping_and_metrics() -> None:
    """Verifies PathologyDataScience/BCSS meta/gtruth_codes.tsv 22-class remapping fixes the off-by-one shift."""
    pipeline = WSILowMagHighMagPipeline(device="cpu")

    # Construct raw BCSS 22-class ROI mask:
    # 0=outside_roi, 1=tumor, 19=angioinvasion, 20=dcis, 2=stroma, 3=lymphocyte, 10=plasma_cells, 4=necrosis, 6=blood_vessel
    raw_bcss_gt = np.zeros((100, 100), dtype=np.uint8)
    raw_bcss_gt[0:20, :] = 1    # tumor -> 0 (TUM)
    raw_bcss_gt[20:30, :] = 20  # dcis -> 0 (TUM)
    raw_bcss_gt[30:55, :] = 2   # stroma -> 1 (STR)
    raw_bcss_gt[55:75, :] = 3   # lymphocyte -> 2 (LYM)
    raw_bcss_gt[75:80, :] = 10  # plasma_cells -> 2 (LYM)
    raw_bcss_gt[80:95, :] = 4   # necrosis -> 3 (NEC)
    raw_bcss_gt[95:98, :] = 6   # blood_vessel -> 4 (outside_roi / ignored)
    raw_bcss_gt[98:100, :] = 0  # outside_roi -> 4 (ignored)

    # Model prediction in WSSS-Tissue 0-indexed convention [0=TUM, 1=STR, 2=LYM, 3=NEC, 4=BACK]
    pred_wsss = np.full((100, 100), 4, dtype=np.uint8)
    pred_wsss[0:30, :] = 0
    pred_wsss[30:55, :] = 1
    pred_wsss[55:80, :] = 2
    pred_wsss[80:95, :] = 3

    metrics = pipeline.evaluate_wsi_with_ground_truth_roi(pred_wsss, raw_bcss_gt)
    assert metrics["mIoU_4cls"] == 100.0
    assert metrics["mDice_4cls"] == 100.0
    assert metrics["Pixel_Accuracy"] == 100.0
    assert metrics["FwIoU"] == 100.0


def test_pareto_router_4cls_oracle_and_concave_gain() -> None:
    """Verifies Pareto router evaluates 4-class mIoU, computes Oracle upper bound, and achieves positive AUC gain."""
    router = CoarseToFineAdaptiveRouter()
    gt = np.zeros((64, 64), dtype=np.int64)
    gt[:32, :] = 2  # LYM
    gt[32:, :] = 3  # NEC

    # Patch 0: Easy patch where coarse is already 100% accurate
    coarse_easy = gt.copy()
    fine_easy = gt.copy()

    # Patch 1: Hard minority-class patch where coarse fails (predicts 1=STR) and fine succeeds (100%)
    coarse_hard = np.ones((64, 64), dtype=np.uint8)
    fine_hard = gt.copy()

    res = router.evaluate_pareto_curve(
        coarse_preds=[coarse_easy, coarse_hard],
        fine_preds=[fine_easy, fine_hard],
        gt_masks=[gt, gt],
        difficulty_scores=[0.10, 0.90],
        budgets=[0.0, 0.5, 1.0],
        num_random_trials=20,
    )

    assert res["adaptive_miou"][1] > res["random_miou"][1]
    assert res["oracle_miou"][1] >= res["adaptive_miou"][1]
    assert res["auc_gain"] > 0.0


def test_biomarker_icc_ccc_and_tumor_positive_subgroup() -> None:
    """Verifies ICC(2,1), Lin's CCC, Bland-Altman LoA, and zero-filtered tumor-positive subgroup."""
    records = [
        {"patient_id": "P_ZERO_1", "pred_vtp": 0.0, "gt_vtp": 0.0, "pred_tsr": 0.0, "gt_tsr": 0.0},
        {"patient_id": "P_ZERO_2", "pred_vtp": 0.0, "gt_vtp": 0.0, "pred_tsr": 0.0, "gt_tsr": 0.0},
        {"patient_id": "P_TUM_1", "pred_vtp": 25.0, "gt_vtp": 27.0, "pred_tsr": 40.0, "gt_tsr": 42.0},
        {"patient_id": "P_TUM_2", "pred_vtp": 55.0, "gt_vtp": 52.0, "pred_tsr": 65.0, "gt_tsr": 62.0},
        {"patient_id": "P_TUM_3", "pred_vtp": 80.0, "gt_vtp": 78.0, "pred_tsr": 85.0, "gt_tsr": 83.0},
    ]
    report = compute_patient_biomarker_correlation(records)
    assert report["patient_count"] == 5
    assert report["tumor_positive_patient_count"] == 3
    assert report["vtp_icc"] > 0.95
    assert report["vtp_ccc"] > 0.95
    assert "tumor_positive_subgroup" in report
    assert report["tumor_positive_subgroup"]["patient_count"] == 3
    assert report["tumor_positive_subgroup"]["tsr_icc"] > 0.95


# -------------------------------------------------------------------------
# Scientific Upgrades: Beer-Lambert, SG-DAP, LMP-PPC, IntraPatch, Ablation
# -------------------------------------------------------------------------

def test_stain_deconv_layer_mathematical_consistency() -> None:
    """Verifies exact Beer-Lambert transpose inversion, ImageNet unnormalization, Fano factor, and Sobel gradient."""
    import torch.nn.functional as F

    from src.models.stain_deconv import StainDeconvLayer

    deconv = StainDeconvLayer()
    dummy_rgb = torch.rand(2, 3, 64, 64, dtype=torch.float32).clamp(0.01, 1.0)
    he_channels, fano_factor, stain_features = deconv(dummy_rgb)

    assert he_channels.shape == (2, 2, 64, 64)
    assert fano_factor.shape == (2, 1, 64, 64)
    assert stain_features.shape == (2, 6, 64, 64)
    assert torch.all(fano_factor >= 0.0)

    # Pure Hematoxylin and pure Eosin optical vectors must unmix cleanly with minimal cross-talk
    h_unit = F.normalize(torch.tensor([0.650, 0.704, 0.286], dtype=torch.float32), dim=0).view(1, 3, 1, 1)
    e_unit = F.normalize(torch.tensor([0.072, 0.990, 0.105], dtype=torch.float32), dim=0).view(1, 3, 1, 1)
    rgb_h = torch.pow(10.0, -0.80 * h_unit).expand(1, 3, 16, 16)
    rgb_e = torch.pow(10.0, -0.70 * e_unit).expand(1, 3, 16, 16)

    stains_h, _, _ = deconv(rgb_h)
    stains_e, _, _ = deconv(rgb_e)
    assert torch.allclose(stains_h[:, 0], torch.full((1, 16, 16), 0.80), atol=1e-3)
    assert torch.allclose(stains_h[:, 1], torch.zeros((1, 16, 16)), atol=1e-3)
    assert torch.allclose(stains_e[:, 1], torch.full((1, 16, 16), 0.70), atol=1e-3)
    assert torch.allclose(stains_e[:, 0], torch.zeros((1, 16, 16)), atol=1e-3)

    # Boundary Sobel gradient response must exceed zero on H&E transition
    mixed = torch.cat([rgb_h[:, :, :, :8], rgb_e[:, :, :, 8:]], dim=-1)
    he_mix, _, _ = deconv(mixed)
    grad = deconv.compute_stain_gradient(he_mix)
    assert float(grad[:, :, :, 7:9].mean()) > float(grad[:, :, :, :4].mean())

    # ImageNet-normalized inputs (with negative values) must unnormalize instead of saturating at min=1e-3
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    stains_h_norm, _, _ = deconv((rgb_h - mean) / std)
    assert torch.allclose(stains_h_norm, stains_h, atol=1e-3)


def test_sg_dap_affinity_refiner_and_ignore_index() -> None:
    """Verifies SG-DAP normalization, mismatched feature alignment, zero-feat probability conservation, and BACK=4 preservation."""
    from src.models.sg_dap import StainGuidedAffinityRefiner

    refiner = StainGuidedAffinityRefiner(
        kernel_size=5,
        alpha_rw=0.85,
        num_iters=3,
        conf_ignore_thresh=0.60,
        margin_ignore_thresh=0.20,
    )

    b, c, h, w = 2, 4, 32, 32
    dummy_rgb = torch.rand(b, 3, h, w, dtype=torch.float32).clamp(0.01, 1.0)
    # Deliberately test spatially mismatched low-res backbone features (8x8 vs 32x32 CAM)
    mismatched_feats = torch.randn(b, 64, 8, 8, dtype=torch.float32)

    # Unnormalized CAM with ambiguous boundary region
    init_cam = torch.zeros(b, c, h, w, dtype=torch.float32)
    init_cam[:, 0, :, :16] = 4.5
    init_cam[:, 1, :, 16:] = 4.5
    init_cam[:, 0, :, 15:17] = 1.0
    init_cam[:, 1, :, 15:17] = 1.0

    # 1. Evaluation mode (return_ignore_mask=False): strictly classes in {0, 1, 2, 3} and sum == 1
    refined_cam, discrete_eval = refiner(dummy_rgb, mismatched_feats, init_cam, return_ignore_mask=False)
    assert refined_cam.shape == (b, c, h, w)
    assert torch.allclose(refined_cam.sum(dim=1), torch.ones(b, h, w), atol=1e-4)
    assert discrete_eval.shape == (b, h, w)
    assert 255 not in discrete_eval
    assert torch.all((discrete_eval >= 0) & (discrete_eval < c))

    # 2. All-zero deep features must conserve probability mass via center self-loop fallback
    zero_feats = torch.zeros(1, 32, h, w, dtype=torch.float32)
    zero_cam, _ = refiner(dummy_rgb[:1], zero_feats, init_cam[:1], return_ignore_mask=False)
    assert torch.allclose(zero_cam.sum(dim=1), torch.ones(1, h, w), atol=1e-4)

    # 3. Training pseudo-mask mode (return_ignore_mask=True): uncertain pixels receive 255
    _, discrete_train = refiner(dummy_rgb, mismatched_feats, init_cam, return_ignore_mask=True)
    assert 255 in discrete_train


def test_sg_dap_stain_only_mode_and_deep_affinity_toggle() -> None:
    """Verifies default use_deep_affinity=False stain-only fast path vs explicit use_deep_affinity=True."""
    from src.models.sg_dap import StainGuidedAffinityRefiner

    torch.manual_seed(17)
    refiner_stain = StainGuidedAffinityRefiner(
        kernel_size=5,
        alpha_rw=0.85,
        num_iters=3,
        conf_ignore_thresh=0.60,
        margin_ignore_thresh=0.20,
    )
    assert refiner_stain.use_deep_affinity is False

    b, c, h, w = 2, 4, 24, 24
    rgb = torch.rand(b, 3, h, w, dtype=torch.float32).clamp(0.05, 0.95)
    # Create spatial stain regions with an ambiguous transition band at columns 10..14
    rgb[0, :, :, :10] = torch.tensor([0.85, 0.35, 0.55]).view(3, 1, 1)
    rgb[0, :, :, 10:14] = torch.tensor([0.60, 0.30, 0.65]).view(3, 1, 1)
    rgb[0, :, :, 14:] = torch.tensor([0.35, 0.25, 0.75]).view(3, 1, 1)

    init_cam = torch.zeros(b, c, h, w, dtype=torch.float32)
    init_cam[0, 0, :, :10] = 4.0
    init_cam[0, 1, :, 14:] = 4.0
    init_cam[0, 0, :, 10:14] = 1.0
    init_cam[0, 1, :, 10:14] = 1.0
    init_cam[1, 2, :, :] = 3.5  # single active class sample
    init_cam[:, 3, :, :] = 2.0  # spurious absent-class activation

    active_labels = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    # 1. Stain-only with deep_feats=None, return_ignore_mask=False
    probs_none, mask_none = refiner_stain(
        rgb, None, init_cam, return_ignore_mask=False, active_labels=active_labels
    )
    assert probs_none.shape == (b, c, h, w)
    assert torch.isfinite(probs_none).all()
    assert torch.allclose(probs_none.sum(dim=1), torch.ones(b, h, w), atol=1e-5)
    assert float(probs_none[0, 2:].abs().max()) == 0.0
    assert float(probs_none[1, [0, 1, 3]].abs().max()) == 0.0
    assert 255 not in mask_none
    assert set(torch.unique(mask_none[0]).tolist()).issubset({0, 1})
    assert set(torch.unique(mask_none[1]).tolist()) == {2}

    # 2. Stain-only with deep_feats=None, return_ignore_mask=True
    probs_ign, mask_ign = refiner_stain(
        rgb, None, init_cam, return_ignore_mask=True, active_labels=active_labels
    )
    assert torch.isfinite(probs_ign).all()
    assert 255 in mask_ign[0]
    assert set(torch.unique(mask_ign[0]).tolist()).issubset({0, 1, 255})
    assert torch.all(mask_ign[1] == 2)  # zero absent-class leak on single-class patch

    # 3. Passing structured deep_feats when use_deep_affinity=False MUST be bypassed identically to None
    deep_struct = torch.zeros(b, 32, 6, 6, dtype=torch.float32)
    deep_struct[:, :16, :, :3] = 3.0
    deep_struct[:, 16:, :, 3:] = 3.0
    probs_with_feats, mask_with_feats = refiner_stain(
        rgb, deep_struct, init_cam, return_ignore_mask=False, active_labels=active_labels
    )
    assert torch.equal(probs_none, probs_with_feats)
    assert torch.equal(mask_none, mask_with_feats)

    # 4. use_deep_affinity=True MUST use structured deep_feats (modulating output vs stain-only)
    refiner_deep = StainGuidedAffinityRefiner(
        kernel_size=5,
        alpha_rw=0.85,
        num_iters=3,
        conf_ignore_thresh=0.60,
        margin_ignore_thresh=0.20,
        use_deep_affinity=True,
    )
    assert refiner_deep.use_deep_affinity is True
    probs_deep, _ = refiner_deep(
        rgb, deep_struct, init_cam, return_ignore_mask=False, active_labels=active_labels
    )
    assert torch.isfinite(probs_deep).all()
    assert torch.allclose(probs_deep.sum(dim=1), torch.ones(b, h, w), atol=1e-5)
    assert not torch.allclose(probs_deep, probs_none, atol=1e-5)

    # 5. use_deep_affinity=True with all-zero deep_feats still conserves mass via center self-loop
    zero_feats = torch.zeros(1, 32, h, w, dtype=torch.float32)
    zero_cam, _ = refiner_deep(rgb[:1], zero_feats, init_cam[:1], return_ignore_mask=False)
    assert torch.allclose(zero_cam.sum(dim=1), torch.ones(1, h, w), atol=1e-4)


def test_multi_prototype_ppc_loss_and_ema_update() -> None:
    """Verifies MultiPrototypePPCLoss EMA update in train() vs strict buffer immutability in eval()."""
    from src.losses_metrics.losses import MultiPrototypePPCLoss

    loss_fn = MultiPrototypePPCLoss(
        num_classes=4,
        num_prototypes=3,
        feat_dim=32,
        max_samples=256,
    )

    b, d, h, w = 2, 32, 16, 16
    proj_feats = torch.randn(b, d, h, w, dtype=torch.float32, requires_grad=True)
    cam_probs = torch.softmax(torch.randn(b, 4, h, w) * 3.0, dim=1)
    img_labels = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])

    # In eval() mode, prototypes MUST remain frozen
    loss_fn.eval()
    protos_before_eval = loss_fn.prototypes.clone()
    _ = loss_fn(proj_feats, cam_probs, img_labels)
    assert torch.equal(protos_before_eval, loss_fn.prototypes)

    # In train() mode, backward works and EMA updates prototypes
    loss_fn.train()
    initial_protos = loss_fn.prototypes.clone()
    loss = loss_fn(proj_feats, cam_probs, img_labels)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert proj_feats.grad is not None
    assert not torch.equal(initial_protos, loss_fn.prototypes)


def test_intra_patch_dual_branch_attention_identity_init() -> None:
    """Verifies M9 identity initialization, zero norm-buffer drift, step-1 gamma grad, and step-2 full grad flow."""
    from src.models.intra_patch_attention import IntraPatchDualBranchAttention

    attn_module = IntraPatchDualBranchAttention(
        in_channels=32,
        macro_size=(8, 8),
        num_heads=2,
    )
    attn_module.train()

    dummy_feats = torch.randn(2, 32, 16, 16, dtype=torch.float32, requires_grad=True)
    enhanced, spatial_gate = attn_module(dummy_feats)

    # Exact identity at gamma == 0
    assert torch.allclose(enhanced, dummy_feats, atol=1e-6)
    assert spatial_gate.shape == (2, 1, 16, 16)
    assert torch.all((spatial_gate >= 0.0) & (spatial_gate <= 1.0))

    # Verify zero BatchNorm running_mean buffers inside gate_conv
    assert not any("running_mean" in k for k in attn_module.state_dict().keys())

    # Step 1 backward gives non-zero gradient to gamma; step 2 (gamma != 0) flows into query/key/value/gate
    opt = torch.optim.SGD(attn_module.parameters(), lr=0.1)
    opt.zero_grad()
    loss = (enhanced * 2.0).sum()
    loss.backward()
    assert dummy_feats.grad is not None
    assert attn_module.gamma.grad is not None and float(attn_module.gamma.grad.abs().sum()) > 0.0
    opt.step()

    opt.zero_grad()
    feats2 = torch.randn(2, 32, 16, 16, dtype=torch.float32)
    enh2, _ = attn_module(feats2)
    (enh2 ** 2).mean().backward()
    assert attn_module.query_conv.weight.grad is not None
    assert float(attn_module.query_conv.weight.grad.abs().sum()) > 0.0


def test_comparative_ablation_evaluator_dry_run(tmp_path: Path) -> None:
    """Verifies ComparativeAblationEvaluator across all modes, M9 exact identity, composable pipeline, and Wilcoxon stats."""
    from src.clinical.comparative_ablation_evaluator import ComparativeAblationEvaluator

    report_file = tmp_path / "test_ablation_report.json"
    evaluator = ComparativeAblationEvaluator(output_report_path=str(report_file))

    # 1. Verify exact M9 residual identity when gamma == 0
    rgb = torch.rand(1, 3, 32, 32).clamp(0.01, 1.0)
    feats = torch.randn(1, 64, 32, 32)
    init_cam = torch.softmax(torch.randn(1, 4, 32, 32), dim=1)
    p1, _ = evaluator.run_inference_patch(rgb, init_cam, feats, method="m1")
    p9_only, _ = evaluator.run_inference_patch(rgb, init_cam, feats, method="m9_only")
    p8, _ = evaluator.run_inference_patch(rgb, init_cam, feats, method="m8")
    p9, _ = evaluator.run_inference_patch(rgb, init_cam, feats, method="m9")
    assert float(torch.max(torch.abs(p9_only - p1))) < 1e-6
    assert float(torch.max(torch.abs(p9 - p8))) < 1e-6

    # 2. Verify arbitrary composable operator ordering
    p_comp, m_comp = evaluator.compose_pipeline(rgb, init_cam, feats, steps=["od_prior", "sg_dap", "mpppc"])
    assert torch.allclose(p_comp.sum(dim=1), torch.ones(1, 32, 32), atol=1e-4)
    assert set(torch.unique(m_comp).tolist()).issubset({0, 1, 2, 3})

    # 3. Verify all standalone & cumulative modes in dry-run suite
    report = evaluator.evaluate_synthetic_or_dataset_suite(num_patches=10, patch_size=32)
    assert report_file.is_file()
    assert "comparative_benchmarks" in report
    benchmarks = report["comparative_benchmarks"]

    expected_modes = ["m1", "m6", "m7_only", "m8_only", "m9_only", "od_only", "m7", "m8", "m9", "m_combined"]
    for method_key in expected_modes:
        assert method_key in benchmarks
        res = benchmarks[method_key]
        assert "four_class_miou" in res
        assert "five_class_miou" in res
        assert "pixel_accuracy" in res
        assert "boundary_f1" in res
        assert "latency_ms_per_patch" in res
        assert "wilcoxon_p_vs_m1" in res
        assert "wilcoxon_p_vs_m6" in res
        assert "patient_wilcoxon_p_vs_m1" in res
        per_class = res["per_class_iou"]
        assert "tumor" in per_class
        assert "stroma" in per_class
        assert "inflammatory" in per_class
        assert "necrosis" in per_class


def test_evaluator_scale_alpha_sweep_and_ablation_edge_cases(tmp_path: Path) -> None:
    """Regression tests for 100x delta_miou fix, m6_cam_fusion_alpha sweep, cgm no-op, and fn_to_bg consistency."""
    from src.clinical.comparative_ablation_evaluator import ComparativeAblationEvaluator
    from src.clinical.exhaustive_evaluator import ExhaustiveBenchmarkEvaluator, SegmentationMetricAccumulator
    from src.colab_pkg.evaluator import run_multiscale_baseline_comparison
    from src.colab_pkg.model_loader import (
        MultiMethodWSSSEngineV2 as ColabEngineV2,
    )
    from src.models.wsss_inference_methods import (
        MultiMethodWSSSEngineV2 as SrcEngineV2,
    )

    # 1. delta_miou in run_multiscale_baseline_comparison must be in percentage points [0, 100], never 100x inflated
    img_dir = tmp_path / "ms_img"
    mask_dir = tmp_path / "ms_mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    img_p = img_dir / "TCGA-A1-A0SD_p0.png"
    mask_p = mask_dir / "TCGA-A1-A0SD_p0.png"
    Image.fromarray(np.full((32, 32, 3), 160, dtype=np.uint8)).save(img_p)
    gt_arr = np.zeros((32, 32), dtype=np.uint8)
    gt_arr[16:, :] = 1
    Image.fromarray(gt_arr).save(mask_p)

    class _PerfectM6Engine:
        def predict_m6(self, model, tensor, raw_rgb=None, model_stage1=None):
            return gt_arr.copy()

    ms_res = run_multiscale_baseline_comparison(
        engine=_PerfectM6Engine(),
        model_stage2=DummySegmentationModel(),
        test_files=[img_p],
        test_mask_dir=mask_dir,
        device="cpu",
    )
    assert ms_res["delta_miou"] == pytest.approx(75.0, abs=1e-4)

    # 2. m6_cam_fusion_alpha must affect predict_all_methods M6 output in both src/models and src/colab_pkg
    class _Stage2Class0(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, h, w = x.shape
            out = torch.zeros((b, 4, h, w), device=x.device)
            out[:, 0] = 2.0
            out[:, 1] = 1.0
            return out

    class _Stage1Class1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            b, _, h, w = x.shape
            cam = torch.zeros((b, 4, max(1, h // 4), max(1, w // 4)), device=x.device)
            cam[:, 0, 0, 0] = 5.0
            cam[:, 1, :, :] = 6.0
            cam[:, 1, 0, 0] = 0.0
            y_cls = torch.tensor([[0.9, 0.9, 0.01, 0.01]], device=x.device).expand(b, 4)
            return cam, y_cls

    x_in = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    raw_rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
    s2_mock = _Stage2Class0().eval()
    s1_mock = _Stage1Class1().eval()
    for EngineCls in (SrcEngineV2, ColabEngineV2):
        eng = EngineCls(model_stage1=s1_mock, include_v2=True)
        assert hasattr(eng, "m6_cam_fusion_alpha")
        eng.m6_cam_fusion_alpha = 0.0
        p_a = eng.predict_all_methods(s2_mock, x_in, raw_rgb=raw_rgb, model_stage1=s1_mock, include_v2=True)[
            "M6_MultiScale_Pyramid_D4_Ensemble"
        ]
        eng.m6_cam_fusion_alpha = 0.99
        p_b = eng.predict_all_methods(s2_mock, x_in, raw_rgb=raw_rgb, model_stage1=s1_mock, include_v2=True)[
            "M6_MultiScale_Pyramid_D4_Ensemble"
        ]
        assert np.all(p_a[16:, :] == 0) and np.all(p_b[16:, :] == 1), (
            f"{EngineCls.__module__} ignored m6_cam_fusion_alpha in predict_all_methods"
        )

    # 3. compose_pipeline("cgm") with cgm_gate=None must be a clean no-op, not ValueError
    ev = ComparativeAblationEvaluator(device=torch.device("cpu"))
    cam = torch.softmax(torch.randn(1, 4, 32, 32), dim=1)
    feat = torch.randn(1, 64, 16, 16)
    pr_noop, _ = ev.compose_pipeline(x_in[:, :, :32, :32], cam, feat, steps=("cgm",), precomputed_d4=cam, cgm_gate=None)
    assert torch.allclose(pr_noop, cam, atol=1e-5)

    # 4. _accumulate_confusion must NOT fall back to p4 when allow_p4_fallback=False and p5 predicts 100% background
    cm4 = np.zeros((4, 4), dtype=np.int64)
    cm5 = np.zeros((5, 5), dtype=np.int64)
    fn_bg = np.zeros(4, dtype=np.int64)
    p4_only = np.zeros((8, 8), dtype=np.uint8)
    p5_all_bg = np.full((8, 8), 4, dtype=np.uint8)
    gt_fg = np.zeros((8, 8), dtype=np.int64)
    miou_bg = ComparativeAblationEvaluator._accumulate_confusion(
        cm4, cm5, fn_bg, p4_only, p5_all_bg, gt_fg, allow_p4_fallback=False
    )
    assert miou_bg == 0.0
    assert int(cm4.sum()) == 0
    assert int(fn_bg[0]) == 64

    # 5. SegmentationMetricAccumulator & ColabSegmentationEvaluator include fn_to_bg_4cls in PA_4cls denominator
    acc = SegmentationMetricAccumulator(num_classes=5)
    pred_half_bg = np.array([[0, 0], [4, 4]], dtype=np.int64)
    gt_all_tum = np.zeros((2, 2), dtype=np.int64)
    acc.update(pred_half_bg, gt_all_tum)
    m_acc = acc.compute_metrics()
    assert m_acc["Pixel_Accuracy_4cls"] == pytest.approx(50.0, abs=1e-4)
    assert acc.patch_miou_4cls[0] == pytest.approx(0.5, abs=1e-4)

    ceval = ColabSegmentationEvaluator(num_classes=5)
    ceval.update(pred_half_bg, gt_all_tum, patient_id="TCGA-A1-A0SD")
    m_ceval = ceval.compute_metrics()
    assert m_ceval["Accuracy"] == pytest.approx(50.0, abs=1e-4)
    assert ceval.patch_miou_4cls[0] == pytest.approx(50.0, abs=1e-4)

    # 6. ExhaustiveBenchmarkEvaluator._compute_patient_level_tests scales scores to [0, 100]
    ex_ev = ExhaustiveBenchmarkEvaluator(dataset_dir=Path("data/sample_dataset"), output_dir=tmp_path, device="cpu")
    ex_ev.eval_medsam.patch_patient_ids = ["P1", "P1", "P2", "P2", "P3", "P3"]
    scores_a = np.array([0.50, 0.50, 0.60, 0.60, 0.70, 0.70])
    scores_m = np.array([0.60, 0.60, 0.70, 0.70, 0.80, 0.80])
    pt_res = ex_ev._compute_patient_level_tests(scores_a, scores_a, scores_m)
    assert pt_res["proposed_vs_author"]["mean_diff"] == pytest.approx(10.0, abs=1e-4)


def test_colab_evaluator_patient_level_fn_to_bg_and_zero_cm4_guard() -> None:
    """Regression test: patient-level mIoU/bootstrap and compute_metrics must honor fn_to_bg_4cls."""
    gt = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.int64)
    pred_half_bg = np.array([[0, 4, 1, 4], [2, 4, 3, 4]], dtype=np.int64)

    # 1. Exact reproduction: 50% of each ROI class predicted as background -> 50.0 across all 3 metrics
    ev1 = ColabSegmentationEvaluator(num_classes=5)
    ev1.update(pred_half_bg, gt, patient_id="P1")
    m1 = ev1.compute_metrics()
    assert m1["mIoU"] == pytest.approx(50.0, abs=1e-4)
    assert m1["patch_miou_mean"] == pytest.approx(50.0, abs=1e-4)
    assert m1["patient_miou_mean"] == pytest.approx(50.0, abs=1e-4)
    assert m1["patient_count"] == 1

    # 2. 100%-background-predicted patient must NOT be dropped and bootstrap CI must include fn_to_bg
    ev2 = ColabSegmentationEvaluator(num_classes=5)
    ev2.update(pred_half_bg, gt, patient_id="P1")
    ev2.update(np.full((2, 4), 4, dtype=np.int64), gt, patient_id="P_BG")
    m2 = ev2.compute_metrics()
    assert m2["patient_count"] == 2
    assert m2["patient_miou_mean"] == pytest.approx(25.0, abs=1e-4)
    assert m2["patient_bootstrap_miou_ci95"][1] <= 50.0 + 1e-4

    # 3. Single 100%-background-predicted patient alone -> patient_count == 1, patient_miou_mean == 0.0
    ev_bg = ColabSegmentationEvaluator(num_classes=5)
    ev_bg.update(np.full((2, 4), 4, dtype=np.int64), gt, patient_id="P_ONLY_BG")
    m_bg = ev_bg.compute_metrics()
    assert m_bg["patient_count"] == 1
    assert m_bg["patient_miou_mean"] == pytest.approx(0.0, abs=1e-4)

    # 4. cm4.sum() == 0 with fn_to_bg_4cls.sum() > 0 must NOT fall back to 5-class cm[:4, :]
    ev3 = ColabSegmentationEvaluator(num_classes=5)
    ev3.update(
        np.zeros((2, 4), dtype=np.int64),
        gt,
        pred_raw_4cls=np.full((2, 4), 4, dtype=np.int64),
        patient_id="P_RAW_BG",
    )
    m3 = ev3.compute_metrics()
    assert m3["mIoU"] == pytest.approx(0.0, abs=1e-4)
    assert m3["Accuracy"] == pytest.approx(0.0, abs=1e-4)
    assert m3["patient_count"] == 1
    assert m3["patient_miou_mean"] == pytest.approx(0.0, abs=1e-4)
    ev3.reset()
    assert len(ev3.patient_fn_bg_4cls) == 0


def test_surrogate_prompt_encoder_empty_and_label_4_mapping() -> None:
    """Regression test: SurrogatePromptEncoder handles N==0/B==0 without stack crash and maps label 4 distinctly."""
    from src.models.sam_wsss_adapter import SurrogatePromptEncoder

    torch.manual_seed(0)
    enc = SurrogatePromptEncoder(embed_dim=256)

    # 1. points=None -> (1, 0, 256)
    out_none, dense_none = enc(points=None)
    assert out_none.shape == (1, 0, 256)
    assert dense_none is None

    # 2. N == 0 -> (B, 0, 256) without RuntimeError from torch.stack
    coords_n0 = torch.zeros((2, 0, 2), dtype=torch.float32)
    labels_n0 = torch.zeros((2, 0), dtype=torch.int64)
    out_n0, dense_n0 = enc(points=(coords_n0, labels_n0))
    assert out_n0.shape == (2, 0, 256)
    assert dense_n0 is None

    # 3. B == 0 -> (0, N, 256) without RuntimeError from torch.stack
    coords_b0 = torch.zeros((0, 3, 2), dtype=torch.float32)
    labels_b0 = torch.zeros((0, 3), dtype=torch.int64)
    out_b0, dense_b0 = enc(points=(coords_b0, labels_b0))
    assert out_b0.shape == (0, 3, 256)
    assert dense_b0 is None

    # 4. Label 4 must map onto point_embeddings distinctly from pad label -1, neg 0, and pos 1
    coords_1 = torch.tensor([[[12.0, 24.0]]], dtype=torch.float32)
    out_neg1, _ = enc(points=(coords_1, torch.tensor([[-1]], dtype=torch.int64)))
    out_0, _ = enc(points=(coords_1, torch.tensor([[0]], dtype=torch.int64)))
    out_1, _ = enc(points=(coords_1, torch.tensor([[1]], dtype=torch.int64)))
    out_4, _ = enc(points=(coords_1, torch.tensor([[4]], dtype=torch.int64)))

    assert not torch.allclose(out_4, out_neg1, atol=1e-6)
    assert not torch.allclose(out_4, out_0, atol=1e-6)
    assert not torch.allclose(out_4, out_1, atol=1e-6)


def test_downloader_extract_patient_id_parity_with_patient_split() -> None:
    """Regression test: colab_pkg.downloader.extract_patient_id matches datasets.patient_split.extract_patient_id."""
    from src.datasets.patient_split import extract_patient_id as canonical_extract_patient_id

    cases = [
        ("TCGA-A2-A0T2_1024_2048_[1 0 0 1].png", "TCGA-A2-A0T2"),
        ("tcga-d8-a27f_patch_01.png", "TCGA-D8-A27F"),
        ("C:/data/test/TCGA-XX-XXXX+001+[1 0 0 0].png", "TCGA-XX-XXXX"),
        ("C:/dir_with_underscore/GDPH_1234_patch_01.png", "GDPH"),
        ("LUAD+001+[1 0 0 0].png", "LUAD"),
        ("simple.png", "simple"),
        ("nested/sub_dir/caseA+tile_2.png", "caseA"),
    ]
    for raw_path, expected in cases:
        got_dl = extract_patient_id(raw_path)
        got_canon = canonical_extract_patient_id(raw_path)
        assert got_dl == expected, f"downloader extract_patient_id({raw_path!r})={got_dl!r} != {expected!r}"
        assert got_dl == got_canon, (
            f"Parity mismatch on {raw_path!r}: downloader={got_dl!r} vs patient_split={got_canon!r}"
        )


def test_fp16_amp_losses_no_nan_on_extreme_logits() -> None:
    """Regression test: custom losses, PDA, and Stage-1 head stay finite under autocast & float16."""
    from src.losses_metrics.losses import (
        MultiLayerPseudoSupervisionLoss,
        MultiPrototypePPCLoss,
        UncertaintyAwareBoundaryLoss,
        UncertaintyRegularizationLoss,
    )
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
    from src.training.train_proposed_sam import CompositeWSSSLoss
    from src.training.train_stage1 import AsymmetricLoss, AttentionModule, Stage1Trainer

    torch.manual_seed(7)

    # 1. AsymmetricLoss under float16 extreme logits (-50, 0, +50) and exact FP32 parity on logits < -16.1
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    logits_f16 = torch.tensor(
        [[-50.0, 50.0, 0.0, -35.0], [50.0, -50.0, 25.0, 0.0]],
        dtype=torch.float16,
        requires_grad=True,
    )
    targets_f16 = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]], dtype=torch.float16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss_asl = asl(logits_f16, targets_f16)
    assert loss_asl.dtype == torch.float32
    assert torch.isfinite(loss_asl), f"AsymmetricLoss returned non-finite: {loss_asl}"
    loss_asl.backward()
    assert logits_f16.grad is not None and torch.isfinite(logits_f16.grad).all()

    logits_neg = torch.tensor([[-25.0, 2.0, -18.0, 0.5]], dtype=torch.float32)
    targets_neg = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    probs_ref = torch.sigmoid(logits_neg)
    expected_ref = -(
        targets_neg * torch.log(probs_ref.clamp(min=1e-8))
        + (1.0 - targets_neg)
        * probs_ref.pow(4.0)
        * torch.log((1.0 - (probs_ref - 0.05).clamp(min=0.0)).clamp(min=1e-8))
    ).mean()
    assert torch.allclose(asl(logits_neg, targets_neg), expected_ref, atol=1e-6)

    # 2. AttentionModule (PDA) non-uniform extreme activations inside autocast (zero >65504 overflow)
    attn = AttentionModule()
    x_f16 = torch.full((2, 16, 4, 4), 100.0, dtype=torch.float16)
    x_f16[:, :, 0, 0] = 50.0
    x_f16.requires_grad_(True)
    w_f16 = torch.full((4, 16, 1, 1), 10.0, dtype=torch.float16, requires_grad=True)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out_attn = attn(x_f16, w_f16, gamma=0.75)
    assert out_attn.dtype == torch.float32
    assert torch.isfinite(out_attn).all()

    # 3. WSSSTissueStage1NetCAM & Stage1Trainer.forward_with_pda inside autocast return FP32 finite outputs
    s1_net = WSSSTissueStage1NetCAM(n_class=4).eval()
    trainer = Stage1Trainer(
        model=s1_net,
        optimizer=torch.optim.AdamW(s1_net.parameters(), lr=1e-4),
        criterion=asl,
        enable_amp=False,
    )
    img_bf = torch.rand((1, 3, 64, 64), dtype=torch.float32)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        y_fwd = s1_net(img_bf)
        cam_fc, y_fc = s1_net.forward_cam(img_bf)
        logits_pda, probs_pda = trainer.forward_with_pda(img_bf, enable_pda=True)
    assert y_fwd.dtype == torch.float32 and torch.isfinite(y_fwd).all()
    assert cam_fc.dtype == torch.float32 and torch.isfinite(cam_fc).all()
    assert y_fc.dtype == torch.float32 and torch.isfinite(y_fc).all()
    assert logits_pda.dtype == torch.float32 and torch.isfinite(logits_pda).all()
    assert probs_pda.dtype == torch.float32 and torch.isfinite(probs_pda).all()

    # 4. MLPS + UncertaintyRegularization + MP-PPC (extreme cos diff) + UABL + CompositeWSSSLoss inside autocast
    seg_logits = (torch.randn(2, 5, 16, 16) * 30.0).to(torch.float16).detach().requires_grad_(True)
    pm = torch.randint(0, 5, (2, 16, 16), dtype=torch.long)
    pm[0, 0, :4] = 255
    mlps = MultiLayerPseudoSupervisionLoss()
    unc = UncertaintyRegularizationLoss()
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        l_mlps = mlps(seg_logits, {"bn7": pm, "b5_2": pm, "b4_5": pm})
        l_unc = unc(seg_logits, pm)
    assert l_mlps.dtype == torch.float32 and torch.isfinite(l_mlps)
    assert l_unc.dtype == torch.float32 and torch.isfinite(l_unc)

    ppc = MultiPrototypePPCLoss(num_classes=4, num_prototypes=2, feat_dim=8, temp=0.05, conf_thresh=0.1)
    with torch.no_grad():
        ppc.prototypes.zero_()
        ppc.prototypes[0, :, 0] = -1.0  # Opposite class 0 feature
        ppc.prototypes[1, :, 0] = 1.0   # Aligned with class 1 negative -> pos_sim - sim_max <= -20 at temp=0.05
    feats_ppc = torch.zeros((2, 8, 8, 8), dtype=torch.float32)
    feats_ppc[:, 0, :, :] = 1.0
    feats_ppc.requires_grad_(True)
    cam_ppc = torch.zeros((2, 4, 8, 8), dtype=torch.float32)
    cam_ppc[:, 0, :, :] = 0.99
    cam_ppc[:, 1, :, :] = 0.01
    img_lbl_f16 = torch.ones((2, 4), dtype=torch.float16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        l_ppc = ppc(feats_ppc, cam_ppc, img_lbl_f16)
    assert l_ppc.dtype == torch.float32 and torch.isfinite(l_ppc)
    l_ppc.backward()
    assert feats_ppc.grad is not None and torch.isfinite(feats_ppc.grad).all()

    uabl = UncertaintyAwareBoundaryLoss()
    probs_f16 = torch.softmax(seg_logits.detach(), dim=1).to(torch.float16).requires_grad_(True)
    grad_f16 = torch.rand((2, 1, 16, 16), dtype=torch.float16)
    unc_map_f16 = torch.rand((2, 1, 16, 16), dtype=torch.float16)
    comp = CompositeWSSSLoss()
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        l_uabl = uabl(probs_f16, grad_f16, unc_map_f16)
        comp_out = comp(seg_logits, pm, iou_predictions=torch.rand((2, 5), dtype=torch.float16))
    assert l_uabl.dtype == torch.float32 and torch.isfinite(l_uabl)
    assert comp_out["total_loss"].dtype == torch.float32 and torch.isfinite(comp_out["total_loss"])







