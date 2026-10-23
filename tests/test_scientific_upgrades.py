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

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from src.colab_pkg.downloader import (
    ensure_luad_checkpoints,
    ensure_luad_dataset,
    extract_patient_id,
    resolve_split_dirs,
)
from src.colab_pkg.evaluator import (
    ColabSegmentationEvaluator,
    LUAD_CLASS_CODES,
    run_biomarker_correlation_study,
    run_cross_dataset_luad_benchmark,
    run_multimag_and_pareto_benchmark,
    run_val_to_test_scientific_protocol,
)
from src.colab_pkg.model_loader import (
    DeepLabV3Plus,
    WSSSTissueStage1NetCAM,
    load_luad_models,
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
    WSICandidatePatch,
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
            src = src.replace('Path("/content" if IN_COLAB else "./se374_scientific_work")', f"Path(r'{tmp_path}')")
            src = src.replace("DRY_RUN_SAMPLES = 50", "DRY_RUN_SAMPLES = 4")
        exec(compile(src, f"<notebook_cell_{idx}>", "exec"), ns)

    assert "zip_out" in ns
    assert Path(ns["zip_out"]).is_file()
    assert Path(ns["zip_out"]).stat().st_size > 0

