"""Unit & integration tests for Gate-7 Clinical Biomarkers, GPU Efficiency, LUAD Suite, Notebooks & Docker."""

import ast
import csv
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import nbformat
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.stage2_v2.gate7_clinical_bench import (
    _SHORT_LABELS,
    LUAD_TO_BCSS_LUT,
    LUADBCSSAlignedDataset,
    benchmark_gate7_gpu_efficiency,
    compute_bland_altman_and_pearson,
    compute_cohens_kappa,
    compute_gate7_clinical_biomarkers,
    extract_luad_case_id,
    remap_luad_mask_to_bcss,
    run_gate7_clinical_and_luad_benchmark,
)
from src.stage2_v2.visual_audit import BCSS_PNG_PALETTE_PREFIX, GATE7_ARM_ORDER

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAD_DASHES = ("\u2014", "\u2013", "\u2012", "\u2015", "\u2212")


def _save_indexed_mask(mask_u8: np.ndarray, path: Path) -> None:
    im = Image.fromarray(mask_u8.astype(np.uint8), mode="P")
    pal: List[int] = []
    for rgb in BCSS_PNG_PALETTE_PREFIX:
        pal.extend(rgb)
    pal.extend([0] * (768 - len(pal)))
    im.putpalette(pal)
    im.save(path)


def _assert_monochrome_bw_png(path: Path, expected_dpi: int = 300) -> None:
    assert path.is_file() and path.stat().st_size > 2000
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
        dpi = im.info.get("dpi")
    if dpi is not None:
        assert round(dpi[0]) == expected_dpi and round(dpi[1]) == expected_dpi
    # Strict monochrome B&W: zero chroma saturation across all pixels + pure white corner background
    max_chroma = int(np.max(np.abs(rgb[:, :, 0] - rgb[:, :, 1])))
    max_chroma = max(max_chroma, int(np.max(np.abs(rgb[:, :, 1] - rgb[:, :, 2]))))
    assert max_chroma <= 2, f"{path.name} is not strict monochrome B&W (max_chroma={max_chroma})"
    assert int(rgb[0, 0, 0]) >= 250 and int(rgb[0, 0, 1]) >= 250 and int(rgb[0, 0, 2]) >= 250


class _TinyDualHeadStub(nn.Module):
    input_norm = "imagenet"
    track = "track_a_clean"
    tcga_pretrained = False

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            self.conv.weight[0, 0, 0, 0] = 2.0
            self.conv.weight[1, 1, 0, 0] = 2.0
            self.conv.weight[2, 2, 0, 0] = 2.0
            self.conv.bias[3] = -0.2

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg = self.conv(x)
        cls = seg.mean(dim=(2, 3))
        return {"seg_logits": seg, "cls_logits": cls}


class _TinyStage1Stub(nn.Module):
    def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        cams = torch.ones((b, 4, max(1, h // 4), max(1, w // 4)), device=x.device)
        probs = torch.full((b, 4), 0.9, device=x.device)
        return cams, probs


class _TinyBCSSDataset(Dataset):
    def __init__(self, items: List[Tuple[np.ndarray, np.ndarray, str]]) -> None:
        self.items = items
        self.files = [Path(fn) for _, _, fn in items]
        self.mask_dir = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        rgb, gt, fn = self.items[idx]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = (rgb.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
        return torch.from_numpy(arr), torch.from_numpy(gt.astype(np.int64)), fn


def test_gate7_clinical_bench_line_budget_and_ascii_dashes() -> None:
    mod_p = PROJECT_ROOT / "src" / "stage2_v2" / "gate7_clinical_bench.py"
    text = mod_p.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 400, f"gate7_clinical_bench.py exceeds 400 lines: {len(text.splitlines())}"
    for bad in BAD_DASHES:
        assert bad not in text

    plots_p = PROJECT_ROOT / "src" / "stage2_v2" / "clinical_bench_plots.py"
    plots_txt = plots_p.read_text(encoding="utf-8")
    assert len(plots_txt.splitlines()) <= 350, f"clinical_bench_plots.py exceeds 350 lines: {len(plots_txt.splitlines())}"
    for bad in BAD_DASHES:
        assert bad not in plots_txt


def test_bland_altman_pearson_and_cohens_kappa_edge_cases() -> None:
    # Perfect identity
    st_id = compute_bland_altman_and_pearson([10.0, 40.0, 70.0, 90.0], [10.0, 40.0, 70.0, 90.0])
    assert st_id["pearson_r"] == pytest.approx(1.0, abs=1e-4)
    assert st_id["mean_bias"] == pytest.approx(0.0, abs=1e-4)
    assert st_id["loa_lower"] == pytest.approx(0.0, abs=1e-4)
    assert st_id["loa_upper"] == pytest.approx(0.0, abs=1e-4)
    assert st_id["mae"] == pytest.approx(0.0, abs=1e-4)

    # Constant +5% bias
    st_off = compute_bland_altman_and_pearson([15.0, 45.0, 75.0, 95.0], [10.0, 40.0, 70.0, 90.0])
    assert st_off["pearson_r"] == pytest.approx(1.0, abs=1e-4)
    assert st_off["mean_bias"] == pytest.approx(5.0, abs=1e-4)
    assert st_off["mae"] == pytest.approx(5.0, abs=1e-4)

    # Variable disagreement -> non-zero LoA and RMSE >= MAE
    st_var = compute_bland_altman_and_pearson([12.0, 35.0, 74.0, 80.0], [10.0, 40.0, 70.0, 90.0])
    assert st_var["loa_lower"] < st_var["mean_bias"] < st_var["loa_upper"]
    assert st_var["rmse"] >= st_var["mae"] > 0.0

    # Degenerate empty & constant arrays (zero NaN)
    st_empty = compute_bland_altman_and_pearson([], [])
    assert st_empty["n"] == 0 and st_empty["pearson_p"] == 1.0
    st_const = compute_bland_altman_and_pearson([50.0, 50.0, 50.0], [50.0, 50.0, 50.0])
    assert st_const["pearson_r"] == 0.0 and st_const["pearson_p"] == 1.0

    # Cohen's kappa: perfect, chance, inverse, degenerate single-class, empty
    assert compute_cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert compute_cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert compute_cohens_kappa([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)
    assert compute_cohens_kappa([1, 1, 1], [0, 0, 0]) == pytest.approx(0.0)
    assert compute_cohens_kappa([], []) == pytest.approx(0.0)


def test_luad_bcss_label_remapping_and_nested_dataset(tmp_path: Path) -> None:
    raw = np.array([[0, 1, 2], [3, 4, 255]], dtype=np.int64)
    mapped = remap_luad_mask_to_bcss(raw)
    assert mapped.tolist() == [[0, 3, 2], [1, 4, 255]]
    assert LUAD_TO_BCSS_LUT.tolist() == [0, 3, 2, 1, 4]

    # Defensive nested rglob discovery + mode='P' preservation
    img_nested = tmp_path / "luad" / "test" / "img" / "nested_sub"
    msk_nested = tmp_path / "luad" / "test" / "mask" / "nested_sub"
    img_nested.mkdir(parents=True)
    msk_nested.mkdir(parents=True)
    fn = "GDPH_001+patch_01.png"
    Image.fromarray(np.full((64, 64, 3), (180, 120, 150), dtype=np.uint8)).save(img_nested / fn)
    m_raw = np.zeros((64, 64), dtype=np.uint8)
    m_raw[:32, :32] = 1  # LUAD NEC -> BCSS 3
    m_raw[:32, 32:] = 3  # LUAD TAS -> BCSS 1
    m_raw[32:, :32] = 2  # LUAD LYM -> BCSS 2
    m_raw[32:, 32:] = 4  # BACK -> 4
    _save_indexed_mask(m_raw, msk_nested / fn)

    ds = LUADBCSSAlignedDataset(tmp_path / "luad" / "test" / "img", tmp_path / "luad" / "test" / "mask", target_size=(64, 64))
    assert len(ds) == 1
    img_t, gt_t, name = ds[0]
    assert name == fn and img_t.shape == (3, 64, 64) and gt_t.shape == (64, 64)
    assert int(gt_t[0, 0]) == 3 and int(gt_t[0, 63]) == 1 and int(gt_t[63, 0]) == 2 and int(gt_t[63, 63]) == 4

    # LUAD case/patient ID extraction across TCGA, GDPH coord (<case>-<x>-<y>), and GDPH_<id> formats
    assert extract_luad_case_id("TCGA-55-7724-01Z_10_20.png") == "TCGA-55-7724"
    assert extract_luad_case_id("242129-14432-46368.png") == "242129"
    assert extract_luad_case_id("242129-14656-46368.png") == "242129"
    assert extract_luad_case_id("GDPH_001+patch_01.png") == "GDPH_001"
    assert extract_luad_case_id("GDPH_002+patch_01.png") == "GDPH_002"

    with pytest.raises(FileNotFoundError, match="Zero paired PNG"):
        LUADBCSSAlignedDataset(tmp_path / "empty_img", tmp_path / "empty_msk")


def test_clinical_biomarkers_cap_survival_and_monochrome_plots(tmp_path: Path) -> None:
    # Four real TCGA-BRCA 23-patient IDs spanning both Deceased (vital_status=1) and Living (0)
    pids = ["TCGA-E2-A1LK", "TCGA-EW-A1P8", "TCGA-E2-A14N", "TCGA-EW-A1P1"]
    gt_tum_ratios = [0.25, 0.35, 0.70, 0.80]  # First 2 Stroma-Rich (High Risk, TSR<=50%), last 2 Low Risk
    raw_rgbs, gt_list, pr_arm5_list, pr_ens4_list = [], [], [], []
    for r_tum in gt_tum_ratios:
        rgb = np.full((64, 64, 3), (180, 110, 150), dtype=np.uint8)
        rgb[56:, :] = 252  # Pure white optical glass region (> 512 px -> GT-free Pure-Glass Guard sets to 4)
        gt = np.ones((64, 64), dtype=np.uint8)  # STR
        split_row = int(56 * r_tum)
        gt[:split_row, :] = 0  # TUM
        gt[56:, :] = 4  # BACK glass
        # Prediction predicts foreground even on glass (must be stripped by GT-free Pure-Glass Guard!)
        pr4 = gt.copy()
        pr4[56:, :] = 0
        pr5 = pr4.copy()
        pr5[max(0, split_row - 2) : split_row + 2, :] = 1
        raw_rgbs.append(rgb)
        gt_list.append(gt)
        pr_arm5_list.append(pr5)
        pr_ens4_list.append(pr4)

    res = compute_gate7_clinical_biomarkers(
        patient_ids=pids,
        raw_rgbs=raw_rgbs,
        gt_masks=np.stack(gt_list, axis=0),
        model_preds={"arm5_cgm": np.stack(pr_arm5_list, axis=0), "ens4_cgm": np.stack(pr_ens4_list, axis=0)},
        output_dir=tmp_path,
        primary_model="ens4_cgm",
        dpi=300,
    )
    assert res["n_patients"] == 4 and res["n_patches"] == 4
    ens_st = res["models"]["ens4_cgm"]
    assert ens_st["vtp"]["pearson_r"] > 0.99 and ens_st["tsr"]["pearson_r"] > 0.99
    assert ens_st["cap_concordance"]["accuracy"] == pytest.approx(1.0)
    assert ens_st["cap_concordance"]["cohens_kappa"] == pytest.approx(1.0)
    assert 0.0 <= ens_st["km_logrank_p"] <= 1.0
    assert res["kaplan_meier"]["gt"]["n_high_risk"] == 2 and res["kaplan_meier"]["gt"]["n_low_risk"] == 2

    # Unmeasurable patient (TUM=0, STR=0; pure LYM) must get cap_code=-1 (Indeterminate), never High Risk (1)
    unm_dir = tmp_path / "unmeasurable"
    rgb_lym = np.full((64, 64, 3), (120, 100, 180), dtype=np.uint8)
    gt_lym = np.full((64, 64), 2, dtype=np.uint8)
    res_unm = compute_gate7_clinical_biomarkers(
        patient_ids=["TCGA-E2-A1LK"],
        raw_rgbs=[rgb_lym],
        gt_masks=gt_lym[None, ...],
        model_preds={"ens4_cgm": gt_lym[None, ...]},
        output_dir=unm_dir,
        primary_model="ens4_cgm",
        dpi=150,
    )
    assert res_unm["patients"][0]["gt_cap_code"] == -1
    assert res_unm["patients"][0]["ens4_cgm_cap_code"] == -1
    assert res_unm["models"]["ens4_cgm"]["cap_concordance"]["n_measurable"] == 0

    # CSV & JSON artifacts + strict monochrome 300-DPI B&W figures
    with open(tmp_path / "patient_clinical_risk_summary.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert (tmp_path / "gate7_clinical_biomarkers.json").is_file()
    _assert_monochrome_bw_png(tmp_path / "bland_altman_tsr_vtp_bw.png", expected_dpi=300)
    _assert_monochrome_bw_png(tmp_path / "km_survival_tsr_bw.png", expected_dpi=300)


def test_end_to_end_gate7_clinical_gpu_luad_benchmark_and_zip(tmp_path: Path) -> None:
    bcss_items = []
    for idx, pid in enumerate(("TCGA-E2-A1LK", "TCGA-EW-A1P8", "TCGA-E2-A14N", "TCGA-EW-A1P1")):
        rgb = np.full((32, 32, 3), (120, 120, 120), dtype=np.uint8)
        gt = np.ones((32, 32), dtype=np.uint8)
        if idx < 2:
            rgb[:8, :] = (230, 60, 60)
            gt[:8, :] = 0
        else:
            rgb[:24, :] = (230, 60, 60)
            gt[:24, :] = 0
        bcss_items.append((rgb, gt, f"{pid}_0_0.png"))

    luad_img_d, luad_msk_d = tmp_path / "luad_img", tmp_path / "luad_msk"
    luad_img_d.mkdir()
    luad_msk_d.mkdir()
    for i in range(2):
        fn = f"CASE0{i}+patch_{i}.png"
        rgb_l = np.full((32, 32, 3), (60, 220, 60), dtype=np.uint8)
        rgb_l[:16, :] = (220, 60, 60)
        Image.fromarray(rgb_l).save(luad_img_d / fn)
        m_l = np.full((32, 32), 3, dtype=np.uint8)  # LUAD TAS -> BCSS STR (1)
        m_l[:16, :] = 0  # LUAD TE -> BCSS TUM (0)
        _save_indexed_mask(m_l, luad_msk_d / fn)

    arm_specs = [{"name": nm, "model": _TinyDualHeadStub()} for nm in GATE7_ARM_ORDER]
    bcss_loader = DataLoader(_TinyBCSSDataset(bcss_items), batch_size=2, shuffle=False)
    luad_loader = DataLoader(LUADBCSSAlignedDataset(luad_img_d, luad_msk_d, target_size=(32, 32)), batch_size=2, shuffle=False)
    out_dir = tmp_path / "bench_out"
    zip_out = tmp_path / "SE374_Stage2_Training_Results.zip"

    combined = run_gate7_clinical_and_luad_benchmark(
        arm_specs=arm_specs,
        bcss_test_loader=bcss_loader,
        luad_test_loader=luad_loader,
        stage1_cls_ckpt=None,
        author_stage1_ckpt=None,
        author_stage2_ckpt=None,
        output_dir=out_dir,
        zip_path=zip_out,
        device=torch.device("cpu"),
        amp_dtype=None,
        selected_ttas=("raw_1x", "raw_1x", "raw_1x", "raw_1x"),
        n_warmup=1,
        n_iters=2,
        dpi=300,
        stage1_model_override=_TinyStage1Stub(),
        author_stage1_model_override=_TinyStage1Stub(),
        author_model_override=_TinyDualHeadStub(),
    )
    assert "author_stage2" in combined["gpu_efficiency"]["models"]
    assert "arm5_dinov2_large_r2" in combined["gpu_efficiency"]["models"]
    assert "primary_4arm_ensemble" in combined["gpu_efficiency"]["models"]
    assert combined["gpu_efficiency"]["models"]["primary_4arm_ensemble"]["measurement_mode"] == "resident_multi_arm_forward"
    # Every arm must receive its REAL computed BCSS mIoU (zero hardcoded 71.0 fallback!)
    for arm_nm in GATE7_ARM_ORDER:
        assert arm_nm in combined["bcss_pooled_miou_pct"]
        assert combined["gpu_efficiency"]["models"][arm_nm]["reference_miou_pct"] == combined["bcss_pooled_miou_pct"][arm_nm]
    assert _SHORT_LABELS["primary_4arm_ensemble"] == "4-Arm Ens (no-CGM)"
    assert combined["gpu_efficiency"]["models"]["primary_4arm_ensemble"]["reference_miou_pct"] == combined["bcss_pooled_miou_pct"]["ens4_no_cgm"]
    # Primary 4-arm ensemble prefers ens4_no_cgm over ens4_cgm when both exist, and falls back to ens4_cgm when absent
    both_ref = benchmark_gate7_gpu_efficiency(
        arm_specs=[{"name": "arm8_deeplab_r101_r2", "model": _TinyDualHeadStub()}],
        author_stage2_ckpt=None,
        sample_imgs=torch.zeros((2, 3, 32, 32)),
        device=torch.device("cpu"),
        amp_dtype=None,
        output_dir=tmp_path / "both_ref_gpu",
        reference_mious={"arm8_deeplab_r101_r2": 70.96, "ens4_no_cgm": 72.80, "ens4_cgm": 72.81},
        n_warmup=1,
        n_iters=1,
        dpi=150,
    )
    assert both_ref["models"]["primary_4arm_ensemble"]["reference_miou_pct"] == pytest.approx(72.80)
    fallback_ref = benchmark_gate7_gpu_efficiency(
        arm_specs=[{"name": "arm8_deeplab_r101_r2", "model": _TinyDualHeadStub()}],
        author_stage2_ckpt=None,
        sample_imgs=torch.zeros((2, 3, 32, 32)),
        device=torch.device("cpu"),
        amp_dtype=None,
        output_dir=tmp_path / "fallback_ref_gpu",
        reference_mious={"arm8_deeplab_r101_r2": 70.96, "ens4_cgm": 72.81},
        n_warmup=1,
        n_iters=1,
        dpi=150,
    )
    assert fallback_ref["models"]["primary_4arm_ensemble"]["reference_miou_pct"] == pytest.approx(72.81)
    zero_ref = benchmark_gate7_gpu_efficiency(
        arm_specs=[{"name": "arm8_deeplab_r101_r2", "model": _TinyDualHeadStub()}],
        author_stage2_ckpt=None,
        sample_imgs=torch.zeros((2, 3, 32, 32)),
        device=torch.device("cpu"),
        amp_dtype=None,
        output_dir=tmp_path / "zero_ref_gpu",
        reference_mious={"ens4_no_cgm": 0.0, "ens4_cgm": 72.81},
        n_warmup=1,
        n_iters=1,
        dpi=150,
    )
    assert zero_ref["models"]["primary_4arm_ensemble"]["reference_miou_pct"] == pytest.approx(0.0)
    none_val_ref = benchmark_gate7_gpu_efficiency(
        arm_specs=[{"name": "arm8_deeplab_r101_r2", "model": _TinyDualHeadStub()}],
        author_stage2_ckpt=None,
        sample_imgs=torch.zeros((2, 3, 32, 32)),
        device=torch.device("cpu"),
        amp_dtype=None,
        output_dir=tmp_path / "none_val_ref_gpu",
        reference_mious={"author_step0": None, "ens4_no_cgm": None, "ens4_cgm": 72.81},
        n_warmup=1,
        n_iters=1,
        author_model_override=_TinyDualHeadStub(),
        dpi=150,
    )
    assert none_val_ref["models"]["author_stage2"]["reference_miou_pct"] is None
    assert none_val_ref["models"]["primary_4arm_ensemble"]["reference_miou_pct"] == pytest.approx(72.81)
    # Zero fabricated fallback when reference_mious=None
    no_ref = benchmark_gate7_gpu_efficiency(
        arm_specs=[{"name": "arm8_deeplab_r101_r2", "model": _TinyDualHeadStub()}],
        author_stage2_ckpt=None,
        sample_imgs=torch.zeros((2, 3, 32, 32)),
        device=torch.device("cpu"),
        amp_dtype=None,
        output_dir=tmp_path / "no_ref_gpu",
        reference_mious=None,
        n_warmup=1,
        n_iters=1,
        author_model_override=_TinyDualHeadStub(),
        dpi=150,
    )
    assert no_ref["models"]["author_stage2"]["reference_miou_pct"] is None
    assert no_ref["models"]["arm8_deeplab_r101_r2"]["reference_miou_pct"] is None
    assert no_ref["models"]["primary_4arm_ensemble"]["reference_miou_pct"] is None
    assert combined["luad_cross_organ"]["n_patches"] == 2
    assert combined["luad_cross_organ"]["n_patients"] == 2
    assert combined["luad_cross_organ"]["models"]["ens4_cgm"]["mIoU"] >= 0.0
    _assert_monochrome_bw_png(out_dir / "gpu_pareto_efficiency_bw.png", expected_dpi=300)

    with zipfile.ZipFile(zip_out, "r") as zf:
        names = set(zf.namelist())
    for req in (
        "STAGE2_TRAINING_METRICS.json",
        "GATE7_CLINICAL_BENCH_METRICS.json",
        "patient_clinical_risk_summary.csv",
        "gate7_clinical_biomarkers.json",
        "bland_altman_tsr_vtp_bw.png",
        "km_survival_tsr_bw.png",
        "gate7_gpu_efficiency.json",
        "gpu_pareto_efficiency_bw.png",
        "gate7_luad_cross_organ.json",
        "stage2_training_loss.png",
    ):
        assert req in names, f"Missing {req} in ZIP"


def test_04b_and_02d_notebooks_nbformat_ast_local_bundle_zero_drive_mount() -> None:
    for rel, must_contain, forbid in (
        (
            "notebooks/04b_Stage2_Gate7_Clinical_LUAD_Bench_G4.ipynb",
            (
                "/content/local_repo.bundle",
                "STAGE2_GIT_REF",
                "khoinguyen59/medical-wsss.git",
                "run_gate7_clinical_and_luad_benchmark",
                "GATE7_CLINICAL_BENCH_METRICS.json",
                "STAGE2_TRAINING_METRICS.json",
                "SE374_Stage2_Training_Results.zip",
            ),
            ("drive.mount", "/content/drive"),
        ),
        (
            "notebooks/02d_Stage2_Visual_Overlay_Audit_G4.ipynb",
            (
                "/content/local_repo.bundle",
                "STAGE2_GIT_REF",
                "khoinguyen59/medical-wsss.git",
                "run_stage2_v2_visual_audit",
            ),
            ("drive.mount", "/content/drive"),
        ),
    ):
        nb_path = PROJECT_ROOT / rel
        raw = nb_path.read_text(encoding="utf-8")
        for bad in BAD_DASHES:
            assert bad not in raw, f"Non-ASCII dash in {rel}"
        for token in must_contain:
            assert token in raw, f"Missing {token!r} in {rel}"
        for bad_tok in forbid:
            assert bad_tok not in raw, f"Forbidden {bad_tok!r} in {rel}"
        nb_obj = nbformat.reads(raw, as_version=4)
        nbformat.validate(nb_obj)
        assert nb_obj["nbformat"] == 4 and nb_obj["nbformat_minor"] == 5
        code_cells = [c for c in nb_obj["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 6
        for idx, cell in enumerate(code_cells):
            ast.parse(cell["source"], filename=f"{rel}:cell_{idx}")


def test_dockerfile_and_docker_compose_production_contracts() -> None:
    df_p = PROJECT_ROOT / "Dockerfile"
    dc_p = PROJECT_ROOT / "docker-compose.yml"
    df_txt = df_p.read_text(encoding="utf-8")
    dc_txt = dc_p.read_text(encoding="utf-8")
    for text in (df_txt, dc_txt):
        for bad in BAD_DASHES:
            assert bad not in text
    assert "FROM python:" in df_txt
    assert "EXPOSE 8000" in df_txt
    assert "HEALTHCHECK" in df_txt and "/health" in df_txt
    assert "src.visualization.api_server:app" in df_txt
    assert "USER appuser" in df_txt
    assert "8000:8000" in dc_txt
    assert "./checkpoints:/app/checkpoints" in dc_txt
    assert "./data:/app/data" in dc_txt
    assert "/health" in dc_txt


def test_pareto_axis_labels_and_gate7_monochrome_figures_regressions(tmp_path: Path, monkeypatch) -> None:
    """Lock in Defect 2A Pareto axis labels and Defects 5A-5D + dynamic SHA256 count in monochrome figures."""
    from src.stage2_v2 import gate7_clinical_bench as bench_mod
    from src.visualization import draw_gate7_monochrome_figures as fig_mod

    captured: Dict[str, object] = {}
    orig_save = bench_mod.va._atomic_save_figure

    def _capture_save(fig, save_path, dpi=300):
        ax = fig.axes[0]
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        bboxes = [t.get_bbox_patch().get_window_extent(rend) for t in ax.texts]
        ax_box = ax.get_window_extent(rend)
        leg = ax.get_legend()
        leg_box = leg.get_window_extent(rend) if leg is not None else None
        captured["xlabel"] = ax.get_xlabel()
        captured["ylabel"] = ax.get_ylabel()
        captured["n_labels"] = len(bboxes)
        captured["any_overlap"] = any(bboxes[i].overlaps(bboxes[j]) for i in range(len(bboxes)) for j in range(i + 1, len(bboxes)))
        captured["any_clipped"] = any(b.x0 < ax_box.x0 or b.x1 > ax_box.x1 or b.y0 < ax_box.y0 or b.y1 > ax_box.y1 for b in bboxes)
        captured["any_leg_overlap"] = bool(leg_box is not None and any(b.overlaps(leg_box) for b in bboxes))
        return orig_save(fig, save_path, dpi=dpi)

    monkeypatch.setattr(bench_mod.va, "_atomic_save_figure", _capture_save)
    pareto_png = bench_mod.render_gpu_pareto_bw(
        {
            "device_name": "TestGPU",
            "models": {
                "author_stage2": {"latency_ms_per_patch": 3.2, "throughput_fps": 312.5, "peak_vram_mb": 420.0, "reference_miou_pct": 70.43},
                "arm8_deeplab_r101_r2": {"latency_ms_per_patch": 4.1, "throughput_fps": 243.9, "peak_vram_mb": 510.0, "reference_miou_pct": 70.96},
                "arm6_segformer_b5_r2": {"latency_ms_per_patch": 8.5, "throughput_fps": 117.6, "peak_vram_mb": 980.0, "reference_miou_pct": 72.04},
                "arm7_convnext_large_r2": {"latency_ms_per_patch": 9.8, "throughput_fps": 102.0, "peak_vram_mb": 1450.0, "reference_miou_pct": 71.80},
                "arm5_dinov2_large_r2": {"latency_ms_per_patch": 18.4, "throughput_fps": 54.3, "peak_vram_mb": 2650.0, "reference_miou_pct": 72.53},
                "primary_4arm_ensemble": {"latency_ms_per_patch": 41.2, "throughput_fps": 24.3, "peak_vram_mb": 5500.0, "reference_miou_pct": 72.80},
            },
        },
        tmp_path / "pareto_check.png",
        dpi=300,
    )
    assert captured["xlabel"] == "1x Forward Latency per Patch (ms, lower is faster)"
    assert captured["ylabel"] == "BCSS TEST TTA 4-Class mIoU (%)"
    assert captured["n_labels"] == 6 and not captured["any_overlap"] and not captured["any_clipped"] and not captured["any_leg_overlap"]
    _assert_monochrome_bw_png(pareto_png, expected_dpi=300)

    # Real G4 Blackwell tight bottom-left cluster (Arm-8 1.33ms/70.47% vs Author 1.44ms/70.43%) must also have zero label overlap
    pareto_g4_png = bench_mod.render_gpu_pareto_bw(
        {
            "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "models": {
                "author_stage2": {"latency_ms_per_patch": 1.436, "throughput_fps": 696.54, "peak_vram_mb": 351.86, "reference_miou_pct": 70.43},
                "arm8_deeplab_r101_r2": {"latency_ms_per_patch": 1.330, "throughput_fps": 751.79, "peak_vram_mb": 440.24, "reference_miou_pct": 70.47},
                "arm6_segformer_b5_r2": {"latency_ms_per_patch": 4.425, "throughput_fps": 225.96, "peak_vram_mb": 915.09, "reference_miou_pct": 72.04},
                "arm7_convnext_large_r2": {"latency_ms_per_patch": 1.488, "throughput_fps": 672.00, "peak_vram_mb": 1360.52, "reference_miou_pct": 71.80},
                "arm5_dinov2_large_r2": {"latency_ms_per_patch": 2.988, "throughput_fps": 334.66, "peak_vram_mb": 1872.21, "reference_miou_pct": 72.53},
                "primary_4arm_ensemble": {"latency_ms_per_patch": 9.064, "throughput_fps": 110.33, "peak_vram_mb": 3202.10, "reference_miou_pct": 72.80},
            },
        },
        tmp_path / "pareto_g4_check.png",
        dpi=300,
    )
    assert captured["n_labels"] == 6 and not captured["any_overlap"] and not captured["any_clipped"] and not captured["any_leg_overlap"]
    _assert_monochrome_bw_png(pareto_g4_png, expected_dpi=300)

    # Adversarial Pareto stress cases: 3-point tight cluster, 2 near-zero-ms models, and near-right-edge penultimate model
    for case_nm, case_models in (
        (
            "tight3",
            {
                "arm8_deeplab_r101_r2": {"latency_ms_per_patch": 1.33, "throughput_fps": 751.8, "peak_vram_mb": 440.2, "reference_miou_pct": 70.47},
                "author_stage2": {"latency_ms_per_patch": 1.44, "throughput_fps": 696.5, "peak_vram_mb": 351.9, "reference_miou_pct": 70.43},
                "arm7_convnext_large_r2": {"latency_ms_per_patch": 1.49, "throughput_fps": 672.0, "peak_vram_mb": 1360.5, "reference_miou_pct": 70.55},
                "arm5_dinov2_large_r2": {"latency_ms_per_patch": 2.99, "throughput_fps": 334.7, "peak_vram_mb": 1872.2, "reference_miou_pct": 72.53},
                "arm6_segformer_b5_r2": {"latency_ms_per_patch": 4.42, "throughput_fps": 226.0, "peak_vram_mb": 915.1, "reference_miou_pct": 72.04},
                "primary_4arm_ensemble": {"latency_ms_per_patch": 9.06, "throughput_fps": 110.3, "peak_vram_mb": 3202.1, "reference_miou_pct": 72.80},
            },
        ),
        (
            "near_zero",
            {
                "arm8_deeplab_r101_r2": {"latency_ms_per_patch": 0.01, "throughput_fps": 100000.0, "peak_vram_mb": 440.0, "reference_miou_pct": 70.47},
                "author_stage2": {"latency_ms_per_patch": 0.02, "throughput_fps": 50000.0, "peak_vram_mb": 351.0, "reference_miou_pct": 70.43},
            },
        ),
        (
            "right_penult",
            {
                "author_stage2": {"latency_ms_per_patch": 1.0, "throughput_fps": 1000.0, "peak_vram_mb": 351.0, "reference_miou_pct": 70.43},
                "arm8_deeplab_r101_r2": {"latency_ms_per_patch": 1.2, "throughput_fps": 833.0, "peak_vram_mb": 440.0, "reference_miou_pct": 70.96},
                "arm5_dinov2_large_r2": {"latency_ms_per_patch": 8.9, "throughput_fps": 112.0, "peak_vram_mb": 1872.0, "reference_miou_pct": 72.53},
                "primary_4arm_ensemble": {"latency_ms_per_patch": 9.0, "throughput_fps": 111.0, "peak_vram_mb": 3202.0, "reference_miou_pct": 72.80},
            },
        ),
    ):
        bench_mod.render_gpu_pareto_bw({"device_name": case_nm, "models": case_models}, tmp_path / f"pareto_{case_nm}.png", dpi=150)
        assert captured["n_labels"] == len(case_models)
        assert not captured["any_overlap"] and not captured["any_clipped"] and not captured["any_leg_overlap"], f"{case_nm}: {captured}"

    ev_sentinel = PROJECT_ROOT / "checkpoints" / "vps_evidence" / "se374_g7_ensemble_tracka" / "artifacts" / "STAGE2_TRAINING_METRICS.json"
    if not ev_sentinel.is_file():
        pytest.skip("Gitignored checkpoints/vps_evidence/ not present on clean CI checkout.")

    sha_file = PROJECT_ROOT / "checkpoints" / "vps_evidence" / "SHA256SUMS.txt"
    expected_sha_count = sum(1 for ln in sha_file.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
    ev = fig_mod.load_evidence(PROJECT_ROOT)
    assert expected_sha_count >= 123 and ev["sha256_count"] == expected_sha_count
    assert len((PROJECT_ROOT / "src" / "visualization" / "draw_gate7_monochrome_figures.py").read_text(encoding="utf-8").splitlines()) <= 400
    sens = fig_mod._compute_patient_sensitivity(ev["patient_json"], ev.get("true_gt_counts"))
    assert sens["n_json_records"] == 12
    assert sens["asym_nec_clr"] == 7 and sens["asym_nec_tot"] == 7
    assert sens["asym_lym_fp"] == 1 and sens["asym_lym_tot"] == 1 and sens["asym_lym_fp_ids"] == ["EW-A3U0"]
    assert len(sens["gt0_info"]) == 8
    d8_idx = sens["pids"].index("TCGA-D8-A27H")
    ew_idx = sens["pids"].index("TCGA-EW-A3U0")
    assert sens["gt0_info"][d8_idx] == ("NEC", 425, 0)
    assert sens["arm5_gt0_fp"] == {ew_idx}
    assert d8_idx not in sens["arm5_gt0_fp"]

    fig_mod.configure_monochrome_style()
    fig3_png, fig3_svg = fig_mod.draw_fig3_patient_gt0_sensitivity(ev, tmp_path)
    fig4_png, fig4_svg = fig_mod.draw_fig4_pipeline_architecture(ev, tmp_path)
    _assert_monochrome_bw_png(fig3_png, expected_dpi=300)
    _assert_monochrome_bw_png(fig4_png, expected_dpi=300)
    svg3_txt = fig3_svg.read_text(encoding="utf-8")
    svg4_txt = fig4_svg.read_text(encoding="utf-8")
    for bad in BAD_DASHES:
        assert bad not in svg3_txt and bad not in svg4_txt
    for req3 in (
        "Raw vs Asym-Free (-8 Asym GT=0)",
        "Excl. 8 Asym GT=0 (min union=0)",
        "Excl. 8 asym GT=0 cells (min union=0)",
        "Asym GT=0 NEC cell cleared by 4-DoF CGM",
    ):
        assert req3 in svg3_txt, f"Missing {req3!r} in regenerated fig3 SVG"
    for forb3 in ("Clears 7/7 GT=0 NEC", "1/1 GT=0 LYM left"):
        assert forb3 not in svg3_txt, f"Stale D10 token {forb3!r} found in fig3 SVG"
    if ev.get("true_gt_counts") is not None:
        assert sens["true_gt0_count"] == 11
        assert sens["true_nec_absent"] == 8 and sens["true_lym_absent"] == 3
        assert sens["arm5_true_nec_cleared"] == 7 and sens["arm5_true_lym_fp"] == 3
        assert sens["true_nec_fp_ids"] == ["E2-A573"]
        assert sens["true_gt_stats"][0] == pytest.approx(0.9383, abs=1e-3)
        for req3_true in (
            "Clears 7/8 NEC-Absent (7/7 Asym) to 0 px",
            "Excl. All 11 True GT=0 (on-disk)",
            "Clears 7/8 true NEC-absent (7/7 asym; E2-A573 FP)",
            "3/3 LYM-absent left (asym: EW-A3U0)",
        ):
            assert req3_true in svg3_txt, f"Missing {req3_true!r} in regenerated fig3 SVG"
    for required in (
        "0% TCGA-BRCA Pretraining Overlap",
        "4. GATE-7 TRACK A 4-ARM (R2 EMA-ONLY, 224/336px)",
        "Primary 4-Arm (no-CGM): 72.80% | p_Holm=0.0005",
        "Secondary Simplex (+CGM): 72.90% TEST (exploratory)",
        "WSI Multi-Tile Stitch, VTP/TSR %, CAP PDF",
        f"{expected_sha_count}/{expected_sha_count} SHA256 Artifacts (BAD = 0)",
    ):
        assert required in svg4_txt, f"Missing {required!r} in regenerated fig4 SVG"
    for forbidden in ("p=0.0004", "sTILs", "0% TCGA Pretraining |"):
        assert forbidden not in svg4_txt, f"Stale token {forbidden!r} found in fig4 SVG"

    # Safe fallback when SHA256SUMS.txt or raw test masks are absent, partial, or corrupted RGB
    ev_fallback = {**ev, "sha256_count": None, "true_gt_counts": None}
    fig3_fb_png, fig3_fb_svg = fig_mod.draw_fig3_patient_gt0_sensitivity(ev_fallback, tmp_path / "fallback")
    _assert_monochrome_bw_png(fig3_fb_png, expected_dpi=300)
    assert "Clears 7/7 Asym GT=0 NEC to 0 px" in fig3_fb_svg.read_text(encoding="utf-8")
    _, fig4_fb_svg = fig_mod.draw_fig4_pipeline_architecture(ev_fallback, tmp_path / "fallback")
    assert "SHA256 Artifacts Verified (BAD = 0)" in fig4_fb_svg.read_text(encoding="utf-8")
    partial_root = tmp_path / "partial_masks"
    partial_dir = partial_root / "data" / "BCSS-WSSS" / "test" / "mask"
    partial_dir.mkdir(parents=True)
    for pid in sens["pids"]:
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(partial_dir / f"{pid}-DX1_0+0.png")
    assert fig_mod._load_true_test_gt_counts(partial_root, sens["pids"], expected_total=4986) is None
    rgb_root = tmp_path / "rgb_masks"
    rgb_dir = rgb_root / "data" / "BCSS-WSSS" / "test" / "mask"
    rgb_dir.mkdir(parents=True)
    for pid in sens["pids"]:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(rgb_dir / f"{pid}-DX1_0+0.png")
    assert fig_mod._load_true_test_gt_counts(rgb_root, sens["pids"], expected_total=len(sens["pids"])) is None


def test_phase1a_gate7_clinical_bench_upgrades(tmp_path: Path, monkeypatch) -> None:
    """Verify Lin's CCC, Mesker/UNITED attribution, ROI-bounded+Cand1+VTP_RCB+HotspotTSR, calibration, 2-panel Pareto, LUAD 2x zoom & provenance."""
    from src.stage2_v2 import clinical_bench_plots as plots_mod
    from src.stage2_v2 import gate7_clinical_bench as bench_mod

    # 1. Lin's CCC edge cases: identity=1.0, constant shift < 1.0 even though r=1.0, empty=0.0
    assert plots_mod.compute_lins_ccc([10.0, 40.0, 70.0], [10.0, 40.0, 70.0]) == pytest.approx(1.0, abs=1e-4)
    ccc_shift = plots_mod.compute_lins_ccc([20.0, 50.0, 80.0], [10.0, 40.0, 70.0])
    assert 0.0 < ccc_shift < 0.99
    assert plots_mod.compute_lins_ccc([], []) == pytest.approx(0.0)
    assert "Mesker" in plots_mod.GUIDELINE_ATTRIBUTION and "UNITED" in plots_mod.GUIDELINE_ATTRIBUTION

    # 2. PixelCalibrationAccumulator: perfect predictions vs uniform predictions
    cal_perf = plots_mod.PixelCalibrationAccumulator()
    gt_c = np.array([[[0, 1], [2, 3]], [[0, 4], [1, 4]]], dtype=np.uint8)
    pr_perf = np.full((2, 4, 2, 2), 1e-4, dtype=np.float32)
    for b in range(2):
        for r in range(2):
            for c in range(2):
                if gt_c[b, r, c] < 4:
                    pr_perf[b, gt_c[b, r, c], r, c] = 0.9997
    cal_perf.update(pr_perf, gt_c)
    s_perf = cal_perf.compute()
    assert s_perf["n_valid_pixels"] == 6 and s_perf["ece"] < 0.01 and s_perf["brier"] < 0.01 and s_perf["nll"] < 0.01

    cal_unif = plots_mod.PixelCalibrationAccumulator()
    cal_unif.update(np.full((2, 4, 2, 2), 0.25, dtype=np.float32), gt_c)
    s_unif = cal_unif.compute()
    assert s_unif["brier"] == pytest.approx(0.75, abs=1e-4)

    # 3. 2-panel GPU Pareto zero-collision across BOTH panels (raw_1x & deployed_tta) + strict B&W
    panel_checks: List[Dict[str, object]] = []
    orig_save = bench_mod.va._atomic_save_figure

    def _capture_2panel(fig, save_path, dpi=300):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for ax in fig.axes:
            bboxes = [t.get_bbox_patch().get_window_extent(rend) for t in ax.texts]
            ax_box = ax.get_window_extent(rend)
            leg = ax.get_legend()
            leg_box = leg.get_window_extent(rend) if leg is not None else None
            panel_checks.append(
                {
                    "n_labels": len(bboxes),
                    "any_overlap": any(bboxes[i].overlaps(bboxes[j]) for i in range(len(bboxes)) for j in range(i + 1, len(bboxes))),
                    "any_clipped": any(b.x0 < ax_box.x0 or b.x1 > ax_box.x1 or b.y0 < ax_box.y0 or b.y1 > ax_box.y1 for b in bboxes),
                    "any_leg_overlap": bool(leg_box is not None and any(b.overlaps(leg_box) for b in bboxes)),
                }
            )
        return orig_save(fig, save_path, dpi=dpi)

    monkeypatch.setattr(bench_mod.va, "_atomic_save_figure", _capture_2panel)
    two_panel_data = {
        "device_name": "NVIDIA RTX PRO 6000 Blackwell",
        "amp_dtype": "torch.bfloat16",
        "raw_1x": {
            "author_stage2": {"latency_ms_per_patch": 0.90, "throughput_fps": 1111.0, "peak_vram_mb": 350.0, "reference_miou_pct": 69.35},
            "arm8_deeplab_r101_r2": {"latency_ms_per_patch": 1.05, "throughput_fps": 952.0, "peak_vram_mb": 440.0, "reference_miou_pct": 70.10},
            "arm6_segformer_b5_r2": {"latency_ms_per_patch": 3.80, "throughput_fps": 263.0, "peak_vram_mb": 915.0, "reference_miou_pct": 71.50},
            "arm7_convnext_large_r2": {"latency_ms_per_patch": 1.35, "throughput_fps": 740.0, "peak_vram_mb": 1360.0, "reference_miou_pct": 71.20},
            "arm5_dinov2_large_r2": {"latency_ms_per_patch": 2.85, "throughput_fps": 350.0, "peak_vram_mb": 1872.0, "reference_miou_pct": 72.10},
            "primary_4arm_ensemble": {"latency_ms_per_patch": 8.95, "throughput_fps": 111.7, "peak_vram_mb": 3200.0, "reference_miou_pct": 72.45},
        },
        "deployed_tta": {
            "author_stage2": {"latency_ms_per_patch": 6.40, "throughput_fps": 156.0, "peak_vram_mb": 520.0, "reference_miou_pct": 70.43},
            "arm8_deeplab_r101_r2": {"latency_ms_per_patch": 7.10, "throughput_fps": 140.0, "peak_vram_mb": 590.0, "reference_miou_pct": 70.47},
            "arm6_segformer_b5_r2": {"latency_ms_per_patch": 31.20, "throughput_fps": 32.0, "peak_vram_mb": 1100.0, "reference_miou_pct": 72.04},
            "arm7_convnext_large_r2": {"latency_ms_per_patch": 9.40, "throughput_fps": 106.0, "peak_vram_mb": 1500.0, "reference_miou_pct": 71.80},
            "arm5_dinov2_large_r2": {"latency_ms_per_patch": 18.50, "throughput_fps": 54.0, "peak_vram_mb": 2100.0, "reference_miou_pct": 72.80},
            "primary_4arm_ensemble": {"latency_ms_per_patch": 66.80, "throughput_fps": 15.0, "peak_vram_mb": 3400.0, "reference_miou_pct": 72.80},
        },
    }
    p2_path = bench_mod.render_gpu_pareto_bw(two_panel_data, tmp_path / "pareto_2panel.png", dpi=300)
    assert len(panel_checks) == 2
    for idx_p, chk in enumerate(panel_checks):
        assert chk["n_labels"] == 6 and not chk["any_overlap"] and not chk["any_clipped"] and not chk["any_leg_overlap"], f"panel {idx_p}: {chk}"
    _assert_monochrome_bw_png(p2_path, expected_dpi=300)
    monkeypatch.setattr(bench_mod.va, "_atomic_save_figure", orig_save)

    # 4. Full E2E check of new deliverables: ROI-bounded+Cand1+VTP_RCB+HotspotTSR, BCSS calibration, LUAD linear/logprob + 2x zoom + overlay + provenance
    bcss_items = []
    for idx, pid in enumerate(("TCGA-E2-A1LK", "TCGA-EW-A1P8", "TCGA-E2-A14N", "TCGA-EW-A1P1")):
        rgb = np.full((32, 32, 3), (120, 120, 120), dtype=np.uint8)
        gt = np.ones((32, 32), dtype=np.uint8)
        if idx < 2:
            rgb[:10, :] = (230, 50, 50)
            gt[:10, :] = 0
        else:
            rgb[:22, :] = (230, 50, 50)
            gt[:22, :] = 0
        bcss_items.append((rgb, gt, f"{pid}_0_0.png"))

    luad_img_d, luad_msk_d = tmp_path / "l_img", tmp_path / "l_msk"
    luad_img_d.mkdir()
    luad_msk_d.mkdir()
    for i in range(4):
        fn = f"CASE0{i}+patch_{i}.png"
        rgb_l = np.full((32, 32, 3), (50, 220, 50), dtype=np.uint8)
        rgb_l[:16, :] = (220, 50, 50)
        Image.fromarray(rgb_l).save(luad_img_d / fn)
        m_l = np.full((32, 32), 3, dtype=np.uint8)
        m_l[:16, :] = 0
        _save_indexed_mask(m_l, luad_msk_d / fn)

    arm_specs = [{"name": nm, "model": _TinyDualHeadStub()} for nm in GATE7_ARM_ORDER]
    res = run_gate7_clinical_and_luad_benchmark(
        arm_specs=arm_specs,
        bcss_test_loader=DataLoader(_TinyBCSSDataset(bcss_items), batch_size=2, shuffle=False),
        luad_test_loader=DataLoader(LUADBCSSAlignedDataset(luad_img_d, luad_msk_d, target_size=(32, 32)), batch_size=2, shuffle=False),
        stage1_cls_ckpt=None,
        author_stage1_ckpt=None,
        author_stage2_ckpt=None,
        output_dir=tmp_path / "e2e_v2",
        zip_path=tmp_path / "e2e_v2.zip",
        device=torch.device("cpu"),
        amp_dtype=None,
        selected_ttas=("raw_1x", "raw_1x", "raw_1x", "raw_1x"),
        n_warmup=1,
        n_iters=2,
        n_repeats=2,
        dpi=150,
        stage1_model_override=_TinyStage1Stub(),
        author_stage1_model_override=_TinyStage1Stub(),
        author_model_override=_TinyDualHeadStub(),
    )
    assert "provenance" in res and "git_commit_sha" in res["provenance"]
    assert "bcss_raw_1x_miou_pct" in res and "ens4_raw_1x" in res["bcss_raw_1x_miou_pct"] and "author_raw" in res["bcss_raw_1x_miou_pct"]
    assert {"author_step0", "ens4_uncal_linear", "ens4_cal_logprob_eps1e3"}.issubset(set(res["bcss_pixel_calibration"].keys()))
    cb = res["clinical_biomarkers"]
    assert cb["guideline_attribution"] == plots_mod.GUIDELINE_ATTRIBUTION
    assert "models_roi_bounded" in cb and "vtp_rcb" in cb["models"]["ens4_cgm"] and "hotspot_tsr" in cb["models"]["ens4_cgm"]
    assert "lins_ccc" in cb["models"]["ens4_cgm"]["vtp"] and "lins_ccc" in cb["models"]["ens4_cgm"]["tsr"]
    assert "raw_1x" in res["gpu_efficiency"] and "deployed_tta" in res["gpu_efficiency"]
    assert "latency_ms_std" in res["gpu_efficiency"]["raw_1x"]["author_stage2"]
    # Verify Primary 4-Arm (no-CGM) does NOT charge Stage-1 CGM overhead while Author Step-0 DOES
    assert res["gpu_efficiency"]["deployed_tta"]["author_stage2"]["uses_stage1_cgm"] is True
    assert res["gpu_efficiency"]["deployed_tta"]["primary_4arm_ensemble"]["uses_stage1_cgm"] is False
    assert res["gpu_efficiency"]["deployed_tta"]["primary_4arm_ensemble"]["tta_mode"] == "4arm_tta_no_cgm"
    luad = res["luad_cross_organ"]
    for mk in (*GATE7_ARM_ORDER, "author_raw", "author_step0", "arm5_no_cgm", "arm5_cgm", "ens4_uncal_linear", "ens4_no_cgm", "ens4_cgm"):
        assert mk in luad["models"], f"Missing LUAD model {mk}"
    assert "arm5_no_cgm_2x_zoom" in luad["magnification_2x_zoom_ablation"] and "ens4_uncal_linear_2x_zoom" in luad["magnification_2x_zoom_ablation"]
    assert "arm5_no_cgm_vs_author_step0" in luad["paired_patient_stats"] and "ens4_uncal_linear_vs_author_step0" in luad["paired_patient_stats"]
    assert (tmp_path / "e2e_v2" / "luad_per_patient_metrics.json").is_file()
    assert (tmp_path / "e2e_v2" / "luad_zero_shot_overlays.png").is_file()
    with zipfile.ZipFile(tmp_path / "e2e_v2.zip", "r") as zf:
        znames = set(zf.namelist())
        assert "luad_per_patient_metrics.json" in znames and "luad_zero_shot_overlays.png" in znames

    # 5. Verify author_step0 calibration applies CGM gate (zeroes out gated FP class) & LUAD 2x zoom keeps batch_size <= B
    class _SpyBatchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_bs = 0
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            self.max_bs = max(self.max_bs, int(x.shape[0]))
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), device=x.device, dtype=torch.float32)
            lg[:, 3] = 2.0  # Strong false-positive NEC before CGM
            lg[:, 0] = 1.0  # True TUM
            return {"seg_logits": lg, "cls_logits": torch.tensor([[3.0, -3.0, -3.0, -3.0]], device=x.device).expand(b, 4)}

    class _GateOffNecS1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            b = int(x.shape[0])
            return torch.zeros((b, 4, 8, 8), device=x.device), torch.tensor([[0.95, 0.01, 0.01, 0.01]], device=x.device).expand(b, 4)

    spy_arm = _SpyBatchModel()
    gt_tum_items = [(np.full((32, 32, 3), 120, dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8), f"TCGA-E2-A1LK_{i}.png") for i in range(4)]
    stream_cal = bench_mod._stream_gate7_predictions(
        arm_specs=[{"name": "arm5_dinov2_large_r2", "model": spy_arm}],
        loader=DataLoader(_TinyBCSSDataset(gt_tum_items), batch_size=2, shuffle=False),
        stage1_cls_ckpt=None,
        author_stage1_ckpt=None,
        author_stage2_ckpt=None,
        device=torch.device("cpu"),
        amp_dtype=None,
        selected_ttas=("raw_1x",),
        stage1_model_override=_GateOffNecS1(),
        author_stage1_model_override=_GateOffNecS1(),
        author_model_override=_SpyBatchModel(),
        collect_raw_1x_and_cal=True,
    )
    # With NEC gated off by Stage-1, author_step0 predicts 100% TUM (NLL ~ 0) whereas author_raw predicts NEC (NLL > 1.0)
    assert stream_cal["pixel_calibration"]["author_raw"]["nll"] > 1.0
    assert stream_cal["pixel_calibration"]["author_step0"]["nll"] < 0.05

    spy_zoom = _SpyBatchModel()
    bench_mod._run_luad_2x_zoom_ablation(
        [{"name": "arm5_dinov2_large_r2", "model": spy_zoom}],
        DataLoader(LUADBCSSAlignedDataset(luad_img_d, luad_msk_d, target_size=(32, 32)), batch_size=2, shuffle=False),
        np.zeros((4, 32, 32), dtype=np.uint8),
        np.arange(4, dtype=np.int32),
        4,
        torch.device("cpu"),
        None,
    )
    assert spy_zoom.max_bs == 2, f"Expected quadrant sequential batch_size=2 (no 4x batch explosion), got {spy_zoom.max_bs}"
