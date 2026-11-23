"""
Comprehensive unit & integration tests for Phase-1 Visual Overlay, Gate-7 Compatibility,
Pipeline Integrity, API Tiling, and Visual Audit Suite (Deliverables A-E).

All hyphens strictly use standard ASCII 0x2D ('-').
"""

import ast
import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.colab_pkg.visualizer import (
    compute_unified_patch_miou,
    plot_whole_roi_ground_truth_comparison,
    render_clinical_comparison_figure,
)
from src.datasets.otsu_filter import compute_otsu_tissue_mask
from src.models.wsss_inference_methods import MultiMethodWSSSEngineV2
from src.pipeline.wsi_lowmag_highmag_pipeline import (
    COLOR_PALETTE,
    WSILowMagHighMagPipeline,
)
from src.stage2_v2.models import (
    load_stage2_v2_checkpoint,
    remap_hf_state_dict_keys,
)
from src.stage2_v2.visual_audit import (
    BCSS_PNG_PALETTE_PREFIX,
    assert_translation_and_orientation_alignment,
    compute_boundary_f1_at_radius,
    compute_translation_shift_matrix,
    compute_tum_centroid_offset,
    run_stage2_v2_visual_audit,
    stitch_column_major_roi,
    verify_flip_rot90_equivariance,
    verify_gt_mask_contract,
    verify_pooled_miou_reconciliation,
    verify_zero_gt_leakage_glass_guard,
)
from src.visualization.clinical_viewer import (
    blend_overlay,
    compute_agreement_error_overlay,
    compute_blurred_dual_verification_overlay,
    compute_cam_overlay,
    compute_patch_iou_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# (A) Bidirectional transformers v4 <-> v5 state_dict remap & Gate-7 ckpts
# ---------------------------------------------------------------------------


def test_bidirectional_dinov2_v4_v5_remap_roundtrip():
    v5_sd = {
        "backbone.encoder.layer.0.attention.q_proj.weight": torch.randn(4, 4),
        "backbone.encoder.layer.0.attention.k_proj.bias": torch.randn(4),
        "backbone.encoder.layer.0.attention.v_proj.weight": torch.randn(4, 4),
        "backbone.encoder.layer.0.attention.o_proj.weight": torch.randn(4, 4),
        "cls_head.weight": torch.randn(4, 4),
    }
    v4_keys = {
        "backbone.encoder.layer.0.attention.attention.query.weight",
        "backbone.encoder.layer.0.attention.attention.key.bias",
        "backbone.encoder.layer.0.attention.attention.value.weight",
        "backbone.encoder.layer.0.attention.output.dense.weight",
        "cls_head.weight",
    }

    # Matching keys -> exact no-op
    noop = remap_hf_state_dict_keys(v5_sd, set(v5_sd.keys()))
    assert set(noop.keys()) == set(v5_sd.keys())

    # v5 -> v4
    to_v4 = remap_hf_state_dict_keys(v5_sd, v4_keys)
    assert set(to_v4.keys()) == v4_keys
    assert torch.equal(
        to_v4["backbone.encoder.layer.0.attention.attention.query.weight"],
        v5_sd["backbone.encoder.layer.0.attention.q_proj.weight"],
    )

    # v4 -> v5 round-trip
    back_to_v5 = remap_hf_state_dict_keys(to_v4, set(v5_sd.keys()))
    assert set(back_to_v5.keys()) == set(v5_sd.keys())
    for k in v5_sd:
        assert torch.equal(back_to_v5[k], v5_sd[k])


def test_bidirectional_segformer_v4_v5_all_14_rules_roundtrip():
    v5_sd = {
        "segformer.decode_head.linear_projections.0.proj.weight": torch.randn(4, 4),
        "segformer.segformer.stages.1.patch_embeddings.proj.weight": torch.randn(4, 4, 3, 3),
        "segformer.segformer.stages.1.layer_norm.weight": torch.randn(4),
        "segformer.segformer.stages.2.blocks.3.layernorm_before.weight": torch.randn(4),
        "segformer.segformer.stages.2.blocks.3.layernorm_after.bias": torch.randn(4),
        "segformer.segformer.stages.2.blocks.3.attention.q_proj.weight": torch.randn(4, 4),
        "segformer.segformer.stages.2.blocks.3.attention.k_proj.weight": torch.randn(4, 4),
        "segformer.segformer.stages.2.blocks.3.attention.v_proj.weight": torch.randn(4, 4),
        "segformer.segformer.stages.2.blocks.3.attention.o_proj.weight": torch.randn(4, 4),
        "segformer.segformer.stages.2.blocks.3.attention.sequence_reduction.sequence_reduction.weight": torch.randn(4, 4, 1, 1),
        "segformer.segformer.stages.2.blocks.3.attention.sequence_reduction.layer_norm.weight": torch.randn(4),
        "segformer.segformer.stages.2.blocks.3.mlp.fc1.weight": torch.randn(8, 4),
        "segformer.segformer.stages.2.blocks.3.mlp.fc2.weight": torch.randn(4, 8),
        "segformer.segformer.stages.2.blocks.3.mlp.dwconv.dwconv.weight": torch.randn(8, 1, 3, 3),
        "segformer.decode_head.linear_fuse.weight": torch.randn(4, 16, 1, 1),
        "segformer.decode_head.batch_norm.weight": torch.randn(4),
        "segformer.decode_head.classifier.weight": torch.randn(4, 4, 1, 1),
        "cls_head.weight": torch.randn(4, 4),
    }
    v4_keys = {
        "segformer.decode_head.linear_c.0.proj.weight",
        "segformer.segformer.encoder.patch_embeddings.1.proj.weight",
        "segformer.segformer.encoder.layer_norm.1.weight",
        "segformer.segformer.encoder.block.2.3.layer_norm_1.weight",
        "segformer.segformer.encoder.block.2.3.layer_norm_2.bias",
        "segformer.segformer.encoder.block.2.3.attention.self.query.weight",
        "segformer.segformer.encoder.block.2.3.attention.self.key.weight",
        "segformer.segformer.encoder.block.2.3.attention.self.value.weight",
        "segformer.segformer.encoder.block.2.3.attention.output.dense.weight",
        "segformer.segformer.encoder.block.2.3.attention.self.sr.weight",
        "segformer.segformer.encoder.block.2.3.attention.self.layer_norm.weight",
        "segformer.segformer.encoder.block.2.3.mlp.dense1.weight",
        "segformer.segformer.encoder.block.2.3.mlp.dense2.weight",
        "segformer.segformer.encoder.block.2.3.mlp.dwconv.dwconv.weight",
        "segformer.decode_head.linear_fuse.weight",
        "segformer.decode_head.batch_norm.weight",
        "segformer.decode_head.classifier.weight",
        "cls_head.weight",
    }

    to_v4 = remap_hf_state_dict_keys(v5_sd, v4_keys)
    assert set(to_v4.keys()) == v4_keys
    assert len(to_v4) == len(v5_sd)

    back_v5 = remap_hf_state_dict_keys(to_v4, set(v5_sd.keys()))
    assert set(back_v5.keys()) == set(v5_sd.keys())
    for k in v5_sd:
        assert torch.equal(back_v5[k], v5_sd[k])


def test_real_gate7_checkpoints_load_strict_when_present():
    ev_root = PROJECT_ROOT / "checkpoints" / "vps_evidence"
    ckpts = sorted(ev_root.glob("se374_g7_arm*_*/stage2_retrained_best.pth"))
    if len(ckpts) < 4:
        pytest.skip("Local Gate-7 Track A checkpoints not present in this environment.")

    for p in ckpts:
        model, payload = load_stage2_v2_checkpoint(p, device=torch.device("cpu"), strict=True)
        assert model is not None
        assert isinstance(payload, dict)


# ---------------------------------------------------------------------------
# (B) Pipeline, Palette, ROI Standardization, Border Padding & Engine V2
# ---------------------------------------------------------------------------


def test_pipeline_palette_matches_clinical_standard_no_str_lym_swap():
    assert COLOR_PALETTE[0] == (228, 26, 28)   # TUM Red
    assert COLOR_PALETTE[1] == (77, 175, 74)   # STR Green
    assert COLOR_PALETTE[2] == (55, 126, 184)  # LYM Blue
    assert COLOR_PALETTE[3] == (255, 127, 0)   # NEC Orange
    assert COLOR_PALETTE[4] == (153, 153, 153) # BACK Gray


def test_standardize_bcss_roi_mask_no_0_4_scrambling_and_preserves_255():
    # Binary TUM (0) + BACK (4) mask must NOT be scrambled into raw BCSS!
    m_04 = np.array([[0, 0, 4], [4, 0, 255]], dtype=np.uint8)
    pred_dummy = np.full_like(m_04, 3)
    out = WSILowMagHighMagPipeline.standardize_bcss_roi_mask(m_04, pred_mask=pred_dummy, label_convention="auto")
    assert np.array_equal(out, m_04)

    # Raw 22-class BCSS mask (values 5..21) must remap properly while preserving 255
    raw_bcss = np.array([[0, 1, 2, 3, 4], [5, 10, 19, 20, 255]], dtype=np.uint8)
    out_raw = WSILowMagHighMagPipeline.standardize_bcss_roi_mask(raw_bcss, label_convention="auto")
    assert int(out_raw[0, 0]) == 4     # 0 outside ROI -> BACK(4)
    assert int(out_raw[0, 1]) == 0     # 1 tumor -> TUM(0)
    assert int(out_raw[0, 2]) == 1     # 2 stroma -> STR(1)
    assert int(out_raw[0, 3]) == 2     # 3 lymphocyte -> LYM(2)
    assert int(out_raw[0, 4]) == 3     # 4 necrosis -> NEC(3)
    assert int(out_raw[1, 0]) == 4     # 5 gland -> BACK(4)
    assert int(out_raw[1, 1]) == 2     # 10 plasma cells -> LYM(2)
    assert int(out_raw[1, 2]) == 0     # 19 angioinvasion -> TUM(0)
    assert int(out_raw[1, 3]) == 0     # 20 dcis -> TUM(0)
    assert int(out_raw[1, 4]) == 255   # ignore preserved


def test_evaluate_wsi_with_gt_png_preserves_p_mode_and_valid_fg_miou(tmp_path):
    # Create an indexed 'P' mode PNG with TUM(0), STR(1), and BACK(4)
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[:, 16:] = 1
    gt[24:, :] = 4  # outside ROI
    im = Image.fromarray(gt, mode="P")
    flat_pal: List[int] = []
    for rgb in BCSS_PNG_PALETTE_PREFIX:
        flat_pal.extend(rgb)
    flat_pal.extend([0] * (768 - len(flat_pal)))
    im.putpalette(flat_pal)
    gt_path = tmp_path / "gt_indexed.png"
    im.save(gt_path)

    # Prediction matches TUM & STR perfectly inside valid_fg, predicts TUM on BACK=4
    pred = np.zeros((32, 32), dtype=np.uint8)
    pred[:, 16:] = 1
    pred[24:, :] = 0  # outside-ROI prediction must NOT pollute canonical 4-class mIoU!

    pipe = WSILowMagHighMagPipeline(device="cpu")
    res = pipe.evaluate_wsi_with_ground_truth_roi(pred, gt_path, label_convention="auto")
    assert res["mIoU_4cls"] == pytest.approx(100.0, abs=1e-2)
    assert res["mDice_4cls"] == pytest.approx(100.0, abs=1e-2)


def test_wsi_pipeline_nonsquare_aspect_ratio_and_border_padding():
    # Non-square slide (W=300, H=250) -> border patches <224px must be reflect-padded & unpadded
    slide = np.full((250, 300, 3), 140, dtype=np.uint8)
    slide[10:240, 10:290] = [180, 90, 140]  # stained H&E region

    class _ConstModel(nn.Module):
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), dtype=torch.float32, device=x.device)
            lg[:, 0] = 3.0
            return {"seg_logits": lg}

    pipe_ar = WSILowMagHighMagPipeline(
        patch_size=224,
        lowmag_target_size=(128, 128),
        tissue_threshold=0.05,
        device="cpu",
        preserve_aspect_ratio=True,
    )
    stitched, report = pipe_ar.run_end_to_end_wsi(
        slide,
        model_stage2=_ConstModel(),
        model_stage1=None,
        method="M1_Baseline_SinglePass",
        max_patches_to_infer=16,
    )
    mw, mh = report.macro_dimensions
    assert mw > mh  # preserves W > H aspect ratio
    assert stitched.shape == (mh, mw)


def test_wsss_engine_dict_unwrap_imagenet_norm_and_bright_glass_guard():
    class _DictStage2(nn.Module):
        input_norm = "imagenet"

        def __init__(self) -> None:
            super().__init__()
            self.seen_mins: List[float] = []

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            self.seen_mins.append(float(x.min().item()))
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), dtype=torch.float32, device=x.device)
            lg[:, 1] = 2.0
            return {"seg_logits": lg}

    m = _DictStage2()
    x01 = torch.full((1, 3, 32, 32), 0.1, dtype=torch.float32)
    x01_clone = x01.clone()
    p_orig, p_d4 = MultiMethodWSSSEngineV2.forward_d4_tta(m, x01)
    assert p_orig.shape == (4, 32, 32)
    assert p_d4.shape == (4, 32, 32)
    # Must have ImageNet-normalized (negative min for 0.1 input) WITHOUT mutating x01!
    assert min(m.seen_mins) < -0.5
    assert torch.equal(x01, x01_clone)

    # Cand1 Bright-Glass Guard: suppresses >=256px bright white glass while sparing stained tissue & small holes
    raw_rgb = np.full((64, 64, 3), [175, 95, 145], dtype=np.uint8)
    raw_rgb[:32, :32] = [245, 244, 245]  # 1024 px bright glass -> should become BACK=4
    raw_rgb[40:48, 40:48] = [245, 245, 245]  # 64 px (< 256) bright lumen -> should be spared!
    pred = np.zeros((64, 64), dtype=np.int64)
    guarded = MultiMethodWSSSEngineV2.apply_bright_glass_guard(pred, raw_rgb, background_class_idx=4)
    assert np.all(guarded[:32, :32] == 4)
    assert np.all(guarded[40:48, 40:48] == 0)
    assert np.all(guarded[32:, :32] == 0)


# ---------------------------------------------------------------------------
# (C) Clinical Viewer, Colab Visualizer & Otsu Filter
# ---------------------------------------------------------------------------


def test_canonical_valid_fg_miou_and_outside_roi_error_map():
    gt = np.zeros((32, 32), dtype=np.int64)
    gt[:16, :] = 0  # TUM
    gt[16:, :] = 4  # BACK (split half stained outside-ROI, half bright glass)

    pred = np.zeros((32, 32), dtype=np.int64)  # predicts TUM everywhere

    raw_rgb = np.full((32, 32, 3), [170, 80, 130], dtype=np.uint8)  # stained H&E everywhere
    raw_rgb[16:, 16:] = [248, 248, 248]  # bright glass on bottom-right

    # Canonical valid_fg 4-class mIoU must be 100.0% (zero outside-ROI FP pollution!)
    m_cv = compute_patch_iou_metrics(pred, gt)
    assert m_cv["miou_4cls"] == pytest.approx(100.0, abs=1e-2)
    assert m_cv["miou_gt_present"] == pytest.approx(100.0, abs=1e-2)
    assert m_cv["mdice_4cls"] == pytest.approx(100.0, abs=1e-2)
    assert "miou_penalized_4cls" in m_cv
    assert "miou_5cls" in m_cv

    m_col = compute_unified_patch_miou(pred, gt)
    assert m_col["miou_4cls"] == pytest.approx(100.0, abs=1e-2)
    assert m_col["miou_gt_present"] == pytest.approx(100.0, abs=1e-2)
    assert "miou_penalized" in m_col

    # Agreement error map: stained outside-ROI must NOT be Crimson/Magenta; bright glass stays Magenta
    _, summ = compute_agreement_error_overlay(raw_rgb, pred, gt)
    assert summ["outside_roi_tissue_px"] == 16 * 16
    assert summ["fp_bright_glass_px"] == 16 * 16
    assert summ["fp_glass_px"] == 16 * 32  # backward-compatible total gt==4 FP count
    assert summ["miou_4cls_roi_canonical"] == pytest.approx(100.0, abs=1e-2)

    blurred_rgb = compute_blurred_dual_verification_overlay(raw_rgb, pred, gt, blur_sigma=0.0)
    assert blurred_rgb.shape == (32, 32, 3)


def test_cam_overlay_4x7x7_blend_ignore255_visualizer_int64_and_otsu_chw(tmp_path):
    raw_rgb = np.full((32, 32, 3), [180, 100, 150], dtype=np.uint8)
    cam_477 = np.random.RandomState(0).rand(4, 7, 7).astype(np.float32)
    cam_vis, cam_meta = compute_cam_overlay(raw_rgb, cam_477)
    assert cam_vis.shape == (32, 32, 3)
    assert isinstance(cam_meta, dict)

    # blend_overlay leaves ignore=255 untouched (not painted black)
    mask_ign = np.full((32, 32), 255, dtype=np.int64)
    blended = blend_overlay(raw_rgb, mask_ign, alpha=0.5)
    assert np.array_equal(blended, raw_rgb)

    # colab_pkg/visualizer.py int64 mask resize works cleanly without TypeError (mismatched 16x16 forces resize!)
    int64_pred = np.zeros((16, 16), dtype=np.int64)
    int64_gt = np.zeros((16, 16), dtype=np.int64)
    fig, _ = plot_whole_roi_ground_truth_comparison(
        rgb_image=raw_rgb,
        pred_mask=int64_pred,
        gt_mask=int64_gt,
        slide_id="Test Int64",
    )
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)

    # colab_pkg/visualizer.py dynamic ImageNet norm & transparent BACK=4
    class _ImNetModel(nn.Module):
        input_norm = "imagenet"

        def __init__(self) -> None:
            super().__init__()
            self.seen_mins: List[float] = []

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            self.seen_mins.append(float(x.min().item()))
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), dtype=torch.float32, device=x.device)
            lg[:, 2] = 3.0
            return {"seg_logits": lg}

    img_dir = tmp_path / "vis_img"
    mask_dir = tmp_path / "vis_mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    img_p = img_dir / "TCGA-TEST-0001+0.png"
    mask_p = mask_dir / "TCGA-TEST-0001+0.png"
    Image.fromarray(np.full((32, 32, 3), 25, dtype=np.uint8), mode="RGB").save(img_p)
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(mask_p)

    im_model = _ImNetModel()
    fig_c, _ = render_clinical_comparison_figure(
        active_test_files=[img_p],
        test_mask_dir=mask_dir,
        model=im_model,
        wsss_engine=MultiMethodWSSSEngineV2(),
        best_method_key="M1_Baseline_SinglePass",
        device="cpu",
        sample_count=1,
        show_plots=False,
    )
    assert fig_c is not None
    assert min(im_model.seen_mins) < -0.5
    plt.close(fig_c)

    # otsu_filter handles (3, H, W) CHW NumPy + negative ImageNet float safely
    chw_rgb = np.transpose(raw_rgb, (2, 0, 1))
    chw_rgb[:, :16, :16] = 250
    t_mask = compute_otsu_tissue_mask(chw_rgb)
    assert t_mask.shape == (32, 32)

    imnet_f = (raw_rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    t_mask_f = compute_otsu_tissue_mask(imnet_f)
    assert t_mask_f.shape == (32, 32)


# ---------------------------------------------------------------------------
# (D) API Server Gate-7 Registry, Sliding-Window Tiling, GT Decoder & UI
# ---------------------------------------------------------------------------


def test_api_server_gate7_tiling_fail_closed_gt_decode_and_compare(monkeypatch):
    from src.visualization import api_server

    class _DummyGate7(nn.Module):
        input_norm = "imagenet"

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), dtype=torch.float32, device=x.device)
            lg[:, 0] = 2.0
            lg[:, 1] = 0.5
            cls_lg = torch.tensor([[2.5, 1.0, -2.0, -2.0]], dtype=torch.float32, device=x.device).repeat(b, 1)
            return {"seg_logits": lg, "cls_logits": cls_lg}

    # Verify RGB-palette GT decoder handles both BCSS palette (153,0,255) & clinical palette (255,127,0)
    rgb_gt = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb_gt[0:4, :] = (255, 0, 0)      # TUM 0
    rgb_gt[4:8, :] = (0, 255, 0)      # STR 1
    rgb_gt[8:12, :] = (153, 0, 255)   # NEC 3 (BCSS)
    rgb_gt[12:16, :] = (255, 127, 0)  # NEC 3 (Clinical)
    decoded = api_server._decode_gt_pil_to_index_mask(Image.fromarray(rgb_gt, mode="RGB"))
    assert np.all(decoded[0:4, :] == 0)
    assert np.all(decoded[4:8, :] == 1)
    assert np.all(decoded[8:12, :] == 3)
    assert np.all(decoded[12:16, :] == 3)

    # Inject dummy Gate-7 arm5 into manager and test 320x320 sliding-window tiling + softmax probs
    mgr = api_server._manager
    mgr.ensure_initialized()
    orig_models = dict(mgr._gate7_models)
    orig_paths = dict(mgr.gate7_ckpt_paths)
    try:
        mgr._gate7_models["arm5"] = _DummyGate7()
        mgr.gate7_ckpt_paths["arm5"] = Path("dummy_arm5.pth")

        # 320x320 image triggers 224x224 feather-blended sliding-window tiling (> 256px)
        big_img = np.full((320, 320, 3), [175, 95, 145], dtype=np.uint8)
        payload = api_server.process_single_patch(big_img, mode="gate7_arm5", enforce_otsu=False)
        assert payload["method_used"] == "gate7_arm5"
        assert payload["tiling_used"] is True
        assert payload["mask_shape"] == [320, 320]
        assert 0.0 <= payload["mean_entropy"] <= 1.0
        probs_arr = payload["_probs_arr"]
        assert float(probs_arr.min()) >= 0.0 and float(probs_arr.max()) <= 1.0 + 1e-5

        # Verify /api/v1/compare resizes mismatched GT cleanly & /api/v1/predict/wsi allows gate7_arm5
        client = TestClient(api_server.app)
        buf_img = io.BytesIO()
        Image.fromarray(np.full((64, 64, 3), [175, 95, 145], dtype=np.uint8)).save(buf_img, format="PNG")
        img_b64 = base64.b64encode(buf_img.getvalue()).decode("ascii")
        buf_gt = io.BytesIO()
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(buf_gt, format="PNG")
        gt_b64 = base64.b64encode(buf_gt.getvalue()).decode("ascii")

        resp_cmp = client.post(
            "/api/v1/compare",
            json={
                "image_base64": img_b64,
                "gt_mask_base64": gt_b64,
                "methods": ["gate7_arm5"],
                "apply_otsu_gating": False,
            },
        )
        assert resp_cmp.status_code == 200
        cmp_data = resp_cmp.json()
        assert cmp_data["gt_color_base64"] is not None
        gt_b64_raw = cmp_data["gt_color_base64"].split(",", 1)[-1]
        gt_dec = Image.open(io.BytesIO(base64.b64decode(gt_b64_raw)))
        assert gt_dec.size == (64, 64)

        resp_wsi = client.post(
            "/api/v1/predict/wsi",
            json={
                "slide_id": "TEST_GATE7_WSI",
                "mode": "gate7_arm5",
                "tiles": [{"id": "t0", "x": 0, "y": 0, "image_base64": img_b64}],
            },
        )
        assert resp_wsi.status_code == 200
        assert resp_wsi.json()["region_summary"]["method_used"] == "gate7_arm5"

        # Fail-closed HTTP 503 when explicit WSSS_STAGE2_V2_CKPT is missing
        monkeypatch.setenv("WSSS_STAGE2_V2_CKPT", str(PROJECT_ROOT / "nonexistent_v2_ckpt_999.pth"))
        resp_503 = client.post(
            "/api/v1/predict/patch",
            json={"image_base64": img_b64, "mode": "gate7_arm5"},
        )
        assert resp_503.status_code == 503
    finally:
        mgr._gate7_models.clear()
        mgr._gate7_models.update(orig_models)
        mgr.gate7_ckpt_paths.clear()
        mgr.gate7_ckpt_paths.update(orig_paths)


def test_index_html_has_gate7_modes_crisp_defaults_and_hygiene():
    html = (PROJECT_ROOT / "src" / "visualization" / "static" / "index.html").read_text(encoding="utf-8")
    assert "gate7_arm5" in html
    assert "gate7_4arm_cgm" in html
    assert "strict_weights" in html
    assert "applySeparableBoxBlurRGBA" in html
    assert "wsiRowsInput" in html
    assert "alert(" not in html
    assert "SE374" not in html
    assert "UIT" not in html
    for bad_dash in ("\u2014", "\u2013", "\u2015", "\u2212"):
        assert bad_dash not in html


# ---------------------------------------------------------------------------
# (E) Visual Audit Suite & 02d Notebook End-to-End Verification
# ---------------------------------------------------------------------------


class _EquivariantColorModel(nn.Module):
    """Exact spatial pixel-wise equivariant dual-head model for fast deterministic visual audit tests."""

    def __init__(self, input_norm: str = "imagenet") -> None:
        super().__init__()
        self.input_norm = input_norm
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.input_norm == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            rgb = (x * std + mean).clamp(0.0, 1.0)
        else:
            rgb = x.clamp(0.0, 1.0)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        # Deterministic pixel-wise mapping: R->TUM(0), G->STR(1), B->LYM(2), dark/yellow->NEC(3)
        l0 = (r - 0.5 * g - 0.5 * b) * 6.0
        l1 = (g - 0.5 * r - 0.5 * b) * 6.0
        l2 = (b - 0.5 * r - 0.5 * g) * 6.0
        l3 = ((r + g) * 0.5 - b - 0.35) * 6.0
        seg = torch.stack([l0, l1, l2, l3], dim=1)
        cls_lg = seg.amax(dim=(2, 3))
        return {"seg_logits": seg, "cls_logits": cls_lg}


class _DummyStage1Cls(nn.Module):
    def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, hh, ww = x.shape
        cams = torch.ones((b, 4, max(1, hh // 4), max(1, ww // 4)), device=x.device)
        probs = torch.full((b, 4), 0.9, device=x.device)
        return cams, probs


class _SyntheticBCSSDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, Path, str]]) -> None:
        self.samples = [(str(a), str(b)) for a, b, _ in items]
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_p, mask_p, fname = self._items[idx]
        rgb = np.asarray(Image.open(img_p).convert("RGB"), dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = np.transpose((rgb - mean) / std, (2, 0, 1))
        gt = np.asarray(Image.open(mask_p), dtype=np.int64)
        return torch.from_numpy(norm), torch.from_numpy(gt), fname


def _save_bcss_indexed_mask(mask: np.ndarray, path: Path) -> None:
    im = Image.fromarray(mask.astype(np.uint8), mode="P")
    flat_pal: List[int] = []
    for rgb in BCSS_PNG_PALETTE_PREFIX:
        flat_pal.extend(rgb)
    flat_pal.extend([0] * (768 - len(flat_pal)))
    im.putpalette(flat_pal)
    im.save(path)


def test_visual_audit_assertions_shift_detection_and_seam_check(tmp_path):
    # 1. Verify translation-shift catches an intentional 5px horizontal shift!
    gt = np.zeros((2, 64, 64), dtype=np.uint8)
    gt[:, 8:32, 10:34] = 1
    gt[:, 32:56, 20:50] = 2
    gt[:, 8:28, 36:56] = 3

    rep_ok = compute_translation_shift_matrix({"aligned": gt.copy()}, gt, shifts=(-6, -4, -2, -1, 0, 1, 2, 4, 6))
    a_ok = assert_translation_and_orientation_alignment(rep_ok, enforce_strict=True)
    assert a_ok["passed"] is True
    assert rep_ok["per_model"]["aligned"]["peak_dx"] == 0
    assert rep_ok["per_model"]["aligned"]["peak_dy"] == 0

    shifted_pred = np.roll(gt, shift=5, axis=2)  # deliberate 5px dx shift -> must FAIL!
    rep_bad = compute_translation_shift_matrix({"shifted": shifted_pred}, gt, shifts=(-6, -4, -2, -1, 0, 1, 2, 4, 6))
    with pytest.raises(RuntimeError, match=r"\[STAGE2 ERROR\]"):
        assert_translation_and_orientation_alignment(rep_bad, enforce_strict=True)

    # 2. Verify column-major ROI stitcher seam check passes on smooth grid and fails on scrambled row-major tiles
    full_img = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    full_img[..., 0] = (xx * 3).astype(np.uint8)
    full_img[..., 1] = (yy * 3).astype(np.uint8)
    full_img[..., 2] = 120

    # Column-major packing: idx = col * n_rows + row
    tiles_cm = {
        0: full_img[0:32, 0:32],
        1: full_img[32:64, 0:32],
        2: full_img[0:32, 32:64],
        3: full_img[32:64, 32:64],
    }
    stitched, meta = stitch_column_major_roi(tiles_cm, 2, 2, tile_size=32, enforce_seam=True, max_seam_l1=20.0)
    assert np.array_equal(stitched, full_img)
    assert meta["h_seam_l1"] < 5.0 and meta["v_seam_l1"] < 5.0

    # Scrambled high-contrast tiles -> seam L1 >= 20 raises [STAGE2 ERROR]
    tiles_bad = {
        0: np.full((32, 32, 3), 10, dtype=np.uint8),
        1: np.full((32, 32, 3), 240, dtype=np.uint8),
        2: np.full((32, 32, 3), 240, dtype=np.uint8),
        3: np.full((32, 32, 3), 10, dtype=np.uint8),
    }
    with pytest.raises(RuntimeError, match=r"\[STAGE2 ERROR\]"):
        stitch_column_major_roi(tiles_bad, 2, 2, tile_size=32, enforce_seam=True, max_seam_l1=20.0)

    # 3. Verify GT mask contract fails on RGB PNG and passes on indexed 'P' BCSS PNG
    p_ok = tmp_path / "ok.png"
    _save_bcss_indexed_mask(gt[0], p_ok)
    res_gt = verify_gt_mask_contract([p_ok], expected_hw=(64, 64), check_palette=True)
    assert res_gt["passed"] is True

    p_rgb = tmp_path / "bad_rgb.png"
    Image.fromarray(full_img, mode="RGB").save(p_rgb)
    with pytest.raises(RuntimeError, match=r"\[STAGE2 ERROR\]"):
        verify_gt_mask_contract([p_rgb], expected_hw=(64, 64), check_palette=True)

    # 4. Verify real BCSS-WSSS test PNGs pass Assertion 1 when dataset exists locally
    real_mask_dir = PROJECT_ROOT / "data" / "BCSS-WSSS" / "test" / "mask"
    if real_mask_dir.is_dir():
        real_masks = sorted(real_mask_dir.glob("*.png"))[:10]
        if real_masks:
            real_res = verify_gt_mask_contract(real_masks, expected_hw=(224, 224), check_palette=True)
            assert real_res["passed"] is True
            assert real_res["palette_verified_count"] == len(real_masks)

    # 5. Verify Boundary F1, TUM centroid offset, zero-GT glass guard, and reconciliation checks
    bf1 = compute_boundary_f1_at_radius(gt, gt, radius=2)
    assert bf1["macro_boundary_f1"] == pytest.approx(1.0, abs=1e-4)
    cent = compute_tum_centroid_offset(gt, gt, min_tum_px=16)
    assert cent["median_px"] == pytest.approx(0.0, abs=1e-4)
    assert verify_zero_gt_leakage_glass_guard([gt[0]], [full_img], [gt[0]])["passed"] is True
    assert verify_pooled_miou_reconciliation({"a": 0.7280}, {"a": 0.7281}, tol=0.0015)["passed"] is True
    with pytest.raises(RuntimeError, match=r"\[STAGE2 ERROR\]"):
        verify_pooled_miou_reconciliation({"a": 0.7000}, {"a": 0.7281}, tol=0.0015, enforce_strict=True)


def test_run_stage2_v2_visual_audit_end_to_end_synthetic(tmp_path):
    img_dir = tmp_path / "img"
    mask_dir = tmp_path / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Build 8 distinct asymmetric 64x64 synthetic patches (including a complete 2x2 column-major ROI!)
    items: List[Tuple[Path, Path, str]] = []
    roi_prefix = "TCGA-TEST-ROI1-DX1_xmin100_ymin200_MPP-0.2500"
    specs = [
        (f"{roi_prefix}+0.png", False, False),
        (f"{roi_prefix}+1.png", False, False),
        (f"{roi_prefix}+2.png", False, False),
        (f"{roi_prefix}+3.png", False, False),
        ("TCGA-E2-A1LL-DX1_xmin10_ymin20_MPP-0.2500+0.png", True, False),
        ("TCGA-GI-A2C9-DX1_xmin10_ymin20_MPP-0.2500+0.png", False, True),
        ("TCGA-EW-A3U0-DX1_xmin10_ymin20_MPP-0.2500+0.png", False, False),
        ("TCGA-EW-A6SB-DX1_xmin10_ymin20_MPP-0.2500+0.png", False, False),
    ]

    for idx, (fname, heavy_glass, heavy_outside) in enumerate(specs):
        gt = np.zeros((64, 64), dtype=np.uint8)
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        # Asymmetric 4-class pattern so flip/rot90/transpose collapse clearly
        gt[0:28, 0:34] = 0
        rgb[0:28, 0:34] = (230, 30, 30)
        gt[0:30, 34:64] = 1
        rgb[0:30, 34:64] = (30, 220, 30)
        gt[28:64, 0:26] = 2
        rgb[28:64, 0:26] = (30, 40, 230)
        gt[30:64, 26:64] = 3
        rgb[30:64, 26:64] = (230, 210, 20)

        if heavy_glass:
            gt[32:, :] = 4
            rgb[32:, :] = (248, 248, 248)
        elif heavy_outside:
            gt[32:, :] = 4
            rgb[32:, :] = (160, 90, 140)
        elif idx < 4:
            # Make the 2x2 ROI tiles share identical border color for <20 seam L1
            rgb[0, :, :] = (180, 120, 140)
            rgb[-1, :, :] = (180, 120, 140)
            rgb[:, 0, :] = (180, 120, 140)
            rgb[:, -1, :] = (180, 120, 140)

        ip = img_dir / fname
        mp = mask_dir / fname
        Image.fromarray(rgb, mode="RGB").save(ip)
        _save_bcss_indexed_mask(gt, mp)
        items.append((ip, mp, fname))

    ds = _SyntheticBCSSDataset(items)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    arm_specs = [
        {"name": "arm8_deeplab_r101_r2", "model": _EquivariantColorModel("imagenet")},
        {"name": "arm6_segformer_b5_r2", "model": _EquivariantColorModel("imagenet")},
        {"name": "arm7_convnext_large_r2", "model": _EquivariantColorModel("imagenet")},
        {"name": "arm5_dinov2_large_r2", "model": _EquivariantColorModel("imagenet")},
    ]

    out_dir = tmp_path / "stage2_results"
    zip_path = tmp_path / "SE374_Stage2_Training_Results.zip"
    metrics = run_stage2_v2_visual_audit(
        arm_specs=arm_specs,
        test_loader=loader,
        output_dir=out_dir,
        zip_path=zip_path,
        device=torch.device("cpu"),
        selected_ttas=("raw_1x", "raw_1x", "raw_1x", "raw_1x"),
        enforce_strict_reconciliation=False,
        enforce_sha=False,
        enforce_palette_check=True,
        enforce_equivariance=True,
        enforce_shift_alignment=True,
        dpi=100,
        stage1_model_override=_DummyStage1Cls(),
        author_model_override=_EquivariantColorModel("zero_one"),
        whole_roi_targets=((roi_prefix, 2, 2, "09_whole_roi_synthetic_2x2.png", "multiclass"),),
    )

    # All 6 assertions recorded & passed (reconciliation not strict on synthetic)
    for k in (
        "assertion_1_gt_png_contract",
        "assertion_2_ckpt_sha256_and_norm",
        "assertion_3_flip_rot90_equivariance",
        "assertion_4_translation_shift_alignment",
        "assertion_5_zero_gt_leakage_glass_guard",
    ):
        assert metrics["assertions"][k]["passed"] is True

    assert metrics["pooled_test_metrics"]["ens4_simplex_cgm"]["secondary_exploratory"] is True
    assert (out_dir / "STAGE2_TRAINING_METRICS.json").is_file()
    assert (out_dir / "stage2_training_loss.png").is_file()
    assert (out_dir / "11_spatial_alignment_monochrome_300dpi.png").is_file()
    assert (out_dir / "09_whole_roi_synthetic_2x2.png").is_file()
    assert zip_path.is_file()

    # Verify Academic Monochrome Invariant-24: 11_spatial_alignment_monochrome_300dpi.png is strictly grayscale!
    mono_arr = np.asarray(Image.open(out_dir / "11_spatial_alignment_monochrome_300dpi.png").convert("RGB"))
    assert np.array_equal(mono_arr[..., 0], mono_arr[..., 1])
    assert np.array_equal(mono_arr[..., 1], mono_arr[..., 2])

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for req in (
            "STAGE2_TRAINING_METRICS.json",
            "stage2_training_loss.png",
            "11_spatial_alignment_monochrome_300dpi.png",
            "01_quantile_P90.png",
            "07_glass_guard_comparison_gt_free.png",
            "09_whole_roi_synthetic_2x2.png",
            "per_patch_visual_audit.csv",
            "per_patient_class_ious_all_models.json",
            "translation_shift_matrix.csv",
            "whole_roi_audit.json",
        ):
            assert req in names, f"Missing {req} in visual audit ZIP"


def test_02d_visual_audit_notebook_valid_json_ast_and_contracts():
    nb_path = PROJECT_ROOT / "notebooks" / "02d_Stage2_Visual_Overlay_Audit_G4.ipynb"
    assert nb_path.is_file()
    raw = nb_path.read_text(encoding="utf-8")
    assert "khoinguyen59/medical-wsss.git" in raw
    assert "medical-ai-consortium/bcss-wsss" not in raw
    assert "run_stage2_v2_visual_audit" in raw
    for bad_dash in ("\u2014", "\u2013", "\u2015", "\u2212"):
        assert bad_dash not in raw

    nb = json.loads(raw)
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 6
    for idx, cell in enumerate(code_cells):
        src_txt = "".join(cell["source"])
        ast.parse(src_txt, filename=f"02d_cell_{idx}")
    assert "stage1_cls_ckpt=STAGE1_BEST_CLS_PATH" in raw
    assert "SHA256_STAGE1_BEST_CLS" in raw
    assert "author_stage1_ckpt=STAGE1_RETRAINED_PATH" in raw
    assert "SHA256_STAGE1_RETRAINED" in raw


def test_visual_audit_bcss_val_dataset_and_disk_ckpt_loading(tmp_path):
    """Regression test for Bugs E1 & E2: disk ckpt_path loading + BCSSStage2ValDataset (.files/.mask_dir)."""
    from src.stage2_v2.models import SegFormerDualHead
    from src.training.dataset_loader import BCSSStage2ValDataset

    img_dir = tmp_path / "test_img"
    mask_dir = tmp_path / "test_mask"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Create 4 patches from 2 distinct patients (including a complete 2x1 ROI)
    roi_prefix = "TCGA-EW-A1P4-DX1_xmin17256_ymin35430_MPP-0.2500"
    names = [
        f"{roi_prefix}+0.png",
        f"{roi_prefix}+1.png",
        "TCGA-E2-A1LL-DX1_xmin10_ymin20_MPP-0.2500+0.png",
        "TCGA-EW-A6SB-DX1_xmin10_ymin20_MPP-0.2500+0.png",
    ]
    for fn in names:
        rgb = np.full((224, 224, 3), (180, 110, 145), dtype=np.uint8)
        gt = np.zeros((224, 224), dtype=np.uint8)
        gt[:112, 112:] = 1
        gt[112:, :112] = 2
        gt[112:, 112:] = 3
        Image.fromarray(rgb, mode="RGB").save(img_dir / fn)
        _save_bcss_indexed_mask(gt, mask_dir / fn)

    # Use REAL BCSSStage2ValDataset (has .files + .mask_dir, NOT .samples, returns 2-tuple)
    ds = BCSSStage2ValDataset(img_dir, mask_dir, strict_patient_filter=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    # Save a real Stage-2 v2 tiny SegFormer checkpoint to disk to exercise ckpt_path loading!
    tiny_sf = SegFormerDualHead(
        backbone_id="nvidia/mit-b5",
        num_classes=4,
        pretrained=False,
        tiny_offline_config=True,
        arch="segformer_b5",
    )
    ckpt_p = tmp_path / "arm6_disk.pth"
    torch.save({**tiny_sf.model_metadata(), "state_dict": tiny_sf.state_dict()}, ckpt_p)

    arm_specs = [
        {"name": "arm8_deeplab_r101_r2", "model": _EquivariantColorModel("imagenet")},
        {"name": "arm6_segformer_b5_r2", "ckpt_path": ckpt_p},
        {"name": "arm7_convnext_large_r2", "model": _EquivariantColorModel("imagenet")},
        {"name": "arm5_dinov2_large_r2", "model": _EquivariantColorModel("imagenet")},
    ]

    metrics = run_stage2_v2_visual_audit(
        arm_specs=arm_specs,
        test_loader=loader,
        output_dir=tmp_path / "out",
        zip_path=tmp_path / "bundle.zip",
        device=torch.device("cpu"),
        selected_ttas=("raw_1x", "raw_1x", "raw_1x", "raw_1x"),
        enforce_strict_reconciliation=False,
        enforce_sha=False,
        enforce_palette_check=True,
        enforce_equivariance=False,
        enforce_shift_alignment=False,
        dpi=72,
        stage1_model_override=_DummyStage1Cls(),
        author_model_override=_EquivariantColorModel("zero_one"),
        whole_roi_targets=((roi_prefix, 2, 1, "09_whole_roi_2x1.png", "nec_fp"),),
    )

    # Must have discovered all 3 real TCGA patients & verified all 4 BCSS PNG palettes & stitched 2x1 ROI!
    assert metrics["n_test_patients"] == 3
    assert metrics["assertions"]["assertion_1_gt_png_contract"]["palette_verified_count"] == 4
    assert len(metrics["whole_roi_audit"]) == 1
    assert metrics["whole_roi_audit"][0].get("skipped_incomplete") is not True
    assert (tmp_path / "out" / "09_whole_roi_2x1.png").is_file()


def test_equivariance_catches_axis_inversion_bug():
    """Verify Assertion 3 fails fast when a model breaks flip/rot90 equivariance."""

    class _NonEquivariantModel(nn.Module):
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), dtype=torch.float32, device=x.device)
            # Ignores input orientation: always paints top-left=0, top-right=1, bot-left=2, bot-right=3
            lg[:, 0, : hh // 2, : ww // 2] = 5.0
            lg[:, 1, : hh // 2, ww // 2 :] = 5.0
            lg[:, 2, hh // 2 :, : ww // 2] = 5.0
            lg[:, 3, hh // 2 :, ww // 2 :] = 5.0
            return {"seg_logits": lg, "cls_logits": torch.ones((b, 4), device=x.device)}

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:16, :16] = (220, 30, 30)
    rgb[:16, 16:] = (30, 220, 30)
    rgb[16:, :16] = (30, 30, 220)
    rgb[16:, 16:] = (220, 200, 30)

    with pytest.raises(RuntimeError, match=r"\[STAGE2 ERROR\]"):
        verify_flip_rot90_equivariance(
            {"broken_model": _NonEquivariantModel()},
            [rgb],
            {"broken_model": "imagenet"},
            device=torch.device("cpu"),
            enforce_strict=True,
        )


def test_api_server_tiled_cgm_preservation_and_strict_weights_export_wsi(monkeypatch):
    """Regression test for Bugs D1 & D4: tiled Gate-7 preserves CGM + strict_weights on export/wsi."""
    from src.visualization import api_server

    class _CGMProbeArm5(nn.Module):
        input_norm = "imagenet"

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), dtype=torch.float32, device=x.device)
            # Pixel logits strongly prefer NEC (class 3), with TUM (class 0) second
            lg[:, 3] = 4.0
            lg[:, 0] = 1.5
            # Patch cls_head strongly gates NEC OFF (-6.0 -> sigmoid ~0.002 < 0.65) and TUM ON (+6.0)
            cls_lg = torch.tensor([[6.0, -6.0, -6.0, -6.0]], dtype=torch.float32, device=x.device).repeat(b, 1)
            return {"seg_logits": lg, "cls_logits": cls_lg}

    mgr = api_server._manager
    mgr.ensure_initialized()
    orig_models = dict(mgr._gate7_models)
    orig_paths = dict(mgr.gate7_ckpt_paths)
    orig_s2_loaded = mgr.stage2_loaded
    try:
        mgr._gate7_models["arm5"] = _CGMProbeArm5()
        mgr.gate7_ckpt_paths["arm5"] = Path("dummy_arm5.pth")

        # 320x320 triggers sliding-window tiling (> 256px): CGM MUST suppress NEC (class 3) completely!
        big_img = np.full((320, 320, 3), [175, 95, 145], dtype=np.uint8)
        res = api_server.process_single_patch(big_img, mode="gate7_arm5", enforce_otsu=False)
        assert res["tiling_used"] is True
        assert np.all(res["pred_mask"] == 0), "Tiled gate7_arm5 dropped CGM gating!"
        assert float(res["_probs_arr"][3].max()) == pytest.approx(0.0, abs=1e-6)

        # Verify strict_weights=True returns HTTP 503 on /api/v1/export/report & /api/v1/predict/wsi when Stage-2 absent
        mgr.stage2_loaded = False
        client = TestClient(api_server.app)
        buf = io.BytesIO()
        Image.fromarray(np.full((32, 32, 3), [175, 95, 145], dtype=np.uint8)).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        r_exp = client.post(
            "/api/v1/export/report",
            json={"image_base64": img_b64, "mode": "m6", "strict_weights": True, "format": "json"},
        )
        assert r_exp.status_code == 503

        r_wsi = client.post(
            "/api/v1/predict/wsi",
            json={
                "slide_id": "STRICT_TEST",
                "mode": "m6",
                "strict_weights": True,
                "tiles": [{"id": "t0", "x": 0, "y": 0, "image_base64": img_b64}],
            },
        )
        assert r_wsi.status_code == 503
    finally:
        mgr._gate7_models.clear()
        mgr._gate7_models.update(orig_models)
        mgr.gate7_ckpt_paths.clear()
        mgr.gate7_ckpt_paths.update(orig_paths)
        mgr.stage2_loaded = orig_s2_loaded


def test_otsu_bright_imagenet_glass_and_visualizer_no_double_norm(tmp_path):
    """Regression test for Bugs C1 & B1: pure-white ImageNet glass + no double ImageNet norm on mid-tone H&E."""
    # Pure white RGB=[255,255,255] after ImageNet normalization is [+2.249, +2.429, +2.640]
    white_01 = np.ones((64, 64, 3), dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    white_imnet = (white_01 - mean) / std
    t_mask = compute_otsu_tissue_mask(white_imnet)
    assert int(t_mask.sum()) == 0, "Pure-white ImageNet float patch was misclassified as tissue!"

    # Mid-tone uniform H&E RGB=[140, 140, 140] -> single ImageNet norm R=(0.549-0.485)/0.229 = +0.28
    # Double normalization would push R to (0.28-0.485)/0.229 = -0.90!
    class _MidToneProbeModel(nn.Module):
        input_norm = "imagenet"

        def __init__(self) -> None:
            super().__init__()
            self.seen_r_means: List[float] = []

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            self.seen_r_means.append(float(x[:, 0].mean().item()))
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), dtype=torch.float32, device=x.device)
            lg[:, 0] = 2.0
            return {"seg_logits": lg}

    img_dir = tmp_path / "mt_img"
    mask_dir = tmp_path / "mt_mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    ip = img_dir / "TCGA-TEST-MIDT+0.png"
    mp = mask_dir / "TCGA-TEST-MIDT+0.png"
    Image.fromarray(np.full((32, 32, 3), 140, dtype=np.uint8), mode="RGB").save(ip)
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(mp)

    probe = _MidToneProbeModel()
    fig, _ = render_clinical_comparison_figure(
        active_test_files=[ip],
        test_mask_dir=mask_dir,
        model=probe,
        wsss_engine=MultiMethodWSSSEngineV2(),
        best_method_key="M1_Baseline_SingleScale",
        device="cpu",
        sample_count=1,
        show_plots=False,
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
    expected_r = (140.0 / 255.0 - 0.485) / 0.229  # ~ +0.2795
    assert probe.seen_r_means[0] == pytest.approx(expected_r, abs=0.02), (
        f"Double ImageNet normalization detected: got R mean {probe.seen_r_means[0]:.3f}, expected {expected_r:.3f}"
    )


def test_visual_audit_6_surgical_blockers_regression(tmp_path):
    """Comprehensive regression test covering all 6 G4 visual audit blockers and polish fixes."""
    from src.stage2_v2.visual_audit import (
        GATE7_ARM_ORDER,
        GATE7_FULL_VAL_BETAS,
        GATE7_SELECTED_TTAS,
        GATE7_SIMPLEX_WEIGHTS,
        SHIFT_GRID,
        _denorm_imagenet_to_uint8,
        _render_glass_guard_ablation_sheet,
        _select_5x5_zoom_origin,
        select_audit_sheet_indices,
        verify_checkpoints_and_norm_contract,
    )

    # Blocker 1: Widened SHIFT_GRID includes +/-16 and calibrated min_orient_margin=0.05
    assert -16 in SHIFT_GRID and 16 in SHIFT_GRID
    synthetic_shift_ok = {
        "shifts": list(SHIFT_GRID),
        "margin": 16,
        "per_model": {
            "arm8": {
                "miou_zero_shift": 0.720,
                "peak_dx": 2,
                "peak_dy": 2,
                "miou_peak": 0.724,
                "far_border_max_miou": 0.690,
                "max_orient_miou": 0.640,  # margin = 0.080 > 0.05 -> PASS
            }
        },
    }
    assert assert_translation_and_orientation_alignment(synthetic_shift_ok, enforce_strict=True)["passed"] is True
    synthetic_shift_bad = {
        "shifts": list(SHIFT_GRID),
        "margin": 16,
        "per_model": {
            "arm8": {
                "miou_zero_shift": 0.660,
                "peak_dx": 0,
                "peak_dy": 0,
                "miou_peak": 0.660,
                "far_border_max_miou": 0.620,
                "max_orient_miou": 0.640,  # margin = 0.020 <= 0.05 -> FAIL
            }
        },
    }
    with pytest.raises(RuntimeError, match=r"\[STAGE2 ERROR\]"):
        assert_translation_and_orientation_alignment(synthetic_shift_bad, enforce_strict=True)

    # Blocker 2: verify_checkpoints_and_norm_contract enforces author_stage1_ckpt SHA-256
    fake_s1 = tmp_path / "stale_stage1_retrained.pth"
    fake_s1.write_bytes(b"stale_weights_payload")
    with pytest.raises(RuntimeError, match=r"Author Stage-1 SHA-256 mismatch"):
        verify_checkpoints_and_norm_contract(
            arm_specs=[{"name": n} for n in GATE7_ARM_ORDER],
            author_stage1_ckpt=fake_s1,
            enforce_sha=True,
        )

    # Blocker 3: Mid-tone ImageNet tensor [140, 140, 140] -> [+0.28, +0.42, +0.64] must still denormalize!
    mid_rgb = np.full((32, 32, 3), 140, dtype=np.uint8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    mid_imnet = (np.transpose(mid_rgb.astype(np.float32) / 255.0, (2, 0, 1)) - mean) / std
    assert float(mid_imnet.min()) > 0.0 and float(mid_imnet.max()) < 1.0
    restored = _denorm_imagenet_to_uint8(mid_imnet, images_are_imagenet_norm=True)
    assert np.allclose(restored, mid_rgb, atol=1)

    # Blocker 4 & 5: Heavy-BACK requires fg_px >= 5000 (rejecting fg_px=2/28), and Sheet 08 uses no_cgm + distinct LYM patients
    def _mk_rec(
        idx: int,
        fn: str,
        pid: str,
        fg_px: int,
        bg_frac: float = 0.0,
        or_frac: float = 0.0,
        arm5_n: int = 0,
        no_cgm_n: int = 0,
        cgm_n: int = 0,
        arm5_l: int = 0,
        no_cgm_l: int = 0,
        cgm_l: int = 0,
    ) -> Dict[str, object]:
        return {
            "index": idx,
            "filename": fn,
            "patient_id": pid,
            "fg_px": fg_px,
            "gt_tum_px": fg_px // 2,
            "gt_str_px": fg_px // 4,
            "gt_lym_px": fg_px // 8,
            "gt_nec_px": fg_px - (fg_px // 2 + fg_px // 4 + fg_px // 8),
            "back_frac": bg_frac + or_frac,
            "bright_glass_frac": bg_frac,
            "outside_roi_frac": or_frac,
            "author_step0_miou": 0.70,
            "arm5_cgm_miou": 0.72,
            "ens4_no_cgm_miou": 0.72,
            "ens4_cgm_miou": 0.72 + idx * 0.001,
            "arm5_cgm_nec_fp_px": arm5_n,
            "ens4_no_cgm_nec_fp_px": no_cgm_n,
            "ens4_cgm_nec_fp_px": cgm_n,
            "arm5_cgm_lym_fp_px": arm5_l,
            "ens4_no_cgm_lym_fp_px": no_cgm_l,
            "ens4_cgm_lym_fp_px": cgm_l,
        }

    records = [
        # Degenerate fg_px=2 / 28 patches with 99% glass / outside-ROI MUST be rejected when fg_px >= 5000 exists!
        _mk_rec(0, "TCGA-A1-0001+0.png", "TCGA-A1-0001", fg_px=2, bg_frac=0.99),
        _mk_rec(1, "TCGA-A2-0002+0.png", "TCGA-A2-0002", fg_px=28, or_frac=0.99),
        _mk_rec(2, "TCGA-B1-0001+0.png", "TCGA-B1-0001", fg_px=8000, bg_frac=0.65),
        _mk_rec(3, "TCGA-B2-0002+0.png", "TCGA-B2-0002", fg_px=7500, bg_frac=0.55),
        _mk_rec(4, "TCGA-C1-0001+0.png", "TCGA-C1-0001", fg_px=9000, or_frac=0.70),
        _mk_rec(5, "TCGA-C2-0002+0.png", "TCGA-C2-0002", fg_px=8500, or_frac=0.60),
        # Divergence: A1P4 NEC FP lives in ens4_no_cgm while ens4_cgm suppresses it to 0
        _mk_rec(6, "TCGA-EW-A1P4+0.png", "TCGA-EW-A1P4", fg_px=20000, arm5_n=0, no_cgm_n=12000, cgm_n=0),
        _mk_rec(7, "TCGA-EW-A1P4+1.png", "TCGA-EW-A1P4", fg_px=20000, arm5_n=0, no_cgm_n=8000, cgm_n=0),
        _mk_rec(8, "TCGA-EW-A1P4+2.png", "TCGA-EW-A1P4", fg_px=20000, arm5_n=0, no_cgm_n=50, cgm_n=50),
        # LYM FP: two patches in A159 with highest FP, one in A6SB -> distinct patient selection picks 1 from each!
        _mk_rec(9, "TCGA-E2-A159+0.png", "TCGA-E2-A159", fg_px=20000, arm5_l=0, no_cgm_l=9000, cgm_l=100),
        _mk_rec(10, "TCGA-E2-A159+1.png", "TCGA-E2-A159", fg_px=20000, arm5_l=0, no_cgm_l=8500, cgm_l=100),
        _mk_rec(11, "TCGA-EW-A6SB+0.png", "TCGA-EW-A6SB", fg_px=20000, arm5_l=0, no_cgm_l=6000, cgm_l=100),
    ]
    sel = select_audit_sheet_indices(records, k=4, min_fg_px=1000)
    assert all(records[i]["fg_px"] >= 5000 for i in sel["heavy_back"]), f"Degenerate heavy_back: {sel['heavy_back']}"
    assert sel["divergence"][:2] == [6, 7]
    lym_pids = {records[i]["patient_id"] for i in sel["divergence"][2:]}
    assert lym_pids == {"TCGA-E2-A159", "TCGA-EW-A6SB"}

    # Blocker 4 (Sheet 09 fallback): when nec_fp has 0 FP everywhere, _select_5x5_zoom_origin falls back to multiclass (not stuck at 0,0)
    gt_roi = np.zeros((6 * 16, 6 * 16), dtype=np.uint8)
    # Put rich 4-class patch block strictly at bottom-right (r0=1, c0=1 -> pixels 16:96, 16:96)
    gt_roi[16:56, 16:56] = 0
    gt_roi[16:56, 56:96] = 1
    gt_roi[56:96, 16:56] = 2
    gt_roi[56:96, 56:96] = 3
    pred_no_nec_fp = gt_roi.copy()
    r0_fb, c0_fb = _select_5x5_zoom_origin(gt_roi, pred_no_nec_fp, 6, 6, mode="nec_fp", tile_size=16)
    assert (r0_fb, c0_fb) == (1, 1)

    # Polish 2: Sheet 07 records None (displayed as N/A) when bg_mask.sum() == 0 and nec_mask.sum() == 0
    rgb_or = np.full((32, 32, 3), (150, 80, 130), dtype=np.uint8)  # stained tissue, zero bright glass
    gt_no_nec = np.zeros((32, 32), dtype=np.uint8)
    gt_no_nec[16:, :] = 4  # stained outside-ROI, zero NEC
    pred_z = np.zeros((32, 32), dtype=np.uint8)
    s7_rep = _render_glass_guard_ablation_sheet(
        indices=[0],
        raw_rgbs=[rgb_or],
        gt_masks=gt_no_nec[None, ...],
        preds_ens4_cgm=pred_z[None, ...],
        patch_records=[_mk_rec(0, "TCGA-C1-0001+0.png", "TCGA-C1-0001", fg_px=512, bg_frac=0.0, or_frac=0.5)],
        save_path=tmp_path / "sheet07_na.png",
        dpi=72,
    )
    assert s7_rep["patches"][0]["cand1_bg_supp_pct"] is None
    assert s7_rep["patches"][0]["cand1_nec_erode_pct"] is None

    # Blocker 6 + Blocker 2 (separate author Stage-1): scrambled arm order reorders to GATE7_ARM_ORDER
    img_dir = tmp_path / "s_img"
    mask_dir = tmp_path / "s_mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    items = []
    for k_i in range(4):
        fn = f"TCGA-P{k_i}-0001+0.png"
        gt_p = np.zeros((64, 64), dtype=np.uint8)
        rgb_p = np.zeros((64, 64, 3), dtype=np.uint8)
        gt_p[0:32, 0:32] = 0
        rgb_p[0:32, 0:32] = (230, 30, 30)
        gt_p[0:32, 32:64] = 1
        rgb_p[0:32, 32:64] = (30, 220, 30)
        gt_p[32:64, 0:32] = 2
        rgb_p[32:64, 0:32] = (30, 40, 230)
        gt_p[32:64, 32:64] = 3
        rgb_p[32:64, 32:64] = (220, 220, 20)
        ip = img_dir / fn
        mp = mask_dir / fn
        Image.fromarray(rgb_p, mode="RGB").save(ip)
        _save_bcss_indexed_mask(gt_p, mp)
        items.append((ip, mp, fn))

    class _AuthorOnlyStage1(nn.Module):
        def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            b = x.shape[0]
            # Suppress classes 2 & 3 ONLY for Author Step-0 while ensemble Stage-1 keeps all 4 active!
            probs = torch.tensor([[0.95, 0.95, 0.01, 0.01]], device=x.device).repeat(b, 1)
            return torch.ones((b, 4, 4, 4), device=x.device), probs

    class _BiasedColorModel(_EquivariantColorModel):
        """Distinct per-arm logit bias so wrong arm-to-weight/beta mapping changes simplex fusion."""

        def __init__(self, bias_cls: int, strength: float) -> None:
            super().__init__("imagenet")
            self.bias_cls = bias_cls
            self.strength = strength

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            out = super().forward(x)
            seg = out["seg_logits"].clone()
            seg[:, self.bias_cls] += self.strength
            return {"seg_logits": seg, "cls_logits": out["cls_logits"]}

    arm_models = {
        "arm8_deeplab_r101_r2": _BiasedColorModel(0, 0.5),
        "arm6_segformer_b5_r2": _BiasedColorModel(1, 0.5),
        "arm7_convnext_large_r2": _BiasedColorModel(2, 0.5),
        "arm5_dinov2_large_r2": _EquivariantColorModel("imagenet"),
    }
    canonical_arms = [{"name": n, "model": arm_models[n]} for n in GATE7_ARM_ORDER]
    scrambled_arms = [
        {"name": "arm5_dinov2_large_r2", "model": arm_models["arm5_dinov2_large_r2"]},
        {"name": "arm7_convnext_large_r2", "model": arm_models["arm7_convnext_large_r2"]},
        {"name": "arm6_segformer_b5_r2", "model": arm_models["arm6_segformer_b5_r2"]},
        {"name": "arm8_deeplab_r101_r2", "model": arm_models["arm8_deeplab_r101_r2"]},
    ]
    loader = DataLoader(_SyntheticBCSSDataset(items), batch_size=2, shuffle=False, num_workers=0)
    res_canon = run_stage2_v2_visual_audit(
        arm_specs=canonical_arms,
        test_loader=loader,
        output_dir=tmp_path / "canon_out",
        zip_path=tmp_path / "canon.zip",
        device=torch.device("cpu"),
        enforce_strict_reconciliation=False,
        enforce_sha=False,
        enforce_palette_check=True,
        enforce_equivariance=False,
        enforce_shift_alignment=False,
        dpi=72,
        stage1_model_override=_DummyStage1Cls(),
        author_stage1_model_override=_AuthorOnlyStage1(),
        author_model_override=_EquivariantColorModel("zero_one"),
        whole_roi_targets=(),
    )
    res = run_stage2_v2_visual_audit(
        arm_specs=scrambled_arms,
        test_loader=loader,
        output_dir=tmp_path / "reordered_out",
        zip_path=tmp_path / "reordered.zip",
        device=torch.device("cpu"),
        enforce_strict_reconciliation=False,
        enforce_sha=False,
        enforce_palette_check=True,
        enforce_equivariance=False,
        enforce_shift_alignment=False,
        dpi=72,
        stage1_model_override=_DummyStage1Cls(),
        author_stage1_model_override=_AuthorOnlyStage1(),
        author_model_override=_EquivariantColorModel("zero_one"),
        whole_roi_targets=(),
    )
    # Arm specs MUST be canonically reordered to GATE7_ARM_ORDER with exact aligned TTAs, betas, and simplex weights!
    fp = res["frozen_protocol"]
    assert fp["arm_order"] == list(GATE7_ARM_ORDER)
    assert fp["selected_ttas"] == list(GATE7_SELECTED_TTAS)
    assert fp["full_val_betas"] == list(GATE7_FULL_VAL_BETAS)
    assert fp["simplex_weights"] == list(GATE7_SIMPLEX_WEIGHTS)
    assert res["pooled_test_metrics"]["ens4_cgm"]["mIoU_raw"] == pytest.approx(
        res_canon["pooled_test_metrics"]["ens4_cgm"]["mIoU_raw"], abs=1e-6
    )
    assert res["pooled_test_metrics"]["ens4_simplex_cgm"]["mIoU_raw"] == pytest.approx(
        res_canon["pooled_test_metrics"]["ens4_simplex_cgm"]["mIoU_raw"], abs=1e-6
    )
    # Author Step-0 was gated by _AuthorOnlyStage1 (classes 2 & 3 suppressed -> ~0.50 mIoU), while ens4_cgm kept all 4 (~0.95+)!
    assert res["pooled_test_metrics"]["ens4_cgm"]["mIoU"] > 0.90
    assert res["pooled_test_metrics"]["author_step0"]["mIoU"] < 0.60
    assert "author_stage1" in res["assertions"]["assertion_2_ckpt_sha256_and_norm"]["input_norm_contract"]


def test_visual_audit_reviewer_edge_cases_and_fixed_gt_shift(tmp_path):
    """Regression test for fixed-GT shift invariance, nested Subset unwrap, float32 defaults, zoom fg guard, and lock safety."""
    import concurrent.futures
    import time

    from torch.utils.data import Subset

    from src.stage2_v2.visual_audit import (
        GATE7_ARM_ORDER,
        GATE7_FULL_VAL_BETAS,
        GATE7_SIMPLEX_WEIGHTS,
        _collect_test_samples_from_loader,
        _select_5x5_zoom_origin,
        select_audit_sheet_indices,
    )
    from src.visualization import api_server

    # 1. Fixed-GT shift invariance: mutating GT pixels strictly outside [margin:h-margin, margin:w-margin]
    # must NOT alter any (dx, dy) shift-matrix mIoU (invariant GT valid_fg across all shifts!)
    gt_a = np.zeros((2, 64, 64), dtype=np.uint8)
    gt_a[:, 8:32, 10:34] = 1
    gt_a[:, 32:56, 20:50] = 2
    gt_a[:, 8:28, 36:56] = 3
    pred_a = gt_a.copy()

    gt_b = gt_a.copy()
    gt_b[:, :6, :] = 4   # mutate border outside margin=6
    gt_b[:, -6:, :] = 3  # mutate border outside margin=6
    gt_b[:, :, :6] = 2
    gt_b[:, :, -6:] = 4

    shifts_6 = (-6, -4, -2, -1, 0, 1, 2, 4, 6)
    rep_a = compute_translation_shift_matrix({"m": pred_a}, gt_a, shifts=shifts_6)
    rep_b = compute_translation_shift_matrix({"m": pred_a}, gt_b, shifts=shifts_6)
    assert rep_a["per_model"]["m"] == rep_b["per_model"]["m"]

    # Deliberate +5px right-shift of pred still peaks at dx=-5 (Wait: -6,-4 closest is -4 or -6 -> fails +/-4 or gap!)
    pred_shifted = np.roll(gt_a, shift=5, axis=2)
    rep_sh = compute_translation_shift_matrix({"sh": pred_shifted}, gt_a, shifts=(-6, -5, -4, -2, 0, 2, 4, 5, 6))
    assert rep_sh["per_model"]["sh"]["peak_dx"] == -5
    with pytest.raises(RuntimeError, match=r"\[STAGE2 ERROR\]"):
        assert_translation_and_orientation_alignment(rep_sh, enforce_strict=True)

    # 2. Nested Subset(Subset(ds, ...), ...) preserves filenames, mask paths & direct disk RGB loading
    img_dir = tmp_path / "nest_img"
    mask_dir = tmp_path / "nest_mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    items = []
    for k_i in range(4):
        fn = f"TCGA-N{k_i}-0001+0.png"
        rgb_p = np.full((64, 64, 3), (140, 140, 140), dtype=np.uint8)
        gt_p = np.zeros((64, 64), dtype=np.uint8)
        ip = img_dir / fn
        mp = mask_dir / fn
        Image.fromarray(rgb_p, mode="RGB").save(ip)
        _save_bcss_indexed_mask(gt_p, mp)
        items.append((ip, mp, fn))

    base_ds = _SyntheticBCSSDataset(items)
    nested_ds = Subset(Subset(base_ds, [1, 2, 3]), [0, 2])  # selects indices 1 and 3
    fns, pids, mps, rgbs, _ = _collect_test_samples_from_loader(DataLoader(nested_ds, batch_size=2, shuffle=False))
    assert fns == ["TCGA-N1-0001+0.png", "TCGA-N3-0001+0.png"]
    assert pids == ["TCGA-N1-0001", "TCGA-N3-0001"]
    assert all(p is not None and p.is_file() for p in mps)
    assert all(np.array_equal(r, np.full((64, 64, 3), 140, dtype=np.uint8)) for r in rgbs)

    # 3. _select_5x5_zoom_origin multiclass fg guard: tiny 0.6%-FG 4-class block must NOT beat rich tissue block
    # Use a 10x10 tile grid (tile_size=16) so (0,0) covers [0:80, 0:80] and (5,5) covers disjoint [80:160, 80:160]
    gt_roi = np.full((10 * 16, 10 * 16), 4, dtype=np.uint8)
    # Top-left 5x5 block (0,0) has only 40 foreground pixels (0.6% of 80x80) evenly split across 4 classes
    gt_roi[0:2, 0:5] = 0
    gt_roi[0:2, 5:10] = 1
    gt_roi[2:4, 0:5] = 2
    gt_roi[2:4, 5:10] = 3
    # Bottom-right 5x5 block (5,5) has 100% foreground evenly split across 4 quadrants;
    # without the fg_frac guard, corner (0,0) would tie at 47.64 and win on iteration order!
    gt_roi[80:120, 80:120] = 0
    gt_roi[80:120, 120:160] = 1
    gt_roi[120:160, 80:120] = 2
    gt_roi[120:160, 120:160] = 3
    r0_z, c0_z = _select_5x5_zoom_origin(gt_roi, gt_roi, 10, 10, mode="multiclass", tile_size=16)
    assert (r0_z, c0_z) == (5, 5)

    # 4. Heavy-BACK prefers distinct patients between bright-glass and outside-ROI rows (4 distinct patients total)
    def _rec(i: int, pid: str, bg: float, out_r: float) -> Dict[str, object]:
        return {
            "index": i,
            "filename": f"{pid}+{i}.png",
            "patient_id": pid,
            "fg_px": 8000,
            "gt_tum_px": 4000,
            "gt_str_px": 2000,
            "gt_lym_px": 1000,
            "gt_nec_px": 1000,
            "back_frac": bg + out_r,
            "bright_glass_frac": bg,
            "outside_roi_frac": out_r,
            "author_step0_miou": 0.70,
            "arm5_cgm_miou": 0.72,
            "ens4_no_cgm_miou": 0.72,
            "ens4_cgm_miou": 0.72,
            "arm5_cgm_nec_fp_px": 0,
            "ens4_no_cgm_nec_fp_px": 0,
            "ens4_cgm_nec_fp_px": 0,
            "arm5_cgm_lym_fp_px": 0,
            "ens4_no_cgm_lym_fp_px": 0,
            "ens4_cgm_lym_fp_px": 0,
        }

    hb_recs = [
        _rec(0, "TCGA-P1-0001", bg=0.70, out_r=0.05),
        _rec(1, "TCGA-P2-0002", bg=0.65, out_r=0.05),
        _rec(2, "TCGA-P1-0001", bg=0.05, out_r=0.80),  # same patient as bg_pick[0], higher OR
        _rec(3, "TCGA-P3-0003", bg=0.05, out_r=0.60),
        _rec(4, "TCGA-P4-0004", bg=0.05, out_r=0.55),
    ]
    sel_hb = select_audit_sheet_indices(hb_recs, k=4, min_fg_px=1000)
    hb_pids = [hb_recs[i]["patient_id"] for i in sel_hb["heavy_back"]]
    assert len(set(hb_pids)) == 4

    # 5. Float32 numpy default betas/weights on scrambled canonical arms must NOT swap arm params
    scrambled = [{"name": n, "model": _EquivariantColorModel("imagenet")} for n in reversed(GATE7_ARM_ORDER)]
    res_f32 = run_stage2_v2_visual_audit(
        arm_specs=scrambled,
        test_loader=DataLoader(base_ds, batch_size=2, shuffle=False),
        output_dir=tmp_path / "f32_out",
        zip_path=tmp_path / "f32.zip",
        device=torch.device("cpu"),
        selected_ttas=("raw_1x", "raw_1x", "raw_1x", "raw_1x"),
        full_val_betas=np.asarray(GATE7_FULL_VAL_BETAS, dtype=np.float32),
        simplex_weights=np.asarray(GATE7_SIMPLEX_WEIGHTS, dtype=np.float32),
        enforce_strict_reconciliation=False,
        enforce_sha=False,
        enforce_palette_check=True,
        enforce_equivariance=False,
        enforce_shift_alignment=False,
        dpi=72,
        stage1_model_override=_DummyStage1Cls(),
        author_model_override=_EquivariantColorModel("zero_one"),
        whole_roi_targets=(),
    )
    assert res_f32["frozen_protocol"]["arm_order"] == list(GATE7_ARM_ORDER)
    assert np.allclose(res_f32["frozen_protocol"]["full_val_betas"], list(GATE7_FULL_VAL_BETAS), atol=1e-5)
    assert np.allclose(res_f32["frozen_protocol"]["simplex_weights"], list(GATE7_SIMPLEX_WEIGHTS), atol=1e-5)

    # 6. API server thread-safety: self._lock strictly serializes s1_for_cam.forward_cam under concurrent threads
    class _LockProbeS1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0

        def forward_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.015)
            b, _, hh, ww = x.shape
            cams = torch.ones((b, 4, max(1, hh // 4), max(1, ww // 4)), device=x.device)
            probs = torch.full((b, 4), 0.9, device=x.device)
            self.active -= 1
            return cams, probs

    class _FastG7(nn.Module):
        input_norm = "imagenet"

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            b, _, hh, ww = x.shape
            lg = torch.zeros((b, 4, hh, ww), device=x.device)
            lg[:, 0] = 2.0
            return {"seg_logits": lg, "cls_logits": torch.full((b, 4), 2.0, device=x.device)}

    mgr = api_server._manager
    mgr.ensure_initialized()
    orig_g7 = dict(mgr._gate7_models)
    orig_p = dict(mgr.gate7_ckpt_paths)
    orig_s1 = mgr._gate7_stage1_cls
    probe_s1 = _LockProbeS1()
    try:
        mgr._gate7_models["arm5"] = _FastG7()
        mgr.gate7_ckpt_paths["arm5"] = Path("dummy_arm5.pth")
        mgr._gate7_stage1_cls = probe_s1
        rgb_t = np.full((64, 64, 3), [175, 95, 145], dtype=np.uint8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(api_server.process_single_patch, rgb_t, "gate7_arm5", False) for _ in range(8)]
            for fut in futs:
                fut.result()
        assert probe_s1.max_active == 1
    finally:
        mgr._gate7_models.clear()
        mgr._gate7_models.update(orig_g7)
        mgr.gate7_ckpt_paths.clear()
        mgr.gate7_ckpt_paths.update(orig_p)
        mgr._gate7_stage1_cls = orig_s1
