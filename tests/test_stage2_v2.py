"""Comprehensive unit and 4-arm E2E dry-run tests for src/stage2_v2/ and 02b notebook."""

import ast
import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.ops import StochasticDepth

from src.models.wsss_inference_methods import DeepLabV3Plus, WSSSTissueStage1NetCAM
from src.stage2_v2 import (
    STAGE2_V2_PRESETS,
    BCSSSoftMemmapDataset,
    ConvNeXtFPNDualHead,
    DeepLabV3PlusDualHead,
    DINOv2AdapterDualHead,
    KLSoftOEEMLoss,
    MILPatchPresenceLoss,
    SegFormerDualHead,
    SoftCrossEntropyLoss,
    SoftMulticlassDiceLoss,
    SoftSymmetricCELoss,
    Stage2ModelEMA,
    build_stage2_v2_model,
    generate_bn7_soft_memmap,
    generate_round2_ema_memmap,
    get_stage2_v2_preset,
    load_stage2_v2_checkpoint,
    resolve_stage2_v2_config_from_env,
    run_step0_calibration_and_author_anchor,
    train_stage2_v2,
)
from src.stage2_v2 import trainer as s2_trainer
from src.stage2_v2.evaluator import _eval_author_stage2_split
from src.stage2_v2.memmap_dataset import _cams_to_soft_probs
from src.training.dataset_loader import BCSSStage2ValDataset, load_bcss_patient_splits
from src.training.train_stage2 import prepare_stage1_input


def test_stage2_v2_presets_and_env_resolution() -> None:
    expected = {
        "arm1_deeplab_cleankd",
        "arm2_segformer_oeem",
        "arm3_convnext_selftrain",
        "arm4_dinov2_adapter",
    }
    assert set(STAGE2_V2_PRESETS.keys()) == expected
    cfg_default = get_stage2_v2_preset("arm1_deeplab_cleankd")
    assert cfg_default.optimizer == "sgd" and cfg_default.scheduler == "poly"
    assert cfg_default.to_dict()["eval_scales"] == [0.75, 1.0, 1.25]

    cfg_env = resolve_stage2_v2_config_from_env(
        environ={
            "STAGE2_EXP_PRESET": "arm4_dinov2_adapter",
            "STAGE2_SEED": "123",
            "STAGE2_EPOCHS": "3",
            "STAGE2_BATCH_SIZE": "8",
            "STAGE2_NUM_WORKERS": "0",
            "STAGE2_SOFT_MODE": "raw_l1",
            "STAGE2_SOFT_TEMP": "0.40",
            "STAGE2_DRY_RUN": "1",
        }
    )
    assert cfg_env.preset_name == "arm4_dinov2_adapter"
    assert cfg_env.unfreeze_last_n_blocks == 4
    assert cfg_env.seed == 123 and cfg_env.epochs == 3 and cfg_env.batch_size == 8
    assert cfg_env.num_workers == 0 and cfg_env.soft_mode == "raw_l1"
    assert cfg_env.soft_temp == pytest.approx(0.40) and cfg_env.dry_run is True

    cfg_arm = resolve_stage2_v2_config_from_env(environ={"STAGE2_ARM": "arm3_convnext_selftrain"})
    assert cfg_arm.preset_name == "arm3_convnext_selftrain"
    assert cfg_arm.self_train_round2_start_epoch == 10

    with pytest.raises(ValueError):
        get_stage2_v2_preset("nonexistent_arm")
    with pytest.raises(ValueError):
        resolve_stage2_v2_config_from_env(environ={"STAGE2_SOFT_MODE": "bad_mode"})
    with pytest.raises(ValueError):
        resolve_stage2_v2_config_from_env(environ={"STAGE2_SOFT_TEMP": "-0.1"})


@pytest.mark.parametrize(
    "preset_name",
    [
        "arm1_deeplab_cleankd",
        "arm2_segformer_oeem",
        "arm3_convnext_selftrain",
        "arm4_dinov2_adapter",
    ],
)
@pytest.mark.parametrize("hw", [168, 224, 280])
def test_all_4_arms_forward_backward_multires_and_b1_eval(preset_name: str, hw: int) -> None:
    cfg = get_stage2_v2_preset(preset_name)
    model = build_stage2_v2_model(cfg, pretrained=False)
    model.train()
    x = torch.randn(2, 3, hw, hw)
    out = model(x)
    assert set(out.keys()) == {"seg_logits", "cls_logits"}
    assert out["seg_logits"].shape == (2, 4, hw, hw)
    assert out["cls_logits"].shape == (2, 4)

    loss = out["seg_logits"].mean() + out["cls_logits"].mean()
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)

    model.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 3, hw, hw)
        out1 = model(x1)
        lg1 = model.forward_logits(x1)
        assert out1["seg_logits"].shape == (1, 4, hw, hw)
        assert out1["cls_logits"].shape == (1, 4)
        assert lg1.shape == (1, 4, hw, hw)


def test_convnext_stochastic_depth_and_dinov2_pad14_and_freeze() -> None:
    cnx = ConvNeXtFPNDualHead(num_classes=4, pretrained=False)
    assert cnx.stochastic_depth_prob == pytest.approx(0.2)
    sd_probs = [m.p for m in cnx.modules() if isinstance(m, StochasticDepth)]
    assert len(sd_probs) > 0 and max(sd_probs) == pytest.approx(0.2)

    dino = DINOv2AdapterDualHead(num_classes=4, pretrained=False, unfreeze_last_n_blocks=2)
    assert len(dino.stage_norms) == 4
    assert all(isinstance(ln, torch.nn.LayerNorm) for ln in dino.stage_norms)
    assert not any(p.requires_grad for p in dino.backbone.embeddings.parameters())
    assert not any(p.requires_grad for p in dino.backbone.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in dino.backbone.encoder.layer[-1].parameters())
    assert all(p.requires_grad for p in dino.backbone.layernorm.parameters())

    dino.eval()
    with torch.no_grad():
        x_225 = torch.randn(1, 3, 225, 225)
        out_225 = dino(x_225)
        assert out_225["seg_logits"].shape == (1, 4, 225, 225)
        assert out_225["cls_logits"].shape == (1, 4)


def test_stage2_model_ema_clean_export_no_bloat() -> None:
    model = DeepLabV3PlusDualHead(num_classes=4, pretrained=False)
    ema = Stage2ModelEMA(model, decay=0.9)
    with torch.no_grad():
        for p in model.cls_head.parameters():
            p.add_(1.0)
    ema.update(model)
    clean_sd = ema.clean_model_state_dict()
    assert set(clean_sd.keys()) == set(model.state_dict().keys())
    before = model.cls_head.weight.detach().clone()
    with ema.swap_to_ema(model):
        assert not torch.allclose(model.cls_head.weight, before)
    assert torch.allclose(model.cls_head.weight, before)


def test_losses_oeem_zero_bias_mil_sce() -> None:
    oeem = KLSoftOEEMLoss(tau=1.5, oeem_warmup_epochs=2)
    q = torch.tensor([[[[0.7, 0.2], [0.1, 0.8]], [[0.2, 0.7], [0.8, 0.1]], [[0.05, 0.05], [0.05, 0.05]], [[0.05, 0.05], [0.05, 0.05]]]], dtype=torch.float32)
    logits_exact = torch.log(q)

    # Multi-class patch with p == q -> KL == 0 and w == 1.0 even after warmup
    y_multi = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    kl_eq, w_eq = oeem.compute_kl_and_weights(logits_exact, q, y_gt=y_multi, epoch=5)
    assert torch.allclose(kl_eq, torch.zeros_like(kl_eq), atol=1e-5)
    assert torch.allclose(w_eq, torch.ones_like(w_eq), atol=1e-5)

    # Single-class patch with p != q -> w == 1.0 strictly even after warmup
    logits_off = torch.randn(1, 4, 8, 8, requires_grad=True)
    q_1cls = torch.zeros(1, 4, 8, 8)
    q_1cls[:, 0] = 1.0
    y_single = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _, w_single = oeem.compute_kl_and_weights(logits_off, q_1cls, y_gt=y_single, epoch=5)
    assert torch.allclose(w_single, torch.ones_like(w_single))

    # Multi-class during warmup -> w == 1.0
    _, w_warm = oeem.compute_kl_and_weights(logits_off, q_1cls, y_gt=y_multi, epoch=1)
    assert torch.allclose(w_warm, torch.ones_like(w_warm))

    # Multi-class after warmup -> weights in [0.25, 2.0] and differentiable loss
    loss_oeem = oeem(logits_off, q_1cls, y_gt=y_multi, epoch=3)
    assert torch.isfinite(loss_oeem)
    loss_oeem.backward()
    assert logits_off.grad is not None and torch.isfinite(logits_off.grad).all()

    # MIL B=1 + SCE + SoftCE + Dice
    mil = MILPatchPresenceLoss()
    for y_vec in ([[1.0, 0.0, 1.0, 0.0]], [[1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0]]):
        lg = torch.randn(1, 4, 16, 16, requires_grad=True)
        l_m = mil(lg, torch.tensor(y_vec))
        assert torch.isfinite(l_m)
        l_m.backward()

    sce = SoftSymmetricCELoss(alpha=1.0, beta=0.15)
    lg_s = torch.randn(2, 4, 15, 17, requires_grad=True)
    q_s = torch.softmax(torch.randn(2, 4, 15, 17), dim=1)
    l_sce = sce(lg_s, q_s)
    assert torch.isfinite(l_sce)
    l_sce.backward()
    assert torch.isfinite(SoftCrossEntropyLoss()(lg_s.detach(), q_s))
    assert torch.isfinite(SoftMulticlassDiceLoss()(lg_s.detach(), q_s))


def _make_synthetic_bcss_cohort(root: Path, n_train: int = 64, n_val: int = 16, n_test: int = 16):
    tr_dir = root / "training"
    val_img = root / "val" / "img"
    val_msk = root / "val" / "mask"
    te_img = root / "test" / "img"
    te_msk = root / "test" / "mask"
    for d in (tr_dir, val_img, val_msk, te_img, te_msk):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    train_pids = ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003", "TCGA-AA-0004"]
    val_pids = ["TCGA-BB-0001", "TCGA-BB-0002"]
    test_pids = ["TCGA-CC-0001", "TCGA-CC-0002"]

    for i in range(n_train):
        pid = train_pids[i % len(train_pids)]
        if i % 3 == 0:
            lbl = [1, 0, 0, 0]
            arr = rng.integers(235, 255, size=(224, 224, 3), dtype=np.uint8)
            arr[:112, :112] = rng.integers(80, 180, size=(112, 112, 3), dtype=np.uint8)
        else:
            lbl = [1, 1, 0, 0] if i % 2 == 0 else [0, 1, 1, 1]
            arr = rng.integers(40, 220, size=(224, 224, 3), dtype=np.uint8)
        fname = f"{pid}_x{i * 224}_y0_[{lbl[0]}{lbl[1]}{lbl[2]}{lbl[3]}].png"
        Image.fromarray(arr, mode="RGB").save(tr_dir / fname)

    for split_pids, idir, mdir, count in ((val_pids, val_img, val_msk, n_val), (test_pids, te_img, te_msk, n_test)):
        for i in range(count):
            pid = split_pids[i % len(split_pids)]
            arr = rng.integers(40, 220, size=(224, 224, 3), dtype=np.uint8)
            msk = np.zeros((224, 224), dtype=np.uint8)
            msk[56:112, :] = 1
            msk[112:168, :] = 2
            msk[168:200, :] = 3
            msk[200:215, :] = 4
            msk[215:, :] = 255
            fname = f"{pid}_x{i}_y0_[1111].png"
            Image.fromarray(arr, mode="RGB").save(idir / fname)
            p_msk = Image.fromarray(msk, mode="P")
            p_msk.save(mdir / fname)

    return tr_dir, val_img, val_msk, te_img, te_msk, train_pids, val_pids, test_pids


def test_memmap_dataset_spawn_pickling_antileakage_and_d4(tmp_path: Path) -> None:
    tr_dir, *_ = _make_synthetic_bcss_cohort(tmp_path / "bcss", n_train=8, n_val=2, n_test=2)
    s1 = WSSSTissueStage1NetCAM(n_class=4).eval()
    mm_dir = tmp_path / "mm"
    dat_p, man_p, man = generate_bn7_soft_memmap(
        stage1_model=s1,
        train_img_dir=tr_dir,
        output_dir=mm_dir,
        patient_ids=["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003", "TCGA-AA-0004"],
        stage1_sha256="abc123",
        soft_mode="softmax_t",
        soft_temp=0.25,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert dat_p.is_file() and man["num_patches"] == 8 and man["single_label_short_circuit"] >= 2

    # Cache hit on identical args
    dat_p2, _, _ = generate_bn7_soft_memmap(
        stage1_model=s1,
        train_img_dir=tr_dir,
        output_dir=mm_dir,
        patient_ids=["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003", "TCGA-AA-0004"],
        stage1_sha256="abc123",
        soft_mode="softmax_t",
        soft_temp=0.25,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert dat_p2 == dat_p

    # Guard cam_sum <= 1e-6 fallback
    zero_cam = torch.zeros(1, 4, 16, 16)
    y_2 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    for mode in ("raw_l1", "softmax_t", "sg_dap_soft"):
        q_fb = _cams_to_soft_probs(zero_cam, y_2, torch.rand(1, 3, 16, 16), mode, 0.25)
        assert torch.allclose(q_fb.sum(dim=1), torch.ones(1, 16, 16), atol=1e-4)

    with pytest.raises(RuntimeError, match="ANTI-LEAKAGE"):
        BCSSSoftMemmapDataset(tr_dir, dat_p, man_p, patient_ids=["TCGA-ZZ-9999"], strict_patient_filter=True)

    ds = BCSSSoftMemmapDataset(tr_dir, dat_p, man_p, is_train=True)
    assert ds._memmap is None
    unpickled = pickle.loads(pickle.dumps(ds))
    assert unpickled._memmap is None
    img_t, q_t, y_t, pw_t = unpickled[0]
    assert img_t.shape == (3, 224, 224) and q_t.shape == (4, 224, 224)
    assert y_t.shape == (4,) and pw_t.shape == (224, 224)
    # Single-class patch 0 has white glass region weighted 0.25 and tissue weighted 1.0
    assert float(pw_t.min()) == pytest.approx(0.25) and float(pw_t.max()) == pytest.approx(1.0)

    # Round-2 cannot overwrite s1 memmap
    model_r2 = DeepLabV3PlusDualHead(num_classes=4, pretrained=False).eval()
    with pytest.raises(ValueError, match="differ"):
        generate_round2_ema_memmap(model_r2, tr_dir, dat_p, man_p, dat_p, man_p, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def e2e_64_patch_bundle(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("e2e_64_bundle")
    cohort_dir = root / "bcss"
    tr_dir, val_img, val_msk, te_img, te_msk, tr_pids, val_pids, te_pids = _make_synthetic_bcss_cohort(
        cohort_dir, n_train=64, n_val=16, n_test=16
    )
    s1 = WSSSTissueStage1NetCAM(n_class=4).eval()
    mm_dir = root / "mm"
    dat_p, man_p, _ = generate_bn7_soft_memmap(
        stage1_model=s1,
        train_img_dir=tr_dir,
        output_dir=mm_dir,
        patient_ids=tr_pids,
        stage1_sha256="sha_test",
        soft_mode="softmax_t",
        soft_temp=0.25,
        batch_size=16,
        device=torch.device("cpu"),
    )
    return tr_dir, val_img, val_msk, te_img, te_msk, tr_pids, val_pids, te_pids, s1, dat_p, man_p


@pytest.mark.parametrize(
    "preset_name",
    [
        "arm1_deeplab_cleankd",
        "arm2_segformer_oeem",
        "arm3_convnext_selftrain",
        "arm4_dinov2_adapter",
    ],
)
def test_e2e_64_patch_dry_run_all_4_arms(tmp_path: Path, e2e_64_patch_bundle, preset_name: str) -> None:
    tr_dir, val_img, val_msk, te_img, te_msk, tr_pids, val_pids, te_pids, s1, src_dat, src_man = e2e_64_patch_bundle
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir(parents=True, exist_ok=True)
    dat_p = mm_dir / src_dat.name
    man_p = mm_dir / src_man.name
    shutil.copy2(src_dat, dat_p)
    shutil.copy2(src_man, man_p)

    val_ds = BCSSStage2ValDataset(val_img, val_msk, patient_ids=val_pids, strict_patient_filter=True)
    test_ds = BCSSStage2ValDataset(te_img, te_msk, patient_ids=te_pids, strict_patient_filter=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    base_cfg = get_stage2_v2_preset(preset_name)
    r2_ep = 2 if preset_name == "arm3_convnext_selftrain" else None
    ep_count = 2 if preset_name == "arm3_convnext_selftrain" else 1
    cfg = replace(
        base_cfg,
        epochs=ep_count,
        batch_size=8,
        num_workers=0,
        dry_run=True,
        self_train_round2_start_epoch=r2_ep,
    )
    model = build_stage2_v2_model(cfg, pretrained=False)
    ckpt_dir = tmp_path / f"ckpts_{preset_name}"
    out_dir = tmp_path / f"res_{preset_name}"
    zip_p = tmp_path / f"res_{preset_name}.zip"

    metrics = train_stage2_v2(
        cfg=cfg,
        model=model,
        train_img_dir=tr_dir,
        memmap_path=dat_p,
        manifest_path=man_p,
        val_loader=val_loader,
        test_loader=test_loader,
        stage1_model=s1,
        stage1_cls_model=s1,
        stage1_norm_mode="zero_one",
        stage1_sha256="sha_test",
        train_patient_ids=tr_pids,
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
        zip_path=zip_p,
        device=torch.device("cpu"),
        strict_patient_filter=True,
    )

    assert (ckpt_dir / "stage2_retrained_best.pth").is_file()
    assert (ckpt_dir / "stage2_latest.pth").is_file()
    assert (out_dir / "STAGE2_TRAINING_METRICS.json").is_file()
    assert (out_dir / "stage2_training_loss.png").is_file()
    assert zip_p.is_file()
    if preset_name == "arm3_convnext_selftrain":
        assert (mm_dir / "pm_soft_r2.dat").is_file()

    reloaded, meta = load_stage2_v2_checkpoint(ckpt_dir / "stage2_retrained_best.pth", strict=True)
    assert meta["arch"] == cfg.arch and meta["non_finite_tensors"] == 0
    assert reloaded.forward_logits(torch.randn(1, 3, 224, 224)).shape == (1, 4, 224, 224)
    assert "val_eval_calibrated" in metrics and "test_eval_calibrated" in metrics
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    zip_p.unlink(missing_ok=True)
    shutil.rmtree(mm_dir, ignore_errors=True)


def test_step0_calibration_and_author_anchor_smoke(tmp_path: Path) -> None:
    cohort_dir = tmp_path / "bcss0"
    _, val_img, val_msk, te_img, te_msk, _, val_pids, te_pids = _make_synthetic_bcss_cohort(
        cohort_dir, n_train=4, n_val=8, n_test=8
    )
    val_loader = DataLoader(BCSSStage2ValDataset(val_img, val_msk, patient_ids=val_pids), batch_size=4)
    test_loader = DataLoader(BCSSStage2ValDataset(te_img, te_msk, patient_ids=te_pids), batch_size=4)

    author_m = DeepLabV3Plus(num_classes=4, output_stride=16)
    author_ckpt = tmp_path / "stage2_checkpoint_trained_on_bcss.pth"
    torch.save({"state_dict": author_m.state_dict()}, author_ckpt)

    s1 = WSSSTissueStage1NetCAM(n_class=4).eval()
    out_json = tmp_path / "step0_metrics.json"
    rep = run_step0_calibration_and_author_anchor(
        stage1_model=s1,
        val_loader=val_loader,
        test_loader=test_loader,
        author_stage2_ckpt=author_ckpt,
        stage1_norm_mode="zero_one",
        device=torch.device("cpu"),
        output_json_path=out_json,
    )
    assert out_json.is_file()
    assert "val" in rep["author_stage2_anchor"] and "test" in rep["author_stage2_anchor"]
    assert {"raw_l1", "softmax_t_0.15", "softmax_t_0.25", "softmax_t_0.40", "sg_dap_soft"} == set(
        rep["bn7_soft_recipe_sweep_val"].keys()
    )


def test_notebook_02b_json_ast_device_and_ascii_dashes() -> None:
    nb_path = Path("notebooks/02b_Stage2_Breakthrough_G4.ipynb")
    assert nb_path.is_file()
    raw = nb_path.read_text(encoding="utf-8")
    for bad in ("\u2014", "\u2013", "\u2015", "\u2212"):
        assert bad not in raw
    nb = json.loads(raw)
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] == 5
    cells = nb["cells"]
    assert len(cells) >= 8

    for idx, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            ast.parse(code, filename=f"02b_cell_{idx}.py")

    setup_code = "".join("".join(c["source"]) + "\n" for c in cells[:6] if c["cell_type"] == "code")
    lines = setup_code.splitlines()
    dev_idx = next((i for i, ln in enumerate(lines) if "DEVICE =" in ln and not ln.strip().startswith("#")), None)
    use_idx = next((i for i, ln in enumerate(lines) if ("{DEVICE}" in ln or "to(DEVICE)" in ln) and not ln.strip().startswith("#")), None)
    assert dev_idx is not None
    if use_idx is not None:
        assert dev_idx < use_idx


def test_notebook_02b_cell2_train_val_test_resolution_synthetic_and_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    monkeypatch.setattr(
        "urllib.request.urlretrieve",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Unexpected network download in test")),
    )
    nb = json.loads(Path("notebooks/02b_Stage2_Breakthrough_G4.ipynb").read_text(encoding="utf-8"))
    cell2_code = "".join(
        next(c for c in nb["cells"] if c.get("id") == "cell-03-dataset")["source"]
    )

    # 1) Synthetic flat-training/*.png + val/img + test/img layout using real cohort patient IDs
    splits = load_bcss_patient_splits()
    tr_pids = splits["train"][:3]
    val_pids = splits["val"][:2]
    te_pids = splits["test"][:2]
    synth_root = tmp_path / "BCSS-WSSS"
    (synth_root / "training").mkdir(parents=True, exist_ok=True)
    for d in (synth_root / "val" / "img", synth_root / "val" / "mask", synth_root / "test" / "img", synth_root / "test" / "mask"):
        d.mkdir(parents=True, exist_ok=True)
    # Also create an excluded sample_dataset directory with >= 10 PNGs to verify exclusion
    (synth_root / "sample_dataset" / "training").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(12):
        arr = rng.integers(30, 220, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(
            synth_root / "training" / f"{tr_pids[i % len(tr_pids)]}_x{i}_y0_[1100].png"
        )
        Image.fromarray(arr, mode="RGB").save(
            synth_root / "sample_dataset" / "training" / f"{tr_pids[i % len(tr_pids)]}_x{i}_y0_[1100].png"
        )
    for pids, idir, mdir in ((val_pids, synth_root / "val" / "img", synth_root / "val" / "mask"), (te_pids, synth_root / "test" / "img", synth_root / "test" / "mask")):
        for i, pid in enumerate(pids):
            arr = rng.integers(30, 220, size=(64, 64, 3), dtype=np.uint8)
            msk = np.zeros((64, 64), dtype=np.uint8)
            msk[:32, :] = 1
            Image.fromarray(arr, mode="RGB").save(idir / f"{pid}_x{i}_y0_[1100].png")
            Image.fromarray(msk, mode="P").save(mdir / f"{pid}_x{i}_y0_[1100].png")

    synth_code = cell2_code.replace('Path("./data/BCSS-WSSS")', f'Path(r"{synth_root}")').replace(
        'Path("./data/BCSS-WSSS.zip")', f'Path(r"{tmp_path / "BCSS-WSSS.zip"}")'
    )
    ns_synth = {
        "IN_COLAB": False,
        "os": __import__("os"),
        "Path": Path,
        "subprocess": __import__("subprocess"),
        "cfg": replace(get_stage2_v2_preset("arm1_deeplab_cleankd"), dry_run=True, num_workers=0),
        "SEED": 42,
        "DEVICE": torch.device("cpu"),
    }
    exec(synth_code, ns_synth)
    assert Path(ns_synth["TRAIN_DIR"]).resolve() == (synth_root / "training").resolve()
    assert Path(ns_synth["VAL_IMG_DIR"]).resolve() == (synth_root / "val" / "img").resolve()
    assert Path(ns_synth["TEST_IMG_DIR"]).resolve() == (synth_root / "test" / "img").resolve()
    assert len(ns_synth["val_ds"]) == len(val_pids)
    assert len(ns_synth["test_ds"]) == len(te_pids)

    # Verify helper excludes sample_dataset and returns None when isolated with no data
    empty_dir = tmp_path / "empty_bcss"
    (empty_dir / "sample_dataset" / "training").mkdir(parents=True, exist_ok=True)
    for i in range(12):
        Image.new("RGB", (16, 16)).save(
            empty_dir / "sample_dataset" / "training" / f"{tr_pids[i % len(tr_pids)]}_x{i}_y0_[1100].png"
        )
    assert ns_synth["_resolve_bcss_train_dir"](empty_dir, extra_roots=()) is None
    empty_zip = tmp_path / "empty_dataset.zip"
    with zipfile.ZipFile(empty_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("placeholder.txt", b"0" * 1_000_005)
    with pytest.raises(FileNotFoundError):
        bad_code = cell2_code.replace('Path("./data/BCSS-WSSS")', f'Path(r"{empty_dir}")').replace(
            'Path("./data/BCSS-WSSS.zip")', f'Path(r"{empty_zip}")'
        ).replace(
            "_resolve_bcss_train_dir(DATASET_DIR, min_images=10, min_patients=2)",
            "_resolve_bcss_train_dir(DATASET_DIR, min_images=10, min_patients=2, extra_roots=())",
        )
        ns_err = dict(ns_synth)
        exec(bad_code, ns_err)

    # 2) Real data/BCSS-WSSS/ layout if present in workspace
    real_train = Path("data/BCSS-WSSS/training")
    if real_train.is_dir():
        ns_real = {
            "IN_COLAB": False,
            "os": __import__("os"),
            "Path": Path,
            "subprocess": __import__("subprocess"),
            "cfg": replace(get_stage2_v2_preset("arm1_deeplab_cleankd"), dry_run=True, num_workers=0),
            "SEED": 42,
            "DEVICE": torch.device("cpu"),
        }
        exec(cell2_code, ns_real)
        assert Path(ns_real["TRAIN_DIR"]).resolve() == real_train.resolve()
        assert Path(ns_real["VAL_IMG_DIR"]).resolve() == Path("data/BCSS-WSSS/val/img").resolve()
        assert Path(ns_real["TEST_IMG_DIR"]).resolve() == Path("data/BCSS-WSSS/test/img").resolve()
        assert len(ns_real["val_ds"]) > 0 and len(ns_real["test_ds"]) > 0


def test_author_stage2_eval_receives_zero_one_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    val_img = tmp_path / "val" / "img"
    val_msk = tmp_path / "val" / "mask"
    val_img.mkdir(parents=True, exist_ok=True)
    val_msk.mkdir(parents=True, exist_ok=True)

    # Build synthetic RGB patches spanning full [0, 255] dynamic range
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:75, :, 0] = 0
    arr[75:150, :, 1] = 128
    arr[150:, :, 2] = 255
    msk = np.zeros((224, 224), dtype=np.uint8)
    msk[112:, :] = 1
    for i, pid in enumerate(["TCGA-BB-0001", "TCGA-BB-0002"]):
        Image.fromarray(arr, mode="RGB").save(val_img / f"{pid}_x{i}_y0_[1100].png")
        Image.fromarray(msk, mode="P").save(val_msk / f"{pid}_x{i}_y0_[1100].png")

    val_loader = DataLoader(
        BCSSStage2ValDataset(val_img, val_msk, patient_ids=["TCGA-BB-0001", "TCGA-BB-0002"]),
        batch_size=2,
        shuffle=False,
    )
    raw_imgs, _ = next(iter(val_loader))
    assert float(raw_imgs.min()) < -1.0 and float(raw_imgs.max()) > 1.0
    expected_01 = prepare_stage1_input(raw_imgs, stage1_norm_mode="zero_one", images_are_imagenet_norm=True)

    author_m = DeepLabV3Plus(num_classes=4, output_stride=16).eval()
    recorded_inputs = []
    hook = author_m.register_forward_pre_hook(
        lambda _mod, args: recorded_inputs.append(args[0].detach().cpu().clone())
    )
    try:
        s1 = WSSSTissueStage1NetCAM(n_class=4).eval()
        _eval_author_stage2_split(
            author_model=author_m,
            stage1_model=s1,
            loader=val_loader,
            device=torch.device("cpu"),
            stage1_norm_mode="imagenet",
        )
    finally:
        hook.remove()

    assert len(recorded_inputs) == 6  # 1x raw + 5 non-identity views in 6-view MS-TTA
    assert torch.allclose(recorded_inputs[0], expected_01, atol=1e-5)
    for x_in in recorded_inputs:
        assert float(x_in.min()) >= -1e-6 and float(x_in.max()) <= 1.0 + 1e-6

    # Verify via run_step0_calibration_and_author_anchor as well
    author_ckpt = tmp_path / "stage2_checkpoint_trained_on_bcss.pth"
    torch.save({"state_dict": author_m.state_dict()}, author_ckpt)
    step0_inputs = []
    hook2 = author_m.register_forward_pre_hook(
        lambda _mod, args: step0_inputs.append(args[0].detach().cpu().clone())
    )
    monkeypatch.setattr(
        "src.stage2_v2.evaluator.build_and_load_deeplabv3plus",
        lambda checkpoint_path, device=None, strict=True: author_m,
    )
    try:
        rep = run_step0_calibration_and_author_anchor(
            stage1_model=s1,
            val_loader=val_loader,
            test_loader=None,
            author_stage2_ckpt=author_ckpt,
            stage1_norm_mode="imagenet",
            device=torch.device("cpu"),
            output_json_path=tmp_path / "step0.json",
        )
    finally:
        hook2.remove()
    assert "val" in rep["author_stage2_anchor"]
    assert len(step0_inputs) == 6
    assert torch.allclose(step0_inputs[0], expected_01, atol=1e-5)


def test_arm3_round2_disconnect_resume_regenerates_and_rebinds_r2_memmap(
    tmp_path: Path, e2e_64_patch_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    tr_dir, val_img, val_msk, te_img, te_msk, tr_pids, val_pids, te_pids, s1, src_dat, src_man = e2e_64_patch_bundle
    mm_dir = tmp_path / "mm_r2_resume"
    mm_dir.mkdir(parents=True, exist_ok=True)
    dat_p = mm_dir / src_dat.name
    man_p = mm_dir / src_man.name
    shutil.copy2(src_dat, dat_p)
    shutil.copy2(src_man, man_p)

    val_loader = DataLoader(BCSSStage2ValDataset(val_img, val_msk, patient_ids=val_pids), batch_size=8, shuffle=False)
    test_loader = DataLoader(BCSSStage2ValDataset(te_img, te_msk, patient_ids=te_pids), batch_size=8, shuffle=False)

    base_cfg = get_stage2_v2_preset("arm3_convnext_selftrain")
    cfg_p1 = replace(base_cfg, epochs=2, batch_size=8, num_workers=0, dry_run=True, self_train_round2_start_epoch=2)
    ckpt_dir = tmp_path / "ckpts_r2"
    out_dir = tmp_path / "out_r2"

    m1 = build_stage2_v2_model(cfg_p1, pretrained=False)
    train_stage2_v2(
        cfg=cfg_p1,
        model=m1,
        train_img_dir=tr_dir,
        memmap_path=dat_p,
        manifest_path=man_p,
        val_loader=val_loader,
        test_loader=test_loader,
        stage1_model=s1,
        stage1_norm_mode="zero_one",
        stage1_sha256="sha_r2_test",
        train_patient_ids=tr_pids,
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
        zip_path=tmp_path / "r2_p1.zip",
        device=torch.device("cpu"),
    )
    r2_dat = mm_dir / "pm_soft_r2.dat"
    r2_man = mm_dir / "pm_soft_r2_manifest.json"
    latest_ckpt = ckpt_dir / "stage2_latest.pth"
    assert r2_dat.is_file() and r2_man.is_file() and latest_ckpt.is_file()
    ckpt_obj = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    assert ckpt_obj["epoch"] == 2 and ckpt_obj["round2_active"] is True

    # Simulate fresh VM disconnect: delete Round-2 memmap + manifest, keeping only pm_soft_s1.dat + stage2_latest.pth
    r2_dat.unlink()
    r2_man.unlink()
    assert not r2_dat.exists()

    built_loaders = []
    orig_build = s2_trainer._build_train_loader

    def _spy_build(*args, **kwargs):
        dl = orig_build(*args, **kwargs)
        built_loaders.append(dl)
        return dl

    monkeypatch.setattr(s2_trainer, "_build_train_loader", _spy_build)

    cfg_p2 = replace(base_cfg, epochs=3, batch_size=8, num_workers=0, dry_run=True, self_train_round2_start_epoch=2)
    m2 = build_stage2_v2_model(cfg_p2, pretrained=False)
    metrics_p2 = train_stage2_v2(
        cfg=cfg_p2,
        model=m2,
        train_img_dir=tr_dir,
        memmap_path=dat_p,
        manifest_path=man_p,
        val_loader=val_loader,
        test_loader=test_loader,
        stage1_model=s1,
        stage1_norm_mode="zero_one",
        stage1_sha256="sha_r2_test",
        train_patient_ids=tr_pids,
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
        zip_path=tmp_path / "r2_p2.zip",
        device=torch.device("cpu"),
        resume_from=latest_ckpt,
    )
    assert r2_dat.is_file() and r2_man.is_file()
    assert len(built_loaders) >= 2
    assert built_loaders[0].dataset.memmap_path.name == "pm_soft_s1.dat"
    assert built_loaders[-1].dataset.memmap_path.name == "pm_soft_r2.dat"
    assert metrics_p2["history"]["epoch"] == [1, 2, 3]
    shutil.rmtree(mm_dir, ignore_errors=True)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)


def test_soft_multiclass_dice_loss_linear_valid_weight_and_gradient_direction() -> None:
    dice_fn = SoftMulticlassDiceLoss(eps=1e-5)
    # 1) Exact match p == q with valid_weight = 0.25 must achieve loss == 0.0 (within 1e-5)
    q_exact = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    q_exact[0, 0, 0, 0] = 1.0
    q_exact[0, 1, 0, 1] = 1.0
    q_exact[0, 2, 1, 0] = 1.0
    q_exact[0, 3, 1, 1] = 1.0
    logits_exact = torch.where(q_exact > 0.5, torch.full_like(q_exact, 20.0), torch.full_like(q_exact, -20.0))
    vw = torch.full((1, 2, 2), 0.25, dtype=torch.float32)
    loss_exact = dice_fn(logits_exact, q_exact, valid_weight=vw)
    assert float(loss_exact.item()) == pytest.approx(0.0, abs=1e-5)

    # 2) With q_0 = 1.0 at (0,0), vw = 0.25, and batch Dice > 0.50, class-0 logit grad must be negative
    logits = (2.0 * q_exact).detach().clone().requires_grad_(True)
    loss = dice_fn(logits, q_exact, valid_weight=vw)
    batch_dice = 1.0 - float(loss.item())
    assert batch_dice > 0.50
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 0, 0, 0].item()) < 0.0


def test_cgm_author_mul0_diverges_from_cgm_neg1e4_on_negative_active_logits() -> None:
    class _DummyAuthorModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, h, w = x.shape
            logits = torch.full((b, 4, h, w), -2.0, dtype=torch.float32, device=x.device)
            # Active class 0 has negative logit (-0.5), which stays < 0.0 of zeroed inactive classes
            logits[:, 0] = -0.5
            return logits

    class _DummyStage1Model(torch.nn.Module):
        def forward_cam(self, x: torch.Tensor):
            b, _, h, w = x.shape
            cams = torch.zeros((b, 4, max(1, h // 16), max(1, w // 16)), dtype=torch.float32, device=x.device)
            probs = torch.tensor([[0.90, 0.01, 0.01, 0.01]], dtype=torch.float32, device=x.device).repeat(b, 1)
            return cams, probs

    imgs = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    gts = torch.zeros((1, 16, 16), dtype=torch.long)
    loader = DataLoader(list(zip(imgs, gts)), batch_size=1, shuffle=False)

    res = _eval_author_stage2_split(
        author_model=_DummyAuthorModel().eval(),
        stage1_model=_DummyStage1Model().eval(),
        loader=loader,
        device=torch.device("cpu"),
        stage1_norm_mode="zero_one",
        scales=(1.0,),
    )
    assert res["cgm_neg1e4"]["mIoU"] == pytest.approx(1.0, abs=1e-4)
    assert res["cgm_author_mul0"]["mIoU"] == pytest.approx(0.0, abs=1e-4)
    assert res["cgm_author_mul0"]["mIoU"] != res["cgm_neg1e4"]["mIoU"]
