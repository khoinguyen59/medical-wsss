"""Gate-7 Track A + Addendum unit and integration tests (CPU-fast, zero-leakage)."""

import ast
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Dinov2Config, Dinov2Model, SegformerConfig, SegformerForSemanticSegmentation

from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.stage2_v2 import (
    DEFAULT_ARM_FAMILIES,
    STAGE2_V2_PRESETS,
    VALID_MEMMAP_SOURCES,
    VALID_SOFT_MODES,
    VALID_TRACKS,
    BaseStage2DualHead,
    BCSSSoftMemmapDataset,
    ConvNeXtFPNDualHead,
    DeepLabV3PlusDualHead,
    DINOv2AdapterDualHead,
    SegFormerDualHead,
    Stage2ModelEMA,
    build_stage2_v2_model,
    evaluate_r2_pseudo_mask_val_diagnostic,
    generate_bn7_soft_memmap,
    generate_cross_family_r2_memmap,
    generate_round2_ema_memmap,
    get_stage2_v2_preset,
    load_stage2_v2_checkpoint,
    resolve_stage2_v2_config_from_env,
    run_stage2_v2_ensemble_eval,
    save_ensemble_comparison_plot,
    train_stage2_v2,
    validate_track_a_arm_spec,
    validate_track_a_teacher_checkpoint,
    validate_track_name,
)
from src.stage2_v2.ensemble import TRACK_B_DENY_REGEX as ENS_TRACK_B_DENY_REGEX
from src.stage2_v2.memmap_dataset import TRACK_B_DENY_REGEX as MM_TRACK_B_DENY_REGEX
from src.stage2_v2.memmap_dataset import _cams_to_soft_probs
from src.stage2_v2.models import _compute_evenly_spaced_stages
from src.training.dataset_loader import load_bcss_patient_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TinyDualHead(BaseStage2DualHead):
    """Tiny synthetic dual-head for fast CPU memmap/trainer/ensemble tests."""

    def __init__(
        self,
        num_classes: int = 4,
        bias_logits: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        has_frozen: bool = False,
        track: str = "track_a_clean",
        tcga_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.arch = "deeplabv3plus_r101"
        self.backbone_id = "resnet101"
        self.num_classes = int(num_classes)
        self.track = str(track)
        self.tcga_pretrained = bool(tcga_pretrained)
        self.frozen_conv = nn.Conv2d(3, 4, kernel_size=1)
        if has_frozen:
            for p in self.frozen_conv.parameters():
                p.requires_grad = False
        self.head_conv = nn.Conv2d(4, self.num_classes, kernel_size=1)
        with torch.no_grad():
            self.head_conv.bias.copy_(torch.tensor(bias_logits, dtype=torch.float32))
        self.cls_head = nn.Linear(4, self.num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = torch.relu(self.frozen_conv(x))
        seg_logits = self.head_conv(feat)
        cls_feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return {"seg_logits": seg_logits, "cls_logits": self.cls_head(cls_feat)}

    def backbone_parameters(self):
        return self.frozen_conv.parameters()

    def head_parameters(self):
        return list(self.head_conv.parameters()) + list(self.cls_head.parameters())


class _TinySimpleDataset(torch.utils.data.Dataset):
    def __init__(self, files, img_size: int = 224, make_missing_class: bool = False) -> None:
        self.files = list(files)
        self.img_size = int(img_size)
        self.make_missing_class = bool(make_missing_class)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB").resize((self.img_size, self.img_size)), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr.transpose(2, 0, 1))
        gt = torch.zeros((self.img_size, self.img_size), dtype=torch.int64)
        h2 = self.img_size // 2
        gt[:h2, :h2] = 0
        gt[:h2, h2:] = 1
        gt[h2:, :h2] = 2
        gt[h2:, h2:] = 2 if (self.make_missing_class and idx % 2 == 0) else 3
        gt[:2, :2] = 4
        return img_t, gt


def test_gate7_presets_and_env_overrides() -> None:
    expected = {
        "arm1_deeplab_cleankd",
        "arm2_segformer_oeem",
        "arm3_convnext_selftrain",
        "arm4_dinov2_adapter",
        "arm5_dinov2_large_r2",
        "arm6_segformer_b5_r2",
        "arm7_convnext_large_r2",
        "arm8_deeplab_r101_r2",
        "arm9_phikon_v2_r2",
    }
    assert set(STAGE2_V2_PRESETS.keys()) == expected
    assert VALID_SOFT_MODES == ("raw_l1", "softmax_t", "sg_dap_soft")
    assert "ens4_r2_soft" not in VALID_SOFT_MODES
    assert VALID_MEMMAP_SOURCES == ("stage1", "ens4_r2")
    assert VALID_TRACKS == ("track_a_clean", "track_b_tcga_pretrained")

    for a in (
        "arm1_deeplab_cleankd",
        "arm2_segformer_oeem",
        "arm3_convnext_selftrain",
        "arm4_dinov2_adapter",
    ):
        c = get_stage2_v2_preset(a)
        assert c.memmap_source == "stage1"
        assert c.ema_only_best_selection is False
        assert c.track == "track_a_clean"
        assert c.tcga_pretrained is False
        assert c.internal_scale == 1.0

    for a in (
        "arm5_dinov2_large_r2",
        "arm6_segformer_b5_r2",
        "arm7_convnext_large_r2",
        "arm8_deeplab_r101_r2",
        "arm9_phikon_v2_r2",
    ):
        c = get_stage2_v2_preset(a)
        assert c.memmap_source == "ens4_r2"
        assert c.soft_mode == "sg_dap_soft"
        assert c.soft_temp == pytest.approx(0.25)
        assert c.self_train_round2_start_epoch is None
        assert c.ema_only_best_selection is True

    c5 = get_stage2_v2_preset("arm5_dinov2_large_r2")
    assert c5.arch == "dinov2_vitl14_adapter"
    assert c5.backbone_id == "facebook/dinov2-large"
    assert c5.internal_scale == pytest.approx(1.5)
    assert c5.fpn_dim == 192
    assert c5.unfreeze_last_n_blocks == 8
    assert c5.drop_path_rate == pytest.approx(0.10)
    assert c5.track == "track_a_clean" and c5.tcga_pretrained is False

    c6 = get_stage2_v2_preset("arm6_segformer_b5_r2")
    assert c6.arch == "segformer_b5" and c6.backbone_id == "nvidia/mit-b5"
    assert c6.drop_path_rate == pytest.approx(0.15)

    c7 = get_stage2_v2_preset("arm7_convnext_large_r2")
    assert c7.arch == "convnext_large_fpn" and c7.backbone_id == "convnext_large"
    assert c7.stochastic_depth_prob == pytest.approx(0.3)

    c9 = get_stage2_v2_preset("arm9_phikon_v2_r2")
    assert c9.track == "track_b_tcga_pretrained" and c9.tcga_pretrained is True

    c_env = resolve_stage2_v2_config_from_env(
        environ={
            "STAGE2_EXP_PRESET": "arm1_deeplab_cleankd",
            "STAGE2_MEMMAP_SOURCE": "ens4_r2",
            "STAGE2_EMA_ONLY_BEST_SELECTION": "1",
        }
    )
    assert c_env.memmap_source == "ens4_r2"
    assert c_env.ema_only_best_selection is True

    with pytest.raises(ValueError, match="STAGE2_MEMMAP_SOURCE"):
        resolve_stage2_v2_config_from_env(environ={"STAGE2_MEMMAP_SOURCE": "bad_src"})
    with pytest.raises(ValueError, match="STAGE2_EMA_ONLY_BEST_SELECTION"):
        resolve_stage2_v2_config_from_env(environ={"STAGE2_EMA_ONLY_BEST_SELECTION": "maybe"})


def test_dinov2_crop_before_resize_unfreeze_and_new_ckpt_roundtrip(tmp_path: Path) -> None:
    model = DINOv2AdapterDualHead(
        backbone_id="facebook/dinov2-large",
        num_classes=4,
        pretrained=False,
        unfreeze_last_n_blocks=8,
        tiny_offline_config=True,
        arch="dinov2_vitl14_adapter",
        fpn_dim=192,
        internal_scale=1.5,
        drop_path_rate=0.10,
    )
    # 4-layer tiny offline config unfreezes all 4; verify 24-layer unfreeze_last_n_blocks=8 touches stages 18 & 24
    cfg_24 = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=24,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=224,
        patch_size=14,
    )
    m24 = DINOv2AdapterDualHead.__new__(DINOv2AdapterDualHead)
    BaseStage2DualHead.__init__(m24)
    m24.backbone = Dinov2Model(cfg_24)
    m24.unfreeze_last_n_blocks = 8
    m24.stage_indices = _compute_evenly_spaced_stages(int(cfg_24.num_hidden_layers))
    m24._apply_freeze_policy()
    assert m24.stage_indices == (6, 12, 18, 24)
    # Stage 18 is layer index 17 (in layers[-8:], i.e. indices 16..23) -> trainable!
    assert all(p.requires_grad for p in m24.backbone.encoder.layer[17].parameters())
    assert all(p.requires_grad for p in m24.backbone.encoder.layer[23].parameters())
    assert all(not p.requires_grad for p in m24.backbone.encoder.layer[15].parameters())

    # Spy on _pad_to_patch_multiple to verify exact scaled dims + crop-before-resize
    seen_work_shapes = []
    orig_pad = model._pad_to_patch_multiple

    def _spy_pad(x_w: torch.Tensor):
        seen_work_shapes.append(tuple(x_w.shape[-2:]))
        return orig_pad(x_w)

    model._pad_to_patch_multiple = _spy_pad  # type: ignore[method-assign]
    model.eval()
    for hw, exp_scaled in (
        ((224, 224), (336, 336)),
        ((168, 168), (252, 252)),
        ((280, 280), (420, 420)),
        ((225, 225), (338, 338)),
    ):
        x = torch.randn(1, 3, hw[0], hw[1])
        out = model(x)
        assert out["seg_logits"].shape == (1, 4, hw[0], hw[1])
        assert out["cls_logits"].shape == (1, 4)
    assert seen_work_shapes == [(336, 336), (252, 252), (420, 420), (338, 338)]

    # Test tiny_offline_config=True roundtrip
    ckpt_tiny = tmp_path / "dino_tiny.pth"
    meta_out = model.model_metadata()
    torch.save({**meta_out, "state_dict": model.state_dict()}, ckpt_tiny)
    reloaded_tiny, meta_tiny = load_stage2_v2_checkpoint(ckpt_tiny, device="cpu", strict=True)
    assert reloaded_tiny.fpn_dim == 192
    assert reloaded_tiny.internal_scale == pytest.approx(1.5)
    assert meta_tiny["model_kwargs"]["drop_path_rate"] == pytest.approx(0.10)

    # Test __new__ non-tiny config path + legacy Wave-2 payload without new model_kwargs
    saved_hf = tmp_path / "local_hf_dino"
    Dinov2Config(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=224,
        patch_size=14,
    ).save_pretrained(saved_hf)
    model_nontiny = DINOv2AdapterDualHead(
        backbone_id="facebook/dinov2-large",
        num_classes=4,
        pretrained=False,
        unfreeze_last_n_blocks=2,
        tiny_offline_config=True,
        arch="dinov2_vitl14_adapter",
        fpn_dim=192,
        internal_scale=1.5,
        drop_path_rate=0.10,
    )
    # Build matching non-tiny model via saved_hf
    cfg_d = Dinov2Config.from_pretrained(saved_hf)
    model_nontiny.backbone = Dinov2Model(cfg_d)
    model_nontiny.hidden_size = 32
    model_nontiny.stage_norms = nn.ModuleList([nn.LayerNorm(32) for _ in range(4)])
    model_nontiny.stage_projs = nn.ModuleList([nn.Conv2d(32, 192, 1, bias=False) for _ in range(4)])
    model_nontiny.cls_head = nn.Linear(32, 4)
    model_nontiny.tiny_offline_config = False
    model_nontiny.backbone_id = str(saved_hf)
    ckpt_nontiny = tmp_path / "dino_nontiny.pth"
    nt_meta = model_nontiny.model_metadata()
    torch.save({**nt_meta, "state_dict": model_nontiny.state_dict()}, ckpt_nontiny)
    reloaded_nt, _ = load_stage2_v2_checkpoint(ckpt_nontiny, device="cpu", strict=True)
    assert reloaded_nt.fpn_dim == 192 and reloaded_nt.internal_scale == pytest.approx(1.5)

    # Legacy Wave-2 payload missing fpn_dim / internal_scale / drop_path_rate
    legacy_ckpt = tmp_path / "dino_legacy_w2.pth"
    m_leg = DINOv2AdapterDualHead(
        backbone_id=str(saved_hf), num_classes=4, pretrained=False, unfreeze_last_n_blocks=2, tiny_offline_config=True
    )
    m_leg.backbone = Dinov2Model(cfg_d)
    m_leg.hidden_size = 32
    m_leg.stage_norms = nn.ModuleList([nn.LayerNorm(32) for _ in range(4)])
    m_leg.stage_projs = nn.ModuleList([nn.Conv2d(32, 128, 1, bias=False) for _ in range(4)])
    m_leg.cls_head = nn.Linear(32, 4)
    torch.save(
        {
            "arch": "dinov2_vitb14_adapter",
            "backbone_id": str(saved_hf),
            "num_classes": 4,
            "model_kwargs": {"unfreeze_last_n_blocks": 2, "tiny_offline_config": False},
            "state_dict": m_leg.state_dict(),
        },
        legacy_ckpt,
    )
    leg_loaded, _ = load_stage2_v2_checkpoint(legacy_ckpt, device="cpu", strict=True)
    assert leg_loaded.fpn_dim == 128 and leg_loaded.internal_scale == pytest.approx(1.0)


def test_convnext_large_segformer_b5_and_deeplab_arm8(tmp_path: Path) -> None:
    cfg7 = replace(get_stage2_v2_preset("arm7_convnext_large_r2"))
    m7 = build_stage2_v2_model(cfg7, pretrained=False)
    assert isinstance(m7, ConvNeXtFPNDualHead)
    assert m7.in_channels_list == [192, 384, 768, 1536]
    assert m7.cls_head.in_features == 1536
    assert m7.stochastic_depth_prob == pytest.approx(0.3)
    out7 = m7(torch.randn(1, 3, 64, 64))
    assert out7["seg_logits"].shape == (1, 4, 64, 64) and out7["cls_logits"].shape == (1, 4)

    ckpt7 = tmp_path / "arm7.pth"
    torch.save({**m7.model_metadata(), "state_dict": m7.state_dict()}, ckpt7)
    r7, _ = load_stage2_v2_checkpoint(ckpt7, device="cpu", strict=True)
    assert isinstance(r7, ConvNeXtFPNDualHead) and r7.in_channels_list == [192, 384, 768, 1536]

    cfg6 = get_stage2_v2_preset("arm6_segformer_b5_r2")
    m6 = build_stage2_v2_model(cfg6, pretrained=False)
    assert isinstance(m6, SegFormerDualHead)
    assert m6.arch == "segformer_b5" and m6.drop_path_rate == pytest.approx(0.15)
    assert m6.segformer.config.drop_path_rate == pytest.approx(0.15)
    # Also test SegFormer __new__ non-tiny reload
    saved_sf = tmp_path / "local_sf"
    SegformerConfig(
        num_labels=4,
        depths=[1, 1, 1, 1],
        hidden_sizes=[16, 32, 64, 128],
        decoder_hidden_size=32,
        num_attention_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.15,
    ).save_pretrained(saved_sf)
    m6_nt = SegFormerDualHead(
        backbone_id=str(saved_sf), num_classes=4, pretrained=False, tiny_offline_config=True, arch="segformer_b5", drop_path_rate=0.15
    )
    m6_nt.tiny_offline_config = False
    ckpt6 = tmp_path / "arm6_nt.pth"
    torch.save({**m6_nt.model_metadata(), "state_dict": m6_nt.state_dict()}, ckpt6)
    r6, meta6 = load_stage2_v2_checkpoint(ckpt6, device="cpu", strict=True)
    assert r6.arch == "segformer_b5" and meta6["model_kwargs"]["drop_path_rate"] == pytest.approx(0.15)

    cfg8 = get_stage2_v2_preset("arm8_deeplab_r101_r2")
    m8 = build_stage2_v2_model(cfg8, pretrained=False)
    assert isinstance(m8, DeepLabV3PlusDualHead) and m8.arch == "deeplabv3plus_r101"


def test_ema_trainable_only_resume_and_ema_only_best_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1) Stage2ModelEMA trainable_only=True save & resume parity with frozen params
    m_orig = _TinyDualHead(has_frozen=True)
    ema = Stage2ModelEMA(m_orig, decay=0.9)
    with torch.no_grad():
        m_orig.head_conv.weight.add_(1.0)
    ema.update(m_orig)

    sd_full = ema.state_dict(trainable_only=False)
    sd_train = ema.state_dict(trainable_only=True, model=m_orig)
    assert sd_train["trainable_only"] is True
    assert "frozen_conv.weight" not in sd_train["shadow"]
    assert "head_conv.weight" in sd_train["shadow"]
    assert len(sd_train["shadow"]) < len(sd_full["shadow"])

    # Simulate resume into a fresh model where frozen weights are reloaded first
    m_resumed = _TinyDualHead(has_frozen=True)
    ema_resumed = Stage2ModelEMA(m_resumed, decay=0.9)
    m_resumed.load_state_dict(m_orig.state_dict(), strict=True)
    ema_resumed.load_state_dict(sd_train, model=m_resumed)
    for k, v in ema.clean_model_state_dict().items():
        assert torch.allclose(ema_resumed.clean_model_state_dict()[k], v, atol=1e-6)

    # 2) ema_only_best_selection=True selects EMA even when raw_miou > ema_miou
    splits = load_bcss_patient_splits()
    tr_pids = splits["train"][:2]
    val_pids = splits["val"][:1]
    img_dir = tmp_path / "train"
    img_dir.mkdir()
    rng = np.random.default_rng(11)
    for i, pid in enumerate(tr_pids):
        arr = rng.integers(40, 200, size=(224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(img_dir / f"{pid}_dx1_x{i}_[1100].png")
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    Image.fromarray(rng.integers(40, 200, size=(224, 224, 3), dtype=np.uint8), mode="RGB").save(
        val_dir / f"{val_pids[0]}_dx1_v0_[1111].png"
    )
    val_loader = DataLoader(_TinySimpleDataset(sorted(val_dir.glob("*.png")), img_size=64), batch_size=1)

    s1_m = WSSSTissueStage1NetCAM(n_class=4)
    mm_dir = tmp_path / "memmap"
    s1_dat, s1_man, _ = generate_bn7_soft_memmap(
        s1_m, img_dir, mm_dir, patient_ids=tr_pids, soft_mode="sg_dap_soft", batch_size=2, device=torch.device("cpu")
    )
    # Build ens4_r2 memmap via 1 tiny SegFormer teacher (real load_stage2_v2_checkpoint)
    t_ckpt = tmp_path / "t1.pth"
    t_mod = SegFormerDualHead(num_classes=4, pretrained=False, tiny_offline_config=True)
    torch.save({**t_mod.model_metadata(), "state_dict": t_mod.state_dict()}, t_ckpt)
    r2_dat, r2_man, r2_manifest = generate_cross_family_r2_memmap(
        [t_ckpt], img_dir, s1_dat, s1_man, mm_dir, patient_ids=tr_pids, batch_size=2, device=torch.device("cpu")
    )

    from src.stage2_v2 import trainer as s2_trainer

    # Monkeypatch evaluate_stage2_v2_model to return raw_miou=0.8000 vs ema_miou=0.7200
    call_idx = {"n": 0}

    def _fake_eval(*args, **kwargs):
        call_idx["n"] += 1
        # Epoch 1: call 1 = raw (0.80), call 2 = ema (0.72); final step-8 eval calls return 0.72
        miou = 0.80 if call_idx["n"] == 1 else 0.72
        base = {"mIoU": miou, "mDice": miou, "per_class_iou": {"tum": miou, "str": miou, "lym": miou, "nec": miou}, "class_ious": [miou] * 4}
        return {"num_patches": 1, "raw": base, "cgm_self": base, "cgm_s1": base, "cgm_dual": base, "ms_tta": base, "ms_tta_cgm": base}

    monkeypatch.setattr(s2_trainer, "evaluate_stage2_v2_model", _fake_eval)
    monkeypatch.setattr(
        s2_trainer,
        "calibrate_cgm_thresholds_val_only",
        lambda **_kw: {"gate_source": "self", "thresholds": [0.1, 0.1, 0.1, 0.1], "per_class_thresholds": {}, "val_miou": 0.72},
    )
    monkeypatch.setattr(
        s2_trainer,
        "load_stage2_v2_checkpoint",
        lambda ckpt_p, device="cpu", strict=True: (_TinyDualHead(has_frozen=True), torch.load(ckpt_p, map_location="cpu", weights_only=False)),
    )

    cfg_ema_only = replace(
        get_stage2_v2_preset("arm5_dinov2_large_r2"),
        epochs=1,
        num_workers=0,
        batch_size=2,
        ema_only_best_selection=True,
    )
    m_train = _TinyDualHead(has_frozen=True)
    ckpt_dir = tmp_path / "ckpts"
    out_dir = tmp_path / "out"
    res = train_stage2_v2(
        cfg=cfg_ema_only,
        model=m_train,
        train_img_dir=img_dir,
        memmap_path=r2_dat,
        manifest_path=r2_man,
        val_loader=val_loader,
        stage1_sha256="abc",
        train_patient_ids=tr_pids,
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
        zip_path=tmp_path / "res.zip",
        device=torch.device("cpu"),
    )
    assert res["best_selected_weights"] == "ema"
    assert res["best_val_miou"] == pytest.approx(0.72)
    assert res["memmap_source"] == "ens4_r2"
    assert res["r2_manifest_sha256"] == compute_file_sha256_local(r2_man)

    best_p = torch.load(ckpt_dir / "stage2_retrained_best.pth", map_location="cpu", weights_only=False)
    latest_p = torch.load(ckpt_dir / "stage2_latest.pth", map_location="cpu", weights_only=False)
    assert best_p["selected_weights"] == "ema"
    assert best_p["memmap_source"] == "ens4_r2"
    assert best_p["r2_manifest_sha256"] == res["r2_manifest_sha256"]
    assert "frozen_conv.weight" in best_p["state_dict"]
    assert latest_p["ema_state_dict"]["trainable_only"] is True
    assert "frozen_conv.weight" not in latest_p["ema_state_dict"]["shadow"]
    assert r2_manifest["stage1_sha256"] == ""


def compute_file_sha256_local(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_r2_memmap_guards_tail_noise_leakage_and_val_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = load_bcss_patient_splits()
    tr_pids = splits["train"][:2]
    val_pids = splits["val"][:1]
    te_pids = splits["test"][:1]

    # 1) y_gt.sum() == 0 uniform fallback in _cams_to_soft_probs
    cam_z = torch.rand(1, 4, 8, 8)
    y_z = torch.zeros(1, 4)
    rgb_z = torch.rand(1, 3, 8, 8)
    for sm in ("raw_l1", "softmax_t", "sg_dap_soft"):
        q_u = _cams_to_soft_probs(cam_z, y_z, rgb_z, soft_mode=sm, soft_temp=0.25)
        assert torch.allclose(q_u, torch.full_like(q_u, 0.25), atol=1e-4)

    # 2) Build train patches: 1 single-class, 1 zero-label, 2 multi-label
    img_dir = tmp_path / "train"
    img_dir.mkdir()
    rng = np.random.default_rng(5)
    fns = [
        f"{tr_pids[0]}_dx1_x0_[1000].png",
        f"{tr_pids[0]}_dx1_x1_[0000].png",
        f"{tr_pids[0]}_dx1_x2_[1100].png",
        f"{tr_pids[1]}_dx1_x3_[1010].png",
    ]
    for fn in fns:
        Image.fromarray(rng.integers(20, 220, size=(224, 224, 3), dtype=np.uint8), mode="RGB").save(img_dir / fn)

    s1_m = WSSSTissueStage1NetCAM(n_class=4)
    mm_dir = tmp_path / "mm"
    s1_dat, s1_man, _ = generate_bn7_soft_memmap(
        s1_m, img_dir, mm_dir, patient_ids=tr_pids, stage1_sha256="s1sha_test", soft_mode="raw_l1", batch_size=2, device=torch.device("cpu")
    )

    # Check BCSSSoftMemmapDataset y_gt.sum() == 0 uniform fallback & generate_round2_ema_memmap
    ds = BCSSSoftMemmapDataset(img_dir, s1_dat, s1_man, patient_ids=tr_pids, is_train=False)
    _, q0, y0, _ = ds[1]
    assert y0.sum().item() == 0.0
    assert torch.allclose(q0, torch.full_like(q0, 0.25), atol=0.02)

    t_tiny = _TinyDualHead(bias_logits=(-20.0, -20.0, 10.0, 10.0))
    generate_round2_ema_memmap(
        t_tiny, img_dir, s1_dat, s1_man, mm_dir / "pm_soft_r2.dat", mm_dir / "pm_soft_r2_manifest.json", batch_size=2, device=torch.device("cpu")
    )

    # Save 4 tiny teachers whose logits are overwhelmingly on class 2 & 3 (so on patch [1100] p_sum < 1e-2!)
    import src.stage2_v2.models as s2_models

    orig_load = s2_models.load_stage2_v2_checkpoint

    def _stub_aware_load(ckpt_p, device="cpu", strict=True):
        pl = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        if pl.get("tiny_test_stub"):
            m_stub = _TinyDualHead(
                track=str(pl.get("track", "track_a_clean")),
                tcga_pretrained=bool(pl.get("tcga_pretrained", False)),
            )
            m_stub.load_state_dict(pl["state_dict"], strict=True)
            return m_stub.to(device).eval(), pl
        return orig_load(ckpt_p, device=device, strict=strict)

    monkeypatch.setattr(s2_models, "load_stage2_v2_checkpoint", _stub_aware_load)

    t_paths = []
    for i in range(4):
        tp = tmp_path / f"teacher_{i}.pth"
        tm = _TinyDualHead(bias_logits=(-12.0, -12.0, 8.0, 8.0))
        with torch.no_grad():
            tm.frozen_conv.weight.zero_()
            tm.head_conv.weight.zero_()
        torch.save({**tm.model_metadata(), "tiny_test_stub": True, "state_dict": tm.state_dict()}, tp)
        t_paths.append(tp)

    # Anti-leakage guard: val/test patient raises RuntimeError
    with pytest.raises(RuntimeError, match="ANTI-LEAKAGE"):
        generate_cross_family_r2_memmap(
            t_paths, img_dir, s1_dat, s1_man, mm_dir, patient_ids=[val_pids[0]], device=torch.device("cpu")
        )

    # Track A guard rejects tcga_pretrained=True or phikon name
    bad_t = tmp_path / "phikon_v2_teacher.pth"
    torch.save({**_TinyDualHead(track="track_b_tcga_pretrained", tcga_pretrained=True).model_metadata(), "tiny_test_stub": True, "state_dict": _TinyDualHead().state_dict()}, bad_t)
    with pytest.raises(RuntimeError, match="TRACK A QUARANTINE"):
        generate_cross_family_r2_memmap(
            [bad_t], img_dir, s1_dat, s1_man, mm_dir, patient_ids=tr_pids, track="track_a_clean", device=torch.device("cpu")
        )
    # Explicitly allowed under Track B
    r2_b_dat, _, _ = generate_cross_family_r2_memmap(
        [bad_t], img_dir, s1_dat, s1_man, mm_dir, patient_ids=tr_pids, track="track_b_tcga_pretrained", dat_filename="pm_b.dat", manifest_filename="pm_b.json", device=torch.device("cpu")
    )
    assert r2_b_dat.is_file()

    # Run Track A R2 generation
    r2_dat, r2_man, man = generate_cross_family_r2_memmap(
        t_paths, img_dir, s1_dat, s1_man, mm_dir, patient_ids=tr_pids, batch_size=2, device=torch.device("cpu")
    )
    assert r2_dat.name == "pm_soft_ens4_r2.dat" and r2_man.name == "pm_soft_ens4_r2_manifest.json"
    assert man["stage1_sha256"] == "s1sha_test"
    assert len(man["teacher_sha256s"]) == 4
    assert man["single_label_short_circuit"] == 1
    assert man["zero_label_patches"] == 1
    assert man["multi_label_computed"] == 2
    assert man["frac_pixels_psum_lt_0_01"] > 0.4
    assert "mean_entropy_p_ens4_uncal" in man
    assert "mean_entropy_q_r2" in man
    assert "argmax_disagreement_p_ens4_vs_s1" in man
    # Verify no leftover temp FP32 memmaps on Windows
    assert list(mm_dir.glob(".tmp_ens4_r2_fp32_*.dat")) == []

    # Verify absent-class tail-noise guard: patch 2 ([1100]) must equal s1 memmap exactly!
    mm_s1_r = np.memmap(s1_dat, dtype=np.uint8, mode="r", shape=(4, 4, 224, 224))
    mm_r2_r = np.memmap(r2_dat, dtype=np.uint8, mode="r", shape=(4, 4, 224, 224))
    assert np.allclose(mm_r2_r[2], mm_s1_r[2], atol=1)
    assert int(mm_r2_r[0, 0].min()) == 255 and int(mm_r2_r[0, 1:].max()) == 0
    assert np.all(mm_r2_r[1] == 64)
    mm_s1_r._mmap.close()
    mm_r2_r._mmap.close()

    # Simulated exception cleans up temp memmap
    def _boom(*_a, **_kw):
        raise ValueError("simulated crash")

    monkeypatch.setattr(s2_models, "load_stage2_v2_checkpoint", _boom)
    with pytest.raises(ValueError, match="simulated crash"):
        generate_cross_family_r2_memmap(
            t_paths, img_dir, s1_dat, s1_man, mm_dir, patient_ids=tr_pids, force=True, device=torch.device("cpu")
        )
    assert list(mm_dir.glob(".tmp_ens4_r2_fp32_*.dat")) == []
    monkeypatch.setattr(s2_models, "load_stage2_v2_checkpoint", _stub_aware_load)

    # VAL diagnostic (zero TEST access)
    val_dir = tmp_path / "val_d"
    val_dir.mkdir()
    for i in range(2):
        Image.fromarray(rng.integers(20, 220, size=(32, 32, 3), dtype=np.uint8), mode="RGB").save(
            val_dir / f"{val_pids[0]}_dx1_v{i}_[1111].png"
        )
    val_loader = DataLoader(_TinySimpleDataset(sorted(val_dir.glob("*.png")), img_size=32), batch_size=2)
    diag_json = tmp_path / "diag.json"
    diag = evaluate_r2_pseudo_mask_val_diagnostic(
        val_loader=val_loader,
        teacher_ckpt_paths=t_paths,
        stage1_model=s1_m,
        stage1_sha256="s1sha_test",
        soft_mode="raw_l1",
        output_json_path=diag_json,
        device=torch.device("cpu"),
    )
    assert diag["split"] == "val_only"
    assert set(diag["candidates"].keys()) == {"q_s1_sgdap", "p_ens4_uncal_ygated", "q_r2"}
    for ck in ("q_s1_sgdap", "p_ens4_uncal_ygated", "q_r2"):
        c_rep = diag["candidates"][ck]
        assert "mIoU" in c_rep and "multiclass_mIoU" in c_rep
        assert set(c_rep["per_class_iou"].keys()) == {"tum", "str", "lym", "nec"}
        assert "ECE" in c_rep and "multiclass_entropy" in c_rep
    assert "frac_psum_lt_0_05" in diag
    assert isinstance(diag["go_decision"], bool) and isinstance(diag["ceiling_gate_pass"], bool)
    assert diag_json.is_file()

    # TEST patient inside val_loader raises RuntimeError
    te_dir = tmp_path / "te_d"
    te_dir.mkdir()
    Image.fromarray(rng.integers(20, 220, size=(32, 32, 3), dtype=np.uint8), mode="RGB").save(
        te_dir / f"{te_pids[0]}_dx1_t0_[1111].png"
    )
    bad_loader = DataLoader(_TinySimpleDataset(sorted(te_dir.glob("*.png")), img_size=32), batch_size=1)
    with pytest.raises(RuntimeError, match="ANTI-LEAKAGE"):
        evaluate_r2_pseudo_mask_val_diagnostic(bad_loader, t_paths, output_json_path=None, device=torch.device("cpu"))


def test_ensemble_track_a_quarantine_8shades_and_per_patient_class_ious(tmp_path: Path) -> None:
    for a in ("arm5_dinov2_large_r2", "arm6_segformer_b5_r2", "arm7_convnext_large_r2", "arm8_deeplab_r101_r2", "arm9_phikon_v2_r2"):
        assert a in DEFAULT_ARM_FAMILIES

    # Track A quarantine blocks Track B arm
    with pytest.raises(ValueError, match="TRACK A QUARANTINE"):
        validate_track_a_arm_spec({"name": "arm9_phikon_v2_r2"}, track="track_a_clean")
    validate_track_a_arm_spec({"name": "arm9_phikon_v2_r2"}, track="track_b_tcga_pretrained")

    # 8 distinct grayscale monochrome bars
    plot_p = tmp_path / "bars8.png"
    entries = [(f"B{i}", 0.70 + 0.01 * i, 0.68 + 0.01 * i, 0.72 + 0.01 * i) for i in range(8)]
    save_ensemble_comparison_plot(entries, plot_p)
    assert plot_p.is_file()

    # Inspect matplotlib bar facecolors directly to confirm 8 distinct R==G==B shades
    shades = ["#E0E0E0", "#C4C4C4", "#A8A8A8", "#8C8C8C", "#707070", "#545454", "#343434", "#000000"]
    assert len(set(shades)) == 8
    for hx in shades:
        rgb = matplotlib.colors.to_rgb(hx)
        assert rgb[0] == rgb[1] == rgb[2]

    # Tiny E2E ensemble verifying per_patient_class_ious.json (with null for union==0) + ZIP bundle
    splits = load_bcss_patient_splits()
    v_pids = splits["val"][:4]
    t_pids = splits["test"][:4]
    rng = np.random.default_rng(19)
    v_files = []
    t_files = []
    for sub, pids, out_list in (("val", v_pids, v_files), ("test", t_pids, t_files)):
        d = tmp_path / sub
        d.mkdir()
        for i, pid in enumerate(pids):
            p = d / f"{pid}_dx1_p{i}.png"
            Image.fromarray(rng.integers(20, 220, size=(16, 16, 3), dtype=np.uint8), mode="RGB").save(p)
            out_list.append(p)

    val_loader = DataLoader(_TinySimpleDataset(v_files, img_size=16), batch_size=4)
    # Test set has missing class 3 (NEC) on patient indices 0 and 2 -> union == 0 -> null in JSON
    test_loader = DataLoader(_TinySimpleDataset(t_files, img_size=16, make_missing_class=True), batch_size=4)

    arms = [
        {"name": "arm5_dinov2_large_r2", "family": "vit", "model": _TinyDualHead(bias_logits=(1.0, 0.0, 0.0, -25.0))},
        {"name": "arm6_segformer_b5_r2", "family": "vit", "model": _TinyDualHead(bias_logits=(0.0, 1.0, 0.0, -25.0))},
        {"name": "arm7_convnext_large_r2", "family": "cnn", "model": _TinyDualHead(bias_logits=(0.0, 0.0, 1.0, -25.0))},
        {"name": "arm8_deeplab_r101_r2", "family": "cnn", "model": _TinyDualHead(bias_logits=(0.5, 0.5, 0.5, -25.0))},
    ]
    for a_spec in arms:
        with torch.no_grad():
            a_spec["model"].head_conv.weight.zero_()
    out_dir = tmp_path / "ens_out"
    zip_p = tmp_path / "ens.zip"

    # Guard triggers in run_stage2_v2_ensemble_eval when Track B arm is passed under track_a_clean
    bad_arms = list(arms[:3]) + [{"name": "arm9_phikon_v2_r2", "family": "vit", "model": _TinyDualHead(track="track_b_tcga_pretrained", tcga_pretrained=True)}]
    with pytest.raises(ValueError, match="TRACK A QUARANTINE"):
        run_stage2_v2_ensemble_eval(bad_arms, val_loader, test_loader, output_dir=out_dir, zip_path=zip_p, track="track_a_clean", n_bootstrap=20, n_perm=20)

    rep = run_stage2_v2_ensemble_eval(
        arms,
        val_loader,
        test_loader,
        output_dir=out_dir,
        zip_path=zip_p,
        track="track_a_clean",
        n_bootstrap=50,
        n_perm=50,
        device=torch.device("cpu"),
    )
    assert rep["track"] == "track_a_clean"
    pp_path = out_dir / "per_patient_class_ious.json"
    assert pp_path.is_file()
    pp_obj = json.loads(pp_path.read_text(encoding="utf-8"))
    for req_k in (
        "patient_ids",
        "class_names",
        "class_keys",
        "primary_4arm",
        "equal_4arm",
        "simplex_cv_4arm",
        "individual_arms",
        "val_best_cnn_vit_pair",
        "author_anchors",
    ):
        assert req_k in pp_obj
    assert len(pp_obj["individual_arms"]) == 4
    prim_u = np.asarray(pp_obj["primary_4arm"]["union_p"])
    assert prim_u.shape == (4, 4)
    # Confirm null where union_p == 0
    has_zero_union = False
    for p_idx in range(4):
        for c_idx in range(4):
            if prim_u[p_idx, c_idx] == 0:
                has_zero_union = True
                assert pp_obj["primary_4arm"]["per_patient_class_ious"][p_idx][c_idx] is None
    assert has_zero_union
    with zipfile.ZipFile(zip_p, "r") as zf:
        assert "per_patient_class_ious.json" in zf.namelist()


def test_notebooks_02b_and_02c_gate7_structure_and_ast() -> None:
    nb02b = json.loads((PROJECT_ROOT / "notebooks" / "02b_Stage2_Breakthrough_G4.ipynb").read_text(encoding="utf-8"))
    assert len(nb02b["cells"]) == 9
    for idx, cell in enumerate(nb02b["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"02b_cell_{idx}.py")
    c1_02b = "".join(nb02b["cells"][1]["source"])
    c5_02b = "".join(nb02b["cells"][5]["source"])
    c6_02b = "".join(nb02b["cells"][6]["source"])
    c7_02b = "".join(nb02b["cells"][7]["source"])
    assert "_ensure_transformers" in c1_02b and "min_ver=(5, 0, 0)" in c1_02b
    assert "diag_r2_val" in c1_02b and "diag_r2_val" in c5_02b
    assert "ens4_r2" in c5_02b and "generate_cross_family_r2_memmap" in c5_02b
    assert "evaluate_r2_pseudo_mask_val_diagnostic" in c5_02b
    assert "diag_r2_val" in c6_02b and "diag_r2_val" in c7_02b

    nb02c = json.loads((PROJECT_ROOT / "notebooks" / "02c_Stage2_Ensemble_Eval_G4.ipynb").read_text(encoding="utf-8"))
    assert len(nb02c["cells"]) == 6
    for idx, cell in enumerate(nb02c["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"02c_cell_{idx}.py")
    c3_02c = "".join(nb02c["cells"][2]["source"])
    c4_02c = "".join(nb02c["cells"][3]["source"])
    c6_02c = "".join(nb02c["cells"][5]["source"])
    assert "STAGE2_ENSEMBLE_ARMS" in c3_02c and "STAGE2_TRACK" in c3_02c
    assert "validate_track_a_arm_spec" in c3_02c
    assert "track=STAGE2_TRACK" in c4_02c
    assert "per_patient_class_ious.json" in c6_02c

    # P1-1 / P1-4 / P1-5 checks in 02b
    assert 'diag_soft_mode = os.environ.get("STAGE2_SOFT_MODE", "").strip().lower() or "sg_dap_soft"' in c5_02b
    assert "soft_mode=diag_soft_mode" in c5_02b
    assert "os.replace(_diag_tmp, _diag_json)" in c5_02b
    assert "pm_soft_ens4_r2_trackb.dat" in c5_02b
    assert "pm_soft_ens4_r2_trackb_manifest.json" in c5_02b
    assert "and not ZIP_EXPORT_S2.is_file()" not in c7_02b
    assert ".zip.tmp" in c7_02b and "os.replace(_tmp_zip, ZIP_EXPORT_S2)" in c7_02b


def test_iter2_remediation_p1_p2_and_challenger_defense(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torch.utils.data import Subset

    import src.stage2_v2.models as s2_models
    from src.stage2_v2 import trainer as s2_trainer

    # [P1-3] Legacy ens4_r2_soft alias in env + explicit rejection in memmap_dataset
    c_alias = resolve_stage2_v2_config_from_env(environ={"STAGE2_SOFT_MODE": "ens4_r2_soft"})
    assert c_alias.soft_mode == "sg_dap_soft"
    assert c_alias.memmap_source == "ens4_r2"
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_stage2_v2_config_from_env(
            environ={"STAGE2_SOFT_MODE": "ens4_r2_soft", "STAGE2_MEMMAP_SOURCE": "stage1"}
        )
    with pytest.raises(ValueError, match="ens4_r2_soft"):
        _cams_to_soft_probs(torch.rand(1, 4, 8, 8), torch.ones(1, 4), torch.rand(1, 3, 8, 8), "ens4_r2_soft", 0.25)
    with pytest.raises(ValueError, match="ens4_r2_soft"):
        generate_bn7_soft_memmap(WSSSTissueStage1NetCAM(n_class=4), tmp_path, tmp_path, soft_mode="ens4_r2_soft")

    # [P2-3] TRACK_B_DENY_REGEX allows uniform/unified/community while rejecting uni-2/MahmoodLab/UNI
    for ok_tok in ("uniform_arm5.pth", "unified_model", "community_resnet"):
        assert MM_TRACK_B_DENY_REGEX.search(ok_tok) is None
        assert ENS_TRACK_B_DENY_REGEX.search(ok_tok) is None
        validate_track_a_arm_spec({"name": ok_tok, "arch": ok_tok}, track="track_a_clean")
    for bad_tok in ("uni-2.pth", "MahmoodLab/UNI", "uni_v2_teacher", "phikon-v2", "gigapath", "virchow2", "conch"):
        assert MM_TRACK_B_DENY_REGEX.search(bad_tok) is not None
        assert ENS_TRACK_B_DENY_REGEX.search(bad_tok) is not None
        with pytest.raises(ValueError, match="TRACK A QUARANTINE"):
            validate_track_a_arm_spec({"name": "arm5_dinov2_large_r2", "ckpt_path": bad_tok}, track="track_a_clean")

    # [P2-2] Fail closed on unknown track strings across modules
    assert validate_track_name("track_a") == "track_a_clean"
    for bad_trk in ("track_c", "track_a_typo", ""):
        with pytest.raises(ValueError, match="track"):
            validate_track_name(bad_trk)
        with pytest.raises(ValueError, match="track"):
            validate_track_a_arm_spec({"name": "arm1_deeplab_cleankd"}, track=bad_trk)
        with pytest.raises(ValueError, match="track"):
            generate_cross_family_r2_memmap(
                [tmp_path / "missing.pth"], tmp_path, tmp_path / "s1.dat", tmp_path / "s1.json", tmp_path, track=bad_trk
            )

    # [P2-8] Stage2ModelEMA.load_state_dict raises ValueError if trainable_only=True and model=None
    m_frozen = _TinyDualHead(has_frozen=True)
    ema = Stage2ModelEMA(m_frozen, decay=0.9)
    sd_tr = ema.state_dict(trainable_only=True, model=m_frozen)
    with pytest.raises(ValueError, match="model="):
        ema.load_state_dict(sd_tr, model=None)

    # [P2-7] _probe_interp_pos_support matches dummy probe dtype to backbone dtype (bfloat16)
    m_dino = DINOv2AdapterDualHead(
        backbone_id="facebook/dinov2-base", num_classes=4, pretrained=False, tiny_offline_config=True
    )
    m_dino.backbone = m_dino.backbone.to(dtype=torch.bfloat16)
    assert isinstance(m_dino._probe_interp_pos_support(), bool)

    # [P1-5] Track B memmap separation & trainer manifest["track"] validation
    splits = load_bcss_patient_splits()
    tr_pids = splits["train"][:2]
    val_pids = splits["val"][:2]
    te_pids = splits["test"][:1]
    img_dir = tmp_path / "tr_imgs"
    img_dir.mkdir()
    rng = np.random.default_rng(77)
    for idx_i, pid in enumerate(tr_pids):
        Image.fromarray(rng.integers(30, 210, size=(224, 224, 3), dtype=np.uint8), mode="RGB").save(
            img_dir / f"{pid}_dx1_p{idx_i}_[1100].png"
        )
    s1_m = WSSSTissueStage1NetCAM(n_class=4)
    mm_dir = tmp_path / "mm_sep"
    s1_dat, s1_man, _ = generate_bn7_soft_memmap(
        s1_m, img_dir, mm_dir, patient_ids=tr_pids, soft_mode="raw_l1", batch_size=2, device=torch.device("cpu")
    )

    def _stub_load(ckpt_p, device="cpu", strict=True):
        pl = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        if pl.get("tiny_test_stub"):
            m_st = _TinyDualHead(
                track=str(pl.get("track", "track_a_clean")),
                tcga_pretrained=bool(pl.get("tcga_pretrained", False)),
            )
            m_st.load_state_dict(pl["state_dict"], strict=True)
            return m_st.to(device).eval(), pl
        raise AssertionError("unexpected non-stub load")

    monkeypatch.setattr(s2_models, "load_stage2_v2_checkpoint", _stub_load)
    tb_ckpt = tmp_path / "tb_stub.pth"
    tm_b = _TinyDualHead(bias_logits=(-12.0, -12.0, 8.0, 8.0), track="track_b_tcga_pretrained", tcga_pretrained=True)
    with torch.no_grad():
        tm_b.frozen_conv.weight.zero_()
        tm_b.head_conv.weight.zero_()
    torch.save({**tm_b.model_metadata(), "tiny_test_stub": True, "state_dict": tm_b.state_dict()}, tb_ckpt)
    tb_dat, tb_man, tb_manifest = generate_cross_family_r2_memmap(
        [tb_ckpt], img_dir, s1_dat, s1_man, mm_dir, patient_ids=tr_pids, track="track_b_tcga_pretrained", device=torch.device("cpu")
    )
    assert tb_dat.name == "pm_soft_ens4_r2_trackb.dat"
    assert tb_man.name == "pm_soft_ens4_r2_trackb_manifest.json"
    assert tb_manifest["track"] == "track_b_tcga_pretrained"

    cfg_b = get_stage2_v2_preset("arm9_phikon_v2_r2")
    res_mm_b, res_man_b, _, _ = s2_trainer._resolve_memmap_source(cfg_b, mm_dir / "pm_soft_ens4_r2.dat", mm_dir / "pm_soft_ens4_r2_manifest.json")
    assert res_mm_b.name == "pm_soft_ens4_r2_trackb.dat"
    assert res_man_b.name == "pm_soft_ens4_r2_trackb_manifest.json"

    # Poisoned Track A manifest pointing to Track B must raise ValueError under Track A
    poison_man = mm_dir / "pm_soft_ens4_r2_manifest.json"
    poison_man.write_text(json.dumps({"track": "track_b_tcga_pretrained", "filenames": []}), encoding="utf-8")
    cfg_a = get_stage2_v2_preset("arm5_dinov2_large_r2")
    with pytest.raises(ValueError, match="TRACK A QUARANTINE"):
        s2_trainer._resolve_memmap_source(cfg_a, mm_dir / "pm_soft_ens4_r2.dat", poison_man)

    # [P1-1], [P1-2], [P2-4] diag_r2_val strict multi-class psum_*, Subset unwrap, train/test/empty rejection, soft_mode/soft_temp + sidecar JSON
    ta_ckpt = tmp_path / "ta_stub.pth"
    tm_a = _TinyDualHead(bias_logits=(-12.0, -12.0, 8.0, 8.0), track="track_a_clean", tcga_pretrained=False)
    with torch.no_grad():
        tm_a.frozen_conv.weight.zero_()
        tm_a.head_conv.weight.zero_()
    torch.save({**tm_a.model_metadata(), "tiny_test_stub": True, "state_dict": tm_a.state_dict()}, ta_ckpt)

    val_mix_dir = tmp_path / "val_mix"
    val_mix_dir.mkdir()
    for fn_v in (
        f"{val_pids[0]}_dx1_single_[1000].png",
        f"{val_pids[1]}_dx1_multi_[1100].png",
    ):
        Image.fromarray(rng.integers(20, 220, size=(32, 32, 3), dtype=np.uint8), mode="RGB").save(val_mix_dir / fn_v)

    class _SingleThenMultiDataset(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = list(files)

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = torch.zeros(3, 32, 32, dtype=torch.float32)
            gt = torch.zeros((32, 32), dtype=torch.int64)
            if idx == 1:
                gt[:, 16:] = 1  # multi-class [0, 1] -> p_sum < 1e-2 everywhere since teacher predicts [2, 3]!
            return img, gt

    mix_ds = _SingleThenMultiDataset(sorted(val_mix_dir.glob("*.png")))
    # batch_size=1 ensures batch 0 has 0 multi-class patches and batch 1 has 1 multi-class patch
    mix_loader = DataLoader(Subset(Subset(mix_ds, [0, 1]), [0, 1]), batch_size=1)
    metrics_out = tmp_path / "diag_out" / "STAGE2_TRAINING_METRICS.json"
    diag_rep = evaluate_r2_pseudo_mask_val_diagnostic(
        val_loader=mix_loader,
        teacher_ckpt_paths=[ta_ckpt],
        stage1_model=s1_m,
        soft_mode="raw_l1",
        soft_temp=0.25,
        output_json_path=metrics_out,
        device=torch.device("cpu"),
    )
    assert diag_rep["soft_mode"] == "raw_l1"
    assert diag_rep["soft_temp"] == pytest.approx(0.25)
    assert diag_rep["num_multiclass_val_patches"] == 1
    assert diag_rep["frac_psum_lt_0_05"] == pytest.approx(1.0)
    assert metrics_out.is_file()
    assert (metrics_out.parent / "STAGE2_R2_VAL_DIAGNOSTIC.json").is_file()

    # Nested Subset containing a TRAIN patient must fail anti-leakage check
    mix_plus_train = _SingleThenMultiDataset(
        sorted(val_mix_dir.glob("*.png")) + [img_dir / f"{tr_pids[0]}_dx1_p0_[1100].png"]
    )
    with pytest.raises(RuntimeError, match="ANTI-LEAKAGE"):
        evaluate_r2_pseudo_mask_val_diagnostic(
            DataLoader(Subset(mix_plus_train, [0, 2]), batch_size=1),
            [ta_ckpt],
            output_json_path=None,
            device=torch.device("cpu"),
        )
    # Empty val_pids_found (missing .files/.samples) must fail closed
    with pytest.raises(RuntimeError, match="ANTI-LEAKAGE"):
        evaluate_r2_pseudo_mask_val_diagnostic(
            DataLoader(torch.utils.data.TensorDataset(torch.zeros(1, 3, 32, 32), torch.zeros(1, 32, 32, dtype=torch.long))),
            [ta_ckpt],
            output_json_path=None,
            device=torch.device("cpu"),
        )
    # Unknown track or ens4_r2_soft must fail closed
    with pytest.raises(ValueError, match="track"):
        evaluate_r2_pseudo_mask_val_diagnostic(mix_loader, [ta_ckpt], track="bad_track", output_json_path=None)
    with pytest.raises(ValueError, match="soft_mode"):
        evaluate_r2_pseudo_mask_val_diagnostic(mix_loader, [ta_ckpt], soft_mode="ens4_r2_soft", output_json_path=None)

    # [Challenger Defense-in-Depth] Forged raw checkpoint metadata fails closed upfront in run_stage2_v2_ensemble_eval
    forged_ckpt = tmp_path / "arm5_dinov2_large_r2.pth"
    torch.save(
        {
            "arch": "deeplabv3plus_r101",
            "backbone_id": "resnet101",
            "track": "track_b_tcga_pretrained",
            "tcga_pretrained": True,
            "state_dict": tm_b.state_dict(),
        },
        forged_ckpt,
    )
    te_file = tmp_path / "te_only" / f"{te_pids[0]}_dx1_t0.png"
    te_file.parent.mkdir()
    Image.fromarray(rng.integers(20, 220, size=(16, 16, 3), dtype=np.uint8), mode="RGB").save(te_file)
    v_loader_s = DataLoader(_TinySimpleDataset([val_mix_dir / f"{val_pids[0]}_dx1_single_[1000].png"], img_size=16), batch_size=1)
    t_loader_s = DataLoader(_TinySimpleDataset([te_file], img_size=16), batch_size=1)
    benign_arm = {"name": "arm6_segformer_b5_r2", "family": "vit", "model": _TinyDualHead()}
    forged_arms = [
        {"name": "arm5_dinov2_large_r2", "family": "vit", "ckpt_path": forged_ckpt},
        benign_arm,
        benign_arm,
        benign_arm,
    ]
    with pytest.raises(RuntimeError, match="TRACK A QUARANTINE"):
        run_stage2_v2_ensemble_eval(
            forged_arms, v_loader_s, t_loader_s, output_dir=tmp_path / "ens_f", zip_path=tmp_path / "ens_f.zip", track="track_a_clean"
        )
    with pytest.raises(ValueError, match="track"):
        run_stage2_v2_ensemble_eval(
            [benign_arm] * 4, v_loader_s, t_loader_s, output_dir=tmp_path / "ens_b", zip_path=tmp_path / "ens_b.zip", track="invalid_track"
        )


def test_r2_blend_config_and_notebook_wiring() -> None:
    """Verify r2_blend defaults to 1.0, validates [0,1]/finite, and is wired in 02b notebook."""
    from src.stage2_v2.config import resolve_stage2_v2_config_from_env

    cfg_default = resolve_stage2_v2_config_from_env(environ={"STAGE2_EXP_PRESET": "arm5_dinov2_large_r2"})
    assert cfg_default.r2_blend == pytest.approx(1.0)
    assert cfg_default.to_dict()["r2_blend"] == pytest.approx(1.0)

    cfg_env = resolve_stage2_v2_config_from_env(
        environ={"STAGE2_EXP_PRESET": "arm6_segformer_b5_r2", "STAGE2_R2_BLEND": "0.85"}
    )
    assert cfg_env.r2_blend == pytest.approx(0.85)

    for bad in ("-0.1", "1.2", "nan", "inf"):
        with pytest.raises(ValueError, match="STAGE2_R2_BLEND"):
            resolve_stage2_v2_config_from_env(environ={"STAGE2_EXP_PRESET": "arm5_dinov2_large_r2", "STAGE2_R2_BLEND": bad})

    with pytest.raises(ValueError, match="r2_blend"):
        resolve_stage2_v2_config_from_env(environ={}, overrides={"r2_blend": float("nan")})

    nb_path = Path("notebooks/02b_Stage2_Breakthrough_G4.ipynb")
    nb_obj = json.loads(nb_path.read_text(encoding="utf-8"))
    cell_src = "\n".join("".join(c.get("source", [])) for c in nb_obj.get("cells", []))
    assert "blend=cfg.r2_blend" in cell_src
    assert "[R2 TELEMETRY]" in cell_src
    assert "blend=0.5" not in cell_src
