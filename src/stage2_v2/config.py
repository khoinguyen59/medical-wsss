"""
Stage-2 v2 Experiment Presets & Environment Resolution for BCSS-WSSS.

Defines immutable, reproducible configuration presets for the 4-arm Stage-2
Gate-5 breakthrough suite:
- arm1_deeplab_cleankd: DeepLabV3+ R101 + PolySGD + Clean bn7 Soft-KD + Dice + MIL
- arm2_segformer_oeem: SegFormer-B3 Dual-Head + AdamW + KL-Soft OEEM + Dice + MIL
- arm3_convnext_selftrain: ConvNeXt-Base FPN + AdamW + Soft-SCE + Round-2 EMA Self-Train
- arm4_dinov2_adapter: DINOv2-ViT-B/14 Adapter + AdamW + KL-Soft OEEM + Dice + MIL

All hyphens strictly use standard ASCII 0x2D ('-').
"""

import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

VALID_SOFT_MODES: Tuple[str, ...] = ("raw_l1", "softmax_t", "sg_dap_soft")
VALID_LOSS_MODES: Tuple[str, ...] = ("soft_kd_dice_mil", "kl_oeem_dice_mil", "soft_sce_dice_mil")


@dataclass(frozen=True)
class Stage2V2Config:
    """Immutable configuration contract for a Stage-2 v2 experimental arm."""

    preset_name: str = "arm1_deeplab_cleankd"
    arch: str = "deeplabv3plus_r101"
    backbone_id: str = "resnet101"
    num_classes: int = 4
    unfreeze_last_n_blocks: int = 0
    optimizer: str = "sgd"
    lr_backbone: float = 1e-3
    lr_head: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "poly"
    poly_power: float = 0.9
    eta_min: float = 1e-6
    warmup_epochs: int = 1
    epochs: int = 20
    patience: int = 6
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    loss_mode: str = "soft_kd_dice_mil"
    sce_alpha: float = 1.0
    sce_beta: float = 0.15
    lambda_dice: float = 0.3
    lambda_mil: float = 0.25
    lambda_cls: float = 0.15
    use_oeem: bool = False
    oeem_warmup_epochs: int = 2
    oeem_tau: float = 1.5
    use_ema: bool = True
    ema_decay: float = 0.995
    self_train_round2_start_epoch: Optional[int] = None
    self_train_blend: float = 0.5
    self_train_lr_scale: float = 0.3
    soft_mode: str = "softmax_t"
    soft_temp: float = 0.25
    dry_run: bool = False
    git_ref: str = ""
    eval_scales: Tuple[float, ...] = (0.75, 1.0, 1.25)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain JSON-friendly dict."""
        data = asdict(self)
        data["eval_scales"] = list(self.eval_scales)
        return data


STAGE2_V2_PRESETS: Dict[str, Stage2V2Config] = {
    "arm1_deeplab_cleankd": Stage2V2Config(
        preset_name="arm1_deeplab_cleankd",
        arch="deeplabv3plus_r101",
        backbone_id="resnet101",
        num_classes=4,
        unfreeze_last_n_blocks=0,
        optimizer="sgd",
        lr_backbone=1e-3,
        lr_head=1e-2,
        momentum=0.9,
        weight_decay=5e-4,
        scheduler="poly",
        poly_power=0.9,
        warmup_epochs=1,
        epochs=20,
        patience=6,
        batch_size=32,
        loss_mode="soft_kd_dice_mil",
        lambda_dice=0.3,
        lambda_mil=0.25,
        lambda_cls=0.15,
        use_oeem=False,
        use_ema=True,
        ema_decay=0.995,
    ),
    "arm2_segformer_oeem": Stage2V2Config(
        preset_name="arm2_segformer_oeem",
        arch="segformer_b3",
        backbone_id="nvidia/mit-b3",
        num_classes=4,
        unfreeze_last_n_blocks=0,
        optimizer="adamw",
        lr_backbone=6e-5,
        lr_head=6e-4,
        weight_decay=0.01,
        scheduler="cosine",
        warmup_epochs=1,
        epochs=18,
        patience=6,
        batch_size=32,
        loss_mode="kl_oeem_dice_mil",
        use_oeem=True,
        oeem_warmup_epochs=2,
        oeem_tau=1.5,
        lambda_dice=0.3,
        lambda_mil=0.20,
        lambda_cls=0.20,
        use_ema=True,
        ema_decay=0.995,
    ),
    "arm3_convnext_selftrain": Stage2V2Config(
        preset_name="arm3_convnext_selftrain",
        arch="convnext_base_fpn",
        backbone_id="convnext_base",
        num_classes=4,
        unfreeze_last_n_blocks=0,
        optimizer="adamw",
        lr_backbone=1e-4,
        lr_head=5e-4,
        weight_decay=0.05,
        scheduler="cosine",
        warmup_epochs=1,
        epochs=18,
        patience=6,
        batch_size=32,
        loss_mode="soft_sce_dice_mil",
        sce_alpha=1.0,
        sce_beta=0.15,
        lambda_dice=0.3,
        lambda_mil=0.20,
        lambda_cls=0.20,
        use_oeem=False,
        self_train_round2_start_epoch=10,
        self_train_blend=0.5,
        self_train_lr_scale=0.3,
        use_ema=True,
        ema_decay=0.995,
    ),
    "arm4_dinov2_adapter": Stage2V2Config(
        preset_name="arm4_dinov2_adapter",
        arch="dinov2_vitb14_adapter",
        backbone_id="facebook/dinov2-base",
        num_classes=4,
        unfreeze_last_n_blocks=4,
        optimizer="adamw",
        lr_backbone=2e-5,
        lr_head=5e-4,
        weight_decay=0.01,
        scheduler="cosine",
        warmup_epochs=1,
        epochs=15,
        patience=5,
        batch_size=32,
        loss_mode="kl_oeem_dice_mil",
        use_oeem=True,
        oeem_warmup_epochs=2,
        oeem_tau=1.5,
        lambda_dice=0.3,
        lambda_mil=0.20,
        lambda_cls=0.20,
        use_ema=True,
        ema_decay=0.995,
    ),
}


def get_stage2_v2_preset(name: str = "arm1_deeplab_cleankd") -> Stage2V2Config:
    """Return a validated Stage2V2Config preset by name."""
    key = str(name or "arm1_deeplab_cleankd").strip().lower()
    if key not in STAGE2_V2_PRESETS:
        valid = ", ".join(sorted(STAGE2_V2_PRESETS.keys()))
        raise ValueError(f"Unknown Stage-2 v2 preset {name!r}. Valid presets: {valid}.")
    return STAGE2_V2_PRESETS[key]


def resolve_stage2_v2_config_from_env(
    overrides: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Stage2V2Config:
    """Resolve Stage2V2Config from environment variables and optional overrides.

    Supported environment variables:
    - STAGE2_EXP_PRESET / STAGE2_ARM: preset key
    - STAGE2_SEED: integer random seed override
    - STAGE2_EPOCHS: integer epochs override
    - STAGE2_BATCH_SIZE: integer batch size override
    - STAGE2_NUM_WORKERS: integer dataloader worker count override
    - STAGE2_SOFT_MODE: 'raw_l1' | 'softmax_t' | 'sg_dap_soft'
    - STAGE2_SOFT_TEMP: positive float temperature (default 0.25)
    - STAGE2_DRY_RUN: '1'/'true' vs '0'/'false'
    - STAGE2_GIT_REF: optional git ref metadata string
    """
    env = os.environ if environ is None else environ
    override_dict = dict(overrides) if overrides is not None else {}

    preset_raw = override_dict.pop("preset_name", None)
    if preset_raw is None:
        preset_raw = env.get("STAGE2_EXP_PRESET") or env.get("STAGE2_ARM") or "arm1_deeplab_cleankd"
    cfg = get_stage2_v2_preset(str(preset_raw))

    updates: Dict[str, Any] = {}

    def _read_int(env_key: str, field_name: str, min_val: int = 0) -> None:
        raw = env.get(env_key, "")
        if raw is not None and str(raw).strip() != "":
            val = int(str(raw).strip())
            if val < min_val:
                raise ValueError(f"{env_key} must be >= {min_val}, got {val}.")
            updates[field_name] = val

    _read_int("STAGE2_SEED", "seed", min_val=0)
    _read_int("STAGE2_EPOCHS", "epochs", min_val=1)
    _read_int("STAGE2_BATCH_SIZE", "batch_size", min_val=1)
    _read_int("STAGE2_NUM_WORKERS", "num_workers", min_val=0)

    raw_soft_mode = env.get("STAGE2_SOFT_MODE", "")
    if raw_soft_mode is not None and str(raw_soft_mode).strip() != "":
        sm_val = str(raw_soft_mode).strip().lower()
        if sm_val not in VALID_SOFT_MODES:
            raise ValueError(
                f"Unsupported STAGE2_SOFT_MODE={raw_soft_mode!r}. Expected one of {VALID_SOFT_MODES}."
            )
        updates["soft_mode"] = sm_val

    raw_soft_temp = env.get("STAGE2_SOFT_TEMP", "")
    if raw_soft_temp is not None and str(raw_soft_temp).strip() != "":
        st_val = float(str(raw_soft_temp).strip())
        if st_val <= 0.0:
            raise ValueError(f"STAGE2_SOFT_TEMP must be > 0, got {st_val}.")
        updates["soft_temp"] = st_val

    raw_dry = env.get("STAGE2_DRY_RUN", "")
    if raw_dry is not None and str(raw_dry).strip() != "":
        dry_s = str(raw_dry).strip().lower()
        if dry_s in ("1", "true", "yes"):
            updates["dry_run"] = True
        elif dry_s in ("0", "false", "no"):
            updates["dry_run"] = False
        else:
            raise ValueError(f"Unsupported STAGE2_DRY_RUN={raw_dry!r}.")

    raw_git = env.get("STAGE2_GIT_REF", "")
    if raw_git is not None and str(raw_git).strip() != "":
        updates["git_ref"] = str(raw_git).strip()

    for k, v in override_dict.items():
        if v is not None:
            if k == "eval_scales" and not isinstance(v, tuple):
                v = tuple(float(s) for s in v)
            updates[k] = v

    if updates:
        cfg = replace(cfg, **updates)
    return cfg
