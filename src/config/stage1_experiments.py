"""
Stage-1 Experiment Presets & Environment Resolution for SE374.R11.

Defines versioned, reproducible experimental configurations for the 3-arm
controlled Stage-1 scientific benchmark:
- default: Backward-compatible AdamW baseline with multiplicative PDA + EMA.
- arm_a_polysgd: PolySGD + Label-Gated Multiplicative PDA + Full D4 + EMA.
- arm_b_equiv_adamw: Calibrated AdamW + Label-Gated PDA + Full D4 + Scale/D4 Equivariant CAM Consistency + EMA.
- arm_c_mpppc_polysgd: Arm A PolySGD base + training-only ProjectionHead + warm-started MP-PPC(K=1).

Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Stage1ExperimentConfig:
    """Immutable configuration contract for a Stage-1 experimental arm."""

    preset_name: str = "default"
    optimizer_type: str = "adamw"          # "adamw" | "polysgd"
    scheduler_type: str = "cosine"         # "cosine" | "poly"
    lr_bb: float = 1.4e-4
    lr_fc8: float = 1.4e-3
    wd_bb: float = 1e-4
    wd_fc8: float = 1e-4
    momentum: float = 0.9
    warmup_epochs: float = 0.0
    poly_power: float = 0.9
    eta_min: float = 1e-6
    batch_size: int = 32
    max_epochs: int = 25
    stop_after_epoch: Optional[int] = None
    seed: int = 42
    num_workers: int = 2
    input_norm: str = "zero_one"
    loss_type: str = "soft_margin"
    use_d4_aug: bool = False
    pda_mode: str = "multiplicative"       # "mean" | "multiplicative"
    pda_start_epoch: int = 3               # 0-indexed (3 -> Epoch 4)
    initial_gamma: float = 1.0
    gamma_decay: float = 0.985
    min_gamma: float = 0.65
    use_ema: bool = True
    ema_decay: float = 0.995
    use_equiv_consistency: bool = False
    equiv_scales: Tuple[float, ...] = (1.0, 1.25)
    lambda_eq_max: float = 0.0
    eq_ramp_start_epoch: int = 2           # 0-indexed (2 -> Epoch 3)
    eq_ramp_end_epoch: int = 5             # 0-indexed (5 -> Epoch 6)
    use_mpppc: bool = False
    lambda_ppc: float = 0.0
    ppc_start_epoch: int = 3               # 0-indexed (3 -> Epoch 4)
    ppc_num_prototypes: int = 1
    ppc_conf_thresh: float = 0.70
    min_epochs_early_stop: int = 12
    early_stop_patience: int = 8
    git_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain JSON-friendly dict."""
        data = asdict(self)
        data["equiv_scales"] = list(self.equiv_scales)
        return data


STAGE1_PRESETS: Dict[str, Stage1ExperimentConfig] = {
    "default": Stage1ExperimentConfig(
        preset_name="default",
        optimizer_type="adamw",
        scheduler_type="cosine",
        lr_bb=1.4e-4,
        lr_fc8=1.4e-3,
        wd_bb=1e-4,
        wd_fc8=1e-4,
        momentum=0.9,
        warmup_epochs=0.0,
        poly_power=0.9,
        eta_min=1e-6,
        batch_size=32,
        max_epochs=25,
        use_d4_aug=False,
        pda_mode="multiplicative",
        pda_start_epoch=3,
        initial_gamma=1.0,
        gamma_decay=0.985,
        min_gamma=0.65,
        use_ema=True,
        ema_decay=0.995,
        use_equiv_consistency=False,
        use_mpppc=False,
        min_epochs_early_stop=12,
        early_stop_patience=8,
    ),
    "arm_a_polysgd": Stage1ExperimentConfig(
        preset_name="arm_a_polysgd",
        optimizer_type="polysgd",
        scheduler_type="poly",
        lr_bb=1.5e-3,
        lr_fc8=1.5e-2,
        wd_bb=5e-4,
        wd_fc8=5e-4,
        momentum=0.9,
        warmup_epochs=0.5,
        poly_power=0.9,
        eta_min=0.0,
        batch_size=32,
        max_epochs=20,
        use_d4_aug=True,
        pda_mode="multiplicative",
        pda_start_epoch=3,
        initial_gamma=1.0,
        gamma_decay=0.98,
        min_gamma=0.65,
        use_ema=True,
        ema_decay=0.995,
        use_equiv_consistency=False,
        use_mpppc=False,
        min_epochs_early_stop=12,
        early_stop_patience=8,
    ),
    "arm_b_equiv_adamw": Stage1ExperimentConfig(
        preset_name="arm_b_equiv_adamw",
        optimizer_type="adamw",
        scheduler_type="cosine",
        lr_bb=8e-5,
        lr_fc8=2.4e-4,
        wd_bb=1e-3,
        wd_fc8=5e-2,
        momentum=0.9,
        warmup_epochs=2.0,
        poly_power=0.9,
        eta_min=1e-6,
        batch_size=32,
        max_epochs=18,
        use_d4_aug=True,
        pda_mode="multiplicative",
        pda_start_epoch=3,
        initial_gamma=1.0,
        gamma_decay=0.98,
        min_gamma=0.65,
        use_ema=True,
        ema_decay=0.995,
        use_equiv_consistency=True,
        equiv_scales=(1.0, 1.25),
        lambda_eq_max=0.25,
        eq_ramp_start_epoch=2,
        eq_ramp_end_epoch=5,
        use_mpppc=False,
        min_epochs_early_stop=12,
        early_stop_patience=8,
    ),
    "arm_c_mpppc_polysgd": Stage1ExperimentConfig(
        preset_name="arm_c_mpppc_polysgd",
        optimizer_type="polysgd",
        scheduler_type="poly",
        lr_bb=1.5e-3,
        lr_fc8=1.5e-2,
        wd_bb=5e-4,
        wd_fc8=5e-4,
        momentum=0.9,
        warmup_epochs=0.5,
        poly_power=0.9,
        eta_min=0.0,
        batch_size=32,
        max_epochs=20,
        use_d4_aug=True,
        pda_mode="multiplicative",
        pda_start_epoch=3,
        initial_gamma=1.0,
        gamma_decay=0.98,
        min_gamma=0.65,
        use_ema=True,
        ema_decay=0.995,
        use_equiv_consistency=False,
        use_mpppc=True,
        lambda_ppc=0.10,
        ppc_start_epoch=3,
        ppc_num_prototypes=1,
        ppc_conf_thresh=0.70,
        min_epochs_early_stop=12,
        early_stop_patience=8,
    ),
}


def get_stage1_preset(name: str = "default") -> Stage1ExperimentConfig:
    """Return a validated Stage1ExperimentConfig preset by name."""
    key = str(name or "default").strip().lower()
    if key not in STAGE1_PRESETS:
        valid = ", ".join(sorted(STAGE1_PRESETS.keys()))
        raise ValueError(f"Unknown Stage-1 preset {name!r}. Valid presets: {valid}.")
    return STAGE1_PRESETS[key]


def resolve_stage1_config_from_env(
    overrides: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Stage1ExperimentConfig:
    """Resolve Stage1ExperimentConfig from environment variables and optional overrides.

    Environment variables supported:
    - STAGE1_EXP_PRESET: preset key (default, arm_a_polysgd, arm_b_equiv_adamw, arm_c_mpppc_polysgd)
    - STAGE1_MAX_EPOCHS: integer max epochs override (full LR schedule horizon)
    - STAGE1_STOP_AFTER_EPOCH: optional 1-indexed integer epoch to stop cleanly after
    - STAGE1_SEED: integer random seed override
    - STAGE1_BATCH_SIZE: integer batch size override
    - STAGE1_INPUT_NORM: 'zero_one' or 'imagenet'
    - STAGE1_LOSS_TYPE: 'soft_margin', 'asl', or 'bce'
    - STAGE1_GIT_REF: optional git ref metadata string
    - STAGE1_NUM_WORKERS: integer dataloader worker override
    """
    env = os.environ if environ is None else environ
    override_dict = dict(overrides) if overrides is not None else {}

    preset_raw = override_dict.pop("preset_name", None)
    if preset_raw is None:
        preset_raw = env.get("STAGE1_EXP_PRESET", "default")
    cfg = get_stage1_preset(str(preset_raw or "default"))

    updates: Dict[str, Any] = {}

    def _read_int(env_key: str, field_name: str, min_val: int = 0) -> None:
        raw = env.get(env_key, "")
        if raw is not None and str(raw).strip() != "":
            val = int(str(raw).strip())
            if val < min_val:
                raise ValueError(f"{env_key} must be >= {min_val}, got {val}.")
            updates[field_name] = val

    _read_int("STAGE1_MAX_EPOCHS", "max_epochs", min_val=1)
    _read_int("STAGE1_STOP_AFTER_EPOCH", "stop_after_epoch", min_val=1)
    _read_int("STAGE1_SEED", "seed", min_val=0)
    _read_int("STAGE1_BATCH_SIZE", "batch_size", min_val=1)
    _read_int("STAGE1_NUM_WORKERS", "num_workers", min_val=0)

    raw_norm = env.get("STAGE1_INPUT_NORM", "")
    if raw_norm is not None and str(raw_norm).strip() != "":
        norm_val = str(raw_norm).strip().lower()
        if norm_val not in ("zero_one", "imagenet"):
            raise ValueError(f"Unsupported STAGE1_INPUT_NORM={raw_norm!r}.")
        updates["input_norm"] = norm_val

    raw_loss = env.get("STAGE1_LOSS_TYPE", "")
    if raw_loss is not None and str(raw_loss).strip() != "":
        loss_val = str(raw_loss).strip().lower()
        if loss_val not in ("soft_margin", "asl", "bce"):
            raise ValueError(f"Unsupported STAGE1_LOSS_TYPE={raw_loss!r}.")
        updates["loss_type"] = loss_val

    raw_git = env.get("STAGE1_GIT_REF", "")
    if raw_git is not None and str(raw_git).strip() != "":
        updates["git_ref"] = str(raw_git).strip()

    for k, v in override_dict.items():
        if v is not None:
            if k == "equiv_scales" and not isinstance(v, tuple):
                v = tuple(float(s) for s in v)
            updates[k] = v

    if updates:
        cfg = replace(cfg, **updates)
    return cfg
