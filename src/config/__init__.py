"""Configuration package for SE374.R11 WSSS pipeline."""

from src.config.stage1_experiments import (
    STAGE1_PRESETS,
    Stage1ExperimentConfig,
    get_stage1_preset,
    resolve_stage1_config_from_env,
)

__all__ = [
    "STAGE1_PRESETS",
    "Stage1ExperimentConfig",
    "get_stage1_preset",
    "resolve_stage1_config_from_env",
]
