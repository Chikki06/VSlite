"""Mandatory training configuration contract for all VSlite model stacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

# IR-only: 1/3/5 (Vis+ band subsets) / 6/10. Vis+: 4=Vis+1, 6=Vis+3, 8=Vis+5. Full IR+VIS: 13.
# VIS-only: 3 with num_ir_channels=0 (VIS_ONLY=1).
VALID_CHANNEL_COUNTS = frozenset({1, 3, 4, 5, 6, 8, 10, 13})
VALID_COLOR_SPACES = frozenset({"srgb", "linear"})
VALID_LR_SCHEDULES = frozenset({"fixed", "cosine", "plateau"})
VALID_OPTIMIZERS = frozenset({"adam", "adamw"})


@dataclass
class VSliteTrainConfig:
    """Required fields for every training run. Validation fails if invalid."""

    model: str
    run: str
    num_channels: int
    patch_size: int
    batch_size: int
    epochs: int
    grad_accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    dropout: float = 0.0
    he_color_space: str = "srgb"
    augment: bool = True
    aug_strength: float = 1.0
    optimizer: str = "adam"
    lr_schedule: str = "fixed"
    warmup_epochs: int = 0
    lambda_l1: float = 1.0
    lambda_nce: float = 10.0
    lambda_perc: float = 0.0
    lambda_gan: float = 1.0
    no_gan: bool = False
    ema_decay: float = 0.0
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4
    progressive_schedule: str = ""
    output_dir: str = ""
    dataroot: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size_effective(self) -> int:
        return int(self.batch_size) * max(1, int(self.grad_accum_steps))


def validate_train_config(cfg: VSliteTrainConfig) -> None:
    """Raise ValueError if required ablation fields are missing or invalid."""
    if not str(cfg.model).strip():
        raise ValueError("VSliteTrainConfig.model is required")
    if not str(cfg.run).strip():
        raise ValueError("VSliteTrainConfig.run is required")
    if int(cfg.num_channels) not in VALID_CHANNEL_COUNTS:
        raise ValueError(
            f"num_channels must be one of {sorted(VALID_CHANNEL_COUNTS)}, got {cfg.num_channels}"
        )
    if int(cfg.patch_size) < 32:
        raise ValueError(f"patch_size must be >= 32, got {cfg.patch_size}")
    if int(cfg.batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
    if int(cfg.epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg.epochs}")
    if str(cfg.he_color_space).lower() not in VALID_COLOR_SPACES:
        raise ValueError(f"he_color_space must be srgb|linear, got {cfg.he_color_space}")
    if str(cfg.lr_schedule).lower() not in VALID_LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be fixed|cosine|plateau, got {cfg.lr_schedule}")
    if str(cfg.optimizer).lower() not in VALID_OPTIMIZERS:
        raise ValueError(f"optimizer must be adam|adamw, got {cfg.optimizer}")


def config_from_env(model: str, run: str, env: Optional[Mapping[str, str]] = None) -> VSliteTrainConfig:
    """Build config from standard ablation environment variables."""
    import os

    e = dict(os.environ) if env is None else dict(env)

    def _get(name: str, default: str = "") -> str:
        return str(e.get(name, default)).strip()

    def _int(name: str, default: int) -> int:
        v = _get(name)
        return int(v) if v else default

    def _float(name: str, default: float) -> float:
        v = _get(name)
        return float(v) if v else default

    def _bool(name: str, default: bool = False) -> bool:
        v = _get(name).lower()
        if not v:
            return default
        return v in {"1", "true", "yes", "y", "on"}

    num_ch = _int("NUM_CHANNELS", _int("INPUT_NC", 13))
    return VSliteTrainConfig(
        model=model,
        run=run,
        num_channels=num_ch,
        patch_size=_int("PATCH_SIZE", _int("IMG_SIZE", 512)),
        batch_size=_int("BATCH_SIZE", 8),
        grad_accum_steps=_int("GRAD_ACCUM_STEPS", 1),
        epochs=_int("EPOCHS", 50),
        lr=_float("LR", 2e-4),
        weight_decay=_float("WEIGHT_DECAY", 0.0),
        dropout=_float("DROPOUT", 0.0),
        he_color_space=_get("HE_COLOR_SPACE", "srgb").lower() or "srgb",
        augment=not _bool("NO_AUG"),
        aug_strength=_float("AUG_STRENGTH", 1.0),
        optimizer=_get("OPTIMIZER", "adam").lower() or "adam",
        lr_schedule=_get("LR_SCHEDULE", "fixed").lower() or "fixed",
        warmup_epochs=_int("WARMUP_EPOCHS", 0),
        lambda_l1=_float("LAMBDA_L1", 1.0),
        lambda_nce=_float("LAMBDA_NCE", 10.0),
        lambda_perc=_float("LAMBDA_PERC", 0.0),
        lambda_gan=_float("LAMBDA_GAN", 1.0),
        no_gan=_bool("NO_GAN"),
        ema_decay=_float("EMA_DECAY", 0.0),
        early_stop_patience=_int("EARLY_STOP_PATIENCE", 0),
        early_stop_min_delta=_float("EARLY_STOP_MIN_DELTA", 1e-4),
        progressive_schedule=_get("PROGRESSIVE_SCHEDULE"),
        output_dir=_get("OUTPUT_DIR"),
        dataroot=_get("DATAROOT", _get("VS_DATAROOT")),
    )


@runtime_checkable
class VSliteTrainingHooks(Protocol):
    """Optional protocol documenting expected trainer responsibilities."""

    def eval_l1(self, loader: Any) -> float: ...
    def save_checkpoint(self, kind: str) -> None: ...
    def forward_preview(self, fixed_batch: Any) -> tuple[Any, Any]: ...
