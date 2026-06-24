"""Configuration management using Pydantic settings and YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Training hyperparameters loaded from YAML config files."""

    # Model
    model_name: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.3
    freeze_backbone_epochs: int = 2

    # Data
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    seed: int = 42

    # Training
    epochs: int = 50
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine_with_warmup"
    warmup_epochs: int = 3
    early_stopping_patience: int = 8
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    label_smoothing: float = 0.05

    # Logging
    use_wandb: bool = False
    wandb_project: str = "deepfake-recognition"
    save_top_k: int = 3
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load config from a YAML file, with defaults for missing keys."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Flatten nested YAML structure into flat dict
        flat: dict[str, Any] = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)

        return cls(**{k: v for k, v in flat.items() if k in cls.model_fields})


def get_project_root() -> Path:
    """Return the project root directory (where pyproject.toml lives)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def get_device() -> str:
    """Return the best available device string."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
