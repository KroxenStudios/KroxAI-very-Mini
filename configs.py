from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    n_layers: int
    n_heads: int
    ff_hidden: int
    max_len: int


PRESETS: Dict[str, ModelConfig] = {
    # lightweight for tests / CPU
    "tiny": ModelConfig(dim=64, n_layers=2, n_heads=4, ff_hidden=128, max_len=128),
    "small": ModelConfig(dim=128, n_layers=4, n_heads=4, ff_hidden=256, max_len=256),
    # suitable starting point for training on mid-range GPUs
    "base": ModelConfig(dim=256, n_layers=8, n_heads=8, ff_hidden=1024, max_len=512),
    # larger, needs more VRAM
    "large": ModelConfig(dim=512, n_layers=12, n_heads=8, ff_hidden=2048, max_len=1024),
    # long-context variants (require training with these lengths)
    "small_long": ModelConfig(dim=128, n_layers=4, n_heads=4, ff_hidden=256, max_len=1024),
    "base_long": ModelConfig(dim=256, n_layers=8, n_heads=8, ff_hidden=1024, max_len=2048),
    "large_long": ModelConfig(dim=512, n_layers=12, n_heads=8, ff_hidden=2048, max_len=4096),
}


def get_preset(name: str, default: str = "small") -> ModelConfig:
    return PRESETS.get(name, PRESETS[default])
