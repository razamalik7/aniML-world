"""Species-level constants (mirror ``configs/species/deer.yaml`` for Python access)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeerSpeciesCfg:
    name: str = "deer"
    total_joints: int = 12
    locomotion_obs_dim: int = 148
    ecosystem_obs_dim: int = 159
    action_dim: int = 12


def deer_yaml_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "species" / "deer.yaml"
