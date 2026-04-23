"""Species-level constants for wolf (Phase 2 ecosystem)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WolfSpeciesCfg:
    name: str = "wolf"
    total_joints: int = 12
    locomotion_obs_dim: int = 148
    ecosystem_obs_dim: int = 159
    action_dim: int = 12


def wolf_yaml_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "species" / "wolf.yaml"
