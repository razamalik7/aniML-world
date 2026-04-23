"""Env configs (Isaac Lab + isaaclab_tasks required when classes are imported)."""

from __future__ import annotations

from typing import Any

__all__ = ["DeerLocomotionRoughEnvCfg"]


def __getattr__(name: str) -> Any:
    if name == "DeerLocomotionRoughEnvCfg":
        from training.envs.deer_locomotion_env_cfg import DeerLocomotionRoughEnvCfg

        return DeerLocomotionRoughEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
