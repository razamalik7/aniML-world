"""Agent / RL configs (Isaac Lab Kit Python required when classes are imported)."""

from __future__ import annotations

from typing import Any

__all__ = ["DeerLocomotionRoughPPORunnerCfg"]


def __getattr__(name: str) -> Any:
    if name == "DeerLocomotionRoughPPORunnerCfg":
        from training.agents.deer_rsl_rl_cfg import DeerLocomotionRoughPPORunnerCfg

        return DeerLocomotionRoughPPORunnerCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
