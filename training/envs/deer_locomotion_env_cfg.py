"""Deer locomotion target environment.

Currently subclasses Isaac Lab's rough ANYmal-C velocity task (same observation/action
structure as quadruped locomotion). Replace the scene robot with your deer USD and align
joint naming in actions/observations when assets are ready (Phase 1).
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
    AnymalCRoughEnvCfg,
)


# Default parallel envs: tuned for ~8GB laptop VRAM with headroom (was 1024, ~<5GB observed).
# Push higher on 16GB+ GPUs or override CLI: --num_envs 2048 (may OOM on 8GB).
_DEFAULT_NUM_ENVS = 1500


@configclass
class DeerLocomotionRoughEnvCfg(AnymalCRoughEnvCfg):
    """Rough ANYmal-C MDP; ``num_envs`` below balances throughput vs VRAM on 8GB class GPUs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = _DEFAULT_NUM_ENVS
