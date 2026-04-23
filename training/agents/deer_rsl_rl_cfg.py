"""rsl_rl PPO runner for deer locomotion (inherits ANYmal rough hyperparameters)."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)


@configclass
class DeerLocomotionRoughPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """Same MLP [512, 256, 128] as ANYmal rough; distinct experiment name for aniML logs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_locomotion_rough"
