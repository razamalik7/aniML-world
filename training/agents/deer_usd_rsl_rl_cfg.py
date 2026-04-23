"""PPO experiment names for deer USD tasks (separate from ANYmal stand-in logs)."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)


@configclass
class DeerUsdRoughPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_usd_rough"


@configclass
class DeerUsdFlatPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_usd_flat"
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.1,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )


@configclass
class DeerUsdStandPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """Phase-0 standing: near-zero init noise, zero entropy, conservative LR."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_stand_p0"
        self.max_iterations = 500
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.05,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
        )


@configclass
class DeerUsdStandV2PPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """Standing with the SAME network architecture as walk ([512,256,128]).

    Identical rewards to the best standing run (2026-04-10_15-08-53) but with
    the larger network so the checkpoint can be directly warm-started into
    DeerUsdWalkPPORunnerCfg without architecture mismatch.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_stand_v2"
        self.max_iterations = 500
        self.save_interval = 50
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.05,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
        )


@configclass
class DeerUsdWalkPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """Phase-1 walking: warm-started from CoM-fixed standing policy.

    Strategy (Option A):
      - Load standing policy weights (knows how to stay upright).
      - std_param reset to 0.4 in the patched checkpoint (tools/patch_ckpt_for_walk.py)
        so PPO has real exploration capacity from step 1.
      - Critic and optimizer cleared in the patch so stale standing value estimates
        don't corrupt early walk gradients.
      - init_noise_std=0.4 must match the patched checkpoint std or rsl_rl will
        overwrite the loaded std on the first forward pass.

    Expected milestones:
      ~100 iters:  policy reliably stands (already learned, just adapting to walk env)
      ~400 iters:  shuffling/trotting gait visible
      ~800 iters:  stable walk
      ~1200 iters: consistent trot, raise feet_air_time to 0.05 then 0.125
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_walk_p1"
        self.max_iterations = 3000
        self.save_interval = 100
        # Phase 3: resume from iter 1000 (training was interrupted).
        # velocity_error_l2 and stationary_penalty active to break standing-still optimum.
        self.resume = True
        self.load_run = "2026-04-14_16-34-53"
        self.load_checkpoint = "model_1000.pt"
        # Replicates 2026-04-13_17-03-57 exactly — the best walk run we have.
        # That run loaded model_800 which had std=0.10 after 800 walk iters.
        # rsl_rl 4.0 overrides checkpoint std with init_noise_std, so we set 0.10
        # directly. entropy_coef=0.001 grows std slowly as walking needs it.
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.1,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,     # matches 17-03-57 exactly — slow std growth
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
        )
