"""PPO runner configs for the post-AMP deer locomotion stages."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)

# v4.7: mirror-symmetry augmentation. Doubles the PPO training batch with
# L<->R-mirrored copies, forcing the network to be equivariant under
# bilateral reflection. Targets the residual front-leg L/R asymmetry that
# v4.6's per-step reward couldn't fully kill.
from training.envs.deer_mirror_aug import mirror_obs_actions


_NETWORK = RslRlPpoActorCriticCfg(
    # init_noise_std=1.0 was too high for the 21-DOF deer — actions in 21 dims
    # at 1σ each were destabilizing balance immediately at fresh init. 0.5 is
    # the sweet spot that keeps exploration alive without wrecking balance.
    init_noise_std=0.5,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)


def _alg(
    entropy_coef: float,
    learning_rate: float = 1.0e-3,
    symmetry_cfg: RslRlSymmetryCfg | None = None,
) -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=entropy_coef,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=learning_rate,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=symmetry_cfg,
    )


# v4.7: symmetry config for Stage 3 mirror-aug runs.
_SYMMETRY_CFG = RslRlSymmetryCfg(
    use_data_augmentation=True,
    use_mirror_loss=False,         # data aug only; loss-form is alternative
    data_augmentation_func=mirror_obs_actions,
    mirror_loss_coeff=0.0,
)


@configclass
class DeerUsdLocoStage1PPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """Stage 1: discover trot from scratch.

    init_noise_std=1.0 + entropy 0.005 -> exploration room early; PPO
    naturally tightens std as the policy converges.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_loco_stage1"
        self.max_iterations = 2000
        self.save_interval = 100
        self.policy = _NETWORK
        self.algorithm = _alg(entropy_coef=0.005, learning_rate=1.0e-3)


@configclass
class DeerUsdLocoStage2PPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """Stage 2: full 3-DOF + pushes. Warm-start from Stage 1."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_loco_stage2"
        self.max_iterations = 2000
        self.save_interval = 100
        self.policy = _NETWORK
        self.algorithm = _alg(entropy_coef=0.002, learning_rate=5.0e-4)


@configclass
class DeerUsdLocoStage3PPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """Stage 3: terrain. Warm-start from Stage 2.

    v4.7: enables L<->R mirror-aug data augmentation via symmetry_cfg.
    The augmentation func lives in training/envs/deer_mirror_aug.py.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_loco_stage3"
        self.max_iterations = 2000
        self.save_interval = 100
        self.policy = _NETWORK
        self.algorithm = _alg(
            entropy_coef=0.001,
            learning_rate=3.0e-4,
            symmetry_cfg=_SYMMETRY_CFG,
        )


@configclass
class DeerUsdLocoStablePPORunnerCfg(DeerUsdLocoStage3PPORunnerCfg):
    """v5.0 stable-base pivot. Identical to Stage3 runner (same network,
    mirror data-augmentation, entropy) but writes to its own experiment dir
    so stable runs don't intermix with the naturalness Stage3 checkpoints.

    Mirror augmentation is retained deliberately — it is a policy-equivariance
    data aug (helps a symmetric, stable gait), NOT a naturalness reward.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_loco_stable"


@configclass
class DeerUsdLocoStableFlatPPORunnerCfg(DeerUsdLocoStage1PPORunnerCfg):
    """v5.2 FRESH stable base. Stage1 recipe (entropy 0.005, lr 1e-3, from
    scratch) + mirror symmetry augmentation (we want a symmetric stable gait).
    Own experiment dir so fresh stable runs don't mix with the Stage1
    naturalness runs.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "animl_deer_loco_stable_flat"
        self.algorithm = _alg(
            entropy_coef=0.005,
            learning_rate=1.0e-3,
            symmetry_cfg=_SYMMETRY_CFG,
        )
