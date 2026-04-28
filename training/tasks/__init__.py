"""Gymnasium registration for aniML tasks.

Import this module (or ``import training.tasks``) before invoking Isaac Lab's
``rsl_rl/train.py`` so ``--task AniML-Deer-Locomotion-Rough-v0`` resolves.
"""

from __future__ import annotations

import gymnasium as gym

_ANYMAL_AGENTS = "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents"

gym.register(
    id="AniML-Deer-Locomotion-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_locomotion_env_cfg:DeerLocomotionRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_rsl_rl_cfg:DeerLocomotionRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{_ANYMAL_AGENTS}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerWithSymmetryCfg"
        ),
        "rl_games_cfg_entry_point": f"{_ANYMAL_AGENTS}:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_ANYMAL_AGENTS}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="AniML-Deer-USD-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_rough_env_cfg:DeerUsdRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_rsl_rl_cfg:DeerUsdRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{_ANYMAL_AGENTS}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerWithSymmetryCfg"
        ),
        "rl_games_cfg_entry_point": f"{_ANYMAL_AGENTS}:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_ANYMAL_AGENTS}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="AniML-Deer-USD-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_rough_env_cfg:DeerUsdFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_rsl_rl_cfg:DeerUsdFlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{_ANYMAL_AGENTS}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerWithSymmetryCfg"
        ),
        "rl_games_cfg_entry_point": f"{_ANYMAL_AGENTS}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_ANYMAL_AGENTS}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="AniML-Deer-USD-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_rough_env_cfg:DeerUsdStandEnvCfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_rsl_rl_cfg:DeerUsdStandPPORunnerCfg",
    },
)

gym.register(
    id="AniML-Deer-USD-Stand-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_rough_env_cfg:DeerUsdStandEnvCfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_rsl_rl_cfg:DeerUsdStandV2PPORunnerCfg",
    },
)

gym.register(
    id="AniML-Deer-USD-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_rough_env_cfg:DeerUsdWalkEnvCfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_rsl_rl_cfg:DeerUsdWalkPPORunnerCfg",
    },
)

# AMP (Adversarial Motion Priors) walk task. Uses amp-rsl-rl's
# AMPOnPolicyRunner; launch via training/scripts/train_amp.py (NOT Isaac Lab's
# default rsl_rl train.py, which doesn't know about AMP_PPO).
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfg",
        # Entry point used by our custom AMP driver; not consumed by Isaac
        # Lab's stock rsl_rl script.
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG",
    },
)

# Run 2: same AMP setup but env disables standing-reward shortcuts, and the
# runner cfg warm-starts from run 1 via --resume_from on the CLI.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV2",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V2",
    },
)

# Run 3: same env as v2 but AMP reward scale dropped to 0.3 so task reward
# (velocity tracking) can compete with the style reward. Warm-start from v2.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV2",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V3",
    },
)

# Run 5 (v4 task id): AMP with reference-state initialization, no pelvis
# termination, linear forward-velocity reward, and 5s episodes.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV3",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V4",
    },
)

# Run 6 (v5 task id): anatomy-aligned rewards (body -Z forward), orientation
# termination, re-baked RSI (world -Y forward). Trained from scratch because
# prior policies learned to produce body-+X velocity (anatomical-lateral),
# fundamentally incompatible with this new objective.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV4",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V5",
    },
)

# Run 7 (v6 task): consolidate run 6's iter-800 walking gait with lower
# entropy so noise std doesn't ballon and kill it.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV4",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V6",
    },
)

# Run 9: action_scale 1.0 + entropy_coef 0.0
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV5",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V7",
    },
)

# Run 10: warm-start r9 + small entropy + leg-joint-activity reward
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV6",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V8",
    },
)

# Run 11: soft spine/neck Kp + leg_activity weight 2.5 + entropy 0.005
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v9",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV6",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V9",
    },
)

# Run 12: anti-belly-slide. Tilt 35°, pelvis_too_low, upright_bonus,
# discriminator [64,64], entropy 0.002, 2500 envs. Warm-start r11b.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV7",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V10",
    },
)

# Run 13: anti-crouch-shuffle. Linear pelvis_height_ramp, nose_pitch_penalty,
# pelvis_too_low tightened to 0.70m, entropy 0.001. Warm-start r12.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV8",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V11",
    },
)

# Run 14: attack neck-bend cheat. Drop nose_pitch_penalty, add head_height_ramp
# on skull body (z_min=0.60, z_max=1.20). Warm-start r13.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v12",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV9",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG_V12",
    },
)

# Run 16: shuffle->walk fix. Inherits base v0 env (clean reward set, lean
# config that gave us the breakthrough in r15) but reduces action_rate_l2 +
# dof_torques_l2 weights so the policy can swing legs through the action
# space instead of minimizing change. Reuses v0 agent cfg (init_noise_std=1.0).
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v13",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV10",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG",
    },
)

# Run 19: zero out the tiny-weight L2 smoothness penalties (dof_acc_l2,
# dof_torques_l2) that overflow when widened URDF limits push joint state
# through occasional PhysX numeric spikes. Used with widened URDF/USD/.npy.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v14",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV11",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_CFG",
    },
)
