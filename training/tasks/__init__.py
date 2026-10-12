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

# ---------------------------------------------------------------------------
# Pure reward-shaping locomotion tasks (URDF v2 21-DOF, post-AMP).
# Three-stage curriculum: discover trot -> 3DOF + pushes -> rough terrain.
# ---------------------------------------------------------------------------
gym.register(
    id="AniML-Deer-USD-Loco-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_loco_env_cfg:DeerUsdLocoStage1Cfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_loco_cfg:DeerUsdLocoStage1PPORunnerCfg",
    },
)
gym.register(
    id="AniML-Deer-USD-Loco-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_loco_env_cfg:DeerUsdLocoStage2Cfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_loco_cfg:DeerUsdLocoStage2PPORunnerCfg",
    },
)
gym.register(
    id="AniML-Deer-USD-Loco-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_loco_env_cfg:DeerUsdLocoStage3Cfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_loco_cfg:DeerUsdLocoStage3PPORunnerCfg",
    },
)

# v5.0 stable-base pivot: Stage3 terrain + control, naturalness rewards zeroed.
# Reuses Stage3's PPO runner cfg (same obs/action dims, same network).
gym.register(
    id="AniML-Deer-USD-Loco-Stable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_loco_env_cfg:DeerUsdLocoStableCfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_loco_cfg:DeerUsdLocoStablePPORunnerCfg",
    },
)

# v5.2 fresh-from-scratch stable base (forward-only, Stage1 recipe, naturalness
# rewards stripped). Trained WITHOUT warm-start to avoid inheriting the
# naturalness policy's reward-dependent gait (which destabilized v5.0/v5.1).
gym.register(
    id="AniML-Deer-USD-Loco-Stable-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_loco_env_cfg:DeerUsdLocoStableFlatCfg",
        "rsl_rl_cfg_entry_point": "training.agents.deer_usd_loco_cfg:DeerUsdLocoStableFlatPPORunnerCfg",
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

# Run 21: multi-clip walk + run motion library, V12 env with wider velocity
# command range (0.3-2.5 m/s) so policy is asked to span both gait speeds.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v15",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV12",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_RUN_CFG",
    },
)

# Run 22: walk + run + idle three-clip library. Idle adds rest-state /
# breathing pose diversity; weight reduced to 0.5 so its 331-frame length
# doesn't drown the 31/25-frame active-gait clips.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v16",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV12",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WALK_RUN_IDLE_CFG",
    },
)

# Run 23: narrow command range (0.5-1.2 m/s walk speeds only) + 3-clip
# dataset. Fixes r21/r22's velocity-tracking regression while keeping the
# multi-clip body-motion benefit. Discriminator's run-clip pose info still
# influences the manifold even though command is walk-only.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v17",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV13",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WRI_NARROW_CFG",
    },
)

# Run 26: FIX the velocity-tracking axis bug + add linear forward reward.
# r1-r25 used Isaac Lab's stock track_lin_vel_xy_exp which measures body
# +X (= deer's anatomical LEFT). This task uses the anatomy-aligned
# rewards (forward = body -Z) and adds forward_velocity_linear_anatomical
# at weight 2.0 to break the "stand still and look natural" local optimum.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v18",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV14",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WRI_FWDFIX_CFG",
    },
)

# Run 27: V15 env (RSI from walk + bumped fwd reward 4.0 + axis fix +
# pelvis_too_low term off) + walk+run dataset (no idle). Triple-pronged
# attack on the standing-still local optimum.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v19",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV15",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WR_RSI_CFG",
    },
)

# Run 28: V16 = V15 + pelvis_too_low termination ON. r27's regression
# (all 4 envs fell and lay there for 1000 steps each) traced to that
# termination being the load-bearing piece preventing the policy from
# gaming ep_len with collapsed posture.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v20",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV16",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_WR_RSI_TERM_CFG",
    },
)

# Run 29: V17 with locked command (forward only at 1.0 m/s).
# r28 produced bidirectional walking — moves but not consistently
# in commanded direction. Lock command to single value to force
# directional consistency.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v21",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV17",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_LOCKED_FWD_CFG",
    },
)

# Run 30: V18 = V17 + action_rate_l2 zeroed. Free up high-freq joint
# changes needed for real walking speed. r29 plateaued at 0.04 m/s
# despite commanded 1.0 m/s — the smoothness penalty was preventing
# the rapid leg-target changes a real walk gait needs.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v22",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV18",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_FREE_ACTION_CFG",
    },
)

# Run 31: V19 = lowered command (0.5 m/s) + mild action_rate (-0.005) +
# tighter exp std (0.3 instead of 0.5). Run with task_reward_ratio 0.85
# via CLI for max velocity-tracking pressure.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v23",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV19",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_LOW_CMD_CFG",
    },
)

# Run 32: V20 = V19 + forward_velocity_linear weight 4.0 -> 8.0.
# Run with --task_reward_ratio 0.95 to make forward speed unambiguously
# the dominant reward source. If still plateaus at 0.05 m/s, the
# in-place reference clip is the binding constraint.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v24",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV20",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_MAX_PUSH_CFG",
    },
)

# Run 33: V21 = V20 + episode_length 20s -> 40s. Force learning of
# sustained balance, not just 20s bursts. Pair with task_ratio 0.85
# (less aggressive than r32's 0.95). Warm-start from r32 model_2999.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v25",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV21",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_LONG_EP_CFG",
    },
)

# Run 35: V22 = V20 + tightened pelvis_too_low termination (min_height
# 0.10 -> 0.55) + base_height_l2 reward (target z=0.85, weight -30).
# Fixes the belly-slide local optimum r32-r34 fell into.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v26",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV22",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_NO_BELLY_CFG",
    },
)

# Run 36: V23 = V22 + variable command (0.4-0.7 m/s). Test if r35's
# upright walking generalizes to variable speeds. Warm-start r35.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v27",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV23",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_VARSPEED1_CFG",
    },
)

# Run 37: V24 = V22 + broader command (0.3-1.0 m/s). Warm-start r36.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v28",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV24",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_VARSPEED2_CFG",
    },
)

# Run 38: V25 = V24 + yaw rate (-0.3, 0.3). Add turning capability.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v29",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV25",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_TURN_CFG",
    },
)

# Run 39: V26 = V25 + lateral cmds (lin_vel_y -0.2..0.2). Full 3-DOF
# locomotion. Pair with FULL_LOCO_CFG (entropy_coef 0.002 to deflate
# noise_std). Warm-start from r38 turn checkpoint.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v30",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV26",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_FULL_LOCO_CFG",
    },
)

# Run 40: V27 = V26 + push_robot interval events (8-12s) for recovery
# learning. Mild ±0.4 m/s impulses simulate light shoves; pushes
# emergent disturbance-recovery behavior the Unity ecosystem game wants.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v31",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV27",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_PUSH_CFG",
    },
)

# Run 41: V28 = V27 + stronger ±0.6 m/s pushes + entropy 0.001.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v32",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV28",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_PUSH2_CFG",
    },
)

# Run 42: V29 = V28 + mild heightfield terrain (±0.5–2 cm bumps).
# First terrain pass — push policy off the flat-plane assumption.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v33",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV29",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_TERRAIN1_CFG",
    },
)

# Run 43: V30 = V28 + feet_air_time + feet_slide. Fix the shuffle/skate
# gait by adding stride mechanics rewards. Pair with STRIDE_CFG. Warm-
# start from r41b (animl_deer_amp_push2 model_14993).
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v34",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV30",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_STRIDE_CFG",
    },
)

# Run 44: V31 = V30 + trot_phase reward. Diagonal-pair alternation
# reward to break the bunny-hop / pronk gait.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v35",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV31",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_TROT_CFG",
    },
)

# Run 45: V32 = V30 + trot_phase_v2 (pronk-penalizing). From-scratch
# training to avoid the bunny-hop local optimum r43b/r44 baked in.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v36",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV32",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_FROMSCRATCH_TROT_CFG",
    },
)

# Run 46: V33 = V32 + 4x stronger action_rate_l2. Pair with TROT_ANNEAL_CFG
# (entropy_coef 10x lower). Cleans up r45's spasming gait.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v37",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV33",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_TROT_ANNEAL_CFG",
    },
)


# Run 47b (r45 PORT): V34 = r45's V32 with the three stride-mechanics rewards
# remapped from *_foot (cannon on the v4 rig) to *_toe (the actual hoof), plus
# light hip-abduction regularization for the 10 DOFs AMP cannot supervise.
# Fresh, entropy 0.005. Purpose: replicate r45 on the current 25-DOF COM-fixed
# rig, changing only what the rig change forces.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v38",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV34",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_R45PORT_CFG",
    },
)


# Run 48 (CANDIDATE, NOT LAUNCHED): same env as r47b (V34) but trained against
# the procedurally rebuilt, physically-grounded reference clip. Tests whether
# the reference data -- not the reward stack -- is the binding constraint.
gym.register(
    id="AniML-Deer-USD-Walk-AMP-v39",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.envs.deer_usd_amp_env_cfg:DeerUsdWalkAMPEnvCfgV34",
        "amp_rsl_rl_cfg_entry_point": "training.agents.deer_usd_amp_cfg:DEER_USD_AMP_PROCREF_CFG",
    },
)


# ===========================================================================
# Tier-3 (Poly Art / Malbers rig) tasks — the NEW generation. Kept entirely in
# its own subpackage (training/polyart/) so the rebuilt 38-DOF rig's configs
# never mix with the legacy deer_usd_* ones above. See training/polyart/.
# ===========================================================================
from training.polyart import tasks as _polyart_tasks  # noqa: E402,F401
