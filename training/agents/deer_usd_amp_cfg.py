"""Config for amp-rsl-rl's AMPOnPolicyRunner on the deer walk task.

Unlike rsl_rl's stock ``OnPolicyRunner``, ``AMPOnPolicyRunner`` expects a
plain ``dict`` (not a dataclass) with specific subsections: ``policy``,
``algorithm``, ``discriminator``, ``dataset``, plus top-level runner keys.

We provide both:

- ``DEER_USD_AMP_WALK_CFG``: the raw dict amp-rsl-rl consumes directly.
- ``DeerUsdWalkAMPRunnerCfg``: an isaaclab ``@configclass`` wrapper that
  holds the same values in dataclass form. If Isaac Lab's train.py is
  adapted to route AMP tasks to the AMP runner, it can introspect this
  config; otherwise the bespoke ``training/scripts/train_amp.py`` driver
  uses the dict directly.

The ``dataset.amp_data_path`` resolves to ``assets/motions`` relative to
repo root. Override via env var ``ANIML_AMP_DATA_PATH``.

See tools/convert_unity_anim_to_npy.py for how ``deer_walk.npy`` is
produced from the Unity .anim file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Resolve the motion dataset path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_amp_data_path() -> Path:
    raw = os.environ.get("ANIML_AMP_DATA_PATH", "").strip()
    if raw:
        p = Path(raw)
        if p.is_dir():
            return p.resolve()
    return (_REPO_ROOT / "assets" / "motions").resolve()


AMP_DATA_PATH = _resolve_amp_data_path()


# ---------------------------------------------------------------------------
# The dict the runner consumes
# ---------------------------------------------------------------------------

DEER_USD_AMP_WALK_CFG: dict[str, Any] = {
    # ---------------- runner top-level ----------------
    "seed": 42,
    "num_steps_per_env": 24,
    "max_iterations": 3000,
    "save_interval": 100,
    "experiment_name": "animl_deer_amp_walk",
    "run_name": "",
    "logger": "tensorboard",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "empirical_normalization": False,
    # rsl_rl 3.x requires this to be a dict. Keys = "set name" used by the
    # algorithm, values = list of observation group names from the env.
    # The AMP runner needs "policy", "critic", and "amp".
    "obs_groups": {
        # rsl_rl >= 4 (amp-rsl-rl 2.x) constructs the actor with obs_set
        # "actor"; rsl_rl 3.x used "policy". MLPModel does obs_groups[obs_set],
        # so the missing "actor" key raised KeyError. Provide BOTH names so
        # either library branch resolves. All three read the env's "policy"
        # observation group; "amp" reads the AMP group the discriminator uses.
        "actor":  ["policy"],
        "policy": ["policy"],
        "critic": ["policy"],
        "amp":    ["amp"],
    },

    # ---------------- policy (actor-critic) ----------------
    "policy": {
        # Short name resolved via _BUILTIN_CLASSES in amp_on_policy_runner.
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,  # AMP can tolerate high initial noise -- it
                                # supplies a gait prior via the style reward.
                                # If warm-starting from the standing ckpt,
                                # drop to 0.3.
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },

    # ---------------- actor / critic (rsl_rl >= 4 style) ----------------
    # amp-rsl-rl 2.x branches on RSL_RL_V4_PLUS. We run rsl_rl 5.x, so it
    # takes THIS branch and reads train_cfg["actor"] / ["critic"]; the
    # legacy "policy" section above is only used on rsl_rl 3.x. Keep both so
    # the config works on either. Values mirror "policy" exactly (same
    # 512-256-128 elu nets, same init noise) so r38 semantics are preserved.
    #
    # GOTCHA: the runner filters these dicts to keys accepted by
    # MLPModel.__init__ (obs/obs_groups/obs_set/output_dim are supplied
    # positionally; only hidden_dims / activation / obs_normalization /
    # distribution_cfg survive). Unknown keys are DROPPED SILENTLY -- so
    # init_noise_std must live inside distribution_cfg, not at this level,
    # or the actor would quietly fall back to default noise.
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,        # == old policy.init_noise_std
            "std_type": "scalar",
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
    },

    # Native replacement for train_amp.py's --task_reward_ratio monkey-patch
    # (amp-rsl-rl 2.x): reward = (1-style_weight)*task + style_weight*style.
    # 0.5 reproduces the old hardcoded 50/50 blend r38 trained with.
    "style_weight": 0.5,

    # ---------------- PPO + AMP algorithm ----------------
    "algorithm": {
        "class_name": "AMP_PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,           # higher than pure PPO -- AMP
                                        # benefits from broader exploration
                                        # early on.
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "amp_replay_buffer_size": 100_000,
        "use_smooth_ratio_clipping": False,
    },

    # ---------------- discriminator ----------------
    "discriminator": {
        "hidden_dims": [128, 128],
        # reward_scale balances style vs task reward. AMPOnPolicyRunner
        # combines them as ``0.5 * task + 0.5 * style`` -- we keep the
        # discriminator's own scale at 1.0 and let the runner do the blend.
        "reward_scale": 1.0,
        "loss_type": "BCEWithLogits",
        "empirical_normalization": True,
    },

    # ---------------- dataset ----------------
    "dataset": {
        "amp_data_path": AMP_DATA_PATH,
        # Map dataset filename stem -> sampling weight.
        "datasets": {
            "deer_walk": 1.0,
        },
        # 1 = use raw dataset fps (30). Increase to slow playback for slower
        # gaits; decrease isn't supported (would require upsampling beyond
        # nyquist). Walk is natural at 1x.
        "slow_down_factor": 1,
    },
}


# ---------------------------------------------------------------------------
# Dataclass wrapper (optional -- used if Isaac Lab's entry-point wants one)
# ---------------------------------------------------------------------------

# Run 2 cfg: warm-start from run 1 final ckpt, lower noise, stronger
# velocity pull via env (see DeerUsdWalkAMPEnvCfgV2). Same discriminator and
# AMP params as v1 so the learned discriminator is directly reusable.
DEER_USD_AMP_WALK_CFG_V2: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG,
    "experiment_name": "animl_deer_amp_walk_v2",
    "max_iterations": 1500,
    "save_interval": 100,
    "seed": 43,
    # rsl_rl v3's OnPolicyRunner supports resume via `load()`; amp-rsl-rl's
    # runner doesn't expose a resume flag in the cfg, so we do manual
    # ckpt loading in train_amp.py when args.resume_from is set.
}
# Override the actor-critic for V2 to use lower init_noise_std.
DEER_USD_AMP_WALK_CFG_V2["policy"] = {
    **DEER_USD_AMP_WALK_CFG["policy"],
    "init_noise_std": 0.4,   # warm-start friendly
}
# Slightly lower entropy coef to keep the warm-started policy focused.
DEER_USD_AMP_WALK_CFG_V2["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG["algorithm"],
    "entropy_coef": 0.005,
}


# Run 3 cfg: if run 2 plateaus, crank velocity weight further and lower noise.
# Intended to warm-start from the best model_*.pt under run 2.
DEER_USD_AMP_WALK_CFG_V3: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V2,
    "experiment_name": "animl_deer_amp_walk_v3",
    "max_iterations": 2000,
    "seed": 44,
}
DEER_USD_AMP_WALK_CFG_V3["policy"] = {
    **DEER_USD_AMP_WALK_CFG_V2["policy"],
    "init_noise_std": 0.25,  # tighter exploration once policy is already OK
}
DEER_USD_AMP_WALK_CFG_V3["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V2["algorithm"],
    "entropy_coef": 0.002,
    "learning_rate": 5.0e-4,  # slower, refining
}
# Drop AMP's dominance of the reward. The discriminator.reward_scale scales
# the style reward BEFORE the 0.5 runtime blend. With scale 0.3 the style
# contribution drops from ~96% to ~55-60%, giving task rewards room to
# actually shape the policy.
DEER_USD_AMP_WALK_CFG_V3["discriminator"] = {
    **DEER_USD_AMP_WALK_CFG_V2["discriminator"],
    "reward_scale": 0.3,
}


# Run 5 (task AniML-Deer-USD-Walk-AMP-v4): paired with DeerUsdWalkAMPEnvCfgV3.
# RSI + linear velocity reward + shorter episodes + no pelvis termination.
# We go BACK to the default 1.0 reward_scale because with RSI the deer spends
# more time near expert poses (closer to fooling the discriminator), so the
# style gradient should actually matter. Similarly, the monkey-patched
# task/style blend isn't needed: the linear forward-velocity reward provides
# direct signal regardless of scaling.
DEER_USD_AMP_WALK_CFG_V4: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V2,
    "experiment_name": "animl_deer_amp_walk_v4",
    "max_iterations": 2000,
    "seed": 45,
}
DEER_USD_AMP_WALK_CFG_V4["policy"] = {
    **DEER_USD_AMP_WALK_CFG_V2["policy"],
    "init_noise_std": 0.3,
}
DEER_USD_AMP_WALK_CFG_V4["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V2["algorithm"],
    "entropy_coef": 0.005,
    "learning_rate": 5.0e-4,
}
DEER_USD_AMP_WALK_CFG_V4["discriminator"] = {
    **DEER_USD_AMP_WALK_CFG_V2["discriminator"],
    "reward_scale": 1.0,   # full style strength now that RSI provides good poses
}


# Run 6 (task AniML-Deer-USD-Walk-AMP-v5): paired with DeerUsdWalkAMPEnvCfgV4.
# Anatomy-aligned rewards (body -Z forward), orientation termination,
# re-baked RSI along world -Y. Trained FROM SCRATCH because the prior
# policy (v3/v4 runs) learned to produce body-+X velocity, which is
# anatomical-lateral and fundamentally misaligned with the new objective.
DEER_USD_AMP_WALK_CFG_V5: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V4,
    "experiment_name": "animl_deer_amp_walk_v5",
    "max_iterations": 2000,
    "save_interval": 100,
    "seed": 46,
}
DEER_USD_AMP_WALK_CFG_V5["policy"] = {
    **DEER_USD_AMP_WALK_CFG_V4["policy"],
    "init_noise_std": 1.0,   # from-scratch friendly; RSI provides physics
}
DEER_USD_AMP_WALK_CFG_V5["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V4["algorithm"],
    "entropy_coef": 0.01,     # more exploration to find the correct gait
    "learning_rate": 1.0e-3,
}


# Run 7 (task AniML-Deer-USD-Walk-AMP-v6): warm-start from run 6's best
# checkpoint (iter 800, where ep_len 191 / base_tilt 40% / track_fwd 1.28)
# with MUCH lower entropy to keep the noise from ballooning and destroying
# the gait. Aim is to consolidate the walking behavior that emerged in run 6.
DEER_USD_AMP_WALK_CFG_V6: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V5,
    "experiment_name": "animl_deer_amp_walk_v6",
    "max_iterations": 2000,
    "seed": 47,
}
DEER_USD_AMP_WALK_CFG_V6["policy"] = {
    **DEER_USD_AMP_WALK_CFG_V5["policy"],
    "init_noise_std": 0.5,   # warm-start friendly, don't destroy the gait
}
DEER_USD_AMP_WALK_CFG_V6["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V5["algorithm"],
    "entropy_coef": 0.002,    # 5x lower than V5 -- stop std from climbing
    "learning_rate": 5.0e-4,
}


# Run 9 (task AMP-v7): action_scale 1.0 (was 0.5) so knees can reach full
# expert range; entropy_coef=0 so noise_std stays put instead of ballooning.
DEER_USD_AMP_WALK_CFG_V7: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V5,
    "experiment_name": "animl_deer_amp_walk_v7",
    "max_iterations": 3000,
    "save_interval": 100,
    "seed": 48,
}
DEER_USD_AMP_WALK_CFG_V7["policy"] = {
    **DEER_USD_AMP_WALK_CFG_V5["policy"],
    "init_noise_std": 0.5,
}
DEER_USD_AMP_WALK_CFG_V7["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V5["algorithm"],
    "entropy_coef": 0.0,
    "learning_rate": 1.0e-3,
}


# Run 10 (task AMP-v8): warm-start run 9 + small entropy + leg activity reward
DEER_USD_AMP_WALK_CFG_V8: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V7,
    "experiment_name": "animl_deer_amp_walk_v8",
    "max_iterations": 3000,
    "seed": 49,
}
DEER_USD_AMP_WALK_CFG_V8["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V7["algorithm"],
    "entropy_coef": 0.002,
    "learning_rate": 5.0e-4,
}


# Run 11 (task AMP-v9): per-joint soft spine/neck + leg_activity weight 2.5 +
# entropy 0.005. Same env cfg as v8 (with the updated reward weight).
DEER_USD_AMP_WALK_CFG_V9: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V8,
    "experiment_name": "animl_deer_amp_walk_v9",
    "max_iterations": 3000,
    "seed": 50,
}
DEER_USD_AMP_WALK_CFG_V9["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V8["algorithm"],
    "entropy_coef": 0.005,
    "learning_rate": 5.0e-4,
}


# Run 12 (task AMP-v10): anti-belly-slide.
# Env cfg V7 adds tilt-35°, pelvis_too_low, upright_bonus, 2500 envs.
# Agent side: shrink discriminator [128,128] → [64,64] so it can't trivially
# saturate (r11b had amp_loss=0.004, style_r=0.002 — disc gave up).
# Entropy 0.005 → 0.002 to stop noise_std drifting upward (0.40 → 0.59 over r11b).
DEER_USD_AMP_WALK_CFG_V10: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V9,
    "experiment_name": "animl_deer_amp_walk_v10",
    "max_iterations": 3000,
    "seed": 51,
}
DEER_USD_AMP_WALK_CFG_V10["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V9["algorithm"],
    "entropy_coef": 0.002,
    "learning_rate": 5.0e-4,
}
# NOTE: discriminator hidden_dims kept at [128,128] (V9 inherited).
# Tried [64,64] for anti-saturation but the saved checkpoint stores disc
# weights and PyTorch refuses size-mismatched load_state_dict even with
# strict=False. Shrinking the disc would require either training from
# scratch or patching train_amp.py to skip disc state on resume.


# Run 13 (task AMP-v11): anti-crouch-shuffle.
# Env cfg V8 adds linear pelvis_height_ramp + nose_pitch_penalty,
# tightens pelvis_too_low 0.55 → 0.70m.
# Agent side: entropy 0.002 → 0.001 to stop noise_std drift
# (0.40 → 0.59 → 0.72 across r11b/r12).
DEER_USD_AMP_WALK_CFG_V11: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V10,
    "experiment_name": "animl_deer_amp_walk_v11",
    "max_iterations": 3000,
    "seed": 52,
}
DEER_USD_AMP_WALK_CFG_V11["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V10["algorithm"],
    "entropy_coef": 0.001,
    "learning_rate": 5.0e-4,
}


# Run 14 (task AMP-v12): attack neck-bend cheat from r13.
# Env cfg V9 drops nose_pitch_penalty, adds head_height_ramp on skull body.
# Agent side: same hyperparams as V11 (entropy 0.001, seed bump).
DEER_USD_AMP_WALK_CFG_V12: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG_V11,
    "experiment_name": "animl_deer_amp_walk_v12",
    "max_iterations": 3000,
    "seed": 53,
}
DEER_USD_AMP_WALK_CFG_V12["algorithm"] = {
    **DEER_USD_AMP_WALK_CFG_V11["algorithm"],
    "entropy_coef": 0.001,
    "learning_rate": 5.0e-4,
}


# ---------------------------------------------------------------------------
# r21+ multi-clip configs: walk + run motion library so the AMP discriminator
# sees richer body motion (esp. through the spine/neck/head joints, which the
# walk clip alone barely animates: pelvis_to_chest amplitude ~2.4° in walk vs
# ~11° in run/idle). Used with V11 env (zeroed L2) + the widened URDF/USD.
# ---------------------------------------------------------------------------

DEER_USD_AMP_WALK_RUN_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG,
    "experiment_name": "animl_deer_amp_walk_run",
    "max_iterations": 3000,
    "seed": 48,
}
DEER_USD_AMP_WALK_RUN_CFG["dataset"] = {
    **DEER_USD_AMP_WALK_CFG["dataset"],
    # Equal-weight mixture. Walk clip is 31 frames, run is 25; weight=1.0
    # each gives roughly equal sampling. Names are filename stems (without
    # .npy) and must match files under amp_data_path.
    "datasets": {
        "deer_walk": 1.0,
        "deer_run":  1.0,
    },
}


# Run + walk + idle three-clip mix. Idle has 331 frames (the longest by far)
# and lots of subtle weight-shift / breathing motion that should pull the
# policy toward natural rest-state postures even when the velocity command
# is low. Reserve for r22 onward once we know walk+run plays well.
DEER_USD_AMP_WALK_RUN_IDLE_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_CFG,
    "experiment_name": "animl_deer_amp_walk_run_idle",
    "max_iterations": 3000,
    "seed": 49,
}
DEER_USD_AMP_WALK_RUN_IDLE_CFG["dataset"] = {
    **DEER_USD_AMP_WALK_CFG["dataset"],
    "datasets": {
        "deer_walk": 1.0,
        "deer_run":  1.0,
        "deer_idle": 0.5,   # less weight: idle is 10x longer, don't drown
                            #              the active gait clips
    },
}


# r23: same 3-clip dataset but narrower command range (V13 env, 0.5-1.2 m/s).
# Separate experiment_name so logs don't collide with r22's animl_deer_amp_walk_run_idle.
DEER_USD_AMP_WRI_NARROW_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_IDLE_CFG,
    "experiment_name": "animl_deer_amp_wri_narrow",
    "seed": 50,
}


# r26: same 3-clip narrow setup, but the env now uses anatomy-aligned
# velocity tracking (body -Z = forward) instead of the wrong-axis stock
# Isaac Lab rewards. Separate experiment_name so r26's logs are isolated
# (the policy semantics differ — comparing to r25 isn't apples-to-apples).
DEER_USD_AMP_WRI_FWDFIX_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_IDLE_CFG,
    "experiment_name": "animl_deer_amp_wri_fwdfix",
    "seed": 53,
}


# r27: walk+run dataset (no idle — that was the standing-still attractor)
# paired with V15 env (RSI from walk + bumped fwd reward + axis fix).
DEER_USD_AMP_WR_RSI_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_wr_rsi",
    "seed": 54,
}


# r28: same as r27 but env is V16 (pelvis_too_low termination back ON).
DEER_USD_AMP_WR_RSI_TERM_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_wr_rsi_term",
    "seed": 55,
}


# r29: locked-command setup — V17 env with lin_vel_x=(1.0, 1.0), no
# lateral, no yaw. Force directional consistency.
DEER_USD_AMP_LOCKED_FWD_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_locked_fwd",
    "seed": 56,
}


# r30: V18 (V17 + action_rate_l2 zeroed). Fresh seed.
DEER_USD_AMP_FREE_ACTION_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_free_action",
    "seed": 57,
}


# r31: V19 (lowered command 0.5 m/s, mild action_rate=-0.005, tighter std)
DEER_USD_AMP_LOW_CMD_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_low_cmd",
    "seed": 58,
}


# r32: V20 (V19 + fwd weight 4.0->8.0). Pair with task_reward_ratio 0.95.
DEER_USD_AMP_MAX_PUSH_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_max_push",
    "seed": 59,
}


# r33: V21 (V20 + episode_length_s 20 -> 40). Pair with task_ratio 0.85.
DEER_USD_AMP_LONG_EP_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_long_ep",
    "seed": 60,
}


# r35: V22 (V20 + tighter pelvis_too_low + base_height_l2 reward).
DEER_USD_AMP_NO_BELLY_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_no_belly",
    "seed": 62,
}


# r36: V23 (V22 + variable command 0.4-0.7).
DEER_USD_AMP_VARSPEED1_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_varspeed1",
    "seed": 63,
}


# r37: V24 (V22 + broader command 0.3-1.0).
DEER_USD_AMP_VARSPEED2_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_varspeed2",
    "seed": 64,
}


# r38: V25 (V24 + yaw rate commands).
DEER_USD_AMP_TURN_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_turn",
    "seed": 65,
}


# r39: V26 (V25 + lateral cmd lin_vel_y -0.2..0.2). Full 3-DOF locomotion.
# r36-r38 inherited entropy_coef=0.01 and noise_std ballooned from 1.0 to
# 3.82 over ~9000 iters (per CLAUDE.md, 0.01 destroys learned gaits). Drop
# to 0.002 so the bloated std starts decaying instead of growing further.
# rsl_rl v3 preserves the checkpoint's std_param on resume, but lowering
# entropy_coef makes future updates pull it back down.
DEER_USD_AMP_FULL_LOCO_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_full_loco",
    "seed": 66,
}
DEER_USD_AMP_FULL_LOCO_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.002,
}


# r40: V27 (V26 + interval push events). Recovery-from-disturbance training.
# Keep entropy_coef=0.002 so noise_std continues deflating from r39's 2.57.
DEER_USD_AMP_PUSH_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_push",
    "seed": 67,
}
DEER_USD_AMP_PUSH_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.002,
}


# r41: V28 (V27 + stronger ±0.6 m/s pushes). entropy_coef 0.001 to keep
# deflating noise_std from r40's 2.12. Warm-start r40.
DEER_USD_AMP_PUSH2_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_push2",
    "seed": 68,
}
DEER_USD_AMP_PUSH2_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.001,
}


# r42: V29 (V28 + mild rough terrain). First step toward terrain-
# adaptive emergent behavior. Keep entropy_coef 0.001.
DEER_USD_AMP_TERRAIN1_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_terrain1",
    "seed": 69,
}
DEER_USD_AMP_TERRAIN1_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.001,
}


# r43: V30 (V28 + feet_air_time + feet_slide rewards). Targets the
# "shuffle / skating" gait pattern by adding explicit stride mechanics:
# air-time bonus per stride + slide penalty on contact. Warm-start r41b.
# v2: weights bumped 5x (slide -0.5->-3.0, air_time +1.0->+5.0, threshold
# 0.25->0.35s). Entropy_coef pumped 0.001->0.002 to give the policy more
# exploration room to escape the deep shuffle local optimum.
DEER_USD_AMP_STRIDE_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_stride",
    "seed": 70,
}
DEER_USD_AMP_STRIDE_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.002,
}


# r44: V31 (V30 + trot_phase reward). Targets bunny-hop -> trot
# transition by adding diagonal-pair alternation reward.
DEER_USD_AMP_TROT_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_trot",
    "seed": 71,
}
DEER_USD_AMP_TROT_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.002,
}


# r45: V32 (V30 + trot_phase_v2 with pronk penalty). Plan-B from-scratch
# training so the policy never gets to encode bunny-hop as a local
# optimum. Higher init_noise_std + entropy_coef for fresh exploration.
DEER_USD_AMP_FROMSCRATCH_TROT_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_fromscratch_trot",
    "seed": 72,
    "max_iterations": 4000,
}
DEER_USD_AMP_FROMSCRATCH_TROT_CFG["policy"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["policy"],
    "init_noise_std": 1.0,    # standard fresh-init exploration
}
DEER_USD_AMP_FROMSCRATCH_TROT_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.005,    # higher early entropy to prevent premature convergence
}


# r46: V33 (V32 + stronger action_rate_l2). Anneal pass to clean up
# the spasming r45 gait. entropy_coef 10x lower so std collapses.
DEER_USD_AMP_TROT_ANNEAL_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_trot_anneal",
    "seed": 73,
}
DEER_USD_AMP_TROT_ANNEAL_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.0005,    # 10x lower than r45 to deflate std
}


try:
    from isaaclab.utils import configclass

    @configclass
    class DeerUsdWalkAMPRunnerCfg:
        """Thin dataclass wrapper carrying the same values as
        :data:`DEER_USD_AMP_WALK_CFG`. Exists for symmetry with other
        Deer USD runner configs; the AMPOnPolicyRunner actually needs
        the dict form. Use ``to_dict()`` to get the runner payload."""

        cfg: dict = field(default_factory=lambda: dict(DEER_USD_AMP_WALK_CFG))

        def to_dict(self) -> dict:
            return self.cfg

        @property
        def experiment_name(self) -> str:
            return self.cfg["experiment_name"]

        @property
        def max_iterations(self) -> int:
            return self.cfg["max_iterations"]

except ImportError:  # isaaclab not available in this process
    DeerUsdWalkAMPRunnerCfg = None  # type: ignore[assignment]


# r47b: PORT OF r45 to the current 25-DOF rig (env V34).
# Agent params copied from r45 (FROMSCRATCH_TROT) verbatim: entropy 0.005
# (inside CLAUDE.md's safe 0.002-0.005 band), style_weight 0.5, lr 1e-3,
# walk+run datasets, FROM SCRATCH.
# NOTE: r45's cfg set init_noise_std inside "policy", which rsl_rl >=4
# IGNORES (it reads actor.distribution_cfg.init_std). Our base actor already
# has init_std 1.0, which is what r45 intended, so fresh-init matches.
DEER_USD_AMP_R45PORT_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_r45port",
    "seed": 80,
    "max_iterations": 4000,
}
DEER_USD_AMP_R45PORT_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.005,
}


# r48 (CANDIDATE, not launched): identical to the r45-port agent config except
# it trains against the PROCEDURALLY REBUILT reference clip instead of the
# shipped one. Purpose: test the night's central conclusion -- that the
# reference data, not the reward stack, is what caps this project.
#
# Shipped deer_walk vs assets/motions/deer_walk_proc.npy:
#   joints out of range   2 (fetlock, 0.90 rad)  ->  0
#   min hoof height       +0.271 (floats)        ->  -0.063 (grounded)
#   stance fraction       17%                    ->  61%
#   root_z range          0.0000 (flat)          ->  0.0300
# r47b showed feet_slide/feet_air_time are INERT against a reference that never
# makes contact, so this is the change that should let them finally bite.
DEER_USD_AMP_PROCREF_CFG: dict[str, Any] = {
    **DEER_USD_AMP_WALK_RUN_CFG,
    "experiment_name": "animl_deer_amp_procref",
    "seed": 81,
    "max_iterations": 4000,
}
DEER_USD_AMP_PROCREF_CFG["algorithm"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["algorithm"],
    "entropy_coef": 0.005,
}
# ONLY the procedural clip. deer_run is deliberately excluded: it has its own
# unresolved hind-fetlock defect (NIGHT_LOG section 2c/3c) and would reintroduce
# ungrounded feet, defeating the point of the experiment.
DEER_USD_AMP_PROCREF_CFG["dataset"] = {
    **DEER_USD_AMP_WALK_RUN_CFG["dataset"],
    "datasets": {"deer_walk_proc": 1.0},
}
