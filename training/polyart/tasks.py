"""Gym registration for the Tier-3 (Poly Art) deer tasks.

Imported by ``training/tasks/__init__.py`` so the ids register alongside the
legacy tasks. Launch via the AMP driver (``training/scripts/train_amp.py``),
NOT Isaac Lab's stock rsl_rl train.py (which doesn't know AMP_PPO).
"""

from __future__ import annotations

import gymnasium as gym

# AMP walk, phase 1: flat ground, forward walk, 38-DOF Poly Art rig.
gym.register(
    id="AniML-Deer-PolyArt-Walk-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "training.polyart.amp_env_cfg:DeerPolyArtWalkAMPEnvCfg",
        "amp_rsl_rl_cfg_entry_point": "training.polyart.amp_runner_cfg:DEER_POLYART_AMP_WALK_CFG",
    },
)
