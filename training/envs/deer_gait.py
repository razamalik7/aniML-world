"""Walk-These-Ways-style gait conditioning for the deer locomotion stack.

After 5 attempts at pure reward-shaping S1 (kneeling, tuck-and-freeze,
rigid-front wheelbarrow, downward-dog) we hit the same wall MIT's
"Walk These Ways" team did: free-form quadruped policies find an
unbounded number of degenerate gaits when given only velocity
tracking + smoothness rewards. Their solution: PRESCRIBE a target
foot-contact pattern (which feet should be in stance vs swing at any
given moment) and reward matching it.

This module provides:
  - A per-env gait clock (computed on the fly from
    ``env.episode_length_buf`` so it auto-resets each episode).
  - Per-foot phase offsets defining the gait pattern (default = trot:
    diagonal pairs, FL/HR sync, FR/HL sync, two pairs offset 0.5 cycles).
  - ``gait_phase_obs`` term: feeds (cos, sin) of each foot's phase into
    the policy observation so the policy knows where in the stride it is.
  - ``gait_contact_match`` reward: per step, +1 per foot whose actual
    contact state matches the target (target_in_stance = phase < duty).
  - ``gait_swing_velocity`` reward: encourage forward foot velocity
    during swing, discourage during stance.

Gait parameters (frequency, duty_cycle, phase_offsets) are fixed per
training run via reward-term params. Walk-These-Ways samples these per
episode for diverse gaits; we keep them fixed for now as the simplest
form. (Future: gait command channel.)

Foot ordering convention (must match the contact sensor body_names you
pass in): ``[FL_foot, HR_foot, FR_foot, HL_foot]`` so trot offsets
(0.0, 0.0, 0.5, 0.5) put diagonals in sync.
"""

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


# ---------------------------------------------------------------------------
# Phase calculation (single source of truth used by obs + rewards).
# ---------------------------------------------------------------------------

def _compute_foot_phases(
    env: ManagerBasedRLEnv,
    frequency: float,
    foot_phase_offsets: Sequence[float],
) -> torch.Tensor:
    """Return per-foot phase in [0, 1) — shape (num_envs, num_feet).

    Phase advances at ``frequency`` Hz from episode start. Each foot has
    its own offset to define the gait pattern (e.g. trot: diagonal pairs
    in sync, opposite pairs offset 0.5).
    """
    # episode_length_buf is steps since this env's last reset.
    t = env.episode_length_buf.float() * env.step_dt   # (num_envs,)
    base_phase = (t * frequency) % 1.0                 # (num_envs,)
    offsets = torch.tensor(
        list(foot_phase_offsets),
        device=env.device,
        dtype=torch.float32,
    )                                                   # (num_feet,)
    foot_phases = (base_phase.unsqueeze(-1) + offsets.unsqueeze(0)) % 1.0
    return foot_phases                                  # (num_envs, num_feet)


# ---------------------------------------------------------------------------
# Observation: feed phase into the policy so it knows where in the stride.
# ---------------------------------------------------------------------------

def gait_phase_obs(
    env: ManagerBasedRLEnv,
    frequency: float = 2.0,
    foot_phase_offsets: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 0.5),
) -> torch.Tensor:
    """Periodic encoding of each foot's phase: (cos, sin) per foot.

    Output shape: (num_envs, 2 * num_feet) — i.e. 8 floats for 4-legged
    deer. Plug into the AMP-style observation group via ``ObsTerm`` with
    this function as ``func``.
    """
    foot_phases = _compute_foot_phases(env, frequency, foot_phase_offsets)
    angles = 2.0 * torch.pi * foot_phases               # (num_envs, num_feet)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


# ---------------------------------------------------------------------------
# Reward: per-foot contact state vs target schedule.
# ---------------------------------------------------------------------------

def gait_contact_match(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    frequency: float = 2.0,
    foot_phase_offsets: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 0.5),
    duty_cycle: float = 0.5,
    contact_threshold: float = 5.0,
) -> torch.Tensor:
    """Per-step reward: how many feet have correct contact state.

    For each foot:
        target_stance = (phase < duty_cycle)
        actual_contact = (foot_force > threshold)
        match = (target_stance == actual_contact)
    Reward = sum_feet (match * 1.0). Range [0, num_feet].

    With trot offsets (0, 0, 0.5, 0.5) and duty=0.5:
      Phase 0.0-0.5: FL/HR in stance, FR/HL in swing  (one diagonal grounded)
      Phase 0.5-1.0: FR/HL in stance, FL/HR in swing  (other diagonal grounded)
    Forces the deer to alternate diagonals -- which IS what trotting is.

    The sensor_cfg.body_names ORDER must match foot_phase_offsets order:
    [FL_foot, HR_foot, FR_foot, HL_foot] is the convention.
    """
    foot_phases = _compute_foot_phases(env, frequency, foot_phase_offsets)
    target_stance = (foot_phases < duty_cycle)          # (num_envs, num_feet) bool

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    actual_contact = torch.norm(forces, dim=-1) > contact_threshold  # (num_envs, num_feet) bool

    match = (target_stance == actual_contact).float()
    return match.sum(dim=-1)                            # (num_envs,) in [0, num_feet]


# ---------------------------------------------------------------------------
# Reward: shaped contact force pattern (Walk-These-Ways' "shaped_force").
# Penalizes contact forces during swing and rewards moderate during stance.
# ---------------------------------------------------------------------------

def gait_shaped_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    frequency: float = 2.0,
    foot_phase_offsets: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 0.5),
    duty_cycle: float = 0.5,
    sigma: float = 100.0,
) -> torch.Tensor:
    """Penalize contact force when foot SHOULD be swinging.

    `match` reward above is binary; this is a smoother gradient:
       force = ||contact force||
       in_swing_target = (phase >= duty_cycle)
       penalty per foot = exp(-(force/sigma)**2) ONLY when target is swing
    Returns a positive value to be summed; pair with negative weight.

    Actually per Walk-These-Ways naming, the shaped reward is positive for
    LOW force in swing and NEUTRAL during stance. We return:
        sum_feet  in_swing_target * exp(-(force/sigma)**2)
    => +1 per swinging-foot whose force is near zero. Use positive weight.
    """
    foot_phases = _compute_foot_phases(env, frequency, foot_phase_offsets)
    in_swing_target = (foot_phases >= duty_cycle).float()  # (num_envs, num_feet)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    f_mag = torch.norm(forces, dim=-1)                     # (num_envs, num_feet)
    swing_quality = torch.exp(-(f_mag / sigma).square())   # near 1 when force small

    return (in_swing_target * swing_quality).sum(dim=-1)   # (num_envs,) in [0, num_feet]
