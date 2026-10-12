"""Custom reward terms for the Tier-3 deer AMP env.

Kept separate from the legacy ``training/envs/deer_rewards.py`` (which carries the
old rig's anatomy-axis shims). The Tier-3 rig is +X-forward, so these are simple.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def forward_velocity_linear(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """LINEAR reward for moving along the commanded direction, capped at the
    commanded speed.

    The stock ``track_lin_vel_xy_exp`` is ``exp(-error²/std²)`` — at large error
    (a standing deer vs a 1 m/s command) it's ≈0 and FLAT, giving no gradient to
    start moving, so the policy settles into "stand still + match style". This
    term is linear in achieved forward speed, so it pays off for every bit of
    forward progress — a constant "go" gradient that breaks that local optimum.

      reward = clamp( v·d̂ , 0 , |cmd| )

    where d̂ is the commanded planar direction (body frame) and v is the body-
    frame root linear velocity. No reward for standing (0), backing up (0), or
    overspeeding (capped) — the exp term still shapes precise tracking on top.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)          # (N, 3) body frame
    cmd_xy = cmd[:, :2]
    cmd_speed = torch.norm(cmd_xy, dim=1)                        # (N,)
    dir_xy = cmd_xy / cmd_speed.clamp(min=1e-3).unsqueeze(1)     # unit commanded dir
    v_along = (asset.data.root_lin_vel_b[:, :2] * dir_xy).sum(dim=1)
    return torch.clamp(v_along.clamp(min=0.0), max=cmd_speed)


def head_too_low(
    env: ManagerBasedRLEnv,
    target_height: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Head"),
) -> torch.Tensor:
    """One-sided penalty for carrying the SKULL below ``target_height`` (m above
    the env floor). Returns ``max(0, target - head_z)²`` (>=0; use a NEGATIVE
    weight).

    r5's pelvis-height terms fixed the BODY height but the deer still games
    posture by keeping the HEAD down (the "posture-metrics-get-gamed" failure:
    the pelvis metric is satisfied while the skull sags). This constrains the
    skull directly. One-sided so it only pushes the head UP when low — it never
    forces an unnatural head-high carriage, so the AMP walk style is preserved.
    """
    asset = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2] - env.scene.env_origins[:, 2]
    below = (target_height - head_z).clamp(min=0.0)
    return below ** 2


# --------------------------------------------------------------------------
# Reference-range conformance
# --------------------------------------------------------------------------
_REF_RANGE_CACHE: dict = {}


def _reference_joint_range(motion_path, joint_names, device):
    """Per-joint (min, max) of the reference clip, ordered to the SIM's joints."""
    key = (str(motion_path), tuple(joint_names), str(device))
    if key in _REF_RANGE_CACHE:
        return _REF_RANGE_CACHE[key]
    import numpy as np
    from pathlib import Path
    p = Path(motion_path) if motion_path else (
        Path(__file__).resolve().parents[2] / "assets" / "motions" / "deer_polyart_walk.npy")
    d = np.load(str(p), allow_pickle=True).item()
    ref_names = list(d["joints_list"])
    jp = np.asarray(d["joint_positions"], dtype=np.float32)
    idx = [ref_names.index(n) if n in ref_names else None for n in joint_names]
    lo = np.zeros(len(joint_names), dtype=np.float32)
    hi = np.zeros(len(joint_names), dtype=np.float32)
    for k, i in enumerate(idx):
        if i is None:                      # joint absent from the clip: no constraint
            lo[k], hi[k] = -1e6, 1e6
        else:
            lo[k], hi[k] = jp[:, i].min(), jp[:, i].max()
    out = (torch.tensor(lo, device=device), torch.tensor(hi, device=device))
    _REF_RANGE_CACHE[key] = out
    return out


def joint_outside_reference_range(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_path: str | None = None,
    margin: float = 0.087,          # ~5 deg of free play before any penalty
) -> torch.Tensor:
    """Penalise joint angles that leave the range the REFERENCE clip actually uses.

    Returns the summed radians outside ``[ref_min - margin, ref_max + margin]``
    (>=0; use a NEGATIVE weight).

    WHY (r10, user-spotted). r10's ankles looked visibly wrong on video, and the
    measurement was stark: 95% of the left hoof-ankle's samples fell OUTSIDE the
    range the reference walk ever uses, and the policy's range was essentially the
    URDF LIMIT range rather than the walk range -- on 16 of 22 leg joints. Nothing
    penalised extreme joint use (`dof_torques_l2`/`dof_acc_l2` zeroed since r3), so
    any range the skeleton permitted got used.

    The obvious fix -- clamping the URDF limits down to the walk -- is the WRONG
    LAYER. It bakes one skill into the skeleton: training a run later would need a
    different rig, a single policy could never do both, and emergent gait
    transitions become impossible. Limits should encode ANATOMY; what is
    APPROPRIATE for the current task belongs in the reward. This is that reward.

    It also re-derives itself: point ``motion_path`` at a run clip and the
    admissible range becomes the run's, with no rig change.

    NOTE this is a backstop, not the primary mechanism. AMP's discriminator should
    already reject out-of-distribution joint angles; it failed to because
    amp-rsl-rl updates its observation normaliser with POLICY as well as expert
    data, so policy outliers inflate the variance and hide themselves. Fixing that
    (see train_amp.py) is the real repair; this term catches what slips through.
    """
    asset = env.scene[asset_cfg.name]
    lo, hi = _reference_joint_range(motion_path, tuple(asset.data.joint_names), asset.device)
    q = asset.data.joint_pos
    out = (lo - margin - q).clamp(min=0.0) + (q - (hi + margin)).clamp(min=0.0)
    return torch.sum(out, dim=1)


def feet_air_time_positive(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """``mdp.feet_air_time`` with the payout clamped at zero.

    THE PERVERSE INCENTIVE (r16, measured). The stock term is

        sum( (last_air_time - threshold) * first_contact )

    and it only pays ON LANDING. For a leg that never lifts, ``first_contact``
    never fires, so it contributes exactly 0. For a leg that takes a SHORT step
    it fires with ``last_air_time < threshold``, so it contributes a NEGATIVE
    number. Doing nothing therefore scores strictly better than trying:

        never lifts (r12/r13 dead legs, duty 0.03)  ->   0.0
        tentative 0.06 s step                        ->  (0.06-0.12) = -0.06
        proper 0.35 s stride                         ->  (0.35-0.12) = +0.23

    That is the dead-leg pathology written as a reward. It also explains why the
    term has been NEGATIVE since r7 on a rig whose median swing (0.16 s) sits near
    the threshold: lowering 0.2 -> 0.12 at r12 moved the break-even point but left
    the sign flip in place. Raising the WEIGHT makes it strictly worse -- at 20x
    it went to -0.05 and r16p1 fell behind probe-N at matched iterations (ep_len
    40.2 vs 45.9), which is why that run was killed at iter ~130.

    Clamping at zero makes a tentative step worth the same as no step (never
    worse), while long strides still pay. Trying can no longer be punished.

    NOT ``feet_air_time_positive_biped``: that one enforces SINGLE STANCE, which is
    a biped assumption. A quadruped walk keeps 2-3 feet down at a time, so it would
    pay ~0 for a correct deer gait.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold).clamp(min=0.0) * first_contact, dim=1)
    # no reward for a zero command (matches the stock term)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
