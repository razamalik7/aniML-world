"""Extra AMP observation terms — things the discriminator is currently BLIND to.

WHY THIS EXISTS. The AMP observation is
    joint_pos(38) + joint_vel(38) + base_lin_vel(3) + base_ang_vel(3) = 82
Nothing in there encodes GROUND RELATIONSHIP. A deer with textbook joint angles that
is walking on the front of its ankles, or lying on its side, looks *perfect* to the
discriminator. exp2 measured 53.3% of its load carried on the ankle/pastern rather
than the hoof, and Raza spotted it on video before any metric did -- there was no
metric, because the discriminator has no term for it either. `non_hoof_contact` at
-0.75 was the only objection in the whole system and it was not enough.

WHY HEIGHT AND NOT CONTACT. Raza's objection, which was correct: the reference clips
are pure animation with no physics, so there are no contact forces to compare against.
Any contact label would have to be inferred from a height threshold -- brittle, and
wrong wherever the animation floats or interpenetrates.

Height sidesteps that entirely. It is geometry, not dynamics: FK gives it on the
reference side, the simulator gives it directly on the policy side, and the SAME link
origin is used for both. Measured on the reference walk, the hoof link origins sit
0.029-0.054 m at their lowest (the origin is the joint, not the sole) and swing up to
0.19-0.21 m. That constant per-hoof offset does not matter, because it is present
identically on both sides and cancels in the comparison. What survives is the 13-18 cm
swing trajectory -- which encodes footfall timing continuously, strictly more
information than a binary contact flag.

CORRECTION (2026-07-29). The paragraph that used to sit here predicted that height
would also expose LEG NEGLECT, on the reasoning that "an under-used leg has a hoof that
barely rises". That was measured and it is FALSE. In A2 the two front hooves reach the
same minimum height -- 0.0 cm and 0.1 cm above ground -- while carrying 0.02 and 0.32
of the load, a 16x difference. The neglected leg is neither frozen nor held high: its
range of motion is 0.95x the expert's. It descends all the way down and simply never
stops. So height cannot separate a supporting leg from a passenger, and hoof_heights
should be expected to fix ankle-walking (a genuine POSE signature) and nothing else.

What DOES separate them is `hoof_planar_speed` below. A hoof in stance is stationary in
the world while the body moves over it; a hoof that is merely brushing past keeps
moving. Normalised by body speed, the reference plants all four hooves at 0.04-0.44
across walk, trot and run -- while our policies skim the FRONT hooves at 0.7-1.55, six
to thirteen times the reference, and the slip tracks the missing load inversely (A2:
slip 1.55 -> share 0.02; slip 0.47 -> share 0.32). It is the largest policy/reference
discrepancy found so far, and the discriminator cannot see any of it.

Like height, it is pure geometry: FK plus a finite difference on the reference side,
`body_lin_vel_w` on the sim side. No contact forces, no physics on the animation side.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def hoof_heights(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-hoof link-origin height above this env's ground plane. Shape (num_envs, 4).

    Subtracts the env origin so the value is height above the LOCAL ground, not world
    Z -- otherwise every env in the grid would report a different constant offset and
    the discriminator would learn the grid layout instead of the gait.
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return z - env.scene.env_origins[:, 2].unsqueeze(1)


def hoof_planar_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-hoof horizontal speed in the world frame, m/s. Shape (num_envs, 4).

    Horizontal only, and a MAGNITUDE, so the value is invariant to which way the deer
    happens to be facing -- the discriminator must not be able to identify a clip by its
    heading. Vertical motion is deliberately excluded: a hoof lifting is normal swing,
    whereas a hoof sliding along the ground is the defect.

    World frame, not body frame, is the whole point. In stance the hoof is planted and
    the BODY moves over it, so world speed goes to ~0 while body-relative speed is large
    and uninformative. Env origins do not enter: velocity is translation-invariant.

    Absolute rather than normalised by body speed, because ``base_lin_vel`` is already
    in the AMP observation -- the discriminator can form the ratio itself, and giving it
    the raw quantity keeps the reference-side computation a plain finite difference.
    """
    asset = env.scene[asset_cfg.name]
    v = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.linalg.norm(v, dim=-1)
