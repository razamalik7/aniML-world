"""Custom reward and termination functions for deer standing / locomotion.

Geometry-based terms (heights, spreads) plus **contact-force** rewards so the policy
cannot rest stable weight on the torso: feet must bear load (see V6 plan).
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

_IDX_CACHE: dict[str, torch.Tensor] = {}


def _body_indices(robot: Articulation, body_names: list[str], device) -> torch.Tensor | None:
    key = f"{id(robot)}_{'_'.join(body_names)}"
    if key not in _IDX_CACHE:
        all_names = robot.body_names
        idxs = [all_names.index(n) for n in body_names if n in all_names]
        _IDX_CACHE[key] = torch.tensor(idxs, device=device, dtype=torch.long) if idxs else None
    return _IDX_CACHE[key]


def _body_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, body_names: list[str]) -> torch.Tensor:
    """Return mean world-Z of the named bodies.  Shape: (num_envs,)."""
    robot: Articulation = env.scene[asset_cfg.name]
    idx_t = _body_indices(robot, body_names, env.device)
    if idx_t is None or len(idx_t) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    return robot.data.body_pos_w[:, idx_t, 2].mean(dim=-1)


def _each_body_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, body_names: list[str]) -> torch.Tensor:
    """Return world-Z of each named body. Shape: (num_envs, num_bodies)."""
    robot: Articulation = env.scene[asset_cfg.name]
    idx_t = _body_indices(robot, body_names, env.device)
    if idx_t is None or len(idx_t) == 0:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return robot.data.body_pos_w[:, idx_t, 2]


# ------------------------------------------------------------------
# Feet low: penalize EACH foot being above ground, not just average
# ------------------------------------------------------------------
def feet_height_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot", "FL_foot", "FR_foot"),
    target_z: float = 0.05,
) -> torch.Tensor:
    """Mean L2 penalty across all individual feet (catches single high foot)."""
    each_z = _each_body_z(env, asset_cfg, list(foot_names))
    return (each_z - target_z).square().mean(dim=-1)


# ------------------------------------------------------------------
# Head high: reward skull being above pelvis
# ------------------------------------------------------------------
def head_above_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_names: list[str] = ("skull",),
    base_names: list[str] = ("pelvis",),
    target_diff: float = 0.5,
) -> torch.Tensor:
    """L2 penalty when head-minus-pelvis height is below target_diff."""
    head_z = _body_z(env, asset_cfg, list(head_names))
    base_z = _body_z(env, asset_cfg, list(base_names))
    diff = head_z - base_z
    return (diff - target_diff).square()


# ------------------------------------------------------------------
# Feet below pelvis: all 4 feet should be lower than pelvis
# ------------------------------------------------------------------
def feet_below_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot", "FL_foot", "FR_foot"),
    body_names: list[str] = ("pelvis",),
) -> torch.Tensor:
    """Reward proportional to how much the pelvis is above the feet."""
    foot_z = _body_z(env, asset_cfg, list(foot_names))
    body_z = _body_z(env, asset_cfg, list(body_names))
    gap = body_z - foot_z
    return torch.clamp(gap, min=0.0)


# ------------------------------------------------------------------
# Alive bonus: constant reward for pelvis being above a threshold
# ------------------------------------------------------------------
def upright_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_names: list[str] = ("pelvis",),
    min_height: float = 0.25,
) -> torch.Tensor:
    """Binary +1 reward when pelvis is above min_height, else 0."""
    base_z = _body_z(env, asset_cfg, list(base_names))
    return (base_z > min_height).float()


# ------------------------------------------------------------------
# Joint pos regularization: stay close to the standing IK pose
# ------------------------------------------------------------------
def joint_pos_standing_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation of joint positions from the initial (standing) pose."""
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_pos - robot.data.default_joint_pos).square().mean(dim=-1)


# ------------------------------------------------------------------
# Foot spread: penalize difference between highest and lowest foot
# ------------------------------------------------------------------
def foot_height_spread(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot", "FL_foot", "FR_foot"),
) -> torch.Tensor:
    """Penalize Z spread among feet (all feet should be at same height)."""
    each_z = _each_body_z(env, asset_cfg, list(foot_names))
    spread = each_z.max(dim=-1).values - each_z.min(dim=-1).values
    return spread.square()


# ------------------------------------------------------------------
# Hind joint extension: penalize hind leg joints deviating from standing angles
# ------------------------------------------------------------------
_HIND_JOINT_IDX_CACHE: dict[int, torch.Tensor] = {}


def hind_joint_pos_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] = (
        # URDF v2 names: hip-flex joints are now downstream of the abduction frame.
        "HR_hipabd_to_HR_thigh", "HR_thigh_to_HR_shin",
        "HL_hipabd_to_HL_thigh", "HL_thigh_to_HL_shin",
    ),
) -> torch.Tensor:
    """L2 penalty on hind leg joints only — forces hip extension to standing angles.

    The general joint_pos_standing_l2 averages across all joints, letting the policy
    keep hind hips horizontal while front legs compensate. This isolates the 4 hind
    joints so their deviation cannot be diluted.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    rid = id(robot)
    if rid not in _HIND_JOINT_IDX_CACHE:
        all_names = robot.joint_names
        idxs = [all_names.index(n) for n in joint_names if n in all_names]
        _HIND_JOINT_IDX_CACHE[rid] = torch.tensor(idxs, device=env.device, dtype=torch.long)
    idx = _HIND_JOINT_IDX_CACHE[rid]
    if len(idx) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    cur = robot.data.joint_pos[:, idx]
    default = robot.data.default_joint_pos[:, idx]
    return (cur - default).square().mean(dim=-1)


# ------------------------------------------------------------------
# Hind feet height: penalize hind feet being above ground (prevents tucking)
# ------------------------------------------------------------------
def hind_feet_height_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot"),
    target_z: float = 0.05,
) -> torch.Tensor:
    """Mean L2 penalty on hind feet only — catches back legs tucked under body."""
    each_z = _each_body_z(env, asset_cfg, list(foot_names))
    return (each_z - target_z).square().mean(dim=-1)


# ------------------------------------------------------------------
# Hind feet contact: fraction of hind feet bearing weight
# ------------------------------------------------------------------
def hind_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Fraction of hind feet with net contact force above threshold (0-1).

    Specifically targets the back legs that tend to tuck and avoid ground contact
    while front legs carry all the weight.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    norms = torch.norm(forces, dim=-1)
    in_contact = (norms > threshold).float()
    return in_contact.mean(dim=-1)


# ------------------------------------------------------------------
# Feet under body: penalize horizontal offset of foot centroid from pelvis
# ------------------------------------------------------------------
def feet_under_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot", "FL_foot", "FR_foot"),
    body_names: list[str] = ("pelvis",),
) -> torch.Tensor:
    """Squared horizontal distance between pelvis and foot centroid.

    Standing: feet roughly beneath pelvis -> small.
    Sitting: feet splayed forward -> large.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    foot_idx = _body_indices(robot, list(foot_names), env.device)
    body_idx = _body_indices(robot, list(body_names), env.device)
    if foot_idx is None or body_idx is None:
        return torch.zeros(env.num_envs, device=env.device)
    foot_xy = robot.data.body_pos_w[:, foot_idx, :2].mean(dim=1)
    body_xy = robot.data.body_pos_w[:, body_idx, :2].mean(dim=1)
    return ((body_xy - foot_xy) ** 2).sum(dim=-1)


# ------------------------------------------------------------------
# Feet below body (target gap): penalize if gap deviates from standing value
# ------------------------------------------------------------------
def feet_below_body_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot", "FL_foot", "FR_foot"),
    body_names: list[str] = ("pelvis",),
    target_gap: float = 0.28,
) -> torch.Tensor:
    """L2 penalty when vertical pelvis-to-foot gap deviates from ``target_gap``.

    IK standing pose has ~0.286m gap; sitting gives ~0.17m.
    """
    foot_z = _body_z(env, asset_cfg, list(foot_names))
    body_z = _body_z(env, asset_cfg, list(body_names))
    gap = body_z - foot_z
    return (gap - target_gap).square()


# ------------------------------------------------------------------
# Termination: pelvis too low (collapsed)
# ------------------------------------------------------------------
def pelvis_too_low(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_names: list[str] = ("pelvis",),
    min_height: float = 0.10,
) -> torch.Tensor:
    """Return True (terminate) when pelvis drops below min_height.

    On terrain, ``env.scene.env_origins[:, 2]`` carries the local tile
    elevation; we threshold pelvis_z RELATIVE to that, not absolute.
    On a flat plane env_origins[:, 2] is 0, so behavior is unchanged.
    """
    base_z = _body_z(env, asset_cfg, list(base_names))
    origin_z = env.scene.env_origins[:, 2]
    return (base_z - origin_z) < min_height


# ------------------------------------------------------------------
# Contact forces: weight on feet, not on torso (PhysX net contact forces)
# ------------------------------------------------------------------
def all_feet_in_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Fraction of feet with net contact force norm above ``threshold`` (0–1 per env).

    Uses unfiltered :attr:`ContactSensorData.net_forces_w` (GPU pipeline does not
    support ground-only filtering in current PhysX). For quadrupeds feet mainly
    contact the terrain.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    norms = torch.norm(forces, dim=-1)
    in_contact = (norms > threshold).float()
    return in_contact.mean(dim=-1)


def feet_swing_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.05,
    contact_threshold: float = 5.0,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Reward each foot for being lifted to ``target_height`` during swing phase.

    Walk-These-Ways uses this to prevent the policy from satisfying
    `feet_air_time` with feet barely-off-the-ground. Without this, a deer
    can earn air-time reward by lifting a foot 1 mm and putting it back —
    a "shuffle" or "rigid-front wheelbarrow" gait that doesn't match real
    locomotion.

    For each foot in SWING (not in contact):
        reward += exp(-(height - target_height)^2 / sigma^2)
    Stance feet contribute 0. Reward is summed across all feet, so a deer
    with all 4 feet hitting target during their respective swings gets the
    biggest reward.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = (torch.norm(forces, dim=-1) > contact_threshold).float()
    in_swing = 1.0 - in_contact   # (num_envs, num_feet)

    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]   # (num_envs, num_feet)
    # Account for terrain elevation (env_origin Z carries tile height; on flat = 0)
    origin_z = env.scene.env_origins[:, 2].unsqueeze(-1)        # (num_envs, 1)
    height_above_ground = foot_z - origin_z

    err = height_above_ground - target_height
    per_foot_reward = torch.exp(-(err.square()) / (sigma * sigma)) * in_swing
    return per_foot_reward.sum(dim=-1)


def trot_phase_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    foot_order: tuple[str, ...] = ("FL_foot", "RH_foot", "FR_foot", "LH_foot"),
) -> torch.Tensor:
    """Reward diagonal-pair alternation (trot pattern), penalize pronk.

    Diagnosis from r43b playback: deer learned to lift all four feet
    simultaneously (bunny hop / pronk) instead of alternating diagonal
    pairs like a real quadruped. ``feet_air_time`` rewarded foot lift
    but said nothing about WHICH feet.

    A trot has two diagonal pairs that alternate:
       Pair A = front-left + hind-right  (FL + HR)
       Pair B = front-right + hind-left  (FR + HL)

    When trotting:
       Within a pair: contact states match (both stance OR both swing).
       Between pairs: contact states differ (one stance, the other swing).

    Reward formula (per env, per step, with c_X in {0,1}):
       intra_diff = |c_FL - c_HR| + |c_FR - c_HL|        # want 0
       inter_diff = |c_FL - c_FR|                          # want 1
       reward = inter_diff - intra_diff
       range: [-2, +1] per step
       perfect trot phase A: c_FL=c_HR=1, c_FR=c_HL=0 -> 1 - 0 = +1
       perfect trot phase B: c_FL=c_HR=0, c_FR=c_HL=1 -> 1 - 0 = +1
       pronk (all 1 or all 0):                          -> 0 - 0 =  0
       random:                                          -> roughly 0

    The penalty for being out-of-phase within a diagonal pair is the
    *teeth* of this reward — without that the policy could still
    pronk with one diagonal in the air at random times.

    Pass ``sensor_cfg`` with body_names in this exact order:
        [FL_foot, HR_foot, FR_foot, HL_foot]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contact_sensor.data.net_forces_w shape: (num_envs, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    norms = torch.norm(forces, dim=-1)
    in_contact = (norms > threshold).float()
    # in_contact: (num_envs, 4) ordered [FL, HR, FR, HL]
    c_fl = in_contact[:, 0]
    c_hr = in_contact[:, 1]
    c_fr = in_contact[:, 2]
    c_hl = in_contact[:, 3]
    intra_diff = torch.abs(c_fl - c_hr) + torch.abs(c_fr - c_hl)
    inter_diff = torch.abs(c_fl - c_fr)
    return inter_diff - intra_diff


def trot_phase_reward_v2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    pronk_penalty: float = 0.5,
) -> torch.Tensor:
    """v2: like ``trot_phase_reward`` but ACTIVELY penalizes pronking
    (all 4 feet in same state).

    r44 showed the original formula gave pronk a reward of 0 (neutral),
    so the policy could satisfy every other reward with smooth pronking
    and never had to break the synchronized-feet local optimum. Per-
    step trot_phase plateaued at -0.00013 over 2000 iters of training.

    This version returns:
        trot_score - pronk_penalty * pronk_indicator

    where:
        trot_score = pair1_sync * pair2_sync * pairs_differ   (in [0,1])
        pronk_indicator = 1 if all 4 feet in same state, else 0

    Examples (FL, HR, FR, HL):
        Perfect trot phase A (1,1,0,0):    score=1, pronk=0  ->  +1.0
        Perfect trot phase B (0,0,1,1):    score=1, pronk=0  ->  +1.0
        Pronk grounded (1,1,1,1):          score=0, pronk=1  ->  -0.5
        Pronk in air   (0,0,0,0):          score=0, pronk=1  ->  -0.5
        Bound (1,0,1,0) front pair down:   score=0, pronk=0  ->   0.0
        Mixed/transition:                  score~0, pronk=0  ->   0.0

    Body order for sensor_cfg MUST be [FL_foot, HR_foot, FR_foot, HL_foot].
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    norms = torch.norm(forces, dim=-1)
    in_contact = (norms > threshold).float()
    c_fl = in_contact[:, 0]
    c_hr = in_contact[:, 1]
    c_fr = in_contact[:, 2]
    c_hl = in_contact[:, 3]

    # Pair-1 sync: FL == HR; Pair-2 sync: FR == HL.
    pair1_sync = 1.0 - torch.abs(c_fl - c_hr)
    pair2_sync = 1.0 - torch.abs(c_fr - c_hl)
    # Inter-pair differ: FL != FR.
    pairs_differ = torch.abs(c_fl - c_fr)
    trot_score = pair1_sync * pair2_sync * pairs_differ

    # Pronk: all 4 same state.
    pronk_grounded = c_fl * c_hr * c_fr * c_hl
    pronk_air = (1.0 - c_fl) * (1.0 - c_hr) * (1.0 - c_fr) * (1.0 - c_hl)
    pronk = pronk_grounded + pronk_air

    return trot_score - pronk_penalty * pronk


def body_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 200.0,
    max_violation: float = 400.0,
) -> torch.Tensor:
    """Penalize any contact force on torso bodies (pelvis sitting on ground).

    With ``enabled_self_collisions=False`` the only contact source for the pelvis
    is the terrain, so no foot-force gate is needed.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    norms = torch.norm(forces, dim=-1).max(dim=-1).values
    violation = torch.clamp(norms - threshold, min=0.0, max=max_violation)
    return violation


def min_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_force: float = 20.0,
) -> torch.Tensor:
    """Continuous penalty on the *least loaded* foot — forces every foot to bear weight.

    Returns ``clamp(1 - min_force / target_force, min=0)`` so the penalty is 1.0
    when the weakest foot has zero contact, smoothly decreases as force approaches
    ``target_force``, and is 0.0 when all feet bear at least ``target_force``.
    Continuous gradient avoids the dead-signal trap of a binary threshold.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    norms = torch.norm(forces, dim=-1)  # (N, num_feet)
    min_force = norms.min(dim=-1).values  # weakest foot per env
    return torch.clamp(1.0 - min_force / max(target_force, 1.0), min=0.0)


def worst_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot", "FL_foot", "FR_foot"),
    target_z: float = 0.05,
) -> torch.Tensor:
    """L2 penalty on the *highest* foot — catches tucked legs that escape the mean.

    ``feet_height_l2`` averages across feet so one bad foot is diluted.
    This term specifically penalizes the worst offender.
    """
    each_z = _each_body_z(env, asset_cfg, list(foot_names))
    max_z = each_z.max(dim=-1).values
    return (max_z - target_z).square()


def foot_weight_bearing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_force: float | None = None,
) -> torch.Tensor:
    """Reward ~1 when total foot contact force magnitude matches expected weight."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_norms = torch.norm(foot_forces, dim=-1)
    total = foot_norms.sum(dim=-1)

    if target_force is not None:
        tgt = torch.full_like(total, float(target_force))
    else:
        raw = os.environ.get("ANIML_DEER_TARGET_FOOT_FORCE", "").strip()
        if raw:
            tgt = torch.full_like(total, float(raw))
        else:
            # Approximate weight: default_mass may live on CPU in some builds — match contact tensor device.
            mg = robot.data.default_mass.sum(dim=-1).to(device=total.device, dtype=total.dtype) * 9.81
            tgt = mg

    return torch.clamp(total / torch.clamp(tgt, min=1.0), max=1.0)


# ------------------------------------------------------------------
# Leg extension: reward pelvis-to-foot vertical gap matching standing value.
# Standing gap ~0.89m. Splayed/lying gap ~0.15m. Cannot be faked.
# ------------------------------------------------------------------
def leg_extension_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot", "FL_foot", "FR_foot"),
    body_names: list[str] = ("pelvis",),
    target_gap: float = 0.89,
) -> torch.Tensor:
    """Reward when pelvis-to-foot vertical gap matches the standing value.

    gap = pelvis_Z - avg_foot_Z.
    Standing: ~0.89m -> reward near 1.0.
    Splayed flat: ~0.15m -> reward near 0.0.
    Uses exp(-error^2 / sigma^2) so reward is smooth and dense.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    foot_idx = _body_indices(robot, list(foot_names), env.device)
    body_idx = _body_indices(robot, list(body_names), env.device)
    if foot_idx is None or body_idx is None:
        return torch.zeros(env.num_envs, device=env.device)
    avg_foot_z = robot.data.body_pos_w[:, foot_idx, 2].mean(dim=1)
    pelvis_z = robot.data.body_pos_w[:, body_idx, 2].mean(dim=1)
    gap = pelvis_z - avg_foot_z
    error = gap - target_gap
    return torch.exp(-error.square() / 0.1)  # sigma=0.316m, ~0 when gap off by >0.5m


def feet_under_chest(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_foot_names: list[str] = ("FL_foot", "FR_foot"),
    chest_names: list[str] = ("chest",),
) -> torch.Tensor:
    """Penalize Y-offset between chest and front feet.

    When front feet are directly under the chest (standing properly), Y-offset ~ 0.
    When front feet are splayed forward, Y-offset is large and positive.
    This directly fixes the 'leaning forward on angled front legs' exploit.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    foot_idx = _body_indices(robot, list(front_foot_names), env.device)
    chest_idx = _body_indices(robot, list(chest_names), env.device)
    if foot_idx is None or chest_idx is None:
        return torch.zeros(env.num_envs, device=env.device)
    # Y = forward direction in Isaac frame
    avg_foot_y = robot.data.body_pos_w[:, foot_idx, 1].mean(dim=1)
    chest_y = robot.data.body_pos_w[:, chest_idx, 1].mean(dim=1)
    y_offset = avg_foot_y - chest_y  # positive = feet forward of chest
    return y_offset.square()


# ------------------------------------------------------------------
# Head height: penalize skull dropping below a minimum world Z.
# Prevents butt-in-air / head-down postures.
# ------------------------------------------------------------------
def head_min_height_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_names: list[str] = ("skull",),
    min_height: float = 0.90,
) -> torch.Tensor:
    """L2 penalty when skull world-Z drops below min_height.

    Standing deer skull is ~1.4m up. Butt-in-air pose drops head to ~0.5m.
    Penalising below 0.9m blocks that exploit without constraining normal posture.
    """
    head_z = _body_z(env, asset_cfg, list(head_names))
    violation = torch.clamp(min_height - head_z, min=0.0)
    return violation.square()


# ------------------------------------------------------------------
# Foot Z height: reward all four feet being at ground level (Z near 0)
# ------------------------------------------------------------------
def all_feet_near_ground(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names: list[str] = ("HL_foot", "HR_foot", "FL_foot", "FR_foot"),
    target_z: float = 0.03,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Reward when all four feet are close to ground level (world Z near target_z).

    Returns fraction of feet within tolerance of target_z, so range is [0, 1].
    Complements contact-force rewards: if no contact sensors, this still drives
    the policy to place feet on the ground rather than dangling them.
    """
    each_z = _each_body_z(env, asset_cfg, list(foot_names))
    near = (torch.abs(each_z - target_z) < tolerance).float()
    return near.mean(dim=-1)


# ------------------------------------------------------------------
# Curriculum terms: linear reward-weight ramps
# ------------------------------------------------------------------

def velocity_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """L2 penalty on XY velocity error. Unlike exp-tracking, this is never zero
    when the deer is stationary and a non-zero command is active.
    Returns negative values (penalty); weight should be negative."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_vel = cmd[:, :2]                                      # (N, 2) commanded XY vel
    body_vel = asset.data.root_lin_vel_b[:, :2]               # (N, 2) actual XY vel in body frame
    return torch.sum((cmd_vel - body_vel) ** 2, dim=-1)       # (N,)


# ------------------------------------------------------------------
# v4.2: skull pitch L2 — penalizes chin-tuck behavior.
#
# `head_above_base` only catches skull POSITION vs pelvis (the deer can
# satisfy that by tucking the chin while keeping head overall raised).
# This term penalizes the actual skull JOINT angle deviating from level,
# which is the direct fix for the chin-tuck-down failure mode observed
# in v4.1 trained policy.
# ------------------------------------------------------------------
def skull_pitch_l2(
    env: ManagerBasedRLEnv,
    joint_name: str = "neck_to_skull",
) -> torch.Tensor:
    """Penalty: square of skull joint position. Encourages level head.

    With weight -1.0, the deer pays ~0.04 per step for a 0.2 rad chin tuck.
    Should drive the skull joint toward neutral while the upstream neck
    handles drinking/browsing reach.
    """
    robot: Articulation = env.scene["robot"]
    cache_key = f"_skull_pitch_idx_{joint_name}"
    if not hasattr(env, cache_key) or getattr(env, cache_key) is None:
        idx = robot.data.joint_names.index(joint_name)
        setattr(env, cache_key, idx)
    idx = getattr(env, cache_key)
    return robot.data.joint_pos[:, idx].square()


def neck_posture_track(
    env: ManagerBasedRLEnv,
    joint_targets: tuple[tuple[str, float], ...] = (
        ("neck_yaw_to_neck", -0.225),  # FK-verified head-UP (URDF sign is flipped
        ("neck_to_skull",     0.326),  # vs the reference clip's convention)
    ),
    sigma: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v7: imitate the reference's NECK/HEAD carriage (completes the imitation).

    joint_track_reference_walk tracks ONLY the 12 leg joints, leaving the neck
    unconstrained -> the head droops to the ground while the legs match the gait,
    and joint_track sees nothing wrong (the collapse root cause). The reference
    (deer_walk.npy) walks with the head UP: chest_to_neck +12.9deg, neck_to_skull
    -18.7deg, level body. This rewards the URDF neck joints matching that
    carriage, so head-up is part of the imitation itself rather than something we
    fight for with ad-hoc head-height penalties. exp kernel -> reward in
    [0, n_joints]; pair with a POSITIVE weight.

    NOTE: neck sign convention (reference -> URDF) is a first guess; verify early
    via head_min_height (sign-free skull world-Z) and flip targets if the head
    drops instead of rising.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    cache = "_neck_posture_idx"
    if not hasattr(env, cache):
        names = [n for n, _ in joint_targets]
        idxs = torch.tensor(
            [robot.data.joint_names.index(n) for n in names],
            device=env.device, dtype=torch.long,
        )
        tgts = torch.tensor([t for _, t in joint_targets], device=env.device)
        setattr(env, cache, (idxs, tgts))
    idxs, tgts = getattr(env, cache)
    q = robot.data.joint_pos[:, idxs]
    return torch.exp(-(q - tgts.unsqueeze(0)).square() / (sigma ** 2)).sum(dim=-1)


# ------------------------------------------------------------------
# v4.2: knee flex during swing — direct positive incentive for visible
# knee bending during foot lift. Without this, the policy converges to
# a hip-only stilt walk (knees stay near default position throughout
# the gait cycle, achieving foot clearance via hip rotation alone).
# This rewards the knee joint deviating from its default angle WHEN the
# foot is supposed to be in the swing phase per the gait clock.
# ------------------------------------------------------------------
def knee_flex_during_swing(
    env: ManagerBasedRLEnv,
    knee_joint_names: tuple[str, ...] = (
        "FL_thigh_to_FL_shin",
        "HR_thigh_to_HR_shin",
        "FR_thigh_to_FR_shin",
        "HL_thigh_to_HL_shin",
    ),
    knee_default_pos: tuple[float, ...] = (0.5, -0.7, 0.5, -0.7),
    frequency: float = 2.0,
    foot_phase_offsets: tuple[float, ...] = (0.0, 0.0, 0.5, 0.5),
    duty_cycle: float = 0.5,
    flex_target: float = 0.4,
) -> torch.Tensor:
    """Reward knee flexion during gait swing phase.

    For each leg, when the foot is supposed to be in swing (phase >=
    duty_cycle), reward the knee joint angle deviating from its default
    by ~flex_target rad. tanh-saturated so the reward is bounded per leg.

    knee_joint_names ORDER MUST MATCH foot_phase_offsets order (the gait
    clock uses [FL, HR, FR, HL] for trot).

    Returns shape (num_envs,) in [0, num_legs]. Pair with positive weight
    (suggested ~+1.0 to +1.5).
    """
    # Compute foot phases (same as gait_contact_match for consistency)
    t = env.episode_length_buf.float() * env.step_dt
    base_phase = (t * frequency) % 1.0
    offsets = torch.tensor(list(foot_phase_offsets), device=env.device, dtype=torch.float32)
    foot_phases = (base_phase.unsqueeze(-1) + offsets.unsqueeze(0)) % 1.0  # (N, num_legs)
    in_swing = (foot_phases >= duty_cycle).float()

    # Knee joint indices (cache per-env after first call)
    robot: Articulation = env.scene["robot"]
    cache_key = "_knee_flex_idx"
    if not hasattr(env, cache_key) or getattr(env, cache_key) is None:
        all_joints = robot.data.joint_names
        idxs = torch.tensor(
            [all_joints.index(n) for n in knee_joint_names],
            device=env.device, dtype=torch.long,
        )
        setattr(env, cache_key, idxs)
    knee_idxs = getattr(env, cache_key)
    knee_pos = robot.data.joint_pos[:, knee_idxs]                        # (N, num_legs)
    knee_defaults = torch.tensor(list(knee_default_pos), device=env.device, dtype=torch.float32)
    knee_flex = (knee_pos - knee_defaults.unsqueeze(0)).abs()             # (N, num_legs)
    flex_quality = torch.tanh(knee_flex / flex_target)                    # in [0, 1)

    return (in_swing * flex_quality).sum(dim=-1)                          # (N,) in [0, num_legs]


def stationary_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    min_cmd_speed: float = 0.3,
) -> torch.Tensor:
    """Penalize near-zero body speed when a significant velocity command is active.
    Only fires when |cmd_vel| > min_cmd_speed and body speed < min_cmd_speed.
    This directly punishes the standing-still local optimum."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=-1)                 # (N,)
    body_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)  # (N,)
    # Only penalize when a real command exists but body isn't moving
    active = (cmd_speed > min_cmd_speed).float()
    not_moving = torch.clamp(min_cmd_speed - body_speed, min=0.0)
    return active * not_moving                                 # (N,) — weight should be negative


class LinearRampCommandParam(ManagerTermBase):
    """Linearly ramp a command manager parameter from start_val to end_val.

    Usage::

        self.curriculum.ramp_standing = CurrTerm(
            func=LinearRampCommandParam,
            params={
                "command_name": "base_velocity",
                "param_name": "rel_standing_envs",
                "start_val": 1.0,
                "end_val": 0.1,
                "ramp_start_step": 0,
                "ramp_end_step": 1_800_000,
            },
        )
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._cmd_name: str = cfg.params["command_name"]
        self._param: str = cfg.params["param_name"]
        self._start_v: float = cfg.params["start_val"]
        self._end_v: float = cfg.params["end_val"]
        self._start_step: int = cfg.params["ramp_start_step"]
        self._end_step: int = cfg.params["ramp_end_step"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        param_name: str,
        start_val: float,
        end_val: float,
        ramp_start_step: int,
        ramp_end_step: int,
    ) -> float:
        step = env.common_step_counter
        if step <= self._start_step:
            new_v = self._start_v
        elif step >= self._end_step:
            new_v = self._end_v
        else:
            t = (step - self._start_step) / (self._end_step - self._start_step)
            new_v = self._start_v + t * (self._end_v - self._start_v)

        cmd_term = env.command_manager.get_term(self._cmd_name)
        if hasattr(cmd_term.cfg, self._param):
            setattr(cmd_term.cfg, self._param, new_v)
        if hasattr(cmd_term, self._param):
            setattr(cmd_term, self._param, new_v)
        return new_v


class LinearRampRewardWeight(ManagerTermBase):
    """Linearly ramp a reward term's weight from start_weight to end_weight
    over [ramp_start_step, ramp_end_step] environment steps.

    Before ramp_start_step the weight is held at start_weight.
    After ramp_end_step the weight is held at end_weight.

    Usage in env cfg::

        from training.envs.deer_rewards import LinearRampRewardWeight
        from isaaclab.managers import CurriculumTermCfg as CurrTerm

        self.curriculum.ramp_velocity = CurrTerm(
            func=LinearRampRewardWeight,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "start_weight": 0.3,
                "end_weight": 1.5,
                "ramp_start_step": 0,
                "ramp_end_step": 500_000,   # env steps (not iters)
            },
        )
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_name: str = cfg.params["term_name"]
        self._start_w: float = cfg.params["start_weight"]
        self._end_w: float = cfg.params["end_weight"]
        self._start_step: int = cfg.params["ramp_start_step"]
        self._end_step: int = cfg.params["ramp_end_step"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        start_weight: float,
        end_weight: float,
        ramp_start_step: int,
        ramp_end_step: int,
    ) -> float:
        step = env.common_step_counter
        if step <= self._start_step:
            new_w = self._start_w
        elif step >= self._end_step:
            new_w = self._end_w
        else:
            t = (step - self._start_step) / (self._end_step - self._start_step)
            new_w = self._start_w + t * (self._end_w - self._start_w)

        term_cfg = env.reward_manager.get_term_cfg(self._term_name)
        term_cfg.weight = new_w
        env.reward_manager.set_term_cfg(self._term_name, term_cfg)
        return new_w


# ---------------------------------------------------------------------------
# Anatomy-aligned reward + termination functions (used by AMP v5+)
# ---------------------------------------------------------------------------
# The deer URDF was authored with:
#   +X body = anatomical LEFT     (HL_thigh at +X, HR_thigh at -X)
#   +Y body = anatomical DOWN     (feet at ~+0.6 below pelvis)
#   -Z body = anatomical FORWARD  (skull at -0.94 from pelvis)
# Isaac Lab's standard locomotion rewards assume +X body = forward, which is
# incompatible with this URDF's native frame. These helpers re-project
# velocities onto the deer's anatomical axes so the command manager's
# "lin_vel_x" actually means "anatomical forward speed."


def track_forward_vel_anatomical_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """Exponential tracking of the commanded forward velocity.

    cmd[:, 0] is treated as commanded anatomical-forward speed.
    Actual anatomical forward speed = -root_lin_vel_b[:, 2]  (body -Z axis).
    Returns exp(-err^2 / std^2), shape (N,), always positive. Use positive weight.
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_fwd = cmd[:, 0]
    actual_fwd = -asset.data.root_lin_vel_b[:, 2]
    err2 = (cmd_fwd - actual_fwd) ** 2
    return torch.exp(-err2 / (std * std))


def track_lateral_vel_anatomical_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """Exponential tracking of the commanded lateral (anatomical right) velocity.

    cmd[:, 1] is treated as commanded anatomical-right speed (positive = right).
    Actual anatomical right speed = -root_lin_vel_b[:, 0]  (body -X axis).
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_right = cmd[:, 1]
    actual_right = -asset.data.root_lin_vel_b[:, 0]
    err2 = (cmd_right - actual_right) ** 2
    return torch.exp(-err2 / (std * std))


def track_yaw_rate_anatomical_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """Exponential tracking of commanded yaw rate around anatomical-up axis.

    cmd[:, 2] is commanded yaw rate (positive = turn left = CCW viewed from above).
    Anatomical up = body -Y. Yaw rate in that frame = -root_ang_vel_b[:, 1].
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_yaw = cmd[:, 2]
    actual_yaw = -asset.data.root_ang_vel_b[:, 1]
    err2 = (cmd_yaw - actual_yaw) ** 2
    return torch.exp(-err2 / (std * std))


# ------------------------------------------------------------------
# v4.4: L2 (unbounded-gradient) yaw/lateral anchoring.
# The exp-weighted track_*_exp terms above saturate at large errors —
# err2/std^2 = 4 already gives reward ~0.018, and beyond that the
# gradient is essentially flat. v4.3 exhibited 13°/s yaw drift at zero
# command, meaning the policy was sitting in the saturated tail and
# getting no pull back to straight. These L2 versions have linear
# gradient in error magnitude, so the further off-track the policy
# drifts, the harder these terms pull it back.
# Use with negative weight (cost). Sign-symmetric — punishes left and
# right drift equally.
# ------------------------------------------------------------------
def track_yaw_rate_anatomical_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Squared yaw-rate tracking error. Apply with negative weight."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_yaw = cmd[:, 2]
    actual_yaw = -asset.data.root_ang_vel_b[:, 1]
    return (cmd_yaw - actual_yaw).square()


def track_lateral_vel_anatomical_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Squared lateral-velocity tracking error. Apply with negative weight."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_right = cmd[:, 1]
    actual_right = -asset.data.root_lin_vel_b[:, 0]
    return (cmd_right - actual_right).square()


# ------------------------------------------------------------------
# v4.5: Carpus deep-flex bonus (deadbanded linear).
# Companion to `knee_flex_during_swing` (which uses bounded tanh and saturates
# near 23-29 deg of flex, letting the policy park there for near-max reward).
# This term gives ZERO reward below `deadband` rad of flex and ramps linearly
# to 1.0 across the next `scale` rad. Policy can't game it by oscillating
# shallow — it has to actually fold the cannon past the threshold.
# Use with positive weight, applied to the visible-knee carpus joints
# (shin->foot in our URDF, anatomically the wrist on front legs and the
# hock on hind legs).
# ------------------------------------------------------------------
def carpus_deep_flex_bonus(
    env: ManagerBasedRLEnv,
    carpus_joint_names: tuple = (
        "FL_shin_to_FL_foot", "HR_shin_to_HR_foot",
        "FR_shin_to_FR_foot", "HL_shin_to_HL_foot",
    ),
    frequency: float = 2.0,
    foot_phase_offsets: tuple = (0.0, 0.0, 0.5, 0.5),
    duty_cycle: float = 0.5,
    deadband: float = 0.5,   # rad — ~29 deg threshold
    scale: float = 0.6,      # rad above deadband for full reward
) -> torch.Tensor:
    """Deadbanded linear carpus flex bonus during swing phase.

    Returns 0 when |flex| < deadband; ramps linearly to 1.0 as flex goes from
    `deadband` to `deadband + scale`; clamps at 1.0 above. Sums across the
    4 carpus joints, masked by per-leg swing-phase indicator.
    """
    robot: Articulation = env.scene["robot"]
    cache_key = "_carpus_deep_idx_" + "_".join(carpus_joint_names)
    if not hasattr(env, cache_key):
        idxs = torch.tensor(
            [robot.data.joint_names.index(j) for j in carpus_joint_names],
            device=env.device, dtype=torch.long,
        )
        defaults = robot.data.default_joint_pos[0, idxs].clone()
        setattr(env, cache_key, (idxs, defaults))
    idxs, defaults = getattr(env, cache_key)

    t = env.episode_length_buf.float() * env.step_dt
    base_phase = (t * frequency) % 1.0
    offsets = torch.tensor(list(foot_phase_offsets), device=env.device, dtype=torch.float32)
    foot_phases = (base_phase.unsqueeze(-1) + offsets.unsqueeze(0)) % 1.0
    in_swing = (foot_phases >= duty_cycle).float()

    q = robot.data.joint_pos[:, idxs]
    flex = (q - defaults.unsqueeze(0)).abs()
    flex_past = torch.clamp(flex - deadband, min=0.0)
    flex_quality = torch.clamp(flex_past / scale, max=1.0)

    return (in_swing * flex_quality).sum(dim=-1)


# ------------------------------------------------------------------
# v4.5: Bilateral joint-deviation symmetry penalty (EMA-based).
# Penalizes sustained L/R asymmetry of joint deviations from default.
# A symmetric trot has L and R reaching equal peak deviations with phase
# shift; the EMA washes out phase, leaving only magnitude asymmetry.
# Apply with NEGATIVE weight (cost). Persistent EMA state per env;
# ~50-step half-life (decay 0.95) is short enough to refresh post-reset
# without explicit hooks (1000-step episodes).
# ------------------------------------------------------------------
def bilateral_joint_symmetry_ema(
    env: ManagerBasedRLEnv,
    joint_pairs: tuple = (
        ("HL_thigh_to_HL_shin",   "HR_thigh_to_HR_shin"),    # hind knee
        ("FL_thigh_to_FL_shin",   "FR_thigh_to_FR_shin"),    # front "elbow"
        ("HL_shin_to_HL_foot",    "HR_shin_to_HR_foot"),     # hind hock
        ("FL_shin_to_FL_foot",    "FR_shin_to_FR_foot"),     # front carpus
        ("HL_hipabd_to_HL_thigh", "HR_hipabd_to_HR_thigh"),  # hind hip flex
        ("FL_hipabd_to_FL_thigh", "FR_hipabd_to_FR_thigh"),  # front shoulder flex
    ),
    decay: float = 0.95,
) -> torch.Tensor:
    """Squared L/R EMA-deviation difference, summed across joint pairs.

    NOTE (v4.5 retrospective): this function FAILED to reduce asymmetry in
    v4.5. Root cause: PPO's value function can't see the EMA state (it's
    not in the observation), so the policy can't predict how current
    actions affect future EMA-based reward. The effective gradient signal
    for asymmetry reduction is ~5% of the nominal weight (only the
    instantaneous (1-decay) contribution flows back). v4.6+ uses
    `bilateral_joint_symmetry_l2` (per-step, no EMA, Markov-clean).
    Kept here for archive / reproducibility of v4.5 behavior.
    """
    robot: Articulation = env.scene["robot"]
    cache_key = "_bilateral_sym_state"
    if not hasattr(env, cache_key):
        L_names = [p[0] for p in joint_pairs]
        R_names = [p[1] for p in joint_pairs]
        L_idxs = torch.tensor(
            [robot.data.joint_names.index(n) for n in L_names],
            device=env.device, dtype=torch.long,
        )
        R_idxs = torch.tensor(
            [robot.data.joint_names.index(n) for n in R_names],
            device=env.device, dtype=torch.long,
        )
        L_defs = robot.data.default_joint_pos[0, L_idxs].clone()
        R_defs = robot.data.default_joint_pos[0, R_idxs].clone()
        setattr(env, cache_key, {
            "L_idxs": L_idxs, "R_idxs": R_idxs,
            "L_defs": L_defs, "R_defs": R_defs,
            "ema_L": torch.zeros(env.num_envs, len(joint_pairs), device=env.device),
            "ema_R": torch.zeros(env.num_envs, len(joint_pairs), device=env.device),
        })
    state = getattr(env, cache_key)

    q = robot.data.joint_pos
    abs_dev_L = (q[:, state["L_idxs"]] - state["L_defs"].unsqueeze(0)).abs()
    abs_dev_R = (q[:, state["R_idxs"]] - state["R_defs"].unsqueeze(0)).abs()

    state["ema_L"] = decay * state["ema_L"] + (1.0 - decay) * abs_dev_L
    state["ema_R"] = decay * state["ema_R"] + (1.0 - decay) * abs_dev_R

    return (state["ema_L"] - state["ema_R"]).square().sum(dim=-1)


# ------------------------------------------------------------------
# v4.6: Per-step bilateral L/R deviation symmetry penalty (no EMA).
# Replaces the v4.5 EMA-based version, which failed because PPO's
# value function couldn't see the EMA state. This per-step variant
# is Markov-clean: at each step, penalize the squared difference of
# *instantaneous* magnitudes |q_L - default_L| and |q_R - default_R|.
# Phase-shifted alternation in a symmetric trot still has |L| and |R|
# tracking each other in MAGNITUDE (just at offset times) over enough
# steps, so a sustained asymmetric drive shows up as a positive
# expected value of (|L| - |R|)^2 across all time.
# Apply with NEGATIVE weight (cost).
# ------------------------------------------------------------------
def bilateral_joint_symmetry_l2(
    env: ManagerBasedRLEnv,
    joint_pairs: tuple = (
        ("HL_thigh_to_HL_shin",   "HR_thigh_to_HR_shin"),    # hind knee
        ("FL_thigh_to_FL_shin",   "FR_thigh_to_FR_shin"),    # front "elbow"
        ("HL_shin_to_HL_foot",    "HR_shin_to_HR_foot"),     # hind hock
        ("FL_shin_to_FL_foot",    "FR_shin_to_FR_foot"),     # front carpus
        ("HL_hipabd_to_HL_thigh", "HR_hipabd_to_HR_thigh"),  # hind hip flex
        ("FL_hipabd_to_FL_thigh", "FR_hipabd_to_FR_thigh"),  # front shoulder flex
    ),
) -> torch.Tensor:
    """Per-step squared L/R |q-default| difference, summed across pairs."""
    robot: Articulation = env.scene["robot"]
    cache_key = "_bilateral_sym_l2_idx"
    if not hasattr(env, cache_key):
        L_names = [p[0] for p in joint_pairs]
        R_names = [p[1] for p in joint_pairs]
        L_idxs = torch.tensor(
            [robot.data.joint_names.index(n) for n in L_names],
            device=env.device, dtype=torch.long,
        )
        R_idxs = torch.tensor(
            [robot.data.joint_names.index(n) for n in R_names],
            device=env.device, dtype=torch.long,
        )
        L_defs = robot.data.default_joint_pos[0, L_idxs].clone()
        R_defs = robot.data.default_joint_pos[0, R_idxs].clone()
        setattr(env, cache_key, (L_idxs, R_idxs, L_defs, R_defs))
    L_idxs, R_idxs, L_defs, R_defs = getattr(env, cache_key)

    q = robot.data.joint_pos
    abs_dev_L = (q[:, L_idxs] - L_defs.unsqueeze(0)).abs()
    abs_dev_R = (q[:, R_idxs] - R_defs.unsqueeze(0)).abs()

    return (abs_dev_L - abs_dev_R).square().sum(dim=-1)


def forward_velocity_linear_anatomical(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    clip_max: float = 1.5,
) -> torch.Tensor:
    """Linear reward for forward velocity along the anatomical-forward axis.

    Projects actual velocity onto the commanded-forward direction (body -Z),
    clamped to the magnitude of the command. Gives a non-zero gradient
    everywhere, unlike exp tracking which saturates at large error.

    Only fires when |cmd_fwd| > 1e-3 (zero command = zero reward, no
    pressure to move).
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_fwd = cmd[:, 0]
    actual_fwd = -asset.data.root_lin_vel_b[:, 2]

    abs_cmd = torch.abs(cmd_fwd)
    active = (abs_cmd > 1e-3).float()
    # Projection with sign: moving in the commanded direction is positive.
    projection = actual_fwd * torch.sign(cmd_fwd)
    reward = torch.clamp(projection, min=-clip_max)
    reward = torch.minimum(reward, abs_cmd)
    return reward * active


def leg_joint_activity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the mean absolute velocity of the 12 leg joints.

    Directly punishes "rigid-lever" local optima where the policy holds legs
    static and bounces forward. We target the 12 leg joints (hip / shin / foot
    × 4 legs), excluding spine / neck / skull which naturally stay still.

    Returns mean |joint_vel| across those 12 joints, in rad/s. Use positive
    weight around 0.3-0.5 so 1 rad/s of average leg motion earns ~0.4 reward
    per step (~100/episode at 250 steps).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # URDF v2: 12 X-axis flex/extension joints (hip flex, knee, ankle per leg).
    # The 4 NEW HAA joints are excluded — they're for lateral balance, not
    # forward propulsion, so we don't want this reward to punish their stillness.
    leg_joint_names = [
        "HR_hipabd_to_HR_thigh", "HR_thigh_to_HR_shin", "HR_shin_to_HR_foot",
        "HL_hipabd_to_HL_thigh", "HL_thigh_to_HL_shin", "HL_shin_to_HL_foot",
        "FR_hipabd_to_FR_thigh", "FR_thigh_to_FR_shin", "FR_shin_to_FR_foot",
        "FL_hipabd_to_FL_thigh", "FL_thigh_to_FL_shin", "FL_shin_to_FL_foot",
    ]
    name_to_idx = {n: i for i, n in enumerate(asset.data.joint_names)}
    leg_ids = torch.tensor(
        [name_to_idx[n] for n in leg_joint_names],
        dtype=torch.long, device=asset.device,
    )
    joint_vel = asset.data.joint_vel[:, leg_ids]        # (N, 12)
    return joint_vel.abs().mean(dim=-1)                  # (N,)


def pelvis_height_ramp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_names: list[str] = ("pelvis",),
    z_min: float = 0.55,
    z_max: float = 1.00,
) -> torch.Tensor:
    """Linear height-of-standing reward, dense gradient between thresholds.

    Returns ``clip((pelvis_z - z_min) / (z_max - z_min), 0, 1)`` per env.
    Replaces the binary ``upright_bonus`` so the policy can't camp at
    pelvis_z right above the termination cutoff. Every cm of standing up
    is worth more reward, with no flat zone to nest in.
    """
    base_z = _body_z(env, asset_cfg, list(base_names))
    return torch.clamp((base_z - z_min) / max(z_max - z_min, 1e-3), 0.0, 1.0)


def nose_pitch_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body pitch (nose down OR nose up).

    With the URDF axis convention (body +Y = anatomical down, body -Z =
    anatomical forward), an upright + level deer has
    ``projected_gravity_b ≈ (0, +1, 0)`` — no projection on the X (lateral)
    or Z (forward) axes. A nose-down pitch rotates the forward axis off
    horizontal, producing a non-zero Z component in projected_gravity_b.

    Return value is ``|projected_gravity_b[:, 2]|`` per env (range [0, 1]).
    Use NEGATIVE weight (e.g. -2.0) so pitch is penalized.

    This is independent of the lateral roll captured by
    ``base_orientation_too_tilted`` — it specifically attacks the
    'nose-down strut' geometry seen in r11b/r12 where the body tilts
    just under the global tilt cutoff to game the reward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b  # (N, 3)
    return gravity_b[:, 2].abs()                # (N,)


def base_orientation_too_tilted(
    env: ManagerBasedRLEnv,
    max_angle_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination: fire when the body's anatomical-up axis is > max_angle_deg
    away from world up.

    When upright, body +Y is gravity-aligned (anatomical down). So
    ``projected_gravity_b ≈ (0, 1, 0)``. The more the body tilts, the more
    this vector rotates away from the Y-axis. We compare against
    ``cos(max_angle_deg)``.
    """
    import math as _math
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b  # (N, 3), unit-length
    # Dot product with body +Y; == 1 when upright, 0 on the side, -1 upside-down.
    cos_tilt = gravity_b[:, 1]
    cos_threshold = _math.cos(_math.radians(max_angle_deg))
    return cos_tilt < cos_threshold


# ---------------------------------------------------------------------------
# v4.13: hard joint-angle range penalty (hind knees specifically)
# ---------------------------------------------------------------------------
# Across v4.7-v4.12 the HR knee was stuck at mean -55 to -61 deg (reference
# -14, ref range -28 to 0). Joint tracking with both exp and Cauchy reward
# shapes failed; RSI moved it only ~5 deg. The policy is in a structural
# equilibrium where HR carries more load than HL during diagonal-pair
# trot, so HR has to deeply flex.
#
# The Cauchy/exp tracking rewards SATURATE — even with a clean gradient
# at all error scales, the policy is choosing this configuration because
# it's stable. We need a HARD penalty that grows quadratically when the
# joint goes BEYOND the reference's actual range, so any deeper flex
# costs more than the equilibrium savings.
#
# Applied to BOTH hind knees by symmetry. HL is currently within range,
# so the penalty only fires on HR.
def anti_parallel_thigh_l2(
    env: ManagerBasedRLEnv,
    thigh_pairs: tuple[tuple[str, str], ...] = (
        ("HL_hipabd_to_HL_thigh", "HR_hipabd_to_HR_thigh"),  # hind pair
        ("FL_hipabd_to_FL_thigh", "FR_hipabd_to_FR_thigh"),  # front pair
    ),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v4.28: directly penalize parallel hindquarter motion.

    Across v4.7-v4.27, bilateral_symmetry enforced |L-default| = |R-default|
    in magnitude but did NOT distinguish mirror (L_dev = -R_dev) from
    parallel (L_dev = +R_dev). Policy ended up with HL/HR hip correlation
    -0.22 vs walk reference -0.93 — partly parallel.

    This term penalizes (L_dev + R_dev)^2 per pair. For a mirror gait
    (L_dev = -R_dev), the sum is 0 → no penalty. For parallel (L_dev =
    +R_dev), the sum is 2*L_dev → quadratic penalty. Targets the
    "//" parallel-leg gait directly.

    Returns (N,) sum of squared (L+R-2*default) across pairs.
    Pair with negative weight (suggested -1.0 to -2.0).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    cache_key = "_anti_parallel_thigh_idx_" + "_".join(
        n for pair in thigh_pairs for n in pair
    )
    if not hasattr(env, cache_key):
        L_names = [p[0] for p in thigh_pairs]
        R_names = [p[1] for p in thigh_pairs]
        L_idxs = torch.tensor(
            [robot.data.joint_names.index(n) for n in L_names],
            device=env.device, dtype=torch.long,
        )
        R_idxs = torch.tensor(
            [robot.data.joint_names.index(n) for n in R_names],
            device=env.device, dtype=torch.long,
        )
        L_defs = robot.data.default_joint_pos[0, L_idxs].clone()
        R_defs = robot.data.default_joint_pos[0, R_idxs].clone()
        setattr(env, cache_key, (L_idxs, R_idxs, L_defs, R_defs))
    L_idxs, R_idxs, L_defs, R_defs = getattr(env, cache_key)

    q = robot.data.joint_pos
    # Note: URDF has MIRRORED joint axes for hip flexion (per CLAUDE.md),
    # which means same q on both sides = mirror physical pose. So mirror
    # gait has q_L = q_R (NOT q_L = -q_R). We want q_L - L_def == q_R - R_def
    # for mirror, and q_L - L_def != q_R - R_def for parallel.
    # Actually — for mirror PHYSICAL pose under mirrored axes: q_L == q_R.
    # In a TROTTING/WALKING gait, the swing-stance offset means q_L cycles
    # vs q_R with a phase shift. At any moment they SHOULD be different.
    # Anti-correlation in time means: at any instant, L_dev and R_dev have
    # opposite signs (one in swing forward, other in stance back).
    # So we want L_dev + R_dev ≈ 0 at any instant (when defaults equal,
    # this means q_L + q_R ≈ 2*default → q_L ≈ -q_R when default=0).
    L_dev = q[:, L_idxs] - L_defs.unsqueeze(0)
    R_dev = q[:, R_idxs] - R_defs.unsqueeze(0)
    parallel_sum = L_dev + R_dev  # 0 for mirror, large for parallel
    return parallel_sum.square().sum(dim=-1)  # (N,) — pair with negative weight


def hipabd_zero_l2(
    env: ManagerBasedRLEnv,
    hipabd_joint_names: tuple[str, ...] = (
        "pelvis_to_HL_hipabd", "pelvis_to_HR_hipabd",
        "chest_to_FL_hipabd", "chest_to_FR_hipabd",
    ),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v4.17: penalize squared deviation of hip abduction joints from 0.

    v4.16 added hip ABDUCTION joints to bilateral_joint_symmetry, which
    forced L and R to mirror each other. But the policy then chose a
    COMMON-MODE offset where both shoulders cocked the SAME direction
    (front L=+5-8°, R=+10-22°). Symmetric in `|L-R|` sense but both
    nonzero — body leans → lateral drift → "stepping sideways".

    This term penalizes magnitude, pushing each hipabd toward 0 unless
    the policy actively needs abduction (e.g., commanded lateral motion).
    Pair with `bilateral_joint_symmetry_l2` (which still enforces L=R)
    so the combined effect is L=R=0.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    cache_key = "_hipabd_zero_idx_" + "_".join(hipabd_joint_names)
    if not hasattr(env, cache_key):
        idxs = torch.tensor(
            [robot.data.joint_names.index(n) for n in hipabd_joint_names],
            device=env.device, dtype=torch.long,
        )
        setattr(env, cache_key, idxs)
    idxs = getattr(env, cache_key)
    q = robot.data.joint_pos[:, idxs]                            # (N, n_abd)
    return q.square().sum(dim=-1)                                # (N,) — pair with negative weight


def _quat_to_yaw_w(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Yaw angle (rad) around world Z from a wxyz quaternion. Vectorized.
    yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2)).
    """
    qw = quat_wxyz[..., 0]
    qx = quat_wxyz[..., 1]
    qy = quat_wxyz[..., 2]
    qz = quat_wxyz[..., 3]
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny, cosy)


def heading_hold_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v4.16: penalize cumulative heading drift over the episode.

    The existing ``track_yaw_rate_l2`` is per-step: it penalizes
    instantaneous angular velocity vs the commanded yaw rate. But the deer
    can drift slowly (low yaw rate every step) while accumulating large
    heading errors over the 1000-step episode. Net result: the policy
    finds gaits that satisfy the per-step rate penalty while still
    veering 50-150° off course over 30 seconds.

    This term tracks each env's initial heading on reset, then penalizes
    `(cur_yaw - target_yaw)^2` where target_yaw = initial_yaw + integrated
    commanded yaw rate over time. So the penalty grows with cumulative
    drift, providing direct anti-veer pressure.

    State is kept on env attributes:
      env._heading_hold_initial: (N_envs,) initial yaw, set on reset/episode 0
      env._heading_hold_command_int: (N_envs,) running integral of cmd_yaw
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cur_quat = asset.data.root_quat_w                            # (N, 4) wxyz
    cur_yaw = _quat_to_yaw_w(cur_quat)                           # (N,)

    init_key = "_heading_hold_initial"
    int_key = "_heading_hold_cmd_integral"

    # Allocate state if missing or wrong device/size
    n = cur_yaw.shape[0]
    if not hasattr(env, init_key) or getattr(env, init_key).shape[0] != n:
        setattr(env, init_key, cur_yaw.detach().clone())
        setattr(env, int_key, torch.zeros(n, device=cur_yaw.device))
    initial = getattr(env, init_key)
    cmd_int = getattr(env, int_key)

    # On reset (episode_length_buf == 0), reseed initial heading and zero the integral.
    just_reset = env.episode_length_buf == 0                     # (N,) bool
    if just_reset.any():
        initial = torch.where(just_reset, cur_yaw, initial)
        cmd_int = torch.where(just_reset, torch.zeros_like(cmd_int), cmd_int)
        setattr(env, init_key, initial)
        setattr(env, int_key, cmd_int)

    # Integrate commanded yaw rate over time (Euler).
    cmd = env.command_manager.get_command(command_name)          # (N, 3) [vx, vy, wz]
    cmd_yaw_rate = cmd[:, 2] if cmd.shape[-1] >= 3 else torch.zeros_like(cur_yaw)
    cmd_int = cmd_int + cmd_yaw_rate * env.step_dt
    setattr(env, int_key, cmd_int)

    target_yaw = initial + cmd_int                               # (N,)

    # Wrap angular difference to (-pi, pi].
    import math as _math
    diff = cur_yaw - target_yaw
    diff_wrapped = (diff + _math.pi) % (2.0 * _math.pi) - _math.pi
    return diff_wrapped.square()                                  # (N,) — pair with negative weight


def front_carpus_fold_penalty(
    env: ManagerBasedRLEnv,
    carpus_joint_names: tuple[str, str] = ("FL_shin_to_FL_foot", "FR_shin_to_FR_foot"),
    range_min: float = 0.10,    # rad, ref min after sign-fix ~+11 deg
    range_max: float = 1.30,    # rad, ref max after sign-fix ~+69 deg, with slack
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v4.15: Hard quadratic penalty for front carpus going outside the SIGN-CORRECTED
    reference range [+0.10, +1.30] rad (~+6° to +75°).

    Across v4.7-v4.13 the front carpus was tracking the WRONG SIGN; v4.14
    fixed the sign convention but the policy spends 78-98% of time clamped
    at the URDF lower limit -0.3 rad (extension direction). Same structural
    issue as the HR knee in v4.12 — joint_track reward saturates and the
    stiff actuator (Kp=60) pulls toward extension. Hard penalty for being
    OUTSIDE the fold direction forces the policy to keep carpus in [+0.10, +1.30].

    See ``hind_knee_range_penalty`` for the same approach on the hind knee.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    cache_key = f"_front_carpus_fold_idx_{'_'.join(carpus_joint_names)}"
    if not hasattr(env, cache_key):
        all_names = robot.data.joint_names
        idxs = torch.tensor(
            [all_names.index(n) for n in carpus_joint_names],
            device=env.device, dtype=torch.long,
        )
        setattr(env, cache_key, idxs)
    knee_idxs = getattr(env, cache_key)
    q = robot.data.joint_pos[:, knee_idxs]                    # (N, 2)
    below = torch.clamp(range_min - q, min=0.0)               # 0 if q >= range_min
    above = torch.clamp(q - range_max, min=0.0)               # 0 if q <= range_max
    violation = below + above                                  # (N, 2) >= 0
    return violation.square().sum(dim=-1)                      # (N,) — pair with negative weight


def hind_knee_range_penalty(
    env: ManagerBasedRLEnv,
    knee_joint_names: tuple[str, str] = ("HL_thigh_to_HL_shin", "HR_thigh_to_HR_shin"),
    range_min: float = -0.55,   # rad, ref min ~-28 deg = -0.5 rad, with 3 deg slack
    range_max: float = 0.05,    # rad, ref max ~0 deg, with 3 deg slack
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty for hind knees going outside the reference's range.

    For each hind knee joint, computes:
        violation = max(range_min - q, 0) + max(q - range_max, 0)
        penalty = violation^2  (sum across the 2 hind knees)
    Returns penalty (>= 0). Use with NEGATIVE weight to discourage the
    out-of-range region.

    Reference range for HL/HR knee: -0.5 to 0 rad (from deer_walk.npy).
    With 3 deg slack on each side: [-0.55, 0.05] rad.

    At HR mean -0.96 rad (v4.12): below range_min by 0.41 rad, penalty
    contribution = 0.41^2 = 0.168 per step. With weight -5.0 = -0.84
    per step = -840 per episode — DOMINATES the joint_track signal.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    cache_key = f"_hind_knee_range_idx_{'_'.join(knee_joint_names)}"
    if not hasattr(env, cache_key):
        all_names = robot.data.joint_names
        idxs = torch.tensor(
            [all_names.index(n) for n in knee_joint_names],
            device=env.device, dtype=torch.long,
        )
        setattr(env, cache_key, idxs)
    knee_idxs = getattr(env, cache_key)
    q = robot.data.joint_pos[:, knee_idxs]                    # (N, 2)
    below = torch.clamp(range_min - q, min=0.0)               # 0 if q >= range_min
    above = torch.clamp(q - range_max, min=0.0)               # 0 if q <= range_max
    violation = below + above                                  # (N, 2) >= 0
    return violation.square().sum(dim=-1)                      # (N,) — pair with negative weight


# ---------------------------------------------------------------------------
# v4.8: per-joint reference-walk tracking
# ---------------------------------------------------------------------------
# Replaces the soft swing-phase rewards (knee_flex_during_swing,
# carpus_flex_during_swing, carpus_deep_flex_bonus) with explicit per-joint
# imitation of the reference walk in `assets/motions/deer_walk.npy`.
#
# Why: every previous "encourage leg motion" reward was a soft positive bonus
# on a derived feature (knee deviation, foot height, etc.). The policy kept
# finding low-effort local optima that satisfied those features without
# actually folding the leg through a swing — front cannons stayed rigid.
#
# Reference indexing strategy: the deer_walk.npy is one cycle (31 frames at
# 30 fps = 1.033 s). We index by the SAME global gait phase that the rest of
# the gait machinery uses (`phase = (t * GAIT_FREQ) mod 1.0` with
# GAIT_FREQ = 2 Hz). This compresses the reference's 1-Hz cycle to 2 Hz —
# joint-angle SHAPES stay identical, just played at 2x. That's fine: animals
# preserve gait shape across speed; what we care about is the leg
# articulation pattern, not its absolute timing.
#
# Joint mapping is by NAME (not index) so it's robust to URDF reorderings.
# Reference has 15 joints; we track the 12 LEG joints by default and skip
# the 3 spine/neck joints (which carry little leg-motion signal and risk
# fighting the head-pose rewards).
#
# Reward per step:
#   r = sum_j exp(-(q_j - q_ref_j(phase))^2 / sigma^2)
# Bounded in [0, num_tracked_joints]. With 12 joints + sigma=0.3 rad (~17 deg
# tolerance) and weight 0.3, max contribution is +3.6/step — comparable to
# forward_velocity_linear, large enough to dominate the "shuffle" local
# optimum but not so large it overrides command tracking.
# ---------------------------------------------------------------------------

# Reference joint name → policy joint name. The 12 leg joints come first
# (indices 0..11 in the tracked subset); the 3 spine/neck joints are listed
# at the end and only used when leg_only=False.
_REF_TO_POLICY_JOINT_MAP: tuple[tuple[str, str], ...] = (
    # Hind right (3 joints)
    ("pelvis_to_HR_thigh",   "HR_hipabd_to_HR_thigh"),    # ref[1]
    ("HR_thigh_to_HR_shin",  "HR_thigh_to_HR_shin"),      # ref[2]
    ("HR_shin_to_HR_foot",   "HR_shin_to_HR_foot"),       # ref[3]
    # Hind left (3 joints)
    ("pelvis_to_HL_thigh",   "HL_hipabd_to_HL_thigh"),    # ref[4]
    ("HL_thigh_to_HL_shin",  "HL_thigh_to_HL_shin"),      # ref[5]
    ("HL_shin_to_HL_foot",   "HL_shin_to_HL_foot"),       # ref[6]
    # Front right (3 joints)
    ("chest_to_FR_thigh",    "FR_hipabd_to_FR_thigh"),    # ref[7]
    ("FR_thigh_to_FR_shin",  "FR_thigh_to_FR_shin"),      # ref[8]
    ("FR_shin_to_FR_foot",   "FR_shin_to_FR_foot"),       # ref[9]
    # Front left (3 joints)
    ("chest_to_FL_thigh",    "FL_hipabd_to_FL_thigh"),    # ref[10]
    ("FL_thigh_to_FL_shin",  "FL_thigh_to_FL_shin"),      # ref[11]
    ("FL_shin_to_FL_foot",   "FL_shin_to_FL_foot"),       # ref[12]
    # --- spine + neck (excluded when leg_only=True) ---
    ("pelvis_to_chest",      "chest_yaw_to_chest"),       # ref[0]  midline pitch
    ("chest_to_neck",        "neck_yaw_to_neck"),         # ref[13] midline pitch
    ("neck_to_skull",        "neck_to_skull"),            # ref[14] midline pitch
)
_NUM_LEG_JOINTS_TRACKED = 12

# v4.9: per-foot reference templates.
# We extract ONE leg's joint trajectory from the reference and apply it to
# both legs in that pair (front: FL traj for both FL and FR; hind: HL traj
# for both HL and HR). This works because:
#   1. The URDF mirrored-axes convention means +q rad on either side of a
#      pair has the same physical interpretation (e.g. "knee bent forward").
#   2. Mirror equivariance from v4.7 means the policy treats L/R as
#      symmetric — using one template removes any L/R bias the animator
#      may have baked into the .npy.
#
# Each foot indexes its template by ITS own gait phase, so joint tracking
# is consistent with whatever gait pattern the env enforces (trot in our
# case via gait_contact_match). v4.8 used a single global phase for all
# joints, which forced the policy to satisfy walk-cycle joint angles at
# trot-cycle stance/swing timing — they conflict and the conflict surfaces
# as one hind leg spasming while the other tracks normally.

# Foot order MUST match GAIT_OFFSETS in deer_usd_loco_env_cfg.py
# (currently (0.0, 0.0, 0.5, 0.5) for trot in [FL, HR, FR, HL] order).
_FOOT_ORDER: tuple[str, ...] = ("FL", "HR", "FR", "HL")

# For each foot, the policy joint names for its 3 tracked joints
# (hipabd→thigh, thigh→shin, shin→foot). Order is hip-to-toe.
_POLICY_JOINTS_PER_FOOT: dict[str, tuple[str, ...]] = {
    "FL": ("FL_hipabd_to_FL_thigh", "FL_thigh_to_FL_shin", "FL_shin_to_FL_foot"),
    "FR": ("FR_hipabd_to_FR_thigh", "FR_thigh_to_FR_shin", "FR_shin_to_FR_foot"),
    "HL": ("HL_hipabd_to_HL_thigh", "HL_thigh_to_HL_shin", "HL_shin_to_HL_foot"),
    "HR": ("HR_hipabd_to_HR_thigh", "HR_thigh_to_HR_shin", "HR_shin_to_HR_foot"),
}

# Reference joint names for the FRONT template (used for FL and FR) and
# HIND template (used for HL and HR). We pull from one side of the
# reference to avoid cross-foot phase contamination.
_REF_FRONT_TEMPLATE: tuple[str, ...] = (
    "chest_to_FL_thigh",
    "FL_thigh_to_FL_shin",
    "FL_shin_to_FL_foot",
)
_REF_HIND_TEMPLATE: tuple[str, ...] = (
    "pelvis_to_HL_thigh",
    "HL_thigh_to_HL_shin",
    "HL_shin_to_HL_foot",
)

# v4.14: per-joint sign correction for ref→policy mapping.
# The URDF has OPPOSITE sign conventions for front carpus vs hind hock:
#   - Front carpus FL/FR_shin_to_FL/FR_foot: URDF limit [-0.3, +1.6] (positive = fold)
#   - Hind hock   HL/HR_shin_to_HL/HR_foot: URDF limit [-1.6, +0.3] (negative = fold)
# The reference uses a single convention (negative = fold for all legs). So
# for the front carpus the reference values are LITERALLY OUTSIDE the URDF
# joint limits — and v4.7-v4.13's joint_track reward was pulling the policy
# toward unreachable angles, which the policy gave up on (settling at limit
# clamp ~-0.3 rad and beyond into the OTHER direction). This produced the
# "leg held straight out forward" failure mode the user observed.
#
# Templates are 3 joints in hip→toe order:
#   FRONT [chest→thigh, thigh→shin (knee), shin→foot (carpus)]
#   HIND  [pelvis→thigh, thigh→shin (knee), shin→foot (hock)]
# Sign array is applied AFTER reference lookup, BEFORE error computation.
_FRONT_TEMPLATE_SIGN: tuple[float, ...] = (1.0, 1.0, -1.0)   # carpus negated
_HIND_TEMPLATE_SIGN: tuple[float, ...] = (1.0, 1.0, 1.0)    # hock direct match


def joint_track_reference_walk(
    env: ManagerBasedRLEnv,
    motion_path: str = "assets/motions/deer_walk.npy",
    frequency: float = 2.0,
    foot_phase_offsets: tuple[float, ...] = (0.0, 0.0, 0.5, 0.5),
    sigma: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v4.9: per-foot phase-indexed reference-walk tracking.

    For each foot k ∈ {FL, HR, FR, HL}, looks up the front- or hind-leg
    reference joint angles at that foot's gait phase, and rewards the
    policy's matching 3 joints for being close to those targets:

        phase_k = (t * frequency + foot_phase_offsets[k]) mod 1
        r_k = sum_{j=0..2} exp(-(q_kj - template_kj(phase_k))^2 / sigma^2)
        return sum_k r_k

    Bounded in [0, 12]. Pair with positive weight (~+0.5).

    v4.8 -> v4.9 change: v4.8 used ONE global phase to look up all 12
    joint targets simultaneously, which forced the policy to satisfy
    walk-cycle joint kinematics at the env's enforced trot-cycle stance/
    swing timing. The conflict made one hind leg spasm in mid-air while
    the other tracked normally. Per-foot indexing puts each foot's
    joint targets in sync with its own gait phase, so joint tracking and
    gait_contact_match are consistent regardless of the reference's
    underlying gait pattern.

    Args:
        motion_path: relative path from repo root to the .npy.
        frequency: gait clock Hz. MUST match GAIT_FREQ in env config.
        foot_phase_offsets: per-foot phase offsets in [0, 1). MUST match
            GAIT_OFFSETS in env config (defaults to trot).
        sigma: per-joint angular tolerance (rad). 0.3 ≈ 17 deg.
    """
    cache_key = "_v49_ref_track_state"
    state = getattr(env, cache_key, None)

    if state is None:
        # ---- Load reference once ----
        import os
        import numpy as np
        if not os.path.isabs(motion_path):
            here = os.path.dirname(os.path.abspath(__file__))
            repo = os.path.abspath(os.path.join(here, "..", ".."))
            motion_path = os.path.join(repo, motion_path)
        if not os.path.isfile(motion_path):
            raise FileNotFoundError(
                f"joint_track_reference_walk: motion file not found at {motion_path}"
            )
        data = np.load(motion_path, allow_pickle=True).item()
        ref_joints_list = list(data["joints_list"])
        ref_jp = data["joint_positions"]    # (T_ref, 15)
        T_ref = int(ref_jp.shape[0])

        robot: Articulation = env.scene[asset_cfg.name]
        policy_joint_names = list(robot.data.joint_names)

        # ---- Resolve template indices in the reference (3 front + 3 hind) ----
        missing: list[str] = []
        front_ref_idxs: list[int] = []
        hind_ref_idxs: list[int] = []
        for n in _REF_FRONT_TEMPLATE:
            if n not in ref_joints_list:
                missing.append(f"ref-front:{n}")
            else:
                front_ref_idxs.append(ref_joints_list.index(n))
        for n in _REF_HIND_TEMPLATE:
            if n not in ref_joints_list:
                missing.append(f"ref-hind:{n}")
            else:
                hind_ref_idxs.append(ref_joints_list.index(n))

        # ---- Resolve policy joint indices per foot (4 feet × 3 joints = 12) ----
        per_foot_policy_idxs: list[list[int]] = []
        for foot in _FOOT_ORDER:
            joint_names_for_foot = _POLICY_JOINTS_PER_FOOT[foot]
            this_foot_idxs: list[int] = []
            for n in joint_names_for_foot:
                if n not in policy_joint_names:
                    missing.append(f"policy:{n}")
                    continue
                this_foot_idxs.append(policy_joint_names.index(n))
            per_foot_policy_idxs.append(this_foot_idxs)

        if missing:
            raise RuntimeError(
                f"joint_track_reference_walk (v4.9): missing joint mappings: "
                f"{missing}. Ref joints: {ref_joints_list}. "
                f"First 5 policy joints: {policy_joint_names[:5]}"
            )

        # ---- Build the two templates, resampled to 100 phase bins ----
        N_PHASE = 100
        phases_dense = torch.arange(N_PHASE, dtype=torch.float32) / N_PHASE
        f_idx = phases_dense * T_ref                              # (N_PHASE,)
        f_lo = f_idx.floor().long() % T_ref
        f_hi = (f_lo + 1) % T_ref
        w_phase = (f_idx - f_idx.floor()).unsqueeze(-1)           # (N_PHASE, 1)

        front_jp = torch.from_numpy(ref_jp[:, front_ref_idxs].astype("float32"))
        hind_jp = torch.from_numpy(ref_jp[:, hind_ref_idxs].astype("float32"))
        front_template = front_jp[f_lo] * (1 - w_phase) + front_jp[f_hi] * w_phase
        hind_template = hind_jp[f_lo] * (1 - w_phase) + hind_jp[f_hi] * w_phase
        # Shapes: each (N_PHASE, 3)

        # v4.14: apply per-joint sign correction so reference values match the
        # URDF's joint sign convention (front carpus is sign-inverted; see
        # _FRONT_TEMPLATE_SIGN comment block above).
        front_sign = torch.tensor(_FRONT_TEMPLATE_SIGN, dtype=torch.float32)
        hind_sign = torch.tensor(_HIND_TEMPLATE_SIGN, dtype=torch.float32)
        front_template = front_template * front_sign.unsqueeze(0)
        hind_template = hind_template * hind_sign.unsqueeze(0)

        # ---- Stack templates into ONE (4, N_PHASE, 3) tensor in foot order ----
        # Foot order is [FL, HR, FR, HL]: F→front, H→hind.
        templates_stacked = torch.stack([
            front_template if f.startswith("F") else hind_template
            for f in _FOOT_ORDER
        ], dim=0)                                                  # (4, N_PHASE, 3)

        # ---- Flat policy joint idx tensor: 12 ints in foot order ----
        flat_policy_idxs: list[int] = []
        for idxs in per_foot_policy_idxs:
            flat_policy_idxs.extend(idxs)
        # Shape (12,) — joint_pos[:, this] gives (N, 12) which we reshape to (N, 4, 3).

        device = env.device
        templates_stacked = templates_stacked.to(device)            # (4, N_PHASE, 3)
        offsets_t = torch.tensor(
            list(foot_phase_offsets), device=device, dtype=torch.float32,
        )                                                           # (4,)
        policy_idxs_t = torch.tensor(
            flat_policy_idxs, device=device, dtype=torch.long,
        )                                                           # (12,)
        # Pre-build the foot-index pattern for advanced indexing:
        #   foot_idx[k] = k for all envs.
        foot_idx_pattern = torch.arange(
            len(_FOOT_ORDER), device=device, dtype=torch.long,
        )                                                           # (4,)

        state = {
            "templates": templates_stacked,            # (4, N_PHASE, 3)
            "offsets": offsets_t,                      # (4,)
            "policy_idxs_flat": policy_idxs_t,         # (12,)
            "foot_idx_pattern": foot_idx_pattern,      # (4,)
            "n_phase": N_PHASE,
            "n_phase_t": torch.tensor(
                float(N_PHASE), device=device, dtype=torch.float32,
            ),
            "frequency": float(frequency),
            "sigma_sq": float(sigma) * float(sigma),
        }
        setattr(env, cache_key, state)

    # ---- Per-step lookup, fully vectorized (no Python loop) ----
    robot: Articulation = env.scene[asset_cfg.name]
    t = env.episode_length_buf.float() * env.step_dt              # (N,)
    base_phase = (t * state["frequency"]) % 1.0                   # (N,)
    # (N, 4) foot phases via broadcast
    foot_phases = (base_phase.unsqueeze(-1) + state["offsets"].unsqueeze(0)) % 1.0

    n_phase = state["n_phase"]
    bin_f = foot_phases * n_phase                                 # (N, 4)
    bin_lo = bin_f.floor().long() % n_phase                       # (N, 4)
    bin_hi = (bin_lo + 1) % n_phase                               # (N, 4)
    w = (bin_f - bin_f.floor()).unsqueeze(-1)                     # (N, 4, 1)

    # Advanced indexing: templates is (4, N_PHASE, 3).
    # We want, for each (env, foot k), templates[k, bin_lo[env,k], :].
    # Broadcast foot_idx_pattern (4,) against bin_lo (N, 4) → both (N, 4).
    N = foot_phases.shape[0]
    foot_idx = state["foot_idx_pattern"].expand(N, -1)            # (N, 4)
    templates = state["templates"]                                 # (4, P, 3)
    q_ref_lo = templates[foot_idx, bin_lo]                         # (N, 4, 3)
    q_ref_hi = templates[foot_idx, bin_hi]                         # (N, 4, 3)
    q_ref = q_ref_lo * (1 - w) + q_ref_hi * w                      # (N, 4, 3)

    # Gather all 12 tracked joints in one shot, reshape to (N, 4, 3).
    q_now_flat = robot.data.joint_pos[:, state["policy_idxs_flat"]]  # (N, 12)
    q_now = q_now_flat.view(N, 4, 3)                                # (N, 4, 3)

    err_sq = (q_now - q_ref).square()                              # (N, 4, 3)
    # v4.11: Cauchy/Lorentzian shape replaces exp(-err²/σ²).
    # Cauchy decays much more slowly than exp — even at err=4σ it gives
    # 0.06 (vs exp gives ≈0). This restores gradient when joints are far
    # from the reference, fixing the vanishing-gradient that kept HR knee
    # stuck at -61° across v4.7/v4.8/v4.9/v4.10. Max value still 1.0 at
    # err=0; total reward bounded in [0, 12] same as before.
    return (1.0 / (1.0 + err_sq / state["sigma_sq"])).sum(dim=(-2, -1))


def forward_velocity_linear(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    clip_max: float = 1.5,
) -> torch.Tensor:
    """Linear reward for tracking the commanded forward velocity.

    Unlike ``exp(-err**2 / sigma**2)`` which has near-zero gradient when the
    error is large, this reward is linear in actual body velocity projected
    onto the commanded direction. It gives the policy a consistent gradient
    toward motion even when far from the target.

    For each env:
        v_cmd  = commanded body-frame XY velocity (length = commanded speed)
        v_body = actual body-frame XY velocity
        reward = clamp( dot(v_body, v_cmd / |v_cmd|), min=-clip_max, max=cmd_speed )

    When the command is zero we return zero (no preference over speed).

    Use positive weight. Scale with something like 1.0 so one-unit of matching
    forward velocity earns one reward unit per step.
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_vel = cmd[:, :2]                                          # (N, 2)
    body_vel = asset.data.root_lin_vel_b[:, :2]                   # (N, 2)

    cmd_speed = torch.norm(cmd_vel, dim=-1, keepdim=True)         # (N, 1)
    # Avoid div-by-zero when no command is active
    safe_speed = torch.where(cmd_speed > 1e-3, cmd_speed, torch.ones_like(cmd_speed))
    cmd_dir = cmd_vel / safe_speed                                # (N, 2)

    # Projection of actual velocity onto commanded direction
    projection = torch.sum(body_vel * cmd_dir, dim=-1)            # (N,)

    # Gate: only reward when a real command is active
    active = (cmd_speed.squeeze(-1) > 1e-3).float()

    # Clamp: at most commanded speed (full credit), at least -clip_max
    # (prevents unbounded negative reward if deer rockets backward).
    reward = torch.clamp(projection, min=-clip_max, max=float("inf"))
    reward = torch.minimum(reward, cmd_speed.squeeze(-1))

    return reward * active                                        # (N,)


# ---------------------------------------------------------------------------
# Reference-state initialization (used by AMP v4+)
# ---------------------------------------------------------------------------
# Module-level cache so we load the .pt file once per process.
_RSI_CACHE: dict[str, torch.Tensor] | None = None
_RSI_JOINT_INDICES_CACHE: torch.Tensor | None = None


# v4.12: bake-to-asset joint name mapping. The baked .pt was built with
# an OLDER URDF that used joint names like "pelvis_to_HR_thigh"; the v2
# URDF (current 25-DOF) uses "HR_hipabd_to_HR_thigh". The mapping handles
# both renames and the joint-count mismatch (bake 15 -> asset 25). Joints
# in the asset that have no bake counterpart (hipabd / chest_yaw / toes /
# neck_yaw — 10 joints) get the asset's default_joint_pos at reset.
_BAKE_TO_ASSET_JOINT_NAME: dict[str, str] = {
    "pelvis_to_chest":      "chest_yaw_to_chest",
    "pelvis_to_HR_thigh":   "HR_hipabd_to_HR_thigh",
    "HR_thigh_to_HR_shin":  "HR_thigh_to_HR_shin",
    "HR_shin_to_HR_foot":   "HR_shin_to_HR_foot",
    "pelvis_to_HL_thigh":   "HL_hipabd_to_HL_thigh",
    "HL_thigh_to_HL_shin":  "HL_thigh_to_HL_shin",
    "HL_shin_to_HL_foot":   "HL_shin_to_HL_foot",
    "chest_to_FR_thigh":    "FR_hipabd_to_FR_thigh",
    "FR_thigh_to_FR_shin":  "FR_thigh_to_FR_shin",
    "FR_shin_to_FR_foot":   "FR_shin_to_FR_foot",
    "chest_to_FL_thigh":    "FL_hipabd_to_FL_thigh",
    "FL_thigh_to_FL_shin":  "FL_thigh_to_FL_shin",
    "FL_shin_to_FL_foot":   "FL_shin_to_FL_foot",
    "chest_to_neck":        "neck_yaw_to_neck",
    "neck_to_skull":        "neck_to_skull",
}

# v4.14: per-joint sign correction for bake→asset mapping. Front carpus
# joints (FL/FR_shin_to_FL/FR_foot) have URDF axis SIGN-INVERTED relative
# to the bake's animation convention. Without this fix, RSI tries to write
# bake values like q=-1.0 rad which the URDF clamps at q=-0.3 rad — the
# deer effectively starts every episode in the wrong front-carpus pose.
# Same fix as in joint_track_reference_walk.
_BAKE_JOINT_SIGN: dict[str, float] = {
    "FL_shin_to_FL_foot": -1.0,
    "FR_shin_to_FR_foot": -1.0,
}


def _load_reset_states(
    device: torch.device,
    asset_joint_names: list[str],
    asset_default_joint_pos: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Lazy-load baked reset-states file and align joint ordering to the sim.

    The baked .pt has 15 joint columns named per the OLD URDF
    ("pelvis_to_HR_thigh", etc.). The current asset has 25 joints with
    different names ("HR_hipabd_to_HR_thigh"). We map by name via
    ``_BAKE_TO_ASSET_JOINT_NAME``; joints in the asset with no bake
    counterpart (hipabd / chest_yaw / toes / neck_yaw — 10 of them) are
    filled with the asset's default joint position so the reset is still
    well-defined.

    Args:
        asset_default_joint_pos: (n_dof,) tensor of asset default joint
            positions. Required when bake/asset joint sets differ. If None,
            unmapped joints fill with 0 (the legacy behaviour, but unsafe
            if the deer's stable rest pose isn't all zeros).
    """
    global _RSI_CACHE, _RSI_JOINT_INDICES_CACHE
    if _RSI_CACHE is not None:
        return _RSI_CACHE

    import os as _os
    from pathlib import Path as _P

    # Resolve path: env var override or canonical location in repo.
    override = _os.environ.get("ANIML_AMP_RESET_STATES", "").strip()
    if override and _P(override).is_file():
        path = _P(override)
    else:
        here = _P(__file__).resolve()
        repo_root = None
        for parent in here.parents:
            if (parent / "assets" / "motions").is_dir():
                repo_root = parent
                break
        if repo_root is None:
            raise RuntimeError(
                "Can't locate repo root from deer_rewards.py; set "
                "ANIML_AMP_RESET_STATES to the .pt file."
            )
        path = repo_root / "assets" / "motions" / "deer_walk_reset_states.pt"
        if not path.is_file():
            raise RuntimeError(
                f"Baked reset states not found at {path}. Run "
                "`python tools/bake_reset_states.py` first."
            )

    raw = torch.load(str(path), map_location=device, weights_only=False)
    npy_order: list[str] = list(raw["joints_list"])
    bake_jp = raw["joint_positions"]    # (T, n_bake)
    bake_jv = raw["joint_velocities"]   # (T, n_bake)
    T = bake_jp.shape[0]
    n_asset = len(asset_joint_names)

    # Build asset-aligned joint_positions / joint_velocities.
    # For each asset joint, decide its source: bake column (mapped) or
    # default (unmapped).
    bake_lookup = {name: i for i, name in enumerate(npy_order)}
    # Build a (n_asset,) src_idx tensor: src_idx[i] = bake column for
    # asset joint i, or -1 if no bake source.
    src_idx_list: list[int] = []
    unmapped: list[str] = []
    for asset_name in asset_joint_names:
        # Look for a bake name that maps to this asset name.
        matched_bake_idx = -1
        for bake_name, mapped_asset_name in _BAKE_TO_ASSET_JOINT_NAME.items():
            if mapped_asset_name == asset_name and bake_name in bake_lookup:
                matched_bake_idx = bake_lookup[bake_name]
                break
        # Fallback: direct name match (in case bake and asset names align).
        if matched_bake_idx < 0 and asset_name in bake_lookup:
            matched_bake_idx = bake_lookup[asset_name]
        src_idx_list.append(matched_bake_idx)
        if matched_bake_idx < 0:
            unmapped.append(asset_name)

    src_idx = torch.tensor(src_idx_list, dtype=torch.long, device=device)
    has_bake = src_idx >= 0                                       # (n_asset,)

    # Build full (T, n_asset) tensors. Use bake column for mapped joints,
    # default joint pos for unmapped.
    aligned_jp = torch.zeros(T, n_asset, device=device, dtype=bake_jp.dtype)
    aligned_jv = torch.zeros(T, n_asset, device=device, dtype=bake_jv.dtype)

    if has_bake.any():
        mapped_idxs = torch.nonzero(has_bake, as_tuple=False).squeeze(-1)
        bake_cols = src_idx[mapped_idxs]
        aligned_jp[:, mapped_idxs] = bake_jp[:, bake_cols].to(device)
        aligned_jv[:, mapped_idxs] = bake_jv[:, bake_cols].to(device)

        # v4.14: apply per-joint sign correction. Front carpus joints have
        # OPPOSITE sign convention between bake (animation) and URDF.
        for asset_idx_int in mapped_idxs.cpu().tolist():
            asset_name = asset_joint_names[asset_idx_int]
            sign = _BAKE_JOINT_SIGN.get(asset_name, 1.0)
            if sign != 1.0:
                aligned_jp[:, asset_idx_int] *= sign
                aligned_jv[:, asset_idx_int] *= sign

    if (~has_bake).any():
        if asset_default_joint_pos is None:
            # Legacy fallback: zero-fill (only safe if deer's rest pose has
            # those joints at 0 — usually true for chest_yaw / neck_yaw /
            # toe / hipabd which spec defaults near 0).
            pass
        else:
            unmapped_idxs = torch.nonzero(~has_bake, as_tuple=False).squeeze(-1)
            defaults = asset_default_joint_pos[unmapped_idxs].to(device)
            aligned_jp[:, unmapped_idxs] = defaults.unsqueeze(0).expand(T, -1)
            # joint velocities for unmapped joints stay 0 (no motion at init)

    raw["joint_positions"]  = aligned_jp
    raw["joint_velocities"] = aligned_jv

    if unmapped:
        # Log once on first call so we know what got defaulted.
        print(
            f"[reset_from_amp_motion] {len(unmapped)} asset joints not in bake; "
            f"using defaults: {unmapped}"
        )

    _RSI_CACHE = raw
    return raw


def reset_from_amp_motion(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_height_m: float = 0.93,
    pose_jitter: float = 0.0,
    vel_jitter: float = 0.0,
) -> None:
    """Reset asset state by sampling a random frame from the baked walk cycle.

    For each ``env_id``:
      - Sample a random frame index from the baked trajectory.
      - Set joint positions / velocities to that frame's values (after
        reordering to match sim joint order).
      - Set root pose: XY from env origin (preserves the per-env spawn
        location), Z = ``base_height_m``, orientation = the URDF spawn
        quat Rx(-90) (same as ``_DEFAULT_ROT``) -- we don't rotate the
        base from the expert clip because the clip is itself in-place.
      - Set root linear velocity (world frame) to the expert frame's
        forward velocity. Root angular velocity matches expert.

    ``pose_jitter`` and ``vel_jitter`` add small uniform noise on top, for
    exploration around the expert init.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    # Pass default_joint_pos so unmapped joints (hipabd / yaw / toes — not
    # in the 15-DOF bake) get filled with the asset's stable rest values.
    asset_default_jp = asset.data.default_joint_pos[0]  # (n_dof,) — same for all envs
    data = _load_reset_states(
        device,
        list(asset.data.joint_names),
        asset_default_joint_pos=asset_default_jp,
    )
    T = data["joint_positions"].shape[0]

    n = env_ids.shape[0]
    # Random frame indices
    frame_ids = torch.randint(0, T, (n,), device=device)

    # Joint state
    jp = data["joint_positions"][frame_ids].to(device)  # (n, 15)
    jv = data["joint_velocities"][frame_ids].to(device)

    if pose_jitter > 0:
        jp = jp + (torch.rand_like(jp) * 2 - 1) * pose_jitter
    if vel_jitter > 0:
        jv = jv + (torch.rand_like(jv) * 2 - 1) * vel_jitter

    # Clamp to joint limits
    lo = asset.data.soft_joint_pos_limits[:, :, 0][env_ids]
    hi = asset.data.soft_joint_pos_limits[:, :, 1][env_ids]
    jp = torch.clamp(jp, lo, hi)

    asset.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

    # Root pose: at env origin (xy) + base_height_m above local terrain.
    # env_origins[:, 2] carries terrain tile elevation; on flat plane it's 0.
    origins = env.scene.env_origins[env_ids]                      # (n, 3)
    positions = origins.clone()
    positions[:, 2] = origins[:, 2] + base_height_m
    # Use the asset's default root orientation (post Rx(-90)). This is
    # already set as init_state.rot in deer_usd_articulation.py.
    default_rot = asset.data.default_root_state[env_ids, 3:7]     # (n, 4), wxyz
    asset.write_root_pose_to_sim(
        torch.cat([positions, default_rot], dim=-1), env_ids=env_ids,
    )

    # Root velocity: expert forward speed. Work in world frame since
    # root_quat Rx(-90) rotation maps body X to world X (deer facing +X).
    lin_vel_w = data["root_lin_vel_w"][frame_ids].to(device)     # (n, 3)
    ang_vel_w = data["root_ang_vel_w"][frame_ids].to(device)     # (n, 3)
    if vel_jitter > 0:
        lin_vel_w = lin_vel_w + (torch.rand_like(lin_vel_w) * 2 - 1) * vel_jitter
        ang_vel_w = ang_vel_w + (torch.rand_like(ang_vel_w) * 2 - 1) * vel_jitter
    root_vel = torch.cat([lin_vel_w, ang_vel_w], dim=-1)          # (n, 6)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
