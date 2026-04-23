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
        "pelvis_to_HR_thigh", "HR_thigh_to_HR_shin",
        "pelvis_to_HL_thigh", "HL_thigh_to_HL_shin",
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
    """Return True (terminate) when pelvis drops below min_height."""
    base_z = _body_z(env, asset_cfg, list(base_names))
    return base_z < min_height


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
    # Find the 12 leg joints by name.
    leg_joint_names = [
        "pelvis_to_HR_thigh", "HR_thigh_to_HR_shin", "HR_shin_to_HR_foot",
        "pelvis_to_HL_thigh", "HL_thigh_to_HL_shin", "HL_shin_to_HL_foot",
        "chest_to_FR_thigh",  "FR_thigh_to_FR_shin", "FR_shin_to_FR_foot",
        "chest_to_FL_thigh",  "FL_thigh_to_FL_shin", "FL_shin_to_FL_foot",
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


def _load_reset_states(device: torch.device, asset_joint_names: list[str]) -> dict[str, torch.Tensor]:
    """Lazy-load baked reset-states file to the given device and align joint
    ordering to the sim articulation.
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
        # Find the repo root by climbing until we see "assets/motions"
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

    # Reorder joint_positions / joint_velocities to match asset's joint order.
    npy_order = raw["joints_list"]
    if list(npy_order) != list(asset_joint_names):
        # Build an index map from npy-order to asset-order.
        lookup = {name: i for i, name in enumerate(npy_order)}
        idx = torch.tensor(
            [lookup[name] for name in asset_joint_names],
            dtype=torch.long, device=device,
        )
        raw["joint_positions"]  = raw["joint_positions"][:, idx]
        raw["joint_velocities"] = raw["joint_velocities"][:, idx]

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

    data = _load_reset_states(device, list(asset.data.joint_names))
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

    # Root pose: at env origin + base_height_m, spawn-quat orientation
    origins = env.scene.env_origins[env_ids]                      # (n, 3)
    positions = origins.clone()
    positions[:, 2] = base_height_m
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
