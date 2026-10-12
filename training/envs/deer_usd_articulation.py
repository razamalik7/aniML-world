"""Build :class:`ArticulationCfg` for an arbitrary quadruped USD (deer, etc.)."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg


def deer_dc_actuator_cfg() -> DCMotorCfg:
    """Per-joint DC motor PD with anatomical front/hind split (URDF v3).

    Per cervid biomechanics (escape-gait literature): forelimbs act as elastic
    *struts* (passive energy absorbers / decelerators), hindlimbs as active
    *drivers* (extensors that produce propulsive force). The gain split below
    encodes that:

      Front legs (FR_*, FL_*):  lower stiffness, higher damping  -> compliant
      Hind  legs (HR_*, HL_*):  higher stiffness, normal damping -> active
      Spine + neck + skull:     very soft (passive sway)

    Env vars (override defaults):
      Legs (per-side gains)
        ANIML_DEER_FRONT_STIFFNESS / _FRONT_DAMPING
        ANIML_DEER_HIND_STIFFNESS  / _HIND_DAMPING
      Effort/velocity limits (separate front/hind, hind gets more headroom
      because hindlimbs do the propulsion)
        ANIML_DEER_FRONT_EFFORT / _FRONT_VELOCITY
        ANIML_DEER_HIND_EFFORT  / _HIND_VELOCITY
      Soft (spine/neck/skull)
        ANIML_DEER_SOFT_STIFFNESS / _SOFT_DAMPING
      Legacy (backward compat — used by tests when front/hind not specified)
        ANIML_DEER_STIFFNESS / _DAMPING (treated as a baseline if specific
        front/hind aren't set)
    """
    sat = float(os.environ.get("ANIML_DEER_SATURATION_EFFORT", "350"))
    # Per-leg-class effort + velocity (hind gets more torque/speed for propulsion).
    front_eff = float(os.environ.get("ANIML_DEER_FRONT_EFFORT",   "240"))
    hind_eff  = float(os.environ.get("ANIML_DEER_HIND_EFFORT",    "360"))
    front_vel = float(os.environ.get("ANIML_DEER_FRONT_VELOCITY", "80"))
    hind_vel  = float(os.environ.get("ANIML_DEER_HIND_VELOCITY",  "120"))
    # The DCMotorCfg has scalar effort_limit/velocity_limit, not per-joint. Take
    # the larger to size the motor model; per-joint torque is still bounded by
    # PhysX joint drive limits and saturation_effort below.
    eff = max(front_eff, hind_eff)
    vel = max(front_vel, hind_vel)
    # Per-leg-class stiffness/damping (THE biomech split).
    legacy_stiff = float(os.environ.get("ANIML_DEER_STIFFNESS", "60"))
    legacy_damp  = float(os.environ.get("ANIML_DEER_DAMPING",   "4"))
    front_stiff = float(os.environ.get("ANIML_DEER_FRONT_STIFFNESS", "40"))
    front_damp  = float(os.environ.get("ANIML_DEER_FRONT_DAMPING",   "4"))   # v4.1: 6 -> 4 to let front knees bend more visibly (less stilt-like)
    hind_stiff  = float(os.environ.get("ANIML_DEER_HIND_STIFFNESS",  str(legacy_stiff + 10)))
    hind_damp   = float(os.environ.get("ANIML_DEER_HIND_DAMPING",    str(legacy_damp)))
    soft_stiff  = float(os.environ.get("ANIML_DEER_SOFT_STIFFNESS",  "5"))
    soft_damp   = float(os.environ.get("ANIML_DEER_SOFT_DAMPING",    "0.5"))

    return DCMotorCfg(
        joint_names_expr=[".*"],
        saturation_effort=sat,
        effort_limit=eff,
        velocity_limit=vel,
        # Mutually-exclusive joint patterns — DCMotorCfg raises ValueError on
        # multi-match. Order: HAA frames first (explicit), then leg flex/ankle
        # (regex grouped by L/R prefix), then spine/neck/skull.
        stiffness={
            # Hind hip abduction (load-bearing, active)
            "pelvis_to_HR_hipabd":   hind_stiff,
            "pelvis_to_HL_hipabd":   hind_stiff,
            # Front hip abduction (load-bearing, more compliant)
            "chest_to_FR_hipabd":    front_stiff,
            "chest_to_FL_hipabd":    front_stiff,
            # Hind leg flex/ankle (active drivers)
            r"H(R|L)_(thigh|shin|foot|toe)_to_H(R|L)_(thigh|shin|foot|toe)":  hind_stiff,
            r"H(R|L)_hipabd_to_H(R|L)_thigh":                          hind_stiff,
            # Front leg flex/ankle (compliant struts)
            r"F(R|L)_(thigh|shin|foot|toe)_to_F(R|L)_(thigh|shin|foot|toe)":  front_stiff,
            r"F(R|L)_hipabd_to_F(R|L)_thigh":                          front_stiff,
            # Spine yaw + pitch (near-passive)
            "pelvis_to_chest_yaw":   soft_stiff,
            "chest_yaw_to_chest":    soft_stiff,
            # Neck yaw + pitch + skull (near-passive)
            "chest_to_neck_yaw":     soft_stiff,
            "neck_yaw_to_neck":      soft_stiff,
            "neck_to_skull":         soft_stiff,
        },
        damping={
            "pelvis_to_HR_hipabd":   hind_damp,
            "pelvis_to_HL_hipabd":   hind_damp,
            "chest_to_FR_hipabd":    front_damp,
            "chest_to_FL_hipabd":    front_damp,
            r"H(R|L)_(thigh|shin|foot|toe)_to_H(R|L)_(thigh|shin|foot|toe)":  hind_damp,
            r"H(R|L)_hipabd_to_H(R|L)_thigh":                          hind_damp,
            r"F(R|L)_(thigh|shin|foot|toe)_to_F(R|L)_(thigh|shin|foot|toe)":  front_damp,
            r"F(R|L)_hipabd_to_F(R|L)_thigh":                          front_damp,
            "pelvis_to_chest_yaw":   soft_damp,
            "chest_yaw_to_chest":    soft_damp,
            "chest_to_neck_yaw":     soft_damp,
            "neck_yaw_to_neck":      soft_damp,
            "neck_to_skull":         soft_damp,
        },
    )


def _env_flag(name: str, *, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


# deer_world.usd: identity USD root, URDF Y-up lying flat.
# init_state.rot = Rx(-90) rotates upright at EVERY episode reset.
# Rx(-90) quaternion (w,x,y,z): 0.7071068, -0.7071068, 0, 0
# See .cursor/rules/deer-orientation.mdc for why this must not change.
_DEFAULT_ROT = "0.7071068,-0.7071068,0,0"

# URDF v2 default joint pose (21 DOF) — "ready to walk", not "balanced standing".
#
# Walk-These-Ways uses default joint angles that put the robot close to a stride
# pose, not centered-stand. This makes small action offsets produce stride-like
# motion. Our previous defaults (front legs tucked back, hind legs swung forward,
# knees straight) biased the policy toward stillness — small offsets stayed near
# stationary stance, never spontaneously cycling through a stride.
#
# New defaults:
#   - All 4 hips (X-axis flex) at 0.0 (legs straight down, vertically)
#   - Front knees at +0.5 rad (forward-bent — natural for FR/FL whose limit is [0, 2.2])
#   - Hind knees at -0.7 rad (backward-bent — natural for HR/HL whose limit is [-2.2, 0])
#   - All 4 ankles at 0 (foot tip pointing down naturally from shin)
#   - All HAA / spine yaw / neck yaw at 0
#
# This puts the deer in a "loaded spring" stance where every leg is bent —
# extending any joint by a small action drives the foot down or forward.
_DEFAULT_JOINT_POS = {
    # Spine
    "pelvis_to_chest_yaw":     0.0,
    "chest_yaw_to_chest":      0.0,
    # Hind right
    "pelvis_to_HR_hipabd":     0.0,
    "HR_hipabd_to_HR_thigh":  +0.30,   # v8 stance fix: hind hip back -> feet under hips (was 0.0)
    "HR_thigh_to_HR_shin":    -0.50,   # v8 stance fix: less bend, un-scrunch (was -0.7)
    "HR_shin_to_HR_foot":      0.0,
    "HR_foot_to_HR_toe":       0.0,    # pastern (fetlock) — neutral stance
    # Hind left
    "pelvis_to_HL_hipabd":     0.0,
    "HL_hipabd_to_HL_thigh":  +0.30,   # v8 stance fix (was 0.0)
    "HL_thigh_to_HL_shin":    -0.50,   # v8 stance fix (was -0.7)
    "HL_shin_to_HL_foot":      0.0,
    "HL_foot_to_HL_toe":       0.0,
    # Neck
    "chest_to_neck_yaw":       0.0,
    "neck_yaw_to_neck":        0.0,
    "neck_to_skull":           0.0,
    # Front right
    "chest_to_FR_hipabd":      0.0,
    "FR_hipabd_to_FR_thigh":  -0.45,   # v8 stance fix: front feet UNDER shoulders, FK+eye verified (was 0.0=feet 20cm behind)
    "FR_thigh_to_FR_shin":    +0.5,    # knee bent forward (was 0)
    "FR_shin_to_FR_foot":      0.0,
    "FR_foot_to_FR_toe":       0.0,    # pastern (fetlock) — neutral stance
    # Front left
    "chest_to_FL_hipabd":      0.0,
    "FL_hipabd_to_FL_thigh":  -0.45,   # v8 stance fix (was 0.0)
    "FL_thigh_to_FL_shin":    +0.5,
    "FL_shin_to_FL_foot":      0.0,
    "FL_foot_to_FL_toe":       0.0,
}


def _parse_rot() -> tuple[float, float, float, float]:
    """Read ANIML_DEER_SPAWN_ROT as 'w,x,y,z' quaternion."""
    raw = os.environ.get("ANIML_DEER_SPAWN_ROT", _DEFAULT_ROT).strip()
    parts = [float(x) for x in raw.split(",")]
    if len(parts) == 4:
        return tuple(parts)  # type: ignore[return-value]
    return (1.0, 0.0, 0.0, 0.0)


def deer_articulation_cfg(usd_path: str) -> ArticulationCfg:
    """Spawn the USD as ``{ENV_REGEX_NS}/Robot`` (prim_path set by caller).

    ``activate_contact_sensors`` on the spawn config walks the subtree for
    ``UsdPhysics.RigidBodyAPI``. DCC exports without PhysX setup have **no**
    rigid bodies, which raises *No contact sensors added*. Default is **off**;
    set ``ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1`` after the asset has rigid
    bodies (Isaac Sim → Physics / collision / articulation root).
    """
    # deer_world.usd: after Rx(-90), lowest foot (FR) is ~0.638m below pelvis.
    # Spawn Z=0.93 puts feet at ground with ~1cm margin. (April 10 working value.)
    z = float(os.environ.get("ANIML_DEER_SPAWN_Z", "0.93"))
    rot = _parse_rot()
    contact_sensors = _env_flag("ANIML_DEER_ACTIVATE_CONTACT_SENSORS", default=False)
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.05,
                angular_damping=0.05,
                # Tightened from 1000.0 / 1000.0. Stock Isaac Lab quadruped
                # configs use these huge caps for robustness, but on the
                # deer URDF with self_collisions=True they let PhysX produce
                # NaN-sized velocities during depenetration of overlapping
                # mesh fragments, which propagates to "Failed to get DOF
                # velocities from backend" exceptions and corrupts the
                # cooked GPU articulation state on disk between runs.
                # 50 m/s and 20 rad/s are generous above realistic deer motion
                # (sprint ~15 m/s, head shake ~10 rad/s). Tightened from 30 ->
                # 20 rad/s for r18 after r17 crashed at iter 735 with widened
                # joint limits — wider hip/fetlock range gave the policy more
                # opportunities to drive joint vel into NaN territory.
                # v4.6: tightened from 50/20 -> 30/15 to further suppress
                # PhysX NaN-drift seen in v4.5 silent death (iter time crept
                # from 4.3s to 5.7s before kit.exe died). Tighter caps trade
                # a little physical realism for stability margin.
                max_linear_velocity=30.0,
                max_angular_velocity=15.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                # Bumped from 8/0 to 16/2 for r18. Widened joint limits push
                # PhysX harder; more solver iterations = better numerical
                # stability at limit edges, at ~10-15% throughput cost.
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, z),
            rot=rot,
            joint_pos=_DEFAULT_JOINT_POS,
        ),
        actuators={"legs": deer_dc_actuator_cfg()},
        soft_joint_pos_limit_factor=0.95,
    )
