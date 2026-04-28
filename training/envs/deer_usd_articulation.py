"""Build :class:`ArticulationCfg` for an arbitrary quadruped USD (deer, etc.)."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg


def deer_dc_actuator_cfg() -> DCMotorCfg:
    """Per-joint DC motor PD. Legs get firm gains so they can push off the
    ground; spine/neck/skull get very soft gains so they passively sway with
    gravity/momentum instead of being actively held rigid by the servo.

    Env vars:
      ANIML_DEER_STIFFNESS / _DAMPING            -> legs (load-bearing)
      ANIML_DEER_SOFT_STIFFNESS / _SOFT_DAMPING  -> spine + neck + skull
    """
    sat = float(os.environ.get("ANIML_DEER_SATURATION_EFFORT", "350"))
    eff = float(os.environ.get("ANIML_DEER_EFFORT_LIMIT", "300"))
    vel = float(os.environ.get("ANIML_DEER_VELOCITY_LIMIT", "100"))
    leg_stiff  = float(os.environ.get("ANIML_DEER_STIFFNESS",       "60"))
    leg_damp   = float(os.environ.get("ANIML_DEER_DAMPING",         "4"))
    soft_stiff = float(os.environ.get("ANIML_DEER_SOFT_STIFFNESS",  "5"))
    soft_damp  = float(os.environ.get("ANIML_DEER_SOFT_DAMPING",    "0.5"))
    return DCMotorCfg(
        joint_names_expr=[".*"],
        saturation_effort=sat,
        effort_limit=eff,
        velocity_limit=vel,
        stiffness={
            # Legs: anything with thigh/shin/foot in the name (12 joints)
            ".*(thigh|shin|foot).*": leg_stiff,
            # Spine/neck/skull: near-passive
            "pelvis_to_chest":       soft_stiff,
            "chest_to_neck":         soft_stiff,
            "neck_to_skull":         soft_stiff,
        },
        damping={
            ".*(thigh|shin|foot).*": leg_damp,
            "pelvis_to_chest":       soft_damp,
            "chest_to_neck":         soft_damp,
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

# CoM fix: hind feet are +0.245m behind pelvis, front feet -0.593m forward.
# Foot centroid is 0.245m behind CoM -> deer tips backward.
# Bend hind legs forward (+) and tuck front legs back (-) to center the support polygon.
_DEFAULT_JOINT_POS = {
    "pelvis_to_chest":      0.0,
    # Hind legs: swing forward so hind feet move from Y=+0.10 toward Y=-0.15
    "pelvis_to_HR_thigh":   0.4,
    "HR_thigh_to_HR_shin":  0.0,
    "HR_shin_to_HR_foot":   0.0,
    "pelvis_to_HL_thigh":   0.4,
    "HL_thigh_to_HL_shin":  0.0,
    "HL_shin_to_HL_foot":   0.0,
    # Front legs: tuck back so front feet sit under the chest
    "chest_to_FR_thigh":   -0.4,
    "FR_thigh_to_FR_shin":  0.0,
    "FR_shin_to_FR_foot":   0.0,
    "chest_to_FL_thigh":   -0.4,
    "FL_thigh_to_FL_shin":  0.0,
    "FL_shin_to_FL_foot":   0.0,
    "chest_to_neck":        0.0,
    "neck_to_skull":        0.0,
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
                max_linear_velocity=50.0,
                max_angular_velocity=20.0,
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
