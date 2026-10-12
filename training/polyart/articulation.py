"""ArticulationCfg for the Poly Art / Malbers deer (Tier-3 rebuild, 38 DOF).

Built FRESH against the validated USD ``assets/deer_polyart_urdf/deer_polyart.usd``
— *not* forked from ``deer_usd_articulation.py`` (the old 21-DOF rig). Key
differences from the old rig, all consequences of the Tier-3 rebuild:

  * The USD imports **Z-up, +X-forward, standing at joint-zero**. There is no
    ``Rx(-90)`` reset rotation and no "loaded-spring" default pose — spawn
    rotation is identity and the default joint pose is all zeros (validated as a
    correct symmetric standing stance in ``validate_usd.py``).
  * 38 explicit joints (2-DOF joints are two stacked revolutes with a massless
    ``*_dof0`` intermediate; see ``docs/USD_BUILD_PIPELINE.md``).
  * Actuator groups are defined by **explicit joint-name lists** that partition
    all 38 joints exactly once — no regex. The old rig's ``DCMotorCfg`` blew up
    on regex multi-matches; explicit lists make the partition auditable.

Carried forward from the old rig because they are hard-won PhysX-stability
lessons (see CLAUDE.md "PhysX silent-death pattern"):
  * ``max_linear_velocity=30`` / ``max_angular_velocity=15`` /
    ``max_depenetration_velocity=1.0`` — NOT the stock 1000/1000 that let PhysX
    yank self-colliding bodies apart at NaN speeds.
  * ``solver_position_iteration_count=16`` / ``velocity=2`` for stability at
    joint limits under self-collision.
  * ``enabled_self_collisions=True`` (the USD already carries the 34 rest-overlap
    filter pairs, so this is safe — legs still can't pass through each other).

Actuator model: ``ImplicitActuatorCfg`` (pure PD + effort clamp). Chosen over
``DCMotorCfg`` for transparency — no torque-speed curve to reason about, and it
is exactly what ``validate_usd.py`` held the rig with. A DCMotor torque-speed
model is a possible phase-2 realism upgrade.

Gains are DELIBERATE starting points (anatomical hind-driver / front-strut /
soft-torso split), all env-overridable for tuning during the first AMP runs.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ---------------------------------------------------------------------------
# Joint-group partition — explicit names, verified to cover all 38 joints once.
# Names are the URDF/USD spelling (underscores, "_to_", "_dof0" suffix), which
# is what the actuator config and SceneEntityCfgs must target. The reference
# .npy uses a cosmetically different spelling in the SAME order — map by
# position with preserve_order=True, never by string (see USD_BUILD_PIPELINE.md).
# ---------------------------------------------------------------------------

# Hind legs — active drivers (extensors produce propulsion): 12 joints.
HIND_LEG_JOINTS = [
    "Pelvis_to_L_Thigh_dof0", "L_Thigh_dof0_to_L_Thigh",   # hip (2-DOF)
    "L_Thigh_to_L_Calf",                                   # stifle (knee)
    "L_Calf_to_L_HorseLink",                               # hock
    "L_HorseLink_to_L_Foot",                               # fetlock (ankle)
    "L_Foot_to_L_Toe0",                                    # hoof (toe)
    "Pelvis_to_R_Thigh_dof0", "R_Thigh_dof0_to_R_Thigh",
    "R_Thigh_to_R_Calf",
    "R_Calf_to_R_HorseLink",
    "R_HorseLink_to_R_Foot",
    "R_Foot_to_R_Toe0",
]

# Front legs — compliant struts (elastic energy absorbers): 12 joints.
FRONT_LEG_JOINTS = [
    "Spine2_to_L_Clavicle",                                        # shoulder
    "L_Clavicle_to_L_UpperArm_dof0", "L_UpperArm_dof0_to_L_UpperArm",  # (2-DOF)
    "L_UpperArm_to_L_Forearm",                                     # elbow
    "L_Forearm_to_L_Hand",                                         # carpus (wrist)
    "L_Hand_to_L_Finger0",                                         # hoof (toe)
    "Spine2_to_R_Clavicle",
    "R_Clavicle_to_R_UpperArm_dof0", "R_UpperArm_dof0_to_R_UpperArm",
    "R_UpperArm_to_R_Forearm",
    "R_Forearm_to_R_Hand",
    "R_Hand_to_R_Finger0",
]

# Spine — soft passive sway (Pelvis->Spine is 2-DOF): 4 joints.
SPINE_JOINTS = [
    "Pelvis_to_Spine_dof0", "Spine_dof0_to_Spine",
    "Spine_to_Spine1",
    "Spine1_to_Spine2",
]

# Neck + head — soft (all three segments are 2-DOF): 6 joints.
NECK_HEAD_JOINTS = [
    "Spine2_to_Neck_dof0", "Neck_dof0_to_Neck",
    "Neck_to_Neck1_dof0", "Neck1_dof0_to_Neck1",
    "Neck1_to_Head_dof0", "Head_dof0_to_Head",
]

# Cosmetic — actuated but PHASE-1-LOCKED (held at default; unlock later for
# learned eating / alert / tail). 4 joints. See dof_spec.phase1_locked_at_reference.
COSMETIC_JOINTS = [
    "Head_to_Jaw", "Head_to_L_Ear", "Head_to_R_Ear", "Pelvis_to_Tail",
]

ALL_JOINTS = HIND_LEG_JOINTS + FRONT_LEG_JOINTS + SPINE_JOINTS + NECK_HEAD_JOINTS + COSMETIC_JOINTS
assert len(ALL_JOINTS) == 38, f"expected 38 joints, partition has {len(ALL_JOINTS)}"
assert len(set(ALL_JOINTS)) == 38, "joint partition has a duplicate — groups must be disjoint"

# ---------------------------------------------------------------------------
# Two derived orderings the AMP env needs (kept here so joint-name truth lives
# in ONE module):
#
#   REFERENCE_JOINT_ORDER — the 38 joints in URDF *declaration* order, which is
#     identical to the reference .npy's joints_list column order. The AMP
#     observation MUST read joints in this order (preserve_order=True) so the
#     discriminator lines up with the dataset columns.
#
#   ACTIVE_JOINTS — the 34 joints the policy actuates in phase 1 (all except the
#     4 cosmetic Jaw/Ears/Tail, which are held at default). This is the action
#     space; matches dof_spec.phase1_active_dof = 34.
# ---------------------------------------------------------------------------
REFERENCE_JOINT_ORDER = [
    "Pelvis_to_Spine_dof0", "Spine_dof0_to_Spine", "Spine_to_Spine1", "Spine1_to_Spine2",
    "Spine2_to_Neck_dof0", "Neck_dof0_to_Neck", "Neck_to_Neck1_dof0", "Neck1_dof0_to_Neck1",
    "Neck1_to_Head_dof0", "Head_dof0_to_Head", "Head_to_L_Ear", "Head_to_R_Ear", "Head_to_Jaw",
    "Spine2_to_L_Clavicle", "L_Clavicle_to_L_UpperArm_dof0", "L_UpperArm_dof0_to_L_UpperArm",
    "L_UpperArm_to_L_Forearm", "L_Forearm_to_L_Hand", "L_Hand_to_L_Finger0",
    "Spine2_to_R_Clavicle", "R_Clavicle_to_R_UpperArm_dof0", "R_UpperArm_dof0_to_R_UpperArm",
    "R_UpperArm_to_R_Forearm", "R_Forearm_to_R_Hand", "R_Hand_to_R_Finger0",
    "Pelvis_to_L_Thigh_dof0", "L_Thigh_dof0_to_L_Thigh", "L_Thigh_to_L_Calf",
    "L_Calf_to_L_HorseLink", "L_HorseLink_to_L_Foot", "L_Foot_to_L_Toe0",
    "Pelvis_to_R_Thigh_dof0", "R_Thigh_dof0_to_R_Thigh", "R_Thigh_to_R_Calf",
    "R_Calf_to_R_HorseLink", "R_HorseLink_to_R_Foot", "R_Foot_to_R_Toe0",
    "Pelvis_to_Tail",
]
assert len(REFERENCE_JOINT_ORDER) == 38
assert set(REFERENCE_JOINT_ORDER) == set(ALL_JOINTS), "reference order must be a permutation of the 38 joints"

ACTIVE_JOINTS = [j for j in REFERENCE_JOINT_ORDER if j not in set(COSMETIC_JOINTS)]
assert len(ACTIVE_JOINTS) == 34, f"expected 34 active joints, got {len(ACTIVE_JOINTS)}"

# Distal hoof links that make ground contact (for foot contact sensors / rewards).
# Hind hooves = *_Toe0, front hooves = *_Finger0. (Validated as the ground-contact
# links in validate_usd.py.)
FEET_BODIES = ["L_Toe0", "R_Toe0", "L_Finger0", "R_Finger0"]
HIND_FEET_BODIES = ["L_Toe0", "R_Toe0"]
FRONT_FEET_BODIES = ["L_Finger0", "R_Finger0"]

# Leg segments that should NEVER touch the ground — only the hooves should. A
# penalty on contact here stops the deer propping itself up on folded shins /
# cannon bones / "knees" instead of standing on its feet.
NON_HOOF_LEG_BODIES = [
    "L_Thigh", "L_Calf", "L_HorseLink", "L_Foot",
    "R_Thigh", "R_Calf", "R_HorseLink", "R_Foot",
    "L_UpperArm", "L_Forearm", "L_Hand",
    "R_UpperArm", "R_Forearm", "R_Hand",
]

# Default composed USD (override with ANIML_DEER_USD, per CLAUDE.md convention).
_DEFAULT_USD = os.path.join(
    "assets", "deer_polyart_urdf", "deer_polyart.usd"
)


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _flag(name: str, *, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def deer_polyart_actuators() -> dict[str, ImplicitActuatorCfg]:
    """Five actuator groups partitioning the 38 joints, deliberate PD gains.

    Anatomical split (env-overridable):
      hind legs  : firm active drivers
      front legs : softer compliant struts
      spine      : soft sway
      neck+head  : softer sway
      cosmetic   : hold-at-default (phase-1 lock enforced at the action level)

    Effort/velocity limits are safety clamps well above the PD torque these gains
    produce at normal errors; they exist to cap runaway torque, not to shape the
    nominal motion.
    """
    return {
        # r7: stiffness ~2x (hind 100->200, front 80->160), damping ~2x. At the
        # old gains the legs had to sag ~0.4-0.8 rad off target to make enough
        # torque to hold the body up, so they folded to the joint limits and the
        # deer propped on the hard stops instead of standing. Stiffer legs hold
        # their pose (a real deer stands on near-rigid tendon-locked columns).
        "hind_legs": ImplicitActuatorCfg(
            joint_names_expr=HIND_LEG_JOINTS,
            stiffness=_f("ANIML_DEER_HIND_STIFFNESS", 200.0),
            damping=_f("ANIML_DEER_HIND_DAMPING", 12.0),
            effort_limit=_f("ANIML_DEER_HIND_EFFORT", 320.0),
            velocity_limit=_f("ANIML_DEER_HIND_VELOCITY", 20.0),
        ),
        "front_legs": ImplicitActuatorCfg(
            joint_names_expr=FRONT_LEG_JOINTS,
            stiffness=_f("ANIML_DEER_FRONT_STIFFNESS", 160.0),
            damping=_f("ANIML_DEER_FRONT_DAMPING", 12.0),
            effort_limit=_f("ANIML_DEER_FRONT_EFFORT", 260.0),
            velocity_limit=_f("ANIML_DEER_FRONT_VELOCITY", 20.0),
        ),
        "spine": ImplicitActuatorCfg(
            joint_names_expr=SPINE_JOINTS,
            stiffness=_f("ANIML_DEER_SPINE_STIFFNESS", 50.0),
            damping=_f("ANIML_DEER_SPINE_DAMPING", 4.0),
            effort_limit=_f("ANIML_DEER_SPINE_EFFORT", 150.0),
            velocity_limit=_f("ANIML_DEER_SPINE_VELOCITY", 15.0),
        ),
        "neck_head": ImplicitActuatorCfg(
            joint_names_expr=NECK_HEAD_JOINTS,
            stiffness=_f("ANIML_DEER_NECK_STIFFNESS", 40.0),
            damping=_f("ANIML_DEER_NECK_DAMPING", 3.0),
            effort_limit=_f("ANIML_DEER_NECK_EFFORT", 120.0),
            velocity_limit=_f("ANIML_DEER_NECK_VELOCITY", 15.0),
        ),
        "cosmetic": ImplicitActuatorCfg(
            joint_names_expr=COSMETIC_JOINTS,
            stiffness=_f("ANIML_DEER_COSMETIC_STIFFNESS", 50.0),
            damping=_f("ANIML_DEER_COSMETIC_DAMPING", 3.0),
            effort_limit=_f("ANIML_DEER_COSMETIC_EFFORT", 60.0),
            velocity_limit=_f("ANIML_DEER_COSMETIC_VELOCITY", 10.0),
        ),
    }


def deer_polyart_articulation_cfg(usd_path: str | None = None) -> ArticulationCfg:
    """ArticulationCfg for the Tier-3 deer. Caller sets ``prim_path``.

    Spawn: identity rotation (USD is already upright, +X-forward), joints at
    zero (validated standing pose). ``spawn_z`` default 0.95 puts the front
    hooves at the ground; the hind hooves sit ~6 cm higher in the rest pose and
    settle on the first physics step (a rest-pose property of the rig, harmless —
    RSI resets from the reference clip put all four feet in contact per frame).
    """
    usd_path = usd_path or os.environ.get("ANIML_DEER_USD", _DEFAULT_USD)
    z = _f("ANIML_DEER_SPAWN_Z", 0.95)
    # Contact reporting ON by default — the new USD has real rigid bodies, and
    # the AMP env attaches foot contact sensors. (Old rig defaulted this OFF
    # because DCC-exported meshes lacked rigid bodies; not our case.)
    contact_sensors = _flag("ANIML_DEER_ACTIVATE_CONTACT_SENSORS", default=True)

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.05,
                angular_damping=0.05,
                # PhysX NaN-drift guard — see module docstring / CLAUDE.md.
                max_linear_velocity=30.0,
                max_angular_velocity=15.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, z),
            rot=(1.0, 0.0, 0.0, 0.0),      # identity — USD is already upright
            joint_pos={".*": 0.0},          # joint-zero == standing (validated)
            joint_vel={".*": 0.0},
        ),
        actuators=deer_polyart_actuators(),
        soft_joint_pos_limit_factor=0.95,
    )
