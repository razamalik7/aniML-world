"""Rough-terrain velocity locomotion with **your** deer (or quadruped) USD.

Requires ``ANIML_DEER_USD`` or a default file: ``.usd`` / ``.usda`` / ``.usdc``.

The parent MDP matches Isaac Lab's ANYmal velocity task, but body names for contacts
rewards/terminations are **not** hard-coded to ANYmal: set env vars if defaults miss
your rig (inspect Stage in Isaac Sim → Physics → rigid bodies / links).

**Honest scope:** swapping USD is the right *engineering* step; learning a *stable,
natural-looking* gait usually needs **many GPU-hours** and PD/actuator tuning—not a
single afternoon—unless you fine-tune from a strong prior or use motion capture /
reference trajectories.

**USD must be sim-ready:** Isaac Lab expects a subtree under the spawned robot with
``UsdPhysics.ArticulationRootAPI`` and rigid bodies (collision + ``RigidBodyAPI``).
Pure SkelRoot / render meshes will fail at articulation init or produce no useful
physics. Prepare the asset in Isaac Sim (or export with physics), then set
``ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1`` so contact reporting is enabled for
rewards/terminations.
"""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.envs.mdp import rewards as mdp_rewards
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from training.envs.deer_locomotion_env_cfg import _DEFAULT_NUM_ENVS
from training.envs.deer_usd_articulation import deer_articulation_cfg


def _params_copy(term) -> dict:
    p = getattr(term, "params", None) or {}
    try:
        return dict(p)
    except (TypeError, ValueError):
        return {k: p[k] for k in p}


def _require_usd_path() -> str:
    raw = os.environ.get("ANIML_DEER_USD", "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
    repo = Path(__file__).resolve().parents[2]
    for rel in (
        Path("deer_world.usd"),
        Path("deer_fixed_robot.usd"),
        Path("deer_standing_robot.usd"),
        Path("deer_robot_clean.usd"),
        Path("deer_robot.usd"),
        Path("assets") / "deer" / "deer.usd",
        Path("deer1.usdc"),
        Path("deer1.usd"),
        Path("deer.usdc"),
    ):
        candidate = (repo / rel).resolve()
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "Deer USD not found. Set ANIML_DEER_USD, or place one of: "
        "assets/deer/deer.usd, deer1.usdc, deer1.usd, deer.usdc at repo root "
        f"(repo={repo})."
    )


@configclass
class DeerUsdRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Same rough velocity MDP as ANYmal-C; robot + contact regexes come from your USD."""

    def __post_init__(self) -> None:
        super().__post_init__()

        usd_path = _require_usd_path()
        self.scene.robot = deer_articulation_cfg(usd_path).replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = _DEFAULT_NUM_ENVS

        imu_link = os.environ.get("ANIML_DEER_IMU_LINK", "pelvis").strip()
        if self.scene.height_scanner is not None:
            self.scene.height_scanner = self.scene.height_scanner.replace(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{imu_link}"
            )

        base_body = os.environ.get("ANIML_DEER_BODY_BASE", "pelvis").strip()
        foot_rx = os.environ.get(
            "ANIML_DEER_BODY_FOOT",
            r".*(FOOT|foot|Foot|hoof|Hoof|toe|Toe|ankle|Ankle|_f_|_F_|ToeLink)",
        ).strip()
        thigh_rx = os.environ.get(
            "ANIML_DEER_BODY_THIGH",
            r".*(thigh|shin|THIGH|Thigh|calf|Calf|knee|Knee|shank|Shank|UPPER|upper_leg)",
        ).strip()

        base_asset = SceneEntityCfg("robot", body_names=base_body)

        m = self.events.add_base_mass
        self.events.add_base_mass = EventTerm(
            func=m.func, mode=m.mode, params={**_params_copy(m), "asset_cfg": base_asset}
        )
        m = self.events.base_com
        self.events.base_com = EventTerm(
            func=m.func, mode=m.mode, params={**_params_copy(m), "asset_cfg": base_asset}
        )
        m = self.events.base_external_force_torque
        self.events.base_external_force_torque = EventTerm(
            func=m.func, mode=m.mode, params={**_params_copy(m), "asset_cfg": base_asset}
        )

        f = self.rewards.feet_air_time
        self.rewards.feet_air_time = RewTerm(
            func=f.func,
            weight=f.weight,
            params={
                **_params_copy(f),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_rx),
            },
        )
        u = self.rewards.undesired_contacts
        self.rewards.undesired_contacts = RewTerm(
            func=u.func,
            weight=u.weight,
            params={
                **_params_copy(u),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=thigh_rx),
            },
        )

        base_contact_threshold = float(os.environ.get("ANIML_DEER_BASE_CONTACT_THRESHOLD", "1500"))
        if base_contact_threshold > 0:
            t = self.terminations.base_contact
            self.terminations.base_contact = DoneTerm(
                func=t.func,
                params={
                    **_params_copy(t),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=base_body),
                    "threshold": base_contact_threshold,
                },
            )
        else:
            self.terminations.base_contact = None


@configclass
class DeerUsdFlatEnvCfg(DeerUsdRoughEnvCfg):
    """Flat plane with body-part height rewards that teach standing posture.

    Instead of fighting coordinate transforms, we reward:
      - feet low (near ground)
      - head above body
      - body above feet (upright)
      - spine level (chest ≈ pelvis height)
    The agent discovers the correct standing pose from any spawn orientation.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            all_feet_in_contact,
            body_ground_contact_penalty,
            feet_below_body_target,
            feet_height_l2,
            feet_under_body,
            foot_height_spread,
            foot_weight_bearing,
            head_above_base,
            hind_feet_contact,
            hind_feet_height_l2,
            hind_joint_pos_l2,
            joint_pos_standing_l2,
            min_foot_contact,
            pelvis_too_low,
            upright_bonus,
            worst_foot_height,
        )

        robot_cfg = SceneEntityCfg("robot")
        foot_list = ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]
        hind_foot_list = ["HL_foot", "HR_foot"]
        foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=foot_list)
        hind_foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=hind_foot_list)
        # Pelvis only: chest links pick up huge non-ground artifacts on GPU contact.
        torso_bodies = os.environ.get("ANIML_DEER_TORSO_CONTACT_BODIES", "pelvis").strip()
        torso_list = [b.strip() for b in torso_bodies.split(",") if b.strip()]
        torso_sensor_cfg = SceneEntityCfg("contact_forces", body_names=torso_list)

        rj = self.events.reset_robot_joints
        self.events.reset_robot_joints = EventTerm(
            func=rj.func,
            mode=rj.mode,
            params={**_params_copy(rj), "position_range": (0.9, 1.1)},
        )

        # -- Deterministic upright spawn: no pose or velocity randomization --
        self.events.reset_base = EventTerm(
            func=self.events.reset_base.func,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            },
        )
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None

        # -- Positive rewards --
        self.rewards.upright_bonus = RewTerm(
            func=upright_bonus, weight=5.0,
            params={"asset_cfg": robot_cfg, "base_names": ["pelvis"], "min_height": -0.2},
        )

        self.rewards.feet_below_body = RewTerm(
            func=feet_below_body_target, weight=-15.0,
            params={"asset_cfg": robot_cfg, "foot_names": foot_list,
                    "body_names": ["pelvis"], "target_gap": 0.05},
        )

        # -- Penalties --
        self.rewards.feet_height_l2 = RewTerm(
            func=feet_height_l2, weight=-20.0,
            params={"asset_cfg": robot_cfg, "foot_names": foot_list, "target_z": 0.0},
        )

        self.rewards.foot_spread = RewTerm(
            func=foot_height_spread, weight=-15.0,
            params={"asset_cfg": robot_cfg, "foot_names": foot_list},
        )

        self.rewards.head_above_base = RewTerm(
            func=head_above_base, weight=-8.0,
            params={"asset_cfg": robot_cfg, "head_names": ["skull"],
                    "base_names": ["pelvis"], "target_diff": 0.4},
        )

        self.rewards.joint_standing = RewTerm(
            func=joint_pos_standing_l2, weight=-20.0,
            params={"asset_cfg": robot_cfg},
        )

        self.rewards.feet_under_body = RewTerm(
            func=feet_under_body, weight=-25.0,
            params={"asset_cfg": robot_cfg, "foot_names": foot_list, "body_names": ["pelvis"]},
        )

        self.rewards.worst_foot_height = RewTerm(
            func=worst_foot_height, weight=-25.0,
            params={"asset_cfg": robot_cfg, "foot_names": foot_list, "target_z": 0.0},
        )

        self.rewards.hind_feet_height = RewTerm(
            func=hind_feet_height_l2, weight=-20.0,
            params={"asset_cfg": robot_cfg, "foot_names": hind_foot_list, "target_z": 0.0},
        )

        self.rewards.hind_joint_standing = RewTerm(
            func=hind_joint_pos_l2, weight=-30.0,
            params={"asset_cfg": robot_cfg},
        )

        foot_ct_thresh = float(os.environ.get("ANIML_DEER_FOOT_CONTACT_THRESH", "5.0"))
        self.rewards.all_feet_contact = RewTerm(
            func=all_feet_in_contact, weight=8.0,
            params={"sensor_cfg": foot_sensor_cfg, "threshold": foot_ct_thresh},
        )
        self.rewards.foot_load = RewTerm(
            func=foot_weight_bearing, weight=5.0,
            params={"sensor_cfg": foot_sensor_cfg, "asset_cfg": robot_cfg},
        )
        self.rewards.min_foot_contact = RewTerm(
            func=min_foot_contact, weight=-8.0,
            params={"sensor_cfg": foot_sensor_cfg, "target_force": 20.0},
        )
        self.rewards.hind_feet_contact = RewTerm(
            func=hind_feet_contact, weight=10.0,
            params={"sensor_cfg": hind_foot_sensor_cfg, "threshold": foot_ct_thresh},
        )
        torso_viol_thresh = float(os.environ.get("ANIML_DEER_TORSO_CONTACT_FORCE_THRESH", "5"))
        torso_cap = float(os.environ.get("ANIML_DEER_TORSO_PENALTY_CAP", "400"))
        self.rewards.torso_ground_contact = RewTerm(
            func=body_ground_contact_penalty, weight=0.0,
            params={
                "sensor_cfg": torso_sensor_cfg,
                "threshold": torso_viol_thresh,
                "max_violation": torso_cap,
            },
        )

        target_h = float(os.environ.get("ANIML_DEER_TARGET_HEIGHT", "-0.03"))
        self.rewards.base_height_l2 = RewTerm(
            func=mdp_rewards.base_height_l2, weight=-25.0,
            params={"target_height": target_h, "asset_cfg": robot_cfg},
        )

        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.undesired_contacts.weight = -0.25
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # -- No early termination: min_height=-999 so it never fires.
        # Terminating on fall teaches the policy to collapse intentionally.
        # Rewards handle all posture incentives; episodes always run to time_out.
        self.terminations.pelvis_too_low = DoneTerm(
            func=pelvis_too_low,
            params={"asset_cfg": robot_cfg, "base_names": ["pelvis"], "min_height": -999.0},
        )

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class DeerUsdStandEnvCfg(DeerUsdRoughEnvCfg):
    """Phase-0 trivial standing: minimal rewards, tiny action scale, no randomization.

    The optimal policy is approximately "output zeros" — the PD controller holds
    the IK standing joint angles. This should converge in <200 iterations and
    proves the full pipeline works before adding complexity.

    Progression plan:
      Phase 0 (this): deterministic spawn, 4 rewards, action_scale 0.25
      Phase 1: add joint randomization (0.95, 1.05)
      Phase 2: add orientation perturbations
      Phase 3: velocity commands → walking
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from isaaclab.envs.mdp import actions as mdp_actions

        from training.envs.deer_rewards import (
            pelvis_too_low,
            leg_extension_reward,
            feet_under_chest,
        )

        robot_cfg = SceneEntityCfg("robot")

        # Disable self-collision for performance (not needed for static standing)
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        # Action scale: policy output × 1.0 = joint offset in radians
        self.actions.joint_pos = mdp_actions.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=1.0,
            use_default_offset=True,
        )

        # -- Deterministic spawn: always start at exact standing pose --
        rj = self.events.reset_robot_joints
        self.events.reset_robot_joints = EventTerm(
            func=rj.func,
            mode=rj.mode,
            params={**_params_copy(rj), "position_range": (1.0, 1.0)},
        )

        # -- Disable ALL pose and velocity randomization on reset --
        self.events.reset_base = EventTerm(
            func=self.events.reset_base.func,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            },
        )
        self.events.push_robot = None

        # -- Standing reward set --
        # Best performing base: leg_extension (pelvis height) + base_height + orientation.
        # New addition: feet_under_chest penalizes front feet splayed forward of chest.

        # 1. Pelvis-to-foot vertical gap: can't be faked by lying flat or splooting
        self.rewards.leg_extension = RewTerm(
            func=leg_extension_reward, weight=20.0,
            params={"asset_cfg": robot_cfg,
                    "foot_names": ["HL_foot", "HR_foot", "FL_foot", "FR_foot"],
                    "body_names": ["pelvis"],
                    "target_gap": 0.89},
        )

        # 2. Pelvis world height: target 0.92m
        self.rewards.base_height_l2 = RewTerm(
            func=mdp_rewards.base_height_l2, weight=-15.0,
            params={"target_height": 0.92, "asset_cfg": robot_cfg},
        )

        # 3. Front feet under chest: penalize Y-offset between chest and front feet.
        #    Directly fixes "leaning forward on angled front legs" — feet must be
        #    below the chest, not forward of it.
        self.rewards.feet_under_chest = RewTerm(
            func=feet_under_chest, weight=-40.0,
            params={"asset_cfg": robot_cfg,
                    "front_foot_names": ["FL_foot", "FR_foot"],
                    "chest_names": ["chest"]},
        )

        # 4. Flat orientation: penalize roll/pitch
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp_rewards.flat_orientation_l2, weight=-5.0,
            params={"asset_cfg": robot_cfg},
        )

        # 5. Small torque penalty
        self.rewards.dof_torques_l2.weight = -1.0e-4

        # -- Zero out all parent rewards not explicitly set above --
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0
        for attr in [
            "feet_below_body", "feet_height_l2", "foot_spread",
            "head_above_base", "feet_under_body", "worst_foot_height",
            "hind_feet_height", "hind_joint_standing", "all_feet_contact",
            "foot_load", "min_foot_contact", "hind_feet_contact",
            "torso_ground_contact", "upright_bonus", "joint_standing",
            "all_feet_near_ground", "head_min_height",
        ]:
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # -- No early termination for standing training --
        # Terminating on fall teaches the policy to collapse on purpose to escape penalties.
        # Set min_height to -999 so it never fires — episodes always run to time_out.
        self.terminations.pelvis_too_low = DoneTerm(
            func=pelvis_too_low,
            params={"asset_cfg": robot_cfg, "base_names": ["pelvis"], "min_height": -999.0},
        )

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class DeerUsdWalkEnvCfg(DeerUsdRoughEnvCfg):
    """Phase 1 walking: flat terrain, velocity commands, natural gait emergence.

    Trains from scratch — no warm-start from standing (collapses exploration).
    Velocity commands sample 0→1.2 m/s uniformly so the policy learns both
    standing still AND walking from a single training run.

    Progression:
      Phase 1 (this): flat terrain, 0-1.2 m/s, feet_air_time=0.01
      Phase 2: increase feet_air_time to 0.05-0.125 once gait stabilises (~800 iters)
      Phase 3: enable terrain curriculum for robustness
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from isaaclab.envs.mdp import actions as mdp_actions
        from training.envs.deer_rewards import pelvis_too_low, leg_extension_reward, velocity_error_l2, stationary_penalty

        robot_cfg = SceneEntityCfg("robot")

        # Flat terrain, no height scan
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Action scale 0.5: standard for locomotion
        self.actions.joint_pos = mdp_actions.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )

        # Conservative velocity range for a large/heavy deer
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        # 10% standing envs: the iter-800 checkpoint has already adapted to velocity
        # commands. The ramp_standing curriculum will handle initial runs; for resumed
        # runs the step counter resets so ramp_standing completes early.
        self.commands.base_velocity.rel_standing_envs = 0.1

        # Small joint randomization on reset for robustness
        rj = self.events.reset_robot_joints
        self.events.reset_robot_joints = EventTerm(
            func=rj.func, mode=rj.mode,
            params={**_params_copy(rj), "position_range": (0.95, 1.05)},
        )

        # No pose randomization — the deer's init_state.rot is Rx(-90°), so adding
        # any Euler offsets (even yaw) gets interpreted incorrectly and spawns them
        # sideways or upside-down. Yaw-invariance comes from command sampling (all dirs),
        # not from spawn randomization.
        self.events.reset_base = EventTerm(
            func=self.events.reset_base.func,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            },
        )
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None

        # -- Walk reward balance (phase 2: break standing-still local optimum) --
        # orientation loosened to -1.0: forward lean is needed for locomotion.
        # feet_air_time raised to 2.0 to strongly reward actual stepping.
        # velocity tracking final weight raised to 1.5.
        # leg_extension fixed at 3.0 (just enough to prevent sploot, not dominate).
        self.rewards.flat_orientation_l2.weight = -1.0    # allow walking lean
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.undesired_contacts.weight = -0.1

        self.rewards.track_lin_vel_xy_exp.weight = 0.3
        self.rewards.track_ang_vel_z_exp.weight = 0.15

        # Keep feet_air_time as secondary gait signal
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.2

        # Hard velocity error penalty: directly punishes not matching the commanded velocity.
        # Unlike exp-tracking (which gives ~0.6 reward just standing), this is always
        # non-zero when the deer isn't moving as commanded.
        self.rewards.velocity_error = RewTerm(
            func=velocity_error_l2,
            weight=-0.5,
            params={"command_name": "base_velocity"},
        )

        # Stationary penalty: fires when a velocity command > 0.3 m/s is active
        # but the deer isn't moving. Directly breaks the standing-still local optimum.
        self.rewards.stationary_penalty = RewTerm(
            func=stationary_penalty,
            weight=-2.0,
            params={"command_name": "base_velocity", "min_cmd_speed": 0.3},
        )

        # Height floor: gentler than standing — walking naturally shifts CoM
        self.rewards.base_height_l2 = RewTerm(
            func=mdp_rewards.base_height_l2, weight=-5.0,
            params={"target_height": 0.85, "asset_cfg": robot_cfg},
        )

        # Leg extension: fixed at 3.0 — just enough to stay upright, not dominate.
        self.rewards.leg_extension = RewTerm(
            func=leg_extension_reward, weight=3.0,
            params={"asset_cfg": robot_cfg,
                    "foot_names": ["HL_foot", "HR_foot", "FL_foot", "FR_foot"],
                    "body_names": ["pelvis"],
                    "target_gap": 0.89},
        )

        # Remove other standing-specific rewards inherited from parent
        for attr in [
            "feet_under_chest", "upright_bonus",
            "all_feet_near_ground", "head_min_height", "joint_standing",
        ]:
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Pelvis at ~0.93m standing. 0.25m = truly fallen. Matches April 10 run.
        self.terminations.pelvis_too_low = DoneTerm(
            func=pelvis_too_low,
            params={"asset_cfg": robot_cfg, "base_names": ["pelvis"], "min_height": 0.25},
        )

        # Disable pelvis-contact termination for walking: the pelvis can graze ground
        # on spawn due to joint randomization causing a large first-step impulse (>1500N),
        # which kills every episode on step 1. pelvis_too_low handles actual falls.
        self.terminations.base_contact = None

        # -- Reward curriculum: velocity ramp only --
        # leg_extension is held FIXED at 10.0 for the entire run.
        # Previous runs decayed it 10→5 while velocity ramped 0.3→1.0 over the same
        # 500 iters — this caused the upright signal to weaken exactly as velocity
        # pressure increased, leading to permanent collapse at iter ~100.
        # The April 13 run that worked kept leg_extension=20.0 fixed throughout.
        from training.envs.deer_rewards import LinearRampRewardWeight

        # NOTE: common_step_counter counts env.step() calls (NOT total env steps).
        # 1 iter = 24 env.step() calls. 1500 envs run in parallel per step call.
        # 500 iters = 12,000 steps.
        self.curriculum.ramp_vel_xy = CurrTerm(
            func=LinearRampRewardWeight,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "start_weight": 0.3,
                "end_weight": 1.5,   # raised from 1.0 to push forward motion harder
                "ramp_start_step": 0,
                "ramp_end_step":   12_000,
            },
        )
        self.curriculum.ramp_vel_yaw = CurrTerm(
            func=LinearRampRewardWeight,
            params={
                "term_name": "track_ang_vel_z_exp",
                "start_weight": 0.15,
                "end_weight": 0.5,
                "ramp_start_step": 0,
                "ramp_end_step":   12_000,
            },
        )
        # leg_extension fixed at 3.0 — no decay needed, it's already low.
