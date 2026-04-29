"""AMP-augmented walk environment for the deer USD.

Extends ``DeerUsdWalkEnvCfg`` with:

1. Simplified task rewards (AMP's discriminator supplies the "how to move
   naturally" signal so we don't need leg_extension, flat_orientation_l2,
   feet_air_time, velocity_error_l2, or stationary_penalty any more).
2. An ``amp`` observation group that amp-rsl-rl's ``AMPOnPolicyRunner``
   reads via ``env.cfg.observations.amp.joint_pos.params['asset_cfg'].joint_names``.

The AMP observation concatenates per-step:
  - joint_pos        (15)
  - joint_vel        (15)
  - base_lin_vel     (3,  body frame)
  - base_ang_vel     (3,  body frame)
Total amp obs dim = 36. AMPLoader feeds the discriminator transition pairs
``(s_t, s_{t+1})`` → discriminator input dim = 72.

Task is registered as ``AniML-Deer-USD-Walk-AMP-v0`` in
``training/tasks/__init__.py``.
"""

from __future__ import annotations

from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from training.envs.deer_usd_rough_env_cfg import DeerUsdWalkEnvCfg


# The 15 URDF joint names in the same canonical order used by the motion
# converter (tools/convert_unity_anim_to_npy.py). amp-rsl-rl matches dataset
# joints against this list positionally, so ordering must stay in sync with
# the .npy file.
DEER_URDF_JOINT_NAMES: list[str] = [
    "pelvis_to_chest",
    "pelvis_to_HR_thigh",
    "HR_thigh_to_HR_shin",
    "HR_shin_to_HR_foot",
    "pelvis_to_HL_thigh",
    "HL_thigh_to_HL_shin",
    "HL_shin_to_HL_foot",
    "chest_to_FR_thigh",
    "FR_thigh_to_FR_shin",
    "FR_shin_to_FR_foot",
    "chest_to_FL_thigh",
    "FL_thigh_to_FL_shin",
    "FL_shin_to_FL_foot",
    "chest_to_neck",
    "neck_to_skull",
]


@configclass
class AMPObservationsCfg(ObsGroup):
    """Observation group consumed by the AMP discriminator.

    Each term's ``params.asset_cfg`` selects which asset + joint names are
    extracted. amp-rsl-rl pulls ``joint_pos.params['asset_cfg'].joint_names``
    to align dataset joint order with sim joint order.

    No noise/offsets here — the discriminator sees clean state.
    """

    joint_pos = ObsTerm(
        func=mdp_obs.joint_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=DEER_URDF_JOINT_NAMES),
        },
    )
    joint_vel = ObsTerm(
        func=mdp_obs.joint_vel,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=DEER_URDF_JOINT_NAMES),
        },
    )
    base_lin_vel = ObsTerm(func=mdp_obs.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp_obs.base_ang_vel)

    def __post_init__(self) -> None:
        # Discriminator uses raw concatenation, no history stacking.
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DeerUsdWalkAMPEnvCfgV2(DeerUsdWalkEnvCfg):
    """Run 2: kill standing-reward penalties, crank velocity tracking.

    Inherits from the base walk env (not from v1) to give us a clean slate.
    Same AMP observation group as v1, but task rewards rebalanced so the
    policy can't earn reward by staying still.

    Key deltas from V1:
      - track_lin_vel_xy_exp weight 1.0 -> 3.0 (stronger pull to move)
      - action_rate_l2 = 0.0 (was -0.01 — was rewarding stillness)
      - dof_torques_l2 = 0.0 (was -1e-5)
      - dof_acc_l2 = 0.0 (was -2.5e-8)
      - rel_standing_envs = 0.0 (every env gets a nonzero command)
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # ---- Add AMP observation group ----
        self.observations.amp = AMPObservationsCfg()

        # ---- Drop all the shaping rewards (AMP handles them) ----
        for attr in [
            "leg_extension",
            "feet_air_time",
            "velocity_error",
            "stationary_penalty",
            "base_height_l2",
            "flat_orientation_l2",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "undesired_contacts",
        ]:
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Task reward: strong velocity tracking, nothing else.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Remove smoothness penalties so the policy isn't rewarded for not
        # moving. AMP's discriminator implicitly enforces smoothness via the
        # natural-motion prior.
        self.rewards.action_rate_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0

        # Velocity command range: aim for the synthesized 0.8 m/s forward.
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Kill the velocity curriculum — we don't want the policy chasing a
        # moving target. Keep commands at their final range from step 0.
        # DeerUsdWalkEnvCfg installs ramp_vel_xy / ramp_vel_yaw via parent;
        # disable any that exist.
        for attr in ("ramp_vel_xy", "ramp_vel_yaw"):
            if hasattr(self.curriculum, attr):
                setattr(self.curriculum, attr, None)


@configclass
class DeerUsdWalkAMPEnvCfgV3(DeerUsdWalkAMPEnvCfgV2):
    """Run 5 (aka AMP-v4 task): aggressive changes aimed at breaking the
    standing-still local optimum.

    Inherits V2's reward simplifications, then layers:
      1. Reference-state initialization: each env resets to a random frame
         of the expert walk (joint poses, joint velocities, root forward
         velocity all match the expert clip). Standard AMP recipe.
      2. No pelvis_too_low termination: removes the specific cost of
         falling that trapped V2/V3 in standing.
      3. Linear forward-velocity reward: adds a reward term with gradient
         everywhere (unlike exp-tracking which saturates at high error).
      4. Shorter episodes (20s -> 5s): the survival reward stacks with
         episode length, so shorter episodes make standing-still less
         lucrative per episode.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            forward_velocity_linear,
            reset_from_amp_motion,
        )

        # --- (1) Reference-state initialization ---
        # Replace both reset_robot_joints and reset_base with a single RSI
        # event that sets joint + root state together from the baked walk.
        # (Leaving reset_robot_joints alone would later overwrite our joint
        # state with default_joint_pos + small noise.)
        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints = None
        self.events.reset_base = EventTerm(
            func=reset_from_amp_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_height_m": 0.93,
                "pose_jitter": 0.05,
                "vel_jitter": 0.1,
            },
        )

        # --- (2) No pelvis_too_low termination ---
        if hasattr(self.terminations, "pelvis_too_low"):
            self.terminations.pelvis_too_low = None

        # --- (3) Linear forward-velocity reward ---
        # Weight is picked so at full command-match we get ~1.0 per step,
        # matching the magnitude of style reward (~0.3) and exp-tracking
        # (~1.0 at error=0). Use weight 2.0 so the linear reward dominates
        # at non-zero error where exp-tracking goes to zero.
        self.rewards.forward_velocity_linear = RewTerm(
            func=forward_velocity_linear,
            weight=2.0,
            params={"command_name": "base_velocity", "clip_max": 1.5},
        )

        # --- (4) Shorter episodes ---
        # Default is 20s (from LocomotionVelocityRoughEnvCfg). 5s is enough
        # to show 2-3 strides if the deer walks.
        self.episode_length_s = 5.0


@configclass
class DeerUsdWalkAMPEnvCfgV4(DeerUsdWalkAMPEnvCfgV3):
    """Run 6 (AMP-v5 task): fixes the axis mismatch that made V3's walking
    policy fake.

    Inherits V3 (RSI + no pelvis termination + 5s episodes + linear vel
    reward) but:
      1. Replaces Isaac Lab's default ``track_lin_vel_xy_exp`` /
         ``track_ang_vel_z_exp`` rewards (which assume body +X = forward)
         with anatomy-aligned versions that use body -Z = forward.
      2. Replaces ``forward_velocity_linear`` with
         ``forward_velocity_linear_anatomical`` for the same reason.
      3. Adds ``base_orientation_too_tilted`` termination so the ep_len
         metric is no longer gamed by fallen-and-sliding deer.
      4. Uses the re-baked ``deer_walk_reset_states.pt`` (body -Z forward),
         inherited automatically via RSI.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            track_forward_vel_anatomical_exp,
            track_lateral_vel_anatomical_exp,
            track_yaw_rate_anatomical_exp,
            forward_velocity_linear_anatomical,
            base_orientation_too_tilted,
        )

        # ---- Remove the body-+X-assuming rewards from the inherited chain ----
        for attr in [
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "forward_velocity_linear",   # the body-+X version added in V3
        ]:
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # ---- Anatomy-aligned task rewards ----
        # Weights chosen so this is net dominant over AMP style (~0.3/step)
        # at reasonable tracking errors.
        self.rewards.track_forward_vel = RewTerm(
            func=track_forward_vel_anatomical_exp,
            weight=3.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_lateral_vel = RewTerm(
            func=track_lateral_vel_anatomical_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_yaw_rate = RewTerm(
            func=track_yaw_rate_anatomical_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        # Linear forward-velocity reward (dense gradient everywhere)
        self.rewards.forward_velocity_linear = RewTerm(
            func=forward_velocity_linear_anatomical,
            weight=2.0,
            params={"command_name": "base_velocity", "clip_max": 1.5},
        )

        # ---- Orientation-based fall termination ----
        self.terminations.base_tilt = DoneTerm(
            func=base_orientation_too_tilted,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_angle_deg": 60.0},
        )


@configclass
class DeerUsdWalkAMPEnvCfgV6(DeerUsdWalkAMPEnvCfgV4):
    """Run 10: adds leg_joint_activity reward on top of V4's anatomy-aligned
    setup, to break the 'rigid-lever hop' local optimum seen in run 9.
    Uses action_scale=1.0 inherited via V5-style override below.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        from isaaclab.envs.mdp import actions as mdp_actions
        from training.envs.deer_rewards import leg_joint_activity

        # Match V5's action_scale bump
        self.actions.joint_pos = mdp_actions.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=1.0,
            use_default_offset=True,
        )

        # Reward leg-joint motion magnitude to punish rigid-lever strategies.
        # Weight 2.5 so ~1 rad/s average joint velocity per episode earns
        # ~600 total reward — roughly matching track_forward_vel scale.
        self.rewards.leg_joint_activity = RewTerm(
            func=leg_joint_activity,
            weight=2.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class DeerUsdWalkAMPEnvCfgV7(DeerUsdWalkAMPEnvCfgV6):
    """Run 12 (AMP-v10 task): anti-belly-slide fixes.

    R11b produced a belly-down motorboat gait: ep_len pegged at 87% time_out
    with leg_activity=15 but pelvis dragging on the ground.  The old 60° tilt
    termination was too permissive (nose-down-40° passed).  This subclass:

      1. Tightens ``base_tilt`` cutoff 60° → 35°.
      2. Re-enables ``pelvis_too_low`` at min_height=0.55m (belly cutoff).
      3. Adds ``upright_bonus`` reward (pelvis > 0.70m, weight 2.0) so
         there's a dense gradient toward standing up, not just a cliff at
         the termination threshold.
      4. Bumps num_envs 1500 → 2500 (5070 has ~2 GiB headroom at 1500).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            pelvis_too_low,
            upright_bonus,
        )

        # ---- Tighter tilt cutoff ----
        self.terminations.base_tilt.params["max_angle_deg"] = 35.0

        # ---- Pelvis-height termination (belly on ground = reset) ----
        self.terminations.pelvis_too_low = DoneTerm(
            func=pelvis_too_low,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_names": ["pelvis"],
                "min_height": 0.55,
            },
        )

        # ---- Dense upright reward (+1/step when pelvis > 0.70m) ----
        self.rewards.upright_bonus = RewTerm(
            func=upright_bonus,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_names": ["pelvis"],
                "min_height": 0.70,
            },
        )

        # ---- Throughput: use the 5070's headroom ----
        self.scene.num_envs = 2500


@configclass
class DeerUsdWalkAMPEnvCfgV8(DeerUsdWalkAMPEnvCfgV7):
    """Run 13 (AMP-v11 task): kill the crouch-shuffle that r12 converged to.

    R12 produced a frozen pose: pelvis hovering in the 0.55-0.70m gap
    between termination and binary upright_bonus, body pitched nose-down
    just under the 35° tilt cutoff, antlers near the ground.  Two binary
    cliffs gave the policy a sweet spot to camp in.

    Fixes:
      1. Replace binary ``upright_bonus`` with linear ``pelvis_height_ramp``
         (dense gradient 0.55 → 1.00 m).  No flat zone to nest in.
      2. Add ``nose_pitch_penalty`` (weight -2.0) — directly attacks the
         nose-down strut geometry that the magnitude-only tilt cutoff
         couldn't distinguish from lateral roll.
      3. Tighten ``pelvis_too_low`` 0.55 → 0.70 m.  Eliminates the gap
         between termination and reward.  Aggressive — initial term
         rate may spike >90% before policy adapts.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            pelvis_height_ramp,
            nose_pitch_penalty,
        )

        # ---- Replace binary upright_bonus with linear ramp ----
        if hasattr(self.rewards, "upright_bonus"):
            delattr(self.rewards, "upright_bonus")
        self.rewards.pelvis_height_ramp = RewTerm(
            func=pelvis_height_ramp,
            weight=4.0,        # dense ramp, scale matches track_forward_vel
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_names": ["pelvis"],
                "z_min": 0.55,
                "z_max": 1.00,
            },
        )

        # ---- Penalize body pitch (forward-axis projection of gravity) ----
        self.rewards.nose_pitch_penalty = RewTerm(
            func=nose_pitch_penalty,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ---- Tighter pelvis cutoff (no gap between term and reward) ----
        self.terminations.pelvis_too_low.params["min_height"] = 0.70


@configclass
class DeerUsdWalkAMPEnvCfgV9(DeerUsdWalkAMPEnvCfgV8):
    """Run 14 (AMP-v12 task): attack the neck-bend cheat seen in r13.

    R13 playback showed crouched-prop pose: pelvis horizontal (satisfying
    both tilt-35° and nose_pitch_penalty), front legs splayed forward as
    static struts, neck folded down so skull / antlers graze the ground.
    The pitch penalty was near-zero (-0.065/ep) because it measures the
    pelvis body axis — but the pose puts the head down via neck joints,
    not base pitch.

    Fixes:
      1. Drop ``nose_pitch_penalty`` — it was measuring the wrong body.
      2. Add ``head_height_ramp`` — reuses ``pelvis_height_ramp`` on the
         skull body with z_min=0.60, z_max=1.20m (adult deer standing head
         is ~1.4m).  Dense gradient rewarding head-up posture; policy can
         no longer camp with pelvis level + neck folded.
      3. Keep everything else from V8 (pelvis_too_low 0.70m,
         pelvis_height_ramp, tilt-35°, 2500 envs).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import pelvis_height_ramp

        # ---- Drop pitch penalty (measured wrong body) ----
        if hasattr(self.rewards, "nose_pitch_penalty"):
            delattr(self.rewards, "nose_pitch_penalty")

        # ---- Reward head height (dense ramp on skull body) ----
        # Skull world-Z: adult deer standing ~1.4m, crouched ~0.9m,
        # nose-on-ground prop pose ~0.3-0.5m. Ramp 0.60→1.20 gives
        # full reward for standing and zero for the prop pose.
        # Weight 3.0 so this dominates: 1.0 * 3.0 * 250 = 750 vs
        # track_forward_vel's peak ~750.
        self.rewards.head_height_ramp = RewTerm(
            func=pelvis_height_ramp,
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_names": ["skull"],
                "z_min": 0.60,
                "z_max": 1.20,
            },
        )


@configclass
class DeerUsdWalkAMPEnvCfgV5(DeerUsdWalkAMPEnvCfgV4):
    """Run 9+: same as V4 but action_scale bumped from 0.5 to 1.0 so joints
    can reach the expert walk clip's full knee-bend range (~60° = 1.0 rad).
    With action_scale=0.5 the policy was physically capped at half the needed
    range, producing stiff-looking legs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        from isaaclab.envs.mdp import actions as mdp_actions
        self.actions.joint_pos = mdp_actions.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=1.0,
            use_default_offset=True,
        )


@configclass
class DeerUsdWalkAMPEnvCfg(DeerUsdWalkEnvCfg):
    """Walk env with AMP discriminator observation group and simplified rewards.

    The inherited DeerUsdWalkEnvCfg keeps: track_lin_vel_xy_exp,
    track_ang_vel_z_exp, base_height_l2, action_rate_l2, dof_torques_l2,
    dof_acc_l2, undesired_contacts, lin_vel_z_l2, ang_vel_xy_l2, and the
    pelvis_too_low termination. We remove everything the discriminator
    implicitly handles so the task reward stays clean.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # ---- Add AMP observation group ----
        # ManagerBasedRLEnvCfg.observations is a top-level namespace; we
        # attach a new attribute that the runner looks up by name "amp".
        self.observations.amp = AMPObservationsCfg()

        # ---- Simplify rewards ----
        # AMP's style reward handles "move like the expert" implicitly, so
        # we drop the shaping terms we added to fight the standing-still
        # optimum. Keep only the velocity tracking + the minimum stability
        # / smoothness penalties.
        for attr in [
            "leg_extension",
            "feet_air_time",
            "velocity_error",
            "stationary_penalty",
            "base_height_l2",         # AMP reconstructs desired height via motion prior
            "flat_orientation_l2",    # AMP handles uprightness
            "lin_vel_z_l2",           # small-signal, let AMP shape
            "ang_vel_xy_l2",          # small-signal, let AMP shape
            "undesired_contacts",     # keep commented-out -- enable if legs clip
        ]:
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Primary task reward: follow velocity commands. AMP multiplies the
        # total reward by its style factor internally (0.5 task + 0.5 style
        # by default -- see amp_on_policy_runner.py).
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        # Minimal smoothness penalties (kept small so AMP style dominates):
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-8

        # Velocity command range: matches the motion's synthesized forward
        # speed (0.8 m/s). Keep a small window so the AMP discriminator's
        # forward-speed prior is meaningful.
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # 0% standing envs -- AMP should not see "stand still" as matching
        # the walk motion. Re-enable later with a separate idle clip.
        self.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class DeerUsdWalkAMPEnvCfgV12(DeerUsdWalkAMPEnvCfg):
    """Run 21+: multi-clip motion library (walk + run, optionally + idle).

    Inherits V11's reward shape (zeroed L2 smoothness), then widens the
    velocity command range so the policy is asked to span both gait speeds
    in the dataset (walk = 0.8 m/s, run = 2.5 m/s).

    Designed to be paired with DEER_USD_AMP_WALK_RUN_CFG or
    DEER_USD_AMP_WALK_RUN_IDLE_CFG agent dicts (which add the run/idle
    .npy files to the AMP discriminator's dataset).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # V11's L2 zeroing
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        # Span walk (0.8 m/s) AND run (2.5 m/s) gait speeds
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.7, 0.7)


@configclass
class DeerUsdWalkAMPEnvCfgV18(DeerUsdWalkAMPEnvCfg):
    """Run 30: V17 + zero action_rate_l2 weight to free up high-frequency
    joint motion needed for real walking speed.

    r29 produced consistent forward direction (all envs went -Y)
    but speed was still ~0.04 m/s vs commanded 1.0 m/s. The bidirectional
    bursts within episodes ("walk forward, pause, sometimes reverse")
    suggest the policy can't sustain the rapid alternation of joint
    targets that real walking requires.

    Hypothesis: action_rate_l2 = -0.01 was penalizing exactly the high-
    frequency action changes a walking gait needs (each leg cycle has
    several inflection points where target changes sign rapidly).

    V18 zeros action_rate_l2.weight so the policy is free to swing legs
    fast. Risk: noise_std might drift up like r16. Defensive std-clamp
    in train_amp.py is in place to prevent NaN crash regardless.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            track_forward_vel_anatomical_exp,
            track_yaw_rate_anatomical_exp,
            forward_velocity_linear_anatomical,
            reset_from_amp_motion,
        )

        # ZERO ALL smoothness penalties — free joint motion
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0   # NEW: was -0.01

        # Locked command (V17 inherited)
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Disable wrong-axis curriculum
        for attr in ("ramp_vel_xy", "ramp_vel_yaw"):
            if hasattr(self.curriculum, attr):
                setattr(self.curriculum, attr, None)

        # Remove wrong-axis stock rewards
        for attr in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Anatomy-aligned tracking
        self.rewards.track_forward_vel = RewTerm(
            func=track_forward_vel_anatomical_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_yaw_rate = RewTerm(
            func=track_yaw_rate_anatomical_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.forward_velocity_linear = RewTerm(
            func=forward_velocity_linear_anatomical,
            weight=4.0,
            params={"command_name": "base_velocity", "clip_max": 1.5},
        )

        # RSI from walk.npy
        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints = None
        self.events.reset_base = EventTerm(
            func=reset_from_amp_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_height_m": 0.93,
                "pose_jitter": 0.05,
                "vel_jitter": 0.1,
            },
        )


@configclass
class DeerUsdWalkAMPEnvCfgV17(DeerUsdWalkAMPEnvCfg):
    """Run 29: V16 + locked command range to single forward value.

    r28 produced real walking (1.2-1.5m of forward motion in 30s) but
    bidirectional — sometimes the deer walked +Y, sometimes -Y, never
    locking onto the commanded direction's sign. Hypothesis: the random
    lateral / yaw / variable speed commands confuse the policy enough
    that it ignores command sign in favor of "look natural while moving".

    V17 locks the command range to a single value: forward at 1.0 m/s
    only. No lateral, no yaw, no speed variability. Forces the policy
    to learn one consistent forward-tracking behavior. Once that's solid
    we can re-broaden in r30+.

    Otherwise inherits V16: RSI from walk, anatomy-aligned tracking,
    forward_velocity_linear w=4.0, pelvis_too_low termination ON.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            track_forward_vel_anatomical_exp,
            track_yaw_rate_anatomical_exp,
            forward_velocity_linear_anatomical,
            reset_from_amp_motion,
        )

        # V11 zeroed L2
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0

        # LOCKED COMMAND: forward at 1.0 m/s, nothing else.
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Disable curriculum (we have no range to ramp now anyway)
        for attr in ("ramp_vel_xy", "ramp_vel_yaw"):
            if hasattr(self.curriculum, attr):
                setattr(self.curriculum, attr, None)

        # Remove wrong-axis stock rewards
        for attr in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Anatomy-aligned tracking
        self.rewards.track_forward_vel = RewTerm(
            func=track_forward_vel_anatomical_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_yaw_rate = RewTerm(
            func=track_yaw_rate_anatomical_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.forward_velocity_linear = RewTerm(
            func=forward_velocity_linear_anatomical,
            weight=4.0,
            params={"command_name": "base_velocity", "clip_max": 1.5},
        )

        # RSI from walk.npy
        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints = None
        self.events.reset_base = EventTerm(
            func=reset_from_amp_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_height_m": 0.93,
                "pose_jitter": 0.05,
                "vel_jitter": 0.1,
            },
        )

        # pelvis_too_low termination stays ON (inherited from base)


@configclass
class DeerUsdWalkAMPEnvCfgV16(DeerUsdWalkAMPEnvCfg):
    """Run 28: V15 + pelvis_too_low termination back ON.

    r27 (V15) removed pelvis_too_low termination thinking RSI made it
    unnecessary. Result: catastrophic regression — all envs fell over
    and stayed down to game ep_len. Comparing across the run history,
    the ONLY runs that produced forward motion (r19 et al.) had
    pelvis_too_low termination ON. It's load-bearing: it keeps episodes
    short for fallen deer, removing the perverse incentive to lie down.

    Otherwise inherits V15: RSI from walk.npy, anatomy-aligned tracking
    rewards (forward = body -Z), forward_velocity_linear at weight 4.0,
    narrow command range 0.5-1.2 m/s.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            track_forward_vel_anatomical_exp,
            track_yaw_rate_anatomical_exp,
            forward_velocity_linear_anatomical,
            reset_from_amp_motion,
        )

        # V11's L2 zeroing
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        # V13's narrow command range
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # Disable wrong-axis curriculum
        for attr in ("ramp_vel_xy", "ramp_vel_yaw"):
            if hasattr(self.curriculum, attr):
                setattr(self.curriculum, attr, None)

        # Remove wrong-axis stock rewards
        for attr in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Anatomy-aligned tracking
        self.rewards.track_forward_vel = RewTerm(
            func=track_forward_vel_anatomical_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_yaw_rate = RewTerm(
            func=track_yaw_rate_anatomical_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.forward_velocity_linear = RewTerm(
            func=forward_velocity_linear_anatomical,
            weight=4.0,
            params={"command_name": "base_velocity", "clip_max": 1.5},
        )

        # RSI from walk.npy
        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints = None
        self.events.reset_base = EventTerm(
            func=reset_from_amp_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_height_m": 0.93,
                "pose_jitter": 0.05,
                "vel_jitter": 0.1,
            },
        )

        # KEEP pelvis_too_low termination ON (the difference vs V15).
        # Inherited default from base config kicks in if pelvis Z drops
        # below ~0.55 m, terminating the episode and removing the perverse
        # "fall and lie there for 1000 steps of free reward" incentive.


@configclass
class DeerUsdWalkAMPEnvCfgV15(DeerUsdWalkAMPEnvCfg):
    """Run 27: kick the deer out of the standing-still basin via three
    independent levers stacked together.

    r26 had the right reward shape (anatomy-aligned tracking + linear
    fwd reward) but the policy STILL converged to standing-still because:
      (a) base AMP env spawns deer in default standing pose — episode
          starts at the local optimum with no incentive to leave
      (b) idle.npy in the discriminator dataset rewards ~standing as
          natural-looking, so style reward is high at zero motion
      (c) linear forward weight 2.0 was outweighed by style+exp_track
          combination at standing

    Triple fix:
      1. RSI (reference state initialization): each reset draws a random
         frame from the baked walk reset states (.pt). Deer arrives in
         episode mid-walk — has to keep walking or fall.
      2. Pair with walk+run dataset (no idle) — `DEER_USD_AMP_WALK_RUN_CFG`.
         Removes idle's standing-as-natural attractor from style reward.
      3. Bump forward_velocity_linear weight 2.0 → 4.0. Make moving
         forward the dominant reward.

    Also drops pelvis_too_low termination (RSI starts deer at full speed,
    early-step "fall recovery" episodes are useful learning, not bugs).
    Inherits V11's zeroed L2 and V13's narrow command range otherwise.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from training.envs.deer_rewards import (
            track_forward_vel_anatomical_exp,
            track_yaw_rate_anatomical_exp,
            forward_velocity_linear_anatomical,
            reset_from_amp_motion,
        )

        # V11's L2 zeroing
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        # V13's narrow command range (walk speeds only)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # Disable wrong-axis curriculum
        for attr in ("ramp_vel_xy", "ramp_vel_yaw"):
            if hasattr(self.curriculum, attr):
                setattr(self.curriculum, attr, None)

        # Remove wrong-axis stock rewards
        for attr in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Anatomy-aligned exp tracking (body -Z forward)
        self.rewards.track_forward_vel = RewTerm(
            func=track_forward_vel_anatomical_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_yaw_rate = RewTerm(
            func=track_yaw_rate_anatomical_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        # Bumped from r26's 2.0 → 4.0. Forward motion is now the dominant
        # reward signal even when the discriminator likes standing-still.
        self.rewards.forward_velocity_linear = RewTerm(
            func=forward_velocity_linear_anatomical,
            weight=4.0,
            params={"command_name": "base_velocity", "clip_max": 1.5},
        )

        # --- THE KEY FIX: RSI ---
        # Replace default reset events with one that initializes from
        # a random frame of the baked walk motion. Deer arrives mid-walk
        # with appropriate joint vels and forward body velocity.
        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints = None
        self.events.reset_base = EventTerm(
            func=reset_from_amp_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_height_m": 0.93,
                "pose_jitter": 0.05,
                "vel_jitter": 0.1,
            },
        )

        # Drop pelvis_too_low termination — RSI starts deer at full speed,
        # so first-step fall-recovery episodes are useful learning.
        if hasattr(self.terminations, "pelvis_too_low"):
            self.terminations.pelvis_too_low = None


@configclass
class DeerUsdWalkAMPEnvCfgV14(DeerUsdWalkAMPEnvCfg):
    """Run 26: FIX the wrong-axis velocity tracking reward + add a linear
    forward-velocity reward to break the standing-still local optimum.

    Critical bug found 2026-04-28: Isaac Lab's stock track_lin_vel_xy_exp
    measures body +X velocity, but the deer URDF has body -Z = anatomical
    forward (per CLAUDE.md). So r1-r25 all "tracked velocity" along the
    deer's anatomical-LEFT axis. Metrics looked OK because the deer
    sometimes drifted sideways during shuffling, but the policy was never
    actually rewarded for moving forward.

    Combined with the multi-clip discriminator giving good "natural-pose"
    style reward to standing-still configurations (matching idle.npy),
    the policy converged to "stand still and look natural" in r21-r25.

    Fixes:
      1. Remove track_lin_vel_xy_exp / track_ang_vel_z_exp (wrong axis).
      2. Add track_forward_vel_anatomical_exp (right axis: body -Z).
      3. Add track_yaw_rate_anatomical_exp (right axis).
      4. Add forward_velocity_linear_anatomical (weight 2.0): linear
         reward, ZERO when standing still, breaks the standing-still
         local optimum that saturating-exp can't escape.

    Inherits V11's zeroed L2 + V13's narrow command range. Pairs with
    multi-clip agent cfg (DEER_USD_AMP_WRI_NARROW_CFG or similar).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # V11's L2 zeroing (overflow-prone with widened limits)
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        # V13's narrow command range (walk speeds only)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # ---- THE FIX: anatomy-aligned velocity tracking ----
        from training.envs.deer_rewards import (
            track_forward_vel_anatomical_exp,
            track_yaw_rate_anatomical_exp,
            forward_velocity_linear_anatomical,
        )
        # Remove the wrong-axis stock rewards
        for attr in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Disable the curriculum terms that ramp the now-deleted
        # track_lin_vel_xy_exp / track_ang_vel_z_exp rewards
        for attr in ("ramp_vel_xy", "ramp_vel_yaw"):
            if hasattr(self.curriculum, attr):
                setattr(self.curriculum, attr, None)

        # Anatomy-aligned exp tracking on the correct (-Z) axis
        self.rewards.track_forward_vel = RewTerm(
            func=track_forward_vel_anatomical_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_yaw_rate = RewTerm(
            func=track_yaw_rate_anatomical_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        # Linear forward-velocity reward — dense gradient, zero at stop.
        # This is the term that aggressively pulls policy off the
        # "stand still and look natural" local optimum.
        self.rewards.forward_velocity_linear = RewTerm(
            func=forward_velocity_linear_anatomical,
            weight=2.0,
            params={"command_name": "base_velocity", "clip_max": 1.5},
        )


@configclass
class DeerUsdWalkAMPEnvCfgV13(DeerUsdWalkAMPEnvCfg):
    """Run 23: narrow command range (walk speeds only) + multi-clip dataset.

    r21/r22 showed: multi-clip dataset gave best-ever style metrics but
    velocity tracking regressed (lin_vel_R 0.33 -> 0.11). The V12 wide
    command range (0.3-2.5 m/s) made it impossible for one policy to
    actually track the command — it settled at a self-consistent gait the
    discriminator approved of, regardless of commanded speed.

    V13 keeps the multi-clip discriminator (which gets to see walk + run
    body-motion shapes) but constrains the command to walk speeds only
    (0.5-1.2 m/s). Hypothesis: the policy can match command at walk speed
    AND benefit from the run clip's richer body-motion signal in the
    discriminator manifold (the discriminator scores body shape, not
    speed; a slow walker that uses spine/legs like a runner still scores
    well).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # V11's L2 zeroing
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        # Walk-speed command range, lets task reward pull velocity to match
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class DeerUsdWalkAMPEnvCfgV11(DeerUsdWalkAMPEnvCfg):
    """Run 19: kill the tiny-weight L2 smoothness penalties that overflow
    when widened URDF limits + occasional PhysX numeric spikes feed huge
    joint accelerations into them.

    Pattern from r18: with the wider joint range, transient joint vel spikes
    near limit edges produce dof_acc values around 1e10. Multiplied by the
    -2.5e-8 weight, the per-step reward becomes -1e2 = OK. But when MULTIPLE
    spikes happen in one batch, their squared L2 sum overflows fp32 and
    cascades to -inf gradient -> std<0 -> torch crash.

    These terms were never load-bearing (weights were tiny by design). Only
    action_rate_l2 actually constrains policy behavior; keep that at the v0
    value to prevent noise_std drift (the r16 lesson).

    Deltas vs. base v0:
      - dof_torques_l2 weight: -1e-5 -> 0.0
      - dof_acc_l2 weight    : -2.5e-8 -> 0.0
      - action_rate_l2 weight: unchanged (-0.01)
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        # action_rate_l2 stays at -0.01 (inherited from base v0)


@configclass
class DeerUsdWalkAMPEnvCfgV10(DeerUsdWalkAMPEnvCfg):
    """Run 16: loosen the smoothness penalties so the policy can produce a
    real walk instead of a shuffle.

    Diagnosis from r15 visual playback: deer walks but with a stiff-legged
    shuffle. Reward breakdown showed Episode_Reward/action_rate_l2 = -0.94
    and dof_torques_l2 = -0.64 — those penalties were collectively bigger
    than the +0.39 AMP style reward, so the policy learned to minimize
    action changes rather than produce big leg swings (which require fast
    target changes during swing phase).

    Deltas vs. base v0:
      - action_rate_l2 weight: -0.01 -> -0.001 (10x lower)
      - dof_torques_l2 weight: -1e-5 -> -3e-6 (~3x lower)
      - dof_acc_l2 weight: unchanged (already negligible)
    Everything else (rewards, velocity range, RSI off, etc.) inherits from
    base v0.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.001
        self.rewards.dof_torques_l2.weight = -3.0e-6
        # dof_acc_l2 stays at -2.5e-8 (inherited from base)
