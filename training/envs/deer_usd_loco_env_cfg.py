"""Pure reward-shaping locomotion env configs for the URDF v2 (21-DOF) deer.

This is the post-AMP path: no reference clips, no discriminator. Reward stack
follows the published Lee/Margolis/Hwangbo recipe (Isaac Lab port of
``legged_gym``) — what every shipping legged robot actually uses.

Three stages of curriculum:
  - Stage 1: discover trot — flat ground, forward-only command, no pushes.
  - Stage 2: full 3-DOF + push events for recovery training.
  - Stage 3: rough heightfield terrain.
"""

from __future__ import annotations

from isaaclab.envs.mdp import rewards as mdp_rewards
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg as ObsTerm

from training.envs.deer_usd_rough_env_cfg import DeerUsdRoughEnvCfg


@configclass
class DeerUsdLocoStage1Cfg(DeerUsdRoughEnvCfg):
    """Stage 1: discover trot on flat ground from scratch.

    - Forward-only velocity command [0, 0.8 m/s], 20% standing envs.
    - No push events.
    - Flat plane terrain.
    - Reward stack: tracking + feet_air_time + smoothness + slide-penalty +
      orientation + base-height + standing-pose deviation.

    Terminations: pelvis_too_low (0.55m), base_orientation_too_tilted (60°).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Local imports kept inside __post_init__ to avoid circulars during
        # package init (matches the pattern used in other deer env configs).
        from isaaclab.envs.mdp import actions as mdp_actions
        from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
            feet_air_time,
            feet_slide,
        )
        from training.envs.deer_rewards import (
            anti_parallel_thigh_l2,        # v4.28: penalize parallel hindquarter motion
            base_orientation_too_tilted,
            bilateral_joint_symmetry_ema,  # v4.5 (deprecated, kept for archive)
            bilateral_joint_symmetry_l2,   # v4.6: per-step L/R asymmetry penalty
            carpus_deep_flex_bonus,        # v4.5: deadbanded carpus deep-flex bonus
            feet_swing_clearance,
            forward_velocity_linear_anatomical,
            front_carpus_fold_penalty,     # v4.15: hard penalty for carpus out of fold range
            head_above_base,
            head_min_height_l2,            # v5.4: absolute skull-height floor (anti downward-dog)
            heading_hold_penalty,          # v4.16: anti-veer cumulative heading drift penalty
            leg_extension_reward,          # v5.4: un-fakeable standing pelvis-to-foot gap
            hind_knee_range_penalty,       # v4.13: hard penalty for hind knee out of ref range
            hipabd_zero_l2,                # v4.17: push hip abduction joints toward 0
            joint_pos_standing_l2,
            joint_track_reference_walk,    # v4.8: per-joint imitation of deer_walk.npy
            knee_flex_during_swing,    # v4.2: forces visible knee bend during swing phase
            pelvis_height_ramp,
            pelvis_too_low,
            reset_from_amp_motion,         # v4.12: RSI for breaking HR-knee local optimum
            neck_posture_track,        # v7: imitate reference neck/head carriage (head up)
            skull_pitch_l2,            # v4.2: chin-tuck penalty
            track_forward_vel_anatomical_exp,
            track_lateral_vel_anatomical_exp,
            track_yaw_rate_anatomical_exp,
        )

        # ---------------- Scene: mild rough heightfield from iter 0 ----------------
        # Per Walk-These-Ways / legged_gym: train on terrain from the start.
        # Random ground height is a *natural perturbation* that prevents the
        # policy from converging to degenerate stationary solutions (tuck-and-
        # freeze, rigid-front wheelbarrow). Mild ±0.5–4 cm bumps on a 40-cm grid.
        import isaaclab.terrains as terrain_gen
        from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
        rough = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=10.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.4,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "mild": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(0.005, 0.04),
                    noise_step=0.005,
                    border_width=0.5,
                ),
            },
        )
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = rough
        # Disable height scanner (we don't need to observe terrain — let the
        # policy learn proprioceptive blind walking).
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        # ---------------- Actions: scale 0.5 (Walk-These-Ways recipe value).
        # We had dropped to 0.3 fearing HAA destabilization. With the
        # ready-to-walk default pose + terrain perturbation, the policy needs
        # bigger action magnitude to swing limbs through stride.
        self.actions.joint_pos = mdp_actions.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )

        # ---------------- Commands: forward-only, NO standing envs ----------------
        # r3 hit a "tuck and freeze" local optimum: 20% standing envs gave
        # +1.0 reward each for not moving, exp-kernel tracking std=0.5 was
        # too forgiving on the other 80%, so the policy converged to ignoring
        # commands and standing still in a tucked pose. Forcing a positive
        # lower bound on lin_vel_x AND zeroing rel_standing_envs makes
        # "stand still" strictly suboptimal — every env must move.
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # ---------------- Events ----------------
        # Pushes from iter 0 (Walk-These-Ways recipe). Pushes are a natural
        # perturbation against degenerate stationary solutions, just like
        # terrain. Mild ±0.4 m/s — child-shove magnitude.
        from isaaclab.envs import mdp as mdp_events
        self.events.push_robot = EventTerm(
            func=mdp_events.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={
                "velocity_range": {
                    "x": (-0.4, 0.4),
                    "y": (-0.4, 0.4),
                    "z": (-0.3, 0.3),
                    "roll": (-0.4, 0.4),
                    "pitch": (-0.4, 0.4),
                    "yaw": (-0.4, 0.4),
                },
            },
        )
        # Deterministic upright spawn (handled by init_state.rot in
        # deer_usd_articulation; zero out the parent's pose/vel jitter so we
        # don't fight that with random spawn rotations).
        rb = self.events.reset_base
        self.events.reset_base = EventTerm(
            func=rb.func,
            mode=rb.mode,
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
        # v4.12: RSI (Reference State Initialization) replaces the default
        # multiplicative-jitter joint reset. v4.7-v4.11 all showed the same
        # HR-knee structural local optimum: HR init at default -0.7 rad
        # (-40 deg, deeply bent), reference target -0.25 rad (-14 deg);
        # multiplicative jitter (0.85, 1.15) keeps init in [-0.84, -0.60] —
        # deer NEVER starts near reference. With RSI sampling random frames
        # from deer_walk.npy, the deer sometimes starts at reference HR
        # values (~-0.25 rad), giving the policy direct experience with
        # those poses. Pairs perfectly with the joint_track_reference_walk
        # reward — same data drives both init and target tracking.
        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints = None
        self.events.reset_base = EventTerm(
            func=reset_from_amp_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_height_m": 0.93,
                "pose_jitter": 0.05,   # ±0.05 rad jitter on joint angles
                "vel_jitter": 0.1,     # ±0.1 rad/s and ±0.1 m/s on velocities
            },
        )

        # ---------------- Rewards: clean legged_gym stack ----------------
        # Drop the parent's stock tracking rewards (they use Isaac's xy frame
        # which doesn't match our anatomical convention; replace with our own).
        for attr in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        # Anatomical tracking (body -Z = anatomical forward in our URDF).
        self.rewards.track_forward_vel = RewTerm(
            func=track_forward_vel_anatomical_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_lateral_vel = RewTerm(
            func=track_lateral_vel_anatomical_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_yaw_rate = RewTerm(
            func=track_yaw_rate_anatomical_exp,
            weight=0.75,
            params={"command_name": "base_velocity", "std": 0.5},
        )

        # feet_air_time: parent already wires the deer foot regex; just bump
        # weight + threshold to legged_gym recipe (0.5s air time per stride).
        if hasattr(self.rewards, "feet_air_time"):
            f = self.rewards.feet_air_time
            self.rewards.feet_air_time = RewTerm(
                func=f.func,
                weight=1.0,
                params={**({k: f.params[k] for k in f.params}), "threshold": 0.5},
            )

        # URDF v4 (Phase 2A): foot link split into FOOT (cannon/metacarpus,
        # SHOULD NOT touch ground) + TOE (phalanges/hoof, the actual ground
        # contact). All foot-contact rewards now read TOE bodies; cannon
        # contact is penalized via `undesired_contacts` (extended below).
        foot_names = ["HL_toe", "HR_toe", "FL_toe", "FR_toe"]   # actual hooves
        cannon_names = ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]  # NOT to touch
        self.rewards.feet_slide = RewTerm(
            func=feet_slide,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_names),
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_names),
            },
        )

        # feet_swing_clearance: reward each foot for being lifted to ~5 cm
        # above the ground during swing phase. Without this, `feet_air_time`
        # rewards "foot away from ground for >=0.5s" but ANY clearance
        # qualifies — including the rigid-front wheelbarrow gait where the
        # front foot lifts 1 mm. Walk-These-Ways uses ~5 cm target.
        # Per-foot exp reward, summed across 4 feet -> max ~+4 per step.
        self.rewards.feet_swing_clearance = RewTerm(
            func=feet_swing_clearance,
            weight=1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_names),
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_names),
                "target_height": 0.08,    # v4.2: was 0.05; raised to force higher swing arc → must bend knee to lift foot that high
                "contact_threshold": 5.0,
                "sigma": 0.05,
            },
        )

        # Override the parent's `feet_air_time` sensor to read TOE contacts
        # (not foot/cannon). Otherwise feet_air_time fires on cannon contact
        # and rewards the front-knee-walk. Same for the underlying contact
        # forces reading.
        if hasattr(self.rewards, "feet_air_time"):
            f = self.rewards.feet_air_time
            self.rewards.feet_air_time = RewTerm(
                func=f.func,
                weight=f.weight,
                params={
                    **{k: f.params[k] for k in f.params},
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_names),
                },
            )

        # ---------------- GAIT-TARGET CONDITIONING (Walk-These-Ways) ----------------
        # v4.18: CRITICAL FIX. Originally these offsets enforced a TROT
        # (diagonal pairs synced, FL+HR offset 0, FR+HL offset 0.5).
        # But the deer_walk.npy reference is actually a 4-beat lateral-
        # sequence WALK with measured offsets:
        #     FL: 0.000  HR: 0.677  FR: 0.516  HL: 0.161
        # (cross-correlation > 0.93 — clean signal). Since v4.8 wired in
        # joint_track_reference_walk, the env has been FIGHTING ITSELF:
        # gait_contact_match said trot, joint_track said walk. Policy
        # produced compromise gaits (parallel legs, drift) that satisfied
        # neither cleanly. This was the root of much pain.
        #
        # Walk offsets (FL, HR, FR, HL): rounded to phase fractions that
        # match the measured reference within ±0.02:
        #     FL=0.000, HR=0.677, FR=0.516, HL=0.161
        #
        # Foot order [FL, HR, FR, HL] is preserved.
        from training.envs.deer_gait import (
            gait_contact_match,
            gait_phase_obs,
            gait_shaped_force,
        )
        gait_foot_order = ["FL_toe", "HR_toe", "FR_toe", "HL_toe"]
        gait_sensor_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=gait_foot_order,
            preserve_order=True,
        )
        GAIT_FREQ = 2.0
        # v4.18: WALK pattern matching deer_walk.npy reference (was TROT (0,0,0.5,0.5))
        GAIT_OFFSETS = (0.0, 0.677, 0.516, 0.161)
        # v4.25: duty cycle 0.5 (trot) -> 0.65 (slow walk). Walks have
        # overlapping stance — each foot is on the ground for ~65-75% of
        # the cycle, not 50%. This was the last walk-vs-trot mismatch in
        # the gait_contact_match enforcement; matching it to walk physics
        # should let the policy commit to the lateral-sequence pattern.
        GAIT_DUTY = 0.65

        # v4.22: weight 2.0 -> 1.0. After v4.18 unified gait_contact_match
        # and joint_track on the walk pattern, the per-step rewards from
        # gait_contact_match (~4.80/ep) were dominant enough that the policy
        # satisfied them via foot contact timing without deeply committing
        # to joint_track's per-foot phase shape. Halving lets joint_track
        # (now at weight 2.0, ~9.8 ep_reward) dominate the gait-pattern
        # decision. Stage 3 reward stack v4.21 hit local optimum at HL/HR
        # hip correlation -0.10 vs ref -0.93 — need a different intervention.
        self.rewards.gait_contact_match = RewTerm(
            func=gait_contact_match,
            weight=1.0,
            params={
                "sensor_cfg": gait_sensor_cfg,
                "frequency": GAIT_FREQ,
                "foot_phase_offsets": GAIT_OFFSETS,
                "duty_cycle": GAIT_DUTY,
                "contact_threshold": 5.0,
            },
        )
        self.rewards.gait_shaped_force = RewTerm(
            func=gait_shaped_force,
            weight=0.5,
            params={
                "sensor_cfg": gait_sensor_cfg,
                "frequency": GAIT_FREQ,
                "foot_phase_offsets": GAIT_OFFSETS,
                "duty_cycle": GAIT_DUTY,
                "sigma": 100.0,
            },
        )

        # Tell the policy WHERE in the stride it is, via the observation
        # group. (cos, sin) per foot -> 8 floats. Without this the policy
        # would see the contact-match reward as random noise (no input
        # signaling the gait target).
        self.observations.policy.gait_phase = ObsTerm(
            func=gait_phase_obs,
            params={
                "frequency": GAIT_FREQ,
                "foot_phase_offsets": GAIT_OFFSETS,
            },
        )

        # Standing-pose deviation: gently regularize joints back toward the
        # default pose. Important for stillness when commanded zero.
        self.rewards.joint_pos_standing = RewTerm(
            func=joint_pos_standing_l2,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Smoothness penalties — re-weight the stock terms to legged_gym values.
        if hasattr(self.rewards, "lin_vel_z_l2"):
            self.rewards.lin_vel_z_l2.weight = -2.0
        if hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2.weight = -0.05
        if hasattr(self.rewards, "dof_torques_l2"):
            self.rewards.dof_torques_l2.weight = -2.5e-5
        if hasattr(self.rewards, "dof_acc_l2"):
            self.rewards.dof_acc_l2.weight = -2.5e-7
        if hasattr(self.rewards, "action_rate_l2"):
            # v4.2: -0.01 -> -0.005. The stronger penalty was punishing knee
            # variation, driving the policy to a hip-only stilt walk. Halving
            # the penalty lets knee bend cycles cost less reward.
            self.rewards.action_rate_l2.weight = -0.005

        # Orientation: dense cosine penalty (Lee/Margolis use -5 here).
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = -5.0

        # Base height: dense pull toward standing height.
        self.rewards.base_height = RewTerm(
            func=mdp_rewards.base_height_l2,
            weight=-20.0,
            params={"target_height": 0.85},
        )

        # Pelvis height ramp: positive reward for standing height (survival
        # incentive). Lowered weight 5.0 -> 3.0 in r4 because at +5.0 it
        # SATURATED in r3 (4.99/5.0) and the policy could max it without
        # walking — tucked-and-frozen poses with pelvis at 0.95m got the
        # full reward. Still keep it as a survival buffer but no longer
        # dominant.
        self.rewards.pelvis_height_ramp = RewTerm(
            func=pelvis_height_ramp,
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_names": ["pelvis"],
                "z_min": 0.40,
                "z_max": 0.85,
            },
        )

        # r7: head_above_base. r6's downward-dog pose collapses head/chest
        # but pelvis stays up. This term penalizes head-pelvis VERTICAL gap
        # falling below target_diff (0.5m). Negative L2 — head must stay
        # roughly that high above pelvis or pay reward.
        # r8: head_above_base weight reduced -3.0 -> -1.0. r7 reached upright
        # posture (head IS above base now, target_diff hits ≈0.5) so the
        # penalty's job is mostly done; large weight was crowding out the
        # forward_velocity_linear reward. Keep at -1.0 as a residual nudge.
        # v5.4: bumped -1.0 -> -3.0. The r8 reduction to -1.0 assumed the deer
        # was reliably upright, but a FRESH-from-scratch run (v5.3) fell back
        # into the r6 "downward-dog" collapse — pelvis held up + level while the
        # neck droops the skull to the ground and the front legs drag (a
        # hind-push slide). Neither termination catches it (pelvis stays above
        # 0.55m and under 60deg tilt), so it went undetected in metrics.
        # v7: ZEROED — replaced by neck_posture_track (imitating the reference's
        # actual head carriage) instead of a hand-set head-above-pelvis target.
        self.rewards.head_above_base = RewTerm(
            func=head_above_base,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "head_names": ["skull"],
                "base_names": ["pelvis"],
                "target_diff": 0.5,
            },
        )

        # v5.4: two purpose-built anti-collapse rewards that were sitting UNUSED
        # in deer_rewards.py. head_min_height_l2 penalizes absolute skull world-Z
        # below 0.9m (docstring: "Butt-in-air pose drops head to ~0.5m.
        # Penalising below 0.9m blocks that exploit"). leg_extension_reward
        # rewards the pelvis-to-foot standing gap (~0.89m) which a splayed/
        # dragging pose CANNOT fake (splayed gap ~0.15m -> reward ~0). Together
        # they make the downward-dog slide strictly worse than standing/walking.
        # v7: weight -8 -> -4. Kept as a sign-free BACKSTOP + arbiter for the new
        # neck_posture_track term (if the neck sign guess is wrong, this still
        # blocks a face-plant; watch it early to verify the neck sign).
        self.rewards.head_min_height = RewTerm(
            func=head_min_height_l2,
            weight=-4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "head_names": ["skull"],
                "min_height": 0.90,
            },
        )
        self.rewards.leg_extension = RewTerm(
            func=leg_extension_reward,
            weight=5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "foot_names": ["HL_foot", "HR_foot", "FL_foot", "FR_foot"],
                "body_names": ["pelvis"],
                "target_gap": 0.89,
            },
        )

        # v7: COMPLETE THE IMITATION. joint_track tracks only the 12 leg joints,
        # leaving the neck unconstrained -> head droops (collapse root cause).
        # This imitates the reference's upright head carriage directly, making
        # head-up part of the imitation rather than a penalty we fight for.
        self.rewards.neck_posture = RewTerm(
            func=neck_posture_track,
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                # signs FLIPPED vs the raw reference values: URDF neck convention
                # is opposite the reference clip's. FK-verified head-UP targets
                # (skull COM -0.448 -> -0.511, higher). Magnitudes = reference
                # carriage (neck raise 12.9deg, head 18.7deg).
                "joint_targets": (
                    ("neck_yaw_to_neck", -0.225),
                    ("neck_to_skull",    0.326),
                ),
                "sigma": 0.35,
            },
        )

        # v4.24: weight 10.0 -> 5.0. v4.20 showed the policy will trade
        # walk-pattern coordination for higher forward speed (1.04 m/s
        # at the cost of gait correlations). At weight 10, this reward
        # contributes ~4.5/ep which competes directly with joint_track
        # (~9.8/ep) — strong incentive to "go fast even if it breaks the
        # walk shape". Halving lets joint_track win the trade-off.
        self.rewards.forward_velocity_linear = RewTerm(
            func=forward_velocity_linear_anatomical,
            weight=5.0,
            params={"command_name": "base_velocity", "clip_max": 1.0},
        )

        # KEEP `undesired_contacts` ENABLED at full strength (legged_gym /
        # Walk-These-Ways recipe weight = -1.0). The parent already wires it
        # to a deer-anatomy contact regex (thigh|shin|...). This is THE
        # signal that prevents kneeling-on-calves: any leg segment touching
        # the ground costs reward every step. We previously zeroed it
        # ("noisy on deer geometry") and the policy converged to lying on
        # its front shins with the pelvis still up — pelvis_height_ramp by
        # itself is fakeable that way.
        if hasattr(self.rewards, "undesired_contacts"):
            # v4 (revised after warm-start regression): extend regex to
            # include FOOT (cannon/metacarpus). Cannon contact = the
            # front-knee-walk exploit. BUT keep weight modest so the
            # warm-start v3 policy isn't crushed for its existing solution
            # before it has a chance to discover toe-walking. v3-warm-start
            # at -2.5 caused policy collapse (ep_len 229 -> 52 over 250 iters)
            # — the deer learned to fall fast rather than lift cannons.
            # Lowered to -0.5 — a gradient pull toward toe-use without
            # making cannon contact strictly catastrophic.
            # v4.4 REVERT: -0.3 -> -0.5. v4.3 softened this to enable
            # deeper carpus folding, but the new carpus_flex term it
            # was paired with is now removed. Restoring -0.5 to keep
            # cannon-walk discouraged.
            cannon_names = ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]
            shin_names   = ["HL_shin", "HR_shin", "FL_shin", "FR_shin"]
            thigh_names  = ["HL_thigh", "HR_thigh", "FL_thigh", "FR_thigh"]
            self.rewards.undesired_contacts = RewTerm(
                func=self.rewards.undesired_contacts.func,
                weight=-0.5,
                params={
                    **{k: self.rewards.undesired_contacts.params[k] for k in self.rewards.undesired_contacts.params if k != "sensor_cfg"},
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces",
                        body_names=cannon_names + shin_names + thigh_names,
                    ),
                },
            )
        # Drop dof_pos_limits if present — joint limits already enforce this.
        if hasattr(self.rewards, "dof_pos_limits"):
            self.rewards.dof_pos_limits.weight = 0.0

        # v7: ZEROED. skull_pitch_l2 drove neck_to_skull toward 0 (level head),
        # but the reference walk carriage is neck_to_skull = -18.7deg. It was
        # actively FIGHTING the reference posture. neck_posture_track now imitates
        # the real carriage instead.
        self.rewards.skull_pitch_l2 = RewTerm(
            func=skull_pitch_l2,
            weight=0.0,
            params={"joint_name": "neck_to_skull"},
        )

        # v4.8 -> v4.9: replace the soft swing-phase bonuses with explicit
        # per-joint imitation of deer_walk.npy. v4.9 fixes a gait-pattern
        # conflict that crippled v4.8: the reference is a walk, the env
        # enforces a trot via gait_contact_match, and v4.8 indexed all 12
        # joints by ONE global phase — forcing the policy to satisfy walk-
        # cycle joint angles at trot-cycle stance/swing timing. The conflict
        # surfaced as one hind leg spasming mid-air while the other tracked
        # normally; HL/HR asymmetry blew out from 0.99 to 1.61 and yaw
        # drift worsened.
        #
        # Fix: per-foot phase indexing. Each foot looks up its joint targets
        # at ITS own gait phase (using foot_phase_offsets matching the
        # trot pattern). FL's reference trajectory is used for both FL and
        # FR (URDF mirrored axes preserve L/R angle equivalence); HL's for
        # both HL and HR. See deer_rewards.joint_track_reference_walk.
        #
        # v4.9 -> v4.10: weight bumped 0.5 -> 1.0. v4.9 fixed hind L/R
        # asymmetry (1.61 -> 0.98) but the asymmetry migrated to the front
        # legs (FL/FR ROM ratio 0.88 -> 1.35) and HR knee remained stuck at
        # -58 deg mean (vs reference -14). The HR-stuck pattern is consistent
        # across v4.7/v4.8/v4.9 — joint default is -0.7 rad which is already
        # deeper than reference target, and at sigma=0.3 the exp() reward is
        # vanishingly small for the policy at large errors (no gradient).
        # Doubling weight makes even small reward differences impactful.
        # Max contribution 12 * 1.0 = 12/step, comparable to or larger than
        # forward_velocity_linear (max 10/step capped).
        # v4.20 -> v4.21 REVERT: 3.0 -> 2.0. v4.20 weight 3.0 BACKFIRED:
        # FL/FR hip -0.58 -> -0.45 (regressed), straightness 0.323 -> 0.160
        # (halved), policy traded walk-coordination for faster forward speed
        # (1.04 m/s vs v4.19 0.67 m/s) earning more forward_velocity_linear
        # reward. Sweet spot is weight 2.0 (v4.19). Reverting and giving
        # the policy more training time from v4.19's state to see if it
        # continues converging or had already plateaued at v4.19's metrics.
        # v4.24: revert sigma 0.2 -> 0.3 (v4.23 regression — tighter sigma
        # made the policy give up rather than commit harder). Back to v4.22
        # baseline for this term.
        self.rewards.joint_track_reference_walk = RewTerm(
            func=joint_track_reference_walk,
            weight=2.0,
            params={
                "motion_path": "assets/motions/deer_walk.npy",
                "frequency": GAIT_FREQ,                # MUST match gait clock
                "foot_phase_offsets": GAIT_OFFSETS,    # MUST match gait pattern
                "sigma": 0.3,                          # ~17 deg per-joint tolerance
            },
        )

        # v4.8: zero out the deprecated soft swing-phase bonuses. The
        # tracker subsumes them — keeping them adds redundant gradients
        # that conflict with explicit per-joint targets. Kept as RewTerm
        # entries (weight=0) so any inherited config can re-enable them
        # for ablation.
        self.rewards.knee_flex_during_swing = RewTerm(
            func=knee_flex_during_swing,
            weight=0.0,   # v4.8: subsumed by joint_track_reference_walk
            params={
                "knee_joint_names": (
                    "FL_thigh_to_FL_shin", "HR_thigh_to_HR_shin",
                    "FR_thigh_to_FR_shin", "HL_thigh_to_HL_shin",
                ),
                "knee_default_pos": (0.5, -0.7, 0.5, -0.7),
                "frequency": GAIT_FREQ,
                "foot_phase_offsets": GAIT_OFFSETS,
                "duty_cycle": GAIT_DUTY,
                "flex_target": 0.4,
            },
        )
        self.rewards.carpus_flex_during_swing = RewTerm(
            func=knee_flex_during_swing,  # generic — works on any 4 joints
            weight=0.0,   # v4.8: subsumed by joint_track_reference_walk
            params={
                "knee_joint_names": (
                    "FL_shin_to_FL_foot", "HR_shin_to_HR_foot",
                    "FR_shin_to_FR_foot", "HL_shin_to_HL_foot",
                ),
                "knee_default_pos": (0.0, 0.0, 0.0, 0.0),
                "frequency": GAIT_FREQ,
                "foot_phase_offsets": GAIT_OFFSETS,
                "duty_cycle": GAIT_DUTY,
                "flex_target": 0.5,
            },
        )
        self.rewards.carpus_deep_flex_bonus = RewTerm(
            func=carpus_deep_flex_bonus,
            weight=0.0,   # v4.8: subsumed by joint_track_reference_walk
            params={
                "carpus_joint_names": (
                    "FL_shin_to_FL_foot", "HR_shin_to_HR_foot",
                    "FR_shin_to_FR_foot", "HL_shin_to_HL_foot",
                ),
                "frequency": GAIT_FREQ,
                "foot_phase_offsets": GAIT_OFFSETS,
                "duty_cycle": GAIT_DUTY,
                "deadband": 0.5,
                "scale": 0.6,
            },
        )

        # v4.6: bilateral joint-deviation symmetry penalty (per-step L2).
        # v4.5 used an EMA-based version which FAILED because PPO's value
        # function couldn't see the EMA state — only ~5% of the nominal
        # weight made it back as gradient. v4.6 uses a per-step Markov-clean
        # version: at each step, penalize squared difference of |q_L-default|
        # and |q_R-default| summed across 6 joint pairs. Sustained
        # asymmetric drive shows up as positive expected value over time;
        # natural phase-shifted trot has tracked magnitudes and is fine.
        # Bumped weight 0.5 -> 1.0 to make the signal more decisive.
        self.rewards.bilateral_symmetry = RewTerm(
            func=bilateral_joint_symmetry_l2,
            weight=-1.0,
            params={
                "joint_pairs": (
                    ("HL_thigh_to_HL_shin",   "HR_thigh_to_HR_shin"),    # hind knee
                    ("FL_thigh_to_FL_shin",   "FR_thigh_to_FR_shin"),    # front elbow
                    ("HL_shin_to_HL_foot",    "HR_shin_to_HR_foot"),     # hind hock
                    ("FL_shin_to_FL_foot",    "FR_shin_to_FR_foot"),     # front carpus
                    # v4.16: include hip ABDUCTION joints (these were unmapped in
                    # the reference, so the policy was free to abduct asymmetrically
                    # — likely cause of the "stepping sideways with one pair"
                    # behavior the user reported). Adding them to bilateral symmetry
                    # forces L/R hip abduction to mirror.
                    ("pelvis_to_HL_hipabd",   "pelvis_to_HR_hipabd"),    # hind hip abduction
                    ("chest_to_FL_hipabd",    "chest_to_FR_hipabd"),     # front shoulder abduction
                    ("HL_hipabd_to_HL_thigh", "HR_hipabd_to_HR_thigh"),  # hind hip flex
                    ("FL_hipabd_to_FL_thigh", "FR_hipabd_to_FR_thigh"),  # front shoulder flex
                ),
            },
        )

        # v4.28: anti-parallel hindquarter penalty. v4.27 analysis showed
        # FL/HR diagonal correlation +0.52 (ref -0.43) and HL/HR hip
        # correlation only -0.26 (ref -0.93) — same-side and same-time
        # legs moving in sync rather than mirrored. bilateral_symmetry
        # enforces |L-default|=|R-default| in MAGNITUDE but allows the
        # policy to satisfy that with L_dev = +R_dev (parallel) instead
        # of L_dev = -R_dev (mirror). This term directly penalizes
        # (L_dev + R_dev)^2 — zero for mirror, quadratic for parallel.
        #
        # v4.29: extended to KNEE pairs because v4.28 showed the policy
        # "bought" HL/HR hip mirror (-0.26 -> -0.34) by sacrificing knee
        # coordination (+0.34 -> +0.43, both legs flexing TOGETHER).
        # All 4 pairs now constrained → no escape valve.
        #
        # v4.29 crashed at iter ~320 with action_std NaN. anti_parallel
        # ep_reward had grown to -0.61 (3.5x v4.28). Halving weight to
        # -0.25 (v4.29r) avoided the crash but regressed FL/FR hip from
        # -0.61 to -0.24 — too soft. v4.30 splits the difference at
        # -0.35: strong enough to push hip mirror, weak enough to avoid
        # NaN. Watch action_std for instability.
        self.rewards.anti_parallel_thigh = RewTerm(
            func=anti_parallel_thigh_l2,
            weight=-0.35,
            params={
                "thigh_pairs": (
                    ("HL_hipabd_to_HL_thigh", "HR_hipabd_to_HR_thigh"),  # hind hip
                    ("FL_hipabd_to_FL_thigh", "FR_hipabd_to_FR_thigh"),  # front hip
                    ("HL_thigh_to_HL_shin",   "HR_thigh_to_HR_shin"),    # hind knee
                    ("FL_thigh_to_FL_shin",   "FR_thigh_to_FR_shin"),    # front knee
                ),
            },
        )

        # v4.4: bilateral straight-line anchoring via L2 tracking errors.
        # The existing track_yaw_rate_exp / track_lateral_vel_exp use
        # exp(-err^2/std^2) which saturates near 0 once err exceeds ~2*std.
        # In v4.3 the policy drifted ~13 deg/s at zero yaw command — the
        # exp term had no gradient left to pull it back. These L2 terms
        # have linear gradient in error magnitude, so the further off-track
        # the deer drifts, the harder they pull back. Sign-symmetric, so
        # they don't favor a left or right bias.
        from training.envs.deer_rewards import (
            track_lateral_vel_anatomical_l2,
            track_yaw_rate_anatomical_l2,
        )
        # v4.10 -> v4.11 REVERT: -3.0 -> -1.5.
        self.rewards.track_yaw_rate_l2 = RewTerm(
            func=track_yaw_rate_anatomical_l2,
            weight=-1.5,
            params={"command_name": "base_velocity"},
        )

        # v4.17: penalize hip abduction joint magnitudes to push them
        # toward 0 unless actively used for sidestep/turn. v4.16's
        # bilateral_symmetry extension forced L=R but the policy chose a
        # COMMON-MODE offset (both shoulders cocked same direction at
        # +5-22°) — symmetric in |L-R| but body leans → lateral drift →
        # "stepping sideways". This pulls each hipabd toward 0.
        # Weight -1.5 means at common +15° (=0.26 rad) on all 4 hipabd:
        # penalty = 4 * 0.067 * 1.5 = 0.4/step = 400/episode. Strong.
        # When commanded lateral motion is genuine, the deer can still
        # abduct asymmetrically; the cost is just the magnitude itself.
        self.rewards.hipabd_zero_l2 = RewTerm(
            func=hipabd_zero_l2,
            weight=-1.5,
            params={},
        )

        # v4.16: cumulative heading-drift penalty. track_yaw_rate_l2 is
        # per-step (penalizes instantaneous angular velocity error). The
        # deer can satisfy that while accumulating large heading errors
        # over the 1000-step episode (drift 5°/s -> 150° off course over
        # 30s). This term tracks each env's initial yaw on reset and
        # penalizes (cur_yaw - target_yaw)^2 where target_yaw includes
        # integrated commanded yaw rate. Direct anti-veer pressure.
        # Weight tuned so a 30° drift costs ~0.27 squared rad * weight.
        # v4.31 crashed NaN at iter ~88 when bumped to -2.5 from a
        # warm-start whose policy had 213° yaw drift baked in (v4.30):
        # huge instantaneous gradient. v4.31r: smaller bump to -2.0
        # AND warm-start from cleaner v4.29r (yaw 68°) instead.
        self.rewards.heading_hold = RewTerm(
            func=heading_hold_penalty,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )

        # v4.13: hard penalty for hind knees going outside the reference
        # walk's range [-0.55, 0.05] rad. The smooth tracking rewards
        # saturate; we need a HARD quadratic penalty that grows when the
        # joint goes BEYOND the ref envelope.
        self.rewards.hind_knee_range_penalty = RewTerm(
            func=hind_knee_range_penalty,
            weight=-5.0,
            params={
                "knee_joint_names": ("HL_thigh_to_HL_shin", "HR_thigh_to_HR_shin"),
                "range_min": -0.55,
                "range_max": 0.05,
            },
        )

        # v4.15: hard penalty for FRONT carpus joints going outside the
        # SIGN-CORRECTED reference range [+0.10, +1.30] rad. v4.14 fixed
        # the sign-convention bug for FL/FR_shin_to_FL/FR_foot (URDF
        # positive = fold direction, while bake convention had negative =
        # fold). After the fix the carpus CAN physically fold (reaches
        # +30 to +67 deg peaks during swing) but the policy spends 78-98%
        # of time clamped at the URDF lower limit -0.3 rad — the stiff
        # actuator (Kp=60) pulls back to extension and joint_track alone
        # can't overcome it. Same hard-penalty solution that worked for
        # the HR knee in v4.13.
        self.rewards.front_carpus_fold_penalty = RewTerm(
            func=front_carpus_fold_penalty,
            weight=-5.0,
            params={
                "carpus_joint_names": ("FL_shin_to_FL_foot", "FR_shin_to_FR_foot"),
                "range_min": 0.10,    # ~+6 deg, above the URDF clamp -0.3
                "range_max": 1.30,    # ~+75 deg, just below ref max +69 deg
            },
        )
        self.rewards.track_lateral_vel_l2 = RewTerm(
            func=track_lateral_vel_anatomical_l2,
            weight=-1.0,  # softer than yaw — lateral drift is less impactful visually
            params={"command_name": "base_velocity"},
        )

        # r7: Extend undesired-contact penalty to TORSO bodies (chest, neck,
        # skull). r6 deer found "downward-dog" pose — chest dropped to ground,
        # only hind legs trotting. The leg-only undesired_contacts regex
        # didn't fire on chest contact, so the policy was free to rest
        # the front body on the ground. With this term firing -2.0/step
        # whenever chest/neck/skull contact > 50 N (above self-collision
        # noise, below load-bearing chest contact ~350 N) the chest-drop
        # collapse becomes strictly worse than standing.
        from isaaclab.envs import mdp as _mdp
        self.rewards.torso_contact_penalty = RewTerm(
            func=_mdp.undesired_contacts,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["chest", "neck", "skull"],
                ),
                "threshold": 50.0,
            },
        )

        # ---------------- Terminations ----------------
        # Disable base_contact termination — tried with chest/neck/skull body
        # names at thresholds 1, 200; both fired 100% at spawn, indicating the
        # deer's enabled_self_collisions setting produces high inter-mesh
        # contact forces between chest and front-leg thighs that aren't real
        # ground contacts. We rely on the `undesired_contacts` REWARD penalty
        # (-1.0 weight on thigh/shin) — same signal but graded, doesn't kill
        # episodes, and matches Walk-These-Ways' default
        # (terminate_after_contacts_on = [] empty, collision = -1.0).
        if hasattr(self.terminations, "base_contact"):
            self.terminations.base_contact = None
        # Relaxed to 0.30m for Stage 1 (was 0.55m) — gives the policy room to
        # explore lower poses without instant termination, so the survival
        # gradient via pelvis_height_ramp can build before the policy can stand.
        # Stage 2/3 can re-tighten to 0.55m once standing is solid.
        self.terminations.pelvis_too_low = DoneTerm(
            func=pelvis_too_low,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "min_height": 0.30,
            },
        )
        self.terminations.base_tilted = DoneTerm(
            func=base_orientation_too_tilted,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_angle_deg": 60.0},
        )


@configclass
class DeerUsdLocoStage2Cfg(DeerUsdLocoStage1Cfg):
    """Stage 2: full 3-DOF locomotion + push events for recovery.

    Inherits everything from Stage 1, then:
      - Widens command ranges to lateral + yaw.
      - Adds push_robot interval events (mild ±0.4 m/s).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from isaaclab.envs import mdp as mdp_events

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (-3.14, 3.14)

        self.events.push_robot = EventTerm(
            func=mdp_events.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={
                "velocity_range": {
                    "x": (-0.4, 0.4),
                    "y": (-0.4, 0.4),
                    "z": (-0.3, 0.3),
                    "roll": (-0.4, 0.4),
                    "pitch": (-0.4, 0.4),
                    "yaw": (-0.4, 0.4),
                },
            },
        )


@configclass
class DeerUsdLocoStage3Cfg(DeerUsdLocoStage2Cfg):
    """Stage 3: rough heightfield terrain on top of Stage 2.

    Mild ±0.5–4 cm bumps at 40 cm grid spacing. The pelvis_too_low
    termination already accounts for terrain elevation per env_origin
    (fixed in deer_rewards.py during the AMP-era r42 work), so it stays
    tight even with terrain.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        import isaaclab.terrains as terrain_gen
        from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

        rough = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=10.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.4,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "mild": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(0.005, 0.04),
                    noise_step=0.005,
                    border_width=0.5,
                ),
            },
        )
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = rough


@configclass
class DeerUsdLocoStableCfg(DeerUsdLocoStage3Cfg):
    """v5.0 — STABLE-BASE pivot (2026-05-14).

    Strategic reframe: after v4.7-v4.33 (26 reward-shaping iterations chasing
    natural gait), the policy still porpoises ~48 cm/stride, pitches nose-down
    ~10 deg, and weaves thousands of degrees. The research literature is
    unanimous that hand-designed reward RL is a known dead end for gait
    NATURALNESS, and the user's actual complaints ("uphill", "swerving") are
    STABILITY problems, not coordination problems.

    Root-cause insight: the naturalness reward terms (joint tracking, bilateral
    symmetry, anti-parallel, carpus fold, gait-phase matching) fight each other
    AND fight stability — they are what destabilize the deer. Stripping them
    back to the standard legged-robot locomotion stack (velocity tracking +
    upright + height + anti-bounce + anti-weave + smoothness + don't-fall)
    should yield a robotic-but-SOLID walk: no bounce, no weave, goes where
    commanded, doesn't fall. That is the right foundation for the ecosystem
    MVP. Naturalness is parked for a later AMP pass (now unblocked — the r15
    axis bug and playback are fixed).

    This class ONLY zeros naturalness weights + strengthens anti-porpoise; it
    inherits Stage3's stable terms unchanged. Stage3 (naturalness) stays fully
    intact and recoverable. Best naturalness checkpoint preserved at
    logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_23-39-02/model_30396.pt.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # ---- Zero ONLY the cosmetic-naturalness reward terms ----
        # v5.1 revision: the v5.0 total-strip collapsed (base_tilted -> 99.8%,
        # deer fell every episode). Root cause: (1) removing gait_contact_match
        # + gait_shaped_force also removed the STEPPING-RHYTHM scaffolding that
        # keeps a quadruped upright (rhythmic contact timing is a stability
        # feature, not just a naturalness one), and (2) lin_vel_z_l2 = -4.0
        # over-penalized the vertical motion that stepping requires. Too big a
        # discontinuity for a warm-start.
        #
        # So strip ONLY the cosmetic terms (pose/coordination realism) and KEEP
        # the gait-timing rewards for stepping rhythm + stability.
        _naturalness_terms = (
            "joint_track_reference_walk",   # per-joint imitation of reference walk
            "bilateral_symmetry",           # L/R magnitude-symmetry penalty
            "anti_parallel_thigh",          # sign-aware mirror penalty (v4.28)
            "hind_knee_range_penalty",      # forces hind knee into ref range
            "front_carpus_fold_penalty",    # forces front carpus to fold
            "knee_flex_during_swing",       # already 0, keep explicit
            "carpus_flex_during_swing",     # already 0
            "carpus_deep_flex_bonus",       # already 0
        )
        for name in _naturalness_terms:
            if hasattr(self.rewards, name):
                getattr(self.rewards, name).weight = 0.0

        # KEPT for stepping rhythm / stability (NOT zeroed):
        #   gait_contact_match, gait_shaped_force, feet_swing_clearance,
        #   feet_air_time, feet_slide.

        # ---- Mild anti-porpoise ----
        # Stage3 has lin_vel_z_l2 at -2.0. Nudge to -2.5 (NOT -4.0 — that
        # fought stepping and helped collapse v5.0). With the cosmetic terms
        # gone but stepping rhythm retained, this should trim the bounce
        # without destabilizing.
        if hasattr(self.rewards, "lin_vel_z_l2"):
            self.rewards.lin_vel_z_l2.weight = -2.5

        # ---- Keep everything else as-is (the stable/control stack) ----
        # Retained from Stage3: track_forward_vel, track_lateral_vel,
        # track_yaw_rate, forward_velocity_linear, base_height,
        # pelvis_height_ramp, head_above_base, heading_hold, feet_slide,
        # feet_swing_clearance, feet_air_time, joint_pos_standing,
        # torso_contact_penalty, undesired_contacts, hipabd_zero_l2,
        # track_yaw_rate_l2, track_lateral_vel_l2, skull_pitch_l2,
        # flat_orientation_l2, ang_vel_xy_l2, dof_torques_l2, dof_acc_l2,
        # action_rate_l2, and the fall terminations.


def _apply_stable_reward_profile(rewards, lin_vel_z_weight: float = -2.5) -> None:
    """v5.x stable-base reward profile (shared).

    Zeros the cosmetic-naturalness terms (pose/coordination realism), KEEPS the
    gait-timing rewards (stepping rhythm = stability), and applies a mild
    anti-porpoise weight. Same profile as DeerUsdLocoStableCfg, factored out so
    the flat/from-scratch variant reuses it.
    """
    _cosmetic_naturalness = (
        "joint_track_reference_walk",
        "bilateral_symmetry",
        "anti_parallel_thigh",
        "hind_knee_range_penalty",
        "front_carpus_fold_penalty",
        "knee_flex_during_swing",
        "carpus_flex_during_swing",
        "carpus_deep_flex_bonus",
    )
    for _name in _cosmetic_naturalness:
        if hasattr(rewards, _name):
            getattr(rewards, _name).weight = 0.0
    if hasattr(rewards, "lin_vel_z_l2"):
        rewards.lin_vel_z_l2.weight = lin_vel_z_weight


@configclass
class DeerUsdLocoStableFlatCfg(DeerUsdLocoStage1Cfg):
    """v5.2 — STABLE base, FRESH from scratch on FLAT ground.

    Why this exists: warm-starting the stable reward from the naturalness
    policy (v4.31r) proved unstable — v5.0 collapsed outright, and v5.1 drifted
    (base_tilted 15% -> 55% as it trained away from the propped-up naturalness
    gait). The naturalness gait depended on the very rewards we strip, so
    warm-starting inherits a body that falls apart without them.

    Fix: learn the stable walk FROM SCRATCH under the stable reward, on the
    EASIEST setting — Stage1's flat ground + forward-only command, no pushes.
    Stage1 already reliably discovers walking from scratch WITH the naturalness
    rewards; stripping them (fewer competing objectives) makes it easier, not
    harder. This yields a clean, un-entangled stable base we can later
    curriculum up to terrain (Stage2/3).

    Launch FRESH (no --resume / no warm-start).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_stable_reward_profile(self.rewards)
