"""AMP-augmented walk environment for the Tier-3 (Poly Art) deer — 38 DOF.

Built FRESH on Isaac Lab's stock ``LocomotionVelocityRoughEnvCfg`` scaffolding,
reconfigured for the deer with clean, purpose-built manager cfgs (no inheritance
of the old ``deer_usd_amp_env_cfg`` accumulation). Phase-1 scope: walk forward on
flat ground, naturalness supplied by the AMP discriminator.

Why this is much simpler than the old rig's env:
  * The USD imports **+X = forward, +Z = up** (R_ISAAC baked in at build time),
    so Isaac Lab's **stock** ``track_lin_vel_xy_exp`` / ``track_ang_vel_z_exp``
    work directly — none of the old rig's anatomy-aligned reward shims.
  * Spawns upright at joint-zero, so no ``Rx(-90)`` reset gymnastics.

Two observation groups:
  * ``policy`` — proprioception + velocity command (what the actor sees).
  * ``amp``    — clean ``joint_pos(38) + joint_vel(38) + base_lin_vel(3) +
    base_ang_vel(3)`` = **82**, in REFERENCE_JOINT_ORDER with preserve_order,
    matching ``deer_polyart_walk.npy`` column-for-column. amp-rsl-rl reads it via
    ``env.cfg.observations.amp.joint_pos.params['asset_cfg'].joint_names``.

Root link is **Pelvis** (not ``base``); the fall termination is orientation-
based (``bad_orientation`` 60°), which CLAUDE.md found more robust than height/
contact for this animal.

Phase-1 locked joints (Jaw/Ears/Tail) are simply excluded from the action space
(ACTIVE_JOINTS = 34); their actuators hold them at default. They remain in the
AMP obs (the reference has all 38) — being near-constant in the walk clip, they
cost no style penalty.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from training.polyart.articulation import (
    deer_polyart_articulation_cfg,
    ACTIVE_JOINTS,
    REFERENCE_JOINT_ORDER,
    FEET_BODIES,
    NON_HOOF_LEG_BODIES,
)
from training.polyart.rsi import reset_from_reference
from training.polyart.observations import hoof_heights as amp_hoof_heights
from training.polyart.observations import hoof_planar_speed as amp_hoof_speed
from training.polyart.ase import observe_latent as ase_observe_latent
from training.polyart.ase import resample_latent as ase_resample_latent
from training.polyart.rewards import (feet_air_time_positive, forward_velocity_linear,
                                      head_too_low, joint_outside_reference_range)


# OVERRIDABLE (2026-07-26). `joint_pos_limits` and `joint_ref_range` are the ONLY
# reward difference between r10 -- the last run that learned to stand FROM SCRATCH --
# and r14/r15, which never did. Both are penalties a random policy cannot escape: a
# flailing deer is always outside the reference walk's range and always near its
# limits, so from step one it pays a tax with no reachable escape. rsl_rl does not
# bootstrap value on failure termination, so a dead episode is worth 0 and falling
# over immediately beats accumulating the tax -- measured net reward per step went
# -0.0017 (r10) -> -0.0094 (r15), episode length pinned flat at ~18 of 400 steps.
# Warm-started this never bites: r12/r13's policy is already inside the range, so the
# term costs ~0 and only nudges. Exposed as env vars so from-scratch runs can anneal
# them in rather than pay them throughout exploration.
# NOTE: these MUST live at module scope -- @configclass treats every class attribute
# as a reward term and rejects a bare float with
#   TypeError: Configuration for the term '_JPL_W' is not of type RewardTermCfg.
_JPL_W = float(os.environ.get("ANIML_DEER_W_JOINT_POS_LIMITS", "-2.0"))
_JRR_W = float(os.environ.get("ANIML_DEER_W_JOINT_REF_RANGE", "-1.0"))
_FAT_W = float(os.environ.get("ANIML_DEER_W_FEET_AIR_TIME", "1.0"))

# EXPERIMENT HARNESS (2026-07-27). Weights for terms the AMP discriminator can ALREADY
# SEE, so they duplicate the style signal in a cruder language. AMP obs is
# joint_pos(38)+joint_vel(38)+base_lin_vel(3)+base_ang_vel(3): vertical velocity and
# roll/pitch rate are both in there. Every proxy that has caused a pathology this week
# was one the discriminator could already see -- joint_ref_range (joint angles)
# suppressed leg activity and killed two legs; feet_air_time (footfalls follow from
# joint trajectories) first punished short steps, then at 20x rewarded a 22%-flight
# HOP, because all four feet airborne at once is the highest-paying thing per second.
# Kept as env vars rather than deleted so the "style does the work" hypothesis is a
# config change, not a code change.
# NOT included here: base_height and non_hoof_contact. AMP is genuinely BLIND to
# pelvis height and to contact state, so those are blind-spot patches, not duplicates.
_LVZ_W = float(os.environ.get("ANIML_DEER_W_LIN_VEL_Z", "-2.0"))
_AVXY_W = float(os.environ.get("ANIML_DEER_W_ANG_VEL_XY", "-0.05"))
_HH_W = float(os.environ.get("ANIML_DEER_W_HEAD_HEIGHT", "-5.0"))
# Yaw command range. Must be widened BEFORE the turning walk takes (DWalk R / DWalk L)
# enter the expert set, or they are motion the command never asks for -- the r9
# turn-contamination failure.
_CMD_YAW = float(os.environ.get("ANIML_DEER_CMD_ANG_VEL_Z", "0.0"))

# ASE stage 1 (2026-07-27). ANIML_DEER_ASE_DIM > 0 conditions the actor on a random
# unit latent z, resampled per episode. Opt-in so it cannot affect the AMP experiments.
# The MI reward and encoder training are patched on in train_amp.py; here we only give
# the actor sight of z and resample it on reset. z MUST be in the actor's observation
# at construction time, or the network has no input to condition on.
_ASE_DIM = int(os.environ.get("ANIML_DEER_ASE_DIM", "0"))

# B (2026-07-28): add per-hoof HEIGHT to the AMP observation, closing the
# discriminator's blind spot to ground relationship. ANIML_DEER_AMP_HOOF_H=1 enables.
# Changes the AMP obs width 82 -> 86, so every existing checkpoint is incompatible --
# this is a from-scratch-only change, and train_amp.py must patch the DATASET side to
# match or AMPLoader and the env disagree on the observation and the run is garbage.
_AMP_HOOF_H = os.environ.get("ANIML_DEER_AMP_HOOF_H", "0") == "1"

# G (2026-07-29): add per-hoof horizontal SPEED. Measured after height was shown NOT to
# separate a supporting leg from a passenger -- A2's two front hooves reach the same
# minimum height while carrying 0.02 and 0.32 of the load. Slip does separate them:
# normalised by body speed the reference plants every hoof at 0.04-0.44 across walk,
# trot and run, while our policies skim the FRONT hooves at 0.7-1.55, and the slip
# tracks the missing load inversely. See docs/LEG_NEGLECT_findings.md.
# Composes with HOOF_H, so the AMP obs is 82 / 86 / 86 / 90.
_AMP_HOOF_V = os.environ.get("ANIML_DEER_AMP_HOOF_V", "0") == "1"

# Commanded forward-speed range. Measured gait speeds in the clip library:
#   walk 1.14 m/s | trot 2.99 m/s | run 7.57 m/s
# These are cleanly separated, which is what makes VELOCITY the implicit conditioning
# variable for a multi-gait dataset: base_lin_vel is part of the AMP observation, so a
# command of 1.1 puts the deer where only walk frames live and 3.0 where only trot
# frames do. That is the same mechanism ASE uses, with a hand-specified 3-number dial
# instead of a learned latent -- and its limit is that it can only separate behaviours
# that DIFFER IN SPEED.
_CMD_VX_LO = float(os.environ.get("ANIML_DEER_CMD_VX_LO", "0.8"))
_CMD_VX_HI = float(os.environ.get("ANIML_DEER_CMD_VX_HI", "1.2"))


# ---------------------------------------------------------------------------
# USD path
# ---------------------------------------------------------------------------
def _resolve_usd() -> str:
    raw = os.environ.get("ANIML_DEER_USD", "").strip()
    if raw and Path(raw).is_file():
        return str(Path(raw).resolve())
    repo = Path(__file__).resolve().parents[2]
    return str((repo / "assets" / "deer_polyart_urdf" / "deer_polyart.usd").resolve())


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@configclass
class AMPObservationsCfg(ObsGroup):
    """The clean state the AMP discriminator compares against the reference.

    Term ORDER is load-bearing: amp-rsl-rl builds the reference obs as
    (joint_pos, joint_vel, base_lin_vel, base_ang_vel) — keep this order.
    ``joint_pos`` is ABSOLUTE (matches the reference's absolute joint angles),
    not ``joint_pos_rel``. ``preserve_order=True`` keeps our columns aligned to
    the dataset's — without it SceneEntityCfg silently reorders to the sim's
    internal joint order and the discriminator learns from shuffled garbage.
    """

    joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=REFERENCE_JOINT_ORDER, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=REFERENCE_JOINT_ORDER, preserve_order=True)},
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    # THESE TWO MUST STAY LAST AND IN THIS ORDER: the dataset-side patch appends
    # heights then speeds after the four stock blocks, and the two orders have to agree
    # column-for-column. Swapping them compares a height against a speed -- dimensional
    # nonsense that trains perfectly happily and produces a quietly meaningless run.
    hoof_height = ObsTerm(
        func=amp_hoof_heights,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES,
                                            preserve_order=True)},
    )
    hoof_speed = ObsTerm(
        func=amp_hoof_speed,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES,
                                            preserve_order=True)},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False   # discriminator sees clean state
        self.concatenate_terms = True
        if not _AMP_HOOF_H:
            self.hoof_height = None
        if not _AMP_HOOF_V:
            self.hoof_speed = None


@configclass
class DeerPolyArtObservationsCfg:
    """policy (actor proprioception) + amp (discriminator) groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # ASE latent — disabled (set to None) unless ANIML_DEER_ASE_DIM > 0.
        ase_latent = ObsTerm(func=ase_observe_latent, params={"dim": max(_ASE_DIM, 1)})
        # NOTE: no height_scan term — flat ground, no height scanner.

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    amp: AMPObservationsCfg = AMPObservationsCfg()


# ---------------------------------------------------------------------------
# Commands — phase 1: forward walk only
# ---------------------------------------------------------------------------
@configclass
class DeerPolyArtCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # r8: (0.5,1.0) -> (0.8,1.2). The repaired reference walks at a
            # measured 1.05 m/s (median frame speed 1.118). Commanding slower
            # than the style target put the task and style rewards in tension.
            lin_vel_x=(_CMD_VX_LO, _CMD_VX_HI),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-_CMD_YAW, _CMD_YAW),
            heading=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Actions — 34 active joints (cosmetics locked), full-range scale
# ---------------------------------------------------------------------------
@configclass
class DeerPolyArtActionsCfg:
    # scale=1.0 (not the stock 0.5): expert gaits need ~60° knee bends, which
    # ±0.5 rad caps off (CLAUDE.md). Cosmetics (Jaw/Ears/Tail) are absent from
    # joint_names, so the action manager never targets them — their actuators
    # hold them at default (phase-1 lock).
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ACTIVE_JOINTS, scale=1.0, use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Rewards — lean; AMP supplies "natural". Stock velocity tracking works (X-fwd).
# ---------------------------------------------------------------------------

# probe-L (2026-07-26) isolated the pair above: restoring joint_pos_limits to -2.0 on
# top of probe-N changed nothing (ep_len 59.8 vs 60.3 at iter 270), so the hard
# end-stop penalty is cold-start-SAFE and stays on. joint_ref_range is the sole
# reward-side blocker -- it binds joints to the WALK CLIP's range, which a flailing
# policy can never satisfy.


@configclass
class DeerPolyArtRewardsCfg:
    # -- task
    # LINEAR forward-velocity reward: constant "go" gradient the exp term lacks,
    # to break the stand-still local optimum (see rewards.py). This is the phase-1
    # change after the shakeout showed reward rising while locomotion fell.
    # r8: 5.0 -> 2.0. This was inflated (2.5->4.0->5.0) to drag the deer forward
    # against a style reward that — on the corrupt clip — was actively teaching it
    # to STAND STILL (59.8% of expert frames had ~0 root velocity). With the
    # repaired clip the discriminator supplies the "go" itself, so this only needs
    # to be a gentle gradient, not a counterweight.
    forward_velocity = RewTerm(
        func=forward_velocity_linear, weight=2.0,
        params={"command_name": "base_velocity"},
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # -- penalties (keep small; naturalness is the discriminator's job)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=_LVZ_W)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=_AVXY_W)
    # r5 anti-crouch, r8 RECALIBRATED: target 0.85 -> 0.94.
    # 0.85 was BELOW the reference's own minimum pelvis height (measured: mean
    # 0.946, min 0.890) — we were rewarding a permanent slight crouch. The term
    # works: at r7 the policy held the pelvis within 5 mm of 0.85, so it tracks
    # this target closely. Pointing it at the true reference height should lift
    # the body ~9 cm on its own.
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-20.0,
        params={"target_height": 0.94, "asset_cfg": SceneEntityCfg("robot")},
    )
    # r6 anti-head-down, r8 RECALIBRATED: target 1.0 -> 1.05, weight -8 -> -5.
    # Reference head height is mean 1.180, min 1.020. Set as a FLOOR just above
    # the reference minimum so it catches genuine head-droop without fighting the
    # natural head bob (a target at the 1.18 mean would penalize half the real
    # walk). With clean style data the discriminator now shapes carriage, so this
    # is a safety net rather than the primary driver — hence the lower weight.
    head_height = RewTerm(
        func=head_too_low, weight=_HH_W,
        params={"target_height": 1.05, "asset_cfg": SceneEntityCfg("robot", body_names="Head")},
    )
    # r7 anti-crumple: the render + joint data showed the deer folds its legs to
    # their limits and props on the shins/cannon bones/"knees", not its hooves.
    # Penalize ground contact on any leg segment EXCEPT the hooves so only the
    # feet may bear weight.
    # r8: -2.0 -> -0.75. At r7 this was the LARGEST single reward term by far
    # (-1.28 episodic, vs +0.50 for forward_velocity and ~-1.1 for all task terms
    # combined) — it existed to stop the deer propping on its shins, which was
    # itself downstream of the corrupt style target. A penalty that large on any
    # non-hoof ground contact also punishes the failed steps a policy must make
    # while LEARNING to step, so it suppresses exploration. Kept as a guardrail,
    # no longer a dominant force.
    non_hoof_contact = RewTerm(
        func=mdp.undesired_contacts, weight=-0.75,
        # r12: threshold 1.0 -> 50.0 N. The deer weighs ~809 N, so 1 N (~0.1 kgf, a
        # small apple) fired on the faintest graze -- taxing exactly the compliant leg
        # motion we want to KEEP, while not specifically targeting knee-walking. A limb
        # genuinely PROPPING carries 160-240 N. 50 N cleanly separates "bearing load"
        # from "brushed the ground". This was deliberately NOT fixed earlier: until the
        # self-collision fix, this term was partly reading 100+ kN of phantom TORSO
        # contact, so tuning its threshold would have been tuning a broken measurement.
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=NON_HOOF_LEG_BODIES),
                "threshold": 50.0},
    )
    # r7 stepping: reward hoof air-time (when commanded to move) so the deer
    # lifts and places its feet — a real step cycle — instead of a static prop-up.
    # r10: threshold 0.3 -> 0.2. This term pays `air_time - threshold` on each
    # touchdown, so it only rewards anything once a foot stays airborne LONGER than
    # the threshold. Through r7-r9 it measured persistently NEGATIVE (-0.018), i.e.
    # it was a pure penalty and never once rewarded a step -- the single term meant
    # to encourage stepping was structurally unable to. (It was also the clearest
    # early warning that no gait was forming, and I under-weighted it.) The
    # reference stride is 1.10 s, so a real swing is ~0.35-0.4 s; 0.2 s is reachable
    # from a standing start and still requires a genuine lift, giving a usable
    # gradient instead of a flat penalty.
    # r12: 0.2 -> 0.12, now MEASURED rather than guessed. r10's actual swing durations
    # (PhysX contacts) were median 0.160 s against the reference walk's ~0.35-0.40 s, so
    # at 0.2 s only 35.7% of swings paid out and the term stayed net-negative (-0.016)
    # exactly as in r7-r9. At 0.12 s ~55% pay out, so it finally acts as a gradient.
    # Raise it back toward the reference as swings lengthen.
    # r16: weight is env-overridable. At weight 1.0 this term paid +0.003 per episode
    # against forward_velocity's +1.99 -- 1/700th. Stepping was therefore worth almost
    # nothing, and the measured consequence is that two of four legs stopped bearing
    # load entirely (r12/r13 duty 0.03-0.07) while the diagonal pair did all the work:
    # idling is free. Raising it is the r16 gait experiment. Overshoot buys r10's
    # failure instead (duty too LOW, liftoffs too HIGH -- a prance), so read the
    # duty/liftoff numbers from gait_report rather than pushing the weight blindly.
    feet_air_time = RewTerm(
        func=feet_air_time_positive, weight=_FAT_W,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
                "command_name": "base_velocity", "threshold": 0.12},
    )
    # r11: NEW. r10's policy RODE ITS JOINT LIMITS -- 95% of the left hoof-ankle's
    # samples were outside the range the reference walk ever uses, and its measured
    # range [-112.8, +57.3] was essentially the limit range [-112.7, +55.8] rather than
    # the walk's [-65.6, +16.8]. The same held on nearly every leg joint. That is what
    # made the ankles look wrong in the render. Two causes: limits fitted to the union
    # of walk+run+jump (running needs far more ankle travel than walking) and NOTHING
    # penalising extreme joint use, since dof_torques_l2/dof_acc_l2 have been zeroed
    # since r3 and action_rate is tiny. Limits are now refitted to DWalk+15 deg; this
    # term adds the missing pressure so the policy does not simply ride the new ones.
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=_JPL_W)
    # r12: the PRIMARY shaping term. Limits are back to ANATOMICAL (the union of
    # all round-trip-validated clips), so the skeleton no longer encodes 'walk' --
    # a future run/jump policy uses the same rig. What is appropriate for THIS task
    # is expressed here instead: stay inside the range the reference walk uses,
    # with ~5 deg of free play. Re-point its motion_path at a run clip and it
    # re-derives, no rig rebuild.
    joint_ref_range = RewTerm(func=joint_outside_reference_range, weight=_JRR_W)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)  # r3: -0.01->-0.005
    # r3: ZEROED. At r2 iter ~800 these "tiny" penalties dominated the reward
    # (dof_torques_l2 = -4.1, action_rate = -1.2) vs forward_velocity +0.25 —
    # the exact overflow CLAUDE.md warns about (torques huge on an 80 kg animal
    # with noisy actions), shackling the policy at a slow-walk plateau. AMP
    # style supplies the smoothness these were meant to.
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    # NOTE: no feet_air_time / undesired_contacts / flat_orientation here —
    # those need per-rig body names and AMP handles gait style. FEET_BODIES are
    # ready in articulation.py if an anti-shuffle term is needed later.


# ---------------------------------------------------------------------------
# Terminations — timeout + orientation fall (root = Pelvis)
# ---------------------------------------------------------------------------
@configclass
class DeerPolyArtTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    # r5 anti-crouch: the 60° tilt check misses a CROUCH (body stays roughly
    # level, just low). The render showed r4 learned a head-down crouch-shuffle
    # that evades fell_over while barely translating. Terminate if the pelvis
    # drops below 0.6 m (a normal walk keeps it ~0.85+; RSI spawns ~0.9).
    pelvis_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Events — phase 1: friction randomization + gentle standing reset
# ---------------------------------------------------------------------------
@configclass
class DeerPolyArtEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # Reference-state initialization: each reset spawns the deer mid-stride from
    # a random valid frame of deer_polyart_walk.npy (natural joint pose + root
    # height/orientation/velocity). One combined event writes joint AND root
    # state, so no separate reset_root/reset_joints terms. See training/polyart/rsi.py.
    reset = EventTerm(
        func=reset_from_reference,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # NOTE: no base-mass / COM / external-force / push events in phase 1 — those
    # are phase-2 domain randomization (they also assumed a "base" link).


# ---------------------------------------------------------------------------
# Curriculum — none on flat ground
# ---------------------------------------------------------------------------
@configclass
class DeerPolyArtCurriculumCfg:
    pass


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@configclass
class DeerPolyArtWalkAMPEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat-ground AMP walk env for the 38-DOF Poly Art deer."""

    observations: DeerPolyArtObservationsCfg = DeerPolyArtObservationsCfg()
    actions: DeerPolyArtActionsCfg = DeerPolyArtActionsCfg()
    commands: DeerPolyArtCommandsCfg = DeerPolyArtCommandsCfg()
    rewards: DeerPolyArtRewardsCfg = DeerPolyArtRewardsCfg()
    terminations: DeerPolyArtTerminationsCfg = DeerPolyArtTerminationsCfg()
    events: DeerPolyArtEventCfg = DeerPolyArtEventCfg()
    curriculum: DeerPolyArtCurriculumCfg = DeerPolyArtCurriculumCfg()

    def __post_init__(self) -> None:
        # --- scene overrides (before super: super reads terrain/curriculum) ---
        self.scene.num_envs = int(os.environ.get("ANIML_DEER_NUM_ENVS", "2048"))
        self.scene.env_spacing = 2.5
        # robot
        self.scene.robot = deer_polyart_articulation_cfg(_resolve_usd())
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        # flat ground — drop the rough generator and the height scanner
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        super().__post_init__()

        # --- ASE stage 1 wiring (opt-in via ANIML_DEER_ASE_DIM) --------------
        # The latent obs term is declared unconditionally so the configclass stays
        # static; disable it by setting the term to None when ASE is off, which is how
        # Isaac Lab skips a term. When ON, also register the per-episode resample as a
        # reset event -- without it z is constant for the whole run and the encoder's
        # task becomes trivial (one z, one behaviour), which defeats the point.
        if _ASE_DIM > 0:
            self.observations.policy.ase_latent.params["dim"] = _ASE_DIM
            self.events.ase_resample = EventTerm(
                func=ase_resample_latent, mode="reset", params={"dim": _ASE_DIM},
            )
            print(f"[ase] stage-1 ON: latent dim {_ASE_DIM}, resampled per episode")
        else:
            self.observations.policy.ase_latent = None

        # --- post-super general settings ---
        self.decimation = 4
        # r2: 20s -> 8s. At 20s the policy learned to "survive by standing"
        # (falls dropping, ep_len -> 756, but error_vel RISING to 1.1 = drifting
        # backward), banking style reward over long standing episodes. Short
        # episodes + RSI (spawns mid-stride at ~0.7 m/s) keep the deer in the
        # moving phase and deny the standing-accumulation payoff.
        self.episode_length_s = 8.0
        self.sim.dt = 0.005   # 200 Hz physics, 50 Hz control
