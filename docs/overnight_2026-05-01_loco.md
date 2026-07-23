# Overnight 2026-05-01 — URDF v2 + Pure Reward-Shaping Locomotion

## TL;DR

Migrated off AMP (the reference-data ceiling we'd been hitting for 46 runs) to the
proven Lee/Margolis/Hwangbo reward-shaping recipe. Simultaneously upgraded the
deer URDF from 15 → 21 joints to give the policy actual lateral DOFs (4 hip
abductions + spine yaw + neck yaw — every joint in v1 rotated around the same X
axis, so the deer literally could not spread legs sideways or twist).

Three training stages chained automatically. Final result: **a deer that walks at
3-DOF velocity commands on rough terrain, recovers from pushes, with the new
HAA joints actively engaged for lateral balance.** All from-scratch with no
reference clips.

## Pipeline changes

### Stage 0 — URDF v2 + USD regen + tooling fixes
- Edited `assets/deer_urdf/deer_world.urdf`: added 6 zero-length frame bones
  (`HR_hip_abd`, `HL_hip_abd`, `FR_hip_abd`, `FL_hip_abd`, `chest_yaw_link`,
  `neck_yaw_link`) and 6 new revolute joints (4 HAA on body Z, spine yaw + neck
  yaw on body Y). The 6 existing X-axis joints kept their axes but got renamed
  + reparented to flow through the new frame bones.
- Backups in `assets/deer_urdf/deer_world.pre_v2.urdf` and `deer_world.pre_v2.usd`.
- Regenerated USD via `tools/import_world_urdf_v3.py` (the existing
  RTX-50-safe minimal-kit pattern) — 21 joints, 22 rigid bodies, defaultPrim
  `/deer`, identity root rotation (Rx(-90°) still applied at spawn via
  `init_state.rot`, unchanged).
- Cleaned orphan `/visuals` references on frame bones (stops the "Unresolved
  reference prim path" warnings on every load).
- Updated `training/envs/deer_usd_articulation.py`: actuator regex/explicit
  names cover all 21 joints with mutually-exclusive matching (firm DC gains on
  HAA + leg flex/ankle, soft on spine yaw / pitch / neck / skull). `_DEFAULT_JOINT_POS`
  expanded to 21 entries (new joints at 0).
- Updated `training/envs/deer_rewards.py`: `hind_joint_pos_l2` and
  `leg_joint_activity` now reference the renamed hip-flex joint names.
- Updated `DEER_URDF_JOINT_NAMES` in `training/envs/deer_usd_amp_env_cfg.py`
  to the canonical 21-name v2 ordering (kept for tooling compatibility even
  though the AMP pipeline is now quarantined).
- Upgraded `rsl-rl-lib` 3.0.1 → 5.0.1 (Isaac Lab 2.7 expects RSL-RL 5.0 API
  fields like `share_cnn_encoders`). Safe because we no longer use
  `amp-rsl-rl`.

### New training infrastructure
- `training/envs/deer_usd_loco_env_cfg.py` — three env classes (`DeerUsdLocoStage1Cfg`,
  `Stage2`, `Stage3`) inheriting from `DeerUsdRoughEnvCfg`. Reward stack and
  termination set in __post_init__ overrides.
- `training/agents/deer_usd_loco_cfg.py` — three vanilla PPO runner configs.
- `training/scripts/play_loco.py` — adapted from `play_amp.py`, uses
  `OnPolicyRunner` (vanilla rsl_rl 5.0+) instead of `AMPOnPolicyRunner`. Calls
  `handle_deprecated_rsl_rl_cfg` to convert config dataclass to the dict
  format the new PPO `construct_algorithm` expects.
- 3 task registrations in `training/tasks/__init__.py`: `AniML-Deer-USD-Loco-Stage1-v0`,
  `Stage2-v0`, `Stage3-v0`.

## Reward stack (Lee/Margolis/Hwangbo recipe + deer-specific tweaks)

Tracking (positive):
- `track_forward_vel_anatomical_exp` (+1.5, std=0.5)
- `track_lateral_vel_anatomical_exp` (+0.5, std=0.5)
- `track_yaw_rate_anatomical_exp` (+0.75, std=0.5)
- `feet_air_time` (+1.0, threshold=0.5s)
- **`pelvis_height_ramp` (+5.0)** — the survival-incentive fix that broke S1's
  give-up exploit. See "Failure & fix" below.

Smoothness / efficiency (negative):
- `lin_vel_z_l2` (−2.0), `ang_vel_xy_l2` (−0.05), `flat_orientation_l2` (−5.0),
  `base_height_l2` (−20, target=0.85), `dof_torques_l2` (−2.5e-5),
  `dof_acc_l2` (−2.5e-7), `action_rate_l2` (−0.01), `feet_slide` (−0.5),
  `joint_pos_standing_l2` (−0.5).

Terminations:
- `pelvis_too_low` (0.30m for S1, room to learn; tightenable in later stages)
- `base_orientation_too_tilted` (60°)

Action scale: 0.3 (down from the legged_gym default 0.5; the 21-DOF deer is
more sensitive to joint magnitude than the 12-DOF ANYmal).

## Failure & fix on first S1 attempt

Initial S1 (init_noise_std=1.0, action_scale=0.5, no pelvis_height_ramp)
crashed via the **give-up exploit**: at iter 1516, ep_len had collapsed from 60
to 19 steps, std had dropped 1.0→0.24. The policy learned that ending episodes
fast minimized accumulated negative reward, since the small positive task
rewards weren't compelling enough to outweigh the smoothness penalties.

**Fix**: added `pelvis_height_ramp` (+5 weight, dense ramp from z_min=0.40 to
z_max=0.85). At full standing height over a 1000-step episode this contributes
up to +5000 reward — swamps the small negatives so survival becomes the dominant
gradient signal. Also relaxed `pelvis_too_low` termination to 0.30m (gives the
policy room to explore lower poses without instant termination). Lowered
`init_noise_std` to 0.5 (1.0 in 21 dims was destabilizing).

After fix, smoke test went from ep_len 60 + 100% pelvis_too_low termination to
ep_len 94 + 0% pelvis_too_low at iter 30. Full S1-r2 succeeded.

## Stage results

| metric | S1 final | S2 final | S3 final |
|---|---|---|---|
| Mean reward | 72.5 | **77.0** | 62.3 |
| Mean ep_len | 819 | 915 | **931** |
| time_out fraction | 0.41 | 0.86 | 0.86 |
| base_tilted fraction | 0.59 | 0.14 | **0.14** |
| pelvis_too_low fraction | 0.0 | 0.0 | 0.003 |
| track_forward_vel | 0.79 | 0.76 | 0.71 |
| track_lateral_vel | n/a (cmd=0) | 0.37 | 0.38 |
| track_yaw_rate | n/a (cmd=0) | 0.31 | 0.31 |
| pelvis_height_ramp | 4.15 | 4.61 | 4.46 |
| Mean action std | 0.42 | 0.28 | (preserved from S2) |

**Key narrative**: The new HAA joints really are getting used. S1 trained
fwd-only and the deer learned to walk forward — but had nothing forcing
lateral balance, so 60% of episodes ended via lateral tilt. S2 added lateral
+ yaw commands + push events, which gave the policy reason to actually use
the HAA joints — base_tilted dropped from 60% → 14%. S3 added rough terrain
and the deer kept the same gait quality.

## Outputs

- Checkpoints:
  - `C:/isaac-lab/IsaacLab/logs/rsl_rl/animl_deer_loco_stage1/2026-05-01_10-15-33/model_1999.pt`
  - `C:/isaac-lab/IsaacLab/logs/rsl_rl/animl_deer_loco_stage2/2026-05-01_11-43-05/model_1999.pt`
  - `C:/isaac-lab/IsaacLab/logs/rsl_rl/animl_deer_loco_stage3/2026-05-01_13-12-13/model_1999.pt`
- Trajectory dumps: `logs/videos/play_loco_2026-05-01_15-{00-01,04-35,09-22}.npz`
- Videos:
  - `logs/videos/loco_s1.mp4` (flat, fwd-only)
  - `logs/videos/loco_s2.mp4` (flat, full 3DOF + pushes)
  - `logs/videos/loco_s3.mp4` (rough terrain, full 3DOF + pushes)
- Visual sanity verification:
  - `logs/videos/v2_static_pose.mp4` (5-second URDF v2 sweep showing default
    pose + HAA spread + spine/neck yaw)
- This summary: `docs/overnight_2026-05-01_loco.md`

## What to do next

1. **Watch the videos.** That's the actual deliverable. Compare to
   `r45_fromscratch.mp4` and `r46_anneal.mp4` (the AMP-era runs you saw
   yesterday). Specifically look for:
   - Does the trot look more natural? (HAA joints should produce visible
     lateral hip motion now, vs the rigid forward-only stiff legs of v1)
   - Is the body twisting/banking on turns? (spine yaw)
   - Is the head turning to track motion? (neck yaw)
   - On terrain (S3), is foot placement adapting to bumps?

2. **Decide direction**:
   - If gait looks much better → Stage 4 ideas: idle/standing-still natural
     behaviors when cmd=0 (current weight-shift logic is undertrained), head
     "look at" command channel, multiple gait modes.
   - If gait improved but still odd → next lever is hip internal/external
     rotation (HIR) joints (+4 more joints, full ANYmal-style 3-DOF hips).
     Less impact than HAA but adds polish.
   - If gait looks similar to before → likely the morphology / mass
     distribution of the Unity rig is the next ceiling, and we'd need to
     adjust mass/inertia values per `docs/ANIMAL_PHYSICS_PIPELINE.md`.

3. **Quarantined AMP path** stays available in the repo
   (`training/envs/deer_usd_amp_env_cfg.py`, `training/agents/deer_usd_amp_cfg.py`,
   `training/scripts/train_amp.py`, `assets/motions/*.npy`) if we want to
   layer AMP polish on top of the reward-shaped policy later (e.g., with
   retargeted Google `motion_imitation` dog mocap).
