# Overnight 2026-05-02 — S1 Run Series r3 → r8

## TL;DR

Eight S1 attempts to discover trotting with pure reward-shaping. Each
attempt fixed one local optimum and exposed the next. Best single result:
**r7 — first run where the deer stood upright on all 4 feet** (a real
deer-shaped silhouette). Forward speed remains slow (~0.05 m/s). The
fundamental finding: with this 21-DOF deer + free-form reward shaping,
any reward weight pushed high enough to drive the desired behavior gets
gamed; any weight low enough to not be gamed isn't strong enough to
drive behavior. Walk-These-Ways' gait conditioning helped (r6 onwards
showed real diagonal-pair alternation) but didn't break the trade-off.

Best checkpoint: `logs/rsl_rl/animl_deer_loco_stage1/2026-05-02_02-23-32/model_1999.pt` (r7).
Video: `logs/videos/loco_s1_r7.mp4`.

## Run-by-run timeline

| run | env change | result | failure mode |
|---|---|---|---|
| r2 | base recipe | reward 60.9, ep_len 689 | kneeling on calves |
| r3 | re-enable undesired_contacts | reward 67.7, ep_len 992 | tuck-and-freeze (0.003 m/s) |
| r4 | + forward_velocity_linear, no standing envs | reward 48, fwd 0.18 m/s | rigid-front wheelbarrow |
| r5 | + ready-to-walk default pose, terrain, pushes, action 0.5, feet_swing_clearance | reward 73, fwd 0.03 m/s | downward-dog (chest drop) |
| r6 | + Walk-These-Ways gait conditioning (phase obs + contact match + shaped force) | reward 113, gait_match 2.57 | hind-only trot, chest still down |
| **r7** | + torso_contact_penalty, head_above_base | **reward 76, ep_len 800, gait_match 2.41, all 4 legs upright** | slow (~0.05 m/s) |
| r8 | warm from r7, fwd_vel_linear 5→10, head_above_base −3→−1 | mid-run reward 146 then regressed to 85 | reverted to chest-drop chasing speed |

## Key technical additions (kept in repo)

### Code
- `training/envs/deer_gait.py` — Walk-These-Ways-style gait conditioning:
  - `gait_phase_obs(env, frequency, foot_phase_offsets)` — periodic encoding of per-foot phase fed into policy obs
  - `gait_contact_match(env, ...)` — reward per step for matching prescribed contact schedule (target_stance = phase < duty_cycle)
  - `gait_shaped_force(env, ...)` — reward low contact force during prescribed swing phase
  - Phase computed from `env.episode_length_buf * env.step_dt * frequency`, auto-resets per episode
- `training/envs/deer_rewards.py:feet_swing_clearance` — exp reward each foot for being lifted to ~5 cm during swing
- `training/envs/deer_usd_loco_env_cfg.py` — current state has all r7 + r8 reward terms wired

### Reward stack (final r8 state)
| term | weight | source |
|---|---|---|
| track_forward_vel | +1.5 | exp tracking |
| track_lateral_vel | +0.5 | exp tracking |
| track_yaw_rate | +0.75 | exp tracking |
| feet_air_time | +1.0 | stock |
| feet_swing_clearance | +1.0 | NEW |
| pelvis_height_ramp | +3.0 | survival |
| **forward_velocity_linear** | +10.0 | r4 reintro, r8 doubled |
| **gait_contact_match** | +1.0 | r6 NEW (Walk-These-Ways) |
| **gait_shaped_force** | +0.5 | r6 NEW |
| flat_orientation_l2 | -5.0 | stock |
| base_height_l2 | -20.0 | stock |
| dof_torques_l2 | -2.5e-5 | stock |
| dof_acc_l2 | -2.5e-7 | stock |
| action_rate_l2 | -0.01 | stock |
| feet_slide | -0.5 | stock |
| joint_pos_standing_l2 | -0.5 | custom |
| **undesired_contacts** | -1.0 | re-enabled (was zeroed early) |
| **torso_contact_penalty** | -2.0 | r7 NEW (chest/neck/skull) |
| **head_above_base** | -1.0 | r7 NEW |

### Observation
- Standard 75 obs from parent (lin_vel/ang_vel/gravity/cmd/joint_pos/joint_vel/last_action)
- + 8 gait phase observations (cos, sin per foot for 4 feet) — total 83

### Asset / config
- URDF v2 (21 joints with HAA × 4, spine yaw, neck yaw)
- Default joint pose updated to "ready-to-walk" (front knees +0.5 rad, hind knees -0.7 rad)
- action_scale = 0.5 (Walk-These-Ways value)
- Terrain enabled from iter 0 (mild ±0.5–4 cm heightfield)
- Push events enabled from iter 0 (±0.4 m/s every 8-12 s)

## What we learned (for future you / the user)

1. **Recipe-shaped recipes need full perturbations from the start** — disabling terrain + pushes for "easier S1" exposed many degenerate stationary solutions the recipe was designed to prevent.

2. **The 21-DOF deer has more degenerate gaits than ANYmal** (kneeling, tucking, wheelbarrow, downward-dog). Each requires explicit prevention. ANYmal's 12-DOF makes some of these mechanically infeasible.

3. **Gait conditioning is necessary, not sufficient** — gait_contact_match emerged at 2.57/4.0 (64% per-foot accuracy on trot) but the deer satisfied it with hind-only trotting + chest drop. The policy will exploit any unconstrained DOF.

4. **Reward weight tuning is brittle** — bumping forward_velocity_linear 5→10 (r8) broke the upright posture from r7. Each weight increase opens a new failure mode.

5. **Best single artifact (r7)** comes from layering ALL the constraints: gait + torso + head + leg + height + standing-pose + tracking. Backing off any one releases a degenerate solution.

## Best result: r7

`logs/rsl_rl/animl_deer_loco_stage1/2026-05-02_02-23-32/model_1999.pt`

Final metrics:
- Mean reward: 76 (lower than r6's 113 because of head_above_base penalty)
- ep_len: 800
- time_out: 74%, base_tilted: 26%, pelvis_too_low: 0%
- gait_contact_match: 2.41 (60% trot accuracy)
- forward_velocity_linear: 0.23 (slow)
- head_above_base: -0.70 (head IS up but with residual penalty)
- torso_contact_penalty: -0.10 (chest mostly off ground)

Visual (`logs/videos/loco_s1_r7.mp4`): deer standing on all 4 legs, head upright,
body horizontal, hind legs visibly cycling through trot stance/swing, front
legs partly active (knee std 1-5° vs hind 22-29° — fronts noticeably more
rigid). Forward distance over 30s: 1.4-1.9 m per env. Slow walk pace.

## CORRECTION (after user re-watched r7 + r8)

The "upright on all 4 feet" claim for r7 was wrong on closer inspection.
The deer is **wrist-walking**: front foot bent flat against the ground at
the shin-to-foot joint, with the foot pad acting as the contact patch.
Looks like "kneeling on front knees" anatomically. The contact body IS
the foot (which is what `undesired_contacts` *desires*) so the penalty
doesn't fire — but the foot orientation is wrong. The hind legs push
forward, the front "kneels", deer creeps along.

r8 made it worse: with `head_above_base` weight reduced -3.0 → -1.0, the
deer's head dropped further toward the floor while the wrist-walking
gait stayed.

Adding to the local-optimum catalog (now 9 distinct failure modes):
- r7 / r8 mode: **wrist-walking** — foot lays flat at shin-to-foot joint.
  Defeats `undesired_contacts` (regex matches thigh/shin/calf/knee/shank,
  NOT foot).

**Implication**: every reward we add opens the next loophole. The deer
is *finding the cleverest local optimum* that satisfies all explicit
rewards while violating an *unstated* assumption we forgot to encode.
After 8 attempts and ~10 distinct reward terms, the search keeps
discovering exploits faster than we can patch them.

## Recommended next moves (for tomorrow)

In order of likely impact:

1. **Sample gait parameters per episode** (Walk-These-Ways' main contribution we haven't done). Fixed gait freq 2 Hz + offsets (0,0,0.5,0.5) = trot only. Sampling freq ∈ [1.5, 3.5] Hz and stance fraction ∈ [0.4, 0.6] would make the policy learn a *family* of gaits, which they observe makes individual gaits more robust because the policy can't memorize one solution.

2. **Asymmetric leg-segment penalty** — front knees keep locking. Add a penalty for front-knee acceleration variance below threshold, or a positive reward for front-knee angular velocity matching the gait phase. Force the front legs to participate.

3. **Lower forward_velocity_linear back to 5.0**, add separate `forward_velocity_target_clip` reward that ONLY pays when actual matches cmd within a tolerance — so going faster doesn't get more reward. This breaks r8's "lean and drag for speed" exploit.

4. **Re-render r7 + r8 as side-by-side visual comparison** to clearly show the trade-off (r7 upright slow vs r8 fast but collapsed).

5. **Honest alternative**: revisit AMP with retargeted Google `motion_imitation` dog mocap. The "tame the local optima" battle has eaten 6 runs. AMP's discriminator generically suppresses non-natural patterns from the reference distribution.

## Files / artifacts

- All r2-r8 logs in `logs/loco_s1_r{N}_*.log` (renamed for posterity)
- Best video: `logs/videos/loco_s1_r7.mp4`
- All 8 attempt videos exist for comparison: `loco_s1_r{2..8}.mp4`
- This summary: `docs/overnight_2026-05-02_loco_gait.md`
- Yesterday's summary: `docs/overnight_2026-05-01_loco.md`
- Plan history: `C:/Users/nars7/.claude/plans/crispy-gliding-cupcake.md`
