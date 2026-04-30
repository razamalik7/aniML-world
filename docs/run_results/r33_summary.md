# Run 33 — V21 (40s episodes) + warm-start r32 + task_ratio 0.85

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v25 (V21, walk+run, 40s eps, task=0.85)
**Duration:** 90 min for 2000 iters (5000 total cumulative from r32)

## Result: REGRESSION on stability AND direction

| | r32 | **r33** |
|---|---|---|
| reward | 118 | 122 |
| task_r/step | +0.213 | +0.177 |
| ep_len | 1000 | 975 (out of 2000 max) |
| pelvis_too_low | 0.01% | **2.96%** ← exposing falls |
| err_vel_xy | 1.10 | 1.99 |

## Trajectory (30s playback)
| env | dy | spd_y | fell_at |
|---|---|---|---|
| 0 | -0.27m | -0.009 m/s | t=1.2s ← fell instantly |
| 1 | -1.51m | -0.050 m/s | t=1.4s |
| 2 | +5.03m | **+0.168 m/s** | t=2.2s ← went BACKWARD fast |
| 3 | +1.91m | +0.064 m/s | t=1.3s ← also backward |

Two envs went BACKWARD direction at 0.06-0.17 m/s. All 4 fell within 2s.

## Why this is worse than r32
The warm-start + new objective combination destabilized the policy:
- Episode length 20→40s changed the value function targets (more
  reward integrated per non-fall episode, but falls give less)
- task_ratio 0.95→0.85 brought back ~30% style influence, which
  conflicts with the fast-rush gait r32 learned
- Combined effect: policy lost coherence near the warm-start state

## Lesson
Don't change too many variables at once on a warm-start. r32's policy
is brittle but functional; r33's perturbation broke it.

## r34 plan options
1. **Revert to r32's exact settings** + warm-start r32 + train more iters
   (refine without changing variables)
2. **Terrain randomization** — add bumpy ground to force balance
   adaptation. The deer can't stand still on uneven terrain; might
   force genuinely robust locomotion.
3. **Fresh start with V21** (no warm-start) — let the policy learn
   long-episode balance from scratch with the new constraints.

Going with option 1 first — least risky, most likely to recover the
r32 speed without destabilizing.

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_long_ep/2026-04-29_16-01-17/model_4998.pt`
- Mesh playback: `logs/videos/r33_meshed.mp4`
