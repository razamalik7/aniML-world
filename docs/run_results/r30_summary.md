# Run 30 — V18 (V17 + action_rate_l2 = 0)

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v22 (V18 env, walk+run, 14000 envs)
**Duration:** 130 min for 3000 iters

## Result: same bidirectional ~0.04 m/s pattern, task_r finally positive

| | r28 | r29 | **r30** |
|---|---|---|---|
| reward | 195 | 194 | 209 |
| style_r/step | 0.408 | 0.404 | 0.416 |
| **task_r/step** | -0.009 | -0.008 | **+0.012** ← first positive |
| ep_len | 986 | 989 | 994 |
| pelvis_too_low | 0.49% | 0.73% | 0.37% |
| err_vel_xy | 1.68 | 1.97 | 2.01 |
| **noise_std** | 1.72 | 1.79 | **4.21** ← exploded |

## Trajectory
| env | dy | spd_y | fell |
|---|---|---|---|
| 0 | -1.54 | -0.051 | No |
| 1 | -1.39 | -0.046 | No |
| 2 | -1.06 | -0.035 | Yes |
| 3 | -1.10 | -0.037 | No |

Same ~0.04 m/s as r29. Same systematic bidirectional pattern in 5s windows
(+1.35, 0, 0, -1.37, +1.26 m). The motion is suspiciously periodic —
suggests a slow bounding gait cycle of ~10-14s, way too slow for
natural walking (real walk cycle = ~1s).

## Defensive layer fired
noise_std drifted to 4.21 (from 1.79) but std-clamp safeguard prevented
the NaN crash. Action_rate_l2 was helping keep std bounded; without it
the policy explores wide action range.

## Hypothesis for r31
The deer can't escape the slow-bounce local optimum. Try:
1. Lower commanded speed to 0.5 m/s (more achievable target)
2. Boost task_reward_ratio to 0.85 (force velocity tracking pressure)
3. Possibly bring back action_rate_l2 = -0.005 (half of original) to
   tame the wild noise_std without re-blocking fast joint motion

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_free_action/2026-04-29_08-49-38/model_2999.pt`
- Mesh playback: `logs/videos/r30_meshed.mp4`
