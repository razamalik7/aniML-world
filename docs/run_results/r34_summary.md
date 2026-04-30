# Run 34 — Warm-start r32 with identical settings

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v24 (V20, walk+run, task=0.95, warm r32)
**Duration:** 90 min for 2000 iters

## Result: identical to r32 — refining didn't help, BELLY-SLIDE EXPOSED

| | r32 | **r34** |
|---|---|---|
| reward | 118 | 119 |
| task_r/step | +0.213 | +0.216 |
| ep_len | 1000 | 991 |
| pelvis_too_low (training) | 0.01% | 0.01% |
| err_vel_xy | 1.10 | 1.11 |
| noise_std | 0.75 | **0.66** |

## Trajectory (30s)
| env | dy | spd_y | "fell_at" |
|---|---|---|---|
| 0 | -5.00m | -0.167 m/s | t=1.5s |
| 1 | -4.71m | -0.157 m/s | t=1.1s |
| 2 | -4.66m | -0.155 m/s | t=1.3s |
| 3 | -4.54m | -0.151 m/s | t=1.1s |

## CRITICAL DIAGNOSIS — pelvis Z timeline

```
    t=  0.0s   2.0s   4.0s   6.0s   8.0s ... 28.0s
env 0:  0.93  0.34  0.31  0.31  0.31 ...  0.31
env 1:  0.93  0.30  0.31  0.31  0.30 ...  0.31
env 2:  0.93  0.31  0.30  0.30  0.31 ...  0.31
env 3:  0.93  0.31  0.32  0.31  0.31 ...  0.31
```

The deer's pelvis sits at **z = 0.30-0.32 for the entire episode**.
It's **belly-sliding** at 0.16 m/s, not walking. Limbs flail, body
slides forward on the ground.

## Root cause
`pelvis_too_low` termination's default `min_height = 0.10` (10cm).
The deer's belly at z=0.30 doesn't trigger termination because
30cm > 10cm. So the policy learned belly-slide as the "stable" gait
that maximizes forward velocity reward without violating any
termination constraint.

This explains the recurring belly-slide failure mode across many
historical runs (r11b etc. in prior sessions). The lenient threshold
has been a hidden enabler all along.

## r35 plan
V22:
1. Tighten `pelvis_too_low` min_height 0.10 → 0.55 (catches
   belly-slides — deer pelvis must stay above 55cm)
2. Add `base_height_l2` reward at target ~0.85 with negative weight
   to actively pull pelvis up to walking height
3. Warm-start from r34's model_4998

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_max_push/2026-04-29_17-37-21/model_4998.pt`
- Mesh playback: `logs/videos/r34_meshed.mp4`
