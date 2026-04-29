# Run 28 — V16 (V15 + pelvis_too_low term back ON)

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v20 (V16 env, walk+run dataset, RSI)
**Duration:** 130 min for 3000 iters at 14000 envs

## Big shift: forward motion is HAPPENING (just not directional)

| Metric | r25 (stationary) | r28 |
|---|---|---|
| reward | 245 | 195 |
| style_r/step | 0.496 | 0.408 |
| ep_len | 998 | **986** (term firing) |
| pelvis_too_low | 0.01% | **0.49%** |
| err_vel_xy | 1.77 | 1.68 |
| **falls in trajectory** | 2/4 | **0/4** ✓ |

## Trajectory analysis (30s playback, 4 envs)

| env | dx | dy | dz | x_speed | y_speed | fell |
|---|---|---|---|---|---|---|
| 0 | -0.10 | **-1.20** | -0.17 | -0.003 | -0.040 | No |
| 1 | -0.16 | **-1.46** | -0.10 | -0.005 | -0.049 | No |
| 2 | +0.02 | **-1.16** | -0.14 | +0.001 | -0.039 | No |
| 3 | -0.15 | **-1.30** | -0.10 | -0.005 | -0.043 | No |

**All 4 envs moved roughly 1.2-1.5m forward** in 30s = ~0.04-0.05 m/s.
Still slow vs commanded 0.5-1.2 m/s, but **actually moving**.

## Bidirectional motion in 5s windows
```
t= 0-5s:  +1.32m forward  ✓
t= 5-10s: +0.01m         (stalled)
t=10-15s: -0.01m         (stalled)
t=15-20s: -1.33m         BACKWARD
t=20-25s: +1.25m forward ✓
```

The deer can walk in both directions and isn't consistently matching
the commanded direction's sign. Net motion ~0 because forward and
backward cancel. This is a MUCH better failure mode than r25-r27 —
the policy learned to walk, just doesn't lock onto command direction.

## Diagnosis for r29
The lateral velocity command range (-0.2, 0.2) and yaw rate (-0.5, 0.5)
sampled randomly may be confusing the policy. Many commands ask the
deer to mostly stop or sidle. Try locking the command range to
forward-only (lin_vel_y=0, ang_vel_z=0) to force directional
consistency. Warm-start from r28's model_2999.

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_wr_rsi_term/2026-04-29_04-55-53/model_2999.pt`
- Mesh playback: `logs/videos/r28_meshed.mp4`
