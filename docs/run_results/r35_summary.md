# Run 35 — V22 BREAKTHROUGH: deer finally walks upright

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v26 (V22, walk+run, task=0.85)
**Duration:** 120 min for 3000 iters at 14000 envs
**Status:** ⭐ BEST RUN OF ENTIRE PROJECT

## Trajectory — actual walking

| env | dy | spd_y | pelvis z (min/avg/max) |
|---|---|---|---|
| 0 | -2.81m | -0.094 m/s | 0.57 / **0.85** / 1.00 |
| 1 | -9.38m | **-0.313 m/s** | 0.55 / 0.86 / 1.21 |
| 2 | -3.45m | -0.115 m/s | 0.57 / 0.84 / 0.98 |
| 3 | -1.22m | -0.041 m/s | 0.55 / 0.86 / 1.14 |

**Pelvis z averages 0.84-0.86** — target was 0.85. The deer stands upright.
**No belly-slide, no falls in 30s playback.**
env 1 walked **9.4 meters** at sustained 0.31 m/s — actual locomotion.

## What finally worked
After 35 runs of debugging, the full stack of fixes:
1. r15: coord-frame bug fix (.npy axis +X → -Y)
2. r19-r20: URDF joint limits widened
3. r21-r22: multi-clip motion library
4. r23: narrow command range
5. r28: pelvis_too_low termination back ON
6. r29: locked single-direction command
7. r32: max task pressure (0.95 ratio + linear fwd 8.0)
8. **r35: tightened pelvis_too_low (0.10 → 0.55) + base_height_l2 reward (-30 weight)**

The last step killed the belly-slide local optimum that was hiding behind
the lenient termination threshold for many runs.

## Training metrics
| | r34 (belly-slide) | **r35 (upright walking)** |
|---|---|---|
| ep_len | 991 | 796 (terminations firing) |
| pelvis_too_low | 0.01% | 36.4% |
| err_vel_xy | 1.11 | **0.85** |
| reward | 119 | 95 (display scaled) |

## r36 plan
This is a strong baseline. Next steps:
- Warm-start r35 + train more iters to refine
- Consider adding terrain randomization (DeerUsdRoughEnvCfg-style) for
  emergent terrain-adaptive behavior
- Or push events for recovery learning

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_no_belly/2026-04-29_19-13-44/model_2999.pt`
- Mesh playback: `logs/videos/r35_meshed.mp4`
