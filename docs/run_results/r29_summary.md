# Run 29 — V17 (locked command) + warm-start r28

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v21 (V17 env, locked command, walk+run)
**Duration:** 90 min for 2000 iters at 14000 envs (warm-start from r28)

## What was changed
- Command range LOCKED: lin_vel_x=(1.0, 1.0), lin_vel_y=0, ang_vel_z=0
- Warm-started from r28 model_2999

## Result: directional consistency ✓ , speed ✗

| env | dy (forward) | spd_y | fell |
|---|---|---|---|
| 0 | -1.65m | -0.055 m/s | No |
| 1 | -1.01m | -0.034 m/s | No |
| 2 | -1.14m | -0.038 m/s | **Yes** |
| 3 | -1.43m | -0.048 m/s | No |

**All 4 envs moved in -Y (forward)** — directional consistency
fixed by the locked command.
**Speed still ~25× below commanded** (0.04 vs 1.0 m/s).
3/4 stayed upright, 1 fell.

## Bidirectional motion still present in 5s windows
```
t= 0-5s:  +1.31m  ✓
t=15-20s: -1.43m  BACKWARD
t=20-25s: +1.27m  ✓
```

So the deer oscillates: walks forward briefly, pauses or reverses,
walks forward again. Net 5s windows can be negative.

## Hypothesis for r30
The deer "shuffles" forward (slow joint motions) but can't accelerate
to commanded speed because `action_rate_l2 = -0.01` penalizes the fast
joint target changes needed for an actual walk gait.

r30 plan: V18 = V17 with `action_rate_l2.weight = 0`. Free up the
policy to make fast action changes. Train fresh (warm-start might
preserve the slow-shuffle local optimum).

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_locked_fwd/2026-04-29_07-13-35/model_4998.pt`
- Mesh playback: `logs/videos/r29_meshed.mp4`
