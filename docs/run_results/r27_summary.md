# Run 27 — RSI + walk+run + fwd reward 4.0 (REGRESSION)

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v19 (V15 env, walk+run dataset)
**Duration:** 130 min for 3000 iters at 14000 envs

## What was changed from r26
1. RSI (reset_from_amp_motion) — deer arrives mid-walk in each episode
2. Removed idle.npy from discriminator dataset
3. Bumped forward_velocity_linear weight 2.0 → 4.0
4. Removed pelvis_too_low termination (mistake!)

## Result: WORSE THAN r26

Trajectory of model_2999.pt (4 envs, 30s each):
- env 0: y_speed -0.041 m/s, **fell** (z 0.93→0.43)
- env 1: y_speed -0.034 m/s, **fell** (z 0.93→0.37)
- env 2: y_speed -0.017 m/s, **fell** (z 0.93→0.28)
- env 3: y_speed -0.036 m/s, **fell** (z 0.93→0.41)

All 4 envs fell over and stayed down. Commanded speed was 0.5-1.2 m/s
fwd; actual was ~0.03 m/s.

## Why it failed
Removing `pelvis_too_low` termination eliminated the natural counter-
incentive to falling. Once a deer fell, the episode kept running for
1000 steps while it accumulated style reward (run.npy contains some
near-ground frames during stride extension that match a fallen-deer
pose well enough). Falling became the easiest way to maximize ep_len
without active locomotion.

## Lesson learned
`pelvis_too_low` termination was load-bearing in r19's success — it
forced the policy to maintain upright posture as a precondition to
earning any reward. Removing it tripped a perverse incentive.

r19 (which actually walked forward despite being a shuffle) had:
  - pelvis_too_low termination ON ✓
  - track_lin_vel_xy_exp (wrong axis but gave SOME signal)
  - No RSI

r27 has:
  - RSI ✓ (good)
  - axis fix ✓ (good)
  - linear fwd reward ✓ (good)
  - pelvis_too_low termination OFF ✗ (catastrophic)

## r28 plan
V16 env = V15 + pelvis_too_low termination ON. Keep all the structural
fixes; restore the upright-posture constraint that was the load-bearing
piece of r19's success.

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_wr_rsi/2026-04-29_02-38-12/model_2999.pt`
- Mesh playback: `logs/videos/r27_meshed.mp4`
