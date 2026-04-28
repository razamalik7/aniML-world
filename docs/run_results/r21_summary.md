# Run 21 — Multi-clip walk + run

**Date:** 2026-04-28
**Task:** AniML-Deer-USD-Walk-AMP-v15 (V12 env, walk+run dataset)
**Duration:** 110 min for 3000 iters at 12000 envs

## Hypothesis
r20 visual feedback: even with widened torso joints, walk.npy's spine
amplitude (2.4°) is too small for the discriminator to teach whole-body
motion. Multi-clip = richer reference manifold = should pull policy
toward bigger amplitude motion across body.

## Setup
- `DEER_USD_AMP_WALK_RUN_CFG`: AMP runner sees walk + run datasets
  (weights 1.0 each).
- `DeerUsdWalkAMPEnvCfgV12`: V11 + widened velocity command range
  (0.3-2.5 m/s) to span both gait speeds.
- Same widened URDF/USD as r20 (no further joint changes).

## Results

| Metric | r19 | r20 | **r21** |
|---|---|---|---|
| Iters | 3000 | 3000 | 3000 |
| mean_reward | 174 | 195 | **223** (+28% vs r19) |
| style_r/step | 0.377 | 0.403 | **0.461** (+22%) |
| ep_len | 999 | 996 | **1000/1000** |
| pelvis_too_low | 0.02% | 0.11% | **0.00%** (never fell) |
| amp_loss | 0.38 | 0.41 | **0.47** (disc challenged) |
| noise_std | 1.98 | 1.67 | **1.53** (most stable) |
| **lin_vel_R** | 0.33 | 0.33 | **0.11** ⚠ regression |
| **err_vel_xy** | 1.40 | 1.38 | **2.88** ⚠ regression |

## Trade-off observed
Style metrics jumped substantially — discriminator finds 22% more signal
on whole-body motion, deer is more stable (never falls). But velocity
tracking regressed: the policy seems to settle at a comfortable speed
between walk (0.8 m/s) and run (2.5 m/s) rather than tracking the actual
command. Possible interpretations:
  (a) Wider command range (0.3-2.5) makes command harder to follow;
      policy averages.
  (b) Discriminator now rewards motion that doesn't strongly correlate
      with commanded speed.
  (c) Without speed conditioning in the AMP discriminator (it sees
      base_lin_vel but not command), it can't differentiate "looks right
      at speed X" from "looks right at speed Y" — so the policy is OK
      with one self-consistent gait at any speed.

If (c) is the issue, we'd want:
  - Speed conditioning baked into AMP obs (richer obs group)
  - OR keep narrower command range matched to a single clip
  - OR reweight task vs style (more task pressure)

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_walk_run/2026-04-28_10-50-50/model_2999.pt`
- Mesh playback: `logs/videos/r21_meshed.mp4` (5.3 MB)
