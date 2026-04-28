# Run 22 — Three-clip walk + run + idle

**Date:** 2026-04-28
**Task:** AniML-Deer-USD-Walk-AMP-v16 (V12 env, walk+run+idle dataset, idle weight 0.5)
**Duration:** 110 min for 3000 iters at 12000 envs

## Results

| Metric | r19 | r20 | r21 | **r22** |
|---|---|---|---|---|
| Iters | 3000 | 3000 | 3000 | 3000 |
| mean_reward | 174 | 195 | 223 | **234** (+34% vs r19) |
| style_r/step | 0.377 | 0.403 | 0.461 | **0.476** |
| ep_len | 999 | 996 | 1000 | 1000 |
| pelvis_too_low | 0.02% | 0.11% | 0.00% | 0.02% |
| amp_loss | 0.38 | 0.41 | 0.47 | **0.49** |
| noise_std | 1.98 | 1.67 | 1.53 | 1.52 |
| **lin_vel_R** | 0.33 | 0.33 | 0.11 | **0.10** |
| **err_vel_xy** | 1.40 | 1.38 | 2.88 | **2.87** |

## Conclusion of the r19→r22 sequence

| Change | Style ∆ | Vel-track ∆ |
|---|---|---|
| r19→r20 (torso wider) | +6% | flat |
| r20→r21 (add run clip) | +14% | -67% (broke) |
| r21→r22 (add idle clip) | +3% | flat |

Multi-clip clearly improves AMP-side metrics (the discriminator finds richer
signal across body), but the wider command range (0.3–2.5 m/s in V12) makes
velocity-tracking a fool's errand. Policy settles into a self-consistent
"natural-looking deer" that the discriminator likes regardless of speed.

## Path forward (r23)

Best of both: keep the multi-clip dataset (style_r 0.48 is real) but
narrow the command range so velocity-tracking task reward can pull the
policy onto the actual commanded speed. Expected: 0.5-1.2 m/s range
should align with walk gait while letting the discriminator's broader
body-motion signal still apply (run clip's spine/leg pose info doesn't
require commanding run speed to be useful).

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_walk_run_idle/2026-04-28_12-46-22/model_2999.pt`
- Mesh playback: `logs/videos/r22_meshed.mp4` (2.6 MB)
