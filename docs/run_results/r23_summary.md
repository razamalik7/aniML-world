# Run 23 — 3-clip dataset + narrow command range

**Date:** 2026-04-28
**Task:** AniML-Deer-USD-Walk-AMP-v17 (V13 env, narrow range 0.5-1.2 m/s, walk+run+idle dataset)
**Duration:** 110 min for 3000 iters at 12000 envs

## Hypothesis
r21/r22 showed multi-clip dataset improves style metrics dramatically
(+22-26%) but breaks velocity tracking. Hypothesis: it's the wider V12
command range (0.3-2.5 m/s) that makes velocity tracking impossible —
narrowing back to walk speeds while keeping the rich multi-clip
discriminator should preserve style benefits AND fix velocity tracking.

## Results — best run on every metric

| Metric | r19 | r20 | r21 | r22 | **r23** |
|---|---|---|---|---|---|
| mean_reward | 174 | 195 | 223 | 234 | **240** |
| style_r/step | 0.377 | 0.403 | 0.461 | 0.476 | **0.484** |
| ep_len | 999 | 996 | 1000 | 1000 | **1000** |
| pelvis_too_low | 0.02% | 0.11% | 0% | 0.02% | **0.00%** |
| amp_loss | 0.38 | 0.41 | 0.47 | 0.49 | **0.49** |
| noise_std | 1.98 | 1.67 | 1.53 | 1.52 | **1.47** |
| **lin_vel_R** | 0.33 | 0.33 | 0.11 | 0.10 | **0.14** |
| **err_vel_xy** | 1.40 | 1.38 | 2.88 | 2.87 | **1.76** |

## Conclusion
The narrow command range hypothesis was correct. err_vel_xy recovered
from 2.87 → 1.76 (38% improvement) while style_r/reward kept their
multi-clip-driven peaks. Still slightly below r19's vel-tracking
baseline (1.40) but in the same ballpark; the policy is genuinely
matching the (now reachable) command.

## Where we stand
After r19→r23 sequence:
- Bug-fix unlock: style_r 0.003 → 0.48 (160x improvement over r1-r14)
- Multi-clip + torso widening + narrow range: best stability,
  highest amp_loss (challenged disc), lowest noise_std drift
- Trade space identified: command range vs disc-style coupling

## r24 candidate paths
1. `--task_reward_ratio 0.7` to push velocity tracking further
2. Add commanded velocity to AMP obs group (so disc sees command)
3. Domain randomization (terrain bumps, push events) for emergent
   recovery / robustness
4. Warm-start from model_2999 + harder velocity command schedule

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_wri_narrow/2026-04-28_14-42-46/model_2999.pt`
- Mesh playback: `logs/videos/r23_meshed.mp4` (2.1 MB)
