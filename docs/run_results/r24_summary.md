# Run 24 — task_reward_ratio 0.7 (task-heavy blend)

**Date:** 2026-04-28
**Task:** AniML-Deer-USD-Walk-AMP-v17 + `--task_reward_ratio 0.7`
**Duration:** 120 min for 3000 iters at 14000 envs

## Hypothesis
r23 fixed most of the velocity-tracking regression but lin_vel_R was
still 0.14 (vs r19 baseline 0.33). amp-rsl-rl's default 0.5/0.5 task/style
blend may be giving too much weight to style now that style is producing
strong signal. Bumping to 0.7/0.3 should put more pressure on the policy
to actually track the velocity command.

## Results

| Metric | r23 | **r24 (task=0.7)** | Note |
|---|---|---|---|
| mean_reward | 240 | 149 | reduced — style scaled 0.6× in display |
| style_r/step | 0.484 | 0.292 | display scaled; raw ≈ 0.49 (≈ r23) |
| **task_r/step** | -0.005 | **+0.0022** | first ever **positive** raw task reward |
| ep_len | 1000 | 999 | flat |
| lin_vel_R | 0.14 | 0.14 | flat |
| err_vel_xy | 1.76 | 1.73 | trivial improvement |
| amp_loss | 0.49 | **0.50** | highest disc challenge yet |
| noise_std | 1.47 | **1.01** | task-dominant => more deterministic |

## Conclusion
The task_reward_ratio knob did exactly what it claims (boosted task reward
contribution 1.4× while halving style 0.6×), and the deer's task_r went
positive for the first time. But velocity-tracking metrics (lin_vel_R,
err_vel_xy) basically didn't move — there's a soft ceiling around 0.14 /
1.7 in the current setup. Likely root cause: with multi-clip discriminator,
the policy has many "natural-looking" gait basins to sit in. Velocity
matching only requires being IN one of those basins; the discriminator
can't tell which one matches the command.

The lower noise_std (1.01 vs r23's 1.47) is notable — task-pressure
encourages deterministic policy. That's actually GOOD for predictable
deployment behaviors but reduces exploration.

## Hardware
14000 envs at 11 GB / 16 GB VRAM (68%), 94% GPU util. Properly using
the 5080 now. Per-iter time slightly longer than 12000 envs, total run
~10% longer.

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_wri_narrow/2026-04-28_16-38-28/model_2999.pt`
- Mesh playback: `logs/videos/r24_meshed.mp4` (3.8 MB)
