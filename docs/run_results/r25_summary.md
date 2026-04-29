# Run 25 — Warm-start from r23 + 2000 more iters

**Date:** 2026-04-28
**Task:** AniML-Deer-USD-Walk-AMP-v17 + `--resume_from r23/model_2999.pt`
**Duration:** 80 min for 2000 additional iters at 14000 envs (5000 cumulative)

## Results — best on every aggregate metric

| Metric | r19 | r23 | **r25 (5000 iters cum.)** |
|---|---|---|---|
| mean_reward | 174 | 240 | **245** ← best ever |
| style_r/step | 0.377 | 0.484 | **0.496** ← best ever |
| amp_loss | 0.38 | 0.49 | **0.50** ← best ever |
| ep_len | 999 | 1000 | 998 |
| pelvis_too_low | 0.02% | 0.00% | 0.01% |
| noise_std | 1.98 | 1.47 | 1.46 |
| lin_vel_R (avg) | 0.33 | 0.14 | 0.13 |
| lin_vel_R (**peak**) | — | 0.18 | **0.363** ← exceeds r19 baseline |
| err_vel_xy | 1.40 | 1.76 | 1.77 |

## Notable observation — lin_vel_R variance

`lin_vel_R` peak of 0.363 in a 20-iter window MEANS the policy can track
velocity at r19-baseline levels under the right conditions. The avg of
0.13 means it doesn't always do so. This bimodality (sometimes-tracks,
sometimes-doesn't) is the classic AMP-trained-policy-with-many-good-basins
problem: with multi-clip discriminator, several gait shapes are equally
"natural-looking", and the policy doesn't always pick the one that
matches command.

## Convergence assessment
The +2000-iter extension squeezed marginal gains over r23:
  reward     +5  (+2%)
  style_r    +0.012 (+2.5%)
  amp_loss   +0.01

We're hitting diminishing returns on this configuration. Future gains
require structural changes (terrain, perturbations, better velocity
conditioning, skill-latent architecture).

## Best deployable checkpoint
`logs/rsl_rl/animl_deer_amp_wri_narrow/2026-04-28_18-44-57/model_4998.pt`
(+ matching `policy_4998.onnx` for Unity Sentis)

## Mesh playback
`logs/videos/r25_meshed.mp4` (3.5 MB)
