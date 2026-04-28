# Run 20 — Torso widening (single walk clip)

**Date:** 2026-04-28
**Task:** AniML-Deer-USD-Walk-AMP-v14 (V11 env: zeroed L2)
**Duration:** 110 min for 3000 iters at 12000 envs

## Hypothesis
r19 visual feedback: deer hips swung but knees and torso were stiff. We
suspected the URDF's tight torso joint limits (pelvis_to_chest ±11.5°,
chest_to_neck -17/+29°, neck_to_skull -23/+34°) were preventing the
discriminator from teaching whole-body motion. Widened those:

| Joint | Before | After |
|---|---|---|
| pelvis_to_chest | ±0.2 rad (±11.5°) | ±0.5 rad (±28.6°) |
| chest_to_neck | -0.3/+0.5 rad | -0.6/+0.8 rad (-34/+46°) |
| neck_to_skull | -0.4/+0.6 rad | -0.8/+0.8 rad (±46°) |

Re-converted walk/run/idle .npy with new limits, patched USD limits
in place via pxr (avoiding the broken Isaac Sim URDF importer).

## Results

| Metric | r19 | **r20** |
|---|---|---|
| Iters completed | 3000 | 3000 |
| mean_reward (last20) | 174 | **195** (+12%) |
| mean_reward peak | 209 | 206 |
| style_r/step | 0.377 | **0.403** (+7%) |
| task_r/step | -0.014 | **-0.007** (better) |
| ep_len | 999/1000 | 996/1000 |
| pelvis_too_low | 0.02% | 0.11% |
| noise_std | 1.98 | 1.67 (more stable) |
| amp_loss | 0.38 | **0.41** |

## Notes
- Walk clip's pelvis_to_chest amplitude is only **2.4°** — even with
  wider limits, single-clip walk barely contains spine motion. Multi-clip
  (r21+) should be the bigger lever for whole-body articulation.
- Pure metric improvement vs r19 suggests the wider torso limits did help
  the discriminator find slightly more signal, but the visual quality is
  the real test (TBD on user review).

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_walk/2026-04-28_08-55-23/model_2999.pt`
- Mesh playback: `logs/videos/r20_meshed.mp4`
