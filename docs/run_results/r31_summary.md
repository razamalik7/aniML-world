# Run 31 — V19 lower command (0.5 m/s) + task_ratio 0.85

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v23 (V19 env, walk+run, --task_reward_ratio 0.85)
**Duration:** 140 min for 3000 iters at 14000 envs

## Result: best velocity tracking yet

| | r28 | r29 | r30 | **r31** |
|---|---|---|---|---|
| reward | 195 | 194 | 209 | 86 (display scaled) |
| task_r/step | -0.009 | -0.008 | +0.012 | **+0.040** ← biggest ever |
| style_r/step (raw) | 0.408 | 0.404 | 0.416 | ~0.45 |
| **err_vel_xy** | 1.68 | 1.97 | 2.01 | **1.09** ← halved |
| ep_len | 986 | 989 | 994 | 971 |
| pelvis_too_low | 0.49% | 0.73% | 0.37% | 0.27% |
| **noise_std** | 1.72 | 1.79 | 4.21 | **0.89** ← controlled |
| amp_loss | 0.42 | 0.41 | 0.42 | 0.45 |

## Per-env trajectory (30s playback)

| env | dy | spd_y | fell |
|---|---|---|---|
| 0 | -1.66 | -0.055 m/s | No |
| 1 | -1.51 | -0.050 m/s | No |
| 2 | -1.89 | -0.063 m/s | **Yes** |
| 3 | -1.32 | -0.044 m/s | No |

**All 4 envs consistently moved forward in -Y direction.** The
"5s windows show backward motion" pattern from earlier runs was an
artifact of averaging across envs at different gait phases — per-env
the motion is consistent.

Speed ~0.05 m/s vs commanded 0.5 m/s = 10× too slow. But:
- Direction is consistent
- Forward motion is genuinely happening
- err_vel_xy 1.09 is best of any run since the pre-bug-fix era

## Hypothesis on the speed plateau
The Unity .anim source clips were authored as IN-PLACE walking (no root
displacement). We synthesized forward velocity in the .npy by setting
root_position along world -Y at FORWARD_SPEED. The discriminator can
only score body-shape natural-ness; if the body shape is consistent
with walking-in-place gait (which it is, since that's what the source
contains), style reward stays high regardless of actual forward
displacement.

Combined with the synthesized FORWARD_SPEED being only 0.8 m/s (walk),
the policy's "good enough" zone is roughly 0-0.8 m/s — anything in
that range gets reasonable style reward and the policy settles low to
minimize action expenditure.

## r32 plan
Try pushing task_reward_ratio further (0.95) AND boost
forward_velocity_linear weight to 8.0. Make forward speed unambiguously
the dominant reward source. If still plateaus at 0.05 m/s, the
in-place reference is fundamentally limiting our reachable speed and
we'd need to RE-AUTHOR walk.npy with bone-level forward translation.

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_low_cmd/2026-04-29_11-06-35/model_2999.pt`
- Mesh playback: `logs/videos/r31_meshed.mp4`
