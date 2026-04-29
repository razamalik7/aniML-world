# Run 32 — V20 (fwd weight 8.0) + task_reward_ratio 0.95

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v24 (V20 env, walk+run, --task_ratio 0.95)
**Duration:** 140 min for 3000 iters at 14000 envs

## Result: SPEED breakthrough but stability regressed

| | r31 | **r32** |
|---|---|---|
| reward | 86 | 118 |
| **task_r/step** | +0.040 | **+0.213** ← 5x bigger |
| ep_len (training) | 971 | 1000 |
| pelvis_too_low (training) | 0.27% | 0.01% |
| **err_vel_xy** | 1.09 | 1.10 |

## Trajectory (30s playback)
| env | dy | spd_y | fell |
|---|---|---|---|
| 0 | -4.89 | -0.163 m/s | **Yes** |
| 1 | -4.99 | -0.166 m/s | **Yes** |
| 2 | -4.33 | -0.144 m/s | **Yes** |
| 3 | -5.13 | -0.171 m/s | **Yes** |

**Speed jumped 3× to ~0.16 m/s** — finally getting close to commanded
0.5 m/s. But all 4 envs fell during the 30s playback.

## Why falls didn't show in training
Training has pelvis_too_low termination ON. Fallen episode ends → RSI
respawns mid-walk → new episode starts with fresh body velocity.
Episode length cap is 20s (1000 steps at 50Hz). The policy maintains
balance for ~20s windows reliably; 30s is just past the trained
window.

The policy chose a high-speed "lean forward and rush" gait that earns
maximum task reward but accumulates instability if not reset
periodically.

## r33 plan
1. Dial task_reward_ratio back to 0.85 (less aggressive, keeps style as
   regularizer on body shape)
2. Keep forward_velocity_linear weight 8.0
3. Extend episode_length_s from 20s → 40s so training has to maintain
   balance for longer windows
4. Warm-start from r32's model_2999 (preserves the speed breakthrough)

If r33 produces sustained walking at 0.15+ m/s without falling, that's
deployable. If still falls, try terrain randomization (which forces
balance-aware locomotion) or pure RL without AMP.

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_max_push/2026-04-29_13-34-20/model_2999.pt`
- Mesh playback: `logs/videos/r32_meshed.mp4`
