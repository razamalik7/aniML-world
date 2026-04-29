# Run 26 — Anatomy-axis fix + linear forward reward (PARTIAL FIX)

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v18 (V14 env, walk+run+idle dataset)
**Duration:** 130 min for 3000 iters at 14000 envs

## What was fixed
- Removed wrong-axis `track_lin_vel_xy_exp` (was measuring body +X = anatomical LEFT)
- Added `track_forward_vel_anatomical_exp` (correct: body -Z), weight 1.5
- Added `track_yaw_rate_anatomical_exp`, weight 0.5
- Added `forward_velocity_linear_anatomical`, weight 2.0 (linear, zero-at-stop)

## Result: NOT ENOUGH

Trajectory inspection of model_2999.pt:
- env 0: avg fwd speed **0.005 m/s** (commanded ~0.85 m/s)
- env 1: avg fwd speed 0.000 m/s + **fell over** (root z 0.93 → 0.33)
- env 2: avg fwd speed 0.013 m/s
- env 3: avg fwd speed 0.016 m/s + **fell over**

Deer is essentially **stationary** in 2/4 envs and **falling over** in the
other 2. Same failure mode as r25 despite the axis fix and linear reward.

## Why it didn't work
The "stand still and look natural" basin is too strong:
- Standing still earns ~0.48/step style reward (idle.npy frames are ~standing)
- Standing still earns exp(-0.85²) × 1.5 = 0.74/step from track_forward_vel_exp
- Standing still earns 0 from forward_velocity_linear
- Total standing reward: ~1.2/step × 1000 = 1200/episode (rough)

To beat this, the policy needs to discover walking AND not fall while
exploring. Initial walking attempts fall → lower reward → policy retreats
to standing. Local optimum trap.

## Diagnosis
The base AMP env has **RSI (reference state initialization) OFF**.
Episodes start with deer in standing pose. With idle in the discriminator
dataset, standing pose is a high-reward attractor. The policy has no
reason to leave it.

## r27 plan
Triple-pronged fix:
1. **Add RSI** (V15 env): start each episode mid-walk from a random
   walk.npy frame. Deer arrives in episode already walking — has to
   continue or fall.
2. **Drop idle.npy** from discriminator dataset (use walk+run only).
   No more standing-as-natural reward.
3. **Bump linear forward weight 2.0 → 4.0**. Make moving forward
   dominant.

## Artifacts
- Checkpoint: `logs/rsl_rl/animl_deer_amp_wri_fwdfix/2026-04-29_00-20-12/model_2999.pt`
- Mesh playback: `logs/videos/r26_meshed.mp4`
