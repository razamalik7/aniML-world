# Night 2 — Best Walking Policy (Run 9)

## Watch it

```powershell
$env:PYTHONPATH = 'C:\Users\nars7\Downloads\aniML_world'
$env:ANIML_DEER_USD = 'C:\Users\nars7\Downloads\aniML_world\deer_world.usd'
$env:ANIML_DEER_ACTIVATE_CONTACT_SENSORS = '1'
$env:ANIML_DEER_STIFFNESS = '60'
$env:ANIML_DEER_DAMPING = '4'
cd C:\isaac-lab\IsaacLab
.\isaaclab.bat -p C:\Users\nars7\Downloads\aniML_world\training\scripts\play_amp.py `
    --task AniML-Deer-USD-Walk-AMP-v7 --num_envs 8
```

Checkpoint: `logs/rsl_rl/animl_deer_amp_walk_v7/2026-04-15_23-50-13/model_2999.pt`

## Final metrics (run 9, best of all runs)

| metric | run 9 (best) | run 8c (prev best) | run 7 (stiff Kp=200) |
|---|---|---|---|
| ep_len (of 250) | **237** | 233 | 228 |
| tilt termination | **9%** | 17% | 19% |
| track_forward_vel | **2.33** | 0.97 | 1.13 |
| noise_std | **0.31** (stable) | 2.53 (ballooned) | 1.23 |
| style_reward/step | 0.003 | 0.28 | 0.38 |

## What changed in run 9

Three fixes on top of night 1's axis-alignment work:

1. **`action_scale` 0.5 → 1.0** — the biggest single fix. The expert walk clip
   has knee bends of 30-60° (0.5-1.0 rad), but at action_scale=0.5 the policy
   was physically capped at ±28°. Legs looked stiff because they literally
   couldn't bend far enough. Now they can reach the full expert range.

2. **`entropy_coef` 0.01 → 0.0** — every prior run had noise_std balloon from
   1.0 to 1.5-2.5, destroying learned gaits. With entropy=0 the std stayed
   put at 0.31, giving the policy stable, low-noise actions throughout.

3. **`Kp=60, Kd=4`** (was Kp=200, Kd=6) — softer PD gains so joints flex on
   ground contact instead of bouncing the whole body. Effect was modest vs
   the action_scale fix but contributed to smoother motion.

## Caveat: AMP style reward collapsed

`style_reward/step = 0.003` means the AMP discriminator effectively
disengaged. With entropy=0 the policy doesn't explore enough to find
expert-matching transitions. The deer walks forward (task reward is
excellent) but the gait may not look "natural" in the AMP sense — it's
optimizing velocity tracking without style guidance.

Next step if you want more natural motion: reintroduce a small entropy
(0.001-0.002) or warm-start from this checkpoint with AMP reward_scale
boosted, so the discriminator re-engages while preserving the walking gait.

## All runs summary

| Run | Config | Key result |
|---|---|---|
| 1-4 | Various AMP configs, Kp=200 | Standing only. Body +X ≠ anatomical forward. |
| 5 (night 1) | Axis fix + RSI + no pelvis term + 5s episodes | Fake walking (deer slid on ground) |
| 6 | + anatomy-aligned rewards | Breakthrough at iter 800, regressed (noise exploded) |
| 7 | + lower entropy warm-start from r6 | Consolidated at ep_len 228, tilt 19% |
| 8c | + soft actuators Kp=60 Kd=4 | Similar to r7 but more robust |
| **9** | **+ action_scale=1.0 + entropy=0** | **ep_len 237, tilt 9%, trk_fwd 2.33** |

## Process state

No training processes running. All checkpoints preserved.
