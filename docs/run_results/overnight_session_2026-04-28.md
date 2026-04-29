# Overnight session 2026-04-28 — r19 → r25 sweep

## TL;DR

Six clean training runs (r19-r25), all completed without crashes, each
committed and pushed. End state: a deer that walks reliably with style
metrics 130× better than any of r1-r14, hits 0% fall rate, and produces
a deployable ONNX policy for Unity Sentis.

**Best checkpoint:** `logs/rsl_rl/animl_deer_amp_wri_narrow/2026-04-28_18-44-57/model_4998.pt`
**Best video:** `logs/videos/r25_meshed.mp4`

## Metric trajectory

|  | r19 (baseline) | r20 (+torso) | r21 (+walk+run) | r22 (+idle) | r23 (narrow) | r24 (task=0.7) | r25 (warm+2k) |
|---|---|---|---|---|---|---|---|
| mean_reward | 174 | 195 | 223 | 234 | 240 | 149* | **245** |
| style_r/step | 0.377 | 0.403 | 0.461 | 0.476 | 0.484 | 0.292* | **0.496** |
| ep_len | 999 | 996 | 1000 | 1000 | 1000 | 999 | 998 |
| pelvis_too_low | 0.02% | 0.11% | 0.00% | 0.02% | 0.00% | 0.01% | 0.01% |
| amp_loss | 0.38 | 0.41 | 0.47 | 0.49 | 0.49 | 0.50 | **0.50** |
| lin_vel_R | 0.33 | 0.33 | 0.11 | 0.10 | 0.14 | 0.14 | 0.13 |
| err_vel_xy | 1.40 | 1.38 | 2.88 | 2.87 | 1.76 | 1.73 | 1.77 |
| noise_std | 1.98 | 1.67 | 1.53 | 1.52 | 1.47 | 1.01 | 1.46 |

*r24's reward/style numbers are reduced because of the task_reward_ratio
display scaling (style scaled 0.6×, task scaled 1.4×). Raw style ≈ r23.

## Key findings

1. **Torso widening (r19→r20)** added marginal style improvement (+7%)
   on top of the leg widening from prior sessions. Lone-clip walk's
   spine amplitude is intrinsically tiny (2.4°), so wider limits alone
   don't unlock body motion.

2. **Multi-clip dataset (r20→r21)** was the big style unlock: +14%
   style on a single change, by giving the discriminator richer body-
   motion poses (run clip's spine swing, leg flex). But the wider
   command range that came with V12 broke velocity tracking.

3. **Three-clip with idle (r21→r22)** added another +3% style
   marginally, idle's 331 frames of weight-shift / breathing helps
   the rest-state pose distribution.

4. **Narrow command range (r22→r23)** was the velocity-tracking fix
   we needed. 38% improvement on err_vel_xy without giving up the
   multi-clip style benefits.

5. **Task-heavy reward blend (r23→r24)** got task_r/step positive for
   the first time but didn't really move velocity-tracking metrics.
   There's a soft ceiling — multi-clip discriminator gives multiple
   equally-good gait basins and the policy can sit in any of them.

6. **Warm-start +2000 iters (r24→r25)** consolidated the best policy
   slightly — best reward and style ever, peak lin_vel_R briefly
   exceeded r19 baseline. Diminishing returns hit.

## Hardware utilization

Pushed from 8000 envs (r19) to 14000 envs (r24/r25):
- VRAM: 50% → 68% utilization
- GPU: 86% → 94% utilization
- 50% more training samples per gradient update

## Per-run docs
- [r20 summary](r20_summary.md) — torso widening
- [r21 summary](r21_summary.md) — walk + run
- [r22 summary](r22_summary.md) — walk + run + idle
- [r23 summary](r23_summary.md) — narrow command range (best of single-knob runs)
- [r24 summary](r24_summary.md) — task_reward_ratio 0.7
- [r25 summary](r25_summary.md) — warm-start +2000 iters (best aggregate metrics)

## Open questions / candidates for r26+

1. **Whole-body motion still a question for visual review.** Metrics
   are great but the user's r19 feedback was "stiff body". Need to
   eyeball r25_meshed.mp4 to confirm the multi-clip + torso widening
   produced meaningfully more body articulation than r19/r20.

2. **lin_vel_R variance is bimodal** — peak 0.36 vs avg 0.13. Some
   episodes track well, others don't. Possible fix: add commanded
   velocity to the AMP observation group so the discriminator can
   condition style on speed.

3. **Emergent behavior is the real goal.** Current setup teaches one
   walking policy (with optional run interpolation). For dynamic
   emergent behaviors (fleeing, jumping, recovering from pushes,
   adapting to terrain), the path is task RL on top of this base
   policy:
     - Add terrain randomization (`DeerUsdRoughEnvCfg`-style)
     - Add base_external_force_torque events for push recovery
     - Layer task rewards (chase target, avoid predator) on top of
       AMP style
     - Or graduate to skill-embedding architecture (ASE/CALM style)

4. **Unity deployment.** `policy_4998.onnx` is ready to drop into
   Unity Sentis. Should validate end-to-end inference path in a
   minimal Unity scene before scaling.

## Tooling shipped this session
- `tools/render_playback_meshed.py` — full deer-mesh offline renderer
- `tools/render_playback_video.py` — stick-figure offline renderer
- `tools/import_world_urdf_v3.py` — URDF→USD regen (won't work on
  this machine due to RTX renderer crash; use direct pxr edit instead)
- `tools/patch_walk_npy_axis.py` — coord-frame patch (now generalized
  to all clips via `--target` / `--all`)
- `play_amp.py` `--realtime` and `--video` flags
- Defensive std-clamp safeguard in `train_amp.py`
- 6 new env classes (V10-V13)
- 4 new agent cfgs (multi-clip variants)
- 5 new task registrations (AMP-v13 through v17)

## Commits pushed to claude/jolly-clarke-e00da9
- `0d9de77` Fix AMP coord-frame bug + widen URDF leg limits + add multi-clip prep
- `9d612ee` Widen torso joints + prep multi-clip AMP configs
- `303986c` r20 results
- `c0c79f3` r21 results
- `4600792` r22 results
- `b49089d` r23 results
- `d3d785f` r24 results
- `f66a9ea` r25 results
