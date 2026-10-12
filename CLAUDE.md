# CLAUDE.md — aniML World

Quick-reference for Claude sessions. Most of the domain context lives in
`.cursor/rules/*.mdc` (auto-loaded) and `ARCHITECTURE.md` (ecosystem design). Older plans
and session logs are in `archive/`.
This file covers gotchas discovered during AMP training iterations.

## Running training

- Training launches via `isaaclab.bat -p <script>` from `C:\isaac-lab\IsaacLab`. Never just `python`.
- Required env vars: `PYTHONPATH=<repo>`, `ANIML_DEER_USD=<path>`, `ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1`.
- Actuator gains tuned via env: `ANIML_DEER_STIFFNESS` / `_DAMPING` (legs), `ANIML_DEER_SOFT_STIFFNESS` / `_SOFT_DAMPING` (spine + neck + skull).
- Launch in background with `nohup ./isaaclab.bat -p ... > run.log 2>&1 &` — the shell exits immediately, `kit.exe` runs independently.
- Kill training: `powershell Get-Process kit -EA SilentlyContinue | Stop-Process -Force`.
- Check training state: `powershell Get-Process -Name kit -EA SilentlyContinue` (empty = not running).

## Dependencies (applied in Isaac Lab's Python)

- ~~`rsl-rl-lib==3.0.0` pinned — amp-rsl-rl 1.2.0 needs legacy `ActorCritic`~~ **STALE (fixed 2026-07-22).** That pin applied to amp-rsl-rl **1.2.0** only. **amp-rsl-rl 2.2.3 requires just `rsl-rl-lib>=3.0.0` and runs fine on the shipped 5.0.1 — NO downgrade needed, AMP and loco coexist.** Believing the old pin cost us a long detour; don't reintroduce it.
  - Porting 1.2.0 → 2.2.3 needed 3 repo-side changes (all done): (1) add `actor`/`critic` config sections — 2.x branches on `RSL_RL_V4_PLUS` and reads `train_cfg["actor"]`/`["critic"]` (`MLPModel`, keys `hidden_dims`/`activation`/`obs_normalization`/`distribution_cfg`; **unknown keys are silently dropped**, so `init_noise_std` must live in `distribution_cfg` as `init_std`); (2) add an `"actor"` key to `obs_groups` (rsl_rl ≥4 renamed the actor obs set `policy` → `actor`; missing it = `KeyError: 'actor'` from `mlp_model.py`); (3) the hardcoded 0.5/0.5 task/style blend is now the native **`style_weight`** config key — the `--task_reward_ratio` monkey-patch in `train_amp.py` is obsolete.
- **AMP obs must match the reference dataset exactly.** `DEER_URDF_JOINT_NAMES` must list only the **15** joints in `deer_walk.npy`/`deer_run.npy`, in the same order, with `preserve_order=True` on the `SceneEntityCfg`. The v2 rig's 6 extra joints (spine yaw, 4× hip abduction, neck yaw) have no reference data; including them gave AMP obs 48 vs dataset 36 → `torch.cat` size 96-vs-72 error. Without `preserve_order` the columns silently shuffle (no error, garbage style learning).
- Install into Isaac Lab's Python: `cd /c/isaac-lab/IsaacLab/_isaac_sim && ./python.bat -m pip install <pkg>`.

## URDF body frame is NOT Isaac Lab's convention

The deer URDF has:
- Body `+X` = anatomical LEFT  (HL_thigh at +X, HR_thigh at -X)
- Body `+Y` = anatomical DOWN   (feet at +Y ~0.6 below pelvis)
- Body `-Z` = anatomical FORWARD (skull at -Z ~0.94)

Isaac Lab's stock `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` assume body +X = forward — **wrong for this URDF**. Use the anatomy-aligned versions in `training/envs/deer_rewards.py`: `track_forward_vel_anatomical_exp`, `track_lateral_vel_anatomical_exp`, `track_yaw_rate_anatomical_exp`, `forward_velocity_linear_anatomical`.

When spawning mid-stride via RSI, root velocity must be expressed along the *actual* facing direction (world -Y after the Rx(-90°) spawn rotation), not world +X.

## DCMotorCfg regex gotcha

Stiffness/damping dict keys are regex-matched against joint names, and **multiple matches on one joint = ValueError**. Can't combine `.*` with specific names. Use mutually exclusive patterns, e.g. `.*(thigh|shin|foot).*` for legs + explicit names for spine/neck.

## amp-rsl-rl quirks

- `discriminator.loss_type` must be `"BCEWithLogits"` not `"BCEWithLogitsLoss"`.
- Reward blend is hardcoded `0.5*task + 0.5*style` in `amp_on_policy_runner.py:384`. Override in `train_amp.py` via `--task_reward_ratio` (monkey-patches discriminator.predict_reward and env.step).
- Discriminator weights are **not** in the checkpoint — auto-fresh on warm-start. Usually desirable (policy/discriminator co-adapt from scratch).
- `rsl_rl` v3 **preserves** checkpoint `std_param` on resume; config's `init_noise_std` is IGNORED on warm-start.
- `entropy_coef=0` → AMP discriminator saturates to accuracy=1.0, style reward drops to ~0, policy freezes into a local optimum. Use 0.002–0.005 range.
- `entropy_coef=0.01` ballooned noise_std from 1.0 → 2.5 over 3000 iters and destroyed learned gaits.

## Training heuristics (hard-won)

- `action_scale=0.5` caps joint offsets at ±0.5 rad (±28°); expert walk needs knee bends of ~60°. Use `action_scale=1.0`.
- Soft spine/neck (Kp=5, Kd=0.5) lets the body passively sway; stiff legs (Kp=60, Kd=4) keep push-off. Uniform stiffness looks robotic.
- Orientation-based fall termination (`base_orientation_too_tilted`, 60°) is better than `pelvis_too_low` — height-based lets a fallen-sliding deer run out the episode clock.
- RSI via `reset_from_amp_motion` requires a baked `.pt` — rebuild with `python tools/bake_reset_states.py` after motion data or axis changes.

## Monitoring tools

- `python tools/amp_status.py --experiment <name>` — quick last-20 average of reward / ep_len / tilt / AMP metrics. Use this, not stdout parsing.
- TensorBoard event accumulator pattern works for custom trajectory tables (see `amp_status.py`).

## Task registration pattern

New AMP task workflow:
1. New env cfg: `@configclass class DeerUsdWalkAMPEnvCfgVN(DeerUsdWalkAMPEnvCfgVN-1)` in `training/envs/deer_usd_amp_env_cfg.py`.
2. New runner cfg: Python dict `DEER_USD_AMP_WALK_CFG_VN` in `training/agents/deer_usd_amp_cfg.py` (NOT dataclass — amp-rsl-rl needs a plain dict).
3. Register in `training/tasks/__init__.py` with `amp_rsl_rl_cfg_entry_point` kwarg.
4. Launch with `--task AniML-Deer-USD-Walk-AMP-vN`.

## AMP failure-mode fingerprints

- Style reward collapses to ~0 + policy stops improving → discriminator saturated; lower AMP reward_scale or add entropy.
- ep_len pegged at max + err_vel_xy below threshold + style very low → deer is sliding on ground, not walking. Check with `play_amp.py`; re-enable termination if killed.
- Leg-activity reward tiny (~2/episode) despite leg tracking looking ok → policy found rigid-lever hop. Bump leg_activity weight to match task reward magnitudes.
- **Style_r stuck at ~0.002 across many warm-starts** → check the .npy's `root_position` axis convention. After `Rx(-90°)` spawn rotation, the deer faces world -Y, so the .npy must put `root_position` along world -Y (matches `bake_reset_states.py`). Body-frame velocity should land at body -Z. The original converter wrote +X by mistake — see r15 entry in archive/logs/SESSION_LOG.md.

## PhysX silent-death pattern (Isaac Sim 5.1 + RTX 50-series)

`Exception: Failed to get DOF velocities from backend` from `omni/physics/tensors/impl/api.py` is downstream of NaN propagation in the cooked GPU PhysX articulation cache. Root causes seen:
- `RigidBodyPropertiesCfg(max_linear_velocity=1000, max_angular_velocity=1000)` — Isaac Lab quadruped default that lets PhysX yank self-colliding bodies apart at NaN-inducing speeds. **Cap to 50 / 30 in `deer_usd_articulation.py`.**
- The crash writes degenerate state to on-disk caches (`%LOCALAPPDATA%\ov\cache`, `...\NVIDIA\DXCache`), so subsequent runs die earlier than the last. **Clear those caches between investigation passes** if you see the "each run dies sooner" pattern.
- Reboot does NOT fix this — corruption is on disk, not in RAM.
- Smoke tests (small num_envs, few iters) hide the problem. Full training runs at 6000+ envs surface it.

## Ghost-process trap

When killing training, `Get-Process kit | Stop-Process` is NOT enough — the parent `python.exe` wrapper that launched kit can hold up to 8 GB of GPU/system memory. Always also `Get-Process python | Where { $_.WorkingSet64 -gt 500MB } | Stop-Process` before relaunching. Otherwise the next launch's VRAM read includes the leftover and looks much higher than reality (cost me hours of "OOM at 8000 envs" misdiagnosis until I noticed VRAM dropped from 15.5 GB to 569 MiB after killing the orphan python procs).

## Playback (`play_amp.py`) currently broken

As of r15 (2026-04-26), playback dies silently in env init with the `terrain_importer + FabricManager mismatched prototypes` warnings as the last log line. Both viewer and `--headless` modes affected. Training works fine on the same machine state. Not yet diagnosed; possibly related to `runner.load()` reshape from 8000-env trained checkpoint to 4-env playback, or a viewport-init code path that runs even in headless. Metric-based validation (style_r, ep_len, etc.) is reliable.

## Long-horizon runs need TWO independent failsafes (learned the hard way 2026-07-30)

A queue script and an event-based monitor are **the same failsafe**, not two. Both key off
the queue emitting output. When `.run_queue19.sh` hung, it printed nothing, the monitor
had nothing to fire on, and the silence was indistinguishable from healthy progress. The
D2b smoke had passed cleanly and the full run simply never started; it sat idle until
Raza asked.

The bug itself is worth remembering too:

    wait_idle () { while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader                    | grep -qE "[1-9][0-9]? ?%|100 ?%"; do sleep 60; done; }

This machine idles at **1-2%** from desktop compositing, so `"2 %"` matches `[1-9]` and
the wait never ends. Every earlier queue carried the same bug and survived only because
the GPU happened to read exactly `0 %` at the moment it looked.

Rules:
1. **Never gate on GPU utilisation.** Test whether the training PROCESS is alive:
   `Get-Process python | Where WorkingSet64 -gt 800MB`. Gate on the thing itself, not a
   noisy proxy for it.
2. **Run a WATCHDOG alongside the event monitor.** The event monitor reports things that
   happen; the watchdog reports things that STOP happening. It must assert progress
   positively -- log bytes growing, or a trainer alive -- and alarm on their absence.
   Silence is not evidence of health.
3. **Poll the state yourself periodically on multi-hour work.** Do not outsource all
   liveness detection to the automation being monitored.
