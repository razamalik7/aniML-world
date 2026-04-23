# CLAUDE.md — aniML World

Quick-reference for Claude sessions. Most of the domain context lives in
`.cursor/rules/*.mdc` (auto-loaded) and `ARCHITECTURE.md` / `PROGRESS.md` /
`AMP_PLAN.md` / `NIGHT_SUMMARY.md`. Read those for the full picture.
This file covers gotchas discovered during AMP training iterations.

## Running training

- Training launches via `isaaclab.bat -p <script>` from `C:\isaac-lab\IsaacLab`. Never just `python`.
- Required env vars: `PYTHONPATH=<repo>`, `ANIML_DEER_USD=<path>`, `ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1`.
- Actuator gains tuned via env: `ANIML_DEER_STIFFNESS` / `_DAMPING` (legs), `ANIML_DEER_SOFT_STIFFNESS` / `_SOFT_DAMPING` (spine + neck + skull).
- Launch in background with `nohup ./isaaclab.bat -p ... > run.log 2>&1 &` — the shell exits immediately, `kit.exe` runs independently.
- Kill training: `powershell Get-Process kit -EA SilentlyContinue | Stop-Process -Force`.
- Check training state: `powershell Get-Process -Name kit -EA SilentlyContinue` (empty = not running).

## Dependencies (applied in Isaac Lab's Python)

- `rsl-rl-lib==3.0.0` pinned — amp-rsl-rl 1.2.0 needs legacy `ActorCritic` import; 4.x/5.x removed it. Isaac Lab ships 5.0.1, downgrade is safe (Isaac Lab itself only uses `VecEnv` from rsl_rl).
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
