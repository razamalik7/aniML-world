# aniML World — Session Log

**Purpose:** continuous record of what we're building, what we've tried, and
what's currently in flight. Open this first when resuming on a new machine
or after a long break.

---

## Index

1. [Project Overview](#project-overview)
2. [Current State (At A Glance)](#current-state-at-a-glance)
3. [How To Resume On A New Machine](#how-to-resume-on-a-new-machine)
4. [Key Files & Where To Look](#key-files--where-to-look)
5. [Run History](#run-history)
6. [Known Gotchas](#known-gotchas)
7. [Backlog / Ideas Deferred](#backlog--ideas-deferred)

---

## Project Overview

We're teaching a custom **deer USD model** to walk in NVIDIA Isaac Lab using
**Adversarial Motion Priors (AMP)** layered on PPO via the `amp-rsl-rl`
package. The reference motion is a Unity `.anim` walk clip converted to
`.npy` via `tools/convert_unity_anim_to_npy.py`.

### The stack
- **Isaac Sim 5.1 + Isaac Lab 2.3.x** (installed at `C:\isaac-lab\IsaacLab`)
- **amp-rsl-rl 1.2.0** (needs `rsl-rl-lib==3.0.0` pinned — newer versions removed `ActorCritic`)
- **Hardware:** RTX 5070 Laptop GPU, 8 GiB VRAM
- **Repo root:** `C:\Users\nars7\Downloads\aniML_world`

### The URDF has weird axes (memorize this or get confused)
- Body **+X** = anatomical **LEFT**
- Body **+Y** = anatomical **DOWN** (feet are at +Y ~0.6m below pelvis)
- Body **−Z** = anatomical **FORWARD** (skull at −Z ~0.94m)

Isaac Lab's stock `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` assume
body +X = forward. That is **wrong** for this URDF. We ship anatomy-aligned
rewards in `training/envs/deer_rewards.py` (see `track_forward_vel_anatomical_exp`
etc.). After the `Rx(−90°)` spawn rotation the deer faces **world −Y**,
not world +X.

### The core loop
1. Define an env cfg subclass in `training/envs/deer_usd_amp_env_cfg.py`
   (versioned `DeerUsdWalkAMPEnvCfgV1..VN`).
2. Define an agent cfg **dict** (NOT dataclass — amp-rsl-rl wants plain dict)
   in `training/agents/deer_usd_amp_cfg.py` (`DEER_USD_AMP_WALK_CFG_VN`).
3. Register a gym task in `training/tasks/__init__.py`
   (`AniML-Deer-USD-Walk-AMP-vN`).
4. Launch with `training/scripts/train_amp.py --task ...`.
5. Inspect with `tools/amp_status.py --experiment ...` or visually with
   `training/scripts/play_amp.py`.

---

## Current State (At A Glance)

**As of 2026-04-26 (late evening):** 15 training runs completed. **Run 15
is a major breakthrough** — first run where the AMP discriminator
actually trained, with style_r/step ~70× higher than every prior run.

**Diagnosis of the prior 14-run saturation:** the .npy reference motion's
`root_position` was advancing along world **+X** while the policy's
runtime base-frame velocity faces world **−Y** post-spawn-rotation. After
`AMPLoader`'s `R.T @ v_world` transform, the expert's `base_lin_vel_b`
landed on body **+X** while the policy's lands on body **−Z**. That is a
**constant 3-vector mismatch** the discriminator could drive its
expert-vs-policy decision off without ever inspecting the gait cycle —
explains the chronic style_r ≈ 0.002 across r1-r14. Bake_reset_states.py
already had the right convention (world −Y); the bug was only in
`tools/convert_unity_anim_to_npy.py:418-434`. Fix landed via
`tools/patch_walk_npy_axis.py` + converter source update.

**New machine** (this session): RTX 5080 / 16 GB VRAM / Core Ultra 9 285K,
~2-3× the prior compute. Throughput at 8000 envs hit **147k steps/s**,
~7× the laptop's 21k. Iter pace stable at ~42-46 iter/min.

**New unrelated bug discovered:** Isaac Sim 5.1 has intermittent silent
deaths on this hardware — `Exception: Failed to get DOF velocities from
backend` from `omni.physics.tensors`, downstream of NaN propagation
through the cooked GPU PhysX articulation cache. Made worse by
`max_linear_velocity=1000.0` / `max_angular_velocity=1000.0` defaults
that let PhysX yank self-colliding bodies apart at NaN-inducing speeds.
Fixed by tightening to 50 / 30 in
`training/envs/deer_usd_articulation.py`. Each crash also wrote
degenerate state to disk caches (`%LOCALAPPDATA%\ov\cache`,
`...\NVIDIA\DXCache`), causing subsequent runs to die earlier — clearing
those caches plus the velocity-cap fix together kept r15 stable for full
training.

**Active issue:** Headless and viewer playback (`play_amp.py`) still
crashes silently in env init — same `terrain_importer + FabricManager`
warning pattern then process death. Training works; playback path has
some additional fragility. Not yet diagnosed.

**Next likely run (r16):** investigate playback crash, then consider:
(a) adding `Deer_001_run.anim` as a second motion clip for a
2-gait library, (b) testing the policy under varied terrain via
`DeerUsdRoughEnvCfg`, (c) extending to 5000+ iters since training is
stable now.

### Currently running / queued
- **r14 playback** running in Isaac Sim (process started in current session).
- No training process currently running.

---

## How To Resume On A New Machine

### Prerequisites
1. Windows laptop with NVIDIA GPU (≥8 GiB VRAM recommended).
2. **Isaac Sim 5.1 + Isaac Lab** installed at `C:\isaac-lab\IsaacLab`. See
   Isaac Lab's installer docs.
3. Inside Isaac Lab's bundled Python, install pinned deps:
   ```bash
   cd /c/isaac-lab/IsaacLab/_isaac_sim
   ./python.bat -m pip install rsl-rl-lib==3.0.0 amp-rsl-rl==1.2.0
   ```
4. Clone this repo:
   ```bash
   git clone https://github.com/razamalik7/aniML-world.git
   cd aniML-world
   ```
5. Set env vars each shell session (or put in `.envrc` / profile):
   ```bash
   export PYTHONPATH="$(pwd)"                # or "C:\Users\...\aniML_world"
   export ANIML_DEER_USD="$(pwd)/assets/deer_world.usd"
   export ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
   ```
6. **Checkpoints are NOT in git** (`logs/` is gitignored — too large).
   To resume training from a specific historical checkpoint, either:
   (a) rsync the `logs/rsl_rl/<exp>/<run>/model_*.pt` from the source machine, or
   (b) re-train from scratch, or
   (c) skip warm-start and start fresh.

### Everyday commands
```bash
# Start a training run (headless, in background)
cd /c/isaac-lab/IsaacLab
nohup ./isaaclab.bat -p C:\path\to\aniML_world\training\scripts\train_amp.py \
    --task AniML-Deer-USD-Walk-AMP-v12 \
    --num_envs 2500 \
    --headless \
    --resume_from C:\path\to\aniML_world\logs\rsl_rl\<exp>\<run>\model_N.pt \
    > /path/to/train.log 2>&1 &

# Watch metrics
python tools/amp_status.py --experiment animl_deer_amp_walk_v12

# Visual playback of a checkpoint
./isaaclab.bat -p C:\path\to\aniML_world\training\scripts\play_amp.py \
    --task AniML-Deer-USD-Walk-AMP-v12 \
    --num_envs 4 \
    --checkpoint <path/to/model_N.pt>

# Kill everything (ghost kit.exe processes are a recurring issue)
powershell Get-Process kit -EA SilentlyContinue | Stop-Process -Force
```

---

## Key Files & Where To Look

| File | What's in it |
|---|---|
| `CLAUDE.md` | Tight gotchas reference for Claude sessions — read first |
| `SESSION_LOG.md` | This file — big-picture state + run history |
| `ARCHITECTURE.md` | System diagram & design decisions |
| `AMP_PLAN.md` | Original strategic plan for the AMP rollout |
| `PROGRESS.md` | Earlier progress notes (pre-AMP locomotion work) |
| `training/envs/deer_usd_amp_env_cfg.py` | All `DeerUsdWalkAMPEnvCfgVN` env subclasses |
| `training/envs/deer_rewards.py` | All custom reward / termination functions |
| `training/envs/deer_usd_articulation.py` | DCMotorCfg with per-joint Kp/Kd |
| `training/agents/deer_usd_amp_cfg.py` | All `DEER_USD_AMP_WALK_CFG_VN` agent dicts |
| `training/tasks/__init__.py` | gym.register() lines for each AMP-vN task |
| `training/scripts/train_amp.py` | Main training driver (has `--resume_from`, `--task_reward_ratio`) |
| `training/scripts/play_amp.py` | Playback with auto-latest-checkpoint |
| `tools/convert_unity_anim_to_npy.py` | Unity `.anim` → `.npy` motion conversion |
| `tools/bake_reset_states.py` | Pre-bake RSI states from motion `.npy` |
| `tools/amp_status.py` | Last-20-iter summary from TensorBoard events |
| `assets/motions/deer_walk.npy` | AMP reference motion |
| `assets/motions/deer_walk_reset_states.pt` | Baked RSI states |

---

## Run History

Keep this table append-only. **Latest runs first.**

### Run 15 — 2026-04-26 (BREAKTHROUGH: AMP discriminator finally trains)

- **Hardware**: NEW machine — RTX 5080 (16 GB VRAM) / Core Ultra 9 285K /
  32 GB RAM. ~2-3× the prior laptop's compute.
- **Task:** `AniML-Deer-USD-Walk-AMP-v0` (clean ablation — wanted the
  bare-bones config to isolate the bug-fix effect; v12's heavy reward
  stack carries r1-r14 baggage).
- **Env cfg:** `DeerUsdWalkAMPEnvCfg` (the original lean version)
- **Agent cfg:** `DEER_USD_AMP_WALK_CFG` (original; entropy 0.01,
  init_noise_std 1.0, fresh start — no warm-start from any prior run).
- **Iters:** 3000 (paused at 2500 for user gaming, resumed via
  `--resume_from model_2500.pt`, completed last 500 cleanly).
- **Num envs:** **8000** (the new machine ceiling — VRAM at 51%, GPU
  util 88%; previous laptop was capped at 2500).
- **Throughput:** **147k steps/s** (vs 21k on the laptop).
- **Final stats (last 20 iters of completed run):**
  - mean_reward **184** (peak 194)
  - **style_r/step 0.389** (peak **0.605**) ← vs r14's 0.003
  - ep_len 1000 / 1000 (max — deer never falls)
  - pelvis_too_low **0.0%** (vs r14's 1%)
  - amp_loss **0.40** (real discriminator gradient throughout)
  - noise_std 1.47 (healthy exploration without drift)
- **Final checkpoint:**
  `logs/rsl_rl/animl_deer_amp_walk/2026-04-26_23-26-43/model_2999.pt`
  (+ matching `policy_2999.onnx` for Unity Sentis).

#### Bug fixes that enabled this run

1. **Coord-frame fix in motion data** (the headline fix):
   - Before: `tools/convert_unity_anim_to_npy.py:418-434` wrote
     `root_position` along world +X with constant Rx(-90°) quaternion.
     `AMPLoader.R.T @ v_world` then put the expert's forward velocity at
     body +X.
   - But the deer's actual sim runtime velocity (URDF body -Z =
     anatomical forward, mapped through Rx(-90°) spawn rotation) sits at
     body -Z.
   - Discriminator could decide expert vs policy off this constant
     3-vector slot — explains why r1-r14 all had style_r ≈ 0.002.
   - Fix: converter source updated, existing .npy patched in-place via
     `tools/patch_walk_npy_axis.py` (backup at
     `assets/motions/deer_walk.npy.preaxisfix.bak.npy`).
   - Validation: `R.T @ (0, -0.8, 0)` now yields `(0, 0, -0.8)` matching
     Isaac Lab's `root_lin_vel_b`.

2. **PhysX velocity cap fix** (debugging side-quest):
   - Before: `RigidBodyPropertiesCfg(max_linear_velocity=1000.0,
     max_angular_velocity=1000.0)` — Isaac Lab quadruped boilerplate
     defaults. With self_collisions=True, PhysX yanked overlapping
     mesh fragments apart at NaN-inducing speeds.
   - Effect: silent training/playback crashes with
     `Exception: Failed to get DOF velocities from backend`. Each crash
     wrote degenerate state to on-disk PhysX cooked caches (in
     `%LOCALAPPDATA%\ov\cache` and `...\NVIDIA\DXCache`), causing the
     next run to die earlier than the last.
   - Fix: capped to 50 m/s / 30 rad/s in
     `training/envs/deer_usd_articulation.py`. Plus cleared corrupted
     caches before relaunch.
   - Symptom hash: silent kit.exe death, no Python traceback (until you
     get the late traceback from one specific run that tipped past the
     C++/Python boundary cleanly).

3. **Restored deleted USD payloads:** commit `876e0ae` had pruned
   `configuration/deer_world_{base,physics,sensor,robot}.usd` as
   "experimental USDs"; they were actually the live payloads that
   `deer_world.usd` wraps. Restored from `6624f98^!`.

#### Open issues from this run

- **Playback (`play_amp.py`) crashes silently** even after the velocity-cap
  fix. Both viewer and `--headless` modes die mid-init at the
  `terrain_importer + FabricManager mismatched prototypes` warning
  step. Training survived this same warning fine, but playback's load
  path (4 envs vs 8000-env trained checkpoint, plus possibly
  `runner.load()` reshape) hits some additional fragility. Needs
  investigation before we can visually verify the gait quality. Metric-
  based evidence is overwhelming, but the visual is still TBD.
- **Two intermittent multi-minute stalls during training**:
  iter 1300→1400 took 47 min, iter 2200→2300 took 1h 54min, while
  every other 100-iter chunk was 130-160s. Pattern is fully recoverable
  (pace returns to normal after stall). Most likely Windows Defender
  scan, telemetry pause, or a transient PhysX hiccup. Total stall time:
  ~2h 42min added to what should have been a 70-min run.

#### Useful artifacts produced

- `tools/patch_walk_npy_axis.py` — one-off .npy axis-swap patch with
  pre/post-fix value validation. Won't be needed again unless we
  regenerate the .npy from a fresh Unity export.
- `assets/motions/deer_walk.npy.preaxisfix.bak.npy` — backup of the
  pre-fix .npy. Safe to delete once we trust the fix.

### Run 14 — 2026-04-21
- **Task:** `AniML-Deer-USD-Walk-AMP-v12`
- **Env cfg:** `DeerUsdWalkAMPEnvCfgV9` (inherits V8; drops `nose_pitch_penalty`, adds `head_height_ramp` on skull body)
- **Agent cfg:** `DEER_USD_AMP_WALK_CFG_V12` (inherits V11; entropy 0.001, seed 53)
- **Warm-start:** r13 `model_11996.pt`
- **Iters:** 3000 new (cumulative 11997 → 14996)
- **Num envs:** 2500 (5070 full utilization)
- **Final stats:** mean_reward **56.7** (best-ever numerically), task_r/step 0.48, style_r/step 0.003 (peak 0.046 — first life signal from disc), ep_len 236, pelvis_too_low 1%, noise_std 0.71.
- **Motivation:** r13 playback showed deer with level pelvis + bent neck + head-down. The r13 `nose_pitch_penalty` was measuring pelvis-axis orientation — pelvis was level so penalty didn't fire, but the neck was folded. Switched to rewarding **skull world-Z** directly.
- **Checkpoint:** `logs/rsl_rl/animl_deer_amp_walk_v12/2026-04-21_14-12-45/model_14995.pt`
- **Visual:** under review at time of writing.
- **Learned:** (TBD after playback — will update with whether gait finally appeared or next failure mode emerged.)

### Run 13 — 2026-04-16
- **Task:** `AniML-Deer-USD-Walk-AMP-v11`
- **Env cfg:** `DeerUsdWalkAMPEnvCfgV8` (linear `pelvis_height_ramp`, `nose_pitch_penalty`, `pelvis_too_low` tightened 0.55 → 0.70 m)
- **Agent cfg:** `DEER_USD_AMP_WALK_CFG_V11` (entropy 0.001, seed 52)
- **Warm-start:** r12 `model_8997.pt`
- **Iters:** 3000 new (cumulative 8998 → 11997)
- **Final stats:** mean_reward 48.9, task_r/step 0.43, style_r/step 0.002 (peak 0.005), ep_len 225, pelvis_too_low term 2% (down from peak 98%), noise_std 0.65 (actually dropped from r12's 0.72 — entropy 0.001 finally clamping drift).
- **Motivation:** r12 produced the "crouch-shuffle" — pelvis hovering in the 0.55–0.70m gap between termination and binary upright_bonus. Replaced binary with linear ramp + added pitch penalty + tightened cutoff. Policy adapted fast (98% → 2% termination rate).
- **Checkpoint:** `logs/rsl_rl/animl_deer_amp_walk_v11/2026-04-16_15-45-30/model_11996.pt`
- **Visual:** crouched-prop pose — pelvis level (cleared tilt-35° AND pitch penalty), front legs splayed forward as static struts, neck folded down so skull / antlers graze ground. Front knees fully rigid.
- **Learned:** `nose_pitch_penalty` measured the wrong body. It watches pelvis `projected_gravity_b[:, 2]` — when pelvis is level and only neck is bent, this is ~0 and penalty doesn't fire. Need skull-height reward, not base-pitch penalty.

### Run 12 — 2026-04-16
- **Task:** `AniML-Deer-USD-Walk-AMP-v10`
- **Env cfg:** `DeerUsdWalkAMPEnvCfgV7` (tilt 60° → 35°, `pelvis_too_low` z<0.55, binary `upright_bonus` at z>0.70, num_envs 1500 → 2500)
- **Agent cfg:** `DEER_USD_AMP_WALK_CFG_V10` (entropy 0.002, seed 51, disc stayed [128,128] — tried shrinking but warm-start incompatible)
- **Warm-start:** r11b `model_5998.pt`
- **Iters:** 3000 new (cumulative 5999 → 8998)
- **Final stats:** mean_reward 49.1, task_r/step 0.43, style_r/step 0.002, ep_len 225, pelvis_too_low 0%, noise_std 0.72 (drift continued despite entropy drop).
- **Motivation:** r11b visual: belly-sliding / motorboat. Wanted to kill it with geometric constraints. Tilt term 60° → 35° + height term. Also first run with 2500 envs: throughput went 13.5k → 20.7k steps/s.
- **Checkpoint:** `logs/rsl_rl/animl_deer_amp_walk_v10/2026-04-16_14-46-08/model_8997.pt`
- **Visual:** crouch-shuffle. Pelvis sitting at ~0.60m, nose-down near 35°, front legs prop-extended, hind splayed — the "sweet spot" between the two binary cliffs of pelvis_too_low (0.55) and upright_bonus (0.70). Deer was visibly gaming both thresholds.
- **Learned:** binary thresholds 15cm apart create a campable gap. Need dense gradient (→ linear ramp in r13).

### Run 11b — 2026-04-16
- **Task:** `AniML-Deer-USD-Walk-AMP-v9`
- **Env cfg:** `DeerUsdWalkAMPEnvCfgV6` (adds `leg_joint_activity` weight 2.5 to V4's base)
- **Agent cfg:** `DEER_USD_AMP_WALK_CFG_V9` (entropy 0.005, seed 50)
- **Warm-start:** r9 `model_2999.pt`
- **Iters:** 3000 new
- **Final stats:** mean_reward 48.1, task_r/step 0.42, style_r/step 0.002, leg_joint_activity 15.2/ep, noise_std drifted 0.40 → 0.59.
- **Motivation:** r9/r10 produced "rigid-lever hop" — knees locked, deer bouncing forward. Added leg_joint_activity reward to punish static legs.
- **Visual:** belly-sliding. The leg_activity reward did its job TOO literally — legs thrashing maximally regardless of whether they touched ground. Deer found belly-down + leg-thrashing as a local optimum because the 60° tilt cutoff was too permissive.
- **Learned:** activity-based rewards don't distinguish productive motion from thrash. Need posture constraints (→ r12).
- **This run also:** we set the per-joint Kp split — legs Kp=60 Kd=4, spine+neck+skull Kp=5 Kd=0.5. Using `.*(thigh|shin|foot).*` regex + explicit spine/neck names because DCMotorCfg errors on multiple regex matches per joint.

### Runs 1–10 (earlier 2026-04-15 / early 2026-04-16)
High-level arc:
- **Runs 1–6:** initial AMP pipeline bring-up. Had `action_scale=0.5` (joint range capped at half expert) and stock body-+X-forward rewards (wrong for this URDF). Several failed warm-starts due to axis mismatch.
- **Run 6 specifically:** anatomy-aligned rewards introduced (`track_forward_vel_anatomical_exp` etc.) + re-baked RSI along world -Y.
- **Run 7 (v5 task, action_scale=1.0):** first action_scale=1.0 run. Let knees reach expert range.
- **Run 8 series:** entropy tuning. entropy=0.01 ballooned noise_std 1.0 → 2.5 → learned gaits destroyed. entropy=0 saturated discriminator. Settled on 0.002–0.005 sweet spot.
- **Run 9 (v7):** best warm-start anchor for many later runs. entropy=0, action_scale=1.0, from r8c.
- **Run 10 (v8):** warm-start r9 with entropy=0.002 + leg_joint_activity weight 1.0. Mild improvement (leg_activity 2.2/ep — too small to matter).

See CLAUDE.md "AMP failure-mode fingerprints" for the diagnostic patterns these runs calibrated.

---

## Known Gotchas

See **CLAUDE.md** for the comprehensive list. Highlights:

- `isaaclab.bat -p` is the ONLY way to run training scripts. Bare `python` misses kit plugins.
- `rsl-rl-lib` pinned to 3.0.0. Newer versions removed `ActorCritic` that amp-rsl-rl imports.
- DCMotorCfg regex keys must be **mutually exclusive** across joints. Combining `.*` with specific names raises `ValueError: Multiple matches for joint`.
- `train_amp.py` default `--num_envs` is now 2500 (was 1024 — patched 2026-04-21). Override on CLI if OOM.
- `amp-rsl-rl` uses a **single shared Adam optimizer** for actor-critic + discriminator. No separate disc LR. Gradient penalty `lambda_=10` hardcoded. Only disc tunables via cfg: `hidden_dims`, `reward_scale`, `loss_type`.
- Shrinking disc `hidden_dims` **cannot** be combined with warm-start from a larger-disc checkpoint — PyTorch `load_state_dict` errors on size mismatch even with `strict=False`.
- Discriminator weights **are** persisted in newer amp-rsl-rl checkpoints. (CLAUDE.md previously said they weren't — that's outdated.)
- After a training run crashes, check for ghost `kit.exe` via `Get-Process kit`.

---

## Backlog / Ideas Deferred

- **Foot-alternation reward.** Reward when contralateral feet (FL/HR vs FR/HL) have phase-offset contact pattern matching a walk. Most likely r15 fix — directly shapes gait, not just posture.
- **Train discriminator from scratch (no warm-start of disc weights).** Patch `train_amp.py` to skip disc state on resume. Would allow shrinking disc `hidden_dims` for anti-saturation, plus gives the disc a chance to un-saturate as policy improves.
- **Swap AMP reward_scale dynamically.** Start low (0.3) while disc is catching up, ramp to 1.0 once style_r sustains above ~0.02/step.
- **Dedicated Claude skill.** A `.claude/skills/isaac-lab-training.md` with launch incantations, decision trees for failure modes, and common edit recipes. ~30 min to set up, saves rederivation per session.
- **Isaac Sim MCP server.** Community MCPs let Claude query live sim state (body positions, joints) over MCP. Useful for "is the deer upright at iter 500" without spinning up play_amp.py. ~1 hr setup, quality varies.
- **Diagnostic script.** `tools/inspect_run.py --run <path>` that dumps per-reward-component stats, termination breakdown, joint-vel histograms, CoM-Z histogram in one command. Currently requires cobbling amp_status.py + log-tail + playback.
- **CLAUDE.md split.** Once > 300 lines, split into per-topic `.claude/rules/*.mdc` files.

---

## Continuous Update Protocol

**Every meaningful turn should update this file:**
- New run started → append to Run History (status: "in progress").
- Run completed → fill in final stats, visual, learned.
- Failure mode diagnosed → add to Run History entry + update Current State.
- New gotcha discovered → add to Known Gotchas or push to CLAUDE.md.
- Backlog item picked up → move to Run History.

Commit + push after each update so the other machine has latest context:
```bash
git add SESSION_LOG.md CLAUDE.md training/...
git commit -m "log: <one-line summary>"
git push
```
