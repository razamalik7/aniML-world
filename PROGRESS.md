# aniML World — Project Progress

> Last updated: 2026-04-15 (AMP integration night)

## What Is This Project?

aniML World is an intelligent ecosystem simulation where 50+ AI agents (deer, wolves, and eventually other species) are trained via reinforcement learning in NVIDIA Isaac Lab and deployed in a Unity 6 environment. The goal is a living, breathing virtual ecosystem where animals exhibit learned, emergent behaviors — locomotion, predation, reproduction — rather than hand-scripted animations.

The simulation uses small policy networks (1-5M parameters, MLP [512, 256, 128]) that process state vectors, trained with PPO (Proximal Policy Optimization) using continuous action spaces. A rule-based World Director manages ecosystem-level events. Each animal carries a heritable genome (16-32 genes) for morphology and behavior, enabling emergent evolution through natural selection.

### Tech Stack

| Component | Technology |
|-----------|------------|
| Training | NVIDIA Isaac Lab (Isaac Sim 4.5+), rsl_rl (PPO), PyTorch 2.5+ |
| Animal Model | Custom deer USD/URDF with 15 joints, 4 legs |
| Deployment | Unity 6 (URP), Unity Sentis for ONNX inference |
| Hardware | RTX 5070 Laptop (12GB VRAM, 8GB dedicated) |
| OS | Windows 10 (native, not WSL) |

### Directory Layout

```
configs/          YAML configs for species, training hyperparams, environment
training/         Isaac Lab Python code (envs, agents, rewards, scripts)
  envs/           Environment configs and custom reward functions
  agents/         PPO runner configurations
  scripts/        train.py, play.py entry points
  tasks/          Gymnasium task registration
models/           PyTorch network definitions + ONNX exports
world_director/   Rule-based event system and context tokens
unity/            C# scripts for Unity 6 deployment
shared/           Cross-language state/action schemas
tools/            VRAM budget calculator, benchmarks, utility scripts
assets/           Deer URDF, meshes, USD files
logs/rsl_rl/      TensorBoard logs and model checkpoints
```

---

## Progress Timeline

### Phase 0: Standing (Completed)

**Goal:** Get the custom deer mesh to stand reliably in Isaac Lab.

**Key challenges solved:**
- **Coordinate system mismatch**: The deer URDF uses Y-up (ROS convention) but Isaac Lab uses Z-up. Solved with an Rx(-90°) rotation in `init_state.rot = (0.7071, -0.7071, 0, 0)`.
- **USD orientation**: After URDF→USD conversion, the root transform needed to be set to identity (via `fix_deer_world_usd.py`) so Isaac Lab's `init_state.rot` controls orientation correctly.
- **Reward design**: Developed `leg_extension_reward` (pelvis-to-foot vertical gap), `feet_under_chest`, `base_height_l2`, and `flat_orientation_l2` to teach upright posture without the policy gaming termination conditions.
- **Self-collision**: Enabled for realism but disabled during standing-only training for performance.
- **Joint configuration**: 15 DOF articulation with PD control (stiffness=200, damping=6), default joint positions tuned for natural standing pose.

**Result:** Policy reliably stands in <200 iterations. Standing checkpoint (`model_199.pt`) used as warm-start for walking.

### Phase 0.5: Center of Mass Fix (Completed)

**Goal:** Fix the deer being "top heavy" and falling sideways during training.

**Key challenges solved:**
- **CoM calculation**: Wrote a script to compute volume-weighted geometric centroids from STL mesh data for every link in `deer_world.urdf`.
- **URDF patching**: Applied computed CoM values to each link's `<inertial><origin>` tag.
- **USD regeneration**: Re-converted patched URDF to USD, verified root transform was identity.
- **Orientation regression**: Had to carefully preserve the known-good Rx(-90°) spawn rotation after the CoM fix, which had been lost during USD regeneration.

**Result:** Dramatically improved stability. The deer's center of mass is now physically accurate, leading to natural weight distribution across all four legs.

### Phase 1: Walking — Initial Attempts (Completed, multiple iterations)

**Goal:** Train the deer to walk forward in response to velocity commands.

**Environment configuration (`DeerUsdWalkEnvCfg`):**
- Flat terrain, velocity commands 0-1.2 m/s
- 1500 parallel environments
- 20-second episodes (4000 steps at 0.02s step size)
- Curriculum: velocity tracking weight ramps from 0.3→1.0 over 500 iterations
- `pelvis_too_low` termination at 0.25m (actual falls only)
- `rel_standing_envs = 0.1` (10% get zero velocity commands)

**Training runs and learnings:**

| Run | Config | Result | Key Learning |
|-----|--------|--------|-------------|
| Walk from scratch (init_std=1.0) | Random init | Collapsed at iter ~40, ep_len stuck at 60 | Too much exploration noise causes chaotic torques → instant falls → entropy collapse |
| Warm-start (std=0.4, critic reset) | Patched standing ckpt | Collapsed at iter ~100, ep_len=64 | `ramp_leg_ext` (10→5 decay) weakened upright signal during critical learning period |
| Warm-start (std=0.4, fixed leg_ext=15) | Removed leg_ext decay | Still collapsed at iter ~100 | Discovered rsl_rl v4.0 overrides checkpoint std with `init_noise_std` from config |
| Warm-start (init_noise_std=0.05) | Matching April run's std | Collapsed at iter ~40, high dof_acc_l2 | CoM-fixed physics has different dynamics than old model |
| **Replicate 17-03-57** (init_noise_std=0.1, entropy=0.001) | Exact match of best historical run | **ep_len peaked at 686 (iter 3732)** | The key was matching effective std=0.10 + low entropy_coef=0.001 for slow std growth |

**Critical discovery — rsl_rl v4.0 std loading behavior:**
When `resume=True`, rsl_rl loads actor MLP weights from the checkpoint, but **overwrites** `distribution.std_param` with `init_noise_std` from the config. This means the checkpoint's learned per-joint standard deviation is discarded. To replicate a run's effective std, you must set `init_noise_std` in the config to the desired value explicitly.

### Phase 1 Continuation: Extended Training (Completed)

Resumed the best phase 1 run for an additional 2800 iterations (total 5600).

**Result:** ep_len plateau at 400-550, peak of 686. `pelvis_too_low` stabilized at ~0.59-0.71.

**Observation when visualized:** Deer stand in good posture, occasionally lift a foot, shuffle slightly, but **no forward locomotion**. The policy learned to balance perfectly but not to walk.

### Phase 2: Reward Rebalancing (Completed)

**Goal:** Break the standing-still local optimum.

**Changes from Phase 1:**
- `flat_orientation_l2` weight: -3.0 → **-1.0** (allow more forward lean)
- `feet_air_time` weight: 0.5 → **2.0**, threshold 0.5s → **0.2s** (reward shorter steps)
- `track_lin_vel_xy_exp` ramp ceiling: 1.0 → **1.5** (stronger velocity signal)
- `leg_extension` weight: 10.0 → **3.0** fixed (just prevent sploot, don't dominate)
- Removed `ramp_leg_ext` curriculum entirely

**Result:** Massive stability improvement:
- ep_len: 425-500 → **750-866** (peak 866 at iter 2271)
- `pelvis_too_low`: 0.71 → **0.30**

**But when visualized:** Still standing still. The deer is incredibly stable but not moving. `feet_air_time` remained negative throughout — feet never cleared even the 0.2s threshold.

### Phase 4: AMP (Adversarial Motion Priors) Integration — 2026-04-15 overnight

**Decision:** pure RL was stuck in a standing-still local optimum. Switched to AMP with the ithappy pack's Unity `Deer_001_walk.anim` as motion prior.

**Motion pipeline (`tools/` new scripts):**
1. `probe_walk_anim.py` — discovered clip is in-place (root doesn't translate) and that `thigh.*` bones are static in walk; hip-joint animation lives in `shoulder.*`. Bone mapping revised.
2. `probe_idle_anim.py` — idle[0] used as Unity-space rest pose (idle better matches URDF standing than walk[0]).
3. `probe_delta_twist.py` — confirmed that after rest subtraction, URDF's single-X-axis joint captures 11/12 leg bones with swing/twist ratio < 0.3; asymmetric `front_shoulder.R` needs composition with `front_thigh.R`.
4. `convert_unity_anim_to_npy.py` — full Hermite evaluation, swing-twist extraction, forward velocity synthesized at 0.8 m/s, exports `assets/motions/deer_walk.npy` (31 frames @ 30 fps).
5. `visualize_foot_trajectories.py` — verified foot phasing matches a real quadruped walk: HL (0.16) → FL (0.32) → HR (0.65) → FR (0.81).

**Framework integration:**
- `amp-rsl-rl 1.2.0` installed into Isaac Lab's Python (downgraded bundled `rsl-rl-lib` 5.0.1 → 3.0.0 so `ActorCritic` import works; Isaac Lab only imports `VecEnv` so this is safe).
- `training/envs/deer_usd_amp_env_cfg.py` — `DeerUsdWalkAMPEnvCfg` adds an `amp` observation group (joint_pos, joint_vel, base_lin_vel, base_ang_vel = 36 dims) and removes shaping rewards that AMP handles implicitly.
- `training/agents/deer_usd_amp_cfg.py` — `DEER_USD_AMP_WALK_CFG` dict consumed directly by `AMPOnPolicyRunner`.
- `training/scripts/train_amp.py` — bespoke driver (Isaac Lab's stock rsl_rl train.py doesn't know about AMP_PPO). Uses `RslRlVecEnvWrapper` to adapt Isaac Lab env to rsl_rl's VecEnv API.
- `training/scripts/play_amp.py` — checkpoint-playback driver.
- Task: `AniML-Deer-USD-Walk-AMP-v0`.

**Run 1 results (512 envs, std=1.0 from scratch, 500 iters, ~7 min on RTX 5070 Laptop):**
- Iter 0–50: exploration phase, deer falls 99% of the time, reward ~10.
- Iter 50–85: breakthrough — AMP gradient kicks in, reward 10 → 104, episode length 135 → 1000 (max), pelvis_too_low 99% → 5%.
- Iter 85–250: plateau at reward ~120–132, episode length 1000, pelvis_too_low 3–5%.
- Iter 250–500: curriculum ramps commanded velocity to 1.25 m/s (curriculum was untouched from inherited DeerUsdWalkEnvCfg), policy destabilizes and recovers twice. Final iter 485: reward 118, pelvis_too_low 1–3%, ep_len 1000.
- **Caveat:** TensorBoard shows `Train/mean_task_reward ≈ -0.04` (negative — dominated by `action_rate_l2`/`dof_torques_l2` penalties) and `Train/mean_style_reward ≈ 0.31` (positive but modest). **The deer gets most of its reward by fooling AMP's discriminator slightly**, not by velocity tracking (`error_vel_xy ≈ 1.36 m/s` at final iter, well above zero). This means the policy is standing mostly still while wiggling enough to score some AMP style points.

**Run 2 (warm-start from run 1 model_499.pt, 800 iters, killed):**
- New env `DeerUsdWalkAMPEnvCfgV2`: curriculum disabled, `action_rate_l2` / `dof_torques_l2` / `dof_acc_l2` zeroed, `track_lin_vel_xy_exp` weight 1.0 → 3.0, `rel_standing_envs = 0.0`.
- Runner cfg `DEER_USD_AMP_WALK_CFG_V2`: `init_noise_std` 1.0 → 0.4, `entropy_coef` 0.01 → 0.005.
- **Results:** reward peaked 183, `task_reward/step` +0.021 (first time positive), `lin_vel_R` peak 0.81 (2x run 1). Episode length near max, falls 5%.
- **But plateaued** at reward ~145 for 300+ iters. TensorBoard showed 96% of reward came from AMP style and only 4% from velocity tracking. The deer found a "wiggle-in-place" local optimum where it fools AMP slightly without actually moving.

**Run 3 (warm-start from run 2 model_800.pt, `reward_scale` 1.0 → 0.3, 2000 iters, in progress):**
- Same env as v2, but AMP `discriminator.reward_scale` dropped to **0.3** so style reward doesn't dominate.
- `init_noise_std` 0.4 → 0.25, `entropy_coef` 0.005 → 0.002, `learning_rate` 1e-3 → 5e-4 for refinement.
- **Early results (iter 200):** `err_vel_xy` dropping from 1.87 → 1.18 (progress!), `lin_vel_R` peak **1.32** per episode, pelvis_too_low ~1%, episode length still near max. Deer appears to be tracking velocity more of the time but noisily.

**Run 4 (warm-start from R3 model_1000, 0.85/0.15 task/style blend via monkey-patch, 834 iters, killed):**
- Brief breakthrough at iter 1689: `err_vel_xy` dropped to 0.99, `pelvis_too_low` spiked to 29% — the deer tried walking, fell too much, policy retreated to standing within 150 iters.
- Diagnosis: the termination cost-of-falling (29% ep end rate) outweighed the velocity-tracking reward gain. Standing 1000 steps at small reward beats walking 400 steps at larger reward.

**Run 5 — the walking run (warm-start from R4 model_1800, 2000 iters completed):**

All four recommended fixes applied together:

1. **Reference-state initialization**: `tools/bake_reset_states.py` pre-bakes expert pose+velocity per frame to `.pt`; new `reset_from_amp_motion` event in `deer_rewards.py` samples a random frame on each env reset. Standard AMP recipe.
2. **Removed `pelvis_too_low` termination**: episodes never terminate early for falls — deer can fall and keep trying.
3. **New `forward_velocity_linear` reward** (weight 2.0): linear in body-velocity projection onto command direction, gradient everywhere (unlike exp-tracking which saturates).
4. **Shorter episodes**: `episode_length_s` 20s → 5s, so standing-still reward stacks less.

New files: `DeerUsdWalkAMPEnvCfgV3`, `DEER_USD_AMP_WALK_CFG_V4`, task `AniML-Deer-USD-Walk-AMP-v4`, `tools/bake_reset_states.py`, additions to `deer_rewards.py`.

**Results:**
- `err_vel_xy` final: **0.38** (from runs 1-4's plateau at 1.3-1.5).
- `ep_len`: **250 / 250** (max — deer completes every episode without falling).
- `lin_vel_R` avg: **0.58**.
- `style_r/step`: **0.41** (up from run 4's 0.03 — style-reward also succeeding; discriminator getting fooled at times).
- Final checkpoint: `logs/rsl_rl/animl_deer_amp_walk_v4/2026-04-15_11-06-10/model_3799.pt` (iter 3799 total since run 1).
- ONNX exports saved alongside every 100th checkpoint, ready for Unity Sentis.

**The deer walks.** First time in the project.

**Run 2 (planned, warm-start from run 1 model_499.pt):**
- New task: `AniML-Deer-USD-Walk-AMP-v2` with `DeerUsdWalkAMPEnvCfgV2`.
- Velocity curriculum **disabled** (commands at 0.4–1.0 m/s from step 0).
- Task rewards: `track_lin_vel_xy_exp` weight 1.0 → **3.0**, `track_ang_vel_z_exp` → 1.0, `action_rate_l2`/`dof_torques_l2`/`dof_acc_l2` **zeroed** so the policy can't earn reward by staying still.
- `init_noise_std` 1.0 → 0.4, `entropy_coef` 0.01 → 0.005 (warm-start-friendly).

### Phase 3: Stationary Penalty (Completed, superseded by AMP)

**Goal:** Directly punish the standing-still behavior.

**New reward functions added to `deer_rewards.py`:**

1. **`velocity_error_l2`** (weight=-0.5): L2 penalty on XY velocity error between command and actual body velocity. Unlike `track_lin_vel_xy_exp` (which uses exp(-error²) and gives ~0.6 reward for standing), this is always non-zero when the deer isn't matching the command.

2. **`stationary_penalty`** (weight=-2.0): Fires when commanded velocity > 0.3 m/s but body speed < 0.3 m/s. Directly targets the standing-still optimum.

**Status at iter ~924/3000:**
- ep_len: ~500 (down from 866 — harder penalties)
- `stationary_penalty`: -0.15 to -0.23 (still firing — deer still mostly standing)
- `velocity_error`: -0.10 to -0.16 (active)
- `pelvis_too_low`: 0.53 (more falls than phase 2, penalties destabilizing)

**Run still in progress.** Training at ~1s/iter, ETA ~35 more minutes.

---

## Current State

### What Works
- Custom deer USD model spawns correctly with proper orientation in Isaac Lab
- CoM is physically accurate (computed from mesh geometry)
- Standing policy is rock-solid — the deer balances indefinitely
- Training pipeline is fully automated (env configs, reward functions, checkpoints, curriculum)
- Warm-starting from standing checkpoints consistently produces stable walkers
- Episode lengths of 750-866 steps (15-17 seconds) achieved

### What Doesn't Work Yet
- **No forward locomotion**: The policy has found a local optimum where standing still maximizes reward. Despite trying multiple reward configurations, the deer hasn't learned to take steps and move forward.
- **feet_air_time always negative**: Feet never leave the ground long enough to register as "stepping."
- The `track_lin_vel_xy_exp` Gaussian reward gives partial credit even at zero velocity, enabling the standing-still exploit.

### Key Files

| File | Purpose |
|------|---------|
| `training/envs/deer_usd_rough_env_cfg.py` | All environment configs (DeerUsdStandEnvCfg, DeerUsdWalkEnvCfg) |
| `training/envs/deer_usd_articulation.py` | ArticulationCfg with spawn rotation, default joints, actuator settings |
| `training/envs/deer_rewards.py` | Custom reward/termination functions (leg_extension, pelvis_too_low, velocity_error, stationary_penalty, etc.) |
| `training/agents/deer_usd_rsl_rl_cfg.py` | PPO runner configs (network architecture, hyperparameters, checkpoint loading) |
| `training/scripts/train.py` | Training entry point (registers tasks, delegates to Isaac Lab's RSL-RL trainer) |
| `training/scripts/play.py` | Visualization entry point |
| `assets/deer_urdf/deer_world.urdf` | Source URDF with CoM-patched inertials |
| `deer_world.usd` | Sim-ready USD (generated from URDF, root transform = identity) |
| `tools/patch_ckpt_for_walk.py` | Checkpoint manipulation (reset std, clear critic/optimizer) |

### Best Checkpoints

| Checkpoint | Run | ep_len | Notes |
|-----------|-----|--------|-------|
| `model_199.pt` | `animl_deer_stand_v2/2026-04-13_22-09-04` | N/A | Best standing policy (CoM-fixed) |
| `model_3700.pt` | `animl_deer_walk_p1/2026-04-14_12-55-18` | 686 | Phase 1 peak — good balance, no forward motion |
| `model_2200.pt` | `animl_deer_walk_p1/2026-04-14_15-37-59` | 866 | Phase 2 peak — excellent stability, no forward motion |

### Hyperparameters (Current Walk Config)

```yaml
Network: MLP [512, 256, 128], activation=elu
init_noise_std: 0.1
entropy_coef: 0.001
learning_rate: 1.0e-3 (adaptive schedule, desired_kl=0.008)
gamma: 0.99, lam: 0.95
clip_param: 0.2
num_envs: 1500
episode_length: 20s
action_scale: 0.5
```

---

## Next Steps / Open Questions

1. **Does the stationary penalty work?** Phase 3 is still training. If `stationary_penalty` decreases while `velocity_error` improves, the deer is learning to move. If both stay high, stronger measures needed.

2. **Alternative approaches if current strategy fails:**
   - **Reference motion tracking**: Use motion capture data of a real deer walking as a reference trajectory. This is the industry-standard approach (DeepMimic, AMP) for natural locomotion.
   - **Asymmetric reward**: Give very large one-time bonus for first step, then scale down.
   - **Phase the velocity command**: Start with very slow commands (0.05 m/s) where even a slight lean counts, then gradually increase.
   - **Action space restructuring**: Use a CPG (Central Pattern Generator) as action space instead of raw joint positions — this bakes in the concept of alternating leg phases.

3. **ONNX export**: Once a walking policy exists, export to ONNX for Unity Sentis deployment.

4. **Multi-species pipeline**: The URDF→USD→Isaac Lab pipeline is now documented and repeatable. Adding a new species requires: (a) rigged mesh with collision geometry, (b) URDF with correct joint limits and CoM, (c) USD conversion, (d) species-specific env config inheriting from DeerUsdRoughEnvCfg.

5. **Ecosystem integration**: World Director events, predator-prey dynamics, reproduction/genetics — all blocked on having at least one species that can walk.
