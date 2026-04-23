# AMP Locomotion Training Plan

## Goal

Train the deer to walk and run using **Adversarial Motion Priors (AMP)** — a technique where a discriminator network learns what "natural deer motion" looks like from reference animation data, and the policy gets rewarded for producing motion that fools the discriminator, alongside standard task rewards (velocity tracking).

We have a massive advantage: **the original deer model (ithappy Animals_FREE) shipped with walk, run, and idle animations** that use the exact same skeleton. No mocap retargeting or synthetic motion generation needed.

---

## Why AMP (and Why Pure RL Failed)

Our pure-RL attempts hit a fundamental problem: the policy learned to balance perfectly but found that **standing still** was the reward-maximizing strategy. Adding velocity rewards, stationary penalties, and curriculum tricks all failed because:

- The reward landscape has a deep local minimum at "stand still and collect balance rewards"
- The policy never experiences what walking *feels like*, so it can't discover the coordinated leg cycling pattern through random exploration alone
- Real quadruped labs (ETH, IIT, Berkeley) almost universally use motion priors for natural locomotion

AMP solves this by giving the policy a continuous "style" signal: "this is what moving your legs should look like." The policy still optimizes task rewards, but now it has a gradient toward natural gait patterns instead of having to discover them from scratch.

---

## Framework Choice: `amp-rsl-rl`

**Repository**: [github.com/ami-iit/amp-rsl-rl](https://github.com/ami-iit/amp-rsl-rl) (294 stars, v1.2.0, March 2026)

Why this over alternatives:
- **Drop-in replacement** for our existing `rsl_rl` training pipeline. Same config pattern, same training loop — just swap `PPO` → `AMP_PPO`.
- **pip-installable**: `pip install amp-rsl-rl`
- **Clean `.npy` motion data format** with well-documented structure
- **Built for quadrupeds** — used with Aliengo at IIT DLS Lab
- **Actively maintained** by the Italian Institute of Technology robotics group

Alternatives considered:
- **Isaac Lab native AMP via `skrl`**: Would require switching our entire pipeline from rsl_rl to skrl. Too disruptive.
- **beyondAMP**: Good but more complex setup, geared toward humanoids.

---

## Motion Data: Unity `.anim` → `.npy` Conversion

### Source Files

Located in the Unity project:
```
C:\Users\nars7\Downloads\Unity Projects\aniML-world\Assets\ithappy\Animals_FREE\Animations\Other_Animations\
├── Deer_001_walk.anim    ← primary training clip
├── Deer_001_run.anim     ← secondary clip (add later)
├── Deer_001_idle.anim    ← standing reference
```

### Target Format (amp-rsl-rl `.npy`)

```python
{
    "joints_list": ["pelvis_to_chest", "pelvis_to_HR_thigh", ...],  # 15 joint names
    "joint_positions": [np.array([...]), ...],   # per-frame, shape (15,) each
    "root_position": [np.array([x, y, z]), ...], # per-frame, 3D, Z-up (Isaac Lab)
    "root_quaternion": [np.array([x, y, z, w]), ...],  # per-frame, xyzw (SciPy convention)
    "fps": 30.0
}
```

### Skeleton Mapping

Unity animation bones → URDF joints. The deer has **15 revolute joints, ALL on X-axis**.

> **⚠ Mapping revised after Phase 0a probe.** The original plan had `hip ← shoulder × thigh`. The probe showed `thigh.*` bones are static (2 keyframes, zero amplitude) in the walk clip — all hip rotation actually lives in `shoulder.*` / `front_shoulder.*`. We read shoulder only and ignore thigh for the hip joint.
>
> Asymmetry note: on the **L front leg** rotation is on `front_shoulder.L` only, but on the **R front leg** both `front_shoulder.R` and `front_thigh.R` have keyframes. We'll handle both sides symmetrically by reading shoulder only and logging a warning if `thigh.*` has non-trivial amplitude in a future clip.

**Leg joints:**

| URDF Joint | Unity Bone | Notes |
|---|---|---|
| `pelvis_to_HR_thigh` | `shoulder.R` | Hip — rear right. Shoulder bone carries the swing. |
| `HR_thigh_to_HR_shin` | `shin.R` | Knee — local rotation of shin relative to thigh |
| `HR_shin_to_HR_foot` | `foot.R` | Ankle — local rotation of foot relative to shin |
| `pelvis_to_HL_thigh` | `shoulder.L` | Hip — rear left |
| `HL_thigh_to_HL_shin` | `shin.L` | Knee |
| `HL_shin_to_HL_foot` | `foot.L` | Ankle |
| `chest_to_FR_thigh` | `front_shoulder.R` | Shoulder — front right |
| `FR_thigh_to_FR_shin` | `front_shin.R` | Elbow |
| `FR_shin_to_FR_foot` | `front_foot.R` | Wrist |
| `chest_to_FL_thigh` | `front_shoulder.L` | Shoulder — front left |
| `FL_thigh_to_FL_shin` | `front_shin.L` | Elbow |
| `FL_shin_to_FL_foot` | `front_foot.L` | Wrist |

**Spine/head joints (multi-bone compose):**

| URDF Joint | Unity Bone(s) | Notes |
|---|---|---|
| `pelvis_to_chest` | `spine.004` × `spine.005` | Two spine segments compose to one |
| `chest_to_neck` | `spine.006` × `spine.007` × `spine.008` | Three segments → one neck joint |
| `neck_to_skull` | `scull` | Direct (note: rig uses "scull" not "skull") |

### ⚠ Non-X rotation amplitudes are large — bone frames ≠ URDF frames

The probe found `front_shoulder.R` has `x_amp=21°`, `y_amp=46°`, `z_amp=46°`. That is not small off-axis noise — it's the majority of the motion. The only sane interpretation: **the Unity bone's local X-axis is not the same direction as the URDF joint's X-axis.** Bone roll during rigging/export reorients the local frame.

This makes swing-twist extraction non-optional and bind-pose calibration essential: we must compute each URDF joint axis expressed in its Unity bone's local frame (via the bind-pose relative transform between bone and URDF link), then twist the animated rotation around **that** axis, not around bone-local X.

### Conversion Algorithm

```
For each frame t in [0, 1] at 30 fps (31 frames for a 1-second cycle):

1. Parse all m_RotationCurves from the .anim YAML
2. Evaluate each curve at time t. Unity defaults to cubic Hermite with per-key
   in/out tangents — re-implementing this correctly is fiddly. Preferred path:
   run a 2-minute headless Unity editor script that samples the AnimationClip
   at 60 fps and dumps bone local TRS to CSV/JSON. Falls back to hand-rolled
   Hermite eval only if that isn't viable.
3. For each URDF joint:
   a. Identify the contributing Unity bone(s)
   b. Get their local quaternions at time t
   c. If multiple bones: compose quaternions (q_parent × q_child)
   d. Subtract the bind-pose rotation to get the delta quaternion
   e. Extract the joint angle via SWING-TWIST decomposition around the joint's
      axis (URDF X for all 15 joints). Take the twist angle only; discard swing.
      Do NOT use naive `2*atan2(qx, qw)` — baked animations have small off-axis
      components from IK cleanup that this silently drops.
   f. Clamp to URDF joint limits, log any frame that required clamping
4. For root body:
   a. Get position from m_PositionCurves of the root bone
   b. Handle handedness: Unity is LEFT-handed, URDF/Isaac is RIGHT-handed.
      The axis swap alone (Y-up → Z-up) isn't sufficient; we also need a
      handedness flip. Validate by rendering one frame in both frames and
      confirming the deer faces the expected direction.
   c. Transform Unity (x, y, z) → Isaac Z-up with handedness correction
   d. Add spawn height offset (0.93m) to Z component
   e. Get orientation from m_RotationCurves of root bone chain
   f. Compose with Rx(-90°) to convert to Z-up
```

### Key Technical Challenges

1. **Bind pose comes from the armature, not `idle.anim[0]`.** The animation quaternions represent absolute local rotations, and we need the bind (T-pose) rotation per bone to compute deltas. Using idle.anim's first frame is risky: if idle isn't the true bind pose, every joint angle ends up with a constant offset and the discriminator learns a weirdly-posed "natural" deer. **Pull `m_LocalRotation` directly from the armature object** (from the `.fbx`/`.blend` or from the Unity scene's bind pose data). Confirm by checking that applying the bind rotations produces the URDF default pose.

2. **Coordinate frames — handedness matters.** Unity is **Y-up, left-handed**. URDF is Y-up, right-handed (before Rx(-90°) to Isaac Z-up). The axis swap `(x,y,z) → (x,z,-y)` only handles up-axis — you may also need to flip X or Z for handedness. Joint angles are local and frame-independent (swing-twist handles it), but root pose needs explicit handedness correction. Sanity-check by visualizing one frame.

3. **Multi-bone composition is the single biggest risk in the whole plan.** The URDF collapses Unity's `shoulder.R × thigh.R` into one X-axis `pelvis_to_HR_thigh` joint. Real skeletal animations use shoulders for scapula slide — Y and Z rotation components that cannot be expressed as an X-axis rotation. Projecting loses motion, or produces nonsense.
   **Measure before converting.** Dump shoulder and front_shoulder quaternions across the full walk cycle and inspect the amplitude of their non-X components. Two branches:
     - **If scapula motion is small (≲ 5°):** treat the hip joint as thigh-only, ignore shoulder. Simpler and usually correct for a walk cycle.
     - **If scapula motion is significant:** either extend the URDF with an extra hip DoF, or bake the shoulder contribution into root pose (lose it from the joint signal but preserve the root trajectory it generates).

4. **⚠ CRITICAL PRE-CHECK: is `Deer_001_walk.anim` in-place?** Most Unity walk clips are **root-motion-stripped** so they loop cleanly; forward motion is applied at runtime by a Root Motion driver. If the clip is in-place, `root_position` is stationary and **the discriminator will learn that "real walking = root stays put"** — reproducing the same standing-still optimum pure RL hit, now with a prettier gait.
   **Check this before writing a single line of the converter.** If in-place:
     - Integrate foot-contact kinematics to recover forward velocity (sum stance-phase displacements per cycle), OR
     - Apply a synthetic forward velocity equal to the clip's intended speed, OR
     - Use Unity's Root Motion bake to produce a displaced version and dump that.

### Conversion Script

We'll write `tools/convert_unity_anim_to_npy.py` that:
- Reads bind pose from the armature (not from idle.anim)
- Samples Unity curves at uniform timesteps (preferably via headless Unity editor script)
- Applies the bone → joint mapping with swing-twist extraction
- Applies handedness correction for the root
- Outputs `assets/motions/deer_walk.npy` (and later `deer_run.npy`)
- Logs per-joint angle histograms and clamp counts for validation

**Time budget: 1 full day, not 1–2 hours.** The original estimate didn't account for bind-pose calibration, Hermite eval, handedness debugging, or per-joint validation.

---

## Training Setup Changes

### 1. Install `amp-rsl-rl`

```bash
pip install amp-rsl-rl
```

This installs alongside our existing `rsl-rl-lib`. The `AMP_PPO` class extends `PPO` with:
- A **discriminator network** (small MLP, ~128×128) trained to distinguish real vs. generated motion transitions
- A **style reward** from the discriminator added to the task reward
- A **gradient penalty** on the discriminator for training stability

### 2. New Training Config: `training/agents/deer_usd_amp_cfg.py`

```python
from amp_rsl_rl.algorithms import AMP_PPO

@configclass
class DeerUsdWalkAMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMP_PPO"          # ← the key change
    num_steps_per_env = 24
    max_iterations = 3000
    experiment_name = "deer_amp_walk"

    # Same actor/critic as before.
    # NOTE: init_noise_std choice depends on warm-start strategy (see below).
    #   - From scratch:                 init_noise_std = 1.0 (AMP provides its own gait prior)
    #   - Warm-start from standing ckpt: init_noise_std ≈ 0.3
    # Do NOT combine warm-start with std=1.0 — rsl_rl v4 overwrites the ckpt
    # std with the config value, instantly destroying the prior you loaded.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # AMP-specific config
    amp_motion_files = ["assets/motions/deer_walk.npy"]
    amp_reward_coef = 0.5            # weight of style reward vs task reward
    amp_disc_hidden_dims = [128, 128]
    amp_disc_grad_penalty = 5.0      # most important AMP hyperparameter (range: 0.1–10)
    amp_disc_learning_rate = 1.0e-4
    amp_replay_buffer_size = 100000
```

> **Note**: The exact config field names may differ based on `amp-rsl-rl`'s API. We'll check the source after installing.

### 3. Environment Config: `DeerUsdWalkAMPEnvCfg`

Mostly the same as `DeerUsdWalkEnvCfg` but with simplified rewards. AMP handles the "how to move naturally" part, so we can strip down to just:

- **`track_lin_vel_xy_exp`**: Primary task reward — follow velocity commands (weight: 1.0)
- **`track_ang_vel_z_exp`**: Turn tracking (weight: 0.5)
- **`base_height_l2`**: Don't fly or sink (weight: -1.0)
- **`dof_torques_l2`**: Energy penalty (weight: -0.0001, very light)
- **`action_rate_l2`**: Smoothness (weight: -0.01)

**Remove**: `leg_extension`, `flat_orientation_l2`, `feet_air_time`, `velocity_error_l2`, `stationary_penalty` — the AMP discriminator handles all of these implicitly.

### 4. AMP Observation Space

The discriminator needs to see state transitions to judge if they look natural. The standard AMP observation for a quadruped includes:
- Joint positions (15 values)
- Joint velocities (15 values)
- Root body linear velocity (3 values, body frame)
- Root body angular velocity (3 values, body frame)
- Root body height (1 value)
- **Foot contact indicators (4 values, binary or normalized force)** — critical for a quadruped. This is what gives the discriminator its strong gait signal; without contacts the discriminator mostly learns pose, not phase. Confirm what IIT's Aliengo recipe uses before omitting.

Total AMP observation dim: ~41 per timestep.

**Transition pairing**: amp-rsl-rl uses transition pairs `(s_t, s_{t+1})` for the discriminator, not a stacked history window. Indexing the motion buffer and reward computation must use pair semantics — verify against the library source before wiring up the obs group.

This is separate from the policy observation space (which stays the same as our current setup).

---

## Execution Plan

### Phase 0a: Pre-Converter Reality Checks ✅ COMPLETE

See `assets/motions/deer_walk_probe.md` and `deer_walk_twist_report.md`. Key findings:

1. **Root motion: IN-PLACE.** Every bone has 0 position keyframes in the entire clip. We must synthesize forward velocity.
2. **Thigh bones are mostly static.** Hip-joint rotation lives in `shoulder.*` / `front_shoulder.*`, not in `thigh.*`. Bone mapping revised (see Skeleton Mapping table above).
3. **Once rest-pose is subtracted, 11/12 leg bones produce near-pure X-axis twist** (ratios < 0.3, i.e. swing ≪ twist). The scary 45° non-X components are just bind-pose offsets, not true off-axis motion. A single X-axis URDF joint is sufficient.
4. **One asymmetric bone** — `front_shoulder.R`. The animator split its rotation across `front_shoulder.R` + `front_thigh.R` (unlike the L side, where `front_thigh.L` is static). Composing the two recovers a clean twist (range 28°) matching the L side's range 30°. Converter must handle this special case.
5. **Bind pose source: use `Deer_001_walk.anim` frame 0.** Neither idle nor the Blender-imported FBX rest matches Unity's `.anim` frame-0 values (Blender and Unity apply different axis conversions on import). For AMP we don't need the "true" bind — just a consistent zero. Walk frame 0 is that zero, and the URDF/USD `init_state.joint_pos` can be set to match it.

### Phase 0b: Motion Data Conversion ✅ COMPLETE

Artifacts produced:
- `tools/probe_walk_anim.py` — pre-checks (root motion, shoulder amplitude)
- `tools/probe_idle_anim.py` — Unity-space rest pose extraction from idle frame 0
- `tools/probe_delta_twist.py` — verify X-twist assumption after rest subtraction
- `tools/blender_dump_bind_pose.py` — Blender-space bind pose dump (reference)
- `tools/convert_unity_anim_to_npy.py` — the actual converter
- `tools/visualize_motion_npy.py` — stick-figure FK viz
- `tools/visualize_foot_trajectories.py` — foot-path + joint-curve viz
- `assets/motions/deer_walk.npy` — final AMP dataset (31 frames @ 30 fps)
- `assets/motions/deer_walk_probe.md`, `deer_walk_twist_report.md`, `deer_walk_conversion_report.md` — analysis records
- `assets/motions/deer_walk_viz/_foot_trajectories.png`, `_joint_curves.png` — verification plots

Key choices made:
- **Rest pose:** `Deer_001_idle.anim` frame 0 (better match to URDF standing than walk[0]).
- **Bone mapping:** hip joint ← shoulder only (thigh bones are static in walk), except `front_shoulder.R` which uses `shoulder × thigh` composition (asymmetric rig export).
- **Forward velocity synthesis:** 0.8 m/s constant along Isaac +X (clip is in-place).
- **Joint angle extraction:** swing-twist decomposition around bone-local X after subtracting rest rotation.

Remaining minor issue: front feet (`F{R,L}_shin_to_F{R,L}_foot`) clamp at ±28.6° URDF limit on roughly 11/31 and 22/31 frames. Peaks are exactly at the limit, not beyond by much. Mitigation options if AMP training suffers:
- Widen URDF foot limits from ±0.5 rad to ±0.6 rad (requires URDF regen + USD re-export).
- Scale motion amplitudes down by ~5 % (quick hack).

### Original time estimate (superseded)

1. Extract bind pose from the armature (NOT from idle.anim frame 0).
2. Write `tools/convert_unity_anim_to_npy.py`. Prefer a short Unity editor script for curve sampling; fall back to PyYAML + Hermite eval only if needed.
3. Implement swing-twist angle extraction per joint.
4. Apply handedness-aware root transform (Unity left-handed → Isaac right-handed, Y-up → Z-up).
5. Convert walk animation to `assets/motions/deer_walk.npy`.
6. **Validate**:
   - Per-joint angle histograms, clamp counts per joint (hard error if > 5% of frames clamped on any joint).
   - Plot the walk cycle (matplotlib stick-figure or re-import into Isaac Sim viewer).
   - Confirm root trajectory direction matches expected walk direction.

### Phase 1: Framework Integration ✅ CODE WRITTEN, SMOKE TEST PENDING

Artifacts produced:
- `training/envs/deer_usd_amp_env_cfg.py` — `DeerUsdWalkAMPEnvCfg` with `AMPObservationsCfg` group and simplified rewards (removed `leg_extension`, `flat_orientation_l2`, `feet_air_time`, `velocity_error`, `stationary_penalty`, `base_height_l2`, `lin_vel_z_l2`, `ang_vel_xy_l2`).
- `training/agents/deer_usd_amp_cfg.py` — `DEER_USD_AMP_WALK_CFG` dict (amp-rsl-rl runner consumes dict, not dataclass) plus a `@configclass` wrapper.
- `training/tasks/__init__.py` — registered `AniML-Deer-USD-Walk-AMP-v0` with `amp_rsl_rl_cfg_entry_point`.
- `training/scripts/train_amp.py` — bespoke driver that instantiates `AMPOnPolicyRunner` directly (Isaac Lab's stock rsl_rl train.py doesn't know about AMP_PPO).

**Dependency gotcha:** `amp-rsl-rl` 1.2.0 imports `from rsl_rl.modules import ActorCritic`, which was removed after `rsl-rl-lib` 3.x. Pin: `pip install 'rsl-rl-lib==3.0.0'`. If Isaac Lab vendors its own rsl_rl, you'll want a dedicated venv for AMP training so pins don't collide.

### Smoke test recipe (to run on Isaac Lab machine)

```powershell
$env:PYTHONPATH = 'C:\Users\nars7\Downloads\aniML_world'
$env:ISAACLAB_ROOT = 'C:\isaac-lab\IsaacLab'
# Ensure the URDF/USD is available the usual way (ANIML_DEER_USD env var or default file at repo root)

# 10-iteration smoke test
.\isaaclab.bat -p C:\Users\nars7\Downloads\aniML_world\training\scripts\train_amp.py `
    --task AniML-Deer-USD-Walk-AMP-v0 --num_envs 64 --max_iterations 10 --headless
```

Expected outcomes if it works:
- Training loop starts, rollouts proceed, `AMPOnPolicyRunner` prints per-iteration stats.
- TensorBoard logs write under `logs/rsl_rl/animl_deer_amp_walk/<timestamp>/`.
- No exceptions from `AMPLoader` (motion dataset is already validated locally).

Likely failure modes and where to look:
- **Missing `amp` obs group** → runner errors at `env.cfg.observations.amp.joint_pos.params['asset_cfg'].joint_names`. Check `DeerUsdWalkAMPEnvCfg.__post_init__` is attaching `AMPObservationsCfg`.
- **Joint-name mismatch** → `AMPLoader` expects the 15-joint order we set in both the .npy and the env cfg (same constant list).
- **`ActorCritic` import error** → rsl-rl-lib version > 3.x.
- **TensorDict obs vs dict obs** → Isaac Lab returns an observation namespace; runner calls `.clone().detach()` so it must be tensor-like. If this breaks, wrap via `VecEnvAdapter`.

### Phase 2: Training (2-4 hours)

1. Launch headless AMP training: ~3000 iterations, 1024 envs
2. Monitor discriminator loss (should decrease then stabilize around 0.5 — meaning it can't easily tell real from fake)
3. Monitor episode length (should increase as the deer learns to walk without falling)
4. Monitor velocity tracking reward (should increase as the deer follows commands)
5. **Key milestone**: By iteration 500-1000, the deer should show visibly coordinated leg motion

### Phase 3: Tuning (if needed)

The single most important hyperparameter is **`disc_grad_penalty`** (or `amp_disc_grad_penalty`):
- Too high → discriminator is too weak, style reward is meaningless, policy reverts to pure RL behavior
- Too low → discriminator is too strong, policy can't satisfy it, training collapses
- Sweet spot is typically 1.0–10.0, start at 5.0

If the deer walks but looks unnatural:
- Add `Deer_001_run.anim` as a second motion clip to give the discriminator a broader sense of "natural"
- Increase `amp_reward_coef` to weight style more

If the deer falls over:
- Warm-start from our existing standing checkpoint (load actor weights, keep AMP discriminator fresh)
- Reduce velocity command range initially

### Phase 4: Run Animation (optional, later)

Once walk works:
1. Convert `Deer_001_run.anim` to `assets/motions/deer_run.npy`
2. Add to `amp_motion_files` list alongside walk
3. Increase velocity command range
4. The discriminator will learn both gaits and the policy will naturally transition between walk and run based on commanded speed

---

## File Changes Summary

| File | Action |
|---|---|
| `tools/convert_unity_anim_to_npy.py` | **New** — Unity .anim → .npy converter |
| `assets/motions/deer_walk.npy` | **New** — Converted walk motion data |
| `training/agents/deer_usd_amp_cfg.py` | **New** — AMP_PPO runner config |
| `training/envs/deer_usd_rough_env_cfg.py` | **Edit** — Add `DeerUsdWalkAMPEnvCfg` |
| `training/envs/__init__.py` | **Edit** — Register new AMP task |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Motion data conversion produces wrong joint angles | Medium | Validate against URDF limits, visualize before training |
| `amp-rsl-rl` API doesn't match our rsl_rl version | Low | Check compatibility, pin versions if needed |
| Discriminator overpowers task rewards | Medium | Tune `disc_grad_penalty`, adjust `amp_reward_coef` |
| 12GB VRAM insufficient for AMP discriminator overhead | Low | Discriminator is tiny (~128×128 MLP), adds <100MB |
| Walk animation doesn't translate well to physics sim | Medium | The animation is from the same skeleton, just needs correct calibration |

---

## Fallbacks

Ordered by preference if AMP stalls or motion conversion produces bad data.

### Fallback A: Residual Policy over a Hand-Crafted Trot

Middle ground between AMP and CPG. Bake a fixed sinusoidal trot (same phase pattern ithappy's walk clip uses). Policy outputs residuals on top of the baked trajectory. Same "forced cyclic motion" property as CPG, no oscillator math, and warm-starts against the gait prior trivially. Cheapest rescue path — usually a day of work.

### Fallback B: CPG-Based Action Space

Instead of outputting raw joint positions, the policy outputs oscillator parameters (amplitude, frequency, phase offset per leg). A CPG module converts them to coordinated sinusoidal joint trajectories. Guarantees cyclic leg motion, no reference motion needed, but produces a more mechanical look than AMP.
