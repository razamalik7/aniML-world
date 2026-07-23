# Research memo — how much reference noise can AMP absorb?

**2026-07-23.** Evidence labels: **VERIFIED** (read in primary source — code or
paper text) · **REPORTED** (secondary/summary only) · **UNVERIFIED**.

Distilled into `docs/CV_PIPELINE_DESIGN.md`. This file keeps the evidence.

---

## 1. Bottom line

**The global-root-trajectory stage can be cut.** No AMP discriminator in any
implementation read observes absolute root position. In `amp-rsl-rl` specifically
it observes neither root height nor root orientation. Root data enters only as a
3-vector of linear velocity (finite difference of root position, rotated to body
frame) and a 3-vector of angular velocity.

**The bar that binds is per-frame joint-angle jitter**, because `amp-rsl-rl`
finite-differences the clip into joint velocities and feeds them to the
discriminator. Computed on this project's own `deer_walk.npy` (**VERIFIED** — ran
the numbers; true RMS joint speed 1.63 rad/s, joint-angle amplitude 10.5°):

| jitter σ | spurious joint-vel std | % of true RMS joint speed | % of joint-angle amplitude |
|---|---|---|---|
| 0.5° | 0.37 rad/s | 23 % | 4.8 % |
| 1° | 0.74 rad/s | **45 %** | 9.6 % |
| 2° | 1.48 rad/s | 91 % | 19 % |
| 3° | 2.22 rad/s | 136 % | 29 % |
| 5° | 3.70 rad/s | 227 % | 48 % |

At 30 fps jitter is amplified **√2·fps ≈ 42×** into the velocity channel. Since
the policy cannot physically produce white-noise joint velocity, the velocity
columns become a free separating feature → discriminator saturates → style reward
→ 0. That is the failure fingerprint already in `CLAUDE.md`.

### Acceptance bar

| Quantity | Target |
|---|---|
| Joint-angle jitter σ after filtering | **≤ 1°**, ideally ≤ 0.5° |
| Bone-length CV over time | **0 by construction** (fit rig angles, never free 3D points) |
| Stance foot slide | **< ~5 % of stride length** |
| Root speed | **±10–20 %**, derived from foot odometry |
| Global root XY / heading / scale | **not needed** |
| Absolute root height | **not needed**; FK-ground it |
| Missing DOFs | tolerable — **exclude, never zero-fill** |

---

## 2. Discriminator observation specs (code citations)

### 2.1 `amp-rsl-rl` v2.2.3 — the one that matters here
`amp_rsl_rl/utils/motion_loader.py`, `MotionData.get_amp_dataset_obs`, L172–206.
**VERIFIED — read the installed source.**

```python
if velocity_representation == VelocityRepresentation.MIXED_REPRESENTATION:
    return torch.cat((
        self.joint_positions[indices],
        self.joint_velocities[indices],
        self.base_lin_velocities_mixed[indices],
        self.base_ang_velocities_mixed[indices],
    ), dim=1)
return torch.cat((
    self.joint_positions[indices],
    self.joint_velocities[indices],
    self.base_lin_velocities_local[indices],
    self.base_ang_velocities_local[indices],
), dim=1)
```

That is the **entire** observation. No root position, height, or orientation.

Where root data enters at all (`load_data`, L502–527): `root_position` is
resampled then passed to `_compute_raw_derivative` → `base_lin_vel`;
`root_quaternion` only produces angular velocity and rotates linear velocity into
body frame. Neither survives into the observation. `get_state_for_reset`
(L208–239, the RSI path) returns `(quat, joint_pos, joint_vel, lin_vel, ang_vel)`
— **no position**.

Two further facts:
- Joint velocity is a **raw finite difference** (`_compute_raw_derivative`, L450)
  after **linear** resampling (`interp1d`, L414). No smoothing anywhere.
- `simulation_dt = env.cfg.sim.dt * env.cfg.decimation` (`amp_on_policy_runner.py:293`)
  = 0.02 s. A 31-frame / 30 fps clip becomes ~51 discriminator transitions, all on
  straight-line interpolants between 31 poses.

Project side, `training/envs/deer_usd_amp_env_cfg.py::AMPObservationsCfg`
(**VERIFIED**): `joint_pos(15) + joint_vel(15) + base_lin_vel(3) +
base_ang_vel(3) = 36`; discriminator input 72 for the `(s, s')` pair.

### 2.2 `nv-tlabs/ASE` — `humanoid_amp.py::build_amp_observations` (**VERIFIED**)
```python
root_h = root_pos[:, 2:3]
heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
...
if (not root_height_obs): root_h_obs = torch.zeros_like(root_h)
local_key_body_pos = key_body_pos - root_pos.unsqueeze(-2)   # root-relative
obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel,
                 dof_obs, dof_vel, flat_local_key_pos), dim=-1)
```
`root_pos` used **only** for `[:, 2:3]` and as the origin subtracted from key-body
positions. Horizontal position never appears; heading explicitly divided out.
`root_height_obs` is a flag that can zero height out entirely.

### 2.3 IsaacGymEnvs `humanoid_amp.py` (**VERIFIED**)
Identical. `NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12 = 105`.

### 2.4 `escontrela/AMP_for_hardware` (**VERIFIED**)
```python
def get_amp_observations(self):
    ...
    return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel,
                      joint_vel, z_pos), dim=-1)
```
`rsl_rl/datasets/motion_loader.py` carries the comment **"Remove first 7
observation dimensions (root_pos and root_orn)"** — dataset frames literally drop
root position and orientation, re-adding only root z. 43-dim.

### 2.5 Isaac Lab official `humanoid_amp_env.py::compute_obs` (**VERIFIED**)
Includes `root_positions[:, 2:3]` and the **full** root quaternion, and feeds
**world-frame** velocities — so this variant *is* heading-sensitive. A known
quirk; do not copy.

### 2.6 AMP paper §5.3 (**VERIFIED** via ar5iv)
> "Linear velocity and angular velocity of the root, represented in the
> character's local coordinate frame. Local rotation of each joint. Local velocity
> of each joint. 3D positions of the end-effectors … represented in the
> character's local coordinate frame."

### Verdict
**Absolute root position is never a discriminator input in any of five
implementations or in the paper spec.** Root height appears in 4 of 5 (optional in
ASE); `amp-rsl-rl` omits it. Heading is explicitly removed or absent by
construction.

### 2.7 Contrast — DeepMimic (**VERIFIED** via ar5iv, arXiv 1804.02717)
Imitation reward includes an end-effector term using **"the 3D world position in
meters of end-effector"** and a **centre-of-mass** term on global positions, with
a phase variable φ∈[0,1] for per-frame synchronisation. Per-frame reference error
enters the reward directly. VideoMimic (CoRL 2025) is a batched DeepMimic variant
and **does** require the global root reference at train time (**REPORTED**). This
is the structural reason tracking needs the CV stage AMP does not.

---

## 3. Evidence table

| # | Source | Finding | Status |
|---|---|---|---|
| E1 | AMP paper §5 | **"the observed state transitions may not be physically consistent for the agent, and therefore impossible for the agent to perfectly reproduce. Despite this discrepancy, we show that the discriminator still provides an effective objective."** | **VERIFIED** |
| E2 | AMP paper §8.5 | **"Gradient penalty proves to be the most vital component."** Without it, "large performance fluctuations… noticeable visual artifacts." GP is what buys tolerance. | **VERIFIED** |
| E3 | AMP §8.5, Fig. 8 | Velocity features are load-bearing — consecutive poses alone "was insufficient for some motions, such as rolling." You cannot just drop the noisy velocity channel. | **VERIFIED** |
| E4 | AMP Table 2 | Reference dataset sizes: Locomotion 434.1 s / 56 clips; Walk 229.6 s / 9 clips; Cartwheel 13.6 s / 3 clips. **Two orders of magnitude more than this project's 1.03 s.** | **VERIFIED** |
| E5 | Escontrela et al. (2203.15103) | Real A1 locomotion learned from **4.5 s of German Shepherd mocap**. No data-quantity ablation. | **VERIFIED** |
| E6 | WASABI, CoRL 2022 (2206.11693) | Discriminator saw **only 10 dims: base lin vel, base ang vel, gravity vector, base height — no joint angles at all.** Demonstrations were a human hand-holding an unactuated Solo 8. Still learned a backflip. | **VERIFIED** |
| E7 | WASABI Table 1 (DTW, lower better) | LSGAN (= AMP's objective) vs WASABI: BackFlip **813.76 → 477.43**; StandUp 678.21 → 351.13; Leap 155.31 → 131.70; **Wave 230.91 → 247.29 (LSGAN better)**. AMP's objective degrades as the reference becomes infeasible, but wins on near-feasible ones. **The disconfirming data point.** | **VERIFIED** |
| E8 | WASABI | "when faced with demonstrations that initially seem beyond what the agent can achieve, the discriminator is prone to be driven to optimality, prohibiting a more fine-grained evaluation" — the saturation mechanism in the authors' words. | **VERIFIED** |
| E9 | SFV Table 3 | Motion-Reconstruction ablation, normalised return. Frontflip: RSI **0.404 → 0.658**, ASI **0.403 → 0.708**. Handspring A: RSI 0.391 → 0.464. **Raw per-frame pose-estimator output still trains a working policy at 60–90 % of cleaned.** | **VERIFIED** |
| E10 | SFV Table 2 | Initial-state strategy matters *more* than motion cleanup: Backflip FSI **0.086** vs RSI 0.602 vs ASI **0.741**. Largest single lever in that paper. | **VERIFIED** |
| E11 | SFV | What MR fixes: "poses are predicted independently for every frame… sequencing… produce[s] motions that exhibit artifacts due to inconsistent predictions across adjacent frames"; physics "refine[s] imperfect and noisy pose sequences, compensate[s] for missing frames." | **VERIFIED** |
| E12 | I-CTRL (2405.08726) Table VII | Gaussian noise on reference joint angles: **σ=0° → 91.1 % success, 0.10±0.06 m root error; σ=5° → 73.1 %, 0.13±0.10; σ=10° → 60.6 %, 0.15±0.11.** ⚠️ *Evaluation-time* noise on a policy trained clean, and **tracking-based**. Measures a tracking controller, not AMP. | **VERIFIED** |
| E13 | VAIL (1810.00821) | Video imitation: "Both GAIL and the pixel-loss are unable to learn the running gait. VAIL is the only method that successfully learns." Mechanism: "a discriminator that achieves very high accuracy will produce relatively uninformative gradients." | **VERIFIED** |
| E14 | CoRe | Humanoid imitation success: Walking **73.3 % vs 23.3 % / 26.7 %**; complex self-collision 66.7 % vs 6.7 % / 20.0 %. ⚠️ Page notes no explicit ablation isolating contact-aware refinement. | **REPORTED** |
| E15 | *Retargeting Matters* / GMR (2510.02252) | "artifacts in retargeted data significantly reduce policy robustness, particularly for dynamic or long sequences"; PHC shows "severe ground penetration, which can make it difficult for the policy to learn to track." Tracking-based. | **REPORTED** |
| E16 | STMR (2404.11557) | 44.9–82.3 % reduction in keypoint tracking error (48.7 mm vs 88.4–275.1 mm); targets "noisy motion sources, such as hand-held camera videos". **Numerical contact-preservation ablation not obtainable** from HTML/PDF. | **REPORTED** — contact claim not independently verified |
| E17 | VideoMimic (2505.03729) | Full monocular stack: ViTPose → VIMO → MegaSaM/MonST3R → GeoCalib → BSTRO (foot contact) → temporal smoothness. Tracking-based; needs global root at train time. Only ablation is MoCap pre-training. **No per-CV-stage ablation.** | **REPORTED** |
| E18 | H2O (2403.04436) | "Sim-to-data" cleaning: train a privileged imitator on uncleaned retargeted data; sequences it fails on are deemed infeasible. **~8.5k of 10k** AMASS sequences survived. Cheap mechanical feasibility filter. | **REPORTED** |
| E19 | Uni-Mo (2606.28237) | Production CV gates: CLIP semantic ≥0.85; **geometric gate reprojection error <20 px mean, <100 px max** (70.2 % pass); of survivors **97.6 % track successfully in sim.** | **REPORTED** |
| E20 | npj Robotics 3:32 (2025) | DeepLabCut 2D → Kalman → ST-GCN 3D lift → retarget → AlienGo gallop/backflip. Beats VideoPose3D/LiftPose3D/SemGCN/GLA-GCN on MPJPE. **Full text not accessible** (paywall); discriminator spec unconfirmed; **no noise-tolerance ablation found.** | **REPORTED** — do not lean on this |
| E21 | Frontiers survey (2025) | "effective action priors can be extracted from only a single open-loop gait cycle"; VIAN video imitation "80 % success rate for dog imitation versus 55 % for standard RL". No AMP-vs-DeepMimic robustness comparison exists. | **REPORTED** |
| E22 | HumanMimic (2309.14225) | BCE loss showed "considerable relative fluctuation… volatile changes to the style reward and destabilizing the training phase", with collapse — "the robot only learned a tiptoe walking gait close to the standing posture". W-1 + soft boundary fixed it. | **VERIFIED** |
| E23 | MimicKit (2510.13794) | "AMP exhibits poor tracking performance, since the policies are trained using a general distribution-matching objective"; "more prone to converging to local optima, especially for challenging, highly dynamic motions." **The honest cost of AMP's tolerance.** | **REPORTED** |

**The honest gap:** no study deliberately corrupts a reference clip and measures
AMP policy quality vs corruption level. E12 is closest and is tracking-based and
evaluation-time. **The AMP dose-response curve is genuinely unmeasured.**

---

## 4. Error modes ranked (for `amp-rsl-rl`'s narrow obs vector)

**Rank 1 — high-frequency joint-angle jitter.** Mechanism verified by code +
arithmetic (§1). 1° iid jitter already puts ~45 % noise power into 15 of 36 obs
dims. Supported indirectly by E2, E8, E13, E11. **Entirely fixable by filtering,
at no cost** — which is why it is rank 1 *and* cheap.

**Rank 2 — foot sliding / contact-schedule violation.** `joint_vel` and
`base_lin_vel` sit in the *same* vector, so a clip where legs cycle at speed A
while the root translates at speed B encodes a transition the policy structurally
cannot match. The one error mode **not** filtered for free by physics. Evidence
E14, E15, E16 — ⚠️ **all tracking-based**; extension to AMP is a mechanistic
argument, not a measurement.

**Rank 3 — wrong root velocity magnitude.** 3 of 36 dims. Low impact alone. Real
risk is second-order: the task reward tracks a *commanded* velocity, so if the
reference's implied speed disagrees with the command, task and style rewards
fight.

**Rank 4 — missing DOFs.** Low impact if handled right (E6: 10 base-only dims
sufficed for a backflip; 15-of-25 is luxurious). **Catastrophic if handled
wrong** — `motion_loader.py` L470–485 silently zero-fills:
```python
for frame in data["joint_positions"]:
    arr = np.zeros((len(idx_map),), dtype=frame.dtype)
    for i, src_idx in enumerate(idx_map):
        if src_idx is not None:
            arr[i] = frame[src_idx]
```
The discriminator then learns "real motion has spine yaw pinned at exactly 0".
**Excluding uncovered joints from the AMP obs group is correct — do not regress
it.**

**Rank 5 — systematic bias / bone lengths / scale.** Largely free. Overall scale
is dimensionless in this obs vector (joint angles are scale-free; scale affects
only `base_lin_vel`, which we synthesise). Limb-length *ratios* matter, **but
disappear entirely if we fit the rig's own joint angles to 2D keypoints** rather
than lifting free 3D points. Constant joint offsets survive as a stable but odd
gait — annoying, not fatal.

**Rank 6 — temporal gaps.** Genuinely unmeasured. AMP has no phase variable, so a
gap just removes samples. The hazard is interpolating *across* a gap and creating
an implausibly fast transition. **Mitigation: split into several clean sub-clips
rather than interpolate.** `AMPLoader` takes `datasets: Dict[str, float]` with
per-clip weights (L373–380, **VERIFIED**) — multi-clip is first-class and free.

**Boundary on "physics filters errors for free":** E1 says infeasible transitions
still yield an effective objective; E7 draws the line — as the reference moves off
the feasible manifold the LSGAN objective degrades sharply, while on near-feasible
references it is fine or better. **Errors that keep the reference near the
dynamically feasible manifold are absorbed; contact violations and white-noise
velocities are not.**

---

## 5. Recommended preprocessing stack

1. **Fit rig joint angles directly to 2D keypoints; never lift free 3D then
   retarget.** Kills bone-length, scale and depth error at once. Gate on
   reprojection error at Uni-Mo's **<20 px mean / <100 px max** (E19).
2. **Temporal filtering — mandatory.** Target ≤1° residual. Practitioner defaults:
   **Savitzky-Golay degree 3, window ≈ 1/8 s** ("a degree three polynomial assumes
   locally that the data has a jerk of zero, which is a good prior for motion
   capture data" — **REPORTED**, Ubisoft neural mocap solver). Biomechanics work
   prefers **zero-phase Butterworth** when derivatives matter (**REPORTED**,
   Sensors 21:4580) — and derivatives are exactly what matters here. Recommend
   `filtfilt` Butterworth, cutoff ~6–8 Hz for a walk, and **validate on the
   differentiated signal, not the angle signal.**
3. **Contact-based foot locking.** Detect stance from hoof height + velocity;
   re-solve so stance hooves are world-stationary and root translation is the
   integral of stance-foot motion. Fixes rank-2 and rank-3 simultaneously and
   makes them consistent by construction.
4. **FK grounding for root height.** Set per-frame root Z so the lowest hoof
   touches z=0. Also fixes an existing hazard: `reset_from_amp_motion` spawns at a
   hardcoded `base_height_m=0.93` regardless of pose.
5. **Sim-based feasibility filtering** (E18). Run candidates through
   `validate_motion.py` + a short kinematic replay; discard failures. H2O kept
   85 % of AMASS this way. High value/effort ratio.
6. **Split into multiple short clean clips** rather than one long gappy one.
7. **Discriminator-side regularisation is part of the budget.** Keep the gradient
   penalty on (E2; `compute_grad_pen`, `lambda_=10` + R1, **VERIFIED**), keep the
   discriminator small, keep `entropy_coef` 0.002–0.005, and consider
   **`loss_type="Wasserstein"`** — `amp-rsl-rl`'s `Discriminator` supports it with
   a soft `tanh(eta_wgan · logit)` bound (**VERIFIED**). That is the WASABI (E7) /
   HumanMimic (E22) fix for unreachable references, and it is a one-line config
   change. Worth an A/B.
8. **Full physics-based trajectory optimisation (STMR's TMR)** — high cost, and
   given E1 plus AMP not tracking per-frame, **probably not worth it for the AMP
   path.** Worth it if switching to tracking.

**Also: keep RSI.** E10 is the largest single effect size in this memo.

---

## 6. Unknowns

1. **The AMP noise dose-response curve does not exist.** §1's table is arithmetic
   on our own clip plus a mechanistic argument. **Cheap to measure:** noise
   `deer_walk.npy` at σ ∈ {0, 0.5, 1, 2, 3, 5}°, train six short AMP runs, plot
   style reward. One day of GPU.
2. **Whether foot-slide/contact error poisons AMP (vs tracking) is unmeasured.**
   Testable: scale root speed to 0.5× and 2× foot-odometry and watch style reward.
3. **Minimum clip length is not established.** Data points: 4.5 s (E5), 13.6 s,
   434 s (E4), "a single open-loop gait cycle" (E21, secondary). **Our 1.03 s /
   31 frames is below every measured point.** Mitigation: concatenate cycles, add
   the mirrored clip via `symmetry_cfg` (supported, **VERIFIED**).
4. **Temporal-gap tolerance: no data at all.**
5. **npj Robotics (E20) full text not obtained** — the closest prior work.
6. **STMR's contact-preservation claim** is plausible and mechanistically sensible
   but **not** independently verified numerically.
7. **Disconfirming evidence not dismissed:** E7's Wave row (LSGAN wins), E10
   (initial-state strategy dominates motion cleanup), E23 (AMP's tolerance is
   bought by *not tracking well* — if the goal is for the deer to look like the
   video deer, AMP is the wrong tool and this whole analysis changes).

---

## Sources

Code read locally: `amp_rsl_rl` v2.2.3 (`utils/motion_loader.py`,
`networks/discriminator.py`, `runners/amp_on_policy_runner.py`); Isaac Lab
`humanoid_amp_env.py`; project `training/envs/deer_usd_amp_env_cfg.py`,
`training/envs/deer_rewards.py`, `assets/motions/deer_walk.npy`.

- AMP — https://arxiv.org/abs/2104.02180 · https://ar5iv.labs.arxiv.org/html/2104.02180
- ASE `humanoid_amp.py` — https://raw.githubusercontent.com/nv-tlabs/ASE/main/ase/env/tasks/humanoid_amp.py
- IsaacGymEnvs — https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/humanoid_amp.py
- AMP_for_hardware — https://github.com/escontrela/AMP_for_hardware
- AMP as reward substitutes — https://ar5iv.labs.arxiv.org/html/2203.15103
- SFV — https://ar5iv.labs.arxiv.org/html/1810.03599
- WASABI — https://ar5iv.labs.arxiv.org/html/2206.11693 · https://github.com/martius-lab/wasabi
- I-CTRL — https://arxiv.org/html/2405.08726
- VAIL — https://ar5iv.labs.arxiv.org/html/1810.00821
- HumanMimic — https://arxiv.org/html/2309.14225v3
- STMR — https://arxiv.org/html/2404.11557
- VideoMimic — https://arxiv.org/html/2505.03729v1
- Retargeting Matters / GMR — https://arxiv.org/abs/2510.02252
- CoRe — https://tmjeong1103.github.io/CoRe/
- H2O — https://arxiv.org/abs/2403.04436
- Uni-Mo — https://arxiv.org/html/2606.28237
- npj Robotics 3:32 — https://www.nature.com/articles/s44182-025-00048-x *(full text not obtained)*
- MimicKit — https://arxiv.org/html/2510.13794v1
- DeepMimic — https://ar5iv.labs.arxiv.org/html/1804.02717
- Imitation-learning survey — https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1678567/full
- Neural mocap solver (SG settings) — https://theorangeduck.com/media/uploads/other_stuff/neural_solver.pdf
- Butterworth vs SG for derivatives — https://www.mdpi.com/1424-8220/21/13/4580
