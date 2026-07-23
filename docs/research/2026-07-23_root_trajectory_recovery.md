# Research memo — global root recovery for large-translation locomotion

**2026-07-23.** **VERIFIED** (read in primary source / code) · **REPORTED** ·
**UNVERIFIED**.

Distilled into `docs/CV_PIPELINE_DESIGN.md`. This file keeps the evidence.

---

## 1. Bottom line

**The global-root problem, as posed, does not exist for this project.**

Three facts, all **VERIFIED by code** in `amp-rsl-rl`:

1. `MotionData.get_amp_dataset_obs()` returns exactly
   `cat(joint_positions, joint_velocities, base_lin_velocities_{local|mixed},
   base_ang_velocities_{local|mixed})` (L172–206). No root position, height, or
   quaternion.
2. `root_position` is consumed at exactly **one** place:
   `_compute_raw_derivative(resampled_base_positions, simulation_dt)` →
   `base_lin_velocities_mixed`, then rotated into body frame (L502–524).
   `root_quaternion` enters only as that rotation and a log-map angular velocity.
3. `get_state_for_reset()` (L208–239, the RSI path) returns
   `(base_quat, joint_pos, joint_vel, base_lin_vel, base_ang_vel)` — **no base
   position**. Spawn height comes from env config.

So `root_position` is a **velocity carrier**. Absolute position, accumulated
drift, a constant offset, a slow curve — all invisible. Only the first difference
survives.

Cross-check on the other major quadruped-AMP codebase (**VERIFIED**):
Escontrela's `AMP_for_hardware` slices AMP obs as joint pos, local toe pos, lin
vel, ang vel, joint vel, and separately concatenates `root_pos[2]` — **root height
only**. Root x/y and quaternion excluded there too.

**Two independent AMP implementations, neither of which can see where the animal
is.**

### The corollary that inverts the problem

`_compute_raw_derivative` is a bare forward difference with **no smoothing**
(L450–452), applied *after* `interp1d` (default **linear**) upsamples to
`simulation_dt` (L408–415, L497–511). **VERIFIED.**

- Linear interp + forward difference ⇒ `base_lin_vel` is **piecewise-constant**,
  stepping once per *source* frame. Upsampling 30 → 200 Hz adds resolution, not
  information.
- Per-frame root noise passes through at gain `1/dt_source`. At 30 fps, **1 cm of
  root jitter between adjacent frames = 0.30 m/s velocity error** — 37 % of the
  0.8 m/s reference speed. At 60 fps, 0.60 m/s.

Every video root estimator surveyed has per-frame noise well above 1 cm at
wildlife distances. **Piping a raw video-derived root into this loader would be
actively worse than the current synthetic constant.** Root *accuracy* is
irrelevant; root *smoothness* is everything.

### Recommendation

**Do not build global-root recovery. Build a root-velocity synthesiser, in two
stages.**

- **Stage 0 (current, defensible):** constant forward speed, straight line.
- **Stage 1 (~1 day):** replace the hand-picked 0.8 m/s with `stride length ×
  cadence` from the footage; add a yaw-rate track if the animal turns. Still
  synthetic, now grounded in the clip.
- **Stage 1b (hours, genuinely recommended):** add a sinusoidal surge at the
  measured stride frequency, amplitude from our own sim rollouts. See §5 — this
  removes a specific degeneracy at near-zero cost and is the control condition
  against which Stage 2 must be measured.
- **Stage 2 (~1–2 weeks, the real win):** **contact integration.** Fit the URDF to
  2D keypoints, detect the planted hoof, integrate root displacement from leg
  kinematics. Recovers the thing constant-speed genuinely lacks: the within-stride
  surge/heave ripple. Needs no SLAM, no background, no metric depth, no camera
  model — only correct bone lengths (we have them) and correct contact timing.

Contact integration's famous weakness is **cumulative drift**. Our pipeline
differentiates immediately, so drift is exactly the error we do not care about.
**That coincidence is the whole argument.**

**Do not** adapt WHAM/TRAM/GVHMR — all welded to SMPL + AMASS-trained human
priors, all would need a deer motion prior that does not exist, all
licence-hostile to a public repo.

---

## 2. Survey — human global-motion recovery

EMDB-2 numbers as re-evaluated in one place by GVHMR (third-party normalised
comparison, more trustworthy than cross-quoting self-reported tables).
**VERIFIED** from https://arxiv.org/html/2409.06662v1.

| Method | Mechanism | Static bg? | Metric scale? | Subject-agnostic? | Code / licence | WA-MPJPE₁₀₀ / W-MPJPE₁₀₀ / RTE% |
|---|---|---|---|---|---|---|
| **GLAMR** (CVPR'22) | **No SLAM.** cVAE + biLSTM trajectory predictor maps local pose → egocentric root trajectory; joint optimisation of trajectory + camera extrinsics against 2D keypoints | No | Yes | ❌ AMASS, SMPL | NVlabs/GLAMR · **NVIDIA non-commercial** | 280.8 / 726.6 / 11.4 % |
| **TRACE** (CVPR'23) | End-to-end 5D regression in a learned world memory | No | Weak | ❌ | — | 529.0 / 1702.3 / 17.7 % (jitter 2987) |
| **SLAHMR** (CVPR'23) | DROID-SLAM on background → relative camera up-to-scale; **HuMoR** prior resolves scale | Effectively yes | Yes, from the prior | ❌ PHALP+, SMPL-H, HuMoR | vye16/slahmr · **MIT** (SMPL-H is not) | 326.9 / 776.1 / 10.2 % · ~40 min/100 frames |
| **PACE** (3DV'23) | Joint human+camera optimisation with learned prior | Partly | Yes | ❌ | NVlabs | — |
| **COIN** (2024) | Control-inpainting diffusion prior disentangles human/camera | Partly | Yes | ❌ | — | beats PACE by 125.5 mm W-MPJPE on RICH |
| **WHAM** (CVPR'24) | Uses **only camera angular velocity** from SLAM or gyro; AMASS decoder predicts egocentric root velocity; **contact-aware refinement** | **No** — rotation only | Yes | ❌ SMPL, AMASS, AMASS contact labels | yohanshin/WHAM · **MIT** | 135.6 / 354.8 / 6.0 % |
| **TRAM** (ECCV'24) | DROID-SLAM with **dual masking** (YOLOv7→SAM) to excise the subject; **ZoeDepth** on the *background* gives scale by robust least squares | **Yes, hard requirement** | **Yes** | ⚠️ camera half yes, body half SMPL | yufu-wang/tram · **MIT** | self-reported 76.4 / 222.4 / **1.4 %** *(not directly comparable)* |
| **GVHMR** (SIGGRAPH Asia'24) | **Gravity-View coordinates** — per-frame frame with y‖gravity, z‖view axis. Needs **camera rotation only**; no translation, no SLAM scale | **No** | Yes | ❌ SMPL-X, AMASS+BEDLAM | zju3dv/GVHMR · **ZJU non-commercial, derivatives must be open-source + non-commercial** | **111.0 / 276.5 / 2.0 %** (best) |
| **OfCaM** (2024) | **Closed-form**: `s = d_absolute / d_slam` using the depth of a *contact joint* vs SLAM keypoint depth; median over frames | Yes | **Yes** | ⚠️ idea agnostic, impl SMPL | "upon acceptance" — **UNVERIFIED** | 60 % RTE reduction |
| **HumanMM / HTD-Refine** (2025–26) | multi-shot / high-order temporal alignment | — | — | ❌ | — | REPORTED to beat GVHMR/TRAM |

### What would have to be replaced for a deer

| Component | Human-specific? | Deer replacement | Feasible? |
|---|---|---|---|
| SLAM on background (DROID/DPVO) | **No** | none needed | ✅ reusable |
| Dynamic-subject masking (SAM) | No | none | ✅ |
| ZoeDepth background scale | No | none | ✅ — but see §3 on telephoto |
| SMPL / SMPL-X | **Yes** | SMAL, or our URDF | ⚠️ SMAL is MPI non-commercial; our URDF is better |
| AMASS trajectory decoder (WHAM/GVHMR/GLAMR) | **Yes — the crux** | a deer mocap corpus of AMASS scale | ❌ **does not exist** |
| HuMoR transition prior | **Yes** | deer cVAE prior | ❌ same blocker |
| AMASS foot-contact labels | **Yes** | our own sim rollouts | ⚠️ possible but circular |
| 2D keypoint detector | **Yes** | DeepLabCut / SuperAnimal / APT-36K | ✅ |

**The universal blocker is the learned motion prior.** Everything beating ~10 %
RTE does so by knowing what human motion looks like. GVHMR's Gravity-View trick —
the most elegant idea here, needing camera *rotation only* — is architecturally
subject-agnostic, but its trajectory head is AMASS+BEDLAM-trained. Retraining it
for deer is the same "we need a big deer mocap dataset" problem that drove this
project to video. **Circular.**

---

## 3. SLAM / camera-pose on wildlife footage

Recipe: mask subject → SLAM on background → camera trajectory up to scale → pin
scale → place subject.

**What breaks:**

| Failure mode | Evidence |
|---|---|
| **Long focal length / telephoto** | TRAM verbatim: *"SLAM's dependence on known focal length limits its applicability in many in-the-wild cases"*; depth estimation degrades at extreme focal lengths. **VERIFIED** |
| **Low parallax** | SLAHMR fails on *"predominantly rotational camera motion or co-linear motion between humans and cameras"* and *"cannot distinguish translational from rotational motion with large focal length."* **VERIFIED.** This is the canonical wildlife setup: long lens, tripod pan, animal moving parallel to the sensor. |
| **Subject fills frame** | TRAM cannot work *"without visible static background structure."* **VERIFIED** |
| **Moving foliage** | Violates static-scene assumption; DROID-W exists because DROID-SLAM *"assume[s] a static environment, leading to tracking failures in the presence of motion."* **VERIFIED** |
| **Unknown/varying intrinsics (zoom)** | DROID-SLAM assumes known focal length; MegaSaM added focal optimisation inside BA. **VERIFIED** |

**The strongest disconfirming datapoint**, from a wildlife-domain paper rather
than a CV benchmark: the plains-zebra drone study observes that for naturalist
videos cameras are typically stationary or shot from a long distance so parallax
is minimal, and therefore **abandons SfM entirely** for homography-based frame
registration, with metric scale set by physically measuring reference distances
at the site. https://arxiv.org/pdf/2505.16882 ·
https://github.com/neuroinformatics-unit/zebras-stitching (**MIT**). **VERIFIED.**
A domain expert independently concluding SLAM is the wrong tool for this footage
class.

Second: **WildLIFT** (2026), a species-agnostic drone-wildlife 3D pipeline,
rejected both monocular depth networks (*"suffer domain shift on oblique aerial
viewpoints"*) and SfM (*"fail when dynamic animals dominate the frame"*), settling
on CUT3R — then states it is **scale-agnostic by design**, offering no metric
mechanism. **VERIFIED** (https://arxiv.org/html/2604.24718). Repo
`vandyshukla04/wildlift-framework`: 0 stars, no licence, three ~600-byte entry
scripts, `backends/` and `wildlift/` empty. **Effectively a shell. VERIFIED via
GitHub API.**

### Feed-forward geometry models (2025–26)

| Model | Metric? | Licence | Wildlife suitability |
|---|---|---|---|
| DUSt3R | No — scale-invariant | **CC BY-NC-SA 4.0** | dynamic-scene failure |
| MASt3R | Partly | **CC BY-NC-SA 4.0** | same |
| VGGT | Camera + pointmaps | **Meta bespoke research** | *"in dynamic scenes the same global attention can mix camera ego-motion with object motion"* |
| CUT3R | No | **CC BY-NC-SA 4.0** | used by WildLIFT |
| MegaSaM | **Yes** | project page only, release **UNVERIFIED** | best for low parallax: joint focal optimisation, uncertainty-aware global BA, motion-probability masks |
| DROID-W | camera pose | **Apache-2.0** (407★) | fixes dynamic distractors; does *not* address focal/parallax/telephoto |

**Every non-commercial-friendly feed-forward option is CC BY-NC-SA or bespoke.**

### Metric scale — mechanisms assessed

1. **Learned metric depth on background** (TRAM's trick). **Unreliable at wildlife
   range.** KITTI caps ~80 m and far-field is weakest; one concrete failure:
   ground truth 65.8 m predicted 44.6 m (~32 % error) [REPORTED]. **No paper
   evaluates Metric3D/UniDepth/Depth Pro at 10–100 m on telephoto wildlife
   imagery — a genuine UNKNOWN**, and the absence is itself informative.
2. **Known subject dimensions** (OfCaM generalised). Conceptually strongest: we
   know shoulder height and limb lengths from the URDF. `s = known /
   apparent`. **The mechanism to steal, not the method.**
3. **Known intrinsics + known subject size.** Same, without SLAM. Reduces to §5.
4. **Physical reference measured on site** — what the zebra study did. Boring,
   works.

---

## 4. The contact-integration route — deep assessment

**Recommended. Under-appreciated in the vision literature precisely because vision
papers are graded on the metric it is worst at (absolute drift) and ignore the one
it is best at (instantaneous velocity fidelity).**

**Principle:** a planted foot is stationary in the world. Given joint angles and
known bone lengths, FK gives the stance foot in body frame; root world
displacement is the negation of the stance foot's body-frame displacement, rotated
by root orientation. The legs are an odometer.

### Evidence — humans

**TransPose** (SIGGRAPH 2021) — **VERIFIED** (https://ar5iv.labs.arxiv.org/html/2105.04605):
- biRNN predicts per-foot contact probability from leaf-joint positions.
- Ground-truth contact labels by thresholding foot displacement at **0.008
  m/frame** — pure heuristic, no manual labels.
- Two estimators fused by contact probability `s`: `s<0.5` → learned RNN velocity;
  `0.5≤s<0.9` → linear blend; `s≥0.9` → supporting-foot velocity. The learned
  branch exists solely to cover flight phases.
- Cumulative distance error on TotalCapture: **~12 cm @ 1 s, ~41 cm @ 5 s** (vs
  50.8 cm RNN-only). Read as velocity error: ~0.12 m/s, ~15 % of a 0.8 m/s walk —
  and mostly slow drift, not per-frame noise.

**Rempe et al., ECCV 2020** — **VERIFIED** (https://ar5iv.labs.arxiv.org/html/2007.11678):
- Contact detector: 5-layer MLP (1024/512/128/32/20), trained on 65 mocap
  sequences retargeted to 13 Mixamo characters, >100k frames, **automatically
  labelled** (movement <2 cm AND within 5 cm of ground plane).
- **Accuracy: 94.6 % synthetic, 92.7 % real video, F1 = 0.949.**
- Then **centroidal-dynamics** trajectory optimisation for COM position +
  orientation with no-slip and no-penetration constraints.
- Cost: **30–60 min per 2-second clip.** Assumes unoccluded feet, single ground
  plane, no object interaction.
- Code: https://github.com/davrempe/contact-human-dynamics — populated, custom
  licence, last push 2021.

**The contact detector is subject-agnostic in construction** — it consumes joint
positions/velocities, not SMPL parameters, and its labels come from a geometric
heuristic. **We could regenerate its training set from our own sim rollouts.**

### Evidence — legged robots in production

- Hartley et al., **Contact-Aided Invariant EKF** (IJRR 2020,
  https://arxiv.org/pdf/1805.10410) — canonical; Lie-group estimator fusing IMU +
  kinematics + contact.
- Drift **REPORTED**: ~**10 % of distance travelled** for naive leg odometry;
  **<1 %** with calibrated kinematics (Cerberus).
- Dominant failure sources **REPORTED**: inaccurate leg kinematics and
  **fault-prone contact detection** — "wrong contact estimation is one of the main
  challenges."

Asymmetry worth noting: robots get IMU + joint encoders + force sensors and still
drift 1–10 %. We get none of those — but we also do not need the absolute position
their operators need.

### Contact detection from video, quadrupeds

- Horses: hoof contacts by **peak-and-valley detection on Butterworth-smoothed 2D
  hoof traces** centred on the animal's back. Markerless. **REPORTED** — no F1.
- IMU reference for the same task: stance-phase bias **<1.79 / 1.46 / 3.45 /
  −1.94 %** of stride duration. **REPORTED.**
- Hard case: *"In the fast running gait of a quadruped, the total gait cycle could
  be less than 300 ms, and the contact period for each leg can be less than 40
  ms."* **REPORTED.** At 30 fps that is **1.2 frames of stance** for a gallop.
  **Viable for walk/trot, marginal for canter, unusable for gallop at consumer
  frame rates.**
- Deer-specific contact-detection accuracy: **UNKNOWN.**

### Concrete algorithm

1. **2D keypoints** per frame including all four hooves.
2. **Per-frame URDF fit** by 2D reprojection IK. Bone lengths are known and
   metric — a far stronger depth cue than generic monocular depth: limb
   foreshortening + known segment lengths pins root depth.
3. **Stance detection**: hoof at a local minimum in fitted-rig height AND image
   velocity below threshold. Optionally train Rempe's MLP on our own rollouts.
4. **Integrate**: `Δroot_world[t] = −R_root[t] · Δ(FK(stance_hoof))_body[t]`; blend
   across double-support with TransPose's 0.5/0.9 schedule.
5. **Heading**: integrate root yaw from the fitted rig, not SLAM.
6. **Smooth before writing** — spline or Savitzky-Golay on `root_position` at
   source frame rate. Given §1, **not optional**.
7. **Cross-check** mean speed against Alexander's relation
   `v = 0.25·g^0.5·SL^1.67·h^−1.17` (**20–30 % uncertainty**, validated mouse to
   elephant). Disagreement >30 % means contact detection is wrong.

**Needs:** correct bone lengths ✅, correct contact timing ⚠️, adequate frame rate
⚠️.
**Does not need:** SLAM, static background, metric depth, known intrinsics, known
focal length, a static camera, an AMASS-equivalent prior, or a visible background.

**For a panning/tracking/handheld camera — the case Uni-Mo leaves unsolved —
contact integration is completely indifferent to camera motion, because it never
uses the background. The strongest single argument here.**

---

## 5. The "cheapest sufficient" analysis

**What the discriminator sees from root fields:** `base_lin_vel` (3, body frame)
and `base_ang_vel` (3), per frame.

**What constant-speed straight-line synthesis gets RIGHT:** mean forward speed (if
chosen correctly); zero lateral velocity and yaw rate (correct for straight
walking); **zero noise** — a real and underrated virtue given §1.

**What it gets WRONG — one thing:** the **within-stride velocity ripple.** A real
walking quadruped surges and decelerates within each stride and heaves vertically
at 2× stride frequency. A constant synthesis has *zero* oscillation in
`base_lin_vel`. That is a trivially learnable discriminator feature: the reference
has a dead-flat velocity channel while any simulated deer has an oscillating one.
The discriminator can win on that channel alone, saturate, and collapse style
reward — **exactly the fingerprint in `CLAUDE.md`.**

**This reframes the whole question.** Recovering "global root trajectory" buys
almost nothing. Recovering **the per-stride velocity profile** buys the one
channel the current reference is degenerate in. Contact integration produces
exactly that profile as its primary output. Learned camera-frame root regressors
(npj/STGNet, GLAMR-style) produce it *badly* — trained on MPJPE-type losses that
reward mean accuracy and are indifferent to the high-frequency component.

| Tier | Effort | Gets you | Still breaks |
|---|---|---|---|
| **0. Constant 0.8 m/s** (current) | done | valid mean; smooth | flat velocity channel → saturation risk |
| **1. Speed from stride × cadence** | hours | mean grounded in the clip; Alexander cross-check ±20–30 % | still flat |
| **1b. + sinusoidal ripple at measured stride frequency** | hours | non-degenerate velocity channel at ~zero cost | ripple *shape* is a guess; phase-locking approximate |
| **2. Contact integration** | 1–2 weeks | true ripple, true phase alignment, true mean, handles turns | contact detection on fast gaits; needs good 2D keypoints |
| **3. Full world-frame SLAM** | months | absolute position — **which nothing consumes** | telephoto/parallax/foliage/scale, non-commercial licences |

**Tier 1b is a serious suggestion:** a few hours, not a research problem, removes
the specific degeneracy, and is the control against which Tier 2 is measured.

**When the animal turns:** yaw rate becomes non-zero, `base_ang_vel.z` goes live,
and `base_lin_vel` acquires a body-frame lateral component. A straight-line
synthesis is then wrong in two channels. Turning is recoverable cheaply *without*
3D reconstruction if the camera is a static tripod (yaw directly observable from
2D keypoint layout). **With a panning camera it needs camera rotation estimation**
— and this is where GVHMR's insight applies: **rotation only**, no translation, no
scale. Rotation-only estimation from a pan is the *easy* part of SLAM (it is what
image stabilisation does, and what the zebra study used homography for) and dodges
the parallax/telephoto failure modes. **UNVERIFIED for this use case**, but
mechanically sound and cheap to test.

---

## 6. Filming protocol — making the problem free

**Tier A — "root trajectory is free"**

1. **Rigid tripod. No pan, tilt, or zoom.** Removes camera-subject ambiguity
   entirely; makes GVHMR/WHAM/TRAM irrelevant.
2. **Fixed focal length**, recorded. Prefer *moderate* over telephoto — telephoto
   kills parallax, compresses depth, breaks metric depth models.
3. **Camera axis roughly perpendicular to the animal's path**, animal crossing
   left-to-right. Maximises observability of forward displacement.
4. **Camera elevated 2–4 m, tilted down 15–30°**, so the ground plane is
   well-conditioned for homography. A flat-on view makes ground-plane fitting
   ill-posed.
5. **Four known-position ground markers** in the crossing zone → **ground-plane
   homography** directly. No calibration object → no scale.
6. **A vertical scale reference** (1 m rod) at the animal's working distance.
7. **≥60 fps**, ideally 120 for anything faster than a trot (§4, 40 ms stance).
8. **Whole animal in frame including all four hooves**, unoccluded by vegetation.
   Rempe's assumption is the binding one.
9. **Flat, uniform ground** — every contact method assumes a single plane.

With Tier A, root recovery reduces to: detect hoof-ground contact pixel → apply
the ground homography → world position. **Closed form. No SLAM, no depth model,
no learned prior.** This is the monoplotting technique used for UAV wildlife
geolocalisation, and the scale strategy the zebra study used.

**Tier B — acceptable degradation:** handheld but static-intent (homography
stabilisation recovers the ground plane); pan-only tripod (recover **rotation
only** — cheap, robust, this is stabilisation not SLAM). Add a known-length object
at the subject's distance as scale fallback.

**Tier C — now doing research:** camera translates (dolly, walking operator,
drone) → real SLAM, §3's failure modes live. Telephoto + tripod pan + moving
foliage is worst case — **use contact integration and ignore the background.**

**Tier A and contact integration are complementary, not redundant.** Tier A gives
an independent ground-truth root trajectory to *validate* the contact integrator
against. Building both and comparing is the strongest evidence available, and a
good portfolio artifact.

---

## 7. Quadruped prior work — repo state verified

**All repo states VERIFIED via GitHub API on 2026-07-23.**

| Work | World-frame root? | Repo state | Licence |
|---|---|---|---|
| **"Unleashing Infinite Motion"** (arXiv 2606.28237) — the paper the brief calls *Uni-Mo*; actual title differs | Yes: state `s_t = (p_t, φ_t, θ_t)` via URDF-anchored kinematic optimisation with reprojection + foot-contact constraints | ⚠️ **GaoLii/Quad-Imaginarium: README.md only. 9 KB. 3 stars. No licence file. No code.** Data "upon paper acceptance." | claimed CC BY 4.0 in paper; **no LICENSE in repo** |
| **npj Robotics 2025**, Zhao et al. | ⚠️ **Learned camera-frame regression, not world frame.** STGNet has two heads: local joint positions, and "the trajectory of the root joint to obtain the global position". Trained on **AI4Animation + DigiDogs mocap projected through 4 virtual cameras with known K, E**. Loss = normalised MPJPE on the root. It *hallucinates* root depth from a learned prior. Paper admits *"performance may degrade under non-frontal camera angles."* | ⚠️ **arclab-hku/Imitation_from_video: populated but it is the RL half ONLY** — a `legged_gym`/`AMP_for_hardware` fork. **The STGNet video→3D pipeline is not in the repo.** Shipped data: `datasets/video_motion_limp_aliengo/` = **two .txt files** | **GPLv2**, "free for academic usage with several patents," commercial contact required. ⛔ portfolio-hostile |
| **DogMo** (2510.24117) | Multi-view **RGB-D** capture, 1.2k sequences, 10 dogs; three-stage SMAL fitting. World-frame root comes from the *rig*, not monocular inference. Monocular is a benchmark *setting*, not a solved capability | release **UNVERIFIED** | SMAL ⇒ MPI non-commercial |
| **PPR** (ICCV'23) — differentiable physics + rendering | Yes in principle: joint global scale alignment, contact estimation, physically plausible root | ⚠️ `gengshan-y/ppr` main = LICENSE + README only. Not a shell exactly — a *pointer*: real code in `lab4d-org/lab4d` `ppr` branch + `gengshan-y/ppr-diffphys`. Both real. Must assemble three repos; shipped data is one cat sequence | MIT (all three) ✅ |
| **Lab4d / BANMo / RAC** | Test-time-optimisation 4D reconstruction; object-centric, not world-frame locomotion | populated, 354★, active to 2025-02 | **MIT** ✅ |
| **Animal Avatars** (ECCV'24, Meta) | SMAL + CSE; appearance-focused | populated | Meta custom; SMAL non-commercial |
| **MagicPony / Ponymation** | category-level shape; no world-frame root | — | — |
| **ReWild4D** (2607.00157) | **UNVERIFIED** — PDF not text-extractable; https://syl-322.github.io/ReWild4D/ | unknown | unknown |
| **RLWAV** (2412.04273) — **the important negative result** | **Extracts no 3D motion at all.** Trains a video *action classifier* on wild animal footage, then uses the classifier's score on renders of the simulated robot as the RL reward. Learned still/walk/run/jump on a real Solo-12 | no code link | — |

**RLWAV deserves emphasis.** A peer-reviewed demonstration (CoRL LocoLearn 2024
best paper) that quadruped locomotion skills can be learned from wild animal video
while **completely sidestepping motion extraction, root trajectory, camera motion,
scale, and contact.** Not a drop-in for AMP, but strong evidence this problem
class is over-specified by default.

**Pattern:** every quadruped precedent claiming a global root either (a) regresses
it from a learned prior in camera frame assuming a near-fixed frontal camera (npj,
Uni-Mo), or (b) gets it from a physical capture rig (DogMo). **Nobody has
recovered world-frame quadruped root motion from moving-camera monocular wildlife
video.** The gap is real and is not closing.

### Licensing summary

| Safe (permissive) | Contaminating |
|---|---|
| DROID-SLAM (BSD-3), DPVO (MIT), DROID-W (Apache-2.0) | **SMPL / SMPL-X / SMAL** — MPI non-commercial |
| WHAM (MIT), TRAM (MIT), SLAHMR (MIT) — *but all pull SMPL* | **GVHMR** — ZJU academic/non-profit; derivatives must be open-source AND non-commercial |
| Lab4d / PPR / ppr-diffphys (MIT) | **DUSt3R / MASt3R / CUT3R** — CC BY-NC-SA 4.0 |
| **amp-rsl-rl (BSD-3)** ✅ current stack | **VGGT** — Meta bespoke |
| zebras-stitching (MIT) | **arclab-hku/Imitation_from_video** — GPLv2 + patents |
| contact-human-dynamics — custom, read LICENSE before vendoring | GLAMR — NVIDIA non-commercial |

**The contact-integration route touches none of the contaminating licences.** It
uses our own URDF, a 2D keypoint detector, and code we write. A further argument
for it.

---

## 8. Unknowns

1. ~~Whether env-side AMP obs matches dataset-side.~~ **RESOLVED after this memo
   was written** — checked directly: env side is
   `joint_pos(15)+joint_vel(15)+base_lin_vel(3)+base_ang_vel(3)`, matching the
   dataset side. No root position on either. §1 holds for our pinned setup.
2. **Contact-detection accuracy on deer from video.** No published numbers for
   cervids. Horse/dog analogues suggest tractable for walk/trot. Hooves are small
   and often occluded by grass.
3. **Whether 30 fps suffices.** Gallop stance may be ~40 ms; no deer-specific
   stance-duration measurements found. For walk, almost certainly fine.
4. **Metric-depth accuracy at 10–100 m on telephoto wildlife imagery.** No
   evaluation exists. The one far-field datapoint (65.8 → 44.6 m) is automotive.
5. **Whether the discriminator actually exploits the flat-velocity degeneracy.**
   Inference from architecture, not measurement. **Cheaply testable**: ablate
   `base_lin_vel` from the AMP obs and see whether style-reward behaviour changes.
   Do this before investing in Tier 2.
6. **ReWild4D (2607.00157)** — could not extract; worth a look given recency.
7. **MegaSaM code release** — project page exists; repo/licence **not verified**.
   If permissive it is the best answer for the low-parallax case.
8. **OfCaM code** — "upon acceptance," **not verified**. The *idea* is reusable
   without their code.
9. **Whether npj's STGNet would transfer to deer.** Trained on dogs; deer
   morphology differs substantially. Code unreleased, so untestable.

---

## Sources

**AMP stack (code-verified):** https://github.com/ami-iit/amp-rsl-rl (BSD-3) ·
https://github.com/escontra/AMP_for_hardware · https://arxiv.org/pdf/2104.02180 ·
https://arxiv.org/pdf/2203.15103

**Human global motion:** https://arxiv.org/abs/2112.01524 (GLAMR) ·
https://github.com/vye16/slahmr (SLAHMR) · https://arxiv.org/pdf/2312.07531
(WHAM) · https://arxiv.org/html/2403.17346v2 (TRAM) ·
https://arxiv.org/html/2409.06662v1 (GVHMR) · https://arxiv.org/html/2407.00574v1
(OfCaM) · https://arxiv.org/html/2310.13768 (PACE) ·
https://arxiv.org/abs/2408.16426 (COIN) · https://arxiv.org/pdf/2306.02850 (TRACE)

**SLAM / geometry:** https://arxiv.org/pdf/2108.10869 (DROID-SLAM) ·
https://github.com/MoyangLi00/DROID-W · https://arxiv.org/html/2412.04463v1
(MegaSaM) · https://arxiv.org/abs/2503.11651 (VGGT) · https://github.com/naver/dust3r
· https://github.com/naver/mast3r · https://github.com/CUT3R/CUT3R ·
https://arxiv.org/html/2403.18913v1 (UniDepth) · https://arxiv.org/html/2404.15506v4
(Metric3Dv2) · https://arxiv.org/html/2410.02073v1 (Depth Pro)

**Contact / odometry:** https://ar5iv.labs.arxiv.org/html/2007.11678 ·
https://github.com/davrempe/contact-human-dynamics ·
https://ar5iv.labs.arxiv.org/html/2105.04605 (TransPose) ·
https://arxiv.org/pdf/1805.10410 (contact-aided InEKF) ·
https://arxiv.org/pdf/2208.04598 (UnderPressure) ·
https://github.com/Shimingyi/MotioNet · https://arxiv.org/pdf/2008.08880 (PhysCap)

**Quadruped / animal:** https://www.nature.com/articles/s44182-025-00048-x ·
https://github.com/arclab-hku/Imitation_from_video · https://arxiv.org/html/2606.28237
· https://github.com/GaoLii/Quad-Imaginarium · https://arxiv.org/abs/2510.24117
(DogMo) · https://github.com/gengshan-y/ppr · https://github.com/lab4d-org/lab4d ·
https://arxiv.org/abs/2412.04273 (RLWAV) · https://arxiv.org/abs/2004.00784 ·
https://arxiv.org/pdf/1810.03599 (SFV) · https://arxiv.org/pdf/2505.16882 ·
https://github.com/neuroinformatics-unit/zebras-stitching ·
https://arxiv.org/html/2604.24718 (WildLIFT)

**Biomechanics:** Alexander 1976 ·
https://cdnsciencepub.com/doi/10.1139/z93-095 (escape gaits of white-tailed deer)
· https://doi.org/10.3390/s22030703 (equine stance-phase detection)
