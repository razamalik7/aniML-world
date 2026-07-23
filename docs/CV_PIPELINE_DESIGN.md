# CV workstream — revised design after research

**Written 2026-07-23 (overnight session). Supersedes the architecture proposed in
`docs/CV_PIPELINE_HANDOFF.md`.**

Read in this order:
1. **This file** — what the pipeline should actually be, and why it is much
   smaller than the brief assumed.
2. `docs/CV_PIPELINE_PROGRESS.md` — the concurrent session's findings, including
   three verified defects in the current reference clips. Still accurate except
   where §4 of that file recommends a SLAM stage; see §3 below.
3. `docs/REFERENCES.md` — source status.
4. `docs/CV_PIPELINE_HANDOFF.md` — the original brief. Historical now; two of its
   stated facts are wrong (§0) and its stage list is superseded (§5).

All work this session was CPU-only. The GPU stayed occupied by locomotion
training (13.6 / 16.3 GB) throughout. Nothing under `training/` was modified.

---

## 0. TL;DR — five findings that change what to build

**1. The AMP discriminator never observes root position. The global-root stage
can be deleted.** [VERIFIED by reading the installed `amp_rsl_rl` v2.2.3 and the
project's own env config, independently by two agents.]

`MotionData.get_amp_dataset_obs()` returns exactly
`cat(joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)`.
No root position, no root height, no root orientation. `get_state_for_reset()`
(the RSI path) likewise returns no base position — spawn height comes from the
env config. Our env side matches: `joint_pos(15) + joint_vel(15) +
base_lin_vel(3) + base_ang_vel(3) = 36`.

`root_position` in the `.npy` is consumed at exactly one place: a finite
difference into linear velocity. **It is a velocity carrier and nothing else.**
Absolute position, accumulated drift, world registration and metric scale are all
invisible. The handoff calls the missing global root a "UNIVERSAL GAP" that makes
a root-recovery stage "mandatory regardless of extractor". That is wrong for AMP.
It would be right for DeepMimic-style tracking, which does use world-frame
end-effector and centre-of-mass terms.

**2. Hand-labelling ~150 real frames beats the entire synthetic-render pipeline.**
[VERIFIED from ScarceNet, CVPR 2023, Table 3.]

On TigDog, **36 hand-labelled real images per category scores 73.83 mAP —
matching the full synthetic-render + domain-adaptation pipeline at 73.66**, and
beating it by 7 points on tiger. The authors state it outright: better
performance is achievable by labelling a few images, which reduces human labour
compared to generating synthetic data.

Supporting numbers: synthetic-only training is catastrophically weak everywhere
(Mu et al. 25.3 PCK@0.05 on horses; 21.2 on tigers), and domain adaptation, not
asset quality, supplies the recovery (+54 points, to 79.5). Since we would have
to build the domain-adaptation stage ourselves, and labelling ~150 frames is one
evening, the synthetic path loses on effort *and* on outcome.

**3. The premise of the brief's "#1 open risk" is factually wrong.** [VERIFIED.]

The handoff worries that published work "used professional CAD" while ours is
hobby-quality. Mu et al.'s footnote points at the UE4 Marketplace *Animal Pack
Ultra 2* — **horse 3,299 vertices, tiger 3,124 vertices**, ~6 canned animations,
no LODs. A ~$50 indie game pack. Our artist mesh is 977 verts / 1,238 polys with
a proper skinned 33-bone rig — the same class. The gating risk the brief wanted
answered was largely a non-issue.

**4. Deer is the easiest quadruped in the literature, not a hard case.**
[VERIFIED from ScarceNet Table 2 and AP-10K Table 9.]

Deer is *in* AP-10K (794 labelled images) and therefore transitively inside
SuperAnimal's Quadruped-80K. Trained on Bovidae only, with **zero deer labels**,
ScarceNet reaches **0.767 mAP on deer — the best-transferring species in the
table, and higher than training on all deer labels (0.751)**. Cervidae retains
82% of within-family performance. Compare chimpanzee at 0.057. We are not in the
unseen-species regime; we are in the best-covered corner of it.

(Caveat: moose scores 0.547, much worse than deer. Bulkier silhouette and heavier
antlers. A heavily-antlered buck may be closer to the moose case. Antler effects
on head keypoints are **entirely unstudied** — see §7.)

**5. The binding accuracy constraint is joint-angle jitter, and it is severe.**
[VERIFIED by code reading plus arithmetic on our own clip.]

`amp-rsl-rl` linearly resamples the clip and then takes a **raw finite
difference** — no smoothing anywhere — to produce the joint-velocity half of the
discriminator observation. Jitter is amplified by `√2 · fps ≈ 42×` at 30 fps:

| jitter σ on joint angles | spurious joint-vel std | % of true RMS joint speed (1.63 rad/s) |
|---|---|---|
| 0.5° | 0.37 rad/s | 23 % |
| 1° | 0.74 rad/s | **45 %** |
| 2° | 1.48 rad/s | 91 % |
| 3° | 2.22 rad/s | 136 % |
| 5° | 3.70 rad/s | 227 % |

A physically actuated deer cannot produce white-noise joint velocity, so a noisy
velocity channel becomes a **perfectly separating feature**: the discriminator
saturates and style reward collapses to ~0. That is precisely the failure
fingerprint already in `CLAUDE.md`. **Temporal filtering is not a polish step; it
is load-bearing.**

---

## 1. The acceptance bar for the CV pipeline

This is the concrete answer to "how accurate does extraction have to be", which
the brief asked for and which sets scope for every upstream stage.

| Quantity | Target | Rationale |
|---|---|---|
| Joint-angle jitter σ, **after filtering** | **≤ 1°**, ideally ≤ 0.5° | keeps spurious joint velocity below ~half the true signal |
| Bone-length variation over time | **exactly 0** | achieved by construction if we fit rig joint angles rather than lifting free 3D points |
| Stance-phase foot slide | **< ~5 % of stride length** | contact/velocity consistency; see §3 |
| Root translational speed | **±10–20 %**, and *derived from foot odometry* | only 3 of 36 obs dims, but must agree with leg cadence |
| Global root XY / heading / world scale | **not needed at all** | never observed (§0.1) |
| Absolute root height | **not needed**; derive by FK grounding | not in the obs; RSI uses a constant `base_height_m=0.93` |
| Missing DOFs | tolerable — but **exclude, never zero-fill** | §4 |
| 2D reprojection error | **< 20 px mean, < 100 px max** | Uni-Mo's production gate; 70.2 % of clips passed it, and 97.6 % of survivors trained successfully |

On missing DOFs, the reassuring datapoint: **WASABI (CoRL 2022) trained a Solo-8
backflip with a discriminator that saw only 10 dimensions — base linear/angular
velocity, gravity vector and base height, and *no joint angles at all*.** Our
15-of-25 joint coverage is luxurious by that standard.

---

## 2. Revised architecture

The brief's seven stages collapse to four. Stages [1]–[3] (render 10k images,
train a 2D net on them, domain-adapt on real video) are replaced by "label 150
real frames". Stage [5] (root + contact) shrinks from a SLAM pipeline to foot
odometry.

```
[A] Collect real deer footage; hand-label ~150 frames at OUR keypoint set,
    bootstrapped by SuperAnimal-Quadruped zero-shot predictions
      |
[B] Fine-tune a COCO-pretrained 2D pose net on those labels
    (+ optional semi-supervised refinement on the unlabelled remainder)
      |
[C] Fit OUR URDF's joint angles directly to the 2D keypoints, per frame,
    temporally regularised.  NEVER lift free 3D points and then retarget.
      |
[D] Post-process into a clip:
      - Butterworth zero-phase filter; validate on the DIFFERENTIATED signal
      - detect stance; foot-lock; derive root speed from foot odometry
      - FK-ground the root height so the lowest hoof touches z = 0
      - write .npy (xyzw!), run the validation gates, train
```

**Why [C] is stated so emphatically.** Fitting the rig's own joint angles makes
bone lengths hard constraints. That single choice eliminates, at once: bone-length
drift, absolute-scale error, and most monocular depth ambiguity (limb
foreshortening against known segment lengths pins depth far better than any
generic monocular depth model). Lifting free 3D points and retargeting afterwards
reintroduces all three. This is also why the existing
`validate_motion.py::bone_length_consistency` gate is the right instinct.

### What happens to the synthetic renderer

It is no longer the path to a 2D pose network. It keeps three real uses:

1. **The fitting model for stage [C].** We need to render/FK the rig to compute
   reprojection error regardless. This is required work, not optional.
2. **Augmentation on top of the labelled set.** PASyn: 99 real + 3K crude
   synthetic scored 92.4 vs 91.4 for 8,000 real images. Crude synthetic is
   useful *with* real labels, useless *instead of* them.
3. **Conditioning signal for a generative appearance upgrade**, if synthetic
   scale is ever wanted. Render depth + Canny edges from the rig, feed as
   ControlNet/FLUX conditioning. SPAC-Net took synthetic-only zebra from 46.6 to
   **80.4** this way; Generative Zoo trained exclusively on such data and beat
   real-data baselines. The mesh only has to be geometrically right — appearance
   comes from the diffusion model. **This is the field's actual answer to "my
   renderer is crude", and it is a far better use of effort than mesh work.**

**Do not spend any time on fur, subsurface scattering, or material authoring.**
It is the lowest-ranked lever in every study that measured it (Eversberg &
Lambrecht: flat grey 0.642 vs best realistic material 0.653 AP — a 1.1-point
swing across the entire realism range, while background choice swings 5.3).

---

## 3. Root recovery — what to do instead of SLAM

`CV_PIPELINE_PROGRESS.md` §4 recommends DROID-SLAM + metric depth (TRAM/WHAM).
Given §0.1, **that recommendation is superseded**: those methods recover absolute
world-frame position, which nothing in our stack consumes. They also carry SMPL
and non-commercial licence exposure, and their failure modes (telephoto, low
parallax from a tripod pan, moving foliage, subject filling frame) are exactly
wildlife footage.

**Use foot odometry instead.** A planted hoof is stationary in the world;
given per-frame joint angles and known bone lengths, the root's displacement is
the negation of the stance hoof's body-frame displacement. The animal's legs are
the odometer. This needs no SLAM, no static background, no metric depth, no known
intrinsics — and it is **completely indifferent to camera motion**, which is
precisely the case Uni-Mo declares unsolved.

Foot odometry's classic weakness is cumulative drift. Our pipeline
differentiates the trajectory immediately, so drift is the one error mode we do
not care about. That coincidence is the whole argument.

Evidence it works: TransPose (SIGGRAPH 2021) reports ~12 cm cumulative error at
1 s using contact-based integration; Rempe et al. (ECCV 2020) get 94.6 %
synthetic / 92.7 % real contact-detection accuracy from a small MLP whose
training labels are generated by a pure geometric heuristic (movement < 2 cm and
within 5 cm of the ground plane) — so it is subject-agnostic and we could
regenerate its training set from our own sim rollouts. Legged robots do the same
thing in production (contact-aided invariant EKF), drifting 1–10 % of distance.

**Frame-rate caveat.** Quadruped gallop stance can be < 40 ms — about 1.2 frames
at 30 fps. Contact integration is fine for walk and trot, marginal for canter,
unusable for gallop at consumer frame rates. Walk is our target, so this is fine,
but film at 60 fps or better if there is a choice.

### The degeneracy worth fixing right now

Our current clip synthesises a **constant** forward speed. The discriminator sees
`base_lin_vel`, so a constant gives it a **dead-flat velocity channel**, while any
physically simulated deer has one that oscillates within each stride. That is a
trivially learnable separating feature and a plausible contributor to the
documented style-reward collapse.

Cheap fix, hours not weeks: add a sinusoidal surge at the measured stride
frequency, amplitude taken from our own sim rollouts. It is not the true ripple,
but it removes the degeneracy and is a necessary control condition before
anything more ambitious is judged.

---

## 4. Hazards to preserve (do not regress these)

- **`root_quaternion` on disk is XYZW, not WXYZ.** The handoff §1 says wxyz; it
  is wrong. Verified in `motion_loader.py` (`Rotation.from_quat`, with the
  comment "expected in the dataset as xyzw format (SciPy default)") and against
  the shipped clip, whose constant `[-0.7071, 0, 0, 0.7071]` is Rx(−90°) only
  under the xyzw reading. Under wxyz it would be a 270° yaw, which is not a
  coordinate fix. **The wrong reading raises no error.** This project already
  lost ~14 runs to one root-axis convention bug.
- **Silent zero-fill on joint-name mismatch.** `AMPLoader.load_data` aligns joints
  **by name** and fills any expected-but-absent joint with **constant zeros, with
  no warning**. The discriminator then learns "real deer keep their spine yaw at
  exactly 0" and penalises any motion of it. Currently dormant only by accident
  (`rsl_rl` 5.0.1 makes `amp_joint_names` resolve to `None`, so the loader falls
  back to the dataset's own list, and the old and new orderings happen to be
  positionally identical). It fires the moment anyone sets `amp_joint_names`
  explicitly — which the library's own warning tells you to do. Six joints would
  go to zero: the spine, all four hips, and the neck.
  `tools/cv/motion_npy.py --expect-joints` checks for it.
- **`joint_velocities` in the file is ignored** — the loader recomputes them from
  positions. Do not spend effort tuning them; do spend effort on the smoothness
  of `joint_positions`, which is what actually determines them.
- **Keep RSI.** In SFV, initial-state strategy was the single largest effect
  measured (backflip normalised return 0.086 fixed-start vs 0.741 adaptive) —
  larger than motion cleanup.

---

## 5. Immediate wins that need no CV at all

`CV_PIPELINE_PROGRESS.md` §1 found three verified defects in the clips AMP is
training on today. They are upstream of every naturalness complaint and are
hours of work. Listing them here because §0.5 raises their importance: the
discriminator is unusually sensitive to exactly these.

1. **Front fetlock reference is unreachable on 100 % of frames.**
   `FR_shin_to_FR_foot` spans −1.20…−0.40 rad in the clip; the URDF allows
   −0.30…+1.60. The clip is entirely negative, the limit is mostly positive.
   `convert_unity_anim_to_npy.py` carries a hard-coded limits table that has gone
   stale against `deer_world.urdf`. **Fix: derive limits from the URDF at
   runtime**, and re-check the fetlock sign convention.
2. **Walk-clip root speed contradicts the legs by 1.7×.** Stored 0.77 m/s;
   foot-odometry implies 1.32 m/s; feet slide **3.93× farther than the body
   travels**. `deer_run.npy` is fine on this axis (ratio 1.06), which corroborates
   the diagnosis — the walk clip's `FORWARD_SPEED = 0.8` was simply guessed wrong.
   Given §0.1, `base_lin_vel` is 3 of the 36 discriminator dims *and* the only
   channel root data reaches, so this is a direct, systematic error in the
   reference.
3. **The clip stands 10–17 cm underground.** `spawn_height = 0.93`; the correct
   standing root height for this rig is ≈ 1.03 m.

Two more, from this session:

4. **The clip is 1.03 s / 31 frames — below every measured data point in the
   literature** (AMP's own datasets: 13.6 s cartwheel, 229.6 s walk, 434 s
   locomotion; Escontrela's real-robot A1 result used 4.5 s). Whether one gait
   cycle suffices because gait is cyclic, or whether the discriminator just
   memorises ~51 transitions, is unknown. Cheap mitigation: concatenate several
   cycles and add the mirrored clip.
5. **The flat velocity channel** (§3).

**These are the locomotion agent's territory, not this workstream's — I did not
touch `convert_unity_anim_to_npy.py`. Owner decides.**

---

## 6. The experiment that should run first

Both the noise memo and the progress doc independently converged on this, and it
is the highest information-per-GPU-hour experiment available:

> Take `deer_walk.npy`, inject controlled corruption — Gaussian joint noise at
> σ ∈ {0, 0.5, 1, 2, 3, 5}°, a systematic 10 % limb-scale error, and a root speed
> scaled to 0.5× and 2× the foot-odometry value — and train short AMP runs on
> each. Plot style reward and `amp_status.py` metrics against corruption level.

This measures *our* tolerance on *our* rig, and it directly validates or refutes
the §0.5 jitter table, which is currently arithmetic plus a mechanistic argument
rather than a measurement. **The AMP noise dose-response curve does not exist in
the literature** — nobody has published it. So this is both the thing that sets
our scope and a genuinely novel result.

`tools/cv/` has everything needed to synthesise the corrupted clips. It needs the
GPU.

Second-priority experiment, also cheap and CPU-only to prepare: label 200 real
deer frames once, then train on 10/25/50/100/200 subsets and evaluate on a
held-out 50. That produces our actual labels-vs-accuracy curve instead of one
extrapolated from AP-10K.

---

## 7. Honest unknowns

- **The AMP noise dose-response curve is unmeasured** in the literature. §0.5 is
  a well-grounded prediction, not a result.
- **Whether foot-slide/contact error poisons *AMP* specifically is unmeasured.**
  All the supporting evidence (CoRe, GMR, STMR) is from tracking-based imitation.
  The extension to AMP rests on the structural argument that `joint_vel` and
  `base_lin_vel` sit in the same observation vector.
- **Nobody has ablated mesh/texture fidelity for animal 2D pose.** The
  "silhouette dominates because texture gets randomised away" argument is
  inference from method, not measurement. The closest real evidence is SPAC-Net's
  zebra-vs-rhino contrast, which is why §0.4's "deer is the rhino case" matters.
- **BEDLAM's clothing-realism claim has no numbers anywhere in the paper** —
  main text or supplementary. It is widely cited as proving realism matters; it
  is a prose claim. What BEDLAM *does* quantify is backbone pretraining
  (from-scratch 61.8 → ImageNet 51.8 → COCO 47.4 PA-MPJPE), which dwarfs it.
- **Antler effects on head keypoints are entirely unstudied.** Moose (0.547) vs
  deer (0.767) is a weak proxy. 20 minutes of SuperAnimal zero-shot on antlered
  vs antlerless frames would settle it. Partly mitigated now: SuperAnimal
  supervises antlers explicitly, and our schema emits antler tips, so they can be
  trained as signal rather than ignored as clutter.
- **Temporal-gap tolerance: no data at all**, for AMP or tracking.
- **The npj Robotics 2025 paper is the closest prior work and neither agent could
  read its full text** (paywall). Its extraction-accuracy numbers and
  discriminator spec remain unconfirmed. Worth institutional access before
  treating anything in it as load-bearing.
- ~~SuperAnimal-Quadruped's ordered 39-keypoint list was not obtained.~~
  **RESOLVED.** Read from the primary source —
  `DeepLabCut/deeplabcut/modelzoo/project_configs/superanimal_quadruped.yaml`
  (the model card does not publish it; the figures repo it defers to 404s).
  `deer_keypoints.SUPERANIMAL_TO_CANONICAL` maps 26 of the 39 slots.
  Notably that vocabulary **includes four antler keypoints**, so the model was
  trained with antlers as labelled structure rather than unlabelled clutter —
  which is better than AP-10K for deer and softens the antler unknown below.
  Antler *tips* are now supplied by `MESH_KEYPOINTS` (lateral extremes of the
  skull vertex group, read per frame from the posed mesh), taking the schema to
  37 keypoints.

---

## 8. Licensing — for a public portfolio repo

| Asset | Licence | Risk |
|---|---|---|
| **The deer mesh + `.anim` clips** (ithappy *Animals_FREE*, Unity Asset Store) | Asset Store EULA — use in projects permitted, redistribution not | **Flagged, unresolved.** Renders derived from it are arguably fine as product output; publishing a large image dataset of the mesh is closer to redistribution. Worth a decision before anything is published. |
| SuperAnimal-Quadruped weights / Quadruped-80K | **Modified MIT — academic, non-commercial only**; commercial use needs an EPFL agreement | **HIGH.** Fine for a hobby portfolio if stated; do not vendor weights. Avoid entirely by fine-tuning a COCO-pretrained HRNet. |
| `JitengMu/Learning-from-Synthetic-Animals` | **GPL-3.0** | **HIGH.** Copying their code makes our repo GPL-3.0. |
| TigDog | Non-commercial; creators do not own source video copyright | **HIGH.** Do not redistribute. |
| AP-10K | CC-BY-4.0 | LOW. |
| `chaneyddtt/UDA-Animal-Pose` | MIT | LOW. |
| SMPL / SMPL-X / SMAL (pulled in by WHAM, TRAM, GLAMR, DogMo) | MPI research licence, non-commercial | **HIGH — and avoided entirely** by the §3 decision not to use them. |
| DUSt3R / MASt3R / CUT3R | CC BY-NC-SA 4.0 | HIGH — also avoided. |
| `amp-rsl-rl` | BSD-3 | LOW. Current stack. |
| **Our URDF + our own hand labels** | ours | **NONE.** |

Worth noting: the §2 architecture — our own rig, our own labels, a COCO-pretrained
backbone, foot odometry instead of SLAM — yields a repo with **essentially no
third-party data encumbrance**. The synthetic + DA + SLAM path would have pulled
in GPL-3.0, non-commercial weights and SMPL. That is a real secondary argument
for the revised design.

---

## 9. Code state and deconfliction

Two agents built in `tools/cv/` simultaneously this session. Nothing was deleted;
both halves are complementary but there is genuine duplication to resolve.

| File | Author | Role |
|---|---|---|
| `deer_rig.py` | other | URDF parse + FK, **verified exact-parity with `render_playback_meshed.py`** in its self-test. The URDF is what AMP consumes, so this is needed regardless. |
| `keypoint_schema.py` | other | 31 keypoints anchored to **URDF links**, mapped to **AP-10K**. This is the schema stage [C] fitting needs. |
| `render.py` | other | Pure-numpy pinhole camera + z-buffer rasteriser; gives part-segmentation and depth for free. Useful for reprojection in stage [C]. |
| `synth_dataset.py` | other | Mu et al. recipe over the **URDF** meshes. Demoted by §2 — and it renders convex-hull proxies (§below). |
| `motion_npy.py` | other | `.npy` IO with verified conventions + the zero-fill check. Keep. |
| `validate_motion.py` | other | The acceptance gates: bone length, foot slide, ground penetration, root-vs-foot-odometry consistency, joint limits, reprojection. Keep — this is what found the §5 defects. |
| `deer_keypoints.py` | this session | 35 keypoints anchored to the **artist armature bones**. Needed for Blender-side labelling. |
| `blender_synth.py` | this session | Blender headless generator over the **artist mesh**. Verified working; see below. |
| `overlay_keypoints.py` | this session | Draws labels on renders for visual QA. |

**The two keypoint schemas are in different spaces and are NOT yet reconciled.**
One anchors to URDF link origins, the other to artist armature bone heads/tails.
A spot check says they do not trivially align — e.g. the front hip in armature
space sits at `(0.122, −0.051, −0.516)` relative to the spine root, while the URDF
places the corresponding point at `(0.122, 0.227, −0.649)` relative to the pelvis.
X agrees exactly; Y and Z do not. Establishing the correspondence (or proving the
rigs disagree) is **the next integration task**, and it matters: it is the link
between "labels produced in image space" and "joint angles the URDF fit solves
for".

**Mesh source, resolved.** The URDF's `<visual>` meshes are convex-hull physics
proxies — **1,693 triangles for the entire animal**, of which the skull is 1,080;
no toe, hoof or antler visual geometry at all; and the other session measured a
10.5 cm gap between pelvis and chest, 9.9 cm at each shoulder. Renders of the
URDF show a disassembled deer. The artist mesh in
`assets/blender/deer_standing.blend` (`Deer_001`, 977 verts / 1,238 polys, skinned
to the 33-bone `Deer_001_rig`) is continuous and correct. **Any render intended to
look like a deer must use the artist mesh; anything reasoning in URDF space must
use the URDF.** Worth knowing when judging policy videos by eye:
`tools/render_playback_meshed.py` renders the URDF, so the deer it shows is
always slightly disassembled.

### `blender_synth.py` — verified working

```bash
blender -b assets/blender/deer_standing.blend --python tools/cv/blender_synth.py -- \
    --out <dir> --num 200 --backgrounds <coco_dir>
python tools/cv/overlay_keypoints.py --dataset <dir> --num 8 --contact-sheet
```

Cycles CPU-only (never EEVEE — it would allocate GPU memory), ~0.5 s/image, so
10k images ≈ 1.5 CPU-hours. Poses are sampled from the **three real Unity clips
already present in the .blend as Blender Actions** (idle / walk / run) plus
bounded jitter, not from the rig's rest pose. Emits per sample: PNG, 2D keypoints,
3-level visibility (ray-cast against the mesh, with the occlusion threshold taken
from each joint's own depth below the skin rather than a single body-radius
constant), 3D keypoints, full camera intrinsics/extrinsics, and the bone
quaternions that produced the image.

Camera distance is **derived from focal length** so the animal fills a controlled
fraction of frame; sampling them independently let long lenses silently crop the
subject to a torso.

**Verified visually, not just numerically** — the overlay contact sheet shows
keypoints landing on the animal across viewpoints and gait poses, which is the
check that catches frame-convention errors. Two defects were found and fixed that
way: camera framing off-centre (it had been computed from mesh bounds, which live
in a different object transform than the bones and the projection), and an
over-conservative visibility test.

Known gaps: no background images were available this session, so the tested
renders used flat random-colour worlds — which **overstates** achievable accuracy,
since a network can cheat on "silhouette against flat colour". The generator warns
about this on every run. The original texture is also missing (the source Unity
project folder is gone), so only randomised procedural textures are possible.

---

## 10. What I would do next, in order

1. **Fix the three reference-clip defects** (§5.1–5.3) plus clip length and the
   flat velocity channel. Hours of work, no CV, upstream of everything, and it
   plausibly explains a chunk of the "everything rigid" review.
2. **Run the noise-tolerance experiment** (§6) when the GPU frees. It sets the
   accuracy bar for every stage above and could cut scope further.
3. **Label 200 real deer frames** (§2[A]) and produce the labels-vs-accuracy
   curve. CPU-only, and it replaces the biggest remaining extrapolation.
4. **Reconcile the two keypoint schemas** (§9) — the blocking integration task.
5. Only then build stage [C], the URDF fit.

Note the ordering: three of the top four items are not the video pipeline. The
research's net effect was to shrink this workstream and push value upstream into
the clips we already have.

---

## 11. Addendum — second CPU session, same night

Later work that changes or extends the above. Details in
`docs/REFERENCE_CLIP_FINDINGS.md` and `docs/RIG_DEFECT_hind_foot_collision.md`.

**The two rigs are geometrically identical, and the correspondence is exact.**
§9 asked for this and treated it as open. Anchoring the fit on true joint centres
rather than the axial set gives **scale 1.0000, rotation 0.00 deg, residual
0.00 cm on all 12 limb joints** — a pure translation of `(0, -1.0206, 0.2905)`,
which is the artist `spine.003` head. The URDF **skips the armature's `shoulder.*`
bone**, so `URDF hip <-> armature thigh.head`, one segment further down than the
bone names suggest. The residual axial disagreements in §9 were entirely an
artifact of anchoring on URDF *surface* keypoints, which are extreme vertices of
convex-hull proxy meshes (`withers` was 19 cm out). Run
`python tools/cv/rig_correspondence.py --anchor limbs`.

**Two verified rig defects, both upstream of every training run, neither fixed:**

1. **Hind foot/toe collision meshes are mirrored** (`RIG_DEFECT_hind_foot_collision.md`).
   The hind legs have no collision geometry at the hoof; front reaches +0.977 m,
   hind +0.647 m. The correct mesh is already on disk as the pre-v3 version. The
   training USD inherited it.
2. **Front fetlock joint limits are wrong.** The clip asks for [-1.20, -0.40] rad;
   the URDF allows [-0.30, +1.60], so the joint is pinned at its stop and can never
   reach the reference. A concrete mechanism for the long-reported "front legs
   rigid / static struts".

**Two claims from `CV_PIPELINE_PROGRESS.md` §1 are corrected.** The walk clip's
`FORWARD_SPEED = 0.8` is **correct** — the "1.7x too slow, feet slide 3.9x"
finding is an artifact of stance-based estimators, mine included. Measuring duty
factors directly from the artist source shows **0.16-0.39 where a walk requires
>0.5**: the source animation has no physically valid stance phase, so every
stance-based metric on it is unreliable. `deer_run.npy`'s 2.5 m/s does look ~46%
too fast, inverting the earlier read that run was the healthy clip.

**Consequence for §3.** Contact-based root recovery is still the right design, but
it **cannot be validated on these clips** — it needs motion that actually has
stance phases. Validate on video-derived motion or a sim rollout. This is also the
strongest concrete argument for the video workstream: real footage has a real
stance phase by construction, and this hand-authored clip never will.

**New tooling:** `check_rig_geometry.py` (regression guard, exits non-zero),
`rig_correspondence.py`, `blender_dump_rig.py`, `rebuild_reference_clips.py`
(writes corrected copies to `assets/motions/v2/`, never in place),
`make_noise_suite.py` (14 corrupted variants + manifest, ready for the GPU).
