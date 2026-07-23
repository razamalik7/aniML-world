# CV workstream — video → motion. Progress & design log

Running doc for the "ordinary video of an animal → reference `.npy` for our own
rig" workstream. Started 2026-07-23. Read together with
`docs/CV_PIPELINE_HANDOFF.md` (the brief) and `docs/REFERENCES.md` (source
status). **Where this doc and the handoff disagree, this doc wins** — it records
things checked against the installed code and the actual rig, and two of the
handoff's stated facts turned out to be wrong.

Everything here was done CPU-only; the GPU stayed occupied by locomotion
training (13.6 GB / 16 GB, 75 % util) for the whole session. Nothing under
`training/` was touched.

---

## 0. TL;DR for whoever picks this up

1. **Three defects in the existing reference clips are almost certainly costing
   more naturalness than any video pipeline would buy.** They are cheap to fix
   and are upstream of everything. See §1. Fix these first.
2. **The rig's visual meshes are not a connected body** — a 10.5 cm hole between
   pelvis and chest, 9.9 cm at each shoulder, no hooves. Any 2D pose net trained
   on renders of it learns a silhouette that does not exist in nature. This is
   the real form of the handoff's "#1 open risk", and it is a *geometry* problem,
   not a texture problem. See §2.
3. **Rendering alone will not work.** Hard number: Mu et al. got **25.3 PCK@0.05**
   on real horses from synthetic-only training and **70.8** after self-training
   on unlabelled real video. The domain-adaptation stage is where the accuracy
   is. See §3.
4. **The global-root problem is solved in the human literature and the handoff
   does not mention it.** TRAM / WHAM recover world-frame root trajectory from a
   *moving* camera via SLAM on the background + metric-depth scaling. Both MIT,
   both live. This is the missing piece Uni-Mo declared unsolved. See §4.
5. Built and self-tested: `tools/cv/` — rig+FK, 31-keypoint schema, CPU
   rasterizer, synthetic dataset generator, `.npy` I/O with *verified*
   conventions, and a validation suite. See §6.

---

## 1. VERIFIED DEFECTS in the current reference clips

All three found with `tools/cv/motion_npy.py` and `tools/cv/validate_motion.py`,
reproducible with:

```
python tools/cv/validate_motion.py --clip assets/motions/deer_walk.npy \
    --map-legacy --render docs/cv_walk_validation.png
```

### 1a. Front fetlock reference is physically unreachable — 100 % of frames

| joint | clip range (rad) | URDF limit | frames outside |
|---|---|---|---|
| `FR_shin_to_FR_foot` | −1.20 … −0.40 | −0.30 … +1.60 | **31/31** |
| `FL_shin_to_FL_foot` | −1.20 … −0.20 | −0.30 … +1.60 | 19/31 |

`tools/convert_unity_anim_to_npy.py` carries its own hard-coded `URDF_LIMITS`
table which says `±1.20` for these joints. **The table is stale relative to
`deer_world.urdf`, which says `[-0.30, +1.60]`** — the range is mostly *positive*
and the clip is entirely *negative*. The AMP discriminator has been shown a
front-fetlock reference the rig can never match, on every frame, for the entire
project. That is a permanent, un-closeable style-reward gap on the front legs,
and it fits the human review ("everything rigid", front-leg weirdness).

*Recommended fix:* re-derive the limits table from the URDF at runtime instead of
hard-coding it, and re-check the sign convention for the fetlock DOF.

### 1b. The synthesised root speed contradicts the leg motion by 1.7×

The clip is in-place; `convert_unity_anim_to_npy.py` synthesises
`root_position` as a straight line at `FORWARD_SPEED = 0.8 m/s`. Measured:

| quantity | value |
|---|---|
| stored root speed | 0.77 m/s |
| speed implied by stance-foot odometry | **1.32 m/s** |
| mean foot slide during stance | **1.71 m/s** |
| total foot slide ÷ distance travelled | **3.93×** |

The feet slide almost four times farther than the body travels. This is a
*skating* reference — exactly the failure mode `docs/REFERENCES.md` warns about
for naive retargeting, except it is baked into our hand-authored clip. Setting
`FORWARD_SPEED` to the foot-odometry value (~1.3 m/s) would make the reference
self-consistent for free.

**`deer_run.npy` is fine on this axis and that corroborates the diagnosis**:
stored 2.40 m/s vs foot-odometry 2.55 m/s (ratio 1.06), slide/travel 0.62. Its
`FORWARD_SPEED = 2.5` was guessed nearly right; the walk clip's 0.8 was not. So
this is a per-clip constant that happens to be wrong for walk — not a systemic
converter bug — and the walk clip is the one AMP trains on most.
(`deer_run.npy` has its own problem: **all four** fetlocks out of limits, 16–22
frames of 25, and 26.8 cm ground penetration.)

### 1c. The clip stands ~10–17 cm underground

`spawn_height = 0.93` in the converter. At that root height the rig's hooves sit
at **z = −0.08 … −0.10 m at rest** and reach **−0.17 m** during the clip. Nothing
floats (`max_float = 0.0`), so it is a uniform offset: the correct standing root
height for this rig is **≈ 1.03 m**, not 0.93.

### 1d. Latent landmine — silent zero-fill on joint-name mismatch (NOT currently firing)

`assets/motions/*.npy` still use the pre-v2 joint names (`pelvis_to_chest`,
`pelvis_to_HR_thigh`, …). The trainer asks for the v2 names
(`chest_yaw_to_chest`, `HR_hipabd_to_HR_thigh`, …). Only **9 of 15 match**.

`AMPLoader.load_data` aligns **by name** and fills any unmatched joint with
**zeros, silently**:

```python
for j in expected_joint_names:
    idx_map.append(dataset_joint_names.index(j) if j in dataset_joint_names else None)
...
if src_idx is not None: arr[i] = frame[src_idx]     # else stays 0.0
```

Right now this does **not** fire, because `rsl_rl` 5.0.1 sets `RSL_RL_V4_PLUS =
True`, so `amp_joint_names` resolves to `None`, so `AMPLoader` falls back to
building the joint list from the *dataset's own* `joints_list` — and the old and
new orderings happen to be positionally identical. It works by accident.

**It breaks the moment anyone sets `amp_joint_names` explicitly in `dataset_cfg`**
— which is exactly what the library's own warning tells you to do. Six joints
would go to constant zero: the spine, **all four hips**, and the neck. Renaming
the clips' `joints_list` to the current URDF names removes the landmine
permanently. `tools/cv/motion_npy.py --expect-joints ...` checks for this.

### 1e. Handoff spec correction — quaternion order

`docs/CV_PIPELINE_HANDOFF.md` §1 says `root_quaternion (T,4) wxyz`. **It is
xyzw.** `motion_loader.py::_resample_data_SO3` calls
`Rotation.from_quat(raw_quaternions)` with the comment *"the quaternion is
expected in the dataset as xyzw format (SciPy default)"*. `wxyz` is what
`MotionData.base_quat` exposes *after* internal conversion. Our existing clips
are correctly xyzw. Anything new must be written xyzw.

---

## 2. The rig is not renderable as a deer yet (the real "#1 risk")

The handoff frames the top risk as *"sim-to-real gap for a hobby-quality mesh …
silhouette and proportions are accurate but surface appearance is crude."*
Checked directly. **The surface appearance is fine; the silhouette is the
problem.**

Rest-pose minimum surface distance between adjacent links' visual meshes:

| link pair | gap |
|---|---|
| `pelvis` → `chest` | **10.5 cm** |
| `chest` → `FR_thigh` / `FL_thigh` | **9.9 cm** |
| `FR_shin` → `FR_foot` / `FL_shin` → `FL_foot` | **10.7 cm** |
| `chest` → `neck` | 4.9 cm |
| all hind-leg joints | 0.0 cm (fine) |

The deer is 1.49 m long, so the torso hole is ~7 % of body length. See
`docs/cv_render_check.png` and `docs/cv_walk_validation.png` — the body reads as
two disconnected boxes with daylight between them, and the front legs are
floating segments.

Additionally, **10 links have no visual mesh at all**: all four `*_toe` links,
`antlers`, the 4 `*_hip_abd` and 2 `*_yaw` helper links. So renders have **no
hooves**, and the toe DOF is invisible in image space — a 2D pose net can never
learn to see it. (Antlers are fine: they are baked into `skull_visual.stl`.)

Why this matters more than texture: the standard recipe (Mu et al.) **randomises
texture away on purpose** — half the training images get random colours. What
survives randomisation, and therefore what the network actually keys on, is
**shape and silhouette**. A hole in the torso and missing hooves are precisely
the part that is not randomised.

### IMPORTANT QUALIFICATION (added after discovering concurrent work — see §9)

These gaps are a property of the **per-link visual STL decomposition that the
URDF uses**, not necessarily of the original artwork.
`assets/blender/deer_standing.blend` (8 MB, Apr 2026) is the artist source and is
a skinned mesh — almost certainly continuous. **So the right source for synthetic
training renders is the artist blend, not the URDF's per-link STLs.** A
concurrent agent is building exactly that (`tools/cv/blender_synth.py`).

I have **not** opened the blend to confirm continuity — treat that as unverified.

What remains true regardless:
- Anything that renders **the URDF** shows a disassembled deer. That includes
  `tools/render_playback_meshed.py`, which is how policy quality gets judged by
  eye. Worth knowing when interpreting those videos.
- The URDF still has **no hoof or toe geometry at all**, so URDF-space contact
  reasoning and any URDF-based reprojection is missing the distal foot.
- **Cheap fixes:** promote the existing `*_toe_collision.stl` and
  `antler_collision_v2.stl` to `<visual>` elements (they already exist), and
  close the torso/shoulder gaps.

`DeerRig.missing_visuals()` reports this, and `synth_dataset.py` prints it as a
warning on every run so it cannot be forgotten.

---

## 3. Q1 — sim-to-real gap for low-fidelity meshes

The handoff says *"nobody has measured this."* Partly true; here is what exists.

**Mu et al., CVPR 2020 (`Learning-from-Synthetic-Animals`)** — PCK@0.05 on real
TigDog horses/tigers, extracted from the paper's Table 1:

| training regime | horse | tiger |
|---|---|---|
| synthetic only | **25.33** | **21.17** |
| + CycleGAN | 51.86 | 46.47 |
| + BDL | 62.33 | 52.26 |
| + CyCADA | 55.57 | 51.48 |
| **+ CC-SSL (self-training)** | **70.77** | **64.14** |
| real data only | 78.98 | 81.99 |
| CC-SSL + real labels | 82.43 | 84.00 |

They used **professional purchased UE4-marketplace CAD** and still got 25 PCK
synthetic-only. **The dominant term is the domain gap, not the asset.** Stage [3]
of the handoff's architecture (domain adaptation) is confirmed **mandatory**, and
it is worth ~+45 PCK — far more than any plausible mesh upgrade.

**Counter-example, and why it does not transfer.** *ZebraPose* (arXiv 2408.10831)
trained detection + pose on **18k renders of a Sketchfab hobby model** — no
domain adaptation, no real labels — and did generalise to real zebras
(PCK@0.05 = 0.466, 0.608 after filtering misalignments; detection mAP50 0.512).
That is a genuine low-fidelity-asset success. But the setting is much easier:
zebras have a *uniquely* distinctive stripe texture, and the imagery is aerial
drone footage over open savanna (near-trivial backgrounds). Deer in cluttered
forest/field footage is the Mu et al. regime, not the ZebraPose regime. **Do not
plan around ZebraPose's result.**

One transferable ZebraPose finding, though: their biggest single win came from
**viewpoint/scale coverage, not realism** — adding cropped+rescaled variants
moved detection mAP50 from 0.150 → 0.512 (+36 pts). `sample_camera()` in
`synth_dataset.py` implements wide distance/elevation/FOV sampling for this
reason.

**Net conclusion on Q1:** mesh *texture* fidelity is second-order (it gets
randomised away). Mesh *geometry* fidelity is first-order, and ours is currently
broken (§2). Domain adaptation is non-negotiable. The handoff's proposed
one-day decisive experiment is still the right experiment, but **it will
under-report** unless the mesh gaps are fixed first — otherwise it measures our
broken silhouette, not the method.

*Evidence quality:* Mu et al. numbers read directly from the paper PDF — solid.
ZebraPose numbers via the HTML version, single source, not independently
replicated — medium. No study anywhere ablates mesh geometry quality directly;
the "silhouette dominates because texture is randomised" argument is **my
inference from the method, not a measured result.** Flagging that explicitly.

---

## 4. Q2 — large-translation locomotion from a moving/tracking camera

The handoff calls this unsolved, citing Uni-Mo's limitation (*"extending to
large-translation locomotion would require tracking cameras or a multi-view
setup"*). **For the camera half, this is solved in the human-motion literature,
and neither the handoff nor `REFERENCES.md` mentions it.**

| method | approach | repo | licence | last push |
|---|---|---|---|---|
| **TRAM** (ECCV 2024) | DROID-SLAM on background with dual masking of the subject → camera trajectory; scale fixed by aligning SLAM depth to a **metric depth** prediction; subject regressed in camera frame; compose. Claims **−60 % global motion error** vs prior work. | `yufu-wang/tram` | **MIT** | 2025-06-08 |
| **WHAM** (CVPR 2024) | Lifts 2D keypoint sequences to 3D with an AMASS-trained motion prior, fuses camera angular velocity from SLAM, predicts **foot contact** to refine global trajectory. | `yohanshin/WHAM` | **MIT** | 2024-04-18 |
| DROID-SLAM | the SLAM backbone both rely on | `princeton-vl/DROID-SLAM` | BSD-3 | 2025-05-05 |

Repo state and licences checked via the GitHub API on 2026-07-23 — all three
live, permissive, non-empty. **Not verified:** whether they run on our hardware,
and their weights' licences (WHAM/TRAM depend on SMPL, which is
research-licensed — *that* is a real portfolio-repo risk and needs checking
before adoption).

**Why this matters:** the camera-trajectory half is species-agnostic. DROID-SLAM
on the *background* does not care that the foreground is a deer. So the
architecture that actually works for our case is:

```
background SLAM  ->  metric camera trajectory   (species-agnostic, off-the-shelf)
        +
deer pose in camera frame                       (our rig-specific fit, stage [4])
        +
foot-contact constraint                         (locks stance feet in world frame)
        =  world-frame root trajectory
```

The **foot-contact term is the cheap and robust part** and is worth doing even
without SLAM: if you know which hoof is planted, the root translation between
frames is determined (that is "foot odometry", and `root_consistency()` in
`validate_motion.py` already measures it — it is what caught defect 1b). For a
locomotion clip filmed side-on, foot odometry alone may be sufficient and avoids
the SLAM dependency entirely.

**Practical alternative worth pricing:** deliberately shoot/find **fixed-camera,
side-on** footage of deer crossing frame. That reduces the problem to Uni-Mo's
*solved* case plus a 1-D translation along the image axis, recoverable from
known limb length as a scale reference. Much lower risk than a SLAM pipeline.

---

## 5. Q3 — how much reference noise can physics/RL absorb?

**This one I could not answer well, and it remains the highest-value open
question.** What I have:

- **AMP's mechanism is inherently noise-tolerant in a specific way**: the
  discriminator scores the *distribution* of (state, next-state) transitions,
  with no per-frame temporal alignment. So *jitter* and *phase* errors are far
  less damaging than for DeepMimic-style per-frame tracking. But *systematic*
  errors (a joint with the wrong sign, a limb with the wrong range, a skating
  contact schedule) are learned faithfully as style — see defects 1a–1c, which
  are exactly systematic errors. **Rule of thumb: AMP forgives noise, not bias.**
- **ReActor** (arXiv 2605.06593, SIGGRAPH 2026, ~2 months old) is the most
  relevant new work: a **bilevel optimisation that adapts the reference motion to
  the robot's morphology while training the tracking policy**, needing only *"a
  sparse set of semantic rigid-body correspondences"*, demonstrated **retargeting
  onto a quadruped**. If it works, it collapses the handoff's hand-defined
  25-DOF correspondence problem *and* absorbs reference error. **No code found,
  no quantitative error-tolerance numbers in the abstract, unrefereed at time of
  writing. Promising, unverified — do not plan around it, but re-check.**
- I found **no study that quantifies an accuracy threshold** (e.g. "3D keypoint
  error below X cm suffices"). The npj Robotics paper is paywalled to me and I
  could not extract its extraction-accuracy numbers.

**Recommended way to actually answer it — and it is cheap and CPU-free to
prepare:** take the existing clean clip, inject controlled noise (Gaussian
per-joint at σ = 1°, 3°, 10°; add a systematic 10 % limb-scale error; corrupt the
contact schedule), and train short AMP runs on each. That measures *our* tolerance
on *our* rig directly, and it sets the accuracy bar for every upstream stage.
`tools/cv/` has everything needed to synthesise the corrupted clips. **This is
the single experiment I would run first once the GPU frees** — it could cut the
whole upstream scope.

---

## 6. What was built — `tools/cv/`

All CPU-only, no torch/cv2 required, each module self-tests via
`python <module>.py`.

| file | what it is |
|---|---|
| `deer_rig.py` | URDF parse → 25 DOF / 27 links, FK, mesh loading, `missing_visuals()`. **FK verified to exact 0.0 parity against `tools/render_playback_meshed.py`** on random poses — that assertion runs in its self-test, so a future rig change that breaks agreement fails loudly. |
| `keypoint_schema.py` | **31 keypoints** = 20 joint-centres + 11 mesh-derived surface landmarks (nose, eyes, antler tips, withers, tail root, 4 hooves). Superset of **AP-10K's 17**, with `AP10K_MAP` so off-the-shelf detectors and hand-labelled frames line up. Ships `SKELETON` (30 bones) and `FLIP_PAIRS` for symmetry augmentation. Self-test asserts anatomical ordering (nose fwd of withers fwd of tail; antlers above withers above hooves; left/right not swapped) and bone-length invariance under random poses + random root transforms. |
| `render.py` | OpenCV-convention pinhole `Camera` (`look_at`, `project`, `intrinsics`) + a pure-numpy z-buffer rasterizer returning **RGB + part-segmentation + depth**. ~0.1 s per 640×480 frame. |
| `synth_dataset.py` | The Mu et al. recipe end-to-end: poses sampled by interpolating real reference-clip frames + bounded jitter (keeps poses biomechanically plausible), wide viewpoint/scale/FOV randomisation, 50/50 random-vs-original texture, background pool (COCO val2017 or procedural), **occlusion-aware keypoint visibility from the depth buffer** (0=absent / 1=occluded / 2=visible), part-seg masks, full camera intrinsics+extrinsics and ground-truth joint angles per sample. **10k images ≈ 17 min single-core.** |
| `motion_npy.py` | `.npy` read/write with the **verified** conventions, plus `check_joint_alignment()` (the §1d zero-fill trap) and `check_clip()` (limits, NaN, unit-norm quats, constant-column detection, xyzw/wxyz smell test). |
| `validate_motion.py` | The acceptance gates: bone-length consistency, **foot slide**, ground penetration/float, **root-vs-foot-odometry consistency**, joint-limit violations, velocity spikes, and 2D reprojection error. `--render` always emits a filmstrip next to the numbers. |

Generated artefacts: `docs/cv_render_check.png`, `docs/cv_keypoint_check.png`,
`docs/cv_walk_validation.png`, `docs/cv_synth_samples.png`.

### A methodology note worth keeping

While writing §1 I rendered the walk clip and the deer looked strongly
nose-down; I nearly wrote that up as a fourth defect. Measuring it showed the
nose is **above** the tail root on all 31 frames — the tilt was an artefact of my
camera being placed 0.55 m above its look-at target. The visual lied, and the
numbers corrected it. `MEMORY.md` records the opposite failure (posture metrics
lying, corrected by rendering). **Neither direction is safe alone**; the fix in
`validate_motion.py` is a forced-level side camera, with a comment saying why.

---

## 7. Open questions that need the owner's decision

1. **Fix the reference clips first?** §1a–1c are three small edits to
   `convert_unity_anim_to_npy.py` (limits from URDF, `FORWARD_SPEED` ≈ 1.3,
   `spawn_height` ≈ 1.03). They are upstream of every naturalness complaint. But
   they change the AMP reference mid-training-campaign, and that file is adjacent
   to the locomotion agent's territory. **I did not touch it. Whose call?**
2. **Is the rig geometry in scope for this workstream?** §2 says the meshes must
   be fixed before synthetic rendering is worth anything. That is Blender/asset
   work, not CV work. Who does it, and do you want the cheap version (promote
   collision STLs to visuals, close gaps) or a proper re-mesh?
3. **Run the noise-tolerance experiment before building more pipeline?** §5. It
   is the highest-information-per-GPU-hour experiment available and it could
   dramatically cut scope. Needs the GPU.
4. **Fixed-camera footage vs SLAM?** §4. Committing to side-on fixed-camera deer
   footage removes the hardest component entirely. Do you have/can you get such
   footage, or must the pipeline handle arbitrary internet video?
5. **SMPL licence exposure.** If we adopt TRAM/WHAM, their model weights depend
   on SMPL (research-only). Fine for private training; a problem for a public
   portfolio repo — same shape of problem as the PFERD licence already flagged.
   Needs checking before adoption, not after.

## 9. ⚠ CONCURRENT WORK IN `tools/cv/` — needs deconfliction

Discovered at the end of the session: **another agent was building the same
workstream in the same directory at the same time.** Files not authored by me,
created 03:29–03:34 alongside mine:

| their file | overlaps my |
|---|---|
| `deer_keypoints.py` (267 lines, `CANONICAL_KEYPOINTS`, `SUPERANIMAL_QUADRUPED_FROM_CANONICAL`) | `keypoint_schema.py` |
| `blender_synth.py` (32 KB, renders `assets/blender/deer_standing.blend` via `bpy`) | `synth_dataset.py` |
| `overlay_keypoints.py` (label QA contact sheets) | my inline overlay checks |

I did not modify or delete any of them.

**Their Blender approach is better for dataset generation** — it renders the
artist mesh (continuous, textured, properly skinned) rather than the URDF's
per-link STLs, which sidesteps §2 entirely. Their keypoint schema maps to
**SuperAnimal-Quadruped**; mine maps to **AP-10K**. Both are useful; they should
be merged into one file, not maintained in parallel.

**What is worth keeping from my side regardless of which renderer wins:**
`deer_rig.py` (URDF FK with verified parity — the artist blend is not the thing
AMP consumes, the URDF is), `motion_npy.py`, `validate_motion.py`, and the
findings in §1 / §3 / §4 / §5.

**Owner action needed:** decide the merge, and in future give parallel agents
disjoint directories.

## 8. Do NOT re-derive

- `root_quaternion` is **xyzw** (§1e).
- `joint_velocities` in the file is **ignored** by `AMPLoader` — it resamples
  positions and finite-differences them itself.
- The rig is **25 actuated DOF**, 27 links, 16 visual meshes, 10 links without
  visuals.
- FK in `deer_rig.py` is exact-parity with the production renderer; don't write a
  third FK implementation.
- Mu et al. synthetic-only = 25.3 PCK, CC-SSL = 70.8 (§3). Domain adaptation is
  mandatory; this is settled.
