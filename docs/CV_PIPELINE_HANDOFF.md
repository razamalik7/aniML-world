# HANDOFF — Video → Motion pipeline (CV workstream)

**Your job:** research and prototype a reusable system that turns ordinary
videos of animals into **reference motion clips for our own custom rigs**.
You own this workstream. A separate agent is concurrently iterating on
locomotion RL training — **do not touch that** (see Constraints).

Owner context: solo hobby/portfolio project ("aniML World") — an ecosystem sim
with RL-trained animals (deer now, wolves later) destined for Unity. Half
passion project, half ML-resume project, so the ML sophistication matters.

---

## 1. The goal, concretely

Produce `.npy` clips in the format our AMP training already consumes:

```
joint_positions   (T, J)   per-joint angles, radians
joint_velocities  (T, J)   NOTE: ignored by amp-rsl-rl's loader
root_position     (T, 3)   WORLD frame
root_quaternion   (T, 4)   *** xyzw *** (SciPy order) - NOT wxyz
joints_list       (J,)     joint names matching our URDF
fps               float
```

**CORRECTION (2026-07-23):** an earlier version of this doc said `wxyz`. That
was WRONG. amp-rsl-rl's `motion_loader.py` documents the dataset quaternion as
**xyzw** and converts internally to wxyz. Independently confirmed by an
uprightness test on `deer_walk.npy`: read as xyzw the skull sits at +1.185 above
a pelvis at +0.930 (upright); read as wxyz the skull is at -0.096, i.e. the deer
is inverted. **Use xyzw.** Getting this wrong silently invalidates any FK /
foot-contact / skating analysis built on the clip.

Reference: `assets/motions/deer_walk.npy` (current clip — 31 frames, 15 joints,
hand-authored from a bought Unity animation). `tools/convert_unity_anim_to_npy.py`
shows how one is built; `tools/visualize_motion_npy.py` sanity-checks one.

**Why this matters:** AMP's discriminator makes the policy match the reference
distribution, so **the reference clip is a hard ceiling on naturalness.** Our
current clip is a stiff synthetic cycle covering only 15 of the rig's 25 joints.
Human review of the best trained policy: *"head does a weird nod and is held
really high; hind legs asymmetric; everything rigid — no torso/neck flexion, no
lateral hip motion, nothing fluid or muscle-like."* Much of that is inherited
from the reference. **Better reference motion is the only way to raise the ceiling.**

Target rig: 25-DOF deer URDF (`assets/deer_urdf/deer_world.urdf`) — spine
yaw+pitch, per-leg hip abduction + hip flex + knee + ankle + toe, neck yaw+pitch,
skull.

---

## 2. ALREADY VERIFIED — do not re-derive these

Three deep-research passes with adversarial verification produced these. Full
detail in `docs/REFERENCES.md` (read it first).

- **MoCapAnything V2** (arXiv 2604.28130, repo `phongdaot/MocapAnything`): code +
  464MB weights genuinely released (MIT), BUT a direct read of
  `inference/video2pose2rot.py` confirms the reference skeleton is a **fixed
  per-species dataset lookup** — *"users cannot supply arbitrary skeletons; only
  species names select pre-built reference rigs."* Output is camera-space
  **root-relative**. Full 73-species dataset unreleased. **Not usable for a custom URDF.**
- **SLoMo**: repo contains **zero source code**, abandoned 2024. Ruled out.
- **npj Robotics 2025** (`nature.com/articles/s44182-025-00048-x`): best conceptual
  match (DeepLabCut 2D → ST-GCN 3D lift → constrained-IK retarget → AMP). Its repo
  `arclab-hku/Imitation_from_video` IS live but contains **only AMP training + already-
  extracted motion** — no 2D pose, no 3D lift, no IK. Treat as a paper to reimplement.
- **UNIVERSAL GAP**: every off-the-shelf arbitrary-skeleton extractor (MoCapAnything V2,
  TopoCap — both verified) outputs camera-space root-relative motion with **no global
  root trajectory and no foot contact**. A root + contact stage is mandatory regardless.
- **Retargeting is solved by nothing**: no released tool emits joint rotations for an
  arbitrary URDF; correspondences + IK constraints stay hand-defined. Naive keypoint
  (UVM) transfer breaks contact schedules → foot sliding/penetration; **STMR**
  (IEEE T-RO 2025, arXiv 2404.11557) adds contact-preservation + foot-locking.
- **Licensing landmines**: PFERD horse mocap (Harvard Dataverse `doi:10.7910/DVN/2EXONE`)
  is real, high-quality, WITH world-frame root — but **non-commercial AND forbids
  redistributing modified forms** (verified via Dataverse API). SuperAnimal's
  Quadruped-80K (Zenodo 14016777) is **academic/non-commercial**. Matters for a public
  portfolio repo.
- **Biomechanics-literature shortcut: dead.** Only ~6 sagittal DOFs as summary
  scalars/plots; no spine/neck/toe/abduction, no world root, no per-frame data.
- **Reference-free approaches** (RLWAV, CPGs, energy-min): **no claim survived
  verification.** Unevidenced, not disproven.

---

## 3. Proposed architecture (stress-test it, don't assume it)

The central idea: **make the model's output space BE our rig**, so retargeting —
historically this project's biggest bug source (axis conventions) — never happens.

```
[0] AMP policy on the existing hand-authored clip   ← already running
      Literature says a clean-reference warm start is REQUIRED before
      video-derived refs are trainable (VideoMimic ablation; PhysHMR
      distills from a mocap-trained expert). Our existing clip IS that asset.
[1] Render OUR rig → ~10k images, randomized texture/background/lighting/view
      → auto-labels: 2D keypoints AND our exact joint angles (free, exact)
[2] Fine-tune a 2D pose net on those renders
[3] Domain-adapt on unlabeled REAL deer video (self-training)  ← MANDATORY
[4] Temporally-constrained fit against OUR URDF → joint angles directly
[5] Root/contact: back-project 2D keypoints to camera rays; condition on
      joint-to-ray displacement (sidesteps metric-root entirely)
[6] Fine-tune the stage-0 policy on video-derived refs
```

**Key precedents (verified):**
- **Uni-Mo** (arXiv 2606.28237) — closest precedent: fine-tunes ViTPose on **rendered
  images of the robot's own model**, then fits the **robot's own URDF** (root pos +
  quat + joint angles), 96.7% real-hardware success. **BUT** its stated limitation is
  that the subject must stay fully visible in a **fixed camera**, so it's biased to
  in-place behaviors; *"extending to large-translation locomotion would require tracking
  cameras or a multi-view setup"* — **that's exactly our locomotion case, unsolved.**
  Also a rigid CAD robot, not a furry deer; 1-month-old unrefereed preprint.
- **Mu et al. CVPR 2020** (`Learning-from-Synthetic-Animals`) — the synthetic recipe:
  **5k random-texture + 5k original-texture images per animal, 640×480, backgrounds
  randomized from COCO val2017, 4:1 split**, auto-labeled with keypoints, dense 2D+3D,
  depth, part segmentation. Used professional Unreal CAD assets (better than our rig).
- **UDA-Animal-Pose** (CVPR 2021 Oral, repo live) — the domain-adaptation half.
  Rendering alone does NOT close the gap.
- **STMR** — contact-preservation/foot-locking for the retarget stage.

**Biggest open risk (this is your #1 question to answer):** the sim-to-real gap for a
**hobby-quality mesh**. Published work used professional CAD. Our deer's silhouette and
proportions are accurate but surface appearance is crude. **Nobody has measured this.**

**Cheapest decisive experiment:** render our deer → train 2D pose on synthetic only →
run on real deer footage → hand-label ~30 frames → measure PCK. ~1 day, answers the
question that gates everything downstream.

---

## 4. Constraints (important)

- **THE GPU IS OCCUPIED.** A locomotion training run (Isaac Lab, ~13.5 GB of a 16 GB
  RTX 5080) is running and must not be disturbed. **Do CPU-only work** until told the
  GPU is free. That means: research, design, scaffolding, data-pipeline code, converter
  code, validation tooling, dataset acquisition planning. Do NOT launch GPU training or
  heavy inference. Check with `nvidia-smi` before anything GPU-touching.
- **Do not modify** anything under `training/` (envs, agents, tasks, scripts) —
  that's the locomotion agent's workstream. Your code goes in `tools/cv/` (new).
- Windows + Git Bash. Isaac's python: `/c/isaac-lab/env_isaaclab/Scripts/python.exe`
  (has torch/cuda). A separate system python exists for light work.
- Prefer legally clean, well-documented, reproducible approaches. Flag licensing
  explicitly for anything that could end up in a public portfolio repo.

---

## 5. What to deliver

1. **Deepen the research** where evidence is thin — especially: (a) the sim-to-real gap
   for low-fidelity meshes, (b) what actually works for large-translation locomotion from
   a moving/tracking camera (Uni-Mo's unsolved case), (c) whether physics/RL can absorb
   enough reference noise that a cruder extractor suffices — this sets the accuracy bar
   for every upstream stage and could dramatically reduce scope.
2. **Build what's buildable CPU-side.** Concretely: the synthetic-render pipeline design
   (we have the rig, meshes, and a working offline renderer — see
   `tools/render_playback_meshed.py` for FK+mesh rendering, and note Blender MCP is
   available in the owner's session), the keypoint schema for our rig, the
   `.npy` converter, and validation tooling (2D reprojection error, bone-length
   consistency, foot-slide metrics).
3. **Write findings to `docs/`** as you go — append to `docs/REFERENCES.md` for verified
   source status, and keep a running design/progress doc. Assume your context will be
   lost; write for the next agent.
4. **Flag thin evidence explicitly.** This project has been burned repeatedly by
   confident claims that dissolved under verification (a stale dependency pin cost a long
   detour; posture metrics lied three separate times). Prefer "unverified" over polish.

---

## 6. Hard-won lessons that apply to you

- **Verify claims against primary sources**, not project pages. Four separate
  MoCapAnything marketing claims were refuted 0-3 under adversarial verification.
- **Metrics lie about quality.** Posture metrics repeatedly showed a healthy deer that was
  actually face-planted or skating. Always validate visually / against ground truth.
- **Check the repo state, not the paper.** Two "released" pipelines turned out to be
  empty or partial.
- Field moves fast here: MoCapAnything V1 (Dec 2025) → V2 (Apr 2026) → TopoCap (Jun 2026)
  in seven months. Re-check before committing engineering time.
