# References — key sources for aniML World

Curated papers/methods that actually inform this project's direction. Grouped
by the role they play. Annotations say **why it matters to us**, not just what
it is.

Last updated: 2026-07-22.

---

## Video → motion (the "we have no animations" unlock)

- **Learning aggressive animal locomotion skills for quadrupedal robots solely
  from monocular videos** — npj Robotics, Sept 2025.
  <https://www.nature.com/articles/s44182-025-00048-x>
  (open PDF via ResearchGate: <https://www.researchgate.net/publication/395561813>,
  video: <https://www.youtube.com/watch?v=SGf0Nkx8t9A>)
  **Why it matters:** the closest thing to exactly what we want — take ordinary
  monocular video of an animal, extract a physics-usable reference motion, and
  learn the skill with imitation RL. Pipeline: robust **2D pose** → Kalman
  fusion on a skeleton graph → **spatio-temporal GCN** lifts to **3D joint
  trajectories** → **retarget** to the robot joint space → **generative
  imitation learning** in sim. They learned gallop, tripod, bipedal, backflip
  and deployed on a real robot. This is the template for our Phase 4 data path.

- **SFV: Reinforcement Learning of Physical Skills from Videos** — Peng et al.,
  SIGGRAPH Asia 2018. <https://arxiv.org/pdf/1810.03599>
  **Why it matters:** the foundational "video → 3D pose → motion-imitation RL"
  paper (same lead author as AMP). Establishes the whole pattern and its
  failure modes. Read first — it's the conceptual ancestor of the npj paper.

- **SMAL / BARC / hSMAL** — parametric 3D shape+pose models for quadrupeds.
  SMAL (toy-scan quadruped model), BARC (dogs, breed-aware):
  <https://www.researchgate.net/publication/363906392>, hSMAL (horses):
  <https://arxiv.org/pdf/2106.10102>
  **Why it matters:** these are the 3D-lifting engines for the pipeline above.
  **BARC ≈ dogs ⇒ our WOLF maps onto the best-validated model.** Deer is the
  harder case (no deer-specific SMAL; would lean on generic SMAL or hSMAL-style
  fitting). See `docs/RESEARCH_NOTES.md` for the feasibility breakdown.

- **DeepLabCut / ViTPose-animal (AP-10K)** — pretrained 2D animal keypoint
  detectors. **Why it matters:** the first stage (2D pose) is basically solved
  by off-the-shelf models; we don't train this from scratch.

---

### >>> BEST PATH FOUND: PFERD horse mocap (deep-research #2, 2026-07-23) <<<

- **PFERD — Poses for Equine Research Dataset.** Harvard Dataverse
  `doi:10.7910/DVN/2EXONE`; descriptor Nature Sci Data 2024 (s41597-024-03312-1);
  loader `github.com/Celiali/PFERD`. 5 horses, **56 Qualisys cams @240Hz**,
  117-132 markers (~50 anatomical). **`MODEL_DATA/[Trial]_hsmal.npz` = beta(10
  shape) + theta (per-joint relative rotations, 3x36 = 108-dim) + gamma
  (translation).**
  **WHY IT WINS:** fits were optimized against calibrated ARENA marker
  trajectories, so **gamma is a true WORLD-frame root trajectory** — the exact
  thing no video extractor (monocular or multi-view) can supply. Plus
  horse->deer is ungulate->ungulate (far closer body plan than dog->deer).
  - GOTCHA: **the FBX is NOT a rig** — it's a marker cloud + camera objects.
    Articulated pose lives ONLY in the hSMAL npz.
  - GOTCHA: hSMAL was learned from **37 horse TOYS**; approximate anatomy.
  - GOTCHA: repertoire claim REFUTED (0-3) — it's walk/trot/piaffe/passage/
    pirouette/jump/rear/kick (horse+dressage), **NOT deer startle/flee**.
  - **!! LICENSE (verified via Dataverse API, 2026-07-23) — DECISION-CRITICAL !!**
    NOT CreativeCommons. Custom *"Data Copyright License for non-commercial
    scientific research purposes"*: commercial use **prohibited** (only
    "non-commercial scientific research, non-commercial education, or
    non-commercial artistic projects"); **"The Data shall not be copied,
    shared, distributed, re-sold... or sub-licensed in whole or in part"**;
    **"may not be reproduced, modified and/or made available in any form to
    any third party without the Licensors prior written permission."**
    => Private training use looks OK; **publishing retargeted .npy clips to a
    public repo = redistribution of a modified form = NOT permitted** without
    written permission. Options: keep clips local + publish only policy/videos;
    or ask the authors (license contemplates written permission). Note the
    video fallback has the same flavour of problem: SuperAnimal's Quadruped-80K
    (Zenodo 14016777) is Modified-MIT **academic/non-commercial**.
- **Ranked alternatives:** monocular SuperAnimal-Quadruped (Nature Comms 2024,
  HF `mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped`) + **FMPose3D** in DLC
  3.x is near-turnkey and cheaper than assumed (Horse-10 zero-shot 0.228 NE vs
  AP-10K 0.287) BUT its model card says *"not suitable... for applications that
  require high keypoint precision"* — IK is exactly that; fine-tuning reaches
  0.07 NE (~3x better), so budget hand-labeling deer frames; deer not
  benchmarked. **Anipose** (BSD-2) multi-view = lowest ML risk, native joint
  angles, but stops at 3D keypoints, unmaintained since 2021, risk moves to
  filming logistics. **RGBD-Dog** = real 43-joint/95-DOF BVH but academic-only,
  unobtainable, dog. **DogMo** (2510.24117) DOES recover global root (useful
  later for WOLVES).
- **Biomechanics-literature shortcut: DEAD (0-3 on the strong claims).** Only
  ~6 **sagittal** flex/ext DOFs as summary scalars + plotted curves; no
  spine/neck/toe/abduction, no world root, no downloadable per-frame series.
  Useful only to regularize/sanity-check, not to author a clip.
- **Reference-free (RLWAV/CPG/energy-min) + generative motion priors: NO claim
  survived verification.** Treat as UNEVIDENCED (not disproven).
- **npj repo correction:** `arclab-hku/Imitation_from_video` IS live, but direct
  inspection shows only legged_gym/rsl_rl/AMP training + already-extracted
  motion — **no 2D pose, no 3D lift, no IK/retarget.** It ships only the stage
  we already have.

---

### >>> VERIFIED STATUS (deep-research + code spike, 2026-07-23) — READ BEFORE BUILDING <<<

- **MoCapAnything / MoCapAnything V2** (arXiv 2512.10881 / 2604.28130,
  repo `phongdaot/MocapAnything`, weights `kehong/MoCapAnythingV2-weights`).
  **NOT USABLE FOR US (as of 2026-07-23).** Code+weights ARE real (MIT, 464 MB
  ckpt, 4 inference entrypoints) — not vaporware. BUT the headline
  "arbitrary/custom skeleton reference prompt" does **not** exist in released
  code. Four claims asserting it were refuted 0-3/1-2, and a direct read of
  `inference/video2pose2rot.py` confirms: `ref_seq` is a **fixed dataset
  lookup** (`derive_wild_ref` = "any sequence of matching **species**"),
  skeleton/rest-pose come from THEIR `bvh/`+`bvh_pose/`+`species_info_dict.npy`
  — *"Users cannot supply arbitrary skeletons; only species names select
  pre-built reference rigs."* Output is **root-relative** camera-space. Also:
  full 73-species dataset unreleased (demo subset only), repo may be an
  UNOFFICIAL reimplementation (project page is `animotionlab`, repo is
  `phongdaot`), weights license unconfirmed, and **zero** independent
  validation (TopoCap competes on the same benchmark and omits it entirely).
- **SLoMo — RULED OUT.** Repo has **zero source code** (`language: null`);
  only pointers to PPR/Lab4d + a Julia CI-MPC pkg + PyBullet-era
  motion_imitation. Dead since 2024-07. Its two open unanswered issues are
  literally our question ("How to adapt keypoints trajectory to the robot
  configuration?").
- **npj Robotics pipeline** — still the best *conceptual* match (DeepLabCut 2D
  + Kalman -> ST-GCN 3D lift -> constrained-IK retarget -> AMP, real AlienGo).
  BUT the claim its code is released at `arclab-hku/Imitation_from_video` was
  **REFUTED (0-3)** — treat as a paper to reimplement, not code to run.
- **UNIVERSAL GAP (the real design work):** every current arbitrary-skeleton
  extractor (MoCapAnything V2 **and** TopoCap, both verified) outputs
  camera-space **root-relative** motion with **no global root trajectory and no
  foot contact**. Our AMP `.npy` needs world root pos+quat. So a
  **root-trajectory + contact-cleanup stage is mandatory regardless of
  extractor.** Design it independently of the extractor choice.
- **Retargeting is solved by nothing.** No released method emits rotations for
  an arbitrary URDF; correspondences + IK constraints stay hand-defined (npj
  matched only hip+foot, knee via IK, tuned for a ~12-DOF dog — no guidance for
  our spine/neck/skull DOFs). Naive keypoint transfer (UVM) breaks contact
  schedules -> foot sliding/penetration; budget an explicit **STMR**-style
  (IEEE T-RO 2025, `arXiv 2404.11557`) contact-preservation + foot-locking
  stage. Reference impl worth reading: `google-research/motion_imitation`
  `retarget_motion/retarget_motion.py`.
- **Scope reality check:** with a mandatory root+contact stage and hand-defined
  25-DOF correspondences, video->motion is a **multi-week** project, not free
  parallel work. Field is moving fast (V1 Dec-2025 -> V2 Apr-2026 -> TopoCap
  Jun-2026); re-check repo state before committing engineering time.

---

### >>> CV workstream pass, 2026-07-23 — global root SOLVED, synthetic-only NOT <<<

Full analysis + repo-state checks in `docs/CV_PIPELINE_PROGRESS.md`.

- **THE "NO GLOBAL ROOT" GAP IS NOT UNIVERSAL — the human-motion literature
  solved it, and we had missed it.** `REFERENCES.md` previously stated that
  every extractor is camera-space root-relative. True of *animal/arbitrary-
  skeleton* extractors; **false in general.**
  - **TRAM** (ECCV 2024, `arXiv 2403.17346`, repo `yufu-wang/tram`, **MIT**,
    pushed 2025-06-08): DROID-SLAM on the background with dual masking of the
    subject → camera trajectory; metric scale from aligning SLAM depth to a
    metric-depth prediction; compose with camera-frame body motion. Claims
    **−60 % global motion error** vs prior work.
  - **WHAM** (CVPR 2024, `arXiv 2312.07531`, repo `yohanshin/WHAM`, **MIT**):
    lifts 2D keypoint sequences with a motion prior, fuses SLAM camera angular
    velocity, and **predicts foot contact** to refine the global trajectory.
  - **DROID-SLAM** (`princeton-vl/DROID-SLAM`, BSD-3) is the shared backbone.
  - **Why it matters to us:** background SLAM is *species-agnostic* — it does not
    care that the foreground is a deer. This is the answer to Uni-Mo's stated
    "large-translation locomotion is unsolved" limitation. Repo state + licences
    verified via GitHub API 2026-07-23. **UNVERIFIED:** whether they run on our
    hardware, and their weights' dependence on **SMPL (research-only licence)** —
    same public-repo exposure as the PFERD problem. Check before adopting.
  - **Cheaper alternative:** foot-contact odometry alone determines root
    translation when a stance foot is known, no SLAM needed. Implemented as a
    metric in `tools/cv/validate_motion.py::root_consistency`.

- **Synthetic-only rendering is NOT sufficient — quantified.** Mu et al. CVPR
  2020 Table 1, PCK@0.05 on real TigDog: **synthetic-only 25.33 (horse) /
  21.17 (tiger)** → CycleGAN 51.9/46.5 → BDL 62.3/52.3 → **CC-SSL self-training
  70.77/64.14**; real-data-only 78.98/81.99. They used **purchased professional
  UE4 CAD** and still got 25 PCK synthetic-only. **The domain gap dominates the
  asset quality.** The handoff's stage [3] domain adaptation is confirmed
  MANDATORY and worth ~+45 PCK. Also: part-segmentation as an auxiliary task is
  worth ~+1.4 PCK (their Table 3). Numbers read from the paper PDF directly.

- **ZebraPose** (`arXiv 2408.10831`) — the one genuine low-fidelity-asset
  success: 18k renders of a **Sketchfab hobby model**, **no domain adaptation,
  no real labels**, generalises to real zebras (PCK@0.05 0.466, 0.608 filtered;
  detection mAP50 0.512). **Do not plan around it** — zebras have uniquely
  distinctive stripes and the footage is aerial drone over open savanna. Deer in
  clutter is the Mu et al. regime. One transferable lesson: their largest single
  gain came from **viewpoint/scale coverage, not realism** (mAP50 0.150 → 0.512
  from cropped+rescaled variants).

- **ReActor** (`arXiv 2605.06593`, SIGGRAPH 2026) — bilevel optimisation that
  **adapts the reference motion to the robot morphology while training the
  tracking policy**, from only *"a sparse set of semantic rigid-body
  correspondences"*, demonstrated **retargeting onto a quadruped**. If real, it
  collapses both the hand-defined-correspondence problem and the
  reference-noise problem. **No code found, no quantitative tolerance numbers,
  unrefereed, ~2 months old. PROMISING / UNVERIFIED.**

- **Reference-noise tolerance: still UNANSWERED.** No published study gives an
  accuracy threshold. Best available reasoning: AMP scores a *distribution* of
  transitions with no per-frame alignment, so it forgives **jitter** but learns
  **systematic bias** faithfully. Recommended: measure it on our own rig by
  injecting controlled noise into the existing clean clip.

- **Handoff spec correction:** the `.npy` `root_quaternion` is **xyzw**, not
  wxyz. Verified in `amp_rsl_rl/utils/motion_loader.py::_resample_data_SO3`
  (`Rotation.from_quat`, "xyzw format (SciPy default)"). Also: the file's
  `joint_velocities` is **ignored** by the loader, which resamples positions and
  finite-differences them itself.

---

## Imitation / motion priors (the naturalness engine)

- **AMP: Adversarial Motion Priors for Stylized Physics-Based Character
  Control** — Peng et al., SIGGRAPH 2021.
  **Why it matters:** the tool for *naturalness*. A discriminator learns
  "does this look like the reference animal"; the policy is rewarded for
  fooling it. We already have AMP infrastructure (`amp-rsl-rl`); the two
  blockers that made us park it (root-axis bug, broken playback) are now fixed.
  Tolerant of noisy references — pairs well with video-extracted motion.

- **ASE: Large-Scale Reusable Adversarial Skill Embeddings** — Peng et al.,
  SIGGRAPH 2022. <https://xbpeng.github.io/projects/ASE/index.html>
  **Why it matters:** the frontier architecture for us. Learn a reusable
  **latent skill space** once (adversarial imitation + unsupervised RL); a
  high-level policy then composes skills by picking latent points. This is the
  right long-term shape: motor control becomes a reusable module the ecosystem
  brain commands. Résumé-gold.

- **C·ASE: Conditional Adversarial Skill Embeddings** — SIGGRAPH Asia 2023.
  <https://frank-zy-dou.github.io/projects/CASE/index.html>
  **Why it matters:** adds direct controllability over which skill fires —
  relevant once we want the ecosystem brain to deliberately select gaits
  (walk vs. flee-sprint vs. pounce).

- **Lifelike Agility and Play in Quadrupedal Robots using RL and Generative
  Pre-trained Models** — 2023. <https://arxiv.org/pdf/2308.15143>
  **Why it matters:** end-to-end example of pretraining a motion prior from
  animal data then reusing it for downstream quadruped tasks. A concrete
  blueprint for the "primitive + high-level controller" split (our Phase 1).

---

## Locomotion RL baselines (the stable base)

- **Learning Robust, Agile, Natural Legged Locomotion Skills in the Wild** —
  2023. <https://arxiv.org/html/2304.10888>
  **Why it matters:** states plainly that *reward-engineering produces
  unnatural gaits and is laborious* — the empirical backing for our stable-base
  pivot and for using imitation (not reward-shaping) for naturalness.

- **legged_gym / Isaac Lab velocity-tracking recipe** (Rudin/Margolis/Hwangbo
  lineage). **Why it matters:** the standard massively-parallel PPO locomotion
  stack our Stage1–3 + Stable configs are built on. What every shipping legged
  robot actually uses. Great for *stability*, weak for *naturalness*.

- **Imitation learning for legged robot locomotion: a survey** — 2025.
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12702866/>
  **Why it matters:** the map of the whole space (tracking vs. adversarial vs.
  skill-latent). Good orientation doc when choosing the Phase 4 method.

---

---

### >>> VERIFIED STATUS UPDATE (overnight research, 2026-07-23) <<<

Three parallel deep-research passes. Full memos with evidence tables and all
URLs live in `docs/research/`; decisions in `docs/CV_PIPELINE_DESIGN.md`.
Highlights that **change or retire** entries above:

- **The AMP discriminator never observes root position.** VERIFIED twice
  independently by reading the installed `amp-rsl-rl` v2.2.3
  (`get_amp_dataset_obs` = joint pos + joint vel + base lin vel + base ang vel,
  full stop) and cross-checked against our own env config. Also confirmed in
  ASE, IsaacGymEnvs, `AMP_for_hardware`, and the AMP paper's §5.3 spec.
  ⇒ **The "UNIVERSAL GAP" entry above is wrong for AMP.** Root-relative,
  camera-space output is *fine*; only root **velocity** matters, and only as 3
  of 36 observation dimensions. A world-frame root trajectory stage is not
  needed. (It *would* be needed for DeepMimic-style tracking, which uses
  world-frame end-effector and CoM terms — VERIFIED in the DeepMimic paper.)
- **TRAM / WHAM / GVHMR / GLAMR — investigated and rejected.** They do solve
  global root from a moving camera, but (a) nothing in our stack consumes the
  output, (b) all are welded to SMPL/SMPL-X + AMASS-trained human motion priors
  that have no deer equivalent, and (c) licences are portfolio-hostile (GVHMR:
  ZJU non-commercial with copyleft-ish derivative terms; GLAMR: NVIDIA
  non-commercial; all pull SMPL). **Use contact/foot odometry instead** — it is
  indifferent to camera motion, needs no background, and touches no
  contaminating licence.
- **`npj Robotics 2025` — further degraded.** Beyond the earlier finding that
  the repo ships only the RL half: its root recovery is **learned camera-frame
  regression** (STGNet trained on dog mocap projected through virtual cameras),
  not world-frame reconstruction, and the paper admits *"performance may degrade
  under non-frontal camera angles."* Repo is **GPLv2 + patent claims**. Full
  text still not obtained (paywall).
- **Uni-Mo / "Unleashing Infinite Motion" (2606.28237) — repo is empty.**
  `GaoLii/Quad-Imaginarium` is README.md only, 9 KB, no licence, no code, data
  "upon acceptance." Its useful contribution to us is a **production acceptance
  gate**: reprojection error <20 px mean / <100 px max (70.2 % of clips pass;
  97.6 % of survivors then train successfully).
- **PFERD is not needed.** The licence problem flagged earlier (non-commercial,
  no redistribution of modified forms) is moot — with root position unobserved,
  its headline advantage (a true world-frame root trajectory) buys us nothing.
- **SuperAnimal-Quadruped — licence confirmed restrictive.** Modified MIT,
  **academic/non-commercial only**; commercial use requires an EPFL agreement
  (VERIFIED on the model card). Its "not suitable for applications requiring
  high keypoint precision" caveat applies to *zero-shot* use; the authors state
  fine-tuning is the remedy. Use it to **bootstrap labelling**, not as a final
  model. For a fully permissive repo, fine-tune a COCO-pretrained HRNet instead.
- **NEW — ScarceNet (CVPR 2023, arXiv 2303.15023) is the decisive result for the
  synthetic-render question.** 36 hand-labelled real images per category score
  **73.83 mAP on TigDog, matching the full synthetic + domain-adaptation
  pipeline (73.66)**. The authors say plainly that labelling a few images beats
  generating synthetic data. ⇒ **the render-and-domain-adapt architecture is
  superseded by "hand-label ~150 frames and fine-tune."**
- **NEW — Mu et al. used a ~$50 UE4 game pack**, not professional CAD (horse
  3,299 verts / tiger 3,124). The brief's "#1 open risk" (hobby-mesh fidelity)
  was largely a non-issue.
- **NEW — deer is the easiest quadruped in the literature.** In AP-10K (794
  images) and transitively in Quadruped-80K. ScarceNet reaches **0.767 mAP on
  deer with zero deer labels** — best-transferring species in its table.
- **NEW — WASABI (CoRL 2022, arXiv 2206.11693):** an AMP-family discriminator
  seeing **only 10 dimensions and no joint angles at all** still learned a Solo-8
  backflip. Our 15-of-25 joint coverage is generous by that standard.
- **NEW — the AMP noise dose-response curve does not exist in the literature.**
  Nobody has published "corrupt the reference by σ, measure policy quality."
  Running it on our own rig is both the thing that sets our scope and a novel
  result. See `docs/CV_PIPELINE_DESIGN.md` §6.

---

## How to use this file

Add a source when it actually changes a decision, with a one-line "why it
matters to us." Keep it curated — this is a decision log, not a bibliography.
Deeper analysis of any of these goes in `docs/RESEARCH_NOTES.md`; full research
memos go in `docs/research/`.
