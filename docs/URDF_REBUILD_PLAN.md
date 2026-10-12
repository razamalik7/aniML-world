# URDF (re)generation — investigation & Tier-3 plan

**Date:** 2026-07-23
**Question:** How should we generate a new deer URDF from the Poly Art / Malbers
rig (Tier 3), how did we do it before, and how do we make it **replicable across
all 10 animals** in the pack?

**One-line answer:** There is already a config-driven Blender→URDF pipeline built
for exactly this. The reason the first build was painful was **not** the
pipeline — it was (a) fixing coordinate frames *after* export instead of in
Blender first, and (b) a separate, fragile reference-clip converter whose
hand-maintained sign/limit/root assumptions drifted from the URDF. Tier 3
eliminates (b) entirely by extracting reference motion from the *same* rig the
URDF is built from.

---

## PART A — How the URDF was built before (retrospective)

### A1. The intended pipeline (good, and reusable)

`tools/blender_urdf_pipeline/rig_to_urdf.py` (689 lines) + `PIPELINE_GUIDE.md` +
`configs/species/deer.json`. Design:

- Input: a rigged mesh in Blender (armature + skinned mesh with vertex weights).
- For each link, collect the vertices weighted to that link's bones →
  **convex hull = collision**; the actual skinned **triangles = visual** STL.
- Walk the configured bone→link map; each link's joint origin = its
  `joint_bone` head in **Blender world space**; joint axis = config or an
  auto-heuristic (⊥ bone & world-up); masses distributed from config.
- Emit URDF + per-link STLs. Then `import_world_urdf_v3.py` → USD.

Config-driven, ~30 s to run, "~10 min per new species." A `wolf.yaml` already
sits next to `deer.json`, so multi-species was always the intent. **This is the
right skeleton to build on.**

### A2. Why the reality was messy — three root causes

**(1) Coordinate frames fixed *after* export, not in Blender first.**
`rig_to_urdf.py` authors everything via `arm.matrix_world @ bone.head_local` —
i.e. the URDF inherits whatever orientation the Blender scene has. The source
model came in mis-oriented (FBX Y-up / odd transforms) and instead of fixing the
Blender scene up front, the URDF was patched afterward. The fossil record of
that is a pile of one-off scripts:
`prerotate_urdf.py`, `export_zup_urdf.py`, `export_world_space_urdf.py`,
`fix_deer_rotation.py`, `fix_mesh_rpy.py`, `create_linklocal_visual_meshes.py`
vs `create_visual_only_meshes.py` (world-space vs link-local mesh confusion),
`reset_deer_xform.py`, `fix_deer_world_usd.py`, … This is also the origin of the
CLAUDE.md "body +X = anatomical left, +Y = down, −Z = forward" convention — a
*symptom* of never normalising the frame, which then forced anatomy-aware
reward variants (`track_forward_vel_anatomical_exp`, etc.).

**(2) The rig grew by hand-rebuilds, not by config.**
The pipeline's `legs` schema only models `thigh/shin/foot` (3 seg). Hip
abduction, toes, and the spine/neck yaw joints that make up today's **25-DOF**
rig were bolted on afterward by `rebuild_deer_urdf_v3.py` / `_v4.py`. So the
"single pipeline" degraded into pipeline + manual surgery.

**(3) The reference clips came from a *separate*, fragile converter.**
`tools/convert_unity_anim_to_npy.py` builds the `.npy` reference from a bought
Unity animation. It:
- keeps a **hand-maintained `URDF joint → Unity bone path` map**,
- extracts each joint angle as the **twist about bone-local X**,
- applies a **per-joint SIGN table** ("the twist direction may or may not match
  URDF X"),
- **clamps to hardcoded URDF joint limits** that "MUST stay in sync" with the
  URDF by hand, and
- **synthesizes `root_position` as linear forward motion at a constant
  `FORWARD_SPEED`** (the root is *invented*, not measured).

### A3. The consequences — most of ~40 runs of pain trace to A2

Cross-referencing `RETROSPECTIVE_v4.md` and `NIGHT_LOG_2026-07-23.md`:

| Symptom | Real cause | Which A2 root cause |
|---|---|---|
| Front carpus extends instead of folding (v4.14) | converter sign table had FL/FR fetlock inverted vs URDF | (3) |
| Feet float 0.27 m, never plant; flat `root_z` | root was **synthesized**, not measured | (3) |
| Reference joints clamped every frame | converter limits ≠ actual rig reach | (3) |
| HR knee stuck at −58° | URDF default pose −0.7 + stiff actuators hold it | (2) |
| "Everything rigid, no torso motion" | reference clip barely moved spine (converter/source) | (3) |
| Hind hooves have no collision (found 2026-07-23) | a `*_v3` convex-hull regeneration mirrored the hind meshes | (2) |
| Anatomy-aware reward variants needed at all | body frame never normalised to Isaac's | (1) |

**The theme: the URDF and the reference were built by two different tools with
two different sets of assumptions, and every drift between them became a bug that
took iterations to find.** Reward tuning then fought symptoms of these upstream
mismatches for months (v4.7–v4.33).

---

## PART B — How to build the new URDF (Tier 3), replicably

### B0. The governing principle

**One rig, one source of truth.** Build the URDF *and* extract the reference
clips from the **same** Malbers FBX skeleton, so joint axes, signs, limits, and
the root trajectory are shared **by construction**. This structurally deletes
root cause (3) — the fragile converter — which was the single largest source of
reference-quality bugs. It is the real reason Tier 3 beats Tier 2:

> **Tier 2 (retarget Malbers anim → our current rig) would RE-CREATE
> `convert_unity_anim_to_npy.py` for a new rig — the exact fragile hand-mapped
> converter that caused the bugs. Tier 3 removes the need for any mapping.**

### B1. What the Malbers deer rig gives us (verified 2026-07-23)

From headless-Blender inspection of `Deer Walk.FBX`:
- Skeleton (Empty hierarchy): `Root` / `CG → Pelvis → {Spine → Spine1 → Spine2 →
  Neck → Neck1 → Head}`, hind `Thigh → Calf → HorseLink(hock) → Foot → Toe0`,
  front `Clavicle → UpperArm → Forearm → Hand → Finger0`, plus `Tail`, ears, jaw.
- **3 spine + 2 neck segments** (ours: 2 spine + 3 neck) — richer torso, the
  exact DOF whose absence read as "rigid."
- Walk clip: **root motion present** (0.89 m/s real world path), **feet plant**
  (toe z→0.06 m, lift 0.18 m), lively spine/neck/tail. Verified by render
  (`docs/polyart_deer_walk_bodyframe.png`).
- All 10 animals in the pack share this Malbers rig convention → configs will be
  near-identical across species.

### B2. Coordinate discipline — do it ONCE, in Blender (kills root cause 1)

Before any export, normalise the scene so the pipeline emits a clean
Isaac-Z-up URDF with **zero** post-hoc fixups:
1. Import the animal FBX; set scene units = **meters**.
2. Orient to **Z-up, facing +X** (Isaac convention), then **Apply All
   Transforms** (location/rotation/scale) so bone world matrices are clean.
3. Pose a natural **standing** stance and **Pose → Apply → Apply Pose as Rest
   Pose** (PIPELINE_GUIDE Step 0). Joint-zero then = standing, so no artificial
   `_DEFAULT_JOINT_POS` offsets or fragile spawn-Z hacks.
4. Save `.blend`. *Every* frame-fix script from A2(1) becomes unnecessary.

### B3. Generalise the pipeline (one-time engineering)

Extend `rig_to_urdf.py` from the fixed `thigh/shin/foot` schema to a **generic
bone-tree walker**:
- Each **actuated** bone → one link (convex-hull collision + skinned visual) +
  one 1-DOF revolute joint to its parent; axis = config or auto-hinge.
- A per-animal **prune/merge list** folds non-actuated bones into their parent
  link as fixed geometry (ears, eyes, jaw, fingers; tail optional).
- Multi-segment legs and 3-segment spine fall out naturally (no hand-rebuild
  like `rebuild_deer_urdf_v4`).
- Joints needing 2 DOF (hip flex + abduction) → two stacked revolutes with a
  zero-length intermediate link (standard URDF quadruped trick).

Because all 10 Malbers animals share the rig, the per-animal config is mostly
the same bone names + a different `total_mass_kg` → this is what finally makes
"~10 min per animal" real.

### B4. Reference extraction from the SAME rig (kills root cause 3)

New converter (replaces `convert_unity_anim_to_npy.py`), driven by the *same*
config:
- For each animation FBX, read each actuated bone's rotation per frame and
  extract the joint angle **about the URDF joint's own axis** (same axis the
  URDF declares — no separate sign table, no drift).
- **Measure** the root trajectory from the animation (Malbers has real root
  motion) → convert to body-frame `base_lin_vel` / `base_ang_vel` (what AMP
  consumes). The Malbers walk is circular, so we take body-frame velocities and
  optionally straighten by removing yaw — **no synthesized flat root.**
- Trim the ~40-frame turn-in; emit `.npy` in the exact URDF joint order.
- Do this for walk / trot / run / idle → a full reference set per animal, all
  guaranteed in-range and grounded because they came from the rig itself.

### B5. Validation gates (tooling we now have, that we didn't the first time)

Gate every animal before spending GPU:
- `tools/cv/check_rig_geometry.py` — visual/collision same-side, left/right
  symmetry, per-leg reach, missing geometry (this is what caught the hind-foot
  mirror).
- FK + render skull-Z / foot-ground plane, and the body-frame gait render used
  on the Malbers clip.
- A "reference sanity" pass: 0 joint-limit violations, min hoof ≈ 0, stance
  fraction 60–75 %, non-flat root — the same metrics that condemned the old
  clips.

### B6. Physics + USD (reuse what works)

- URDF→USD via `import_world_urdf_v3.py` (per-link convex-decomp already tuned).
  Decide up front: regenerate the monolithic USD and re-apply physics tuning, or
  rebuild the base/physics/robot/sensor variant composition (the current split
  was assembled by separate tooling — see `configuration/deer_world_*.usd`).
- Masses/inertia from config (crude box inertia; expect one round of tuning).
- Actuators via the existing env-var infra (`ANIML_DEER_STIFFNESS` etc.,
  `DCMotorCfg`); velocity caps 50/30 per the PhysX silent-death note in CLAUDE.md.
- Spawn Z from a physics settle on the fixed rig (NOT a pre-fix measurement).

### B7. Per-animal replication workflow (the payoff)

1. Import `<Animal>.FBX` → orient + standing rest pose (B2) → save `.blend`.
2. Copy `configs/species/deer_polyart.json` → `<animal>.json`; adjust the
   handful of bone-name differences + `total_mass_kg`.
3. Run the generalised pipeline → URDF + meshes.
4. Run the reference extractor on that animal's anim FBXs → `.npy` set.
5. `check_rig_geometry.py` + render gates.
6. `import_world_urdf_v3.py` → USD; settle for spawn-Z.
7. Train (AMP env is already parameterised by env vars / task registration).

---

## Honest risks & open questions

- **Pipeline generalisation is real work.** The current pipeline is
  thigh/shin/foot only; the generic bone-tree walker + 2-DOF-joint handling +
  prune/merge is the main new engineering. Estimate: the bulk of the Tier-3
  effort.
- **Joint-axis agreement.** URDF joint axis and the extractor's angle axis must
  be defined from the *same* bone data or B4's "no sign bug" guarantee weakens.
  1-DOF revolutes drop off-axis motion; hips likely need 2 DOF.
- **Skinned weights present?** Poly Art mesh must have vertex weights for the
  hull/visual step (standard for a skinned FBX — confirm on import).
- **USD composition.** Whether to keep the variant composition or go monolithic
  is an open call; the current split wasn't produced by `import_world_urdf_v3`.
- **Inertia realism.** Config box-inertia is approximate; budget one tuning pass.
- **DOF scope.** Decide per animal what to actuate (spine 3, legs 5/leg, tail?)
  vs freeze (ears/jaw/eyes/fingers).

## Bottom line

The rebuild is **more tractable than the first time** because: the pipeline
already exists, the coordinate mess has a known one-time cure (B2), we now have
validation tooling that didn't exist before (B5), and — most importantly —
building URDF + reference from **one rig** removes the converter that caused the
majority of historical bugs. The cost is the pipeline generalisation (B3/B4) and
a physics-tuning pass. The reward is a **repeatable per-animal recipe** (B7) that
turns the 10-animal roadmap from ten hand-builds into ten config edits.
