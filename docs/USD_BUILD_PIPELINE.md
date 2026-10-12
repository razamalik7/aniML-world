# USD build pipeline (Poly Art / Malbers rig → validated Isaac USD)

**Status:** COMPLETE & VALIDATED for the deer, 2026-07-23.
**Purpose:** the exact, reproducible sequence that turns a Malbers Poly Art rig
(model FBX + animation FBXs) into a physics-validated Isaac Lab USD, so we can
replay it for the other 9 animals in the Forest Pack with a per-species config
swap and (almost) nothing else.

This is the **execution** doc. For *why* Tier 3 (rebuild from the same rig that
supplies reference motion) and the retrospective on the first painful build, see
[`URDF_REBUILD_PLAN.md`](URDF_REBUILD_PLAN.md). For the DOF/role decisions, see
[`rig_introspection_deer.md`](rig_introspection_deer.md) and
[`rig_rom_survey_deer.md`](rig_rom_survey_deer.md).

---

## 0. Ground rules learned the hard way

1. **Fix coordinate frames in Blender, before export — never after.**
   `R_ISAAC` = +90° about Z (Blender Z-up/−Y-forward → Isaac +X-forward /
   +Y-left / Z-up) is applied to every read. The model imports in a clean
   standing rest pose so **joint-zero == standing**.
2. **Extract reference motion from the same rig the URDF is built from.** This
   deletes the entire sign/axis/root-convention bug class that plagued the old
   `convert_unity_anim_to_npy.py`. The extractor measures each joint's axis from
   the *same* relative-rotation decomposition the URDF uses, so sim `joint_pos`
   and the reference `.npy` are in the same units, same signs, same order.
3. **Measure axes self-consistently.** `relL = D(parent)⁻¹ @ D(child)` where
   `D(bone) = q_anim @ q_rest⁻¹`. Measuring the axis from the same `relL` the
   extractor later decomposes fixed a 168 mm → 82 mm FK round-trip error. The
   residual 82 mm is the cost of 1-DOF fetlocks, not a bug.
4. **One mesh source for visual AND collision.** Per-link real-mesh STLs are
   written once and used for both `<visual>` and `<collision>`. This makes a
   mirror bug (the old rig's hind feet) structurally impossible — you can't
   mirror one without the other.
5. **Validate the validator.** See §6 — the biggest "USD explosion" turned out
   to be my test harness never writing the default joint state after reset.

---

## 1. Inputs

Per-species config is the single source of truth:
`configs/species/<species>_polyart.json` (deer version FINALIZED, human-approved
2026-07-23). It names the model + anim FBXs and, per bone: role, parent, measured
axis, ROM, purity, `dof`, `kind` (`joint` / `base_link` / `merge_geometry`), and
`phase1_active`. The `dof_spec` block is authoritative:

- `total_dof: 38`, `phase1_active_dof: 34`
- `two_dof_joints`: `L Thigh, R Thigh, L UpperArm, R UpperArm, Spine, Neck, Neck1, Head`
- `merge_into_head`: `Eye L, Eye R` (render-mesh only)
- `merge_into_base`: `CG`
- `phase1_locked_at_reference`: `Jaw, L Ear, R Ear, Tail` (actuated but locked
  in phase 1; unlock later for learned eating / alert / tail).

**Environment for all Isaac/Blender steps:**
- Isaac Sim Python: `C:\isaac-lab\env_isaaclab\Scripts\python.exe`
- `OMNI_KIT_ACCEPT_EULA=YES` (else the URDF import bootstrap hangs on an EOF EULA prompt)
- Blender 5.1 headless; FBX blend-shape import is monkey-patched off:
  `imp.blen_read_shapes = lambda *a, **k: {}`

---

## 2. Pipeline stages (6 tools, `tools/rig_pipeline/`)

Run in order. Stages 1–2 are one-time analysis; 3–6 are the build.

### Stage 1 — introspect (`introspect_rig.py`, Blender)
Reads model + anim FBXs, applies the canonical Malbers bone→role map, measures
per-joint axis / range / purity (SVD "purity" of the rotation vectors decides
1-DOF vs 2-DOF), and emits the draft `configs/species/<species>_polyart.json`
plus `docs/rig_introspection_<species>.md`.

### Stage 2 — ROM survey (`survey_joint_rom.py`, Blender)
Measures each joint's range across **all** clips and splits LOCOMOTION vs
EXPRESSIVE. This is what validated "animations cheat" — expressive clips drive
joints to 180°, eyes read 360° garbage — so cosmetic joints get locked in phase 1.
Emits `docs/rig_rom_survey_<species>.md`.

> After stage 2 the config is reviewed and **finalized by a human** (the deer's
> 38-DOF decision was checked joint-by-joint, not by count). Everything below
> consumes the frozen config.

### Stage 3 — build URDF (`build_urdf.py`, Blender)
The generator. For each link: collects the skinned mesh faces assigned to that
bone (majority-vertex `Counter`), writes a real-mesh STL used for **both**
visual and collision, measures the joint axis from the walk anim, applies scale,
and writes anatomical per-role mass (`ROLE_W`, not hull volume — hull volume
inflated the antlered head to 63% of body mass). 2-DOF joints become two stacked
revolutes with a massless (0.01 kg) `<bone>_dof0` intermediate link.

It also computes, **data-driven**, the collision metadata the importer needs:
- `HULL_BUDGET`: per-link `maxConvexHulls` by contact relevance (feet/toes 4–6,
  body 2–4, ears 1).
- `filter_pairs`: pairs of link geometries that **overlap at rest** (adjacent
  body regions bulge into each other in fat hulls; the two halves of each 2-DOF
  joint are co-located). Detected by nearest-point distance < 0.04 m between
  down-sampled link vertex clouds, unioned with the 2-DOF halves. **Left/right
  leg pairs and distal-limb-vs-body pairs do NOT overlap at rest → not filtered
  → different limbs still collide** (the "legs never pass through each other"
  guarantee is preserved, provably).

```
OMNI_KIT_ACCEPT_EULA=YES C:\isaac-lab\env_isaaclab\Scripts\python.exe \
  -m ... build_urdf.py \        # run via Blender's python; see tool header
  --config configs/species/deer_polyart.json \
  --model  "<Deer.FBX>" --anim "<Deer Walk.FBX>" \
  --axes-file assets/deer_polyart_urdf/build_axes.json \
  --outdir assets/deer_polyart_urdf --species deer --mass 80 --scale 0.8
```
Outputs: `assets/deer_polyart_urdf/deer_polyart.urdf`, `meshes/*.stl` (31),
`import_config.json` (`hull_budget` + `filter_pairs`, 34 pairs), `build_axes.json`.

### Stage 4 — extract reference (`extract_reference.py`, Blender)
Extracts joint angles about the **self-measured axes** + measured root
trajectory → `assets/motions/deer_polyart_walk.npy`. Keys: `joints_list` (38),
`joint_positions (T,38)`, `root_position (T,3)`, `root_quaternion (T,4)` xyzw,
`fps`. Runs an FK round-trip validation (168 mm → 82 mm after the self-consistent
axis fix). Also writes `deer_polyart_axes.json`.

> **Naming is load-bearing — the npy `joints_list` MUST equal the URDF/USD joint
> names verbatim.** amp-rsl-rl 2.2.3's `MotionLoader.load_data` aligns reference
> columns to sim joints by **string equality** (`if j in dataset_joint_names`);
> any joint that doesn't match resolves to an all-**zero** reference column — no
> error, just garbage style learning. The extractor now emits URDF `safe`
> spelling (`Pelvis_to_Spine_dof0`, `Spine_dof0_to_Spine`, …) exactly as
> `build_urdf.py` writes it. The first deer extraction used an older
> `{parent}->{child}` spelling and had to be relabeled post-hoc against the URDF
> (structural proof: normalize each name, assert it equals the URDF name or the
> URDF name minus a trailing `_dof0`; all 38 must match). If you ever change the
> extractor's naming, re-run that proof.

### Stage 5 — import to USD (`import_urdf.py`, Isaac)
`URDFParseAndImportFile` with `set_convex_decomp(True)`, `set_self_collision(True)`,
`set_make_instanceable(False)`, merge-fixed-joints off, fix-base off. Produces the
composed USD `deer_polyart.usd` + `configuration/deer_polyart_{base,physics,robot,sensor}.usd`.

> The default drive gains copied here (`1047.19751 / 52.35988`) are Isaac's URDF
> defaults and are **faith-based / placeholder** — the *training* actuator gains
> are set deliberately in the articulation config (§7), not inherited from here.

### Stage 6 — apply collision + filters (`apply_collision.py`, Isaac)
The importer produces an **instanced** USD whose instance proxies are read-only,
so per-link edits in stage 5 silently don't stick. This stage re-opens the USD,
**un-instances** (`SetInstanceable(False)` — flattens the prototype so collision
APIs become writable), then applies per-link `convexDecomposition` +
`maxConvexHulls` from `hull_budget`, and `FilteredPairsAPI` for the 34
`filter_pairs`. Verified from a freshly re-opened stage:
`decomp_meshes=31, approximations={convexDecomposition}, self_collision=True,
APPLIED_FILTERS:34/34`.

```
C:\isaac-lab\env_isaaclab\Scripts\python.exe tools/rig_pipeline/import_urdf.py \
  --urdf assets/deer_polyart_urdf/deer_polyart.urdf \
  --usd  assets/deer_polyart_urdf/deer_polyart.usd \
  --config assets/deer_polyart_urdf/import_config.json
C:\isaac-lab\env_isaaclab\Scripts\python.exe tools/rig_pipeline/apply_collision.py \
  --usd assets/deer_polyart_urdf/deer_polyart.usd \
  --config assets/deer_polyart_urdf/import_config.json
```

---

## 3. Validation (`validate_usd.py`, Isaac)

Standalone physics settle test — **no training env cfg**. Drops the Articulation
on a ground plane, writes the default root+joint state, holds the rest pose with
**deliberate** PD gains (not inherited numbers), settles 2 s under gravity, and
reports finite / upright / stands / quiet(|jvel|<2).

```
OMNI_KIT_ACCEPT_EULA=YES C:\isaac-lab\env_isaaclab\Scripts\python.exe \
  tools/rig_pipeline/validate_usd.py \
  --usd assets/deer_polyart_urdf/deer_polyart.usd \
  --spawn_z 0.95 --stiffness 200 --damping 15 --fix_base 1
```

**Result: PASS.** Spawns standing (joints ≈ 0, symmetric), finite, self-collision
quiet (|jvel| → 0.07 with feet clear), inter-limb collision preserved, front
hooves make correct ground contact. Free-base under gravity tips — expected; a
passive PD hold cannot balance a tall quadruped, that's the RL policy's job.

---

## 4. Artifacts produced (per species)

```
configs/species/<sp>_polyart.json            # frozen source of truth
assets/<sp>_polyart_urdf/
  <sp>_polyart.urdf
  meshes/*.stl                               # real mesh, visual+collision
  import_config.json                         # hull_budget + filter_pairs
  build_axes.json
  <sp>_polyart.usd                           # composed, validated
  configuration/<sp>_polyart_{base,physics,robot,sensor}.usd
assets/motions/<sp>_polyart_walk.npy         # 38-joint reference, fps=30
assets/motions/<sp>_polyart_axes.json
docs/rig_introspection_<sp>.md
docs/rig_rom_survey_<sp>.md
```

---

## 5. Replicating for the next animal (wolf, fox, …)

1. Point a new `configs/species/<sp>_polyart.json` at that animal's model + anim
   FBXs; run stages 1–2; **human-review** the roles/DOF (Malbers rigs share the
   Biped-style skeleton, so the role map largely carries over, but paw/digit
   counts and tail length differ — check, don't assume).
2. Run stages 3–6 unchanged; tune only `--mass` / `--scale`.
3. Validate (§3). Then build the training config from the produced USD (§7).

Everything downstream of the config is data-driven (hull budgets, filter pairs,
axes, masses), so the per-animal human work is the config review, not the code.

---

## 6. The two real bugs found (and the ghost-process gotcha)

**Bug A — the test harness, not the USD (mine).** `validate_usd.py` initially
never wrote the default joint state after `sim.reset()`, so joints spawned with
uninitialized velocity (~12.8 rad/s) and *everything* flailed — faking ~8
"explosion / fly-up" tests. Fix: after `sim.reset()`, write
`write_root_pose_to_sim` / `write_root_velocity_to_sim` /
`write_joint_state_to_sim(default_joint_pos, default_joint_vel)` then
`robot.reset()`. **Lesson: validate the validator before trusting a failure.**

**Bug B — 2-DOF / fat-hull self-collision (real).** The massless intermediate
link in each 2-DOF joint hides the two halves of that joint from PhysX's
auto-adjacency filter, and fat convex hulls of adjacent body regions overlap at
rest → co-located hulls violently repel. Fix: the data-driven `filter_pairs`
(§ Stage 3) filters exactly the rest-overlapping pairs and nothing else, so
inter-limb collision is untouched.

**Ghost-process gotcha.** `nohup` Isaac runs do **not** fully release GPU/system
memory after `sim_app.close()` — up to a dozen orphan `python.exe` holding
~14 GB. Always, after any Isaac run, verify with
`Get-Process python,kit | sort WS desc` and kill the large (>300 MB) leftovers;
small (1–64 MB) MCP/tooling python are fine. GPU should return to ~2.9 GB baseline.
Also: when piping Isaac output through `grep | tail`, the pipe **buffers to EOF**,
so you see nothing until the run exits — you can't tell "cooking" from "hung".
Prefer `python -u … > logfile 2>&1` and `Read` the log for live progress.

**amp-rsl-rl 2.2.3 MotionLoader contract** (verified against installed source,
`utils/motion_loader.py::load_data`). The reference `.npy` must be a pickled dict
with keys: `joints_list` (list[str], = URDF joint names), `joint_positions`
`(T,N)`, `root_position` `(T,3)`, `root_quaternion` `(T,4)` xyzw, `fps` (float).
The loader **computes** joint & base velocities by finite-difference after
resampling to `simulation_dt` — it does **not** read a `joint_velocities` key
(that was a 1.2.0-era field; harmless if present, unused). It matches
`expected_joint_names` (from the env's AMP obs `asset_cfg.joint_names`) to
`joints_list` by string equality and reorders/zero-fills — see the naming note in
Stage 4. AMP obs term order (env side) must stay `joint_pos, joint_vel,
base_lin_vel, base_ang_vel`.

---

## 7. Training config — built fresh, DONE & validated (2026-07-23)

All new training code lives in **`training/polyart/`** (mirrors `tools/rig_pipeline/`
on the build side), walled off from the legacy `deer_usd_*` configs. The only
touch to old code is one import line at the bottom of `training/tasks/__init__.py`.

| Piece | File | Gate result |
|---|---|---|
| Articulation (38 joints, 5 actuator groups, deliberate gains) | `training/polyart/articulation.py` | Isaac loads it; all group gains verified in sim; spawns onto feet — PASS |
| AMP env (flat, fwd walk, policy+amp obs, X-fwd stock rewards) | `training/polyart/amp_env_cfg.py` | constructs/resets/steps; obs policy 122 / **amp 82**; action 34 — PASS |
| RSI (mid-stride reference-state init, no separate bake) | `training/polyart/rsi.py` | spawns z 0.89–0.95, momentum ≤1.15 m/s, resampling works — PASS |
| Runner cfg (amp-rsl-rl **2.2.3** schema) + task reg | `training/polyart/amp_runner_cfg.py`, `tasks.py` | `train_amp.py` drives 3 AMP_PPO iters clean — PASS |

Key simplifications the Tier-3 rebuild bought: **+X-forward rig ⇒ stock Isaac Lab
velocity rewards work** (no anatomy shims); **upright at joint-zero ⇒ no Rx(-90)**;
**RSI reads the `.npy` directly** (no `.pt` bake to keep in sync).

Validation harnesses (all in `tools/rig_pipeline/`):
`validate_articulation.py`, `validate_env.py` (+ the 3-iter `train_amp.py` smoke).

Launch the first real AMP run (multi-hour, full GPU):
```
$env:PYTHONPATH='<repo>'; $env:ANIML_DEER_USD='<repo>\assets\deer_polyart_urdf\deer_polyart.usd'
C:\isaac-lab\env_isaaclab\Scripts\python.exe -u training\scripts\train_amp.py `
    --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs 2048 --headless
```

Minor open item: RSI worst-case foot penetration ~3 cm at spawn (one frame,
one env) — harmless (PhysX depenetrates, capped at 1 m/s); tighten with a small
spawn-Z offset if it shows up in early training.
