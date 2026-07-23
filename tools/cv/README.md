# `tools/cv/` — video → reference-motion workstream

Start with **`docs/CV_PIPELINE_DESIGN.md`**. It explains why this workstream is
much smaller than the original brief assumed, and it supersedes
`docs/CV_PIPELINE_HANDOFF.md`.

Everything here is CPU-only and needs no torch or cv2. The GPU belongs to
locomotion training.

## Two spaces, and why there are two of almost everything

The rig exists in two forms and **they are not the same geometry**:

| | URDF (`assets/deer_urdf/deer_world.urdf`) | Artist blend (`assets/blender/deer_standing.blend`) |
|---|---|---|
| what it is | 25 actuated DOF, 27 links | `Deer_001` mesh (977 v / 1,238 poly) skinned to the 33-bone `Deer_001_rig` |
| visual meshes | **convex-hull physics proxies** — 1,693 tris total, gaps between links, no hooves/toes/antlers | continuous, correct deer silhouette, UV-mapped |
| what it's for | **what AMP actually consumes.** FK, IK fitting, motion validation | **anything that must look like a deer.** Synthetic renders |
| module | `deer_rig.py`, `keypoint_schema.py`, `render.py` | `blender_synth.py`, `deer_keypoints.py` |

**Do not render the URDF and expect a deer.** `tools/render_playback_meshed.py`
renders the URDF, so the animal in policy-review videos is always slightly
disassembled — worth remembering when judging gait by eye.

✅ **The two keypoint schemas are reconciled, and the rigs are geometrically
identical.** `keypoint_schema.py` anchors 31 keypoints to URDF link origins;
`deer_keypoints.py` anchors 35 to artist armature bone heads/tails. Anchoring the
fit on true joint centres gives **scale 1.0000, rotation 0.00°, residual 0.00 cm
on all 12 limb joints** — a pure translation of `(0, -1.0206, 0.2905)`.

The one thing to know: **the URDF skips the armature's `shoulder.*` bone**, so
`URDF hip ↔ armature thigh.head`, one segment further down than the bone names
suggest. Verify with `python tools/cv/rig_correspondence.py --anchor limbs`.

The URDF's *surface* keypoints (nose, withers, eyes, antler tips, hooves,
tail_root) are derived from convex-hull proxy meshes and are **not** reliable —
`withers` sits 19 cm from its armature counterpart. Use joint centres for
anything quantitative.

## Modules

| file | what it does | self-test |
|---|---|---|
| `deer_rig.py` | URDF parse → 25 DOF / 27 links, FK, mesh loading, `missing_visuals()`. **FK verified to exact 0.0 parity with `render_playback_meshed.py`** — the assertion runs in the self-test, so a rig change that breaks agreement fails loudly. | `python deer_rig.py` |
| `keypoint_schema.py` | 31 keypoints in **URDF space** = 20 joint centres + 11 mesh-derived surface landmarks. Superset of AP-10K's 17 via `AP10K_MAP`. Ships `SKELETON` and `FLIP_PAIRS`. Asserts anatomical ordering and bone-length invariance. | `python keypoint_schema.py` |
| `deer_keypoints.py` | 35 keypoints in **artist-armature space**, for Blender-side labelling. AP-10K mapping verified; SuperAnimal mapping deliberately left `None` (the ordered bodypart list could not be obtained — guessing it would silently corrupt every label). | `python deer_keypoints.py` |
| `render.py` | Pure-numpy pinhole camera + z-buffer rasteriser over the URDF; returns RGB + part-segmentation + depth. ~0.1 s per 640×480 frame. Useful for reprojection in the URDF-fitting stage. | `python render.py` |
| `synth_dataset.py` | Mu et al. recipe over the **URDF** meshes. **Demoted** — see design doc §2; it renders convex hulls. | |
| `blender_synth.py` | Blender headless generator over the **artist mesh**. Poses sampled from the three real Unity clips already in the .blend as Actions. Emits image + 2D keypoints + 3-level visibility + 3D keypoints + camera intrinsics/extrinsics + the bone quaternions that produced it. | see below |
| `overlay_keypoints.py` | Draws labels on renders. **The verification step — not optional.** | see below |
| `motion_npy.py` | `.npy` read/write with **verified** conventions, `check_joint_alignment()` (the silent zero-fill trap) and `check_clip()`. | `python motion_npy.py --check assets/motions/deer_walk.npy` |
| `validate_motion.py` | The acceptance gates: bone length, foot slide, ground penetration, root-vs-foot-odometry, joint limits, velocity spikes, reprojection. `--render` emits a filmstrip beside the numbers. | see below |

## Typical invocations

Generate a small synthetic batch and **look at it**:

```bash
blender -b assets/blender/deer_standing.blend --python tools/cv/blender_synth.py -- --out /tmp/synth --num 8 --seed 1
```

```bash
python tools/cv/overlay_keypoints.py --dataset /tmp/synth --num 8 --contact-sheet
```

Validate a reference clip:

```bash
python tools/cv/validate_motion.py --clip assets/motions/deer_walk.npy --map-legacy --render docs/cv_walk_validation.png
```

Check a clip for the silent zero-fill trap before training on it:

```bash
python tools/cv/motion_npy.py --check assets/motions/deer_walk.npy --expect-joints chest_yaw_to_chest,HR_hipabd_to_HR_thigh
```

## Non-negotiables

- **`root_quaternion` on disk is XYZW.** The handoff says wxyz; it is wrong.
  Verified against `motion_loader.py` and the shipped clip. The wrong reading
  raises no error and silently ruins style learning.
- **Never zero-fill a missing joint.** `AMPLoader` aligns by name and fills
  absent joints with constant zeros, silently. The discriminator then learns
  "real deer keep their spine yaw at exactly 0."
- **Look at the overlay before trusting any dataset.** A frame-convention error
  produces label files that are internally consistent, numerically plausible and
  completely wrong. Two real defects were caught this way while building
  `blender_synth.py`; both would have passed any numeric check.
- **Cycles CPU only, never EEVEE.** EEVEE allocates GPU memory and the GPU is
  training. Check `nvidia-smi` before anything GPU-touching.

## Second-session additions

| file | what it does |
|---|---|
| `blender_dump_rig.py` | Dumps the artist armature's rest pose + actions to `data/artist_rig_rest.json` so the rest of the tooling needs no Blender. |
| `rig_correspondence.py` | Measures URDF <-> armature correspondence. Anchored on joint centres it is **exact**: scale 1.0000, 0.00 deg, 0.00 cm. Use `--anchor limbs`. |
| `check_rig_geometry.py` | Rig sanity gate. Catches mirrored/misplaced meshes, leg asymmetry, missing geometry. **Exits non-zero**; currently fails on the hind-foot defect. |
| `rebuild_reference_clips.py` | Corrected clips as **copies** in `assets/motions/v2/`. Never touches live clips or the converter. |
| `make_noise_suite.py` | The 14-variant corrupted-reference suite for the AMP noise experiment, + `MANIFEST.md`. |

**Two rig defects are currently open** — see `docs/RIG_DEFECT_hind_foot_collision.md`
and `docs/REFERENCE_CLIP_FINDINGS.md` §1. `check_rig_geometry.py` fails until the
first is fixed; that is intentional.
