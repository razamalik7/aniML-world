# Night session 2026-07-23 — CV workstream

Branch **`cv-workstream-overnight`** (5 commits on top of `ab10b6e`).
CPU-only throughout; the GPU stayed on locomotion training the whole time.
**Nothing live was modified** — verified by diffing `training/`, `deer_world.usd`,
`configuration/`, `assets/deer_urdf/`, `convert_unity_anim_to_npy.py` and the
shipped `.npy` clips against the base branch: empty.

## Findings, most consequential first

1. **Hind foot/toe collision meshes are mirrored** (`RIG_DEFECT_hind_foot_collision.md`).
   The hind legs have no collision geometry at the hoof; front reaches +0.977 m,
   hind +0.647 m. The correct mesh is already on disk (pre-v3). The training USD
   inherited the flipped one. **Proposed fix verified on a scratch copy**:
   failures 3 → 0, spread 33 cm → 5.4 cm.
2. **Front fetlock joint limits are wrong** (`REFERENCE_CLIP_FINDINGS.md` §1).
   The clip asks for [−1.20, −0.40] rad; the URDF allows [−0.30, +1.60], so the
   joint is pinned at its stop and can never reach the reference — a mechanism
   for the long-reported rigid front legs.
3. **The AMP discriminator never observes root position.** Deletes the
   global-root stage the brief called mandatory.
4. **Hand-labelling ~150 real frames beats the whole synthetic pipeline**
   (ScarceNet). Deer is the best-transferring species in the literature.
5. **Two earlier claims corrected.** The walk clip's 0.8 m/s is right; the
   "1.7× too slow" finding was a stance-estimator artifact. Duty factors from the
   artist source are 0.16–0.39 where a walk needs >0.5 — the source animation has
   no valid stance phase, so stance-based metrics on it are unreliable and
   contact-based root recovery cannot be validated here.
6. **The two rigs are geometrically identical** — scale 1.0000, 0.00°, 0.00 cm on
   all 12 limb joints. The URDF skips the armature's `shoulder.*` bone.
7. **SuperAnimal's 39-keypoint list resolved** from the primary source; it
   includes antler keypoints, and our schema now emits antler tips.

## A note on method

Two hypotheses of mine were refuted by measurement and are recorded as such: the
fetlock "sign fix" (the geometry said the clip was right and the rig was wrong)
and my first foot-odometry estimator (it disagreed with itself once the duty
factor was checked). Both would have been plausible to ship. The pattern that
caught them was the project's own rule — render it, measure the physical
quantity, don't trust the number that merely looks consistent.

## Where to pick up

1. Fix the two rig defects (both one-liners, both verified, both upstream of
   every run). `python tools/cv/check_rig_geometry.py` gates it and currently
   exits 1.
2. Run the noise suite when the GPU frees — `assets/motions/noise_suite/`,
   14 variants, `MANIFEST.md` explains each. It measures a curve nobody has
   published and sets the accuracy bar for the whole CV pipeline.
3. Label ~200 real deer frames and produce the labels-vs-accuracy curve.
   `export_coco_keypoints.py` and `eval_pck.py` are ready.
