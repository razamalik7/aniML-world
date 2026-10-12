# VERIFIED RIG DEFECT — hind foot/toe collision meshes are mirrored

> **STATUS: FIXED 2026-07-23 (Raza-authorized).** Applied the clean-scope fix �
> all 4 feet collisions repointed to legacy(==visual) meshes, both hind toes
> Y-mirrored (`*_toe_collision_fixed.stl`). Patched both the URDF and the loaded
> `deer_world_base.usd` in place (no re-import). Checker now exits 0 (3 failures
> -> 0, spread 33cm -> 2.7cm). Stance-pose front/hind hoof gap 69cm -> 3.5cm.
> See `NIGHT_LOG_2026-07-23.md` �6 and `docs/collision_fix_stance_compare.png`.
> Backups: `configuration/deer_world_base.usd.pre_footfix.bak`,
> `assets/deer_urdf/deer_world.pre_footfix.urdf`.

**Found 2026-07-23 (CV workstream, incidentally). Affects LIVE locomotion
training. Nothing was changed — this is for the locomotion owner to decide on.**

Reproduce:

```bash
python tools/cv/check_rig_geometry.py
```

---

## The defect in one line

The **hind** foot and toe *collision* meshes are mirrored along the anatomical
down axis, so the deer's hind legs have **no collision geometry at the hoof** —
their collision volume sits ~0.32–0.48 m *above* the ankle, up by the belly. The
front legs are unaffected.

## Evidence

All within a single link, where the URDF declares `<origin xyz="0 0 0" rpy="0 0
0"/>` for both visual and collision — so the two meshes are in the same frame and
must agree. Bounds are link-local, `+Y` = anatomical down.

| link | mesh | local Y min | local Y max |
|---|---|---|---|
| `HL_foot` | `HL_foot_visual.stl` | **+0.027** | **+0.434** |
| `HL_foot` | `HL_foot_collision_v3.stl` | **−0.479** | **−0.323** |
| `HR_foot` | `HR_foot_visual.stl` | +0.027 | +0.434 |
| `HR_foot` | `HR_foot_collision_v3.stl` | −0.479 | −0.323 |
| `FL_foot` | `FL_foot_visual.stl` | +0.122 | +0.372 |
| `FL_foot` | `FL_foot_collision_v3.stl` | +0.199 | +0.343 |
| `HL_toe` | `HL_toe_collision.stl` | −0.419 | −0.281 |
| `FL_toe` | `FL_toe_collision.stl` | +0.250 | +0.384 |

Front: visual and collision both positive — consistent. Hind: visual positive,
collision entirely negative — **on the opposite side of the ankle.**

**The correct hind geometry is already on disk.** The pre-v3 collision mesh
matches the visual exactly:

| mesh | verts | local Y min | local Y max |
|---|---|---|---|
| `HL_foot_collision.stl` (legacy) | 41 | **+0.027** | **+0.434** |
| `HL_foot_visual.stl` | 41 | +0.027 | +0.434 |
| `HL_foot_collision_v3.stl` (in use) | 36 | −0.479 | −0.323 |

So the "v3" regeneration of the collision hulls corrupted the hind feet only.

**The training USD inherited it.** Reading the USD's mesh points directly, in
local space (frame-independent, so no coordinate-convention confusion):

```
/deer/HL_foot/visuals/HL_foot_visual/.../mesh              216 pts   y [+0.027, +0.434]
/deer/HL_foot/collisions/HL_foot_collision_v3/.../mesh     180 pts   y [-0.479, -0.323]
/deer/HL_toe/collisions/HL_toe_collision/.../mesh           84 pts   y [-0.419, -0.281]
/deer/FL_foot/collisions/FL_foot_collision_v3/.../mesh     180 pts   y [+0.199, +0.343]
```

The USD's hind-foot collision points match the flipped STL exactly. `deer_world.usd`
is what training loads, so this is live.

## Consequence

Lowest collision point per leg, URDF rest pose (larger = lower):

| leg | lowest collision point |
|---|---|
| FL | +0.977 m |
| FR | +0.982 m |
| **HL** | **+0.647 m** |
| **HR** | **+0.647 m** |

**A 33 cm front/hind asymmetry.** For the hind limbs, the lowest collision
geometry is the *shin*; there is nothing at the hoof. For both hind and front
feet to rest on the same ground plane, the rear of the animal must sit ~33 cm
lower than the geometry intends — or the hind legs must extend abnormally, or the
hind end drags.

The artist armature says the hind hoof should reach *lower* than the front
(Y = 1.21 vs 1.12), so the sign of the error is unambiguous: the hind legs should
be the longer pair, and in collision they are 33 cm shorter.

Note the **visual** meshes are fine on all four legs (lowest points 1.006 / 1.004
/ 1.031 / 1.031). So playback videos render a normal-looking deer while physics
sees a hind-legless one. That is precisely the kind of mismatch this project's
`MEMORY.md` warns about — except here it is the *render* that lies, not the
metric.

See `docs/cv_urdf_collision_rest.png` (collision, defect visible: the red hind
foot/toe clusters sit up by the pelvis) versus `docs/cv_urdf_visual_rest.png`
(visual, all four legs correct).

## Why this may matter a great deal

This is a plausible mechanical contributor to the hind-limb pathologies
documented across roughly 40 training runs — belly-sliding, hind splay, crouch,
rear-end dragging, the "deer scoots forward on hind push" pattern. A policy whose
hind hooves cannot contact the ground has no way to push off with them, and would
be expected to discover exactly those workarounds.

**Stated as a hypothesis, not a conclusion.** The geometry is verified; the causal
link to specific observed behaviours is not, and the pathologies have other
documented contributors (the chest-COM error, reward shaping). The honest claim
is: this is a real defect, it is upstream of every locomotion run to date, and it
should be fixed before more reward-shaping iterations are attributed to policy
behaviour.

## Suggested fix (NOT applied)

Cheapest correct option — point the hind foot collisions at the legacy meshes,
which are already correct and already on disk:

```xml
<!-- assets/deer_urdf/deer_world.urdf, links HL_foot and HR_foot -->
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry><mesh filename="meshes/world/HL_foot_collision.stl"/></geometry>
</collision>
```

The hind **toe** collision meshes are also mirrored and have no legacy
counterpart, so they need regenerating (or mirroring in Y) rather than swapping.

Then regenerate the USD (`tools/import_world_urdf_v3.py`) and re-run
`python tools/cv/check_rig_geometry.py` — it should report zero failures.

Whoever regenerated the `*_v3` collision hulls should also check *why* only the
hind pair flipped; if the generator is still around, it will do it again.

## Secondary observation, weaker

`check_rig_geometry.py` also warns that **all four** foot links, and the skull,
have near-zero visual/collision bounding-box overlap. For the front feet the
collision hull is displaced ~0.3 m along Z (forward) relative to the visual. That
may be legitimate — a hoof does extend forward of the cannon bone, and the v3
hulls are longer than the visuals — but it is not obviously right either. Worth a
look; unlike the hind mirroring it is **not** confirmed as a defect.

## Regression guard

`tools/cv/check_rig_geometry.py` encodes all of this and exits non-zero on
failure, so it can gate a rig change. It checks, using only relative comparisons
within the rig (no external ground truth, no coordinate-convention assumptions):

- visual and collision on the same side of the link origin,
- left/right leg symmetry,
- per-leg collision reach along the down axis,
- visual/collision bounding-box overlap,
- links with missing geometry.

A mirrored mesh raises no error anywhere in the URDF → USD → PhysX → training
chain. The only symptom is a policy that behaves oddly for reasons nobody can
pin down.

---

## The suggested fix is VERIFIED to work (tested on a throwaway copy)

Applied to a **copy** of the URDF in a scratch directory — nothing live was
modified — with exactly two changes:

1. `HL_foot` / `HR_foot` collision meshes repointed from `*_collision_v3.stl` to
   the legacy `*_collision.stl`;
2. `FL_shin_to_FL_foot` / `FR_shin_to_FR_foot` limits changed from
   `lower="-0.3" upper="1.6"` to `lower="-1.6" upper="0.3"` (the separate
   fetlock defect, see `docs/REFERENCE_CLIP_FINDINGS.md` §1).

`python tools/cv/check_rig_geometry.py --urdf <patched copy>`:

| | before | after |
|---|---|---|
| mirrored-mesh failures | 3 | **0** |
| FL lowest collision point | +0.977 m | +0.977 m |
| FR | +0.982 m | +0.982 m |
| **HL** | **+0.647 m** | **+1.031 m** |
| **HR** | **+0.647 m** | **+1.031 m** |
| front/hind spread | **33 cm** | **5.4 cm** |
| exit code | 1 | **0** |

The hind hooves now reach *slightly lower* than the front, which is what the
artist rig says is correct (hind hoof 1.21 m vs front 1.12 m). The fix needs no
new geometry — the correct mesh has been sitting in the repo all along.

### Two caveats

- **The hind toe collision meshes are still mirrored** and this patch does not
  address them; there is no legacy toe mesh to swap to, so they need regenerating
  or mirroring in Y. They no longer dominate the leg's reach, so the checker
  passes, but the geometry is still wrong.
- **`check_rig_geometry.py` cannot detect toe mirroring**, because its
  visual-vs-collision test needs both meshes and the toe links have no visual.
  That is a real gap in the guard. Promoting the existing `*_toe_collision.stl`
  and `antler_collision_v2.stl` to `<visual>` elements would close it and fix the
  missing hooves in playback renders at the same time.

After applying to the real URDF, regenerate the USD
(`tools/import_world_urdf_v3.py`) and re-run the checker against the USD as well
— the current USD demonstrably inherited the defect, so the URDF fix alone is not
sufficient.
