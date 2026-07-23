# Deer Physics Audit — 2026-05-04 (revised after deeper checks)

**Source files audited:**
- `assets/deer_urdf/deer_world.urdf` (current physics, 22 links / 21 joints)
- `assets/blender/deer_standing.blend` (source rig: armature `Deer_001_rig`, mesh `Deer_001`, 33 bones, 977 verts)

## Headline findings

1. **Hind legs are anatomically way too light.** Each hind leg = 3.0 kg vs each front leg = 5.0 kg. Real deer have *heavier, more muscular* hind legs (110-120% of front mass). We have 60% of front. **This biases the policy toward front-leg propulsion**, plausibly the root cause of the front-knee-scoot exploit we keep hitting.

2. **L/R asymmetry in the thighs.** `HR_thigh` CoM at `(0.011, 0.099, -0.142)`, `HL_thigh` CoM at `(0.029, 0.082, -0.160)` — these should be exact mirrors (HL.x = -HR.x). They are not. `FR_thigh` and `FL_thigh` similarly asymmetric. Shins and hind feet *are* mirrored correctly. So we have a directional bias baked into the inertials.

3. **All inertia tensors are spherical** (ixx=iyy=izz, off-diagonals zero). This is wrong for any limb segment. A 30cm thigh should have ~5× larger transverse inertia than longitudinal. Limbs swing too freely about their long axis, won't pendulum properly.

4. **Source rig has 5 leg segments per leg**; URDF has 4. We dropped the **pastern (fetlock) joint**, which is the natural shock absorber on landing in real ungulates.

5. **Source rig has 9 spine segments**; URDF has 1 spine yaw + 1 chest pitch (combined into the body, not a real chain). We collapsed the entire dorsal spine.

6. **Mesh is head-heavy** (542 of 977 verts in skull). If we ever derive inertia from mesh density, head mass would be hugely over-inflated. Currently our skull mass (5 kg) is set by hand, ignoring this.

7. **Neck ROM is non-anatomical.** Yaw ±34°, pitch -34° to 46°. Real deer turns head ~180° to look behind, drinks at ~90° pitch down, browses at ~90° pitch up. Currently neck cannot drink, browse, or look behind.

8. **All collisions are mesh-based.** Slow, prone to bad contact resolution, and the antler mesh in the skull collision is *currently capable of self-colliding* with the back/neck during sharp head turns.

9. **No tail in rig.** Visual mesh has a tail but the rig doesn't bone it. So no tail-flag alarm signaling.

10. **No ear control in URDF.** Ears in the rig (`ear.R/L`) but not represented in URDF.

## Detailed inventory

### URDF link masses (current)

| Link | Mass (kg) | % total | Notes |
|---|---|---|---|
| pelvis | 18.0 | 24.7% | |
| chest | 28.0 | 38.4% | Includes ribcage, organs |
| neck | 6.0 | 8.2% | |
| skull | 5.0 | 6.8% | Including antlers |
| HR_thigh + HL_thigh | 1.2×2 = 2.4 | 3.3% | |
| HR_shin + HL_shin | 1.0×2 = 2.0 | 2.7% | |
| HR_foot + HL_foot | 0.8×2 = 1.6 | 2.2% | Foot+toe fused |
| FR_thigh + FL_thigh | 2.5×2 = 5.0 | 6.8% | **Heavier than hind?!** |
| FR_shin + FL_shin | 1.5×2 = 3.0 | 4.1% | |
| FR_foot + FL_foot | 1.0×2 = 2.0 | 2.7% | |
| 6 zero-mass frame links | 0.006 | 0.0% | HAA/yaw frames |
| **Total** | **73.0** | 100% | Reasonable for adult deer |

### Anatomical targets (white-tail deer biomechanics, 73 kg adult)

| Region | Target % | Target kg | Current kg | Δ |
|---|---|---|---|---|
| Torso (pelvis+chest) | 55-60% | 40-44 | 46.0 | +slightly heavy |
| Head + neck | 8-10% | 6-7 | 11.0 | +heavy by ~4 kg |
| Each forelimb | 6-7% | 4.4-5.1 | 5.0 | ✓ ok |
| Each hindlimb | 7-8% | 5.1-5.8 | 3.0 | **−2 kg per leg, too light** |

The mass shifted from hindlimbs into torso+head. Sum of misallocation: +2.5 kg torso+head, −5 kg hind legs = ~5 kg in the wrong place out of 73 kg total.

### L/R inertial asymmetry (CoM offsets in link frame)

| Link | HR/FR CoM | HL/FL CoM | Mirror? |
|---|---|---|---|
| HR/HL thigh | (0.011, 0.100, -0.142) | (0.029, 0.082, -0.160) | **NO** — both +x, y/z differ |
| HR/HL shin | (0.000, 0.058, 0.087) | (-0.000, 0.058, 0.087) | ✓ |
| HR/HL foot | (0.003, 0.242, -0.061) | (-0.003, 0.242, -0.061) | ✓ |
| FR/FL thigh | (-0.015, 0.087, 0.101) | (-0.015, 0.099, 0.104) | **NO** — both -x, y/z differ |
| FR/FL shin | (0.015, 0.137, -0.068) | (-0.014, 0.137, -0.070) | ✓ |
| FR/FL foot | (-0.000, 0.229, -0.020) | (-0.001, 0.233, +0.007) | **NO** — z differs by 0.027 |

3 of 12 limb segments have asymmetric inertials. Probably an artifact of mesh-derived CoM computation on a slightly-non-symmetric mesh.

### Joint limits review

| Joint | Current | Anatomical target | Status |
|---|---|---|---|
| Spine yaw | ±0.3 (±17°) | ±0.5 (±29°) | tight |
| Chest pitch | ±0.5 (±29°) | -0.7 to +0.5 | OK; bow-up needed for jumping |
| Hip abd (×4) | ±0.4 (±23°) | ±0.6 (±34°) | tight |
| Hip flex (×4) | ±1.4 (±80°) | -1.6 to +1.0 | OK-ish |
| Stifle hind (×2) | -2.2 to 0 (-126° to 0°) | match | ✓ |
| Stifle front (×2) | 0 to 2.2 (0 to 126°) | match | ✓ (correct sign for ungulate) |
| Hock (×4) | ±1.2 (±69°) | -2.5 to +0.3 | wrong: hock only flexes one way |
| Neck yaw | ±0.6 (±34°) | ±1.6 (±90°+) | **way too tight** |
| Neck pitch | -0.6 to +0.8 | -1.6 to +1.6 | **way too tight** — can't drink or browse |
| Skull | ±0.8 | ±0.5 nod | OK; need separate yaw though |

### Joint origin spot-check (URDF vs Blender bone heads)

Comparing URDF joint origins to Blender bone head positions in armature-local space:

| URDF joint | URDF origin (rel parent) | Blender bone head | Match? |
|---|---|---|---|
| pelvis_to_HR_hipabd | (-0.127, 0.190, 0.188) | shoulder.R head: (-0.127, 0.333, 1.012) − pelvis-equivalent | x ✓, y/z need spawn-frame conversion |
| pelvis_to_HL_hipabd | (+0.127, 0.190, 0.188) | shoulder.L head: (+0.127, 0.333, 1.012) | symmetric ✓ |
| chest_to_FR_hipabd | (-0.122, 0.215, -0.472) | front_shoulder.R: (-0.122, -0.226, 1.072) | x ✓ |

Lateral hip placement matches the rig. Y/Z need a coordinate-frame audit (URDF uses post-Rx(-90°) deer-forward = world -Y; Blender armature has Rx(+90°) applied at the object level). Suspect minor offsets but no major errors.

### Collision shapes

All 16 mass-bearing links use `<mesh filename="meshes/world/*_collision.stl"/>`. Per-link:
- pelvis, chest, neck, skull = mesh
- 12 leg segments = mesh

No primitives. Skull collision currently includes antlers (542 verts in scull region) — these can self-collide with neck/back during sharp head turns and burn PhysX time.

### Volume sanity check

Total mesh volume (bmesh): 0.207 m³. At deer density (~700 kg/m³): would imply ~145 kg mass. Our URDF total is 73 kg, so densities are coming out at 352 kg/m³ — half of biological. Either the mesh is non-manifold (volume calc inflated) or the masses are too low. Doesn't matter much for relative gait dynamics, but worth noting.

---

# Revised plan (now with Blender access)

What Blender unlocks:
- **Per-link volume estimation from vertex-group regions** (instead of trusting mesh-derived inertials)
- **Visual joint-pivot verification** with screenshots/renders
- **Pose-mode reachability tests** — set joint angles to drinking pose / browsing pose / look-behind, verify in viewport that limits + geometry actually allow it
- **Anatomical landmark identification** by looking at the actual mesh (e.g., where is the real hip socket, not just where the bone happens to be)
- **In-Blender modifications** with re-export to URDF via existing `tools/export_world_space_urdf.py` pipeline

## Phase 0 — Audit ✅ DONE
This document.

## Phase 1 — Fixes, ranked by likely impact

| # | Fix | Blender needed? | Effort | Why it matters |
|---|---|---|---|---|
| 1 | **Mass redistribution** — shift 5 kg from torso+head into hind legs (per anatomical %) | Reference only | 30 min | Likely THE cause of front-leg-dominant exploits |
| 2 | **L/R symmetry** — force HL = mirror(HR), FL = mirror(FR) for all inertials | No | 15 min | Removes directional gait bias |
| 3 | **Realistic inertia tensors** — replace spherical with limb-segment ellipsoid (axes from bone length + mesh cross-section) | Yes (cross-section measurement) | 1 hr | Limbs will pendulum correctly |
| 4 | **Replace mesh collision with primitives** — capsule per limb, box for chest/pelvis, sphere for skull (no antlers in collision) | Yes (capsule dim measurement) | 1 hr | Faster + no antler self-collision |
| 5 | **Anatomical joint limits** — expand neck especially (±90° yaw, ±90° pitch); fix hock asymmetric range | No | 20 min | Enables drinking/browsing/look-behind, removes non-physical hyperextension |
| 6 | **Re-import URDF→USD** + smoke-test 100 iters | No | 30 min | Validate nothing regressed |

Total: ~3-4 hours of work, with Blender acting as visual ground truth.

## Phase 2 — Optional additions (after Phase 1 validates)

| # | Add | Effort | Behavior unlocked |
|---|---|---|---|
| A | **Pasterns** — split foot into foot+toe (the rig already has this) | 1 hr | Shock absorption on landing; correct hoof flex |
| B | **Spine pitch joint** at sacrum-lumbar | 30 min | Pronk, jumping, rear-up |
| C | **3-DOF neck** (separate yaw + pitch + skull nod) | 30 min | Full head-pose control for behaviors |
| D | **Tail joint(s)** — 1-2 segments | 1 hr | Tail-flag alarm signal (future behavior) |
| E | **Ear joints** (or visual-only driven rotation) | 1 hr | Alert posture (future) |

## Phase 3 — Validation
- `tools/check_usd_masses.py` — confirm CoM target met (<2cm vs foot polygon centroid)
- `tools/measure_rest_pose.py` — confirm spawn pose matches viewer expectation
- 100-iter smoke train on Loco-S1 vs current baseline — no regression
- Visual: render side-by-side static pose + animation comparison vs current

## Recommended order

Start with **#1 (mass redistribution) + #2 (symmetry)**. They're 45 min combined, no Blender required, and they directly address the most likely cause of the front-knee-scoot exploit. We can smoke-test that before doing anything else, see if the failure mode shifts. If yes — strong evidence we're on the right track. Then proceed with #3-6.

## Open questions for the user

1. **Adult deer mass target?** 73 kg fits a small-to-medium white-tail buck. Is that the intent, or should we target a different species/size (mule deer ~90 kg, red deer ~150 kg)?
2. **Antlers?** Currently included in skull mass + skull mesh collision. Real bucks add 1-3 kg of bone on the head. Want to model them as a separate rigid body (cleaner) or keep fused?
3. **Are pasterns + spine pitch in scope for "Phase 1"** or do we explicitly defer them?

---

# Revisions after deeper verification (2026-05-04 cont'd)

## What changed in my conclusions

### 1. The "hindlimbs too light" claim — partially walked back

I asserted hindlimb mass should be 110-120% of forelimb mass. Web searches turned up no specific cervid limb-mass percentages I could cite. What the literature *does* say (Sellers/Manning, Hudson, Lopez-Calderon 2023):

> "Fore- and hindlimb mass scale according to geometric similarity (limb mass ∝ body mass¹·⁰) across terrestrial mammals."

So fore/hind mass ratio is roughly consistent across species, but the literature I can access doesn't give me a clean cervid number. **What I can confidently say:** our 6 kg hind / 10 kg fore (60% ratio) is unusually fore-heavy compared to general quadruped patterns where hind ≥ fore. The direction of the fix is right; the magnitude (target ratio) is uncertain.

### 2. Bigger insight from biomechanics search — actuator-side, not mass-side

From the deer escape-gait literature (research summary):

> "The accelerating impulse of both forelimbs combined did not differ significantly from that of both hindlimbs, while the forelimbs applied a greater decelerating impulse than the hindlimbs... **forelimbs appear to act as elastic struts rather than being actively retracted, while the hindlimbs are more actively extended or retracted during stance**."

**Translation:** in real deer, forelimbs are passive springs (energy absorbers), hindlimbs are active drivers (propulsion). They produce equal-magnitude force impulses but in opposite signs. The structural difference is *control-side*, not just mass-side.

**Our config has uniform stiffness 60 + damping 4 across all 12 leg joints.** That's not how real deer work. A more biology-aligned recipe:
- Front leg joints: lower stiffness (~30-40), more compliant — let them act as struts
- Hind leg joints: higher stiffness/torque budget — let them be the active drivers

This is potentially a bigger lever than mass redistribution. Hasn't been tried.

### 3. Computed CoM check (was missing from initial audit)

Approximate CoM in armature-local space using URDF masses + Blender bone positions:

```
Foot polygon centroid Y: -0.133  (forward = -Y)
Computed CoM Y:          -0.094
Difference:              +0.039 m  (CoM is 4 cm BEHIND foot polygon centroid)
```

For target 60:40 forelimb:hindlimb load distribution at standstill:
- Forefoot avg Y: -0.47
- Hindfoot avg Y: +0.21
- Required CoM Y: 0.6 × (-0.47) + 0.4 × (+0.21) = **-0.20**

Currently at -0.094, target -0.20. **CoM is ~0.10 m too far back** for proper anatomical weight distribution. This means hindlimbs bear *more* static load than they should, which biases the policy to use them for support rather than propulsion — leaving the front legs to do "work" via degenerate mechanisms (the front-knee scoot).

This is actually the strongest single piece of evidence we have for a physics-side fix.

### 4. Isaac Lab pipeline check — sound

Compared our `deer_articulation_cfg` to `ANYMAL_C_CFG`:

| Param | ANYmal-C | Our deer | Status |
|---|---|---|---|
| Stiffness (legs) | 40 | 60 | We're 50% stiffer (more aggressive) |
| Damping (legs) | 5 | 4 | Comparable |
| effort_limit | 80 | 300 | We allow much more torque |
| velocity_limit | 7.5 | 100 | We allow much more speed |
| max_lin_vel | 1000 | 50 | We tightened (RTX 50 PhysX stability) |
| max_ang_vel | 1000 | 20 | We tightened |
| solver_pos_iters | 4 | 16 | 4× more iterations (cost ~10-15%) |
| solver_vel_iters | 0 | 2 | More |
| self_collisions | True | True | same |
| soft_joint_pos_limit_factor | 0.95 | 0.95 | same |

**Verdict:** our configuration is in the same regime as the working Isaac Lab quadruped baseline. Tightened velocity caps and increased solver iterations are *correct* for our hardware (RTX 50-series PhysX issue documented in CLAUDE.md). No issues with the pipeline itself. ANYmal uses an LSTM actuator network from real motor data; we use simple DC — this affects sim2real but is fine for pure-sim training.

### 5. Joint limit feasibility test (Blender-based)

Computed whether our URDF joint limits permit canonical deer behaviors:

| Behavior | Required ROM | Available (sum chain) | Feasible? |
|---|---|---|---|
| Drinking (head pitch +90°) | +1.57 rad | +1.6 rad (neck pitch + skull) | ✓ marginal |
| Browsing (head pitch -90°) | -1.57 rad | -1.4 rad (neck pitch + skull) | ✗ short by 10° |
| Look behind (head yaw 180°) | 3.14 rad | 0.9 rad (spine + neck yaw) | ✗ off by 2.2 rad |
| Look sideways (head yaw 90°) | 1.57 rad | 0.9 rad | ✗ short by 0.7 rad |

So drinking *barely* works; browsing, look-side, and look-behind are mechanically impossible at current limits. Confirmed.

## Revised plan — sharper, evidence-based

### Phase 1 fixes, re-ranked by evidence strength

| # | Fix | Evidence | Effort |
|---|---|---|---|
| **1** | **Front:hind actuator gain split** — front legs Kp=30/Kd=3 (compliant struts), hind legs Kp=70/Kd=5 (active drivers). Differentiate via joint regex. | Biomech literature on real deer | 30 min |
| **2** | **Move CoM forward by ~0.10 m** — adjust pelvis CoM offset within link frame, optionally redistribute pelvis ↔ chest mass (e.g., -3 kg pelvis, +3 kg chest) | Computed: 4 cm off polygon centroid, 10 cm off 60:40 target | 30 min |
| **3** | **L/R symmetric inertials** — force HL.thigh = mirror(HR.thigh), FL.thigh = mirror(FR.thigh), FL_foot = mirror(FR_foot) | Read URDF — confirmed asymmetric | 15 min |
| **4** | **Realistic ellipsoid inertias** — replace spherical (ixx=iyy=izz) with measured limb-axis tensors using Blender bone length + mesh cross-section | Spherical is objectively wrong for elongated bodies | 1 hr |
| **5** | **Expand neck joint limits** — yaw ±1.6 rad each segment (chest_yaw + neck_yaw stack to ±π), pitch -1.6 to +1.6 | Confirmed: cannot browse, look-side, or look-behind currently | 15 min |
| 6 | Replace mesh collision with primitives (capsule limbs, box torso, sphere skull, antlers OUT of collision) | Slower + antler self-collision exists | 1 hr |
| 7 | Re-import URDF→USD + smoke test 100 iters | Validation | 30 min |

**Order of operations**: do **1 + 2 + 3** first (75 min), smoke test. That's the highest-evidence trio (biomech + computed CoM + observed asymmetry). If front-knee-scoot fades or shifts, we know we're on the right path.

### Re: the rotation/axis issue

Yes — with Blender access I can fix it cleanly:
1. In Blender, rotate the armature so +X = forward, +Z = up (standard URDF convention)
2. Re-export URDF via `tools/export_world_space_urdf.py`
3. Re-import to USD
4. Remove `init_state.rot=Rx(-90°)` from `deer_articulation_cfg`
5. Replace `track_forward_vel_anatomical_exp` with stock `track_lin_vel_xy_exp` in rewards
6. Re-bake reset states (`tools/bake_reset_states.py`)

**Cost:** invalidates all current AMP `.npy` motion files (axis bakes in), invalidates current trained checkpoints, and the AMP track is gone until we re-convert. The Loco track is fine since checkpoints retrain anyway. Recommend doing this *only* if we commit to abandoning current AMP work — otherwise defer.

### Phase 2 (later, after Phase 1 validates)

- **Front-leg compliance refinement** — could go further than just lower Kp: add explicit joint spring-damper that biases toward standing pose
- **Pasterns** (4 joints) — easy add, source rig has them
- **Spine pitch joint** — needed for pronk/rear/jump
- **3-DOF neck** — needed regardless
- **Tail joints, ear joints, jaw** — visual/behavioral polish, future

## Revised recommendation

Start with the **gain split (Fix #1) + CoM shift (Fix #2) + symmetry (Fix #3)** as the first 75-minute pass. These three are now backed by:
- Published biomechanics (gain split)
- Direct computation showing 10 cm CoM offset (mass shift)
- Direct file inspection showing asymmetric inertials (symmetry)

Then re-import to USD and smoke test. If the failure mode shifts, do Fixes 4-6 next.

The rotation fix gets queued for whenever we're ready to scrap AMP — independently from physics.

Sources for the biomech claims:
- [Escape gaits of white-tailed deer, mule deer, and their hybrids — ResearchGate](https://www.researchgate.net/publication/238029092_Escape_gaits_of_white-tailed_deer_mule_deer_and_their_hybrids_Body_configuration_biomechanics_and_function)
- [Musculature of the hip and thigh of the white-tailed deer — PubMed](https://pubmed.ncbi.nlm.nih.gov/1200389/)
- [Scale Effects between Body Size and Limb Design in Quadrupedal Mammals — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3832634/)
