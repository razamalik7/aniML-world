# Animal Physics Body Pipeline

How to go from a Unity Asset Store mesh to a sim-ready physics body in Isaac Lab.
Documented from the deer pipeline (April 2026). Follow these steps for each new species.

---

## Prerequisites

- **Blender 5.0+** (free, any OS)
- **Isaac Lab** installed with `isaaclab.bat` working
- The animal mesh imported into Blender (`.fbx`, `.obj`, etc. from Unity Asset Store)
- Familiarity with Blender's armature/bone system

## Overview

```
Unity Asset Store mesh (.fbx)
  → Blender: rig armature, assign mesh to bones, define link/joint hierarchy
    → Python script: export URDF + per-link STL meshes from Blender
      → Python script: import URDF → USD via Isaac Sim's URDF importer
        → Isaac Lab: spawn with Rx(-90) rotation, train RL
```

Total time per species (after first): ~2-4 hours mostly manual Blender work.

---

## Step 1: Prepare the Blender File

### 1a. Import the mesh

Open Blender, import the animal mesh. If it comes with an armature (skeleton), great — skip to 1c. If not, you'll need to create one.

### 1b. Create/verify the armature

The armature needs bones for every rigid link you want in physics. For a quadruped:

| Link name | Blender bone(s) | Notes |
|-----------|-----------------|-------|
| `pelvis` | spine base + hip bones | Root of the articulation |
| `chest` | mid-spine + shoulder bones | Heaviest single link |
| `neck` | neck vertebrae | Can be 1-3 bones merged |
| `skull` | head + ears + antlers | Everything above neck |
| `{F/H}{L/R}_thigh` | upper leg | One per leg |
| `{F/H}{L/R}_shin` | lower leg | One per leg |
| `{F/H}{L/R}_foot` | foot + toes | One per leg |

**Critical:** make sure the armature is parented to the mesh with automatic weights or manual vertex groups. Every vertex must belong to exactly one bone's vertex group (for clean per-link mesh export).

### 1c. Check transforms

Run the inspection script to verify your Blender file:

```bash
blender --background your_animal.blend --python tools/inspect_blend.py
```

Key things to verify:
- **Armature world matrix**: note the rotation. Our deer had the armature rotated so it stood upright in armature-local space but lay flat in world space.
- **Bone positions in world space**: these become joint origins in the URDF.
- **Coordinate system**: Blender world is Y-up. Isaac Sim is Z-up. We handle this with a spawn rotation, NOT by rotating geometry.

### 1d. Define the link-bone mapping

Create a Python dict mapping link names to Blender bone names. Example from deer:

```python
LINK_BONE_MAP = {
    "pelvis":    ["spine.003", "shoulder.R", "shoulder.L"],
    "chest":     ["spine.004", "spine.005", "front_shoulder.R", "front_shoulder.L"],
    "neck":      ["spine.006", "spine.007", "spine.008"],
    "skull":     ["scull", "ear.R", "ear.L"],
    "HR_thigh":  ["thigh.R"],
    "HR_shin":   ["shin.R"],
    "HR_foot":   ["foot.R", "toe.R"],
    # ... etc for all legs
}
```

The **first bone** in each list is the "primary" bone whose head position becomes the link origin. Additional bones contribute mesh vertices to that link.

---

## Step 2: Set Anatomically Realistic Masses

This is critical. Wrong masses = wrong center of mass = the animal tips over regardless of RL rewards.

### Rules of thumb for quadrupeds

- **Total mass**: look up the real animal's weight (e.g. 70 kg adult doe)
- **Front:rear weight split**: ~60:40 for most ungulates (deer, horses). Front legs bear more weight.
- **Chest > Pelvis**: the chest carries the ribcage, shoulders, and supports the neck/head which hangs far forward
- **Head + neck**: surprisingly heavy and far forward — major CoM contributor
- **Front leg links > Hind leg links**: heavier because they're weight-bearing

### Deer reference values (73 kg total)

```python
MASSES = {
    "pelvis": 18.0, "chest": 28.0, "neck": 6.0, "skull": 5.0,
    "HR_thigh": 1.2, "HR_shin": 1.0, "HR_foot": 0.8,
    "HL_thigh": 1.2, "HL_shin": 1.0, "HL_foot": 0.8,
    "FR_thigh": 2.5, "FR_shin": 1.5, "FR_foot": 1.0,
    "FL_thigh": 2.5, "FL_shin": 1.5, "FL_foot": 1.0,
}
```

### Validate CoM position

After setting masses, calculate the weighted center of mass and compare to the foot contact polygon centroid. **The CoM must be within ~2cm of the foot centroid** or the animal will tip over in simulation:

```python
# Pseudocode
foot_centroid_y = average(all_foot_Y_positions)
com_y = sum(mass[link] * position[link].Y for link in all_links) / total_mass
offset = com_y - foot_centroid_y
assert abs(offset) < 0.02, f"CoM is {offset}m off — adjust masses"
```

---

## Step 3: Export URDF from Blender

### Why world-space export

We tried multiple approaches:
1. ~~Armature-local space export~~ — Isaac's URDF importer misinterprets it
2. ~~Pre-rotated Z-up export~~ — produces face-planting models
3. **World-space export + Isaac spawn rotation** ✓ — most reliable

The animal lies flat in Blender world space. We export bone positions and mesh vertices as-is. Isaac Lab applies an Rx(-90°) rotation at spawn to stand it up.

### Export script structure

See `tools/export_world_space_urdf.py` for the full implementation. Key steps:

```python
# For each link:
# 1. Get the primary bone's world-space head position = link origin
# 2. Collect all mesh vertices belonging to this link's bone vertex groups
# 3. Transform vertices to world space, then subtract link origin = link-local
# 4. Write link-local vertices as binary STL

# For each joint:
# 1. Get child bone's world-space head position = joint origin
# 2. Joint origin is relative to parent link origin
# 3. Rotation axis is X (left-right in Blender world = left-right in Isaac after Rx rotation)
# 4. Joint limits: typically [-1.5, 1.5] radians for legs, [-0.5, 0.5] for spine

# URDF structure:
# - All joints are revolute with axis="1 0 0" (X-axis bending)
# - Visual and collision meshes are identical (the STL files)
# - Inertia tensors: simple sphere approximation I = 0.4 * m * r^2
```

### Run the export

```bash
blender --background your_animal.blend --python tools/export_world_space_urdf.py
```

Output:
- `assets/deer_urdf/deer_world.urdf`
- `assets/deer_urdf/meshes/world/*.stl` (one per link)

---

## Step 4: Import URDF → USD

Isaac Lab reads USD, not URDF. The URDF importer adds physics APIs (ArticulationRootAPI, RigidBodyAPI, CollisionAPI).

See `tools/import_world_urdf_v2.py`. Key settings:

```python
importer.set_merge_fixed_joints(False)    # Keep all links separate
importer.set_fix_base(False)              # Free-floating root (not bolted down)
importer.set_import_inertia_tensor(True)  # Use our URDF inertia values
importer.set_distance_scale(1.0)          # Meters
importer.set_density(0.0)                 # Use explicit masses, not density
importer.set_self_collision(True)         # Links can collide with each other
```

After import, set `defaultPrim` on the USD stage:

```python
stage = Usd.Stage.Open(usd_path)
stage.SetDefaultPrim(stage.GetPrimAtPath("/animal_name"))
stage.Save()
```

### Run the import

```bash
# Must use Isaac Lab's Python (has pxr/omni modules)
isaaclab.bat -p tools/import_world_urdf_v2.py
```

### Verify the USD

```bash
isaaclab.bat -p tools/check_usd_masses.py
```

Confirm:
- All links have correct masses
- ArticulationRootAPI is on the root link (pelvis)
- RigidBodyAPI is on every link
- Total mass matches expectations

---

## Step 5: Configure Isaac Lab Environment

### Spawn rotation

The animal lies flat in the USD (exported from Blender world space). Isaac Lab's `init_state.rot` rotates it upright at every episode reset:

```python
# Rx(-90°) quaternion (w, x, y, z): stands the animal up from lying flat
_DEFAULT_ROT = "0.7071068,-0.7071068,0,0"
```

**Do NOT bake this rotation into the USD root prim.** Isaac Lab's `init_state` overrides it at every reset, causing compounding rotations.

### Spawn height

Measure the vertical distance from pelvis to feet in the rest pose:

```python
# In Blender (after mapping to Isaac frame):
# foot Z relative to pelvis = -0.920m for deer
# So spawn pelvis at Z = 0.93m (feet at ~0.01m above ground)
_DEFAULT_SPAWN_Z = "0.93"
```

### Default joint positions

If the URDF rest pose (all joints = 0) has the feet centered under the CoM, use all zeros. If not, set joint offsets to balance the foot polygon under the CoM:

```python
# Deer needed hind legs forward, front legs back to center feet under CoM
_DEFAULT_JOINT_POS = {
    "pelvis_to_HR_thigh": 0.4,   # hind forward
    "pelvis_to_HL_thigh": 0.4,
    "chest_to_FR_thigh": -0.4,   # front backward
    "chest_to_FL_thigh": -0.4,
    # all others: 0.0
}
```

### Physics properties

```python
RigidBodyPropertiesCfg(
    disable_gravity=False,
    linear_damping=0.05,
    angular_damping=0.05,
    max_depenetration_velocity=1.0,
)
```

Joint stiffness ~200, damping ~6 (env vars `ANIML_DEER_STIFFNESS`, `ANIML_DEER_DAMPING`).

---

## Step 6: Validate Before Training

### Quick visual check

```bash
isaaclab.bat -p training/scripts/play.py --task AniML-Deer-USD-Stand-v0 --num_envs 4
```

Verify:
- Animal spawns upright with feet on the ground
- No floating/displaced mesh pieces
- Orientation is correct (head forward, feet down)
- When you press X (reset), it returns to the same pose consistently

### Measurement script

Use `tools/measure_rest_pose.py` to dump actual world-Z positions after one physics step. Confirms spawn height and joint angles are correct.

---

## Lessons Learned (Mistakes to Avoid)

### 1. Never bake rotation into the USD root prim
Isaac Lab's `init_state.rot` **overrides** the USD root's `xformOp:orient` at every episode reset. If you bake Rx(-90) into both the USD AND init_state, they compound and the animal spawns sideways.

### 2. Always export in Blender world space
Armature-local space, pre-rotated Z-up, and other "clever" coordinate transforms all failed. The most reliable approach is: export raw world-space → let Isaac Lab handle the rotation.

### 3. Set masses BEFORE training
Wrong masses → wrong CoM → animal tips over → RL reward tuning becomes impossible. Get the physics right first.

### 4. Validate CoM vs foot centroid
The single most important physics check. If the CoM is more than ~2cm from the foot support polygon centroid, the animal WILL tip over no matter what rewards you use.

### 5. Don't use early termination for standing training
If you terminate episodes when the animal falls below a height threshold, PPO learns to intentionally collapse to escape penalty accumulation. Use rewards only; let every episode run to timeout.

### 6. Front and hind knees bend in opposite directions
For ungulates (deer, horses, etc.), **front knees bend forward** (like a human knee) and **hind knees bend backward** (like a dog's hock). If you set symmetric joint limits like `[-2.0, 0.2]` for all four legs, the front knees will hyperextend backward. Fix:
```python
"FR_thigh_to_FR_shin": ( 0.0,  1.8),  # front knee: forward only
"FL_thigh_to_FL_shin": ( 0.0,  1.8),
"HR_thigh_to_HR_shin": (-1.8,  0.0),  # hind knee: backward only
"HL_thigh_to_HL_shin": (-1.8,  0.0),
```

### 7. Reward simplicity wins
Many weak, competing rewards create exploitable local minima (e.g., "sploot" — lying flat satisfies foot-on-ground, low-torque, and upright-bonus simultaneously). One strong, unfakeable signal (pelvis-to-foot vertical gap) works better.

---

## File Reference

| File | Purpose |
|------|---------|
| `tools/inspect_blend.py` | Inspect Blender armature transforms and bone positions |
| `tools/export_world_space_urdf.py` | Export URDF + STL meshes from Blender world space |
| `tools/patch_urdf_masses.py` | Patch URDF with anatomically realistic masses |
| `tools/import_world_urdf_v2.py` | Import URDF → USD with correct physics APIs |
| `tools/check_usd_masses.py` | Verify USD has correct mass values |
| `tools/measure_rest_pose.py` | Dump actual spawn positions from Isaac Lab |
| `training/envs/deer_usd_articulation.py` | ArticulationCfg: spawn rotation, height, joint defaults |
| `training/envs/deer_usd_rough_env_cfg.py` | Environment configs: rewards, terminations, events |
| `training/envs/deer_rewards.py` | Custom reward functions |

---

## Adapting for a New Species

1. Import mesh into Blender, create/verify armature
2. Copy `export_world_space_urdf.py`, update `LINK_BONE_MAP`, `JOINTS`, and `MASSES` for the new animal
3. Look up real animal mass, set anatomically correct mass distribution
4. Run the export, then the URDF→USD import
5. Calculate CoM vs foot centroid — adjust masses until <2cm offset
6. Copy `deer_usd_articulation.py`, update joint names and spawn height
7. Set `_DEFAULT_SPAWN_Z` = distance from pelvis to feet + small margin
8. Visual check with `play.py`
9. Train with `AniML-{Species}-USD-Stand-v0`
