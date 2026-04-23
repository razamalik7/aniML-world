# Rigged Mesh → URDF → Isaac Sim Pipeline

Automated pipeline that converts any rigged animal mesh into a physics-ready
URDF for Isaac Sim RL training. One-time setup, then ~10 minutes per new species.

---

## How It Works (plain English)

Your artist gives you a rigged FBX — a 3D mesh with a skeleton inside it.
Each bone in the skeleton "owns" a region of the mesh surface via **vertex
weights** (the same data that makes the skin deform when the bone moves).

This pipeline exploits that data:

1. **For each body part** (torso, hip, thigh, shin, foot), the script finds
   mesh vertices influenced by that part's bones.
2. It wraps those vertices in a **convex hull** for **collision** — the
   tightest convex shape that contains them (PhysX-friendly).
3. It optionally exports **visual** STLs: actual mesh **triangles** assigned to
   each link (same bone-weight rules), so Isaac Sim shows a recognizable animal
   instead of blobs.
4. It writes a **URDF** file that describes the full robot: which links exist,
   how they connect via joints, joint limits, mass, inertia, and where each
   collision STL lives.
5. Isaac Sim's URDF importer reads this and auto-generates all the PhysX
   schemas (ArticulationRootAPI, RigidBodyAPI, CollisionAPI, joints) — the
   exact stuff that was missing from the raw FBX.

**Result:** a physics-ready USD you can train on.

---

## Files

```
tools/blender_urdf_pipeline/
  rig_to_urdf.py        ← The main script (runs inside Blender's Python)
  PIPELINE_GUIDE.md     ← This file

tools/isaac_windows/
  Run-BlenderToUrdf.ps1 ← PowerShell launcher (finds Blender, handles paths)

configs/species/
  deer.json             ← Deer bone mapping (template for other species)
```

---

## Species Config Reference (`configs/species/<name>.json`)

| Field | Type | What it does |
|-------|------|-------------|
| `species` | string | Name used in URDF and output filenames |
| `mesh_object` | string | Exact name of the mesh object in Blender (case-sensitive) |
| `armature_object` | string | Exact name of the armature object in Blender |
| `total_mass_kg` | float | Total body mass, distributed proportionally across links |
| `collision_method` | string | `"convex_hull"` (only collision option; visuals use raw tris) |
| `export_visual_meshes` | bool | Default `true`: write `meshes/<link>_visual.stl` per link; set `false` or use `--no-visual-meshes` to use collision hull for both |
| `torso_bones` | string[] | All bones that merge into the `base` link (torso/spine/head) |
| `legs` | object | Map of leg prefix → segment bone names |
| `legs.<prefix>.hip` | string | Bone name for the hip/shoulder of this leg |
| `legs.<prefix>.thigh` | string | Bone name for the upper leg |
| `legs.<prefix>.shin` | string | Bone name for the lower leg / knee |
| `legs.<prefix>.foot` | string | Bone name for the foot / hoof |
| `joint_defaults` | object | Per-segment-role joint limits and axis |
| `joint_defaults.<role>.axis` | [x,y,z] | Rotation axis in local frame |
| `joint_defaults.<role>.lower_deg` | float | Lower joint limit (degrees) |
| `joint_defaults.<role>.upper_deg` | float | Upper joint limit (degrees) |
| `joint_defaults.<role>.effort` | float | Max torque (N·m) |
| `joint_defaults.<role>.velocity` | float | Max angular velocity (rad/s) |
| `weight_threshold` | float | Min vertex weight to count (0.0–1.0, default 0.15) |
| `min_hull_vertices` | int | If fewer verts found, use fallback cube (default 8) |

### Leg prefix convention

| Prefix | Meaning |
|--------|---------|
| `HR` | Hind Right |
| `HL` | Hind Left |
| `FR` | Front Right |
| `FL` | Front Left |

### Finding your bone names

In Blender:
1. Select the armature object
2. Go to **Object Data Properties** (green bone icon in Properties panel)
3. Expand the bone hierarchy — these are your exact bone names
4. Copy them into the config JSON

Or run the script with `--dry-run` — if names don't match, it prints all
available bones so you can fix the config without guessing.

---

## Output Structure

```
<output-dir>/
  <species>.urdf              ← Main file for Isaac Sim import
  meshes/
    pelvis_collision.stl      ← Convex hull (physics)
    pelvis_visual.stl         ← Skinned mesh triangles (viewport)
    chest_collision.stl
    chest_visual.stl
    ... (one *_collision + one *_visual per link when export_visual_meshes is true)
```

### URDF link naming

The script produces standardized names that match the Isaac Lab training
environment regex patterns automatically:

| Link name pattern | Matches training env regex |
|-------------------|---------------------------|
| `pelvis` (or your torso base name) | `ANIML_DEER_BODY_BASE` (default `pelvis`) |
| `*_foot` | `ANIML_DEER_BODY_FOOT` (contains "foot") |
| `*_thigh` | `ANIML_DEER_BODY_THIGH` (contains "thigh") |

Set `ANIML_DEER_BODY_BASE` if your root link name differs from `pelvis`.

---

## Running the Pipeline

### Step 0: Standing rest pose in Blender (strongly recommended)

RL and Isaac Lab spawn the robot at **URDF joint angles = 0**. If your rig's
default rest pose is a bind pose (legs folded, spine vertical in Blender Z, etc.),
the animal will look "cooked" in sim no matter how you tune spawn height.

1. Open your rigged `.blend` (mesh + armature).
2. Select the **armature** → **Pose Mode**.
3. Pose the deer in a natural **standing** stance (four feet on the ground,
   spine roughly horizontal / natural).
4. Menu **Pose → Apply → Apply Pose as Rest Pose**.
   - This bakes the stand into the skeleton; joint zero = that stance in the URDF.
5. (Optional) **Object Mode** → select mesh + armature → **Apply** scale/rotation
   if you imported from FBX with odd transforms; scene units should be **meters**.
6. **Save** the `.blend**, then run the pipeline below.

Without this step, you can still export a URDF, but you will need artificial
`joint_pos` offsets and spawn heights — fragile and easy to get wrong.

### Prerequisites

- **Blender 4.2 LTS** (or 4.x compatible) installed
- A `.blend` file with the rigged animal (import FBX first if needed)
- The mesh must have vertex weights (standard for any rigged model)

### Step 1: Dry run (verify config)

```powershell
.\tools\isaac_windows\Run-BlenderToUrdf.ps1 `
    -BlendFile  "C:\path\to\deer.blend" `
    -SpeciesConfig "configs\species\deer.json" `
    -OutputDir  "assets\deer_urdf" `
    -DryRun
```

What to check:
- **No bone warnings** = config matches rig, proceed to step 2
- **BONE NAME WARNINGS** = some names don't match; the script prints all
  available bones — update the config JSON and re-run dry-run

### Step 2: Generate URDF

Same command without `-DryRun`:

```powershell
.\tools\isaac_windows\Run-BlenderToUrdf.ps1 `
    -BlendFile  "C:\path\to\deer.blend" `
    -SpeciesConfig "configs\species\deer.json" `
    -OutputDir  "assets\deer_urdf"
```

Takes ~30 seconds. Check the console output for per-link hull and visual triangle counts.

**After import (once):** With the deer standing on the ground plane in Isaac Sim,
note the **pelvis/root** world **Z** height. Set training default
`ANIML_DEER_SPAWN_Z` to that value (or update the default in
`training/envs/deer_usd_articulation.py`) so feet rest on the floor without penetration.

### Step 3: Import in Isaac Sim

1. Open Isaac Sim
2. **File → Import → URDF**
3. Browse to `assets/deer_urdf/deer.urdf`
4. Settings:
   - **Fix Base Link** = **unchecked** (we want a floating base for locomotion)
   - **Joint Drive Type** = **None** (Isaac Lab's DCMotor actuator handles this)
   - **Self Collision** = **checked**
5. Click Import
6. **File → Save As** → save the scene (e.g. `deer_physics.usd`)

### Step 2b: Re-run after rest-pose or mesh changes

Whenever you change the **rest pose** in Blender or regenerate STLs:

1. Re-run **Step 2** (Generate URDF) — overwrites `deer.urdf` and `meshes/`.
2. Re-do **Step 3** (Import URDF in Isaac Sim) — re-save `deer_physics.usd`.
3. Re-run **Step 3b** (`fix_usd_default_prim.py`) — refresh `deer_robot.usd` /
   `deer_robot_clean.usd`.
4. Update **`ANIML_DEER_SPAWN_Z`** / `deer_usd_articulation.py` default if
   standing height changed.
5. Start a **new** training run (old checkpoints do not match new kinematics).

### Step 3b: Extract robot prim (REQUIRED)

Isaac Sim's "Save As" saves the **entire scene** (ground plane, lights,
physics scene, etc.) with `/World` as the default prim. Isaac Lab expects
**only the robot** as the default prim. Run the extraction script:

```powershell
Push-Location C:\isaac-lab\IsaacLab
& .\_isaac_sim\python.bat `
    "C:\Users\nars7\Downloads\aniML_world\tools\fix_usd_default_prim.py" `
    "deer_physics.usd" "/World/deer" "deer_robot.usd"
Pop-Location
```

This copies just the robot subtree into a clean USD with `/deer` as the
default prim. Use `deer_robot.usd` (not `deer_physics.usd`) for training.

**Why:** Without this step, `activate_contact_sensors` can't find rigid
bodies at the expected paths and training crashes with "could not find
any bodies with contact reporter API".

### Step 4: Train

```powershell
$env:ANIML_DEER_USD = "C:\Users\nars7\Downloads\aniML_world\assets\deer\deer_physics.usd"
$env:ANIML_DEER_ACTIVATE_CONTACT_SENSORS = "1"
.\tools\isaac_windows\Run-AniMLDeerTrain.ps1
```

---

## Adding a New Species (the 10-minute version)

1. **Import FBX** into Blender, save as `.blend`
2. **Copy config:** `deer.json` → `wolf.json`
3. **Edit bone names:** select armature, read bone list, update JSON
4. **Adjust mass** (`total_mass_kg`) and **joint limits** if needed
5. **Run** the pipeline (same two commands, different file paths)
6. **Import URDF** in Isaac Sim, save USD
7. **Train** (update `ANIML_DEER_USD` path or create a species-specific launch script)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Mesh object not found" | Wrong `mesh_object` name in config | Script prints available objects — copy the right name |
| "Bone not found" warnings | Bone name typo or different naming convention | Script prints all armature bones — update config |
| Tiny collision on a link | Few vertices weighted to that bone | Lower `weight_threshold` (e.g. 0.05) or check weight painting |
| Physics sim explodes | Scale mismatch (cm vs m) | Set Blender scene unit to Meters before running |
| Deer falls through floor | `Fix Base Link` was checked during import | Re-import with it unchecked (floating base) |
| Joints don't move in training | Joint Drive Type was set during import | Re-import with Joint Drive Type = None |
| Torso collision too blobby | Convex hull of wide torso | Split torso into chest + abdomen sub-links in config |

---

## What This Doesn't Do (and what handles it instead)

| Thing | Why not here | Where it lives |
|-------|-------------|----------------|
| Visual mesh for Unity | Unity uses the original FBX/skinned mesh | Unity asset import |
| Multi-DOF joints (ball-and-socket) | URDF revolute joints are 1-DOF | Add extra links per DOF in config if needed |
| Automatic joint limit discovery | Needs biomechanical reference data | Manual config tuning after first training run |
| Texture/material | Isaac RL doesn't use textures | Unity handles visual fidelity |
