"""
Export deer URDF using WORLD-space bone positions and mesh vertices.

Key insight: Isaac Sim's URDF importer uses world-space convention.
In deer_standing.blend, the deer lies along -Y in world space (feet near Y=0).
Spawning with Rx(-90deg) root rotation stands the deer upright in Isaac's Z-up world.

This script:
1. Uses bone.head (world space) for joint positions
2. Transforms mesh vertices to world space, then makes them link-local
3. Outputs deer_world.urdf with identity visual/collision origins
4. Spawn rotation in Isaac: w=0.707, x=-0.707, y=0, z=0 (Rx -90deg)
"""
import bpy
import struct
import os
import math
from mathutils import Vector

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "deer_urdf", "meshes", "world")
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "deer_urdf", "deer_world.urdf")
os.makedirs(OUT_DIR, exist_ok=True)

# Bone -> link name mapping (primary bone determines link origin)
LINK_BONE_MAP = {
    "pelvis":    ["spine.003", "shoulder.R", "shoulder.L"],
    "chest":     ["spine.004", "spine.005", "front_shoulder.R", "front_shoulder.L"],
    "neck":      ["spine.006", "spine.007", "spine.008"],
    "skull":     ["scull", "ear.R", "ear.L"],
    "HR_thigh":  ["thigh.R"],
    "HR_shin":   ["shin.R"],
    "HR_foot":   ["foot.R", "toe.R"],
    "HL_thigh":  ["thigh.L"],
    "HL_shin":   ["shin.L"],
    "HL_foot":   ["foot.L", "toe.L"],
    "FR_thigh":  ["front_thigh.R"],
    "FR_shin":   ["front_shin.R"],
    "FR_foot":   ["front_foot.R", "front_toe.R"],
    "FL_thigh":  ["front_thigh.L"],
    "FL_shin":   ["front_shin.L"],
    "FL_foot":   ["front_foot.L", "front_toe.L"],
}

# (parent_link, child_link, bone_at_joint_origin)
JOINTS = [
    ("pelvis",   "chest",    "spine.004"),
    ("pelvis",   "HR_thigh", "thigh.R"),
    ("pelvis",   "HL_thigh", "thigh.L"),
    ("chest",    "neck",     "spine.006"),
    ("chest",    "FR_thigh", "front_thigh.R"),
    ("chest",    "FL_thigh", "front_thigh.L"),
    ("neck",     "skull",    "scull"),
    ("HR_thigh", "HR_shin",  "shin.R"),
    ("HR_shin",  "HR_foot",  "foot.R"),
    ("HL_thigh", "HL_shin",  "shin.L"),
    ("HL_shin",  "HL_foot",  "foot.L"),
    ("FR_thigh", "FR_shin",  "front_shin.R"),
    ("FR_shin",  "FR_foot",  "front_foot.R"),
    ("FL_thigh", "FL_shin",  "front_shin.L"),
    ("FL_shin",  "FL_foot",  "front_foot.L"),
]

# In world space the deer lies along -Y. After Rx(-90) spawn:
#   world X -> Isaac X (left/right)
#   world Y -> Isaac -Z (goes down, i.e. legs hang down correctly -- wait)
#   world -Y -> Isaac Z (up)
#   world Z -> Isaac Y (forward)
# Knee bending: thigh points in (-Y, -Z) world = after rotation (-Z, +Y) isaac
# The primary bending axis in world space is X (left-right), which stays X after Rx rotation.
# So joint axis xyz="1 0 0" is still correct for knee/hip flexion.

# (lower_rad, upper_rad, effort, velocity)
# Front knees bend FORWARD only (0 to +1.8) — anatomically correct for ungulates.
# Hind knees bend BACKWARD only (-1.8 to 0) — like a dog's hock joint.
JOINT_LIMITS = {
    "pelvis_to_chest":    (-0.2,  0.2, 120, 8),
    "pelvis_to_HR_thigh": (-0.8,  0.8, 100, 10),
    "pelvis_to_HL_thigh": (-0.8,  0.8, 100, 10),
    "chest_to_neck":      (-0.3,  0.5,  50,  8),
    "chest_to_FR_thigh":  (-0.8,  0.8, 100, 10),
    "chest_to_FL_thigh":  (-0.8,  0.8, 100, 10),
    "neck_to_skull":      (-0.4,  0.6,  30,  8),
    "HR_thigh_to_HR_shin":(-1.8,  0.0,  80, 10),  # hind knee: backward only
    "HR_shin_to_HR_foot": (-0.5,  0.5,  60, 10),
    "HL_thigh_to_HL_shin":(-1.8,  0.0,  80, 10),  # hind knee: backward only
    "HL_shin_to_HL_foot": (-0.5,  0.5,  60, 10),
    "FR_thigh_to_FR_shin":( 0.0,  1.8,  80, 10),  # front knee: forward only
    "FR_shin_to_FR_foot": (-0.5,  0.5,  60, 10),
    "FL_thigh_to_FL_shin":( 0.0,  1.8,  80, 10),  # front knee: forward only
    "FL_shin_to_FL_foot": (-0.5,  0.5,  60, 10),
}

# Anatomically realistic masses (~73 kg doe).
# Chest > pelvis because it carries ribcage, withers, and supports neck+skull far forward.
# Front legs heavier (60% weight-bearing). CoM validated to sit within 1cm of foot centroid.
MASSES = {
    "pelvis": 18.0, "chest": 28.0, "neck": 6.0, "skull": 5.0,
    "HR_thigh": 1.2, "HR_shin": 1.0, "HR_foot": 0.8,
    "HL_thigh": 1.2, "HL_shin": 1.0, "HL_foot": 0.8,
    "FR_thigh": 2.5, "FR_shin": 1.5, "FR_foot": 1.0,
    "FL_thigh": 2.5, "FL_shin": 1.5, "FL_foot": 1.0,
}

print("\n" + "=" * 70)
print("WORLD-SPACE URDF EXPORT")
print("=" * 70)

arm_obj = mesh_obj = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        arm_obj = obj
    elif obj.type == 'MESH':
        mesh_obj = obj

armature = arm_obj.data

# --- Bone world positions ---
print("\n--- Bone positions in WORLD space ---")
bone_world_pos = {}
for bone in armature.bones:
    wp = arm_obj.matrix_world @ bone.head_local
    bone_world_pos[bone.name] = wp.copy()

for name in ["Root", "spine.003", "thigh.R", "shin.R", "foot.R", "toe.R",
             "front_thigh.R", "front_shin.R", "front_foot.R", "front_toe.R"]:
    if name in bone_world_pos:
        p = bone_world_pos[name]
        print(f"  {name:20s}: ({p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f})")

# --- Get evaluated mesh in world space ---
depsgraph = bpy.context.evaluated_depsgraph_get()
mesh_eval = mesh_obj.evaluated_get(depsgraph)
mesh_data = mesh_eval.to_mesh()

world_verts = [mesh_obj.matrix_world @ v.co for v in mesh_data.vertices]
world_normals = [(mesh_obj.matrix_world.to_3x3() @ v.normal).normalized() for v in mesh_data.vertices]

xs = [v.x for v in world_verts]; ys = [v.y for v in world_verts]; zs = [v.z for v in world_verts]
print(f"\nMesh world bounds: X=[{min(xs):.3f},{max(xs):.3f}] Y=[{min(ys):.3f},{max(ys):.3f}] Z=[{min(zs):.3f},{max(zs):.3f}]")

# --- Vertex group assignment ---
vg_names = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
vert_groups = {}
for v in mesh_data.vertices:
    best_group = None
    best_weight = -1
    for g in v.groups:
        if g.weight > best_weight:
            best_weight = g.weight
            gname = vg_names.get(g.group)
            if gname:
                best_group = gname
    if best_group:
        vert_groups[v.index] = best_group

# --- STL writer ---
def write_stl(filepath, tris):
    with open(filepath, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(tris)))
        for n, v1, v2, v3 in tris:
            f.write(struct.pack('<12fH', n[0], n[1], n[2],
                v1[0], v1[1], v1[2], v2[0], v2[1], v2[2],
                v3[0], v3[1], v3[2], 0))

polys = mesh_data.polygons
loops = mesh_data.loops

print("\n--- Exporting per-link STL files (world space, link-local) ---")
link_world_pos = {}

for link_name, bone_names in LINK_BONE_MAP.items():
    tris = []
    for poly in polys:
        loop_idx = list(poly.loop_indices)
        vis = [loops[li].vertex_index for li in loop_idx]
        link_verts = sum(1 for vi in vis if vert_groups.get(vi) in bone_names)
        if link_verts <= len(vis) // 2:
            continue
        n = world_normals[vis[0]]
        for i in range(1, len(vis) - 1):
            tris.append((
                (n.x, n.y, n.z),
                tuple(world_verts[vis[0]]),
                tuple(world_verts[vis[i]]),
                tuple(world_verts[vis[i+1]]),
            ))

    if not tris:
        print(f"  WARNING: {link_name} has 0 triangles!")
        continue

    # Link origin = primary bone world position
    lpos = bone_world_pos[bone_names[0]]
    link_world_pos[link_name] = lpos

    # Make link-local (subtract link world position)
    local_tris = [(n,
        (v1[0]-lpos.x, v1[1]-lpos.y, v1[2]-lpos.z),
        (v2[0]-lpos.x, v2[1]-lpos.y, v2[2]-lpos.z),
        (v3[0]-lpos.x, v3[1]-lpos.y, v3[2]-lpos.z))
        for n, v1, v2, v3 in tris]

    vis_path = os.path.join(OUT_DIR, f"{link_name}_visual.stl")
    col_path = os.path.join(OUT_DIR, f"{link_name}_collision.stl")
    write_stl(vis_path, local_tris)
    write_stl(col_path, local_tris)

    all_z = [c for _, v1, v2, v3 in local_tris for c in [v1[2], v2[2], v3[2]]]
    print(f"  {link_name:12s}: {len(local_tris):4d} tris, origin=({lpos.x:+.3f},{lpos.y:+.3f},{lpos.z:+.3f})")

# --- Compute joint offsets (world space, relative to parent link origin) ---
print("\n--- Joint offsets (world space, relative to parent link) ---")
joint_data = []
for parent_link, child_link, joint_bone in JOINTS:
    if parent_link not in link_world_pos or joint_bone not in bone_world_pos:
        print(f"  SKIP: {parent_link} -> {child_link} (missing data)")
        continue
    joint_world = bone_world_pos[joint_bone]
    parent_world = link_world_pos[parent_link]
    rel = joint_world - parent_world
    print(f"  {parent_link:12s} -> {child_link:12s}: ({rel.x:+.4f}, {rel.y:+.4f}, {rel.z:+.4f})")
    joint_data.append((parent_link, child_link, rel))

# --- Write URDF ---
print(f"\n--- Writing URDF: {URDF_PATH} ---")

def simple_inertia(mass, r=0.15):
    return 0.4 * mass * r * r

lines = [
    '<?xml version="1.0"?>',
    '<!-- deer_world.urdf: world-space export from deer_standing.blend',
    '     Spawn rotation in Isaac Sim: w=0.7071, x=-0.7071, y=0, z=0 (Rx -90deg)',
    '     This rotates the lying-flat world-space model to stand upright (Z-up). -->',
    '<robot name="deer">',
]

for link_name in LINK_BONE_MAP:
    if link_name not in link_world_pos:
        continue
    mass = MASSES.get(link_name, 1.0)
    ii = simple_inertia(mass)
    lines += [
        f'  <link name="{link_name}">',
        f'    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.1f}"/>',
        f'      <inertia ixx="{ii:.6f}" ixy="0" ixz="0" iyy="{ii:.6f}" iyz="0" izz="{ii:.6f}"/></inertial>',
        f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="meshes/world/{link_name}_collision.stl"/></geometry></collision>',
        f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="meshes/world/{link_name}_visual.stl"/></geometry></visual>',
        f'  </link>',
    ]

for parent_link, child_link, rel in joint_data:
    jname = f"{parent_link}_to_{child_link}"
    lim = JOINT_LIMITS.get(jname, (-1.5708, 1.5708, 80, 10))
    lines += [
        f'  <joint name="{jname}" type="revolute">',
        f'    <parent link="{parent_link}"/><child link="{child_link}"/>',
        f'    <origin xyz="{rel.x:.6f} {rel.y:.6f} {rel.z:.6f}" rpy="0 0 0"/>',
        f'    <axis xyz="1 0 0"/>',
        f'    <limit lower="{lim[0]:.4f}" upper="{lim[1]:.4f}" effort="{lim[2]:.1f}" velocity="{lim[3]:.1f}"/>',
        f'  </joint>',
    ]

lines.append('</robot>')

with open(URDF_PATH, 'w') as f:
    f.write('\n'.join(lines) + '\n')

mesh_eval.to_mesh_clear()

print(f"DONE.")
print(f"URDF: {URDF_PATH}")
print(f"Meshes: {OUT_DIR}")
print(f"Spawn with: ANIML_DEER_SPAWN_ROT=0.7071,-0.7071,0,0  ANIML_DEER_SPAWN_Z=1.05")
