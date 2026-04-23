"""
Export a clean, consistent URDF from deer_standing.blend.

Key insight: In the armature's LOCAL space, the deer is already standing
with Z-up, feet near Z=0. The armature Rx(90°) and mesh Rx(88.53°) are
OBJECT-LEVEL rotations we should IGNORE — we want the local-space data.

Steps:
1. Use bone.head_local for joint positions (armature local space)
2. Transform mesh vertices into armature local space
3. Split mesh by vertex groups
4. Write clean URDF — no root rotation needed in Isaac Sim
"""
import bpy
import struct
import os
import math
from mathutils import Matrix

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "deer_urdf", "meshes", "clean")
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "deer_urdf", "deer_clean.urdf")
os.makedirs(OUT_DIR, exist_ok=True)

LINK_BONE_MAP = {
    "pelvis": ["spine.003", "shoulder.R", "shoulder.L"],
    "chest": ["spine.004", "spine.005", "front_shoulder.R", "front_shoulder.L"],
    "neck": ["spine.006", "spine.007", "spine.008"],
    "skull": ["scull", "ear.R", "ear.L"],
    "HR_thigh": ["thigh.R"],
    "HR_shin": ["shin.R"],
    "HR_foot": ["foot.R", "toe.R"],
    "HL_thigh": ["thigh.L"],
    "HL_shin": ["shin.L"],
    "HL_foot": ["foot.L", "toe.L"],
    "FR_thigh": ["front_thigh.R"],
    "FR_shin": ["front_shin.R"],
    "FR_foot": ["front_foot.R", "front_toe.R"],
    "FL_thigh": ["front_thigh.L"],
    "FL_shin": ["front_shin.L"],
    "FL_foot": ["front_foot.L", "front_toe.L"],
}

JOINTS = [
    ("pelvis", "chest", "spine.004"),
    ("pelvis", "HR_thigh", "thigh.R"),
    ("pelvis", "HL_thigh", "thigh.L"),
    ("chest", "neck", "spine.006"),
    ("chest", "FR_thigh", "front_thigh.R"),
    ("chest", "FL_thigh", "front_thigh.L"),
    ("neck", "skull", "scull"),
    ("HR_thigh", "HR_shin", "shin.R"),
    ("HR_shin", "HR_foot", "foot.R"),
    ("HL_thigh", "HL_shin", "shin.L"),
    ("HL_shin", "HL_foot", "foot.L"),
    ("FR_thigh", "FR_shin", "front_shin.R"),
    ("FR_shin", "FR_foot", "front_foot.R"),
    ("FL_thigh", "FL_shin", "front_shin.L"),
    ("FL_shin", "FL_foot", "front_foot.L"),
]

JOINT_LIMITS = {
    "pelvis_to_chest": (-0.5236, 0.5236, 120, 8),
    "pelvis_to_HR_thigh": (-1.5708, 1.5708, 100, 10),
    "pelvis_to_HL_thigh": (-1.5708, 1.5708, 100, 10),
    "chest_to_neck": (-0.6981, 0.6981, 50, 8),
    "chest_to_FR_thigh": (-1.5708, 1.5708, 100, 10),
    "chest_to_FL_thigh": (-1.5708, 1.5708, 100, 10),
    "neck_to_skull": (-0.5236, 0.8727, 30, 8),
    "HR_thigh_to_HR_shin": (-2.0944, 0.1745, 80, 10),
    "HR_shin_to_HR_foot": (-0.7854, 0.7854, 60, 10),
    "HL_thigh_to_HL_shin": (-2.0944, 0.1745, 80, 10),
    "HL_shin_to_HL_foot": (-0.7854, 0.7854, 60, 10),
    "FR_thigh_to_FR_shin": (-2.0944, 0.1745, 80, 10),
    "FR_shin_to_FR_foot": (-0.7854, 0.7854, 60, 10),
    "FL_thigh_to_FL_shin": (-2.0944, 0.1745, 80, 10),
    "FL_shin_to_FL_foot": (-0.7854, 0.7854, 60, 10),
}

MASSES = {
    "pelvis": 25.0, "chest": 22.0, "neck": 3.5, "skull": 2.5,
    "HR_thigh": 1.5, "HR_shin": 1.5, "HR_foot": 1.2,
    "HL_thigh": 1.5, "HL_shin": 1.5, "HL_foot": 1.2,
    "FR_thigh": 2.0, "FR_shin": 1.2, "FR_foot": 1.2,
    "FL_thigh": 2.0, "FL_shin": 1.2, "FL_foot": 1.2,
}

print("\n" + "=" * 70)
print("CLEAN URDF EXPORT (ARMATURE LOCAL SPACE)")
print("=" * 70)

arm_obj = mesh_obj = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        arm_obj = obj
    elif obj.type == 'MESH':
        mesh_obj = obj

print(f"Armature: {arm_obj.name}, rotation: {[math.degrees(r) for r in arm_obj.rotation_euler]}")
print(f"Mesh: {mesh_obj.name}, rotation: {[math.degrees(r) for r in mesh_obj.rotation_euler]}")

# --- Bone positions in armature LOCAL space (Z-up, deer standing) ---
armature = arm_obj.data
bone_local_pos = {}
print("\n--- Bone positions (armature local space, Z-up) ---")
for bone in armature.bones:
    pos = bone.head_local
    bone_local_pos[bone.name] = pos.copy()
    z_tag = " <-- GROUND" if pos.z < 0.15 else ""
    print(f"  {bone.name:25s}: ({pos.x:+.4f}, {pos.y:+.4f}, {pos.z:+.4f}){z_tag}")

# Check: feet should be near Z=0
for name in ["toe.R", "toe.L", "front_toe.R", "front_toe.L"]:
    z = bone_local_pos[name].z
    print(f"  ** {name} Z = {z:.3f} {'(near ground OK)' if abs(z) < 0.2 else 'WARNING: not near ground!'}")

# --- Transform mesh vertices into armature local space ---
# mesh_world = arm_world * parent_inverse * mesh_local_transform
# We want: v_in_arm_local = arm_world_inv * mesh_world * v_mesh_local
arm_world_inv = arm_obj.matrix_world.inverted()
mesh_world_mat = mesh_obj.matrix_world

# Combined: mesh-local -> armature-local
to_arm_local = arm_world_inv @ mesh_world_mat
print(f"\nMesh-to-armature-local transform:")
for row in to_arm_local:
    print(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

# Get evaluated mesh
depsgraph = bpy.context.evaluated_depsgraph_get()
mesh_eval = mesh_obj.evaluated_get(depsgraph)
mesh_data = mesh_eval.to_mesh()

# Transform vertices to armature local space
arm_local_verts = [to_arm_local @ v.co for v in mesh_data.vertices]
arm_local_normals = [(to_arm_local.to_3x3() @ v.normal).normalized() for v in mesh_data.vertices]

xs = [v.x for v in arm_local_verts]
ys = [v.y for v in arm_local_verts]
zs = [v.z for v in arm_local_verts]
print(f"\nMesh in armature local space: {len(arm_local_verts)} verts")
print(f"  X: [{min(xs):.4f}, {max(xs):.4f}]  (width)")
print(f"  Y: [{min(ys):.4f}, {max(ys):.4f}]  (depth/forward)")
print(f"  Z: [{min(zs):.4f}, {max(zs):.4f}]  (height)")
print(f"  ** Lowest Z = {min(zs):.4f} (should be near 0 for feet on ground)")

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

# --- Export per-link STL ---
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

print("\n--- Exporting per-link STL files (armature local space) ---")
link_pos = {}

for link_name, bone_names in LINK_BONE_MAP.items():
    tris = []
    for poly in polys:
        loop_idx = list(poly.loop_indices)
        vis = [loops[li].vertex_index for li in loop_idx]
        link_verts = sum(1 for vi in vis if vert_groups.get(vi) in bone_names)
        if link_verts <= len(vis) // 2:
            continue
        n = to_arm_local.to_3x3() @ poly.normal
        for i in range(1, len(vis) - 1):
            tris.append((
                (n.x, n.y, n.z),
                tuple(arm_local_verts[vis[0]]),
                tuple(arm_local_verts[vis[i]]),
                tuple(arm_local_verts[vis[i+1]]),
            ))

    if not tris:
        print(f"  WARNING: {link_name} has 0 triangles!")
        continue

    # Link position = primary bone head in armature local
    lpos = bone_local_pos[bone_names[0]]
    link_pos[link_name] = lpos

    # Make link-local (subtract link position)
    local_tris = []
    for n, v1, v2, v3 in tris:
        local_tris.append((n,
            (v1[0]-lpos.x, v1[1]-lpos.y, v1[2]-lpos.z),
            (v2[0]-lpos.x, v2[1]-lpos.y, v2[2]-lpos.z),
            (v3[0]-lpos.x, v3[1]-lpos.y, v3[2]-lpos.z)))

    vis_path = os.path.join(OUT_DIR, f"{link_name}_visual.stl")
    col_path = os.path.join(OUT_DIR, f"{link_name}_collision.stl")
    write_stl(vis_path, local_tris)
    write_stl(col_path, local_tris)

    all_z = [c for _, v1, v2, v3 in local_tris for c in [v1[2], v2[2], v3[2]]]
    print(f"  {link_name:12s}: {len(local_tris):4d} tris, pos=({lpos.x:+.3f},{lpos.y:+.3f},{lpos.z:+.3f}), local_z=[{min(all_z):+.3f},{max(all_z):+.3f}]")

# --- Compute joints ---
print("\n--- Joint offsets (armature local, relative to parent) ---")
joint_data = []
for parent_link, child_link, joint_bone in JOINTS:
    joint_pos = bone_local_pos[joint_bone]
    parent_pos = link_pos[parent_link]
    rel = joint_pos - parent_pos
    print(f"  {parent_link:12s} -> {child_link:12s}: ({rel.x:+.6f}, {rel.y:+.6f}, {rel.z:+.6f})")
    joint_data.append((parent_link, child_link, rel))

# Verify: can legs reach the ground?
print("\n--- Leg reach analysis ---")
for side, prefix, shoulder_bone in [("Hind R", "", "shoulder.R"), ("Front R", "front_", "front_shoulder.R")]:
    chain = []
    if "front" in prefix:
        chain_bones = [f"{prefix}thigh.R", f"{prefix}shin.R", f"{prefix}foot.R", f"{prefix}toe.R"]
    else:
        chain_bones = ["thigh.R", "shin.R", "foot.R", "toe.R"]
    
    zs = [bone_local_pos[b].z for b in chain_bones if b in bone_local_pos]
    pelvis_z = bone_local_pos["spine.003"].z
    lowest_z = min(zs) if zs else 999
    print(f"  {side}: pelvis Z={pelvis_z:.3f}, lowest joint Z={lowest_z:.3f}, reaches ground={'YES' if lowest_z < 0.15 else 'NO'}")

# --- Write URDF ---
print(f"\n--- Writing URDF ---")

def simple_inertia(mass, r=0.15):
    i = 0.4 * mass * r * r
    return i

lines = ['<?xml version="1.0"?>',
'<!-- deer_clean.urdf: exported from deer_standing.blend armature local space.',
'     Mesh and skeleton consistent. Z-up. No root rotation needed. -->',
'<robot name="deer">']

for link_name in LINK_BONE_MAP:
    mass = MASSES.get(link_name, 1.0)
    ii = simple_inertia(mass)
    lines.append(f'  <link name="{link_name}">')
    lines.append(f'    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.1f}"/>')
    lines.append(f'      <inertia ixx="{ii:.6f}" ixy="0" ixz="0" iyy="{ii:.6f}" iyz="0" izz="{ii:.6f}"/></inertial>')
    lines.append(f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="meshes/clean/{link_name}_collision.stl"/></geometry></collision>')
    lines.append(f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="meshes/clean/{link_name}_visual.stl"/></geometry></visual>')
    lines.append(f'  </link>')

for parent_link, child_link, rel in joint_data:
    jname = f"{parent_link}_to_{child_link}"
    lim = JOINT_LIMITS.get(jname, (-1.5708, 1.5708, 80, 10))
    lines.append(f'  <joint name="{jname}" type="revolute">')
    lines.append(f'    <parent link="{parent_link}"/><child link="{child_link}"/>')
    lines.append(f'    <origin xyz="{rel.x:.6f} {rel.y:.6f} {rel.z:.6f}" rpy="0 0 0"/>')
    lines.append(f'    <axis xyz="1 0 0"/>')
    lines.append(f'    <limit lower="{lim[0]:.6f}" upper="{lim[1]:.6f}" effort="{lim[2]:.1f}" velocity="{lim[3]:.1f}"/>')
    lines.append(f'  </joint>')

lines.append('</robot>')

with open(URDF_PATH, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"URDF: {URDF_PATH}")
print(f"Meshes: {OUT_DIR}")
print("DONE. Use identity root rotation in Isaac Sim.")

mesh_eval.to_mesh_clear()
