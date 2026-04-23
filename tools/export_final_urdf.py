"""
Export deer_final.urdf where joint position 0 = natural standing pose.

Strategy:
- Use world-space bone positions for joint origins (importer will Rx90 them)
- For joint AXES: the natural bending axis for each joint in world space is X (left/right)
  After importer Rx90: world-X stays Isaac-X. So axis="1 0 0" is still correct.
- For joint LIMITS: symmetric around 0 (natural pose IS 0)
- Meshes: world-space vertices made link-local by subtracting bone world pos

This means: at joint_pos=0 the deer stands in its natural Blender rest pose.
The importer Rx90 stands it upright. Spawn Z = foot offset from pelvis after Rx90.

After Rx90:  world(x,y,z) -> isaac(x, z, -y)
Pelvis world Y = -1.02 -> isaac Z = 1.02  (pelvis height above ground origin)
Toe world Y   = -0.09 -> isaac Z = 0.09  (foot height)
Pelvis-to-foot drop in isaac Z = 1.02 - 0.09 = 0.93m
So SPAWN_Z = 0.93 + small margin ~ 0.95
"""
import bpy, struct, os, math
from mathutils import Vector

OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "deer_urdf", "meshes", "final")
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "deer_urdf", "deer_final.urdf")
os.makedirs(OUT_DIR, exist_ok=True)

# Primary bone = link origin. Bone must exist in armature.
LINK_BONE_MAP = {
    "pelvis":   ["spine.003", "shoulder.R", "shoulder.L"],
    "chest":    ["spine.004", "spine.005", "front_shoulder.R", "front_shoulder.L"],
    "neck":     ["spine.006", "spine.007", "spine.008"],
    "skull":    ["scull", "ear.R", "ear.L"],
    "HR_thigh": ["thigh.R"],
    "HR_shin":  ["shin.R"],
    "HR_foot":  ["foot.R", "toe.R"],
    "HL_thigh": ["thigh.L"],
    "HL_shin":  ["shin.L"],
    "HL_foot":  ["foot.L", "toe.L"],
    "FR_thigh": ["front_thigh.R"],
    "FR_shin":  ["front_shin.R"],
    "FR_foot":  ["front_foot.R", "front_toe.R"],
    "FL_thigh": ["front_thigh.L"],
    "FL_shin":  ["front_shin.L"],
    "FL_foot":  ["front_foot.L", "front_toe.L"],
}

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

# Limits are now RELATIVE to the natural standing pose (0 = standing)
JOINT_LIMITS = {
    "pelvis_to_chest":    (-0.30,  0.30, 120,  8),
    "pelvis_to_HR_thigh": (-0.80,  0.80, 100, 10),
    "HR_thigh_to_HR_shin":(-1.20,  0.60,  80, 10),
    "HR_shin_to_HR_foot": (-0.60,  0.60,  60, 10),
    "pelvis_to_HL_thigh": (-0.80,  0.80, 100, 10),
    "HL_thigh_to_HL_shin":(-1.20,  0.60,  80, 10),
    "HL_shin_to_HL_foot": (-0.60,  0.60,  60, 10),
    "chest_to_neck":      (-0.50,  0.50,  50,  8),
    "chest_to_FR_thigh":  (-0.80,  0.80, 100, 10),
    "FR_thigh_to_FR_shin":(-1.20,  0.60,  80, 10),
    "FR_shin_to_FR_foot": (-0.60,  0.60,  60, 10),
    "chest_to_FL_thigh":  (-0.80,  0.80, 100, 10),
    "FL_thigh_to_FL_shin":(-1.20,  0.60,  80, 10),
    "FL_shin_to_FL_foot": (-0.60,  0.60,  60, 10),
    "neck_to_skull":      (-0.50,  0.70,  30,  8),
}

MASSES = {
    "pelvis": 25.0, "chest": 22.0, "neck": 3.5, "skull": 2.5,
    "HR_thigh": 1.5, "HR_shin": 1.5, "HR_foot": 1.2,
    "HL_thigh": 1.5, "HL_shin": 1.5, "HL_foot": 1.2,
    "FR_thigh": 2.0, "FR_shin": 1.2, "FR_foot": 1.2,
    "FL_thigh": 2.0, "FL_shin": 1.2, "FL_foot": 1.2,
}

print("\n" + "="*70)
print("FINAL URDF EXPORT  (joint 0 = natural standing pose)")
print("="*70)

arm_obj = mesh_obj = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE': arm_obj = obj
    elif obj.type == 'MESH':   mesh_obj = obj

armature = arm_obj.data

# World-space bone positions
bone_w = {}
for bone in armature.bones:
    bone_w[bone.name] = arm_obj.matrix_world @ bone.head_local

# Report key positions
pelvis_wy = bone_w['spine.003'].y
toe_wy    = bone_w['toe.R'].y
print(f"Pelvis world Y={pelvis_wy:.4f} -> Isaac Z={-pelvis_wy:.4f}")
print(f"Toe    world Y={toe_wy:.4f}    -> Isaac Z={-toe_wy:.4f}")
print(f"Ideal SPAWN_Z = {-pelvis_wy - (-toe_wy):.4f} + margin")

# World-space mesh vertices
depsgraph  = bpy.context.evaluated_depsgraph_get()
mesh_eval  = mesh_obj.evaluated_get(depsgraph)
mesh_data  = mesh_eval.to_mesh()
world_verts   = [mesh_obj.matrix_world @ v.co for v in mesh_data.vertices]
world_normals = [(mesh_obj.matrix_world.to_3x3() @ v.normal).normalized()
                 for v in mesh_data.vertices]

# Vertex group -> link
vg_names   = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
vert_group = {}
for v in mesh_data.vertices:
    best, bw = None, -1
    for g in v.groups:
        if g.weight > bw:
            bw = g.weight
            n  = vg_names.get(g.group)
            if n: best = n
    if best: vert_group[v.index] = best

def write_stl(path, tris):
    with open(path, 'wb') as f:
        f.write(b'\0'*80)
        f.write(struct.pack('<I', len(tris)))
        for n, v1, v2, v3 in tris:
            f.write(struct.pack('<12fH',
                n[0],n[1],n[2],
                v1[0],v1[1],v1[2], v2[0],v2[1],v2[2], v3[0],v3[1],v3[2], 0))

polys = mesh_data.polygons
loops = mesh_data.loops

print("\n--- Exporting per-link STLs ---")
link_wpos = {}

for link, bones in LINK_BONE_MAP.items():
    tris = []
    for poly in polys:
        lix  = list(poly.loop_indices)
        vis  = [loops[li].vertex_index for li in lix]
        hits = sum(1 for vi in vis if vert_group.get(vi) in bones)
        if hits <= len(vis)//2: continue
        n = world_normals[vis[0]]
        for i in range(1, len(vis)-1):
            tris.append((
                (n.x, n.y, n.z),
                tuple(world_verts[vis[0]]),
                tuple(world_verts[vis[i]]),
                tuple(world_verts[vis[i+1]]),
            ))

    if not tris:
        print(f"  WARNING: {link} 0 tris"); continue

    lpos = bone_w[bones[0]]
    link_wpos[link] = lpos

    local = [(n,
        (v1[0]-lpos.x, v1[1]-lpos.y, v1[2]-lpos.z),
        (v2[0]-lpos.x, v2[1]-lpos.y, v2[2]-lpos.z),
        (v3[0]-lpos.x, v3[1]-lpos.y, v3[2]-lpos.z))
        for n,v1,v2,v3 in tris]

    write_stl(os.path.join(OUT_DIR, f"{link}_visual.stl"),    local)
    write_stl(os.path.join(OUT_DIR, f"{link}_collision.stl"), local)
    print(f"  {link:12s}: {len(local):4d} tris  origin_world=({lpos.x:+.3f},{lpos.y:+.3f},{lpos.z:+.3f})")

# Joint offsets in world space (parent link origin -> joint bone head)
print("\n--- Joint offsets (world space) ---")
joint_data = []
for parent, child, jbone in JOINTS:
    if parent not in link_wpos or jbone not in bone_w:
        print(f"  SKIP {parent}->{child}"); continue
    rel = bone_w[jbone] - link_wpos[parent]
    print(f"  {parent:12s}->{child:12s}: ({rel.x:+.4f},{rel.y:+.4f},{rel.z:+.4f})")
    joint_data.append((parent, child, rel))

# Write URDF
print(f"\n--- Writing {URDF_PATH} ---")

def inertia(mass, r=0.15): return 0.4*mass*r*r

lines = [
    '<?xml version="1.0"?>',
    '<!-- deer_final.urdf',
    '     joint_pos=0 means natural standing pose from Blender.',
    '     URDF importer applies Rx(90) -> deer stands in Isaac Z-up.',
    '     SPAWN_Z=0.95, SPAWN_ROT=1,0,0,0 -->',
    '<robot name="deer">',
]

for link in LINK_BONE_MAP:
    if link not in link_wpos: continue
    m  = MASSES.get(link, 1.0)
    ii = inertia(m)
    lines += [
        f'  <link name="{link}">',
        f'    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{m:.1f}"/>',
        f'      <inertia ixx="{ii:.6f}" ixy="0" ixz="0" iyy="{ii:.6f}" iyz="0" izz="{ii:.6f}"/></inertial>',
        f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry>'
        f'<mesh filename="meshes/final/{link}_collision.stl"/></geometry></collision>',
        f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry>'
        f'<mesh filename="meshes/final/{link}_visual.stl"/></geometry></visual>',
        f'  </link>',
    ]

for parent, child, rel in joint_data:
    jname = f"{parent}_to_{child}"
    lo, hi, eff, vel = JOINT_LIMITS.get(jname, (-1.0, 1.0, 80, 10))
    lines += [
        f'  <joint name="{jname}" type="revolute">',
        f'    <parent link="{parent}"/><child link="{child}"/>',
        f'    <origin xyz="{rel.x:.6f} {rel.y:.6f} {rel.z:.6f}" rpy="0 0 0"/>',
        f'    <axis xyz="1 0 0"/>',
        f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="{eff:.1f}" velocity="{vel:.1f}"/>',
        f'  </joint>',
    ]

lines.append('</robot>')
with open(URDF_PATH, 'w') as f:
    f.write('\n'.join(lines)+'\n')

mesh_eval.to_mesh_clear()
print("DONE.")
print(f"URDF  : {URDF_PATH}")
print(f"Meshes: {OUT_DIR}")
print("Spawn with: ANIML_DEER_SPAWN_Z=0.95  ANIML_DEER_SPAWN_ROT=1,0,0,0")
print("Default joint positions: ALL ZEROS (natural standing = 0)")
