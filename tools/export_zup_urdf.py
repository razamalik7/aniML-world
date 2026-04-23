"""
Export deer_zup.urdf: all geometry pre-rotated into Isaac's Z-up standing frame.

Transform applied to every vertex and joint position:
  Rx(-90deg): world(x, y, z) -> isaac(x, -z, y)
  
Result: in the exported URDF, the deer already stands upright in Z-up.
- Pelvis at origin
- Feet at Z = -(original Y) = -(−0.93) wait...

World-space (Blender): deer lies along -Y. Pelvis at Y=-1.02, toes at Y=-0.09.
Rx(-90): (x,y,z) -> (x, -z, y)
  pelvis: (0, -1.02, 0.29) -> (0, -0.29, -1.02)  -- pelvis goes to -Z. Wrong, that's down.

Let's try Rx(+90): (x,y,z) -> (x, z, -y)
  pelvis: (0, -1.02, 0.29) -> (0, 0.29, 1.02)   -- pelvis at Z=1.02. UP! Correct.
  toe.R:  (-0.127, -0.09, 0.39) -> (-0.127, 0.39, 0.09)  -- toe at Z=0.09. Near ground!
  skull:  (0, -1.38, -0.65) -> (0, -0.65, 1.38)  -- skull at Z=1.38. UP! Correct.

Rx(+90) quaternion: w=cos(45deg)=0.7071, x=sin(45deg)=0.7071, y=0, z=0

So: apply Rx(+90) to all coords when writing the URDF.
Spawn: identity rotation, Z = pelvis_isaac_Z - toe_isaac_Z = 1.02 - 0.09 = 0.93
       Add small margin: SPAWN_Z = 0.95
"""
import bpy, struct, os, math
from mathutils import Vector, Matrix

OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "deer_urdf", "meshes", "zup")
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "deer_urdf", "deer_zup.urdf")
os.makedirs(OUT_DIR, exist_ok=True)

def rx90(v):
    """Rx(+90deg): (x,y,z) -> (x, z, -y)"""
    return (v[0], v[2], -v[1])

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
print("Z-UP URDF EXPORT  (Rx+90 pre-applied, identity spawn)")
print("="*70)

arm_obj = mesh_obj = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE': arm_obj = obj
    elif obj.type == 'MESH':   mesh_obj = obj

armature = arm_obj.data

# World-space bone positions, then rotate to Z-up Isaac frame
bone_w = {}
bone_isaac = {}
for bone in armature.bones:
    w = arm_obj.matrix_world @ bone.head_local
    bone_w[bone.name]     = w
    bone_isaac[bone.name] = rx90((w.x, w.y, w.z))

# Verify key positions
print("\nKey bone positions in Isaac Z-up frame (after Rx+90):")
for name in ["spine.003", "toe.R", "front_toe.R", "scull"]:
    if name in bone_isaac:
        p = bone_isaac[name]
        print(f"  {name:20s}: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

pelvis_z = bone_isaac["spine.003"][2]
toe_z    = bone_isaac["toe.R"][2]
print(f"\nPelvis Isaac Z = {pelvis_z:.4f}")
print(f"Toe    Isaac Z = {toe_z:.4f}")
print(f"SPAWN_Z = {pelvis_z - toe_z + 0.02:.4f}  (pelvis height above ground + 2cm margin)")

# Mesh
depsgraph = bpy.context.evaluated_depsgraph_get()
mesh_eval  = mesh_obj.evaluated_get(depsgraph)
mesh_data  = mesh_eval.to_mesh()
world_verts   = [mesh_obj.matrix_world @ v.co for v in mesh_data.vertices]
world_normals = [(mesh_obj.matrix_world.to_3x3() @ v.normal).normalized() for v in mesh_data.vertices]

# Apply Rx+90 to all vertices
isaac_verts   = [rx90((v.x, v.y, v.z)) for v in world_verts]
isaac_normals = [rx90((n.x, n.y, n.z)) for n in world_normals]

# Vertex group assignment
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

print("\n--- Exporting per-link STLs (Isaac Z-up frame) ---")
link_isaac_pos = {}

for link, bones in LINK_BONE_MAP.items():
    tris = []
    for poly in polys:
        lix  = list(poly.loop_indices)
        vis  = [loops[li].vertex_index for li in lix]
        hits = sum(1 for vi in vis if vert_group.get(vi) in bones)
        if hits <= len(vis)//2: continue
        n = isaac_normals[vis[0]]
        for i in range(1, len(vis)-1):
            tris.append((n,
                isaac_verts[vis[0]],
                isaac_verts[vis[i]],
                isaac_verts[vis[i+1]]))

    if not tris:
        print(f"  WARNING: {link} 0 tris"); continue

    lpos = bone_isaac[bones[0]]
    link_isaac_pos[link] = lpos

    # Make link-local
    local = [(n,
        (v1[0]-lpos[0], v1[1]-lpos[1], v1[2]-lpos[2]),
        (v2[0]-lpos[0], v2[1]-lpos[1], v2[2]-lpos[2]),
        (v3[0]-lpos[0], v3[1]-lpos[1], v3[2]-lpos[2]))
        for n,v1,v2,v3 in tris]

    write_stl(os.path.join(OUT_DIR, f"{link}_visual.stl"),    local)
    write_stl(os.path.join(OUT_DIR, f"{link}_collision.stl"), local)
    print(f"  {link:12s}: {len(local):4d} tris  isaac_pos=({lpos[0]:+.3f},{lpos[1]:+.3f},{lpos[2]:+.3f})")

# Joint offsets
print("\n--- Joint offsets (Isaac Z-up frame) ---")
joint_data = []
for parent, child, jbone in JOINTS:
    if parent not in link_isaac_pos or jbone not in bone_isaac:
        print(f"  SKIP {parent}->{child}"); continue
    jp = bone_isaac[jbone]
    pp = link_isaac_pos[parent]
    rel = (jp[0]-pp[0], jp[1]-pp[1], jp[2]-pp[2])
    print(f"  {parent:12s}->{child:12s}: ({rel[0]:+.4f},{rel[1]:+.4f},{rel[2]:+.4f})")
    joint_data.append((parent, child, rel))

# Write URDF
print(f"\n--- Writing URDF ---")

def inertia(mass, r=0.15): return 0.4*mass*r*r

lines = [
    '<?xml version="1.0"?>',
    '<!-- deer_zup.urdf: geometry pre-rotated into Isaac Z-up standing frame.',
    '     joint_pos=0 = natural standing. SPAWN_Z=0.95, SPAWN_ROT=1,0,0,0.',
    '     No root rotation needed anywhere. -->',
    '<robot name="deer">',
]

for link in LINK_BONE_MAP:
    if link not in link_isaac_pos: continue
    m  = MASSES.get(link, 1.0)
    ii = inertia(m)
    lines += [
        f'  <link name="{link}">',
        f'    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{m:.1f}"/>',
        f'      <inertia ixx="{ii:.6f}" ixy="0" ixz="0" iyy="{ii:.6f}" iyz="0" izz="{ii:.6f}"/></inertial>',
        f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
        f'<geometry><mesh filename="meshes/zup/{link}_collision.stl"/></geometry></collision>',
        f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
        f'<geometry><mesh filename="meshes/zup/{link}_visual.stl"/></geometry></visual>',
        f'  </link>',
    ]

for parent, child, rel in joint_data:
    jname = f"{parent}_to_{child}"
    lo, hi, eff, vel = JOINT_LIMITS.get(jname, (-1.0, 1.0, 80, 10))
    lines += [
        f'  <joint name="{jname}" type="revolute">',
        f'    <parent link="{parent}"/><child link="{child}"/>',
        f'    <origin xyz="{rel[0]:.6f} {rel[1]:.6f} {rel[2]:.6f}" rpy="0 0 0"/>',
        f'    <axis xyz="1 0 0"/>',
        f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="{eff:.1f}" velocity="{vel:.1f}"/>',
        f'  </joint>',
    ]

lines.append('</robot>')
with open(URDF_PATH, 'w') as f:
    f.write('\n'.join(lines)+'\n')

mesh_eval.to_mesh_clear()
print("DONE.")
print(f"URDF:   {URDF_PATH}")
print(f"Meshes: {OUT_DIR}")
print(f"Spawn:  ANIML_DEER_SPAWN_Z={pelvis_z - toe_z + 0.02:.2f}  ANIML_DEER_SPAWN_ROT=1,0,0,0")
