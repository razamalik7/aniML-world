"""
Create PROPER link-local visual meshes for each link.

For each link:
  v_local = R_VISUAL * v_blender - FK_position

Where:
  R_VISUAL = Rx(-91deg) — rotates from Blender Y-up to Isaac Z-up
  v_blender = original mesh vertices from Blender (in deer.urdf frame)
  FK_position = link world position at rest, from deer_standing.urdf joint chain

The result is mesh vertices centered on the link's joint origin.
When attached to the skeleton, they move naturally with joints — no drift.
"""
import struct
import os
import xml.etree.ElementTree as ET
import math

BASE = os.path.join(os.path.dirname(__file__), "..", "assets", "deer_urdf")
MESH_DIR = os.path.join(BASE, "meshes")
OUT_DIR = os.path.join(MESH_DIR, "visual_local")
os.makedirs(OUT_DIR, exist_ok=True)


def quat_to_matrix(w, x, y, z):
    return [
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ]


def mat_vec(M, v):
    return tuple(sum(M[i][j] * v[j] for j in range(3)) for i in range(3))


R_VISUAL = quat_to_matrix(0.7018, -0.7124, 0.0, 0.0)


def read_stl(path):
    with open(path, "rb") as f:
        header = f.read(80)
        (num_tri,) = struct.unpack("<I", f.read(4))
        tris = []
        for _ in range(num_tri):
            data = struct.unpack("<12fH", f.read(50))
            tris.append((data[0:3], data[3:6], data[6:9], data[9:12], data[12]))
        return header, tris


def write_stl(path, header, tris):
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(tris)))
        for n, v1, v2, v3, attr in tris:
            f.write(struct.pack("<12fH", *n, *v1, *v2, *v3, attr))


# --- Compute FK positions from deer_standing.urdf ---

tree = ET.parse(os.path.join(BASE, "deer_standing.urdf"))
root = tree.getroot()

joints = {}
for joint in root.findall("joint"):
    name = joint.get("name")
    parent = joint.find("parent").get("link")
    child = joint.find("child").get("link")
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.get("xyz").split()]
    joints[name] = {"parent": parent, "child": child, "xyz": xyz}

parent_to_joints = {}
for jname, jdata in joints.items():
    parent_to_joints.setdefault(jdata["parent"], []).append((jname, jdata))

link_world_pos = {"pelvis": (0.0, 0.0, 0.0)}

def compute_fk(link_name):
    if link_name in link_world_pos:
        return link_world_pos[link_name]
    for jname, jdata in joints.items():
        if jdata["child"] == link_name:
            parent_pos = compute_fk(jdata["parent"])
            pos = tuple(parent_pos[i] + jdata["xyz"][i] for i in range(3))
            link_world_pos[link_name] = pos
            return pos
    raise ValueError(f"No joint leads to {link_name}")

LINKS = [
    "pelvis", "chest", "neck", "skull",
    "HR_thigh", "HR_shin", "HR_foot",
    "HL_thigh", "HL_shin", "HL_foot",
    "FR_thigh", "FR_shin", "FR_foot",
    "FL_thigh", "FL_shin", "FL_foot",
]

for link in LINKS:
    compute_fk(link)

print("=== FK positions (deer_standing.urdf, rest pose) ===")
for link in LINKS:
    p = link_world_pos[link]
    print(f"  {link:12s}: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

# --- Create link-local visual meshes ---

print("\n=== Creating link-local visual meshes ===\n")
for link in LINKS:
    fk_pos = link_world_pos[link]

    src = os.path.join(MESH_DIR, f"{link}_visual.stl")
    if not os.path.exists(src):
        src = os.path.join(MESH_DIR, f"{link}_collision.stl")
    if not os.path.exists(src):
        print(f"  SKIP {link}")
        continue

    header, tris = read_stl(src)
    new_tris = []
    for n, v1, v2, v3, attr in tris:
        nn = mat_vec(R_VISUAL, n)
        rv1 = mat_vec(R_VISUAL, v1)
        rv2 = mat_vec(R_VISUAL, v2)
        rv3 = mat_vec(R_VISUAL, v3)
        nv1 = tuple(rv1[i] - fk_pos[i] for i in range(3))
        nv2 = tuple(rv2[i] - fk_pos[i] for i in range(3))
        nv3 = tuple(rv3[i] - fk_pos[i] for i in range(3))
        new_tris.append((nn, nv1, nv2, nv3, attr))

    dst = os.path.join(OUT_DIR, f"{link}_visual.stl")
    write_stl(dst, header, new_tris)

    zs = []
    for _, v1, v2, v3, _ in new_tris:
        zs.extend([v1[2], v2[2], v3[2]])
    print(f"  {link:12s}: {len(new_tris)} tris, z=[{min(zs):+.3f}, {max(zs):+.3f}]  (should be centered near 0)")

print(f"\nOutput: {OUT_DIR}")
