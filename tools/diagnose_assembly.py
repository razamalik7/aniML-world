"""Diagnose whether the deer assembles correctly at different spawn rotations.

Computes world positions of all links via FK, and checks mesh bounding boxes
to see if meshes form a coherent deer shape.

Run: python tools/diagnose_assembly.py
"""
import math
import struct
from pathlib import Path

MESH_DIR = Path(r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\meshes")

JOINTS = [
    ("pelvis", None, [0, 0, 0]),
    ("chest", "pelvis", [0, -0.965180, 0.511591]),
    ("neck", "chest", [0, -0.024067, -0.849597]),
    ("skull", "neck", [0, -0.368394, -0.350125]),
    ("HR_thigh", "pelvis", [-0.127009, -0.842842, 0.456906]),
    ("HR_shin", "HR_thigh", [-0.000001, 0.238963, -0.232403]),
    ("HR_foot", "HR_shin", [-0.008380, 0.168601, 0.217347]),
    ("HL_thigh", "pelvis", [0.127008, -0.842841, 0.456906]),
    ("HL_shin", "HL_thigh", [-0.000001, 0.238963, -0.232403]),
    ("HL_foot", "HL_shin", [0.008379, 0.168600, 0.217347]),
    ("FR_thigh", "chest", [-0.122245, 0.187920, -0.889417]),
    ("FR_shin", "FR_thigh", [0.000050, 0.141754, 0.179375]),
    ("FR_foot", "FR_shin", [0.001356, 0.269873, -0.095854]),
    ("FL_thigh", "chest", [0.122205, 0.187955, -0.889340]),
    ("FL_shin", "FL_thigh", [-0.000067, 0.141706, 0.179413]),
    ("FL_foot", "FL_shin", [-0.000251, 0.268422, -0.099855]),
]


def rx_mat(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def mat_vec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def read_stl_bbox(path):
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        mins = [1e9] * 3
        maxs = [-1e9] * 3
        for _ in range(n):
            f.read(12)
            for _ in range(3):
                vx, vy, vz = struct.unpack("<fff", f.read(12))
                for i, v in enumerate((vx, vy, vz)):
                    mins[i] = min(mins[i], v)
                    maxs[i] = max(maxs[i], v)
            f.read(2)
    return [(mins[i] + maxs[i]) / 2 for i in range(3)]


def fk(root_deg):
    R = rx_mat(root_deg)
    pos = {}
    for name, parent, offset in JOINTS:
        world_offset = mat_vec(R, offset)
        if parent is None:
            pos[name] = world_offset
        else:
            pos[name] = [pos[parent][i] + world_offset[i] for i in range(3)]
    return pos


print("=" * 70)
print("MESH BOUNDING BOX CENTERS (after STL rotation)")
print("=" * 70)
for name, _, _ in JOINTS:
    stl = MESH_DIR / f"{name}_collision.stl"
    if stl.exists():
        c = read_stl_bbox(stl)
        print(f"  {name:12s}: ({c[0]:7.3f}, {c[1]:7.3f}, {c[2]:7.3f})")

print()

for angle in [0, -91, 90, 160, 180, 225]:
    pos = fk(angle)
    print("=" * 70)
    print(f"ROOT Rx = {angle}°")
    print("=" * 70)

    foot_names = ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]
    feet_z = [pos[f][2] for f in foot_names]
    pelvis_z = pos["pelvis"][2]
    skull_z = pos["skull"][2]

    print(f"  {'Link':12s}  {'X':>7s}  {'Y':>7s}  {'Z':>7s}")
    for name, _, _ in JOINTS:
        p = pos[name]
        marker = ""
        if name in foot_names:
            marker = " <-- foot"
        elif name == "pelvis":
            marker = " <-- root"
        elif name == "skull":
            marker = " <-- head"
        print(f"  {name:12s}  {p[0]:7.3f}  {p[1]:7.3f}  {p[2]:7.3f}{marker}")

    standing = (min(feet_z) < pelvis_z < skull_z
                and pelvis_z - min(feet_z) > 0.2
                and skull_z - pelvis_z > 0.1)
    print(f"\n  Feet Z range: {min(feet_z):.3f} to {max(feet_z):.3f}")
    print(f"  Pelvis Z: {pelvis_z:.3f}, Skull Z: {skull_z:.3f}")
    print(f"  STANDING: {'YES' if standing else 'NO'}")
    print()
