"""
Create rotation-only visual meshes in a SEPARATE directory.
These have Rx(-91deg) rotation applied but NO per-link position offset,
so they stay attached to their links during joint rotation.

The collision meshes in meshes/standing/ are NOT touched.
"""
import struct
import os

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

BASE = os.path.join(os.path.dirname(__file__), "..", "assets", "deer_urdf", "meshes")
OUT = os.path.join(BASE, "visual_rotonly")
os.makedirs(OUT, exist_ok=True)

LINKS = [
    "pelvis", "chest", "neck", "skull",
    "HR_thigh", "HR_shin", "HR_foot",
    "HL_thigh", "HL_shin", "HL_foot",
    "FR_thigh", "FR_shin", "FR_foot",
    "FL_thigh", "FL_shin", "FL_foot",
]

print("=== Creating rotation-only visual meshes (no per-link offset) ===\n")
for link in LINKS:
    src = os.path.join(BASE, f"{link}_visual.stl")
    if not os.path.exists(src):
        src = os.path.join(BASE, f"{link}_collision.stl")
    if not os.path.exists(src):
        print(f"  SKIP {link}")
        continue
    header, tris = read_stl(src)
    new_tris = []
    for n, v1, v2, v3, attr in tris:
        new_tris.append((
            mat_vec(R_VISUAL, n),
            mat_vec(R_VISUAL, v1),
            mat_vec(R_VISUAL, v2),
            mat_vec(R_VISUAL, v3),
            attr
        ))
    dst = os.path.join(OUT, f"{link}_visual.stl")
    write_stl(dst, header, new_tris)
    print(f"  {link}_visual.stl ({len(new_tris)} tris)")

print(f"\nOutput: {OUT}")
print("These meshes have ROTATION ONLY - no per-link offset.")
print("Collision meshes in meshes/standing/ are UNTOUCHED.")
