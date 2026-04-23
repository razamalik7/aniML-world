"""
Fix visual mesh drift during animation.

Problem: The standing meshes have per-link position offsets baked in. These
offsets rotate when joints rotate, causing visual meshes to float away from
their links during animation.

Fix: Create visual meshes with ROTATION ONLY (Rx(-91deg)) - no per-link
position offset. This means each link's visual mesh will be correctly
ORIENTED but centered at the deer_rotated link position. There will be
small gaps between segments (inherent to the articulated model with different
joint frames), but no floating/drifting panels.

Collision meshes are left unchanged (they have the full transform and the
trained policy depends on them for correct physics contact).
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
        triangles = []
        for _ in range(num_tri):
            data = struct.unpack("<12fH", f.read(50))
            triangles.append((data[0:3], data[3:6], data[6:9], data[9:12], data[12]))
        return header, triangles

def write_stl(path, header, triangles):
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for n, v1, v2, v3, attr in triangles:
            f.write(struct.pack("<12fH", *n, *v1, *v2, *v3, attr))


MESH_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "deer_urdf", "meshes")
OUT_DIR = os.path.join(MESH_DIR, "standing")

LINKS = [
    "pelvis", "chest", "neck", "skull",
    "HR_thigh", "HR_shin", "HR_foot",
    "HL_thigh", "HL_shin", "HL_foot",
    "FR_thigh", "FR_shin", "FR_foot",
    "FL_thigh", "FL_shin", "FL_foot",
]

print("=== Creating rotation-only visual meshes ===\n")
for link in LINKS:
    src = os.path.join(MESH_DIR, f"{link}_visual.stl")
    if not os.path.exists(src):
        src = os.path.join(MESH_DIR, f"{link}_collision.stl")
    if not os.path.exists(src):
        print(f"  SKIP {link} (no source mesh)")
        continue

    dst = os.path.join(OUT_DIR, f"{link}_visual.stl")
    header, triangles = read_stl(src)
    new_tris = []
    for normal, v1, v2, v3, attr in triangles:
        nn = mat_vec(R_VISUAL, normal)
        nv1 = mat_vec(R_VISUAL, v1)
        nv2 = mat_vec(R_VISUAL, v2)
        nv3 = mat_vec(R_VISUAL, v3)
        new_tris.append((nn, nv1, nv2, nv3, attr))
    write_stl(dst, header, new_tris)
    print(f"  {link}_visual.stl -> rotation only ({len(new_tris)} tris)")

print("\nDone. Visual meshes now have rotation only, no per-link offset.")
print("Collision meshes are UNCHANGED (physics preserved).")
