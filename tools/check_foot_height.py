"""Check the lowest point of the deer mesh to determine correct spawn height.

Reads the standing STL files for all foot links and finds the minimum Z
coordinate across all vertices. The spawn height should be set so that
min_z + spawn_z >= 0 (feet at or above ground).
"""
import struct
import os

MESH_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "deer_urdf", "meshes", "standing")

# deer_rotated.urdf FK at zero angles (link world positions)
LINK_POS = {
    "pelvis":   (0, 0, 0),
    "chest":    (0, 0.732, -0.811),
    "neck":     (0, 1.045, -0.021),
    "skull":    (0, 1.511, 0.182),
    "HR_thigh": (-0.127, 0.636, -0.718),
    "HR_shin":  (-0.127, 0.491, -0.418),
    "HR_foot":  (-0.135, 0.258, -0.564),
    "HL_thigh": (0.127, 0.636, -0.718),
    "HL_shin":  (0.127, 0.491, -0.418),
    "HL_foot":  (0.135, 0.258, -0.564),
    "FR_thigh": (-0.122, 0.860, 0.089),
    "FR_shin":  (-0.122, 0.665, -0.031),
    "FR_foot":  (-0.121, 0.444, 0.151),
    "FL_thigh": (0.122, 0.860, 0.089),
    "FL_shin":  (0.122, 0.665, -0.031),
    "FL_foot":  (0.122, 0.444, 0.151),
}


def read_stl_z_range(path):
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        min_z, max_z = float("inf"), float("-inf")
        for _ in range(n):
            data = struct.unpack("<12fH", f.read(50))
            for vi in range(3):
                z = data[3 + vi * 3 + 2]
                min_z = min(min_z, z)
                max_z = max(max_z, z)
    return min_z, max_z


print("=== Foot height analysis (standing meshes, link-local) ===\n")
global_min_z = float("inf")
global_max_z = float("-inf")

for link in ["pelvis", "chest", "HR_foot", "HL_foot", "FR_foot", "FL_foot",
             "HR_shin", "HL_shin", "FR_shin", "FL_shin", "HR_thigh", "HL_thigh",
             "FR_thigh", "FL_thigh", "neck", "skull"]:
    col = os.path.join(MESH_DIR, f"{link}_collision.stl")
    if not os.path.exists(col):
        continue
    min_z, max_z = read_stl_z_range(col)
    lp = LINK_POS.get(link, (0, 0, 0))
    world_min_z = lp[2] + min_z
    world_max_z = lp[2] + max_z
    global_min_z = min(global_min_z, world_min_z)
    global_max_z = max(global_max_z, world_max_z)
    print(f"{link:12s}  link_z={lp[2]:+.3f}  mesh_z=[{min_z:+.3f}, {max_z:+.3f}]  "
          f"world_z=[{world_min_z:+.3f}, {world_max_z:+.3f}]")

print(f"\nGlobal Z range: [{global_min_z:+.3f}, {global_max_z:+.3f}]")
print(f"Recommended spawn Z: {-global_min_z + 0.02:.3f} (feet 2cm above ground)")
print(f"Standing pelvis height: {-global_min_z:.3f}")
