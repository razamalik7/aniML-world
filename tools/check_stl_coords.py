"""Check if STL mesh data is in global or link-local coordinates.

Reads each STL file's bounding box and compares to joint positions
from deer.urdf to determine the coordinate frame.

Run: python tools/check_stl_coords.py
"""
import struct
from pathlib import Path

MESH_DIR = Path(r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\meshes")

JOINT_ORIGINS = {
    "pelvis": (0, 0, 0),
    "chest": (0, -0.965, 0.512),
    "neck": (0, -0.965 - 0.024, 0.512 - 0.850),
    "skull": (0, -0.965 - 0.024 - 0.368, 0.512 - 0.850 - 0.350),
    "HR_thigh": (-0.127, -0.843, 0.457),
    "HR_shin": (-0.127, -0.843 + 0.239, 0.457 - 0.232),
    "HR_foot": (-0.127 - 0.008, -0.843 + 0.239 + 0.169, 0.457 - 0.232 + 0.217),
    "HL_thigh": (0.127, -0.843, 0.457),
    "HL_shin": (0.127, -0.843 + 0.239, 0.457 - 0.232),
    "HL_foot": (0.127 + 0.008, -0.843 + 0.239 + 0.169, 0.457 - 0.232 + 0.217),
    "FR_thigh": (-0.122, -0.965 + 0.188, 0.512 - 0.889),
    "FR_shin": (-0.122 + 0.000, -0.965 + 0.188 + 0.142, 0.512 - 0.889 + 0.179),
    "FR_foot": (-0.122 + 0.001, -0.965 + 0.188 + 0.142 + 0.270, 0.512 - 0.889 + 0.179 - 0.096),
    "FL_thigh": (0.122, -0.965 + 0.188, 0.512 - 0.889),
    "FL_shin": (0.122, -0.965 + 0.188 + 0.142, 0.512 - 0.889 + 0.179),
    "FL_foot": (0.122, -0.965 + 0.188 + 0.142 + 0.268, 0.512 - 0.889 + 0.179 - 0.100),
}


def read_stl_bbox(path):
    with open(path, "rb") as f:
        f.read(80)
        n_tri = struct.unpack("<I", f.read(4))[0]
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        for _ in range(n_tri):
            f.read(12)  # normal
            for _ in range(3):
                vx, vy, vz = struct.unpack("<fff", f.read(12))
                for i, v in enumerate((vx, vy, vz)):
                    mins[i] = min(mins[i], v)
                    maxs[i] = max(maxs[i], v)
            f.read(2)  # attr
    center = [(mins[i] + maxs[i]) / 2 for i in range(3)]
    size = [maxs[i] - mins[i] for i in range(3)]
    return mins, maxs, center, size, n_tri


for stl_file in sorted(MESH_DIR.glob("*_collision.stl")):
    link_name = stl_file.stem.replace("_collision", "")
    joint_pos = JOINT_ORIGINS.get(link_name, None)
    mins, maxs, center, size, n_tri = read_stl_bbox(stl_file)

    print(f"\n{link_name} ({n_tri} triangles):")
    print(f"  STL center:  ({center[0]:7.3f}, {center[1]:7.3f}, {center[2]:7.3f})")
    print(f"  STL size:    ({size[0]:7.3f}, {size[1]:7.3f}, {size[2]:7.3f})")
    if joint_pos:
        dist = sum((center[i] - joint_pos[i]) ** 2 for i in range(3)) ** 0.5
        print(f"  Joint pos:   ({joint_pos[0]:7.3f}, {joint_pos[1]:7.3f}, {joint_pos[2]:7.3f})")
        print(f"  Distance center-to-joint: {dist:.3f}")
        if dist < 0.3:
            print(f"  -> GLOBAL frame (mesh centered near its joint's world position)")
        else:
            link_dist = sum(center[i] ** 2 for i in range(3)) ** 0.5
            if link_dist < 0.3:
                print(f"  -> LINK-LOCAL frame (mesh centered near origin)")
            else:
                print(f"  -> UNCLEAR (dist to origin: {link_dist:.3f}, dist to joint: {dist:.3f})")
