"""Rotate all deer STL mesh files by Rx(-90°) to align mesh frame with joint frame.

The STL data was exported from Blender in the mesh object's local frame (Y-based from FBX),
but the URDF joint origins are in the armature frame (Z-up). This script bakes the -90° X
rotation into the vertex/normal data so mesh and joint coordinates match.

Backs up originals to meshes_backup/ before modifying.

Run: python tools/rotate_stl_files.py
"""
import math
import shutil
import struct
from pathlib import Path

MESH_DIR = Path(r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\meshes")
BACKUP_DIR = MESH_DIR.parent / "meshes_backup"

ANGLE_DEG = -90.0
ANGLE_RAD = math.radians(ANGLE_DEG)
COS_A = math.cos(ANGLE_RAD)
SIN_A = math.sin(ANGLE_RAD)


def rx(x, y, z):
    """Rotate point by Rx(-90°): (x,y,z) -> (x, z, -y)"""
    return (x, y * COS_A - z * SIN_A, y * SIN_A + z * COS_A)


def rotate_stl(src: Path, dst: Path):
    with open(src, "rb") as f:
        header = f.read(80)
        n_tri = struct.unpack("<I", f.read(4))[0]
        triangles = []
        for _ in range(n_tri):
            nx, ny, nz = struct.unpack("<fff", f.read(12))
            v1 = struct.unpack("<fff", f.read(12))
            v2 = struct.unpack("<fff", f.read(12))
            v3 = struct.unpack("<fff", f.read(12))
            attr = f.read(2)
            triangles.append((rx(*[nx, ny, nz]), rx(*v1), rx(*v2), rx(*v3), attr))

    with open(dst, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n_tri))
        for normal, v1, v2, v3, attr in triangles:
            f.write(struct.pack("<fff", *normal))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<fff", *v3))
            f.write(attr)


def main():
    stl_files = list(MESH_DIR.glob("*.stl"))
    if not stl_files:
        print(f"No STL files found in {MESH_DIR}")
        return

    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True)
        for f in stl_files:
            shutil.copy2(f, BACKUP_DIR / f.name)
        print(f"Backed up {len(stl_files)} STL files to {BACKUP_DIR}")
    else:
        print(f"Backup already exists at {BACKUP_DIR}, skipping backup")

    for stl_file in sorted(stl_files):
        backup = BACKUP_DIR / stl_file.name
        if not backup.exists():
            shutil.copy2(stl_file, backup)

        rotate_stl(backup, stl_file)
        orig_size = backup.stat().st_size
        new_size = stl_file.stat().st_size
        print(f"  Rotated: {stl_file.name} ({orig_size} -> {new_size} bytes)")

    print(f"\nDone. Rotated {len(stl_files)} STL files by Rx({ANGLE_DEG}°)")
    print("Original files preserved in meshes_backup/")


if __name__ == "__main__":
    main()
