"""Rotate visual STL mesh vertices to align with Isaac Lab's physics frame.

The STL meshes from Blender are in Y-up coordinates. When Isaac Lab runs the
physics at 160 deg X base rotation (overriding the USD's baked -91 deg), the
collision shapes work correctly but the visual meshes appear rotated.

This script reads each *_visual.stl (or *_collision.stl used as visual), rotates
all vertices, and writes the result to a 'visual_rotated/' subdirectory.

The rotation angle = 160 - (-91) = 251 deg about X, applied as -251 = +109 deg.

Usage:
    python tools/rotate_visual_stls.py [--angle 109]
"""
import argparse
import math
import struct
from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "meshes"
OUT_DIR = MESH_DIR / "visual_rotated"


def rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read binary STL, return (normals [N,3], vertices [N,3,3])."""
    data = path.read_bytes()
    n_triangles = struct.unpack_from("<I", data, 80)[0]
    normals = np.zeros((n_triangles, 3), dtype=np.float32)
    vertices = np.zeros((n_triangles, 3, 3), dtype=np.float32)
    offset = 84
    for i in range(n_triangles):
        normals[i] = struct.unpack_from("<3f", data, offset)
        offset += 12
        for j in range(3):
            vertices[i, j] = struct.unpack_from("<3f", data, offset)
            offset += 12
        offset += 2  # attribute byte count
    return normals, vertices


def write_binary_stl(path: Path, normals: np.ndarray, vertices: np.ndarray):
    """Write binary STL."""
    n = len(normals)
    buf = bytearray(80)  # header
    buf += struct.pack("<I", n)
    for i in range(n):
        buf += struct.pack("<3f", *normals[i])
        for j in range(3):
            buf += struct.pack("<3f", *vertices[i, j])
        buf += struct.pack("<H", 0)
    path.write_bytes(buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=109.0,
                        help="Rotation angle in degrees about X axis")
    args = parser.parse_args()

    R = rot_x(args.angle)
    OUT_DIR.mkdir(exist_ok=True)

    stl_files = sorted(MESH_DIR.glob("*_collision.stl"))
    for stl in stl_files:
        normals, vertices = read_binary_stl(stl)

        rot_normals = (R @ normals.T).T
        rot_vertices = np.zeros_like(vertices)
        for j in range(3):
            rot_vertices[:, j, :] = (R @ vertices[:, j, :].T).T

        out_name = stl.name.replace("_collision.stl", "_visual_rotated.stl")
        out_path = OUT_DIR / out_name
        write_binary_stl(out_path, rot_normals, rot_vertices)
        print(f"  {stl.name} -> {out_name}  ({len(normals)} triangles)")

    print(f"\nWrote {len(stl_files)} rotated STLs to {OUT_DIR}")
    print(f"Rotation: +{args.angle} deg about X axis")


if __name__ == "__main__":
    main()
