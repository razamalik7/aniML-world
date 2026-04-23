"""
Patch deer_world.urdf IN PLACE with mesh-computed CoM offsets from the STL collision meshes.

This keeps everything else identical (joint tree, masses, joint limits, mesh references)
and only adds/updates the <origin xyz="..."/> inside each <inertial> element.

The CoM offsets are computed as volume-weighted centroids of the collision STL meshes.
These are in each link's LOCAL coordinate frame (which is what URDF <inertial> expects).
"""

import struct
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_PATH = Path(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf')
MESH_DIR  = Path(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\meshes\world')


def read_stl_binary(path):
    with open(path, 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        triangles = []
        for _ in range(n):
            f.read(12)
            v = [struct.unpack('<3f', f.read(12)) for _ in range(3)]
            f.read(2)
            triangles.append(v)
    return np.array(triangles, dtype=np.float64)


def stl_centroid(path):
    tris = read_stl_binary(path)
    if len(tris) == 0:
        return np.zeros(3)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    signed_vol = np.einsum('ni,ni->n', v0, np.cross(v1, v2)) / 6.0
    total_vol = signed_vol.sum()
    if abs(total_vol) > 1e-10:
        tet_centroids = (v0 + v1 + v2) / 4.0
        return (signed_vol[:, None] * tet_centroids).sum(axis=0) / total_vol
    else:
        tri_centroids = tris.mean(axis=1)
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_area = areas.sum()
        if total_area < 1e-10:
            return np.zeros(3)
        return (areas[:, None] * tri_centroids).sum(axis=0) / total_area


ET.register_namespace('', '')
tree = ET.parse(URDF_PATH)
root = tree.getroot()

print(f"Patching {URDF_PATH.name} with mesh-computed CoM offsets")
print(f"  {'Link':<20} {'Old CoM':>25} {'New CoM (from STL)':>35}")
print(f"  {'-'*85}")

patched = 0
for link in root.findall('link'):
    name = link.get('name')
    stl_path = MESH_DIR / f"{name}_collision.stl"
    if not stl_path.exists():
        continue

    com = stl_centroid(stl_path)
    inertial = link.find('inertial')
    if inertial is None:
        continue

    origin = inertial.find('origin')
    old_xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
    new_xyz = f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}"

    if origin is None:
        origin = ET.SubElement(inertial, 'origin')
    origin.set('xyz', new_xyz)
    origin.set('rpy', '0 0 0')
    patched += 1

    print(f"  {name:<20} {old_xyz:>25} -> ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f})")

tree.write(URDF_PATH, xml_declaration=True, encoding='utf-8')
print(f"\nPatched {patched} links. Saved to {URDF_PATH}")
