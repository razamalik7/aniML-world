"""
Compute the geometric centroid of each collision STL mesh for deer_world.urdf
and write those centroids as physics:centerOfMass offsets in the URDF.

The centroid of an STL = volume-weighted average of triangle centroids
(signed volume method for watertight meshes). For non-watertight meshes
we fall back to area-weighted triangle centroid.

The CoM offsets in URDF <inertial><origin xyz="..."/> are expressed in the
LINK's local coordinate frame. For deer_world.urdf all joint RPY = 0, so
the link frame is axis-aligned with the URDF world frame at rest pose.
The STL files are exported in link-local coordinates (Blender export places
each mesh piece with its own local origin at the joint location).
"""

import struct, os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_PATH  = Path(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf')
MESH_DIR   = Path(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\meshes\world')
OUT_PATH   = Path(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world_comfix.urdf')


def read_stl_binary(path):
    """Return Nx3x3 array of triangles from a binary STL file."""
    with open(path, 'rb') as f:
        f.read(80)  # header
        n = struct.unpack('<I', f.read(4))[0]
        triangles = []
        for _ in range(n):
            f.read(12)  # normal
            v = [struct.unpack('<3f', f.read(12)) for _ in range(3)]
            f.read(2)   # attr
            triangles.append(v)
    return np.array(triangles, dtype=np.float64)  # (N, 3, 3)


def stl_centroid(path):
    """
    Compute volume-weighted centroid of a closed STL mesh.
    Falls back to area-weighted if mesh is open (signed volume ≈ 0).
    Returns centroid as (x, y, z) in mesh local coordinates.
    """
    tris = read_stl_binary(path)
    if len(tris) == 0:
        return np.zeros(3)

    # Triangle centroids
    tri_centroids = tris.mean(axis=1)   # (N, 3)

    # Signed volume of each tetrahedron (origin + triangle)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    signed_vol = np.einsum('ni,ni->n', v0, np.cross(v1, v2)) / 6.0  # (N,)
    total_vol  = signed_vol.sum()

    if abs(total_vol) > 1e-10:
        # Volume-weighted centroid (correct for watertight mesh)
        # centroid of each tet = (v0 + v1 + v2 + origin) / 4 ≈ tri_centroid * 3/4
        # More precisely: centroid_tet = (v0 + v1 + v2) / 4 (origin is 0)
        tet_centroids = (v0 + v1 + v2) / 4.0
        com = (signed_vol[:, None] * tet_centroids).sum(axis=0) / total_vol
    else:
        # Area-weighted fallback
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_area = areas.sum()
        if total_area < 1e-10:
            return np.zeros(3)
        com = (areas[:, None] * tri_centroids).sum(axis=0) / total_area

    return com


# Parse URDF
ET.register_namespace('', '')
tree = ET.parse(URDF_PATH)
root = tree.getroot()

print(f"Computing CoM from STL meshes for deer_world.urdf")
print(f"Mesh dir: {MESH_DIR}")
print()
print(f"  {'Link':<20} {'mesh file':<35} {'CoM_local (x,y,z)':>40}  {'|CoM|':>8}")
print(f"  {'-'*110}")

patched = 0
for link in root.findall('link'):
    name = link.get('name')
    stl_path = MESH_DIR / f"{name}_collision.stl"

    if not stl_path.exists():
        print(f"  {name:<20} {'-- no STL --':<35}")
        continue

    com = stl_centroid(stl_path)
    mag = np.linalg.norm(com)
    print(f"  {name:<20} {stl_path.name:<35} ({com[0]:+.6f}, {com[1]:+.6f}, {com[2]:+.6f})  {mag:.4f}")

    # Write into URDF inertial origin
    inertial = link.find('inertial')
    if inertial is not None:
        origin = inertial.find('origin')
        if origin is None:
            origin = ET.SubElement(inertial, 'origin')
        origin.set('xyz', f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
        origin.set('rpy', '0 0 0')
        patched += 1

print()
print(f"Patched {patched} links with mesh-computed CoM offsets.")

# Verify sanity: show mass * |CoM| for biggest contributors
print()
print("  Top heavy contributors (mass × |CoM| in link-local frame):")
for link in root.findall('link'):
    inertial = link.find('inertial')
    if inertial is None:
        continue
    m   = float(inertial.find('mass').get('value'))
    org = inertial.find('origin')
    if org is None:
        continue
    xyz = list(map(float, org.get('xyz', '0 0 0').split()))
    com = np.array(xyz)
    print(f"    {link.get('name'):<20} mass={m:.1f}  |CoM_local|={np.linalg.norm(com):.3f}  contribution={m * np.linalg.norm(com):.3f}")

# Save
tree.write(OUT_PATH, xml_declaration=True, encoding='utf-8')
print()
print(f"Saved: {OUT_PATH}")
print()
print("Next step: regenerate USD from deer_world_comfix.urdf using:")
print("  isaaclab.bat -p C:\\isaac-lab\\IsaacLab\\scripts\\tools\\convert_urdf.py \\")
print("    assets/deer_urdf/deer_world_comfix.urdf deer_world_comfix.usd \\")
print("    --joint-stiffness 200.0 --joint-damping 6.0")
