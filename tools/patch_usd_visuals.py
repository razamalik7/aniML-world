"""
Patch visual mesh vertex data in deer_fixed_robot.usd IN-PLACE.

Reads rotation-only visual STLs from meshes/visual_rotonly/ and replaces
ONLY the vertex positions and normals in the /visuals/ and /meshes/*_visual/
prims. All collision and physics prims are completely untouched.

This preserves the trained policy while fixing the floating-panel artifact
caused by per-link offsets that drift when joints rotate.
"""
import struct
import os
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf, Vt  # noqa: E402

LINKS = [
    "pelvis", "chest", "neck", "skull",
    "HR_thigh", "HR_shin", "HR_foot",
    "HL_thigh", "HL_shin", "HL_foot",
    "FR_thigh", "FR_shin", "FR_foot",
    "FL_thigh", "FL_shin", "FL_foot",
]

MESH_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "deer_urdf", "meshes", "visual_rotonly")


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)  # header
        (num_tri,) = struct.unpack("<I", f.read(4))
        tris = []
        for _ in range(num_tri):
            data = struct.unpack("<12fH", f.read(50))
            tris.append((data[0:3], data[3:6], data[6:9], data[9:12], data[12]))
        return tris


def set_mesh_from_stl(mesh_prim, tris):
    """Replace points and normals on a UsdGeom.Mesh with STL triangle data."""
    mesh = UsdGeom.Mesh(mesh_prim)

    new_pts = []
    new_norms = []
    for normal, v1, v2, v3, _ in tris:
        new_pts.append(Gf.Vec3f(*v1))
        new_pts.append(Gf.Vec3f(*v2))
        new_pts.append(Gf.Vec3f(*v3))
        new_norms.append(Gf.Vec3f(*normal))
        new_norms.append(Gf.Vec3f(*normal))
        new_norms.append(Gf.Vec3f(*normal))

    mesh.GetPointsAttr().Set(Vt.Vec3fArray(new_pts))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray(new_norms))

    n = len(tris)
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * n))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(list(range(3 * n))))


def find_mesh_prim(stage, base_path):
    """Walk under base_path to find the first Mesh typed prim."""
    root = stage.GetPrimAtPath(base_path)
    if not root:
        return None
    for p in Usd.PrimRange(root):
        if p.GetTypeName() == "Mesh":
            return p
    return None


usd_path = sys.argv[1] if len(sys.argv) > 1 else "deer_fixed_robot.usd"
stage = Usd.Stage.Open(usd_path)

patched = 0
for link in LINKS:
    stl_path = os.path.join(MESH_DIR, f"{link}_visual.stl")
    if not os.path.exists(stl_path):
        print(f"  SKIP {link} - no STL")
        continue

    tris = read_stl(stl_path)

    paths_to_patch = [
        f"/visuals/{link}/{link}_visual",
        f"/meshes/{link}_visual",
    ]

    for base in paths_to_patch:
        mesh_prim = find_mesh_prim(stage, base)
        if mesh_prim:
            old_pts = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            old_count = len(old_pts) if old_pts else 0
            set_mesh_from_stl(mesh_prim, tris)
            new_pts = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            print(f"  {mesh_prim.GetPath()}: {old_count} -> {len(new_pts)} pts ({len(tris)} tris)")
            patched += 1
        else:
            print(f"  NOT FOUND: {base}")

stage.Save()
print(f"\nPatched {patched} visual mesh prims in {usd_path}")
print("Collision / physics prims are UNTOUCHED.")

app.close()
