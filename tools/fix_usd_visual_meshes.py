"""Create a deer USD with properly rotated visual mesh vertices.

The physics articulation uses a 160 deg X base rotation while the USD was
authored with -91 deg. The collision shapes work fine (physics engine
handles the transform), but the visual meshes need their vertex data
rotated so they visually match the physics pose.

Strategy:
  1. Open deer_base.usd
  2. Create NEW mesh prims under /meshes_visual/ with rotated point data
  3. Update /visuals/ references to point to /meshes_visual/ instead of /meshes/
  4. Save a copy — collision refs stay pointing at original /meshes/

Usage:
    isaaclab.bat -p tools/fix_usd_visual_meshes.py [--angle 109]
"""
from __future__ import annotations
import argparse
import math
import shutil
from pathlib import Path

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: E402
import numpy as np  # noqa: E402

SRC_DIR = Path(r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer")
DST_DIR = Path(r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_visualfix")
ROOT_USD = Path(r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd")
DST_ROOT = Path(r"C:\Users\nars7\Downloads\aniML_world\deer_visualfix2_robot.usd")


def rot_x_matrix(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=109.0)
    args = parser.parse_args()

    R = rot_x_matrix(args.angle)
    lines = [f"Visual mesh rotation: +{args.angle} deg X\n"]

    # --- Step 1: Copy the USD hierarchy ---
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    shutil.copytree(SRC_DIR, DST_DIR)
    lines.append(f"Copied {SRC_DIR} -> {DST_DIR}")

    base_usd_path = DST_DIR / "configuration" / "deer_base.usd"
    stage = Usd.Stage.Open(str(base_usd_path))

    # --- Step 2: Create /meshes_visual/ with rotated vertex data ---
    meshes_scope = stage.GetPrimAtPath("/meshes")
    if not meshes_scope:
        print("ERROR: /meshes scope not found!")
        app.close()
        return

    meshes_vis_scope = stage.DefinePrim("/meshes_visual", "Scope")
    rotated_count = 0

    for mesh_prim in meshes_scope.GetChildren():
        mesh_name = mesh_prim.GetName()
        src_mesh_path = f"/meshes/{mesh_name}/node_STL_BINARY_/mesh"
        src_mesh = stage.GetPrimAtPath(src_mesh_path)
        if not src_mesh or not src_mesh.IsA(UsdGeom.Mesh):
            lines.append(f"  SKIP {mesh_name}: no mesh at {src_mesh_path}")
            continue

        src_geom = UsdGeom.Mesh(src_mesh)
        pts = src_geom.GetPointsAttr().Get()
        fvc = src_geom.GetFaceVertexCountsAttr().Get()
        fvi = src_geom.GetFaceVertexIndicesAttr().Get()
        normals = src_geom.GetNormalsAttr().Get()

        if not pts:
            lines.append(f"  SKIP {mesh_name}: no points")
            continue

        pts_arr = np.array(pts)
        rot_pts = (R @ pts_arr.T).T

        new_parent = stage.DefinePrim(f"/meshes_visual/{mesh_name}", "Xform")
        new_stl = stage.DefinePrim(f"/meshes_visual/{mesh_name}/node_STL_BINARY_", "Xform")
        xf = UsdGeom.Xformable(new_stl)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1, 0, 0, 0))
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1, 1, 1))

        new_mesh_prim = stage.DefinePrim(
            f"/meshes_visual/{mesh_name}/node_STL_BINARY_/mesh", "Mesh"
        )
        new_geom = UsdGeom.Mesh(new_mesh_prim)

        new_points = Vt.Vec3fArray([Gf.Vec3f(*p) for p in rot_pts])
        new_geom.CreatePointsAttr(new_points)

        if fvc:
            new_geom.CreateFaceVertexCountsAttr(fvc)
        if fvi:
            new_geom.CreateFaceVertexIndicesAttr(fvi)
        if normals:
            normals_arr = np.array(normals)
            rot_normals = (R @ normals_arr.T).T
            new_normals = Vt.Vec3fArray([Gf.Vec3f(*n) for n in rot_normals])
            new_geom.CreateNormalsAttr(new_normals)
            new_geom.SetNormalsInterpolation(src_geom.GetNormalsInterpolation())

        src_looks = stage.GetPrimAtPath(
            f"/meshes/{mesh_name}/node_STL_BINARY_/Looks"
        )
        if src_looks:
            looks = stage.DefinePrim(
                f"/meshes_visual/{mesh_name}/node_STL_BINARY_/Looks", "Scope"
            )
            mat = stage.DefinePrim(
                f"/meshes_visual/{mesh_name}/node_STL_BINARY_/Looks/DefaultMaterial",
                "Material",
            )
            shader = stage.DefinePrim(
                f"/meshes_visual/{mesh_name}/node_STL_BINARY_/Looks/DefaultMaterial/DefaultMaterial",
                "Shader",
            )

        rotated_count += 1
        old_c = np.mean(pts_arr, axis=0)
        new_c = np.mean(rot_pts, axis=0)
        lines.append(
            f"  {mesh_name}: {len(pts)} pts, "
            f"centroid ({old_c[0]:.3f},{old_c[1]:.3f},{old_c[2]:.3f}) -> "
            f"({new_c[0]:.3f},{new_c[1]:.3f},{new_c[2]:.3f})"
        )

    lines.append(f"\nCreated {rotated_count} rotated mesh prims under /meshes_visual/")

    # --- Step 3: Update /visuals/ references to point to /meshes_visual/ ---
    ref_count = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/visuals/") or prim.GetTypeName() != "Xform":
            continue
        parts = path.split("/")
        if len(parts) != 4:
            continue
        link_name = parts[2]
        collision_name = parts[3]

        old_ref_path = f"/meshes/{collision_name}"
        new_ref_path = f"/meshes_visual/{collision_name}"

        new_mesh_prim = stage.GetPrimAtPath(new_ref_path)
        if not new_mesh_prim:
            lines.append(f"  SKIP ref update for {path}: {new_ref_path} not found")
            continue

        refs = prim.GetReferences()
        refs.ClearReferences()
        refs.AddInternalReference(new_ref_path)
        ref_count += 1
        lines.append(f"  Updated ref: {path} -> {new_ref_path}")

    lines.append(f"\nUpdated {ref_count} visual references")

    stage.GetRootLayer().Save()
    lines.append(f"\nSaved: {base_usd_path}")

    # --- Step 4: Create root USD that references the new hierarchy ---
    src_root_text = ROOT_USD.read_text(encoding="utf-8", errors="replace")
    dst_root_text = src_root_text.replace(
        "assets/deer_urdf/deer/", "assets/deer_urdf/deer_visualfix/"
    ).replace(
        "deer_urdf/deer/", "deer_urdf/deer_visualfix/"
    )
    DST_ROOT.write_text(dst_root_text, encoding="utf-8")
    lines.append(f"Created root: {DST_ROOT}")

    text = "\n".join(lines)
    print(text, flush=True)
    log_path = DST_ROOT.with_suffix(".log.txt")
    with open(log_path, "w") as f:
        f.write(text + "\n")
    print(f"\nLog: {log_path}")

    app.close()


if __name__ == "__main__":
    main()
