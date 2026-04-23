"""Compare USD mesh vertices with raw STL vertices to determine coordinate frame.

If USD vertices == STL vertices, meshes are in Blender world frame.
If USD vertices == STL vertices - link_offset, meshes are in link-local frame.
"""
from __future__ import annotations
import struct
import math
from pathlib import Path

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402
import numpy as np  # noqa: E402

USD_PATH = r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd"
MESH_DIR = Path(r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\meshes")
LOG_PATH = r"C:\Users\nars7\Downloads\aniML_world\tools\frame_comparison.txt"


def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    verts = []
    offset = 84
    for i in range(n):
        offset += 12  # skip normal
        for j in range(3):
            verts.append(struct.unpack_from("<3f", data, offset))
            offset += 12
        offset += 2
    return np.array(verts, dtype=np.float32)


def main():
    stage = Usd.Stage.Open(USD_PATH)
    lines = []

    link_translates = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/deer/") or "/" in path[6:]:
            if path.count("/") != 2:
                continue
        if prim.GetTypeName() != "Xform":
            continue
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        t = ops.get("xformOp:translate")
        if t:
            link_name = path.split("/")[-1]
            link_translates[link_name] = np.array(t.Get())

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/meshes/") or prim.GetTypeName() != "Mesh":
            continue

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        usd_pts = np.array(pts)

        parts = path.split("/")
        mesh_name = parts[2]  # e.g. "pelvis_collision"
        stl_path = MESH_DIR / f"{mesh_name}.stl"
        if not stl_path.exists():
            lines.append(f"\n{mesh_name}: STL not found")
            continue

        stl_pts = read_stl_vertices(stl_path)

        link_name = mesh_name.replace("_collision", "")
        link_t = link_translates.get(link_name, np.zeros(3))

        usd_mean = usd_pts.mean(axis=0)
        stl_mean = stl_pts.mean(axis=0)
        diff_world = np.linalg.norm(usd_mean - stl_mean)
        diff_local = np.linalg.norm(usd_mean - (stl_mean - link_t))

        lines.append(f"\n{mesh_name}:")
        lines.append(f"  Link translate: ({link_t[0]:.4f}, {link_t[1]:.4f}, {link_t[2]:.4f})")
        lines.append(f"  STL centroid:   ({stl_mean[0]:.4f}, {stl_mean[1]:.4f}, {stl_mean[2]:.4f})")
        lines.append(f"  USD centroid:   ({usd_mean[0]:.4f}, {usd_mean[1]:.4f}, {usd_mean[2]:.4f})")
        lines.append(f"  ||USD - STL||  = {diff_world:.6f}  (world frame match)")
        lines.append(f"  ||USD - (STL - link_t)|| = {diff_local:.6f}  (local frame match)")

        if diff_world < 0.001:
            lines.append(f"  => WORLD FRAME (vertices are in Blender world coords)")
        elif diff_local < 0.001:
            lines.append(f"  => LOCAL FRAME (vertices are relative to link origin)")
        else:
            lines.append(f"  => UNKNOWN FRAME")

    text = "\n".join(lines)
    print(text, flush=True)
    with open(LOG_PATH, "w") as f:
        f.write(text + "\n")

    app.close()


if __name__ == "__main__":
    main()
