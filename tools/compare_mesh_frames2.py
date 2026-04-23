"""Compare mesh vertices between deer_base.usd and raw STL files."""
from __future__ import annotations
import struct
from pathlib import Path

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Sdf  # noqa: E402
import numpy as np  # noqa: E402

BASE_USD = r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer\configuration\deer_base.usd"
MESH_DIR = Path(r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\meshes")
LOG_PATH = r"C:\Users\nars7\Downloads\aniML_world\tools\frame_comparison.txt"


def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    verts = []
    offset = 84
    for i in range(n):
        offset += 12
        for j in range(3):
            verts.append(struct.unpack_from("<3f", data, offset))
            offset += 12
        offset += 2
    return np.array(verts, dtype=np.float32)


def main():
    stage = Usd.Stage.Open(BASE_USD)
    lines = []

    link_translates = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/deer/"):
            continue
        parts = path.split("/")
        if len(parts) != 3:
            continue
        if prim.GetTypeName() != "Xform":
            continue
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        t = ops.get("xformOp:translate")
        if t:
            link_translates[parts[2]] = np.array(t.Get())
            lines.append(f"Link {parts[2]}: translate=({t.Get()[0]:.4f}, {t.Get()[1]:.4f}, {t.Get()[2]:.4f})")

    lines.append(f"\nFound {len(link_translates)} link translates")
    lines.append("\n--- Mesh comparison ---")

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/meshes/") or prim.GetTypeName() != "Mesh":
            continue

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            lines.append(f"\n{path}: no points!")
            continue
        usd_pts = np.array(pts)

        parts = path.split("/")
        mesh_name = parts[2]
        stl_path = MESH_DIR / f"{mesh_name}.stl"
        if not stl_path.exists():
            lines.append(f"\n{mesh_name}: STL not found at {stl_path}")
            continue

        stl_pts = read_stl_vertices(stl_path)
        stl_unique = np.unique(stl_pts, axis=0)
        usd_unique = np.unique(usd_pts, axis=0)

        link_name = mesh_name.replace("_collision", "")
        link_t = link_translates.get(link_name, np.zeros(3))

        usd_mean = usd_pts.mean(axis=0)
        stl_mean = stl_unique.mean(axis=0)
        diff_world = np.linalg.norm(usd_mean - stl_mean)
        diff_local = np.linalg.norm(usd_mean - (stl_mean - link_t))

        lines.append(f"\n{mesh_name} ({len(usd_pts)} USD pts, {len(stl_unique)} STL unique verts):")
        lines.append(f"  Link translate: ({link_t[0]:.4f}, {link_t[1]:.4f}, {link_t[2]:.4f})")
        lines.append(f"  STL centroid:   ({stl_mean[0]:.4f}, {stl_mean[1]:.4f}, {stl_mean[2]:.4f})")
        lines.append(f"  USD centroid:   ({usd_mean[0]:.4f}, {usd_mean[1]:.4f}, {usd_mean[2]:.4f})")
        lines.append(f"  ||USD - STL||  = {diff_world:.6f}  (world frame)")
        lines.append(f"  ||USD - (STL-link_t)|| = {diff_local:.6f}  (local frame)")

        if diff_world < 0.01:
            lines.append(f"  => WORLD FRAME")
        elif diff_local < 0.01:
            lines.append(f"  => LOCAL FRAME")
        else:
            lines.append(f"  => UNKNOWN FRAME")

    text = "\n".join(lines)
    print(text, flush=True)
    with open(LOG_PATH, "w") as f:
        f.write(text + "\n")

    app.close()


if __name__ == "__main__":
    main()
