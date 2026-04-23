"""Inspect the deer_standing_robot.usd to understand visual mesh structure.

Finds all Mesh prims, reports their parent path, point count, and which
sublayer they are defined in.
"""
from __future__ import annotations
import argparse

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Sdf  # noqa: E402

USD_PATH = r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd"
LOG_PATH = r"C:\Users\nars7\Downloads\aniML_world\tools\usd_mesh_report.txt"


def main():
    stage = Usd.Stage.Open(USD_PATH)
    lines = []

    lines.append(f"Root layer: {stage.GetRootLayer().identifier}")
    lines.append(f"Sublayers:")
    for sl in stage.GetRootLayer().subLayerPaths:
        lines.append(f"  {sl}")

    lines.append("\n--- All Mesh prims ---")
    mesh_count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh_count += 1
        path = str(prim.GetPath())
        is_visual = "/visuals" in path
        is_collision = "/collisions" in path

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        n_pts = len(pts) if pts else 0

        fvc = mesh.GetFaceVertexCountsAttr().Get()
        n_faces = len(fvc) if fvc else 0

        tag = "VISUAL" if is_visual else ("COLLISION" if is_collision else "OTHER")

        lines.append(f"\n  [{tag}] {path}")
        lines.append(f"    Points: {n_pts}, Faces: {n_faces}")

        if pts and n_pts > 0:
            import numpy as np
            pts_arr = np.array(pts)
            mn = pts_arr.min(axis=0)
            mx = pts_arr.max(axis=0)
            lines.append(f"    Bounds: min=({mn[0]:.4f},{mn[1]:.4f},{mn[2]:.4f}) max=({mx[0]:.4f},{mx[1]:.4f},{mx[2]:.4f})")

        stack = prim.GetPrimStack()
        for spec in stack:
            lines.append(f"    Layer: {spec.layer.identifier}")

    lines.insert(3, f"\nTotal Mesh prims: {mesh_count}")

    text = "\n".join(lines)
    print(text, flush=True)
    with open(LOG_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved to: {LOG_PATH}")

    app.close()


if __name__ == "__main__":
    main()
