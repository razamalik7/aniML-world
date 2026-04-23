"""Fix visual mesh alignment in the working USD with per-link transforms.

The mesh STL data is in Blender's world frame but the physics operates at a
different base rotation. Each link's visual needs a rotation around Blender's
world origin (0,0,0), not around the link's own origin.

For a link at position p, the visual transform is:
  T(p - R * p) * R   (where R is the compensation rotation)

This rotates the mesh around (0,0,0) in Blender world while keeping it
attached to its physics link.

Usage:
    isaaclab.bat -p tools/fix_usd_visuals.py
"""
from __future__ import annotations
import math
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom  # noqa: E402
import numpy as np  # noqa: E402

SRC = r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd"
DST = r"C:\Users\nars7\Downloads\aniML_world\deer_visualfix_robot.usd"

ANGLE_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 109.0
ANGLE_RAD = math.radians(ANGLE_DEG)

c, s = math.cos(ANGLE_RAD), math.sin(ANGLE_RAD)
R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

stage = Usd.Stage.Open(SRC)

visual_count = 0
log_lines = []

for prim in stage.Traverse():
    path = str(prim.GetPath())
    if not path.endswith("/visuals") or prim.GetTypeName() != "Xform":
        continue

    parent = prim.GetParent()
    if not parent:
        continue

    parent_xf = UsdGeom.Xformable(parent)
    parent_ops = {op.GetOpName(): op for op in parent_xf.GetOrderedXformOps()}
    translate_op = parent_ops.get("xformOp:translate")
    if translate_op is None:
        log_lines.append(f"  SKIP {path}: no translate on parent")
        continue

    p = np.array(translate_op.Get())

    Rp = R @ p
    offset = p - Rp

    xf = UsdGeom.Xformable(prim)

    translate_vis = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_vis.Set(Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2])))

    q = Gf.Quatd(math.cos(ANGLE_RAD / 2), math.sin(ANGLE_RAD / 2), 0, 0)
    orient_vis = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    orient_vis.Set(q)

    visual_count += 1
    log_lines.append(
        f"  {path}: parent_pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) "
        f"offset=({offset[0]:.4f},{offset[1]:.4f},{offset[2]:.4f})"
    )

log_lines.insert(0, f"Modified {visual_count} visual Xforms with +{ANGLE_DEG} deg X rotation + pivot offset")

stage.GetRootLayer().Export(DST)
log_lines.append(f"\nExported to: {DST}")

log_text = "\n".join(log_lines)
print(log_text, flush=True)

with open(DST.replace(".usd", "_log.txt"), "w") as f:
    f.write(log_text + "\n")

app.close()
