"""List all prims in a USD file with their types and xform ops."""
from __future__ import annotations
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

usd_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd"
out_path = usd_path.replace(".usd", "_prims.txt")

stage = Usd.Stage.Open(usd_path)
lines = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    ptype = prim.GetTypeName()
    flags = []
    if UsdPhysics.RigidBodyAPI(prim):
        flags.append("RigidBody")
    if UsdPhysics.CollisionAPI(prim):
        flags.append("Collision")
    if UsdPhysics.ArticulationRootAPI(prim):
        flags.append("ArticRoot")
    xf = UsdGeom.Xformable(prim)
    if xf:
        ops = [f"{op.GetOpName()}={op.Get()}" for op in xf.GetOrderedXformOps()]
    else:
        ops = []
    line = f"{path}  type={ptype}  flags={flags}  xformOps={ops}"
    lines.append(line)

with open(out_path, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {len(lines)} prims to {out_path}", flush=True)
app.close()
