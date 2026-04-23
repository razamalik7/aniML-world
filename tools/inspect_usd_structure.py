"""Inspect the USD prim hierarchy to understand visual vs collision mesh structure."""
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

usd_path = sys.argv[1] if len(sys.argv) > 1 else "deer_fixed_robot.usd"
stage = Usd.Stage.Open(usd_path)

for prim in stage.Traverse():
    path = str(prim.GetPath())
    ptype = prim.GetTypeName()
    depth = path.count("/")
    indent = "  " * min(depth - 1, 8)
    extras = []
    if UsdPhysics.CollisionAPI(prim):
        extras.append("COL")
    if UsdPhysics.RigidBodyAPI(prim):
        extras.append("RB")
    tag = f" [{','.join(extras)}]" if extras else ""
    print(f"{indent}{path} ({ptype}){tag}", flush=True)

app.close()
