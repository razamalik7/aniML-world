"""Open deer USD, apply Rx(160deg) to root, render statically.

Run: _isaac_sim\\python.bat view_usd_rotated.py <usd_path>
"""
import sys, time, math

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import omni  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

usd_path = sys.argv[1]
print(f"Opening: {usd_path}", flush=True)

omni.usd.get_context().open_stage(usd_path)
for _ in range(30):
    app.update()

stage = omni.usd.get_context().get_stage()

root = stage.GetDefaultPrim()
print(f"Default prim: {root.GetPath()}", flush=True)

xform = UsdGeom.Xformable(root)

existing_ops = xform.GetOrderedXformOps()
print(f"Existing xform ops: {[str(op.GetOpName()) for op in existing_ops]}", flush=True)

xform.ClearXformOpOrder()

translate_op = xform.AddTranslateOp()
translate_op.Set(Gf.Vec3d(0, 0, 1.0))

orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
w_val, x_val = 1.0, 0.0
orient_op.Set(Gf.Quatd(w_val, x_val, 0, 0))
deg = 0.0

print(f"Applied: Rx({deg}deg), Z=1.0m", flush=True)
print(f"Quaternion: w={w_val:.6f}, x={x_val:.6f}, y=0, z=0", flush=True)

for _ in range(10):
    app.update()

print("\n--- Rotated static view. Close window or Ctrl+C to exit. ---", flush=True)
while app.is_running():
    app.update()
    time.sleep(0.05)

app.close()
