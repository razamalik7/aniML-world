"""Reset deer root to identity transform."""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom, UsdPhysics

stage = Usd.Stage.Open(sys.argv[1])
deer = stage.GetPrimAtPath("/deer")
xf = UsdGeom.Xformable(deer)
xf.ClearXformOpOrder()

trans = xf.AddTranslateOp()
trans.Set(Gf.Vec3d(0, 0, 0))
orient = xf.AddOrientOp()
orient.Set(Gf.Quatf(1, 0, 0, 0))
scale = xf.AddScaleOp()
scale.Set(Gf.Vec3f(1, 1, 1))

stage.GetRootLayer().Save()
print("Reset /deer to identity transform", flush=True)

rb = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
print(f"Rigid bodies: {rb}", flush=True)
app.close()
