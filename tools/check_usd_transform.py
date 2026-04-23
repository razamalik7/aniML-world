"""Print the root prim transform and all body Z positions from a deer USD.

Run: _isaac_sim\\python.bat check_usd_transform.py <usd_path>
"""
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

usd_path = sys.argv[1]
stage = Usd.Stage.Open(usd_path)

dp = stage.GetDefaultPrim()
print(f"Default prim: {dp.GetPath()}", flush=True)

for prim in stage.Traverse():
    xf = UsdGeom.Xformable(prim)
    if not xf:
        continue
    ops = xf.GetOrderedXformOps()
    if ops:
        local = xf.GetLocalTransformation()
        translate = local.ExtractTranslation()
        rot = local.ExtractRotation()
        quat = rot.GetQuat()
        real = quat.GetReal()
        imag = quat.GetImaginary()
        path = str(prim.GetPath())
        has_rb = bool(UsdPhysics.RigidBodyAPI(prim))
        has_art = bool(UsdPhysics.ArticulationRootAPI(prim))
        tags = []
        if has_rb:
            tags.append("RigidBody")
        if has_art:
            tags.append("ArticulationRoot")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(
            f"{path}{tag_str}:  "
            f"translate=({translate[0]:.3f}, {translate[1]:.3f}, {translate[2]:.3f})  "
            f"rot_axis=({rot.GetAxis()[0]:.3f}, {rot.GetAxis()[1]:.3f}, {rot.GetAxis()[2]:.3f})  "
            f"rot_angle={rot.GetAngle():.1f}deg  "
            f"quat_wxyz=({real:.4f}, {imag[0]:.4f}, {imag[1]:.4f}, {imag[2]:.4f})",
            flush=True,
        )

app.close()
