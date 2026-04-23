"""Check transform on the deer root prim in the extracted USD."""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom

stage = Usd.Stage.Open(sys.argv[1])
print(f"Up axis: {UsdGeom.GetStageUpAxis(stage)}", flush=True)
print(f"Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}", flush=True)

for path in ["/deer", "/deer/pelvis", "/deer/chest", "/deer/HR_thigh", "/deer/HR_foot"]:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print(f"{path}: NOT FOUND", flush=True)
        continue
    xf = UsdGeom.Xformable(prim)
    local = xf.GetLocalTransformation()
    translate = local.ExtractTranslation()
    rot = local.ExtractRotation()
    print(f"{path}:  translate={translate}  rotation_axis={rot.GetAxis()}  angle={rot.GetAngle():.1f}", flush=True)

app.close()
