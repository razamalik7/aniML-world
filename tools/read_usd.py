from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd
s = Usd.Stage.Open(r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd")
print("=== ROOT LAYER ===")
print(s.GetRootLayer().ExportToString())
print("\n=== ROOT_JOINT CHECK ===")
rj = s.GetPrimAtPath("/deer/root_joint")
if rj and rj.IsValid():
    print(f"root_joint exists, active={rj.IsActive()}, type={rj.GetTypeName()}")
else:
    print("root_joint: NOT FOUND (good!)")
deer = s.GetPrimAtPath("/deer")
if deer:
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(deer)
    for op in xf.GetOrderedXformOps():
        print(f"  xformOp: {op.GetName()} = {op.Get()}")
app.close()
