"""Copy the deer's transform from the original Isaac Sim scene save to the extracted USD."""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

original = sys.argv[1]  # deer_physics.usd (full scene)
robot = sys.argv[2]     # deer_robot.usd (extracted)
src_path = sys.argv[3] if len(sys.argv) > 3 else "/World/deer"

orig_stage = Usd.Stage.Open(original)
orig_prim = orig_stage.GetPrimAtPath(src_path)
orig_xf = UsdGeom.Xformable(orig_prim)

print("Original xform ops:", flush=True)
for op in orig_xf.GetOrderedXformOps():
    print(f"  {op.GetOpType()}: {op.Get()}", flush=True)

robot_stage = Usd.Stage.Open(robot)
deer_prim = robot_stage.GetPrimAtPath("/deer")
robot_xf = UsdGeom.Xformable(deer_prim)

robot_xf.ClearXformOpOrder()

for op in orig_xf.GetOrderedXformOps():
    optype = op.GetOpType()
    if optype == UsdGeom.XformOp.TypeTranslate:
        new_op = robot_xf.AddTranslateOp()
        new_op.Set(Gf.Vec3d(0, 0, 0))
        print(f"Set translate to (0,0,0) — scene-level offset not needed", flush=True)
    elif optype == UsdGeom.XformOp.TypeOrient:
        new_op = robot_xf.AddOrientOp()
        new_op.Set(op.Get())
        print(f"Copied orient: {op.Get()}", flush=True)
    elif optype == UsdGeom.XformOp.TypeScale:
        new_op = robot_xf.AddScaleOp()
        new_op.Set(op.Get())
        print(f"Copied scale: {op.Get()}", flush=True)

print("\nBody world positions with copied transform:", flush=True)
positions = {}
for prim in robot_stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        xfable = UsdGeom.Xformable(prim)
        world_tf = xfable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = world_tf.ExtractTranslation()
        name = str(prim.GetPath()).split("/")[-1]
        positions[name] = (pos[0], pos[1], pos[2])
        print(f"  {name:20s}  x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}", flush=True)

feet = {k: v for k, v in positions.items() if "foot" in k.lower()}
if feet:
    min_foot_z = min(v[2] for v in feet.values())
    max_foot_z = max(v[2] for v in feet.values())
    pelvis_z = positions.get("pelvis", (0,0,0))[2]
    print(f"\nFoot Z range: {min_foot_z:.3f} to {max_foot_z:.3f}", flush=True)
    print(f"Pelvis Z: {pelvis_z:.3f}", flush=True)
    print(f"Feet below pelvis: {min_foot_z < pelvis_z}", flush=True)
    print(f"Suggested spawn Z: {abs(min_foot_z) + 0.10:.2f}", flush=True)

robot_stage.GetRootLayer().Save()
print(f"\nSaved: {robot}", flush=True)

app.close()
