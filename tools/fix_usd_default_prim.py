"""Fix a saved Isaac Sim USD so the robot is the default prim.

When you save an Isaac Sim scene with File → Save As, the default prim is
/World (the whole scene). Isaac Lab expects the default prim to be the
robot root so it can spawn copies in parallel envs.

Usage:
  C:\isaac-lab\IsaacLab\_isaac_sim\python.bat tools\fix_usd_default_prim.py \
      deer_physics.usd /World/deer deer_physics_fixed.usd
"""
import sys
import os

if len(sys.argv) < 3:
    print("Usage: fix_usd_default_prim.py <input.usd> <robot_prim_path> [output.usd]")
    print("Example: fix_usd_default_prim.py deer_physics.usd /World/deer deer_robot.usd")
    sys.exit(1)

input_path = sys.argv[1]
robot_path = sys.argv[2]
output_path = sys.argv[3] if len(sys.argv) > 3 else input_path.replace(".usd", "_robot.usd")

print(f"Input:  {input_path}", flush=True)
print(f"Robot:  {robot_path}", flush=True)
print(f"Output: {output_path}", flush=True)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

try:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(input_path)
    if not stage:
        print("ERROR: could not open stage", flush=True)
        sys.exit(1)

    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        print(f"ERROR: prim not found at {robot_path}", flush=True)
        print(f"Available root prims: {[str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]}", flush=True)
        sys.exit(1)

    out_stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(out_stage, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(out_stage, UsdGeom.GetStageMetersPerUnit(stage))

    Sdf.CopySpec(stage.GetRootLayer(), robot_prim.GetPath(), out_stage.GetRootLayer(), Sdf.Path("/deer"))

    out_stage.SetDefaultPrim(out_stage.GetPrimAtPath("/deer"))

    phys_scene = stage.GetPrimAtPath("/physicsScene")
    if phys_scene and phys_scene.IsValid():
        Sdf.CopySpec(stage.GetRootLayer(), phys_scene.GetPath(), out_stage.GetRootLayer(), Sdf.Path("/physicsScene"))

    out_stage.GetRootLayer().Save()
    print(f"\nWrote: {output_path}", flush=True)
    print(f"Default prim: /deer", flush=True)

    rb_count = 0
    art_count = 0
    for p in out_stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_count += 1
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_count += 1
    print(f"Rigid bodies: {rb_count}", flush=True)
    print(f"Articulation roots: {art_count}", flush=True)

except Exception as ex:
    print(f"EXCEPTION: {ex}", flush=True)
    import traceback
    traceback.print_exc()

app.close()
