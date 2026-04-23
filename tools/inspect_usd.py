"""Quick USD inspector — writes results to a file to avoid stdout issues."""
import sys

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
sim = launcher.app

from pxr import Usd, UsdGeom, UsdPhysics

out_path = r"C:\Users\nars7\Downloads\aniML_world\usd_inspect_results.txt"
lines = []

def log(msg=""):
    lines.append(msg)
    print(msg, flush=True)

for path in [
    r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd",
    r"C:\Users\nars7\Downloads\aniML_world\deer_standing_physics.usd",
    r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer\deer.usd",
    r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer\configuration\deer_robot.usd",
]:
    try:
        stage = Usd.Stage.Open(path)
    except Exception as e:
        log(f"FAILED to open {path}: {e}")
        continue

    log(f"\n{'='*60}")
    log(f"File: {path}")
    log(f"Up axis: {UsdGeom.GetStageUpAxis(stage)}")

    root = stage.GetDefaultPrim()
    if root:
        log(f"Default prim: {root.GetPath()}")
    else:
        log("No default prim!")
        continue

    def inspect(prim, depth=0):
        indent = "  " * depth
        xf = UsdGeom.Xformable(prim)
        if xf:
            ops = xf.GetOrderedXformOps()
            if ops:
                log(f"{indent}Xform on {prim.GetPath()}:")
                for o in ops:
                    log(f"{indent}  {o.GetOpName()} ({o.GetTypeName()}): {o.Get()}")

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            log(f"{indent}** ArticulationRootAPI on {prim.GetPath()}")

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            log(f"{indent}** RigidBodyAPI on {prim.GetPath()}")

        for child in prim.GetChildren():
            if depth < 3:
                inspect(child, depth + 1)

    inspect(root)

with open(out_path, "w") as f:
    f.write("\n".join(lines))

log(f"\nResults written to {out_path}")
sim.close()
