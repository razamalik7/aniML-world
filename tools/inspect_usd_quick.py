"""Quick USD physics inspector that works with Isaac Sim's Python."""
import sys
import os

usd_path = sys.argv[1] if len(sys.argv) > 1 else None
if not usd_path:
    print("Usage: python inspect_usd_quick.py <path_to_usd>", flush=True)
    sys.exit(1)

print(f"[INSPECT] Target: {usd_path}", flush=True)
print(f"[INSPECT] Exists: {os.path.isfile(usd_path)}", flush=True)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

try:
    from pxr import PhysxSchema, Usd, UsdPhysics

    print(f"[INSPECT] Opening stage...", flush=True)
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[INSPECT] ERROR: could not open stage", flush=True)
        app.close()
        sys.exit(1)

    dp = stage.GetDefaultPrim()
    print(f"[INSPECT] Default prim: {dp.GetPath() if dp and dp.IsValid() else '(none)'}", flush=True)

    root_children = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
    print(f"[INSPECT] Root children: {root_children}", flush=True)

    found_any = False
    for prim in stage.Traverse():
        apis = []
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            apis.append("ArticulationRoot")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RigidBody")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            apis.append("Collision")
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            apis.append("ContactReport")
        is_joint = prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        if is_joint:
            apis.append("Joint")
        if apis:
            found_any = True
            print(f"[INSPECT]   {str(prim.GetPath()):60s}  {', '.join(apis)}", flush=True)

    if not found_any:
        print("[INSPECT] WARNING: No physics APIs found on any prim!", flush=True)
        print("[INSPECT] Listing all prims:", flush=True)
        for prim in stage.Traverse():
            print(f"[INSPECT]   {prim.GetPath()}  type={prim.GetTypeName()}", flush=True)

    print("[INSPECT] Done.", flush=True)
except Exception as ex:
    print(f"[INSPECT] EXCEPTION: {ex}", flush=True)
    import traceback
    traceback.print_exc()

app.close()
