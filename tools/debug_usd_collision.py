"""Check whether convex decomp was applied to collision meshes during import."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "experience": r"C:\isaac-lab\IsaacLab\apps\isaaclab.python.headless.kit"})

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
USD_PATH = str(REPO / "deer_world.usd")
stage = Usd.Stage.Open(USD_PATH)

with open(r"C:\Users\nars7\AppData\Local\Temp\decomp_check.txt", "w") as f:
    found = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            found += 1
            mca = UsdPhysics.MeshCollisionAPI(prim)
            approx = mca.GetApproximationAttr().Get()
            has_decomp = prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
            max_hulls = None
            if has_decomp:
                pdc = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
                max_hulls = pdc.GetMaxConvexHullsAttr().Get()
            f.write(f"{prim.GetPath()}  approx={approx}  has_decomp_api={has_decomp}  max_hulls={max_hulls}\n")
    f.write(f"\nFound {found} prims with MeshCollisionAPI\n")

print("[debug] dump complete", flush=True)
app.close()
