"""Debug: find toe collision meshes and report what's keeping per-link tuning from applying."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "experience": r"C:\isaac-lab\IsaacLab\apps\isaaclab.python.headless.kit"})

from pxr import Usd, UsdPhysics, PhysxSchema
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
USD_PATH = str(REPO / "deer_world.usd")
stage = Usd.Stage.Open(USD_PATH)

with open(r"C:\Users\nars7\AppData\Local\Temp\toe_debug.txt", "w") as f:
    # Find all prims with CollisionAPI, classify
    f.write("=== All CollisionAPI prims (via TraverseInstanceProxies) ===\n")
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # walk up
        cur = prim; chain = []
        while cur and cur.IsValid() and cur.GetPath().pathString != "/":
            chain.append(cur.GetName())
            cur = cur.GetParent()
        has_decomp = prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
        f.write(f"  {prim.GetPath()}\n")
        f.write(f"    chain={list(reversed(chain))}\n")
        f.write(f"    is_proxy={prim.IsInstanceProxy()}  has_decomp_api={has_decomp}\n")

    # Same with stage.Traverse() (no proxies)
    f.write("\n=== With plain stage.Traverse() ===\n")
    n = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        n += 1
        f.write(f"  {prim.GetPath()}  is_proxy={prim.IsInstanceProxy()}\n")
    f.write(f"  total: {n}\n")

    # Look specifically at toe link's children
    f.write("\n=== /deer/HR_toe descendants ===\n")
    toe = stage.GetPrimAtPath("/deer/HR_toe")
    if toe:
        for desc in Usd.PrimRange(toe):
            apis = list(desc.GetAppliedSchemas())
            f.write(f"  {desc.GetPath()}  type={desc.GetTypeName()}  apis={apis}\n")

print("[debug] dump done", flush=True)
app.close()
