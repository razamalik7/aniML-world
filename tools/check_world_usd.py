"""Check deer_world.usd root prim transform and defaultPrim."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, Gf

LOG = open(r"C:\Users\nars7\Downloads\aniML_world\tools\world_usd_check.txt", "w")
def log(s): LOG.write(s+"\n"); LOG.flush()

stage = Usd.Stage.Open(r"C:\Users\nars7\Downloads\aniML_world\deer_world.usd")
log(f"defaultPrim: {stage.GetDefaultPrim().GetPath()}")

log("\n=== All prims depth<=2 with xformOps ===")
for prim in stage.Traverse():
    depth = prim.GetPath().pathString.count('/')
    if depth > 2: continue
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps() if xf else []
    log(f"  {prim.GetPath()} [{prim.GetTypeName()}]")
    for o in ops:
        log(f"    {o.GetOpName()} = {o.Get()}")

LOG.close()
app.close()
