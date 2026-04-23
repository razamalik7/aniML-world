"""Check USD orientation WITHOUT launching full Isaac Sim."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, Gf

import sys
LOG = r"C:\Users\nars7\Downloads\aniML_world\tools\usd_check_output.txt"
stage = Usd.Stage.Open(r"C:\Users\nars7\Downloads\aniML_world\deer_clean.usd")
out = open(LOG, 'w')

def log(msg):
    out.write(msg + '\n')
    out.flush()

log("\n=== Top-level prims ===")
for prim in stage.Traverse():
    depth = prim.GetPath().pathString.count('/')
    if depth > 2:
        continue
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        ops = xformable.GetOrderedXformOps()
        log(f"  {prim.GetPath()}: {prim.GetTypeName()}")
        for o in ops:
            log(f"    {o.GetOpName()} = {o.Get()}")

log("\n=== Pelvis prim ===")
for prim in stage.Traverse():
    if 'pelvis' in prim.GetPath().pathString.lower() and prim.GetTypeName() in ('Xform', 'Mesh'):
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            ops = xformable.GetOrderedXformOps()
            log(f"  {prim.GetPath()}: {prim.GetTypeName()}")
            for o in ops:
                log(f"    {o.GetOpName()} = {o.Get()}")

log("\n=== All joints (depth<=4) ===")
for prim in stage.Traverse():
    depth = prim.GetPath().pathString.count('/')
    if depth > 4:
        continue
    if prim.GetTypeName() in ('PhysicsRevoluteJoint', 'PhysicsFixedJoint'):
        log(f"  {prim.GetPath()}")
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            for o in xformable.GetOrderedXformOps():
                log(f"    {o.GetOpName()} = {o.Get()}")

out.close()
app.close()
