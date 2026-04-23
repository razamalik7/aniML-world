"""Inspect the root prim transforms in deer_clean.usd to diagnose orientation."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf

stage = Usd.Stage.Open(r"C:\Users\nars7\Downloads\aniML_world\deer_clean.usd")

print("=== Root-level prims and their xform ops ===")
for prim in stage.Traverse():
    depth = prim.GetPath().pathString.count('/')
    if depth > 3:
        continue
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps() if xformable else []
    op_strs = []
    for o in ops:
        val = o.Get()
        op_strs.append(f"{o.GetOpName()}={val}")
    indent = "  " * (depth - 1)
    print(f"{indent}{prim.GetPath()}: {prim.GetTypeName()}")
    for s in op_strs:
        print(f"{indent}  -> {s}")

print("\n=== Looking for pelvis link world transform ===")
# Find the pelvis prim
for prim in stage.Traverse():
    if 'pelvis' in prim.GetPath().pathString.lower() and prim.GetTypeName() in ('Xform', 'Mesh'):
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            ops = xformable.GetOrderedXformOps()
            print(f"{prim.GetPath()}: {prim.GetTypeName()}")
            for o in ops:
                print(f"  {o.GetOpName()} = {o.Get()}")

app.close()
