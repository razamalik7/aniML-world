"""Inspect the root prim's xform ops and all link positions in the USD."""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

usd_path = sys.argv[1]
print(f"Inspecting: {usd_path}", flush=True)

omni.usd.get_context().open_stage(usd_path)
for _ in range(20):
    app.update()

stage = omni.usd.get_context().get_stage()
root = stage.GetDefaultPrim()

print(f"\nRoot prim: {root.GetPath()}", flush=True)
xform = UsdGeom.Xformable(root)
ops = xform.GetOrderedXformOps()
print(f"Xform ops ({len(ops)}):", flush=True)
for op in ops:
    print(f"  {op.GetOpName()} = {op.Get()}", flush=True)

print("\n--- Link xform ops ---", flush=True)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if prim.GetTypeName() in ("Xform",) and path.count("/") == 2:
        xf = UsdGeom.Xformable(prim)
        link_ops = xf.GetOrderedXformOps()
        if link_ops:
            print(f"\n{path}:", flush=True)
            for op in link_ops:
                print(f"  {op.GetOpName()} = {op.Get()}", flush=True)

print("\n--- Joint prims ---", flush=True)
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsRevoluteJoint":
        print(f"  {prim.GetPath()}", flush=True)
        for attr_name in ["physics:localPos0", "physics:localPos1",
                          "physics:localRot0", "physics:localRot1",
                          "physics:axis"]:
            attr = prim.GetAttribute(attr_name)
            if attr and attr.HasValue():
                print(f"    {attr_name} = {attr.Get()}", flush=True)

app.close()
