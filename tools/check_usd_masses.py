from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("C:/Users/nars7/Downloads/aniML_world/deer_world.usd")
print("=== USD Mass values ===")
total = 0
for prim in stage.Traverse():
    api = UsdPhysics.MassAPI(prim)
    if api:
        attr = api.GetMassAttr()
        if attr and attr.HasValue():
            m = attr.Get()
            if m and m > 0:
                print(f"  {prim.GetPath()}: {m:.2f} kg")
                total += m
print(f"  Total: {total:.2f} kg")
app.close()
