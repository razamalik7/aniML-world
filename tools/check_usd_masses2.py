from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("C:/Users/nars7/Downloads/aniML_world/deer_world.usd")

out = []
out.append("=== USD Mass values (physics:mass attribute) ===")
total = 0
for prim in stage.Traverse():
    # Check MassAPI
    if prim.HasAPI(UsdPhysics.MassAPI):
        api = UsdPhysics.MassAPI(prim)
        attr = api.GetMassAttr()
        m = attr.Get() if attr else None
        out.append(f"  {prim.GetPath()}: {m} kg")
        if m:
            total += m
    elif prim.IsA(UsdPhysics.RigidBodyAPI):
        out.append(f"  {prim.GetPath()}: RigidBody but no MassAPI")

out.append(f"\nTotal: {total:.2f} kg")

with open("C:/Users/nars7/Downloads/aniML_world/tools/mass_check_out.txt", "w") as f:
    f.write("\n".join(out))

print("\n".join(out))
app.close()
