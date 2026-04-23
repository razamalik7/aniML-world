from pxr import Usd
import sys

for usd_file in [
    r"C:\Users\nars7\Downloads\aniML_world\deer_world.usd",
    r"C:\Users\nars7\Downloads\aniML_world\deer_fixed_robot.usd",
]:
    print(f"\n===== {usd_file} =====")
    stage = Usd.Stage.Open(usd_file)
    content = stage.GetRootLayer().ExportToString()
    print(f"Layer length: {len(content)}")
    print(content[:1500])
    print("...")
    print("Prims:")
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")
        if prim.GetTypeName() == "PhysicsArticulationRootAPI" or hasattr(prim, 'GetMassAttr'):
            pass
