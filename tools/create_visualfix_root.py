"""Create the deer_visualfix2_robot.usd root file.

Copies the structure of deer_standing_robot.usd but references the
deer_visualfix hierarchy instead of deer for the visual mesh fix.
"""
from __future__ import annotations

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, Sdf, UsdGeom, UsdPhysics  # noqa: E402

SRC = r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd"
DST = r"C:\Users\nars7\Downloads\aniML_world\deer_visualfix2_robot.usd"


def main():
    src_stage = Usd.Stage.Open(SRC)
    src_layer = src_stage.GetRootLayer()

    dst_layer = Sdf.Layer.CreateNew(DST)

    Sdf.CopySpec(src_layer, Sdf.Path.absoluteRootPath, dst_layer, Sdf.Path.absoluteRootPath)

    txt = dst_layer.ExportToString()
    txt = txt.replace(
        "deer_urdf/deer/",
        "deer_urdf/deer_visualfix/"
    ).replace(
        "@./assets/deer_urdf/deer/",
        "@./assets/deer_urdf/deer_visualfix/"
    )

    dst_layer.ImportFromString(txt)
    dst_layer.Save()

    test_stage = Usd.Stage.Open(DST)
    deer = test_stage.GetPrimAtPath("/deer")
    if deer:
        print(f"Root prim /deer exists: {deer.IsValid()}")
        print(f"Type: {deer.GetTypeName()}")
        children = [c.GetName() for c in deer.GetChildren()]
        print(f"Children: {children[:5]}...")
    else:
        print("WARNING: /deer prim not found!")

    used = test_stage.GetUsedLayers()
    for l in used:
        print(f"  Layer: {l.identifier}")

    print(f"\nCreated: {DST}")
    app.close()


if __name__ == "__main__":
    main()
