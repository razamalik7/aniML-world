"""Create deer_visualfix2_robot.usd that references the visualfix hierarchy.

Opens the original deer_standing_robot.usd, reads its structure, and creates
a new root USD pointing to the deer_visualfix configuration layers.
"""
from __future__ import annotations
from pathlib import Path

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

SRC = r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd"
DST = r"C:\Users\nars7\Downloads\aniML_world\deer_visualfix2_robot.usd"


def main():
    src_stage = Usd.Stage.Open(SRC)
    src_layer = src_stage.GetRootLayer()

    print(f"Source root layer: {src_layer.identifier}")
    print(f"Source sublayers: {src_layer.subLayerPaths}")

    src_deer = src_stage.GetPrimAtPath("/deer")
    src_xf = UsdGeom.Xformable(src_deer)
    ops = [(op.GetOpName(), op.Get()) for op in src_xf.GetOrderedXformOps()]
    print(f"Root /deer xform ops: {ops}")

    print("\nSource layer text (first 2000 chars):")
    txt = src_layer.ExportToString()
    print(txt[:2000])

    deer_spec = src_layer.GetPrimAtPath("/deer")
    if deer_spec:
        print(f"\n/deer spec type: {deer_spec.typeName}")
        print(f"/deer spec refs: {deer_spec.referenceList}")
        print(f"/deer spec payloads: {deer_spec.payloadList}")

        for attr in deer_spec.attributes:
            print(f"  attr: {attr}")
        for prop in deer_spec.properties:
            print(f"  prop: {prop}")

    new_txt = txt.replace(
        "assets/deer_urdf/deer/",
        "assets/deer_urdf/deer_visualfix/"
    )

    if "deer_visualfix" not in new_txt:
        new_txt = txt.replace(
            "deer_urdf/deer/",
            "deer_urdf/deer_visualfix/"
        )

    if "deer_visualfix" not in new_txt:
        print("\nWARNING: Could not find path to replace in USDA text!")
        print("Checking for other path formats...")
        for line in txt.split("\n"):
            if "deer" in line.lower() and ("asset" in line.lower() or "ref" in line.lower() or "@" in line):
                print(f"  {line.strip()}")

    dst_layer = Sdf.Layer.CreateNew(DST)
    result = dst_layer.ImportFromString(new_txt)
    print(f"\nImport result: {result}")
    dst_layer.Save()
    print(f"Saved: {DST}")
    print(f"Size: {Path(DST).stat().st_size} bytes")

    verify = Usd.Stage.Open(DST)
    v_deer = verify.GetPrimAtPath("/deer")
    if v_deer:
        print(f"Verify: /deer valid={v_deer.IsValid()}")
        used = verify.GetUsedLayers()
        for l in used:
            print(f"  Layer: {l.identifier}")
    else:
        print("WARNING: /deer not found in output!")

    app.close()


if __name__ == "__main__":
    main()
