"""Trace the complete transform chain from /deer/{link}/visuals to the mesh data.

Opens deer_base.usd directly (the sublayer that contains visual geometry) to
inspect all intermediate Xform ops in the reference chain.
"""
from __future__ import annotations

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Sdf  # noqa: E402

BASE_USD = r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer\configuration\deer_base.usd"
LOG_PATH = r"C:\Users\nars7\Downloads\aniML_world\tools\visual_chain_report.txt"


def get_xform_ops(prim):
    if not prim.IsA(UsdGeom.Xformable):
        return []
    xf = UsdGeom.Xformable(prim)
    return [(op.GetOpName(), str(op.Get())) for op in xf.GetOrderedXformOps()]


def main():
    stage = Usd.Stage.Open(BASE_USD)
    lines = []

    lines.append("=== All prims in deer_base.usd with xform ops ===\n")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        typ = prim.GetTypeName()
        ops = get_xform_ops(prim)
        refs_info = ""

        refs = prim.GetMetadata("references")
        if refs:
            refs_info = f" REFS={refs}"

        if ops or "/visuals" in path or "/meshes" in path:
            lines.append(f"{path} [{typ}]{refs_info}")
            if ops:
                for name, val in ops:
                    lines.append(f"    {name} = {val}")
            else:
                lines.append(f"    (no xform ops)")

            children = [f"{c.GetName()}[{c.GetTypeName()}]" for c in prim.GetChildren()]
            if children:
                lines.append(f"    children: {children}")

    lines.append("\n\n=== Reference arcs on /deer/*/visuals ===\n")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.endswith("/visuals"):
            continue

        lines.append(f"\n{path}:")
        ops = get_xform_ops(prim)
        lines.append(f"  xform ops: {ops if ops else 'NONE (identity)'}")

        stack = prim.GetPrimStack()
        for spec in stack:
            lines.append(f"  spec layer: {spec.layer.identifier.split('/')[-1]}")
            spec_refs = spec.referenceList
            if spec_refs:
                for ref_item in spec_refs.prependedItems:
                    lines.append(f"    prepended ref: asset={ref_item.assetPath} prim={ref_item.primPath}")
                for ref_item in spec_refs.appendedItems:
                    lines.append(f"    appended ref: asset={ref_item.assetPath} prim={ref_item.primPath}")

    lines.append("\n\n=== /visuals scope prims ===\n")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/visuals"):
            continue
        ops = get_xform_ops(prim)
        lines.append(f"{path} [{prim.GetTypeName()}]")
        if ops:
            for name, val in ops:
                lines.append(f"    {name} = {val}")
        else:
            lines.append(f"    (no xform ops)")

        stack = prim.GetPrimStack()
        for spec in stack:
            refs = spec.referenceList
            if refs:
                for ref_item in refs.prependedItems:
                    lines.append(f"    ref: asset={ref_item.assetPath} prim={ref_item.primPath}")

    lines.append("\n\n=== /meshes scope prims ===\n")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/meshes"):
            continue
        ops = get_xform_ops(prim)
        lines.append(f"{path} [{prim.GetTypeName()}]")
        if ops:
            for name, val in ops:
                lines.append(f"    {name} = {val}")
        else:
            lines.append(f"    (no xform ops)")

    text = "\n".join(lines)
    print(text[:4000], flush=True)
    with open(LOG_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\n...full report at: {LOG_PATH}")

    app.close()


if __name__ == "__main__":
    main()
