"""Deep inspection of deer_standing_robot.usd — find where mesh data lives."""
from __future__ import annotations

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Sdf  # noqa: E402

USD_PATH = r"C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd"
LOG_PATH = r"C:\Users\nars7\Downloads\aniML_world\tools\usd_deep_report.txt"


def main():
    lines = []

    stage = Usd.Stage.Open(USD_PATH)
    root_layer = stage.GetRootLayer()

    lines.append(f"=== Root layer: {root_layer.identifier}")
    lines.append(f"Sublayers: {root_layer.subLayerPaths}")

    lines.append("\n=== All prims (composed stage) ===")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        typ = prim.GetTypeName()
        refs = prim.GetReferences()
        has_refs = bool(prim.GetPrimStack())

        children = [c.GetName() for c in prim.GetChildren()]
        xf = ""
        if prim.IsA(UsdGeom.Xformable):
            xfable = UsdGeom.Xformable(prim)
            ops = [(op.GetOpName(), str(op.Get())) for op in xfable.GetOrderedXformOps()]
            if ops:
                xf = f" xform={ops}"

        mesh_info = ""
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = m.GetPointsAttr().Get()
            mesh_info = f" MESH(pts={len(pts) if pts else 0})"

        spec_layers = []
        for spec in prim.GetPrimStack():
            spec_layers.append(spec.layer.identifier.split("/")[-1])

        lines.append(f"{path} [{typ}]{mesh_info}{xf}")
        if children:
            lines.append(f"  children: {children}")
        if spec_layers:
            lines.append(f"  layers: {spec_layers}")

    lines.append("\n=== Checking sublayer files ===")
    all_layers = stage.GetUsedLayers()
    for layer in all_layers:
        lines.append(f"\n--- Layer: {layer.identifier} ---")
        for prim_path in layer.rootPrims:
            prim_spec = layer.GetPrimAtPath(prim_path.path)
            lines.append(f"  {prim_path.path} type={prim_spec.typeName if prim_spec else '?'}")
            if prim_spec:
                for child in prim_spec.nameChildren:
                    lines.append(f"    {child.path} type={child.typeName}")
                    for gchild in child.nameChildren:
                        lines.append(f"      {gchild.path} type={gchild.typeName}")
                        for ggchild in gchild.nameChildren:
                            lines.append(f"        {ggchild.path} type={ggchild.typeName}")

    text = "\n".join(lines)
    print(text[:5000], flush=True)
    with open(LOG_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\n...saved full report to: {LOG_PATH}")

    app.close()


if __name__ == "__main__":
    main()
