"""Inspect visual mesh prims in the USD to understand their structure."""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402

usd_path = sys.argv[1] if len(sys.argv) > 1 else "deer_fixed_robot.usd"
stage = Usd.Stage.Open(usd_path)

for prim in stage.Traverse():
    path = str(prim.GetPath())
    ptype = prim.GetTypeName()
    if ptype == "Mesh":
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        norms = UsdGeom.Mesh(prim).GetNormalsAttr().Get()
        fvc = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
        fvi = UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()
        n_pts = len(pts) if pts else 0
        n_norms = len(norms) if norms else 0
        n_faces = len(fvc) if fvc else 0
        print(f"{path}  pts={n_pts}  norms={n_norms}  faces={n_faces}", flush=True)
        if n_pts > 0:
            print(f"  first pt: {pts[0]}", flush=True)
            zs = [p[2] for p in pts]
            print(f"  z range: [{min(zs):.4f}, {max(zs):.4f}]", flush=True)

app.close()
