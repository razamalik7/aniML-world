"""Fix visual orientation of the deer USD."""

import math
import shutil
import sys
import traceback
from pathlib import Path

print("SCRIPT START", flush=True)

try:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
except Exception as e:
    print(f"SimulationApp failed: {e}", flush=True)
    sys.exit(1)

print("SimulationApp ready", flush=True)

try:
    from pxr import Gf, Usd, UsdGeom, Vt
    print("pxr imported OK", flush=True)
except Exception as e:
    print(f"pxr import failed: {e}", flush=True)
    app.close()
    sys.exit(1)

ROOT = Path(r"c:\Users\nars7\Downloads\aniML_world")
USD_FILE = ROOT / "deer_standing_robot.usd"
DEER_BASE = ROOT / "assets" / "deer_urdf" / "deer" / "configuration" / "deer_base.usd"

ANGLE_DEG = 109.0
ANGLE_RAD = math.radians(ANGLE_DEG)
COS_A = math.cos(ANGLE_RAD)
SIN_A = math.sin(ANGLE_RAD)


def _rx(x, y, z):
    return (x, y * COS_A - z * SIN_A, y * SIN_A + z * COS_A)


try:
    # Step 1: Fix root orient
    print(f"\n[1/2] Fix root orient in {USD_FILE.name}", flush=True)
    stage = Usd.Stage.Open(str(USD_FILE))
    deer = stage.GetPrimAtPath("/deer")
    xf = UsdGeom.Xformable(deer)
    for op in xf.GetOrderedXformOps():
        if "orient" in op.GetName():
            print(f"  Old: {op.Get()}", flush=True)
            op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            print("  New: identity (1,0,0,0)", flush=True)
            break
    stage.GetRootLayer().Save()
    print(f"  Saved {USD_FILE.name}", flush=True)

    # Step 2: Rotate visual meshes
    print(f"\n[2/2] Rotate visual meshes in {DEER_BASE.name}", flush=True)
    print(f"  Path: {DEER_BASE}", flush=True)
    print(f"  Exists: {DEER_BASE.exists()}", flush=True)

    bak = DEER_BASE.parent / (DEER_BASE.stem + "_visual_bak.usd")
    if not bak.exists():
        shutil.copy2(DEER_BASE, bak)
        print(f"  Backed up -> {bak.name}", flush=True)

    stage2 = Usd.Stage.Open(str(DEER_BASE))
    print(f"  Stage opened: {stage2 is not None}", flush=True)

    count = 0
    for prim in stage2.Traverse():
        ps = str(prim.GetPath())
        if "/visuals/" not in ps:
            continue
        if prim.GetTypeName() != "Mesh":
            continue

        pts_attr = prim.GetAttribute("points")
        if not pts_attr:
            continue
        pts = pts_attr.Get()
        if not pts or len(pts) == 0:
            continue

        new_pts = Vt.Vec3fArray([Gf.Vec3f(*_rx(p[0], p[1], p[2])) for p in pts])
        pts_attr.Set(new_pts)

        nrm_attr = prim.GetAttribute("normals")
        if nrm_attr:
            nrm = nrm_attr.Get()
            if nrm and len(nrm) > 0:
                new_nrm = Vt.Vec3fArray([Gf.Vec3f(*_rx(n[0], n[1], n[2])) for n in nrm])
                nrm_attr.Set(new_nrm)

        ext_attr = prim.GetAttribute("extent")
        if ext_attr:
            ext = ext_attr.Get()
            if ext and len(ext) == 2:
                corners = [
                    _rx(ext[i][0], ext[j][1], ext[k][2])
                    for i in range(2) for j in range(2) for k in range(2)
                ]
                mn = [min(c[d] for c in corners) for d in range(3)]
                mx = [max(c[d] for c in corners) for d in range(3)]
                ext_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*mn), Gf.Vec3f(*mx)]))

        count += 1
        print(f"  Rotated: {ps}  ({len(pts)} verts)", flush=True)

    if count > 0:
        stage2.GetRootLayer().Save()
        print(f"  Saved {DEER_BASE.name} ({count} meshes)", flush=True)
    else:
        print("  WARNING: 0 visual meshes found!", flush=True)

    print("\nDONE - SUCCESS" if count > 0 else "\nDONE - NO MESHES FOUND", flush=True)

except Exception:
    traceback.print_exc()
    print("FAILED", flush=True)

app.close()
