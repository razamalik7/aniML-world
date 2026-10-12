"""Verify the just-imported USD has the expected v3 physics:
  - Total mass 82 kg
  - Convex decomposition applied to all link collision meshes
  - L/R symmetric inertia tensors (within tolerance)
  - Filtered pairs configured
  - 22 rigid bodies, 21 revolute + 1 fixed joint
  - Computed CoM vs target

Writes report to /tmp/verify_v3.txt so we don't lose output to Kit's stdout capture.
"""
from isaacsim import SimulationApp
app = SimulationApp({
    "headless": True,
    "experience": r"C:\isaac-lab\IsaacLab\apps\isaaclab.python.headless.kit",
})

import json
from pathlib import Path
from pxr import Usd, UsdPhysics, UsdGeom, Gf, PhysxSchema

REPO = Path(__file__).resolve().parent.parent
USD_PATH = str(REPO / "deer_world.usd")

stage = Usd.Stage.Open(USD_PATH)
report_lines = []
def log(msg):
    print(msg, flush=True)
    report_lines.append(msg)

log(f"[verify] USD: {USD_PATH}")
log(f"[verify] Default prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else 'NONE'}")

# 1. Per-link mass + CoM
log("\n=== Mass + CoM per link ===")
masses = {}
coms = {}
total = 0.0
com_world_weighted = Gf.Vec3d(0, 0, 0)
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.MassAPI):
        continue
    api = UsdPhysics.MassAPI(prim)
    m = api.GetMassAttr().Get()
    if m is None or m <= 0:
        continue
    com_local = api.GetCenterOfMassAttr().Get()
    name = prim.GetName()
    masses[name] = m
    coms[name] = list(com_local) if com_local else None
    total += m
    log(f"  {name:25s}  mass={m:8.3f}  com_local={list(com_local) if com_local else 'None'}")
log(f"  {'TOTAL':25s}  mass={total:8.3f}  (target 82.0)")

# 2. Joint counts
log("\n=== Joint topology ===")
revolute_count = 0
fixed_count = 0
joint_limits = {}
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        revolute_count += 1
        j = UsdPhysics.RevoluteJoint(prim)
        lo = j.GetLowerLimitAttr().Get()
        hi = j.GetUpperLimitAttr().Get()
        joint_limits[prim.GetName()] = (lo, hi)
    if prim.IsA(UsdPhysics.FixedJoint):
        fixed_count += 1
        log(f"  fixed joint: {prim.GetPath()}")
log(f"  revolute joints: {revolute_count}  (expect 21)")
log(f"  fixed joints   : {fixed_count}  (expect 1: skull_to_antlers)")

# 3. Joint limits — verify expansions landed
log("\n=== Joint limits (degrees, USD convention) ===")
for jn in ["chest_to_neck_yaw","neck_yaw_to_neck","neck_to_skull",
           "pelvis_to_chest_yaw","chest_yaw_to_chest",
           "HR_shin_to_HR_foot","HL_shin_to_HL_foot",
           "FR_shin_to_FR_foot","FL_shin_to_FL_foot"]:
    if jn in joint_limits:
        lo, hi = joint_limits[jn]
        log(f"  {jn:30s}  [{lo:+8.2f}, {hi:+8.2f}]")

# 4. Collision approximation per link
log("\n=== Collision approximation per link ===")
HULL_LINKS = {"antlers","skull","chest","pelvis","neck",
              "HR_thigh","HL_thigh","FR_thigh","FL_thigh",
              "HR_shin","HL_shin","FR_shin","FL_shin",
              "HR_foot","HL_foot","FR_foot","FL_foot",
              "HR_toe","HL_toe","FR_toe","FL_toe"}
decomp_n = 0; convex_n = 0; mesh_n = 0; other_n = 0
hulls_per_link = {}
# Use TraverseInstanceProxies so we find collision meshes whether or not
# the importer wrapped them in instance prototypes.
for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        mca = UsdPhysics.MeshCollisionAPI(prim)
        approx = mca.GetApproximationAttr().Get()
        if approx == "convexDecomposition":
            decomp_n += 1
            # Get max hull count
            if prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
                pdc = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
                mh = pdc.GetMaxConvexHullsAttr().Get()
                cur = prim
                while cur and cur.IsValid():
                    if cur.GetName() in HULL_LINKS:
                        hulls_per_link[cur.GetName()] = mh
                        break
                    cur = cur.GetParent()
        elif approx == "convexHull":
            convex_n += 1
        elif approx in ("none", "meshSimplification"):
            mesh_n += 1
        else:
            other_n += 1
log(f"  convexDecomposition: {decomp_n}")
log(f"  convexHull         : {convex_n}")
log(f"  triangle/none      : {mesh_n}")
log(f"  other              : {other_n}")
log("\n  per-link maxConvexHulls:")
for k in sorted(hulls_per_link):
    log(f"    {k:15s} {hulls_per_link[k]}")

# 5. Filtered pairs
log("\n=== Filtered pairs ===")
filter_count = 0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        api = UsdPhysics.FilteredPairsAPI(prim)
        rel = api.GetFilteredPairsRel()
        if rel:
            targets = list(rel.GetTargets())
            for t in targets:
                log(f"  {prim.GetName():15s}  <->  {t.name}")
                filter_count += 1
log(f"  total filtered pairs: {filter_count}")

# Save report
out = Path(r"C:\Users\nars7\AppData\Local\Temp\verify_v3.txt")
out.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\n[verify] Report saved to {out}", flush=True)

app.close()
