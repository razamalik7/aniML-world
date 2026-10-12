"""Apply per-link convex decomposition to an already-imported composed USD.

The URDF importer produces an instanced, composed USD (base/physics/... layers).
Instance proxies are read-only, so we un-instance first (flattens the prototype
into the prim, making per-link collision APIs writable) -- the pattern the old
import_world_urdf_v3.py used. Then set convexDecomposition + per-link maxConvexHulls.

  /c/isaac-lab/env_isaaclab/Scripts/python.exe tools/rig_pipeline/apply_collision.py \
      --usd assets/deer_polyart_urdf/deer_polyart.usd \
      --config assets/deer_polyart_urdf/import_config.json
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True,
                     "experience": r"C:\isaac-lab\IsaacLab\apps\isaaclab.python.headless.kit"})
import json, argparse, sys
from pxr import Usd, UsdPhysics, PhysxSchema

p = argparse.ArgumentParser()
p.add_argument("--usd", required=True)
p.add_argument("--config", required=True)
a, _ = p.parse_known_args()
budget = json.load(open(a.config))["hull_budget"]
stage = Usd.Stage.Open(a.usd)

# Step A: un-instance so collision prims become directly editable
un = 0
for prim in stage.Traverse():
    if prim.IsInstance():
        prim.SetInstanceable(False); un += 1
sys.stdout.write(f"UNINSTANCED:{un}\n"); sys.stdout.flush()

# Step B: per-link convex decomposition
def link_of(prim):
    cur = prim
    while cur and cur.IsValid():
        if cur.GetName() in budget:
            return cur.GetName()
        cur = cur.GetParent()
    return None

applied = 0; per_link = {}
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    link = link_of(prim)
    if link is None:
        continue
    mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mca.GetApproximationAttr().Set("convexDecomposition")
    pdc = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    pdc.CreateMaxConvexHullsAttr(int(budget[link]))
    pdc.CreateVoxelResolutionAttr(200000)
    pdc.CreateShrinkWrapAttr(True)
    applied += 1; per_link[link] = per_link.get(link, 0) + 1
sys.stdout.write(f"APPLIED_DECOMP:{applied} over {len(per_link)} links\n"); sys.stdout.flush()

# Step C: explicit self-collision filter pairs (2-DOF parent<->child halves).
filter_pairs = json.load(open(a.config)).get("filter_pairs", [])
art = next((p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)), None)
deer = art.GetParent() if art else stage.GetDefaultPrim()
def find_link(name):
    p = deer.GetChild(name)
    return p if (p and p.IsValid()) else None
nf = 0
for aname, bname in filter_pairs:
    pa, pb = find_link(aname), find_link(bname)
    if pa and pb:
        api = UsdPhysics.FilteredPairsAPI.Apply(pa)
        rel = api.CreateFilteredPairsRel()
        tgt = list(rel.GetTargets())
        if pb.GetPath() not in tgt:
            tgt.append(pb.GetPath()); rel.SetTargets(tgt)
        nf += 1
    else:
        sys.stdout.write(f"  filter pair not found: {aname},{bname}\n")
sys.stdout.write(f"APPLIED_FILTERS:{nf}/{len(filter_pairs)}\n"); sys.stdout.flush()

stage.Save()

# verify from a freshly re-opened stage
v = Usd.Stage.Open(a.usd)
cd = sum(1 for pr in v.Traverse() if pr.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI))
apx = set()
for pr in v.Traverse():
    at = pr.GetAttribute('physics:approximation')
    if at and at.Get(): apx.add(at.Get())
sc = None
for pr in v.Traverse():
    at = pr.GetAttribute('physxArticulation:enabledSelfCollisions')
    if at and at.Get() is not None: sc = at.Get()
sys.stdout.write(f"VERIFY decomp_meshes={cd} approximations={apx} self_collision={sc}\n"); sys.stdout.flush()
app.close()
