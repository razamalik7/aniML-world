"""URDF -> USD import for the Poly Art rig (runs in Isaac Sim's Python).

Applies, per import_config.json from the generator:
  * per-link convex DECOMPOSITION with maxConvexHulls tuned by contact-relevance
    (feet high, body low) -- collision detail where contacts happen,
  * self-collision ENABLED (legs/body can't pass through; the articulation
    auto-filters adjacent links sharing a joint),
  * visual stays the real mesh (the importer keeps the <visual> mesh as-is;
    only the <collision> mesh gets decomposed).

  /c/isaac-lab/env_isaaclab/Scripts/python.exe tools/rig_pipeline/import_urdf.py \
      --urdf assets/deer_polyart_urdf/deer_polyart.urdf \
      --usd  assets/deer_polyart_urdf/deer_polyart.usd \
      --config assets/deer_polyart_urdf/import_config.json
"""
from isaacsim import SimulationApp
app = SimulationApp({
    "headless": True,
    "experience": r"C:\isaac-lab\IsaacLab\apps\isaaclab.python.headless.kit",
})

import omni.kit.app
ext_mgr = omni.kit.app.get_app().get_extension_manager()
for ext_id in ("isaacsim.asset.importer.urdf",):
    if not ext_mgr.is_extension_enabled(ext_id):
        ext_mgr.set_extension_enabled_immediate(ext_id, True)

import argparse
import json
from pathlib import Path
import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf

p = argparse.ArgumentParser()
p.add_argument("--urdf", required=True)
p.add_argument("--usd", required=True)
p.add_argument("--config", required=True)
args, _ = p.parse_known_args()
urdf_path = str(Path(args.urdf).resolve())
usd_path = str(Path(args.usd).resolve())
cfg = json.load(open(args.config))
budget = cfg["hull_budget"]                     # {safe_link_name: maxConvexHulls}
filter_pairs = cfg.get("filter_pairs", [])
print(f"[import] URDF: {urdf_path}")
print(f"[import] hull budgets for {len(budget)} links, total {sum(budget.values())} hulls")

status, importer = omni.kit.commands.execute("URDFCreateImportConfig")
importer.set_merge_fixed_joints(False)
importer.set_fix_base(False)
importer.set_import_inertia_tensor(True)
importer.set_distance_scale(1.0)
importer.set_density(0.0)
importer.set_convex_decomp(True)
importer.set_self_collision(True)               # legs/body collide; adjacent auto-filtered
for setter in ("set_make_instanceable", "set_create_instanceable_asset", "set_instanceable"):
    if hasattr(importer, setter):
        getattr(importer, setter)(False); break
importer.set_default_drive_type(_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION)
importer.set_default_drive_strength(1047.19751)
importer.set_default_position_drive_damping(52.35988)

omni.kit.commands.execute("URDFParseAndImportFile",
                          urdf_path=urdf_path, import_config=importer, dest_path=usd_path)

from pxr import Usd, UsdPhysics, PhysxSchema
stage = Usd.Stage.Open(usd_path)

# find the articulation root prim
root = None
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        root = prim; break
if root is None:
    root = stage.GetDefaultPrim()
print(f"[import] articulation root: {root.GetPath()}")
stage.SetDefaultPrim(root if root.GetParent().IsPseudoRoot() else stage.GetPrimAtPath(str(root.GetPath()).rsplit('/', 1)[0] or '/'))

# enable self-collision on the articulation
pa = PhysxSchema.PhysxArticulationAPI.Apply(root)
pa.CreateEnabledSelfCollisionsAttr(True)
print("[import] enabledSelfCollisions = True")

# per-link convex decomposition (walk up from each collision prim to its link name)
def link_of(prim):
    cur = prim
    while cur and cur.IsValid():
        if cur.GetName() in budget:
            return cur.GetName()
        cur = cur.GetParent()
    return None

applied = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    link = link_of(prim)
    if link is None:
        continue
    n = budget[link]
    mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mca.GetApproximationAttr().Set("convexDecomposition")
    pdc = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    pdc.CreateMaxConvexHullsAttr(n)
    pdc.CreateVoxelResolutionAttr(200000)
    pdc.CreateShrinkWrapAttr(True)
    applied += 1
print(f"[import] applied convex decomposition to {applied} collision meshes")

# explicit self-collision filter pairs (antler<->body deferred; usually empty now)
def find_link_prim(name):
    for prim in stage.Traverse():
        if prim.GetName() == name and prim.GetParent() == root:
            return prim
    return None
nf = 0
for a, b in filter_pairs:
    pa_, pb_ = find_link_prim(a), find_link_prim(b)
    if pa_ and pb_:
        api = UsdPhysics.FilteredPairsAPI.Apply(pa_)
        rel = api.CreateFilteredPairsRel()
        t = list(rel.GetTargets())
        if pb_.GetPath() not in t:
            t.append(pb_.GetPath()); rel.SetTargets(t)
        nf += 1
print(f"[import] added {nf} explicit filter pairs")

stage.Save()

# audit
rb = sum(1 for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI))
jt = sum(1 for pr in stage.Traverse() if pr.IsA(UsdPhysics.RevoluteJoint))
cd = sum(1 for pr in stage.Traverse()
         if pr.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI))
print(f"[import] SAVED {usd_path}")
print(f"[import] RIGID_BODIES:{rb}  REVOLUTE_JOINTS:{jt}  DECOMP_MESHES:{cd}")
app.close()
