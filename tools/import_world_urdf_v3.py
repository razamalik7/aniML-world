"""Regenerate deer_world.usd from the (widened) URDF.

Adapted from import_world_urdf_v2.py to use the current Projects layout
instead of the legacy Downloads layout. Runs in Isaac Sim's bundled Python:

    /c/isaac-lab/env_isaaclab/Scripts/python.exe \
        tools/import_world_urdf_v3.py [--urdf <path>] [--usd <path>]

Default IO:
  in : <repo>/assets/deer_urdf/deer_world.urdf
  out: <repo>/deer_world.usd
"""

from isaacsim import SimulationApp
# Use Isaac Lab's minimal headless experience to avoid the omni.hydra.rtx
# scenerenderer crash that hits us on RTX 50-series + Isaac Sim 5.1.
# Default SimulationApp({"headless":True}) loads a kit experience that
# initializes the RTX scene renderer even though we don't render — and
# crashes in carbOnPluginStartup on this hardware. The minimal kit doesn't
# include the URDF importer extension, so we enable it manually below.
app = SimulationApp({
    "headless": True,
    "experience": r"C:\isaac-lab\IsaacLab\apps\isaaclab.python.headless.kit",
})  # NOQA: must come before isaaclab imports

# Enable the URDF importer extension (not in the minimal kit by default).
import omni.kit.app
ext_mgr = omni.kit.app.get_app().get_extension_manager()
for ext_id in ("isaacsim.asset.importer.urdf",):
    if not ext_mgr.is_extension_enabled(ext_id):
        ext_mgr.set_extension_enabled_immediate(ext_id, True)

import argparse
from pathlib import Path

import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URDF = REPO_ROOT / "assets" / "deer_urdf" / "deer_world.urdf"
DEFAULT_USD = REPO_ROOT / "deer_world.usd"

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=str(DEFAULT_URDF))
parser.add_argument("--usd",  default=str(DEFAULT_USD))
args, _ = parser.parse_known_args()

urdf_path = str(Path(args.urdf).resolve())
usd_path  = str(Path(args.usd).resolve())
print(f"[import] URDF in : {urdf_path}")
print(f"[import] USD out : {usd_path}")

status, importer = omni.kit.commands.execute("URDFCreateImportConfig")
importer.set_merge_fixed_joints(False)   # keep antler as separate body (fixed joint)
importer.set_fix_base(False)
importer.set_import_inertia_tensor(True)
importer.set_distance_scale(1.0)
importer.set_density(0.0)
# v3.1: enable convex decomposition globally. Per-link hull counts are tuned
# post-import (antlers get high hull count for tine fidelity, limbs get few).
importer.set_convex_decomp(True)
importer.set_self_collision(True)
# v3.2: disable instanceable-asset generation. The default importer wraps
# each link's mesh in instance prototypes for memory deduplication. That blocks
# per-prim API writes (instance proxies aren't directly writable). Disabling
# costs a small amount of GPU memory but lets us tune per-link
# maxConvexHulls — a much bigger memory + speed win.
if hasattr(importer, "set_make_instanceable"):
    importer.set_make_instanceable(False)
elif hasattr(importer, "set_create_instanceable_asset"):
    importer.set_create_instanceable_asset(False)
elif hasattr(importer, "set_instanceable"):
    importer.set_instanceable(False)
else:
    # Older API: write the underlying setting directly
    try:
        importer.make_instanceable = False
    except AttributeError:
        print("[import] WARNING: could not disable instanceable assets — per-link tuning may not apply")
importer.set_default_drive_type(_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION)
importer.set_default_drive_strength(1047.19751)
importer.set_default_position_drive_damping(52.35988)

omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=importer,
    dest_path=usd_path,
)

from pxr import Usd, UsdPhysics, Sdf
# PhysxSchema is loaded via the Isaac Sim app extension cache; import after app boot.
from pxr import PhysxSchema
stage = Usd.Stage.Open(usd_path)

# Set defaultPrim to /deer so Isaac Lab references resolve correctly
root_prim = stage.GetPrimAtPath("/deer")
if root_prim:
    stage.SetDefaultPrim(root_prim)
    print(f"[import] Set defaultPrim to /deer")

# ============================================================================
# Per-link convex decomposition tuning. With instanceable=False above, the
# collision mesh prims are directly writable, so we can apply
# PhysxConvexDecompositionCollisionAPI per link with link-specific hull counts.
#
# Hull counts chosen to preserve gameplay-relevant geometry:
#   - Antlers: 24 — every tine + connecting beam; sparring tine-on-tine fidelity
#   - Skull: 6 — cranium + jaw + snout cavity + eye sockets (concave)
#   - Chest: 6 — ribcage shape, brisket, shoulder pockets
#   - Pelvis: 4 — hip socket curvature
#   - Neck: 3 — sausage shape with slight curve
#   - Limbs: 2 each — above-knee + below-knee bulge (essentially convex anyway)
#
# Total hulls: 24 + 6 + 6 + 4 + 3 + 12*2 = 67 per deer
# vs default 32-each = 17*32 = 544 per deer — ~8x fewer collision pieces.
# Same convex-decomp approximation method, just sized appropriately.
# ============================================================================
HULL_COUNTS = {
    "antlers":  24,
    "skull":    6,
    "chest":    6,
    "pelvis":   4,
    "neck":     3,
    "HR_thigh": 2, "HL_thigh": 2, "FR_thigh": 2, "FL_thigh": 2,
    "HR_shin":  2, "HL_shin":  2, "FR_shin":  2, "FL_shin":  2,
    "HR_foot":  2, "HL_foot":  2, "FR_foot":  2, "FL_foot":  2,
    # v4 toes (hooves) — small simple shapes
    "HR_toe":   3, "HL_toe":   3, "FR_toe":   3, "FL_toe":   3,
}

VOXEL_RESOLUTION = 200_000   # 200k voxels for decomposition fidelity
SHRINK_WRAP = True            # project hulls onto mesh surface

def _find_link_name(prim):
    """Walk up parents until we find a prim whose name matches a known link."""
    cur = prim
    while cur and cur.IsValid():
        nm = cur.GetName()
        if nm in HULL_COUNTS:
            return nm
        cur = cur.GetParent()
    return None

# Step A: un-instance any /deer/<link>/{visuals,collisions} that the importer
# made instanceable. Instance proxies are read-only via the standard API; we
# need editable prims to apply per-link convex-decomp APIs. Setting
# instanceable=False on the instance root flattens its prototype subtree into
# the prim itself, making descendants directly editable.
uninstanced = 0
for prim in stage.Traverse():
    if prim.IsInstance():
        prim.SetInstanceable(False)
        uninstanced += 1
print(f"[import] Un-instanced {uninstanced} instance prims (now directly editable)")

# Step B: now that meshes are real prims under /deer/.../collisions/.../mesh,
# apply per-link convex-decomp APIs. Use plain Traverse() since prims are
# no longer instance proxies.
decomp_count = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    link_name = _find_link_name(prim)
    if link_name is None:
        continue
    n_hulls = HULL_COUNTS[link_name]
    mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mca.GetApproximationAttr().Set("convexDecomposition")
    pdc = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    pdc.CreateMaxConvexHullsAttr(n_hulls)
    pdc.CreateVoxelResolutionAttr(VOXEL_RESOLUTION)
    pdc.CreateShrinkWrapAttr(SHRINK_WRAP)
    decomp_count += 1
    print(f"[import]   {link_name:15s}  maxConvexHulls={n_hulls}")
print(f"[import] Applied per-link convex decomp to {decomp_count} collision meshes")

# ============================================================================
# Self-collision filter pairs.
# `enabled_self_collisions=True` allows ALL link-link collisions; we
# explicitly exclude pairs that produce noisy/non-physical contacts.
# ============================================================================
FILTERED_PAIRS = [
    # Antler ↔ neck (sharp head turn would clip antler tines into neck mesh)
    ("antlers", "neck"),
    # Antler ↔ chest (deep nod would clip antlers into chest)
    ("antlers", "chest"),
    # Adjacent-leg same-side cross-collisions cause jitter at extreme yoga poses
    # (no benefit — real legs can't physically reach). Same-side hind-front:
    ("HR_thigh", "FR_thigh"),
    ("HL_thigh", "FL_thigh"),
    # Hip abd ↔ thigh of same leg (parent-child but importer sometimes emits)
    # Default behavior usually filters; explicit to be safe.
]

def _add_filter_pair(prim_a, prim_b):
    """Apply UsdPhysics.FilteredPairsAPI to prim_a, adding prim_b to its filter list.
    The FilteredPairs rel is one-way at the schema level but PhysX treats the
    pairing as symmetric (no need to also filter from prim_b)."""
    api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
    rel = api.CreateFilteredPairsRel()
    targets = list(rel.GetTargets())
    path_b = prim_b.GetPath()
    if path_b not in targets:
        targets.append(path_b)
        rel.SetTargets(targets)

# Locate link prims by name under /deer
def find_link_prim(name: str):
    target_path = f"/deer/{name}"
    p = stage.GetPrimAtPath(target_path)
    return p if p and p.IsValid() else None

filter_count = 0
for a_name, b_name in FILTERED_PAIRS:
    a = find_link_prim(a_name)
    b = find_link_prim(b_name)
    if not (a and b):
        print(f"[import]   skipping filter pair ({a_name}, {b_name}): prim missing")
        continue
    _add_filter_pair(a, b)
    filter_count += 1
print(f"[import] Added {filter_count} self-collision filter pairs")

stage.Save()
print(f"[import] Saved USD")

# Sanity: enumerate physics APIs
print("\n[import] Physics API summary:")
rigid_count = 0
art_count = 0
joint_count = 0
fixed_joint_count = 0
for prim in stage.Traverse():
    if UsdPhysics.RigidBodyAPI(prim):
        rigid_count += 1
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_count += 1
    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint_count += 1
    if prim.IsA(UsdPhysics.FixedJoint):
        fixed_joint_count += 1
print(f"  rigid bodies      : {rigid_count}")
print(f"  articulation roots: {art_count}")
print(f"  revolute joints   : {joint_count}")
print(f"  fixed joints      : {fixed_joint_count}  (expect 1: skull_to_antlers)")

# Total mass check
print("\n[import] Per-link mass:")
total = 0.0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(prim)
        m = mass_api.GetMassAttr().Get()
        if m and m > 0:
            print(f"  {prim.GetPath().name:20s} {m:8.3f} kg")
            total += m
print(f"  {'TOTAL':20s} {total:8.3f} kg  (target 82.0)")

# Print joint limits to verify widening propagated
print("\n[import] Revolute joint limits:")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint = UsdPhysics.RevoluteJoint(prim)
        lo_attr = joint.GetLowerLimitAttr()
        hi_attr = joint.GetUpperLimitAttr()
        lo = lo_attr.Get() if lo_attr else None
        hi = hi_attr.Get() if hi_attr else None
        # Limits are in DEGREES in USD! URDF was rad, USD is deg.
        print(f"  {prim.GetPath().name:30s} [{lo:+8.2f}, {hi:+8.2f}]")

app.close()
print("[import] done")
