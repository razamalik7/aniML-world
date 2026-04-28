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
importer.set_merge_fixed_joints(False)
importer.set_fix_base(False)
importer.set_import_inertia_tensor(True)
importer.set_distance_scale(1.0)
importer.set_density(0.0)
importer.set_convex_decomp(False)
importer.set_self_collision(True)
importer.set_default_drive_type(_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION)
importer.set_default_drive_strength(1047.19751)
importer.set_default_position_drive_damping(52.35988)

omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=importer,
    dest_path=usd_path,
)

from pxr import Usd, UsdPhysics
stage = Usd.Stage.Open(usd_path)

# Set defaultPrim to /deer so Isaac Lab references resolve correctly
root_prim = stage.GetPrimAtPath("/deer")
if root_prim:
    stage.SetDefaultPrim(root_prim)
    stage.Save()
    print(f"[import] Set defaultPrim to /deer and saved")

# Sanity: enumerate physics APIs
print("\n[import] Physics API summary:")
rigid_count = 0
art_count = 0
joint_count = 0
for prim in stage.Traverse():
    if UsdPhysics.RigidBodyAPI(prim):
        rigid_count += 1
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_count += 1
    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint_count += 1
print(f"  rigid bodies      : {rigid_count}")
print(f"  articulation roots: {art_count}")
print(f"  revolute joints   : {joint_count}")

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
