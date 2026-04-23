"""Import deer_world.urdf with correct physics settings for Isaac Lab."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf

LOG = open(r"C:\Users\nars7\Downloads\aniML_world\tools\import_v2_log.txt", "w")

def log(msg):
    LOG.write(msg + "\n")
    LOG.flush()

urdf_path = r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf"
usd_path  = r"C:\Users\nars7\Downloads\aniML_world\deer_world.usd"

status, importer = omni.kit.commands.execute("URDFCreateImportConfig")

log("Config attrs:")
for attr in dir(importer):
    if not attr.startswith('_'):
        try:
            log(f"  {attr} = {getattr(importer, attr)}")
        except:
            pass

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
# Z-up: our URDF is in Blender world space, Y is "back" not "up"
# The importer's up_vector tells it which axis in the URDF file is "up"
# Blender world: Z is actually sideways for the lying deer, but the joint
# offsets are consistent. Leave default (Z-up) so importer doesn't rotate.

omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=importer,
    dest_path=usd_path,
)

from pxr import Usd, UsdPhysics, Sdf
stage = Usd.Stage.Open(usd_path)

# Set defaultPrim so Isaac Lab can reference the USD correctly
root_prim = stage.GetPrimAtPath("/deer")
if root_prim:
    stage.SetDefaultPrim(root_prim)
    stage.Save()
    log("Set defaultPrim to /deer and saved.")
log("\n=== Physics APIs ===")
rigid_count = 0
for prim in stage.Traverse():
    has_rb  = bool(UsdPhysics.RigidBodyAPI(prim))
    has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    if has_rb or has_col or has_art:
        tags = []
        if has_rb:  tags.append("RigidBody"); rigid_count += 1
        if has_col: tags.append("Collision")
        if has_art: tags.append("ArticulationRoot")
        log(f"  {prim.GetPath()}: {', '.join(tags)}")

log(f"\nTotal rigid bodies: {rigid_count}")
LOG.close()
app.close()
