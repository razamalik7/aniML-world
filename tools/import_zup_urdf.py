"""Import deer_zup.urdf -> deer_zup.usd. No root rotation needed."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf
from pxr import Usd, UsdPhysics

urdf = r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_zup.urdf"
usd  = r"C:\Users\nars7\Downloads\aniML_world\deer_zup.usd"

_, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.set_merge_fixed_joints(False)
cfg.set_fix_base(False)
cfg.set_import_inertia_tensor(True)
cfg.set_distance_scale(1.0)
cfg.set_density(0.0)
cfg.set_convex_decomp(False)
cfg.set_self_collision(True)
cfg.set_default_drive_type(_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION)
cfg.set_default_drive_strength(1047.0)
cfg.set_default_position_drive_damping(52.0)

omni.kit.commands.execute("URDFParseAndImportFile",
    urdf_path=urdf, import_config=cfg, dest_path=usd)

stage = Usd.Stage.Open(usd)
root  = stage.GetPrimAtPath("/deer")
stage.SetDefaultPrim(root)
stage.Save()

rb = sum(1 for p in stage.Traverse() if UsdPhysics.RigidBodyAPI(p))
print(f"Rigid bodies: {rb}  USD: {usd}")
app.close()
