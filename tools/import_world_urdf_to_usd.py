"""Import deer_world.urdf into deer_world.usd via Isaac Sim's URDF importer."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf

urdf_path = r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf"
usd_path  = r"C:\Users\nars7\Downloads\aniML_world\deer_world.usd"

status, importer = omni.kit.commands.execute(
    "URDFCreateImportConfig"
)

importer.merge_fixed_joints        = False
importer.fix_base                  = False
importer.import_inertia_tensor     = True
importer.distance_scale            = 1.0
importer.density                   = 0.0
importer.default_drive_type        = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
importer.default_drive_strength    = 1047.19751
importer.default_position_drive_damping = 52.35988
importer.default_drive_strength    = 1047.19751
importer.convex_decomp             = False
importer.self_collision            = True
omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=importer,
    dest_path=usd_path,
)

print(f"Imported to: {usd_path}")
app.close()
