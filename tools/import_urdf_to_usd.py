"""Import URDF → USD via Isaac Sim's native URDF importer.

Run with: _isaac_sim\\python.bat import_urdf_to_usd.py <urdf> <output.usd> [robot_name]
"""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

for _ in range(20):
    app.update()

from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402

urdf_path = sys.argv[1]
output_path = sys.argv[2]
robot_name = sys.argv[3] if len(sys.argv) > 3 else "deer"

print(f"URDF:   {urdf_path}", flush=True)
print(f"Output: {output_path}", flush=True)

import_config = _urdf.ImportConfig()
import_config.fix_base = False
import_config.make_default_prim = True
import_config.create_physics_scene = True
try:
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_NONE
except (AttributeError, TypeError):
    pass
import_config.self_collision = False
import_config.distance_scale = 1.0
import_config.density = 0.0

import os
urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
urdf_file = os.path.basename(urdf_path)

urdf_interface = _urdf.acquire_urdf_interface()
parse_result = urdf_interface.parse_urdf(urdf_dir, urdf_file, import_config)
dest_path = f"/{robot_name}"
prim_path = urdf_interface.import_robot(
    urdf_dir, urdf_file, parse_result, import_config, dest_path
)
print(f"Imported at: {prim_path}", flush=True)

for _ in range(10):
    app.update()

stage = omni.usd.get_context().get_stage()

robot_prim = stage.GetPrimAtPath(dest_path)
if not robot_prim.IsValid():
    for p in stage.Traverse():
        if UsdPhysics.ArticulationRootAPI(p):
            robot_prim = p
            dest_path = str(p.GetPath())
            break

if robot_prim.IsValid():
    stage.SetDefaultPrim(robot_prim)
    stage.GetRootLayer().Export(output_path)

    out_stage = Usd.Stage.Open(output_path)
    print(f"\nWrote: {output_path}", flush=True)
    print(f"Default prim: {out_stage.GetDefaultPrim().GetPath()}", flush=True)

    rc = sum(1 for p in out_stage.Traverse() if UsdPhysics.RigidBodyAPI(p))
    ac = sum(1 for p in out_stage.Traverse() if UsdPhysics.ArticulationRootAPI(p))
    print(f"Rigid bodies: {rc}", flush=True)
    print(f"Articulation roots: {ac}", flush=True)
else:
    print("ERROR: robot prim not found!", flush=True)
    sys.exit(1)

app.close()
