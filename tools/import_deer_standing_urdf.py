"""
Import deer_standing.urdf into Isaac Sim to produce deer_standing_robot_v2.usd.

deer_standing.urdf has:
  - Mesh-computed CoM offsets for all 16 links
  - Total mass 70 kg
  - CoM at 0.652m above ground — matches real white-tailed deer

Run with:
  isaaclab.bat -p tools/import_deer_standing_urdf.py
"""
import argparse
import sys
import os
from pathlib import Path

# Isaac Sim app must be initialized before importing omni modules
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import omni.kit.commands
from omni.isaac.urdf import _urdf as omni_urdf

REPO = Path(__file__).resolve().parents[1]
URDF_PATH = str(REPO / "assets" / "deer_urdf" / "deer_standing.urdf")
OUT_USD  = str(REPO / "deer_standing_v2.usd")

print(f"[import_urdf] Input:  {URDF_PATH}")
print(f"[import_urdf] Output: {OUT_USD}")

# Configure import
cfg = omni_urdf.ImportConfig()
cfg.merge_fixed_joints         = False
cfg.convex_decomp              = False
cfg.import_inertia_tensor      = True   # use URDF inertia values
cfg.fix_base                   = False
cfg.make_default_prim          = True
cfg.self_collision             = True
cfg.create_physics_scene       = True
cfg.default_drive_type         = omni_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
cfg.default_drive_strength     = 200.0
cfg.default_position_drive_damping = 6.0
cfg.distance_scale             = 1.0
cfg.density                    = 0.0   # use URDF mass values, not density

result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=cfg,
    dest_path=OUT_USD,
)

if result:
    print(f"[import_urdf] SUCCESS: {OUT_USD}")
    print(f"[import_urdf] Root prim: {prim_path}")
else:
    print("[import_urdf] FAILED!")
    sys.exit(1)

app.close()
