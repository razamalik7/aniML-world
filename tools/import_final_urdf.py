"""Import deer_final.urdf -> deer_final.usd with physics, defaultPrim, and correct orientation."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import math
import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf
from pxr import Usd, UsdGeom, UsdPhysics, Gf

urdf = r"C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_final.urdf"
usd  = r"C:\Users\nars7\Downloads\aniML_world\deer_final.usd"

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

# Set defaultPrim
stage.SetDefaultPrim(root)

# Bake Rx(-90deg) into the root prim so deer spawns standing upright.
# USD has feet in +Y, skull in -Y (importer stored world-space as-is).
# Rx(-90): rotates +Y -> -Z (feet go down), -Y -> +Z (skull goes up).
# Quaternion Rx(-90): w=cos(-45deg), x=sin(-45deg), y=0, z=0
w = math.cos(math.radians(-45))
x = math.sin(math.radians(-45))
xf = UsdGeom.Xformable(root)
xf.ClearXformOpOrder()
orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
orient_op.Set(Gf.Quatd(float(w), float(x), 0.0, 0.0))

stage.Save()

LOG = open(r"C:\Users\nars7\Downloads\aniML_world\tools\import_final_log.txt", "w")
def log(s): LOG.write(s+"\n"); LOG.flush()

# Verify
rb = sum(1 for p in stage.Traverse() if UsdPhysics.RigidBodyAPI(p))
hr_foot = stage.GetPrimAtPath("/deer/HR_foot")
xf2 = UsdGeom.Xformable(hr_foot)
for o in xf2.GetOrderedXformOps():
    if "translate" in o.GetOpName():
        t = o.Get()
        log(f"HR_foot translate: Y={t[1]:.4f}")
log(f"Root orient baked: Rx(-90) w={w:.4f} x={x:.4f}")
log(f"Rigid bodies: {rb}")
log(f"SPAWN_Z recommendation: {t[1]+0.03:.4f}")
log(f"USD: {usd}")

# Also verify the baked orient
root2 = stage.GetPrimAtPath("/deer")
xf3 = UsdGeom.Xformable(root2)
for o in xf3.GetOrderedXformOps():
    log(f"Root op: {o.GetOpName()} = {o.Get()}")
LOG.close()
app.close()
