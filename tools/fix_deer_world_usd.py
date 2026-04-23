"""
Reset deer_world.usd root orient back to identity.
Isaac Lab episode resets use init_state.rot, NOT the USD baked orient.
The baked orient was only affecting the initial preview frame, not physics.
We'll set init_state.rot = Rx(-90) in the articulation config instead.
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import math
from pxr import Usd, UsdGeom, Gf

LOG = open(r"C:\Users\nars7\Downloads\aniML_world\tools\fix_world_usd_log.txt", "w")
def log(s): LOG.write(s+"\n"); LOG.flush()

usd_path = r"C:\Users\nars7\Downloads\aniML_world\deer_world.usd"
stage = Usd.Stage.Open(usd_path)
root = stage.GetPrimAtPath("/deer")

# Set root to identity — Isaac Lab will apply init_state.rot at every reset
xf = UsdGeom.Xformable(root)
xf.ClearXformOpOrder()
orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))  # identity

stage.SetDefaultPrim(root)
stage.Save()

log("deer_world.usd root orient set to IDENTITY (1,0,0,0)")
log("Isaac Lab will apply Rx(-90) via init_state.rot at every episode reset")
log("Rx(-90) quaternion: w=0.7071068, x=-0.7071068, y=0, z=0")
log("")

# Verify foot position to determine correct spawn Z
hr_foot = stage.GetPrimAtPath("/deer/HR_foot")
xf2 = UsdGeom.Xformable(hr_foot)
for o in xf2.GetOrderedXformOps():
    if "translate" in o.GetOpName():
        t = o.Get()
        log(f"HR_foot USD translate: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
        log(f"After Rx(-90): foot goes to Z = -{t[1]:.4f} = {-t[1]:.4f} below pelvis")
        log(f"SPAWN_Z = {t[1] + 0.03:.4f}")

LOG.close()
app.close()
