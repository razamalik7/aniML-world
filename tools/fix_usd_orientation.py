"""
Fix deer_final.usd: bake correct root orientation so deer spawns standing.

Current state: HR_foot Y=+0.597 (above pelvis), skull Y=-0.361 (below pelvis).
The deer is upside down in USD space. Need Rx(180) to flip: Y -> -Y, Z -> -Z.

BUT: Isaac Lab applies the spawn rotation ON TOP of the USD root orient.
Isaac Lab's UsdFileCfg spawning references the USD and places it, then applies
init_state rotation. So the USD root orient is NOT used by Isaac Lab directly -
it references /deer as defaultPrim and re-orients via init_state.

The actual issue: Isaac Lab references the USD, which puts pelvis at origin.
The joint tree has feet at Y=+0.597 (up in world), skull at Y=-0.362 (down).
Isaac's world is Z-up. So spawn_rot needs to rotate this to be Z-up standing.

Current world layout (USD, identity orientation):
  - Pelvis: origin
  - Feet: +Y direction (0.597 above)  
  - Skull: -Y direction (0.362 below) and -Z
  - Body: runs in +Y/-Y

To make this stand upright (feet under body, skull up):
  - We need to rotate so that +Y (feet direction) becomes -Z (down in Isaac)
  - That's Rx(-90): world-X stays, world-Y -> -Isaac-Z, world-Z -> Isaac-Y
  - Rx(-90) quaternion: w=cos(-45deg)=0.7071, x=sin(-45deg)=-0.7071, y=0, z=0
  - BUT: feet need to go DOWN, skull UP.
  - Skull is at (0, -0.361, -0.944) -> after Rx(-90): (0, +0.944, +0.361) -- skull goes UP and forward. 
  - Feet at (x, +0.597, z) -> after Rx(-90): (x, -z, -0.597) -- feet go DOWN. CORRECT!

So spawn_rot = (w=0.7071, x=-0.7071, y=0, z=0) = Rx(-90)
And spawn_z = 0.597 + small margin = 0.65 (feet touch ground when pelvis at 0.65)

Let's verify by baking this into the USD root so it's consistent for all instances.
Isaac Lab uses the USD's defaultPrim xform as the BASE, then applies init_state on top.
If we bake Rx(-90) into the USD root prim, spawn_rot can be identity.
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf
import math

usd_path = r"C:\Users\nars7\Downloads\aniML_world\deer_final.usd"
stage = Usd.Stage.Open(usd_path)

LOG = open(r"C:\Users\nars7\Downloads\aniML_world\tools\fix_orient_log.txt", "w")
def log(s): LOG.write(s+"\n"); LOG.flush()

# Verify the analysis
hr_foot = stage.GetPrimAtPath("/deer/HR_foot")
skull   = stage.GetPrimAtPath("/deer/skull")
for p, name in [(hr_foot, "HR_foot"), (skull, "skull")]:
    xf = UsdGeom.Xformable(p)
    for o in xf.GetOrderedXformOps():
        if "translate" in o.GetOpName():
            t = o.Get()
            log(f"{name}: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")

# Bake Rx(-90) into root: w=0.7071068, x=-0.7071068, y=0, z=0
root = stage.GetPrimAtPath("/deer")
xf_root = UsdGeom.Xformable(root)

# Clear existing ops and set Rx(-90)
xf_root.ClearXformOpOrder()
w = math.cos(math.radians(-90)/2)
x = math.sin(math.radians(-90)/2)
orient_op = xf_root.AddOrientOp()
orient_op.Set(Gf.Quatf(w, x, 0, 0))

# Verify
log(f"\nBaked orient: w={w:.6f}, x={x:.6f}, y=0, z=0  (Rx -90deg)")
log("After this rotation:")
log("  USD +Y (feet direction) -> Isaac -Z (downward) CORRECT")
log("  USD -Y (skull direction) -> Isaac +Z (upward) CORRECT")
log("  Feet were at Y=+0.597 -> Isaac Z=-0.597 from pelvis")
log("  So SPAWN_Z = 0.62 puts feet right at ground level")

stage.Save()
log(f"\nSaved to {usd_path}")
LOG.close()
app.close()
