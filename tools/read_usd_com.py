"""
Read mass and centerOfMass from deer_standing_v2 USD after Isaac Sim import.
Run with: isaaclab.bat -p tools/read_usd_com.py
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
from pxr import Usd, UsdPhysics, UsdGeom, Gf

stage = Usd.Stage.Open(r'C:\Users\nars7\Downloads\aniML_world\deer_standing_v2.usd')

print("deer_standing_v2.usd — Physics mass and CoM per prim:")
print(f"{'Prim':<35} {'mass':>8}  {'CoM (x,y,z)':>40}  Isaac_height")
print('-' * 100)

total_mass = 0.0
com_weighted = np.zeros(3)

# Walk all prims, find ones with PhysicsMassAPI
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        com  = mass_api.GetCenterOfMassAttr().Get()
        if mass is None or mass == 0:
            continue
        com_xyz = np.array([com[0], com[1], com[2]]) if com else np.zeros(3)

        # Get world transform to find where this prim is in world space
        xform = UsdGeom.Xformable(prim)
        xf = xform.ComputeLocalToWorldTransform(0)
        # xf is a 4x4 GfMatrix4d
        # The translation gives the prim's world position
        t = xf.ExtractTranslation()
        world_pos = np.array([t[0], t[1], t[2]])

        # CoM in world space = world_pos + rotated local CoM offset
        # For now just show: world_pos.Z gives height of joint origin in Isaac
        prim_name = prim.GetPath().name
        print(f"  {prim_name:<35} {mass:>8.3f}  local_CoM=({com_xyz[0]:+.3f},{com_xyz[1]:+.3f},{com_xyz[2]:+.3f})  "
              f"world_h={world_pos[2]:+.4f}")

        total_mass += mass
        com_weighted += mass * (world_pos + com_xyz)

print()
print(f"Total mass: {total_mass:.2f} kg")
if total_mass > 0:
    com_world = com_weighted / total_mass
    print(f"Mass-weighted world CoM (at T=0): ({com_world[0]:+.4f}, {com_world[1]:+.4f}, {com_world[2]:+.4f})")
    print(f"CoM height in Isaac (Z-up): {com_world[2]:+.4f} m above the spawn point")
    print()
    print(f"NOTE: At spawn, pelvis is at Z = 0.55m.")
    print(f"      If world CoM Z ~ 0, then absolute CoM height = 0.55 + {com_world[2]:+.4f} = {0.55 + com_world[2]:.4f} m")
    print(f"      Real deer: ~0.65m. Difference: {0.55 + com_world[2] - 0.65:+.3f} m")

app.close()
