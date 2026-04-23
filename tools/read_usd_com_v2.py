"""
Spawn deer_standing_v2.usd in a live Isaac Lab sim, then read the actual
rigid body CoM positions in world space after settling.

Run with: isaaclab.bat -p tools/read_usd_com_v2.py
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
from pxr import Usd, UsdPhysics, Gf, Sdf

usd_path = r'C:\Users\nars7\Downloads\aniML_world\deer_standing_v2.usd'

# Open stage and traverse ALL prims, including through sublayers
stage = Usd.Stage.Open(usd_path)

print(f"Stage identifier: {stage.GetRootLayer().identifier}")
print(f"SubLayers: {stage.GetRootLayer().subLayerPaths}")
print()

# Compose the full stage (USD composes all sublayers)
all_prims = list(stage.TraverseAll())
print(f"Total prims in composed stage: {len(all_prims)}")
print()

# Find prims with physics:mass
mass_prims = []
for prim in all_prims:
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass_prims.append(prim)

print(f"Prims with PhysicsMassAPI: {len(mass_prims)}")
print()

if not mass_prims:
    print("No MassAPI prims found. Listing all prim paths:")
    for p in all_prims[:50]:
        print(f"  {p.GetPath()}  type={p.GetTypeName()}")
    print()
    # Try checking sublayer content
    for layer_path in stage.GetRootLayer().subLayerPaths:
        print(f"\nChecking sublayer: {layer_path}")
        sub_layer = Sdf.Layer.FindOrOpen(layer_path)
        if sub_layer:
            sub_stage = Usd.Stage.Open(sub_layer.identifier)
            for sp in sub_stage.TraverseAll():
                if sp.HasAPI(UsdPhysics.MassAPI):
                    mass_api = UsdPhysics.MassAPI(sp)
                    mass = mass_api.GetMassAttr().Get()
                    com  = mass_api.GetCenterOfMassAttr().Get()
                    print(f"  {sp.GetPath().name:<30} mass={mass}  CoM={com}")
else:
    print(f"{'Prim':<30} {'mass':>8}  {'local CoM (x,y,z)':>35}")
    print('-' * 80)
    total_mass = 0.
    for prim in mass_prims:
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get() or 0.
        com  = mass_api.GetCenterOfMassAttr().Get()
        com_str = f"({com[0]:+.4f},{com[1]:+.4f},{com[2]:+.4f})" if com else "None"
        print(f"  {prim.GetPath().name:<30} {mass:>8.3f}  {com_str}")
        total_mass += mass
    print(f"\nTotal mass: {total_mass:.2f} kg")

app.close()
