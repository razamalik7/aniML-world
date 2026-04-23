"""Compare physics properties between deer_world.usd and deer_fixed_robot.usd."""
from pxr import Usd, UsdPhysics, UsdGeom
import sys

def inspect_usd(path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {path}")
    print('='*60)
    stage = Usd.Stage.Open(path)
    
    total_mass = 0.0
    com_x_sum = 0.0
    com_z_sum = 0.0
    body_count = 0
    
    for prim in stage.Traverse():
        # Check for mass API
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            com_attr = mass_api.GetCenterOfMassAttr()
            density_attr = mass_api.GetDensityAttr()
            
            mass = mass_attr.Get() if mass_attr.IsValid() else None
            com = com_attr.Get() if com_attr.IsValid() else None
            density = density_attr.Get() if density_attr.IsValid() else None
            
            name = prim.GetName()
            path_str = str(prim.GetPath())
            
            print(f"\n  [{name}]")
            if mass is not None:
                print(f"    mass    = {mass:.4f} kg")
                total_mass += mass
                body_count += 1
            if density is not None and density > 0:
                print(f"    density = {density:.4f}")
            if com is not None:
                print(f"    CoM     = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")
                if mass:
                    com_x_sum += mass * com[0]
                    com_z_sum += mass * com[2]
    
    print(f"\n  TOTAL: {body_count} bodies, {total_mass:.2f} kg")
    if total_mass > 0:
        print(f"  Mass-weighted CoM X = {com_x_sum/total_mass:.4f} m")
        print(f"  Mass-weighted CoM Z = {com_z_sum/total_mass:.4f} m")

inspect_usd(r"C:\Users\nars7\Downloads\aniML_world\deer_world.usd", "deer_world.usd")
inspect_usd(r"C:\Users\nars7\Downloads\aniML_world\deer_fixed_robot.usd", "deer_fixed_robot.usd")
