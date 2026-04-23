"""
Patch deer_world.urdf with the mesh-computed CoM offsets and masses from deer_standing.urdf.

deer_world.urdf has:
  - round-number masses (18, 28, 6, 5 kg etc.)
  - all CoM offsets = 0 0 0

deer_standing.urdf has:
  - mesh-computed masses (total 70 kg)
  - proper CoM offsets per link (computed from mesh geometry)

We keep deer_world.urdf's joint structure (it was used for the successful April 10 stand run)
but replace the mass/CoM values with deer_standing.urdf's values.
"""
import xml.etree.ElementTree as ET
import copy

ET.register_namespace('', '')

# Load both URDFs
world_tree = ET.parse(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf')
world_root = world_tree.getroot()

standing_tree = ET.parse(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf')
standing_root = standing_tree.getroot()

# Build lookup: link_name -> inertial element from deer_standing
standing_inertials = {}
for link in standing_root.findall('link'):
    name = link.get('name')
    inertial = link.find('inertial')
    if inertial is not None:
        standing_inertials[name] = inertial

# Patch deer_world.urdf
patched = 0
skipped = []
for link in world_root.findall('link'):
    name = link.get('name')
    if name in standing_inertials:
        # Remove existing inertial
        old_inertial = link.find('inertial')
        if old_inertial is not None:
            link.remove(old_inertial)
        # Add the standing inertial (deep copy)
        new_inertial = copy.deepcopy(standing_inertials[name])
        link.append(new_inertial)
        
        mass_val = float(new_inertial.find('mass').get('value'))
        origin = new_inertial.find('origin')
        xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
        print(f'  Patched {name:<20} mass={mass_val:.2f} kg  CoM={xyz}')
        patched += 1
    else:
        skipped.append(name)

print(f'\nPatched {patched} links')
if skipped:
    print(f'Skipped (not in deer_standing): {skipped}')

# Save as new file
out_path = r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world_comfix.urdf'
world_tree.write(out_path, xml_declaration=True, encoding='utf-8')
print(f'\nSaved to: {out_path}')

# Verify total mass
total = sum(float(link.find('inertial').find('mass').get('value')) 
            for link in world_root.findall('link') 
            if link.find('inertial') is not None)
print(f'New total mass: {total:.2f} kg')
