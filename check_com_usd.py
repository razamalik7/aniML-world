"""
Read mass and CoM from the generated deer_standing_v2 USD files directly.
These are plain-text USDA equivalents embedded in the binary USDC.
We parse configuration/deer_standing_v2_physics.usd (which is USDA / text).
"""
import re
from pathlib import Path

physics_usd = Path(r'C:\Users\nars7\Downloads\aniML_world\configuration\deer_standing_v2_physics.usd')
base_usd    = Path(r'C:\Users\nars7\Downloads\aniML_world\configuration\deer_standing_v2_base.usd')

def try_read_text(path):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return f.read()
    except:
        return None

# Try reading both as text (they may be USDA)
for path, label in [(physics_usd, 'physics'), (base_usd, 'base')]:
    content = try_read_text(path)
    if content and '#usda' in content[:50]:
        print(f'{label}.usd is USDA (text format) — size={len(content)}')
        # Find mass and centerOfMass entries
        masses    = re.findall(r'physics:mass\s*=\s*([\d.eE+-]+)', content)
        coms      = re.findall(r'physics:centerOfMass\s*=\s*\((.*?)\)', content, re.DOTALL)
        prim_names = re.findall(r'def\s+\w+\s+"(\w+)"', content)
        print(f'  Found {len(masses)} mass values, {len(coms)} CoM values')
        for i, (m, c) in enumerate(zip(masses, coms)):
            name = prim_names[i] if i < len(prim_names) else f'link_{i}'
            print(f'    {name:<20} mass={m:<8} CoM=({c.strip()})')
    else:
        print(f'{label}.usd is binary USDC — size={path.stat().st_size} bytes (cannot read as text)')

# Also show the URDF comparison summary directly
print()
print('='*60)
print('GROUND TRUTH: deer_standing.urdf mass-weighted CoM analysis')
print('='*60)

import xml.etree.ElementTree as ET
import numpy as np

def analyze_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    masses, com_offsets_local = {}, {}
    for link in root.findall('link'):
        name = link.get('name')
        inertial = link.find('inertial')
        if inertial is not None:
            m = float(inertial.find('mass').get('value'))
            masses[name] = m
            origin = inertial.find('origin')
            xyz = list(map(float, origin.get('xyz', '0 0 0').split())) if origin is not None else [0,0,0]
            com_offsets_local[name] = np.array(xyz)

    joints = {}
    for j in root.findall('joint'):
        parent = j.find('parent').get('link')
        child  = j.find('child').get('link')
        origin = j.find('origin')
        xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
        joints[child] = (parent, np.array(xyz))

    cache = {}
    def world_pos(link_name):
        if link_name in cache:
            return cache[link_name]
        if link_name not in joints:
            cache[link_name] = np.zeros(3)
            return np.zeros(3)
        parent, offset = joints[link_name]
        pos = world_pos(parent) + offset
        cache[link_name] = pos
        return pos

    total_mass = 0.0
    com_world  = np.zeros(3)
    print(f"\n  {'Link':<20} {'mass':>6}  {'URDF CoM (x,y,z)':>32}  Isaac_h")
    print(f"  {'-'*20} {'-'*6}  {'-'*32}  {'-'*7}")
    for name, mass in masses.items():
        joint_origin = world_pos(name)
        com_world_link = joint_origin + com_offsets_local.get(name, np.zeros(3))
        total_mass += mass
        com_world  += mass * com_world_link
        h = -com_world_link[1]   # URDF Y-up -> Isaac Z height
        print(f"  {name:<20} {mass:>6.2f}  ({com_world_link[0]:+.3f},{com_world_link[1]:+.3f},{com_world_link[2]:+.3f})  {h:+.3f}")

    com = com_world / total_mass
    isaac_height = -com[1]
    isaac_fwdback = com[2]

    print(f"\n  Total: {total_mass:.1f} kg")
    print(f"  Mass-weighted CoM (URDF Y-up): ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f})")
    print(f"\n  In Isaac Z-up (pelvis joint at world origin):")
    print(f"    CoM lateral   = {com[0]:+.4f} m   (0 = perfectly centered)")
    print(f"    CoM fore/aft  = {isaac_fwdback:+.4f} m   (- = toward hindquarters)")
    print(f"    CoM above pelvis joint = {isaac_height:+.4f} m")

    # Pelvis joint spawns at Z = 0.55 in Isaac
    pelvis_z = 0.55
    com_abs  = pelvis_z + isaac_height
    print(f"\n    Pelvis spawn Z = {pelvis_z} m")
    print(f"    => CoM absolute height = {com_abs:.3f} m above ground")
    print(f"    => Real white-tailed deer CoM  ~0.65 m above ground")
    print(f"    => Difference: {com_abs - 0.65:+.3f} m  ({'higher' if com_abs > 0.65 else 'lower'} than real deer)")
    print()

    if com_abs > 0.75:
        print("  !! WARNING: CoM too high — deer will be unstable / tip easily")
    elif com_abs > 0.70:
        print("  !! MARGINAL: CoM slightly high — may need wider stance or lower body")
    else:
        print("  OK: CoM height is in realistic range for a deer")

    if abs(com[0]) > 0.02:
        print(f"  !! LATERAL IMBALANCE: {com[0]:+.4f} m")
    else:
        print(f"  OK: lateral balance {com[0]:+.4f} m")

    if isaac_fwdback < -0.10:
        print(f"  !! TIPS BACKWARD: fore/aft CoM = {isaac_fwdback:+.4f} m")
    elif isaac_fwdback > 0.05:
        print(f"  !! TIPS FORWARD: fore/aft CoM = {isaac_fwdback:+.4f} m")
    else:
        print(f"  OK: fore/aft balance {isaac_fwdback:+.4f} m")

analyze_urdf(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf')

# Also compare old vs new
print()
print('='*60)
print('COMPARISON: old deer_world.urdf vs new deer_standing.urdf')
print('='*60)

for urdf, label, pelvis_z in [
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf',    'deer_world.urdf (OLD, broken CoM)',   0.93),
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf', 'deer_standing.urdf (NEW, correct CoM)', 0.55),
]:
    tree = ET.parse(urdf)
    root = tree.getroot()
    masses2, coms2 = {}, {}
    for link in root.findall('link'):
        name = link.get('name')
        inertial = link.find('inertial')
        if inertial is not None:
            m = float(inertial.find('mass').get('value'))
            masses2[name] = m
            origin = inertial.find('origin')
            xyz = list(map(float, origin.get('xyz', '0 0 0').split())) if origin is not None else [0,0,0]
            coms2[name] = np.array(xyz)
    joints2 = {}
    for j in root.findall('joint'):
        parent = j.find('parent').get('link')
        child  = j.find('child').get('link')
        origin = j.find('origin')
        xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
        joints2[child] = (parent, np.array(xyz))
    cache2 = {}
    def wp2(ln):
        if ln in cache2: return cache2[ln]
        if ln not in joints2:
            cache2[ln] = np.zeros(3)
            return np.zeros(3)
        parent, offset = joints2[ln]
        pos = wp2(parent) + offset
        cache2[ln] = pos
        return pos
    total2 = 0.; cv2 = np.zeros(3)
    for name, mass in masses2.items():
        c = wp2(name) + coms2.get(name, np.zeros(3))
        total2 += mass; cv2 += mass * c
    com2 = cv2 / total2
    abs_h = pelvis_z + (-com2[1])
    print(f"  {label}")
    print(f"    spawn Z={pelvis_z}, CoM absolute height={abs_h:.3f}m, fore/aft={com2[2]:+.3f}m")
    print()
