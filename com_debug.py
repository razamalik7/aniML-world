"""
Debug: print the raw joint chain and CoM offset numbers for deer_standing.urdf
to understand why the computed CoM is near the feet.
"""
import xml.etree.ElementTree as ET
import numpy as np

urdf = r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf'
tree = ET.parse(urdf)
root = tree.getroot()

print("=== RAW JOINT CHAIN (URDF Y-up, pelvis at origin) ===")
print(f"{'Joint':<35} {'type':<10} {'parent':<15} {'child':<15} {'xyz offset'}")
print('-'*100)
for j in root.findall('joint'):
    parent = j.find('parent').get('link')
    child  = j.find('child').get('link')
    origin = j.find('origin')
    xyz = origin.get('xyz','0 0 0') if origin is not None else '0 0 0'
    print(f"  {j.get('name'):<35} {j.get('type'):<10} {parent:<15} {child:<15} {xyz}")

print()
print("=== RAW INERTIAL / CoM DATA per link ===")
print(f"{'Link':<20} {'mass':>8}  {'CoM origin xyz (LOCAL frame)'}")
print('-'*70)
for link in root.findall('link'):
    name = link.get('name')
    inertial = link.find('inertial')
    if inertial is not None:
        mass = float(inertial.find('mass').get('value'))
        origin = inertial.find('origin')
        xyz = origin.get('xyz','0 0 0') if origin is not None else '0 0 0'
        rpy = origin.get('rpy','0 0 0') if origin is not None else '0 0 0'
        print(f"  {name:<20} {mass:>8.3f}  xyz=({xyz})  rpy=({rpy})")

print()
print("=== INTERPRETATION ===")
print("""
In URDF, <inertial><origin xyz='x y z'/> means:
  The CoM of this link is at position (x, y, z) relative to the LINK FRAME,
  where the link frame origin is defined by the JOINT that connects to this link.

So 'chest' CoM at (-0.000028, -0.622695, 1.286595) means:
  Starting from the chest joint origin,
  go -0.62m in local Y and +1.29m in local Z to reach the chest CoM.

BUT: if the local frame is not aligned with the world frame (due to joint rpy or
the mesh export orientation), these offsets are NOT simply addable to world pos.

Let's check what coordinate system Blender exported in.
""")

# The skull CoM: let's see if it makes physical sense
# skull joint origin (world, accumulated from pelvis):
#   pelvis->chest offset + chest->neck offset + neck->skull offset
joints_xyz = {}
for j in root.findall('joint'):
    child = j.find('child').get('link')
    origin = j.find('origin')
    xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
    joints_xyz[child] = xyz

# Manual accumulation
pelvis_world = np.array([0,0,0])
chest_world  = pelvis_world + np.array(joints_xyz.get('chest', [0,0,0]))
neck_world   = chest_world  + np.array(joints_xyz.get('neck', [0,0,0]))
skull_world  = neck_world   + np.array(joints_xyz.get('skull', [0,0,0]))
print(f"Joint world positions (simple xyz accumulation, Y-up, pelvis at origin):")
print(f"  pelvis: {pelvis_world}")
print(f"  chest:  {chest_world}   (joint, not CoM)")
print(f"  neck:   {neck_world}")
print(f"  skull:  {skull_world}")

# What does the skull local CoM of (+0.000, +0.150, -0.061) look like?
# If local frame = world frame at rest (no rpy), skull CoM is at:
skull_com_local = np.zeros(3)
for link in root.findall('link'):
    if link.get('name') == 'skull':
        origin = link.find('inertial').find('origin')
        skull_com_local = np.array(list(map(float, origin.get('xyz','0 0 0').split())))
skull_com_world = skull_world + skull_com_local
print(f"\n  skull CoM local offset: {skull_com_local}")
print(f"  skull CoM world (naive): {skull_com_world}")
print(f"  Skull CoM Y={skull_com_world[1]:.3f}")
print(f"  Isaac height of skull CoM (Z=-URDF_Y) = {-skull_com_world[1]:.3f}m relative to pelvis joint")

print()
print("SANITY CHECK:")
print(f"  Skull joint world Y = {skull_world[1]:.3f}")
print(f"  In Isaac (Z=-Y): skull joint is {-skull_world[1]:.3f}m BELOW pelvis")
print(f"  => skull is BELOW the pelvis in Isaac space")
print(f"  That means the spine goes DOWNWARD in this URDF, which is WRONG for a standing deer.")
print()
print(f"  Chest joint world Y = {chest_world[1]:.3f}")
print(f"  => chest is {-chest_world[1]:.3f}m below pelvis")
print(f"  => The CHEST is below the PELVIS in this URDF!")
print()
print(f"This means deer_standing.urdf has the skeleton organized differently:")
print(f"  - Pelvis is at the TOP (highest Y)")
print(f"  - Chest, neck, skull chain goes DOWNWARD (lower Y)")
print(f"  - Legs also go downward (lower Y)")
print()
print(f"CONCLUSION: In this URDF, Y-up means:")
print(f"  Pelvis is at height 0 (joint origin)")
print(f"  Everything below (chest, neck, skull, legs) has negative or positive Y")
print(f"  depending on mesh export orientation in Blender.")
print()
print("The CoM local offsets make sense WITHIN each link's own coordinate frame,")
print("but the link frames may be rotated relative to the world frame.")
print()
print("==> The CoM calculation requires knowing each link's rotation matrix at rest,")
print("    which depends on the joint RPY values AND the joint angles (all zero at rest).")

# Check if there are any non-trivial rpy values
print()
print("Joint RPY values:")
for j in root.findall('joint'):
    if j.get('type') == 'fixed':
        continue
    origin = j.find('origin')
    if origin is not None:
        rpy = origin.get('rpy', '0 0 0')
        if rpy != '0 0 0' and rpy != '0.0 0.0 0.0':
            print(f"  {j.get('name')}: rpy={rpy}  *** NON-ZERO ***")
        else:
            print(f"  {j.get('name')}: rpy={rpy}")
    else:
        print(f"  {j.get('name')}: no origin")
