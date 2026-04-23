"""Analyze CoM for a URDF, accounting for per-link CoM offsets and joint tree."""
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

def analyze_urdf(urdf_path, label):
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

    # Build joint tree: child -> (parent, xyz_offset)
    joints = {}
    for joint in root.findall('joint'):
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        origin = joint.find('origin')
        xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
        joints[child] = (parent, np.array(xyz))

    # Compute world position of each link's joint origin (URDF Y-up, pelvis at origin)
    cache = {}
    def world_pos(link_name):
        if link_name in cache:
            return cache[link_name]
        if link_name not in joints:  # root
            cache[link_name] = np.zeros(3)
            return np.zeros(3)
        parent, offset = joints[link_name]
        pos = world_pos(parent) + offset
        cache[link_name] = pos
        return pos

    print(f"\n{'='*65}")
    print(f"  {label}")
    print('='*65)
    print(f"  {'Link':<20} {'mass':>6}  {'URDF CoM (x,y,z)':>30}  h_isaac")
    print(f"  {'-'*20} {'-'*6}  {'-'*30}  {'-'*7}")

    total_mass = 0.0
    com_world = np.zeros(3)

    for name, mass in masses.items():
        joint_origin = world_pos(name)
        # CoM in world = joint origin + local CoM offset (both in URDF Y-up space)
        com_world_link = joint_origin + com_offsets_local.get(name, np.zeros(3))
        total_mass += mass
        com_world += mass * com_world_link
        # URDF Y-up -> Isaac Z-up via Rx(-90): height = -URDF_y
        h_isaac = -com_world_link[1]
        print(f"  {name:<20} {mass:>6.2f}  ({com_world_link[0]:+.3f}, {com_world_link[1]:+.3f}, {com_world_link[2]:+.3f})  {h_isaac:+.3f}")

    com = com_world / total_mass
    # URDF -> Isaac: X stays X (fore/aft), Y->Z (lateral), -Y->height
    # In URDF: X = lateral, Y = vertical (up), Z = fore/aft
    # After Rx(-90): Isaac X = URDF X (lateral), Isaac Y = URDF Z (fore/aft), Isaac Z = -URDF Y (height)
    # Wait - need to check coordinate conventions carefully
    # URDF: X=right, Y=up, Z=forward (ROS standard)
    # Isaac Z-up after Rx(-90): the Rx(-90) rotates the model so:
    #   URDF +Y (up) -> Isaac +Z (up)  [Rx(-90) rotates Y toward -Z, so -Y->+Z... hmm]
    # Actually Rx(-90) = rotation of -90 degrees around X axis:
    #   [1,  0,    0  ]   URDF(x,y,z) -> Isaac(x, z, -y)
    #   [0,  0,    1  ]
    #   [0, -1,    0  ]
    # So Isaac X = URDF X (lateral/left-right)
    #    Isaac Y = URDF Z (fore/aft)
    #    Isaac Z = -URDF Y (height, URDF -Y becomes Isaac +Z)

    isaac_lateral = com[0]   # left-right (should be ~0)
    isaac_fwdback = com[2]   # fore/aft (URDF Z, after Rx(-90) this becomes Isaac Y)
    isaac_height  = -com[1]  # height above pelvis (URDF -Y = Isaac +Z)

    print(f"\n  Total mass: {total_mass:.1f} kg")
    print(f"  CoM URDF space (Y-up): ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f})")
    print(f"\n  In Isaac Z-up (pelvis joint at origin):")
    print(f"    Lateral (left/right)  = {isaac_lateral:+.4f} m  (should be ~0)")
    print(f"    Fore/aft              = {isaac_fwdback:+.4f} m  (- = toward hind, + = toward front)")
    print(f"    Height above pelvis   = {isaac_height:+.4f} m  (+ = above pelvis joint)")
    print(f"    Absolute height (pelvis@0.93m) = {0.93 + isaac_height:.3f} m above ground")

    print(f"\n  DIAGNOSIS:")
    if abs(isaac_lateral) > 0.02:
        print(f"  !! LATERAL IMBALANCE: {isaac_lateral:+.4f} m — will tip sideways")
    else:
        print(f"  OK  lateral balance: {isaac_lateral:+.4f} m")
    if abs(isaac_fwdback) > 0.05:
        print(f"  !! FORE/AFT IMBALANCE: {isaac_fwdback:+.4f} m — tips {'forward' if isaac_fwdback>0 else 'backward'}")
    else:
        print(f"  OK  fore/aft balance: {isaac_fwdback:+.4f} m")
    if isaac_height > 0.15:
        print(f"  !! TOP-HEAVY: CoM is {isaac_height:.3f}m above pelvis — easy to topple laterally")
    elif isaac_height < -0.3:
        print(f"  !! TOO LOW: CoM is {abs(isaac_height):.3f}m below pelvis")
    else:
        print(f"  OK  CoM height vs pelvis: {isaac_height:+.4f} m")

    # Pendulum stability estimate: CoM height above feet
    feet_height = 0.93  # pelvis Z in Isaac
    com_abs_height = feet_height + isaac_height
    print(f"\n  Pendulum: CoM at {com_abs_height:.3f}m, feet at 0.00m -> stability ratio = {com_abs_height:.2f}m")
    print(f"  (A real white-tailed deer CoM is ~0.65m above ground for a 70kg animal)")

for path, label in [
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf', 'deer_world.urdf (old — zero CoM offsets)'),
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world_comfix.urdf', 'deer_world_comfix.urdf (PATCHED — mesh CoM from deer_standing)'),
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf', 'deer_standing.urdf (source of CoM values)'),
]:
    analyze_urdf(path, label)
