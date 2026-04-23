"""
Definitive CoM analysis for deer_standing.urdf.

Key insight: In URDF, the <origin xyz="..."> tag inside <inertial> gives the CoM
position in the LINK's local coordinate frame — NOT relative to the joint origin
added to world space.

For the deer, all joint rotations at rest pose = 0, so the link frames ARE
aligned with the parent frame at rest. This means we CAN accumulate:
  world_com_link = world_joint_origin + com_offset_in_link_frame

BUT: the link's local frame axes may differ from the parent's axes if there is a
joint <origin rpy="..."> rotation. Let's check those.

Also: we need to figure out what Z the pelvis actually spawns at.
The key is: what is the Y coordinate of the BOTTOM of each foot in URDF space,
where "bottom" means the lowest surface of the foot mesh.

Since deer_standing.urdf was exported from Blender with proper CoM,
the CoM offset *within* each link = offset from joint origin to geometric centroid.
The foot mesh bottom = joint_origin + 2 * CoM_offset (centroid ~ midpoint of a roughly
symmetric shape like a hoof).

Alternatively, we can use the inertia tensor to estimate link extents, but
that requires knowing the shape. Let's use a simpler bound:

For URDF Y-up coordinates, the GROUND is the plane with the HIGHEST Y value
(since Y is up, lower Y = higher above ground = actually more negative in Isaac).

Wait — let me re-examine the coordinate conventions carefully.

In URDF (Y-up, ROS convention):
  - +Y is UP
  - +Z is FORWARD  
  - +X is LEFT

After applying Rx(-90) to go from URDF to Isaac Z-up:
  - URDF +Y (up) → Isaac +Z (up)
  - URDF +Z (fwd) → Isaac -Y
  - URDF +X → Isaac +X

So: Isaac_Z = URDF_Y (height), Isaac_Y = -URDF_Z, Isaac_X = URDF_X

In the URDF file, the pelvis is the root link at (0,0,0).
The foot joints should be BELOW the pelvis, meaning LOWER URDF_Y (smaller or negative Y).
The GROUND is at the lowest point of the feet = minimum URDF_Y of the foot mesh bottoms.

From our skeleton output:
  FR_foot joint origin:  URDF_Y = +0.444  (above pelvis? That can't be feet on the ground)
  
Wait — this means in this URDF's Y-up coordinate system, the feet are ABOVE
the pelvis (higher Y). But in a Y-up system, higher Y = higher up. So the feet
are higher than the pelvis? That would mean the model is upside down in URDF space!

Let me check: if the pelvis is the body center (hip area), and the feet joint origins
are at Y=0.44 while the skull is at Y=1.51, then:
  - skull (head) is at the HIGHEST Y point = highest up = head is on top ✓ 
  - pelvis (hip) is at Y=0 = middle ✓
  - feet are at Y=0.44 = ABOVE the pelvis!?

That would mean the legs go UP from the pelvis to the feet — i.e., the model 
was exported standing on its BACK (hooves in the air). This matches the known
issue that we need Rx(-90) — but wait, Rx(-90) rotates Y toward -Z:
  Isaac_Z = -URDF_Y
  Isaac_Y = URDF_Z  
  Isaac_X = URDF_X

So if foot joint URDF_Y = +0.44, then Isaac foot joint Z = -0.44 (below origin).
Pelvis at URDF_Y = 0, Isaac_Z = 0.
So feet ARE below the pelvis in Isaac space. ✓

So the transform is: Isaac_Z = -URDF_Y (height = negative of URDF Y).

This means:
- Pelvis spawns at Isaac_Z = z_spawn
- Foot joint in Isaac = z_spawn + (-URDF_Y_of_foot)
- For feet to be at ground (Isaac_Z = 0): z_spawn = URDF_Y_of_foot

Max foot joint URDF_Y was 0.447 → spawn_z = 0.447 + foot_mesh_below_joint

Let me compute correctly.
"""

import xml.etree.ElementTree as ET
import numpy as np

urdf_path = r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf'
tree = ET.parse(urdf_path)
root = tree.getroot()

# Build joint tree (parent -> child, xyz offset in PARENT frame at rest pose = 0)
joints = {}
for j in root.findall('joint'):
    parent = j.find('parent').get('link')
    child  = j.find('child').get('link')
    origin = j.find('origin')
    xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
    rpy = list(map(float, origin.get('rpy','0 0 0').split())) if origin is not None else [0,0,0]
    joints[child] = (parent, np.array(xyz), np.array(rpy))

# Masses and local CoM offsets
masses, com_local = {}, {}
for link in root.findall('link'):
    name = link.get('name')
    inertial = link.find('inertial')
    if inertial is not None:
        masses[name] = float(inertial.find('mass').get('value'))
        origin = inertial.find('origin')
        xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
        com_local[name] = np.array(xyz)

# Compute world joint positions (rest pose, all joint angles = 0)
# At rest, joint rotations = 0, so just accumulate xyz offsets
cache = {}
def world_joint_pos(ln):
    if ln in cache: return cache[ln]
    if ln not in joints:
        cache[ln] = np.zeros(3)
        return np.zeros(3)
    parent, xyz_offset, rpy = joints[ln]
    parent_pos = world_joint_pos(parent)
    pos = parent_pos + xyz_offset  # valid at rest (joint angle = 0)
    cache[ln] = pos
    return pos

# Isaac conversion: Isaac_Z = -URDF_Y, Isaac_X = URDF_X, Isaac_Y = -URDF_Z
def to_isaac(urdf_xyz):
    """Convert URDF (Y-up) coords to Isaac (Z-up) coords."""
    x, y, z = urdf_xyz
    return np.array([x, -z, -y])  # Isaac: X=URDF_X, Y=-URDF_Z, Z=-URDF_Y

print("deer_standing.urdf — joint origins in Isaac space (pelvis joint at Isaac origin)")
print(f"{'Link':<20}  {'URDF joint (x,y,z)':>30}  {'Isaac joint (x,y,z)':>30}  Isaac_h")
print('-'*100)

foot_links = ['HR_foot', 'HL_foot', 'FR_foot', 'FL_foot']
for name in masses:
    jp = world_joint_pos(name)  # URDF space
    ip = to_isaac(jp)           # Isaac space
    com_l = com_local.get(name, np.zeros(3))  # local CoM offset (URDF link frame)
    com_w = jp + com_l          # world CoM (approx, valid for rest pose with aligned frames)
    ic = to_isaac(com_w)
    marker = ' <-- FOOT' if name in foot_links else ''
    print(f"  {name:<20}  ({jp[0]:+.3f},{jp[1]:+.3f},{jp[2]:+.3f})  -> ({ip[0]:+.3f},{ip[1]:+.3f},{ip[2]:+.3f})  h={ip[2]:+.3f}{marker}")

print()

# Find the lowest foot joint in Isaac (lowest Z = smallest Isaac_Z)
print("Foot joints in Isaac space:")
min_foot_z = 999.
for name in foot_links:
    jp = world_joint_pos(name)
    ip = to_isaac(jp)
    com_l = com_local.get(name, np.zeros(3))
    # Foot CoM local offset in Isaac (approximately)
    com_i = to_isaac(com_l)
    # The foot mesh extends BELOW the joint (lower Z in Isaac) by about 2x the CoM offset
    foot_bottom_z = ip[2] + com_i[2] * 2.0  # estimate mesh bottom below joint
    print(f"  {name}: joint_z={ip[2]:+.4f}  CoM_local_z={com_i[2]:+.4f}  est_foot_bottom={foot_bottom_z:+.4f}")
    if foot_bottom_z < min_foot_z:
        min_foot_z = foot_bottom_z

print()
print(f"Lowest estimated foot bottom (Isaac Z relative to pelvis joint) = {min_foot_z:+.4f} m")
spawn_z = -min_foot_z + 0.02  # spawn pelvis high enough that feet clear ground by 2cm
print(f"Recommended spawn Z = {spawn_z:.4f} m (feet 2cm above ground)")

# Mass-weighted CoM in Isaac
total_m = 0.; com_sum = np.zeros(3)
for name, mass in masses.items():
    jp = world_joint_pos(name)
    com_l = com_local.get(name, np.zeros(3))
    com_w = jp + com_l
    com_sum += mass * to_isaac(com_w)
    total_m += mass
com_isaac = com_sum / total_m

print()
print(f"Mass-weighted CoM (Isaac, relative to pelvis joint at origin):")
print(f"  Lateral (X):  {com_isaac[0]:+.4f} m")
print(f"  Fore/aft (Y): {com_isaac[1]:+.4f} m")
print(f"  Height  (Z):  {com_isaac[2]:+.4f} m  (above pelvis joint)")

com_abs_h = spawn_z + com_isaac[2]
print()
print(f"With spawn_z = {spawn_z:.4f} m:")
print(f"  Absolute CoM height = {com_abs_h:.4f} m above ground")
print(f"  Real white-tailed deer: ~0.65 m")
print(f"  Difference: {com_abs_h - 0.65:+.3f} m ({'too high' if com_abs_h > 0.65 else 'too low' if com_abs_h < 0.55 else 'good'})")

print()
print("="*50)
print("COMPARISON: old vs new")
print("="*50)

for urdf, label, old_spawn_z in [
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf',
     'deer_world.urdf (OLD, zero CoM)', 0.93),
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf',
     'deer_standing.urdf (NEW)', spawn_z),
]:
    t = ET.parse(urdf)
    r2 = t.getroot()
    ms2, cs2, js2 = {}, {}, {}
    for link in r2.findall('link'):
        nm = link.get('name')
        inertial = link.find('inertial')
        if inertial is not None:
            ms2[nm] = float(inertial.find('mass').get('value'))
            o = inertial.find('origin')
            xyz = list(map(float, o.get('xyz','0 0 0').split())) if o else [0,0,0]
            cs2[nm] = np.array(xyz)
    for j2 in r2.findall('joint'):
        parent = j2.find('parent').get('link')
        child  = j2.find('child').get('link')
        o = j2.find('origin')
        xyz = list(map(float, o.get('xyz','0 0 0').split())) if o else [0,0,0]
        js2[child] = (parent, np.array(xyz))
    cache2 = {}
    def wp2(ln):
        if ln in cache2: return cache2[ln]
        if ln not in js2:
            cache2[ln] = np.zeros(3)
            return np.zeros(3)
        parent, xyz = js2[ln]
        pos = wp2(parent) + xyz
        cache2[ln] = pos
        return pos
    tm2 = 0.; cv2 = np.zeros(3)
    for nm, mass in ms2.items():
        jp = wp2(nm)
        c = jp + cs2.get(nm, np.zeros(3))
        ic = to_isaac(c)
        tm2 += mass
        cv2 += mass * ic
    com2 = cv2 / tm2
    abs_h2 = old_spawn_z + com2[2]
    print(f"\n  {label}")
    print(f"    spawn_z={old_spawn_z:.3f}  CoM_rel_pelvis_z={com2[2]:+.4f}  =>  abs_CoM_h={abs_h2:.4f} m")
    print(f"    lateral={com2[0]:+.4f}  fore/aft={com2[1]:+.4f}")
