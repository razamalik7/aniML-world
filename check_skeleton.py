"""
Map out the full skeleton geometry of deer_standing.urdf to understand
where the pelvis joint is relative to the whole body, and compute
correct spawn Z.
"""
import xml.etree.ElementTree as ET
import numpy as np

urdf = r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf'
tree = ET.parse(urdf)
root = tree.getroot()

# Build joint tree
joints = {}
for j in root.findall('joint'):
    parent = j.find('parent').get('link')
    child  = j.find('child').get('link')
    origin = j.find('origin')
    xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
    joints[child] = (parent, np.array(xyz), j.get('type'), j.get('name'))

cache = {}
def world_pos(ln):
    if ln in cache: return cache[ln]
    if ln not in joints:
        cache[ln] = np.zeros(3)
        return np.zeros(3)
    parent, offset, jtype, jname = joints[ln]
    pos = world_pos(parent) + offset
    cache[ln] = pos
    return pos

# Get CoM offsets
coms = {}
masses = {}
for link in root.findall('link'):
    name = link.get('name')
    inertial = link.find('inertial')
    if inertial is not None:
        masses[name] = float(inertial.find('mass').get('value'))
        origin = inertial.find('origin')
        xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
        coms[name] = np.array(xyz)

# URDF is Y-up. After Rx(-90), Isaac Z = -URDF_Y, Isaac Y = URDF_Z, Isaac X = URDF_X
# So: Isaac height = -URDF_Y (with pelvis at origin)
# feet are at POSITIVE URDF_Y (below pelvis in Isaac = ground direction)

print("deer_standing.urdf skeleton (URDF Y-up, pelvis at origin)")
print(f"{'Link':<20} {'Joint orig (x,y,z)':>30}  {'CoM world (x,y,z)':>30}  Isaac_h")
print('-'*100)

all_links = list(masses.keys())
# also add foot-tip estimate (just joint origin of foot links, no child joint below them)
# The lowest point is the foot joint origin + the foot CoM offset (pointing downward in URDF)

foot_links = ['HR_foot', 'HL_foot', 'FR_foot', 'FL_foot']
max_y_foot_joint = -999   # highest URDF_Y among foot joint origins = lowest Isaac Z
max_y_foot_com   = -999   # accounting for CoM offset

for name in all_links:
    jp = world_pos(name)
    com_world = jp + coms.get(name, np.zeros(3))
    isaac_h = -jp[1]  # joint origin height in Isaac
    isaac_h_com = -com_world[1]  # CoM height in Isaac
    marker = ' <-- FOOT' if name in foot_links else ''
    print(f"  {name:<20} ({jp[0]:+.3f},{jp[1]:+.3f},{jp[2]:+.3f})  CoM({com_world[0]:+.3f},{com_world[1]:+.3f},{com_world[2]:+.3f})  jh={isaac_h:+.3f} ch={isaac_h_com:+.3f}{marker}")
    if name in foot_links:
        if jp[1] > max_y_foot_joint:
            max_y_foot_joint = jp[1]
        if com_world[1] > max_y_foot_com:
            max_y_foot_com = com_world[1]

print()
print(f"Max URDF_Y among foot JOINT origins = {max_y_foot_joint:.4f}")
print(f"Max URDF_Y among foot CoM positions  = {max_y_foot_com:.4f}")
print()
print("In Isaac Z-up, the feet are at:  Isaac_Z = pelvis_spawn_Z - URDF_Y_of_foot")
print("For feet to touch ground (Isaac_Z = 0):  pelvis_spawn_Z = URDF_Y_of_foot_bottom")
print()

# Foot mesh extends below the foot joint. Estimate bottom of foot as:
# foot joint origin Y + foot CoM offset Y (the CoM is toward the bottom of the hoof)
# Actually the CoM offset is the COM within the link's local frame from the joint origin.
# The foot goes from joint origin downward; CoM is roughly at midpoint.
# So bottom of foot ≈ joint_origin.Y + 2 * CoM_offset.Y (roughly double the CoM offset)
# But let's use the simpler bound: foot bottom ≈ foot joint Y + foot CoM Y offset * 1.5
print("Foot geometry estimates (URDF Y-up):")
for name in foot_links:
    jp = world_pos(name)
    com_offset = coms.get(name, np.zeros(3))
    # CoM offset is local to the joint frame. For a foot dangling downward, 
    # the local Y of the CoM offset indicates how far below the joint the CoM is.
    # The bottom of the foot mesh is roughly 2x that distance below the joint.
    foot_bottom_y = jp[1] + com_offset[1] * 1.5
    foot_bottom_isaac = -foot_bottom_y
    print(f"  {name}: joint_Y={jp[1]:+.4f}  CoM_offset_Y={com_offset[1]:+.4f}  "
          f"est_foot_bottom_Y={foot_bottom_y:+.4f}  Isaac_h_below_pelvis={foot_bottom_isaac:+.4f}")

# The lowest foot bottom (highest URDF_Y) defines how high the pelvis must spawn
max_foot_bottom_y = -999
for name in foot_links:
    jp = world_pos(name)
    com_offset = coms.get(name, np.zeros(3))
    foot_bottom_y = jp[1] + com_offset[1] * 1.5
    if foot_bottom_y > max_foot_bottom_y:
        max_foot_bottom_y = foot_bottom_y

margin = 0.02
spawn_z = max_foot_bottom_y + margin
print(f"\nMax estimated foot bottom URDF_Y = {max_foot_bottom_y:.4f}")
print(f"Recommended spawn Z (+ {margin}m margin) = {spawn_z:.4f} m")

# Now compute CoM absolute height with this spawn Z
total_m = 0.; cv = np.zeros(3)
for name, mass in masses.items():
    jp = world_pos(name)
    com_world = jp + coms.get(name, np.zeros(3))
    total_m += mass; cv += mass * com_world
com = cv / total_m
isaac_com_h = -com[1]  # height relative to pelvis
com_abs = spawn_z + isaac_com_h
print(f"\nCoM height relative to pelvis (Isaac) = {isaac_com_h:+.4f} m")
print(f"CoM absolute height = {spawn_z:.4f} + {isaac_com_h:+.4f} = {com_abs:.4f} m above ground")
print(f"Real deer CoM = ~0.65 m  =>  error = {com_abs - 0.65:+.3f} m")
if abs(com_abs - 0.65) < 0.10:
    print("=> GOOD: within 10cm of real deer CoM")
elif com_abs > 0.75:
    print("=> WARNING: still too high")
else:
    print("=> MARGINAL")
