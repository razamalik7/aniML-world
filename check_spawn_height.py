"""Compute foot position at default joint pose to determine correct spawn Z."""
import xml.etree.ElementTree as ET
import numpy as np

def get_foot_height(urdf_path, default_joint_pos=None):
    """Compute the lowest foot position at given joint angles (all others 0)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if default_joint_pos is None:
        default_joint_pos = {}

    # Build joint tree
    joints = {}
    for j in root.findall('joint'):
        jtype = j.get('type')
        name = j.get('name')
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        origin = j.find('origin')
        xyz = list(map(float, origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
        rpy = list(map(float, origin.get('rpy','0 0 0').split())) if origin is not None else [0,0,0]
        axis_el = j.find('axis')
        axis = list(map(float, axis_el.get('xyz','1 0 0').split())) if axis_el is not None else [1,0,0]
        angle = default_joint_pos.get(name, 0.0)
        joints[child] = {'parent': parent, 'xyz': np.array(xyz), 'type': jtype, 'axis': np.array(axis), 'angle': angle}

    def rot_matrix(axis, angle):
        """Rodrigues rotation formula."""
        ax = np.array(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        c, s = np.cos(angle), np.sin(angle)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    cache = {}
    def world_pos(link_name):
        if link_name in cache:
            return cache[link_name]
        if link_name not in joints:
            cache[link_name] = (np.zeros(3), np.eye(3))
            return np.zeros(3), np.eye(3)
        j = joints[link_name]
        parent_pos, parent_rot = world_pos(j['parent'])
        # Transform offset through parent rotation
        local_pos = parent_rot @ j['xyz']
        pos = parent_pos + local_pos
        # Apply joint rotation if revolute
        if j['type'] == 'revolute':
            joint_rot = rot_matrix(parent_rot @ j['axis'], j['angle'])
            rot = parent_rot @ joint_rot
        else:
            rot = parent_rot.copy()
        cache[link_name] = (pos, rot)
        return pos, rot

    # Find foot link positions
    # URDF Y-up: after Rx(-90), Isaac Z = -URDF Y, so lowest foot = max URDF Y
    foot_links = ['HR_foot', 'HL_foot', 'FR_foot', 'FL_foot']
    print(f"\n  Foot positions (URDF Y-up, pelvis at origin):")
    max_urdf_y = -999  # highest URDF Y = lowest Isaac Z offset
    for link in foot_links:
        pos, _ = world_pos(link)
        # In Isaac: height = -URDF_Y + pelvis_Z
        # Foot needs to be at ground (Z=0 in Isaac), so:
        # pelvis_spawn_Z = max(URDF_Y of all feet)
        print(f"    {link}: URDF(x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f})  Isaac_height_rel_pelvis={-pos[1]:.3f}")
        if pos[1] > max_urdf_y:
            max_urdf_y = pos[1]

    spawn_z = max_urdf_y + 0.01  # 1cm margin above ground
    print(f"\n  max foot URDF_Y = {max_urdf_y:.4f}")
    print(f"  => Recommended spawn Z = {spawn_z:.4f} m (puts feet at ground + 1cm)")
    return spawn_z

DEFAULT_JOINT_POS = {
    'pelvis_to_chest': 0.0,
    'pelvis_to_HR_thigh': 0.4,
    'HR_thigh_to_HR_shin': 0.0,
    'HR_shin_to_HR_foot': 0.0,
    'pelvis_to_HL_thigh': 0.4,
    'HL_thigh_to_HL_shin': 0.0,
    'HL_shin_to_HL_foot': 0.0,
    'chest_to_FR_thigh': -0.4,
    'FR_thigh_to_FR_shin': 0.0,
    'FR_shin_to_FR_foot': 0.0,
    'chest_to_FL_thigh': -0.4,
    'FL_thigh_to_FL_shin': 0.0,
    'FL_shin_to_FL_foot': 0.0,
    'chest_to_neck': 0.0,
    'neck_to_skull': 0.0,
}

for path, label in [
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf', 'deer_world.urdf (current)'),
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf', 'deer_standing.urdf (target)'),
]:
    print(f"\n{'='*55}\n  {label}")
    get_foot_height(path, DEFAULT_JOINT_POS)
