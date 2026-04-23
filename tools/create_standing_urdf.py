"""
Create a standing-ready URDF + pre-transformed STL meshes.

Problem: deer.urdf looks correct visually at Rx(-91deg) but can't physically
stand at that orientation. deer_rotated.urdf CAN physically stand at identity
but its meshes are disconnected because it applies rpy=-90deg without the
necessary per-link translation offsets.

Solution: Bake the Rx(-91deg) visual rotation + per-link position correction
directly into new STL mesh files. Then create a clean URDF with
deer_rotated.urdf's joint chain and rpy=0, xyz=0 on all meshes.

Transform per link:
  v_new = [R_visual - R_160] * linkpos_deer + R_visual * v
  n_new = R_visual * n

where R_visual = Rx(-90.86deg) from deer_standing_physics.usd quaternion
      R_160    = Rx(160deg) the rotation that maps deer.urdf -> deer_rotated.urdf
"""

import struct
import math
import os

# --- Rotation matrices ---

def quat_to_matrix(w, x, y, z):
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return [
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ]

def rx_matrix(deg):
    """Rotation matrix around X-axis by deg degrees."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]

def mat_vec(M, v):
    return tuple(sum(M[i][j] * v[j] for j in range(3)) for i in range(3))

def mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(3)] for i in range(3)]


# Exact visual rotation from deer_standing_physics.usd quaternion (w,x,y,z)
R_VISUAL = quat_to_matrix(0.7018, -0.7124, 0.0, 0.0)

# Rotation that maps deer.urdf joint origins -> deer_rotated.urdf joint origins
R_160 = rx_matrix(160.0)

# Delta matrix for per-link position correction
R_DELTA = mat_sub(R_VISUAL, R_160)


# --- Forward kinematics from deer.urdf ---

JOINT_TREE = {
    "pelvis":   (None,       (0, 0, 0)),
    "chest":    ("pelvis",   (0.0, -0.965180, 0.511591)),
    "neck":     ("chest",    (0.000002, -0.024067, -0.849597)),
    "skull":    ("neck",     (0.000003, -0.368394, -0.350125)),
    "HR_thigh": ("pelvis",   (-0.127009, -0.842842, 0.456906)),
    "HR_shin":  ("HR_thigh", (-0.000001, 0.238963, -0.232403)),
    "HR_foot":  ("HR_shin",  (-0.008380, 0.168601, 0.217347)),
    "HL_thigh": ("pelvis",   (0.127008, -0.842841, 0.456906)),
    "HL_shin":  ("HL_thigh", (-0.000001, 0.238963, -0.232403)),
    "HL_foot":  ("HL_shin",  (0.008379, 0.168600, 0.217347)),
    "FR_thigh": ("chest",    (-0.122245, 0.187920, -0.889417)),
    "FR_shin":  ("FR_thigh", (0.000050, 0.141754, 0.179375)),
    "FR_foot":  ("FR_shin",  (0.001356, 0.269873, -0.095854)),
    "FL_thigh": ("chest",    (0.122205, 0.187955, -0.889340)),
    "FL_shin":  ("FL_thigh", (-0.000067, 0.141706, 0.179413)),
    "FL_foot":  ("FL_shin",  (-0.000251, 0.268422, -0.099855)),
}

_world_cache = {}

def world_pos(link):
    if link in _world_cache:
        return _world_cache[link]
    parent, xyz = JOINT_TREE[link]
    if parent is None:
        _world_cache[link] = xyz
        return xyz
    pp = world_pos(parent)
    result = tuple(pp[i] + xyz[i] for i in range(3))
    _world_cache[link] = result
    return result


# --- Binary STL I/O ---

def read_stl(path):
    with open(path, "rb") as f:
        header = f.read(80)
        (num_tri,) = struct.unpack("<I", f.read(4))
        triangles = []
        for _ in range(num_tri):
            data = struct.unpack("<12fH", f.read(50))
            triangles.append((data[0:3], data[3:6], data[6:9], data[9:12], data[12]))
        return header, triangles

def write_stl(path, header, triangles):
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for n, v1, v2, v3, attr in triangles:
            f.write(struct.pack("<12fH", *n, *v1, *v2, *v3, attr))


def transform_stl(src, dst, link_name):
    """Read src STL, apply per-link transform, write to dst."""
    wpos = world_pos(link_name)
    offset = mat_vec(R_DELTA, wpos)

    header, triangles = read_stl(src)
    new_tris = []
    for normal, v1, v2, v3, attr in triangles:
        nn = mat_vec(R_VISUAL, normal)
        nv1 = tuple(offset[i] + mat_vec(R_VISUAL, v1)[i] for i in range(3))
        nv2 = tuple(offset[i] + mat_vec(R_VISUAL, v2)[i] for i in range(3))
        nv3 = tuple(offset[i] + mat_vec(R_VISUAL, v3)[i] for i in range(3))
        new_tris.append((nn, nv1, nv2, nv3, attr))
    write_stl(dst, header, new_tris)


# --- Main ---

MESH_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "deer_urdf", "meshes")
OUT_DIR = os.path.join(MESH_DIR, "standing")
os.makedirs(OUT_DIR, exist_ok=True)

LINKS = [
    "pelvis", "chest", "neck", "skull",
    "HR_thigh", "HR_shin", "HR_foot",
    "HL_thigh", "HL_shin", "HL_foot",
    "FR_thigh", "FR_shin", "FR_foot",
    "FL_thigh", "FL_shin", "FL_foot",
]

print("=== Creating standing-ready mesh files ===\n")
for link in LINKS:
    wpos = world_pos(link)
    offset = mat_vec(R_DELTA, wpos)
    print(f"{link:12s}  world_deer=({wpos[0]:+.4f}, {wpos[1]:+.4f}, {wpos[2]:+.4f})  "
          f"offset=({offset[0]:+.5f}, {offset[1]:+.5f}, {offset[2]:+.5f})")

    for suffix in ("collision", "visual"):
        src = os.path.join(MESH_DIR, f"{link}_{suffix}.stl")
        if not os.path.exists(src):
            print(f"  SKIP {link}_{suffix}.stl (not found)")
            continue
        dst = os.path.join(OUT_DIR, f"{link}_{suffix}.stl")
        transform_stl(src, dst, link)
        h, tris = read_stl(dst)
        print(f"  {suffix:10s} -> {os.path.basename(dst)} ({len(tris)} triangles)")

print(f"\nAll meshes written to: {OUT_DIR}")

# --- Generate URDF ---

URDF_TEMPLATE = """\
<?xml version="1.0"?>
<!--
  deer_standing.urdf
  Auto-generated from deer_rotated.urdf joint chain + pre-transformed meshes.
  Uses visual STLs for <visual>, collision STLs for <collision>.
  All mesh origins: xyz="0 0 0" rpy="0 0 0" (transforms baked into STL data).
  Root at identity -> deer stands upright.
-->
<robot name="deer">
{links}
{joints}
</robot>
"""

# Inertial data from deer_rotated.urdf
INERTIALS = {
    "pelvis":   {"xyz": "-0.000001 0.728145 -0.520502",  "mass": "25.000000", "ixx": "1.461721", "iyy": "1.372512", "izz": "0.844660"},
    "chest":    {"xyz": "-0.000028 -0.622695 1.286595",   "mass": "22.000000", "ixx": "4.047500", "iyy": "3.796640", "izz": "1.030469"},
    "neck":     {"xyz": "-0.000019 -0.921248 0.170690",   "mass": "3.500000",  "ixx": "0.335532", "iyy": "0.160215", "izz": "0.273118"},
    "skull":    {"xyz": "-0.000003 -1.361236 -0.243162",  "mass": "2.500000",  "ixx": "0.239443", "iyy": "0.304650", "izz": "0.404810"},
    "HR_thigh": {"xyz": "0.212361 -0.703494 0.951129",    "mass": "1.533589",  "ixx": "0.070828", "iyy": "0.051351", "izz": "0.049459"},
    "HR_shin":  {"xyz": "0.147567 -0.532824 0.422427",    "mass": "1.468330",  "ixx": "0.069088", "iyy": "0.033817", "izz": "0.046364"},
    "HR_foot":  {"xyz": "0.137587 -0.431249 0.677384",    "mass": "1.174664",  "ixx": "0.014113", "iyy": "0.002552", "izz": "0.012383"},
    "HL_thigh": {"xyz": "-0.212362 -0.703494 0.951128",   "mass": "1.533589",  "ixx": "0.070828", "iyy": "0.051351", "izz": "0.049459"},
    "HL_shin":  {"xyz": "-0.147571 -0.532823 0.422427",   "mass": "1.468330",  "ixx": "0.069088", "iyy": "0.033817", "izz": "0.046364"},
    "HL_foot":  {"xyz": "-0.137586 -0.431250 0.677383",   "mass": "1.174664",  "ixx": "0.014113", "iyy": "0.002552", "izz": "0.012383"},
    "FR_thigh": {"xyz": "0.193567 -1.012330 -0.173887",   "mass": "1.957774",  "ixx": "0.142444", "iyy": "0.104511", "izz": "0.080061"},
    "FR_shin":  {"xyz": "0.131006 -0.767364 0.178798",    "mass": "1.174664",  "ixx": "0.029326", "iyy": "0.012046", "izz": "0.027344"},
    "FR_foot":  {"xyz": "0.124637 -0.588919 -0.099479",   "mass": "1.174664",  "ixx": "0.011062", "iyy": "0.002700", "izz": "0.009295"},
    "FL_thigh": {"xyz": "-0.193570 -1.011733 -0.173943",  "mass": "1.990403",  "ixx": "0.144616", "iyy": "0.106260", "izz": "0.081199"},
    "FL_shin":  {"xyz": "-0.130968 -0.766803 0.178768",   "mass": "1.174664",  "ixx": "0.029230", "iyy": "0.012041", "izz": "0.027251"},
    "FL_foot":  {"xyz": "-0.124912 -0.596287 -0.115251",  "mass": "1.174664",  "ixx": "0.010357", "iyy": "0.002014", "izz": "0.009310"},
}

# Joint data from deer_rotated.urdf
JOINTS_ROTATED = [
    ("pelvis_to_chest",     "pelvis", "chest",    "0.000000 0.731998 -0.810849", "-0.523599", "0.523599", "120.000000", "8.000000"),
    ("chest_to_neck",       "chest",  "neck",     "0.000002 0.313195 0.790129",  "-0.698132", "0.698132", "50.000000",  "8.000000"),
    ("neck_to_skull",       "neck",   "skull",    "0.000003 0.465927 0.203012",  "-0.523599", "0.872665", "30.000000",  "8.000000"),
    ("pelvis_to_HR_thigh",  "pelvis", "HR_thigh", "-0.127009 0.635741 -0.717620", "-1.570796", "1.570796", "100.000000", "10.000000"),
    ("HR_thigh_to_HR_shin", "HR_thigh","HR_shin", "-0.000001 -0.145065 0.300118", "-2.094395", "0.174533", "80.000000",  "10.000000"),
    ("HR_shin_to_HR_foot",  "HR_shin","HR_foot",  "-0.008380 -0.232770 -0.146574", "-0.785398", "0.785398", "60.000000", "10.000000"),
    ("pelvis_to_HL_thigh",  "pelvis", "HL_thigh", "0.127008 0.635740 -0.717620", "-1.570796", "1.570796", "100.000000", "10.000000"),
    ("HL_thigh_to_HL_shin", "HL_thigh","HL_shin", "-0.000001 -0.145065 0.300118", "-2.094395", "0.174533", "80.000000",  "10.000000"),
    ("HL_shin_to_HL_foot",  "HL_shin","HL_foot",  "0.008379 -0.232769 -0.146575", "-0.785398", "0.785398", "60.000000", "10.000000"),
    ("chest_to_FR_thigh",   "chest",  "FR_thigh", "-0.122245 0.127611 0.900051", "-1.570796", "1.570796", "100.000000", "10.000000"),
    ("FR_thigh_to_FR_shin", "FR_thigh","FR_shin", "0.000050 -0.194555 -0.120075", "-2.094395", "0.174533", "80.000000", "10.000000"),
    ("FR_shin_to_FR_foot",  "FR_shin","FR_foot",  "0.001356 -0.220814 0.182375", "-0.785398", "0.785398", "60.000000",  "10.000000"),
    ("chest_to_FL_thigh",   "chest",  "FL_thigh", "0.122205 0.127552 0.899991", "-1.570796", "1.570796", "100.000000", "10.000000"),
    ("FL_thigh_to_FL_shin", "FL_thigh","FL_shin", "-0.000067 -0.194523 -0.120127", "-2.094395", "0.174533", "80.000000", "10.000000"),
    ("FL_shin_to_FL_foot",  "FL_shin","FL_foot",  "-0.000251 -0.218082 0.185639", "-0.785398", "0.785398", "60.000000",  "10.000000"),
]


def make_link(name):
    d = INERTIALS[name]
    vis_stl = f"meshes/standing/{name}_visual.stl"
    col_stl = f"meshes/standing/{name}_collision.stl"
    vis_src = os.path.join(OUT_DIR, f"{name}_visual.stl")
    col_src = os.path.join(OUT_DIR, f"{name}_collision.stl")
    if not os.path.exists(vis_src):
        vis_stl = col_stl
    return f"""  <link name="{name}">
    <inertial>
      <origin xyz="{d['xyz']}" rpy="0 0 0"/>
      <mass value="{d['mass']}"/>
      <inertia ixx="{d['ixx']}" ixy="0" ixz="0" iyy="{d['iyy']}" iyz="0" izz="{d['izz']}"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{col_stl}"/>
      </geometry>
    </collision>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{vis_stl}"/>
      </geometry>
    </visual>
  </link>"""


def make_joint(name, parent, child, xyz, lo, hi, effort, vel):
    return f"""  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz}" rpy="0 0 0"/>
    <axis xyz="1.000000 0.000000 0.000000"/>
    <limit lower="{lo}" upper="{hi}" effort="{effort}" velocity="{vel}"/>
  </joint>"""


links_xml = "\n".join(make_link(l) for l in LINKS)
joints_xml = "\n".join(make_joint(*j) for j in JOINTS_ROTATED)
urdf_content = URDF_TEMPLATE.format(links=links_xml, joints=joints_xml)

urdf_path = os.path.join(os.path.dirname(__file__), "..", "assets", "deer_urdf", "deer_standing.urdf")
with open(urdf_path, "w") as f:
    f.write(urdf_content)

print(f"\nURDF written to: {urdf_path}")
print("\nDone! Next: import deer_standing.urdf into Isaac Sim with identity root rotation.")
