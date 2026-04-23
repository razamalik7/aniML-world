"""
Definitive diagnostic: Are the STL meshes in link-local or global coordinates?

Reads each collision STL, computes its bounding box center, and compares against:
  (a) Link-local: center should be near (0,0,0) for each link
  (b) Global (deer.urdf frame): center should match the cumulative joint chain position
  (c) Global (deer_rotated.urdf frame): same but with pre-rotated joint origins
"""

import struct, math, os, sys

MESH_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "deer_urdf", "meshes")

DEER_URDF_JOINTS = {
    "pelvis":   {"parent": None,     "xyz": (0, 0, 0)},
    "chest":    {"parent": "pelvis", "xyz": (0, -0.965180, 0.511591)},
    "neck":     {"parent": "chest",  "xyz": (0, -0.272830, 0.802370)},
    "skull":    {"parent": "neck",   "xyz": (0, -0.221010, 0.440790)},
    "HR_thigh": {"parent": "pelvis", "xyz": (-0.127009, -0.829960, 0.445530)},
    "HR_shin":  {"parent": "HR_thigh","xyz": (-0.000001, -0.326720, 0.040820)},
    "HR_foot":  {"parent": "HR_shin", "xyz": (-0.008380, -0.200460, -0.248430)},
    "HL_thigh": {"parent": "pelvis", "xyz": (0.127008, -0.829960, 0.445530)},
    "HL_shin":  {"parent": "HL_thigh","xyz": (-0.000001, -0.326720, 0.040820)},
    "HL_foot":  {"parent": "HL_shin", "xyz": (0.008379, -0.200460, -0.248430)},
    "FR_thigh": {"parent": "chest",  "xyz": (-0.122245, -0.552290, 0.786230)},
    "FR_shin":  {"parent": "FR_thigh","xyz": (0.000050, -0.178050, -0.234670)},
    "FR_foot":  {"parent": "FR_shin", "xyz": (0.001356, -0.072370, -0.167570)},
    "FL_thigh": {"parent": "chest",  "xyz": (0.122205, -0.552350, 0.786170)},
    "FL_shin":  {"parent": "FL_thigh","xyz": (-0.000067, -0.178020, -0.234720)},
    "FL_foot":  {"parent": "FL_shin", "xyz": (-0.000251, -0.069650, -0.164300)},
}

DEER_ROTATED_JOINTS = {
    "pelvis":   {"parent": None,     "xyz": (0, 0, 0)},
    "chest":    {"parent": "pelvis", "xyz": (0.0, 0.731998, -0.810849)},
    "neck":     {"parent": "chest",  "xyz": (0.000002, 0.313195, 0.790129)},
    "skull":    {"parent": "neck",   "xyz": (0.000003, 0.465927, 0.203012)},
    "HR_thigh": {"parent": "pelvis", "xyz": (-0.127009, 0.635741, -0.717620)},
    "HR_shin":  {"parent": "HR_thigh","xyz": (-0.000001, -0.145065, 0.300118)},
    "HR_foot":  {"parent": "HR_shin", "xyz": (-0.008380, -0.232770, -0.146574)},
    "HL_thigh": {"parent": "pelvis", "xyz": (0.127008, 0.635740, -0.717620)},
    "HL_shin":  {"parent": "HL_thigh","xyz": (-0.000001, -0.145065, 0.300118)},
    "HL_foot":  {"parent": "HL_shin", "xyz": (0.008379, -0.232769, -0.146575)},
    "FR_thigh": {"parent": "chest",  "xyz": (-0.122245, 0.127611, 0.900051)},
    "FR_shin":  {"parent": "FR_thigh","xyz": (0.000050, -0.194555, -0.120075)},
    "FR_foot":  {"parent": "FR_shin", "xyz": (0.001356, -0.220814, 0.182375)},
    "FL_thigh": {"parent": "chest",  "xyz": (0.122205, 0.127552, 0.899991)},
    "FL_shin":  {"parent": "FL_thigh","xyz": (-0.000067, -0.194523, -0.120127)},
    "FL_foot":  {"parent": "FL_shin", "xyz": (-0.000251, -0.218082, 0.185639)},
}


def global_pos(joints, link):
    pos = [0.0, 0.0, 0.0]
    node = link
    while node is not None:
        xyz = joints[node]["xyz"]
        pos[0] += xyz[0]
        pos[1] += xyz[1]
        pos[2] += xyz[2]
        node = joints[node]["parent"]
    return tuple(pos)


def read_stl_bbox(path):
    with open(path, "rb") as f:
        f.read(80)
        n_tri = struct.unpack("<I", f.read(4))[0]
        xs, ys, zs = [], [], []
        for _ in range(n_tri):
            f.read(12)  # normal
            for _ in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                xs.append(x)
                ys.append(y)
                zs.append(z)
            f.read(2)  # attrib
    if not xs:
        return None
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    ct = ((mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2)
    sz = (mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])
    return {"min": mn, "max": mx, "center": ct, "size": sz, "n_tri": n_tri}


def dist(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))


def rx_minus90(x, y, z):
    """Rx(-90°): (x,y,z) → (x, z, -y)"""
    return (x, z, -y)


print("=" * 90)
print("STL MESH COORDINATE FRAME DIAGNOSTIC")
print("=" * 90)

links = ["pelvis", "chest", "neck", "skull",
         "HR_thigh", "HR_shin", "HR_foot",
         "HL_thigh", "HL_shin", "HL_foot",
         "FR_thigh", "FR_shin", "FR_foot",
         "FL_thigh", "FL_shin", "FL_foot"]

for link in links:
    stl_path = os.path.join(MESH_DIR, f"{link}_collision.stl")
    if not os.path.exists(stl_path):
        print(f"\n{link}: FILE MISSING")
        continue

    bb = read_stl_bbox(stl_path)
    if bb is None:
        print(f"\n{link}: EMPTY STL")
        continue

    gp_orig = global_pos(DEER_URDF_JOINTS, link)
    gp_rot = global_pos(DEER_ROTATED_JOINTS, link)

    ct = bb["center"]
    ct_rotated = rx_minus90(*ct)

    print(f"\n{'-'*90}")
    print(f"  LINK: {link}")
    print(f"  STL center (raw):       ({ct[0]:+.4f}, {ct[1]:+.4f}, {ct[2]:+.4f})")
    print(f"  STL center (Rx -90°):   ({ct_rotated[0]:+.4f}, {ct_rotated[1]:+.4f}, {ct_rotated[2]:+.4f})")
    print(f"  STL size:               ({bb['size'][0]:.4f}, {bb['size'][1]:.4f}, {bb['size'][2]:.4f})  [{bb['n_tri']} tris]")
    print(f"  deer.urdf global pos:   ({gp_orig[0]:+.4f}, {gp_orig[1]:+.4f}, {gp_orig[2]:+.4f})")
    print(f"  deer_rotated global pos:({gp_rot[0]:+.4f}, {gp_rot[1]:+.4f}, {gp_rot[2]:+.4f})")
    print(f"  dist(raw->origin):      {dist(ct, (0,0,0)):.4f}")
    print(f"  dist(raw->deer.urdf):   {dist(ct, gp_orig):.4f}")
    print(f"  dist(Rx90->deer.urdf):  {dist(ct_rotated, gp_orig):.4f}")
    print(f"  dist(raw->rotated):     {dist(ct, gp_rot):.4f}")
    print(f"  dist(Rx90->rotated):    {dist(ct_rotated, gp_rot):.4f}")

print(f"\n{'='*90}")
print("INTERPRETATION:")
print("  If 'dist(raw->origin)' is small for ALL links -> meshes are in link-local coords")
print("  If 'dist(raw->deer.urdf)' is small -> meshes are in deer.urdf global coords (Y-up Blender)")
print("  If 'dist(Rx90->deer.urdf)' is small -> meshes are in Blender frame, need Rx(-90) to match URDF")
print("  If 'dist(raw->rotated)' is small -> meshes are in deer_rotated.urdf global coords")
print("=" * 90)
