"""Verify that deer_rotated.urdf joint origins allow standing at identity root."""
import math
import numpy as np

ROOT_RX = 0.0  # identity for deer_rotated

JOINTS = {
    "pelvis_to_chest":      {"xyz": (0.0, 0.731998, -0.810849), "limits": (-0.524, 0.524)},
    "pelvis_to_HR_thigh":   {"xyz": (-0.127009, 0.635741, -0.717620), "limits": (-1.571, 1.571)},
    "HR_thigh_to_HR_shin":  {"xyz": (-0.000001, -0.145065, 0.300118), "limits": (-2.094, 0.175)},
    "HR_shin_to_HR_foot":   {"xyz": (-0.008380, -0.232770, -0.146574), "limits": (-0.785, 0.785)},
    "pelvis_to_HL_thigh":   {"xyz": (0.127008, 0.635740, -0.717620), "limits": (-1.571, 1.571)},
    "HL_thigh_to_HL_shin":  {"xyz": (-0.000001, -0.145065, 0.300118), "limits": (-2.094, 0.175)},
    "HL_shin_to_HL_foot":   {"xyz": (0.008379, -0.232769, -0.146575), "limits": (-0.785, 0.785)},
    "chest_to_FR_thigh":    {"xyz": (-0.122245, 0.127611, 0.900051), "limits": (-1.571, 1.571)},
    "FR_thigh_to_FR_shin":  {"xyz": (0.000050, -0.194555, -0.120075), "limits": (-2.094, 0.175)},
    "FR_shin_to_FR_foot":   {"xyz": (0.001356, -0.220814, 0.182375), "limits": (-0.785, 0.785)},
    "chest_to_FL_thigh":    {"xyz": (0.122205, 0.127552, 0.899991), "limits": (-1.571, 1.571)},
    "FL_thigh_to_FL_shin":  {"xyz": (-0.000067, -0.194523, -0.120127), "limits": (-2.094, 0.175)},
    "FL_shin_to_FL_foot":   {"xyz": (-0.000251, -0.218082, 0.185639), "limits": (-0.785, 0.785)},
    "chest_to_neck":        {"xyz": (0.000002, 0.313195, 0.790129), "limits": (-0.698, 0.698)},
    "neck_to_skull":        {"xyz": (0.000003, 0.465927, 0.203012), "limits": (-0.524, 0.873)},
}

CHAIN = {
    "pelvis": None,
    "chest": ("pelvis", "pelvis_to_chest"),
    "neck": ("chest", "chest_to_neck"),
    "skull": ("neck", "neck_to_skull"),
    "HR_thigh": ("pelvis", "pelvis_to_HR_thigh"),
    "HR_shin": ("HR_thigh", "HR_thigh_to_HR_shin"),
    "HR_foot": ("HR_shin", "HR_shin_to_HR_foot"),
    "HL_thigh": ("pelvis", "pelvis_to_HL_thigh"),
    "HL_shin": ("HL_thigh", "HL_thigh_to_HL_shin"),
    "HL_foot": ("HL_shin", "HL_shin_to_HL_foot"),
    "FR_thigh": ("chest", "chest_to_FR_thigh"),
    "FR_shin": ("FR_thigh", "FR_thigh_to_FR_shin"),
    "FR_foot": ("FR_shin", "FR_shin_to_FR_foot"),
    "FL_thigh": ("chest", "chest_to_FL_thigh"),
    "FL_shin": ("FL_thigh", "FL_thigh_to_FL_shin"),
    "FL_foot": ("FL_shin", "FL_shin_to_FL_foot"),
}


def rx_mat(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])


def fk(joint_angles):
    root_R = rx_mat(ROOT_RX)
    frames = {"pelvis": (np.array([0,0,0]), root_R.copy())}
    
    order = ["chest", "neck", "skull",
             "HR_thigh", "HR_shin", "HR_foot",
             "HL_thigh", "HL_shin", "HL_foot",
             "FR_thigh", "FR_shin", "FR_foot",
             "FL_thigh", "FL_shin", "FL_foot"]
    
    for link in order:
        parent, jname = CHAIN[link]
        xyz = np.array(JOINTS[jname]["xyz"])
        theta = joint_angles.get(jname, 0.0)
        p_pos, p_R = frames[parent]
        child_pos = p_pos + p_R @ xyz
        child_R = p_R @ rx_mat(math.degrees(theta))
        frames[link] = (child_pos, child_R)
    
    return {n: p for n, (p, _) in frames.items()}


# Zero joints
print("=== deer_rotated.urdf at identity root, zero joints ===")
pos = fk({})
for name in ["pelvis", "chest", "neck", "skull", "HR_foot", "HL_foot", "FR_foot", "FL_foot"]:
    p = pos[name]
    print(f"  {name:12s}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

print(f"\n  Pelvis Z: {pos['pelvis'][2]:.3f}")
print(f"  HR_foot below pelvis: {pos['pelvis'][2] - pos['HR_foot'][2]:.3f}")
print(f"  FR_foot below pelvis: {pos['pelvis'][2] - pos['FR_foot'][2]:.3f}")

# Search for standing angles
print("\n=== Searching for standing angles ===")
best_score = -999
best_angles = {}

for h_hip in np.arange(-1.5, 1.5, 0.3):
    for h_knee in np.arange(-2.0, 0.17, 0.3):
        angles = {}
        angles["pelvis_to_HR_thigh"] = h_hip
        angles["HR_thigh_to_HR_shin"] = h_knee
        angles["pelvis_to_HL_thigh"] = h_hip
        angles["HL_thigh_to_HL_shin"] = h_knee
        
        for f_hip in np.arange(-1.5, 1.5, 0.3):
            for f_knee in np.arange(-2.0, 0.17, 0.3):
                angles["chest_to_FR_thigh"] = f_hip
                angles["FR_thigh_to_FR_shin"] = f_knee
                angles["chest_to_FL_thigh"] = f_hip
                angles["FL_thigh_to_FL_shin"] = f_knee
                
                pos = fk(angles)
                feet_z = [pos[f][2] for f in ["HR_foot","HL_foot","FR_foot","FL_foot"]]
                pelvis_z = pos["pelvis"][2]
                
                min_fz = min(feet_z)
                spread = max(feet_z) - min(feet_z)
                above = pelvis_z - np.mean(feet_z)
                
                if min_fz < pelvis_z and above > 0:
                    score = above * 10 - spread * 5
                    if score > best_score:
                        best_score = score
                        best_angles = dict(angles)

if best_angles:
    print(f"Best score: {best_score:.3f}")
    pos = fk(best_angles)
    for name in ["pelvis", "chest", "HR_foot", "HL_foot", "FR_foot", "FL_foot"]:
        p = pos[name]
        print(f"  {name:12s}: Z={p[2]:+.3f}")
    
    feet_z = [pos[f][2] for f in ["HR_foot","HL_foot","FR_foot","FL_foot"]]
    print(f"\n  Pelvis above mean foot: {pos['pelvis'][2] - np.mean(feet_z):.3f}m")
    print(f"  Foot spread: {max(feet_z)-min(feet_z):.3f}m")
    
    print("\n  Joint angles:")
    for jn, a in sorted(best_angles.items()):
        if abs(a) > 0.01:
            print(f"    {jn}: {a:.3f} rad ({math.degrees(a):.1f} deg)")
    
    spawn_z = -min(feet_z) + 0.03
    print(f"\n  Recommended spawn Z: {spawn_z:.3f}")
else:
    print("No standing config found!")
