"""Compute IK joint angles for deer.urdf standing at Rx(-91deg).

Uses the actual deer.urdf joint chain (from the USD inspection).
Finds joint angles that put all feet below the pelvis at the ground plane.
"""
import math
import numpy as np
from itertools import product

ROOT_RX = -91.0  # degrees

JOINTS = {
    "pelvis_to_chest":      {"parent": "pelvis", "child": "chest",    "xyz": (0, -0.965180, 0.511591), "limits": (-0.524, 0.524)},
    "chest_to_neck":        {"parent": "chest",  "child": "neck",     "xyz": (0.000002, -0.024067, -0.849597), "limits": (-0.698, 0.698)},
    "neck_to_skull":        {"parent": "neck",   "child": "skull",    "xyz": (0.000003, -0.368394, -0.350125), "limits": (-0.524, 0.873)},
    "pelvis_to_HR_thigh":   {"parent": "pelvis", "child": "HR_thigh", "xyz": (-0.127009, -0.842842, 0.456906), "limits": (-1.571, 1.571)},
    "HR_thigh_to_HR_shin":  {"parent": "HR_thigh","child": "HR_shin", "xyz": (-0.000001, 0.238963, -0.232403), "limits": (-2.094, 0.175)},
    "HR_shin_to_HR_foot":   {"parent": "HR_shin", "child": "HR_foot", "xyz": (-0.008380, 0.168601, 0.217347), "limits": (-0.785, 0.785)},
    "pelvis_to_HL_thigh":   {"parent": "pelvis", "child": "HL_thigh", "xyz": (0.127008, -0.842841, 0.456906), "limits": (-1.571, 1.571)},
    "HL_thigh_to_HL_shin":  {"parent": "HL_thigh","child": "HL_shin", "xyz": (-0.000001, 0.238963, -0.232403), "limits": (-2.094, 0.175)},
    "HL_shin_to_HL_foot":   {"parent": "HL_shin", "child": "HL_foot", "xyz": (0.008379, 0.168600, 0.217347), "limits": (-0.785, 0.785)},
    "chest_to_FR_thigh":    {"parent": "chest",  "child": "FR_thigh", "xyz": (-0.122245, 0.187920, -0.889417), "limits": (-1.571, 1.571)},
    "FR_thigh_to_FR_shin":  {"parent": "FR_thigh","child": "FR_shin", "xyz": (0.000050, 0.141754, 0.179375), "limits": (-2.094, 0.175)},
    "FR_shin_to_FR_foot":   {"parent": "FR_shin", "child": "FR_foot", "xyz": (0.001356, 0.269873, -0.095854), "limits": (-0.785, 0.785)},
    "chest_to_FL_thigh":    {"parent": "chest",  "child": "FL_thigh", "xyz": (0.122205, 0.187955, -0.889340), "limits": (-1.571, 1.571)},
    "FL_thigh_to_FL_shin":  {"parent": "FL_thigh","child": "FL_shin", "xyz": (-0.000067, 0.141706, 0.179413), "limits": (-2.094, 0.175)},
    "FL_shin_to_FL_foot":   {"parent": "FL_shin", "child": "FL_foot", "xyz": (-0.000251, 0.268422, -0.099855), "limits": (-0.785, 0.785)},
}


def rx_mat(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])


def fk(joint_angles, root_rx_deg=ROOT_RX):
    """Forward kinematics: returns world positions of all links."""
    root_R = rx_mat(root_rx_deg)
    
    frames = {}  # link -> (pos_world, R_world)
    frames["pelvis"] = (np.array([0.0, 0.0, 0.0]), root_R.copy())
    
    for jname, jdata in JOINTS.items():
        parent = jdata["parent"]
        child = jdata["child"]
        xyz = np.array(jdata["xyz"])
        theta = joint_angles.get(jname, 0.0)
        
        p_pos, p_R = frames[parent]
        child_pos = p_pos + p_R @ xyz
        child_R = p_R @ rx_mat(math.degrees(theta))
        frames[child] = (child_pos, child_R)
    
    return {name: pos for name, (pos, _) in frames.items()}


def score_standing(joint_angles):
    """Score how good a set of joint angles is for standing.
    
    Criteria:
    - All 4 feet at similar Z (close to each other)
    - All feet below pelvis
    - Pelvis as high as possible
    - Chest at similar height to pelvis (level spine)
    """
    pos = fk(joint_angles)
    
    pelvis_z = pos["pelvis"][2]
    chest_z = pos["chest"][2]
    
    feet = ["HR_foot", "HL_foot", "FR_foot", "FL_foot"]
    foot_zs = [pos[f][2] for f in feet]
    
    min_foot_z = min(foot_zs)
    max_foot_z = max(foot_zs)
    mean_foot_z = sum(foot_zs) / 4
    foot_spread = max_foot_z - min_foot_z
    
    pelvis_above_feet = pelvis_z - mean_foot_z
    spine_level = abs(chest_z - pelvis_z)
    
    score = 0.0
    score += pelvis_above_feet * 10.0  # want pelvis above feet
    score -= foot_spread * 5.0  # want feet at same level
    score -= spine_level * 2.0  # want level spine
    if min_foot_z < pelvis_z:
        score += 5.0  # bonus for feet below pelvis
    
    return score, {
        "pelvis_z": pelvis_z,
        "chest_z": chest_z,
        "foot_zs": {f: pos[f][2] for f in feet},
        "foot_spread": foot_spread,
        "pelvis_above_feet": pelvis_above_feet,
    }


# Grid search over key joint angles
print("Searching for optimal standing joint angles...")
print(f"Root rotation: Rx({ROOT_RX}deg)")
print()

# First, show zero-angle baseline
zero_angles = {j: 0.0 for j in JOINTS}
s, info = score_standing(zero_angles)
print(f"ZERO JOINTS: score={s:.3f}")
print(f"  pelvis_z={info['pelvis_z']:.3f}, chest_z={info['chest_z']:.3f}")
for f, z in info['foot_zs'].items():
    print(f"  {f}: z={z:.3f}")
print(f"  pelvis_above_feet={info['pelvis_above_feet']:.3f}, spread={info['foot_spread']:.3f}")
print()

# Search hind leg angles
print("--- Searching hind leg angles (HR) ---")
best_hind_score = -999
best_hind = {}

for hip in np.arange(-1.5, 1.5, 0.2):
    for knee in np.arange(-2.0, 0.17, 0.2):
        for ankle in np.arange(-0.7, 0.7, 0.2):
            angles = dict(zero_angles)
            angles["pelvis_to_HR_thigh"] = hip
            angles["HR_thigh_to_HR_shin"] = knee
            angles["HR_shin_to_HR_foot"] = ankle
            angles["pelvis_to_HL_thigh"] = hip
            angles["HL_thigh_to_HL_shin"] = knee
            angles["HL_shin_to_HL_foot"] = ankle
            
            pos = fk(angles)
            hr_z = pos["HR_foot"][2]
            pelvis_z = pos["pelvis"][2]
            
            if hr_z < pelvis_z:
                leg_reach = pelvis_z - hr_z
                if leg_reach > best_hind_score:
                    best_hind_score = leg_reach
                    best_hind = {"hip": hip, "knee": knee, "ankle": ankle, "foot_z": hr_z}

if best_hind:
    print(f"  Best hind reach: {best_hind_score:.3f}m below pelvis")
    print(f"  hip={best_hind['hip']:.2f}, knee={best_hind['knee']:.2f}, ankle={best_hind['ankle']:.2f}")
    print(f"  foot_z={best_hind['foot_z']:.3f}")
else:
    print("  No hind leg config puts feet below pelvis!")

# Search front leg angles
print("\n--- Searching front leg angles (FR) ---")
best_front_score = -999
best_front = {}

for hip in np.arange(-1.5, 1.5, 0.2):
    for knee in np.arange(-2.0, 0.17, 0.2):
        for ankle in np.arange(-0.7, 0.7, 0.2):
            angles = dict(zero_angles)
            # Need chest position from pelvis_to_chest angle
            for chest_angle in [0.0, -0.5, 0.5]:
                angles["pelvis_to_chest"] = chest_angle
                angles["chest_to_FR_thigh"] = hip
                angles["FR_thigh_to_FR_shin"] = knee
                angles["FR_shin_to_FR_foot"] = ankle
                angles["chest_to_FL_thigh"] = hip
                angles["FL_thigh_to_FL_shin"] = knee
                angles["FL_shin_to_FL_foot"] = ankle
                
                pos = fk(angles)
                fr_z = pos["FR_foot"][2]
                pelvis_z = pos["pelvis"][2]
                
                if fr_z < pelvis_z:
                    leg_reach = pelvis_z - fr_z
                    if leg_reach > best_front_score:
                        best_front_score = leg_reach
                        best_front = {"chest": chest_angle, "hip": hip, "knee": knee, "ankle": ankle, "foot_z": fr_z}

if best_front:
    print(f"  Best front reach: {best_front_score:.3f}m below pelvis")
    print(f"  chest={best_front['chest']:.2f}, hip={best_front['hip']:.2f}, knee={best_front['knee']:.2f}, ankle={best_front['ankle']:.2f}")
    print(f"  foot_z={best_front['foot_z']:.3f}")
else:
    print("  No front leg config puts feet below pelvis!")

# Combine best angles
if best_hind and best_front:
    print("\n--- Combined standing pose ---")
    combined = dict(zero_angles)
    combined["pelvis_to_chest"] = best_front["chest"]
    combined["pelvis_to_HR_thigh"] = best_hind["hip"]
    combined["HR_thigh_to_HR_shin"] = best_hind["knee"]
    combined["HR_shin_to_HR_foot"] = best_hind["ankle"]
    combined["pelvis_to_HL_thigh"] = best_hind["hip"]
    combined["HL_thigh_to_HL_shin"] = best_hind["knee"]
    combined["HL_shin_to_HL_foot"] = best_hind["ankle"]
    combined["chest_to_FR_thigh"] = best_front["hip"]
    combined["FR_thigh_to_FR_shin"] = best_front["knee"]
    combined["FR_shin_to_FR_foot"] = best_front["ankle"]
    combined["chest_to_FL_thigh"] = best_front["hip"]
    combined["FL_thigh_to_FL_shin"] = best_front["knee"]
    combined["FL_shin_to_FL_foot"] = best_front["ankle"]
    
    s, info = score_standing(combined)
    print(f"Score: {s:.3f}")
    print(f"  pelvis_z={info['pelvis_z']:.3f}, chest_z={info['chest_z']:.3f}")
    for f, z in info['foot_zs'].items():
        print(f"  {f}: z={z:.3f}")
    print(f"  pelvis_above_feet={info['pelvis_above_feet']:.3f}, spread={info['foot_spread']:.3f}")
    
    print("\n--- Default joint pos dict for deer_usd_articulation.py ---")
    print("_DEFAULT_JOINT_POS = {")
    for jname, angle in combined.items():
        if abs(angle) > 0.001:
            print(f'    "{jname}": {angle:.4f},')
    print("}")
    
    spawn_z = -min(info['foot_zs'].values()) + 0.02
    print(f"\nRecommended spawn Z: {spawn_z:.3f}")
