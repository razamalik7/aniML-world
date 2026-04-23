"""Compute joint angles that produce a standing pose from the deer URDF.

Reads the URDF, builds the kinematic chains, and uses optimization to find
joint angles where all four feet are at the same height (below the pelvis)
and the skull is above the pelvis.

Run with: python tools/compute_standing_joints.py
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "deer.urdf"


def parse_urdf(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz").split()])
        axis_el = j.find("axis")
        axis = np.array([float(v) for v in axis_el.get("xyz").split()])
        limit = j.find("limit")
        lo = float(limit.get("lower"))
        hi = float(limit.get("upper"))
        joints[name] = {
            "parent": parent, "child": child,
            "xyz": xyz, "axis": axis,
            "lo": lo, "hi": hi,
        }
    return joints


def rot_axis_angle(axis, angle):
    """Rodrigues rotation matrix."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def forward_kinematics(joints, chain_names, angles, base_rot=np.eye(3)):
    """Compute world position of the last link in a chain."""
    pos = np.zeros(3)
    rot = base_rot.copy()
    for i, jname in enumerate(chain_names):
        j = joints[jname]
        pos = pos + rot @ j["xyz"]
        if i < len(angles):
            rot = rot @ rot_axis_angle(j["axis"], angles[i])
    return pos


HIND_RIGHT = ["pelvis_to_HR_thigh", "HR_thigh_to_HR_shin", "HR_shin_to_HR_foot"]
HIND_LEFT = ["pelvis_to_HL_thigh", "HL_thigh_to_HL_shin", "HL_shin_to_HL_foot"]
FRONT_RIGHT = ["pelvis_to_chest", "chest_to_FR_thigh", "FR_thigh_to_FR_shin", "FR_shin_to_FR_foot"]
FRONT_LEFT = ["pelvis_to_chest", "chest_to_FL_thigh", "FL_thigh_to_FL_shin", "FL_shin_to_FL_foot"]

CHAINS = {
    "HR": HIND_RIGHT,
    "HL": HIND_LEFT,
    "FR": FRONT_RIGHT,
    "FL": FRONT_LEFT,
}

SKULL_CHAIN = ["pelvis_to_chest", "chest_to_neck", "neck_to_skull"]


def get_joint_limits(joints, chain):
    return [(joints[j]["lo"], joints[j]["hi"]) for j in chain]


def main():
    joints = parse_urdf(str(URDF_PATH))

    print("=== Foot positions at ZERO joint angles (URDF rest pose) ===")
    for name, chain in CHAINS.items():
        n = len(chain)
        pos = forward_kinematics(joints, chain, [0.0] * n)
        print(f"  {name}_foot: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")

    skull_pos = forward_kinematics(joints, SKULL_CHAIN, [0.0] * 3)
    print(f"  skull:   x={skull_pos[0]:.3f}  y={skull_pos[1]:.3f}  z={skull_pos[2]:.3f}")

    # Try several base X-rotations and optimize joint angles for standing
    best_cost = float("inf")
    best_result = None

    test_angles = list(range(0, 360, 5)) + [269.2, 160.0, 180.0]
    for base_deg in test_angles:
        base_rad = math.radians(base_deg)
        base_rot = rot_axis_angle(np.array([1, 0, 0]), base_rad)

        # Optimization variables: joint angles for all 4 legs
        # HR: 3 joints, HL: 3, FR: 4 (includes pelvis_to_chest), FL: 4 (shares chest joint)
        # But chest joint is shared between FR and FL. Also spine joints affect skull.
        # Simplified: optimize HR(3) + HL(3) + chest(1) + FR_leg(3) + FL_leg(3) + neck(1) + skull(1) = 15 joints
        # Actually, let's just optimize all individual chain angles

        # For simplicity, treat all chains as independent (chest angle duplicated)
        # HR: 3, HL: 3, FR: 4 (chest+3leg), FL: 4 (chest+3leg), skull: 3 (chest+neck+skull)

        # Total: HR(3) + HL(3) + spine_chest(1) + FR_leg(3) + FL_leg(3) + neck(1) + skull_joint(1) = 15
        # But spine_chest is shared by FR, FL, and skull chains.

        # Let's use: [chest_angle, neck_angle, skull_angle,
        #             HR_hip, HR_knee, HR_ankle,
        #             HL_hip, HL_knee, HL_ankle,
        #             FR_hip, FR_knee, FR_ankle,
        #             FL_hip, FL_knee, FL_ankle]

        def objective(x):
            chest_a, neck_a, skull_a = x[0], x[1], x[2]
            hr_a = x[3:6]
            hl_a = x[6:9]
            fr_a = x[9:12]
            fl_a = x[12:15]

            hr_pos = forward_kinematics(joints, HIND_RIGHT, hr_a, base_rot)
            hl_pos = forward_kinematics(joints, HIND_LEFT, hl_a, base_rot)
            fr_pos = forward_kinematics(joints, FRONT_RIGHT, [chest_a] + list(fr_a), base_rot)
            fl_pos = forward_kinematics(joints, FRONT_LEFT, [chest_a] + list(fl_a), base_rot)
            skull_p = forward_kinematics(joints, SKULL_CHAIN, [chest_a, neck_a, skull_a], base_rot)

            foot_zs = np.array([hr_pos[2], hl_pos[2], fr_pos[2], fl_pos[2]])
            mean_foot_z = foot_zs.mean()

            # Goals:
            # 1. All feet at same Z (minimize variance)
            z_var = np.var(foot_zs) * 1000

            # 2. Feet below pelvis (pelvis Z = 0 in body frame, rotated = base_rot @ [0,0,0] = 0)
            pelvis_z = 0.0
            feet_above_pelvis = sum(max(0, fz - pelvis_z) ** 2 for fz in foot_zs) * 100

            # 3. Skull above pelvis
            skull_below = max(0, pelvis_z - skull_p[2]) ** 2 * 50

            # 4. Feet low (want mean foot Z to be as low as possible)
            feet_high = mean_foot_z ** 2 * 10

            # 5. Joint angles small (prefer natural pose)
            joint_penalty = np.sum(x ** 2) * 0.1

            # 6. Pelvis above feet by at least 0.3m
            gap = pelvis_z - mean_foot_z
            gap_penalty = max(0, 0.3 - gap) ** 2 * 200

            return z_var + feet_above_pelvis + skull_below + feet_high + joint_penalty + gap_penalty

        limits = []
        # chest
        limits.append((joints["pelvis_to_chest"]["lo"], joints["pelvis_to_chest"]["hi"]))
        # neck
        limits.append((joints["chest_to_neck"]["lo"], joints["chest_to_neck"]["hi"]))
        # skull
        limits.append((joints["neck_to_skull"]["lo"], joints["neck_to_skull"]["hi"]))
        # HR
        for j in HIND_RIGHT:
            limits.append((joints[j]["lo"], joints[j]["hi"]))
        # HL
        for j in HIND_LEFT:
            limits.append((joints[j]["lo"], joints[j]["hi"]))
        # FR (leg only, chest already included)
        for j in FRONT_RIGHT[1:]:
            limits.append((joints[j]["lo"], joints[j]["hi"]))
        # FL (leg only)
        for j in FRONT_LEFT[1:]:
            limits.append((joints[j]["lo"], joints[j]["hi"]))

        x0 = np.zeros(15)
        res = minimize(objective, x0, method="L-BFGS-B", bounds=limits,
                       options={"maxiter": 500, "ftol": 1e-12})

        if res.fun < best_cost:
            best_cost = res.fun
            best_result = (base_deg, res.x.copy(), res.fun)

    base_deg, x, cost = best_result
    base_rad = math.radians(base_deg)
    base_rot = rot_axis_angle(np.array([1, 0, 0]), base_rad)

    print(f"\n=== BEST STANDING CONFIGURATION ===")
    print(f"Base X rotation: {base_deg}° (rad: {base_rad:.4f})")

    # Convert to quaternion
    half = base_rad / 2
    qw = math.cos(half)
    qx = math.sin(half)
    print(f"Quaternion (w,x,y,z): ({qw:.6f}, {qx:.6f}, 0, 0)")
    print(f"Optimization cost: {cost:.6f}")

    chest_a, neck_a, skull_a = x[0], x[1], x[2]
    hr_a = x[3:6]
    hl_a = x[6:9]
    fr_a = x[9:12]
    fl_a = x[12:15]

    print(f"\nJoint angles (radians):")
    print(f"  pelvis_to_chest:     {chest_a:.4f} ({math.degrees(chest_a):.1f}°)")
    print(f"  chest_to_neck:       {neck_a:.4f} ({math.degrees(neck_a):.1f}°)")
    print(f"  neck_to_skull:       {skull_a:.4f} ({math.degrees(skull_a):.1f}°)")

    for name, angles, chain in [("HR", hr_a, HIND_RIGHT), ("HL", hl_a, HIND_LEFT)]:
        for jname, a in zip(chain, angles):
            print(f"  {jname}: {a:.4f} ({math.degrees(a):.1f}°)")

    for name, angles, chain_tail in [("FR", fr_a, FRONT_RIGHT[1:]), ("FL", fl_a, FRONT_LEFT[1:])]:
        for jname, a in zip(chain_tail, angles):
            print(f"  {jname}: {a:.4f} ({math.degrees(a):.1f}°)")

    # Print final positions
    print(f"\nBody positions after optimization:")
    hr_pos = forward_kinematics(joints, HIND_RIGHT, hr_a, base_rot)
    hl_pos = forward_kinematics(joints, HIND_LEFT, hl_a, base_rot)
    fr_pos = forward_kinematics(joints, FRONT_RIGHT, [chest_a] + list(fr_a), base_rot)
    fl_pos = forward_kinematics(joints, FRONT_LEFT, [chest_a] + list(fl_a), base_rot)
    skull_p = forward_kinematics(joints, SKULL_CHAIN, [chest_a, neck_a, skull_a], base_rot)

    print(f"  pelvis:  z=0.000 (origin)")
    for name, pos in [("HR_foot", hr_pos), ("HL_foot", hl_pos), ("FR_foot", fr_pos), ("FL_foot", fl_pos), ("skull", skull_p)]:
        print(f"  {name}: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")

    foot_zs = [hr_pos[2], hl_pos[2], fr_pos[2], fl_pos[2]]
    print(f"\n  Foot Z range: {min(foot_zs):.3f} to {max(foot_zs):.3f} (spread: {max(foot_zs)-min(foot_zs):.3f})")
    print(f"  Mean foot Z: {np.mean(foot_zs):.3f}")
    print(f"  Pelvis above feet: {0 - np.mean(foot_zs):.3f}m")
    print(f"  Skull above pelvis: {skull_p[2]:.3f}m")
    print(f"\n  Recommended ANIML_DEER_SPAWN_Z: {-np.mean(foot_zs) + 0.05:.3f}")

    # Output in format for deer_usd_articulation.py
    print(f"\n=== CONFIG FOR deer_usd_articulation.py ===")
    print(f'_DEFAULT_ROT = "{qw:.6f},{qx:.6f},0,0"')
    joint_dict = {}
    joint_dict["pelvis_to_chest"] = round(chest_a, 4)
    joint_dict["chest_to_neck"] = round(neck_a, 4)
    joint_dict["neck_to_skull"] = round(skull_a, 4)
    for jname, a in zip(HIND_RIGHT, hr_a):
        joint_dict[jname] = round(a, 4)
    for jname, a in zip(HIND_LEFT, hl_a):
        joint_dict[jname] = round(a, 4)
    for jname, a in zip(FRONT_RIGHT[1:], fr_a):
        joint_dict[jname] = round(a, 4)
    for jname, a in zip(FRONT_LEFT[1:], fl_a):
        joint_dict[jname] = round(a, 4)
    # Remove zero entries
    joint_dict = {k: v for k, v in joint_dict.items() if abs(v) > 0.001}
    print(f"_DEFAULT_JOINT_POS = {joint_dict}")


if __name__ == "__main__":
    main()
