"""Compute initial joint angles for a LEVEL standing deer.

Keeps identity root rotation (body upright → flat_orientation_l2 ≈ 0)
and adjusts leg joint angles so all four feet are at the same Z level.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize


def parse_joints(path: str):
    tree = ET.parse(path)
    joints = {}
    for j in tree.findall("joint"):
        if j.get("type") != "revolute":
            continue
        origin = j.find("origin")
        xyz = np.array([float(x) for x in origin.get("xyz").split()])
        axis = np.array([float(x) for x in j.find("axis").get("xyz").split()])
        lim = j.find("limit")
        joints[j.get("name")] = {
            "xyz": xyz, "axis": axis,
            "lo": float(lim.get("lower")), "hi": float(lim.get("upper")),
        }
    return joints


def rot_axis(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def fk_chain(joints, chain, angles, base_pos):
    pos = base_pos.copy()
    R = np.eye(3)
    for jname in chain:
        j = joints[jname]
        pos = pos + R @ j["xyz"]
        R = R @ rot_axis(j["axis"], angles.get(jname, 0.0))
    return pos


def main():
    urdf = r"c:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer.urdf"
    J = parse_joints(urdf)

    print("=== Zero-angle foot positions (identity root) ===")
    legs = {
        "HR": ["pelvis_to_HR_thigh", "HR_thigh_to_HR_shin", "HR_shin_to_HR_foot"],
        "HL": ["pelvis_to_HL_thigh", "HL_thigh_to_HL_shin", "HL_shin_to_HL_foot"],
        "FR": ["pelvis_to_chest", "chest_to_FR_thigh", "FR_thigh_to_FR_shin", "FR_shin_to_FR_foot"],
        "FL": ["pelvis_to_chest", "chest_to_FL_thigh", "FL_thigh_to_FL_shin", "FL_shin_to_FL_foot"],
    }
    for name, chain in legs.items():
        fp = fk_chain(J, chain, {}, np.zeros(3))
        print(f"  {name}: x={fp[0]:.3f} y={fp[1]:.3f} z={fp[2]:.3f}")

    # x = [hind_thigh, hind_shin, hind_foot, front_thigh, front_shin, front_foot, spine]
    # Left/right symmetric
    def unpack(x):
        return {
            "pelvis_to_HR_thigh": x[0], "pelvis_to_HL_thigh": x[0],
            "HR_thigh_to_HR_shin": x[1], "HL_thigh_to_HL_shin": x[1],
            "HR_shin_to_HR_foot": x[2], "HL_shin_to_HL_foot": x[2],
            "chest_to_FR_thigh": x[3], "chest_to_FL_thigh": x[3],
            "FR_thigh_to_FR_shin": x[4], "FL_thigh_to_FL_shin": x[4],
            "FR_shin_to_FR_foot": x[5], "FL_shin_to_FL_foot": x[5],
            "pelvis_to_chest": x[6],
        }

    bounds = [
        (J["pelvis_to_HR_thigh"]["lo"], J["pelvis_to_HR_thigh"]["hi"]),
        (J["HR_thigh_to_HR_shin"]["lo"], J["HR_thigh_to_HR_shin"]["hi"]),
        (J["HR_shin_to_HR_foot"]["lo"], J["HR_shin_to_HR_foot"]["hi"]),
        (J["chest_to_FR_thigh"]["lo"], J["chest_to_FR_thigh"]["hi"]),
        (J["FR_thigh_to_FR_shin"]["lo"], J["FR_thigh_to_FR_shin"]["hi"]),
        (J["FR_shin_to_FR_foot"]["lo"], J["FR_shin_to_FR_foot"]["hi"]),
        (J["pelvis_to_chest"]["lo"], J["pelvis_to_chest"]["hi"]),
    ]

    def objective(x):
        a = unpack(x)
        origin = np.zeros(3)

        foot_z = {}
        for leg_name, chain in legs.items():
            foot_z[leg_name] = fk_chain(J, chain, a, origin)[2]

        zs = list(foot_z.values())
        mean_z = np.mean(zs)
        level = sum((z - mean_z) ** 2 for z in zs)
        high = mean_z

        reg = sum(xi ** 2 for xi in x) * 0.5

        limit_pen = 0.0
        for i, xi in enumerate(x):
            lo, hi = bounds[i]
            margin = (hi - lo) * 0.15
            if xi < lo + margin:
                limit_pen += ((lo + margin - xi) / margin) ** 2
            if xi > hi - margin:
                limit_pen += ((xi - (hi - margin)) / margin) ** 2

        return (
            level * 500     # feet level
            + high * 100    # feet as low as possible
            + reg           # prefer small angles (natural pose)
            + limit_pen * 5
        )

    best_result = None
    best_cost = float("inf")
    rng = np.random.default_rng(42)

    for trial in range(500):
        if trial == 0:
            x0 = np.zeros(7)
        else:
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res

    x = best_result.x
    a = unpack(x)

    print("\n=== Optimized Standing Pose (identity root) ===\n")
    for leg_name, chain in legs.items():
        fp = fk_chain(J, chain, a, np.zeros(3))
        print(f"  {leg_name} foot: x={fp[0]:.3f} y={fp[1]:.3f} z={fp[2]:.3f}")

    foot_zs = [fk_chain(J, chain, a, np.zeros(3))[2] for chain in legs.values()]
    print(f"\n  Foot Z range: {min(foot_zs):.4f} to {max(foot_zs):.4f}")
    print(f"  Foot Z spread: {max(foot_zs) - min(foot_zs):.4f}")
    print(f"  Spawn Z: {abs(min(foot_zs)) + 0.05:.3f}")

    print("\n=== Joint angles ===\n")
    joint_names = [
        ("pelvis_to_chest", a["pelvis_to_chest"]),
        ("pelvis_to_H.*thigh", a["pelvis_to_HR_thigh"]),
        ("H._thigh_to_H._shin", a["HR_thigh_to_HR_shin"]),
        ("H._shin_to_H._foot", a["HR_shin_to_HR_foot"]),
        ("chest_to_F.*thigh", a["chest_to_FR_thigh"]),
        ("F._thigh_to_F._shin", a["FR_thigh_to_FR_shin"]),
        ("F._shin_to_F._foot", a["FR_shin_to_FR_foot"]),
    ]
    print("_DEFAULT_JOINT_POS = {")
    for pat, val in joint_names:
        print(f'    "{pat}": {val:.4f},  # {math.degrees(val):.1f} deg')
    print("}")

    print(f'\n_DEFAULT_ROT = "1,0,0,0"  # identity — body upright')
    print(f"\nSpawn Z = {abs(min(foot_zs)) + 0.05:.3f}")


if __name__ == "__main__":
    main()
