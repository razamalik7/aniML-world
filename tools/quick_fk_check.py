"""Quick FK check: compute foot/skull positions at various base rotations.

Pure Python (no Isaac Lab needed). Uses deer.urdf joint origins directly.
"""
import math
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET


def rot_x(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def parse_urdf(path):
    tree = ET.parse(str(path))
    root = tree.getroot()
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        origin = j.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz").split()])
        axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
        joints[name] = {"xyz": xyz, "axis": axis,
                        "parent": j.find("parent").get("link"),
                        "child": j.find("child").get("link")}
    return joints


def fk_chain(joints, chain_names, base_rot):
    pos = np.zeros(3)
    rot = base_rot.copy()
    for jname in chain_names:
        pos = pos + rot @ joints[jname]["xyz"]
    return pos


CHAINS = {
    "HL_foot": ["pelvis_to_HL_thigh", "HL_thigh_to_HL_shin", "HL_shin_to_HL_foot"],
    "HR_foot": ["pelvis_to_HR_thigh", "HR_thigh_to_HR_shin", "HR_shin_to_HR_foot"],
    "FL_foot": ["pelvis_to_chest", "chest_to_FL_thigh", "FL_thigh_to_FL_shin", "FL_shin_to_FL_foot"],
    "FR_foot": ["pelvis_to_chest", "chest_to_FR_thigh", "FR_thigh_to_FR_shin", "FR_shin_to_FR_foot"],
    "skull": ["pelvis_to_chest", "chest_to_neck", "neck_to_skull"],
}


def main():
    urdf = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "deer.urdf"
    joints = parse_urdf(urdf)

    print("URDF frame (no rotation) — zero joint angles:")
    R0 = np.eye(3)
    for name, chain in CHAINS.items():
        p = fk_chain(joints, chain, R0)
        print(f"  {name:10s}  X={p[0]:+7.3f}  Y={p[1]:+7.3f}  Z={p[2]:+7.3f}")

    print("\n" + "=" * 70)
    print("Testing various X-axis rotations (zero joint angles)")
    print("Looking for: feet Z < pelvis Z (0), skull Z > pelvis Z (0)")
    print("=" * 70)

    for deg in [0, 90, -90, 180, 160, -91, 269]:
        R = rot_x(deg)
        foot_zs = []
        skull_z = 0
        for name, chain in CHAINS.items():
            p = fk_chain(joints, chain, R)
            if "foot" in name:
                foot_zs.append(p[2])
            if name == "skull":
                skull_z = p[2]

        mean_fz = np.mean(foot_zs)
        standing = mean_fz < 0 and skull_z > 0
        print(f"\n  rot_x({deg:+4d}°):")
        for name, chain in CHAINS.items():
            p = fk_chain(joints, chain, R)
            tag = ""
            if "foot" in name:
                tag = " [FOOT]"
            if name == "skull":
                tag = " [SKULL]"
            print(f"    {name:10s}  X={p[0]:+7.3f}  Y={p[1]:+7.3f}  Z={p[2]:+7.3f}{tag}")
        print(f"    Mean foot Z: {mean_fz:+.3f}  |  Skull Z: {skull_z:+.3f}  "
              f"|  Pelvis-feet: {0-mean_fz:+.3f}  |  "
              f"{'** STANDING **' if standing else 'NOT standing'}")
        if standing:
            print(f"    Recommended spawn_z: {-mean_fz + 0.05:.3f}")


if __name__ == "__main__":
    main()
