"""Pre-rotate a URDF so that a standing pose uses identity base rotation.

The Blender export produces mesh STLs and bone offsets in DIFFERENT coordinate
frames (~70 deg offset about X). So we apply two separate X-axis rotations:

  Kinematic rotation (160 deg):
    - Joint origin XYZ
    - Inertial origin XYZ
  Visual rotation (90 deg):
    - RPY on collision/visual mesh origins

Joint axes (1,0,0) are invariant under X rotation — no change needed.

After this, identity base rotation + IK joint angles = standing deer whose
mesh visuals match the physical skeleton.

Usage:
    python tools/prerotate_urdf.py
"""

import math
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF_IN = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "deer.urdf"
URDF_OUT = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "deer_rotated.urdf"

KINEMATIC_DEG = 160.0
KINEMATIC_RAD = math.radians(KINEMATIC_DEG)

# Mesh data from Blender is ~70° offset from bone data (armature vs mesh object
# transforms differ). The meshes need 90° X rotation to appear standing, while
# the joint offsets need 160° for correct physics.
VISUAL_DEG = -90.0
VISUAL_RAD = math.radians(VISUAL_DEG)


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


R_KIN = rot_x(KINEMATIC_RAD)
R_VIS_RPY = f"{VISUAL_RAD:.6f} 0 0"


def rotate_xyz_kin(xyz_str: str) -> str:
    vals = np.array([float(v) for v in xyz_str.split()])
    rotated = R_KIN @ vals
    return f"{rotated[0]:.6f} {rotated[1]:.6f} {rotated[2]:.6f}"


def main():
    tree = ET.parse(str(URDF_IN))
    root = tree.getroot()

    for link in root.findall("link"):
        name = link.get("name")

        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            if origin is not None:
                old_xyz = origin.get("xyz", "0 0 0")
                origin.set("xyz", rotate_xyz_kin(old_xyz))

        for tag in ("collision", "visual"):
            elem = link.find(tag)
            if elem is not None:
                origin = elem.find("origin")
                if origin is not None:
                    old_xyz = origin.get("xyz", "0 0 0")
                    origin.set("xyz", rotate_xyz_kin(old_xyz))
                    origin.set("rpy", R_VIS_RPY)

    for joint in root.findall("joint"):
        origin = joint.find("origin")
        if origin is not None:
            old_xyz = origin.get("xyz", "0 0 0")
            origin.set("xyz", rotate_xyz_kin(old_xyz))

    ET.indent(tree, space="  ")
    tree.write(str(URDF_OUT), xml_declaration=True, encoding="unicode")
    print(f"Wrote pre-rotated URDF to: {URDF_OUT}")
    print(f"Kinematic rotation baked: {KINEMATIC_DEG} deg about X (joints/inertials)")
    print(f"Visual mesh RPY: {VISUAL_DEG} deg about X")
    print(f"\nNow import this URDF in Isaac Sim (File > Import > URDF)")
    print(f"Then use identity rotation (1,0,0,0) in deer_usd_articulation.py")


if __name__ == "__main__":
    main()
