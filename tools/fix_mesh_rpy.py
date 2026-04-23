"""Add mesh RPY to URDF visual/collision origins WITHOUT changing kinematics.

The mesh STL data from Blender uses a different coordinate frame than the bone
(joint) data. When Isaac Lab renders the physics at 160 deg X base rotation, the
skeleton is correct but the mesh appears rotated within each link frame.

This script adds RPY to each visual/collision mesh origin so the mesh aligns
with the physics skeleton. Joint origins and inertial origins are left unchanged.

Usage:
    python tools/fix_mesh_rpy.py [--rpy ROLL_DEG PITCH_DEG YAW_DEG]
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_IN = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "deer.urdf"
URDF_OUT = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "deer_meshfix.urdf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpy", nargs=3, type=float, default=[-90, 0, 0],
                        help="Mesh RPY in degrees (roll pitch yaw about X Y Z)")
    args = parser.parse_args()

    rpy_rad = [math.radians(d) for d in args.rpy]
    rpy_str = f"{rpy_rad[0]:.6f} {rpy_rad[1]:.6f} {rpy_rad[2]:.6f}"

    tree = ET.parse(str(URDF_IN))
    root = tree.getroot()

    count = 0
    for link in root.findall("link"):
        elem = link.find("visual")
        if elem is not None:
            origin = elem.find("origin")
            if origin is None:
                origin = ET.SubElement(elem, "origin")
                origin.set("xyz", "0 0 0")
            origin.set("rpy", rpy_str)
            count += 1

    ET.indent(tree, space="  ")
    tree.write(str(URDF_OUT), xml_declaration=True, encoding="unicode")
    print(f"Wrote: {URDF_OUT}")
    print(f"Modified {count} visual-only origins with RPY = {args.rpy} deg")
    print(f"Kinematics (joint origins, inertials) unchanged")


if __name__ == "__main__":
    main()
