"""Create a URDF that uses rotated STLs for visuals, original STLs for collisions.

Reads the original deer.urdf and replaces each visual mesh reference with the
corresponding rotated STL from meshes/visual_rotated/.

Usage:
    python tools/create_visualfix_urdf.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_IN = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "deer.urdf"
URDF_OUT = Path(__file__).resolve().parents[1] / "assets" / "deer_urdf" / "deer_visualfix.urdf"
ROTATED_DIR = "meshes/visual_rotated"


def main():
    tree = ET.parse(str(URDF_IN))
    root = tree.getroot()

    count = 0
    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is None:
            continue

        old_file = mesh.get("filename", "")
        stem = Path(old_file).stem  # e.g. "pelvis_collision"
        rotated_name = stem.replace("_collision", "_visual_rotated") + ".stl"
        rotated_path = f"{ROTATED_DIR}/{rotated_name}"

        rotated_full = URDF_IN.parent / rotated_path
        if rotated_full.exists():
            mesh.set("filename", rotated_path)
            count += 1
            print(f"  {link.get('name')}: {old_file} -> {rotated_path}")
        else:
            print(f"  {link.get('name')}: SKIP (no rotated STL at {rotated_full})")

    ET.indent(tree, space="  ")
    tree.write(str(URDF_OUT), xml_declaration=True, encoding="unicode")
    print(f"\nWrote: {URDF_OUT}")
    print(f"Updated {count} visual mesh references")
    print(f"Collision meshes unchanged (original STLs)")


if __name__ == "__main__":
    main()
