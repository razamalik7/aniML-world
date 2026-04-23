"""
Patch deer_world.urdf joint limits based on real deer anatomy.

Key anatomical facts:
- Front knees bend FORWARD only (like human knees but reversed) — no backward hyperextension
- Hind knees bend BACKWARD only (like a dog's hock joint)
- Thigh swing is moderate — deer don't do full splits
- Spine has very limited mobility
"""
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_PATH = Path(__file__).parent.parent / "assets" / "deer_urdf" / "deer_world.urdf"

# (lower_rad, upper_rad) — anatomically grounded limits
# Positive = forward/up in Isaac frame after Rx(-90) spawn
JOINT_LIMITS = {
    # Spine — very stiff in real deer
    "pelvis_to_chest":      (-0.2,  0.2),
    "chest_to_neck":        (-0.3,  0.5),   # neck bends down more than up
    "neck_to_skull":        (-0.4,  0.6),   # head tilts down to graze

    # Hind legs — knee bends backward (negative), hip swings forward/back
    "pelvis_to_HR_thigh":   (-0.8,  0.8),
    "HR_thigh_to_HR_shin":  (-1.8,  0.0),   # hind knee: backward only, no forward hyperextension
    "HR_shin_to_HR_foot":   (-0.5,  0.5),

    "pelvis_to_HL_thigh":   (-0.8,  0.8),
    "HL_thigh_to_HL_shin":  (-1.8,  0.0),   # hind knee: backward only
    "HL_shin_to_HL_foot":   (-0.5,  0.5),

    # Front legs — knee bends FORWARD only (opposite to hind), no backward hyperextension
    "chest_to_FR_thigh":    (-0.8,  0.8),
    "FR_thigh_to_FR_shin":  ( 0.0,  1.8),   # front knee: forward only (positive)
    "FR_shin_to_FR_foot":   (-0.5,  0.5),

    "chest_to_FL_thigh":    (-0.8,  0.8),
    "FL_thigh_to_FL_shin":  ( 0.0,  1.8),   # front knee: forward only (positive)
    "FL_shin_to_FL_foot":   (-0.5,  0.5),
}

tree = ET.parse(URDF_PATH)
root = tree.getroot()

patched = []
for joint in root.findall("joint"):
    name = joint.get("name")
    if name not in JOINT_LIMITS:
        continue
    lim = joint.find("limit")
    if lim is None:
        continue
    old_lo = float(lim.get("lower", 0))
    old_hi = float(lim.get("upper", 0))
    new_lo, new_hi = JOINT_LIMITS[name]
    lim.set("lower", str(new_lo))
    lim.set("upper", str(new_hi))
    patched.append((name, old_lo, old_hi, new_lo, new_hi))

tree.write(URDF_PATH, xml_declaration=True, encoding="unicode")

print("Patched joint limits:")
for name, olo, ohi, nlo, nhi in patched:
    changed = " *** CHANGED ***" if (olo != nlo or ohi != nhi) else ""
    print(f"  {name:30s}: [{olo:.1f}, {ohi:.1f}] -> [{nlo:.1f}, {nhi:.1f}]{changed}")
print(f"\nSaved to: {URDF_PATH}")
print("\nNEXT: Re-import URDF to USD: isaaclab.bat -p tools/import_world_urdf_v2.py")
