"""
Patch deer_world.urdf with anatomically realistic masses.

Real deer (~70 kg doe) CoM facts:
- ~60% of body weight on front legs (chest/shoulder region)
- Head + neck are far forward and heavy, pulling CoM forward
- Current bug: pelvis=25kg, chest=22kg -> CoM sits too far back -> tips backward

References: equine/cervid biomechanics literature; front:rear weight ratio ~60:40.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_PATH = Path(__file__).parent.parent / "assets" / "deer_urdf" / "deer_world.urdf"

# Anatomically realistic masses for a ~70 kg adult doe.
# Chest carries ribcage + front shoulder mass. Skull+neck hang far forward.
# Front legs heavier because they bear 60% of static load.
REALISTIC_MASSES = {
    "pelvis":   18.0,   # hindquarters, sacrum
    "chest":    28.0,   # ribcage, withers, front shoulders — largest single block
    "neck":      6.0,   # muscular cervid neck
    "skull":     5.0,   # head + antler base
    "HR_thigh":  1.2,   # hind legs are lighter (spring-loaded propulsion)
    "HR_shin":   1.0,
    "HR_foot":   0.8,
    "HL_thigh":  1.2,
    "HL_shin":   1.0,
    "HL_foot":   0.8,
    "FR_thigh":  2.5,   # front legs heavier (weight-bearing)
    "FR_shin":   1.5,
    "FR_foot":   1.0,
    "FL_thigh":  2.5,
    "FL_shin":   1.5,
    "FL_foot":   1.0,
}
# Total: 18+28+6+5 + 2*(1.2+1.0+0.8) + 2*(2.5+1.5+1.0) = 57 + 6 + 10 = 73 kg

ET.register_namespace("", "")
tree = ET.parse(URDF_PATH)
root = tree.getroot()

patched = []
for link in root.findall("link"):
    name = link.get("name")
    if name not in REALISTIC_MASSES:
        continue
    inertial = link.find("inertial")
    if inertial is None:
        continue
    mass_elem = inertial.find("mass")
    if mass_elem is None:
        continue
    old = float(mass_elem.get("value"))
    new = REALISTIC_MASSES[name]
    mass_elem.set("value", str(new))

    # Also update inertia tensor to match (simple sphere approx I = 0.4*m*r^2)
    # Use radius proportional to cube root of mass for rough scaling
    r = 0.12 + 0.04 * (new / 25.0) ** (1/3)
    ii = round(0.4 * new * r * r, 5)
    inertia_elem = inertial.find("inertia")
    if inertia_elem is not None:
        inertia_elem.set("ixx", str(ii))
        inertia_elem.set("iyy", str(ii))
        inertia_elem.set("izz", str(ii))
        inertia_elem.set("ixy", "0")
        inertia_elem.set("ixz", "0")
        inertia_elem.set("iyz", "0")

    patched.append((name, old, new))

tree.write(URDF_PATH, xml_declaration=True, encoding="unicode")

print("Patched URDF masses:")
total_old = total_new = 0
for name, old, new in patched:
    print(f"  {name:15s}: {old:.1f} kg -> {new:.1f} kg  ({new-old:+.1f})")
    total_old += old
    total_new += new
print(f"\n  Total: {total_old:.1f} kg -> {total_new:.1f} kg")
print(f"\nSaved to: {URDF_PATH}")
print("\nNEXT: Re-import URDF to USD so physics picks up the new masses.")
print("Run: python tools/import_world_urdf_v2.py")
