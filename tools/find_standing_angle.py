"""Find the Rx(theta) angle that makes the deer stand upright.

Uses actual USD link positions (zero-joint rest pose).
"""
import math

LINKS = {
    "pelvis":   (0, 0, 0),
    "chest":    (0, -0.965, 0.512),
    "neck":     (0, -0.989, -0.338),
    "skull":    (0, -1.358, -0.688),
    "HR_foot":  (-0.135, -0.435, 0.442),
    "HL_foot":  (0.135, -0.435, 0.442),
    "FR_foot":  (-0.121, -0.367, -0.298),
    "FL_foot":  (0.122, -0.367, -0.298),
}

def rx(y, z, theta_deg):
    t = math.radians(theta_deg)
    return y * math.sin(t) + z * math.cos(t)

print(f"{'angle':>6}  {'pelvis_z':>9}  {'chest_z':>9}  {'skull_z':>9}  "
      f"{'HR_ft_z':>9}  {'FR_ft_z':>9}  {'spine_tilt':>11}  {'notes'}")
print("-" * 100)

for angle in range(0, 361, 5):
    pz = rx(0, 0, angle)
    cz = rx(-0.965, 0.512, angle)
    sz = rx(-1.358, -0.688, angle)
    hz = rx(-0.435, 0.442, angle)
    fz = rx(-0.367, -0.298, angle)

    spine_tilt = cz - pz

    notes = []
    if hz < pz and fz < pz:
        notes.append("ALL_FEET_BELOW")
    if abs(spine_tilt) < 0.15:
        notes.append("SPINE_HORIZ")
    if sz > pz + 0.1:
        notes.append("HEAD_UP")
    if hz < pz - 0.2 and fz < pz - 0.2:
        notes.append("LEGS_DOWN")

    note_str = " ".join(notes)
    if note_str:
        print(f"{angle:>6}  {pz:>+9.3f}  {cz:>+9.3f}  {sz:>+9.3f}  "
              f"{hz:>+9.3f}  {fz:>+9.3f}  {spine_tilt:>+11.3f}  {note_str}")

print("\n--- Key angles ---")
for angle in [85, 90, 95, 100, 110, 120, 130, 140, 150, 153, 160]:
    pz = rx(0, 0, angle)
    cz = rx(-0.965, 0.512, angle)
    sz = rx(-1.358, -0.688, angle)
    hz = rx(-0.435, 0.442, angle)
    fz = rx(-0.367, -0.298, angle)
    spine_tilt = cz - pz
    print(f"  Rx({angle:>3}): pelvis_z={pz:+.3f}  chest_z={cz:+.3f}  skull_z={sz:+.3f}  "
          f"HR_foot_z={hz:+.3f}  FR_foot_z={fz:+.3f}  spine_tilt={spine_tilt:+.3f}")
