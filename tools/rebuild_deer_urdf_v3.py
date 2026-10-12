"""Rebuild deer URDF with v3 physics: mass redistribution to 82 kg, ellipsoid limb inertias,
L/R symmetric inertials, expanded neck ROM, and antlers as a separate rigid body via fixed joint.

See `docs/physics_audit_2026-05-04.md` for the rationale behind each change.

Run from project root:
    python tools/rebuild_deer_urdf_v3.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
URDF_IN = REPO / "assets" / "deer_urdf" / "deer_world.urdf"
URDF_OUT = REPO / "assets" / "deer_urdf" / "deer_world.urdf"
URDF_BACKUP = REPO / "assets" / "deer_urdf" / "deer_world.pre_v3.urdf"


# ============================================================================
# Mass plan — 82 kg adult white-tail buck.
# Anatomical proportions: torso ~62%, head+neck ~10%, limbs ~26%, antlers 2 kg.
# Front legs slightly heavier than hind (bio: 60:40 weight bearing).
# ============================================================================
MASSES = {
    "pelvis":   10.0,
    "chest":    41.5,   # +0.5 to absorb the antler mass reduction (total stays 82 kg)
    "neck":      5.0,
    "skull":     3.0,   # without antlers (separated into own body)
    "antlers":   1.5,   # v4.1: was 2.0; reduced for less-aggressive forward moment
    # Hind legs (each side identical for symmetry)
    "HR_thigh":  2.5, "HL_thigh":  2.5,
    "HR_shin":   1.7, "HL_shin":   1.7,
    "HR_foot":   0.8, "HL_foot":   0.8,
    # Front legs (slightly heavier — biomech)
    "FR_thigh":  3.0, "FL_thigh":  3.0,
    "FR_shin":   1.7, "FL_shin":   1.7,
    "FR_foot":   0.8, "FL_foot":   0.8,
}

# ============================================================================
# Symmetric CoM offsets (mirror across x=0 between R and L pairs).
# Use the R-side as canonical, force L = mirror(R).
# Values from current URDF, cleaned up to ensure symmetry.
# ============================================================================
COM_OFFSETS = {
    # Body (centered)
    "pelvis":   (-0.000001, 0.041033,  0.005039),
    "chest":    (-0.000001, 0.025167, -0.209932),
    "neck":     (-0.000009,-0.029927, -0.190513),
    "skull":    ( 0.000031,-0.086540, -0.031164),
    # Antlers — v4.1: CoM moved CLOSER to pedicle to match real-deer mass
    # distribution (heaviest at the burr/lower beam, not at tine tips).
    # was (0, -0.20, -0.25) = 20 cm up + 25 cm forward of pedicle. Too aggressive —
    # caused visible head-led falls and chin-tuck compensation behavior.
    # Now: 10 cm up + 10 cm forward. Antler moment about pedicle drops ~3x.
    # In URDF coord (-Z forward, +Y down), forward=+(-Z), up=-Y.
    "antlers":  ( 0.000000,-0.100000, -0.100000),
    # Hind right (canonical), hind left mirrored
    "HR_thigh": ( 0.011367, 0.099510, -0.142357),
    "HR_shin":  ( 0.000192, 0.058269,  0.086781),
    "HR_foot":  ( 0.003029, 0.241795, -0.060840),
    # Front right (canonical), front left mirrored
    "FR_thigh": (-0.015282, 0.086552,  0.100682),
    "FR_shin":  ( 0.014918, 0.137371, -0.067931),
    "FR_foot":  (-0.000020, 0.228783, -0.020400),
}
# Generate L mirrors from R
for r_name, r_off in list(COM_OFFSETS.items()):
    if r_name.startswith(("HR_", "FR_")):
        l_name = r_name.replace("HR_", "HL_").replace("FR_", "FL_")
        COM_OFFSETS[l_name] = (-r_off[0], r_off[1], r_off[2])


# ============================================================================
# Inertia tensors.
# Limb segments: ellipsoid (long axis = link Z, transverse = X & Y).
# Body links: rectangular block approximation.
# ============================================================================
def ellipsoid_inertia(mass: float, length: float, radius: float):
    """Cylinder inertia: long axis = z, transverse = x = y."""
    i_long = 0.5 * mass * radius**2
    i_trans = (1.0 / 12.0) * mass * (3 * radius**2 + length**2)
    return i_trans, i_trans, i_long  # ixx, iyy, izz


def block_inertia(mass: float, w: float, l: float, h: float):
    """Rectangular block inertia."""
    ixx = (1.0 / 12.0) * mass * (l**2 + h**2)
    iyy = (1.0 / 12.0) * mass * (w**2 + h**2)
    izz = (1.0 / 12.0) * mass * (w**2 + l**2)
    return ixx, iyy, izz


# Limb dimensions estimated from Blender bone lengths + reasonable cross-sections.
# (length_m, radius_m)
LIMB_DIMS = {
    "HR_thigh": (0.33, 0.07), "HL_thigh": (0.33, 0.07),
    "HR_shin":  (0.275, 0.05), "HL_shin":  (0.275, 0.05),
    "HR_foot":  (0.34, 0.04), "HL_foot":  (0.34, 0.04),
    "FR_thigh": (0.23, 0.07), "FL_thigh": (0.23, 0.07),
    "FR_shin":  (0.286, 0.05), "FL_shin":  (0.286, 0.05),
    "FR_foot":  (0.276, 0.04), "FL_foot":  (0.276, 0.04),
}

# Body link dimensions (rough, w x l x h in meters).
BODY_DIMS = {
    "pelvis": (0.30, 0.30, 0.20),
    "chest":  (0.40, 0.60, 0.50),
    "neck":   (0.18, 0.40, 0.18),
    "skull":  (0.18, 0.20, 0.15),
    "antlers":(0.50, 0.40, 0.40),  # branching — bbox approx, mass not uniformly distributed
}


def compute_inertia(link_name: str, mass: float):
    if link_name in LIMB_DIMS:
        L, r = LIMB_DIMS[link_name]
        return ellipsoid_inertia(mass, L, r)
    if link_name == "antlers":
        # Antlers are branching/sparse; reduce moments by ~40% from solid block
        ixx, iyy, izz = block_inertia(mass, *BODY_DIMS["antlers"])
        return ixx * 0.6, iyy * 0.6, izz * 0.6
    if link_name in BODY_DIMS:
        return block_inertia(mass, *BODY_DIMS[link_name])
    raise ValueError(f"Unknown link: {link_name}")


# ============================================================================
# Joint limit overrides.
# Expand neck ROM so deer can actually drink, browse, look behind.
# Hock asymmetric range (real hock flexes one direction).
# Spine pitch slightly expanded for pronk/jump prep.
# ============================================================================
JOINT_LIMITS = {
    # Neck — was -0.6/+0.8 pitch and ±0.6 yaw.
    # Real deer browse posture needs head ~vertical (pi/2 down or up).
    # Look-behind needs combined yaw chain (spine + neck) ~pi.
    "chest_to_neck_yaw":   (-1.5, 1.5),    # was (-0.6, 0.6) — chain with spine yaw -> ~pi total
    "neck_yaw_to_neck":    (-1.6, 1.6),    # was (-0.6, 0.8) — drink/browse pitch
    "neck_to_skull":       (-1.0, 1.0),    # was (-0.8, 0.8) — slight nod expand
    # Spine — slight expansion for pronk/jump
    "pelvis_to_chest_yaw": (-0.5, 0.5),    # was (-0.3, 0.3)
    "chest_yaw_to_chest":  (-0.7, 0.7),    # was (-0.5, 0.5)
    # Hocks — currently ±1.2 (symmetric); real hock flexes mostly one direction.
    "HR_shin_to_HR_foot":  (-1.6, 0.3),    # hind hock: flex inward (toward belly)
    "HL_shin_to_HL_foot":  (-1.6, 0.3),
    "FR_shin_to_FR_foot":  (-0.3, 1.6),    # front carpus: flex back (folds under body)
    "FL_shin_to_FL_foot":  (-0.3, 1.6),
}


def fmt(v: float) -> str:
    """Format float for URDF — 6 decimals, no trailing zeros for clarity."""
    return f"{v:.6f}"


def build_inertial_xml(mass: float, com: tuple, inertia: tuple, indent: str = "    ") -> str:
    cx, cy, cz = com
    ixx, iyy, izz = inertia
    return (
        f'{indent}<inertial><origin xyz="{fmt(cx)} {fmt(cy)} {fmt(cz)}" rpy="0 0 0" /><mass value="{mass}" />\n'
        f'{indent}  <inertia ixx="{fmt(ixx)}" ixy="0" ixz="0" iyy="{fmt(iyy)}" iyz="0" izz="{fmt(izz)}" /></inertial>'
    )


def main():
    if not URDF_IN.exists():
        print(f"ERROR: URDF not found at {URDF_IN}", file=sys.stderr); sys.exit(1)
    if not URDF_BACKUP.exists():
        URDF_BACKUP.write_text(URDF_IN.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup created: {URDF_BACKUP}")
    src = URDF_IN.read_text(encoding="utf-8")

    # 1. Replace inertial blocks per link.
    # Match: <link name="X"> ... <inertial> ... </inertial> ...
    # Use regex on the inertial block per link.
    for link_name in MASSES:
        if link_name == "antlers":
            continue  # added later as new link
        mass = MASSES[link_name]
        com  = COM_OFFSETS[link_name]
        inertia = compute_inertia(link_name, mass)
        new_inertial = build_inertial_xml(mass, com, inertia, indent="    ")

        pattern = re.compile(
            rf'(<link name="{re.escape(link_name)}">)\s*<inertial>.*?</inertial>',
            flags=re.DOTALL,
        )
        match = pattern.search(src)
        if not match:
            print(f"WARNING: could not find inertial for link '{link_name}'", file=sys.stderr)
            continue
        replacement = match.group(1) + "\n" + new_inertial
        src = src[:match.start()] + replacement + src[match.end():]

    # 2. Update skull collision filename to v2 (without antlers)
    src = src.replace("skull_collision.stl", "skull_collision_v2.stl")

    # 3. Joint limits — for each joint we override, replace lower/upper attrs.
    for joint_name, (lower, upper) in JOINT_LIMITS.items():
        pattern = re.compile(
            rf'(<joint name="{re.escape(joint_name)}"[^>]*type="revolute"[^>]*>.*?<limit )'
            rf'lower="[^"]*" upper="[^"]*"',
            flags=re.DOTALL,
        )
        m = pattern.search(src)
        if not m:
            print(f"WARNING: joint '{joint_name}' not matched", file=sys.stderr)
            continue
        new = m.group(1) + f'lower="{fmt(lower)}" upper="{fmt(upper)}"'
        src = src[:m.start()] + new + src[m.end():]

    # 4. Add antler link + fixed joint before </robot>.
    antler_mass = MASSES["antlers"]
    antler_com = COM_OFFSETS["antlers"]
    antler_inertia = compute_inertia("antlers", antler_mass)
    antler_link = (
        '  <!-- ===== Antlers: separate rigid body, fixed joint to skull. ===== -->\n'
        '  <link name="antlers">\n'
        + build_inertial_xml(antler_mass, antler_com, antler_inertia, indent="    ") + "\n"
        '    <collision><origin xyz="0 0 0" rpy="0 0 0" /><geometry><mesh filename="meshes/world/antler_collision_v2.stl" /></geometry></collision>\n'
        '  </link>\n'
        '  <joint name="skull_to_antlers" type="fixed">\n'
        '    <parent link="skull" /><child link="antlers" />\n'
        '    <origin xyz="0 0 0" rpy="0 0 0" />\n'
        '  </joint>\n'
    )
    src = src.replace("</robot>", antler_link + "</robot>")

    URDF_OUT.write_text(src, encoding="utf-8")
    print(f"Wrote {URDF_OUT}")
    print(f"Total mass: {sum(MASSES.values())} kg")


if __name__ == "__main__":
    main()
