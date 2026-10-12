"""URDF v4: split each foot into foot (cannon/metacarpus) + toe (phalanges/hoof)
via a pastern (fetlock) joint. Adds 4 new joints + 4 new links to the URDF.

This addresses the front-knee-walk exploit observed in v3 r2 trained policy:
the deer over-flexed the front carpus joint such that the rigid foot link's
top (cannon) contacted ground while the hoof tucked back. With the foot link
split, contact sensors can distinguish "hoof on ground" (good) from "cannon
on ground" (bad), and `undesired_contacts` regex on `*_foot$` penalizes the
cannon-walk while leaving hoof contact unrewarded for the right reasons.

Anatomical correctness preserved: deer can still fold front legs fully under
for sternal recumbency / sitting / browsing low because the carpus joint
range stays at (-0.3, +1.6 rad). Only the contact geometry is segmented.

Run:
    python tools/rebuild_deer_urdf_v4.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
URDF_IN = REPO / "assets" / "deer_urdf" / "deer_world.urdf"
URDF_OUT = REPO / "assets" / "deer_urdf" / "deer_world.urdf"
URDF_BACKUP = REPO / "assets" / "deer_urdf" / "deer_world.pre_v4.urdf"


# ============================================================================
# Mass redistribution: foot is split into cannon (kept as <prefix>_foot) +
# phalanges (new <prefix>_toe). Cannon ~70%, phalanges ~30% — anatomical.
# Total per leg foot mass stays the same as v3 (0.8 kg) so we don't need to
# rebalance other links.
# ============================================================================
FOOT_NEW_MASS = 0.55   # was 0.8 in v3 — cannon only
TOE_MASS = 0.25        # NEW — phalanges + hoof

# ============================================================================
# Pastern joint origins (pos of toe-link origin in parent foot-link frame).
# Derived from Blender source rig: foot-bone tail = toe-bone head.
# Forced L/R symmetric (mirrored from R-side) to match our v3 inertial
# symmetry policy.
# ============================================================================
HIND_PASTERN_DELTA  = (0.008378, -0.058986, -0.333707)   # HR; HL mirrors x
FRONT_PASTERN_DELTA = (0.003615, -0.046036, -0.271656)   # FR; FL mirrors x

PASTERN_JOINTS = {
    "HR_foot_to_HR_toe": {
        "parent": "HR_foot", "child": "HR_toe",
        "origin": HIND_PASTERN_DELTA,
    },
    "HL_foot_to_HL_toe": {
        "parent": "HL_foot", "child": "HL_toe",
        "origin": (-HIND_PASTERN_DELTA[0], HIND_PASTERN_DELTA[1], HIND_PASTERN_DELTA[2]),
    },
    "FR_foot_to_FR_toe": {
        "parent": "FR_foot", "child": "FR_toe",
        "origin": FRONT_PASTERN_DELTA,
    },
    "FL_foot_to_FL_toe": {
        "parent": "FL_foot", "child": "FL_toe",
        "origin": (-FRONT_PASTERN_DELTA[0], FRONT_PASTERN_DELTA[1], FRONT_PASTERN_DELTA[2]),
    },
}

# Real fetlock ROM: flexes inward ~70°, extends back ~25° during stance shock.
# Use generous range to allow shock absorption and lying-down poses.
PASTERN_LIMIT = (-0.40, 1.20)   # rad ≈ (-23°, +69°)
PASTERN_EFFORT = 50.0
PASTERN_VELOCITY = 10.0

# ============================================================================
# Toe link inertia (slim cylinder along Z axis = limb-length axis).
# Length per side from Blender rig: hind 0.34m, front 0.28m.
# ============================================================================
TOE_DIMS = {
    "HR_toe": (0.339, 0.025), "HL_toe": (0.339, 0.025),
    "FR_toe": (0.276, 0.025), "FL_toe": (0.276, 0.025),
}

# CoM offset within toe link frame (pointing roughly down and forward-ish).
# Toe extends from joint origin in -Y, -Z direction; CoM at midpoint.
TOE_COM_HIND  = (0.0, -0.030, -0.165)    # half of toe-bone delta
TOE_COM_FRONT = (0.0, -0.023, -0.135)


def ellipsoid_inertia(mass, length, radius):
    i_long = 0.5 * mass * radius**2
    i_trans = (1.0 / 12.0) * mass * (3 * radius**2 + length**2)
    return i_trans, i_trans, i_long


def fmt(v): return f"{v:.6f}"


def build_toe_link_xml(name, mass, com, inertia, collision_stl_filename):
    cx, cy, cz = com
    ixx, iyy, izz = inertia
    return (
        f'  <!-- Pastern: cannon ({name.replace("_toe","_foot")}) → hoof ({name}) -->\n'
        f'  <link name="{name}">\n'
        f'    <inertial><origin xyz="{fmt(cx)} {fmt(cy)} {fmt(cz)}" rpy="0 0 0" /><mass value="{mass}" />\n'
        f'      <inertia ixx="{fmt(ixx)}" ixy="0" ixz="0" iyy="{fmt(iyy)}" iyz="0" izz="{fmt(izz)}" /></inertial>\n'
        f'    <collision><origin xyz="0 0 0" rpy="0 0 0" /><geometry><mesh filename="meshes/world/{collision_stl_filename}" /></geometry></collision>\n'
        f'  </link>\n'
    )


def build_pastern_joint_xml(joint_name, info):
    o = info["origin"]
    return (
        f'  <joint name="{joint_name}" type="revolute">\n'
        f'    <parent link="{info["parent"]}" /><child link="{info["child"]}" />\n'
        f'    <origin xyz="{fmt(o[0])} {fmt(o[1])} {fmt(o[2])}" rpy="0 0 0" />\n'
        f'    <axis xyz="1 0 0" />\n'
        f'    <limit lower="{fmt(PASTERN_LIMIT[0])}" upper="{fmt(PASTERN_LIMIT[1])}" effort="{PASTERN_EFFORT}" velocity="{PASTERN_VELOCITY}" />\n'
        f'  </joint>\n'
    )


def main():
    if not URDF_IN.exists():
        print(f"ERROR: URDF not found at {URDF_IN}", file=sys.stderr); sys.exit(1)
    if not URDF_BACKUP.exists():
        URDF_BACKUP.write_text(URDF_IN.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup created: {URDF_BACKUP}")
    src = URDF_IN.read_text(encoding="utf-8")

    # 1. Update foot link mass (0.8 → 0.55) and collision STL filename
    for prefix in ["HR", "HL", "FR", "FL"]:
        link = f"{prefix}_foot"
        # Change mass
        src = re.sub(
            rf'(<link name="{link}">.*?<mass value=")[^"]+"',
            rf'\g<1>{FOOT_NEW_MASS}"',
            src,
            count=1,
            flags=re.DOTALL,
        )
        # Change collision STL ref
        src = src.replace(f"{prefix}_foot_collision.stl", f"{prefix}_foot_collision_v3.stl")

    # 2. Build new toe links + pastern joints
    new_links_xml = []
    new_joints_xml = []
    for prefix in ["HR", "HL", "FR", "FL"]:
        toe_name = f"{prefix}_toe"
        is_hind = prefix.startswith("H")
        L, r = TOE_DIMS[toe_name]
        inertia = ellipsoid_inertia(TOE_MASS, L, r)
        com = TOE_COM_HIND if is_hind else TOE_COM_FRONT
        # Mirror x for L side
        if prefix.endswith("L"):
            com = (-com[0], com[1], com[2])
        new_links_xml.append(build_toe_link_xml(
            toe_name, TOE_MASS, com, inertia,
            collision_stl_filename=f"{prefix}_toe_collision.stl",
        ))
        joint_name = f"{prefix}_foot_to_{prefix}_toe"
        new_joints_xml.append(build_pastern_joint_xml(joint_name, PASTERN_JOINTS[joint_name]))

    addition = (
        "\n  <!-- ===== Pasterns (fetlock joints) — v4 addition ===== -->\n"
        + "".join(new_links_xml)
        + "".join(new_joints_xml)
    )

    # 3. Insert before </robot>
    src = src.replace("</robot>", addition + "</robot>")

    URDF_OUT.write_text(src, encoding="utf-8")
    print(f"Wrote {URDF_OUT}")
    # Total mass check (rough)
    import re as _re
    masses = _re.findall(r'<mass value="([\d.]+)"', src)
    total = sum(float(m) for m in masses)
    print(f"Total mass: {total:.2f} kg (target 82.0)")
    print(f"Added 4 toe links ({TOE_MASS}kg each = {4*TOE_MASS}kg)")
    print(f"Reduced foot mass: 0.8 → {FOOT_NEW_MASS} ({4*(0.8-FOOT_NEW_MASS):.2f}kg redistributed to toes)")


if __name__ == "__main__":
    main()
