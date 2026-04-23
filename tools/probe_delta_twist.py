"""For each relevant bone, compute q_delta = q_rest^-1 * q_anim(t) across the
walk cycle and check whether the delta is dominated by X rotation (as the
URDF's single-axis joints assume).

Uses walk.anim frame 0 as "rest" -- for AMP we just need a consistent zero,
not the true bind. The URDF init_state can be set to match walk[0].

Per bone, reports:
  - max twist amplitude (rotation angle around X_bone)
  - max swing amplitude (rotation angle around the plane orthogonal to X_bone)
  - swing/twist ratio -- if >> 1, the URDF's single X-axis joint can't
    represent this bone's motion and we'd need an extra DoF

Writes report to assets/motions/deer_walk_twist_report.md.

Swing-twist decomposition (around X axis):
  Given q = (qx, qy, qz, qw):
    twist = normalize((qx, 0, 0, qw))        # projection onto X-axis rotation
    swing = q * twist^-1                      # residual
  angle_twist = 2 * atan2(|qx|, qw)
  angle_swing = 2 * atan2(sqrt(qy^2+qz^2), qw_twist)  -- we just compute
    angle_swing = 2 * acos(|swing.w|)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

WALK_PATH = Path(
    r"C:\Users\nars7\Downloads\Unity Projects\aniML-world"
    r"\Assets\ithappy\Animals_FREE\Animations\Other_Animations\Deer_001_walk.anim"
)
OUT_PATH = Path(
    r"C:\Users\nars7\Downloads\aniML_world\assets\motions\deer_walk_twist_report.md"
)

# The 12 leg bones (one per URDF leg joint, after the revised mapping).
LEG_BONE_NAMES = [
    # Rear right
    "Deer_001_rig/Root/spine.003/shoulder.R",                         # hip R
    "Deer_001_rig/Root/spine.003/shoulder.R/thigh.R/shin.R",          # knee R
    "Deer_001_rig/Root/spine.003/shoulder.R/thigh.R/shin.R/foot.R",   # ankle R
    # Rear left
    "Deer_001_rig/Root/spine.003/shoulder.L",
    "Deer_001_rig/Root/spine.003/shoulder.L/thigh.L/shin.L",
    "Deer_001_rig/Root/spine.003/shoulder.L/thigh.L/shin.L/foot.L",
    # Front right
    "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R",
    "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R/front_shin.R",
    "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R/front_shin.R/front_foot.R",
    # Front left
    "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L",
    "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L/front_shin.L",
    "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L/front_shin.L/front_foot.L",
]

# --- parser (same as probe_walk_anim) ---

KEY_RE = re.compile(
    r"- serializedVersion:\s*\d+\s*\n"
    r"\s*time:\s*(?P<t>[-\d.eE+]+)\s*\n"
    r"\s*value:\s*\{x:\s*(?P<x>[-\d.eE+]+),\s*y:\s*(?P<y>[-\d.eE+]+),"
    r"\s*z:\s*(?P<z>[-\d.eE+]+),\s*w:\s*(?P<w>[-\d.eE+]+)\}"
)
PATH_RE = re.compile(r"path:\s*(?P<path>[^\n]+)")


@dataclass
class Key:
    t: float
    x: float
    y: float
    z: float
    w: float


@dataclass
class Curve:
    path: str
    keys: list[Key] = field(default_factory=list)


def parse_rot_curves(text: str) -> list[Curve]:
    sec_re = re.compile(
        r"^  m_RotationCurves:\s*\n(?P<body>.*?)(?=^  m_[A-Z])",
        re.DOTALL | re.MULTILINE,
    )
    m = sec_re.search(text)
    if not m:
        return []
    body = m.group("body")
    entries = re.split(r"\n  - curve:\s*\n", "\n" + body)
    curves: list[Curve] = []
    for entry in entries[1:]:
        p = PATH_RE.search(entry)
        if not p:
            continue
        c = Curve(path=p.group("path").strip())
        for km in KEY_RE.finditer(entry):
            c.keys.append(Key(
                t=float(km.group("t")),
                x=float(km.group("x")),
                y=float(km.group("y")),
                z=float(km.group("z")),
                w=float(km.group("w")),
            ))
        curves.append(c)
    return curves


# --- quaternion helpers ---


def q_norm(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def q_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def q_angle_deg(q) -> float:
    _, _, _, w = q_norm(q)
    w = max(-1.0, min(1.0, abs(w)))
    return 2.0 * math.degrees(math.acos(w))


def swing_twist_x(q):
    """Decompose q into twist about X and swing orthogonal to X.

    Returns (twist_angle_deg (signed), swing_angle_deg (unsigned)).
    Signed twist preserves direction via atan2(qx, qw).
    """
    x, y, z, w = q_norm(q)
    # Twist: project onto X-axis rotation. twist_q = (qx, 0, 0, qw) / |.|
    tn = math.sqrt(x * x + w * w)
    if tn < 1e-12:
        # Pure 180° swing on Y-Z: undefined twist; treat as 0 twist
        twist_ang = 0.0
        swing_ang = 180.0
        return twist_ang, swing_ang
    tx, tw = x / tn, w / tn
    twist_ang = 2.0 * math.degrees(math.atan2(tx, tw))
    # Fold to (-180, 180]
    while twist_ang > 180.0:
        twist_ang -= 360.0
    while twist_ang <= -180.0:
        twist_ang += 360.0
    # Swing = q * twist^-1
    twist_q = (tx, 0.0, 0.0, tw)
    swing_q = q_mul((x, y, z, w), q_conj(twist_q))
    sx, sy, sz, sw = q_norm(swing_q)
    sw_c = max(-1.0, min(1.0, abs(sw)))
    swing_ang = 2.0 * math.degrees(math.acos(sw_c))
    return twist_ang, swing_ang


def main() -> None:
    text = WALK_PATH.read_text(encoding="utf-8")
    curves = parse_rot_curves(text)
    by_path = {c.path: c for c in curves}

    lines: list[str] = []
    lines.append("# Walk cycle — twist vs swing amplitude per leg bone")
    lines.append("")
    lines.append("Delta computed as `q_rest^-1 * q_anim(t)` with rest = walk.anim frame 0.")
    lines.append("")
    lines.append("| Bone | # keys | twist range (°) | swing max (°) | swing/twist ratio |")
    lines.append("|------|--------|-----------------|---------------|-------------------|")

    print(f"{'bone':<24s} {'nkeys':>5s} {'twist_min':>10s} {'twist_max':>10s} "
          f"{'twist_range':>11s} {'swing_max':>10s} {'ratio':>6s}")

    # Set of bones where the animator split rotation across two bones (e.g.
    # front_shoulder.R + front_thigh.R on the right front leg). Map
    # primary_bone_path -> list of contributing bone paths (in parent->child
    # composition order).
    COMPOSED_BONES = {
        "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R": [
            "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R",
            "Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R",
        ],
    }

    def sample_key(curve: Curve, i: int, fallback_t: float, fallback_idx: int):
        """Return the i-th key if it exists, else the last key."""
        if i < len(curve.keys):
            return curve.keys[i]
        # Repeat last key -- bones with fewer keys are effectively constant
        # between their last key and end.
        return curve.keys[-1]

    ok_count = 0
    bad_count = 0
    for path in LEG_BONE_NAMES:
        short = path.split("/")[-1]

        if path in COMPOSED_BONES:
            # Re-sample all contributing curves on a common key index using
            # the primary bone's keycount. This is approximate (ignores
            # exact times), but good enough to test the hypothesis that the
            # composition captures the full hip rotation.
            contribs = COMPOSED_BONES[path]
            curves_list = [by_path.get(p) for p in contribs]
            if any(c is None or not c.keys for c in curves_list):
                print(f"MISSING composed component for {short}")
                continue

            # Compose bind (q_rest) from all contributors, and anim per frame.
            def compose_keys(curves_list, idx_picker):
                q = (0.0, 0.0, 0.0, 1.0)
                for c in curves_list:
                    k = idx_picker(c)
                    q = q_mul(q, q_norm((k.x, k.y, k.z, k.w)))
                return q

            q_rest = compose_keys(curves_list, lambda c: c.keys[0])
            q_rest_inv = q_conj(q_norm(q_rest))

            # Align frames by sample index up to the MAX key count among
            # contributors.
            max_keys = max(len(c.keys) for c in curves_list)
            twists, swings = [], []
            for i in range(max_keys):
                q_a = compose_keys(
                    curves_list,
                    lambda c, ii=i: c.keys[min(ii, len(c.keys) - 1)],
                )
                q_d = q_mul(q_rest_inv, q_norm(q_a))
                tw, sw = swing_twist_x(q_d)
                twists.append(tw)
                swings.append(sw)

            short = short + " (composed)"
        else:
            c = by_path.get(path)
            if c is None or not c.keys:
                print(f"MISSING: {path}")
                continue

            q_rest = (c.keys[0].x, c.keys[0].y, c.keys[0].z, c.keys[0].w)
            q_rest_inv = q_conj(q_norm(q_rest))

            twists, swings = [], []
            for k in c.keys:
                q_a = q_norm((k.x, k.y, k.z, k.w))
                q_d = q_mul(q_rest_inv, q_a)
                tw, sw = swing_twist_x(q_d)
                twists.append(tw)
                swings.append(sw)

        twist_range = max(twists) - min(twists)
        swing_max = max(swings)
        ratio = swing_max / max(twist_range, 0.01)
        verdict = "OK" if ratio < 0.3 else ("MARGINAL" if ratio < 0.7 else "BAD")
        if verdict == "OK":
            ok_count += 1
        else:
            bad_count += 1
        nkeys_disp = len(twists)
        print(f"{short:<32s} {nkeys_disp:5d} {min(twists):10.2f} "
              f"{max(twists):10.2f} {twist_range:11.2f} {swing_max:10.2f} "
              f"{ratio:6.2f}  {verdict}")
        lines.append(
            f"| `{short}` | {nkeys_disp} | "
            f"[{min(twists):+.1f}, {max(twists):+.1f}] "
            f"(range {twist_range:.1f}) | {swing_max:.1f} | "
            f"{ratio:.2f} ({verdict}) |"
        )

    lines.append("")
    lines.append(f"**Summary: {ok_count} OK (ratio < 0.3), {bad_count} marginal/bad.**")
    lines.append("")
    lines.append("Decision rule:")
    lines.append("- ratio < 0.3: single-X URDF joint fully captures this bone's walk motion.")
    lines.append("- 0.3–0.7: significant but maybe tolerable swing — visualize before deciding.")
    lines.append("- > 0.7: URDF joint axis is wrong OR the rig has genuine off-axis motion that")
    lines.append("  can't be represented by an X-only joint. Either extend the URDF or redefine")
    lines.append("  the joint axis per bone.")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
