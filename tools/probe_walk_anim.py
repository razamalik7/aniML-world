"""Phase 0a pre-checks for AMP plan.

Answers two questions about Deer_001_walk.anim before we spend a day writing a
converter:

  1. Is the clip in-place (root motion baked out) or does it have real forward
     displacement? If in-place we need to synthesize velocity or the
     discriminator will learn "real walking = stationary root".

  2. What is the amplitude of the shoulder / front_shoulder bones' non-X
     rotation components? URDF collapses shoulder x thigh into one X-axis hip
     joint, so if the shoulder contributes significant Y/Z rotation we lose it.

Writes findings to assets/motions/deer_walk_probe.md.

The .anim file is Unity-tagged YAML. We don't need full YAML parsing -- the
structure is extremely regular, so regex extraction is safer and faster than
wrestling with a custom PyYAML loader.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ANIM_PATH = Path(
    r"C:\Users\nars7\Downloads\Unity Projects\aniML-world"
    r"\Assets\ithappy\Animals_FREE\Animations\Other_Animations\Deer_001_walk.anim"
)
OUTPUT_MD = Path(
    r"C:\Users\nars7\Downloads\aniML_world\assets\motions\deer_walk_probe.md"
)


@dataclass
class Keyframe:
    time: float
    x: float
    y: float
    z: float
    w: float  # for position curves, w is unused and defaults to 0


@dataclass
class Curve:
    path: str
    keys: list[Keyframe] = field(default_factory=list)


# --- parsing ---------------------------------------------------------------

# A top-level entry inside m_RotationCurves / m_PositionCurves is a block
# starting with "- curve:" and ending before the next "- curve:" or a new
# top-level section (m_ScaleCurves, etc.).
CURVE_BLOCK_RE = re.compile(
    r"- curve:\s*\n(?P<body>.*?)(?=\n  - curve:|\nm_[A-Z])",
    re.DOTALL,
)

KEY_RE = re.compile(
    r"- serializedVersion:\s*\d+\s*\n"
    r"\s*time:\s*(?P<t>[-\d.eE+]+)\s*\n"
    r"\s*value:\s*\{x:\s*(?P<x>[-\d.eE+]+),\s*y:\s*(?P<y>[-\d.eE+]+),"
    r"\s*z:\s*(?P<z>[-\d.eE+]+),\s*w:\s*(?P<w>[-\d.eE+]+)\}"
)

PATH_RE = re.compile(r"path:\s*(?P<path>[^\n]+)")


def parse_curves(text: str, section_header: str) -> list[Curve]:
    """Extract all curves under a given section (m_RotationCurves or m_PositionCurves)."""
    # Slice out just the section body between `section_header:` and the next
    # top-level `m_...` key at column 0.
    # All top-level AnimationClip keys are 2-space indented. Match exactly
    # that indent level on both the section header and the terminating key.
    sec_re = re.compile(
        rf"^{re.escape(section_header)}:\s*\n(?P<body>.*?)(?=^  m_[A-Z])",
        re.DOTALL | re.MULTILINE,
    )
    m = sec_re.search(text)
    if not m:
        return []
    body = m.group("body")

    curves: list[Curve] = []
    # Find each "  - curve:" entry (2-space indent)
    entries = re.split(r"\n  - curve:\s*\n", "\n" + body)
    for entry in entries[1:]:
        # entry is everything up to the next curve or end of section
        path_m = PATH_RE.search(entry)
        if not path_m:
            continue
        path = path_m.group("path").strip()
        keys = []
        for km in KEY_RE.finditer(entry):
            keys.append(
                Keyframe(
                    time=float(km.group("t")),
                    x=float(km.group("x")),
                    y=float(km.group("y")),
                    z=float(km.group("z")),
                    w=float(km.group("w")),
                )
            )
        curves.append(Curve(path=path, keys=keys))
    return curves


# --- analysis --------------------------------------------------------------


def position_extent(curve: Curve) -> dict:
    """Return min/max/range per axis for a position curve."""
    if not curve.keys:
        return {}
    xs = [k.x for k in curve.keys]
    ys = [k.y for k in curve.keys]
    zs = [k.z for k in curve.keys]
    return {
        "x_min": min(xs), "x_max": max(xs), "x_range": max(xs) - min(xs),
        "y_min": min(ys), "y_max": max(ys), "y_range": max(ys) - min(ys),
        "z_min": min(zs), "z_max": max(zs), "z_range": max(zs) - min(zs),
        "num_keys": len(curve.keys),
        "t_start": curve.keys[0].time,
        "t_end": curve.keys[-1].time,
    }


def quat_to_axis_angle_components(kf: Keyframe) -> tuple[float, float, float, float]:
    """For a unit quaternion (x,y,z,w), return (angle_deg, x_component_deg,
    y_component_deg, z_component_deg) where *_component_deg is the rotation
    angle projected onto that axis via 2*atan2(axis_val, w).

    This is a crude decomposition -- for the probe we only want amplitude, not
    an exact swing/twist split. If y or z components are non-trivial, that's
    the signal to worry about."""
    x, y, z, w = kf.x, kf.y, kf.z, kf.w
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return 0.0, 0.0, 0.0, 0.0
    x, y, z, w = x / n, y / n, z / n, w / n
    # Clamp
    w_c = max(-1.0, min(1.0, w))
    angle = 2.0 * math.degrees(math.acos(abs(w_c)))
    # Per-axis "naive" angle extraction
    ax = math.degrees(2.0 * math.atan2(x, w)) if abs(w) > 1e-9 else math.degrees(math.pi if x > 0 else -math.pi)
    ay = math.degrees(2.0 * math.atan2(y, w)) if abs(w) > 1e-9 else math.degrees(math.pi if y > 0 else -math.pi)
    az = math.degrees(2.0 * math.atan2(z, w)) if abs(w) > 1e-9 else math.degrees(math.pi if z > 0 else -math.pi)
    return angle, ax, ay, az


def rotation_component_amplitudes(curve: Curve) -> dict:
    """Return per-component amplitude in degrees across the curve."""
    if not curve.keys:
        return {}
    xs, ys, zs = [], [], []
    for k in curve.keys:
        _, ax, ay, az = quat_to_axis_angle_components(k)
        xs.append(ax)
        ys.append(ay)
        zs.append(az)
    return {
        "num_keys": len(curve.keys),
        "t_start": curve.keys[0].time,
        "t_end": curve.keys[-1].time,
        "x_amp_deg": max(xs) - min(xs),
        "y_amp_deg": max(ys) - min(ys),
        "z_amp_deg": max(zs) - min(zs),
        "x_min_deg": min(xs), "x_max_deg": max(xs),
        "y_min_deg": min(ys), "y_max_deg": max(ys),
        "z_min_deg": min(zs), "z_max_deg": max(zs),
    }


# --- main ------------------------------------------------------------------


def main() -> None:
    text = ANIM_PATH.read_text(encoding="utf-8")
    print(f"Loaded {ANIM_PATH.name} ({len(text):,} chars)")

    rot_curves = parse_curves(text, "  m_RotationCurves")
    pos_curves = parse_curves(text, "  m_PositionCurves")
    # Debug: if either parse hits zero, the regex is wrong -- bail loudly.
    if not rot_curves or not pos_curves:
        print(f"[WARN] parse returned rot={len(rot_curves)} pos={len(pos_curves)} "
              "-- regex likely mis-bounded")

    print(f"Parsed {len(rot_curves)} rotation curves, {len(pos_curves)} position curves")
    print()
    print("First 10 rotation curve paths:")
    for c in rot_curves[:10]:
        print(f"  {c.path}  ({len(c.keys)} keys)")
    print()
    print("Position curve paths:")
    for c in pos_curves:
        print(f"  {c.path}  ({len(c.keys)} keys)")

    # ------ Q1: Root motion? ----------------------------------------------
    root_candidates = [
        c for c in pos_curves
        if c.path in ("Deer_001_rig", "Deer_001_rig/Root")
        or c.path.endswith("/Root")
    ]
    print("\n==== Q1: Root motion ====")
    root_report_lines: list[str] = []
    if not root_candidates:
        msg = "NO position curves found for the root bone. Clip is in-place."
        print(msg)
        root_report_lines.append(msg)
    else:
        for c in root_candidates:
            ext = position_extent(c)
            print(f"  path={c.path}")
            for k, v in ext.items():
                print(f"    {k}: {v}")
            root_report_lines.append(f"### `{c.path}`\n")
            for k, v in ext.items():
                root_report_lines.append(f"- {k}: {v}")
            root_report_lines.append("")

    # Heuristic: Unity Y-up, so forward = Z. X_range or Z_range > ~0.05 m
    # suggests the root actually moves.
    in_place = True
    for c in root_candidates:
        ext = position_extent(c)
        max_horizontal_range = max(ext.get("x_range", 0), ext.get("z_range", 0))
        if max_horizontal_range > 0.05:
            in_place = False

    verdict_q1 = (
        "IN-PLACE (will need to synthesize forward velocity)"
        if in_place
        else "HAS ROOT MOTION (use as-is)"
    )
    print(f"\n  VERDICT: {verdict_q1}")

    # ------ Q2: Shoulder amplitude ----------------------------------------
    print("\n==== Q2: Shoulder rotation amplitude ====")
    shoulder_curves = [
        c for c in rot_curves
        if "shoulder" in c.path.lower()
    ]
    shoulder_report_lines: list[str] = []
    max_non_x_amp = 0.0
    for c in shoulder_curves:
        amps = rotation_component_amplitudes(c)
        short = c.path.split("/")[-1]
        print(f"  {short:24s} x_amp={amps['x_amp_deg']:6.2f}°  "
              f"y_amp={amps['y_amp_deg']:6.2f}°  z_amp={amps['z_amp_deg']:6.2f}°  "
              f"(keys={amps['num_keys']})")
        shoulder_report_lines.append(
            f"- `{c.path}` keys={amps['num_keys']} "
            f"x_amp={amps['x_amp_deg']:.2f}° "
            f"y_amp={amps['y_amp_deg']:.2f}° "
            f"z_amp={amps['z_amp_deg']:.2f}°"
        )
        max_non_x_amp = max(max_non_x_amp, amps["y_amp_deg"], amps["z_amp_deg"])

    verdict_q2 = (
        f"SIGNIFICANT non-X shoulder motion (max {max_non_x_amp:.1f}°) -- "
        "extend URDF or bake into root"
        if max_non_x_amp >= 5.0
        else f"Shoulder non-X amplitude max {max_non_x_amp:.1f}° < 5° -- "
        "treat hip as thigh-only, ignore shoulders"
    )
    print(f"\n  VERDICT: {verdict_q2}")

    # Also report thigh x_amp so we know leg actually swings
    print("\n---- Thigh rotation (sanity: should have real X amplitude) ----")
    thigh_curves = [
        c for c in rot_curves if "thigh" in c.path.lower()
    ]
    thigh_report_lines: list[str] = []
    for c in thigh_curves:
        amps = rotation_component_amplitudes(c)
        short = c.path.split("/")[-1]
        print(f"  {short:24s} x_amp={amps['x_amp_deg']:6.2f}°  "
              f"y_amp={amps['y_amp_deg']:6.2f}°  z_amp={amps['z_amp_deg']:6.2f}°")
        thigh_report_lines.append(
            f"- `{c.path}` "
            f"x_amp={amps['x_amp_deg']:.2f}° "
            f"y_amp={amps['y_amp_deg']:.2f}° "
            f"z_amp={amps['z_amp_deg']:.2f}°"
        )

    # Keyframe density summary
    key_counts = [len(c.keys) for c in rot_curves if c.keys]
    if key_counts:
        print(f"\nRotation curve keyframe counts: "
              f"min={min(key_counts)} max={max(key_counts)} "
              f"median={sorted(key_counts)[len(key_counts)//2]}")
        t_span_curves = [c for c in rot_curves if len(c.keys) >= 2]
        if t_span_curves:
            example = t_span_curves[0]
            print(f"Example time span: {example.path} "
                  f"t=[{example.keys[0].time}, {example.keys[-1].time}]")

    # ------ Write Markdown report ----------------------------------------
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Deer_001_walk.anim — Phase 0a Probe Report",
        "",
        f"Source: `{ANIM_PATH}`",
        "",
        f"- Rotation curves: {len(rot_curves)}",
        f"- Position curves: {len(pos_curves)}",
        f"- Max keyframes per curve: {max(key_counts) if key_counts else 0}",
        f"- Min keyframes per curve: {min(key_counts) if key_counts else 0}",
        "",
        "## Q1: Root motion",
        "",
        f"**VERDICT: {verdict_q1}**",
        "",
        *root_report_lines,
        "",
        "## Q2: Shoulder non-X rotation amplitude",
        "",
        f"**VERDICT: {verdict_q2}**",
        "",
        "Shoulder bones:",
        *shoulder_report_lines,
        "",
        "Thigh bones (sanity check — should swing on X):",
        *thigh_report_lines,
        "",
        "## Decisions",
        "",
        "- [ ] Root handling: IN-PLACE → synthesize velocity / HAS ROOT MOTION → use as-is",
        "- [ ] Shoulder handling: ignore / extend URDF / bake into root",
    ]
    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nReport written to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
