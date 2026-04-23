"""Extract Unity-space bind pose from Deer_001_idle.anim first frame.

The Blender-based bind dump (deer_bind_pose.json) is in Blender's imported
frame, which does NOT match Unity's bone-local frame because both importers
apply their own axis conversions. For the AMP motion converter we need bind
rotations in the SAME basis as the .anim quaternions -- i.e. Unity's.

Approach:
  1. Parse Deer_001_idle.anim the same way as walk.
  2. For each bone path, check whether the rotation curve is constant (or
     nearly so) across the clip. If yes, that constant value is the bind pose.
     If a bone has per-frame motion in idle (e.g. breathing), average or take
     t=0 -- the clip is ~1s, breathing motion is << walk amplitude.
  3. Also dump walk.anim frame-0 values for comparison. If walk[0] == idle[0]
     per bone, we have strong evidence idle IS the bind pose.

Output: assets/motions/deer_bind_pose_unity.json with Unity-frame rest quats.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

IDLE_PATH = Path(
    r"C:\Users\nars7\Downloads\Unity Projects\aniML-world"
    r"\Assets\ithappy\Animals_FREE\Animations\Other_Animations\Deer_001_idle.anim"
)
WALK_PATH = Path(
    r"C:\Users\nars7\Downloads\Unity Projects\aniML-world"
    r"\Assets\ithappy\Animals_FREE\Animations\Other_Animations\Deer_001_walk.anim"
)
OUT_PATH = Path(
    r"C:\Users\nars7\Downloads\aniML_world\assets\motions\deer_bind_pose_unity.json"
)

# Reuse parser from probe_walk_anim.py. Keep it inline so this script stands
# alone.

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


def parse_rotation_curves(text: str) -> list[Curve]:
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


def key_at_t0(curve: Curve) -> Key | None:
    if not curve.keys:
        return None
    # Keys are sorted by time in Unity serialization.
    return curve.keys[0]


def amplitude(curve: Curve) -> float:
    """Max angular distance between any two keyframes, in degrees."""
    if len(curve.keys) < 2:
        return 0.0
    max_d = 0.0
    k0 = curve.keys[0]
    for k in curve.keys[1:]:
        dot = abs(k0.x * k.x + k0.y * k.y + k0.z * k.z + k0.w * k.w)
        dot = max(-1.0, min(1.0, dot))
        ang = 2.0 * math.degrees(math.acos(dot))
        if ang > max_d:
            max_d = ang
    return max_d


def quat_dist_deg(a: Key, b: Key) -> float:
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.degrees(math.acos(dot))


def main() -> None:
    idle = parse_rotation_curves(IDLE_PATH.read_text(encoding="utf-8"))
    walk = parse_rotation_curves(WALK_PATH.read_text(encoding="utf-8"))
    print(f"idle: {len(idle)} rot curves   walk: {len(walk)} rot curves")

    idle_by_path = {c.path: c for c in idle}
    walk_by_path = {c.path: c for c in walk}

    bind: dict[str, dict] = {}
    all_paths = sorted(set(idle_by_path) | set(walk_by_path))

    print(f"\n{'path':<80s} idle_amp  idle_vs_walk0  src")
    for p in all_paths:
        c_idle = idle_by_path.get(p)
        c_walk = walk_by_path.get(p)
        k_idle = key_at_t0(c_idle) if c_idle else None
        k_walk = key_at_t0(c_walk) if c_walk else None

        if k_idle is not None and k_walk is not None:
            cross = quat_dist_deg(k_idle, k_walk)
        else:
            cross = -1.0

        idle_amp = amplitude(c_idle) if c_idle else 0.0
        # Prefer idle[0]; if idle missing, use walk[0]; if both missing, skip.
        src = None
        pick: Key | None = None
        if k_idle is not None:
            pick = k_idle
            src = "idle[0]"
        elif k_walk is not None:
            pick = k_walk
            src = "walk[0]"

        if pick is None:
            continue

        bind[p] = {
            "rest_quat_unity_xyzw": [pick.x, pick.y, pick.z, pick.w],
            "source": src,
            "idle_amplitude_deg": idle_amp,
            "idle_minus_walk0_deg": cross,
        }

        short = p.split("/")[-1]
        cross_s = f"{cross:+7.3f}" if cross >= 0 else "   N/A "
        print(f"{short:<30s}{p:<50s} {idle_amp:7.3f}°  {cross_s}°   {src}")

    # Flag any bone where idle != walk[0] by more than 2° -- that bone's
    # animation starts from a non-rest pose in walk, and we'd be using the
    # wrong bind. (Note: can also happen if the idle clip has real motion at
    # that bone, e.g. head bob.)
    flagged = [
        (p, b) for p, b in bind.items()
        if b["idle_minus_walk0_deg"] > 2.0
    ]
    print(f"\nBones where idle[0] differs from walk[0] by > 2°: {len(flagged)}")
    for p, b in flagged:
        short = p.split("/")[-1]
        print(f"  {short:<24s} diff={b['idle_minus_walk0_deg']:.2f}°  "
              f"idle_amp={b['idle_amplitude_deg']:.2f}°")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "source_idle": str(IDLE_PATH),
        "source_walk": str(WALK_PATH),
        "note": ("rest_quat is in Unity's bone-local frame (xyzw), taken from "
                 "idle.anim frame 0 where available, else walk.anim frame 0"),
        "bones": bind,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
