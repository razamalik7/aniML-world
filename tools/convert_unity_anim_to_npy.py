"""Convert Unity `.anim` clip to amp-rsl-rl `.npy` motion data.

Input:
  Unity 2020+ AnimationClip YAML (e.g. Deer_001_walk.anim) from the ithappy
  Animals_FREE pack. See assets/motions/deer_walk_probe.md for the analysis
  that motivates the mapping choices below.

Output: assets/motions/deer_walk.npy with dict:
  {
    "joints_list":     list[str]       # 15 URDF joint names, canonical order
    "joint_positions": np.ndarray      # (T, 15) float32, radians
    "joint_velocities":np.ndarray      # (T, 15) float32, rad/s (finite diff)
    "root_position":   np.ndarray      # (T, 3)  float32, meters, Isaac Z-up
    "root_quaternion": np.ndarray      # (T, 4)  float32, xyzw, Isaac Z-up
    "fps":             float
  }

Pipeline:
  1. Parse `.anim` YAML for all m_RotationCurves.
  2. Sample each bone's rotation at uniform 30 fps using cubic Hermite
     interpolation (matches Unity's curve playback for non-weighted tangents).
  3. For each URDF joint, pull the corresponding Unity bone (or composed
     pair), compute `q_delta = q_rest^-1 * q_anim(t)` with rest = walk[0]
     (composed where applicable), and extract the twist angle about
     bone-local X. This is the URDF joint angle (signed, radians).
  4. Clamp to URDF joint limits and log clamp counts. Error out if any joint
     requires clamping on more than 5 % of frames.
  5. Finite-difference joint positions -> velocities.
  6. Synthesize root_position as linear forward motion at FORWARD_SPEED along
     world -Y (the clip is in-place). The deer URDF was authored with body -Z
     = anatomical forward, and the Rx(-90°) spawn rotation maps body -Z to
     world -Y; so a deer walking forward in Isaac travels toward world -Y.
     Root orientation is the constant Rx(-90°) spawn rotation. AMPLoader
     computes base-frame velocities as R.T @ v_world, which for v_world =
     (0, -FORWARD_SPEED, 0) and R = Rx(-90°) produces v_body = (0, 0,
     -FORWARD_SPEED) — i.e. forward along body -Z, exactly what the running
     policy's base_lin_vel_b reports when the deer walks correctly.

Run:  python tools/convert_unity_anim_to_npy.py
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_anim_source_dir() -> Path:
    """Locate aniML-anim-source folder. Walks up from REPO_ROOT looking for it,
    so this works whether we run from the main project or a worktree.
    Falls back to the legacy Unity project path on the original machine."""
    cur = REPO_ROOT
    for _ in range(6):
        cand = cur / "aniML-anim-source"
        if cand.is_dir():
            return cand
        cand2 = cur.parent / "aniML-anim-source"
        if cand2.is_dir():
            return cand2
        cur = cur.parent
    legacy = Path(
        r"C:\Users\nars7\Downloads\Unity Projects\aniML-world"
        r"\Assets\ithappy\Animals_FREE\Animations\Other_Animations"
    )
    return legacy


ANIM_SOURCE_DIR = _find_anim_source_dir()
ANIM_PATH = ANIM_SOURCE_DIR / "Deer_001_walk.anim"
REST_ANIM_PATH = ANIM_SOURCE_DIR / "Deer_001_idle.anim"
OUT_PATH = REPO_ROOT / "assets" / "motions" / "deer_walk.npy"
OUT_REPORT = REPO_ROOT / "assets" / "motions" / "deer_walk_conversion_report.md"

FPS = 30.0

# Synthesized forward speed (m/s). Walk clip is in-place (see probe report).
# Middle of DeerUsdWalkEnvCfg velocity command range (0.0, 1.2). Faster ->
# policy learns a brisker walk.
FORWARD_SPEED = 0.8

# URDF joints in their canonical (training-config) order. Must match what the
# AMP discriminator sees from Isaac Lab.
URDF_JOINTS: list[str] = [
    "pelvis_to_chest",
    "pelvis_to_HR_thigh",
    "HR_thigh_to_HR_shin",
    "HR_shin_to_HR_foot",
    "pelvis_to_HL_thigh",
    "HL_thigh_to_HL_shin",
    "HL_shin_to_HL_foot",
    "chest_to_FR_thigh",
    "FR_thigh_to_FR_shin",
    "FR_shin_to_FR_foot",
    "chest_to_FL_thigh",
    "FL_thigh_to_FL_shin",
    "FL_shin_to_FL_foot",
    "chest_to_neck",
    "neck_to_skull",
]

# URDF joint limits (radians). MUST stay in sync with assets/deer_urdf/deer_world.urdf.
# Widened on 2026-04-27 from the original tighter set:
#   fetlock (shin_to_foot): ±0.50 -> ±1.20 rad  (28.6° -> 68.8°)
#   hip (pelvis_to_thigh) : ±0.80 -> ±1.40 rad  (45.8° -> 80.2°)
#   knee one-sided        : 1.80  -> 2.20 rad  (103° -> 126°)
#   front shoulder        : ±0.80 -> ±1.40 rad
# These come from configs/species/deer.json's biomechanical targets, which the
# original URDF was supposed to match but didn't (URDF was tighter for
# unclear reasons). Pre-widen, 11/12 leg joints clamped on deer_run.npy and
# 71% of front-fetlock frames clamped on deer_walk.npy — flat-footing the
# AMP reference. Widening unlocks a real walk/run shape.
URDF_LIMITS: dict[str, tuple[float, float]] = {
    "pelvis_to_chest":      (-0.2, 0.2),
    "pelvis_to_HR_thigh":   (-1.4, 1.4),
    "pelvis_to_HL_thigh":   (-1.4, 1.4),
    "chest_to_FR_thigh":    (-1.4, 1.4),
    "chest_to_FL_thigh":    (-1.4, 1.4),
    "HR_thigh_to_HR_shin":  (-2.2, 0.0),
    "HR_shin_to_HR_foot":   (-1.2, 1.2),
    "HL_thigh_to_HL_shin":  (-2.2, 0.0),
    "HL_shin_to_HL_foot":   (-1.2, 1.2),
    "FR_thigh_to_FR_shin":  (0.0, 2.2),
    "FR_shin_to_FR_foot":   (-1.2, 1.2),
    "FL_thigh_to_FL_shin":  (0.0, 2.2),
    "FL_shin_to_FL_foot":   (-1.2, 1.2),
    "chest_to_neck":        (-0.3, 0.5),
    "neck_to_skull":        (-0.4, 0.6),
}

# URDF joint -> list of Unity bone paths (in parent->child composition order).
# See probe_delta_twist.py for why front_shoulder.R needs a composed mapping.
REAR_R = "Deer_001_rig/Root/spine.003"
REAR_L = "Deer_001_rig/Root/spine.003"
FRONT  = "Deer_001_rig/Root/spine.003/spine.004/spine.005"
SPINE  = "Deer_001_rig/Root/spine.003"
NECK   = "Deer_001_rig/Root/spine.003/spine.004/spine.005"

JOINT_BONE_MAP: dict[str, list[str]] = {
    # Rear right leg
    "pelvis_to_HR_thigh":   [f"{REAR_R}/shoulder.R"],
    "HR_thigh_to_HR_shin":  [f"{REAR_R}/shoulder.R/thigh.R/shin.R"],
    "HR_shin_to_HR_foot":   [f"{REAR_R}/shoulder.R/thigh.R/shin.R/foot.R"],
    # Rear left leg
    "pelvis_to_HL_thigh":   [f"{REAR_L}/shoulder.L"],
    "HL_thigh_to_HL_shin":  [f"{REAR_L}/shoulder.L/thigh.L/shin.L"],
    "HL_shin_to_HL_foot":   [f"{REAR_L}/shoulder.L/thigh.L/shin.L/foot.L"],
    # Front right leg -- composition because animator split rotation
    "chest_to_FR_thigh":    [
        f"{FRONT}/front_shoulder.R",
        f"{FRONT}/front_shoulder.R/front_thigh.R",
    ],
    "FR_thigh_to_FR_shin":  [f"{FRONT}/front_shoulder.R/front_thigh.R/front_shin.R"],
    "FR_shin_to_FR_foot":   [f"{FRONT}/front_shoulder.R/front_thigh.R/front_shin.R/front_foot.R"],
    # Front left leg -- shoulder only; thigh is static
    "chest_to_FL_thigh":    [f"{FRONT}/front_shoulder.L"],
    "FL_thigh_to_FL_shin":  [f"{FRONT}/front_shoulder.L/front_thigh.L/front_shin.L"],
    "FL_shin_to_FL_foot":   [f"{FRONT}/front_shoulder.L/front_thigh.L/front_shin.L/front_foot.L"],
    # Spine -- two-segment compose
    "pelvis_to_chest":      [f"{SPINE}/spine.004", f"{SPINE}/spine.004/spine.005"],
    # Neck -- three-segment compose (006 x 007 x 008)
    "chest_to_neck":        [
        f"{NECK}/spine.006",
        f"{NECK}/spine.006/spine.007",
        f"{NECK}/spine.006/spine.007/spine.008",
    ],
    "neck_to_skull":        [f"{NECK}/spine.006/spine.007/spine.008/scull"],
}

# Per-joint sign: the twist direction may or may not match URDF X. Empirically
# derived. Leave all at +1 for now; the FIRST iteration showed raw twist lands
# within URDF limits for most joints and the remaining issues are rest-pose
# choice, not sign. Revisit after visualizing.
JOINT_SIGN: dict[str, float] = {j: 1.0 for j in URDF_JOINTS}


# --------------------------------------------------------------------------
# Unity .anim parser
# --------------------------------------------------------------------------

KEY_RE = re.compile(
    r"- serializedVersion:\s*\d+\s*\n"
    r"\s*time:\s*(?P<t>[-\d.eE+]+)\s*\n"
    r"\s*value:\s*\{x:\s*(?P<vx>[-\d.eE+]+),\s*y:\s*(?P<vy>[-\d.eE+]+),"
    r"\s*z:\s*(?P<vz>[-\d.eE+]+),\s*w:\s*(?P<vw>[-\d.eE+]+)\}\s*\n"
    r"\s*inSlope:\s*\{x:\s*(?P<ix>[-\d.eE+]+),\s*y:\s*(?P<iy>[-\d.eE+]+),"
    r"\s*z:\s*(?P<iz>[-\d.eE+]+),\s*w:\s*(?P<iw>[-\d.eE+]+)\}\s*\n"
    r"\s*outSlope:\s*\{x:\s*(?P<ox>[-\d.eE+]+),\s*y:\s*(?P<oy>[-\d.eE+]+),"
    r"\s*z:\s*(?P<oz>[-\d.eE+]+),\s*w:\s*(?P<ow>[-\d.eE+]+)\}"
)
PATH_RE = re.compile(r"path:\s*(?P<path>[^\n]+)")


@dataclass
class Key:
    t: float
    v: tuple[float, float, float, float]   # xyzw
    m_in: tuple[float, float, float, float]   # inSlope xyzw (tangent)
    m_out: tuple[float, float, float, float]  # outSlope xyzw


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
        pm = PATH_RE.search(entry)
        if not pm:
            continue
        c = Curve(path=pm.group("path").strip())
        for km in KEY_RE.finditer(entry):
            c.keys.append(Key(
                t=float(km.group("t")),
                v=(float(km.group("vx")), float(km.group("vy")),
                   float(km.group("vz")), float(km.group("vw"))),
                m_in=(float(km.group("ix")), float(km.group("iy")),
                      float(km.group("iz")), float(km.group("iw"))),
                m_out=(float(km.group("ox")), float(km.group("oy")),
                       float(km.group("oz")), float(km.group("ow"))),
            ))
        curves.append(c)
    return curves


# --------------------------------------------------------------------------
# Cubic Hermite interpolation (matches Unity for non-weighted tangents)
# --------------------------------------------------------------------------


def hermite_scalar(
    p0: float, p1: float,
    m0: float, m1: float,
    dt: float,
    s: float,
) -> float:
    """Cubic Hermite in [0,1] with tangents given per-second.

    When tangents are zero this reduces to smoothstep(s) interpolation, which
    is what Unity does for keys with auto/flat tangents (the vast majority of
    our keys, per inspection)."""
    h00 = 2 * s * s * s - 3 * s * s + 1
    h10 = s * s * s - 2 * s * s + s
    h01 = -2 * s * s * s + 3 * s * s
    h11 = s * s * s - s * s
    return h00 * p0 + h10 * (dt * m0) + h01 * p1 + h11 * (dt * m1)


def sample_curve(curve: Curve, t: float) -> tuple[float, float, float, float]:
    """Sample the quaternion at time t (seconds). Returns (x, y, z, w)
    with per-component Hermite interpolation; no re-normalization (caller
    does it when needed)."""
    keys = curve.keys
    if not keys:
        return (0.0, 0.0, 0.0, 1.0)
    if t <= keys[0].t:
        return keys[0].v
    if t >= keys[-1].t:
        return keys[-1].v
    # Find bracketing keys. Linear search is fine for < 20 keys.
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        if k0.t <= t <= k1.t:
            dt = k1.t - k0.t
            if dt <= 0.0:
                return k1.v
            s = (t - k0.t) / dt
            out = []
            for axis in range(4):
                p0 = k0.v[axis]
                p1 = k1.v[axis]
                m0 = k0.m_out[axis]
                m1 = k1.m_in[axis]
                out.append(hermite_scalar(p0, p1, m0, m1, dt, s))
            return tuple(out)  # type: ignore[return-value]
    return keys[-1].v


# --------------------------------------------------------------------------
# Quaternion ops (xyzw, SciPy-style)
# --------------------------------------------------------------------------


def q_norm(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
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


def twist_x_angle(q) -> float:
    """Twist angle about local X-axis in radians, signed, range (-pi, pi]."""
    x, y, z, w = q_norm(q)
    tn = math.sqrt(x * x + w * w)
    if tn < 1e-9:
        return 0.0
    tx, tw = x / tn, w / tn
    ang = 2.0 * math.atan2(tx, tw)
    while ang > math.pi:
        ang -= 2.0 * math.pi
    while ang <= -math.pi:
        ang += 2.0 * math.pi
    return ang


# --------------------------------------------------------------------------
# Core conversion
# --------------------------------------------------------------------------


def compose_bone_chain(
    bone_paths: list[str],
    by_path: dict[str, Curve],
    t: float,
) -> tuple[float, float, float, float]:
    """Sample each bone's rotation at time t, multiply in order, normalize."""
    q = (0.0, 0.0, 0.0, 1.0)
    for bp in bone_paths:
        c = by_path.get(bp)
        if c is None:
            raise KeyError(f"Missing rotation curve for bone: {bp}")
        qi = q_norm(sample_curve(c, t))
        q = q_mul(q, qi)
    return q_norm(q)


def convert(anim_path: Path, out_path: Path, rest_anim_path: Path | None = None) -> dict:
    print(f"Reading {anim_path.name}")
    text = anim_path.read_text(encoding="utf-8")
    curves = parse_rotation_curves(text)
    by_path = {c.path: c for c in curves}
    print(f"  parsed {len(curves)} rotation curves")

    # Optional separate rest-pose source (e.g. idle.anim frame 0). This often
    # better approximates the URDF's default standing pose than walk[0].
    rest_by_path: dict[str, Curve] = by_path
    if rest_anim_path is not None:
        print(f"Reading rest pose from {rest_anim_path.name}")
        rest_text = rest_anim_path.read_text(encoding="utf-8")
        rest_curves = parse_rotation_curves(rest_text)
        rest_by_path = {c.path: c for c in rest_curves}
        print(f"  parsed {len(rest_curves)} rest rotation curves")

    # Confirm all required bones exist in both sources
    missing = []
    for joint, bones in JOINT_BONE_MAP.items():
        for bp in bones:
            if bp not in by_path:
                missing.append((joint, bp, "walk"))
            if bp not in rest_by_path:
                missing.append((joint, bp, "rest"))
    if missing:
        for joint, bp, src in missing:
            print(f"  MISSING bone for joint {joint} in {src}: {bp}")
        raise RuntimeError("Cannot continue with missing bones")

    # Clip time span. We assume one key at t=0, last at t=T. In practice the
    # walk clip spans 0..1 s (33 frames at 30fps).
    t_start = min((c.keys[0].t for c in curves if c.keys), default=0.0)
    t_end   = max((c.keys[-1].t for c in curves if c.keys), default=1.0)
    duration = t_end - t_start
    n_frames = int(round(duration * FPS)) + 1
    print(f"  clip span: t=[{t_start:.3f}, {t_end:.3f}] s  "
          f"({duration:.3f}s, {n_frames} frames at {FPS} fps)")

    # Compute rest quaternion per joint. If rest_anim_path is provided we use
    # its t=0 pose; otherwise walk's t=0.
    rest_t = 0.0
    rest_q: dict[str, tuple[float, float, float, float]] = {}
    for joint, bones in JOINT_BONE_MAP.items():
        rest_q[joint] = compose_bone_chain(bones, rest_by_path, rest_t)

    # Sample every joint at uniform frame times.
    joint_positions = np.zeros((n_frames, len(URDF_JOINTS)), dtype=np.float32)
    clamp_counts: dict[str, int] = {j: 0 for j in URDF_JOINTS}
    twist_deg_max: dict[str, float] = {j: 0.0 for j in URDF_JOINTS}

    for f in range(n_frames):
        t = t_start + f / FPS
        for j_idx, joint in enumerate(URDF_JOINTS):
            q_anim = compose_bone_chain(JOINT_BONE_MAP[joint], by_path, t)
            q_delta = q_mul(q_conj(rest_q[joint]), q_anim)
            ang = twist_x_angle(q_delta) * JOINT_SIGN[joint]
            lo, hi = URDF_LIMITS[joint]
            if ang < lo - 1e-6 or ang > hi + 1e-6:
                clamp_counts[joint] += 1
                ang = max(lo, min(hi, ang))
            joint_positions[f, j_idx] = ang
            deg = math.degrees(ang)
            if abs(deg) > twist_deg_max[joint]:
                twist_deg_max[joint] = abs(deg)

    # Joint velocities via central differences (wrap-around: clip loops).
    joint_velocities = np.zeros_like(joint_positions)
    dt_frame = 1.0 / FPS
    # Treat the clip as a cyclic walk: velocity at frame 0 uses (frame 1 -
    # last_unique_frame). The last frame of a 1s clip at 30fps often duplicates
    # frame 0; we detect that and skip.
    last_unique = n_frames - 1
    if np.allclose(joint_positions[0], joint_positions[-1], atol=1e-4):
        last_unique = n_frames - 2
    for f in range(n_frames):
        fp = (f + 1) % (last_unique + 1)
        fm = (f - 1) % (last_unique + 1)
        joint_velocities[f] = (joint_positions[fp] - joint_positions[fm]) / (2 * dt_frame)

    # Synthesize root trajectory. Isaac Z-up. Clip is in-place so we advance
    # along world -Y at FORWARD_SPEED (deer URDF body -Z = anatomical
    # forward, mapped to world -Y by the Rx(-90°) spawn rotation).
    # See bake_reset_states.py for the matching convention used by RSI.
    spawn_height = 0.93  # matches articulation cfg
    root_position = np.zeros((n_frames, 3), dtype=np.float32)
    for f in range(n_frames):
        t = f / FPS
        root_position[f, 1] = -FORWARD_SPEED * t
        root_position[f, 2] = spawn_height

    # Root orientation: deer spawns with Rx(-90°) to convert URDF Y-up to
    # Isaac Z-up. Keep it constant; any Unity root rotation is baked into
    # per-bone rotations when we did q_delta relative to rest.
    # Rx(-90°) as xyzw: (-0.7071068, 0, 0, 0.7071068)
    # AMPLoader will compute base_lin_velocities_local = R.T @ v_world,
    # which for v_world = (0, -FORWARD_SPEED, 0) and R = Rx(-90°) yields
    # v_body = (0, 0, -FORWARD_SPEED) — i.e. forward along body -Z, the same
    # value the running policy emits as base_lin_vel_b when walking correctly.
    root_quaternion = np.tile(
        np.array([-0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32),
        (n_frames, 1),
    )

    # ---- Report ----
    over_threshold = [
        j for j, cnt in clamp_counts.items()
        if cnt / n_frames > 0.05
    ]

    print("\n  Per-joint twist max (deg) and clamp count:")
    for j in URDF_JOINTS:
        lo, hi = URDF_LIMITS[j]
        print(f"    {j:<24s} max|twist|={twist_deg_max[j]:6.2f}°  "
              f"limits=[{math.degrees(lo):+6.1f}°, {math.degrees(hi):+6.1f}°]  "
              f"clamped={clamp_counts[j]}/{n_frames}")

    if over_threshold:
        print(f"\n  WARNING: joints with > 5% clamp rate: {over_threshold}")
    else:
        print("\n  OK: no joint exceeds 5% clamp rate")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joints_list": URDF_JOINTS,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "root_position": root_position,
        "root_quaternion": root_quaternion,
        "fps": float(FPS),
    }
    np.save(out_path, payload, allow_pickle=True)
    print(f"\n  Saved {out_path}")
    print(f"    joint_positions shape: {joint_positions.shape}")
    print(f"    root_position final Y: {root_position[-1, 1]:.3f} m "
          f"(FORWARD_SPEED={FORWARD_SPEED} m/s along world -Y over {duration:.3f}s)")

    # Write markdown report
    lines = [
        f"# {anim_path.name} → {out_path.name} conversion report",
        "",
        f"- Source: `{anim_path}`",
        f"- Frames: {n_frames} at {FPS} fps ({duration:.3f} s)",
        f"- Forward speed synthesized: **{FORWARD_SPEED} m/s** along world -Y "
        f"(matches deer URDF body -Z = anatomical forward via Rx(-90°) spawn rotation)",
        f"- Spawn height: {spawn_height} m",
        "",
        "## Per-joint twist amplitude and clamp counts",
        "",
        "| URDF joint | max |twist| (°) | URDF limits (°) | clamped frames |",
        "|------------|-----------------:|-----------------|---------------:|",
    ]
    for j in URDF_JOINTS:
        lo, hi = URDF_LIMITS[j]
        lines.append(
            f"| `{j}` | {twist_deg_max[j]:.2f} | "
            f"[{math.degrees(lo):+.1f}, {math.degrees(hi):+.1f}] | "
            f"{clamp_counts[j]}/{n_frames} |"
        )
    lines.append("")
    lines.append(
        "## Verdict"
        if not over_threshold
        else "## ⚠ Clamp issues"
    )
    if over_threshold:
        lines.append("")
        lines.append("Joints exceeding 5 % clamp rate:")
        for j in over_threshold:
            lines.append(f"- `{j}` — {clamp_counts[j]}/{n_frames} frames clamped")
    else:
        lines.append("")
        lines.append("No joint exceeds the 5 % clamp threshold. Motion data is in-range for the URDF.")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {OUT_REPORT}")

    return payload


def convert_many() -> None:
    """Convert every known deer motion clip into .npy form.

    Uses the same REST_ANIM_PATH (idle.anim[0]) as rest pose for all clips,
    so joint angles live in the same reference frame. Speeds are synthesized
    per-clip since the source .anim are in-place (no root translation).
      walk:  0.8 m/s
      run :  2.5 m/s
      idle:  0.0 m/s (stationary)"""
    clips = [
        (ANIM_SOURCE_DIR / "Deer_001_walk.anim",
         REPO_ROOT / "assets" / "motions" / "deer_walk.npy", 0.8),
        (ANIM_SOURCE_DIR / "Deer_001_run.anim",
         REPO_ROOT / "assets" / "motions" / "deer_run.npy", 2.5),
        (ANIM_SOURCE_DIR / "Deer_001_idle.anim",
         REPO_ROOT / "assets" / "motions" / "deer_idle.npy", 0.0),
    ]
    global FORWARD_SPEED, OUT_REPORT
    base_speed, base_report = FORWARD_SPEED, OUT_REPORT
    for anim, out, speed in clips:
        if not anim.is_file():
            print(f"  [skip] {anim.name}: source not found at {anim}")
            continue
        FORWARD_SPEED = speed
        OUT_REPORT = out.with_suffix("").parent / (out.stem + "_conversion_report.md")
        try:
            print(f"\n=== converting {anim.name} at {speed} m/s -> {out.name} ===")
            convert(anim, out, rest_anim_path=REST_ANIM_PATH)
        except Exception as e:
            print(f"  {anim.name}: FAILED -- {e}")
    FORWARD_SPEED, OUT_REPORT = base_speed, base_report


if __name__ == "__main__":
    import sys as _sys
    if "--all" in _sys.argv:
        convert_many()
    else:
        convert(ANIM_PATH, OUT_PATH, rest_anim_path=REST_ANIM_PATH)
