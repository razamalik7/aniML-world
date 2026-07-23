"""Rebuild corrected AMP reference clips **as copies**, never in place.

    python tools/cv/rebuild_reference_clips.py                 # dry run, report only
    python tools/cv/rebuild_reference_clips.py --write         # write assets/motions/v2/

Reads `assets/motions/<clip>.npy`, writes `assets/motions/v2/<clip>_v2.npy`.
It does NOT touch the live clips and does NOT touch
`tools/convert_unity_anim_to_npy.py` -- that file belongs to the locomotion
workstream. This is a post-processor, so the source of truth stays where it is.

WHAT IT FIXES, and why each is justified
----------------------------------------

1. **Joint names -> current URDF names.**
   The shipped clips use pre-v2 names (`pelvis_to_HR_thigh`, ...); the trainer
   asks for v2 names (`HR_hipabd_to_HR_thigh`, ...). `AMPLoader` aligns by name
   and fills anything unmatched with **constant zeros, silently**. It happens not
   to fire today only because `rsl_rl` 5.x makes `amp_joint_names` resolve to
   None, so the loader falls back to the dataset's own list and the two orderings
   are positionally identical. Setting `amp_joint_names` explicitly -- which the
   library's own warning tells you to do -- would send six joints to constant
   zero. Renaming removes the landmine permanently.

2. **Fetlock sign -- DISABLED BY DEFAULT, because measuring it refuted it.**
   `FR/FL_shin_to_*_foot` sit at [-1.20, -0.40] rad while the URDF allows
   [-0.30, +1.60]: outside the limit on **31 of 31 frames**. Negating lands them
   inside, at [+0.40, +1.20], and the URDF's front and hind conventions really are
   mirror images (front knee [0, +2.2] vs hind [-2.2, 0]), so a converter sign
   error looked like the obvious explanation.

   **It is not.** Checking what the angles do to the geometry:

   | variant | L front hoof lift | R front hoof lift |
   |---|---|---|
   | original clip | **0.145 m** | **0.144 m** |
   | negated | 0.515 m | 0.525 m |

   A walking deer lifts a hoof 0.10-0.20 m. The original values produce exactly
   that; negating produces a 0.5 m high-step no deer walks with. **The clip's
   fetlock angles are correct and the URDF's front fetlock LIMITS are wrong.**

   That inverts the fix: correct the rig, not the clip. And it means the front
   fetlock has been pinned at its lower limit (-0.30) throughout training, because
   the joint physically cannot reach the reference's range -- a concrete mechanism
   for the repeatedly-reported "front legs rigid / splayed as static struts".

   `--sign-fix` still exists so the experiment is reproducible, but it defaults
   off and should stay off.

3. **Root trajectory from foot odometry**, replacing the synthesised constant.
   The walk clip stores 0.77 m/s while its own leg motion implies 1.32 m/s, so
   the feet slide 3.9x farther than the body travels. Integrating root
   displacement from the stance foot fixes speed, slide and -- the part that
   matters most -- gives a real **within-stride velocity ripple**. A constant
   speed makes `base_lin_vel` dead flat, and since `base_lin_vel` is 3 of the 36
   dimensions the AMP discriminator sees, a flat channel is a trivially learnable
   "this is the reference" feature that can saturate the discriminator. This is
   the one thing a constant root genuinely gets wrong.

4. **Root height by FK grounding.** `spawn_height = 0.93` puts the hooves 10-17 cm
   underground. Setting root height per frame so the lowest hoof touches z=0 is
   exact and needs no constant.

5. **Optional lengthening** (`--cycles N`, `--mirror`). The clip is 1.03 s / 31
   frames, below every dataset size measured in the AMP literature (13.6 s
   cartwheel, 229.6 s walk, 434 s locomotion; 4.5 s for a real-robot A1 result).
   Concatenating cycles and appending a left/right-mirrored copy is free.

WHAT IT DELIBERATELY DOES NOT FIX
---------------------------------
Clamping-induced information loss. Where the original conversion clamped a joint
against the stale limits table, the true value is gone. Only a re-run of the
converter against URDF-derived limits recovers it. That is the locomotion
owner's call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from deer_rig import DeerRig
from keypoint_schema import DeerKeypoints

REPO = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO / "assets" / "motions"
OUT_DIR = SRC_DIR / "v2"

# pre-v2 clip joint name -> current URDF joint name
NAME_MAP: dict[str, str] = {
    "pelvis_to_chest":    "chest_yaw_to_chest",
    "pelvis_to_HR_thigh": "HR_hipabd_to_HR_thigh",
    "pelvis_to_HL_thigh": "HL_hipabd_to_HL_thigh",
    "chest_to_FR_thigh":  "FR_hipabd_to_FR_thigh",
    "chest_to_FL_thigh":  "FL_hipabd_to_FL_thigh",
    "chest_to_neck":      "neck_yaw_to_neck",
}

# Rx(-90 deg) spawn rotation, xyzw. Maps body -Z (forward) to world -Y.
SPAWN_QUAT_XYZW = np.array([-0.70710678, 0.0, 0.0, 0.70710678],
                           dtype=np.float32)

HOOF_KP = ["L_f_hoof", "R_f_hoof", "L_h_hoof", "R_h_hoof"]

# Left/right joint swaps for the mirrored copy.
MIRROR_PAIRS = [("HR", "HL"), ("FR", "FL")]


def body_to_world(p: np.ndarray) -> np.ndarray:
    """URDF body frame (+X left, +Y down, -Z fwd) -> Isaac world after Rx(-90).

    world = (x, z, -y):  body -Z (forward) -> world -Y, body +Y (down) -> world -Z.
    """
    p = np.asarray(p)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def load_clip(path: Path) -> dict:
    d = np.load(path, allow_pickle=True).item()
    return {
        "joints_list": [str(j) for j in d["joints_list"]],
        "joint_positions": np.asarray(d["joint_positions"], np.float64),
        "joint_velocities": np.asarray(d["joint_velocities"], np.float64),
        "root_position": np.asarray(d["root_position"], np.float64),
        "root_quaternion": np.asarray(d["root_quaternion"], np.float64),
        "fps": float(np.asarray(d["fps"]).reshape(-1)[0]),
    }


def fix_signs(jp: np.ndarray, names: list[str], rig: DeerRig,
              enable: bool) -> tuple[np.ndarray, list[str]]:
    """Negate channels that are out of URDF range but in range when negated."""
    jp = jp.copy()
    flipped: list[str] = []
    if not enable:
        return jp, flipped
    for i, n in enumerate(names):
        if n not in rig.joints:
            continue
        lo, hi = rig.joints[n].lower, rig.joints[n].upper
        c = jp[:, i]
        out_now = (c < lo - 1e-4).any() or (c > hi + 1e-4).any()
        neg = -c
        out_neg = (neg < lo - 1e-4).any() or (neg > hi + 1e-4).any()
        if out_now and not out_neg:
            jp[:, i] = neg
            flipped.append(n)
    return jp, flipped


def clamp_to_limits(jp: np.ndarray, names: list[str],
                    rig: DeerRig) -> tuple[np.ndarray, dict[str, int]]:
    jp = jp.copy()
    counts: dict[str, int] = {}
    for i, n in enumerate(names):
        if n not in rig.joints:
            continue
        lo, hi = rig.joints[n].lower, rig.joints[n].upper
        c = jp[:, i]
        n_out = int(((c < lo - 1e-6) | (c > hi + 1e-6)).sum())
        if n_out:
            counts[n] = n_out
        jp[:, i] = np.clip(c, lo, hi)
    return jp, counts


def hoof_positions(kps: DeerKeypoints, jp: np.ndarray,
                   names: list[str]) -> np.ndarray:
    """(T, 4, 3) hoof positions in the URDF body frame, root at origin."""
    idx = [kps.index(n) for n in HOOF_KP]
    T = jp.shape[0]
    out = np.zeros((T, len(idx), 3))
    for t in range(T):
        q = {n: float(jp[t, i]) for i, n in enumerate(names)}
        out[t] = kps.positions(q)[idx]
    return out


def root_from_foot_odometry(hoofs: np.ndarray, fps: float,
                            stance_frac: float = 0.25
                            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Integrate root displacement from whichever hoof is planted.

    A planted hoof is stationary in the world, so between consecutive frames the
    root moves by minus that hoof's displacement in body coordinates. This yields
    a per-frame root velocity including the within-stride surge that a constant
    speed cannot represent.

    Stance thresholds are computed **per hoof**, not pooled. On this rig the
    front hooves lift ~0.145 m and the hind ~0.10 m, so a pooled threshold
    classifies the hind pair almost arbitrarily. The clip is also in-place and
    cyclic, which means a hoof's swing displacement exactly cancels its stance
    displacement over a cycle -- so letting a single swinging hoof into the
    integration drives net travel to ~0. Hence: per-hoof thresholds, strict
    common stance, and a median rather than a mean across stance hooves.

    Returns (root_pos_world, stance_mask, stats).
    """
    T, K, _ = hoofs.shape
    down = hoofs[:, :, 1]                          # +Y is anatomical down
    lo, hi = down.min(axis=0), down.max(axis=0)    # per hoof
    thresh = hi - stance_frac * (hi - lo)
    stance = down >= thresh[None, :]               # (T, K)

    world_h = body_to_world(hoofs)                 # (T, K, 3)
    d = np.zeros((T, 3))
    for t in range(1, T):
        both = stance[t] & stance[t - 1]
        if not both.any():
            d[t] = d[t - 1]                        # hold velocity through flight
            continue
        step = -(world_h[t][both] - world_h[t - 1][both])
        d[t] = np.median(step, axis=0)

    pos = np.cumsum(d, axis=0)
    pos -= pos[0]

    # Ground the root: lowest hoof touches world z = 0 each frame.
    hoof_world_z = world_h[:, :, 2]          # relative to root
    pos[:, 2] = -hoof_world_z.min(axis=1)

    dt = 1.0 / fps
    vel = np.gradient(pos, dt, axis=0)
    horiz = np.linalg.norm(vel[:, :2], axis=1)
    # Consistency check: each leg pair implies its own stride length, and
    # therefore its own travel speed. A kinematically coherent clip has them
    # agree. If they disagree, no single root speed can satisfy both, and the
    # discriminator will see leg cadence and body speed that cannot co-occur.
    cycle_s = T / fps
    fore_aft = hoofs[:, :, 2]                 # body Z, fore-aft
    front_sweep = float(np.mean([fore_aft[:, 0].ptp(), fore_aft[:, 1].ptp()]))
    hind_sweep = float(np.mean([fore_aft[:, 2].ptp(), fore_aft[:, 3].ptp()]))

    stats = {
        "mean_speed": float(horiz.mean()),
        "speed_std": float(horiz.std()),
        "speed_min": float(horiz.min()),
        "speed_max": float(horiz.max()),
        "mean_root_height": float(pos[:, 2].mean()),
        "stance_fraction": float(stance.mean()),
        "total_travel": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
        "front_stride_m": front_sweep,
        "hind_stride_m": hind_sweep,
        "front_implied_speed": front_sweep / cycle_s,
        "hind_implied_speed": hind_sweep / cycle_s,
        "front_hind_disagreement": (abs(front_sweep - hind_sweep)
                                    / max(front_sweep, hind_sweep, 1e-6)),
        "hoof_lift_m": [float(hi[k] - lo[k]) for k in range(K)],
    }
    return pos, stance, stats


def constant_speed_root(rp: np.ndarray, fps: float,
                        speed: float) -> np.ndarray:
    """Straight-line root at a fixed speed, keeping the FK-grounded height.

    Travel is along world -Y, where the deer faces after the Rx(-90 deg) spawn
    rotation. Height keeps the per-frame value from FK grounding, which is
    exact and independent of the odometry question.
    """
    T = rp.shape[0]
    t = np.arange(T) / float(fps)
    out = rp.copy()
    out[:, 0] = 0.0
    out[:, 1] = -speed * t
    return out


def mirror_clip(jp: np.ndarray, names: list[str]) -> np.ndarray:
    """Left/right mirrored copy: swap paired leg channels."""
    out = jp.copy()
    idx = {n: i for i, n in enumerate(names)}
    for a, b in MIRROR_PAIRS:
        for n in names:
            if not n.startswith(a):
                continue
            partner = b + n[len(a):]
            if partner in idx:
                out[:, idx[n]] = jp[:, idx[partner]]
                out[:, idx[partner]] = jp[:, idx[n]]
    return out


def cyclic_velocity(x: np.ndarray, fps: float) -> np.ndarray:
    T = x.shape[0]
    if T < 2:
        return np.zeros_like(x)
    last = T - 2 if np.allclose(x[0], x[-1], atol=1e-4) else T - 1
    period = last + 1
    dt = 1.0 / fps
    out = np.zeros_like(x)
    for f in range(T):
        out[f] = (x[(f + 1) % period] - x[(f - 1) % period]) / (2 * dt)
    return out


def rebuild(clip: str, args, rig: DeerRig, kps: DeerKeypoints) -> None:
    src = SRC_DIR / f"{clip}.npy"
    if not src.is_file():
        print(f"  [skip] {src} not found")
        return
    d = load_clip(src)
    fps = d["fps"]
    names_old = d["joints_list"]
    names = [NAME_MAP.get(n, n) for n in names_old]
    jp = d["joint_positions"]

    print(f"\n=== {clip} ===")
    print(f"  source: {jp.shape[0]} frames @ {fps:g} fps")
    renamed = [(a, b) for a, b in zip(names_old, names) if a != b]
    print(f"  renamed {len(renamed)} joints to current URDF names")
    unknown = [n for n in names if n not in rig.joints]
    if unknown:
        print(f"  !! still not in URDF: {unknown}")

    jp, flipped = fix_signs(jp, names, rig, args.sign_fix)
    if args.sign_fix:
        print(f"  sign-flipped channels ({len(flipped)}): {flipped}")
    else:
        print("  sign fix: OFF (correct -- see docstring)")

    if args.clamp:
        jp, clamped = clamp_to_limits(jp, names, rig)
        print(f"  clamped to URDF limits: {clamped or 'none'}")
    else:
        _, would = clamp_to_limits(jp, names, rig)
        if would:
            print(f"  OUT OF URDF LIMITS (left unclamped, deliberately): {would}")
            print("    The clip's kinematics are correct here and the RIG's limits")
            print("    are wrong -- see docs/REFERENCE_CLIP_FINDINGS.md.")
            print("    Clamping would degrade a correct reference. Use --clamp to")
            print("    force it if the rig cannot be fixed.")
        else:
            print("  every frame within URDF limits")

    hoofs = hoof_positions(kps, jp, names)
    rp, stance, st = root_from_foot_odometry(hoofs, fps)
    if args.constant_speed is not None:
        rp = constant_speed_root(rp, fps, args.constant_speed)
        print(f"  root: CONSTANT {args.constant_speed:.3f} m/s along world -Y "
              f"(FK-grounded height retained)")

    old_speed = float(np.linalg.norm(
        d["root_position"][-1, :2] - d["root_position"][0, :2]
    ) / (jp.shape[0] / fps))
    print(f"  root speed: stored {old_speed:.3f} m/s  ->  foot-odometry "
          f"{st['mean_speed']:.3f} m/s "
          f"(min {st['speed_min']:.3f}, max {st['speed_max']:.3f}, "
          f"std {st['speed_std']:.3f})")
    print(f"  within-stride ripple: {st['speed_std']/max(st['mean_speed'],1e-6)*100:.1f}% "
          f"of mean (constant-speed reference has 0%)")
    print(f"  root height: stored {d['root_position'][0,2]:.3f} m  ->  "
          f"FK-grounded mean {st['mean_root_height']:.3f} m")
    print(f"  stance fraction {st['stance_fraction']:.2f}   "
          f"hoof lift (m): {[round(v,3) for v in st['hoof_lift_m']]}")
    print(f"  stride implied by FRONT legs {st['front_stride_m']:.3f} m "
          f"-> {st['front_implied_speed']:.3f} m/s")
    print(f"  stride implied by HIND  legs {st['hind_stride_m']:.3f} m "
          f"-> {st['hind_implied_speed']:.3f} m/s")
    dis = st['front_hind_disagreement']
    flag = "  <-- INCONSISTENT" if dis > 0.15 else ""
    print(f"  front/hind stride disagreement {dis*100:.0f}%{flag}")

    # optional lengthening
    if args.mirror:
        jp = np.concatenate([jp, mirror_clip(jp, names)], axis=0)
        hoofs = hoof_positions(kps, jp, names)
        rp, stance, st = root_from_foot_odometry(hoofs, fps)
        print(f"  + mirrored copy -> {jp.shape[0]} frames")
    if args.cycles > 1:
        jp = np.concatenate([jp] * args.cycles, axis=0)
        hoofs = hoof_positions(kps, jp, names)
        rp, stance, st = root_from_foot_odometry(hoofs, fps)
        print(f"  x{args.cycles} cycles -> {jp.shape[0]} frames "
              f"({jp.shape[0]/fps:.2f} s)")

    T = jp.shape[0]
    jv = cyclic_velocity(jp, fps)
    rq = np.tile(SPAWN_QUAT_XYZW, (T, 1))

    if not args.write:
        print("  (dry run -- pass --write to emit)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{clip}_v2.npy"
    np.save(out, {
        "joints_list": names,
        "joint_positions": jp.astype(np.float32),
        "joint_velocities": jv.astype(np.float32),
        "root_position": rp.astype(np.float32),
        "root_quaternion": rq.astype(np.float32),
        "fps": float(fps),
    }, allow_pickle=True)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*",
                    default=["deer_walk", "deer_run", "deer_idle"])
    ap.add_argument("--write", action="store_true",
                    help="actually emit files (default is a dry run)")
    ap.add_argument("--sign-fix", action="store_true",
                    help="Negate channels that are out of URDF range but in "
                         "range when negated. DEFAULT OFF -- measuring the "
                         "resulting hoof kinematics refuted this fix; see the "
                         "module docstring.")
    ap.add_argument("--constant-speed", type=float, default=None,
                    metavar="MPS",
                    help="Use a constant forward speed instead of foot "
                         "odometry. RECOMMENDED for the shipped clips: their "
                         "source animation has no valid stance phase (duty "
                         "factors 0.16-0.39 where a walk needs >0.5), so "
                         "odometry is unreliable on them. See "
                         "docs/REFERENCE_CLIP_FINDINGS.md section 2.")
    ap.add_argument("--clamp", action="store_true",
                    help="Clamp to URDF joint limits. DEFAULT OFF: the front "
                         "fetlock limits are themselves wrong, so clamping "
                         "degrades a correct clip.")
    ap.add_argument("--mirror", action="store_true",
                    help="append a left/right mirrored copy")
    ap.add_argument("--cycles", type=int, default=1,
                    help="repeat the clip N times")
    args = ap.parse_args()

    rig = DeerRig()
    kps = DeerKeypoints(rig)
    print(f"URDF: {rig.urdf_path}")
    print(f"output dir: {OUT_DIR}  (live clips in {SRC_DIR} are never modified)")
    for clip in args.clips:
        rebuild(clip, args, rig, kps)


if __name__ == "__main__":
    main()
