"""Repair the AMP reference clip: drop dead air, drop root-teleport seams, and
stitch the real walk cycles into one continuous trajectory.

WHY (audit 2026-07-24): the extracted `deer_polyart_walk.npy` was
  frames   0-103 : 3 real walk takes of 32 frames (~1.07 s, ~1.09 m/s each)
  frames  32-35  : root TELEPORT seam between takes (spikes to 18.6 m/s)
  frames  69-71  : ditto
  frames 104-248 : byte-identical FROZEN pose -- 58% dead air
amp-rsl-rl's MotionLoader finite-differences `root_position` to build the expert
AMP observation, so the discriminator was trained on a velocity distribution that
was 59.8% ~0 m/s and 1.6% teleport -- i.e. it rewarded the policy for STANDING
STILL, which is exactly the failure we spent 7 runs tuning rewards against. RSI
was hit too: its filter only excludes *fast* frames, so all 145 dead-air frames
passed and 58% of spawns were one identical frozen pose.

The 3 takes head in different world directions (-0.1 deg, -29.4 deg, +29.3 deg)
because each take restarts the root. We yaw-align each take to the previous one
and chain positions continuously, so the stitched clip is a single straight walk.
That matters because a raw concatenation would inject a ~29 deg instantaneous yaw
jump (~15 rad/s of bogus angular velocity) at each join.

Stitching is safe here: the take-to-take pose difference (~0.105 rad) is SMALLER
than the median frame-to-frame joint step (~0.237 rad), so joins fall within the
clip's own natural motion range.

Heights are deliberately left untouched -- the clip was already validated as
in-scale (pelvis 0.89-0.96 vs the sim's 0.95 stand height), and the ground-contact
check used link origins rather than mesh extents, so it isn't a trustworthy basis
for shifting the whole clip. Dead air + seams are the proven, dominant defects.

  python tools/rig_pipeline/repair_reference.py \
      --input assets/motions/deer_polyart_walk.npy \
      --output assets/motions/deer_polyart_walk_fixed.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]


def load_clip(path):
    """extract_reference runs in Blender (numpy 2.x) while this runs in Isaac Lab's
    interpreter (numpy 1.26). numpy 2 pickles reconstructors under `numpy._core`,
    absent in numpy 1 -> ModuleNotFoundError. Alias the moved modules."""
    if not hasattr(np, "_core"):
        import numpy.core as _c
        sys.modules.setdefault("numpy._core", _c)
        for sub in ("multiarray", "umath", "numeric", "_multiarray_umath"):
            mod = getattr(_c, sub, None)
            if mod is not None:
                sys.modules.setdefault(f"numpy._core.{sub}", mod)
    return np.load(str(path), allow_pickle=True).item()

# AUTHORITATIVE clip boundaries, taken from Unity's FBX import metadata
# ("Deer Walk.FBX.meta" -> clipAnimations), NOT inferred from root motion:
#     DWalk    firstFrame 0   lastFrame 32    loopTime 1   <- straight walk
#     DWalk R  firstFrame 36  lastFrame 68    loopTime 1   <- turns ~60 deg right
#     DWalk L  firstFrame 72  lastFrame 104   loopTime 1   <- turns ~60 deg left
# Malbers ships a DIRECTIONAL locomotion set; the extractor swallowed all three
# as one action. Ranges below are end-EXCLUSIVE, so lastFrame+1.
#
# The trailing frame IS kept: for a loopTime clip it is the loop-closing frame,
# measured 3x closer to frame 0 (mean 0.017 rad) than the previous frame is
# (0.057 rad), so dropping it would make a tiled loop seam WORSE, not better.
UNITY_CLIPS = {"DWalk": (0, 33), "DWalk R": (36, 69), "DWalk L": (72, 105)}
# One gait cycle = 33 frames @ 30 fps. Used as the yaw-analysis window: anything
# shorter measures body sway instead of net turning (see validate()).
STRIDE_SECONDS = 33.0 / 30.0
DEFAULT_CYCLES = [UNITY_CLIPS["DWalk"], UNITY_CLIPS["DWalk R"], UNITY_CLIPS["DWalk L"]]


def yaw_of(quat_xyzw: np.ndarray) -> float:
    """World-Z yaw of a quaternion, in radians."""
    return float(Rotation.from_quat(quat_xyzw).as_euler("zyx")[0])


def join_cost(jp, a, b):
    """Worst single-joint discontinuity if take `b` is played right after `a`.
    Joint angles are root-independent, so this is unaffected by yaw alignment."""
    return float(np.abs(jp[a[1] - 1] - jp[b[0]]).max())


def best_order(jp, cycles, verbose=True):
    """Pick the take ordering with the smallest worst-join discontinuity.

    The takes are independent repetitions of the same walk, so their order is
    free -- and choosing well matters: the natural 1,2,3 order happens to use the
    worst available join (a 1.18 rad / ~35 rad/s single-frame jump), which would
    feed the discriminator a bogus joint-velocity sample.
    """
    from itertools import permutations
    scored = []
    for perm in permutations(range(len(cycles))):
        joins = [join_cost(jp, cycles[perm[i]], cycles[perm[i + 1]])
                 for i in range(len(perm) - 1)]
        scored.append((max(joins), list(perm), joins))
    scored.sort(key=lambda x: x[0])
    worst, perm, joins = scored[0]
    if verbose:
        print(f"  take-order search: best={[p+1 for p in perm]} "
              f"worst_join={worst:.3f} rad (joins={[round(j,3) for j in joins]})")
        nat = next(s for s in scored if s[1] == list(range(len(cycles))))
        print(f"                     natural order {[p+1 for p in nat[1]]} "
              f"would be {nat[0]:.3f} rad")
    return [cycles[i] for i in perm]


def repair(jp, rp, rq, cycles, fps, verbose=True):
    """Yaw-align and chain the takes into one continuous trajectory."""
    out_jp, out_rp, out_rq = [], [], []
    for k, (s, e) in enumerate(cycles):
        c_jp, c_rp, c_rq = jp[s:e].copy(), rp[s:e].copy(), rq[s:e].copy()
        if k == 0:
            out_jp.append(c_jp); out_rp.append(c_rp); out_rq.append(c_rq)
            if verbose:
                print(f"  take {k+1}: frames {s}-{e-1} kept as anchor")
            continue

        prev_p, prev_q = out_rp[-1][-1], out_rq[-1][-1]
        # typical continuation step: reuse the previous take's last displacement
        d_typ = out_rp[-1][-1] - out_rp[-1][-2]

        dyaw = yaw_of(prev_q) - yaw_of(c_rq[0])
        R = Rotation.from_euler("z", dyaw)

        c_rq = (R * Rotation.from_quat(c_rq)).as_quat()          # xyzw
        rel = c_rp - c_rp[0]
        c_rp = prev_p + d_typ + R.apply(rel)

        out_jp.append(c_jp); out_rp.append(c_rp); out_rq.append(c_rq)
        if verbose:
            print(f"  take {k+1}: frames {s}-{e-1} yaw-aligned by "
                  f"{np.degrees(dyaw):+.1f} deg and chained")

    return (np.concatenate(out_jp).astype(np.float32),
            np.concatenate(out_rp).astype(np.float32),
            np.concatenate(out_rq).astype(np.float32))


def validate(jp, rp, rq, fps, label):
    """Report the properties that actually drive AMP + RSI."""
    v = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * fps
    w = np.abs(np.diff(Rotation.from_quat(rq).as_euler("zyx")[:, 0])) * fps
    w = np.minimum(w, 2 * np.pi * fps - w)  # unwrap
    jstep = np.abs(np.diff(jp, axis=0)).max(axis=1)

    print(f"\n=== {label} ===")
    print(f"  frames={len(jp)}  dur={len(jp)/fps:.2f}s")
    print(f"  root speed  : mean={v.mean():.3f} median={np.median(v):.3f} "
          f"min={v.min():.3f} max={v.max():.3f} m/s")
    print(f"  frozen frames (<0.01 m/s) : {int((v<0.01).sum())}/{len(v)} "
          f"({100*(v<0.01).mean():.1f}%)")
    print(f"  teleports    (>3 m/s)     : {int((v>3).sum())}/{len(v)}")
    # Yaw must be split into two very different things. A raw per-frame |yaw rate|
    # conflates them and badly overstates the problem: a perfectly straight walk
    # still reads ~0.9 rad/s because the body naturally sways each stride.
    #   * SYSTEMATIC turning = net heading change over a ~1 s window. This is what
    #     fights an ang_vel_z=0 command and makes the policy wander.
    #   * BODY SWAY = oscillation about the local mean. Natural; must be preserved.
    # The window MUST span at least one full stride (~1.1 s here), or the sway
    # does not cancel and a perfectly straight clip reads as heavy turning. (A
    # len//3 window on a 33-frame cycle reported the straight clip as turning
    # 39 deg/s -- more than the actual turn clips. Window size is the whole ballgame.)
    yaw = np.unwrap(Rotation.from_quat(rq).as_euler("zyx")[:, 0])
    win = min(len(yaw) - 1, max(3, int(round(fps * STRIDE_SECONDS))))
    if len(yaw) > win:
        sysd = np.degrees(yaw[win:] - yaw[:-win]) / (win / fps)   # deg per second
    else:
        sysd = np.array([np.degrees(yaw[-1] - yaw[0]) / (len(yaw) / fps)])
    wsign = np.diff(yaw) * fps
    loc = np.convolve(wsign, np.ones(win) / win, mode="same")
    sway = wsign - loc
    print(f"  yaw SYSTEMATIC turning : mean|{np.abs(sysd).mean():.1f}| max|{np.abs(sysd).max():.1f}| deg/s"
          f"   windows >20 deg/s: {int((np.abs(sysd)>20).sum())}/{len(sysd)} "
          f"({100*(np.abs(sysd)>20).mean():.0f}%)")
    print(f"    ^ env commands ang_vel_z=0, so THIS is what fights the task reward "
          f"(turn clips run ~30-60 deg/s; a straight clip is <5)")
    print(f"  yaw BODY SWAY (natural) : std={np.degrees(sway.std()):.1f} deg/s "
          f"[preserve this -- it is not a defect]")
    print(f"  joint step max: {jstep.max():.3f} rad")
    print(f"  speed histogram:")
    for lo, hi in [(0, .01), (.01, .5), (.5, .9), (.9, 1.3), (1.3, 3), (3, 1e9)]:
        n = int(((v >= lo) & (v < hi)).sum())
        bar = "#" * int(40 * n / max(1, len(v)))
        print(f"    [{lo:>5}, {hi:>5}) {n:4d} {bar}")
    net = np.linalg.norm(rp[-1, :2] - rp[0, :2])
    print(f"  net displacement={net:.3f} m over {len(jp)/fps:.2f}s "
          f"=> {net/(len(jp)/fps):.3f} m/s effective")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Defaults are idempotent: read the pristine extractor output (_raw), write
    # the canonical clip the training configs consume. Re-running never repairs
    # an already-repaired clip.
    ap.add_argument("--input", default=str(REPO / "assets/motions/deer_polyart_walk_raw.npy"))
    ap.add_argument("--output", default=str(REPO / "assets/motions/deer_polyart_walk.npy"))
    ap.add_argument("--tile", type=int, default=1,
                    help="Repeat the stitched walk N times (root motion accumulates). "
                         "AMP samples transitions, so 1 is usually enough.")
    ap.add_argument("--clip_range", default=None,
                    help="Explicit LO:HI (end-exclusive) frame range for a single clip, "
                         "overriding --takes. Use for non-walk gaits, whose clip splits "
                         "differ: DTrot=0:25, DRun=0:21 (Unity lastFrame+1, keeping the "
                         "loop-closing frame as for DWalk).")
    ap.add_argument("--takes", default="all",
                    help="Which takes to keep, 1-based (e.g. '1' or '1,3'). "
                         "Deer Walk.FBX is a Malbers DIRECTIONAL set: take 1 is a "
                         "straight walk, take 2 turns ~60 deg left, take 3 turns ~56 "
                         "deg right. With ang_vel_z commanded at 0, keeping the turn "
                         "takes puts ~67%% of expert frames at ~1 rad/s of yaw against "
                         "a zero-yaw command -- style and task rewards then fight, and "
                         "the policy wanders. Use '1' for a pure straight-walk target.")
    args = ap.parse_args()

    d = load_clip(args.input)
    jp = np.asarray(d["joint_positions"], np.float64)
    rp = np.asarray(d["root_position"], np.float64)
    rq = np.asarray(d["root_quaternion"], np.float64)
    fps = float(d["fps"])
    names = list(d["joints_list"])

    validate(jp, rp, rq, fps, f"BEFORE  ({Path(args.input).name})")

    if args.clip_range:
        _lo, _hi = (int(x) for x in str(args.clip_range).split(":"))
        cycles = [(_lo, _hi)]
    elif str(args.takes).lower() == "all":
        cycles = list(DEFAULT_CYCLES)
    else:
        idx = [int(t) - 1 for t in str(args.takes).split(",") if t.strip() != ""]
        cycles = [DEFAULT_CYCLES[i] for i in idx]
    print(f"\nstitching takes {cycles} (dropping dead air + teleport seams):")
    ordered = best_order(jp, cycles) if len(cycles) > 1 else cycles
    n_jp, n_rp, n_rq = repair(jp, rp, rq, ordered, fps)

    if args.tile > 1:
        base_jp, base_rp, base_rq = n_jp.copy(), n_rp.copy(), n_rq.copy()
        d_typ = base_rp[-1] - base_rp[-2]
        span = base_rp[-1] - base_rp[0]
        for i in range(1, args.tile):
            n_jp = np.concatenate([n_jp, base_jp])
            n_rq = np.concatenate([n_rq, base_rq])
            n_rp = np.concatenate([n_rp, base_rp + (span + d_typ) * i])
        print(f"  tiled x{args.tile} -> {len(n_jp)} frames")

    validate(n_jp, n_rp, n_rq, fps, f"AFTER   ({Path(args.output).name})")

    out = {"joints_list": names, "joint_positions": n_jp,
           "root_position": n_rp, "root_quaternion": n_rq, "fps": fps}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out, allow_pickle=True)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
