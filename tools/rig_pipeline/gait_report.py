"""Is the deer WALKING? A quantitative gait report -- the acceptance test.

Built because every judgement up to r9 came from eyeballing renders or from
aggregate metrics that structurally CANNOT see a gait. r9 scored 7.75 m of travel
at 1.29 m/s with straightness 0.99 and a pelvis held at 0.82 -- and was still not
walking: it planted its left legs (duty 0.97, one liftoff in 6 s) and pushed with
the right, a heavy limp. Displacement, speed, straightness and height are all
blind to that. Footfall pattern is not.

Reports, per leg:
  duty factor   fraction of time the hoof bears load  (walk: ~0.6-0.75)
  liftoffs/s    step frequency                        (walk: ~0.9-1.0)
  joint activity how much the leg actually articulates (ref: ~16-18 deg std)
plus whole-body displacement, speed, straightness and pelvis height, and a
footfall diagram so the gait pattern is visible at a glance.

Contact source, in order of preference:
  1. `contact_forces` recorded from PhysX by collect_rollout -- ground truth, and
     exactly what the reward terms read.
  2. FK hoof height (fallback, e.g. for the reference .npy). Less reliable here:
     this rig's four hooves disagree about floor level by up to 5.8 cm (the 82 mm
     FK reconstruction error), so a fixed height threshold mislabels stance/swing.

  python tools/rig_pipeline/gait_report.py --input docs/polyart_r9_final_rollout.npz
  python tools/rig_pipeline/gait_report.py --reference        # ground-truth targets
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LEGS = {
    "L-hind":  dict(hoof="L_Toe0",    links=["L_Thigh", "L_Calf", "L_HorseLink", "L_Foot", "L_Toe0"]),
    "R-hind":  dict(hoof="R_Toe0",    links=["R_Thigh", "R_Calf", "R_HorseLink", "R_Foot", "R_Toe0"]),
    "L-front": dict(hoof="L_Finger0", links=["L_UpperArm", "L_Forearm", "L_Hand", "L_Finger0"]),
    "R-front": dict(hoof="R_Finger0", links=["R_UpperArm", "R_Forearm", "R_Hand", "R_Finger0"]),
}
# A walking quadruped: each foot down ~60-75% of the stride, ~1 step/s at ~1 m/s,
# and all four legs articulating by a similar amount.
TARGET = dict(duty=(0.55, 0.80), liftoff=(0.6, 1.6), activity=(12.0, 24.0))


def _shim():
    if not hasattr(np, "_core"):
        import numpy.core as _c
        sys.modules.setdefault("numpy._core", _c)
        for s in ("multiarray", "umath", "numeric", "_multiarray_umath"):
            m = getattr(_c, s, None)
            if m is not None:
                sys.modules.setdefault(f"numpy._core.{s}", m)


def load_any(path):
    """Return (joints_list, jp, rp, rq_wxyz, fps, contacts|None, contact_bodies|None)."""
    _shim()
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(str(p), allow_pickle=True)
        cf = d["contact_forces"] if "contact_forces" in d.files else None
        cb = list(d["contact_bodies"].astype(str)) if "contact_bodies" in d.files else None
        return (list(d["joints_list"].astype(str)), d["joint_positions"].astype(float),
                d["root_pos"].astype(float), d["root_quat_wxyz"].astype(float),
                float(d["fps"]), cf, cb)
    d = np.load(str(p), allow_pickle=True).item()          # reference .npy
    rq = np.asarray(d["root_quaternion"], float)[:, [3, 0, 1, 2]]   # xyzw -> wxyz
    jp = np.asarray(d["joint_positions"], float)[:, None, :]
    rp = np.asarray(d["root_position"], float)[:, None, :]
    return list(d["joints_list"]), jp, rp, rq[:, None, :], float(d["fps"]), None, None


def hoof_heights_fk(names, jp, rp, rq):
    """Lowest world-Z of each hoof's COLLISION mesh, per frame (fallback path)."""
    import trimesh
    from tools.render_playback_meshed import (
        parse_urdf_full, compute_link_transforms, quat_wxyz_to_R, rpy_to_R)
    urdf = REPO / "assets/deer_polyart_urdf/deer_polyart.urdf"
    text = urdf.read_text(encoding="utf-8")
    origin_map, _ = parse_urdf_full(urdf)
    col = {}
    for m in re.finditer(r'<link name="([^"]+)">(.*?)</link>', text, re.S):
        n, b = m.group(1), m.group(2)
        cm = re.search(r'<collision[^>]*>\s*<origin\s+xyz="([^"]+)"(?:\s+rpy="([^"]+)")?[^>]*/>\s*'
                       r'<geometry>\s*<mesh filename="([^"]+)"', b, re.S)
        if cm and (urdf.parent / cm.group(3)).is_file():
            col[n] = (np.asarray(trimesh.load_mesh(str(urdf.parent / cm.group(3)),
                                                   force="mesh").vertices, float),
                      np.array([float(v) for v in cm.group(1).split()]),
                      rpy_to_R(np.array([float(v) for v in (cm.group(2) or "0 0 0").split()])))
    out = {k: [] for k in LEGS}
    for t in range(len(jp)):
        ja = {n: float(jp[t, i]) for i, n in enumerate(names)}
        lt = compute_link_transforms(ja, origin_map)
        R0 = quat_wxyz_to_R(rq[t]); t0 = rp[t]
        for leg, spec in LEGS.items():
            h = spec["hoof"]
            if h in lt and h in col:
                Rl, tl = lt[h]; V, cx, cR = col[h]
                out[leg].append((V @ (R0 @ Rl @ cR).T + (R0 @ (Rl @ cx + tl) + t0))[:, 2].min())
    return {k: np.array(v) for k, v in out.items()}


def runs_stats(down, fps):
    """duty factor, liftoff rate (down->up transitions per second)."""
    duty = float(down.mean())
    lift = int(((~down[1:]) & (down[:-1])).sum())
    return duty, lift / (len(down) / fps)


def flag(v, lo_hi):
    lo, hi = lo_hi
    return "ok " if lo <= v <= hi else ("LOW" if v < lo else "HIGH")


def report_env(label, names, jp, rp, rq, fps, cf, cb, force_thresh, show_diagram,
               ref_jp=None, ref_names=None):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    T = len(jp)

    # ---- whole-body ----
    p = rp[:, :2]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    net = float(np.linalg.norm(p[-1] - p[0]))
    dur = T / fps
    print(f"  travel {net:5.2f} m in {dur:4.2f}s = {net/dur:5.3f} m/s   "
          f"straightness {net/max(step.sum(),1e-9):4.2f}   "
          f"pelvis z mean {rp[:,2].mean():4.2f} min {rp[:,2].min():4.2f}")

    # ---- contact ----
    trusted = cf is not None and cb is not None
    if trusted:
        src = f"PhysX contact forces (>{force_thresh:.0f} N = bearing load)"
        down = {leg: cf[:, cb.index(s["hoof"])] > force_thresh
                for leg, s in LEGS.items() if s["hoof"] in cb}
    else:
        z = hoof_heights_fk(names, jp, rp, rq)
        src = "FK hoof height (FALLBACK -- duty factors are NOT trustworthy)"
        down = {leg: a < (a.min() + 0.03) for leg, a in z.items()}
    print(f"  contact source: {src}")
    if not trusted:
        # Calibration check: run this on the reference clip (a real deer walk) and the
        # FK path reports the FRONT hooves at duty 0.27/0.36 -- i.e. it would declare
        # genuine ground truth "not a walk". A front hoof swings high and rolls through
        # stance, so "within 3 cm of its own minimum" misses much of its stance phase.
        # So: show the numbers and the pattern, but refuse to issue a verdict. Duty
        # factors are only meaningful from PhysX contacts.
        print("  NOTE: on the reference clip this path under-reports front-hoof duty "
              "(0.27/0.36) and would wrongly call a real walk a non-walk. Treat the "
              "footfall PATTERN as informative and the duty NUMBERS as indicative only.")

    # ---- per leg ----
    print(f"\n  {'leg':<9}{'duty':>7}{'':>5}{'lift/s':>8}{'':>5}{'activity':>10}{'':>5}")
    for leg, spec in LEGS.items():
        if leg not in down:
            continue
        duty, lift = runs_stats(down[leg], fps)
        cols = [i for i, n in enumerate(names)
                if any(n.startswith(l + "_to_") or f"_to_{l}" in n for l in spec["links"])]
        act = float(np.degrees(jp[:, cols].std(axis=0).mean())) if cols else float("nan")
        print(f"  {leg:<9}{duty:>7.2f} {flag(duty,TARGET['duty']):<4}"
              f"{lift:>8.2f} {flag(lift,TARGET['liftoff']):<4}"
              f"{act:>9.1f}d {flag(act,TARGET['activity']):<4}")

    # ---- verdict ----
    ok = all(TARGET['duty'][0] <= runs_stats(down[l], fps)[0] <= TARGET['duty'][1]
             for l in down)
    lifts = [runs_stats(down[l], fps)[1] for l in down]
    balanced = (max(lifts) - min(lifts)) < 0.8 if lifts else False
    print(f"\n  duty in walk range: {'YES' if ok else 'NO'}    "
          f"step rate balanced across legs: {'YES' if balanced else 'NO'}"
          f"  (spread {max(lifts)-min(lifts):.2f}/s)")

    # LEG PARTICIPATION. Added 2026-07-29 after Raza repeatedly described "leg neglect"
    # in runs this harness scored well. The failure is a leg that barely bears load at
    # all, and neither duty-in-range nor step-rate-balance catches it:
    #   * C2 ran ONE hind leg at 0.51 duty with the other three under 0.01 -- three
    #     effectively vestigial legs -- while scoring 0.51 on left-right asymmetry.
    #   * C3 scored 0.10 left-right (near-best) with front/hind load at 0.37.
    # A left-vs-right comparison is blind to a deer that uses both hind legs evenly and
    # lets the front float, which is exactly the shape of the problem.
    duties = {l: float(down[l].mean()) for l in down}
    dead = [l for l, v in duties.items() if v < 0.08]
    hind = sum(v for l, v in duties.items() if "hind" in l.lower())
    front = sum(v for l, v in duties.items() if "front" in l.lower())
    ratio = front / hind if hind > 1e-6 else float("nan")
    print(f"  LEG PARTICIPATION: front/hind load {ratio:.2f} "
          f"(1.0 = balanced, <0.5 = front legs barely bearing)")
    if dead:
        print(f"     *** {len(dead)} leg(s) under 8% duty -- effectively unused: "
              f"{', '.join(dead)} ***")
    if trusted:
        print(f"  VERDICT: {'WALKING' if (ok and balanced) else 'NOT a walk gait'}")
    else:
        print(f"  VERDICT: INDETERMINATE -- FK contact cannot support a verdict. "
              f"Re-run on a collect_rollout .npz (records PhysX contact forces).")

    # --- joint-range conformance ---
    # ADDED after r10: the policy was riding its JOINT LIMITS -- 95% of the left hoof
    # ankle's samples sat outside the range the reference walk ever uses, and its range
    # was the LIMIT range, not the walk range. Nothing here caught it; the user spotted
    # the ankles looking wrong in a video. Duty/liftoff/activity are all blind to a joint
    # that cycles correctly but through the wrong ANGLES. This is that missing check.
    if ref_jp is not None and ref_names is not None:
        print(f"\n  JOINT-RANGE CONFORMANCE (% of samples outside the reference's range):")
        rows = []
        for i, n in enumerate(names):
            if not any(k in n for k in ("Thigh", "Calf", "HorseLink", "Foot", "Toe",
                                        "UpperArm", "Forearm", "Hand", "Finger")):
                continue
            if n not in ref_names:
                continue
            r = ref_jp[:, ref_names.index(n)]
            lo, hi = r.min(), r.max()
            out = float(((jp[:, i] < lo - np.radians(1)) | (jp[:, i] > hi + np.radians(1))).mean())
            rows.append((out, n, np.degrees(lo), np.degrees(hi),
                         np.degrees(jp[:, i].min()), np.degrees(jp[:, i].max())))
        rows.sort(reverse=True)
        for out, n, lo, hi, pa, pb in rows[:6]:
            flagm = " <== OUT OF STYLE" if out > 0.25 else ""
            print(f"    {n:<28} ref[{lo:>7.1f},{hi:>6.1f}] policy[{pa:>7.1f},{pb:>6.1f}]"
                  f"  {100*out:4.0f}%{flagm}")
        bad = sum(1 for r in rows if r[0] > 0.25)
        print(f"    {bad}/{len(rows)} leg joints spend >25% of the time outside the "
              f"reference range" + ("  <-- will look wrong" if bad else "  (good)"))

    if show_diagram:
        print("\n  footfall (#=bearing load, .=swing), one column per frame:")
        w = min(T, 100)
        idx = np.linspace(0, T - 1, w).astype(int)
        for leg in LEGS:
            if leg in down:
                print(f"    {leg:<9}|" + "".join("#" if down[leg][i] else "." for i in idx) + "|")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default=None, help="rollout .npz from collect_rollout")
    ap.add_argument("--reference", action="store_true",
                    help="analyse the reference clip instead (ground-truth targets)")
    ap.add_argument("--envs", default="all")
    ap.add_argument("--force_thresh", type=float, default=20.0,
                    help="N above which a hoof counts as bearing load. The deer weighs "
                         "~809 N, so a leg carrying real weight is >100 N; 20 N excludes "
                         "incidental brushes without missing genuine stance.")
    ap.add_argument("--no_diagram", action="store_true")
    args = ap.parse_args()

    path = REPO / "assets/motions/deer_polyart_walk.npy" if args.reference else args.input
    if path is None:
        raise SystemExit("pass --input <rollout.npz> or --reference")
    names, jp, rp, rq, fps, cf, cb = load_any(path)
    # reference joint ranges, for the conformance check (skip when analysing the ref itself)
    ref_jp = ref_names = None
    if not args.reference:
        _r = np.load(str(REPO / "assets/motions/deer_polyart_walk.npy"), allow_pickle=True).item()
        ref_jp = np.asarray(_r["joint_positions"], float); ref_names = list(_r["joints_list"])
    N = jp.shape[1]
    envs = range(N) if args.envs == "all" else [int(x) for x in str(args.envs).split(",")]
    print(f"source: {path}\nframes={len(jp)} envs={N} fps={fps:.0f}")
    print(f"targets: duty {TARGET['duty']}, liftoffs/s {TARGET['liftoff']}, "
          f"leg activity {TARGET['activity']} deg")
    for e in envs:
        if e >= N:
            continue
        report_env(f"{Path(path).stem}  env {e}", names, jp[:, e, :], rp[:, e, :], rq[:, e, :],
                   fps, cf[:, e, :] if cf is not None else None, cb,
                   args.force_thresh, not args.no_diagram, ref_jp, ref_names)


if __name__ == "__main__":
    main()
