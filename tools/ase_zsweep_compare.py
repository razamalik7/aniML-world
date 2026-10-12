"""Quantify how DIFFERENT an ASE latent's skills actually are.

`ase_z_sweep.py` pins a distinct fixed z per env and records the rollout, but it does
not score anything -- the "22-36 degrees" quoted for exp4 was computed ad hoc. This
makes it repeatable, and fixes a trap in the obvious version of the measurement.

THE TRAP. The naive score is mean pairwise joint-angle distance between envs. Two envs
running the SAME gait at different points in the stride are far apart by that measure,
so a policy that ignores z entirely still scores well merely because the envs fell out
of phase. Every number here is therefore reported twice:

  raw    mean pairwise joint distance, no alignment
  phase  the same after searching over time shifts for the best alignment

`phase` is the honest one. raw >> phase means the envs differ mostly in WHEN, not WHAT.

Gait statistics are reported alongside because they are phase-invariant by construction
and answer a different question -- not "do these poses differ" but "do these move
differently": forward speed, stride frequency, and per-leg duty.

    python tools/ase_zsweep_compare.py --runs C3 F1
    python tools/ase_zsweep_compare.py --runs exp4 C3 C3b F1
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HOOVES = ["L_Toe0", "R_Toe0", "L_Finger0", "R_Finger0"]
CONTACT_N = 20.0


def phase_aligned_distance(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[float, float]:
    """(raw, best-aligned) mean per-joint distance in degrees between two trajectories."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    raw = float(np.abs(a - b).mean())
    best = raw
    for s in range(1, max_shift + 1):
        for x, y in ((a[s:], b[:n - s]), (a[:n - s], b[s:])):
            d = float(np.abs(x - y).mean())
            if d < best:
                best = d
    return raw, best


def stride_hz(contact: np.ndarray, fps: float) -> float:
    """Footfalls per second from contact on/off transitions."""
    down = contact > CONTACT_N
    trans = int(np.sum(down[1:] & ~down[:-1]))
    return trans / (len(down) / fps)


def analyse(tag: str) -> dict | None:
    p = REPO / "docs" / f"{tag}_zsweep.npz"
    if not p.exists():
        print(f"\n{tag}: no z-sweep at {p.name} -- skipped")
        return None
    d = np.load(str(p), allow_pickle=True)
    q = np.degrees(np.asarray(d["joint_positions"], float))   # (T, E, J)
    rp = np.asarray(d["root_pos"], float)
    fps = float(d["fps"])
    bodies = [str(x) for x in d["contact_bodies"]] if "contact_bodies" in d.files else []
    cf = np.abs(np.asarray(d["contact_forces"], float)) if "contact_forces" in d.files else None
    T, E, _ = q.shape

    # Skip the first second: every env starts from the same RSI pose, so early frames
    # are similar no matter what z does, and including them dilutes every pair.
    s0 = min(int(fps), T // 4)
    qq = q[s0:]
    max_shift = min(int(fps * 1.5), (T - s0) // 3)   # up to ~1.5 s of phase search

    raws, aligns = [], []
    for i, j in combinations(range(E), 2):
        r, a = phase_aligned_distance(qq[:, i], qq[:, j], max_shift)
        raws.append(r); aligns.append(a)

    speeds = [float(np.linalg.norm(np.diff(rp[s0:, e, :2], axis=0), axis=1).mean() * fps)
              for e in range(E)]
    gait = {}
    if cf is not None and all(h in bodies for h in HOOVES):
        for e in range(E):
            hz = np.mean([stride_hz(cf[s0:, e, bodies.index(h)], fps) for h in HOOVES])
            duty = np.mean([(cf[s0:, e, bodies.index(h)] > CONTACT_N).mean() for h in HOOVES])
            gait[e] = (hz, duty)

    print(f"\n=== {tag} ({E} latents, {T} frames @ {fps:.0f} fps) ===")
    print(f"  pose separation   raw {np.mean(raws):5.1f}d   phase-aligned "
          f"{np.mean(aligns):5.1f}d   (min pair {np.min(aligns):5.1f}d)")
    frac = np.mean(aligns) / max(np.mean(raws), 1e-6)
    print(f"  {100*frac:.0f}% of the raw separation survives phase alignment"
          + ("  <- mostly PHASE, not skill" if frac < 0.5 else ""))
    print(f"  forward speed     {min(speeds):.2f} - {max(speeds):.2f} m/s "
          f"(spread {max(speeds)-min(speeds):.2f})")
    if gait:
        hzs = [gait[e][0] for e in gait]; dts = [gait[e][1] for e in gait]
        print(f"  stride rate       {min(hzs):.2f} - {max(hzs):.2f} Hz   "
              f"duty {min(dts):.2f} - {max(dts):.2f}")
    return {"tag": tag, "raw": float(np.mean(raws)), "aligned": float(np.mean(aligns)),
            "min": float(np.min(aligns)), "speed_spread": max(speeds) - min(speeds)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["C3", "F1"])
    a = ap.parse_args()
    rows = [r for r in (analyse(t) for t in a.runs) if r]
    if len(rows) > 1:
        print("\n" + "=" * 70)
        print(f"{'run':10s} {'aligned':>9s} {'min pair':>9s} {'spd spread':>11s}")
        for r in sorted(rows, key=lambda x: -x["aligned"]):
            print(f"{r['tag']:10s} {r['aligned']:8.1f}d {r['min']:8.1f}d "
                  f"{r['speed_spread']:10.2f}")
        print("\nHigher phase-aligned separation = genuinely distinct skills.")
        print("High MI with LOW separation is the failure the diversity objective "
              "exists to prevent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
