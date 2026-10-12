"""How much does IK retargeting help ACROSS THE WHOLE ANIMATION LIBRARY?

The walk was always our best clip (86 mm). The clips that actually gate ASE are the
others -- sleep 218 mm, fall 248 mm, death 450 mm -- because ASE learns a latent skill
space FROM a motion dataset and is bounded by the fidelity of every clip in it. A fix
that only helps the walk fixes nothing that was broken.

Runs the same least-squares IK as `retarget_ik.py` on a sample of frames from every
clip and reports analytic-vs-IK error side by side.

  python tools/rig_pipeline/eval_ik_all.py --dir <dir of clips with bone targets>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.rig_pipeline.retarget_ik import Chain, load_clip  # noqa: E402


def eval_clip(path, n_frames, max_nfev):
    d = load_clip(path)
    if "bone_positions" not in d:
        return None
    ch = Chain(d)
    jp = np.asarray(d["joint_positions"], float)
    bp = np.asarray(d["bone_positions"], float)
    rq = np.asarray(d["root_quaternion"], float)
    rp = np.asarray(d["root_position"], float)
    bidx = {b: ch.bone_names.index(b) for b in ch.tracked}
    T = len(jp)
    idx = np.linspace(0, T - 1, min(n_frames, T)).astype(int)

    before, after = [], []
    for t in idx:
        rootD = Rotation.from_quat(rq[t]); rootp = rp[t]
        tgt = np.array([bp[t, bidx[b]] for b in ch.tracked])

        def resid(q):
            P = ch.fk(q, rootD, rootp)
            return (np.array([P[b] for b in ch.tracked]) - tgt).ravel()

        before.append(np.linalg.norm(resid(jp[t]).reshape(-1, 3), axis=1).mean())
        # solve the ROOT jointly -- root error propagates to every bone
        z0 = np.concatenate([jp[t], rootD.as_rotvec(), rootp])

        def resid_full(z):
            P = ch.fk(z[:ch.ndof], Rotation.from_rotvec(z[ch.ndof:ch.ndof + 3]), z[ch.ndof + 3:])
            return (np.array([P[b] for b in ch.tracked]) - tgt).ravel()

        sol = least_squares(resid_full, z0, method="lm", max_nfev=max_nfev)
        after.append(np.linalg.norm(resid_full(sol.x).reshape(-1, 3), axis=1).mean())
    return T, np.mean(before) * 1000, np.mean(after) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--frames", type=int, default=40, help="frames sampled per clip")
    ap.add_argument("--max_nfev", type=int, default=400)
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.npy"))
    print(f"{'clip':<22}{'frames':>7}{'analytic':>11}{'IK':>10}{'reduction':>11}   usable for ASE?")
    rows = []
    for f in files:
        try:
            r = eval_clip(f, args.frames, args.max_nfev)
        except Exception as e:
            print(f"{f.stem:<22}  FAILED: {type(e).__name__}: {e}")
            continue
        if r is None:
            print(f"{f.stem:<22}  (no bone targets -- re-extract)")
            continue
        T, b, a = r
        red = 100 * (1 - a / max(b, 1e-9))
        # <100 mm on a ~1.5 m animal is roughly where reconstruction stops dominating
        verdict = ("yes" if a < 100 else ("marginal" if a < 150 else "NO"))
        was = ("yes" if b < 100 else ("marginal" if b < 150 else "NO"))
        print(f"{f.stem:<22}{T:>7}{b:>9.1f}mm{a:>8.1f}mm{red:>10.0f}%   {was} -> {verdict}")
        rows.append((f.stem, b, a))
    if rows:
        bb = np.mean([r[1] for r in rows]); aa = np.mean([r[2] for r in rows])
        print(f"\n{'MEAN across clips':<22}{'':>7}{bb:>9.1f}mm{aa:>8.1f}mm"
              f"{100*(1-aa/max(bb,1e-9)):>10.0f}%")
        print(f"\nclips usable as ASE skill data (<100 mm): "
              f"{sum(1 for r in rows if r[2] < 100)}/{len(rows)}  "
              f"(was {sum(1 for r in rows if r[1] < 100)}/{len(rows)})")


if __name__ == "__main__":
    main()
