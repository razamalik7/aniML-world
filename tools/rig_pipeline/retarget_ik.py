"""IK retargeting: solve for the joint angles that best reproduce the artist's
animation, instead of reading them off analytically.

THE CEILING. `extract_reference` computes each joint angle analytically -- the twist
of the child's rotation about ONE fitted axis. That is a first-order approximation
twice over: real bones rotate in 3-D while our joints are 1-DOF hinges, and each
hinge axis was fitted by SVD **to the walk**, so accuracy degrades away from it.
Measured FK round-trip error: **86 mm on the walk**, and 218 mm (sleep), 248 mm
(fall), 450 mm (death). For a single walk that is cosmetic. For **ASE** -- which
learns a latent skill space FROM a motion dataset -- it is a blocker, because the
embedding is bounded by the fidelity of every clip in it.

THE IDEA. The analytic extraction optimises nothing; it just reads a number. Here we
instead SOLVE, per frame, for the joint vector q minimising

    sum_bones || FK(q)_bone - bone_target ||^2

subject to joint limits, warm-started from the analytic answer. Same rig, same DOF,
same action space, same joint semantics -- we are only choosing better numbers for
the angles we already have. Error can only go down: the analytic solution is the
initial guess, so the optimiser starts at 86 mm and improves from there.

Requires `bone_positions` / `ik_offsets` in the .npy (added to extract_reference
2026-07-25). FK here mirrors the round-trip in extract_reference exactly:
  D_bone = D_parent @ Rot(axis1, q1) [@ Rot(axis2, q2)]
  p_bone = p_parent + D_parent @ rest_offset

  python tools/rig_pipeline/retarget_ik.py --input <clip_with_targets.npy>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]


def load_clip(path):
    if not hasattr(np, "_core"):          # Blender numpy 2.x -> Isaac numpy 1.x
        import numpy.core as _c
        sys.modules.setdefault("numpy._core", _c)
        for s in ("multiarray", "umath", "numeric", "_multiarray_umath"):
            m = getattr(_c, s, None)
            if m is not None:
                sys.modules.setdefault(f"numpy._core.{s}", m)
    return np.load(str(path), allow_pickle=True).item()


class Chain:
    """Minimal FK over the extracted rig, mirroring extract_reference's round-trip."""

    def __init__(self, d):
        self.jnames = list(d["joints_list"])
        self.offsets = d["ik_offsets"]
        self.axes = d["axes"]
        self.bone_names = list(d["bone_names"])
        # bones that own joints, in a parent-before-child order
        self.bones = [b for b in self.offsets]
        order, seen = [], set()

        def visit(b):
            if b in seen:
                return
            seen.add(b)
            par = self.offsets[b]["parent"]
            if par in self.offsets:
                visit(par)
            order.append(b)

        for b in self.bones:
            visit(b)
        self.order = order
        # column index of each bone's dof(s) in the joint vector
        self.col = {}
        k = 0
        for b in self.bones:
            self.col[b] = k
            k += 1 + (1 if self.offsets[b]["dof"] == 2 else 0)
        self.ndof = k
        self.tracked = [b for b in self.order if b in self.bone_names]

    def fk(self, q, root_D, root_p):
        """Return {bone: world position} for joint vector q."""
        D = {}
        P = {}
        for b in self.order:
            par = self.offsets[b]["parent"]
            Dp = D.get(par, root_D)
            Pp = P.get(par, root_p)
            c = self.col[b]
            r = Rotation.from_rotvec(np.asarray(self.axes[b]["axis1"], float) * q[c])
            if self.offsets[b]["dof"] == 2:
                r = r * Rotation.from_rotvec(np.asarray(self.axes[b]["axis2"], float) * q[c + 1])
            D[b] = Dp * r
            P[b] = Pp + Dp.apply(np.asarray(self.offsets[b]["offset"], float))
        return P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all")
    # ABLATION (2026-07-25): nfev=60 was TERMINATING ON THE CAP, not converging --
    # raising it to 400 took the walk from 63% to 69% error reduction.
    ap.add_argument("--max_nfev", type=int, default=1000)
    ap.add_argument("--root_mode", choices=["none", "rot", "full"], default="rot",
                    help="Which root DOF to solve. Measured on the walk (62.7 mm "
                         "analytic): none = joints only -> 20.09 mm; "
                         "rot = joints + root ROTATION -> 13.42 mm [DEFAULT]; "
                         "full = rot + root POSITION -> 13.27 mm. "
                         "'rot' is the default because root ORIENTATION is DERIVED (it comes "
                         "from the same analytic twist as every joint, so it carries the same "
                         "error and there is real signal to recover) while root POSITION is "
                         "MEASURED ground truth (the pelvis's actual world position), so there "
                         "is nothing to fix -- confirmed by 'full' buying only 0.15 mm. Holding "
                         "position fixed makes it STRUCTURALLY impossible for the solver to "
                         "corrupt the locomotion trajectory, rather than trusting it not to.")
    args = ap.parse_args()

    d = load_clip(args.input)
    for k in ("bone_positions", "ik_offsets", "bone_names", "axes"):
        if k not in d:
            raise SystemExit(f"{args.input} lacks '{k}' -- re-extract with the current "
                             f"extract_reference (it now dumps IK targets).")

    ch = Chain(d)
    jp = np.asarray(d["joint_positions"], float)
    bp = np.asarray(d["bone_positions"], float)
    bidx = {b: ch.bone_names.index(b) for b in ch.tracked}
    root_q = np.asarray(d["root_quaternion"], float)
    root_p = np.asarray(d["root_position"], float)
    T = len(jp) if not args.max_frames else min(args.max_frames, len(jp))
    print(f"frames={T}  dof={ch.ndof}  tracked bones={len(ch.tracked)}")

    # analytic answer is the initial guess; the solve can only improve on it
    base = next(iter(ch.offsets))
    out_q = jp.copy()
    out_rootq = root_q.copy()
    out_rootp = root_p.copy()
    err_before, err_after = [], []
    for t in range(T):
        # root orientation-from-rest, as the round-trip uses
        rootD = Rotation.from_quat(root_q[t])
        rootp = root_p[t]
        tgt = np.array([bp[t, bidx[b]] for b in ch.tracked])

        def resid_j(q, R=rootD, P0=rootp):
            P = ch.fk(q, R, P0)
            return (np.array([P[b] for b in ch.tracked]) - tgt).ravel()

        err_before.append(np.linalg.norm(resid_j(jp[t]).reshape(-1, 3), axis=1).mean())

        if args.root_mode == "none":
            sol = least_squares(resid_j, jp[t], method="lm", max_nfev=args.max_nfev)
            out_q[t] = sol.x
            err_after.append(np.linalg.norm(resid_j(sol.x).reshape(-1, 3), axis=1).mean())
        elif args.root_mode == "rot":
            z0 = np.concatenate([jp[t], rootD.as_rotvec()])

            def resid_rot(z):
                return resid_j(z[:ch.ndof], Rotation.from_rotvec(z[ch.ndof:]), rootp)

            sol = least_squares(resid_rot, z0, method="lm", max_nfev=args.max_nfev)
            out_q[t] = sol.x[:ch.ndof]
            out_rootq[t] = Rotation.from_rotvec(sol.x[ch.ndof:]).as_quat()
            # root POSITION deliberately untouched -> locomotion trajectory preserved exactly
            err_after.append(np.linalg.norm(resid_rot(sol.x).reshape(-1, 3), axis=1).mean())
        else:
            z0 = np.concatenate([jp[t], rootD.as_rotvec(), rootp])

            def resid_full(z):
                return resid_j(z[:ch.ndof],
                               Rotation.from_rotvec(z[ch.ndof:ch.ndof + 3]),
                               z[ch.ndof + 3:])

            sol = least_squares(resid_full, z0, method="lm", max_nfev=args.max_nfev)
            out_q[t] = sol.x[:ch.ndof]
            out_rootq[t] = Rotation.from_rotvec(sol.x[ch.ndof:ch.ndof + 3]).as_quat()
            out_rootp[t] = sol.x[ch.ndof + 3:]
            err_after.append(np.linalg.norm(resid_full(sol.x).reshape(-1, 3), axis=1).mean())
        if t % 20 == 0:
            print(f"  frame {t:>4}: {err_before[-1]*1000:7.1f} mm -> {err_after[-1]*1000:7.1f} mm")

    # MULTI-TURN UNWRAP. Rot(axis, t) == Rot(axis, t + 2pi), so least_squares is free
    # to return a numerically absurd branch for an identical pose. Measured on the
    # first full-library pass: the raw analytic angles peak at 167 deg while the IK
    # output reached 957 deg (Death), 595 (Fall), 400 (Attack). Those are pose-correct
    # and everything-else-wrong: they blow out any limit fitted from them, and the AMP
    # discriminator sees 443 deg and 83 deg as wildly different states.
    # Snap each angle to the 2pi branch nearest the ANALYTIC warm start -- that branch
    # is the physically sensible one, and shifting by a whole multiple of 2pi leaves
    # the pose (and therefore err_after) exactly unchanged.
    turns = np.round((out_q[:T] - jp[:T]) / (2 * np.pi))
    n_wrapped = int((turns != 0).sum())
    if n_wrapped:
        out_q[:T] -= 2 * np.pi * turns
        print(f"unwrapped {n_wrapped} multi-turn samples "
              f"(max |angle| {np.degrees(np.abs(out_q[:T])).max():.1f} deg after)")

    b, a = np.mean(err_before) * 1000, np.mean(err_after) * 1000
    print(f"\nMEAN bone position error: {b:.1f} mm  ->  {a:.1f} mm   "
          f"({100*(1-a/max(b,1e-9)):.0f}% reduction)")
    dq = np.degrees(np.abs(out_q[:T] - jp[:T]))
    print(f"joint angles moved: mean {dq.mean():.2f} deg, max {dq.max():.2f} deg")

    if args.output:
        d2 = dict(d)
        d2["joint_positions"] = out_q.astype(np.float32)
        if args.root_mode != "none":
            d2["root_quaternion"] = out_rootq.astype(np.float32)
        if args.root_mode == "full":
            d2["root_position"] = out_rootp.astype(np.float32)
        d2["ik_retargeted"] = True
        np.save(args.output, d2, allow_pickle=True)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
