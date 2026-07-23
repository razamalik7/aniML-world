"""Fit the deer URDF's joint angles and root pose to 2D keypoints.

    python tools/cv/fit_urdf_to_keypoints.py --selftest
    python tools/cv/fit_urdf_to_keypoints.py --selftest --noise-px 3 --trials 20

This is stage [C] of the pipeline in `docs/CV_PIPELINE_DESIGN.md`, and the one
piece of real algorithmic risk left. Everything upstream (2D keypoints) and
downstream (clip assembly) is comparatively mechanical.

WHY FIT THE RIG DIRECTLY, RATHER THAN LIFT 3D POINTS AND RETARGET
-----------------------------------------------------------------
Because it makes three error modes structurally impossible instead of merely
small:

  * **bone lengths** cannot drift -- they are the URDF's, fixed;
  * **scale** is not a free parameter -- the rig is metric;
  * **depth ambiguity** is heavily constrained -- limb foreshortening against
    known segment lengths pins root depth far better than any generic monocular
    depth model.

The literature is explicit that naive keypoint transfer breaks contact schedules
and produces foot sliding. Solving in joint space sidesteps that entire class.

WHAT IS BEING SOLVED
--------------------
Unknowns per frame: root translation (3) + root rotation (3, axis-angle) +
joint angles (25) = 31. Observations: up to 31 visible 2D keypoints = 62
residuals. So the problem is over-determined when enough keypoints are visible,
and the self-test measures how it degrades as visibility drops.

Residuals:
  * reprojection error, in pixels, for visible keypoints only;
  * a soft joint-limit barrier (quadratic outside the URDF range) -- a hard clamp
    would kill the gradient exactly where the optimiser needs it;
  * optional temporal smoothness across a sequence, which is where most of the
    robustness against per-frame noise comes from.

THE SELF-TEST IS THE POINT
--------------------------
`--selftest` samples a random plausible pose, renders it to 2D through a known
camera, optionally adds pixel noise, then fits and reports the recovered joint
angle error. Because ground truth is known exactly, this measures the fitter
rather than trusting it -- and it converts "how accurate must the 2D pose net be"
from a guess into a number, on our rig, without needing any real footage or GPU.

Read the result against the accuracy bar in `docs/CV_PIPELINE_DESIGN.md` §1:
joint-angle error should stay under ~1 degree for the reference to be usable.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.optimize import least_squares

from deer_rig import DeerRig
from keypoint_schema import DeerKeypoints
from render import Camera


def axis_angle_to_R(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def R_to_axis_angle(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    theta = math.acos(c)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(math.pi - theta) < 1e-6:      # near-180 degrees; use the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        if k[np.argmax(k)] > 0:
            k = k / np.linalg.norm(k)
        return k * theta
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return k / (2 * math.sin(theta)) * theta


class UrdfFitter:
    """Least-squares fit of root pose + joint angles to 2D keypoints."""

    def __init__(self, rig: DeerRig | None = None,
                 kps: DeerKeypoints | None = None,
                 limit_weight: float = 50.0):
        self.rig = rig or DeerRig()
        self.kps = kps or DeerKeypoints(self.rig)
        self.dof = list(self.rig.dof_names)
        self.n_dof = len(self.dof)
        lo, hi = self.rig.joint_limits() if hasattr(self.rig, "joint_limits") \
            else (None, None)
        if lo is None:
            lo = np.array([self.rig.joints[n].lower for n in self.dof])
            hi = np.array([self.rig.joints[n].upper for n in self.dof])
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.limit_weight = limit_weight

    # -- parameter packing -------------------------------------------------

    def pack(self, t: np.ndarray, rvec: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.concatenate([t, rvec, q])

    def unpack(self, x: np.ndarray):
        return x[:3], x[3:6], x[6:]

    def keypoints_3d(self, t, rvec, q) -> np.ndarray:
        R = axis_angle_to_R(rvec)
        local = self.kps.positions(dict(zip(self.dof, q)))
        return local @ R.T + t

    def project(self, cam: Camera, pts3d: np.ndarray) -> np.ndarray:
        cam_pts = (pts3d - cam.C) @ cam.R_cw.T
        z = np.clip(cam_pts[:, 2], 1e-6, None)
        u = cam.fx * cam_pts[:, 0] / z + cam.cx
        v = cam.fy * cam_pts[:, 1] / z + cam.cy
        return np.stack([u, v], axis=1)

    # -- solve -------------------------------------------------------------

    def residuals(self, x, cam, target2d, weights):
        t, rvec, q = self.unpack(x)
        proj = self.project(cam, self.keypoints_3d(t, rvec, q))
        r = ((proj - target2d) * weights[:, None]).reshape(-1)
        below = np.clip(self.lo - q, 0, None)
        above = np.clip(q - self.hi, 0, None)
        return np.concatenate([r, self.limit_weight * (below + above)])

    def fit(self, cam: Camera, target2d: np.ndarray, visible: np.ndarray,
            init=None, max_nfev: int = 200):
        """Fit one frame. `visible` is a boolean mask over keypoints."""
        w = visible.astype(np.float64)
        if init is None:
            # Start at the rest pose, placed at the centroid's back-projection.
            t0 = np.zeros(3)
            r0 = np.zeros(3)
            q0 = np.clip(np.zeros(self.n_dof), self.lo, self.hi)
            # Crude depth init: put the animal in front of the camera at a
            # distance that makes its projected size roughly match.
            rest = self.kps.positions(dict(zip(self.dof, q0)))
            span3d = float(np.linalg.norm(rest.max(0) - rest.min(0)))
            vis_pts = target2d[visible]
            span2d = float(np.linalg.norm(vis_pts.max(0) - vis_pts.min(0))) \
                if len(vis_pts) > 1 else cam.width * 0.5
            depth = cam.fx * span3d / max(span2d, 1.0)
            centre_px = vis_pts.mean(0) if len(vis_pts) else np.array(
                [cam.cx, cam.cy])
            ray = np.array([(centre_px[0] - cam.cx) / cam.fx,
                            (centre_px[1] - cam.cy) / cam.fy, 1.0])
            t0 = cam.C + (cam.R_cw.T @ ray) * depth - rest.mean(0)
            init = self.pack(t0, r0, q0)

        sol = least_squares(
            self.residuals, init, args=(cam, target2d, w),
            method="trf", max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
        t, rvec, q = self.unpack(sol.x)
        proj = self.project(cam, self.keypoints_3d(t, rvec, q))
        err = np.linalg.norm(proj - target2d, axis=1)
        return {
            "x": sol.x, "t": t, "rvec": rvec, "q": q,
            "reproj_px_mean": float(err[visible].mean()) if visible.any()
            else float("nan"),
            "reproj_px_max": float(err[visible].max()) if visible.any()
            else float("nan"),
            "success": bool(sol.success), "nfev": int(sol.nfev),
        }


    # -- sequence solve ----------------------------------------------------

    def fit_sequence(self, cam, target2d_seq, visible_seq,
                     smooth_joint: float = 40.0, smooth_root: float = 40.0,
                     max_nfev: int = 400):
        """Fit a whole sequence jointly with a temporal smoothness prior.

        Per-frame fitting is badly under-constrained in practice (see the
        self-test: ~4 deg median joint error at only 1 px of keypoint noise,
        against a ~1 deg budget). Real motion is smooth, and a first-difference
        penalty injects that as prior information, which is where most of the
        noise robustness has to come from.

        `smooth_joint` / `smooth_root` weight the first difference of joint
        angles and root parameters respectively. Too high and genuine fast
        motion gets flattened -- which would destroy exactly the within-stride
        detail the discriminator cares about -- so they are exposed, not baked in.
        """
        T = len(target2d_seq)
        n = 6 + self.n_dof
        # Independent per-frame fits make the initial guess.
        init = np.zeros((T, n))
        prev = None
        for t in range(T):
            out = self.fit(cam, target2d_seq[t], visible_seq[t], init=prev,
                           max_nfev=120)
            init[t] = out["x"]
            prev = out["x"]

        def residuals_seq(xf):
            X = xf.reshape(T, n)
            parts = []
            for t in range(T):
                parts.append(self.residuals(X[t], cam, target2d_seq[t],
                                            visible_seq[t].astype(float)))
            d = np.diff(X, axis=0)
            parts.append((smooth_root * d[:, :6]).reshape(-1))
            parts.append((smooth_joint * d[:, 6:]).reshape(-1))
            return np.concatenate(parts)

        sol = least_squares(residuals_seq, init.reshape(-1), method="trf",
                            max_nfev=max_nfev, xtol=1e-8, ftol=1e-8)
        X = sol.x.reshape(T, n)
        return {"x": X, "q": X[:, 6:], "t": X[:, :3], "rvec": X[:, 3:6],
                "success": bool(sol.success)}


# ---------------------------------------------------------------------------
# Self-test: synthetic round trip with known ground truth
# ---------------------------------------------------------------------------

def sample_pose(fitter: UrdfFitter, rng: np.random.Generator,
                frac: float = 0.35) -> np.ndarray:
    """A random pose within a fraction of each joint's range, centred on 0."""
    mid = np.zeros(fitter.n_dof)
    half = frac * (fitter.hi - fitter.lo) / 2.0
    return np.clip(mid + rng.uniform(-half, half), fitter.lo, fitter.hi)


def selftest(args) -> int:
    rng = np.random.default_rng(args.seed)
    fitter = UrdfFitter()
    n_kp = len(fitter.kps)
    print(f"rig: {fitter.n_dof} DOF, {n_kp} keypoints "
          f"({2 * n_kp} residuals when all visible)")
    print(f"trials {args.trials}, pixel noise sigma {args.noise_px}, "
          f"visible fraction {args.visible_frac}")
    print()

    joint_errs, root_errs, reproj, fails = [], [], [], 0
    for trial in range(args.trials):
        q_true = sample_pose(fitter, rng)
        t_true = np.array([0.0, 0.0, 0.0])
        r_true = np.array([0.0, 0.0, rng.uniform(-math.pi, math.pi)])

        pts3d = fitter.keypoints_3d(t_true, r_true, q_true)
        centre = pts3d.mean(0)
        radius = float(np.linalg.norm(pts3d.max(0) - pts3d.min(0))) / 2
        azim = rng.uniform(0, 2 * math.pi)
        elev = math.radians(rng.uniform(5, 40))
        dist = radius * rng.uniform(3.0, 6.0)
        eye = centre + np.array([
            dist * math.cos(elev) * math.cos(azim),
            dist * math.cos(elev) * math.sin(azim),
            dist * math.sin(elev)])
        cam = Camera.look_at(eye, centre, width=640, height=480, hfov_deg=50.0)

        target = fitter.project(cam, pts3d)
        if args.noise_px > 0:
            target = target + rng.normal(0, args.noise_px, target.shape)
        visible = rng.random(n_kp) < args.visible_frac
        if visible.sum() < 8:
            visible[rng.choice(n_kp, 8, replace=False)] = True

        out = fitter.fit(cam, target, visible, max_nfev=args.max_nfev)
        if not np.isfinite(out["reproj_px_mean"]):
            fails += 1
            continue
        jerr = np.degrees(np.abs(out["q"] - q_true))
        joint_errs.append(jerr)
        root_errs.append(float(np.linalg.norm(out["t"] - t_true)))
        reproj.append(out["reproj_px_mean"])

    if not joint_errs:
        print("all trials failed")
        return 1
    J = np.stack(joint_errs)
    print(f"{'metric':<34}{'median':>10}{'mean':>10}{'p90':>10}")
    print("-" * 64)
    print(f"{'joint angle error (deg)':<34}{np.median(J):10.3f}"
          f"{J.mean():10.3f}{np.percentile(J, 90):10.3f}")
    print(f"{'worst joint per trial (deg)':<34}"
          f"{np.median(J.max(1)):10.3f}{J.max(1).mean():10.3f}"
          f"{np.percentile(J.max(1), 90):10.3f}")
    print(f"{'root position error (m)':<34}{np.median(root_errs):10.4f}"
          f"{np.mean(root_errs):10.4f}{np.percentile(root_errs, 90):10.4f}")
    print(f"{'reprojection error (px)':<34}{np.median(reproj):10.3f}"
          f"{np.mean(reproj):10.3f}{np.percentile(reproj, 90):10.3f}")
    if fails:
        print(f"\n{fails}/{args.trials} trials failed outright")

    # Per-joint breakdown. The aggregate hides the thing that matters: with a
    # single view some DOFs are simply not observable, and they must be found
    # and given a prior rather than left for the optimiser to wander in.
    print()
    print(f"{'joint':<28}{'median deg':>12}{'p90 deg':>10}")
    print("-" * 50)
    order = np.argsort(-np.median(J, axis=0))
    for i in order:
        m, p = float(np.median(J[:, i])), float(np.percentile(J[:, i], 90))
        flag = "   <-- weakly observable" if m > 2.0 else ""
        print(f"{fitter.dof[i]:<28}{m:12.3f}{p:10.3f}{flag}")

    bar = 1.0
    med = float(np.median(J))
    print()
    if args.noise_px == 0:
        print(f"With noiseless 2D input the fit should be near-exact. "
              f"Median joint error {med:.3f} deg.")
        if med > 0.1:
            print("  ^ that is higher than a clean round trip should give; the "
                  "solver or the parameterisation is leaving error on the table.")
    else:
        print(f"At {args.noise_px:g} px keypoint noise the median joint error is "
              f"{med:.3f} deg, against the ~{bar:g} deg bar in "
              f"docs/CV_PIPELINE_DESIGN.md section 1.")
        print("  Scale this to real detector accuracy: a 2D pose net's error is "
              "usually quoted as PCK, so convert via the bbox size before "
              "comparing.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--visible-frac", type=float, default=0.85)
    ap.add_argument("--max-nfev", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest(args))
    ap.print_help()


if __name__ == "__main__":
    main()
