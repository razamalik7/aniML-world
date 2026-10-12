"""Synthesise variety from ONE motion clip, to fight AMP discriminator saturation.

THE PROBLEM (measured, r16p1 via tools/amp_status.py). ``amp_loss`` peaked at 0.6449 --
near the ln(2)=0.6931 equilibrium where the discriminator genuinely cannot separate
policy from expert -- then fell to **0.2985**, while style reward tracked it down
0.7507 -> 0.2963. The discriminator had memorised the expert set and started winning,
so the style gradient faded. It saturated even under the LEGACY normaliser, which
handicaps it by hiding policy outliers in inflated variance.

The cause is SIZE, not quality: the expert set is **33 frames / 1.1 s / one stride**
against ~98,000 policy timesteps per iteration. Three separate fixes improved the
reference's quality (dead air, turn contamination, IK retargeting); none touched its
quantity.

This is the cheap axis: no new capture, no new rig, no ASE. Three augmentations,
each defensible on physical grounds rather than "add noise and hope":

  --speeds     TIME-WARP. Resample the clip at k*fps. Joint POSITIONS trace the same
               manifold, but joint and base VELOCITIES scale by k -- and velocities
               are 41 of the 82 AMP observation dimensions, so a 0.8x and a 1.2x walk
               are genuinely distinct points in discriminator space. Keep |k-1| small:
               a 2x walk is not a run (real quadrupeds change GAIT, not tempo), so
               aggressive warping manufactures references no policy can physically
               track. +/-20-30% is the defensible band.

  --phases     PHASE ROLL. Cyclically roll the clip so it starts at a different point
               in the stride. Adds no new poses, but a discriminator that memorised
               "frame 0 looks like THIS" can no longer key on absolute phase.

  --jitter     JOINT JITTER, in degrees, temporally SMOOTHED. Unsmoothed per-frame
               noise is the exact defect IK retargeting was built to remove (the
               analytic extraction read each frame independently and produced visible
               tremor), so raw noise would teach the discriminator that tremor is
               deer-like. Smoothing keeps the perturbation on a plausible trajectory.

Writes one .npy per variant, ready to list in ANIML_AMP_DATASETS.

  python tools/rig_pipeline/augment_reference.py \
      --input assets/motions/deer_polyart_walk.npy \
      --outdir assets/motions --tag walkaug \
      --speeds 0.8,0.9,1.1,1.2 --phases 4 --jitter 0.75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.rig_pipeline.retarget_ik import load_clip  # noqa: E402


def _resample(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Linear resample along axis 0 to n_out samples."""
    n_in = len(arr)
    src = np.linspace(0.0, n_in - 1, n_out)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (src - lo).reshape(-1, *([1] * (arr.ndim - 1)))
    return arr[lo] * (1.0 - w) + arr[hi] * w


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win < 2:
        return x
    k = np.ones(win) / win
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(x[:, j], k, mode="same")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default=str(REPO / "assets" / "motions"))
    ap.add_argument("--tag", required=True, help="output prefix, e.g. walkaug")
    ap.add_argument("--speeds", default="0.8,0.9,1.1,1.2",
                    help="time-warp factors; 1.0 (the original) is always emitted")
    ap.add_argument("--phases", type=int, default=1,
                    help="number of cyclic phase rolls per speed (1 = no roll)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="smoothed joint jitter in DEGREES (0 = off)")
    ap.add_argument("--jitter_smooth", type=int, default=5, help="moving-average window")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = load_clip(args.input)
    jp = np.asarray(d["joint_positions"], float)
    rp = np.asarray(d["root_position"], float)
    rq = np.asarray(d["root_quaternion"], float)
    fps = float(np.asarray(d.get("fps", 30.0)).item())
    T = len(jp)
    jn = [str(x) for x in d["joints_list"]]
    print(f"input {Path(args.input).name}: {T} frames @ {fps:.1f} fps ({T/fps:.2f} s)")

    speeds = sorted({1.0} | {float(s) for s in args.speeds.split(",") if s.strip()})
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int]] = []

    for k in speeds:
        n_out = max(4, int(round(T / k)))
        jp_k, rp_k, rq_k = _resample(jp, n_out), _resample(rp, n_out), _resample(rq, n_out)
        # renormalise quaternions after linear interpolation
        rq_k = rq_k / np.linalg.norm(rq_k, axis=1, keepdims=True).clip(min=1e-9)
        for p in range(max(1, args.phases)):
            sh = int(round(p * n_out / max(1, args.phases)))
            jp_p = np.roll(jp_k, sh, axis=0)
            rq_p = np.roll(rq_k, sh, axis=0)
            # root POSITION must stay monotonic along travel -- rolling it would
            # teleport the deer mid-clip. Re-integrate from the rolled per-frame delta.
            step = np.diff(rp_k, axis=0, prepend=rp_k[:1])
            rp_p = np.cumsum(np.roll(step, sh, axis=0), axis=0) + rp_k[0]
            if args.jitter > 0.0:
                noise = rng.normal(0.0, np.radians(args.jitter), size=jp_p.shape)
                jp_p = jp_p + _smooth(noise, args.jitter_smooth)
            name = f"{args.tag}_s{k:g}".replace(".", "p")
            if args.phases > 1:
                name += f"_p{p}"
            np.save(outdir / f"{name}.npy",
                    dict(joint_positions=jp_p.astype(np.float32),
                         root_position=rp_p.astype(np.float32),
                         root_quaternion=rq_p.astype(np.float32),
                         # joints_list must be a PYTHON LIST, not an ndarray:
                         # AMPLoader.load_data calls dataset_joint_names.index(j) to
                         # build its column map, and ndarray has no .index ->
                         #   AttributeError: 'numpy.ndarray' object has no attribute 'index'
                         # The original clips store a list; matching that is required.
                         joints_list=list(jn), fps=np.float32(fps)),
                    allow_pickle=True)
            written.append((name, len(jp_p)))

    total = sum(n for _, n in written)
    print(f"\nwrote {len(written)} variants, {total} frames total "
          f"({total/max(T,1):.1f}x the original {T})")
    for n, f in written:
        print(f"   {n:<28} {f:4d} frames")
    print("\nANIML_AMP_DATASETS=\"" + ",".join(f"{n}:1.0" for n, _ in written) + "\"")


if __name__ == "__main__":
    main()
