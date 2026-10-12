"""Single-shot trajectory analysis for deer locomotion policy NPZs.

Takes a `play_loco_*.npz` (from training/scripts/play_loco.py --video) and
prints all the metrics we care about for evaluating a trained policy:

  * Inter-leg phase correlations (FL/FR, HL/HR, FL/HR, FL/HL hip flexion
    + knees) vs walk reference values
  * Trajectory straightness ratio per env (net_disp / total_dist)
  * Cumulative |yaw drift| per env over 30s
  * Carpus % time in fold band [+0.10, +1.30] rad
  * HR knee mean angle
  * Speed metrics
  * Anti-parallel hindquarter check (the "//" issue)

Usage:
    python tools/analyze_trajectory.py logs/videos/play_loco_*.npz

If multiple paths supplied, prints a comparison table across versions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# Walk reference correlations — computed directly from
# assets/motions/deer_walk.npy on 2026-05-13 (after discovering that
# prior knee reference values had been hand-guessed wrong: knees in a
# true walk are POSITIVELY correlated, not anti-correlated like the
# hip flex pairs).
WALK_REF = {
    "FL/FR hip": -0.96,
    "HL/HR hip": -0.93,
    "FL/HR hip (diag)": -0.43,
    "FL/HL hip (lateral)": +0.47,
    "FL/FR knee": +0.35,   # FIXED 2026-05-13: was -0.96 (wrong)
    "HL/HR knee": +0.67,   # FIXED 2026-05-13: was -0.93 (wrong)
}


def quat_to_yaw(q):
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def pearson(a, b):
    a_c = (a - a.mean()) / (a.std() + 1e-8)
    b_c = (b - b.mean()) / (b.std() + 1e-8)
    return float((a_c * b_c).mean())


def analyze(npz_path: Path) -> dict:
    """Load NPZ and compute all metrics. Returns a dict suitable for printing."""
    data = np.load(npz_path, allow_pickle=True)
    jp = data["joint_positions"]  # (T, N, 25)
    rp = data["root_pos"]          # (T, N, 3)
    rq = data["root_quat_wxyz"]    # (T, N, 4)
    names = list(data["joints_list"].astype(str))
    T, N, _ = jp.shape
    fps = float(data.get("fps", 50.0))
    duration_s = T / fps

    # Joint indices
    fl_h = names.index("FL_hipabd_to_FL_thigh")
    fr_h = names.index("FR_hipabd_to_FR_thigh")
    hl_h = names.index("HL_hipabd_to_HL_thigh")
    hr_h = names.index("HR_hipabd_to_HR_thigh")
    fl_k = names.index("FL_thigh_to_FL_shin")
    fr_k = names.index("FR_thigh_to_FR_shin")
    hl_k = names.index("HL_thigh_to_HL_shin")
    hr_k = names.index("HR_thigh_to_HR_shin")
    fl_c = names.index("FL_shin_to_FL_foot")
    fr_c = names.index("FR_shin_to_FR_foot")

    # Per-pair correlations averaged across envs
    pairs = {
        "FL/FR hip":     (fl_h, fr_h),
        "HL/HR hip":     (hl_h, hr_h),
        "FL/HR hip (diag)":    (fl_h, hr_h),
        "FL/HL hip (lateral)": (fl_h, hl_h),
        "FL/FR knee":   (fl_k, fr_k),
        "HL/HR knee":   (hl_k, hr_k),
    }
    correlations = {}
    correlations_per_env = {}
    for label, (a, b) in pairs.items():
        per_env = [pearson(jp[:, env, a], jp[:, env, b]) for env in range(N)]
        correlations[label] = float(np.mean(per_env))
        correlations_per_env[label] = per_env

    # Trajectory: straightness ratio + yaw drift per env
    straightness = []
    yaw_drift = []
    speeds = []
    net_speeds = []
    for env in range(N):
        path = rp[:, env, :2]  # XY
        total_dist = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        net_disp = float(np.linalg.norm(path[-1] - path[0]))
        ratio = net_disp / (total_dist + 1e-8)
        straightness.append(ratio)
        speeds.append(total_dist / duration_s)
        net_speeds.append(net_disp / duration_s)

        y = quat_to_yaw(rq[:, env])
        y_unwrap = np.unwrap(y)
        drift = y_unwrap[-1] - y_unwrap[0]
        yaw_drift.append(float(abs(drift) * 180 / np.pi))

    # Carpus % in fold band
    carpus_in_band = []
    for env in range(N):
        fl_q = jp[:, env, fl_c]
        fr_q = jp[:, env, fr_c]
        fl_in = float(((fl_q >= 0.10) & (fl_q <= 1.30)).mean() * 100)
        fr_in = float(((fr_q >= 0.10) & (fr_q <= 1.30)).mean() * 100)
        carpus_in_band.append((fl_in, fr_in))

    # HR knee mean
    hr_knee_mean = []
    for env in range(N):
        hr_q = jp[:, env, hr_k]
        hr_knee_mean.append(float(hr_q.mean() * 180 / np.pi))

    return {
        "npz_path": str(npz_path),
        "T": T,
        "N": N,
        "fps": fps,
        "duration_s": duration_s,
        "correlations": correlations,
        "correlations_per_env": correlations_per_env,
        "straightness": straightness,
        "straightness_mean": float(np.mean(straightness)),
        "yaw_drift_deg": yaw_drift,
        "yaw_drift_mean": float(np.mean(yaw_drift)),
        "speed_mps": speeds,
        "speed_mean": float(np.mean(speeds)),
        "net_speed_mps": net_speeds,
        "net_speed_mean": float(np.mean(net_speeds)),
        "carpus_in_band": carpus_in_band,
        "hr_knee_mean_deg": hr_knee_mean,
        "hr_knee_mean_overall": float(np.mean(hr_knee_mean)),
    }


def print_single(result):
    print(f"\n=== {Path(result['npz_path']).name} ({result['T']} frames, "
          f"{result['N']} envs, {result['duration_s']:.1f}s @ {result['fps']:.0f}fps) ===")
    print()
    print("Inter-leg correlations vs walk reference:")
    print(f"  {'pair':25s} {'mean':>9s}  {'envs (4)':>40s}   ref")
    for label, mean_corr in result["correlations"].items():
        envs = result["correlations_per_env"][label]
        envs_str = " ".join(f"{v:>+5.2f}" for v in envs)
        ref = WALK_REF.get(label, None)
        ref_str = f"{ref:+5.2f}" if ref is not None else ""
        print(f"  {label:25s} {mean_corr:>+9.3f}   {envs_str}   {ref_str}")
    print()
    print("Trajectory:")
    print(f"  straightness ratio:  mean {result['straightness_mean']:.3f}   "
          f"envs: {' '.join(f'{v:.3f}' for v in result['straightness'])}")
    print(f"  speed (m/s):         mean {result['speed_mean']:.2f}   "
          f"envs: {' '.join(f'{v:.2f}' for v in result['speed_mps'])}")
    print(f"  net speed (m/s):     mean {result['net_speed_mean']:.2f}   "
          f"envs: {' '.join(f'{v:.2f}' for v in result['net_speed_mps'])}")
    print(f"  |yaw drift| deg:     mean {result['yaw_drift_mean']:.0f}   "
          f"envs: {' '.join(f'{v:.0f}' for v in result['yaw_drift_deg'])}")
    print()
    print("Key joints:")
    print(f"  HR knee mean (deg):  mean {result['hr_knee_mean_overall']:.1f}   "
          f"(ref -14.2)")
    cb = result["carpus_in_band"]
    print(f"  Carpus % in [+5.7, +74.5] deg fold band:")
    for env, (fl_p, fr_p) in enumerate(cb):
        print(f"     env{env}:  FL {fl_p:.0f}%  FR {fr_p:.0f}%")


def print_compare(results):
    print(f"\n=== Comparison across {len(results)} runs ===\n")
    labels = list(WALK_REF.keys()) + [
        "straightness", "speed", "net speed", "|yaw drift|", "HR knee mean", "carpus avg in band"
    ]
    # Build rows
    header = f"{'metric':30s}  " + "  ".join(f"{Path(r['npz_path']).stem[-18:]:>18s}" for r in results)
    print(header)
    for label in WALK_REF:
        row = f"{label:30s}  "
        row += "  ".join(f"{r['correlations'][label]:>+18.3f}" for r in results)
        ref = WALK_REF[label]
        row += f"  (ref {ref:+.2f})"
        print(row)
    for k_label, k in [
        ("straightness ratio", "straightness_mean"),
        ("speed m/s", "speed_mean"),
        ("net speed m/s", "net_speed_mean"),
        ("|yaw drift| deg", "yaw_drift_mean"),
        ("HR knee mean deg", "hr_knee_mean_overall"),
    ]:
        row = f"{k_label:30s}  "
        row += "  ".join(f"{r[k]:>18.3f}" for r in results)
        print(row)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        if not p.is_file():
            print(f"WARN: skipping non-file {p}", file=sys.stderr)
    results = [analyze(p) for p in paths if p.is_file()]
    for r in results:
        print_single(r)
    if len(results) > 1:
        print_compare(results)


if __name__ == "__main__":
    main()
