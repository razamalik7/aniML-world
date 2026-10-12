"""Render the deer_walk.npy reference animation to MP4.

Quick wrapper that converts the reference motion .npy (15 joints, 30 fps,
single cycle) into the .npz format render_playback_meshed.py expects, then
calls that renderer.

Loops the cycle 4 times so the user can see the gait clearly. Single env.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default=str(REPO / "assets" / "motions" / "deer_walk.npy"))
    ap.add_argument("--output", default=str(REPO / "logs" / "videos" / "walk_reference.mp4"))
    ap.add_argument("--loops", type=int, default=4)
    args = ap.parse_args()

    data = np.load(args.motion, allow_pickle=True).item()
    jp = data["joint_positions"].astype(np.float32)      # (T, 15)
    rp = data["root_position"].astype(np.float32)        # (T, 3)
    rq = data["root_quaternion"].astype(np.float32)      # (T, 4)
    fps = float(data["fps"])
    T = jp.shape[0]

    # Loop the cycle. Forward translation is in Y per the convert; we accumulate
    # per loop so the deer keeps walking instead of teleporting.
    if args.loops > 1:
        cycle_dY = float(rp[-1, 1] - rp[0, 1])
        jps = [jp]
        rps = [rp]
        rqs = [rq]
        for k in range(1, args.loops):
            jps.append(jp.copy())
            rp_k = rp.copy()
            rp_k[:, 1] += k * cycle_dY
            rps.append(rp_k)
            rqs.append(rq.copy())
        jp = np.concatenate(jps, axis=0)
        rp = np.concatenate(rps, axis=0)
        rq = np.concatenate(rqs, axis=0)

    # Add the env dim (T, 1, ...) so the renderer is happy.
    jp = jp[:, None, :]
    rp = rp[:, None, :]
    rq = rq[:, None, :]

    npz_path = Path(args.output).with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(npz_path),
        joint_positions=jp,
        root_pos=rp,
        root_quat_wxyz=rq,
        joints_list=np.array(data["joints_list"], dtype=object),
        fps=fps,
    )
    print(f"Wrote .npz: {npz_path}  T={jp.shape[0]} joints={jp.shape[2]}")

    # Now call the renderer
    cmd = [
        sys.executable,
        str(REPO / "tools" / "render_playback_meshed.py"),
        "--input", str(npz_path),
        "--output", args.output,
        "--fps", str(int(fps)),
        "--max_envs", "1",
        "--stride", "1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
