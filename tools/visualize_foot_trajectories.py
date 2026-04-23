"""Plot foot trajectories across one walk cycle, and joint angle curves,
so we can verify the converted motion is a real walk (alternating legs,
correct phase relationships) without fiddling with stick-figure scaling."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_motion_npy as v   # reuse FK

REPO_ROOT = Path(__file__).resolve().parent.parent
NPY_PATH = REPO_ROOT / "assets" / "motions" / "deer_walk.npy"
OUT_DIR = REPO_ROOT / "assets" / "motions" / "deer_walk_viz"


def main() -> None:
    payload = np.load(NPY_PATH, allow_pickle=True).item()
    jp = payload["joint_positions"]
    joints = payload["joints_list"]
    fps = payload["fps"]
    n = jp.shape[0]
    parent_map, origin_map = v.parse_urdf(v.URDF_PATH)

    feet = ["HR_foot", "HL_foot", "FR_foot", "FL_foot"]
    traj = {foot: np.zeros((n, 3)) for foot in feet}
    for f in range(n):
        ja = {name: float(jp[f, i]) for i, name in enumerate(joints)}
        pos = v.compute_link_positions(ja, origin_map)
        for foot in feet:
            traj[foot][f] = pos[foot]

    # ---- Figure 1: foot trajectories in URDF space (side view: Z vs Y) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"HR_foot": "tab:red", "HL_foot": "tab:blue",
              "FR_foot": "tab:green", "FL_foot": "tab:orange"}
    ax = axes[0]
    for foot in feet:
        t = traj[foot]
        ax.plot(t[:, 2], t[:, 1], "-", color=colors[foot], label=foot, lw=1.5)
        ax.plot(t[0, 2], t[0, 1], "o", color=colors[foot], markersize=6)
    ax.set_xlabel("Z (forward ←→ back, URDF)")
    ax.set_ylabel("Y (down ← → up, URDF Y-inverted)")
    ax.set_title("Foot trajectories — side view\n"
                 "(should show oval/arc loops if walking)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # ---- Figure 2: foot Z vs time, colored by side ----
    ax = axes[1]
    t_axis = np.arange(n) / fps
    for foot in feet:
        ax.plot(t_axis, traj[foot][:, 2], "-", color=colors[foot],
                label=f"{foot} (Z)", lw=1.8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Z (forward/back, URDF)")
    ax.set_title("Foot forward-position over time\n"
                 "(walk: diagonal pairs should alternate)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out1 = OUT_DIR / "_foot_trajectories.png"
    fig.savefig(out1, dpi=100)
    plt.close(fig)
    print(f"Wrote {out1}")

    # ---- Figure 3: joint angle curves ----
    fig, axes = plt.subplots(5, 3, figsize=(14, 12))
    for i, name in enumerate(joints):
        ax = axes.flat[i]
        ax.plot(t_axis, np.degrees(jp[:, i]), "-", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("deg", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Joint angle curves across walk cycle")
    fig.tight_layout()
    out2 = OUT_DIR / "_joint_curves.png"
    fig.savefig(out2, dpi=100)
    plt.close(fig)
    print(f"Wrote {out2}")

    # ---- Diagonal pair phase check ----
    # In a trot, diagonal pairs (HR+FL, HL+FR) move together.
    # In a walk, each foot lifts in sequence (LF, RH, RF, LH or similar).
    # Print per-foot "max forward t" and phase differences.
    max_t = {}
    for foot in feet:
        max_t[foot] = t_axis[int(np.argmax(traj[foot][:, 2]))]
    print("\nTime of maximum Z (backward-most) per foot:")
    for foot in feet:
        print(f"  {foot}: t={max_t[foot]:.3f} s  "
              f"(phase {max_t[foot] * fps / n:.2f} of cycle)")


if __name__ == "__main__":
    main()
