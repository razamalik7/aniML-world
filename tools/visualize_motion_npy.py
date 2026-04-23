"""Forward-kinematics visualization of a converted AMP motion .npy.

Reads assets/motions/deer_walk.npy, applies joint angles to the URDF
skeleton (by parsing joint origins and parent/child links from
assets/deer_urdf/deer_world.urdf), and renders a top-and-side view of the
deer per frame + a small animation as PNG frames.

Output:
  assets/motions/deer_walk_viz/frame_000.png ... frame_NNN.png

We keep this minimal -- pure numpy + matplotlib, no pybullet/isaac needed.
Joints rotate around URDF-X (all 15 joints are X-axis revolute, Y-up URDF frame).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
NPY_PATH = REPO_ROOT / "assets" / "motions" / "deer_walk.npy"
URDF_PATH = REPO_ROOT / "assets" / "deer_urdf" / "deer_world.urdf"
OUT_DIR = REPO_ROOT / "assets" / "motions" / "deer_walk_viz"


def parse_urdf(urdf_path: Path):
    """Return (parent_map, origin_map) where parent_map[child_link] = parent_link
    and origin_map[joint_name] = (xyz numpy array, parent_link, child_link)."""
    text = urdf_path.read_text(encoding="utf-8")
    # Each joint block has parent/child + origin xyz.
    joint_pat = re.compile(
        r'<joint\s+name="(?P<name>[^"]+)"[^>]*>\s*'
        r'<parent\s+link="(?P<parent>[^"]+)"\s*/>'
        r'\s*<child\s+link="(?P<child>[^"]+)"\s*/>'
        r'\s*<origin\s+xyz="(?P<xyz>[^"]+)"[^>]*/>',
        re.DOTALL,
    )
    parent_map = {}
    origin_map = {}
    for m in joint_pat.finditer(text):
        name = m.group("name")
        parent = m.group("parent")
        child = m.group("child")
        xyz = np.array([float(v) for v in m.group("xyz").split()], dtype=np.float64)
        parent_map[child] = parent
        origin_map[name] = (xyz, parent, child)
    return parent_map, origin_map


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def compute_link_positions(joint_angles: dict[str, float], origin_map: dict) -> dict:
    """Walk the URDF tree and compute each link's position in pelvis (base) frame."""
    # Build link children list to traverse.
    link_world_pos: dict[str, np.ndarray] = {}
    link_world_rot: dict[str, np.ndarray] = {}
    link_world_pos["pelvis"] = np.zeros(3)
    link_world_rot["pelvis"] = np.eye(3)

    # Repeat until we've placed all children. Order-free BFS.
    placed = {"pelvis"}
    pending = list(origin_map.items())
    for _ in range(len(pending) + 5):
        progress = False
        remaining = []
        for jname, (xyz, parent, child) in pending:
            if parent not in placed:
                remaining.append((jname, (xyz, parent, child)))
                continue
            angle = joint_angles.get(jname, 0.0)
            R_j = rot_x(angle)
            R_parent = link_world_rot[parent]
            p_parent = link_world_pos[parent]
            p_child = p_parent + R_parent @ xyz
            R_child = R_parent @ R_j
            link_world_pos[child] = p_child
            link_world_rot[child] = R_child
            placed.add(child)
            progress = True
        pending = remaining
        if not pending:
            break
        if not progress:
            raise RuntimeError(f"Unplaceable links remain: {[x[0] for x in pending]}")
    return link_world_pos


def draw_frame(link_pos: dict[str, np.ndarray], origin_map: dict,
               title: str, out_path: Path) -> None:
    """Draw side view and front view of the skeleton."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Build edges from origin_map (parent -> child)
    edges = [(parent, child) for (_xyz, parent, child) in origin_map.values()]

    def draw_view(ax, axis_x, axis_y, axis_x_label, axis_y_label):
        for parent, child in edges:
            p = link_pos[parent]
            c = link_pos[child]
            ax.plot([p[axis_x], c[axis_x]], [p[axis_y], c[axis_y]],
                    "-", color="gray", linewidth=1.2)
            ax.plot([p[axis_x]], [p[axis_y]], "o", color="steelblue", markersize=3)
            ax.plot([c[axis_x]], [c[axis_y]], "o", color="steelblue", markersize=3)
        ax.set_xlabel(axis_x_label)
        ax.set_ylabel(axis_y_label)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        # fixed view limits for consistent animation
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.2, 0.8)

    # URDF is Y-up (before Rx(-90)). X = lateral (left-right), Y = vertical,
    # Z = forward/backward. Side view = Z vs Y. Front view = X vs Y.
    draw_view(axes[0], 2, 1, "Z (fwd→back)", "Y (up)")
    axes[0].set_title("Side view")
    draw_view(axes[1], 0, 1, "X (lateral)", "Y (up)")
    axes[1].set_title("Front view")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main() -> None:
    payload = np.load(NPY_PATH, allow_pickle=True).item()
    joints_list = payload["joints_list"]
    jp = payload["joint_positions"]
    fps = payload["fps"]
    n_frames = jp.shape[0]
    print(f"Loaded {NPY_PATH.name}: {n_frames} frames, {len(joints_list)} joints, {fps} fps")

    parent_map, origin_map = parse_urdf(URDF_PATH)
    print(f"Parsed URDF: {len(origin_map)} joints / {len(set(parent_map.values()) | set(parent_map))} links")
    # Sanity: all our motion joints are present in the URDF
    missing = [j for j in joints_list if j not in origin_map]
    if missing:
        print(f"WARNING: URDF missing joints: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in range(n_frames):
        ja = {name: float(jp[f, i]) for i, name in enumerate(joints_list)}
        link_pos = compute_link_positions(ja, origin_map)
        draw_frame(
            link_pos, origin_map,
            title=f"Deer walk — frame {f:02d}/{n_frames-1}  (t={f/fps:.3f}s)",
            out_path=OUT_DIR / f"frame_{f:03d}.png",
        )
    print(f"Wrote {n_frames} frames to {OUT_DIR}")

    # Also dump a compact panel: every Nth frame on a single image
    n_show = min(8, n_frames)
    step = max(1, n_frames // n_show)
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    for i in range(n_show):
        f = i * step
        ja = {name: float(jp[f, j]) for j, name in enumerate(joints_list)}
        link_pos = compute_link_positions(ja, origin_map)
        for row, (ax_idx_x, ax_idx_y, xl, yl, title) in enumerate([
            (2, 1, "Z", "Y", "side"), (0, 1, "X", "Y", "front")
        ]):
            ax = axes[row, i]
            for parent, child in [(p, c) for (_xyz, p, c) in origin_map.values()]:
                p = link_pos[parent]; c = link_pos[child]
                ax.plot([p[ax_idx_x], c[ax_idx_x]], [p[ax_idx_y], c[ax_idx_y]],
                        "-", color="gray", linewidth=1.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.set_xlim(-1, 1); ax.set_ylim(-1.2, 0.8)
            if row == 0:
                ax.set_title(f"f{f}")
            if i == 0:
                ax.set_ylabel(title)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Walk cycle sample frames")
    fig.tight_layout()
    panel_path = OUT_DIR / "_panel.png"
    fig.savefig(panel_path, dpi=80)
    plt.close(fig)
    print(f"Wrote panel: {panel_path}")


if __name__ == "__main__":
    main()
