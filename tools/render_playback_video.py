"""Render a play_amp.py trajectory dump (.npz) to MP4.

Reads (T, N, ...) joint_positions / root_pos / root_quat_wxyz and renders an
animated stick-figure of N parallel deer over T steps, side-by-side.
Completely bypasses Isaac Sim — pure numpy + matplotlib + ffmpeg.

This exists because Isaac Sim 5.1's omni.hydra.rtx renderer crashes during
init/shutdown on RTX 50-series GPUs (both viewer and headless+enable_cameras
modes). Trajectory dump + offline render dodges the entire broken stack.

Usage:
    python tools/render_playback_video.py --input <path>/play_amp_*.npz
                                          [--output <path>.mp4]
                                          [--fps 50]
                                          [--side_only]

Output: an MP4 next to the .npz, named like ``play_amp_*.mp4``.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets" / "deer_urdf" / "deer_world.urdf"


# --- URDF parsing & FK (mirrors tools/visualize_motion_npy.py) ----------------

def parse_urdf(urdf_path: Path):
    text = urdf_path.read_text(encoding="utf-8")
    joint_pat = re.compile(
        r'<joint\s+name="(?P<name>[^"]+)"[^>]*>\s*'
        r'<parent\s+link="(?P<parent>[^"]+)"\s*/>'
        r'\s*<child\s+link="(?P<child>[^"]+)"\s*/>'
        r'\s*<origin\s+xyz="(?P<xyz>[^"]+)"[^>]*/>',
        re.DOTALL,
    )
    origin_map = {}
    for m in joint_pat.finditer(text):
        name = m.group("name")
        parent = m.group("parent")
        child = m.group("child")
        xyz = np.array([float(v) for v in m.group("xyz").split()], dtype=np.float64)
        origin_map[name] = (xyz, parent, child)
    return origin_map


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def compute_link_positions(joint_angles: dict, origin_map: dict) -> dict:
    """FK in URDF (Y-up) frame, pelvis at origin."""
    link_pos = {"pelvis": np.zeros(3)}
    link_rot = {"pelvis": np.eye(3)}
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
            R_parent = link_rot[parent]
            p_parent = link_pos[parent]
            link_pos[child] = p_parent + R_parent @ xyz
            link_rot[child] = R_parent @ R_j
            placed.add(child)
            progress = True
        pending = remaining
        if not pending:
            break
        if not progress:
            raise RuntimeError(f"Unplaceable links: {[x[0] for x in pending]}")
    return link_pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, type=str,
                        help="Path to play_amp_*.npz trajectory dump.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 path. Default: same name as input "
                             "with .mp4 extension.")
    parser.add_argument("--fps", type=int, default=50,
                        help="Output video framerate (default 50, matching "
                             "sim step rate).")
    parser.add_argument("--side_only", action="store_true",
                        help="Render only the side view (default: side+top).")
    parser.add_argument("--max_envs", type=int, default=4,
                        help="Maximum number of parallel envs to display "
                             "(default 4).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".mp4")

    print(f"[render] loading {input_path}")
    data = np.load(input_path, allow_pickle=True)
    jp_all = data["joint_positions"]      # (T, N, 15)
    rp_all = data["root_pos"]             # (T, N, 3) world frame, Z-up Isaac
    rq_all = data["root_quat_wxyz"]       # (T, N, 4) wxyz
    joints_list = list(data["joints_list"].astype(str))
    fps_in = float(data["fps"]) if "fps" in data else 50.0

    T, N, J = jp_all.shape
    N_show = min(N, args.max_envs)
    print(f"[render] T={T} steps, N={N} envs (showing {N_show}), {J} joints, "
          f"fps_in={fps_in}")

    origin_map = parse_urdf(URDF_PATH)
    print(f"[render] parsed URDF: {len(origin_map)} joints")

    # The deer URDF is Y-up (URDF native). Spawn rotation is Rx(-90°) so
    # body Y-up -> world Z-up, and body -Z (anatomical forward) -> world -Y.
    # We'll work in world frame (Isaac Z-up) directly, applying the per-step
    # root quaternion to the FK output.

    # Set up figure: side view (looking from world +X, so we see Y-Z plane)
    # plus top view (Z-X plane) per env. Layout: rows = views, cols = envs.
    n_views = 1 if args.side_only else 2
    fig, axes = plt.subplots(
        n_views, N_show,
        figsize=(3.5 * N_show, 3.5 * n_views),
        squeeze=False,
    )

    # Pre-compute world-frame link positions per (step, env)
    print(f"[render] computing FK for {T} steps × {N_show} envs ...")
    # link names: collect from origin_map
    all_links = sorted({"pelvis", *(p for _, p, _ in origin_map.values()),
                        *(c for _, _, c in origin_map.values())})
    edges = [(p, c) for (_, p, c) in origin_map.values()]

    link_world: dict[str, np.ndarray] = {
        ln: np.zeros((T, N_show, 3), dtype=np.float32) for ln in all_links
    }
    for t in range(T):
        for n in range(N_show):
            ja = {jn: float(jp_all[t, n, i]) for i, jn in enumerate(joints_list)}
            pelvis_local = compute_link_positions(ja, origin_map)
            R_root = quat_wxyz_to_R(rq_all[t, n])
            origin_world = rp_all[t, n]
            for ln, p_local in pelvis_local.items():
                link_world[ln][t, n] = origin_world + R_root @ p_local
        if t % 200 == 0:
            print(f"[render]   FK step {t}/{T}", flush=True)

    # World view bounds (Z-up Isaac). Find bbox per env across whole episode.
    all_pts = np.stack([link_world[ln] for ln in all_links], axis=0)  # (L,T,N,3)
    print(f"[render] world bbox per env (min, max):")
    bounds = []
    for n in range(N_show):
        env_pts = all_pts[:, :, n, :]  # (L,T,3)
        xmin, ymin, zmin = env_pts.reshape(-1, 3).min(0)
        xmax, ymax, zmax = env_pts.reshape(-1, 3).max(0)
        # symmetric padding
        pad = 0.5
        bounds.append({
            "x": (xmin - pad, xmax + pad),
            "y": (ymin - pad, ymax + pad),
            "z": (zmin - pad, zmax + pad),
        })
        print(f"  env {n}: x=[{xmin:.2f},{xmax:.2f}] y=[{ymin:.2f},{ymax:.2f}] "
              f"z=[{zmin:.2f},{zmax:.2f}]")

    # Set up video writer
    print(f"[render] writing {T} frames to {output_path} @ {args.fps} fps ...")
    writer = imageio.get_writer(
        str(output_path), fps=args.fps, codec="libx264",
        quality=8, macro_block_size=1,
    )
    for t in range(T):
        for n in range(N_show):
            for view_idx in range(n_views):
                ax = axes[view_idx, n]
                ax.clear()
                if view_idx == 0:
                    # Side view: Y horizontal (forward = -Y), Z vertical
                    a, b = 1, 2
                    ax.set_xlabel("world Y (deer travels -Y)")
                    ax.set_ylabel("world Z (up)")
                    ax.set_xlim(*bounds[n]["y"])
                    ax.set_ylim(*bounds[n]["z"])
                else:
                    # Top view: Y horizontal, X vertical (lateral)
                    a, b = 1, 0
                    ax.set_xlabel("world Y")
                    ax.set_ylabel("world X (lateral)")
                    ax.set_xlim(*bounds[n]["y"])
                    ax.set_ylim(*bounds[n]["x"])
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)
                # ground line at z=0 in side view
                if view_idx == 0:
                    ax.axhline(0, color="brown", linewidth=0.5, alpha=0.5)
                # draw edges
                for parent, child in edges:
                    pp = link_world[parent][t, n]
                    cc = link_world[child][t, n]
                    ax.plot([pp[a], cc[a]], [pp[b], cc[b]],
                            "-", color="gray", linewidth=1.2)
                # joints as dots
                for ln in all_links:
                    p = link_world[ln][t, n]
                    ax.plot([p[a]], [p[b]], "o",
                            color="steelblue", markersize=2.5)
                if view_idx == 0:
                    ax.set_title(f"env {n} side  t={t/fps_in:.2f}s")
                else:
                    ax.set_title(f"env {n} top")
        fig.suptitle(f"r15 model_2999 playback — frame {t}/{T} "
                     f"({t/fps_in:.2f}s of {T/fps_in:.2f}s)")
        fig.tight_layout()
        # Render figure to a numpy frame
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[..., :3].copy()  # drop alpha
        writer.append_data(frame)
        if t % 50 == 0:
            print(f"[render]   wrote frame {t}/{T}", flush=True)
    writer.close()
    plt.close(fig)
    print(f"[render] done: {output_path}")
    print(f"[render]   open with: start \"\" \"{output_path}\"")


if __name__ == "__main__":
    main()
