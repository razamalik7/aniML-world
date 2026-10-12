"""Render a multi-env rollout .npz to SEPARATE single-env still PNGs.

`render_playback_meshed.py` stitches all envs side-by-side into one wide MP4.
That's great for motion, but a 4-wide panel is hard to read (and didn't display
well for review). This tool renders each env on its own, at a handful of chosen
gait frames, as individual PNGs — so you can judge each spawn state's posture
independently ("does env 2 actually stand on its hooves?").

Pure CPU: numpy FK + trimesh + matplotlib. No Isaac, no GPU — safe to run
alongside a live training job. Reuses the exact FK/mesh helpers from
`render_playback_meshed.py` so the geometry matches the MP4 renderer bit for bit.

  python tools/rig_pipeline/render_env_stills.py \
    --input docs/polyart_r7b_rollout.npz \
    --urdf assets/deer_polyart_urdf/deer_polyart.urdf \
    --outdir docs --tag polyart_r7b \
    --envs 0,1,2,3 --frames 30,60,90,120
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Reuse the exact FK + mesh loading from the MP4 renderer so a still matches a
# video frame exactly.
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.render_playback_meshed import (  # noqa: E402
    parse_urdf_full,
    load_meshes,
    compute_link_transforms,
    quat_wxyz_to_R,
    shade,
)


def render_still(ax, link_meshes, meshes, center, half_win, light_dir,
                 azim, elev):
    """Draw one deer configuration (already FK'd to world verts) into `ax`."""
    ax.clear()
    ax.set_facecolor("#e8eef4")
    for link, world_v in link_meshes.items():
        faces = meshes[link]["faces"]
        tris = world_v[faces]
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        normals = np.cross(e1, e2)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        normals = normals / lengths
        shade_factor = shade(normals, light_dir)
        base_rgb = np.array([0.78, 0.58, 0.38])
        colors = np.clip(base_rgb[None, :] * shade_factor[:, None], 0, 1)
        colors_rgba = np.hstack([colors, np.ones((len(colors), 1))])
        pc = Poly3DCollection(
            tris, facecolors=colors_rgba,
            edgecolors=(0.3, 0.2, 0.1, 0.4), linewidths=0.15,
        )
        ax.add_collection3d(pc)

    cx, cy, cz = center
    xlo, xhi = cx - half_win[0], cx + half_win[0]
    ylo, yhi = cy - half_win[1], cy + half_win[1]
    # Faint ground plane + world-grid lines (every 0.5 m) for a size cue.
    ground = np.array([[xlo, ylo, 0], [xhi, ylo, 0], [xhi, yhi, 0], [xlo, yhi, 0]])
    gpc = Poly3DCollection([ground], facecolors=(0.82, 0.82, 0.84, 0.30),
                           edgecolors=(0, 0, 0, 0.0))
    gpc.set_zorder(-10)
    ax.add_collection3d(gpc)
    step = 0.5
    gc = (0.45, 0.45, 0.50, 0.7)
    x = step * math.floor(xlo / step)
    while x <= step * math.ceil(xhi / step) + 1e-9:
        for L in ax.plot([x, x], [ylo, yhi], [0, 0], color=gc, linewidth=0.8):
            L.set_zorder(-5)
        x += step
    y = step * math.floor(ylo / step)
    while y <= step * math.ceil(yhi / step) + 1e-9:
        for L in ax.plot([xlo, xhi], [y, y], [0, 0], color=gc, linewidth=0.8):
            L.set_zorder(-5)
        y += step

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(-0.05, 1.8)
    ax.set_box_aspect((2 * half_win[0], 2 * half_win[1], 1.85))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)


def fk_world_verts(jp_row, rq_row, rp_row, joints_list, meshes, origin_map):
    """FK one (env, frame): return {link: (V,3) world verts} + bbox center."""
    ja = {jn: float(jp_row[i]) for i, jn in enumerate(joints_list)}
    link_T = compute_link_transforms(ja, origin_map)
    R_root = quat_wxyz_to_R(rq_row)
    t_root = rp_row.astype(np.float64)
    link_meshes = {}
    fmin = np.array([np.inf, np.inf, np.inf])
    fmax = np.array([-np.inf, -np.inf, -np.inf])
    for link, mesh in meshes.items():
        if link not in link_T:
            continue
        R_link, t_link = link_T[link]
        R_full = R_root @ R_link @ mesh["vorigin_R"]
        t_full = R_root @ (R_link @ mesh["vorigin_xyz"] + t_link) + t_root
        world_v = mesh["vertices"] @ R_full.T + t_full
        link_meshes[link] = world_v
        fmin = np.minimum(fmin, world_v.min(0))
        fmax = np.maximum(fmax, world_v.max(0))
    center = 0.5 * (fmin + fmax)
    return link_meshes, center, fmin, fmax


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--urdf", default=str(REPO / "assets/deer_polyart_urdf/deer_polyart.urdf"))
    p.add_argument("--outdir", default=str(REPO / "docs"))
    p.add_argument("--tag", required=True, help="filename prefix, e.g. polyart_r7_i1200")
    p.add_argument("--envs", default="0,1,2,3")
    p.add_argument("--frames", default="30,60,90,120")
    p.add_argument("--azim", type=float, default=0.0)
    p.add_argument("--elev", type=float, default=12.0)
    args = p.parse_args()

    envs = [int(e) for e in str(args.envs).split(",") if e != ""]
    frames = [int(f) for f in str(args.frames).split(",") if f != ""]

    data = np.load(args.input, allow_pickle=True)
    jp_all = data["joint_positions"]     # (T, N, J)
    rp_all = data["root_pos"]            # (T, N, 3)
    rq_all = data["root_quat_wxyz"]      # (T, N, 4) wxyz
    joints_list = list(data["joints_list"].astype(str))
    fps_in = float(data["fps"]) if "fps" in data else 50.0
    T, N, J = jp_all.shape
    print(f"[stills] {args.input}: T={T} N={N} J={J} fps={fps_in}")

    origin_map, visual_map = parse_urdf_full(Path(args.urdf))
    meshes = load_meshes(visual_map, Path(args.urdf).parent)
    print(f"[stills] loaded {len(meshes)} link meshes from {args.urdf}")

    light_dir = np.array([0.4, 0.3, 1.0]); light_dir /= np.linalg.norm(light_dir)
    half_win = np.array([1.2, 1.2, 1.0])
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for e in envs:
        if e >= N:
            print(f"[stills] skip env {e} (only {N} envs)")
            continue
        # Per-env head-height readout across the clip: an honest, metric-free
        # sanity number (max skull Z over the frames we render).
        for f in frames:
            t = min(f, T - 1)
            link_meshes, center, fmin, fmax = fk_world_verts(
                jp_all[t, e], rq_all[t, e], rp_all[t, e],
                joints_list, meshes, origin_map,
            )
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            render_still(ax, link_meshes, meshes, center, half_win,
                         light_dir, args.azim, args.elev)
            top = fmax[2]
            ax.set_title(f"{args.tag}  env {e}  t={t/fps_in:.2f}s  "
                         f"top_z={top:.2f}m", fontsize=10)
            fig.tight_layout()
            out = outdir / f"{args.tag}_env{e}_f{f}.png"
            fig.savefig(out, dpi=110)
            plt.close(fig)
            written.append(out.name)
            print(f"[stills]   wrote {out.name}  top_z={top:.2f}")
    print(f"[stills] done: {len(written)} PNGs in {outdir}")


if __name__ == "__main__":
    main()
