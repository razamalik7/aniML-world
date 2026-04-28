"""Render a play_amp.py trajectory dump (.npz) to MP4 with the actual deer
meshes (loaded from URDF visual STLs).

This bypasses Isaac Sim's broken renderer entirely — we apply per-frame FK to
the joint chain in pure numpy, transform each link's mesh vertices into world
space, project to a 2D scene with matplotlib's Poly3DCollection, and stitch
frames into MP4 via imageio_ffmpeg.

Usage:
    python tools/render_playback_meshed.py --input <path>/play_amp_*.npz
                                            [--output <path>.mp4]
                                            [--fps 50]
                                            [--max_envs 4]
                                            [--stride 2]
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
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets" / "deer_urdf" / "deer_world.urdf"


# -------- URDF parsing (links + joint origins + visual mesh paths) -----------

def parse_urdf_full(urdf_path: Path):
    """Return:
      origin_map[joint] = (xyz, rpy, parent_link, child_link)
      visual_map[link]  = (mesh_filename, xyz, rpy)   for each link with visual
    """
    text = urdf_path.read_text(encoding="utf-8")

    # Joints (parent link, child link, origin xyz/rpy)
    joint_pat = re.compile(
        r'<joint\s+name="(?P<name>[^"]+)"[^>]*>\s*'
        r'<parent\s+link="(?P<parent>[^"]+)"\s*/>'
        r'\s*<child\s+link="(?P<child>[^"]+)"\s*/>'
        r'\s*<origin\s+xyz="(?P<xyz>[^"]+)"(?:\s+rpy="(?P<rpy>[^"]+)")?[^>]*/>',
        re.DOTALL,
    )
    origin_map = {}
    for m in joint_pat.finditer(text):
        name = m.group("name")
        parent = m.group("parent")
        child = m.group("child")
        xyz = np.array([float(v) for v in m.group("xyz").split()], dtype=np.float64)
        rpy_str = m.group("rpy") or "0 0 0"
        rpy = np.array([float(v) for v in rpy_str.split()], dtype=np.float64)
        origin_map[name] = (xyz, rpy, parent, child)

    # Visual meshes per link
    # <link name="X"> ... <visual> ... <origin xyz="..." rpy="..."/>
    #     <geometry><mesh filename="..."/></geometry> ... </visual> ... </link>
    link_pat = re.compile(r'<link\s+name="(?P<name>[^"]+)">(?P<body>.*?)</link>',
                          re.DOTALL)
    visual_pat = re.compile(
        r'<visual[^>]*>\s*'
        r'<origin\s+xyz="(?P<xyz>[^"]+)"(?:\s+rpy="(?P<rpy>[^"]+)")?[^>]*/>\s*'
        r'<geometry>\s*<mesh\s+filename="(?P<filename>[^"]+)"\s*/>\s*</geometry>',
        re.DOTALL,
    )
    visual_map = {}
    for lm in link_pat.finditer(text):
        name = lm.group("name")
        body = lm.group("body")
        vm = visual_pat.search(body)
        if not vm:
            continue
        xyz = np.array([float(v) for v in vm.group("xyz").split()], dtype=np.float64)
        rpy_str = vm.group("rpy") or "0 0 0"
        rpy = np.array([float(v) for v in rpy_str.split()], dtype=np.float64)
        visual_map[name] = (vm.group("filename"), xyz, rpy)

    return origin_map, visual_map


def rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """RPY (URDF Z-Y-X intrinsic) to rotation matrix."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def compute_link_transforms(joint_angles: dict, origin_map: dict) -> dict:
    """Return link_T[link] = (3x3 R, 3 t) in URDF root (pelvis) frame."""
    link_T = {"pelvis": (np.eye(3), np.zeros(3))}
    placed = {"pelvis"}
    pending = list(origin_map.items())
    for _ in range(len(pending) + 5):
        progress = False
        remaining = []
        for jname, (xyz, rpy, parent, child) in pending:
            if parent not in placed:
                remaining.append((jname, (xyz, rpy, parent, child)))
                continue
            R_parent, t_parent = link_T[parent]
            # Joint frame relative to parent: translation xyz + rotation rpy +
            # joint rotation (X-axis revolute for our deer).
            R_joint_origin = rpy_to_R(rpy)
            R_joint = rot_x(joint_angles.get(jname, 0.0))
            R_child_in_parent = R_joint_origin @ R_joint
            t_child_in_parent = xyz
            R_child = R_parent @ R_child_in_parent
            t_child = R_parent @ t_child_in_parent + t_parent
            link_T[child] = (R_child, t_child)
            placed.add(child)
            progress = True
        pending = remaining
        if not pending:
            break
        if not progress:
            raise RuntimeError(f"Unplaceable links: {[x[0] for x in pending]}")
    return link_T


# -------- Mesh loading & shaded face generation -----------------------------

def load_meshes(visual_map: dict, urdf_dir: Path) -> dict:
    """Load STL for each link with a visual, plus its visual-origin transform."""
    meshes = {}
    for link, (filename, xyz, rpy) in visual_map.items():
        path = urdf_dir / filename
        if not path.is_file():
            print(f"[render] WARNING: missing mesh {path}")
            continue
        mesh = trimesh.load_mesh(str(path), force="mesh")
        if mesh.is_empty:
            continue
        meshes[link] = {
            "vertices": np.asarray(mesh.vertices, dtype=np.float64),
            "faces": np.asarray(mesh.faces, dtype=np.int32),
            "vorigin_xyz": xyz,
            "vorigin_R": rpy_to_R(rpy),
        }
    return meshes


def shade(face_normals: np.ndarray, light_dir: np.ndarray) -> np.ndarray:
    """Lambert shading per face. Returns shading factor in [0.2, 1.0]."""
    cos = (face_normals @ light_dir).clip(-1, 1)
    return 0.5 + 0.5 * cos  # [0, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max_envs", type=int, default=2,
                        help="Max envs to render side-by-side. 1-2 keeps "
                             "matplotlib fast; 4 is doable but slow.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth frame (1=all, 2=half, etc.). "
                             "Lower fps if you stride.")
    parser.add_argument("--azim", type=float, default=-75,
                        help="View azimuth degrees. -90 ~= side view "
                             "(looking from world +X).")
    parser.add_argument("--elev", type=float, default=15,
                        help="View elevation degrees.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path = Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + "_meshed.mp4")

    print(f"[render] loading {input_path}")
    data = np.load(input_path, allow_pickle=True)
    jp_all = data["joint_positions"]      # (T, N, 15)
    rp_all = data["root_pos"]             # (T, N, 3) world-frame Isaac Z-up
    rq_all = data["root_quat_wxyz"]       # (T, N, 4) wxyz
    joints_list = list(data["joints_list"].astype(str))
    fps_in = float(data["fps"]) if "fps" in data else 50.0

    T_full, N, J = jp_all.shape
    stride = max(1, args.stride)
    T = (T_full + stride - 1) // stride
    N_show = min(N, args.max_envs)
    out_fps = max(1, int(round(args.fps / stride)))
    print(f"[render] T_full={T_full}, stride={stride}, T_render={T}, "
          f"N={N}->{N_show}, fps_in={fps_in}, out_fps={out_fps}")

    print(f"[render] parsing URDF {URDF_PATH}")
    origin_map, visual_map = parse_urdf_full(URDF_PATH)
    urdf_dir = URDF_PATH.parent
    meshes = load_meshes(visual_map, urdf_dir)
    print(f"[render] loaded {len(meshes)} link meshes "
          f"(of {len(visual_map)} URDF visuals)")

    # Set up figure: one 3D axis per env
    fig = plt.figure(figsize=(6 * N_show, 6))
    axes = [fig.add_subplot(1, N_show, n + 1, projection="3d")
            for n in range(N_show)]

    # Pre-compute world-frame mesh vertices per (frame, env, link)
    # This is memory-heavy; for 1500 frames * 4 envs * ~16 links * ~50 verts,
    # that's ~5M floats — fine.
    print(f"[render] computing FK for {T}*{N_show} = {T*N_show} configs ...")
    world_verts: list = []  # world_verts[t][n][link] = (V,3) vertices
    bounds_per_env = [[(np.inf, -np.inf)] * 3 for _ in range(N_show)]

    light_dir = np.array([0.4, 0.3, 1.0])
    light_dir /= np.linalg.norm(light_dir)

    for ti in range(T):
        t = ti * stride
        env_data = []
        for n in range(N_show):
            ja = {jn: float(jp_all[t, n, i]) for i, jn in enumerate(joints_list)}
            link_T = compute_link_transforms(ja, origin_map)
            R_root = quat_wxyz_to_R(rq_all[t, n])
            t_root = rp_all[t, n].astype(np.float64)
            link_meshes = {}
            for link, mesh in meshes.items():
                if link not in link_T:
                    continue
                R_link, t_link = link_T[link]
                # Compose: world = root . link . visual_origin . mesh_local
                R_full = R_root @ R_link @ mesh["vorigin_R"]
                t_full = R_root @ (R_link @ mesh["vorigin_xyz"] + t_link) + t_root
                world_v = mesh["vertices"] @ R_full.T + t_full
                link_meshes[link] = world_v
                # Update bbox
                vmin = world_v.min(0)
                vmax = world_v.max(0)
                for d in range(3):
                    lo, hi = bounds_per_env[n][d]
                    bounds_per_env[n][d] = (min(lo, vmin[d]), max(hi, vmax[d]))
            env_data.append(link_meshes)
        world_verts.append(env_data)
        if ti % 100 == 0:
            print(f"[render]   FK frame {ti}/{T}", flush=True)

    # Pad bbox slightly
    pad = 0.3
    for n in range(N_show):
        bounds_per_env[n] = [(lo - pad, hi + pad)
                             for lo, hi in bounds_per_env[n]]

    # Render frames to mp4
    print(f"[render] writing video {output_path} @ {out_fps} fps")
    writer = imageio.get_writer(
        str(output_path), fps=out_fps, codec="libx264",
        quality=8, macro_block_size=1,
    )

    for ti in range(T):
        for n in range(N_show):
            ax = axes[n]
            ax.clear()
            ax.set_facecolor("#e8eef4")
            link_meshes = world_verts[ti][n]
            for link, world_v in link_meshes.items():
                faces = meshes[link]["faces"]
                tris = world_v[faces]  # (F, 3, 3)
                # Compute face normals for shading
                e1 = tris[:, 1] - tris[:, 0]
                e2 = tris[:, 2] - tris[:, 0]
                normals = np.cross(e1, e2)
                lengths = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
                normals = normals / lengths
                shade_factor = shade(normals, light_dir)
                base_rgb = np.array([0.78, 0.58, 0.38])  # warm tan
                colors = base_rgb[None, :] * shade_factor[:, None]
                colors = np.clip(colors, 0, 1)
                # alpha column
                colors_rgba = np.hstack([colors, np.ones((len(colors), 1))])
                pc = Poly3DCollection(
                    tris, facecolors=colors_rgba,
                    edgecolors=(0.3, 0.2, 0.1, 0.4), linewidths=0.15,
                )
                ax.add_collection3d(pc)
            # Ground plane (light gray rectangle at z=0)
            xlo, xhi = bounds_per_env[n][0]
            ylo, yhi = bounds_per_env[n][1]
            ground = np.array([
                [xlo, ylo, 0], [xhi, ylo, 0], [xhi, yhi, 0], [xlo, yhi, 0]
            ])
            ax.add_collection3d(Poly3DCollection(
                [ground], facecolors=(0.85, 0.85, 0.85, 0.7),
                edgecolors=(0, 0, 0, 0.0),
            ))
            # Set view
            xlim = bounds_per_env[n][0]
            ylim = bounds_per_env[n][1]
            zlim = bounds_per_env[n][2]
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
            ax.set_box_aspect((xlim[1] - xlim[0],
                               ylim[1] - ylim[0],
                               zlim[1] - zlim[0]))
            ax.view_init(elev=args.elev, azim=args.azim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            t_seconds = ti * stride / fps_in
            ax.set_title(f"env {n} — t={t_seconds:.2f}s")
        fig.suptitle(f"r15 model_2999 playback — frame {ti*stride}/{T_full} "
                     f"({ti*stride/fps_in:.2f}s of {T_full/fps_in:.2f}s)",
                     fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(buf[..., :3].copy())
        if ti % 50 == 0:
            print(f"[render]   wrote frame {ti}/{T}", flush=True)
    writer.close()
    plt.close(fig)
    print(f"[render] done: {output_path}")


if __name__ == "__main__":
    main()
