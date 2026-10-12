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
      origin_map[joint] = (xyz, rpy, parent_link, child_link, axis)
      visual_map[link]  = (mesh_filename, xyz, rpy)   for each link with visual

    URDF v2 has joints on multiple axes (X for flex/extension, Y for spine
    yaw + neck yaw, Z for hip abduction), so we have to read the <axis> tag
    per joint instead of hard-coding X. Default if missing: (1, 0, 0).
    """
    text = urdf_path.read_text(encoding="utf-8")

    # Joints: parent + child + origin (xyz, rpy) + axis xyz.
    # Use a permissive structure: scan the whole <joint>...</joint> block
    # and pick out parent/child/origin/axis sub-elements individually so we
    # don't depend on element ordering.
    joint_block_pat = re.compile(
        r'<joint\s+name="(?P<name>[^"]+)"[^>]*>(?P<body>.*?)</joint>',
        re.DOTALL,
    )
    parent_pat = re.compile(r'<parent\s+link="(?P<v>[^"]+)"\s*/>')
    child_pat = re.compile(r'<child\s+link="(?P<v>[^"]+)"\s*/>')
    origin_pat = re.compile(
        r'<origin\s+xyz="(?P<xyz>[^"]+)"(?:\s+rpy="(?P<rpy>[^"]+)")?[^>]*/>'
    )
    axis_pat = re.compile(r'<axis\s+xyz="(?P<v>[^"]+)"\s*/>')
    origin_map = {}
    for m in joint_block_pat.finditer(text):
        name = m.group("name")
        body = m.group("body")
        pm = parent_pat.search(body)
        cm = child_pat.search(body)
        om = origin_pat.search(body)
        am = axis_pat.search(body)
        if pm is None or cm is None or om is None:
            continue
        parent = pm.group("v")
        child = cm.group("v")
        xyz = np.array(
            [float(v) for v in om.group("xyz").split()], dtype=np.float64
        )
        rpy_str = om.group("rpy") or "0 0 0"
        rpy = np.array([float(v) for v in rpy_str.split()], dtype=np.float64)
        if am is not None:
            axis = np.array(
                [float(v) for v in am.group("v").split()], dtype=np.float64
            )
            n = np.linalg.norm(axis)
            axis = axis / n if n > 0 else np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([1.0, 0.0, 0.0])
        origin_map[name] = (xyz, rpy, parent, child, axis)

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


def rot_axis(theta: float, axis: np.ndarray) -> np.ndarray:
    """Rodrigues' rotation about an arbitrary unit axis."""
    ux, uy, uz = float(axis[0]), float(axis[1]), float(axis[2])
    c = math.cos(theta)
    s = math.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + ux*ux*C,    ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c + uy*uy*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c + uz*uz*C],
    ], dtype=np.float64)


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def compute_link_transforms(joint_angles: dict, origin_map: dict, root: str | None = None) -> dict:
    """Return link_T[link] = (3x3 R, 3 t) in the URDF root frame.

    Root link is auto-detected as the link that is a parent but never a child
    (works for both the old lowercase 'pelvis' rig and the polyart 'Pelvis' rig).
    """
    if root is None:
        children = {c for (_, (_, _, _, c, _)) in origin_map.items()}
        parents = {p for (_, (_, _, p, _, _)) in origin_map.items()}
        roots = list(parents - children)
        root = roots[0] if roots else "pelvis"
    link_T = {root: (np.eye(3), np.zeros(3))}
    placed = {root}
    pending = list(origin_map.items())
    for _ in range(len(pending) + 5):
        progress = False
        remaining = []
        for jname, (xyz, rpy, parent, child, axis) in pending:
            if parent not in placed:
                remaining.append((jname, (xyz, rpy, parent, child, axis)))
                continue
            R_parent, t_parent = link_T[parent]
            # Joint frame relative to parent: translation xyz + rotation rpy +
            # joint rotation about the URDF-declared axis (URDF v2 has joints
            # on X / Y / Z axes, so we cannot hard-code rot_x here).
            R_joint_origin = rpy_to_R(rpy)
            R_joint = rot_axis(joint_angles.get(jname, 0.0), axis)
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
    parser.add_argument("--urdf", type=str, default=None,
                        help="Override the rig used for FK. Needed to render "
                             "historical npz dumps recorded on an older rig "
                             "(e.g. assets/deer_urdf/deer.urdf for the April "
                             "AMP runs). Defaults to the current deer_world.urdf.")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max_envs", type=int, default=2,
                        help="Max envs to render side-by-side. 1-2 keeps "
                             "matplotlib fast; 4 is doable but slow.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth frame (1=all, 2=half, etc.). "
                             "Lower fps if you stride.")
    parser.add_argument("--azim", type=float, default=0,
                        help="View azimuth degrees. 0 = pure side view "
                             "(camera at world +X, deer's left side). "
                             "After URDF Rx(-90) spawn rotation the deer "
                             "faces world -Y, so azim=0 shows leg swings "
                             "clearly with forward motion going right -> left.")
    parser.add_argument("--elev", type=float, default=12,
                        help="View elevation degrees. ~12 keeps the legs "
                             "visible without making it look top-down.")
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

    # --urdf lets us render HISTORICAL npz dumps whose joints_list uses an
    # older rig's joint names (e.g. the April r38-r46 AMP runs, which used
    # assets/deer_urdf/deer.urdf with 15 pre-v2 joint names). Defaults to the
    # current rig so existing usage is unchanged.
    urdf_path = Path(args.urdf) if args.urdf else URDF_PATH
    print(f"[render] parsing URDF {urdf_path}")
    origin_map, visual_map = parse_urdf_full(urdf_path)
    urdf_dir = urdf_path.parent
    meshes = load_meshes(visual_map, urdf_dir)
    print(f"[render] loaded {len(meshes)} link meshes "
          f"(of {len(visual_map)} URDF visuals)")

    # REFUSE TO RENDER A MISMATCHED RIG. `--urdf` defaults to the OLD 15-joint
    # deer_world.urdf, so rendering a polyart rollout without passing --urdf
    # silently maps 38 joint angles onto a 16-link skeleton and draws a pile of
    # scattered body parts that looks exactly like a catastrophic policy failure.
    # It cost a round of "the deer is balancing on its nose" before anyone
    # checked which URDF had been loaded. A rollout's joints_list is the ground
    # truth for which rig produced it -- if the URDF doesn't provide those
    # joints, the picture is meaningless, so do not draw it.
    missing = [j for j in joints_list if j not in origin_map]
    if missing:
        raise SystemExit(
            f"[render] RIG MISMATCH: {urdf_path.name} is missing "
            f"{len(missing)}/{len(joints_list)} of the rollout's joints "
            f"(e.g. {missing[:4]}).\n"
            f"          The rollout was recorded on a different rig. Pass the "
            f"matching --urdf, e.g.\n"
            f"          --urdf assets/deer_polyart_urdf/deer_polyart.urdf"
        )

    # Set up figure: one 3D axis per env
    fig = plt.figure(figsize=(6 * N_show, 6))
    axes = [fig.add_subplot(1, N_show, n + 1, projection="3d")
            for n in range(N_show)]

    # Pre-compute world-frame mesh vertices per (frame, env, link)
    # This is memory-heavy; for 1500 frames * 4 envs * ~16 links * ~50 verts,
    # that's ~5M floats — fine.
    print(f"[render] computing FK for {T}*{N_show} = {T*N_show} configs ...")
    world_verts: list = []  # world_verts[t][n][link] = (V,3) vertices
    # Per-frame deer bounding box (so camera can FOLLOW the deer
    # instead of using a single box for the whole trajectory — that
    # made the deer a tiny speck inside a 10m-wide frame).
    per_frame_bbox = [[None] * N_show for _ in range(T)]

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
            frame_min = np.array([np.inf, np.inf, np.inf])
            frame_max = np.array([-np.inf, -np.inf, -np.inf])
            for link, mesh in meshes.items():
                if link not in link_T:
                    continue
                R_link, t_link = link_T[link]
                # Compose: world = root . link . visual_origin . mesh_local
                R_full = R_root @ R_link @ mesh["vorigin_R"]
                t_full = R_root @ (R_link @ mesh["vorigin_xyz"] + t_link) + t_root
                world_v = mesh["vertices"] @ R_full.T + t_full
                link_meshes[link] = world_v
                frame_min = np.minimum(frame_min, world_v.min(0))
                frame_max = np.maximum(frame_max, world_v.max(0))
            per_frame_bbox[ti][n] = (frame_min, frame_max)
            env_data.append(link_meshes)
        world_verts.append(env_data)
        if ti % 100 == 0:
            print(f"[render]   FK frame {ti}/{T}", flush=True)

    # Smooth the per-frame bbox center across a small temporal window so the
    # camera doesn't jitter, and pin a fixed window size so the deer keeps
    # a consistent on-screen scale.
    smooth_win = max(5, int(out_fps * 0.4))   # ~0.4s window
    centers = np.zeros((T, N_show, 3))
    for ti in range(T):
        for n in range(N_show):
            mn, mx = per_frame_bbox[ti][n]
            centers[ti, n] = 0.5 * (mn + mx)
    # Box-filter centers along time
    if smooth_win > 1 and T > smooth_win:
        kernel = np.ones(smooth_win) / smooth_win
        for n in range(N_show):
            for d in range(3):
                centers[:, n, d] = np.convolve(
                    centers[:, n, d], kernel, mode="same"
                )

    # Fixed half-window size around the deer per axis (meters).
    half_win = np.array([1.6, 1.6, 1.0])    # ~3.2x3.2x2.0m view box

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
            # Camera follows deer: per-frame center, fixed half-window.
            cx, cy, cz = centers[ti, n]
            xlo, xhi = cx - half_win[0], cx + half_win[0]
            ylo, yhi = cy - half_win[1], cy + half_win[1]
            # Ground plane: very faint so it doesn't sort over the feet.
            ground = np.array([
                [xlo, ylo, 0], [xhi, ylo, 0], [xhi, yhi, 0], [xlo, yhi, 0]
            ])
            ground_pc = Poly3DCollection(
                [ground], facecolors=(0.82, 0.82, 0.84, 0.30),
                edgecolors=(0, 0, 0, 0.0),
            )
            ground_pc.set_zorder(-10)
            ax.add_collection3d(ground_pc)
            # Grid lines on ground in WORLD coords, every 0.5 m. With camera
            # tracking the deer, these lines slide past underneath as it
            # walks — gives you the visual cue that motion is happening
            # (without grid the tracking cam looks like a treadmill).
            grid_step = 0.5
            gx_lo = grid_step * math.floor(xlo / grid_step)
            gx_hi = grid_step * math.ceil(xhi / grid_step)
            gy_lo = grid_step * math.floor(ylo / grid_step)
            gy_hi = grid_step * math.ceil(yhi / grid_step)
            grid_color = (0.45, 0.45, 0.50, 0.7)
            grid_lw = 0.8
            x = gx_lo
            while x <= gx_hi + 1e-9:
                ln = ax.plot([x, x], [ylo, yhi], [0, 0],
                             color=grid_color, linewidth=grid_lw)
                for L in ln:
                    L.set_zorder(-5)
                x += grid_step
            y = gy_lo
            while y <= gy_hi + 1e-9:
                ln = ax.plot([xlo, xhi], [y, y], [0, 0],
                             color=grid_color, linewidth=grid_lw)
                for L in ln:
                    L.set_zorder(-5)
                y += grid_step
            # Pelvis-trail: line of every-Nth past pelvis position so the
            # forward progress is unmistakable even from a fixed-window cam.
            past_centers = centers[max(0, ti - 200):ti + 1, n]
            if len(past_centers) >= 2:
                trail_z = np.zeros(len(past_centers))
                trail = ax.plot(
                    past_centers[:, 0], past_centers[:, 1], trail_z,
                    color=(0.20, 0.40, 0.85), linewidth=2.5, alpha=0.9,
                )
                # Bring trail in front of ground
                for line in trail:
                    line.set_zorder(5)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            # Z window: feet near z=0, head ~1.4m. Frame from z=-0.05 (a hair
            # below floor so feet don't touch the bottom) to z=1.8m.
            ax.set_zlim(-0.05, 1.8)
            ax.set_box_aspect((2 * half_win[0], 2 * half_win[1], 1.85))
            ax.view_init(elev=args.elev, azim=args.azim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            # Hide the three axis panes (back walls of the 3D box). Without
            # this, as the deer walks toward a pane, matplotlib's painter
            # algorithm mis-sorts and the white pane draws OVER the deer —
            # what you see as "the deer walking behind a white wall".
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor((0, 0, 0, 0))
            ax.grid(False)
            t_seconds = ti * stride / fps_in
            ax.set_title(f"env {n} — t={t_seconds:.2f}s")
        fig.suptitle(f"AMP deer playback — {input_path.stem} — "
                     f"{ti*stride/fps_in:.2f}s of {T_full/fps_in:.2f}s",
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
