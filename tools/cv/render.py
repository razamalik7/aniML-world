"""CPU-only pinhole camera + z-buffer rasterizer for the deer rig.

Why hand-rolled: the system python has no cv2 / pyrender / moderngl / torch,
and the GPU is reserved for locomotion training. numpy alone gets us ~640x480
renders at a usable rate, and -- more importantly -- gives us PART
SEGMENTATION and DEPTH for free, which Mu et al. (CVPR 2020) showed is worth
+1.4 PCK as a multi-task auxiliary.

Outputs per render:
    rgb   (H,W,3) uint8
    seg   (H,W)   int16   -1 = background, else index into ``part_names``
    depth (H,W)   float32 camera-space Z, +inf on background

Camera model: standard OpenCV pinhole.
    x_cam = R_cw @ (x_world - C)     R_cw rows = camera right/down/forward
    u = fx * x/z + cx ,  v = fy * y/z + cy
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_cw: np.ndarray  # (3,3) world -> camera rotation
    C: np.ndarray  # (3,) camera centre in world

    @staticmethod
    def look_at(
        eye,
        target,
        width: int = 640,
        height: int = 480,
        hfov_deg: float = 50.0,
        up=(0.0, 0.0, 1.0),
        roll_deg: float = 0.0,
    ) -> "Camera":
        """Build a camera at ``eye`` pointing at ``target`` (world Z-up)."""
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        fwd = target - eye
        n = np.linalg.norm(fwd)
        if n < 1e-9:
            raise ValueError("camera eye coincides with target")
        fwd = fwd / n
        up = np.asarray(up, dtype=np.float64)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:  # looking straight up/down
            right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        if roll_deg:
            a = math.radians(roll_deg)
            ca, sa = math.cos(a), math.sin(a)
            right, down = ca * right + sa * down, -sa * right + ca * down
        R_cw = np.stack([right, down, fwd])  # rows
        fx = 0.5 * width / math.tan(0.5 * math.radians(hfov_deg))
        return Camera(width, height, fx, fx, 0.5 * width, 0.5 * height, R_cw, eye)

    # -- projection ------------------------------------------------------

    def to_cam(self, pts_world: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_world, dtype=np.float64))
        return (p - self.C) @ self.R_cw.T

    def project(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (uv (N,2), z_cam (N,)). Points behind the camera get
        z<=0; callers must check. uv is NOT clipped to the image."""
        pc = self.to_cam(pts_world)
        z = pc[:, 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.fx * pc[:, 0] / safe + self.cx
        v = self.fy * pc[:, 1] / safe + self.cy
        return np.stack([u, v], axis=1), z

    def in_frame(self, uv: np.ndarray, z: np.ndarray, margin: int = 0) -> np.ndarray:
        return (
            (z > 1e-6)
            & (uv[:, 0] >= -margin)
            & (uv[:, 0] < self.width + margin)
            & (uv[:, 1] >= -margin)
            & (uv[:, 1] < self.height + margin)
        )

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


# ---------------------------------------------------------------------------
# Rasterizer
# ---------------------------------------------------------------------------


def rasterize(
    cam: Camera,
    parts: list[tuple[str, np.ndarray, np.ndarray]],
    colors: dict[str, np.ndarray] | None = None,
    background: np.ndarray | None = None,
    light_dir=(0.4, 0.3, 1.0),
    ambient: float = 0.35,
) -> dict:
    """Render a list of ``(part_name, vertices_world (V,3), faces (F,3))``.

    Pure numpy, per-triangle scanline over its bounding box with a z-buffer.
    Flat (per-face) Lambert shading -- our meshes are low-poly enough that
    smooth shading would be misleading anyway.
    """
    H, W = cam.height, cam.width
    depth = np.full((H, W), np.inf, dtype=np.float64)
    seg = np.full((H, W), -1, dtype=np.int16)
    shade_buf = np.zeros((H, W), dtype=np.float64)

    part_names = [p[0] for p in parts]
    light = np.asarray(light_dir, dtype=np.float64)
    light = light / (np.linalg.norm(light) + 1e-12)

    for pid, (_name, verts_w, faces) in enumerate(parts):
        if len(faces) == 0:
            continue
        uv, z = cam.project(verts_w)
        tri_uv = uv[faces]  # (F,3,2)
        tri_z = z[faces]  # (F,3)

        # Backface/behind culling + frustum reject
        keep = (tri_z > 1e-6).all(axis=1)
        if not keep.any():
            continue
        # World-space face normals for shading (before culling indices shift)
        tw = verts_w[faces]
        nrm = np.cross(tw[:, 1] - tw[:, 0], tw[:, 2] - tw[:, 0])
        nl = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.where(nl < 1e-12, 1.0, nl)
        face_shade = ambient + (1.0 - ambient) * np.clip(np.abs(nrm @ light), 0, 1)

        idx = np.flatnonzero(keep)
        for f in idx:
            p0, p1, p2 = tri_uv[f]
            z0, z1, z2 = tri_z[f]
            umin = int(max(0, math.floor(min(p0[0], p1[0], p2[0]))))
            umax = int(min(W - 1, math.ceil(max(p0[0], p1[0], p2[0]))))
            vmin = int(max(0, math.floor(min(p0[1], p1[1], p2[1]))))
            vmax = int(min(H - 1, math.ceil(max(p0[1], p1[1], p2[1]))))
            if umin > umax or vmin > vmax:
                continue
            area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
            if abs(area) < 1e-9:
                continue
            us = np.arange(umin, umax + 1) + 0.5
            vs = np.arange(vmin, vmax + 1) + 0.5
            U, V = np.meshgrid(us, vs)
            w0 = ((p1[0] - p0[0]) * (V - p0[1]) - (U - p0[0]) * (p1[1] - p0[1])) / area
            w1 = ((U - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (V - p0[1])) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not inside.any():
                continue
            # Perspective-correct depth
            iz = w2 / z0 + w1 / z1 + w0 / z2
            with np.errstate(divide="ignore", invalid="ignore"):
                zz = np.where(iz > 1e-12, 1.0 / iz, np.inf)
            sub_d = depth[vmin : vmax + 1, umin : umax + 1]
            win = inside & (zz < sub_d)
            if not win.any():
                continue
            sub_d[win] = zz[win]
            seg[vmin : vmax + 1, umin : umax + 1][win] = pid
            shade_buf[vmin : vmax + 1, umin : umax + 1][win] = face_shade[f]

    # Compose RGB
    if background is None:
        rgb = np.full((H, W, 3), 210, dtype=np.uint8)
    else:
        bg = np.asarray(background)
        if bg.shape[:2] != (H, W):
            raise ValueError(f"background must be {(H, W)}, got {bg.shape[:2]}")
        rgb = bg[..., :3].astype(np.uint8).copy()

    if colors is None:
        colors = {}
    fg = seg >= 0
    if fg.any():
        pal = np.array(
            [colors.get(n, np.array([0.62, 0.46, 0.30])) for n in part_names],
            dtype=np.float64,
        )
        col = pal[seg[fg]] * shade_buf[fg][:, None]
        rgb[fg] = np.clip(col * 255.0, 0, 255).astype(np.uint8)

    return {
        "rgb": rgb,
        "seg": seg,
        "depth": depth.astype(np.float32),
        "part_names": part_names,
        "mask": fg,
    }


def render_rig(
    rig,
    cam: Camera,
    q=None,
    root_pos=None,
    root_quat_xyzw=None,
    colors: dict | None = None,
    background: np.ndarray | None = None,
) -> dict:
    """Convenience: FK a ``DeerRig`` and rasterize all its visual meshes."""
    T = rig.fk_world(q, root_pos, root_quat_xyzw)
    parts = []
    for link in sorted(rig.visuals):
        mesh = rig.load_mesh(link)
        if mesh is None or link not in T:
            continue
        R, t = T[link]
        parts.append((link, mesh["vertices"] @ R.T + t, mesh["faces"]))
    return rasterize(cam, parts, colors=colors, background=background)
