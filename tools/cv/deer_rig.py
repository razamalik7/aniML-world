"""Shared rig model: URDF parsing, forward kinematics, keypoint extraction.

This is the single source of truth for "where is link L / keypoint K in world
space, given joint angles + root pose" for the whole CV workstream. Everything
else in ``tools/cv/`` builds on it.

It deliberately duplicates the URDF parser + FK from
``tools/render_playback_meshed.py`` rather than importing it, because that
module is a CLI script owned by the playback workflow and imports matplotlib
at module scope. Keep the two FK implementations numerically identical --
``tests_fk_parity()`` below checks that.

ANATOMY / AXIS CONVENTIONS (see CLAUDE.md -- these are NOT Isaac Lab's):
    body +X = anatomical LEFT
    body +Y = anatomical DOWN   (feet ~+0.6 below pelvis)
    body -Z = anatomical FORWARD (skull ~-0.94)
The Rx(-90 deg) spawn rotation maps body -Z -> world -Y, so a deer walking
forward in Isaac travels toward world -Y.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "assets" / "deer_urdf" / "deer_world.urdf"

ROOT_LINK = "pelvis"

# Anatomical unit vectors in the URDF body frame.
AX_LEFT = np.array([1.0, 0.0, 0.0])
AX_DOWN = np.array([0.0, 1.0, 0.0])
AX_FORWARD = np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float

    @property
    def is_actuated(self) -> bool:
        return self.jtype in ("revolute", "continuous", "prismatic")


@dataclass
class Visual:
    link: str
    mesh_file: str
    xyz: np.ndarray
    rpy: np.ndarray


_JOINT_BLOCK = re.compile(
    r'<joint\s+name="(?P<name>[^"]+)"\s+type="(?P<type>[^"]+)"\s*>'
    r"(?P<body>.*?)</joint>",
    re.DOTALL,
)
_LINK_BLOCK = re.compile(r'<link\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</link>', re.DOTALL)
_PARENT = re.compile(r'<parent\s+link="([^"]+)"')
_CHILD = re.compile(r'<child\s+link="([^"]+)"')
_ORIGIN = re.compile(r'<origin\s+xyz="([^"]+)"(?:\s+rpy="([^"]+)")?')
_AXIS = re.compile(r'<axis\s+xyz="([^"]+)"')
_LIMIT = re.compile(r'<limit[^>]*lower="([^"]+)"\s+upper="([^"]+)"')
_VISUAL = re.compile(
    r"<visual[^>]*>\s*"
    r'<origin\s+xyz="(?P<xyz>[^"]+)"(?:\s+rpy="(?P<rpy>[^"]+)")?[^>]*/>\s*'
    r"<geometry>\s*<mesh\s+"
    r'filename="(?P<file>[^"]+)"',
    re.DOTALL,
)


def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in text.split()], dtype=np.float64)


class DeerRig:
    """Parsed URDF + forward kinematics + keypoint extraction."""

    def __init__(self, urdf_path: Path | str = DEFAULT_URDF):
        self.urdf_path = Path(urdf_path)
        text = self.urdf_path.read_text(encoding="utf-8")

        self.joints: dict[str, Joint] = {}
        for m in _JOINT_BLOCK.finditer(text):
            body = m.group("body")
            pm, cm, om = _PARENT.search(body), _CHILD.search(body), _ORIGIN.search(body)
            if not (pm and cm and om):
                continue
            am = _AXIS.search(body)
            axis = _vec(am.group(1), (1.0, 0.0, 0.0)) if am else np.array([1.0, 0.0, 0.0])
            n = np.linalg.norm(axis)
            axis = axis / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
            lm = _LIMIT.search(body)
            self.joints[m.group("name")] = Joint(
                name=m.group("name"),
                jtype=m.group("type"),
                parent=pm.group(1),
                child=cm.group(1),
                xyz=_vec(om.group(1)),
                rpy=_vec(om.group(2)),
                axis=axis,
                lower=float(lm.group(1)) if lm else -math.pi,
                upper=float(lm.group(2)) if lm else math.pi,
            )

        self.visuals: dict[str, Visual] = {}
        for lm in _LINK_BLOCK.finditer(text):
            vm = _VISUAL.search(lm.group("body"))
            if not vm:
                continue
            self.visuals[lm.group("name")] = Visual(
                link=lm.group("name"),
                mesh_file=vm.group("file"),
                xyz=_vec(vm.group("xyz")),
                rpy=_vec(vm.group("rpy")),
            )

        self.links: list[str] = sorted(
            {j.parent for j in self.joints.values()} | {j.child for j in self.joints.values()}
        )
        # Actuated joints in URDF declaration order == the rig's DOF ordering.
        self.dof_names: list[str] = [j.name for j in self.joints.values() if j.is_actuated]
        self._order = self._topological_joint_order()
        self._mesh_cache: dict[str, dict] = {}

    # -- topology ----------------------------------------------------------

    def _topological_joint_order(self) -> list[Joint]:
        placed = {ROOT_LINK}
        out: list[Joint] = []
        pending = list(self.joints.values())
        for _ in range(len(pending) + 2):
            rest = []
            for j in pending:
                if j.parent in placed:
                    out.append(j)
                    placed.add(j.child)
                else:
                    rest.append(j)
            pending = rest
            if not pending:
                return out
        raise RuntimeError(f"URDF is not a tree rooted at {ROOT_LINK}: {[j.name for j in pending]}")

    # -- kinematics --------------------------------------------------------

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.joints[n].lower for n in self.dof_names])
        hi = np.array([self.joints[n].upper for n in self.dof_names])
        return lo, hi

    def fk(self, q: dict[str, float] | np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Link transforms in the ROOT (pelvis) frame.

        ``q`` may be a name->angle dict, or an array ordered like
        ``self.dof_names``. Returns ``{link: (R 3x3, t 3)}``.
        """
        if q is None:
            qd: dict[str, float] = {}
        elif isinstance(q, dict):
            qd = q
        else:
            arr = np.asarray(q, dtype=np.float64).reshape(-1)
            if arr.shape[0] != len(self.dof_names):
                raise ValueError(f"q has {arr.shape[0]} values, rig has {len(self.dof_names)} DOF")
            qd = dict(zip(self.dof_names, arr))

        out = {ROOT_LINK: (np.eye(3), np.zeros(3))}
        for j in self._order:
            Rp, tp = out[j.parent]
            Rc = Rp @ rpy_to_R(j.rpy) @ rot_axis(float(qd.get(j.name, 0.0)), j.axis)
            tc = Rp @ j.xyz + tp
            out[j.child] = (Rc, tc)
        return out

    def fk_world(
        self,
        q=None,
        root_pos: np.ndarray | None = None,
        root_quat_xyzw: np.ndarray | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Link transforms in WORLD space.

        ``root_quat_xyzw`` is xyzw (SciPy / AMP-dataset convention -- see
        ``tools/cv/motion_npy.py`` for why).
        """
        link_T = self.fk(q)
        R0 = np.eye(3) if root_quat_xyzw is None else quat_xyzw_to_R(root_quat_xyzw)
        t0 = np.zeros(3) if root_pos is None else np.asarray(root_pos, dtype=np.float64)
        return {lk: (R0 @ R, R0 @ t + t0) for lk, (R, t) in link_T.items()}

    # -- meshes ------------------------------------------------------------

    def load_mesh(self, link: str) -> dict | None:
        """Load a link's visual mesh, pre-transformed by its <visual> origin.

        Returns ``{"vertices": (V,3), "faces": (F,3)}`` in LINK frame, or None
        if the link has no visual (see ``missing_visuals()``).
        """
        if link in self._mesh_cache:
            return self._mesh_cache[link]
        vis = self.visuals.get(link)
        if vis is None:
            self._mesh_cache[link] = None
            return None
        import trimesh  # local import: keeps deer_rig importable without trimesh

        path = self.urdf_path.parent / vis.mesh_file
        if not path.is_file():
            self._mesh_cache[link] = None
            return None
        mesh = trimesh.load_mesh(str(path), force="mesh")
        R = rpy_to_R(vis.rpy)
        verts = np.asarray(mesh.vertices, dtype=np.float64) @ R.T + vis.xyz
        entry = {"vertices": verts, "faces": np.asarray(mesh.faces, dtype=np.int32)}
        self._mesh_cache[link] = entry
        return entry

    def missing_visuals(self) -> list[str]:
        """Links that are part of the kinematic tree but have no visual mesh.

        As of 2026-07-23 this is the 4 toe links, the antlers, and the two
        zero-length yaw helper links. The toes and antlers matter: they are
        VISIBLE anatomy on a real deer, so synthetic renders are missing hooves
        and antlers entirely. Collision STLs exist for both
        (``*_toe_collision.stl``, ``antler_collision_v2.stl``) and should be
        promoted to visuals before generating a training set.
        """
        return [lk for lk in self.links if lk not in self.visuals]


# ---------------------------------------------------------------------------
# Rotation helpers (kept dependency-free and identical to render_playback_meshed)
# ---------------------------------------------------------------------------


def rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_axis(theta: float, axis: np.ndarray) -> np.ndarray:
    ux, uy, uz = (float(v) for v in axis)
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s],
            [uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s],
            [uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C],
        ]
    )


def quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_wxyz_to_xyzw(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def quat_xyzw_to_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:], q[..., :3]], axis=-1)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_test() -> None:
    rig = DeerRig()
    print(f"URDF        : {rig.urdf_path}")
    print(f"links       : {len(rig.links)}")
    print(f"joints      : {len(rig.joints)}  (actuated DOF: {len(rig.dof_names)})")
    print(f"visuals     : {len(rig.visuals)}")
    print(f"no visual   : {rig.missing_visuals()}")

    T = rig.fk()
    for lk in ("pelvis", "chest", "skull", "HL_foot", "FL_toe"):
        if lk in T:
            print(f"  rest {lk:10s} t = {np.round(T[lk][1], 4)}")

    # FK parity against the production renderer's implementation.
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from render_playback_meshed import compute_link_transforms, parse_urdf_full  # noqa: E402

    om, _ = parse_urdf_full(rig.urdf_path)
    rng = np.random.default_rng(0)
    lo, hi = rig.joint_limits()
    q = dict(zip(rig.dof_names, rng.uniform(lo, hi)))
    ours = rig.fk(q)
    theirs = compute_link_transforms(q, om)
    worst = max(
        float(np.abs(ours[lk][1] - theirs[lk][1]).max())
        for lk in theirs
        if lk in ours
    )
    print(f"FK parity vs render_playback_meshed: max |dt| = {worst:.3e}")
    assert worst < 1e-9, "FK MISMATCH -- do not trust downstream keypoints"
    print("OK")


if __name__ == "__main__":
    _self_test()
