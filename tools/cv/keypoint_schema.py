"""Keypoint schema for the 25-DOF deer rig.

A keypoint is a fixed point in some link's frame. Given joint angles + root
pose we can therefore compute its exact 3D world position (and, with a camera,
its exact 2D pixel position + visibility). That is the whole point: synthetic
renders come with perfect labels for free.

Two families:

* **JOINT keypoints** -- the origin of a child link, i.e. the rotation centre of
  the joint above it. Exact, stable, and they are what an IK fit actually wants
  to constrain. Defined declaratively below.
* **SURFACE keypoints** -- nose, eyes, tail root, hoof tips, antler tips. These
  have no joint to sit on, so they are derived as the extreme mesh vertex along
  an anatomical direction, evaluated at REST pose and then frozen into the
  owning link's local frame. Recomputed at construction (so they track mesh
  edits) and covered by ``_self_test``.

The schema is a superset of **AP-10K's 17 quadruped keypoints** so that
off-the-shelf detectors (ViTPose-AP10K, SuperAnimal-Quadruped) and any
hand-labelled real deer frames line up with our renders. ``AP10K_MAP`` gives
that correspondence.

AXIS REMINDER (URDF body frame, see CLAUDE.md):
    +X = anatomical LEFT,  +Y = anatomical DOWN,  -Z = anatomical FORWARD
so "up" = -Y, "forward" = -Z, "rearward" = +Z, "left" = +X.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deer_rig import AX_DOWN, AX_FORWARD, AX_LEFT, DeerRig

# ---------------------------------------------------------------------------
# Joint-centre keypoints: name -> link whose ORIGIN is the keypoint
# ---------------------------------------------------------------------------

JOINT_KEYPOINTS: dict[str, str] = {
    # axial
    "pelvis": "pelvis",
    "chest": "chest",
    "neck_base": "neck",
    "skull_base": "skull",
    # left / right, front / hind.  L/R are the ANIMAL's left and right.
    "L_shoulder": "FL_thigh",
    "L_elbow": "FL_shin",
    "L_f_ankle": "FL_foot",
    "L_f_toe": "FL_toe",
    "R_shoulder": "FR_thigh",
    "R_elbow": "FR_shin",
    "R_f_ankle": "FR_foot",
    "R_f_toe": "FR_toe",
    "L_hip": "HL_thigh",
    "L_knee": "HL_shin",
    "L_h_ankle": "HL_foot",
    "L_h_toe": "HL_toe",
    "R_hip": "HR_thigh",
    "R_knee": "HR_shin",
    "R_h_ankle": "HR_foot",
    "R_h_toe": "HR_toe",
}

# ---------------------------------------------------------------------------
# Surface keypoints: name -> (link, direction in ROOT frame to maximise)
# ---------------------------------------------------------------------------

_UP = -AX_DOWN
_REAR = -AX_FORWARD

SURFACE_KEYPOINTS: dict[str, tuple[str, np.ndarray]] = {
    "nose": ("skull", AX_FORWARD),
    "L_eye": ("skull", AX_FORWARD * 0.7 + _UP * 0.3 + AX_LEFT * 0.6),
    "R_eye": ("skull", AX_FORWARD * 0.7 + _UP * 0.3 - AX_LEFT * 0.6),
    "L_antler_tip": ("skull", _UP * 0.8 + AX_LEFT * 0.6),
    "R_antler_tip": ("skull", _UP * 0.8 - AX_LEFT * 0.6),
    "withers": ("chest", _UP),
    "tail_root": ("pelvis", _REAR * 0.8 + _UP * 0.4),
    # Hoof tips fall back to the FOOT mesh because the toe links currently have
    # no visual mesh (see DeerRig.missing_visuals). Once *_toe_collision.stl is
    # promoted to a visual, retarget these to the toe links.
    "L_f_hoof": ("FL_foot", AX_DOWN),
    "R_f_hoof": ("FR_foot", AX_DOWN),
    "L_h_hoof": ("HL_foot", AX_DOWN),
    "R_h_hoof": ("HR_foot", AX_DOWN),
}

# ---------------------------------------------------------------------------
# Skeleton edges (for drawing + for bone-length validation)
# ---------------------------------------------------------------------------

SKELETON: list[tuple[str, str]] = [
    ("tail_root", "pelvis"),
    ("pelvis", "chest"),
    ("chest", "neck_base"),
    ("neck_base", "skull_base"),
    ("skull_base", "nose"),
    ("skull_base", "L_eye"),
    ("skull_base", "R_eye"),
    ("skull_base", "L_antler_tip"),
    ("skull_base", "R_antler_tip"),
    ("chest", "withers"),
    ("chest", "L_shoulder"),
    ("L_shoulder", "L_elbow"),
    ("L_elbow", "L_f_ankle"),
    ("L_f_ankle", "L_f_toe"),
    ("L_f_toe", "L_f_hoof"),
    ("chest", "R_shoulder"),
    ("R_shoulder", "R_elbow"),
    ("R_elbow", "R_f_ankle"),
    ("R_f_ankle", "R_f_toe"),
    ("R_f_toe", "R_f_hoof"),
    ("pelvis", "L_hip"),
    ("L_hip", "L_knee"),
    ("L_knee", "L_h_ankle"),
    ("L_h_ankle", "L_h_toe"),
    ("L_h_toe", "L_h_hoof"),
    ("pelvis", "R_hip"),
    ("R_hip", "R_knee"),
    ("R_knee", "R_h_ankle"),
    ("R_h_ankle", "R_h_toe"),
    ("R_h_toe", "R_h_hoof"),
]

# Left/right mirror pairs -- used for symmetry augmentation and for catching
# left/right swaps, which are the classic animal-pose failure mode.
FLIP_PAIRS: list[tuple[str, str]] = [
    ("L_eye", "R_eye"),
    ("L_antler_tip", "R_antler_tip"),
    ("L_shoulder", "R_shoulder"),
    ("L_elbow", "R_elbow"),
    ("L_f_ankle", "R_f_ankle"),
    ("L_f_toe", "R_f_toe"),
    ("L_f_hoof", "R_f_hoof"),
    ("L_hip", "R_hip"),
    ("L_knee", "R_knee"),
    ("L_h_ankle", "R_h_ankle"),
    ("L_h_toe", "R_h_toe"),
    ("L_h_hoof", "R_h_hoof"),
]

# AP-10K's 17 keypoints, in their canonical order, mapped onto our names.
# Lets us (a) evaluate a pretrained AP-10K detector on our renders and
# (b) ingest hand-labelled real deer frames in a standard tool.
AP10K_ORDER: list[str] = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
AP10K_MAP: dict[str, str] = {
    "L_Eye": "L_eye", "R_Eye": "R_eye", "Nose": "nose",
    "Neck": "neck_base", "Root_of_tail": "tail_root",
    "L_Shoulder": "L_shoulder", "L_Elbow": "L_elbow", "L_F_Paw": "L_f_hoof",
    "R_Shoulder": "R_shoulder", "R_Elbow": "R_elbow", "R_F_Paw": "R_f_hoof",
    "L_Hip": "L_hip", "L_Knee": "L_knee", "L_B_Paw": "L_h_hoof",
    "R_Hip": "R_hip", "R_Knee": "R_knee", "R_B_Paw": "R_h_hoof",
}

KEYPOINT_NAMES: list[str] = list(JOINT_KEYPOINTS) + list(SURFACE_KEYPOINTS)


@dataclass
class Keypoint:
    name: str
    link: str
    local: np.ndarray  # position in the link's local frame
    kind: str  # "joint" | "surface"


class DeerKeypoints:
    """Resolves the schema against a rig and evaluates keypoints under FK."""

    def __init__(self, rig: DeerRig | None = None):
        self.rig = rig or DeerRig()
        rest = self.rig.fk()
        self.keypoints: list[Keypoint] = []

        for name, link in JOINT_KEYPOINTS.items():
            if link not in rest:
                raise KeyError(f"keypoint {name!r} references unknown link {link!r}")
            self.keypoints.append(Keypoint(name, link, np.zeros(3), "joint"))

        for name, (link, direction) in SURFACE_KEYPOINTS.items():
            mesh = self.rig.load_mesh(link)
            if mesh is None:
                raise RuntimeError(
                    f"surface keypoint {name!r} needs a visual mesh on link "
                    f"{link!r}, which has none. See DeerRig.missing_visuals()."
                )
            R, _t = rest[link]
            d = np.asarray(direction, dtype=np.float64)
            d = d / np.linalg.norm(d)
            # Maximise along d expressed in the link's rest-pose local frame.
            d_local = R.T @ d
            v = mesh["vertices"]
            self.keypoints.append(
                Keypoint(name, link, v[int(np.argmax(v @ d_local))].copy(), "surface")
            )

        self.names: list[str] = [k.name for k in self.keypoints]
        self._index = {n: i for i, n in enumerate(self.names)}
        self.skeleton_idx = [
            (self._index[a], self._index[b])
            for a, b in SKELETON
            if a in self._index and b in self._index
        ]
        self.flip_idx = [
            (self._index[a], self._index[b])
            for a, b in FLIP_PAIRS
            if a in self._index and b in self._index
        ]

    def __len__(self) -> int:
        return len(self.keypoints)

    def index(self, name: str) -> int:
        return self._index[name]

    def positions(self, q=None, root_pos=None, root_quat_xyzw=None) -> np.ndarray:
        """(K,3) world positions of every keypoint for one pose."""
        T = self.rig.fk_world(q, root_pos, root_quat_xyzw)
        out = np.zeros((len(self.keypoints), 3))
        for i, kp in enumerate(self.keypoints):
            R, t = T[kp.link]
            out[i] = R @ kp.local + t
        return out

    def positions_seq(self, Q, root_pos=None, root_quat_xyzw=None) -> np.ndarray:
        """(T,K,3) world positions for a motion sequence."""
        Q = np.asarray(Q)
        T = len(Q)
        rp = np.zeros((T, 3)) if root_pos is None else np.asarray(root_pos)
        rq = (
            np.tile([0.0, 0.0, 0.0, 1.0], (T, 1))
            if root_quat_xyzw is None
            else np.asarray(root_quat_xyzw)
        )
        return np.stack([self.positions(Q[t], rp[t], rq[t]) for t in range(T)])

    def ap10k(self, world_kp: np.ndarray) -> np.ndarray:
        """Reorder a (K,3)/(T,K,3) array into AP-10K's 17-keypoint order."""
        idx = [self._index[AP10K_MAP[n]] for n in AP10K_ORDER]
        return np.asarray(world_kp)[..., idx, :]

    def bone_lengths(self, world_kp: np.ndarray) -> np.ndarray:
        """(..., n_bones) lengths of every skeleton edge. Rigid-body bones must
        be constant over time -- that's the bone-length consistency check."""
        w = np.asarray(world_kp)
        a = np.array([i for i, _ in self.skeleton_idx])
        b = np.array([j for _, j in self.skeleton_idx])
        return np.linalg.norm(w[..., a, :] - w[..., b, :], axis=-1)


def _self_test() -> None:
    kps = DeerKeypoints()
    print(f"keypoints: {len(kps)}  ({len(JOINT_KEYPOINTS)} joint + "
          f"{len(SURFACE_KEYPOINTS)} surface)")
    P = kps.positions()
    for n in ("nose", "withers", "tail_root", "L_h_hoof", "L_antler_tip", "pelvis"):
        print(f"  rest {n:14s} {np.round(P[kps.index(n)], 3)}")

    # Sanity: anatomy must be arranged correctly in the body frame.
    fwd = lambda n: -P[kps.index(n)][2]  # noqa: E731  (larger = more forward)
    up = lambda n: -P[kps.index(n)][1]  # noqa: E731
    assert fwd("nose") > fwd("withers") > fwd("tail_root"), "front/back order wrong"
    assert up("L_antler_tip") > up("withers") > up("L_h_hoof"), "up/down order wrong"
    assert P[kps.index("L_hip")][0] > P[kps.index("R_hip")][0], "left/right swapped"
    print("  anatomy ordering: OK")

    # Bone lengths must be invariant to joint angles and root pose.
    rig = kps.rig
    lo, hi = rig.joint_limits()
    rng = np.random.default_rng(1)
    L0 = kps.bone_lengths(kps.positions())
    for _ in range(5):
        q = rng.uniform(lo, hi)
        rp = rng.normal(0, 2, 3)
        rq = rng.normal(size=4)
        rq /= np.linalg.norm(rq)
        L = kps.bone_lengths(kps.positions(q, rp, rq))
        # Only bones that do NOT span an actuated joint are truly rigid; all of
        # them are rigid in the sense that |parent-child| is fixed by the URDF,
        # because a revolute joint rotates about a fixed origin.
        drift = float(np.abs(L - L0).max())
        assert drift < 1e-9, f"bone length drift {drift}"
    print(f"  bone-length invariance over random poses: OK ({len(L0)} bones)")
    print(f"  AP-10K reorder: {kps.ap10k(P).shape}")
    print("OK")


if __name__ == "__main__":
    _self_test()
