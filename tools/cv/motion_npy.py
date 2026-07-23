"""Read/write AMP reference-motion ``.npy`` clips, with the conventions VERIFIED
against the installed ``amp_rsl_rl`` rather than assumed.

Verification done 2026-07-23 by reading
``site-packages/amp_rsl_rl/utils/motion_loader.py``:

* ``root_quaternion`` is **xyzw** (SciPy order). ``_resample_data_SO3`` calls
  ``Rotation.from_quat(raw_quaternions)`` with the comment *"the quaternion is
  expected in the dataset as xyzw format (SciPy default)"*. The
  ``docs/CV_PIPELINE_HANDOFF.md`` spec saying "wxyz" is WRONG -- wxyz is what
  ``MotionData.base_quat`` exposes AFTER internal conversion, not what the file
  holds.
* ``joint_velocities`` in the file is **ignored** by the loader: it resamples
  ``joint_positions`` to the sim timestep and recomputes velocities by finite
  difference. We still write it (the format documents it, and our own
  validation uses it), but do not tune it.
* ``root_position`` is world-frame; linear velocity is derived from it by
  finite difference, then rotated into the body frame by ``R.T @ v``.
* Joint alignment is BY NAME (``dataset_joint_names.index(j)``), and any
  expected joint missing from ``joints_list`` is silently filled with
  **ZEROS** -- no warning. See ``check_joint_alignment`` below; this is the
  single most dangerous failure mode in this format.

Usage:
    python tools/cv/motion_npy.py --check assets/motions/deer_walk.npy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deer_rig import DeerRig

# Spawn rotation Rx(-90 deg) in xyzw -- maps URDF body -Z (anatomical forward)
# to world -Y, matching tools/bake_reset_states.py.
SPAWN_QUAT_XYZW = np.array([-0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)

REQUIRED_KEYS = (
    "joints_list",
    "joint_positions",
    "joint_velocities",
    "root_position",
    "root_quaternion",
    "fps",
)


@dataclass
class MotionClip:
    joints_list: list[str]
    joint_positions: np.ndarray  # (T, J) radians
    joint_velocities: np.ndarray  # (T, J) rad/s
    root_position: np.ndarray  # (T, 3) world, metres
    root_quaternion: np.ndarray  # (T, 4) XYZW
    fps: float

    @property
    def n_frames(self) -> int:
        return int(self.joint_positions.shape[0])

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    # -- io ---------------------------------------------------------------

    @staticmethod
    def load(path: str | Path) -> "MotionClip":
        d = np.load(str(path), allow_pickle=True).item()
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}")
        return MotionClip(
            joints_list=[str(j) for j in d["joints_list"]],
            joint_positions=np.asarray(d["joint_positions"], dtype=np.float32),
            joint_velocities=np.asarray(d["joint_velocities"], dtype=np.float32),
            root_position=np.asarray(d["root_position"], dtype=np.float32),
            root_quaternion=np.asarray(d["root_quaternion"], dtype=np.float32),
            fps=float(np.asarray(d["fps"]).reshape(-1)[0]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            str(path),
            {
                "joints_list": list(self.joints_list),
                "joint_positions": self.joint_positions.astype(np.float32),
                "joint_velocities": self.joint_velocities.astype(np.float32),
                "root_position": self.root_position.astype(np.float32),
                "root_quaternion": self.root_quaternion.astype(np.float32),
                "fps": float(self.fps),
            },
            allow_pickle=True,
        )

    # -- construction -----------------------------------------------------

    @staticmethod
    def from_joint_trajectory(
        joints_list: list[str],
        joint_positions: np.ndarray,
        fps: float,
        root_position: np.ndarray | None = None,
        root_quaternion_xyzw: np.ndarray | None = None,
        cyclic: bool = False,
    ) -> "MotionClip":
        """Build a clip, deriving joint velocities by central difference.

        ``cyclic=True`` wraps the difference at the clip boundary (use for a
        single gait cycle); otherwise the endpoints use one-sided differences.
        """
        jp = np.asarray(joint_positions, dtype=np.float32)
        T = jp.shape[0]
        dt = 1.0 / float(fps)
        jv = np.zeros_like(jp)
        if cyclic:
            last = T - 2 if T > 1 and np.allclose(jp[0], jp[-1], atol=1e-4) else T - 1
            m = last + 1
            for f in range(T):
                jv[f] = (jp[(f + 1) % m] - jp[(f - 1) % m]) / (2 * dt)
        else:
            jv[1:-1] = (jp[2:] - jp[:-2]) / (2 * dt)
            if T > 1:
                jv[0] = (jp[1] - jp[0]) / dt
                jv[-1] = (jp[-1] - jp[-2]) / dt

        rp = (
            np.zeros((T, 3), dtype=np.float32)
            if root_position is None
            else np.asarray(root_position, dtype=np.float32)
        )
        rq = (
            np.tile(SPAWN_QUAT_XYZW, (T, 1))
            if root_quaternion_xyzw is None
            else np.asarray(root_quaternion_xyzw, dtype=np.float32)
        )
        return MotionClip(list(joints_list), jp, jv, rp, rq, float(fps))


# ---------------------------------------------------------------------------
# Alignment + integrity checks
# ---------------------------------------------------------------------------


def check_joint_alignment(clip: MotionClip, expected_joint_names: list[str]) -> dict:
    """Replicate ``AMPLoader.load_data``'s name-based index map.

    Returns ``{"matched": [...], "zero_filled": [...], "unused": [...]}``.
    Anything in ``zero_filled`` will be fed to the discriminator as a CONSTANT
    ZERO reference signal, silently.
    """
    ds = list(clip.joints_list)
    matched = [j for j in expected_joint_names if j in ds]
    zero_filled = [j for j in expected_joint_names if j not in ds]
    unused = [j for j in ds if j not in expected_joint_names]
    return {"matched": matched, "zero_filled": zero_filled, "unused": unused}


def check_clip(clip: MotionClip, rig: DeerRig | None = None) -> list[str]:
    """Structural + physical sanity checks. Returns a list of problem strings."""
    problems: list[str] = []
    T, J = clip.joint_positions.shape

    if len(clip.joints_list) != J:
        problems.append(f"joints_list has {len(clip.joints_list)} names but joint_positions has {J} columns")
    for name, arr, shape in (
        ("joint_velocities", clip.joint_velocities, (T, J)),
        ("root_position", clip.root_position, (T, 3)),
        ("root_quaternion", clip.root_quaternion, (T, 4)),
    ):
        if arr.shape != shape:
            problems.append(f"{name} shape {arr.shape}, expected {shape}")
    for name, arr in (
        ("joint_positions", clip.joint_positions),
        ("root_position", clip.root_position),
        ("root_quaternion", clip.root_quaternion),
    ):
        if not np.isfinite(arr).all():
            problems.append(f"{name} contains non-finite values")

    if clip.root_quaternion.shape == (T, 4):
        n = np.linalg.norm(clip.root_quaternion, axis=1)
        if np.abs(n - 1.0).max() > 1e-3:
            problems.append(f"root_quaternion not unit-norm (max |1-|q|| = {np.abs(n - 1).max():.3e})")
        # xyzw-vs-wxyz smell test: our clips use a constant Rx(-90 deg) spawn
        # rotation, which in xyzw is (-0.707, 0, 0, +0.707). If the file were
        # wxyz the leading component would be the scalar.
        q0 = clip.root_quaternion[0]
        if abs(abs(q0[0]) - 0.7071) < 1e-2 and abs(q0[3] - 0.7071) < 1e-2:
            pass  # consistent with xyzw Rx(+-90)
        elif abs(abs(q0[0]) - 0.7071) < 1e-2 and abs(q0[1]) < 1e-3 and abs(q0[3]) < 1e-3:
            problems.append(
                f"root_quaternion[0]={np.round(q0, 4)} looks like WXYZ; the loader "
                "reads XYZW (scipy). Check the convention."
            )

    if rig is not None:
        unknown = [j for j in clip.joints_list if j not in rig.joints]
        if unknown:
            problems.append(f"joint names not present in {rig.urdf_path.name}: {unknown}")
        for i, j in enumerate(clip.joints_list):
            if j not in rig.joints:
                continue
            lo, hi = rig.joints[j].lower, rig.joints[j].upper
            col = clip.joint_positions[:, i]
            n_out = int(((col < lo - 1e-4) | (col > hi + 1e-4)).sum())
            if n_out:
                problems.append(
                    f"joint {j}: {n_out}/{T} frames outside URDF limits "
                    f"[{lo:.2f}, {hi:.2f}] (range {col.min():.2f}..{col.max():.2f})"
                )
            if float(col.max() - col.min()) < 1e-6:
                problems.append(f"joint {j}: CONSTANT across the clip (no reference signal)")

    return problems


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", required=True, help="path to a motion .npy")
    ap.add_argument("--urdf", default=None)
    ap.add_argument(
        "--expect-joints",
        default=None,
        help="comma-separated joint names the trainer will ask for "
        "(checks the silent zero-fill failure mode)",
    )
    args = ap.parse_args()

    clip = MotionClip.load(args.check)
    rig = DeerRig(args.urdf) if args.urdf else DeerRig()
    print(f"{args.check}")
    print(f"  frames {clip.n_frames} @ {clip.fps} fps = {clip.duration:.2f}s, {len(clip.joints_list)} joints")
    print(f"  root travel {np.round(clip.root_position[-1] - clip.root_position[0], 3)} m")
    print(f"  root_quaternion[0] (xyzw) {np.round(clip.root_quaternion[0], 4)}")

    problems = check_clip(clip, rig)
    if args.expect_joints:
        expected = [s.strip() for s in args.expect_joints.split(",") if s.strip()]
        al = check_joint_alignment(clip, expected)
        print(f"  alignment: {len(al['matched'])}/{len(expected)} matched")
        if al["zero_filled"]:
            problems.append(
                "SILENT ZERO-FILL: the trainer expects these joints but the clip "
                f"does not contain them -> constant-zero reference: {al['zero_filled']}"
            )
        if al["unused"]:
            problems.append(f"clip columns the trainer never reads: {al['unused']}")

    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\n  OK: no problems found")


if __name__ == "__main__":
    _main()
