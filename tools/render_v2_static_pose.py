"""Generate a static-pose .npz for the URDF v2 deer and render it.

Used to visually verify the new 21-joint articulation:
  - frame bones don't introduce mesh distortion
  - HAA at ±0.3 rad still looks plausible (lateral leg spread)
  - spine yaw at ±0.2 rad still looks plausible (torso twist)
  - default standing pose looks anatomically reasonable

Reads the current URDF, builds a few keyframes at extreme HAA / yaw values
to stress-test the skinning, writes a .npz, then invokes the existing meshed
renderer.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
NPZ_OUT = REPO / "logs" / "videos" / "v2_static_pose.npz"
MP4_OUT = REPO / "logs" / "videos" / "v2_static_pose.mp4"

# 21 joint names in canonical (URDF v2) order — must match
# DEER_URDF_JOINT_NAMES in deer_usd_amp_env_cfg.py.
JOINTS = [
    "pelvis_to_chest_yaw",
    "chest_yaw_to_chest",
    "pelvis_to_HR_hipabd",
    "HR_hipabd_to_HR_thigh",
    "HR_thigh_to_HR_shin",
    "HR_shin_to_HR_foot",
    "pelvis_to_HL_hipabd",
    "HL_hipabd_to_HL_thigh",
    "HL_thigh_to_HL_shin",
    "HL_shin_to_HL_foot",
    "chest_to_FR_hipabd",
    "FR_hipabd_to_FR_thigh",
    "FR_thigh_to_FR_shin",
    "FR_shin_to_FR_foot",
    "chest_to_FL_hipabd",
    "FL_hipabd_to_FL_thigh",
    "FL_thigh_to_FL_shin",
    "FL_shin_to_FL_foot",
    "chest_to_neck_yaw",
    "neck_yaw_to_neck",
    "neck_to_skull",
]


def make_pose(j_overrides: dict[str, float]) -> np.ndarray:
    """Build a 21-vector of joint angles, defaults from articulation config."""
    base = {
        "HR_hipabd_to_HR_thigh":  0.4,
        "HL_hipabd_to_HL_thigh":  0.4,
        "FR_hipabd_to_FR_thigh": -0.4,
        "FL_hipabd_to_FL_thigh": -0.4,
    }
    base.update(j_overrides)
    return np.array([base.get(j, 0.0) for j in JOINTS], dtype=np.float32)


def main() -> None:
    fps = 25
    # Build a 5-second sweep that exercises the new DOFs:
    #   t=0.0-1.0 : default standing pose (verify rest)
    #   t=1.0-2.0 : HAA spread to ±0.3 rad on all 4 hips (lateral spread)
    #   t=2.0-3.0 : spine + neck yaw to +0.25 rad (turning right)
    #   t=3.0-4.0 : spine + neck yaw to -0.25 rad (turning left)
    #   t=4.0-5.0 : back to default
    duration = 5.0
    T = int(duration * fps)
    N_envs = 4    # show 4 copies side-by-side

    poses = []
    for i in range(T):
        t = i / fps
        if t < 1.0:
            j = make_pose({})
        elif t < 2.0:
            phase = math.sin(math.pi * (t - 1.0))   # 0 -> 1 -> 0 over 1s
            j = make_pose({
                "pelvis_to_HR_hipabd":  0.3 * phase,
                "pelvis_to_HL_hipabd": -0.3 * phase,
                "chest_to_FR_hipabd":   0.3 * phase,
                "chest_to_FL_hipabd":  -0.3 * phase,
            })
        elif t < 3.0:
            phase = math.sin(math.pi * (t - 2.0))
            j = make_pose({
                "pelvis_to_chest_yaw": 0.25 * phase,
                "chest_to_neck_yaw":   0.25 * phase,
            })
        elif t < 4.0:
            phase = math.sin(math.pi * (t - 3.0))
            j = make_pose({
                "pelvis_to_chest_yaw": -0.25 * phase,
                "chest_to_neck_yaw":   -0.25 * phase,
            })
        else:
            j = make_pose({})
        poses.append(j)

    joint_positions = np.zeros((T, N_envs, 21), dtype=np.float32)
    for n in range(N_envs):
        for t in range(T):
            joint_positions[t, n, :] = poses[t]

    # Root: pelvis at z=0.93, with the canonical Rx(-90°) spawn quaternion
    # (matches deer_usd_articulation._DEFAULT_ROT). Each env spawned 3 m apart
    # along world X so they don't overlap.
    root_pos = np.zeros((T, N_envs, 3), dtype=np.float32)
    for n in range(N_envs):
        root_pos[:, n, 0] = 3.0 * n
    root_pos[:, :, 2] = 0.93

    # quat_wxyz = (0.7071068, -0.7071068, 0, 0)
    root_quat = np.tile(
        np.array([0.7071068, -0.7071068, 0.0, 0.0], dtype=np.float32),
        (T, N_envs, 1),
    )

    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        NPZ_OUT,
        joint_positions=joint_positions,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        joints_list=np.array(JOINTS),
        fps=np.float32(fps),
    )
    print(f"[static] wrote {NPZ_OUT}  shape={joint_positions.shape}  T={T}  N={N_envs}")

    # Now render via the existing meshed renderer.
    cmd = [
        sys.executable,
        str(REPO / "tools" / "render_playback_meshed.py"),
        "--input", str(NPZ_OUT),
        "--output", str(MP4_OUT),
        "--fps", str(fps),
        "--max_envs", "4",
        "--stride", "1",
    ]
    print(f"[static] running renderer: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[static] renderer exit code {rc}")
        sys.exit(rc)
    print(f"[static] wrote {MP4_OUT}")


if __name__ == "__main__":
    main()
