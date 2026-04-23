"""Pre-bake reset states from AMP motion .npy files into a .pt that the
env's reset event can load directly (no AMPLoader dependency in the env).

For each frame of the walk clip, we want:
  - joint_positions (15,)
  - joint_velocities (15,)
  - root_lin_vel_b (3,)   -- body-frame linear velocity
  - root_lin_vel_w (3,)   -- world-frame linear velocity (post spawn rotation)
  - root_ang_vel_w (3,)   -- world-frame angular velocity
  - root_quat_wxyz (4,)   -- Isaac-style quaternion

Axis note (v2 of this script):
  The deer URDF was authored with body -Z = anatomical forward (head at -Z,
  feet cycling fore/aft around the Z axis). With the Rx(-90°) spawn
  rotation, body -Z maps to world -Y. So a deer walking forward in Isaac
  moves toward world -Y, NOT world +X. The previous version of this bake
  placed the expert's forward velocity along world +X, which meant RSI
  spawned the deer with a lateral shove and every env immediately tipped.

Now we synthesize root_position along world -Y and root_lin_vel_b along
body -Z, which matches the actual anatomy-forward direction that the
leg-cycle joint positions produce.

These are what `reset_from_amp_motion` in deer_rewards.py consumes.
Output: assets/motions/deer_walk_reset_states.pt (torch dict of tensors).

Run:
  python tools/bake_reset_states.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
NPY_PATH = REPO_ROOT / "assets" / "motions" / "deer_walk.npy"
OUT_PATH = REPO_ROOT / "assets" / "motions" / "deer_walk_reset_states.pt"

# Sample at simulation rate so reset states match sim timestep; AMPLoader
# uses 60 Hz for the default Isaac Lab sim.
SIM_DT = 1.0 / 60.0


def resample(arr: np.ndarray, t_orig: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_new), arr.shape[1]), dtype=arr.dtype)
    for i in range(arr.shape[1]):
        out[:, i] = np.interp(t_new, t_orig, arr[:, i])
    return out


def main() -> None:
    payload = np.load(NPY_PATH, allow_pickle=True).item()
    joints_list = payload["joints_list"]
    jp_raw = np.asarray(payload["joint_positions"], dtype=np.float32)  # (T, 15)
    fps = float(payload["fps"])

    T_orig = jp_raw.shape[0]
    dt_orig = 1.0 / fps
    t_orig = np.linspace(0, T_orig * dt_orig, T_orig)
    # Resample to simulation dt
    T_new = max(2, int(T_orig * dt_orig / SIM_DT))
    t_new = np.linspace(0, T_orig * dt_orig, T_new)

    jp = resample(jp_raw, t_orig, t_new)                # (T_new, 15)

    # Root pose is synthesized to match the anatomy-forward direction of
    # the deer URDF. Post-spawn (Rx(-90°)), body -Z -> world -Y. So the deer
    # walks in the world -Y direction at FORWARD_SPEED.
    FORWARD_SPEED = 0.8                                 # matches the .npy synthesis
    SPAWN_HEIGHT = 0.93
    # World-frame trajectory
    rp = np.zeros((T_new, 3), dtype=np.float32)
    rp[:, 1] = -FORWARD_SPEED * t_new                    # world -Y
    rp[:, 2] = SPAWN_HEIGHT

    # Quaternion constant: Rx(-90°), xyzw = (-sin45, 0, 0, cos45)
    rq = np.tile(
        np.array([-0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32),
        (T_new, 1),
    )

    # Joint velocities -- finite diff (wrap-around assumed; walk is cyclic)
    jv = np.zeros_like(jp, dtype=np.float32)
    jv[:-1] = (jp[1:] - jp[:-1]) / SIM_DT
    jv[-1]  = (jp[0] - jp[-1]) / SIM_DT

    # Root linear velocity (world frame) -- finite diff of root_position
    rlv_w = np.zeros_like(rp, dtype=np.float32)
    rlv_w[:-1] = (rp[1:] - rp[:-1]) / SIM_DT
    rlv_w[-1]  = rlv_w[-2]

    # Root linear velocity in body frame (body -Z = anatomical forward)
    rlv_b = np.zeros_like(rlv_w, dtype=np.float32)
    for i in range(T_new):
        R = Rotation.from_quat(rq[i]).as_matrix()  # world_from_body, = Rx(-90°)
        rlv_b[i] = R.T @ rlv_w[i]

    # Root angular velocity -- zero (in-place clip, constant root quat)
    rav_w = np.zeros((T_new, 3), dtype=np.float32)

    # Isaac quaternion convention: wxyz
    rq_wxyz = np.stack([rq[:, 3], rq[:, 0], rq[:, 1], rq[:, 2]], axis=1)

    out = {
        "joints_list":      joints_list,
        "joint_positions":  torch.from_numpy(jp).float(),
        "joint_velocities": torch.from_numpy(jv).float(),
        "root_position":    torch.from_numpy(rp).float(),
        "root_quat_wxyz":   torch.from_numpy(rq_wxyz).float(),
        "root_lin_vel_w":   torch.from_numpy(rlv_w).float(),
        "root_lin_vel_b":   torch.from_numpy(rlv_b).float(),
        "root_ang_vel_w":   torch.from_numpy(rav_w).float(),
        "fps":              float(fps),
        "sim_dt":           float(SIM_DT),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print(f"  T_new = {T_new} frames at {1/SIM_DT:.0f} Hz")
    print(f"  joint_positions shape = {out['joint_positions'].shape}")
    print(f"  root_lin_vel_w mean    = {out['root_lin_vel_w'].mean(0).tolist()}")
    print(f"  root_lin_vel_b mean    = {out['root_lin_vel_b'].mean(0).tolist()}")


if __name__ == "__main__":
    main()
