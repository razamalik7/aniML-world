"""Validation metrics for reference motion clips.

These are the acceptance gates a video-derived clip must pass before it is
worth spending GPU time on. They are deliberately *physical* and *visual*
rather than statistical, because this project has repeatedly been misled by
healthy-looking aggregate metrics (see MEMORY: "posture metrics get gamed").
Every metric here therefore has a matching ``--render`` visual output.

Metrics
-------
bone_length_consistency
    Std/mean of every skeleton edge length over time. For a clip produced by
    FK on our own rig this is exactly 0 by construction, so it is really a
    check on *externally lifted 3D keypoints* (a 3D-lift stage that breathes
    bone lengths is a broken 3D-lift stage). Report: max CV over bones.

foot_slide
    Horizontal displacement of a hoof between consecutive frames while that
    hoof is in "stance" (below a height threshold). Real quadrupeds have
    ~0 slide; naive keypoint retargeting notoriously produces skating.
    Report: mean and 95th-percentile slide per stance frame, in metres, and
    total slid distance as a fraction of root travel.

ground_penetration
    How far the lowest hoof goes below z=0, and how far the lowest hoof sits
    ABOVE 0 (floating). Both are retargeting failures.

root_consistency
    Compares the root translation actually stored in the clip against the
    translation implied by integrating stance-foot motion ("foot odometry").
    A mismatch means the root trajectory and the leg motion disagree -- the
    single most common defect when the root is synthesised rather than
    measured. Report: implied vs stored speed.

joint_limits / velocity spikes
    Fraction of frames outside URDF limits; max |joint velocity|.

reprojection_error
    Only meaningful when 2D observations exist. Given (T,K,2) observed image
    keypoints plus a camera, reports the pixel distance between observed and
    FK-projected keypoints. This is the metric that grounds the whole
    video->motion fit; everything else can look fine while the pose is wrong.

Usage:
    python tools/cv/validate_motion.py --clip assets/motions/deer_walk.npy \\
        [--map-legacy] [--render out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deer_rig import DeerRig
from keypoint_schema import DeerKeypoints
from motion_npy import MotionClip, check_clip

# The 15 legacy dataset joint names (deer_walk/run/idle.npy) mapped onto the
# current deer_world.urdf names. Correspondence taken from the comments in
# training/envs/deer_usd_amp_env_cfg.py; it is POSITIONAL and semantic
# (same physical DOF, renamed when the v2 rig split yaw/abduction out).
LEGACY_JOINT_RENAME: dict[str, str] = {
    "pelvis_to_chest": "chest_yaw_to_chest",
    "pelvis_to_HR_thigh": "HR_hipabd_to_HR_thigh",
    "pelvis_to_HL_thigh": "HL_hipabd_to_HL_thigh",
    "chest_to_FR_thigh": "FR_hipabd_to_FR_thigh",
    "chest_to_FL_thigh": "FL_hipabd_to_FL_thigh",
    "chest_to_neck": "neck_yaw_to_neck",
}

HOOVES = ["L_f_hoof", "R_f_hoof", "L_h_hoof", "R_h_hoof"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def bone_length_consistency(kps: DeerKeypoints, world_kp: np.ndarray) -> dict:
    L = kps.bone_lengths(world_kp)  # (T, n_bones)
    mean = L.mean(axis=0)
    cv = L.std(axis=0) / np.maximum(mean, 1e-9)
    worst = int(np.argmax(cv))
    a, b = kps.skeleton_idx[worst]
    return {
        "max_cv": float(cv.max()),
        "mean_cv": float(cv.mean()),
        "worst_bone": f"{kps.names[a]}->{kps.names[b]}",
        "worst_bone_mean_len_m": float(mean[worst]),
    }


def foot_slide(
    world_kp: np.ndarray,
    kps: DeerKeypoints,
    fps: float,
    stance_height: float = 0.06,
    root_travel: float | None = None,
) -> dict:
    idx = [kps.index(h) for h in HOOVES]
    H = world_kp[:, idx, :]  # (T, 4, 3)
    z = H[..., 2]
    stance = z < (z.min(axis=0, keepdims=True) + stance_height)
    d = np.linalg.norm(H[1:, :, :2] - H[:-1, :, :2], axis=-1)  # (T-1, 4)
    both = stance[1:] & stance[:-1]
    if not both.any():
        return {"n_stance_transitions": 0, "mean_slide_m": float("nan")}
    s = d[both]
    total = float(s.sum())
    out = {
        "n_stance_transitions": int(both.sum()),
        "stance_fraction": float(stance.mean()),
        "mean_slide_m_per_frame": float(s.mean()),
        "p95_slide_m_per_frame": float(np.percentile(s, 95)),
        "mean_slide_speed_m_s": float(s.mean() * fps),
        "total_slid_m": total,
    }
    if root_travel:
        out["slid_over_travelled"] = float(total / max(root_travel, 1e-9))
    return out


def ground_contact(world_kp: np.ndarray, kps: DeerKeypoints) -> dict:
    idx = [kps.index(h) for h in HOOVES]
    z = world_kp[:, idx, 2]
    lowest = z.min(axis=1)
    return {
        "min_hoof_z_m": float(z.min()),
        "max_penetration_m": float(max(0.0, -z.min())),
        "mean_lowest_hoof_z_m": float(lowest.mean()),
        "max_float_m": float(max(0.0, lowest.max())),
        "frames_all_feet_airborne": int((z.min(axis=1) > 0.05).sum()),
    }


def root_consistency(
    world_kp: np.ndarray, kps: DeerKeypoints, clip: MotionClip, stance_height: float = 0.06
) -> dict:
    """Compare stored root motion against foot-odometry-implied motion.

    If the feet are planted and the body moves forward, the stance feet must
    move BACKWARD relative to the root by exactly the distance travelled.
    """
    idx = [kps.index(h) for h in HOOVES]
    H = world_kp[:, idx, :]
    root = clip.root_position
    stored = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    stored_speed = stored / max(clip.duration, 1e-9)

    # Foot-relative-to-root displacement during stance, summed = implied travel.
    rel = H[..., :2] - root[:, None, :2]
    z = H[..., 2]
    stance = z < (z.min(axis=0, keepdims=True) + stance_height)
    both = stance[1:] & stance[:-1]
    drel = rel[1:] - rel[:-1]
    if not both.any():
        implied_speed = float("nan")
    else:
        # Each stance foot should recede at the body's forward speed.
        implied_speed = float(np.linalg.norm(drel[both], axis=-1).mean() * clip.fps)
    return {
        "stored_travel_m": stored,
        "stored_speed_m_s": stored_speed,
        "foot_odometry_speed_m_s": implied_speed,
        "speed_ratio_odom_over_stored": float(implied_speed / max(stored_speed, 1e-9))
        if np.isfinite(implied_speed)
        else float("nan"),
    }


def joint_dynamics(clip: MotionClip, rig: DeerRig) -> dict:
    T, J = clip.joint_positions.shape
    dt = 1.0 / clip.fps
    vel = np.abs(np.diff(clip.joint_positions, axis=0)) / dt
    n_out = 0
    worst = None
    for i, name in enumerate(clip.joints_list):
        if name not in rig.joints:
            continue
        lo, hi = rig.joints[name].lower, rig.joints[name].upper
        col = clip.joint_positions[:, i]
        bad = int(((col < lo - 1e-4) | (col > hi + 1e-4)).sum())
        n_out += bad
        if bad and (worst is None or bad > worst[1]):
            worst = (name, bad)
    return {
        "frames_x_joints_out_of_limits": n_out,
        "out_of_limit_fraction": float(n_out / max(T * J, 1)),
        "worst_joint": worst[0] if worst else None,
        "max_joint_speed_rad_s": float(vel.max()) if vel.size else 0.0,
        "p99_joint_speed_rad_s": float(np.percentile(vel, 99)) if vel.size else 0.0,
    }


def reprojection_error(
    kps: DeerKeypoints,
    cam,
    world_kp: np.ndarray,
    observed_uv: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict:
    """Pixel error between observed 2D keypoints and FK-projected keypoints.

    ``observed_uv`` is (T,K,2) in the same keypoint order as ``kps.names``.
    """
    T, K, _ = observed_uv.shape
    errs = []
    for t in range(T):
        uv, z = cam.project(world_kp[t])
        ok = z > 1e-6
        if valid is not None:
            ok &= valid[t].astype(bool)
        if ok.any():
            errs.append(np.linalg.norm(uv[ok] - observed_uv[t][ok], axis=-1))
    if not errs:
        return {"mean_px": float("nan")}
    e = np.concatenate(errs)
    return {
        "mean_px": float(e.mean()),
        "median_px": float(np.median(e)),
        "p95_px": float(np.percentile(e, 95)),
        "max_px": float(e.max()),
        "n_points": int(e.size),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _remap_legacy(clip: MotionClip) -> MotionClip:
    clip.joints_list = [LEGACY_JOINT_RENAME.get(j, j) for j in clip.joints_list]
    return clip


def evaluate(clip: MotionClip, rig: DeerRig, kps: DeerKeypoints) -> dict:
    q = np.zeros((clip.n_frames, len(rig.dof_names)))
    col = {n: i for i, n in enumerate(rig.dof_names)}
    for i, name in enumerate(clip.joints_list):
        if name in col:
            q[:, col[name]] = clip.joint_positions[:, i]
    world_kp = kps.positions_seq(q, clip.root_position, clip.root_quaternion)
    travel = float(np.linalg.norm(clip.root_position[-1, :2] - clip.root_position[0, :2]))
    return {
        "world_kp": world_kp,
        "bone_length": bone_length_consistency(kps, world_kp),
        "foot_slide": foot_slide(world_kp, kps, clip.fps, root_travel=travel),
        "ground": ground_contact(world_kp, kps),
        "root": root_consistency(world_kp, kps, clip),
        "joints": joint_dynamics(clip, rig),
    }


def _render_strip(clip, rig, kps, res: dict, out_path: Path, n: int = 6) -> None:
    """Always produce a picture next to the numbers."""
    import imageio.v2 as imageio

    from render import Camera, render_rig

    q = np.zeros((clip.n_frames, len(rig.dof_names)))
    col = {n_: i for i, n_ in enumerate(rig.dof_names)}
    for i, name in enumerate(clip.joints_list):
        if name in col:
            q[:, col[name]] = clip.joint_positions[:, i]
    frames = np.linspace(0, clip.n_frames - 1, n).astype(int)
    tiles = []
    for f in frames:
        # LEVEL side camera: eye and target at the same height, so the picture
        # cannot fake a body pitch that isn't there. (An earlier version put the
        # eye 0.55 m above the target and every clip "looked" nose-down.)
        centre = clip.root_position[f].astype(float)
        centre[2] = 0.75
        eye = centre + np.array([3.2, 0.0, 0.0])
        cam = Camera.look_at(eye, centre, 380, 300, hfov_deg=45)
        img = render_rig(rig, cam, q[f], clip.root_position[f], clip.root_quaternion[f])["rgb"].copy()
        uv, z = cam.project(res["world_kp"][f])
        vis = cam.in_frame(uv, z)
        for k in np.flatnonzero(vis):
            u, v = int(uv[k, 0]), int(uv[k, 1])
            img[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3] = [255, 30, 30]
        tiles.append(img)
    imageio.imwrite(str(out_path), np.hstack(tiles))
    print(f"  wrote {out_path}")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--map-legacy", action="store_true",
                    help="rename the 6 pre-v2 joint names to their current URDF names")
    ap.add_argument("--render", default=None, help="write a filmstrip PNG here")
    args = ap.parse_args()

    clip = MotionClip.load(args.clip)
    if args.map_legacy:
        clip = _remap_legacy(clip)
    rig = DeerRig(args.urdf) if args.urdf else DeerRig()
    kps = DeerKeypoints(rig)

    print(f"{args.clip}  ({clip.n_frames} frames @ {clip.fps} fps, {clip.duration:.2f}s)")
    for p in check_clip(clip, rig):
        print(f"  [structure] {p}")

    res = evaluate(clip, rig, kps)
    for section in ("bone_length", "foot_slide", "ground", "root", "joints"):
        print(f"  [{section}]")
        for k, v in res[section].items():
            print(f"      {k:34s} {v:.4f}" if isinstance(v, float) else f"      {k:34s} {v}")

    if args.render:
        _render_strip(clip, rig, kps, res, Path(args.render))


if __name__ == "__main__":
    _main()
