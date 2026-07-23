"""Generate a synthetic 2D-keypoint training set by rendering OUR deer rig.

Recipe follows Mu et al., "Learning from Synthetic Animals" (CVPR 2020), the
only quadruped synthetic-pose recipe with published ablations:
  * ~5k random-texture + ~5k original-texture images per animal
  * 640x480, backgrounds randomised from a photo pool (they used COCO val2017)
  * randomised lighting, viewpoint, pose
  * 4:1 train/val split
  * auto-labelled keypoints + PART SEGMENTATION (their Table 3: part-seg as an
    auxiliary task is worth +1.4 PCK on horses, +1.4 on tigers)

CRITICAL CONTEXT before you spend time here -- see docs/CV_PIPELINE_PROGRESS.md:
Mu et al. measured synthetic-only transfer at **25.3 PCK@0.05 on real horses**,
rising to **70.8 with CC-SSL self-training on unlabelled real video**. Rendering
alone is NOT sufficient; the domain-adaptation stage is where the accuracy comes
from. Do not judge this generator by how pretty the renders are.

Also: our rig currently renders with a 10.5 cm hole between pelvis and chest and
no hooves (see ``DeerRig.missing_visuals`` and the progress doc). Fix the mesh
gaps BEFORE generating a real training set, or the net learns a silhouette that
does not exist in nature.

Poses are sampled from the existing reference clips plus bounded random
perturbation, so the pose distribution stays biomechanically plausible instead
of uniform-over-joint-limits (which produces impossible contortions).

Usage:
    python tools/cv/synth_dataset.py --out data/synth_deer --n 200
    python tools/cv/synth_dataset.py --out data/synth_deer --n 10000 \\
        --backgrounds path/to/coco/val2017
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deer_rig import DeerRig
from keypoint_schema import AP10K_ORDER, DeerKeypoints
from motion_npy import SPAWN_QUAT_XYZW, MotionClip
from render import Camera, render_rig
from validate_motion import LEGACY_JOINT_RENAME

REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_DIR = REPO_ROOT / "assets" / "motions"


# ---------------------------------------------------------------------------
# Pose sampling
# ---------------------------------------------------------------------------


def load_pose_bank(rig: DeerRig, clips=("deer_walk", "deer_run", "deer_idle")) -> np.ndarray:
    """(N, DOF) bank of real reference poses, in rig DOF order."""
    col = {n: i for i, n in enumerate(rig.dof_names)}
    out = []
    for name in clips:
        p = MOTION_DIR / f"{name}.npy"
        if not p.is_file():
            continue
        clip = MotionClip.load(p)
        names = [LEGACY_JOINT_RENAME.get(j, j) for j in clip.joints_list]
        q = np.zeros((clip.n_frames, len(rig.dof_names)))
        for i, n in enumerate(names):
            if n in col:
                q[:, col[n]] = clip.joint_positions[:, i]
        out.append(q)
    if not out:
        raise FileNotFoundError(f"no reference clips found in {MOTION_DIR}")
    return np.concatenate(out, axis=0)


def sample_pose(bank: np.ndarray, rig: DeerRig, rng: np.random.Generator,
                jitter: float = 0.12) -> np.ndarray:
    """Interpolate two bank poses and add bounded noise, then clamp to limits."""
    lo, hi = rig.joint_limits()
    i, j = rng.integers(0, len(bank), 2)
    a = rng.uniform()
    q = (1 - a) * bank[i] + a * bank[j]
    q = q + rng.normal(0.0, jitter, size=q.shape) * (hi - lo) / 2.0
    return np.clip(q, lo, hi)


def sample_camera(rng: np.random.Generator, target: np.ndarray, width: int, height: int) -> Camera:
    """Random viewpoint on a hemisphere around the deer.

    ZebraPose's main empirical lesson was that VIEWPOINT/SCALE COVERAGE matters
    more than photorealism: adding cropped+scaled variants of the same renders
    moved detection mAP50 from 0.150 to 0.512. So sample distance and elevation
    widely, and bias elevation toward the near-ground angles a real handheld or
    trail-camera clip would have.
    """
    azim = rng.uniform(0, 2 * np.pi)
    elev = np.deg2rad(rng.uniform(-8, 45) if rng.uniform() < 0.75 else rng.uniform(45, 80))
    dist = rng.uniform(2.2, 9.0)
    eye = target + dist * np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )
    look = target + rng.normal(0, 0.12, 3)
    return Camera.look_at(
        eye, look, width, height,
        hfov_deg=rng.uniform(35, 65),
        roll_deg=rng.normal(0, 4),
    )


def random_palette(rng: np.random.Generator, links, mode: str) -> dict:
    """``mode='original'`` = deer-ish browns; ``'random'`` = per-part random."""
    if mode == "random":
        return {lk: rng.uniform(0.05, 0.95, 3) for lk in links}
    base = np.array([0.42, 0.29, 0.18]) + rng.normal(0, 0.05, 3)
    return {lk: np.clip(base + rng.normal(0, 0.035, 3), 0.03, 0.95) for lk in links}


def load_backgrounds(path: str | None, limit: int = 4000) -> list[Path]:
    if not path:
        return []
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(p)
    files = sorted(
        f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return files[:limit]


def make_background(rng, bgs, width, height):
    if not bgs:
        c = rng.uniform(0.25, 0.85, 3)
        grad = np.linspace(0.85, 1.15, height)[:, None, None]
        img = np.broadcast_to(c[None, None, :] * grad, (height, width, 3))
        noise = rng.normal(0, 0.03, (height, width, 3))
        return np.clip((img + noise) * 255, 0, 255).astype(np.uint8)
    import imageio.v2 as imageio

    img = np.asarray(imageio.imread(str(bgs[rng.integers(len(bgs))])))
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3]
    # random crop then nearest-resize to target (no cv2 available)
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = rng.integers(0, h - s + 1)
    x0 = rng.integers(0, w - s + 1)
    img = img[y0 : y0 + s, x0 : x0 + s]
    yi = (np.linspace(0, s - 1, height)).astype(int)
    xi = (np.linspace(0, s - 1, width)).astype(int)
    return img[yi][:, xi]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate(out_dir: Path, n: int, width: int, height: int, seed: int,
             backgrounds: str | None, random_texture_frac: float = 0.5) -> None:
    import imageio.v2 as imageio

    rig = DeerRig()
    kps = DeerKeypoints(rig)
    bank = load_pose_bank(rig)
    bgs = load_backgrounds(backgrounds)
    rng = np.random.default_rng(seed)

    img_dir = out_dir / "images"
    seg_dir = out_dir / "part_seg"
    img_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    missing = rig.missing_visuals()
    print(f"[synth] rig has {len(rig.dof_names)} DOF, {len(kps)} keypoints")
    print(f"[synth] pose bank: {len(bank)} reference poses")
    print(f"[synth] backgrounds: {len(bgs) or 'procedural'}")
    if missing:
        print(f"[synth] WARNING links without visual mesh (invisible in renders): {missing}")

    records = []
    for i in range(n):
        q = sample_pose(bank, rig, rng)
        root = np.array([0.0, 0.0, rng.uniform(0.95, 1.10)])
        cam = sample_camera(rng, root + np.array([0, 0, -0.15]), width, height)
        mode = "random" if rng.uniform() < random_texture_frac else "original"
        palette = random_palette(rng, list(rig.visuals), mode)
        bg = make_background(rng, bgs, width, height)

        out = render_rig(rig, cam, q, root, SPAWN_QUAT_XYZW,
                         colors=palette, background=bg)
        if out["mask"].sum() < 0.01 * width * height:
            continue  # deer too small / off-frame

        P = kps.positions(q, root, SPAWN_QUAT_XYZW)
        uv, z = cam.project(P)
        in_frame = cam.in_frame(uv, z)

        # Occlusion-aware visibility from the depth buffer: a keypoint is
        # visible if the rendered depth at its pixel is not much closer than
        # the keypoint itself. Without this, labels claim self-occluded far-side
        # legs are visible and the net learns to hallucinate.
        vis = np.zeros(len(P), dtype=np.int8)
        for k in np.flatnonzero(in_frame):
            u, v = int(uv[k, 0]), int(uv[k, 1])
            if not (0 <= u < width and 0 <= v < height):
                vis[k] = 1  # in frame by margin but off-buffer -> labelled, not visible
                continue
            d = out["depth"][v, u]
            vis[k] = 2 if (not np.isfinite(d) or z[k] <= d + 0.05) else 1

        stem = f"{i:06d}"
        imageio.imwrite(str(img_dir / f"{stem}.png"), out["rgb"])
        np.save(str(seg_dir / f"{stem}.npy"), out["seg"].astype(np.int8))
        ys, xs = np.nonzero(out["mask"])
        records.append(
            {
                "id": stem,
                "file": f"images/{stem}.png",
                "part_seg": f"part_seg/{stem}.npy",
                "texture_mode": mode,
                "bbox_xywh": [int(xs.min()), int(ys.min()),
                              int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                # keypoints: [u, v, visibility] with 0=absent 1=labelled-occluded 2=visible
                "keypoints": [[float(uv[k, 0]), float(uv[k, 1]), int(vis[k])]
                              for k in range(len(P))],
                "joint_angles": [float(v) for v in q],
                "root_position": [float(v) for v in root],
                "camera": {
                    "K": cam.intrinsics.tolist(),
                    "R_cw": cam.R_cw.tolist(),
                    "C": cam.C.tolist(),
                },
            }
        )
        if (i + 1) % 50 == 0:
            print(f"[synth]   {i + 1}/{n}", flush=True)

    meta = {
        "n_images": len(records),
        "image_size": [width, height],
        "keypoint_names": kps.names,
        "ap10k_order": AP10K_ORDER,
        "skeleton": [[int(a), int(b)] for a, b in kps.skeleton_idx],
        "flip_pairs": [[int(a), int(b)] for a, b in kps.flip_idx],
        "part_names": sorted(rig.visuals),
        "dof_names": rig.dof_names,
        "urdf": str(rig.urdf_path),
        "links_without_visual_mesh": missing,
        "records": records,
    }
    (out_dir / "annotations.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[synth] wrote {len(records)} samples -> {out_dir}")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backgrounds", default=None,
                    help="directory of background photos (e.g. COCO val2017)")
    ap.add_argument("--random-texture-frac", type=float, default=0.5)
    args = ap.parse_args()
    generate(Path(args.out), args.n, args.width, args.height, args.seed,
             args.backgrounds, args.random_texture_frac)


if __name__ == "__main__":
    _main()
