"""Give AMPLoader's MotionData a hoof-height channel, computed by FK on the reference.

Pairs with `training/polyart/observations.hoof_heights` on the env side. Both must
produce the SAME quantity in the SAME column order or the discriminator compares
mismatched vectors and the whole run is quietly meaningless -- the failure mode that
zero-filled joint columns once produced (preflight check 8).

The reference clips carry joint angles and a root pose, not link positions, so the
hoof heights are reconstructed with the same FK the mesh renderer uses:
    world = root_rot . link_transform ; height = world.z
That is the LINK ORIGIN (the joint), not the sole of the hoof -- it bottoms out around
0.03-0.05 m rather than 0. The offset is constant per hoof and identical on the sim
side, so it cancels; the 13-18 cm swing is what carries the signal.

MotionData is resampled to the simulator dt at load, so heights are computed on the
ORIGINAL frames and resampled the same way, rather than FK'd after resampling.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

FEET = ["L_Toe0", "R_Toe0", "L_Finger0", "R_Finger0"]


def resolve_loader(runner):
    """Find the AMPLoader. It is NOT on the runner.

    `amp_on_policy_runner.py` builds it as a LOCAL (~line 302) and hands it to the
    algorithm, which keeps it as `self.amp_data` (`amp_ppo.py:140`). `runner.amp_data`
    was an assumption, and it crashed B twice: once here, and once again in
    train_amp.py's own width guard, which had its own copy of the same wrong expression.
    One helper, used by both, so a third copy cannot drift.
    """
    loader = getattr(runner, "amp_data", None)
    if loader is None:
        loader = getattr(getattr(runner, "alg", None), "amp_data", None)
    if loader is None or not hasattr(loader, "motion_data"):
        raise SystemExit(
            "[hoof] cannot find the AMPLoader. Looked at runner.amp_data and "
            "runner.alg.amp_data; neither has .motion_data. amp-rsl-rl moved it -- "
            "grep 'amp_data' in amp_rsl_rl/runners/amp_on_policy_runner.py.")
    return loader


def reference_hoof_world(origin_map, joints_list, jp, rp, rq) -> np.ndarray:
    """(T, 4, 3) hoof link-origin positions in world space for one clip."""
    from tools.render_playback_meshed import compute_link_transforms
    out = np.zeros((len(jp), len(FEET), 3), dtype=np.float64)
    for t in range(len(jp)):
        ja = {n: float(jp[t, i]) for i, n in enumerate(joints_list)}
        link_T = compute_link_transforms(ja, origin_map)
        Rr = Rotation.from_quat(rq[t]); pr = rp[t]
        for k, h in enumerate(FEET):
            if h in link_T:
                Rl, tl = link_T[h]
                out[t, k] = Rr.apply(tl) + pr
    return out


def reference_hoof_heights(npy_path, origin_map, joints_list, jp, rp, rq) -> np.ndarray:
    """(T, 4) hoof link-origin heights for one clip."""
    return reference_hoof_world(origin_map, joints_list, jp, rp, rq)[:, :, 2].astype(np.float32)


def reference_hoof_speeds(origin_map, joints_list, jp, rp, rq, fps) -> np.ndarray:
    """(T, 4) hoof horizontal speed, m/s, matching ``observations.hoof_planar_speed``.

    Differenced at the clip's OWN fps and only then resampled, for the same reason the
    heights are: differencing after resampling would rescale every velocity by the
    resample ratio, and the error would be a smooth plausible-looking factor rather than
    anything that announces itself.

    The final frame is held rather than wrapped. These clips are not perfect loops -- a
    wrap would manufacture one enormous fake velocity spike per clip, and AMP samples
    transitions uniformly, so it would be drawn as often as any real frame.
    """
    P = reference_hoof_world(origin_map, joints_list, jp, rp, rq)
    v = np.linalg.norm(np.diff(P[:, :, :2], axis=0), axis=2) * float(fps)
    return np.concatenate([v, v[-1:]], axis=0).astype(np.float32)


def attach(runner, urdf_path, verbose=True, height=True, speed=False) -> int:
    """Attach hoof channels to every MotionData and extend get_amp_dataset_obs.

    Returns the number of columns added. COLUMN ORDER IS HEIGHTS THEN SPEEDS, and the
    env side must build its terms in that same order -- a silent transposition here
    would have the discriminator comparing a height against a speed, which is
    dimensionally meaningless but numerically perfectly happy.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(urdf_path).resolve().parents[2]))
    from tools.render_playback_meshed import parse_urdf_full
    from tools.rig_pipeline.retarget_ik import load_clip

    if not (height or speed):
        return 0
    origin_map, _ = parse_urdf_full(Path(urdf_path))

    loader = resolve_loader(runner)
    ds_root = Path(runner.dataset_cfg["amp_data_path"])
    names = list(runner.dataset_cfg["datasets"].keys())
    if len(names) != len(loader.motion_data):
        raise RuntimeError(f"dataset name/MotionData mismatch: {len(names)} vs {len(loader.motion_data)}")

    def resample(A: np.ndarray, T: int) -> np.ndarray:
        src = np.linspace(0, len(A) - 1, T)
        lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, len(A) - 1)
        w = (src - lo)[:, None]
        return A[lo] * (1 - w) + A[hi] * w

    for name, md in zip(names, loader.motion_data):
        d = load_clip(str(ds_root / f"{name}.npy"))
        jl = [str(x) for x in d["joints_list"]]
        jp = np.asarray(d["joint_positions"], float)
        rp = np.asarray(d["root_position"], float)
        rq = np.asarray(d["root_quaternion"], float)
        T = len(md.joint_positions)
        cols, msg = [], []
        if height:
            H = reference_hoof_heights(None, origin_map, jl, jp, rp, rq)
            cols.append(resample(H, T)); msg.append(f"height {H.min():.3f}-{H.max():.3f} m")
        if speed:
            V = reference_hoof_speeds(origin_map, jl, jp, rp, rq, float(d["fps"]))
            cols.append(resample(V, T)); msg.append(f"speed {V.min():.2f}-{V.max():.2f} m/s")
        md.hoof_extra = torch.as_tensor(np.concatenate(cols, axis=1),
                                        dtype=torch.float32, device=md.device)
        if verbose:
            print(f"[hoof] {name}: {T} frames, " + ", ".join(msg), flush=True)

    # extend the observation
    from amp_rsl_rl.utils.motion_loader import MotionData
    if not getattr(MotionData, "_hoof_patched", False):
        _orig = MotionData.get_amp_dataset_obs

        def _with_hoof(self, indices, *a, _f=_orig, **kw):
            base = _f(self, indices, *a, **kw)
            hh = getattr(self, "hoof_extra", None)
            if hh is None:
                return base
            return torch.cat((base, hh[indices]), dim=1)

        MotionData.get_amp_dataset_obs = _with_hoof
        MotionData._hoof_patched = True

    # REBUILD the loader's flat buffers -- without this the patch above is decorative.
    #
    # AMPLoader precomputes all_obs / all_next_obs ONCE at construction
    # (motion_loader.py:342-370) by calling get_amp_dataset_obs, and
    # `feed_forward_generator` -- the only path `amp_ppo.update()` ever samples expert
    # transitions from (amp_ppo.py:522 -> :781) -- reads those buffers, never the
    # method. Patching the method after the loader is built therefore changed nothing
    # the discriminator sees, and B died with policy width 172 against expert 164.
    # This was not a wiring slip; the original design hooked a function that is called
    # exactly once, before the hook exists.
    dev = loader.all_obs.device
    n_rows, w_before = int(loader.all_obs.shape[0]), int(loader.all_obs.shape[1])
    obs_l, nxt_l = [], []
    for md in loader.motion_data:
        T = len(md)
        idx = torch.arange(T, device=dev)
        nidx = torch.clamp(idx + 1, max=T - 1)
        obs_l.append(md.get_amp_dataset_obs(idx, loader.velocity_representation))
        nxt_l.append(md.get_amp_dataset_obs(nidx, loader.velocity_representation))
    new_obs, new_nxt = torch.cat(obs_l, 0), torch.cat(nxt_l, 0)
    # Row count must be untouched: per_frame_weights was built from the ORIGINAL
    # augmented lengths, and torch.multinomial would index off the end if we changed it.
    # A mismatch means symmetry augmentation is on and this rebuild has to replicate it.
    if new_obs.shape[0] != n_rows:
        raise SystemExit(
            f"[hoof] rebuild changed the expert row count ({n_rows} -> "
            f"{new_obs.shape[0]}). AMP dataset symmetry augmentation is probably "
            f"enabled; the rebuild must apply _apply_symmetry the same way.")
    loader.all_obs, loader.all_next_obs = new_obs, new_nxt
    if verbose:
        print(f"[hoof] expert buffers rebuilt: {n_rows} rows, width "
              f"{w_before} -> {int(new_obs.shape[1])}", flush=True)
    return len(FEET) * (int(height) + int(speed))
