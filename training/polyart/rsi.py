"""Reference-state initialization (RSI) for the Tier-3 deer AMP env.

At each reset, spawn the deer mid-stride from a random frame of the reference
walk clip — natural joint pose + matching momentum — so training starts near the
reference manifold instead of from a cold stand. This is the single biggest
lever for AMP convergence.

Unlike the old rig (``reset_from_amp_motion`` + a pre-baked ``.pt``), this loads
and caches the reference ``.npy`` **directly** in the reset event. No separate
bake step, so it can never go stale relative to the motion data — the footgun
CLAUDE.md warns about ("rebuild the .pt after motion/axis changes").

Why the old rig's baggage is gone:
  * The Tier-3 USD is +X-forward / upright, so there's no ``Rx(-90)`` spawn quat
    and no "synthesize forward velocity along world −Y" hack — the reference's
    own world-frame root velocity is used as-is.
  * The reference is in-scale (pelvis z 0.89–0.96 ≈ the sim's 0.95 stand height).

Robustness: one glitch frame in the clip has an ~18 m/s root spike; frames whose
raw root speed or joint speed exceed sane bounds are excluded from the sample
pool (not just clamped), so RSI never spawns a pathological state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

_POOL_CACHE: dict = {}
_CLAMP_REPORTED = False   # one-shot guard flag, see reset_from_reference()


def _default_motion() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "motions" / "deer_polyart_walk.npy"


def _build_pool(
    path: Path,
    joint_names: tuple[str, ...],
    device: torch.device,
    max_lin_speed: float,
    max_ang_speed: float,
    max_joint_speed: float,
) -> dict:
    """Load the reference clip, derive per-frame reset states in SIM joint order,
    and mark the frames that are safe to sample from. Runs once per (path, rig)."""
    d = np.load(str(path), allow_pickle=True).item()
    ref_names = list(d["joints_list"])
    jp = np.asarray(d["joint_positions"], dtype=np.float32)     # (T, 38) ref order
    rp = np.asarray(d["root_position"], dtype=np.float32)       # (T, 3)
    rq = np.asarray(d["root_quaternion"], dtype=np.float32)     # (T, 4) xyzw
    fps = float(d["fps"])
    T = len(jp)

    # finite-difference velocities (defined on frames 0..T-2)
    jv = (jp[1:] - jp[:-1]) * fps                               # (T-1, 38)
    lin_w = (rp[1:] - rp[:-1]) * fps                            # (T-1, 3) world
    rot = Rotation.from_quat(rq)                                # xyzw
    ang_w = np.zeros((T - 1, 3), dtype=np.float32)
    for i in range(T - 1):
        ang_w[i] = (rot[i + 1] * rot[i].inv()).as_rotvec() * fps
    # align states to the velocity length
    jp, rp, rq = jp[:-1], rp[:-1], rq[:-1]

    # map reference joint columns -> sim joint order
    ref_idx = [ref_names.index(n) for n in joint_names]
    jp = jp[:, ref_idx]
    jv = jv[:, ref_idx]

    # VALID frames: exclude glitch frames (raw speed spikes) from the pool
    raw_lin_speed = np.linalg.norm(lin_w, axis=1)
    raw_jnt_speed = np.abs(jv).max(axis=1)
    valid = np.where((raw_lin_speed <= max_lin_speed) & (raw_jnt_speed <= max_joint_speed))[0]

    # gentle clamp on what's kept (belt-and-suspenders)
    jv = np.clip(jv, -max_joint_speed, max_joint_speed)
    lin_w = np.clip(lin_w, -max_lin_speed, max_lin_speed)
    ang_w = np.clip(ang_w, -max_ang_speed, max_ang_speed)

    quat_wxyz = rq[:, [3, 0, 1, 2]]  # xyzw -> wxyz for Isaac

    def t(x):
        return torch.tensor(x, device=device, dtype=torch.float32)

    return {
        "jp": t(jp), "jv": t(jv),
        "root_z": t(rp[:, 2]),
        "root_quat_wxyz": t(quat_wxyz),
        "lin_w": t(lin_w), "ang_w": t(ang_w),
        "valid": torch.tensor(valid, device=device, dtype=torch.long),
        "n_total": T - 1, "n_valid": int(len(valid)),
    }


def _get_pool(path, joint_names, device, mls, mas, mjs) -> dict:
    key = (str(path), joint_names, str(device), mls, mas, mjs)
    pool = _POOL_CACHE.get(key)
    if pool is None:
        pool = _build_pool(Path(path), joint_names, device, mls, mas, mjs)
        _POOL_CACHE[key] = pool
    return pool


def reset_from_reference(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_path: str | None = None,
    use_reference_height: bool = True,
    base_height: float = 0.95,
    ground_clearance: float = 0.065,
    pose_jitter: float = 0.0,
    vel_jitter: float = 0.0,
    root_vel_scale: float = 1.0,
    max_lin_speed: float = 3.0,
    max_ang_speed: float = 3.0,
    max_joint_speed: float = 20.0,
) -> None:
    """Reset event: spawn each env from a random valid reference frame.

    Places the root at the env origin (x,y) so it stays in its tile, at the
    reference height (or a constant), with the reference orientation and
    reference world-frame velocities. Joint pos/vel come from the same frame,
    clamped to joint limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    pool = _get_pool(
        motion_path or _default_motion(), tuple(asset.data.joint_names), device,
        max_lin_speed, max_ang_speed, max_joint_speed,
    )

    n = env_ids.shape[0]
    valid = pool["valid"]
    sel = valid[torch.randint(0, valid.shape[0], (n,), device=device)]

    # --- joint state ---
    jp = pool["jp"][sel]
    jv = pool["jv"][sel]
    if pose_jitter > 0:
        jp = jp + (torch.rand_like(jp) * 2 - 1) * pose_jitter
    if vel_jitter > 0:
        jv = jv + (torch.rand_like(jv) * 2 - 1) * vel_jitter
    lo = asset.data.soft_joint_pos_limits[env_ids, :, 0]
    hi = asset.data.soft_joint_pos_limits[env_ids, :, 1]
    jp_clamped = torch.clamp(jp, lo, hi)
    # GUARD (added 2026-07-25). This clamp is SILENT, and for every run up to r9 it
    # was mangling 100% of spawns: the URDF's joint limits were derived from a
    # different angle definition than the reference, so e.g. the left hind ankle was
    # forced 56.7 deg away from the reference pose at every reset -- and the policy
    # could never match the reference there, taking a permanent AMP style penalty it
    # was powerless to fix. Never let that hide again: report it loudly, once.
    global _CLAMP_REPORTED
    if not _CLAMP_REPORTED:
        _CLAMP_REPORTED = True
        err = (jp_clamped - jp).abs()
        worst = float(err.max())
        if worst > 0.017:  # ~1 deg
            j = int(err.max(dim=0).values.argmax())
            names = asset.data.joint_names
            n_bad = int((err.max(dim=1).values > 0.017).sum())
            print(f"[rsi] *** WARNING: reference poses are being CLAMPED to joint limits. "
                  f"worst={torch.rad2deg(err.max()):.1f} deg on '{names[j]}'; "
                  f"{n_bad}/{err.shape[0]} spawned envs affected. The rig cannot "
                  f"reproduce the reference -- rerun tools/rig_pipeline/fix_joint_limits.py. ***",
                  flush=True)
        else:
            print(f"[rsi] joint-limit clamp check OK (worst deviation "
                  f"{torch.rad2deg(err.max()):.2f} deg)", flush=True)
    jp = jp_clamped
    asset.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

    # --- root pose: env origin (xy) + reference height, reference orientation ---
    origins = env.scene.env_origins[env_ids]
    z = pool["root_z"][sel] if use_reference_height else torch.full((n,), base_height, device=device)
    # ground_clearance (added 2026-07-25): at the raw reference height the deer's
    # collision hulls are BELOW the floor on 67% of frames, deepest -6.3 cm
    # (measured by FK-ing the collision meshes, not the visuals). PhysX then
    # depenetrates at up to max_depenetration_velocity=1.0 m/s, so every single
    # episode began with the deer being shoved out of the floor for ~60 ms.
    # 0.065 m clears the worst frame by ~2 mm; the deer then settles a few cm,
    # which is gentle and physical, unlike an explosive depenetration.
    # Root cause is probably the rig's 82 mm FK reconstruction error (the four
    # hooves disagree about floor level by up to 5.8 cm, the same order), so this
    # is a spawn-side compensation, not a claim about the animation.
    pos = origins.clone()
    pos[:, 2] = origins[:, 2] + z + ground_clearance
    quat = pool["root_quat_wxyz"][sel]
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)

    # --- root velocity: reference world-frame lin + ang, SCALED ---
    # root_vel_scale=1.0 (r8): spawn the full reference state — pose AND its
    # matching root velocity. This is the standard DeepMimic/AMP RSI and it is
    # correct again now that the clip is repaired.
    #
    # History: r7 set this to 0.0 because the policy appeared to "coast" on
    # injected momentum to game the forward reward. The 2026-07-24 audit showed
    # that was largely an artifact of the CORRUPT clip — 58% of its frames were
    # one frozen dead-air pose, so the velocity pool was bimodal garbage (60% at
    # 0 m/s, the rest ~1.1 m/s) and most spawns paired a mid-stride pose with a
    # nonsense velocity. Worse, zeroing it made every spawn dynamically
    # INCONSISTENT: a mid-swing pose with no body momentum is a state a real deer
    # is never in, and many such states are simply unrecoverable -> instant sink.
    # The repaired clip is uniformly ~1.03-1.26 m/s, so pose and velocity now
    # agree on every frame. Set to 0.0 only to deliberately train standing starts.
    lin = pool["lin_w"][sel] * root_vel_scale
    ang = pool["ang_w"][sel] * root_vel_scale
    if vel_jitter > 0:
        lin = lin + (torch.rand_like(lin) * 2 - 1) * vel_jitter
        ang = ang + (torch.rand_like(ang) * 2 - 1) * vel_jitter
    asset.write_root_velocity_to_sim(torch.cat([lin, ang], dim=-1), env_ids=env_ids)
