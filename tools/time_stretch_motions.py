"""Time-stretch existing motion .npys to bake speed-varied reference clips.

Plan-B for low AMP style_r: the discriminator sees ``base_lin_vel`` in the
observation, but our reference dataset has only ONE walk speed (0.8 m/s) and
ONE run speed (2.4 m/s). When the policy is commanded at, say, 0.5 m/s, its
``base_lin_vel`` doesn't match any reference frame -> discriminator easily
distinguishes policy from expert -> style_r stays low.

This tool reads each existing motion .npy and produces multiple speed-stretched
copies. Time-stretching by factor ``k`` means:
  - frame interval ``dt`` shrinks by ``1/k``  (faster playback at same fps)
  - ``joint_positions`` keep the SAME values per frame (joint poses don't depend on time)
  - ``joint_velocities`` scale by ``k``       (rad/s scales with playback rate)
  - ``root_position`` per frame is REGENERATED at the new effective speed
  - ``root_quaternion`` unchanged (constant Rx(-90deg) spawn quat)
  - reported ``fps`` is also scaled by ``k`` so dataloader integrates correctly

Why this is approximately right: a real animal walking faster has ~proportionally
faster joint motion and proportionally faster body motion. Stretching by a single
``k`` is an approximation (real gait amplitude/frequency relationship is nonlinear)
but it's far better than a single fixed-speed reference.

We DO NOT touch the originals. Outputs go to ``deer_walk_k0p6.npy`` etc.

Usage:
    python tools/time_stretch_motions.py
    # Then update DEER_USD_AMP_WALK_RUN_CFG dataset to include the new files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
MOTIONS_DIR = REPO / "assets" / "motions"


# Speeds we want covered by the discriminator dataset, expressed as
# multipliers of the original clip speed. Walk source = 0.8 m/s,
# run source = 2.4 m/s.
#
# Walk variants -> covers 0.4 - 1.2 m/s (encompasses our 0.3-1.0 cmd range).
# Run  variants -> covers 1.5 - 3.0 m/s (for future broader cmd range).
WALK_FACTORS = (0.5, 0.75, 1.0, 1.25)         # 0.4, 0.6, 0.8, 1.0 m/s
RUN_FACTORS  = (0.625, 0.85, 1.0, 1.25)        # 1.5, 2.0, 2.4, 3.0 m/s


def stretch_clip(payload: dict, k: float) -> dict:
    """Return a new payload with playback time-scaled by factor ``k``.

    Implementation note: amp_rsl_rl's MotionLoader IGNORES the stored
    ``joint_velocities`` field — it recomputes from ``joint_positions``
    and the simulation dt (see motion_loader.py:344-353). Same for
    ``base_lin_velocities``: derived from ``root_position`` differences
    over time. So the only field that drives effective playback speed
    is ``fps``: scaling fps by k makes the loader treat each frame's
    position deltas as occurring over 1/k as much time, multiplying all
    derived velocities by k.

    We keep the joint_positions and root_position values invariant per
    frame (the body has moved the same amount per frame, but each frame
    now represents 1/k less real time -> body & joints move k times
    faster).
    """
    fps_in = float(payload["fps"])

    return {
        "joints_list": list(payload["joints_list"]),
        # Per-frame values unchanged (just played at a different rate).
        "joint_positions": np.asarray(
            payload["joint_positions"], dtype=np.float32
        ).copy(),
        # Stored but ignored by loader; scale anyway for self-consistency
        # in case anything else reads them.
        "joint_velocities": np.asarray(
            payload["joint_velocities"], dtype=np.float32
        ).copy() * k,
        "root_position": np.asarray(
            payload["root_position"], dtype=np.float32
        ).copy(),
        "root_quaternion": np.asarray(
            payload["root_quaternion"], dtype=np.float32
        ).copy(),
        # Only field that actually changes effective playback speed.
        "fps": fps_in * k,
    }


def factor_label(k: float) -> str:
    """Filename-safe label for a stretch factor: 0.5 -> k0p50, 1.25 -> k1p25."""
    s = f"{k:.2f}"
    return "k" + s.replace(".", "p")


def main() -> None:
    sources = [
        ("deer_walk", WALK_FACTORS),
        ("deer_run",  RUN_FACTORS),
    ]
    written = []
    for stem, factors in sources:
        src = MOTIONS_DIR / f"{stem}.npy"
        if not src.is_file():
            print(f"  [skip] {src} not found")
            continue
        payload = np.load(src, allow_pickle=True).item()
        T = payload["joint_positions"].shape[0]
        fps_src = float(payload["fps"])
        speed_src = -float(payload["root_position"][-1, 1]) / (T / fps_src)
        print(f"\n=== {stem}: T={T} @ {fps_src:.1f} fps, speed={speed_src:.3f} m/s ===")
        for k in factors:
            out = stretch_clip(payload, k)
            tag = factor_label(k)
            out_path = MOTIONS_DIR / f"{stem}_{tag}.npy"
            np.save(out_path, out, allow_pickle=True)
            print(f"  k={k:.3f} -> {out_path.name}: "
                  f"speed_out={speed_src*k:.3f} m/s, "
                  f"fps_out={out['fps']:.2f}")
            written.append(out_path.name)

    print(f"\nWrote {len(written)} time-stretched clips:")
    for w in written:
        print(f"  {w}")
    print("\nNext step: add these stems with weights to "
          "training/agents/deer_usd_amp_cfg.py:DEER_USD_AMP_WALK_RUN_CFG['dataset']['datasets'].")


if __name__ == "__main__":
    main()
