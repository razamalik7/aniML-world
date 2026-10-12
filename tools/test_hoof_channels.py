"""CPU test for the hoof height/speed AMP channels and the dataset-side patch.

Run before any queued run that sets ANIML_DEER_AMP_HOOF_H or _V. B (height only) and
G (height+speed) share this code path, and the width guard in train_amp.py CANNOT catch
a height/speed transposition -- 4+4 is 8 either way. That ordering is enforced by
convention in exactly two places, so it gets a test.

    C:/isaac-lab/env_isaaclab/Scripts/python.exe tools/test_hoof_channels.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy.core as _c                                        # noqa: E402
sys.modules.setdefault("numpy._core", _c)
for _s in ("multiarray", "umath", "numeric", "_multiarray_umath"):
    _m = getattr(_c, _s, None)
    if _m is not None:
        sys.modules.setdefault(f"numpy._core.{_s}", _m)

from tools.render_playback_meshed import parse_urdf_full       # noqa: E402
from tools.rig_pipeline.hoof_height_patch import (             # noqa: E402
    FEET, attach, reference_hoof_heights, reference_hoof_speeds,
)

URDF = REPO / "assets/deer_polyart_urdf/deer_polyart.urdf"
CLIP = "deer_polyart_walk"
FAILS: list[str] = []


def check(ok, name, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def fake_runner(n_frames: int):
    """Stand-in for the amp-rsl-rl runner, shaped like the REAL one.

    The first version of this mock exposed `runner.amp_data`, because that is what
    `attach` read. It passed, and B still crashed on its first line: the loader actually
    lives at `runner.alg.amp_data`. A mock built from the code under test only ever
    confirms the code's own assumptions. This one mirrors the library
    (amp_on_policy_runner builds the loader as a local and hands it to the algorithm,
    amp_ppo.py:140 keeps it as self.amp_data), and `test_real_attribute_path` below
    checks that claim against the installed package rather than against this file.
    """
    BASE = 82

    class FakeMotion:
        """Mimics MotionData closely enough to exercise the buffer rebuild."""
        def __init__(self, T):
            self.joint_positions = torch.zeros(T, 38)
            self.device = "cpu"
            self.T = T

        def __len__(self):
            return self.T

        def get_amp_dataset_obs(self, indices, *a, **kw):
            base = torch.zeros(len(indices), BASE)
            hh = getattr(self, "hoof_extra", None)
            return base if hh is None else torch.cat((base, hh[indices]), dim=1)

    md = FakeMotion(n_frames)
    # all_obs as AMPLoader precomputes it: one row per frame, BEFORE any patch
    idx = torch.arange(n_frames)
    loader = SimpleNamespace(
        motion_data=[md],
        velocity_representation=None,
        all_obs=md.get_amp_dataset_obs(idx),
        all_next_obs=md.get_amp_dataset_obs(torch.clamp(idx + 1, max=n_frames - 1)),
    )
    return SimpleNamespace(
        alg=SimpleNamespace(amp_data=loader),
        dataset_cfg={"amp_data_path": str(REPO / "assets/motions"),
                     "datasets": {CLIP: 1.0}},
    ), md, loader


def test_real_attribute_path() -> None:
    """Assert against the INSTALLED amp-rsl-rl, not against our own mock."""
    import inspect
    try:
        from amp_rsl_rl.algorithms import amp_ppo
        from amp_rsl_rl.runners import amp_on_policy_runner
    except Exception as e:
        check(False, "amp-rsl-rl importable", f"{type(e).__name__}: {e}")
        return
    alg_src = inspect.getsource(amp_ppo)
    run_src = inspect.getsource(amp_on_policy_runner)
    check("self.amp_data" in alg_src, "algorithm stores self.amp_data",
          "amp_ppo keeps the loader")
    check("self.amp_data" not in run_src, "runner does NOT store self.amp_data",
          "so runner.amp_data is wrong -- this is the bug that broke B")
    check("self.dataset_cfg" in run_src, "runner stores self.dataset_cfg",
          "attach reads amp_data_path/datasets from it")


def main() -> int:
    om, _ = parse_urdf_full(URDF)
    d = np.load(str(REPO / "assets/motions" / f"{CLIP}.npy"), allow_pickle=True).item()
    jl = [str(x) for x in d["joints_list"]]
    jp = np.asarray(d["joint_positions"], float)
    rp = np.asarray(d["root_position"], float)
    rq = np.asarray(d["root_quaternion"], float)
    fps = float(d["fps"])

    print("\n=== 1. reference channels are well-formed ===")
    H = reference_hoof_heights(None, om, jl, jp, rp, rq)
    V = reference_hoof_speeds(om, jl, jp, rp, rq, fps)
    check(H.shape == (len(jp), 4), "height shape", str(H.shape))
    check(V.shape == (len(jp), 4), "speed shape (no frame lost to the diff)", str(V.shape))
    check(np.isfinite(H).all() and np.isfinite(V).all(), "all finite")
    check((V >= 0).all(), "speed is a magnitude, never negative")
    # A planted hoof must reach near-zero world speed somewhere in the stride, else the
    # channel carries no stance information at all.
    check(V.min(axis=0).max() < 0.25, "every hoof plants at some point",
          f"worst per-hoof minimum {V.min(axis=0).max():.3f} m/s")

    print("\n=== 2. the last frame is held, not wrapped ===")
    # These clips are not loops. Wrapping would manufacture one huge fake velocity
    # spike per clip, and AMP samples transitions uniformly so it would be drawn often.
    check(np.allclose(V[-1], V[-2]), "final frame duplicated rather than wrapped")
    wrap = np.linalg.norm((rp[0] - rp[-1])[:2]) * fps
    check(V.max() < wrap or wrap < 1e-6, "no wrap-sized spike present",
          f"max channel {V.max():.2f} m/s vs a wrap would be ~{wrap:.2f} m/s")

    print("\n=== 2c. the loader really is where attach looks for it ===")
    test_real_attribute_path()

    print("\n=== 3. attach(): column count and ORDER ===")
    for height, speed, want in [(True, False, 4), (False, True, 4), (True, True, 8)]:
        r, md, _ = fake_runner(len(jp))
        n = attach(r, str(URDF), verbose=False, height=height, speed=speed)
        got = md.hoof_extra.shape[1]
        check(n == want and got == want, f"height={height} speed={speed} -> {want} cols",
              f"returned {n}, tensor {got}")

    print("\n=== 3b. the EXPERT BUFFERS are rebuilt (what B actually died on) ===")
    # AMPLoader fills all_obs/all_next_obs ONCE at construction, and
    # feed_forward_generator -- the only path amp_ppo.update samples experts from --
    # reads those buffers, never the method. Patching the method alone left the buffers
    # at 82 while the env produced 86, and B crashed 15 iterations in with
    # "expected 172 but got 164". Asserting the METHOD width is what let that pass.
    r, md, loader = fake_runner(len(jp))
    w0, rows0 = int(loader.all_obs.shape[1]), int(loader.all_obs.shape[0])
    attach(r, str(URDF), verbose=False, height=True, speed=True)
    check(int(loader.all_obs.shape[1]) == w0 + 8, "all_obs widened by 8",
          f"{w0} -> {int(loader.all_obs.shape[1])}")
    check(int(loader.all_next_obs.shape[1]) == w0 + 8, "all_next_obs widened by 8",
          f"{w0} -> {int(loader.all_next_obs.shape[1])}")
    check(int(loader.all_obs.shape[0]) == rows0, "row count UNCHANGED",
          f"{rows0} rows (per_frame_weights would index off the end otherwise)")
    check(bool((loader.all_obs[:, w0:] != 0).any()), "the added columns carry real data",
          "non-zero hoof channels present in the buffer")
    # next_obs must be the frame-SHIFTED view: all_next_obs[i] == all_obs[i+1].
    # If the rebuild used idx for both, the discriminator would be shown transitions
    # that never move, and it would still train perfectly happily.
    shifted = bool(torch.allclose(loader.all_next_obs[:-1, w0:],
                                  loader.all_obs[1:, w0:]))
    check(shifted, "all_next_obs[i] == all_obs[i+1] on the hoof columns",
          "the rebuild uses clamp(idx+1), matching motion_loader.py:350")

    r, md, loader = fake_runner(len(jp))
    attach(r, str(URDF), verbose=False, height=True, speed=True)
    cols = md.hoof_extra.numpy()
    # Heights are ~0.03-0.25 m and speeds reach several m/s. If the halves were swapped
    # the maxima would swap with them.
    hmax, vmax = cols[:, :4].max(), cols[:, 4:].max()
    check(hmax < 1.0 < vmax, "heights occupy cols 0-3, speeds cols 4-7",
          f"max of first half {hmax:.2f} (metres), second half {vmax:.2f} (m/s)")

    print("\n=== 4. resampling to the sim dt preserves the channel ===")
    r, md, loader = fake_runner(len(jp) * 4)          # MotionData resampled 4x longer
    attach(r, str(URDF), verbose=False, height=True, speed=True)
    up = md.hoof_extra.numpy()
    check(up.shape == (len(jp) * 4, 8), "resampled to MotionData length", str(up.shape))
    check(abs(up[:, 4:].max() - vmax) < 0.5 * vmax,
          "speeds NOT rescaled by the resample ratio",
          f"{vmax:.2f} -> {up[:,4:].max():.2f} m/s (a diff-after-resample bug gives ~4x)")

    print("\n" + "=" * 66)
    if FAILS:
        print(f"FAILED — {len(FAILS)} check(s): " + "; ".join(FAILS))
    else:
        print("ALL PASS — hoof channels agree in count, order and scale.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
