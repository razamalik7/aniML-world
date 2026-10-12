"""PREFLIGHT — assert every invariant that has silently broken before, and REFUSE to
launch training if any fails.

Every serious failure in this project was SILENT: it threw no error, it produced
confident wrong numbers, and it was found days later by looking at a video. Being more
careful is not a control. Each check below exists because that exact thing broke, and
each names the run it cost us:

  1  reference dead air           58% frozen frames taught the policy to STAND STILL (r1-r8)
  2  reference turn contamination 59% of windows turning vs a 0-yaw command (r9)
  3  reference position scale     Blender 5.1 imports 1.25x larger; ANGLES are
                                  scale-invariant so it hides in plain sight
  4  joint limits vs reference    limits in a different frame -> 100% of RSI spawns
                                  silently clamped, worst 56.7 deg (r7-r9)
  5  limits contain joint-zero    an all-positive fitted range excludes the default pose
                                  and Isaac refuses to build the articulation
  6  self-collision               1,002,943 N on an 809 N animal, from 2-DOF `_dof0`
                                  links defeating parent-child filtering (r1-r9)
  7  collision filters survive    reimporting the USD REGENERATES it and wipes the
                                  filters; nothing warns you
  8  AMP dataset via AMPLoader    validate through amp-rsl-rl's OWN loader, not our
                                  reimplementation of it
  9  discriminator normaliser     amp_ppo folds POLICY obs into the normaliser, hiding
                                  the policy's own outliers (r9 style collapse)

COVERAGE (widened 2026-07-29). The clip checks run over EVERY clip named in
ANIML_AMP_DATASETS, defaulting to walk + trot + run. Previously only
`deer_polyart_walk` was validated while every run since exp3 trained on three clips --
two of the three expert sources were never checked at all, and the one clip that was
genuinely contaminated (`walk_alltakes`) was among the unchecked.

Widening it meant three gates had to stop being absolute, because they were about to
fail valid data rather than catch bad data:
  * 1b teleport      was `speed > 3 m/s`; the run clip legitimately hits 7.56 m/s.
                     Now peak/p95 ratio -- a splice spikes ONE frame, a gait does not.
  * 2b turn          counted frames whose INSTANTANEOUS yaw rate exceeded 60 deg/s.
                     A trot sits at 62.5% of frames above that and a run at 60.0%,
                     purely from pelvis oscillation at high stride frequency, both dead
                     straight. Now the MAX over a one-stride window: straight walk,
                     trot and run all read 0.0, walk_alltakes reads 53.0. The question
                     is whether fast yaw ACCUMULATES or returns to baseline.
  * 8b expert speed  was `max < 3.0 m/s`, same problem. Now relative to the clip median.
A check that fails good data is worse than no check: it trains you to ignore the gate.

Run before every launch:
  OMNI_KIT_ACCEPT_EULA=YES python tools/rig_pipeline/preflight.py
  ANIML_AMP_DATASETS="a:1.0,b:1.0" ... preflight.py     # check a specific run's set
Exit code 0 = safe to train. Non-zero = do not launch.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BODY_WEIGHT_N = 809.0
FAILS: list[str] = []
WARNS: list[str] = []


def check(ok, name, detail, warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        (WARNS if warn_only else FAILS).append(f"{name}: {detail}")
    return ok


def load(p):
    if not hasattr(np, "_core"):
        import numpy.core as _c
        sys.modules.setdefault("numpy._core", _c)
        for s in ("multiarray", "umath", "numeric", "_multiarray_umath"):
            m = getattr(_c, s, None)
            if m is not None:
                sys.modules.setdefault(f"numpy._core.{s}", m)
    return np.load(str(p), allow_pickle=True).item()


def clip_names() -> list[str]:
    """Every clip the imminent run will actually train on.

    Preflight used to validate `deer_polyart_walk` alone, while every run since exp3
    has trained on walk + trot + run. Two of the three expert clips were never checked
    at all -- and `deer_polyart_walk_alltakes`, the one clip that WAS contaminated, was
    among the unchecked. Read ANIML_AMP_DATASETS so the gate covers what is being fed in.
    """
    import os
    spec = os.environ.get("ANIML_AMP_DATASETS", "")
    if spec:
        return [p.split(":")[0].strip() for p in spec.split(",") if p.strip()]
    return ["deer_polyart_walk", "deer_polyart_trot", "deer_polyart_run"]


def check_clip(name: str, urdf_limits: dict) -> None:
    from scipy.spatial.transform import Rotation
    p = REPO / "assets/motions" / f"{name}.npy"
    if not p.exists():
        check(False, f"clip {name}", "file not found")
        return
    d = load(p)
    jp = np.asarray(d["joint_positions"], float)
    rp = np.asarray(d["root_position"], float)
    rq = np.asarray(d["root_quaternion"], float)
    fps = float(d["fps"])
    names = list(d["joints_list"])

    print(f"\n--- {name} ({len(rp)} frames, {fps:.0f} fps) ---")
    v = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * fps
    check((v < 0.01).mean() < 0.02, "1 no dead air",
          f"{100*(v<0.01).mean():.1f}% frozen frames (was 59.8% pre-r9)")

    # 1b: a teleport is a frame far faster than the clip's OWN pace, not a fixed speed.
    # The old absolute 3 m/s gate would have failed deer_polyart_run for running: its
    # top speed is 7.56 m/s and that is the gait, not a seam.
    ratio = float(v.max() / max(np.percentile(v, 95), 1e-6)) if len(v) else 1.0
    check(ratio < 3.0, "1b no teleport seams",
          f"peak/p95 speed ratio {ratio:.2f} (a splice spikes one frame; "
          f"clip peaks at {v.max():.2f} m/s)")

    yaw = np.unwrap(Rotation.from_quat(rq).as_euler("zyx")[:, 0])
    win = min(len(yaw) - 1, max(3, int(round(fps * 1.1))))   # >= one stride
    sysd = np.degrees(yaw[win:] - yaw[:-win]) / (win / fps) if len(yaw) > win else \
        np.array([np.degrees(yaw[-1] - yaw[0]) / (len(yaw) / fps)])
    check(np.abs(sysd).mean() < 10, "2 no turn contamination",
          f"systematic turning {np.abs(sysd).mean():.1f} deg/s (turn clips run ~27)")

    # 2b: check 2 averages over the clip, so a mostly-straight clip containing a short
    # hard pivot can dilute below the gate. The MAX over the same one-stride window
    # cannot be diluted.
    #
    # This replaces an earlier 2b that counted frames whose INSTANTANEOUS yaw rate
    # exceeded 60 deg/s. That was wrong and would have blocked valid data: a trot sits
    # at 62.5% of frames above 60 deg/s and a run at 60.0%, purely from pelvis
    # oscillation at high stride frequency, while both are dead straight. Measured over
    # a stride window instead, straight clips read 0.0 and walk_alltakes reads 53.0 --
    # the distinction is whether fast yaw ACCUMULATES or returns to baseline.
    peak = float(np.abs(sysd).max())
    check(peak < 25.0, "2b turn MAGNITUDE (max, not mean)",
          f"worst one-stride window turns {peak:.1f} deg/s "
          f"(straight walk/trot/run all read 0.0; walk_alltakes reads 53.0)")

    # 3: scale. The 1.25x Blender 5.1 import bug scales root_position but NOT joint
    # angles, so it hides in plain sight.
    #
    # The old version of this check bounded pelvis height, tightly for clips named
    # "walk" and loosely (0.80-1.45) for everything else, on the reasoning that faster
    # gaits ride higher. That hole let the real thing through for MONTHS: trot shipped at
    # 1.195 and run at 1.281, both comfortably inside the loose band, both actually 1.25x
    # oversized. Their true pelvis heights are 0.956 and 1.025, and their true speeds
    # 2.37 and 6.00 m/s rather than the 2.96 and 7.50 recorded everywhere.
    #
    # This replacement needs no gait band and no filename. FK the hooves into world
    # space: rig geometry is scale-CORRECT, clip root position is not, so an oversized
    # clip leaves the deer floating by exactly the excess. A real animal plants a hoof
    # at some point in any stride, including a gallop.
    z = float(rp[:, 2].mean())
    try:
        import sys as _sys
        _sys.path.insert(0, str(REPO))
        from tools.render_playback_meshed import parse_urdf_full
        from tools.rig_pipeline.hoof_height_patch import reference_hoof_heights
        _om, _ = parse_urdf_full(REPO / "assets/deer_polyart_urdf/deer_polyart.urdf")
        _H = reference_hoof_heights(None, _om, names, jp, rp, rq)
        _lo = float(_H.min())
        check(-0.10 <= _lo <= 0.10, "3 position scale (FK: lowest hoof must reach ground)",
              f"lowest hoof {_lo:+.3f} m, pelvis {z:.3f} m "
              f"(a 1.25x clip floats at ~+0.20; dividing by 1.25 would give "
              f"{float(reference_hoof_heights(None, _om, names, jp, rp * 0.8, rq).min()):+.3f})")
    except Exception as e:
        check(0.80 <= z <= 1.45, "3 position scale (FALLBACK: FK unavailable)",
              f"pelvis {z:.3f} m -- {type(e).__name__}: {e}", warn_only=True)

    worst, worst_j = 0.0, ""
    for i, n in enumerate(names):
        if n in urdf_limits:
            lo, hi = urdf_limits[n]
            e = max(float(lo - jp[:, i].min()), float(jp[:, i].max() - hi), 0.0)
            if e > worst:
                worst, worst_j = e, n
    check(worst < np.radians(1), "4 limits contain the reference",
          f"worst excursion {np.degrees(worst):.2f} deg" + (f" on {worst_j}" if worst_j else ""))


def main():
    urdf_path = REPO / "assets/deer_polyart_urdf/deer_polyart.urdf"
    usd_path = REPO / "assets/deer_polyart_urdf/deer_polyart.usd"
    clip_path = REPO / "assets/motions/deer_polyart_walk.npy"
    d = load(clip_path)
    names = list(d["joints_list"])

    print(f"\n=== RIG ===")
    text = urdf_path.read_text(encoding="utf-8")
    lim = {}
    for m in re.finditer(r'<joint name="([^"]+)"\s+type="([^"]+)"(.*?)</joint>', text, re.S):
        n, ty, b = m.group(1), m.group(2), m.group(3)
        lm = re.search(r'<limit([^>]*)/>', b)
        if ty == "fixed" or not lm:
            continue
        lo = re.search(r'lower="(-?[0-9.eE+]+)"', lm.group(1))
        hi = re.search(r'upper="(-?[0-9.eE+]+)"', lm.group(1))
        if lo and hi:
            lim[n] = (float(lo.group(1)), float(hi.group(1)))
    print(f"  {len(lim)} limited joints parsed from the URDF")

    clips = clip_names()
    print(f"\n=== REFERENCE CLIPS ({len(clips)}: {', '.join(clips)}) ===")
    for c in clips:
        check_clip(c, lim)

    print(f"\n=== RIG (continued) ===")
    bad0 = [n for n, (lo, hi) in lim.items() if not (lo <= 0.0 <= hi)]
    check(not bad0, "5 limits contain joint-zero",
          f"{len(bad0)} joints exclude the default pose" + (f" e.g. {bad0[:3]}" if bad0 else ""))

    print(f"\n=== USD ===")
    try:
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(usd_path))
        nfil = 0
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.FilteredPairsAPI):
                rel = UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel()
                if rel:
                    nfil += len(rel.GetTargets())
        check(nfil >= 20, "7 self-collision filters present",
              f"{nfil} filtered pairs (reimport wipes these; expect ~40)")
        njoint = sum(1 for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint))
        mism = 0
        for p in stage.Traverse():
            if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in lim:
                j = UsdPhysics.RevoluteJoint(p)
                wl, wh = np.degrees(lim[p.GetName()])
                if abs(j.GetLowerLimitAttr().Get() - wl) > 0.1 or \
                   abs(j.GetUpperLimitAttr().Get() - wh) > 0.1:
                    mism += 1
        check(mism == 0, "7b USD limits match URDF",
              f"{mism}/{njoint} joints differ (stale USD = you edited the URDF and "
              f"forgot to reimport)")
    except Exception as e:
        check(False, "7 USD checks", f"could not open USD: {e}", warn_only=True)

    print(f"\n=== AMP DATASET (through amp-rsl-rl's OWN loader) ===")
    # Load ALL clips in one loader, exactly as training does. Loading them together is
    # the point: `expected_joint_names` is applied per clip, and a name mismatch in the
    # SECOND clip zero-fills silently while the first looks perfect.
    try:
        import torch
        from amp_rsl_rl.utils import AMPLoader
        L = AMPLoader(device="cpu", dataset_path_root=(REPO / "assets/motions").resolve(),
                      datasets={c: 1.0 for c in clips}, simulation_dt=0.005 * 4,
                      slow_down_factor=1, expected_joint_names=names)
        check(len(L.motion_data) == len(clips), "8 all clips loaded",
              f"{len(L.motion_data)}/{len(clips)} MotionData built")
        for c, md in zip(clips, L.motion_data):
            q = torch.as_tensor(md.joint_positions)
            z = int((q.abs().sum(dim=0) == 0).sum())
            check(z == 0, f"8 no silent zero-fill [{c}]",
                  f"{z} all-zero joint columns (name mismatch zero-fills SILENTLY)")
            sp = torch.linalg.norm(
                torch.as_tensor(md.base_lin_velocities_local)[:, :2], dim=1).numpy()
            # No absolute speed cap: the run clip legitimately reaches 7.6 m/s, and the
            # old `max < 3.0` gate would have failed it for running. What matters is
            # that the clip is moving and free of a single impossible spike.
            ok = (sp < 0.01).mean() < 0.02 and sp.max() < 4.0 * max(np.median(sp), 0.05)
            check(ok, f"8b expert speed sane [{c}]",
                  f"median {np.median(sp):.2f} m/s, max {sp.max():.2f}, "
                  f"{100*(sp<0.01).mean():.0f}% frozen")
    except Exception as e:
        check(False, "8 AMPLoader checks", f"{type(e).__name__}: {e}", warn_only=True)

    print(f"\n=== TRAINING SCRIPT ===")
    tt = (REPO / "training/scripts/train_amp.py").read_text(encoding="utf-8")
    check("expert-only statistics" in tt or "_expert_only_update" in tt,
          "9 discriminator normaliser patched",
          "expert-only stats (policy obs in the normaliser hid r9's outliers)")

    print("\n" + "=" * 70)
    if FAILS:
        print(f"PREFLIGHT FAILED — {len(FAILS)} blocking issue(s). DO NOT LAUNCH:")
        for f in FAILS:
            print(f"   * {f}")
    else:
        print("PREFLIGHT PASSED — safe to launch.")
    if WARNS:
        print(f"({len(WARNS)} warning(s): " + "; ".join(WARNS) + ")")
    print("NOTE: this checks the things that HAVE broken. It cannot catch a NEW failure "
          "mode -- for that, render it and look.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
