"""Separate "the leg is FROZEN" from "the leg SWINGS but carries no weight".

The leg-participation table in gait_report.py measures CONTACT DUTY -- what fraction of
the time a hoof is loaded. It cannot distinguish two very different failures that both
read as a low number:

  A. the leg is kinematically dead        (joints barely move; the deer drags it)
  B. the leg swings through a normal arc  (but the deer's weight has shifted off it)

Those call for opposite fixes, and the whole phase-2 puzzle turns on which one it is:
exp3 -> exp3b cut ankle-walking from 33.5% to 3.2% but took front/hind load from 1.12
to 0.37. If the front legs went FROZEN, the expert-only normaliser made the
discriminator sharp enough that holding a limb still at a plausible pose scores better
than moving it imperfectly -- the same shape as the feet_air_time bug, where doing
nothing beat trying. If they still SWING, nothing was lost kinematically and the deer
merely re-balanced rearward, which is a far less serious defect.

Range of motion is measured against the EXPERT clips, because "small" is only
meaningful relative to what the reference actually does with that joint.

    python tools/leg_kinematics_vs_load.py --runs exp3 exp3b [--runs ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Front limbs are Clavicle/UpperArm/Forearm/Hand/Finger0; hind are Thigh/Calf/
# HorseLink/Foot/Toe0. Matching on these stems rather than hardcoding indices, so a rig
# change surfaces as "0 joints matched" instead of silently profiling the wrong limb.
LEGS = {
    "L-hind":  (("L_Thigh", "L_Calf", "L_HorseLink", "L_Foot", "L_Toe0"), "L_Toe0", "L_Foot"),
    "R-hind":  (("R_Thigh", "R_Calf", "R_HorseLink", "R_Foot", "R_Toe0"), "R_Toe0", "R_Foot"),
    "L-front": (("L_Clavicle", "L_UpperArm", "L_Forearm", "L_Hand", "L_Finger0"), "L_Finger0", "L_Hand"),
    "R-front": (("R_Clavicle", "R_UpperArm", "R_Forearm", "R_Hand", "R_Finger0"), "R_Finger0", "R_Hand"),
}
CONTACT_N = 5.0   # a hoof is "loaded" above this; body weight is 809 N


def joint_idx(names: list[str], stems: tuple[str, ...]) -> list[int]:
    """Joints whose name mentions one of the limb's links on EITHER side of the '_to_'."""
    out = []
    for i, n in enumerate(names):
        if any(s in n for s in stems):
            out.append(i)
    return out


def slip_ratios(names, q, rp, rq, fps):
    """Per-hoof slip = hoof horizontal speed / body speed, over the lowest quarter of
    that hoof's own height range. 0 = planted, ~2 = swinging.

    This is the metric that actually explains leg neglect (see
    docs/LEG_NEGLECT_findings.md §4): a neglected limb is neither frozen nor held high,
    it descends and never stops. Normalised by body speed because the reference's own
    slip rises with gait -- walk 0.04-0.12, trot 0.35-0.42, run 0.29-0.34.

    Returns None rather than raising if FK is unavailable: this runs as the last step of
    an overnight queue, and a missing renderer import must not take the whole report
    down with it.
    """
    try:
        from scipy.spatial.transform import Rotation

        from tools.render_playback_meshed import compute_link_transforms, parse_urdf_full
        om, _ = parse_urdf_full(REPO / "assets/deer_polyart_urdf/deer_polyart.urdf")
    except Exception:
        return None
    feet = [h for _, (_, h, _) in LEGS.items()]
    P = np.zeros((len(q), len(feet), 3))
    for t in range(len(q)):
        ja = {n: float(q[t, i]) for i, n in enumerate(names)}
        try:
            lt = compute_link_transforms(ja, om)
        except Exception:
            return None
        Rr = Rotation.from_quat(rq[t])
        for k, h in enumerate(feet):
            if h in lt:
                _, tl = lt[h]
                P[t, k] = Rr.apply(tl) + rp[t]
    V = np.linalg.norm(np.diff(P[:, :, :2], axis=0), axis=2) * fps
    body = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * fps
    out = {}
    for k, leg in enumerate(LEGS):
        h = P[:-1, k, 2]
        lo = h < np.percentile(h, 25)
        out[leg] = float(np.median(V[lo, k] / np.maximum(body[lo], 0.05))) if lo.any() else float("nan")
    return out, float(np.median(body))


def rom(q: np.ndarray, idx: list[int]) -> np.ndarray:
    """Per-joint range of motion in degrees, robust to single-frame spikes."""
    if not idx:
        return np.array([])
    a = np.degrees(q[..., idx])
    a = a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a
    return np.percentile(a, 95, axis=0) - np.percentile(a, 5, axis=0)


def load_expert(names: list[str], clips: list[str]) -> dict[str, float]:
    if not hasattr(np, "_core"):
        import numpy.core as _c
        sys.modules.setdefault("numpy._core", _c)
        for s in ("multiarray", "umath", "numeric", "_multiarray_umath"):
            m = getattr(_c, s, None)
            if m is not None:
                sys.modules.setdefault(f"numpy._core.{s}", m)
    qs = []
    for c in clips:
        p = REPO / "assets/motions" / f"{c}.npy"
        if not p.exists():
            continue
        d = np.load(str(p), allow_pickle=True).item()
        jn = list(d["joints_list"])
        q = np.asarray(d["joint_positions"], float)
        # reorder the clip's columns into the rollout's joint order
        col = {n: i for i, n in enumerate(jn)}
        take = [col.get(n, -1) for n in names]
        qq = np.zeros((q.shape[0], len(names)))
        for j, t in enumerate(take):
            if t >= 0:
                qq[:, j] = q[:, t]
        qs.append(qq)
    if not qs:
        return {}
    q = np.concatenate(qs, 0)
    return {leg: float(np.mean(rom(q, joint_idx(names, stems))))
            for leg, (stems, _, _) in LEGS.items()}


def analyse(tag: str, expert: dict[str, float] | None = None, suffix: str = "_rollout"):
    p = REPO / "docs" / f"{tag}{suffix}.npz"
    if not p.exists():
        print(f"\n{tag}: no rollout at {p.name} -- skipped")
        return None
    d = np.load(str(p), allow_pickle=True)
    names = [str(x) for x in d["joints_list"]]
    bodies = [str(x) for x in d["contact_bodies"]]
    q = np.asarray(d["joint_positions"], float)          # (T, E, J)
    f = np.abs(np.asarray(d["contact_forces"], float))   # (T, E, B)
    if expert is None:
        expert = load_expert(names, ["deer_polyart_walk_alltakes",
                                     "deer_polyart_trot", "deer_polyart_run"])

    print(f"\n{tag}")
    # The command is the CONDITION. A left/right load split measured while the deer was
    # commanded to turn left says nothing about neglect -- it says the deer corners.
    if "commands" in d.files:
        c = np.asarray(d["commands"], float)             # (T, E, 3) vx, vy, yaw
        print(f"  commanded  vx {c[:,:,0].mean():+.2f}  vy {c[:,:,1].mean():+.2f}  "
              f"yaw {np.degrees(c[:,:,2]).mean():+.1f} deg/s "
              f"(spread {np.degrees(c[:,:,2]).std():.1f})")
        if abs(np.degrees(c[:, :, 2]).mean()) > 3.0:
            print("  *** NET YAW COMMAND IS NON-ZERO -- left/right load is confounded "
                  "by cornering; compare only against runs at the same command")
    else:
        print("  commanded: NOT RECORDED (pre-2026-07-29 rollout). If this run used "
              "CMD_ANG_VEL_Z>0 its left/right split may be cornering, not neglect.")
    print(f"  {'leg':9s} {'share':>6s} {'meanN':>7s} {'d@5N':>6s} {'d@20N':>6s} "
          f"{'ROM':>7s} {'/exp':>6s}  verdict")
    rows, mean_f = {}, {}
    for leg, (stems, hoof, ankle) in LEGS.items():
        hb = bodies.index(hoof) if hoof in bodies else -1
        mean_f[leg] = float(f[:, :, hb].mean()) if hb >= 0 else 0.0
    tot = max(sum(mean_f.values()), 1e-6)

    for leg, (stems, hoof, ankle) in LEGS.items():
        idx = joint_idx(names, stems)
        r = float(np.mean(rom(q, idx))) if idx else float("nan")
        hb = bodies.index(hoof) if hoof in bodies else -1
        d5 = float((f[:, :, hb] > 5.0).mean()) if hb >= 0 else float("nan")
        d20 = float((f[:, :, hb] > 20.0).mean()) if hb >= 0 else float("nan")
        share = mean_f[leg] / tot
        ex = expert.get(leg, float("nan"))
        ratio = r / ex if ex and ex == ex else float("nan")
        # 0.25 share = its fair quarter. Under ~8% of the load is neglect however it
        # is reached; whether the limb also stopped MOVING is the separate question.
        if share < 0.08 and ratio < 0.5:
            verdict = "FROZEN (dead weight)"
        elif share < 0.08:
            verdict = "swings, carries nothing"
        elif ratio < 0.5:
            verdict = "loaded but stiff"
        else:
            verdict = "working"
        print(f"  {leg:9s} {share:6.2f} {mean_f[leg]:7.0f} {d5:6.2f} {d20:6.2f} "
              f"{r:6.1f}d {ratio:6.2f}  {verdict}")
        rows[leg] = (share, r, ratio, d5, d20)

    # CONFIG-INDEPENDENT NEGLECT METRICS. Which leg gets dropped is emergent, not
    # fixed: across 18 rollouts the lightest leg is L-front 12 times, R-front 4, L-hind
    # 2 -- and r12/r13 loaded L-front at 0.56 while neglecting R-front. So a metric
    # naming a specific leg cannot compare two runs. What IS universal is that some leg
    # is always dropped: every rollout measured has one leg at or below 0.11 of its
    # 0.25 fair share. Track the worst leg and the spread, not the leg's name.
    shares = np.array([rows[l][0] for l in LEGS])
    worst = float(shares.min())
    # normalised entropy: 1.0 = all four legs share the load equally, 0 = one leg
    # carries everything. Independent of WHICH leg, so it compares across runs.
    p = np.clip(shares, 1e-9, None); p = p / p.sum()
    balance = float(-(p * np.log(p)).sum() / np.log(len(p)))
    print(f"  LOAD BALANCE  worst leg {worst:.2f} (fair share 0.25)   "
          f"evenness {balance:.2f} (1.0 = all four equal)   "
          f"lightest = {min(LEGS, key=lambda l: rows[l][0])}")

    # SLIP -- the metric that separates a supporting limb from a passenger.
    sl = slip_ratios(names, q[:, 0], np.asarray(d["root_pos"], float)[:, 0],
                     np.asarray(d["root_quat_wxyz"], float)[:, 0][:, [1, 2, 3, 0]],
                     float(d["fps"]))
    if sl is not None:
        s, body = sl
        ref = ("walk 0.04-0.12" if body < 2.0 else
               "trot 0.35-0.42" if body < 5.0 else "run 0.29-0.34")
        print(f"  SLIP (0=planted) " + "  ".join(f"{k} {s[k]:.2f}" for k in LEGS)
              + f"   at {body:.2f} m/s, reference {ref}")

    fh_load = ((rows["L-front"][0] + rows["R-front"][0]) /
               max(rows["L-hind"][0] + rows["R-hind"][0], 1e-6))
    fh_rom = ((rows["L-front"][2] + rows["R-front"][2]) /
              max(rows["L-hind"][2] + rows["R-hind"][2], 1e-6))
    # How much of the "contact" is featherweight grazing rather than support? If duty
    # at 5 N is far above duty at 20 N, the hoof is brushing the ground, not standing
    # on it -- and a duty-based participation score counts that as a working leg.
    graze = np.mean([1.0 - (rows[l][4] / rows[l][3]) if rows[l][3] > 1e-6 else 0.0
                     for l in LEGS])
    print(f"  front/hind by LOAD {fh_load:.2f}   by MOTION {fh_rom:.2f}   "
          f"grazing contacts {100*graze:.0f}%")
    print(f"  -> {'limbs are FROZEN, not merely unloaded' if fh_rom < 0.6 else 'motion retained -- this is a WEIGHT shift, not a dead limb'}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["exp3", "exp3b"])
    ap.add_argument("--suffix", default="_rollout",
                    help="'_straight' for the matched-condition re-collection")
    a = ap.parse_args()

    first = next((r for r in a.runs
                  if (REPO / "docs" / f"{r}{a.suffix}.npz").exists()), None)
    if first is None:
        print(f"no rollouts found with suffix '{a.suffix}' for {a.runs}")
        return 1
    d = np.load(str(REPO / "docs" / f"{first}{a.suffix}.npz"), allow_pickle=True)
    names = [str(x) for x in d["joints_list"]]
    expert = load_expert(names, ["deer_polyart_walk_alltakes",
                                 "deer_polyart_trot", "deer_polyart_run"])
    print("EXPERT mean range of motion per limb (degrees):")
    for k, v in expert.items():
        print(f"  {k:9s} {v:6.1f}")
    print("\nduty = fraction of frames the HOOF is loaded (>5 N)")
    print("ROM  = mean per-joint 5-95 percentile spread over that limb's joints")

    for tag in a.runs:
        analyse(tag, expert, a.suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
