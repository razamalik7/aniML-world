"""Measure the geometric correspondence between the two deer rigs.

    python tools/cv/rig_correspondence.py [--json <out>]

WHY THIS EXISTS
---------------
The CV pipeline straddles two rigs that are *not* the same geometry:

  * the **artist armature** (`Deer_001_rig`, 33 bones) -- what we render, and
    therefore what image-space labels are defined against;
  * the **URDF** (25 DOF, 27 links) -- what AMP consumes, and therefore what the
    fitting stage must solve for.

If a keypoint means one thing in the renders and a slightly different thing in
the URDF fit, every downstream number is quietly biased and nothing errors. So
before building the fitting stage, measure the correspondence.

METHOD
------
1. Align the two rigs using only the **axial** keypoints -- nose, skull, neck,
   withers, tail root. Those are anatomically unambiguous in both rigs, so they
   give an honest global transform that was not fitted to the limbs.
   Similarity (Umeyama: rotation + uniform scale + translation), because the two
   rigs may differ in overall scale.
2. Then, for each URDF leg keypoint, report the **nearest armature bone point**
   and its residual. Nothing about the limb correspondence is assumed; the
   geometry decides.
3. Report the named-correspondence hypotheses side by side so the off-by-one
   question is settled by numbers rather than by reading bone names.

The hypothesis under test: the URDF's leg chain skips one armature bone. The
Unity->npy converter maps `pelvis_to_HR_thigh` onto armature bone `shoulder.R`,
which would make URDF "hip" = `shoulder.R` head. But the URDF joint *origin*
offsets match `thigh.R`'s geometry exactly, which would make URDF "hip" =
`thigh.R` head -- one bone further down. Those cannot both be right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deer_rig import DeerRig
from keypoint_schema import DeerKeypoints

DATA = Path(__file__).resolve().parent / "data" / "artist_rig_rest.json"

# Anatomically unambiguous in both rigs -- used to fit the global transform.
# Deliberately excludes every limb keypoint.
AXIAL_PAIRS: list[tuple[str, str]] = [
    # (URDF schema name, artist schema name)
    ("nose", "nose"),
    ("skull_base", "skull_base"),
    ("neck_base", "neck_base"),
    ("withers", "withers"),
    ("tail_root", "tail_base"),
]

# The two competing readings of the leg chain.
HYP_BY_NAME: dict[str, str] = {
    "L_shoulder": "FL_shoulder", "L_elbow": "FL_elbow",
    "L_f_ankle": "FL_carpus",    "L_f_toe": "FL_fetlock",
    "R_shoulder": "FR_shoulder", "R_elbow": "FR_elbow",
    "R_f_ankle": "FR_carpus",    "R_f_toe": "FR_fetlock",
    "L_hip": "HL_hip",   "L_knee": "HL_knee",
    "L_h_ankle": "HL_hock", "L_h_toe": "HL_fetlock",
    "R_hip": "HR_hip",   "R_knee": "HR_knee",
    "R_h_ankle": "HR_hock", "R_h_toe": "HR_fetlock",
}

HYP_SHIFTED: dict[str, str] = {
    "L_shoulder": "FL_elbow",  "L_elbow": "FL_carpus",
    "L_f_ankle": "FL_fetlock", "L_f_toe": "FL_hoof",
    "R_shoulder": "FR_elbow",  "R_elbow": "FR_carpus",
    "R_f_ankle": "FR_fetlock", "R_f_toe": "FR_hoof",
    "L_hip": "HL_knee",   "L_knee": "HL_hock",
    "L_h_ankle": "HL_fetlock", "L_h_toe": "HL_hoof",
    "R_hip": "HR_knee",   "R_knee": "HR_hock",
    "R_h_ankle": "HR_fetlock", "R_h_toe": "HR_hoof",
}


def umeyama(src: np.ndarray, dst: np.ndarray,
            with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping src onto dst.

    Returns (scale, R, t) such that `scale * (R @ src.T).T + t ~= dst`.
    Umeyama 1991, including the reflection guard -- without it a
    near-degenerate point set can produce a mirrored fit that looks numerically
    excellent and is anatomically nonsense (it would swap left and right).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / n
        scale = float((D * np.diag(S)).sum() / var_s) if var_s > 0 else 1.0
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def apply_sim(scale: float, R: np.ndarray, t: np.ndarray,
              pts: np.ndarray) -> np.ndarray:
    return scale * (R @ np.asarray(pts, dtype=np.float64).T).T + t


# Limb JOINT-CENTRE correspondences, under the 'shifted' reading. These are
# true joint centres in BOTH rigs -- no mesh geometry involved -- which makes
# them far better alignment anchors than the axial set, whose URDF side is
# derived from convex-hull proxy meshes (`withers` is the extreme vertex of
# `chest_visual.stl`, a 36-triangle hull, and lands ~19 cm out).
LIMB_JOINT_PAIRS: list[tuple[str, str]] = [
    ("L_shoulder", "FL_elbow"), ("L_elbow", "FL_carpus"),
    ("L_f_ankle", "FL_fetlock"),
    ("R_shoulder", "FR_elbow"), ("R_elbow", "FR_carpus"),
    ("R_f_ankle", "FR_fetlock"),
    ("L_hip", "HL_knee"), ("L_knee", "HL_hock"), ("L_h_ankle", "HL_fetlock"),
    ("R_hip", "HR_knee"), ("R_knee", "HR_hock"), ("R_h_ankle", "HR_fetlock"),
]

ANCHOR_SETS = {
    "axial": AXIAL_PAIRS,
    "limbs": LIMB_JOINT_PAIRS,
    "both": AXIAL_PAIRS + LIMB_JOINT_PAIRS,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=None, help="write findings as JSON")
    ap.add_argument("--data", default=str(DATA),
                    help="artist rig dump from blender_dump_rig.py")
    ap.add_argument("--anchor", choices=sorted(ANCHOR_SETS), default="axial",
                    help="which correspondences fit the global transform")
    ap.add_argument("--no-scale", action="store_true",
                    help="fit a rigid transform (no uniform scale)")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(
            f"missing {data_path}. Regenerate with:\n"
            f"  blender -b assets/blender/deer_standing.blend "
            f"--python tools/cv/blender_dump_rig.py -- --out {data_path}")
    art = json.loads(data_path.read_text(encoding="utf-8"))
    art_kp: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float64) for k, v in art["keypoints"].items()}
    art_bones: dict[str, dict] = art["bones"]

    # Every distinct armature bone point, for the nearest-point search.
    art_points: dict[str, np.ndarray] = {}
    for bone, b in art_bones.items():
        art_points[f"{bone}.head"] = np.asarray(b["head"], dtype=np.float64)
        art_points[f"{bone}.tail"] = np.asarray(b["tail"], dtype=np.float64)

    rig = DeerRig()
    kps = DeerKeypoints(rig)
    urdf_pos = kps.positions()          # rest pose, identity root
    urdf_kp = {n: urdf_pos[i] for i, n in enumerate(kps.names)}

    # ---- 1. global alignment ---------------------------------------------
    anchors = ANCHOR_SETS[args.anchor]
    src = np.array([urdf_kp[a] for a, _ in anchors])
    dst = np.array([art_kp[b] for _, b in anchors])
    scale, R, t = umeyama(src, dst, with_scale=not args.no_scale)
    axial_res = np.linalg.norm(apply_sim(scale, R, t, src) - dst, axis=1)

    print("=" * 72)
    print("URDF rig  vs  artist armature  -- rest pose correspondence")
    print("=" * 72)
    print(f"\nGlobal similarity fitted on {len(anchors)} "
          f"'{args.anchor}' correspondences"
          f"{' (rigid, no scale)' if args.no_scale else ''}")
    print(f"  scale (artist / URDF) : {scale:.4f}")
    print(f"  rotation from identity: "
          f"{np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))):.2f} deg")
    print(f"  translation           : {np.round(t, 4).tolist()}")
    print(f"  anchor residual: mean {axial_res.mean()*100:.2f} cm, "
          f"max {axial_res.max()*100:.2f} cm")
    for (a, b), r in zip(anchors, axial_res):
        print(f"     {a:<12s} -> {b:<12s} {r*100:6.2f} cm")

    # Always report the axial set too, so the reader can see whether the body
    # axis agrees under whichever alignment was chosen.
    if args.anchor != "axial":
        ax_src = np.array([urdf_kp[a] for a, _ in AXIAL_PAIRS])
        ax_dst = np.array([art_kp[b] for _, b in AXIAL_PAIRS])
        ax_res = np.linalg.norm(apply_sim(scale, R, t, ax_src) - ax_dst, axis=1)
        print(f"\n  AXIAL keypoints under this alignment "
              f"(mean {ax_res.mean()*100:.2f} cm):")
        for (a, b), r in zip(AXIAL_PAIRS, ax_res):
            print(f"     {a:<12s} -> {b:<12s} {r*100:6.2f} cm")

    # ---- 2. nearest armature point for each URDF limb keypoint ----------
    limb_names = [n for n in kps.names
                  if n.startswith(("L_", "R_")) and "eye" not in n
                  and "antler" not in n]
    print(f"\nNearest armature bone point for each URDF limb keypoint")
    print("(alignment above was fitted WITHOUT limbs, so this is independent)")
    print(f"\n  {'URDF keypoint':<14} {'nearest armature point':<24} {'dist':>8}   "
          f"{'by-name hyp':<14}{'shifted hyp':<14}")
    print(f"  {'-'*14} {'-'*24} {'-'*8}   {'-'*14}{'-'*14}")

    nearest_map: dict[str, str] = {}
    rows = []
    for n in limb_names:
        p = apply_sim(scale, R, t, urdf_kp[n][None, :])[0]
        best, bestd = None, np.inf
        for bn, bp in art_points.items():
            d = float(np.linalg.norm(bp - p))
            if d < bestd:
                best, bestd = bn, d
        nearest_map[n] = best
        h0 = HYP_BY_NAME.get(n)
        h1 = HYP_SHIFTED.get(n)
        d0 = (float(np.linalg.norm(art_kp[h0] - p))
              if h0 in art_kp else float("nan"))
        d1 = (float(np.linalg.norm(art_kp[h1] - p))
              if h1 in art_kp else float("nan"))
        rows.append({"urdf": n, "nearest": best, "dist_cm": bestd * 100,
                     "by_name": h0, "by_name_cm": d0 * 100,
                     "shifted": h1, "shifted_cm": d1 * 100})
        print(f"  {n:<14} {best:<24} {bestd*100:7.2f}cm  "
              f"{d0*100:>7.2f}cm     {d1*100:>7.2f}cm")

    valid0 = [r["by_name_cm"] for r in rows if np.isfinite(r["by_name_cm"])]
    valid1 = [r["shifted_cm"] for r in rows if np.isfinite(r["shifted_cm"])]
    print(f"\n  hypothesis 'by name'  : mean {np.mean(valid0):6.2f} cm  "
          f"max {np.max(valid0):6.2f} cm  (n={len(valid0)})")
    print(f"  hypothesis 'shifted'  : mean {np.mean(valid1):6.2f} cm  "
          f"max {np.max(valid1):6.2f} cm  (n={len(valid1)})")
    winner = "shifted" if np.mean(valid1) < np.mean(valid0) else "by name"
    print(f"  => '{winner}' fits better")

    # ---- 3. limb segment lengths, both rigs -----------------------------
    print("\nLimb segment lengths (URDF, metres) vs the armature bones the "
          "nearest-point search picked:")
    chains = [
        ("front L", ["L_shoulder", "L_elbow", "L_f_ankle", "L_f_toe"]),
        ("hind L",  ["L_hip", "L_knee", "L_h_ankle", "L_h_toe"]),
    ]
    for label, chain in chains:
        segs = []
        for a, b in zip(chain[:-1], chain[1:]):
            segs.append(float(np.linalg.norm(urdf_kp[b] - urdf_kp[a])))
        print(f"  {label:<8} URDF: " +
              "  ".join(f"{s:.3f}" for s in segs) +
              f"   (total {sum(segs):.3f})")

    verdict = {
        "scale_artist_over_urdf": scale,
        "axial_residual_mean_cm": float(axial_res.mean() * 100),
        "axial_residual_max_cm": float(axial_res.max() * 100),
        "limb_rows": rows,
        "hypothesis_by_name_mean_cm": float(np.mean(valid0)),
        "hypothesis_shifted_mean_cm": float(np.mean(valid1)),
        "winner": winner,
        "nearest_map": nearest_map,
    }

    print("\n" + "=" * 72)
    if axial_res.mean() * 100 > 5.0:
        print("VERDICT: the two rigs do NOT agree even on the body axis "
              f"(mean {axial_res.mean()*100:.1f} cm). Treat them as different "
              "animals; a shared keypoint schema is not currently justified.")
    elif np.min([np.mean(valid0), np.mean(valid1)]) * 1.0 > 5.0:
        print("VERDICT: body axis agrees but the LIMBS do not match under "
              "either naming hypothesis. The URDF's leg segmentation genuinely "
              "differs from the armature's -- correspondence must be defined "
              "per-joint, from the nearest-point column above.")
    else:
        print(f"VERDICT: rigs agree; use the '{winner}' correspondence.")
    print("=" * 72)

    if args.json:
        Path(args.json).write_text(json.dumps(verdict, indent=1),
                                   encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
