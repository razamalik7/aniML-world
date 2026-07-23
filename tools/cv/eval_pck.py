"""PCK evaluation for 2D keypoint predictions against ground truth.

    python tools/cv/eval_pck.py --gt labels.json --pred preds.json
    python tools/cv/eval_pck.py --gt labels.json --pred preds.json --alpha 0.05 0.1

This is the measurement the decisive experiment needs: train a 2D pose net on
synthetic renders, run it on hand-labelled real deer frames, and report PCK. Every
number quoted in `docs/research/2026-07-23_synthetic_vs_handlabel.md` is
PCK@0.05, so this must compute it the same way or the comparison is meaningless.

PCK@alpha
---------
A predicted keypoint is correct if it lands within `alpha * s` pixels of ground
truth, where `s` is a size normaliser. **Which normaliser you pick changes the
number by a lot**, and papers are not always explicit, so this tool makes it a
flag and prints which one it used:

  `bbox_max`  (default)  s = max(bbox_w, bbox_h). What AP-10K and the animal-pose
                         papers this project cites generally use.
  `bbox_diag`            s = sqrt(bbox_w^2 + bbox_h^2). More forgiving.
  `torso`                s = distance between two named keypoints. Common in
                         older human-pose work; needs --torso-pair.

Reported per keypoint as well as overall, because the aggregate hides the thing
that matters here. In Mu et al.'s synthetic-only result the head keypoints score
46-54 PCK while the hooves score 17 -- and hooves are exactly what a URDF fit and
foot-contact detection depend on. An overall number that looks survivable can be
carried entirely by the head.

Input format: COCO keypoint JSON for ground truth (as written by
`export_coco_keypoints.py`). Predictions may be either the same format or a plain
`{image_id: [[x, y, score], ...]}` mapping.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_gt(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray],
                                 dict[int, list[float]], list[str]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    names = d["categories"][0]["keypoints"]
    K = len(names)
    kps, vis, bbox = {}, {}, {}
    for a in d["annotations"]:
        flat = np.asarray(a["keypoints"], dtype=np.float64).reshape(K, 3)
        kps[a["image_id"]] = flat[:, :2]
        vis[a["image_id"]] = flat[:, 2]
        bbox[a["image_id"]] = a["bbox"]
    return kps, vis, bbox, names


def load_pred(path: Path, K: int) -> dict[int, np.ndarray]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, np.ndarray] = {}
    if isinstance(d, dict) and "annotations" in d:
        for a in d["annotations"]:
            flat = np.asarray(a["keypoints"], dtype=np.float64).reshape(-1, 3)
            out[a["image_id"]] = flat[:, :2]
    elif isinstance(d, list):
        for a in d:
            flat = np.asarray(a["keypoints"], dtype=np.float64).reshape(-1, 3)
            out[int(a["image_id"])] = flat[:, :2]
    else:
        for k, v in d.items():
            arr = np.asarray(v, dtype=np.float64)
            out[int(k)] = arr[:, :2] if arr.ndim == 2 and arr.shape[1] >= 2 \
                else arr.reshape(-1, 2)
    return out


def normaliser(kind: str, bbox: list[float], gt: np.ndarray,
               names: list[str], torso_pair: tuple[str, str] | None) -> float:
    if kind == "bbox_max":
        return max(bbox[2], bbox[3])
    if kind == "bbox_diag":
        return math.hypot(bbox[2], bbox[3])
    if kind == "torso":
        if torso_pair is None:
            raise SystemExit("--normaliser torso needs --torso-pair A B")
        a, b = torso_pair
        if a not in names or b not in names:
            raise SystemExit(f"--torso-pair {a},{b} not in the schema")
        return float(np.linalg.norm(gt[names.index(a)] - gt[names.index(b)]))
    raise SystemExit(f"unknown normaliser {kind}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--alpha", type=float, nargs="*", default=[0.05, 0.1])
    ap.add_argument("--normaliser", choices=["bbox_max", "bbox_diag", "torso"],
                    default="bbox_max")
    ap.add_argument("--torso-pair", nargs=2, default=None)
    ap.add_argument("--include-occluded", action="store_true",
                    help="score keypoints labelled occluded (vis==1) too. "
                         "Default counts only vis==2, which is the stricter "
                         "and more common convention.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    gt, vis, bbox, names = load_gt(Path(args.gt))
    K = len(names)
    pred = load_pred(Path(args.pred), K)

    shared = sorted(set(gt) & set(pred))
    if not shared:
        raise SystemExit("no image_ids in common between --gt and --pred")
    missing = len(set(gt)) - len(shared)

    min_vis = 1 if args.include_occluded else 2
    dists: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for img in shared:
        g, p = gt[img], pred[img]
        if p.shape[0] != K:
            raise SystemExit(
                f"image {img}: prediction has {p.shape[0]} keypoints, "
                f"ground truth has {K}")
        s = normaliser(args.normaliser, bbox[img], g, names,
                       tuple(args.torso_pair) if args.torso_pair else None)
        if s <= 0:
            continue
        dists.append(np.linalg.norm(p - g, axis=1) / s)
        masks.append(vis[img] >= min_vis)

    D = np.stack(dists)          # (N, K) normalised distance
    M = np.stack(masks)          # (N, K) scored?

    print(f"images scored : {len(D)}"
          + (f"  ({missing} ground-truth images had no prediction)"
             if missing else ""))
    print(f"normaliser    : {args.normaliser}"
          + (f" ({args.torso_pair[0]}-{args.torso_pair[1]})"
             if args.normaliser == "torso" else ""))
    print(f"scoring       : visibility >= {min_vis}"
          f"  ({int(M.sum())} keypoint instances)")
    print()

    results: dict[str, dict] = {}
    header = "  " + f"{'keypoint':<16}" + "".join(
        f"{'PCK@' + str(a):>11}" for a in args.alpha) + f"{'n':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in range(K):
        m = M[:, k]
        n = int(m.sum())
        row = f"  {names[k]:<16}"
        per: dict[str, float] = {}
        for a in args.alpha:
            if n == 0:
                row += f"{'--':>11}"
                continue
            v = float((D[m, k] < a).mean())
            per[f"PCK@{a}"] = v
            row += f"{v * 100:10.1f}%"
        row += f"{n:8d}"
        print(row)
        results[names[k]] = {**per, "n": n}

    print("  " + "-" * (len(header) - 2))
    overall_row = f"  {'ALL':<16}"
    overall: dict[str, float] = {}
    for a in args.alpha:
        v = float((D[M] < a).mean())
        overall[f"PCK@{a}"] = v
        overall_row += f"{v * 100:10.1f}%"
    overall_row += f"{int(M.sum()):8d}"
    print(overall_row)

    # The aggregate hides the failure mode that matters. Call it out.
    feet = [n for n in names if "hoof" in n.lower() or "paw" in n.lower()]
    if feet and args.alpha:
        a = args.alpha[0]
        idx = [names.index(n) for n in feet]
        fm = M[:, idx]
        if fm.sum() > 0:
            fv = float((D[:, idx][fm] < a).mean())
            print(f"\n  feet only (PCK@{a}): {fv * 100:.1f}%  "
                  f"over {int(fm.sum())} instances")
            print("  Feet drive the URDF fit and foot-contact detection. In the")
            print("  published synthetic-only results they score ~17 while head")
            print("  keypoints score ~50, so judge on this number, not the mean.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"per_keypoint": results, "overall": overall,
             "normaliser": args.normaliser, "min_visibility": min_vis,
             "n_images": len(D)}, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
