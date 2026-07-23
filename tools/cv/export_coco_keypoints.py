"""Export a synthetic dataset (or hand labels) to COCO keypoint JSON.

    python tools/cv/export_coco_keypoints.py --dataset <synth_dir> --out labels.json
    python tools/cv/export_coco_keypoints.py --dataset <synth_dir> --out labels.json --schema ap10k

COCO keypoint JSON is what essentially every 2D pose framework reads --
mmpose, DeepLabCut's newer loaders, Detectron2, ViTPose. Emitting it means the
synthetic renders and the hand-labelled real frames can go into the same
training run without any bespoke data code.

TWO SCHEMAS
-----------
`--schema canonical` (default) exports all 35 of our keypoints. Use this for
training a net whose output space is our rig -- the point of the whole exercise.

`--schema ap10k` exports the 17 AP-10K slots. Use this to compare against an
off-the-shelf AP-10K detector, or to fine-tune one. The two eye keypoints have no
bone-anchored equivalent on our rig and are emitted with visibility 0, which COCO
readers treat as "not labelled" and loss functions mask out.

VISIBILITY maps our 3-level flag onto COCO's:
    ours 2 (visible)            -> COCO 2 (labelled, visible)
    ours 1 (in frame, occluded) -> COCO 1 (labelled, not visible)
    ours 0 (outside frame)      -> COCO 0 (not labelled)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deer_keypoints import (
    AP10K_FROM_CANONICAL,
    AP10K_KEYPOINT_ORDER,
    KEYPOINT_NAMES,
    SKELETON_EDGES,
)


def build_categories(schema: str) -> tuple[list[dict], list[str]]:
    if schema == "ap10k":
        names = list(AP10K_KEYPOINT_ORDER)
        idx = {n: i for i, n in enumerate(names)}
        # AP-10K's own skeleton, expressed over its slot names.
        edges = [
            ("L_Eye", "R_Eye"), ("L_Eye", "Nose"), ("R_Eye", "Nose"),
            ("Nose", "Neck"), ("Neck", "Root_of_tail"),
            ("Neck", "L_Shoulder"), ("L_Shoulder", "L_Elbow"),
            ("L_Elbow", "L_F_Paw"),
            ("Neck", "R_Shoulder"), ("R_Shoulder", "R_Elbow"),
            ("R_Elbow", "R_F_Paw"),
            ("Root_of_tail", "L_Hip"), ("L_Hip", "L_Knee"),
            ("L_Knee", "L_B_Paw"),
            ("Root_of_tail", "R_Hip"), ("R_Hip", "R_Knee"),
            ("R_Knee", "R_B_Paw"),
        ]
    else:
        names = list(KEYPOINT_NAMES)
        idx = {n: i for i, n in enumerate(names)}
        edges = list(SKELETON_EDGES)
    skeleton = [[idx[a] + 1, idx[b] + 1] for a, b in edges
                if a in idx and b in idx]
    return [{
        "id": 1,
        "name": "deer",
        "supercategory": "animal",
        "keypoints": names,
        "skeleton": skeleton,
    }], names


def remap_to_ap10k(kp: list[list[float]], vis: list[int],
                   canon_index: dict[str, int]) -> tuple[list[list[float]],
                                                         list[int]]:
    out_kp: list[list[float]] = []
    out_vis: list[int] = []
    for slot in AP10K_KEYPOINT_ORDER:
        canon = AP10K_FROM_CANONICAL[slot]
        if canon is None or canon not in canon_index:
            out_kp.append([0.0, 0.0])
            out_vis.append(0)
            continue
        i = canon_index[canon]
        out_kp.append(kp[i])
        out_vis.append(vis[i])
    return out_kp, out_vis


def bbox_from_keypoints(kp: list[list[float]], vis: list[int],
                        w: int, h: int, pad: float = 0.06) -> list[float]:
    pts = [p for p, v in zip(kp, vis) if v > 0]
    if not pts:
        return [0.0, 0.0, float(w), float(h)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0 = max(0.0, x0 - px); y0 = max(0.0, y0 - py)
    x1 = min(float(w), x1 + px); y1 = min(float(h), y1 + py)
    return [x0, y0, x1 - x0, y1 - y0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True,
                    help="directory written by blender_synth.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", choices=["canonical", "ap10k"],
                    default="canonical")
    ap.add_argument("--min-visible", type=int, default=4,
                    help="skip frames with fewer than this many visible "
                         "keypoints; a near-empty annotation is noise")
    args = ap.parse_args()

    root = Path(args.dataset)
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    canon_index = {n: i for i, n in enumerate(meta["keypoint_names"])}
    categories, names = build_categories(args.schema)

    images: list[dict] = []
    annotations: list[dict] = []
    skipped = 0
    for i, s in enumerate(meta["samples"]):
        label = json.loads((root / s["label"]).read_text(encoding="utf-8"))
        kp = label["keypoints_2d"]
        vis = label["visibility"]
        if args.schema == "ap10k":
            kp, vis = remap_to_ap10k(kp, vis, canon_index)
        n_vis = sum(1 for v in vis if v > 0)
        if n_vis < args.min_visible:
            skipped += 1
            continue
        img_id = i + 1
        w, h = label["width"], label["height"]
        images.append({
            "id": img_id,
            "file_name": label["image"],
            "width": w,
            "height": h,
        })
        flat: list[float] = []
        for (x, y), v in zip(kp, vis):
            flat += [round(float(x), 2), round(float(y), 2), int(v)]
        bbox = bbox_from_keypoints(kp, vis, w, h)
        annotations.append({
            "id": img_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": flat,
            "num_keypoints": n_vis,
            "bbox": [round(v, 2) for v in bbox],
            "area": round(bbox[2] * bbox[3], 2),
            "iscrowd": 0,
        })

    out = {
        "info": {
            "description": f"aniML deer, schema={args.schema}",
            "source_dataset": str(root),
            "source_blend": meta.get("source_blend"),
            "renderer": meta.get("renderer"),
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  schema      : {args.schema} ({len(names)} keypoints)")
    print(f"  images      : {len(images)}")
    print(f"  annotations : {len(annotations)}")
    if skipped:
        print(f"  skipped     : {skipped} frames with < {args.min_visible} "
              f"visible keypoints")


if __name__ == "__main__":
    main()
