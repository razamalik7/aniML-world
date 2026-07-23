"""Draw auto-generated keypoint labels onto their renders, so a human can
confirm the labels are actually on the animal.

Blender's bundled Python has no Pillow, so `blender_synth.py --debug-overlay`
usually cannot draw. Run this afterwards with the system Python:

    python tools/cv/overlay_keypoints.py --dataset <out_dir> [--num 8]
    python tools/cv/overlay_keypoints.py --dataset <out_dir> --contact-sheet

THIS IS NOT OPTIONAL. A frame-convention error produces label files that are
internally consistent, numerically plausible, and completely wrong. The only
detector is looking at a picture. This project has lost multiple days to
metrics that agreed with each other while the geometry was broken.

Read the output as:
    green dot  = labelled visible
    orange dot = labelled self-occluded
    blue lines = skeleton edges
If dots sit off the body, or left/right limbs are swapped, stop and fix the
frame convention before generating a real dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def draw_one(img_path: Path, label: dict, edges: list[tuple[str, str]],
             label_points: bool) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    names = label["keypoint_names"]
    kp = label["keypoints_2d"]
    vis = label["visibility"]
    idx = {n: i for i, n in enumerate(names)}

    for a, b in edges:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None or vis[ia] == 0 or vis[ib] == 0:
            continue
        draw.line([tuple(kp[ia]), tuple(kp[ib])], fill=(40, 90, 220), width=2)

    for i, (x, y) in enumerate(kp):
        if vis[i] == 0:
            continue
        colour = (30, 220, 90) if vis[i] == 2 else (240, 170, 40)
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour,
                     outline=(0, 0, 0))
        if label_points:
            draw.text((x + 5, y - 5), names[i], fill=(255, 255, 255))
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, type=str,
                   help="Directory written by blender_synth.py")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--names", action="store_true",
                   help="Print keypoint names next to each dot (cluttered "
                        "but decisive when checking left/right swaps).")
    p.add_argument("--contact-sheet", action="store_true",
                   help="Write one grid image instead of N separate files.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    root = Path(args.dataset)
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    edges = [tuple(e) for e in meta["skeleton_edges"]]
    samples = meta["samples"][: args.num]
    if not samples:
        raise SystemExit("dataset has no samples")

    out_dir = Path(args.out) if args.out else root / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for s in samples:
        label = json.loads((root / s["label"]).read_text(encoding="utf-8"))
        img = draw_one(root / s["image"], label, edges, args.names)
        images.append((s["stem"], img))
        if not args.contact_sheet:
            path = out_dir / f"{s['stem']}.png"
            img.save(path)
            print(f"wrote {path}")

    if args.contact_sheet:
        cols = min(4, len(images))
        rows = (len(images) + cols - 1) // cols
        w, h = images[0][1].size
        sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 24))
        for i, (_stem, img) in enumerate(images):
            sheet.paste(img, ((i % cols) * w, (i // cols) * h))
        path = out_dir / "contact_sheet.png"
        sheet.save(path)
        print(f"wrote {path}  ({len(images)} samples, {cols}x{rows})")

    n_vis = [sum(1 for v in json.loads(
        (root / s["label"]).read_text(encoding="utf-8"))["visibility"] if v == 2)
        for s in samples]
    total = len(meta["keypoint_names"])
    print(f"visible keypoints per sample: min={min(n_vis)} "
          f"max={max(n_vis)} mean={sum(n_vis)/len(n_vis):.1f} of {total}")


if __name__ == "__main__":
    main()
