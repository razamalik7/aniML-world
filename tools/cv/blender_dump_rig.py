"""Dump the artist armature's rest-pose geometry to JSON, so the rest of the
CV tooling can reason about it without needing Blender installed.

    blender -b assets/blender/deer_standing.blend \
        --python tools/cv/blender_dump_rig.py -- --out tools/cv/data/artist_rig_rest.json

Writes, in `rig.matrix_world` space (which uses the same anatomical convention
as the deer URDF: +X left, +Y down, -Z forward):

    bones[name] = {head, tail, parent, length}
    keypoints[name] = [x, y, z]        # the deer_keypoints.py schema
    actions[name] = {frame_start, frame_end}

Committed output lives in `tools/cv/data/` so `rig_correspondence.py` and any
future URDF-fitting work can run in plain Python.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deer_keypoints import CANONICAL_KEYPOINTS  # noqa: E402

RIG_OBJ = "Deer_001_rig"
MESH_OBJ = "Deer_001"


def parse_args(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args(list(sys.argv))
    rig = bpy.data.objects[RIG_OBJ]
    mw = rig.matrix_world

    bones = {}
    for b in rig.data.bones:
        h, t = mw @ b.head_local, mw @ b.tail_local
        bones[b.name] = {
            "head": [round(v, 6) for v in h],
            "tail": [round(v, 6) for v in t],
            "parent": b.parent.name if b.parent else None,
            "length": round(b.length, 6),
        }

    keypoints = {}
    for name, (bone, end) in CANONICAL_KEYPOINTS:
        if bone not in bones:
            continue
        keypoints[name] = bones[bone][end]

    actions = {}
    for a in bpy.data.actions:
        lo, hi = a.frame_range
        actions[a.name] = {"frame_start": int(math.floor(lo)),
                           "frame_end": int(math.ceil(hi))}

    mesh = bpy.data.objects.get(MESH_OBJ)
    mesh_info = None
    if mesh is not None:
        mesh_info = {
            "name": MESH_OBJ,
            "vertices": len(mesh.data.vertices),
            "polygons": len(mesh.data.polygons),
            "uv_layers": [l.name for l in mesh.data.uv_layers],
        }

    payload = {
        "source_blend": bpy.data.filepath,
        "rig": RIG_OBJ,
        "frame": "rig.matrix_world (+X left, +Y down, -Z forward)",
        "n_bones": len(bones),
        "bones": bones,
        "keypoints": keypoints,
        "actions": actions,
        "mesh": mesh_info,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[dump] wrote {out} ({len(bones)} bones, {len(keypoints)} keypoints, "
          f"{len(actions)} actions)")


if __name__ == "__main__":
    main()
