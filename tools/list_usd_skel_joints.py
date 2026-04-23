#!/usr/bin/env python3
"""Print UsdSkel skeleton joint names and parent indices (Isaac Lab / isaaclab Python).

  .\\isaaclab.bat -p C:\\path\\to\\aniML_world\\tools\\list_usd_skel_joints.py -- C:\\path\\to\\stage.usd
"""
from __future__ import annotations

import argparse
import sys

from pxr import Usd, UsdSkel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("usd_path")
    args = p.parse_args()

    stage = Usd.Stage.Open(args.usd_path)
    if not stage:
        print(f"ERROR: could not open {args.usd_path}", file=sys.stderr)
        return 1

    found = False
    for prim in stage.Traverse():
        if not prim.IsA(UsdSkel.Skeleton):
            continue
        found = True
        skel = UsdSkel.Skeleton(prim)
        path = prim.GetPath().pathString
        print(f"\nSkeleton: {path}")

        names = []
        ja = skel.GetJointsAttr()
        if ja and ja.HasAuthoredValue():
            names = list(ja.Get())
        if not names:
            print("  (no joints attribute or empty)")
            continue

        print(f"  Joint count: {len(names)}")
        try:
            topo = UsdSkel.Topology(names)
            parents = topo.GetParentIndices()
            for i, name in enumerate(names):
                pi = int(parents[i]) if i < len(parents) else -1
                parent_name = names[pi] if pi >= 0 else "(root)"
                print(f"  [{i:3d}] {name}  (parent: {parent_name})")
        except Exception as ex:  # noqa: BLE001
            print(f"  (could not build topology: {ex})")
            for i, name in enumerate(names):
                print(f"  [{i:3d}] {name}")

    if not found:
        print("No UsdSkel.Skeleton prims found in stage.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
