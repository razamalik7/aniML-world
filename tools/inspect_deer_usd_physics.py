#!/usr/bin/env python3
"""Inspect a USD for Isaac Lab–relevant physics (run with Isaac Sim / isaaclab Python).

Usage (from Isaac Lab root, with repo on PYTHONPATH optional):

  .\\isaaclab.bat -p C:\\path\\to\\aniML_world\\tools\\inspect_deer_usd_physics.py -- C:\\path\\to\\deer.usdc
"""
from __future__ import annotations

import argparse
import sys

from pxr import PhysxSchema, Usd, UsdPhysics


def main() -> int:
    p = argparse.ArgumentParser(description="List articulation roots and rigid bodies in a USD file.")
    p.add_argument("usd_path", help="Path to .usd / .usda / .usdc")
    p.add_argument("--max-list", type=int, default=40, help="Max rigid body paths to print")
    args = p.parse_args()

    stage = Usd.Stage.Open(args.usd_path)
    if not stage:
        print(f"ERROR: could not open stage: {args.usd_path}", file=sys.stderr)
        return 1

    default_prim = stage.GetDefaultPrim()
    print(f"Stage: {args.usd_path}")
    print(f"Default prim: {default_prim.GetPath() if default_prim.IsValid() else '(none)'}")

    roots: list[str] = []
    rigid_paths: list[str] = []
    joints: list[str] = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_paths.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            joints.append(str(prim.GetPath()))

    print(f"\nArticulationRootAPI count: {len(roots)}")
    for r in roots:
        print(f"  {r}")

    print(f"\nRigidBodyAPI count: {len(rigid_paths)}")
    for r in rigid_paths[: args.max_list]:
        has_contact = prim_has_contact_report(stage, r)
        print(f"  {r}  (PhysxContactReportAPI: {has_contact})")
    if len(rigid_paths) > args.max_list:
        print(f"  ... and {len(rigid_paths) - args.max_list} more")

    print(f"\nRevolute/Prismatic joint count: {len(joints)}")
    for j in joints[: args.max_list]:
        print(f"  {j}")
    if len(joints) > args.max_list:
        print(f"  ... and {len(joints) - args.max_list} more")

    ok = len(roots) >= 1 and len(rigid_paths) >= 1
    print("\nSummary: ", end="")
    if ok:
        print("Looks like physics data is present (still validate collision/drives in Isaac Sim Play).")
    else:
        print("NOT sim-ready for Isaac Lab articulation: need ArticulationRootAPI and RigidBodyAPI prims.")
    return 0 if ok else 2


def prim_has_contact_report(stage: Usd.Stage, path_str: str) -> bool:
    prim = stage.GetPrimAtPath(path_str)
    if not prim.IsValid():
        return False
    return prim.HasAPI(PhysxSchema.PhysxContactReportAPI)


if __name__ == "__main__":
    raise SystemExit(main())
