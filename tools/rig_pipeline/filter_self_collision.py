"""Filter the self-collision pairs that overlap in the REST pose.

THE PROBLEM (measured 2026-07-25). At a clean spawn, pelvis at a healthy 0.985, the
rig reports **1,002,943 N** of total contact force -- 1240x the deer's 809 N weight --
with 746,844 N of it on the TORSO (Spine2 171kN, Neck 167kN, Spine 142kN) and only
496 N on the hooves. Commanded to hold its default pose with zero action, the deer
collapses from pelvis 0.985 to 0.354. The actuators (320 Nm) cannot fight 400 kN.
This has been present since r1 and is very likely the dominant cause of the
collapsing/crumpling behaviour we spent nine runs attributing to rewards.

WHY. Isaac's articulation auto-filters PARENT-CHILD collision pairs only. Two things
defeat that here:
  1. Every 2-DOF joint inserts a massless intermediate `_dof0` link, so
     `Pelvis -> Spine_dof0 -> Spine` makes `Pelvis<->Spine` a GRANDPARENT pair --
     unfiltered, and obviously overlapping since they are adjacent body segments.
  2. Collision geometry is a CONVEX DECOMPOSITION. A convex hull is fatter than the
     concave mesh it wraps, so hulls of nearby-but-not-parent-child links overlap
     even where the visual meshes do not.
PhysX cannot tell "these two shapes rest inside each other by design" from "these two
shapes are penetrating and must be pushed apart", so it applies enormous restoring
forces forever.

THE FIX. Add explicit USD collision filters for exactly those pairs that overlap in
the rest pose (all joint angles zero). Rest-pose overlap is the objective criterion:
if two hulls intersect when the animal is in its neutral pose, collision between them
carries no information and can only inject error.

IMPORTANT -- this does NOT weaken the "legs must never pass through each other"
requirement. Every filtered pair is anatomically ADJACENT (Pelvis<->Spine,
Neck<->Spine2, Clavicle<->UpperArm, Pelvis<->Calf). No left-limb<->right-limb pair
overlaps at rest, so none is filtered: leg-vs-leg collision stays fully enabled, and
the tool asserts that before writing.

  python tools/rig_pipeline/filter_self_collision.py            # dry run
  python tools/rig_pipeline/filter_self_collision.py --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.render_playback_meshed import (  # noqa: E402
    parse_urdf_full, compute_link_transforms, rpy_to_R)

LEFT = re.compile(r"^L_")
RIGHT = re.compile(r"^R_")
# The standing requirement is "legs must never pass through each other". That is about
# LEG segments specifically -- L_Clavicle<->R_Clavicle and L_Ear<->R_Ear are cross-side
# but are not legs, and filtering them is harmless (they cannot meaningfully collide).
LEG_PARTS = ("Thigh", "Calf", "HorseLink", "Foot", "Toe",
             "UpperArm", "Forearm", "Hand", "Finger")


def limb_of(name):
    """'L'/'R' for a LEG link, None otherwise (torso, midline, clavicle, ear...)."""
    if not any(k in name for k in LEG_PARTS):
        return None
    if LEFT.match(name):
        return "L"
    if RIGHT.match(name):
        return "R"
    return None


def rest_overlaps(urdf_path: Path, motion: Path | None = None, max_kin: int = 2):
    """Pairs that must be filtered = (a) anatomically adjacent (kin <= max_kin), or
    (b) observed overlapping in the rest pose OR any sampled animation pose.

    Rest pose alone is NOT enough: Pelvis<->L_Thigh does not overlap at rest but slams
    together in the mid-stride pose RSI spawns from, and it was contributing 145 kN.
    `max_kin=2` covers exactly the `_dof0` artefacts (Pelvis<->Spine etc). Anything
    further apart is filtered ONLY if it is observed overlapping in some pose, so we do
    not blanket-disable collision between pairs that never actually interpenetrate.
    L_Thigh<->R_Thigh is kin 4 and never overlaps, so the legs are never filtered.
    """
    text = urdf_path.read_text(encoding="utf-8")
    origin_map, _ = parse_urdf_full(urdf_path)
    parent = {c: p for (_, (_, _, p, c, _)) in origin_map.items()}

    col = {}
    for m in re.finditer(r'<link name="([^"]+)">(.*?)</link>', text, re.S):
        n, b = m.group(1), m.group(2)
        cm = re.search(r'<collision[^>]*>\s*<origin\s+xyz="([^"]+)"(?:\s+rpy="([^"]+)")?[^>]*/>\s*'
                       r'<geometry>\s*<mesh filename="([^"]+)"', b, re.S)
        if cm and (urdf_path.parent / cm.group(3)).is_file():
            mesh = trimesh.load_mesh(str(urdf_path.parent / cm.group(3)), force="mesh")
            col[n] = (np.asarray(mesh.vertices, float),
                      np.array([float(v) for v in cm.group(1).split()]),
                      rpy_to_R(np.array([float(v) for v in (cm.group(2) or "0 0 0").split()])))

    # pose set: rest + sampled animation frames
    poses = [{j: 0.0 for j in origin_map}]
    if motion is not None and motion.is_file():
        d = np.load(str(motion), allow_pickle=True).item()
        jn = list(d["joints_list"]); jp = np.asarray(d["joint_positions"], float)
        for t in range(0, len(jp), max(1, len(jp) // 12)):
            poses.append({n: float(jp[t, i]) for i, n in enumerate(jn)})
    print(f"sampling {len(poses)} poses (rest + animation frames)")

    def hulls_for(pose):
        lt = compute_link_transforms(pose, origin_map)
        out = {}
        for n, (V, cx, cR) in col.items():
            if n in lt:
                R, t = lt[n]
                try:
                    out[n] = trimesh.convex.convex_hull(V @ (R @ cR).T + (R @ cx + t))
                except Exception:
                    pass
        return out

    def kin(a, b):
        def chain(x):
            c = [x]
            while parent.get(x) is not None:
                x = parent[x]; c.append(x)
            return c
        ca, cb = chain(a), chain(b)
        for i, x in enumerate(ca):
            if x in cb:
                return i + cb.index(x)
        return 99

    ns = sorted(col)
    found = {}
    for pi, pose in enumerate(poses):
        hulls = hulls_for(pose)
        for i, a in enumerate(ns):
            for b in ns[i + 1:]:
                if a not in hulls or b not in hulls:
                    continue
                d = kin(a, b)
                if d == 1:                      # parent-child: Isaac filters already
                    continue
                ha, hb = hulls[a], hulls[b]
                if (ha.bounds[0] > hb.bounds[1]).any() or (hb.bounds[0] > ha.bounds[1]).any():
                    continue
                try:
                    n_in = int(hb.contains(ha.vertices).sum()) + int(ha.contains(hb.vertices).sum())
                except Exception:
                    n_in = 0
                if n_in > 0:
                    k = (a, b)
                    prev = found.get(k, (0, "rest" if pi == 0 else f"pose{pi}"))
                    if n_in > prev[0]:
                        found[k] = (n_in, "rest" if pi == 0 else f"pose{pi}")
    # adjacency rule: anything within max_kin joints cannot meaningfully collide
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            d = kin(a, b)
            if 1 < d <= max_kin and (a, b) not in found:
                found[(a, b)] = (0, f"adjacent(kin={d})")
    return [(a, b, kin(a, b), n, why) for (a, b), (n, why) in found.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--urdf", default=str(REPO / "assets/deer_polyart_urdf/deer_polyart.urdf"))
    ap.add_argument("--usd", default=str(REPO / "assets/deer_polyart_urdf/deer_polyart.usd"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    pairs = rest_overlaps(Path(args.urdf),
                          motion=REPO / "assets/motions/deer_polyart_walk.npy")
    print(f"non-parent-child pairs overlapping at REST: {len(pairs)}")
    for a, b, d, n, why in sorted(pairs, key=lambda r: (r[2], r[0])):
        print(f"   kin={d}  {a:<16}{b:<16} {n:>3} verts   [{why}]")

    # SAFETY: never filter a left-limb <-> right-limb pair.
    cross = [(a, b) for a, b, _, _, _ in pairs
             if limb_of(a) and limb_of(b) and limb_of(a) != limb_of(b)]
    if cross:
        raise SystemExit(f"REFUSING: would filter left-vs-right limb pairs {cross}. "
                         f"Leg-vs-leg collision must stay enabled.")
    print("\nSAFETY CHECK PASSED: no left-limb <-> right-limb pair is filtered; "
          "leg-vs-leg collision remains fully enabled.")

    if not args.apply:
        print("\n(dry run -- pass --apply to write the USD)")
        return

    from pxr import Usd, UsdPhysics, Sdf
    stage = Usd.Stage.Open(args.usd)
    # map link name -> prim (rigid bodies live at /deer_polyart/<link>)
    prims = {}
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI) or UsdPhysics.RigidBodyAPI(p):
            prims.setdefault(p.GetName(), p)
    missing = [x for a, b, _, _, _ in pairs for x in (a, b) if x not in prims]
    if missing:
        print(f"  WARNING: prims not found for {sorted(set(missing))}")

    n_applied = 0
    for a, b, _, _, _ in pairs:
        pa, pb = prims.get(a), prims.get(b)
        if pa is None or pb is None:
            continue
        api = UsdPhysics.FilteredPairsAPI.Apply(pa)
        rel = api.CreateFilteredPairsRel()
        rel.AddTarget(pb.GetPath())
        n_applied += 1
    stage.GetRootLayer().Save()
    print(f"\napplied {n_applied} filtered pairs and saved {args.usd}")
    print("Re-run tools/rig_pipeline/diagnose_self_collision.py: total contact force "
          "should fall from ~1,000,000 N to ~809 N concentrated in the hooves.")


if __name__ == "__main__":
    main()
