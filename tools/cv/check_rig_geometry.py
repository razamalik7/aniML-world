"""Geometric sanity checks on the deer rig. Catches the class of bug where a
mesh is silently mirrored, mis-scaled, or attached to the wrong side of a link.

    python tools/cv/check_rig_geometry.py
    python tools/cv/check_rig_geometry.py --urdf assets/deer_urdf/deer_world.urdf

Exits non-zero if any check fails, so it can gate a rig change.

WHY
---
A mirrored collision mesh raises no error anywhere. The URDF parses, the USD
converts, PhysX simulates, training runs, rewards go up. The only symptom is a
policy that behaves strangely for reasons nobody can pin down -- which is a
fair description of a long run of this project's locomotion work.

The checks below are all *relative* comparisons within the rig, so they need no
external ground truth and no particular coordinate convention:

  visual_vs_collision  Within one link, the visual and collision meshes must sit
                       on the same side of the link origin. They describe the
                       same physical part; if their bounding boxes are on
                       opposite sides of the origin along any axis, one of them
                       is mirrored.
  leg_symmetry         Left and right legs of the same pair must have collision
                       geometry at the same distance along the down axis.
  limb_reach           Every leg's lowest collision point should be near the
                       lowest point of the whole rig. A leg whose collision
                       geometry stops far short has no foot as far as physics is
                       concerned.
  visual_collision_overlap
                       Per link, the visual and collision bounding boxes should
                       overlap substantially.

ANATOMY (see CLAUDE.md): +X left, +Y down, -Z forward.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

from deer_rig import DeerRig

DOWN = 1          # index of the anatomical "down" axis (+Y)
LEG_GROUPS = ["FL", "FR", "HL", "HR"]


def link_meshes(urdf_path: Path) -> dict[str, dict[str, str]]:
    """link -> {"visual": file, "collision": file} (basenames)."""
    text = urdf_path.read_text(encoding="utf-8")
    out: dict[str, dict[str, str]] = {}
    for m in re.finditer(r'<link name="([^"]+)">(.*?)</link>', text, re.DOTALL):
        link, body = m.group(1), m.group(2)
        entry: dict[str, str] = {}
        for tag in ("visual", "collision"):
            tm = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", body, re.DOTALL)
            if not tm:
                continue
            fm = re.search(r'filename="([^"]+)"', tm.group(1))
            if fm:
                entry[tag] = fm.group(1)
        if entry:
            out[link] = entry
    return out


def load(urdf_dir: Path, rel: str) -> np.ndarray | None:
    p = urdf_dir / rel
    if not p.is_file():
        return None
    try:
        return np.asarray(trimesh.load_mesh(str(p), force="mesh").vertices,
                          dtype=np.float64)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="metres of slack before a sign disagreement counts")
    args = ap.parse_args()

    rig = DeerRig(args.urdf) if args.urdf else DeerRig()
    urdf_path = rig.urdf_path
    urdf_dir = urdf_path.parent
    meshes = link_meshes(urdf_path)
    rest = rig.fk()

    failures: list[str] = []
    warnings: list[str] = []

    # ---- 1. visual vs collision on the same side of the link origin ------
    print("[1] visual vs collision, same side of link origin")
    for link, files in sorted(meshes.items()):
        if "visual" not in files or "collision" not in files:
            continue
        v = load(urdf_dir, files["visual"])
        c = load(urdf_dir, files["collision"])
        if v is None or c is None:
            continue
        for ax, axname in ((0, "X"), (1, "Y"), (2, "Z")):
            vlo, vhi = v[:, ax].min(), v[:, ax].max()
            clo, chi = c[:, ax].min(), c[:, ax].max()
            # Disjoint ranges on opposite sides of 0 = mirrored.
            if vlo > args.tol and chi < -args.tol:
                failures.append(
                    f"{link}: visual spans {axname} [{vlo:+.3f},{vhi:+.3f}] but "
                    f"collision spans [{clo:+.3f},{chi:+.3f}] -- opposite sides "
                    f"of the link origin. Collision mesh "
                    f"{Path(files['collision']).name} looks MIRRORED.")
            elif vhi < -args.tol and clo > args.tol:
                failures.append(
                    f"{link}: visual spans {axname} [{vlo:+.3f},{vhi:+.3f}] but "
                    f"collision spans [{clo:+.3f},{chi:+.3f}] -- opposite sides "
                    f"of the link origin. Visual mesh "
                    f"{Path(files['visual']).name} looks MIRRORED.")
    print(f"    {len(failures)} problem(s)")

    # ---- 2. per-leg lowest collision point ------------------------------
    print("[2] per-leg reach along the down axis (collision geometry)")
    reach: dict[str, float] = {}
    for grp in LEG_GROUPS:
        lows = []
        for link, files in meshes.items():
            if not link.startswith(grp + "_") or "collision" not in files:
                continue
            m = load(urdf_dir, files["collision"])
            if m is None or link not in rest:
                continue
            R, t = rest[link]
            lows.append(float((m @ R.T + t)[:, DOWN].max()))
        if lows:
            reach[grp] = max(lows)
    for grp, r in reach.items():
        print(f"    {grp}: lowest collision point at {r:+.3f} m")
    if len(reach) == 4:
        spread = max(reach.values()) - min(reach.values())
        if spread > 0.10:
            worst = min(reach, key=reach.get)
            failures.append(
                f"leg reach differs by {spread*100:.0f} cm across legs "
                f"(min {worst} at {reach[worst]:+.3f}). A leg whose collision "
                f"geometry stops short has no foot as far as physics is "
                f"concerned.")
        for a, b in (("FL", "FR"), ("HL", "HR")):
            if abs(reach[a] - reach[b]) > 0.02:
                failures.append(
                    f"{a}/{b} asymmetric: {reach[a]:+.3f} vs {reach[b]:+.3f}")

    # ---- 3. visual/collision bbox overlap -------------------------------
    print("[3] visual/collision bounding-box overlap")
    for link, files in sorted(meshes.items()):
        if "visual" not in files or "collision" not in files:
            continue
        v = load(urdf_dir, files["visual"])
        c = load(urdf_dir, files["collision"])
        if v is None or c is None:
            continue
        lo = np.maximum(v.min(0), c.min(0))
        hi = np.minimum(v.max(0), c.max(0))
        inter = float(np.prod(np.clip(hi - lo, 0, None)))
        vol_v = float(np.prod(v.max(0) - v.min(0)))
        frac = inter / vol_v if vol_v > 1e-12 else 0.0
        if frac < 0.05:
            warnings.append(
                f"{link}: visual and collision bounding boxes overlap only "
                f"{frac*100:.1f}% of the visual volume")

    # ---- 4. links with no visual ----------------------------------------
    no_visual = sorted(l for l, f in meshes.items() if "visual" not in f)
    all_links = {j.parent for j in rig.joints.values()} | \
                {j.child for j in rig.joints.values()}
    no_geom = sorted(all_links - set(meshes))
    print(f"[4] {len(no_visual)} link(s) with collision but no visual: "
          f"{no_visual}")
    print(f"    {len(no_geom)} link(s) with no geometry at all: {no_geom}")

    print()
    if warnings:
        print(f"{len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"  ! {w}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  X {f}")
        return 1
    print("OK: no geometric inconsistencies found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
