"""Rewrite the URDF's joint limits so they actually CONTAIN the motion the rig has
to reproduce.

WHY (audit 2026-07-25): `build_urdf.measure_axes()` derives `lo`/`hi` from a
different definition of "joint angle" than `extract_reference` emits --
zero at animation frame 0 instead of the rest pose, parent-relative absolute
orientation instead of change-from-rest, and a linearised rotation-vector
projection instead of an exact twist. The AXES agree between the two tools
(verified: all 30 bones, dot=1.0000), so only the limits are wrong, and the error
is non-linear (an additive offset correction does NOT fix it).

Consequence: 19 of 38 joints had limits that excluded poses the animation needs.
`rsi.py` clamps to joint limits, so 100% of reference frames were silently
distorted at spawn -- worst case 56.7 deg on the left hind ankle -- and the
policy could never match the reference on those joints, leaving a permanent
AMP style penalty it was powerless to fix.

POLICY: **widen only, never narrow.**
A deer really can fold its legs right up under itself -- that is what `DLie` /
`DSleep` are -- so "unused" range in the walk is NOT evidence of a bad limit. It
is real anatomy that we will need for lying, sleeping and eating later.
Limits encode ANATOMY; discouraging collapse during walking is the reward
function's job (`pelvis_too_low`, `base_height`), not the skeleton's.

Death clips are excluded by default: those are authored ragdoll//drama poses, not
reliable anatomy. Every bound is attributed to the clip that set it, so nothing
is widened on the strength of one suspicious animation without it being visible.

  python tools/rig_pipeline/fix_joint_limits.py --rom_dir <dir of per-clip .npy>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXCLUDE = ["Deer_Death", "Deer_Death_Opp"]


def _numpy2_pickle_shim():
    """extract_reference runs inside Blender (numpy 2.3.4) but this tool runs in
    Isaac Lab's interpreter (numpy 1.26.0). numpy 2 pickles array reconstructors
    under `numpy._core`, which does not exist in numpy 1 -> ModuleNotFoundError on
    load. Alias the moved modules so the pickles are readable across both."""
    if hasattr(np, "_core"):
        return
    import numpy.core as _c
    sys.modules.setdefault("numpy._core", _c)
    for sub in ("multiarray", "umath", "numeric", "_multiarray_umath"):
        mod = getattr(_c, sub, None)
        if mod is not None:
            sys.modules.setdefault(f"numpy._core.{sub}", mod)


def load_clip(path):
    _numpy2_pickle_shim()
    return np.load(str(path), allow_pickle=True).item()


def read_limits(text):
    """joint -> (lower, upper, full <limit .../> tag text)"""
    out = {}
    for m in re.finditer(r'<joint name="([^"]+)"\s+type="([^"]+)"(.*?)</joint>', text, re.S):
        name, typ, body = m.group(1), m.group(2), m.group(3)
        lm = re.search(r'<limit([^>]*)/>', body)
        if typ == "fixed" or not lm:
            continue
        lo = re.search(r'lower="(-?[0-9.eE+]+)"', lm.group(1))
        hi = re.search(r'upper="(-?[0-9.eE+]+)"', lm.group(1))
        if lo and hi:
            out[name] = (float(lo.group(1)), float(hi.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rom_dir", required=True,
                    help="directory of per-animation .npy files from extract_reference")
    ap.add_argument("--urdf", default=str(REPO / "assets/deer_polyart_urdf/deer_polyart.urdf"))
    ap.add_argument("--out", default=None, help="default: patch --urdf in place (writes .bak)")
    ap.add_argument("--margin_deg", type=float, default=10.0)
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE)
    ap.add_argument("--include", nargs="*", default=None,
                    help="Whitelist of clip stems to trust. Use this to admit only "
                         "clips whose FK ROUND-TRIP error is low. Our joint axes were "
                         "measured FROM THE WALK, so the rig is a walk-optimised "
                         "approximation: it reproduces walk-like motion at ~86 mm but "
                         "lying/eating/falling at 217-248 mm. Deriving a limit from a "
                         "clip the rig cannot faithfully represent widens the skeleton "
                         "on the strength of extraction error, not anatomy.")
    ap.add_argument("--mode", choices=["widen", "fit"], default="fit",
                    help="widen: only ever enlarge a limit (keeps whatever slack the "
                         "current URDF has). fit: set the limit to the observed range "
                         "+/- margin, which also REMOVES slack.\n"
                         "'fit' is the default because the existing bounds are not "
                         "evidence -- they came from the buggy measurement. The "
                         "counter-argument for keeping slack was that a deer really can "
                         "fold its legs to lie down, but that range came from Deer_Sleep, "
                         "which extracts at 218 mm round-trip and is not trustworthy. So "
                         "there is no good evidence for the extra travel, and it is "
                         "exactly the room a collapsing deer folds into. Re-widen "
                         "deliberately (with validated clips) when training lying/eating.")
    ap.add_argument("--percentile", type=float, default=100.0,
                    help="Use the [100-p, p] percentile of each joint's samples instead "
                         "of the raw min/max. 100 = min/max (original behaviour).\n"
                         "Reason: IK retargeting solves each frame independently, and a "
                         "handful of frames land on a distant local minimum. After the "
                         "multi-turn unwrap the library still shows a 198.7 deg elbow "
                         "(Deer_Sleep) and a -193.2 deg neck (Deer_WalkBack) -- solver "
                         "outliers, not deer anatomy, and min/max is maximally sensitive "
                         "to exactly those. 99.9 keeps the real extremes (a lying deer "
                         "does fold hard) while dropping the ~0.1%% that are artefacts.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.rom_dir).glob("*.npy"))
    used, skipped = [], []
    for f in files:
        if args.include is not None:
            ok = any(x.lower() == f.stem.lower() for x in args.include)
        else:
            ok = not any(x.lower() in f.stem.lower() for x in args.exclude)
        (used if ok else skipped).append(f)
    if not used:
        raise SystemExit(f"no .npy found in {args.rom_dir}")
    print(f"clips used   ({len(used)}): {[f.stem for f in used]}")
    print(f"clips EXCLUDED ({len(skipped)}): {[f.stem for f in skipped]}")

    # --- gather per-joint range, attributed to the clip that set it ---
    names = None
    rng = {}          # joint -> [min, max, who_min, who_max]
    excl_rng = {}     # same, for the excluded clips (reported only)
    for f in used + skipped:
        d = load_clip(f)
        jn = list(d["joints_list"]); jp = np.asarray(d["joint_positions"], float)
        if names is None:
            names = jn
        elif jn != names:
            raise SystemExit(f"{f.stem}: joints_list differs from {used[0].stem} -- "
                             f"axes/config mismatch, refusing to mix conventions")
        tgt = excl_rng if f in skipped else rng
        for i, n in enumerate(jn):
            if args.percentile >= 100.0:
                a, b = float(jp[:, i].min()), float(jp[:, i].max())
            else:
                q = args.percentile
                a, b = float(np.percentile(jp[:, i], 100.0 - q)), float(np.percentile(jp[:, i], q))
            if n not in tgt:
                tgt[n] = [a, b, f.stem, f.stem]
            else:
                if a < tgt[n][0]: tgt[n][0], tgt[n][2] = a, f.stem
                if b > tgt[n][1]: tgt[n][1], tgt[n][3] = b, f.stem

    text = Path(args.urdf).read_text(encoding="utf-8")
    cur = read_limits(text)
    M = np.radians(args.margin_deg)

    print(f"\nmargin = {args.margin_deg:.0f} deg; mode = {args.mode.upper()}"
          f"{' (never narrows)' if args.mode == 'widen' else ' (fits to observed range; removes slack)'}")
    print(f"{'joint':<28}{'current':>18}{'needed':>18}{'new':>18}  set-by")
    changes = {}
    for n, (lo, hi) in cur.items():
        if n not in rng:
            continue
        a, b, wmin, wmax = rng[n]
        if args.mode == "widen":
            nlo, nhi = min(lo, a - M), max(hi, b + M)
        else:                      # fit
            nlo, nhi = a - M, b + M
        # The window MUST contain the joint's DEFAULT position (0 for this rig, which
        # spawns at joint-zero). Some joints -- e.g. Head_to_R_Ear -- only ever move to
        # one side in the animation, so a pure fit produces an all-positive range that
        # excludes zero, and Isaac Lab rejects the articulation outright:
        #   ValueError: default positions out of the limits: 'Head_to_R_Ear': 0.000
        #               not in [0.032, 2.034]
        # Keep ~1 deg of slack around zero so the rest pose is always legal.
        Z = np.radians(1.0)
        nlo, nhi = min(nlo, -Z), max(nhi, Z)
        if abs(nlo - lo) < 1e-9 and abs(nhi - hi) < 1e-9:
            continue
        changes[n] = (nlo, nhi)
        who = f"{wmin if nlo < lo - 1e-9 else ''}{'/' if nlo < lo-1e-9 and nhi > hi+1e-9 else ''}{wmax if nhi > hi + 1e-9 else ''}"
        print(f"{n:<28}[{np.degrees(lo):>7.1f},{np.degrees(hi):>7.1f}]"
              f"[{np.degrees(a):>7.1f},{np.degrees(b):>7.1f}]"
              f"[{np.degrees(nlo):>7.1f},{np.degrees(nhi):>7.1f}]  {who}")
    print(f"\n{len(changes)}/{len(cur)} joints widened")

    # what the excluded clips WOULD have added -- visible, not silent
    extra = []
    for n, (lo, hi) in cur.items():
        if n not in excl_rng or n not in rng:
            continue
        ea, eb, w1, w2 = excl_rng[n]
        a, b, _, _ = rng[n]
        if ea < a - np.radians(5) or eb > b + np.radians(5):
            extra.append((n, np.degrees(min(ea - a, 0)), np.degrees(max(eb - b, 0)), w1, w2))
    if extra:
        print(f"\nEXCLUDED clips would additionally widen {len(extra)} joints (>5 deg), e.g.:")
        for n, dlo, dhi, w1, w2 in sorted(extra, key=lambda r: -(abs(r[1]) + r[2]))[:8]:
            print(f"  {n:<28} lower {dlo:+7.1f}d  upper {dhi:+7.1f}d   ({w1}/{w2})")

    if args.dry_run:
        print("\n--dry_run: no file written")
        return

    # --- patch the URDF text, per joint block ---
    def patch(m):
        name, typ, body = m.group(1), m.group(2), m.group(3)
        if name not in changes:
            return m.group(0)
        nlo, nhi = changes[name]
        nb = re.sub(r'(<limit[^>]*?)lower="(-?[0-9.eE+]+)"', rf'\1lower="{nlo:.4f}"', body, count=1)
        nb = re.sub(r'(<limit[^>]*?)upper="(-?[0-9.eE+]+)"', rf'\1upper="{nhi:.4f}"', nb, count=1)
        return f'<joint name="{name}" type="{typ}"{nb}</joint>'

    new_text = re.sub(r'<joint name="([^"]+)"\s+type="([^"]+)"(.*?)</joint>', patch, text, flags=re.S)
    out = Path(args.out or args.urdf)
    if out == Path(args.urdf):
        bak = out.with_suffix(".urdf.prelimitfix.bak")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8")
            print(f"\nbackup -> {bak.name}")
    out.write_text(new_text, encoding="utf-8")
    print(f"wrote {out}")

    # --- verify: every used clip frame must now be inside the limits ---
    fixed = read_limits(new_text)
    bad = 0
    for f in used:
        d = load_clip(f)
        jp = np.asarray(d["joint_positions"], float)
        for i, n in enumerate(list(d["joints_list"])):
            if n not in fixed:
                continue
            lo, hi = fixed[n]
            if jp[:, i].min() < lo - 1e-6 or jp[:, i].max() > hi + 1e-6:
                bad += 1
                print(f"  STILL OUT OF RANGE: {f.stem} {n}")
    print(f"\nVERIFY: {'PASS - every used clip frame is inside the new limits' if not bad else f'FAIL ({bad})'}")


if __name__ == "__main__":
    main()
