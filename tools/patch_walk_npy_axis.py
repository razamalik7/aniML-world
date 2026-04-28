"""Patch motion .npy: swap root_position axis from world +X to -Y.

The original converter wrote `root_position` advancing along world +X with a
constant Rx(-90°) `root_quaternion`. Combined with AMPLoader's
`base_lin_velocities_local = R.T @ v_world`, that put the expert's forward
velocity at body +X — but the running policy faces world -Y after the spawn
rotation and emits forward velocity at body -Z. The discriminator could drive
its expert-vs-policy decision off this constant 3-vector mismatch alone, which
is exactly the chronic style_r ≈ 0.002 saturation observed across runs 1-14.

This script loads each existing motion .npy, moves the X-axis component of
`root_position` onto the -Y axis, leaves everything else untouched, and saves
back. After this fix, AMPLoader will compute body-frame velocity as
(0, 0, -FORWARD_SPEED), matching the policy's runtime observation.

The Unity .anim source files aren't on this machine so we can't re-run the
full converter (tools/convert_unity_anim_to_npy.py); the source is updated to
the correct convention going forward, but existing .npy files need this
in-place patch.

Run:
    /c/isaac-lab/env_isaaclab/Scripts/python.exe tools/patch_walk_npy_axis.py
    /c/isaac-lab/env_isaaclab/Scripts/python.exe tools/patch_walk_npy_axis.py --target run
    /c/isaac-lab/env_isaaclab/Scripts/python.exe tools/patch_walk_npy_axis.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MOTIONS_DIR = REPO_ROOT / "assets" / "motions"

KNOWN_TARGETS = {
    "walk": "deer_walk.npy",
    "run":  "deer_run.npy",
    "idle": "deer_idle.npy",
}


def patch_one(npy_path: Path) -> bool:
    """Patch a single .npy in place. Returns True if patched, False if skipped."""
    if not npy_path.is_file():
        print(f"[skip] {npy_path.name}: not found")
        return False

    print(f"\nLoading {npy_path}")
    payload = np.load(npy_path, allow_pickle=True).item()
    rp = np.asarray(payload["root_position"], dtype=np.float32).copy()  # (T, 3)
    rq = np.asarray(payload["root_quaternion"], dtype=np.float32)        # (T, 4)

    print(f"  Before: root_position[ 0]={rp[0].tolist()}")
    print(f"          root_position[-1]={rp[-1].tolist()}")
    print(f"          quat[0] (xyzw)   ={rq[0].tolist()}")

    # Heuristic: pre-fix version moves along +X with ~zero Y. Skip if already
    # along -Y (already patched) or stationary.
    moved_x = rp[-1, 0] - rp[0, 0]
    moved_y = rp[-1, 1] - rp[0, 1]
    if abs(moved_x) < 1e-3 and abs(moved_y) > 1e-3:
        print(f"  [skip] {npy_path.name}: looks already patched "
              f"(moved -Y={moved_y:.3f})")
        return False
    if abs(moved_x) < 1e-3 and abs(moved_y) < 1e-3:
        print(f"  [skip] {npy_path.name}: stationary clip "
              f"(no axis swap needed)")
        return False
    if moved_x <= 0.05:
        print(f"  [skip] {npy_path.name}: doesn't look like pre-fix "
              f"(moved_x={moved_x:.3f}). Refusing to patch.")
        return False

    # Swap +X movement onto -Y. Z (height) is left alone.
    new_rp = np.zeros_like(rp)
    new_rp[:, 0] = 0.0
    new_rp[:, 1] = -rp[:, 0]
    new_rp[:, 2] = rp[:, 2]

    print(f"  After:  root_position[ 0]={new_rp[0].tolist()}")
    print(f"          root_position[-1]={new_rp[-1].tolist()}")

    backup_path = npy_path.with_suffix(".npy.preaxisfix.bak")
    if not backup_path.exists() and not str(backup_path).endswith(".bak.npy"):
        # np.save adds .npy automatically; explicitly check for the .bak.npy form
        backup_check = npy_path.with_name(npy_path.stem + ".npy.preaxisfix.bak.npy")
        if not backup_check.exists():
            np.save(backup_path, payload, allow_pickle=True)
            print(f"  Wrote backup -> {backup_path}.npy")
        else:
            print(f"  Backup already exists at {backup_check}")
    else:
        print(f"  Backup already exists")

    payload["root_position"] = new_rp.astype(np.float32)
    np.save(npy_path, payload, allow_pickle=True)
    print(f"  Saved patched -> {npy_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--target", choices=list(KNOWN_TARGETS), default="walk",
        help="Which motion clip to patch.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Patch all known clips in assets/motions.",
    )
    args = parser.parse_args()

    if args.all:
        targets = [(name, MOTIONS_DIR / fn) for name, fn in KNOWN_TARGETS.items()]
    else:
        targets = [(args.target, MOTIONS_DIR / KNOWN_TARGETS[args.target])]

    patched = 0
    for name, path in targets:
        print(f"\n=== Target: {name} ({path.name}) ===")
        if patch_one(path):
            patched += 1
    print(f"\nPatched {patched}/{len(targets)} clips.")


if __name__ == "__main__":
    main()
