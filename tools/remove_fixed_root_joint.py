"""Remove the fixed root joint from the deer USD so the base floats freely.

Modifies deer_standing_robot.usd in-place (backs up to .bak first).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd  # noqa: E402

SRC = ROOT / "deer_standing_robot.usd"
BAK = ROOT / "deer_standing_robot.usd.bak"


def main():
    if not BAK.exists():
        shutil.copy2(SRC, BAK)
        print(f"Backed up: {SRC} -> {BAK}")

    stage = Usd.Stage.Open(str(SRC))
    if not stage:
        print(f"ERROR: Could not open {SRC}")
        app.close()
        return

    rj = stage.GetPrimAtPath("/deer/root_joint")
    if rj and rj.IsValid():
        print(f"Found: /deer/root_joint  type={rj.GetTypeName()}")
        stage.RemovePrim("/deer/root_joint")
        print("Removed /deer/root_joint")
    else:
        print("No /deer/root_joint found")

    # Also change the root orient from -91° to 180° so the deer spawns upright
    deer = stage.GetPrimAtPath("/deer")
    if deer:
        from pxr import Gf, UsdGeom  # noqa: E402
        xf = UsdGeom.Xformable(deer)
        # 180° about X:  quat(w=0, x=1, y=0, z=0)
        orient_op = None
        for op in xf.GetOrderedXformOps():
            if "orient" in op.GetName():
                orient_op = op
                break
        if orient_op:
            old_val = orient_op.Get()
            print(f"Old orient: {old_val}")
            orient_op.Set(Gf.Quatd(0.0, 1.0, 0.0, 0.0))
            print(f"New orient: 180° X = (w=0, x=1, y=0, z=0)")

    stage.GetRootLayer().Save()
    print(f"Saved: {SRC}")

    # Verify
    stage2 = Usd.Stage.Open(str(SRC))
    rj2 = stage2.GetPrimAtPath("/deer/root_joint")
    if rj2 and rj2.IsValid():
        print(f"\nWARNING: root_joint still present (type={rj2.GetTypeName()})")
    else:
        print("\nVERIFIED: /deer/root_joint is gone")

    deer2 = stage2.GetPrimAtPath("/deer")
    if deer2:
        xf2 = UsdGeom.Xformable(deer2)
        for op in xf2.GetOrderedXformOps():
            if "orient" in op.GetName():
                print(f"Verified orient: {op.Get()}")

    app.close()


if __name__ == "__main__":
    main()
