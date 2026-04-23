"""Verify deer_fixed_robot.usd: no fixed joint, check root orient, FK standing.

Run: _isaac_sim\python.bat tools\verify_deer_fixed_usd.py
"""
import math
import sys
import traceback

print("SCRIPT START", flush=True)

try:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
except Exception as e:
    print(f"SimulationApp failed: {e}", flush=True)
    sys.exit(1)

print("SimulationApp ready", flush=True)

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    print("pxr imported OK", flush=True)
except Exception as e:
    print(f"pxr import failed: {e}", flush=True)
    app.close()
    sys.exit(1)

USD_PATH = r"C:\Users\nars7\Downloads\aniML_world\deer_fixed_robot.usd"

try:
    stage = Usd.Stage.Open(USD_PATH)
    print(f"\nOpened: {USD_PATH}", flush=True)

    # 1. Check for fixed joints
    print("\n=== FIXED JOINTS ===", flush=True)
    fixed_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.FixedJoint):
            print(f"  FOUND fixed joint: {prim.GetPath()}", flush=True)
            fixed_count += 1
        tp = prim.GetTypeName()
        if "FixedJoint" in tp:
            print(f"  FOUND typed FixedJoint: {prim.GetPath()}", flush=True)
            fixed_count += 1
    if fixed_count == 0:
        print("  NONE found (good!)", flush=True)

    # 2. Root prim orientation
    print("\n=== ROOT ORIENT ===", flush=True)
    deer = stage.GetPrimAtPath("/deer")
    if deer.IsValid():
        xf = UsdGeom.Xformable(deer)
        for op in xf.GetOrderedXformOps():
            print(f"  {op.GetName()} = {op.Get()}", flush=True)
    else:
        print("  /deer prim NOT FOUND!", flush=True)

    # 3. Articulation root
    print("\n=== ARTICULATION ROOT ===", flush=True)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  ArticulationRoot: {prim.GetPath()}", flush=True)

    # 4. Rigid bodies
    print("\n=== RIGID BODIES ===", flush=True)
    rb_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_count += 1
            print(f"  {prim.GetPath()}", flush=True)
    print(f"  Total: {rb_count}", flush=True)

    # 5. Check joint prims (revolute)
    print("\n=== REVOLUTE JOINTS ===", flush=True)
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            body0 = prim.GetRelationship("physics:body0").GetTargets()
            body1 = prim.GetRelationship("physics:body1").GetTargets()
            print(f"  {prim.GetPath().name}: {body0} -> {body1}", flush=True)

    # 6. Check visual mesh transforms (verify rpy was applied)
    print("\n=== VISUAL MESH TRANSFORMS (sample) ===", flush=True)
    checked = 0
    for prim in stage.Traverse():
        ps = str(prim.GetPath())
        if "/visuals/" in ps and prim.GetTypeName() in ("Xform", "Mesh"):
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            if ops:
                for op in ops:
                    print(f"  {ps}: {op.GetName()} = {op.Get()}", flush=True)
                checked += 1
                if checked >= 6:
                    break
    if checked == 0:
        print("  No visual mesh xform ops found", flush=True)

    # 7. FK analysis at different root rotations
    print("\n=== FK STANDING CHECK ===", flush=True)
    import numpy as np

    joints_chain = {
        "pelvis": {"parent": None, "offset": [0, 0, 0]},
        "chest": {"parent": "pelvis", "offset": [0, -0.965180, 0.511591]},
        "neck": {"parent": "chest", "offset": [0, -0.024067, -0.849597]},
        "skull": {"parent": "neck", "offset": [0, -0.368394, -0.350125]},
        "HR_thigh": {"parent": "pelvis", "offset": [-0.127009, -0.842842, 0.456906]},
        "HR_shin": {"parent": "HR_thigh", "offset": [-0.000001, 0.238963, -0.232403]},
        "HR_foot": {"parent": "HR_shin", "offset": [-0.008380, 0.168601, 0.217347]},
        "HL_thigh": {"parent": "pelvis", "offset": [0.127008, -0.842841, 0.456906]},
        "HL_shin": {"parent": "HL_thigh", "offset": [-0.000001, 0.238963, -0.232403]},
        "HL_foot": {"parent": "HL_shin", "offset": [0.008379, 0.168600, 0.217347]},
        "FR_thigh": {"parent": "chest", "offset": [-0.122245, 0.187920, -0.889417]},
        "FR_shin": {"parent": "FR_thigh", "offset": [0.000050, 0.141754, 0.179375]},
        "FR_foot": {"parent": "FR_shin", "offset": [0.001356, 0.269873, -0.095854]},
        "FL_thigh": {"parent": "chest", "offset": [0.122205, 0.187955, -0.889340]},
        "FL_shin": {"parent": "FL_thigh", "offset": [-0.000067, 0.141706, 0.179413]},
        "FL_foot": {"parent": "FL_shin", "offset": [-0.000251, 0.268422, -0.099855]},
    }

    def rx_matrix(deg):
        r = math.radians(deg)
        c, s = math.cos(r), math.sin(r)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def compute_fk(root_rx_deg):
        R = rx_matrix(root_rx_deg)
        positions = {}
        for name, info in joints_chain.items():
            offset = np.array(info["offset"])
            if info["parent"] is None:
                positions[name] = R @ offset
            else:
                positions[name] = positions[info["parent"]] + R @ offset
        return positions

    for angle in [-91, 0, 90, 160, 180]:
        pos = compute_fk(angle)
        feet_z = [pos[f][2] for f in ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]]
        pelvis_z = pos["pelvis"][2]
        skull_z = pos["skull"][2]
        min_foot_z = min(feet_z)
        max_foot_z = max(feet_z)
        standing = min_foot_z < pelvis_z < skull_z and (max_foot_z - min_foot_z) < 0.1

        print(f"\n  Root Rx={angle:>4}°:", flush=True)
        print(f"    Pelvis Z = {pelvis_z:.3f}", flush=True)
        print(f"    Skull  Z = {skull_z:.3f}", flush=True)
        print(f"    Feet   Z = {min_foot_z:.3f} to {max_foot_z:.3f}", flush=True)
        print(f"    Standing = {'YES' if standing else 'NO'}", flush=True)

    # Find best standing angle by sweep
    print("\n=== SWEEP: Finding best standing angle ===", flush=True)
    best_angle = None
    best_score = float('inf')
    for a in range(0, 360):
        pos = compute_fk(a)
        feet_z = [pos[f][2] for f in ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]]
        pelvis_z = pos["pelvis"][2]
        skull_z = pos["skull"][2]
        min_fz = min(feet_z)
        spread = max(feet_z) - min_fz
        score = abs(min_fz) + spread * 5
        if pelvis_z > min_fz and skull_z > pelvis_z:
            if score < best_score:
                best_score = score
                best_angle = a

    if best_angle is not None:
        pos = compute_fk(best_angle)
        feet_z = [pos[f][2] for f in ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]]
        print(f"  Best angle: Rx={best_angle}°", flush=True)
        print(f"  Pelvis Z: {pos['pelvis'][2]:.3f}", flush=True)
        print(f"  Skull  Z: {pos['skull'][2]:.3f}", flush=True)
        print(f"  Feet Z: {[f'{z:.3f}' for z in feet_z]}", flush=True)

        half = math.radians(best_angle / 2.0)
        qw = math.cos(half)
        qx = math.sin(half)
        print(f"  Quaternion (w,x,y,z): ({qw:.6f}, {qx:.6f}, 0, 0)", flush=True)
    else:
        print("  No valid standing angle found!", flush=True)

    print("\nDONE - SUCCESS", flush=True)

except Exception:
    traceback.print_exc()
    print("FAILED", flush=True)

app.close()
