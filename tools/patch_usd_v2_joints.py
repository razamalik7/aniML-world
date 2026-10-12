"""Patch deer_world.usd to v2 (21 DOF) by adding 6 new joints + 6 zero-length frame bones.

Why this script: Isaac Sim's URDF importer crashes on RTX 50-series, so we modify
the existing 15-DOF USD in-place via pxr instead of regenerating from URDF.

Adds:
  4 hip-abduction frame bones (HR_hip_abd, HL_hip_abd, FR_hip_abd, FL_hip_abd) with
    Z-axis revolute joints from their respective parent (pelvis or chest).
  1 spine-yaw frame bone (chest_yaw_link) with Y-axis revolute from pelvis.
  1 neck-yaw  frame bone (neck_yaw_link)  with Y-axis revolute from chest.

The 6 existing X-axis joints (pelvis_to_chest, pelvis_to_HR_thigh, ...) get RE-PARENTED
to the new frame bones and renamed (e.g. pelvis_to_HR_thigh -> HR_hipabd_to_HR_thigh).
Their localPos0 becomes (0,0,0) since the frame bone is positioned at the joint origin.

Source URDF: assets/deer_urdf/deer_world.urdf  (already updated to v2 by hand)
Source USD:  deer_world.pre_v2.usd             (backup of original 15-DOF USD)
Output USD:  deer_world.usd                    (overwrites current in place)
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


REPO = Path(__file__).resolve().parent.parent
SRC_USD = REPO / "deer_world.pre_v2.usd"   # the backup we made just before editing
DST_USD = REPO / "deer_world.usd"


# Stage 0 spec (matches the URDF v2 we just wrote).
# (joint_name, parent_link, child_frame_link, [localPos0 in parent frame], axis, limit_lower_deg, limit_upper_deg, effort, max_velocity)
NEW_FRAME_JOINTS = [
    # Spine yaw: pelvis -> chest_yaw_link  (Y-axis)
    ("pelvis_to_chest_yaw",  "pelvis", "chest_yaw_link",
     (-0.000001, 0.011858, -0.177186), "Y", -17.19, 17.19, 120.0, 458.366),
    # Hind right HAA
    ("pelvis_to_HR_hipabd",  "pelvis", "HR_hip_abd",
     (-0.127008, 0.189755, 0.187936), "Z", -22.918, 22.918, 100.0, 572.958),
    # Hind left HAA
    ("pelvis_to_HL_hipabd",  "pelvis", "HL_hip_abd",
     ( 0.127008, 0.189755, 0.187934), "Z", -22.918, 22.918, 100.0, 572.958),
    # Neck yaw: chest -> neck_yaw_link (Y-axis)
    ("chest_to_neck_yaw",    "chest",  "neck_yaw_link",
     (0.000008, 0.004488, -0.426567), "Y", -34.377, 34.377, 50.0, 458.366),
    # Front right HAA
    ("chest_to_FR_hipabd",   "chest",  "FR_hip_abd",
     (-0.122050, 0.215239, -0.471956), "Z", -22.918, 22.918, 100.0, 572.958),
    # Front left HAA
    ("chest_to_FL_hipabd",   "chest",  "FL_hip_abd",
     ( 0.121934, 0.215348, -0.471727), "Z", -22.918, 22.918, 100.0, 572.958),
]

# (old_joint_name, new_joint_name, new_parent_frame_link)
# These are the 6 existing X-axis joints that get RE-PARENTED to the new frame bones.
# Their localPos0 becomes (0,0,0) since the frame bone IS at the original joint origin.
RENAMED_JOINTS = [
    ("pelvis_to_chest",      "chest_yaw_to_chest",   "chest_yaw_link"),
    ("pelvis_to_HR_thigh",   "HR_hipabd_to_HR_thigh", "HR_hip_abd"),
    ("pelvis_to_HL_thigh",   "HL_hipabd_to_HL_thigh", "HL_hip_abd"),
    ("chest_to_neck",        "neck_yaw_to_neck",     "neck_yaw_link"),
    ("chest_to_FR_thigh",    "FR_hipabd_to_FR_thigh", "FR_hip_abd"),
    ("chest_to_FL_thigh",    "FL_hipabd_to_FL_thigh", "FL_hip_abd"),
]


def main() -> None:
    print(f"[patch] source: {SRC_USD}")
    print(f"[patch] dest:   {DST_USD}")
    if not SRC_USD.is_file():
        raise FileNotFoundError(f"Backup USD not found at {SRC_USD}. "
                                f"Make sure you cp'd deer_world.usd to deer_world.pre_v2.usd first.")

    # Load source as a stage we can edit.
    stage = Usd.Stage.Open(str(SRC_USD))
    deer_path = "/deer"
    joints_path = f"{deer_path}/joints"

    # ---- 1. Create 6 new zero-length frame-bone Xforms with minimal rigid body physics ----
    print("\n[patch] adding 6 frame-bone links")
    for _, parent_link, frame_link, _, _, _, _, _, _ in NEW_FRAME_JOINTS:
        bone_path = f"{deer_path}/{frame_link}"
        bone_prim = stage.DefinePrim(bone_path, "Xform")

        # Make it a rigid body so PhysX can use it as an articulation link.
        UsdPhysics.RigidBodyAPI.Apply(bone_prim)
        UsdPhysics.MassAPI.Apply(bone_prim)

        # Tiny mass + identity inertia so it doesn't add meaningful dynamics.
        bone_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(0.001)
        bone_prim.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(1e-6, 1e-6, 1e-6))
        bone_prim.CreateAttribute("physics:centerOfMass", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0, 0, 0))
        bone_prim.CreateAttribute("physics:principalAxes", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
        bone_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        bone_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        bone_prim.CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(0.0)
        bone_prim.CreateAttribute("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0, 0, 0))
        bone_prim.CreateAttribute("physics:startsAsleep", Sdf.ValueTypeNames.Bool).Set(False)
        print(f"  + link {bone_path} (parent {parent_link})")

    # ---- 2. Re-parent the 6 existing X-axis joints to the new frame bones, rename them ----
    print("\n[patch] re-parenting + renaming 6 existing joints")
    for old_name, new_name, new_parent in RENAMED_JOINTS:
        old_path = f"{joints_path}/{old_name}"
        new_path = f"{joints_path}/{new_name}"
        old_prim = stage.GetPrimAtPath(old_path)
        if not old_prim.IsValid():
            raise RuntimeError(f"Expected joint {old_path} not found in source USD")

        # Snapshot attributes/relationships so we can rebuild under the new name.
        attrs = {a.GetName(): a.Get() for a in old_prim.GetAttributes() if a.HasAuthoredValue()}
        rels = {r.GetName(): list(r.GetTargets()) for r in old_prim.GetRelationships()}

        # Remove the old joint prim, create fresh under new name.
        stage.RemovePrim(Sdf.Path(old_path))
        new_prim = stage.DefinePrim(new_path, "PhysicsRevoluteJoint")

        # Copy attributes verbatim, EXCEPT localPos0 which becomes (0,0,0)
        # (the frame bone is positioned at the original joint origin).
        for k, v in attrs.items():
            if k == "physics:localPos0":
                v = Gf.Vec3f(0, 0, 0)
            try:
                tn = old_prim.GetAttribute(k).GetTypeName() if old_prim.GetAttribute(k) else None
                if tn is not None:
                    new_prim.CreateAttribute(k, tn).Set(v)
                else:
                    # Fallback: infer type from value
                    new_prim.CreateAttribute(k, Sdf.ValueTypeNames.Float).Set(v)
            except Exception:
                pass  # skip attributes we can't copy
        # Wait — we removed the prim before reading. Need to do this in two phases.
        # Bug — fix below.

    # The above has a bug because we cleared the prim before we tried to GetAttribute().
    # Re-implement with two phases: collect first, mutate after.
    raise RuntimeError("see fixed version below")


def main_v2() -> None:
    print(f"[patch] source: {SRC_USD}")
    print(f"[patch] dest:   {DST_USD}")
    if not SRC_USD.is_file():
        raise FileNotFoundError(f"Backup USD not found at {SRC_USD}.")

    stage = Usd.Stage.Open(str(SRC_USD))
    deer_path = "/deer"
    joints_path = f"{deer_path}/joints"

    # ---- Phase A: Snapshot existing joint state for the 6 X-axis joints to be modified ----
    snapshots = []
    for old_name, new_name, new_parent in RENAMED_JOINTS:
        old_path = f"{joints_path}/{old_name}"
        old_prim = stage.GetPrimAtPath(old_path)
        if not old_prim.IsValid():
            raise RuntimeError(f"Expected joint {old_path} not found in source USD")
        attrs = {}
        for a in old_prim.GetAttributes():
            if a.HasAuthoredValue():
                attrs[a.GetName()] = (a.GetTypeName(), a.Get())
        rels = {r.GetName(): list(r.GetTargets()) for r in old_prim.GetRelationships()}
        snapshots.append((old_name, new_name, new_parent, attrs, rels))

    # ---- Phase B: Add 6 new frame-bone links ----
    print("\n[patch] adding 6 frame-bone links")
    for _, parent_link, frame_link, _, _, _, _, _, _ in NEW_FRAME_JOINTS:
        bone_path = f"{deer_path}/{frame_link}"
        bone_prim = stage.DefinePrim(bone_path, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(bone_prim)
        UsdPhysics.MassAPI.Apply(bone_prim)
        bone_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(0.001)
        bone_prim.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(1e-6, 1e-6, 1e-6))
        bone_prim.CreateAttribute("physics:centerOfMass", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0, 0, 0))
        bone_prim.CreateAttribute("physics:principalAxes", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
        bone_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        bone_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        bone_prim.CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(0.0)
        bone_prim.CreateAttribute("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0, 0, 0))
        bone_prim.CreateAttribute("physics:startsAsleep", Sdf.ValueTypeNames.Bool).Set(False)
        print(f"  + link {bone_path} (parent {parent_link})")

    # ---- Phase C: Add 6 new HAA/yaw revolute joints ----
    print("\n[patch] adding 6 new HAA/yaw revolute joints")
    for jname, parent, child, lpos0, axis, lo_deg, hi_deg, effort, max_vel in NEW_FRAME_JOINTS:
        jpath = f"{joints_path}/{jname}"
        j = stage.DefinePrim(jpath, "PhysicsRevoluteJoint")
        j.CreateAttribute("physics:axis", Sdf.ValueTypeNames.Token).Set(axis)
        j.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*lpos0))
        j.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0, 0, 0))
        j.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
        j.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
        j.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(lo_deg)
        j.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(hi_deg)
        j.CreateAttribute("physics:breakForce", Sdf.ValueTypeNames.Float).Set(3.4028234663852886e+38)
        j.CreateAttribute("physics:breakTorque", Sdf.ValueTypeNames.Float).Set(3.4028234663852886e+38)
        j.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        j.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(False)
        j.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        j.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(max_vel)
        # Drive
        UsdPhysics.DriveAPI.Apply(j, "angular")
        j.CreateAttribute("drive:angular:physics:type", Sdf.ValueTypeNames.Token).Set("force")
        j.CreateAttribute("drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float).Set(effort)
        j.CreateAttribute("drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float).Set(0.0)
        j.CreateAttribute("drive:angular:physics:damping", Sdf.ValueTypeNames.Float).Set(0.0)
        j.CreateAttribute("drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float).Set(0.0)
        j.CreateAttribute("drive:angular:physics:targetVelocity", Sdf.ValueTypeNames.Float).Set(0.0)
        # Body relationships
        body0_rel = j.CreateRelationship("physics:body0")
        body0_rel.SetTargets([Sdf.Path(f"{deer_path}/{parent}")])
        body1_rel = j.CreateRelationship("physics:body1")
        body1_rel.SetTargets([Sdf.Path(f"{deer_path}/{child}")])
        print(f"  + joint {jpath}: {parent} -> {child} axis={axis}")

    # ---- Phase D: Remove + recreate (rename + re-parent) the 6 existing X-axis joints ----
    print("\n[patch] re-parenting + renaming 6 existing joints")
    root_layer = stage.GetRootLayer()
    for old_name, new_name, new_parent, attrs, rels in snapshots:
        old_path = f"{joints_path}/{old_name}"
        new_path = f"{joints_path}/{new_name}"
        # Remove the old prim BOTH from the stage (memory) AND the layer (on-disk).
        # stage.RemovePrim alone only deactivates; we need the primSpec gone too
        # so it doesn't get exported.
        stage.RemovePrim(Sdf.Path(old_path))
        joints_spec = root_layer.GetPrimAtPath(joints_path)
        if joints_spec is not None and old_name in joints_spec.nameChildren:
            del joints_spec.nameChildren[old_name]
        # Build the new prim
        new_prim = stage.DefinePrim(new_path, "PhysicsRevoluteJoint")
        for k, (tn, v) in attrs.items():
            if k == "physics:localPos0":
                v = Gf.Vec3f(0, 0, 0)
            try:
                new_prim.CreateAttribute(k, tn).Set(v)
            except Exception as e:
                print(f"    warn: failed to copy {k}: {e}")
        # Rebuild body0/body1 relationships, with body0 pointing to the new frame bone
        body0_rel = new_prim.CreateRelationship("physics:body0")
        body0_rel.SetTargets([Sdf.Path(f"{deer_path}/{new_parent}")])
        body1_rel = new_prim.CreateRelationship("physics:body1")
        # body1 stays the original child
        if "physics:body1" in rels:
            body1_rel.SetTargets(rels["physics:body1"])
        # Apply DriveAPI schema (so authoring drive:angular attrs is well-typed)
        UsdPhysics.DriveAPI.Apply(new_prim, "angular")
        print(f"  ~ {old_name} -> {new_name}  (parent now {new_parent})")

    # ---- Phase E: Save out ----
    stage.GetRootLayer().Export(str(DST_USD))
    print(f"\n[patch] saved {DST_USD}")
    print("\n[patch] verifying joint count + structure ...")
    verify = Usd.Stage.Open(str(DST_USD))
    n_joints = sum(1 for p in verify.Traverse() if p.GetTypeName() == "PhysicsRevoluteJoint")
    n_bodies = sum(1 for p in verify.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    print(f"  joints in v2 USD: {n_joints}  (expected 21)")
    print(f"  rigid bodies   : {n_bodies}  (expected 22 = 16 original + 6 frame bones)")
    if n_joints != 21:
        raise RuntimeError(f"Expected 21 joints, got {n_joints}")


if __name__ == "__main__":
    main_v2()
