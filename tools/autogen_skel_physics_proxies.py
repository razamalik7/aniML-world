#!/usr/bin/env python3
"""Auto-place capsule rigid-body links + optional revolute joints from UsdSkel.Skeleton.

Uses joint **rest** (or bind) matrices as **parent-local** chains, then multiplies the
Skeleton prim's world Xform to place proxies in stage space. This matches many DCC
exports; if capsules land wrong, try ``--rest-mode absolute`` (see below).

Run with Isaac Sim / Isaac Lab Python (``pxr`` + ``UsdPhysics``):

  cd C:\\isaac-lab\\IsaacLab
  .\\isaaclab.bat -p C:\\path\\aniML_world\\tools\\autogen_skel_physics_proxies.py -- \\
      --input path\\deer_rl_workspace.usd \\
      --output path\\deer_rl_workspace_physics.usd \\
      --rig-root /World/deer1/Deer_001_rig \\
      --joints

Then open the **output** USD in Isaac Sim, verify in **Play**, tune axes/limits in UI.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import Optional

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel


def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    s = s.strip("_")
    return s or "joint"


def _find_skeleton(stage: Usd.Stage, explicit: Optional[str]) -> Usd.Prim:
    if explicit:
        p = stage.GetPrimAtPath(explicit)
        if not p.IsValid():
            raise SystemExit(f"Invalid --skeleton path: {explicit}")
        if not p.IsA(UsdSkel.Skeleton):
            raise SystemExit(f"Prim is not UsdSkel.Skeleton: {explicit}")
        return p
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton):
            return prim
    raise SystemExit("No UsdSkel.Skeleton found; pass --skeleton /path")


def _compose_local_chain(
    joints: list[str],
    parents: list[int],
    local_mats: list[Gf.Matrix4d],
) -> list[Gf.Matrix4d]:
    """Assume local_mats[i] is parent-local transform for joint i."""
    n = len(joints)
    world_skel = [Gf.Matrix4d(1.0) for _ in range(n)]
    world_skel[0] = local_mats[0]
    for i in range(1, n):
        p = parents[i]
        if p < 0:
            world_skel[i] = local_mats[i]
        else:
            world_skel[i] = world_skel[p] * local_mats[i]
    return world_skel


def _compose_absolute(local_mats: list[Gf.Matrix4d]) -> list[Gf.Matrix4d]:
    """Each local_mats[i] is already in skeleton space (no parent multiply)."""
    return list(local_mats)


def _get_joint_local_matrices(skel: UsdSkel.Skeleton) -> tuple[list[Gf.Matrix4d], str]:
    rest_attr = skel.GetRestTransformsAttr()
    bind_attr = skel.GetBindTransformsAttr()
    if rest_attr and rest_attr.HasAuthoredValue():
        data = list(rest_attr.Get())
        return data, "rest"
    if bind_attr and bind_attr.HasAuthoredValue():
        data = list(bind_attr.Get())
        return data, "bind"
    raise SystemExit("Skeleton has no restTransforms or bindTransforms.")


def _stage_world_xform(prim: Usd.Prim, t: Usd.TimeCode) -> Gf.Matrix4d:
    xf = UsdGeom.Xformable(prim)
    return xf.ComputeLocalToWorldTransform(t)


def _quat_from_basis(x: Gf.Vec3d, y: Gf.Vec3d, z: Gf.Vec3d) -> Gf.Quatd:
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(x[0], x[1], x[2], 0.0))
    m.SetRow(1, Gf.Vec4d(y[0], y[1], y[2], 0.0))
    m.SetRow(2, Gf.Vec4d(z[0], z[1], z[2], 0.0))
    return m.ExtractRotationQuat()


def _ortho_bases(bone: Gf.Vec3d, world_up: Gf.Vec3d) -> tuple[Gf.Vec3d, Gf.Vec3d, Gf.Vec3d]:
    b = bone.GetNormalized()
    if b.GetLength() < 1e-8:
        b = Gf.Vec3d(0, 1, 0)
    up = world_up.GetNormalized()
    side = Gf.Cross(b, up)
    if side.GetLength() < 1e-6:
        side = Gf.Cross(b, Gf.Vec3d(1.0, 0.0, 0.0))
    side = side.GetNormalized()
    up2 = Gf.Cross(side, b).GetNormalized()
    return side, b, up2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input USD/USDA/USDC")
    ap.add_argument(
        "--output",
        default="",
        help="Output USD path (required unless --dry-run; must differ from input)",
    )
    ap.add_argument("--rig-root", required=True, help="Articulation root Xform (e.g. /World/deer1/Deer_001_rig)")
    ap.add_argument("--skeleton", default="", help="Skeleton prim path (default: first Skeleton in stage)")
    ap.add_argument("--scope-name", default="physics_autogen", help="Child scope under rig-root for links/joints")
    ap.add_argument("--radius-scale", type=float, default=0.18, help="Capsule radius = bone_len * scale")
    ap.add_argument("--min-bone-length", type=float, default=0.02, help="Skip shorter bones (meters-ish)")
    ap.add_argument("--mass-per-link", type=float, default=3.0, help="UsdPhysics.MassAPI mass per capsule link")
    ap.add_argument("--include-regex", default=".*", help="Only joints whose full name matches")
    ap.add_argument("--exclude-regex", default="", help="Drop matching joint names")
    ap.add_argument(
        "--rest-mode",
        choices=("local_chain", "absolute"),
        default="local_chain",
        help="How to compose skeleton matrices (see docstring)",
    )
    ap.add_argument("--joints", action="store_true", help="Author PhysicsRevoluteJoint between links (experimental)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = ap.parse_args()
    if not args.dry_run and not args.output.strip():
        print("ERROR: pass --output file.usd (or use --dry-run)", file=sys.stderr)
        return 1

    stage = Usd.Stage.Open(args.input)
    if not stage:
        print(f"ERROR: could not open {args.input}", file=sys.stderr)
        return 1

    rig = stage.GetPrimAtPath(args.rig_root)
    if not rig.IsValid():
        print(f"ERROR: invalid --rig-root {args.rig_root}", file=sys.stderr)
        return 1

    skel_prim = _find_skeleton(stage, args.skeleton or None)
    skel = UsdSkel.Skeleton(skel_prim)
    joints_attr = skel.GetJointsAttr()
    if not joints_attr or not joints_attr.HasAuthoredValue():
        print("ERROR: skeleton has no joints[]", file=sys.stderr)
        return 1
    joint_names = [str(t) for t in joints_attr.Get()]
    topo = UsdSkel.Topology(joint_names)
    parents = [int(topo.GetParentIndices()[i]) for i in range(len(joint_names))]

    local_mats, src = _get_joint_local_matrices(skel)
    if len(local_mats) != len(joint_names):
        print(f"ERROR: {src}Transforms length {len(local_mats)} != joints {len(joint_names)}", file=sys.stderr)
        return 1

    if args.rest_mode == "local_chain":
        skel_space = _compose_local_chain(joint_names, parents, local_mats)
    else:
        skel_space = _compose_absolute(local_mats)

    tcode = Usd.TimeCode.Default()
    skel_world = _stage_world_xform(skel_prim, tcode)

    world_xf: list[Gf.Matrix4d] = [skel_world * m for m in skel_space]

    inc = re.compile(args.include_regex) if args.include_regex else None
    exc = re.compile(args.exclude_regex) if args.exclude_regex else None

    scope_path = f"{args.rig_root.rstrip('/')}/{args.scope_name}"
    if args.dry_run:
        print(f"Skeleton: {skel_prim.GetPath()} ({len(joint_names)} joints, {src}Transforms, mode={args.rest_mode})")
        print(f"Scope: {scope_path}")
        for i, name in enumerate(joint_names):
            p = parents[i]
            par_name = joint_names[p] if p >= 0 else "(root)"
            if p < 0:
                continue
            pi = world_xf[p].ExtractTranslation()
            ji = world_xf[i].ExtractTranslation()
            bl = (ji - pi).GetLength()
            ok = inc.match(name) if inc else True
            if exc and exc.search(name):
                ok = False
            print(f"  [{i}] {name} parent={par_name} bone_len={bl:.4f} include={ok}")
        return 0

    in_abs = os.path.abspath(os.path.normpath(args.input))
    out_abs = os.path.abspath(os.path.normpath(args.output))
    if in_abs == out_abs:
        print("ERROR: --output must not be the same file as --input", file=sys.stderr)
        return 1
    shutil.copy2(args.input, args.output)

    stage = Usd.Stage.Open(args.output, Usd.Stage.LoadAll)
    if not stage:
        print(f"ERROR: could not open output {args.output}", file=sys.stderr)
        return 1

    rig = stage.GetPrimAtPath(args.rig_root)
    if not rig.IsValid():
        print(f"ERROR: invalid --rig-root {args.rig_root}", file=sys.stderr)
        return 1

    skel_prim = _find_skeleton(stage, args.skeleton or None)
    skel = UsdSkel.Skeleton(skel_prim)
    joint_names = [str(t) for t in skel.GetJointsAttr().Get()]
    topo = UsdSkel.Topology(joint_names)
    parents = [int(topo.GetParentIndices()[i]) for i in range(len(joint_names))]
    local_mats, src = _get_joint_local_matrices(skel)
    if args.rest_mode == "local_chain":
        skel_space = _compose_local_chain(joint_names, parents, local_mats)
    else:
        skel_space = _compose_absolute(local_mats)
    tcode = Usd.TimeCode.Default()
    skel_world = _stage_world_xform(skel_prim, tcode)
    world_xf = [skel_world * m for m in skel_space]

    scope = UsdGeom.Scope.Define(stage, scope_path)
    links_scope = UsdGeom.Scope.Define(stage, f"{scope_path}/links")
    joints_scope = UsdGeom.Scope.Define(stage, f"{scope_path}/joints")

    link_path_by_index: dict[int, Sdf.Path] = {}

    for i, jname in enumerate(joint_names):
        p = parents[i]
        if p < 0:
            continue
        if inc and not inc.match(jname):
            continue
        if exc and exc.search(jname):
            continue

        pw = world_xf[p].ExtractTranslation()
        jw = world_xf[i].ExtractTranslation()
        bone = jw - pw
        bone_len = bone.GetLength()
        if bone_len < args.min_bone_length:
            continue

        radius = max(bone_len * args.radius_scale, 0.01)
        cyl_h = max(bone_len - 2.0 * radius, 1e-4)
        mid = (pw + jw) * 0.5

        side, yaxis, zaxis = _ortho_bases(bone, Gf.Vec3d(0.0, 1.0, 0.0))
        rot = _quat_from_basis(side, yaxis, zaxis)

        link_name = f"link_{_sanitize(jname)}_{i}"
        link_path = f"{links_scope.GetPrim().GetPath()}/{link_name}"
        link_xf = UsdGeom.Xform.Define(stage, link_path)
        UsdGeom.Xformable(link_xf).ClearXformOpOrder()
        UsdGeom.Xformable(link_xf).AddTranslateOp().Set(Gf.Vec3d(mid[0], mid[1], mid[2]))
        UsdGeom.Xformable(link_xf).AddOrientOp().Set(rot)

        cap_path = f"{link_path}/CollisionCapsule"
        cap = UsdGeom.Capsule.Define(stage, cap_path)
        cap.CreateRadiusAttr(radius)
        cap.CreateHeightAttr(float(cyl_h))
        cap.CreateAxisAttr(UsdGeom.Tokens.y)
        UsdGeom.Xformable(cap).AddRotateXOp().Set(0.0)

        UsdPhysics.RigidBodyAPI.Apply(link_xf.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        UsdPhysics.MassAPI.Apply(link_xf.GetPrim()).CreateMassAttr(args.mass_per_link)

        link_path_by_index[i] = Sdf.Path(link_path)

    def body_for_index(idx: int) -> Sdf.Path:
        if idx <= 0:
            return Sdf.Path(args.rig_root)
        cur = idx
        while cur >= 0:
            if cur in link_path_by_index:
                return link_path_by_index[cur]
            cur = parents[cur]
        return Sdf.Path(args.rig_root)

    if args.joints:
        ji = 0
        for i, jname in enumerate(joint_names):
            p = parents[i]
            if p < 0:
                continue
            if i not in link_path_by_index:
                continue
            body0 = body_for_index(p)
            body1 = link_path_by_index[i]
            pivot = world_xf[p].ExtractTranslation()

            T0 = _stage_world_xform(stage.GetPrimAtPath(body0), tcode)
            T1 = _stage_world_xform(stage.GetPrimAtPath(body1), tcode)
            inv0 = T0.GetInverse()
            inv1 = T1.GetInverse()
            lp0 = inv0.TransformAffine(Gf.Vec3d(pivot))
            lp1 = inv1.TransformAffine(Gf.Vec3d(pivot))

            bone = world_xf[i].ExtractTranslation() - pivot
            side, _y, _z = _ortho_bases(bone, Gf.Vec3d(0.0, 1.0, 0.0))
            axis_world = side
            try:
                axis_local0 = inv0.TransformDir(Gf.Vec3d(axis_world)).GetNormalized()
            except AttributeError:
                r = inv0.ExtractRotationMatrix()
                axis_local0 = (r * Gf.Vec3d(axis_world)).GetNormalized()

            jpath = f"{joints_scope.GetPrim().GetPath()}/rev_{ji}_{_sanitize(jname)}"
            ji += 1
            rj = UsdPhysics.RevoluteJoint.Define(stage, jpath)
            rj.CreateBody0Rel().SetTargets([body0])
            rj.CreateBody1Rel().SetTargets([body1])
            rj.CreateLocalPos0Attr(Gf.Vec3f(float(lp0[0]), float(lp0[1]), float(lp0[2])))
            rj.CreateLocalPos1Attr(Gf.Vec3f(float(lp1[0]), float(lp1[1]), float(lp1[2])))
            rj.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            rj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            rj.CreateAxisAttr(Gf.Vec3f(float(axis_local0[0]), float(axis_local0[1]), float(axis_local0[2])))
            rj.CreateLowerLimitAttr(float(-3.12413936107))  # ~-179 deg rad
            rj.CreateUpperLimitAttr(float(3.12413936107))

    print(f"Wrote {args.output}")
    print(f"  Links created: {len(link_path_by_index)}  (source={src}, rest_mode={args.rest_mode})")
    if args.joints:
        print(f"  Joints: experimental — verify axes/limits in Isaac Sim.")
    stage.GetRootLayer().Save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
