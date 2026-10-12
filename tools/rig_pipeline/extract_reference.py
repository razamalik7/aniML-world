"""Extract an AMP reference clip from a Malbers animation, and VALIDATE it by
FK round-trip (runs in Blender headless).

The joint angle for each URDF joint is the twist of the child link's motion,
relative to the parent link's motion, about the joint's stored axis. "Motion"
means rotation-from-rest, where rest = the MODEL bind pose the URDF was built on.
Because the URDF uses world-aligned links, a link's orientation-from-rest equals
the bone's world rotation from its bind orientation.

Round-trip: FK the extracted (root + joint angles) back through the URDF and
compare each link's world position to the animation's actual bone position. If
the axis/rest convention is right, the error is ~mm. If it's wrong, the error is
large and tells us so — no guessing.

  blender --background --python extract_reference.py -- \
      --config configs/species/deer_polyart.json \
      --axes assets/deer_polyart_urdf/build_axes.json \
      --model ".../Deer.FBX" --anim ".../Deer Walk.FBX" \
      --out assets/motions/deer_polyart_walk.npy
"""
import bpy, sys, json, math, os
import numpy as np
from mathutils import Quaternion, Vector, Matrix

R_ISAAC = Matrix(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
Rq = R_ISAAC.to_quaternion()


def reset_patch():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import addon_utils
    try: addon_utils.enable("io_scene_fbx")
    except Exception: pass
    import io_scene_fbx.import_fbx as imp
    imp.blen_read_shapes = lambda *a, **k: {}


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return {argv[i].lstrip("-").replace("-", "_"): argv[i + 1] for i in range(0, len(argv) - 1, 2)}


def twist_angle(q, axis):
    """Signed twist angle of quaternion q about unit `axis` (swing-twist)."""
    a = Vector(axis).normalized()
    v = Vector((q.x, q.y, q.z))
    p = v.dot(a) * a                       # projection of the vector part onto axis
    tw = Quaternion((q.w, p.x, p.y, p.z))
    if tw.magnitude < 1e-8:
        return 0.0
    tw.normalize()
    ang = 2.0 * math.atan2(p.dot(a), tw.w)
    # wrap to (-pi, pi]
    while ang > math.pi: ang -= 2 * math.pi
    while ang <= -math.pi: ang += 2 * math.pi
    return ang


def main():
    a = parse_argv()
    # POSITION SCALE. Blender 5.1 imports this FBX at 1.25x the scale earlier Blender
    # versions did (measured: every root_position element is EXACTLY 1.25x the values in
    # the clip the rig was built from, std 0.0; pelvis rest height 1.1928 vs 0.9542).
    # Joint ANGLES are scale-invariant and byte-identical across the two, so only
    # POSITIONS are affected -- which is precisely why this hid for so long: the AMP
    # dataset's joint columns looked fine while root heights and speeds were inflated.
    # Everything downstream (RSI spawn height, commanded speed, ground clearance) is
    # calibrated to the 0.95 m pelvis, so positions must be emitted in THAT scale.
    pos_scale = float(a.get("position_scale", 1.0))
    if pos_scale != 1.0:
        print(f"[extract] scaling all POSITIONS by {pos_scale} "
              f"(angles are scale-invariant and unaffected)")
    cfg = json.load(open(a["config"]))
    axes = json.load(open(a["axes"]))
    J = cfg["joints"]
    joint_bones = [b for b, j in J.items() if j.get("kind") == "joint"]
    base = next(b for b, j in J.items() if j.get("kind") == "base_link")

    # ---- MODEL rest: bone world orientation (Isaac) + head position ----
    reset_patch(); bpy.ops.import_scene.fbx(filepath=a["model"])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    rest_q = {}; rest_p = {}
    for b in arm.data.bones:
        M = arm.matrix_world @ b.matrix_local
        rest_q[b.name] = (Rq @ M.to_quaternion())
        w = arm.matrix_world @ b.head_local
        rest_p[b.name] = R_ISAAC @ Vector((w.x, w.y, w.z))

    # ---- ANIM: per-frame bone world orientation + head position (Isaac) ----
    reset_patch(); bpy.ops.import_scene.fbx(filepath=a["anim"])
    sc = bpy.context.scene
    objs = {o.name: o for o in bpy.data.objects if o.type == "EMPTY"}
    parent_of = {n: (o.parent.name if o.parent else None) for n, o in objs.items()}
    # Sample the ACTION's real keyframe range, not the SCENE range.
    # BUG FIXED 2026-07-24: using sc.frame_start/frame_end sampled 250 frames of a
    # ~104-frame walk action, so frames 104-249 just held the final pose. That made
    # the emitted clip 58% dead air -- and since amp-rsl-rl finite-differences
    # root_position, the discriminator learned an "expert" that is motionless 60%
    # of the time and rewarded the policy for STANDING STILL. Cost ~7 training runs
    # of reward tuning against a corrupt target. Falls back to the scene range only
    # if nothing has animation data.
    _ranges = []
    for _o in bpy.data.objects:
        _ad = getattr(_o, "animation_data", None)
        if _ad is not None and _ad.action is not None:
            _s, _e = _ad.action.frame_range
            _ranges.append((int(round(_s)), int(round(_e))))
    if _ranges:
        f0, f1 = min(r[0] for r in _ranges), max(r[1] for r in _ranges)
        print(f"[extract] action keyframe range: {f0}..{f1} "
              f"(scene range was {sc.frame_start}..{sc.frame_end})")
    else:
        f0, f1 = sc.frame_start, sc.frame_end
        print(f"[extract] WARNING: no action data found; falling back to scene "
              f"range {f0}..{f1} -- verify the clip for dead air!")
    frames = list(range(f0, f1 + 1))
    Q = {n: [] for n in objs}; P = {n: [] for n in objs}
    for f in frames:
        sc.frame_set(f); bpy.context.view_layer.update()
        for n, o in objs.items():
            Q[n].append(Rq @ o.matrix_world.to_quaternion())
            t = o.matrix_world.translation
            P[n].append(R_ISAAC @ Vector((t.x, t.y, t.z)))

    # link orientation-from-rest D(t) = q_anim(t) @ q_rest^-1  (world frame)
    def D(bone, i):
        if bone not in Q or bone not in rest_q:
            return Quaternion()
        return Q[bone][i] @ rest_q[bone].inverted()

    # parent-link resolver (skip merges)
    def plink(bone):
        par = J[bone]["parent"]
        while par is not None and par in J and J[par].get("kind") == "merge_geometry":
            par = J[par]["parent"]
        return par

    # ---- joint axes ----
    # BUG FIXED 2026-07-25: this ALWAYS re-measured the axes by SVD and then
    # overwrote the axes json, silently ignoring the --axes file the docstring says
    # to pass. That is fine for a single clip, but it makes every ADDITIONAL clip
    # (trot, run, idle, ...) get its OWN axis convention -- so their joint angles
    # would not be comparable to each other, to the URDF, or to the AMP dataset.
    # Default now: USE the supplied axes. Pass --remeasure_axes 1 to re-derive
    # them (only correct when (re)building the rig from a single canonical clip).
    remeasure = str(a.get("remeasure_axes", "0")).lower() in ("1", "true", "yes")
    axis1 = {}; axis2 = {}
    if not remeasure:
        missing = [b for b in joint_bones if b not in axes]
        if missing:
            raise SystemExit(f"[extract] --axes is missing {len(missing)} joint bones "
                             f"(e.g. {missing[:4]}). Pass --remeasure_axes 1 to derive them.")
        for bone in joint_bones:
            axis1[bone] = list(axes[bone]["axis1"])
            axis2[bone] = list(axes[bone]["axis2"])
        print(f"[extract] using SUPPLIED axes for {len(joint_bones)} joints "
              f"(consistent with the URDF; not overwriting the axes file)")
    else:
        # Measure each joint axis from the SAME quantity extraction decomposes:
        # relL(t) = D(parent)^-1 @ D(child). Self-consistent by construction, so the
        # twist about this axis reproduces the motion (validated by round-trip).
        for bone in joint_bones:
            par = plink(bone)
            rv = []
            for i in range(len(frames)):
                r = D(par, i).inverted() @ D(bone, i)
                ax, ang = r.to_axis_angle()
                if ang > math.pi: ang -= 2 * math.pi
                rv.append([ax.x * ang, ax.y * ang, ax.z * ang])
            rv = np.array(rv)
            if np.allclose(rv, 0):
                axis1[bone] = [0, 0, 1]; axis2[bone] = [1, 0, 0]; continue
            U, S, Vt = np.linalg.svd(rv - rv.mean(0), full_matrices=False)
            a1 = Vt[0]
            if (rv @ a1).mean() < 0: a1 = -a1
            axis1[bone] = [float(x) for x in a1]
            axis2[bone] = [float(x) for x in Vt[1]]
        print(f"[extract] RE-MEASURED axes for {len(joint_bones)} joints")

    # ---- extract angles about the self-measured axes ----
    jnames = []; cols = []
    for bone in joint_bones:
        j = J[bone]; par = plink(bone)
        a1 = axis1[bone]
        series1 = []; series2 = [] if j["dof"] == 2 else None
        for i in range(len(frames)):
            relL = D(par, i).inverted() @ D(bone, i)
            t1 = twist_angle(relL, a1)
            series1.append(t1)
            if j["dof"] == 2:
                res = Quaternion(Vector(a1), t1).inverted() @ relL
                series2.append(twist_angle(res, axis2[bone]))
        # Emit URDF joint names (safe spelling: spaces->underscores, _dof0
        # intermediate for 2-DOF joints) EXACTLY as build_urdf.py writes them.
        # amp-rsl-rl's MotionLoader.load_data matches reference joints to sim
        # joints by STRING EQUALITY; if these don't equal the URDF/USD joint
        # names, every column silently resolves to zeros (garbage style). The
        # old "{par}->{bone}" spelling did NOT match and had to be relabeled
        # post-hoc -- do not reintroduce it. (Cross-check: the relabel-with-
        # structural-proof in the pipeline notes validates this each rig.)
        safe = lambda s: s.replace(" ", "_")
        if j["dof"] == 2:
            jnames.append(f"{safe(par)}_to_{safe(bone)}_dof0")
            cols.append(series1)
            jnames.append(f"{safe(bone)}_dof0_to_{safe(bone)}")
            cols.append(series2)
        else:
            jnames.append(f"{safe(par)}_to_{safe(bone)}")
            cols.append(series1)

    joint_pos = np.array(cols, dtype=np.float32).T   # (T, 38)
    # expose the axes to the round-trip; only WRITE them when we re-measured
    # (otherwise we would clobber the rig's canonical axes with a per-clip copy).
    axes = {b: {"axis1": axis1[b], "axis2": axis2[b]} for b in joint_bones}
    if remeasure:
        json.dump(axes, open(os.path.join(os.path.dirname(a["out"]), "deer_polyart_axes.json"), "w"), indent=1)

    # ---- root = pelvis (base) world pos + orientation-from-rest ----
    root_pos = np.array([[P[base][i].x, P[base][i].y, P[base][i].z] for i in range(len(frames))], dtype=np.float32) * pos_scale
    root_quat = []
    for i in range(len(frames)):
        d = D(base, i)  # base link orientation (world)
        root_quat.append([d.x, d.y, d.z, d.w])   # xyzw
    root_quat = np.array(root_quat, dtype=np.float32)

    # ---- BONE TARGETS for IK retargeting -------------------------------------
    # The analytic extraction above compresses each joint's true 3-D rotation into a
    # twist about ONE fitted axis, so the reconstructed pose drifts from the artist's:
    # 86 mm on the walk, and 218-450 mm on sleep/fall/death (the axes were fitted to
    # the WALK, so accuracy degrades away from it). That ceiling is cosmetic for a
    # single walk but blocking for ASE, which learns a latent skill space FROM a motion
    # dataset and is therefore bounded by reconstruction fidelity across many clips.
    # Dumping the animation's true per-frame bone world positions lets a downstream
    # least-squares IK solve for joint angles that MINIMISE bone-position error
    # directly, on the same rig -- no extra DOF, no larger action space.
    bone_names = [b for b in J if b in P]
    bone_pos = np.array(
        [[[P[b][i].x, P[b][i].y, P[b][i].z] for b in bone_names] for i in range(len(frames))],
        dtype=np.float32) * pos_scale                      # (T, n_bones, 3)
    # rest offsets + parent map so the IK tool can rebuild FK without Blender
    ik_offsets = {}
    for bone in joint_bones:
        par = plink(bone)
        ik_offsets[bone] = {
            "parent": par,
            "offset": [float(x) * pos_scale for x in (rest_p[bone] - rest_p.get(par, rest_p[base]))],
            "dof": int(J[bone]["dof"]),
        }

    fps = float(sc.render.fps)

    # ---- self-check: the emitted clip IS the AMP expert distribution ----
    # amp-rsl-rl derives root velocity by finite-differencing root_position, so
    # frozen frames teach "stand still" and seams teach teleports. Both are silent
    # failures downstream (no error, just a policy that won't walk), so shout here.
    _v = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1) * fps
    _frozen = float((_v < 0.01).mean()) if len(_v) else 1.0
    _tele = int((_v > 3.0).sum())
    _still = int((np.abs(np.diff(joint_pos, axis=0)).max(axis=1) < 1e-6).sum())
    print(f"[extract] frames={len(joint_pos)} fps={fps}  root speed "
          f"mean={_v.mean():.3f} median={np.median(_v):.3f} m/s")
    print(f"[extract] frozen-root frames={100*_frozen:.1f}%  teleports(>3 m/s)={_tele}  "
          f"identical-pose frames={_still}")
    if _frozen > 0.10 or _tele or _still:
        print("[extract] *** WARNING: clip has dead air / teleport seams. amp-rsl-rl "
              "will train the discriminator on this distribution -- run "
              "tools/rig_pipeline/repair_reference.py before training. ***")

    out = {"joints_list": jnames, "joint_positions": joint_pos,
           "root_position": root_pos, "root_quaternion": root_quat, "fps": fps,
           # IK-retargeting payload (see the BONE TARGETS block above)
           "bone_names": bone_names, "bone_positions": bone_pos,
           "ik_offsets": ik_offsets, "axes": axes}
    # NOTE: the FK round-trip runs AFTER this save, so its error is not embedded here.
    # It is printed as ROUNDTRIP_ERR_M / OVERALL_MEAN_MM -- capture it from the log.
    np.save(a["out"], out, allow_pickle=True)

    # ============ FK ROUND-TRIP VALIDATION ============
    # reconstruct each link's world transform from (root, angles) and compare to
    # the animation's actual bone head positions.
    offset = {}
    for bone in joint_bones:
        par = J[bone]["parent"]
        while par is not None and par in J and J[par].get("kind") == "merge_geometry":
            par = J[par]["parent"]
        offset[bone] = (rest_p[bone] - rest_p.get(par, rest_p[base]))
    check_bones = ["L Toe0", "R Toe0", "L Finger0", "R Finger0", "Head", "L Foot", "Neck1"]
    errs = {b: [] for b in check_bones}
    col_of = {}
    k = 0
    for bone in joint_bones:
        col_of[bone] = k; k += 1 + (1 if J[bone]["dof"] == 2 else 0)
    for i in range(0, len(frames), 3):
        Dl = {base: D(base, i)}; pos = {base: P[base][i]}
        # forward pass in tree order (joint_bones is roughly root-first)
        for bone in joint_bones:
            par = J[bone]["parent"]
            while par is not None and par in J and J[par].get("kind") == "merge_geometry":
                par = J[par]["parent"]
            if par not in Dl:
                Dl[par] = D(par, i); pos[par] = P.get(par, {i: rest_p.get(par, rest_p[base])})[i] if par in P else rest_p.get(par, rest_p[base])
            ci = col_of[bone]; a1 = axes[bone]["axis1"]
            rot = Quaternion(a1, float(joint_pos[i, ci]))
            if J[bone]["dof"] == 2:
                rot = rot @ Quaternion(axes[bone]["axis2"], float(joint_pos[i, ci + 1]))
            Dl[bone] = Dl[par] @ rot
            pos[bone] = pos[par] + (Dl[par] @ offset[bone])
        for b in check_bones:
            if b in pos and b in P:
                errs[b].append((pos[b] - P[b][i]).length)
    print("ROUNDTRIP_ERR_M (mean position error per bone):")
    allm = []
    for b in check_bones:
        if errs[b]:
            m = float(np.mean(errs[b])); allm.append(m)
            print(f"  {b:10s} {m*1000:7.1f} mm")
    print(f"OVERALL_MEAN_MM:{np.mean(allm)*1000:.1f}")
    print(f"WROTE_NPY:{a['out']}  shape={joint_pos.shape}  fps={fps}")


main()
