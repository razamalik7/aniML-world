"""Survey every joint's range-of-motion across ALL animation clips (runs in Blender).

Answers "which joints are real DOF vs dead weight, and how much do they move" from
EVIDENCE instead of anatomical guessing. For each clip it measures, per joint, the
rotation range about that joint's own dominant axis, plus the 2-DOF purity.

Clips are split into LOCOMOTION vs EXPRESSIVE so a joint that only moves in a
theatrical clip (jump/death) is visible as such and not used to size the rig
("animations can cheat"). The actuate/freeze recommendation is driven by the
LOCOMOTION range; expressive range is reported for context only.

  blender --background --python survey_joint_rom.py -- \
      --anims-dir ".../Deer/Anims" --out-report docs/rig_rom_survey_deer.md
"""
import bpy, sys, os, json, math, glob
import numpy as np

LOCOMOTION = {"walk", "trot", "run", "walkback", "turn", "idle", "idle mirrored"}
ACTUATE_MIN_DEG = 6.0   # locomotion ROM below this on a leaf => candidate freeze
PURITY_2DOF = 0.90


def reset_patch():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import addon_utils
    try: addon_utils.enable("io_scene_fbx")
    except Exception: pass
    import io_scene_fbx.import_fbx as imp
    imp.blen_read_shapes = lambda *a, **k: {}


def measure_clip(path):
    reset_patch()
    bpy.ops.import_scene.fbx(filepath=path)
    sc = bpy.context.scene
    objs = {o.name: o for o in bpy.data.objects if o.type == "EMPTY"}
    parent = {n: (o.parent.name if o.parent else None) for n, o in objs.items()}
    f0, f1 = sc.frame_start, sc.frame_end
    frames = list(range(f0, f1 + 1, max(1, (f1 - f0) // 60)))
    quats = {n: [] for n in objs}
    for f in frames:
        sc.frame_set(f); bpy.context.view_layer.update()
        for n, o in objs.items():
            quats[n].append(o.matrix_world.to_quaternion())
    out = {}
    for child, par in parent.items():
        if not par or par not in quats:
            continue
        rv = []
        q0 = None
        for i in range(len(frames)):
            rrel = quats[par][i].inverted() @ quats[child][i]
            if q0 is None: q0 = rrel.copy()
            dR = q0.inverted() @ rrel
            ax, ang = dR.to_axis_angle()
            if ang > math.pi: ang -= 2 * math.pi
            rv.append([ax.x * ang, ax.y * ang, ax.z * ang])
        rv = np.array(rv)
        if np.allclose(rv, 0):
            out[child] = {"rom": 0.0, "purity": 1.0}; continue
        U, S, Vt = np.linalg.svd(rv - rv.mean(0), full_matrices=False)
        proj = rv @ Vt[0]
        out[child] = {"rom": round(float(np.degrees(proj.max() - proj.min())), 1),
                      "purity": round(float(S[0] ** 2 / (S ** 2).sum()), 3)}
    return out


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = {}
    for i in range(0, len(argv) - 1, 2):
        a[argv[i].lstrip("-").replace("-", "_")] = argv[i + 1]
    return a


def clip_name(path):
    b = os.path.basename(path)
    n = os.path.splitext(b)[0]
    return n[5:].strip().lower() if n.lower().startswith("deer ") else n.lower()


def main():
    a = parse_argv()
    fbxs = sorted(glob.glob(os.path.join(a["anims_dir"], "*.[fF][bB][xX]")))
    clips = {}
    for p in fbxs:
        nm = clip_name(p)
        try:
            clips[nm] = measure_clip(p)
        except Exception as e:
            print(f"SKIP {nm}: {str(e)[:80]}")
    # union of joints
    joints = sorted(set().union(*[set(c) for c in clips.values()]))
    loco = [c for c in clips if c in LOCOMOTION]
    expr = [c for c in clips if c not in LOCOMOTION]

    rows = []
    for j in joints:
        loco_rom = [clips[c][j]["rom"] for c in loco if j in clips[c]]
        expr_rom = [clips[c][j]["rom"] for c in expr if j in clips[c]]
        loco_pur = [clips[c][j]["purity"] for c in loco if j in clips[c] and clips[c][j]["rom"] > 5]
        maxloco = max(loco_rom) if loco_rom else 0.0
        maxexpr = max(expr_rom) if expr_rom else 0.0
        pur = round(float(np.mean(loco_pur)), 3) if loco_pur else 1.0
        dof = 2 if (pur < PURITY_2DOF and maxloco > 8) else 1
        rec = "actuate" if maxloco >= ACTUATE_MIN_DEG else (
              "FREEZE (dead in locomotion)" if maxexpr < ACTUATE_MIN_DEG else
              "expressive-only (freeze for loco)")
        rows.append((j, round(maxloco, 1), round(maxexpr, 1), pur, dof, rec))

    # order: biggest locomotion movers first
    rows.sort(key=lambda r: -r[1])
    lines = [
        "# Joint ROM survey across ALL deer clips",
        "",
        f"- clips measured: {len(clips)} ({', '.join(sorted(clips))})",
        f"- locomotion clips: {sorted(loco)}",
        f"- expressive clips: {sorted(expr)}",
        "",
        "`maxLoco` = biggest range in any locomotion clip (the number that should",
        "drive DOF choice). `maxExpr` = biggest range in any expressive/extreme clip",
        "(context only — animations can cheat there). `purity` <0.9 => 2-DOF.",
        "",
        "| joint | maxLoco° | maxExpr° | purity | DOF | recommendation |",
        "|---|---|---|---|---|---|",
    ]
    for j, ml, me, pur, dof, rec in rows:
        flag = ""
        if "FREEZE" in rec: flag = ""
        lines.append(f"| {j} | {ml} | {me} | {pur} | {dof} | {rec} |")
    lines += [
        "",
        "## Per-clip ROM matrix (degrees, dominant axis)",
        "",
        "| joint | " + " | ".join(sorted(clips)) + " |",
        "|---|" + "---|" * len(clips),
    ]
    for j in joints:
        cells = [str(clips[c][j]["rom"]) if j in clips[c] else "-" for c in sorted(clips)]
        lines.append(f"| {j} | " + " | ".join(cells) + " |")
    with open(a["out_report"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("WROTE_REPORT:" + a["out_report"])
    print("N_CLIPS:" + str(len(clips)))


main()
