"""Stage 0 — rig introspection / dry-run (runs inside Blender headless).

Reads a Malbers/Poly-Art animal FBX set and produces:
  1. a proposed per-species config (JSON) that the URDF builder + reference
     extractor both consume, and
  2. a human-reviewable Markdown report.

Everything about the rig is MEASURED from the files — nothing is hand-typed:
  * bone hierarchy + rest transforms come from the model FBX's armature,
  * per-joint hinge AXIS, range, and DOF-count come from an SVD of the bone's
    actual rotation across the supplied animation clips,
  * per-link geometry coverage comes from the skinned mesh's vertex weights.

The only hand-curated thing is CANONICAL_ROLES: a bone-name -> role map for the
Malbers rig family. Because all 10 Forest-Pack animals share this rig, that map
is written once and reused; any bone NOT in the map is reported as UNKNOWN for a
human to resolve (never silently guessed).

Run (see tools/rig_pipeline/run_introspect.sh):
  blender --background --python introspect_rig.py -- \
      --species deer \
      --model  ".../Deer/Models/Poly Art/Deer.FBX" \
      --anims  ".../Deer Walk.FBX" ".../Deer Trot.FBX" ".../Deer Run.FBX" \
      --out-config configs/species/deer_polyart.json \
      --out-report docs/rig_introspection_deer.md
"""
import bpy, sys, json, math
import numpy as np
from mathutils import Quaternion

# --------------------------------------------------------------------------
# Canonical Malbers-rig role map (hand-curated ONCE, reused for every animal).
# role -> how the bone is treated when building the URDF.
# --------------------------------------------------------------------------
CANONICAL_ROLES = {
    "Root":     "root",         # FK/world root; carries root motion, not a link
    "CG":       "base_helper",  # center-of-gravity null; folds into base
    "Pelvis":   "base",         # URDF root link
    "Spine":    "spine",        "Spine1": "spine", "Spine2": "spine",
    "Neck":     "neck",         "Neck1":  "neck",
    "Head":     "head",
    "Jaw":      "cosmetic",     "Eye L": "cosmetic", "Eye R": "cosmetic",
    "L Ear":    "cosmetic",     "R Ear": "cosmetic",
    "Tail":     "tail",
    # front legs (arm naming)
    "L Clavicle": "front_shoulder", "R Clavicle": "front_shoulder",
    "L UpperArm": "front_upper",    "R UpperArm": "front_upper",
    "L Forearm":  "front_lower",    "R Forearm":  "front_lower",
    "L Hand":     "front_ankle",    "R Hand":     "front_ankle",
    "L Finger0":  "front_toe",      "R Finger0":  "front_toe",
    # hind legs (leg naming)
    "L Thigh":    "hind_upper",     "R Thigh":    "hind_upper",
    "L Calf":     "hind_lower",     "R Calf":     "hind_lower",
    "L HorseLink":"hind_hock",      "R HorseLink":"hind_hock",
    "L Foot":     "hind_ankle",     "R Foot":     "hind_ankle",
    "L Toe0":     "hind_toe",       "R Toe0":     "hind_toe",
}
# roles that become fixed geometry merged into their parent link
MERGE_ROLES = {"cosmetic", "base_helper"}
# roles that are the free-floating base / not a joint
NONJOINT_ROLES = {"root", "base"}

PURITY_2DOF = 0.90   # below this measured purity (and enough range) => flag 2-DOF
MIN_RANGE_DEG = 5.0  # ignore near-static joints for DOF flagging


# --------------------------------------------------------------------------
# FBX loading (with the blend-shape skip that Deer.FBX needs)
# --------------------------------------------------------------------------
def _reset_and_patch():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import addon_utils
    try:
        addon_utils.enable("io_scene_fbx")
    except Exception:
        pass
    # Blender 5.1's importer hard-errors on FBX in-between blend shapes, which we
    # do not need for a URDF. No-op the shape reader so the mesh still imports.
    import io_scene_fbx.import_fbx as imp
    imp.blen_read_shapes = lambda *a, **k: {}


def load_fbx(path):
    _reset_and_patch()
    bpy.ops.import_scene.fbx(filepath=path)


# --------------------------------------------------------------------------
# Model FBX: armature rest data + per-bone mesh vertex coverage
# --------------------------------------------------------------------------
def read_model(path):
    load_fbx(path)
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError(f"no armature in model FBX {path}")
    bones = {}
    for b in arm.data.bones:
        bones[b.name] = {
            "parent": b.parent.name if b.parent else None,
            "head": [round(v, 5) for v in b.head_local],
            "tail": [round(v, 5) for v in b.tail_local],
            "length": round(b.length, 5),
        }
    # per-bone weighted-vertex count across all skinned meshes
    coverage = {n: 0 for n in bones}
    meshes = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        gi_to_name = {vg.index: vg.name for vg in o.vertex_groups}
        n_w = 0
        for v in o.data.vertices:
            if not v.groups:
                continue
            n_w += 1
            # attribute vertex to its highest-weight group
            best = max(v.groups, key=lambda g: g.weight)
            nm = gi_to_name.get(best.group)
            if nm in coverage:
                coverage[nm] += 1
        meshes.append({"name": o.name, "verts": len(o.data.vertices),
                       "skinned": len(o.vertex_groups) > 0, "weighted": n_w})
    return {"armature": arm.name, "bones": bones, "coverage": coverage, "meshes": meshes}


# --------------------------------------------------------------------------
# Animation FBX: per-joint measured rotation axis / range / DOF purity
# --------------------------------------------------------------------------
def measure_anim(path):
    load_fbx(path)
    sc = bpy.context.scene
    objs = {o.name: o for o in bpy.data.objects if o.type == "EMPTY"}
    parent = {n: (o.parent.name if o.parent else None) for n, o in objs.items()}
    f0, f1 = sc.frame_start, sc.frame_end
    frames = list(range(f0, f1 + 1, 2))
    quats = {n: [] for n in objs}
    for f in frames:
        sc.frame_set(f)
        bpy.context.view_layer.update()
        for n, o in objs.items():
            quats[n].append(o.matrix_world.to_quaternion())

    out = {}
    for child, par in parent.items():
        if not par or par not in quats:
            continue
        rotvecs = []
        q0 = None
        for i in range(len(frames)):
            rrel = quats[par][i].inverted() @ quats[child][i]
            if q0 is None:
                q0 = rrel.copy()
            dR = q0.inverted() @ rrel
            ax, ang = dR.to_axis_angle()
            if ang > math.pi:
                ang -= 2 * math.pi
            rotvecs.append([ax.x * ang, ax.y * ang, ax.z * ang])
        rv = np.array(rotvecs)
        if np.allclose(rv, 0):
            out[child] = {"range_deg": 0.0, "purity": 1.0, "offaxis_deg": 0.0,
                          "axis": [0.0, 0.0, 0.0]}
            continue
        U, S, Vt = np.linalg.svd(rv - rv.mean(0), full_matrices=False)
        axis = Vt[0]
        # sign the axis so the animation's dominant motion is +
        if (rv @ axis).mean() < 0:
            axis = -axis
        proj = rv @ axis
        out[child] = {
            "range_deg": round(float(np.degrees(proj.max() - proj.min())), 1),
            "purity": round(float(S[0] ** 2 / (S ** 2).sum()), 3),
            "offaxis_deg": round(float(np.degrees((rv @ Vt[1]).max() - (rv @ Vt[1]).min())), 1),
            "axis": [round(float(v), 3) for v in axis],
        }
    return out


def _avg_axis(vlist):
    """Average unit axes (sign-aligned) and report agreement."""
    A = np.array([v["axis"] for v in vlist])
    if np.allclose(A, 0):
        return [0.0, 0.0, 0.0], 1.0
    ref = A[np.argmax([np.linalg.norm(a) for a in A])]
    aligned = np.array([a if a @ ref >= 0 else -a for a in A])
    mean = aligned.mean(0)
    n = np.linalg.norm(mean)
    mean = (mean / n).tolist() if n > 1e-6 else [0.0, 0.0, 0.0]
    # agreement = mean cosine to the mean axis (1.0 = clips agree perfectly)
    agree = float(np.mean([abs(a @ np.array(mean)) for a in aligned])) if n > 1e-6 else 0.0
    return [round(v, 3) for v in mean], round(agree, 3)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = {"anims": []}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--anims":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                a["anims"].append(argv[i]); i += 1
            continue
        a[k.lstrip("-").replace("-", "_")] = argv[i + 1]
        i += 2
    return a


def main():
    a = parse_argv()
    species = a.get("species", "animal")
    model = read_model(a["model"])
    anims = {p.split("\\")[-1].split("/")[-1]: measure_anim(p) for p in a["anims"]}

    # ---- consistency gate: model bones vs animation bones ----
    model_bones = set(model["bones"])
    anim_bones = set().union(*[set(m) for m in anims.values()]) if anims else set()
    only_model = sorted(model_bones - anim_bones - {"Root"})
    only_anim = sorted(anim_bones - model_bones - {"Root"})

    # ---- roles + measured per-joint axis (averaged over clips) ----
    joints = {}
    unknown = []
    for bone, info in model["bones"].items():
        role = CANONICAL_ROLES.get(bone)
        if role is None:
            unknown.append(bone)
            role = "UNKNOWN"
        meas = [anims[c][bone] for c in anims if bone in anims[c]]
        axis, agree = _avg_axis(meas) if meas else ([0.0, 0.0, 0.0], 0.0)
        rng = round(float(np.mean([m["range_deg"] for m in meas])), 1) if meas else 0.0
        pur = round(float(np.mean([m["purity"] for m in meas])), 3) if meas else 1.0
        off = round(float(np.mean([m["offaxis_deg"] for m in meas])), 1) if meas else 0.0
        dof = 2 if (role not in MERGE_ROLES | NONJOINT_ROLES
                    and pur < PURITY_2DOF and rng > MIN_RANGE_DEG) else 1
        joints[bone] = {"role": role, "parent": info["parent"], "axis": axis,
                        "clip_agreement": agree, "range_deg": rng, "purity": pur,
                        "offaxis_deg": off, "dof_recommended": dof,
                        "coverage_verts": model["coverage"].get(bone, 0)}

    proposed = {
        "species": species, "rig_family": "malbers_forest",
        "model_fbx": a["model"], "anim_fbxs": a["anims"],
        "note": "DRAFT from introspect_rig.py — review before building URDF.",
        "merge_into_parent": [b for b, j in joints.items() if j["role"] in MERGE_ROLES],
        "joints": joints,
    }
    with open(a["out_config"], "w") as f:
        json.dump(proposed, f, indent=2)

    # ---- Markdown report ----
    def row(b, j):
        flag = ""
        if j["role"] == "UNKNOWN": flag = " ⚠️UNKNOWN"
        elif j["dof_recommended"] == 2: flag = " ⚠️2-DOF"
        elif j["role"] not in MERGE_ROLES | NONJOINT_ROLES and j["coverage_verts"] == 0:
            flag = " ⚠️no-geom"
        return (f"| {b} | {j['role']} | {j['parent']} | {j['range_deg']} | "
                f"{j['purity']} | {j['offaxis_deg']} | {j['axis']} | "
                f"{j['clip_agreement']} | {j['coverage_verts']} | {j['dof_recommended']}{flag} |")

    order = [b for b in model["bones"]]
    lines = [
        f"# Rig introspection — {species} (Malbers Forest rig)",
        "",
        f"- model: `{a['model']}`",
        f"- anims: {', '.join(anims)}",
        f"- armature bones: {len(model['bones'])}",
        "",
        "## Consistency gates",
        f"- bones in model but not animated: {only_model or 'none'}",
        f"- bones animated but not in model: {only_anim or 'none'}",
        f"- UNKNOWN bones (not in canonical map): {unknown or 'none'}",
        "",
        "## Per-joint (measured from animation)",
        "",
        "purity 1.0 = clean 1-DOF hinge; <0.9 => 2-DOF. clip_agreement = do walk/trot/run agree on the axis (1.0 best). coverage = mesh verts weighted to this bone.",
        "",
        "| bone | role | parent | range° | purity | off° | axis | clip_agree | cov | DOF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [row(b, joints[b]) for b in order]
    lines += [
        "",
        "## Meshes",
        "",
        "| mesh | verts | skinned | weighted |",
        "|---|---|---|---|",
    ] + [f"| {m['name']} | {m['verts']} | {m['skinned']} | {m['weighted']} |" for m in model["meshes"]]
    lines += [
        "",
        "## What a reviewer should check",
        "1. No ⚠️UNKNOWN bones (else extend CANONICAL_ROLES).",
        "2. ⚠️2-DOF joints — confirm which get stacked revolutes (hips, neck, spine-root).",
        "3. ⚠️no-geom actuated links — a bone with 0 weighted verts has no collision.",
        "4. clip_agreement low (<0.8) on a joint => its axis is unstable across gaits; inspect.",
        "5. Consistency gates all empty.",
    ]
    with open(a["out_report"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"WROTE_CONFIG:{a['out_config']}")
    print(f"WROTE_REPORT:{a['out_report']}")
    print(f"UNKNOWN:{unknown}")
    print(f"ONLY_MODEL:{only_model}")
    print(f"ONLY_ANIM:{only_anim}")


main()
