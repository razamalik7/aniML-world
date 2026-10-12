"""Build a physics URDF from a Malbers rig + the finalized species config.

Runs in Blender headless. Reads:
  * the MODEL fbx  -> rest skeleton + skinned mesh (collision/visual geometry)
  * a WALK anim    -> per-joint rotation axes (SVD) and joint limits (measured)
  * configs/species/<name>.json -> the human-approved DOF spec (kind/dof/merge)

Coordinate discipline (kills the old fix-script mess): everything read in Blender
world (Z-up, metres, faces -Y after FBX import) is transformed by R = +90 deg about
Z so the URDF is Isaac-native (+X forward, +Y left, Z up). All link frames are
world-aligned (joint origin rpy="0 0 0"), so a joint axis is simply a world vector
and the reference extractor can use the identical axis -> no sign drift.

  blender --background --python build_urdf.py -- \
      --config configs/species/deer_polyart.json \
      --model  ".../Deer.FBX" --anim ".../Deer Walk.FBX" \
      --outdir assets/deer_polyart_urdf --species deer
"""
import bpy, sys, os, json, math, struct
import numpy as np
from mathutils import Vector, Matrix, Quaternion

# +90 deg about Z: Blender-world (-Y forward) -> Isaac (+X forward, +Y left, Z up)
R_ISAAC = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def reset_patch():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import addon_utils
    try: addon_utils.enable("io_scene_fbx")
    except Exception: pass
    import io_scene_fbx.import_fbx as imp
    imp.blen_read_shapes = lambda *a, **k: {}


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = {}
    for i in range(0, len(argv) - 1, 2):
        a[argv[i].lstrip("-").replace("-", "_")] = argv[i + 1]
    return a


# ---------------------------------------------------------------- geometry io
def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\0" * 80); f.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0]); L = np.linalg.norm(n)
            n = n / L if L > 0 else np.zeros(3)
            f.write(struct.pack("<3f", *n))
            for v in t: f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")


def convex_hull(points):
    """Return (verts, faces) of the convex hull via Blender bmesh."""
    import bmesh
    bm = bmesh.new()
    for p in points:
        bm.verts.new(p)
    try:
        bmesh.ops.convex_hull(bm, input=bm.verts)
    except Exception:
        pass
    bm.verts.ensure_lookup_table()
    verts = [np.array(v.co) for v in bm.verts]
    idx = {v: i for i, v in enumerate(bm.verts)}
    faces = []
    for fc in bm.faces:
        vs = [idx[v] for v in fc.verts]
        for k in range(1, len(vs) - 1):
            faces.append((vs[0], vs[k], vs[k + 1]))
    bm.free()
    return verts, faces


def hull_volume(verts, faces):
    v = 0.0
    for a, b, c in faces:
        v += np.dot(verts[a], np.cross(verts[b], verts[c])) / 6.0
    return abs(v)


# --------------------------------------------------------- read model geometry
def read_model(path, weld_unskinned=None):
    reset_patch()
    bpy.ops.import_scene.fbx(filepath=path)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    # rest bone head in Isaac frame
    bone_head = {}
    for b in arm.data.bones:
        w = arm.matrix_world @ b.head_local
        bone_head[b.name] = R_ISAAC @ np.array([w.x, w.y, w.z])
    # assign every skinned vertex to its highest-weight bone; collect per bone
    body_meshes = [o for o in bpy.data.objects
                   if o.type == "MESH" and o.name.startswith("Deer")]
    from collections import Counter
    bone_pts = {}    # per-bone vertices (for bbox / inertia)
    bone_faces = {}  # per-bone real mesh triangles (for visual + collision source)
    for o in body_meshes:
        gi = {vg.index: vg.name for vg in o.vertex_groups}
        vbone = {}
        for v in o.data.vertices:
            if not v.groups:
                continue
            g = max(v.groups, key=lambda x: x.weight)
            nm = gi.get(g.group)
            vbone[v.index] = nm
            w = o.matrix_world @ v.co
            bone_pts.setdefault(nm, []).append(R_ISAAC @ np.array([w.x, w.y, w.z]))
        # assign each face to the bone owning the majority of its vertices; keep
        # the real triangles so visual + collision use the actual mesh, not a hull
        for poly in o.data.polygons:
            vs = list(poly.vertices)
            bs = [vbone[vi] for vi in vs if vi in vbone]
            if not bs:
                continue
            nm = Counter(bs).most_common(1)[0][0]
            co = []
            for vi in vs:
                w = o.matrix_world @ o.data.vertices[vi].co
                co.append(R_ISAAC @ np.array([w.x, w.y, w.z]))
            for k in range(1, len(vs) - 1):   # fan triangulation
                bone_faces.setdefault(nm, []).append([co[0], co[k], co[k + 1]])

    # Weld UNSKINNED meshes (e.g. the stag's antlers) rigidly to a target bone.
    # Deer.FBX ships antlers as separate objects with NO vertex groups, so the
    # skin-weight pass above skips them entirely — without this the stag rig has
    # no rack. Each welded mesh's triangles/verts are transformed into the Isaac
    # frame and folded into the target bone's link (visual + collision source),
    # exactly as if they were skinned 100% to that bone.
    for spec in (weld_unskinned or []):
        mesh_name, target = spec["mesh"], spec["bone"]
        o = bpy.data.objects.get(mesh_name)
        if o is None or o.type != "MESH":
            print(f"WELD_WARN: unskinned mesh {mesh_name!r} not found"); continue
        nv = nf = 0
        for v in o.data.vertices:
            w = o.matrix_world @ v.co
            bone_pts.setdefault(target, []).append(R_ISAAC @ np.array([w.x, w.y, w.z])); nv += 1
        for poly in o.data.polygons:
            vs = list(poly.vertices)
            co = []
            for vi in vs:
                w = o.matrix_world @ o.data.vertices[vi].co
                co.append(R_ISAAC @ np.array([w.x, w.y, w.z]))
            for k in range(1, len(vs) - 1):
                bone_faces.setdefault(target, []).append([co[0], co[k], co[k + 1]]); nf += 1
        print(f"WELDED {mesh_name!r} -> {target}: {nv} verts, {nf} tris")
    return bone_head, bone_pts, bone_faces


# ----------------------------------------------- measure axes + limits from anim
def measure_axes(path, joint_bones):
    reset_patch()
    bpy.ops.import_scene.fbx(filepath=path)
    sc = bpy.context.scene
    objs = {o.name: o for o in bpy.data.objects if o.type == "EMPTY"}
    parent = {n: (o.parent.name if o.parent else None) for n, o in objs.items()}
    f0, f1 = sc.frame_start, sc.frame_end
    frames = list(range(f0, f1 + 1, 2))
    quat = {n: [] for n in objs}
    for f in frames:
        sc.frame_set(f); bpy.context.view_layer.update()
        for n, o in objs.items():
            quat[n].append(o.matrix_world.to_quaternion())
    Rq = Matrix(((0, -1, 0), (1, 0, 0), (0, 0, 1))).to_quaternion()  # R_ISAAC as quat
    out = {}
    for child in joint_bones:
        par = parent.get(child)
        if not par or par not in quat:
            out[child] = None; continue
        rv = []; angs = []
        q0 = None
        for i in range(len(frames)):
            rrel = quat[par][i].inverted() @ quat[child][i]
            if q0 is None: q0 = rrel.copy()
            dR = q0.inverted() @ rrel
            ax, ang = dR.to_axis_angle()
            if ang > math.pi: ang -= 2 * math.pi
            # express axis in Isaac world: rotate parent-frame axis by parent world * R
            axw = (Rq @ quat[par][i]) @ ax
            rv.append([axw.x * ang, axw.y * ang, axw.z * ang])
        rv = np.array(rv)
        if np.allclose(rv, 0):
            out[child] = {"axis1": [0, 0, 1], "axis2": [1, 0, 0], "lo": -0.1, "hi": 0.1}
            continue
        U, S, Vt = np.linalg.svd(rv - rv.mean(0), full_matrices=False)
        a1 = Vt[0]
        if (rv @ a1).mean() < 0: a1 = -a1
        a2 = Vt[1]
        proj = rv @ a1
        out[child] = {"axis1": [round(float(x), 4) for x in a1],
                      "axis2": [round(float(x), 4) for x in a2],
                      "lo": round(float(proj.min()), 4), "hi": round(float(proj.max()), 4)}
    return out


# ------------------------------------------------------------------ urdf build
def vec(v): return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


def main():
    a = parse_argv()
    cfg = json.load(open(a["config"]))
    J = cfg["joints"]
    outdir = a["outdir"]; mesh_dir = os.path.join(outdir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    species = a.get("species", "animal")
    total_mass = float(a.get("mass", 70.0))

    joint_bones = [b for b, j in J.items() if j.get("kind") == "joint"]
    weld_unskinned = cfg.get("weld_unskinned", [])
    welded_links = {s["bone"] for s in weld_unskinned}
    bone_head, bone_pts, bone_faces = read_model(a["model"], weld_unskinned)
    # uniform scale to realistic size (positions + geometry; angles/axes unaffected)
    scale = float(a.get("scale", 1.0))
    if scale != 1.0:
        bone_head = {k: v * scale for k, v in bone_head.items()}
        bone_pts = {k: [p * scale for p in v] for k, v in bone_pts.items()}
        bone_faces = {k: [[v * scale for v in tri] for tri in f] for k, f in bone_faces.items()}
    axes = measure_axes(a["anim"], joint_bones)
    # CLEANUP: use the extractor's self-consistent axes if provided, so the URDF
    # and the reference clip decompose motion about identical axes.
    if a.get("axes_file") and os.path.exists(a["axes_file"]):
        ext = json.load(open(a["axes_file"]))
        for b in axes:
            if b in ext and axes[b]:
                axes[b]["axis1"] = ext[b]["axis1"]
                axes[b]["axis2"] = ext[b]["axis2"]
        print("USING_UNIFIED_AXES:" + a["axes_file"])

    # resolve which link a bone's geometry belongs to (merges redirect)
    def geom_link(bone):
        j = J.get(bone, {})
        if j.get("kind") == "merge_geometry":
            return j["merge_into"]
        return bone
    # accumulate REAL mesh triangles + verts per link (merges fold in). Visual AND
    # collision both come from these same triangles, in one step, so there is no
    # separately-regenerated collision file that could mirror independently.
    link_faces = {}; link_verts = {}
    for bone, faces in bone_faces.items():
        if bone not in J: continue
        link_faces.setdefault(geom_link(bone), []).extend(faces)
    for bone, pts in bone_pts.items():
        if bone not in J: continue
        link_verts.setdefault(geom_link(bone), []).extend(pts)
    geom_links = [l for l in link_faces if link_faces[l] and l in bone_head]

    base = next(b for b, j in J.items() if j.get("kind") == "base_link")

    # mass from ANATOMICAL per-role fractions (robust; identical across animals)
    ROLE_W = {"base": 16, "spine": 11, "neck": 3, "head": 5,
              "hind_upper": 5, "hind_lower": 2, "hind_hock": 0.8, "hind_ankle": 0.6, "hind_toe": 0.4,
              "front_shoulder": 1.5, "front_upper": 4, "front_lower": 2, "front_ankle": 0.6, "front_toe": 0.4}
    def bone_weight(link):
        if link == "Jaw": return 0.4
        if "Ear" in link: return 0.15
        if link == "Tail": return 0.8
        return ROLE_W.get(J.get(link, {}).get("role", ""), 1.0)
    raw = {link: bone_weight(link) for link in geom_links}
    sw = sum(raw.values())
    link_mass = {link: max(0.05, total_mass * raw[link] / sw) for link in geom_links}
    # Welded racks add real mass on top (~3% of body per rack). Antler geometry
    # already extends the link's bbox (so box-inertia captures the moment); this
    # gives that geometry actual mass so the head-heavy balance is faithful.
    for link in welded_links:
        if link in link_mass:
            link_mass[link] += 0.03 * total_mass

    # per-link convex-decomposition budget for the USD-import step. Detail is spent
    # where CONTACTS happen (feet), kept cheap where they don't (body/head).
    HULL_BUDGET = {"base": 4, "spine": 3, "neck": 2, "head": 4,
                   "hind_upper": 2, "hind_lower": 2, "hind_hock": 2, "hind_ankle": 4, "hind_toe": 6,
                   "front_shoulder": 2, "front_upper": 2, "front_lower": 2, "front_ankle": 4, "front_toe": 6}
    def hull_budget(link):
        if link in welded_links: return 8   # head+antler: more hulls for the rack
        if link == "Jaw": return 2
        if "Ear" in link: return 1
        if link == "Tail": return 2
        return HULL_BUDGET.get(J.get(link, {}).get("role", ""), 3)

    def safe(n): return n.replace(" ", "_")

    links_xml = []; joints_xml = []; budget = {}
    for link in geom_links:
        fn = os.path.join(mesh_dir, f"{safe(link)}.stl")
        origin = bone_head[link]
        tris = [[np.asarray(v) - origin for v in face] for face in link_faces[link]]
        write_stl(fn, tris)
        verts = np.array(link_verts[link]) - origin
        mass = link_mass[link]
        pmin = verts.min(0); pmax = verts.max(0)
        sx, sy, sz = np.maximum(pmax - pmin, 0.02)
        ix = mass * (sy * sy + sz * sz) / 12; iy = mass * (sx * sx + sz * sz) / 12
        iz = mass * (sx * sx + sy * sy) / 12
        com = verts.mean(0)
        budget[safe(link)] = hull_budget(link)
        links_xml.append(f'''  <link name="{safe(link)}">
    <inertial><origin xyz="{vec(com)}" rpy="0 0 0"/><mass value="{mass:.4f}"/>
      <inertia ixx="{ix:.5f}" iyy="{iy:.5f}" izz="{iz:.5f}" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry>
      <mesh filename="meshes/{safe(link)}.stl"/></geometry></collision>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry>
      <mesh filename="meshes/{safe(link)}.stl"/></geometry></visual>
  </link>''')

    # tiny intermediate links for 2-DOF joints
    def add_intermediate(name, pos):
        links_xml.append(f'''  <link name="{name}">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
      <inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>''')

    # joints (walk the tree from base)
    used_axes = {}
    dof2_filter_pairs = []   # parent<->child of each 2-DOF joint: two halves of one
    # anatomical joint, co-located, but NOT auto-filtered (a massless intermediate
    # link sits between them so PhysX doesn't see them as adjacent). Their hulls
    # overlap and violently repel unless filtered — the same filtering PhysX does
    # automatically for a 1-DOF joint's two links. Does NOT affect inter-limb collision.
    for bone in joint_bones:
        j = J[bone]
        # find parent link: climb config parents skipping merges to a real link/base
        par = j["parent"]
        while par is not None and par in J and J[par].get("kind") == "merge_geometry":
            par = J[par]["parent"]
        parent_link = par if (par in geom_links or J.get(par, {}).get("kind") == "base_link") else base
        pos = bone_head[bone]
        parent_pos = bone_head.get(parent_link, bone_head[base])
        offset = pos - parent_pos
        ax = axes.get(bone) or {"axis1": [0, 0, 1], "axis2": [1, 0, 0], "lo": -0.5, "hi": 0.5}
        lo, hi = ax["lo"], ax["hi"]
        marg = 0.15
        used_axes[bone] = ax
        if j["dof"] == 1:
            joints_xml.append(f'''  <joint name="{safe(parent_link)}_to_{safe(bone)}" type="revolute">
    <parent link="{safe(parent_link)}"/><child link="{safe(bone)}"/>
    <origin xyz="{vec(offset)}" rpy="0 0 0"/><axis xyz="{vec(ax['axis1'])}"/>
    <limit lower="{lo-marg:.4f}" upper="{hi+marg:.4f}" effort="80" velocity="12"/></joint>''')
        else:
            inter = f"{safe(bone)}_dof0"
            add_intermediate(inter, pos)
            dof2_filter_pairs.append([safe(parent_link), safe(bone)])
            joints_xml.append(f'''  <joint name="{safe(parent_link)}_to_{inter}" type="revolute">
    <parent link="{safe(parent_link)}"/><child link="{inter}"/>
    <origin xyz="{vec(offset)}" rpy="0 0 0"/><axis xyz="{vec(ax['axis1'])}"/>
    <limit lower="{lo-marg:.4f}" upper="{hi+marg:.4f}" effort="80" velocity="12"/></joint>''')
            joints_xml.append(f'''  <joint name="{inter}_to_{safe(bone)}" type="revolute">
    <parent link="{inter}"/><child link="{safe(bone)}"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="{vec(ax['axis2'])}"/>
    <limit lower="-0.7" upper="0.7" effort="60" velocity="12"/></joint>''')

    urdf = f'''<?xml version="1.0"?>
<robot name="{species}_polyart">
{chr(10).join(links_xml)}
{chr(10).join(joints_xml)}
</robot>'''
    urdf_path = os.path.join(outdir, f"{species}_polyart.urdf")
    open(urdf_path, "w").write(urdf)

    # Self-collision filter pairs: DATA-DRIVEN. Detect which link geometries overlap
    # at the rest pose and filter exactly those. Convex hulls bulge past the mesh, so
    # connected body regions (chain links, shoulder<->neck, thigh<->spine at the
    # hindquarters) overlap and would falsely repel under full self-collision.
    # DISTAL inter-limb pairs (L leg vs R leg, foot vs opposite leg) do NOT overlap at
    # rest, so they are NOT filtered -> different limbs still cannot pass through.
    link_world = {l: np.array([v for face in link_faces[l] for v in face]) for l in geom_links}
    gl = list(geom_links)
    overlap = set()
    for i in range(len(gl)):
        A = link_world[gl[i]][::3]
        for j in range(i + 1, len(gl)):
            B = link_world[gl[j]][::3]
            dmin = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
            if dmin < 0.04:   # 4 cm: mesh proxy for hull overlap (hulls bulge ~1-2 cm + contact offset)
                overlap.add((safe(gl[i]), safe(gl[j])))
    for a_, b_ in dof2_filter_pairs:      # guarantee the co-located 2-DOF halves are in
        overlap.add((a_, b_))
    filter_pairs = sorted(overlap)
    import_cfg = {
        "enable_self_collision": True,
        "collision_approximation": "convexDecomposition",
        "hull_budget": budget,
        "filter_pairs": [list(p) for p in filter_pairs],
        "note": "Full self-collision. filter_pairs = link geometries overlapping at rest "
                "(connected body regions + 2-DOF halves). Left/right legs and distal limb-vs-body "
                "pairs do NOT overlap at rest and are NOT filtered, so different limbs still collide.",
    }
    open(os.path.join(outdir, "import_config.json"), "w").write(json.dumps(import_cfg, indent=1))
    # audit
    ndof = sum(J[b]["dof"] for b in joint_bones)
    n_revolute = urdf.count('type="revolute"')
    open(os.path.join(outdir, "build_axes.json"), "w").write(json.dumps(used_axes, indent=1))
    print("WROTE_URDF:" + urdf_path)
    print(f"LINKS:{len(geom_links)} bodies + {sum(1 for b in joint_bones if J[b]['dof']==2)} intermediates")
    print(f"REVOLUTE_JOINTS:{n_revolute}  EXPECTED_DOF:{ndof}")
    print(f"BASE:{base}  no_geom_links:{[b for b in joint_bones if b not in geom_links]}")
    print(f"HULL_BUDGET_TOTAL:{sum(budget.values())} across {len(budget)} links (feet={[budget.get(k) for k in budget if 'Toe' in k or 'Finger' in k]})")


main()
