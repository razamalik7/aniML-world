#!/usr/bin/env python3
"""Blender script: rigged mesh + species config → URDF with auto collision.

Reads the armature's skeleton and the mesh's vertex weights to generate
**per-link convex-hull collision** meshes and a valid **URDF** that Isaac
Sim's URDF importer turns into a PhysX articulation in one click.

Usage (headless — recommended):
  blender --background deer.blend --python rig_to_urdf.py -- \
      --config configs/species/deer.json \
      --output-dir assets/deer_urdf

Usage (open Blender with the file already loaded):
  Run this script from Blender's Text Editor with the args hard-coded
  (see __main__ block at bottom).

The output folder will contain:
  deer.urdf              — ready for Isaac Sim File → Import → URDF
  meshes/*_collision.stl — convex-hull collision per link
  meshes/*_visual.stl    — actual skinned triangles per link (for viewport)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from collections import OrderedDict
from xml.dom import minidom
from xml.etree import ElementTree as ET

import bmesh
import bpy
from mathutils import Matrix, Vector

# ---------------------------------------------------------------------------
# STL writer (no bpy.ops dependency — works headless reliably)
# ---------------------------------------------------------------------------

def _write_stl_binary(filepath: str, verts: list[tuple], faces: list[tuple]) -> None:
    with open(filepath, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            v0, v1, v2 = [verts[i] for i in face[:3]]
            e1 = [v1[j] - v0[j] for j in range(3)]
            e2 = [v2[j] - v0[j] for j in range(3)]
            n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ]
            ln = math.sqrt(sum(x * x for x in n))
            if ln > 1e-12:
                n = [x / ln for x in n]
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


# ---------------------------------------------------------------------------
# Convex hull from a point cloud (uses Blender bmesh)
# ---------------------------------------------------------------------------

def _convex_hull(points: list[Vector]) -> tuple[list[tuple], list[tuple]]:
    """Return (verts, tri_faces) of the convex hull of *points*."""
    if len(points) < 4:
        return [], []

    bm = bmesh.new()
    for p in points:
        bm.verts.new(p)
    bm.verts.ensure_lookup_table()

    try:
        result = bmesh.ops.convex_hull(bm, input=bm.verts[:])
    except Exception:
        bm.free()
        return [], []

    interior = set(result.get("geom_interior", []))
    unused = set(result.get("geom_unused", []))
    to_del = [g for g in (interior | unused) if isinstance(g, bmesh.types.BMVert)]
    if to_del:
        bmesh.ops.delete(bm, geom=to_del, context="VERTS")

    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    vmap = {v: i for i, v in enumerate(bm.verts)}
    verts = [tuple(v.co) for v in bm.verts]
    faces = [tuple(vmap[v] for v in f.verts) for f in bm.faces]
    bm.free()
    return verts, faces


# ---------------------------------------------------------------------------
# Vertex-weight collection
# ---------------------------------------------------------------------------

def _vertex_link_assignment(
    mesh: bpy.types.Mesh,
    mesh_obj: bpy.types.Object,
    link_order: list[str],
    links: OrderedDict[str, dict],
    threshold: float,
) -> list[int]:
    """Assign each mesh vertex to exactly one link index (into *link_order*) or -1.

    The winning link is the one whose bone set gets the highest vertex group
    weight for that vertex (ties: lowest index). If the best weight is below
    *threshold*, the vertex is unassigned (-1).
    """
    vg_index_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    bone_sets = [set(links[name]["bones"]) for name in link_order]
    assignment: list[int] = []
    for v in mesh.vertices:
        scores = [
            max(
                (
                    g.weight
                    for g in v.groups
                    if vg_index_to_name.get(g.group, "") in bones
                ),
                default=0.0,
            )
            for bones in bone_sets
        ]
        bi = max(range(len(scores)), key=lambda i: scores[i])
        if scores[bi] < threshold:
            assignment.append(-1)
        else:
            assignment.append(bi)
    return assignment


def _visual_tris_for_link(
    mesh: bpy.types.Mesh,
    mesh_obj_matrix_world: Matrix,
    link_index: int,
    vertex_assignment: list[int],
    origin_world: Vector,
) -> tuple[list[tuple], list[tuple]]:
    """Build (verts, faces) in link-local frame from mesh triangles owned by *link_index*.

    A triangle is included iff all three vertices are assigned to *link_index*.
    """
    verts: list[tuple] = []
    faces: list[tuple] = []
    for tri in mesh.loop_triangles:
        i0, i1, i2 = tri.vertices
        if (
            vertex_assignment[i0] != link_index
            or vertex_assignment[i1] != link_index
            or vertex_assignment[i2] != link_index
        ):
            continue
        for vi in (i0, i1, i2):
            p_w = mesh_obj_matrix_world @ mesh.vertices[vi].co
            pl = p_w - origin_world
            verts.append((pl.x, pl.y, pl.z))
        n = len(verts)
        faces.append((n - 3, n - 2, n - 1))
    return verts, faces


def _get_weighted_verts(
    mesh_obj: bpy.types.Object,
    bone_names: list[str],
    threshold: float = 0.15,
    exclusive: bool = False,
) -> list[Vector]:
    """Collect world-space vertex positions weighted to *bone_names*.

    *exclusive* (old behaviour): vertex goes to whichever bone has the
    highest weight — produces small, non-overlapping hulls.

    Default (overlap): any vertex where a target bone has weight ≥ threshold
    is included.  Produces larger hulls that fill gaps between links.
    """
    mesh = mesh_obj.data
    vg_index_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    target_names = set(bone_names)
    world = mesh_obj.matrix_world

    result: list[Vector] = []
    for v in mesh.vertices:
        if exclusive:
            best_w = 0.0
            best_name = ""
            for g in v.groups:
                name = vg_index_to_name.get(g.group, "")
                if g.weight > best_w:
                    best_w = g.weight
                    best_name = name
            if best_name in target_names and best_w >= threshold:
                result.append(world @ v.co)
        else:
            for g in v.groups:
                name = vg_index_to_name.get(g.group, "")
                if name in target_names and g.weight >= threshold:
                    result.append(world @ v.co)
                    break
    return result


# ---------------------------------------------------------------------------
# Bone helpers
# ---------------------------------------------------------------------------

def _bone_head_world(arm_obj: bpy.types.Object, bone_name: str) -> Vector:
    bone = arm_obj.data.bones.get(bone_name)
    if bone is None:
        raise ValueError(f"Bone '{bone_name}' not found in armature '{arm_obj.name}'")
    return arm_obj.matrix_world @ bone.head_local


def _bone_auto_axis(arm_obj: bpy.types.Object, bone_name: str) -> Vector:
    """Heuristic joint axis: perpendicular to bone direction and world-up."""
    bone = arm_obj.data.bones.get(bone_name)
    if bone is None:
        return Vector((1, 0, 0))
    bone_dir = (bone.tail_local - bone.head_local).normalized()
    up = Vector((0, 0, 1))
    axis = bone_dir.cross(up)
    if axis.length < 0.01:
        axis = Vector((1, 0, 0))
    return axis.normalized()


# ---------------------------------------------------------------------------
# Inertia from bounding box approximation
# ---------------------------------------------------------------------------

def _bbox_of(pts: list[tuple]) -> tuple[Vector, Vector]:
    if not pts:
        return Vector((0, 0, 0)), Vector((0.01, 0.01, 0.01))
    lo = Vector(pts[0])
    hi = Vector(pts[0])
    for p in pts[1:]:
        for i in range(3):
            lo[i] = min(lo[i], p[i])
            hi[i] = max(hi[i], p[i])
    return lo, hi


def _box_inertia(mass: float, sx: float, sy: float, sz: float):
    ixx = mass / 12.0 * (sy * sy + sz * sz)
    iyy = mass / 12.0 * (sx * sx + sz * sz)
    izz = mass / 12.0 * (sx * sx + sy * sy)
    return ixx, iyy, izz


# ---------------------------------------------------------------------------
# URDF builder
# ---------------------------------------------------------------------------

def _sf(v: float) -> str:
    return f"{v:.6f}"


def _vec3(v) -> str:
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


class UrdfBuilder:
    def __init__(self, robot_name: str):
        self.robot = ET.Element("robot", name=robot_name)

    def add_link(
        self,
        name: str,
        mass: float,
        origin_world: Vector,
        collision_stl: str,
        hull_verts: list[tuple],
        visual_stl: str | None = None,
    ):
        link = ET.SubElement(self.robot, "link", name=name)

        lo, hi = _bbox_of(hull_verts)
        sx, sy, sz = max(hi.x - lo.x, 0.01), max(hi.y - lo.y, 0.01), max(hi.z - lo.z, 0.01)
        cx = (lo.x + hi.x) / 2 - origin_world.x
        cy = (lo.y + hi.y) / 2 - origin_world.y
        cz = (lo.z + hi.z) / 2 - origin_world.z
        ixx, iyy, izz = _box_inertia(mass, sx, sy, sz)

        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=_vec3((cx, cy, cz)), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=_sf(mass))
        ET.SubElement(
            inertial,
            "inertia",
            ixx=_sf(ixx), ixy="0", ixz="0",
            iyy=_sf(iyy), iyz="0", izz=_sf(izz),
        )

        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = ET.SubElement(collision, "geometry")
        ET.SubElement(geom, "mesh", filename=collision_stl)

        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        vgeom = ET.SubElement(visual, "geometry")
        ET.SubElement(vgeom, "mesh", filename=collision_stl)

    def add_joint(
        self,
        name: str,
        joint_type: str,
        parent: str,
        child: str,
        origin_in_parent: Vector,
        axis: tuple,
        lower_rad: float,
        upper_rad: float,
        effort: float,
        velocity: float,
    ):
        joint = ET.SubElement(self.robot, "joint", name=name, type=joint_type)
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz=_vec3(origin_in_parent), rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz=_vec3(axis))
        if joint_type == "revolute":
            ET.SubElement(
                joint,
                "limit",
                lower=_sf(lower_rad),
                upper=_sf(upper_rad),
                effort=_sf(effort),
                velocity=_sf(velocity),
            )

    def write(self, filepath: str):
        rough = ET.tostring(self.robot, encoding="unicode")
        pretty = minidom.parseString(rough).toprettyxml(indent="  ")
        lines = [l for l in pretty.splitlines() if l.strip()]
        lines[0] = '<?xml version="1.0"?>'
        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _verify_bones(arm_obj: bpy.types.Object, cfg: dict) -> list[str]:
    """Check that every bone referenced in the config exists in the armature.
    Returns a list of warning strings (empty = all good).
    """
    available = set(arm_obj.data.bones.keys())
    referenced: set[str] = set()

    # New multi-segment torso format
    for seg in cfg.get("torso", []):
        for b in seg.get("bones", []):
            referenced.add(b)
        jb = seg.get("joint_bone")
        if jb:
            referenced.add(jb)

    # Legacy flat torso format
    for b in cfg.get("torso_bones", []):
        referenced.add(b)

    for _leg, segs in cfg.get("legs", {}).items():
        for _role, bone in segs.items():
            if _role == "parent":
                continue
            referenced.add(bone)

    missing = referenced - available
    warnings: list[str] = []
    if missing:
        warnings.append(f"Bones in config but NOT in armature: {sorted(missing)}")
        warnings.append(f"Available bones: {sorted(available)}")
    return warnings


def run(
    config_path: str,
    output_dir: str,
    dry_run: bool = False,
    export_visual_meshes: bool = True,
):
    with open(config_path) as f:
        cfg = json.load(f)

    export_visual_meshes = export_visual_meshes and bool(cfg.get("export_visual_meshes", True))

    species = cfg["species"]
    mesh_name = cfg["mesh_object"]
    arm_name = cfg["armature_object"]
    total_mass = cfg.get("total_mass_kg", 70.0)
    threshold = cfg.get("weight_threshold", 0.15)
    min_hull = cfg.get("min_hull_vertices", 8)

    mesh_obj = bpy.data.objects.get(mesh_name)
    arm_obj = bpy.data.objects.get(arm_name)
    if mesh_obj is None:
        all_objects = [o.name for o in bpy.data.objects]
        raise RuntimeError(
            f"Mesh object '{mesh_name}' not found in scene. "
            f"Available objects: {all_objects}"
        )
    if arm_obj is None:
        all_objects = [o.name for o in bpy.data.objects]
        raise RuntimeError(
            f"Armature object '{arm_name}' not found in scene. "
            f"Available objects: {all_objects}"
        )

    bone_warnings = _verify_bones(arm_obj, cfg)
    if bone_warnings:
        print("\n=== BONE NAME WARNINGS ===")
        for w in bone_warnings:
            print(f"  {w}")
        print("  Fix the config JSON bone names to match your rig, then re-run.")
        print("  (Continuing anyway — missing bones will produce fallback collision.)\n")

    # Evaluate mesh (applies modifiers like armature in rest pose)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = mesh_eval.data

    # ---- Collect links ----
    # OrderedDict: link_name -> {bones, role, parent_link, joint_cfg, bone_for_joint}
    links: OrderedDict[str, dict] = OrderedDict()

    # Torso — supports both new multi-segment and legacy flat format
    torso_segments = cfg.get("torso")
    if torso_segments:
        base_name = None
        for seg in torso_segments:
            name = seg["name"]
            parent = seg.get("parent")
            is_base = seg.get("is_base", False)
            if is_base:
                base_name = name
            links[name] = {
                "bones": seg["bones"],
                "role": "torso",
                "parent": parent,
                "joint_cfg": seg.get("joint"),
                "joint_bone": seg.get("joint_bone"),
                "origin_bone": seg["bones"][0],
                "mass_kg": seg.get("mass_kg"),
            }
        if base_name is None:
            base_name = torso_segments[0]["name"]
    else:
        # Legacy flat format
        base_name = "base"
        links["base"] = {
            "bones": cfg["torso_bones"],
            "role": "base",
            "parent": None,
            "joint_cfg": None,
            "joint_bone": None,
            "origin_bone": cfg["torso_bones"][0],
        }

    joint_defaults = cfg.get("joint_defaults", {})
    seg_order = ["hip", "thigh", "shin", "foot"]

    for leg_prefix, leg_bones in cfg["legs"].items():
        leg_parent = leg_bones.get("parent", base_name)
        parent = leg_parent
        for seg in seg_order:
            bone_name = leg_bones.get(seg)
            if not bone_name:
                continue
            link_name = f"{leg_prefix}_{seg}"
            jd = joint_defaults.get(seg, {})
            links[link_name] = {
                "bones": [bone_name],
                "role": seg,
                "parent": parent,
                "joint_cfg": jd,
                "joint_bone": bone_name,
                "origin_bone": bone_name,
            }
            parent = link_name

    link_order = list(links.keys())
    vertex_assignment: list[int] | None = None
    if export_visual_meshes and link_order:
        vertex_assignment = _vertex_link_assignment(
            eval_mesh, mesh_obj, link_order, links, threshold
        )

    # ---- Distribute mass ----
    # Links with explicit mass_kg get that value; the rest split the
    # remaining mass proportional to vertex count.
    vert_counts: dict[str, int] = {}
    for name, info in links.items():
        verts = _get_weighted_verts(mesh_eval, info["bones"], threshold)
        vert_counts[name] = max(len(verts), 1)

    masses: dict[str, float] = {}
    fixed_mass = 0.0
    for name, info in links.items():
        m = info.get("mass_kg")
        if m is not None:
            masses[name] = m
            fixed_mass += m

    remaining_mass = max(total_mass - fixed_mass, 1.0)
    auto_links = [n for n in links if n not in masses]
    auto_verts = sum(vert_counts[n] for n in auto_links) or 1
    for name in auto_links:
        masses[name] = max(remaining_mass * vert_counts[name] / auto_verts, 0.1)

    # ---- Dry run ----
    if dry_run:
        print(f"\n=== DRY RUN: {species} ===")
        print(f"Links: {len(links)}")
        for name, info in links.items():
            vn = vert_counts[name]
            m = masses[name]
            print(f"  {name:20s}  bones={info['bones'][:3]}{'...' if len(info['bones'])>3 else ''}  "
                  f"verts={vn}  mass={m:.1f}kg  parent={info['parent']}")
        return

    # ---- Build ----
    os.makedirs(output_dir, exist_ok=True)
    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    urdf = UrdfBuilder(species)

    link_origins: dict[str, Vector] = {}

    for li, (name, info) in enumerate(links.items()):
        verts = _get_weighted_verts(mesh_eval, info["bones"], threshold)
        if len(verts) < min_hull:
            if info["joint_bone"]:
                center = _bone_head_world(arm_obj, info["joint_bone"])
            else:
                center = _bone_head_world(arm_obj, info["bones"][0])
            half = 0.02
            verts = [
                center + Vector((dx, dy, dz))
                for dx in (-half, half)
                for dy in (-half, half)
                for dz in (-half, half)
            ]
            print(f"  [WARN] {name}: too few weighted verts, using fallback cube at bone head")

        if info.get("joint_bone"):
            origin = _bone_head_world(arm_obj, info["joint_bone"])
        elif info.get("origin_bone"):
            origin = _bone_head_world(arm_obj, info["origin_bone"])
        else:
            origin = Vector((0, 0, 0))

        link_origins[name] = origin

        local_verts = [(Vector(v) - origin) for v in verts]
        hull_verts, hull_faces = _convex_hull(local_verts)

        if not hull_faces:
            print(f"  [WARN] {name}: convex hull failed, skipping collision mesh")
            hull_verts = [(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0), (0, 0, 0.01)]
            hull_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

        stl_name = f"{name}_collision.stl"
        stl_path = os.path.join(mesh_dir, stl_name)
        _write_stl_binary(stl_path, hull_verts, hull_faces)

        visual_rel: str | None = None
        if export_visual_meshes and vertex_assignment is not None:
            v_verts, v_faces = _visual_tris_for_link(
                eval_mesh,
                mesh_eval.matrix_world,
                li,
                vertex_assignment,
                origin,
            )
            visual_name = f"{name}_visual.stl"
            visual_path = os.path.join(mesh_dir, visual_name)
            if v_faces:
                _write_stl_binary(visual_path, v_verts, v_faces)
                visual_rel = f"meshes/{visual_name}"
                print(
                    f"  {name}: collision {len(hull_verts)} hull verts / "
                    f"visual {len(v_verts)} verts {len(v_faces)} tris, mass={masses[name]:.1f}kg"
                )
            else:
                print(
                    f"  {name}: {len(hull_verts)} hull verts (no visual tris — "
                    f"using collision mesh for visual), mass={masses[name]:.1f}kg"
                )
        else:
            print(
                f"  {name}: {len(hull_verts)} hull verts, {len(hull_faces)} tris, "
                f"mass={masses[name]:.1f}kg → {stl_name}"
            )

        urdf.add_link(
            name=name,
            mass=masses[name],
            origin_world=origin,
            collision_stl=f"meshes/{stl_name}",
            hull_verts=hull_verts,
            visual_stl=visual_rel,
        )

    # ---- Joints ----
    for name, info in links.items():
        if info["parent"] is None:
            continue
        parent = info["parent"]
        jc = info["joint_cfg"] or {}

        parent_origin = link_origins[parent]
        child_origin = link_origins[name]
        offset = child_origin - parent_origin

        axis_cfg = jc.get("axis")
        if axis_cfg:
            axis = tuple(axis_cfg)
        else:
            axis = tuple(_bone_auto_axis(arm_obj, info["joint_bone"]))

        lower = math.radians(jc.get("lower_deg", -90))
        upper = math.radians(jc.get("upper_deg", 90))
        effort = jc.get("effort", 80)
        velocity = jc.get("velocity", 10)

        urdf.add_joint(
            name=f"{parent}_to_{name}",
            joint_type="revolute",
            parent=parent,
            child=name,
            origin_in_parent=offset,
            axis=axis,
            lower_rad=lower,
            upper_rad=upper,
            effort=effort,
            velocity=velocity,
        )

    urdf_path = os.path.join(output_dir, f"{species}.urdf")
    urdf.write(urdf_path)
    print(f"\n  Wrote {urdf_path}")
    print(f"  Links: {len(links)}, Joints: {len(links) - 1}")
    print(f"  Meshes in: {mesh_dir}")
    print(f"\n  Next: Isaac Sim → File → Import → URDF → pick {urdf_path}")
    print(f"        Set base to 'floating/moveable', import.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Rig → URDF with auto collision")
    p.add_argument("--config", required=True, help="Species JSON config (e.g. deer.json)")
    p.add_argument("--output-dir", required=True, help="Output folder for URDF + meshes")
    p.add_argument("--dry-run", action="store_true", help="Print plan without writing files")
    p.add_argument(
        "--no-visual-meshes",
        action="store_true",
        help="Skip per-link visual STLs (use convex hull for both collision and visual)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.config,
        args.output_dir,
        dry_run=args.dry_run,
        export_visual_meshes=not args.no_visual_meshes,
    )
