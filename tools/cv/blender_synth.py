"""Synthetic deer dataset generator -- renders the ARTIST mesh from randomised
viewpoints and auto-labels exact 2D/3D keypoints and bone rotations.

Runs inside Blender (needs `bpy`), headless, CPU-only:

    blender -b assets/blender/deer_standing.blend \
        --python tools/cv/blender_synth.py -- \
        --out <dir> --num 200 [--seed 0] [--debug-overlay] ...

WHY BLENDER AND NOT THE EXISTING MATPLOTLIB RENDERER
----------------------------------------------------
`tools/render_playback_meshed.py` renders the URDF's *visual* STLs. Those are
convex-hull physics proxies -- 1,693 triangles for the whole animal, no ears, no
tail, no toes, no antlers, and a bulged silhouette. Training a 2D pose network
on that would be training it on the wrong shape.

The artist mesh in `assets/blender/deer_standing.blend` (`Deer_001`, 1,238 polys,
skinned to the 33-bone `Deer_001_rig`) has the correct deer silhouette. It is the
only asset we have that a pose network could plausibly transfer from. See
docs/CV_PIPELINE_DESIGN.md.

CPU-ONLY IS DELIBERATE
----------------------
The GPU is running RL training (~13.6 of 16.3 GB). Cycles is pinned to CPU and
EEVEE is never used, since EEVEE would allocate GPU memory. Do not "optimise"
this by switching to GPU without checking `nvidia-smi` first.

OUTPUT LAYOUT
-------------
    <out>/images/deer_000000.png
    <out>/labels/deer_000000.json
    <out>/dataset.json          # schema, config, per-sample index
    <out>/overlays/...          # only with --debug-overlay

Each label JSON:
    {
      "image": "images/deer_000000.png",
      "width": 640, "height": 480,
      "keypoint_names": [...],                 # canonical order
      "keypoints_2d": [[x, y], ...],           # pixels, origin top-left
      "visibility":   [2|1|0, ...],            # 2=visible, 1=in-frame but
                                               #   occluded, 0=outside frame
      "keypoints_3d": [[x, y, z], ...],        # armature world frame
      "camera": {"K": [[..]], "R_wc": [[..]], "t_wc": [..], "focal_mm":,
                 "sensor_mm":},
      "pose": {"bone_quats_wxyz": {bone: [w,x,y,z]}, "source": "...",
               "clip_frame": int|null}
    }

The `pose` block is what makes this trainable for direct pose regression: the
label carries the exact articulation that produced the image.

VERIFY BEFORE YOU TRUST
-----------------------
Run with `--debug-overlay --num 8` and LOOK at `<out>/overlays/`. If keypoints
do not sit on the animal, the frame convention is wrong and every downstream
number is garbage. This project has been burned three separate times by metrics
that looked healthy while the geometry was wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy  # type: ignore
import mathutils  # type: ignore
from mathutils import Euler, Matrix, Quaternion, Vector  # type: ignore

# Make `tools/cv` importable when Blender runs this file by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deer_keypoints import (  # noqa: E402
    CANONICAL_KEYPOINTS,
    KEYPOINT_NAMES,
    MESH_KEYPOINTS,
    MESH_KEYPOINT_NAMES,
    REQUIRED_BONES,
    SKELETON_EDGES,
)

MESH_OBJ = "Deer_001"
RIG_OBJ = "Deer_001_rig"


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def purge_cameras_and_lights() -> None:
    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def make_camera(name: str = "SynthCam"):
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def make_sun(name: str = "SynthSun"):
    light_data = bpy.data.lights.new(name, type="SUN")
    light = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light)
    return light


def configure_render(width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    # GPU is occupied by RL training. Never switch this to GPU blindly.
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False


# ---------------------------------------------------------------------------
# Randomised appearance
# ---------------------------------------------------------------------------

def build_random_material(rng: random.Random, randomize_texture: bool):
    """A fresh randomised material for the deer body.

    Mu et al. (CVPR 2020) randomise the ANIMAL's texture, not just the
    background. The stated effect is that the network stops relying on
    appearance and keys on shape/silhouette instead -- which is the property
    that makes a crude-surface mesh potentially usable. Our mesh's original
    texture file is missing (the source Unity project folder is gone), so
    randomised procedural texture is not merely the published recipe here, it
    is the only option.
    """
    mat = bpy.data.materials.new("SynthDeerMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    base = (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95),
            rng.uniform(0.05, 0.95), 1.0)
    bsdf.inputs["Base Color"].default_value = base
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = rng.uniform(0.25, 1.0)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = rng.uniform(0.0, 0.15)

    if randomize_texture:
        tex_kind = rng.choice(["noise", "voronoi", "musgrave", "wave", "flat"])
        if tex_kind != "flat":
            coord = nt.nodes.new("ShaderNodeTexCoord")
            mapping = nt.nodes.new("ShaderNodeMapping")
            mapping.inputs["Scale"].default_value = (
                rng.uniform(0.5, 12.0), rng.uniform(0.5, 12.0),
                rng.uniform(0.5, 12.0))
            nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

            node_type = {
                "noise": "ShaderNodeTexNoise",
                "voronoi": "ShaderNodeTexVoronoi",
                "musgrave": "ShaderNodeTexNoise",  # Musgrave folded into Noise
                "wave": "ShaderNodeTexWave",
            }[tex_kind]
            tex = nt.nodes.new(node_type)
            nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
            if "Scale" in tex.inputs:
                tex.inputs["Scale"].default_value = rng.uniform(1.0, 20.0)

            ramp = nt.nodes.new("ShaderNodeValToRGB")
            for _ in range(2):
                pass
            ramp.color_ramp.elements[0].color = (
                rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), 1)
            ramp.color_ramp.elements[1].color = (
                rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), 1)
            out_socket = "Fac" if "Fac" in tex.outputs else "Color"
            nt.links.new(tex.outputs[out_socket], ramp.inputs["Fac"])
            nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def set_world_background(rng: random.Random, bg_images: list[Path]) -> None:
    """Randomise the world background.

    Published recipe (Mu et al.) composites the animal over random COCO
    val2017 images. Pass a directory of natural images via --backgrounds to
    follow it. Without one we fall back to a random-colour/gradient world,
    which is materially weaker: uniform backgrounds let a network cheat on
    "blob against flat colour" and will overstate synthetic-only accuracy.
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("SynthWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Strength"].default_value = rng.uniform(0.3, 2.0)

    if bg_images:
        path = rng.choice(bg_images)
        img = bpy.data.images.load(str(path), check_existing=True)
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = img
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (
            0.0, 0.0, rng.uniform(0.0, 2 * math.pi))
        coord = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (
            rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), 1.0)


def randomize_lighting(sun, rng: random.Random) -> None:
    sun.data.energy = rng.uniform(1.0, 7.0)
    sun.data.angle = rng.uniform(0.005, 0.3)   # soft/hard shadow
    sun.rotation_euler = Euler((
        math.radians(rng.uniform(5.0, 85.0)),
        0.0,
        rng.uniform(0.0, 2 * math.pi),
    ))
    if hasattr(sun.data, "color"):
        warmth = rng.uniform(0.0, 0.25)
        sun.data.color = (1.0, 1.0 - warmth * 0.4, 1.0 - warmth)


# ---------------------------------------------------------------------------
# Pose sampling
# ---------------------------------------------------------------------------

def collect_pose_bones(rig):
    return {pb.name: pb for pb in rig.pose.bones}


def reset_pose(rig) -> None:
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
        pb.location = Vector((0, 0, 0))
        pb.scale = Vector((1, 1, 1))


def bake_current_pose(rig) -> dict[str, Quaternion]:
    """Snapshot pose-bone rotations, then detach the action.

    Needed because we want to perturb a clip frame. While an action is bound,
    the depsgraph re-evaluates bone channels from the F-curves and silently
    discards any jitter we write. Snapshot -> detach -> write back makes the
    pose ours to modify.
    """
    snapshot: dict[str, Quaternion] = {}
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        snapshot[pb.name] = pb.rotation_quaternion.copy()
    if rig.animation_data is not None:
        rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.rotation_quaternion = snapshot[pb.name]
    return snapshot


def jitter_pose(rig, rng: random.Random, max_deg: float) -> None:
    """Compose a small random rotation onto every bone's current rotation.

    Pose diversity is the cheapest axis of variation we control. Applied on top
    of a real clip frame this widens the pose distribution without wandering
    into configurations a deer cannot adopt -- which is what unbounded jitter
    around the rest pose produces.
    """
    if max_deg <= 0.0:
        return
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        axis = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
        if axis.length < 1e-8:
            continue
        axis.normalize()
        angle = math.radians(rng.gauss(0.0, max_deg / 2.0))
        angle = max(-math.radians(max_deg), min(math.radians(max_deg), angle))
        pb.rotation_quaternion = pb.rotation_quaternion @ Quaternion(axis,
                                                                     angle)


def list_actions(rig, only: list[str] | None = None):
    """Animation clips available to pose from.

    `assets/blender/deer_standing.blend` ships the three ithappy Unity clips
    already imported as Blender Actions (idle / walk / run). Posing from these
    is strictly better than random jitter around the rest pose: the rest pose
    is an authoring T-pose-alike that no live deer ever adopts, and a pose
    network trained on it would be learning a distribution that does not occur
    in the footage we intend to run on.
    """
    actions = list(bpy.data.actions)
    if only:
        wanted = [w.lower() for w in only]
        actions = [a for a in actions
                   if any(w in a.name.lower() for w in wanted)]
    return actions


def action_frame_range(action) -> tuple[int, int]:
    lo, hi = action.frame_range
    return int(math.floor(lo)), int(math.ceil(hi))


def apply_clip_frame(rig, action, frame: int) -> None:
    """Set the armature to a frame of a Blender action."""
    if action is None:
        return
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = action
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# Camera sampling and projection
# ---------------------------------------------------------------------------

def subject_bounds(rig) -> tuple[Vector, Vector]:
    """Bounds of the posed SKELETON, in the same frame as the keypoints.

    Deliberately not the mesh bounds. The mesh object and the armature object
    carry different object-level transforms in this .blend, so mesh-derived
    bounds live in a different frame than `rig.matrix_world @ bone.head`, which
    is what the keypoints and the camera projection use. Framing the camera on
    mesh bounds put the animal off-centre and at odd apparent orientations.
    Using bone bounds keeps camera, keypoints and projection in one frame.
    """
    mw = rig.matrix_world
    pts: list[Vector] = []
    for pb in rig.pose.bones:
        pts.append(mw @ pb.head)
        pts.append(mw @ pb.tail)
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts),
                 min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts),
                 max(p.z for p in pts)))
    return mn, mx


def ground_level(rig) -> float:
    """World Y of the lowest hoof. +Y is anatomical DOWN, so this is a max."""
    mw = rig.matrix_world
    ys = []
    for bone in ("toe.L", "toe.R", "front_toe.L", "front_toe.R"):
        pb = rig.pose.bones.get(bone)
        if pb is not None:
            ys.append((mw @ pb.tail).y)
    return max(ys) if ys else 0.0


def make_ground(rig, rng: random.Random, size: float):
    """A ground plane under the hooves.

    Every real photograph of a deer contains ground, usually meeting the
    animal at the hooves. Rendering the subject floating in a void removes the
    single most reliable contextual cue a pose network can use, and makes
    foot keypoints sit against empty background instead of against a contact
    surface -- which is exactly where we most need accuracy, since foot
    contact drives the root-trajectory recovery downstream.
    """
    y = ground_level(rig)
    mesh = bpy.data.meshes.new("GroundMesh")
    obj = bpy.data.objects.new("Ground", mesh)
    bpy.context.scene.collection.objects.link(obj)
    s = size
    verts = [(-s, 0.0, -s), (s, 0.0, -s), (s, 0.0, s), (-s, 0.0, s)]
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    obj.location = Vector((0.0, y, 0.0))

    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (
            rng.uniform(0.05, 0.6), rng.uniform(0.05, 0.5),
            rng.uniform(0.03, 0.35), 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = rng.uniform(0.6, 1.0)
    obj.data.materials.append(mat)
    return obj


def place_camera(cam, target: Vector, radius: float, rng: random.Random,
                 elev_range: tuple[float, float],
                 focal_range: tuple[float, float],
                 roll_deg: float, width: int, height: int,
                 fill_range: tuple[float, float] = (0.45, 0.92)) -> None:
    """Sample a camera on a sphere around the subject.

    Azimuth is uniform over the full circle. Elevation is restricted to a band
    that reflects how animals are actually filmed -- a wildlife camera is
    roughly at or slightly above the animal, essentially never below ground and
    rarely straight down. Training on impossible viewpoints spends capacity on
    poses the real distribution never contains.

    Distance is DERIVED from the focal length so the animal fills a controlled
    fraction of the frame, rather than sampled independently. Sampling them
    independently couples "how much of the deer is in frame" to "which lens" --
    long lenses silently crop the animal to a torso. Deriving distance keeps
    focal length free to vary (telephoto compression is a real and important
    property of wildlife footage) while framing stays consistent.
    """
    azim = rng.uniform(0.0, 2 * math.pi)
    elev = math.radians(rng.uniform(*elev_range))

    focal = rng.uniform(*focal_range)
    cam.data.lens = focal
    cam.data.sensor_width = 36.0
    sensor_w = 36.0
    sensor_h = sensor_w * height / width
    half_fov = min(math.atan(sensor_w / (2.0 * focal)),
                   math.atan(sensor_h / (2.0 * focal)))
    fill = rng.uniform(*fill_range)
    dist = radius / (fill * math.sin(half_fov))
    dist = max(dist, radius * 1.2)

    # +Y is anatomical DOWN in this rig, so "above the animal" is -Y.
    horiz = dist * math.cos(elev)
    cam.location = target + Vector((
        horiz * math.cos(azim),
        -dist * math.sin(elev),
        horiz * math.sin(azim),
    ))
    look = (target - cam.location).normalized()
    # Aim -Z at the target, keeping world -Y (anatomical UP) as the camera's
    # up vector, so the horizon is level and the deer stands upright in frame.
    up = Vector((0.0, -1.0, 0.0))
    right = look.cross(up)
    if right.length < 1e-6:                     # looking straight up/down
        right = look.cross(Vector((0.0, 0.0, 1.0)))
    right.normalize()
    true_up = right.cross(look).normalized()
    rot = Matrix((right, true_up, -look)).transposed().to_4x4()
    cam.rotation_euler = rot.to_euler()
    if roll_deg > 0.0:
        roll = math.radians(rng.uniform(-roll_deg, roll_deg))
        cam.rotation_euler.rotate_axis("Z", roll)


def camera_intrinsics(cam, width: int, height: int) -> list[list[float]]:
    """Pinhole K for the current camera and render resolution."""
    sensor_w = cam.data.sensor_width
    sensor_h = sensor_w * height / width
    if cam.data.sensor_fit == "VERTICAL":
        sensor_h = cam.data.sensor_height
        sensor_w = sensor_h * width / height
    fx = cam.data.lens * width / sensor_w
    fy = cam.data.lens * height / sensor_h
    cx = width / 2.0 + cam.data.shift_x * width
    cy = height / 2.0 - cam.data.shift_y * width
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def project_point(cam, scene, world_pt: Vector, width: int,
                  height: int) -> tuple[float, float, float]:
    """World point -> (pixel_x, pixel_y, depth). Origin top-left.

    Uses Blender's own projection so it cannot drift from what was rendered.
    """
    from bpy_extras.object_utils import world_to_camera_view  # type: ignore
    ndc = world_to_camera_view(scene, cam, world_pt)
    px = ndc.x * width
    py = (1.0 - ndc.y) * height   # Blender NDC y is bottom-up
    return px, py, ndc.z


def keypoint_visibility(cam, mesh_obj, world_pt: Vector, px: float, py: float,
                        depth: float, width: int, height: int,
                        eps: float = 0.03) -> int:
    """2 = visible, 1 = in frame but self-occluded, 0 = out of frame / behind.

    Joint centres sit INSIDE the mesh volume, so "did the camera ray hit the
    mesh before reaching the keypoint" reports *every* keypoint occluded. The
    question we actually want answered is whether some OTHER body part lies in
    front of the limb this joint belongs to.

    So we measure how deep the joint sits below its own skin -- the distance to
    the nearest mesh surface -- and only call it occluded when the first ray
    hit is closer than that by more than `eps`. A fixed body-radius constant
    cannot work here: the chest is ~0.3 m thick while a fetlock is ~0.03 m, and
    any single constant mislabels one end or the other.
    """
    if depth <= 0.0 or not (0 <= px < width and 0 <= py < height):
        return 0
    origin = cam.matrix_world.translation
    direction = (world_pt - origin)
    dist = direction.length
    if dist < 1e-6:
        return 2
    direction = direction / dist

    dg = bpy.context.evaluated_depsgraph_get()
    mw = mesh_obj.matrix_world
    mw_inv = mw.inverted()
    local_origin = mw_inv @ origin
    local_target = mw_inv @ world_pt
    local_dir = (local_target - local_origin).normalized()

    hit, loc, _nrm, _idx = mesh_obj.ray_cast(local_origin, local_dir,
                                             depsgraph=dg)
    if not hit:
        return 2
    hit_dist = ((mw @ loc) - origin).length

    # How far below its own skin does this joint sit?
    ok, surf, _n, _i = mesh_obj.closest_point_on_mesh(local_target,
                                                      depsgraph=dg)
    skin_depth = ((mw @ surf) - world_pt).length if ok else 0.05

    if hit_dist < dist - skin_depth - eps:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Keypoint extraction
# ---------------------------------------------------------------------------

def bone_world_points(rig) -> dict[str, Vector]:
    """Canonical keypoint name -> world position for the CURRENT pose."""
    mw = rig.matrix_world
    pose_bones = {pb.name: pb for pb in rig.pose.bones}
    out: dict[str, Vector] = {}
    for name, (bone, end) in CANONICAL_KEYPOINTS:
        pb = pose_bones.get(bone)
        if pb is None:
            raise KeyError(
                f"keypoint {name!r} needs bone {bone!r} which is absent from "
                f"the armature. Present bones: {sorted(pose_bones)}")
        out[name] = mw @ (pb.head if end == "head" else pb.tail)
    return out


def resolve_mesh_keypoints(mesh_obj) -> dict[str, int]:
    """Pick the fixed vertex index for each mesh-anchored keypoint, once.

    Resolved at REST by an extremum rule over a vertex group (e.g. the most +X
    vertex weighted to the skull = the left antler tip). The index is then fixed
    for the whole run, and the position is read per frame from the evaluated
    mesh -- so skinning does the posing and we never do frame algebra.
    """
    groups = {g.name: g.index for g in mesh_obj.vertex_groups}
    out: dict[str, int] = {}
    for name, group, axis, sign, min_w in MESH_KEYPOINTS:
        gi = groups.get(group)
        if gi is None:
            print(f"[synth] WARNING: vertex group {group!r} absent; "
                  f"mesh keypoint {name!r} skipped")
            continue
        best_i, best_v = None, -1e18
        for v in mesh_obj.data.vertices:
            w = 0.0
            for g in v.groups:
                if g.group == gi:
                    w = g.weight
                    break
            if w < min_w:
                continue
            val = sign * v.co[axis]
            if val > best_v:
                best_i, best_v = v.index, val
        if best_i is None:
            print(f"[synth] WARNING: no vertex in {group!r} above weight "
                  f"{min_w}; mesh keypoint {name!r} skipped")
            continue
        out[name] = best_i
    return out


def mesh_keypoint_positions(mesh_obj, indices: dict[str, int]
                            ) -> dict[str, Vector]:
    """World positions of the mesh-anchored keypoints at the CURRENT pose."""
    if not indices:
        return {}
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh_obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = mesh_obj.matrix_world
    out = {name: mw @ me.vertices[i].co.copy() for name, i in indices.items()}
    ev.to_mesh_clear()
    return out


def bone_rotations(rig) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for pb in rig.pose.bones:
        q = pb.rotation_quaternion if pb.rotation_mode == "QUATERNION" \
            else pb.matrix_basis.to_quaternion()
        out[pb.name] = [q.w, q.x, q.y, q.z]
    return out


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def draw_overlay(image_path: Path, out_path: Path, kp2d: list[list[float]],
                 vis: list[int], names: list[str]) -> bool:
    """Draw keypoints + skeleton on the render. Returns False if unavailable.

    This is the verification step. Numbers agreeing with themselves proves
    nothing; a picture with dots on the animal's actual joints does.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        return False

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    idx = {n: i for i, n in enumerate(names)}
    for a, b in SKELETON_EDGES:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue
        if vis[ia] == 0 or vis[ib] == 0:
            continue
        draw.line([tuple(kp2d[ia]), tuple(kp2d[ib])], fill=(40, 90, 220),
                  width=2)
    for i, (x, y) in enumerate(kp2d):
        if vis[i] == 0:
            continue
        colour = (30, 220, 90) if vis[i] == 2 else (240, 170, 40)
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour,
                     outline=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(
        prog="blender_synth.py",
        description="Render randomised, auto-labelled deer images.")
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--num", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--samples", type=int, default=24,
                   help="Cycles samples. 16-32 is plenty for pose training; "
                        "noise is a form of augmentation.")
    p.add_argument("--backgrounds", type=str, default=None,
                   help="Directory of natural images used as world "
                        "backgrounds. The published recipe uses COCO "
                        "val2017. Strongly recommended -- see docstring.")
    p.add_argument("--pose-jitter-deg", type=float, default=4.0,
                   help="Per-bone random rotation composed onto the clip "
                        "pose. Keep small when posing from clips.")
    p.add_argument("--clips", type=str, nargs="*", default=None,
                   help="Substrings selecting which actions to pose from "
                        "(e.g. walk run). Default: all actions in the file.")
    p.add_argument("--no-clips", action="store_true",
                   help="Ignore actions and pose from rest+jitter. Not "
                        "recommended -- see --clips.")
    p.add_argument("--no-ground", action="store_true",
                   help="Omit the ground plane. Not recommended.")
    p.add_argument("--elev-min", type=float, default=-10.0)
    p.add_argument("--elev-max", type=float, default=55.0)
    p.add_argument("--focal-min", type=float, default=28.0)
    p.add_argument("--focal-max", type=float, default=85.0)
    p.add_argument("--roll-deg", type=float, default=6.0)
    p.add_argument("--fill-min", type=float, default=0.45,
                   help="Smallest fraction of the frame's short side the "
                        "animal's bounding sphere should span.")
    p.add_argument("--fill-max", type=float, default=0.92)
    p.add_argument("--no-texture-randomization", action="store_true")
    p.add_argument("--debug-overlay", action="store_true",
                   help="Also write overlays/ with keypoints drawn on. "
                        "USE THIS on a small --num before any real run.")
    p.add_argument("--start-index", type=int, default=0)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args(list(sys.argv))
    rng = random.Random(args.seed)

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    mesh_obj = bpy.data.objects.get(MESH_OBJ)
    rig = bpy.data.objects.get(RIG_OBJ)
    if mesh_obj is None or rig is None:
        raise SystemExit(
            f"expected objects {MESH_OBJ!r} and {RIG_OBJ!r} in the .blend; "
            f"found {[o.name for o in bpy.data.objects]}")

    have = {b.name for b in rig.data.bones}
    missing = REQUIRED_BONES - have
    if missing:
        raise SystemExit(
            f"armature is missing bones required by the keypoint schema: "
            f"{sorted(missing)}")

    bg_images: list[Path] = []
    if args.backgrounds:
        bg_dir = Path(args.backgrounds)
        exts = {".jpg", ".jpeg", ".png", ".hdr", ".exr"}
        bg_images = sorted(p for p in bg_dir.rglob("*")
                           if p.suffix.lower() in exts)
        print(f"[synth] {len(bg_images)} background images from {bg_dir}")
    if not bg_images:
        print("[synth] WARNING: no background images. Falling back to flat "
              "random-colour worlds. Synthetic-only accuracy measured on "
              "this data will be OPTIMISTIC -- a network can cheat on "
              "'silhouette against flat colour'. Pass --backgrounds.")

    purge_cameras_and_lights()
    cam = make_camera()
    sun = make_sun()
    configure_render(args.width, args.height, args.samples)
    scene = bpy.context.scene

    actions = [] if args.no_clips else list_actions(rig, args.clips)
    if actions:
        print(f"[synth] posing from {len(actions)} clip(s): "
              f"{[a.name for a in actions]}")
    else:
        print("[synth] WARNING: posing from the REST pose + jitter only. The "
              "rest pose is an authoring pose no live deer adopts; a network "
              "trained on it learns the wrong pose distribution. The shipped "
              ".blend contains idle/walk/run actions -- prefer those.")

    mesh_kp_idx = resolve_mesh_keypoints(mesh_obj)
    all_kp_names = KEYPOINT_NAMES + [n for n in MESH_KEYPOINT_NAMES
                                     if n in mesh_kp_idx]
    if mesh_kp_idx:
        print(f"[synth] mesh-anchored keypoints resolved: {mesh_kp_idx}")

    mn, mx = subject_bounds(rig)
    print(f"[synth] skeleton bbox min={tuple(round(v,3) for v in mn)} "
          f"max={tuple(round(v,3) for v in mx)}")

    ground = None
    index: list[dict] = []
    for i in range(args.num):
        n = args.start_index + i
        stem = f"deer_{n:06d}"

        reset_pose(rig)
        pose_source = "rest"
        clip_frame = None
        if actions:
            action = rng.choice(actions)
            lo, hi = action_frame_range(action)
            clip_frame = rng.randint(lo, hi)
            apply_clip_frame(rig, action, clip_frame)
            bake_current_pose(rig)
            pose_source = action.name
        jitter_pose(rig, rng, args.pose_jitter_deg)
        bpy.context.view_layer.update()

        if ground is not None:
            bpy.data.objects.remove(ground, do_unlink=True)
            ground = None
        if not args.no_ground:
            # Large enough that the plane's own edge never enters frame at the
            # longest camera distance the focal/fill sampling can produce.
            ground = make_ground(rig, rng, size=120.0)

        mat = build_random_material(rng, not args.no_texture_randomization)
        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = mat
        else:
            mesh_obj.data.materials.append(mat)

        set_world_background(rng, bg_images)
        randomize_lighting(sun, rng)

        mn_i, mx_i = subject_bounds(rig)
        centre_i = (mn_i + mx_i) / 2.0
        radius_i = (mx_i - mn_i).length / 2.0     # bounding-sphere radius
        place_camera(cam, centre_i, radius_i, rng,
                     (args.elev_min, args.elev_max),
                     (args.focal_min, args.focal_max), args.roll_deg,
                     args.width, args.height,
                     (args.fill_min, args.fill_max))
        bpy.context.view_layer.update()

        img_rel = f"images/{stem}.png"
        img_path = out_dir / img_rel
        scene.render.filepath = str(img_path)
        bpy.ops.render.render(write_still=True)

        world_kp = bone_world_points(rig)
        world_kp.update(mesh_keypoint_positions(mesh_obj, mesh_kp_idx))
        kp2d: list[list[float]] = []
        kp3d: list[list[float]] = []
        vis: list[int] = []
        for name in all_kp_names:
            wp = world_kp[name]
            px, py, depth = project_point(cam, scene, wp, args.width,
                                          args.height)
            v = keypoint_visibility(cam, mesh_obj, wp, px, py, depth,
                                    args.width, args.height)
            kp2d.append([round(px, 3), round(py, 3)])
            kp3d.append([round(wp.x, 5), round(wp.y, 5), round(wp.z, 5)])
            vis.append(v)

        R_wc = [list(row) for row in cam.matrix_world.inverted().to_3x3()]
        t_wc = list(cam.matrix_world.inverted().translation)
        label = {
            "image": img_rel,
            "width": args.width,
            "height": args.height,
            "keypoint_names": all_kp_names,
            "keypoints_2d": kp2d,
            "visibility": vis,
            "keypoints_3d": kp3d,
            "camera": {
                "K": camera_intrinsics(cam, args.width, args.height),
                "R_wc": R_wc,
                "t_wc": [round(v, 6) for v in t_wc],
                "focal_mm": cam.data.lens,
                "sensor_mm": cam.data.sensor_width,
                "location": [round(v, 6) for v in cam.matrix_world.translation],
            },
            "pose": {
                "bone_quats_wxyz": bone_rotations(rig),
                "source": pose_source,
                "jitter_deg": args.pose_jitter_deg,
                "clip_frame": clip_frame,
            },
        }
        (out_dir / "labels" / f"{stem}.json").write_text(
            json.dumps(label), encoding="utf-8")

        if args.debug_overlay:
            ok = draw_overlay(img_path, out_dir / "overlays" / f"{stem}.png",
                              kp2d, vis, all_kp_names)
            if not ok and i == 0:
                print("[synth] NOTE: Pillow unavailable inside Blender; "
                      "overlays skipped. Run tools/cv/overlay_keypoints.py "
                      "afterwards with the system Python instead.")

        n_vis = sum(1 for v in vis if v == 2)
        index.append({"stem": stem, "image": img_rel,
                      "label": f"labels/{stem}.json", "num_visible": n_vis})
        print(f"[synth] {i+1}/{args.num} {stem} visible={n_vis}/"
              f"{len(all_kp_names)}", flush=True)

    (out_dir / "dataset.json").write_text(json.dumps({
        "schema_version": 1,
        "keypoint_names": all_kp_names,
        "skeleton_edges": SKELETON_EDGES,
        "mesh_keypoints": mesh_kp_idx,
        "num_samples": len(index),
        "config": vars(args),
        "source_blend": bpy.data.filepath,
        "source_mesh": MESH_OBJ,
        "source_rig": RIG_OBJ,
        "renderer": "cycles-cpu",
        "samples": index,
    }, indent=1), encoding="utf-8")
    print(f"[synth] wrote {len(index)} samples to {out_dir}")


if __name__ == "__main__":
    main()
