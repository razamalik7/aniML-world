"""Dump bind-pose bone rotations from Deer_001.fbx.

Run with:
  "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe" \
    --background --python tools/blender_dump_bind_pose.py

Produces assets/motions/deer_bind_pose.json:

  {
    "unity_axis_forward": "-Z",   // FBX import settings used
    "unity_axis_up":      "Y",
    "bones": {
      "<bone_name>": {
        "parent":          "<parent_name> or null",
        "local_rest_quat": [x, y, z, w],   // parent-relative bind rotation (xyzw)
        "local_rest_head": [x, y, z],      // position of bone head in parent frame
        "world_rest_quat": [x, y, z, w],   // world-space bind rotation (xyzw)
        "world_rest_head": [x, y, z]       // world-space head position
      },
      ...
    }
  }

Used by tools/convert_unity_anim_to_npy.py to do bind-relative delta and
swing-twist extraction per URDF joint.
"""

import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

# Resolve project root relative to this script. When run headless from the
# repo root, __file__ may not be set -- fall back to the first CWD-relative
# guess.
try:
    THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = THIS_DIR.parent
except NameError:
    REPO_ROOT = Path(os.getcwd()).resolve()
    if REPO_ROOT.name != "aniML_world":
        # Try the known path
        alt = Path(r"C:\Users\nars7\Downloads\aniML_world")
        if alt.exists():
            REPO_ROOT = alt

FBX_PATH = Path(
    r"C:\Users\nars7\Downloads\Unity Projects\aniML-world"
    r"\Assets\ithappy\Animals_FREE\Meshes\Deer_001.fbx"
)
OUT_PATH = REPO_ROOT / "assets" / "motions" / "deer_bind_pose.json"


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_fbx(fbx_path: Path) -> bpy.types.Object:
    # Use "default" axis convention -- we will document it and stay consistent.
    # Unity FBX export is Y-up, so keep that.
    bpy.ops.import_scene.fbx(
        filepath=str(fbx_path),
        automatic_bone_orientation=False,  # keep original bone axes as authored
        ignore_leaf_bones=False,
        force_connect_children=False,
        use_custom_normals=True,
        axis_forward="-Z",
        axis_up="Y",
    )
    # Find the armature
    arm = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            arm = obj
            break
    if arm is None:
        raise RuntimeError("No armature found after FBX import")
    return arm


def quat_xyzw(q) -> list[float]:
    # Blender Quaternion is (w, x, y, z); convert to SciPy-style xyzw.
    return [q.x, q.y, q.z, q.w]


def main() -> None:
    clear_scene()
    arm = import_fbx(FBX_PATH)
    print(f"Imported armature: {arm.name} with {len(arm.data.bones)} bones")

    bones_out = {}
    for bone in arm.data.bones:
        # `bone.matrix_local` is the rest (bind) matrix of this bone in
        # armature space. To get the parent-relative local matrix we need
        # parent^-1 * this.
        world_mat: Matrix = bone.matrix_local.copy()
        if bone.parent is not None:
            parent_world_mat = bone.parent.matrix_local
            local_mat = parent_world_mat.inverted() @ world_mat
        else:
            local_mat = world_mat

        loc_quat = local_mat.to_quaternion()
        loc_head = local_mat.to_translation()

        world_quat = world_mat.to_quaternion()
        world_head = world_mat.to_translation()

        bones_out[bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "local_rest_quat": quat_xyzw(loc_quat),
            "local_rest_head": [loc_head.x, loc_head.y, loc_head.z],
            "world_rest_quat": quat_xyzw(world_quat),
            "world_rest_head": [world_head.x, world_head.y, world_head.z],
        }

    payload = {
        "source_fbx": str(FBX_PATH),
        "unity_axis_forward": "-Z",
        "unity_axis_up": "Y",
        "blender_version": bpy.app.version_string,
        "armature_name": arm.name,
        "bones": bones_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH} with {len(bones_out)} bones")

    # Quick sanity print: shoulder bones' parent-relative rest rotations
    for name in ("shoulder.R", "shoulder.L", "front_shoulder.R",
                 "front_shoulder.L", "thigh.R", "spine.003", "Root"):
        if name in bones_out:
            q = bones_out[name]["local_rest_quat"]
            h = bones_out[name]["local_rest_head"]
            print(f"  {name:24s} local_rest_quat=[{q[0]:+.4f}, {q[1]:+.4f}, "
                  f"{q[2]:+.4f}, {q[3]:+.4f}]  head=({h[0]:+.3f},"
                  f"{h[1]:+.3f},{h[2]:+.3f})")


if __name__ == "__main__":
    main()
