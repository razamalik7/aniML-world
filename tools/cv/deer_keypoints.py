"""Canonical keypoint schema for the aniML deer, plus mappings to the public
animal-pose vocabularies we might consume pretrained detectors from.

WHY THIS FILE EXISTS
--------------------
The video->motion pipeline needs one fixed answer to "what is a keypoint on our
deer, and where exactly is it?". Everything downstream depends on it:

  * the synthetic renderer auto-labels these points (exactly, for free),
  * a 2D pose net is trained to predict them,
  * the URDF fit minimises reprojection error of these points,
  * PCK against hand-labelled real frames is measured on these points.

Keypoints are anchored to the ARTIST ARMATURE (`Deer_001_rig`, 33 bones) rather
than to the URDF, because:

  1. The artist mesh is what we can actually render (the URDF's "visual" meshes
     are convex-hull physics proxies -- see docs/CV_PIPELINE_DESIGN.md).
  2. The source `.anim` clips are natively in armature space, so we can pose and
     render without first solving the URDF<->armature retarget.

Each keypoint is (bone_name, 'head'|'tail'). Blender gives us the world-space
position of any bone head/tail in the posed armature for free, so the labels are
exact by construction -- no annotation, no annotator bias.

COORDINATE FRAME (verified 2026-07-23 by dumping the rest pose)
---------------------------------------------------------------
In `rig.matrix_world` space the armature uses the SAME anatomical convention as
the deer URDF documented in CLAUDE.md:

    +X = anatomical LEFT     (.L bones at +X, .R bones at -X)
    +Y = anatomical DOWN     (leg chains increase monotonically in Y)
    -Z = anatomical FORWARD  (skull at -Z, tail at +Z)

This is NOT Blender's usual Z-up convention, and it is not Isaac's either. Do
not assume; the render-overlay debug mode in blender_synth.py is the check that
keeps us honest.

NAMING
------
Anatomical where a deer vet would recognise it, generic where the mapping to
public vocabularies matters more than precision:

    hind leg : hip -> knee (stifle) -> hock (tarsus) -> fetlock -> hoof
    fore leg : shoulder -> elbow -> carpus -> fetlock -> hoof

STATUS OF THE PUBLIC-VOCABULARY MAPPINGS
----------------------------------------
Both VERIFIED against primary sources on 2026-07-23.

`AP10K_FROM_CANONICAL` against the AP-10K paper's 17-keypoint definition
(arXiv 2108.12617): 15/17 mapped, the two eyes excluded.

`SUPERANIMAL_TO_CANONICAL` against the ordered `bodyparts:` list in
`DeepLabCut/deeplabcut/modelzoo/project_configs/superanimal_quadruped.yaml`
(the HuggingFace model card does not publish it): 26/39 mapped. That vocabulary
includes four ANTLER keypoints, which matters for deer -- antler tips are
supplied here by `MESH_KEYPOINTS`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

# keypoint name -> (armature bone, which end of the bone)
# Order here IS the canonical index order. Append-only: downstream label files
# and trained checkpoints index into this list positionally.
CANONICAL_KEYPOINTS: list[tuple[str, tuple[str, str]]] = [
    # --- head ---
    ("nose",          ("scull",     "tail")),
    ("skull_base",    ("scull",     "head")),   # poll, between the ears
    ("ear_L_base",    ("ear.L",     "head")),
    ("ear_L_tip",     ("ear.L",     "tail")),
    ("ear_R_base",    ("ear.R",     "head")),
    ("ear_R_tip",     ("ear.R",     "tail")),
    # --- neck / spine / tail ---
    ("neck_upper",    ("spine.008", "head")),
    ("neck_mid",      ("spine.007", "head")),
    ("neck_base",     ("spine.006", "head")),
    ("withers",       ("spine.005", "head")),   # top of shoulder
    ("back_mid",      ("spine.004", "head")),
    ("croup",         ("spine.003", "head")),   # sacrum / top of hip
    ("tail_base",     ("spine.002", "head")),
    ("tail_mid",      ("spine.001", "head")),
    ("tail_tip",      ("spine",     "tail")),
    # --- fore left ---
    ("FL_shoulder",   ("front_shoulder.L", "head")),
    ("FL_elbow",      ("front_thigh.L",    "head")),
    ("FL_carpus",     ("front_shin.L",     "head")),
    ("FL_fetlock",    ("front_foot.L",     "head")),
    ("FL_hoof",       ("front_toe.L",      "tail")),
    # --- fore right ---
    ("FR_shoulder",   ("front_shoulder.R", "head")),
    ("FR_elbow",      ("front_thigh.R",    "head")),
    ("FR_carpus",     ("front_shin.R",     "head")),
    ("FR_fetlock",    ("front_foot.R",     "head")),
    ("FR_hoof",       ("front_toe.R",      "tail")),
    # --- hind left ---
    ("HL_hip",        ("shoulder.L", "head")),
    ("HL_knee",       ("thigh.L",    "head")),
    ("HL_hock",       ("shin.L",     "head")),
    ("HL_fetlock",    ("foot.L",     "head")),
    ("HL_hoof",       ("toe.L",      "tail")),
    # --- hind right ---
    ("HR_hip",        ("shoulder.R", "head")),
    ("HR_knee",       ("thigh.R",    "head")),
    ("HR_hock",       ("shin.R",     "head")),
    ("HR_fetlock",    ("foot.R",     "head")),
    ("HR_hoof",       ("toe.R",      "tail")),
]

KEYPOINT_NAMES: list[str] = [name for name, _ in CANONICAL_KEYPOINTS]
KEYPOINT_ANCHORS: dict[str, tuple[str, str]] = dict(CANONICAL_KEYPOINTS)
KEYPOINT_INDEX: dict[str, int] = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# Bones referenced by the schema; handy for validating a .blend before a run.
REQUIRED_BONES: set[str] = {bone for _, (bone, _) in CANONICAL_KEYPOINTS}


# Skeleton edges, for drawing overlays and for bone-length consistency checks.
# Each entry is a pair of canonical keypoint names whose 3D distance should be
# CONSTANT across every frame of a rigid-bone motion. validate_motion.py uses
# this as a falsification test on extracted motion.
SKELETON_EDGES: list[tuple[str, str]] = [
    ("nose", "skull_base"),
    ("skull_base", "ear_L_base"), ("ear_L_base", "ear_L_tip"),
    ("skull_base", "ear_R_base"), ("ear_R_base", "ear_R_tip"),
    ("skull_base", "neck_upper"),
    ("neck_upper", "neck_mid"), ("neck_mid", "neck_base"),
    ("neck_base", "withers"), ("withers", "back_mid"), ("back_mid", "croup"),
    ("croup", "tail_base"), ("tail_base", "tail_mid"), ("tail_mid", "tail_tip"),
    ("withers", "FL_shoulder"), ("withers", "FR_shoulder"),
    ("FL_shoulder", "FL_elbow"), ("FL_elbow", "FL_carpus"),
    ("FL_carpus", "FL_fetlock"), ("FL_fetlock", "FL_hoof"),
    ("FR_shoulder", "FR_elbow"), ("FR_elbow", "FR_carpus"),
    ("FR_carpus", "FR_fetlock"), ("FR_fetlock", "FR_hoof"),
    ("croup", "HL_hip"), ("croup", "HR_hip"),
    ("HL_hip", "HL_knee"), ("HL_knee", "HL_hock"),
    ("HL_hock", "HL_fetlock"), ("HL_fetlock", "HL_hoof"),
    ("HR_hip", "HR_knee"), ("HR_knee", "HR_hock"),
    ("HR_hock", "HR_fetlock"), ("HR_fetlock", "HR_hoof"),
]

# The four ground-contact points. Contact detection, foot-slide metrics and the
# leg-odometry root-recovery route all key off these.
FOOT_KEYPOINTS: list[str] = ["FL_hoof", "FR_hoof", "HL_hoof", "HR_hoof"]

# Points on the body axis, used to estimate heading and body pitch robustly
# (they move least relative to the trunk).
TRUNK_KEYPOINTS: list[str] = ["withers", "back_mid", "croup", "neck_base"]


# ---------------------------------------------------------------------------
# Public vocabulary mappings
# ---------------------------------------------------------------------------

# AP-10K's 17-keypoint layout. VERIFIED against the AP-10K paper (NeurIPS 2021
# Datasets & Benchmarks, arXiv 2108.12617) keypoint definition.
# Value None = we have no bone-anchored equivalent (see EYE note below).
AP10K_KEYPOINT_ORDER: list[str] = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]

AP10K_FROM_CANONICAL: dict[str, str | None] = {
    # The armature has no eye bones. Eyes must come from mesh geometry (see
    # MESH_DERIVED_KEYPOINTS) or be excluded from PCK. Excluding them is fine:
    # they carry no articulation information for a locomotion reference.
    "L_Eye":        None,
    "R_Eye":        None,
    "Nose":         "nose",
    "Neck":         "neck_base",
    "Root_of_tail": "tail_base",
    "L_Shoulder":   "FL_shoulder",
    "L_Elbow":      "FL_elbow",
    "L_F_Paw":      "FL_hoof",
    "R_Shoulder":   "FR_shoulder",
    "R_Elbow":      "FR_elbow",
    "R_F_Paw":      "FR_hoof",
    "L_Hip":        "HL_hip",
    "L_Knee":       "HL_knee",
    "L_B_Paw":      "HL_hoof",
    "R_Hip":        "HR_hip",
    "R_Knee":       "HR_knee",
    "R_B_Paw":      "HR_hoof",
}

# SuperAnimal-Quadruped (DeepLabCut model zoo, Nature Comms 2024), 39 keypoints.
#
# VERIFIED 2026-07-23 against the primary source: the ordered `bodyparts:` list in
#   DeepLabCut/deeplabcut/modelzoo/project_configs/superanimal_quadruped.yaml
# (the HuggingFace model card does not publish it; the figures repo it defers to
# did not resolve). Order below is the file's order and must not be reordered --
# an off-by-one in keypoint order produces silent garbage, which is exactly the
# failure mode behind the AMP `preserve_order` gotcha in CLAUDE.md.
#
# Licence: Modified MIT, academic/non-commercial; commercial use requires
# contacting the authors. Fine to use for bootstrapping labels; do not vendor the
# weights into a public repo.
#
# NOTE FOR DEER: this vocabulary includes four ANTLER keypoints. The model was
# trained with antlers as labelled structure rather than unlabelled clutter,
# which is better than AP-10K (17 keypoints, no antlers) for our species.
SUPERANIMAL_QUADRUPED_ORDER: list[str] = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend",
    "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend",
    "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end",
    "back_base", "back_end", "back_middle",
    "tail_base", "tail_end",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai", "back_right_thai",
    "back_left_knee", "back_right_knee", "back_right_paw",
    "belly_bottom", "body_middle_right", "body_middle_left",
]

# SuperAnimal slot -> our canonical keypoint, or None where we have no
# bone-anchored equivalent. Used to seed hand-labelling from zero-shot
# predictions (the model card says it is not precise enough to use directly, but
# correcting pre-placed points is much faster than placing them).
#
# "thai" is the upstream spelling of "thigh". SuperAnimal gives 3 points per
# limb where we have 5, so the mapping is deliberately lossy: their per-limb
# "knee" is the visually prominent middle joint, which is the carpus on a foreleg
# and the hock on a hind leg.
SUPERANIMAL_TO_CANONICAL: dict[str, str | None] = {
    "nose": "nose",
    "upper_jaw": None, "lower_jaw": None,
    "mouth_end_right": None, "mouth_end_left": None,
    "right_eye": None, "left_eye": None,        # no eye bones; see MESH_DERIVED
    "right_earbase": "ear_R_base", "right_earend": "ear_R_tip",
    "left_earbase": "ear_L_base", "left_earend": "ear_L_tip",
    # Antler TIPS come from MESH_KEYPOINTS (lateral extremes of the skull vertex
    # group), so SuperAnimal's antler supervision is usable. The antler BASES have
    # no reliable extremum rule and stay unmapped.
    "right_antler_base": None, "left_antler_base": None,   # no bone/extremum
    "right_antler_end": "antler_R_tip", "left_antler_end": "antler_L_tip",
    "neck_base": "neck_base", "neck_end": "neck_upper",
    "throat_base": None, "throat_end": None,
    "back_base": "withers", "back_middle": "back_mid", "back_end": "croup",
    "tail_base": "tail_base", "tail_end": "tail_tip",
    "front_left_thai": "FL_shoulder", "front_left_knee": "FL_carpus",
    "front_left_paw": "FL_hoof",
    "front_right_thai": "FR_shoulder", "front_right_knee": "FR_carpus",
    "front_right_paw": "FR_hoof",
    "back_left_thai": "HL_hip", "back_left_knee": "HL_hock",
    "back_left_paw": "HL_hoof",
    "back_right_thai": "HR_hip", "back_right_knee": "HR_hock",
    "back_right_paw": "HR_hoof",
    "belly_bottom": None,
    "body_middle_right": None, "body_middle_left": None,
}

# ---------------------------------------------------------------------------
# Mesh-anchored keypoints
# ---------------------------------------------------------------------------
# Some landmarks have no bone to hang on. Antler tips are the important case:
# SuperAnimal-Quadruped supervises four antler keypoints, and antlers are the one
# deer-specific feature the animal-pose literature has never studied, so having
# them is worth the extra machinery.
#
# These are resolved ONCE to fixed mesh vertex indices by an extremum rule, then
# read per frame from the EVALUATED (skinned, posed) mesh. That is exact and
# needs no frame algebra: Blender's skinning has already put the vertex where it
# belongs, so `deer.matrix_world @ evaluated_mesh.vertices[i].co` is the answer at
# any pose. Deriving a bone-local offset instead would reintroduce exactly the
# rig-frame confusion this module warns about.
#
# Entries: (keypoint name, vertex group, axis index, sign, min weight)
# axis/sign select the extremum, e.g. (0, +1) = most +X vertex = anatomical LEFT.
# X is shared between the mesh and armature object transforms in this .blend
# (both matrices have row0 = (1,0,0)), so the lateral extremes are unambiguous.
MESH_KEYPOINTS: list[tuple[str, str, int, float, float]] = [
    ("antler_L_tip", "scull", 0, +1.0, 0.5),
    ("antler_R_tip", "scull", 0, -1.0, 0.5),
]

MESH_KEYPOINT_NAMES: list[str] = [k[0] for k in MESH_KEYPOINTS]

# Still unresolved, and only needed for full AP-10K coverage. Eyes carry no
# articulation information for a locomotion reference, so leaving them out costs
# nothing; both are excluded from PCK rather than guessed.
UNRESOLVED_MESH_KEYPOINTS: list[tuple[str, str]] = [
    ("L_Eye", "no automatic extremum rule finds an eye; would need a "
              "hand-picked vertex index captured once in rest pose"),
    ("R_Eye", "mirror of L_Eye across the X=0 plane"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ap10k_indices() -> list[int | None]:
    """Canonical index for each AP-10K slot, or None where unmapped.

    Lets you slice a canonical (T, 35, D) array down to AP-10K order so a
    pretrained AP-10K detector's output can be compared against our labels.
    """
    out: list[int | None] = []
    for slot in AP10K_KEYPOINT_ORDER:
        canon = AP10K_FROM_CANONICAL[slot]
        out.append(None if canon is None else KEYPOINT_INDEX[canon])
    return out


def edge_indices() -> list[tuple[int, int]]:
    """SKELETON_EDGES as canonical index pairs."""
    return [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]) for a, b in SKELETON_EDGES]


def foot_indices() -> list[int]:
    return [KEYPOINT_INDEX[n] for n in FOOT_KEYPOINTS]


def describe() -> str:
    lines = [
        f"aniML deer canonical keypoint schema: {NUM_KEYPOINTS} keypoints, "
        f"{len(SKELETON_EDGES)} skeleton edges",
        "",
        f"{'idx':>3}  {'name':<14} {'bone':<20} end",
        f"{'---':>3}  {'-'*14} {'-'*20} ----",
    ]
    for i, (name, (bone, end)) in enumerate(CANONICAL_KEYPOINTS):
        lines.append(f"{i:>3}  {name:<14} {bone:<20} {end}")
    mapped = sum(1 for v in AP10K_FROM_CANONICAL.values() if v is not None)
    sa_mapped = sum(1 for v in SUPERANIMAL_TO_CANONICAL.values()
                    if v is not None)
    sa_unmapped = [k for k, v in SUPERANIMAL_TO_CANONICAL.items() if v is None]
    lines += [
        "",
        f"AP-10K coverage: {mapped}/{len(AP10K_KEYPOINT_ORDER)} slots mapped "
        f"(unmapped: {[k for k, v in AP10K_FROM_CANONICAL.items() if v is None]})",
        f"SuperAnimal-Quadruped coverage: {sa_mapped}/"
        f"{len(SUPERANIMAL_QUADRUPED_ORDER)} slots mapped",
        f"  unmapped: {sa_unmapped}",
        "  (their order VERIFIED against DeepLabCut's project config; the four "
        "antler slots are unmapped only because our armature has no antler "
        "bones -- the antlers are baked into skull_visual.stl)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
