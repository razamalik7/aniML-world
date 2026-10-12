# Rig introspection — deer (Malbers Forest rig)

- model: `C:/Users/nars7/My project/Assets/Malbers Animations/Animals Packs/01 Forest Pack/Deer/Models/Poly Art/Deer.FBX`
- anims: Deer Walk.FBX, Deer Trot.FBX, Deer Run.FBX
- armature bones: 34

## Consistency gates
- bones in model but not animated: ['CG']
- bones animated but not in model: none
- UNKNOWN bones (not in canonical map): none

## Per-joint (measured from animation)

purity 1.0 = clean 1-DOF hinge; <0.9 => 2-DOF. clip_agreement = do walk/trot/run agree on the axis (1.0 best). coverage = mesh verts weighted to this bone.

| bone | role | parent | range° | purity | off° | axis | clip_agree | cov | DOF |
|---|---|---|---|---|---|---|---|---|---|
| CG | base_helper | None | 0.0 | 1.0 | 0.0 | [0.0, 0.0, 0.0] | 0.0 | 0 | 1 |
| Pelvis | base | CG | 32.0 | 0.881 | 14.9 | [-0.954, 0.294, 0.064] | 0.795 | 78 | 1 |
| Spine | spine | Pelvis | 39.7 | 0.813 | 18.7 | [-0.294, -0.933, -0.208] | 0.837 | 56 | 2 ⚠️2-DOF |
| Spine1 | spine | Spine | 30.4 | 0.945 | 6.9 | [-0.021, -0.88, 0.474] | 0.755 | 113 | 1 |
| Spine2 | spine | Spine1 | 36.1 | 0.927 | 11.9 | [0.056, -0.883, 0.466] | 0.753 | 85 | 1 |
| Neck | neck | Spine2 | 42.9 | 0.854 | 14.3 | [0.311, -0.631, -0.71] | 0.837 | 140 | 2 ⚠️2-DOF |
| Neck1 | neck | Neck | 56.3 | 0.877 | 18.3 | [-0.245, -0.349, 0.905] | 0.416 | 78 | 2 ⚠️2-DOF |
| Head | head | Neck1 | 28.7 | 0.713 | 17.8 | [-0.398, -0.171, -0.901] | 0.731 | 437 | 2 ⚠️2-DOF |
| L Ear | cosmetic | Head | 37.6 | 0.982 | 8.3 | [0.31, 0.917, 0.251] | 0.846 | 47 | 1 |
| R Ear | cosmetic | Head | 37.2 | 0.981 | 8.3 | [-0.302, -0.947, 0.113] | 0.825 | 47 | 1 |
| Jaw | cosmetic | Head | 0.0 | 1.0 | 0.0 | [0.0, 0.0, 0.0] | 1.0 | 89 | 1 |
| Eye L | cosmetic | Head | 0.0 | 1.0 | 0.0 | [0.0, 0.0, 0.0] | 1.0 | 48 | 1 |
| Eye R | cosmetic | Head | 0.0 | 1.0 | 0.0 | [0.0, 0.0, 0.0] | 1.0 | 48 | 1 |
| L Clavicle | front_shoulder | Spine2 | 44.2 | 0.964 | 14.0 | [-0.003, 0.999, 0.042] | 0.986 | 65 | 1 |
| L UpperArm | front_upper | L Clavicle | 63.0 | 0.689 | 43.6 | [-0.074, 0.073, -0.995] | 0.774 | 31 | 2 ⚠️2-DOF |
| L Forearm | front_lower | L UpperArm | 94.3 | 1.0 | 0.0 | [0.0, 0.0, -1.0] | 1.0 | 41 | 1 |
| L Hand | front_ankle | L Forearm | 110.5 | 0.928 | 36.5 | [-0.002, -0.079, -0.997] | 0.995 | 10 | 1 |
| L Finger0 | front_toe | L Hand | 67.8 | 0.999 | 3.2 | [0.002, 0.003, 1.0] | 1.0 | 23 | 1 |
| R Clavicle | front_shoulder | Spine2 | 54.4 | 0.963 | 13.2 | [0.014, -0.998, -0.054] | 0.997 | 61 | 1 |
| R UpperArm | front_upper | R Clavicle | 62.2 | 0.732 | 46.4 | [0.094, 0.279, -0.956] | 0.698 | 31 | 2 ⚠️2-DOF |
| R Forearm | front_lower | R UpperArm | 92.0 | 1.0 | 0.0 | [0.0, 0.0, -1.0] | 1.0 | 39 | 1 |
| R Hand | front_ankle | R Forearm | 113.0 | 0.868 | 43.4 | [0.182, 0.027, -0.983] | 0.982 | 12 | 2 ⚠️2-DOF |
| R Finger0 | front_toe | R Hand | 77.2 | 0.999 | 2.4 | [0.004, 0.001, 1.0] | 1.0 | 23 | 1 |
| L Thigh | hind_upper | Pelvis | 58.3 | 0.884 | 20.9 | [-0.677, -0.01, 0.736] | 0.936 | 56 | 2 ⚠️2-DOF |
| L Calf | hind_lower | L Thigh | 56.2 | 1.0 | 0.0 | [0.0, 0.0, -1.0] | 1.0 | 41 | 1 |
| L HorseLink | hind_hock | L Calf | 89.8 | 1.0 | 0.0 | [0.0, 0.0, 1.0] | 1.0 | 17 | 1 |
| L Foot | hind_ankle | L HorseLink | 94.3 | 0.985 | 21.6 | [-0.014, 0.08, 0.997] | 0.998 | 19 | 1 |
| L Toe0 | hind_toe | L Foot | 57.8 | 0.966 | 4.5 | [-0.066, -0.04, -0.997] | 0.995 | 17 | 1 |
| R Thigh | hind_upper | Pelvis | 70.1 | 0.804 | 25.0 | [0.26, -0.039, 0.965] | 0.834 | 58 | 2 ⚠️2-DOF |
| R Calf | hind_lower | R Thigh | 73.1 | 1.0 | 0.0 | [0.0, 0.0, -1.0] | 1.0 | 41 | 1 |
| R HorseLink | hind_hock | R Calf | 103.2 | 1.0 | 0.0 | [0.0, 0.0, -1.0] | 1.0 | 17 | 1 |
| R Foot | hind_ankle | R HorseLink | 95.3 | 0.96 | 32.5 | [0.071, -0.011, -0.997] | 0.995 | 18 | 1 |
| R Toe0 | hind_toe | R Foot | 71.3 | 0.971 | 12.6 | [-0.019, -0.034, 0.999] | 0.997 | 18 | 1 |
| Tail | tail | Pelvis | 45.1 | 0.843 | 25.0 | [-0.689, 0.664, 0.292] | 0.769 | 19 | 2 ⚠️2-DOF |

## Meshes

| mesh | verts | skinned | weighted |
|---|---|---|---|
| Antlers 01 | 200 | False | 0 |
| Antlers 02 | 412 | False | 0 |
| Antlers 03 | 262 | False | 0 |
| Antlers 04 | 186 | False | 0 |
| Antlers 05 | 204 | False | 0 |
| Deer 003 | 913 | True | 913 |
| Magic001 | 824 | True | 824 |
| Spots | 186 | True | 186 |

## What a reviewer should check
1. No ⚠️UNKNOWN bones (else extend CANONICAL_ROLES).
2. ⚠️2-DOF joints — confirm which get stacked revolutes (hips, neck, spine-root).
3. ⚠️no-geom actuated links — a bone with 0 weighted verts has no collision.
4. clip_agreement low (<0.8) on a joint => its axis is unstable across gaits; inspect.
5. Consistency gates all empty.