# Deer_001_walk.anim → deer_walk.npy conversion report

- Source: `C:\Users\nars7\Downloads\Unity Projects\aniML-world\Assets\ithappy\Animals_FREE\Animations\Other_Animations\Deer_001_walk.anim`
- Frames: 31 at 30.0 fps (1.000 s)
- Forward speed synthesized: **0.8 m/s** along Isaac +X
- Spawn height: 0.93 m

## Per-joint twist amplitude and clamp counts

| URDF joint | max |twist| (°) | URDF limits (°) | clamped frames |
|------------|-----------------:|-----------------|---------------:|
| `pelvis_to_chest` | 2.40 | [-11.5, +11.5] | 0/31 |
| `pelvis_to_HR_thigh` | 37.39 | [-45.8, +45.8] | 0/31 |
| `HR_thigh_to_HR_shin` | 28.65 | [-103.1, +0.0] | 1/31 |
| `HR_shin_to_HR_foot` | 20.49 | [-28.6, +28.6] | 0/31 |
| `pelvis_to_HL_thigh` | 38.61 | [-45.8, +45.8] | 0/31 |
| `HL_thigh_to_HL_shin` | 28.38 | [-103.1, +0.0] | 0/31 |
| `HL_shin_to_HL_foot` | 19.32 | [-28.6, +28.6] | 0/31 |
| `chest_to_FR_thigh` | 15.38 | [-45.8, +45.8] | 0/31 |
| `FR_thigh_to_FR_shin` | 38.77 | [+0.0, +103.1] | 0/31 |
| `FR_shin_to_FR_foot` | 28.65 | [-28.6, +28.6] | 22/31 |
| `chest_to_FL_thigh` | 17.82 | [-45.8, +45.8] | 0/31 |
| `FL_thigh_to_FL_shin` | 27.34 | [+0.0, +103.1] | 0/31 |
| `FL_shin_to_FL_foot` | 28.65 | [-28.6, +28.6] | 11/31 |
| `chest_to_neck` | 14.17 | [-17.2, +28.6] | 0/31 |
| `neck_to_skull` | 22.59 | [-22.9, +34.4] | 0/31 |

## ⚠ Clamp issues

Joints exceeding 5 % clamp rate:
- `FR_shin_to_FR_foot` — 22/31 frames clamped
- `FL_shin_to_FL_foot` — 11/31 frames clamped