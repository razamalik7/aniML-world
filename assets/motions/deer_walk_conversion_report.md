# Deer_001_walk.anim → deer_walk.npy conversion report

- Source: `C:\Users\nars7\Projects\aniML-world\aniML-anim-source\Deer_001_walk.anim`
- Frames: 31 at 30.0 fps (1.000 s)
- Forward speed synthesized: **0.8 m/s** along world -Y (matches deer URDF body -Z = anatomical forward via Rx(-90°) spawn rotation)
- Spawn height: 0.93 m

## Per-joint twist amplitude and clamp counts

| URDF joint | max |twist| (°) | URDF limits (°) | clamped frames |
|------------|-----------------:|-----------------|---------------:|
| `pelvis_to_chest` | 2.40 | [-28.6, +28.6] | 0/31 |
| `pelvis_to_HR_thigh` | 37.39 | [-80.2, +80.2] | 0/31 |
| `HR_thigh_to_HR_shin` | 28.65 | [-126.1, +0.0] | 1/31 |
| `HR_shin_to_HR_foot` | 20.49 | [-68.8, +68.8] | 0/31 |
| `pelvis_to_HL_thigh` | 38.61 | [-80.2, +80.2] | 0/31 |
| `HL_thigh_to_HL_shin` | 28.38 | [-126.1, +0.0] | 0/31 |
| `HL_shin_to_HL_foot` | 19.32 | [-68.8, +68.8] | 0/31 |
| `chest_to_FR_thigh` | 15.38 | [-80.2, +80.2] | 0/31 |
| `FR_thigh_to_FR_shin` | 38.77 | [+0.0, +126.1] | 0/31 |
| `FR_shin_to_FR_foot` | 68.75 | [-68.8, +68.8] | 6/31 |
| `chest_to_FL_thigh` | 17.82 | [-80.2, +80.2] | 0/31 |
| `FL_thigh_to_FL_shin` | 27.34 | [+0.0, +126.1] | 0/31 |
| `FL_shin_to_FL_foot` | 68.75 | [-68.8, +68.8] | 2/31 |
| `chest_to_neck` | 14.17 | [-34.4, +45.8] | 0/31 |
| `neck_to_skull` | 22.59 | [-45.8, +45.8] | 0/31 |

## ⚠ Clamp issues

Joints exceeding 5 % clamp rate:
- `FR_shin_to_FR_foot` — 6/31 frames clamped
- `FL_shin_to_FL_foot` — 2/31 frames clamped