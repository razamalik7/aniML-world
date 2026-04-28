# Deer_001_run.anim → deer_run.npy conversion report

- Source: `C:\Users\nars7\Projects\aniML-world\aniML-anim-source\Deer_001_run.anim`
- Frames: 25 at 30.0 fps (0.800 s)
- Forward speed synthesized: **2.5 m/s** along world -Y (matches deer URDF body -Z = anatomical forward via Rx(-90°) spawn rotation)
- Spawn height: 0.93 m

## Per-joint twist amplitude and clamp counts

| URDF joint | max |twist| (°) | URDF limits (°) | clamped frames |
|------------|-----------------:|-----------------|---------------:|
| `pelvis_to_chest` | 11.46 | [-11.5, +11.5] | 2/25 |
| `pelvis_to_HR_thigh` | 52.63 | [-80.2, +80.2] | 0/25 |
| `HR_thigh_to_HR_shin` | 62.18 | [-126.1, +0.0] | 6/25 |
| `HR_shin_to_HR_foot` | 68.75 | [-68.8, +68.8] | 4/25 |
| `pelvis_to_HL_thigh` | 63.69 | [-80.2, +80.2] | 0/25 |
| `HL_thigh_to_HL_shin` | 53.78 | [-126.1, +0.0] | 4/25 |
| `HL_shin_to_HL_foot` | 51.82 | [-68.8, +68.8] | 0/25 |
| `chest_to_FR_thigh` | 80.21 | [-80.2, +80.2] | 3/25 |
| `FR_thigh_to_FR_shin` | 82.08 | [+0.0, +126.1] | 2/25 |
| `FR_shin_to_FR_foot` | 68.75 | [-68.8, +68.8] | 13/25 |
| `chest_to_FL_thigh` | 80.21 | [-80.2, +80.2] | 4/25 |
| `FL_thigh_to_FL_shin` | 95.40 | [+0.0, +126.1] | 0/25 |
| `FL_shin_to_FL_foot` | 68.75 | [-68.8, +68.8] | 13/25 |
| `chest_to_neck` | 28.65 | [-17.2, +28.6] | 9/25 |
| `neck_to_skull` | 21.57 | [-22.9, +34.4] | 0/25 |

## ⚠ Clamp issues

Joints exceeding 5 % clamp rate:
- `pelvis_to_chest` — 2/25 frames clamped
- `HR_thigh_to_HR_shin` — 6/25 frames clamped
- `HR_shin_to_HR_foot` — 4/25 frames clamped
- `HL_thigh_to_HL_shin` — 4/25 frames clamped
- `chest_to_FR_thigh` — 3/25 frames clamped
- `FR_thigh_to_FR_shin` — 2/25 frames clamped
- `FR_shin_to_FR_foot` — 13/25 frames clamped
- `chest_to_FL_thigh` — 4/25 frames clamped
- `FL_shin_to_FL_foot` — 13/25 frames clamped
- `chest_to_neck` — 9/25 frames clamped