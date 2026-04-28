# Deer_001_idle.anim → deer_idle.npy conversion report

- Source: `C:\Users\nars7\Projects\aniML-world\aniML-anim-source\Deer_001_idle.anim`
- Frames: 331 at 30.0 fps (11.000 s)
- Forward speed synthesized: **0.0 m/s** along world -Y (matches deer URDF body -Z = anatomical forward via Rx(-90°) spawn rotation)
- Spawn height: 0.93 m

## Per-joint twist amplitude and clamp counts

| URDF joint | max |twist| (°) | URDF limits (°) | clamped frames |
|------------|-----------------:|-----------------|---------------:|
| `pelvis_to_chest` | 14.74 | [-28.6, +28.6] | 0/331 |
| `pelvis_to_HR_thigh` | 13.15 | [-80.2, +80.2] | 0/331 |
| `HR_thigh_to_HR_shin` | 15.75 | [-126.1, +0.0] | 90/331 |
| `HR_shin_to_HR_foot` | 23.03 | [-68.8, +68.8] | 0/331 |
| `pelvis_to_HL_thigh` | 10.63 | [-80.2, +80.2] | 0/331 |
| `HL_thigh_to_HL_shin` | 8.46 | [-126.1, +0.0] | 50/331 |
| `HL_shin_to_HL_foot` | 23.60 | [-68.8, +68.8] | 0/331 |
| `chest_to_FR_thigh` | 34.99 | [-80.2, +80.2] | 0/331 |
| `FR_thigh_to_FR_shin` | 32.59 | [+0.0, +126.1] | 165/331 |
| `FR_shin_to_FR_foot` | 62.01 | [-68.8, +68.8] | 0/331 |
| `chest_to_FL_thigh` | 3.58 | [-80.2, +80.2] | 0/331 |
| `FL_thigh_to_FL_shin` | 7.05 | [+0.0, +126.1] | 275/331 |
| `FL_shin_to_FL_foot` | 21.96 | [-68.8, +68.8] | 0/331 |
| `chest_to_neck` | 20.64 | [-34.4, +45.8] | 0/331 |
| `neck_to_skull` | 0.00 | [-45.8, +45.8] | 0/331 |

## ⚠ Clamp issues

Joints exceeding 5 % clamp rate:
- `HR_thigh_to_HR_shin` — 90/331 frames clamped
- `HL_thigh_to_HL_shin` — 50/331 frames clamped
- `FR_thigh_to_FR_shin` — 165/331 frames clamped
- `FL_thigh_to_FL_shin` — 275/331 frames clamped