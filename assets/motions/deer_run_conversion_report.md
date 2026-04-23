# Deer_001_run.anim → deer_run.npy conversion report

- Source: `C:\Users\nars7\Downloads\Unity Projects\aniML-world\Assets\ithappy\Animals_FREE\Animations\Other_Animations\Deer_001_run.anim`
- Frames: 25 at 30.0 fps (0.800 s)
- Forward speed synthesized: **2.5 m/s** along Isaac +X
- Spawn height: 0.93 m

## Per-joint twist amplitude and clamp counts

| URDF joint | max |twist| (°) | URDF limits (°) | clamped frames |
|------------|-----------------:|-----------------|---------------:|
| `pelvis_to_chest` | 11.46 | [-11.5, +11.5] | 2/25 |
| `pelvis_to_HR_thigh` | 45.84 | [-45.8, +45.8] | 8/25 |
| `HR_thigh_to_HR_shin` | 62.18 | [-103.1, +0.0] | 6/25 |
| `HR_shin_to_HR_foot` | 28.65 | [-28.6, +28.6] | 13/25 |
| `pelvis_to_HL_thigh` | 45.84 | [-45.8, +45.8] | 8/25 |
| `HL_thigh_to_HL_shin` | 53.78 | [-103.1, +0.0] | 4/25 |
| `HL_shin_to_HL_foot` | 28.65 | [-28.6, +28.6] | 14/25 |
| `chest_to_FR_thigh` | 45.84 | [-45.8, +45.8] | 10/25 |
| `FR_thigh_to_FR_shin` | 82.08 | [+0.0, +103.1] | 2/25 |
| `FR_shin_to_FR_foot` | 28.65 | [-28.6, +28.6] | 20/25 |
| `chest_to_FL_thigh` | 45.84 | [-45.8, +45.8] | 13/25 |
| `FL_thigh_to_FL_shin` | 95.40 | [+0.0, +103.1] | 0/25 |
| `FL_shin_to_FL_foot` | 28.65 | [-28.6, +28.6] | 18/25 |
| `chest_to_neck` | 28.65 | [-17.2, +28.6] | 9/25 |
| `neck_to_skull` | 21.57 | [-22.9, +34.4] | 0/25 |

## ⚠ Clamp issues

Joints exceeding 5 % clamp rate:
- `pelvis_to_chest` — 2/25 frames clamped
- `pelvis_to_HR_thigh` — 8/25 frames clamped
- `HR_thigh_to_HR_shin` — 6/25 frames clamped
- `HR_shin_to_HR_foot` — 13/25 frames clamped
- `pelvis_to_HL_thigh` — 8/25 frames clamped
- `HL_thigh_to_HL_shin` — 4/25 frames clamped
- `HL_shin_to_HL_foot` — 14/25 frames clamped
- `chest_to_FR_thigh` — 10/25 frames clamped
- `FR_thigh_to_FR_shin` — 2/25 frames clamped
- `FR_shin_to_FR_foot` — 20/25 frames clamped
- `chest_to_FL_thigh` — 13/25 frames clamped
- `FL_shin_to_FL_foot` — 18/25 frames clamped
- `chest_to_neck` — 9/25 frames clamped