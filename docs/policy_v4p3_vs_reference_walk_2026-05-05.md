# Stage 3 Policy vs Reference deer_walk — Stride Comparison

- **Reference**: `deer_walk.npy` — 31 frames @ 30 fps = 1.03 s, 15 DOF
- **Policy**:    `play_loco_2026-05-05_10-52-48.npz` — 1500 frames @ 50 fps = 30.00 s, 25 DOF (env 0 of 4)

## Root locomotion

| Metric | Reference | Policy |
|---|---|---|
| Mean horizontal speed (m/s) | 0.800 | 0.629 |
| Net-displacement speed (m/s) | 0.800 | 0.201 |
| Total yaw drift | +0.000 rad (+0.0°) | -6.943 rad (-397.8°) |
| Yaw drift per sec | +0.00°/s | -13.27°/s |

## Knee joints (the user-flagged 'knees not bending')

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_thigh_to_FL_shin | ref | 0.477 (27.3°) | 0.329 (18.8°) | 0.105 | 1.94 |
| FL_thigh_to_FL_shin | pol | 0.536 (30.7°) | 0.539 (30.9°) | 0.086 | 2.00 |
| FR_thigh_to_FR_shin | ref | 0.677 (38.8°) | 0.402 (23.0°) | 0.112 | 2.90 |
| FR_thigh_to_FR_shin | pol | 0.543 (31.1°) | 0.543 (31.1°) | 0.099 | 2.00 |
| HL_thigh_to_HL_shin | ref | 0.495 (28.4°) | 0.373 (21.3°) | 0.109 | 1.94 |
| HL_thigh_to_HL_shin | pol | 2.002 (114.7°) | 1.355 (77.6°) | 0.158 | 2.00 |
| HR_thigh_to_HR_shin | ref | 0.500 (28.6°) | 0.500 (28.6°) | 0.134 | 1.94 |
| HR_thigh_to_HR_shin | pol | 1.492 (85.5°) | 0.766 (43.9°) | 0.116 | 4.00 |

**Avg knee peak |q|:** ref 30.8°  vs  policy 65.5°  (policy is 213% of reference)

**Avg knee ROM:**     ref 23.0°  vs  policy 45.9°  (policy is 200% of reference)

## Cannon (shin->foot) joints

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_shin_to_FL_foot | ref | 1.200 (68.8°) | 1.005 (57.6°) | 0.326 | 0.97 |
| FL_shin_to_FL_foot | pol | 0.519 (29.7°) | 0.819 (46.9°) | 0.135 | 2.00 |
| FR_shin_to_FR_foot | ref | 1.200 (68.8°) | 0.804 (46.1°) | 0.317 | 0.97 |
| FR_shin_to_FR_foot | pol | 0.654 (37.5°) | 0.954 (54.7°) | 0.137 | 2.00 |
| HL_shin_to_HL_foot | ref | 0.337 (19.3°) | 0.540 (30.9°) | 0.123 | 1.94 |
| HL_shin_to_HL_foot | pol | 0.684 (39.2°) | 0.984 (56.4°) | 0.088 | 0.10 |
| HR_shin_to_HR_foot | ref | 0.358 (20.5°) | 0.562 (32.2°) | 0.125 | 1.94 |
| HR_shin_to_HR_foot | pol | 0.802 (46.0°) | 1.102 (63.2°) | 0.093 | 0.30 |

## Head pitch (`neck_to_skull`)

| Source | mean | peak\|q\| | ROM |
|---|---|---|---|
| ref | -0.326 rad (-18.7°) | +0.394 rad (+22.6°) | +0.147 rad (+8.4°) |
| pol | -0.042 rad (-2.4°) | +0.220 rad (+12.6°) | +0.429 rad (+24.6°) |

## Front vs hind knee activity (RMS)

| Source | Front-avg RMS | Hind-avg RMS | front/hind ratio |
|---|---|---|---|
| ref | 0.1086 | 0.1212 | 0.90 |
| pol | 0.0922 | 0.1367 | 0.67 |

## L/R asymmetry (hind L vs hind R; front L vs front R)

| Source | hindL/hindR RMS ratio | frontL/frontR RMS ratio |
|---|---|---|
| ref | 0.81 | 0.94 |
| pol | 1.36 | 0.87 |

## Bottom-line interpretation

- **Knee bend depth**: policy reaches 200% of reference ROM. Comparable to reference.
- **Yaw drift**: policy drifts -13.3°/s vs reference +0.0°/s. CONSISTENT WITH user observation of veering.
