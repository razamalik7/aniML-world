# Stage 3 Policy vs Reference deer_walk — Stride Comparison

- **Reference**: `deer_walk.npy` — 31 frames @ 30 fps = 1.03 s, 15 DOF
- **Policy**:    `play_loco_2026-05-05_20-18-54.npz` — 1500 frames @ 50 fps = 30.00 s, 25 DOF (env 0 of 4)

## Root locomotion

| Metric | Reference | Policy |
|---|---|---|
| Mean horizontal speed (m/s) | 0.800 | 0.674 |
| Net-displacement speed (m/s) | 0.800 | 0.076 |
| Total yaw drift | +0.000 rad (+0.0°) | +4.571 rad (+261.9°) |
| Yaw drift per sec | +0.00°/s | +8.74°/s |

## Knee joints (the user-flagged 'knees not bending')

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_thigh_to_FL_shin | ref | 0.477 (27.3°) | 0.329 (18.8°) | 0.105 | 1.94 |
| FL_thigh_to_FL_shin | pol | 0.502 (28.8°) | 0.506 (29.0°) | 0.080 | 2.00 |
| FR_thigh_to_FR_shin | ref | 0.677 (38.8°) | 0.402 (23.0°) | 0.112 | 2.90 |
| FR_thigh_to_FR_shin | pol | 0.426 (24.4°) | 0.435 (24.9°) | 0.079 | 2.00 |
| HL_thigh_to_HL_shin | ref | 0.495 (28.4°) | 0.373 (21.3°) | 0.109 | 1.94 |
| HL_thigh_to_HL_shin | pol | 1.941 (111.2°) | 1.410 (80.8°) | 0.230 | 2.00 |
| HR_thigh_to_HR_shin | ref | 0.500 (28.6°) | 0.500 (28.6°) | 0.134 | 1.94 |
| HR_thigh_to_HR_shin | pol | 1.591 (91.2°) | 0.954 (54.6°) | 0.124 | 4.00 |

**Avg knee peak |q|:** ref 30.8°  vs  policy 63.9°  (policy is 208% of reference)

**Avg knee ROM:**     ref 23.0°  vs  policy 47.3°  (policy is 206% of reference)

## Cannon (shin->foot) joints

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_shin_to_FL_foot | ref | 1.200 (68.8°) | 1.005 (57.6°) | 0.326 | 0.97 |
| FL_shin_to_FL_foot | pol | 0.521 (29.9°) | 0.821 (47.1°) | 0.126 | 2.00 |
| FR_shin_to_FR_foot | ref | 1.200 (68.8°) | 0.804 (46.1°) | 0.317 | 0.97 |
| FR_shin_to_FR_foot | pol | 0.726 (41.6°) | 1.026 (58.8°) | 0.159 | 2.00 |
| HL_shin_to_HL_foot | ref | 0.337 (19.3°) | 0.540 (30.9°) | 0.123 | 1.94 |
| HL_shin_to_HL_foot | pol | 0.832 (47.6°) | 1.132 (64.9°) | 0.104 | 0.10 |
| HR_shin_to_HR_foot | ref | 0.358 (20.5°) | 0.562 (32.2°) | 0.125 | 1.94 |
| HR_shin_to_HR_foot | pol | 0.586 (33.6°) | 0.886 (50.8°) | 0.076 | 0.30 |

## Head pitch (`neck_to_skull`)

| Source | mean | peak\|q\| | ROM |
|---|---|---|---|
| ref | -0.326 rad (-18.7°) | +0.394 rad (+22.6°) | +0.147 rad (+8.4°) |
| pol | +0.141 rad (+8.1°) | +0.383 rad (+21.9°) | +0.653 rad (+37.4°) |

## Front vs hind knee activity (RMS)

| Source | Front-avg RMS | Hind-avg RMS | front/hind ratio |
|---|---|---|---|
| ref | 0.1086 | 0.1212 | 0.90 |
| pol | 0.0793 | 0.1771 | 0.45 |

## L/R asymmetry (hind L vs hind R; front L vs front R)

| Source | hindL/hindR RMS ratio | frontL/frontR RMS ratio |
|---|---|---|
| ref | 0.81 | 0.94 |
| pol | 1.85 | 1.02 |

## Bottom-line interpretation

- **Knee bend depth**: policy reaches 206% of reference ROM. Comparable to reference.
- **Yaw drift**: policy drifts +8.7°/s vs reference +0.0°/s. CONSISTENT WITH user observation of veering.
