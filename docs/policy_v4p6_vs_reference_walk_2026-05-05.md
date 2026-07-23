# Stage 3 Policy vs Reference deer_walk — Stride Comparison

- **Reference**: `deer_walk.npy` — 31 frames @ 30 fps = 1.03 s, 15 DOF
- **Policy**:    `play_loco_2026-05-05_23-10-29.npz` — 1500 frames @ 50 fps = 30.00 s, 25 DOF (env 0 of 4)

## Root locomotion

| Metric | Reference | Policy |
|---|---|---|
| Mean horizontal speed (m/s) | 0.800 | 0.970 |
| Net-displacement speed (m/s) | 0.800 | 0.160 |
| Total yaw drift | +0.000 rad (+0.0°) | -0.436 rad (-25.0°) |
| Yaw drift per sec | +0.00°/s | -0.83°/s |

## Knee joints (the user-flagged 'knees not bending')

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_thigh_to_FL_shin | ref | 0.477 (27.3°) | 0.329 (18.8°) | 0.105 | 1.94 |
| FL_thigh_to_FL_shin | pol | 0.496 (28.4°) | 0.512 (29.3°) | 0.090 | 2.00 |
| FR_thigh_to_FR_shin | ref | 0.677 (38.8°) | 0.402 (23.0°) | 0.112 | 2.90 |
| FR_thigh_to_FR_shin | pol | 0.499 (28.6°) | 0.500 (28.7°) | 0.068 | 2.00 |
| HL_thigh_to_HL_shin | ref | 0.495 (28.4°) | 0.373 (21.3°) | 0.109 | 1.94 |
| HL_thigh_to_HL_shin | pol | 1.837 (105.3°) | 1.147 (65.7°) | 0.134 | 2.00 |
| HR_thigh_to_HR_shin | ref | 0.500 (28.6°) | 0.500 (28.6°) | 0.134 | 1.94 |
| HR_thigh_to_HR_shin | pol | 1.593 (91.3°) | 0.871 (49.9°) | 0.120 | 4.00 |

**Avg knee peak |q|:** ref 30.8°  vs  policy 63.4°  (policy is 206% of reference)

**Avg knee ROM:**     ref 23.0°  vs  policy 43.4°  (policy is 189% of reference)

## Cannon (shin->foot) joints

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_shin_to_FL_foot | ref | 1.200 (68.8°) | 1.005 (57.6°) | 0.326 | 0.97 |
| FL_shin_to_FL_foot | pol | 0.563 (32.3°) | 0.863 (49.4°) | 0.153 | 2.00 |
| FR_shin_to_FR_foot | ref | 1.200 (68.8°) | 0.804 (46.1°) | 0.317 | 0.97 |
| FR_shin_to_FR_foot | pol | 0.774 (44.4°) | 1.075 (61.6°) | 0.164 | 2.00 |
| HL_shin_to_HL_foot | ref | 0.337 (19.3°) | 0.540 (30.9°) | 0.123 | 1.94 |
| HL_shin_to_HL_foot | pol | 0.860 (49.2°) | 1.160 (66.4°) | 0.100 | 0.10 |
| HR_shin_to_HR_foot | ref | 0.358 (20.5°) | 0.562 (32.2°) | 0.125 | 1.94 |
| HR_shin_to_HR_foot | pol | 0.743 (42.6°) | 1.044 (59.8°) | 0.088 | 0.20 |

## Head pitch (`neck_to_skull`)

| Source | mean | peak\|q\| | ROM |
|---|---|---|---|
| ref | -0.326 rad (-18.7°) | +0.394 rad (+22.6°) | +0.147 rad (+8.4°) |
| pol | +0.018 rad (+1.0°) | +0.352 rad (+20.2°) | +0.648 rad (+37.1°) |

## Front vs hind knee activity (RMS)

| Source | Front-avg RMS | Hind-avg RMS | front/hind ratio |
|---|---|---|---|
| ref | 0.1086 | 0.1212 | 0.90 |
| pol | 0.0791 | 0.1271 | 0.62 |

## L/R asymmetry (hind L vs hind R; front L vs front R)

| Source | hindL/hindR RMS ratio | frontL/frontR RMS ratio |
|---|---|---|
| ref | 0.81 | 0.94 |
| pol | 1.11 | 1.32 |

## Bottom-line interpretation

- **Knee bend depth**: policy reaches 189% of reference ROM. Comparable to reference.
- **Yaw drift**: policy drifts -0.8°/s vs reference +0.0°/s. Roughly straight.
