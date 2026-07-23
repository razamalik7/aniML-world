# Stage 3 Policy vs Reference deer_walk — Stride Comparison

- **Reference**: `deer_walk.npy` — 31 frames @ 30 fps = 1.03 s, 15 DOF
- **Policy**:    `play_loco_2026-05-05_07-17-20.npz` — 1000 frames @ 50 fps = 20.00 s, 25 DOF (env 0 of 4)

## Root locomotion

| Metric | Reference | Policy |
|---|---|---|
| Mean horizontal speed (m/s) | 0.800 | 1.386 |
| Net-displacement speed (m/s) | 0.800 | 0.000 |
| Total yaw drift | +0.000 rad (+0.0°) | -0.006 rad (-0.4°) |
| Yaw drift per sec | +0.00°/s | -0.02°/s |

## Knee joints (the user-flagged 'knees not bending')

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_thigh_to_FL_shin | ref | 0.477 (27.3°) | 0.329 (18.8°) | 0.105 | 1.94 |
| FL_thigh_to_FL_shin | pol | 0.516 (29.6°) | 0.519 (29.7°) | 0.093 | 2.00 |
| FR_thigh_to_FR_shin | ref | 0.677 (38.8°) | 0.402 (23.0°) | 0.112 | 2.90 |
| FR_thigh_to_FR_shin | pol | 0.479 (27.4°) | 0.482 (27.6°) | 0.085 | 2.00 |
| HL_thigh_to_HL_shin | ref | 0.495 (28.4°) | 0.373 (21.3°) | 0.109 | 1.94 |
| HL_thigh_to_HL_shin | pol | 2.039 (116.8°) | 1.238 (71.0°) | 0.176 | 2.00 |
| HR_thigh_to_HR_shin | ref | 0.500 (28.6°) | 0.500 (28.6°) | 0.134 | 1.94 |
| HR_thigh_to_HR_shin | pol | 1.502 (86.1°) | 0.778 (44.6°) | 0.139 | 4.00 |

**Avg knee peak |q|:** ref 30.8°  vs  policy 65.0°  (policy is 211% of reference)

**Avg knee ROM:**     ref 23.0°  vs  policy 43.2°  (policy is 188% of reference)

## Cannon (shin->foot) joints

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_shin_to_FL_foot | ref | 1.200 (68.8°) | 1.005 (57.6°) | 0.326 | 0.97 |
| FL_shin_to_FL_foot | pol | 0.477 (27.3°) | 0.777 (44.5°) | 0.150 | 2.00 |
| FR_shin_to_FR_foot | ref | 1.200 (68.8°) | 0.804 (46.1°) | 0.317 | 0.97 |
| FR_shin_to_FR_foot | pol | 0.629 (36.0°) | 0.929 (53.2°) | 0.191 | 2.00 |
| HL_shin_to_HL_foot | ref | 0.337 (19.3°) | 0.540 (30.9°) | 0.123 | 1.94 |
| HL_shin_to_HL_foot | pol | 0.643 (36.8°) | 0.943 (54.0°) | 0.075 | 0.05 |
| HR_shin_to_HR_foot | ref | 0.358 (20.5°) | 0.562 (32.2°) | 0.125 | 1.94 |
| HR_shin_to_HR_foot | pol | 0.731 (41.9°) | 1.032 (59.1°) | 0.080 | 0.20 |

## Head pitch (`neck_to_skull`)

| Source | mean | peak\|q\| | ROM |
|---|---|---|---|
| ref | -0.326 rad (-18.7°) | +0.394 rad (+22.6°) | +0.147 rad (+8.4°) |
| pol | -0.050 rad (-2.9°) | +0.233 rad (+13.4°) | +0.430 rad (+24.7°) |

## Front vs hind knee activity (RMS)

| Source | Front-avg RMS | Hind-avg RMS | front/hind ratio |
|---|---|---|---|
| ref | 0.1086 | 0.1212 | 0.90 |
| pol | 0.0891 | 0.1571 | 0.57 |

## L/R asymmetry (hind L vs hind R; front L vs front R)

| Source | hindL/hindR RMS ratio | frontL/frontR RMS ratio |
|---|---|---|
| ref | 0.81 | 0.94 |
| pol | 1.27 | 1.10 |

## Bottom-line interpretation

- **Knee bend depth**: policy reaches 188% of reference ROM. Comparable to reference.
- **Yaw drift**: policy drifts -0.0°/s vs reference +0.0°/s. Roughly straight.
