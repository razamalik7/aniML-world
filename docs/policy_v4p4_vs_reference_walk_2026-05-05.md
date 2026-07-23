# Stage 3 Policy vs Reference deer_walk — Stride Comparison

- **Reference**: `deer_walk.npy` — 31 frames @ 30 fps = 1.03 s, 15 DOF
- **Policy**:    `play_loco_2026-05-05_18-26-43.npz` — 1500 frames @ 50 fps = 30.00 s, 25 DOF (env 0 of 4)

## Root locomotion

| Metric | Reference | Policy |
|---|---|---|
| Mean horizontal speed (m/s) | 0.800 | 1.192 |
| Net-displacement speed (m/s) | 0.800 | 0.193 |
| Total yaw drift | +0.000 rad (+0.0°) | +2.372 rad (+135.9°) |
| Yaw drift per sec | +0.00°/s | +4.53°/s |

## Knee joints (the user-flagged 'knees not bending')

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_thigh_to_FL_shin | ref | 0.477 (27.3°) | 0.329 (18.8°) | 0.105 | 1.94 |
| FL_thigh_to_FL_shin | pol | 0.571 (32.7°) | 0.585 (33.5°) | 0.104 | 2.00 |
| FR_thigh_to_FR_shin | ref | 0.677 (38.8°) | 0.402 (23.0°) | 0.112 | 2.90 |
| FR_thigh_to_FR_shin | pol | 0.556 (31.8°) | 0.557 (31.9°) | 0.077 | 2.00 |
| HL_thigh_to_HL_shin | ref | 0.495 (28.4°) | 0.373 (21.3°) | 0.109 | 1.94 |
| HL_thigh_to_HL_shin | pol | 2.002 (114.7°) | 1.424 (81.6°) | 0.215 | 2.00 |
| HR_thigh_to_HR_shin | ref | 0.500 (28.6°) | 0.500 (28.6°) | 0.134 | 1.94 |
| HR_thigh_to_HR_shin | pol | 1.649 (94.5°) | 0.921 (52.8°) | 0.145 | 4.00 |

**Avg knee peak |q|:** ref 30.8°  vs  policy 68.4°  (policy is 222% of reference)

**Avg knee ROM:**     ref 23.0°  vs  policy 49.9°  (policy is 217% of reference)

## Cannon (shin->foot) joints

| Joint | Source | peak\|q\| | ROM | RMS | dominant Hz |
|---|---|---|---|---|---|
| FL_shin_to_FL_foot | ref | 1.200 (68.8°) | 1.005 (57.6°) | 0.326 | 0.97 |
| FL_shin_to_FL_foot | pol | 0.484 (27.7°) | 0.783 (44.9°) | 0.178 | 2.00 |
| FR_shin_to_FR_foot | ref | 1.200 (68.8°) | 0.804 (46.1°) | 0.317 | 0.97 |
| FR_shin_to_FR_foot | pol | 0.695 (39.8°) | 0.996 (57.1°) | 0.207 | 2.00 |
| HL_shin_to_HL_foot | ref | 0.337 (19.3°) | 0.540 (30.9°) | 0.123 | 1.94 |
| HL_shin_to_HL_foot | pol | 0.543 (31.1°) | 0.844 (48.3°) | 0.074 | 0.10 |
| HR_shin_to_HR_foot | ref | 0.358 (20.5°) | 0.562 (32.2°) | 0.125 | 1.94 |
| HR_shin_to_HR_foot | pol | 0.753 (43.1°) | 1.053 (60.4°) | 0.094 | 0.40 |

## Head pitch (`neck_to_skull`)

| Source | mean | peak\|q\| | ROM |
|---|---|---|---|
| ref | -0.326 rad (-18.7°) | +0.394 rad (+22.6°) | +0.147 rad (+8.4°) |
| pol | -0.027 rad (-1.5°) | +0.284 rad (+16.3°) | +0.541 rad (+31.0°) |

## Front vs hind knee activity (RMS)

| Source | Front-avg RMS | Hind-avg RMS | front/hind ratio |
|---|---|---|---|
| ref | 0.1086 | 0.1212 | 0.90 |
| pol | 0.0905 | 0.1800 | 0.50 |

## L/R asymmetry (hind L vs hind R; front L vs front R)

| Source | hindL/hindR RMS ratio | frontL/frontR RMS ratio |
|---|---|---|
| ref | 0.81 | 0.94 |
| pol | 1.49 | 1.35 |

## Bottom-line interpretation

- **Knee bend depth**: policy reaches 217% of reference ROM. Comparable to reference.
- **Yaw drift**: policy drifts +4.5°/s vs reference +0.0°/s. Roughly straight.
