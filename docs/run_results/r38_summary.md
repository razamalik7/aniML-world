# Run 38 — V25 + yaw rate (-0.3, 0.3) + warm-start r37

## Trajectory
| env | dy | spd_y | z avg | dyaw |
|---|---|---|---|---|
| 0 | -9.40 | 0.31 | **0.92** | +5° |
| 1 | -8.65 | 0.29 | 0.88 | -10° |
| 2 | -9.43 | 0.31 | **0.94** | +19° ← turning |
| 3 | -5.35 | 0.18 | 0.93 | +4° |

Yaw integration works. Pelvis z avg highest yet (0.88-0.94). All 4
upright, walking forward, modest turning.

r39: add lateral commands lin_vel_y (-0.2, 0.2).
