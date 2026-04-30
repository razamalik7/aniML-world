# Run 37 — V24 broader command (0.3-1.0) + warm-start r36

**Date:** 2026-04-29
**Best run yet** — sustained walking at up to 0.37 m/s.

## Trajectory
| env | dy | spd_y | z avg |
|---|---|---|---|
| 0 | -11.17m | **0.37 m/s** | 0.89 |
| 1 | -7.12m | 0.24 m/s | 0.91 |
| 2 | -9.80m | 0.33 m/s | 0.87 |
| 3 | -4.73m | 0.16 m/s | 0.86 |

## Progressive broadening worked
| | r35 (locked) | r36 (0.4-0.7) | r37 (0.3-1.0) |
|---|---|---|---|
| best speed | 0.31 m/s | 0.20 m/s | **0.37 m/s** |
| max distance | 9.4m | 5.9m | **11.2m** |
| pelvis_too_low | 36% | 26% | **21%** |

## r38 plan
Add yaw rate commands so deer can turn. Range (-0.3, 0.3) rad/s
to start.
