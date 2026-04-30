# Run 36 — V23 variable command (0.4-0.7) + warm-start r35

**Date:** 2026-04-29
**Task:** AniML-Deer-USD-Walk-AMP-v27
**Duration:** 80 min, 2000 iters

## Trajectory
| env | dy | spd_y | z avg |
|---|---|---|---|
| 0 | -1.29 | 0.04 | 0.89 |
| 1 | -5.90 | **0.20** | **0.91** |
| 2 | -4.46 | 0.15 | **0.93** |
| 3 | -1.35 | 0.04 | 0.93 |

**Pelvis posture better than r35** (z avg 0.89-0.93 vs 0.84-0.86).
Variable command tolerated. Speeds 0.04-0.20 m/s.

| | r35 | **r36** |
|---|---|---|
| ep_len | 796 | 889 |
| pelvis_too_low | 36.4% | 25.9% |
| reward | 95 | 104 |

## r37 plan
Broaden command range further: (0.3, 1.0). Warm-start r36.
