# v4.11 contingency plan (pre-staged)

Decision tree based on v4.10 outcome. Pre-staged so v4.11 launches in <5 min after v4.10 render.

## Likely scenarios

### Scenario A: v4.10 fixes HR knee + reduces asymmetries → SUCCESS
Action: extend training (no reward change). Warm-start v4.10 with another 800 iters to consolidate.
Launch: `tools/launch_v4p11_extend.sh` (TBD — copy v4.10 script with new src)

### Scenario B: yaw drift improved but HR still stuck at -50°+
Diagnosis: vanishing gradient. Single-shape (exp) reward gives ~0 signal at large errors.
Action: switch to **Cauchy/Lorentzian reward shape** in `joint_track_reference_walk`:
```python
# Replace:
per_joint = torch.exp(-err_sq / sigma_sq)
# With:
per_joint = 1.0 / (1.0 + err_sq / sigma_sq)
```
Decay is much slower:
| err | exp(-err²/σ²) | 1/(1+err²/σ²) |
|---|---|---|
| 0 | 1.0 | 1.0 |
| σ | 0.37 | 0.50 |
| 2σ | 0.018 | 0.20 |
| 4σ | ~0 | 0.06 |

Even at 1.2 rad error, Cauchy gives 0.06/joint = 0.72 reward across 12 joints. exp gives ~0.

Keep weight 1.0, sigma 0.3.

### Scenario C: v4.10 destabilizes — ep_len drops below 700, base_tilted > 50%
Action: reduce weight back to 0.7 (compromise between v4.9's 0.5 and v4.10's 1.0). Keep yaw_rate_l2 at -3.0.

### Scenario D: front asymmetry worse than ever (FL/FR > 1.5)
Action: add explicit per-foot ROM symmetry penalty. Track running max-min per foot, penalize L/R diff.
Implementation: needs per-env state — harder. Punt to v4.12 if needed.

## Quick assessment criteria

Run after v4.10 completes:
1. `joint_track_reference_walk` reward (compare to v4.9's 3.40 and v4.10 expected ~7-8 with doubled weight)
2. mean reward (compare to v4.9's 265)
3. ep_len (must stay > 950)
4. From the comparison report:
   - HR knee mean (target: trending toward -25°, was -58.6° v4.9)
   - HL/HR knee asym ratio (target: stays near 1.0, v4.9 was 0.977)
   - FL/FR knee asym ratio (target: drops below 1.2, v4.9 was 1.350)
   - yaw drift (target: |drift| < 1°/s, v4.9 was -2.8°/s)
   - net displacement (target: > 0.3 m/s, v4.9 was 0.05)

## Pre-built decision logic

```
if HR_mean > -0.5 rad AND yaw_drift < 1.0:
    SUCCESS → extend training (Scenario A)
elif HR_mean still < -0.8 rad:
    vanishing gradient → Cauchy reward (Scenario B)
elif ep_len < 700 OR base_tilted > 50%:
    instability → reduce weight 1.0 → 0.7 (Scenario C)
elif FL/FR > 1.5:
    asymmetry migration → explicit ROM penalty (Scenario D, may defer to v4.12)
else:
    partial progress → continue same direction with v4.10 model_pt warm-start
```
