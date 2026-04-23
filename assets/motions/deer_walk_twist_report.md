# Walk cycle — twist vs swing amplitude per leg bone

Delta computed as `q_rest^-1 * q_anim(t)` with rest = walk.anim frame 0.

| Bone | # keys | twist range (°) | swing max (°) | swing/twist ratio |
|------|--------|-----------------|---------------|-------------------|
| `shoulder.R` | 11 | [-16.7, +31.8] (range 48.5) | 3.6 | 0.08 (OK) |
| `shin.R` | 11 | [-18.6, +9.7] (range 28.3) | 4.9 | 0.17 (OK) |
| `foot.R` | 11 | [-16.6, +15.6] (range 32.2) | 3.4 | 0.11 (OK) |
| `shoulder.L` | 10 | [-12.4, +25.5] (range 37.9) | 3.7 | 0.10 (OK) |
| `shin.L` | 12 | [-13.7, +7.0] (range 20.7) | 4.2 | 0.20 (OK) |
| `foot.L` | 10 | [-30.9, +0.0] (range 30.9) | 6.2 | 0.20 (OK) |
| `front_shoulder.R (composed)` | 12 | [-28.3, +0.0] (range 28.3) | 19.8 | 0.70 (BAD) |
| `front_shin.R` | 11 | [+0.0, +22.0] (range 22.0) | 3.4 | 0.15 (OK) |
| `front_foot.R` | 13 | [-55.5, +0.0] (range 55.5) | 14.2 | 0.26 (OK) |
| `front_shoulder.L` | 8 | [-1.9, +28.7] (range 30.6) | 5.3 | 0.17 (OK) |
| `front_shin.L` | 9 | [+0.0, +18.8] (range 18.8) | 1.7 | 0.09 (OK) |
| `front_foot.L` | 14 | [-54.4, +3.9] (range 58.3) | 8.3 | 0.14 (OK) |

**Summary: 11 OK (ratio < 0.3), 1 marginal/bad.**

Decision rule:
- ratio < 0.3: single-X URDF joint fully captures this bone's walk motion.
- 0.3–0.7: significant but maybe tolerable swing — visualize before deciding.
- > 0.7: URDF joint axis is wrong OR the rig has genuine off-axis motion that
  can't be represented by an X-only joint. Either extend the URDF or redefine
  the joint axis per bone.