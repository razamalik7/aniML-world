# The trot and run reference clips are 1.25x oversized (found 2026-08-10)

## What is wrong

`extract_reference.py` reads `position_scale` from **argv**, not from the species config,
defaulting to 1.0. Blender 5.1 imports this FBX 1.25x larger than 4.2 did. Whoever
extracted `deer_polyart_walk` passed `--position_scale 0.8`; whoever extracted
`deer_polyart_trot` and `deer_polyart_run` did not.

Joint ANGLES are scale-invariant, so nothing looked wrong.

| clip | pelvis Z | lowest hoof (FK) | ÷1.25 pelvis | ÷1.25 lowest hoof | verdict |
|---|---|---|---|---|---|
| walk | 0.940 | **+0.029 planted** | 0.752 | −0.162 sunk | CORRECT |
| trot | 1.195 | +0.203 floating | **0.956** | **−0.028 planted** | 1.25x OVERSIZED |
| run | 1.281 | +0.195 floating | **1.025** | −0.077 | 1.25x OVERSIZED |

Dividing walk by 1.25 sinks it; dividing trot by 1.25 plants it. And a deer does not
raise its pelvis 27% to trot or 36% to run.

## Why it matters more than a height error

`root_position` carries the SPEED. Inflating position inflates velocity:

* trot is really **2.37 m/s**, not the 2.96 recorded everywhere
* run is really **6.00 m/s**, not 7.50

`base_lin_vel` is part of the AMP observation, so **the discriminator has been shown
1.25x inflated root velocities for two of our three clips** in every multi-clip run:
exp3, exp3b, D2, D2b, B, G, C3, C3b, F1, exp4.

Also downstream: CLAUDE.md's "reference speeds walk 1.14 | trot 2.99 | run 7.57" is
wrong for two of three; the slip reference bands used throughout
`docs/LEG_NEGLECT_findings.md` were computed at the inflated speeds; and
`bake_reset_states.py` RSI spawn heights come from these clips, so trot/run spawn poses
sit ~0.24 m too high.

## A hypothesis it may explain (NOT yet tested)

Only 9 of 16 runs track the commanded speed within ±0.3 m/s, and several overshoot badly
(exp4 hit 3.26 m/s when commanded 1.15; exp3 hit 2.01). If the style reward is pulling
toward a trot that *claims* to move at 2.96 m/s while the task reward asks for 1.15, the
two objectives are in direct conflict and the policy splits the difference by running
too fast. That would make the command-tracking problem a DATA bug rather than a tuning
problem. Testable: rescale the clips and re-measure achieved-vs-commanded speed.

## Why preflight missed it

Check 3 bounded pelvis height tightly only for clips named "walk", and loosely
(0.80–1.45) for everything else, on the reasoning that faster gaits ride higher. Trot at
1.195 and run at 1.281 both fit inside the loose band.

**Fixed.** Check 3 now FKs the hooves into world space and requires the lowest one to
reach the ground (±0.10 m). Rig geometry is scale-correct and clip root position is not,
so an oversized clip floats by exactly the excess. No gait band, no filename dependency.
Verified: passes walk, fails trot and run.

## The fix, not yet applied

Rescaling is a one-line change (`root_position *= 0.8`) but it alters the training data
for every future run, so it should be a deliberate decision rather than a silent repair:

1. Re-extract with `--position_scale 0.8`, or rescale the existing `.npy` in place
2. Rebuild the RSI bake (`tools/bake_reset_states.py`) — spawn heights change
3. Correct the reference speeds in CLAUDE.md
4. Recompute the slip reference bands (they anchor every gait number reported so far)
5. Re-run the command-tracking measurement to test the hypothesis above
