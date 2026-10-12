
## mi_scale and the missing diversity objective (2026-07-28)

Measured on C2 (ASE warm-started onto exp3's policy, mi_scale=0.5):

| iter | 50 | 150 | 250 | 400 | 550 | 700 | 850 |
|---|---|---|---|---|---|---|---|
| mi_reward | 0.358 | 0.596 | 0.696 | 0.753 | 0.780 | 0.799 | 0.807 |

Chance level for 8-dim unit vectors (clamped cosine) is **0.147**, so the encoder ends
at ~5.5x chance. It CONVERGES BY ~ITERATION 500 and then flattens -- so ASE's skill
separation is not training-limited, and exp4's 1500 iterations were ample. An earlier
claim that exp4 was "undertrained" was wrong on this axis.

COST: C2 vs exp3b (same policy, ASE added) -- err_vel_xy 0.378 -> 1.180, non_hoof
-0.072 -> -0.314. Skill diversity is bought with locomotion quality, and `mi_scale`
is the dial. 0.5 was chosen arbitrarily and has never been swept.

THE MISSING PIECE. ASE has THREE reward components; this implementation has two:
  1. style (AMP discriminator)                      -- implemented
  2. skill discovery / MI (encoder q(z|s,s'))       -- implemented
  3. DIVERSITY                                      -- NOT implemented
(3) compares the policy's action distribution under two different latents at the SAME
state and pushes them apart in proportion to the latent distance. It exists precisely
to stop the policy satisfying the encoder with small, non-semantic differences -- which
matches Raza's read of the z-sweep ("a bit diff, not insanely diff") and the MI plateau.

Why it is not implemented yet: it needs a hook inside rsl_rl's PPO update, where both
the actor and the POLICY observations are in scope. The current patches only see AMP
observations at reward time (via discriminator.predict_reward). An approximation bolted
on elsewhere would be exactly the kind of half-correct change that produced three
retracted conclusions this week.

Cheaper first: sweep mi_scale (E015 = 0.15, E15 = 1.5, vs C3's 0.5) which moves the
same tradeoff and needs no new code.

---

## UPDATE 2026-07-29 — the diversity objective is now implemented

`DiversityRegularizer` in `training/polyart/ase.py`, enabled by
`ANIML_DEER_ASE_DIVERSITY=0.01`. Formulation follows the reference implementation
(nv-tlabs/ASE, `ase/learning/ase_agent.py`):

    a_diff = mean_j ( clamp(mu_z1)_j - clamp(mu_z2)_j )^2     behavioural distance
    z_diff = 0.5 - 0.5 * <z1, z2>                             in [0,1] for unit z
    loss   = coef * ( target - a_diff / (z_diff + 1e-5) )^2   coef 0.01, target 1.0

The hook was the hard part, and the note above ("needs a hook inside rsl_rl's PPO
update") was right about why. rsl_rl 5.0.1 assembles its loss at ppo.py:313 and
backwards it at :363 with nothing in between. Rather than copy 150 lines of someone
else's update loop, wrap `optimizer.step`: at that point the graph is gone but the
gradients are live and unstepped, so a second `backward()` ACCUMULATES the diversity
gradient into the same buffers. The actor is then re-clipped, because our contribution
lands after rsl_rl's own `clip_grad_norm_` and would otherwise be the one unbounded
gradient in the system. Observations are captured off the actor's forward, keyed on
`stochastic_output=True and torch.is_grad_enabled()` -- true only inside `update()`,
never during the inference-mode rollout.

Verified CPU-only before queueing (`tools/test_ase_diversity.py`, 12 checks): a z-blind
policy scores bonus 0 and pays the full target penalty; 60 steps of training raise
spread 0.002 -> 0.79; a wrong latent slice raises SystemExit rather than silently
optimising joint-velocity columns.

**F1** = C3 + diversity, one variable changed.

## UPDATE 2026-07-29 — C3b: ASE phase 2 is a NEGATIVE result

C3b warm-started C3 under the expert-only normaliser, asking whether it would clean up
C3's 66.4% ankle load the way exp3b cleaned exp3's. It made ASE worse on both axes.

MI reward, compared at matched ENCODER AGE (the encoder is rebuilt fresh on warm-start,
so C3b's 1200 iterations must not be read against C3's 3000):

| iter~ | C3 | C3b |
|---|---|---|
| 200 | 0.845 | 0.440 |
| 600 | 0.920 | 0.468 |
| 1200 | 0.910 | 0.493 |

Roughly half, at every age. Leg balance fell with it: evenness 0.70 -> 0.31, worst leg
0.04 -> 0.01.

A caution about reading these two scalars together. C3b's encoder TRAINING loss is
*better* than C3's (0.045 vs 0.088 at iter~1200), which looks like a contradiction. It
is not: `encoder_loss` is measured during the update, after the encoder has trained on
that buffer, while `mi_reward` is measured on the NEXT rollout with the previous
weights. For C3 the two agree (loss 0.088 -> cos 0.912, measured MI 0.910); for C3b
they diverge hard. An encoder that fits its buffer but not the next rollout is one whose
z -> behaviour mapping does not hold on fresh data. **Do not report encoder_loss as
evidence of skill separation** -- it can be driven down by an encoder memorising a
collapsed policy.

That is still an inference from two logged numbers, so queue 13 z-sweeps C3b and
measures pose separation directly.

If it confirms, the ASE recipe is: **legacy normaliser for cold start, and do NOT
follow it with expert-only.** It would also be the second instance of the exp3 -> exp3b
pattern -- a sharper discriminator buys what it can measure by giving up what it cannot.
There it was front-leg load; here it is skill diversity. Both are invisible in the
82-dim AMP observation.

## UPDATE 2026-07-29 — F1 is running; how to read it WITHOUT fooling yourself

The diversity term is confirmed live (it is not enough that the config printed; the
proof is the logged scalar):

    ASE/diversity_bonus   0.0002 -> 0.0003   over the 40-iteration smoke, target 1.0
    ASE/diversity_loss    0.9997 -> 0.9993

At iteration 40 the policy sits at the fully collapsed end -- essentially zero action
spread across latents -- which is expected from scratch. The CPU test needed 60 gradient
steps at coef=1.0 to reach 0.79; F1 runs at coef=0.01, so the pressure is 100x weaker,
against roughly 20 optimiser steps per iteration over 3000 iterations. That should be
ample, but 0.01 is the reference default from nv-tlabs/ASE, tuned for a HUMANOID with a
different action scale and reward magnitudes. It is not obviously right for a deer.

**Check the bonus before the z-sweep.** There are two failure modes that look identical
in a z-sweep and imply opposite next steps:

  * final bonus still <0.05  -> the objective never got traction. NOT "diversity does
    not work" -- the coefficient is too weak. Next step is a coef sweep (0.1, 1.0).
  * bonus rose toward 1.0 but separation is flat -> the spread really is limited by the
    3-gait dataset, and more clips is the only path.

Reporting the first as the second would retire a working idea on the strength of an
untuned hyperparameter. That is the same shape as the `entropy_coef=0` and
`action_scale=0.5` mistakes already recorded in CLAUDE.md: a default carried over from a
different body, mistaken for a property of the method.

## CORRECTION 2026-07-30 — "C3b halved skill separation" was WRONG

I reported C3b as making ASE worse on both axes, on the strength of its MI reward being
about half C3's at matched encoder age. The z-sweep, built specifically to test that
inference, refutes it:

| run | phase-aligned separation | min pair | speed spread |
|---|---|---|---|
| exp4 | 30.0° | 23.1° | 3.40 |
| C3b | **17.7°** | **12.9°** | **1.50** |
| C3 | 17.6° | 8.9° | 0.82 |

C3b is EQUAL on mean separation and BETTER on the worst pair and on speed spread. Its
skills are not collapsed. What survives of the original claim is the gait half only —
matched-condition leg balance really is worse (evenness 0.29 vs 0.37, worst leg 0.01 vs
0.03), while slip is a wash (4.0× vs 4.2× reference).

**The lesson is about the metric, not about C3b: MI reward is NOT a measure of
behavioural diversity.** I used it as a proxy for separation and it pointed the wrong
way. That is the same failure as `amp_status`'s "peak", as net-vs-magnitude yaw, and as
duty-vs-load — a summary standing in for the thing it does not actually measure. The
z-sweep is a direct behavioural measurement and is the only trustworthy separation
number; `encoder_loss` and `mi_reward` describe the ENCODER's fit, not the policy's range.

Practical consequence: exp4 still holds the separation record (30.0°) at the LOWEST MI of
the three (0.82). Across exp4 / C3 / C3b, MI and separation are uncorrelated or inversely
related. Anything reported as "skill diversity" must come from a z-sweep.

## RETRACTION 2026-07-30 — "exp4 reached MI 0.82" was never measured

That figure appears throughout these notes and the handoff as the motivating evidence for
building the diversity objective ("MI 0.82 with only 22-36° of separation → high
information, near-identical motion"). It has no source. exp4's log contains exactly one
ASE line, `[ase] stage-1 ON`, and its event file has NO ASE scalars at all — both the
stdout print and the TB logging were added in C1, *after* exp4 ran. The number was
carried forward as if measured.

What IS measured for exp4 is its z-sweep: 30.0° phase-aligned separation, min pair 23.1°,
speed spread 3.40 m/s. That stands, and it is still the best separation on record.

## A REAL confound in the exp4-vs-C3 comparison, and a hypothesis worth testing

exp4 and C3 have **identical env-var configs**. The only difference between them is
training length: exp4 ran 1500 iterations, C3 ran 3000. So the separation gap
(30.0° vs 17.6°) is not evidence about any config choice — it is 1500 iterations versus
3000, and it suggests **skill separation DECAYS with training** while MI keeps rising.

That would be the "cheaply detectable" failure in its purest form: as training proceeds
the policy makes its skills more subtly distinguishable rather than more different, which
satisfies the encoder while shrinking behavioural range. It also predicts that early
checkpoints are behaviourally richer than final ones — worth knowing before anyone builds
stage 2 on a final checkpoint.

Testable directly and cheaply: C3 kept `model_1500.pt`. Z-sweeping C3 at exp4's exact age
isolates training length from everything else.
  * C3@1500 ≈ 30° -> separation decays with training; exp4 was not special
  * C3@1500 ≈ 17.6° -> the gap is something else and this hypothesis is dead

## RESULT 2026-07-30 — F1: the diversity objective works, modestly

| run | diversity | iters | aligned sep | min pair | spd spread | final MI |
|---|---|---|---|---|---|---|
| exp4 | no | **1500** | **30.0°** | **23.1°** | **3.40** | not logged |
| F1 | **yes (bonus 0.81)** | 3000 | **19.8°** | 10.3° | **1.35** | 0.970 |
| C3b | no | 4199 | 17.7° | 12.9° | 1.50 | 0.968 |
| C3 | no | 3000 | 17.6° | 8.9° | 0.82 | 0.973 |

**F1 vs C3 is a clean single-variable comparison** — identical config, identical 3000
iterations, diversity coefficient the only difference:
* phase-aligned separation 17.6° → 19.8° (+12.5%)
* worst pair 8.9° → 10.3° (+16%)
* speed spread 0.82 → 1.35 m/s (+65%)
* MI held: 0.973 → 0.970, so the two objectives genuinely coexist at coef 0.01

So the objective does what the paper claims: it converts latent distance into behavioural
distance, and it does not fight the encoder. Confirmed positive.

**But it is small compared to training length.** exp4 — same config, no diversity
objective, HALF the training — sits at 30.0°/23.1°/3.40, above everything else by a wide
margin. If the decay hypothesis holds (queue 18 tests it on C3's own model_1500.pt), then
*when you stop training* dominates *whether you added the objective*, and the honest
ranking of ASE levers becomes:
  1. stop earlier / prevent the decay
  2. the diversity objective (+12.5%)
  3. the expert-only normaliser phase (neutral on skills, mildly negative on gait)
That is not the ordering I expected when I built the objective, and it is worth stating
plainly rather than leading with the +12.5%.

## CONFIRMED 2026-07-30 — ASE skill separation DECAYS with training

| run | iters | aligned sep | min pair | spd spread |
|---|---|---|---|---|
| **F1 @1500** | 1500 | **34.5°** | **24.1°** | 2.28 |
| exp4 | 1500 | 30.0° | 23.1° | 3.40 |
| C3 @1500 | 1500 | 24.3° | 19.2° | 2.69 |
| F1 final | 3000 | 19.8° | 10.3° | 1.35 |
| C3b | 4199 | 17.7° | 12.9° | 1.50 |
| C3 final | 3000 | 17.6° | 8.9° | 0.82 |

**Within-run, so no config or seed confound is possible:**
* C3: 24.3° @1500 → 17.6° @3000 (**−28%**)
* F1: 34.5° @1500 → 19.8° @3000 (**−43%**)

Every 1500-iteration checkpoint beats every final checkpoint with no overlap. Meanwhile MI
keeps CLIMBING over the same interval (F1: 0.909 at 10% → 0.970 final). That is the
"cheaply detectable" failure in its purest measured form: **as training proceeds the policy
makes its skills more subtly distinguishable rather than more different.** The encoder gets
happier while behavioural range collapses.

### Two consequences

1. **I understated the diversity objective.** Compared at matched age it is +42%
   (F1 34.5° vs C3 24.3° @1500), not the +12.5% I reported from final checkpoints — I was
   comparing two already-decayed policies. **F1@1500 is the best ASE checkpoint produced
   so far** (34.5° / 24.1° / 2.28). The objective also does not prevent the decay, only
   starts higher.
2. **Do NOT build stage 2 on a final stage-1 checkpoint.** It throws away ~40% of the
   skill range the policy briefly had. Either stop stage 1 near 1500 iterations or select
   the checkpoint by z-sweep separation rather than by final reward.

### Caveat, stated rather than buried
exp4 and C3@1500 are the SAME config at the SAME age (identical env vars, seed 42) and
still differ 30.0° vs 24.3° — roughly 21% run-to-run spread, presumably GPU
nondeterminism. The +42% diversity effect sits above that noise but deserves replication.
The decay finding is immune to it, being a same-run comparison.

### Revised ranking of ASE levers
1. **When you stop** (or checkpoint selection) — up to +40% separation, free
2. **The diversity objective** — +42% at matched age, one env var
3. The expert-only normaliser phase — neutral on skills, mildly negative on gait
