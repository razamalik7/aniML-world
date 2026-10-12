# Leg neglect: what it actually is (2026-07-29)

Raza's video reads kept returning to the same thing — "definitely prone to some leg
neglect", "one leg barely being used", "other legs seeming more completely vestigial".
This is the investigation of that, and it changes two things I had previously reported.

## 1. No leg is kinematically dead. Anywhere.

The leg-participation score in `gait_report.py` measures **contact duty** — how often a
hoof is loaded. A low number is compatible with two very different failures:

* **A. frozen** — the joints stop moving, the deer drags the limb
* **B. unloaded** — the limb swings through a normal arc but carries no weight

They need opposite fixes, so `tools/leg_kinematics_vs_load.py` measures both: contact
load share, *and* range of motion per limb against the expert clips.

Every limb in every run is in category B. Range of motion, as a fraction of expert:

| run | L-hind | R-hind | L-front | R-front |
|---|---|---|---|---|
| exp1 | 1.11 | 1.51 | 1.12 | 1.11 |
| exp3 | 1.41 | 1.40 | 1.06 | 1.23 |
| exp3b | 1.28 | 1.25 | 0.88 | 0.94 |
| A1 | 1.37 | 1.37 | 1.12 | 1.17 |
| A2 | 1.27 | 1.24 | 0.95 | 0.98 |
| C2 | 0.90 | 1.46 | 1.10 | 0.88 |
| C3 | 0.80 | 0.79 | 0.85 | 1.04 |

Nothing is below 0.79. Even C2 — the run I described as having "3 dead legs" — moves
all four limbs through 0.88–1.46× the expert's arc. Its L-front swings at 1.10× expert
while carrying **1 N**. The limbs are not dead; the deer is standing on a subset of them.

That reframes the fix. "Make the leg move" is not the problem to solve. "Make the deer
put weight on it" is. And nothing in the current reward set or in the AMP observation
mentions load at all — which is the same gap experiment B (hoof height) was built for.

## 2. The participation numbers I reported were confounded by the turn command

**All seven runs were rolled out with `CMD_ANG_VEL_Z=0.5`** — a yaw command sampled in
±0.5 rad/s, i.e. ±29 °/s. Measured net yaw over each rollout:

| run | mean yaw °/s | L/R load |
|---|---|---|
| exp3 | −1.1 | 0.97 |
| exp3b | +2.3 | 1.15 |
| A2 | +3.9 | 0.95 |
| C2 | **+7.9** | **0.01** |
| C3 | **+14.1** | **0.14** |

C3's rollout averaged a **sustained 14 °/s left turn across all four envs**. A deer
that is correctly cornering unloads its inside legs. So "C3 neglects its left legs" and
"C3 was commanded to turn left" are indistinguishable in the data I had — because
**the command was never recorded in the rollout**.

This is the same failure shape as every other one on this project, and it is worth
naming precisely: the metric was fine, the *condition it was measured under* was
invisible. Turn contamination in the expert data (exp1) was this. The `amp_status`
"peak" was this. A summary is only interpretable next to the condition that produced it.

Two fixes, both landed:

* `tools/rig_pipeline/collect_rollout.py` now records `commands` (T, N, 3) in every
  rollout, and `leg_kinematics_vs_load.py` prints it and **refuses to let a >3 °/s net
  yaw command pass without a warning**.
* `.run_queue9.sh` re-collects every finished run under one matched condition — yaw
  command pinned to 0, speed pinned to the reference walk band 1.0–1.3 m/s — into
  `docs/*_straight.npz`, then writes `docs/LEG_PARTICIPATION_matched.txt`.

**Until queue 9 completes, treat every front/hind and left/right number in the earlier
tables as provisional.** The ROM finding in §1 is not affected — range of motion does
not depend on which way the deer was told to go.

## 3. The duty threshold was doing real work

`gait_report.py` calls a hoof loaded above **20 N**; my first pass used 5 N, and the
two disagreed enough to change the story. The reason is that **23–40% of all "contacts"
sit between 5 N and 20 N** — on an 809 N animal, that is a hoof brushing the ground,
not standing on it. A duty-based score counts a grazing touch as a working leg.

Load *share* — mean force on that hoof over the sum across all four — needs no
threshold and is what the tables above now use. A fair share is 0.25.

## 4. The mechanism: the neglected leg never stops moving

If a limb is neither frozen nor held high, what is it doing wrong? It never establishes
stance. In stance a hoof is planted and the body travels over it, so its speed **in the
world frame** falls to ~0. A hoof that is merely brushing past keeps moving.

Slip ratio = hoof horizontal speed / body speed, taken over the lowest quarter of that
hoof's own height range. 0 = planted, ~2 = swinging.

| | L_Toe0 | R_Toe0 | L_Finger0 | R_Finger0 | body m/s |
|---|---|---|---|---|---|
| REF walk | 0.04 | 0.11 | 0.12 | 0.08 | 1.11 |
| REF trot | 0.20 | 0.44 | 0.35 | 0.42 | 2.96 |
| REF run | 0.26 | 0.34 | 0.34 | 0.29 | 7.50 |
| exp3 | 0.11 | 0.07 | **0.73** | 0.19 | 1.74 |
| exp3b | 0.06 | 0.12 | **0.99** | **0.86** | 3.29 |
| A2 | 0.06 | 0.45 | **1.55** | 0.47 | 1.54 |
| C3 | **0.71** | 0.17 | **0.91** | **1.06** | 2.72 |

The reference plants all four hooves at 0.04–0.44 **across all three gaits**, including
the 7.5 m/s run. Our hind hooves match it (0.06–0.17). Our front hooves skim at
0.73–1.55 — six to thirteen times the reference.

And slip predicts the missing load. A2: L_Finger0 slip 1.55 → share 0.02; R_Finger0
slip 0.47 → share 0.32. exp3: L_Finger0 slip 0.73 → share 0.06; R_Finger0 slip 0.19 →
share 0.36. Same leg, both metrics, every run.

**The AMP observation contains none of this.** It is joint angles, joint velocities, and
base velocity — nothing that locates a hoof relative to the ground it is sliding along.
This is the largest measured policy/reference discrepancy in the project, and the
discriminator has no term for it.

Experiment **G** adds it: per-hoof horizontal world speed, 4 columns, alongside B's 4
height columns. It stays inside Raza's physics-free constraint — FK plus a finite
difference on the reference side, `body_lin_vel_w` in sim, no contact forces claimed of
an animation clip that has none. `ANIML_DEER_AMP_HOOF_V=1`.

> **G RAN AND FAILED — see the RESULT section at the end of this file.** The diagnosis in
> this section holds (slip is the discrepancy, and it does track load), but the
> prediction that feeding it to the discriminator would fix the gait is falsified: slip
> got monotonically WORSE along D2 → B → G (3.7× → 6.9× → 10.7× reference). Do not read
> this section as a live prediction.

## 5. Why phase 2 cleaned the hooves and dropped the front legs

This was the open question from exp3 → exp3b: the expert-only normaliser took ankle
load from 33.5% to 3.2%, and front/hind load from 1.12 to 0.37. Both real, apparently
contradictory. Broken out by channel:

| | ankle load | hind slip | front slip | front/hind load |
|---|---|---|---|---|
| exp3 | 33.5% | 0.09 | 0.46 | 0.74 |
| exp3b | **3.2%** | 0.09 | **0.93** | **0.31** |

Phase 2 fixed the hind foot-strike and left hind slip untouched at an excellent 0.09 —
while front slip roughly doubled and front load halved. The mechanism follows from what
the discriminator can and cannot see:

**Ankle-walking is visible in the AMP observation. Front-hoof skimming is not.**
Standing on the ankle is a distinctive *joint angle*, and joint angles are 38 of the 82
columns. Skimming is a relationship between the hoof and the ground, and there is no
column for that at all. So sharpening the discriminator — which is exactly what
expert-only normalisation does, by refusing to let the policy's own outliers inflate
the variance that hides them — tightened the error it could measure, and the policy
paid for that tightening on the axis nothing was watching.

The general form is worth keeping: **a sharper discriminator on an incomplete
observation does not merely fail to fix the unobserved defect — it can make it worse,
because the policy trades away what is not measured to buy what is.** That is an
argument for fixing the observation (B, G) *before* reaching for a sharper critic, and
it is the same lesson as the reward proxies, one level up.

Two caveats, stated rather than buried. exp3's rollout ran at 1.74 m/s and exp3b's at
3.29, and the reference's own slip rises with speed (walk 0.04–0.12, trot 0.35–0.42),
so against the speed-appropriate reference the degradation is ~1.8× → ~2.4×, not the
2× the raw numbers suggest. And both were rolled out under uncontrolled yaw. The
matched re-collection in queue 10 settles both; until then this is the mechanism, not
the measurement.

## 5b. The veer is not universal — it tracks how much VARIETY the run trained on

The veer (uncommanded heading drift) had been open since before compaction, described
as ~25-37° per 8 s with no known cause. It could not be attributed before, because no
rollout recorded its yaw command. D2 and D1 are the first runs both TRAINED and ROLLED
OUT at yaw exactly 0, so the residual is finally attributable.

| run | expert set | cmd band | per-env net yaw °/s | mean | spread |
|---|---|---|---|---|---|
| D2 | walk + trot + run | 0.8–3.2 | −1.0, −1.1, **+2.7**, −0.4 | **+0.04** | 1.57 |
| D1 | walk only | 1.0–1.3 | −5.4, −6.8, −5.1, −8.2 | **−6.36** | 1.23 |

**D2 does not veer.** Signs are mixed and the mean is 0.04 °/s — the residual is
per-episode noise, not a bias. **D1 veers hard and consistently**: four envs out of
four in the same direction, at a magnitude five times the spread.

The walk clip is not the culprit — it measures net 0.0 °/s with a max one-stride
window of 0.0 (preflight checks 2 and 2b). D1 is not imitating a turn. It is imitating
a *single 33-frame stride*, roughly one gait cycle, whose front limbs differ ~20% in
range between left and right (§6). With one clip and a 0.3 m/s command band there is
nothing to force the policy off whichever asymmetric solution it first finds; D2's three
clips across a 2.4 m/s band keep pushing it to re-adapt.

So the answer to "what targets the veer" is **variety, not a yaw penalty** — which is
also the ASE-compatible answer, since a heading-holding reward would be exactly the kind
of task-specific shaping stage 1 must not carry.

Caveat worth stating: this is 4 envs per run and one rollout each. The sign consistency
is strong (4/4, mean 5× the spread) but it should be re-checked in the matched
re-collection, where both runs are re-rolled under identical conditions.

## 6. The neglect is structural; WHICH leg is neglected is emergent

I nearly concluded the wrong thing here twice, so both corrections are recorded.

**First wrong idea: a rig defect favouring the left front.** L-front was lightest in
the 7 runs I first looked at, which seemed too consistent for noise. Ruled out:

* **mass asymmetry** — front-limb L/R masses identical to 5 decimal places
* **reference asymmetry** — on the clean walk clip the FK'd hoof-height minima differ
  by 2 mm (hind) and 8 mm (front) between sides. In joint space the reference actually
  uses the LEFT front slightly *more* (L/R range ratio 1.20 at walk, 1.10 at trot), so
  the bias is not inherited from the data
* **joint-limit asymmetry** — real, but not front-specific: once flipped joint axes are
  accounted for, mean L/R limit mismatch is 18.2° front vs 14.9° hind. `Forearm_to_Hand`
  is the one genuine outlier (~45°), and the *left* has the wider range, which does not
  explain the left being unloaded

**Second wrong idea: seed-driven symmetry breaking.** Every run uses `seed: 42`
(`amp_runner_cfg.py:77`, never overridden), and a quadruped must break left-right
symmetry to walk — it has to choose which diagonal leads. Identical initialisation
breaking identically would explain perfect consistency. It is a clean story and it is
wrong: widening to all 18 historical rollouts, the lightest leg is L-front 12 times,
**R-front 4 times** (exp4, r12, r13, r16p1) and **L-hind twice** (D1, r10). r12 and r13
loaded L-front at 0.56 — the *most* loaded leg — while neglecting R-front. D1 and D2
share a seed, a rig and a codebase and disagree on which leg they drop.

**What is actually universal** is that *some* leg is always dropped. Every rollout ever
measured has one leg at or below 0.11 against a fair share of 0.25. The identity is
emergent from data, command band and dynamics; the neglect itself is not.

This changes how the fix must be read. "Did L-front improve?" cannot compare two runs,
so `leg_kinematics_vs_load.py` now reports **worst-leg share** and **evenness** (the
normalised entropy of the four shares, 1.0 = perfectly balanced) instead. Current
standings, best first:

| run | worst leg | evenness | lightest | yaw cmd |
|---|---|---|---|---|
| D1 | **0.10** | 0.78 | L-hind | 0 (recorded) |
| D2 | **0.08** | 0.87 | L-front | 0 (recorded) |
| exp3 | 0.06 | 0.86 | L-front | ±0.5, unrecorded |
| exp1 / A1 / C3 | 0.04 | 0.70–0.88 | L-front | ±0.5, unrecorded |
| exp3b | 0.03 | 0.82 | L-front | ±0.5, unrecorded |
| A2 | 0.02 | 0.81 | L-front | ±0.5, unrecorded |
| C2 | 0.00 | 0.10 | L-front | ±0.5, unrecorded |

D1 and D2 lead, but they are also the only two measured at a known yaw of 0, so part of
that margin is the rollout condition rather than the training. Queue 10 re-rolls every
run at yaw 0 and 1.0–1.3 m/s, which is what makes this table trustworthy.

---

# RESULT 2026-07-30: the hoof channels FAILED. My hypothesis is falsified.

Matched conditions (yaw 0, command ~1.15 m/s, every run re-rolled identically), slip
normalised by the reference at each run's ACHIEVED speed:

| run | AMP obs | achieved | slip | ×reference | worst leg | evenness |
|---|---|---|---|---|---|---|
| exp3 | 82 | 2.01 | 0.72 | **3.3×** | 0.06 | 0.86 |
| D2 | 82 | 1.42 | 0.48 | **3.7×** | 0.07 | 0.89 |
| C3 | 82 | 1.86 | 0.81 | 4.2× | 0.03 | 0.37 |
| F1 | 82 | 1.77 | 0.80 | 4.4× | 0.01 | 0.49 |
| D1 | 82 | 1.10 | 0.56 | 6.4× | 0.03 | 0.78 |
| B | 86 | 1.43 | 0.92 | **6.9×** | 0.02 | 0.55 |
| A2 | 82 | 1.22 | 0.89 | 8.6× | 0.01 | 0.72 |
| G | 90 | 0.96 | 0.94 | **10.7×** | 0.02 | 0.78 |
| exp3b | 82 | 1.03 | 1.10 | 12.6× | 0.01 | 0.68 |

**The D2 → B → G ladder gets monotonically WORSE at the one thing the channels were
built to fix**: 3.7× → 6.9× → 10.7×. Robust to the normalisation — G is second-worst on
absolute slip too (0.94 vs D2's 0.48).

## Why the reasoning was right and the prediction still wrong

The diagnosis held up: slip IS the largest policy/reference discrepancy, it DOES track
missing load inversely, and height genuinely cannot separate a supporting leg from a
passenger. What did not follow is the step from "the discriminator can now see the
defect" to "the policy will fix the defect".

The mechanism is a coupling between observation width and reward scale, and two
independent monotonic ladders agree on it:

| run | AMP obs | amp_loss | style_r | slip ×ref |
|---|---|---|---|---|
| D2 | 82 | 0.4449 | 0.4419 | 3.7× |
| B | 86 | 0.3883 | 0.3846 | 6.9× |
| G | 90 | 0.3561 | 0.3551 | 10.7× |

Slip is 6–13× off reference: an extremely easy feature to discriminate on. Handing the
discriminator that feature lets it win more cheaply (amp_loss falls further below
ln 2 = 0.693), so the style reward it emits shrinks, so the policy receives LESS
pressure toward the expert. **Adding an observation channel spent style signal to buy
discriminator confidence.**

That is a real, reusable lesson: in AMP the observation and the reward scale are not
independent knobs. A channel that separates too well is a liability.

## What is still unsolved, stated plainly

* **Leg neglect.** No run of nine exceeds 0.07 worst-leg share against a fair share of
  0.25. Nine configurations — data variety, command band, normaliser phase, ASE, two
  observation channels — and not one moved it.
* **Foot-strike.** The best run is 3.3× reference slip. Nothing is close.

## The next lever is the REWARD, not the observation

The evidence now points away from adding channels. Options, in order of how directly the
evidence supports them:
1. **`style_weight`** (currently 0.5). If the style reward shrank ~20% across the ladder,
   restore its influence rather than removing the information. Single variable, testable
   on G's own checkpoint config.
2. **Weaken the discriminator** (smaller hidden dims, lower lr, gradient penalty) so it
   cannot win as cheaply on the new channel. Standard AMP practice.
3. **More reference clips.** Independently supported: D1 (1 clip) veers −6.4 °/s and D2
   (3 clips) does not; exp3 (3 clips) is the best run in this table.
Do NOT read this table as "hoof channels are a dead end" — read it as "they were added
without retuning the reward scale they perturb."

## CORRECTION: achieved speed must be controlled for too

I briefly reported exp2 and exp4 as the best runs on leg balance (both worst leg 0.11).
Both are artifacts, and the reason is the same one that has bitten every other metric here
— the condition was not controlled.

Slip is a RATIO to body speed, so achieved speed changes it two ways: it scales the
denominator, and it selects which reference band the run is compared against.

* **exp2 achieved 0.37 m/s** against a 1.15 command. A nearly stationary deer distributes
  load evenly for free; 0.11/0.93 measures standing still, not walking.
* **exp4 achieved 3.26 m/s** against the same command. It ignores the command and runs,
  which earns it the lenient trot reference band (0.35 vs 0.09) and its apparent 1.9×.

Only **9 of 16** runs land within ±0.3 m/s of the command; achieved speeds span
0.37–3.26. Restricted to those 9, the ranking is:

| run | speed | slip ×ref | worst leg | evenness |
|---|---|---|---|---|
| A1 | 1.29 | **3.2×** | 0.03 | 0.86 |
| **D2** | 1.42 | **3.7×** | **0.07** | **0.89** |
| exp1b | 1.13 | 5.1× | 0.04 | 0.83 |
| D1 | 1.10 | 6.4× | 0.03 | 0.78 |
| B | 1.43 | 6.9× | 0.02 | 0.55 |
| A2 | 1.22 | 8.6× | 0.01 | 0.72 |
| exp1 | 1.10 | 9.0× | 0.07 | 0.85 |
| G | 0.96 | 10.7× | 0.02 | 0.78 |
| exp3b | 1.03 | 12.6× | 0.01 | 0.68 |

**D2 is the best policy.** And D2 / B / G all sit inside the tracking band
(1.42 / 1.43 / 0.96), so the monotonic ladder 3.7× → 6.9× → 10.7× is like-for-like and
the G falsification is robust to this correction.

**New open problem: command tracking is poor project-wide** and has been invisible because
nothing reported achieved-vs-commanded speed. Any future gait metric must be read next to
the achieved speed, not the commanded one.
