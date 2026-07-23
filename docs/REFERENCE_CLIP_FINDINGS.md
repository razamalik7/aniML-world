# Reference clips — what is actually wrong with them

**2026-07-23, CPU-only session. Nothing live was modified.** Corrected clips are
written as **copies** to `assets/motions/v2/` by
`tools/cv/rebuild_reference_clips.py`; `assets/motions/*.npy` and
`tools/convert_unity_anim_to_npy.py` were not touched.

**This file corrects two claims in `docs/CV_PIPELINE_PROGRESS.md` §1.** Read this
one where they disagree — the corrections are backed by an independent
measurement path (the artist source animation in Blender) that the earlier
analysis did not have.

---

## Summary

| # | Claim | Status |
|---|---|---|
| 1a | Front fetlock reference is outside URDF limits on 31/31 frames | **CONFIRMED — and the cause is the opposite of what was assumed.** The clip is right; the *rig's limits* are wrong. See §1. |
| 1b | Walk clip's stored speed is 1.7× too slow; feet slide 3.9× further than the body travels | **NOT ESTABLISHED — do not act on it.** The estimator that produced it is unreliable on this clip. See §2. |
| 1c | Clip stands 10–17 cm underground | **CONFIRMED.** See §3. |
| 1d | Silent zero-fill landmine on joint-name mismatch | **CONFIRMED**, dormant by accident. Fixed in the v2 copies. |
| 1e | On-disk `root_quaternion` is XYZW, not WXYZ | **CONFIRMED.** |
| NEW | The source animation has no physically valid stance phase | **MEASURED.** Duty factors 0.16–0.39 where a walk requires >0.5. See §2. |
| NEW | `deer_run.npy`'s speed is ~46% too fast | **LIKELY**, on the one estimator that is trustworthy here. See §2. |

---

## 1. Front fetlock — the clip is correct, the rig is wrong

`FR_shin_to_FR_foot` spans **[−1.20, −0.40] rad** in the walk clip; the URDF
allows **[−0.30, +1.60]**. Outside the limit on **31 of 31 frames**.

The obvious reading is a converter sign error — the URDF's front and hind
conventions genuinely are mirror images (front knee [0, +2.2] vs hind [−2.2, 0]),
and negating lands the channel neatly inside its limit. **That reading is wrong.**
Negating the channel and measuring what it does to the geometry:

| variant | L front hoof lift | R front hoof lift |
|---|---|---|
| **original clip** | **0.145 m** | **0.144 m** |
| negated | 0.515 m | 0.525 m |
| *artist source animation, for reference* | *0.205 m* | *0.137 m* |

A walking deer lifts a hoof 0.10–0.20 m. The original values land there and match
the artist source; negating produces a half-metre high-step no deer walks with.

**So the clip's fetlock kinematics are correct and the URDF's front fetlock
limits are wrong.** The hind fetlock limit is [−1.60, +0.30] and the hind clip
data sits comfortably inside it. The front limit is its mirror image, [−0.30,
+1.60], and excludes the entire correct range. Front and hind fetlocks bend the
same way on a real quadruped, so mirroring that particular limit looks like the
error.

**Consequence, and it is significant:** in Isaac the front fetlock is clamped to
[−0.30, +1.60] while the reference asks for [−1.20, −0.40]. The joint is pinned
at its lower stop and can never move into the reference range. That is a concrete
mechanism for the repeatedly-reported *"front legs splayed forward as static
struts, front knees fully rigid"* and the human review's *"everything rigid"*.

**Suggested fix (NOT applied):** set the front fetlock limits to match the hind,
`lower="-1.6" upper="0.3"`, in `assets/deer_urdf/deer_world.urdf` for
`FL_shin_to_FL_foot` and `FR_shin_to_FR_foot`, then regenerate the USD. Worth
checking the front knee limits by the same method while in there.

`rebuild_reference_clips.py` deliberately leaves these frames **unclamped** —
clamping would degrade a correct reference to satisfy a wrong limit.

---

## 2. Speed and foot slide — the earlier numbers are estimator artifacts

`CV_PIPELINE_PROGRESS.md` §1b reports stored 0.77 m/s vs foot-odometry 1.32 m/s,
foot slide 1.71 m/s, and slide/travel = 3.93×, concluding `FORWARD_SPEED` should
be raised to ~1.3. **Do not act on this.**

I built an independent foot-odometry estimator and got 1.18–1.46 m/s depending on
the stance threshold — same ballpark, and also wrong. Both estimators rest on
detecting when a hoof is planted. So I measured the duty factor directly from the
**artist source animation** in Blender, which removes the converter from the loop:

| clip | FL duty | FR duty | HL duty | HR duty |
|---|---|---|---|---|
| walk | 0.39 | 0.26 | 0.16 | 0.16 |
| run | 0.28 | 0.28 | 0.48 | 0.44 |

**A walk is *defined* by a duty factor above 0.5 on every leg** — each foot is on
the ground more than half the cycle. Measured 0.16–0.39. The source animation has
no physically valid stance phase: the hooves never plant and hold.

That invalidates every stance-based number on this clip — both estimators' speeds,
and the foot-slide and slide/travel figures, which are computed over frames
labelled "stance" that are not stance. Stance-speed estimates on the artist source
scatter across 1.23–1.75 m/s per hoof with a 4.03 m/s cross-check; nothing
converges.

**What can still be said.** The one estimator that needs no stance detection is
the hoof's peak-to-peak fore-aft excursion over a cycle, which brackets the
depicted speed:

| clip | front stride | hind stride | implied speed band | stored |
|---|---|---|---|---|
| walk | 0.683 m | 0.912 m | 0.66 – 0.88 m/s | **0.774 m/s** ✅ |
| run | 1.517 m | 1.222 m | 1.47 – 1.82 m/s | **2.400 m/s** ❌ |

(This brackets rather than pins the speed, since excursion equals stride only at
duty factor 1. It is a lower bound on the depicted speed, and the most robust
number available here.)

**Recommendation: leave `FORWARD_SPEED = 0.8` for the walk clip alone.** It sits
inside the band and there is no better estimate. **`deer_run.npy`'s 2.5 m/s does
look too fast** — roughly 46% above the band — which inverts the earlier
conclusion that run was the healthy one.

**The deeper finding.** The source is a stylised game animation, not physically
grounded motion. Its front and hind legs disagree about stride length by **25% in
the artist rig itself** (front 0.683 m vs hind 0.912 m), so *no* single root speed
can satisfy both ends, and no post-processing fixes that. This is a hard ceiling
on how self-consistent any clip derived from it can be — and it is the strongest
concrete argument for the video→motion workstream: real footage of a real deer has
a real stance phase by construction.

It also means **contact-based root recovery, recommended in
`CV_PIPELINE_DESIGN.md` §3, cannot be validated on this clip.** It needs motion
that actually has stance phases. Validate it on video-derived motion, or on a sim
rollout, not here.

---

## 3. Ground contact — confirmed and fixed in the copies

`spawn_height = 0.93` puts the hooves 10–17 cm below z=0. FK-grounding the root
per frame (set root height so the lowest hoof touches z=0) gives a mean of
**1.071 m** for the walk clip. After grounding, `validate_motion.py` reports
`max_penetration_m = 0.0000` and `max_float_m = 0.0000`.

---

## 4. What `rebuild_reference_clips.py` produces

```bash
python tools/cv/rebuild_reference_clips.py                 # dry run
python tools/cv/rebuild_reference_clips.py --write         # -> assets/motions/v2/
```

Applied by default:
- joint names remapped to current URDF names (kills the silent zero-fill trap);
- root height FK-grounded per frame;
- root trajectory from foot odometry, **with the caveat in §2** — on this clip
  that is not trustworthy, and `--constant-speed` should be preferred until the
  reference has a real stance phase;
- velocities recomputed cyclically;
- out-of-limit frames reported but **not** clamped (§1).

Off by default, deliberately: `--sign-fix` (refuted in §1), `--clamp` (would
degrade a correct clip), `--mirror` and `--cycles N` (available for lengthening
the 1.03 s clip, which is shorter than any dataset measured in the AMP
literature).

---

## 5. What I would actually change, in priority order

1. **Front fetlock limits in the URDF** (§1). Highest value, one line, and it
   plausibly unlocks the front legs after ~40 runs of rigidity.
2. **Hind foot/toe collision meshes** — separate defect, see
   `docs/RIG_DEFECT_hind_foot_collision.md`. Also upstream of everything.
3. **Rename the clips' `joints_list`** to current URDF names (§1d). Removes a
   silent-failure landmine; the v2 copies already do it.
4. **Leave the walk speed alone** (§2). Reconsider the run clip's 2.5 m/s.
5. **Do not invest further in foot-odometry post-processing of these clips** (§2).
   The source lacks the stance phase it depends on.
