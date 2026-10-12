# aniML World

Physically simulated animals that learn to move from motion capture–style reference
animation, trained with adversarial imitation learning in NVIDIA Isaac Lab. The long-term
target is an ecosystem simulation in Unity 6 where every animal's body is driven by a
learned controller rather than canned animation.

The first species is a deer: a 38-joint articulated body with full self-collision,
trained with Adversarial Motion Priors (AMP) and then Adversarial Skill Embeddings (ASE).

<p align="center">
  <img src="docs/media/trot_d2.gif" width="560" alt="Learned deer trot in Isaac Lab">
</p>

<p align="center"><em>A learned trot. No keyframes at runtime: every joint is torque-driven
by a policy trained against the reference animation below.</em></p>

<p align="center">
  <img src="docs/media/reference.gif" width="360" alt="Reference animation">
</p>

## Progress

The first few months were mostly about getting a simulated deer to stay on its feet at
all. These are from an earlier rig, before the current 38-joint build:

<table>
  <tr>
    <td align="center" width="25%" valign="top"><b>Can't get up</b><br><img src="docs/media/early_sprawl.gif" width="180"><br><sub>Chest on the ground, legs splayed, never rises</sub></td>
    <td align="center" width="25%" valign="top"><b>Falls over</b><br><img src="docs/media/early_fall.gif" width="180"><br><sub>Stands for a moment, collapses within two seconds</sub></td>
    <td align="center" width="25%" valign="top"><b>Stands up the wrong way</b><br><img src="docs/media/early_rear.gif" width="180"><br><sub>Gets off the ground by rearing onto its hind legs</sub></td>
    <td align="center" width="25%" valign="top"><b>Upright, stiff-legged</b><br><img src="docs/media/early_stiff.gif" width="180"><br><sub>Stays up, but shuffles on locked, splayed legs</sub></td>
  </tr>
</table>


Getting from there to a gait took a rebuilt rig and a few dozen more training runs. Four
points along the way on the current rig:

<table>
  <tr>
    <td align="center" width="25%" valign="top"><b>Early</b><br><img src="docs/media/stage1_r4.gif" width="180"><br><sub>Scrambles forward in a crouch</sub></td>
    <td align="center" width="25%" valign="top"><b>Collapse</b><br><img src="docs/media/stage2_r9.gif" width="180"><br><sub>Pitches onto its head</sub></td>
    <td align="center" width="25%" valign="top"><b>Upright, no gait</b><br><img src="docs/media/stage3_r13.gif" width="180"><br><sub>Balances, but shuffles instead of stepping</sub></td>
    <td align="center" width="25%" valign="top"><b>Trotting</b><br><img src="docs/media/stage4_d2.gif" width="180"><br><sub>Sequential footfalls, steady rhythm, holds a straight line</sub></td>
  </tr>
</table>


Most of the distance between the second and fourth columns was not reward tuning. It was
finding and fixing problems in the rig and the reference data that the training metrics
could not see (see [What went wrong](#what-went-wrong-and-what-fixed-it)).

## How it works

```
Malbers deer (FBX)
  -> Blender, headless: build a 38-DOF URDF from the skinned mesh
  -> Blender, headless: extract joint-angle reference clips, validated by FK round-trip
  -> URDF -> USD, collision filtering, preflight checks
  -> Isaac Lab (PhysX, 4096 parallel envs): AMP, then ASE
  -> ONNX -> Unity 6 Sentis   (planned)
```

**Rig.** The URDF is generated from the skinned model rather than hand-built, so joint
axes come from the same bone motion the reference clips are extracted from. The extractor
reconstructs every link position from the extracted angles and compares it to the source
animation; a wrong axis convention shows up as centimetres of error instead of a subtly
wrong gait.

**AMP.** A discriminator scores state transitions against the reference clips, and its
output becomes a style reward blended with a velocity-tracking task reward. The reward set
was deliberately cut down to the task terms plus a few safety terms; hand-shaped gait
rewards (air time, joint-range targets) each produced a distinct pathology and were
removed.

**ASE.** Stage 1 trains a policy conditioned on a random latent `z` on the unit sphere, with
an encoder rewarded for recovering `z` from the motion. I also implemented the paper's
diversity objective, which needed a hook inside rsl_rl's PPO update that the library does
not expose. Measured with a phase-aligned z-sweep, it increased skill separation by about
12% at matched training length without reducing the encoder's mutual information.

<p align="center">
  <img src="docs/media/ase_zsweep.gif" width="640" alt="ASE z-sweep">
</p>
<p align="center"><em>ASE stage 1: four copies of the same policy, each pinned to a
different latent.</em></p>

## What went wrong, and what fixed it

The recurring lesson was that a training curve can look healthy while the animal is doing
something wrong. Each of these was invisible in TensorBoard and found by measuring the
right thing directly, often after watching a render.

- **The reference clip was 58% frozen frames.** The style reward was teaching the policy
  to stand still. Fixed by trimming to the real keyframe range and checking velocity
  histograms of every clip before training.
- **About a million newtons of phantom self-collision** on an 809 N animal, from helper
  links that defeated PhysX's parent-child collision filtering. Found by comparing total
  contact force to body weight.
- **Mirrored hind-leg collision meshes.** The render looked perfect; the hooves had no
  contact geometry.
- **A "walk" clip that was two-thirds turning.** It was three takes chained together, two
  of them deliberate turns. Net yaw cancelled out, so a net-turn check passed it. The
  policy that trained on it veered.
- **Two of three reference clips were 1.25x oversized,** from a Blender version change that
  scales FBX imports. Joint angles are scale-invariant, so nothing looked wrong, but the
  root velocities the discriminator saw were 25% too fast. Caught by forward-kinematics
  checking that the lowest hoof actually reaches the ground.
- **Adding observations made the gait worse.** Giving the discriminator hoof height and
  hoof speed made it more confident and the style reward weaker, and foot slip got
  steadily worse with each channel added. In AMP the observation space and the reward
  scale are coupled and have to be tuned together.

The data and rig problems are now checks in `tools/rig_pipeline/preflight.py`, which is run
before every training launch and exits non-zero if any of them fail.

<p align="center">
  <img src="docs/media/failure_exp4.gif" width="300" alt="A failed policy">
</p>
<p align="center"><em>Not every run went well.</em></p>

## Evaluation

Evaluating a gait from logged rewards turned out to be unreliable, so every policy is
rolled out under a fixed command and scored on measurements of the motion itself:

- **Load share per leg**, from contact forces, to catch legs that swing but carry no weight
- **Hoof slip**: hoof speed while planted, relative to body speed, compared with the reference
- **Ankle contact share**, to catch walking on the pastern instead of the hoof
- **Stride regularity** and **heading drift**
- **Phase-aligned z-sweep separation** for ASE skills

Sixteen policies were re-evaluated under identical conditions. The current best holds a
straight line (no measurable heading drift) with regular stride timing at around 3 m/s.

## Current limitations

- The deer lands on the front of the ankle rather than the hoof on three legs.
- Every policy trots well and walks slowly badly; slow, near-static gaits are still open.
- Command tracking is loose: several policies settle at their own preferred speed.

## Next

The Malbers pack ships about 60 animation takes and the deer currently uses three. Next
is a behaviour model (grazing, drinking, idle variants, turns, reactions) built as ASE
stage 2 on top of the locomotion controller, then export to Unity.

## Stack

Isaac Sim 5.1 and Isaac Lab, PhysX on GPU, PyTorch, rsl_rl and amp-rsl-rl, Blender 5.1
(headless Python) for rig and reference extraction, Unity 6 and Sentis for deployment.
Trained on a single RTX 5080.

## Repository layout

```
training/polyart/        environment config, rewards, observations, ASE
training/scripts/        training entry point (AMP / ASE patches on amp-rsl-rl)
tools/rig_pipeline/      URDF build, reference extraction, preflight, gait analysis
tools/                   evaluation and rendering
docs/                    write-ups of individual findings
```

## Assets

The deer model and animations come from the Malbers Animations Forest Pack, which is a
paid asset and is not redistributed here. The pipeline expects a local copy; the path is
set in `configs/species/deer_polyart.json`.
