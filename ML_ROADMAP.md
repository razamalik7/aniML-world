# ML Roadmap — aniML World

A living plan. This is half a passion project, half an ML-portfolio project, so
each phase is annotated with the **ML skills / résumé value** it generates and
the **product value** it adds. We update this continuously.

Last updated: 2026-07-22. Owner: Raza.

**North star:** an emergent predator–prey ecosystem where animals *learned* to
move, hunt, and survive — with naturalistic motion learned from real wildlife
video — running in real time, and evolving over generations.

**One-sentence pitch (the cohesive product):** "Animals that learned to move,
then learned to hunt and survive together, with motion learned from monocular
video, deployed to run in real time, and evolving across generations."

---

## Where things stand (2026-07-22)

- Phase 0 (stable locomotion base): **in progress.** v5.0 stable config
  training (naturalness rewards stripped, anti-porpoise strengthened). Pivoted
  here after 26 reward-shaping iterations (v4.7–v4.33) hit the naturalness
  ceiling of pure reward engineering. See `RETROSPECTIVE_v4.md`,
  `NIGHT_LOG_2026-05-13.md`.
- Everything past Phase 0 is design + stubs today.
- Best naturalness checkpoint (parked, recoverable): v4.31r
  `logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_23-39-02/model_30396.pt`.

---

## The phases

### Phase 0 — Stable locomotion base  ← WE ARE HERE
Deer walks / turns / stops on command, robustly, **no bounce, no weave, no
fall.** Robotic is fine; solid is the bar.
- **ML skills:** deep RL (PPO), reward design, domain randomization,
  massively-parallel sim (Isaac Lab), curriculum learning, sim debugging.
- **Product value:** the foundational unit everything else stands on.
- **Done when:** vertical bounce < ~10 cm, cumulative heading drift low,
  straightness > 0.6, commanded-velocity tracking solid, low fall rate.

### Phase 1 — Locomotion as a reusable primitive + hierarchical control
Freeze the walk policy as a **velocity/heading-conditioned low-level
controller**; add a high-level policy that outputs *where to go*, not torques.
This decoupling is the single most important architectural move — it makes
everything above it about *behavior*, not motor control.
- **ML skills:** hierarchical RL, goal-conditioned policies, options framework.
- **Product value:** the interface the ecosystem brain plugs into. Also the
  seam that lets us later swap the low-level controller (Phase 4) without
  touching behavior.
- **Design rule:** keep the policy I/O interface **engine-agnostic** from here
  on, so the eventual Unity port is a port, not a rewrite.

### Phase 2 — Single-animal ecology brain
Deer: navigate to food, eat, flee threats, rest. Wolf: detect, pursue, catch.
Design the sensory observations (raycasts, vision/scent proxies) and survival
rewards.
- **ML skills:** multi-task / goal-conditioned RL, observation engineering,
  reward design for emergent behavior.
- **Product value:** individual animals that *do things* for reasons.

### Phase 3 — Multi-agent ecosystem  ← THE MVP
N deer + M wolves in one scene. Predator–prey dynamics **emerge** rather than
being scripted: herds, chases, population oscillations.
- **ML skills:** multi-agent RL (MARL), self-play, population-based training,
  emergent dynamics.
- **Product value:** the demonstrable "wow." The natural first impressive
  deliverable and a strong interview centerpiece.
- **Done when:** a rendered clip shows believable emergent predator–prey
  behavior with animals that don't fall over.

### Phase 4 — Naturalness upgrade (imitation + video-extracted motion)
Swap the robotic low-level controller for one trained with **AMP → skill
embeddings (ASE)**, fed by **motion extracted from monocular wildlife video**.
Behavior layer above it is unchanged — it just commands a prettier body.
- **ML skills:** adversarial imitation learning, latent skill spaces, monocular
  animal pose estimation (computer vision), motion retargeting.
- **Product value:** turns a functional demo into a *beautiful* one; turns "we
  have no animations" into a headline feature.
- **Feasibility notes:** see `docs/RESEARCH_NOTES.md`. Wolf is the easier case
  (maps to the dog-validated BARC model); deer is harder for 3D lifting.

### Phase 5 — Deployment + showcase (the cohesive piece)
ONNX export → Unity Sentis, LOD/batched inference so dozens of agents run in
real time, a watchable/playable demo. Then genetics + heritable traits as the
flourish (emergent selection, fast-forward generations).
- **ML skills:** model export/optimization, real-time inference at scale,
  open-ended evolution.
- **Product value:** ships the thing. The deployment story is itself a strong
  résumé line (real-time ML at scale).

---

## Open decisions (revisit as we go)

- **MVP demo home:** Isaac Lab (fast, all-Python, where the ML lives) vs. Unity
  (prettier, matches original vision, real deployment work). Current lean:
  **build + validate the MVP in Isaac Lab; treat Unity as a separate Phase-5
  productization milestone.** Keep policy I/O portable from Phase 1 so the port
  isn't a rewrite. (See RESEARCH_NOTES.)
- **Naturalness data source:** bought animation packs (rejected — look robotic,
  hard to find good ones) vs. **video-to-motion extraction (preferred).**
- **Deer vs. wolf first** for the video pipeline: wolf is technically easier
  (dog-validated models) — possibly prototype the pipeline on wolf.

## Bang-for-buck

Phases 0–3 = the complete MVP with the broadest ML surface (RL, hierarchical
RL, MARL). Phases 4–5 = the flashy upgrades (imitation, CV, deployment,
evolution). The MVP stands alone without 4–5; do those for passion + extra
résumé punch.
