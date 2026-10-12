# aniML World — Architecture Reference

> **Note (2026-10):** Section 1 (version pins, WSL2/Docker setup) predates the current
> native-Windows Isaac Sim 5.1 / Isaac Lab setup and is out of date. The ecosystem design
> sections are current.


> Master reference for all AI models working on this project.
> This file is the single source of truth for version pins, tensor layouts, API patterns, and architectural decisions.

## 1. Version Pins

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10 | Isaac Lab requirement |
| Isaac Sim | 4.5+ | Check compatibility matrix |
| Isaac Lab | 2.3+ | `pip install isaaclab` |
| rsl_rl | 2.3+ | `pip install isaaclab_rl` (ETH RSL fork bundled with Isaac Lab) |
| PyTorch | 2.5+ | CUDA 12.x, ships with Isaac Sim |
| ONNX opset | 17 | Sentis 2.5 compatible |
| Unity | 6 (6000.x) | URP pipeline |
| Unity Sentis | 2.5+ | Package: `com.unity.ai.inference` |
| .NET | C# 9+ | Unity 6 default |

### Isaac Lab ↔ Isaac Sim Compatibility

Isaac Lab versions are tightly coupled to Isaac Sim versions. Always check:
https://isaac-sim.github.io/IsaacLab/latest/source/setup/installation/index.html

### WSL2 / Docker Setup (Windows)

Isaac Lab runs on Linux. On Windows, use WSL2 + Docker:
- Base image: `nvcr.io/nvidia/isaac-sim:4.5.0` (check NVIDIA NGC for latest)
- NVIDIA Container Toolkit required for GPU passthrough
- Minimum NVIDIA driver: 570+
- WSL2 with Ubuntu 22.04 or 24.04

---

## 2. Observation Space (Full Specification)

### Locomotion Phase (Phase 1) — 148 dimensions

| Field | Indices | Dim | Source | Notes |
|-------|---------|-----|--------|-------|
| Joint positions (relative) | [0:12] | 12 | `mdp.joint_pos_rel` | 4 legs × 3 joints |
| Joint velocities (relative) | [12:24] | 12 | `mdp.joint_vel_rel` | rad/s |
| Body orientation | [24:28] | 4 | `mdp.root_quat_w` | Quaternion (w,x,y,z) |
| Body angular velocity | [28:31] | 3 | `mdp.base_ang_vel` | Body frame |
| Gravity projection | [31:34] | 3 | `mdp.projected_gravity` | Gravity in body frame |
| Velocity command | [34:37] | 3 | Command manager | (vx, vy, yaw_rate) |
| Body linear velocity | [37:39] | 2 | `mdp.base_lin_vel` | vx, vy in body frame |
| Last action | [39:48] | 9 | `mdp.last_action` | For action smoothness |
| Height scan raycasts | [48:148] | 100 | Custom raycast term | 10×10 grid under body |

### Ecosystem Phase (Phase 2+) — 159 dimensions

Appends to locomotion observation:

| Field | Indices | Dim | Source | Notes |
|-------|---------|-----|--------|-------|
| Context tokens | [148:159] | 11 | World Director | Multi-hot encoded |

### Future Extensions

If adding inter-agent observations (Phase 2), insert BEFORE context tokens:

| Field | Proposed Indices | Dim | Notes |
|-------|-----------------|-----|-------|
| Nearest 3 agents relative pos | [148:157] | 9 | 3 agents × (dx, dy, dz) |
| Nearest 3 agents relative vel | [157:166] | 9 | 3 agents × (dvx, dvy, dvz) |
| Agent type one-hot | [166:171] | 5 | Species encoding per neighbor |
| Context tokens | [171:182] | 11 | Shifted to end |

This would make total obs dim = 182.

---

## 3. Action Space

| Field | Indices | Dim | Type | Range |
|-------|---------|-----|------|-------|
| Joint position targets | [0:12] | 12 | Continuous | [-1, 1] scaled by PD controller |

**Joint ordering:**
- [0] FL_hip_yaw, [1] FL_hip_pitch, [2] FL_knee
- [3] FR_hip_yaw, [4] FR_hip_pitch, [5] FR_knee
- [6] RL_hip_yaw, [7] RL_hip_pitch, [8] RL_knee
- [9] RR_hip_yaw, [10] RR_hip_pitch, [11] RR_knee

Actions are joint position targets fed to a PD controller (not raw torques).
The PD controller computes: `torque = Kp * (target - current) - Kd * velocity`

---

## 4. Reward Functions

### Phase 1: Locomotion

| Reward Term | Weight | Function | Purpose |
|-------------|--------|----------|---------|
| Velocity tracking (lin) | 1.0 | exp(-error²/0.25) | Match commanded velocity |
| Velocity tracking (ang) | 0.5 | exp(-error²/0.25) | Match commanded yaw rate |
| Body height | -2.0 | \|h - h_target\|² | Stay at target height |
| Orientation penalty | -1.0 | \|roll\|² + \|pitch\|² | Stay upright |
| Energy penalty | -0.001 | Σ\|torque × velocity\| | Efficiency |
| Feet air time | 0.5 | Σ(air_time - 0.5)+ | Encourage gait cycling |
| Action rate | -0.01 | \|a_t - a_{t-1}\|² | Smooth actions |
| Feet stumble | -1.0 | Contact force > threshold | Penalize collisions |

### Phase 2: Ecosystem (additional)

| Reward Term | Weight | Species | Purpose |
|-------------|--------|---------|---------|
| Survival bonus | 0.1/step | All | Stay alive |
| Forage (eat) | 2.0 | Herbivore | Reach food source |
| Flee predator | 3.0 | Herbivore | Distance from wolves |
| Catch prey | 5.0 | Predator | Contact with deer |
| Pack proximity | 0.5 | Predator | Stay near other wolves |

---

## 5. PPO Training Configuration (rsl_rl)

Reference implementation from Isaac Lab ANYmal-B:

```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class LocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "deer_locomotion_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

**VRAM-safe settings for RTX 5070 (12GB):**
- `num_envs`: 1024 (safe) or 2048 (test first)
- NOT 4096 — that typically requires 24GB+

---

## 6. ONNX Export Specification

Exported models go to `models/exported/`.

**Naming convention:** `<species>_<phase>_v<version>.onnx`
- Example: `deer_locomotion_v1.onnx`, `wolf_ecosystem_v3.onnx`

**Required ONNX properties:**
- Input name: `"observations"`, shape: `[batch, obs_dim]` (dynamic batch)
- Output name: `"action_mean"`, shape: `[batch, action_dim]` (dynamic batch)
- Opset version: 17
- Only export the actor (policy) network, NOT the critic

**Export code:**
```python
import torch

def export_policy(actor_model, obs_dim, action_dim, path):
    actor_model.eval()
    dummy_input = torch.randn(1, obs_dim)
    torch.onnx.export(
        actor_model,
        dummy_input,
        path,
        input_names=["observations"],
        output_names=["action_mean"],
        dynamic_axes={
            "observations": {0: "batch"},
            "action_mean": {0: "batch"},
        },
        opset_version=17,
    )
```

---

## 7. Unity Sentis Integration

### Package and Namespace

```csharp
using Unity.InferenceEngine;  // Sentis 2.5+ namespace
```

### Loading and Running a Model

```csharp
ModelAsset modelAsset;  // Drag ONNX file in Inspector
Model runtimeModel = ModelLoader.Load(modelAsset);
Worker worker = new Worker(runtimeModel, BackendType.GPUCompute);

// Create input tensor
var input = new Tensor<float>(new TensorShape(batchSize, OBS_DIM));
// ... fill input data ...

// Run inference
var output = worker.Schedule(input).PeekOutput() as Tensor<float>;

// Read results (blocking)
var cpuOutput = output.ReadbackAndClone();

// ALWAYS clean up
input.Dispose();
worker.Dispose();
```

### Batching Strategy

All agents in the same LOD zone are batched into a SINGLE forward pass:
1. Collect observations from all Zone 1 agents into one `float[,]` array
2. Create one `Tensor<float>` with shape `[numZone1Agents, OBS_DIM]`
3. Run one `worker.Schedule()` call
4. Scatter action outputs back to individual agents

NEVER run per-agent inference — this defeats GPU parallelism.

---

## 8. LOD AI System

**Critical distinction:** **Rendering LOD** (what the player sees) is **independent** from **ecological simulation fidelity** (whether animals hunt, flee, reproduce, and change the world). Distant or unrendered agents may use **cheaper** inference and **coarser** physics, but they must still **advance plausible ecosystem state** so that when the player arrives, the world reads as if life continued off-camera. See **Section 13**.

### Zone Classification (inference + physics scheduling near the camera)

These zones govern **how often** full neural control and **full** articulation run for agents the engine is actively stepping. They do **not** mean “far animals stop living.”

```
Zone 1: distance < 50m    → Full policy, every frame; full physics
Zone 2: 50m ≤ d < 150m   → Full or distilled policy, lower rate + interpolation; full/light physics
Zone 3: d ≥ 150m         → Coarse policy OR abstract/macro sim tick (see §13); simplified physics optional
Off-screen / unrendered  → Abstract ecological layer + batched macro updates (see §13); NOT “frozen world”
```

### Rendering LOD (visual only)

| Zone | Rendering | Animation |
|------|-----------|-----------|
| 1 | Full mesh | Animator (skeletal) |
| 2 | Reduced mesh | Vertex Animation Textures (GPU instanced) |
| 3 | Billboard | Baked sprite frames |

GPU Resident Drawer does NOT support Animator components. Do not attempt to use it for animated creatures.

**Rule:** Never equate “not drawn” with “not simulated.” Only **cost** changes off-screen; **causal continuity** (predation, reproduction, resource use, terrain/world flags) is a **product requirement**.

---

## 9. World Director Protocol

Rule-based system. NOT a neural network.

### Context Token IDs

```
0=CLEAR_SKY  1=LIGHT_RAIN  2=HEAVY_RAIN  3=THUNDERSTORM
4=METEOR_INCOMING  5=METEOR_IMPACT  6=NIGHT  7=DAWN
8=DUSK  9=DROUGHT  10=WILDFIRE
```

Multi-hot encoding. Multiple tokens can be active simultaneously.

### Event Trigger Rules

| Event | Trigger Condition | Duration |
|-------|------------------|----------|
| Weather transitions | Markov chain, 60s intervals | Until next transition |
| Meteor | Random, ~1% per sim-hour | INCOMING: 5s, IMPACT: 2s |
| Day/Night | Time-of-day cycle | Continuous |
| Drought | Herbivore pop > 80% capacity | Until pop drops below 50% |
| Wildfire | Lightning (thunderstorm) + drought | 30-120s |

---

## 10. Domain Randomization (Sim-to-Sim Transfer)

Applied during Isaac Lab training to ensure policies transfer to Unity:

| Parameter | Range | Method |
|-----------|-------|--------|
| Ground friction | [0.5, 1.5] | Uniform per reset |
| Body mass | ±15% | Uniform per reset |
| Motor strength (Kp) | ±10% | Uniform per reset |
| Motor damping (Kd) | ±10% | Uniform per reset |
| Observation noise | ±0.01 | Gaussian per step |
| Action delay | 0-2 steps | Uniform per reset |
| Gravity | ±5% | Uniform per reset |
| Push perturbation | 0-50N | Random impulse every 5-10s |

---

## 11. File Naming Conventions

- Python: `snake_case.py` (e.g., `locomotion_env.py`)
- C#: `PascalCase.cs` (e.g., `AgentBrain.cs`)
- YAML configs: `snake_case.yaml` (e.g., `deer.yaml`)
- ONNX models: `<species>_<phase>_v<version>.onnx`
- USD assets: `<species>_<variant>.usd`

---

## 12. Directory Structure

```
aniML_world/
├── ARCHITECTURE.md              ← YOU ARE HERE
├── README.md
├── pyproject.toml
├── requirements.txt
├── .cursor/rules/               ← Cursor AI rules (auto-injected)
│
├── configs/
│   ├── species/
│   │   ├── deer.yaml
│   │   └── wolf.yaml
│   ├── training/
│   │   ├── locomotion_rough.yaml
│   │   └── ecosystem.yaml
│   └── environment.yaml
│
├── training/                    ← Isaac Lab (Python)
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── locomotion_env_cfg.py
│   │   └── ecosystem_env_cfg.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── deer_cfg.py
│   │   └── wolf_cfg.py
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── locomotion_rewards.py
│   │   └── ecosystem_rewards.py
│   ├── terrain/
│   │   ├── __init__.py
│   │   └── procedural_terrain.py
│   └── scripts/
│       ├── train.py
│       └── export_onnx.py
│
├── models/
│   ├── __init__.py
│   ├── networks.py              ← ActorCritic MLP definition
│   └── exported/
│       └── .gitkeep
│
├── world_director/
│   ├── __init__.py
│   ├── context_tokens.py
│   ├── event_system.py
│   └── weather.py
│
├── shared/
│   ├── __init__.py
│   ├── schemas.py               ← Python source of truth
│   └── schemas.json             ← Auto-generated for Unity
│
├── unity/
│   └── Runtime/
│       ├── AI/
│       │   ├── BatchedPolicyInference.cs
│       │   ├── AgentBrain.cs
│       │   └── AILODManager.cs
│       ├── Simulation/          ← off-screen ecological continuity (§13)
│       │   ├── EcologicalRegion.cs
│       │   ├── AbstractEcosystemTick.cs
│       │   ├── SimulationHydrator.cs
│       │   └── ContinuityLedger.cs
│       ├── Environment/
│       │   ├── TerrainManager.cs
│       │   ├── WeatherSystem.cs
│       │   └── MeteorSystem.cs
│       ├── WorldDirector/
│       │   └── EventBroadcaster.cs
│       └── Creatures/
│           ├── CreatureController.cs
│           ├── DeerController.cs
│           └── WolfController.cs
│
├── tools/
│   ├── vram_budget.py
│   └── benchmark.py
│
└── docker/
    └── Dockerfile.isaac
```

---

## 13. Off-Screen and Far-Field Ecological Continuity

### Product requirement (user-facing)

Animals outside the player’s view (or far from the camera) must still **produce consequences**: hunting, being hunted, reproduction, movement that affects **logical** world state (e.g. trampling, carcass placement, resource depletion), and responses to World Director events. When the player **moves into** a region, what they see should be **consistent** with that ongoing simulation—as if those animals had been **living there the whole time**, not spawned from a static snapshot.

This is analogous to **simulating a sports season or match** in a management game: the engine produces **plausible outcomes and state** while the player is elsewhere; when they “watch” a game, stats and positions align with what the sim already decided. **aniML World** needs a **richer** version of that idea: not a single scoreline, but **spatially grounded** populations, poses, trails of interaction, and environment deltas.

### Design principle: two axes

| Axis | Question | Options |
|------|----------|---------|
| **Presentation** | Is the agent drawn with full skeletal animation? | Rendering LOD (§8) |
| **Fidelity** | Does this agent’s behavior still affect the ecosystem truthfully? | Always **yes**; implementation tier varies |

### Implementation tiers (recommended direction)

These can be combined per **region** or **cell** (spatial partitioning):

1. **Player bubble (near camera)**  
   Full physics, full (or throttled) neural policy, full animation. Source of truth for **fine** motion.

2. **Mid field (rendered but far)**  
   Same **species policies** at lower frequency, **distilled** smaller policy on **compressed observations**, or simplified articulation (e.g. root motion + procedural legs). Physics may be **light** (capsule + steering) while preserving **contacts** that matter (e.g. kill range, feeding).

3. **Abstract / off-screen layer**  
   No per-frame full ragdoll required. Advance **macro state** on a **sim clock** (e.g. 1–5 Hz or event-driven):
   - **Parcel or cell** state: herbivore/predator counts, energy, rough positions, “last conflict” flags, pregnancy/reproduction timers, carcass/resource nodes.
   - **Transitions**: learned or hand-authored **macro MDP** (who eats whom, who flees, birth/death rates) **calibrated** against full sim or offline rollouts so bulk statistics match near-field behavior.
   - **Neural optional**: small networks predicting **distribution** of outcomes (e.g. encounter resolution) from macro features—still **actions with effects**, not only a static population model.

4. **Handoff when the player enters a region**  
   **Hydrate** full agents from abstract state: spawn or wake bodies at **consistent** positions, carry over **vitals** (health, hunger, group id), apply **short stabilization** (pose settle, clamp penetration) so physics does not explode. Optionally run **1–2 seconds** of warm-up with **high-level goals** fixed (flee, rest) before handing control to the full policy.

### What to document in code (future modules)

Suggested Unity-side concepts (names indicative):

- `EcologicalRegion` / spatial hash: which agents are in abstract vs full mode.
- `AbstractEcosystemTick`: advances off-screen parcels (predation, reproduction, resources).
- `SimulationHydrator`: player enters cell → instantiate/enable full `AgentBrain` + physics from saved macro + meso state.
- `ContinuityLedger`: append-only or periodic log of **world-affecting** events (kills, births, major moves) for replay or debug.

Python/Isaac side: train policies in **full** fidelity; optionally train **distillation** targets or **macro models** using logged trajectories from full multi-agent episodes.

---

## 14. Future Improvements (Non-Binding Roadmap)

Items discussed for **behavioral depth** and **reactivity** that are **not** required for v1 but align with the original vision:

### Architecture and learning

- **Hierarchical control:** Small **locomotion** policy (joint targets) + separate **high-level** head or module (modes: stalk, chase, rest, herd) outputting velocity commands or goal vectors. Improves **role diversity** without scaling a single flat MLP to hundreds of millions of parameters.
- **Richer interaction observations:** Fixed **K nearest agents** slots (relative position, velocity, species one-hot, optional “attention” over K—still compatible with small nets if K is bounded).
- **Centralized training, decentralized execution:** During training only, extra info for critics or value baselines; execution stays local per animal for deployment.
- **Policy distillation:** Train a **smaller** or **lower-rate** student to mimic the full policy for Zone 2/3 so distant animals stay **neurally aligned** with near-camera behavior (better than generic FSM if budget allows).

### Sensing and World Director

- **Engineered social/anim features:** Expose compact signals from animation/state machines (dominance, submissive pose, alert stance) into observations for **body language** without full vision.
- **Optional vision later:** Image encoder for stalking, occlusion, camouflage—adds **texture/line-of-sight** nuance; keep policies small by **freezing** a pretrained backbone or bottlenecking to a low-dim embedding.
- **Optional high-level language model (later):** A **small** LLM or VLM used only to propose **scenario tags** or narrative context, still exported as **context tokens** consumed by the same small policies—not one giant VLA per deer.

### Action representation

- **FAST / OAT revisit:** Only if adding **language-conditioned** or **tokenized high-level** plans; **not** the default for raw joint control in sim.

### Off-screen fidelity

- **Learned macro models:** Train neural or hybrid models so **abstract-layer** outcomes match **full sim** statistics (predation rate, herd dispersion) under stress tests.
- **Deterministic replay:** For debugging continuity, log abstract events and replay when entering a cell (similar to sports sim “box score” + underlying event list).

### Evolution & genetics (see §16–§18 for current design)

- **Speciation detection:** When genetic distance between sub-populations exceeds a threshold, flag a divergence event (proto-speciation). Visual + behavioral differences would already be apparent from phenotype mapping.
- **Epigenetics / lifetime adaptation:** Traits partially modified by an agent's experiences (e.g. a deer that survived many wolf encounters passes a slight alertness bonus). Lamarckian flavour, biologically hand-wavy but gameplay-interesting.
- **Sexual dimorphism genes:** Phenotype mapping conditioned on a sex gene, enabling size/appearance differences within a species.
- **Player-driven selective breeding:** Allow the player to “tag” preferred individuals and increase their mating probability — artificial selection sandbox mode.

---

## 15. Behavioral Quality Levers (Design, Not Model Size)

For **natural motion**, **predator–prey**, and **social dynamics**, prioritize:

1. **Observation design** — What each agent knows about neighbors, threats, and resources (structure beats parameter count).
2. **Multi-agent training** — Curriculum from locomotion → pairwise interactions → full ecosystem; diverse opponents.
3. **Reward shaping and sparsification** — So chase/catch/escape/social terms are learnable.
4. **Hierarchy** — Separate low-level motor control from high-level intent when flat policies plateau.

Hardware constraints are secondary to these for **intended project quality**; they only affect **how many** parallel training envs or **how often** full inference runs, not the **design** of the above.

---

## 16. Genetics, Evolution & Reproduction

### Design goal

Natural selection should **emerge** from the simulation: agents with traits that help them survive and reproduce pass those traits to offspring; maladapted lineages die out. Over many in-sim generations the player should observe populations visibly shifting — faster deer, larger wolves, coat-color drift, etc.

### 16.1 Genome Representation

Each agent carries a **genome**: a fixed-length float vector of heritable trait genes.

```
Genome (per agent):  float[NUM_GENES]   (proposed NUM_GENES = 16–32)
```

| Gene Group | Genes (examples) | Dimension | Phenotype Effect |
|------------|-------------------|-----------|------------------|
| Morphology | body_mass_scale, limb_length_ratio, torso_length | 3 | Scales USD skeleton joint offsets and physics mass |
| Locomotion | max_speed_mult, stride_frequency, turn_agility | 3 | Multiplies PD Kp/Kd and action scaling |
| Metabolism | stamina_pool, hunger_rate, energy_efficiency | 3 | Governs energy drain/gain curves |
| Sensory | vision_range, hearing_range, smell_range | 3 | Modulates observation mask radii (nearest-K cutoff, raycast range) |
| Reproduction | maturity_age, gestation_length, litter_size_bias, mating_drive | 4 | Timers and thresholds for reproduction FSM |
| Appearance | coat_hue, coat_value, pattern_id, antler_scale (deer) | 4+ | Drives material property blocks / vertex color in Unity |
| Species-specific | (reserved) | 0–8 | Wolf: jaw_strength, pack_cohesion; Deer: alertness_bias, sprint_burst |

**Encoding:** Each gene is a float in **[0, 1]** mapped to a species-specific phenotype range via a config table (`configs/species/<species>.yaml`). Example: `body_mass_scale` gene value 0.0 → 80 % of base mass, 1.0 → 120 % of base mass.

### 16.2 Trait ↔ Domain Randomization Synergy

The domain randomization ranges already in §10 (mass ±15 %, friction, Kp/Kd, gravity, etc.) were chosen so **trained policies generalise** over physical variation. The genome simply **replaces uniform-random sampling** with **inherited values** at spawn time. No policy retraining is needed as long as gene ranges stay within (or close to) the DR envelope. If a gene range is expanded beyond DR bounds, retrain with the wider range.

### 16.3 Reproduction Mechanics

Reproduction is driven by the **ecosystem simulation** (both near-field and abstract layer §13):

| Stage | Condition | Timer / Cost | Where Simulated |
|-------|-----------|-------------|-----------------|
| **Eligibility** | Agent age ≥ genome `maturity_age`; energy > 60 % | — | All tiers |
| **Mate search** | Two eligible same-species agents within `mating_range` | Courtship: 5–15 s sim-time | Zone 1–2: full policy; Zone 3+/abstract: probabilistic roll |
| **Mating event** | Courtship completed; energy cost deducted from both parents | Energy cost ≈ 20 % each | All tiers (logged in ContinuityLedger) |
| **Gestation** | Mother carries pregnancy timer (genome `gestation_length`) | Species default 30–90 s sim-time | All tiers |
| **Birth** | Timer expires → 1–N offspring spawned (bias from genome) | Energy cost ≈ 30 % | Near-field: instantiate; abstract: add to parcel counts |
| **Offspring genome** | Crossover + mutation (see §16.4) | — | All tiers |

**Mating selection heuristic:** Prefer mates with complementary or high-fitness traits. Simple implementation: weighted random among eligible neighbours, with probability proportional to a **mate score** (configurable per species — e.g. wolves may prefer larger packmates, deer may prefer fastest neighbours).

### 16.4 Inheritance: Crossover & Mutation

```
offspring_genome = crossover(parent_A.genome, parent_B.genome) + mutation
```

| Operator | Method | Parameters |
|----------|--------|------------|
| **Crossover** | Uniform crossover: per-gene, pick parent A or B with 50 % probability | — |
| **Mutation** | Gaussian perturbation: `gene += N(0, σ_mut)`, clamped to [0, 1] | `σ_mut` ≈ 0.02–0.05 (configurable per gene) |
| **Rare macro-mutation** | With probability `p_macro` ≈ 0.5 %, one random gene resampled uniformly | Introduces novelty, prevents convergence stagnation |

Crossover and mutation rates are defined in `configs/training/evolution.yaml` and tunable per species.

### 16.5 Selection Pressure (Emergent)

No explicit fitness function is evaluated. Selection pressure arises **naturally** from the simulation loop:

1. **Survival selection** — Agents that die (starved, predated, environmental) before reproducing contribute zero offspring. Traits that improve survival (speed for deer, strength for wolves) propagate.
2. **Fecundity selection** — Agents that reproduce more often pass genes at higher rates. Metabolism genes that allow efficient energy use → more reproduction events.
3. **Balancing feedback** — Over-specialization is checked by ecosystem dynamics: if all deer become very fast, wolves starve and deer overpopulate → food scarcity → starvation selects for efficient metabolism over raw speed.

### 16.6 Tracking Lineage & Population Genetics

- Each agent has a unique `agent_id` and stores `parent_ids[2]`.
- `ContinuityLedger` (§13) logs birth/death events with genome snapshots.
- **Population stats** computed periodically (every N sim-seconds): mean/variance of each gene across the living population, per species. Exposed to the player as optional ecology overlay / graphs.
- Optional: **species richness index**, **genetic diversity** (heterozygosity) metric for World Director to trigger events (e.g. inbreeding depression debuff when diversity drops below threshold).

### 16.7 Observation Space Extension for Genetics

Genetic traits that affect **what the agent perceives or how it acts** are folded into existing observation/action pipelines:

- **Sensory genes** modify observation mask radii (e.g. `vision_range` gene scales the nearest-K agent detection distance). The obs vector size stays the same; fewer or more neighbours fall within range.
- **Physical genes** are already captured by the physics engine (changed mass, Kp, Kd). The policy sees their effects through joint velocity / body velocity observations.
- **No genome vector in the obs space.** Agents don't "know" their own DNA — they experience its phenotypic consequences through physics and sensory limits.

### 16.8 Data Flow Summary

```
Birth event
  → crossover(parentA.genome, parentB.genome) + mutation
  → offspring.genome
  → phenotype_map(offspring.genome, species_config)
  → set physics params (mass, Kp, Kd, limb scale)
  → set sensory params (detection radii)
  → set appearance params (material property block)
  → agent enters simulation with same trained policy (generalised via DR)
```

---

## 17. Time Control & Fast-Forward

### Design goal

Allow the player (and developers during testing) to **accelerate simulation time** so that ecological and evolutionary dynamics play out over minutes of real time rather than hours. A deer generation that takes ~90 s sim-time at 1× should resolve in ~9 s at 10× — enabling the player to watch populations evolve, herds migrate, and seasons turn.

### 17.1 Speed Tiers

| Tier | Multiplier | Use Case | Rendering | AI Tier |
|------|-----------|----------|-----------|---------|
| **Pause** | 0× | Inspect, plan | Last frame frozen | No ticks |
| **Normal** | 1× | Gameplay | Full (§8) | Full LOD zones |
| **Fast** | 2–5× | Watch a hunt play out, skip rain | Reduced framerate target (30 fps visual), skip particles | Zone 1 shrinks to 25 m; more agents abstract |
| **Very Fast** | 5–20× | Generational observation | Minimal: top-down or stylized overview cam; agents as dots/icons | Most agents abstract-tier; only player-focused cell gets Zone 1 |
| **Ultra / Headless** | 20–100× | Dev testing, overnight evolution runs | Optional: off or 1 fps snapshot | All abstract / macro sim |

### 17.2 Implementation Strategy (Unity Side)

**Do NOT simply set `Time.timeScale` to 20.** Physics and animation break at high multipliers. Instead:

1. **Sim-tick multiplier (`simTicksPerFrame`):** Each `Update` frame, run the ecosystem simulation loop N times before rendering. Physics sub-steps remain at fixed dt (e.g. 1/60 s) but are batched.
2. **Adaptive LOD demotion:** As speed increases, aggressively push agents to cheaper tiers. At 10×+, only the camera-focused cell retains Zone 1; everything else is abstract-layer macro ticks.
3. **Rendering budget cap:** At Fast+ tiers, cap GPU work: disable shadows, reduce draw distance, switch all agents to billboard or dot representation. Optionally offer a **data view** (map overlay with population heat-maps, lineage trees) instead of 3D rendering.
4. **Audio:** Pitch-shift ambient sounds up to ~2×; above that, crossfade to a simplified time-lapse soundscape or mute.

### 17.3 Simulation Consistency at High Speed

- **Fixed-dt accumulator:** Sim advances in fixed `sim_dt` increments regardless of render framerate. At high speed, many sim steps per render frame.
- **Event ordering:** World Director events, births, deaths, and state transitions must process in **sim-time order** even when batched. Use a priority queue keyed on sim-time.
- **Determinism (optional):** If seeded RNG is used throughout, a fast-forward run produces the **same** result as real-time — useful for debugging and replay.

### 17.4 Player Interaction During Fast-Forward

| Speed | Player Can... |
|-------|--------------|
| Pause | Pan camera, inspect any agent stats/genome, read ledger |
| 1× | Full gameplay (move, observe, tag animals) |
| 2–5× | Pan camera, click agents for info; no precise interactions |
| 5–20× | Observe data overlays, population graphs; camera auto-follows selected herd or region |
| 20×+ | Dev/sandbox only; headless or minimal HUD |

Pressing any movement key or interaction key auto-drops speed to 1×.

### 17.5 Evolution-Friendly Defaults

For a player who wants to "watch evolution happen":

- **Suggested flow:** Set speed to 10×, enable population-stats overlay, select a species. Watch trait distributions shift over 5–10 generations. Slow to 1× to observe the resulting animals in detail.
- **Generation counter:** UI shows current average generation number per species (incremented at birth based on max parent generation + 1).
- **Trait drift graph:** Real-time line chart of population mean for key genes (speed, mass, sensory range) over sim-time. Visible at any speed tier.

---

## 18. Directory & Config Additions for Evolution / Time Control

New files implied by §16–§17:

```
configs/
  └── training/
      └── evolution.yaml            ← mutation rates, crossover params, gene ranges per species

shared/
  ├── genome_schema.py              ← Gene layout, phenotype mapping functions (Python source)
  └── genome_schema.json            ← Auto-generated for Unity

unity/
  └── Runtime/
      ├── Genetics/
      │   ├── Genome.cs             ← Genome struct, crossover, mutation operators
      │   ├── PhenotypeMapper.cs    ← Genome → physics/visual params
      │   ├── ReproductionManager.cs← Mating eligibility, gestation, birth spawning
      │   └── PopulationStats.cs    ← Per-species gene distribution tracking
      ├── Simulation/
      │   └── (existing files updated: AbstractEcosystemTick handles reproduction + genetics)
      └── TimeControl/
          ├── SimulationClock.cs    ← Sim-tick multiplier, fixed-dt accumulator
          ├── TimeScaleManager.cs   ← Speed tier switching, LOD demotion triggers
          └── EvolutionOverlayUI.cs ← Population graphs, generation counter, trait drift

training/
  └── evolution/
      ├── __init__.py
      └── genome_utils.py           ← Genome ops for Isaac Lab (spawn with inherited traits)
```

---

## 18. Asset Pipeline — Rigged Mesh → Physics-Ready USD

### Problem

Rigged FBX/blend models from artists or asset stores have **no PhysX physics**
(no ArticulationRootAPI, no rigid bodies, no collision). Isaac Lab requires all
of these. Manually adding them per bone per species doesn't scale.

### Solution: Automated Blender → URDF → Isaac Pipeline

```
Artist FBX ──► Blender (.blend) ──► rig_to_urdf.py ──► URDF + STL meshes
                                         │
                                    species config
                                    (deer.json)
                                         │
                   Isaac Sim URDF Import ◄┘
                         │
                    Physics-ready USD ──► Isaac Lab training
```

**Per-species work** (5–15 min after first setup):
1. Import FBX into Blender, save as `.blend`
2. Write a species JSON config mapping bone names → standardized link roles
3. Run one command (headless Blender)
4. Import URDF in Isaac Sim
5. Train

### Key Files

| File | Purpose |
|------|---------|
| `configs/species/<species>.json` | Bone-name mapping, joint limits, mass |
| `tools/blender_urdf_pipeline/rig_to_urdf.py` | Blender Python: vertex-weight convex-hull collision + URDF |
| `tools/isaac_windows/Run-BlenderToUrdf.ps1` | Headless Blender launcher (Windows) |

### Species Config Format

```json
{
  "species": "deer",
  "mesh_object": "Deer_001",
  "armature_object": "Deer_001_rig",
  "total_mass_kg": 70.0,
  "collision_method": "convex_hull",
  "torso_bones": ["Root", "spine.003", "..."],
  "legs": {
    "HR": { "hip": "shoulder.R", "thigh": "thigh.R", "shin": "shin.R", "foot": "foot.R" },
    "...": "..."
  },
  "joint_defaults": {
    "hip":   { "axis": [1,0,0], "lower_deg": -60,  "upper_deg": 60,  "effort": 100 },
    "thigh": { "axis": [1,0,0], "lower_deg": -90,  "upper_deg": 90,  "effort": 80  },
    "shin":  { "axis": [1,0,0], "lower_deg": -120, "upper_deg": 10,  "effort": 60  },
    "foot":  { "axis": [1,0,0], "lower_deg": -45,  "upper_deg": 45,  "effort": 40  }
  }
}
```

### URDF Link Naming Convention

The pipeline produces standardized names that match the training env's body
regex patterns out of the box:

- `base` — torso (matches `ANIML_DEER_BODY_BASE`)
- `{HR,HL,FR,FL}_hip` — hip links
- `{HR,HL,FR,FL}_thigh` — upper leg (matches thigh regex)
- `{HR,HL,FR,FL}_shin` — lower leg
- `{HR,HL,FR,FL}_foot` — foot (matches foot regex)

### Collision Strategy

For each link, the script collects all mesh vertices where the mapped bone(s)
have the **highest vertex weight** (above a configurable threshold). It then
computes the **convex hull** of those vertices and exports it as a binary STL.
This produces collision shapes that **follow the actual mesh silhouette** per
body segment — much more accurate than hand-placed capsules, fully automated.

### Isaac Sim URDF Import Settings

After generating the URDF, import in Isaac Sim:
- File → Import → URDF
- **Fix Base Link** = unchecked (floating base for locomotion)
- **Joint Drive Type** = None (Isaac Lab's actuator model handles this)
- **Self Collision** = checked
- Save the resulting USD — this is your training asset

---

## 19. Player Possession (Animal Takeover)

*Added 2026-07-23 (Unity planning session). Design feature: the player can take
direct control of any animal in the world.*

### Why it is nearly free under this architecture

1. **Policies are command-conditioned.** The observation vector carries a
   velocity command slot (vx, vy, yaw-rate — see §2) that training randomizes.
   Possession = routing player input (WASD / stick) into that slot instead of
   the decision layer's output. The locomotion policy executes the command with
   learned legs — no new training needed for basic possession.
2. **Perception symmetry.** Other animals observe the possessed animal through
   the same nearest-K agent slots as any animal (position, velocity, species).
   They cannot tell a human is steering; predator/prey reactions to the
   possessed animal are the same learned behaviors — emergent, unscripted.

### Unity-side integration

- `PlayerBrain` is just another `ICreatureBrain` implementation on the creature
  seam (unity/aniML-world: `CreatureController` + brain/motor split). Possess =
  swap decision layer; release = swap back. Motor (the policy) is untouched.
- The possessed animal is **pinned to the full-physics body tier** (see §8/§13
  body-fidelity LOD) regardless of camera logic, plus a third-person follow
  camera mode alongside the god camera.

### Known issue: players are out-of-distribution command generators

Training samples smooth-ish random commands; players do instant reversals,
wiggle, and sprint into obstacles. Mitigations, cheap → expensive:
1. **Command envelope filter** (day one): clamp/smooth player input to the
   magnitude and rate-of-change ranges seen in training.
2. **Player-like command curriculum** (later, training-side): add jerky
   reversals and stop-start patterns to command sampling so policies are
   robust to human steering.

### Compute doctrine this slots into (2026-07-23 analysis)

Policy inference is negligible (tiny MLPs, batched: ~0.4 MFLOPs/animal/tick —
every animal in the world can run its brain every tick). The scarce resource is
**CPU articulated physics** (~0.1–0.3 ms/animal, PhysX). Body tiers:
full articulation near camera (~25–40 animals) → kinematic "policy puppets"
(policy-driven joints, no contact solve, feet IK) for the mid-field hundreds →
macro parcel state beyond (§13). Minds everywhere, always; expensive bodies
only where the player is looking. The possessed animal always gets a full body.

**Collision-fidelity doctrine (2026-07-23):** near-tier animals keep FULL
collision fidelity — vertex-weighted convex hulls (same pipeline as training)
+ self-collision ON + full solver iterations. Self-collision is enabled in
training, so runtime-on is *plant fidelity*, not a luxury. Visible clipping is
a hard product no-go (user standard). Collision simplification is allowed only
as a distance tier (mid-ring+), and only if the articulation stress harness
shows the budget requires it. Scale capacity via island parallelism, the
adaptive physics-ms governor, and control-rate LOD — not by cutting collision
quality where the player can see it.

---

## 20. Physics Substrate Strategy (researched 2026-07-23)

### Why no game engine has GPU rigid-body physics (verified)

1. **CUDA/console lock:** PhysX GPU pipeline is CUDA-only (Windows/Linux,
   NVIDIA); PS5/Xbox are AMD. Core gameplay physics can't exclude consoles.
2. **Callback architecture:** gameplay code needs synchronous mid-frame
   raycasts/triggers/contact callbacks; even NVIDIA's own GPU-rigid-body
   design kept islands, triggers, CCD, callbacks, and scene queries on CPU.
3. **Determinism & networking:** GPU float reduction order breaks cross-run/
   cross-vendor determinism; Chaos was built *for* networked CPU physics.
4. **Demand:** GPU wins appear at thousands of active bodies; games ship
   5–20 visible ragdolls and fake crowds with animation. The market split on
   the callback/latency boundary — robotics sims went GPU, games stayed CPU.

### Options matrix conclusion (mid-2026)

- Unreal/Chaos: CPU-only, NO reduced-coordinate articulation, slower than
  PhysX on Epic's own 64-ragdoll benchmark. Switching engines buys nothing.
- PhysX 5 embed in Unity: SDK is BSD-3 open source (incl. GPU kernels,
  Apr 2025) but both C# binding projects are ARCHIVED (MagicPhysX 6/2025,
  Embark physx-rs 5/2026). CPU embed feasible-but-costly; GPU embed ruled out
  (CUDA → no consoles, dead bindings).
- Jolt: healthiest C# bindings but maintainer declined Featherstone;
  maximal-coordinate ragdolls = worst solver match to training. Ruled out.
- Newton (NVIDIA+DeepMind+Disney, Warp/MJWarp, v1.4.0 7/2026): GPU, very
  active, Python-only — a TRAINING-side engine, not embeddable in Unity.
- **MuJoCo: official Apache-2.0 Unity plugin steps `mj_step` inside Unity's
  process**; island-threaded CPU solver (islands ≈ one animal each); core
  extremely active (3.10.0, 6/2026). Plugin self-described "not fully mature."

### The convergence play (adopted as strategic direction, gated)

Isaac Lab 3.0 (experimental) integrates **Newton/MJWarp as a training
backend** with validated Newton↔PhysX policy transfer. If it matures:
train on the MuJoCo solver family (GPU, via Newton) → run the SAME solver
family in Unity (CPU MuJoCo plugin). One move closes most of the sim-to-sim
solver gap AND lifts the throughput ceiling via island threading.

**Deep-dive findings (2026-07-23, second research pass):**
- Isaac Lab 3.0's Newton backend is **beta** (Beta 2, 6/2026): manager-based
  workflow + contact sensors supported, same USD assets, but breaking changes
  (quaternions WXYZ→XYZW, wp.array data, joint reordering), limited sensors,
  no AMP demonstrated on it, no GA date. amp-rsl-rl's rsl-rl 3.x pin IS
  compatible with Isaac Lab 3.0 (which pins rsl-rl 3.1.2).
- **mjlab (Zakka et al., Berkeley) may be the better training-side host**:
  Isaac Lab's manager API rebuilt on MuJoCo-Warp, MJCF-native (single source
  of truth with the Unity runtime), rsl-rl, AMP proven adjacent (beyondAMP),
  Unitree ships an official RL repo on it. Young framework, NVIDIA GPU req.
- Our port costs regardless of host: MJCF re-audit of the deer asset
  (collision meshes again), RSI re-bake (joint remap + quat convention),
  contact retune (MuJoCo soft contacts vs PhysX rigid), policies retrain from
  scratch. GOOD news: DCMotor explicit actuators compute PD in Isaac Lab
  Python — our actuator code is engine-agnostic.
- **Unity plugin reality check (the weaker link):** Unity raycasts/triggers
  CANNOT see MuJoCo bodies (DIY kinematic-collider mirroring needed);
  spawn/despawn triggers full scene recompile (mitigate: pre-allocated animal
  pool); the plugin's mj_step is effectively single-threaded-solver today
  (island threading is not wired up — would need patching); ~50 physical
  animals plausible, 200 UNPROVEN; no console story; no shipped game uses it.

**Phased plan (revised):**
- **Phase A (now, unchanged):** build on Unity ArticulationBody. Stress
  harness measures the real ceiling.
- **Phase B spikes (cheap, independent):**
  (1) Runtime spike: one deer MJCF in the Unity MuJoCo plugin — per-animal
  step cost, pooling pattern, collider-mirroring friction. ~1-2 days.
  (2) Training spike (coordinates with training workstream): trial-port the
  walk env to mjlab + AMP via the beyondAMP pattern. After current lineage
  resolves.
- **Phase C (only on green spikes):** converge on the MuJoCo family —
  preferred route mjlab → MJCF → Unity plugin (single asset format end to
  end). Do NOT bet the project on two betas at once (Isaac-Lab-Newton +
  immature plugin simultaneously).
- **Watch:** Unity swappable physics backends (~6.5, namechecks MuJoCo);
  Isaac Lab 3.0 GA; MuJoCo plugin mjSpec support (would fix spawn cost).

**Precedent note:** no shipped game runs RL-policy articulated characters at
this scale (TABS/Gang Beasts = scripted ragdolls, no RL). This is frontier.

---

## 21. Terrain Representation Decision (researched 2026-07-23)

**Decision: heightfield stays the world truth. Hybrid carve-stamps for caves
when needed. Full voxel/marching-cubes/dual-contouring: rejected.**

Scoring vs project factors (world-truth contract, RL locomotion/colliders,
editing+disasters, erosion/hydrology, solo budget, faceted style):
pure heightfield 28/30, heightfield+stamps hybrid 27/30, voxel+MC 16/30,
SDF+DC 16/30.

Decisive arguments:
1. **Single-valued sim contract**: everything (scatter, sim, spawning,
   moisture) asks "what's at (x,z)" — voxels make that question ambiguous
   (cave under meadow), a sim-wide refactor, not a terrain swap.
2. **The authenticity toolkit is heightfield-native**: priority-flood,
   flow accumulation, thermal/hydraulic erosion, and every pro tool
   (Houdini HeightField Erode, World Machine, Gaea) operate on 2D grids.
3. **RL alignment**: Isaac Lab terrains are heightfields; PhysX heightfield
   colliders are cheap for articulated ragdolls; voxel = cooked mesh
   colliders everywhere + train/deploy representation mismatch.
4. **Shipped evidence**: BotW is heightfield (TotK added caves as AUTHORED
   MESHES + custom LOD, not voxels); Witcher 3 = heightmap tiles + cave
   meshes; Valheim = destructible heightfield + skybox-instanced dungeons;
   even UE Voxel Plugin's own docs say "height stamps whenever possible."
   Every shipped smooth-voxel game (Astroneer, NMS, Enshrouded, DRG) had a
   team or engine-specialist-years, AND dig-anywhere as the core verb —
   which an ecosystem sim is not. NMS ships a hard cap that silently
   REVERTS player terrain edits (~15k) — voxel persistence is a cost center
   even for successful studios.

Escape hatches (adopt only when a feature demands):
- Gorges w/ overhangs: mesh stamps or bounded second ceiling layer.
- Animal dens/caves: local analytic carve stamps (capsule/sphere SDF list,
  pure f(x,y,z)) meshed with MC only inside stamp bounds, entered through
  terrain holes — or Valheim-style instanced interiors.
- Craters/sculpting/flatten: native heightfield writes + saved deltas
  (write the height buffer directly; do NOT replay edit logs — Valheim's
  shipped mistake).
Flip condition: only if dig-anywhere becomes a core game verb.

---

## 22. Worldgen v2 Plan (agreed 2026-07-23)

Waves (build order):
1. **Climate + biomes**: moisture = distance-from-water field; temperature =
   DIRECTIONAL gradient (latitude-like, north cold / south warm — NOT radial;
   user rejected Valheim-style spawn-centric rings, and a directional axis
   breaks island symmetry) − altitude lapse − rain shadow (leeward dry).
   Whittaker-style (temp × moisture × elevation) table → biome enum consumed
   by color, splat, scatter, sim. Domain-warp all field inputs. Forests as
   cluster objects (seeds + cores + tapered edges + clearings), NOT per-tree
   dice rolls.
2. **Water network + terrain truth**: priority-flood lakes/ponds (+ watershed
   labels), flow-accumulation river widths, canyon/gorge carving, estuaries,
   thermal erosion passes (hydraulic droplets optional later).
   **Water budget (user wants sources REAL in accounting terms):** weather
   rain falls into watersheds → catchment storage + groundwater scalar →
   river level/flow = f(recent catchment rain + groundwater base flow).
   Springs are where the budget daylights. Drought = deficit → rivers shrink,
   ponds dry, mud banks widen. State bookkeeping, not fluid physics.
3. **Natural wonders** (NO man-made structures — hard user rule): singular
   one-per-world features at hydrology focal points — THE waterfall, THE
   gorge, THE giant tree, hot springs, sea cliffs on one coast. Poisson
   spacing tuned to travel time; ridgeline/shore visibility bias.
4. **Live climate**: time-varying climate as shader uniforms (snowline,
   dryness tint, wind) + day/night + weather feeding creature behavior.
   Priorities per user: seasons YES, microclimates YES, wind field YES,
   underwater life PARKED (far future).
