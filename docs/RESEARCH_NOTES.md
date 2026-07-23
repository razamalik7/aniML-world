# Research Notes — findings, progress & loose thoughts

<!-- ================================================================= -->
## >>> HANDOFF (2026-07-22 ~23:00) — v8 RESULT + CRITICAL head-droop finding <<<

**AWAITING RAZA'S FORK DECISION: (A) one more reward patch vs (B) AMP pivot.**

**v8 (COM fix + neck imitation + stance action-offset) ran to ~iter 1000, killed for eval.**
Training metrics looked great (base_tilted 65%->12%, ep_len 952, reward 400, neck_posture
4.5, head_min_height penalty shrinking). BUT they LIED.

**FAITHFUL RENDER + FK-on-npz revealed the truth (this is why Raza insisted on a real render):**
Rolled out model_900 via play_loco.py (--headless! else Hydra viewport crash), dumped npz,
rendered with tools/render_playback_meshed.py, then FK'd skull/chest/pelvis Z from the raw
joint angles. Result across all 4 envs, near-identical:
- pelvis 0.97->0.73 (up-ish), chest 0.68, **skull 1.23 -> ~0.40m and HOLDS (head dragging near ground)**
- tilt (pelvis) only ~17deg; coasts fwd ~0.9m on RSI spawn velocity in 1st sec then FREEZES.
- **= downward-dog collapse NOT fixed, just disguised.** Pelvis held higher so the pelvis-only
  base_tilted metric (12%) is blind; deer stands still (upright-stander plateau) AND head droops.

**WHY THE METRICS GOT GAMED (key lesson — metrics gaming, whack-a-mole #2):**
- `base_tilted` = PELVIS tilt only -> blind to a drooping head/neck.
- `neck_posture` rewards the neck JOINT ANGLE rel. to chest -> policy matches angle locally
  (earns 4.5) while the whole chest/front sinks, so ABSOLUTE skull still ends at 0.40. GAMED.
- `head_min_height` (absolute skull Z) is the ONLY term that caught it (the -0.85 penalty) but
  weight -4 too weak to prevent droop; its shrinking penalty was MISREAD as "head rising".
- **RULE: never trust base_tilted/neck_posture for posture. Verify absolute skull_Z via FK on a
  play_loco npz. tools: play_loco.py --headless --video -> render_playback_meshed.py; then FK
  skull/chest Z (see the one-off in scratchpad / this session's bash).**

**THE FORK:**
- **(A) reward patch:** head_min_height -4 -> ~-12, ADD positive absolute-skull-height reward,
  + walking terms (feet_air_time 1->2.5, track_forward_vel 1.5->3.5 std 0.5->0.35). Warm-start
  model_900. Faster but whack-a-mole risk (loopholes patched one at a time).
- **(B) AMP pivot (LEAN):** full-body adversarial imitation -> "pelvis-up but head-dragging" is
  not a reachable pose; closes all loopholes structurally. r38 ckpt EXISTS
  (logs/rsl_rl/animl_deer_amp_turn/2026-04-30_00-09-50/model_8996.pt), AMP tasks registered.
  BLOCKER: rsl_rl is 5.x now (loco); AMP needs downgrade to 3.0 (global env swap, breaks loco
  until reverted). Rationale: posture rewards gamed TWICE (v7.1 neck droop, v8 head-droop).

**Machine state: CLEAN.** Training killed (no ghosts; VRAM 2.2GB/16.3, verified no python/kit
>500MB). TensorBoard still up on :6006. Faithful render at logs/videos/v8_model900_faithful.mp4.
The upright-stander/forward-speed plateau ALSO real (fwd vel flat ~0.17 for 550+ iters) — any
restart must fix BOTH head-droop AND walking.

**v8 is a REGRESSION from v7.1 (rollout-confirmed 2026-07-22 ~23:00):**
- v8 model_900/400: 0/8 walking, universal face-plant (skull ~0.40, not a productive dip —
  model_400 vs 900 both collapsed + fwd speed ~0; slowly optimizing the collapse, NOT walking).
- **v7.1 model_1400** (logs/rsl_rl/animl_deer_loco_stage1/2026-07-22_15-38-26/, OLD stance):
  faithful 8-env replay = **3/8 envs WALK 5.5-5.9m at 0.54 m/s** (env3,4,7; low head skull~0.60,
  pelvis 0.65), 5/8 face-plant-scoot. Matches Raza's memory. **The fresh v8 restart + stance
  change DESTROYED this 3/8-walking basin.** To replay v7.1 must temporarily revert the 6 v8
  stance values in deer_usd_articulation.py (front hipabd 0, hind hipabd 0, hind knee -0.7) —
  use_default_offset makes the policy stance-specific. (Reverted+restored cleanly this session.)
- **OLD AMP line r37/r38 was our BEST walker:** r37 11.2m@0.37m/s upright; r38 pelvis 0.88-0.94
  (best posture EVER) + yaw turning. On disk (animl_deer_amp_turn/2026-04-30_00-09-50/model_8996.pt).
  We WALKED AWAY from it (chose loco/PPO for controllability + AMP's rsl_rl-3.0 dep) — nothing
  broke it. The loco line never matched AMP's posture.

**REVISED FORK (A now weakest):**
- **B — revive AMP r38** (our actual best walker; needs rsl_rl 5.x->3.0 swap, revisit why we left).
- **C — warm-start v7.1 model_1400** (cheapest; preserves live 3/8-walking basin; use OLD stance;
  add anti-faceplant/head-up + walking-reward fixes). 
- A (fresh reward patch) throws away a walking basin a 3rd time.
- ROOT CAUSE of collapse: front can't bear DYNAMIC load — spawns upright, RSI hands it 0.7m/s,
  front pitches down & buckles within 0.8s -> faceplant -> freezes (stable optimum). Suspects:
  COM still 57% fwd (>50% on forelegs), front-leg actuator stiffness too low. AWAITING Raza B-vs-C.

<!-- ================================================================= -->
## >>> HANDOFF / NEXT ACTION (2026-07-22 ~20:00, pre-compaction) <<<

**IMMEDIATE NEXT STEP: launch the retrain to test the stance fix (APPLIED, Raza-approved by eye).**

The default-stance fix is **already applied** to `training/envs/deer_usd_articulation.py`
`_DEFAULT_JOINT_POS` (v8): FL/FR_hipabd_to_thigh 0->-0.45, HL/HR_hipabd_to_thigh 0->+0.30,
HL/HR_thigh_to_shin -0.7->-0.50. (Front feet now UNDER shoulders, hind UNDER hips, level
4-square. Neck & front knee +0.5 unchanged.) So the code is ready — just launch:

```
cd <repo>; TS=$(date +%Y-%m-%d_%H-%M-%S)
PYTHONPATH=$PWD ANIML_DEER_USD=$PWD/deer_world.usd ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1 OMNI_KIT_ACCEPT_EULA=YES \
nohup /c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train.py \
  --task AniML-Deer-USD-Loco-Stage1-v0 --headless --num_envs 12000 --max_iterations 2000 \
  > logs/loco_v8p0_stancefix_${TS}.log 2>&1 &
```
Watch: head_min_height ->~0, base_tilted low, forward_velocity climbs; STOP at convergence
(~iter 1500, efficiency). Then playback+render (see workflow below) + VISUAL check for an
upright four-square walk (no more front collapse). Compare to v7.1 (collapsed/crouched).
**Metrics lie about posture — ALWAYS render + eyeball (this bit us repeatedly).**

**Stance-fix safety audit (DONE, clean):** symmetric change keeps symmetry-reward zeros valid;
only MAPPED joints changed so joint_track's unmapped-joint ref is intact; spawn compatible +
MORE level (foot spread 3.5cm->1cm); loaded-spring (bent knees) preserved.

**SPAWN MECHANISM — verified 2026-07-22 (autonomous, Raza asked "make sure start is straight down"):**
Stage1 does NOT spawn from `_DEFAULT_JOINT_POS`. It uses **RSI** (`reset_from_amp_motion`,
loco cfg L188-197): (a) body orientation = FIXED upright Rx(-90) quat every reset (L2140-44,
clip orientation ignored) -> base ALWAYS spawns straight/level, no weird angle; (b) legs
(hip-flex,knee,ankle x4) + spine/neck = sampled from a real `deer_walk.npy` frame; (c) hip-
ABDUCTION x4 + 2 yaws + 4 toes = filled from `_DEFAULT_JOINT_POS` (all 0). base_height=0.93.
**KEY:** all 6 v8 stance edits are on MAPPED joints (`*_hipabd_to_*_thigh`=hip-flex,
`*_thigh_to_*_shin`=knee) -> they spawn from the reference clip, NOT from defaults. So v8
stance changes the **ACTION OFFSET only** (`use_default_offset=True`, loco L121) = the zero-
action PD attractor (Walk-These-Ways "ready-to-walk default" mechanism), NOT the literal
spawn pose. Still a real anti-collapse bias, just via the learned-pose channel.
Verified with pure-Python FK render (no Isaac): `docs/spawn_pose_verification_v8.png` +
`tools/` one-off — every frame shows body level, skull up (z 1.21 vs pelvis 0.93), no hoof
penetration. **The old "weird angle when it fell" was post-spawn COM pitch (now fixed), NOT
the spawn.** So spawn is clean by construction; z=0.93 unchanged/correct.

### The diagnosis chain (why we're here) — condensed
Collapse = "downward-dog / belly-slide": head+front drop to ground, front legs drag, deer
scoots forward on hind push. base_tilted/forward_velocity DON'T catch it (pelvis stays level).
Fixes found, in order:
1. **Front-heavy COM (67% fwd) — FIXED & APPLIED.** chest inertial Z -0.21->-0.0736 in
   `assets/deer_urdf/deer_world.urdf`; USD regenerated+verified (57% fwd). Old USD backed up
   `deer_world.pre_comfix57.usd`.
2. **Incomplete imitation — APPLIED (v7).** joint_track imitates only the 12 LEG joints, not
   neck/posture/root-orientation. Added `neck_posture_track` (deer_rewards.py + wired in
   loco cfg); zeroed head_above_base + skull_pitch_l2 (skull_pitch FOUGHT ref carriage);
   head_min_height -8->-4. Neck sign FK-verified (URDF opp. reference): neck_yaw_to_neck
   -0.225, neck_to_skull +0.326. Helped head-droop but collapse persisted.
3. **DEFAULT STANCE = the real root cause (found via static-pose renders + FK + Raza's eye).**
   Old default: front feet ~20cm BEHIND shoulders (under belly) + hind ~19cm forward +
   over-bent -> front legs literally can't bear weight -> sag. A STALE compensation for the
   OLD front-heavy COM ("front legs back to center feet under CoM", per
   docs/ANIMAL_PHYSICS_PIPELINE.md), never updated after we balanced the COM. **v8 stance fix
   corrects it.** Neck targets are FINE (Raza: all carriages decent) — do NOT touch.

### If v8 stance fix does NOT dissolve the collapse -> PIVOT TO AMP
r38 checkpoint `logs/rsl_rl/animl_deer_amp_turn/2026-04-30_00-09-50/model_8996.pt` walked
UPRIGHT + turning on the OLD bad rig (AMP imitates whole-body posture natively -> sidesteps
the reward-shaping whack-a-mole). Now on the corrected rig it should be even better. Blocker:
amp-rsl-rl needs rsl_rl 3.0 but env has 5.x -> use separate env / downgrade, OR the actor-
reconstruct path `tools/play_r38_reconstruct.py` (works but r38 render blocked by obs-dim
drift: current env V25 emits 87-dim vs r38's 57 — needs git checkout ab10b6e for faithful render).

### Standing facts / gotchas
- Efficiency (Raza-corrected): 12000 envs is FINE — near 16GB VRAM ceiling, sim is latency-
  bound (91% util/43% power), MORE envs = more throughput. DON'T reduce. Real levers:
  stop-at-convergence + cheap tests (FK/static-render instead of full training runs).
- Reward stack still has muddle to trim (double vel-tracking, ~4 dead terms) — deferred; do
  with next run for clarity, not urgent.
- MVP framing: goal is ecosystem sim (deer+wolf predator-prey); locomotion is the blocking
  gate. "Good enough upright walk" unblocks it; naturalness is later (AMP/video pipeline).
- Playback: `training/scripts/play_loco.py` (has h5py stub + needs --headless). Render:
  `tools/render_playback_meshed.py --input <npz> --output <mp4>`. Static-pose FK-render trick
  for stance/posture debugging = very high value (see stance_final.mp4).
- Artifacts: v7.1 ckpt `.../animl_deer_loco_stage1/2026-07-22_15-38-26/model_1400.pt`; videos
  in logs/videos/ (v7p1_neckimit_model1400.mp4 = current collapse; stance_final.mp4 = OLD vs
  NEW stance; neck_pose_candidates.mp4).
- Env: isaac py = `/c/isaac-lab/env_isaaclab/Scripts/python.exe`. Kill: `Get-Process kit,python
  | Stop-Process`. PhysX silent-death recurs (clear %LOCALAPPDATA%\ov\cache + NVIDIA\DXCache,
  resume from last ckpt). Windows/PowerShell + Bash tools.
<!-- ================================================================= -->



A running lab notebook. Dated entries, newest ideas kept near the top of each
theme. Half-formed thoughts are welcome here — this is where ideas incubate
before they graduate into `ML_ROADMAP.md`. Not everything here is decided.

---

## Standing findings (things we now believe)

- **Reward-shaping RL is a naturalness dead end.** 26 iterations (v4.7–v4.33)
  moved gait-coordination metrics only marginally and always via trade-offs
  (fix mirror → break yaw; add anti-parallel → porpoise + weave). The research
  literature says this is expected, not bad luck. Reward engineering is good for
  *stability/task*, weak for *naturalness*. → drove the v5.0 stable-base pivot.
- **The user's visual complaints were stability, not coordination.** "Uphill" =
  48 cm/stride vertical porpoise + ~10° nose-down pitch. "Swerving" = thousands
  of degrees of cumulative heading change. Both confirmed from trajectory data.
  Neither is about leg mirroring. And the naturalness rewards were *causing* the
  instability. So the stable pivot fixes what he actually sees.
- **AMP was parked for incidental reasons, not fundamental ones.** The two
  blockers were a root-axis convention bug (cost ~14 runs, fixed at r15) and
  broken playback (fixed this week). AMP produced the *first* walking deer. It
  remains the right tool for the eventual naturalness pass — but *later*, not
  now (the MVP doesn't need naturalness).

---

## RIG AUDIT pt2 (2026-07-22 13:15): prior weight work found; MAGNITUDES don't fix it, POSITIONS do

Raza remembered doing weight redistribution before — CONFIRMED:
- `docs/ANIMAL_PHYSICS_PIPELINE.md` has the full methodology + intended MASSES
  (pelvis 18, chest 28, neck 6, skull 5, ... = 73kg, "60:40 front:rear").
- `deer_world_comfix.urdf` exists; antlers were lightened (v4.1: "1.5kg, CoM closer").
- BUT the prior fix was mostly REST-POSE centering (move feet under the COM to
  pass the doc's "CoM within 2cm of foot centroid" STANDING check), NOT a true
  dynamic mass fix. The deer stayed front-heavy -> collapse recurs when WALKING.

Two problems found:
1. **Current training URDF DRIFTED worse than the doc intent:** chest 41.5kg (vs
   doc 28), pelvis 10 (vs 18), total 82 (vs 73). Even more front-heavy.
2. **KEY: changing mass MAGNITUDES barely moves the COM.** Measured COM fore-aft:
   - current deer_world.urdf: 67% fwd
   - deer_world_comfix.urdf:  65% fwd
   - current + DOC masses:     66% fwd  (!!)
   The COM is dominated by POSITIONS, not kg: the chest link's COM sits far
   forward (Z=-0.39) and head/neck are cantilevered (-0.79/-0.97). Swapping
   pelvis<->chest mass does almost nothing. **The real lever is WHERE the mass
   sits (inertial origins), not the magnitudes.** Doc target is 60:40 (~60% fwd,
   natural); current 67% overshoots even that.

**Better approach than hand-tuning (which has drifted twice):** recompute each
link's inertial (COM position) FROM ITS MESH GEOMETRY (centroid of that link's
vertices) + anatomical mass fractions targeting ~60% fwd + light distal limbs.

## v7.1 PLAYBACK VERDICT (2026-07-22 17:59): fix confirmed, but gait still poor — whack-a-mole

Playback of model_1400 (metrics that TRAINING numbers hid, again):
- pelvis height mean **0.70** (min 0.47, max 1.01) — CROUCHED (standing ~0.85-0.95)
- **vertical bounce 0.39 m** (target <0.10) — porpoise is BACK
- **body pitch 15.5 deg** nose-down — still front-low
- straightness 0.33, net speed 0.12 m/s

Honest read: the neck-imitation hypothesis was CONFIRMED (head_min_height best-
ever, neck engaged - good science), BUT the OVERALL gait is still crouched +
bouncing + nose-down. We've now fixed the front-heavy rig AND the head-droop
imitation gap, and the deer STILL doesn't walk cleanly. Each reward-shaping fix
reveals the next problem (crouch, bounce, pitch) = classic whack-a-mole. Meanwhile
r38 (AMP, full-body imitation) achieved UPRIGHT walking without this cascade.
STRATEGIC INFLECTION: continuing to hand-patch reward-shaping term-by-term is
looking like the slow path; AMP imitates whole-body posture natively and already
proved upright. Strongly worth weighing AMP as the main path vs more reward
patches. Video: logs/videos/v7p1_neckimit_model1400.mp4.

## v7.1 RESULT (2026-07-22 17:27): neck-imitation fix WORKED (partial) — residual crouch

Completed-imitation fix (neck_posture_track, head-up sign FK-verified after a
first-guess sign error) on the rebalanced rig. At iter 1157 (converged posture):
- base_tilted 7% (upright), forward_velocity 0.71 (walking), time_out 93%.
- neck_posture 4.2 (neck strongly at head-up carriage — v6.0 never had this).
- **head_min_height -0.73 = BEST of any run (v6.0 -1.08), but PLATEAUED, not ->0.**
Verdict: the head-droop ROOT CAUSE (unconstrained neck) is fixed - head ~0.35
higher than v6.0. But a RESIDUAL CROUCH keeps absolute skull height under the
0.9m floor: neck is up, whole body rides low. Next thread: why crouched when
base_height(-20)/pelvis_height_ramp(+3) should push it tall? (candidate: the
default/standing pose height, or base_height target too low, or leg_extension
target 0.89 vs actual crouch). Need PLAYBACK for the real visual.

EFFICIENCY (per Raza 16:58): sim-bound (collection 5.6s vs learning 0.22s/iter);
GPU 91% util but only 43% power (154/360W) = latency-bound, not compute-saturated;
steps/sec dropped 61k->49k (partly the 33-term reward muddle). Leaner next runs:
~4096 envs, ~1500 max iters, un-muddled stack -> est ~2x faster. Pending Raza ok
to make default.

## ROOT CAUSE of the collapse (2026-07-22 15:15): we only imitate LEGS, not posture

Reward-stack audit (~33 terms) + reference-posture check nailed it:
- `joint_track_reference_walk` (+14 ep, DOMINANT, ~10x every other term) tracks
  ONLY the 12 LEG joints (3/leg). It does NOT track the neck, skull, spine, or
  ROOT ORIENTATION.
- Reference clip (deer_walk.npy) is UPRIGHT: root pitch 0deg, chest_to_neck
  +12.9deg (neck raised), neck_to_skull -18.7deg (natural head carriage). NOT
  head-low. Antlers only 1.5kg (ruled out as a cause).
- => The policy can trace the reference LEG gait perfectly while pitching its
  whole body nose-down + drooping the (unconstrained) neck -> head to ground,
  and joint_track sees nothing wrong. **The collapse is matching the part of the
  reference we imitate (legs), on a posture we never imitate.**
- The ~11 posture/anti-collapse penalties (head_above_base, head_min_height,
  skull_pitch, flat_orientation, leg_extension, torso_contact, base_height...)
  all exist to MANUALLY re-impose the upright posture we discarded — and they
  LOSE to the +14 leg reward. The muddle and the collapse have the SAME root:
  incomplete imitation.
- **Explains r38(AMP)-upright vs loco-collapse:** AMP imitates the WHOLE body
  state (posture included) via the discriminator; joint_track imitates legs only.

v6.0 (rebalanced rig, fresh Stage1) data point @ iter 681: base_tilted 37%->10%
(rig fix gave a more LEVEL body - real gain) BUT head penalties still firing
(-1.1) = head still carried low. Consistent: rig fixed body-tip, head-low
persists because imitation doesn't cover head/posture.

FIX OPTIONS (pending Raza decision, leaning a):
  (a) COMPLETE THE IMITATION: extend joint_track to also match root pitch~0 +
      the 2 neck joints. Makes upright+head-carriage directly rewarded by the
      clip we already have; lets us DELETE most posture penalties. Root-cause +
      un-muddle in one. ("make reward-shaping do what AMP does for free.")
  (b) Head-low TERMINATION (blunt, treats symptom, keeps compensation penalties).
  (c) Go back to AMP (full-body imitation native; r38 proof; heavier path).

## RIG AUDIT pt3 (2026-07-22 13:30): mesh analysis -> it's ONE bad number (chest COM)

Computed each link's true geometric COM from its visual mesh (trimesh) vs the
hand-set inertial origin:
- **chest: inertial Z=-0.210 but mesh centroid Z=+0.035 -> 0.245m too far FWD.**
- pelvis / neck / skull inertial origins all match their mesh centroids (~0 drift).
So the ENTIRE front-heaviness is one hand-placed error on the chest link.
Whole-body COM: current 67% fwd; chest COM -> mesh centroid = **49% fwd** (balanced).

=> "full mesh recompute" and "minimal chest nudge" are the SAME fix (one number).
Low-risk AND rigorous. (Volume-proportional masses are NOT viable — the neck
mesh is oversized, giving 15.5kg neck; keep anatomical mass magnitudes.)

THE FIX (one-line URDF edit, git-reversible, pending Raza's accurate-vs-easy pick):
- chest inertial origin Z: -0.21 -> **-0.07** => ~57% fwd (anatomically accurate;
  real deer forequarters denser than uniform mesh => a bit fwd of the 49% centroid)
- OR chest inertial Z -> +0.035 => 49% (mesh-balanced, most forgiving for RL)
Then regenerate USD (import_world_urdf_v3.py) + retrain fresh Stage1 (proven
recipe) and check if the downward-dog collapse dissolves on its own. Raza lean +
mine: -0.07 (accurate).

Raza flagged rig fidelity (DOF, mass, joint locations) as foundational for
naturalism. Audited `assets/deer_urdf/deer_world.urdf`:
- **DOF: fine.** 25 actuated (5/leg incl. actuated toe + 2 spine [yaw+sagittal]
  + 3 neck). Joint limits sane (hip 160deg, knee 126, hock 109). DOF count is
  NOT the problem; if anything it's rich. (AMP used 15, loco 21, body has 25.)
- **MASS DISTRIBUTION: the problem.** Total 82 kg (plausible). But COM computed
  via URDF forward-kinematics sits at **67% of the wheelbase toward the FRONT**
  (balanced=50%, natural deer/horse ~55-58%). Culprits: chest 41.5 kg (51% of
  mass) forward at Z=-0.39, PLUS neck 5 kg + skull 3 kg cantilevered far forward
  (Z=-0.79 / -0.97). ~60% of all mass is in the front half + a heavy head hung
  out front. **The body physically wants to tip onto its face.**

**Why this matters hugely:** almost certainly a real driver of the recurring
downward-dog/belly-slide collapse (the body is biased toward it), explains why
it recurs across unrelated policies and why r35 needed a specific fix (V22) to
beat it, and caps naturalism (dynamics don't match a real deer). Likely an
auto-gen-pipeline artifact: lumped uniform-density mass on the biggest mesh
volume (the barrel) + an over-heavy, over-cantilevered head.

**Recommendation:** bounded URDF mass-rebalance BEFORE committing to r38/AMP —
redistribute chest mass rearward toward abdomen/pelvis + lighten/pull-in
head+neck, target COM ~55-58% forward. Analytical edit, no retraining to find
it. Could dissolve the collapse for ANY policy + raise the naturalism ceiling.
**REBALANCE PROPOSAL (worked up 2026-07-22 13:00, pending Raza review before
editing URDF):**
- Current ~64-67% fwd. Target natural deer ~55-58%.
- **Scenario A (minimal, recommended):** move the CHEST link's inertial origin
  from Z=-0.39 to ~-0.26 (the barrel's actual center). Single inertial-origin
  edit, NO mass changes -> ~55% forward. Justified: the chest link IS the whole
  barrel (no separate abdomen link), so its COM belongs near the barrel middle,
  not shoved forward at -0.39. The forward placement is an auto-gen artifact
  (uniform-density lumping toward the front of the mesh).
- Scenario B: A + skull 3->2kg, neck 5->4kg, pulled in -> ~50%.
- Scenario C: B + chest 41.5->34kg / pelvis 10->17.5kg -> ~47%.
- Next after edit: retrain (fresh Stage1 loco, current proven recipe) on the
  rebalanced rig and check if the downward-dog collapse dissolves on its own.
  That's the real test of the hypothesis.

r38 RENDER DEFERRED: current DeerUsdWalkAMPEnvCfgV25 now emits 87-dim policy obs
but r38's actor expects 57 (obs composition drifted since 2026-04-30). Faithful
render needs the r38 commit's code (git checkout ab10b6e) + rsl_rl 3.0 downgrade.
Not worth it now; r38 uprightness is well-documented. tools/play_r38_reconstruct
.py works mechanically (reconstructs actor, dodges rsl_rl) — just blocked on the
obs-dim drift.

Note: onnxruntime NOT needed for r38 playback — reconstructed the actor MLP
(plain 57->512->256->128->15 ELU, no obs normalizer) directly from the .pt in
tools/play_r38_reconstruct.py, zero env changes.

## MAJOR (2026-07-22 12:40): we ALREADY built the upright walker — it's r38 on the PARKED AMP path

Raza's hunch ("we made a model that walks ok and upright, discarded for
naturalism") is CORRECT. Git history + run archaeology:
- **r35 (2026-04-29) "BREAKTHROUGH: deer walks upright, no belly-slide"** —
  "V22 kills the belly-slide local optimum." The downward-dog/belly-slide
  collapse (the exact thing that reappeared in v5.3) was SOLVED here.
- r36: "posture even better." r37: "best run ever — 11.2m at 0.37 m/s, upright."
- **r38: "pelvis z avg 0.88-0.94 (best ever upright), yaw turning works."**
  Summary: all 4 envs walk forward 5-9.4m at 0.18-0.31 m/s, integrating yaw,
  env2 turned 19deg. **This is genuinely upright (0.88-0.94 vs v5.3's collapsed
  0.73) with turning.**
- **r38 checkpoint:** `logs/rsl_rl/animl_deer_amp_turn/2026-04-30_00-09-50/
  model_8996.pt` — AND it has **ONNX exports** (`policy_8996.onnx`), i.e. it's
  already in Unity-deployable format.

**The strategic miss:** the AMP lineage (r35-r38) produced the best UPRIGHT
walker, then got parked in favor of the loco reward-shaping path (v4.x), which
chased naturalness gait-correlations but regressed on the fundamental
uprightness. My recent v5.x work (strip naturalness / fresh-from-scratch) made
it worse — full downward-dog collapse. We detoured hard from a working base.

**Recommendation:** make the r38 AMP policy the MVP locomotion base (upright +
turning + already ONNX-exported), STOP the loco-stripping experiments. It even
already beat the belly-slide via V22 — whatever V22 did is the anti-collapse
recipe (likely akin to the head_min_height/leg_extension rewards). Naturalness
polish comes later. To SEE r38 we need to revive AMP playback (match the V25 env
version from the run's git snapshot + port the h5py-stub/headless fixes from
play_loco.py to play_amp.py) — bounded task, pending Raza's go.

## CRITICAL (2026-07-22 12:10): v5.3 "clean walker" was a DOWNWARD-DOG COLLAPSE — metrics lied

Raza watched the v5.3 video: the deer is NOT walking upright. It holds its
pelvis up + level while the neck droops the skull to the GROUND, front legs drag
limply tucked under the body, and it SLIDES forward on hind-leg push. Classic
"downward-dog / butt-in-air" collapse — a known recurring exploit ("behavior
we've seen before").

**Why every metric missed it** (important lesson): both terminations are
pelvis-based — `pelvis_too_low` fires at 0.55m (pelvis sat at 0.66-0.97, never
triggers) and `base_tilted` at 60deg (pelvis pitch only ~14deg, never triggers).
The collapse is a NECK droop, invisible to pelvis checks. So base_tilted read
"94% upright" while the deer was face-down sliding, and `forward_velocity` got
earned by the slide. **`base_tilted` + `forward_velocity` can BOTH be satisfied
by a hind-push slide. Never trust them alone — watch head/skull height and
ALWAYS visually verify.** (In hindsight the data hinted: nose-down 14deg pitch,
head_above_base ~-0.05, front carpus 100% folded/tucked, pelvis 0.73m low-ish.)

**Root cause in config:** the Loco stack had `head_above_base` weakened to -1.0
(from -3.0, in an old run that assumed the deer was reliably upright), and left
the purpose-built anti-collapse rewards `head_min_height_l2` and
`leg_extension_reward` UNUSED. `head_min_height_l2`'s own docstring:
"Butt-in-air pose drops head to ~0.5m. Penalising below 0.9m blocks that
exploit." It was written for exactly this and never wired in.

**Fix (v5.4, launched 12:13, fresh Stage1):**
- head_above_base -1.0 -> -3.0
- + head_min_height_l2 weight -8.0 (absolute skull floor at 0.9m)
- + leg_extension_reward weight +5.0 (un-fakeable pelvis-to-foot standing gap;
  splayed/dragging scores ~0)
Fresh so the collapse is penalized from iter 0, never learned. **New monitoring
signal:** watch `head_min_height` (near 0 = head up) and `leg_extension` (near
max = legs extended/standing) in the training log — these detect the collapse
that base_tilted can't.

**TODO:** add skull-height + leg-extension to analyze_trajectory.py / the eval so
this can't slip through on metrics again.

## v5.3 EVAL (2026-07-22 11:53): clean fresh baseline vs old v4.31r lineage

Fresh Stage1 (proven full recipe, 2100 iters, model_2099) evaluated vs v4.31r
(old naturalness lineage, ~30k iters, Stage3). **Answer to "is porpoise/weave
inherent or lineage-specific?" — BOTH, partly.**

| metric | v5.3 fresh | v4.31r old | target |
|---|---|---|---|
| vertical bounce range | **0.30 m** | 0.44 m | <0.10 |
| \|body pitch\| avg | 14° | 7.8° | ~0 |
| cumulative heading (weave) | **350°** | 503° | low |
| straightness | **0.34** | 0.29 | >0.6 |
| \|yaw drift\| | **26°** | 68° | 0 |
| falls (training) | ~6% | — | low |
| carpus fold band | 100% | ~95% | >80 |

Read: the fresh clean baseline is BETTER on bounce (-33%), weave (-30%),
straightness, and yaw drift — so a chunk of the pathology WAS lineage baggage
from the long warm-start-tweak chain. BUT bounce is still 0.30 m (3x target)
and pitch is actually worse (14°) — so bounce is at least partly INHERENT to
the reward/gait, not just lineage. Note FL/FR hip corr = +0.27 (in-phase front
legs) + persistent bounce => the gait has a slight BOUND/HOP component. Caveat:
v5.3 is a young Stage1 policy (forward-only, 2100 iters) vs v4.31r's mature
Stage3 — that a fresh Stage1 already beats the heavily-tuned v4.31r on most
stability metrics is encouraging and leaves headroom.

**Next-step fork (for Raza):** (A) curriculum forward Stage1->2->3 (proven,
safe, adds turning+terrain, doesn't fix bounce) vs (B) attack the bounce now —
warm-start a refinement with MODERATE anti-bounce (lin_vel_z -2 -> -3, gentle
since gait is established, unlike the from-scratch v5.0 that broke at -4).
Recommendation: do B's bounce-fix as a short targeted pass FIRST (it's the #1
visual complaint and now low-risk from a walking policy), then curriculum the
cleaned gait up. Video: logs/videos/v5p3_stage1_model2099.mp4.

## KEY FINDING (2026-07-22 03:47): naturalness rewards are also *bootstrapping scaffolding*

The whole "strip naturalness -> get stability" strategy has a flaw we only saw
by trying it three ways:
- Warm-start + strip (v5.0/v5.1): destabilized (collapse / drift to 55% falls).
- **Fresh + strip (v5.2): could NOT learn to walk at all** — stuck at 100% falls
  with forward velocity flat-lined at ~0.09 from iter 53 to 331. Degenerate
  "fall slowly in place" optimum.

Root cause: **`joint_track_reference_walk` (the per-joint reference-pose reward)
is DUAL-PURPOSE.** At convergence it's cosmetic (shapes gait realism). But from
random init it's the dense *pose-guidance scaffold* that lets the policy
discover walking at all. Strip it and pure "reward forward velocity + don't
fall" is too sparse to bootstrap locomotion from scratch. I conflated its two
roles.

**Strategy reframe:** don't strip the pose/naturalness rewards to get stability.
Instead: (1) get a CLEAN walking baseline from the proven full recipe (fresh
Stage1, v5.3), (2) evaluate whether the porpoise/weave are *inherent to the
reward design* or an *artifact of the v4.x warm-start-chain lineage* (v4.31r was
the end of a long tweak chain — a clean fresh run may just be better), (3) if
the pathologies persist, target them SPECIFICALLY (tuned anti-bounce /
anti-weave on top of the walking recipe) rather than by stripping. Sequence the
risk: solve "walk at all" first, "walk stably" second — trying both from scratch
is what failed. *This is a good candidate to discuss with Raza in the morning —
it changes the stable-base plan.*

## Method choices & rationale (working understanding, 2026-07-22)

- **Data form decides the method — not quality.** Behavior cloning needs
  `(our-obs -> our-action)` pairs (teleop-style); you can't collect that for a
  simulated animal (no way to puppeteer 21 joints into an expert gait). Our data
  is *kinematic motion* (clips / video) -> that is exactly the regime for
  imitation-via-RL (AMP / DeepMimic). This is *why* manipulation (teleop data)
  uses BC / π-style models and animal locomotion (mocap / video) uses AMP. Same
  reason we can't just "collect BC data" for the deer.
- **Base RL algo = PPO (on-policy, model-free, actor-critic) — confirmed the
  right choice.** With a fast 12k-env sim, sample-inefficiency is a non-issue and
  PPO's stability wins. Off-policy (SAC) is for sample-limited / real-robot;
  model-based (Dreamer) is needless complexity here. We run rsl_rl PPO + mirror
  symmetry aug = the standard Isaac / legged_gym recipe.
- **Method-per-phase:** stable base = RL-from-reward (PPO). Naturalness = try
  DeepMimic tracking first (less finicky for a clean single clip) then AMP
  (noise-tolerant; needed for video data). Ceiling = ASE / NPMP skill embeddings.
  BC only ever as a *downstream distillation* tool, never primary.
- **RL is NOT replay/lookup — it's the emergent one.** The policy is a
  generalizing neural function state->action, not a memorized scenario table.
  Behavior (flee the wolf) EMERGES from rewarding the *goal* (survive), never
  from a scripted "if wolf then run." Reactive at runtime (feedforward reflex)
  but generalizes to unseen states, bounded by training diversity (-> why we use
  domain randomization). The "reproduce what was demonstrated" intuition fits
  *behavior cloning*; RL *discovers* behavior from reward.
- **SMAL bottleneck (deer only):** SMAL = toy-scan model of cats/dogs/horses/
  cows/hippos — no cervids, no antlers. It's the video 3D-lifting stage; deer fit
  poorly -> noisy 3D. Wolf ~= dog (BARC) fits clean. Workarounds: (1) asset clips
  for deer (no SMAL at all), (2) retarget a well-fit quadruped's gait (horse/dog)
  onto the deer, (3) custom deer shape (down the line), (4) prove pipeline on wolf
  first. Retargeting onto our URDF is the part we already own; only *video-3D-of-
  a-deer* is finicky.

## Video → motion feasibility (Phase 4 data path)

**Idea:** we lack good animal animations (bought packs look robotic and are hard
to source). Instead, extract reference motion from ordinary monocular wildlife
video using CV, then imitate it in sim. This solves the data problem *and* adds
a computer-vision résumé line. See `docs/REFERENCES.md` (npj Robotics 2025, SFV,
SMAL/BARC).

**Standard pipeline (well-established, not exotic):**
1. Collect monocular videos of the target animal (wildlife footage / YouTube).
2. **2D keypoints** — off-the-shelf (DeepLabCut, ViTPose-animal / AP-10K). ~solved.
3. **3D lifting** — fit a SMAL-family parametric model (SMAL / BARC / hSMAL) to
   the 2D keypoints + silhouettes → 3D joint trajectories over time. *Hardest,
   noisiest stage.*
4. **Retarget** the animal skeleton → our URDF skeleton (joint correspondence +
   IK). **We already own this** — the Unity-anim → npy converter does exactly
   this (swing-twist decomposition + bind-pose calibration).
5. **Imitate** in sim with AMP (or DeepMimic-style tracking).

**Feasibility read: MEDIUM, and de-risked for us because:**
- We already own the retargeting + imitation half of the pipeline.
- **AMP tolerates noisy references** (needs the motion *distribution* to look
  right, not exact per-frame tracking) — a natural fit for imperfect monocular
  3D lifting.
- **Wolf ≈ dog ⇒ maps onto the best-validated model (BARC).** Wolf is the
  easier first target, not deer.

**Main risks / unknowns:**
- **Deer-specific 3D lifting is weaker** — no deer SMAL; generic SMAL or an
  hSMAL-style fit. Antlers/body proportions may fit poorly. (Prototype on wolf.)
- Monocular depth ambiguity → noisy 3D. Mitigated by AMP's tolerance + temporal
  smoothing (the npj paper uses a spatio-temporal GCN + Kalman fusion for this).
- Foot-contact / global-root recovery from video is notoriously hard; may need
  contact heuristics.
- Effort is real: 2D→3D→retarget→imitate is a multi-stage CV+RL project. It's a
  *phase*, not an afternoon.

**Smallest sane prototype:** one wolf, a handful of clean side-view trot videos,
existing 2D pose model → SMAL/BARC fit → retarget to URDF → feed as AMP
reference → check the discriminator learns a plausible trot. If that closes,
the pipeline is proven and scales to more clips/animals.

---

## MVP demo — where should it live?

**Decision (current):** build + validate the MVP (emergent predator–prey) in
**Isaac Lab**, render shareable clips via our offline meshed renderer. Treat
**Unity** as a separate, later Phase-5 productization milestone.

**Why Isaac Lab for the MVP:**
- It's where the ML lives and where we can iterate fastest (all-Python, no
  export step, training + eval in one place).
- MARL / many-agent sim runs natively.
- We already have the offline meshed renderer for shareable video (Isaac Sim's
  live viewer is broken on this RTX-50 box anyway).

**Why NOT Unity yet:**
- The Unity runtime is entirely stubs today.
- Needs an ONNX export pipeline + re-implementing observations/env in C#.
- **Sim-to-sim gap:** a policy trained in Isaac PhysX may not transfer cleanly
  to Unity's engine — that's a real, separate research/eng problem. Doing it too
  early would stall ML progress.

**But:** keep the policy I/O interface engine-agnostic from Phase 1 so the Unity
port later is a port, not a rewrite. Unity is worth doing eventually — the
real-time-deployment story is itself résumé-valuable — just not on the MVP's
critical path.

---

## Loose ideas / parking lot (unfiltered)

- **Prototype the video pipeline on wolf first** (dog-validated CV models), then
  transfer method to deer.
- **Physics-only naturalness as a cheap fallback:** energy/effort minimization +
  symmetry + biomechanical joint constraints get surprisingly natural gaits with
  *no* reference data. Worth a small experiment before committing to the full
  video pipeline.
- **The stable-base retrospective is a good blog post:** "why reward-shaping
  fails for natural gait, and what imitation does instead" — concrete, honest,
  and demonstrates depth. Low effort, high signal for a portfolio.
- **Skill-embedding (ASE) as the unifying architecture:** if we train one latent
  motor space per species, the ecosystem brain outputs latents and we get
  walk/run/turn/pounce from one model. Ambitious but the *right* long-term shape.
- Consider whether the "off-screen macro-sim" (ARCHITECTURE §13) can be an early,
  cheap way to demo ecosystem dynamics at scale before locomotion is pretty.

---

## Open questions to resolve

- ~~Does the warm-start shed the porpoise/weave?~~ **NO — warm-starting the
  stable reward from the naturalness policy (v4.31r) is the wrong tool, proven
  twice.** v5.0 (total strip) collapsed to 99.8% falls. v5.1 (surgical strip:
  cosmetic terms only, keep gait timing, lin_vel_z -2.5) held ~15% briefly then
  **drifted up to 55% falls** as it trained away from the naturalness gait.
  Root cause: that gait was propped up by the very rewards we remove, so the
  warm-started body falls apart without them. (Also lost ~30 min to a PhysX
  Fatal crash mid-run.)
  **Pivot (v5.2, 2026-07-22 03:13): FRESH from scratch, no warm-start.** New
  task `AniML-Deer-USD-Loco-Stable-Flat-v0` = Stage1 forward-only recipe +
  naturalness stripped, trained from zero. Stage1 reliably discovers walking
  from scratch even *with* naturalness rewards; stripping them (fewer competing
  objectives) makes it easier. Expect a clean, un-entangled stable walk we can
  later curriculum up to full 3-DOF + terrain. If this converges, warm-start is
  abandoned for the base entirely.
- Wolf asset/URDF: do we have one, or is deer the only rigged animal so far?
  (Predator–prey MVP needs a second body.)
- **PhysX silent/Fatal crash is a recurring infra tax** (killed v4.26, v4.32,
  v4.33, and now v5.1 at iter ~30742 / 2026-07-22 02:15 with a breakpad Fatal in
  python311.dll). Remediation that works: kill orphans, clear %LOCALAPPDATA%\ov
  \cache + NVIDIA\DXCache, resume from last checkpoint. This costs real time on
  long runs — worth budgeting for, and a reason to checkpoint often (we save
  every 100 iters, good). Not a blocker, but a persistent drag on iteration
  speed on this RTX-50 box.
- For the video pipeline: which 2D pose model + which SMAL variant give usable
  deer/wolf 3D? Needs a small spike.
