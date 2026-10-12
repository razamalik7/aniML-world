#!/bin/bash
# F1 -- C3 with the ASE DIVERSITY OBJECTIVE turned on. Exactly one variable changed.
#
# WHY. C3 reached MI 0.92: the encoder can tell its skills apart almost perfectly. But
# exp4 reached MI 0.82 while its z-sweep separated skills by only 22-36 degrees of pose
# -- Raza's read was "a bit diff, not insanely diff". High mutual information does not
# imply visible behavioural range, because the encoder is satisfied by ANY consistent
# fingerprint, however small. ASE's paper carries a third component for exactly this,
# and we had not implemented it.
#
# The diversity objective requires the ACTIONS under two latents to differ in
# proportion to the latents' own distance (bonus = a_diff / z_diff, driven to a target
# of 1.0). It is a term on the policy loss, not a reward, so it needed a hook inside
# rsl_rl's PPO update -- see train_amp.py. Verified on CPU first:
# `tools/test_ase_diversity.py` (12 checks, incl. that a z-blind policy scores 0 and
# that training on the term raises spread 0.002 -> 0.79).
#
# HOW TO READ IT. The comparison is F1's z-sweep against C3's, both at 3000 iters:
#   * pairwise pose separation UP with MI held near 0.92  -> the diversity objective
#     was the missing piece; ASE stage 1 is real and stage 2 becomes worth building
#   * separation up but MI collapses -> the two objectives fight at coef 0.01; retune
#   * separation flat                -> spread is limited by the DATA (3 gaits), not
#     the objective, and more clips is the only path
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO" OMNI_KIT_ACCEPT_EULA=YES ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
export ANIML_DEER_W_FEET_AIR_TIME=0.0 ANIML_DEER_W_JOINT_REF_RANGE=0.0
export ANIML_DEER_W_LIN_VEL_Z=0.0 ANIML_DEER_W_ANG_VEL_XY=0.0
export ANIML_DEER_W_JOINT_POS_LIMITS=-2.0
export ANIML_DEER_CMD_ANG_VEL_Z=0.5
export ANIML_DEER_CMD_VX_LO=0.8 ANIML_DEER_CMD_VX_HI=3.2
export ANIML_DEER_AMP_LEGACY_NORM=1
export ANIML_DEER_ASE_DIM=8
export ANIML_DEER_ASE_MI_SCALE=0.5
# THE single change vs C3. 0.01 / 1.0 are the reference implementation's defaults
# (nv-tlabs/ASE, ase/learning/ase_agent.py).
export ANIML_DEER_ASE_DIVERSITY=0.01
export ANIML_DEER_ASE_DIVERSITY_TAR=1.0
export ANIML_AMP_DATASETS="deer_polyart_walk_alltakes:1.0,deer_polyart_trot:1.0,deer_polyart_run:0.5"
ITERS="${F1_ITERS:-3000}"
ENVS="${F1_ENVS:-4096}"
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs "$ENVS" --max_iterations "$ITERS" --headless
