#!/bin/bash
# D2b -- D2 plus PHASE 2 (the expert-only discriminator normaliser).
#
# THE POINT. D2 and exp3b are each half of the deer we want, and the halves have never
# been in the same run:
#
#   at their own ~3 m/s          foot-strike (ankle share)   veer      stride CV   slip
#   D2   (phase 1, CLEAN walk)   60/29/63/0%  -- lands on    +0.0 deg/s   0.31    1.1x
#                                   its ankles on 3 legs
#   exp3b(phase 2, DIRTY walk)    2/ 5/13/1%  -- lands       +2.3 deg/s   0.36    1.4x
#                                   squarely on the hooves
#
# Raza saw both on sight: "springy on the hoofs amazing" for exp3b, "feet bottoms arent
# touching" for D2. Phase 2 is what cleaned exp3b's hooves -- exp3 was 33.5% ankle load
# and exp3b 3.2%. D2 never got phase 2; exp3b never got the clean walk clip.
#
# So: D2's data and command set (which give the steady rhythm and zero veer) + phase 2
# (which gives the hoof landing). One variable on top of our best base.
#
# WHAT MAKES IT PHASE 2: dropping ANIML_DEER_AMP_LEGACY_NORM. Without it train_amp.py
# patches the discriminator's normaliser to use EXPERT-ONLY statistics, so the policy's
# own outliers stop inflating the variance that hides them. That is the whole change.
#
# DEVIATION FROM exp3b's RECIPE, deliberately: exp3b resumed from exp3's model_1499
# (mid-training). This resumes from D2's FINAL model_2699, because the properties we are
# trying to KEEP -- zero veer, steady stride timing -- emerged over D2's full training,
# and restarting from 1499 would put them back in play. The risk is that a converged
# policy is less plastic and phase 2 fails to move the foot-strike. If the hooves do not
# clean up, retry from model_1499 before concluding phase 2 needs the clean walk clip.
#
# READ IT ON: ankle share per limb (target: exp3b's ~2-13%, not D2's 60-63%), while
# holding veer near 0.0 deg/s and stride CV near 0.31.
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO" OMNI_KIT_ACCEPT_EULA=YES ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
export ANIML_DEER_W_FEET_AIR_TIME=0.0 ANIML_DEER_W_JOINT_REF_RANGE=0.0
export ANIML_DEER_W_LIN_VEL_Z=0.0 ANIML_DEER_W_ANG_VEL_XY=0.0
export ANIML_DEER_W_JOINT_POS_LIMITS=-2.0
export ANIML_DEER_CMD_ANG_VEL_Z=0.0
export ANIML_DEER_CMD_VX_LO=0.8 ANIML_DEER_CMD_VX_HI=3.2
# NO ANIML_DEER_AMP_LEGACY_NORM here -- its absence IS the experiment.
export ANIML_AMP_DATASETS="deer_polyart_walk:1.0,deer_polyart_trot:1.0,deer_polyart_run:0.5"
ITERS="${D2B_ITERS:-1200}"
ENVS="${D2B_ENVS:-4096}"
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs "$ENVS" --max_iterations "$ITERS" --headless \
  --resume_from logs/rsl_rl/animl_deer_polyart_walk/2026-07-29_00-33-54/model_2699.pt
