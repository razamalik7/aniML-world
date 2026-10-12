#!/bin/bash
# probe-L: is joint_pos_limits (hard end-stop penalty) cold-start-safe on its own?
# Conditions = probe-N EXCEPT joint_pos_limits restored to -2.0.
#   probe-N (both terms 0, legacy norm) reached ep_len 60.3 @270.
#   If probe-L tracks that, joint_pos_limits is harmless -> keep it on permanently.
#   If it drops toward probe-A (26.2) or r15 (17.5), it is a blocker too -> anneal it.
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMNI_KIT_ACCEPT_EULA=YES
export ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
export ANIML_DEER_W_JOINT_POS_LIMITS=-2.0    # <-- THE ONLY CHANGE vs probe-N
export ANIML_DEER_W_JOINT_REF_RANGE=0.0
export ANIML_DEER_AMP_LEGACY_NORM=1
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs 4096 --max_iterations 300 --headless
