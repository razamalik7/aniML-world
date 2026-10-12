#!/bin/bash
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMNI_KIT_ACCEPT_EULA=YES
export ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
# THE TEST: zero the two terms that are the only reward difference vs r10.
export ANIML_DEER_W_JOINT_POS_LIMITS=0.0
export ANIML_DEER_W_JOINT_REF_RANGE=0.0
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs 4096 --max_iterations 300 --headless
