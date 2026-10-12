#!/bin/bash
# EXP 3 -- multi-gait expert set. walk_alltakes(96) + trot(25) + run(21).
# Measured gait speeds: walk 1.14 | trot 2.99 | run 7.57 m/s -- cleanly separated, so
# base_lin_vel in the AMP obs acts as the implicit conditioning variable.
# Command 0.8-3.2 spans walk->trot. Run is IN the dataset but far above command: that
# is deliberate, it tests whether a flight-phase gait's mere presence re-legitimises
# the hop we just eliminated. Yaw widened for the turning walk takes.
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO" OMNI_KIT_ACCEPT_EULA=YES ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
export ANIML_DEER_W_FEET_AIR_TIME=0.0 ANIML_DEER_W_JOINT_REF_RANGE=0.0
export ANIML_DEER_W_LIN_VEL_Z=0.0 ANIML_DEER_W_ANG_VEL_XY=0.0
export ANIML_DEER_W_JOINT_POS_LIMITS=-2.0
export ANIML_DEER_CMD_ANG_VEL_Z=0.0
export ANIML_DEER_CMD_VX_LO=1.0 ANIML_DEER_CMD_VX_HI=1.3
export ANIML_DEER_AMP_LEGACY_NORM=1
export ANIML_AMP_DATASETS="deer_polyart_walk:1.0"
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs 4096 --max_iterations 2700 --headless
