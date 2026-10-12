#!/bin/bash
# EXP 2 -- fabricated variety from ONE clip. Same reward set as Exp 1; the ONLY
# difference is the expert data: 15 synthetic variants of the single DWalk take
# (speeds 0.8-1.2x, 3 phase rolls each, 0.75 deg smoothed jitter) = 507 frames vs 33.
# Validated before use: no teleports, no NaNs, unit quaternions, speeds 0.89-1.35 m/s.
# Yaw stays 0 -- these are all STRAIGHT walks, unlike Exp 1's turning takes.
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO" OMNI_KIT_ACCEPT_EULA=YES ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
export ANIML_DEER_W_FEET_AIR_TIME=0.0 ANIML_DEER_W_JOINT_REF_RANGE=0.0
export ANIML_DEER_W_LIN_VEL_Z=0.0 ANIML_DEER_W_ANG_VEL_XY=0.0
export ANIML_DEER_W_JOINT_POS_LIMITS=-2.0
export ANIML_DEER_CMD_ANG_VEL_Z=0.0
export ANIML_DEER_AMP_LEGACY_NORM=1
export ANIML_AMP_DATASETS="walkaug_s0p8_p0:1.0,walkaug_s0p8_p1:1.0,walkaug_s0p8_p2:1.0,walkaug_s0p9_p0:1.0,walkaug_s0p9_p1:1.0,walkaug_s0p9_p2:1.0,walkaug_s1_p0:1.0,walkaug_s1_p1:1.0,walkaug_s1_p2:1.0,walkaug_s1p1_p0:1.0,walkaug_s1p1_p1:1.0,walkaug_s1p1_p2:1.0,walkaug_s1p2_p0:1.0,walkaug_s1p2_p1:1.0,walkaug_s1p2_p2:1.0"
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs 4096 --max_iterations 1500 --headless
