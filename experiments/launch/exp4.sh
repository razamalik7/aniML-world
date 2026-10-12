#!/bin/bash
# EXP 4 -- ASE stage 1. Policy conditioned on a random unit latent z (resampled per
# episode); reward = style + MI (encoder recovering z). Task reward is NOT removed
# outright -- probe-N/probe-A showed a from-scratch deer needs SOME drive or it never
# leaves the floor -- but it is the thin command set only, no gait proxies.
# Dataset = locomotive clips only. Sleep/Fall/Death/Actions excluded: a latent built
# from collapsed poses would put "lie down" next to "walk" on the dial.
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
export ANIML_AMP_DATASETS="deer_polyart_walk_alltakes:1.0,deer_polyart_trot:1.0,deer_polyart_run:0.5"
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs 4096 --max_iterations 1500 --headless
