#!/bin/bash
# EXP 1 -- "style does the work": strip discriminator-visible proxies, and fix the
# MEASURED problem (saturation: amp_loss 0.298 vs ln2 0.693, style_r 0.75 -> 0.30)
# by tripling the expert set 33 -> 96 frames.
#   feet_air_time 0   : caused the 22%-flight hop; footfalls are implied by joint traj
#   lin_vel_z_l2  0   : base_lin_vel.z is IN the AMP obs -- duplicate
#   ang_vel_xy_l2 0   : base_ang_vel is IN the AMP obs -- duplicate
#   joint_ref_range 0 : joint_pos is IN the AMP obs -- duplicate (probe-L: cold-start blocker)
#   KEPT: base_height / head_height / non_hoof_contact (AMP is BLIND to height,
#         attitude and contact state), joint_pos_limits (cheap safety, probe-L-safe),
#         and the 3 command terms (controllability, not style).
#   walk_alltakes + yaw command widened TOGETHER (turning takes need a turn command).
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMNI_KIT_ACCEPT_EULA=YES
export ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
export ANIML_DEER_W_FEET_AIR_TIME=0.0
export ANIML_DEER_W_JOINT_REF_RANGE=0.0
export ANIML_DEER_W_LIN_VEL_Z=0.0
export ANIML_DEER_W_ANG_VEL_XY=0.0
export ANIML_DEER_W_JOINT_POS_LIMITS=-2.0
export ANIML_AMP_DATASETS="deer_polyart_walk_alltakes:1.0"
export ANIML_DEER_CMD_ANG_VEL_Z=0.5
export ANIML_DEER_AMP_LEGACY_NORM=1
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs 4096 --max_iterations 2500 --headless
