#!/bin/bash
# r16 PHASE 1 -- from scratch, forgiving entry.
#   joint_ref_range 0    : dropped entirely (probe-L: sole reward-side cold-start blocker;
#                          backstop for a normaliser bug that is now fixed; ASE-incompatible)
#   joint_pos_limits -2.0: kept (probe-L: cold-start-safe, protects hard end-stops)
#   legacy normaliser    : forgiving discriminator during infancy (probe-N)
#   feet_air_time 20.0   : the gait experiment -- was 1/700th of forward_velocity
# Phase 2 warm-starts from this run's checkpoint with the expert-only normaliser,
# switched adaptively once ep_len says the policy is ready. Reverting = relaunch
# phase 2 from the same checkpoint with different settings.
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMNI_KIT_ACCEPT_EULA=YES
export ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
export ANIML_DEER_W_JOINT_REF_RANGE=0.0
export ANIML_DEER_W_JOINT_POS_LIMITS=-2.0
export ANIML_DEER_W_FEET_AIR_TIME=20.0
export ANIML_DEER_AMP_LEGACY_NORM=1
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs 4096 --max_iterations 3000 --headless
