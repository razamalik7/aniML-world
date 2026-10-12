#!/bin/bash
# G -- B, plus per-hoof horizontal SPEED in the AMP observation. One variable.
#
# WHY. B adds hoof HEIGHT to close the discriminator's blindness to the ground. That
# should fix ankle-walking, which is a genuine pose signature. It will NOT fix leg
# neglect, and this is measured rather than guessed: in A2 the two front hooves reach
# the SAME minimum height -- 0.0 cm and 0.1 cm -- while carrying 0.02 and 0.32 of the
# load. A neglected leg is not frozen (range of motion 0.95x expert) and is not held
# high. It descends all the way and never stops.
#
# Slip is what separates them. In stance a hoof is planted and the body travels over
# it, so its WORLD speed goes to zero; a hoof merely brushing past keeps moving.
# Normalised by body speed:
#     reference   0.04-0.44   all four hooves, across walk, trot AND run
#     exp3b       0.06 0.12 | 0.99 0.86      <- hind plant, front skim
#     A2          0.06 0.45 | 1.55 0.47      <- and slip tracks the missing load:
#                                               slip 1.55 -> share 0.02
#                                               slip 0.47 -> share 0.32
# Six to thirteen times the reference, on the exact legs Raza kept calling out on video,
# and the discriminator cannot see one bit of it. Largest policy/reference discrepancy
# found so far. See docs/LEG_NEGLECT_findings.md.
#
# Physics-free, which was Raza's constraint: this is FK plus a finite difference on the
# reference side and body_lin_vel_w on the sim side. No contact forces are claimed of
# an animation clip that has none.
#
# READ IT AGAINST B, not against exp3 -- G is B plus four columns, nothing else.
#   front-hoof slip falls toward the reference AND load share rises -> the mechanism is
#     confirmed and this is the fix for leg neglect
#   slip falls but load share does not -> the deer learned to hold the hoof still in
#     the air; slip alone is gameable and needs pairing with height (it is already on)
#   neither moves -> the signal is present but too small a share of an 90-dim obs;
#     the next lever is weighting, not more channels
REPO="C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9"
cd "$REPO"
export PYTHONPATH="$REPO" OMNI_KIT_ACCEPT_EULA=YES ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1
export ANIML_DEER_USD="$REPO/assets/deer_polyart_urdf/deer_polyart.usd"
export ANIML_DEER_W_FEET_AIR_TIME=0.0 ANIML_DEER_W_JOINT_REF_RANGE=0.0
export ANIML_DEER_W_LIN_VEL_Z=0.0 ANIML_DEER_W_ANG_VEL_XY=0.0
export ANIML_DEER_W_JOINT_POS_LIMITS=-2.0
export ANIML_DEER_CMD_ANG_VEL_Z=0.0
export ANIML_DEER_CMD_VX_LO=0.8 ANIML_DEER_CMD_VX_HI=3.2
export ANIML_DEER_AMP_LEGACY_NORM=1
export ANIML_DEER_AMP_HOOF_H=1
export ANIML_DEER_AMP_HOOF_V=1        # THE single change vs B
export ANIML_AMP_DATASETS="deer_polyart_walk:1.0,deer_polyart_trot:1.0,deer_polyart_run:0.5"
ITERS="${G_ITERS:-2700}"
ENVS="${G_ENVS:-4096}"
/c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train_amp.py \
  --task AniML-Deer-PolyArt-Walk-AMP-v0 --num_envs "$ENVS" --max_iterations "$ITERS" --headless
