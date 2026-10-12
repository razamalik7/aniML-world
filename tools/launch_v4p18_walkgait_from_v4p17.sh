#!/usr/bin/env bash
# v4.18 = WALK gait offset fix.
# Warm-start from v4.17 model_14985.pt (has most structural fixes baked in).
#
# CRITICAL ROOT CAUSE FIX. Across v4.8-v4.17 the env had been enforcing
# a TROT pattern (GAIT_OFFSETS = (0,0,0.5,0.5)) via gait_contact_match,
# while joint_track_reference_walk indexed the deer_walk.npy reference —
# which is actually a 4-BEAT LATERAL-SEQUENCE WALK with measured
# offsets (FL=0, HR=0.677, FR=0.516, HL=0.161). For 10 iterations the
# two reward signals were FIGHTING each other: gait_contact_match said
# "FL+HR feet on ground together (trot diagonal)", joint_track said
# "FL_thigh tracking phase 0 vs HR_thigh tracking phase 0.677 (walk
# antiphase)". The policy produced a compromise that satisfied neither
# cleanly — the user observed visually as "front legs parallel (//)
# instead of mirrored (/\)", "stepping sideways with one pair", and
# persistent veering.
#
# Fix: GAIT_OFFSETS is now (0.0, 0.677, 0.516, 0.161) in [FL, HR, FR, HL]
# order. Both gait_contact_match and joint_track now agree on the
# walk pattern. No other reward changes.
#
# Extended 800 -> 1500 iters since the policy was trained with TROT
# gait_phase observations and now needs to relearn the action mapping
# under WALK gait_phase observations (cos/sin values shift).
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-08_15-41-04/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p18_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.17 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-08_15-41-04/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.18 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p18_warmstart_${TS}.log"
SEED_NAME=$(basename "$SEED_DIR")
CKPT_NAME=$(basename "$SRC")

cd "$REPO" && \
PYTHONPATH="$REPO" \
ANIML_DEER_USD="$REPO/deer_world.usd" \
ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1 \
OMNI_KIT_ACCEPT_EULA=YES \
nohup /c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train.py \
    --task AniML-Deer-USD-Loco-Stage3-v0 \
    --headless \
    --num_envs 12000 \
    --max_iterations 1500 \
    --resume \
    --load_run "$SEED_NAME" \
    --checkpoint "$CKPT_NAME" > "$LOG" 2>&1 &

echo "v4.18 launched. Log: $LOG"
echo "PID: $!"
