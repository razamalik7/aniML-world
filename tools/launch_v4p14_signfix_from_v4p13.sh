#!/usr/bin/env bash
# v4.14 = SIGN CORRECTION for FL/FR carpus joints.
# Warm-start from v4.13 model_11789.pt (preserves hind-knee fix and RSI).
#
# CRITICAL BUG FOUND in v4.13 review: the URDF has OPPOSITE sign conventions
# for front carpus (FL/FR_shin_to_FL/FR_foot) vs hind hock (HL/HR_shin_to_HL/HR_foot):
#   - Front carpus URDF limit: [-0.3, +1.6]  (POSITIVE = fold direction)
#   - Hind hock   URDF limit: [-1.6, +0.3]  (NEGATIVE = fold direction)
# The reference walk uses a SINGLE convention (negative = fold for all legs).
# So the reference's FL carpus values [-1.2, -0.2] were LITERALLY OUTSIDE the
# URDF joint limits — the policy was being pulled toward unreachable angles.
# It gave up and the natural extension took over, producing the "leg held
# straight out forward" failure mode the user observed visually.
#
# Fix: negate the reference values for FL/FR_shin_to_FL/FR_foot before
# computing tracking error. Same correction in _load_reset_states so RSI
# inits the deer with carpus FOLDED (positive q in URDF) instead of
# extending and getting clamped at -0.3 rad.
#
# v4.7-v4.13's "carpus ROM matches reference" reports were misleading:
# the ROM magnitude was right but the DIRECTION was wrong. Real test is
# whether the deer visually folds its front cannons properly during swing.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_10-52-26/model_11789.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p14_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.13 checkpoint not found: $SRC" >&2
    SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_10*/model_*.pt | head -1)
    echo "Falling back to: $SRC"
    if [ ! -f "$SRC" ]; then
        echo "FATAL: no v4.13 checkpoints found" >&2
        exit 1
    fi
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
SRC_DIR=$(dirname "$SRC")
cp -r "$SRC_DIR/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.14 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p14_warmstart_${TS}.log"
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
    --max_iterations 800 \
    --resume \
    --load_run "$SEED_NAME" \
    --checkpoint "$CKPT_NAME" > "$LOG" 2>&1 &

echo "v4.14 launched. Log: $LOG"
echo "PID: $!"
