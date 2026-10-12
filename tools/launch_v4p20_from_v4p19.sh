#!/usr/bin/env bash
# v4.20 = bump joint_track weight 2.0 -> 3.0.
# Warm-start from v4.19 model_17983.pt.
#
# v4.19 (weight 2.0) showed clear progress on the gait pattern:
#   FL/FR hip correlation:  -0.40 -> -0.58 (target -0.96)
#   HL/HR hip correlation:  +0.21 -> -0.10 (no longer parallel, target -0.93)
#   Straightness ratio:    0.132 -> 0.323 (target > 0.5)
#   Front-knee correlation: +0.16 -> -0.33 (anti-phase achieved)
#
# But HL/HR still at -0.10 (not anti-phase yet), FL/HL at -0.35 (wrong
# direction; should be +0.47 for ipsilateral walk sync). Continuing the
# same intervention with stronger pull. Single change: weight 2 -> 3.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_11-23-33/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p20_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.19 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_11-23-33/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.20 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p20_warmstart_${TS}.log"
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

echo "v4.20 launched. Log: $LOG"
echo "PID: $!"
