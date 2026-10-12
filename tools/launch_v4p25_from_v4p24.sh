#!/usr/bin/env bash
# v4.25 = GAIT_DUTY 0.5 -> 0.65 (walk-correct stance fraction).
# Warm-start from v4.24 model_22480.pt.
#
# v4.24 (forward_velocity halved) didn't unlock the gait pattern —
# joint_track 83% unchanged, correlations didn't move much. The
# remaining walk-vs-trot mismatch is the duty cycle: trot has 50%
# stance per foot, walks have 65-75%. With GAIT_DUTY=0.5 the
# gait_contact_match was enforcing trot-like contact durations even
# though the offsets were walk. v4.25 fixes this final inconsistency.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_21-40-05/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p25_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.24 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_21-40-05/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.25 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p25_warmstart_${TS}.log"
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

echo "v4.25 launched. Log: $LOG"
echo "PID: $!"
