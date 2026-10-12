#!/usr/bin/env bash
# Launch Stage 3 (rough terrain) warm-start from a Stage 2 checkpoint.
# Usage:
#   bash tools/launch_stage3_from_stage2.sh <stage2_run_dir_name> <checkpoint_name>
# Example:
#   bash tools/launch_stage3_from_stage2.sh 2026-05-05_03-42-40 model_5497.pt
set -euo pipefail

STAGE2_RUN="${1:?need stage2 run dir name (e.g. 2026-05-05_03-42-40)}"
CHECKPOINT="${2:?need checkpoint name (e.g. model_5497.pt)}"

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage2/$STAGE2_RUN/$CHECKPOINT"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_from_stage2_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: Stage 2 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage2/$STAGE2_RUN/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded Stage 3 from $SRC"
echo "Seed dir: $SEED_DIR"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_stage3_warmstart_${TS}.log"
SEED_NAME=$(basename "$SEED_DIR")

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
    --checkpoint "$CHECKPOINT" > "$LOG" 2>&1 &

echo "Stage 3 launched as background process. Log: $LOG"
echo "PID: $!"
