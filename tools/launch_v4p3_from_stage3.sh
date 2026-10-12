#!/usr/bin/env bash
# Launch v4.3 warm-start from Stage 3 final checkpoint.
# Same Stage 3 task (rough terrain + 3-DOF mobility), but with v4.3 reward edits:
#   * knee_flex_during_swing  flex_target 0.4 -> 0.9
#   * NEW carpus_flex_during_swing on shin->foot joints (the visual "front knee")
#   * undesired_contacts cannon penalty -0.5 -> -0.3
# Action / observation / joint dims are unchanged, so no checkpoint padding needed.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-05_05-32-03/model_6995.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p3_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: Stage 3 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-05_05-32-03/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.3 from $SRC"
echo "Seed dir: $SEED_DIR"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p3_warmstart_${TS}.log"
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

echo "v4.3 launched as background process. Log: $LOG"
echo "PID: $!"
