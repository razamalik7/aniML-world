#!/usr/bin/env bash
# v5.2 FRESH stable base — train from scratch, NO warm-start.
# Task: AniML-Deer-USD-Loco-Stable-Flat-v0 (Stage1 forward-only recipe +
# naturalness rewards stripped). Learns a clean stable walk under the stable
# reward directly, avoiding the warm-start instability that sank v5.0/v5.1.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v5p2_stableflat_${TS}.log"

cd "$REPO" && \
PYTHONPATH="$REPO" \
ANIML_DEER_USD="$REPO/deer_world.usd" \
ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1 \
OMNI_KIT_ACCEPT_EULA=YES \
nohup /c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train.py \
    --task AniML-Deer-USD-Loco-Stable-Flat-v0 \
    --headless \
    --num_envs 12000 \
    --max_iterations 2000 > "$LOG" 2>&1 &

echo "v5.2 stableflat (fresh) launched. Log: $LOG"
echo "PID: $!"
