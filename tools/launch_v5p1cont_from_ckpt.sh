#!/usr/bin/env bash
# v5.1-continue: resume the stable run after a PhysX Fatal crash at iter ~30742
# (2026-07-22 02:15). Config unchanged (v5.1 reward was healthy: base_tilted
# ~15%, walking forward). Resume from the last good checkpoint to preserve the
# ~300 iters of adaptation. Caches already cleared per PhysX playbook.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
EXP_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stable/2026-07-22_01-46-17"
CKPT=$(ls -1t "$EXP_DIR"/model_*.pt | head -1)
CKPT_NAME=$(basename "$CKPT")
RUN_NAME=$(basename "$EXP_DIR")

if [ ! -f "$CKPT" ]; then echo "ERROR: no v5.1 checkpoint found" >&2; exit 1; fi
echo "Resuming from $RUN_NAME / $CKPT_NAME"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v5p1cont_${TS}.log"

cd "$REPO" && \
PYTHONPATH="$REPO" \
ANIML_DEER_USD="$REPO/deer_world.usd" \
ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1 \
OMNI_KIT_ACCEPT_EULA=YES \
nohup /c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train.py \
    --task AniML-Deer-USD-Loco-Stable-v0 \
    --headless \
    --num_envs 12000 \
    --max_iterations 1200 \
    --resume \
    --load_run "$RUN_NAME" \
    --checkpoint "$CKPT_NAME" > "$LOG" 2>&1 &

echo "v5.1-continue launched. Log: $LOG"
echo "PID: $!"
