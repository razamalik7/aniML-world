#!/usr/bin/env bash
# v4.29r = v4.29 retry with weight halved (-0.5 -> -0.25).
# v4.29 crashed at iter ~320 with `normal expects std >= 0.0`:
# anti_parallel ep_reward grew to -0.61 (3.5x v4.28's -0.17) which
# blew up PPO action_std to NaN.
#
# Warm-start from v4.28 model_27398.pt (NOT from v4.29's broken
# partial run, since that was probably already producing degenerate
# policy states near the crash).
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_15-19-05/model_27398.pt
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p29r_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.28 checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_15-19-05/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.29r from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p29r_warmstart_${TS}.log"
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

echo "v4.29r launched. Log: $LOG"
echo "PID: $!"
