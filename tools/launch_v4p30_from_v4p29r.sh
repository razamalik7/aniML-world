#!/usr/bin/env bash
# v4.30 = anti_parallel weight -0.35 (between v4.28's -0.5 crash and
# v4.29r's -0.25 too-soft). Warm-start from v4.29r model_28897.pt.
#
# v4.29r at -0.25 (4 pairs): no NaN, but FL/FR hip regressed -0.61 -> -0.24.
# v4.29 at -0.5 (4 pairs): crashed iter 320 (action_std NaN).
# Midpoint -0.35: keeps 70% of v4.29's signal magnitude.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_18-19-47/model_28897.pt
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p30_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.29r checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_18-19-47/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.30 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p30_warmstart_${TS}.log"
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

echo "v4.30 launched. Log: $LOG"
echo "PID: $!"
