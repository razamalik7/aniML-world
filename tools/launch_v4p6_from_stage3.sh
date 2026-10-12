#!/usr/bin/env bash
# v4.6 = Option A: per-step L2 bilateral symmetry penalty + crash mitigations.
# Warm-start from Stage 3 model_6995.pt (cleanest L/R baseline).
#
# Changes vs v4.5:
#   * bilateral_symmetry: EMA -> per-step L2, weight -0.5 -> -1.0
#   * max_linear_velocity / angular: 50/20 -> 30/15 (PhysX NaN suppression)
#   * --max_iterations 1500 -> 800 (avoid the silent-death zone)
#   * OV + DX caches cleared before launch (separate PowerShell call)
#
# Action / observation / joint dims unchanged; no padding needed.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-05_05-32-03/model_6995.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p6_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: Stage 3 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-05_05-32-03/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.6 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p6_warmstart_${TS}.log"
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

echo "v4.6 launched. Log: $LOG"
echo "PID: $!"
