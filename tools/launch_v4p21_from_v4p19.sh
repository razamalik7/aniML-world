#!/usr/bin/env bash
# v4.21 = revert weight 3.0 -> 2.0, MORE training time from v4.19.
# Warm-start from v4.19 model_17983.pt (NOT v4.20 which regressed).
#
# v4.20 weight=3.0 backfired:
#   FL/FR hip:     -0.58 -> -0.45  (regressed from walk ref -0.96)
#   straightness:  0.32  -> 0.16   (halved)
#   speed:         0.67  -> 1.04   m/s (faster, less coordinated)
# Policy found a higher-speed local optimum that earns forward_velocity_linear
# reward at the cost of walk pattern coordination.
#
# Reverting to v4.19's weight (2.0). Hypothesis: v4.19 was at the right
# weight but needed more training time to fully converge to the walk pattern.
# 1500 more iters from v4.19's state to see if it continues improving.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_11-23-33/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p21_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.19 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_11-23-33/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.21 from $SRC (v4.19 checkpoint)"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p21_warmstart_${TS}.log"
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

echo "v4.21 launched. Log: $LOG"
echo "PID: $!"
