#!/usr/bin/env bash
# v4.32 = continue v4.31r config for more training time.
# Warm-start from v4.31r model_30396.pt.
#
# v4.31r results (anti_parallel -0.35 + heading_hold -2.0):
#   HL/HR hip   -0.43 -> -0.49  BEST ever  (ref -0.93)
#   FL/HR diag  +0.38 -> +0.30  best, closing to ref -0.43
#   FL/HL lat   -0.29 -> -0.24  best, closing to ref +0.47
#   straightness 0.16 -> 0.29   biggest jump yet (toward >0.5)
#   |yaw drift| 213° -> 93°    recovered (still > v4.27's 68°)
#   speed       0.61 -> 0.54   regressing (concerning)
#   FL/FR hip   -0.36 -> -0.31  still weak vs v4.28's -0.61
#
# Trajectory is positive. Hypothesis: more iterations at current
# reward weights = continued convergence of all correlations.
# No reward changes this round.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_23-39-02/model_30396.pt
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p32_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.31r checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_23-39-02/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.32 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p32_warmstart_${TS}.log"
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

echo "v4.32 launched. Log: $LOG"
echo "PID: $!"
