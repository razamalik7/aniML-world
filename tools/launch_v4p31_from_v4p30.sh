#!/usr/bin/env bash
# v4.31 = heading_hold weight -1.5 -> -2.5 to fix v4.30's |yaw drift| 213°.
# anti_parallel_thigh stays at v4.30's -0.35 (4 pairs).
# Warm-start from v4.30 model_30396.pt.
#
# v4.30 results: bilateral correlations got best-ever HL/HR hip -0.43,
# FL/HR diag closing to +0.38, FL/HL lat closing to -0.29. BUT
# |yaw drift| went to 213° — anti_parallel removed asymmetric
# steering freedom, heading_hold couldn't compensate at -1.5.
#
# Hypothesis: 67% bump in heading_hold weight tips the balance back
# toward heading control without abandoning coordination gains.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_20-28-15/model_30396.pt
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p31_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.30 checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_20-28-15/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.31 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p31_warmstart_${TS}.log"
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

echo "v4.31 launched. Log: $LOG"
echo "PID: $!"
