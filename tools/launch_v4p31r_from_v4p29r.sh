#!/usr/bin/env bash
# v4.31r = v4.31 retry. Two changes together:
#   - heading_hold weight -1.5 -> -2.0 (smaller bump than the -2.5 that crashed)
#   - anti_parallel_thigh weight -0.35 (4 pairs; same as v4.30)
# Warm-start from v4.29r model_28897 (clean: yaw 68°, action_std 0.82
# stable). Avoid v4.30's broken-yaw policy as start state.
#
# v4.31 crashed because: bumping heading_hold to -2.5 ON A POLICY THAT
# ALREADY DRIFTS 213° → cumulative ep penalty huge → NaN gradient.
# v4.31r mitigates both:
#   (a) cleaner start state (yaw 68° not 213°)
#   (b) modest bump (33% instead of 67%)
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_18-19-47/model_28897.pt
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p31r_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.29r checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_18-19-47/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.31r from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p31r_warmstart_${TS}.log"
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

echo "v4.31r launched. Log: $LOG"
echo "PID: $!"
