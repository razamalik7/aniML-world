#!/usr/bin/env bash
# v4.9 = per-foot phase-indexed reference tracking.
# Warm-start from v4.7 model_8593.pt (NOT v4.8 — v4.8's hind-leg spasm is
# a hard local optimum and v4.7's clean balance is a better starting point).
#
# Fix vs v4.8: v4.8 indexed all 12 joints by ONE global phase, so reference's
# walk-cycle joint angles fought env's enforced trot timing. One hind leg
# spasmed mid-air while the other tracked normally; HL/HR asymmetry blew up
# from 0.99 to 1.61 and yaw drift worsened.
#
# v4.9: each foot looks up its joint targets at its OWN gait phase using
# the same foot_phase_offsets the trot uses (0,0,0.5,0.5). Joint tracking
# becomes consistent with gait_contact_match, no fighting.
#
# Weight 0.3 -> 0.5 since the conflict is gone, safe to commit harder.
#
# Crash mitigations from v4.6/v4.7/v4.8 stay in place:
#   * max_linear_velocity / angular: 30/15
#   * --max_iterations 800 per session
#   * OV+DX caches cleared before launch (separate PowerShell call)
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-06_15-47-36/model_8593.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p9_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.7 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-06_15-47-36/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.9 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p9_warmstart_${TS}.log"
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

echo "v4.9 launched. Log: $LOG"
echo "PID: $!"
