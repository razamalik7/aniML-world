#!/usr/bin/env bash
# v4.7 = Option B: rsl_rl mirror-symmetry data augmentation.
# Warm-start from v4.6 model_7794.pt (best-symmetry checkpoint so far).
#
# Mirror aug doubles every PPO training batch with L<->R-mirrored copies,
# forcing the network to be equivariant under bilateral reflection. Targets
# the residual front-leg L/R asymmetry (1.32 hindL/hindR ratio) that v4.6's
# per-step reward couldn't fully kill.
#
# Crash mitigations from v4.6 stay in place:
#   * max_linear_velocity / angular: 30/15
#   * --max_iterations 800 per session
#   * OV+DX caches cleared before launch (separate PowerShell call)
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-05_22-09-48/model_7794.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p7_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.6 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-05_22-09-48/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.7 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p7_warmstart_${TS}.log"
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

echo "v4.7 launched. Log: $LOG"
echo "PID: $!"
