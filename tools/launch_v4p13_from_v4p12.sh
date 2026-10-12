#!/usr/bin/env bash
# v4.13 = hard hind-knee range penalty.
# Warm-start from v4.12 model_10990.pt (preserves RSI, Cauchy, per-foot).
#
# Across v4.7-v4.12 the HR knee was stuck at -55 to -61 deg regardless of:
#   - per-foot phase indexing (v4.9)
#   - doubled tracking weight (v4.10)
#   - Cauchy reward shape with 40x more gradient (v4.11)
#   - RSI random init from reference frames (v4.12)
#
# Diagnosis: structural equilibrium where HR carries diagonal-pair load
# during trot. Smooth tracking rewards SATURATE — even with gradient at
# all error scales, the policy chooses this configuration because it's a
# stable balance.
#
# Fix: HARD quadratic penalty for hind knees going past ref range
# [-0.55, 0.05] rad. At HR -0.96 rad (v4.12 worst), penalty = 0.41^2 * 5
# = 0.84/step = 840/episode. Dominates joint_track (~8/ep) — forces escape.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_04-58-05/model_10990.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p13_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.12 checkpoint not found: $SRC" >&2
    # Fallback to whatever model_*.pt exists
    SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_04-58-05/model_*.pt | head -1)
    echo "Falling back to: $SRC"
    if [ ! -f "$SRC" ]; then
        echo "FATAL: no v4.12 checkpoints found" >&2
        exit 1
    fi
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_04-58-05/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.13 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p13_warmstart_${TS}.log"
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

echo "v4.13 launched. Log: $LOG"
echo "PID: $!"
