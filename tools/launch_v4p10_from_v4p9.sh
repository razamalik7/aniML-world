#!/usr/bin/env bash
# v4.10 = stronger commitment to reference + anti-circling.
# Warm-start from v4.9 model_9392.pt (preserves per-foot phase mechanism
# and the hind L/R symmetry win, 1.61 -> 0.98).
#
# Changes vs v4.9:
#   * joint_track_reference_walk weight: 0.5 -> 1.0 (force commitment, fix
#     vanishing-gradient at HR knee deep flex)
#   * track_yaw_rate_l2 weight: -1.5 -> -3.0 (anti-circling)
#
# Hypothesis: doubled tracking weight pulls FL ROM down toward FR (and
# both toward reference 18.8 deg ROM), and pulls HR knee out of -58 deg
# stuck local optimum toward reference -14 mean. Doubled yaw penalty
# stops the circling that's been present in v4.7/v4.8/v4.9.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_00-54-05/model_9392.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p10_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.9 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_00-54-05/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.10 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p10_warmstart_${TS}.log"
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

echo "v4.10 launched. Log: $LOG"
echo "PID: $!"
