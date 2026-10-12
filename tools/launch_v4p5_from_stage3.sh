#!/usr/bin/env bash
# Launch v4.5 warm-start from Stage 3 final (model_6995.pt — cleanest L/R baseline).
# v4.5 adds three new reward terms on top of v4.4:
#   * carpus_flex_during_swing (tanh, weight 0.7) — smooth gradient toward modest flex
#   * carpus_deep_flex_bonus (deadbanded linear, weight 1.5) — kicks in past 29 deg flex,
#     can't be gamed by parking shallow
#   * bilateral_joint_symmetry_ema (weight -0.5) — penalize sustained L/R asymmetry of
#     |q-default| across 6 joint pairs (hind/front knees, hocks/carpi, hip flexes)
# Warm-start from Stage 3 (1.27 hindL/hindR ratio, 0.4 deg/s drift) instead of v4.4
# (1.49 ratio, 4.5 deg/s) — symmetry penalty has less work to undo.
# Action / observation / joint dims unchanged — no padding needed.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-05_05-32-03/model_6995.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p5_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: Stage 3 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-05_05-32-03/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.5 from $SRC"
echo "Seed dir: $SEED_DIR"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p5_warmstart_${TS}.log"
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

echo "v4.5 launched as background process. Log: $LOG"
echo "PID: $!"
