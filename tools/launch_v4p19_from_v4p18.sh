#!/usr/bin/env bash
# v4.19 = bump joint_track weight 1.0 -> 2.0.
# Warm-start from v4.18 model_16484.pt (walk gait offsets baked in).
#
# v4.18 unified gait_contact_match and joint_track to both target the
# walk pattern, but the policy STILL didn't produce a walk gait —
# inter-leg correlations stayed near 0 instead of moving toward the
# walk reference (-0.96 for FL/FR, -0.93 for HL/HR, +0.47 for FL/HL).
# Diagnosis: at weight 1.0 the Cauchy reward saturates around 9/12
# even when single joints mistrack by 30+ degrees (Cauchy 0.26 at
# 0.5 rad err). The policy can fake the average while individual
# gait-pattern joints drift.
#
# Doubling weight to 2.0 makes joint_track contribute up to 24/step.
# Single-joint mistracking now costs 2x more reward, forcing the
# policy to commit to the reference's per-joint phase shape.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-09_02-30-38/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p19_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.18 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-09_02-30-38/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.19 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p19_warmstart_${TS}.log"
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

echo "v4.19 launched. Log: $LOG"
echo "PID: $!"
