#!/usr/bin/env bash
# v4.26 = continue v4.25 for more training time at GAIT_DUTY=0.65.
# Warm-start from v4.25 model_23979.pt.
#
# v4.25 introduced GAIT_DUTY=0.65 (walk-correct stance fraction) but
# the gait_contact_match reward dropped 2.31 -> 1.68 ep_reward,
# indicating the policy is still doing ~50% stance (trot-like) and
# violating the new 65% target. Joint_track unchanged at 83%.
# Possibility A: policy needs more training time to adapt.
# Possibility B: the new duty target is too aggressive and policy
# will never converge.
#
# v4.26 tests A by simply continuing training.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_23-45-06/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p26_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.25 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_23-45-06/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.26 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p26_warmstart_${TS}.log"
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

echo "v4.26 launched. Log: $LOG"
echo "PID: $!"
