#!/usr/bin/env bash
# v4.22 = lower gait_contact_match weight 2.0 -> 1.0.
# Warm-start from v4.21 model_19482.pt (essentially v4.19 + more training).
#
# v4.19/v4.21 plateaued at HL/HR hip correlation -0.10 (target -0.93).
# Diagnosis: gait_contact_match at weight 2.0 contributes ~4.80/ep —
# the policy satisfies foot-contact timing via this signal without
# deeply committing to joint_track's per-foot phase shape (which
# specifies the actual leg motion). Halving its weight forces
# joint_track (weight 2.0, ~9.8 ep_reward) to dominate the gait pattern.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_15-40-09/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p22_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.21 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-10_15-40-09/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.22 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p22_warmstart_${TS}.log"
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

echo "v4.22 launched. Log: $LOG"
echo "PID: $!"
