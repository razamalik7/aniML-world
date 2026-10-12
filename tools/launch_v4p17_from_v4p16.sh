#!/usr/bin/env bash
# v4.17 = hipabd_zero_l2 anti-lateral-drift penalty.
# Warm-start from v4.16 model_14186.pt.
#
# v4.16 fixed yaw drift (317° -> 117° drift over 30s) but the deer
# started SIDESTEPPING — straightness ratio went DOWN (0.263 -> 0.214).
# Diagnosis: front hip abduction joints found a COMMON-MODE offset
# (L=+5-8°, R=+10-22° — both shoulders cocked the SAME way). bilateral
# symmetry made L=R but didn't push them to 0. Body leans → lateral
# drift → looks like "stepping sideways with one pair of legs".
#
# Fix: hipabd_zero_l2 (weight -1.5). Penalize magnitude of all 4 hipabd
# joints. Combined with bilateral_symmetry (which forces L=R), the net
# pull is L=R=0. The policy can still abduct ASYMMETRICALLY when
# commanded sideways motion (the cost is just the magnitude).
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-08_13-35-57/model_14186.pt"
if [ ! -f "$SRC" ]; then
    SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-08_13-35-57/model_*.pt | head -1)
fi
if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.16 checkpoint not found" >&2
    exit 1
fi

SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p17_$(date +%Y-%m-%d)"
mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
SRC_DIR=$(dirname "$SRC")
cp -r "$SRC_DIR/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.17 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p17_warmstart_${TS}.log"
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

echo "v4.17 launched. Log: $LOG"
echo "PID: $!"
