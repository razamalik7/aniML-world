#!/usr/bin/env bash
# v4.29 = extend anti_parallel_thigh_l2 to KNEE pairs.
# Warm-start from v4.28 model_27398.pt.
#
# v4.28 results (anti_parallel only on hip flex pairs, weight -0.5):
#   FL/FR hip   -0.57 -> -0.61   ✓
#   HL/HR hip   -0.26 -> -0.34   ✓ (best ever; ref -0.93)
#   FL/HR diag  +0.52 -> +0.44   ✓ (still wrong sign)
#   HL/HR knee  +0.34 -> +0.43   ✗ (wrong sign GROWING — policy traded knee for hip)
#   straightness 0.18 -> 0.24    ✓
#   |yaw drift|  68° -> 115°     ✗ (policy lost steering freedom)
#   speed        1.00 -> 0.79    ✗
#
# Diagnosis: Adding anti_parallel constraint on HIP only let the policy
# satisfy it by routing the parallel motion into the KNEES. The HL/HR
# knee correlation flipped further wrong (+0.34 -> +0.43) compensating
# for hip improvement. Need to constrain all 4 pairs simultaneously so
# the policy has no escape valve.
#
# v4.29 fix: anti_parallel_thigh_l2 now applies to HL/HR + FL/FR for
# BOTH hip flex AND knee (4 pairs total). Same weight -0.5.
# Watch for HL/HR knee to flip negative.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_15-19-05/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p29_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.28 checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_15-19-05/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.29 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p29_warmstart_${TS}.log"
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

echo "v4.29 launched. Log: $LOG"
echo "PID: $!"
