#!/usr/bin/env bash
# v4.16 = anti-veer (heading hold + hip abduction symmetry).
# Warm-start from v4.15 model_13387.pt (best baseline so far: carpus folds,
# hind knees correct, taking real steps).
#
# User feedback on v4.15: steps look good, taking real strides, BUT:
#   1. Veering is still terrible — "stepping sideways with one pair of legs"
#   2. Front carpus folds but not deeply enough (deferred to v4.17)
#
# Two changes to address #1:
#   1. NEW heading_hold_penalty (weight -1.5): tracks each env's initial yaw
#      on reset, penalizes cumulative heading drift over the episode. This
#      is fundamentally different from track_yaw_rate_l2 which is per-step.
#      Per-step rate penalty doesn't catch slow accumulating drift.
#
#   2. EXTEND bilateral_symmetry to cover hip ABDUCTION joints:
#      - pelvis_to_HL_hipabd <-> pelvis_to_HR_hipabd (hind hip abd)
#      - chest_to_FL_hipabd  <-> chest_to_FR_hipabd  (front shoulder abd)
#      These were unmapped in the reference, so the policy was free to
#      abduct asymmetrically. Likely the cause of "stepping sideways with
#      one pair" — diagonal pair abducting while opposite pair doesn't.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-08_00-47-08/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p16_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.15 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-08_00-47-08/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.16 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p16_warmstart_${TS}.log"
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

echo "v4.16 launched. Log: $LOG"
echo "PID: $!"
