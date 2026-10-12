#!/usr/bin/env bash
# v4.28 = add anti_parallel_thigh_l2 reward (weight -0.5).
# Warm-start from v4.27 model_25800.pt.
#
# v4.27 final (1500 iters at GAIT_DUTY=0.65) metrics:
#   |yaw drift|  85° -> 68°            (improved)
#   forward speed 0.80 -> 1.00 m/s     (improved)
#   FL/FR hip   -0.61 -> -0.57         (plateau, ref -0.96)
#   HL/HR hip   -0.22 -> -0.26         (plateau, ref -0.93)
#   FL/HR diag  +0.46 -> +0.52         (WRONG SIGN, ref -0.43)
#   FL/HL lat   -0.34 -> -0.40         (WRONG SIGN, ref +0.47)
#   HL/HR knee  +0.31 -> +0.34         (WRONG SIGN, ref -0.93)
#   carpus fold preserved 95%+
#
# Diagnosis: the policy is satisfying bilateral_symmetry's |L-default|=|R-default|
# constraint via PARALLEL motion (both legs deviate the same way) instead of
# MIRROR motion (opposite deviations). FL/HR moving together visually presents
# as "//" parallel hindquarters (user's observation, confirmed by data).
#
# v4.28 fix: anti_parallel_thigh_l2 penalizes (L_dev + R_dev)^2 — zero for
# perfect mirror, quadratic for parallel. Targets the hip flexion pairs
# directly. Watch for HL/HR hip correlation -0.26 -> closer to -0.93.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_13-14-03/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p28_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.27 checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_13-14-03/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.28 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p28_warmstart_${TS}.log"
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

echo "v4.28 launched. Log: $LOG"
echo "PID: $!"
