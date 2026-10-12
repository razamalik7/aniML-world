#!/usr/bin/env bash
# v4.15 = front_carpus_fold_penalty (sister to hind_knee_range_penalty).
# Warm-start from v4.14 model_12588.pt (sign fix in place).
#
# v4.14 fixed the carpus sign-convention bug — FL/FR carpus can now
# physically fold (reaches +30 to +67 deg peaks during swing). But the
# policy spends 78-98% of time CLAMPED at the URDF lower limit -0.3 rad
# (extension direction). Stiff actuator (Kp=60) pulls toward extension;
# joint_track signal too weak vs the stiffness cost.
#
# Same fix as v4.13's HR knee: HARD quadratic penalty when carpus is
# OUTSIDE the fold range [+0.10, +1.30] rad. At -0.3 rad (current clamp),
# violation = 0.4 rad, penalty = 0.4^2 * 5 = 0.8/step = 800/episode.
# Dominates joint_track (~8.5/ep) — forces escape from the extension trap.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_19-36-52/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p15_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.14 checkpoint not found" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_19-36-52/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.15 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p15_warmstart_${TS}.log"
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

echo "v4.15 launched. Log: $LOG"
echo "PID: $!"
