#!/usr/bin/env bash
# v4.12 = RSI (Reference State Initialization) on top of v4.11.
# Warm-start from v4.11 model_10191.pt (Cauchy reward + per-foot phase).
#
# Diagnosis: Across v4.7-v4.11, HR knee was stuck at -58 to -61 deg mean
# regardless of reward shape (exp or Cauchy) or weight (0.5 or 1.0).
# Reason: HR joint default is -0.7 rad (-40 deg). Default reset jitter is
# multiplicative (0.85-1.15), so HR init range is [-0.84, -0.60] — never
# near reference target -0.25 rad. Stiff legs (Kp=60) hold the deer at
# that init pose, and the policy settles into a low-effort balance there.
#
# Fix: Replace reset_robot_joints with reset_from_amp_motion (RSI).
# At each reset, sample a random frame from deer_walk.npy and set joint
# positions to that frame's values. Now HR will sometimes init near -0.25
# rad — the reference target. Combined with v4.11's joint tracking, the
# policy sees the target poses both as init AND as tracking target.
#
# Crash mitigations stay in place:
#   * max_linear_velocity / angular: 30/15
#   * --max_iterations 800 per session
#   * OV+DX caches cleared before launch
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_03-28-58/model_10191.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p12_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.11 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_03-28-58/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.12 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p12_warmstart_${TS}.log"
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

echo "v4.12 launched. Log: $LOG"
echo "PID: $!"
