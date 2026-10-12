#!/usr/bin/env bash
# v4.8 = Option 3: per-joint reference-walk tracking.
# Warm-start from v4.7 model_8593.pt (best-symmetry checkpoint, mirror-aug).
#
# Replaces the soft swing-phase bonuses that the policy kept gaming
# (knee_flex_during_swing, carpus_flex_during_swing, carpus_deep_flex_bonus
# all set to weight=0) with a single explicit reward that imitates the
# deer_walk.npy reference cycle joint-by-joint:
#
#   r = sum_j exp(-(q_j - q_ref_j(phase))^2 / sigma^2)
#
# 12 leg joints tracked, sigma=0.3 rad (~17 deg tolerance), weight=0.3.
# Phase comes from the existing 2 Hz gait clock (matches GAIT_FREQ).
#
# Why this should beat AMP: AMP's discriminator was a black box the policy
# could fool via proxy features (root velocity profile, contact timing) — it
# learned to *look like* a deer walk to the discriminator without actually
# bending legs. Direct tracking has no proxy: a joint either matches
# q_ref(phase) or it doesn't. No way to game the reward by sliding.
#
# Crash mitigations from v4.6/v4.7 stay in place:
#   * max_linear_velocity / angular: 30/15
#   * --max_iterations 800 per session
#   * OV+DX caches cleared before launch (separate PowerShell call)
#
# Mirror-aug from v4.7 stays on (RslRlSymmetryCfg in deer_usd_loco_cfg.py).
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-06_15-47-36/model_8593.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p8_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.7 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-06_15-47-36/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.8 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p8_warmstart_${TS}.log"
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

echo "v4.8 launched. Log: $LOG"
echo "PID: $!"
