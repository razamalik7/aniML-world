#!/usr/bin/env bash
# v4.11 = Cauchy reward shape + revert yaw weight.
# Warm-start from v4.9 model_9392.pt (cleanest baseline; v4.10 destabilized
# without fixing HR — start from a known-good state).
#
# Changes vs v4.10:
#   * joint_track_reference_walk: exp(-err²/σ²) → 1/(1+err²/σ²) (Cauchy)
#     - At err=0.7 rad (typical HR error), exp gives 0.004 — nothing.
#       Cauchy gives 0.155 — 40× more gradient.
#   * track_yaw_rate_l2 weight: -3.0 → -1.5 (revert; -3.0 made it worse)
#
# Reward weight stays at 1.0; sigma stays at 0.3.
#
# Hypothesis: Cauchy provides gradient at the large errors HR knee is
# stuck at, so the policy can finally pull HR up toward reference's -14°
# mean. Once HR comes up, the asymmetric loading driving the circling
# disappears, and yaw drift fixes itself.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC="$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_00-54-05/model_9392.pt"
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p11_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then
    echo "ERROR: v4.9 checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-07_00-54-05/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.11 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p11_warmstart_${TS}.log"
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

echo "v4.11 launched. Log: $LOG"
echo "PID: $!"
