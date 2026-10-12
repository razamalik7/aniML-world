#!/usr/bin/env bash
# v4.27 = continue v4.26 (which died from PhysX silent death at 421 iters).
# Warm-start from v4.26 model_24400.pt.
#
# v4.26 partial results showed:
#   |yaw drift| 306° -> 85° (env 2 only 11°!) — GAIT_DUTY=0.65 helping
#   FL/FR hip -0.53 -> -0.61 (modest)
#   HL/HR hip -0.21 -> -0.22 (no real change)
# Run was cut short by PhysX silent death. Continue with cleaner cache.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$(ls -1t $REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-11_01-50-28/model_*.pt | head -1)
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stage3/seed_v4p27_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.26 checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-11_01-50-28/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v4.27 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v4p27_warmstart_${TS}.log"
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

echo "v4.27 launched. Log: $LOG"
echo "PID: $!"
