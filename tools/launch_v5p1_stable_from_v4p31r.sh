#!/usr/bin/env bash
# v5.1 STABLE-BASE pivot. Task: AniML-Deer-USD-Loco-Stable-v0.
# Naturalness rewards zeroed, anti-porpoise (lin_vel_z_l2) strengthened -2 -> -4.
#
# Warm-start from v4.31r model_30396 (the best naturalness policy — it can
# already walk forward, stay upright, and turn). Removing the destabilizing
# naturalness rewards should let it relax OUT of the 48cm porpoise + heading
# weave toward a solid, boring, controllable walk. Fast signal expected:
# watch lin_vel_z_l2 ep_reward climb toward 0 (less bounce) and the base
# height range tighten.
#
# If the pathologies DON'T shed within ~800 iters (policy too entrenched),
# fall back to a fresh Stage1->2->3 curriculum run of the Stable config.
set -euo pipefail

REPO=C:/Users/nars7/Projects/aniML-world/.claude/worktrees/jolly-clarke-e00da9
SRC=$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_23-39-02/model_30396.pt
# NOTE: seed dir MUST live under the runner cfg's experiment_name
# ("animl_deer_loco_stable") so --resume/--load_run resolves it.
SEED_DIR="$REPO/logs/rsl_rl/animl_deer_loco_stable/seed_v5p1_$(date +%Y-%m-%d)"

if [ ! -f "$SRC" ]; then echo "ERROR: v4.31r checkpoint not found" >&2; exit 1; fi

mkdir -p "$SEED_DIR"
cp "$SRC" "$SEED_DIR/"
cp -r "$REPO/logs/rsl_rl/animl_deer_loco_stage3/2026-05-13_23-39-02/params" "$SEED_DIR/" 2>/dev/null || true
echo "Seeded v5.1 from $SRC"

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/loco_v5p1_stable_${TS}.log"
SEED_NAME=$(basename "$SEED_DIR")
CKPT_NAME=$(basename "$SRC")

cd "$REPO" && \
PYTHONPATH="$REPO" \
ANIML_DEER_USD="$REPO/deer_world.usd" \
ANIML_DEER_ACTIVATE_CONTACT_SENSORS=1 \
OMNI_KIT_ACCEPT_EULA=YES \
nohup /c/isaac-lab/env_isaaclab/Scripts/python.exe training/scripts/train.py \
    --task AniML-Deer-USD-Loco-Stable-v0 \
    --headless \
    --num_envs 12000 \
    --max_iterations 1500 \
    --resume \
    --load_run "$SEED_NAME" \
    --checkpoint "$CKPT_NAME" > "$LOG" 2>&1 &

echo "v5.1 stable launched. Log: $LOG"
echo "PID: $!"
