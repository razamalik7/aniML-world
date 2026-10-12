#!/usr/bin/env bash
# One-shot: roll out a polyart AMP checkpoint (GPU) and render each env as
# separate still PNGs (CPU). Run ONLY when no training is using the GPU.
#
#   tools/rig_pipeline/render_checkpoint.sh <ckpt.pt> <tag> [envs] [frames] [steps]
#
# e.g. tools/rig_pipeline/render_checkpoint.sh \
#        logs/rsl_rl/animl_deer_polyart_walk/2026-07-24_12-34-26/model_3500.pt \
#        polyart_r7_i3500
set -e
CKPT="$1"; TAG="$2"; ENVS="${3:-0,1,2,3}"; FRAMES="${4:-30,60,90,120,180}"; STEPS="${5:-200}"
PY="C:/isaac-lab/env_isaaclab/Scripts/python.exe"
URDF="assets/deer_polyart_urdf/deer_polyart.urdf"
OUT="docs/${TAG}_rollout.npz"

echo "[driver] rollout $CKPT -> $OUT (num_envs=4 steps=$STEPS)"
OMNI_KIT_ACCEPT_EULA=YES "$PY" -u tools/rig_pipeline/collect_rollout.py \
  --checkpoint "$CKPT" --num_envs 4 --steps "$STEPS" --out "$OUT"

echo "[driver] stills $OUT -> docs/${TAG}_env*_f*.png"
"$PY" -u tools/rig_pipeline/render_env_stills.py \
  --input "$OUT" --urdf "$URDF" --outdir docs --tag "$TAG" \
  --envs "$ENVS" --frames "$FRAMES"
echo "[driver] done: $TAG"
