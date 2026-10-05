#!/bin/bash
# RARA Stage 1: align the vision encoder of the pre-trained policy on pill frames to geometry-matched frames of the
# retrieved pre-training data (one GPU, < 1 h). Output: $RUNS/<NAME>/checkpoints/aligned.pt
# Optional: P (default the downloaded pre-trained policy), NAME, SEED, STEPS (3000), WANDB_PROJECT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SEED=${SEED:-42}; P=${P:-$DATA/weights/pretrain_ws2k.pt}; NAME=${NAME:-pill_stage1_s${SEED}}
$DP_PY $REPO/rara/stage1.py --init $P --out $RUNS/$NAME \
  --target $DATA/zarr/pill_target.zarr --target_geom $DATA/geometry/pill_target.npz \
  --pool $DATA/zarr/pill_retrieved.zarr --pool_geom $DATA/geometry/pill_retrieved.npz \
  --steps ${STEPS:-3000} --lam 1.0 --mpos 4 --tau 0.07 --lr 1e-5 --wandb "${WANDB_PROJECT:-}"
echo "done: $RUNS/$NAME/checkpoints/aligned.pt"
