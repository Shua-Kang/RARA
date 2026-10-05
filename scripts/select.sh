#!/bin/bash
# Select the 250 pick-and-place pre-training episodes closest to the pill demonstrations (CPU, a few minutes).
# Output: $DATA/zarr/pill_retrieved.zarr and $DATA/geometry/pill_retrieved.npz (used by Co-FT and RARA).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p $DATA/zarr $DATA/geometry $DATA/select
$DP_PY $REPO/rara/select.py --pool $DATA/zarr/pretrain.zarr --pool_geom $DATA/geometry/pretrain.npz \
  --pool_episodes $DATA/geometry/pretrain.episodes.json --target_geom $DATA/geometry/pill_target.npz \
  --task place_a2b_right --k ${K:-250} --out $DATA/select/pill_retrieved
ln -sfn $DATA/select/pill_retrieved.zarr $DATA/zarr/pill_retrieved.zarr
ln -sfn $DATA/select/pill_retrieved.npz  $DATA/geometry/pill_retrieved.npz
