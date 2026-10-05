#!/bin/bash
# Download data and checkpoints from Hugging Face into $DATA.
#   bash scripts/download.sh          pill demonstrations, pre-training data (77 GB) and all checkpoints
#   bash scripts/download.sh small    skip the pre-training data; use the released 250 selected episodes instead
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
dl() { $DP_PY - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rtype, dst = sys.argv[1:4]
snapshot_download(repo, repo_type=rtype, local_dir=dst, allow_patterns=sys.argv[4:] or None)
PY
}
mkdir -p $DATA/zarr $DATA/geometry
dl ShuaKang/robotwin_pill_rara dataset $DATA/hf_pill
dl ShuaKang/RARA-robotwin-pretrain model $DATA/weights
dl ShuaKang/RARA-robotwin-pill model $DATA/weights
ln -sfn $DATA/hf_pill/dp_zarr/narrow_move_pillbottle_pad.zarr $DATA/zarr/pill_target.zarr
ln -sfn $DATA/hf_pill/geometry/narrow_move_pillbottle_pad.npz $DATA/geometry/pill_target.npz
if [[ ${1:-} == small ]]; then
  ln -sfn $DATA/hf_pill/dp_zarr/ret250L_move_pillbottle_pad_ws2k.zarr $DATA/zarr/pill_retrieved.zarr
  ln -sfn $DATA/hf_pill/geometry/ret250L_move_pillbottle_pad_ws2k.npz $DATA/geometry/pill_retrieved.npz
else
  dl ShuaKang/my_roboTwin2.0_training dataset $DATA/hf_pretrain "dp_zarr_ws/pre_ws_2k.*"
  ln -sfn $DATA/hf_pretrain/dp_zarr_ws/pre_ws_2k.zarr          $DATA/zarr/pretrain.zarr
  ln -sfn $DATA/hf_pretrain/dp_zarr_ws/pre_ws_2k.npz           $DATA/geometry/pretrain.npz
  ln -sfn $DATA/hf_pretrain/dp_zarr_ws/pre_ws_2k.episodes.json $DATA/geometry/pretrain.episodes.json
fi
ls -l $DATA/zarr $DATA/geometry $DATA/weights
