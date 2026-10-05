#!/bin/bash
# Train one policy. Uses all visible GPUs (torchrun/DDP when more than one).
#   MODE=pretrain                         pre-training on the 2000-episode set (60 epochs, batch 256)
#   MODE=ft    [P=<pretrain.pt>]          FT:    fine-tune on the 50 pill demonstrations (300 epochs)
#   MODE=coft  [P=<pretrain.pt>]          Co-FT: fine-tune on pill demos + retrieved pre-training data (64 + 64 per batch)
#   MODE=rara  [P=...] [S1=<stage1.pt>]   RARA Stage 2: start from the Stage-1 encoder, BC on pill demos, feature
#                                         anchor to P on the retrieved data (run scripts/stage1.sh first)
# Optional: NAME, SEED (42), EPOCHS, EXTRA (Hydra overrides), WANDB_MODE (online|offline|disabled).
# Output: $RUNS/<NAME>/checkpoints/epochXXXX.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${MODE:?MODE=pretrain|ft|coft|rara}"
SEED=${SEED:-42}; P=${P:-$DATA/weights/pretrain_ws2k.pt}; NAME=${NAME:-pill_${MODE}_s${SEED}}
TGT=$DATA/zarr/pill_target.zarr; RET=$DATA/zarr/pill_retrieved.zarr
OV=()
case $MODE in
  pretrain) NAME=${NAME/pill_pretrain/pretrain}; EPOCHS=${EPOCHS:-60}
            ZP="[$DATA/zarr/pretrain.zarr]"; MR="[1]"; OV+=(dataloader.batch_size=256 training.checkpoint_every=10);;
  ft)       ZP="[$TGT]"; MR="[1]"; OV+=(training.init_ckpt=$P);;
  coft)     ZP="[$TGT,$RET]"; MR="[1,1]"; OV+=(training.init_ckpt=$P);;
  rara)     S1=${S1:-$RUNS/pill_stage1_s${SEED}/checkpoints/aligned.pt}
            [[ -f $S1 ]] || { echo "missing Stage-1 weights $S1 (run scripts/stage1.sh)"; exit 2; }
            ZP="[$TGT,$RET]"; MR="[1,1]"
            OV+=(training.init_ckpt=$S1 training.anchor_ref_ckpt=$P "training.anchor_parts=[1]" training.anchor_lambda=1.0);;
  *) echo "unknown MODE $MODE"; exit 2;;
esac
EPOCHS=${EPOCHS:-300}
NG=$(nvidia-smi -L 2>/dev/null | wc -l)
if (( NG > 1 )); then LAUNCH=("$(dirname "$DP_PY")/torchrun" --standalone --nproc_per_node=$NG); else LAUNCH=("$DP_PY"); fi
"${LAUNCH[@]}" $REPO/rara/workspace.py output_dir=$RUNS/$NAME run_name=$NAME logging.mode=${WANDB_MODE:-online} \
  "task.dataset.zarr_paths=$ZP" "task.dataset.mix_ratios=$MR" training.num_epochs=$EPOCHS training.seed=$SEED \
  "${OV[@]}" ${EXTRA:-}
echo "done: $RUNS/$NAME/checkpoints"
