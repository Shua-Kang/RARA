#!/bin/bash
# Standard protocol for one run: last three checkpoints x two seed blocks x 25 episodes = 150 rollouts per split,
# then print the success rate.   bash scripts/evaluate.sh <run dir> [id|ood]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN=$1; SPLIT=${2:-id}
for CKPT in $(ls $RUN/checkpoints/epoch*.pt | sort | tail -n 3); do
  for SEEDSTART in 100000 200000; do CKPT=$CKPT SPLIT=$SPLIT SEEDSTART=$SEEDSTART bash $REPO/scripts/eval.sh; done
done
$DP_PY $REPO/scripts/summarize.py $EVAL "$(basename $RUN)_epoch*_${SPLIT}_*"
