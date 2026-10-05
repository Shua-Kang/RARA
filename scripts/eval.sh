#!/bin/bash
# Closed-loop evaluation of one checkpoint on the pill task: 25 episodes from one block of environment seeds.
#   CKPT=<checkpoint.pt> [SPLIT=id|ood] [SEEDSTART=100000] [N=25] [TAG=<name>] bash scripts/eval.sh
# Output: $EVAL/<TAG>/summary.json (per-seed success) and rollout videos in $EVAL/<TAG>/video/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${CKPT:?CKPT=<checkpoint.pt>}"; SPLIT=${SPLIT:-id}; SEEDSTART=${SEEDSTART:-100000}; N=${N:-25}
TAG=${TAG:-$(basename $(dirname $(dirname $CKPT)))_$(basename $CKPT .pt)_${SPLIT}_${SEEDSTART}}
PORT=${PORT:-$((15000 + RANDOM % 2000))}
$DP_PY $REPO/eval/server.py --ckpt "$CKPT" --port $PORT & SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 60); do
  $DP_PY -c "import socket,sys;s=socket.socket();sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" && break
  kill -0 $SRV 2>/dev/null || { echo "policy server died"; exit 3; }; sleep 5
done
mkdir -p $EVAL/$TAG; cd $RT
PYTHONPATH=$RT $RT_PY $REPO/eval/robotwin_client.py --task move_pillbottle_pad --variant joint --history --replan 6 \
  --config eval_$SPLIT --n $N --port $PORT --cameras head,right_wrist --seed-start $SEEDSTART \
  --out $EVAL/$TAG --video-dir $EVAL/$TAG/video
