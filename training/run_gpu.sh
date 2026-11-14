#!/usr/bin/env bash
# Train X and Z models in parallel on one GPU. Usage: bash run_gpu.sh [width] [depth] [steps]
# Logs go to logs/, weights to weights/{X,Z}_w<width>_d<depth>.npz
set -euo pipefail
cd "$(dirname "$0")"
W=${1:-1024}; D=${2:-8}; S=${3:-100000}
CORES=$(nproc); WORKERS=$(( (CORES - 4) / 2 )); [ "$WORKERS" -lt 2 ] && WORKERS=2
mkdir -p logs
for B in X Z; do
  nohup python train.py --basis $B --device cuda --steps $S --batch 16384 --width $W --depth $D \
    --workers $WORKERS --eval-every 2500 --compile > logs/${B}_w${W}_d${D}.log 2>&1 &
done
echo "started; follow with: tail -f logs/*_w${W}_d${D}.log"
