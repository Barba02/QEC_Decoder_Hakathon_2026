#!/usr/bin/env bash
# Round 2 on one GPU: (A) continue the width-1024 depth-8 models, (B) train depth-16 models from scratch.
# All four processes share the GPU. Logs: logs/*_run2.log and logs/*_d16.log
set -euo pipefail
cd "$(dirname "$0")"
if pgrep -f "train.py --basis" > /dev/null; then
  echo "ERROR: training is already running (see: pgrep -af train.py). Not starting again."; exit 1
fi
for B in X Z; do
  [ -f weights/${B}_w1024_d8.npz ] || { echo "ERROR: weights/${B}_w1024_d8.npz not found"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
mkdir -p logs
for B in X Z; do
  nohup python train.py --basis $B --device cuda --steps 300000 --batch 16384 --width 1024 --depth 8 \
    --workers 6 --eval-every 5000 --compile --init weights/${B}_w1024_d8.npz --seed 1 --lr 3e-4 \
    --out weights/${B}_w1024_d8_run2.npz > logs/${B}_run2.log 2>&1 &
  nohup python train.py --basis $B --device cuda --steps 400000 --batch 16384 --width 1024 --depth 16 \
    --workers 6 --eval-every 5000 --compile --seed 2 \
    --out weights/${B}_w1024_d16.npz > logs/${B}_d16.log 2>&1 &
done
sleep 90
echo "=== started 4 runs; first lines of each log:"
tail -n 3 logs/X_run2.log logs/Z_run2.log logs/X_d16.log logs/Z_d16.log
echo "=== check progress later with: tail -n 2 logs/*_run2.log logs/*_d16.log"
