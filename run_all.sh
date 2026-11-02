#!/bin/bash
# Whole grid, strictly sequential, low priority, resumable (skips finished checkpoints / eval files).
# Seed-major so that a time cut drops seeds, not methods.
# Start:  setsid nohup ./run_all.sh > logs/run_all.log 2>&1 < /dev/null &      Resume: run the same line again.
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs ckpt results/eval
for seed in 0 1 2; do
  for algo in dp bcc bc; do
    [ -f ckpt/${algo}_s${seed}.pt ] && continue
    while [ "$(awk '{print int($1)}' /proc/loadavg)" -gt 30 ]; do sleep 60; done
    nice -n 19 ionice -c3 $PY train.py --algo $algo --seed $seed --steps 40000 || exit 1
  done
  nice -n 19 ionice -c3 $PY eval.py dp_s${seed} bcc_s${seed} bc_s${seed} || exit 1
done
