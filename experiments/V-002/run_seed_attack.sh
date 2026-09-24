#!/usr/bin/env bash
# V-002 seed attack: arm2 (fine-tune the shipped head) exists for seed 0 only. Train and evaluate
# it at seeds 1 and 2 with the recipe the seed-0 run recorded, from the same pinned revision.
#
# Recipe copied from the raw artifact runs/arm2_shipped_init/seed0/training.json:
#   epochs=12  lr=5e-4  weight_decay=0.01  token_budget=12288  max_batch=8  train_mode=uniform
# train.py's own defaults for everything not named here are used (the seed-0 run used them too).
#
# Outputs land in experiments/V-002/ (this agent's directory); checkpoints land in the pinned
# scratch worktree's own gitignored runs/ -- nothing is written into the engineer's artifact dir.
#
# Usage: flock -n <lab>/.gpu.lock -c 'bash experiments/V-002/run_seed_attack.sh' || echo REFUSED
set -euo pipefail

LAB=/home/parshu/projects/contri/laya-lab
REV="$LAB/.scratch/V-002/rev54cff08"
PY="$LAB/env/venv/bin/python"
OUT="$LAB/experiments/V-002"

cd "$REV"
for SEED in 1 2; do
  echo "== train arm2_shipped_init seed=$SEED (epochs=12 lr=5e-4) =="
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G \
    "$PY" experiments/h5-adapter/train.py --arm arm2_shipped_init --seed "$SEED" \
          --epochs 12 --lr 5e-4 --token-budget 12288 --max-batch 8
  echo "== eval arm2_shipped_init seed=$SEED =="
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G \
    "$PY" experiments/h5-adapter/eval.py --arm arm2_shipped_init --seed "$SEED" \
          --out "$OUT/repro_arm2_shipped_init-seed$SEED.jsonl"
done
echo "DONE"
