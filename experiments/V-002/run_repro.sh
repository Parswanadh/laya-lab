#!/usr/bin/env bash
# V-002: reproduce one full eval arm end to end from a PINNED revision of the committed code.
#
# Why pinned: the live working tree has uncommitted edits from the engineer's init-fair redesign
# (arms.py/eval.py/train.py). Verifying against a moving tree proves nothing, so this runs from a
# detached git worktree at 54cff08 -- the commit whose tree contains the prediction files under
# verification (blobs verified equal to the working-tree copies).
#
# The worktree symlinks models/, env/, worktrees/ and experiments/h5-adapter/cache back to the main
# checkout (gitignored, large), and runs/<arm>/seed0/head.pt is a file-level symlink to the head
# checkpoint the engineer's stored predictions were produced from. Nothing here writes into another
# agent's artifacts: outputs go to experiments/V-002/.
#
# Usage: flock -n <lab>/.gpu.lock -c 'bash experiments/V-002/run_repro.sh' || echo REFUSED
set -euo pipefail

LAB=/home/parshu/projects/contri/laya-lab
REV="$LAB/.scratch/V-002/rev54cff08"
PY="$LAB/env/venv/bin/python"
OUT="$LAB/experiments/V-002"

cd "$REV"
echo "== pinned revision =="
git rev-parse HEAD
echo "== fork worktree (dirty, as the runs were made) =="
git -C "$LAB/worktrees/h5" rev-parse HEAD
sha256sum "$LAB/worktrees/h5/laya/common.py"

echo "== arm1_frozen (shipped model, live encoder path, no cache) =="
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G \
  "$PY" experiments/h5-adapter/eval.py --arm arm1_frozen --out "$OUT/repro_arm1_frozen.jsonl"

echo "== arm2_shipped_init seed0 (committed head.pt over the committed feature cache) =="
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G \
  "$PY" experiments/h5-adapter/eval.py --arm arm2_shipped_init --seed 0 \
        --out "$OUT/repro_arm2_shipped_init-seed0.jsonl"

echo "== fork worktree hash after (must be unchanged) =="
sha256sum "$LAB/worktrees/h5/laya/common.py"
echo "DONE"
