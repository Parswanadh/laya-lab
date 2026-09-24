#!/usr/bin/env bash
# V-001 GPU reproduction of experiments/orch-baseline and experiments/orch-diagnostic.
#
# Run under the GPU flock:
#   flock -n /home/parshu/projects/contri/laya-lab/.gpu.lock \
#     -c 'bash experiments/V-001/gpu_repro.sh'
#
# The two run.py files write to a hard-coded path (experiments/<run>/results.json), so the
# artifacts under verification are copied to experiments/V-001/stored/ first and restored byte
# for byte afterwards. The scripts themselves are executed unmodified; this file only orchestrates.
set -u
cd /home/parshu/projects/contri/laya-lab

export USE_TF=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=env/venv/bin/python
V=experiments/V-001
mkdir -p "$V/stored" "$V/fresh" "$V/logs"
touch "$V/logs/.lock-acquired"

echo "== free -h at start =="; free -h

# Caches: the only thing these runs write outside their own results.json is interpreter bytecode.
rm -rf fork/laya/__pycache__ fork/laya/*/__pycache__ 2>/dev/null || true

echo "== frozen inputs =="
sha256sum experiments/orch-baseline/run.py experiments/orch-diagnostic/run.py | tee "$V/logs/sha256-scripts.txt"
for f in experiments/orch-baseline/results.json experiments/orch-diagnostic/results.json; do
  cp -n "$f" "$V/stored/$(echo "$f" | tr '/' '_')"
done
sha256sum experiments/orch-baseline/results.json experiments/orch-diagnostic/results.json \
  | tee "$V/logs/sha256-stored.txt"
cp "$V/logs/sha256-stored.txt" "$V/logs/sha256-stored-before.txt"

echo "== 1/3 orch-baseline (fresh process) =="
"$PY" experiments/orch-baseline/run.py > "$V/logs/fresh-orch-baseline.log" 2>&1
echo "baseline exit=$?"
cp experiments/orch-baseline/results.json "$V/fresh/orch-baseline-results.json"
cp "$V/stored/experiments_orch-baseline_results.json" experiments/orch-baseline/results.json

echo "== 2/3 orch-diagnostic (fresh process) =="
"$PY" experiments/orch-diagnostic/run.py > "$V/logs/fresh-orch-diagnostic.log" 2>&1
echo "diagnostic exit=$?"
cp experiments/orch-diagnostic/results.json "$V/fresh/orch-diagnostic-results.json"
cp "$V/stored/experiments_orch-diagnostic_results.json" experiments/orch-diagnostic/results.json

echo "== 3/3 instrumented repeat + per-item (fresh process) =="
"$PY" experiments/V-001/repeat.py > "$V/logs/repeat.log" 2>&1
echo "repeat exit=$?"

echo "== restored artifacts =="
sha256sum experiments/orch-baseline/results.json experiments/orch-diagnostic/results.json \
  | tee "$V/logs/sha256-restored.txt"
echo "== done =="
