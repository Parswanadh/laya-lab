#!/usr/bin/env bash
# Wait for the GPU lock, then run the H5 pipeline while holding it.
#
# `AGENTS.md` requires the lock be *held* for the whole GPU job, not merely checked once. This
# script holds fd 9 on .gpu.lock from acquisition until the pipeline exits. Every attempt uses
# `flock -n`, so a refusal is observed and logged rather than silently stalling, and nothing is
# launched while another agent holds the lock.
#
# The same gate is applied to free RAM: a run started below 3 GB available would fight the desktop
# baseline and the OOM killer would decide the outcome (this box has taken 31 OOM kills in a
# session). The memory cap follows the AGENTS.md table for the RAM available at acquisition.
set -u
LAB=/home/parshu/projects/contri/laya-lab
RUN="$LAB/experiments/h5-adapter"
LOCK="$LAB/.gpu.lock"
LOG="$RUN/launch.log"
PY="$LAB/env/venv/bin/python"
MAX_WAIT_MIN=${MAX_WAIT_MIN:-420}
INTERVAL=${INTERVAL:-120}
STAGES=${STAGES:-}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
avail_gb() { free -g | awk '/^Mem:/ {print $7}'; }

log "waiter started pid=$$ max_wait=${MAX_WAIT_MIN}min poll=${INTERVAL}s stages='${STAGES:-all}'"
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
attempt=0
exec 9>"$LOCK"
while true; do
  attempt=$(( attempt + 1 ))
  if flock -n 9; then
    a=$(avail_gb)
    if [ "$a" -lt 3 ]; then
      log "attempt $attempt: lock acquired but only ${a}G RAM available (<3G) -- releasing, waiting"
      flock -u 9
    else
      log "attempt $attempt: LOCK ACQUIRED (${a}G RAM available)"
      break
    fi
  else
    holder=$(fuser "$LOCK" 2>&1 | tr -s ' ' | tail -1)
    log "attempt $attempt: GPU busy, another agent holds the lock -- not launching (holder: ${holder:-unknown})"
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "gave up after ${MAX_WAIT_MIN} min without the lock; nothing was launched"
    exit 3
  fi
  sleep "$INTERVAL"
done

a=$(avail_gb)
if [ "$a" -ge 6 ]; then CAP=6G; else CAP=4G; fi
log "available=${a}G -> MemoryMax=$CAP (AGENTS.md memory table)"

cd "$LAB"
rc=0
if [ -n "$STAGES" ]; then
  systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 --unit=laya-h5-pilot-$$ \
      env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True USE_TF=0 \
      "$PY" experiments/h5-adapter/run_all.py --only "$STAGES" \
      >> "$RUN/run.log" 2>&1 || rc=$?
else
  systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 --unit=laya-h5-adapter-$$ \
      env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True USE_TF=0 \
      "$PY" experiments/h5-adapter/run_all.py --seeds "${SEEDS:-0,1,2}" --epochs "${EPOCHS:-8}" \
      --lr "${LR:-5e-4}" \
      >> "$RUN/run.log" 2>&1 || rc=$?
fi
log "run_all.py exited $rc"
exit "$rc"
