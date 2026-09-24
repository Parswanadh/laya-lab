#!/usr/bin/env bash
# Take the GPU lock, then run one stage while holding it. -- the H5 arm-3r driver's launcher.
#
#   stage_arm3r.sh <stage-name> <max_wait_min> <command...>
#
# `AGENTS.md` requires the lock to be *held* for the whole GPU job, not merely checked once, so this
# script keeps fd 9 on `.gpu.lock` from acquisition until the command exits. Every attempt uses
# `flock -n`, so a refusal is observed and logged rather than silently stalling, and nothing is
# launched while another agent holds the lock. The free-RAM gate and the memory cap are the
# AGENTS.md table: a run started below 3 GB available would fight the desktop baseline and let the
# OOM killer decide the outcome.
#
# Logs are written inside the repository (`_<stage>.out`), never /tmp. The log records whether the
# lock was held and how much RAM was available, which is what the finding cites for "locked run".
set -u
LAB=/home/parshu/projects/contri/laya-lab
RUN="$LAB/experiments/h5-adapter"
LOCK="$LAB/.gpu.lock"
NAME="${1:?stage name required}"
WAIT_MIN="${2:?max wait minutes required}"
shift 2
LOG="$RUN/_${NAME}.out"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
avail_gb() { free -g | awk '/^Mem:/ {print $7}'; }

log "stage=$NAME waiting for the GPU lock (max ${WAIT_MIN} min): $*"
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
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
    log "gave up after ${WAIT_MIN} min without the lock; nothing was launched"
    exit 3
  fi
  sleep 60
done

a=$(avail_gb)
if [ "$a" -ge 6 ]; then CAP=6G; else CAP=4G; fi
log "available=${a}G -> MemoryMax=$CAP (AGENTS.md memory table)"

cd "$LAB"
rc=0
systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 --unit="laya-h5-${NAME}-$$" \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True USE_TF=0 \
  "$@" >> "$LOG" 2>&1 || rc=$?
log "stage=$NAME exited $rc (lock held for the whole run: yes)"
exit "$rc"
