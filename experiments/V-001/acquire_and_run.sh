#!/usr/bin/env bash
# Retry-with-backoff launcher for the V-001 reproduction. Never blocks on the lock: each attempt
# uses `flock -n` and yields immediately, so it cannot stall behind another agent's GPU job.
# The inner script touches logs/.lock-acquired, which distinguishes "lock refused" from
# "ran and failed".
set -u
cd /home/parshu/projects/contri/laya-lab
V=experiments/V-001
MARK="$V/logs/.lock-acquired"
ATTEMPTS=${ATTEMPTS:-40}
SLEEP=${SLEEP:-60}

rm -f "$MARK"
for i in $(seq 1 "$ATTEMPTS"); do
  echo "== attempt $i at $(date +%T) =="
  flock -n .gpu.lock -c 'bash experiments/V-001/gpu_repro.sh'
  rc=$?
  if [ -f "$MARK" ]; then
    echo "REPRO_RC=$rc (ran under the lock on attempt $i)"
    exit $rc
  fi
  echo "attempt $i refused at $(date +%T) — another agent holds the GPU; waiting ${SLEEP}s"
  sleep "$SLEEP"
done
echo "gave up after $ATTEMPTS attempts"
exit 2
