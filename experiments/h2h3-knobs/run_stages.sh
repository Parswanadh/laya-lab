#!/bin/bash
# Run one or more E-002 stages under the single-writer GPU lock.
#
#   experiments/h2h3-knobs/run_stages.sh pilot stage1
#
# Each stage is its own process (fresh model load, ~40 s). The lock is taken once for the whole
# chain with a bounded wait (flock -w): if the timeout expires the chain does not run at all and
# exits 1, so a refused launch is visible rather than silently queued.
set -u
cd "$(dirname "$0")/../.." || exit 1
LOG_DIR=experiments/h2h3-knobs/logs
mkdir -p "$LOG_DIR"
WAIT="${FLOCK_WAIT:-1500}"

CMD=""
for stage in "$@"; do
  CMD="$CMD env/venv/bin/python experiments/h2h3-knobs/run.py --stage $stage >> $LOG_DIR/$stage.log 2>&1; echo \"[$stage] exit=\$?\" >> $LOG_DIR/$stage.log;"
done
CMD="echo \"[stages] start \$(date -Is) args=$*\" >> $LOG_DIR/stages.log; $CMD echo \"[stages] done \$(date -Is)\" >> $LOG_DIR/stages.log"

flock -w "$WAIT" .gpu.lock -c "systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 /bin/bash -c '$CMD'"
rc=$?
if [ $rc -ne 0 ]; then
  echo "[stages] NOT RUN: flock/systemd-run rc=$rc for args=$*" >> "$LOG_DIR/stages.log"
fi
exit $rc
