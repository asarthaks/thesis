#!/usr/bin/env bash
#
# reset_incomplete.sh STATUSDIR OUTDIR
#
# Clears the .lock and .failed markers for any job that has no result JSON, so the
# next run_queue.sh pass picks them up again. Use after a crash, an OOM, or a manual
# kill. Jobs that actually finished (have a JSON) are left alone.

set -euo pipefail
STATUS="${1:-queue_status}"
OUTDIR="${2:-results_rerun}"

n=0
shopt -s nullglob
for lock in "$STATUS"/*.lock; do
  job="$(basename "$lock" .lock)"
  if [[ ! -f "$OUTDIR/$job.json" ]]; then
    rm -rf "$STATUS/$job.lock" "$STATUS/$job.failed"
    echo "requeue: $job"
    n=$((n+1))
  fi
done
echo "requeued $n job(s)"
