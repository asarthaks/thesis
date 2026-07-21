#!/usr/bin/env bash
#
# worker.sh GPU VRAM MANIFEST OUTDIR STATUSDIR
#
# One worker pinned to one GPU. It scans the manifest top to bottom and claims the
# first job that (a) fits this GPU's VRAM, (b) is not already done (its result JSON
# exists), and (c) is not already claimed by another worker. Claiming is atomic via
# mkdir, so many workers can share one manifest without stepping on each other.
#
# Resume is automatic: finished jobs have a JSON and are skipped. A job that fails
# leaves a .failed marker and keeps its .lock so it is not retried in a loop; use
# reset_incomplete.sh to requeue failures.

set -uo pipefail

GPU="$1"; VRAM="$2"; MANIFEST="$3"; OUTDIR="$4"; STATUS="$5"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUTDIR" "$STATUS"

echo "[gpu $GPU] worker up, vram=${VRAM}GB, manifest=$MANIFEST"

while true; do
  claimed=0
  while IFS=$'\t' read -r job vram cmd; do
    [ -z "${job:-}" ] && continue
    # VRAM gate
    if [ "$vram" -gt "$VRAM" ]; then continue; fi
    # already finished?
    if [ -f "$OUTDIR/$job.json" ]; then continue; fi
    # already claimed / failed?
    if [ -e "$STATUS/$job.lock" ]; then continue; fi
    # try to claim atomically
    if mkdir "$STATUS/$job.lock" 2>/dev/null; then
      claimed=1
      echo "[gpu $GPU] START $job  $(date +%H:%M:%S)"
      if eval "$cmd" > "$STATUS/$job.log" 2>&1; then
        touch "$STATUS/$job.done"
        echo "[gpu $GPU] DONE  $job  $(date +%H:%M:%S)"
      else
        touch "$STATUS/$job.failed"
        echo "[gpu $GPU] FAIL  $job  (tail below, full log: $STATUS/$job.log)"
        tail -n 5 "$STATUS/$job.log" | sed 's/^/[gpu '"$GPU"']   /'
      fi
      break   # rescan from the top so priorities stay fresh
    fi
  done < "$MANIFEST"

  if [ "$claimed" -eq 0 ]; then
    echo "[gpu $GPU] nothing left to claim; exiting"
    break
  fi
done
