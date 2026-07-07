#!/usr/bin/env bash
#
# run_queue.sh --manifest M --gpus "0 1 2" --vram 24 --out_dir D --status S --env gfn
#
# Launches one worker per GPU in a tmux session, all pulling from the same manifest.
# Free GPUs grab the next fitting job, so long Llama jobs and short GPT-2 jobs balance
# themselves. Re-run the exact same command to resume: finished jobs are skipped.
#
# Attach:   tmux attach -t $SESSION   (window "monitor" shows progress + nvidia-smi)

set -uo pipefail
# no 'errexit' here on purpose: a launcher should not die silently if a pre-flight
# stat (like counting result files in an empty dir) returns non-zero.

SESSION="ctg_queue"
GPUS=""
VRAM=24
MANIFEST="manifest.tsv"
OUTDIR="results_rerun"
STATUS="queue_status"
ENVN="gfn"
PER_GPU=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --gpus)     GPUS="$2"; shift 2 ;;
    --vram)     VRAM="$2"; shift 2 ;;
    --out_dir)  OUTDIR="$2"; shift 2 ;;
    --status)   STATUS="$2"; shift 2 ;;
    --env)      ENVN="$2"; shift 2 ;;
    --session)  SESSION="$2"; shift 2 ;;
    --per_gpu)  PER_GPU="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# default to every visible GPU
if [[ -z "$GPUS" ]]; then
  GPUS=$(nvidia-smi -L | sed -n 's/^GPU \([0-9]*\).*/\1/p' | tr '\n' ' ')
fi

if [[ ! -f "$MANIFEST" ]]; then echo "manifest not found: $MANIFEST"; exit 1; fi
mkdir -p "$OUTDIR" "$STATUS"

NJOBS=$(grep -c . "$MANIFEST" 2>/dev/null || echo 0)
NDONE=$(find "$OUTDIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "manifest=$MANIFEST  jobs=$NJOBS  already_done=$NDONE"
echo "gpus=[$GPUS]  per_gpu=$PER_GPU  vram=${VRAM}GB  out=$OUTDIR  status=$STATUS"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is not installed or not on PATH. Install it or load the module."
  exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n monitor
tmux send-keys -t "$SESSION:monitor" \
  "watch -n 20 'echo done: \$(find $OUTDIR -maxdepth 1 -name \"*.json\" | wc -l)/$NJOBS; echo failed: \$(find $STATUS -maxdepth 1 -name \"*.failed\" | wc -l); nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'" C-m

for g in $GPUS; do
  for i in $(seq 1 "$PER_GPU"); do
    win="gpu${g}w${i}"
    tmux new-window -t "$SESSION" -n "$win"
    if [[ -n "$ENVN" ]]; then
      tmux send-keys -t "$SESSION:$win" "conda activate $ENVN 2>/dev/null || source activate $ENVN 2>/dev/null || true" C-m
    fi
    tmux send-keys -t "$SESSION:$win" "bash worker.sh $g $VRAM $MANIFEST $OUTDIR $STATUS" C-m
  done
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  NWIN=$(tmux list-windows -t "$SESSION" | wc -l | tr -d ' ')
  echo "launched session '$SESSION' with $NWIN windows (1 monitor + workers)."
  echo "attach with:  tmux attach -t $SESSION"
  echo "if a worker window shows nothing, run 'bash worker.sh <gpu> $VRAM $MANIFEST $OUTDIR $STATUS' in it by hand to see the error."
else
  echo "ERROR: tmux session '$SESSION' was not created. Try 'tmux new -s test' to check tmux works here."
  exit 1
fi
