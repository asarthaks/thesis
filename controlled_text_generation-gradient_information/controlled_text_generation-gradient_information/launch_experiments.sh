#!/usr/bin/env bash
#
# launch_experiments.sh
#
# Fans the thesis experiment matrix (from gen_jobs.py) out across your GPUs using
# tmux. One tmux window per GPU; each window runs its assigned jobs sequentially,
# so parallelism equals the number of GPUs. Sequential-per-GPU is the safe default
# for Llama-8B (one fits per card); if GPT-2 Large leaves headroom you can raise
# JOBS_PER_GPU or just list a GPU twice in GPUS.
#
# Usage:
#   ./launch_experiments.sh                 # all models, all tiers
#   ./launch_experiments.sh --tier1         # only the runs the MH bug invalidated
#   ./launch_experiments.sh --models gpt2-large
#   ./launch_experiments.sh --dry-run       # print the plan, launch nothing
#
# Attach with:   tmux attach -t $SESSION
# Watch wandb:   https://wandb.ai/<you>/ctg-langevin-thesis

set -euo pipefail

# ---- config ---------------------------------------------------------------
SESSION="ctg_rerun"
GPUS=(0 1 2 3 4 5 6 7)                 # edit to your available GPUs
CONDA_ENV="gfn"                # conda env to activate in each window; "" to skip
LOG_DIR="rerun_logs"
GEN_ARGS=()                    # forwarded to gen_jobs.py (e.g. --tier1, --models ...)
DRY_RUN=0

# ---- parse a couple of pass-through flags ---------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) GEN_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$LOG_DIR"

# ---- build the job list ---------------------------------------------------
mapfile -t JOBS < <(python gen_jobs.py "${GEN_ARGS[@]}")
NJOBS=${#JOBS[@]}
NGPU=${#GPUS[@]}
if [[ "$NJOBS" -eq 0 ]]; then echo "no jobs generated"; exit 1; fi
echo "Generated $NJOBS jobs across $NGPU GPUs (${GPUS[*]})"

# ---- assign jobs to GPUs round-robin --------------------------------------
declare -A BUCKET
for i in "${!JOBS[@]}"; do
  g=$(( i % NGPU ))
  BUCKET[$g]+="${JOBS[$i]}"$'\n'
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  for g in "${!GPUS[@]}"; do
    echo "=== GPU ${GPUS[$g]} ==="
    printf '%s' "${BUCKET[$g]:-}" | nl -ba
  done
  exit 0
fi

# ---- spin up tmux ---------------------------------------------------------
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "overview"
tmux send-keys -t "$SESSION:overview" "watch -n 30 nvidia-smi" C-m

for g in "${!GPUS[@]}"; do
  gpu=${GPUS[$g]}
  win="gpu${gpu}"
  tmux new-window -t "$SESSION" -n "$win"

  # activate env
  if [[ -n "$CONDA_ENV" ]]; then
    tmux send-keys -t "$SESSION:$win" "conda activate $CONDA_ENV" C-m
  fi
  tmux send-keys -t "$SESSION:$win" "export CUDA_VISIBLE_DEVICES=$gpu" C-m

  # queue each job for this GPU, tee-ing output to a per-run log
  idx=0
  while IFS= read -r job; do
    [[ -z "$job" ]] && continue
    # derive a log filename from the --wandb_group and --method for readability
    tag=$(echo "$job" | grep -oP '(?<=--model_tag )\S+')
    grp=$(echo "$job" | grep -oP '(?<=--wandb_group )\S+')
    meth=$(echo "$job" | grep -oP '(?<=--method )\S+')
    mh=$(echo "$job" | grep -q -- '--mh' && echo mh || echo nomh)
    log="$LOG_DIR/${grp}.${meth}.${mh}.log"
    full="$job 2>&1 | tee $log"
    tmux send-keys -t "$SESSION:$win" "echo '>>> START $grp $meth $mh'; $full; echo '<<< DONE $grp $meth $mh'" C-m
    idx=$((idx+1))
  done <<< "${BUCKET[$g]:-}"

  tmux send-keys -t "$SESSION:$win" "echo 'ALL JOBS ON GPU $gpu FINISHED'" C-m
done

echo "launched. attach with:  tmux attach -t $SESSION"
echo "logs in:                $LOG_DIR/"
