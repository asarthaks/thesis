#!/usr/bin/env bash
# run_gprime_slate.sh - Phase 5 Stage 1a: on-domain trust-region guided SEDD.
# Sharded over held-out SST-2 prompts across the free A6000s (one shard per GPU),
# JSON-existence resume (run_gprime.py skips a shard whose JSON exists), then aggregate.
#
# Usage: ./run_gprime_slate.sh [num_shards]
set -u
cd "$(dirname "$0")"

export HF_HOME=/mount/studenten-temp1/users/singhsk/thesis/thesis/hf/cache
export SEDD_REPO=/mount/studenten-temp1/users/singhsk/thesis/thesis/Score-Entropy-Discrete-Diffusion
GPT2SFT=/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output
HEAD=/mount/arbeitsdaten/studenten1/singhsk/models/sentiment_constrained_ft_gpt2_large/sentiment_head.pt
OUT=results_revision
LOGDIR=unifiedruns/gprime_slate
mkdir -p "$LOGDIR"

N_PROMPTS=300
SCALE=medium
GAMMAS=2,4
DELTA=5.0
SPAN=20
STEPS=64

GPUS=(0 1 2 3 4 5 6 7 8)
NG=${#GPUS[@]}
NSHARDS=${1:-9}

for ((s=0; s<NSHARDS; s++)); do
  rn="rev_gprime_${SCALE}.shard${s}of${NSHARDS}"
  if [ -f "$OUT/$rn.json" ]; then echo "[skip] $rn (json exists)"; continue; fi
  gpu=${GPUS[$((s % NG))]}
  echo "[launch] $rn on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup python diagnostics/run_gprime.py \
    --run_name "$rn" --out_dir "$OUT" --gpt2sft_path "$GPT2SFT" --head "$HEAD" \
    --scale "$SCALE" --n_prompts "$N_PROMPTS" --num_shards "$NSHARDS" --shard_idx "$s" \
    --gammas "$GAMMAS" --delta "$DELTA" --span_len "$SPAN" --steps "$STEPS" \
    > "$LOGDIR/$rn.log" 2>&1 &
done

echo "[wait] all gprime shard jobs..."; wait
echo "[shards done] aggregating..."
python diagnostics/aggregate_gprime.py --out_dir "$OUT" \
  --shard_glob "rev_gprime_${SCALE}.shard*of${NSHARDS}" --final_run_name "rev_gprime"
echo "[gprime slate done]"
