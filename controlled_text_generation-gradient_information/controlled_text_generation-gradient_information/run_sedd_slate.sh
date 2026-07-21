#!/usr/bin/env bash
# run_sedd_slate.sh - Phase 4 capability slate, sharded across the free A6000s.
#
# Jobs: recovery {small,medium}, hybrid {small,medium} (SEDD arm), hybrid_refs
# (scale-independent reference arms: left_conditional, dls_policy, dls_random).
# Each shard writes its own atomic JSON (resume contract: run_sedd_cap.py skips a
# shard whose JSON exists), so re-running resumes. After all shards finish, merge
# jobs assemble final JSON+CSV with bootstrap CIs over the merged rows.
#
# Usage: ./run_sedd_slate.sh [num_shards]
set -u
cd "$(dirname "$0")"

export HF_HOME=/mount/studenten-temp1/users/singhsk/thesis/thesis/hf/cache
export SEDD_REPO=/mount/studenten-temp1/users/singhsk/thesis/thesis/Score-Entropy-Discrete-Diffusion
GPT2SFT=/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output
OUT=results_revision
LOGDIR=unifiedruns/sedd_slate
mkdir -p "$LOGDIR"

GPUS=(0 1 2 3 4 5 6 7 8)
NG=${#GPUS[@]}
NSHARDS=${1:-6}
gi=0

launch () {  # run_name  full-command...
  local rn="$1"; shift
  if [ -f "$OUT/$rn.json" ]; then echo "[skip] $rn (json exists)"; return; fi
  local gpu=${GPUS[$((gi % NG))]}; gi=$((gi+1))
  echo "[launch] $rn on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$@" > "$LOGDIR/$rn.log" 2>&1 &
}

cap () { python diagnostics/run_sedd_cap.py --out_dir "$OUT" --gpt2sft_path "$GPT2SFT" "$@"; }

for ((s=0; s<NSHARDS; s++)); do
  # recovery (native SEDD), both scales
  for scale in small medium; do
    rn="rev_sedd_recovery_${scale}.shard${s}of${NSHARDS}"
    launch "$rn" python diagnostics/run_sedd_cap.py --out_dir "$OUT" --gpt2sft_path "$GPT2SFT" \
      --exp recovery --scale "$scale" --run_name "$rn" --shard_idx "$s" --num_shards "$NSHARDS"
    # hybrid SEDD arm, both scales
    rn="rev_sedd_hybrid_${scale}.shard${s}of${NSHARDS}"
    launch "$rn" python diagnostics/run_sedd_cap.py --out_dir "$OUT" --gpt2sft_path "$GPT2SFT" \
      --exp hybrid --scale "$scale" --run_name "$rn" --shard_idx "$s" --num_shards "$NSHARDS"
  done
  # hybrid reference arms (scale-independent, gpt2sft only) - run once per shard
  rn="rev_sedd_hybridrefs.shard${s}of${NSHARDS}"
  launch "$rn" python diagnostics/run_sedd_cap.py --out_dir "$OUT" --gpt2sft_path "$GPT2SFT" \
    --exp hybrid_refs --scale small --run_name "$rn" --shard_idx "$s" --num_shards "$NSHARDS"
done

echo "[wait] all shard jobs..."; wait
echo "[shards done] merging..."

for scale in small medium; do
  python diagnostics/merge_sedd_cap.py --out_dir "$OUT" \
    --shard_glob "rev_sedd_recovery_${scale}.shard*of${NSHARDS}" \
    --final_run_name "rev_sedd_recovery_${scale}" --experiment recovery --scale "$scale"
done
# hybrid final: SEDD arms (small,medium) + reference arms, grouped by the arm column
python diagnostics/merge_sedd_cap.py --out_dir "$OUT" \
  --shard_glob "rev_sedd_hybrid*.shard*of${NSHARDS}" \
  --final_run_name "rev_sedd_hybrid" --experiment hybrid --scale both
echo "[slate done]"
