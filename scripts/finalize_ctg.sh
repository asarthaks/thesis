#!/usr/bin/env bash
#
# finalize_ctg.sh OUT_DIR JUDGE_DIR SUMMARY_JSON [GPU]
#
# The scoring half of the generate/score split, run as one command once a family of
# chains has finished. Order matters and is enforced here:
#
#   1. merge the shards (rates recomputed from totals, never averaged over shards);
#   2. score the merged text with the HELD-OUT sentiment judge, which also computes the
#      diversity metrics, since those are properties of the accumulated text;
#   3. score it with the external fluency judge (Llama-3), loaded only after the
#      sentiment judge has been released so the two never sit in VRAM together;
#   4. re-merge to join the judge outputs into the summary index.
#
# No generator model is loaded by this script. That is the separation the programme
# depends on: the classifier that steers is never the classifier that scores.

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

OUT_DIR="${1:-results/ctg/phase2}"
JUDGE_DIR="${2:-results/ctg/judged}"
SUMMARY="${3:-results/ctg/phase2_summary.json}"
GPU="${4:-0}"
PY=gfn-lm-tuning/gfn/bin/python
LLAMA="${LLAMA3_PATH:-/mount/arbeitsdaten/studenten1/singhsk/models/llama3-8b}"

mkdir -p "$JUDGE_DIR"

echo "== 1. merge shards in $OUT_DIR"
$PY revision/analyze_ctg.py --out_dir "$OUT_DIR" --judge_dir "$JUDGE_DIR" \
    --out "$SUMMARY" | tail -40

echo
echo "== 2. held-out SENTIMENT judge (adherence + diversity)"
# only the merged CSVs, not the per-shard ones: diversity is a property of the whole
# family and a shard holds too few sequences per prompt to estimate it.
CUDA_VISIBLE_DEVICES="$GPU" $PY diagnostics/run_ctg_judge.py \
    --glob "$OUT_DIR/*.csv" --exclude_shards --out_dir "$JUDGE_DIR" --stage sentiment 2>&1 \
    | grep -v Warning | tail -40

echo
echo "== 3. external FLUENCY judge ($LLAMA)"
if [ -d "$LLAMA" ]; then
  CUDA_VISIBLE_DEVICES="$GPU" $PY diagnostics/run_ctg_judge.py \
      --glob "$OUT_DIR/*.csv" --exclude_shards --out_dir "$JUDGE_DIR" --stage fluency \
      --judge_path "$LLAMA" 2>&1 | grep -v Warning | tail -40
else
  echo "SKIPPED: no judge model at $LLAMA"
fi

echo
echo "== 4. re-merge with judge columns joined"
$PY revision/analyze_ctg.py --out_dir "$OUT_DIR" --judge_dir "$JUDGE_DIR" \
    --out "$SUMMARY" | tail -40
echo
echo "wrote $SUMMARY"
