#!/usr/bin/env bash
#
# run_constraint_ablation.sh MODEL_PATH HEAD_PT [GPU]
#
# The constraint ablation, across TWO tasks x TWO setups x FIVE arms.
#
#   task  : continuation (MuCoLa's actual task: prompt + 20 free tokens)
#           infill       (ours: corrupt N tokens in a real sentence, recover them)
#   setup : mucola (300 steps, noise 5.0->0.05, no MH, no gradnorm, CLS + CENTROID init
#                   -- CLS because they optimize a CONTINUOUS simplex over the vocab,
#                      and centroid because their "zeros" init IS the embedding-table
#                      centroid, a point belonging to no particular token)
#           ours   (50 steps, eps 10.5->0.1, MH on, gradnorm on, DLS + random-token init)
#   arm   : lm_only | full | cons_only | cons_random | random
#
# THE TEST, within every cell:      full   vs   cons_random
# Same gradient magnitude, real direction vs random direction. If `full` steers
# sentiment and `cons_random` does not, the CONSTRAINT gradient's DIRECTION carries
# real information, exactly where the LM gradient's direction did not.

set -uo pipefail

MODEL="${1:?usage: run_constraint_ablation.sh MODEL_PATH HEAD_PT [GPU]}"
HEAD="${2:?need sentiment head .pt from train_sentiment_head.py}"
GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

OUT=results_constrained
mkdir -p "$OUT" logs_constrained

for TASK in continuation infill; do
  for SETUP in mucola ours; do
    for MODE in lm_only full cons_only cons_random random; do
      TAG="$TASK.$SETUP.$MODE"
      echo "=================================================================="
      echo ">>> $TAG   $(date +%H:%M:%S)"
      echo "=================================================================="
      # sampler + init are chosen by --setup (cls/centroid for mucola, dls/random for ours)
      python run_constrained.py \
        --model_path "$MODEL" --head "$HEAD" \
        --task "$TASK" --setup "$SETUP" --constraint_mode "$MODE" \
        --target_label 1 \
        --samples_per_prompt 10 --num_masks 8 --n_samples 100 \
        --out_dir "$OUT" 2>&1 | tee "logs_constrained/$TAG.log"
    done
  done
done

echo ""
echo "=================================================================="
echo "ATTRIBUTION TABLE"
echo "=================================================================="
python - <<'PY'
import json, glob
rows = {}
for f in glob.glob('results_constrained/*.json'):
    j = json.load(open(f)); c = j['config']
    rows[(c['task'], c['setup'], c['constraint_mode'])] = (
        j['steering_gain'], j.get('final_kl'), c.get('sampler'), c.get('init'))
arms = ['lm_only', 'full', 'cons_only', 'cons_random', 'random']
for task in ['continuation', 'infill']:
    for setup in ['mucola', 'ours']:
        meta = next((v for k, v in rows.items() if k[0] == task and k[1] == setup), None)
        smp = f"{meta[2]}/{meta[3]}" if meta else "?"
        print(f"\n--- task={task}  setup={setup}  ({smp})  steering gain, pts ---")
        for a in arms:
            v = rows.get((task, setup, a))
            if v is None:
                print(f"  {a:<13}   (missing)")
            else:
                g, k = v[0], v[1]
                kk = f"{k:.2f}" if k is not None else "n/a"
                print(f"  {a:<13} {g:+6.1f}    final_kl={kk}")
        f_ = rows.get((task, setup, 'full'))
        cr = rows.get((task, setup, 'cons_random'))
        if f_ and cr:
            print(f"  >>> full - cons_random = {f_[0] - cr[0]:+.1f} pts")
print("""
READING IT
  full - cons_random  LARGE POSITIVE
     -> the CONSTRAINT gradient's DIRECTION carries real signal. That is exactly
        what the LM gradient failed to provide. It explains why MuCoLa/COLD work:
        the CLASSIFIER steers them, not the LM likelihood gradient.
  full - cons_random  NEAR ZERO
     -> even the classifier gradient does not steer via direction; the method is
        driven by noise + projection.
  lm_only ~ 0 in every cell
     -> our existing null result, reproduced on a sentiment task.

  Also compare setup=mucola (CLS/centroid) against setup=ours (DLS/random-token)
  WITHIN a task: that isolates the parameterization (continuous simplex-like vs
  discrete) from the task itself.
""")
PY
