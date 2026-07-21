# Revision experiments and analyses

This adds the code for the examiner revision plan on top of the existing harness.
Everything follows the same contract as `run_experiment.py` and
`run_diagnostic.py`: a job is a `run_name`, it is DONE when
`<out_dir>/<run_name>.json` exists, and JSON is written atomically at the end, so
`run_queue.sh` picks up all of it unchanged and resumes on restart.

## New files

Runners (GPU):
- `diagnostics/run_revision.py` with `--exp kl_baselines | model_divergence | continuation`
- `diagnostics/run_external_judge.py` with `--stage generate | judge`
- `diagnostics/run_sedd_linearization.py` (the SEDD positive control, with `--dry_run`)

Analyses (no GPU, run inside a worker window on CPU):
- `revision/analyze_stats.py` (has a `--selftest`)
- `revision/analyze_likelihood_trap.py`
- `revision/reconcile_numbers.py`
- `revision/analyze_constrained.py`

Manifest generator:
- `gen_manifest_revision.py`

## What maps to what

| Concern | Job / run_name | Code | Needs GPU |
|--------|-----------------|------|-----------|
| 1 statistics behind the null | `rev_stats_<dir>` | analyze_stats.py | no |
| 2 missing KL baselines | `rev_klbase_*` | run_revision kl_baselines | yes, cheap |
| 3 circular evaluation (rescoring) | `rev_judge_gen_*`, `rev_judge_score_*` | run_external_judge.py | yes |
| 4 non-gradient sampler on same energy | comes free from concern 2 (the `gibbs` baseline) | run_revision kl_baselines | yes, cheap |
| 5 did tuning change the energy | `rev_divergence_*` | run_revision model_divergence | yes, cheap |
| 6 numerical inconsistencies | `rev_reconcile` | reconcile_numbers.py | no |
| 8 SEDD positive control | `rev_sedd_*` | run_sedd_linearization.py | yes |
| 9 task generality (continuation) | `rev_continuation_*` | run_revision continuation | yes, cheap |
| 10 likelihood trap confound + Llama | `rev_ltrap_within` + `diag_likelihood_trap_llama3-8b` | analyze_likelihood_trap.py + run_diagnostic | mixed |
| 11 constraint CIs + anomaly | `rev_constrained_ci` | analyze_constrained.py | no |
| 12 oracle fairness | `gpt2-large.dls.*.oracle.s50` | run_experiment (per method) | yes, cheap |
| 15 config count | inside `rev_reconcile` | reconcile_numbers.py | no |
| 16 seeds and variance | `*.seed{0..3}` | run_experiment (varied data_seed) | yes, cheap |
| 17 Spearman phrasing | inside `rev_reconcile` | reconcile_numbers.py | no |

Writing only, no code (do these in LaTeX):
- 3 step 1 (write the KL equation), 3 step 4 (human eval)
- 4 (narrow the headline claim, add the Mix-and-Match / twisted-SMC paragraph)
- 7 (write the CLS energy equation and the within-cell acceptance derivation; the
  measured within-cell vs boundary numbers come out of `rev_reconcile` reading the
  existing `*_mh.csv`)
- 13, 14, 18, 19 (definitions, coverage sentence, prose pass, cost table)

The exact KL definition for concern 3 step 1: the metric is
KL(p_ref || p_pred) averaged over masked positions m with m < L-1, where
p_ref = softmax(logits(orig)[m]) is the next-token distribution after the
GROUND-TRUTH fill and p_pred is the same after the recovered fill. This is
replicated verbatim in `avg_kl_for_fill` in `run_revision.py`. Note this means
the ground-truth baseline is 0 by construction, which is the honest answer to the
circularity objection: the reference is the ground-truth-conditioned distribution,
not the model's own preferred token.

## Order of execution (matches the plan's weeks)

Week 1, no GPU. Validate the stats math, then run the analysis phase on the logs
you already have:

```
python revision/analyze_stats.py --selftest
python gen_manifest_revision.py --phase analysis \
    --grid_dirs results_gpt2 results_llama --diag_dir results_diag \
    --constrained_dir results_constrained \
    --out_dir results_revision > manifest_analysis.tsv
./run_queue.sh --manifest manifest_analysis.tsv --gpus "0" --per_gpu 4 \
    --vram 24 --out_dir results_revision --status status_rev --env gfn
```

The `--grid_dirs` must point at the directories holding the main grid `*.csv`
files (the ones `run_experiment.py` wrote). analyze_stats pairs policy against
grad-norm-preserved and against random by `sample_idx` within each config, so it
needs those per-sample CSVs, not just the JSON summaries.

Week 2, light GPU:

```
python gen_manifest_revision.py --phase light --llama_bf16 --gfn_baselines \
    --out_dir results_revision --diag_dir results_diag > manifest_light.tsv
./run_queue.sh --manifest manifest_light.tsv --gpus "0 1 2 3" --per_gpu 2 \
    --vram 24 --out_dir results_revision --status status_rev --env gfn
```

Week 3, targeted experiments. First validate the SEDD loop with the dry run, then
point it at a real checkpoint:

```
python diagnostics/run_sedd_linearization.py --run_name rev_sedd_dryrun \
    --out_dir results_revision --dry_run --n_seqs 20
# then, after cloning + downloading a SEDD checkpoint:
python gen_manifest_revision.py --phase experiments --llama_bf16 \
    --sedd_dir /path/to/sedd-small > manifest_experiments.tsv
./run_queue.sh --manifest manifest_experiments.tsv --gpus "0 1 2 3" --per_gpu 1 \
    --vram 40 --out_dir results_revision --status status_rev --env gfn
```

The judge score job depends on the generate job's `_gen.csv`, so run the light
phase (which produces `rev_judge_gen_gpt2sft_gen.csv`) before the experiments
phase, or put both manifests in the same queue and let the vram gate order them
(generate is 8 GB, score is 22/40 GB).

## Things to confirm before launching

- Model and adapter paths at the top of `gen_manifest_revision.py` (copied from
  your `gen_manifest.py`).
- The two SEDD hooks in `run_sedd_linearization.py` marked `ADAPT`. Run `--dry_run`
  first; it wires a fake model through the whole loop so the CSV and JSON come out
  before you spend a GPU hour. This is the one experiment with real interface risk.
- `analyze_constrained.py` autodetects its columns because `run_constrained.py`
  is not in this bundle. If it cannot find the gain or arm column it will tell you
  the available columns; pass `--gain_col` / `--arm_col` to fix.

## Cheapest high-value path if the deadline is tight

The plan's own non-negotiable core is concerns 1, 2, 6, and the scope narrowing
(4/7, which are writing plus the free Gibbs baseline). In jobs:

```
rev_stats_*          # concern 1, minutes, no GPU
rev_klbase_gpt2sft   # concerns 2 and 4, a few GPU-hours on one card
rev_reconcile        # concern 6/15/17, minutes, no GPU
```

Those three turn the headline from "we did not find an effect" into "we bound the
effect below X, and a single forward pass beats every Langevin configuration."
Concern 8 (SEDD) is the highest-upside optional item on top of that.
