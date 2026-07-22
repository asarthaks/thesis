> **Superseded by `README.md`** (the new top-level guide, which absorbs this file). Kept
> in place for its detailed concern-by-concern run notes; nothing here has been deleted.

# Revision experiments and analyses: exact run guide

Every script here follows the harness contract: a job is DONE when
`<out_dir>/<run_name>.json` exists, JSON is written atomically at the end, and
`run_queue.sh` resumes by skipping finished jobs. So you can re-run any command
below and it picks up where it stopped.

How resume actually works, so you can reason about reruns. `worker.sh` claims a
job by `mkdir`-ing `$STATUS/<run_name>.lock`, and it skips any job whose lock
already exists. That lock, keyed by `run_name`, is the real ledger:

- Rerunning the same queue command skips finished and in-flight jobs by their lock,
  regardless of which `out_dir` each job writes to.
- Two invariants follow. Every `run_name` sharing one `$STATUS` dir must be unique
  (they are here). And a crash leaves a stale lock with no result JSON, which
  blocks that job until you clear it.

Clearing is `reset_incomplete.sh $STATUS $OUTDIR`, and here is the one sharp edge:
it removes the lock for any job whose `$OUTDIR/<run_name>.json` is missing. So you
must call it with the SAME `--out_dir` you queued with. If a job wrote its JSON to
a different folder than the `$OUTDIR` you pass to reset, reset treats it as
unfinished and requeues a job that actually completed. That is the real reason to
keep one queue call's jobs aligned to one `out_dir`. The phases below each use a
single `(status, out_dir)` pair, and the generator routes jobs to match, so
`reset_incomplete.sh` with that same pair is always safe. Use a fresh `$STATUS`
dir for this revision effort so stale locks from earlier runs do not shadow jobs
that share a name.

## State of your folders (from the inspection)

Ready to analyze now, no reruns:
- `results_gpt2_v2`, `results_llama`, `results_gfn` are grid folders
  (`sample_idx,method,mh_enabled,...,avg_kl_div,...`). Concern 1 reads these.
- `results_constrained` and `results_probe` hold the constrained JSONs
  (`config` + `steering_gain` + `sentiment_acc`). Concern 11 reads both.

Empty, must be generated on GPU before its analysis can run:
- `results_diagnosis` is empty. The linearization Spearman (concern 6/17) and the
  likelihood-trap correlations (concern 10) come from here, so the diagnostics
  phase has to run before `rev_reconcile` and `rev_ltrap_within` have inputs.

## Step 0. Validate the stats math (seconds, no GPU)

```bash
python revision/analyze_stats.py --selftest
```

## Phase A. What you can run today, no GPU

Concern 1, paired statistics behind the null, one per grid folder:

```bash
python revision/analyze_stats.py --results_dir results_gpt2_v2 --run_name rev_stats_gpt2 --out_dir results_revision
python revision/analyze_stats.py --results_dir results_llama   --run_name rev_stats_llama --out_dir results_revision
python revision/analyze_stats.py --results_dir results_gfn     --run_name rev_stats_gfn  --out_dir results_revision
```

Concern 11, constraint-direction contrast across both constrained folders:

```bash
python revision/analyze_constrained.py --results_dirs results_constrained results_probe --run_name rev_constrained --out_dir results_revision
```

That prints, per (model, task, setup, target), the `cons_only` minus
`cons_random` gap (the value of the constraint gradient's direction), the
`lm_only` gain (your null result reproduced on sentiment), and any one-sided
artifact flags. Your existing `summarize_constrained.py` gives the same contrast
as a quick console table if you want a second view:

```bash
python summarize_constrained.py --dir results_constrained
```

Note: pairing in concern 1 needs the three methods in the same folder. In
`results_gpt2_v2` you have 22 policy, 22 random, 14 grad_norm_preserved. Configs
that have a grad_norm run get both the policy-vs-gradnorm and policy-vs-random
pairing; configs missing it still get policy-vs-random. Nothing breaks, you just
get fewer comparators on those configs. Run the same one-line check on the other
two folders so you know what to expect:

```bash
for d in results_llama results_gfn; do echo "== $d =="; ls "$d" | grep -oE 'policy|random|grad_norm_preserved_random_dir' | sort | uniq -c; done
```

## Phase B. Generate the diagnostics (GPU), then finish the no-GPU analysis

This populates `results_diagnosis` with linearization, likelihood-trap, and
anisotropy across the model family. `--llama_bf16` tags the Llama jobs at 22 GB so
a 24 GB card runs them.

```bash
python gen_manifest_revision.py --phase diagnostics --llama_bf16 > manifest_diag.tsv
./run_queue.sh --manifest manifest_diag.tsv --gpus "0 1 2 3" --per_gpu 1 \
    --vram 24 --out_dir results_diagnosis --status status_diag --env gfn
tmux attach -t ctg_queue      # watch progress; detach with Ctrl-b then d
```

If a job OOMs or you kill the session, requeue only the unfinished ones with the
matching pair, then relaunch the same queue command:

```bash
bash reset_incomplete.sh status_diag results_diagnosis     # diagnostics phase
# bash reset_incomplete.sh status_rev results_revision     # light / experiments phases
```

When that finishes, run the two no-GPU analyses that depend on it:

```bash
python revision/reconcile_numbers.py --results_dirs results_gpt2_v2 results_llama results_gfn results_diagnosis --run_name rev_reconcile --out_dir results_revision
python revision/analyze_likelihood_trap.py --results_dir results_diagnosis --run_name rev_ltrap_within --out_dir results_revision
```

`rev_reconcile` writes `results_revision/rev_reconcile.json` plus a flat
`results_revision/numbers.json` you diff against the LaTeX. It gives you the
length slope censored vs uncensored (resolves the -1.12 vs -0.11 question), the MH
acceptance split into within-cell vs boundary crossing (concern 6a and 7), the
config-count vs factorial reconciliation (concern 15), and the corrected Spearman
phrasing sentence (concern 17). If it reports the Spearman as missing, the
linearization diag jobs did not finish, check `status_diag` for `.failed` files.

## Phase C. Light GPU experiments

Concern 2 (KL baselines, includes the non-gradient Gibbs sampler for concern 4),
concern 5 (energy divergence base vs tuned), concern 3 generate stage, concern 12
per-method oracle. `--gfn_baselines` adds the KL baselines on the three GFlowNet
variants too.

```bash
python gen_manifest_revision.py --phase light --gfn_baselines --llama_bf16 > manifest_light.tsv
./run_queue.sh --manifest manifest_light.tsv --gpus "0 1 2 3" --per_gpu 2 \
    --vram 24 --out_dir results_revision --status status_rev --env gfn
```

For concern 5 you also have a zero-rerun visual already in the repo, which
overlays the SFT and GFlowNet curves on identical axes and dumps a final-KL table:

```bash
python compare_models.py --results_dirs results_gpt2_v2 results_gfn --out_dir figures_compare
```

## Phase D. Targeted experiments

Run the SEDD dry-run first. It threads a fake model through the whole loop so the
CSV and JSON come out before you spend a GPU hour. This is the one experiment with
real interface risk (its two model-call hooks are marked ADAPT and must match the
repo you clone from github.com/louaaron/Score-Entropy-Discrete-Diffusion).

```bash
python diagnostics/run_sedd_linearization.py --run_name rev_sedd_dryrun --out_dir results_revision --dry_run --n_seqs 20
```

Then the experiments phase. The judge score job (concern 3) reads the `_gen.csv`
the light phase produced, so run Phase C first. Point `--sedd_dir` at a downloaded
SEDD checkpoint to swap the dry-run for the real run.

```bash
python gen_manifest_revision.py --phase experiments --llama_bf16 --sedd_dir /path/to/sedd-small > manifest_exp.tsv
./run_queue.sh --manifest manifest_exp.tsv --gpus "0 1 2 3" --per_gpu 1 \
    --vram 24 --out_dir results_revision --status status_rev --env gfn
```

If you have not cloned SEDD yet, drop `--sedd_dir` and the manifest emits the
dry-run job instead, so the rest of the phase still runs.

## Concern to job map

| Concern | How | Code or writing |
|--------|-----|-----------------|
| 1 statistics behind the null | `rev_stats_*` (Phase A) | code |
| 2 missing KL baselines | `rev_klbase_*` (Phase C) | code |
| 3 circular evaluation | `rev_judge_gen_*` (C) then `rev_judge_score_*` (D); KL equation | code + writing |
| 4 non-gradient sampler | the `gibbs` baseline inside concern 2; scope narrowing | code + writing |
| 5 did tuning change the energy | `rev_divergence_*` (C) and `compare_models.py` | code |
| 6 numerical inconsistencies | `rev_reconcile` (Phase B) | code |
| 7 CLS energy and within-cell acceptance | numbers from `rev_reconcile`; derivation | code + writing |
| 8 SEDD positive control | `rev_sedd_*` (Phase D) | code |
| 9 task generality | `rev_continuation_*` (Phase D) | code |
| 10 likelihood-trap confound | diagnostics (B) then `rev_ltrap_within` | code |
| 11 constraint CIs and anomaly | `rev_constrained` (Phase A) | code |
| 12 oracle fairness | `*.oracle.s50` per method (Phase C) | code |
| 13, 14, 18, 19 | definitions, softening, prose, cost table | writing |
| 15 config count | inside `rev_reconcile` | code |
| 16 seeds and variance | `*.seed{0..3}` (Phase D) | code |
| 17 Spearman phrasing | inside `rev_reconcile` | code |

Writing-only, no job will produce these: concern 3 step 1 (write the KL equation,
which is implemented verbatim as `avg_kl_for_fill` in `run_revision.py` so you can
transcribe it), concern 3 step 4 (human eval), concern 4 (narrow the headline
claim, add the Mix-and-Match and twisted-SMC paragraph), concern 7 (write the CLS
energy equation and the within-cell acceptance derivation, using the measured
numbers from `rev_reconcile`), and concerns 13, 14, 18, 19.

## Confidence-interval note for concern 11

The constrained JSONs store one aggregate `steering_gain` per cell, not per-example
gains, so there is nothing to bootstrap over. The evidence there is the
`cons_only` minus `cons_random` gap and its consistency across the two target
labels, which is what `rev_constrained` reports. If you want per-cell CIs, have
`run_constrained` dump per-example gains as `<run_name>_samples.csv` and pass
`--per_sample_dir` to `analyze_constrained.py`; it will add bootstrap CIs
automatically.

## Cheapest high-value path if the deadline is tight

The non-negotiable core is concerns 1, 2, 6, plus the scope narrowing (4 and 7,
which are writing plus the free Gibbs baseline). In commands:

```bash
python revision/analyze_stats.py --results_dir results_gpt2_v2 --run_name rev_stats_gpt2 --out_dir results_revision
# Phase B diagnostics queue (for reconcile), then:
python revision/reconcile_numbers.py --results_dirs results_gpt2_v2 results_llama results_gfn results_diagnosis --run_name rev_reconcile --out_dir results_revision
# one GPU job for concern 2 and 4 together:
python gen_manifest_revision.py --phase light --llama_bf16 > m.tsv   # then run only rev_klbase_gpt2sft from it
```

Those turn the headline from "we found no effect" into "we bound the effect below
X, and a single forward pass beats every Langevin configuration." Concern 8 (SEDD)
is the highest-upside optional item on top of that.
