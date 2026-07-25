# CLAUDE.md

Guidance for Claude Code working in this repository. Read this fully before
running anything. The single most important rule: audit what already exists
before you launch any GPU job. Experiments here are expensive and many have
already been run. Do not re-run a completed job.

## What this project is

A master's thesis with a diagnostic, negative-result thesis: frozen
autoregressive LM likelihoods are not usable energy functions for gradient-guided
or amortized sampling on discrete text. The two load-bearing claims are the
Gradient Fallacy (the LM likelihood gradient carries no usable directional signal;
policy vs random direction is statistically indistinguishable) and the MH
Breakdown (near-total rejection of boundary-crossing moves in continuous Langevin).
The current work is a revision pass answering 19 examiner concerns. The plan is in
`thesis_revision_plan.md`. The run guide with exact commands is `REVISION_README.md`.

## Repository map

- `core/` samplers and energy. `base_sampler.py` (optimize loop + exact KL metric),
  `dls.py` (discrete Langevin), `cls.py` (continuous Langevin), `constraint.py`
  (sentiment constraint), `prep.py` (model loading, including PEFT/LoRA merge).
- `run_experiment.py` the main grid runner. Writes `<out_dir>/<run_name>.csv`
  (per-sample) and `<out_dir>/<run_name>.json` (per-step mean curves + scalars).
- `diagnostics/run_diagnostic.py` the standalone diagnostics: `--exp linearization
  | likelihood_trap | anisotropy | trajectory`. Supports `--adapter_path`.
- `diagnostics/run_revision.py` NEW. `--exp kl_baselines | model_divergence |
  continuation` (concerns 2, 4, 5, 9).
- `diagnostics/run_external_judge.py` NEW. `--stage generate | judge` (concern 3).
- `diagnostics/run_sedd_linearization.py` NEW. SEDD positive control (concern 8),
  has `--dry_run`. Its two model-call hooks are marked ADAPT and MUST be verified
  against github.com/louaaron/Score-Entropy-Discrete-Diffusion before a real run.
- `revision/analyze_stats.py` concern 1 paired stats (has `--selftest`).
- `revision/analyze_likelihood_trap.py` concern 10 within-strategy correlation.
- `revision/reconcile_numbers.py` concerns 6, 15, 17 (numbers index + slopes + MH split).
- `revision/analyze_constrained.py` concern 11 constraint-direction contrast.
- `gen_manifest_revision.py` emits the queue manifest for all of the above.
- `run_queue.sh` + `worker.sh` + `reset_incomplete.sh` the tmux job queue.
- Utility scripts: `compare_models.py` (concern 5 visual), `summarize_constrained.py`,
  `train_sentiment_head.py`, `replot.py`, `notebook_plotting.py`.

## Thesis source (the ground truth for what was written)

The thesis LaTeX lives in the `Doc/` folder (IMS report-class, chapters
Introduction, Background, Related Work, Methodology, Results, Discussion,
Conclusion, Appendix). Treat these `.tex` files as authoritative for what the
thesis actually claims. Whenever there is any doubt about what a number, table,
figure, definition, or sentence in the thesis says, read the relevant `.tex`
section instead of guessing. This matters most for:

- Concern 6, 15, 17: the reconcile step compares reported numbers against the
  result files. Pull the reported values (acceptance rates, the length slope,
  inter-token distances, the config count, the Spearman phrasing) from the `.tex`
  source, not from memory, then diff against `numbers.json`.
- Concern 7: the CLS energy equation and the within-cell acceptance claim need to
  match how Section 2.4 and Section 5.3 actually state them.
- Concerns 3.1, 4, 13, 14, 18, 19: these are writing edits, so quote the exact
  current wording from the `.tex` before proposing a change, and note the file and
  line so the edit can be applied precisely.

If `Doc/` is named differently, locate the `.tex` files first (grep for
`\\chapter{` or the thesis title) and use whatever folder holds them.

## Result folders

- Grid results (per-sample CSV + JSON): `results_gpt2_v2`, `results_llama`, `results_gfn`.
  Run-name schema: `{model}.{sampler}.{method}.{mh}.{gn}.{oracle}.s{steps}`, e.g.
  `gpt2-large.dls.policy.mh.gn.free.s50`. Methods: `policy`,
  `grad_norm_preserved_random_dir`, `random`.
- Constrained results (aggregate JSON): `results_constrained`, `results_probe`.
  Schema: `config{task,setup,target_label,constraint_mode,model_tag}` +
  `steering_gain`, `sentiment_acc_before`, `sentiment_acc`, `final_kl`.
- Diagnostics: DISCOVER these. They may live in `results_diag`, `results_diagnosis`,
  a probe folder, or elsewhere, and may use ad-hoc run_names. Do not assume the
  folder is empty just because `gen_manifest_revision.py`'s default points at
  `results_diagnosis`. Grep for `spearman_surrogate_vs_true` and `likelihood_trap`
  JSONs across all `results_*` folders first.
- Revision analysis outputs: `results_revision`.

## Queue mechanics (get this right or you will waste GPU hours)

- A worker claims a job by `mkdir "$STATUS/<run_name>.lock"` and skips any job whose
  lock exists. That lock, keyed by run_name, is the real resume ledger. Rerunning
  the same queue command skips finished and in-flight jobs regardless of out_dir.
- `<out_dir>/<run_name>.json` is a secondary done-check. A job is truly done when
  that JSON exists.
- `reset_incomplete.sh $STATUS $OUTDIR` clears the lock for any job whose
  `$OUTDIR/<run_name>.json` is missing. Call it with the SAME out_dir the jobs
  wrote to, or it will requeue finished work.
- Every run_name sharing one $STATUS dir must be unique. Use a FRESH $STATUS dir for
  new work so stale locks from earlier grid runs (which share names like
  `gpt2-large.dls.policy...`) do not shadow new jobs.
- Manifest line schema is TSV: `run_name <TAB> min_vram_gb <TAB> command`.

## Harness conventions

- Resume by existence: a job is done when its result JSON exists; JSON is written
  atomically at the end. Analysis scripts take `--run_name` and `--out_dir` and
  follow the same contract.
- Corruption is deterministic per `sample_idx` (seed = data_seed + running index),
  identical across the three method runs of a config, so `analyze_stats.py` pairs
  policy against the comparators on `sample_idx` with no reruns needed.
- The exact KL metric (see `core/base_sampler.py` optimize, replicated as
  `avg_kl_for_fill` in `run_revision.py`): at each masked position m with
  m < seq_len - 1, p_ref = softmax(logits(orig)[m]) is the next-token distribution
  under the GROUND-TRUTH fill, p_pred the same under the recovered fill,
  avg_kl = mean KL(p_ref || p_pred). Ground-truth fill gives KL = 0 by construction.

## Known issues to fix

1. Dtype crash in `diagnostics/run_diagnostic.py`, `exp_linearization`, under a
   bfloat16 model load. `surrogate = (delta_e @ g)` mixes float32 and bfloat16 and
   raises "addmv input tensors must have the same dtype". Fix by casting both
   operands to float32 at the matmul, and audit the rest of the function
   (true_delta and any other einsum/matmul) for the same mismatch under bf16.
   Either run linearization in float32, or apply the cast so bf16 works. Prefer the
   cast so 24 GB cards can run Llama.
2. Constrained raw gains are bias-dominated on the continuation/mucola setup: every
   arm, including lm_only and random, flips sign with the target label, which means
   a fixed sentiment drift dominates the raw gain. The trustworthy number is the
   paired `cons_only - cons_random` contrast, which cancels the shared bias
   (roughly +27 to +37 points on mucola-continuation, near zero on the DLS "ours"
   setup). The one-sided artifact flag in `analyze_constrained.py` currently fires
   on sign-flips too, so treat it as "raw gains are bias-contaminated here, use the
   contrast" rather than an arm-specific defect. This is the correct concern-11
   story: the constraint gradient's direction carries signal, the LM gradient's
   does not.

## Status

The 19-concern revision's code work is COMPLETE. See the final report at the
bottom of REVISION_LOG.md: concerns 1, 2, 3 (minus human eval), 4, 5, 6, 7, 9,
10, 11, 12, 15, 16, 17 all have their numbers in results_revision/ (flat index:
results_revision/numbers.json); SEDD (8) is dry-run validated, real run deferred;
13, 14, 18, 19 and the WRITE halves of the others are LaTeX work. Do NOT re-run
any of it.

Current phase files:
- PROMPT_PHASE2_EXPERIMENTS.md: the last-token experiment (zero-gradient theorem,
  energy-only working sampler, position-condition ablation), the mathematical
  audit, and the optional SEDD real run (needs explicit user go-ahead).
- THESIS_WRITING_GUIDE.md: the authoritative list of every LaTeX edit, with the
  verified numbers inline.
- PROMPT_PHASE3_THESIS.md: instructions for applying the guide to Doc/.

Key mathematical fact for the current phase: under causal attention with
inputs_embeds, the gradient of the sequence log-likelihood with respect to the
FINAL token's input embedding is exactly zero (its embedding only feeds logits
that predict a nonexistent next token), while the energy difference between
candidate final tokens is exactly the model's conditional log-ratio. So at the
last position the energy is maximally usable and the gradient provably useless,
which is the cleanest statement of the thesis's central claim.

## Working rules

- Audit before running. Produce a done/missing/failed table mapped to the 19
  concerns before launching anything.
- Reuse existing results. If a diagnostic already ran under a different run_name,
  point the analysis at it or symlink it to the expected name rather than re-running.
- Keep a running log in `REVISION_LOG.md`: timestamped entries for what you audited,
  what you decided, what you launched, what you fixed, and the resulting numbers.
- Do not launch anything above single-experiment cost, or anything destructive,
  without stating the plan and the GPU cost first.
- No em-dashes in any prose you write into the thesis or logs.
