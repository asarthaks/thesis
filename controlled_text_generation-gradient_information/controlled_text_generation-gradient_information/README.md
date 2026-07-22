# Controlled text generation and the gradient-information question

This repository holds the code, results, and thesis source for a diagnostic,
negative-result master's thesis: **frozen autoregressive LM likelihoods are not usable
energy functions for gradient-guided or amortized sampling on discrete text.** Two
load-bearing empirical claims support it, and one constructive counterfactual closes it.

This README documents the repository **as it currently stands** (a working research tree,
not a cleaned package) and gives the exact commands to reproduce every experiment family
and every number, table, and figure in the thesis. It supersedes `REVISION_README.md`
(kept in place for its detailed concern-by-concern run notes).

## The claim map

- **Necessity (the Gradient Fallacy).** The LM likelihood gradient carries no usable
  directional signal. Policy versus random directions are statistically
  indistinguishable (paired mean KL diff +0.17, 95% CI [-0.29, +0.62], Wilcoxon p 0.40);
  the linearized surrogate is uncorrelated with the true energy change (|rho| < 0.06
  across five models, 400k candidate pairs); and at the final token position the gradient
  of the sequence log-likelihood with respect to the input embedding is provably exactly
  zero, while the energy difference between candidate tokens is exactly the model's
  conditional log-ratio. The gradient is defective; the energy is fine.
- **The MH breakdown.** In continuous Langevin the Metropolis-Hastings correction rejects
  almost every boundary-crossing move (within-cell acceptance ~100% for DLS but the
  continuous state sits >100 units off a token manifold whose tokens are ~1.8-2.8 apart),
  so the sampler is either quenched or wanders off the fluent manifold.
- **Sufficiency and the energy is searchable without the gradient.** A single forward
  pass with top-k rescoring beats every Langevin configuration, and a gradient-free
  Metropolized Gibbs sampler on the *same* energy recovers tokens the gradient samplers
  never reach (Langevin 0/10 on the qualitative showcase; Gibbs / top-k / hybrid 2-4/10).
- **Guidance and the constructive counterfactual.** The constraint classifier's gradient
  *direction* carries measurable signal (paired cons_only - cons_random +27 to +37 pts on
  the MuCoLa continuation) but cannot rescue generation, because following it off the
  fluent manifold breaks the classifiers themselves. A trained-objective pilot (SEDD,
  score-entropy discrete diffusion) confirms the diagnosis: swapping only the direction
  signal (AR left-conditional -> SEDD bidirectional proposal) inside the same
  MH-corrected chain lifts exact recovery 0% -> 39%, and a noisy classifier steers
  absorbing-diffusion generation. On-domain, trust-region guidance (Phase 5) removes the
  fluency cost and steers a held-out judge in one direction, with a residual
  instrument-alignment asymmetry.

The authoritative statement of every claim, number, table, and figure is the thesis
LaTeX under `Doc/` (compile with `latexmk -pdf` in `Doc/`).

## Environment

- Python 3.11, `torch 2.12.0+cu130` (CUDA 13.0), on a university GPU server with 49 GB
  A6000-class cards. The active virtualenv is `gfn-lm-tuning/gfn`.
- Key Python deps: `transformers`, `datasets`, `peft`, `numpy`, `pandas`, `scipy`,
  `matplotlib`, `scikit-learn` (t-SNE trajectory panel), `wandb` (optional; `--no_wandb`
  disables). SEDD needs the patched clone at `$SEDD_REPO`
  (`Score-Entropy-Discrete-Diffusion`) and `HF_HOME` pointing at `hf/cache`.
- Frozen base model (`gpt2sft`):
  `gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output`. Sentiment judge head:
  `/mount/arbeitsdaten/studenten1/singhsk/models/sentiment_constrained_ft_gpt2_large/sentiment_head.pt`.
- SEDD datasets load with custom code: export `HF_DATASETS_TRUST_REMOTE_CODE=1` for the
  ROCStories / trajectory jobs.

## Repository layout (current, as-is)

```
core/                samplers + energy. base_sampler.py (optimize loop + exact KL metric),
                     dls.py, cls.py, constraint.py (sentiment head), prep.py (model + PEFT load)
run_experiment.py    main grid runner -> <out_dir>/<run_name>.csv + .json
diagnostics/         run_diagnostic.py (linearization|likelihood_trap|anisotropy|trajectory),
                     run_revision.py (kl_baselines|model_divergence|continuation|last_token),
                     run_external_judge.py, collect_traces.py, plot_diagnostics.py, analyze_mh.py,
                     sedd_lib.py, run_sedd_cap.py, run_sedd_guided.py, run_gprime.py (Phase 5),
                     aggregate_guided.py, aggregate_gprime.py (Phase 5), merge_sedd_cap.py,
                     train_noisy_classifier.py, run_sedd_linearization.py
revision/            analyze_stats.py, analyze_constrained.py, analyze_likelihood_trap.py,
                     reconcile_numbers.py, build_showcase.py (Phase 5),
                     make_showcase_tex.py (Phase 5), plot_trajectories.py (Phase 5)
Doc/                 thesis LaTeX (thesis.tex master; chapters/ 01..08 + 05a + abstract; references.bib)
figures/             thesis figures (pdf/png)
gen_manifest*.py     manifest emitters for the queue
run_queue.sh worker.sh reset_incomplete.sh   the tmux job queue
run_sedd_slate.sh run_gprime_slate.sh         SEDD / G-prime shard launchers (one shard per GPU)
results_gpt2_v2/ results_llama/ results_gfn/  grid results (per-sample CSV + JSON)
results_constrained/ results_probe/           constrained-generation aggregate JSONs
results_diag/ results_diagnosis/              diagnostics (linearization, trap, anisotropy, traces)
results_revision/    revision analysis outputs (flat index: numbers.json; Phase 4/5 rev_*.json,
                     sedd_capability_summary.json, qualitative_showcase.json)
status_*/            queue lock dirs (one per experiment family)
unifiedruns/ rerun_logs/ wandb/               run logs
```

Nothing has been moved, renamed, or archived; the layout is intentionally left as the
research produced it (see the TODO at the bottom).

## How the queue works

- A worker claims a job by `mkdir "$STATUS/<run_name>.lock"` and skips any job whose lock
  exists. **That lock, keyed by run_name, is the real resume ledger.** Re-running the same
  queue command skips finished and in-flight jobs regardless of `out_dir`.
- `<out_dir>/<run_name>.json` is the secondary done-check; a job is truly done when that
  JSON exists (written atomically at the end).
- `reset_incomplete.sh $STATUS $OUTDIR` clears the lock for any job whose
  `$OUTDIR/<run_name>.json` is missing. **Call it with the SAME out_dir the jobs wrote
  to**, or it requeues finished work.
- Every run_name sharing one `$STATUS` dir must be unique. Use a **fresh `$STATUS` dir**
  for new work so stale locks from earlier grids (which share names like
  `gpt2-large.dls.policy...`) do not shadow new jobs.
- Manifest line schema is TSV: `run_name <TAB> min_vram_gb <TAB> command`.

The SEDD and G-prime slates (`run_sedd_slate.sh`, `run_gprime_slate.sh`) do not use the
lock queue; they assign one shard per GPU via `CUDA_VISIBLE_DEVICES` + `nohup` and resume
by JSON existence, then aggregate.

## Reproducing each experiment family

### Grid (necessity + MH breakdown, 145 configs)

```bash
python gen_manifest.py > manifest_grid.tsv     # or gen_jobs.py / launch_experiments.sh
./run_queue.sh --manifest manifest_grid.tsv --gpus "0 1 2 3" --per_gpu 2 \
    --vram 24 --out_dir results_gpt2_v2 --status status_gpt2_v2 --env gfn
# run_name schema: {model}.{sampler}.{method}.{mh}.{gn}.{oracle}.s{steps}
# e.g. gpt2-large.dls.policy.mh.gn.free.s50 ; methods: policy, grad_norm_preserved_random_dir, random
```

### Diagnostics (linearization, likelihood trap, anisotropy)

```bash
python gen_manifest_revision.py --phase diagnostics --llama_bf16 > manifest_diag.tsv
./run_queue.sh --manifest manifest_diag.tsv --gpus "0 1 2 3" --per_gpu 1 \
    --vram 24 --out_dir results_diagnosis --status status_diag --env gfn
python diagnostics/plot_diagnostics.py --res_dir results_diagnosis --fig_dir figures
```

### MH-acceptance + trajectory traces

`collect_traces.py` uses the patched `core/dls.py` and `core/cls.py`, which add three
optional recorders (`mh_log`, `traj_log`, `proposal_log`), all `None` by default. When
unset the samplers are bit-identical to the originals (no extra `torch.randn`, RNG stream
aligned, so the 145 grid runs stay reproducible). Verify with `python
verify_equivalence_suite.py` before trusting a patched run. If backups exist
(`core/dls.py.bak`, `core/cls.py.bak`) they restore the pre-patch files.

```bash
export HF_DATASETS_TRUST_REMOTE_CODE=1
python diagnostics/collect_traces.py --model_path <gpt2sft> --core_path . \
    --run_name traces_gpt2sft --out_dir results_diag --n_seqs 200 --n_traj_seqs 6 \
    --steps 50 --n_masks 1        # KEEP --n_masks 1 (single-position MH decision)
python diagnostics/analyze_mh.py --csv results_diag/traces_gpt2sft_mh.csv --fig_dir figures
# Phase 5 trajectory PCA/t-SNE (adds dls_random, stores gt_emb; new run_name, canonical untouched):
python diagnostics/collect_traces.py --model_path <gpt2sft> --core_path . \
    --run_name traces_gpt2sft_plot --out_dir results_diag --n_seqs 6 --n_traj_seqs 6 \
    --steps 50 --n_masks 1 --overwrite
python revision/plot_trajectories.py --npz results_diag/traces_gpt2sft_plot_traj.npz --fig_dir figures
```

### Revision analyses (no GPU)

```bash
python revision/analyze_stats.py --results_dir results_gpt2_v2 --run_name rev_stats_gpt2 --out_dir results_revision
python revision/analyze_constrained.py --results_dirs results_constrained results_probe --run_name rev_constrained --out_dir results_revision
python revision/reconcile_numbers.py --results_dirs results_gpt2_v2 results_llama results_gfn results_diagnosis --run_name rev_reconcile --out_dir results_revision
python revision/analyze_likelihood_trap.py --results_dir results_diagnosis --run_name rev_ltrap_within --out_dir results_revision
```

### Revision experiments (GPU)

```bash
# concern 2 baselines + gibbs, concern 5 divergence, concern 9 continuation, concern 3 judge
python gen_manifest_revision.py --phase light --gfn_baselines --llama_bf16 > manifest_light.tsv
python gen_manifest_revision.py --phase experiments --llama_bf16 --sedd_dir /path/to/sedd-small > manifest_exp.tsv
./run_queue.sh --manifest manifest_light.tsv --gpus "0 1 2 3" --per_gpu 2 --vram 24 \
    --out_dir results_revision --status status_rev --env gfn
# last-token (concern 20 / zero-gradient theorem):
python diagnostics/run_revision.py --exp last_token --run_name rev_last_token_gpt2sft \
    --out_dir results_revision --model_path <gpt2sft> --n_samples 200
```

### SEDD capability slate (Phase 4) and G-prime (Phase 5)

```bash
export HF_HOME=<repo>/hf/cache SEDD_REPO=<repo>/Score-Entropy-Discrete-Diffusion
# gates / recovery / hybrid, both scales, sharded one-per-GPU:
./run_sedd_slate.sh 6
# guided generation gamma sweep (Phase 4):
python diagnostics/run_sedd_guided.py --run_name rev_sedd_guided_g2 --gamma 2 \
    --gpt2sft_path <gpt2sft> --head <head> --scale medium
python diagnostics/aggregate_guided.py --shard_glob "rev_sedd_guided_g2.shard*" --final_run_name rev_sedd_guided_g2
# on-domain trust-region guided generation (Phase 5, held-out SST-2 prompts):
./run_gprime_slate.sh 9        # -> rev_gprime.json via aggregate_gprime.py
# qualitative showcase (Phase 5):
python revision/build_showcase.py                 # -> results_revision/qualitative_showcase.json
python revision/make_showcase_tex.py --tex_out results_revision/showcase_appendix.tex
```

## Artifact map (thesis table/figure -> producing script -> result file)

Anchored on `results_revision/numbers.json` (grid + reconcile, keyed by run_name) and
`results_revision/sedd_capability_summary.json` (SEDD gates/linearization/recovery/hybrid/guided).

| Thesis object | Producing script | Result file |
|---|---|---|
| Table `tab:full-grid` (145-config grid) | `run_experiment.py` (grid) | `results_gpt2_v2`, `results_llama`, `results_gfn`; `numbers.json` |
| Table `tab:fallacy` (concern 1 paired stats) | `revision/analyze_stats.py` | `results_revision/rev_stats_gpt2.json` |
| Table `tab:baselines` (KL baselines + Gibbs) | `run_revision.py --exp kl_baselines` | `results_revision/rev_klbase_gpt2sft.json` |
| Table `tab:gfn-unify` (GFlowNet variants) | `run_revision.py --exp kl_baselines` (gfn) | `results_revision/rev_klbase_gfn-*.json` |
| Table `tab:divergence` (base vs tuned energy) | `run_revision.py --exp model_divergence` | `results_revision/rev_divergence_*.json` |
| Table `tab:constrained` (constraint contrast) | `revision/analyze_constrained.py` | `results_revision/rev_constrained.json` |
| Table `tab:lasttoken` (position conditions) | `run_revision.py --exp last_token` | `results_revision/rev_last_token_gpt2sft.json` |
| Table `tab:diffusion-lin` (SEDD linearization) | `run_sedd_linearization.py` / `run_sedd_cap.py` | `rev_sedd_lin_{small,medium}.json`; `sedd_capability_summary.json` |
| Table `tab:diffusion-recovery` (SEDD recovery) | `run_sedd_cap.py --exp recovery` | `rev_sedd_recovery_{small,medium}.json` |
| Table `tab:diffusion-hybrid` (hybrid sufficiency) | `run_sedd_cap.py --exp hybrid` | `rev_sedd_hybrid.json` |
| Table `tab:diffusion-guided` (guided steering) | `run_sedd_guided.py` (Phase 4) + `run_gprime.py` (Phase 5) | `rev_sedd_guided_g{1,2,4}.json`; `rev_gprime.json` |
| Table `tab:cost` (compute cost) | hand-derived + wall-clocks | `rev_continuation`, `rev_klbase`, `rev_judge` JSONs |
| Figs `fig:lin-radius/scatter/decomp/topk` | `run_diagnostic.py --exp linearization` -> `plot_diagnostics.py` | `results_diag(nosis)/diag_linearization_*` |
| Figs `fig:mh-accept`, `fig:mh-decomp` | `collect_traces.py` -> `analyze_mh.py` | `results_diag/traces_gpt2sft_mh.csv` |
| Figs `fig:trap`, `fig:trap-length` | `run_diagnostic.py --exp likelihood_trap` -> `plot_diagnostics.py` | `results_diag(nosis)/diag_ltrap_*` |
| Fig `fig:aniso` | `run_diagnostic.py --exp anisotropy` -> `plot_diagnostics.py` | `results_diag/diag_anisotropy_*` |
| Figs `fig:traj-dls-pca`, `fig:traj-cls-pca` (Phase 5) | `revision/plot_trajectories.py` | `figures/fig_traj_pca_{dls,cls}.png` from `traces_gpt2sft_plot_traj.npz` |
| Figs `fig:dls-traj-50/100` | `run_experiment.py` / `replot.py` | `figures/gpt2-large.dls.gn.free.s{50,100}_new_trajectories.png` |
| Fig `fig:lasttoken` | `run_revision.py --exp last_token` | `results_revision/last_token_figure.png` |
| Appendix showcase (Phase 5) | `build_showcase.py` + `make_showcase_tex.py` | `results_revision/qualitative_showcase.json`, `showcase_appendix.tex` |

## Known caveats

- **SEDD runs are excluded from the AR reconcile globs.** `reconcile_numbers.py` counts
  and diffs only the autoregressive grid; SEDD (`rev_sedd_*`) lives outside those globs by
  design. Do not fold SEDD run_names into the AR config count (145 = 5 x 29).
- **The `gn=on` bitwise `gradnorm == random` artifact.** With grad-normalization on, the
  grad-norm-preserved-random-dir arm and the random arm are bitwise identical in several
  measurements (e.g. judge ppl 181.32 for both; continuation 8.850 for both). This is
  expected: normalizing the gradient magnitude and then substituting a random direction of
  the same norm collapses the two arms. Report them as one where this holds.
- **Concern 6a attribution.** The reported CLS acceptance pair (0.03% within / 3.7%
  boundary) matches the CLS-policy *no-MH* split in the refreshed reconcile
  (0.034 / 3.665); the MH=True split is 0.627% / 8.56%, and the DLS-MH within-cell /
  boundary contrast is 100% / 9.3%. The thesis now reports all three explicitly.
- **Trajectory figures use a clean 5-config regeneration** (`traces_gpt2sft_plot`), not the
  canonical `traces_gpt2sft` npz, because the torch RNG carries across configs (so a single
  config cannot be spliced at the canonical RNG state) and the figures were placeholders.
  CLS states escape >100 units off the token manifold (MH-off to a max of ~979); the PCA
  panels clip extreme excursions so the anisotropy cone stays visible.

## TODO

Deferred by author decision: repository reorganization (move results_* under results/,
code under scripts/, strays to archive/). Do NOT perform this until the author explicitly
asks for it and confirms. When it happens: git mv only, Doc/ never moves, nothing is
deleted, and a full reference sweep across *.py, *.sh, *.md, *.tex (including % source
comments and reconcile globs) plus the Stage 3 verification gate must follow.
