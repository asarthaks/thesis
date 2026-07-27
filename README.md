# Controlled text generation and the gradient-information question

This repository holds the code, results, and thesis source for a diagnostic,
negative-result master's thesis: **frozen autoregressive LM likelihoods are not usable
energy functions for gradient-guided or amortized sampling on discrete text.** Two
load-bearing empirical claims support it, and one constructive counterfactual closes it.

This README documents the repository **as it currently stands** (a research tree that was
reorganized on 2026-07-25; see the "Repository layout" note at the bottom) and gives the
exact commands to reproduce every experiment family and every number, table, and figure in
the thesis. It supersedes `REVISION_README.md` (kept in place for its detailed
concern-by-concern run notes).

## The claim map

- **Necessity, scoped to the INPUT-EMBEDDING gradient.** Differentiating the frozen
  likelihood with respect to the input embedding, which is what MuCoLa- and COLD-style
  samplers do, yields no usable directional signal. Policy versus norm-matched random
  directions are statistically indistinguishable (paired mean KL diff +0.17, 95% CI
  [-0.29, +0.62], Wilcoxon p 0.40; on chain statistics +0.002, p 0.75); the linearized
  surrogate is uncorrelated with the true energy change (|rho| < 0.06 across five models,
  400k candidate pairs); a 5x5 (epsilon, temperature) sweep spanning proposal entropies
  from uniform to deterministic finds no configuration where it helps; and at the final
  token position that gradient is provably exactly zero while the energy difference
  between candidate tokens is exactly the model's conditional log-ratio.
- **But the usable direction EXISTS, on the output side.** The one-hot / simplex gradient,
  which is what the discrete samplers this family adapts actually differentiate, has
  v-th coordinate log p(v | x_<i) + g^T e(v). Re-analysing the SAME linearization
  measurements with that surrogate gives Spearman 0.37 to 0.59 overall and 0.60 to 0.73
  at admissible substitution distances, against ~0 for the input-embedding surrogate
  (`revision/analyze_onehot_surrogate.py` -> `results/revision/rev_onehot_surrogate.json`).
  The energy is fine and so is its output-side derivative; the input-embedding Jacobian
  slice is what discards the signal. This is why top-k rescoring, Gibbs and SEDD all work.
  DEMONSTRATED AT THE SAMPLER LEVEL: substituting the one-hot surrogate into the same DLS
  sampler over the same (eps, temperature) sweep reaches 40% exact recovery, against a
  maximum of 2% for the input-embedding gradient anywhere in that sweep, and comparable to
  score-trained SEDD at 39%. Cross-model on the GFlowNet energies: 25/21/9% sharp vs 0.5%
  calibrated. See `revision/analyze_rev3.py` -> `results/revision/rev_rev3_summary.json`.
- **Bidirectional conditioning, not score training, is the operative variable.** In the same
  exact-energy MH chain: uniform proposal 0.5%, AR left-conditional 23.5%, SEDD 39%,
  RoBERTa-large MLM (never score-trained) 44.5%. The uniform arm reproduces the Langevin
  flagship (0.0%, KL 6.538 vs 6.541, accept 9.48 vs 9.98%) from an independent
  implementation, confirming the near-uniform-proposal finding. See
  `diagnostics/run_mlm_control.py` -> `results/revision/rev_mlm_control.json`.
- **The gradient is NOT actively harmful.** Earlier readings that the true gradient was
  reliably WORSE than a random direction (the Llama-3 contrast, the sharp sweep cells) were
  artifacts of applying the MH reverse-proposal term to the policy arm only. With
  `--mh_exact_all_arms` the effect disappears (+2.34 -> -0.25 nats). The claim is
  indifference, not anti-guidance.
- **The proposal is near-uniform in the main grid.** Measured proposal entropy is 10.8248
  nats against log|V| = 10.8249, with a floor of 10.22 over the whole schedule, so the
  flagship ablation could not have detected an informative gradient had one been present.
  Exact recovery is 0.0% in 139 of 145 configurations, and across the sweep the chain never
  even VISITS the ground-truth token (`ever_accuracy` 0.135% mean). The "quenching effect"
  formerly reported in Section 5.2 is WITHDRAWN: a bit-identical fresh re-run shows mean KL
  rising (8.76 -> 9.50), no late convergence, and tokens still changing on 99% of
  final-decile steps.
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
LaTeX under `Doc/final/thesis/` (compile with `latexmk -pdf thesis.tex` in
`Doc/final/thesis/`; figures resolve from `Doc/figures/`). The proposal is at
`Doc/final/proposal/` and the defense talk at `Doc/final/beamer/`.

## Environment

- Python 3.11, `torch 2.12.0+cu130` (CUDA 13.0), on a university GPU server with 49 GB
  A6000-class cards. The active virtualenv is `gfn-lm-tuning/gfn` (never moved; venvs
  hardcode absolute paths).
- Two install paths: exact reproduction is
  `gfn-lm-tuning/gfn/bin/python -m pip install -r requirements.txt.lock` (the frozen
  pin set, 141 packages, `torch==2.12.0+cu130`, `transformers==4.38.2`,
  `datasets==2.19.2`, `peft==0.9.0`); loose setup is `pip install -r requirements.txt`.
- Key Python deps: `transformers`, `datasets`, `peft`, `numpy`, `pandas`, `scipy`,
  `matplotlib`, `scikit-learn` (t-SNE trajectory panel), `wandb` (optional; `--no_wandb`
  disables). SEDD needs the clone at `$SEDD_REPO`
  (`external/Score-Entropy-Discrete-Diffusion`) and `HF_HOME` pointing at
  `external/hf/cache`.
- **Path anchors live in `env.sh`.** The workflow is `cd <repo root> && source env.sh`,
  which exports `REPO_ROOT`, `RESULTS_DIR`, `STATUS_DIR`, `LOGS_DIR`, `SEDD_REPO`,
  `HF_HOME`, `GPT2SFT`, `SENTIMENT_HEAD`, `PYTHONPATH`, and
  `HF_DATASETS_TRUST_REMOTE_CODE=1`. A future move of the tree is then a single edit.
- Frozen base model (`gpt2sft` / `$GPT2SFT`):
  `gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output`. Sentiment judge head
  (`$SENTIMENT_HEAD`, a server-specific absolute path outside the repo):
  `/mount/arbeitsdaten/studenten1/singhsk/models/sentiment_constrained_ft_gpt2_large/sentiment_head.pt`.
- SEDD datasets load with custom code: `HF_DATASETS_TRUST_REMOTE_CODE=1` (exported by
  `env.sh`) for the ROCStories / trajectory jobs.

## Repository layout

The runnable repo root is the `thesis/` workspace (the git root). Run everything from
there with `source env.sh` first. Packages stay at the root so imports resolve; the entry
runners and shell launchers live in `scripts/`; results, status locks, and logs are
grouped.

```
core/                samplers + energy. base_sampler.py (optimize loop + exact KL metric),
                     dls.py, cls.py, constraint.py (sentiment head), prep.py (model + PEFT load)
diagnostics/         run_diagnostic.py (linearization|likelihood_trap|anisotropy|trajectory),
                     run_revision.py (kl_baselines|model_divergence|continuation|last_token),
                     run_external_judge.py, collect_traces.py, plot_diagnostics.py, analyze_mh.py,
                     sedd_lib.py, run_sedd_cap.py, run_sedd_guided.py, run_gprime.py (Phase 5),
                     aggregate_guided.py, aggregate_gprime.py (Phase 5), merge_sedd_cap.py,
                     train_noisy_classifier.py, run_sedd_linearization.py
revision/            analyze_stats.py, analyze_constrained.py, analyze_likelihood_trap.py,
                     reconcile_numbers.py, build_showcase.py (Phase 5),
                     make_showcase_tex.py (Phase 5), plot_trajectories.py (Phase 5)
Methods/ Experiments/  legacy samplers (imported by the equivalence suite) + old aux experiment
scripts/             entry runners + launchers, all run from the repo root:
                     run_experiment.py (grid runner -> <out_dir>/<run_name>.csv + .json),
                     run_constrained.py, gen_manifest*.py, gen_diag_manifest.py, gen_jobs.py,
                     compare_models.py, replot.py, notebook_plotting.py, summarize_constrained.py,
                     evaluate.py, train_sentiment_head.py, test_sentiment_head.py,
                     verify_equivalence_suite.py, verify_logic*.py, verify_constraint_live.py,
                     smoke_test_gfn.py,
                     run_queue.sh worker.sh reset_incomplete.sh   the tmux job queue,
                     run_sedd_slate.sh run_gprime_slate.sh        SEDD / G-prime shard launchers,
                     launch_experiments.sh launch_multi.sh run_constraint_ablation.sh gpu_monitor.sh
Doc/final/thesis/    canonical thesis LaTeX: thesis.tex master (article/template class),
                     chapters/ 01..08 + 05a + abstract + showcase/gprime/tab_confusion,
                     references.bib. Figures resolve from Doc/figures/ (graphicspath ../../)
Doc/final/proposal/  proposal.tex (amended: contingency section + errata)
Doc/final/beamer/    Presentation.tex defense talk (+ bg.png)
Doc/figures/         thesis figure PDFs/PNGs (LaTeX build inputs); Doc/prev_version/ pre-revision source
figures/             code-output figures (pdf/png; plot scripts write here via --fig_dir figures);
                     figures/{gpt2,gfn,llama,compare}/ hold the per-model plot outputs
results/
  grid/{gpt2_v2,llama,gfn,rerun}   grid results (per-sample CSV + JSON)
  constrained/{main,probe}          constrained-generation aggregate JSONs
  diagnostics/{diag,diagnosis}      diagnostics (linearization, trap, anisotropy, traces)
  revision/                         revision analysis outputs (flat index: numbers.json;
                                    Phase 4/5 rev_*.json, sedd_capability_summary.json,
                                    qualitative_showcase.json)
  legacy/                           one legacy pre-v2 CSV
status/{gpt2_v2,llama,gfn,constrained,diag,rev,rev_v2}   queue lock dirs (one per family)
logs/{unifiedruns,rerun_logs,wandb,manifests}            run logs + historical manifest snapshots
external/            Score-Entropy-Discrete-Diffusion (SEDD clone), hf/ (HF cache); no venv inside
gfn-lm-tuning/       GFlowNet tree + the live `gfn` virtualenv + gpt2sft checkpoint (never moved)
refs/  meetings/     reference PDFs + reference thesis; meeting transcripts
env.sh               path anchors (source this first); requirements.txt.lock  frozen pin set
archive/             emptied nesting shells, duplicate root files, *.zip, strays (nothing deleted)
```

The tree was reorganized on 2026-07-25 by a git-mv-only pass; the full old-to-new move
map and the nothing-deleted certification are in `REVISION_RESTRUCTURING.md`, and the
pre-reorg commit `4bd1ee4` is the rollback point.

## How the queue works

- A worker claims a job by `mkdir "$STATUS/<run_name>.lock"` and skips any job whose lock
  exists. **That lock, keyed by run_name, is the real resume ledger.** Re-running the same
  queue command skips finished and in-flight jobs regardless of `out_dir`.
- `<out_dir>/<run_name>.json` is the secondary done-check; a job is truly done when that
  JSON exists (written atomically at the end).
- `scripts/reset_incomplete.sh $STATUS $OUTDIR` clears the lock for any job whose
  `$OUTDIR/<run_name>.json` is missing. **Call it with the SAME out_dir the jobs wrote
  to**, or it requeues finished work.
- Every run_name sharing one `$STATUS` dir must be unique. Use a **fresh `$STATUS` dir**
  for new work so stale locks from earlier grids (which share names like
  `gpt2-large.dls.policy...`) do not shadow new jobs.
- Manifest line schema is TSV: `run_name <TAB> min_vram_gb <TAB> command`.
- The launchers live in `scripts/` and `cd` to the repo root themselves, so run them as
  `./scripts/run_queue.sh ...` from the root; the `python scripts/run_experiment.py ...`
  commands they eval resolve `import core` from the root.

The SEDD and G-prime slates (`scripts/run_sedd_slate.sh`, `scripts/run_gprime_slate.sh`)
do not use the lock queue; they assign one shard per GPU via `CUDA_VISIBLE_DEVICES` +
`nohup` and resume by JSON existence, then aggregate.

## Reproducing each experiment family

First, from the repo root: `source env.sh` (exports the path anchors below). All commands
run from the root.

### Grid (necessity + MH breakdown, 145 configs)

```bash
python scripts/gen_manifest.py > manifest_grid.tsv     # or gen_jobs.py / launch_experiments.sh
./scripts/run_queue.sh --manifest manifest_grid.tsv --gpus "0 1 2 3" --per_gpu 2 \
    --vram 24 --out_dir results/grid/gpt2_v2 --status status/gpt2_v2 --env gfn
# run_name schema: {model}.{sampler}.{method}.{mh}.{gn}.{oracle}.s{steps}
# e.g. gpt2-large.dls.policy.mh.gn.free.s50 ; methods: policy, grad_norm_preserved_random_dir, random
```

### Diagnostics (linearization, likelihood trap, anisotropy)

```bash
python scripts/gen_manifest_revision.py --phase diagnostics --llama_bf16 > manifest_diag.tsv
./scripts/run_queue.sh --manifest manifest_diag.tsv --gpus "0 1 2 3" --per_gpu 1 \
    --vram 24 --out_dir results/diagnostics/diagnosis --status status/diag --env gfn
python diagnostics/plot_diagnostics.py --res_dir results/diagnostics/diagnosis --fig_dir figures
```

### MH-acceptance + trajectory traces

`collect_traces.py` uses the patched `core/dls.py` and `core/cls.py`, which add three
optional recorders (`mh_log`, `traj_log`, `proposal_log`), all `None` by default. When
unset the samplers are bit-identical to the originals (no extra `torch.randn`, RNG stream
aligned, so the 145 grid runs stay reproducible). Verify with `python
scripts/verify_equivalence_suite.py` before trusting a patched run. If backups exist
(`core/dls.py.bak`, `core/cls.py.bak`) they restore the pre-patch files.

```bash
export HF_DATASETS_TRUST_REMOTE_CODE=1     # (env.sh already exports this)
python diagnostics/collect_traces.py --model_path <gpt2sft> --core_path . \
    --run_name traces_gpt2sft --out_dir results/diagnostics/diag --n_seqs 200 --n_traj_seqs 6 \
    --steps 50 --n_masks 1        # KEEP --n_masks 1 (single-position MH decision)
python diagnostics/analyze_mh.py --csv results/diagnostics/diag/traces_gpt2sft_mh.csv --fig_dir figures
# Phase 5 trajectory PCA/t-SNE (adds dls_random, stores gt_emb; new run_name, canonical untouched):
python diagnostics/collect_traces.py --model_path <gpt2sft> --core_path . \
    --run_name traces_gpt2sft_plot --out_dir results/diagnostics/diag --n_seqs 6 --n_traj_seqs 6 \
    --steps 50 --n_masks 1 --overwrite
python revision/plot_trajectories.py --npz results/diagnostics/diag/traces_gpt2sft_plot_traj.npz --fig_dir figures
```

### Revision analyses (no GPU)

```bash
python revision/analyze_stats.py --results_dir results/grid/gpt2_v2 --run_name rev_stats_gpt2 --out_dir results/revision
python revision/analyze_constrained.py --results_dirs results/constrained/main results/constrained/probe --run_name rev_constrained --out_dir results/revision
python revision/reconcile_numbers.py --results_dirs results/grid/gpt2_v2 results/grid/llama results/grid/gfn results/diagnostics/diagnosis --run_name rev_reconcile --out_dir results/revision
python revision/analyze_likelihood_trap.py --results_dir results/diagnostics/diagnosis --run_name rev_ltrap_within --out_dir results/revision
```

### Revision experiments (GPU)

```bash
# concern 2 baselines + gibbs, concern 5 divergence, concern 9 continuation, concern 3 judge
python scripts/gen_manifest_revision.py --phase light --gfn_baselines --llama_bf16 > manifest_light.tsv
python scripts/gen_manifest_revision.py --phase experiments --llama_bf16 --sedd_dir /path/to/sedd-small > manifest_exp.tsv
./scripts/run_queue.sh --manifest manifest_light.tsv --gpus "0 1 2 3" --per_gpu 2 --vram 24 \
    --out_dir results/revision --status status/rev --env gfn
# last-token (concern 20 / zero-gradient theorem):
python diagnostics/run_revision.py --exp last_token --run_name rev_last_token_gpt2sft \
    --out_dir results/revision --model_path <gpt2sft> --n_samples 200
```

### SEDD capability slate (Phase 4) and G-prime (Phase 5)

```bash
# env.sh already exports HF_HOME=<repo>/external/hf/cache and
# SEDD_REPO=<repo>/external/Score-Entropy-Discrete-Diffusion (the slates also set them).
# gates / recovery / hybrid, both scales, sharded one-per-GPU:
./scripts/run_sedd_slate.sh 6
# guided generation gamma sweep (Phase 4):
python diagnostics/run_sedd_guided.py --run_name rev_sedd_guided_g2 --gamma 2 \
    --gpt2sft_path <gpt2sft> --head <head> --scale medium
python diagnostics/aggregate_guided.py --shard_glob "rev_sedd_guided_g2.shard*" --final_run_name rev_sedd_guided_g2
# on-domain trust-region guided generation (Phase 5, held-out SST-2 prompts):
./scripts/run_gprime_slate.sh 9        # -> rev_gprime.json via aggregate_gprime.py
# qualitative showcase (Phase 5):
python revision/build_showcase.py                 # -> results/revision/qualitative_showcase.json
python revision/make_showcase_tex.py --tex_out results/revision/showcase_appendix.tex
```

## Artifact map (thesis table/figure -> producing script -> result file)

Anchored on `results/revision/numbers.json` (grid + reconcile, keyed by run_name) and
`results/revision/sedd_capability_summary.json` (SEDD gates/linearization/recovery/hybrid/guided).

| Thesis object | Producing script | Result file |
|---|---|---|
| Table `tab:full-grid` (145-config grid) | `scripts/run_experiment.py` (grid) | `results/grid/gpt2_v2`, `results/grid/llama`, `results/grid/gfn`; `numbers.json` |
| Table `tab:fallacy` (concern 1 paired stats) | `revision/analyze_stats.py` | `results/revision/rev_stats_gpt2.json` |
| Table `tab:baselines` (KL baselines + Gibbs) | `run_revision.py --exp kl_baselines` | `results/revision/rev_klbase_gpt2sft.json` |
| Table `tab:gfn-unify` (GFlowNet variants) | `run_revision.py --exp kl_baselines` (gfn) | `results/revision/rev_klbase_gfn-*.json` |
| Table `tab:divergence` (base vs tuned energy) | `run_revision.py --exp model_divergence` | `results/revision/rev_divergence_*.json` |
| Table `tab:constrained` (constraint contrast) | `revision/analyze_constrained.py` | `results/revision/rev_constrained.json` |
| Table `tab:lasttoken` (position conditions) | `run_revision.py --exp last_token` | `results/revision/rev_last_token_gpt2sft.json` |
| Table `tab:diffusion-lin` (SEDD linearization) | `run_sedd_linearization.py` / `run_sedd_cap.py` | `rev_sedd_lin_{small,medium}.json`; `sedd_capability_summary.json` |
| Table `tab:diffusion-recovery` (SEDD recovery) | `run_sedd_cap.py --exp recovery` | `rev_sedd_recovery_{small,medium}.json` |
| Table `tab:diffusion-hybrid` (hybrid sufficiency) | `run_sedd_cap.py --exp hybrid` | `rev_sedd_hybrid.json` |
| Table `tab:diffusion-guided` (guided steering) | `run_sedd_guided.py` (Phase 4) + `run_gprime.py` (Phase 5) | `rev_sedd_guided_g{1,2,4}.json`; `rev_gprime.json` |
| Table `tab:cost` (compute cost) | hand-derived + wall-clocks | `rev_continuation`, `rev_klbase`, `rev_judge` JSONs |
| Figs `fig:lin-radius/scatter/decomp/topk` | `run_diagnostic.py --exp linearization` -> `plot_diagnostics.py` | `results/diagnostics/{diag,diagnosis}/diag_linearization_*` |
| Figs `fig:mh-accept`, `fig:mh-decomp` | `collect_traces.py` -> `analyze_mh.py` | `results/diagnostics/diag/traces_gpt2sft_mh.csv` |
| Figs `fig:trap`, `fig:trap-length` | `run_diagnostic.py --exp likelihood_trap` -> `plot_diagnostics.py` | `results/diagnostics/{diag,diagnosis}/diag_ltrap_*` |
| Fig `fig:aniso` | `run_diagnostic.py --exp anisotropy` -> `plot_diagnostics.py` | `results/diagnostics/diag/diag_anisotropy_*` |
| Figs `fig:traj-dls-pca`, `fig:traj-cls-pca` (Phase 5) | `revision/plot_trajectories.py` | `figures/fig_traj_pca_{dls,cls}.png` from `traces_gpt2sft_plot_traj.npz` |
| Figs `fig:dls-traj-50/100` | `scripts/run_experiment.py` / `scripts/replot.py` | `figures/gpt2-large.dls.gn.free.s{50,100}_new_trajectories.png` |
| Fig `fig:lasttoken` | `run_revision.py --exp last_token` | `results/revision/last_token_figure.png` |
| Table `tab:proposal-sharpness` (entropy, t1/t2) | `scripts/run_experiment.py --log_proposal_stats` | `results/grid/smoke/flagship_stats.json` |
| Table `tab:chainstats` + Fig `fig:forest` | `revision/analyze_chain_stats.py` | `results/revision/rev_chain_stats.json` |
| n=1000 certified equivalence (5.5) | `revision/analyze_power.py` | `results/revision/rev_power.json` |
| Table `tab:onehot` (one-hot surrogate) | `revision/analyze_onehot_surrogate.py` | `results/revision/rev_onehot_surrogate.json` |
| Tables `tab:mhfix`, `tab:onehot-sweep` (sweeps) | `scripts/gen_manifest_rev3.py` + `revision/analyze_rev3.py` | `results/grid/rev3`; `results/revision/rev_rev3_summary.json` |
| Table `tab:mlm` (bidirectional MLM control) | `diagnostics/run_mlm_control.py` | `results/revision/rev_mlm_control.json`, `rev_mlm_uniform.json` |
| Quenching withdrawal (5.2) | `scripts/run_experiment.py` re-run | `results/grid/verify/` (bit-identical to `results/grid/gpt2_v2`) |
| Appendix showcase (Phase 5) | `build_showcase.py` + `make_showcase_tex.py` | `results/revision/qualitative_showcase.json`, `showcase_appendix.tex` |

## Known caveats

- **SEDD runs are excluded from the AR reconcile globs.** `reconcile_numbers.py` counts
  and diffs only the autoregressive grid; SEDD (`rev_sedd_*`) lives outside those globs by
  design. Do not fold SEDD run_names into the AR config count (145 = 5 x 29).
- **The `gn=on` bitwise `gradnorm == random` artifact.** With grad-normalization on, the
  grad-norm-preserved-random-dir arm and the random arm are bitwise identical in several
  measurements (e.g. judge ppl 181.32 for both; continuation 8.850 for both). This is
  expected: normalizing the gradient magnitude and then substituting a random direction of
  the same norm collapses the two arms. Report them as one where this holds.
- **CLS acceptance attribution (corrected 2026-07-26, Part 1 Alarm 1).** The two CLS
  acceptance pairs are BOTH MH-on; they differ by gradient normalization, not by the MH
  switch. `reconcile_numbers.py:115` groups by `["sampler","method","grad_norm"]`, so the
  boolean in `rev_reconcile.json`'s `('cls','policy',False/True)` keys is `grad_norm`:
  grad-norm off (`cls_policy_gnoff_mh`) is 0.034% within / 3.665% boundary, grad-norm on
  (`cls_policy_gnon_mh`) is 0.627% / 8.557%. There is no MH-off acceptance rate in the
  trace file, because `collect_traces.py:259` logs acceptance only when MH is on (under
  MH-off every proposal is accepted, rate 1.0). The DLS-MH within-cell / boundary contrast
  is 100% / 9.3%. Section 5.3 now attributes the two numbers to the grad-norm split under
  the correction, and `fig_mh_accept` is regenerated for the single config
  `cls_policy_gnoff_mh`.
- **Trajectory figures use a clean 5-config regeneration** (`traces_gpt2sft_plot`), not the
  canonical `traces_gpt2sft` npz, because the torch RNG carries across configs (so a single
  config cannot be spliced at the canonical RNG state) and the figures were placeholders.
  CLS states escape >100 units off the token manifold (MH-off to a max of ~979); the PCA
  panels clip extreme excursions so the anisotropy cone stays visible.
- **`results/diagnostics/{diag,diagnosis}` are a near-duplicate pair, kept distinct.** The
  old `results_diag` (traces, mh csv, the full linearization / likelihood-trap runs) became
  `results/diagnostics/diag`; the old `results_diagnosis` (a smaller diagnostics set, holds
  a pre-existing 0-byte `diag_likelihood_trap_llama3-8b.csv`) became
  `results/diagnostics/diagnosis`. They are NOT merged because they may hold different runs;
  point each analysis at the one it needs (traces / mh come from `diag`).

## Repository layout note

The repository was reorganized on 2026-07-25 (git-mv-only; nothing deleted). The full
old-to-new move map and the nothing-deleted certification are in
`REVISION_RESTRUCTURING.md`. The pre-reorg commit `4bd1ee4` is the rollback point
(`git reset --hard 4bd1ee4`). Code imports resolve from the repo root with CWD there and
`source env.sh`; the entry runners moved to `scripts/` and carry a repo-root `sys.path`
shim so `import core` works from the new location.
