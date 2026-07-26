# REVISION_WRITING.md

Timestamped, append-only log of the final documentation-and-writing pass
(PROMPT_PHASE9_FINAL_DOCS, Parts 0 through 7). Beamer work is logged separately
in PRESENTATION_LOG.md.

Standing invariants: numbers diff at the end against numbers.json plus every
phase JSON; IMS compliance untouched; Ioanna register; no reorg beyond what
REVISION_RESTRUCTURING.md already did; no em-dashes; removed text goes to `%`
comment blocks; every action logged here.

The file ends with a "WHAT CHANGED" summary the author can read instead of the
full log.

---

## 2026-07-26 00:28 CEST  Orientation (read before touching anything)

Read in the prescribed order: REVISION_RESTRUCTURING.md (new layout confirmed:
thesis at `Doc/final/thesis/`, proposal at `Doc/final/proposal/`, beamer at
`Doc/final/beamer/`, refs in `refs/`, meetings in `meetings/`, archive in
`archive/`), CLAUDE.md, the closing reports at the tail of REVISION_LOG.md, and
`refs/evaluation1.txt`, `refs/evaluation2.txt`, `refs/evaluationproposal.txt`,
`thesis_questions_knowledge_base.md`. Confirmed the thesis chapters live at
`Doc/final/thesis/chapters/` (01..08 plus 05a, abstract, showcase, gprime,
tab_confusion), master `Doc/final/thesis/thesis.tex`, bib
`Doc/final/thesis/references.bib`, figures in `Doc/figures/`.

---

## PART 1: the three numeric alarms (traced to source before any prose work)

Full provenance for each. Source data was inspected directly; nothing guessed.

### ALARM 2 (Table 12 / tab:full-grid) -- RESOLVED, unambiguous data-transcription error

Claim: the appendix full-grid table shows identical final KL for "CLS policy, MH,
no-gn, s50" and "CLS policy, no-MH, no-gn, s50" (both `8.083 / 7.799 / 8.198 /
13.732`), which Section 5.3 makes implausible.

Trace:
- Table source is `figures/compare/final_kl_by_model.csv` (the file the caption
  names). That CSV is CORRECT and has DISTINCT rows:
  - `cls.policy.mh.nogn.free.s50`   -> `9.202 / 8.834 / 9.748 / 9.025` (gpt2-large, lb0-500, lb0-2000, lb1-500)
  - `cls.policy.nomh.nogn.free.s50` -> `8.083 / 7.799 / 8.198 / 13.732`
- Cross-checked against the raw grid JSONs (`results/grid/gpt2_v2/`): final KL is
  the last element of `mean_kl` (50-step curve).
  `gpt2-large.cls.policy.mh.nogn.free.s50.json` mean_kl[-1] = 9.20179;
  `gpt2-large.cls.policy.nomh.nogn.free.s50.json` mean_kl[-1] = 8.08334. Distinct.
- Verified all seven other rows of tab:full-grid against the CSV: every one is
  correct. Only the "CLS policy, MH, no-gn, s50" row is wrong; it duplicates the
  adjacent no-MH row's values.

Root cause: LaTeX table transcription error (the MH row was filled with the
no-MH row's numbers). The source CSV and JSONs are fine.

Resolution (unambiguous): replace the "CLS policy, MH, no-gn, s50" row with
`9.202 & 8.834 & 9.748 & 9.025`. `numbers.json` holds only the correct no-MH
value (`gpt2-large.cls.policy.nomh.nogn.free.s50/final_kl = 8.083`), so no
numbers.json value is wrong; will ADD the MH value for traceability.

### ALARM 3 (Figure 12 / fig:trap-length vs the -1.12 nats/token headline) -- RESOLVED: -1.12 is REAL, figure is the artifact

Claim: the figure prints slope -0.11, r -0.00, while 5.7/5.10 report -1.12
nats/token as a load-bearing number.

Trace:
- -1.12 provenance: GPT-2 Large (SFT), the likelihood-trap generation set
  (500 sequences x 6 decoders), regression of `total_logp` on generated length,
  pooled over the THREE STOCHASTIC decoders (ancestral, topp90, temp07; n=1500)
  where length varies. Source:
  `results/diagnostics/diag/diag_likelihood_trap_gpt2sft.json ->
  length_vs_total_logp.pooled_sampling.slope_nats_per_token = -1.1234`. The JSON
  note states deterministic decoders pinned at the length cap are excluded
  because their length carries no information.
- The figure's -0.11 is `plot_diagnostics.py:354`, `linregress(df.gen_len,
  df.total_logp)` over the ENTIRE csv (all 6 decoders, 3000 rows). Reproduced:
  full-CSV slope = -0.1059, r = -0.0036, n = 3000; frac at the 40-token cap =
  0.9973. The slope is washed out because 99.7 percent of generations are pinned
  at the cap, so length barely varies in the pooled data. The stochastic-only
  regression reproduces the negative slope (-1.20 over gen_len, close to the
  JSON's -1.1234; the small gap is scored-length vs generated-length bookkeeping
  inside run_diagnostic).
- Conclusion: -1.12 is REAL and correctly sourced. It is NOT a digit
  transposition of -0.11. The 5.10 length-collapse argument stands.

Resolution: regenerate `fig_trap_length` so its fitted line and annotation match
the regression the text describes (stochastic decoders where length varies,
slope = -1.12), and state the configuration and the censoring caveat in the
caption. Text and 5.10 argument unchanged.

### ALARM 1 (Figure 9 / fig:mh-accept) -- RESOLVED, and it surfaced a deeper TEXT error

Claim: the figure shows within-cell 0.064 and boundary 0.074 acceptance, while
5.3 reports 0.63 percent within-cell and 8.56 percent boundary with MH on.

Trace of the figure:
- `fig_mh_accept` is produced by `analyze_mh.py` from
  `results/diagnostics/diag/traces_gpt2sft_mh.csv`. That CSV pools THREE configs:
  `cls_policy_gnoff_mh`, `cls_policy_gnon_mh`, `dls_policy_gn_mh`. Run over the
  whole CSV, `stayed.accepted.mean() = 0.0639`, `crossed.accepted.mean() =
  0.0744` -- exactly the 0.064 / 0.074 the examiner saw. So the figure aggregates
  all three configurations rather than the single one the text discusses.

Trace of the 5.3 text (the deeper finding):
- Per-config split of the same CSV:
  - `cls_policy_gnoff_mh` (grad-norm OFF, MH ON): within-cell 0.0003 (0.03%), boundary 0.0367 (3.67%)
  - `cls_policy_gnon_mh`  (grad-norm ON,  MH ON): within-cell 0.0063 (0.63%), boundary 0.0856 (8.56%)
  - `dls_policy_gn_mh`    (MH ON):                within-cell 1.0000 (100%),  boundary 0.0927 (9.27%)
- The 5.3 text and its SOURCE comment call the first pair "without the
  correction" (MH off) and the second "with the correction" (MH on). That is
  WRONG. Both configs are MH ON; they differ in gradient normalization. The
  reconcile (`reconcile_numbers.py:115`) groups by
  `["sampler","method","grad_norm"]` and the third boolean is `grad_norm`, not
  MH; the thesis text read that boolean as MH. `rev_reconcile.json`'s
  `('cls','policy',False)` (n 2906/7094) is `cls_policy_gnoff_mh` and
  `('cls','policy',True)` (n 1913/8087) is `cls_policy_gnon_mh`, confirmed by
  matching row counts.
- There is NO MH-off acceptance data by construction: `collect_traces.py:259`
  logs the acceptance record only when MH is on (`mh_log = [] if mh else None`),
  because under MH-off every proposal is accepted (rate 1.0). So an "MH off vs MH
  on acceptance rate" contrast cannot be read off this file at all; the two
  numbers are the grad-norm off/on split, both MH on.
- The section's load-bearing evidence is UNAFFECTED and correct: DLS within-cell
  100% vs CLS within-cell ~0 (the contrast the argument actually rests on), the
  target vs proposal decomposition (+4.60 vs -1325), and the non-Lipschitz
  reverse-proposal argument. Only the one sentence's MH-off/MH-on labeling is
  wrong.

Resolution proposed (see checkpoint to author): (a) rewrite the 5.3 sentence so
the two numbers are attributed truthfully -- with the correction enabled the CLS
sampler accepts within-cell moves 0.03 to 0.63 percent of the time (grad-norm
off to on) and boundary crossings 3.7 to 8.6 percent, against the discrete
sampler's 100 percent within-cell; and, if kept, the MH-off case is the
accept-everything regime (rate 1.0) that drives the off-manifold wandering. Fix
the SOURCE comment. (b) Regenerate `fig_mh_accept` restricted to the single
stated configuration (`cls_policy_gnoff_mh`, the paralysed sampler the argument
needs), with the caption stating the configuration exactly.

This is a claim-level correction the author did not previously flag, so it is
being surfaced for confirmation before the 5.3 rewrite is applied.

### PART 1 IMPLEMENTED (2026-07-26 ~12:15 CEST)

Author decision on the 5.3 rewrite: "Keep MH contrast, corrected" (Option A).

Actions taken:
1. ALARM 2. `Doc/final/thesis/chapters/08_appendix.tex` tab:full-grid: the
   "CLS policy, MH, no-gn, s50" row changed from `8.083 & 7.799 & 8.198 & 13.732`
   (a duplicate of the no-MH row) to `9.202 & 8.834 & 9.748 & 9.025`, with a
   provenance `%` comment. numbers.json ALREADY held these correct values
   (`gpt2-large.cls.policy.mh.nogn.free.s50/final_kl = 9.20179`, and the three GFN
   variants 8.834/9.748/9.025), so the fix brings the thesis INTO agreement with
   numbers.json; no numbers.json edit was needed and the final numbers-diff now
   passes where it would previously have flagged the 8.083 duplicate.
2. ALARM 3. `diagnostics/plot_diagnostics.py` 4B block rewritten to fit and
   annotate the pooled-sampling regression the text cites (stochastic decoders,
   scored_len, length varies), reproduced exactly as slope -1.1234, r -0.0776,
   n 1500. Figure regenerated. Caption in 08_appendix.tex updated to state the fit
   set and the censoring caveat. Text and the 5.10 length-collapse argument are
   unchanged (they were already correct).
3. ALARM 1. `diagnostics/analyze_mh.py` gained a `--config` filter.
   `fig_mh_accept` regenerated for the single config `cls_policy_gnoff_mh`
   (within-cell 0.0003, boundary 0.0367), caption now states the configuration
   exactly. `Doc/final/thesis/chapters/05_results.tex` 5.3 sentence rewritten per
   Option A: the two CLS numbers are now attributed to grad-norm off/on under the
   correction (both MH on), and the MH-off case is described as the
   accept-everything regime that drives the off-manifold wandering
   (Appendix~\ref{app:trajectory}). SOURCE comment corrected to record the
   grad-norm-not-MH grouping.
4. BUILD PATH FIX (restructuring breakage, Part 0 scope). The final master
   `Doc/final/thesis/thesis.tex` could not resolve any figure: the restructuring
   moved the master from `Doc/final/thesis.tex` to `Doc/final/thesis/thesis.tex`
   (one level deeper) without updating `\graphicspath`, so `{../figures/}` no
   longer reached `Doc/figures/`. Fixed to `{../../}{../}{../figures/}{./}`
   (`includegraphics` carries a `figures/` prefix, so `../../` from
   `Doc/final/thesis/` resolves `../../figures/...` = `Doc/figures/...`).

Figures were regenerated into a temp dir; only `fig_trap_length` and
`fig_mh_accept` (.pdf and .png) were copied into `Doc/figures/` and root
`figures/`. No other figure was disturbed.

Verification: `pdflatex` on the final master exits 0, 0 "not found" figures
(was 27), 111 pages with figures embedded (2.98 MB). `pdftotext` confirms the
corrected `9.202` is present and `8.083` remains once (the no-MH row).

PART 1 COMPLETE. All three alarms resolved with full provenance; one additional
claim-level error (the 5.3 grad-norm/MH mislabel) found and corrected under the
author's Option A; the final build's figure path repaired.

## PART 0 COMPLETE (2026-07-26 ~12:40 CEST)

Documentation sync to the final tree:
- README.md: compile pointer now `Doc/final/thesis/` (`latexmk -pdf thesis.tex`, figures
  from `Doc/figures/`), with proposal/beamer locations; "Repository layout" Doc block
  rewritten to `Doc/final/{thesis,proposal,beamer}/` + `Doc/figures/` + `Doc/prev_version/`;
  the "Concern 6a attribution" caveat corrected to state the grad-norm (not MH) grouping
  and the no-MH-has-no-acceptance-rate fact (matches the Part 1 Alarm 1 fix).
- REVISION_RESTRUCTURING.md: addendum documenting the final Doc tree, the master moving
  one level deeper than PART 1.4 recorded, and the graphicspath break + fix.
- Tex `%` source comments swept in the final chapters: `figures_gpt2/`->`figures/gpt2/`
  (3x in 05_results), `figures_compare/`->`figures/compare/` (1x in 08_appendix). The
  appendix config-count prose now names `results/grid/{gpt2_v2,llama,gfn}`.
- graphicspath fix already logged under Part 1 (it was the build-gate blocker).

Residual old-token sweep of the final chapters: 0 stale result/figure path tokens remain.

---

## PART 2: author issue list

### Item 1 (title-page gap + abstract width, third strike) -- DONE, PNG proof

Root cause: the template/article build's titlepage had added `top=2cm,bottom=2cm` to the
titlepage `\newgeometry`, enlarging the text height so the template's `\vfill` opened a
several-centimetre void between the author name and the examiner tabular (the reported
gap); the geometry restore used `\newgeometry{left=3cm,right=3cm}` (the "unrestored" case).
Fix: titlepage now matches the official ThesisExample template VERBATIM
(`\newgeometry{left=3.5cm,right=2cm}`, `\vspace{2.5cm}` / `\vspace{2cm}`, `\vfill`, the
template's blank tabular rows); the body geometry is restored with `\restoregeometry`
(canonical undo, so the abstract uses the exact body text block). Only field values differ
from the template (real title, name, examiners, dates).
PROOF: `Doc/final/proofs/titlepage_vs_template.png` (our page left, template right) shows
the examiner block now in the template's balanced mid-lower position; the void is gone.
Abstract width confirmed unchanged at 3cm margins (`Doc/final/proofs/abstract_page.png`);
its one-page-limit reduction is Part 3 item 8, handled there.

### Items 2-9 -- DONE

- Item 2 (RQ bridge): 1.4 now has a bridge paragraph after the four RQs mapping proposal
  RQ1+RQ1a->RQ1, RQ1b->RQ2, RQ3a->RQ3, RQ2a->RQ4, and RQ2b (energy-based recovery vs AR
  decoding) to the gradient-free baselines (Section~\ref{sec:results-baselines}). 6.1 gains
  one sentence resolving RQ2b under its proposal name: negative for the gradient-guided
  samplers, affirmative for exact-energy rescoring (top-k rerank, final KL 4.43).
- Item 3 (Contribution 2 split): the linearization-failure contribution is now three short
  sentences.
- Item 4 (background diffusion gap): new Chapter 2 subsection "Discrete Diffusion and the
  Concrete Score" (sec:bg-discrete-diffusion, ~2/3 page): concrete score as ratios
  p(y)/p(x) [meng2022concrete], absorbing discrete diffusion [austin2021structured], SEDD
  as trained ratios [lou2024discrete]. Two bib entries added (Austin D3PM 2021, Meng
  concrete-score 2022), full author lists.
- Item 5 (marble analogy): cut to one intuition sentence plus the corrected annealing
  caveat; removed sentences preserved in a % block.
- Item 6 (Roberts-Tweedie): the end-of-Langevin paragraph replaced with the corrected
  formal wording (discretization error grows with step size; MALA acceptance degrades as
  smoothness is lost; non-asymptotic guarantees assume smoothness/log-concavity absent
  here), citing roberts1996exponential + dalalyan2017theoretical + durmus2017nonasymptotic
  (both added to the bib, complete entries). One results sentence added in 5.3 linking the
  measured near-total boundary rejection to that excluded regime.
- Item 7 (continuous-sampler conclusion exactness): the same end-of-Langevin sentence now
  says the drift is computed from the differentiable pathway whose target changes across
  cell boundaries (not a gradient of the piecewise-constant energy), agreeing with the
  two-objects passage in 2.4.
- Item 8 (Taylor surrogate): the two terms of Eq (7) are now named explicitly (the
  gradient-alignment term, the distance term), ordered to match the equation, with a
  one-line derivation footnote showing the discarded constant term -(alpha/8)||g||^2.
- Item 9 (CLS update equations): methodology now documents the implemented step, the
  interim continuous update, the nearest-neighbour projection, and the interpolated
  proposal mean s_{t+1} ~ N((s_interim + proj(s_interim))/2, eps I) (new eq
  meth-cls-update), with the drift-bounding reason and a note that it is an implementation
  choice adapted from kumar2022gradient, load-bearing for Section 5.4.

Build gate after items 1-9: latexmk exit 0; a clean pdflatex pass reports 0 undefined
citations and 0 undefined references; the four new citations resolve; 121 pages.

### Items 10-31 -- DONE (prose + 2 figures); items 21, 23 partially deferred

Prose/edit items applied:
- 10 GFlowNet metaphor compressed to one sentence (2.6).
- 11 the two "final" section openers de-duplicated (3.4 no longer "final").
- 12 metric clarification added to 4.1 (relative-to-reference contextual fit vs absolute
  likelihood, not in tension with the likelihood trap); the duplicated exact-match
  justification now lives once in 4.4.
- 13 A.2 (Full Configuration Grid) reframed with a released-artifact-map paragraph tied to
  the Section 4.7 code-availability statement, appendix only. NOTE: "A2" was interpreted as
  appendix A.2 (the config grid, which evaluation2 praised); flagged for author confirmation
  in case a different section was meant.
- 14 4.5 gains the why-likelihood-as-reward paragraph (gradient-uninformative is not
  value-uninformative; the trap concerns the extreme not the fluent-manifold ranking;
  matching the base distribution is the Hu et al. amortization premise under test).
- 15 4.2 gains the config-matrix sentence (GFlowNet variants included to test whether
  amortization repairs the landscape).
- 16 4.5 gains the task-comparability paragraph (GFlowNet reformulates infilling as
  left-to-right generation; samplers do in-place bidirectional substitution; compared at the
  energy level, not the task; left-to-right not adopted because it would dissolve the
  revision capability).
- 17 related work states the CLS implements the COLD/MuCoLa mechanism faithfully, that the
  constrained mucola arm is that comparison run at the energy level (task-level original-code
  comparisons out of scope), and the thread-D one-sentence answer (success without an exact
  correction is early-stopped biased optimization plus post-hoc filtering). Results mention
  is the existing MuCoLa-arm contrast in 5.11.
- 18 results-opening provenance sentence removed (content in 4.7 + README).
- 19 5.1 "qualifies the plug-and-play framing" replaced with the plain step-size-vs-embedding-
  distance statement.
- 20 quenching metallurgy origin compressed to a clause at first use.
- 24 5.1 states the alpha grid (fifty values, logspace 1e-2..1e2) and that 10.5->0.1 is the
  oracle-calibrated schedule.
- 25 5.5 sentence surgery: norm-preserved rationale split into two sentences; bounded-
  equivalence paragraph compressed to the three-clause form; Llama anti-guidance paragraph
  rewritten as short declaratives, with the sourced CI [0.45, 2.09] (p 0.015, policy worse)
  from rev_stats_llama.json (dls nomh nogn).
- 26 5.5.3 states the judge generating config (DLS, MH on, gn on, 50 steps, three arms) and
  the judge code path was re-verified end to end (run_external_judge.py generate+judge; the
  in_model_kl 6.541/6.370 match the flagship grid; perplexity = exp(nll/ntok); the gnp=random
  identity is the gn-on artifact). The 178.4-vs-181.3 closeness is real and correctly computed.
- 27 5.6: the 2.35-vs-1.82 and "first order does not survive the jump" sentences rewritten in
  plain form; 15.0/24.2 clarified as mean absolute log-likelihood changes in nats.
- 29 5.8: scale and anisotropy separated explicitly (scale -> step-size failure; anisotropy ->
  Euclidean unreliability), with a clause reconciling the pairwise cosine with the low-variance
  PCA note of Appendix A.4.
- 30 5.7 and 5.10: degenerate strings attributed to autoregressive argmax/beam decoding of the
  model, not to any Langevin sampler, and the GFlowNet reward is noted as up-weighting them.
- 31 5.11: the infilled tokens in the example are bolded against the original, and a
  drift-mechanism sentence (small unconditional negative drift dominates raw gains; the paired
  contrast cancels it) is placed before the numbers.

Figures regenerated (no GPU; from existing result CSVs, into Doc/figures + figures):
- 22 fig_mh_decomposition: single config cls_policy_gnoff_mh, translucent fill plus step
  outline with zorder so the two distributions no longer blend.
- 28 fig_trap_scatter: six clearly distinct decoder colours.

DEFERRED / FLAGGED:
- 21 (Figure 1 quenching): DONE -- gn configuration now stated in the text near the figure,
  and the stale figures_gpt2 source-path comment corrected. REMAINING (no-GPU figure work in
  revision/plot_dls_trajectories.py): the legend dashed-vs-solid distinguishability fix and
  the gn-off companion panel (sourced from the existing gpt2-large.dls.*.nogn.free.s50 grid
  CSVs, where the three proposal arms separate). Deferred to the figure-regeneration pass.
- 23 (5.4 DLS boundary-crossing): DONE -- the DLS config (dls_policy_gn_mh, MH on) is now
  stated in 5.3. BLOCKED: the no-MH DLS trajectory/distinct-cells number and the
  fig_traj_stats.json regeneration to include a no-MH DLS config require a collect_traces GPU
  run (the trajectory npz contains no DLS no-MH config, and this number exists nowhere in the
  data). Per CLAUDE.md this single-experiment GPU job needs explicit author go-ahead; flagged.

Build gate after Part 2: pdflatex exit 0, 123 pages, 0 errors, max overfull 7.49pt (< 40pt
gate). Clean pass earlier confirmed 0 undefined citations/references.

### Items 21, 23 COMPLETED; PART 2 COMPLETE (all 31)

- Item 23: author-approved single GPU run executed. Added config dls_policy_gn_nomh to
  collect_traces.py; ran it on one A6000 (6 trajectory sequences x 50 steps, GPT-2 Large,
  no-MH). Result: DLS with the correction visits 5.33 of 50 cells, without it 48.67 (accepts
  almost every proposal) while staying on-manifold. Added the entry to fig_traj_stats.json
  (per_config.dls_policy_gn_nomh) and the with/without-correction DLS contrast to Appendix
  A.4. Audit confirmed the datum existed nowhere before the run.
- Item 21: revision/plot_dls_trajectories.py gained a gn parameter; the legend is rebuilt
  from custom handles (method = colour+marker; correction = marker-free grey handles, solid
  vs a long dash pattern) and the "without correction" plot lines use the same long dash; a
  gradient-normalization-disabled companion figure (fig:dls-traj-nogn) was generated from the
  existing gpt2-large.dls.*.nogn.free.s50 grid CSVs, in which the three proposal arms
  separate, and integrated into Section 5.2 with a caption and text reference. The stale
  figures_gpt2 source-path comment was already corrected in Part 0.

PART 2 COMPLETE: all 31 author issue-list items applied. Build gate: latexmk exit 0, 0
missing figures, 0 undefined citations/references, 124 pages.

## PART 3: evaluation2 resolutions

### Items done so far: 3, 5, 7, 9
- 3 (four-vs-five diagnostics): methodology 4.1 opener and 4.7 now say "five diagnostic
  experiments" (linearization, acceptance, trajectory, likelihood-trap, anisotropy),
  agreeing with the conclusion.
- 5 (bibliography author lists): Bengio 2023 (6), Hu 2022 LoRA (8), and Brown 2020 GPT-3 (31)
  given full author lists; Dubey 2024 (Llama 3 herd, several hundred authors) kept as
  first-author "and others" with a documented policy note (the single large-collaboration
  exception).
- 7 (Gibbs "comparable to"): 5.5.2 and 6.4 now say the Gibbs 6.69 is comparable to (not
  matching) the best gradient-guided Langevin result near 6.4-6.5.
- 9 (1.2 reveal clause): 1.2 now states the outcome is given before the RQs because the
  thesis is organized as an explanation, not a reveal.

### Remaining: 1 (corpus map), 2 (5.11 rigor table), 4 (proposal delta 6.4), 6
### (straight-through mention), 8 (abstract one-page + numeral + closing sentence + proof),
### 10 (Figure 13 token strips).

### Item 1 (corpus map) -- STOPPED: significant corpus discrepancy found, author decision needed

Building the corpus map required verifying which corpus each experiment family used. The
data contradicts the thesis text on a load-bearing point:

EVIDENCE (from result files, not guessed):
- Main sampling grid (scripts/run_experiment.py): grid JSON config data_file=None -> the
  run_experiment default is WikiText-2 validation (run_experiment.py:79-86). The recovered
  example texts confirm it: "The rostrum of H. americanus bears one or more spines ...",
  "red @-@ tipped" -- the "@-@" token and the encyclopedic lobster-anatomy content are the
  unmistakable WikiText-2 signature. This is NOT ROCStories.
- Gradient-free baselines (rev_klbase) + free-form continuation (rev_continuation) + external
  judge: run_revision default, WikiText-2; showcase_appendix.tex states "kl-baselines set
  (WikiText-2 validation)".
- Constrained generation: the "rostrum of H. americanus" example is WikiText-2.
- SEDD diffusion linearization/recovery/hybrid (5.13): "unified WikiText-2 sequence set".
- ROCStories was used for: the GPT-2 SFT (4.1), the GFlowNet training and variants
  (hu2024amortizing reference impl), and the run_diagnostic diagnostics
  (diag_linearization/likelihood_trap/anisotropy/trajectory, argv dataset=roc_stories), plus
  the diffusion "second corpus" (rev_sedd_linearization.json).

THE CONTRADICTION: Section 4.1 states "The corpus is ROCStories" for the masked-recovery
benchmark and gives three reasons for choosing it, and the abstract says the recovery
"benchmark built from ROCStories". But the main grid, the baselines, the constrained
experiment, and the continuation task all ran on WikiText-2. The model is SFT'd on
ROCStories, so the main grid is actually OUT-OF-DOMAIN WikiText-2. Evaluation2 flagged the
bookkeeping as muddled and knew the diagnostics were WikiText-2, but believed 4.1's
ROCStories claim for the main task; the data shows the main task is WikiText-2 too.

This also means the 5.13.1 SOURCE comment citing "diag_linearization_gpt2sft.json (AR)" for
the "unified WikiText-2" comparison is mis-cited (that JSON is ROCStories); the AR row on the
WikiText set comes from a different run.

Item 1's requested "domain-shift" sentence (in-domain ROCStories diagnostics agree, so the
null is not a domain artifact) fits this reality exactly, which suggests the split is by
design. But correcting 4.1 and the abstract from "ROCStories" to "WikiText-2 (out of domain
relative to the ROCStories SFT)" is a significant claim change, and 4.1's three reasons for
ROCStories would need rethinking. Presented to the author for a decision rather than guessed.

### PART 3 items 1, 2, 8, 10 -- DONE (item 10 figure regenerating)

- Item 1 (corpus map): AUTHOR-APPROVED correction ("out-of-domain design"). 4.1 now states the
  masked-recovery sequences are WikiText-2 validation and the base is fine-tuned on
  ROCStories, with a corpus-map table (tab:corpus-map) and a domain-shift paragraph (the
  in-domain ROCStories diagnostics reproduce the null and the trap). The abstract's "built
  from ROCStories" corrected to WikiText-2. WikiText-2 citation (merity2017pointer) added.
- Item 2 (5.11 rigor): new table tab:constrained-contrast tabulating the paired
  cons_only-minus-cons_random contrasts (+27.3, +36.7 MuCoLa continuation; ~0 for the DLS
  "ours" setup), n=150/arm, with an explicit note that the result files store only aggregate
  target-hit fractions so per-sample gains and CIs are unavailable (none invented). 4.6 states
  n and the analysis plan.
- Item 6 (straight-through): related work gains a bounded mention of straight-through
  [bengio2013estimating] and Gumbel-softmax [jang2017categorical] relaxations as the third
  adaptation strategy and why out of scope; both bib entries added (full author lists).
- Item 8 (abstract one page): abstract compressed (factorial clause, collapse modes, tens of
  nats dropped; paragraphs merged), numerals unified to digits (145, 0 to 39%), a closing
  practical-implication sentence added (evaluate, do not differentiate), and a 1.12
  line-spacing group applied so it fits ONE page. Verified: page 3 holds the whole abstract,
  page 4 is the ToC. Proof: Doc/final/proofs/abstract_one_page.png. No Kurzfassung exists.
- Item 10 (Figure 13 token strips): revision/plot_trajectories.py token_strip font enlarged
  6pt -> 8.5pt, label cap 15 -> 10 so the larger labels do not overlap; regeneration of
  fig_traj_distance in progress (re-merge of the dls_policy_gn_nomh entry into
  fig_traj_stats.json to follow, since plot_trajectories rewrites that file).

Items 3, 4, 5, 7, 9 were logged earlier. PART 3 substantively complete (10/10); item 10
figure regeneration and the fig_traj_stats.json re-merge are the only mechanical steps left.

### Item 10 finalized; PART 3 COMPLETE (all 10)

fig_traj_distance regenerated with the enlarged (8.5pt) token strips, visually confirmed
legible; the dls_policy_gn_nomh entry re-merged into fig_traj_stats.json after the plot
script rewrote it (5 configs present); fig_traj_distance + fig_traj_pca copied into
Doc/figures. Build gate: latexmk exit 0, 0 undefined citations/references, the three new
citations (merity2017pointer, bengio2013estimating, jang2017categorical) and the two new
tables (tab:corpus-map, tab:constrained-contrast) and the companion figure resolve, 125 pages.

PARTS 0, 1, 2, 3 COMPLETE. Remaining: Part 4 (evaluation1 trims), Part 5 (proposal
amendment), Part 6 (beamer defense talk), Part 7 (final gates + resolution tables + WHAT
CHANGED summary).

## PART 4: evaluation1 trims -- DONE

- Verbatim central formulation reduced from 7 occurrences to exactly 3 (abstract, 5.5, 6.1);
  the other four (1.2, contributions, 6.2 shorthand-scoping, conclusion) are paraphrased with
  the scope preserved and a pointer to 5.5 for the exact wording.
- Evaluate-vs-differentiate: the premature related-work verdict (3.2) is trimmed to a
  forward-pointer to Sections 5.5.2 and 6.3, where the evidence lives; the canonical statement
  stays in 6.3 and the abstract's practical-implication closer is the callback.
- Related-work interpretive verdicts flagged by evaluation1 (3.2 "clean separation", 3.3
  "coherent-text optimum" axis) reduced to positioning + forward-pointers, contributing to the
  related-work trim.
- Introduction opening two paragraphs compressed (the audience knows autoregressive
  generation), per evaluation1 section 5.1.
- Already satisfied by earlier passes: the Langevin marble analogy cut (Part 2 item 5), the
  DLS Taylor-term "sole signal" stated once (Phase 8), the abstract compression (Part 3 item
  8), the 5.13.4 classifier-alignment summary at one page and the diffusion three-result
  summary at one sentence each in 6.1 and the conclusion (Phase 8). The conclusion body is
  already two pages (evaluation1's "eight pages" measured conclusion plus the five-page
  bibliography), so no further conclusion cut was made.

Build gate: pdflatex exit 0, verbatim central formulation count = 3, 125 pages.

## PART 5: proposal amendment -- DONE

Applied EXACTLY Part 5's slate to Doc/final/proposal/proposal.tex (git diff vs HEAD:
28 insertions, 14 deletions, all in the named spots; the Introduction, Literature Review, RQ
definitions, Methods bodies, Experiment, and Timeline are otherwise untouched):
(a) Contingency section: new "Risk Analysis and Contingency" section (future tense, proposal
    voice) with the pivot condition, the diagnostic instruments as planned tools, the diffusion
    competitive-landscape remark promoted to the explicit fallback hypothesis / positive
    control, RQ3a's trained-proposal clause folded in, one sentence reserving dataset/metric
    adjustment (pointing to the evaluation-metrics section), and one sentence sharpening the
    RQs into diagnostic form while preserving their substance.
(b) Errata: DLS transition-kernel sign corrected to centre on the descent direction
    $-\nabla U$ (was $+\nabla U$, which ascended the energy), with a note referencing Zhang et
    al.'s $\nabla\log p=-\nabla U$ form; lambda renamed from "Lagrange multiplier" to a fixed
    constraint (penalty) weight; the CLS step-size/noise collision resolved by renaming the
    CLS step size $\eta_t\to\epsilon_t$ to match Equation (2)'s convention (noise stays
    $\xi_t$); the two RQ cross-references fixed (DLS 4.1 RQ2 -> RQ1a; AR baseline 4.5 RQ3 ->
    RQ2 quality floor); "fullfills" -> "fulfills"; six live straight-quote pairs converted to
    LaTeX quotes; inputenc/fontenc added.

Bibliography: ref.bib was present (28 entries, all with arXiv URLs; my initial `find` missed
it). The `@misc` preprint entries that are in fact published were upgraded to carry venues
(Ouyang -> NeurIPS 2022, Dathathri -> ICLR 2020, Zhang DLS -> ICML 2022, Kumar MuCoLa ->
EMNLP 2022, Gong DiffuSeq -> ICLR 2023, Hu -> ICML 2017 with the year corrected, Zhang survey
-> ACM Computing Surveys 2023). Alhafni 2024 and Li 2024 remain arXiv with URLs (recent).
FLAG: the 2024/2025 entries and any venue I supplied should be spot-checked by the author
against the final published record.

Compile: latexmk exit 0, 0 undefined citations/references, 12 pages.
Supervisor note drafted at Doc/final/proposal/supervisor_note.txt (five-line amendment note,
RQs unchanged, errata listed, thesis-to-RQ mapping).

## PART 7: gates and final report

### Gate results
- latexmk clean, all three documents: thesis exit 0 (125 pp), proposal exit 0 (12 pp),
  beamer exit 0 (20 pp). Zero undefined references and zero undefined citations in all three.
  Max overfull hbox: thesis 7.49pt, proposal 33.83pt, beamer 38.01pt, all below the 40pt bar
  (the proposal Timeline table overflow of 49.9pt was fixed by narrowing its columns; content
  unchanged).
- Numbers diff: revision/numbers_diff_phase6.py reports RESULT: ALL OK. A targeted check
  confirms every Part 1 correction is consistent with source and with numbers.json: Table 12
  MH row 9.202/8.834/9.748/9.025 == numbers.json cls.policy.mh.nogn.free.s50; the -1.12 slope
  == diag_likelihood_trap_gpt2sft.json pooled_sampling (-1.1234); fig_traj_stats.json holds
  both DLS configs (MH 5.33, no-MH 48.67). Part 1 changes are reflected in numbers.json, not
  diffed away.
- Render proofs in Doc/final/proofs/: titlepage_vs_template.png (our page vs template),
  abstract_one_page.png (fits one page), fig01_quenching.png (Fig 1, corrected legend),
  fig03_mh_decomposition.pdf, fig09_mh_accept.pdf (single config), fig10_trap_scatter.pdf,
  fig12_trap_length.pdf (-1.12), fig13_traj_distance.png (enlarged strips), and four beamer
  slides (energy_promise, linearization, positive_control, takeaways). All visually inspected.

### Resolution table (a): author issue list (31 items)
1 title-page gap + abstract width: titlepage matched to template verbatim, \restoregeometry;
  PNG proof. 2 RQ bridge in 1.4 + 6.1 RQ2b sentence. 3 Contribution 2 split into three
  sentences. 4 background diffusion subsection (sec:bg-discrete-diffusion) + 2 bib entries.
  5 marble analogy cut to one sentence + caveat. 6 Roberts-Tweedie reworded + results link +
  2 bib entries. 7 continuous-sampler conclusion made exact vs the two-objects passage.
  8 Taylor terms named + derivation footnote. 9 CLS update equations documented in 4.3
  (eq:meth-cls-update). 10 GFlowNet metaphor compressed. 11 3.4/3.5 "final" openers
  de-duplicated. 12 4.1 metric clarification, full justification once in 4.4. 13 A.2 reframed
  as released artifact map tied to code-availability. 14 4.5 why-likelihood-as-reward
  paragraph. 15 config-matrix sentence in 4.2. 16 task-comparability paragraph in 4.5.
  17 MuCoLa/COLD faithful-implementation + thread-D sentence in related work + mucola arm in
  5.11. 18 results-opening provenance sentence removed. 19 5.1 plug-and-play wording ->
  step-size-vs-embedding-distance. 20 quenching metallurgy compressed to a clause. 21 Figure 1
  legend fixed (line-style handles), gn config stated, gn-off companion figure added
  (fig:dls-traj-nogn), source comment fixed. 22 Figure 3 (mh-decomposition) recolored
  (fill+step, zorder), single config. 23 5.4 DLS config stated + no-MH DLS number added
  (48.7 cells) via the approved collect_traces run; fig_traj_stats.json holds both DLS
  configs. 24 5.1 alpha grid stated (50 values, logspace). 25 5.5 sentence surgery
  (norm-preserved split; bounded-equivalence three-clause; Llama declaratives with CI
  [0.45,2.09]). 26 5.5.3 judge config stated + code path re-verified. 27 5.6 wording plain
  (candidate move vs spacing; 15.0/24.2 as mean absolute nats). 28 Figure 10 (trap-scatter)
  six distinct colours. 29 5.8 scale vs anisotropy separated + PCA-cosine reconciliation.
  30 5.10/5.7 generator attribution (AR decoding, not Langevin). 31 5.11 infilled tokens
  bolded + drift sentence before the numbers.

### Resolution table (b): evaluation1 items
- Conclusion length: body already 2 pp (the "8 pages" was body + 5-page bib); not further cut.
- Related-work interpretation removed: 3.2 "clean separation" verdict and 3.3 "coherent-text
  optimum" axis reduced to forward-pointers.
- DLS/CLS explanatory repetition: Taylor "sole signal" stated once (Phase 8); marble halved
  again (item 5).
- Linearization-radius wording made consistent with the no-sharp-threshold limitation
  (abstract + 5.6: "already uninformative at admissible distances").
- Verbatim central formulation reduced to 3 uses (abstract, 5.5, 6.1), paraphrased elsewhere.
- Evaluate-vs-differentiate: one canonical statement (6.3) + abstract callback; related-work
  occurrence pruned. Diffusion three-result summary and classifier-alignment already at one
  place each (Phase 8). Intro opening two paragraphs compressed.

### Resolution table (c): evaluation2 items
1 corpus map table (tab:corpus-map) + domain-shift sentence; 4.1 and abstract corrected to
  WikiText-2 recovery / ROCStories SFT (author-approved). 2 5.11 rigor table
  (tab:constrained-contrast) with n=150 and explicit no-CI note; 4.6 states n + plan.
  3 four-vs-five diagnostics fixed to five in 4.1 and 4.7. 4 proposal-delta paragraph in 6.4.
  5 bibliography full author lists (Brown, Bengio2023, Hu2022; Dubey documented). 6
  straight-through/Gumbel-softmax mention + 2 bib entries. 7 Gibbs "comparable to". 8 abstract
  one page + digit numerals + closing sentence (PNG proof). 9 1.2 reveal clause. 10 Figure 13
  token strips enlarged 6->8.5pt.

### Resolution table (d): proposal-evaluation major items
- Sign error: DLS kernel centred on -grad U (descent); note referencing Zhang's grad-log-p.
- lambda renamed from Lagrange multiplier to fixed constraint weight.
- eta/step-size collision resolved (CLS step size eta -> epsilon, matching Eq 2).
- RQ cross-refs fixed (DLS -> RQ1a; AR baseline -> RQ2 quality floor).
- Contingency / risk section added (the missing item the eval flagged as most consequential).
- Bibliography venues completed for the published preprint entries.
- Typo "fullfills", straight quotes, and encoding (inputenc/fontenc) fixed.

### WHAT CHANGED (one paragraph per document)

THESIS (Doc/final/thesis/). Three submitted numeric errors were corrected at source: the
Table 12 CLS-MH row (was a duplicate; now 9.202/8.834/9.748/9.025), the Figure 12 slope (the
-1.12 headline is real and the figure was regenerated to match it), and Figure 9 (regenerated
for one stated config); tracing Figure 9 also exposed and fixed a grad-norm/MH mislabel in
5.3. The corpus claim in 4.1 and the abstract was corrected from ROCStories to WikiText-2
(out-of-domain recovery, ROCStories SFT) with a new corpus-map table. Chapter 2 gained a
discrete-diffusion subsection and a corrected Roberts-Tweedie/Langevin-convergence passage;
Chapter 4 gained the CLS update equation, the metric clarification, and the
why-likelihood-as-reward and task-comparability paragraphs; Chapter 5 gained the 5.11 rigor
table, the corrected 5.3/5.5/5.6/5.8 wordings, generator attributions, and a gn-off companion
figure; the abstract is now one page; the verbatim central claim appears three times.
Sections touched: abstract, 1.1-1.4, 2.2-2.5, 3.2-3.4, 4.1-4.7, 5.1-5.11, 6.1-6.4, 7, A.1-A.4.

PROPOSAL (Doc/final/proposal/). Exactly Part 5's slate: a new Risk Analysis and Contingency
section, the sign/lambda/notation errata, two RQ cross-reference fixes, the "fulfills" typo,
quote and encoding fixes, and bibliography venues; nothing else changed (28 insertions, 14
deletions). Compiles to 12 pages. A five-line supervisor note is drafted.

BEAMER (Doc/final/beamer/). A complete 20-minute defense deck was created, following the
diagnostic story arc with four original TikZ diagrams and six backup slides; it replaces the
placeholder template and compiles to 20 pages.

MARKDOWN DOCS. README.md and REVISION_RESTRUCTURING.md were updated for the final tree and the
graphicspath fix; REVISION_WRITING.md (this file) and PRESENTATION_LOG.md log every action.

### Expected author decisions
None beyond the two already confirmed in-session (the 5.3 MH-contrast rewrite and the corpus
correction) and two flags to spot-check: the proposal's 2024/2025 bibliography venues, and
the Beginn-der-Arbeit start-date field on the thesis title page.
