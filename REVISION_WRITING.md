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
- 17 [SUPERSEDED 2026-07-28, see CLAUDE.md item (f)] related work states the CLS implements the COLD/MuCoLa mechanism faithfully, that the
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

---

# PHASE 10: the evaluation4 rewrite (2026-07-26, ~20:40 to ~22:30 CEST)

Read in the prescribed order: REVISION_RESTRUCTURING.md, the evaluation3-pass entries of
REVISION_LOG.md (Parts A through D5), refs/evaluation4.md in full, the resolution tables
above, and refs/{Guidelines_for_academic_thesis_writing_at_the_IMS,Checklist_Masterthesis}.pdf.
No experiment was run. Every number below comes from an existing JSON or CSV; where an edit
would have needed a run, the honest sentence the existing data supports was written instead
and the fact is logged.

Rollback point for this phase: commit `087ab37`.

---

## PART 1: the hypothesis architecture

The thesis is now organized as an elimination of three named candidate explanations, stated
once in 1.2 and referred back to by name thereafter.

- **1.2** gains a paragraph naming **Hypothesis A** (the target energy is unusable),
  **Hypothesis B** (autoregressive training does not contain what a local proposal needs) and
  **Hypothesis C** (the input-embedding parameterization discards accessible information, and
  right-context conditioning supplies more), and states that the results chapter eliminates
  them in that order.
- **2.1** now says explicitly that Hypothesis B is stated in the terms of the teacher-forcing
  paragraph, is the natural first explanation, and is found too strong.
- **5.3.3 (gradient-free baselines)** opens by naming Hypothesis A and closes by rejecting it
  and handing the remaining choice to 5.4.
- **5.4.1 (the output-side surrogate)** closes by rejecting Hypothesis B ("the usable local
  direction exists inside the frozen model, it survives its training objective untouched")
  and pointing at 5.9 for the second half of C.
- **5.9 (the proposal ladder)** opens by splitting C into its two halves and closes by
  supporting both, with the replacement of the objective attribution stated as a finding.
- **1.5** rewrites the structure paragraph as that progression.

The three places where the document previously "changed its mind" now read as designed tests.
The withdrawal sentences are kept as findings, not errata: the quenching withdrawal stays in
5.1.1, the anti-guidance withdrawal in 5.3.2, and the attribution replacement in 5.9.2.

### RQ reformulation (evaluation4 items 3, 8, 9)

| old | new |
|---|---|
| RQ1 "Can the frozen likelihood be sampled effectively with theoretically faithful Langevin dynamics ... and if not, why not?" | **RQ1** "Does the input-embedding gradient of a frozen autoregressive sequence likelihood provide a useful proposal direction for discrete token revision, and if not, which alternative local quantities do?" |
| RQ2 | unchanged in wording; the over-broad answer narrowed (below) |
| RQ3 "Does amortizing the energy with a GFlowNet ... escape the failures?" | **RQ3a** (does the policy generate high-reward text?) and **RQ3b** (does the tuned energy become more navigable by the local input-embedding surrogate?), with one sentence saying why the split is not pedantic |
| RQ4 (central) | **E**, posed in a separate block as an extension, with one sentence saying it is an application of the machinery whose prerequisite RQ1 tests and that the contributions do not rest on it |

The proposal-to-final-RQ mapping is preserved in compressed form as a footnote on the
extension paragraph, updated to the reformulated numbering (RQ1+RQ1a -> RQ1; RQ1b -> RQ2;
RQ3a -> RQ3a and RQ3b; RQ2a -> E; RQ2b -> the gradient-free baselines).

6.1 answers RQ1, RQ2, RQ3a, RQ3b and E under exactly these headings.

### Contributions (evaluation4 item 4)

Recounted and renumbered from six-listed-as-five to **five**, one to two sentences each,
following evaluation4's suggested structure: (1) the ablation plus the sweep, with the
certified equivalence; (2) what the derivative discards and that it is recoverable, folding
the old contributions 2 and 3 together; (3) the MH decomposition in both samplers plus the
likelihood trap; (4) the GFlowNet taxonomy plus the energy-level experiment; (5) the
controlled proposal ladder. The old contribution 6, which described diffusion as supplying
"the direction the autoregressive gradient lacks", is gone: diffusion is one rung of (5).
Abstract, contributions and conclusion were checked against each other for the same claim set.

### 6.2 rewritten (evaluation4 item 6)

Retitled **"One Problem, Four Mechanisms"**. It now states one shared high-level problem (the
plug-and-play construction asks the raw likelihood to be a quality objective, a locally
navigable energy and a reward at once, and the three roles come apart) and then four
*distinct* lower-level mechanisms: proposal-information failure, continuous-state correctness
failure, objective-quality failure, amortized-training failure. Their independence is argued
explicitly (the GFlowNet never differentiates its reward, so a missing self term cannot cause
its collapse; the likelihood trap concerns the optimum's location and survives a perfect
proposal; the projection geometry belongs to the continuous relaxation alone). The claim that
one mechanism unifies everything is withdrawn; what survives as a single thread is the
statement about proposals, and the ladder is its positive form.

---

## PART 2: terminology (evaluation4 item 2)

The quantity formerly called the "one-hot input gradient" is renamed the **relaxed
token-indicator derivative** (short form: token-indicator derivative). The name states which
derivative, with respect to which coordinates, under which parameterization.

At the definition site (5.4.1) the object is now defined before it is named:

- a relaxed objective is introduced, `\tilde{L}(z_i) = sum_v z_i[v] log p(v | x_<i) +
  L_future(E^T z_i)`, with `z_i` relaxed from the simplex vertex of the current token
  (new equation `eq:relaxed-objective`);
- it is stated that `\tilde{L}` coincides with the sequence log-likelihood at a vertex, so
  this is a relaxation of the thesis's own target and not a different objective;
- the closed form `d\tilde{L}/dz_i[v] = log p(v | x_<i) + g^T e(v)` follows
  (`eq:onehot-grad`, label kept so existing cross-references resolve);
- and the distinction the evaluation asked for is made explicit: this is **not** the ordinary
  derivative through the embedding lookup, which by the chain rule is `E^T g` and carries the
  future term alone. The whole difference is the self term that only the two-role relaxation
  exposes.

Swept globally. Occurrences of the old name in body prose, table column headings, table
captions, the abstract and the conclusion were replaced; "one-hot" now survives in exactly
two places, both inside the paragraph that explains why the looser name is being avoided.
Table 7's columns became "Input emb." / "Token ind." with the caption defining them (this
also fixed a 91.6pt overfull box the longer headings had created).

---

## PART 3: repetition purge and length

### Structural changes

| change | evaluation4 item |
|---|---|
| 5.1 and 5.2 merged into "Proposal Calibration and the Near-Uniform Main Grid", the entropy measurement becoming 5.1.1 (label `sec:results-quench` kept) | 11.5 |
| The load-bearing sweep became a subsection of the null section rather than a sibling section | 11.6 |
| "The Linearization Radius" retitled "Why the Input-Embedding Surrogate Fails"; its subsection retitled "Recovering the Missing Term with an Output-Side Surrogate" | 11.7 |
| The classifier-guided steering summary moved out from between the diffusion result and the ladder, into the extension section, so the ladder now follows the diffusion arms immediately | 9, 11.8 |
| 5.14 and 5.15 merged into one section, "The Proposal Ladder", with the diffusion arms and the MLM control as its two subsections | 11.6, 11.9 |
| The constrained-generation section moved after the ladder and retitled as the extension | 9 |
| "Sampler Trajectories in Embedding Space" folded into the MH section as a subsection | item 10 |
| Section "Consistency Across Models and Architectures" and Table `tab:crossmodel` REMOVED as pure repetition (both of its columns are stated in full elsewhere); its two interpretive clauses folded into 5.5 and 5.3 | item 10 |
| Figure `fig:lasttoken` REMOVED: its two series were the lower block of Table 11, which it sat beside | item 10 |
| Appendix showcase table CONSOLIDATED: it had reprinted the entire sequence in all ten method rows of all four sequences; the sequence is now given once per block and the rows carry the recovered token alone. Same four sequences, same ten methods, same seeded draw | appendix trimming |
| `gprime_examples` trimmed to one pair per cell; a second longtable that was declared but empty in the submitted source (header rows only) deleted | appendix trimming |
| Appendix A.5 opener de-duplicated against the body summary it had repeated verbatim | item 10 |

Every removal is recorded as a `%` comment at the removal site, naming what was removed, why,
and where the content now lives. Nothing was deleted silently.

### Prose compression

Roughly 90 paragraphs across all eight chapter files were rewritten shorter, at a typical
25 to 35 percent reduction, with every number, citation and qualification preserved. The
heaviest reductions were in 2.4 (DLS/CLS), 3.2 (energy-based decoding), 4.3, 4.4, 4.5, 4.7,
5.1, 5.3, 5.4, 5.7, 5.8, 6.1, 6.3, 6.4 and the conclusion.

### Length, reported honestly

| chapter | pages at 142-page start | now | delta |
|---|---|---|---|
| front matter | 9 | 9 | 0 |
| 1 Introduction | 6 | 7 | +1 |
| 2 Background | 11 | 10 | -1 |
| 3 Related Work | 8 | 7 | -1 |
| 4 Methodology | 14 | 11 | -3 |
| 5 Results | 43 | 40 | -3 |
| 6 Discussion | 9 | 9 | 0 |
| 7 Conclusion | 4 | 3 | -1 |
| Bibliography | 7 | 7 | 0 |
| Appendix | 31 | 22 | -9 |
| **total** | **142** | **126** | **-16** |

The introduction grows by one page because it now carries the hypothesis architecture, the
split RQ3 and the extension block; that is the change Part 1 asked for and it pays for itself
in the recaps it removes downstream.

**The hard limit of 115 to 119 pages is NOT met. The document is 126 pages, seven over.**
Per the standing instruction, the full Part 3 programme was completed first and no further
substance or floats were cut; the remaining candidates are listed for the author's decision
in the closing section below.

---

## PART 4: consistency sweep for the changed findings

Swept the full source and the rendered text for each named orphan.

| orphan | hits in rendered PDF | disposition |
|---|---|---|
| quenching as a live mechanism | 3, all inside 5.1.1's withdrawal paragraph | correct by the constraint |
| anti-guidance assertion | 1, "the evidence supports indifference, not anti-guidance" | correct |
| "training objective is the cause" | 0 as an assertion | the corpus-map row "Diffusion positive control (SEDD)" was renamed "Diffusion and masked-LM proposal ladder"; "positive control" no longer appears anywhere |
| "0 to 39" without comparators | 0 | the ladder table carries the gradient (0.0), uniform (0.5) and AR-conditional (23.5) rows, and 5.3.3 now states the gradient-free exact-match comparator (top-$k$ rescore 33.0, Gibbs 18.5) with the explicit sentence that 39 is a gain over 33, not over nothing |
| gradient "carries no usable signal" unscoped | 1, in 4.7, rescoped to "the input-embedding gradient" | fixed |
| RQ4 | 0 | all references now read "the extension question E" |
| "evaluate, do not differentiate" | 0 | replaced everywhere by the corrected implication |

The final claim set now reads identically in the abstract, 1.2, 1.4, 6.1, 6.2, 6.5 and the
conclusion: certified equivalence for the input-embedding-gradient proposal; the output-side
derivative of the same frozen model recovers 40 percent on GPT-2 and 41 on Llama-3; the
conditioning ladder attributes the remaining gap to right-context access; and the practical
implication is "differentiate the right object, or propose from the output side".

Numbers diff: `revision/numbers_diff_phase6.py` reports **RESULT: ALL OK** (46 checks, 0 failures).

---

## AUTHOR-REQUESTED FIXES (2026-07-26, after the first pass)

1. **4.7 closing paragraph removed.** The job manifest, the file-queue lock protocol, the
   stranded-lock failure mode and the per-configuration dispatch limit are implementation
   detail of the harness, which the IMS guidelines keep out of the methods section. The
   engineering account remains in REVISION_LOG.md (C6, C7, D3) and in the README. Removal
   recorded as a `%` comment at the site.
2. **4.8 equivalence-suite sentence removed.** The named script and the README artifact-map
   sentence are gone; the section now says only that the code and configuration definitions
   are retained and that each table and figure is linked to the result file it draws from.
   The verification itself was run and is recorded in REVISION_LOG.md.
3. **5.1.1: the "an earlier draft ... withdrawn here" sentence removed.** The withdrawal now
   stands on the measurement alone, with no reference to a previous version.
4. **5.2.1: the correction setting is now stated for the discrete sampler too.** The sentence
   reads "the continuous sampler with the correction enabled visits about two ... with the
   correction disabled about 45 of a possible 50. The discrete sampler splits the same way but
   at a different scale: about five cells with the correction enabled and about 49 with it
   disabled ... and in both settings its state remains a genuine token embedding."
   Source: `figures/fig_traj_stats.json` per_config `dls_policy_gn_mh` 5.33 and
   `dls_policy_gn_nomh` 48.67.
5. **Figure 3 caption overflowing the folio: fixed.** The forest plot is now sized by height
   (`height=0.58\textheight,keepaspectratio`) rather than width. Root cause: the 35-row plot
   at `0.85\textwidth` was tall enough to push its four-line caption onto the page number.
   The same pass had already moved the figure back into the body: it, `fig:lin-scatter` and
   `fig:lin-radius` had floated out of Chapter 5 into the appendix region because ten tables
   and four figures in that chapter carried a bare `[t]` specifier. All `[t]` floats in
   `05_results`, `08_appendix` and `tab_confusion` were changed to `[htbp]`; every core float
   is now in the body, zero float promotions are reported, and the change also recovered two
   pages of whitespace.
6. **Table 9 extended with the output-side surrogate** (the author's question, answered in
   the affirmative from existing data). `tab:gfn-unify` now carries four columns: the
   input-embedding gradient and the token-indicator derivative at the calibrated
   configuration, and the token-indicator derivative in a surrogate-driven configuration with
   its exact-recovery rate. Values and provenance:

   | energy | input emb.\ (calib.) | token ind.\ (calib.) | token ind.\ (surrogate-driven) KL / exact |
   |---|---|---|---|
   | GPT-2 Large | 6.541 | 6.335 | 3.229 / 40.0% |
   | GFN lb0-500 | 6.306 | 6.022 | 4.539 / 25.0% |
   | GFN lb0-2000 | 6.721 | 6.496 | 4.999 / 21.0% |
   | GFN lb1-500 | 6.415 | 5.931 | 5.343 / 9.0% |

   Sources: `results/grid/rev3/onehot_mh_gn.json` and `xm_onehot_gfn-*.json` (calibrated,
   n=200, eps 10.5->0.1, T=5, gn on); `ohsweep_e10p5_t1p0.json` (base, n=50, T=1, gn off) and
   `xm_onehot_gfn-*_sharp.json` (n=200, eps 1050->10, T=1, gn off). The n and configuration
   differences are stated in the caption. The prose now answers RQ3b for both surrogates.
7. **Table 11 was NOT given a token-indicator arm**, because none was run at those three
   positions and running one is a new experiment. Instead 5.8 now states the algebraic fact
   that makes the arm unnecessary there: at the final position the future term of
   Equation (11) vanishes identically, so the token-indicator derivative reduces exactly to
   `log p(v | x_<i)`, which is what the conditional-argmax and top-$k$ rows already rank by.
   The energy-only rows of Table 11 are therefore that surrogate under another name, and the
   34.5 to 40.0 percent they recover is what it attains where the input-embedding gradient is
   exactly zero.
8. **Classifier-guided steering retitled and rescoped.** "on a Diffusion Landscape" became
   "on a Navigable Landscape", and the section now states that the diffusion model was the
   carrier because at the time it was the only proposal in the study that recovered anything,
   names the masked language model and the token-indicator proposal as the two carriers found
   later that would serve the same purpose, and says plainly that neither is tested here
   because doing so would be a further experiment. No steering run with either exists.

---

## PART 5: IMS guidelines and checklist re-verification

| item | status |
|---|---|
| Title page: institute name and address, title, author, two examiners, supervisor, start and end dates | present and untouched; geometry, spacing and tabular are the template's verbatim |
| Declaration of authorship, signed form with place and date | untouched, first page after the title page, German with the non-binding translation footnote |
| Table of contents, list of figures, list of tables | regenerated: 15 figures, 22 tables, matching the 37 float environments in source |
| Introduction: topic, problem, narrowing, goal, research questions, relevance, approach, outline | all present; the RQ block and the outline were rewritten this pass |
| Main part: background, related work, materials and methods, experiments/results/discussion | present |
| Conclusion answers every question raised in the introduction | RQ1, RQ2, RQ3a, RQ3b and E answered in 6.1 under the reformulated wording; the conclusion summarizes without re-arguing |
| Bibliography complete and consistent | 54 entries, all 54 cited, none uncited; no entry added or altered this pass |
| ArXiv only where no published version exists | unchanged from the Phase 9 audit |
| All abbreviations introduced at first use | verified in reading order after the moves: GFlowNet 1.3, MH 2.3, DLS and CLS 2.4, SEDD 2.5, LoRA 2.6, KL 4.4 |
| Abbreviations avoided unless frequent | unchanged |
| Appendix items cross-commented | every appendix subsection names the body section it supports; A.5 now points at 5.10.1 rather than the retired label |
| Abstract on one page | verified: page 4 of the PDF is the table of contents |

---

## PART 6: proposal verified, NOT rewritten

`Doc/final/proposal/proposal.tex` was read and checked sentence by sentence against the final
story. Verdict: **NO CHANGE**. The file was not edited; it still compiles to 12 pages with
zero undefined references or citations.

Contingency section, sentence by sentence:

1. "The programme above presupposes that the frozen model's gradient supplies a usable search
   direction ..." — CONSISTENT. The thesis tests exactly this presupposition and reports that
   it fails for the input-embedding gradient. Future tense, proposal voice, no claim about
   which derivative.
2. "Should that presupposition fail ... the thesis will be redirected rather than abandoned
   ... it will turn to mechanistic diagnosis." — CONSISTENT, and this is what happened.
3. "The instruments for that diagnosis are named here as planned tools rather than improvised
   later ... a controlled ablation that separates the direction of the gradient from its
   magnitude; a linearization test ...; a decomposition of the Metropolis--Hastings acceptance
   ratio ...; and a measurement of whether maximizing the model's own likelihood corresponds
   to good text." — CONSISTENT. All four were run and all four are reported.
4. "Under this branch the diffusion discussion ... is promoted ... into an explicit fallback
   hypothesis and positive control: **if** the difficulty is that the autoregressive
   likelihood was never trained to provide a score, **then** a score-trained diffusion model
   ... should supply the local direction ... and substituting it as the proposal would isolate
   the training objective as the cause." — CONSISTENT, and this is the sentence to look at
   hardest. It is a conditional. The thesis ran the test, found the consequent true (the
   diffusion proposal does work) and the antecedent false as an explanation (a model with no
   score objective works better). A conditional whose antecedent is disconfirmed is not
   contradicted by that disconfirmation; the contingency section proposed the experiment and
   the thesis reports its result, including the result that the isolation the sentence hoped
   for is not achieved by that design alone. No adjustment is warranted, and adjusting it
   would misrepresent what was planned.
5. "The amortization step of RQ3a folds into the same branch, since a GFlowNet that only ever
   evaluates the reward ... tests whether learning to sample sidesteps a gradient pathology."
   — CONSISTENT with RQ3a and RQ3b as now split.
6. "the datasets and metrics may be adjusted as the diagnostic questions sharpen" — CONSISTENT
   with the corpus split and the dropped deliverables accounted for in 6.4.
7. "the research questions will be sharpened into diagnostic form while preserving their
   substance: RQ1 becomes whether and why the gradient is usable, RQ2 becomes whether steering
   is possible on the landscape as it stands, and RQ3 becomes whether amortization repairs it"
   — CONSISTENT with the Phase 10 reformulation, which is exactly this sharpening carried one
   step further.

No sentence positively contradicts the thesis. Diff summary: none, the file is byte-identical
to its Phase 9 state.

---

## PART 8: gates

| gate | result |
|---|---|
| `latexmk -pdf thesis.tex` | exit 0 |
| `latexmk -pdf proposal.tex` | exit 0, 12 pages |
| `latexmk -pdf Presentation.tex` | exit 0, 25 pages |
| undefined references / citations (all three) | 0 / 0 |
| multiply defined labels | 0 |
| float promotions (`[h]` changed) | 0 |
| max overfull hbox, thesis | 20.84pt (gate 40pt) |
| max overfull hbox, beamer | 38.01pt (gate 40pt; the pre-existing TikZ annotation) |
| bibliography | 54 entries, 54 cited, 0 uncited |
| abstract | one page (page 4 is the ToC) |
| numbers diff | `RESULT: ALL OK`, 46 checks, 0 failures |
| total page count | **126** (target 115 to 119: NOT met, see below) |

### Body-float check: every core result keeps its float in the main text

| core result (constraint 6) | float | printed page |
|---|---|---|
| certified equivalence, flagship | Table 3 `tab:chainstats` | 54 |
| certified equivalence, whole grid | Figure 3 `fig:forest` | 55 |
| near-uniform proposal | Table 2 `tab:proposal-sharpness` | 47 |
| main ablation | Table 4 `tab:fallacy` | 56 |
| anti-guidance withdrawn | Table 5 `tab:mhfix` | 59 |
| gradient-free baselines (Hypothesis A) | Table 6 `tab:baselines` | 61 |
| token-indicator surrogate | Table 7 `tab:onehot` | 66 |
| token-indicator sampler, and the temperature | Table 8 `tab:onehot-sweep` | 67 |
| uniform control and the conditioning ladder | Table 13 `tab:mlm` | 82 |

Chapter 5 runs from page 45 to page 84, so all nine are inside the body. No body float was
moved to the appendix at any point in this pass.

### Prior-fix preservation audit

Every resolution recorded in the tables above was re-checked after the rewrite.

| group | still holds | superseded |
|---|---|---|
| Phase 9 author list, items 1 to 31 | 29 of 31 | item 13 (A.2 released-artifact-map paragraph) survives but its README artifact-map sentence in 4.8 was cut at the author's instruction this pass; item 20 (quenching metallurgy compressed to a clause) is subsumed by the withdrawal, which supersedes PHASE9 Part 2 item 20 as already logged in the evaluation3 pass and confirmed by the author (REVISION_LOG D1.3) |
| evaluation1 items | all hold | the "cut the conclusion" item rests on a miscount of eight pages that included the five-page bibliography; the conclusion body is now 3 pages and was rewritten rather than cut further |
| evaluation2 items 1 to 10 | all hold | none |
| proposal-evaluation items | all hold | none |
| evaluation3 findings of record | all hold | none |

Two supersessions are recorded rather than applied silently: the quenching override (already
logged), and the removal of the 4.8 artifact-map sentence, which narrows Phase 9 item 13 to
the appendix where the released-artifact paragraph still lives.

### evaluation4 resolution table

| # | item | action | location |
|---|---|---|---|
| 1 | causal explanation changes | rewritten as a designed elimination; 2.5 no longer presents score training as the expected repair, 3.5 frames diffusion as a comparative family, 5.9 states the replacement of the attribution, 6.2 makes derivative choice and conditioning the operative variables, the conclusion drops any suggestion that score matching is necessary | 1.2, 2.1, 2.5, 3.5, 5.3.3, 5.4.1, 5.9, 6.1, 6.2, 7 |
| 2 | "one-hot input gradient" terminology | renamed the relaxed token-indicator derivative, defined via an explicit relaxed objective and distinguished from `E^T g` | 5.4.1, swept globally |
| 3 | RQ1 imprecise | reformulated in the recommended form | 1.4, 6.1 |
| 4 | five contributions versus six | recounted to five, renumbered, diffusion folded into the ladder contribution | 1.4 |
| 5 | flow weakens at the late revision | the elimination architecture; withdrawals kept as findings | 1.2, chapter 5 transitions |
| 6 | "unified mechanism" is not unified | 6.2 rewritten as one shared problem plus four distinct mechanisms | 6.2 |
| 7 | RQ2 over-broad statement | narrowed in 2.3, 3.2, 6.1 and 6.3 to the implementations and landscapes tested here | 2.3, 3.2, 6.1, 6.3 |
| 8 | RQ3 conflates two questions | split into RQ3a and RQ3b, answered separately | 1.4, 5.7, 6.1 |
| 9 | RQ4 formally central, substantively secondary | repositioned as extension E, section moved after the ladder and retitled | 1.4, 5.10, 6.1 |
| 10 | repetition | purge as tabulated in Part 3 | throughout |
| 11.1 | proposal mapping paragraph | compressed to a footnote | 1.4 |
| 11.2 | contributions | see item 4 | 1.4 |
| 11.3 | compress 2.5 and 2.6 | merged into one subsection and shortened | 2.5 |
| 11.4 | shorten related work 3.2 | compressed; the verdict moved to 6.3 | 3.2, 6.3 |
| 11.5 | merge 5.1 and 5.2 | done | 5.1 |
| 11.6 | connect 5.5 and 5.6 | the sweep is now a robustness subsection of the null | 5.3.2 |
| 11.7 | reorganize 5.7 and 5.7.1 | retitled around the failure and its repair | 5.4, 5.4.1 |
| 11.8 | move 5.14.4 to the appendix | body summary reduced to one paragraph and relocated to the extension | 5.10.1, A.5 |
| 11.9 | make 5.15 central, after the diffusion result | both are subsections of one Proposal Ladder section | 5.9 |
| 11.10 | cut the conclusion | rewritten to report rather than synthesize; 3 pages | 7 |
| 12 | final RQ evaluation | reflected in the reformulated RQs and 6.1 | 1.4, 6.1 |

### WHAT CHANGED

**THESIS.** The document now argues by elimination rather than by accumulation: three
candidate explanations are named in 1.2 and closed in order, with the gradient-free baselines
disposing of the target, the token-indicator re-analysis disposing of the model, and the
proposal ladder confirming that the derivative and the conditioning are what matter. RQ1 is
reformulated to the input-embedding gradient, RQ3 is split into a policy question and an
energy question, RQ4 becomes an extension, and the contributions are recounted to five. The
central new quantity is renamed the relaxed token-indicator derivative and defined through an
explicit relaxed objective so that it is unambiguous which derivative is taken in which
coordinates. 6.2 no longer claims a single unifying mechanism. Sixteen pages were removed,
nine of them from the appendix, by merging four pairs of sections, deleting one wholly
repetitive section and two redundant floats, consolidating the showcase tables, and rewriting
roughly ninety paragraphs shorter without losing a number or a qualification. Table 9 gained
the output-side surrogate from existing runs; Table 11 did not, and the algebraic reason it
does not need one is now stated. Every claim of the old story was swept from the source and
from the rendered text.

**PROPOSAL.** Verified sentence by sentence against the final story and not edited. The
contingency section's diffusion sentence is a conditional whose antecedent the thesis
disconfirms, which is a finding the contingency permits rather than a contradiction.

**BEAMER.** Rebuilt on the current findings; logged in PRESENTATION_LOG.md.

### Expected author decisions

**Length.** The document is 126 pages against the 115-to-119 limit. The full Part 3 programme
is complete and no further prose can be cut without thinning explanation the guidelines
require. The remaining candidates, none of which was applied:

| candidate | saving | cost |
|---|---|---|
| Appendix A.1: reduce the eight support figures from 0.68 to 0.58 textwidth | ~2 pages | none to content; figure labels get smaller, all are vector PDFs |
| Remove `fig:traj-pca` (A.4), whose own caption says it captures 3.3% of the variance and "cannot be the evidence" | ~1 page | loses one illustrative panel; the full-space distances that carry the argument stay |
| Remove the six `\clearpage` commands in the appendix | ~2 to 3 pages | none to content; float placement becomes less predictable |
| Merge A.5.2 and A.5.3 (agreement ladder and confusion analysis) | ~1 page | the two analyses run together; evaluation2 praised neither specifically |
| Move Table 15 (representative configuration grid) to the released artifact set only | ~1 page | the appendix would state the count without showing a sample of the grid |
| Cut A.7.1 (guided-generation examples) to the four highest-contrast pairs | ~1 page | fewer qualitative examples; the selection policy and A.7.2 are untouched |

Applying the first three alone reaches 120 to 121; all six reach 117 to 119. The first, third
and fourth touch no content at all. This is put to the author rather than decided here,
because the instruction was to stop at this point rather than cut substance or floats.

---

## 2026-07-26 ~23:20 CEST  AUTHOR ROUND 2: print legibility, AI-tools declaration

Priority set by the author: figures must be understandable when the thesis is printed on A4.
The length candidates listed at the end of the Phase 10 report were explicitly NOT applied.

### Figure sizing

Text block measured from the build: `\textwidth` 426.79pt, `\textheight` 591.53pt. Every
size below was chosen backwards from that box and verified by rendering the page.

| figure | was | now | note |
|---|---|---|---|
| 1 DLS trajectories, 50 steps | 0.78 tw | 0.92 tw | three stacked panels |
| 2 MH decomposition | 0.80 tw | 0.95 tw | |
| **3 forest plot** | 0.58 th | **REDRAWN**, 0.90 tw, own float page | see below |
| 4 linearization scatter | 0.72 tw | 0.92 tw | |
| 5 linearization radius | 0.72 tw | 0.92 tw | |
| **6 self/future decomposition** | 0.68 tw | **full tw** | source aspect 0.475, so 203pt tall |
| 7 top-$k$ recall | 0.68 tw | 0.92 tw | |
| 8 acceptance by boundary | 0.68 tw | 0.92 tw | |
| **9 trap scatter** | 0.68 tw | **full tw** | aspect 0.548, 234pt tall |
| **10 anisotropy histograms** | 0.68 tw | **full tw** | aspect 0.505, 216pt tall |
| 11 trap length | 0.68 tw | 0.92 tw | |
| 12, 13 trajectory panels | 0.68 tw | 0.85 tw | aspect 1.059, 384pt tall |
| 14 trajectory distances | 0.68 tw | 0.80 tw | aspect 1.254 |
| 15 PCA projection | 0.68 tw | 0.85 tw | |

No figure overflows: zero overfull vboxes, max overfull hbox 20.84pt (gate 40pt), and every
enlarged figure was rendered and inspected.

**Figure 3 was redrawn, not merely rescaled.** Enlarging it alone could not work: the source
was 7.6 x 11.9in with 6.4pt tick labels, so any width that left room for its four-line
caption inside the text height put the 35 configuration labels below 5pt on the page. New
script `revision/replot_forest_chain.py` redraws it from the CACHED contrasts in
`results/revision/rev_chain_stats.json`. It is plot-only: no bootstrap is re-run and no
statistic is recomputed, so every point and interval is identical to the figure it replaces.
The new source is 8.0 x 9.8in with 12pt labels, sized backwards from a 0.90 textwidth slot
(384pt, a reduction of 0.667) so the labels print at 8pt; it takes a float page so the caption
clears the folio. Verified against the cached JSON: the flagship
`gpt2-large.dls.mh.gn.free.s50` chain-mean contrast still sits at $+0.002$ and exactly two of
the 35 intervals exclude zero, matching `summary.n_ci_excludes_zero_chain_mean = 2`.

Page count 126 -> 128. The author's instruction was that legibility takes priority.

### Appendix A.8, Use of AI Tools

Rewritten to the author's dictated content: Google AI Studio (Gemini) and Claude, used for
clarification of concepts checked against the primary sources and other material found
online; to paraphrase and copy-edit drafts of the manuscript; and for engineering support in
building the multi-GPU parallel infrastructure and the manifest and queueing mechanism that
make the long configuration grids practical to run. The closing responsibility paragraph is
unchanged, verbatim as the author specified.

FLAGGED: the previous version additionally disclosed that Claude was used through an agentic
coding interface to apply an author-specified list of edits to the LaTeX sources, to
transcribe verified numbers into tables, and to add cross-references. That sentence is not in
the dictated replacement. It is a declaration-of-authorship matter and therefore the author's
call, so it was removed as instructed and is recorded here rather than reinstated silently.

### Table 22 and Table 21: what cannot be added without a run

The author asked for the output-side surrogate and the RoBERTa proposal to appear in the
qualitative tables. Neither can be produced from stored data:

- The grid JSONs store an `examples` field of only the FIRST 8 samples. The showcase draws
  sequence indices 3, 34, 98 and 199 from a seeded `rng(0).choice(200, 10)`. Only index 3 is
  within the stored 8, and there the token-indicator run does recover the ground truth
  (`fish`), but the other three are not recoverable from any file.
- `rev_mlm_control.json` stores aggregates only (exact 44.5, ever 59.0, KL 2.737, accept
  50.27); it holds no per-sequence text at all.
- Classifier-guided steering was run only with the diffusion carrier. No guided run exists
  with the token-indicator or masked-LM proposal, so Table 21 cannot gain those columns at all
  without a genuinely new experiment.

Reported to the author rather than fabricated. The cheap option, a re-run of the two
recovering configurations with example logging on the showcase indices, is a GPU job and
therefore needs an explicit go-ahead.

---

## 2026-07-26 ~23:55 CEST  evaluation5: items 2, 3A, 3B, 3C, 3D, 3E, 4, 5/RQ2 applied

Author-selected subset of `refs/evaluation5.md`. Applied exactly these eight; the remaining
recommendations (8A remove the proposal-mapping footnote, 8B trim the roadmap, 8C demote
anisotropy, 6/7 cut the conclusion) were NOT applied, the last of them because it rests on a
page miscount recorded below.

### 3D, "pre-registered" (highest-risk wording)

`pre-registered` removed everywhere it described the equivalence margin, and replaced with
"fixed in advance" or "at a margin fixed in advance". Three sites: the abstract, the
paired-contrast paragraph in 5.3, and the caption of Figure 3. Zero occurrences remain in the
source or the rendered text. The margin was chosen before this comparison was run but there
is no timestamped registration, so the stronger term was not defensible.

### 3A, "the target is not at fault" scoped to the recovery task

- Abstract: "The target is not responsible for the recovery failure: a top-$k$ rescoring pass
  reaches 4.43 and a gradient-free Gibbs sampler 6.69 on the identical energy."
- 5.3.3 gains an explicit scope paragraph: Hypothesis A is rejected "for the failure under
  investigation", and the text now says in the same breath that Section 5.7 will show the same
  energy behaving badly in a different regime, that what is established is only that the energy
  does not explain why a gradient-guided search fails at in-place revision, and that it is not
  established, and is not true, that the energy is a good objective to maximize in open-ended
  generation.
- Conclusion: same scoping, with one sentence naming the likelihood trap as the counterpart.

This removes the tension evaluation5 identified between the abstract and Section 5.7.

### 3B, "the model is not at fault either" narrowed

Abstract and conclusion now read "Nor is the failure due to a complete absence of useful local
information in the frozen model", which is exactly what the token-indicator experiment
establishes and leaves room for the ladder's finding that a left-to-right factorization still
withholds the right context.

### 3E, Hypothesis B rejected in its strong form

5.4.1 now reads "the strong form of Hypothesis B is rejected", and a following clause states
what survives: a left-to-right factorization gives a proposal no path from the right context
to the position it is filling, so the frozen model contains some of what an in-place revision
needs and not all of it, with a forward pointer to the ladder that measures how much. The
chapter opener and the introduction were brought into line.

### 3C, the conditioning claim softened

- Abstract: "Conditioning access, in particular to the output distribution and the right
  context, explains that ordering better than score training does" (was "The operative
  variable is what a proposal may condition on, not the objective it was trained with").
- 5.6.2: "Score training is therefore not what the ordering tracks", followed by a new
  sentence listing what is NOT controlled: RoBERTa-large, SEDD-small and SEDD-medium differ
  from one another and from GPT-2 Large in scale, corpus and architecture, and only the
  tokenizer and the chain are held fixed, so the ladder establishes an ordering rather than an
  isolation.
- Conclusion: the same phrasing in both the finding and the uncertainty paragraph, the latter
  now naming scale, corpus and architecture as the uncontrolled differences.

### Item 2, what "output side" does and does not mean

A short paragraph added at the definition site, before the numbers: the missing direct
token-fit term is read off the output conditional, but the token-indicator derivative is the
SUM of that output-side self term and the same input-side future term the embedding gradient
computes, so the shorthand must not be read as saying the future term has disappeared. Where a
plain output-side quantity is meant instead, as in the left-conditional arms of the ladder,
the text now says so.

### 5/RQ2, "theoretically correct" replaced

Four sites. RQ2 in 1.4 now asks whether "equipping them with the exact accept-reject
correction for the proposal they implement" also makes them work; 4.3 says "the effect of the
exact accept-reject step"; 6.1 says "the exact accept-reject step and empirical performance
align"; and 5.2 closes with the distinction spelled out: the correction is exact for the
proposal it corrects, so the corrected chain is $\pi$-invariant by construction, and what
fails is the regularity the Langevin construction assumes underneath it, which the older
phrasing papered over.

### Item 4, Results reordered so the RQ1 argument is uninterrupted

New order, matching evaluation5's recommendation exactly:

| new | section | old position |
|---|---|---|
| 5.1 | Proposal Calibration and the Near-Uniform Main Grid | 5.1 |
| 5.2 | The Metropolis--Hastings Breakdown in Continuous Space | 5.2 |
| 5.3 | Gradient Direction Against a Norm-Matched Random Direction | 5.3 |
| 5.4 | Why the Input-Embedding Surrogate Fails (+ 5.4.1 Recovering the Missing Term) | 5.4 |
| **5.5** | **The Final-Position Case** | was 5.8 |
| **5.6** | **The Proposal Ladder** | was 5.9 |
| **5.7** | **The Likelihood Trap** | was 5.5 |
| **5.8** | **Embedding Anisotropy** | was 5.6 |
| **5.9** | **GFlowNet Fine-Tuning and the Amortized Energy** | was 5.7 |
| 5.10 | Extension: Constrained Generation | 5.10 |

Pure block move of the source; no section was rewritten to accommodate it. Cross-reference
directions were re-checked: the brevity slope is still established (5.7) before the GFlowNet
section that consumes it (5.9); the anisotropy forward reference from 5.1 still points
forward; 5.3.3's "Section 5.7 will show" is still a forward reference; and the final-position
and ladder sections cite only material that now precedes them. Zero undefined references.

Two transitions were rewritten for the new order: the chapter opener, which now says the RQ1
answer is developed without interruption and that the section turns afterwards to the results
standing apart from that argument, and the close of 5.4.1, which now hands off to the exact
check at the final position and then to the ladder.

### Abstract

Recompressed to hold the one-page limit after the 3A, 3B and 3C rewrites lengthened it.
Verified: page 4 of the PDF is the table of contents.

### Gates after this round

latexmk exit 0; 130 pages; 0 undefined references or citations; 0 multiply-defined labels;
0 float promotions; max overfull hbox 20.84pt; 54 bibliography entries, all cited; abstract on
one page; numbers diff RESULT: ALL OK. All ten core floats remain in the body of Chapter 5
(pages 45 to 86): proposal sharpness p47, chain statistics p55, main ablation p55, forest plot
p56, MH-fix p59, gradient-free baselines p60, token-indicator correlation p67, token-indicator
sweep p68, final-position p72, conditioning ladder p76.

### Not applied, with reasons

- **evaluation5 section 6 and 7, "the conclusion still spans eleven pages, 94 to 104", cut it
  by 3 to 5 pages.** The conclusion is pages 96 to 99, three pages; pages 100 to 106 are the
  bibliography. This is the third evaluation to make the same miscount (evaluation1 read it as
  eight pages, evaluation4 as eleven), and it is recorded here so a fourth reader does not
  repeat it. There is nothing to cut.
- **8A, remove the proposal-mapping footnote.** The author's standing constraint requires the
  proposal mapping to survive; it is already compressed to a footnote.
- **8B, shorten the roadmap; 8C, demote anisotropy from a Results section.** Both are length
  measures, and length work is on hold at the author's instruction.

---

## 2026-07-27 CORRECTION TO THE LENGTH PREMISE (author)

The author states that **the appendix does not count toward the page limit**. This overrides
the Phase 10 brief, which said "The appendix is also in scope for trimming ... since appendix
pages count toward the total", and it invalidates two conclusions recorded above.

Composition of the current 130-page build:

| segment | PDF pages | count |
|---|---|---|
| front matter (title, Erklaerung, abstract, ToC, LoF, LoT) | 1 to 9 | 9 |
| Chapters 1 to 7 | 10 to 100 | 91 |
| Bibliography | 101 to 107 | 7 |
| Appendix A.1 to A.8 | 108 to 130 | 23 |

**Countable length excluding the appendix: 107 pages.** That is eight pages BELOW the 115-page
floor of the stated range, not eleven above the ceiling. The two statements this supersedes:

1. "The hard limit of 115 to 119 pages is NOT met. The document is 126 pages, seven over."
   SUPERSEDED. Under the author's counting rule the limit was already met, and is met now.
2. "Moving a body float to the appendix is NOT a permitted way to shorten the body" and the
   reasoning that relocation buys nothing. SUPERSEDED in its arithmetic: relocation to the
   appendix now reduces the countable length one page for one page. The substantive half of
   that constraint still holds, since a core result named in constraint 6 must keep its float
   in the body for the reader's sake, not for the page count.

No further cutting is required for length. Three items deleted in this pass under length
pressure are therefore candidates for RESTORATION into the appendix at zero cost to the
countable total, which is strictly better than deletion; they are put to the author rather
than reinstated unilaterally, because each was also justified as a repetition fix:

| deleted item | where it was | why it could return |
|---|---|---|
| `tab:crossmodel` and its section, the five-energy summary of linearization $\rho$ and the maximum within-strategy trap correlation | body 5.8 | a genuine reader aid; it was cut because both columns appear elsewhere, not because it was wrong |
| `fig:lasttoken`, gradient norm and independence-MH acceptance against downstream context | body, after Table 11 | cut as duplicating the lower block of Table 11; as an appendix figure the duplication costs nothing |
| the harness paragraph: job manifest, lock protocol, stranded locks, per-configuration dispatch throughput | 4.7 closing paragraph | removed from Methodology at the author's instruction; the IMS guidelines explicitly permit implementation detail in an appendix, so an "Implementation and Harness" appendix subsection is a better home than a `%` comment |

---

## 2026-07-27  PART 6 RE-VERIFIED: the proposal against the FINAL thesis

The Phase 10 verification of the proposal was performed before the evaluation5 round, which
reformulated RQ1, split RQ3, moved constrained generation to an extension, renamed the central
derivative, reordered Chapter 5, and softened the conditioning claim from an isolation to an
ordering. The proposal is therefore re-verified here against the thesis as it now stands.
`proposal.tex` and `ref.bib` are unchanged since the Phase 9 amendment (`git diff` empty), and
the file compiles to 12 pages with zero undefined references or citations.

**VERDICT: NO CHANGE.** One sentence is flagged as the one an examiner comparing the two
documents will land on, with the reasoning for leaving it alone given in full.

### Contingency section, sentence by sentence

| # | sentence (abridged) | check against the final thesis | verdict |
|---|---|---|---|
| S1 | "The programme above presupposes that the frozen model's gradient supplies a usable search direction, so that a faithful Langevin sampler can first be built and then controlled." | The thesis scopes its null to the INPUT-EMBEDDING gradient, so the question is whether the proposal's unqualified "the frozen model's gradient" now over-reaches. It does not: the proposal's own Methods section fixes the object at line 401, "we utilize gradients of the energy function with respect to the continuous token embeddings", which is exactly the derivative the thesis tests and exactly the one it finds to fail. | CONSISTENT |
| S2 | "Should that presupposition fail ... the thesis will be redirected rather than abandoned." | This is what happened, and 6.4 records the turn. | CONSISTENT |
| S3 | "Under that branch it will turn to mechanistic diagnosis, treating a negative answer as a finding to be characterized ... established by direct measurement." | The thesis is that diagnosis, now arranged as an explicit elimination of three hypotheses. | CONSISTENT |
| S4 | "The instruments for that diagnosis are named here as planned tools rather than improvised later." | Verified against the thesis: all four named instruments were built and reported. | CONSISTENT, and to the proposal's credit |
| S5 | four instruments: direction-versus-magnitude ablation; linearization test as a function of embedding distance; MH acceptance decomposed into target and proposal terms; whether maximizing the likelihood corresponds to good text | (a) Section 5.3, Table 4. (b) Section 5.4, Figure 5, the within-bin correlation against distance. (c) Section 5.2, target $+4.60$ against proposal $-1325$. (d) Section 5.7, the likelihood trap. All four ran; none was improvised. | CONSISTENT |
| S6 | "the diffusion discussion ... is promoted into an explicit fallback hypothesis and positive control: \emph{if} the difficulty is that the autoregressive likelihood was never trained to provide a score, \emph{then} a score-trained diffusion model on the same tokenizer should supply the local direction the autoregressive gradient lacks, and substituting it as the proposal would isolate the training objective as the cause." | See the extended note below. | FLAGGED, NO CHANGE |
| S7 | "The amortization step of RQ3a folds into the same branch, since a GFlowNet that only ever evaluates the reward and never differentiates it tests whether learning to sample sidesteps a gradient pathology; a trained proposal is therefore available both as an efficiency measure and as a diagnostic instrument." | Matches the thesis's RQ3a and RQ3b. The proposal's original framing of RQ3a was efficiency; this sentence already reserves the diagnostic reading, which is the one the thesis took, so the shift is licensed rather than contradicted. | CONSISTENT |
| S8 | "Two reservations are stated in advance." | -- | CONSISTENT |
| S9 | "First, the datasets and metrics may be adjusted as the diagnostic questions sharpen, as the evaluation plan of Section 4.3 already anticipates in reserving the choice of the most relevant subset." | The thesis did adjust both, and 6.4 accounts for every dropped deliverable by name (RealToxicityPrompts, MAUVE and Self-BLEU, the 70B judge, posterior coverage, CommonGen). Consistent with the thesis. One INTERNAL looseness in the proposal, noted and not touched: the cross-referenced section reserves the choice of \emph{metrics} ("we will select the most relevant subset"), not of datasets, so the appeal to it for datasets is slightly wider than the section it cites. This is a proposal-internal matter and not a conflict with the thesis. | CONSISTENT |
| S10 | "the research questions will be sharpened into diagnostic form while preserving their substance: RQ1 becomes whether and why the gradient is usable, RQ2 becomes whether steering is possible on the landscape as it stands, and RQ3 becomes whether amortization repairs it" | All three land, and two land more precisely than promised. Thesis RQ1 is "whether and why", plus the constructive clause "which alternative local quantities do". Proposal RQ2, the steering question, is answered as extension E; the mapping is explicit in the 1.4 footnote, and this sentence is one reason that footnote is worth keeping. Proposal RQ3 is answered as RQ3b, with RQ3a split off, which is a refinement of the promise rather than a departure. | CONSISTENT |

### S6: the flagged sentence, in full, and why it is not amended

Exact sentence, as it stands in `proposal.tex` line 588:

> "Under this branch the diffusion discussion of the competitive-landscape section is promoted
> from a competitor to be avoided into an explicit \emph{fallback hypothesis} and positive
> control: if the difficulty is that the autoregressive likelihood was never trained to provide
> a score, then a score-trained diffusion model on the same tokenizer should supply the local
> direction the autoregressive gradient lacks, and substituting it as the proposal would isolate
> the training objective as the cause."

Three things are true of it at once, and they have to be separated.

1. **The consequent is confirmed.** A score-trained diffusion model on the GPT-2 tokenizer does
   supply a usable local direction, and substituted as the proposal inside the thesis's own
   exact-energy chain it lifts exact recovery from $0.0$ to $39.0$ percent (Section 5.6.1).
2. **The antecedent is disconfirmed.** The thesis finds the difficulty is not that the objective
   withholds a score: RoBERTa, never score-trained, reaches $44.5$ percent, and the frozen
   autoregressive model's own conditional reaches $23.5$ percent read from the output side.
   A conditional whose antecedent turns out false is not falsified by that, and the contingency
   section explicitly invited the test that decides the antecedent.
3. **The final clause overstates what the design could deliver, and the thesis says so.** The
   clause claims the substitution "would isolate the training objective as the cause". Section
   5.6.1 states the opposite about that design: SEDD differs from GPT-2 Large in objective,
   conditioning direction, scale and corpus at once, so it "cannot by itself separate score
   training from bidirectional conditioning", and it took the RoBERTa control of Section 5.6.2
   to separate them.

Item 3 is the only place in the proposal where a clause conflicts with a statement in the
thesis rather than merely being a hypothesis the thesis disconfirms. It is nonetheless left
unamended, for three reasons.

- It is a **plan**, in future conditional voice, describing what an experiment was expected to
  show. Rewriting it to already know about the conditioning confound would credit the proposal
  with foresight it did not have and would misrepresent the record. A proposal being naive
  about a confound that the thesis then identifies is not an error in the proposal; it is the
  thesis doing its job.
- The correction already exists **in the right document, at the right place**. Section 5.6.1
  bounds the attribution before reporting the result, and Section 5.6.2 runs the control that
  removes the confound. Moving that insight backwards into the proposal would duplicate it and
  weaken it.
- The proposal has already been amended once, in the Phase 9 pass, and it is a submitted
  document. A second amendment that encodes a later finding is a worse outcome than a flagged
  historical record.

What this means practically: if an examiner reads the two documents side by side and asks about
this sentence, the answer is that the thesis tested the conditional the proposal set out,
confirmed its consequent, disconfirmed its antecedent, and found in the process that the design
named in the proposal was not by itself identifying, which is why the masked-language-model
control was added. That answer is already written into Sections 5.6.1 and 5.6.2.

### Outside the contingency section: one emphasis shift, no contradiction

`\section{Goal and research questions}` line 332 says "the study explores the internal mechanics
of the Llama 3 (8B) model, specifically focusing on its gradient signals and embedding
distributions". In the thesis, Llama-3 8B is the cross-architecture control and the SFT'd GPT-2
Large is the reference energy. The proposal's promise is nonetheless kept: Llama-3's gradient
signals are measured (linearization $\rho = 0.021$; the token-indicator substitution recovering
$41.0$ percent against $0.0$) and so are its embedding distributions (nearest-neighbour $0.585$,
pairwise $0.835$, mean pairwise cosine $0.0185$). The proposal does not say "exclusively", the
sentence is in proposal voice, and 4.2 explains the division of roles. Recorded as an emphasis
shift, not a contradiction, and not amended.

### Diff summary

None. `proposal.tex` is byte-identical to its Phase 9 state; the only artefacts touched are the
regenerated build files. Compiles clean at 12 pages, 0 undefined references, 0 undefined
citations, max overfull hbox 33.83pt (below the 40pt gate).

---

## 2026-07-27  AUTHOR ROUND 3: code terminology and artifact references removed from the PDF

Two instructions: strip code-level terminology from the rendered text (the author cited the
`collect_traces` mention on the appendix trajectory page and `dls_policy_gn_mh` in Section 5.2),
and strip every reference to files that will not be submitted, since only the thesis PDF is
handed in. Then sweep the whole document for anything similar.

A full inventory was taken first, over `\texttt{}` uses and over file-extension, script-name and
artifact words in every non-comment line, rather than fixing only the two pages named.

### Code identifiers removed from the rendered text

| what | where | now reads |
|---|---|---|
| `dls_policy_gn_mh` | 5.2, the discrete-sampler acceptance sentence | "in the policy configuration with the correction and gradient normalization both enabled" |
| `cls_policy_gnoff_mh` | caption of the acceptance figure | "with gradient normalization off and the correction enabled, for that single configuration rather than pooled" |
| `collect_traces` | A.4, the shared-source-sentence explanation | "the trace-collection procedure" |
| `np.random.default_rng(0)`, `default_rng(0)` | A.4, A.7 opener, guided-examples caption | "a fixed random seed of zero" |
| `np.random.default_rng(0).choice(200, 10, replace=False)` | showcase selection policy | "ten indices sampled without replacement from the two hundred sequences of the reference set ... under a fixed random seed of zero, then sorted" |
| `data_seed` | showcase selection policy | "corruption seed zero" |
| `no_grad` | A.5, the calibration-clean argument | "seen the split only for an accuracy readout with no parameter update" |
| `--mh_exact_all_arms` | 4.3, the corrected-mode disclosure | the flag name dropped; the mode is described |
| `len_beta`, 17 occurrences across 4.5, 5.9 and three table blocks | GFlowNet variant names | promoted to thesis notation, $\beta_{\text{len}}$, so the variants read "GFlowNet, $\beta_{\text{len}} = 0$, 500" |
| `lm_only`, `cons_only`, `full`, `random`, `cons_random` | 4.6, 5.10, Table 14 header and caption, Table 20 rows | renamed to English: fluency-only, constraint-only, combined, fully random, randomized-constraint |
| `DLS policy, MH, gn, s50` style row labels | Table 15 | expanded to "DLS policy, corr., grad-norm, 50 steps", with the caption defining "corr." and "grad-norm" |
| `Sequence idx 3` | showcase blocks | "Sequence 3" |

### Artifact and file references removed

The thesis no longer promises or names anything outside the PDF.

- **A.2 opening** dropped "drawn from the aggregated results file `final_kl_by_model.csv`".
- **A.2 table caption** dropped "from `final_kl_by_model.csv`".
- **A.2 closing paragraph** was "The full grid is part of the released artifact set: under the
  code-availability statement of Section 4.8, the complete per-configuration results, the
  aggregated `final_kl_by_model.csv` from which this table is drawn, and the artifact map are
  retained with the code, so the entire grid can be regenerated ...". It now explains what the
  eight rows are a subset of and notes that the arithmetic, not the table, establishes the count.
- **A.2 count sentence** "verified against the result folders, which hold 29 configurations for
  GPT-2 Large" became "counting the completed runs confirms it: 29 configurations for GPT-2
  Large".
- **4.8** was "The sampler and diagnostic code is retained with the thesis together with the
  configuration definitions ... and the accompanying documentation links each table and figure to
  the result file it draws from, so every quantitative claim is traceable from the number in the
  text to the file on disk." It now says that every configuration is specified by the design axes
  together with the corruption seed and the schedule, and that each claim is recorded against the
  run that produced it, so any individual result can be regenerated from the description given in
  the thesis. The reproducibility statement survives; the promised deliverables do not.
- **"retained in the source of this document"**, used three times to explain that unshown draws
  still exist somewhere, became "are not shown". With a PDF-only submission the earlier phrasing
  pointed at something the reader cannot open.

`% SOURCE:` provenance comments in the `.tex` files are untouched. They never render, so they do
not reach the submitted PDF, and they remain the traceability trail for the author.

### Verification, against the rendered PDF rather than the source

`pdftotext` over all 130 pages, searched for: `.csv`, `.json`, `.py`, `.npz`, `README`, "artifact
map", "released artifact", "code availability", `np.random`, `default_rng`, `no_grad`,
`collect_traces`, `len_beta`, `cons_only`, `cons_random`, `lm_only`, `_gn_mh`, `gnoff`,
`data_seed`, `verify_equivalence`, `base_sampler`, `run_experiment`, `mask_indices`, "source of
this document", "retained with the", "traceable", `mh_exact`, `idx`, "on disk", "repository",
"supplementary", "with the code", "recorded seeds", "kl-baselines", "aggregated results file".
**Every count is zero.** The only remaining monospace in the document is the garbled-byte marker
`[?]` and the one infilled-span example, both of which are typesetting conventions rather than
code.

### Gates
latexmk exit 0; 130 pages; 0 undefined references or citations; 0 multiply-defined labels; 0 float
promotions; max overfull hbox 20.34pt (the widened Table 15 briefly pushed this to 30.95pt and the
row labels were abbreviated to bring it back); abstract on one page; 54 bibliography entries, all
cited; numbers diff RESULT: ALL OK.

---

## 2026-07-27  NUMERIC AUDIT of the Llama cross-model paragraph (author query)

The author questioned the figure $3.898$ in the closing paragraph of 5.4.1 and then the $4.108$
in the paragraph after it. Both were traced to source. **The $4.108$ is correct. The $3.898$ is
correct as a number but was being used to support a comparison the data does not support**, and a
second, larger problem was found in the same sentence.

### What the sentence said

> "on Llama-3 8B the token-indicator proposal recovers $41.0$ percent at a final KL of $1.908$,
> the lowest divergence any proposal in this study attains, against $0.0$ percent and $3.898$ for
> the input-embedding gradient on the same model and sequences."

### Finding 1: the comparison is not configuration-matched, and no matched comparison exists

| arm | source | configuration | result |
|---|---|---|---|
| token-indicator | `xm_onehot_llama_sharp.json` | $\epsilon: 10.0 \to 0.1$, $T = 1.0$ | 41.0%, KL 1.908 |
| input-embedding | `llama3-8b.dls.policy.mh.gn.free.s50.json` | $\epsilon: 0.1 \to 0.001$, $T = 5.0$ | 0.0%, KL 3.898 |

The two cells differ in step size and temperature as well as in the derivative. An exhaustive
search of `results/grid/**` for a Llama input-embedding run at $T = 1.0$ returns **none**, so the
matched comparison was never run and cannot be recovered from stored data. "On the same model and
sequences" was true and therefore not false, but it read as a matched pair and it is not one.

This matters more than usual here because the *very next paragraph* is about exactly this
confound, that sharpness from the distance term and sharpness from the surrogate are different
things. The section was arguing against its own preceding sentence.

### Finding 2: the superlative broke the no-shared-axis rule

"the lowest divergence any proposal in this study attains" ranked a Llama divergence against the
GPT-2 family. Section 4.2 states that Llama "is never placed on a shared numerical axis with the
GPT-2 family", and 5.3 repeats it. The lowest GPT-2-side divergence is 3.229; the ladder's best
arm is RoBERTa at 2.737, also on the GPT-2 energy. Ranking 1.908 above those is precisely the
comparison the thesis forbids elsewhere. The superlative is removed.

### Finding 3: the one matched Llama cell was omitted, and it cuts the other way

At Llama's calibrated setting the two surrogates *are* matched: same model, same sequences, same
step size and temperature, differing only in the derivative. There the token-indicator proposal is
**marginally worse**, KL 4.108 against 3.898, with both at 0.0 percent exact. That is the honest
cross-model datum and it was not stated. It is now, in the paragraph that follows, because it
strengthens rather than weakens the section's argument: it shows that on Llama the 41.0 percent is
bought by the derivative and the configuration together, not by the derivative alone.

### Finding 4: the GPT-2 comparison, by contrast, IS matched

Table 8 compares the two surrogates cell for cell over one $(\epsilon, T)$ grid, same corrupted
sequences, correction on, gradient normalization off. At $\epsilon = 10.5$, $T = 1$ the
token-indicator arm recovers 40.0 percent and the input-embedding arm 0.0. There the derivative is
the only variable. The distinction between the strong GPT-2 evidence and the weaker Llama evidence
is now drawn explicitly wherever the pair of numbers appears.

### Edits applied

- **5.4.1**, closing paragraph rewritten: the superlative removed; the Llama reference restated as
  "every input-embedding arm in Llama's archived grid recovers 0.0 percent, its best cell reaching
  a KL of 3.898"; and a new paragraph states both qualifications, the configuration mismatch and
  the off-axis divergence.
- **5.4.1**, following paragraph: the matched calibrated cell added, 4.108 against 3.898 with both
  at zero, and the closing sentence now says why the cross-model claim is conditional.
- **7, conclusion**: "against at most 2 percent for the input-embedding gradient anywhere in the
  sweep" became a split statement, cell-for-cell on GPT-2 and "every input-embedding configuration
  that was run" on Llama.
- **6.4, limitation 1**: gains the second qualification, that the cross-architecture evidence is
  weaker than the GPT-2 evidence because no matched Llama run exists.
- **Beamer slide 13**: "Nothing retrained; only the derivative changed" was false for the Llama
  column. Now "GPT-2 Large, compared cell for cell over one $(\epsilon, T)$ grid. Llama-3 8B
  reaches 41.0% where every input-embedding configuration run gives 0.0%."
- **TALK_TRANSCRIPT.md**: the Say block no longer implies a matched Llama pair, and a new
  "If asked here" carries the 4.108-against-3.898 cell.
- **TALK_QNA.md**: new entry N1b, "Is the Llama-3 41 percent a matched comparison?", which
  volunteers the matched cell and the off-axis caution.

The abstract and 6.1 were checked and needed no change: neither asserts a matched Llama
comparison, and 6.1 points at the section that qualifies it.

### Gates after the correction
latexmk exit 0; 131 pages; 0 undefined references or citations; max overfull hbox 20.34pt;
abstract on one page; numbers diff RESULT: ALL OK. Beamer exit 0, 25 pages.

---

## 2026-07-27  AUDIT of Table 11, the conditioning ladder (author query)

The author asked whether every arm of Table 11 is single-token recovery or something else. Traced
to source: `results/revision/rev_sedd_hybrid.csv` (1000 rows, five arms x 200 sequences),
`rev_mlm_control.json`, `rev_mlm_uniform.json`, and the code paths in
`diagnostics/run_sedd_cap.py` and `diagnostics/run_mlm_control.py`.

### Answer: yes, all seven rows are single-token recovery

Verified rather than assumed:

- Both harnesses call the grid's own `build_corruption` with `num_masks = 1`, so exactly one
  interior position is corrupted per sequence, chosen by the same deterministic rule seeded on the
  sample index.
- In the hybrid CSV, every one of the 200 sample indices has **exactly one** masked position, and
  all arms agree on it: 0 samples where the arms disagree.
- All arms: $n = 200$, the same WikiText-2 sequence set, corruption seed 0, 50 steps, and the same
  exact GPT-2 Large sequence-log-likelihood energy for acceptance.

So the answer to the question as asked is that the table is homogeneous in task.

### But the audit surfaced an over-claim in the caption, now corrected

The caption said "Every proposal run inside the same Metropolis--Hastings chain ... **Only the
proposal differs.**" That is exact for five of the seven rows and **not** for the other two.

| rows | chain |
|---|---|
| uniform draw, AR conditional, SEDD-small, SEDD-medium, RoBERTa | position-wise **independence** Metropolis-Hastings, differing only in the proposal distribution |
| input-embedding gradient, norm-matched random | the thesis's own `DiscreteLangevinSampler`, that is the Equation (6) proposal with the correction, $\epsilon: 10.5 \to 0.1$, $T = 5$, gradient normalization on |

These differ in the transition **kernel**, not only in the proposal. The code confirms it
independently: the hybrid CSV records an acceptance rate for the five independence arms and leaves
the field empty for the two Langevin arms, because they are not independence chains.

This matters for the same reason the Llama audit did: the caption asserted a controlled comparison
that is controlled across five rows and not across seven, and the section text repeated it
("differences between arms are differences in what the proposal can see, and nothing else").

### Edits applied

- **Ladder opener**: now states the task explicitly, single interior token, same deterministic
  corruption rule and seed, same 200 sequences, 50 steps, same accepting energy; then states that
  five arms are independence samplers differing only in the proposal, that the two gradient arms
  are the Langevin sampler and therefore differ in kernel as well, and why they are nonetheless
  included, namely to reproduce the null from separate code on this sequence set.
- **Table 11 caption** rewritten to say the same thing, and retitled from "in the same exact-energy
  chain" to "accepted by the same exact energy", which is what is actually common to all seven.
- **The paragraph introducing the table** now describes the two gradient arms as the reference the
  ladder is read against rather than as rungs of it.

The claim the ladder supports is unaffected: the five controlled rows still run 0.5, 23.5, 38.5,
39.0 and 44.5 percent under a common kernel, and that ordering is the finding. What changes is that
the two Langevin rows are no longer presented as though they belonged to the same controlled series.

### One quirk recorded, not used anywhere

In `exp_hybrid_refs` the `top5` column is computed from the autoregressive left-context conditional
for *every* arm, so it is a property of the sequence rather than of the arm, which is why the
gradient and left-conditional arms all report 34.5. No surviving table or sentence in the thesis
quotes a top-5 figure from that file; the removed `tab:diffusion-hybrid` did, and the 34.5 values
now in the text are the final-position experiment's exact-match rates from a different run.

### Gates
latexmk exit 0; 131 pages; 0 undefined references or citations; max overfull hbox 20.34pt;
numbers diff RESULT: ALL OK.

---

## 2026-07-27  TABLE CAPTIONS cut to the Ioanna house style (author-approved)

Applied after an author review of the exact before/after list. Three approved changes: the 20
caption rewrites, the four abbreviation decoders retained, and the unreferenced showcase table
given a prose reference.

### The rule, and a correction to how it was justified

The standing rule is `PROMPT_PHASE8` item 12 and the style memo at `REVISION_LOG` ~2276: naming
clause, plus a legend gloss only if needed; interpretation lives in the prose. The author added a
stricter condition, that a caption must not repeat a fact the prose already states.

CORRECTION TO AN EARLIER CLAIM IN THIS LOG AND IN THE REVIEW. In proposing the first version I
wrote that "the IMS checklist wants tables readable as objects" and used it to justify keeping
caption text. That is wrong. `Checklist_Masterthesis.pdf` mentions tables exactly once, as "Table
of contents", and says nothing about tables, figures or captions. The document that bears on this
is `Guidelines_for_academic_thesis_writing_at_the_IMS.pdf` section 2.5, whose three relevant
bullets are: tables prepared so the reader can easily see how they help answer a research
question; the reader able to repeat the experiments from the available information; and "Reader
not left alone with interpreting the results. **I talk them through each table and depiction with
text.**" The third assigns interpretation to the prose explicitly, so short captions are what the
guidelines describe rather than a departure from them.

### Redundancy audit

Every caption clause was checked against the prose with comments, captions and float bodies
stripped. Confirmed already in the prose and therefore cut: `50{,}257`, `10.8249`, "temperature of
5.0", "standard deviation across the vocabulary", the Equation (6) reference, `n = 200`,
"grad-norm-preserved ... coincide", "gradient normalization disabled", "50 steps", "policy minus",
"three sharpest", `n = 50`, "norm-matched random", "never take its gradient", "masked-recovery
metric", `400{,}000`, `1.94`, "same (eps, T) grid", "changing nothing else", "scored terms",
"calibrated configuration", `1050`, all three divergence column names, `n = 150`, "percentage
points", "fifty steps", "two hundred WikiText-2 sequences", "same corruption seed", the S/M/T/k
definitions, `n = 300`, "trust region", "span negative log-likelihood", `71.7`, and the g/j gloss.

Not in the prose and therefore kept: "lower is better", `100{,}000`, `0.45`, `T = 1`.

### The four decoders, kept, on corrected reasoning

The first justification for keeping them was the false checklist claim. Re-running the search with
float bodies properly stripped gives the real reason: `Input emb.`, `Token ind.`, `corr.`,
`self gain`, `judge gain` and `Upper block` appear **nowhere in the prose**. They exist only inside
the tables, because the prose names the concepts ("the relaxed token-indicator derivative", "the
guiding classifier's own verdict") and never the printed abbreviations. Removing the decoders would
leave those strings undefined anywhere in the thesis, which is a real failure of the "easily
understand" bullet. This is also exactly what Ioanna does ("ASV emb. denotes ASV embedding
similarity", "DR=dECE speaker rank"). Only `grad-norm` is defined in prose.

### Result

| | before | after | Ioanna |
|---|---|---|---|
| mean words | 52.2 | **18.7** | 32.4 |
| median | ~47 | **15** | ~32 |
| max | 115 (`tab:gfn-unify`) | **47** (`tab:onehot`) | 64 |

20 of 22 captions rewritten; `tab:diffusion-lin` and `tab:full-grid` unchanged, their content
being legend that the local prose does not carry. One short caption changed: `tab:mhfix` read "The
apparent gradient penalty is an artifact of the asymmetric correction", a finding, and therefore
put a conclusion into the List of Tables; it is now "Final KL under two treatments of the
reverse-proposal term."

No prose was added or altered to compensate. Every cut clause was verified present in the
surrounding text before removal.

### The separate guideline violation, fixed

`tab:showcase-infill` was **never referenced in the prose**, the only table in the thesis with zero
`\ref`. That contravenes "I talk them through each table and depiction with text" independently of
caption length. The selection-policy paragraph in A.7.2 now opens the second sentence with
"Table~\ref{tab:showcase-infill} gives them." A re-check confirms every one of the 22 tables is now
referenced at least once.

### Gates
latexmk exit 0; **130 pages**, down from 131; 0 undefined references or citations; 0
multiply-defined labels; max overfull hbox 20.34pt; abstract on one page; List of Tables 22
entries, one line each; numbers diff RESULT: ALL OK.

---

## 2026-07-27  FIGURE CAPTIONS cut to the same rule (author-approved)

Same treatment as the tables, applied after an author review of the exact before/after list: cut
any clause the prose already states, keep decoders that appear nowhere else.

### Redundancy audit

Confirmed in the prose and therefore cut: the three panel quantities of Figure 1, "within-cell
against boundary-crossing", "35 configurations", "$\pm 0.327$", `400{,}000`, `2.35`, the two
acceptance rates $3.7\%$ and "essentially never", the $-1.12$ slope with its censoring caveat and
the $0.997$ cap fraction, "unlike the enabled case ... the three methods coincide", "symmetric-log",
"identically zero", "$3.3\%$ ... illustration only", and six occurrences of the trailing "This
figure supports Section~X", a construction Ioanna never uses and which duplicates the prose
sentence that already points at each figure.

Not in the prose and therefore kept: every line-style, panel and colour decoder (solid versus
long-dashed, marker shape, left versus right panel, grey cloud, open circle, filled dot, star, red
ticks, the darker colour marking intervals that exclude zero), "percentile-bootstrap", `n = 200`
per row, the $0.1$ correlation threshold, and "for that single configuration rather than pooled",
which is the Alarm-1 correction and exists only in that caption.

### Result

| | before | after | Ioanna |
|---|---|---|---|
| figure captions, mean | 42.3 | **31.7** | ~15 |
| median | 44 | **29** | ~11 |
| max | 67 | **55** | 28 |

14 of 15 rewritten; `fig:dls-traj-100` unchanged. The outcome sits above Ioanna's figure norm and
that is deliberate: our figures are multi-panel with line-style and marker encodings that hers are
not, and every surviving word is a decoder that appears nowhere else in the document. Cutting
further would remove the only definition of what a dashed line or a red tick means.

CORRECTION TO THE PROPOSAL AS FIRST WRITTEN. In the earlier round I recorded that `fig:forest`
would need a compensating sentence added to Section 5.3. Under the stricter rule it does not:
"percentile-bootstrap", the $n = 200$ per row and the darker-colour gloss are absent from the prose,
so they stay in the caption where they belong, and only "35 configurations" and the margin value
were redundant. No prose was added anywhere.

### The one prose edit

Appendix A.4 said of the PCA panel that "it is shown for intuition only, and **its own caption
records why** it cannot be the evidence: the two components capture just $3.3\%$ ...". Since the
caption no longer records it, the pointer became false. The clause now reads "it is shown for
intuition only, and it cannot be the evidence: the two components capture just $3.3\%$ ...". Five
words; the fact itself was already in that sentence.

### One pre-existing gap found and NOT yet fixed

`fig:mh-accept` is **never referenced in the prose**. It is the only figure in the thesis with no
`\ref`, exactly the condition found for `tab:showcase-infill` in the previous round, and it
contravenes the same guideline bullet, "I talk them through each table and depiction with text".

Verified against the pre-pass backup of `chapters/`: the figure had zero prose references there
too, so this is pre-existing and was not introduced by removing its "This figure supports Section
5.2" tail. That tail was a caption pointing at the prose, which is the wrong direction and does not
satisfy the guideline.

Left unfixed pending the author's decision, since the standing instruction is to ask before
editing. The fix would be one clause in Section 5.2, where the acceptance split is already
discussed, for instance appending "(Figure~\ref{fig:mh-accept})" to the sentence reporting the
$0.03$ to $0.63$ percent within-cell and $3.7$ to $8.6$ percent boundary-crossing rates.

### Gates
latexmk exit 0; 130 pages; 0 undefined references or citations; 0 multiply-defined labels; 0 float
promotions; max overfull hbox 20.34pt; abstract on one page; numbers diff RESULT: ALL OK. All 22
tables referenced in prose; 14 of 15 figures referenced, the exception recorded above.

---

## 2026-07-27  AUTHOR ROUND 4: figure legends, float sizing, Table 21, and A.8

Six author instructions, all applied.

### 1. `fig:mh-accept` referenced in prose

The gap recorded at the end of the figure-caption round is closed. Section 5.2 now ends its
acceptance-rate sentence with "(Figure~\ref{fig:mh-accept}, Appendix~\ref{app:figures})", placed
where the $0.03$ to $0.63$ percent within-cell and $3.7$ to $8.6$ percent boundary-crossing rates
are reported, which is the sentence the figure illustrates.

### 2. Figures 1, 5, 13 and 14: legend text removed from the captions

Verified by rendering each page first: all four plots carry their own in-image legend, so the
caption was restating a key the reader can already see. Figure 5's plot even prints "mean candidate
distance = 2.35" and "first bin with rho < 0.1, at 0.89" inside the axes, which the caption was
duplicating in words.

| figure | removed |
|---|---|
| 1 `fig:dls-traj-50` | "Solid lines are runs with the correction, long-dashed lines without it. Marker shape distinguishes the three proposal methods where their curves coincide." |
| 5 `fig:lin-radius` | "Dashed line: the mean linearization-candidate distance. Dotted line: the smallest binned distance at which the correlation falls below $0.1$." |
| 13 `fig:dls-traj-nogn` | the line-style clause, keeping "Panels and axes as in Figure~\ref{fig:dls-traj-50}" |
| 14 `fig:traj-distance` | the solid/dashed/red-tick key and the not-drawn note |

Figure captions now average **22.5 words** (from 31.7 after the previous round, 42.3 originally),
median 21, max 55. Figures 2 and 15 keep their keys because those plots have no in-image legend.

### 3. Figures 7 and 8 onto one page

Both were at $0.92$ textwidth and each took a page of its own. Sized backwards from the text
block: sources are $384 \times 272$pt and $415 \times 282$pt, so at $0.70$ textwidth ($299$pt)
they are $212$ and $204$pt tall, and with both captions the pair occupies about $460$pt of the
$591.5$pt height. They now share page 108. On-page scale is $0.78$ of source, so the axis labels
are unchanged in relative size and remain legible in print. Verified by rendering.

### 4. Table 21 matched to Table 22 and fitted on one page

`tab:gprime-examples` was the only table in the document at `\normalsize`; Table 22 uses
`\footnotesize` and Table 20 `\small`. Wrapped in `\footnotesize` to match Table 22. That alone
still spilled three lines onto a second page, so three further changes, none of which touches a
datum:

- The verdict marker was set in `\small`, which inside a `\footnotesize` table is **larger** than
  the body text. Changed to `\scriptsize`.
- The marker was preceded by `\newline`, forcing an extra line for each of the eight generation
  rows. It is now inline and flows with the text.
- The label column went from $0.16$ to $0.11$ of the line width, giving the text column $0.83$,
  and the extra $2$pt skip after each pair was dropped.

The table now fits on one page. Page count 130 to **128**.

### 5. Appendix A.8, Use of AI Tools, rewritten to the author's specification

Order and content as dictated: the infrastructure use first and named as the most substantial;
then debugging, stated as checking that the implemented samplers agreed with the mathematics they
are derived from and locating where they did not, explicitly after the experimental code had been
written by the author; then the manuscript work, the agentic coding interface used to edit the
LaTeX sources and lay the document out quickly after the text had been written by the author, with
paraphrasing and copy-editing folded into the same clause. The concept-clarification paragraph is
removed. No gendered pronoun appears; every reference is to "the author". The closing
responsibility paragraph is unchanged.

### A bug in this log's own verification, corrected

The "never referenced in prose" check used in the previous two rounds stripped LaTeX comments with
`%.*$`, which also truncates any line containing an escaped percent sign such as `$3.7\%$`. It
therefore reported `fig:mh-accept` as unreferenced even after the reference had been added, because
the reference sits on a line that contains `\%`.

Re-run with `(?<!\\)%.*$` and with float environments stripped rather than caption commands:

| | figures unreferenced | tables unreferenced |
|---|---|---|
| pre-caption-pass backup | `fig:mh-accept` | `tab:showcase-infill` |
| current | **NONE** | **NONE** |

So the two gaps reported earlier were real, both are now closed, and all 37 floats are referenced
in prose. Any earlier statement in this log that rested on the faulty checker should be read
against this table.

### Gates
latexmk exit 0; **128 pages**; 0 undefined references or citations; 0 multiply-defined labels; 0
float promotions; max overfull hbox 20.34pt; abstract on one page; numbers diff RESULT: ALL OK.

---

## 2026-07-27  SESSION CLOSE-OUT: complete change inventory

Written at the author's request, to capture everything from this session that the entries above
did not record. The entries above cover the substance of each edit; this one covers the file
inventory, the artifacts, the rollback points, the corrections this log made to itself, and the
decisions that were taken but deliberately NOT applied. Audited against `git`, not from memory.

### Rollback points

| commit | when | meaning |
|---|---|---|
| `087ab37` "evaluated" | before this session | state before any Phase 10 work |
| `737a5a4` "beamer and thesis final draft" | 2026-07-27 01:59 | author's mid-session commit; everything from Phase 10 through the talk documents is in it |

Work after `737a5a4` (the code-terminology sweep, the Llama and Table 11 audits, the caption
passes, and author round 4) is uncommitted in the working tree. A copy of `chapters/` as it stood
immediately before the table-caption pass is in the session scratchpad.

### Every file changed in this session

**Thesis source, all 13 chapter files:** `abstract`, `01_introduction`, `02_background`,
`03_related_work`, `04_methodology`, `05_results`, `05a_diffusion_control`, `06_discussion`,
`07_conclusion`, `08_appendix`, `gprime_examples`, `showcase_appendix`, `tab_confusion`.

**Beamer:** `Doc/final/beamer/Presentation.tex`.

**New files created:**
- `revision/replot_forest_chain.py`, the plot-only redraw of Figure 3 from cached statistics.
- `Doc/final/beamer/TALK_TRANSCRIPT.md`, the speaker script.
- `Doc/final/beamer/TALK_QNA.md`, the defence question bank.

**Figures regenerated:** `fig_forest_chain.{pdf,png}` in both `figures/` and `Doc/figures/`. No
other figure file was regenerated; the print-legibility pass changed `\includegraphics` widths
only.

### Deliberately NOT touched, and verified so by `git diff`

- `Doc/final/thesis/thesis.tex`, the master and preamble. The template constraint holds: no change
  to `baselinestretch`, geometry, margins, title page or the Erklaerung.
- `Doc/final/proposal/proposal.tex` and its bibliography. Part 6 was a verification, not a rewrite,
  and the file is byte-identical to its Phase 9 state.
- `references.bib`. No entry added, removed or altered this session; it stands at 54 entries, all
  cited.
- `core/`, `diagnostics/`, `scripts/`, `Methods/`. No sampler, harness or experiment code was
  modified. The only Python written was the new plotting script, which reads a cached JSON.
- `results/`. No result file was written, regenerated or edited. Every number in the thesis still
  traces to the artefacts produced before this session.
- `thesis_questions_knowledge_base.md`, read as the source for the Q&A document and left unchanged.

`refs/evaluation5.md` and `PROMPT_PHASE10_FINAL_REWRITE.md` appear in the session diff but were
added by the author, not by this pass.

### Proof artefacts

69 rendered PNGs now sit in `Doc/final/proofs/`. They are working evidence for specific decisions
rather than deliverables, and they group as follows.

| prefix | what it verifies |
|---|---|
| `fig3_check`, `fig3_after`, `forest_v2`, `forest_final` | Figure 3 overflowing the folio, and the redraw that fixed it |
| `apx_*`, `apxv2_*`, `chk_*` | appendix figure sizes before and after the print-legibility pass |
| `beta_check`, `a2_check`, `a2_v2`, `cons_v2` | the $\beta_{\text{len}}$ notation, and the A.2 and A.6 tables after the artifact-reference sweep |
| `cap_check48`, `cap_check84` | table captions after the caption pass |
| `insp_*` | the four figures whose in-image legends were checked before removing the caption keys |
| `r4_*`, `r5_*` | Figures 7 and 8 sharing a page, and Table 21 fitted to one page |
| `beamer_slide_*`, `bslide_*`, `bs2_*`, `bs3_*`, `bs4_*`, `bfin_*`, `bdone_*` | beamer slides across the two deck passes |

### Corrections this log made to itself during the session

Recorded together because each supersedes something written earlier in this same file.

1. **The length premise.** The Phase 10 report concluded "the hard limit of 115 to 119 pages is NOT
   met, seven over". The author then stated that the appendix does not count, which put the
   countable length at 107 and below the floor. Logged under the correction entry; the conclusion
   and the "relocation buys nothing" reasoning that followed from it are both withdrawn.
2. **The IMS justification for long captions.** I wrote that "the IMS checklist wants tables
   readable as objects" and used it to defend caption text. The checklist mentions tables once, as
   "Table of contents". The Guidelines section 2.5 in fact assign interpretation to the prose.
3. **Figure 3 needing a compensating sentence.** Stated twice in the caption proposals; false under
   the stricter rule, because the bootstrap, the per-row $n$ and the colour gloss are absent from
   the prose and therefore stay in the caption. No prose was added.
4. **The unreferenced-float checker.** Its comment stripper truncated any line containing an escaped
   percent, so it misreported `fig:mh-accept` as unreferenced after the reference had been added.
   Re-run correctly: both gaps found this session were real, and both are now closed.

### Decisions taken and deliberately NOT applied

These are live options, not oversights. Each is the author's call.

| item | status |
|---|---|
| The six length-reduction candidates listed at the end of the Phase 10 report, worth roughly 9.5 pages | not applied; superseded in urgency by the counting-rule correction and then by the author's instruction to prioritise print legibility |
| Moving Section 5.10, the constrained-generation extension, into the appendix | argued against and not applied: it is the only place the thesis attempts the control its title promises, and both external reviews endorse its current placement |
| Restoring `tab:crossmodel`, `fig:lasttoken` and the 4.7 harness paragraph into the appendix, now that appendix pages are free | proposed, not applied, awaiting the author |
| Adding the token-indicator and RoBERTa rows to Tables 21 and 22 | cannot be done from stored data; needs a re-run of two configurations with example logging, which is a GPU job awaiting an explicit go-ahead |
| Classifier-guided steering with a masked-LM or token-indicator carrier | never run; named in 5.10.1 as not tested rather than silently omitted |
| Applying the strict no-repetition rule to the four table decoders as well | offered; not applied, because those abbreviations appear nowhere else in the document |

### Outstanding, awaiting the author

1. The page target, pending the supervisor's reply. Current composition: front matter 9, Chapters
   1 to 7 89, bibliography 7, appendix 23, total 128.
2. Whether to run the two example-logging jobs for the qualitative tables.
3. Whether to reinstate the three items listed above into the appendix.

### State at close

latexmk exit 0 on all three documents. Thesis 128 pages, proposal 12, beamer 25. Zero undefined
references and citations everywhere. Zero multiply-defined labels, zero float promotions, max
overfull hbox 20.34pt against the 40pt gate. Abstract on one page. All 22 tables and all 15 figures
referenced in prose. 54 bibliography entries, none uncited. `revision/numbers_diff_phase6.py`
reports RESULT: ALL OK.

---

# PHASE 11, PART 1: FUTURE-WORK AUDIT AND DEFENSE-EXPERIMENT PROPOSAL (audit only, no GPU, no thesis edits)

Read in the prescribed order: REVISION_RESTRUCTURING.md, the evaluation3-pass entries of
REVISION_LOG.md (Parts A through D5), this file in full (Phase 9 through the 2026-07-27 session
close-out), and the current `Doc/final/thesis/chapters/*.tex`. No experiment was run, no GPU job
was launched, and no `.tex` file was touched. This section is the report Part 1 asks for; it ends
with a numbered decision list and stops there.

## 1. STATE OF PLAY

**Page count.** `pdfinfo` on the current build reports **128 pages**. From `thesis.toc`
(`\contentsline{section}`): front matter pages 1 to 9; Introduction from 10; Background from 17;
Related Work from 27; Methodology from 34; Results from 44; Discussion from 87; Conclusion from
96; Appendix from 107. So: front matter 9 pages, Chapters 1 to 6 (Introduction through Discussion)
pages 10 to 95 (86 pages), Chapter 7 (Conclusion) plus the bibliography together pages 96 to 106
(11 pages, of which the last logged breakdown records the conclusion itself as 3 pages and the
bibliography as 7 to 8), and the Appendix pages 107 to 128 (22 pages). Countable length under the
author's own 2026-07-27 ruling that the appendix does not count toward the limit: roughly 9 + 90 +
7 = 106 pages, eight to nine below the 115-page floor that same ruling references.

**A standing discrepancy, flagged rather than resolved.** This session's own prompt restates "the
hard limit from Phase 10 stands: 115 to 119 pages total... no layout tricks", i.e. counting the
appendix. But the 2026-07-27 entry above ("CORRECTION TO THE LENGTH PREMISE") records the author
overriding exactly that reading, on the grounds that the appendix does not count, which moved the
document from "126 pages, seven over" to "107 pages, eight under the floor". The document has since
grown to 128 total (130 at that point, then 128 after the Table 21 fix), so under the two readings
the current state is either **about 9 to 13 pages over** (counting the appendix, the reading this
session's prompt restates) or **about 8 to 9 pages under** (excluding it, the author's last word on
the question). This is not something to resolve unilaterally here; it is decision 1 below, and it
governs how seriously to take the page-budget section that follows.

**Current structure of Chapter 6 (Discussion).** Five subsections: 6.1 Answers to the Research
Questions (`sec:disc-rqs`), 6.2 One Problem, Four Mechanisms (`sec:disc-unified`), 6.3 Connections to
the Related Work (`sec:disc-related`), 6.4 Scope and Limitations (`sec:disc-scope`), 6.5
**Implications** (`sec:disc-future`, 360 words, about one page). There is no subsection or
paragraph anywhere in the thesis titled "Future Work"; a full-text grep of `Doc/final/thesis/chapters/`
for "future work", "next step", "left for future", "natural extension" and similar returns nothing
except one unrelated methodological aside in 5.3.3. Section 6.5, whose label `sec:disc-future`
is the only trace of the word "future" in the label namespace, is therefore the material that plays
the future-work role, even though its title and content are framed as practical implications of what
was found rather than as a statement of what is proposed next. Chapter 7 (Conclusion) has no
future-work material either; its seven `\paragraph*` blocks summarize what was tested, found and
remains uncertain, and end on "the main practical implication", which restates 6.5's practical
reading rather than proposing new work.

**Exact current wording of Section 6.5** (`Doc/final/thesis/chapters/06_discussion.tex:60-66`):

> \subsection{Implications}
> \label{sec:disc-future}
>
> The most consequential implication is the one the study did not set out to find. An earlier
> reading of these results located the defect in the training objective and predicted that a model
> trained to supply a score would not fail in the same way. A score-trained diffusion model does
> indeed succeed where the input-embedding gradient fails, but the ladder of Table~\ref{tab:mlm}
> shows that the prediction was right for the wrong reason: a bidirectional masked language model,
> never trained with any score objective, does better still, and the frozen autoregressive model's
> own conditional already recovers a quarter of the tokens its gradient never reaches. What
> separates a working proposal from a failing one is the derivative it is built from and the
> context it may condition on, not the objective that trained it.
>
> Two implications follow for practice. The first concerns what inference-time control on a frozen
> autoregressive model can be built from, and it is a correction of the slogan an earlier version of
> this work would have offered. It is not simply that one should evaluate the energy and not
> differentiate it, because a derivative of the same frozen likelihood does work: what one must not
> do is differentiate it in the input-embedding coordinates. Either differentiate the right object,
> taking the relaxed token-indicator derivative whose self term is one forward pass away, or bypass
> the derivative and propose from the model's output side directly; both routes are available on a
> frozen model, and Appendix~\ref{app:cost} shows that both are cheaper than the backward pass they
> replace. The second concerns what a further model change could buy. Since the remaining gap after
> the derivative is fixed is bought by right-context access, the intervention that addresses it is
> not a score-matching fine-tune but any procedure that gives the proposal a bidirectional view of
> the position it is filling, which an off-the-shelf masked language model already supplies at no
> training cost. Only once a landscape is known to be navigable does the additive constraint term of
> the plug-and-play framework become meaningful, since only then is there a fluency energy a sampler
> can follow for the constraint to be added to.

The last sentence of the first paragraph in Section 6.4 ("Scope and Limitations",
`06_discussion.tex:56`) already states the honest status this Part 1 is asked to sharpen: "What it
delivers is a diagnosis of the failure of a prerequisite, with the mechanisms identified, measured
and cross-checked, and with the extension included as evidence of the downstream consequence,
rather than a demonstration of successful control." Item 2 below drafts a tighter version of this
same sentence for the same location, naming the bridge explicitly rather than describing only what
was delivered.

## 2. DRAFT REPLACEMENT (drafted here only; no `.tex` file touched)

The three paragraphs below are drafted to slot into Section 6.5, after its two existing paragraphs,
under a retitled heading. Reasoning for that placement rather than a new subsection is under
decision 2. Each paragraph names the experiment, what it measures, and why it is the next
experiment this specific thesis owes rather than a generic wish; each is cross-checked against
material already in the finalized thesis so it reads as a continuation of an argument already made,
not a new promise.

### (a) Revision scale

> The recovery task studied throughout corrupts exactly one interior token per sequence; the
> `num_masks` design axis was fixed at one across all $145$ configurations, and the proposal ladder
> of Table~\ref{tab:mlm} is single-position for the same reason (Section~\ref{sec:meth-configs}). But
> the motivating scenario of Section~\ref{sec:intro-routes}, a model needing to revise an earlier
> decision once a later one reveals it was wrong, is a multi-position problem by construction, and
> nothing in this thesis speaks to whether the token-indicator ordering established here survives
> when several positions must be revised jointly. The natural next experiment holds the corruption
> machinery fixed and varies only the number of masked positions, $M \in \{1, 2, 4, 8, 16\}$, contiguous
> and scattered, measuring exact and ever recovery and the same contextual-fit KL against the same
> four proposals (input-embedding gradient, token-indicator derivative, the frozen model's own
> conditional, a bidirectional proposal). The open design question, worth stating rather than
> deciding in advance, is whether positions update jointly or by random-scan single-position moves
> within a step; the two differ in whether one revision can see another's outcome before it commits,
> which is exactly the capability Section~\ref{sec:intro-routes} says an autoregressive model lacks.

Verified before drafting: `scripts/run_experiment.py:build_corruption` already takes `num_masks` (CLI
default 1) and selects positions via `rng.choice(valid, size=num_masks, replace=False)`, sorted,
which is a **scattered** draw; there is no contiguous-span option in the function as it stands. A
contiguous variant would be a small, additive change to `build_corruption` (choose a random start and
take `num_masks` consecutive valid positions), not a reimplementation, but it does not exist today
and would need its own equivalence check against the scattered path before use (Part 2 ground rule:
"never reimplement `build_corruption`").

### (b) Transfer to a constrained energy

> The token-indicator identity of Section~\ref{sec:results-onehot} is specific to the likelihood term:
> the closed form $\log p(v \mid \vx_{<i}) + \vg^\top \ve(v)$ exists because the vocabulary index being
> relaxed is the same index the likelihood's own output softmax scores. An arbitrary differentiable
> constraint $C(\cdot)$, such as the sentiment classifier of Section~\ref{sec:results-constrained}, has
> no equivalent decomposition; its "self" term is not recoverable from a forward pass the classifier
> already computes for some other purpose, because nothing else needs that forward pass to exist.
> Section~\ref{sec:results-guided} already states that whether classifier guidance transfers to a
> token-indicator or masked-language-model carrier "is left open"; this is the experiment that
> answers it. The natural architecture is not a single substitution but a composite one: a proposal
> formed from the output-side surrogate for the likelihood term, a shortlist of its top-$k$
> candidates rescored by the constraint classifier on full sequences (since the classifier has no
> local closed form to substitute directly into the proposal), and a Metropolis--Hastings accept
> computed on the exact combined energy of Equation~\eqref{eq:intro-energy}. Running this against the
> existing input-embedding-constraint-gradient arm on the same sequences and the same $\lambda$ grid
> is the test of whether the diagnosis generalizes beyond the likelihood term specifically, or is a
> fact about the likelihood's own parameterization that does not extend to an arbitrary constraint.

### (c) The sampling claim

> Every experiment in this thesis scores a single chain's final state or its trajectory statistics;
> none measures whether the chain returns a genuinely diverse set of acceptable completions rather
> than repeatedly finding one. Section~\ref{sec:disc-scope} already records that "the posterior-coverage
> evaluand was abandoned because the samplers never reached a regime in which coverage is a meaningful
> question", and Section~\ref{sec:disc-rqs}'s RQ2 answer distinguishes a sampler from "an early-stopped
> stochastic optimizer" without ever measuring which one the corrected chain behaves like. That
> distinction is the reason a faithful sampler was worth building in the first place, rather than a
> search procedure with a fluency term attached, and it is the one property of the plug-and-play
> proposal this thesis promised implicitly and never operationalized. The next experiment draws $N$
> generations per prompt from the best chain of experiment (b) at a fixed control level, against a
> compute-matched best-of-$N$ optimizer at the same control level, and reports distinct-$n$, self-BLEU
> and an embedding-based semantic spread, all read at matched adherence so a diversity gain cannot be
> bought by weaker control. Whichever way it falls, it closes the one gap in this thesis's own
> vocabulary between "sampling" and "optimizing" that no measurement here yet fills.

### Honest status sentence for 6.4

Replacing or sharpening the existing closing sentence of 6.4's first paragraph:

> The failure of the plug-and-play construction's central presupposition is diagnosed here, and for
> the likelihood term specifically it is repaired and the repair demonstrated inside this thesis's own
> sampler on its own diagnostic task. The bridge from that repair back to the controllable generation
> the original programme promised, to more than one revised position at a time, to an arbitrary
> constraint rather than the likelihood term alone, and to a measured diversity advantage over search
> rather than an assumed one, is specified in Section~\ref{sec:disc-future} but not crossed here.

## 3. THE EFFICIENCY POINT, verified in code

**Finding: `policy_onehot` is self-plus-gradient, not self-only, and the current code does NOT save
the backward pass.** Read `core/dls.py` and `core/base_sampler.py` directly (not from memory).

- `core/base_sampler.py:get_gradient_and_log_joint` (lines 36 to 72) computes `log_joint`, then
  unconditionally calls `grad_s = torch.autograd.grad(log_joint, s, ...)` at line 54, **before** the
  `return_self_logprobs` branch is even checked. Only after that backward pass does it, under
  `torch.no_grad()`, run one **additional** forward pass (`self.model(inputs_embeds=inputs_embeds)`,
  line 63) to read $\log p(v \mid \vx_{<m})$ at each masked position. So a call with
  `return_self_logprobs=True` costs the same backward pass as a plain gradient call, **plus** one
  extra forward pass. Nothing is skipped.
- `core/dls.py:_step` (lines 41 to 68) builds the proposal logits as `t1 + t2`, where for the
  `policy_onehot` method `t2 = 0.5 * (grad_dot_emb - grad_dot_s) + self._onehot_bonus(self_lp)` (line
  65 to 67), and `_onehot_bonus` (lines 31 to 39) returns `0.5 * self_logprobs`. That is the
  embedding-gradient term **added to** the self term, not the self term alone. The MH backward
  evaluation (lines 133 to 156) repeats exactly the same pattern at the proposed point when the
  correction is on.
- Consequence: as implemented today, one `policy_onehot` step costs **one backward pass plus two
  forward passes** (one to obtain `log_joint` for the backward, one extra for `self_logprobs`),
  against **one backward pass plus one forward pass** for plain `policy`. `policy_onehot` is strictly
  **more** expensive per step than the method it is compared against in Table~\ref{tab:onehot}, not
  less, and with the correction on (which doubles everything, per `tab:cost`'s own row for "DLS/CLS,
  with correction") the gap doubles too.

**The self term alone, however, genuinely is forward-pass-only and requires no code that exists
today.** $\log p(v \mid \vx_{<m})$ is read off `self.model(inputs_embeds=...).logits[0]` at line 63,
under `no_grad`, with no dependency on `grad_s` at all; a method that used `t2 = self._onehot_bonus(self_lp)`
alone, skipping `torch.autograd.grad` entirely, would cost one forward pass and zero backward passes
per step, the same row `tab:cost` already gives "Conditional argmax/sample". **This method does not
exist in the code.** Building it (Part 2's proposed `policy_self`) is a small, additive change: branch
before the `torch.autograd.grad` call in `get_gradient_and_log_joint` so that when only the self term
is wanted, the backward call is skipped outright, not merely unused after being computed.

**An existing overclaim in the finalized thesis, found by this check and independent of whether
Part 2 runs.** Section 6.5's current text states "Appendix~\ref{app:cost} shows that both are cheaper
than the backward pass they replace", referring to the token-indicator derivative and the output-side
bypass together. `Appendix~\ref{app:cost}`'s Table~\ref{tab:cost} (`08_appendix.tex:143-160`) has rows
for DLS/CLS with and without the correction, conditional argmax/sample, top-$k$ rescore, and Gibbs; it
has **no row at all for `policy_onehot` or the token-indicator derivative**. The appendix table does
not show the token-indicator route is cheaper because it does not cover that route, and per the
finding above, the route it would have to cover currently costs more, not less. The half of the
sentence about the output-side bypass (conditional/top-$k$/Gibbs) is correctly supported by
`tab:cost`; the half about the token-indicator derivative is not. This is a correctness issue in
already-finalized prose, independent of the future-work rewrite, and is listed as decision 5 below.

**Draft sentences**, to be placed only after the author decides how to proceed (decision 5):

For **5.6.1**, appended after the closed-form derivation (`eq:onehot-grad`) and before the
correlation numbers:

> The self term alone is available from a forward pass already computed for the energy, at no
> backward-pass cost. The token-indicator derivative as substituted into the sampler here, however,
> still forms the future term via the embedding gradient $\vg$ and so still pays the same backward
> pass as the input-embedding proposal it corrects; a proposal built from the self term alone, with
> no backward pass at all, is specified by this decomposition but is not the method run below.

For the **discussion (6.5)**, replacing the current final clause of its second paragraph:

> Either differentiate the right object, taking the relaxed token-indicator derivative, or bypass the
> derivative and propose from the model's output side directly; the self term either route would use
> is one forward pass away and costs nothing to read, though the token-indicator sampler run in this
> thesis adds that term to the embedding gradient rather than replacing it, so the saving the self
> term makes possible is specified here but not yet realized by the implementation, and
> Appendix~\ref{app:cost}'s accounting does not yet extend to it.

If Part 2's E0 is run and confirms `policy_self` (self term alone) performs at least as well as
`policy_onehot` (self plus gradient), these draft sentences would be replaced again, this time to
report the saving as realized with a measured wall-clock number and a new `tab:cost` row, which is
exactly E0's stated purpose. The sentences above are for the case where Part 2 does not run, or has
not yet reached that result: they correct the current overclaim without asserting a saving that has
not been measured.

## 4. PAGE BUDGET AND TRADES

**Cost of item 2.** The three experiment paragraphs plus the honest 6.4 sentence total approximately
650 to 750 words (drafted above; exact count depends on final wording). Section 6.5 currently runs
360 words in roughly one page, and Chapter 6 as a whole averages about 355 words per page (3198
words over 9 pages, `06_discussion.tex` against `thesis.toc`), so 650 to 750 new words is
approximately 1.8 to 2.1 pages of new material.

- If the three paragraphs and the sharpened 6.4 sentence are **appended to** Section 6.5, keeping its
  existing two paragraphs intact (retitling the subsection, e.g. "Implications and Future Work"), the
  net addition is the full 1.8 to 2.1 pages.
- If they instead **replace** Section 6.5's content outright, the net addition is roughly 1.8 to 2.1
  pages minus the 360 words (about one page) removed, i.e. about 0.8 to 1.1 pages. This was not the
  recommendation put to the author (see decision 2): the existing Implications paragraphs answer a
  question ("what does this mean for practice") that the three new paragraphs do not, and deleting
  them to make room is not asked for here.

**Cost of item 3.** Two short insertions: roughly 55 words at 5.6.1, and a same-length replacement of
an existing sentence at 6.5 (net word count roughly unchanged, since it swaps one sentence for
another of similar length). Combined cost: under 0.3 pages, effectively free relative to item 2.

**Total estimated cost: about 1.1 to 2.4 pages**, depending on the decision 2 placement choice.

**Against the limit.** Under the appendix-excluded counting the author last confirmed (countable
length about 106 pages against a 115-page floor), this addition is trivial and needs no trade at
all; it moves the countable length to perhaps 108, still under the floor. Under the appendix-included
reading this session's prompt restates (128 pages against a 119-page ceiling, already 9 pages over
before this addition), 1.1 to 2.4 more pages is a rounding change against a discrepancy that already
exists and is unrelated to this rewrite; manufacturing a fresh trade list sized to exactly 1 to 2
pages does not resolve that the two readings disagree by 21 pages about where the document currently
stands. Decision 1 needs an answer before a trade decision here means anything.

**If a trade is nonetheless wanted, ranked, with one important caveat.** The six-item trade list at
the end of the original Phase 10 report (this file, "Expected author decisions", 2026-07-26) is
**largely stale**: three of its six items proposed shrinking Appendix figures from 0.68 to 0.58
textwidth or removing `fig:traj-pca`, and the print-legibility pass two rounds later enlarged exactly
those figures (Appendix A.1 and A.4 figures now run 0.70 to full textwidth, verified in
`08_appendix.tex` today) at the author's explicit instruction that legibility takes priority over
length. Reviving that list without adjustment would silently fight a standing instruction. Of the
original six, the ones that do not touch figure sizing are still live:

1. Merge A.5.2 and A.5.3 (the guide-judge agreement ladder and the per-class confusion analysis) into
   one subsection. Estimated saving about 1 page. Cost: the two analyses run together; evaluation2
   praised neither specifically, so this was already the lowest-cost item on the original list.
2. Move Table 15 (the representative configuration-grid sample, A.2) to a stated count with no sample
   rows shown. Estimated saving about 1 page. Cost: the appendix states the grid's size without
   showing what a row of it looks like.
3. Remove forced `\clearpage` breaks in the appendix. Seven currently exist in `08_appendix.tex`
   (verified by grep). Estimated saving 1 to 2 pages, likely less than the 2 to 3 originally
   estimated at 130 pages, since the appendix is now 22 pages against 23 then. Cost: float placement
   becomes less predictable, as the earlier attempt at a broader version of this trade found when it
   was tried and reverted (Part D, evaluation3 pass).

None of these is recommended over resolving decision 1 first; they are listed because the prompt
asks for them regardless.

## 5. CONSISTENCY CHECK

- **Conclusion (Chapter 7).** No future-work material exists there today (verified by full read); its
  closing paragraph, "The main practical implication", already restates 6.5's practical reading in
  summary form. Adding the three-experiment material to 6.5 does not by itself force a change to
  Chapter 7, since the conclusion's stated job (per the evaluation5 resolution, "rewritten to report
  rather than synthesize") is to summarize what was done, not to preview what is proposed. A single
  forward-pointing clause could be added to the conclusion's close ("the discussion specifies three
  next experiments this diagnosis motivates but does not run") but this is optional, not forced;
  flagged as decision 3.
- **Abstract.** No future-work language exists (verified by full read of `chapters/abstract.tex`); its
  closing sentence ("differentiate the right object, or propose from the output side, not from the
  input embeddings") is a practical-implication summary and does not promise anything the rewrite
  would need to fulfil or walk back. No change forced.
- **RQ answers in 6.1.** None of RQ1, RQ2, RQ3a, RQ3b or E promise multi-position revision,
  constraint-energy transfer, or a diversity measurement as forthcoming work; 6.4 mentions the
  abandoned posterior-coverage evaluand (ties directly to item 2c) and 5.9's guided-steering
  subsection already states that transfer to a token-indicator or masked-LM carrier "is left open"
  (ties directly to item 2b). These are consistent anchors for the new paragraphs to cite, not
  promises the paragraphs must satisfy or contradict. No change forced.
- **Beamer's future-work slide.** None exists. The deck's content sections run Motivation through
  Takeaways with no dedicated future-work frame in the main flow, and the Backup section (checked in
  full) has no future-work-titled slide either; Part 2's own closing deliverable list already plans
  backup slides for E0 through E3, which would be the first future-work material the deck carries.
  Flagged as decision 4: whether a short future-work slide (mirroring items 2a/b/c) belongs in the
  main deck now, independent of whether Part 2 runs, since a defense audience commonly asks "what's
  next" and the deck currently has nothing to point to.

## 6. DECISIONS NEEDED FROM THE AUTHOR

1. **Which page-counting rule governs.** This session's prompt restates the appendix-included
   115-to-119-page limit from Phase 10; the 2026-07-27 log entry above records the author overriding
   that in favour of an appendix-excluded rule, under which the document already sits below the
   floor. The two readings disagree by about 21 pages on where the document currently stands. Confirm
   which rule is in force before the page-budget numbers in item 4 mean anything.
2. **Where the three new paragraphs go.** Recommended: append them, plus the sharpened 6.4 sentence,
   to the end of Section 6.5 and retitle it (e.g. "Implications and Future Work"), keeping its
   existing two paragraphs. Alternative: a new subsection 6.6. Alternative: replace 6.5 outright
   (not recommended, since its practical-implications content answers a distinct question evaluators
   have not flagged as redundant).
3. **Whether Chapter 7's closing paragraph gains a one-clause forward pointer** to the future-work
   material, or stays exactly as it is (a pure summary with no preview of proposed work).
4. **Whether a short future-work slide belongs in the main beamer deck now**, stating items 2a/b/c in
   three lines, independent of whether Part 2's experiments run, or whether this should wait for
   Part 2's own backup-slide deliverable.
5. **The existing overclaim in Section 6.5** ("Appendix~\ref{app:cost} shows that both are cheaper
   than the backward pass they replace"), found while verifying item 3. This is a correctness issue in
   already-finalized text, separate from the future-work rewrite and from whether Part 2 runs. Confirm
   whether to fix it now (using the drafted replacement sentence in item 3) as a standalone correction,
   fold it into whatever edit session applies the future-work material, or hold it until Part 2's E0
   result is known so the fix and the eventual "saving realized" sentence can be written together.
6. **Whether to proceed to Part 2** (the defense experiments E0 through E3), and if so, whether to
   build the `policy_self` variant specified in item 3 as E0's first step, per the gate already
   defined in the Phase 11 prompt.

STOP. Awaiting the author's answers to 1 through 6 before any thesis edit or GPU job.
