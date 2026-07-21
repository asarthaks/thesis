# REVISION_LOG_THESIS.md

Phase 3: applying THESIS_WRITING_GUIDE.md to Doc/. Logged here (not REVISION_LOG.md,
which another session owns). No em-dashes anywhere. Numbers sourced from
results_revision/numbers.json and the rev_*.json files, cited in % comments at each use.

## 2026-07-21 - Session start: baseline and setup

- Baseline compile: `latexmk -pdf` in Doc/ succeeds, exit 0, **81 pages**. This is the
  page-count baseline for the whole pass.
- Tools present: latexmk, pdflatex, bibtex (TeX Live on the host).
- Doc/ structure: report class, chapters/ holds abstract + 01..08. thesis.tex is the
  master. references.bib already gives every entry a URL.
- IMPORTANT discovery: much of guide section B is ALREADY present in this version of
  Doc/ (the guide predates the current .tex). Already done: `\listoffigures` +
  `\listoftables` (thesis.tex:101-103), updated statement of authorship with KI-Tools
  clause (thesis.tex:79-87), RQ-to-subsection map (06_discussion.tex sec:disc-rqs),
  related-work-to-results links (06_discussion.tex sec:disc-related), AI-tools appendix
  (08_appendix.tex:74-77), full-config-grid appendix (08_appendix.tex:46-71). These are
  audited below and only gaps are filled.
- Heading style decision (author, this session): use **titlesec on report** to render
  compact single-line headings ("5  Methods", no "Chapter 5" banner). Keep report class,
  all `\chapter`, all `Chapter~\ref{...}` prose and cross-refs unchanged. (Ioanna uses
  article+\section for the same visual effect, but converting would touch sectioning and
  reword "Chapter"->"Section" everywhere; titlesec achieves the intent at far lower risk.)

## Verified numbers index (source -> value), for the edits below

- Concern 1 (rev_stats_gpt2.json, comparisons[0], gpt2-large dls mh gn s50, comparator
  grad_norm_preserved_random_dir): policy_mean_kl 6.5408, comparator 6.3696,
  mean_diff +0.17122, CI95 [-0.28523, +0.61853], wilcoxon_p 0.40024,
  min_detectable_diff_80pow 0.65208, tost.margin 0.32704, tost.equivalent false,
  margin_rule "0.05 x policy_mean_kl (per config)", n_paired 200. len_beta=1 gap +0.670
  lives in rev_stats_gfn.json (per REVISION_LOG final report).
- Concern 2 (rev_klbase_gpt2sft.json, by_baseline.<name>.mean_kl): ground_truth 0.00
  (exact_match 100%), untouched 9.142, random_token 9.392, cond_argmax 8.236 (em 18%),
  cond_sample 8.622 (em 15%), cond_topk_rescore 4.427 (em 33%), gibbs 6.691 (em 18.5%).
  wall_time_sec 215.3 for n=200.
- Concern 3 (rev_judge_score_gpt2sft.json, by_method): judge_ppl policy 178.42,
  gradnorm 181.32, random 181.32 (gradnorm==random exactly); nll/tok 4.438 vs 4.450.
  judge=llama3-8b. rank_spearman -1.0 over 3 means: DO NOT quote. wall 24.78 s / 600 seqs.
  KL equation: transcribe avg_kl_for_fill from diagnostics/run_revision.py into Sec 4.4.
- Concern 5 (rev_divergence_*): mean_abs_loglik_diff 30.98 / 35.98 / 57.31 nats;
  nexttok_kl 1.023 / 1.345 / 3.053; pearson_base_tuned 0.983 / 0.979 / 0.770 for
  lb0-500 / lb0-2000 / lb1-500. Linearization on tuned energies stays flat (0.057, 0.024,
  0.046). Story: energy moved substantially, landscape not repaired. Do NOT soften.
- Concern 6a (rev_reconcile.mh_acceptance_by_boundary, traces_gpt2sft_mh.csv):
  cls policy NOMH within 0.0344% / boundary 3.665%; cls policy MH within 0.627% /
  boundary 8.557%; dls policy MH within 100% / boundary 9.272%. The current text
  (05_results.tex:59) reports 0.03%/3.7% and calls it "correction enabled", which
  matches the NOMH split here, though the .tex's own % SOURCE cites the old
  traces_gpt2sft.json cls_policy_gnoff_mh (=0.000344/0.03665). CONFLICT -> flag to
  author, do not silently change (see Decisions).
- Concern 6b (rev_reconcile.length_slopes): gpt2sft uncensored -1.1234 (r -0.078,
  n 1500), censored null (cap binds, frac_hit_cap 0.997). gfn-lb1-500 uncensored -0.5046,
  censored -2.361 (n 1148). Headline -1.12 stands with censoring caveat.
- Concern 6c inter-token distances (rev_reconcile.numbers_index): NN 1.822, mean pairwise
  2.773 (diag_anisotropy_gpt2sft); linearization candidate mean 2.354
  (diag_linearization_gpt2sft.mean_inter_token_distance). Llama NN 0.585, pairwise 0.835,
  lin 0.659. Use one name per statistic.
- Concern 7 (rev_reconcile, same as 6a): within-cell vs boundary splits above. CLS energy
  eq to write: E(s) = -log p_LM(proj(s)) [confirm auxiliary terms against core/cls.py].
- Concern 9 (rev_continuation_gpt2sft.json): policy 8.818 [8.612, 9.019], gradnorm 8.850
  [8.652, 9.044], random 8.850 (==gradnorm). n=100, span=20, steps=50. wall 2157 s.
- Concern 10 (rev_ltrap_within.json): pooled / max-within-strategy pearson: gpt2sft
  0.361 / 0.511 (greedy); gfn-lb0-500 0.626 / 0.830 (greedy); gfn-lb0-2000 0.785 / 0.821
  (ancestral, greedy 0.766); gfn-lb1-500 0.628 / 0.909 (greedy); llama3-8b 0.218 / 0.772
  (greedy). beam20 weakest. Present within+between-strategy (greedy/beam worst) primary.
- Concern 11 (rev_constrained.json): mucola/continuation contrast cons_only - cons_random
  = +27.33 (neg target, label0) and +36.67 (pos target, label1); DLS/"ours" continuation
  ~0 (label0 0.0, label1 +0.67). Raw arms flip sign with target (bias-dominated),
  one_sided_artifact_flag true -> report the paired contrast.
- Concern 15: 145 = 5 energy functions x 29 (8 CLS: policy,random x 4 gn/schedule
  variants; 21 DLS: policy,gradnorm,random x 7 variants). Verified gpt2_v2 29 + llama 29
  + gfn 87 = 145. Ignore rev_reconcile.config_count.distinct_run_names_present=158
  (globs grid+diag).
- Concern 16 (REVISION_LOG final report): seeds 1000-1003 of gpt2-large.dls.policy.mh.gn.
  free.s50: final KL 6.348 / 6.589 / 6.150 / 6.291, mean 6.345, sd 0.183, range 0.438.
  sd 0.183 < concern-1 margin 0.327 -> justifies margin. Hardware: university GPU server,
  49 GB A6000-class GPUs, torch 2.12 + cu130. Availability: bitwise-equivalence suite
  (verify_equivalence_suite.py).
- Concern 17 (rev_reconcile.spearman_phrasing): max |rho| 0.0573; |rho| < 0.06 across all
  five (gpt2 -0.011, llama 0.021, gfn 0.024 / 0.046 / 0.057), n = 400,000 pairs.
- Concern 19 wall-clocks: continuation 2157 s / 100 seqs; judge 24.78 s / 600 seqs;
  gibbs+baselines 215.3 s / 200 seqs (rev_klbase). Forward/backward passes per token:
  read from core samplers.
- Concern 20 (last-token): NO Phase 2 results in results_revision. -> write marked TODO
  block, tell author (see Decisions).

---

## Guide-item inventory (target file : lines : current wording : plan)

### A. Content edits

- **A1 (concern 1 stats)** -> 05_results.tex tab:fallacy (116-131) + text 110-114; margin
  justification into 04_methodology.tex sec:meth-metrics (29-38). Current table has only
  Policy/GNP/Difference columns, no CI/p/MDD. Plan: add paired mean diff, 95% CI,
  Wilcoxon p, MDD@80% for the flagship gpt2 dls/mh/gn/s50 cell in text; state the
  equivalence margin (0.05 x policy mean KL) and its empirical justification (across-seed
  sd 0.183) in methodology so it does not read post hoc; treat len_beta=1 +0.670 as a
  separate real harmful effect (already partly noted at 05:114/126).
- **A2 (concern 2 baselines)** -> 05_results.tex, new reference-baseline table near the
  fallacy/quench sections + reference note. Current: none. Plan: add ground_truth 0.00,
  untouched 9.14, random_token 9.39, cond_argmax 8.24, cond_sample 8.62,
  cond_topk_rescore 4.43, gibbs 6.69; land the sentence "one forward pass with top-k
  rescoring beats every Langevin config, and a gradient-free Gibbs sampler on the same
  energy works, so the energy evaluation does the work and the gradient adds nothing."
- **A3 (concern 3 KL eq + judge)** -> 04_methodology.tex sec:meth-metrics (35-36) add the
  exact avg_kl_for_fill equation; add external-judge paragraph in 05_results.tex (after
  the fallacy or crossmodel section) with ppl 178.4 vs 181.3, <2%, no winner. Do not quote
  the 3-point rank correlation.
- **A4 (concern 4 scope + RW)** -> 07_conclusion.tex:4 narrow "not a usable energy
  function for sampling on discrete text" to "does not admit local gradient-based
  navigation; its gradient is not a usable search direction". Add RW paragraph in
  03_related_work.tex sec:rel-energy (after 20 or 22) on Mix-and-Match (Mireshghallah et
  al. 2022) and twisted SMC (Lew et al.; Zhao et al. 2024) as frozen-LM-energy samplers
  that avoid the energy gradient, citing the in-house Gibbs result. New bib entries needed.
- **A5 (concern 5 divergence table)** -> 05_results.tex sec:results-gfn (around 275-278)
  or a short new subsection. Current: none. Plan: table of mean abs loglik diff / nexttok
  KL / pearson for the 3 gfn variants + the sentence tying to flat linearization.
- **A6 (concern 6 residuals)** -> (a) 05_results.tex:59-64 attribution FLAG (do not
  change; add %TODO-AUTHOR). (b) 05_results.tex:211-213 + appendix trap-length caption
  (08:41) add censoring caveat. (c) inter-token distance naming: unify NN 1.82 / pairwise
  2.77 / linearization 2.35 across 05:11, 05:163-165, 05:220. Currently uses all three but
  check one-name-per-statistic.
- **A7 (concern 7 CLS energy eq + derivation)** -> 02_background.tex sec:bg-samplers
  (69) add E(s)=-log p(proj(s)) equation; add within-cell acceptance derivation paragraph;
  compare to measured split (within 0.034% / boundary 3.665% nomh; 0.627/8.56 mh; DLS mh
  100/9.27). Confirm auxiliary energy terms against core/cls.py first.
- **A8 (concern 8 SEDD)** -> 06_discussion.tex sec:disc-future (already discusses diffusion
  positive control conceptually, 55-57). Add explicit status: pipeline dry-run validated,
  real run designed-but-not-run, what would be measured, decisive either way. No results
  exist -> designed-but-not-run variant.
- **A9 (concern 9 continuation null)** -> 05_results.tex sec:results-robust (135-141) add
  one row + two sentences: null reproduces on 20-token free-form continuation (policy
  8.818 [8.61,9.02] vs random 8.850 [8.65,9.04]).
- **A10 (concern 10 within-strategy)** -> 05_results.tex sec:results-trap (186-189) and
  crossmodel (243). Current reports pooled-ish +0.65..+0.93. Plan: report within-strategy
  max |r| (gpt2sft 0.51, gfn 0.83/0.82/0.91, llama 0.77) as primary + between-strategy
  (greedy/beam worst); note llama pooled weaker (0.218) and frac_hit_cap 1.0.
- **A11 (concern 11 contrast)** -> 05_results.tex sec:results-constrained (302-327).
  Current table is DLS infill raw gains. Plan: add the paired contrast cons_only -
  cons_random = +27.3 (neg) / +36.7 (pos) on mucola continuation, ~0 on DLS; explain raw
  gains bias-dominated (sign-flips with target); add cons_only-anomaly paragraph
  (constraint-only leaves the fluent manifold, classifier outputs meaningless there).
- **A12 (concern 12 oracle per method)** -> 04_methodology.tex sec:meth-configs (27) add
  one sentence: oracle step-size sweep run independently per proposal method.
- **A13 (concern 13 define "worked")** -> 05_results.tex sec:results-stepsize (9-11) or 18.
  Add one sentence defining "worked/work" = calibrated motion, not successful guidance.
- **A14 (concern 14 soften len_beta)** -> 05_results.tex:273 "no setting between them was
  found to produce good text" -> "the two endpoints produce opposite degeneracies, and no
  intermediate setting was tested that resolves both".
- **A15 (concern 15 config count)** -> 08_appendix.tex app:grid (46-51). Add counting
  formula 145 = 5 x 29 with the 8 CLS + 21 DLS breakdown and why 240 overcounts.
- **A16 (concern 16 seeds/hw/availability)** -> 04_methodology.tex (end of sec:meth-diagnostics
  63, or a short reproducibility paragraph). Add seeds 1000-1003, across-seed KL + sd 0.183,
  hardware, code/data availability naming the bitwise-equivalence suite.
- **A17 (concern 17 spearman phrasing)** -> 05_results.tex:148 "which is to say zero" ->
  "negligible in magnitude (|rho| < 0.06 across all five models: gpt2 -0.011, llama 0.021,
  gfn 0.024/0.046/0.057), n = 400,000 candidate pairs"; add n to fig:lin-scatter caption
  (159).
- **A18 (concern 18 compress recaps)** -> tighten opening mini-recaps of 05_results.tex
  sec:results-fallacy(103-106?) actually 5.5/5.10/6.2 per guide: opening of
  sec:results-fallacy is 5.5-ish; 5.10 = sec:results-constrained; 6.2 = sec:disc-unified.
  Keep full recap once at top of Ch6; compress chapter transitions to one sentence.
- **A19 (concern 19 cost table)** -> 08_appendix.tex new section. Forward/backward passes
  per recovered token for DLS, CLS, conditional resampling, Gibbs + wall-clock
  (continuation 2157 s/100, judge 24.8 s/600, baselines 215 s/200).
- **A20 (concern 20 last-token)** -> NEW section in 05_results.tex + closing figure. NO
  Phase 2 results exist -> write a clearly marked TODO block, tell author.

### B. Structural / formatting

- List of tables + figures: DONE (thesis.tex:101-103). Verify order acceptable.
- RQ-to-subsection map: DONE (06_discussion.tex sec:disc-rqs). Verify every RQ named +
  subsection named. Currently maps to Ch5 sections; good.
- Discussion links related work: DONE (sec:disc-related) for Holtzman, anisotropy,
  MH/COLD/MuCoLa, Hu. GAPS: add Mix-and-Match/twisted-SMC vs Gibbs, and Welling&Teh vs
  MH breakdown per guide B. -> extend sec:disc-related.
- Every bib entry has URL: DONE (references.bib all 30 entries carry url). Fix long-URL
  overflow on bib page 2 -> add xurl + \urlstyle{same} (template pass C).
- Why-chosen for each metric/method: audit sec:meth-metrics, sec:meth-models,
  sec:meth-configs, sec:meth-gfn. Mostly present; fill any gap and log which were missing.
- Appendix material commented on in results: figures (lin-decomp, lin-topk, mh-accept,
  trap-length) all referenced in Ch5; config grid referenced; NEW appendix items (cost
  table A19, config-count A15, trajectory D) must be cross-referenced from Results.

### C. Template (vs IoannaThesis/Ioannathesis.tex)

- Ioanna: article class, \section top-level -> compact "N. Title". No \titleformat (the
  compact form is inherent to article). Decision: titlesec on report (see above).
- xurl + \urlstyle{same}: Ioanna preamble lines 63, 65. Copy into thesis.tex preamble.
- baselinestretch 1.3, parskip medskip, frenchspacing: already identical in both.
- Statement of authorship: current thesis already has the updated KI-Tools wording
  (thesis.tex:83). Do not alter text; author supplies final file if different.

### D. Embedding-trajectory section

- Add subsection in Results (or appendix, cross-referenced) with [PLOT PLACEHOLDER]
  figures + % regeneration instructions (s_ids_history from traces npz / collect_traces;
  PCA 2-comp on vocab embeddings; policy vs random MH-on; CLS MH on/off). Prose connects
  to anisotropy NN 1.82 / pairwise 2.77. Note: DLS trajectory PNGs already exist
  (figures/gpt2-large.dls.gn.free.s50_new_trajectories.png) but these are metric-vs-step,
  not the PCA embedding-space plots the guide wants -> placeholders for the PCA plots.

### E. AI-tools appendix

- 08_appendix.tex:74-77 already lists Gemini + Claude. Guide E requires: adapt Ioanna's
  structure and add the Phase 3 LaTeX editing/formatting assistance accurately. -> revise
  to name Google AI Studio (Gemini 3.1 Pro) and Claude (Anthropic), and add that Claude
  Code assisted with editing/formatting the LaTeX in the revision. Keep declaration honest.

### F. Voice / why-spine

- After local edits, read Results+Discussion end to end; ensure a Discussion-only reader
  can state the why in one sentence: energy fine (Gibbs + one forward pass work), gradient
  defective (zero at last position by causality, uninformative elsewhere by measurement),
  diffusion the counterfactual. The A2 Gibbs result and A20 last-token feed this spine.

---

## Decisions the author must confirm (running list)

1. Concern 6a attribution (05_results.tex:59-64): the 0.03%/3.7% pair matches the CLS
   policy NO-MH trace in rev_reconcile (0.034/3.665), but the text says "correction
   enabled" and its % SOURCE cites the old traces cls_policy_gnoff_mh. Which run produced
   the reported figure? Left unchanged pending confirmation; %TODO-AUTHOR added.
2. SEDD (A8): real run absent -> writing the designed-but-not-run variant. Confirm.
3. A20 last-token section: Phase 2 results absent -> TODO block only. Confirm whether to
   wait for Phase 2 or ship without.
4. A4 new citations (Mix-and-Match Mireshghallah 2022; twisted SMC Lew/Zhao 2024): added
   to references.bib with URLs; confirm exact venue/year.

## Progress log (per item)

(entries appended below as each item lands + compile status)

- **C (template pass) DONE.** thesis.tex preamble: added `\usepackage{titlesec}` with
  `\titleformat{\chapter}[hang]{...\huge\bfseries}{\thechapter}{0.6em}{}` +
  `\titlespacing*` so chapter headings render as compact "1  Introduction" (verified by
  rendering physical page 11 to PNG: single-line heading, no "Chapter" banner). Added
  `\usepackage{xurl}` + `\urlstyle{same}` (copied from Ioanna's preamble): verified on
  bibliography page 2 (physical page 73) that long DOIs/arXiv URLs now break within the
  margin. Compile exit 0, 0 errors. Page count 81 -> 80.

- **A3 (KL equation) DONE.** 04_methodology.tex sec:meth-metrics: added eq:meth-kl
  transcribed from avg_kl_for_fill (mean over masked positions m<L-1 of KL(p_ref||p_pred),
  batchmean), with the ground-truth-fill floor of 0. (Judge paragraph is a Results edit,
  done under task 4.)
- **A12 (oracle per method) DONE.** 04_methodology.tex sec:meth-configs: added the
  per-method independent oracle-sweep sentence; also forward-refs the config-count appendix.
- **A16 (seeds/hardware/availability) DONE.** New \section{Reproducibility, Seeds, and
  Hardware} (sec:meth-repro): seeds 1000-1003, KL 6.348/6.589/6.150/6.291, mean 6.345,
  sd 0.183, range 0.438; determinism per sample index; hardware (49 GB A6000-class, torch
  2.12 + cu130); availability naming verify_equivalence_suite.py.
- **A1 (methodology half) DONE.** sec:meth-metrics: equivalence margin = 0.05 x policy
  mean KL, justified by seed sd 0.183 < margin 0.327 (not post hoc). Table/text half of
  A1 is a Results edit (task 4).
- **A7 (CLS energy + within-cell derivation) DONE.** 02_background.tex sec:bg-samplers:
  added eq:bg-cls-energy E_CLS(s) = -log p_LM(proj_V(s)) with a % note that the impl
  feeds continuous s as inputs_embeds while proj(s) is the target (so within-cell surface
  is nearly, not exactly, flat; boundary target-jump is the load-bearing discontinuity),
  plus the within-cell acceptance derivation (target ratio ~1 within cell -> acceptance
  governed by proposal ratio; DLS same-token move accepted w.p. 1, CLS continuous move
  not protected), forward-referencing the measured split in sec:results-mh.
- Compile after methodology+background: exit 0, refs resolved on rerun, 82 pages.

- **A13 DONE** (05 sec:results-stepsize): defined "work" = calibrated motion (tokens
  change, metrics evolve), distinct from guidance.
- **A17 DONE** (05 sec:results-linradius:148 + fig:lin-scatter caption): replaced "which
  is to say zero" with |rho| < 0.06 across all five models (gpt2 -0.011, llama 0.021, gfn
  0.024/0.046/0.057), n = 400,000; added n to caption.
- **A6c DONE** (05 linradius + aniso): named the three distances once in sec:results-aniso
  (NN 1.82, pairwise 2.77, linearization-candidate 2.35 / llama 0.59, 0.84, 0.66); fixed
  the linradius argument so 2.35 = candidate mean and 1.82 = nearest-neighbour (the
  smallest-substitution scale), corrected fig:lin-radius caption to "candidate distance".
- **A14 DONE** (05:273): softened to "no intermediate setting was tested that resolves
  both; only the two endpoints were run".
- **A6a FLAGGED, not changed** (05 sec:results-mh:59): added %TODO-AUTHOR documenting the
  reconcile-vs-traces label conflict (0.03/3.7 matches NO-MH split; MH split 0.627/8.56).
- **A6b DONE** (05 sec:results-trap): added censoring caveat to -1.12 slope (frac_hit_cap
  0.997 -> censored null for gpt2; gfn-lb1-500 uncensored -0.505 vs censored -2.361).
- **A10 DONE** (05 trap + crossmodel): within-strategy max |r| primary (0.51/0.83/0.82/
  0.91/0.77), greedy/beam worst, llama pooled 0.22 weaker + frac_hit_cap 1.0.
- **A1 (results half) DONE** (05 sec:results-fallacy): added paired-stats paragraph
  (mean diff +0.171, CI [-0.285,+0.619], Wilcoxon p 0.40, MDD 0.652, margin 0.327,
  bounded-effect reading); flagged len_beta=1 +0.670 as a real harmful effect exceeding
  seed noise.
- **A2 DONE** (05 new sec:results-baselines + tab:baselines): ground_truth 0/untouched
  9.14/random 9.39/cond_argmax 8.24/cond_sample 8.62/topk_rescore 4.43/gibbs 6.69; landed
  the spine sentence (topk beats every Langevin cell; gibbs matches; energy is fine).
- **A3 (judge half) DONE** (05 new sec:results-judge): Llama judge ppl 178.4 vs 181.3,
  <2%, gradnorm==random; did NOT quote the 3-point rank correlation.
- **A9 DONE** (05 sec:results-robust): continuation null (policy 8.818 [8.61,9.02] vs
  random 8.850 [8.65,9.04], gradnorm==random) as fourth robustness variation.
- **A5 DONE** (05 sec:results-gfn + tab:divergence): mean abs loglik diff 31/36/57 nats,
  nexttok KL 1.02/1.35/3.05, pearson 0.98/0.98/0.77; energy moved yet linearization flat.
- **A11 DONE** (05 sec:results-constrained): raw gains bias-dominated (sign-flip with
  target); paired contrast cons_only-cons_random = 0 on DLS, +27.3/+36.7 on mucola
  continuation; cons_only off-manifold anomaly paragraph. NOTE: this nuance (constraint
  direction carries SOME signal on mucola, unlike the fluency gradient) is softer than the
  abstract's flat "constraint gradient carries no reliable steering signal" -> author
  decision logged.
- **A18 REVIEWED** (no destructive cuts): section openings 5.5/5.10/6.2 are already one-
  to-two-sentence recaps; the end-of-section forward transitions are the intentional
  investigative device set up in sec:intro-shape; kept the full recap at top of Ch6.
- Compile after all Results edits: exit 0, table refs resolved on rerun, 88 pages.

- **A4 DONE.** references.bib: added mireshghallah2022mixmatch, lew2023sequential,
  zhao2024probabilistic (with URLs; venue/year flagged for author confirm). 03_related_work
  sec:rel-energy: new paragraph on gradient-free frozen-LM samplers (Mix-and-Match, twisted
  SMC) + in-house Gibbs separation. 07_conclusion:4 narrowed to "does not admit local
  gradient-based navigation; its gradient is not a usable search direction" + the energy-is-
  searchable-by-gradient-free-means clause.
- **A8 DONE.** 06_discussion sec:disc-future: SEDD (lou2024discrete) designed-but-not-run
  status (pipeline dry-run validated, hooks marked for verification, real run deferred),
  the sharp prediction, decisive either way.
- **B (discussion RW links) DONE.** sec:disc-related: added Welling&Teh vs the quenching/
  biased-optimizer finding, and a Mix-and-Match/twisted-SMC vs Gibbs paragraph.
- **F (why-spine) DONE (analytic part).** sec:disc-unified: added the causal core, the
  input-embedding gradient is blind to token fitness (which enters via a discrete output
  index), total blindness at the last position where the gradient is exactly zero while the
  energy difference is the conditional log-ratio. Empirical last-token table/sampler is the
  A20 TODO (Phase 2 absent). Spine now: energy fine (Gibbs+topk, tab:baselines) / gradient
  defective (fallacy + linearization + zero-at-last-position) / diffusion counterfactual
  (sec:disc-future). Discussion-only reader can state the why.
- Compile after RW/discussion/conclusion: exit 0, all 3 new citations in bbl, 92 pages.

- **A15 DONE** (08 app:grid): counting formula 145 = 5 x 29 (8 CLS + 21 DLS), why 240
  overcounts, verified 29+29+87.
- **A6b appendix DONE** (08 fig:trap-length caption): censoring caveat added.
- **A19 DONE** (08 new app:cost + tab:cost): passes/token (DLS/CLS S fwd+S bwd, 2S under
  MH; conditional 1 fwd; topk 1+k; gibbs M*T; all gradient-free 0 bwd) + wall-clocks
  (continuation 2157 s/100, baselines+gibbs 215 s/200, judge 24.8 s/600).
- **D DONE** (08 new app:trajectory): PCA plot placeholders (Plot 1 DLS policy/random MH;
  Plot 2 CLS MH on/off) with % regeneration instructions (s_ids_history from
  traces_gpt2sft_traj.npz / collect_traces, PCA on vocab embedding matrix), prose tied to
  anisotropy NN 1.82 / pairwise 2.77. Cross-referenced from sec:results-traj.
- **E DONE** (08 app:ai-tools): rewrote to name Google AI Studio (Gemini 3.1 Pro) + Claude
  (Anthropic); added the Phase 3 agentic LaTeX-editing paragraph accurately (template
  match, transcribing verified numbers, formatting, copy-edit; author-specified list,
  every number traced, author reviewed). Already cross-referenced from the authorship
  statement (thesis.tex), matching Ioanna.
- **A20 TODO-BLOCK** (05 new sec:results-lasttoken): wrote the provable zero-gradient
  analytic derivation (complete, no experiment needed) + clearly-marked [PENDING PHASE 2]
  block for the position-condition table, energy-only-vs-gradient sampler comparison, and
  closing figure fig:lasttoken. Phase 2 results absent from results_revision -> author must
  decide whether to wait for Phase 2 or ship the analytic-only version.
- **A19/D cross-refs DONE** (B, no orphan appendix): sec:results-baselines -> app:cost;
  sec:results-traj -> app:trajectory.
- Compile after appendices+last-token: exit 0, no undefined refs, 98 pages.

- **Template running header:** added `\renewcommand{\chaptermark}` so the running header
  is compact ("5. Results", "A. Appendix") rather than "CHAPTER 5. RESULTS". Verified by
  render.

---

# FINAL REPORT (Phase 3) - 2026-07-21

## Compile + gate
- Clean-from-scratch build (`latexmk -C` then latexmk): exit 0, **0 undefined
  citations/references**, no LaTeX errors.
- **Page count: 81 (baseline) -> 98 (final), +17 pages.**
- Em-dash check: no literal em-dash and no `---` in any chapter prose;
  `Metropolis--Hastings` en-dashes intact.
- Numbers diff: every numeric claim written was pulled programmatically from its source
  file and matched exactly (rev_stats_gpt2, rev_klbase, rev_judge, rev_continuation,
  rev_divergence, rev_ltrap_within/per_model, rev_constrained, rev_reconcile). **No
  mismatches.** Each number carries a `% SOURCE:` comment in the .tex.
- Visual checks (rendered to PNG): compact chapter heading ("1  Introduction"), compact
  running header, bibliography page 2 URLs breaking within margin (xurl), last-token
  PENDING box, trajectory placeholder box, all render correctly.

## Guide-item status
- A1 DONE (methodology margin + results paired stats). A2 DONE (baselines table + spine
  sentence). A3 DONE (KL equation + judge). A4 DONE (narrowed conclusion + RW paragraph +
  3 new cites). A5 DONE (divergence table). A6a FLAGGED for author (attribution). A6b DONE
  (censoring caveat, text + appendix caption). A6c DONE (three distances named once).
  A7 DONE (CLS energy eq + within-cell derivation). A8 DONE (SEDD designed-but-not-run).
  A9 DONE (continuation null). A10 DONE (within-strategy). A11 DONE (paired contrast +
  anomaly; see decision 5). A12 DONE (oracle per method). A13 DONE (define "work").
  A14 DONE (soften len_beta). A15 DONE (config-count formula). A16 DONE (seeds/hw/avail).
  A17 DONE (spearman phrasing + caption n). A18 REVIEWED (openings already compact;
  investigative transitions intentionally preserved; no destructive cuts). A19 DONE (cost
  table). A20 TODO-BLOCK (analytic derivation written; empirical parts pending Phase 2).
- B: list of tables/figures DONE (pre-existing, verified in ToC). RQ-to-subsection map
  DONE (pre-existing sec:disc-rqs, verified all four RQ answered with named sections).
  RW-to-results links DONE (extended: Welling&Teh, Mix-and-Match/twisted-SMC/Gibbs). Bib
  URLs DONE (all entries have url; overflow fixed via xurl). Why-chosen audit: DONE, all
  present already (task 3 reasons for ROCStories; KL over exact-match; Euclidean
  unreliability; BERTScore rationale; why both DLS+CLS; why GFlowNet over RL; oracle
  purpose) - no gaps found. No-orphan-appendix DONE (app:cost and app:trajectory
  cross-referenced from Results). Per-number source comments maintained.
- C: compact headings DONE (titlesec + chaptermark). Vertical spacing already matched
  Ioanna (baselinestretch 1.3, parskip medskip, frenchspacing). xurl URL overflow DONE.
  Statement of authorship already carried the updated KI-Tools wording; not altered.
- D DONE (trajectory appendix, PCA placeholders + regen instructions + prose). E DONE
  (AI-tools appendix rewritten incl. Phase 3 editing). F DONE (why-spine: energy fine /
  gradient defective incl. zero-at-last-position causal core / diffusion counterfactual).

## Decisions the author must confirm
1. **Concern 6a attribution** (05_results.tex, sec:results-mh, %TODO-AUTHOR): the reported
   0.03% / 3.7% acceptance pair matches the CLS-policy NO-MH split in the refreshed
   reconcile (0.034 / 3.665), but the sentence says "correction enabled" and its old
   %SOURCE cited the MH trace. MH split is 0.627% / 8.56%. Left unchanged. Confirm which
   run produced the figure and relabel or swap accordingly.
2. **A20 last-token section**: Phase 2 results are absent from results_revision. The
   provable analytic zero-gradient derivation is written in full; the position-condition
   table, the energy-only-vs-gradient sampler comparison, and Figure fig:lasttoken are
   clearly-marked [PENDING PHASE 2] placeholders. Decide whether to run Phase 2 and fill
   them, or ship the analytic-only version and remove the placeholders.
3. **SEDD (A8)**: wrote the "designed but not run" variant (pipeline dry-run validated,
   real run deferred). Confirm this is the intended presentation.
4. **New citations (A4)**: added mireshghallah2022mixmatch (ACL 2022), lew2023sequential
   (2023), zhao2024probabilistic (ICML 2024) with URLs. Confirm exact venue/year.
5. **A11 vs the abstract** (IMPORTANT): the revised constrained-generation section reports,
   per concern 11, that the constraint gradient's DIRECTION carries some signal on the
   MuCoLa continuation setup (paired contrast +27.3 / +36.7), unlike the fluency gradient,
   though confounded by the off-manifold cons_only anomaly and ~0 on the DLS sampler. This
   is softer than the abstract's flat "the constraint gradient carries no reliable steering
   signal on this energy landscape" (abstract.tex). The abstract and intro were NOT changed.
   Decide whether to soften the abstract/intro wording to match the nuanced results text.
6. **A7 "flat plateau" idealization**: the implementation feeds the continuous state as the
   input embedding while the projected token is the target, so the within-cell surface is
   nearly (not exactly) flat. Kept the author's "flat plateau" framing and added a % note +
   one clause; the load-bearing boundary discontinuity is unaffected. Confirm the framing.

## Files touched (Doc/ only, plus this log)
thesis.tex (preamble), references.bib (+3 entries), chapters/02_background.tex,
03_related_work.tex, 04_methodology.tex, 05_results.tex, 06_discussion.tex,
07_conclusion.tex, 08_appendix.tex. abstract.tex NOT changed (see decision 5).
