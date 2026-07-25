# Thesis writing guide

The single reference for every edit to the LaTeX in Doc/. Everything here comes
from three sources: the WRITE items in REVISION_LOG.md's final report (with the
verified numbers), the author's formatting requirements, and the template rules
from Ioanna's thesis. Numbers cited below were verified against
results_revision/numbers.json; when in doubt, re-check there, never from memory.

## A. Content edits from the revision (apply each, with its number)

1. Concern 1 stats into Table 5.1 and text. Add paired mean difference, 95%
   bootstrap CI, and the min-detectable-difference to the policy-vs-random
   comparisons. Frame as bounded effects: for the flagship gpt2 dls/mh/gn/s50
   cell, mean diff +0.171 KL, CI [-0.285, +0.619], Wilcoxon p 0.40, minimum
   detectable difference at 80% power 0.652. State the equivalence margin (0.05 x
   policy mean KL) and its empirical justification (across-seed sd 0.183 from
   concern 16) in the METHODOLOGY chapter so it does not look post hoc. Treat the
   len_beta=1 gap (+0.670) as a separate, real, harmful effect: it exceeds seed
   noise.

2. Concern 2 baselines into Table 5.1 and trajectory figures as reference rows or
   lines: ground_truth 0.00 (metric floor, 100% exact match), untouched 9.14,
   random_token 9.39, cond_argmax 8.24, cond_sample 8.62, cond_topk_rescore 4.43,
   gibbs 6.69 (gpt2sft). The sentence to land: one forward pass with top-k
   rescoring beats every Langevin configuration in the grid, and a gradient-free
   Gibbs sampler on the same energy also works, so the exact energy evaluation
   does the work and the gradient adds nothing.

3. Concern 3. Write the exact KL equation in Section 4.4, transcribed from
   avg_kl_for_fill in diagnostics/run_revision.py: at each masked position m with
   m < L-1, p_ref is the next-token distribution under the ground-truth fill,
   p_pred the same under the recovered fill, the metric is mean KL(p_ref ||
   p_pred). Note the ground-truth fill gives 0 by construction. Then add the
   external-judge paragraph: an independent Llama-3 judge scores the recovered
   sequences at perplexity 178.4 (policy) vs 181.3 (random), differences under 2%,
   no method wins, so the methods are indistinguishable under a judge that is not
   the model being sampled. Do not quote the 3-point rank correlation.

4. Concern 4 scope. The abstract already says the narrow claim. Fix the
   conclusion (07_conclusion.tex, first sentence area) from "not a usable energy
   function for sampling on discrete text" to the narrow form: the frozen
   likelihood does not admit local gradient-based navigation; its gradient is not
   a usable search direction. Add the related-work paragraph in Chapter 3 on
   samplers over frozen-LM energies that do not use the energy gradient:
   Mix-and-Match (Mireshghallah et al. 2022) and twisted SMC (Lew et al.; Zhao et
   al. 2024), and cite the in-house Gibbs result as the in-platform version of
   the same separation. Keep the likelihood-trap argument as a separate claim that
   applies to any method reaching the minimum.

5. Concern 5. Report the model-divergence table: mean absolute log-likelihood
   difference 31.0 / 36.0 / 57.3 nats, next-token KL 1.02 / 1.35 / 3.05, base-vs-
   tuned Pearson 0.98 / 0.98 / 0.77 for lb0-500 / lb0-2000 / lb1-500. Conclusion
   stands as written: the tuning moved the energy substantially, yet the
   linearization correlations on the tuned energies stay flat (0.057, 0.024,
   0.046), so amortization does not repair the landscape. Do not soften this to
   "tuning left the energy unchanged"; the opposite is true.

6. Concern 6 residuals. (a) The 0.03% / 3.7% acceptance pair matches the
   mh=FALSE trace (0.034 / 3.665); the text currently attributes it to the
   correction being enabled. Confirm with the author which run it came from and
   fix the attribution; the mh=TRUE trace reads 0.627% / 8.56%. (b) Keep the
   -1.12 nats/token slope but add the censoring caveat: nearly all gpt2
   generations hit the 40-token cap (frac 0.997), the censored slope is null, and
   on lb1-500, where length varies, uncensored -0.505 vs censored -2.361 shows
   the cap matters. (c) Define the three inter-token distance statistics once,
   each with its number: nearest-neighbour 1.82, mean pairwise 2.77, linearization
   candidate mean 2.35, and use one name per statistic everywhere.

7. Concern 7. Write the CLS energy as an equation in Section 2.4 (E(s) =
   -log p(proj(s)) plus whatever auxiliary terms the implementation actually has;
   read core/cls.py, state exactly what is there). Add the one-paragraph
   within-cell acceptance derivation and compare against the measured split
   (within-cell 0.034%, boundary 3.665% for the no-MH trace; 0.627% / 8.56% with
   MH; DLS with MH: within 100%, boundary 9.27%).

8. Concern 8. Present SEDD honestly at its current status: pipeline validated by
   dry-run; if the Phase 2 real run lands, report it as the pilot positive
   control section; if not, write the designed-but-not-run paragraph explaining
   what would be measured and why it is decisive either way.

9. Concern 9. One table row plus two sentences: the null reproduces on free-form
   20-token continuation (policy 8.818 [8.61, 9.02] vs random 8.850 [8.65,
   9.04]), so the finding is not an artifact of the masked-recovery task.

10. Concern 10. Report within-strategy correlations as primary: max within-
    strategy |r| gpt2sft 0.51, gfn variants 0.82 to 0.91, llama 0.77; the pooled
    llama number (0.218) is weaker, so present within-strategy plus the between-
    strategy pattern (greedy and beam worst). Note the Llama degeneracy signal:
    frac_hit_cap 1.0, never emits EOS.

11. Concern 11. Report the paired contrast as the number: cons_only minus
    cons_random is +27.3 (negative target) and +36.7 (positive target) on the
    mucola continuation setup and near zero on the DLS setup; raw gains are
    dominated by a fixed sentiment drift (every arm flips sign with the target
    label), which is why the paired contrast is the honest statistic. Add the one
    paragraph on the cons_only anomaly: constraint-only optimization leaves the
    fluent manifold and classifier outputs on that text are meaningless.

12. Concern 12. One sentence in 4.3: the oracle step-size sweep was run
    independently per proposal method (the per-method oracle runs exist in the
    grid), so the schedule selection does not favor any method.

13. Concern 13. Define "worked" in 5.1 in one sentence using the section's own
    vocabulary: calibrated motion, not successful guidance.

14. Concern 14. Soften the len_beta sentence (05_results.tex around line 273) to:
    the two endpoints produce opposite degeneracies, and no intermediate setting
    was tested that resolves both.

15. Concern 15. Appendix paragraph with the counting formula: 145 = 5 energy
    functions x 29 configurations each, where 29 = 8 CLS (policy and random
    across 4 grad-norm and schedule variants) + 21 DLS (three methods across 7
    variants); the nominal factorial 240 overcounts because CLS drops the
    grad-norm-preserved method and some cells. Verified against the result
    folders (29 + 29 + 87).

16. Concern 16. In Chapter 4: seeds 1000 to 1003, across-seed final KL 6.348 /
    6.589 / 6.150 / 6.291 (sd 0.183), hardware (university GPU server, 49 GB
    A6000-class GPUs, torch 2.12 + cu130), and a code and data availability
    statement naming the bitwise equivalence suite.

17. Concern 17. Replace "which is to say zero" (05_results.tex around line 148)
    with: negligible in magnitude (|rho| < 0.06 across all five models: gpt2
    -0.011, llama 0.021, gfn 0.024 / 0.046 / 0.057), n = 400,000 candidate pairs,
    with n also in the caption.

18. Concern 18. One editing pass: keep the full argument recap once at the top of
    Chapter 6; compress the mini-recaps opening 5.5, 5.10, 6.2 and the chapter
    transitions to one sentence each.

19. Concern 19. Small cost table: forward and backward passes per recovered token
    for DLS, CLS, conditional resampling, Gibbs, plus wall-clock per sequence
    (data points on hand: continuation 2157 s per 100 sequences, judge scoring
    24.8 s per 600 sequences; pull Gibbs wall-clock from the rev_klbase logs).

20. NEW, from Phase 2 when it lands: the last-token section. The zero-gradient
    derivation, the position-condition table, the working energy-only sampler vs
    the failing gradient arm on the same task, and the closing figure (gradient
    information vs downstream context length). This is the section that answers
    the supervisor's "why does it not work" with a proof: the only likelihood
    term that knows how well a token fits does not depend on that token's input
    embedding, so the input-embedding gradient is causally blind to token
    fitness, exactly zero at the last position and diluted everywhere else.

## B. Structural and formatting requirements (the author's rules)

- Front matter must include a list of tables and a list of figures (add
  \listoftables and \listoffigures after the table of contents if absent), and
  the thesis must have an appendix.
- In the Discussion chapter, every research question is answered explicitly:
  name the RQ being addressed and state in which subsection of the discussion it
  is answered. An RQ-to-subsection map at the top of the chapter is the cleanest
  form.
- The Discussion must link related work to the obtained results: for each major
  result, connect it to the Chapter 3 literature it confirms, contradicts, or
  extends (Mix-and-Match and twisted SMC vs the Gibbs result; Holtzman et al. vs
  the likelihood trap; Welling and Teh vs the MH breakdown; Hu et al. vs the
  GFlowNet failure modes).
- Every bibliography entry carries a URL so it is trackable. Fix the entry on
  bibliography page 2 whose URL overflows the margin (use \usepackage{xurl} or
  \sloppy plus \urlstyle, whichever matches Ioanna's preamble; check how her
  bibliography handles long URLs and copy that).
- For every metric and every method or algorithm used, state why it was chosen,
  with references. This already largely exists per the thesis conventions; audit
  each metric and method paragraph and fill gaps.
- Everything placed in the appendix is commented on in the relevant results
  subsection: no orphan appendix material.
- Every number in the text carries its source file in a % LaTeX comment
  (existing convention; maintain it for all new numbers, sourcing from
  results_revision/numbers.json paths).

## C. Template matching against Ioanna's thesis (IoannaThesis/Ioannathesis.tex)

- Chapter headings must match her style: "5. Methods" in her compact form, not
  the large "Chapter 5" display. Inspect her class options and any \titleformat
  or \chapterformat redefinitions in the preamble and copy them verbatim.
- Match her vertical spacing (line spread, spacing around headings and floats).
  Diff the preambles and adopt hers wherever the two differ, unless the IMS
  class forbids it.
- Keep the overall structure aligned with hers (already the case: Introduction,
  Background, Related Work, Methodology, Results, Discussion, Conclusion,
  Appendix).
- Use the updated statement of authorship the author provided.

## D. The embedding-trajectory section (plots to be recreated)

Add a subsection in Results (or an appendix section commented on in Results) on
the geometry of sampler trajectories in reduced-dimensionality embedding space.
The original plots were lost; write the section with clearly marked
[PLOT PLACEHOLDER] figures and give the author exact regeneration instructions in
a % comment block at each placeholder:

- Data: s_ids_history from a stored trajectory run (the traces npz in
  results_diag, or rerun collect_traces on one config).
- Plot 1: PCA (2 components, fit on the vocabulary embedding matrix) of the
  masked-position state over the 50 steps, one line per method (policy, random),
  MH on, with the ground-truth token embedding marked. Expectation to describe:
  trajectories wander within the anisotropy cone without directed motion toward
  the target.
- Plot 2: the same for CLS with MH on and off, showing the with-MH trajectory
  pinned at its start (the flatline) and the no-MH trajectory jumping between
  Voronoi cells.
- Describe in prose what these show and connect to the anisotropy numbers (NN
  1.82, pairwise 2.77) so the section stands even before the plots are recreated.

## E. The AI tools appendix

Adapt the structure of the corresponding section from Ioanna's thesis. List the
tools actually used: Google AI Studio (Gemini 3.1 Pro) and Claude (Anthropic).
Describe the actual uses accurately: generating and debugging GPU experiment
scripts and infrastructure, analysis tooling, and, if Claude Code edits thesis
prose in Phase 3, assistance with editing and formatting the LaTeX. The
declaration must match what was really done, including this phase; an inaccurate
declaration next to a signed statement of authorship is the one risk in this
thesis that no result can offset. Cross-reference the appendix from the methods
or introduction per the university's convention, matching Ioanna's placement.

## F. Voice and style

- No em-dashes anywhere.
- Long, motivated paragraphs in the established voice of the existing chapters;
  match Ioanna's register and rhythm. Vary sentence length; avoid formulaic
  scaffolding (symmetric triads, "delve", stacked hedges, bullet-like prose in
  running text).
- Bold on first use of named concepts; dense citation; every claim about a
  number sourced.
- The supervisor's dissatisfaction was that the WHY was unclear. The narrative
  spine of the revised Results and Discussion is: the energy is fine (Gibbs and
  one forward pass work), the gradient is the defect (zero at the last position
  by causality, uninformative elsewhere by measurement), and training objectives
  that produce scores (diffusion) are the counterfactual. Every chapter should
  advance that spine.
