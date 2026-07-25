# Prompt for Claude Code, Phase 8: resolve the external evaluation, tighten to final

One session, Opus 4.8, highest reasoning setting. The full external evaluation is
saved in the repo as evaluation_feedback.md (put the text there before starting).
Paste the block below.

---

Read CLAUDE.md, the closing reports in REVISION_LOG.md, and
evaluation_feedback.md in full before touching anything. This pass resolves
every point in the external evaluation. The verdict to internalize: the spine is
strong, the RQs are answered, the defects are over-explanation, repetition, and
claim overreach. The remedy is compression and scoping, never cutting
experiments. Log everything to REVISION_LOG.md, timestamped. No em-dashes.
Removed text goes to % comment blocks. All prior invariants hold: numbers and
source comments unchanged (numbers diff at the end), IMS compliance fixes
untouched, Ioanna register maintained, no sentence of hers reused.

Note: the previously planned style pass (Phase 7) was never run; this session
does its work too. So begin with STYLE CAPTURE, before any rewriting: read
Ioanna's thesis (IoannaThesis/Ioannathesis.tex and includes) end to end and
write a style memo into REVISION_LOG.md with verbatim examples of (a) her
section-title grammar (expect noun phrases, never questions; quote five
headings), (b) how she opens chapters and sections, (c) her sentence rhythm
and voice, (d) her caption format (quote three), (e) how much interpretation
lives in text vs caption, (f) paragraph length and transitions. Also list the
AI-flavored patterns to remove from our text (question-form headings,
formulaic openers like Notably/Crucially/Importantly, symmetric triads,
stacked signposting). Then rewrite ONE section (5.2) and put the before and
after in the log as the calibration sample. All rewriting in this session
happens in that register. Her thesis is a style reference only: imitate
register and format, never reuse any sentence or phrase of hers.

## A. Claim scoping (do this first; it propagates everywhere)

Replace the four overreaching formulations with the evaluation's scoped
versions, consistently in abstract, introduction, results, discussion, and
conclusion. The same scoped sentence appears everywhere the claim appears:

1. Central claim: "In the tested token-substitution settings, the
   input-embedding gradient of frozen autoregressive sequence likelihood
   provided no reliable proposal advantage over a norm-matched random
   direction." Never the universal "carries no usable search direction on
   discrete text" without the scope.
2. Last-token result: always specify which gradient, with respect to which
   representation (the final token's input embedding), under the shifted
   causal likelihood indexing. Never a bare "provably zero at the final token".
3. GFlowNet: "The combined GFlowNet and diffusion results support the
   interpretation that the training objective, rather than only the sampling
   algorithm, is a key source of the failure." Replace every causal "because
   it is a property of the training objective".
4. Training-free premise: "The experiments refute the premise that the frozen
   autoregressive sequence-likelihood gradient can supply the required local
   score for this class of Langevin-based methods without additional
   training." Note explicitly that gradient-free training-free methods
   (conditional resampling, Gibbs) remained viable, which the thesis's own
   results show.

Also sharpen the RQ4 answer into the evaluation's two-statement split, stated
in the discussion's RQ answer: on the frozen autoregressive landscape,
plug-and-play constraint steering failed; on a score-trained diffusion
landscape, constrained steering became partly possible, subject to fluency and
classifier-alignment limits. Do not blend them.

## B. Rhetoric dial-down

- Neutral section headings everywhere. Retain exactly two coined labels in the
  running text: "likelihood trap" and "linearization failure". "Gradient
  fallacy" survives in ONE discussion paragraph that introduces it as the
  thesis's shorthand and immediately scopes it (per A1); it disappears from
  all headings and recurring use. Flag this in the report as the one default
  the author may overrule.
- Remove or rewrite the theatrical recurring phrases the evaluation quotes
  ("the central business of this thesis", "a closed case against", "protected
  from their own objective", "the strange silver lining", "the falsifying
  experiment", and kin). One memorable phrasing per idea, at first use, then
  plain reporting.
- Kill the default five-beat paragraph pattern (result, why it matters, how it
  fits, what it raises, mechanism recap). A results paragraph states the
  result and the one reading the reader needs; the mechanism synthesis lives
  in the discussion, once.

## B2. Retitling and float pruning (from the unrun style pass)

- RETITLE: convert every question-form or AI-flavored heading to Ioanna-style
  noun-phrase headings, per the style memo. Full old-to-new heading table in
  the log. Keep every \label key unchanged so \ref and the RQ-to-subsection
  map survive; update the map's displayed titles.
- FLOAT PRUNING, RQ-gated: a figure or table stays in the main text only if
  the running text cites it as direct evidence for a named RQ or the diagnosis
  chain answering them. Expected body set: trajectory distance figure,
  last-token table, KL baselines table, stats table with CIs, hybrid table,
  linearization table, and the one-page exploratory steering summary's single
  table. Secondary PCA, showcases, examples, per-strategy likelihood-trap
  detail, cost table live in the appendix with one cross-reference sentence
  each. Anything serving no RQ and no diagnosis step goes to % comments. Log
  the float-by-float decision table. This dovetails with C10 and C13: apply
  them together so floats move once, not twice.

## C. Structural cuts, in the evaluation's priority order

1. ABSTRACT: hard limit one page, target 350-500 words. Keep: problem, tested
   models and samplers, the scoped central result (A1), the principal
   mechanism in one sentence, GFlowNet outcome in one sentence, diffusion
   positive control in one sentence, one restrained closing sentence. Cut:
   the three-claim hierarchy, classifier-transfer qualifications, the full
   last-token explanation. The scoped claims from A keep it honest at this
   length. Update the Kurzfassung in lockstep if present. Render the abstract
   page and confirm it fits one page.
2. Section 1.4 "The Shape of the Investigation": remove; replace with one
   sentence at the end of the introduction ("Because the initial sampling
   experiments failed, the study developed into a diagnostic investigation of
   the underlying mechanisms.").
3. Section 1.5 contributions: compress to four or five tight contributions,
   one to two sentences each; no miniature results chapter.
4. Thesis roadmap: one concise paragraph.
5. Background de-anticipation: keep every derivation; remove the repeated
   "this surrogate is the sampler's entire claim" sentences (state once) and
   ALL measured outcomes from the background (the CLS acceptance numbers,
   "well under a percent", boundary splits move to Results only). Halve the
   marble analogy and correct its convergence claim: annealed step sizes and
   the "fair draw from the landscape" interpretation hold only under
   technical conditions; say so in one clause rather than implying it
   unconditionally.
6. Results recaps: every section opener that summarizes the previous section
   compresses to at most one sentence. The likelihood-trap section reports
   its result and drops the re-derivation of the central claim.
7. Section 5.4 trajectory analysis: main text becomes a one-page summary with
   the off-manifold numbers and the geometric conclusion; the figures and
   full walkthrough are already in the appendix; point there.
8. Section 5.9 cross-model consistency: one summary table, one paragraph on
   the GPT-2 family, one paragraph on Llama-3 scope.
9. Last-token section: reframe as the structural confirmation of the already
   established result (its distinct contribution is the exactness), not
   another full central-result chapter; compress accordingly.
10. Guided diffusion (5.13.4, 5.13.5): the positive-control core stays in the
    main text (5.13.1 signal, 5.13.2 recovery, 5.13.3 hybrid). The steering
    study compresses to a ONE-PAGE main-text subsection explicitly labeled as
    exploratory follow-up, stating: the constraint direction carries signal,
    guidance steers the guiding classifier, external transfer is bounded by
    guide-judge agreement, and the trust region removes the fluency cost.
    Full tables, agreement ladder, confusion analysis, and examples move to
    the appendix with one cross-reference. The RQ4 two-statement split (A)
    carries the honest summary.
11. Discussion and conclusion: 6.1 (direct RQ answers) stays as is. Merge the
    re-synthesis overlap in 6.2 through 6.5: mechanism synthesis once,
    literature links once, limitations once, implications once. Conclusion
    rewritten to two to three pages: what was tested, what was found, what
    remains uncertain, the main practical implication. No third recap of the
    evidence chain.
12. Captions: adopt the rule "concise caption plus full prose" everywhere
    (the checklist requires prose walkthroughs, so prose wins). Use
    \caption[short]{...} so the lists of figures and tables show one-line
    entries. Any caption still carrying interpretation, limitations, or
    comparisons loses them to the adjacent text.
13. Qualitative appendices: consolidate the overlapping showcases
    (guided-generation examples, recovery showcase, trust-region
    before/after, ten seeded sequences) into ONE appendix section with a
    single stated selection policy and a smaller representative set; the
    remainder goes to % comments.

## D. Emphasis triage (within the existing chapter structure)

Do NOT renumber chapters or merge Background and Related Work; the evaluation
calls the current separation defensible and a restructure this late is risk
without benefit (log this as considered and declined, with this reason). But
apply the evaluation's three-tier emphasis inside the existing structure: core
evidence (gradient vs random, linearization, gradient-free baselines, MH
decomposition, diffusion positive control) gets full sections; supporting
diagnostics (anisotropy, quenching, trajectories, likelihood trap, cross-model
calibration) get compact sections that point to appendix detail; exploratory
extensions (GFlowNet failure taxonomy detail, guided diffusion steering,
trust region, guide-judge agreement) are labeled exploratory where they
appear. The label is honest signposting, one clause, not a new rhetorical
device.

## E. Mathematical precision item

In the CLS discussion, keep the implemented surrogate pathway and the target
density carefully distinct: the projected energy is piecewise constant (no
classical gradient), the differentiable pathway is smooth within cells, and
the drift discontinuity claim attaches to the pathway across cell boundaries.
Audit every sentence in the CLS sections against this distinction and fix any
wording that implies a classical gradient of the piecewise-constant energy is
being used.

## E2. The two layout bugs (from the unrun style pass)

1. Title page: remove the oversized vertical gap between the author name and
   the Studiengang table; match the official template's titlepage (name, then
   \vfill, then the tabular; hunt the stray \vspace or doubled \vfill).
   Render and compare side by side against the template PDF's page 1.
2. Abstract width: the abstract is set wider than the text block; find the
   cause (unrestored \newgeometry, minipage, or custom environment) and give
   the page the body's 3cm/3cm geometry. While there, sweep for any other
   page whose geometry deviates (an unrestored \newgeometry propagates
   silently).

## F. Page targets and the gate

Target after all cuts: main text (before references and appendix) reduced by
roughly 20 percent, cutting prose not experiments; abstract one page;
conclusion two to three pages. Per-chapter page deltas logged. Then the gate:
latexmk clean, zero undefined refs and citations, lists of figures and tables
regenerated with the new short captions, numbers diff clean against
results_revision/numbers.json and all phase JSONs, invariants verified,
render the abstract page, title page, one core results page, the exploratory
subsection page, and the conclusion opening to PNG and inspect.

## G. The resolution report

Close with a point-by-point table in REVISION_LOG.md addressed to the
evaluation: every numbered concern and every section-8 formulation from
evaluation_feedback.md, the action taken, the location in Doc/, and done or
author-decision status. Expected author decisions: the gradient-fallacy
default (B), and nothing else. The report ends with the new page count and
the statement that every evaluation item is resolved or explicitly deferred
to the author.

Constraints: compression and scoping only; no experiment, number, or result
removed from the record (appendix and % comments preserve everything);
maintain Ioanna's register from this session's style memo in all rewritten
prose, and check the 5.2 calibration sample against the memo before
proceeding past it; no em-dashes; if any cut would conflict with an IMS checklist item or
an invariant, keep the content and log the conflict.
