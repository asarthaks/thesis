# Prompt for Claude Code, Phase 9: docs sync, all evaluations resolved, proposal amendment, beamer

One session, Opus 4.8, highest reasoning setting. Paste the block below.

---

Read, in this order, before touching anything: REVISION_RESTRUCTURING.md (the
new layout: thesis at Doc/final/thesis/, proposal at Doc/final/proposal/, beamer
to be created at Doc/final/beamer/, references in refs/), CLAUDE.md, the closing
reports in REVISION_LOG.md, then refs/evaluation1.txt, refs/evaluation2.txt,
refs/evaluationproposal.txt, and thesis_questions_knowledge_base.md. Create
REVISION_WRITING.md at the repo root and log every action there, timestamped;
the beamer work logs to PRESENTATION_LOG.md. Both logs end with a "WHAT CHANGED"
summary section the author can read instead of the full log. No em-dashes.
Removed text goes to % comment blocks. All standing invariants hold: numbers
diff at the end, IMS compliance untouched, Ioanna register, no reorg beyond
what REVISION_RESTRUCTURING.md already did.

## Part 0: documentation sync

Update README.md and add an addendum to REVISION_RESTRUCTURING.md so both
reflect the final tree (Doc/final/thesis/, Doc/final/proposal/, Doc/final/
beamer/, refs/, meetings/, archive/). Sweep every % source comment in the
thesis tex for paths broken by the restructuring and update them (the
restructuring log says historical logs are not hand-edited; tex comments ARE
updated). Update the artifact map in README for the new paths.

## Part 1: the three numeric alarms from evaluation2 (before any prose work)

These may be real errors. Trace each to source data; fix data or text; log
full provenance for each.
1. Figure 9 vs Section 5.3: the figure shows within-cell 0.064 and boundary
   0.074 acceptance while the text reports 0.63% within-cell and 8.56%
   boundary with MH on. Determine what the figure actually aggregates (which
   configs, which trace files), regenerate it from the correct source or fix
   the text, and make the caption state the configuration exactly.
2. Table 12: CLS policy with MH and without MH show identical final KL (8.083)
   across all four models to three decimals, which 5.3 makes implausible.
   Inspect the table-generation code and the underlying JSONs for a join or
   duplicated-read bug; regenerate the table from verified reads; log the
   before and after values.
3. Figure 12 vs the -1.12 nats/token headline: the figure prints slope -0.11,
   r -0.00. Establish the exact provenance of -1.12 (which model, which
   generation set, censored or not); if it is a digit transposition of the
   GPT-2 -0.11, correct EVERY occurrence including 5.10's length-collapse
   argument and re-derive that argument's wording from the true numbers (the
   lb1-500 uncensored -0.505 and censored -2.361 are available); if -1.12 is
   real, put its source in the text and the figure. This is the single most
   important correction in this session.

## Part 2: the author's issue list (decisions encoded; apply as written)

1. TITLE PAGE GAP and ABSTRACT WIDTH, third strike: fix against the official
   template titlepage and body geometry, and this time the gate REQUIRES
   side-by-side PNG renders (our page vs template page) in the report. Hunt
   the stray \vfill/\vspace and the unrestored \newgeometry.
2. RQ BRIDGE: in 1.4, immediately after the four RQs, add the short paragraph
   stating these are the operationalized forms of the proposal's questions,
   refined when the study became diagnostic, followed by the compact mapping:
   proposal RQ1+RQ1a -> thesis RQ1; RQ1b -> RQ2; RQ3a -> RQ3; RQ2a -> RQ4;
   RQ2b (energy-based recovery vs AR decoding) answered by the gradient-free
   baselines, cite the section. Add one sentence in 6.1 noting the RQ2b
   resolution under its proposal name (negative for the samplers, affirmative
   for exact-energy rescoring).
3. Contribution 2: split into two or three shorter sentences.
4. BACKGROUND DIFFUSION GAP: add a compact subsection to Chapter 2 (score
   matching, absorbing discrete diffusion, the concrete score as trained
   probability ratios) sufficient for a student to follow the 5.13 positive
   control without external reading. Two thirds of a page, dense citation.
5. Marble analogy in 2.2: cut it; keep one plain sentence of intuition and
   the corrected annealing caveat (fair-draw interpretation holds only under
   technical conditions).
6. ROBERTS-TWEEDIE: replace the flagged sentence with the corrected wording
   the author supplied (discretization error grows with step size; MALA
   acceptance degrades under lost smoothness; convergence guarantees assume
   smoothness/log-concavity absent here; cite Roberts and Tweedie 1996 plus
   Dalalyan 2017 or Durmus and Moulines 2017, published venues). Then add ONE
   sentence where the boundary-rejection measurement is reported in results,
   linking the measured collapse to that predicted regime. Both papers into
   the bibliography, complete entries.
7. End of 2.2 continuous-sampler conclusion: audit the claim against the
   implemented CLS (piecewise-constant projected target vs smooth
   differentiable pathway within cells) and make the wording exact; it must
   agree with the two-objects passage in 2.4 that evaluation2 says to keep as
   is.
8. Taylor surrogate in 2.4: name the two terms explicitly (the distance term
   and the gradient-alignment term) and order the prose to match Equation 7's
   term order. Add the one-line footnote deriving Eq 7 (which constant terms
   are discarded), per evaluation2.
9. CLS UPDATE EQUATIONS: document the implemented step in methods: the
   interim continuous update, the nearest-neighbour projection, and the
   interpolated mean s_{t+1} = (s_interim + proj(s_interim))/2, with the
   reason (bounding off-manifold drift) and a note that it is an
   implementation choice adapted from prior work. This is currently missing
   and it is load-bearing for 5.4's off-manifold numbers.
10. GFlowNet metaphor in 2.6: keep, compressed to its shortest working form.
11. 3.4/3.5: de-duplicate the two "final" transitions.
12. 4.1 METRIC CLARIFICATION: add two or three sentences: the KL metric
    compares the recovered token's induced next-token conditional against the
    ground-truth-conditioned reference, so fitting synonyms score well; this
    is relative-to-reference contextual fit, not absolute likelihood as
    quality, and is therefore not in tension with the likelihood-trap
    finding, which concerns absolute sequence likelihood as a maximization
    objective. Forward-point to 4.4; remove the duplicated justification so
    the full version lives once in 4.4 (author issue and dedup in one move).
13. A2: keep, reframed as the released artifact map tied to the
    code-availability statement; appendix only.
14. 4.5 WHY LIKELIHOOD AS REWARD: add a short paragraph: gradient-
    uninformative does not imply value-uninformative (the thesis's own
    gradient-free results); the trap concerns the likelihood's extreme, not
    its ranking near the fluent manifold; and matching the base distribution
    is the standard amortization formulation (Hu et al.) whose premise is
    exactly what is under test.
15. CONFIG MATRIX: one sentence where the five-energy matrix is introduced:
    the GFlowNet-tuned models are included to test whether amortization
    repairs the landscape (the tuning demonstrably moved the energy; the
    samplers test whether it became navigable).
16. TASK-FORMULATION COMPARABILITY: add a short paragraph in the GFlowNet
    methods or limitations: the GFlowNet reformulates infilling as
    left-to-right generation of the blank under a restructured prompt, the
    samplers perform in-place substitution with bidirectional context; the
    comparison is therefore run at the energy level (final KL under the same
    base model), not the task level; and the left-to-right reformulation is
    not adopted for the samplers because it would dissolve the revision
    capability under study rather than repair its metric.
17. MUCOLA/COLD: state explicitly in related work and once in results that
    the CLS implementation follows the COLD/MuCoLa mechanism faithfully and
    the constrained "mucola" arm is that comparison; one sentence on why
    original-code task-level comparisons were out of scope. Mine the
    knowledge base thread D for the "how did they work without MH" answer
    and place its one-sentence version in related work.
18. Results-opening provenance sentence: remove; its content merges into the
    reproducibility statement.
19. 5.1 "qualifies the plug-and-play framing": replace with the plain
    statement that the step sizes required are much larger than the
    embedding-space distances between tokens.
20. Quenching: keep the term; compress the metallurgy origin to a clause.
21. FIGURE 1: fix the legend so dashed vs solid is distinguishable (legend
    handles without markers or longer dash patterns); state the gn
    configuration in the text near the figure; add or repair the gn-off
    companion panel where the three proposal arms separate, sourced from the
    gn-off runs; correct the stale source-path % comment to the actual
    generating script and path.
22. FIGURE 3: regenerate with distinguishable colors and alpha/zorder so
    overlapping lines do not blend into a third color.
23. 5.4: state the config for the DLS boundary-crossing count (MH on), add
    the no-MH DLS number, and regenerate fig_traj_stats.json to include both
    DLS configs (the missing key the author flagged).
24. 5.1: add the sentence naming the alpha grid searched and that 10.5->0.1
    came from the oracle calibration.
25. 5.5 SENTENCE SURGERY, per the author's list: split the norm-preserved
    rationale into two short sentences; compress the bounded-equivalence
    paragraph to the three-clause form (the difference is small, not
    significant, and the experiment could only have detected a gap of at
    least 0.652); rewrite the Llama anti-guidance paragraph as short
    declaratives (gradient direction hurts; random direction is neutral;
    ranking flips vs GPT-2; the CI excludes zero).
26. 5.5.3: state the generating configuration (sampler, mh, gn, arms) in the
    text; trace it from the judge scripts; and re-verify the judge analysis
    code path end to end, logging the verification, since the 178.4 vs 181.3
    proximity made the author uneasy.
27. 5.6: rewrite the 2.35-vs-1.82 sentence and the "first order does not
    survive the jump" sentence in plain form (candidate moves are longer
    than nearest-neighbour spacing; the linear approximation has broken down
    at that distance); clarify 15.0/24.2 as mean absolute log-likelihood
    changes in nats.
28. FIGURE 10: regenerate the scatter with clearly distinct colors.
29. 5.8: separate scale from anisotropy per the author's wording, and add
    the clause reconciling the cosine measure with the PCA note in A.4.
30. 5.10 and 5.7: add the generator attribution for degenerate strings (the
    argmax decoder under the frozen LM produced them; the GFlowNet then
    up-weights them under the length-biased reward).
31. 5.11: bold or otherwise mark the infilled token in the example, and add
    the drift-mechanism sentence (a small unconditional negative drift
    dominates raw gains; the paired contrast cancels it) before the numbers.

## Part 3: evaluation2 resolutions (beyond Part 1)

1. CORPUS MAP: one small table in Chapter 4 mapping every experiment family
   to its corpus (ROCStories vs WikiText-2), plus one sentence addressing
   whether the domain shift could inflate the null (note the in-domain
   ROCStories results agree, so it does not).
2. 5.11 RIGOR: tabulate the paired contrasts (+27.3, +36.7) with n and
   whatever uncertainty the aggregate JSONs support; where per-sample gains
   do not exist, say so in the table note explicitly rather than inventing
   CIs; state the n and analysis plan for the constrained extension in 4.6.
3. Four-vs-five diagnostics count: fix 4.7 and the conclusion to agree.
4. PROPOSAL DELTA in 6.4: one sentence per dropped deliverable (toxicity/
   RealToxicityPrompts, MAUVE, Self-BLEU/diversity suite, 70B judge,
   posterior-coverage metric) explaining why each became moot when the study
   turned diagnostic.
5. Bibliography: full author lists (no et-al truncation in the reference
   list) or one named style's policy applied uniformly; fix Brown, Dubey,
   Bengio 2023, Hu 2022.
6. Related work: one bounded mention of straight-through/Gumbel-softmax
   relaxations as the third adaptation strategy and why out of scope.
7. Gibbs wording: "comparable to", not "matching", for 6.69 vs 6.4-6.5.
8. Abstract: ONE PAGE HARD LIMIT (author requirement; it is currently ~2
   pages), one numeral convention throughout, and one closing sentence on
   the practical implication (evaluate the energy, do not differentiate
   it). Kurzfassung in lockstep if present. Render the page as proof.
9. 1.2 reveal: add the one clause acknowledging the outcome is stated before
   the RQs because the thesis is organized as an explanation, not a reveal.
10. Figure 13 token strips: enlarge or widen for print legibility.

## Part 4: evaluation1 remaining trims

Apply its specific cuts list (introduction 0.5-1 page; Langevin background
0.5; DLS/CLS background 1; related work 1-1.5; plus its conclusion trims).
Reduce the verbatim central formulation to two or three occurrences (abstract,
5.5, 6.1) and paraphrase elsewhere. Prune the evaluate-vs-differentiate
repetition to one canonical statement plus one callback. State the diffusion
three-result summary once in 5.13's intro and once in 6.1, nowhere else.
Classifier-alignment explanation: once in full, elsewhere one clause.

## Part 5: the proposal amendment (Doc/final/proposal/; bounded, minimal)

Apply ONLY: (a) the contingency section at the end, future tense, proposal
voice: the pivot condition (should the rigorous frameworks fail to improve on
heuristics, the thesis turns to mechanistic diagnosis, treating negative RQ
answers as findings to characterize), the diagnostic instruments named as
planned tools, section 2.4's diffusion remark promoted to the explicit
fallback hypothesis (a score-trained model as positive control), RQ3a's
trained-proposal clause folded in, one sentence reserving dataset/metric
adjustment (5.3 already allows it), and one sentence that under the
diagnostic branch the RQs will be sharpened into diagnostic form preserving
their substance. (b) The proposal evaluation's major fixes: the sign error
(state the update in ascent form on log p or descent on U, consistently, and
fix theta_t - eta grad log p -> plus, or restate via U = -log p), rename
lambda from "penalty weight" to "constraint weight", resolve the eta/step
size notation collision with Eq 1, correct the two RQ cross-references (4.1
-> RQ1a; 4.5 also serves RQ2's quality floor), complete the bibliography
entries with venues and years, fix "fullfills", the quote marks, and any
encoding artifacts. NOTHING else changes; produce a diff summary proving the
rest is untouched; compile the proposal; draft the five-line note to the
supervisor (amendment adds the agreed contingency section, RQs unchanged,
minor errata fixed, one-line thesis-to-RQ mapping).

## Part 6: the beamer presentation (Doc/final/beamer/, PRESENTATION_LOG.md)

Locate the beamer template (in Doc/final/beamer/ or refs/; if absent, STOP
and ask the author rather than inventing one). Read
refs/08_ims-theses_handout.pdf for the department's talk expectations. Build
a 20-minute defense talk: 14-16 content slides plus backups. Story arc: the
revision problem AR cannot solve natively; the energy-based promise and its
untested assumption; the faithful implementation (DLS/CLS, MH, oracle
calibration); the null (gradient vs norm-matched random, one slide, one
figure); the why (linearization failure with the jump-size picture; MH
breakdown with the within-cell vs boundary decomposition; one slide each);
the energy still works (rescoring, Gibbs); GFlowNet does not escape it (one
slide, three failure modes compressed); the last-token exact result (one
slide, the theorem in one line plus the table row); the diffusion positive
control and the hybrid 0-to-39 percent (one slide, the flagship); takeaway
(evaluate, do not differentiate; training objectives determine landscapes).
Graphics: clean TikZ or SVG diagrams for the energy-landscape cartoon, the
Voronoi cell picture with step sizes to scale, the sampler loop, and the
hybrid architecture (SEDD proposes, AR energy accepts); regenerate result
figures at slide resolution with large fonts; no dense tables, one number
per claim. Backup slides: mine thesis_questions_knowledge_base.md, one
backup slide per confusion hotspot (linearization radius and self-term
blindness, MH chain, likelihood-is-bad-yet-everywhere, MuCoLa/COLD, task
design, GFlowNet framing), each answering the question in one visual plus
three lines. Ioanna-register slide text: short noun phrases, no bullet
walls, no AI-flavored phrasing. Compile clean; render four representative
slides to PNG for the report; log design decisions and the slide-by-slide
minute budget in PRESENTATION_LOG.md.

## Part 7: gates and the final report

latexmk clean on thesis, proposal, and beamer; zero undefined refs and
citations; numbers diff against numbers.json plus every phase JSON plus the
regenerated fig_traj_stats.json and any Part 1 corrections (Part 1 changes
must be reflected in numbers.json, not diffed away); render proofs: title
page vs template, one-page abstract, Figures 1, 3, 9, 10, 12, 13, four
beamer slides. Resolution tables in REVISION_WRITING.md: (a) every item of
the author's issue list, (b) every evaluation1 item, (c) every evaluation2
item, (d) every proposal-evaluation major item, each with action and
location. Then the WHAT CHANGED summary: one paragraph per document (thesis,
proposal, beamer, markdown docs) naming the sections that changed and why,
so the author re-reads only those. Expected author decisions in the report:
none beyond confirming the Part 1 corrections if any number changed.

Constraints: Part 1 before all prose work; nothing beyond this slate; no
em-dashes; removed text recoverable; Ioanna register throughout; the
proposal changes are exactly Part 5's list and nothing more; if any Part 1
trace is ambiguous, present the evidence and stop rather than guess.
