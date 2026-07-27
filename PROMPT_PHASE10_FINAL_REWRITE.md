# Prompt for Claude Code, Phase 10: the final rewrite around the changed findings

One session, Opus 4.8, highest reasoning setting. This is the finalization pass:
after it there are NO further experiments and no further structural changes.
Paste the block below.

---

Read, in order: REVISION_RESTRUCTURING.md (layout), the evaluation3-pass entries
in REVISION_LOG.md (Parts A through D5; these define the current findings),
refs/evaluation4.md in full, REVISION_WRITING.md's resolution tables, and the
refs/ guidelines (Guidelines_for_academic_thesis_writing_at_the_IMS.pdf,
Checklist_Masterthesis.pdf, the template). Append all work to REVISION_WRITING.md,
timestamped, ending with a WHAT CHANGED summary. Beamer work logs to
PRESENTATION_LOG.md.

## Hard constraints (non-negotiable, from the author)

1. NO NEW EXPERIMENTS. Every number comes from existing JSONs and CSVs. If an
   edit seems to need a run, write the honest sentence the existing data
   supports instead, and log it.
2. TEMPLATE UNTOUCHED. baselinestretch 1.3, margins, geometry, title page,
   Erklaerung: exactly as they are. Length is won by cutting prose, never by
   layout tricks.
3. IOANNA REGISTER throughout, per the established style memo: noun-phrase
   headings, no formulaic scaffolding, no em-dashes, long motivated paragraphs.
4. PRIOR FIXES PRESERVED. Every resolution in REVISION_WRITING.md's tables
   (the Phase 9 author list, evaluation1/2/3 items) must still hold after the
   rewrite, EXCEPT where a changed finding supersedes it; each supersession is
   logged explicitly in the resolution table (precedent: the quenching
   override), never silent. Knowledge-base answers that referenced withdrawn
   findings are stale by definition; do not re-litigate them, note the
   supersession once.
5. Abstract stays within one page.
5b. TOTAL DOCUMENT LENGTH: 115 to 119 pages, hard limit, reached WITHOUT losing
   concrete and important content and WITHOUT layout tricks (constraint 2) and
   WITHOUT exiling the evidence to the appendix: the main text keeps its
   figures and tables. Every core result named in constraint 6 remains
   presented in the body with its figure or table. The length is won from
   prose (the Part 3 purge), from merging overlapping sections, and from the
   hypothesis architecture removing recaps, not from stripping the body of
   its evidence. If, after the full Part 3 program, the document still
   exceeds 119 pages, do NOT start cutting substance or floats: stop, report
   the page count and the remaining candidates with a one-line cost
   assessment each, and let the author choose.
6. The findings of record are the evaluation3-pass results: quenching
   withdrawn; near-uniform proposal with the accept/reject filter doing the
   work (uniform control corroborates); equivalence CERTIFIED at n=1000;
   anti-guidance withdrawn (MH asymmetry in sharp cells; Llama contrast hedged
   on multiplicity and the MH-off optimizer point, non-reproduction on GPT-2);
   the one-hot/output-side result (surrogate rho 0.60-0.73 near; sampler 40.0%
   GPT-2, 41.0% Llama, vs 0.0% embedding gradient); the conditioning ladder
   with RoBERTa 44.5% > SEDD 39.0% > AR conditional 23.5%; attribution is
   "what the proposal may condition on", not the training objective; the
   temperature was the binding constraint on the calibrated grid.

## Part 1: the new argumentative architecture (the heart of this pass)

Adopt evaluation4's hypothesis skeleton as the thesis's explicit structure:

- In 1.2, after the assumption is isolated, state the three candidate
  explanations ONCE: (A) the target energy itself is unusable; (B)
  autoregressive training fails to contain the information a local proposal
  needs; (C) the input-embedding parameterization discards accessible
  information, and right-context conditioning supplies more of it.
- Reorganize the results narrative (prose and transitions, not the section
  numbering beyond what Part 3's merges require) so it reads as sequential
  elimination: the gradient-free baselines and exact-energy results eliminate
  A; the one-hot re-analysis and one-hot sampler eliminate B, since the same
  frozen weights yield 40 percent when differentiated on the output side; the
  conditioning ladder confirms and refines C. Every place the document
  currently "changes its mind" (the 5.2 withdrawal, the attribution shift from
  objective to conditioning, the anti-guidance hedge) is rewritten to read as
  the designed next test in this progression, with the withdrawal sentences
  kept (honesty) but placed as findings, not as errata.
- The conditioning-ladder table becomes the organizing artifact of the late
  results: introduce it once, refer back to it, never rebuild it in prose.
- REFORMULATE THE RQs per evaluation4: RQ1 in its recommended precise form;
  RQ2 with the over-broad statement narrowed (its item 7); RQ3 split into its
  two conflated questions (its item 8), answered separately in 6.1; RQ4
  repositioned as an EXTENSION (its item 9, the author's decision), with 5.11
  framed accordingly. Preserve the proposal bridge in its compressed one-
  paragraph form (evaluation4 item 11.1) so the proposal mapping survives the
  reformulation; update the mapping to the reformulated RQs.
- Rewrite 6.2 per evaluation4 item 6: one shared high-level problem (the
  proposal cannot see token fitness), distinct lower-level mechanisms named
  separately (the Jacobian slice, the linearization radius, the MH geometry,
  the conditioning direction); drop the claim that a single mechanism unifies
  everything.
- Contributions (evaluation4 item 4): recount, renumber, one to two sentences
  each, consistent with the abstract and the conclusion.

## Part 2: terminology precision (evaluation4 item 2)

Adopt evaluation4's recommended terminology for the one-hot result exactly,
and apply it globally: the quantity must be named so that it is mathematically
unambiguous which derivative, with respect to which coordinates, under which
parameterization (the relaxed one-hot/simplex input coordinates versus the
embedding coordinates), and the closed-form identity (log p(v | x_<i) plus the
embedding-gradient inner product) is stated once where the object is defined.
Sweep every occurrence, including captions, the abstract, and the conclusion,
for the imprecise name and replace it.

## Part 3: repetition purge and length (evaluation4 items 10 and 11, complete)

Apply every reduction in evaluation4's item 11 list, and resolve every
repetition it names in item 10: the 5.5/5.6/5.7/5.13-family overlaps collapse
into the Part 1 progression (each result stated once in full, thereafter one
clause); 5.14/5.15 merged or trimmed per its instruction; 6.1 answers RQs with
references into Chapter 5 rather than re-arguing; the conclusion reports
without a third synthesis; Background 2.5 and 2.6 and Related Work 3.2
compressed as it specifies. The hypothesis architecture itself removes the
need for most recaps; where a recap survives, it is one sentence. Target: the
hard limit of constraint 5b, TOTAL 115 to 119 pages (from 142), body
correspondingly toward the high 70s to low 80s, with NOTHING the evaluators
praised removed (6.4 limitations, the failure taxonomy, A.6 selection policy,
the statistics apparatus stay) and the body's figures and tables staying in
the body. The appendix is also in scope for trimming (consolidated showcases,
superseded exploratory detail), since appendix pages count toward the total;
but moving a body float to the appendix is NOT a permitted way to shorten the
body. Log per-chapter and appendix page deltas against the 142-page start.

## Part 4: consistency sweep for the changed findings

Grep the entire thesis for orphaned claims of the old story and fix every
hit: quenching as a live mechanism (the term may appear only in 5.2's
withdrawal statement), any anti-guidance assertion, any "training objective is
the cause" attribution, any "0 to 39" contrast lacking the gradient-free and
one-hot comparators, any statement that the gradient "carries no usable
signal" unscoped to the input-embedding parameterization. The abstract,
introduction, 6.1, 6.2 and the conclusion must all carry the same final claim
set: certified equivalence for the embedding-gradient proposal; the output-
side derivative of the same frozen model recovers 40 to 41 percent; the
conditioning ladder attributes the remaining gap to right-context access; the
practical implication updated from "evaluate, do not differentiate" to its
corrected form (differentiate the right object, or condition on the output
side). Numbers diff at the end.

## Part 5: guidelines re-verification (refs/)

After the rewrite, re-walk the IMS checklist and guidelines quickly: title
page fields, Erklaerung untouched and correctly placed, lists of tables and
figures regenerated (they changed), bibliography rules (published venues,
full author lists, URLs), every abbreviation introduced at first use in its
possibly-new first location, every RQ raised in 1.4 answered in 6.1 under the
REFORMULATED wording, every appendix item cross-commented. Log the checklist
table.

## Part 6: the proposal (Doc/final/proposal/)

Verify, do not rewrite: check the amended proposal (contingency section)
against the final story for contradiction. The contingency promised
mechanistic diagnosis and named a score-trained model as a positive control;
the final attribution is conditioning, not objective. That is a finding the
contingency permits, not a contradiction, so the expected verdict is NO
CHANGE; log the verification sentence by sentence for the contingency
section. Only if a sentence positively contradicts the thesis (log the exact
sentence) may it be minimally adjusted, future tense, proposal voice, and the
diff summarized.

## Part 7: the beamer (Doc/final/beamer/, PRESENTATION_LOG.md)

The deck was built on the old story and now contradicts the thesis. Update:
the quenching slide becomes the near-uniform proposal and temperature slide
(entropy floor 10.28 of 10.82; uniform control reproduces the flagship); the
attribution slide becomes the conditioning ladder including RoBERTa; ADD the
one-hot headline slide (0 to 40 percent on GPT-2, 0 to 41 on Llama, same
frozen model, same sampler, different derivative; this is now the talk's
flagship alongside certified equivalence); the takeaway slide carries the
corrected practical implication; the hybrid slide gains the 33 percent
gradient-free comparator so the 39 is honestly framed. Re-check every backup
slide against the final claims; retire the ones answering withdrawn-finding
questions and add backups for the new obvious questions (why did the
temperature flatten everything; what exactly is the one-hot derivative; why
does RoBERTa beat SEDD). Minute budget re-verified for 20 minutes. Compile;
render the changed slides to PNG.

## Part 8: gates and the final certification

latexmk exit 0 on thesis, proposal, beamer; zero undefined references,
citations, duplicate labels; abstract page render proving one page; total
page count within the 115-to-119 hard limit (constraint 5b), with the
body-floats check: every core result of constraint 6 still has its figure or
table in the main text, verified by listing them with their page numbers;
numbers diff ALL OK; the prior-fix preservation audit (Part 0 constraint 4) as a
table: every prior resolution, still-holds or superseded-with-log-reference;
evaluation4 resolution table, every numbered item, action and location; the
WHAT CHANGED summary per document; and the closing statement: findings of
record integrated, story coherent under the hypothesis architecture, no
experiments outstanding, thesis final.

Constraints: no experiments; template untouched; Ioanna register; no
em-dashes; removed text to % comments; every supersession of a prior fix
logged, never silent; if a required edit conflicts with a hard constraint,
keep the constraint and log the conflict for the author.
