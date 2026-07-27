# PRESENTATION_LOG.md

Timestamped, append-only log of the defense-talk (beamer) build, Part 6 of
PROMPT_PHASE9_FINAL_DOCS. Design decisions, the slide-by-slide minute budget,
and the four rendered proof slides are recorded here. Thesis and proposal work
is logged in REVISION_WRITING.md.

Standing invariants: Ioanna register (short noun phrases, no bullet walls, no
AI-flavored phrasing); one number per claim; no em-dashes; numbers agree with
numbers.json and the regenerated figures.

The file ends with a "WHAT CHANGED" summary.

---

## 2026-07-26 00:28 CEST  Not yet started

Beamer build begins after Part 1 (numeric alarms) is confirmed and the thesis
prose passes are underway, per the Part 6 sequencing. Template located at
`Doc/final/beamer/Presentation.tex` (plus `bg.png`); `refs/08_ims-theses_handout.pdf`
holds the department talk expectations and will be read before slide design.

---

## 2026-07-26 ~15:30 CEST  PART 6: beamer defense talk BUILT

Reference deck evaluated first (Doc/final/beamer/Final_presentation_SU.pptx, Sergei Ukhov's
defense, same examiners Vu and Schweitzer, 21 slides). Adopted from it: the three-act
structure with section dividers (Motivation / Method / Results / Conclusions), the rhythm of
stating the claim or question before showing each result, a concrete worked example placed
early, and crisp closing takeaways. Avoided: its text-dense slides (full-sentence hypothesis
walls); the Ioanna register here is short noun phrases, one number per claim, no bullet walls.

Template: built on the located Doc/final/beamer template (Madrid/whale beamer), retitled for
the IMS with a deep-blue structure colour, the institute and both examiners on the title
page, and the UNIBG logo removed. inputenc switched latin1 -> utf8; bm added.

Story arc (follows Part 6 exactly), 14 content slides + 6 backups:
 1 title; 2 the revision problem AR cannot solve; 3 the energy-based promise and its one
 untested assumption (TikZ energy-landscape cartoon); 4 the four RQs; 5 the faithful
 implementation, DLS/CLS, MH, oracle calibration (TikZ sampler loop); 6 the null, gradient vs
 norm-matched random (paired CI + the four-energy table); 7 why 1, linearization failure with
 the jump-size picture (TikZ Voronoi cells with the valid radius vs the smallest move to
 scale, plus fig_lin_radius); 8 why 2, the MH breakdown, target +4.60 vs proposal -1325
 (fig_mh_decomposition); 9 the energy still works, rescore 4.43 and Gibbs 6.69; 10 GFlowNet,
 three failure modes compressed; 11 the last-token exact result, the zero-gradient theorem in
 one line plus the position table; 12 the diffusion positive control and the hybrid 0->39
 percent (TikZ hybrid architecture, SEDD proposes / AR energy accepts); 13 takeaways
 (evaluate, do not differentiate; training objectives determine landscapes); 14 thank you.
 Backups (mined from thesis_questions_knowledge_base.md, one per confusion hotspot):
 linearization radius and self-term blindness; the MH chain; likelihood-bad-yet-everywhere;
 MuCoLa/COLD and why they appear to work; task design and corpus; why a GFlowNet and the
 task-comparability caveat.

TikZ diagrams built (large fonts, controlled): energy-landscape cartoon (jagged 1D energy,
"gradient points where?"); Voronoi/token-spacing picture with the valid radius and the
smallest admissible move to scale; the sampler loop (state -> gradient -> MH -> project);
the hybrid architecture (SEDD score -> proposal -> MH accept by AR energy -> recovered token).
Result figures embedded as the thesis vector PDFs (fig_lin_radius, fig_mh_decomposition,
fig_lin_scatter), legible at slide scale; conceptual figures are the purpose-built TikZ.

Minute budget (target 20 min, IMS allows 25-30): title 0.5; motivation 4.5 (1.5 + 2 + 1);
method 2; null 2; why 5 (2.5 + 2.5); energy-works 1.5; amortization + last-token + positive
control 5 (1.5 + 1.5 + 2); takeaways 1. Total ~21.5 min, leaving buffer inside 25 min.

Compile: latexmk exit 0, 20 pages, 0 missing figures, no overfull above 40pt. Four proof
slides rendered to Doc/final/proofs/: beamer_energy_promise.png (slide 3),
beamer_linearization.png (slide 7), beamer_positive_control.png (slide 12),
beamer_takeaways.png (slide 13). Two minor right-margin TikZ text overflows found on first
render (slide 7 caption, slide 12 annotation) and fixed.

### WHAT CHANGED (beamer)
A complete 20-minute defense deck was created at Doc/final/beamer/Presentation.tex, replacing
the placeholder template content. It follows the diagnostic story arc from the revision
problem through the null, the two mechanisms, the gradient-free baselines, the GFlowNet and
last-token results, to the diffusion positive control and the takeaway, with four original
TikZ diagrams and six knowledge-base-derived backup slides. It compiles clean and is rendered
to four proof slides. No thesis number was changed by this part.

---

## 2026-07-26 ~21:30 CEST  PHASE 10: deck rebuilt on the evaluation3/evaluation4 findings

The Phase 9 deck was built on the old story and contradicted the thesis in three places
(the attribution to the training objective, the anti-guidance claim in a backup, and the
absence of both the near-uniform measurement and the token-indicator result). Rewritten
against the findings of record. Thesis work for this phase is logged in REVISION_WRITING.md.

### Slides changed, added, retired

| # | slide | action |
|---|---|---|
| 4 | Research questions | REWRITTEN to the reformulated RQ1 (input-embedding scoped), RQ2, the RQ3a/RQ3b split, and E as an extension; closing line names the three candidate causes A/B/C |
| 5 | The proposal the main grid actually draws from | NEW. Entropy 10.8248 at step 0, floor 10.28 against the uniform ceiling 10.8249, effective support ~29,000 tokens, gradient term ~0.009 nats; side table gradient 0.0%/6.541 vs uniform control 0.5%/6.538; block stating the 5x5 sweep that re-tests the null |
| 6 | Two samplers | minor wording (overfull fix) |
| 11 | GFlowNet | closing line changed from "the failure is a property of the training objective" to the RQ3a/RQ3b split, both negative |
| 13 | The headline: same model, same sampler, different derivative | NEW, replaces the old positive-control slide as the flagship. Two-role relaxation, the closed form, rho 0.03 -> 0.73 near, exact 0.0% -> 40.0% GPT-2 and 41.0% Llama, and the temperature note (at T=5 even this proposal falls to 2%) |
| 14 | The conditioning ladder | NEW. Full ladder incl. RoBERTa 44.5% > SEDD 39.0% > AR conditional 23.5% > uniform 0.5% > gradient 0.0%, with the gradient-free comparator (top-k rescore 33%, Gibbs 18.5%) so the 39 is honestly framed; TikZ shows only the proposal changing |
| 15 | Takeaways | REWRITTEN. Certified equivalence; not the target and not the model; right-context access rather than a score objective. Closing line changed from "Evaluate the energy; do not differentiate it" to "Differentiate the right object, or propose from the output side", with the explicit correction of the old slogan |
| B5 | Backup: task design and corpus | anti-guidance claim RETIRED; now states multiplicity, the MH-off confound, and non-reproduction on GPT-2 sharp cells, concluding indifference |
| B6 | Backup: why did the temperature flatten everything? | NEW |
| B7 | Backup: what exactly is the token-indicator derivative? | NEW (relaxed objective, and why it is not E^T g) |
| B8 | Backup: why does RoBERTa beat SEDD? | NEW (size/native query, the ordering is by conditioning, bridge 99.95%, uniform control 0.5%) |

The old positive-control slide ("The missing direction was the training objective, not the
sampler") is retired: its claim is superseded by the ladder.

### Minute budget, re-verified for 20 minutes

title 0.5; revision problem 1.5; energy promise 1.5; RQs 1; near-uniform proposal 1.5;
two samplers 1.5; the null 2; why 1 (linearization) 1.5; why 2 (MH) 1.5; energy still
works 1; GFlowNet 1; last-token 1; the headline 2; the ladder 1.5; takeaways 1;
thank you 0. Total 20.0 min, 9 backups held in reserve.

### Gates
latexmk exit 0; 25 pages (16 content + 9 backup); 0 undefined references or citations;
max overfull hbox 38.01pt, below the 40pt bar (it is the pre-existing TikZ annotation on
the sampler-loop slide, unchanged since Phase 9). Rendered proofs in Doc/final/proofs/:
beamer_slide_5 (near-uniform), 13 (headline), 14 (ladder), 15 (takeaways), 22, 23, 24
(the three new backups); all visually inspected.

### WHAT CHANGED (beamer)
The deck now tells the elimination story rather than the training-objective story: the
near-uniform proposal bounds the null, the token-indicator derivative is the flagship
constructive result, the conditioning ladder carries the attribution, and the takeaway is
the corrected practical implication. One backup was retired for asserting a withdrawn
finding and three were added for the questions the new results invite. No thesis number
was changed by this part.

---

## 2026-07-27  PHASE 10b: deck re-audited against the FINAL thesis (post-evaluation5)

The Phase 10 rebuild logged above predates the evaluation5 round, which reformulated RQ1,
split RQ3, moved constrained generation to an extension, renamed the central derivative,
reordered Chapter 5, replaced "theoretically correct", dropped "pre-registered", and softened
the conditioning claim from an isolation to an ordering. The deck was re-audited claim by
claim against the thesis as it now stands. Eight edits; no slide added or removed.

### Edits

| slide | change | driver |
|---|---|---|
| 6 Two samplers | "pre-registered margin" -> "equivalence margin fixed in advance" | evaluation5 3D |
| 15 Takeaways | same, "certified over 1000 paired sequences at a margin fixed in advance" | evaluation5 3D |
| **7 The central result** | the block now leads with the **$n = 1000$ certification** as a three-row table (last iterate $+0.133$ $[-0.063,+0.326]$, chain mean $+0.136$ $[-0.012,+0.287]$, chain minimum $+0.033$ $[-0.079,+0.145]$) with the line "all three inside the $\pm 0.327$ margin fixed in advance: equivalence certified, not merely non-significant". It previously showed only the $n = 200$ non-significance ($+0.171$, CI straddling zero, $p = 0.40$, detectable only if $\geq 0.652$), which is the weaker of the two results and no longer what the thesis leads with. Closing line "no reliable advantage over noise of the same norm" -> "the gradient direction is equivalent to noise of the same norm" | thesis 5.3 |
| **10** | retitled "The energy is fine; its gradient is not" -> **"The target is not what fails the recovery"**; closing line "Evaluate the energy: works. Differentiate it: does not." -> "Hypothesis A rejected: the energy is searchable, just not by this derivative", with the scope stated underneath: as an objective to \emph{maximize} in free generation the same energy is degenerate, pointing at the likelihood-trap backup | evaluation5 3A |
| 9 Why part 2 | "Faithful correction disables CLS" -> "The exact correction disables CLS", plus "Exact for the proposal it corrects. What fails is the regularity underneath." Bullets tightened so the caveat fits the frame | evaluation5 5/RQ2 |
| 13 The headline | "Relaxed token-indicator derivative. One forward pass" -> "the self term \emph{added to} the same future term, not in place of it", so the slide cannot be read as saying the future term was swapped away | evaluation5 item 2 |
| 14 The ladder | closing line gains "An ordering, not an isolation (backup)" | evaluation5 3C |
| 24 Backup, RoBERTa vs SEDD | the ordering caveat stated in full where the question is actually asked: an ordering by conditioning, not an isolation of it, since the arms differ in scale, corpus and architecture and only the tokenizer and the chain are held fixed | evaluation5 3C |
| 20 Backup, MuCoLa/COLD | "Where they report success without the exact correction, it comes from an early-stopped biased optimizer plus post-hoc filtering" -> the same statement scoped to the mechanism \emph{as reimplemented and measured here}, with the explicit disclaimer that it is not a reinterpretation of those systems' published results, whose tasks, tuning and evaluation this study did not reproduce | evaluation4 item 7 |
| section divider | "Amortization and positive control" -> "Amortization, and which proposals work"; "positive control" now appears nowhere in the deck | evaluation4 item 1 |

### Orphan sweep of the deck

| term | occurrences | status |
|---|---|---|
| quenching | 0 | clean |
| "positive control" | 0 | clean |
| "training objective" as the cause | 0 | clean |
| "operative variable" | 0 | clean |
| "theoretically correct" | 0 | clean |
| "pre-registered" | 0 | clean |
| "Evaluate the energy; do not differentiate it" | 0 | clean |
| anti-guidance | 1, the backup that retires it | correct |
| "one-hot" | 1, the backup explaining why the name is avoided | correct |

Every number in the deck was checked against the thesis: 0.0, 0.5, 2, 18.5, 23.5, 33, 39.0,
40.0, 41.0, 44.5, 55, 99.95 percent, and the KL and log-ratio figures. All agree.

### Minute budget, unchanged at 20.0 minutes

title 0.5; revision problem 1.5; energy promise 1.5; RQs 1; near-uniform proposal 1.5; two
samplers 1.5; the null 2; why 1 (linearization) 1.5; why 2 (MH) 1.5; the target is not at
fault 1; GFlowNet 1; last-token 1; the headline 2; the ladder 1.5; takeaways 1; thank you 0.
Nine backups in reserve. Slide 7 carries one more table row than before and slide 10 one more
line, neither of which changes the spoken time.

### Gates
latexmk exit 0; 25 pages (16 content, 9 backup); 0 undefined references or citations; 0
missing figures; max overfull hbox 38.01pt, below the 40pt bar and unchanged since Phase 9
(the TikZ annotation on the sampler-loop slide); 2 residual overfull vboxes of 4.3pt and
6.0pt, both invisible at slide scale and both pre-existing. Embedded result figures are the
current ones from Doc/figures, including the redrawn versions from the print-legibility pass.

Proofs rendered and inspected in Doc/final/proofs/: bs2_7 (the null, now $n=1000$),
bs3_9 (MH), bdone_5 (near-uniform), bfin_13 (headline), bfin_14 (ladder), bdone_15
(takeaways), bdone_22 and bdone_23 and bfin_24 (the three new backups), and
Presentation_final.pdf.

### WHAT CHANGED (beamer, round 2)
The deck's claims now match the thesis exactly. The two substantive changes are that the null
slide leads with certified equivalence at $n = 1000$ rather than with non-significance at
$n = 200$, and that the "energy is fine" slide is scoped to the recovery task so it no longer
contradicts the likelihood trap. The rest is terminology brought into line: no
"pre-registered", no "theoretically correct", no "positive control", the token-indicator
derivative described as a sum rather than a substitution, and the MuCoLa/COLD verdict narrowed
to the mechanism this study reimplemented. No thesis number was changed by this part.

---

## 2026-07-27  Talk documents added: TALK_TRANSCRIPT.md and TALK_QNA.md

Two markdown companions written into `Doc/final/beamer/`, at the author's request.

### `TALK_TRANSCRIPT.md` (461 lines)

A speakable script for all 15 content slides, keyed to the deck's frame order. Each slide has a
**Say** block written to be read more or less as written, short sentences on purpose, plus
**Point at** for what to gesture to, **If asked here** for the question that usually arrives
mid-slide with its one-line answer, and **Cut first** for what to drop when running long. Ends
with a timing card summing to 20.0 minutes, a running-long and running-short plan, four sentences
to have word-perfect, and three phrasings to avoid.

The one standing rule stated at the top: never say "the gradient does not work", always "the
input-embedding gradient", because the whole contribution lives in that qualifier and the
unqualified form is refuted by the thesis's own data.

### `TALK_QNA.md` (535 lines)

Derived from `thesis_questions_knowledge_base.md` in the repository root, and it follows that
document's own ranking: its section 1 "confusion hotspots" become Part 1 here, in the same order,
because that list is the ranked record of what actually lost a careful reader.

- **Part 1**, the seven hotspots: self-term blindness; "the gradient is noise yet something
  works"; the slope-versus-zero-gradient paradox; the author's own GFlowNet-framing idea, answered
  as a considered alternative rather than a misunderstanding; whether a smaller step size fixes
  the continuous sampler; the statistics paragraph; scale versus anisotropy.
- **Part 2**, by chapter, following the knowledge base's per-section entries, including the
  `PUSHBACK` items it records: the MH formula with a good and a bad move, the Taylor surrogate,
  the KL-is-built-on-likelihood circularity, positional dependence, the corpus split, the missing
  external-judge configuration, the "left context used for two opposite outcomes" catch, the
  Figure 1 overlap, and the generator attribution for the degenerate strings.
- **Part 3**, ten questions the *new* results invite that the knowledge base predates: why the
  grid is not built on the working derivative, why this is not the ordinary one-hot gradient, why
  RoBERTa beats SEDD, whether 44.5 percent is leakage, why the temperature flattened everything,
  the pre-registration wording, the target-not-at-fault versus likelihood-trap tension, the Llama
  contrast, what to do next, and the one-sentence contribution.
- **Part 4**, five questions to hold a line on, with the line to hold.
- A closing coverage table mapping every knowledge-base hotspot and cross-cutting thread to the
  answer that covers it, and naming the two things deliberately not carried over.

**Supersession handling.** The knowledge base was written against an earlier draft, so several of
its entries ask about findings the thesis no longer makes. A table at the top of the document
records the three that changed, quenching, anti-guidance and the objective attribution, and every
affected answer is tagged `SUPERSEDED` and answered with the current finding rather than the old
one. This is deliberate: those questions will still be asked by anyone who read the earlier draft,
and the right answer is the withdrawal plus its evidence.

### Verification

Every headline number in both documents was checked programmatically against the result files:
22 checks over `rev_klbase_gpt2sft.json`, `rev_mlm_control.json`, `rev_mlm_uniform.json`, the
flagship and no-correction grid JSONs, `diag_anisotropy_gpt2sft.json`,
`diag_linearization_gpt2sft.json`, `ohsweep_e10p5_t1p0.json` and `xm_onehot_llama_sharp.json`.
Zero failures. Slide and backup numbers cross-checked against the deck's frame inventory. No
em-dashes in either file.
