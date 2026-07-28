
---

## 2026-07-28  AUTHOR ROUND 5: deck restructured to the thesis's own argument order

The author raised eight points against the Phase 10b deck. All eight are applied. The deck grows
from 16 content slides to 20 and from 9 backups to 10; no slide was dropped.

### The eight points and what was done

| # | author's point | resolution |
|---|---|---|
| 1 | introduce Langevin dynamics first, after slide 3 or inside it | NEW slide 4, "Langevin dynamics, and how it is made discrete". The continuous update with drift and diffusion named, the Lipschitz condition flagged (it is consumed twice later), then the discrete adaptation |
| 2 | do not go straight into the comparison with the uniform distribution | the uniform-draw control (0.5 percent, KL 6.538 vs 6.541) is REMOVED from the near-uniform slide and moved to the ladder, slide 16, where the rungs above it make it mean something. The near-uniform slide now carries only the entropy measurement |
| 3 | the slide never says why "a perfectly informative gradient would have produced the same null" | the reasoning is now ON the slide, in its own alert block: two proposals both within $10^{-4}$ nats of uniform are two draws from the same distribution, therefore the main grid bounds what it could detect rather than establishing the null. It was previously only in the transcript |
| 4 | is the loop really "run till convergence"? | NO, and the claim was wrong. Corrected to "a fixed annealed schedule of 50 or 100 steps, never stopped early". Thesis 4.3 sets 50 or 100 steps; thesis 5.1.1 shows that without the correction the chain does not converge at all (mean KL rises 8.765 to 9.499). A new backup, 23, carries that measurement, and the transcript gains an "If asked here" and a fourth entry in the traps list |
| 5 | slide 6 before slide 5, then 5 before 7 | applied. Samplers, then setup, then the near-uniform measurement, then the null |
| 6 | list the five energies before the experiments | NEW slide 8, "The setup: one task, five energies, one metric". Task, the five energies named individually with their roles, the KL metric, the margin fixed in advance |
| 7 | slide 8 never says what linearization IS | the linearization slide now defines it in the first line and shows both quantities being correlated, the Taylor surrogate against the true log-likelihood change, before giving the correlation |
| 8 | put the gradient equation at the start so the output surrogate has context | the DLS proposal logit is on the new slide 4 with both terms underbraced, and the closing line names $\bm{g}$ as the input-embedding gradient and the object under test. Slide 12 and slide 14 now both refer back to it |

### Results order, matched to Chapter 5 and to the author's stated preference

Ablation through to the input-embedding surrogate failing, then the output surrogate and the
recovery, then the ladder, then classifier-guided steering, with the hypotheses stated up front:

| slide | content | thesis section |
|---|---|---|
| 6 | three candidate explanations A, B, C and the test that decides each | 1.2, 5 opener |
| 9 | the calibrated proposal is numerically uniform | 5.1.1 |
| 10 | the null, certified equivalence, plus the sweep result | 5.3 |
| 11 | Hypothesis A: gradient-free baselines on the identical energy | 5.3.4 |
| 12 | why 1, the linearization | 5.4 |
| 13 | why 2, the MH breakdown | 5.2 |
| 14 | Hypothesis B: the token-indicator derivative | 5.4.1 |
| 15 | the final-position case | 5.5 |
| 16 | Hypothesis C: the proposal ladder | 5.6 |
| 17 | GFlowNet | 5.9 |
| 18 | the extension: constrained generation and classifier-guided steering | 5.10 |

Slide 6 is new and slide 18 is new. Slide 18 restores the material that had no slide at all in
the Phase 10 deck: the plug-and-play test the whole programme was built for, reported through the
paired constraint-direction contrast (MuCoLa continuation $+27.3$ and $+36.7$, discrete sampler
$0.0$ and $+0.7$) with the off-manifold caveat, and the classifier-guided steering result on the
navigable carrier ($+9.3$ to $+25.7$ points on the guide's own verdict, agreement ladder 56 to 80
percent, the trust-region fluency result against $7.1 \to 11.0$ nats without it).

### Minute budget

Baseline 23.5 minutes over 20 content slides. A marked 20-minute cut set is in the transcript's
timing card and brings it to 20.15 without touching slides 6, 9, 14 or 16. The author should pick
the target: 20 minutes needs the cut set, 25 does not.

### Companions updated

`TALK_TRANSCRIPT.md` rewritten to the new 20-slide order, with new Say blocks for slides 4, 6, 8
and 18, the uniform-control moved to slide 16, a note on slide 9 recording where the control went,
a two-column timing card (baseline and cut), and a fourth trap: never say the sampler runs to
convergence. `TALK_QNA.md` backup page references renumbered for the ten-backup appendix
(17-25 to 21,22,24,25,26,27,28,29,30), the two "slide 9" references retargeted to slide 13, and
the quenching answer pointed at the new backup 23.

### Gates
latexmk exit 0; 30 pages (20 content, 10 backup); 0 undefined references or citations; 0 missing
figures; max overfull hbox 38.01pt, the pre-existing TikZ annotation on the sampler-loop slide,
unchanged since Phase 9; max overfull vbox 6.00pt, also pre-existing. Orphan sweep clean:
0 occurrences of quenching, "positive control", "training objective", "theoretically correct",
"pre-registered", "run to convergence", or em-dashes in the deck or either companion. Slides 4, 6,
8, 9, 12, 16, 18 and 19 rendered and visually inspected.

### WHAT CHANGED (beamer, round 3)
The deck now opens with the machinery (Langevin, the discrete proposal, the gradient in a formula)
and the three hypotheses, then runs the results as the elimination the thesis runs, and closes on
controllable generation rather than stopping at the ladder. The two substantive corrections are
the withdrawn "run to convergence" claim and the near-uniform slide, which now states on the slide
why it bounds rather than establishes the null. No thesis number was changed by this round.

---

## 2026-07-28 AUTHOR ROUND 6: two mechanism slides, the running example

Author request: two new slides built on one running example, "The dog barked loudly because",
tracking the embedding at position 3. One early slide showing how an autoregressive model scores
a fixed sequence, one later slide showing why the derivative reaches only half of that score and
how the token-indicator derivative repairs it.

### What was added

**New slide 4, "Sequence scoring: the dual role of a token."** Placed between the energy promise
(3) and Langevin (5), so the order is now define the energy, show how it is computed, then show
how it is differentiated. Content: a left-to-right TikZ flow of four embeddings into a causal
transformer emitting logit vectors, plus a zoom on the logit vector at position 2 drawn as a
column of cells with the integer id 452 pointing into the highlighted one. Three facts on the
slide: the energy is the sum of the log-probabilities of the tokens that are there; the future
route, where e_3 shapes the logits at step 3; the self route, where the score of "barked" sits in
the logits at step 2 and is fetched by an integer index.

Deliberately NOT an argument. The closing line is "Both routes are in the energy. Only one of
them is a continuous function of e_3." The finding is not stated here. Tipping it early would
spend the effect twice, once without evidence.

**New slide 15, "The gradient fallacy, and the token-indicator repair."** Placed immediately
before the Hypothesis B results, which is now slide 16. Left half: the gradient at e_3 splitting
into two paths, a teal check on the path to the score of "loudly" from Z_3 and a red cross on the
dashed path to the score of "barked" from Z_2, with the two reasons stated (masked out of Z_2,
and the score is an array index, so the derivative is exactly zero). Right half: the repair as
two stacked boxes, the future term the proposal already had plus the self term read straight off
the logits. Closing line ties it to the rho = 0.03 already shown on the linearization slide.

### Consequential edits

- Slide 16 (Hypothesis B) no longer re-derives the dual role, since slide 15 now does it in full.
  Its left column opens "Written out, with the token indicator z_i relaxed in both of its roles"
  and the slide is now the numbers rather than the argument. Its budget drops 2.0 to 1.5 min.
- `\usepackage{pifont}` added for the check and cross glyphs.
- Every slide number from 4 onward shifts. Mapping applied to both companions: 1 to 3 unchanged,
  4 to 13 shift by +1, 14 to 30 shift by +2. Backup pages 21 to 30 become 23 to 32.
- TALK_QNA.md: all nine Backup pointers renumbered, both "slide 13" references retargeted to 14.
- TALK_TRANSCRIPT.md: two new Say sections, the timing card rebuilt, the header updated to 32
  slides, "by minute seven" to "by minute eight" for the hypotheses slide, and the traps heading
  corrected from three to four (it has listed four since round 5).

### Timing

Baseline 23.5 to 25.25 min. Slide 4 costs 1.0 and slide 15 costs 1.25; slide 16 gives back 0.5.
The short set now lands at 20.65 rather than 20.15. A hard 20-minute cap needs slide 17 dropped
entirely and slide 19 folded into one sentence off slide 18, which lands at 19.9. The never-
compress list is now 7, 10, 15, 16, 18.

### Gates

latexmk exit 0, 32 pages. No new overfull boxes: the eight reported are all pre-existing (18.15
and 9.26 at the energy-promise frame, 38.01 on the sampler-loop TikZ annotation, 4.98 on the
Hypothesis B equation, and four sub-5pt vboxes). Zero undefined references. Slides 4 and 15
rendered to PNG and inspected; the only fix needed was vertical room around the plus sign between
the two repair boxes. No thesis number was changed by this round.

---

## 2026-07-28 AUTHOR ROUND 7: the scoring slide rebuilt on the real code, and the order fixed

Three author corrections to round 6, applied together.

### 1. The scoring slide now shows the target_ids array and the offset

Round 6 drew only the forward pass and a zoom on one logit vector. The author asked for the
target array alongside it, and for the length alignment as it is actually done in the code. Read
from `core/prep.py`, `joint_log_prob_from_inputs_embeds`:

    logits  = outputs.logits[:, :-1, :].contiguous()
    targets = target_ids[:, 1:].contiguous()

So the slide is now three rows: `inputs_embeds` at the bottom feeding the transformer, `logits`
above it, `target_ids` above that, with diagonal arrows pairing logit t to id t+1. The dropped
last logit and the dropped first id are greyed out, which makes the offset the visual point of the
slide. The two slice lines appear underneath, then the energy as a gather:

    E(x) = - sum_t log softmax(Z_t)[ id_{t+1} ]

Title changed to "Scoring a sequence: two arrays, offset by one", which is also the phrase the
callback slide reuses.

### 2. No position is singled out on the scoring slide

The author will return to this slide when the output surrogate is introduced, so it must not
spend the argument early. Every mention of a specific position, of "the dual role", and of the
future-versus-self split is gone from it. The closing line is now only "The same sequence enters
twice: once as continuous vectors, once as integers." The transcript section carries the same
instruction in bold.

Slide 15 was rebuilt as the literal callback. It is now titled "Back to the two arrays: what the
gradient can reach" and its left half is a compressed redraw of the same three rows, with a teal
check on `inputs_embeds` to `logits` and a red cross on the path to `target_ids`. The words
"barked" and "loudly" moved out of the slide and into the Say block. This is less text on both
slides and a stronger return.

### 3. Langevin before scoring, and the DLS proposal off the Langevin slide

Order is now 3 energy promise, 4 Langevin, 5 scoring, 6 RQs, 7 hypotheses, 8 samplers. The
Langevin slide is a quick intro only: the continuous update, drift, diffusion, the Lipschitz
requirement, a small walk-on-a-smooth-curve figure, and the bridge that text has no small step.

The DLS proposal logit, which is sampler-specific, moved to slide 8 where the samplers are
introduced, together with the definition of g and the "object under test" line. Slide 8 is the
right home for it: by then the audience has seen Langevin and has seen how the energy is computed.

### Timing

Baseline 25.5 to 25.25 (Langevin 1.75 to 1.0, samplers 1.25 to 1.75, slide 15 1.25 to 1.5).
Short set 20.25. A hard 20-minute cap now needs only slide 17 dropped, which lands at 19.75.
The hypotheses slide is back at minute seven in the baseline, and the transcript says so again.

### Gates

latexmk exit 0, 32 pages, zero undefined references. The deck's worst overfull box is now
18.15pt, down from 44.86: moving the "the step the field skips" annotation below the MH node
fixed the long-standing 38pt overflow on the sampler slide as a side effect. Slides 4, 5, 8 and
15 rendered and inspected. All cross-references in both companions renumbered for the swap:
the alignment-term callback moved to slide 8, the Lipschitz callback to slide 4, the linearization
callback to slide 8, and the Q&A H1 gesture pointer to slide 5.

---

## 2026-07-28 AUTHOR ROUND 8: setup placement, the mask on slide 5, slide 15 redrawn, text cut

### The placement question, and the answer

The author asked where the task setup belongs, given that the research questions already use the
term "input-embedding gradient". Answer: it belongs after the hypotheses and BEFORE the samplers.
Order is now 3 energy promise, 4 Langevin, 5 scoring, 6 RQs, 7 hypotheses, 8 setup, 9 samplers,
10 calibrated proposal, 11 central result, 12 Hypothesis A.

Three reasons, in order of weight:

1. Slide 5 already grounds the term. Once the audience has seen the `inputs_embeds` row, "the
   input-embedding gradient" means "the derivative with respect to that row" and the RQs can use
   it. What slide 9 adds is not the meaning of g but how a sampler consumes it, which is
   implementation, not vocabulary. So slide 5 stays before the RQs.
2. The old order broke an adjacency. Slide 10 is about the very proposal that the sampler slide
   introduces, and the setup was sitting between them. Moving the setup up puts the proposal and
   the measurement of that proposal back-to-back. This was the cohesiveness problem.
3. It matches the thesis. Chapter 4 runs Task, Energy Functions, Samplers and Configurations,
   Evaluation Metrics. Task and energies precede the samplers there too.

### Slide 5: the mask and the soft token

Position 3 now carries s over `[MASK]` in red on the `inputs_embeds` row, and `id(s)` in red on
the `target_ids` row, so the slide states that the masked position's embedding is the soft token
being optimized and its id is whatever s projects to. Everything else is unchanged.

### Slide 15 redrawn on slide 5's picture

The abstract three-box version is gone. It is now the same diagram as slide 5, four positions,
with three marks: a teal arrow and check on s feeding the transformer and reaching Z_3 onward; a
red dashed barrier with a cross between positions 2 and 3, labelled "causal mask: nothing flows
left"; and a thick red gather arrow from Z_2 to id(s) labelled "the score of position 3 is a
gather at this integer". One sentence underneath, then the two repair boxes side by side.

The barrier was first drawn across the transformer bar, where it collided with the bar's own
label, and the earlier version had a dashed arrow from s to Z_2 that crossed the teal arrow and
read as a contradiction. Both fixed.

### Text reduction, now a standing rule

Content slides carry as little text as the argument allows; backup slides are exempt. Applied to
slides 4, 6, 7, 8, 9, 10, 11, 12, 16, 18, 19, 20, 21. Mean prose per content slide fell from 99
words to 66, and the cut prose moved into the transcript, which is where speech belongs. The
hypotheses slide is now a three-row table (A/B/C, what is at fault, decided by) plus one line.

### New backup 33: how this differs from MuCoLa and COLD

Author request. Four-row table: MuCoLa and COLD differentiate the input embedding and discard the
self term with no correction; CLS and DLS here differentiate the same thing but apply the exact
correction; Grathwohl 2021 and Zhang 2022 differentiate the one-hot and keep the self term; the
token-indicator derivative keeps it too. So the answer to "do they lack it as well" is yes, and
the slide says why that is structural rather than a defect: the gap is a borrowing from the
embedding-space side of the literature rather than the discrete-EBM side whose proposal the DLS
sampler otherwise uses. It also states what would change it, namely relaxing the target as well
so the score is a soft dot product rather than a gather, which is what the token-indicator
derivative does. Sourced from `revision/analyze_onehot_surrogate.py` (the docstring states the
input-embedding attribution for MuCoLa and COLD and the one-hot attribution for Grathwohl and
Zhang), `core/constraint.py` (the Lagrangian-versus-weighted-sum deviation), and thesis 3.1, 2.4
and 6.2.

ONE ITEM TO VERIFY BEFORE THE DEFENCE: the claim that MuCoLa gathers its self term at a discrete
index rests on this repository's own characterization, not on a re-reading of the paper's energy.
If MuCoLa scores a soft target through the tied output matrix, its self term is differentiable
and the first table row needs a footnote. The slide is worded so that the structural argument
survives either way, but check it.

### Gates

latexmk exit 0, 33 pages, zero undefined references. Five overfull boxes, all pre-existing, worst
18.15pt. Slides 5, 7, 15 and 33 rendered and inspected. Transcript sections 8 and 9 swapped, the
slide 15 section rewritten to the new marks, timing card updated, and the Q&A given a pointer to
backup 33 on the MuCoLa deflection row. Baseline 25.25 min, short set 20.25, unchanged by this
round.
