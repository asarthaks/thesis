# Prompt for Claude Code, Phase 11: future-work audit (gate), then defense experiments

Opus 4.8, highest reasoning. This file has TWO parts separated by a HARD GATE.
Do Part 1 and STOP. Do not begin Part 2 until the author replies "approved,
run Part 2". Paste the block below.

---

Read REVISION_RESTRUCTURING.md, the evaluation3-pass and Phase 10 entries in
REVISION_LOG.md, REVISION_WRITING.md, and the current thesis in
Doc/final/thesis/. Standing rules: Ioanna register, no em-dashes, template
untouched, removed text to % comments, nothing moved or deleted.

# PART 1: audit and proposal (NO GPU, NO THESIS EDITS, ends in a report)

Produce a report in REVISION_WRITING.md, and nothing else. No .tex file is
modified in Part 1.

1. STATE OF PLAY. Current page count (total and body), current structure, and
   the exact current wording of the future-work material wherever it lives
   (6.5 or 7 or both). Quote it.

2. DRAFT THE REPLACEMENT, in the report, not in the thesis. A future-work
   section built around three concrete next experiments, in this order, each
   with one paragraph saying what it measures and why it is the next necessary
   step rather than a wish:
   a. Revision scale: recovery and contextual fit as the number of masked
      positions grows (1, 2, 4, 8, 16; contiguous and scattered), which is
      what the introduction's motivating scenario actually requires.
   b. Transfer to a constrained energy: the one-hot identity is specific to
      the likelihood term, since the v-th coordinate is log p(v | x_<i) plus
      the embedding inner product; an arbitrary classifier has no such closed
      form, so the natural architecture is proposal from the output-side
      surrogate, top-k candidates rescored by the classifier on full
      sequences, and MH accepting on the exact combined energy. State it as
      the experiment that would test whether the diagnosis generalizes beyond
      the likelihood.
   c. The sampling claim: diversity at matched control, measured against a
      compute-matched best-of-N optimizer, which is the property that
      motivated insisting on the correction in the first place and which no
      metric in this thesis yet measures.
   Also draft the honest status sentence for 6.4: the failure is diagnosed and
   the repair demonstrated on the diagnostic task; the bridge to controllable
   generation is specified but not crossed.

3. THE EFFICIENCY POINT (verify in code before writing anything). Read
   core/dls.py and core/base_sampler.py and establish precisely what
   `policy_onehot` computes: whether the proposal uses the self term ALONE or
   the self term ADDED to the embedding-gradient term, and therefore whether
   the current implementation still requires a backward pass. Report the
   finding. If the self term is available from the forward logits already
   computed for the energy, then the working surrogate is backward-pass-free
   in principle while COLD and MuCoLa backpropagate at every step, and that is
   an efficiency result as well as a correctness result. Draft the one or two
   sentences for 5.6.1 and the discussion, and say plainly whether the current
   code realizes that saving or merely could.

4. PAGE BUDGET AND TRADES. Estimate the page cost of items 2 and 3. The hard
   limit from Phase 10 stands: 115 to 119 pages total, body keeps its figures
   and tables, no layout tricks. If the additions would breach it, list
   candidate trades with a one-line cost each (what is lost, and which
   evaluator praised it if any), ranked, and recommend one. Do not execute any
   trade in Part 1.

5. CONSISTENCY CHECK. List anything else the future-work rewrite forces:
   the conclusion's outlook paragraph, the abstract if it mentions future
   work, the RQ answers in 6.1 if any promise this material, and the beamer's
   future-work slide.

6. STOP. End the report with a numbered list of decisions you need from the
   author, and wait.

# HARD GATE. Do not proceed past this line without the author's approval.

# PART 2: the defense experiments (only after approval)

Purpose, and it defines the boundary: these results are for the DEFENCE, so
the author can answer "what next" with measurements rather than intentions.
Default destination is the beamer backup slides plus, at most, one
"preliminary work indicates" clause in future work. Results do NOT enter
Chapter 5 and do not reopen the finalized argument unless the author
explicitly says so.

## Ground rules

- No training of any model. Frozen models, existing checkpoints, existing
  harness only. No new corpora.
- Import `build_corruption`, `load_texts`, `seed_all` from the grid scripts;
  never reimplement them (a reimplementation once consumed the RNG
  differently and had to be discarded).
- Strict role separation for any classifier: the model that steers never
  scores. The scoring classifier is the concern-11 judge, unchanged.
- Pre-register a prediction for every run before launching it, in the log.
- Use the sharding added in D3 and `revision/merge_shards.py`; a fresh status
  dir; `--per_gpu 1` or 2 on the 3090 host (36 concurrent GPT-2 Large loads
  exhausted host RAM and the OOM killer took them without a traceback); never
  blanket `reset_incomplete.sh` while other hosts hold live locks, clear only
  locks whose log has been silent past a threshold.
- If any result contradicts a claim the finalized thesis makes, STOP, write
  the contradiction plainly at the top of the status block, and ask. Do not
  continue the slate and do not quietly reconcile it.

## Status reporting (the author wants to check in at any moment)

Create DEFENSE_EXPERIMENTS_LOG.md. Its first section is a STATUS block that
you OVERWRITE after every job family completes, containing: current family,
jobs done/running/queued, wall-clock elapsed, the headline number so far in
one line, the next decision point, and any red flag. Below it, append-only
detail: pre-registration, launch commands, results tables, and interpretation.
Update the STATUS block at least every completed family, and immediately on
any failure or contradiction.

## E0 (run first, cheapest, highest value): is the repair backward-pass-free?

From Part 1 item 3 you know whether `policy_onehot` is self-only or
self-plus-gradient. If it is the latter, add a `policy_self` variant that uses
the self term ALONE (log p(v | x_<i) read from the forward logits, no backward
pass), guarded so all existing behaviour is bit-identical with the flag off,
and verify that with the equivalence suite before running anything.

Run at the sharp configuration that works (large eps so the distance term is
suppressed, T = 1.0), on GPT-2 Large, n = 200, MH on, and at the calibrated
cell as a control. Report exact %, ever %, KL, acceptance, t2/t1, and measured
wall-clock per sample against `policy_onehot` and against the
input-embedding arm.

Prediction to register first: pure self performs at least as well as
self-plus-gradient (the re-analysis showed the embedding term slightly
degrades the correlation), at lower cost per step. If it does, the thesis's
repair is also strictly cheaper than the methods it corrects, and that is a
defence slide on its own.

REGISTER THE ALTERNATIVE TOO. The autoregressive left-conditional independence
sampler reaches 23.5 percent on this task while the one-hot sampler reaches
40, so pure self is not self-evidently equivalent to the left-conditional arm;
the discrete kernel also carries the distance term and the temperature. If
pure self lands nearer 23.5 than 40, the embedding term contributes after all,
the efficiency claim weakens, and that must be reported as the finding rather
than reframed. Report the three arms side by side (pure self, self plus
gradient, left-conditional independence) so the comparison is explicit.

## E1: revision scale

Masks M in {1, 2, 4, 8, 16}, contiguous and scattered, same sequences and
corruption machinery, n = 200 (shard it). Arms: the best output-side proposal
from E0, RoBERTa-large (the strongest proposal you have), uniform (the floor),
and the input-embedding gradient (the control that scored zero). Metrics:
exact %, ever %, the existing KL, acceptance. State the multi-position update
scheme explicitly in the log (random-scan single-position updates or joint),
justify the choice, and keep it fixed across arms.

Prediction: degradation with M for every arm; the question is the shape and
whether the ordering of proposals is preserved. Report the curve, not a
verdict.

## E2: transfer to a constrained energy (the flagship of this slate)

Energy: the exact combined energy, likelihood plus lambda times the sentiment
constraint, with the MH accept computed on it exactly. Arms, all sharing the
same sequences, the same lambda grid (a small grid, 3 values), and both target
labels:
  a. output-side proposal, top-k candidates rescored by the STEERING
     classifier on full sequences, MH on the exact combined energy;
  b. output-side proposal, constraint in the ACCEPT step only (no rescoring),
     which isolates what rescoring buys;
  c. input-embedding constraint gradient inside the proposal, the approach
     that failed, as the control;
  d. uniform proposal with the same rescoring, which isolates what the
     surrogate contributes over a blind proposal;
  e. mixture proposal: half the shortlist drawn by likelihood rank, half drawn
     by the constraint's own preference over the vocabulary where one is
     available, which tests whether shortlist coverage rather than the
     mechanism is the binding limit;
  f. THE REPAIRED LANGEVIN STEP, and the arm this slate exists for: one
     proposal formed from both terms of the combined energy, each computed
     with the best signal available for it. The likelihood contribution is the
     EXACT output-side self term log p(v | x_<i), free in the forward pass;
     the constraint contribution is the classifier's Taylor surrogate, its
     gradient with respect to the input embedding at the masked position
     dotted with e(v) - e(x_i), scaled by lambda, because an arbitrary
     classifier has no closed form to substitute. Distance term, temperature
     and exact-combined-energy accept exactly as in the other arms. This is
     structurally the method the field uses (COLD, MuCoLa) with ONE
     substitution: the input-embedding likelihood gradient replaced by the
     exact output-side term.
Metrics: control adherence under the HELD-OUT judge, fluency under gpt2sft,
both labels, bootstrap CIs, plus the guide-versus-judge agreement number you
already know to report.

SHORTLIST COVERAGE DIAGNOSTIC, run alongside and reported with every arm.
The likelihood shortlist is the design's real vulnerability: a token that
satisfies the constraint but sits outside the top k is unreachable, and the
arm would then fail for a reason unrelated to the mechanism under test.
Measure, per step: the rank under the likelihood of the candidate the
constraint most prefers, the fraction of steps where the constraint-preferred
candidate falls inside the shortlist, and the same at k in {16, 64, 256} so
the curve is visible. Report this BEFORE interpreting any adherence number.
If coverage is low, the honest reading of a weak arm (a) is a proposal-support
limitation, not evidence against exact-evaluation steering, and the write-up
must say so.

Prediction: (a) and (f) lead, then (b), then (d), then (c), at matched
fluency, with (e) above (a) only if coverage is the binding constraint. Any
other ordering is interesting and must be reported as measured. Register also
the compute accounting: forward and backward passes per accepted move for each
arm, including (c), since the comparison against gradient-based steering is an
efficiency claim as well as an adherence claim. Note that (f) costs one
classifier BACKWARD per step while (a) costs one batched classifier FORWARD
over k candidates, so which is cheaper depends on k and must be measured
rather than asserted.

WHAT THE (a) VERSUS (f) CONTRAST DECIDES, and it is the point of the slate.
Both arms hold the likelihood side fixed at the exact output-side term and
differ only in how the constraint enters: exact evaluation on a shortlist
against a first-order surrogate of the classifier. If (f) tracks (a), the
thesis's diagnosis is specific to the likelihood parameterization and gradients
remain usable for arbitrary differentiable constraints, which is a materially
narrower and more careful claim than the thesis currently makes and must be
reported as such. If (f) collapses toward (c), the linearization failure
generalizes beyond the likelihood, and exact evaluation on a shortlist is the
general remedy. Report the constraint-side linearization diagnostic alongside
it, the Spearman correlation between the classifier's Taylor surrogate and the
true change in C over the same candidate sets, so the outcome has a mechanism
behind it rather than only a ranking. Note for the write-up: the existing
constrained results already showed the constraint gradient's DIRECTION
carries signal (the paired cons_only minus cons_random contrast), so (f)
working is a live possibility, not a straw arm.

## E3: the sampling claim

N generations per prompt from the best chain of E2 at a fixed control level,
against a compute-matched best-of-N optimizer at the same control level.
Metrics: distinct-n, self-BLEU, and an embedding-based semantic spread, all at
MATCHED adherence, with the compute accounting stated in forward passes.

Prediction: register one. Note in the log that this is the metric the proposal
promised as "posterior coverage" and never operationalized, so whichever way it
falls it closes that gap.

## Adaptive continuation (the author asked for this explicitly)

After each family completes, write an interim entry containing: prediction
versus outcome, what the result rules in or out, and a PROPOSED next batch of
at most 12 jobs with a one-line rationale each and an estimated wall-clock.
Then apply this rule:
  - If the proposed batch is a variation within the current family (another
    lambda, another mask count, another temperature cell, a seed repeat for
    power) and fits the budget, LAUNCH IT and log that you did.
  - If it introduces a new model, a new corpus, any training, or would exceed
    the budget, ASK FIRST.
Budget for the whole of Part 2 without asking again: 300 GPU-hours aggregate
and no single family beyond 24 hours wall-clock. Track and report consumption
in the STATUS block.

## Closing deliverable for Part 2

- DEFENSE_EXPERIMENTS_LOG.md complete, with a final summary table: every
  experiment, its prediction, its outcome, and the one sentence the author
  should say at the defence if asked about it.
- Beamer backup slides for E0 to E3, built in the existing deck's style, each
  a single visual plus three lines, logged in PRESENTATION_LOG.md.
- A one-paragraph draft, for the author's decision only, of the "preliminary
  work indicates" clause for future work, with an explicit note on whether any
  result would require a change to the finalized thesis text (expected: none;
  if otherwise, say so loudly).

Constraints: no training; no thesis edits in Part 2 without a separate
instruction; stop and ask on any contradiction with a finalized claim; every
number traceable to a JSON; no em-dashes.
