# Prompt for Claude Code, Phase 13: controllable-generation experiments (post-submission, code only)

Opus 4.8, highest reasoning. Paste the block below.

---

The thesis has been SUBMITTED. This session runs experiments only.

## Absolute constraint, verified at the end

NOTHING under Doc/ is read-write. Do not edit, create, delete or reformat any
file in Doc/final/thesis/, Doc/final/proposal/, or Doc/final/beamer/. Reading
them for context is fine. At the end, run a status check on Doc/ and paste the
output proving it is untouched. If any tool call would write under Doc/, stop.
Slide material is SUGGESTED in a separate file for the author to decide on; it
is never inserted into the deck.

Read CLAUDE.md, the evaluation3-pass entries in REVISION_LOG.md (current
findings of record), and DEFENSE_EXPERIMENTS_LOG.md if present. Create
CTG_EXPERIMENTS_LOG.md. No em-dashes.

## What this phase is for

The thesis diagnosed the failure and demonstrated a repair on the diagnostic
task. It never crossed the bridge to controllable generation. This phase
crosses it, or measures exactly why it cannot be crossed, so the author can
answer "is your sampler actually useful" with numbers. Four questions, in
priority order:

Q1 COST. The output-side surrogate has a self term, log p(v | x_<i), free in
   the forward pass, and a future term describing how the substitution changes
   everything downstream, which costs either a backward pass (linearized) or k
   forward passes (exact). Is the future term worth its cost?
Q2 CONDITIONING. Do bidirectional proposals (RoBERTa, SEDD), which see right
   context, beat the autoregressive output-side proposal on a task where the
   controlled attribute is global?
Q3 MECHANISM. How should a constraint enter: exact evaluation on a shortlist,
   a first-order surrogate of the classifier, both, or only the accept step?
Q4 USEFULNESS. At matched compute, does the exact-energy MH chain beat MuCoLa
   on the adherence-fluency trade-off, and does it produce more distinct valid
   solutions, which is the property that motivated the correction in the first
   place?

## Part 0: audit and design (no GPU, ends in a plan the author can skim)

1. Read run_constrained.py, core/dls.py, core/cls.py, core/constraint.py,
   diagnostics/run_revision.py and the sharding/merge utilities. Report the
   task setup already implemented for constrained generation (prompts,
   initialization, sequence length, arms, metrics) and REUSE it; extend arms,
   do not rebuild the task.
2. Establish exactly what policy_onehot computes today: self term alone, or
   self term plus the embedding-gradient term. This decides the Study A arm
   list.
3. MUCOLA-FAITHFUL ARM. The thesis's CLS is a variant of MuCoLa, not MuCoLa:
   MuCoLa's language-model term also carries a direct path to the next
   position's embedding through the negative log-likelihood, since with tied
   embeddings that vector acts as an output embedding, whereas the CLS
   implementation differentiates inputs_embeds against discrete labels and has
   no such path. Implement a mucola_faithful arm that includes the direct
   output-embedding path, keeps the continuous embedding state with projection
   to the nearest embedding, uses a fixed constraint weight, and runs WITHOUT
   MH, as the method does. Verify it against the paper's update before
   running. Keep the existing CLS as its own labelled arm so the difference
   between them is itself a measurement. Report both as distinct arms
   everywhere; never call the CLS "MuCoLa".
4. Write the full job matrix with per-job VRAM, shard counts, and an estimated
   wall-clock, and the exact queue commands. Then start Study A without
   waiting for approval; the budget rule below governs.

## Study A (first): is the future term worth its cost?

Arms, all inside the exact-energy MH chain, sharp configuration:
  A1 self term only, no backward pass at all;
  A2 self term plus linearized future term (one backward pass), i.e. the
     current policy_onehot if that is what it computes;
  A3 self term plus EXACT future term for the shortlisted k candidates
     (k forward passes, the expensive ceiling);
  A4 input-embedding gradient only (the control that scored zero);
  A5 uniform proposal (the floor).
Run on BOTH the recovery task (cheap, and comparable to the thesis numbers)
and the constrained-generation task. Report per arm: exact and ever recovery
where applicable, adherence under the held-out judge, external fluency,
acceptance rate, and the cost triple (forward passes, backward passes,
wall-clock per accepted move).

Pre-register: A1 approximately equals A2 at strictly lower cost, and A3 buys
little over A1 relative to its cost. If A1 collapses relative to A2, the
future term matters and the efficiency story dies; report that plainly.

## Study B: proposals, including bidirectional

Fix the best constraint mechanism from an early Study C probe, then compare
proposal sources: output-side (the Study A winner), RoBERTa-large, SEDD, the
autoregressive left-conditional, uniform, and the input-embedding gradient.
Same sequences, same energy, same accept, same budget. Report the same metric
and cost sets. This tests whether right-context access, which mattered on
recovery, matters more or less when the target attribute is global.

## Study C: how the constraint should enter

Fix the proposal at the Study A winner. Arms:
  C1 constraint in the ACCEPT step only;
  C2 exact classifier rescoring of the top-k shortlist;
  C3 classifier gradient as a first-order term inside the proposal (the
     MuCoLa-style constraint handling, but on a repaired likelihood side);
  C4 C2 and C3 combined;
  C5 mixture shortlist, half by likelihood rank and half by the constraint's
     own preference.
Run the SHORTLIST COVERAGE diagnostic alongside every arm and report it BEFORE
any adherence number: the rank under the likelihood of the candidate the
constraint most prefers, and the fraction of steps where it falls inside the
shortlist, at k in {16, 64, 256}. A weak C2 with poor coverage is a
proposal-support limitation, not evidence about the mechanism, and must be
reported as such. Report also the constraint-side linearization correlation
(Spearman between the classifier's first-order surrogate and the true change
in the classifier score over the same candidates), so C3's outcome has a
mechanism behind it.

## Study D: the head-to-head, at matched compute

Best configuration from A, B and C against mucola_faithful and against the
thesis CLS arm. Sweep the constraint weight on a small grid for every method
so the comparison is a CURVE, adherence under the held-out judge against
external fluency, not a single point. Equalize compute: report the curve at
matched wall-clock and at matched total forward-equivalent passes, stating the
accounting. This is the table the defense question "is your sampler useful"
gets answered from.

## Study E: the sampling claim

At a fixed adherence level, generate N sequences per prompt from the best
chain and from a compute-matched best-of-N optimizer over the same energy.
Measure distinct-n, self-BLEU, and an embedding-based semantic spread, all at
MATCHED adherence and matched compute. Note in the log that this is the
"posterior coverage" the proposal promised and never operationalized, so
whichever way it falls it closes that gap.

## Adaptive continuation

After each study, write prediction versus outcome, what it rules in or out,
and a proposed next batch of at most 12 jobs with one-line rationales and an
estimated wall-clock. Then: if the batch is a variation within the current
programme (another constraint weight, another k, another temperature cell,
another seed for power, an extra proposal source already loaded) and fits the
budget, LAUNCH IT and log that you did. If it needs a new model, a new corpus,
any training, or exceeds the budget, ASK FIRST. Budget without asking: 400
GPU-hours aggregate, no single study beyond 24 hours wall-clock. Track
consumption in the STATUS block.

## Parallelization (the author flags this as very important)

- Shard every study by sequence; one job per arm and shard; merge with the
  existing merge utility. Use a fresh status dir.
- SPLIT GENERATION FROM SCORING, the pattern that already worked: generation
  uses the small models (GPT-2 Large, RoBERTa, SEDD) and shards across all
  cards at 2 per card where VRAM allows; the Llama-3 judge runs afterwards as
  a single batched scoring pass over the accumulated text, never co-resident
  with the generators.
- On the 3090 host use per_gpu 1 or 2; 36 concurrent GPT-2 Large loads
  exhausted host RAM and the OOM killer removed workers without a traceback.
- Never blanket-reset locks while other hosts hold live claims; clear only
  locks whose log has been silent past a threshold.
- Report GPU utilization in the STATUS block; if cards are idle while jobs
  queue, fix the shard granularity rather than accepting it.

## Status reporting

CTG_EXPERIMENTS_LOG.md opens with a STATUS block, OVERWRITTEN after every
completed arm or family: current study, jobs done/running/queued, GPUs busy,
elapsed and budget consumed, the headline number so far in one line, the next
decision point, red flags. Below it, append-only: pre-registrations, commands,
result tables, interpretation.

## Contradiction handling (the thesis is submitted; this is defense prep)

If any result contradicts a claim the submitted thesis makes, nothing needs
fixing, but the author must know before the defense. Put it at the top of the
STATUS block under CONTRADICTS SUBMITTED CLAIM, quote the thesis claim and the
contradicting number, and continue the programme. Do not soften either one.

## Deliverables

1. CTG_EXPERIMENTS_LOG.md with a final table: every study, prediction,
   outcome, and the one sentence the author should say if asked about it.
2. SUGGESTED_BACKUP_SLIDES.md: for each result worth showing, a proposed slide
   as a short description plus the figure or table it would use plus three
   lines of speaker notes, ranked by how likely the question is. SUGGESTIONS
   ONLY, in this file, nothing added to the deck.
3. Every number traceable to a JSON, with paths listed.
4. The Doc/ untouched proof.

Constraints: nothing under Doc/ modified; no training of any model beyond what
Study C's arms already require from existing checkpoints; import the existing
corruption, loading and seeding helpers rather than reimplementing them; strict
separation between the classifier that steers and the classifier that scores;
pre-register every run; stop and ask on anything outside the budget rule.
