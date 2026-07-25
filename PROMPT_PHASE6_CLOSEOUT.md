# Prompt for Claude Code, Phase 6 (the last one): mechanism proof, examples, trajectory redesign, thesis done

One session, Opus 4.8, highest reasoning setting (`/model`, Opus 4.8, high or
extended thinking). After this session the thesis is DONE: no TODOs, no
placeholders, no open experiments. Paste the block below.

---

Read CLAUDE.md, the Phase 5 entries at the bottom of REVISION_LOG.md, and the
Stage 4 G-section plan logged there, before doing anything. Log everything to
REVISION_LOG.md, appended, timestamped. Pre-register predictions before every
analysis. No em-dashes anywhere. Everything here is CPU except, at most, one
tiny regeneration noted in Part 2. The repo reorg remains DEFERRED: move,
rename, archive nothing.

## Part 1: the per-class confusion analysis (the mechanism proof; run first)

Purpose: the G write-up currently INFERS that residual guide-judge disagreement
falls asymmetrically by class. This analysis demonstrates or refutes it, so it
runs before one word of the G section is written.

1. LABEL-ALIGNMENT SANITY FIRST. On the real held-out SST-2 validation
   sentences, compute per-class accuracy for the guide and for the judge
   separately. This exposes any label-mapping flip (a silent flip would
   perfectly mimic the predicted asymmetry for a trivial reason). Overall
   accuracies must reproduce the logged 79.7% (guide) and 88% (judge); report
   the per-class split.
2. THE CONFUSION TABLE, computed on the UNGUIDED on-domain generations only
   (the neutral calibration surface; measuring alignment on guided text would
   be circular). Two-by-two guide-verdict vs judge-verdict counts, plus the two
   conditional agreement rates: P(judge says pos | guide says pos) and
   P(judge says neg | guide says neg), with bootstrap CIs.
3. Report guided-text agreement alongside as labeled context, not evidence.
4. Pre-registered prediction: the conditional agreement is asymmetric by class
   and the HIGHER channel matches the direction that transferred in G-prime
   (positive). Decision rule: asymmetry confirmed means the instrument-alignment
   mechanism is demonstrated and the table goes in the thesis; symmetric means
   the G section says the residual cause is not fully identified, and no new
   explanation is invented post hoc.
5. Artifact: results_revision/rev_confusion.json plus the LaTeX table, staged.

## Part 2: the guided-generation examples (so the steering is visible in text)

- Selection, immutable: with default_rng(0), draw 3 pair indices per cell
  (gamma in {2,4} x label in {neg,pos}) from the G-prime pairs, 12 pairs total.
  No filtering, no redrawing, no swapping, even if a draw is dull or contains
  crude review language; the caption states the draws are seeded and
  unfiltered. Statistics live in Part 1's table; examples are illustration.
- Each pair shows: prompt, unguided continuation, guided continuation, the
  guide's verdict on each, the judge's verdict on each. Pairs where guide and
  judge disagree are the instrument-alignment mechanism made visible; do not
  annotate them as failures, annotate them as what they are.
- Trust-region before-and-after: add 2-3 guided pairs from the Phase 4
  OFF-domain run at high gamma, where the NLL climbed, next to their G-prime
  counterparts. If Phase 4 per-item text was not stored, regenerate only those
  pairs deterministically from the Phase 4 config and seeds (the one possible
  GPU use this session; minutes).
- Artifacts: results_revision/gprime_examples.md (readable, printed in full in
  the closing report so the author can eyeball immediately) and a staged LaTeX
  appendix block, cross-commented from the G section.

## Part 3: the trajectory figure, redesigned (projection demoted, distances promoted)

The author's objection is correct and becomes part of the thesis: a 2D
projection of a 1280-dimensional space cannot carry the argument. The primary
figure is exact in the full space.

1. DECODING TRAP, do not fall in: the npz vocab_embeddings array is a 6000-token
   SUBSAMPLE kept for the background cloud. All nearest-token decoding and all
   distance computation use the FULL embedding matrix loaded from the gpt2sft
   checkpoint (CPU, wte weights only). The subsample is decoration only.
2. PRIMARY FIGURE, one panel per config (dls_policy, dls_random, cls MH on, cls
   MH off; the gn-on cls panel optional), SAME seeded sequence across panels
   (default_rng(0) picks one of the 6 traced sequences; verify it exists in all
   configs). X axis: step 0-50. Y axis, symlog (linthresh 1, so DLS's exact
   zeros are visible, not dropped): L2 distance to the ground-truth token
   embedding in the FULL 1280-dim space. Second line for CLS: distance to the
   nearest token of any kind (the off-manifold measure; identically zero for
   DLS, state it in the caption rather than plotting a flat zero line if it
   clutters). Token strip under the axis: the decoded token (DLS state itself;
   CLS nearest-token projection), annotated at every change, capped at ~15
   labels with elision marked. CLS rejected steps inferred EXACTLY from state
   repetition (identical consecutive states = rejected proposal); mark them as
   ticks. Final landing token and ground truth printed at the right edge with
   match/no-match.
3. SECONDARY: keep ONE PCA cone figure, one sequence per panel (no more
   six-star overlays), with the explained-variance ratio of the two components
   computed from the fitted PCA and PRINTED IN THE CAPTION, plus the sentence
   that the projection is illustration and the full-space distances are the
   evidence. DROP the t-SNE figures entirely.
4. The script (revision/plot_trajectories.py, reworked) also writes
   figures/fig_traj_stats.json with the per-config distance statistics (the
   115-128 MH-on off-manifold distances, the 979 no-MH max, final distances,
   token-change counts), so every number quoted in the thesis has a source file
   for its % comment. The numbers diff in Part 4 includes this JSON.

## Part 4: thesis edits (Doc/), everything, nothing left open

1. G SECTION per the logged decision-case (b) plan, now with: the Phase 4
   off-domain table and the G-prime on-domain table coexisting, each labeled by
   domain; the agreement ladder; the trust-region NLL result; the flip stated
   carefully as setting-contingent (the working direction tracks instrument
   alignment; it is NOT a general "positive steers, negative does not" claim);
   Part 1's confusion table with its conclusion at the strength the data
   licenses; Part 2's examples cross-referenced.
2. ABSTRACT softened to the agreed three-claim shape, old text in a % comment:
   the LM-likelihood gradient claim at full strength (theorem plus
   measurement); the constraint-direction claim precise (signal exists, cannot
   rescue generation off-manifold, and even with fluency controlled the
   transfer is bounded by instrument alignment); the training-free premise
   tested and refuted with the trained-objective pilot confirming the diagnosis.
   Check the front matter for a German Kurzfassung or Zusammenfassung and
   update it in lockstep if present. Verify no sentence in the intro or
   conclusion now contradicts the softened abstract.
3. TRAJECTORY: the new primary figure and the one PCA secondary replace the
   placeholders; the off-manifold numbers (115-128 vs token spacing 1.8-2.8, a
   factor of 40-70; max 979 without MH) are promoted into the body text of the
   CLS results section and the discussion, sourced to fig_traj_stats.json; the
   epistemic sentence about projections goes in the text, not only the caption.
4. SHOWCASE appendix (already staged) placed, cross-commented from results.
5. CITATIONS: resolve the twisted-SMC venue/year leftovers with the always-true
   rule: cite as arXiv preprints with the IDs already in the URLs unless a
   peer-reviewed venue is positively confirmed; never guess a venue. Sweep the
   bibliography once for any other entry with a guessed-looking venue.
6. AVAILABILITY statement points at README.md and the artifact map. RQ map
   updated for any new subsection.
7. COMPLETENESS SWEEP, the "nothing incomplete" gate: grep Doc/ for TODO,
   PLACEHOLDER, TODO-AUTHOR, FIXME, XXX, \todo and resolve every hit or list it
   as an explicit author decision; latexmk clean with zero undefined references
   and citations; lists of tables and figures populated and current;
   bibliography page 2 overflow still fixed; render every changed page plus
   bibliography page 2 to PNG and inspect.
8. NUMBERS DIFF against results_revision/numbers.json plus rev_gprime.json,
   rev_confusion.json, fig_traj_stats.json, and every Phase 4-5 JSON. Report
   any mismatch, never silently fix.

## Part 5: the closing certification

Append the final report to REVISION_LOG.md: every concern 1-19 with its final
status and where it landed in the thesis; every phase experiment with its
result file; the artifact map confirmed complete; the completeness sweep
result (zero open items, or the explicit short list needing the author);
compile status; and the closing statement that nothing experimental remains
open. Print gprime_examples.md in full at the end so the author can read the
steering examples directly. Do not commit; the author commits.

Constraints: nothing beyond this slate; move, rename, archive, delete nothing;
do not touch core/ samplers; Doc/ edits keep removed text in % comments; seeded
selections are immutable once drawn; established voice, no formulaic
scaffolding, no em-dashes. If Part 1 comes back symmetric or any prediction
fails, write the honest version, not the convenient one.
