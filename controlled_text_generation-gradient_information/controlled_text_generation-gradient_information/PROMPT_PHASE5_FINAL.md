# Prompt for Claude Code, Phase 5 (final close-out): G-prime, figures, qualitative, reorg, README, thesis

One session, Opus 4.8, highest reasoning setting (`/model`, Opus 4.8, high or
extended thinking). Paste the block below.

---

Read CLAUDE.md, the Phase 4 entries at the bottom of REVISION_LOG.md, and
REVISION_LOG_THESIS.md before doing anything. Log everything to REVISION_LOG.md,
appended, timestamped. Pre-register predictions before every run. No em-dashes
anywhere.

SEQUENCING IS A RULE, NOT A SUGGESTION. Stage 1 (experiments and artifacts) runs
first. Stage 2 (README) happens only after Stage 1's outputs exist. Stage 3
(verification) gates Stage 4 (thesis edits). REPO REORGANIZATION IS DEFERRED BY
THE AUTHOR'S DECISION: do not move, rename, or archive any file or folder in
this session, no matter how messy the layout looks. It goes on the TODO list
only (see Stage 2) and happens in a future session only when the author
explicitly asks for it and confirms.
Use the 9 A6000s: everything shardable is sharded through the queue as in Phase
4, fresh status dir.

## Stage 1a: G-prime, on-domain guided generation with a trust region

The Phase 4 diagnosis said the G asymmetry is measurement transfer off-domain:
guide and judge agree only 56-64% on the generated text, and gamma pushes text
further off-manifold where they diverge. G-prime tests the steering where the
instruments are calibrated, and tests the diagnosis itself.

Design:
- Prompts: held-out sentences from the SAME sentiment corpus behind
  train_sentiment_head.py, from a validation or test split that NEITHER the
  guide nor the judge saw in training (verify this in the loading code before
  running; if the corpus has no clean held-out split, construct one by hashing
  and excluding training ids, and log the construction). Prompt = the first 8
  to 12 tokens of a sentence; generation length matched to the Phase 4 G runs.
- Mechanism unchanged: commitment-time top-k reweighting (k=32), noisy
  classifier guides, concern-11 judge scores, strict role separation.
- NEW, the trust region: at each commitment, only candidates whose SEDD log
  posterior is within delta = 5 nats of the top candidate are eligible for
  reweighting; others keep zero adjusted mass. Record delta. If more than 90%
  of commitments have one or zero eligible candidates, widen to 8 nats once and
  log it. This caps the fluency cost by construction.
- Arms: unguided vs guided at gamma in {2, 4}, both target labels, n = 300
  pairs per label per gamma, sharded.
- Metrics: judge hit% per label with bootstrap CIs (the headline), NLL under
  gpt2sft, SELF gain (the guide's own verdict, mechanism check), and the
  DIAGNOSIS TEST: guide-judge agreement on the unguided on-domain generations.

Pre-registered predictions, in order of importance:
1. Agreement on unguided on-domain text rises clearly above the off-domain
   56-64%. If it does NOT rise, the domain diagnosis was wrong: STOP, report,
   and do not interpret the steering numbers until the cause is understood.
2. At gamma 2, guided beats unguided on the judge at BOTH labels, CIs
   excluding zero, with bounded NLL cost.
3. The trust region holds NLL bounded even at gamma 4 (unlike Phase 4's 7.1 to
   11.0 climb).
If prediction 1 holds but 2 still fails on the positive label, the honest
conclusion is final: symmetric held-out steering is not reachable with this
guide at this scale, and the Phase 4 asymmetric finding stands as the result.
Either outcome closes G; there is no third round.

## Stage 1b: qualitative showcase (for eyeballing, in the thesis and the repo)

- Selection rule, immutable once drawn: with a seeded RNG (seed 0), draw 10
  sample_idx values from the kl_baselines sequence set. No filtering, no
  redrawing, failures stay in. State the rule in the table caption.
- For each showcase sequence, produce: original text, corrupted text (corrupted
  token marked), and the recovery of each method: dls_policy (mh, gn),
  dls_random, the CLS flagship config, gibbs, cond_argmax, cond_topk_rescore,
  left_conditional independence MH, sedd_recovery small and medium,
  hybrid_medium. Pull recovered text from existing per-item CSVs where stored;
  where only ids or nothing was stored, regenerate ONLY the showcase sequences
  (deterministic corruption makes this exact; verify the regenerated corrupted
  tokens match the stored CSV rows before trusting the regeneration).
- Same treatment for G-prime: 3 guided-vs-unguided pairs per label, same
  seeded rule, seed 0 over the G-prime indices.
- Artifacts: results_revision/qualitative_showcase.json plus a LaTeX longtable
  appendix section (filled position in bold), cross-commented from the relevant
  results subsections per the thesis convention.

## Stage 1c: trajectory figures (the deferred PCA placeholders, plus t-SNE)

- Data check first: results_diag npz traces must cover DLS policy vs random
  (MH on) and CLS with MH on vs off, gpt2sft. Rerun collect_traces for any
  missing config only (cheap, one GPU).
- Method guards: PCA (2 components) is fit on the model's vocabulary embedding
  matrix, trajectories are PROJECTED into it, never fit on the trajectory
  points. Background: 2000 seeded-random vocabulary embeddings as a grey cloud
  so the anisotropy cone is visible. Mark the start state and the ground-truth
  token. t-SNE is the SECONDARY panel: fixed seed 0, perplexity 30, and a
  caption note that t-SNE distorts global distances; PCA is the primary
  evidence.
- Script: revision/plot_trajectories.py, standalone, writes
  figures/fig_traj_pca_dls.png, fig_traj_pca_cls.png and the t-SNE variants at
  thesis quality. Reproducible from the npz alone.
- Prose hook: connect what the figures show to the anisotropy numbers already
  in the thesis (nearest-neighbour 1.82, mean pairwise 2.77) and to the
  rubber-band and quenching narratives.

## Stage 2: README (on the CURRENT layout; no files move)

- README: write a new top-level README.md that absorbs and supersedes
  REVISION_README.md (leave the old file in place; add a one-line banner at its
  top saying it is superseded by README.md). Document the repo AS IT IS, messy
  or not; do not "clean up" anything. Contents: one-paragraph
  thesis summary and the claim map (necessity, sufficiency, guidance);
  environment setup; the CURRENT repo layout as it stands; how the queue works
  (manifest, worker locks, reset_incomplete with matching out_dir); exact
  commands to rerun every experiment family (grid, diagnostics, revision
  analyses, last_token, SEDD slate, G and G-prime); and THE ARTIFACT MAP:
  every table and figure number in the thesis mapped to the producing script
  and the result file path, anchored on results_revision/numbers.json and
  sedd_capability_summary.json. Known caveats section: SEDD runs excluded from
  AR reconcile, the gn=on bitwise gradnorm==random artifact, the concern 6a
  attribution note.
- TODO section at the bottom of the README, verbatim: "Deferred by author
  decision: repository reorganization (move results_* under results/, code
  under scripts/, strays to archive/). Do NOT perform this until the author
  explicitly asks for it and confirms. When it happens: git mv only, Doc/
  never moves, nothing is deleted, and a full reference sweep across *.py,
  *.sh, *.md, *.tex (including % source comments and reconcile globs) plus
  the Stage 3 verification gate must follow." Mirror the same entry in
  REVISION_LOG.md so it survives even if the README is rewritten.

## Stage 3: verification gate (must pass before any thesis edit)

1. python -m py_compile on every script in the tree.
2. Rerun the no-GPU analyses and numerically diff their JSONs against the
   existing copies: identical or explained (this catches anything Stage 1
   accidentally disturbed).
3. One tiny queue smoke test (one shard, n=5) through run_queue.sh with a
   fresh status dir.
4. Thesis compiles clean, all figures resolve (including the new Stage 1c
   figures), lists of tables and figures update.
5. Numbers diff against numbers.json plus all Phase 4 and 5 JSONs.
Log the gate results. If anything fails, fix before Stage 4.

## Stage 4: thesis edits (Doc/)

1. G section, per the G-prime decision table: if predictions 1 and 2 held,
   upgrade to the symmetric on-domain result with the off-domain asymmetry
   kept as the instrument-validity finding (it dovetails with concern 11's
   off-manifold theme); if prediction 1 held but 2 failed on one label, keep
   the Phase 4 section and add G-prime as the on-domain control showing the
   limit is real, not instrumental; if prediction 1 failed, report the refuted
   diagnosis honestly and leave the Phase 4 asymmetric finding as the result
   with cause marked open. In all cases the trust-region NLL result is
   reported.
2. Abstract softening, three claims at their true strength, old text kept in a
   % comment: (i) the LM-likelihood gradient carries no usable directional
   signal, full strength, theorem plus measurement; (ii) the constraint
   classifier's gradient direction carries measurable signal but cannot rescue
   generation, because following it off the fluent manifold breaks the
   classifiers themselves; (iii) the training-free premise, tested, fails for
   a structural cause, and a trained-objective pilot confirms the diagnosis
   and shows the capability at a training cost. Check the front matter for a
   German Kurzfassung or Zusammenfassung; if present, update it in lockstep.
   Keep the intro consistent (the hypothesis-under-test reframing from Phase 4
   already aligns; verify no sentence now contradicts the softened abstract).
3. Trajectory placeholders replaced with the real figures from Stage 1c,
   regeneration instructions in the % comments now pointing at
   plot_trajectories.py, cross-comments in the results text.
4. Qualitative appendix from Stage 1b, cross-commented from results, caption
   stating the seeded selection rule.
5. Code and data availability statement updated to reference the README and
   the artifact map.
6. Final pass: recompile clean, render every changed page plus bibliography
   page 2 to PNG and inspect, rerun the numbers diff, update the
   RQ-to-subsection map if G-prime added a subsection, append the closing
   report to REVISION_LOG.md with the remaining author items (expected: none
   beyond the twisted-SMC venue/year confirm if still open).

Constraints: nothing beyond this slate; no sample-many-and-filter under any
name; do not touch core/ samplers except reading; do not move, rename, delete,
or archive ANY existing file or folder (the reorganization is deferred and
gated on explicit author confirmation); Doc/ edits keep removed text in %
comments; seeded selection rules are
immutable once drawn; established voice, no formulaic scaffolding, no
em-dashes. If any gate fails or any outcome contradicts its pre-registered
prediction, stop, report, and do not write it into the thesis until the cause
is understood.
