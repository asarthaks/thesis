# Revision Plan: Experiments and Fixes for Examiner Concerns

Each section states the issue, the concrete steps or experiments to run, and what the fix buys the thesis. Concerns are ordered by priority. A suggested timeline is at the end.

---

## Priority 1: Must fix before submission

### 1. No statistical machinery behind the null result

**Issue.** The central claim is an equivalence claim ("gradient is indistinguishable from random") but Table 5.1 and the surrounding text report point estimates only. "Within noise" is asserted, never demonstrated. The 0.670 gap on the len_beta=1 variant is dismissed without a test.

**What to do.**

1. For every configuration, keep the per-sequence final KL values (n=200 per cell). You almost certainly already have these in the raw logs, so no reruns needed.
2. Compute paired differences per sequence between policy and grad-norm-preserved (same sequence, same corruption seed, same schedule). Report the mean difference with a 95% bootstrap CI (10k resamples) for every row of Table 5.1.
3. Run a paired Wilcoxon signed-rank test per row as a sanity check on the bootstrap.
4. Run a formal equivalence test (TOST) with a pre-declared equivalence margin. A defensible margin: the standard deviation of final KL across random seeds of the fully random method, or 5% of the baseline KL. State the margin and its justification in the methodology chapter, not the results chapter, so it does not look post hoc.
5. Do a retrospective power analysis: given the observed per-sequence variance and n=200, what is the smallest true policy-vs-random gap the design could detect at 80% power? Report that number once in Section 5.5.
6. Treat the len_beta=1 result (random beats gradient by 0.670) as its own finding. Test it separately. If significant, it strengthens the thesis: the gradient is not just uninformative there, it is actively harmful, which fits the mis-specified surrogate story from Section 5.6.

**What it buys you.** The difference between "we did not find an effect" and "we can bound the effect below X." The first is a shrug, the second is a result. This is the cheapest fix in the whole list because it is pure analysis on data you already have, and it upgrades the headline claim of the entire thesis.

---

### 2. Missing baselines for the KL metric

**Issue.** Final KL values around 6 to 8 float in a vacuum. There is no anchor for what good, trivial, or terrible looks like, and the obvious non-MCMC baseline (resample the masked position from the model's own conditional) is absent.

**What to do.** Run four cheap baselines on the identical 200 sequences and corruption seeds:

1. **Ground-truth anchor.** Insert the original token back and compute the final KL. This is the floor any method should be compared against.
2. **Conditional resampling (the killer baseline).** For single-token recovery, take one forward pass at the masked position, sample (or take argmax) from p(x_i | x_<i), optionally rescored by the full-sequence likelihood over the top-k candidates. For multi-token recovery, do a few sweeps of Metropolized Gibbs: propose from the conditional, accept with the exact energy. This is exactly the Gibbs baseline Grathwohl et al. compare against, so its absence is conspicuous.
3. **Random-token floor.** Substitute a uniformly random vocabulary token and compute the final KL. This calibrates the top of the scale.
4. **Untouched-corruption reference.** The KL of the corrupted sequence before any sampling, so the reader can see how much any method improves at all.

Add all four as horizontal reference lines or extra rows in Table 5.1 and the trajectory figures.

**What it buys you.** Two things. First, interpretability: every number in Chapter 5 gains meaning. Second, and more importantly, if conditional resampling beats every Langevin configuration (it very likely will), it sharpens your own argument: the exact energy evaluation inside the MH accept step is doing all the work in your best configuration, and the gradient contributes nothing on top of what one forward pass provides. That is a stronger and more quotable version of the gradient fallacy. Cost: a few GPU-hours.

---

### 3. Circular evaluation (model grades its own homework)

**Issue.** The primary metric is a KL computed under the same model whose likelihood Section 5.7 argues is a bad measure of quality. The thesis says "likelihood is not quality" and then measures quality with the model's own predictive distribution. Also, the reference distribution in the KL is not precisely defined in Section 4.4.

**What to do.**

1. Write out the KL definition exactly in Section 4.4: which distribution, at which position, conditioned on what, compared against what. One equation ends the ambiguity.
2. **External-judge rescoring.** Take the final recovered sequences from a representative subset of configurations (policy, grad-norm-preserved, random, on the base model, ~200 sequences each) and score them under Llama-3 (perplexity of the recovered sequence, or KL at the recovered position under Llama). If the ranking of methods is preserved under an independent judge, the circularity objection dissolves.
3. Report exact-match and top-5 recovery accuracy as secondary metrics. You argued against exact match in 4.1, and the argument is fine, but reporting it costs nothing and shows the null result is not an artifact of the metric choice.
4. **Small human evaluation.** 100 infilled sequences, three conditions (policy, random, ground truth), shuffled, rated for coherence on a 3-point scale by 2 to 3 raters. Report agreement (Krippendorff's alpha). Even a small study closes the loop between "KL says no difference" and "humans see no difference."

**What it buys you.** Immunity to the single most predictable defense question: "your metric comes from the very object you claim is broken, so why should I believe it?" The external-judge rescoring is an afternoon of compute; the human eval is a weekend.

---

### 4. Overclaimed scope against successful non-gradient samplers

**Issue.** The thesis tests one gradient parameterization (with respect to the input embedding of the masked position) and concludes the frozen likelihood is not a usable energy. But Mix-and-Match (Mireshghallah et al. 2022) does MH with MLM proposals on a frozen product-of-experts energy and works, and twisted SMC methods (Lew et al.; Zhao et al. 2024) sample frozen-LM posteriors successfully. Their existence shows the energy can be a fine target when the proposal is good.

**What to do.**

1. Add both lines of work to Chapter 3 with a dedicated paragraph: "samplers on frozen-LM energies that do not use the energy gradient."
2. Narrow the headline claim wherever it appears (abstract, intro, conclusion): from "the frozen likelihood is not a usable energy function" to "the frozen likelihood does not admit local gradient-based navigation; its gradient is not a usable search direction." The likelihood-trap argument (low energy is degenerate text) survives at full strength and should be stated separately, because it applies to any method that reaches the minimum, gradient or not.
3. Optional but high-value experiment: run Metropolized conditional/Gibbs proposals (which is baseline 2 from concern 2) and frame it explicitly as "a non-gradient sampler on the same energy." If it works where the gradient fails, you have demonstrated, inside your own experimental platform, that the failure lives in the gradient and not the energy per se. This is a cleaner separation than the thesis currently has.
4. In the COLD/MuCoLa discussion, note explicitly that those methods differentiate through soft token distributions where the self term is visible to the gradient, so your structural-blindness result applies to your parameterization and the transfer to theirs is an argument, not a measurement. One honest sentence.

**What it buys you.** The claim becomes both smaller and stronger. Smaller because it no longer contradicts working systems in the literature; stronger because the Gibbs experiment turns "the energy is broken" vs "the gradient is broken" from a rhetorical distinction into a measured one.

---

### 5. GFlowNet unifying experiment: did tuning change the energy at all?

**Issue.** Table 5.2 shows Langevin on the tuned energies matches Langevin on the base and concludes amortization cannot repair the landscape. But if 500 to 2000 steps of LoRA barely moved the joint likelihood, indistinguishability is trivially expected and the experiment proves little.

**What to do.**

1. On a held-out set of ~1000 sequences, compute per-sequence log-likelihood under the base model and each GFlowNet variant. Report the mean absolute difference, the correlation between base and tuned scores, and the KL between the two models' next-token distributions averaged over positions.
2. Repeat the anisotropy and linearization-correlation measurements on the tuned models' effective energy (you already have the linearization correlations: 0.024, 0.057, 0.046; just present them alongside the model-divergence numbers).
3. Then write the conclusion conditionally: if the tuned models diverge substantially from the base in likelihood yet the landscape diagnostics are unchanged, the "amortization does not repair the energy" claim stands and is now supported. If the tuned models are nearly identical to the base, retitle the finding honestly as "light-touch LoRA amortization leaves the energy unchanged" and note that heavier tuning is an open question.
4. If time permits, one run with a larger LoRA rank or full fine-tuning for the same step budget would show whether the result is capacity-limited.

**What it buys you.** Closes the logical gap in the single experiment the whole second half of the thesis leans on. Steps 1 and 2 are pure measurement on existing checkpoints, roughly one GPU-day.

---

### 6. Internal numerical inconsistencies

**Issue.** Five concrete contradictions between text, figures, and tables that any examiner will find:

a. Section 5.3 reports acceptance rates of 0.03% (within-cell) and 3.7% (boundary), Figure A.3 shows 6.4% and 7.4%.
b. Text says length slope is about -1.12 nats/token; Figure A.4's legend says -0.11 with r = -0.00, and most points sit at the length cap of 40.
c. Section 5.6 uses mean inter-token distance 2.35; Section 5.8 reports 1.82 (nearest-neighbour) and 2.77 (pairwise).
d. Figure 5.5 annotates "linearization radius r ≈ 0.89" while the caption denies a threshold exists.
e. Table A.1's CLS rows are identical to three decimals with and without MH, but Section 5.4 says the uncorrected CLS visits 46 cells per run, so the two chains cannot end in the same place.

**What to do.**

1. Trace each number back to its results file (the thesis says every figure is annotated with its source file, so this is grep work). For (a), determine which configuration each pair came from; if they are different configurations, label both fully; if the same, one is a bug.
2. For (b), rerun the regression excluding capped-length generations, report both the censored and uncensored slope, and reconcile the legend with the text. If the true slope is -0.11, the brevity-incentive argument in 5.7 and 5.10 needs to be reweighted, so this one matters beyond cosmetics. Consider a Tobit-style censored regression given the cap, or at minimum acknowledge the censoring.
3. For (c), define which statistic "inter-token distance" means once, in Section 4.4 or 5.8, and use that word for that statistic everywhere. If 2.35 is a third quantity (e.g. mean distance to the top-k proposal candidates), define it.
4. For (d), either derive 0.89 from a stated criterion (e.g. the distance at which binned correlation drops below 0.05) and say so in the caption, or delete the annotation. The caption's honest framing is right; the figure must match it.
5. For (e), diff the two run configurations and the logged trajectories. If the corrected chain never moves, its final state is the corrupted input and its KL should equal the untouched-corruption reference from concern 2, which gives you an independent check. If the uncorrected chain's number is genuinely identical, something in the logging or evaluation pipeline collapsed the two runs, and you need to find it before submission because it casts doubt on the whole grid.
6. After fixing, write a small script that regenerates every reported number from the results files and diffs against the LaTeX source. Run it as a final pre-submission check.

**What it buys you.** Credibility. A diagnostic thesis whose contribution is careful measurement cannot afford measurable self-contradictions. Item (e) especially needs an answer, not a patch.

---

### 7. The CLS energy is underspecified and its within-cell behavior is puzzling

**Issue.** If the CLS energy depends on the continuous state only through the projected token, it is piecewise constant, the within-cell gradient is zero, MALA degenerates to a random walk within a cell, and within-cell acceptance should be near 1 (as in the discrete case). The reported near-total rejection of within-cell moves implies the energy has additional state-dependence (a distance-to-embedding term, a soft projection, temperature-scaled mixture) that Section 2.4 never states.

**What to do.**

1. Write the CLS energy as an equation in Section 2.4: exactly how the continuous state enters, whether a soft or hard projection is used for the energy vs for the readout, and any auxiliary terms.
2. Add a one-paragraph derivation of what MALA's acceptance should look like under that energy for within-cell moves, so the measured 0.03% (or 6.4%, per concern 6) is compared against a theoretical expectation rather than left as a bare number.
3. If the implementation follows MuCoLa, check whether their distance/regularization term is present and say so; that term would explain nonzero within-cell drift and low acceptance.

**What it buys you.** The MH-breakdown section (5.3) is one of the thesis's best results, and it currently rests on an energy the reader cannot reconstruct. One equation and one paragraph make it airtight.

---

### 8. The diffusion positive control was designed but never run

**Issue.** The thesis's central causal claim (the training objective is the culprit) makes a falsifiable prediction, argues the test is cheap, notes that off-the-shelf models sharing the GPT-2 tokenizer exist, and then leaves it as future work. That is the strongest card left unplayed.

**What to do.** Run a pilot, not the full grid:

1. Take SEDD-small or SEDD-medium (Lou et al. 2024, GPT-2 tokenizer, weights public).
2. Run only the **linearization diagnostic** on it: for the same 200 ROCStories sequences and masked positions, compute the model's score/ratio-based analogue of the surrogate against the true change, and the binned correlation-by-distance curve, mirroring Figures 5.4 and 5.5. No sampler tuning needed, no annealing schedules, no MH machinery.
3. If feasible in the remaining time, add the direction-vs-magnitude ablation with the model's native denoising-based proposal at a single noise level.
4. Report it as a pilot section (5.12 or an appendix): "a preliminary positive control."

Expected outcomes and how to write each: if the correlation is substantially positive where the AR models gave ~0, the causal diagnosis is confirmed and the thesis's contribution jumps a tier. If it is also ~0, the diagnosis is refuted at the level of training objective and the honest conclusion is that the cause lies deeper (discreteness itself, embedding geometry), which is still a publishable and more surprising finding. Either way the thesis wins, which is the definition of a good control.

**What it buys you.** Converts the thesis from "a diagnosis plus a proposed test" into "a diagnosis plus its first confirmation (or refutation)." Estimated cost: 2 to 4 days including reading the SEDD codebase. If genuinely infeasible before the deadline, add a paragraph explaining precisely why (e.g. score parameterization differences make the surrogate non-comparable without care) so the omission reads as considered rather than convenient.

---

## Priority 2: Should fix, moderate effort

### 9. Task generality (masked recovery vs free-form generation)

**Issue.** All sampling results come from 1 to 3 token recovery in five-sentence ROCStories, while the plug-and-play promise concerns free-form generation.

**What to do.** Two options, either suffices. Cheap: add a limitations paragraph in 6.4 stating explicitly that the diagnostic task was chosen for controllability and that the constrained-generation extension (which includes continuation) is the only free-form evidence. Better: rerun the core ablation (policy vs grad-norm-preserved, DLS, MH on, one schedule) on a prefix-continuation task with 20-token spans, 100 sequences, one model. One table row per method.

**What it buys you.** Removes the "you only showed this on a toy task" objection, or at least pre-empts it honestly.

### 10. Likelihood trap: pooled correlation and missing Llama measurement

**Issue.** The +0.65 correlation pools generations across decoding strategies, so strategy and likelihood are confounded; and the trap is never measured on Llama-3, weakening the cross-architecture story.

**What to do.** Report within-strategy correlations (likelihood vs repetition inside ancestral sampling alone, inside temperature sampling alone). Run the same decoding sweep on Llama-3, which is a few hours of generation, and add one row to the results. If the within-strategy correlation weakens, present the between-strategy pattern (greedy/beam worst) as the primary evidence, which is what Holtzman et al. did anyway.

**What it buys you.** The trap is a pillar of the unified argument in Chapter 6; this makes the pillar load-bearing on both models and removes a confound.

### 11. Constraint experiment: one direction, no uncertainty, unexplained anomaly

**Issue.** Section 5.11 tests only the positive-sentiment target, reports no error bars, and the cons_only mode steering the wrong way (-8.0) is left unexplained.

**What to do.** Run the negative-sentiment target under the same five modes (symmetric design, halves the chance the result is target-specific). Bootstrap the steering gains over sequences and report CIs; with ~200 generations, gains of a few points are likely within noise and the CIs will say so. For the cons_only anomaly, inspect 20 generations manually and check the classifier's confidence distribution on them; the likely story is that constraint-only optimization leaves the fluent-text manifold entirely and the classifier's outputs on gibberish are meaningless, which is worth one paragraph because it reinforces the thesis.

**What it buys you.** RQ4's answer currently rests on the weakest table in the thesis. This makes it as defensible as the others.

### 12. Oracle step-size fairness

**Issue.** If the oracle sweep tuned the schedule on the policy method and reused it for the random methods (or vice versa), the ablation is mildly biased in an unknown direction.

**What to do.** State in 4.3 exactly how the oracle and the fixed schedules were selected and for which method. If shared, rerun the oracle sweep independently per proposal method for one model (GPT-2 base) and show the conclusion is unchanged. That is a small sweep, not the full grid.

**What it buys you.** Removes a quiet asymmetry from the thesis's central comparison.

### 13. "The schedule worked on Llama-3" is undefined

**Issue.** Section 5.1's framing assumes a Llama-tuned schedule "worked," but later results show the gradient is useless (or harmful) on Llama too. What did "working" mean?

**What to do.** Define it in 5.1 in one sentence: presumably "the sampler moved and metrics changed over the run," i.e. calibrated motion, not successful guidance. This is exactly the motion-vs-guidance distinction the section itself introduces, so the fix is to apply the thesis's own vocabulary to its own opening anecdote.

**What it buys you.** Internal consistency; costs one sentence.

### 14. len_beta coverage

**Issue.** The claim "no setting between them was found to produce good text" is made having reported only len_beta = 0 and 1.

**What to do.** Either list the intermediate values actually tried (if any were, with a sentence on what they produced), or train one variant at len_beta = 0.5 for 500 steps and report its decoding behavior, or soften the sentence to "the two endpoints produce opposite degeneracies, and we did not find an intermediate setting that resolves both" with the coverage stated.

**What it buys you.** The interpolation claim currently outruns the reported evidence; any of the three options fixes that.

### 15. Configuration count arithmetic

**Issue.** The full factorial (2 samplers x 3 proposals x 2 MH x 2 gn x 2 schedules x 5 models = 240) does not obviously give 145, and the pruning is never explained.

**What to do.** Add a short appendix table or paragraph enumerating which cells were excluded and why (e.g. oracle variants counted separately, Llama run on a subset, CLS proposal methods reduced). A counting formula that lands on 145 is enough.

**What it buys you.** Reproducibility of the headline "145 configurations" claim, which appears in the abstract.

---

## Priority 3: Minor, low effort

### 16. Seeds, variance, and code availability

**What to do.** Rerun one representative configuration with 3 to 5 seeds and report the across-seed standard deviation, so the reader can see run-to-run noise (this also feeds the equivalence margin in concern 1). State seeds and hardware in Chapter 4. Add a code/data availability statement; the bitwise-equivalence test suite you mention is a selling point, so name it and where it lives.

### 17. Spearman phrasing

**What to do.** With ~400k candidate pairs, rho = -0.011 is statistically nonzero but negligible; phrase all correlation claims as effect-size claims ("negligible in magnitude, |rho| < 0.06 across all models") rather than "zero," and give the n once.

### 18. Prose repetition

**What to do.** One editing pass targeting the argument-arc restatements: keep the full recap once at the top of Chapter 6, and compress the mini-recaps that open Sections 5.5, 5.10, 6.2 and the chapter transitions to a sentence each. Likely saves 4 to 6 pages with no loss.

### 19. Computational cost comparison

**What to do.** Add a small table: forward+backward passes per recovered token for DLS, CLS, conditional resampling, and Gibbs, plus wall-clock per sequence on your hardware. Since avoiding enumeration is the main selling point of gradient guidance, showing that the gradient methods are also more expensive than the trivial baseline (from concern 2) completes the practical indictment.

---

## Suggested order of execution

**Week 1 (analysis only, no GPU):** Concern 6 (reconcile numbers, this gates everything), then concern 1 (bootstrap, TOST, power) on existing logs, concern 3 step 1 (define the KL), concerns 13, 15, 17.

**Week 2 (light compute):** Concern 2 (all four baselines), concern 5 steps 1 and 2 (model-divergence measurements), concern 10 (within-strategy and Llama trap), concern 3 step 2 (Llama rescoring).

**Week 3 (targeted experiments):** Concern 8 (SEDD linearization pilot), concern 11 (negative-sentiment target plus CIs), concern 12 (per-method oracle check), concern 16 (seed reruns).

**Week 4 (writing):** Concerns 4 and 7 (related work paragraph, scope narrowing, CLS equation), concern 9 (limitations paragraph or continuation run if time allows), concerns 14, 18, 19, and the regenerate-all-numbers script from concern 6 as the final gate.

If the timeline is tighter than four weeks, the non-negotiable core is: concerns 6, 1, 2, and 7. Those four turn the thesis from "convincing narrative" into "verified measurement," and everything else builds on them. Concern 8 is the highest-upside optional item: a two-day pilot that could confirm the thesis's central causal claim.
