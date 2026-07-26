# Master's Thesis Evaluation (Evaluation 3)

**Candidate:** Sarthak Singh
**Title:** *Energy-Guided Sampling for Controllable Text Generation: From Langevin Dynamics to Amortized Inference*
**Institution:** Institut für Maschinelle Sprachverarbeitung, Universität Stuttgart
**Documents reviewed:** `Doc/final/thesis/` (125 pp.), `Doc/final/proposal/` (12 pp.), plus the source code and result files, which were consulted to verify claims.
**Reviewer role:** strict, fair supervisor and examiner, IMS grading criteria.
**Date of review:** 2026-07-26

---

## 1. Executive Summary

This is a serious, ambitious, and unusually honest piece of work. The candidate set out to build controllable generation on faithful Langevin dynamics, found the foundation did not hold, and converted the project into a diagnostic study of *why* frozen autoregressive likelihoods fail as energy functions for gradient-guided sampling. That pivot is handled with more discipline than most students manage: the research questions are re-derived explicitly from the proposal, the failure is attacked from several independent angles, and the thesis ends with a positive control (a score-entropy diffusion model) that most published papers in this area would not have run.

The craftsmanship of the empirical apparatus is genuinely impressive. Numbers in the thesis were spot-checked against `results/revision/numbers.json` and `rev_stats_gpt2.json`: the flagship values (6.541, 6.474, 6.370, 6.430, paired mean difference 0.171, CI [-0.285, 0.619], Wilcoxon p = 0.400, MDE 0.652) all match the result files exactly. Provenance is annotated in the LaTeX source for essentially every quantity. Reproducibility, replicability, and documentation are at or above the level expected from a good conference submission.

Against this, there are four substantive scientific problems, one of which is serious. **The most important is that the discrete sampler's proposal distribution is, at the hyperparameters used, numerically indistinguishable from uniform over the 50,257-token vocabulary** (measured mean proposal entropy 10.8248 nats against log|V| = 10.82491). At that configuration neither the gradient term nor the distance term of the proposal can influence the sampled token, so the headline "policy versus norm-matched random" null is guaranteed a priori by the configuration rather than established by it. The thesis's central conclusion is nonetheless probably correct, because it is independently supported by the linearization measurement, the final-position argument, and the gradient-free baselines. But the experiment the thesis foregrounds is not the experiment that carries the argument, and that distinction is never made.

Second, the single most striking number in the study is buried: **exact-match recovery is 0.0% in 139 of the 145 configurations and never exceeds 1% in the remaining six.** It appears once, as a reference row in a diffusion table on p. 78. Third, the Metropolis-Hastings correction is applied asymmetrically across the compared arms in the code, which confounds precisely the comparison the thesis rests on. Fourth, the "gradient" tested is one specific Jacobian slice, and a different, arguably more faithful, instantiation of the cited samplers would not have the structural blindness the thesis diagnoses.

None of these overturn the thesis, and all four are fixable largely by re-analysis and re-reporting rather than by new GPU work. But they mean that the strongest version of this argument has not quite been made, and a careful examiner will press on all four in the defence.

**Estimated grade: 1.7 (gut, upper band).** Justification and the conditions that would move it in either direction are in Section 6.

---

## 2. Strengths

### Scientific

- **A real research question, sharply isolated.** The thesis identifies an assumption that the COLD/MuCoLa line of work makes and never tests (that the frozen likelihood's gradient is a navigable search direction), and it builds an experiment specifically to test that assumption rather than a proxy for it. The three-way ablation (true gradient / norm-matched random direction / fully random) with the normalization confound explicitly removed is a genuinely well-designed contrast, and the reasoning for it in Section 5.5 is exemplary.
- **Convergent evidence from independent routes.** Linearization correlation (rho approximately 0 over 400,000 candidate pairs, on all five energies), the MH acceptance decomposition (target +4.60 against proposal -1325), the GFlowNet unification experiment, the analytic final-position result, and the diffusion control do not share machinery. That is exactly the right architecture for a diagnostic thesis.
- **The final-position analysis (Section 5.12) is the intellectual high point.** The observation that under shifted causal indexing the gradient with respect to the last token's input embedding is exactly zero, while the energy difference between candidate final tokens is exactly the conditional log-ratio, is elegant, correct, and precisely scoped (which gradient, with respect to which representation, under which indexing). The empirical companion, showing a measured gradient norm of exactly 0.0000 over roughly 15,000 evaluations and 100.00% acceptance for the independence sampler, is a beautiful confirmation that also certifies the implementation.
- **The positive control was run, not merely proposed.** Turning "the training objective is the cause" into a falsifiable prediction and then testing it with SEDD on the same tokenizer is what distinguishes a diagnosis from a complaint. The hybrid experiment (same exact-energy chain, swap only the proposal) is the correct design for isolating the direction signal.
- **Gradient-free baselines are the right control and were the right instinct.** Showing that top-k rescoring (KL 4.43) beats every one of 145 Langevin configurations, and that Gibbs (6.69) matches the best of them, is what licenses narrowing the claim from "the energy" to "the gradient". Many theses would have stopped at the null.
- **Statistical practice well above MSc norm.** Paired comparisons on identical corruption seeds, bootstrap CIs, Wilcoxon tests, a pre-specified equivalence margin calibrated to measured seed noise, a reported minimum detectable effect, and a four-seed stability run (sd 0.183). The thesis also correctly refuses to claim equivalence when TOST does not certify it.
- **Honest scoping.** The scoped formulation ("in the tested token-substitution settings, the input-embedding gradient provided no reliable proposal advantage") is used consistently across abstract, introduction, results, discussion, and conclusion. The "gradient fallacy" label is used exactly once and immediately bounded. Section 6.4 explicitly distinguishes "the training-free premise is refuted for this class of Langevin methods" from "the training-free premise is refuted", and points at the thesis's own gradient-free results as counter-evidence. This is the kind of discipline examiners rarely see.
- **Explicit accounting of dropped deliverables.** Section 6.4 names every proposal commitment that became moot (RealToxicityPrompts, MAUVE/Self-BLEU, the 70B judge, posterior coverage, CommonGen) and gives a reason for each. This is the correct way to handle a pivot.

### Sustainability and reproducibility

- Deterministic per-index corruption seeding, so paired analysis needs no reruns; a bitwise-equivalence suite certifying the instrumented samplers against the reference; a resumable job queue with per-job memory annotations; a README artifact map linking each table and figure to the script and the result file. The appendix even documents the grid arithmetic (145 = 5 x 29) and reconciles it against folder counts. File-level traceability was verified on several numbers and it holds.

### Written document

- **The golden thread is real.** Every results subsection is motivated by a question the previous one leaves open, and Section 6.1 answers all four RQs against the specific sections that establish them. Nothing raised in the introduction is left unanswered.
- **Background is written to expose assumptions, not to survey.** Section 2.4's separation of the piecewise-constant projected energy from the differentiable pathway, and the statement that all non-Lipschitz claims attach to the latter, is mathematically careful in a way that most student work is not.
- **The qualitative appendix (A.6) is exemplary.** A stated, seeded, unfiltered selection policy, with failures kept in, and an explicit statement that no number in the thesis derives from these tables. This is better practice than most published papers.
- Prose is fluent and grammatical, terminology is consistent, cross-references resolve, all 47 bibliography entries are cited and no citation is missing an entry, and only seven overfull boxes remain in a 125-page document.

---

## 3. Critical Issues and Weaknesses

### 3.1 Severe: the discrete sampler's proposal is numerically uniform

This is the finding an examiner will lead with. From `results/grid/gpt2_v2/gpt2-large.dls.policy.mh.gn.free.s50.json`, the mean proposal entropy is **10.8248 nats at every step for the first roughly 44 of 50 steps**, against log(50257) = **10.82491**. The proposal distribution is within roughly 1e-4 nats of the uniform distribution over the entire vocabulary. Figure 1's bottom panel shows this plainly, and the thesis does not comment on it.

The consequence is structural, not cosmetic. In Equation (2.9) the token is chosen from a softmax over `t1 + t2`, divided by `temperature = 5.0` (`scripts/run_experiment.py:144`, `core/base_sampler.py:7`). With `grad_normalization` on, the gradient-alignment term has magnitude on the order of ||e(v) - e(x)||/2, approximately 1.2 nats before the temperature divide, and the distance term at epsilon approximately 10.5 contributes about -0.37; after dividing by 5, both are small relative to a 10.82-nat entropy budget, and the measured entropy confirms the net effect is nil. The sampler is therefore, for roughly 90% of its schedule, a uniform-proposal independence chain with an MH filter, not a gradient-guided Langevin sampler.

Three implications:

1. **The central ablation cannot detect what it was built to detect.** Comparing "policy" and "norm-matched random" proposals when both are within 1e-4 nats of uniform is comparing two copies of the same object. A perfectly informative gradient would produce the same null at this configuration. The null is therefore uninformative *at the sampling level*.
2. **The thesis's own code logs the diagnostic that would have caught this and never reports it.** `core/dls.py` records `t1_std`, `t2_std`, and `t2_over_t1` with the comment "if t1 dominates, the proposal is essentially a distance-weighted random walk and the gradient is decorative." That quantity appears in no result file, no table, and no figure in the thesis. It should be the first number in Section 5.5.
3. **The step-size calibration of Section 5.1 optimised the wrong criterion.** Section 5.1 defines a schedule as "working" if it produces *calibrated motion* (tokens change), and explicitly notes this is weaker than *guided motion*. That distinction is correct and well made, but the study then never checks whether any schedule exists that produces both. The oracle sweep searches epsilon alone; the temperature is never swept. A two-dimensional sweep over (epsilon, temperature) reporting proposal entropy and `t2_over_t1` alongside final KL would settle whether the null survives a configuration in which the gradient term is actually load-bearing. On the reported timing (21.6 s/sequence), a 5 x 5 sweep at n = 50 costs under two GPU-hours.

To be fair to the candidate: the thesis's conclusion very likely survives this, because the linearization experiment measures the surrogate-truth correlation *directly at the proposal-logit level*, where temperature is a monotone rescaling and therefore irrelevant to a Spearman correlation. That result (|rho| < 0.06 on all five energies, n = 400,000 each) is the real evidence, and so is the final-position theorem. The problem is one of argumentative architecture: the thesis presents the sampling ablation as the primary result and the linearization as its explanation, when in fact the linearization is the primary result and the sampling ablation is, at this configuration, close to vacuous. Reordering the argument would make the thesis both more honest and considerably stronger.

### 3.2 Severe: zero exact recovery is never reported in the results chapter

From `results/revision/numbers.json`, across all 145 configurations at n = 200 each, `accuracy` is exactly 0.0 in 139 configurations, and is 0.5% or 1.0% (one or two sequences) in the remaining six. In roughly 29,000 recovery attempts the Langevin samplers essentially never restore the corrupted token.

This appears in the thesis exactly once, as the row "Flagship Langevin | 0.0 | -- | approximately 6.5" inside Table 11 in the diffusion section on p. 78, and once in the final-position table. It is absent from Section 5.5, from Table 2, from the abstract, and from the conclusion's "what was found".

This matters for two reasons. First, it is the most rhetorically powerful negative result in the study and it is being under-sold. Second, and more importantly, it undermines the primary metric: the reader is asked to interpret a fall in mean KL from 9.14 to 6.4 as meaningful sampler behaviour, in a setting where not one token is ever recovered. The thesis owes the reader an explicit reconciliation of those two facts. Report exact-match and top-5 alongside KL for the whole grid, as is already done for the diffusion and final-position experiments.

### 3.3 Serious: the Metropolis-Hastings correction is applied asymmetrically across compared arms

In `core/dls.py` and `core/cls.py`, the reverse-proposal term is computed exactly only when `method == "policy"`. For the random and norm-preserved arms the code sets `log_q_ratio = 0.0`, commented "Random-direction baselines are symmetric random walks: q(x|x') = q(x'|x), so the proposal ratio cancels."

That justification does not hold. For CLS the proposal mean is `0.5 * (s + 0.5*eps*g + proj_V(s + 0.5*eps*g))`, which involves a projection and a drift; the reverse kernel is not the forward kernel even for a random `g`, and the thesis's own headline finding is that this term reaches -1325 nats. For DLS the distance term is symmetric in the numerator, but the softmax normalisers differ between forward and backward, and the random direction is redrawn, so the ratio is only approximately zero.

The consequence is that in every MH-enabled comparison, **the policy arm is charged an exactly computed reversibility penalty from which its comparators are exempted.** This includes Table 2 (`tab:fallacy`, MH on, gn off) and the flagship paired test of Section 5.5. It is a plausible partial explanation for why the random arms are numerically better in three of four cells, and for the one Llama cell where the gradient is "reliably worse". This must be disclosed in Section 4.3 at minimum, and ideally either fixed (compute the exact ratio for all arms) or bounded (report the distribution of the policy arm's `log_q_ratio` and show that removing it does not change the ordering). Note that the code comments show the candidate was aware of the detailed-balance subtlety in the policy arm and fixed an earlier bug there, so this is an oversight of symmetry, not of understanding.

### 3.4 Serious: "the gradient" is one Jacobian slice, and its deficiency is partly a design choice

The gradient tested throughout is d log p(x) / d e_i, taken with respect to the input embedding at the masked position (`core/base_sampler.py:36-45`). The thesis correctly identifies that this is structurally blind to the "self" term, quantifies the blindness (mean |self| = 15.0 nats against mean |future| = 24.2 nats), and proves the extreme case at the final position.

But the natural adaptation of the two cited samplers to a language-model energy does not have this property. In Grathwohl et al. (2021) and Zhang et al. (2022), the proposal is built from the gradient of the energy with respect to the **one-hot (or simplex-relaxed) input**, not with respect to an embedding. For an autoregressive LM, that gradient's v-th coordinate contains log p(v | x_{<i}) directly, because the token index enters the energy through the output softmax as well as through the embedding lookup. In other words, the standard formulation of the very method the thesis implements would include exactly the self term whose absence the thesis diagnoses, and it is exactly the signal that the thesis's own successful baselines (conditional argmax, top-k rescore, Gibbs, independence MH) exploit.

Section 3.2 excludes only straight-through and Gumbel-softmax relaxations as "altering the model's forward computation". That exclusion does not cover the one-hot/simplex gradient, which alters nothing about the forward computation and leaves the frozen likelihood untouched. The alternative is never named.

This does not invalidate the thesis: it studies a real, published family of methods (MuCoLa and COLD do operate in embedding space), and it scopes its claim to the input-embedding gradient nearly everywhere. But it does mean the reader can reasonably ask whether the diagnosis is "autoregressive likelihoods do not admit gradient guidance" or "differentiating with respect to the input embedding discards the informative half of the energy, and that is a modelling choice". The thesis should name the alternative, say why it was not chosen, and ideally run it: the surrogate becomes g_hat(v) = log p(v | x_{<i}) + (gradient term), and the linearization diagnostic can be re-run at negligible cost. The likely outcome is that it correlates well, which would sharpen the thesis's message considerably: the usable direction is available, but it lives on the output side, not the input side. That is a more interesting and more defensible thesis than the one currently stated, and the data to support it is already in hand.

### 3.5 Moderate: the primary metric is a last-iterate statistic for a sampling method

Every headline number is the KL at the final step of the chain. Inspecting the per-sample trajectory in `gpt2-large.dls.policy.mh.gn.free.s50.csv`, sample 0 passes through KL 0.996 at step 4, sits at **0.346** for steps 5 to 10, then drifts to 3.75, back to 0.58, and terminates at **8.09**. The chain visits near-perfect states and the reported statistic discards them.

For an MCMC method this is the wrong summary. Report at least: the chain-averaged KL over the second half of the schedule, the minimum over the trajectory, and the fraction of steps below some threshold. This is pure re-analysis on data already on disk. Note also what the trajectory says substantively: an MH-corrected chain on the exact sequence likelihood accepted a move from KL 0.35 to KL 8.09. That is direct evidence that the *energy itself*, not only its gradient, disagrees with the metric, which is a more nuanced picture than "the energy is fine and only the gradient is broken", and deserves discussion.

### 3.6 Moderate: statistical power and multiplicity

- With n = 200 the minimum detectable difference at 80% power is 0.652 nats, against observed differences of 0.008 to 0.171. TOST correctly fails to certify equivalence. The thesis handles this honourably by reporting "no reliable difference". But at 21.6 s/sequence, raising n to 1,000 on the single flagship configuration costs about six GPU-hours and would very likely have converted the central claim from "we could not detect a difference" into "we can certify equivalence at the pre-registered margin". For the claim on which the entire thesis rests, that omission is hard to defend, especially in a study that spent roughly 175 GPU-hours on breadth.
- The equivalence margin (5% of the policy mean KL, that is 0.327 nats) is calibrated to pipeline noise, which is defensible, but is never connected to any notion of *practical* significance. What does 0.33 nats of next-token KL mean for text quality? The thesis has an external judge and a set of qualitative examples that could have been used to anchor this and does not use them.
- No correction for multiple comparisons. The one statistically significant contrast reported (Llama-3, nomh/nogn, CI [0.45, 2.09], p = 0.015) is selected from a family of dozens of paired comparisons across 145 configurations. At alpha = 0.05 uncorrected, one expects a few such hits by chance. The thesis calls this contrast "statistically clear" and builds an interpretation on it (a mis-specified surrogate can point away from good moves). That interpretation may well be right, but it needs either a corrected p-value or a much more hedged presentation.

### 3.7 Moderate: confounds in the GFlowNet and diffusion comparisons

- **Domain.** The base model is SFT'd on ROCStories, the GFlowNets are trained on ROCStories, and the unification experiment evaluates them all on WikiText-2. If the GFlowNet objective reshaped the energy in a domain-specific way, the null on WikiText is partly expected. Section 4.1 handles the domain-shift objection for the *gradient* null (in-domain diagnostics reproduce it), but not for the GFlowNet unification result, which is the load-bearing GFlowNet experiment. The divergence table (31 to 57 nats) is offered as evidence that tuning moved the energy, but the corpus on which that divergence was measured is not stated in the thesis.
- **The diffusion control changes more than the objective.** SEDD differs from GPT-2 Large in training objective, model scale, training corpus, tokenizer usage pattern, *and* conditioning direction. Section 5.13 addresses the last point explicitly and well (the bidirectional revision query is the operative difference), but the section, the discussion, and the conclusion then still attribute the effect to "the training objective". The candidate cannot have it both ways. The cheap clean control is missing and would take an afternoon: a bidirectional masked LM (BERT/RoBERTa) is not score-trained but *is* bidirectional; if it also supplies a usable proposal, the causal variable is bidirectional conditioning, not score matching. Wang and Cho (2019) is the obvious reference and the obvious baseline, and it is neither cited nor run.
- **The "0 to 39%" framing overstates.** The comparator for 39% is the DLS arm at 0%. On the same sequence set, the thesis's own gradient-free top-k rescore reaches 33% (Table 11). The honest headline is "39% against 33% for the best gradient-free autoregressive baseline", a modest gain. The abstract's "raises exact recovery from 0 to 39%" invites the reader to attribute the whole 39 points to the diffusion score. Fix the abstract and the conclusion.

### 3.8 Moderate: the constrained-generation extension (RQ4) is the weakest chapter

- n = 150 per arm, aggregate target-hit fractions only, no per-generation labels stored, therefore no confidence intervals and no paired tests anywhere. Every number in Tables 7 and 8 is an unqualified point estimate. Section 4.6 discloses this, which is right, but disclosing an evidential limitation does not remove it, and RQ4 is one of four research questions.
- Table 8 reports raw steering gains and then the text immediately declares them untrustworthy. If they are untrustworthy, move them to the appendix and lead with the contrast.
- The +27.3 and +36.7 point contrasts on the MuCoLa arm are then themselves discounted as off-manifold artifacts. After both discountings, RQ4's empirical content reduces to two near-zero contrasts on the DLS arm at n = 150. That is thin for a headline research question.
- One attribute only (sentiment), and the constraint classifier is trained on the base model's own representations while the "judge" is also a frozen-model sentiment judge, so the independence of the evaluation is limited. The proposal's toxicity arm was dropped for defensible reasons, but the effect is that RQ4 rests on a single, weakly instrumented attribute.

### 3.9 Related work: real gaps for a thesis with this specific claim

47 references is thin for an IMS MSc thesis, and the omissions are not random. A thesis whose central empirical claim is "the highest-likelihood text is degenerate, so the sampling target is undesirable" must engage with the machine-translation literature that established this most rigorously:

- **Stahlberg and Byrne (2019)**, "On NMT search errors and model errors", the empty-string-is-the-mode result. This is the likelihood trap in its sharpest published form and is not cited.
- **Eikema and Aziz (2020)**, "Is MAP decoding all you need?", which makes exactly the argument that the mode of a well-trained model is not where the quality is.
- **Meister et al.** on typicality and uniform information density, which gives a principled account of why high-probability text is bad.

Further, given that the thesis's most successful baseline is a Gibbs sampler over a frozen LM:

- **Wang and Cho (2019)**, "BERT has a mouth and it must speak", the direct antecedent of a Gibbs sampler on a frozen model, is absent.
- **Miao et al. (2019)**, CGMH, Metropolis-Hastings for constrained sentence generation, is absent and is arguably the closest precedent for "gradient-free MH on a frozen LM works".
- **Deng et al. (2020)** residual EBMs for text, and **Goyal et al. (2022)** on implicit energy networks in masked LMs, are the two most relevant "LM as EBM" papers and are absent.
- **Han et al. (2023)** SSD-LM, the semi-autoregressive simplex diffusion model, is directly relevant to the diffusion section.
- NeuroLogic and DExperts were cited in the proposal and dropped in the thesis without replacement.

Eight bibliography entries use `and others` in the author field, producing truncated author lists in the rendered bibliography (`ouyang2022training`, `vaswani2017attention`, `dathathri2020plug`, `krause2021gedi`, `madan2023learning`, `mostafazadeh2016corpus`, `dubey2024llama`). Complete the author lists.

### 3.10 Presentation and format

- **Length.** 125 pages total (body pp. 9-91, bibliography to 98, appendix 99-125). An IMS MSc thesis is typically 60-80 pages. With `baselinestretch 1.3` and 3 cm margins the text is airy, but this is still long, and the length is not distributed where the evidence is.
- **Figure economy is poor.** Seven figures in a ninety-page body, three of which (Figures 1, 2, 3) are near-identical full-page trajectory plots occupying three consecutive pages and conveying, between them, one fact.
- **There is no figure for the central result.** The claim on which the thesis rests is delivered as a four-row table of point estimates (Table 2). The obvious and missing figure is a forest plot of paired policy-minus-random differences with bootstrap CIs, one row per configuration, across all five energies, with the equivalence margin drawn as a band. That single figure would do more work than Figures 1 to 3 combined and would also address the multiplicity concern visually.
- **Figures 1 to 3 do not visibly support the claim they are cited for.** The text says the drop moves "from around step 40 to around step 85" when the schedule is extended, and calls this the signature of quenching. In both figures the visible drop in the dashed curves occurs at the *final* step (approximately 49 and approximately 99), not at 80% and 85% of the way through. Either annotate the drop points on the figures or give the per-step numbers; as it stands the figure and the text disagree.
- **The full grid is never shown.** The reader is told there are 145 configurations and shown eight rows. Given that the "no reliable difference" claim is a claim about the whole grid, the whole grid (or a compact heatmap of it) belongs in the appendix.
- Thirteen `h` float specifiers were silently promoted to `ht`; seven overfull boxes remain. Both minor.
- **"Background Work"** is a non-standard section heading; use "Background".
- Minor factual slips to check: "49 GB A6000-class accelerators" (the A6000 is 48 GB), and "PyTorch 2.12 with CUDA 13.0" should be verified against the actual environment file.
- No German *Kurzfassung*: acceptable for an English thesis. Erklärung is present, on the first page, in the current KI-Tools wording, with a translation footnote. Correct.

### 3.11 The LaTeX sources and the AI-tools declaration

The submitted `.tex` files contain several hundred lines of revision-provenance comments of the form `% REMOVED (Phase 8, evaluation section 5 item 2: "...")` followed by verbatim quotations of an external evaluator's critique and the superseded wording. `abstract.tex` alone carries roughly sixty lines of commented-out prior versions.

Two observations, in order of importance.

1. **Strip these before submission.** A submitted artefact should not contain its own revision archaeology. Keep the history in version control, where it belongs. The `% SOURCE:` annotations linking numbers to result files are a different matter and are a genuine strength; keep those, or better, move them into a machine-checkable artifact map.
2. The Appendix A.7 declaration is present, specific, and states that Claude was used through an agentic interface to apply an author-specified edit list, transcribe verified numbers, and copy-edit. That is the right disclosure and it is taken at face value. Note only that the density and character of the source comments (edits keyed to an external evaluator's numbered checklist, with the tool preserving both the critique and the replaced text) means the process was more iterative than "copy-editing" conveys, and that an examiner reading the sources will notice. The candidate is advised to (a) clean the sources, and (b) be prepared to defend the mathematics of Section 2.4, the acceptance decomposition, and the final-position derivation entirely unaided, because those are the passages a committee will probe.

### 3.12 Smaller points

- Section 5.5's paired test is run on the gn-*enabled* configuration; Table 2, the unconfounded gn-disabled comparison, gets point estimates only. The formal statistics should be on the configuration the thesis argues is the clean one, or on both.
- Section 5.5 argues that the 0.670 gap on the length-normalised GFlowNet "exceeds the across-seed noise scale of about 0.18 nats, so it is not run-to-run variation". Comparing a paired difference against an across-seed standard deviation of a mean is not a valid test. Run the paired test.
- Table 12 lists "Top-5 (%) 34.5" for arms with 0.0% exact recovery; explain what the top-5 column measures for a sampler that returns a single token.
- Section 5.4 reports "about two cells" for the corrected continuous sampler where the prior canonical trace gave exactly one; the source comment records the discrepancy between two runs of the same configuration. The thesis should say in the text that these are two independently seeded trace runs, not silently in a comment.
- The masked-recovery task operates at the subword level, and the appendix examples show masks landing inside words ("star**moments**" for *starfish*, "@-@ **ede**s" for *1990s*). This is a legitimate design, but the fraction of masks that are word-internal subwords should be reported, since it bears directly on how attainable exact recovery is.
- "Section" is used for what the IMS criteria call chapters. This follows the official article-class template and is fine, but state it once.

---

## 4. Section-by-Section Feedback

**Abstract (p. 7).** One page, well compressed, and correctly scoped. Two fixes: (i) state that exact recovery was 0% across the grid, which is the strongest sentence available and is currently missing; (ii) change "raises exact recovery from 0 to 39%" to give the gradient-free comparator (33%), otherwise the reader mis-attributes the gain.

**1 Introduction (pp. 9-15).** Excellent. The motivation from the left-to-right factorisation to global constraints to the additive energy is clean and needs no revision. The decision to state the negative outcome before posing the RQs is the right call and is well justified. The RQ-to-proposal mapping in Section 1.4 is a model of how to document a pivot. The five contributions are appropriately compact. Only change: contribution 1 should say "the input-embedding gradient's direction", matching the scoping used everywhere else.

**2 Background (pp. 15-26).** The strongest expository chapter. Section 2.1's distinction between what teacher forcing shapes and what it does not is the conceptual spine of the whole thesis and is stated once, precisely, in the right place. Section 2.4's separation of the piecewise-constant projected energy from the differentiable pathway, with the explicit statement that every non-Lipschitz claim attaches to the latter, is exactly right and rare. Two additions needed: (i) a paragraph on the alternative of taking the gradient with respect to the one-hot/simplex input, with a stated reason for not doing so (Section 3.4 above); (ii) the derivation in Section 2.4 should note that the proposal's *temperature* rescales the gradient-alignment and distance terms jointly, and that the resulting proposal sharpness is an empirical quantity to be measured, not assumed.

**3 Related Work (pp. 26-33).** Well organised around a narrative rather than a catalogue, and Section 3.2's paragraph isolating the gradient-free samplers (Mix-and-Match, twisted SMC) as anticipating the diagnosis is genuinely insightful. The gap statement in Section 3.6 is precise. Fill the gaps listed in Section 3.9 above, particularly Stahlberg and Byrne, Eikema and Aziz, and Wang and Cho: the first two are load-bearing for the likelihood trap and the third for the Gibbs baseline. Expand the section by roughly a page and a half.

**4 Methodology (pp. 33-45).** Thorough and well motivated; the practice of giving the reason for each design choice at the point of choice is good scientific writing. Four additions: (i) state the temperature and report the resulting proposal entropy, with `t2_over_t1`; (ii) disclose the asymmetric MH treatment in Section 4.3; (iii) state which corpus the base-versus-tuned divergence measurement used; (iv) in Section 4.4, add exact-match and top-5 as secondary metrics for the whole grid, and say plainly that exact match is 0 everywhere. Section 4.8 (reproducibility) is very good and should stay as is.

**5 Results (pp. 46-81).** The best-organised results chapter one could expect from an MSc candidate, and also the one whose emphasis most needs rebalancing.

- *5.1* is fine, and the calibrated-versus-guided motion distinction is well made. Extend the oracle sweep to temperature.
- *5.2* Figures 1 to 3 should be merged into one figure with three columns; the drop points must be annotated or tabulated.
- *5.3* is excellent and needs nothing beyond the asymmetry disclosure.
- *5.4* is appropriately compressed to the main geometric conclusion.
- *5.5* needs the largest revision. Lead with the proposal-entropy diagnostic, state the 0% recovery, add the forest-plot figure, run the paired test on the gn-off configuration, and either correct or hedge the Llama contrast.
- *5.6* is the strongest section in the chapter and should be promoted to the primary evidence. Say explicitly that this measurement is invariant to the temperature and step-size choices that limit Section 5.5, because that is exactly why it carries the argument.
- *5.7* is convincing; add the MT citations.
- *5.8* is clean and correctly separates scale from anisotropy.
- *5.9* the cross-model table is good.
- *5.10* the taxonomy is honestly labelled exploratory; the unification experiment needs the domain caveat.
- *5.11* needs the treatment in Section 3.8 above.
- *5.12* is excellent. Do not change it.
- *5.13* is a strong chapter that over-attributes. Add the bidirectional-MLM control or explicitly bound the attribution to "bidirectional conditioning and/or score training, which this design cannot separate".

**6 Discussion (pp. 81-90).** Section 6.1 answers all four RQs against specific evidence, which the criteria require and which many theses fail. Section 6.2's account of the mechanism is well argued and correctly hedged ("support the interpretation" rather than "because"). Section 6.4 is unusually candid and is a credit to the candidate. Add to Section 6.4: the proposal-sharpness limitation, the MH asymmetry, the input-embedding-versus-one-hot choice, and the last-iterate metric. A limitations section that names four real threats is worth more than one that names ten cosmetic ones.

**7 Conclusion (pp. 90-91).** The four-heading structure (tested / found / uncertain / implication) is exactly right and should be kept. Add the 0% recovery to "what was found" and fix the "0 to 39%" framing.

**Appendix (pp. 99-125).** A.1 through A.4 are appropriate. A.3 (computational cost) is a nice touch and makes the practical argument concrete. A.4's decision to make the load-bearing figure exact in the full space and demote the PCA projection to illustration, with the 3.3% explained variance stated in the caption, is exemplary self-criticism. A.6's selection policy is the best thing in the appendix. A.7 is adequate but see Section 3.11. Add the full 145-row grid.

---

## 5. Assessment of the Proposal

Reviewed as a separate deliverable against the IMS proposal criteria.

**Strengths.** The motivation is well constructed and the "research gap" paragraph (that published methods omit MH and anneal improperly, so they are noisy gradient descent rather than sampling) is a sharp, testable, correctly identified gap: it is the gap the thesis then addresses. The taxonomy of CTG by stage of intervention and constraint granularity is a sensible organising frame. The Methods section specifies both samplers concretely, with the transition kernel written out, plus MH, Gibbs, and an AR baseline. The "Competitive Landscape: Diffusion Models" subsection turned out to be prescient. The Risk Analysis and Contingency section is genuinely strong and, unusually, names in advance the four diagnostic instruments the thesis actually used.

**Weaknesses.**

- **Length: 12 pages against the 6-10 page specification.** Over by 20%.
- **No explicit "Statement and Contributions" section.** The criteria ask for one; the proposal has "Goal and research questions", which covers the goal but never states the anticipated contributions as such.
- **RQ structure is convoluted.** Three top-level questions with sub-questions RQ1a, RQ1b, RQ2a, RQ2b, RQ3a, no RQ3b, and RQ1 and RQ2 having both a top-level formulation and sub-questions that do not exhaust them. Flatten to four numbered questions.
- **Evaluation metrics are under-committed.** Six metric families are listed with "we will select the most relevant subset". A proposal should nominate a primary metric and justify it; four of the six were subsequently dropped, which the thesis had to account for in Section 6.4. Committing earlier would have avoided that.
- **Datasets are over-committed.** CommonGen, ROCStories, IMDb, RealToxicityPrompts are named; three were not used. WikiText-2, which carried the main grid, is not mentioned.
- **Timeline is thin.** Six month-blocks with bulleted activities, no deliverable per milestone, no explicit dependencies, and "Month 5: Optimization and Framework Extension" is not a milestone.
- **Citation hygiene.** Keys are inconsistent between auto-generated DOIs (`10.5555/3104482.3104568`, `10.5555/3600270.3600963`) and readable keys; `ref.bib` contains 28 entries of which 17 are cited. Equations are referenced informally ("Equation (2)").
- **Source file quality.** The `.tex` contains roughly 160 lines of commented-out draft prose, several of them alternative versions of the same paragraph. Invisible in the PDF, but clean it up.
- **The amendment.** The supervisor note discloses that the Risk Analysis section and the errata were added after review, so the disclosure is correct and the document is above board. Note only that the amended proposal now names the thesis's actual diagnostic instruments with a specificity that a genuinely ex-ante document would rarely achieve, and that a reader who has both documents will see this. Since it is disclosed, it is not a problem of integrity; it is a problem of the proposal no longer serving as evidence of foresight. Consider dating the amendment inside the document itself.

**Proposal grade, if assessed separately: 2.0.**

---

## 6. Estimated Grade and Justification

### Grade: 1.7 (gut / good, upper band)

Mapped to the IMS criteria:

| Criterion | Assessment | Band |
|---|---|---|
| **Hypotheses and goals** | The assumption under test is isolated with unusual precision; four RQs, explicitly mapped from the proposal, explicitly answered in the discussion. | 1.0-1.3 |
| **Coherence of research steps** | Systematic and self-motivating; each experiment answers a question the previous one raised. Two samplers chosen precisely so that a shared finding is a finding about the energy. | 1.3 |
| **Completeness** | All four RQs answered; dropped proposal deliverables individually accounted for. Related work is competently reviewed but has real gaps (Stahlberg and Byrne, Eikema and Aziz, Wang and Cho, Miao et al., Deng et al.) that a thesis with this specific claim should not have. No human evaluation anywhere. | 2.0 |
| **Correctness** | This is where the marks are lost. The implementations are careful and verified, the numbers are accurate and traceable, and the analytic results are correct. But the central sampling ablation runs at a configuration whose proposal is numerically uniform; the MH correction is applied asymmetrically across the compared arms; the primary statistic is a last-iterate value for an MCMC method; the equivalence claim is underpowered when the fix cost hours; and the tested gradient is a Jacobian slice whose deficiency is partly a design choice. The conclusion survives, because other evidence carries it, but the flagship experiment does not establish what it is presented as establishing. | 2.3-2.7 |
| **Originality** | High. The direction-versus-magnitude ablation, the acceptance decomposition by boundary crossing, the final-position zero-gradient result, and running rather than proposing the positive control are all genuinely the candidate's own contributions. | 1.0-1.3 |
| **Sustainability** | Outstanding. Seeds, manifests, an equivalence test suite, a resumable queue, an artifact map, per-number provenance, and an unfiltered qualitative appendix with a stated selection policy. Traceability was verified and it holds. | 1.0 |
| **Meaningful structure** | Sound, but unbalanced: 125 pages, three near-duplicate full-page figures, a ninety-page body with seven figures and no visualisation of the central result. | 2.0 |
| **Coherence / golden thread** | Very strong. Motivation, mechanism, and conclusion connect without a break, and the pivot is handled openly. | 1.3 |
| **Quality of writing** | Fluent, precise, consistent terminology, disciplined scoping language, essentially free of grammatical error. | 1.0-1.3 |
| **Quality of presentation** | Figures are clean and professional where they exist, but too few, poorly economised, and in one case (Figures 1 to 3) not visibly supporting the claim they illustrate. Several tables give point estimates without uncertainty. | 2.3 |
| **Metalevel** | Introduction, bigger picture, limitations, and implications are all handled well; the limitations section is unusually candid, and the implications are actionable. | 1.3 |
| **References** | Internally consistent, all cited, all present, correctly formatted. Thin at 47 entries with substantive topical gaps and eight truncated author lists. | 2.3 |

**Why 1.7 and not 1.3.** The four issues in Sections 3.1 to 3.4 are scientific, not cosmetic. A thesis whose headline claim is "following the gradient is no better than following noise" must demonstrate that the experiment could have detected the difference had it existed, and this one cannot: at 10.8248 nats of proposal entropy against a 10.8249-nat ceiling, no gradient however informative could have shown up. That the conclusion is nonetheless probably correct is fortunate, not earned by that experiment. Combined with the asymmetric MH treatment in exactly the comparison at issue, the buried 0% recovery, and the last-iterate metric, this is a thesis whose argument is right for reasons partly other than the ones it gives.

**Why 1.7 and not 2.3.** The candidate has done a great deal more than the null. The linearization measurement is independent of every configuration concern raised above and is decisive on its own terms. The final-position result is analytically exact. The gradient-free baselines correctly narrow the claim from the energy to the gradient, and the candidate saw that and acted on it. The diffusion positive control is real work that most authors skip. The scoping discipline, the limitations section, the unfiltered examples, and the reproducibility infrastructure all reflect genuine scientific maturity. The writing is good. And a diagnostic negative result, delivered with a mechanism and a falsifiable prediction that was then tested, is a more valuable contribution than most positive results at this level.

### What would move the grade

- **To 1.3, without new experiments:** report proposal entropy and `t2_over_t1`; state the 0% recovery in the abstract, Section 5.5, and the conclusion; disclose the MH asymmetry and bound its effect; re-report the KL as a chain statistic rather than a last iterate; reorder Section 5 so that the linearization result is the primary evidence and the sampling ablation is its corroboration; add the forest-plot figure; fill the four related-work gaps. All of this is re-analysis and rewriting on data already on disk.
- **To 1.0-1.3, with about a day of compute:** a (epsilon, temperature) sweep showing the null survives at configurations where the gradient term is load-bearing; n = 1,000 on the flagship configuration to certify equivalence under the pre-registered margin; the one-hot/simplex-gradient variant of the linearization diagnostic; a bidirectional-MLM arm to separate score training from bidirectional conditioning in the diffusion control.
- **Down to 2.3:** if, at the defence, the candidate cannot account for the uniform-proposal measurement or the MH asymmetry, since both go to whether the central experiment measures what it claims.

### Questions the committee should ask at the defence

1. Your measured proposal entropy is 10.8248 nats against a ceiling of 10.8249. At that configuration, could any gradient have produced a measurable difference? What does your `t2_over_t1` diagnostic show?
2. Exact recovery is 0.0% in 139 of 145 configurations. Why is that not in the abstract, and what does a fall in mean KL from 9.14 to 6.4 mean in a setting where no token is ever recovered?
3. Your code sets the MH proposal ratio to zero for the random arms and computes it exactly for the policy arm. Why is that valid, given that the same term reaches -1325 nats in your own CLS measurement?
4. Grathwohl and Zhang take the gradient with respect to the one-hot input, whose v-th coordinate contains log p(v | x_{<i}). You take it with respect to the input embedding, which discards exactly that term. Is your negative result about autoregressive likelihoods or about that choice?
5. SEDD differs from GPT-2 in objective, scale, corpus, and conditioning direction. What would a bidirectional masked LM, which is not score-trained, do in your hybrid chain?
6. Sample 0 of your flagship run reaches KL 0.346 at step 5 and terminates at 8.09. Why report the last iterate?

---

## 7. Overall

A thoughtful, honest, technically substantial thesis on a question worth asking, undermined in its central experiment by a configuration issue the candidate's own instrumentation was built to catch and then never reported. The diagnosis is very likely right and the work deserves to be read; the argument for it needs to be rebuilt around the evidence that actually supports it.

**Thesis: 1.7. Proposal (separately): 2.0.**
