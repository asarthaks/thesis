I've read the full text. Let me visually inspect a few key figures before writing the evaluation.I read the full submitted PDF (117 pages), including all appendices, and inspected the key figures visually. Here is the examiner's report.

---

## Executive Summary

This is a mature, unusually rigorous Master's thesis. It set out to build controllable generation on faithful Langevin dynamics, found the foundational premise broken, and converted that failure into a genuinely diagnostic contribution: the input-embedding gradient of a frozen autoregressive likelihood carries no usable proposal signal on discrete text, and the thesis explains why at three levels (statistical, geometric, and analytic), cross-checks the explanation across five energy functions and two architectures, shows the failure survives amortization, and then runs a diffusion positive control that isolates the training objective as the cause. The methodological honesty (equivalence margins, power analysis, seeded and unfiltered qualitative examples, a bitwise-equivalence verification suite, an artifact map from every number to its source file) is well above what I normally see at MSc level and would not embarrass a conference submission.

It is not flawless. The headline sampling-level comparison is underpowered by the thesis's own analysis, several figure-to-text numeric inconsistencies survive in the submitted version, the corpus bookkeeping (ROCStories vs. WikiText-2 per experiment) is not laid out in one place, and the constrained-generation experiment is noticeably less rigorous than the rest. These are fixable before the defense and mostly cosmetic relative to the strength of the core argument.

---

## Strengths

- The central claim is defended at multiple independent levels: the sampler-level null (5.5), the linearization diagnostic at n = 400,000 candidate pairs with |ρ| < 0.06 on all five energies (5.6), the exact zero-gradient theorem at the final position realized inside the live sampler (5.12), and gradient-free baselines that cleanly separate "the energy is fine" from "its gradient is not" (5.5.2). This triangulation is the thesis's real strength; even if any single leg were questioned, the argument stands.
- The diffusion positive control (5.13) elevates the work from a negative result to a causal diagnosis. The hybrid sampler in 5.13.3 is the best experiment in the thesis: same energy, same Metropolis chain, same sequences, only the proposal direction swapped, recovery moves from 0% to 39%. That is textbook experimental isolation.
- Statistical practice is exemplary for the level: pre-registered equivalence margin calibrated to measured seed noise (4.4, 4.8), paired bootstrap CIs, Wilcoxon tests, an explicit power statement, and the honest admission that the equivalence test "stops just short" of certifying equivalence. The thesis consistently argues from the absence of a gap rather than over-reading small differences.
- Reproducibility and sustainability are outstanding: deterministic per-index corruption seeds, the bitwise-equivalence suite, the manifest-driven resumable job queue, per-number source-file annotations in the LaTeX, and a stated selection policy for every qualitative example (A.6), including kept failures. Table 18's inclusion of cases where every method fails (sequence 98, 199) is exactly the right instinct.
- The scoping discipline is excellent. The "gradient fallacy" shorthand is explicitly bounded (6.2), the Llama-3 control is never placed on a shared numerical axis, the GFlowNet claim is bounded to "the variants tested," the diffusion results are labeled pilots, and 6.4 distinguishes refuting the gradient premise from refuting the training-free premise. This is how negative results should be written.
- Related work is a narrative rather than a catalogue, and Section 3.2's isolation of the gradient-free branch (Mix-and-Match, twisted SMC) as an anticipation of the diagnosis is a genuinely insightful piece of literature synthesis.
- The three GFlowNet failure modes are correctly framed as findings about the reward, not implementation excuses, and the "did the energy actually move" check in Table 5 preempts the obvious triviality objection. Good scientific instinct.
- The writing is precise, the golden thread is unbroken from motivation to conclusion, and every RQ raised in 1.4 is explicitly answered in 6.1.

---

## Critical Issues and Weaknesses

**1. Figure-to-text numeric inconsistencies (must fix before submission).** Three cases, in increasing severity:

- Figure 9 shows acceptance rates of 0.064 (stayed in cell) and 0.074 (crossed a boundary), while Section 5.3 reports 0.63% within-cell and 8.56% boundary-crossing with the correction enabled. As printed, the figure and the text disagree by roughly an order of magnitude on the within-cell rate, and the ordering pattern differs. If the figure aggregates a different configuration set, the caption must say so; otherwise one of the two is wrong.
- Table 12 reports identical final KL for CLS policy with MH and without MH (8.083 in both rows, on all four models). Given that Section 5.3 shows these two regimes behave completely differently (frozen vs. wandering), identical values to three decimals are implausible and look like a results-file join error or a duplicated read. This needs to be verified; it currently undermines confidence in the grid table.
- Figure 12's annotation reads slope = −0.11 nats/token, r = −0.00, while the text of 5.7 states the headline slope as −1.12 nats/token, and 5.10's entire length-collapse explanation leans on the 1.12 figure. The censoring caveat explains why the GPT-2 Large panel would be null, but then the provenance of −1.12 is nowhere visible: it is not the GPT-2 slope shown, not the length-normalized variant's −0.505, and not the censored −2.361. Either state exactly which generations produce −1.12, or the digit transposition (0.11 vs 1.12) is a real error propagating into a load-bearing argument. This is the single most important correction.

**2. The headline comparison is underpowered, by the thesis's own account.** With n = 200 the CI on the paired difference ([−0.285, +0.619]) is wider than the equivalence margin (0.327), and the experiment could only detect an effect of 0.652 at 80% power. The thesis handles this with commendable honesty, and the 400k-pair linearization result carries the burden instead. But an examiner must note that raising n to 500 or 1000 sequences was computationally feasible on the stated hardware (Appendix A.3 shows ~22 s/sequence) and would have let the equivalence test actually close. As it stands, the sampler-level evidence is "bounded, not certified," and the thesis is rescued by its diagnostics rather than its primary grid.

**3. Corpus bookkeeping is muddled.** Section 4.1 states the corpus is ROCStories, chosen partly for consistency with Hu et al. Yet the reference-baseline set, the linearization "unified corpus," and the seeded qualitative examples are WikiText-2 validation (Table 8, A.6.2), with "a second corpus of narrative text" mentioned in passing in 5.13.1. The reader has to reconstruct which experiment ran on which data, and the domain question is not innocent: the energy model is SFT'd on ROCStories but several core diagnostics are run on out-of-domain WikiText-2 text. A single table mapping experiment to corpus, and a sentence addressing whether the domain shift could inflate the null, are needed.

**4. The constrained-generation experiment (5.11) is the weakest section.** Table 6 reports steering gains in whole percentage points with no n, no CIs, and no significance analysis; the decisive paired contrasts (+27.3, +36.7 on the continuous baseline) appear only in prose, untabulated and without intervals. Given the exceptional statistical hygiene everywhere else, this section reads as rushed. Since RQ4 is one of four headline questions, it deserves the same treatment as 5.5.

**5. Minor internal count inconsistency.** Section 4.7 announces "Four additional diagnostic experiments" and then describes five (linearization, acceptance, trajectory, likelihood-trap, anisotropy); the Conclusion says "five diagnostic experiments." Trivial, but it is the kind of thing an examiner notices.

**6. Deliverables relative to the proposal.** The proposal promised sentiment and toxicity control on IMDb and RealToxicityPrompts, with MAUVE, Self-BLEU, and LLM-judge evaluation. Toxicity, MAUVE, and the diversity suite were dropped. Section 6.4's paragraph on the title-content relationship justifies the pivot well, and a diagnostic thesis is a legitimate outcome, but the proposal-to-thesis delta should be acknowledged even more explicitly, ideally with one sentence per dropped deliverable explaining why it became moot. As written, a reader without the proposal cannot see what changed.

**7. Metric circularity is only partially closed.** The external-judge rescoring (5.5.3) covers the gradient-vs-random null, which is the right place to spend it, but the GFlowNet comparisons of Tables 4 and 5 rest entirely on the internal KL metric. One external-judge column on Table 4 would have closed the loop.

**8. Stylistic repetition.** The sentence "the input-embedding gradient of frozen autoregressive sequence likelihood provided no reliable proposal advantage over a norm-matched random direction" appears verbatim at least six times (abstract, 1.2, contributions, 5.5, 6.1, conclusion). I understand the intent, a fixed, carefully scoped formulation that cannot drift, but by the fourth occurrence it reads formulaic. Two or three verbatim uses, with paraphrase elsewhere, would serve the same purpose. A few sentences also run long enough to strain parsing (several exceed 60 words), and phrases like "run rather than proposed" verge on mannered.

**9. Bibliography.** Several entries truncate author lists with "et al." in the reference list itself (Brown et al., Dubey et al., Bengio 2023, Hu 2022). A thesis bibliography should give full author lists or follow one consistent truncation policy from a named style. URLs are present throughout, which is good; venue formatting is otherwise consistent.

---

## Section-by-Section Feedback

**Abstract.** Dense but complete: task, scale, central result, mechanism, both follow-ups. The spelled-out "one hundred and forty-five" and "thirty-nine percent" alongside numerals elsewhere is inconsistent; pick one convention. Consider adding one sentence on practical implication (evaluate, don't differentiate).

**Introduction (Sec. 1).** Very strong. The move from "the factorization is why they work" to "it is also why one approach to controlling them does not" is an excellent opening. 1.2's isolation of the central assumption is the intellectual heart of the thesis and is exactly where it belongs. One improvement: the answer to the thesis question is revealed in 1.2 before the RQs are even posed. That is a defensible choice for a diagnostic thesis, but say so explicitly ("the reader is told the outcome now because the thesis is organized as an explanation, not a reveal").

**Background (Sec. 2).** The best background chapter I have read from a student in some time. Section 2.1's paragraph on what teacher forcing does and does not shape is the conceptual key and is placed early, correctly. The marble analogy in 2.2 is used and then explicitly bounded, which is the right way to use analogies. The two-objects distinction in 2.4 (piecewise-constant projected energy vs. the differentiable pathway) preempts the most likely technical objection to the whole thesis; keep it exactly as is. Minor: equation (7) would benefit from a one-line derivation footnote showing the discarded constant terms.

**Related Work (Sec. 3).** Narrative structure works. 3.2's paragraph on the routinely omitted MH correction, and the observation that "including the correction faithfully often makes these methods stop working," is a sharp and fair reading of the literature. 3.6 states the gap precisely. Missing: a brief mention of straight-through/Gumbel-softmax relaxations as a third adaptation strategy, if only to say why they were out of scope.

**Methodology (Sec. 4).** The order (task, models, samplers, metrics, then diagnostics with their motivating questions) is well judged, and stating that diagnostics were designed post hoc in response to results is honest and correct practice. 4.4's equivalence-margin construction is excellent. Fix the corpus mapping (issue 3), the four-vs-five count (issue 5), and state the n and statistical plan for the constrained extension in 4.6, not just for the main grid.

**Results (Sec. 5).** Organization is logical and each subsection ends by naming the RQ it bears on, which maintains the thread across 40 pages. 5.3's acceptance-ratio decomposition (target +4.60 vs. proposal −1325) is a beautifully clean measurement. 5.5's three-way proposal design with the normalization confound removed is the methodological centerpiece and is explained clearly. 5.6 and Figure 5 are convincing; the self/future decomposition is a genuinely original mechanistic finding. 5.12 is elegant, converting a statistical null into an exact statement. 5.13 is well firewalled as pilots. Required fixes: Figure 9 (issue 1), Table 12 (issue 1), Figure 12 provenance (issue 1), and the rigor gap in 5.11 (issue 4). Smaller point: 5.5.2 says Gibbs at 6.69 is "matching the best gradient-guided Langevin result" near 6.4-6.5; 6.69 is slightly worse, so "comparable to" would be more accurate than "matching."

**Discussion (Sec. 6).** 6.1 answers each RQ with section references, satisfying the structural requirement fully. 6.2's unified mechanism is the payoff and is well argued; the two-routes-no-shared-machinery point is strong. 6.4's limitations are unusually candid, including the "linearization radius names a decay, not a threshold" self-correction. 6.5 is appropriately forward-looking without overreaching. No substantive complaints here.

**Conclusion (Sec. 7).** The what-was-tested / what-was-found / what-remains-uncertain / practical-implication structure is effective and every RQ closes. Good.

**Appendix.** A.2's configuration-count arithmetic, A.3's cost accounting, A.4's refusal to let a 3.3%-variance PCA carry an argument, and A.6's selection policy are all model practice. A.5 is a substantial piece of work in its own right and is correctly demoted to exploratory status.

**Figures generally.** Professionally executed (I inspected Figures 1, 4, and 13 at full resolution): consistent styling, informative captions that state what supports what, symlog axes used where zeros matter. The token strips in Figure 13 are borderline too small in print; consider enlarging or moving to a wider layout.

---

## Estimated Grade and Justification

**Estimated grade: 1.3 (Sehr gut).**

Mapping to the IMS criteria:

*Scientific work.* Hypotheses and goals: clearly defined, explicitly revisited, honestly rescoped (excellent). Coherence: the diagnostic chain from null to mechanism to cross-check to positive control is systematic and unusually tight (excellent). Completeness: all four RQs answered; the pivot from the proposal's control agenda is justified, but toxicity/MAUVE deliverables were dropped and the constrained experiment is thin (very good, not perfect). Correctness: methodology is sound and self-verifying, but the underpowered headline comparison and the three unresolved figure/table inconsistencies prevent a perfect mark here (good to very good). Originality: the direction-vs-magnitude ablation, the self/future decomposition, the final-position theorem, and the hybrid-proposal experiment are original and valuable (excellent). Sustainability: seeds, manifests, equivalence suite, artifact map; among the best I have seen at this level (excellent).

*Written document.* Structure and balance: excellent. Golden thread: excellent, arguably a model example. Quality of writing: very good; precise and controlled, with some over-long sentences and formulaic repetition of the headline claim. Presentation: very good figures and tables, marred by the three numeric inconsistencies. Metalevel: excellent handling of motivation, bigger picture, and future work. References: complete with URLs, minor "et al." truncation inconsistency.

*Structural adherence.* All required thesis components present; every RQ from the introduction is answered in the discussion and conclusion; the AI-tools appendix meets the university's declaration requirements and is unusually transparent.

Why not 1.0: the Figure 12 / −1.12 provenance problem sits under a load-bearing argument, Table 12's duplicated CLS rows and the Figure 9 mismatch are the kind of errors that must not survive into a submitted thesis, the equivalence test does not formally close at n = 200 when a larger n was affordable, and RQ4's experiment lacks the statistical treatment the rest of the thesis sets as its own standard.

Why not lower: the scientific contribution is real, mechanistically grounded, and defended with a level of rigor, honesty, and reproducibility that clearly exceeds the expectations for an MSc thesis, and the identified flaws are localized rather than structural.

If the three numeric inconsistencies are resolved, the −1.12 provenance is documented, and Table 6 gets n and confidence intervals, I would move this to 1.0-1.2 without hesitation. You have five weeks until the August 31 deadline; those fixes are one or two days of work, and I would treat them as mandatory rather than optional.