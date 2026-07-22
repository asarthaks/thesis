Overall verdict

This is a strong, ambitious, and intellectually coherent thesis, but it is also too long and too insistent about its own central argument. It does not contain meaningless filler in the usual sense. Most sections have a legitimate analytical purpose. However, it repeatedly explains the same core conclusion, often with only slightly different framing:

the autoregressive likelihood gradient is not a useful search direction, because the model was not trained to define a navigable global energy landscape.

That idea is important enough to repeat, but the thesis currently repeats it so often that the repetition begins to weaken rather than strengthen the argument.

My honest assessment is:

The central research questions are answered clearly.
The experimental story is unusually well connected.
The thesis has a genuine central contribution rather than merely presenting many experiments.
It can probably be shortened by 15 to 25 pages of main-text material without losing any scientific contribution.
The largest problem is not “blabbering” in the sense of empty writing. It is over-explanation, repeated interpretation, and argumentative over-signposting.
Some later experiments, especially parts of the diffusion and guided-generation extension, make the thesis less focused even though they are individually interesting.

I evaluated the uploaded 142-page thesis, including its introduction, methodology, results structure, explicit research-question answers, discussion, conclusion, figures, tables, and appendix organization.

1. Does it contain blabbering?
Yes, but not in the sense of content-free writing

The author generally knows what every paragraph is supposed to establish. The problem is that many paragraphs do three things when one would be enough:

State the result.
explain why the result matters,
explain how it fits the overall thesis,
explain what question it raises next,
repeat the central mechanism.

This creates a persuasive, essay-like narrative, but a technical thesis should sometimes simply present the evidence and move on.

A typical pattern is:

The experiment shows X.
X matters because it challenges assumption Y.
This connects to the central claim that Z.
It also motivates the next experiment.
Therefore, the same underlying defect explains X.

That structure works occasionally. Here, it becomes a default paragraph pattern.

The writing is rhetorically strong, but sometimes too theatrical

Phrases such as these are effective once:

“the central business of this thesis”
“the gradient fallacy”
“the likelihood trap”
“a correction that breaks the sampler”
“the lowest-energy text is the worst text”
“the strange silver lining in that failure”
“protected from their own objective”
“a closed case against the gradient-guided samplers”
“the sufficiency half of the diagnosis”
“the falsifying experiment”

They give the thesis personality and make the logic memorable. However, the cumulative effect is slightly overstated. It can make the thesis sound as though it is prosecuting a case rather than reporting a carefully bounded empirical study.

The term “gradient fallacy” is particularly risky. The experiments support the claim that the tested autoregressive gradient is not useful in the investigated sampling settings. Calling it a “fallacy” suggests a broader conceptual error across the field. That may be defensible in the discussion, but it is rhetorically stronger than the evidence warrants as a recurring results-section label.

I would retain perhaps two memorable labels, for example “linearization failure” and “likelihood trap”, but use more neutral section headings elsewhere.

2. Is it repeating itself?
Yes, substantially

The repetition appears at several levels.

A. The abstract already contains almost the entire thesis

The abstract is approximately three pages long and includes:

the problem,
model and sampler setup,
configuration count,
central result,
three mechanisms,
GFlowNet outcome,
three explicitly differentiated claims,
diffusion positive control,
final contribution.

It functions more like an extended executive summary than a conventional abstract. A normal master’s thesis abstract could present the same content in approximately 350 to 500 words. The current abstract appears closer to 900 to 1,000 words.

The three-claim paragraph is intellectually precise, but much too detailed for the abstract. The classifier-agreement and trust-region qualifications should remain in the results and discussion.

Potential reduction: 1.5 to 2 pages.

B. The introduction gives away and interprets nearly every later result

The introduction does more than motivate the study. It already tells the reader:

that the central assumption fails,
why it fails,
that GFlowNet does not escape it,
that the energy rather than the search is responsible,
that the results form an investigation,
that two independent routes reach the same defect,
that diffusion repairs the missing signal.

It is acceptable to preview the findings, but this introduction repeatedly states the final interpretation before the evidence is presented.

For example, Section 1.3 says that GFlowNet fails because the difficulty “was never in the search procedure at all but in the energy”. That is already a discussion-level causal conclusion.

Section 1.4 then explains the investigative narrative and defends the way the results are presented. This section is largely unnecessary. A thesis does not need to argue that its narrative organization is “the honest way” to present a diagnostic result. The structure should demonstrate that on its own.

Section 1.5 contains a long contributions paragraph that once again previews almost every major result.

The introduction could be reduced from around seven pages to approximately four or five.

Most removable section: 1.4, “The Shape of the Investigation”.

C. Background repeatedly anticipates the results

The background chapter is clear and pedagogically strong, but it often does not stop at explaining concepts. It explains the later failure in advance.

Examples include statements that:

autoregressive training does not shape a smooth global energy,
the Lipschitz condition will fail,
the MH correction will paralyze the sampler,
the proposal term will collapse,
the first-order surrogate is the central object tested later,
the GFlowNet is the ideal instrument for locating the problem.

Some forward references are useful, but here the background frequently becomes a pre-discussion of results.

The section on DLS derives the proposal, explains the surrogate, states that everything depends on it, and then explains why Section 5.6 tests it. This is repeated in several consecutive paragraphs. The technical derivation is necessary. The repeated explanation of why it matters is not.

The CLS discussion similarly explains the Voronoi-cell geometry, discontinuity, reverse proposal issue, predicted acceptance collapse, and the later measured split. By the time the reader reaches the actual results, much of the interpretation has already been supplied.

Potential reduction: 2 to 3 pages.

D. Results sections repeatedly recap previous sections

The results are structured as a chain:

this result leaves a question,
the next experiment answers it,
that answer points to another mechanism,
the mechanism ties back to the central claim.

This creates coherence, but it also means each section spends considerable space summarizing the preceding one.

For example, the likelihood-trap section does not merely report the likelihood and repetition relationship. It again summarizes why the gradient cannot reach low energy, asks whether low energy would be desirable, answers that it would not, and then connects this to GFlowNet and the complete programme.

The cross-model consistency section is useful, but it largely consolidates findings already reported in the robustness subsection and individual experiments. It could be shortened to one compact table and two paragraphs.

The last-token analysis is valuable, especially the exact-zero observation. However, it partially re-establishes the already well-supported conclusion that the gradient is uninformative. It should be presented explicitly as a formal or structural confirmation, not another full central-result chapter.

E. The Discussion and Conclusion repeat the same synthesis again

Section 6.1 directly answers the four research questions. This is excellent and should remain.

However:

Section 6.2 again unifies all failures under one defect.
Section 6.3 again connects findings to the literature.
Section 6.4 qualifies the claims.
Section 6.5 again states implications and the decisive positive control.
The conclusion then restates the entire mechanism and contribution once more.

This is the biggest macro-level repetition. Once Section 6.1 has answered the research questions and Section 6.2 has synthesized the mechanism, the conclusion should be brief, perhaps two to three pages. The current conclusion begins on page 108 and the appendix begins at 116, suggesting roughly eight pages, which is too long after an already extensive discussion.

Potential reduction across Discussion and Conclusion: 4 to 6 pages.

3. Are the central questions answered?
RQ1: Can frozen autoregressive likelihood be sampled effectively with faithful Langevin dynamics, and if not, why?

Yes, strongly answered.

The thesis gives several forms of evidence:

step-size calibration does not transfer cleanly,
gradient direction is compared against a norm-preserved random direction,
the gradient-guided version does not reliably outperform random direction,
the Taylor surrogate is uncorrelated with the true energy change,
the surrogate is structurally blind to the candidate token’s self-term,
gradient-free rescoring performs better,
the last-token case provides an exact structural argument,
diffusion supplies a positive control.

This is the most convincing and best-developed RQ.

One qualification is needed: the answer is strongest for the tested model families, token-substitution setup, energy definition, and recovery task. The thesis does recognize scope limitations, but the repeated phrase “the frozen autoregressive likelihood carries no usable search direction on discrete text” can sound universal.

A safer formulation would be:

Across the tested autoregressive models and discrete token-recovery settings, the input-embedding gradient of sequence likelihood did not provide a useful proposal direction.

The current thesis often moves from “all tested models” to a broader “autoregressive likelihood does not provide a usable score over sequences”. The positive control makes that inference plausible, but it remains an inference rather than a proof covering all autoregressive language models.

RQ2: What is the role of MH correction in the two samplers?

Yes, clearly answered.

The thesis distinguishes:

DLS, where MH correction helps prevent misleading quenching behaviour and improves theoretical validity;
CLS, where the proposal ratio collapses around token-changing boundary moves.

This is a precise and useful answer. It is one of the strongest parts because it does not merely state that “MH hurts”. It decomposes the acceptance ratio and identifies the proposal term as the mechanism.

My only concern is that the thesis sometimes says the continuous sampler’s correction fails because the drift is “non-Lipschitz” or discontinuous while also describing the projected energy as piecewise constant and the differentiable network input pathway as smooth within cells. This distinction must remain mathematically exact. The thesis appears aware of it, but the wording should avoid making it sound as though a classical gradient of the projected piecewise-constant energy is directly being used. The implemented surrogate pathway and the actual target density need to remain carefully distinguished.

RQ3: Does GFlowNet amortization escape the problem?

Mostly yes, but the interpretation should be narrowed.

The empirical answer is no for the tested variants. The thesis identifies:

capacity collapse,
length collapse,
reward hacking,
failure of the tuned energy to become more navigable for Langevin sampling.

The evidence shows that the specific GFlowNet training attempts did not solve the problem.

However, the inference that the pathology “survives amortization because it is a property of the training objective of autoregressive language models” is somewhat stronger than the direct evidence. A GFlowNet’s failure can also depend on:

reward design,
training stability,
policy parameterization,
trajectory-balance implementation,
insufficient capacity,
exploration,
termination handling,
reward scaling.

The thesis discusses three such failure modes, but then unifies them strongly under the inherited energy. The experiment showing that the tuned energy changed substantially yet remained non-navigable is useful, but it does not establish that no amortized method could work with such a reward. It establishes that these GFlowNet variants did not repair gradient navigability.

Therefore, RQ3 is answered empirically, but the explanation should be phrased as:

Amortization did not escape the observed problems in the tested setup, and modifying the model through the GFlowNet objective did not produce an energy whose local gradient became more useful.

That is more defensible than implying a general failure of amortized inference.

RQ4: Can an additive constraint steer generation?

Answered, but the answer becomes complicated and slightly unstable.

Initially, the thesis appears to answer “no”: sentiment steering gains under the discrete sampler are small and inconsistent.

Later, the diffusion experiments show a more nuanced outcome:

the constraint gradient can contain a signal,
guidance affects the guiding classifier,
transfer to an external judge is asymmetric,
off-domain steering harms fluency,
on-domain trust-region steering can work in one direction,
classifier agreement limits observed transfer.

This means the ultimate answer is not simply “no”. It is closer to:

The additive constraint carries directional information, but successful external steering depends on preserving fluency, operating in a model with an appropriate learned score, and achieving sufficient alignment between guide and evaluator.

That is a valuable answer, but it is not fully aligned with the original formulation of RQ4, which asks whether an additive constraint can steer “on this energy landscape”. The later diffusion model uses a different score-trained landscape. It is partly a repair experiment rather than a direct answer about the original autoregressive energy.

The thesis should separate two conclusions more sharply:

On the frozen autoregressive energy landscape, plug-and-play constraint steering failed.
On a score-trained diffusion landscape, constrained steering became partly possible, subject to fluency and classifier-alignment limitations.

The material is there, but the current thesis risks blending those statements.

4. Is the thesis too long?
Yes

The main text runs to approximately page 115 before the appendix. For a master’s thesis, that is not automatically excessive, especially with many experiments, but here the length is partly caused by repeated prose rather than essential scientific content.

A more focused version could likely have:

1 to 1.5-page abstract
5-page introduction
8 to 10 pages of background and related work combined
8 to 10 pages methodology
30 to 35 pages results
8 to 10 pages discussion and conclusion
appendices for extensive diagnostics and examples

That would produce approximately 65 to 80 substantive pages before references and appendices, rather than about 108 pages before the conclusion ends.

Realistic reduction target

I would recommend cutting 15 to 20% of the prose, not 15 to 20% of the experiments.

The experiments are mostly worthwhile. The explanations around them are what should be compressed.

5. What can be shortened or removed?
High-priority cuts
1. Abstract

Reduce by at least 40%.

Keep:

problem,
tested models and samplers,
central gradient result,
principal mechanism,
GFlowNet outcome,
diffusion positive control,
one restrained concluding sentence.

Remove or compress:

detailed three-claim hierarchy,
classifier-transfer qualifications,
last-token explanation in full,
long causal interpretation.
2. Section 1.4, “The Shape of the Investigation”

This can be removed almost entirely.

A single sentence at the end of the introduction is enough:

Because the initial sampling experiments failed, the study developed into a diagnostic investigation of the underlying mechanisms.

The current section explains and defends the rhetorical structure instead of advancing the scientific argument.

3. Contributions paragraph in Section 1.5

Convert the long paragraph into four or five compact contributions. Currently it reproduces the complete Results chapter in miniature.

4. Structure of the thesis

The section-by-section roadmap is overly detailed. One concise paragraph is enough.

5. Repeated explanations in DLS background

Retain the mathematical proposal and Taylor approximation. Remove sentences repeatedly explaining that the surrogate is “the sampler’s entire claim to be gradient-guided” and that Section 5.6 will test it.

State that once.

6. Repeated CLS acceptance-ratio anticipation

The background should explain the geometry and theoretical concern. It should not already state the measured acceptance outcome in detail. Save “well under a percent” and boundary-crossing comparisons for Results.

7. Section 5.4 trajectory analysis

The trajectory analysis is visually useful but may not require a full standalone main-text section. A concise subsection or appendix reference would be sufficient unless it contributes independent quantitative evidence.

The full seeded trajectory and PCA illustration are already placed in the appendix. The main text can summarize the relevant geometric conclusion in one page.

8. Section 5.9 cross-model consistency

Reduce to:

one summary table,
one paragraph on GPT-2 variants,
one paragraph on Llama-3 scope.

Much of the existing prose re-reports prior numbers.

9. Guided diffusion subsections

Sections 5.13.4 and 5.13.5 are interesting, but they begin to resemble a second thesis:

classifier-guided diffusion,
independent judges,
off-domain continuation,
on-domain SST-2,
fluency trust region,
class-dependent transfer,
guide-judge agreement.

The core thesis is about whether a frozen autoregressive energy supplies a usable gradient and whether amortization repairs it. The cleanest positive control needs only show:

diffusion produces a meaningful local score;
it improves token recovery;
substituting its proposal signal into the existing chain improves recovery.

Those are Sections 5.13.1 to 5.13.3, and they strongly complete the causal story.

Sections 5.13.4 and 5.13.5 could be:

shortened substantially,
moved to an appendix,
or presented as exploratory follow-up work.

They are not useless, but they broaden the thesis from sampling diagnosis into a separate study of classifier-guided diffusion and evaluator alignment.

This is probably the single largest opportunity to improve focus.

10. Discussion and conclusion overlap

The Discussion should:

answer the RQs,
synthesize the mechanism,
discuss relation to prior work,
state limitations and implications.

The Conclusion should not repeat all evidence. It should state:

what was tested,
what was found,
what remains uncertain,
what the main practical implication is.

Approximately two to three pages would be enough.

6. Which parts are unnecessary or lower-value?

I would classify the following as lower-value rather than completely unnecessary:

“The Shape of the Investigation”

Mostly unnecessary.

Extensive physical intuition for Langevin dynamics

The marble-and-vibrating-surface analogy is clear, but somewhat long for a Computational Linguistics master’s audience. It can be reduced by half.

Also, the statement that gradually reducing vibration causes the final position to become “a fair draw from the landscape” should be handled carefully. Annealing step sizes in stochastic-gradient Langevin contexts and simulated annealing-like intuition are not identical, and the final sample interpretation depends on technical conditions. The analogy is pedagogically effective but may oversimplify the convergence claim.

Long prose before and after every table

Several table captions are already unusually detailed, sometimes functioning like mini-results sections. Then the surrounding paragraphs explain the same result again.

Choose one:

detailed caption plus compact prose, or
concise caption plus full prose.

Currently the thesis often does both.

Extensive list-of-figure and list-of-table captions

This is generated from the captions, but the captions themselves are extremely long. Some run to full paragraphs. Captions should be self-contained, but several include interpretation, limitations, comparisons, and conclusions. Shortening them would make the lists and the results chapter easier to scan.

Qualitative examples

The appendices contain several overlapping qualitative sections:

guided-generation examples,
qualitative recovery and steering showcase,
infill recovery showcase,
trust-region before/after,
ten seeded sequences.

These should be consolidated. A smaller representative set with an explicit selection policy is better than many showcases.

7. Does the thesis maintain a clear central line?
Yes, unusually clearly

Despite the length, the thesis has a strong conceptual spine:

A frozen autoregressive LM is treated as an energy.
Langevin sampling assumes a useful local direction.
That direction fails empirically.
The failure is explained geometrically and structurally.
Exact energy evaluation remains useful even when its gradient is not.
GFlowNet tuning does not make the resulting energy locally navigable.
A model explicitly trained to learn a score provides the missing signal.

That is a coherent and potentially valuable thesis contribution.

The problem is that the thesis sometimes adds too many secondary claims around that spine:

likelihood degeneration,
brevity incentives,
embedding anisotropy,
MH discontinuity,
amortization failure modes,
last-token proof,
external judges,
off-domain guidance,
on-domain trust regions,
classifier alignment.

Most are relevant, but not all deserve equal prominence. The thesis would improve if it clearly divided them into:

Core evidence

gradient versus random direction,
linearization correlation,
gradient-free baseline,
MH decomposition,
diffusion positive control.

Supporting diagnostics

embedding anisotropy,
quenching,
trajectory plots,
likelihood trap,
cross-model calibration.

Exploratory extensions

GFlowNet failure taxonomy,
classifier-guided diffusion,
trust-region experiments,
guide-judge agreement.

Currently, the Results chapter treats almost all of them as parts of one escalating definitive argument.

8. Are any claims too strong?

A few formulations should be moderated.

“The gradient of the frozen autoregressive likelihood carries no usable search direction on discrete text”

This is the thesis’s central claim, but as written it is universal.

Better:

In the tested token-substitution settings, the input-embedding gradient of frozen autoregressive sequence likelihood provided no reliable proposal advantage over a norm-matched random direction.

“Provably identically zero at the final token”

This is likely correct for the gradient with respect to the final token’s input embedding under a shifted causal language-model likelihood, because that embedding does not influence its own predicted probability and there are no future predictions. But it must always specify:

which gradient,
with respect to which representation,
under which likelihood indexing convention.

Otherwise readers may incorrectly interpret it as saying that all gradients involving the final token are zero.

“The pathology survives amortization because it is a property of the training objective”

The evidence supports this as a plausible mechanism, especially with the diffusion control, but “because” is stronger than “consistent with”.

Better:

The combined GFlowNet and diffusion results support the interpretation that the training objective, rather than only the sampling algorithm, is a key source of the failure.

“The training-free premise is refuted”

It is refuted for the tested framework, not necessarily for every possible training-free control method built on autoregressive models. Methods that do not differentiate raw joint likelihood, use lookahead, use conditional resampling, or operate in another representation may remain viable.

Better:

The experiments refute the premise that the frozen autoregressive sequence-likelihood gradient can supply the required local score for this class of Langevin-based methods without additional training.

9. Suggested revised structure

A tighter structure could be:

1 Introduction
Controllable generation problem
Energy-based promise
central untested assumption
four RQs
concise contributions

Remove “Shape of the Investigation”.

2 Background and Related Work

Combine the current Chapters 2 and 3 where possible:

autoregressive likelihood as energy
DLS and CLS
MH correction
GFlowNets
diffusion and score matching
gap in prior work

The current separate background and related-work chapters are defensible, but they overlap conceptually.

3 Methodology

Keep largely intact.

4 Core Sampling Results
calibration and quenching, compressed
MH correction comparison
gradient-versus-random result
linearization diagnosis
gradient-free baselines
likelihood trap as supporting evidence
5 Amortization and Positive Control
GFlowNet experiments
last-token structural analysis
diffusion signal
diffusion recovery
hybrid proposal test

Move classifier-guided diffusion and trust-region details to appendix or label them clearly as exploratory.

6 Discussion
direct RQ answers
integrated mechanism
scope and limitations
implications
7 Conclusion

Two to three pages.

10. Final honest assessment

This thesis is not empty, superficial, or poorly conceived. On the contrary, it may suffer from having too much analysis and too much desire to make every connection explicit.

The author has found a strong central narrative and then explains it repeatedly because they want to make sure the reader cannot miss it. The reader will not miss it. By roughly the middle of the Results chapter, the central result is already unmistakable.

The thesis would become more authoritative if it trusted the experiments more and the rhetoric less.

My concise judgment:

Blabbering: Moderate. Mostly over-explanation rather than meaningless content.
Repetition: High, especially between abstract, introduction, results transitions, discussion, and conclusion.
Research questions answered: Yes. RQ1 and RQ2 very strongly; RQ3 convincingly within the tested setup; RQ4 with a more nuanced answer than its initial formulation suggests.
Can it be shortened: Definitely.
Safe reduction: About 15% with no scientific loss.
Aggressive but still sensible reduction: About 20 to 25%.
Most expendable material: Section 1.4, repeated conceptual previews, long transition paragraphs, repeated result summaries, oversized captions, and some of the classifier-guided diffusion extension.
Core quality: Strong, coherent, and potentially excellent after tightening.