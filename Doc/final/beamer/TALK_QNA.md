# Defence Q&A: clarification questions and answers

Companion to `Presentation.tex` and `TALK_TRANSCRIPT.md`.

**Where these came from.** The question list is derived from `thesis_questions_knowledge_base.md`
in the repository root, which collates every point of confusion raised across the Gemini and
Claude reading sessions and the author's own reading-doubt notes. Its section 1, "confusion
hotspots", is the ranked list of what most reliably loses a careful reader, so the same ordering
is used here: **hotspots first**, then by chapter, then the questions the *new* results invite.

**Important, read once.** The knowledge base was written against an earlier draft. Three of its
findings have since been withdrawn or replaced, so several of its entries ask about things the
thesis no longer claims. Those are marked `SUPERSEDED` below and answered with the current
finding, not the old one. The three:

| old finding in the knowledge base | current finding |
|---|---|
| the **quenching effect**: annealing collapses the update into deterministic descent and freezes the chain | withdrawn. The proposal is **numerically uniform at every step**; nothing freezes, and the mean divergence *rises* over the schedule |
| the gradient is **reliably worse** than random (Llama-3, "anti-guidance") | withdrawn. It was the reversibility term charged to the policy arm alone. The supported claim is **indifference** |
| the **training objective** is the cause, and diffusion is the positive control that isolates it | replaced. A masked LM with **no** score objective does best, so the operative factor is **conditioning access**, and the ladder is an ordering rather than an isolation |

Format: **Q** as an examiner would put it, **Say** for the answer out loud in two to four
sentences, **Detail** for the follow-up if pressed, **Backup** for the slide, **Thesis** for
where it lives.

---

# Part 1: the confusion hotspots

These are the seven the knowledge base ranks highest. If you rehearse nothing else, rehearse
H1, H2 and H3.

---

### H1. The self-term blindness. "Why can the gradient not see how well the candidate token fits?"

*Knowledge base hotspot 1, the deepest: raised in the Gemini linearization thread and reopened
four times.*

> **Say.** Because the token plays two roles and the derivative only reaches one of them. As an
> *input*, it is a continuous embedding that flows forward and shapes every later prediction, and
> a derivative with respect to that embedding sees this perfectly. As a *target*, it is the thing
> the previous position was trying to predict, and it enters that term as a discrete index into
> the output softmax. There is no continuous quantity there to differentiate, so the derivative
> misses it entirely.

**Detail.** The decomposition measures both halves. Averaged over candidates, the future term is
24.2 nats and the self term is 15.0, so the invisible half is a substantial fraction of the
total. And the correlation confirms the structural argument: the surrogate correlates with the
future term at 0.03 and with the self term at minus 0.10, which is what you would predict if it
sees one and not the other.

**If pressed on "why an integer index".** Write the likelihood as a sum of `log p(x_t | x_<t)`.
The masked token at position `i` appears in that sum twice over: inside `x_<t` for every `t > i`,
where it is an embedding, and as the argument `x_i` of the term `log p(x_i | x_<i)`, where it is
the row you select out of a softmax. Selecting a row is not a differentiable operation in the
input.

**Backup** 17 (linearization radius and self-term blindness), 23 (what exactly the derivative is).
**Thesis** 5.4 and 5.4.1, Figure 6.

---

### H2. "You say the gradient is noise, yet the sampler clearly does something. How?"

*Knowledge base hotspot 2, recurring across five sections. The sharpest form: how does
correction-on reach 6.541 when correction-off reaches 9.499, if the gradient is useless?*

> **Say.** Because it is not the gradient doing the work, it is the accept/reject filter. The
> proposal is numerically uniform, so with the correction off you have a uniform random walk over
> the vocabulary, which does not converge at all: the mean divergence *rises*, from 8.765 to
> 9.499. Turn the correction on and that same uniform proposal becomes an independence sampler,
> and the filter supplies the entire selection pressure. That is where 6.541 comes from.

**Detail.** And there is an independent confirmation. A literal uniform draw, run through separate
code, gives half a percent recovery, 6.538 divergence and 9.48 percent acceptance, against zero,
6.541 and 9.98 for the gradient-guided sampler. Three significant figures. So the improvement
from the correction is real, and it is not evidence for the gradient.

`SUPERSEDED` note. An earlier draft explained this by "quenching", the idea that annealing
collapsed the update into deterministic descent and froze the chain in a local optimum. A
bit-identical re-run refuted it: the final-step proposal still has an effective support of tens of
thousands of tokens, the chain is still changing its token on 99 percent of final-decile steps,
and the divergence rises rather than falls. If a reader raises quenching, that is why the section
now carries an explicit withdrawal.

**Backup** 19 (likelihood is a poor objective yet used everywhere). **Thesis** 5.1.1, 5.6.2.

---

### H3. The slope-versus-zero-gradient paradox. "If the projected energy is piecewise constant, what is being differentiated?"

*Knowledge base `G1-09`, flagged there as hard-won and as the single wording that most confused a
careful reader.*

> **Say.** Two different objects, and this is the distinction the whole continuous half depends
> on. The projected energy is piecewise constant: inside a Voronoi cell the projected token does
> not change, so the energy does not change, and its derivative is zero in the interior and
> undefined on the boundary. Nobody differentiates that, and my sampler does not.
>
> What the sampler differentiates is a *different* function: the log-likelihood evaluated with
> the continuous state supplied as the input embedding, while the projected token supplies the
> prediction target. Inside one cell the target is fixed, so that function is smooth and has a
> well-defined gradient.

**Detail.** The consequence is the entire mechanism of slide 9. Inside a cell the pathway is
smooth. Cross a boundary and the *prediction target* changes, so the pathway itself jumps and the
drift jumps with it. That is the discontinuity that destroys the reverse-proposal term, and it is
a property of the pathway across boundaries, never a claim about a gradient of the piecewise-constant
energy, which does not exist.

**Backup** 18 (the MH chain). **Thesis** 2.4, the "two objects must be kept apart" paragraph.

---

### H4. "Could you not use the GFlowNet prompt framing and steer with the context tokens' gradients?"

*The author's own recurring idea, knowledge base thread A, `G1-24` through `G1-31`. Expect this
from a supervisor who has heard it before, and answer it as a considered alternative, not a
misunderstanding.*

> **Say.** It is a reasonable idea and I looked at it. The GFlowNet reformulates infilling as
> left-to-right generation under a restructured prompt, start, then ending, then the span to
> generate. If you put the span at the end and steer it with gradients from the context tokens,
> the position you are generating is at or near the sequence end, and that is precisely where the
> input-embedding gradient is provably zero. Move the span into the interior to avoid that and you
> are back in the interior case, where the same null holds.

**Detail.** There is a deeper reason it does not escape the diagnosis. What the context tokens'
gradients can tell you is still only the *future* effect of a change; the candidate's own
conditional fit is missing for the same structural reason. The token-indicator derivative is the
version of that idea that does work, and it needs no prompt restructuring at all.

**Also worth saying.** The left-to-right reformulation is not adopted for the samplers because it
would dissolve the capability under study, the ability to edit a token in light of both sides. It
would change the question rather than repair the metric.

**Backup** 25 (why a GFlowNet, and the task-comparability caveat). **Thesis** 4.5, 5.5.

---

### H5. "Would a smaller step size fix the continuous sampler?"

*Knowledge base hotspot 5, attacked from two directions in two separate sessions.*

> **Say.** No, and for a reason that is measured rather than argued. There are two length scales
> and they do not overlap. The gradient's first-order approximation has already decayed to zero
> well before 1.82, which is the mean distance to the nearest token. But 1.82 is the *smallest*
> move that changes a token at all. So any step small enough for the gradient to be valid is too
> small to change anything, and any step large enough to change a token is outside the valid
> range.

**Detail.** And the oracle configuration bounds every step-size policy at once. It picks the ideal
step at each iteration using knowledge of the correct answer, from fifty values spanning four
orders of magnitude, and even under that schedule the gradient does not beat a random direction.
A hand-tuned cap cannot do better than an oracle.

**Second reason.** Step size cannot address the self-term blindness at all. Making the step
smaller does not make the derivative see a term it is structurally unable to see.

**Thesis** 5.1, 5.4.

---

### H6. The statistics paragraph. "Walk me through the equivalence result."

*Knowledge base hotspot 6, `G2-12` to `G2-15`, recorded there as still open and high value.*

> **Say.** Three ingredients. First, it is *paired*: the corruption is deterministic per sample
> index, so the policy arm and the random arm see the identical corrupted sentence and I can
> difference them sentence by sentence rather than comparing two means. Second, three summaries
> instead of one, because the final state of a Markov chain throws away everything the chain
> visited: the last iterate, the mean over the second half, and the best state reached. Third, the
> test is two one-sided tests against a margin fixed in advance, which is how you demonstrate
> equivalence rather than merely failing to demonstrate difference.

**Detail, the numbers.** At a thousand pairs: last iterate plus 0.133 with interval minus 0.063 to
plus 0.326; chain mean plus 0.136 with minus 0.012 to plus 0.287; chain minimum plus 0.033 with
minus 0.079 to plus 0.145. All three inside plus or minus 0.327. At two hundred pairs the tests
declined to certify, because the smallest difference detectable at eighty percent power was 0.652,
twice the margin. Pooling four further corruption seeds fixed the power problem.

**Where the margin comes from.** Five percent of the policy arm's mean divergence, per
configuration. It is calibrated against the pipeline's own noise, not chosen to fit: re-running the
flagship under seeds 1000 to 1003 gives 6.348, 6.589, 6.150, 6.291, a standard deviation of 0.183,
comfortably inside 0.327.

**Say this before you are asked.** I do not call it pre-registered. There is no timestamped
registration; the margin was fixed before this comparison was run, and the thesis says "fixed in
advance" for exactly that reason.

**One disclosure.** The extra sample size comes from independent corruptions of the same 282
WikiText-2 sentences, not from new sentences, so the interval is a statement about variation across
corruptions. Sentence-level variation is still bounded by those 282 items.

**Thesis** 5.3, Table 3, Figure 3, 4.4, 4.8.

---

### H7. "Your anisotropy section reads like a comparison of scales, not of anisotropy."

*Knowledge base 5.8 entry, a `WORDING` correction the author made against the earlier draft.*

> **Say.** That was a fair criticism of the earlier draft and the section now separates the two
> explicitly, because they have different consequences. *Scale* is that GPT-2's inter-token
> distances are about three times Llama's, 1.82 against 0.59 nearest-neighbour. That is what breaks
> the step size: a step calibrated to cross a cell boundary in one geometry cannot cross one in
> the other. *Anisotropy* proper is that within a single model the embeddings occupy a narrow cone,
> and the evidence for it is the mean pairwise cosine, 0.086 and 0.018, well above zero, plus the
> fact that the top principal component explains only 2.3 percent of the variance so it is a broad
> cone rather than one dominant direction. That is what makes Euclidean distance unreliable as a
> metric, which is why the study uses a divergence instead.

**Thesis** 5.8.

---

# Part 2: by chapter, following the knowledge base ordering

## Chapter 2, background

### "Explain the Metropolis-Hastings acceptance ratio with a good and a bad move." `G1-02`

> **Say.** Two factors. The target ratio compares how probable the proposed state is against the
> current one, so it is above one for a good move and below one for a bad one; that is the part
> that expresses preference. The proposal ratio compares how likely the mechanism was to suggest
> the reverse move against the forward move, and that is what enforces detailed balance. It
> penalizes moves that are easy to make and hard to undo, because accepting those freely would let
> the chain drift and distort the distribution you claim to be sampling.

**Concretely, in my data.** A boundary-crossing move has target ratio plus 4.60, a good move, and
proposal ratio minus 1325. The second overwhelms the first and the good move is rejected.

### "What is a first-order Taylor surrogate?" `G1-07`

> **Say.** It is the linear estimate of how much a function changes if you move a little way from
> where you took the derivative: the gradient dotted with the displacement. Here the displacement
> is the difference between two token embeddings, so the surrogate estimates how much the sequence
> log-likelihood would change if you swapped one token for another. Everything the discrete
> sampler knows about which token is *good*, as opposed to merely *near*, is in that one term. If
> it is uninformative, the proposal is a distance-penalized random choice.

### "There is no state between cat and dog. Is that actually true?" `G1-05`

> **Say.** In token space, yes; that is the difficulty the whole thesis is about. The continuous
> sampler manufactures such a state by relaxing into embedding space, and the price is exactly
> what slide 9 measures: it leaves the manifold of real tokens, sits over a hundred units from any
> of them, and then has to either respect the correction and freeze or ignore it and wander.

## Chapter 3 and the diffusion comparison

### "What is a positive control here, and is the diffusion comparison one?" `G1-14`, and the `[OPEN]` 5.13 challenge

> **Say.** A positive control is a condition you expect to succeed, so that if it fails you know
> the instrument is broken rather than the hypothesis. I proposed the diffusion model as one, and
> the honest answer now is that it is not a control that isolates anything: SEDD differs from
> GPT-2 in objective, conditioning direction, scale and corpus all at once. It is one rung of a
> comparative ladder, and the masked-LM arm is the control that actually separates the factors.

`SUPERSEDED` note. The knowledge base records this as a pushback the author raised against an
earlier draft, and it was right. The thesis now bounds the attribution before reporting the result.

### "The failure is a missing self term in a gradient. Isn't that a *sampler* problem, not an objective problem?" `[OPEN]` in the knowledge base

> **Say.** That pushback was correct and the thesis now agrees with it. There are two separate
> mechanisms and they should not be merged. The Metropolis-Hastings breakdown is sampler-side, a
> consequence of projecting into a discrete space. The missing self term is proposal-side, a
> consequence of which coordinates the derivative is taken in. Neither is about the training
> objective, and the discussion chapter now lists four distinct mechanisms rather than claiming
> one unifying cause.

## Chapter 4, methodology

### "The KL metric is built on log-likelihood, which you say is a bad measure. Isn't that circular?" `G1-18`, `G1-23`, and thread B

*This is the tension the knowledge base flags as running through every layer: the KL metric,
SEDD's p(x), the GFlowNet reward and the Gibbs baseline all lean on a quantity the thesis calls
unreliable.*

> **Say.** The resolution is that two different uses of likelihood are being confused, and the
> thesis is careful to separate them. Ranking single-token substitutions in context is something
> the likelihood does well, and that is all the metric, the Gibbs baseline and the rescoring pass
> ask of it. *Maximizing* absolute sequence likelihood over free generation is what is degenerate,
> and that is the likelihood trap. My metric is also relative to a reference: it compares the
> next-token distribution under the recovered fill against the distribution under the
> ground-truth fill, so it is a measure of contextual fit, not of absolute likelihood as quality.

### "Why is the KL metric position-dependent, and does that not undermine the result?" `G1-20`, `G1-21`

> **Say.** Corrupting an early token perturbs the conditional probability of a long suffix and
> yields a large divergence; corrupting a late one yields almost none. That is why no model in the
> thesis is ranked on a small measured difference in divergence. The central results are argued
> from the *absence* of a gap between conditions, and a null gap is robust to positional
> dependence in a way that a narrow measured win would not be.

### "Why WikiText-2 for the grid when the model is fine-tuned on ROCStories?" corpus-map thread

> **Say.** Deliberately out of domain, and it is stated in the methodology with a table mapping
> every experiment family to its corpus. WikiText-2 validation is held out from any fine-tuning,
> which keeps the recovery task independent of the training data. And the domain shift is tested
> rather than assumed: if the null were an artifact of out-of-domain text it should weaken in
> domain, and it does not. The in-domain ROCStories diagnostics reproduce both the null, with the
> correlation below 0.06 throughout, and the likelihood trap.

**Backup** 21 (task design and corpus). **Thesis** 4.1, Table 1.

### "What configuration produced the external judge's numbers?" `5.5.3`, recorded as `MISSING`

> **Say.** The discrete sampler on GPT-2 Large, correction on, gradient normalization on, fifty
> steps, the same three proposal arms. That was missing from an earlier draft and is now stated in
> the text. The check is that the logged in-model divergences, 6.541 and 6.370, equal the flagship
> grid values, which confirms the configuration.

## Chapter 5

### "Untouched versus random-token fill. Conditional argmax versus conditional sample. Why are the conditional ones so poor?" baselines thread

> **Say.** Untouched leaves the corrupted token in place and scores 9.14; random-token fill
> replaces it with a random token and scores 9.39. Those are the two no-effort references. The
> conditional baselines take one forward pass and either take the argmax of the model's next-token
> distribution at the gap or sample from it, giving 8.24 and 8.62. They are poor because they see
> only the *left* context, so they choose a token that fits what came before and ignore what comes
> after.

### "You used 'left context' to explain both why Gibbs succeeds and why conditional argmax fails. That is inconsistent." recorded `PUSHBACK`

> **Say.** That was a fair catch against an earlier explanation, and the distinction is real.
> Conditional argmax proposes from the left context and *stops*. Gibbs proposes from the
> conditional and then *scores the whole sequence*, so the right context enters through the
> scoring step even though the proposal did not see it. The difference is not what the proposal
> sees; it is whether anything downstream checks the proposal against the full sequence.

### "If log-likelihood is a bad measure, how does Gibbs beat Langevin using it?" recorded as the sharpest form of thread B

> **Say.** Because Gibbs *evaluates* it and Langevin *differentiates* it, and those are different
> operations on the same object. The thesis's claim is now scoped exactly that far: the frozen
> likelihood is serviceable to evaluate and to search without gradients on this task, and
> unusable when differentiated in the input-embedding coordinates.

### "Figure 1: why do the norm-matched random and the fully random arms overlap perfectly?" `G1-48`

> **Say.** Because gradient normalization is on in that figure, and under normalization both arms
> become the same object, a random unit vector. That is why the extended ablation repeats the
> whole comparison with normalization *disabled*: only then do the policy arm and the
> norm-matched arm carry the same magnitude and differ in direction alone. The gradient-normalization
> setting is now stated in the caption, and there is a companion figure with it disabled where the
> three arms do separate.

### "What is 'the null', and what is 'final KL'?" `G2-16`, `G2-09`, `G2-10`

> **Say.** The null is the hypothesis of no difference between the gradient arm and the
> norm-matched random arm; "none overturned the null" means no variation produced a difference.
> Final KL is the mean divergence at the last step of the schedule, averaged over sequences. And a
> nat is a unit of information, a logarithm taken to base e rather than base two.

### "The degenerate strings. Which sampler produced them?" 5.10 `MISSING`

> **Say.** None of them. Those come from autoregressive greedy and beam decoding of the tuned
> policy, in the likelihood-trap decoding pass, not from any Langevin sampler. That attribution
> was missing from an earlier draft and is now stated at every degenerate example.

### "The constrained-generation gains flip sign with the target label, in every arm. What does that mean?" 5.11 thread

> **Say.** It means the raw gains are untrustworthy, and it is the reason the thesis does not use
> them. If even the pure-fluency baseline and the fully random arm flip sign when you switch the
> target from positive to negative, then a fixed sentiment drift in the underlying generations is
> swamping any contribution from the constraint. A gain that moves with the target regardless of
> whether the constraint gradient is present at all cannot be read as steering. The statistic that
> cancels the shared drift is the paired contrast between the constraint-only arm and the arm
> where only the constraint gradient is replaced by noise, and that contrast is essentially zero on
> my sampler.

**Analogy if useful.** It is a moving walkway. Everyone on it moves forward whether or not they
are walking, so you cannot tell who is walking by measuring position. You have to compare two
people on the same walkway.

---

# Part 3: what the new results invite

The knowledge base predates these, so nothing there covers them. Expect all of them.

### N1. "If the token-indicator derivative works, why is your grid built on the one that does not?"

> **Say.** Because it was found by re-analysing measurements already taken, after the grid was
> complete, and it needed no new forward passes to find. The self term was already recorded as a
> column in the linearization data; nobody had thought to add it back. Rebuilding the 145-cell grid
> on it is the obvious next study and it is named as such in the limitations. What the thesis
> establishes is that the signal is present and usable, not that any off-the-shelf schedule will
> use it.

**Do not over-claim.** It also carries a condition: it needs a configuration where the surrogate,
not the distance term, shapes the proposal, and neither model's calibrated setting is one.

### N1b. "Is the Llama-3 41 percent a matched comparison?"

> **Say.** On GPT-2 yes, on Llama no, and the thesis says which is which. The GPT-2 result is a
> cell-for-cell comparison of the two surrogates over one step-size and temperature grid, so
> there the derivative is the only variable. On Llama I never ran the input-embedding gradient at
> the surrogate-driven setting, so the forty-one percent is measured against every Llama
> configuration that *was* run, all of which recover nothing.

**Volunteer the matched Llama cell, because it is the honest one.** At Llama's calibrated setting
the two surrogates *are* matched, same model, same sequences, same step size and temperature, and
there the token-indicator proposal is not better but marginally worse: 4.108 against 3.898, with
both at zero percent exact. So on Llama the forty-one percent is bought by the derivative and the
configuration together. That is why the thesis states the cross-architecture claim as conditional.

**One more caution.** The 1.908 divergence at that cell is on Llama's own scale, which the thesis
keeps off the axis the GPT-2 family is measured on, so it should not be ranked against any GPT-2
figure.

### N2. "Is this not just the one-hot gradient, which is standard?"

> **Say.** No, and the distinction is the mathematical heart of it. The ordinary derivative of the
> likelihood with respect to a one-hot input, through the embedding lookup, is `E` transpose times
> `g` by the chain rule, and its v-th coordinate is the *future* term alone. It contains nothing new.
> What I define relaxes the token indicator in *both* of its roles, as the input predicting the
> suffix and as the target of the preceding prediction, and only that two-role relaxation exposes
> the self term. That is why the thesis calls it the relaxed token-indicator derivative rather than
> "the one-hot gradient", which is ambiguous about exactly this point.

**Backup** 23 has the relaxed objective written out.

### N3. "Why does RoBERTa beat SEDD? Does that not undermine your story?"

> **Say.** It is the finding, not a problem for it. Both are bidirectional and only SEDD is
> score-trained, so a masked LM beating both diffusion models is what shows score training is
> sufficient and not necessary. RoBERTa-large is also bigger and trained on more text, and
> masked-token prediction is its native query at any noise level whereas SEDD answers it at a
> schedule point. So I do not claim a ranking of the two models; I claim the ordering by
> conditioning.

**Volunteer the limit.** The ladder is an ordering, not an isolation. These models also differ in
scale, corpus and architecture, and only the tokenizer and the chain are held fixed across all of
them. The thesis says so in the section and again in the limitations.

**Backup** 24.

### N4. "Forty-four percent recovery could just be leakage in your pipeline."

> **Say.** That was my first thought too, which is why there is a control. Replacing the masked-LM
> conditional with a uniform draw over the same bridged vocabulary, through identical code, gives
> half a percent recovery and 6.538 divergence. If the pipeline leaked, the uniform arm would not
> sit at chance. And the tokenizer bridge covers 99.95 percent of the vocabulary with zero
> ground-truth tokens falling outside it, so nothing is unreachable by construction.

### N5. "Why did the temperature flatten everything, and how did you miss it?"

> **Say.** The proposal logit is the two terms divided by a temperature. The surrogate's spread
> across the vocabulary is a few nats, and dividing that by five against a ten-nat entropy budget
> leaves a softmax that is uniform whatever you feed it. I missed it because the temperature was
> calibrated for *motion*, that the sampler's tokens change at all, which it achieves, and motion
> is a weaker property than guidance. The sweep is what exposed it, and the thesis now measures
> proposal entropy before running any comparison.

**Detail worth having.** Sharpness from the distance term and sharpness from the surrogate are
different things. On Llama's calibrated setting the proposal is very sharp, minimum entropy
0.0034, but sharp about *staying put*, because the small step size makes the distance term
enormous; the ratio of surrogate to distance term is only 1.83. In the recovering configuration
that ratio is 229 and the proposal is sharp about the *ranking*. Entropy alone would have been the
wrong diagnostic.

**Backup** 22.

### N6. "You call it certified equivalence. Was the margin pre-registered?"

> **Say.** No, and I do not use that word. There is no timestamped registration. The margin was
> fixed before this comparison was run, and it is calibrated to the pipeline's own seed noise
> rather than chosen to fit the result, so the thesis says "fixed in advance" throughout.

### N7. "You say the target is not at fault, and also that its optimum is degenerate. Which is it?"

> **Say.** Both, and they are about different regimes, which is why the thesis scopes each one. On
> this recovery task the frozen likelihood is a perfectly serviceable thing to evaluate, and
> gradient-free methods search it successfully; that is what rejects the first hypothesis. As an
> objective to *maximize* in open-ended generation it is badly behaved, and I measure that: the
> within-strategy correlation between per-token likelihood and repetition runs up to plus 0.91, and
> the highest-likelihood beam output is literal gibberish. The abstract and the discussion both
> carry the scoped version.

### N8. "Your Llama-3 contrast is significant against the gradient. Why not report it?"

> **Say.** Because I do not think it survives scrutiny, and I would rather say that than build on
> it. It is one contrast out of several dozen paired comparisons across the grid, uncorrected for
> multiplicity, so at the five percent level a handful of hits is expected. It is measured with the
> correction *disabled*, so by my own argument it compares two biased optimizers rather than two
> samplers. And when I tested the same regime on GPT-2, three sharp cells with the correction off,
> the gradient came out *better* than random in all three, so it does not reproduce.

`SUPERSEDED` note. An earlier draft claimed the gradient was reliably worse than noise. That was
an artifact of charging the reversibility term to the policy arm alone; recomputing it exactly for
every arm turns a gap of plus 2.34 nats into minus 0.25. The supported claim is indifference.

**Backup** 21.

### N9. "What would you do differently, and what next?"

> **Say.** Differently: measure the proposal's entropy before running a grid on it. That one
> measurement would have reordered the whole study, and it cost nothing.
>
> Next, three things in order. Rebuild the configuration grid on the token-indicator derivative,
> since everything here says it should work and nothing here has tested it at scale. Then run the
> constraint on a proposal that actually navigates, either the masked LM or the token-indicator
> arm; my classifier-guidance experiment used the diffusion model only because at the time it was
> the only proposal that recovered anything. And third, the honest negative-space question: whether
> any of this survives at instruction-tuned scale, which I cannot answer from a 774-million-parameter
> model and an 8-billion-parameter control.

### N10. "What is the one-sentence contribution?"

> **Say.** That the failure of gradient-guided sampling on a frozen language model is not a
> failure of the model or of the target, but of the coordinates the derivative is taken in, and
> that taking it in the token-indicator coordinates instead turns zero percent recovery into forty
> on the same frozen weights and the same sampler.

---

# Part 4: questions I would not want, and how to hold the line

| question | the trap | hold this line |
|---|---|---|
| "So gradient-guided controllable generation is dead?" | inviting the unscoped claim | "Embedding-space gradient guidance of a frozen autoregressive likelihood, as in the MuCoLa and COLD family, is what I refute. Gradient-guided discrete sampling as such is not, and my own token-indicator result is a counterexample." |
| "You have reinterpreted COLD and MuCoLa as not really sampling." | broader than the evidence | "I measured the mechanism as I reimplemented it. Their published results involve tasks, tuning, constraints and evaluation procedures I did not reproduce, and I make no claim about those." |
| "Your models are small." | true, and fine to concede | "774 million parameters with an 8-billion cross-architecture control. Every qualitative finding holds on both, and the token-indicator result is 40 percent on one and 41 on the other, but scale is a genuine limitation and it is stated as one." |
| "Is n = 50 per cell enough for the sweep?" | conceding too much | "For the sweep, yes, because it is asked to detect a *presence*, and 40 percent recovery in a cell where the alternative shows zero does not need a large sample. The flagship *equivalence* claim is where power matters, and that is the one at a thousand pairs." |
| "Why should I believe a negative result?" | defensiveness | "Because the same design also produces positive results on the same energy and the same chain: 33 percent for rescoring, 40 for the token-indicator arm, 44.5 for the masked LM. An instrument that only ever returns zero is suspect. Mine does not." |

---

## Coverage against the knowledge base

| knowledge base item | covered |
|---|---|
| hotspot 1, self-term blindness | H1 |
| hotspot 2, "gradient is noise yet something works" | H2 |
| hotspot 3, MH correction and the discontinuity trap, slope-vs-zero | H3, and Chapter 2 entries |
| hotspot 4, the GFlowNet-framing idea | H4 |
| hotspot 5, smaller step size | H5 |
| hotspot 6, the statistics paragraph | H6 |
| hotspot 7, scale versus anisotropy; missing 5.5.3 config | H7, and Chapter 4 entries |
| thread B, "likelihood is bad yet everywhere" | Chapter 4, first entry, and N7 |
| thread C, task design and prefix continuation | H4, Chapter 4 corpus entry |
| thread D, how MuCoLa and COLD worked | Part 4, second row |
| thread E, the gradient-fallacy mechanism | H1, then N2 |
| thread F, the MH background chain | H3, Chapter 2 entries |
| 5.2 quenching questions `G1-44`, `G1-45`, `G1-46` | H2, answered with the withdrawal |
| 5.13 open items | Chapter 3 entries, N3 |
| open items on nats, final KL, "the null" | Chapter 5, the definitions entry |

Not carried over: the knowledge base's `G1-11` to `G1-13` code-walkthrough requests, which are
implementation questions better answered at a screen than in a defence, and its section 4
thesis-gap list, all ten items of which have been applied to the document.
