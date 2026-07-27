## Overall judgment

This version contains a **stronger and more interesting scientific result** than the previous one. The new one-hot and bidirectionality experiments substantially improve the thesis because they prevent an overly broad negative conclusion. The thesis no longer merely says “the gradient fails”. It now identifies **which representation of the signal fails, which information is missing, and which alternatives recover it**.

However, the new experiments also create several logical and structural tensions. The thesis has changed its explanation, but not every earlier chapter has been fully rewritten around that change. As a result, the evidence is strong, but the argument sometimes appears to support two different causal stories:

1. The autoregressive training objective fails to produce a usable score.
2. The required information already exists in the autoregressive model, but the input-embedding derivative and lack of right-context conditioning discard it.

The final results support the second explanation more strongly. The thesis needs to make that its consistent central argument from the abstract through the conclusion. 

My honest assessment:

| Aspect                  | Assessment                                                                                                                 |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Scientific contribution | Strong and more original than before                                                                                       |
| Argumentative flow      | Good overall, but disrupted by a late causal reversal                                                                      |
| Logical consistency     | Several fixable inconsistencies, including one potentially important mathematical terminology issue                        |
| Research questions      | Answered, but RQ1 should be reformulated                                                                                   |
| Repetition              | Moderate to high                                                                                                           |
| Length                  | Too long again                                                                                                             |
| Possible reduction      | Around 12 to 20 pages                                                                                                      |
| Main risk               | The thesis sometimes retains the old “training objective causes the failure” explanation after later experiments revise it |

# 1. Most important logical inconsistency: the causal explanation changes

The thesis initially builds a strong causal story around the **training objective**.

The Background argues that autoregressive teacher forcing does not train a global score over sequences, whereas diffusion training does. The diffusion experiment is then introduced as a positive control:

> If a score-trained model supplies the direction that the autoregressive gradient lacks, the training objective is isolated as the cause.

This is the logic behind Sections 2.5, 2.6, 5.14, and parts of the Related Work.

But Section 5.15 then demonstrates something more specific:

* the left-to-right autoregressive conditional recovers 23.5%;
* score-trained diffusion recovers about 39%;
* non-score-trained bidirectional RoBERTa recovers 44.5%;
* therefore, score training is sufficient but not necessary;
* proposal performance follows output-side access and conditioning, not whether the model was trained by score matching.

The thesis itself explicitly admits that this **revises rather than merely qualifies** the earlier attribution:

> “The earlier reading, that the training objective is the source of the failure, is too strong…”

That is an intellectually honest and scientifically valuable correction. But it means that the earlier causal framing is no longer logically sustainable in its current form. 

## Why this matters

The diffusion experiment no longer isolates the training objective because diffusion and autoregressive models differ simultaneously in:

* objective,
* conditioning direction,
* architecture,
* scale,
* corpus,
* revision query.

Section 5.15 correctly introduces RoBERTa as a discriminating control. Once RoBERTa performs best, the diffusion experiment should no longer be described as having established the training objective as the causal variable.

The strongest supported conclusion is now:

> The input-embedding derivative discards the candidate’s direct conditional-fit information. Output-side proposal distributions recover that information, and bidirectional conditioning adds the right context required for in-place revision. Score training is one way to obtain such a proposal, but it is not necessary.

That is a sharper and more convincing contribution than the old training-objective claim.

## What should change

The following parts should be rewritten:

* Section 2.5 should introduce score training as **one possible source of a useful proposal**, not the expected causal repair.
* Section 2.6 should avoid saying that score training supplies exactly what autoregressive models inherently lack.
* Section 3.5 should frame diffusion as a comparative model family, not a causal positive control that isolates objective.
* The beginning of Section 5.14 should state that diffusion tests whether a trained bidirectional denoising proposal works, while acknowledging that objective and conditioning are initially confounded.
* Section 5.14.3 should not call the experiment sufficient to isolate the training objective.
* Section 6.2 should consistently make derivative choice and conditioning access the operative variables.
* The conclusion should avoid any remaining suggestion that score matching is necessary.

The abstract has already moved in the correct direction by saying:

> “The operative variable is what a proposal may condition on, not the objective it was trained with.”

The rest of the thesis should fully align with that sentence. 

# 2. Potential mathematical inconsistency: “one-hot input gradient”

This is the issue I would inspect most carefully before submission.

The thesis writes:

[
\frac{\partial \log p(x)}{\partial x_i[v]}
==========================================

\log p(v\mid x_{<i})
+
g^\top e(v)
]

and calls this the **one-hot input gradient**. It then says this derivative captures:

* the self term, through the candidate’s output conditional probability;
* the future term, through the embedding pathway.

Conceptually, the decomposition is sensible. The full effect of substituting token (v) indeed includes:

1. how probable (v) is given the left context;
2. how using (v) affects later predictions.

But the word **gradient** may not be mathematically justified unless the relaxed variable is defined extremely carefully.

## The problem

In an ordinary autoregressive model:

* the one-hot token is mapped through the embedding matrix into an input embedding;
* differentiating the likelihood with respect to that one-hot input through the embedding lookup gives an embedding-path derivative;
* by the chain rule, it should be closely related to (E^\top g).

That standard derivative does **not automatically include the candidate’s role as the discrete prediction target**, because the target index in a cross-entropy term is normally treated as a fixed label, not as a differentiable input variable.

To obtain the stated self term, the thesis appears to relax the same token indicator in two roles:

* as the model input used to predict future tokens;
* as the one-hot target used in the current-position log-likelihood.

That can be defined, but then it is not simply the ordinary “gradient with respect to the one-hot input”. It is a derivative under a specifically constructed continuous relaxation of the token variable as both input and target.

## Why this needs clarification

Without a precise definition, an examiner may ask:

> How can differentiating the input token recover the cross-entropy term in which that token is the target of the previous position?

This question is central because the strongest new contribution depends on the distinction.

## Recommended terminology

A safer name might be:

* **full one-hot token surrogate**
* **relaxed token-indicator derivative**
* **output-augmented one-hot surrogate**
* **full substitution surrogate**

Then define the relaxed scalar explicitly, for example:

[
\tilde{L}(z_i)
==============

\sum_v z_i[v]\log p(v\mid x_{<i})
+
L_{\text{future}}\left(E^\top z_i\right)
]

with (z_i) relaxed from the simplex vertex corresponding to the current token.

The derivative is then transparently:

[
\frac{\partial \tilde{L}}{\partial z_i[v]}
==========================================

\log p(v\mid x_{<i})
+
\nabla_{e_i}L_{\text{future}}^\top e(v).
]

This would make the mathematics clean. It would also demonstrate that the proposed quantity is not the standard input gradient used by the sampler, but a deliberately expanded surrogate for the complete substitution effect.

At present, calling it simply the “one-hot input gradient” risks appearing internally contradictory with the statement that the useful information lives on the model’s output side. 

# 3. RQ1 is no longer formulated precisely enough

RQ1 asks:

> Can the frozen likelihood of an autoregressive language model be sampled effectively with theoretically faithful Langevin dynamics on the discrete text manifold, and if not, why not?

The final results no longer support a simple negative answer.

The thesis now shows:

* the input-embedding gradient version fails;
* the one-hot or output-augmented surrogate succeeds at around 40%;
* the model’s ordinary autoregressive output conditional reaches 23.5%;
* a masked LM reaches 44.5%;
* the frozen energy can be searched using useful output-side proposals.

Therefore, the frozen likelihood is not generally unsampleable. A particular gradient-based proposal is defective.

The Discussion handles this reasonably well by saying that the negative finding applies to the **input-embedding derivative**, not gradient guidance or frozen models in general. But the wording of RQ1 remains broader than its answer. 

## Better formulation

A revised RQ1 could be:

> **RQ1. Can input-embedding-gradient Langevin proposals effectively sample a frozen autoregressive likelihood on discrete text, and what information do they preserve or discard?**

Or:

> **RQ1. Does the input-embedding gradient of frozen autoregressive sequence likelihood provide a useful proposal direction for discrete token revision, and if not, which alternative local quantities do?**

That formulation now matches the thesis’s actual strongest contribution.

## Why this improves the flow

The thesis would then have a natural progression:

1. Test input-embedding-gradient sampling.
2. Show that it behaves like random.
3. rule out a flat-proposal artefact with the parameter sweep.
4. identify the missing self term.
5. reconstruct a fuller output-side surrogate.
6. show that it works.
7. test how much additional benefit comes from right-context conditioning.

That is a much stronger narrative than “the frozen likelihood cannot be sampled”.

# 4. Contribution count is plainly inconsistent

The Introduction says:

> “the thesis makes five contributions”

and then lists **six** numbered contributions.

This is a straightforward inconsistency and should be corrected. 

More importantly, the contribution list does not reflect the final causal result well enough.

The sixth contribution still describes the diffusion model as supplying the direction “the autoregressive gradient lacks”. That is true for the input-embedding gradient, but incomplete after Sections 5.7.1 and 5.15, because:

* the autoregressive model already contains a usable output-side direction;
* diffusion is not necessary;
* the masked LM performs even better.

The final contribution should probably combine diffusion and the proposal ladder:

> A controlled proposal ladder showing that recovery increases with access to output-side and bidirectional conditional information, from input-embedding gradient and uniform proposals, through the autoregressive conditional, to diffusion and masked-language-model proposals.

That is more original and more consistent with the final thesis.

# 5. The argumentative flow is strong until the late revision

## What works well

The first part has a coherent investigative sequence:

1. establish step-size and geometry problems;
2. inspect annealing and proposal sharpness;
3. analyze the MH correction;
4. compare gradient and random direction;
5. test robustness;
6. introduce gradient-free baselines;
7. test whether the null survives a parameter range in which the gradient actually controls the proposal;
8. examine linearization;
9. decompose self and future terms;
10. test across architectures.

That progression is much better than merely presenting a large hyperparameter grid. Each experiment answers a question raised by the previous result.

The proposal ladder in Section 5.15 is especially strong because it holds the chain and target energy fixed and changes only the proposal. It gives the thesis a clean empirical axis:

* no conditioning,
* left-only conditioning,
* bidirectional conditioning.

That experiment should be treated as one of the thesis’s central results, not as a late appendix-like addition. 

## Where the flow weakens

The main weakness is that the thesis reaches a preliminary causal conclusion too early:

> autoregressive training does not shape a score, therefore training objective is the source of the failure.

Then Section 5.15 overturns that conclusion.

A scientific narrative is allowed to revise its hypothesis. In fact, this can be compelling. But the current version does not present this as a clearly planned sequence of competing hypotheses. It reads more like the thesis was almost complete and a later control changed the answer.

## Better argumentative architecture

Present three candidate explanations from the beginning:

### Hypothesis A: the target energy is unusable

Test with gradient-free rescoring and Gibbs. Rejected, because exact energy evaluation works.

### Hypothesis B: autoregressive training fails to contain the relevant information

Test with the one-hot/output-side conditional. Rejected, because useful information already exists in the model.

### Hypothesis C: the input-embedding proposal discards direct token-fit information, and right-context conditioning further improves revision

Supported by:

* self/future decomposition,
* final-position case,
* one-hot surrogate,
* autoregressive conditional,
* diffusion proposal,
* RoBERTa proposal.

Under this structure, diffusion is no longer a “causal positive control” for the training objective. It is one point in a proposal-information ladder.

That would make the flow more logical and remove the current reversal.

# 6. The “unified mechanism” is not entirely unified

Section 6.2 tries to connect:

* input-gradient failure,
* continuous MH breakdown,
* GFlowNet collapse,
* likelihood degeneration,
* constraint steering failure,
* proposal conditioning.

But these are not all manifestations of one identical mechanism.

There are at least four distinct mechanisms:

1. **Proposal-information failure**
   The input-embedding derivative omits the token’s direct conditional-fit term.

2. **Continuous-state correctness failure**
   Projection makes the drift discontinuous and destroys the MH reverse proposal.

3. **Objective-quality failure**
   High sequence likelihood can correspond to degenerate text.

4. **Amortized-training failure**
   GFlowNet variants collapse because of reward, length, capacity, or training dynamics.

They are related because they all challenge the original plug-and-play construction, but they are not caused by one single defect.

For example, the GFlowNet does not use the input-embedding gradient at all. Its failure cannot be explained by the missing self term in that gradient. Likewise, the likelihood trap is not caused by gradient linearization.

The Discussion should therefore use a hierarchy:

## Shared high-level problem

The original plug-and-play construction assumes that raw frozen likelihood can serve simultaneously as:

* a quality objective,
* a locally navigable energy,
* a reward for amortization.

The experiments show that these roles come apart.

## Distinct lower-level mechanisms

* local proposal failure,
* projected continuous dynamics failure,
* reward/objective degeneration,
* amortized training collapse.

This is more logically accurate than claiming one mechanical cause explains everything.

# 7. RQ2 is answered, but one statement is too broad

RQ2 is answered clearly:

* MH helps or regularizes the discrete chain;
* MH disables token-changing moves in the continuous projected chain;
* the proposal term causes the collapse.

That is one of the strongest parts of the thesis.

However, statements like:

> “correction-free methods are biased optimizers rather than samplers”

are defensible in a formal sense, but the further statement that their published successes are “successes of optimization under early stopping” is broader than the thesis’s direct evidence.

The thesis evaluates its own implementations and related mechanisms. It does not necessarily reproduce all published systems, tasks, tuning choices, constraints, or evaluation procedures. Therefore, it should avoid turning its result into a universal reinterpretation of all correction-free methods.

A safer formulation is:

> In the implementations and landscapes tested here, omitting the correction makes the update behave as an early-stopped stochastic optimizer rather than an exact sampler from the stated target.

The Related Work currently contains too much verdict-like interpretation before the thesis has shown its evidence. 

# 8. RQ3 is answered, but two questions are partially conflated

RQ3 asks whether amortization with a GFlowNet escapes the failures.

The thesis appears to test two different things:

1. Does the trained GFlowNet policy successfully generate high-reward text?
2. Does the energy induced by the GFlowNet-tuned model become more navigable by Langevin gradient?

These are related but not identical.

A GFlowNet is an amortized policy. Evaluating Langevin sampling on a GFlowNet-tuned energy tests the geometry of the **tuned model’s energy**, not directly whether the GFlowNet policy successfully amortizes sampling.

The three observed collapse modes answer the first question. The “Langevin on tuned versus base energy” experiment answers the second.

The thesis should state this distinction explicitly:

* **Policy-level result:** the tested GFlowNet variants collapsed or exploited reward pathologies.
* **energy-level result:** tuning changed the model substantially but did not make the input-embedding local surrogate more useful.

Both are good findings. Combining them into “amortization did not repair the landscape” is understandable, but slightly imprecise because amortization and energy smoothing are not the same objective.

# 9. RQ4 is answered, but it has become secondary

RQ4 is answered with an appropriately nuanced conclusion:

* on the frozen autoregressive energy and discrete sampler, the real constraint direction does not outperform a randomized direction;
* on the continuous constraint-only setup, an apparent signal exists but is entangled with off-manifold degeneration;
* diffusion-guided steering is exploratory and concerns another landscape.

This is logically sound. The thesis correctly keeps the two landscapes apart. 

However, RQ4 now feels less integrated into the main contribution than RQ1 and the proposal ladder. The contribution list does not mention it, while a large appendix is devoted to classifier-guided steering.

You have two defensible choices:

### Keep RQ4 central

Then add an explicit contribution concerning the controlled real-versus-random constraint comparison and explain why it is essential to the thesis title’s “controllable text generation”.

### Make it an extension

Then remove it from the four central RQs and label it as an exploratory application after the primary sampling diagnosis.

At present it is formally central but substantively secondary.

# 10. Is it repeating itself?

## Yes, more than the previous 117-page version

The thesis has returned to 142 pages. Some increase is justified because the new controls are scientifically valuable, but the main text has expanded from roughly 84 pages to about 111 pages.

The main repeated ideas are:

* input-embedding gradient behaves like random;
* the self term is missing;
* the final token provides the exact case;
* output-side quantities recover the information;
* energy evaluation works while input-gradient search fails;
* bidirectionality provides right context;
* training objective is not the operative variable;
* MH behaves oppositely in DLS and CLS.

These points appear in:

* abstract,
* introduction,
* contribution list,
* background,
* related work,
* several Results transitions,
* RQ answers,
* unified mechanism,
* implications,
* conclusion.

Some repetition is normal. The problem is that the same point is often fully re-explained rather than merely referenced.

## Particularly repetitive areas

### Sections 5.5, 5.6, 5.7, 5.7.1, and 5.13

These all establish versions of:

> the input-embedding gradient does not capture token fitness.

They should remain separate experiments, but their introductory and concluding prose could be compressed.

### Sections 5.14 and 5.15

Section 5.14 builds a long causal interpretation around diffusion. Section 5.15 then revises it. Rewriting 5.14 as an intermediate comparison would eliminate substantial repeated explanation.

### Sections 6.1 and 6.2

Section 6.1 already gives detailed answers and mechanisms. Section 6.2 then explains much of the same evidence again as a unified account.

Section 6.1 should answer each RQ concisely. Section 6.2 should synthesize only what is not already obvious from the individual answers.

### Conclusion

The conclusion begins around page 101 and the appendix at page 112, making it approximately eleven pages. That is too long after a ten-page Discussion.

The conclusion appears to reproduce:

* all key null results,
* the proposal sweep,
* the asymmetric-correction artefact,
* one-hot recovery,
* MH outcomes,
* likelihood trap,
* GFlowNet results,
* diffusion,
* RoBERTa,
* uncertainty,
* practical implications.

This should be reduced to four or five pages.

# 11. What can be shortened or removed?

A realistic reduction is **12 to 20 pages**, mostly from prose rather than experiments.

## High-priority reductions

### 1. Proposal-to-final-RQ mapping

The paragraph mapping proposal RQ1a, RQ1b, RQ2a, RQ2b, and RQ3a to the final RQs is not useful to most readers.

It interrupts the introduction and creates numbering complexity. Move it to a footnote or remove it.

### 2. Correct and shorten the contributions

Use six contributions if there are six, or preferably consolidate them into five.

Suggested structure:

1. Input-gradient versus random ablation and parameter sweep.
2. Self/future decomposition and output-side surrogate.
3. MH decomposition for discrete and continuous samplers.
4. GFlowNet and raw-likelihood reward pathologies.
5. Controlled proposal ladder separating derivative, output access, bidirectionality, and score training.

This is more coherent than treating diffusion as a separate causal proof.

### 3. Compress Background Sections 2.5 and 2.6

Now that score training is not the operative variable, the two sections devote too much space to preparing a causal explanation that the thesis later rejects.

Keep enough background to understand SEDD, but reduce the claim that denoising is the unique repair.

### 4. Shorten Related Work Section 3.2

It is too long and partly becomes a second Discussion chapter.

Move statements about what prior successes “really” represent to Section 6.3.

### 5. Merge Sections 5.1 and 5.2

Step-size calibration, proposal entropy, annealing, and near-uniform proposal behaviour are tightly connected. A single section called something like:

> “Proposal Calibration and the Near-Uniform Main Grid”

would improve flow.

### 6. Merge or tightly connect 5.5 and 5.6

The main result and the load-bearing sweep are logically one result:

* the original grid’s proposal is too flat;
* the broader sweep fixes this concern;
* the null remains.

Treat them as one central section with a robustness subsection.

### 7. Reorganize 5.7 and 5.7.1

The one-hot result is more important than the generic “linearization radius” framing. A better structure:

* 5.7 Why the input-gradient surrogate fails
* 5.7.1 Self and future decomposition
* 5.7.2 Recovering the missing term with an output-side surrogate

This avoids making the later result feel appended.

### 8. Move 5.14.4 fully to the appendix

Classifier-guided steering is exploratory and not needed for the main causal argument.

A one-paragraph main-text summary is enough.

### 9. Make Section 5.15 a central subsection

It should not appear as a final late addition after the exploratory steering material.

Place it immediately after the diffusion proposal result:

* diffusion proposal works;
* is score training necessary?
* RoBERTa control says no;
* proposal conditioning explains the ordering.

### 10. Cut the conclusion by at least half

This is the easiest large reduction with minimal scientific loss.

# 12. Final evaluation of the research questions

## RQ1

**Answered strongly, but currently misworded.**

The actual answer is:

> Input-embedding-gradient Langevin proposals do not provide useful token-revision guidance in the tested settings, because they omit the candidate’s direct conditional-fit term. A fuller output-side surrogate does contain useful guidance.

That is a very good answer.

## RQ2

**Answered clearly and convincingly.**

The proposal-term decomposition gives a genuine mechanism rather than merely an observed acceptance-rate drop.

## RQ3

**Answered for the tested variants.**

The thesis appropriately limits the generality, but it should separate GFlowNet policy failure from failure to reshape local input-gradient geometry.

## RQ4

**Answered cautiously.**

The direct answer on the frozen autoregressive landscape is negative. The diffusion-guidance material is a separate exploratory repair experiment.

# Final honest verdict

This version is scientifically richer, but editorially less controlled than the previous version.

The new one-hot and RoBERTa experiments are not unnecessary. They are arguably the most important experiments in the thesis because they transform the conclusion from:

> “autoregressive models do not provide a useful local direction”

into:

> “the useful information exists, but the conventional input-embedding derivative discards it, and revision further benefits from bidirectional context.”

That second conclusion is more precise, more constructive, and more defensible.

The thesis’s primary task now is not to add further experiments. It is to **rewrite the earlier causal story around the final evidence**.

The most important changes are:

1. Correct “five contributions” versus six.
2. Define the one-hot surrogate mathematically and avoid ambiguous “one-hot input gradient” terminology.
3. Reformulate RQ1 around the input-embedding proposal.
4. Replace “training objective is the cause” with “derivative choice and conditioning access are the operative variables”.
5. Distinguish the multiple failure mechanisms instead of forcing them into one unified cause.
6. Move the RoBERTa proposal ladder closer to the centre of the Results.
7. Shorten the main text by approximately 12 to 20 pages, especially the Related Work, exploratory guidance, Discussion overlap, and eleven-page Conclusion.

With those changes, I would consider the thesis **strong and potentially excellent**. In its present form, the evidence is better than the narrative consistency. The scientific findings are convincing, but the framing still partly belongs to an earlier version of the thesis that the newest experiments have superseded.