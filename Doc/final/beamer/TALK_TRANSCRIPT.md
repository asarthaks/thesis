# Defence talk: speaker transcript

**Energy-Guided Sampling for Controllable Text Generation: From Langevin Dynamics to Amortized Inference**

Companion to `Presentation.tex` (33 slides: 22 content, 11 backup). Baseline 25.25 minutes, with a
marked short set that brings it to 20.25 and a hard-cap variant at 19.75. See the timing card at
the end.

How to use this. The **Say** blocks are meant to be spoken more or less as written; they are
short sentences on purpose. The **Point at** lines tell you what to gesture to. **If asked
here** flags a question that usually arrives mid-slide, with the one-line answer, so you can
take it without losing your place. **Cut first** marks what to drop if you are running long.
Every number in the Say blocks is a thesis number; do not round them differently on the day.

One decision made once, for the whole talk: **you never say "the gradient does not work".** You
say "the input-embedding gradient does not work", every time. The whole contribution lives in
that qualifier, and an examiner who hears the unqualified version will attack it and be right
to. If you slip, correct yourself out loud. It reads as precision, not as fumbling.

**The shape of the talk.** Motivation, then the machinery, then the hypotheses, then the
experiments that eliminate them one at a time, then control. The audience should know, by minute
seven, exactly what three things could be at fault and in what order you are going to knock them
down. Everything after that is bookkeeping against a promise they have already heard.

---

## Slide 1: title (0.5 min)

> **Say.** Good morning. My thesis is about a family of methods that promise something very
> attractive: take a language model you already have, do not retrain it, and steer it toward
> any property you like at the moment you generate. I set out to build that. What I found is
> that the whole family rests on one assumption nobody had tested, that the assumption is false
> as usually implemented, and, more usefully, that it is false for a fixable reason. That last
> part is what I would like to spend most of the time on.

Do not read your name or the examiners off the slide. Straight into slide 2.

---

## Slide 2: the problem autoregressive models cannot solve (1.25 min)

> **Say.** Language models write left to right, one token at a time, and once a token is out it
> is fixed. There is no mechanism to go back. If the second half of a sentence reveals that the
> first half went wrong, the model can only continue or start over.
>
> That is fine when you just want a fluent continuation. It stops being fine when you want
> control, and the reason is precise. The things we actually want to control are properties of
> the *finished* text. Whether a paragraph is positive in sentiment is not a property of its
> first word. So when the model is choosing token three, the object you would have to check the
> constraint against does not exist yet.

**Point at** the diagram: the red arrow that cannot flow back.

> **Say.** You can of course generate a thousand candidates and keep the ones that satisfy the
> constraint. That works when the constraint is easy and is hopeless when it is not.

**Cut first.** The thousand-candidates sentence.

---

## Slide 3: the energy-based promise, and its one assumption (1.5 min)

> **Say.** So here is the elegant alternative. Stop generating token by token. Instead, define a
> probability distribution over whole sequences and sample from it.
>
> You write the energy as two pieces added together. The first is the negative log-likelihood
> of a frozen language model, and that supplies fluency. The second is any constraint you like,
> with a weight. Because it is just addition, you can swap the constraint at generation time and
> never touch the model. That is the plug-and-play promise, and it is genuinely appealing. COLD
> and MuCoLa are the two published systems closest to what I implemented.

**Point at** the jagged landscape cartoon.

> **Say.** Now the assumption. To sample from that energy with a gradient-guided method, you need
> two things to be true. You need a gradient that points, on average, toward better sequences.
> And you need to be able to follow it, step by step, across the space of texts. Put together,
> that is the presupposition that this energy is a landscape you can walk on: a surface with
> meaningful slopes whose downhill direction you can compute locally and trust.
>
> That is intuitive. It is also almost never tested in the papers that rely on it. Testing it is
> this thesis.

**If asked here:** *"Isn't this just PPLM or FUDGE?"* No. Those stay autoregressive and modify
the per-token decision, so they inherit the no-revision limit. This family abandons left-to-right
order entirely, which is what makes revising any position possible in principle.

---

## Slide 4: the machinery, Langevin dynamics (1.0 min)

New slide, and a quick one. It is a *sketch* of the method, not the sampler. The DLS proposal
formula deliberately does not live here; it belongs on slide 9, where the samplers are introduced.
Do not front-load it, and do not rush the Lipschitz bullet, which you come back to twice.

> **Say.** One slide of machinery before anything else.
>
> Suppose the state were continuous and the energy differentiable. Then there is a standard way to
> sample from it: Langevin dynamics. You step along the gradient of the log-density, and you add
> Gaussian noise. Two parts, and the interplay is what separates sampling from optimization. The
> drift walks uphill in probability. The noise stops it collapsing onto the single most likely
> point, because a distribution is more than its mode.

**Point at** the third bullet.

> **Say.** And note this one now, because I come back to it twice. The guarantees require the
> drift to be Lipschitz: a small step has to land roughly where the gradient at the starting point
> said it would.

**Point at** the picture on the right, then stop moving.

> **Say.** Which is the problem. Text is not continuous. There is no state halfway between two
> tokens, so there is no small step; the nearest other sequence is a whole token away. Every
> method in this family is some way of forcing this update onto discrete tokens, and each of them
> has to decide which gradient to take, and of what.

**Cut first.** The "distribution is more than its mode" clause.

---

## Slide 5: scoring a sequence, two arrays offset by one (1.0 min)

New slide. Pure mechanics, and deliberately not an argument. Everything on it is uncontroversial
and every examiner already knows it; its whole job is that slide 15 can point back at this picture
instead of building one under time pressure. Do not editorialize, do not hint at the finding, and
do not single out any one position. You come back here later and you want the return to land.

> **Say.** One slide on how the model actually puts a number on a sequence, because the argument
> later turns on a detail of this picture.
>
> The sequence goes in twice. Once along the bottom, as continuous embedding vectors, which is
> what the transformer actually reads. And once along the top, as target ids, which is a plain
> list of integers.

**Point at** the transformer, then the logits row.

> **Say.** The model reads the bottom row under a causal mask and emits, at every position, a
> vector of logits over the whole vocabulary.
>
> Then the alignment, which is two lines of code. The logits at a position are a prediction about
> the *next* position. So you drop the last logit vector, because there is nothing after the
> sequence for it to predict, and you drop the first id, because nothing predicted it. Everything
> else pairs up diagonally.

**Point at** the diagonal arrows.

> **Say.** Each pair gives one number: take the log softmax of the logit vector, and index it with
> the integer id. The energy is the sum of those. That is the entire computation.
>
> The thing to hold on to is the shape of it. The same sequence is present as continuous vectors
> and as a list of integers, and the energy needs both.

**If asked here:** *"Is that specific to GPT-2?"* No. It is the standard shifted cross-entropy of
any causal transformer. Llama is identical, which is why the results reproduce there.

**Cut first.** The two lines of code. Point at them rather than reading them out.

---

## Slide 6: research questions (0.75 min)

> **Say.** Four questions. The first, and the one the talk is mostly about, is whether that
> *input-embedding* gradient gives a useful proposal direction for revising a token, and if not,
> which local quantities do instead. The second is what the Metropolis-Hastings correction
> actually does in each of my two samplers. The third splits in two, because I conflated them at
> first: whether a GFlowNet learns a policy that generates high-reward text, and separately
> whether the energy it leaves behind is easier to navigate. And there is an extension question
> about adding a constraint, which is the application the whole programme was built for.

Read these briskly. The next slide is the one that earns its minute.

---

## Slide 7: three candidate explanations (1.0 min)

New slide, and the most useful sixty seconds in the first half. It sets the frame the audience
will hold for the entire results section.

> **Say.** Now, suppose the answer to the first question turns out to be no. That is a negative
> result, and a negative result is only worth anything if you can say *which* thing failed. There
> are three candidates, and they are not equivalent.
>
> Hypothesis A: the target is at fault. The frozen likelihood, evaluated on the counterfactual
> sequences a revision has to consider, is simply not something a local search can profit from.
> If that is true, the choice of proposal mechanism is beside the point.
>
> Hypothesis B: the model is at fault. Teacher forcing shapes next-token predictions on correct
> prefixes and never once asks the model to rank substitutions. On that reading the deficiency is
> in the training objective, and only a differently trained model could repair it.
>
> Hypothesis C: the parameterization is at fault. The information is in the frozen model, and the
> derivative I chose throws part of it away.

**Point at** the right-hand table.

> **Say.** And each one has a decisive test. A is settled by searching the identical energy
> without ever differentiating it. B is settled by re-differentiating the same frozen weights a
> different way. C is settled by a ladder of proposals inside one exact chain. Everything from
> here is that elimination, in that order.

**Never cut this slide.** If you are over time, take the seconds out of slide 19.

---

## Slide 8: the setup, one task, five energies, one metric (0.75 min)

Read it, do not perform it. It sits here, between the hypotheses and the samplers, because the
hypotheses raise the question "tested how?" and the answer has two halves: what I ran it on, then
what I ran on it. It also keeps the sampler slide adjacent to slide 10, which is about the very
proposal that slide introduces.

> **Say.** The setup, quickly. The task is masked-token recovery: I corrupt tokens in a held-out
> sequence and ask the sampler to put them back. Two hundred sequences per configuration, and the
> corruption is seeded so every arm sees exactly the same corrupted sentences, which is what makes
> the comparison paired.
>
> Five energy functions. A GPT-2 Large fine-tuned on ROCStories, which is the reference. Three
> GFlowNet-tuned variants of that same base, so the amortization half shares weights with the
> sampling half and is not confounded by a different network. And a Llama-3 8B as a
> cross-architecture control, which I never put on a shared numerical axis with the GPT-2 family
> because its tokenizer and embedding scale are different.
>
> The metric is a divergence: how far the model's predictions under my recovered fill sit from its
> predictions under the ground-truth fill. Zero is a perfect recovery by construction. And I fixed
> an equivalence margin in advance, 0.327, which is five percent of the policy mean and comfortably
> above the pipeline's own seed noise of 0.183.

**Cut first.** The seeding sentence and the margin sentence; both come back on slide 11 where
they are load-bearing.

---

## Slide 9: two samplers, implemented faithfully (1.75 min)

This slide now carries the DLS proposal formula, which used to sit on the Langevin slide. It is
the right place for it: the audience has seen Langevin, has seen how the energy is computed, and
is now being told how the two samplers actually use it. The last line is the one that matters.

> **Say.** Two samplers, because there are two ways to confront the fact that text is discrete,
> and I wanted a finding about the energy rather than about one implementation.
>
> The discrete sampler never leaves token space. The continuous one does the opposite: it relaxes
> into embedding space, runs genuine Langevin dynamics there, and projects back to the nearest
> token when it needs a sequence.
>
> Both get the Metropolis-Hastings correction, applied exactly. That matters, because the
> published methods in this family usually omit it, and without it you do not have a sampler at
> all, you have noisy gradient descent.

**Point at** the loop diagram, at the MH box.

> **Say.** The step size is not guessed. It comes from an oracle sweep that picks the best schedule
> using knowledge of the correct answer. Deliberately generous: if even an oracle-tuned schedule
> cannot make the gradient beat noise, the failure is not one of tuning. And the chain runs a
> fixed annealed schedule, fifty or a hundred steps, never stopped early.

**Point at** the proposal formula, and slow down. This is the object the whole talk is about.

> **Say.** Here is how the discrete sampler decides where to go. To score a candidate token
> against the one currently in place, it uses two terms. The second is a distance penalty, which
> keeps the proposal local. The first is the gradient dotted with the displacement to the
> candidate, and that is a first-order Taylor estimate of how much the log-likelihood would change
> if you actually made the substitution.
>
> So everything the sampler knows about which token is *good*, as opposed to merely *near*, is in
> that one alignment term. If it is uninformative, the whole method collapses to a
> distance-penalized random choice.
>
> And the g in it is a gradient with respect to the *input embedding* of the masked position.
> That object, that particular derivative, is what the rest of the talk is about.

**If asked here:** *"Why the input embedding and not something else?"* Because that is the
coordinate the published samplers in this family differentiate; they operate in embedding space.
Slide 15 is precisely the argument that this was the wrong choice.

**If asked here:** *"Does it run to convergence?"* Careful, and say this precisely: it runs a
fixed-length annealed schedule, not a convergence criterion. With the correction off it does
*not* converge at all, the divergence rises over the schedule. With the correction on it settles
within about twenty steps. What matters for the comparison is that no arm is stopped early
relative to another.

---

## Slide 10: what the calibrated proposal actually is (1.5 min)

This slide exists because it is the honest thing to put before the null. Say so.

> **Say.** Now the results, and the first one is a measurement about my own method rather than
> about the model. Before any comparison, I asked how sharp the proposal distribution actually is.
> It draws a token from a softmax, so its entropy tells you directly whether anything is shaping
> the draw.
>
> At the first step the entropy is 10.8248 nats. The ceiling, for a uniform draw over the whole
> 50,000-token vocabulary, is 10.8249. Over the entire fifty-step schedule the *lowest* it ever
> gets is 10.28, which is an effective support of about twenty-nine thousand tokens. And the
> gradient alignment term, the one from slide 9, contributes about nine thousandths of a nat of
> spread into a ten-nat budget.
>
> So the proposal I calibrated, and that the whole main grid draws from, is numerically a uniform
> draw over the vocabulary.

**Point at** the red block, and slow down. This is the slide's actual content.

> **Say.** Here is why that has to be said before the ablation rather than after it. If two
> proposals are both within a ten-thousandth of a nat of uniform, then comparing them is comparing
> two draws from the *same* distribution. Which means a *perfectly informative* gradient would
> have produced exactly the same null here. The measurement does not confirm the null. It bounds
> what the main grid was capable of detecting, and I would rather state that myself than have it
> pointed out to me.

**Point at** the "therefore" block.

> **Say.** So the comparison was re-run somewhere it could actually fail: a five-by-five sweep over
> step size and temperature, spanning proposal entropy from uniform all the way down to zero,
> deterministic. That is the configuration where the gradient term does shape the draw. I will give
> you the result on the next slide.

**If asked here:** *"Then why report the main grid at all?"* Because it is the configuration the
literature's own calibration procedure hands you, and because 145 configurations of it is what
establishes that the behaviour is not one unlucky cell.

**Note.** The uniform-draw control, half a percent against zero and 6.538 against 6.541, used to
live on this slide. It now sits on the ladder, slide 18, where the row above it makes it mean
something. If you want it here, say only: "and a literal uniform draw, in separate code,
reproduces this sampler to three significant figures."

---

## Slide 11: the central result (1.75 min)

> **Say.** The ablation. Three arms. The true gradient. A random direction rescaled to exactly
> the gradient's norm, so it differs only in *direction*. And a fully random vector, so it differs
> only in *magnitude*. That split is the point: it separates the two things a gradient could be
> contributing.

**Point at** the four-model table.

> **Say.** The differences are small and inconsistent in sign. Worse on one model by four
> hundredths, better on another by fifteen hundredths.

**Point at** the block.

> **Say.** And this is the statistical version. Paired on the same corrupted sentences, over a
> thousand pairs, under three different summaries of the same chains, because the last state of a
> Markov chain is a poor summary of it. Every interval sits inside the margin I fixed before the
> comparison. So the claim is not "we failed to find a difference". It is **equivalence,
> certified**. The gradient direction and noise of the same norm are the same thing on this task.

**Point at** the last line of the left column.

> **Say.** And the sweep I promised you on the previous slide, the one where the gradient does
> shape the proposal: the null holds in every one of the twenty-five cells, and recovery never
> exceeds two percent anywhere in it.

**If asked here:** *"Where does 0.327 come from?"* Five percent of the policy arm's mean
divergence, and it is calibrated against the pipeline's own noise: re-running the flagship under
four independent corruption seeds gives a standard deviation of 0.183, comfortably inside it.

**Cut first.** The four-model table commentary. The certification is the load-bearing half.

---

## Slide 12: Hypothesis A, is the target at fault? (1.0 min)

> **Say.** So the eliminations begin. Hypothesis A said the energy itself is unusable. If that
> were true, nothing operating on this energy would recover anything, whether it differentiated it
> or not. So I scored the same energy with methods that never take its gradient.
>
> Leaving the corrupted token in place gives 9.14. A single forward pass that reranks the top
> candidates by their effect on the sequence reaches 4.43, which is better than every one of the
> 145 Langevin configurations, whose best cell sits near 6.4. And a gradient-free Gibbs sampler on
> exactly the same energy reaches 6.69, comparable to the best of them, without a single backward
> pass.
>
> So Hypothesis A is out. The energy is searchable. It is just not searchable by this derivative,
> and it is searchable more cheaply without one.

**Point at** the scope line.

> **Say.** One qualification, which I would rather say than be asked. That acquittal is about
> *this* task. As an objective to *maximize* in open-ended generation the same likelihood is badly
> behaved, and I measure that too: the highest-likelihood text my models produce is the most
> degenerate text they produce. Those two statements are about different regimes and both are in
> the thesis.

---

## Slide 13: why, part one, the linearization fails before the first move (1.5 min)

> **Say.** So if the target is fine, why did the proposal fail? Back to the proposal on slide 9.
>
> Linearization is the move where you replace the true change a substitution would cause by its
> first-order Taylor term. That term, the gradient dotted with the displacement, is exactly what
> the proposal ranks candidates by. So the question is a measurable one: does that term agree with
> the truth? I computed both, over four hundred thousand candidate-sentence pairs.
>
> The Spearman correlation is minus one hundredth. Not weak. Absent. And below six hundredths on
> every one of the five energies.

**Point at** the Voronoi picture and then the radius plot.

> **Say.** And here is the length-scale problem underneath it. A Taylor expansion is only good in
> a small neighbourhood, and a Transformer is a strongly nonlinear function of its inputs, so that
> neighbourhood is small. The smallest move the sampler can possibly make is to *another token's*
> embedding, and on average that is 1.82 units away. The correlation has already decayed to zero
> well before 1.82. Every substitution the sampler can make is outside the range where its own
> approximation means anything.
>
> So the gradient is not imprecise here. It is inapplicable. And notice that this measurement is
> made on the proposal directly, so it does not depend on any step size or temperature. That is
> what carries the argument, not the grid.

---

## Slide 14: why, part two, the correction breaks the continuous sampler (1.5 min)

> **Say.** The second mechanism is separate, and it belongs to the continuous sampler only. This
> is the answer to my second research question. When the correction is on, that sampler rejects
> almost every move that actually changes a token. The question is whether it rejects them because
> they are bad moves or for some other reason, and the acceptance ratio has exactly two factors,
> so you can just look.

**Point at** the block.

> **Say.** For moves that cross a cell boundary, the target term is plus 4.60. Positive. Those
> moves *improve* the sequence. The proposal term is minus one thousand three hundred and
> twenty-five. It is three orders of magnitude larger and it kills them.
>
> The reason is the Lipschitz condition from slide 4. The reverse move has to be evaluated from
> the drift on the far side of the boundary, and over there the model is predicting a different
> token, so the drift points somewhere unrelated to the way back. The correction concludes the
> move is irreversible and rejects it.
>
> And I want to be careful about what that means. The correction is exact for the proposal it
> corrects. It is not malfunctioning. What fails is the smoothness the Langevin construction
> assumes underneath it. The same correction rescues the discrete sampler, where a within-cell
> move is its own reverse and is accepted with certainty.

**If asked here:** *"So the correction is the problem, and I should drop it?"* Dropping it is
what the literature does, and then every proposal is accepted and the state wanders hundreds of
units away from any real token. You get output, but it is an early-stopped optimizer, not a
sample from the energy you claimed.

---

## Slide 15: back to the two arrays, what the gradient cannot reach (1.5 min)

The hinge of the talk. It pays off slide 5 and it sets up slide 16. Say "come back to the two
arrays" out loud; the callback is the device. This is the same picture, one word shorter, with
three marks added. Deliver it slowly.

> **Say.** Which brings me to what I think is the actual contribution. Hypothesis B said the model
> does not contain what a local proposal needs. I want to show you that it does, and that the
> derivative was throwing it away.
>
> So come back to the two arrays. The samplers differentiate with respect to the bottom row, so
> follow where that derivative can actually go.

**Point at** the green arrow going up from s.

> **Say.** Forward it is fine. The soft token genuinely feeds the transformer, so it shapes the
> logits at position three and everything after, and the chain rule reaches all of it. That is the
> part about how a token affects what comes later.

**Point at** the red cross on the mask.

> **Say.** Backward there is nothing. Attention is causal, so nothing from position three flows
> left. The logits at position two were finished before the model ever read s.

**Point at** the red arrow from those logits up to the id.

> **Say.** And that matters, because the score of position three *is* the entry you pull out of
> those logits at position two. The only thing position three contributes to its own score is that
> integer id. An integer is not something you differentiate. So that term is not small and it is
> not noisy. It is exactly zero.
>
> Concretely: the gradient knows what "barked" does to "loudly", and knows nothing whatsoever
> about whether "barked" was a good word to follow "the dog".
>
> Decomposing the true energy change, the visible part averages 24.2 nats and the invisible part
> averages 15.0.

**Point at** the two boxes.

> **Say.** The repair follows directly. Stop asking autograd to differentiate an integer lookup.
> Relax the token indicator itself, in both of its roles, and the derivative comes out in closed
> form as two terms. The blue box is exactly what the old gradient computed. The amber box is the
> candidate's own conditional log-probability, which you read straight off the logits row the
> model has already produced. One forward pass, frozen weights, nothing retrained.
>
> I want to be precise, because the name matters. This is not "the output side instead of the
> input side". It is the self term **added to** the same future term. And it is not the ordinary
> one-hot input gradient either; that one, by the chain rule, gives you the future term alone.
>
> That missing term is what the correlation of point-zero-three was measuring.

**If asked here:** *"Isn't the self term just the language model's own prediction, so you have
stopped doing energy-based sampling?"* No. It is one of the two terms of the energy's own
derivative, not a substitute for the energy. The suffix term is still there and still comes from
the gradient. The constraint still enters the same way.

**If asked here:** *"Does MuCoLa have the same problem?"* Yes, and backup 33 is the table. They
differentiate the input embedding, so they discard the same term. Say it as a structural
consequence, not as a criticism of the paper.

**Cut first.** The 24.2 and 15.0 decomposition, then the "barked" sentence. The picture carries
the argument without either.

## Slide 16: Hypothesis B, same model, same sampler, different derivative (1.5 min)

**Point at** the equation, briefly. It is the previous slide written out in general form: the
self term the derivative discarded, plus the future term it kept.

**Point at** the results table.

> **Say.** And it works. Same frozen weights. Same sampler I already had. Nothing retrained. The
> correlation at admissible distances goes from 0.03 to 0.73. Exact recovery goes from zero to
> forty percent on GPT-2 Large, and it reaches forty-one on Llama-3 8B. So the strong form of
> Hypothesis B is rejected: a usable local direction was in the frozen model the whole time, and
> it survived the training objective untouched.

**Say the asymmetry if you have the second, and always if the Llama number is challenged.** On
GPT-2 the two surrogates are compared cell for cell across one step-size and temperature grid, so
that is a matched comparison. On Llama no input-embedding run was made at the surrogate-driven
setting, so there the forty-one percent is against every Llama configuration that *was* run, all
of which recover nothing. It is the weaker of the two comparisons and the thesis says so.

**Point at** the bottom line.

> **Say.** One more thing, and it is the finding I found most humbling. At the temperature I had
> calibrated on, five, even *this* proposal collapses to two percent, because that temperature
> flattens the softmax whatever you feed it. The binding constraint on my entire main grid was a
> temperature, not a model and not an objective.

**If asked here:** *"Is the Llama comparison matched?"* No, and the thesis states it. The
matched Llama cell is the calibrated one, where both surrogates recover zero percent and the
token-indicator arm is in fact marginally worse on divergence, 4.108 against 3.898. So the
forty-one percent is bought by the configuration and the derivative together, not by the
derivative alone. On GPT-2, where the grid is matched, the derivative alone is the variable.

**If asked here:** *"So you had a working method and did not use it?"* It was found by
re-analysing measurements I had already taken, late, after the grid was complete. It is reported
as the constructive finding of the thesis rather than folded back in as the method, and the
limitation section says so plainly.

---

## Slide 17: the exact case, the final-token gradient is zero (1 min)

> **Say.** That claim can be checked exactly at one position, and it is the cleanest version of
> the whole diagnosis.
>
> Take the last token of a sequence. Under causal attention, its input embedding only feeds the
> logits that would predict a *next* token, and there is no next token, so those logits appear in
> no term of the likelihood. The gradient with respect to that embedding is therefore exactly
> zero. Not small. Zero. And I can see it in the live sampler: the gradient norm it consumes is
> 0.0000 in both mean and maximum over about fifteen thousand evaluations.
>
> Meanwhile the energy at that same position is maximally informative. The difference in energy
> between two candidate final tokens is exactly the model's own conditional log-ratio between
> them.

**Point at** the position table.

> **Say.** So at that one position, the energy tells you everything and the gradient tells you
> nothing, provably. The Langevin arms recover zero percent; methods that read the energy recover
> between a third and a half.

This is your strongest slide rhetorically. Slow down.

---

## Slide 18: Hypothesis C, the conditioning ladder (1.5 min)

> **Say.** The last experiment of the diagnosis tests what is left. Hypothesis C also predicted
> that a proposal allowed to condition on the *right* context should do better still. So: same
> Metropolis chain, same exact energy from the frozen GPT-2, same two hundred sentences, same
> corruption. Only the proposal changes.

**Point at** the ladder, bottom to top.

> **Say.** Read it upward. A proposal that sees nothing about the candidate's fit, the
> input-embedding gradient, gets zero. Its norm-matched random control gets zero. A uniform draw
> gets half a percent, and in fact a literal uniform draw, written in separate code, reproduces my
> flagship Langevin sampler to three significant figures, 6.538 against 6.541. So the bottom three
> rungs are the null, reproduced independently.
>
> Now add the model's own left-to-right conditional, read from the output side: twenty-three and
> a half percent. Same frozen model whose gradient got zero.
>
> Add the right context. A score-trained diffusion model gets thirty-nine. And a plain masked
> language model, RoBERTa, which was never trained with any score objective, gets forty-four and a
> half. It beats the diffusion model.

**Point at** the closing line.

> **Say.** Which kills the explanation I had been carrying for months. I thought the training
> objective was the cause and diffusion was going to prove it. Score training turns out to be
> sufficient and not necessary. What the ordering tracks is what the proposal is allowed to look
> at.
>
> And two honesties about that. First, this is an ordering, not an isolation: these models also
> differ in scale, corpus and architecture, and only the tokenizer and the chain are held fixed.
> Second, the thirty-nine percent should be read against the gradient-free comparator on the same
> task, which is thirty-three. It is a gain over thirty-three, not a gain over nothing.

---

## Slide 19: GFlowNet amortization (0.75 min)

> **Say.** Two results that stand apart from that argument, briefly. First, the other route I
> tried. A GFlowNet never differentiates its reward, it only evaluates it, so a gradient pathology
> should be invisible to it.
>
> Two answers, and they are separate questions. The policy does reach high reward, but by
> exploiting the reward rather than by writing well: with the unnormalized reward it learns that
> shorter is better, and with the normalized reward it finds high-probability gibberish. And
> separately, tuning moved the energy tens of nats, and yet running my sampler on the tuned energy
> is indistinguishable from running it on the base.

**Cut first.** This whole slide. Say one sentence: "amortization did not escape it, and the
details are in the thesis."

---

## Slide 20: the extension, does an additive constraint steer anything? (1.25 min)

New slide, and it closes the loop with slide 3. The audience was promised plug-and-play control
in minute three; this is where they find out what happened to it.

> **Say.** And finally the thing the whole programme was built to deliver: adding a constraint.
> This is the extension question, and I report it last because it applies machinery whose
> prerequisite everything before it was testing. A constraint added to a landscape a sampler
> cannot navigate is being added to nothing.
>
> The setup is the plug-and-play one: a sentiment classifier head on the frozen base, added to the
> energy with a weight. One mechanism has to be flagged before any number. The raw steering gains
> are dominated by an unconditional sentiment drift: every arm, including the ones with no
> constraint gradient at all, flips sign when you switch the target label. So the statistic is the
> paired contrast, constraint-only minus randomized-constraint, which cancels that drift.

**Point at** the table.

> **Say.** Measured that way it is setup-dependent. On my discrete sampler the contrast is
> essentially zero on both labels: the constraint direction buys no control there. On a
> MuCoLa-style continuous baseline doing free continuation it is plus twenty-seven and plus
> thirty-seven points, which does say the constraint gradient's direction carries some signal,
> unlike the fluency gradient, whose direction carried none anywhere.
>
> With a caveat I have to give you: with no fluency term holding it down, the constraint-only arm
> leaves the manifold of natural text, and a sentiment classifier scoring that text is reporting
> on a distribution it was never trained for.

**Point at** the right-hand column.

> **Say.** I also looked at what happens once the proposal *can* navigate, using the diffusion
> model as the carrier. There the constraint direction does move things: plus nine to plus
> twenty-six points on the guiding classifier's own verdict. But transfer to an independent judge
> is bounded by how far the two instruments agree about the text at all, which runs from about
> sixty percent off-domain to eighty on real on-domain text. And an on-domain trust region removes
> the fluency cost that unconstrained guidance otherwise incurs. That is a repair experiment on
> another landscape, not an answer about this one.

**Point at** the closing line.

> **Say.** So the honest headline is that on the frozen autoregressive energy, plug-and-play
> constraint steering failed, and it failed for the reason the rest of the talk established.

**Cut first.** The entire right-hand column. Keep the table and the closing line.

---

## Slide 21: takeaways (1 min)

> **Say.** Four things.
>
> One. The input-embedding gradient of a frozen autoregressive likelihood is equivalent to a
> norm-matched random direction, certified over a thousand paired sequences at a margin fixed in
> advance.
>
> Two. That is not because the target is unusable, and not because the model lacks the
> information. The same frozen weights, differentiated in the token-indicator coordinates instead,
> recover forty percent and forty-one percent.
>
> Three. What buys the rest is access to the right context, not a score objective. RoBERTa above
> SEDD above the autoregressive conditional.
>
> Four. And control follows navigation. The constraint direction buys nothing where the proposal
> cannot move, and carries signal where it can.

**Point at** the closing line, and say it deliberately.

> **Say.** So the practical advice is not the one I expected to give. It is not "evaluate the
> energy and do not differentiate it". A derivative of the same frozen model does work. What you
> must not do is take it in the input-embedding coordinates. Differentiate the right object, or
> propose from the output side.
>
> Thank you.

---

## Timing card

| # | slide | baseline | short | cumulative (short) |
|---|---|---:|---:|---:|
| 1 | title | 0.5 | 0.5 | 0.5 |
| 2 | the revision problem | 1.25 | 0.9 | 1.4 |
| 3 | the energy promise and its assumption | 1.5 | 1.25 | 2.65 |
| 4 | the machinery, Langevin dynamics | 1.0 | 0.75 | 3.4 |
| 5 | scoring a sequence, two arrays offset by one | 1.0 | 0.6 | 4.0 |
| 6 | research questions | 0.75 | 0.75 | 4.75 |
| 7 | three candidate explanations | 1.0 | 1.0 | 5.75 |
| 8 | the setup, five energies | 0.75 | 0.4 | 6.15 |
| 9 | two samplers, and the DLS proposal | 1.75 | 1.5 | 7.65 |
| 10 | the proposal is numerically uniform | 1.5 | 1.5 | 9.15 |
| 11 | the central result, certified equivalence | 1.75 | 1.75 | 10.9 |
| 12 | Hypothesis A, the target | 1.0 | 0.75 | 11.65 |
| 13 | why 1, linearization | 1.5 | 1.5 | 13.15 |
| 14 | why 2, the MH breakdown | 1.5 | 1.0 | 14.15 |
| 15 | back to the two arrays, what the gradient reaches | 1.5 | 1.0 | 15.15 |
| 16 | Hypothesis B, different derivative | 1.5 | 1.25 | 16.4 |
| 17 | the exact final-position case | 1.0 | 0.5 | 16.9 |
| 18 | Hypothesis C, the conditioning ladder | 1.5 | 1.5 | 18.4 |
| 19 | GFlowNet | 0.75 | 0.25 | 18.65 |
| 20 | the extension, constraint steering | 1.25 | 0.6 | 19.25 |
| 21 | takeaways | 1.0 | 1.0 | 20.25 |
| | **total** | **25.25** | **20.25** | |

**The short set**, in the order to apply it: slide 19 to one sentence; slide 20 to the table and
the closing line; slide 17 to the zero-gradient statement alone; slide 14 without the "what that
means" paragraph; slide 8 read straight off; slide 16 as numbers only, since 15 now carries the
derivation; slide 2 without the thousand-candidates sentence; slide 5 without the two code lines; slide 12
without the qualification paragraph; slide 4 without the mode clause and the Lipschitz aside;
slide 3 without the PPLM contrast.

**Under a hard 20-minute cap**, drop slide 17 entirely, since slide 15 already makes the same
point qualitatively. That alone lands at 19.75.

**Never compress 7, 10, 15, 16 or 18.** Slide 7 is the frame the whole results section hangs on,
10 is the honesty that protects the null, and 15, 16 and 18 are the contribution.

**Running short?** Expand slide 17, which has the most headroom, or bring up the temperature
backup.

---

## Four sentences to have word-perfect

You will be asked to summarize under pressure. Rehearse these.

1. **The null.** "Following the input-embedding gradient is equivalent to following a
   norm-matched random direction, certified at a margin fixed in advance over a thousand paired
   sequences."
2. **Why.** "That derivative sees a candidate token only as an input to future predictions and
   never as the target of the prediction that was made about it, and the invisible half averages
   fifteen nats."
3. **The fix.** "Relax the token indicator in both roles instead, and the missing term appears in
   closed form as the model's own conditional. Same weights, same sampler, zero to forty percent."
4. **The attribution.** "Score training is sufficient and not necessary. A masked language model
   with no score objective does best, so the ordering tracks conditioning access, not the training
   objective."

---

## Four traps to avoid saying

- **"The gradient carries no usable signal."** Always "the input-embedding gradient". Your own
  data refutes the unqualified version.
- **"The energy is bad" / "the energy is fine."** Both, unscoped, are wrong. It is a serviceable
  thing to *evaluate* on this task and a badly behaved thing to *maximize* in free generation.
- **"The sampler runs to convergence."** It runs a fixed annealed schedule. Without the correction
  it does not converge at all. The defensible claim is that no arm is stopped early relative to
  another.
- **"Evaluate, do not differentiate."** This was the earlier slogan and it is superseded. The
  corrected form is on slide 21.
