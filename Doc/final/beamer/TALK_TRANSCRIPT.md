# Defence talk: speaker transcript

**Energy-Guided Sampling for Controllable Text Generation: From Langevin Dynamics to Amortized Inference**

Companion to `Presentation.tex` (25 slides: 16 content, 9 backup). Target 20 minutes.

How to use this. The **Say** blocks are meant to be spoken more or less as written; they are
short sentences on purpose. The **Point at** lines tell you what to gesture to. **If asked
here** flags a question that usually arrives mid-slide, with the one-line answer, so you can
take it without losing your place. **Cut first** marks what to drop if you are running long.
Every number in the Say blocks is a thesis number; do not round them differently on the day.

One decision made once, for the whole talk: **you never say "the gradient does not work".** You
say "the input-embedding gradient does not work", every time. The whole contribution lives in
that qualifier, and an examiner who hears the unqualified version will attack it and be right
to. If you slip, correct yourself out loud. It reads as precision, not as fumbling.

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

## Slide 2: the problem autoregressive models cannot solve (1.5 min)

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

## Slide 4: research questions (1 min)

> **Say.** Four questions. The first, and the one the talk is mostly about, is whether the
> *input-embedding* gradient of a frozen autoregressive likelihood gives a useful proposal
> direction for revising a token, and if not, which local quantities do instead. The second is
> what the Metropolis-Hastings correction actually does in each of my two samplers. The third
> splits in two, because I conflated them at first: whether a GFlowNet learns a policy that
> generates high-reward text, and separately whether the energy it leaves behind is easier to
> navigate. And there is an extension question about adding a constraint, which is the
> application the whole programme was built for.

**Point at** the bottom line.

> **Say.** And the shape of the answer. A negative result like mine admits three explanations,
> and I test them in order. A: the target energy itself is unusable. B: the model does not
> contain the information a local proposal needs. C: the information is there and the derivative
> I chose throws part of it away. The results chapter eliminates A, then B, and lands on C.

That last framing is the single most useful thing you say in the first five minutes. Do not rush it.

---

## Slide 5: what the main grid's proposal actually is (1.5 min)

This slide exists because it is the honest thing to put before the null. Say so.

> **Say.** Before any comparison, one measurement, and it changes how you should read
> everything after it. I asked how sharp the proposal distribution actually is. It draws a token
> from a softmax, so its entropy tells you directly whether anything is shaping the draw.
>
> At the first step the entropy is 10.8248 nats. The ceiling, for a uniform draw over the whole
> 50,000-token vocabulary, is 10.8249. Over the entire fifty-step schedule the *lowest* it ever
> gets is 10.28, which is an effective support of about twenty-nine thousand tokens. The gradient
> term contributes about nine thousandths of a nat of spread into a ten-nat budget.
>
> So the proposal I calibrated, and that the whole main grid uses, is numerically a uniform draw.

**Point at** the small table.

> **Say.** And here is the control that confirms it from the other direction. If I replace the
> gradient proposal with a literal uniform draw, in a separate implementation, I get half a
> percent exact recovery against zero, and a final divergence of 6.538 against 6.541. Three
> significant figures. The gradient-guided sampler and a coin flip are the same object here.
>
> That has an uncomfortable consequence I want to state myself rather than have it pointed out:
> at this configuration a *perfectly informative* gradient would have produced the same null. So
> this configuration cannot settle the question. That is why I re-ran the comparison across a
> five-by-five sweep of step size and temperature, spanning entropy from uniform all the way down
> to zero, deterministic. The null holds in every cell, and recovery never exceeds two percent
> anywhere.

**If asked here:** *"Then why report the main grid at all?"* Because it is the configuration the
literature's calibration procedure gives you, and because 145 configurations of it is what
establishes that the behaviour is not one unlucky cell.

---

## Slide 6: two samplers, implemented faithfully (1.5 min)

> **Say.** Two samplers, because there are two ways to confront the fact that text is discrete,
> and I wanted a finding about the energy rather than about one implementation.
>
> The discrete sampler never leaves token space. It uses the gradient only as a score for
> building a categorical proposal over which token to put in the gap. The continuous sampler does
> the opposite: it relaxes into embedding space, runs genuine Langevin dynamics there, and
> projects back to the nearest token when it needs a sequence.
>
> Both get the Metropolis-Hastings correction, applied exactly. That matters, because the
> published methods in this family usually omit it, and without it you do not have a sampler at
> all, you have noisy gradient descent.

**Point at** the loop diagram, at the MH box.

> **Say.** The step size is not guessed. It comes from an oracle sweep, fifty values, that picks
> the best schedule using knowledge of the correct answer. That is deliberately generous: if even
> an oracle-tuned schedule cannot make the gradient beat noise, the failure is not one of tuning.

---

## Slide 7: the central result (2 min)

> **Say.** Now the ablation. Three arms. The true gradient. A random direction rescaled to
> exactly the gradient's norm, so it differs only in *direction*. And a fully random vector, so it
> differs only in *magnitude*. That split is the point: it separates the two things a gradient
> could be contributing.

**Point at** the four-model table.

> **Say.** The differences are small and inconsistent in sign. Worse on one model by four
> hundredths, better on another by fifteen hundredths.

**Point at** the block.

> **Say.** And this is the statistical version. Paired on the same corrupted sentences, over a
> thousand pairs, under three different summaries of the same chains, because the last state of a
> Markov chain is a poor summary of it. Every interval sits inside a margin I fixed before the
> comparison. So the claim is not "we failed to find a difference". It is **equivalence,
> certified**. The gradient direction and noise of the same norm are the same thing on this task.

**If asked here:** *"Where does 0.327 come from?"* Five percent of the policy arm's mean
divergence, and it is calibrated against the pipeline's own noise: re-running the flagship under
four independent corruption seeds gives a standard deviation of 0.183, comfortably inside it.

**Cut first.** The four-model table commentary. The certification is the load-bearing half.

---

## Slide 8: why, part one, the linearization fails before the first move (1.5 min)

> **Say.** So why. The discrete proposal ranks candidate tokens by a first-order Taylor term: the
> gradient dotted with the embedding displacement. That is an approximation to how much the
> sequence log-likelihood would change if you made the substitution. I measured the approximation
> against the truth, over four hundred thousand candidate-sentence pairs.
>
> The Spearman correlation is minus one hundredth. Not weak. Absent. And below six hundredths on
> every one of the five energies I tested.

**Point at** the Voronoi picture and then the radius plot.

> **Say.** And here is the length-scale problem underneath it. A Taylor expansion is only good in
> a small neighbourhood, and a Transformer is a strongly nonlinear function of its inputs, so that
> neighbourhood is small. The smallest move the sampler can possibly make is to another token's
> embedding, and on average that is 1.82 units away. The correlation has already decayed to zero
> well before 1.82. Every substitution the sampler can make is outside the range where its own
> approximation means anything. So the gradient is not imprecise here. It is inapplicable.

---

## Slide 9: why, part two, the correction breaks the continuous sampler (1.5 min)

> **Say.** The second mechanism is separate, and it belongs to the continuous sampler only. When
> the correction is on, it rejects almost every move that actually changes a token. The question
> is whether it rejects them because they are bad moves or for some other reason, and the
> acceptance ratio has exactly two factors, so you can just look.

**Point at** the block.

> **Say.** For moves that cross a cell boundary, the target term is plus 4.60. Positive. Those
> moves *improve* the sequence. The proposal term is minus one thousand three hundred and
> twenty-five. It is three orders of magnitude larger and it kills them.
>
> The reason is geometric. The reverse move has to be evaluated from the drift on the far side of
> the boundary, and over there the model is predicting a different token, so the drift points
> somewhere unrelated to the way back. The correction concludes the move is irreversible and
> rejects it.
>
> And I want to be careful about what that means. The correction is exact for the proposal it
> corrects. It is not malfunctioning. What fails is the smoothness the Langevin construction
> assumes underneath it.

**If asked here:** *"So the correction is the problem, and I should drop it?"* Dropping it is
what the literature does, and then every proposal is accepted and the state wanders hundreds of
units away from any real token. You get output, but it is an early-stopped optimizer, not a
sample from the energy you claimed.

---

## Slide 10: the target is not what fails (1 min)

> **Say.** Now the eliminations. Hypothesis A said the energy itself is unusable. If that were
> true, nothing operating on this energy would recover anything. So I scored the same energy with
> methods that never take its gradient.
>
> A single forward pass that reranks the top candidates reaches 4.43, which is better than every
> one of the 145 Langevin configurations. A gradient-free Gibbs sampler reaches 6.69, comparable
> to the best of them, without a single backward pass.
>
> So Hypothesis A is out. The energy is searchable. It is just not searchable by this derivative.

**Point at** the scope line.

> **Say.** One qualification, which I would rather say than be asked. That acquittal is about
> *this* task. As an objective to *maximize* in open-ended generation the same likelihood is
> badly behaved, and I measure that too: the highest-likelihood text my models produce is the
> most degenerate text they produce. Those two statements are about different regimes and both
> are in the thesis.

---

## Slide 11: GFlowNet amortization (1 min)

> **Say.** Before the constructive half, the other route I tried. A GFlowNet never differentiates
> its reward, it only evaluates it, so a gradient pathology should be invisible to it.
>
> Two answers, and they are separate questions. The policy does reach high reward, but it does
> so by exploiting the reward rather than by writing well: with the unnormalized reward it learns
> that shorter is better, because every extra token costs about 1.12 nats, and with the
> normalized reward it finds high-probability gibberish. And separately, tuning moved the energy a
> long way, tens of nats, dropping the base-to-tuned correlation to 0.77, and yet running my
> sampler on the tuned energy is indistinguishable from running it on the base.

**Cut first.** This whole slide, if you are over time. Say one sentence: "amortization did not
escape it, and the details are in the thesis."

---

## Slide 12: the exact case, the final-token gradient is zero (1 min)

> **Say.** Here is the cleanest version of the whole diagnosis, at one position where I can be
> exact instead of statistical.
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
> nothing. The Langevin arms recover zero percent; methods that read the energy recover between
> a third and a half.

This is your strongest slide rhetorically. Slow down.

---

## Slide 13: the headline, same model, same sampler, different derivative (2 min)

> **Say.** Which brings me to what I think is the actual contribution.
>
> Go back to why the gradient is blind. A token enters the likelihood in two different roles. It
> is an *input*, which the model uses to predict everything after it. And it is a *target*, the
> thing the previous position was trying to predict. The input-embedding gradient only sees the
> first role, because the second one enters through a discrete index into the output softmax and
> a derivative with respect to a continuous embedding cannot reach it.
>
> And I can put a number on what is missing. Decomposing the true energy change, the part the
> gradient sees averages 24.2 nats and the part it cannot see averages 15.0. The invisible half is
> not a rounding error.

**Point at** the equation.

> **Say.** So relax the token indicator itself, in both of its roles, rather than relaxing its
> embedding. Do that and the derivative comes out in closed form as two terms. The second is
> exactly what the old gradient computed. The first is the candidate's own conditional
> log-probability, which is one forward pass away, and it is the piece that was being thrown out.
>
> I want to be precise, because the name matters. This is not "the output side instead of the
> input side". It is the self term **added to** the same future term. And it is not the ordinary
> one-hot input gradient either; that one, by the chain rule, gives you the future term alone.

**Point at** the results table.

> **Say.** And it works. Same frozen weights. Same sampler I already had. Nothing retrained. The
> correlation at admissible distances goes from 0.03 to 0.73. Exact recovery goes from zero to
> forty percent on GPT-2 Large, and it reaches forty-one on Llama-3 8B.

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

## Slide 14: the conditioning ladder (1.5 min)

> **Say.** The last experiment holds everything fixed except the one thing I want to vary. Same
> Metropolis chain. Same exact energy from the frozen GPT-2. Same two hundred sentences, same
> corruption. Only the proposal changes.

**Point at** the ladder, bottom to top.

> **Say.** Read it upward. A proposal that sees nothing about the candidate's fit, the
> input-embedding gradient, gets zero. Its norm-matched random control gets zero. A uniform draw
> gets half a percent. So the bottom three rungs are the null, reproduced from separate code.
>
> Now add the model's own left-to-right conditional, read from the output side: twenty-three and
> a half percent. Same frozen model whose gradient got zero.
>
> Add the right context. A score-trained diffusion model gets thirty-nine. And a plain masked
> language model, RoBERTa, which was never trained with any score objective, gets forty-four and a
> half. It beats both diffusion models.

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

## Slide 15: takeaways (1 min)

> **Say.** Three things.
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

**Point at** the closing line, and say it deliberately.

> **Say.** So the practical advice is not the one I expected to give. It is not "evaluate the
> energy and do not differentiate it". A derivative of the same frozen model does work. What you
> must not do is take it in the input-embedding coordinates. Differentiate the right object, or
> propose from the output side.
>
> Thank you.

---

## Timing card

| # | slide | min | cumulative |
|---|---|---:|---:|
| 1 | title | 0.5 | 0.5 |
| 2 | the revision problem | 1.5 | 2.0 |
| 3 | the energy promise and its assumption | 1.5 | 3.5 |
| 4 | research questions and the three hypotheses | 1.0 | 4.5 |
| 5 | the proposal is numerically uniform | 1.5 | 6.0 |
| 6 | two samplers | 1.5 | 7.5 |
| 7 | the central result, certified equivalence | 2.0 | 9.5 |
| 8 | why 1, linearization | 1.5 | 11.0 |
| 9 | why 2, the MH breakdown | 1.5 | 12.5 |
| 10 | the target is not what fails | 1.0 | 13.5 |
| 11 | GFlowNet | 1.0 | 14.5 |
| 12 | the exact final-position case | 1.0 | 15.5 |
| 13 | the headline, different derivative | 2.0 | 17.5 |
| 14 | the conditioning ladder | 1.5 | 19.0 |
| 15 | takeaways | 1.0 | 20.0 |

**Running long?** Drop slide 11 to one sentence (saves 1.0) and trim slide 7 to the certification
block only (saves 0.5). Never compress 5, 13 or 14: 5 is the honesty that protects the null, and
13 and 14 are the contribution.

**Running short?** Expand slide 12, which has the most headroom, or bring up backup 22 on the
temperature.

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

## Three traps to avoid saying

- **"The gradient carries no usable signal."** Always "the input-embedding gradient". Your own
  data refutes the unqualified version.
- **"The energy is bad" / "the energy is fine."** Both, unscoped, are wrong. It is a serviceable
  thing to *evaluate* on this task and a badly behaved thing to *maximize* in free generation.
- **"Evaluate, do not differentiate."** This was the earlier slogan and it is superseded. The
  corrected form is on slide 15.
