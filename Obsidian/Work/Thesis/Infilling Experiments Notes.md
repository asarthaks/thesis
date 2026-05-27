# Notes on Infilling Experiments, GFlowNet Comparisons, and MuCoLa

  

Prepared for later reference.

  

---

  

## Question 1) For our infilling experiments with Llama 3 on WikiText-2, should we have fine-tuned the model on the dataset we were testing on?

  

### Short answer

  

**Not necessarily.** It depends on what claim you want the experiment to support.

  

- If your thesis question is about **training-free inference-time control**, then using a **pretrained Llama 3 directly on WikiText-2** is a defensible and even preferable setup.

- If your question is about whether **amortized inference / GFlowNet-style training can outperform Langevin-based inference for infilling on a specific domain**, then you should also include a setup where the underlying model is **domain-adapted** before the infilling method is tested.

  

### Why the answer is nuanced

  

In your current thesis, the infilling experiments are explicitly framed as a **training-free** study of Langevin-based decoding. You describe the experiments as being run with a **pre-trained Llama-3 (8B) model on the WikiText-2 dataset**, with randomly corrupted tokens that the sampler must recover over 50 steps. You also frame the thesis goal as testing whether “rigorous, training-free Langevin dynamics” can control LLMs at inference time, and later conclude that the gradients of standard autoregressive LLMs are too jagged and uninformative for reliable MCMC-style control. That means the current setup is aligned with the methodological goal of the thesis: it isolates **inference-time behavior** without changing model weights.

  

### When fine-tuning would be appropriate

  

Fine-tuning becomes appropriate when the experiment changes from:

  

> “Can a pretrained AR LM be controlled at inference time using Langevin dynamics?”

  

to:

  

> “Can a model trained or adapted for the target data distribution provide a better posterior / reward landscape for infilling?”

  

That second question is exactly what Hu et al. do in the GFlowNet paper for story infilling. They first adapt GPT-2 Large to ROCStories because the pretrained model was not good enough at assigning high likelihood to plausible stories, and then they use the adapted model as the base/reward model for GFlowNet amortized inference.

  

### My recommendation for your thesis framing

  

For **Llama 3 + WikiText-2**, I would **not retroactively say you should have fine-tuned it first** if the chapter is meant to evaluate **training-free Langevin decoding**. In fact, doing so would partially weaken the central claim, because it would no longer isolate the quality of the inference-time sampler.

  

Instead, I would present it like this:

  

- **Current thesis experiments**: properly test the training-free hypothesis.

- **Follow-up amortized-inference experiments**: should add a domain-adapted model, because amortized methods depend much more strongly on the quality of the base model’s probability landscape.

  

### Best experimental extension

  

A strong extension would be to run **two tracks**:

  

1. **Training-free track**

   - Base Llama 3 on WikiText-2

   - Langevin methods, AR baseline, maybe Gibbs

   - Goal: test inference-time control only

  

2. **Amortized/domain-adapted track**

   - Base model adapted on WikiText-2 infilling-style data or plain WikiText-2 LM data

   - GFlowNet or other amortized policy

   - Goal: test whether training improves the proposal / posterior approximation

  

That gives you a clean scientific story instead of mixing two different questions.

  

### Practical caveat

  

If you fine-tune on WikiText-2 and also evaluate on WikiText-2, make sure the evaluation split is clean. Otherwise, the comparison becomes confounded by memorization or domain leakage.

  

---

  

## Question 2) For comparability, should the original infilling experiments using Langevin dynamics use base GPT-2, GPT-2 fine-tuned on ROCStories, or GPT-2 fine-tuned on WikiText-2?

  

### Short answer

  

For a **fair comparison**, the Langevin baseline and the GFlowNet model should use the **same underlying base/reward model family and the same domain adaptation stage**.

  

So if the GFlowNet experiment uses:

  

- **GPT-2 fine-tuned on ROCStories**,

  

then the directly comparable Langevin baseline should also use:

  

- **the same ROCStories-fine-tuned GPT-2**.

  

Using base GPT-2 against a GFlowNet built on top of ROCStories-tuned GPT-2 is not apples-to-apples. Using WikiText-2-fine-tuned GPT-2 for comparison to a ROCStories-based GFlowNet is also mismatched, because the domain changes.

  

### Why this matters

  

In both Langevin-style methods and GFlowNet amortization, the language model is not just a generator. It defines the **energy**, **reward**, or **posterior landscape**. If you change the underlying LM, you change the problem itself.

  

For story infilling, Hu et al. define the posterior through the LM likelihood of the complete sequence. That means the quality of the inferred infill depends directly on the base LM’s ability to model that domain. If one method sees ROCStories-adapted probabilities and the other sees generic GPT-2 probabilities, then differences in performance could come from the LM, not from Langevin vs GFlowNet.

  

### What is the cleanest comparison?

  

Use a matrix like this:

  

#### A. Domain-matched comparison for ROCStories

- **Base GPT-2** + Langevin

- **ROCStories-fine-tuned GPT-2** + Langevin

- **ROCStories-fine-tuned GPT-2** + GFlowNet fine-tuning

  

This lets you separate two effects:

- gain from **domain adaptation**

- gain from **amortized inference / GFlowNet**

  

#### B. Domain-matched comparison for WikiText-2

- **Base GPT-2 or Llama 3** + Langevin on WikiText-2 infilling

- **WikiText-2-adapted model** + Langevin

- **WikiText-2-adapted model** + amortized model, if you build one

  

### What not to do

  

I would avoid the following as the main comparison:

  

- **Langevin on base GPT-2** vs **GFlowNet on ROCStories-tuned GPT-2**

- **Langevin on WikiText-2-tuned GPT-2** vs **GFlowNet on ROCStories-tuned GPT-2**

  

Those mixes confound method, model, and domain.

  

### My concrete recommendation

  

If you want to compare against the **original Hu et al. story infilling setup**, do this:

  

1. Fine-tune GPT-2 on ROCStories exactly as in their pipeline.

2. Run your Langevin infilling baseline on **that same fine-tuned GPT-2**.

3. Run GFlowNet training on top of **that same fine-tuned GPT-2**.

4. Report base GPT-2 as an additional ablation, not as the primary head-to-head baseline.

  

That gives you the fairest answer to:

  

> Is GFlowNet amortization better than Langevin dynamics for infilling, once both methods are given the same base model?

  

### If your thesis stays centered on WikiText-2

  

Then do not borrow ROCStories only for one side of the comparison. Instead:

  

- either keep **everything on WikiText-2**, or

- keep **everything on ROCStories** for the Hu et al. reproduction.

  

The dataset and base LM should move together.

  

---

  

## Question 3) MuCoCo / MuCoLa seems to make Langevin dynamics work with `Helsinki-NLP/opus-mt-en-de`, but it does not work for us. Do they fine-tune the model? What do they do differently?

  

### Short answer

  

From the paper and repository, **MuCoLa does not appear to fine-tune the underlying translation model for the constrained translation experiment**. Instead, they use an **off-the-shelf English→German MarianMT model** and perform **non-autoregressive constrained sampling in embedding space** on top of that model.

  

So the main reason their method “works” is **not** that they first fine-tune the base model. It is that their setup is materially different from yours.

  

### What they actually do

  

The MuCoLa paper says they sample **non-autoregressively** from a conditional or unconditional LM, initialize the full output sequence with noise, and then refine it using Langevin Dynamics over the energy function. For terminology-constrained translation, they state that they use an **off-the-shelf English-to-German MarianMT model** and apply MuCoLa to a subset of WMT17 en–de with terminology constraints. They first generate an unconstrained beam-search translation, then run MuCoLa to produce constrained candidates, and finally select the generation with the highest length-normalized log-probability.

  

### Why this can work better than your setup

  

There are several important differences.

  

#### 1. Their base model is a seq2seq translation model, not a decoder-only general LM

  

MarianMT for en→de translation has a much more structured conditional objective:

  

- input sentence on the source side,

- output sentence on the target side,

- strong token-level alignment and cross-attention,

- narrower output manifold for each example.

  

That is very different from using a general-purpose decoder-only LM for open-ended infilling. In translation, the conditional distribution is much tighter: many target tokens are strongly supported by the source sentence. That can make the energy landscape easier to navigate.

  

#### 2. Their task has stronger external structure than open infilling

  

Terminology-constrained translation is not the same as “repair a corrupted free-form sequence.”

  

Their task provides:

- a source sentence,

- a base translation model already trained for that mapping,

- explicit lexical terminology constraints.

  

Your infilling setup on WikiText-2 or open text asks the model to recover semantic coherence in a more ambiguous space. There are many valid repairs, and the gradients may be much less informative.

  

#### 3. They do not enforce the kind of strict MH-corrected sampling you analyze in the thesis

  

Your thesis shows that once you insist on a more mathematically faithful MH correction, continuous-space sampling over text can fail because nearest-neighbor projection creates a Voronoi-cell landscape with discontinuous cliffs. That is one of your main findings.

  

MuCoLa is much more pragmatic: it is an energy-based constrained generation method inspired by Langevin sampling, but it is optimized for empirical generation quality rather than for strict asymptotically correct MCMC over a discrete text posterior. That makes the method more like **guided noisy optimization** than the exact target of your thesis.

  

In other words, they may “work” because they are solving an easier empirical optimization problem, not because they have disproved your thesis conclusions about rigorous posterior sampling.

  

#### 4. They optimize a different continuous representation

  

The paper explicitly argues that optimizing directly over vocabulary-sized soft token distributions mixes slowly, and instead they optimize **smaller intermediate token representations / embeddings**. That reduces dimensionality and can make the dynamics much more stable than methods that operate on full vocabulary-simplex relaxations.

  

#### 5. They use task-specific constraints and candidate selection

  

For translation they do not simply run a blind chain and accept the last state. They generate candidates and choose outputs by a scoring rule tied to model probability and constraint satisfaction. This is closer to constrained search plus iterative refinement than to pure posterior sampling.

  

### So do they fine-tune?

  

For the **translation experiment**, I do **not** see evidence that they fine-tune the MarianMT model first. The paper describes it as an **off-the-shelf** model.

  

However, that does **not** mean the setup is “training-free” in the same sense as your experiments, because the base model itself is already **task-trained for translation**. That is a major difference.

  

So the better interpretation is:

  

- **They do not appear to add an extra fine-tuning step for that experiment.**

- But they are still relying on a model whose pretraining/fine-tuning objective is already highly aligned with the task.

  

### Why MuCoLa succeeding does not invalidate your negative results

  

Your thesis conclusion is about a harder claim:

  

- standard autoregressive LLM gradients are jagged,

- the gradient direction is often uninformative,

- strict MH in continuous space breaks due to discrete projection discontinuities,

- random-direction baselines can match true-gradient baselines surprisingly often.

  

MuCoLa does not really test that same claim under the same conditions. Their method uses:

- different models,

- different tasks,

- different constraints,

- different continuous relaxations,

- a more pragmatic rather than fully MCMC-faithful objective.

  

So both results can be true at once:

  

- **Your thesis:** rigorous Langevin-style posterior sampling on pretrained AR LMs is unstable and often breaks.

- **MuCoLa:** a practical continuous-optimization method inspired by Langevin dynamics can improve constrained generation on certain structured tasks.

  

---

  

## Overall recommendation for your next experiments

  

### Recommended comparison strategy

  

#### Track A: defend the thesis claim

Use your current setup to support the statement:

  

> Training-free, theoretically faithful Langevin dynamics is not a reliable solution for infilling on pretrained AR LMs.

  

Keep:

- Llama 3 on WikiText-2

- base model frozen

- Langevin variants, MH, random-direction controls

  

#### Track B: test amortized inference fairly

Build a separate experimental section:

  

1. Choose one dataset/domain: **ROCStories** or **WikiText-2**.

2. Adapt the base LM to that domain.

3. Compare:

   - AR baseline

   - Langevin baseline on the same adapted model

   - GFlowNet/amortized model on the same adapted model

  

This is the fairest way to test whether amortization helps.

  

### Best immediate experiment

  

If you want the most interpretable next result, I would do:

  

- **GPT-2 base** on ROCStories infilling

- **GPT-2 SFT on ROCStories** on the same infilling task

- **GPT-2 SFT + GFlowNet** on the same infilling task

- optionally **GPT-2 SFT + your Langevin baseline**

  

That decomposition cleanly answers:

  

1. Does domain adaptation help the posterior/reward landscape?

2. Does GFlowNet help beyond domain adaptation?

3. Does Langevin still fail when the base model is domain-matched?

  

---

  

## Bottom-line answers

  

### Q1

For the current thesis chapter, **no, you did not need to fine-tune Llama 3 on WikiText-2 first** if your goal was to study **training-free Langevin decoding**. For a later amortized-inference chapter, adding a domain-adapted model would be a strong extension.

  

### Q2

For a fair method comparison, **Langevin and GFlowNet should use the same underlying LM and the same domain adaptation**. If GFlowNet uses ROCStories-tuned GPT-2, then the directly comparable Langevin baseline should also use ROCStories-tuned GPT-2.

  

### Q3

MuCoLa appears to use an **off-the-shelf MarianMT translation model**, not an extra fine-tuned one for that constrained translation experiment. Their success likely comes from the **task structure, model choice, continuous representation, and pragmatic optimization setup**, not from a faithful MCMC treatment of the same problem you study.

  

---

  

## Sources used

  

- Your thesis draft

- Hu et al., *Amortizing Intractable Inference in Large Language Models* (ICLR 2024)

- Kumar et al., *Gradient-Based Constrained Sampling from Language Models* (EMNLP 2022)

- MuCoCo / MuCoLa GitHub repository