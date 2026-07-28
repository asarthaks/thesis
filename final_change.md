# final_change.md

Every reader-visible change to the thesis text made during this Claude Code session, sentence by sentence, against the snapshot commit `ec346e1`. Page numbers are the **printed folio** in the final PDF, so you can open the page and read the sentence in place.

`~~struck~~` is what the sentence said before, **bold** is what it says now. Where a passage was rewritten too heavily for a word-level marker to be readable, the two versions are given in full instead.

Pure `%`-comment bookkeeping is excluded: it does not appear in the rendered document. Figures, the capitalization sweep and the terminology sweep are summarized at the end.

**Total reader-visible sentence changes: 54.** Final document: 128 pages, 2,327,649 bytes, latexmk exit 0, zero undefined references.

---

## Abstract  (`chapters/abstract.tex`)

### 1. abstract page (unnumbered, 3rd PDF page) | changed | Claude session

**Before:**

> Controllable text generation asks a language model for text satisfying a global property that cannot be checked until the sequence is complete. One family pursues this without retraining, treating the frozen likelihood of a pretrained autoregressive model as an energy function, sampling from it with gradient-guided Langevin dynamics and adding a constraint at inference time. This thesis tests the premise it rests on: that the gradient of that likelihood is a usable local search direction on discrete text.

**After:**

> Controllable text generation asks a language model for text satisfying a property of the sequence as a whole, which cannot be verified until the sequence is complete. [footnote: A length constraint is the simplest example: whether a sentence stays under twenty words is settled only once the sentence is finished. One family of controllable generation methods pursues this without retraining the model, by treating its frozen likelihood as an energy function, i.e. a score that is low for text the model considers good, and sampling from that score with Langevin dynamics, which repeatedly nudges the text in the direction the gradient recommends, with an additive constraint supplied at generation time. This thesis tests the assumption on which that construction rests, namely that the gradient indicates which token should be placed where.

### 2. abstract page (unnumbered, 3rd PDF page) | changed | Claude session

**Before:**

> The study covers five energy functions across 145 configurations on masked-token recovery on WikiText-2 validation, with a token-space and an embedding-space sampler. The central result is negative: the input-embedding gradient gave no reliable advantage over a norm-matched random direction, an equivalence certified over 1000 paired sequences at a margin fixed in advance, and exact recovery was 0.0% in 139 cells. A `$5\times5$` sweep over step size and temperature, from a uniform to a deterministic proposal, finds no cell where the gradient helps.

**After:**

> The evaluation uses masked token recovery, in which one token of a real sentence is replaced at random and the sampler must restore it, over `$145$` configurations and five frozen energy functions: a GPT-2 Large fine-tuned on stories, three variants of it tuned further with a GFlowNet, which learns to generate in proportion to a reward, and a Llama-3 8B as a cross-architecture control.

### 3. abstract page (unnumbered, 3rd PDF page) | changed | Claude session

**Before:**

> Three candidate explanations are eliminated in turn. The target is not responsible for the recovery failure: a top-`$k$` rescoring pass reaches 4.43 and a gradient-free Gibbs sampler 6.69 on the identical energy, against 6.4 for the best Langevin cell. Nor is it due to a complete absence of useful local information in the frozen model: a derivative of the same likelihood taken in the relaxed token-indicator coordinates, differing from the input-embedding gradient by the candidate's own conditional log-probability, correlates with the true energy change at `$\rho = 0.60$` to `$0.73$` at admissible distances against `$|\rho| < 0.06$`, and in the same sampler recovers 40% of tokens on GPT-2 and 41% on Llama-3, where that gradient recovers at most 2%. What fails is the choice of coordinates: the missing token-fit term is one forward pass away, in the output conditional. Amortizing the search with a GFlowNet did not escape the failure: the policies attained high reward by exploiting it, and sampling on the tuned energy matched the untuned base. A ladder of proposals inside one exact-energy chain locates what the remaining gap buys: a uniform draw recovers 0.5%, matching the gradient-guided sampler; the left-to-right conditional 23.5%; a score-entropy diffusion model 39%; and a masked language model, not score-trained, 44.5%. Conditioning access, in particular to the output distribution and the right context, explains that ordering better than score training does. The practical implication: differentiate the right object, or propose from the output side, not from the input embeddings.

**After:**

> Results show that the direction of the gradient is not usable: replacing it with a random direction of the same magnitude produced no reliable difference, an equivalence certified over `$1000$` paired sequences, the correct token was recovered in `$0.0\%$` of attempts in `$139$` of the `$145$` configurations, and a sweep over the settings that make the gradient decisive found none at which it helped. The failure is attributable to the coordinate in which the derivative is taken, since differentiating the same likelihood with respect to the token's indicator over the vocabulary rather than its input embedding introduces the candidate's own conditional log-probability, to which that gradient is structurally blind; in the same sampler and on the same weights that derivative recovers `$40\%$` of tokens on GPT-2 and `$41\%$` on Llama-3, where the input-embedding gradient never exceeds `$2\%$`. The gap that remains is associated with conditioning rather than training, as a single chain accepting by the exact energy and varying only the proposal recovers `$0.5\%$` from a uniform draw, `$23.5\%$` from the model's own left-to-right conditional, `$39\%$` from a diffusion model trained to estimate scores, and `$44.5\%$` from a masked language model never trained on scores. These findings indicate that inference-time control on a frozen autoregressive model should differentiate the token indicator, or draw proposals from the output distribution, rather than from the input embeddings.


## 1 Introduction  (`chapters/01_introduction.tex`)

### 4. p. 9, 1 | changed | Claude session

The dominant paradigm behind modern language ~~models~~ **models, by which this thesis throughout means a neural network that assigns a probability to a sequence of tokens,** is autoregressive generation: ~~a~~ **the** model factorizes the probability of a sequence into a product of next-token conditionals and produces text one token at a time, from left to right (radford2019language). This factorization is what makes these models trainable at scale and fast at inference, and it is ~~the~~ **one** reason **among several that** they work as well as they ~~do. It~~ **do; the volume of text they are trained on and the Transformer architecture that consumes it are the other principal ingredients (vaswani2017attention). The factorization** is also, as this thesis argues, the reason one popular approach to controlling them does ~~not.~~ **not work.**

### 5. p. 10, 1.1 | changed | Claude session

This is effective and now standard, and the aligned ~~assistants~~ **language models** in wide use are its product, but the control it provides is static:

### 6. p. 10, 1.1 | added | Claude session

**After:**

> That capability is inference-time control: the model's parameters are fixed once and never revisited, and the only thing that differs between one constraint and the next is a term handed to the decoding procedure while it runs.

### 7. p. 11, 1.2 | changed | Claude session

**Before:**

> where `$\energy(\bm{x})$` is the energy of a sequence `$\bm{x}$`, a scalar that is low for desirable sequences and high for undesirable ones, and `$Z$` is a normalizing constant.

**After:**

> where `$\bm{x} = (x_1, \dots, x_T)$` is a sequence of `$T$` tokens drawn from a vocabulary `$V$`, `$\energy(\bm{x}) \in \R$` is the energy assigned to that sequence, and `$Z \in \R$` is a normalizing constant. The minus sign in the exponent fixes the direction of the correspondence between the two quantities, and it is worth reading off explicitly: since `$\exp(-\energy)$` falls as `$\energy$` rises, low energy corresponds to high probability and high energy to low probability, so every statement about one is the mirror image of a statement about the other, and a sequence the model considers good is one of low energy. Any scalar-valued function of a complete sequence can serve as `$\energy$`; the choice this thesis studies, the negative log-likelihood of a language model, is made in Section [sec:bg-ebm].

### 8. p. 11, 1.2 | changed | Claude session

where the first term is the negative log-likelihood of a frozen, pretrained language model and supplies fluency, pulling samples toward text the model considers natural, and the second term ~~`$C(\bm{x})$`~~ is an arbitrary differentiable ~~constraint,~~ **constraint `$C : V^{T} \to \R$`, which scores a complete sequence for the property wanted,** weighted by a **scalar** coefficient ~~`$\lambda$`~~ **`$\lambda > 0$`** that trades off fluency against constraint satisfaction.

### 9. p. 12, 1.2 | changed | Claude session

**Before:**

> In the settings tested here that presupposition does not hold: following the input-embedding gradient proves no more effective than following a norm-matched random direction, the finding stated precisely in Section [sec:results-fallacy].

**After:**

> In the settings tested here that presupposition does not hold. Following the input-embedding gradient, the derivative of the likelihood with respect to the vector the model looks up for the token at the position being changed, proves no more effective than following a norm-matched random direction, a direction drawn at random and then rescaled to the exact length of the true gradient, so that the two differ in direction and in nothing else. Section [sec:results-fallacy] states the finding precisely.

### 10. p. 13, 1.2 | changed | Claude session

Under Hypothesis B the target is serviceable but the model does not contain what a local proposal needs, because teacher ~~forcing~~ **forcing, the training scheme that always supplies the true prefix and asks only for the next token,** shapes **those** next-token ~~conditionals on correct prefixes~~ **predictions** and never asks the ~~joint~~ likelihood **of a whole sequence** to rank ~~substitutions;~~ **substitutions inside it;**

### 11. p. 13, 1.2 | changed | Claude session

re-analysing the same measurements under a derivative taken ~~in~~ **with respect to** the ~~token-indicator coordinates,~~ **token's identity rather than its embedding,** and substituting that ~~surrogate~~ **alternative** into the same sampler on the same frozen weights, disposes of B;


## 2 Background  (`chapters/02_background.tex`)

### 12. p. 18, 2.1 | changed | Claude session

**Before:**

> where each conditional `$p_{\text{LM}}(x_t \mid x_{<t})$` is produced by a neural network, here a Transformer (vaswani2017attention), applied to the prefix: it reads the prefix, produces logits over the vocabulary, and a softmax turns them into the next-token distribution.

**After:**

> where `$x_t \in V$` is the token at position `$t$`, the vocabulary `$V$` is a finite set of size `$|V|$`, and `$x_{<t} = (x_1, \dots, x_{t-1})$` denotes the prefix preceding position `$t$`, empty when `$t = 1$`. Each conditional `$p_{\text{LM}}(x_t \mid x_{<t})$` is produced by a neural network, here a Transformer (vaswani2017attention), applied to that prefix: it reads the prefix, produces a logit vector in `$\R^{|V|}$`, and a softmax turns it into a distribution over the `$|V|$` tokens. Read in words, [eq:bg-chain] says that the score of a whole sequence is the sum of the scores its own model assigns to each token in turn, each judged against everything to its left and nothing to its right.

### 13. p. 19, 2.2 | changed | Claude session

**Before:**

> where `$\epsilon_t$` is a step size at iteration `$t$`.

**After:**

> where the state `$\bm{s}_t \in \R^D$` and its update `$\bm{s}_{t+1} \in \R^D$` are `$D$`-dimensional vectors, `$\grad_{\bm{s}}\log p(\bm{s}_t) \in \R^D$` is the gradient of the log-density at the current state, `$\epsilon_t > 0$` is a scalar step size at iteration `$t$`, and the noise `$\bm{\xi}_t \in \R^D$` is drawn from a standard Gaussian with the `$D \times D$` identity as its covariance. In words, [eq:bg-langevin] says: take a step downhill in energy, then jog the state by a random amount whose size the step size also sets.

### 14. p. 21, 2.3 | changed | Claude session

**Before:**

> and otherwise rejects it and remains at `$\bm{s}$`, counting the current state again.

**After:**

> where `$\bm{s}, \bm{s}' \in \R^D$` are the current and proposed states, `$p$` is the density being sampled, `$q(\cdot \mid \cdot)$` is the proposal density, and `$\alpha \in [0,1]$` is the resulting acceptance probability; if the move is not accepted the chain rejects it and remains at `$\bm{s}$`, counting the current state again.

### 15. p. 22, 2.4 | changed | Claude session

**Before:**

> Transplanting this to the discrete setting, where the candidate states are token embeddings `$\ve(v)$` and the current state is `$\ve(x_i)$`, gives a categorical proposal whose logit for token `$v$` is

**After:**

> Transplanting this to the discrete setting, the candidate states are no longer arbitrary points of `$\R^D$` but the embeddings of actual tokens. Write `$v \in V$` for a candidate vocabulary item, that is, one of the `$|V|$` tokens that could be placed at the position being resampled, and `$\ve(v) \in \R^D$` for its embedding, the row of the embedding matrix that the model looks up for it; the current occupant of position `$i$` is the token `$x_i \in V$` with embedding `$\ve(x_i) \in \R^D$`. This gives a categorical proposal over all `$|V|$` candidates whose logit for token `$v$` is

### 16. p. 23, 2.4 | changed | Claude session

**Before:**

> where `$\vg = \grad_{\ve_i}\log p_{\text{LM}}(\bm{x})$`.

**After:**

> where `$\vg = \grad_{\ve_i}\log p_{\text{LM}}(\bm{x}) \in \R^D$` is the gradient of the sequence log-likelihood with respect to the input embedding at position `$i$`, the displacement `$\ve(v) - \ve(x_i) \in \R^D$` is the move in embedding space that swapping `$x_i$` for `$v$` would make, and `$\epsilon > 0$` is the step size, the same quantity as in [eq:bg-langevin]. In words, [eq:bg-discrete-proposal] scores every token in the vocabulary by two competing considerations and samples one in proportion to the result.

### 17. p. 23, 2.4 | changed | Claude session

**Let `$M$` be the number of positions being resampled, so that the continuous state is `$\bm{s} \in \R^{M \times D}$`, one `$D$`-dimensional vector per resampled position.** Writing ~~`$\mathrm{proj}_V(\bm{s})$`~~ **`$\mathrm{proj}_V : \R^{M \times D} \to V^{M}$`** for the nearest-neighbour map that sends the continuous state at each ~~masked~~ **such** position to the token whose embedding is closest in Euclidean distance, that target energy is


## 3 Related Work  (`chapters/03_related_work.tex`)

### 18. p. 28, 3.1 | changed | Claude session

These methods work, and the widely used aligned ~~assistants~~ **language models** are their fruit, but the control they produce is fixed once training is complete.

### 19. p. 29, 3.2 | changed | author (MuCoLa correction)

The continuous sampler of this thesis ~~implements~~ **reproduces the state geometry of** that ~~shared~~ mechanism faithfully, **including the projection onto the embedding table after every update, but it differs from MuCoLa in how the target token enters the likelihood: MuCoLa substitutes the continuous vector into the output softmax, whereas the implementation here gathers the score at the projected token index. Section [sec:disc-scope] sets out what that difference does** and ~~the~~ **does not affect. The** constrained mucola arm of the experiments is ~~that comparison, run~~ **the comparison** at the level of the shared energy rather than of the original papers' tasks and metrics, which use different benchmarks and are out of scope.


## 4 Methodology  (`chapters/04_methodology.tex`)

### 20. p. 37, 4.3 | added | Claude session

**After:**

> The update has four parts, and naming them separately is what makes the rest of this section readable. `$\vs_t \in \R^{M \times D}$` is the current state, one `$D$`-dimensional vector for each of the `$M$` positions being resampled, and in general it is not the embedding of any token. `$\vs_{\text{interim}} \in \R^{M \times D}$` is the interim continuous point, the current state moved a half-step of size `$\epsilon_t$` along the gradient `$\grad_{\vs}\log p \in \R^{M \times D}$`; it is where an unmodified Langevin update would put the chain. `$\mathrm{proj}_V(\vs_{\text{interim}}) \in \R^{M \times D}$` is the projection, the embedding of whichever token lies nearest to that interim point, and is by construction a point on the token manifold. Their average is the interpolated mean, the centre of the Gaussian the next state is actually drawn from, with `$\epsilon_t > 0$` the scalar step size and `$\bm{I}$` the identity covariance of that Gaussian.

### 21. p. 40, 4.4 | added | Claude session

**After:**

> Both `$p^{(m)}_{\text{ref}}$` and `$p^{(m)}_{\text{pred}}$` are therefore probability vectors in `$\R^{|V|}$` over the same vocabulary, differing only in what was placed at the masked positions, and the index `$m$` ranges over `$\mathcal{M}'$`.

### 22. p. 44, 4.7 | added | Claude session

**After:**

> That inner product is the object under test, so it is worth saying in words what it is and why. Both of its factors are vectors in `$\R^{D}$`. The first, `$\vg$`, is the gradient of the sequence log-likelihood with respect to the input embedding at the masked position: it points in the direction in embedding space along which the log-likelihood grows fastest, and its length says how fast. The second, `$\ve(v) - \ve(x_i)$`, is the displacement, the vector one would travel along in embedding space by replacing the current token `$x_i$` with the candidate `$v$`. Their inner product `$\vg^\top(\ve(v) - \ve(x_i))$` is a single number: the component of that displacement lying along the uphill direction, multiplied by how steep the slope is. It is therefore the first-order prediction of how much the log-likelihood would change if the substitution were made, and it is exactly the quantity the discrete proposal of [eq:bg-discrete-proposal] uses to decide which token to propose. The experiment asks whether that prediction agrees with what actually happens, which is `$\Delta(v)$`, computed by making the substitution and running the model.


## 5 Results  (`chapters/05_results.tex`)

### 23. p. 4, 5.3.4 | changed | Claude session

Why the Input-Embedding Surrogate ~~Fails~~ **Fails, and What Recovers It**

### 24. p. 5, 5.9 | changed | Claude session

**Before:**

> Constrained Generation with an Additive Sentiment Term

**After:**

> Constrained Generation and Classifier-Guided Steering

### 25. p. 46 (section start), 5.1 | removed | Claude session

**Before:**

> The Calibrated Proposal is Numerically Uniform

*(removed; nothing replaces it in the running text)*

### 26. p. 50, 5.1 | changed | Claude session

[Discrete Langevin ~~sampler~~ **Sampler** trajectories, 50-step schedule.]Discrete Langevin Sampler trajectories on GPT-2 Large over a 50-step annealing schedule, averaged across sequences.

### 27. p. 51 (section start), 5.2 | removed | Claude session

**Before:**

> The Same Breakdown Read Spatially

*(removed; nothing replaces it in the running text)*

### 28. p. 53, 5.2 | added | Claude session

**After:**

> Read spatially, the same breakdown is a statement about where the samplers go.

### 29. p. 62 (section start), 5.4 | removed | Claude session

**Before:**

> Recovering the Missing Term with an Output-Side Surrogate

*(removed; nothing replaces it in the running text)*

### 30. p. 65, 5.4 | changed | Claude session

Everything above differentiates the sequence log-likelihood with respect to the input embedding at the masked position, `$\vg = \partial \log p(\vx) / \partial \ve_i$`, which is what the samplers ~~this thesis reimplements~~ **implemented here** use, because they operate in embedding ~~space.~~ **space and the energy they score gathers at the projected token index.**

### 31. p. 65, 5.4 | changed | Claude session

Let ~~`$\bm{E}$`~~ **`$\bm{E} \in \R^{|V| \times D}$`** be the embedding matrix, whose `$v$`-th row is ~~`$\ve(v)$`,~~ **`$\ve(v) \in \R^{D}$`,** and let ~~`$\bm{z}_i$`~~ **`$\bm{z}_i \in \R^{|V|}$`** be a point on the vocabulary ~~simplex~~ **simplex, so that its coordinates are non-negative and sum to one; it is** relaxed from the vertex ~~of~~ **that** the current token ~~`$x_i$`.~~ **`$x_i$` occupies, where the coordinate `$\bm{z}_i[x_i]$` is one and every other is zero.**

### 32. p. 66, 5.4 | changed | Claude session

~~whose~~ **where the sum runs over all `$|V|$` tokens `$v \in V$`, `$\bm{z}_i[v] \in [0,1]$` is the coordinate of `$\bm{z}_i$` belonging to token `$v$`, and `$L_{\text{future}} : \R^{D} \to \R$` scores the suffix from an input embedding, so that `$\bm{E}^\top\bm{z}_i \in \R^{D}$` is the mixed embedding the relaxation feeds forward. In words, the two terms of [eq:relaxed-objective] are the two roles a token index plays: the** first ~~term~~ is the current position's own contribution to ~~[eq:bg-chain]~~ **[eq:bg-chain],** with the discrete target replaced by the relaxed indicator, and ~~whose~~ **the** second is the log-likelihood of the suffix evaluated with `$\bm{E}^\top\bm{z}_i$` supplied as the input embedding at position `$i$`.

### 33. p. 66, 5.4 | changed | Claude session

**Before:**

> and this thesis calls it the relaxed token-indicator derivative, shortened to the token-indicator derivative where no confusion arises.

**After:**

> one scalar for each of the `$|V|$` candidates, so the whole derivative is a vector in `$\R^{|V|}$` with one entry per token, against the `$\R^{D}$` of the input-embedding gradient `$\vg$`. This thesis calls it the relaxed token-indicator derivative, shortened to the token-indicator derivative where no confusion arises. Read in words, [eq:onehot-grad] says that the worth of putting token `$v$` at position `$i$` is the sum of two things the model already computes: how likely `$v$` is here given everything to its left, and how well `$v$`'s embedding points along the direction that raises the likelihood of everything to its right.

### 34. p. 70, 5.4 | changed | author (MuCoLa correction)

It is that the input-embedding Jacobian ~~slice, which is the object the embedding-space samplers~~ **slice** of ~~this literature differentiate,~~ **a likelihood whose target token enters as a discrete index** discards the half of the energy that carries the signal, and that supplying the discarded half turns the same sampler on the same frozen model from zero recovery into recovery comparable to a purpose-trained diffusion model. **The coordinate choice that produces the blindness belongs to the implementation studied here rather than to the published embedding-space samplers from which it was drawn, a scope restriction Section [sec:disc-scope] states in full.**

### 35. p. 85 (section start), 5.10 | removed | Claude session

**Before:**

> Classifier-Guided Steering on a Navigable Landscape

*(removed; nothing replaces it in the running text)*


## 6 Discussion  (`chapters/06_discussion.tex`)

### 36. p. 90, 6.1 | changed | Claude session

On a score-trained diffusion landscape, steering became partly possible under a fluency trust region, in one direction per setting, with the reachable direction tracking how far the guiding and judging instruments agree on the text being scored ~~(Section [sec:results-guided]).~~ **(Appendix [app:guided]).**

### 37. p. 91, 6.2 | changed | Claude session

the input-embedding derivative omits the candidate's direct conditional-fit term, which is why the surrogate carries no rank information and why supplying the missing term repairs it ~~(Sections [sec:results-linradius] and [sec:results-onehot]).~~ **(Section [sec:results-linradius]).**

### 38. p. 94, 6.4 | changed | author (MuCoLa correction)

**Before:**

> The thesis therefore refutes the premise for embedding-space samplers of the MuCoLa and COLD family and not for gradient-guided discrete sampling as such.

**After:**

> The thesis therefore refutes the premise for a sampler that differentiates the input embedding of a likelihood whose target token enters as a discrete index, and not for gradient-guided discrete sampling as such. That scope is narrower than the family this work set out to test, and the difference is worth stating precisely. MuCoLa (kumar2022gradient) substitutes the continuous vector into the output softmax itself, defining `$P(\tilde{\bm{e}}_{n+1} \mid \tilde{\bm{e}}_{1:n})$` by replacing the looked-up target embedding with the state, so its gradient reaches the token's own score directly and not only through the following hidden states; COLD (qin2022cold) carries a soft sequence of vocabulary logits whose fluency term is a soft cross-entropy against the model's reference distribution, differentiable in the same way. Neither discards the self term. The energy implemented here scores the masked position by gathering at the projected token index, and it is that construction, not theirs, that the null characterizes. The finding is correspondingly reframed rather than weakened: a natural implementation choice inside the embedding-space approach destroys the ranking signal, the published methods avoid it by relaxing the target as well as the input, and Section [sec:results-onehot] measures what that choice is worth. The convergence is close enough to be worth recording. MuCoLa's self term contributes a gradient of `$\bm{h}_n$`, so it ranks a candidate `$v$` against the incumbent by `$\bm{h}_n^\top(\bm{e}(v) - \bm{e}(x_i))$`, the difference of their logits, while the relaxed token-indicator self term contributes `$\log p(v \mid \bm{x}_{<i}) - \log p(x_i \mid \bm{x}_{<i})$`; the two differ only by the shared normalizer, which cancels in the difference. What the continuous sampler of this thesis does reproduce faithfully is the state geometry, a continuous embedding state with a nearest-neighbour projection onto the embedding table after every update, and it is that geometry the Metropolis--Hastings result of Section [sec:results-mh] concerns.

### 39. p. 96, 6.4 | changed | author (MuCoLa correction)

The experiments refute the premise that the frozen autoregressive sequence-likelihood gradient can supply the required local score for ~~this class of Langevin-based methods~~ **a Langevin sampler on a likelihood whose target token enters as a discrete index,** without additional training.


## A Appendix  (`chapters/08_appendix.tex`)

### 40. p. 5, A.5.1 | changed | Claude session

The Guide-Judge Agreement Ladder **and Per-Class Confusion**

### 41. p. 109 (section start), A.1 | removed | Claude session

**Before:**

> 

*(removed; nothing replaces it in the running text)*

### 42. p. 109 (section start), A.2 | removed | Claude session

**Before:**

> 

*(removed; nothing replaces it in the running text)*

### 43. p. 113 (section start), A.3 | removed | Claude session

**Before:**

> 

*(removed; nothing replaces it in the running text)*

### 44. p. 116, A.4 | changed | Claude session

This section examines the spatial paths the samplers trace through the embedding space, a concrete counterpart to the acceptance statistics ~~of Section [sec:results-mh]~~ and ~~the~~ trajectory summaries of Section ~~[sec:results-traj].~~ **[sec:results-mh].**

### 45. p. 116 (section start), A.4 | changed | Claude session

**Before:**

> [width=0.80]figures/fig_traj_distance.png

**After:**

> [width=]figures/fig_traj_distance.png

### 46. p. 116 (section start), A.4 | changed | Claude session

**Before:**

> [width=0.85]figures/fig_traj_pca.png

**After:**

> [width=]figures/fig_traj_pca.png

### 47. p. 116 (section start), A.4 | removed | Claude session

**Before:**

> 

*(removed; nothing replaces it in the running text)*

### 48. p. 121 (section start), A.5.2 | added | Claude session

**After:**

> app:guided-confusion

### 49. p. 121 (section start), A.5.2 | removed | Claude session

**Before:**

> Per-Class Confusion on the Neutral Calibration Surface app:guided-confusion

*(removed; nothing replaces it in the running text)*

### 50. p. 121 (section start), A.5.2 | removed | Claude session

**Before:**

> The two conditional agreement rates are statistically indistinguishable. The confusion table therefore confirms the overall alignment gap that the ladder measures, and does not identify a by-class asymmetry that would explain the transfer direction. The transfer asymmetry is reported as observed and setting-contingent, and no class-level mechanism is claimed beyond the alignment gap.

*(removed; nothing replaces it in the running text)*

### 51. p. 121 (section start), A.5.2 | removed | Claude session

**Before:**

> 

*(removed; nothing replaces it in the running text)*

### 52. p. 123, A.5.2 | changed | Claude session

**Before:**

> confidence interval `$[-18.0, +2.6]$` includes zero and leans, if at all, the other way.

**After:**

> confidence interval `$[-18.0, +2.6]$` includes zero, so the two conditional agreement rates are statistically indistinguishable. No by-class asymmetry explains the transfer direction, which is therefore reported as observed and setting-contingent, with no class-level mechanism claimed beyond the alignment gap.

### 53. p. 124 (section start), A.7.1 | removed | Claude session

**Before:**

> 

*(removed; nothing replaces it in the running text)*

### 54. p. 124 (section start), A.7.1 | changed | Claude session

Use of ~~AI-Tools~~ **AI Tools**


---

## Changes not listed sentence by sentence

These are sweeps and non-prose changes. They are reader-visible but there is nothing to read side by side.

| change | where | detail |
|---|---|---|
| **Four subsections collapsed into their parents** | 5.1, 5.2, 5.4, 5.10 (pp. 46, 51, 62, 81) | The headings *The Calibrated Proposal is Numerically Uniform*, *The Same Breakdown Read Spatially*, *Recovering the Missing Term with an Output-Side Surrogate* and *Classifier-Guided Steering on a Navigable Landscape* are gone from both the contents and the body; their text now runs on inside the parent section. Two parent titles were widened to cover it (5.4 gained *and What Recovers It*; 5.10 became *Extension: Constrained Generation and Classifier-Guided Steering*). |
| **Capitalization sweep** | whole document | 66 headings and 37 short captions audited against one convention (Title Case for headings, sentence case for captions). Two were non-compliant and were corrected: *Use of AI-Tools* to *Use of AI Tools* (p. 126), and the Figure 1 short caption *Discrete Langevin sampler* to *Discrete Langevin Sampler* (list of figures). |
| **Terminology sweep** | whole document | Single term *language model*, defined at first use. Two replacements of *aligned assistants* (pp. 10 and 27). *LLM* verified absent. |
| **Symbol rename in eq. (7)** | 2.4, p. 21 | The DLS step size was called `alpha`, colliding with the Metropolis-Hastings acceptance probability in eq. (6) and disagreeing with Sections 4.3 and 5.3.2, which already called it `epsilon`. Renamed to `epsilon` in the equation and its footnote. No quantity or value changes. |
| **Nine figures regenerated** | Figures 2, 4, 5, 6, 7, 8, 9, 10, 11 | Redrawn at the width they are printed at, so axis text is legible in print; Figure 2's colliding x-axis fixed and its clipped label and overlapping legend repaired; the two dense scatters (Figures 4 and 6) rasterized, cutting the PDF from 3,097,565 to 2,327,649 bytes. No data, no series and no number changed. |
| **Figures 14 and 15 enlarged** | A.4, pp. 116, 117 | Widened to the full text block. They could not be redrawn properly: the GPT-2 Large SFT checkpoint their script needs is no longer on disk. |
| **Seven forced page breaks removed** | appendix A.1 to A.8 | Page breaks only, no content. |
| **Appendix A.5.3 merged into A.5.2** | p. 121 | Heading removed, two restating sentences consolidated into one. Every number, both tables and all source comments kept. |

## What did not change

- **No number's value**, anywhere. Verified by extracting every decimal and every run of three or more digits from the rendered text before and after: the only differences are page numbers, section numbers, two citation years added by the data-and-architecture edit, and statistics deliberately dropped from the abstract that remain stated in the body.

- **No table body, confidence interval or result claim.**

- **The template.** Margins, line spacing, citation style and the title page are untouched.

- **The bibliography.** `thesis.bbl` is identical modulo whitespace.
