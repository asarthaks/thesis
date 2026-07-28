# Proposed corrections: the MuCoLa / COLD attribution

Status: DRAFTED, NOT APPLIED. Nothing in `Doc/final/thesis/` has been touched.

## The verified facts

MuCoLa, Kumar, Paria and Tsvetkov 2022, section 3, "Energy as a function of embeddings",
verbatim:

> the softmax probability is computed as
> `P(y_{n+1}|y_{1:n},x) = exp(h_n^T e_{n+1} + b_{n+1}) / sum_j exp(h_n^T e_j + b_j)`
> ... By replacing `e_{n+1}` with `e~_{n+1}`, we convert the above probability to
> `P(e~_{n+1}|e~_{1:n},x)`. For each position n+1, `e~_{n+1}` receives gradients,
> (a) directly from `-log P` function and (b) through `h_{n+1}` via back-propagation
> through the network layers.

Their (a) is the self term. Self-gradient is `h_n`.

COLD, Qin, Welleck, Khashabi and Choi 2022, equation 3:

> `f_LM(y~) = sum_t sum_v p_LM(v | y~_<t) log softmax(y~_t(v))`

Per-position vocabulary logits, soft cross-entropy, differentiable. Self term kept.

MuCoLa also projects onto the embedding table after every gradient step ("after every
gradient step, we project each updated vector back to a quantized space, that is the
embedding table using Euclidean distance"). So the projection GEOMETRY in this thesis is
faithful to MuCoLa. Only the ENERGY differs.

This repository's energy: `core/base_sampler.py:50-53` sets
`target_ids[0, mask_indices_t] = s_idx` (the projected index) and calls
`core/prep.py:joint_log_prob_from_inputs_embeds`, a hard-target cross-entropy. Blind, as
claimed. Every measurement stands.

---

## 1. `Doc/final/thesis/chapters/05_results.tex`, line 462

BEFORE

> It is that the input-embedding Jacobian slice, which is the object the embedding-space
> samplers of this literature differentiate, discards the half of the energy that carries
> the signal, and that supplying the discarded half turns the same sampler on the same
> frozen model from zero recovery into recovery comparable to a purpose-trained diffusion
> model.

AFTER

> It is that the input-embedding Jacobian slice of a likelihood whose target token enters
> as a discrete index discards the half of the energy that carries the signal, and that
> supplying the discarded half turns the same sampler on the same frozen model from zero
> recovery into recovery comparable to a purpose-trained diffusion model. The coordinate
> choice that produces the blindness belongs to the implementation studied here rather
> than to the published embedding-space samplers from which it was drawn, a scope
> restriction Section~\ref{sec:disc-scope} states in full.

---

## 2. `Doc/final/thesis/chapters/06_discussion.tex`, line 50 (`sec:disc-scope`)

BEFORE

> The thesis therefore refutes the premise for embedding-space samplers of the MuCoLa and
> COLD family and not for gradient-guided discrete sampling as such.

AFTER (one sentence replaced by a passage)

> The thesis therefore refutes the premise for a sampler that differentiates the input
> embedding of a likelihood whose target token enters as a discrete index, and not for
> gradient-guided discrete sampling as such. That scope is narrower than the family this
> work set out to test, and the difference is worth stating precisely. MuCoLa
> \citep{kumar2022gradient} substitutes the continuous vector into the output softmax
> itself, defining $P(\tilde{\bm{e}}_{n+1} \mid \tilde{\bm{e}}_{1:n})$ by replacing the
> looked-up target embedding with the state, so its gradient reaches the token's own score
> directly and not only through the following hidden states; COLD \citep{qin2022cold}
> carries a soft sequence of vocabulary logits whose fluency term is a soft cross-entropy
> against the model's reference distribution, differentiable in the same way. Neither
> discards the self term. The energy implemented here scores the masked position by
> gathering at the projected token index, and it is that construction, not theirs, that the
> null characterizes. The finding is correspondingly reframed rather than weakened: a
> natural implementation choice inside the embedding-space approach destroys the ranking
> signal, the published methods avoid it by relaxing the target as well as the input, and
> Section~\ref{sec:results-onehot} measures what that choice is worth. The convergence is
> close enough to be worth recording. MuCoLa's self term contributes a gradient of
> $\bm{h}_n$, so it ranks a candidate $v$ against the incumbent by
> $\bm{h}_n^\top(\bm{e}(v) - \bm{e}(x_i))$, the difference of their logits, while the
> relaxed token-indicator self term contributes
> $\log p(v \mid \bm{x}_{<i}) - \log p(x_i \mid \bm{x}_{<i})$; the two differ only by the
> shared normalizer, which cancels in the difference. What the continuous sampler of this
> thesis does reproduce faithfully is the state geometry, a continuous embedding state with
> a nearest-neighbour projection onto the embedding table after every update, and it is
> that geometry the Metropolis--Hastings result of Section~\ref{sec:results-mh} concerns.

---

## 3. `Doc/final/thesis/chapters/03_related_work.tex`, line 20

BEFORE

> The continuous sampler of this thesis implements that shared mechanism faithfully, and
> the constrained \emph{mucola} arm of the experiments is that comparison, run at the level
> of the shared energy rather than of the original papers' tasks and metrics, which use
> different benchmarks and are out of scope.

AFTER

> The continuous sampler of this thesis reproduces the state geometry of that mechanism
> faithfully, including the projection onto the embedding table after every update, but it
> differs from MuCoLa in how the target token enters the likelihood: MuCoLa substitutes the
> continuous vector into the output softmax, whereas the implementation here gathers the
> score at the projected token index. Section~\ref{sec:disc-scope} sets out what that
> difference does and does not affect. The constrained \emph{mucola} arm of the experiments
> is the comparison at the level of the shared energy rather than of the original papers'
> tasks and metrics, which use different benchmarks and are out of scope.

---

## 4. `revision/analyze_onehot_surrogate.py`, docstring lines 5-10

BEFORE

> which is the surrogate MuCoLa- and COLD-style samplers actually use, because they
> differentiate with respect to the INPUT EMBEDDING. Grathwohl et al. (2021) and
> Zhang et al. (2022) instead differentiate with respect to the ONE-HOT (or
> simplex-relaxed) input.

AFTER

> which is the surrogate THIS repository's energy induces: it differentiates with respect
> to the INPUT EMBEDDING while the target token enters as a discrete index (see
> core/prep.py:joint_log_prob_from_inputs_embeds and core/base_sampler.py:50-53).
>
> CORRECTION 2026-07-28: an earlier version of this docstring attributed that surrogate to
> MuCoLa and COLD. That is wrong, verified against both papers. MuCoLa (Kumar et al. 2022,
> sec. 3) substitutes the continuous vector into the output softmax numerator, so its self
> term is differentiable; COLD (Qin et al. 2022, eq. 3) carries per-position vocabulary
> logits and a soft cross-entropy, likewise. Grathwohl et al. (2021) and Zhang et al.
> (2022) differentiate the ONE-HOT (or simplex-relaxed) input, which also keeps the self
> term. The self-term-blind object is this repository's energy, not theirs. The measured
> results are unaffected; the attribution was.

---

## 5. `CLAUDE.md`, the "What this project is" paragraph

BEFORE

> the INPUT-EMBEDDING gradient of a frozen autoregressive likelihood, which is what
> MuCoLa/COLD-style samplers differentiate, carries no usable directional signal, but the
> ONE-HOT gradient of the same likelihood does

AFTER

> the INPUT-EMBEDDING gradient of a frozen autoregressive likelihood whose target token
> enters as a discrete index, which is the energy implemented here (see
> core/prep.py:joint_log_prob_from_inputs_embeds), carries no usable directional signal,
> but the ONE-HOT gradient of the same likelihood does

plus a new lettered correction alongside (a) to (e):

> (f) The claim that MuCoLa and COLD differentiate the blind object is WITHDRAWN, verified
>     against both papers on 2026-07-28. MuCoLa substitutes the continuous vector into the
>     output softmax numerator (sec. 3) and COLD uses a soft cross-entropy over per-position
>     vocabulary logits (eq. 3), so both keep the self term. What this repository
>     reproduces faithfully is their projection GEOMETRY, not their ENERGY. No measurement
>     changes; the attribution does. Do NOT write that this thesis refutes the premise for
>     the MuCoLa and COLD family.

---

## 6. `Doc/final/beamer/Presentation.tex`, backup 26

BEFORE

> The continuous sampler follows the COLD/MuCoLa mechanism faithfully, including the
> correction those works typically omit.

AFTER

> The continuous sampler follows their state geometry faithfully, continuous embedding
> state with a projection after every update, and adds the correction those works omit. It
> does not follow their energy: see backup 33.

---

## Not affected, checked

- `abstract.tex`. Attributes nothing to MuCoLa or COLD.
- `01_introduction.tex` lines 49 and 51. The described assumption is about following a
  gradient across the space of texts, which is true of both papers. No coordinate claim.
- `02_background.tex` line 67. The Voronoi and projection geometry is faithful to MuCoLa,
  which projects onto the embedding table after every gradient step.
- `06_discussion.tex` line 39. Concerns the omitted Metropolis correction, which both
  papers do omit. Correct as written.
- `06_discussion.tex` line 43 and `07_conclusion.tex` line 10. Scope the claim to the
  input-embedding coordinates without attributing that choice to anyone.
- Every number, table and figure in Chapter 5.
