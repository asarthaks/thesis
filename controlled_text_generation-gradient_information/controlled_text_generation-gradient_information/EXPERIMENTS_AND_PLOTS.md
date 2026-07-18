# Experiments and Plots to Produce Before Finalising the Thesis

Ordered by importance. Experiment 1 is the one that answers your supervisor.
Everything here is cheap. Nothing requires retraining anything.

---

## EXPERIMENT 1 (CRITICAL) — The Linearization Radius

**This is the experiment that answers "why doesn't it work". Do this one first.**

### What it measures

The DLS proposal scores a candidate token `v` at masked position `i` using a
first-order Taylor surrogate:

    Δ̂(v) = ∇_{e_i} log p(x⁰)ᵀ · ( e(v) − e(x_i⁰) )

The *true* change in energy from making that substitution is:

    Δ(v) = log p(x with x_i = v) − log p(x⁰)

If gradient guidance works, Δ̂ should predict Δ. We measure whether it does.

### Procedure

```
for each of N = 200 sequences from the ROCStories eval set:
    pick one masked position i
    compute g = ∇_{e_i} log p(x⁰)                       # one backward pass
    sample a candidate set V' of 2000 tokens from the vocabulary
        (stratify: 500 nearest neighbours of e(x_i⁰) by L2,
                   500 mid-range,
                   1000 uniformly at random)
    for each v in V':
        Δ̂(v) = gᵀ (e(v) − e(x_i⁰))                      # free, just a dot product
        Δ(v)  = log p(x[i→v]) − log p(x⁰)               # one forward pass, BATCH THESE
    record (Δ̂, Δ, ||e(v) − e(x_i⁰)||) for all v
```

Cost: 200 × 2000 = 400k forward passes of GPT-2 Large at batch 64. A few hours on
the A6000. Entirely tractable. You can drop to N=100 if time is tight.

### Plot 1A — the money figure

Scatter plot. x-axis: `Δ̂(v)` (surrogate predicted energy change). y-axis: `Δ(v)`
(true energy change). One point per (sequence, candidate) pair, subsampled for
legibility. Colour points by `||e(v) − e(x_i⁰)||` on a continuous colourmap.

Overlay the identity line y = x.

**Predicted result:** a formless cloud with near-zero correlation. Report Pearson
r and Spearman ρ in the corner of the plot.

Caption: "The Taylor surrogate used by the discrete Langevin proposal is
uncorrelated with the true change in sequence energy. Ranking candidate tokens by
this surrogate is therefore equivalent to ranking them at random, which explains
the null result of Section [gradient fallacy]."

### Plot 1B — the mechanism figure

This is the one that turns the observation into an explanation.

Bin the candidates by embedding distance `d = ||e(v) − e(x_i⁰)||` into ~15 bins.
For each bin, compute Spearman ρ between Δ̂ and Δ *within that bin*.

Line plot. x-axis: embedding distance d. y-axis: Spearman ρ.

Add a vertical dashed line at d = 4.5, labelled "mean inter-token distance in
GPT-2". Add a second vertical dashed line at d = 0.5 labelled "Llama-3", to
connect back to your step-size calibration section.

**Predicted result:** ρ is meaningfully positive at small d and decays to zero
well before d = 4.5. The point where ρ crosses some threshold (say 0.1) IS the
linearization radius r, and you should report it as a number.

Caption: "The gradient surrogate is informative only within a radius r ≈ [X] of
the current embedding. The smallest possible discrete move in GPT-2's embedding
space has length ≈ 4.5, which lies far outside this radius. Gradient guidance is
therefore not merely difficult on this landscape; it is inapplicable, because
every admissible step leaves the region in which the gradient carries information."

**If you produce nothing else from this list, produce Plot 1B.** It is the single
figure that converts your thesis from a set of negative results into a mechanism.

### Plot 1C — top-k recall (optional but strong)

For each sequence, take the top-10 candidates ranked by Δ̂, and compute how many
appear in the top-10 ranked by true Δ. Compare against:
  - a random ranking baseline (expected overlap ≈ 10 × 10 / |V'|)
  - an oracle

Bar chart with three bars. Predicted result: gradient ranking ≈ random ranking.

---

## EXPERIMENT 2 (CRITICAL) — Acceptance Rate Conditioned on Boundary Crossing

**This turns the CLS/MH result from an observation into a proof.**

### Procedure

Re-run one CLS configuration (gn-off, MH on, 50 steps, N=200 sequences) with
extra logging. At every step, for every proposal, record:

  - `crossed`: did `argmax_nn(s_prop) != argmax_nn(s_t)`? (i.e. did the nearest-
    neighbour token change?)
  - `accepted`: did MH accept?
  - `log_alpha`, and its two components separately:
        `log_target_ratio  = log π(s_prop) − log π(s_t)`
        `log_proposal_ratio = log N(s_t | m_prop, ε) − log N(s_prop | m_s, ε)`

### Plot 2A

Grouped bar chart. Two groups on the x-axis: "proposal stayed in cell" and
"proposal crossed a boundary". y-axis: acceptance rate.

**Predicted result:** high acceptance for within-cell moves, near-zero for
boundary crossings.

Caption, and this is the sentence to build the section around:
"Metropolis-Hastings accepts only those proposals that do not change the
projected token. The set of accepted moves and the set of moves that do anything
are disjoint."

### Plot 2B — decomposing the rejection

Two overlaid histograms of `log_proposal_ratio`, one for within-cell proposals and
one for boundary-crossing proposals. Do the same for `log_target_ratio`.

**Predicted result:** the target ratio is *fine* for boundary crossings (many
crossings improve the sequence), but the proposal ratio collapses to a large
negative number. This proves the rejection is driven by the **proposal** term and
not the target term, which is the precise, non-hand-wavy version of your claim.

This directly supports the theoretical argument: MALA's validity requires
∇ log π to be Lipschitz (Roberts & Tweedie 1996), and under nearest-neighbour
projection it is discontinuous at every cell boundary, so m_prop computed at
s_prop is unrelated to m_s computed at s_t, and the reverse density is evaluated
deep in the tail.

---

## EXPERIMENT 3 (IMPORTANT) — Embedding-Space Trajectory Visualisation

This is the section you lost when the company closed. Recreate it; it is cheap and
it is a strong visual.

### Procedure

Fit a PCA (2 components) on the full GPT-2 Large token embedding matrix (50257 ×
1280). Do NOT fit it on the trajectory; fit on the vocabulary, then project the
trajectory into that fixed basis. This matters: it means the background cloud and
the trajectory live in the same, interpretable space.

Then, for a single representative sequence and a single masked position, log the
state `s_t` at every one of the 50 steps for four configurations:
  (a) DLS, MH on, gn on
  (b) CLS, gn ON  (the "rubber band" case)
  (c) CLS, gn OFF, MH off
  (d) CLS, gn OFF, MH on  (the paralysed case)

### Plot 3A — 2×2 panel

Each panel: scatter the full vocabulary embedding cloud in light grey (subsample
to 5000 points for file size). Overlay the trajectory as a line with markers,
coloured by step index (viridis). Mark the ground-truth token embedding with a
star, and the initial corrupted token with an X.

**Predicted appearance, and what to write about each:**

(a) DLS: discrete hops. The state is always exactly on a vocabulary point,
    because DLS never leaves token space. Shows the sampler is genuinely moving.

(b) CLS gn-on: the trajectory is a tight ball around the starting point. It never
    escapes. This is the rubber-band effect, and now you can *see* it.

(c) CLS gn-off, no MH: the trajectory wanders, and critically, it wanders into the
    *interior* of the embedding cloud where no tokens live. This is the key
    observation: the continuous state spends most of its time off the token
    manifold, in regions the LM has never been trained to evaluate.

(d) CLS gn-off, MH on: a single point. Nothing moves.

### Plot 3B — distance to the token manifold over time

Line plot. x-axis: optimization step. y-axis: `min_v ||s_t − e(v)||`, the distance
from the continuous state to the nearest real token embedding.

Four lines, one per configuration above.

**Why this matters:** it quantifies panel (c). If CLS spends its time at a
distance from the manifold that is comparable to or larger than the inter-token
distance, then the LM is being evaluated on inputs that are nothing like anything
it saw in training, and the "energy" it reports is an extrapolation with no
meaning. That is a *third* independent reason CLS cannot work, and it is worth a
paragraph.

---

## EXPERIMENT 4 (IMPORTANT) — The Likelihood Trap, Demonstrated on Your Own Model

You need to establish, on YOUR model and YOUR data, that low energy does not mean
good text. Otherwise Mechanism 2 is borrowed from Holtzman et al. rather than
demonstrated.

### Procedure

Take 500 ROCStories contexts. For each, generate the middle sentence four ways:
  - ancestral sampling (temperature 1.0)
  - top-p 0.9
  - beam search, beam = 20
  - greedy

For each generated sentence compute:
  - total log p (the energy, negated)
  - mean per-token log p
  - length
  - a quality proxy: BERTScore against the reference, and/or repetition rate
    (fraction of 4-grams that are repeats)

### Plot 4A — the trap, in one chart

Scatter. x-axis: mean per-token log p (i.e. negative energy per token, higher =
"better" according to the energy function). y-axis: repetition rate, or
BERTScore.

Colour by decoding method.

**Predicted result:** beam search and greedy sit at the far right (highest
likelihood) and are the WORST on quality. Ancestral sampling sits in the middle on
likelihood and is the best on quality.

Caption: "Sequences with the lowest energy under the model are not the best
sequences. Beam search, which approximates the mode of p, produces the most
repetitive and least faithful text. Any procedure that concentrates probability
mass on low-energy regions, including annealed Langevin dynamics and a GFlowNet
trained on a tilted likelihood reward, is therefore optimising toward degenerate
output by construction."

### Plot 4B — length vs total log p

Simple scatter of sequence length against total log p, over your ROCStories eval
set (just score the references, plus your generations).

**Predicted result:** a clean, strong negative linear relationship. Fit the line
and report the slope; the slope is `−H̄`, the model's mean per-token entropy.

Caption: "The unnormalised reward used by the GFlowNet is dominated by a length
term, R ≈ −H̄·L with H̄ = [slope] nats. Length collapse is therefore not an
artifact of the optimizer. It is the correct behaviour of an optimizer applied to
this reward."

**This single plot retroactively explains failure mode 2 as a prediction rather
than a surprise, which is a completely different quality of result.**

---

## EXPERIMENT 5 (NICE TO HAVE) — Anisotropy quantification

Cheap, one script, no forward passes.

Compute over the GPT-2 Large embedding matrix E:
  - μ = mean(E), the mean embedding
  - report ||μ|| and mean over v of ||e(v) − μ||
  - the ratio ||μ|| / E||e(v) − μ|| quantifies the cone effect
  - the fraction of total embedding variance in the top 1, 5, 10 principal
    components
  - mean pairwise cosine similarity between random token pairs (should be well
    above 0, this is the anisotropy)
  - THE NUMBER YOU CITE EVERYWHERE: mean pairwise L2 distance. You claim 4.5 for
    GPT-2 and 0.5 for Llama. VERIFY BOTH. They appear in three places in the
    thesis and in your talk.

### Plot 5A
Histogram of pairwise L2 distances between random token pairs, GPT-2 vs Llama,
two panels (different x-scales, do not share the axis).

Also compute, for the gradient g at a masked position: what fraction of ||g|| lies
along the mean-embedding direction μ? If most of the gradient points along the
cone axis, then the component that actually discriminates between tokens is a
small residual, which is a supporting mechanism for the gradient fallacy.

---

## FIXES TO EXISTING PLOTS

### Gradient fallacy bar chart
- Add a y-axis label: "Δ KL divergence (grad-norm-preserved − policy)"
- Add error bars (± 1 SE). Right now you assert "below two standard errors" in
  prose and show no uncertainty in the figure. That mismatch invites attack.
- Either ADD the fifth bar (gfn-lb1-500) so the chart matches the claim of "five
  energy functions", or state explicitly in the caption why it is excluded.
- Add a note stating the configuration: "gradient normalization OFF, MH ON,
  50 steps, DLS."

### Langevin-on-GFlowNet-energy bar chart
- The y-axis is truncated (5.5 to 7.2), which visually exaggerates differences
  while your caption says "within noise". Either start at zero, or add error bars
  so the overlapping intervals do the work, or annotate the truncation explicitly.
- Add a y-axis label: "Final KL divergence after 50 DLS steps"
- Note that this chart uses gn-ON while the gradient fallacy chart uses gn-OFF.
  These are different configurations and the reader must be told.

### All existing trajectory plots
- Every axis needs a label. The grading criteria list "correctness of
  visualization (axes labels, ...)" as an explicit line item. You currently have
  several figures where the y-axis is bare.
- Every figure needs a unique \label. Your current .tex reuses `\label{fig:sample}`
  for SIX different figures, which means every \ref points at the last one.

---

## NUMBERS TO VERIFY BEFORE ANYTHING GOES IN THE THESIS

Everything below is currently marked TODO in your draft or was reconstructed from
our conversation. Pull each from the CSVs and confirm.

| Claim | Current value | Source to check |
|---|---|---|
| Total runs | 145 | manifest |
| Sequences per run | 200 | manifest |
| GPT-2 mean inter-token L2 | 4.5 | recompute |
| Llama-3 mean inter-token L2 | 0.5 | recompute |
| GPT-2 step-size multiplier | 10.5 | oracle sweep |
| KL gap, policy vs gnp, GPT-2 SFT | 0.045 | allruns.csv |
| KL gap, GFN lb0-500 | 0.008 | allruns.csv |
| KL gap, GFN lb0-2000 | 0.146 | allruns.csv |
| KL gap, Llama-3 | 0.016 | allruns.csv |
| Llama no-MH policy penalty | +1.225 KL, 2.5 SE | allruns.csv |
| Final KL, SFT base | 6.54 | final_kl_by_model.csv |
| Final KL, GFN lb0-500 | 6.31 | final_kl_by_model.csv |
| Final KL, GFN lb0-2000 | 6.72 | final_kl_by_model.csv |
| Final KL, GFN lb1-500 | 6.42 | final_kl_by_model.csv |
| SFT mean gen length | 24.9 | gfn eval |
| GFN lb0-500 mean length | 6.9 | gfn eval |
| GFN lb0-500 % ≤8 tokens | 82% | gfn eval |
| GFN lb0-2000 mean length | 10.5 | gfn eval |
| GFN lb1-500 mean length | 24.6 | gfn eval |
| GFN lb1-500 BERTScore F1 | −0.53 | gfn eval |
| CLS MH acceptance rate | "≈0" | needs an actual number |
| DLS MH acceptance rate | not reported | GET THIS. It matters. |

Also: your draft says GPT-2 and Llama "differ in scale by two orders of
magnitude." 774M vs 8B is one order. Fix.

And your Methods section says D = 4096, which is Llama's embedding dimension.
GPT-2 Large is 1280. State both.
