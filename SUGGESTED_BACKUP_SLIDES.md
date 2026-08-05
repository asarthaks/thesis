# SUGGESTED_BACKUP_SLIDES.md

SUGGESTIONS ONLY. Nothing here has been added to `Doc/final/beamer/Presentation.tex` or
to any other file under `Doc/`, and nothing here should be inserted without the author's
decision. The deck is untouched.

Ranked by how likely the question is at the defense. Each entry gives the slide's
content, the figure or table it would use, and three lines of speaker notes.

Every number traces to a JSON listed in `CTG_EXPERIMENTS_LOG.md`.

---

## S1. "Did you actually test the method you criticise?" (MuCoLa vs the thesis energy)

RANK 1. This is the most likely hostile question, the thesis already anticipates it in
Section 6.2 limitation 1, and this is now a measurement rather than an argument.

**Slide**: two columns, one generated continuation each, from IDENTICAL code differing
only in how the target token enters the language-model term.

| | `mucola_faithful` | `cls_thesis` (this repository's energy) |
|---|---|---|
| sample output | "Once upon a time, there was a king who ruled over a large land. His subjects were very happy, and he" | "Once upon a time environmentalCNN Unicornrosso PE ##YD vitality ped Eh Karn bladesActionCodeStreamerBot Rothido Osborne" |
| mean final LM log-prob | -38.5 | -353.9 |
| gradient at the final position | h_{n-1}, norm 50.5592, cosine 1.000000 | exactly 0.000e+00 |
| steering-head "accuracy" | 40.0% | 53.3% |

**Source**: `results/ctg/studyD/D.mucola_faithful.*`, `D.cls_thesis.*`; verification in
`CTG_EXPERIMENTS_LOG.md` section "The `mucola_faithful` arm".

**Speaker notes**
- Same optimizer, same projection, same init, same steps, same constraint weight; the
  only difference is whether the target enters the softmax numerator as the continuous
  state or as a discrete index.
- MuCoLa's energy produces fluent English; the energy I implemented produces word salad,
  and the final-position gradient identity says exactly why: 50.56 against exactly zero.
- Note the trap in the last row: the blind energy scores HIGHER on the steering
  classifier while producing gibberish, which is why every adherence number in my work
  is paired with an external fluency number.

---

## S2. "Your 40 percent headline: how solid is it?" (the n=50 issue)

RANK 2. An examiner who checks sample sizes will find this, and it is better to raise it
first. It is a precision correction, not a reversal.

**Slide**: one small table plus one sentence of interpretation.

| n | exact recovery | 95% CI |
|---|---|---|
| 50 (the sweep cell as published) | 40.0% | [26.4, 53.6] |
| 200 (same configuration, rerun) | 26.5% | [20.4, 32.6] |
| input-embedding gradient, n=200 | 0.0% | -- |

**Source**: `results/ctg/studyA/A.fullvocab.policy_onehot.json` and
`results/grid/rev3/ohsweep_e10p5_t1p0.json`.

**Speaker notes**
- The published number reproduces exactly: rerunning the configuration and taking the
  same first 50 sequences gives 40.0 percent, so nothing is wrong with the code or the
  data.
- At n=200 it settles at 26.5 percent, so the sweep cell was an optimistic estimate, and
  the comparison to SEDD at 39 percent and RoBERTa at 44.5 percent, both measured at
  n=200, does not hold at equal sample size.
- What the thesis actually rests on is unchanged and now measured on four times the
  data: 26.5 percent against 0.0 percent, a paired gap of 24.5 points, CI [18.5, 30.5].

---

## S3. "Is your sampler efficient, or just expensive?" (the cost of the future term)

RANK 3. The natural follow-up to the one-hot result and the cleanest positive finding.

**Slide**: quality-versus-cost table, full vocabulary, n=200.

| arm | exact % | KL | fwd-equiv / seq | backward passes |
|---|---|---|---|---|
| self term only | 24.5 | 4.598 | 100 | **0** |
| self + linearized future (the thesis's one-hot) | 26.5 | 4.584 | 400 | 10,000 |
| input-embedding gradient | 0.0 | 6.243 | 300 | 10,000 |
| uniform | 0.0 | 6.671 | 100 | 0 |

Paired: onehot minus self = +2.0 exact-match points, CI [-0.5, +5.0]; KL -0.015 nats,
CI [-0.361, +0.331].

**Source**: `results/ctg/studyA_summary.json`.

**Speaker notes**
- The one-hot surrogate splits into a self term that the forward pass already computes
  for free and a future term that costs a backward pass.
- The self term alone gets the whole result: the future term's contribution is +2.0
  points with a confidence interval straddling zero, for four times the compute.
- So the constructive half of my thesis does not need a gradient at all; it needs the
  model's own next-token distribution, which every autoregressive model already exposes.

---

## S4. "Doesn't the shortlist do all the work?" (the floor nobody measured)

RANK 4. A sharp methodological question, and the honest answer strengthens the thesis's
null rather than weakening it.

**Slide**: one small table, arms by shortlist width.

| arm | full vocab | k=256 | k=64 | k=16 |
|---|---|---|---|---|
| uniform (no scoring at all) | 0.0 | 8.0 | 18.5 | **31.5** |
| input-embedding gradient | 0.0 | 8.5 | 21.5 | **30.5** |
| self term | 24.5 | 26.5 | 29.5 | 30.0 |

**Source**: `results/ctg/studyA_summary.json`.

**Speaker notes**
- At a 16-token shortlist every arm lands between 30 and 31.5 percent, including the one
  that scores nothing, so at that width the support and the exact accept step are doing
  all the work and no surrogate is demonstrated.
- This does not rehabilitate the input-embedding gradient: its correct control is the
  uniform arm, and it tracks that floor at every width, never separating from it.
- It does mean any claim of the form "surrogate X works in a shortlisted sampler" has to
  be quoted against the uniform-on-the-same-shortlist floor, which is a control my thesis
  did not run and which I would add if I did this again.

---

## S5. "When IS the expensive computation worth it?" (the exact future term)

RANK 5. Shows the programme found a positive result, not only nulls, and that a
pre-registered prediction was refuted and reported.

**Slide**: exact-future arm against the self term, by shortlist width.

| k | delta KL (exact_k minus self) | delta exact (pts) | cost, fwd-equiv | cost, wall |
|---|---|---|---|---|
| 16 | +0.090 [-0.264, +0.442] | +0.5 [-3.0, +4.0] | x1.17 | x1.04 |
| 64 | **-0.555 [-0.977, -0.161]** | +1.5 [-2.0, +5.5] | x1.65 | **x0.86** |
| 256 | **-0.601 [-1.008, -0.205]** | **+4.5 [+0.0, +9.0]** | x3.57 | x1.32 |

**Speaker notes**
- I pre-registered that the exact future term would buy little for its cost; that
  prediction was wrong at k of 64 and above and I am reporting it as wrong.
- Once the shortlist is wide enough for ranking to matter it buys about 0.6 nats, and at
  k=64 it does so at 0.86 times the wall-clock, because its acceptance rate is 73 percent
  against 26 percent so it wastes far fewer steps.
- Caveat I would state unprompted: with a single masked position the candidate energies
  are computable once per sequence, which is a task-specific economy that will not
  survive to a task where many positions move.

---

## S6. Study C: how should a constraint enter? (coverage FIRST)

RANK 4 to 5. A mechanism finding, and the only one in the phase that explains WHY
classifier-gradient guidance is weak rather than just reporting that it is.

**Slide**: one table, coverage before adherence.

| k | coverage of the constraint's top candidate | chance level | median likelihood rank of it | constraint-side linearization Spearman |
|---|---|---|---|---|
| 16 | 0.2% | 0.032% | 29,928 / 50,257 | +0.019 |
| 64 | 0.8% | 0.127% | 34,227 / 50,257 | +0.040 |
| 256 | 1.7% | 0.509% | 29,866 / 50,257 | -0.008 |

For contrast, on the likelihood side: token-indicator surrogate +0.60 to +0.73,
input-embedding gradient about 0.0.

**Source**: `results/ctg/phase2/Ccov.k{16,64,256}.json`.

**Speaker notes**
- Before comparing mechanisms I check whether the shortlist even contains what the
  constraint wants: the median likelihood rank of its preferred token is about thirty
  thousand out of fifty thousand, so the two rankings are nearly independent.
- Coverage is three to six times chance, so the association is real, but it is under two
  percent even at k=256, which means a weak exact-rescoring arm is a support limitation
  and not a verdict on rescoring as a mechanism.
- And the classifier's own first-order expansion has a Spearman near zero against the
  true change it is expanding, so it is in the same condition as the input-embedding
  gradient my thesis is about, which gives the gradient-in-proposal arm a mechanism
  before I even report its number.

---

## S6b. Placeholder, Study C mechanism comparison

Pending. Will carry the shortlist-coverage diagnostic FIRST (the likelihood rank of the
candidate the constraint most prefers, and the fraction of steps where it falls inside
the shortlist at k in {16, 64, 256}), then the mechanism comparison, then the
constraint-side linearization correlation.

Early smoke-scale reading, NOT a result: coverage 0.0 percent at k=64 with a median
likelihood rank of 27,464 out of 50,257 for the constraint's most-preferred token, and a
constraint-side linearization Spearman of 0.14. If that survives at scale it means a
likelihood-ranked shortlist and a sentiment constraint want almost unrelated tokens, and
any weak rescoring arm is a support limitation rather than evidence about the mechanism.

---

## S7. Placeholder, Study D: the head-to-head at matched compute

Pending. Adherence under the held-out judge against external fluency, as a CURVE over the
constraint weight for every method, reported at matched wall-clock and at matched total
forward-equivalent passes.

---

## S8. Placeholder, Study E: the sampling claim

Pending. Distinct-n, self-BLEU and embedding-based semantic spread at matched adherence
and matched compute, chain against a compute-matched best-of-N optimizer over the same
energy.
