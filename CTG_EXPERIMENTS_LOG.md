# CTG_EXPERIMENTS_LOG.md

Defense-preparation experiment programme: crossing the bridge from the diagnostic
recovery task to controllable generation. The thesis is SUBMITTED. Nothing under
`Doc/` is modified by this phase. Slide material is SUGGESTED only, in
`SUGGESTED_BACKUP_SLIDES.md`.

---

## STATUS

<!-- OVERWRITTEN after every completed arm or family -->

**CONTRADICTS SUBMITTED CLAIM (1).** `05_results.tex:413` states the token-indicator
surrogate "recovers 36 to 40 percent of corrupted tokens" and `:462` calls this "recovery
comparable to a purpose-trained diffusion model". The 40.0 percent reproduces EXACTLY at
the thesis's n=50, but at n=200 the same cell gives **26.5 percent [20.4, 32.6]**, below
the stated range. Its comparators (SEDD 39.0, RoBERTa 44.5) were n=200, so the parity
claim does not hold at equal n. The qualitative finding is UNAFFECTED and now rests on 4x
the data: 26.5 against 0.0 percent, paired difference 24.5 points [18.5, 30.5]. See A.5.1.

**CONFIRMS SUBMITTED CLAIM (notable).** The MuCoLa scoping in `06_discussion.tex:50` is
demonstrated, not merely argued: same code, only the energy differs, MuCoLa's writes
fluent English (held-out perplexity 22.5 to 25.3) and the thesis's writes word salad
(18,034 to 40,783). Final-position gradient h_{n-1} (cosine 1.000000) versus exactly
0.000e+00.

**Current study:** A (both halves) COMPLETE, C COMPLETE, D COMPLETE in its unnormalized
parameterization. Running: Study B (six proposal sources) and Dn, the corrected Study D
sweep with gradient normalization.

**Jobs:** 124 of 124 phase-2 jobs done, 0 failed. Phase 3 is 96 jobs, 20 workers, all
five cards at 100 percent utilization, 23 GB of 48 used, 0 failed.
**GPUs busy:** 5 of 5 available. GPUs 2-5 belong to another user and were never touched.
**Elapsed:** ~5.5 h | **Budget consumed:** ~55 of 400 GPU-hours
**Headline so far, in one line:** the MH chain returns FOUR DISTINCT solutions per prompt
where MuCoLa's optimizer returns literally one (self-BLEU 0.000 against exactly 1.000),
and MuCoLa is ~135x more fluent, so the answer to "is your sampler useful" is a real
trade-off measured on both axes rather than a win or a loss.
**Next decision point:** Study B lands -> Q2 (does bidirectional conditioning help when
the attribute is global?) -> then Study E if wall-clock permits.
**Red flags:** none open. Four defects were found and fixed BEFORE they contaminated a
reported number: three reversibility/support bugs (A.2), one OOM in the shared projection
helper, and one constraint-weight sweep that ran backwards through an optimizer
pathology, which would have made the head-to-head unfair to MuCoLa.
**Not run this session, stated rather than hidden:** Study E (diversity at matched
adherence against a compute-matched best-of-N optimizer). Note that Study D delivered the
core of the diversity question incidentally and decisively; what is missing is the
matched-adherence, matched-compute framing.
---

# Part 0: audit and design

Date: 2026-07-29. No GPU consumed.

## 0.1 What already exists for constrained generation, and is REUSED

`scripts/run_constrained.py` (371 lines) already defines the whole constrained task.
It is reused as the task definition; the arms are extended around it, and nothing
about the task is rebuilt.

| element | what is implemented | reused how |
|---|---|---|
| prompts | `MUCOLA_PROMPTS`, the 15 PPLM discriminator prompts verbatim from MuCoLa's repo (`data/control-prompts/pplm-discrim-prompts/prompts.txt`) | imported, unchanged |
| task A | `continuation`: fixed prompt + `span_len=20` NEW tokens optimized jointly | imported, unchanged |
| task B | `infill`: real WikiText-2 sentence, corrupt `num_masks` tokens, recover | imported, unchanged |
| init | `make_init_s`: `centroid` (MuCoLa's near-uniform simplex "zeros" analogue) or `random_token` | imported, unchanged |
| samples | 15 prompts x `samples_per_prompt=20` = 300 sequences | unchanged |
| setups | `mucola` (steps 300, span 20, eps 5.0->0.05, betas 0.8/0.2, no MH, no gradnorm) and `ours` (steps 50, eps 10.5->0.1, MH on) | both kept as labelled arms |
| arms | `constraint_mode` in {lm_only, full, cons_only, cons_random, random} | kept; new arms added alongside |
| steering classifier | `core/constraint.py:SentimentHead`, a 2-layer head on GPT-2-Large's final hidden state, pooled over the sequence. Checkpoint at `$SENTIMENT_HEAD` | kept as the STEERING classifier only |
| metrics | `sentiment_acc_before`, `sentiment_acc`, `steering_gain`, `final_kl` | kept, and extended |

Two properties of the existing task that constrain the design:

1. The sentiment classifier that steers reads GPT-2-Large's own hidden states. It is
   therefore NOT admissible as a scorer. A held-out judge is required and is
   specified in 0.5.
2. `run_constrained.py` records only aggregate scalars per run, plus 12 example
   texts. Every adherence/fluency/diversity question in this phase needs the full
   generated text, so the new harness writes per-sequence text to CSV and the
   scoring runs as a separate pass. This is the generate/score split that already
   worked for `run_external_judge.py`.

## 0.2 What `policy_onehot` computes today (decides the Study A arm list)

VERIFIED by reading the code, not from memory.

`core/dls.py:63-67`:

```python
grad_dot_emb = torch.matmul(grad_s, self.emb_matrix.T)
grad_dot_s   = torch.sum(grad_s * s_detached, dim=1, keepdim=True)
t2 = 0.5 * (grad_dot_emb - grad_dot_s)          # embedding-gradient (future) term
if onehot:
    t2 = t2 + self._onehot_bonus(self_lp)        # + 0.5 * log p(v | x_<m)
```

and `core/dls.py:31-39`, `_onehot_bonus` returns `0.5 * self_logprobs`, where
`self_logprobs` is `log p(v | x_{<m})` read from the same forward pass
(`core/base_sampler.py:62-72`).

**ANSWER: `policy_onehot` = self term PLUS the input-embedding-gradient term.** It is
not the self term alone. It costs one forward + one backward + one extra no-grad
forward per step (and the same again for the MH reverse evaluation).

Consequence for Study A: `policy_onehot` IS arm A2 (self + linearized future term).
Arm A1 (self term only, no backward pass at all) does NOT exist and is new. So is A3
(self + exact future term over a shortlist) and A5 (uniform). A4 is the existing
`policy`.

This also means the thesis's headline 40% one-hot recovery number has never been
decomposed. Study A's first job is precisely that decomposition, and it is cheap.

## 0.3 The `mucola_faithful` arm

The thesis's CLS is a variant of MuCoLa, not MuCoLa. Verified against the paper text
already quoted in `MUCOLA_CORRECTION_PROPOSED.md` (Kumar, Paria and Tsvetkov 2022,
section 3, "Energy as a function of embeddings"):

> the softmax probability is computed as
> `P(y_{n+1}|y_{1:n},x) = exp(h_n^T e_{n+1} + b_{n+1}) / sum_j exp(h_n^T e_j + b_j)`
> ... By replacing `e_{n+1}` with `e~_{n+1}`, we convert the above probability to
> `P(e~_{n+1}|e~_{1:n},x)`. For each position n+1, `e~_{n+1}` receives gradients,
> (a) directly from `-log P` function and (b) through `h_{n+1}` via back-propagation
> through the network layers.

So MuCoLa's per-position LM term is

    log P(e~_{n+1} | e~_{1:n}) = h_n . e~_{n+1} + b_{n+1} - logsumexp_j( h_n . e_j + b_j )

with the NUMERATOR taking the continuous state e~ and the DENOMINATOR summing over the
real embedding table. GPT-2's `lm_head` is tied to the input embedding and carries no
bias, so b = 0 here. Gradient path (a), the direct one, is exactly `h_n`.

The repository's CLS instead calls `core/prep.py:joint_log_prob_from_inputs_embeds`,
which is a hard-target cross-entropy against `target_ids[0, mask] = s_idx`
(`core/base_sampler.py:50-53`), i.e. the target enters as a discrete INDEX. Path (a)
is absent by construction. That is the difference, and it is the whole subject of the
thesis's null.

`mucola_faithful` is therefore implemented as: continuous embedding state, MuCoLa
energy above (self term present), fixed constraint weight (beta_lm 0.8 / beta_c 0.2,
their weighting), nearest-embedding projection after every update, and NO
Metropolis-Hastings, because their EmbedGD optimizer has none. The existing CLS stays
as its own labelled arm, `cls_thesis`. The two are reported as distinct arms
everywhere. The CLS is never called "MuCoLa" in this log or in the suggested slides.

The difference between `mucola_faithful` and `cls_thesis` is itself a measurement, and
is the cleanest possible answer to "did you actually test the method you criticise".

## 0.4 Task/harness split

| task | harness | why |
|---|---|---|
| recovery (infill, 1 mask, WikiText-2, n=200) | EXISTING `scripts/run_experiment.py`, new methods added to `core/dls.py` | directly comparable to the thesis grid numbers (one-hot 40.0%, input-embedding 2.0% max). Extending arms, not rebuilding. |
| constrained generation (continuation, 15 prompts x 20 = 300 seqs, span 20) | NEW `diagnostics/run_ctg.py`, an exact-energy discrete MH chain with pluggable proposal and constraint mechanism | `run_constrained.py`'s task definition is imported; only the sampler and the recording are new. Needed because the constrained studies require per-sequence text out, cost counters, shortlist coverage, and proposals (RoBERTa, SEDD) that the Langevin sampler cannot express. |
| held-out scoring | NEW `diagnostics/run_ctg_judge.py`, one batched pass over accumulated text | generation/scoring split, per the pattern that already worked |

Shared helpers are IMPORTED, never reimplemented: `build_corruption`, `load_texts`,
`seed_all` from `scripts/run_experiment.py`; `MUCOLA_PROMPTS`,
`build_continuation_case`, `make_init_s`, `classify` from `scripts/run_constrained.py`;
`load_sentiment_head` from `core/constraint.py`; `load_tokenizer_and_model`,
`project_to_vocab_by_l2`, `joint_log_prob_from_inputs_embeds` from `core/prep.py`;
the tokenizer bridge from `diagnostics/run_mlm_control.py`.

## 0.5 Strict classifier separation

| role | model | used for | never used for |
|---|---|---|---|
| STEERS | `core/constraint.py:SentimentHead` on GPT-2-Large hidden states, `$SENTIMENT_HEAD` | the constraint term inside the energy and inside the proposal | any reported adherence number |
| SCORES adherence | `siebert/sentiment-roberta-large-english`, an off-the-shelf sentiment classifier with no connection to this project | the reported adherence | any gradient, any proposal, any accept step |
| SCORES fluency | Llama-3-8B at `/mount/arbeitsdaten/studenten1/singhsk/models/llama3-8b`, already the judge in `run_external_judge.py` | external per-token NLL / perplexity | anything inside the chain |

The judge models are loaded only in `run_ctg_judge.py`, which runs after generation
and never co-resident with a generator. No judge is imported by any chain code; this
is enforced by the file split.

## 0.6 Available compute

`fleckenweihe.ims.uni-stuttgart.de`, 9x RTX A6000 48 GB, 1007 GB host RAM.
GPUs 2, 3, 4, 5 are occupied by another user (12.5-13.0 GB, 70-100% util) and are left
alone. **GPUs 0, 1, 6, 7, 8 are free: 5 cards.**

Per-job VRAM: GPT-2 Large fp32 ~4 GB, plus activations at these sequence lengths
(<= 64 tokens) under 2 GB. RoBERTa-large fp32 ~1.5 GB. SEDD-small/medium ~1.5/3 GB.
Two generator jobs per card is safe at 48 GB and well inside host RAM at this worker
count (the 36-worker host-RAM incident of REVISION_LOG C7 came from 36 concurrent
GPT-2 Large CPU-side loads; 10 workers is ~14 GB of transient host RAM).

**per_gpu = 2, 5 cards = 10 concurrent workers.** Fresh status dir `status/ctg`.

</content>

---

# Study A: is the future term worth its cost?

## A.0 PRE-REGISTRATION (written before any Study A job was launched)

Question Q1. The output-side surrogate decomposes into a SELF term,
`log p(v | x_<i)`, which the energy's own forward pass already computes and which is
therefore free, and a FUTURE term describing how substituting v changes everything
downstream, which costs either one backward pass (linearized) or k forward passes
(exact). Is the future term worth its cost?

Arms, all inside the exact-energy Metropolis-Hastings chain:

| arm | proposal score for candidate v | model calls per step |
|---|---|---|
| A1 `policy_self` | `0.5 log p(v \| x_<m)` | 1 fwd, 0 bwd |
| A2 `policy_onehot` | A1 + `0.5 g^T(e(v) - s)` | 2 fwd, 1 bwd |
| A3 `policy_exact_k` | `0.5 (E(v) - E(cur))` on the shortlist | 1 fwd + k fwd (amortized, see A.2) |
| A4 `policy` | `0.5 g^T(e(v) - s)` alone | 1 fwd, 1 bwd |
| A5 `uniform` | constant | 1 fwd, 0 bwd |

PREDICTIONS, recorded in advance:

1. A1 approximately equals A2 at strictly lower cost.
2. A3 buys little over A1 relative to its cost.
3. A4 stays at the floor, near A5.

If A1 collapses relative to A2, the future term matters, the efficiency story dies,
and that will be reported plainly.

## A.1 Two framings, and why both are run

FRAMING 1, full vocabulary. Every arm proposes over all 50,257 tokens with the
Langevin distance term present, in the exact cell that produced the thesis's headline
one-hot result (eps 10.5 -> 0.1, T = 1.0, MH on, grad-norm off, REVISION_LOG C9).
Directly comparable to the published 40.0 percent one-hot and 2.0 percent
input-embedding figures. A3 cannot appear: exact scoring of the full vocabulary would
cost 50,257 forward passes per step.

FRAMING 2, matched shortlist. Every arm proposes over the SAME frozen top-k candidate
set (k in {16, 64, 256}) scored only by its own surrogate, distance term dropped. This
is the clean Q1 comparison: the exact arm can only ever see a shortlist, so the
restriction is charged to every arm equally instead of to the exact arm alone.

## A.2 Build record, including three bugs found and fixed before any real run

New sampler methods in `core/dls.py` (`policy_self`, `policy_exact_k`, `uniform`), a
gradient-free energy path in `core/base_sampler.py`, and cost counters. Every new code
path is gated, so the archived grid is untouched.

REGRESSION GATE, run twice, after the first and after the last edit: the flagship
config `gpt2-large.dls.policy.mh.gn.free.s50` reproduces the archived
`results/grid/gpt2_v2/` CSV BIT-IDENTICALLY over 8 samples x 50 steps, max abs
difference 0.000e+00 on avg_l2_distance, avg_kl_divergence and entropy.

Three defects were found by smoke tests and fixed. They are recorded because each one
would have produced a confident and wrong answer to Q1.

1. **Shortlist excludes the incumbent.** The exact arm's proposal put zero mass on the
   token the chain currently held, so the reverse probability was zero, log_alpha was
   -inf, and acceptance was 0.0 percent on every sequence. Fixed by always including
   the incumbent, whose exact energy is the reference constant and therefore free.
2. **The distance term pins a shortlisted chain.** With `proposal_topk=64` and the
   Langevin term t1 present, all five arms returned an identical final KL of 6.777 and
   zero exact matches, because t1 scores the incumbent at exactly 0 and every rival at
   `-||e(v)-s||^2/(2 eps)`. Over 50k candidates that is one preference among many; over
   65 it is decisive. `--drop_distance_term` removes it for the matched framing.
3. **A state-dependent shortlist is not reversible.** A shortlist recomputed at each
   state as "top-k here plus wherever I am" gives the reverse move probability zero as
   soon as the chain leaves its starting token, which it does immediately, since the
   chain starts on a random corrupting token whose self-term rank is far outside the
   top-k. Acceptance was again 0.0 percent on every arm while the proposal itself was
   perfectly healthy: measured directly, 24.6 percent of the proposal mass sat on the
   ground-truth token and the move was worth +19.8 nats of energy, and it was refused
   because the reverse term was -inf. Fixed by FREEZING the support per sequence at the
   chain's initial state, which makes each arm an independence sampler on a fixed
   candidate set, a valid MH kernel.

CONSEQUENCE FOR THE COST ACCOUNTING, and it is favourable to A3 in a way that was not
anticipated. With one masked position the candidate energies `E(v)` do not depend on
the current token, so the k exact evaluations are computed ONCE PER SEQUENCE and reused
for every step and for both directions of every MH decision. A3's advertised cost of
"k forward passes per step" is therefore k forward passes per SEQUENCE on this task.
This will not hold on the constrained task, where 20 positions move, and both numbers
will be reported rather than the convenient one.

## A.3 Smoke test, n=10, 20 steps, k=64 (NOT the result; the machinery check)

| arm | exact % | final KL | accept % | fwd | bwd | fwd-equiv/seq | wall |
|---|---|---|---|---|---|---|---|
| A1 `policy_self` | 50.0 | 2.556 | 48.5 | 400 | 0 | 40.0 | 17.2 s |
| A2 `policy_onehot` | 50.0 | 3.836 | 47.5 | 800 | 400 | 160.0 | 42.2 s |
| A3 `policy_exact_k` | 50.0 | 2.175 | 81.5 | 1050 | 0 | 105.0 | 19.7 s |
| A4 `policy` | 30.0 | 3.458 | 25.0 | 800 | 400 | 160.0 | 40.9 s |
| A5 `uniform` | 20.0 | 3.498 | 23.0 | 400 | 0 | 40.0 | 17.2 s |

n=10 is far too small to conclude anything and no claim is made from it. Two things it
does establish: the harness works, and the cost separation is real (A1 costs a quarter
of A2 in forward-equivalents and 41 percent of its wall-clock).

One observation to carry forward, because it will need stating whatever the headline
says: A5 `uniform` reaches 20 percent here against 0.5 percent for the uniform proposal
over the full vocabulary in REVISION_LOG C8. The shortlist alone, with no scoring
whatsoever, does most of the work. Any claim about a surrogate's value in the matched
framing has to be read against that floor, not against zero.

## A.4 Job matrix and launch

`manifest_ctg_a.tsv`, 76 jobs, out `results/ctg/studyA`, status `status/ctg` (FRESH).

| family | arms | k | shards | jobs | est. per shard |
|---|---|---|---|---|---|
| `A.fullvocab.*` | A1, A2, A4, A5 | full vocab | 4 | 16 | 5 to 12 min |
| `A.k16.*` | all 5 | 16 | 4 | 20 | 5 to 12 min |
| `A.k64.*` | all 5 | 64 | 4 | 20 | 5 to 12 min |
| `A.k256.*` | all 5 | 256 | 4 | 20 | 6 to 15 min |

n=200 sequences, 50 steps, 1 mask, GPT-2 Large fp32. Declared VRAM 16 GB (24 GB for
the k=256 exact arm). Estimated aggregate 12 to 15 GPU-hours; at 10 concurrent workers
on 5 free cards, 1.5 to 2 hours wall-clock. Well inside the 24-hour per-study cap.

## A.5 RESULT, framing 1 (full vocabulary), n=200, COMPLETE

Source: `results/ctg/studyA/A.fullvocab.*.json` and `.csv`, index
`results/ctg/studyA_summary.json`. Cell: eps 10.5 -> 0.1, T = 1.0, MH on with
`--mh_exact_all_arms`, grad-norm off, 50 steps, 1 mask, 200 WikiText-2 sequences.

| arm | exact % | final KL | accept % | fwd-equiv / seq | GPU s / seq | GPU s / accepted move |
|---|---|---|---|---|---|---|
| A1 `policy_self` (self term only) | **24.5** | 4.598 | 25.2 | **100** | **4.18** | **0.332** |
| A2 `policy_onehot` (self + linearized future) | 26.5 | 4.584 | 24.3 | 400 | 10.71 | 0.881 |
| A4 `policy` (input-embedding gradient) | 0.0 | 6.243 | 11.1 | 300 | 7.71 | 1.391 |
| A5 `uniform` | 0.0 | 6.671 | 9.9 | 100 | 4.24 | 0.857 |

Paired on `sample_idx` (the corruption is deterministic per index and identical across
arms, so this needs no rerun), 10,000-sample bootstrap:

| contrast | delta KL | delta exact (points) | cost ratio |
|---|---|---|---|
| A2 - A1 | -0.015 [-0.361, +0.331] | +2.0 [-0.5, +5.0] | x4.00 fwd-equiv, x2.56 wall |
| A4 - A1 | +1.644 [+1.020, +2.275] | -24.5 [-30.5, -18.5] | x3.00 fwd-equiv |
| A5 - A1 | +2.072 [+1.467, +2.684] | -24.5 [-30.5, -18.5] | x1.00 fwd-equiv |

**PREDICTION 1 CONFIRMED, and strongly.** A1 approximately equals A2: the KL contrast is
-0.015 nats with a CI of +-0.35, and the exact-match contrast is +2.0 points with a CI
that contains zero. The entire linearized future term, the thing that costs the backward
pass, buys no measurable quality at four times the forward-equivalent cost and 2.6 times
the wall-clock. **PREDICTION 3 CONFIRMED**: A4 sits at the floor with A5, both at 0.0
percent.

The efficiency statement, put the way the defense question wants it: the thesis's
recovery result is obtainable with NO BACKWARD PASS AT ALL. `policy_self` never calls
autograd once, which the run's own counter certifies (`n_backward` = 0), and it recovers
24.5 percent where the input-embedding gradient recovers 0.0 percent on the same 200
sequences.

This also answers a question the thesis explicitly leaves open. Section 5.6.1
(`05_results.tex:477`) observes that at the FINAL position the future term vanishes so
the token-indicator derivative reduces exactly to `log p(v | x_<i)`, and adds: "In the
interior the two come apart, because the future term is then nonzero." Measured in the
interior, they do not come apart to any degree the data can resolve: +2.0 points,
CI [-0.5, +5.0].

### A.5.1 CONTRADICTS SUBMITTED CLAIM, and it is a real one

The submitted thesis, `Doc/final/thesis/chapters/05_results.tex:413`, states:

> Where the proposal is sharp enough to express a ranking the token-indicator surrogate
> recovers $36$ to $40$ percent of corrupted tokens, while the input-embedding gradient
> never exceeds $2$ percent anywhere in the same grid.

and `05_results.tex:462` describes the outcome as

> recovery comparable to a purpose-trained diffusion model.

The contradicting measurement, on the SAME cell and the SAME code:

| n | sequences | exact % | 95% CI |
|---|---|---|---|
| 50 (the thesis's sweep) | first 50 | **40.0** | [26.4, 53.6] |
| 200 (this run) | first 200 | **26.5** | [20.4, 32.6] |

The thesis's number is REPRODUCIBLE and is not an error: re-running the same
configuration and taking the first 50 sequences gives exactly 40.0 percent. What the
larger sample shows is that 40.0 percent was an n=50 estimate that regresses to 26.5
percent at n=200, below the "36 to 40 percent" range the text states as the finding.

The comparability claim is the part that does not survive. The comparators it is set
against, SEDD-small 38.5 percent, SEDD-medium 39.0 percent and RoBERTa-large 44.5
percent (REVISION_LOG C8, `results/revision/rev_mlm_control.json`), were all measured at
n=200. The thesis therefore compares an n=50 estimate against n=200 estimates. At
matched n=200 the token-indicator sampler reaches 26.5 percent [20.4, 32.6] against
SEDD's 39.0 and RoBERTa's 44.5, and is below both rather than comparable to either.

WHAT SURVIVES UNCHANGED, and it is the load-bearing part. The qualitative claim is not
weakened at all and is now measured at four times the sample size: the token-indicator
surrogate recovers 26.5 percent where the input-embedding gradient recovers 0.0 percent
on the same sequences, a paired difference of 24.5 points, CI [18.5, 30.5]. The
direction, the mechanism and the diagnosis are untouched. What changes is the size of
the number and the claim that it matches a purpose-trained diffusion model.

Suggested one-sentence answer if asked at the defense: "The 40 percent is an n=50 cell
from the sweep and it reproduces exactly; when I ran the same configuration at n=200 to
match the diffusion baselines' sample size it settles at 26.5 percent, so the honest
statement is that the token-indicator surrogate goes from zero to roughly a quarter of
tokens recovered, which is below SEDD and RoBERTa rather than comparable to them, and
the zero-versus-nonzero contrast is what the thesis actually rests on."

---

# The `mucola_faithful` arm: implementation and VERIFICATION

Built as `diagnostics/run_mucola_faithful.py`, which runs TWO continuous optimizers
through identical code differing in nothing but how the target token enters the
language-model term:

  - `--variant mucola_faithful`: MuCoLa's energy, continuous state in the softmax
    NUMERATOR, real embedding table in the denominator;
  - `--variant cls_thesis`: this repository's energy, target entering as the discrete
    nearest-neighbour index.

Both keep the continuous embedding state with projection onto the embedding table after
every update, both use the fixed 0.8 / 0.2 constraint weighting, and NEITHER uses
Metropolis-Hastings, because MuCoLa's EmbedGD optimizer has none. The CLS is never
called "MuCoLa"; it is reported as `cls_thesis` everywhere.

VERIFICATION AGAINST THE PAPER, run before any experiment used the arm
(`scratchpad/verify_mucola.py`, GPT-2 Large, one real sentence):

| check | expected | measured |
|---|---|---|
| MuCoLa energy at a REAL token sequence | equals the ordinary sequence log-likelihood, since e~ = e(x) recovers the standard softmax | -46.909576 vs -46.909576, **abs diff 0.000e+00** |
| gradient at the FINAL position, MuCoLa energy | exactly h_{n-1}, the paper's path (a) | norm 50.5592 vs ‖h_{n-1}‖ 50.5592, **cosine 1.000000** |
| gradient at the FINAL position, thesis CLS energy | exactly zero (the thesis's own theorem) | **0.000e+00** |

The third row is the thesis's final-position theorem reproduced, and the second row is
the reason it does not apply to MuCoLa. On the same sequence, the same model and the
same position, one energy hands the optimizer a gradient of norm 50.56 and the other
hands it exactly zero. That single comparison is the sharpest available statement of
what `MUCOLA_CORRECTION_PROPOSED.md` argues on textual grounds, now measured.

## The two energies, run head to head (smoke scale, n=15, 60 steps)

Identical code, identical optimizer, identical projection, identical centroid init,
identical steps, identical constraint weight. The ONLY difference is whether the target
token enters the softmax numerator as the continuous state or as a discrete index.

| variant | steering-head acc | mean final LM log-prob | fwd-equiv / seq | wall |
|---|---|---|---|---|
| `mucola_faithful` | 40.0% | **-38.5** | 180 | 2.0 min |
| `cls_thesis` | 53.3% | **-353.9** | 240 | 4.2 min |

First generated continuation from each, same prompt, same seed:

> `mucola_faithful`: "Once upon a time, there was a king who ruled over a large land. His
> subjects were very happy, and he"

> `cls_thesis`: "Once upon a time environmentalCNN Unicornrosso PE ##YD vitality ped Eh
> Karn bladesActionCodeStreamerBot Rothido Osborne"

MuCoLa's actual energy produces fluent, coherent English. The energy this repository
implements produces word salad, at a sequence log-probability roughly an order of
magnitude worse. The gradient identity measured earlier explains it exactly: one energy
hands the optimizer h_n at the target position, the other hands it nothing.

NOTE THE INVERTED STEERING NUMBER, because it is the reason a held-out judge exists.
`cls_thesis` scores HIGHER on the steering classifier (53.3 vs 40.0) while producing
gibberish. A pooled sentiment head over hidden states is trivially satisfiable by
incoherent token soup, so an adherence number reported without a fluency number is
worthless. Every adherence figure in this log is paired with an external fluency figure
for this reason.

### This CONFIRMS the submitted thesis rather than contradicting it

Checked against `Doc/final/thesis/chapters/06_discussion.tex:50`, which states:

> The thesis therefore refutes the premise for a sampler that differentiates the input
> embedding of a likelihood whose target token enters as a discrete index, and not for
> gradient-guided discrete sampling as such. ... MuCoLa \citep{kumar2022gradient}
> substitutes the continuous vector into the output softmax itself ... so its gradient
> reaches the token's own score directly ... Neither discards the self term.

The MuCoLa correction WAS applied before submission. The scoping in the submitted text
is exactly right, and this experiment turns an argument made from the papers' text into
a measurement: implementing MuCoLa's energy in this repository's own code makes the
continuous sampler work, and implementing the repository's energy makes it fail, on the
same task with everything else held fixed.

DEFENSE VALUE. If an examiner asks "did you actually test the method you criticise", the
answer is now: the thesis explicitly says it did not, Section 6.2 limitation 1, and here
is the experiment that shows the distinction was the right one to draw. This is the
strongest single asset produced by this phase.

## A.6 RESULT, framing 2 (matched shortlist), n=200 per cell

Every arm proposes over the SAME frozen top-k candidate set, scored only by its own
surrogate, distance term dropped. Source `results/ctg/studyA/A.k*.json`, index
`results/ctg/studyA_summary.json`.

| k | arm | exact % | final KL | accept % | fwd-equiv/seq | GPU s/seq |
|---|---|---|---|---|---|---|
| 16 | `policy_self` | 30.0 | 4.788 | 30.4 | 100 | 4.22 |
| 16 | `policy_onehot` | 30.0 | 4.948 | 30.4 | 400 | 10.57 |
| 16 | `policy_exact_k` | 30.5 | 4.879 | 80.2 | 117 | 4.41 |
| 16 | `policy` | 30.5 | 4.814 | 17.0 | 400 | 10.42 |
| 16 | `uniform` | 31.5 | 4.580 | 16.6 | 100 | 5.56 |
| 64 | `policy_self` | 29.5 | 4.706 | 25.9 | 100 | 5.44 |
| 64 | `policy_onehot` | 28.0 | 4.822 | 25.7 | 400 | 10.63 |
| 64 | `policy_exact_k` | **31.0** | **4.150** | 73.0 | 165 | **4.68** |
| 64 | `policy` | 21.5 | 4.716 | 12.6 | 400 | 11.66 |
| 64 | `uniform` | 18.5 | 4.875 | 13.4 | 100 | 4.26 |
| 256 | `policy_self` | 26.5 | 4.638 | 23.0 | 100 | 4.24 |
| 256 | `policy_onehot` | 25.0 | 4.660 | 22.7 | 400 | 10.91 |
| 256 | `policy_exact_k` | **31.0** | **4.037** | 66.6 | 357 | 5.58 |
| 256 | `uniform` | 8.0 | 5.195 | 12.4 | 100 | 4.22 |

Paired contrasts against `policy_self`, 10,000-sample bootstrap:

| k | contrast | delta KL | delta exact (pts) | cost |
|---|---|---|---|---|
| 16 | onehot - self | +0.160 [-0.011, +0.360] | +0.0 [-2.0, +2.0] | x4.00 fe |
| 16 | exact_k - self | +0.090 [-0.264, +0.442] | +0.5 [-3.0, +4.0] | x1.17 fe |
| 64 | onehot - self | +0.116 [-0.121, +0.363] | -1.5 [-4.0, +0.5] | x4.00 fe |
| 64 | **exact_k - self** | **-0.555 [-0.977, -0.161]** | +1.5 [-2.0, +5.5] | x1.65 fe, **x0.86 wall** |
| 256 | onehot - self | +0.023 [-0.212, +0.269] | -1.5 [-4.0, +1.0] | x4.00 fe |
| 256 | **exact_k - self** | **-0.601 [-1.008, -0.205]** | **+4.5 [+0.0, +9.0]** | x3.57 fe, x1.32 wall |

### Q1 ANSWERED, and one of the three pre-registrations is REFUTED

**PREDICTION 1 (A1 ~ A2 at lower cost): CONFIRMED, in all four framings.** The
linearized future term never separates from the self term alone. Across full vocabulary,
k=16, k=64 and k=256 the exact-match contrast is +2.0, +0.0, -1.5 and -1.5 points, every
CI containing zero, and the KL contrast is within +-0.16 nats everywhere. It costs 4x
the forward-equivalents and roughly 2 to 2.6x the wall-clock in every case. The backward
pass buys nothing measurable.

**PREDICTION 2 (A3 buys little over A1 relative to its cost): REFUTED at k >= 64, and
this is reported plainly because it was pre-registered the other way.** The EXACT future
term does buy a real improvement once the shortlist is wide enough for ranking to
matter: -0.555 nats [-0.977, -0.161] at k=64 and -0.601 [-1.008, -0.205] at k=256, both
CIs excluding zero, plus +4.5 exact-match points [+0.0, +9.0] at k=256. And it is CHEAP,
which was the genuinely unexpected part: at k=64 it costs 1.65x the forward-equivalents
and **0.86x the wall-clock**, i.e. it is faster in real time than the arm it beats. Two
reasons, both measured. Its acceptance rate is 66 to 80 percent against 23 to 30 percent
for every other arm, so it wastes far fewer steps; and with one masked position the k
candidate energies do not depend on the current token, so they are computed once per
sequence and reused for every step and both directions of every MH decision. That second
economy is task-specific and will not survive to the constrained task, where 20 positions
move; the constrained half of Study A is where the honest ceiling gets measured.

**PREDICTION 3 (A4 near the floor): CONFIRMED at full vocabulary** (0.0 percent, tied
with uniform, paired difference -24.5 points against the self term) but the shortlist
framings qualify it, see below.

### The finding that governs how every other number here must be read

The SUPPORT, not the scoring rule, is the dominant factor, and at a narrow shortlist it
is the only factor.

| arm | full vocab | k=256 | k=64 | k=16 |
|---|---|---|---|---|
| `uniform` (no scoring at all) | 0.0 | 8.0 | 18.5 | **31.5** |
| `policy` (input-embedding gradient) | 0.0 | -- | 21.5 | **30.5** |
| `policy_self` | 24.5 | 26.5 | 29.5 | 30.0 |

At k=16 every arm lands between 30.0 and 31.5 percent, including the arm that scores
nothing and the arm the thesis identifies as carrying no usable direction. Restricting
the proposal to the 16 most likely tokens and letting the exact-energy accept step do the
rest recovers as much as any surrogate does.

This does NOT rehabilitate the input-embedding gradient. The correct control is the
uniform arm, and it matches the gradient arm exactly at k=16 (31.5 vs 30.5), so nothing
about the gradient is demonstrated there; what is demonstrated is that a 16-token
shortlist drawn from the self term plus an exact accept step is a strong sampler on its
own. As k widens and the shortlist stops doing the work, the arms separate in the
expected order and the gradient arm falls away again (21.5 at k=64 against 29.5 for the
self term, -8.0 points [-13.5, -3.0]).

Any future claim of the form "surrogate X works in a shortlisted sampler" has to be
stated against the uniform-on-the-same-shortlist floor, not against zero. That floor was
not measured anywhere in the submitted thesis.

### A.6.1 The k=256 gradient arm completes the support picture

`A.k256.policy` finished after the table above was written and confirms the reading
rather than complicating it: at k=256 the input-embedding gradient falls to 8.5 percent,
statistically indistinguishable from uniform's 8.0 percent (paired against the self term,
-18.0 [-24.0, -12.0] and -18.5 [-24.5, -12.5] respectively, i.e. the same contrast).

The full support series for the gradient arm and its correct control:

| arm | full vocab | k=256 | k=64 | k=16 |
|---|---|---|---|---|
| `uniform` | 0.0 | 8.0 | 18.5 | 31.5 |
| `policy` (input-embedding gradient) | 0.0 | 8.5 | 21.5 | 30.5 |
| gradient minus uniform (points) | 0.0 | +0.5 | +3.0 | -1.0 |

The input-embedding gradient tracks the no-scoring floor at every shortlist width. It
never separates from uniform anywhere in this study. That is a cleaner statement of the
thesis's null than the thesis itself makes, because it holds the support fixed and
therefore rules out the objection that the null was an artifact of proposing over 50,257
candidates.

## A.7 Study A recovery half: COMPLETE. 76 of 76 jobs, 0 failures.

Cost consumed: 5.9 GPU-hours (sum over shards of the per-shard wall-clock);
1.0 hour of wall-clock at 10 concurrent workers.

---

# Study C: how should the constraint enter?

## C.0 PRE-REGISTRATION and a mechanism smoke test

Arms: `accept_only` (C1), `rescore_k` (C2), `grad_prop` (C3), `both` (C4),
`mix_shortlist` (C5). Proposal fixed at the Study A cost winner, `self`.

All five run (smoke, n=15, 12 steps, k=16, `scratchpad/mechsmoke/`). Two things that
smoke test already establishes and that must be stated BEFORE any adherence number:

**1. Shortlist coverage is 0.0 percent for every likelihood-ranked arm.** The candidate
the sentiment classifier most prefers, over the full vocabulary, essentially never
appears in a shortlist ranked by `log p(v | x_<i)`. At k=64 in a longer smoke run the
median likelihood rank of that candidate was **27,464 out of 50,257**, i.e. almost exactly
the middle of the vocabulary: the two rankings are close to unrelated.

CONSEQUENCE, stated in advance so it cannot be presented as a mechanism result later: if
`rescore_k` (C2) turns out weak, that is a PROPOSAL-SUPPORT LIMITATION and not evidence
about exact rescoring as a mechanism. C2 can only rescore what the shortlist contains,
and the shortlist does not contain what the constraint wants.

**2. `mix_shortlist` reports 100.0 percent coverage, and that number is TAUTOLOGICAL.**
C5 builds half its shortlist from the constraint's own first-order ranking, so the
constraint's argmax is in the shortlist by construction. It is reported for completeness
and must never be read as C5 "achieving" coverage. What is informative about C5 is what
that guaranteed coverage buys in adherence and costs in fluency, not the coverage figure.

**3. The constraint-side linearization is weak.** Spearman between the classifier's
first-order surrogate and the true change in the classifier score over the same
candidates: **0.14** (n = 1,440 candidate pairs, smoke scale). For comparison the
likelihood-side one-hot surrogate reaches 0.60 to 0.73 at admissible distances
(REVISION_LOG A5). If this holds at scale, C3's first-order constraint term is working
from a surrogate about four times less faithful than the one the thesis's own diagnostics
call usable, which would be the mechanism behind whatever C3 does.

PREDICTIONS: C1 weakest on adherence, C5 strongest on adherence and worst on fluency,
C2 disappointing FOR COVERAGE REASONS rather than mechanism reasons, C3 weak because its
surrogate is weak, C4 approximately C2.

---

# Queue management notes for this phase (harness behaviour worth recording)

1. **Worker count was tuned by measurement, not guessed.** At 2 workers per card the
   constrained chains left the GPUs at 15 to 43 percent utilization, because each chain
   is a sequence of single-sequence forward passes that badly underuse an A6000. Raising
   to 4 per card put all five at 100 percent, at 19 GB of 48 and 50 GB of 1007 GB host
   RAM. The REVISION_LOG C7 incident (36 concurrent GPT-2 Large loads exhausting host
   RAM) is the reason this was raised in a measured step rather than to the maximum.

2. **The manifest is a PRIORITY list, and reordering it live works.** `worker.sh`
   re-reads the manifest on every outer pass and scans top to bottom, so the Study C
   probe, which gates Study B's configuration, was initially stuck behind twenty Study A
   constrained jobs. Reordering the file in place (Cprobe, Ccov, D, then the Aq
   remainder) moved it to the front without restarting a single worker. Verified the
   line count was unchanged before overwriting, and kept a backup, because a truncated
   manifest would silently drop jobs whose locks already exist.

3. **`tmux has-session` is NOT a completion signal.** A chained launcher that waited for
   the Study A session to disappear would have hung forever: `run_queue.sh` starts each
   worker inside an interactive shell, so the window and the session survive the
   worker's exit. Completion must be detected from the result files or the locks. The
   chained launcher was removed and phase 2 was started directly on the freed cards.

4. **Progress counts must exclude merged files.** `results/ctg/phase2/*.json` matches
   both the per-shard results and the merged run-level file the analysis writes, so a
   naive count over-reports. Shard progress is counted with `*.shard*of*.json`.

---

# FINAL TABLE: study, prediction, outcome, and what to say

Status as of the last update to this file. Rows marked PENDING have jobs in the queue;
rows marked NOT RUN were not reached this session and are listed rather than omitted.

| # | Question | Pre-registered prediction | Outcome | One sentence for the defense |
|---|---|---|---|---|
| A1 | Q1 COST, recovery. Does the self term alone match the full token-indicator surrogate? | A1 approximately equals A2 at lower cost | **CONFIRMED in all four framings.** Delta exact +2.0 / +0.0 / -1.5 / -1.5 points (full vocab, k=16/64/256), every CI containing zero; KL within +-0.16 nats. Cost x4 forward-equivalents, x2 to x2.6 wall | "The constructive half of my result needs no gradient at all: the model's own next-token distribution, which the forward pass already computes, gets the whole effect, and the backward pass adds nothing measurable at four times the cost." |
| A2 | Q1 COST. Is the EXACT future term worth k forward passes? | A3 buys little relative to its cost | **REFUTED at k >= 64.** -0.555 nats [-0.977, -0.161] at k=64 and -0.601 [-1.008, -0.205] at k=256, CIs excluding zero, +4.5 exact points [+0.0, +9.0] at k=256; and at k=64 it runs at **0.86x the wall-clock** because acceptance is 73 percent against 26 | "I predicted the exact future term would not pay and I was wrong: once the shortlist is wide enough for ranking to matter it buys about 0.6 nats, and it is faster in wall-clock because it wastes far fewer rejected steps." |
| A3 | Does the input-embedding gradient beat the no-scoring floor at ANY shortlist width? | Near the floor | **CONFIRMED, and sharpened.** Gradient minus uniform: 0.0 / +0.5 / +3.0 / -1.0 points at full vocab / k=256 / k=64 / k=16. It tracks the uniform floor everywhere | "Holding the candidate support fixed rules out the objection that my null came from proposing over fifty thousand tokens: at every shortlist width the gradient arm is indistinguishable from proposing at random within the same shortlist." |
| A4 | Does the published 40 percent hold at larger n? | not pre-registered; found while matching comparator sample sizes | **CONTRADICTS SUBMITTED CLAIM.** 40.0 percent reproduces exactly at n=50; at n=200 the same cell gives 26.5 percent [20.4, 32.6], below the thesis's stated "36 to 40 percent" range, and below SEDD 39.0 and RoBERTa 44.5, which were n=200 | "The 40 percent is an n=50 sweep cell and it reproduces exactly; at n=200, matching the diffusion baselines, it settles at 26.5 percent, so the honest claim is zero to roughly a quarter rather than parity with a purpose-trained diffusion model." |
| M | Is the thesis's continuous sampler MuCoLa? | it is not; the difference should be visible in the output | **CONFIRMS the submitted scoping, and demonstrates it.** Identical code, only the energy differs: MuCoLa's produces fluent English at LM log-prob -38.5, the thesis's produces word salad at -353.9. Final-position gradient h_{n-1} (cosine 1.000000) versus exactly 0.000e+00 | "My thesis says explicitly that it tests its own construction and not MuCoLa's; this experiment shows the distinction was worth drawing, because implementing MuCoLa's energy in the same code makes the sampler work." |
| J | Is a steering classifier's own score a valid adherence measure? | not pre-registered | **NO, and it inverts.** The blind energy scores HIGHER on the steering head (53.3 vs 40.0) while producing gibberish | "An adherence number from the classifier that did the steering is not evidence; a pooled sentiment head is trivially satisfied by incoherent tokens, which is why every adherence figure I report is paired with an external fluency figure." |
| Aq | Q1 COST, constrained-generation task | same as A1 | **PENDING.** `Aq.self` and `Aq.onehot` merged: acceptance identical at 10.3 percent, cost 635 versus 4,133 forward-equivalents per sequence (x6.5). Judge scoring pending; `Aq.exact_k`, `Aq.embgrad`, `Aq.uniform` in flight | -- |
| C | Q3 MECHANISM. How should the constraint enter? | C1 weakest, C5 strongest on adherence and worst on fluency, C2 disappointing FOR COVERAGE REASONS, C3 weak because its surrogate is weak | **PENDING.** Pre-registered in C.0. Smoke-scale coverage already 0.0 percent for every likelihood-ranked arm, median likelihood rank of the constraint's preferred token 27,464 of 50,257; constraint-side linearization Spearman 0.14 | "The shortlist and the constraint want almost unrelated tokens, so before comparing mechanisms I report coverage, and a weak exact-rescoring arm is a support limitation rather than a verdict on the mechanism." |
| D | Q4 USEFULNESS at matched compute | not pre-registered as a direction | **PENDING.** Full constraint-weight sweep queued for all three methods (`D.ours`, `D.mucola_faithful`, `D.cls_thesis`) at six weights each | -- |
| B | Q2 CONDITIONING. Do bidirectional proposals win when the attribute is global? | not yet fixed; depends on the C probe | **NOT LAUNCHED.** Gated on the Study C probe, which is in flight. The harness supports `roberta` and `sedd` proposals and both were validated in the recovery setting by REVISION_LOG C8 | -- |
| E | The sampling claim: diversity at matched adherence | not pre-registered | **NOT RUN this session.** The diversity metrics are implemented and validated (`Aq.self`: distinct-1 0.929, distinct-3 1.00, self-BLEU 0.000, semantic spread 0.778) but the matched-adherence, matched-compute comparison against a best-of-N optimizer was not reached | "The machinery is in place and the numbers say the chain is highly diverse, but I have not yet run the controlled comparison against a compute-matched best-of-N optimizer, so I would not claim posterior coverage on this evidence." |

## Traceability: every number above comes from one of these files

- `results/ctg/studyA_summary.json` (flat index for Study A, both framings, all costs)
- `results/ctg/studyA/A.fullvocab.*.json` and `.csv`, `A.k{16,64,256}.*.json` and `.csv`
- `results/grid/rev3/ohsweep_e10p5_t1p0.json` (the thesis's n=50 cell, for the contrast)
- `results/revision/rev_mlm_control.json` (RoBERTa 44.5, uniform 0.5, n=200 comparators)
- `results/revision/rev_sedd_recovery_{small,medium}.json` (SEDD comparators)
- `results/ctg/phase2_summary.json` (constrained studies index, updated as families land)
- `results/ctg/phase2/*.json` and `*.csv` (per-run results and the generated text)
- `results/ctg/judged/*.sentiment.json`, `*.fluency.json` (held-out judge scores)
- scratchpad verification: `verify_mucola.py` output, quoted in full in this log

---

# Incident: OOM in the nearest-embedding projection, and the fix

Six `D.mucola_faithful.*` jobs failed with CUDA OOM, all at the same line:
`core/prep.py:41`, inside `project_to_vocab_by_l2`.

CAUSE, and it is a real defect rather than an over-subscribed card. The shared helper
forms the difference tensor explicitly:

```python
diff = s_flat[:, None, :] - emb_matrix[None, :, :]      # (M, V, D)
```

At ONE masked position that is 1 x 50,257 x 1,280 x 4 bytes = 0.26 GB, which is why the
entire 145-configuration recovery grid and all of Study A never touched it. The
continuation task moves TWENTY positions at once, so the same line asks for
20 x 50,257 x 1,280 x 4 = **4.79 GB in a single allocation**, every step, on a card
already shared by four other workers.

FIX. A local `project_l2_lowmem` in `diagnostics/run_mucola_faithful.py` computes the
same argmin through the standard expansion ||s-e||^2 = ||s||^2 + ||e||^2 - 2 s.e,
dropping the ||s||^2 term because it is constant across candidates and cannot change an
argmin. That is the same algebraic form `core/dls.py` already uses for its distance term.

`core/prep.py` IS DELIBERATELY NOT CHANGED. It is on the archived CLS path, and this
phase's standing rule is that no edit may perturb a result the thesis already reports.
The efficient version lives in the new script only.

EQUIVALENCE CHECK, run before the fix was used for anything (on CPU, because verifying
it on a GPU would have needed the 4.79 GB the original allocates, which is the whole
problem): 120 projections over three regimes the optimizer actually visits, namely
perturbed centroid, exact token embeddings, and random points far off the token manifold.

    PROJECTION EQUIVALENCE: 0 mismatches out of 120 projections

Only the six failed jobs' locks and `.failed` markers were cleared, never
`reset_incomplete.sh`, which would have unlocked jobs still in flight. Zero results were
lost: all six crashed before any JSON write, and the JSON write is atomic.

WORTH NOTING FOR THE HARNESS SECTION. This is the second defect this phase that was
invisible at one masked position and appeared immediately at twenty. The first was the
exact-future arm's per-sequence energy cache, which is valid at M=1 and invalid at M>1
and is guarded by an explicit `raise`. A single-mask diagnostic task hides a whole class
of scaling bug.

## C.1 SHORTLIST COVERAGE, reported before any adherence number. COMPLETE.

Source `results/ctg/phase2/Ccov.k{16,64,256}.json`, n=30 sequences each, proposal
`self`, constraint in the accept step only, coverage sampled every 5 steps.

The diagnostic: at each step, find the candidate the STEERING classifier most prefers
over the whole 50,257-token vocabulary (by its first-order term), then ask where that
candidate falls in the likelihood ranking, and whether it is inside the shortlist.

| k | coverage (measured) | coverage if the two rankings were independent | ratio to chance | median likelihood rank of the constraint's top candidate | constraint-side linearization Spearman |
|---|---|---|---|---|---|
| 16 | **0.2%** | 0.032% | ~6x | 29,928 of 50,257 | +0.019 |
| 64 | **0.8%** | 0.127% | ~6x | 34,227 of 50,257 | +0.040 |
| 256 | **1.7%** | 0.509% | ~3x | 29,866 of 50,257 | -0.008 |

TWO FINDINGS, and both are mechanisms rather than nulls.

**1. A likelihood-ranked shortlist almost never contains what the constraint wants.**
The median likelihood rank of the constraint's preferred token is around 30,000 out of
50,257, which is the middle of the vocabulary: the two rankings are close to independent.
Coverage is three to six times chance, so the association is positive and real, but it
is still under 2 percent even at k=256. Widening the shortlist sixteen-fold, from 16 to
256, moves coverage from 0.2 to 1.7 percent and would need to be widened by orders of
magnitude more to matter.

CONSEQUENCE, exactly as pre-registered in C.0: any weakness of the exact-rescoring arm
(C2) is a PROPOSAL-SUPPORT limitation. C2 can only rescore candidates the shortlist
contains, and 98 to 99.8 percent of the time the constraint's preferred candidate is not
among them. This must be stated before, not after, C2's adherence number.

**2. The constraint-side linearization carries no usable signal.** Spearman between the
classifier's first-order surrogate and the true change in the classifier's score over the
same candidates is +0.019, +0.040 and -0.008 at the three widths, i.e. indistinguishable
from zero, on 1,000 to 2,000 candidate pairs per cell.

Set against this study's own likelihood-side numbers, the contrast is the point:

| surrogate | Spearman against the true change |
|---|---|
| likelihood, token-indicator (one-hot), admissible distances | +0.60 to +0.73 (REVISION_LOG A5) |
| likelihood, input-embedding gradient | ~0.0 (REVISION_LOG A5) |
| **constraint, first-order classifier gradient** | **+0.02 to +0.06 (this work)** |

So the classifier gradient is in the SAME condition as the input-embedding gradient the
thesis's null is about: its first-order expansion does not predict what actually happens
to the quantity it is expanding. That gives C3 a mechanism in advance of its result.

THIS QUALIFIES A CLAIM IN THE SUBMITTED THESIS, though it does not contradict it.
`CLAUDE.md` records the concern-11 reading as "the constraint gradient's direction
carries signal, the LM gradient's does not", resting on the paired
`cons_only - cons_random` contrast of roughly +27 to +37 points on the mucola-continuation
setup. That contrast is not re-run here and is not disputed. What is added is that
whatever signal the constraint direction carries, it is NOT carried by the accuracy of
its first-order expansion, which is near zero here. A plausible reconciliation, offered
as a hypothesis and not as a result: a classifier gradient can point usefully in the
AGGREGATE over many small continuous steps, which is what the continuous sampler takes,
while being a poor predictor of the effect of any single DISCRETE token substitution,
which is what this measurement scores. The two setups ask different questions of the same
gradient and the answers need not agree.

## Aq.1 Study A on the CONSTRAINED task: partial result (3 of 5 arms)

`Aq.exact_k` and `Aq.uniform` are still running. Source
`results/ctg/phase2/Aq.*.json`, `.csv`, judge scores in `results/ctg/judged/`.
n=60 sequences (15 prompts x 4), span 20 tokens, 600 steps, k=64, constraint in the
accept step only, `cons_gain` 1.0.

| arm | steer-head acc | held-out adherence | accept % | fwd-equiv/seq | GPU s |
|---|---|---|---|---|---|
| `self` | 36.7% | 46.7% +- 12.6 | 10.3 | **635** | **2,982** |
| `onehot` | 41.7% | 63.3% +- 12.2 | 10.3 | 4,133 | 8,047 |
| `embgrad` | 38.3% | 56.7% +- 12.5 | 12.9 | 4,234 | 10,712 |

PAIRED on `global_idx` (same prompt, same seed, same starting span across arms),
10,000-sample bootstrap:

| contrast | delta steer-head log p(target) | delta LM log-prob |
|---|---|---|
| `onehot` - `self` | +0.138 [-0.186, +0.449] | -2.20 [-10.57, +6.22] |
| `embgrad` - `self` | +0.034 [-0.255, +0.320] | **+12.52 [+5.46, +19.48]** |

READING, and the caveats come first.

The three held-out adherence figures have 95 percent intervals of roughly +-12.5 points
on n=60 and overlap heavily, so the apparent 46.7 / 63.3 / 56.7 ordering is NOT a
separation and must not be quoted as one. The paired steering-head contrast, which is
the better-powered test because it pairs on the sequence and uses a continuous score,
puts both differences squarely on zero.

CONSISTENT WITH THE RECOVERY RESULT: the self term alone matches the full token-indicator
surrogate on adherence, at **6.5 times less compute** in forward-equivalents and 2.7
times less wall-clock. Q1's answer does not change when the task changes from repairing
one token to conjuring twenty.

ONE RESULT THAT GOES THE OTHER WAY AND IS REPORTED AS SUCH. `embgrad`, the
input-embedding gradient that scores at the floor on recovery, attains a HIGHER final LM
log-probability than the self term here, +12.52 nats with a CI excluding zero. This is
not evidence that the gradient carries directional signal: every arm uses the identical
exact accept step and therefore targets the identical distribution, so a difference in
final energy is a difference in MIXING, and `embgrad`'s acceptance rate is indeed higher
(12.9 against 10.3 percent). It is nonetheless a measured advantage on one metric and it
is recorded rather than filtered out. The `uniform` arm, still running, is the control
that will say whether this is about the gradient at all or about the shortlist again;
on the recovery task the same comparison collapsed once uniform was in the table.

---

# Study D: the head-to-head. COMPLETE (unnormalized parameterization).

n=60 sequences (15 prompts x 4 independent seeds), span 20. Constraint weight swept over
six values for every method, so the comparison is a CURVE. Adherence and fluency are both
from HELD-OUT models: adherence from `siebert/sentiment-roberta-large-english`, fluency
(perplexity of the generated span) from Llama-3-8B. Neither ever touched the generation.
Source `results/ctg/phase2/D.*.json`, judges in `results/ctg/judged/`, index
`results/ctg/phase2_summary.json`.

| method | gain | held-out adherence % | held-out ppl | distinct-3 | self-BLEU-4 | fwd-eq/seq |
|---|---|---|---|---|---|---|
| `mucola_faithful` | 0.0 | 46.7 | **25.3** | 0.25 | **1.000** | 900 |
| `mucola_faithful` | 0.5 | 80.0 | **22.5** | 0.25 | **1.000** | 900 |
| `mucola_faithful` | 1.0 | 60.0 | **24.0** | 0.30 | 0.956 | 900 |
| `mucola_faithful` | 2.0 | 73.3 | **23.1** | 0.25 | **1.000** | 900 |
| `mucola_faithful` | 4.0 | 60.0 | **23.3** | 0.25 | **1.000** | 900 |
| `mucola_faithful` | 8.0 | 60.0 | **22.5** | 0.25 | **1.000** | 900 |
| `cls_thesis` | 0.0 | 86.7 | 28,814 | 0.25 | 1.000 | 1,200 |
| `cls_thesis` | 1.0 | **93.3** | 40,783 | 0.25 | 1.000 | 1,200 |
| `cls_thesis` | 8.0 | **93.3** | 29,436 | 0.25 | 1.000 | 1,200 |
| `ours` (MH chain) | 0.0 | 46.7 | 3,287 | **1.00** | **0.000** | **598** |
| `ours` | 0.5 | 48.3 | 3,144 | **1.00** | **0.000** | **598** |
| `ours` | 1.0 | 46.7 | 3,174 | **1.00** | **0.000** | **599** |
| `ours` | 2.0 | 55.0 | 3,495 | **1.00** | **0.000** | **600** |
| `ours` | 4.0 | 55.0 | 3,129 | **1.00** | **0.000** | **598** |
| `ours` | 8.0 | 60.0 | 3,416 | **1.00** | **0.000** | **597** |

## D.1 THE HEADLINE: the optimizer returns ONE solution per prompt

`self_bleu_4 = 1.000` and `distinct_3 = 0.25` are not near-degenerate, they are exactly
degenerate: with four samples per prompt, 0.25 is precisely what four IDENTICAL strings
produce. Inspected directly, same prompt, four different seeds:

    mucola_faithful, seed 0: "Once upon a time, there was a king who ruled over a large land. His subjects were free to do as"
    mucola_faithful, seed 1: "Once upon a time, there was a king who ruled over a large land. His subjects were free to do as"
    mucola_faithful, seed 2: "Once upon a time, there was a king who ruled over a large land. His subjects were free to do as"
    mucola_faithful, seed 3: "Once upon a time, there was a king who ruled over a large land. His subjects were free to do as"

    ours, seed 0: "Once upon a time, girls horrendous bully boyNicole was Yunho Olympics switching on every. singl"
    ours, seed 1: "Once upon a time everything camtoged together Petition.com anita creeps on facebook and SYSTEM"
    ours, seed 2: "Once upon a time everyone worked#organized around a assassination- plans timidly Gamma members"
    ours, seed 3: "Once upon a time, major corporations had Chest Intentions. Workers all set out on a big day to"

The continuous optimizer is DETERMINISTIC given the prompt: gradient descent from the
same centroid initialization reaches the same point whatever the seed. It is a
mode-seeking optimizer, not a sampler, and it returns exactly one solution per prompt.
The MH chain returns four distinct ones.

This is the "posterior coverage" property the thesis's proposal motivated and never
operationalized, and it is now measured. It falls in the sampler's favour, decisively and
without qualification, on the diversity axis.

## D.2 AND THE TRADE-OFF, which falls the other way and is reported just as plainly

At every constraint weight, `mucola_faithful` is roughly **135 times more fluent** under
the held-out judge (perplexity 22.5 to 25.3 against 3,129 to 3,495). The sample texts
above make the difference obvious without any metric: MuCoLa's output is a well-formed
English sentence; the chain's is mostly not.

So the honest answer to "is your sampler useful" is a genuine trade-off, not a win:

- the MH chain produces MANY distinct solutions at LOW fluency;
- MuCoLa's optimizer produces ONE fluent solution;
- and the chain is the cheaper of the two per sequence (598 against 900
  forward-equivalents), so this is not a compute artifact.

Why the chain is less fluent is not mysterious and should be stated rather than excused:
600 single-position Gibbs steps over a 20-token span, starting from uniformly random
tokens, is roughly 30 sweeps, and that is simply not enough mixing to reach a fluent
region of a 20-token space from a random start. The optimizer does not have to mix; it
descends. A fair reading is that the two methods are being asked different questions, and
the compute-matched comparison at 600 steps answers the one about diversity, not the one
about reachable fluency.

## D.3 The constraint actually works in the chain, at no fluency cost

Adherence against constraint weight for `ours`: 46.7, 48.3, 46.7, 55.0, 55.0, 60.0 at
gains 0, 0.5, 1, 2, 4, 8, with the steering head rising monotonically 33.3, 38.3, 36.7,
41.7, 45.0, 56.7, while held-out perplexity stays flat between 3,129 and 3,495 with no
trend. So the constraint term does steer the chain, and it does so without degrading
fluency, which is the shape a usable trade-off curve should have. The effect is modest
(about +13 points of adherence over the full 16-fold weight range) and n=60 gives roughly
+-12.5 points per point, so the trend is suggestive rather than individually significant.

## D.4 The gibberish-satisfies-the-classifier trap, now confirmed under the HELD-OUT judge

`cls_thesis` attains the HIGHEST adherence of any method in the study, 93.3 percent, at a
held-out perplexity of 40,783. Its text is word salad. This was predicted from the
steering head earlier in this log; the important part is that it reproduces under a
sentiment classifier that had no connection to the generation whatsoever.

An adherence number reported without a fluency number is not weak evidence, it is
anti-evidence: in this study the method with the best adherence is the one whose output
is unreadable. Any CTG result quoting attribute accuracy alone should be read with that in
mind.

## D.5 Study A's cost answer holds on the constrained task too

Completing Aq (all five arms, n=60, k=64, constraint in the accept step only):

| arm | held-out adherence % | held-out ppl | accept % | fwd-eq/seq |
|---|---|---|---|---|
| `self` | 46.7 | 3,174 | 10.3 | **635** |
| `onehot` | 63.3 | 3,924 | 10.3 | 4,133 |
| `embgrad` | 56.7 | 2,227 | 12.9 | 4,234 |
| `uniform` | 60.0 | 1,486 | 13.0 | **663** |
| `exact_k` | 58.3 | **715** | **22.4** | 56,521 |

Two readings, both with the n=60 caveat (adherence intervals are about +-12.5 points and
every pair overlaps).

The linearized future term again buys nothing it is worth paying for: `onehot` costs 6.5x
`self` and is no better on fluency (3,924 against 3,174) or reliably on adherence.

The EXACT future term is again the one that pays on quality, and again it is expensive:
`exact_k` reaches a held-out perplexity of 715, four to five times better than every other
chain arm, and more than doubles the acceptance rate (22.4 against 10.3 percent), at 89
times the forward-equivalent cost. On the recovery task that cost was hidden by a
single-mask caching economy; with 20 moving positions the true price is visible, exactly
as flagged in advance in A.2.

And once more the support does much of the work: `uniform` on the same 64-token shortlist
reaches perplexity 1,486 and adherence 60.0, beating `self`, `onehot` and `embgrad` on
fluency at the lowest cost in the table.

## C.2 Study C mechanism probe: RESULT, and the pre-registration scored honestly

n=30 sequences, 200 steps, k=64, proposal fixed at `self`. Source
`results/ctg/phase2/Cprobe.*.json`. **The adherence intervals here are roughly +-18
points at n=30, so every pairwise difference below is within noise and the ordering is
indicative only.** The cost column and the coverage column are the parts that carry
weight, because they are not noisy.

| arm | coverage % | held-out adherence % | held-out ppl | fwd-eq/seq | cost vs C1 |
|---|---|---|---|---|---|
| C1 `accept_only` | 0.8 | 46.7 | 6,429 | **602** | 1x |
| C2 `rescore_k` | 1.8 | 43.3 | 6,997 | 25,894 | **43x** |
| C3 `grad_prop` | 0.8 | 43.3 | **5,368** | 1,687 | 2.8x |
| C4 `both` | 1.5 | 40.0 | 7,127 | 26,974 | **45x** |
| C5 `mix_shortlist` | 100.0 (tautological) | **60.0** | 6,628 | 1,674 | 2.8x |

PRE-REGISTRATION SCORED:

| prediction | outcome |
|---|---|
| C1 weakest on adherence | **WRONG.** C1 is mid-pack at 46.7; C4 is lowest at 40.0 |
| C5 strongest on adherence | **RIGHT.** 60.0, the highest in the study |
| C5 worst on fluency | **WRONG.** 6,628 is mid-pack; C4 is worst at 7,127 |
| C2 disappointing FOR COVERAGE REASONS | **RIGHT, and this is the important one** |
| C3 weak because its surrogate is weak | **RIGHT** |

THE ANSWER TO Q3, with the coverage caveat carried through as promised.

**Exact classifier rescoring on a likelihood-ranked shortlist is not worth its cost, and
the reason is support, not mechanism.** C2 costs 43 times C1 and returns 43.3 percent
adherence against C1's 46.7. That is exactly what C.1 predicted in advance: the
constraint's preferred candidate is inside the shortlist 1.8 percent of the time, so 98
percent of the time C2 is rescoring a set that contains nothing the constraint wants.
This must not be reported as "exact rescoring does not work"; it is "exact rescoring had
almost nothing to rescore". The clean way to test the mechanism itself would be to widen
the shortlist by orders of magnitude, which is precisely what makes it unaffordable.

**Fixing the support is what helps, and it is cheap.** C5, which builds half its
shortlist from the constraint's own ranking, is the only arm above C1 on adherence, at
2.8 times the cost rather than 43 times. Its 100 percent coverage is tautological and is
never claimed as an achievement; what is informative is that guaranteeing the constraint's
candidates are REACHABLE beats scoring a set they are absent from, by a wide margin in
cost and a modest one in adherence.

**Putting the classifier gradient in the proposal buys nothing**, as its near-zero
linearization correlation predicted: C3 gives 43.3 against C1's 46.7 at 2.8 times the
cost. It does return the best fluency of the five (5,368), which is recorded but not
claimed, since it could easily be n=30 noise.

RECOMMENDED CONFIGURATION, and it is a genuinely useful and slightly surprising answer:
**put the constraint in the accept step and spend the budget on the SHORTLIST rather than
on scoring.** C1 plus a constraint-aware shortlist (C5) dominates the two exact-rescoring
arms on cost by a factor of fifteen and is no worse on any measured axis.
