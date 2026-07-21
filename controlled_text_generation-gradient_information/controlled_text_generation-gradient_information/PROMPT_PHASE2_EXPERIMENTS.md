# Prompt for Claude Code, Phase 2: the last-token experiment, a working setup, and a rigorous audit

Set the model to Opus 4.8 with the highest reasoning setting before you start
(`/model`, choose Opus 4.8, enable high or extended thinking). Then paste the block
below.

---

Read CLAUDE.md, REVISION_LOG.md (the full final report at the bottom), and
thesis_revision_plan.md before doing anything. The 19-concern revision is
essentially complete; do not re-run any of it. This phase adds two new experiments
and a mathematical audit that together answer the supervisor's open question: WHY
does gradient guidance not work, stated as a proof rather than a statistic. Keep
logging every audit result, derivation, decision, launch, and number in
REVISION_LOG.md with timestamps, appended at the bottom. No em-dashes in anything
you write.

## Part 1: the mathematical audit (do this first, no GPU needed beyond a probe)

Verify the following claims against the actual code in core/ and write the
verification into REVISION_LOG.md. Be rigorous: derive, then confirm numerically.

1. THE ZERO-GRADIENT THEOREM AT THE LAST POSITION. Claim: under causal attention
   with the inputs_embeds parameterization, the gradient of the full-sequence
   log-likelihood with respect to the input embedding of the final scored token is
   exactly zero, because that embedding influences only the logits at its own
   position, which predict a token that does not exist. Derive this in two
   sentences, then verify numerically: write a small probe (CPU or one GPU, GPT-2
   large, 20 sequences) that computes per-position gradient norms of the sequence
   log-likelihood with respect to inputs_embeds and prints the norm at the final
   position, second-to-last, and a middle position. Expected: exactly 0.0 at the
   final position (up to float noise), small at second-to-last (exactly one
   downstream term), larger in the middle. Check the harness's sequence ending
   first: if scoring includes a trailing token after the last word (period or EOS),
   the truly-final position is that trailing token; identify which position has
   zero downstream terms and report it.

2. THE STRUCTURAL BLINDNESS EXPLANATION. Confirm that GPT-2 large ties input and
   output embeddings (wte and lm_head share weights). Then state the mechanism
   precisely: the term log p(x_L | x_<L) is computed as a softmax over h_{L-1} dot
   the OUTPUT embedding rows, so it depends on which token x_L is only through the
   output embedding path, and backprop to inputs_embeds[L] never sees it. The
   information the sampler needs exists inside the model (h_{L-1} scores every
   candidate exactly) but is invisible to the input-embedding gradient. This is the
   one-sentence answer to why gradient guidance fails, and the last position is
   where it becomes a theorem instead of an empirical claim.

3. THE ENERGY IS EXACT AT THE LAST POSITION. Derive: for a masked final token, the
   energy difference between candidates x and x' is exactly
   log p(x' | prefix) - log p(x | prefix), the model's own conditional. Therefore
   an independence Metropolis sampler that proposes from p(. | prefix) has
   acceptance probability exactly 1 for every move (target proportional to
   proposal). This gives a built-in unit test: if the measured acceptance in Part 2
   deviates from ~100%, the implemented energy contains terms beyond the sequence
   log-likelihood, and you must find them before trusting anything else.

4. RE-VERIFY THE MH RATIOS in core/dls.py and core/cls.py as they exist in the
   repo NOW (the code was fixed once before; confirm the fix is present): forward
   and reverse proposals evaluated at the exact sampled states, method-consistent
   gradients in the reverse proposal, symmetric kernels cancelling for random
   walks.

5. CONFIRM THE KNOWN BITWISE ARTIFACT. In the concern 9 and concern 3 results,
   grad_norm_preserved and random are numerically identical. Confirm this is the
   documented seeded behavior under gn=on (identical unit-norm directions consume
   the same RNG stream), not a new bug, and note that the thesis should present
   policy vs random as the primary comparison wherever gn=on.

6. DOCUMENT THE KL METRIC BOUNDARY. The existing avg_kl metric evaluates masked
   positions m with m < seq_len - 1, so a masked FINAL position contributes zero
   terms and the metric is undefined there. This is why Part 2 defines its own
   metrics. Write this down so nobody wonders why last-token rows have no KL.

## Part 2: the last-token experiment (the professor's suggestion, run properly)

Implement a new experiment (suggested: --exp last_token in
diagnostics/run_revision.py, following the existing conventions: run_name,
out_dir, atomic JSON, per-item CSV, deterministic corruption per sample_idx).

Design:
- Same ROCStories sequences and loading conventions as kl_baselines, n=200.
- Three masked-position conditions on the SAME sequences: final scored token
  (zero downstream terms), second-to-last (exactly one), and a middle position
  (many). The position condition is the experimental variable.
- Arms per condition:
  a. DLS policy, MH on, gn on, standard schedule (10.5 to 0.1, 50 steps)
  b. DLS random, same settings (skip gradnorm: bitwise identical to random here)
  c. cond_argmax, cond_sample, cond_topk_rescore (reuse the kl_baselines code)
  d. independence MH from the conditional p(. | prefix): propose from the
     conditional, accept by exact energy ratio. At the final position, assert
     acceptance ~= 100% per Part 1 item 3 and record the measured rate.
- Metrics (the KL metric does not apply at the final position, define these):
  exact-match %, top-5 recovery %, mean NLL of the recovered token under
  p(. | prefix), mean rank of the recovered token. Report all four per arm per
  position condition, with bootstrap CIs over sequences.
- Also record the mean per-step gradient norm seen by DLS at each position
  condition, tying the audit to the experiment.

State the predictions in REVISION_LOG.md BEFORE running (this is what makes it a
test): at the final position, policy == random within noise because the gradient
is exactly zero, while the independence sampler and cond_topk_rescore recover at
the ceiling set by the model's conditional; at the middle position, the known null
reproduces; second-to-last sits between, and the interesting question is whether
one downstream term already provides measurable signal or not. Then run (cheap:
GPT-2 large only, one GPU, well under an hour) and compare outcome to prediction.

Deliverable: one table (arms x position conditions x four metrics) and one figure
spec (gradient norm and policy-minus-random gap as a function of downstream
context length: 0, 1, many). This is the closing figure of the thesis.

## Part 3: the working-setup comparison (confirm before building anything beyond A)

The user wants a setup where energy-guided generation actually works, to contrast
with the failing one and explain the difference. Two candidates, in cost order:

A. PRE-APPROVED, build it: the energy-only last-token sampler from Part 2 (the
   independence MH arm) IS the working setup. It succeeds by construction on the
   exact task where the gradient provably fails, using the same frozen model, the
   same energy definition, the same harness. The comparison writes itself: the
   energy was never the problem at this position, the input-embedding gradient
   was. Pair it with the existing concern-2 result (cond_topk_rescore 4.43 KL
   beats every Langevin config; Gibbs 6.69 works gradient-free) and the thesis has
   working-vs-failing on both the last-token task and the general infill task.

B. NEEDS EXPLICIT GO-AHEAD, do not start without it: the real SEDD run
   (positive control). Scope if approved: clone
   github.com/louaaron/Score-Entropy-Discrete-Diffusion, download sedd-small
   (GPT-2 tokenizer), verify the two ADAPT hooks in run_sedd_linearization.py
   against the actual repo interfaces, run the dry-run again, then the real
   linearization (n=200). Optionally SEDD's NATIVE infilling on the same corrupted
   sequences, reported as a positive control, not as a harness plug-in. Timebox:
   two days hard, then write it as designed-but-not-run per the plan's fallback.
   Present the verified interface plan and time estimate to the user and WAIT for
   confirmation before running the real thing.

Do not pursue anything heavier (Diffusion-LM integration, score-matching
fine-tuning, any pretraining). The user explicitly ruled these out.

## Part 4: report

Append to REVISION_LOG.md: the audit results with derivations, the prediction vs
outcome table for the last-token experiment, the measured independence-sampler
acceptance rate (and what it certifies about the energy implementation), the
working-vs-failing comparison paragraph in plain language ready to adapt for the
thesis, and the SEDD status. Flag anything where the outcome contradicted the
prediction; that would be the most important finding in the log.


Another session is editing Doc/ in parallel. Do not touch Doc/ or REVISION_LOG_THESIS.md; your log stays REVISION_LOG.md.