# Thesis Q&A Knowledge Base
### Energy-Guided Sampling for Controllable Text Generation — every question, follow-up, and confusion point

## About this document

Collates every question, follow-up, pushback, and confusion you raised across all sources, so it can feed Claude Code later for refining the thesis or building slides. This version incorporates the **full Gemini transcript**, transcribed verbatim by OCR from `gemini_part1.pdf` (50 pages, 50 prompts) and `gemini_part2.pdf` (16 pages, 16 prompts), which contained many follow-ups absent from `questions.txt`.

**Sources**
- `(claude)` this Claude chat: 5.6 through 5.11 deep dives, the oracle, soft-token relaxation, MH breakdown, the knowledge-base build.
- `(gemini)` the earlier Gemini chat, now transcribed in full. Prompt indices below like `G1-07` or `G2-12` map to part/prompt in that transcript.
- `(notes)` your `results_doubts_and_questions.txt` reading-doubts list.
- `(txt)` the `questions.txt` summary of the Gemini chat (kept only where it adds an annotation the raw prompts don't).

**Status legend:** `[ANS-C]` answered in this Claude chat, `[ANS-G]` answered in the Gemini chat, `[OPEN]` captured, not yet answered anywhere.

**Confusion-type tags:** `DEF` definitional gap, `MECH` mechanism/math unclear, `CONTRA` apparent contradiction, `PUSHBACK` you challenged or corrected the explanation, `WORDING` sentence read ambiguously, `MISSING` info absent from thesis/source, `SYNTH` you were synthesising threads, `IDEA` you proposed your own method/alternative.

---

## 1. Confusion hotspots (read first)

The topics you circled back to most, i.e. the parts most likely to lose a reader or examiner.

1. **The self-term gradient blindness (5.6, last paragraph).** Deepest hotspot. Raised in Gemini (the linearization thread) and re-opened four times in this Claude chat: the two limitations + why the input-embedding gradient sees only future tokens; why the token enters as an integer index; where the soft-token equations come from; what pi_i is and why projecting it back revives the MH breakdown.

2. **"The gradient is noise, yet something works."** Recurring `CONTRA` across 2.4, 4.5, 5.1, 5.2, 5.5: if the direction is uninformative, how does CLS ever recover a real token (`G1` "as"/"dorm"), how does MH-on beat MH-off (6.541 vs 9.499), and how did MuCoLa/COLD publish coherent text.

3. **The MH correction and the "discontinuity trap" (2.3, 2.4).** A long Gemini background thread: the acceptance formula with a good vs bad transition, the biased-optimizer point, and especially the **slope-versus-zero-gradient paradox** (`G1-09`): how can the differentiable pathway have a slope if the projected energy has zero gradient.

4. **Your own proposed method (recurring `IDEA`).** Across `G1-24` to `G1-31` you pushed a genuine alternative: use the GFlowNet `[Start][Ending][Generation]` prompt framing and steer the masked-token generation with the gradients of the context tokens under Langevin. This is your idea, revisited several times, and worth treating as a named "considered alternative" in the thesis/defense.

5. **Smaller step size as a fix for CLS (`G2-03,05,06`) and the step cap (`claude`).** Same intuition attacked from two chats: won't a smaller step stop the overshoot / keep the gradient valid. Both resolved to no, for the same overlap-of-scales reason.

6. **The 5.5 statistics paragraph.** Asked in Gemini (`G2-12,13,14,15`) and again in your notes: the paired comparison, CI straddling zero, Wilcoxon p, 80% power, TOST margin, and the Llama-3 "systematically worse than random" contrast. High-value teaching target, still `[OPEN]` in this chat.

7. **Scale versus anisotropy (5.8)** and **missing config (5.5.3)**, as before.

**Where explanations went wrong (meta-signal).** You had to correct the assistant several times, which flags genuinely subtle spots: `G1-34/35` you rejected the AI's equations twice and pasted the thesis text to force the right ones; `G1-43` "don't tell me what can be used, tell me what the author actually used, don't make stuff up"; `G1-46` you pushed back that with `sqrt(epsilon)` noise a smaller step should *increase* noise; `G1-47` the AI wrongly discussed the continuous sampler when the passage was about the discrete one; and in the earlier baselines thread you caught the same reason ("left context") being used for two opposite outcomes.

---

## 2. Questions by chapter and section

### Chapter 2 — Background  `(gemini, mostly new)`

- `G1-01` `[ANS-G]` `DEF` What does **discretization** mean in "the update in (5) is a discretization of a continuous-time process, and any finite step size introduces a bias"?
- `G1-02` `[ANS-G]` `MECH` The **MH acceptance formula** in (6) is unclear. Explain it with an example of a good transition and a bad transition.
- `G1-03` `[ANS-G]` `MECH` Explain the "two facts about the correction" paragraph: (a) correction-free updates are biased optimizers not samplers, (b) the proposal ratio must use the drift at the proposed point.
- `G1-04` `[ANS-G]` `MECH` I didn't get the **discontinuity trap** (reverse proposal collapsing across a drift discontinuity).
- `G1-05` `[ANS-G]` `CONTRA`/`DEF` 2.4 says "there is no state between two tokens: one cannot be halfway between cat and dog." Is it actually like that?
- `G1-06` `[ANS-G]` `DEF` **Grathwohl et al. 2021** is cited here — what did they do and how is it related?
- `G1-07` `[ANS-G]` `DEF`/`MECH` What is a **first-order Taylor surrogate**, what is a Taylor expansion, and what is the "second term" it refers to?
- `G1-08` `[ANS-G]` `MECH` Explain the "two objects must be kept apart" paragraph: the **piecewise-constant energy** (no classical gradient) versus the **differentiable pathway** the sampler actually differentiates.
- `G1-09` `[ANS-G]` `CONTRA` **The slope-versus-zero-gradient paradox (important).** How can the pathway have a "slight smooth slope" inside a cell if you also say the true energy surface is flat plateaus with zero gradient?
- `G1-10` `[ANS-G]` `MECH` Show me the two equations (the differentiable-pathway / proposal equations).
- `G1-11` `[ANS-G]` `MECH` Explain the sampler **code** in detail (what is actually happening).
- `G1-12` `[ANS-G]` `MECH` Explain the two CLS equations, where they are in the code, and how they differ.
- `G1-13` `[ANS-G]` `MECH` Using `prep.py`, re-explain the "two objects" paragraph with a concrete example.

### Chapter 3 — Related work / diffusion (Section 3.5, SEDD)  `(gemini, new; relevant to 5.13)`

- `G1-14` `[ANS-G]` `DEF` Explain 3.5 in detail: what is a **positive control**, what is **score entropy discrete diffusion (SEDD)**, and what are "ratios of the data distribution over a discrete state space"?
- `G1-15` `[ANS-G]` `MECH` What is this **probability ratio**, and how does SEDD find its candidate token?
- `G1-16` `[ANS-G]` `MECH`/`CONTRA` How is `p(x)` calculated, given that AR log-likelihood is argued to be a bad measure (Holtzman, 3.4)?
- `G1-17` `[ANS-G]` `SYNTH` Is this a theoretically sound finding, or is the author saying something contradictory?
  - *These four are the conceptual groundwork for the 5.13 SEDD questions in your notes, which are still `[OPEN]`.*

### Chapter 4 — Methodology  `(gemini, mostly new)`

- `G1-18` `[ANS-G]` `CONTRA` 4.1: open-ended generation gives no handle, infilling does (via ground truth). But synonyms exist, so exact accuracy fails, and KL is on log-likelihood, which is itself argued to be a bad measure. I'm confused.
- `G1-19` `[ANS-G]` `DEF` 4.4: Euclidean distance is called anisotropic (citing Ethayarajh 2019). Tell me more.
- `G1-20` `[ANS-G]` `MECH`/`DEF` 4.4: explain the KL **positional-dependence** limitation and the **equivalence margin** (5% of policy KL; across-seed std 0.183; margin ~0.327). And explain **BERTScore** and how it is computed.
- `G1-21` `[ANS-G]` `MECH` Follow-up: how does corrupting an early token "alter the mathematical context for the words that follow"? What does "absence of a gap = throwing a random dart" mean and how is it proven? What is this "natural background noise" (0.183), how was it computed, and how did they get 5%?
- `G1-22` `[ANS-G]` `MECH` BERTScore follow-up: I got the matrix, but not how precision/recall/F1 are computed, how meaning is accommodated, or how IDF weighting is used.
- `G1-23` `[ANS-G]` `CONTRA` 4.5: the GFlowNet only evaluates rewards, never differentiates, so the gradient pathology is invisible to it. But (1) if the gradient is uninformative, doesn't that already mean the likelihood is uninformative? (2) if the literature says likelihood is a bad measure, why use it as the reward?

**The GFlowNet-prompt-versus-Langevin thread (your `IDEA`, `G1-24` to `G1-31`)** — a sustained, high-value exchange:
- `G1-24` `[ANS-G]` `MECH`/`IDEA` What was the purpose of adding the GFlowNet-trained variants to the Langevin configuration? The GFlowNet infill is a prompt restructuring, different from the Langevin infill — did the author compare them, and are they comparable? Why not use the GFlowNet prompt formulation inside the Langevin samplers to remove the KL positional inconsistency? Wouldn't that also address the introduction's global-property rewriting problem if you use many masks?
- `G1-25` `[ANS-G]` `PUSHBACK` You didn't get my 3rd and 4th questions. GFlowNet formats the prompt as `[Starting][Ending][Generation]`, turning infilling into AR generation — now answer based on that.
- `G1-26` `[ANS-G]` `MECH` How is the final-token gradient exactly zero? And in the Langevin sampler with multiple masks, are all tokens recovered simultaneously while satisfying constraints/fluency?
- `G1-27` `[ANS-G]` `MECH` Show the math behind the zero-gradient claim with an example.
- `G1-28` `[ANS-G]` `MECH`/`IDEA` How does this connect to using the GFlowNet prompt framing in the Langevin infill task? Explain with an example.
- `G1-29` `[ANS-G]` `IDEA` But can't we use the gradient of "and"/"dog" to predict the mask, and fit a constraint as well?
- `G1-30` `[ANS-G]` `MECH` (re-ask) Why do you say the final-token gradient is exactly zero?
- `G1-31` `[ANS-G]` `PUSHBACK`/`IDEA` I'm asking something different and it changes the sampler: is it possible to frame the prompt the GFlowNet way and use the gradients of the **context tokens** to steer the masked-token generation under Langevin, or something similar?

**The linearization-equations thread (`G1-32` to `G1-35`)** — repeated correction of the AI:
- `G1-32` `[ANS-G]` `MECH` Explain the 4.7 experiments, especially the linearization one, in detail.
- `G1-33` `[ANS-G]` `MECH` More detail on linearization, all equations. You skipped the other 4.7 experiments (acceptance, trajectory, likelihood trap, anisotropy). Where was the cosine-similarity test mentioned?
- `G1-34` `[ANS-G]` `PUSHBACK` Your equations differ from the thesis 4.7 ones — use only those. Also, does the likelihood trap apply to both DLS and CLS (they have different gradient mechanisms)?
- `G1-35` `[ANS-G]` `PUSHBACK` Still not the correct equations. Here is the exact text [pastes the linearization paragraph with the self/future decomposition].

### 5.1 — Step-size calibration and embedding geometry

- `G1-36` / `(notes)` `[ANS-G]`/`[OPEN]` `DEF` What does the **calibrated vs guided motion** passage mean?
- `G1-37` / `(notes)` `[ANS-G]`/`[OPEN]` `DEF` What is **mean pairwise distance** (2nd paragraph)?
- `G1-38` `[ANS-G]` `DEF` How is it different from "mean distance between neighbouring token embeddings"?
- `G1-39` `[ANS-G]` `PUSHBACK` Why "a continuous sampler is never truly plug-and-play" — isn't that true for the discrete sampler as well?
- `G1-40` / `(notes)` `[ANS-G]`/`[OPEN]` `WORDING` "which **qualifies** the plug-and-play framing" — shouldn't it be *disqualifies*? *(Resolved: "qualifies" = limits/adds a caveat, not "validates.")* Plus: what selects the step size in **Welling & Teh (2011)**?
- `G1-41` / `(notes)` `[ANS-G]`/`[OPEN]` `MECH` What does the **oracle sentence** mean (ideal step size using the ground truth, separating calibration failure from guidance failure)?
- `(claude)` `[ANS-C]` `MECH` How does the oracle work exactly (DLS/CLS), and would capping the step at the mean inter-token distance fix it? *(No: the gradient-valid radius and the smallest token-changing move don't overlap; the oracle already bounds all step-size policies; Llama-3's tighter geometry is the worst case; a cap ignores the self-term blindness.)*

### 5.2 — Annealing dynamics and the quenching effect

- `G1-42` / `(notes)` `[ANS-G]`/`[OPEN]` `DEF` What are the **per-step metrics**?
- `G1-43` `[ANS-G]` `PUSHBACK` Don't tell me what *can* be used — tell me exactly what the author used, from the document only, don't make stuff up. *(Flags that the prior answer over-generalised.)*
- `G1-44` / `(notes)` `[ANS-G]`/`[OPEN]` `MECH` How does a large step size make the Gaussian noise dominate and accept bad proposals, and how does annealing make the noise vanish?
- `G1-45` / `(notes)` `[ANS-G]`/`[OPEN]` `CONTRA` MH-on converges to a better KL (6.541 vs 9.499) — but if the gradient is useless, how? Is it the MH correction?
- `G1-46` `[ANS-G]` `PUSHBACK`/`MECH` Re-ask of the noise question with a correction: the author uses `sqrt(epsilon)` as the noise, so when the step size goes to zero, shouldn't the noise *increase*, not vanish? *(Subtle: absolute noise sqrt(eps) shrinks, but the point is noise-relative-to-drift and the collapse into deterministic descent; worth a crisp answer.)*
- `G1-47` `[ANS-G]` `CONTRA`/`PUSHBACK` (whole 5.2 pasted) I thought this was about the **discrete** sampler, but you discussed the continuous one. *(The AI mis-attributed; the passage is DLS.)*
- `G1-48` / `(notes)` `[ANS-G]`/`[OPEN]` `MECH`/`MISSING` In Figure 1, grad-norm-preserved-random and random-noise overlap perfectly — why? Is the gradient-normalization state stated anywhere?

### 5.3 — MH breakdown in continuous space

- `G1-49` / `(notes)` `[ANS-G]`/`[OPEN]` `DEF` In Figure 3, what are the y-axis counts, and how do they relate to +4.60 (target log-ratio) and -1325 (proposal log-ratio)?

### 5.4 — Sampler trajectories / off-manifold drift

- `G1-50` / `G2-02` / `(notes)` `[ANS-G]`/`[OPEN]` `DEF`/`MECH` Explain the off-manifold-distance passage (118 to 151 with MH on; ~980 peak, ~17 end with MH off; 65 to 83x the spacing). What is "drift off the manifold" and what is a **manifold**?
- `G2-03` `[ANS-G]` `PUSHBACK`/`IDEA` But if CLS does that, wouldn't a **smaller step size** improve it?
- `G2-04` / `(notes)` `[ANS-G]`/`[OPEN]` `MECH` What does the last sentence mean: "the discrete sampler avoids the choice by never leaving the manifold"?
- `G2-05` `[ANS-G]` `PUSHBACK` I want to understand **why CLS overshoots** even after the step size is calibrated.
- `G2-06` `[ANS-G]` `MISSING` Did the authors try a smaller step size, and do they report exact numbers?
- `G2-07` `[ANS-G]` `MECH` (pastes the `gpt2-large.dls...oracle.s50` config JSON with the mean_l2/mean_kl/entropy arrays) Explain why this schedule.
- `G2-08` `[ANS-G]` `MECH` The starting epsilon is 10.5 and the end is 0.1 — how did they arrive at these?

### 5.5 — Gradient vs a norm-matched random direction (+ statistics)

- `G2-09` / `(notes)` `[ANS-G]`/`[OPEN]` `DEF` What is "**final KL divergence**" — mean KL at the final step or overall?
- `G2-10` / `(notes)` `[ANS-G]`/`[OPEN]` `DEF` What are **nats**?
- `G2-11` / `(notes)` `[ANS-G]`/`[OPEN]` `MECH` "norm-preserved and fully random are indistinguishable in the separate comparison, so the same holds for the magnitude" — which comparison, and what does it mean?
- `G2-12` / `(notes)` `[ANS-G]`/`[OPEN]` `MECH` (whole paragraph) Explain the **paired comparison / equivalence** result (+0.171 mean diff, 95% CI [-0.285, +0.619], Wilcoxon p 0.40, 80%-power min-detectable 0.652, TOST margin 0.327) and the Llama-3 "**local linearization of a landscape that isn't locally linear**" contrast. "Didn't get anything."
- `G2-13` `[ANS-G]` `DEF` What do you mean the "statistics were completely clear" for Llama-3?
- `G2-14` `[ANS-G]` `MISSING` But did they show exact numbers for Llama-3?
- `G2-15` `[ANS-G]` `PUSHBACK` Wouldn't it have helped to run the same three tests (CI, Wilcoxon, power) on Llama-3 too?

### 5.5.1 — Robustness of the null

- `G2-16` / `(txt)` / `(notes)` `[ANS-G]`/`[OPEN]` `DEF` What is "**the null**" ("none overturned the null")?
- `(txt)` `[ANS-G]` `MECH`/`CONTRA` How was the **free-form prefix-continuation** task designed with Langevin, if Langevin can't do open-ended generation? *(And the extended debate: GFlowNet prompt format, "aren't they identical," "why run it at all" — see the cross-cutting thread below.)*

### 5.5.2 — Gradient-free baselines and the Gibbs sampler  `(txt/gemini)`

- `[ANS-G]` `DEF` **Untouched** vs **Random-token fill** — how do they differ?
- `[ANS-G]` `DEF`/`MECH` **Conditional Argmax** and **Conditional Sample** — are they just AR generation, what context, why so poor?
- `[ANS-G]` `MECH` How does the gradient-free **Gibbs sampler** work, and how many candidates per step; was the "exact score via forward pass" just the sentence log-likelihood?
- `[ANS-G]` `CONTRA` **Likelihood-trap contradiction (your sharpest):** if log-likelihood is a bad measure, how does Gibbs beat Langevin using that same measure?
- `[ANS-G]` `PUSHBACK` You reused "left context" as the reason for both Gibbs succeeding and Conditional Argmax failing — that doesn't make sense and feels like justifying the thesis.
- `[ANS-G]` `SYNTH` Synthesis: likelihood is good for single-token recovery, but its gradient is unusable, which is why Langevin fails?

### 5.5.3 — External-judge rescoring

- `(txt)` / `(notes)` `[ANS-G]`/`[OPEN]` `MISSING` The configuration (MH on/off, GN on/off, CLS/DLS) is never stated in the text or the source file. What was it?

### 5.6 — Linearization radius and self-term blindness  *(deepest hotspot)*

- `G1-35` / `(txt)` / `(notes)` `[ANS-G]`/`[OPEN]` `MECH` Why was the Taylor surrogate `Δ̂(v)=g^T(e(v)-e(x_i))` chosen; what does "at the level of the proposal itself" mean; the distance-limits paragraph (decay before 2.35 and 1.82).
- `(claude+gemini)` `[ANS-C]` `MECH`/`PUSHBACK` The last paragraph in full: the two limitations; why the input-embedding gradient registers only future tokens; math + code + example; line-by-line; how to fix. *(Fix: read the self term off the conditional — top-k rescore / Gibbs.)*
- `(claude)` `[ANS-C]` `MECH` Why does the masked token enter the self term as an **integer index** (two hats: continuous input vs integer output target), grounded in `prep.py`.
- `(claude)` `[ANS-C]` `MECH`/`DEF` What is the **decomposition experiment** measuring Δ_self and Δ_future (|self| 15.0, |future| 24.2 nats; surrogate rho 0.03 future, -0.10 self).
- `(claude)` `[ANS-C]` `MECH` Where the **soft-token equations** come from (one-hot inner product -> expectation under pi_i).
- `(claude)` `[ANS-C]` `MECH`/`DEF` What **pi_i** is, and why projecting it back to a corner reintroduces the Voronoi/MH breakdown.
- `(claude)` `[ANS-C]` `PUSHBACK` Shouldn't CLS-without-MH work well? Isn't this a sampler problem? *(MH breakdown is sampler-side; the gradient fallacy is energy/objective-side, proven by DLS + linearization + gradient-free baselines.)*

### 5.7 — The likelihood trap  `(claude+notes)`

- `[ANS-C]` `DEF` Holtzman (2020) **neural-text-degeneration**; **per-token likelihood**; **distinct-2**; what **Figure 10** shows; the **decoding strategies** (greedy, beam-20, ancestral, nucleus); how the experiment was run; line-by-line of 5.7 including the within-strategy caveat and the brevity slope with its censoring caveat.

### 5.8 — Embedding anisotropy  `(claude+notes)`

- `[ANS-C]` `WORDING`/`DEF` The section reads as a **scale** comparison, not anisotropy. The only true anisotropy evidence is the mean pairwise cosine (0.086 vs 0.018) and the broad-PCA sentence. Step-size consequence = scale; Euclidean-unreliability consequence = anisotropy. Fix: separate them.

### 5.10 — GFlowNet reward hacking  `(claude+notes)`

- `[ANS-C]` `MISSING` The len_beta=1 greedy output — generated how? *(Autoregressive greedy from the tuned policy, in the 5.7 decoding pass, not Langevin. Fix: state the generator for every degenerate example.)*

### 5.11 — Constrained generation  `(claude+notes)`

- `[ANS-C]` `MISSING` The infilled example doesn't mark which tokens were infilled. *(Recoverable from `mask_indices`; bold/bracket them.)*
- `[ANS-C]` `MECH` The steering-gain paragraph in simpler terms (sign flip = fixed drift; cons_only minus cons_random cancels it). Moving-walkway analogy.
- `(claude)` `[ANS-C]` `MECH` How the diagnostic was designed and how the labels were flipped (five modes; MH always scores the true energy; label flip = rerun lbl0 vs lbl1 matched, as the control that exposes the drift).

### 5.13 — Diffusion positive control (SEDD)  `(notes, OPEN; groundwork in G1-14..17)`

- `[OPEN]` `PUSHBACK`/`CONTRA` The opening attributes the failure to the training objective, not the sampler/energy — but isn't the zero self-term gradient a **sampler** problem?
- `[OPEN]` `MECH`/`DEF` 5.13.1: how is the experiment done on **SEDD**, which doesn't use a gradient like GPT-2 Large? What is the **(score)** column?
- `[OPEN]` `MECH` 5.13.3: why replace only the **proposal** in the MH correction; why not test with no-MH DLS or no-MH CLS; is there another way to use the SEDD direction?
- `[OPEN]` `MECH`/`CONTRA` 5.13.4: how does the **classifier-guidance** part work and what are the results — didn't we find guidance failed in 5.11? *(Reconcile: 5.11 = frozen-AR energy; 5.13.4 = score-trained landscape; kept apart on purpose.)*

---

## 3. Cross-cutting conceptual threads

### A. Your proposed alternative: GFlowNet framing + context-token gradients to steer Langevin  `(gemini, IDEA)`
`G1-24, 25, 28, 29, 31`. You repeatedly asked whether reformulating the prompt as `[Start][Ending][Generation]` and using the **context tokens' gradients** to steer the masked span under Langevin could sidestep the KL positional inconsistency, the global-property rewriting problem, and the end-token zero-gradient. This is your own research idea and it recurs, so it deserves an explicit "considered alternative and why it does not escape the diagnosis" paragraph. The crux tying it to 5.6/5.12: end-position masks get a zero (or self-blind) gradient, and moving the mask into the interior just returns you to the interior null.

### B. "Likelihood is a bad measure, yet it is everywhere"  `(gemini, CONTRA thread)`
Runs through `G1-16, 18, 23` and the 5.5.2/5.7 baselines. The same tension surfaces at every layer: SEDD's p(x), the KL metric, the GFlowNet reward, and the Gibbs baseline all lean on a quantity the thesis calls unreliable. The resolution (likelihood ranks single-token substitutions well; its gradient is unusable; the trap is a free-generation phenomenon) should be stated once, early and plainly, then referenced.

### C. Task design: can Langevin generate freely, is prefix-continuation meaningful  `(gemini+txt)`
The `[Start][Ending][Generation]` vs prefix-continuation debate, "aren't they identical," and "why run the experiment at all." Same mechanism as 5.6/5.12 (what the input-embedding gradient sees by position).

### D. Prior literature: how did MuCoLa/COLD "work"  `(txt/gemini)`
Post-hoc filtering, early-stopped biased optimizers, real constraint-direction signal vs noise fluency-direction, off-manifold classifier scores. Keep these together.

### E. The gradient-fallacy mechanism  `(claude)`
self/future -> integer-index self term -> soft-token relaxation -> pi_i -> projection revives Voronoi/MH breakdown -> step cap can't fix -> oracle bounds all step policies. A clean self-contained lecture in that order.

### F. The MH-correction background chain  `(gemini)`
`G1-01..13`: discretization bias -> MH accept/reject -> proposal ratio uses drift at proposed point -> discontinuity trap -> piecewise-constant energy vs differentiable pathway -> the slope-vs-zero paradox. This is the theoretical spine that 5.3 and 5.4 pay off, and several of these (`G1-09` especially) were hard-won.

---

## 4. Actionable thesis-gap fixes surfaced

1. **5.6 self-term wording** — say 15.0 and 24.2 nats are mean absolute magnitudes of signed quantities.
2. **5.10 / 5.7 generator attribution** — state that degenerate strings come from greedy AR decoding, not Langevin.
3. **5.11 infilled-token marking** — bold/bracket recovered positions using `mask_indices`.
4. **5.8 scale vs anisotropy** — separate the two and attach each consequence to its true cause.
5. **5.11 label-flip rigor** — confirm prompts/seeds matched; note the guide-and-grade circularity.
6. **5.5.3 missing config** — add MH/GN/sampler to text and source metadata.
7. **5.2 / Figure 1 caption** — state the gradient-normalization setting (explains the overlap).
8. **5.1 wording** — reconsider "qualifies the plug-and-play framing" (readers hit the qualifies/disqualifies trap).
9. **2.4 slope-vs-flat wording** (`G1-09`) — the "almost flat with a slight smooth slope" vs "exactly flat, zero gradient" distinction confused a careful reader; make the pathway/energy distinction unmissable at first use.
10. **4.4 BERTScore and equivalence margin** — these needed two rounds each in Gemini; a one-paragraph primer (or an appendix box) would help readers who are not measurement specialists.

---

## 5. Where explanations needed correction (subtle-spot map)

Points where you pushed back or corrected the assistant. These mark genuinely tricky material worth extra care in the writeup and in any generated explanation:
- `G1-34, 35` — the linearization equations had to be corrected twice against the thesis text.
- `G1-43` — "don't make stuff up," after an answer listed what *could* be used instead of what the author used.
- `G1-46` — the `sqrt(epsilon)` noise objection (absolute vs relative noise as the step anneals).
- `G1-47` — discrete vs continuous sampler mix-up in the 5.2 explanation.
- `G1-25, 31` — your GFlowNet-steering idea was not initially engaged on its own terms.
- baselines thread — the same "left context" reason used for two opposite outcomes.

---

## 6. Open items — priority queue for the next pass

Still unanswered in this Claude chat (many were resolved in Gemini; re-derive if you want them in your own words and verified against the current thesis):

- **Ch 2 background:** MH formula with examples, discontinuity trap, piecewise-constant vs pathway, the slope-vs-zero paradox (`G1-02,03,04,08,09`).
- **Ch 3 / SEDD:** positive control, score-entropy discrete diffusion, probability ratio, p(x) computation (`G1-14..17`) — needed for the 5.13 items below.
- **Ch 4:** anisotropic Euclidean distance, KL positional dependence, equivalence margin and the 0.183/0.327/5% numbers, BERTScore (`G1-19..22`); the GFlowNet-reward-is-likelihood confusion (`G1-23`); your GFlowNet-steering idea (`G1-24..31`).
- **5.1:** calibrated vs guided motion, mean pairwise distance, qualifies/disqualifies, Welling-Teh step selection, the oracle sentence.
- **5.2:** per-step metrics, noise scaling (incl. the sqrt-epsilon objection), MH-on convergence, Figure 1 overlap + gn state.
- **5.3:** Figure 3 y-axis and its link to +4.60 / -1325.
- **5.4:** off-manifold passage and manifold definition; "never leaving the manifold"; why CLS overshoots after calibration; whether smaller steps help; the schedule JSON and how 10.5 -> 0.1 was chosen.
- **5.5:** final KL definition, nats, the "same holds for the magnitude" comparison, the full equivalence paragraph, and the Llama-3 systematic-mis-specification contrast (incl. why the three tests were not run on Llama-3).
- **5.13:** the sampler-vs-objective scoping challenge, the SEDD setup and (score) column, why only the MH proposal was swapped, and the 5.13.4 classifier-guidance results vs 5.11.
