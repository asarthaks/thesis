Your thesis has done excellent diagnostic groundwork — you've identified that gradient direction from LLMs is largely uninformative and that the Voronoi-cliff landscape breaks continuous MCMC. Your Section 6.2 already frames GFlowNets as the motivated next step. Here are concrete, thesis-ready experiments you can run, organized from most essential to more exploratory.

## Core Experiment: GFlowNet as a Proposal Replacement

This is the most natural extension of your existing work. Since you empirically showed that the DLS gradient direction is no better than a random walk, replace the gradient-based proposal entirely with a trained GFlowNet policy. The setup mirrors your current infilling task on Wikitext-2 with Llama-3 8B as the frozen reward function:

$$
R(x) = p_{\text{LLM}}(x) \cdot \exp(\lambda \cdot C(x))
$$

Train a lightweight GFlowNet (e.g., adapter layers over a small transformer) using the **Trajectory Balance (TB)** objective, then compare directly against your already-benchmarked **DLS+MH** and **random-walk** baselines. This is the cleanest apples-to-apples comparison your thesis can make.

## Training Objective Ablation

Your thesis compares discrete vs. continuous sampling spaces. A natural parallel for GFlowNets is comparing training objectives:

- **Trajectory Balance (TB)** — global credit assignment, faster convergence
- **Detailed Balance (DB)** — local, analogous to your MH correction which already works well in DLS
- **Sub-trajectory Balance (SubTB)** — intermediate credit assignment, useful for longer sequences

This ablation directly answers your **RQ3** (improving sampling efficiency) by showing which amortization objective best suits the jagged discrete text landscape you characterized.

## Multi-Constraint Composition

One limitation of Langevin-based methods in your thesis is that handling multiple global constraints requires careful energy function design. GFlowNets naturally handle multiplicative reward composition:

$$
R(x) = p_{\text{LLM}}(x) \cdot \exp(\lambda_1 \cdot \text{sentiment}(x)) \cdot \exp(\lambda_2 \cdot \text{toxicity}(x))
$$

Run this on two simultaneous constraints (e.g., positive sentiment + topic adherence) and measure **KL divergence** and **MAUVE** against your single-constraint Langevin baselines. This directly showcases the advantage of amortization over per-instance optimization.

## Diversity vs. Control Trade-off Analysis

A core advantage GFlowNets have over MCMC — including your DLS — is that they are explicitly designed to sample proportionally from all modes of the reward, not just find a single mode. You should measure:

- **Self-BLEU** (generation diversity within a batch)
- **MAUVE** (already in your setup) against your DLS quenching results

Your *quenching effect* (Section 5.4) showed DLS collapses to a local minimum at the end of annealing. A GFlowNet trained with TB should not exhibit this, making this a strong comparative narrative.

## Amortization Generalization Test

This experiment tests the amortization payoff — train the GFlowNet on one constraint (e.g., positive sentiment) and evaluate zero-shot transfer to a related but unseen constraint (e.g., formal register or topic). Compare:

| Setting | GFlowNet (trained on sentiment) | DLS (per-instance, no training) |
|---|---|---|
| In-distribution (sentiment) | Expected: fast, accurate | Slower, gradient-uninformative |
| Out-of-distribution (topic) | Tests generalization | Same cost, no degradation |

This directly frames GFlowNets as **amortized inference** vs. your **per-instance MCMC**, the key conceptual contrast of Section 6.2.

## GFlowNet + MH Hybrid

Your finding that MH works well in discrete space (DLS) but not continuous (CLS) suggests a promising hybrid: use the GFlowNet as the proposal distribution inside an **MH accept/reject loop**. Since the GFlowNet learns to propose tokens proportionally to reward, the MH acceptance rate should be dramatically higher than random-walk or gradient proposals. This tests whether the theoretical correctness of MH is recovered when the proposal is better-calibrated.

## Practical Recommendation

Given your deadline (**August 2026**), the most feasible path is:

1. **First:** Run the direct GFlowNet-vs-DLS comparison on your existing infilling task — same metrics (**KL, entropy**), same dataset (**Wikitext-2**), same backbone (**Llama-3 8B**).
2. **Second:** Add the **TB vs. DB ablation** — it costs minimal extra compute but significantly strengthens **RQ3**.
3. **Third (if time permits):** Multi-constraint or amortization generalization experiments for a broader contribution.

The GFlowNet policy can be kept deliberately small (e.g., a **2-layer MLP over the LLM's hidden states**) to keep training tractable, which is consistent with the *"lightweight training step"* framing already in your **RQ3**.

---

Prepared using Claude Sonnet 4.6 Thinking

## Follow-ups

- Why GFlowNet training losses like trajectory balance outperform detailed balance and flow matching — key evidence from recent benchmarks and failure modes
- Deep research
- How to implement trajectory balance loss in GFlowNets
- GFlowNets vs reinforcement learning for molecule design
- Code examples for BALD-GFlowNet active learning
- Pretraining GFlowNets with cheap rewards tutorial