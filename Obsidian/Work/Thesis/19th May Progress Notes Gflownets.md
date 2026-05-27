# Detailed Progress Notes: Controllable Text Generation (Langevin vs GFlowNets)

**Date:** May 19, 2026

## 1. Theoretical Insights from GFlowNet Codebase (Moksh Jain et al., ICLR 2024)

We analyzed the gfn-lm-tuning repository to understand how Amortized Inference is achieved.

- **The Backward Probability Hack (**
    
    **`log⁡PB=0logPB​=0`**
    
    **):** Standard GFlowNets require learning a backward policy (
    
    ```
    PBPB​
    ```
    
    ). However, because text generation is a directed tree, stepping backward is deterministic (there is only one way to un-generate a word). Therefore, 
    
    ```
    PB=1PB​=1
    ```
    
     and 
    
    ```
    log⁡PB=0logPB​=0
    ```
    
    . The authors completely omitted the backward probability from their modified_subtb_loss.
    
- **The Infilling "Cheat":** True infilling (updating tokens in the middle of a sequence, as done with Langevin Dynamics) creates a Directed Acyclic Graph (DAG) state space where 
    
    ```
    PB≠1PB​=1
    ```
    
    . To avoid this, the GFlowNet code reformulates infilling as a strictly left-to-right generation task. It generates the middle reasoning (
    
    ```
    ZZ
    ```
    
    ) from the prompt (
    
    ```
    XX
    ```
    
    ) left-to-right, then physically appends the target (
    
    ```
    YY
    ```
    
    ) via a helper function (append_sol_and_remove_eos) and evaluates the reward of the concatenated string 
    
    ```
    X+Z+YX+Z+Y
    ```
    
     using the frozen LLM.
    
- **Superposing Flows (Reward Formulation):** The target energy landscape 
    
    ```
    R(x)R(x)
    ```
    
     is constructed by switching from the LoRA weights to the frozen Base model (lora_to_base), extracting the natural sequence log-probabilities, applying heuristic rule validators (e.g., grammar checks), and merging them using torch.min(reward, invalid_penalty).
    

## 2. Experimental Setup: The Pivot to GPT-2

To make a scientifically rigorous comparison against the GFlowNet codebase, we shifted our Langevin baselines from Llama-3/Wikitext-2 to GPT-2/ROCStories.

- **Infrastructure Strategy:** Moksh Jain confirmed via email that the provided codebase is outdated and does not use vLLM or Distributed Data Parallel (DDP). Instead of migrating the math to a modern library (like trl or prime-rl), we opted to run the original code on a single RTX A6000. GPT-2 is small enough (124M parameters) that multi-GPU communication overhead would actually slow down training.
    
- **Base Model:** We performed Supervised Fine-Tuning (SFT) on the ROCStories dataset to create gpt2_sft_output, which serves as the identical mathematical initialization for both the Langevin samplers and the GFlowNet RL loop.
    

## 3. Langevin Dynamics Baseline Findings on GPT-2

We successfully ran both Discrete Langevin Sampler (DLS) and Continuous Langevin Sampler (CLS) on the GPT-2 SFT model. The results empirically proved all core thesis hypotheses:

- **The "Frozen Sampler" Calibration Issue:** Initially, all samplers flatlined on GPT-2. The DLS entropy immediately dropped to 0.0. We discovered this was a **Hyperparameter Mismatch**. The L2 distance between GPT-2 tokens is massive (~4.5) compared to Llama-3 (~0.5). Step sizes tuned for Llama-3 (
    
    ```
    α≈0.1α≈0.1
    ```
    
    ) were too small to cross GPT-2's Voronoi boundaries.
    
- **The Oracle Fix:** Expanding the alpha_grid and turning the Oracle ON revealed that GPT-2 requires a step size multiplier around 10.5. We implemented a new linear annealing schedule: np.linspace(10.5, 0.1, 50).
    
- **Confirmation of the Gradient Fallacy (DLS):** With the correct schedule, DLS properly "boiled" (high entropy) and "quenched" (locked onto tokens). Crucially, the exact Policy gradient performed almost identically to Random Noise. **Finding:** Local gradients are completely uninformative when traversing the large Euclidean distances required to jump between discrete tokens.
    
- **Confirmation of the MH Breakdown (CLS):** In CLS, the Metropolis-Hastings (MH) correction resulted in a 100% rejection rate (flatline trajectories). **Finding:** Crossing a token's Voronoi cell boundary causes a discontinuous shift in the gradient, making the reverse probability (
    
    ```
    qbackqback​
    ```
    
    ) practically zero. MH mathematically panics and rejects the move. This proves continuous MCMC samplers are fundamentally incompatible with discrete text embeddings, regardless of model size (Llama-3 vs GPT-2).
    

## 4. Current Status: GFlowNet Training Collapse & The "Alignment Tax"

We trained the GFlowNet (PEFT) model using the default repo configs.

- **The Alignment Tax:** When running DLS on the trained GFlowNet, the KL Divergence was higher (~7.8) than on the Base SFT model (~6.5). **Finding:** This is expected. The Base SFT model minimizes pure cross-entropy (perfect local fluency). The GFlowNet trades away pure fluency to satisfy global constraints, resulting in a higher KL divergence (the Alignment Tax).
    
- **The Generation Bug (Mode Collapse):** When evaluating pure autoregressive generation, the Base model produced valid (but repetitive) English, while the GFlowNet output pure garbage (スススス, he).

![[Pasted image 20260519104116.png]]

- **Lukas's Log Diagnosis (Meeting May 4, 2026):**
    
    - The training rewards violently fluctuated (-28 to -100), and the training loss was massive.
        
    - The eval_reward logged exactly 0.0 at every step, which is mathematically suspicious given the garbage output.
        

## 5. Immediate Action Items (From Lukas)

1. **Hyperparameter Check:** Open the ICLR paper's appendix to find the true Learning Rate, batch size, and subtb_lambda used for the ROCStories infilling task. Compare them to the codebase defaults, as the high variance suggests the learning rate was too high.
    
2. **Debug eval_reward:** Inspect main.py / NextSentenceGFNTask to figure out why the evaluation reward is defaulting to zero.
    
3. **"Plan B" (Score Matching):** If the GFlowNet RL math is too brittle, Lukas proposed abandoning the rollout-based Modified SubTB loss.
    
    - Lukas Transcript Quote: "If we can do it more easily... so that the only thing that is doing is that it makes the energy of the full sequence somehow reflect the probability of the sequence. [...] If you can just fine-tune it for score matching, that's what we need, right?"
        
    - Concept: Use Score Matching to directly align the LLM's sequence gradients to the target energy function, bypassing the need for generating thousands of RL rollouts.