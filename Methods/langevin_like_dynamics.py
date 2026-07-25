import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import numpy as np
# from huggingface_hub import login
from torch.distributions.categorical import Categorical

# login(token="PUT_YOUR_KEY_HERE")


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer_and_model(model_id: str, device=device, dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model

def get_embedding_matrix(model):
    # returns tensor (vocab_size, emb_dim)
    emb_layer = model.get_input_embeddings()
    return emb_layer.weight.detach()  # (V, D)

def project_to_vocab_by_l2(s: torch.FloatTensor, emb_matrix: torch.FloatTensor):
    """
    Projects a continuous embedding or batch of embeddings to the nearest token in the
    embedding matrix using Euclidean (L2) distance.

    Args:
        s: (..., D) continuous embedding(s) — can be (D,), (B, D), or (B, T, D)
        emb_matrix: (V, D) vocabulary embedding matrix

    Returns:
        top_idx: tensor of nearest token indices with shape matching s[..., 0]
    """
    # Ensure last dimension is embedding dim
    D = emb_matrix.size(1)
    assert s.size(-1) == D, f"Embedding dim mismatch: s has {s.size(-1)}, emb_matrix has {D}"

    original_shape = s.shape[:-1]  # e.g. (1, 2)
    s_flat = s.view(-1, D)         # Flatten to (N, D)

    # Compute squared Euclidean distance to all vocab embeddings
    diff = s_flat[:, None, :] - emb_matrix[None, :, :]  # (N, V, D)
    dist2 = torch.sum(diff ** 2, dim=-1)                # (N, V)

    # Find nearest vocab index
    top_idx = dist2.argmin(dim=-1)                      # (N,)

    # Reshape back to match original shape (excluding embedding dim)
    top_idx = top_idx.view(*original_shape)             # (B, T) or whatever s was

    return top_idx


def joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids, attention_mask=None):
    """
    Compute total log joint log P(sequence) for a causal LM using inputs_embeds.
    We compute sum_i log p(x_i | x_{<i}) for i=1..L (exclude position 0 where nothing predicts it).
    Implementation details:
      - Forward the model with inputs_embeds to obtain logits.
      - For causal LM, logits[:, :-1, :] predict tokens at positions 1..L-1 (i.e., next-token).
      - We compute the negative log-likelihood of targets shifted accordingly and sum across tokens.
    Returns: scalar log_prob (torch scalar)
    """
    # model outputs logits shape (B, L, V)
    outputs = model(inputs_embeds=inputs_embeds, return_dict=True)
    logits = outputs.logits  # (B, L, V)
    # We want p(x_i | x_<i) for i>=1. So take logits[:, :-1, :] and targets[:, 1:].
    logits_next = logits[:, :-1, :].contiguous()  # (B, L-1, V)
    target_next = target_ids[:, 1:].contiguous()  # (B, L-1)
    B, Lm1, V = logits_next.shape
    logits_flat = logits_next.view(B * Lm1, V)
    targets_flat = target_next.view(B * Lm1)
    # ignore padding if present: assume pad token id is tokenizer.pad_token_id or -100 already applied.
    # We will compute sum of negative log likelihoods for all non-ignored positions
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    nll_sum = loss_fct(logits_flat, targets_flat)  # scalar: sum of negative log-likelihoods
    log_joint = - nll_sum  # sum of log probs
    return log_joint


def langevin_infilling_single_position(
    model,
    input_ids,
    position,
    steps    
):
    """
    Langevin dynamics with decreasing epsilon schedule satisfying:
      Σ ε_k = ∞ and Σ ε_k^2 < ∞.
    """

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    emb_layer = model.get_input_embeddings()
    emb_matrix = emb_layer.weight.detach().to(device)

    # Initialize continuous token s
    with torch.no_grad():
        base_embs = emb_layer(input_ids)

    s = base_embs[:, position[0]:position[1], :]
    s = s.squeeze().clone().detach().requires_grad_(True)
    s_idx = input_ids[:, position[0]:position[1]].squeeze().clone().detach()
    
    def linear_epsilon_schedule(k, total_steps=steps, eps0=1e-1, eps_min=1e-3):
        """
        Linear decay schedule for epsilon.

        Args:
            k (int): current step index (0 <= k < total_steps)
            total_steps (int): total number of Langevin iterations
            eps0 (float): initial epsilon value
            eps_min (float): minimum epsilon value (floor)
        Returns:
            float: epsilon at step k
        """
        frac = k / float(total_steps)
        eps_k = eps0 * (1.0 - frac)
        return max(eps_k, eps_min)
    
    # def poly_step(eps0, t, alpha=0.55):
    #     return eps0 / ((1.0 + t) ** alpha)

    s_states = [s]
    s_ids = [s_idx]

    for k in range(steps):        
        eps_k = linear_epsilon_schedule(k)
        # eps_k = poly_step(1e-1, k)
        print(f"Iteration{k} with epsilon {eps_k}")
              
        # replace embedding at position with s
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[:, position[0]:position[1]] = s_states[-1]
        target_ids = input_ids.clone()
        target_ids[:, position[0]:position[1]] = s_ids[-1]

        # compute log joint and gradient wrt s
        log_joint = joint_log_prob_from_inputs_embeds(
            model, inputs_embeds, target_ids, attention_mask
        )
        
        # calculate gradient to get proposal
        grad_s = torch.autograd.grad(log_joint, s_states[-1], retain_graph=False)[0].squeeze()
        
        # sample new id based on the proposal
        diff = (emb_matrix - s_states[-1].unsqueeze(0))
        # t1 = -torch.norm(diff, dim=-1)/(2*eps_k)
        t1 = -torch.sum(diff * diff, dim=-1) / (2 * eps_k) 
        t2 = 0.5 * diff @ grad_s
        
        new_p = Categorical(logits=t1+t2)
        s_ids.append(new_p.sample().clone().detach())
        s_states.append(emb_matrix[s_ids[-1]].clone().detach().requires_grad_(True)) 
        forward_prob = torch.nn.functional.softmax(t1+t2)[s_ids[-1]]


        # replace embedding at position with s
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[:, position[0]:position[1]] = s_states[-1]
        target_ids = input_ids.clone()
        target_ids[:, position[0]:position[1]] = s_ids[-1]

        # compute log joint and gradient wrt s
        bw_log_joint = joint_log_prob_from_inputs_embeds(
            model, inputs_embeds, target_ids, attention_mask
        )
        
        # calculate gradient to get proposal
        grad_s_b = torch.autograd.grad(bw_log_joint, s_states[-1], retain_graph=False)[0].squeeze()

        backward_diff = (emb_matrix - emb_matrix[s_ids[-1]])
        # t1_b = -torch.norm(backward_diff, dim=-1)/(2*eps_k)
        t1_b = -torch.sum(backward_diff * backward_diff, dim=-1) / (2 * eps_k) 
        t2_b = 0.5 * backward_diff @ grad_s_b
        backward_prob = torch.nn.functional.softmax(t1_b+t2_b)[s_ids[-2]]
       
        accept_prob = torch.minimum(torch.tensor(1), torch.exp(bw_log_joint - log_joint)*(backward_prob/forward_prob))
        print(accept_prob)
        accepted = torch.bernoulli(accept_prob)
        
        # reject the last sample
        if accepted < 1:
            s_states = s_states[:-1]
            s_ids = s_ids[:-1]    
            print("rejected!")

        # --- Approximate MH acceptance via local Taylor expansion ---
        # Instead of recomputing log π(s') and ∇ log π(s') with another model call,
        # reuse the current log π(s) and gradient g_s to get a cheap estimate.

        # Compute log π(s') ≈ log π(s) + (∇ log π(s))ᵀ (s' - s)
        # s_old = s_states[-2].detach()
        # s_new = s_states[-1].detach()
        # delta_s = (s_new - s_old).view(-1)
        # delta_logpi = (grad_s.view(-1) * delta_s).sum()
        # approx_bw_log_joint = log_joint + delta_logpi  # first-order expansion

        # # approximate backward proposal using the same gradient (symmetry)
        # backward_diff = (emb_matrix - s_new.unsqueeze(0))
        # t1_b = -torch.sum(backward_diff * backward_diff, dim=-1) / (2 * eps_k)
        # t2_b = 0.5 * backward_diff @ grad_s
        # backward_logq = torch.log_softmax(t1_b + t2_b, dim=-1)
        # log_q_back = backward_logq[s_ids[-2]]
        # log_q_fwd = torch.log_softmax(t1 + t2, dim=-1)[s_ids[-1]]

        # # compute log acceptance ratio in log-space
        # log_alpha = (approx_bw_log_joint - log_joint) + (log_q_back - log_q_fwd)
        # accept = (torch.log(torch.rand((), device=device)) < log_alpha)

        # if not accept.item():
        #     # reject the last sample
        #     s_states = s_states[:-1]
        #     s_ids = s_ids[:-1]
        #     print("rejected (approx)!")

        # otherwise accept implicitly

    
    return [s.to(torch.int) for s in s_ids]


# ---------- Example usage ----------
if __name__ == "__main__":
    # model_id = "meta-llama/Meta-Llama-3.1-8B"  # demo with small model; swap with large llama-8b family model id
    model_id = "/speech/dbwork/mul/spielwiese3/llm_project/llm_models/pre-trained/Meta-Llama-3-8b/"
    tokenizer, model = load_tokenizer_and_model(model_id, device=device)

    # sample sequence: "Hello world !"
    prompt = "Dogs are considered to be our best friends."
    toks = tokenizer(prompt, return_tensors="pt")
    input_ids = toks.input_ids  # (1, L)
    print("Initial tokens:", tokenizer.decode(input_ids[0]))
    
    # flip some of the tokens
    position = (8, 9)  # last token (0-based)
    input_ids[0, position[0]:position[1]] = 12345 #torch.randint_like(input_ids[0, position[0]:position[1]], 0, 30000)
    print("Distorted tokens:", tokenizer.decode(input_ids[0]))

    token_ids = langevin_infilling_single_position(
        model=model,
        input_ids=input_ids,
        position=position,
        steps=100    
    )
    
    # Make a copy of the tensor
    infilled_sequence = toks.input_ids.clone()   # creates an independent tensor
    
    # Replace the token at `position` for the first (and only) sequence in batch
    infilled_sequence[0, position[0]:position[1]] = token_ids[-1]

    # Decode the full infilled sequence to text
    infilled_text = tokenizer.decode(infilled_sequence[0], skip_special_tokens=True)
    
    for s in token_ids:
        print(tokenizer.decode(s))
    print("Token id:", token_ids[-1])
    print("Token string:", tokenizer.decode(token_ids[-1]))
    print("Infilled sequence token IDs:", infilled_sequence.tolist())
    print("Infilled text:", infilled_text)

    