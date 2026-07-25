import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import numpy as np
from huggingface_hub import login

# login(token="TODO")


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
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
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
    log_joint = -nll_sum  # sum of log probs
    return log_joint


def langevin_infilling_single_position(
    model,
    tokenizer,
    input_ids,
    position,
    steps,
    return_all_states=False,
    do_mh=True,
    mh_burn_in = 40,
    noise_scale = 0.01,
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
    s = s.clone().detach().requires_grad_(True)

    def log_pi(s: torch.Tensor) -> torch.Tensor:
        """
        log π(s): use LM joint log-prob over the *full sequence* while
        setting target_ids at `position` to the *projection* of s.
        """
        with torch.no_grad():
            proj_idx = project_to_vocab_by_l2(s.detach(), emb_matrix)     # [1,1]
            proj_token_id = proj_idx[0].to(torch.int64)

        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[:, position[0]:position[1]] = s
        target_ids = input_ids.clone()
        target_ids[:, position[0]:position[1]] = proj_token_id

        return joint_log_prob_from_inputs_embeds(
            model, inputs_embeds, target_ids, attention_mask
        )  # scalar
    
    def linear_epsilon_schedule(k, total_steps=steps, eps0=1e-1, eps_min=1e-2):
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
    
    def grad_log_pi(s: torch.Tensor) -> torch.Tensor:
        # project to vocab for discrete target (no gradient)
        with torch.no_grad():
            proj_idx = project_to_vocab_by_l2(s.detach(), emb_matrix)
            proj_token_id = proj_idx[0].detach().to(torch.int)

        # replace embedding at position with s
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[0, position[0]:position[1]] = s
        target_ids = input_ids.clone()
        target_ids[0, position[0]:position[1]] = proj_token_id

        # compute log joint and gradient wrt s
        log_joint = joint_log_prob_from_inputs_embeds(
            model, inputs_embeds, target_ids, attention_mask
        )
        grad_s = torch.autograd.grad(log_joint, s, retain_graph=False)[0]
        return grad_s, log_joint
    
    def logq_gaussian(x, mean, eps):
        """
        log N(x; mean, eps I) up to constant:
        = - ||x - mean||^2 / (2 eps)  - (D/2)log(2π eps)
        We keep the full expression for numerical stability in MH ratio.
        """
        diff = (x - mean)
        D = diff.numel()
        quad = (diff.pow(2).sum()) / (2.0 * eps)
        const = 0.5 * D * math.log(2.0 * math.pi * eps)
        return -(quad + const)

    s_states = []
    logpi_traj = []
    token_list = []

    # Precompute current log π(s) once
    with torch.no_grad():
        cur_logpi = log_pi(s)
        # logpi_traj.append(cur_logpi)

    for k in range(steps):        
        eps_k = linear_epsilon_schedule(k)
        # print(f"Iteration{k} with epsilon {eps_k}")
              
        # calculate grad_s
        grad_s, log_joint = grad_log_pi(s)
        

        noise = torch.randn_like(s) * math.sqrt(eps_k*noise_scale)
        
        # m_s = s + 0.5*eps_k * grad_s
        # s_u = (m_s + noise).detach()

        interim = s + 0.5*eps_k*grad_s
        interim_proj = emb_matrix[project_to_vocab_by_l2(interim, emb_matrix=emb_matrix)]

        m_s = 0.5*(interim + interim_proj)
        s_u = (m_s + noise).detach()
        # do LD update
        # s_u = s + eps_k * grad_s + noise
        # soft back projection of s after each step (this is a heuristic modification)
        proj_idx = project_to_vocab_by_l2(s_u, emb_matrix=emb_matrix)
        # s_proj = emb_matrix[proj_idx]
        # s_prop = (0.5*(s_u + s_proj)).detach().clone().requires_grad_(True)
        # s_prop = s
        # try88ing new method discussed with lukas for MH end nov
        s_prop = s_u.detach().clone().requires_grad_(True)

        # Decoding prop token for debugging
        # prop_idx = project_to_vocab_by_l2(s.detach().clone(), emb_matrix)
        prop_token_id = proj_idx[0].to(torch.int)
        prop_token = tokenizer.decode(prop_token_id, skip_special_tokens=True)
        token_list.append(prop_token)

        print(f"Iteration {k} with epsilon {eps_k: .5f} | Proposal token: {prop_token} | Log Joint: {log_joint}")

        if do_mh:
            if k >= mh_burn_in:
                with torch.no_grad():
                    # prop_logpi = log_pi(s_prop)
                    prop_logpi = log_pi(m_s)

                # 5) Reverse proposal log q(s | s')
                # need grad at s' to form mean m' = s' + (eps/2) * g(s')
                # g_prop = grad_log_pi(s_prop)
                # m_prop = s_prop + 0.5 * eps_k * g_prop
                g_prop, log_joint = grad_log_pi(m_s)
                interim_prop = m_s + 0.5 * eps_k * g_prop
                interim_prop_proj = emb_matrix[project_to_vocab_by_l2(interim_prop, emb_matrix=emb_matrix)]
                m_prop = 0.5*(interim_prop + interim_prop_proj)

                log_q_back = logq_gaussian(s, m_prop, eps_k) # Correct

                # 6) Forward proposal log q(s' | s) 
                log_q_fwd = logq_gaussian(s_prop, m_s, eps_k) # Correct

                # 7) MH accept in log-space
                log_alpha = (prop_logpi - cur_logpi) + (log_q_back - log_q_fwd)
                accept = (torch.log(torch.rand((), device=device)) < log_alpha)

                # prop_states = list() 

                # # Decoding prop token for debugging
                # prop_idx = project_to_vocab_by_l2(s_prop.detach().clone(), emb_matrix)
                # prop_token_id = prop_idx[0, 0].to(torch.int)
                # prop_token = tokenizer.decode(prop_token_id, skip_special_tokens=True)

                acc = "Accepted" if accept.item() else "Rejected"
                print(acc)
                # print(f"Iteration {k} with epsilon {eps_k: .5f} | Proposal token: {prop_token} | {acc} ")

                if accept.item():
                    s = s_prop.detach().requires_grad_(True)
                    cur_logpi = prop_logpi.detach()
                    logpi_traj.append(log_joint)
            
            else:
                s = s_prop
        else:
            s = s_prop
            with torch.no_grad():
                # prop_logpi = log_pi(s_prop)
                prop_logpi = log_pi(m_s)
            cur_logpi = prop_logpi.detach()
            logpi_traj.append(log_joint.item())

        if return_all_states:
            s_states.append(s.detach().cpu().clone())

    # Final projection
    final_token_idx = project_to_vocab_by_l2(s, emb_matrix)
    token_id = final_token_idx[0].detach().to(torch.int)

    return token_id, s.detach(), s_states if return_all_states else None, logpi_traj, token_list


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
    position = (3, 4)  # last token (0-based)
    input_ids[0, position[0]:position[1]] = 2000#torch.randint_like(input_ids[0, position[0]:position[1]], 0, 30000)
    print("Distorted tokens:", tokenizer.decode(input_ids[0]))

    token_id, final_s, states, logpi_traj, token_list = langevin_infilling_single_position(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        position=position,
        steps=50,
        return_all_states=True,
        do_mh=False,
        noise_scale=0.01   
    )

    
    # Make a copy of the tensor
    infilled_sequence = toks.input_ids.clone()   # creates an independent tensor

    # Replace the token at `position` for the first (and only) sequence in batch
    infilled_sequence[0, position[0]:position[1]] = token_id

    # Decode the full infilled sequence to text
    infilled_text = tokenizer.decode(infilled_sequence[0], skip_special_tokens=True)

    backprojected_tokens = []
    decoded_tokens = []

    emb_layer = model.get_input_embeddings()
    emb_matrix = emb_layer.weight.detach().to(device)

    for s_step in states:
        # project to nearest token id
        token_idx = project_to_vocab_by_l2(s_step.to(emb_matrix.device), emb_matrix)
        token_id = token_idx[0].detach().to(torch.int)
        backprojected_tokens.append(token_id)
        # decode to string
        decoded_tokens.append(tokenizer.decode(token_id, skip_special_tokens=True))

    # print("Step | Token String")
    # for k, tstr in enumerate(decoded_tokens):
    #     print(f"{k:3d} | {tstr}")

    print("Projected token id:", token_id)
    print("Projected token string:", tokenizer.decode(token_id))
    print("Infilled sequence token IDs:", infilled_sequence.tolist())
    print("Infilled text:", infilled_text)

