import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import numpy as np
from torch.distributions.categorical import Categorical


# from ..Utils.prep import *

device = "cuda" if torch.cuda.is_available() else "cpu"



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





def langevin_infilling_single_position(
    model,
    input_ids,
    position,
    tokenizer,
    steps=30,
    temperature=None,
    oracle=False,
    orig_ids=None,
    alpha_grid=np.logspace(-4, -1, 50),
    method="policy", #options: "policy", "grad_norm_preserved_random_dir", "random",
    mh_sampling=False,
    grad_normalization=False, # <--- ADDED THIS
    debug=True
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

    # --- FIXED SETUP: Safely handle orig_ids decoupled from Oracle ---
    emb_gt = None
    p_ref = None
    
    if orig_ids is not None:
        # Ground truth embedding
        gt_id = orig_ids[0, position[0]].item()
        emb_gt = emb_matrix[gt_id]

        # Pre-compute the Reference Distribution (Ground Truth Context)
        target_idx = position[0]
        seq_len = input_ids.shape[1]
        
        if target_idx < seq_len - 1:
            with torch.no_grad():
                out_ref = model(orig_ids)
                logits_ref = out_ref.logits[0, target_idx, :] 
                p_ref = torch.softmax(logits_ref, dim=-1) # Fixed Target Distribution

    # Initialize continuous token s
    with torch.no_grad():
        base_embs = emb_layer(input_ids)

    s = base_embs[:, position[0]:position[1], :]
    s = s.squeeze().clone().detach().requires_grad_(True)
    s_idx = input_ids[:, position[0]:position[1]].squeeze().clone().detach()

    def joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids, attention_mask=None):
        outputs = model(inputs_embeds=inputs_embeds, return_dict=True)
        logits = outputs.logits  # (B, L, V)
        logits_next = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        target_next = target_ids[:, 1:].contiguous()  # (B, L-1)
        B, Lm1, V = logits_next.shape
        logits_flat = logits_next.view(B * Lm1, V)
        targets_flat = target_next.view(B * Lm1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        nll_sum = loss_fct(logits_flat, targets_flat)  
        log_joint = - nll_sum  
        return log_joint
    
    def linear_epsilon_schedule(k, total_steps=steps, eps0=1e-1, eps_min=1e-3):
        frac = k / float(total_steps)
        eps_k = eps0 * (1.0 - frac)
        return max(eps_k, eps_min)

    def compute_q_logprob(token_emb, theta, grad, alpha):
        m = theta + 0.5 * alpha * grad
        diff = token_emb - m
        dist2 = (diff * diff).sum()
        return - dist2 / (2 * alpha)
    

    s_states = [s]
    s_ids = [s_idx]

    alpha_schedule = []
    chosen_tokens =[]
    theta_states = []
    all_gt_logprobs = []
    entropy_traj =[]
    metrics_history =[]
    # Custom alpha schedule
    alpha_schedule = np.linspace(10.5, 0.1, steps)
    
    # =========================================================================
    # OPTIMIZATION LOOP
    # =========================================================================
    for k in range(steps):
        # eps_k = linear_epsilon_schedule(k)
        #overriding with custom alpha schedule
        eps_k = alpha_schedule[k]
        if debug:
            print(f"****Iteration {k} with epsilon {eps_k}****")
              
        # replace embedding at position with s
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[:, position[0]:position[1]] = s_states[-1]
        target_ids = input_ids.clone()
        target_ids[:, position[0]:position[1]] = s_ids[-1]

        # compute log joint and gradient wrt s
        log_joint = joint_log_prob_from_inputs_embeds(
            model, inputs_embeds, target_ids, attention_mask
        )
        
        # --- FIXED BLOCK C: Method Variations & Gradient Normalization ---
        raw_grad_s = torch.autograd.grad(log_joint, s_states[-1], retain_graph=False)[0].squeeze()

        if method == "policy":
            if grad_normalization:
                grad_norm = raw_grad_s.norm(dim=-1, keepdim=True) + 1e-12
                grad_s = raw_grad_s / grad_norm
            else:
                grad_s = raw_grad_s
                
        elif method == "grad_norm_preserved_random_dir":
            rand_dir = torch.randn_like(raw_grad_s)
            rand_dir_unit = rand_dir / (rand_dir.norm(dim=-1, keepdim=True) + 1e-12)
            if grad_normalization:
                grad_s = rand_dir_unit
            else:
                grad_norms = raw_grad_s.norm(dim=-1, keepdim=True)
                grad_s = rand_dir_unit * grad_norms

        elif method == "random":
            rand_noise = torch.randn_like(raw_grad_s)
            if grad_normalization:
                grad_s = rand_noise / (rand_noise.norm(dim=-1, keepdim=True) + 1e-12)
            else:
                grad_s = rand_noise
        else:
            raise ValueError(f"Method '{method}' not defined")
        
        # --- FIXED ORACLE SAFEGUARD ---
        if oracle and emb_gt is not None:
            gt_logprobs =[]
            for alpha in alpha_grid:
                lp = compute_q_logprob(emb_gt, s_states[-1].detach(), grad_s, alpha)
                gt_logprobs.append(lp.item())

            assert len(gt_logprobs) == len(alpha_grid)

            row = np.array(gt_logprobs.copy())
            row = (row - row.min()) / (row.max() - row.min() + 1e-12)
            all_gt_logprobs.append(row)

            gt_logprobs = np.array(gt_logprobs)
            best_idx = np.argmax(gt_logprobs)
            alpha_star = alpha_grid[best_idx]
            alpha_schedule.append(alpha_star)
            
            if debug:
                print(f"[Step {k}] α* = {alpha_star:.6f}")
            eps_k = alpha_star
        
        # sample new id based on the proposal
        diff = (emb_matrix - s_states[-1].unsqueeze(0))
        t1 = -torch.sum(diff * diff, dim=-1) / (2 * eps_k) 
        t2 = 0.5 * diff @ grad_s

        logits = t1+t2
        scaled_logits = logits/temperature if temperature else logits
        probs = torch.softmax(scaled_logits, dim=0)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Compute entropy
        entropy_t = -(probs * torch.log(probs + 1e-12)).sum().item()
        entropy_traj.append(entropy_t)

        if debug:
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(f"Chosen token at step {k}: '{next_token}' | Value: {logits[next_token_id]}")
            indices = torch.argsort(logits, dim=0, descending=True)[:5]
            print(f"Entropy : {entropy_t}")
            print(f"Top 5 tokens according to the logits are: ")
            for it in indices:
                print(f"Token: {tokenizer.decode([it], skip_special_tokens=True)} | Value: {logits[it]}")

        s_ids.append(next_token_id)
        s_states.append(emb_matrix[next_token_id].clone().detach().requires_grad_(True)) 

        # Metropolis-Hastings Step
        mh_rejected = False 
        if mh_sampling:

            forward_prob = torch.nn.functional.softmax(scaled_logits, dim=0)[s_ids[-1]]

            inputs_embeds = base_embs.clone().detach()
            inputs_embeds[:, position[0]:position[1]] = s_states[-1]
            target_ids = input_ids.clone()
            target_ids[:, position[0]:position[1]] = s_ids[-1]

            bw_log_joint = joint_log_prob_from_inputs_embeds(
                model, inputs_embeds, target_ids, attention_mask
            )
            
            # --- FIXED MH BACKWARD GRADIENT SYMMETRY ---
            raw_grad_s_b = torch.autograd.grad(bw_log_joint, s_states[-1], retain_graph=False)[0].squeeze()

            if method == "policy":
                if grad_normalization:
                    grad_norm_b = raw_grad_s_b.norm(dim=-1, keepdim=True) + 1e-12
                    grad_s_b = raw_grad_s_b / grad_norm_b
                else:
                    grad_s_b = raw_grad_s_b
            elif method == "grad_norm_preserved_random_dir":
                rand_dir_b = torch.randn_like(raw_grad_s_b)
                rand_dir_unit_b = rand_dir_b / (rand_dir_b.norm(dim=-1, keepdim=True) + 1e-12)
                if grad_normalization:
                    grad_s_b = rand_dir_unit_b
                else:
                    grad_norms_b = raw_grad_s_b.norm(dim=-1, keepdim=True)
                    grad_s_b = rand_dir_unit_b * grad_norms_b
            elif method == "random":
                rand_noise_b = torch.randn_like(raw_grad_s_b)
                if grad_normalization:
                    grad_s_b = rand_noise_b / (rand_noise_b.norm(dim=-1, keepdim=True) + 1e-12)
                else:
                    grad_s_b = rand_noise_b

            backward_diff = (emb_matrix - emb_matrix[s_ids[-1]])
            t1_b = -torch.sum(backward_diff * backward_diff, dim=-1) / (2 * eps_k) 
            t2_b = 0.5 * backward_diff @ grad_s_b
            logits_b = t1_b + t2_b
            scaled_logits_b = logits_b/temperature if temperature else logits
            backward_prob = torch.nn.functional.softmax(scaled_logits_b, dim=0)[s_ids[-2]]
        
            accept_prob = torch.minimum(torch.tensor(1), torch.exp(bw_log_joint - log_joint)*(backward_prob/forward_prob))
            accepted = torch.bernoulli(accept_prob)
            
            if accepted < 1:
                s_states = s_states[:-1]
                s_ids = s_ids[:-1]   
                mh_rejected = True
                if debug: 
                    print("rejected!")

        # ---------------------------------------------------------
        # STEP-WISE METRIC LOGGING
        # ---------------------------------------------------------
        if orig_ids is not None:
            current_id = s_ids[-1]
            if isinstance(current_id, torch.Tensor): 
                current_id = current_id.item()

            step_data = {
                "step": k,
                "token_id": current_id,
                "token_str": tokenizer.decode([current_id]),
                "mh_rejected": mh_rejected,
                "l2_distance": None,
                "kl_divergence": None,
                "entropy": entropy_t
            }

            with torch.no_grad():
                if emb_gt is not None:
                    emb_pred = emb_matrix[current_id]
                    l2_dist = torch.norm(emb_pred - emb_gt, p=2).item()
                    step_data["l2_distance"] = l2_dist

                if p_ref is not None:
                    pred_seq = orig_ids.clone()
                    pred_seq[0, position[0]] = current_id
                    
                    out_pred = model(pred_seq)
                    logits_pred = out_pred.logits[0, position[0], :]
                    log_p_pred = torch.log_softmax(logits_pred, dim=-1)
                    
                    kl_val = torch.nn.functional.kl_div(log_p_pred, p_ref, reduction='sum', log_target=False)
                    step_data["kl_divergence"] = kl_val.item()
            
            metrics_history.append(step_data)

    return s_ids, metrics_history, alpha_schedule

def langevin_infilling_multiple_positions_single_log(
    model,
    input_ids,
    mask_indices, # CHANGED: List of integers, e.g., [5, 10, 15]
    tokenizer,
    steps=30,
    temperature=None,
    oracle=False,
    orig_ids=None,
    alpha_grid=np.logspace(-4, -1, 50),
    method="policy",
    mh_sampling=False,
    debug=True
):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Create mask indices tensor for easy indexing
    mask_indices_t = torch.tensor(mask_indices, device=device)
    num_masks = len(mask_indices)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    emb_layer = model.get_input_embeddings()
    emb_matrix = emb_layer.weight.detach().to(device) # (V, D)
    vocab_size, emb_dim = emb_matrix.shape

    # Pre-compute Embedding Norms for efficient distance calculation
    # ||E||^2 term
    emb_matrix_sq_norm = torch.sum(emb_matrix ** 2, dim=1) # (V,)

    # Oracle Setup
    if oracle:
        assert orig_ids is not None
        gt_ids = orig_ids[0, mask_indices_t] # (M,)
        emb_gt = emb_matrix[gt_ids]          # (M, D)

    # --- Initialize s (Continuous Embeddings) ---
    with torch.no_grad():
        base_embs = emb_layer(input_ids) # (1, Seq_Len, D)
    
    # Extract initial embeddings at masked positions
    # s shape: (Num_Masks, Emb_Dim)
    s = base_embs[0, mask_indices_t, :].clone().detach().requires_grad_(True)
    
    # Current discrete token IDs at masked positions
    s_idx = input_ids[0, mask_indices_t].clone().detach() # (M,)

    # --- Helpers ---
    def joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids):
        # Same logic, just standardizing inputs
        outputs = model(inputs_embeds=inputs_embeds, return_dict=True)
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = target_ids[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        # Reshape to (Batch*Seq, Vocab)
        return -loss_fct(logits.view(-1, vocab_size), targets.view(-1))

    def linear_epsilon_schedule(k, total_steps=steps, eps0=1e-1, eps_min=1e-3):
        frac = k / float(total_steps)
        return max(eps0 * (1.0 - frac), eps_min)

    def compute_q_logprob_sum(token_embs, theta, grad, alpha):
        """
        Computes sum of unnormalized log q across all masks.
        """
        m = theta + 0.5 * alpha * grad
        diff = token_embs - m
        # Sum over Dim, then Sum over Masks
        dist2 = (diff * diff).sum() 
        return - dist2 / (2 * alpha)

    # --- History Tracking ---
    s_ids_history = [s_idx.clone()] # Store tensor of IDs
    metrics_history = []
    
    # --- Optimization Loop ---
    for k in range(steps):
        eps_k = linear_epsilon_schedule(k)
        
        # 1. Inject s into the sequence
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[0, mask_indices_t, :] = s # Scatter s into positions
        
        target_ids = input_ids.clone()
        target_ids[0, mask_indices_t] = s_idx

        # 2. Compute Joint Log Prob & Gradient
        log_joint = joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids)
        
        # grad_s shape: (M, D) - PyTorch handles gradients for the slice automatically
        grad_s = torch.autograd.grad(log_joint, s, retain_graph=False)[0]

        # 3. Method Variations
        if method == "grad_norm_preserved_random_dir":
            # Normalize per mask row
            grad_norms = grad_s.norm(dim=1, keepdim=True)
            rand_dir = torch.randn_like(grad_s)
            rand_dir = rand_dir / (rand_dir.norm(dim=1, keepdim=True) + 1e-12)
            grad_s = rand_dir * grad_norms
        elif method == "random":
            grad_s = torch.randn_like(grad_s)

        # 4. Oracle Alpha Selection (Jointly)
        if oracle:
            gt_logprobs = []
            for alpha in alpha_grid:
                # We want to maximize likelihood of *all* GT tokens jointly
                # simple approach: sum of log q for each token
                lp = compute_q_logprob_sum(emb_gt, s.detach(), grad_s, alpha)
                gt_logprobs.append(lp.item())
            
            best_idx = np.argmax(gt_logprobs)
            eps_k = alpha_grid[best_idx]
            if debug and k % 5 == 0:
                print(f"[Step {k}] Joint α* = {eps_k:.6f}")

        # 5. Proposal Step (Memory Efficient Vectorized)
        # We need logits for each mask: shape (M, V)
        # Term 1: -||E - s||^2 / 2eps
        # ||E - s||^2 = ||E||^2 + ||s||^2 - 2 s@E.T
        
        s_detached = s.detach()
        s_sq_norm = torch.sum(s_detached ** 2, dim=1, keepdim=True) # (M, 1)
        
        # Matrix Multiplication: (M, D) @ (D, V) -> (M, V)
        dot_prod = torch.matmul(s_detached, emb_matrix.T) 
        
        # dist_sq shape (M, V)
        dist_sq = emb_matrix_sq_norm.unsqueeze(0) + s_sq_norm - 2 * dot_prod
        t1 = -dist_sq / (2 * eps_k)

        # Term 2: 0.5 * (E - s) @ g 
        # = 0.5 * (E@g - s@g)
        # E@g is actually E @ g.T if we want (V, M), but we need (M, V)
        # So it is (g @ E.T)
        grad_dot_emb = torch.matmul(grad_s, emb_matrix.T) # (M, V)
        grad_dot_s = torch.sum(grad_s * s_detached, dim=1, keepdim=True) # (M, 1)
        
        t2 = 0.5 * (grad_dot_emb - grad_dot_s)

        # Final Logits (M, V)
        logits = t1 + t2
        
        scaled_logits = logits / temperature if temperature else logits
        probs = torch.softmax(scaled_logits, dim=1) # Softmax over vocab
        
        # Sample next tokens for all positions
        # torch.multinomial works on rows, so we sample 1 per row
        next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1) # (M,)
        
        # Entropy tracking (average over masks)
        entropy_t = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean().item()

        # Update State Variables
        # Create new embedding tensor for the next step
        s_next = emb_matrix[next_token_ids].clone().detach().requires_grad_(True)
        s_ids_next = next_token_ids

        # 6. Metropolis-Hastings (Joint)
        accepted_step = True
        mh_rejected = False
        
        if mh_sampling:
            # P(New | Old) = Product of probabilities of chosen tokens
            # Log P(New | Old) = Sum of log_softmax[chosen_indices]
            log_probs_fwd = torch.log_softmax(scaled_logits, dim=1)
            # Gather log probs of the specific tokens we selected
            log_fwd_prob = log_probs_fwd.gather(1, next_token_ids.unsqueeze(1)).sum()

            # --- Backward Step ---
            inputs_embeds_b = base_embs.clone().detach()
            inputs_embeds_b[0, mask_indices_t, :] = s_next
            target_ids_b = input_ids.clone()
            target_ids_b[0, mask_indices_t] = s_ids_next

            bw_log_joint = joint_log_prob_from_inputs_embeds(model, inputs_embeds_b, target_ids_b)
            grad_s_b = torch.autograd.grad(bw_log_joint, s_next, retain_graph=False)[0]

            # Backward Logits
            s_b_detached = s_next.detach()
            s_b_sq_norm = torch.sum(s_b_detached ** 2, dim=1, keepdim=True)
            dot_prod_b = torch.matmul(s_b_detached, emb_matrix.T)
            dist_sq_b = emb_matrix_sq_norm.unsqueeze(0) + s_b_sq_norm - 2 * dot_prod_b
            t1_b = -dist_sq_b / (2 * eps_k)
            
            grad_dot_emb_b = torch.matmul(grad_s_b, emb_matrix.T)
            grad_dot_s_b = torch.sum(grad_s_b * s_b_detached, dim=1, keepdim=True)
            t2_b = 0.5 * (grad_dot_emb_b - grad_dot_s_b)
            
            logits_b = t1_b + t2_b
            scaled_logits_b = logits_b / temperature if temperature else logits_b
            
            log_probs_bw = torch.log_softmax(scaled_logits_b, dim=1)
            # Gather log probs of the PREVIOUS tokens
            log_bw_prob = log_probs_bw.gather(1, s_idx.unsqueeze(1)).sum()

            # Acceptance Ratio (Log Scale)
            # log(A) = log(P_joint_new) - log(P_joint_old) + log(q_back) - log(q_fwd)
            log_alpha = (bw_log_joint - log_joint) + (log_bw_prob - log_fwd_prob)
            accept_prob = torch.exp(torch.minimum(torch.tensor(0.0).to(device), log_alpha))
            
            if torch.rand(1).to(device) > accept_prob:
                accepted_step = False
                mh_rejected = True
                if debug: print("MH Rejected")

        if accepted_step:
            s = s_next
            s_idx = s_ids_next
        
        s_ids_history.append(s_idx.clone())

        # --- Logging ---
        if oracle:
            # Average L2 over all masked tokens
            avg_l2 = torch.norm(s.detach() - emb_gt, dim=1).mean().item()
            
            metrics_history.append({
                "step": k,
                "token_ids": s_idx.tolist(),
                "mh_rejected": mh_rejected,
                "avg_l2_distance": avg_l2,
                "entropy": entropy_t
            })

    return s_ids_history, metrics_history


def langevin_infilling_multiple_positions(
    model,
    input_ids,
    mask_indices, # List[int]
    tokenizer,
    steps=30,
    temperature=None,
    oracle=False,
    orig_ids=None,
    alpha_grid=np.logspace(-4, -1, 50),
    method="policy",
    mh_sampling=False,
    grad_normalization=False,
    debug=True
):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    
    # Create mask indices tensor
    mask_indices_t = torch.tensor(mask_indices, device=device)
    num_masks = len(mask_indices)

    # 1. Setup Embeddings
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    emb_layer = model.get_input_embeddings()
    emb_matrix = emb_layer.weight.detach().to(device) # (V, D)
    emb_matrix_sq_norm = torch.sum(emb_matrix ** 2, dim=1) # (V,) Pre-calc for speed

    # 2. Oracle / Ground Truth Setup
    emb_gt = None
    p_refs = None
    valid_kl_indices = [] # Indices in the sequence where we can calc KL (i.e., not the last token)

    
    assert orig_ids is not None
    # Get GT embeddings for L2 calc
    gt_ids = orig_ids[0, mask_indices_t] # (M,)
    emb_gt = emb_matrix[gt_ids]          # (M, D)

    # Pre-compute Reference Probs (GT) for KL calc
    # We need the probability distribution of the token *immediately following* each mask
    # given the *original* context.
    with torch.no_grad():
        out_gt = model(orig_ids)
        # We want logits at position 'm' because they predict token 'm+1'
        
        ref_logits_list = []
        for m_idx in mask_indices:
            if m_idx < seq_len - 1:
                valid_kl_indices.append(m_idx)
                # Logits at m_idx predict m_idx+1
                ref_logits_list.append(out_gt.logits[0, m_idx, :])
        
        if len(valid_kl_indices) > 0:
            # Stack to (Num_Valid, V)
            ref_logits = torch.stack(ref_logits_list)
            p_refs = torch.softmax(ref_logits, dim=-1)

    # 3. Initialize s (Continuous Embeddings)
    with torch.no_grad():
        base_embs = emb_layer(input_ids) # (1, Seq, D)
    
    # Extract initial state at masked positions
    s = base_embs[0, mask_indices_t, :].clone().detach().requires_grad_(True)
    s_idx = input_ids[0, mask_indices_t].clone().detach() # (M,)

    # --- Helper: Joint Log Prob ---
    def joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids):
        outputs = model(inputs_embeds=inputs_embeds, return_dict=True)
        # Shift logits and targets for causal loss
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = target_ids[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        vocab_size = logits.shape[-1]
        return -loss_fct(logits.view(-1, vocab_size), targets.view(-1))
    
    def linear_epsilon_schedule(k, total_steps=steps, eps0=1e-1, eps_min=1e-3):
        frac = k / float(total_steps)
        return max(eps0 * (1.0 - frac), eps_min)

    def compute_q_logprob_sum(token_embs, theta, grad, alpha):
        m = theta + 0.5 * alpha * grad
        diff = token_embs - m
        dist2 = (diff * diff).sum() 
        return - dist2 / (2 * alpha)

    # --- Metrics History ---
    s_ids_history = [s_idx.clone()]
    metrics_history = []
    # Custom alpha schedule
    alpha_schedule = np.linspace(10.5, 0.1, steps)
    
    # =========================================================================
    # OPTIMIZATION LOOP
    # =========================================================================
    for k in range(steps):
        # eps_k = linear_epsilon_schedule(k)
        #overriding with custom alpha schedule
        eps_k = alpha_schedule[k]
        
        # A. Inject s into sequence
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[0, mask_indices_t, :] = s
        
        target_ids = input_ids.clone()
        target_ids[0, mask_indices_t] = s_idx

        # B. Gradients
        # Calculate gradient EXACTLY ONCE
        log_joint = joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids)
        raw_grad_s = torch.autograd.grad(log_joint, s, retain_graph=False)[0]

        # C. Method Variations
        if method == "policy":
            if grad_normalization:
                # Normalize along the embedding dimension (dim=1)
                grad_norm = raw_grad_s.norm(dim=1, keepdim=True) + 1e-12
                grad_s = raw_grad_s / grad_norm
            else:
                # Use raw gradient (original behavior)
                grad_s = raw_grad_s
                
        elif method == "grad_norm_preserved_random_dir":
            # Generate a unit-length random direction vector
            rand_dir = torch.randn_like(raw_grad_s)
            rand_dir_unit = rand_dir / (rand_dir.norm(dim=1, keepdim=True) + 1e-12)
            
            if grad_normalization:
                # Scale to 1.0 (because the gradient is "normalized")
                grad_s = rand_dir_unit
            else:
                # Scale to match the massive norm of the raw gradient
                grad_norms = raw_grad_s.norm(dim=1, keepdim=True)
                grad_s = rand_dir_unit * grad_norms

        elif method == "random":
            # Pure random walk
            rand_noise = torch.randn_like(raw_grad_s)
            if grad_normalization:
                # Force the random noise to also have a norm of 1.0 for a fair comparison!
                grad_s = rand_noise / (rand_noise.norm(dim=1, keepdim=True) + 1e-12)
            else:
                # Standard gaussian noise
                grad_s = rand_noise
            
        else:
            raise ValueError(f"Unknown method: {method}")

        # D. Oracle Alpha
        if oracle:
            gt_logprobs = []
            for alpha in alpha_grid:
                lp = compute_q_logprob_sum(emb_gt, s.detach(), grad_s, alpha)
                gt_logprobs.append(lp.item())
            best_idx = np.argmax(gt_logprobs)
            eps_k = alpha_grid[best_idx]

        # E. Proposal (Vectorized)
        s_detached = s.detach()
        s_sq_norm = torch.sum(s_detached ** 2, dim=1, keepdim=True)
        dot_prod = torch.matmul(s_detached, emb_matrix.T)
        dist_sq = emb_matrix_sq_norm.unsqueeze(0) + s_sq_norm - 2 * dot_prod
        t1 = -dist_sq / (2 * eps_k)

        grad_dot_emb = torch.matmul(grad_s, emb_matrix.T)
        grad_dot_s = torch.sum(grad_s * s_detached, dim=1, keepdim=True)
        t2 = 0.5 * (grad_dot_emb - grad_dot_s)

        logits = t1 + t2
        scaled_logits = logits / temperature if temperature else logits
        probs = torch.softmax(scaled_logits, dim=1)
        
        next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        entropy_t = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean().item()

        s_next = emb_matrix[next_token_ids].clone().detach().requires_grad_(True)
        s_ids_next = next_token_ids

        # F. Metropolis-Hastings
        mh_rejected = False
        accepted_step = True
        
        if mh_sampling:
            log_probs_fwd = torch.log_softmax(scaled_logits, dim=1)
            log_fwd_prob = log_probs_fwd.gather(1, next_token_ids.unsqueeze(1)).sum()

            inputs_embeds_b = base_embs.clone().detach()
            inputs_embeds_b[0, mask_indices_t, :] = s_next
            target_ids_b = input_ids.clone()
            target_ids_b[0, mask_indices_t] = s_ids_next

            bw_log_joint = joint_log_prob_from_inputs_embeds(model, inputs_embeds_b, target_ids_b)
            grad_s_b = torch.autograd.grad(bw_log_joint, s_next, retain_graph=False)[0]

            s_b_detached = s_next.detach()
            s_b_sq_norm = torch.sum(s_b_detached ** 2, dim=1, keepdim=True)
            dot_prod_b = torch.matmul(s_b_detached, emb_matrix.T)
            dist_sq_b = emb_matrix_sq_norm.unsqueeze(0) + s_b_sq_norm - 2 * dot_prod_b
            t1_b = -dist_sq_b / (2 * eps_k)
            
            grad_dot_emb_b = torch.matmul(grad_s_b, emb_matrix.T)
            grad_dot_s_b = torch.sum(grad_s_b * s_b_detached, dim=1, keepdim=True)
            t2_b = 0.5 * (grad_dot_emb_b - grad_dot_s_b)
            logits_b = t1_b + t2_b
            scaled_logits_b = logits_b / temperature if temperature else logits_b
            
            log_probs_bw = torch.log_softmax(scaled_logits_b, dim=1)
            log_bw_prob = log_probs_bw.gather(1, s_idx.unsqueeze(1)).sum()

            log_alpha = (bw_log_joint - log_joint) + (log_bw_prob - log_fwd_prob)
            accept_prob = torch.exp(torch.minimum(torch.tensor(0.0).to(device), log_alpha))
            
            if torch.rand(1).to(device) > accept_prob:
                accepted_step = False
                mh_rejected = True
                if debug: print(f"Step {k}: MH Rejected")

        if accepted_step:
            s = s_next
            s_idx = s_ids_next
        
        s_ids_history.append(s_idx.clone())

        # =====================================================================
        # LOGGING (L2 & KL)
        # =====================================================================
        if orig_ids is not None:
            # 1. Avg L2 Distance
            # Distance between current continuous 's' and GT embeddings
            current_l2_norm = torch.norm(s.detach() - emb_gt, dim=1) # (M,)
            avg_l2 = current_l2_norm.mean().item()

            # 2. Avg KL Divergence
            avg_kl = 0.0
            
            # Only calculate if we have valid positions (not last token)
            if len(valid_kl_indices) > 0 and p_refs is not None:
                # We must run a forward pass with the DISCRETE tokens we just picked
                # to see what they predict for the *next* token.
                temp_input = orig_ids.clone()
                temp_input[0, mask_indices_t] = s_idx # Plug in current discrete tokens
                
                with torch.no_grad():
                    out_pred = model(temp_input)
                    # Extract logits only at valid positions
                    logits_pred = out_pred.logits[0, valid_kl_indices, :] # (Num_Valid, V)
                    log_p_pred = torch.log_softmax(logits_pred, dim=-1)
                    
                    # Calculate KL batch-wise
                    # KL(Ref || Pred)
                    kl_val = torch.nn.functional.kl_div(
                        log_p_pred, 
                        p_refs, 
                        reduction='batchmean', 
                        log_target=False
                    )
                    avg_kl = kl_val.item()

            metrics_history.append({
                "step": k,
                "token_ids": s_idx.tolist(),
                "mh_rejected": mh_rejected,
                "avg_l2_distance": avg_l2,
                "avg_kl_divergence": avg_kl,
                "entropy": entropy_t
            })

    return s_ids_history, metrics_history