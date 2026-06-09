import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Core Helpers
# -----------------------------------------------------------------------------
def get_embedding_matrix(model):
    emb_layer = model.get_input_embeddings()
    return emb_layer.weight.detach()

def project_to_vocab_by_l2(s: torch.FloatTensor, emb_matrix: torch.FloatTensor):
    D = emb_matrix.size(1)
    assert s.size(-1) == D, f"Embedding dim mismatch: s has {s.size(-1)}, emb_matrix has {D}"

    original_shape = s.shape[:-1] 
    s_flat = s.view(-1, D)         

    diff = s_flat[:, None, :] - emb_matrix[None, :, :]  
    dist2 = torch.sum(diff ** 2, dim=-1)                
    top_idx = dist2.argmin(dim=-1)                      

    return top_idx.view(*original_shape)

def joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids, attention_mask=None):
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    logits_next = logits[:, :-1, :].contiguous()
    target_next = target_ids[:, 1:].contiguous()
    B, Lm1, V = logits_next.shape
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    nll_sum = loss_fct(logits_next.view(-1, V), target_next.view(-1))
    return -nll_sum

def logq_gaussian(x, mean, eps):
    diff = (x - mean)
    D = diff.numel()
    quad = (diff.pow(2).sum()) / (2.0 * eps)
    const = 0.5 * D * math.log(2.0 * math.pi * eps)
    return -(quad + const)

# -----------------------------------------------------------------------------
# SINGLE POSITION CLS (For evaluate_cls_dataset.py)
# -----------------------------------------------------------------------------
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
    method="policy",
    mh_sampling=False,
    grad_normalization=True,
    debug=True,
    mh_burn_in=0,
    noise_scale=0.01
):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    emb_layer = model.get_input_embeddings()
    emb_matrix = emb_layer.weight.detach().to(device)

    # --- Setup Metrics (Decoupled from Oracle) ---
    emb_gt = None
    p_ref = None
    if orig_ids is not None: 
        gt_id = orig_ids[0, position[0]].item()
        emb_gt = emb_matrix[gt_id]
        
        target_idx = position[0]
        if target_idx < input_ids.shape[1] - 1:
            with torch.no_grad():
                out_ref = model(orig_ids)
                p_ref = torch.softmax(out_ref.logits[0, target_idx, :], dim=-1)

    # Initialize continuous token s
    with torch.no_grad():
        base_embs = emb_layer(input_ids)

    s = base_embs[:, position[0]:position[1], :]
    s = s.clone().detach().requires_grad_(True)
    
    # Internal Helpers
    def log_pi(s_val):
        with torch.no_grad():
            proj_idx = project_to_vocab_by_l2(s_val.detach(), emb_matrix)
            proj_token_id = proj_idx[0].to(torch.int64)

        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[:, position[0]:position[1]] = s_val
        target_ids = input_ids.clone()
        target_ids[:, position[0]:position[1]] = proj_token_id

        return joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids, attention_mask)
    
    def grad_log_pi(s_val):
        with torch.no_grad():
            proj_idx = project_to_vocab_by_l2(s_val.detach(), emb_matrix)
            proj_token_id = proj_idx[0].to(torch.int64)

        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[:, position[0]:position[1]] = s_val
        target_ids = input_ids.clone()
        target_ids[:, position[0]:position[1]] = proj_token_id

        log_joint = joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids, attention_mask)
        grad = torch.autograd.grad(log_joint, s_val, retain_graph=False)[0]
        return grad, log_joint

    def linear_epsilon_schedule(k, total_steps=steps, eps0=1e-1, eps_min=1e-3):
        frac = k / float(total_steps)
        return max(eps0 * (1.0 - frac), eps_min)

    # Precompute current log π(s) once
    with torch.no_grad():
        cur_logpi = log_pi(s)

    s_ids_history = []
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
        
        # 1. Gradient Calculation
        raw_grad_s, log_joint = grad_log_pi(s)
        
        # 2. Method Variations & Gradient Normalization
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
            raise ValueError(f"Unknown method: {method}")

        # 3. Oracle Alpha Selection
        if oracle and emb_gt is not None:
            gt_logprobs =[]
            for alpha in alpha_grid:
                interim_alpha = s.detach() + 0.5 * alpha * grad_s
                idx_alpha = project_to_vocab_by_l2(interim_alpha, emb_matrix)
                m_alpha = 0.5 * (interim_alpha + emb_matrix[idx_alpha])
                
                lp = logq_gaussian(emb_gt.unsqueeze(0), m_alpha, alpha * noise_scale)
                gt_logprobs.append(lp.item())
                
            best_idx = np.argmax(gt_logprobs)
            eps_k = alpha_grid[best_idx]

        # 4. Continuous Proposal
        noise = torch.randn_like(s) * math.sqrt(eps_k * noise_scale)
        interim = s + 0.5 * eps_k * grad_s
        interim_proj = emb_matrix[project_to_vocab_by_l2(interim, emb_matrix)]
        
        m_s = 0.5 * (interim + interim_proj)
        s_u = (m_s + noise).detach()
        s_prop = s_u.detach().clone().requires_grad_(True)

        # 5. Metropolis-Hastings (Mathematically Corrected Block)
        mh_rejected = False
        if mh_sampling and k >= mh_burn_in:
            with torch.no_grad():
                # Evaluate target density at actual proposed state
                prop_logpi = log_pi(s_prop)

            # Reverse proposal calculations
            g_prop, _ = grad_log_pi(s_prop)
            
            if method == "policy":
                if grad_normalization:
                    g_prop = g_prop / (g_prop.norm(dim=-1, keepdim=True) + 1e-12)

                interim_prop = s_prop + 0.5 * eps_k * g_prop
                interim_prop_proj = emb_matrix[project_to_vocab_by_l2(interim_prop, emb_matrix)]
                m_prop = 0.5 * (interim_prop + interim_prop_proj)

                log_q_back = logq_gaussian(s, m_prop, eps_k * noise_scale)
                log_q_fwd = logq_gaussian(s_prop, m_s, eps_k * noise_scale)
            else:
                # Random walk proposals are symmetric, so q(s|s') and q(s'|s) cancel out
                log_q_back = 0.0
                log_q_fwd = 0.0

            log_alpha = (prop_logpi - cur_logpi) + (log_q_back - log_q_fwd)
            accept = (torch.log(torch.rand((), device=device)) < log_alpha)

            if accept.item():
                s = s_prop
                cur_logpi = prop_logpi.detach()
            else:
                s = s.detach().clone().requires_grad_(True) 
                mh_rejected = True
        else:
            s = s_prop
            with torch.no_grad():
                cur_logpi = log_pi(s_prop).detach()

        # 6. Logging & Metrics
        with torch.no_grad():
            current_idx = project_to_vocab_by_l2(s, emb_matrix)[0].item()
            s_ids_history.append(current_idx)
            
            step_data = {
                "step": k,
                "token_id": current_idx,
                "mh_rejected": mh_rejected,
                "l2_distance": None,
                "kl_divergence": None,
                "entropy": 0.0
            }

            if orig_ids is not None:
                if emb_gt is not None:
                    emb_pred = emb_matrix[current_idx]
                    step_data["l2_distance"] = torch.norm(emb_pred - emb_gt, p=2).item()

                if p_ref is not None:
                    pred_seq = orig_ids.clone()
                    pred_seq[0, position[0]] = current_idx
                    out_pred = model(pred_seq)
                    log_p_pred = torch.log_softmax(out_pred.logits[0, position[0], :], dim=-1)
                    kl_val = torch.nn.functional.kl_div(log_p_pred, p_ref, reduction='sum', log_target=False)
                    step_data["kl_divergence"] = kl_val.item()

            metrics_history.append(step_data)

    return s_ids_history, metrics_history


# -----------------------------------------------------------------------------
# MULTIPLE POSITIONS CLS (For evaluate_cls_multi.py)
# -----------------------------------------------------------------------------
def langevin_infilling_multiple_positions(
    model,
    input_ids,
    mask_indices,
    tokenizer,
    steps=30,
    temperature=None,
    oracle=False,
    orig_ids=None,
    alpha_grid=np.logspace(-4, -1, 50),
    method="policy",
    mh_sampling=False,
    grad_normalization=True,
    debug=True,
    mh_burn_in=0,
    noise_scale=0.01
):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    
    mask_indices_t = torch.tensor(mask_indices, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    emb_layer = model.get_input_embeddings()
    emb_matrix = emb_layer.weight.detach().to(device)

    # --- Setup Metrics (Decoupled from Oracle) ---
    emb_gt = None
    p_refs = None
    valid_kl_indices =[]

    if orig_ids is not None: 
        gt_ids = orig_ids[0, mask_indices_t] 
        emb_gt = emb_matrix[gt_ids]          

        with torch.no_grad():
            out_gt = model(orig_ids)
            ref_logits_list =[]
            for m_idx in mask_indices:
                if m_idx < seq_len - 1:
                    valid_kl_indices.append(m_idx)
                    ref_logits_list.append(out_gt.logits[0, m_idx, :])
            
            if len(valid_kl_indices) > 0:
                p_refs = torch.softmax(torch.stack(ref_logits_list), dim=-1)

    with torch.no_grad():
        base_embs = emb_layer(input_ids) 
    
    # Initialize s: (Num_Masks, Emb_Dim)
    s = base_embs[0, mask_indices_t, :].clone().detach().requires_grad_(True)
    
    def log_pi(s_val):
        with torch.no_grad():
            proj_idx = project_to_vocab_by_l2(s_val.detach(), emb_matrix)
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[0, mask_indices_t, :] = s_val
        target_ids = input_ids.clone()
        target_ids[0, mask_indices_t] = proj_idx
        return joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids, attention_mask)
    
    def grad_log_pi(s_val):
        with torch.no_grad():
            proj_idx = project_to_vocab_by_l2(s_val.detach(), emb_matrix)
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[0, mask_indices_t, :] = s_val
        target_ids = input_ids.clone()
        target_ids[0, mask_indices_t] = proj_idx
        
        log_joint = joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids, attention_mask)
        grad = torch.autograd.grad(log_joint, s_val, retain_graph=False)[0]
        return grad, log_joint

    def linear_epsilon_schedule(k, total_steps=steps, eps0=1e-1, eps_min=1e-3):
        frac = k / float(total_steps)
        return max(eps0 * (1.0 - frac), eps_min)

    with torch.no_grad():
        cur_logpi = log_pi(s)

    s_ids_history =[]
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
        
        raw_grad_s, log_joint = grad_log_pi(s)

        # 2. Method Variations & Gradient Normalization
        if method == "policy":
            if grad_normalization:
                # Dim 1 because shape is (Num_Masks, Emb_Dim)
                grad_norm = raw_grad_s.norm(dim=1, keepdim=True) + 1e-12
                grad_s = raw_grad_s / grad_norm
            else:
                grad_s = raw_grad_s
        elif method == "grad_norm_preserved_random_dir":
            rand_dir = torch.randn_like(raw_grad_s)
            rand_dir_unit = rand_dir / (rand_dir.norm(dim=1, keepdim=True) + 1e-12)
            if grad_normalization:
                grad_s = rand_dir_unit
            else:
                grad_norms = raw_grad_s.norm(dim=1, keepdim=True)
                grad_s = rand_dir_unit * grad_norms
        elif method == "random":
            rand_noise = torch.randn_like(raw_grad_s)
            if grad_normalization:
                grad_s = rand_noise / (rand_noise.norm(dim=1, keepdim=True) + 1e-12)
            else:
                grad_s = rand_noise
        else:
            raise ValueError(f"Unknown method: {method}")

        # 3. Oracle Alpha
        if oracle and emb_gt is not None:
            gt_logprobs =[]
            for alpha in alpha_grid:
                interim_alpha = s.detach() + 0.5 * alpha * grad_s
                idx_alpha = project_to_vocab_by_l2(interim_alpha, emb_matrix)
                m_alpha = 0.5 * (interim_alpha + emb_matrix[idx_alpha])
                
                lp = logq_gaussian(emb_gt, m_alpha, alpha * noise_scale)
                gt_logprobs.append(lp.item())
            best_idx = np.argmax(gt_logprobs)
            eps_k = alpha_grid[best_idx]

        # 4. Continuous Proposal
        noise = torch.randn_like(s) * math.sqrt(eps_k * noise_scale)
        interim = s + 0.5 * eps_k * grad_s
        interim_proj = emb_matrix[project_to_vocab_by_l2(interim, emb_matrix)]
        
        m_s = 0.5 * (interim + interim_proj)
        s_u = (m_s + noise).detach()
        s_prop = s_u.detach().clone().requires_grad_(True)

        # 5. Metropolis-Hastings (Mathematically Corrected Block)
        mh_rejected = False
        if mh_sampling and k >= mh_burn_in:
            with torch.no_grad():
                prop_logpi = log_pi(s_prop)

            g_prop, _ = grad_log_pi(s_prop)
            
            if method == "policy":
                if grad_normalization:
                    g_prop = g_prop / (g_prop.norm(dim=1, keepdim=True) + 1e-12)

                interim_prop = s_prop + 0.5 * eps_k * g_prop
                interim_prop_proj = emb_matrix[project_to_vocab_by_l2(interim_prop, emb_matrix)]
                m_prop = 0.5 * (interim_prop + interim_prop_proj)

                log_q_back = logq_gaussian(s, m_prop, eps_k * noise_scale)
                log_q_fwd = logq_gaussian(s_prop, m_s, eps_k * noise_scale)
            else:
                # Cancel out for symmetric proposals
                log_q_back = 0.0
                log_q_fwd = 0.0

            log_alpha = (prop_logpi - cur_logpi) + (log_q_back - log_q_fwd)
            accept = (torch.log(torch.rand((), device=device)) < log_alpha)

            if accept.item():
                s = s_prop
                cur_logpi = prop_logpi.detach()
            else:
                s = s.detach().clone().requires_grad_(True)
                mh_rejected = True
        else:
            s = s_prop
            with torch.no_grad():
                cur_logpi = log_pi(s_prop).detach()

        # 6. Logging & Metrics
        with torch.no_grad():
            current_ids = project_to_vocab_by_l2(s, emb_matrix)
            s_ids_history.append(current_ids.clone())
            
            avg_l2 = 0.0
            avg_kl = 0.0
            
            if orig_ids is not None:
                if emb_gt is not None:
                    emb_preds = emb_matrix[current_ids]
                    avg_l2 = torch.norm(emb_preds - emb_gt, dim=1).mean().item()

                if len(valid_kl_indices) > 0 and p_refs is not None:
                    temp_input = orig_ids.clone()
                    temp_input[0, mask_indices_t] = current_ids
                    
                    out_pred = model(temp_input)
                    logits_pred = out_pred.logits[0, valid_kl_indices, :]
                    log_p_pred = torch.log_softmax(logits_pred, dim=-1)
                    
                    kl_val = torch.nn.functional.kl_div(log_p_pred, p_refs, reduction='batchmean', log_target=False)
                    avg_kl = kl_val.item()

            metrics_history.append({
                "step": k,
                "token_ids": current_ids.tolist(),
                "mh_rejected": mh_rejected,
                "avg_l2_distance": avg_l2,
                "avg_kl_divergence": avg_kl,
                "entropy": 0.0 
            })

    return s_ids_history, metrics_history