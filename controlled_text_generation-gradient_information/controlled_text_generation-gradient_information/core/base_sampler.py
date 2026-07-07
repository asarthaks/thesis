import torch
import math
import numpy as np
from core.prep import get_embedding_matrix, project_to_vocab_by_l2, joint_log_prob_from_inputs_embeds

class BaseLangevinSampler:
    def __init__(self, model, tokenizer, steps=50, temperature=5.0, oracle=False, 
                 alpha_grid=np.logspace(-2, 2, 50), method="policy", mh_sampling=False, 
                 grad_normalization=True, debug=False, noise_scale=0.01,
                 epsilon_schedule=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.steps = steps
        self.temperature = temperature
        self.oracle = oracle
        self.alpha_grid = alpha_grid
        self.method = method
        self.mh_sampling = mh_sampling
        self.grad_normalization = grad_normalization
        self.debug = debug
        self.noise_scale = noise_scale
        # Injectable step-size schedule. Pass a length-`steps` array/list (indexed by k)
        # or a callable k -> eps. When None we fall back to the built-in linear schedule.
        # This is what lets GPT-2 Large use linspace(10.5, 0.1) while Llama uses its own
        # scale, instead of the fixed 0.1 -> 0.001 default that flatlines on GPT-2.
        self.epsilon_schedule = epsilon_schedule
        
        self.emb_matrix = get_embedding_matrix(model).to(self.device)
        self.emb_matrix_sq_norm = torch.sum(self.emb_matrix ** 2, dim=1) # Pre-calc for efficiency

    def linear_epsilon_schedule(self, k, eps0=10.5, eps_min=1e-1):
        frac = k / float(self.steps)
        return max(eps0 * (1.0 - frac), eps_min)

    def get_gradient_and_log_joint(self, s, s_idx, base_embs, input_ids, mask_indices_t):
        """Standardized gradient extraction for (M, D) tensor."""
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[0, mask_indices_t, :] = s
        target_ids = input_ids.clone()
        target_ids[0, mask_indices_t] = s_idx

        log_joint = joint_log_prob_from_inputs_embeds(self.model, inputs_embeds, target_ids)
        grad_s = torch.autograd.grad(log_joint, s, retain_graph=False)[0]
        return grad_s, log_joint

    def apply_method_variation(self, raw_grad_s):
        """Applies normalization and direction ablations cleanly to (M, D) tensor."""
        if self.method == "policy":
            if self.grad_normalization:
                grad_norm = raw_grad_s.norm(dim=1, keepdim=True) + 1e-12
                return raw_grad_s / grad_norm
            return raw_grad_s
                
        elif self.method == "grad_norm_preserved_random_dir":
            rand_dir = torch.randn_like(raw_grad_s)
            rand_dir_unit = rand_dir / (rand_dir.norm(dim=1, keepdim=True) + 1e-12)
            if self.grad_normalization:
                return rand_dir_unit
            else:
                grad_norms = raw_grad_s.norm(dim=1, keepdim=True)
                return rand_dir_unit * grad_norms

        elif self.method == "random":
            rand_noise = torch.randn_like(raw_grad_s)
            if self.grad_normalization:
                return rand_noise / (rand_noise.norm(dim=1, keepdim=True) + 1e-12)
            return rand_noise
            
        raise ValueError(f"Unknown method: {self.method}")

    def optimize(self, input_ids, mask_indices, orig_ids=None):
        """The Universal Optimization Loop"""
        seq_len = input_ids.shape[1]
        mask_indices_t = torch.tensor(mask_indices, device=self.device)
        
        # Ground Truth & KL Reference setup
        emb_gt, p_refs, valid_kl_indices = None, None,[]
        if orig_ids is not None:
            emb_gt = self.emb_matrix[orig_ids[0, mask_indices_t]]
            with torch.no_grad():
                out_gt = self.model(orig_ids)
                ref_logits_list =[]
                for m_idx in mask_indices:
                    if m_idx < seq_len - 1:
                        valid_kl_indices.append(m_idx)
                        ref_logits_list.append(out_gt.logits[0, m_idx, :])
                if valid_kl_indices:
                    p_refs = torch.softmax(torch.stack(ref_logits_list), dim=-1)

        # Base states
        with torch.no_grad():
            base_embs = self.model.get_input_embeddings()(input_ids)
            
        s = base_embs[0, mask_indices_t, :].clone().detach().requires_grad_(True)
        s_idx = input_ids[0, mask_indices_t].clone().detach()

        s_ids_history =[s_idx.clone()]
        metrics_history =[]

        for k in range(self.steps):
            if self.epsilon_schedule is None:
                eps_k = self.linear_epsilon_schedule(k)
            elif callable(self.epsilon_schedule):
                eps_k = self.epsilon_schedule(k)
            else:
                eps_k = float(self.epsilon_schedule[k])
            
            # Delegate exactly how to step to CLS or DLS
            s, s_idx, mh_rejected, entropy_t = self._step(
                k, eps_k, s, s_idx, base_embs, input_ids, mask_indices_t, emb_gt
            )
            
            s_ids_history.append(s_idx.clone())

            # Log Metrics universally
            avg_l2, avg_kl = 0.0, 0.0
            if orig_ids is not None:
                emb_preds = self.emb_matrix[s_idx]
                avg_l2 = torch.norm(emb_preds - emb_gt, dim=1).mean().item()
                if valid_kl_indices and p_refs is not None:
                    temp_input = orig_ids.clone()
                    temp_input[0, mask_indices_t] = s_idx 
                    with torch.no_grad():
                        out_pred = self.model(temp_input)
                        log_p_pred = torch.log_softmax(out_pred.logits[0, valid_kl_indices, :], dim=-1)
                        # Universally standardized to batchmean
                        avg_kl = torch.nn.functional.kl_div(log_p_pred, p_refs, reduction='batchmean', log_target=False).item()

            metrics_history.append({
                "step": k,
                "token_ids": s_idx.tolist(),
                "mh_rejected": mh_rejected,
                "avg_l2_distance": avg_l2,
                "avg_kl_divergence": avg_kl,
                "entropy": entropy_t
            })

        return s_ids_history, metrics_history

    def _step(self, k, eps_k, s, s_idx, base_embs, input_ids, mask_indices_t, emb_gt):
        raise NotImplementedError("Subclasses must implement _step")