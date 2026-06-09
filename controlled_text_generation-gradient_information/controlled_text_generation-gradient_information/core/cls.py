import torch
import math
import numpy as np
from core.base_sampler import BaseLangevinSampler
from core.prep import project_to_vocab_by_l2

class ContinuousLangevinSampler(BaseLangevinSampler):

    def _logq_gaussian(self, x, mean, eps):
        diff = (x - mean)
        return -(diff.pow(2).sum() / (2.0 * eps) + 0.5 * diff.numel() * math.log(2.0 * math.pi * eps))

    def _step(self, k, eps_k, s, s_idx, base_embs, input_ids, mask_indices_t, emb_gt):
        # 1. Gradients & Method Variation
        raw_grad_s, log_joint = self.get_gradient_and_log_joint(s, s_idx, base_embs, input_ids, mask_indices_t)
        grad_s = self.apply_method_variation(raw_grad_s)

        # 2. Oracle Alpha Search
        if self.oracle and emb_gt is not None:
            gt_logprobs =[]
            for alpha in self.alpha_grid:
                interim = s.detach() + 0.5 * alpha * grad_s
                idx_alpha = project_to_vocab_by_l2(interim, self.emb_matrix)
                m_alpha = 0.5 * (interim + self.emb_matrix[idx_alpha])
                gt_logprobs.append(self._logq_gaussian(emb_gt, m_alpha, alpha * self.noise_scale).item())
            eps_k = self.alpha_grid[np.argmax(gt_logprobs)]

        # 3. Continuous Proposal
        noise = torch.randn_like(s) * math.sqrt(eps_k * self.noise_scale)
        interim = s + 0.5 * eps_k * grad_s
        interim_proj = self.emb_matrix[project_to_vocab_by_l2(interim, self.emb_matrix)]
        
        m_s = 0.5 * (interim + interim_proj)
        s_prop = (m_s + noise).detach().clone().requires_grad_(True)
        s_idx_prop = project_to_vocab_by_l2(s_prop.detach(), self.emb_matrix)

        # 4. Metropolis-Hastings
        mh_rejected = False
        if self.mh_sampling:
            raw_grad_s_b, bw_log_joint = self.get_gradient_and_log_joint(s_prop, s_idx_prop, base_embs, input_ids, mask_indices_t)
            
            if self.method == "policy":
                grad_s_b = self.apply_method_variation(raw_grad_s_b)
                interim_b = s_prop + 0.5 * eps_k * grad_s_b
                m_prop = 0.5 * (interim_b + self.emb_matrix[project_to_vocab_by_l2(interim_b, self.emb_matrix)])
                
                log_q_back = self._logq_gaussian(s, m_prop, eps_k * self.noise_scale)
                log_q_fwd = self._logq_gaussian(s_prop, m_s, eps_k * self.noise_scale)
            else:
                log_q_back, log_q_fwd = 0.0, 0.0 # Symmetric cancellation

            log_alpha = (bw_log_joint - log_joint) + (log_q_back - log_q_fwd)
            if torch.log(torch.rand((), device=self.device)) > log_alpha:
                mh_rejected = True
                return s.detach().clone().requires_grad_(True), s_idx, mh_rejected, 0.0

        return s_prop, s_idx_prop, mh_rejected, 0.0