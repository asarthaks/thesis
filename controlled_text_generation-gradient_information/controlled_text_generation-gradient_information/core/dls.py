import torch
import numpy as np
from core.base_sampler import BaseLangevinSampler

class DiscreteLangevinSampler(BaseLangevinSampler):
    
    def _compute_q_logprob_sum(self, token_embs, theta, grad, alpha):
        m = theta + 0.5 * alpha * grad
        diff = token_embs - m
        return - (diff * diff).sum() / (2 * alpha)

    def _step(self, k, eps_k, s, s_idx, base_embs, input_ids, mask_indices_t, emb_gt):
        # 1. Gradients & Method Variation
        raw_grad_s, log_joint = self.get_gradient_and_log_joint(s, s_idx, base_embs, input_ids, mask_indices_t)
        grad_s = self.apply_method_variation(raw_grad_s)

        # 2. Oracle Alpha Search
        if self.oracle and emb_gt is not None:
            gt_logprobs =[self._compute_q_logprob_sum(emb_gt, s.detach(), grad_s, alpha).item() for alpha in self.alpha_grid]
            eps_k = self.alpha_grid[np.argmax(gt_logprobs)]

        # 3. Compute Logits & Proposal (Vectorized for M masks)
        s_detached = s.detach()
        s_sq_norm = torch.sum(s_detached ** 2, dim=1, keepdim=True)
        dot_prod = torch.matmul(s_detached, self.emb_matrix.T)
        dist_sq = self.emb_matrix_sq_norm.unsqueeze(0) + s_sq_norm - 2 * dot_prod
        t1 = -dist_sq / (2 * eps_k)

        grad_dot_emb = torch.matmul(grad_s, self.emb_matrix.T)
        grad_dot_s = torch.sum(grad_s * s_detached, dim=1, keepdim=True)
        t2 = 0.5 * (grad_dot_emb - grad_dot_s)

        scaled_logits = (t1 + t2) / self.temperature if self.temperature else (t1 + t2)
        probs = torch.softmax(scaled_logits, dim=1)
        
        next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        entropy_t = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean().item()

        s_next = self.emb_matrix[next_token_ids].clone().detach().requires_grad_(True)

        # 4. Metropolis-Hastings
        mh_rejected = False
        if self.mh_sampling:
            if self.method == "policy":
                # Detailed balance requires the reverse proposal to be evaluated under the
                # SAME kernel that produced the forward move, i.e. the method-varied
                # (here normalized when grad_normalization is on) gradient at s_next.
                # The previous legacy DLS used the raw backward gradient here, which broke
                # detailed balance whenever the forward proposal was normalized.
                log_fwd_prob = torch.log_softmax(scaled_logits, dim=1).gather(1, next_token_ids.unsqueeze(1)).sum()

                raw_grad_s_b, bw_log_joint = self.get_gradient_and_log_joint(s_next, next_token_ids, base_embs, input_ids, mask_indices_t)
                grad_s_b = self.apply_method_variation(raw_grad_s_b)

                s_b_detached = s_next.detach()
                dist_sq_b = self.emb_matrix_sq_norm.unsqueeze(0) + torch.sum(s_b_detached ** 2, dim=1, keepdim=True) - 2 * torch.matmul(s_b_detached, self.emb_matrix.T)
                t1_b = -dist_sq_b / (2 * eps_k)
                t2_b = 0.5 * (torch.matmul(grad_s_b, self.emb_matrix.T) - torch.sum(grad_s_b * s_b_detached, dim=1, keepdim=True))

                scaled_logits_b = (t1_b + t2_b) / self.temperature if self.temperature else (t1_b + t2_b)
                log_bw_prob = torch.log_softmax(scaled_logits_b, dim=1).gather(1, s_idx.unsqueeze(1)).sum()
                log_q_ratio = log_bw_prob - log_fwd_prob
            else:
                # Random-direction baselines are symmetric random walks: q(x|x') = q(x'|x),
                # so the proposal ratio cancels. This matches the CLS treatment and thesis
                # Sec 4.3. Crucially we do NOT call apply_method_variation on the backward
                # gradient here, so no extra torch.randn is drawn and the RNG stream stays
                # aligned with the no-MH path. Only the likelihood term is needed.
                _, bw_log_joint = self.get_gradient_and_log_joint(s_next, next_token_ids, base_embs, input_ids, mask_indices_t)
                log_q_ratio = 0.0

            log_alpha = (bw_log_joint - log_joint) + log_q_ratio
            accept_prob = torch.exp(torch.minimum(torch.tensor(0.0).to(self.device), log_alpha))

            if torch.rand(1).to(self.device) > accept_prob:
                mh_rejected = True
                return s, s_idx, mh_rejected, entropy_t

        return s_next, next_token_ids, mh_rejected, entropy_t