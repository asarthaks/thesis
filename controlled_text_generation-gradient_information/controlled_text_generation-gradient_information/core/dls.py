import torch
import numpy as np
from core.base_sampler import BaseLangevinSampler


class DiscreteLangevinSampler(BaseLangevinSampler):
    """
    Diagnostic build. Same two recorders as the patched CLS:

        sampler.mh_log   = []
        sampler.traj_log = []

    Both default to None, in which case this file is bit-identical to the
    original. Note in particular that the `else` branch of the MH block still
    avoids calling apply_method_variation on the backward gradient, so no extra
    torch.randn is drawn and the RNG stream stays aligned with the no-MH path.
    The diagnostics do not draw from the RNG at all.

    DLS is the control condition for Experiment 2. The prediction is that the
    acceptance rate here is healthy and largely independent of whether the
    proposal changed the token, because DLS never leaves the token manifold and
    therefore never evaluates the target across a discontinuity. The contrast
    between this acceptance rate and the CLS one is the measurement.
    """

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
            gt_logprobs = [self._compute_q_logprob_sum(emb_gt, s.detach(), grad_s, alpha).item() for alpha in self.alpha_grid]
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

        # ---------------- DIAGNOSTIC: EXPERIMENT 1, IN SITU ----------------
        # The t2 term IS the Taylor surrogate, up to the constant grad_dot_s:
        #     t2[v] = 0.5 * ( g . e(v) - g . s ) = 0.5 * g^T ( e(v) - s )
        # so we can record the proposal's own ranking here for free and compare it
        # against the true energy change measured offline by run_diagnostic.py.
        # We only record the rank the proposal assigned to the token it actually
        # chose, plus the entropy, which is cheap.
        if getattr(self, "proposal_log", None) is not None:
            with torch.no_grad():
                chosen_logit = scaled_logits.gather(1, next_token_ids.unsqueeze(1)).squeeze(1)
                rank = (scaled_logits > chosen_logit.unsqueeze(1)).sum(dim=1)
                self.proposal_log.append(dict(
                    seq_id=int(getattr(self, "_diag_seq_id", -1)),
                    step=int(k),
                    epsilon=float(eps_k),
                    entropy=float(entropy_t),
                    mean_rank_of_chosen=float(rank.float().mean().item()),
                    # how much of the proposal logit comes from the DISTANCE term (t1)
                    # versus the GRADIENT term (t2)? if t1 dominates, the proposal is
                    # essentially a distance-weighted random walk and the gradient is
                    # decorative.
                    t1_std=float(t1.std().item()),
                    t2_std=float(t2.std().item()),
                    t2_over_t1=float((t2.std() / (t1.std() + 1e-12)).item()),
                ))
        # -------------------------------------------------------------------

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

            rejected = bool((torch.rand(1).to(self.device) > accept_prob).item())

            # ---------------- DIAGNOSTIC: EXPERIMENT 2 ----------------
            if getattr(self, "mh_log", None) is not None:
                with torch.no_grad():
                    crossed_mask = (next_token_ids != s_idx)
                    self.mh_log.append(dict(
                        sampler="dls",
                        seq_id=int(getattr(self, "_diag_seq_id", -1)),
                        step=int(k),
                        epsilon=float(eps_k),
                        method=str(self.method),
                        grad_norm=bool(getattr(self, "grad_normalization", False)),
                        crossed=int(bool(crossed_mask.any().item())),
                        n_positions=int(s_idx.numel()),
                        n_crossed=int(crossed_mask.sum().item()),
                        accepted=int(not rejected),
                        log_alpha=float(log_alpha),
                        log_target_ratio=float(bw_log_joint - log_joint),
                        log_proposal_ratio=float(log_q_ratio) if isinstance(log_q_ratio, float) else float(log_q_ratio.item()),
                        log_q_back=float("nan"),
                        log_q_fwd=float("nan"),
                        step_norm=float((s_next.detach() - s.detach()).norm().item()),
                        drift_norm=float("nan"),
                        noise_norm=float("nan"),
                    ))
            # ---------------------------------------------------------

            if rejected:
                mh_rejected = True
                self._diag_record_traj(k, eps_k, s, s_idx, mh_rejected)
                return s, s_idx, mh_rejected, entropy_t

        self._diag_record_traj(k, eps_k, s_next, next_token_ids, mh_rejected)
        return s_next, next_token_ids, mh_rejected, entropy_t

    # -------------------- DIAGNOSTIC: EXPERIMENT 3 --------------------
    def _diag_record_traj(self, k, eps_k, s_state, idx_state, mh_rejected):
        if getattr(self, "traj_log", None) is None:
            return
        with torch.no_grad():
            sd = s_state.detach()
            d = torch.cdist(sd, self.emb_matrix)
            dmin, _ = d.min(dim=1)
            self.traj_log.append(dict(
                sampler="dls",
                seq_id=int(getattr(self, "_diag_seq_id", -1)),
                step=int(k),
                epsilon=float(eps_k),
                mh_rejected=int(bool(mh_rejected)),
                state=sd.float().cpu().numpy().copy(),
                token_ids=idx_state.detach().cpu().numpy().copy(),
                dist_to_manifold=dmin.float().cpu().numpy().copy(),
            ))
