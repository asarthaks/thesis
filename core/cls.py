import torch
import math
import numpy as np
from core.base_sampler import BaseLangevinSampler
from core.prep import project_to_vocab_by_l2


class ContinuousLangevinSampler(BaseLangevinSampler):
    """
    Diagnostic build.

    Two optional recorders. Both are None by default, in which case this file is
    bit-identical to the original: no extra tensors are allocated, no extra RNG is
    drawn, and every branch that runs in production runs unchanged.

        sampler.mh_log   = []     # one dict per MH decision
        sampler.traj_log = []     # one dict per step, for the embedding trajectory

    Set sampler._diag_seq_id before each sequence so the records can be grouped.

    Why the mh_log matters
    ----------------------
    MALA is only valid when the drift term grad log pi is Lipschitz continuous
    (Roberts and Tweedie, 1996). Here the target density is defined through a
    nearest-neighbour projection into the vocabulary, so grad log pi is
    discontinuous at every Voronoi boundary. The consequence is visible in the
    acceptance ratio: when a proposal crosses a boundary, m_prop is computed from
    a gradient evaluated on the far side of that boundary and therefore lands
    somewhere unrelated, so s falls deep in the tail of the reverse Gaussian and
    log_q_back diverges to minus infinity.

    We log log_q_back and log_q_fwd separately from the target ratio so that the
    rejection can be attributed to the proposal term rather than to the target
    term. The prediction is that the target term is often POSITIVE for a boundary
    crossing (the move genuinely improves the sequence) while the proposal term is
    catastrophically negative. If that holds, MH is rejecting good moves for
    purely kinematic reasons, and the set of accepted moves and the set of useful
    moves are disjoint.
    """

    def _logq_gaussian(self, x, mean, eps):
        diff = (x - mean)
        return -(diff.pow(2).sum() / (2.0 * eps) + 0.5 * diff.numel() * math.log(2.0 * math.pi * eps))

    def _step(self, k, eps_k, s, s_idx, base_embs, input_ids, mask_indices_t, emb_gt):
        # 1. Gradients & Method Variation
        raw_grad_s, log_joint = self.get_gradient_and_log_joint(s, s_idx, base_embs, input_ids, mask_indices_t)
        grad_s = self.apply_method_variation(raw_grad_s)

        # 2. Oracle Alpha Search
        if self.oracle and emb_gt is not None:
            gt_logprobs = []
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
                log_q_back, log_q_fwd = 0.0, 0.0  # Symmetric cancellation

            log_alpha = (bw_log_joint - log_joint) + (log_q_back - log_q_fwd)
            rejected = bool(torch.log(torch.rand((), device=self.device)) > log_alpha)

            # ---------------- DIAGNOSTIC: EXPERIMENT 2 ----------------
            if getattr(self, "mh_log", None) is not None:
                with torch.no_grad():
                    crossed_mask = (s_idx_prop != s_idx)
                    self.mh_log.append(dict(
                        sampler="cls",
                        seq_id=int(getattr(self, "_diag_seq_id", -1)),
                        step=int(k),
                        epsilon=float(eps_k),
                        method=str(self.method),
                        grad_norm=bool(getattr(self, "grad_normalization", False)),
                        # did the proposal move into a different Voronoi cell?
                        crossed=int(bool(crossed_mask.any().item())),
                        n_positions=int(s_idx.numel()),
                        n_crossed=int(crossed_mask.sum().item()),
                        accepted=int(not rejected),
                        log_alpha=float(log_alpha),
                        # the decomposition. this is the whole point.
                        log_target_ratio=float(bw_log_joint - log_joint),
                        log_proposal_ratio=float(log_q_back - log_q_fwd),
                        log_q_back=float(log_q_back),
                        log_q_fwd=float(log_q_fwd),
                        step_norm=float((s_prop.detach() - s.detach()).norm().item()),
                        drift_norm=float((0.5 * eps_k * grad_s).norm().item()),
                        noise_norm=float(noise.norm().item()),
                    ))
            # ---------------------------------------------------------

            if rejected:
                mh_rejected = True
                s_out = s.detach().clone().requires_grad_(True)
                self._diag_record_traj(k, eps_k, s_out, s_idx, mh_rejected)
                return s_out, s_idx, mh_rejected, 0.0

        self._diag_record_traj(k, eps_k, s_prop, s_idx_prop, mh_rejected)
        return s_prop, s_idx_prop, mh_rejected, 0.0

    # -------------------- DIAGNOSTIC: EXPERIMENT 3 --------------------
    def _diag_record_traj(self, k, eps_k, s_state, idx_state, mh_rejected):
        """Records the raw continuous state so it can be projected with PCA / t-SNE.

        Also records the distance from the state to the nearest real token
        embedding, which quantifies how far off the token manifold the continuous
        sampler drifts. This matters because the language model has never been
        evaluated on inputs that are not token embeddings, so any energy it
        reports out there is an extrapolation with no defined meaning.
        """
        if getattr(self, "traj_log", None) is None:
            return
        with torch.no_grad():
            sd = s_state.detach()
            d = torch.cdist(sd, self.emb_matrix)          # M x V
            dmin, _ = d.min(dim=1)
            self.traj_log.append(dict(
                sampler="cls",
                seq_id=int(getattr(self, "_diag_seq_id", -1)),
                step=int(k),
                epsilon=float(eps_k),
                mh_rejected=int(bool(mh_rejected)),
                state=sd.float().cpu().numpy().copy(),     # M x D
                token_ids=idx_state.detach().cpu().numpy().copy(),
                dist_to_manifold=dmin.float().cpu().numpy().copy(),
            ))
