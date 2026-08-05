import torch
import numpy as np
from core.base_sampler import BaseLangevinSampler

# CTG phase, Study A. Arms whose proposal uses NO backward pass at all. They are
# handled together at the top of _step: the gradient is set to exactly zero, which
# makes the embedding-gradient half of t2 vanish identically, and no autograd call
# and no RNG draw occurs on their behalf. Every pre-existing method is untouched, so
# the archived grid still reproduces bit-identically.
#
#   policy_self    t2 = 0.5 * log p(v | x_<m)                    (A1: self term only)
#   policy_exact_k t2 = 0.5 * (E(v) - E(cur)) on a top-k shortlist (A3: exact future)
#   uniform        all proposal logits equal                      (A5: the floor)
GRADFREE_METHODS = ("policy_self", "policy_exact_k", "uniform")


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

    def _exact_shortlist_t2(self, self_lp, input_ids, mask_indices_t, s_idx, cur_log_joint):
        """A3: the EXACT future term, on a top-k shortlist. Returns (t2, shortlist).

        The linearized arm approximates the energy change of substituting candidate v
        by a first-order expansion in the input embedding. This computes it exactly,
        for the k candidates the self term ranks highest, by evaluating the energy of
        each substituted sequence:

            t2[v] = 0.5 * ( E(v) - E(current) )   for v in the shortlist, -inf outside

        The 0.5 and the "difference against the incumbent" match the linearized t2
        term exactly (a constant shift is a no-op inside a softmax anyway), so the two
        arms differ ONLY in how the future term is obtained and not in scaling.

        SINGLE MASK ONLY. With one masked position the substituted sequences differ
        from each other at that position alone, so E(v) does not depend on the current
        token and the whole vector can be cached and reused for the Metropolis reverse
        evaluation at no extra cost. With several masked positions that is false and
        the reverse term would need its own k passes, so the method refuses to run.

        The k candidate sequences are evaluated as ONE batched forward call. The cost
        counter is still incremented by k, because k sequences' worth of compute is
        what the arm actually spends; batching changes wall-clock, not FLOPs, and the
        cost table reports both.
        """
        if mask_indices_t.numel() != 1:
            raise ValueError("policy_exact_k supports num_masks=1 only "
                             f"(got {mask_indices_t.numel()})")
        m = int(mask_indices_t[0].item())

        # The candidate set is the FROZEN per-sequence shortlist, not a fresh top-k.
        # Using a different set here would give a token that the frozen support says is
        # reachable an exact score of -inf, which reintroduces exactly the irreversibility
        # that _shortlist_mask exists to remove.
        candidates = getattr(self, "_shortlist_indices", None)
        if candidates is None:
            raise ValueError("policy_exact_k requires --proposal_topk: its candidate set "
                             "IS the frozen shortlist.")

        # Cache key: the sequence with the masked position blanked out. Only position m
        # varies during the chain, so a hit means the cached candidate energies are the
        # energies of exactly these candidate sequences. Keying on the content rather
        # than on a caller-supplied sequence id means the cache cannot leak across
        # sequences even if the caller forgets to reset anything.
        fingerprint = input_ids.clone()
        fingerprint[0, m] = -1
        cache = getattr(self, "_exact_cache", None)
        if (cache is not None and cache["m"] == m
                and cache["fingerprint"].shape == fingerprint.shape
                and bool(torch.equal(cache["fingerprint"], fingerprint))):
            shortlist, lj = cache["shortlist"], cache["log_joint"]
        else:
            shortlist = candidates
            # Chunked so a large shortlist cannot blow up VRAM: the logits tensor is
            # chunk x seq_len x |V|, which at k=256 and float32 would be several GB in
            # one allocation, and the log_softmax doubles it.
            chunk = int(getattr(self, "exact_batch", 32))
            with torch.no_grad():
                parts = []
                for i0 in range(0, shortlist.numel(), chunk):
                    sub = shortlist[i0:i0 + chunk]
                    batch = input_ids.repeat(sub.numel(), 1)
                    batch[:, m] = sub
                    logits = self.model(batch).logits
                    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                    parts.append(lp.gather(2, batch[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=1))
                lj = torch.cat(parts)
            self.n_forward += int(shortlist.numel())
            self.n_exact_candidate_evals = getattr(self, "n_exact_candidate_evals", 0) + int(shortlist.numel())
            self._exact_cache = dict(m=m, fingerprint=fingerprint,
                                     shortlist=shortlist, log_joint=lj)

        t2 = torch.full((1, self.emb_matrix.shape[0]), -float("inf"),
                        device=self.device, dtype=self.emb_matrix.dtype)
        ref = cur_log_joint.detach()
        t2[0, shortlist] = (0.5 * (lj - ref)).to(t2.dtype)

        # THE INCUMBENT IS ALWAYS IN THE SHORTLIST. Without this the proposal cannot
        # return to the state it came from whenever that state is outside the top-k,
        # which is the normal case: the chain starts at a random corrupting token. The
        # reverse probability is then exactly zero, log_alpha is -inf, and every single
        # move is rejected. Measured before the fix: acceptance 0.0% on every sequence.
        # Its exact energy costs nothing: E(current) is the reference already passed in,
        # so its entry is 0.5 * (E(cur) - ref) = 0 by construction.
        t2[0, s_idx] = 0.0
        return t2, shortlist

    def _shortlist_mask(self, self_lp, incumbent, input_ids, mask_indices_t):
        """The frozen per-sequence proposal support: top-k by the self term, plus the
        token the chain started from. Returns a boolean keep-mask, or None when off.

        Set `sampler.proposal_topk` to a positive integer to turn this on; it is off by
        default and every archived run is unaffected.

        WHY THE SUPPORT IS FROZEN. The shortlist has to be the SAME set forward and
        backward or the chain is not reversible. A shortlist recomputed at each state as
        "top-k here, plus wherever I currently am" is not: the chain starts on a random
        corrupting token whose self-term rank is far outside the top-k, so once it moves
        away the reverse move has probability exactly zero, log_alpha is -inf, and every
        proposal is rejected forever. Measured before this fix: acceptance 0.0% on every
        arm, with the proposal itself perfectly healthy (24.6% of its mass on the ground
        truth and a candidate worth +19.8 nats of energy, refused because the reverse
        term was -inf).

        Freezing the support at the chain's initial state fixes it exactly rather than
        approximately. The result is an independence sampler on a fixed candidate set,
        which is a valid Metropolis-Hastings kernel; the proposal does not have to be
        the true conditional, since the accept step corrects it. The set is keyed on the
        sequence with its masked positions blanked, so it cannot leak across sequences.

        WHY IT EXISTS AT ALL. The exact-future arm can only ever score a shortlist:
        scoring the whole vocabulary exactly would cost |V| forward passes per step.
        Comparing it against arms that propose over all 50,257 tokens would confound the
        value of the exact future term with the cost of restricting the support, and the
        first smoke test showed exactly that confound. Applying the SAME restriction to
        every arm makes the comparison clean, and the shortlist is built from the self
        term, which every arm gets for free.
        """
        kk = int(getattr(self, "proposal_topk", 0))
        V = self_lp.shape[1]
        if kk <= 0 or kk >= V:
            return None

        fingerprint = input_ids.clone()
        fingerprint[0, mask_indices_t] = -1
        cache = getattr(self, "_shortlist_cache", None)
        if (cache is not None
                and cache["fingerprint"].shape == fingerprint.shape
                and bool(torch.equal(cache["fingerprint"], fingerprint))):
            return cache["keep"]

        keep = torch.zeros((self_lp.shape[0], V), dtype=torch.bool, device=self_lp.device)
        keep.scatter_(1, torch.topk(self_lp.float(), kk, dim=1).indices, True)
        keep.scatter_(1, incumbent.unsqueeze(1), True)
        self._shortlist_cache = dict(fingerprint=fingerprint, keep=keep)
        # Coverage instrument (Study C reports this before any adherence number): where
        # the ground truth / the constraint's preferred token falls relative to the
        # shortlist is recorded by the caller against this frozen set.
        self._shortlist_indices = keep[0].nonzero().squeeze(-1)
        return keep

    def _restrict_to_shortlist(self, logits, self_lp, incumbent, input_ids=None,
                               mask_indices_t=None):
        keep = self._shortlist_mask(self_lp, incumbent, input_ids, mask_indices_t)
        if keep is None:
            return logits
        return logits.masked_fill(~keep, -float("inf"))

    def _onehot_bonus(self, self_logprobs):
        """The self half of the one-hot gradient, ready to add to t2.

        t2 already carries 0.5 * g^T (e(v) - s), the embedding-gradient half.
        The one-hot gradient adds 0.5 * (log p(v | x_<m) - log p(x_m | x_<m));
        the subtracted term is constant in v and a softmax ignores it, so only
        0.5 * log p(v | x_<m) is returned.
        """
        return 0.5 * self_logprobs

    def _step(self, k, eps_k, s, s_idx, base_embs, input_ids, mask_indices_t, emb_gt):
        # 1. Gradients & Method Variation
        onehot = (self.method == "policy_onehot")
        gradfree = self.method in GRADFREE_METHODS
        # A shortlisted run needs the self term for every arm, including the ones whose
        # proposal does not otherwise use it, because the shortlist is defined by it.
        need_self = onehot or gradfree or int(getattr(self, "proposal_topk", 0)) > 0
        if gradfree:
            # No autograd, no RNG draw. One forward pass supplies both the energy for
            # the MH accept and the self term for the proposal.
            log_joint, self_lp = self.log_joint_and_self_logprobs(
                s, s_idx, base_embs, input_ids, mask_indices_t)
            grad_s = torch.zeros_like(s.detach())
        elif need_self:
            raw_grad_s, log_joint, self_lp = self.get_gradient_and_log_joint(
                s, s_idx, base_embs, input_ids, mask_indices_t, return_self_logprobs=True)
            grad_s = self.apply_method_variation(raw_grad_s)
        else:
            raw_grad_s, log_joint = self.get_gradient_and_log_joint(s, s_idx, base_embs, input_ids, mask_indices_t)
            grad_s = self.apply_method_variation(raw_grad_s)

        # 2. Oracle Alpha Search
        if self.oracle and emb_gt is not None:
            gt_logprobs = [self._compute_q_logprob_sum(emb_gt, s.detach(), grad_s, alpha).item() for alpha in self.alpha_grid]
            eps_k = self.alpha_grid[np.argmax(gt_logprobs)]

        # Freeze the proposal support before any arm scores candidates, so the
        # exact-future arm evaluates precisely the set every other arm proposes over.
        if need_self:
            self._shortlist_mask(self_lp, s_idx, input_ids, mask_indices_t)

        # 3. Compute Logits & Proposal (Vectorized for M masks)
        s_detached = s.detach()
        s_sq_norm = torch.sum(s_detached ** 2, dim=1, keepdim=True)
        dot_prod = torch.matmul(s_detached, self.emb_matrix.T)
        dist_sq = self.emb_matrix_sq_norm.unsqueeze(0) + s_sq_norm - 2 * dot_prod
        t1 = -dist_sq / (2 * eps_k)
        if getattr(self, "drop_distance_term", False):
            # The Langevin distance term scores the incumbent at exactly 0 and every
            # other candidate at -||e(v) - s||^2 / (2 eps). Over the full 50k vocabulary
            # that is one preference among many; over a 65-candidate shortlist it pins
            # the chain to its current token outright. Measured: with proposal_topk=64
            # and t1 present, all five arms returned an identical final KL of 6.777 and
            # an exact-match of zero, because every accepted move re-proposed the
            # incumbent. Dropping t1 turns each arm into a proposal over the shortlist
            # scored by its own surrogate, which is the comparison Study A is about.
            t1 = torch.zeros_like(t1)

        grad_dot_emb = torch.matmul(grad_s, self.emb_matrix.T)
        grad_dot_s = torch.sum(grad_s * s_detached, dim=1, keepdim=True)
        t2 = 0.5 * (grad_dot_emb - grad_dot_s)
        if onehot or self.method == "policy_self":
            t2 = t2 + self._onehot_bonus(self_lp)
        elif self.method == "policy_exact_k":
            t2, _ = self._exact_shortlist_t2(self_lp, input_ids, mask_indices_t,
                                             s_idx, log_joint)
        elif self.method == "uniform":
            # The floor: every candidate equally likely. Zeroing t1 as well as t2 is
            # deliberate; leaving the distance term in would make this a
            # distance-weighted walk, not a uniform draw, and the uniform arm is what
            # reproduced the flagship Langevin sampler in REVISION_LOG C8.
            t1 = torch.zeros_like(t1)
            t2 = torch.zeros_like(t2)

        raw_logits = t1 + t2
        if need_self:
            raw_logits = self._restrict_to_shortlist(raw_logits, self_lp, s_idx,
                                                      input_ids, mask_indices_t)
        scaled_logits = raw_logits / self.temperature if self.temperature else raw_logits
        probs = torch.softmax(scaled_logits, dim=1)

        next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        entropy_t = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean().item()

        s_next = self.emb_matrix[next_token_ids].clone().detach().requires_grad_(True)

        # Proposal sharpness, recorded for every step when asked for. This is the
        # quantity that says whether the gradient term is load-bearing at all:
        # t1 is the distance term, t2 the gradient-alignment term, and if t2 is
        # small against t1 (and both small against log|V|) the proposal is a
        # near-uniform draw over the vocabulary whatever the gradient says.
        if getattr(self, "record_proposal_stats", False):
            with torch.no_grad():
                t1_std = float(t1.std().item())
                t2_std = float(t2.std().item())
                self._last_proposal_stats = dict(
                    entropy=float(entropy_t),
                    t1_std=t1_std,
                    t2_std=t2_std,
                    t2_over_t1=t2_std / (t1_std + 1e-12),
                    logit_std=float(scaled_logits.std().item()),
                )

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
            exact_all = getattr(self, "mh_exact_all_arms", False)
            if gradfree:
                # Same construction as the forward proposal, evaluated at s_next, with
                # one gradient-free forward pass. Exact for every gradfree arm: the
                # embedding-gradient half of t2 is identically zero here, so the only
                # state-dependent pieces are the distance term and the self term, and
                # both are recomputed rather than assumed symmetric.
                log_fwd_prob = torch.log_softmax(scaled_logits, dim=1).gather(
                    1, next_token_ids.unsqueeze(1)).sum()

                bw_log_joint, self_lp_b = self.log_joint_and_self_logprobs(
                    s_next, next_token_ids, base_embs, input_ids, mask_indices_t)

                s_b_detached = s_next.detach()
                dist_sq_b = (self.emb_matrix_sq_norm.unsqueeze(0)
                             + torch.sum(s_b_detached ** 2, dim=1, keepdim=True)
                             - 2 * torch.matmul(s_b_detached, self.emb_matrix.T))
                t1_b = -dist_sq_b / (2 * eps_k)
                if getattr(self, "drop_distance_term", False):
                    t1_b = torch.zeros_like(t1_b)
                t2_b = torch.zeros_like(t1_b)
                if self.method == "policy_self":
                    t2_b = self._onehot_bonus(self_lp_b)
                elif self.method == "policy_exact_k":
                    # Single-mask only, so the exact candidate energies are the SAME
                    # vector forward and backward and come straight out of the cache;
                    # the reference constant differs but cancels in the softmax.
                    t2_b, _ = self._exact_shortlist_t2(self_lp_b, input_ids,
                                                       mask_indices_t, next_token_ids,
                                                       bw_log_joint)
                elif self.method == "uniform":
                    t1_b = torch.zeros_like(t1_b)

                raw_logits_b = self._restrict_to_shortlist(t1_b + t2_b, self_lp_b,
                                                           next_token_ids, input_ids,
                                                           mask_indices_t)
                scaled_logits_b = (raw_logits_b / self.temperature
                                   if self.temperature else raw_logits_b)
                log_bw_prob = torch.log_softmax(scaled_logits_b, dim=1).gather(
                    1, s_idx.unsqueeze(1)).sum()
                log_q_ratio = log_bw_prob - log_fwd_prob
            elif self.method in ("policy", "policy_onehot") or exact_all:
                # Detailed balance requires the reverse proposal to be evaluated under the
                # SAME kernel that produced the forward move, i.e. the method-varied
                # (here normalized when grad_normalization is on) gradient at s_next.
                # The previous legacy DLS used the raw backward gradient here, which broke
                # detailed balance whenever the forward proposal was normalized.
                log_fwd_prob = torch.log_softmax(scaled_logits, dim=1).gather(1, next_token_ids.unsqueeze(1)).sum()

                # need_self covers both the one-hot arm and any shortlisted arm: the
                # shortlist at the reverse state is defined by the self term there, so it
                # has to be read, and it comes out of the same forward pass either way.
                if need_self:
                    raw_grad_s_b, bw_log_joint, self_lp_b = self.get_gradient_and_log_joint(
                        s_next, next_token_ids, base_embs, input_ids, mask_indices_t,
                        return_self_logprobs=True)
                else:
                    raw_grad_s_b, bw_log_joint = self.get_gradient_and_log_joint(s_next, next_token_ids, base_embs, input_ids, mask_indices_t)

                if self.method in ("policy", "policy_onehot"):
                    grad_s_b = self.apply_method_variation(raw_grad_s_b)
                else:
                    # The random arms draw their direction independently of the state, so
                    # for a single step the kernel q_g is a fixed, exactly computable
                    # proposal. Reusing the SAME g backward (rather than redrawing, which
                    # would not be the reverse of anything, or assuming symmetry, which is
                    # false because the softmax normalisers differ) makes each step a valid
                    # MH move under q_g; a mixture of pi-invariant kernels is pi-invariant.
                    grad_s_b = grad_s

                s_b_detached = s_next.detach()
                dist_sq_b = self.emb_matrix_sq_norm.unsqueeze(0) + torch.sum(s_b_detached ** 2, dim=1, keepdim=True) - 2 * torch.matmul(s_b_detached, self.emb_matrix.T)
                t1_b = -dist_sq_b / (2 * eps_k)
                if getattr(self, "drop_distance_term", False):
                    t1_b = torch.zeros_like(t1_b)
                t2_b = 0.5 * (torch.matmul(grad_s_b, self.emb_matrix.T) - torch.sum(grad_s_b * s_b_detached, dim=1, keepdim=True))
                if onehot:
                    t2_b = t2_b + self._onehot_bonus(self_lp_b)

                raw_logits_b = t1_b + t2_b
                if need_self:
                    raw_logits_b = self._restrict_to_shortlist(raw_logits_b, self_lp_b,
                                                               next_token_ids, input_ids,
                                                               mask_indices_t)
                scaled_logits_b = raw_logits_b / self.temperature if self.temperature else raw_logits_b
                log_bw_prob = torch.log_softmax(scaled_logits_b, dim=1).gather(1, s_idx.unsqueeze(1)).sum()
                log_q_ratio = log_bw_prob - log_fwd_prob
            else:
                # LEGACY (thesis grid) TREATMENT. Random-direction baselines were assumed to
                # be symmetric random walks, so the proposal ratio was dropped. That is only
                # approximately true: the distance term is symmetric but the softmax
                # normalisers at x and x' differ. Set mh_exact_all_arms to compute it
                # exactly for every arm instead. Kept as the default so the archived grid
                # remains reproducible. Crucially we do NOT call apply_method_variation on
                # the backward gradient here, so no extra torch.randn is drawn and the RNG
                # stream stays aligned with the no-MH path.
                if int(getattr(self, "proposal_topk", 0)) > 0:
                    raise ValueError(
                        "proposal_topk with the legacy MH treatment is invalid: a "
                        "shortlisted proposal is not symmetric, so dropping the "
                        "reverse-proposal term would not sample the target. Pass "
                        "--mh_exact_all_arms.")
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
