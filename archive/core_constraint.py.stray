#!/usr/bin/env python3
"""
core/constraint.py

Adds a differentiable sentiment constraint to the Langevin energy, so we can ask
the question the literature never isolates:

    The LM-likelihood gradient carries no usable directional signal (we showed this
    across 5 models, with and without normalization). MuCoLa / COLD nevertheless
    "work". Is that because the CONSTRAINT gradient carries the signal that the LM
    gradient does not?

Energy (following MuCoLa's weighting, betas = 0.8 : 0.2):

    U(s) = beta_lm * log p_LM(x)  +  beta_c * log p_classifier(y = target | x)

Both terms are differentiable w.r.t. the same continuous embeddings s that the
sampler moves, so the total gradient decomposes cleanly:

    grad U = beta_lm * grad_LM  +  beta_c * grad_constraint

which is exactly what lets us ablate each half independently.

MuCoLa detail we deliberately do NOT copy: they impose the constraint as a
Lagrangian with epsilon thresholds (log p(neg) - log p(pos) < -2). We use the
simpler weighted sum because our question is about the *gradient's information
content*, not about constraint satisfaction guarantees. The weighted sum keeps the
decomposition above exact and interpretable. Noted in the thesis as a deviation.
"""

import torch
import torch.nn as nn


class SentimentHead(nn.Module):
    """Must match train_sentiment_head.py exactly."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
        )

    def forward(self, hidden, attn_mask=None):
        if attn_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            m = attn_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.proj(pooled)


def load_sentiment_head(path, device, dtype=torch.float32):
    ckpt = torch.load(path, map_location=device)
    head = SentimentHead(ckpt["hidden_size"]).to(device=device, dtype=dtype)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def constraint_log_prob(model, head, inputs_embeds, target_label):
    """log p(target_label | x), differentiable w.r.t. inputs_embeds.

    Runs the frozen LM with output_hidden_states so the head reads the same forward
    pass. Gradients flow: head -> last hidden state -> inputs_embeds.
    """
    out = model(inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1]
    logits = head(hidden)                      # (1, 2)
    logp = torch.log_softmax(logits, dim=-1)
    return logp[0, target_label]


class ConstrainedEnergyMixin:
    """Mix into a sampler to replace the pure-LM energy with LM + constraint.

    Set on the sampler instance:
      self.head           : SentimentHead or None
      self.target_label   : 0 (neg) or 1 (pos)
      self.beta_lm        : weight on the LM likelihood   (MuCoLa: 0.8)
      self.beta_c         : weight on the constraint      (MuCoLa: 0.2)
      self.constraint_mode: which gradient the PROPOSAL is allowed to use:
          "full"        - beta_lm*grad_LM + beta_c*grad_constraint   (the real method)
          "lm_only"     - LM gradient only (our existing baseline)
          "cons_only"   - constraint gradient only (does the classifier alone steer?)
          "random"      - random direction (control)
          "cons_random" - constraint's MAGNITUDE, random DIRECTION.
                          THE KEY ABLATION: if "full" beats this, the constraint
                          gradient's DIRECTION carries real information, which is
                          precisely what the LM gradient failed to do.

    Note the energy used for the MH accept ratio is ALWAYS the true combined energy,
    regardless of constraint_mode. Only the proposal direction is ablated. That keeps
    every arm sampling from the same target distribution, so the comparison is fair.
    """

    def combined_log_joint(self, inputs_embeds, target_ids):
        from core.prep import joint_log_prob_from_inputs_embeds
        lm = joint_log_prob_from_inputs_embeds(self.model, inputs_embeds, target_ids)
        if self.head is None:
            return lm, lm, None
        c = constraint_log_prob(self.model, self.head, inputs_embeds, self.target_label)
        total = self.beta_lm * lm + self.beta_c * c
        return total, lm, c

    def _assert_constraint_is_live(self, lm, c, s):
        """Fail loudly if the constraint cannot possibly affect the proposal.

        This exists because of a real bug we hit: with grad_normalization=True the
        sampler normalizes the combined gradient to a UNIT vector. The LM gradient's
        norm dwarfs the 0.2-weighted classifier gradient, so after normalization the
        combined direction rounds to the LM direction and `full` becomes bitwise
        identical to `lm_only`. The constraint was mathematically present and had
        literally zero effect.

        Note this is not a coding slip so much as a mathematical incompatibility:
        beta_lm/beta_c set the RELATIVE MAGNITUDES of the two terms, and gradient
        normalization discards magnitude by construction. MuCoLa does not normalize
        (their EmbedGD uses the raw weighted sum), which is exactly why their 0.2
        actually means something. A weighted multi-term energy and unit-norm
        gradients cannot both be honoured.
        """
        if not self.grad_normalization:
            return
        raise RuntimeError(
            "grad_normalization=True is incompatible with a weighted multi-term energy.\n"
            "The combined gradient is normalized to a unit vector, so the relative\n"
            "weights beta_lm/beta_c are discarded and the constraint term has no\n"
            "effect on the proposal direction (we verified `full` becomes bitwise\n"
            "identical to `lm_only`). MuCoLa does not normalize either.\n"
            "Run the constraint arms with grad_norm OFF."
        )

    def get_gradient_and_log_joint(self, s, s_idx, base_embs, input_ids, mask_indices_t):
        """Overrides the base method. Returns (proposal_grad, energy_for_MH)."""
        inputs_embeds = base_embs.clone().detach()
        inputs_embeds[0, mask_indices_t, :] = s
        target_ids = input_ids.clone()
        target_ids[0, mask_indices_t] = s_idx

        if self.head is None:
            from core.prep import joint_log_prob_from_inputs_embeds
            log_joint = joint_log_prob_from_inputs_embeds(self.model, inputs_embeds, target_ids)
            grad = torch.autograd.grad(log_joint, s, retain_graph=False)[0]
            return grad, log_joint

        total, lm, c = self.combined_log_joint(inputs_embeds, target_ids)
        self._assert_constraint_is_live(lm, c, s)

        mode = getattr(self, "constraint_mode", "full")
        if mode == "full":
            grad = torch.autograd.grad(total, s, retain_graph=False)[0]
        elif mode == "lm_only":
            grad = self.beta_lm * torch.autograd.grad(lm, s, retain_graph=False)[0]
        elif mode in ("cons_only", "cons_random"):
            g_c = torch.autograd.grad(c, s, retain_graph=False)[0]
            if mode == "cons_random":
                # same magnitude, random direction: isolates the DIRECTION's value
                rand = torch.randn_like(g_c)
                rand = rand / (rand.norm(dim=1, keepdim=True) + 1e-12)
                g_c = rand * g_c.norm(dim=1, keepdim=True)
            grad = self.beta_c * g_c
        elif mode == "random":
            g_any = torch.autograd.grad(total, s, retain_graph=False)[0]
            rand = torch.randn_like(g_any)
            grad = rand / (rand.norm(dim=1, keepdim=True) + 1e-12) * g_any.norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"unknown constraint_mode: {mode}")

        # MH always evaluates the TRUE combined energy, whatever the proposal used.
        return grad, total.detach()
